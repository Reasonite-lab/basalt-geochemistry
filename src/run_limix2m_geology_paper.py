"""Paired chemical-panel and label-budget experiments on the frozen common22 cohort."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.model_selection import train_test_split
import torch

from d22_process_common_2026_10_19 import build_d22_42
from run_d36_srnd_partial_chemistry_2026_11_13 import (
    E36, FROZEN, ID, LABEL, ROOT, SEEDS, build_d36_74, load_cohort,
)
import run_limix2m_missing_d36 as limix


OUT = ROOT / "reports/modeling/limix2m_geology_paper_2026-11-17"
BUDGETS = (512, 2048, 4096)
PANELS = ("D22_42", "D36_74")
MASKS = ("完整实测", "随机遮挡", "实测组合遮挡")


def context_rows(train, y, seed):
    """Nested stratified contexts shared by both models and chemical panels."""
    remaining = train.copy()
    picked = np.array([], dtype=int)
    for total in BUDGETS:
        added, remaining = train_test_split(remaining, train_size=total - len(picked),
                                             stratify=y[remaining], random_state=seed)
        picked = np.sort(np.r_[picked, added])
        assert len(picked) == total and len(np.unique(y[picked])) == len(limix.LABELS)
        yield total, picked.copy()


def probability_metrics(y, p):
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)
    assert np.isfinite(p).all() and p.min() >= 0
    result = limix.evaluate(y, p.argmax(axis=1))
    result["log_loss"] = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())
    onehot = np.eye(len(limix.LABELS))[y]
    result["brier"] = float(np.square(p - onehot).sum(axis=1).mean())
    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    bin_id = np.minimum((conf * 10).astype(int), 9)
    result["ece"] = float(sum(abs(correct[bin_id == b].sum() - conf[bin_id == b].sum())
                              for b in range(10)) / len(y))
    return result


def apply_temperature(p, t):
    return softmax(np.log(np.clip(p, 1e-15, 1)) / t, axis=1)


def fit_temperature(y, p):
    def objective(t):
        q = apply_temperature(p, t)
        return float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-15, 1)).mean())
    return float(minimize_scalar(objective, bounds=(0.25, 5), method="bounded").x)


def tuned_tree(x, y, train, seed, history=None):
    inner, val = train_test_split(train, test_size=0.2, stratify=y[train], random_state=seed)
    params = dict(n_estimators=800, objective="multiclass", learning_rate=0.05, num_leaves=31,
                  colsample_bytree=0.8, subsample=0.8, subsample_freq=1, verbosity=-1,
                  n_jobs=4, deterministic=True, force_col_wise=True, random_state=seed)
    selector = lgb.LGBMClassifier(**params)
    if history is None:
        selector.fit(x[inner], y[inner], eval_set=[(x[val], y[val])],
                     eval_metric="multi_logloss", callbacks=[lgb.early_stopping(45, verbose=False)])
    else:
        selector.fit(x[inner], y[inner], eval_set=[(x[inner], y[inner]), (x[val], y[val])],
                     eval_names=["train", "validation"], eval_metric=["multi_logloss", "multi_error"],
                     callbacks=[lgb.early_stopping(45, first_metric_only=True, verbose=False),
                                lgb.record_evaluation(history)])
    temp = fit_temperature(y[val], selector.booster_.predict(x[val], num_threads=4))
    trees = selector.best_iteration_
    model = lgb.LGBMClassifier(**{**params, "n_estimators": trees}).fit(x[train], y[train])
    return model.booster_, {"rounds": trees, "temperature": temp, "inner_fit_n": len(inner),
                            "inner_validation_n": len(val)}


def score_variants(case, variants):
    rows = []
    for name, subset, labels, keys, probs in variants:
        stats = probability_metrics(labels, probs)
        rows.append({**case, "model": name, "subset": subset, "n": len(keys), **stats})
    return rows


def write_case(path, y, ids, complete, predictions, case, meta):
    base = pd.DataFrame({ID: ids, "true": limix.LABELS[y], "D36_complete": complete})
    variants = []
    for name, p in predictions.items():
        base[name] = limix.LABELS[p.argmax(axis=1)]
        for label, values in zip(limix.LABELS, p.T):
            base[f"{name}_p_{label}"] = values
        for subset, mask in [("全部", np.ones(len(y), bool)), ("D36齐全", complete), ("D36部分缺测", ~complete)]:
            if mask.any():
                variants.append((name, subset, y[mask], ids[mask], p[mask]))
    path.mkdir(parents=True, exist_ok=True)
    base.to_parquet(path / "predictions.parquet", index=False)
    rows = score_variants(case, variants)
    limix.core.dump({"configuration": meta, "results": rows}, path / "result.json")


def test_masks(raw, train, test, complete, seed):
    test = test[complete[test]]
    rng = np.random.default_rng(seed)
    donors = train[~complete[train]]
    donor = rng.choice(donors, len(test), replace=True)
    observed = np.isfinite(raw[donor])
    # Same test samples and same number of masked measurements per sample.
    permutation = rng.random(observed.shape).argsort(axis=1)
    uniform = np.take_along_axis(observed, permutation, axis=1)
    np.testing.assert_array_equal(observed.sum(axis=1), uniform.sum(axis=1))
    masked = [raw[test], np.where(uniform, raw[test], np.nan), np.where(observed, raw[test], np.nan)]
    return test, donor, [build_d36_74(a, allow_missing=True)[0] for a in masked]


def summarize(out):
    rows = []
    for p in sorted(out.glob("seed_*/*/*/result.json")):
        rows.extend(json.loads(p.read_text(encoding="utf-8"))["results"])
    if not rows:
        return
    simple, per_class, confusion = [], [], []
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in {"confusion_matrix", "per_class_f1"}}
        simple.append(flat)
        for i, label in enumerate(limix.LABELS):
            line = np.array(row["confusion_matrix"][i])
            per_class.append({**flat, "class": label, "n_class": line.sum(), "correct": line[i],
                              "recall": line[i] / max(line.sum(), 1), "class_f1": row["per_class_f1"][label]})
            confusion.extend({**{k: flat[k] for k in ("seed", "panel", "budget", "scenario", "model", "subset")},
                              "true": label, "predicted": other, "count": int(n)}
                             for other, n in zip(limix.LABELS, line))
    pd.DataFrame(simple).to_csv(out / "scores.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(confusion).to_csv(out / "confusion.csv", index=False)


def args_and_model():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--phase", choices=["main", "mask", "context"], default="main")
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()
    assert limix.core.sha(args.checkpoint) == limix.WEIGHTS_SHA
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                                       text=True).strip()
    assert revision == limix.UPSTREAM
    torch.set_num_threads(4)
    sys.path[:0] = [str(args.dependencies.resolve()), str(args.upstream.resolve())]
    from inference.predictor import LimiXPredictor

    config = json.loads((args.upstream / "config/cls_default_noretrieval.json").read_text())
    return args, LimiXPredictor(device=torch.device("cuda"), model_path=str(args.checkpoint.resolve()),
                                inference_config=config, seed=limix.SEED)


def limix_predict(model, x, y, context, query, batch):
    parts = []
    for start in range(0, len(query), batch):
        parts.append(model.predict(x[context], y[context], query[start:start + batch]))
    return np.concatenate(parts)


def main():
    args, predictor = args_and_model()
    cohort, d36, d36_names, _ = load_cohort()
    ids = cohort[ID].to_numpy()
    y = pd.Categorical(cohort[LABEL], categories=limix.LABELS).codes
    complete = cohort.D36_complete.to_numpy()
    raw = np.power(10, d36[:, :len(E36)].astype(float))
    d22, d22_names = build_d22_42(raw[:, :22])
    mapping = [d36_names.index(name) for name in d22_names]
    np.testing.assert_allclose(d22, d36[:, mapping], atol=5e-6, rtol=1e-5)
    features = {"D22_42": d36[:, mapping].copy(), "D36_74": d36}
    split = pd.read_csv(FROZEN / "split_manifest.csv.gz").set_index(["seed", ID], verify_integrity=True)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {"samples": len(ids), "D36_complete": int(complete.sum()), "budgets": list(BUDGETS),
                "splits": list(SEEDS), "split_sha": limix.core.sha(FROZEN / "split_manifest.csv.gz"),
                "cohort_sha": limix.core.sha(FROZEN / "cohort_manifest.csv.gz"),
                "feature_panels": {"D22_42": d22_names, "D36_74": d36_names},
                "model_sha": limix.WEIGHTS_SHA, "model_version": limix.UPSTREAM, "batch": args.batch,
                "tree_reselection": "Inner 20% of each matched context, early stopping on log loss; refit all rows at selected rounds.",
                "temperature": "Fit on inner predictions using only originally budgeted labels."}
    limix.core.dump(protocol, args.out / "protocol.json")
    for seed in args.seeds:
        roles = split.loc[seed].loc[ids].role.to_numpy()
        train, test = (np.flatnonzero(roles == k) for k in ("train", "test"))
        assert len(train) == 19400 and len(test) == 4851 and np.intersect1d(train, test).size == 0
        base = args.out / f"seed_{seed}"
        all_contexts = dict(context_rows(train, y, seed))
        if args.phase == "context":
            variants = [(budget, rows, f"上下文复抽{add}") for add in (1, 2)
                        for budget, rows in context_rows(train, y, seed + add)]
        else:
            variants = [(k, v, "原始测试") for k, v in all_contexts.items()]
        for panel, x in features.items():
            if args.phase == "mask" and panel != "D36_74":
                continue
            if args.phase == "main":
                where = base / panel / "full_tree"
                if not (where / "result.json").is_file():
                    model, meta = tuned_tree(x, y, train, seed)
                    prob = model.predict(x[test], num_threads=4)
                    write_case(where, y[test], ids[test], complete[test],
                               {"全量LightGBM": prob, "全量LightGBM校准": apply_temperature(prob, meta["temperature"])},
                               dict(seed=seed, panel=panel, budget=len(train), scenario="原始测试"), meta)
                    print(seed, panel, "full tree", flush=True)
            for budget, context, scenario in variants:
                if args.phase == "mask":
                    if budget != 2048:
                        continue
                    tq, donors, masks = test_masks(raw, train, test, complete, seed)
                    mask_cases = [(s, m) for s, m in zip(MASKS, masks)]
                else:
                    tq, donors, mask_cases = test, None, [(scenario, x[test])]
                fitted = None
                for scenario, query in mask_cases:
                    path = base / panel / f"{budget}_{scenario}"
                    if (path / "result.json").exists():
                        continue
                    start = time.perf_counter()
                    if fitted is None:
                        tree, meta = tuned_tree(x, y, context, seed)
                        fit, val = train_test_split(context, test_size=0.2, stratify=y[context], random_state=seed)
                        val_prob = limix_predict(predictor, x, y, fit, x[val], args.batch)
                        lm_temp = fit_temperature(y[val], val_prob)
                        classifier = None
                        if args.phase == "mask":
                            classifier = lgb.LGBMClassifier(n_estimators=200, verbosity=-1, n_jobs=4, random_state=seed)
                            classifier.fit(np.isfinite(x[context, :36]), y[context])
                        fitted = tree, meta, val, lm_temp, classifier
                    tree, meta, val, lm_temp, classifier = fitted
                    gbm_prob = tree.predict(query, num_threads=4)
                    lm_prob = limix_predict(predictor, x, y, context, query, args.batch)
                    output = {"LimiX-2M": lm_prob, "LightGBM": gbm_prob,
                              "LimiX-2M校准": apply_temperature(lm_prob, lm_temp),
                              "LightGBM校准": apply_temperature(gbm_prob, meta["temperature"])}
                    if classifier is not None:
                        output["仅测量标记"] = classifier.booster_.predict(np.isfinite(query[:, :36]), num_threads=4)
                        marker_path = base / panel / f"{budget}_原始测量标记"
                        if not (marker_path / "result.json").exists():
                            marker_prob = classifier.booster_.predict(np.isfinite(x[test, :36]), num_threads=4)
                            write_case(marker_path, y[test], ids[test], complete[test],
                                       {"仅测量标记": marker_prob},
                                       dict(seed=seed, panel=panel, budget=budget, scenario="原始测量标记"),
                                       {"rounds": 200, "feature_columns": 36})
                    path.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path / "indices.npz", context=context, test=tq, val=val,
                                        donor=donors if donors is not None else np.array([], int))
                    write_case(path, y[tq], ids[tq], complete[tq], output,
                               dict(seed=seed, panel=panel, budget=budget, scenario=scenario),
                               {**meta, "limix_temperature": lm_temp, "wall_seconds": time.perf_counter() - start})
                    summarize(args.out)
                    print(seed, panel, budget, scenario, f"{time.perf_counter()-start:.1f}s", flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
