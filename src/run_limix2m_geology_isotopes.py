"""Compare Sr–Nd–Pb inputs at the frozen 2,048-label geology experiment budget."""

import time

import numpy as np
import pandas as pd

import run_limix2m_geology_paper as paper
from run_d36_srnd_partial_chemistry_2026_11_13 import PAIR
from run_d36_srnd_pb_2026_11_13 import PB, QA, load_pb


BUDGET = 2048
BASE = "D36_74"
SRND = "D36_74+SrNd"
PB_FLAGS = "D36_74+SrNd+Pb_flags"
PB_VALUES = "D36_74+SrNd+Pb"
MAIN = "原始测试"
MODELS = ("LimiX-2M", "LightGBM")
METRICS = ("accuracy", "macro_f1_9_labels", "log_loss", "brier", "ece")
LABELS = paper.limix.LABELS
ID = paper.ID


def baseline(seed, scenario, context, test, donor, ids, y):
    path = paper.OUT / f"seed_{seed}" / BASE / f"{BUDGET}_{scenario}"
    with np.load(path / "indices.npz") as indices:
        np.testing.assert_array_equal(indices["context"], context)
        np.testing.assert_array_equal(indices["test"], test)
        np.testing.assert_array_equal(indices["donor"], donor)
    saved = pd.read_parquet(path / "predictions.parquet")
    np.testing.assert_array_equal(saved[ID], ids[test])
    np.testing.assert_array_equal(saved["true"], LABELS[y[test]])
    return {model: saved[[f"{model}_p_{label}" for label in LABELS]].to_numpy() for model in MODELS}


def array_columns(ids, y, complete, srnd, pb, index):
    tests = pd.DataFrame({
        ID: ids[index], "true": LABELS[y[index]], "D36_complete": complete[index],
        "Sr_available": np.isfinite(srnd[index, 0]), "Nd_available": np.isfinite(srnd[index, 1]),
        "Pb_available": np.isfinite(pb[index]).any(axis=1),
    })
    return tests


def save_case(out, seed, scenario, panel, tests, probabilities, meta):
    path = out / f"seed_{seed}" / panel / scenario
    path.mkdir(parents=True, exist_ok=True)
    frame = tests.copy()
    y = pd.Categorical(tests.true, categories=LABELS).codes
    for model, p in probabilities.items():
        paper.probability_metrics(y, p)
        frame[model] = LABELS[p.argmax(axis=1)]
        for label, values in zip(LABELS, p.T, strict=True):
            frame[f"{model}_p_{label}"] = values
    frame.to_parquet(path / "predictions.parquet", index=False)
    paper.limix.core.dump(meta, path / "configuration.json")


def run_seed(args, predictor, seed, cohort, chemistry, srnd, pb, roles):
    ids = cohort[ID].to_numpy()
    y = pd.Categorical(cohort[paper.LABEL], categories=LABELS).codes
    complete = cohort.D36_complete.to_numpy()
    train, test = (np.flatnonzero(roles == role) for role in ("train", "test"))
    assert len(test) == 4851 and len(train) == 19400
    context = dict(paper.context_rows(train, y, seed))[BUDGET]
    raw = np.power(10, chemistry[:, :len(paper.E36)].astype(float))
    mask_test, donors, masks = paper.test_masks(raw, train, test, complete, seed)
    test_meta = array_columns(ids, y, complete, srnd, pb, test)
    mask_meta = array_columns(ids, y, complete, srnd, pb, mask_test)

    base_cases = [(MAIN, test, np.array([], dtype=int), test_meta)]
    base_cases.extend((scenario, mask_test, donors, mask_meta) for scenario in paper.MASKS)
    for scenario, idx, donor, meta in base_cases:
        saved = baseline(seed, scenario, context, idx, donor, ids, y)
        save_case(args.out, seed, scenario, BASE, meta, saved, {"source": "原化学实验逐样品预测"})

    extras = {
        SRND: srnd,
        PB_FLAGS: np.column_stack((srnd, np.isfinite(pb))),
        PB_VALUES: np.column_stack((srnd, pb)),
    }
    for panel, extra in extras.items():
        cases = [(MAIN, np.column_stack((chemistry[test], extra[test])), test_meta)]
        if panel == PB_VALUES:
            cases.extend((scenario, np.column_stack((masked, extra[mask_test])), mask_meta)
                         for scenario, masked in zip(paper.MASKS, masks, strict=True))
        pending = [(s, x, m) for s, x, m in cases
                   if not (args.out / f"seed_{seed}" / panel / s / "configuration.json").exists()]
        if not pending:
            continue
        x = np.column_stack((chemistry, extra))
        tree, details = paper.tuned_tree(x, y, context, seed)
        for scenario, query, meta in pending:
            start = time.perf_counter()
            probs = {
                MODELS[0]: paper.limix_predict(predictor, x, y, context, query, args.batch),
                MODELS[1]: tree.predict(query, num_threads=4),
            }
            save_case(args.out, seed, scenario, panel, meta, probs, details)
            print(f"{seed} {panel} {scenario} {len(meta)} rows {time.perf_counter()-start:.1f}s", flush=True)


def summarize(out):
    results, classes, keys_by_case = [], [], {}
    for file in sorted(out.glob("seed_*/*/*/predictions.parquet")):
        seed, panel, scenario = file.relative_to(out).parts[:3]
        seed = int(seed.removeprefix("seed_"))
        data = pd.read_parquet(file)
        keys = data[[ID, "true", "D36_complete", "Sr_available", "Nd_available", "Pb_available"]]
        assert data[ID].is_unique
        key = seed, scenario
        if key in keys_by_case:
            pd.testing.assert_frame_equal(keys, keys_by_case[key])
        else:
            keys_by_case[key] = keys
        srnd = data.Sr_available | data.Nd_available
        groups = {
            "全部": data,
            "至少一项SrNd": data[srnd],
            "至少一项Pb": data[data.Pb_available],
            "无可用Pb": data[~data.Pb_available],
            "无可用同位素": data[~(srnd | data.Pb_available)],
            "D36部分缺测": data[~data.D36_complete],
        }
        for subset, rows in groups.items():
            if rows.empty:
                continue
            y = pd.Categorical(rows.true, categories=LABELS).codes
            for model in MODELS:
                p = rows[[f"{model}_p_{label}" for label in LABELS]].to_numpy()
                stats = paper.probability_metrics(y, p)
                np.testing.assert_array_equal(rows[model], LABELS[p.argmax(axis=1)])
                config = dict(seed=seed, panel=panel, scenario=scenario, subset=subset, model=model)
                results.append({**config, "n": len(rows), **{m: stats[m] for m in METRICS}})
                if subset == "全部":
                    for label, line in zip(LABELS, stats["confusion_matrix"], strict=True):
                        j = np.flatnonzero(LABELS == label).item()
                        classes.append({**config, "class": label, "n": sum(line), "correct": line[j],
                                        "f1": stats["per_class_f1"][label]})
    result = pd.DataFrame(results)
    result.to_csv(out / "scores_by_seed.csv", index=False)
    pd.DataFrame(classes).to_csv(out / "per_class.csv", index=False)
    key = ["panel", "scenario", "subset", "model"]
    groups = result.groupby(key, as_index=False)
    summary = groups.agg(n_min=("n", "min"), n_max=("n", "max"), splits=("seed", "size"),
                         **{metric: (metric, "mean") for metric in METRICS})
    summary.to_csv(out / "scores_summary.csv", index=False)
    comparisons = []
    for target, reference in ((SRND, BASE), (PB_VALUES, SRND), (PB_VALUES, PB_FLAGS), (PB_VALUES, BASE)):
        joined = result[result.panel.eq(target)].merge(
            result[result.panel.eq(reference)], on=["seed", "scenario", "subset", "model"],
            suffixes=("_target", "_ref"), validate="one_to_one")
        assert joined.n_target.eq(joined.n_ref).all()
        for metric in METRICS:
            joined[metric] = joined[f"{metric}_target"] - joined[f"{metric}_ref"]
        joined["target"], joined["reference"] = target, reference
        comparisons.append(joined[["seed", "target", "reference", "scenario", "subset", "model", *METRICS]])
    pd.concat(comparisons, ignore_index=True).to_csv(out / "paired_deltas_by_seed.csv", index=False)
    print(summary[summary.subset.eq("全部")].to_string(index=False), flush=True)


def main():
    args, predictor = paper.args_and_model()
    if args.out.resolve() == paper.OUT.resolve():
        raise ValueError("Choose a separate --out directory for the isotope results.")
    if args.batch != 1024:
        raise ValueError("Use the original experiment's GPU query batch of 1024 for paired comparisons.")
    cohort, chemistry, _, _ = paper.load_cohort()
    pb = load_pb(cohort)[PB].to_numpy(np.float32)
    srnd = cohort[PAIR].to_numpy(np.float32)
    assert np.isfinite(pb).any(axis=1).sum() == 5965
    assert (srnd[np.isfinite(srnd)] > 0).all()
    inputs = (paper.FROZEN / "cohort_manifest.csv.gz", paper.FROZEN / "split_manifest.csv.gz", QA)
    sha = paper.limix.core.sha
    split = pd.read_csv(inputs[1]).set_index(["seed", ID], verify_integrity=True)
    print(f"{len(cohort)} samples, Sr={np.isfinite(srnd[:, 0]).sum()}, Nd={np.isfinite(srnd[:, 1]).sum()}, "
          f"any Pb={np.isfinite(pb).any(axis=1).sum()}", flush=True)
    for seed in args.seeds:
        roles = split.loc[seed].loc[cohort[ID]].role.to_numpy()
        run_seed(args, predictor, seed, cohort, chemistry, srnd, pb, roles)
    summarize(args.out)
    groups = [args.out / f"seed_{seed}" for seed in args.seeds]
    for group in groups:
        assert len(list(group.rglob("predictions.parquet"))) == 10
    paper.limix.core.dump({
        "status": "PASS", "budget": BUDGET, "seeds": list(args.seeds),
        "sr_nd_features": PAIR, "pb_features": PB,
        "cohort_isotope_counts": {"Sr": int(np.isfinite(srnd[:, 0]).sum()), "Nd": int(np.isfinite(srnd[:, 1]).sum()),
                                 "any_Pb": int(np.isfinite(pb).any(axis=1).sum())},
        "model_sha": paper.limix.WEIGHTS_SHA, "model_revision": paper.limix.UPSTREAM,
        "inputs_sha": {str(p): sha(p) for p in inputs},
        "predictions_sha": {str(p.relative_to(args.out)): sha(p)
                            for group in groups for p in sorted(group.rglob("predictions.parquet"))},
    }, args.out / "audit.json")


if __name__ == "__main__":
    main()
