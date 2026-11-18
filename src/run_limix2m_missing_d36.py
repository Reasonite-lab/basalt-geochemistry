"""Compare complete and partially measured D36 contexts in held-out domains."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import incomplete_pool_fixed_test_2026_10_01 as core
from run_limix2_d36 import evaluate


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_SHA = "16f385d5201dc17cd6a8eb18becb0b4f8b3603838423b9f06062106acc965513"
UPSTREAM = "a50777da8d41607bc6ea467c7e16b4e051c1fb19"
SEED = 20261117
LABELS = np.array(core.CLASSES)
GUARDS = ("province_10fold", "reference_10fold", "chemical_guard")
SCHEMES = ("province_10fold", "reference_10fold", "natural_missing")
MODELS = ("limix2m", "lightgbm")
CONTEXTS = ("complete", "mixed")


def allowed_rows(groups, test, guards=GUARDS):
    allowed = np.ones(len(groups[GUARDS[0]]), dtype=bool)
    for key in guards:
        allowed &= ~np.isin(groups[key], np.unique(groups[key][test]))
    assert not allowed[test].any()
    return np.flatnonzero(allowed)


def cases(n_anchor, labels, folds, groups):
    for scheme in SCHEMES[:2]:
        for fold in range(10):
            yield scheme, fold, np.flatnonzero(folds[scheme] == fold)
    partial = np.arange(n_anchor, len(labels))
    cv = StratifiedGroupKFold(n_splits=8, random_state=SEED, shuffle=True)
    for fold, (_, heldout) in enumerate(cv.split(partial, labels[partial], groups["province_10fold"][partial])):
        test = partial[heldout]
        if np.bincount(labels[test], minlength=len(LABELS)).min() >= 20:
            yield "natural_missing", fold, test
            return
    raise ValueError("No natural-missing holdout has at least 20 samples in every class.")


def paired_contexts(train, labels, n_anchor, budget, seed):
    complete = train[train < n_anchor]
    extra = train[train >= n_anchor]
    ref, _ = train_test_split(complete, train_size=budget, stratify=labels[complete],
                              random_state=seed)
    rng = np.random.default_rng(seed)
    mixed = ref.copy()
    for label in range(len(LABELS)):
        slots = np.flatnonzero(labels[ref] == label)
        candidates = extra[labels[extra] == label]
        count = min(len(candidates), len(slots) // 2)
        positions = rng.choice(slots, size=count, replace=False)
        mixed[positions] = rng.choice(candidates, size=count, replace=False)
    assert np.array_equal(labels[ref], labels[mixed])
    assert len(np.unique(labels[ref])) == len(LABELS)
    assert len(np.unique(mixed)) == budget and (mixed >= n_anchor).any()
    return {"complete": np.sort(ref), "mixed": np.sort(mixed)}


def random_cases(n_anchor, labels, folds, groups):
    """Extend the existing random D36 folds to the measured-partial pool."""
    chemical = groups["chemical_guard"]
    anchor_folds = folds["random_10fold"]
    linked = pd.DataFrame({"group": chemical[:n_anchor], "fold": anchor_folds})
    if linked.groupby("group").fold.nunique().gt(1).any():
        raise ValueError("An existing D36 random fold splits a chemical duplicate group.")
    group_fold = linked.drop_duplicates("group").set_index("group").fold
    assignments = np.full(len(labels), -1, dtype=int)
    assignments[:n_anchor] = anchor_folds
    attached = pd.Series(chemical[n_anchor:]).map(group_fold)
    assignments[n_anchor:] = attached.fillna(-1).to_numpy(dtype=int)
    unassigned = np.flatnonzero(assignments < 0)
    cv = StratifiedGroupKFold(n_splits=10, random_state=SEED, shuffle=True)
    for fold, (_, test) in enumerate(cv.split(unassigned, labels[unassigned], chemical[unassigned])):
        assignments[unassigned[test]] = fold
    assert not (assignments < 0).any()
    assert pd.DataFrame({"group": chemical, "fold": assignments}).groupby("group").fold.nunique().max() == 1
    return [("random_10fold", fold, np.flatnonzero(assignments == fold)) for fold in range(10)]


def limix_probabilities(predictor, x, y, context, test, query_batch):
    parts = []
    for start in range(0, len(test), query_batch):
        query = test[start:start + query_batch]
        parts.append(predictor.predict(x[context], y[context], x[query], task_type="Classification"))
    return np.concatenate(parts)


def write_summary(out, random_only=False):
    entries = []
    per_class = []
    for scheme in ("random_10fold",) if random_only else SCHEMES:
        files = sorted((out / scheme).glob("fold_*/predictions.csv.gz"))
        files = [p for p in files if p.with_name("metrics.json").is_file()]
        if not files:
            continue
        predictions = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        assert predictions.Physical_sample_ID.is_unique
        subsets = [("all", predictions)]
        if random_only:
            complete = predictions.observed_elements.eq(36)
            subsets.extend([("complete_D36", predictions[complete]),
                            ("natural_missing", predictions[~complete])])
        for subset, rows in subsets:
            labels = pd.Categorical(rows["true"], categories=LABELS).codes
            for context in CONTEXTS:
                for model in MODELS:
                    key = f"{model}_{context}"
                    pred = pd.Categorical(rows[key], categories=LABELS).codes
                    score = evaluate(labels, pred)
                    common = {"split": scheme, "context": context, "model": model}
                    if random_only:
                        common["test_subset"] = subset
                    excluded = {"per_class_f1", "confusion_matrix"}
                    entries.append({**common, "n_folds": len(files), "n_test": len(labels),
                                    **{k: v for k, v in score.items() if k not in excluded}})
                    for label, support in zip(LABELS, np.bincount(labels, minlength=len(LABELS))):
                        per_class.append({**common, "label": label, "support": support,
                                          "f1": score["per_class_f1"][label]})
    pd.DataFrame(entries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)


def run_case(predictor, x, y, ids, observed, groups, n_anchor, args, scheme, fold, test, guards=GUARDS):
    folder = args.out / scheme / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metrics.json").exists():
        previous = pd.read_csv(folder / "predictions.csv.gz")
        np.testing.assert_array_equal(previous.Physical_sample_ID, ids[test])
        return
    train = allowed_rows(groups, test, guards)
    contexts = paired_contexts(train, y, n_anchor, args.context, SEED + fold)
    audit = {"n_test": len(test), "n_test_BAB": int((y[test] == 0).sum()),
             "eligible_complete": int((train < n_anchor).sum()),
             "eligible_partial": int((train >= n_anchor).sum()), "results": {}}
    predictions = pd.DataFrame({"Physical_sample_ID": ids[test], "true": LABELS[y[test]],
                                "observed_elements": observed[test]})
    context_rows = []
    for mode, context in contexts.items():
        for key in guards:
            assert not np.intersect1d(groups[key][context], groups[key][test]).size
        n_partial = (context >= n_anchor).sum()
        print(f"{scheme} fold {fold} {mode}: {len(context)} context, {n_partial} partial, "
              f"{len(test)} query", flush=True)
        context_rows.append(pd.DataFrame({"mode": mode, "Physical_sample_ID": ids[context],
                                         "label": LABELS[y[context]], "observed_elements": observed[context]}))
        start = time.perf_counter()
        p_limix = limix_probabilities(predictor, x, y, context, test, args.batch)
        limix_seconds = time.perf_counter() - start
        gbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             verbosity=-1, n_jobs=4, random_state=SEED)
        columns = core.ELEMENTS
        gbm.fit(pd.DataFrame(x[context], columns=columns), y[context])
        p_gbm = gbm.predict_proba(pd.DataFrame(x[test], columns=columns))
        audit["results"][mode] = {"n_context": len(context), "n_partial": int(n_partial),
                                  "class_counts": dict(zip(LABELS, np.bincount(y[context]).tolist())),
                                  "limix_seconds": limix_seconds}
        for model, p in zip(MODELS, (p_limix, p_gbm)):
            assert p.shape == (len(test), len(LABELS)) and np.isfinite(p).all()
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)
            key = f"{model}_{mode}"
            predictions[key] = LABELS[p.argmax(axis=1)]
            for label, values in zip(LABELS, p.T):
                predictions[f"{key}_p_{label}"] = values
            audit["results"][mode][model] = evaluate(y[test], p.argmax(axis=1))
        print(f"Finished {scheme}/{fold}/{mode} in {limix_seconds:.1f}s", flush=True)
    pd.concat(context_rows).to_csv(folder / "contexts.csv", index=False)
    predictions.to_csv(folder / "predictions.csv.gz", index=False)
    core.dump(audit, folder / "metrics.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--members", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=SCHEMES)
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(10), default=list(range(10)))
    args = parser.parse_args()
    if core.sha(args.checkpoint) != WEIGHTS_SHA:
        raise ValueError("Checkpoint does not match the official LimiX-2M artifact.")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != UPSTREAM:
        raise ValueError("Use the V1.1.0 release implementation for these weights.")
    torch.set_num_threads(4)
    sys.path[:0] = [str(args.dependencies.resolve()), str(args.upstream.resolve())]
    from inference.predictor import LimiXPredictor

    prepared = ROOT / core.OUT_REL / "prepared_v2"
    pool, raw, folds, groups, meta = core.prepare(ROOT, prepared)
    x = np.log10(raw).astype(np.float32)
    y = pd.Categorical(pool.Secondary_label, categories=LABELS).codes
    ids = pool.Physical_sample_ID.to_numpy()
    observed = np.isfinite(x).sum(axis=1)
    assert len(ids) == len(np.unique(ids)) and not np.isinf(x).any() and y.min() == 0
    n_anchor = meta["anchor_n"]
    config = json.loads((args.upstream / "config/cls_default_noretrieval.json").read_text())
    config = config[:args.members]
    guards = ("chemical_guard",) if args.random_only else GUARDS
    procedure = {
        "checkpoint_sha": WEIGHTS_SHA, "script_sha": core.sha(__file__),
        "release": revision, "pytorch": str(torch.__version__), "device": args.device,
        "training_pool": meta, "guards": list(guards), "seed": SEED,
        "features": core.ELEMENTS, "transformation": "log10, unmeasured retained as NaN",
        "classes": LABELS.tolist(), "budget": args.context, "query_batch": args.batch,
        "config": config, "mix": "Replace up to half of each class with partial rows at equal n.",
    }
    if args.random_only:
        procedure["random_test"] = "Original D36 random folds extended to partial rows by chemical groups and stratified random assignment."
    else:
        procedure["natural_test"] = "First of eight province-grouped folds with at least 20 samples per class, selected before inference."
    path = args.out / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != procedure:
        raise ValueError("Output directory belongs to a different protocol.")
    core.dump(procedure, path)
    if args.random_only:
        plan = random_cases(n_anchor, y, folds, groups)
        manifest = [pd.DataFrame({"Physical_sample_ID": ids[test], "label": LABELS[y[test]],
                                  "observed_elements": observed[test], "fold": fold})
                    for _, fold, test in plan]
        pd.concat(manifest).to_csv(args.out / "random_test_manifest.csv.gz", index=False)
    else:
        plan = list(cases(n_anchor, y, folds, groups))
        natural = plan[-1][2]
        pd.DataFrame({"Physical_sample_ID": ids[natural], "label": LABELS[y[natural]],
                      "observed_elements": observed[natural]}).to_csv(args.out / "natural_test_manifest.csv", index=False)
    model = LimiXPredictor(device=torch.device(args.device), model_path=str(args.checkpoint.resolve()),
                           inference_config=config, seed=SEED)
    for scheme, fold, test in plan:
        if (args.random_only or scheme in args.schemes) and (fold in args.folds or scheme == "natural_missing"):
            run_case(model, x, y, ids, observed, groups, n_anchor, args, scheme, fold, test, guards)
            write_summary(args.out, args.random_only)


if __name__ == "__main__":
    main()
