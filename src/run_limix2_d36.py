"""Evaluate the official LimiX-2 checkpoint on the original frozen D36 folds."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import incomplete_pool_fixed_test_2026_10_01 as core


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "data/frozen/experiment_v5_2026-07-30/D36_v9.parquet"
CHECKPOINT_SHA = "ed01f7a0a18d25451bc9d1058b479ba6b68048aa1ebde06156c8fcfaf8b8fe81"
SEED = 20261117


def evaluate(y, predictions):
    scores = f1_score(y, predictions, labels=np.arange(len(core.CLASSES)),
                      average=None, zero_division=0)
    supported = np.bincount(y, minlength=len(core.CLASSES)) > 0
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1_9_labels": float(scores.mean()),
        "macro_f1_supported": float(scores[supported].mean()),
        "BAB_f1": float(scores[0]),
        "per_class_f1": dict(zip(core.CLASSES, scores.tolist())),
        "confusion_matrix": confusion_matrix(y, predictions,
                                            labels=np.arange(len(scores))).tolist(),
    }


def read_inputs():
    anchor = pd.read_parquet(ANCHOR)
    if not anchor.Physical_sample_ID.is_unique:
        raise ValueError("Duplicate physical sample in frozen cohort.")
    folds, manifest, paths = core.load_folds(ROOT, anchor)
    x = core.features(anchor[core.ELEMENTS].to_numpy(dtype=float))
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features in the frozen D36 cohort.")
    y = anchor.Secondary_label.map({label: i for i, label in enumerate(core.CLASSES)})
    return anchor.Physical_sample_ID.to_numpy(), x, y.to_numpy(dtype=int), folds, manifest, paths


def run_fold(model, x, y, sample_ids, fold_map, group, split, fold, context_size, out):
    output = out / split / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").is_file():
        return
    train = np.flatnonzero(fold_map != fold)
    test = np.flatnonzero(fold_map == fold)
    if set(group.iloc[train]) & set(group.iloc[test]):
        raise ValueError(f"Protected group overlap in {split} fold {fold}.")
    context, _ = train_test_split(train, train_size=context_size, stratify=y[train],
                                  random_state=SEED + fold)
    context = np.sort(context)
    if not np.array_equal(np.unique(y[context]), np.arange(len(core.CLASSES))):
        raise ValueError(f"Missing class in {split} fold {fold} context.")
    pd.DataFrame({"Physical_sample_ID": sample_ids[context],
                  "label": np.array(core.CLASSES)[y[context]]}).to_csv(output / "context.csv", index=False)
    print(f"Starting {split} fold {fold}: context={len(context)} query={len(test)}", flush=True)
    start = time.perf_counter()
    probabilities = {"limix2": model.predict(x[context], y[context], x[test], task_type="Classification")}
    limix_seconds = time.perf_counter() - start
    for name, rows in [("lightgbm_context", context), ("lightgbm_full_train", train)]:
        # Fixed settings, with no tuning or early stopping on held-out data.
        baseline = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                  verbosity=-1, n_jobs=4, random_state=SEED)
        baseline.fit(x[rows], y[rows])
        probabilities[name] = baseline.predict_proba(x[test])
    predictions = pd.DataFrame({"Physical_sample_ID": sample_ids[test],
                                "true": np.array(core.CLASSES)[y[test]]})
    metrics = {
        "split": split, "fold": fold, "n_train": len(train), "n_context": len(context),
        "n_test": len(test), "n_BAB_test": int((y[test] == 0).sum()),
        "limix_seconds": limix_seconds, "models": {},
    }
    for name, p in probabilities.items():
        if p.shape != (len(test), len(core.CLASSES)) or not np.isfinite(p).all():
            raise ValueError(f"Invalid shape or values from {name}: {p.shape}")
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)
        predictions[name] = np.array(core.CLASSES)[p.argmax(axis=1)]
        for label, prob in zip(core.CLASSES, p.T):
            predictions[f"{name}_p_{label}"] = prob
        metrics["models"][name] = evaluate(y[test], p.argmax(axis=1))
    predictions.to_csv(output / "predictions.csv.gz", index=False)
    core.dump(metrics, output / "metrics.json")
    print(f"Finished {split} fold {fold} in {limix_seconds:.1f}s, "
          f"LimiX-2 macro-F1 = {metrics['models']['limix2']['macro_f1_9_labels']:.4f}", flush=True)


def summarize(out):
    labels = {label: i for i, label in enumerate(core.CLASSES)}
    rows = []
    for split in core.SPLITS:
        paths = sorted((out / split).glob("fold_*/predictions.csv.gz"))
        paths = [p for p in paths if p.with_name("metrics.json").exists()]
        if not paths:
            continue
        pooled = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        if not pooled.Physical_sample_ID.is_unique:
            raise ValueError(f"Repeated test samples in {split}.")
        for model in ["limix2", "lightgbm_context", "lightgbm_full_train"]:
            score = evaluate(pooled["true"].map(labels), pooled[model].map(labels))
            rows.append({"split": split, "model": model, "n_folds": len(paths),
                         "n_test": len(pooled), **{k: v for k, v in score.items()
                                                  if k not in {"per_class_f1", "confusion_matrix"}}})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=core.SPLITS, default=core.SPLITS)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(10), default=[0])
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--query-batch", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rmsnorm", choices=["torch", "triton"], default="torch")
    args = parser.parse_args()
    if core.sha(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Checkpoint SHA-256 does not match the official LimiX-2 release.")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.upstream.resolve()))
    if args.dependencies:
        sys.path.insert(0, str(args.dependencies.resolve()))
    from inference.predictor import LimiXPredictor

    sample_ids, x, y, folds, manifest, split_paths = read_inputs()
    config = json.loads((args.upstream / "config/cls_default_noretrieval_v2.json").read_text())
    config["pipelines"] = config["pipelines"][:1]
    protocol = {
        "cohort": str(ANCHOR.relative_to(ROOT)), "n_samples": len(x),
        "features": core.FEATURE_NAMES, "classes": core.CLASSES,
        "source_hashes": {str(p.relative_to(ROOT)): core.sha(p) for p in [ANCHOR, *split_paths]},
        "script_sha256": core.sha(__file__),
        "checkpoint_sha256": CHECKPOINT_SHA, "torch_version": str(torch.__version__),
        "upstream_commit": subprocess.check_output(["git", "-C", str(args.upstream),
                                                     "rev-parse", "HEAD"], text=True).strip(),
        "device": args.device, "rmsnorm": args.rmsnorm, "seed": SEED, "context_size": args.context,
        "query_batch": args.query_batch, "inference_config": config,
        "context": "Fixed stratified subset of the training side, one per fold.",
    }
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Use a separate output directory for a different protocol.")
    core.dump(protocol, protocol_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint["config"] = {**checkpoint["config"], "rmsnorm_impl": args.rmsnorm}
    model = LimiXPredictor(device=torch.device(args.device), model_path=str(args.checkpoint),
                           inference_config=config, seed=SEED, categorical_features_indices=[],
                           test_batch_mode="fixed", test_batch_size=args.query_batch,
                           enable_preprocess_parallel=False, ckpt=checkpoint)
    del checkpoint
    for fold in args.folds:
        for split in args.splits:
            group = manifest[split.replace("_10fold", "_component")]
            run_fold(model, x, y, sample_ids, folds[split], group, split, fold, args.context, args.out)
            summarize(args.out)


if __name__ == "__main__":
    main()
