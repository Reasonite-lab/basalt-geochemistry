"""Re-evaluate the tuned tree comparator on the exact 49,676-sample random folds."""

from pathlib import Path
import time

import numpy as np
import pandas as pd

import run_limix2m_geology_paper as paper
import run_limix2m_missing_d36 as run


SOURCE = run.ROOT / "reports/modeling/limix2m_random_5fold_2026-11-17"
OUT = paper.OUT / "expansion"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool, raw, _, groups, metadata = run.core.prepare(run.ROOT, run.ROOT / run.core.OUT_REL / "prepared_v2")
    x = np.log10(raw).astype(np.float32)
    ids = pd.Index(pool.Physical_sample_ID)
    y = pd.Categorical(pool.Secondary_label, categories=run.LABELS).codes
    complete = np.isfinite(x).all(axis=1)
    manifest = pd.read_csv(SOURCE / "split_manifest.csv.gz")
    assert len(manifest) == len(ids) == 49676 and manifest.Physical_sample_ID.is_unique
    all_predictions = []
    for fold in range(5):
        current = SOURCE / "random_5fold" / f"fold_{fold}"
        predictions = pd.read_csv(current / "predictions.csv.gz")
        expected = manifest.loc[manifest.fold.eq(fold + 1), "Physical_sample_ID"]
        np.testing.assert_array_equal(predictions.Physical_sample_ID, expected)
        test = ids.get_indexer(predictions.Physical_sample_ID)
        train = run.allowed_rows(groups, test, ("chemical_guard",))
        context = pd.read_csv(current / "contexts.csv")
        record = pd.DataFrame({"fold": fold + 1, "Physical_sample_ID": ids[test],
                               "D36_complete": complete[test], "true": run.LABELS[y[test]]})
        for mode in run.CONTEXTS:
            start = time.perf_counter()
            fit = ids.get_indexer(context.loc[context["mode"].eq(mode), "Physical_sample_ID"])
            assert len(fit) == 2048 and np.isin(fit, train).all()
            cols = [f"limix2m_{mode}_p_{label}" for label in run.LABELS]
            probability = {"LimiX-2M": predictions[cols].to_numpy()}
            tree, details = paper.tuned_tree(x, y, fit, run.SEED + fold)
            prob = tree.predict(x[test], num_threads=4)
            probability["LightGBM"] = prob
            probability["LightGBM校准"] = paper.apply_temperature(prob, details["temperature"])
            for model, p in probability.items():
                for label, values in zip(run.LABELS, p.T):
                    record[f"{mode}_{model}_{label}"] = values
            run.core.dump(details, OUT / f"fold_{fold+1}_{mode}.json")
            print(f"expansion {fold+1}/5 {mode} rounds={details['rounds']} {time.perf_counter()-start:.1f}s", flush=True)
        all_predictions.append(record)
    data = pd.concat(all_predictions, ignore_index=True)
    assert data.Physical_sample_ID.is_unique and len(data) == len(ids)
    data.to_parquet(OUT / "predictions.parquet", index=False)
    metrics, per_class = [], []
    for split, frame in [("全部", data), ("D36齐全", data[data.D36_complete]),
                         ("D36部分缺测", data[~data.D36_complete])]:
        labels = pd.Categorical(frame.true, categories=run.LABELS).codes
        for mode in run.CONTEXTS:
            for model in ["LimiX-2M", "LightGBM", "LightGBM校准"]:
                p = frame[[f"{mode}_{model}_{name}" for name in run.LABELS]].to_numpy()
                s = paper.probability_metrics(labels, p)
                kv = {"subset": split, "context": mode, "model": model, "n": len(labels)}
                metrics.append({**kv, **{k: v for k, v in s.items() if k not in {"confusion_matrix", "per_class_f1"}}})
                for i, name in enumerate(run.LABELS):
                    line = s["confusion_matrix"][i]
                    per_class.append({**kv, "class": name, "n_class": sum(line),
                                      "recall": line[i] / sum(line), "f1": s["per_class_f1"][name]})
    pd.DataFrame(metrics).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(per_class).to_csv(OUT / "per_class.csv", index=False)
    run.core.dump({"samples": len(data), "cohort_signature": metadata["source_signature"],
                   "split_sha": run.core.sha(SOURCE / "split_manifest.csv.gz"),
                   "source_predictions": [run.core.sha(SOURCE / "random_5fold" / f"fold_{k}" / "predictions.csv.gz")
                                          for k in range(5)],
                   "trees": "Inner 20% validation, log-loss early stopping, refit, train-side temperature.",
                   "limix": "Original recorded probabilities from identical context and holdout.",
                   "implementations": {Path(f).name: run.core.sha(f) for f in [__file__, paper.__file__]}},
                   OUT / "audit.json")


if __name__ == "__main__":
    main()
