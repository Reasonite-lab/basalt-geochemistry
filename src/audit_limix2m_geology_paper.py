"""Verify experimental coverage and consolidate paired results for the manuscript."""

from pathlib import Path

import numpy as np
import pandas as pd

from run_limix2m_geology_paper import (
    BUDGETS, MASKS, OUT, PANELS, SEEDS, FROZEN, ID, LABEL, context_rows, load_cohort,
    probability_metrics,
)
from run_limix2m_missing_d36 import LABELS, core


MODELS = ("LimiX-2M", "LightGBM")
METRICS = ["accuracy", "macro_f1_9_labels", "log_loss", "brier", "ece", "BAB_f1"]


def records(d, keys):
    return d[[*keys, *METRICS]].to_dict(orient="records")


def verify_expansion(scores):
    from run_limix2m_geology_expansion import SOURCE

    manifest = pd.read_csv(SOURCE / "split_manifest.csv.gz").set_index(ID, verify_integrity=True)
    prediction = pd.read_parquet(OUT / "expansion/predictions.parquet").set_index(ID, verify_integrity=True)
    assert len(prediction) == len(manifest) == 49676
    np.testing.assert_array_equal(prediction.fold, manifest.loc[prediction.index].fold)
    np.testing.assert_array_equal(prediction.true, manifest.loc[prediction.index].label)
    for _, row in scores.iterrows():
        data = prediction
        if row.subset != "全部":
            data = data[data.D36_complete.eq(row.subset == "D36齐全")]
        assert len(data) == row.n
        actual = pd.Categorical(data.true, categories=LABELS).codes
        probability = data[[f"{row.context}_{row.model}_{label}" for label in LABELS]]
        metrics = probability_metrics(actual, probability.to_numpy())
        np.testing.assert_allclose([metrics[m] for m in METRICS], row[METRICS].to_numpy(float), atol=1e-9)


def main():
    dataset, _, _, _ = load_cohort()
    y = pd.Categorical(dataset[LABEL], categories=LABELS).codes
    ids = dataset[ID].to_numpy()
    full = dataset.D36_complete.to_numpy()
    roles = pd.read_csv(FROZEN / "split_manifest.csv.gz").set_index(["seed", ID], verify_integrity=True)
    audit_rows = []
    for seed in SEEDS:
        part = roles.loc[seed].loc[ids].role.to_numpy()
        train, test = (np.flatnonzero(part == v) for v in ("train", "test"))
        contexts = dict(context_rows(train, y, seed))
        folder = OUT / f"seed_{seed}"
        cases = [(panel, "full_tree", test, None, ["全量LightGBM", "全量LightGBM校准"])
                 for panel in PANELS]
        cases += [(panel, f"{n}_原始测试", test, contexts[n], [*MODELS, "LimiX-2M校准", "LightGBM校准"])
                  for panel in PANELS for n in BUDGETS]
        cases += [("D36_74", f"2048_{s}", test[full[test]], contexts[2048],
                   [*MODELS, "仅测量标记", "LimiX-2M校准", "LightGBM校准"]) for s in MASKS]
        cases.append(("D36_74", "2048_原始测量标记", test, None, ["仅测量标记"]))
        if seed == SEEDS[0]:
            for add in (1, 2):
                cases += [(panel, f"{n}_上下文复抽{add}", test, context,
                           [*MODELS, "LimiX-2M校准", "LightGBM校准"])
                          for panel in PANELS for n, context in context_rows(train, y, seed + add)]
        for panel, experiment, te, context, models in cases:
            case = folder / panel / experiment
            saved = pd.read_parquet(case / "predictions.parquet")
            np.testing.assert_array_equal(saved[ID], ids[te])
            np.testing.assert_array_equal(saved.true, LABELS[y[te]])
            assert saved[ID].is_unique
            indices = case / "indices.npz"
            if context is not None:
                selection = np.load(indices)
                np.testing.assert_array_equal(context, selection["context"])
                np.testing.assert_array_equal(te, selection["test"])
                assert np.isin(selection["val"], context).all() and not np.intersect1d(context, te).size
                assert np.isin(selection["donor"], train).all() and not np.intersect1d(selection["donor"], test).size
            for model in models:
                probs = saved[[f"{model}_p_{label}" for label in LABELS]].to_numpy()
                stats = probability_metrics(y[te], probs)
                np.testing.assert_array_equal(saved[model], LABELS[probs.argmax(1)])
                audit_rows.append({"seed": seed, "panel": panel, "experiment": experiment,
                                   "model": model, "n": len(te), **{k: stats[k] for k in METRICS}})
    recomputed = pd.DataFrame(audit_rows)
    recomputed.to_csv(OUT / "audited_predictions.csv", index=False)
    files = {f.relative_to(OUT).as_posix(): core.sha(f) for f in OUT.rglob("predictions.parquet")}

    scores = pd.read_csv(OUT / "scores.csv")
    main_scores = scores[scores.scenario.eq("原始测试") & scores.subset.eq("全部")]
    main_summary = main_scores.groupby(["panel", "budget", "model"])[METRICS].agg(["mean", "std"])
    main_summary.columns = [f"{a}_{b}" for a, b in main_summary.columns]
    main_summary.to_csv(OUT / "main_summary.csv")
    main_models = main_scores[main_scores.model.isin(MODELS)]
    matched = main_models.pivot(index=["panel", "budget", "seed"], columns="model", values=METRICS)
    delta = matched.xs("LimiX-2M", level=1, axis=1) - matched.xs("LightGBM", level=1, axis=1)
    delta.to_csv(OUT / "paired_model_delta.csv")
    means = scores.groupby(["panel", "budget", "scenario", "model", "subset"], as_index=False)[METRICS].mean()
    expand = pd.read_csv(OUT / "expansion/scores.csv")
    verify_expansion(expand)
    by_class = pd.read_csv(OUT / "per_class.csv")
    sub = by_class[by_class.panel.eq("D36_74") & by_class.budget.eq(2048) &
                   by_class.scenario.eq("原始测试") & by_class.subset.eq("全部") &
                   by_class.model.isin(MODELS)]
    class_summary = sub.groupby(["model", "class"], as_index=False)[["n_class", "correct"]].sum()
    class_summary["recall"] = class_summary.correct / class_summary.n_class
    f1 = sub.groupby(["model", "class"], as_index=False).class_f1.mean()
    class_summary = class_summary.merge(f1, on=["model", "class"], validate="1:1")
    errors = pd.read_csv(OUT / "confusion.csv")
    err = errors[errors.panel.eq("D36_74") & errors.budget.eq(2048) &
                 errors.scenario.eq("原始测试") & errors.subset.eq("全部") &
                 errors.model.eq("LimiX-2M")].groupby(["true", "predicted"], as_index=False)["count"].sum()
    evidence = {
        "cohort": {"total": len(ids), "complete": int(full.sum()), "incomplete": int((~full).sum()),
                   "splits": len(SEEDS), "test_per_split": 4851},
        "main": main_summary.reset_index().to_dict(orient="records"),
        "mean": records(means, ["panel", "budget", "scenario", "model", "subset"]),
        "expansion": records(expand, ["subset", "context", "model", "n"]),
        "class": class_summary.to_dict(orient="records"),
        "confusion": err.to_dict(orient="records"),
        "first_split_original": records(
            main_models[main_models.seed.eq(SEEDS[0])], ["panel", "budget", "model", "seed"]),
        "raw_context_repeat": records(scores[scores.scenario.str.startswith("上下文复抽") & scores.subset.eq("全部")],
                                      ["panel", "budget", "scenario", "model", "seed"]),
        "matched_deltas": delta.reset_index().to_dict(orient="records"),
    }
    core.dump(evidence, OUT / "manuscript_evidence.json")
    core.dump({"status": "PASS", "checked_cases": len(files), "verified_predictions": len(audit_rows),
               "expansion_verified_metric_rows": len(expand),
               "files_sha256": files}, OUT / "experiment_audit.json")
    print(f"Verified {len(files)} prediction files and {len(audit_rows)} case/model pairs.")


if __name__ == "__main__":
    main()
