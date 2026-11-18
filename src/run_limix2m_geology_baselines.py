"""Fit additional baselines on the matched 2,048-label chemical comparison."""

import json

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

import run_limix2m_geology_paper as paper
from run_limix2m_geology_isotopes import baseline


OUT = paper.OUT.with_name("limix2m_geology_baselines_2026-11-17")


def models(x, y, context, seed):
    inner, val = train_test_split(context, test_size=.2, stratify=y[context], random_state=seed)
    xgb = dict(max_depth=5, learning_rate=.05, subsample=.8, colsample_bytree=.8, tree_method="hist",
               objective="multi:softprob", eval_metric="mlogloss", n_jobs=4, random_state=seed)
    fit_xgb = XGBClassifier(**xgb, n_estimators=600, early_stopping_rounds=45)
    fit_xgb.fit(x[inner], y[inner], eval_set=[(x[val], y[val])], verbose=False)
    yield "XGBoost", XGBClassifier(**xgb, n_estimators=fit_xgb.best_iteration + 1).fit(x[context], y[context])
    cat = dict(depth=6, learning_rate=.08, loss_function="MultiClass", thread_count=4, random_seed=seed,
               verbose=False, allow_writing_files=False)
    fit_cat = CatBoostClassifier(**cat, iterations=600, early_stopping_rounds=45)
    fit_cat.fit(x[inner], y[inner], eval_set=(x[val], y[val]))
    yield "CatBoost", CatBoostClassifier(**cat, iterations=fit_cat.get_best_iteration() + 1).fit(x[context], y[context])
    forest = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                           RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=4, random_state=seed))
    yield "随机森林", forest.fit(x[context], y[context])


def main():
    cohort, x, _, _ = paper.load_cohort()
    ids = cohort[paper.ID].to_numpy()
    y = pd.Categorical(cohort[paper.LABEL], categories=paper.limix.LABELS).codes
    roles = pd.read_csv(paper.FROZEN / "split_manifest.csv.gz").set_index(["seed", paper.ID])
    OUT.mkdir(exist_ok=True, parents=True)
    results = []
    for seed in paper.SEEDS:
        role = roles.loc[seed].loc[ids].role.to_numpy()
        train, test = (np.flatnonzero(role == v) for v in ("train", "test"))
        context = dict(paper.context_rows(train, y, seed))[2048]
        assert len(test) == 4851 and len(train) == 19400
        frame = pd.DataFrame({paper.ID: ids[test], "true": paper.limix.LABELS[y[test]]})
        saved = baseline(seed, "原始测试", context, test, np.empty(0, int), ids, y)
        for label, estimator in models(x, y, context, seed):
            np.testing.assert_array_equal(estimator.classes_, np.arange(9))
            saved[label] = estimator.predict_proba(x[test])
            print(seed, label, flush=True)
        for name, probs in saved.items():
            metrics = paper.probability_metrics(y[test], probs)
            results.append({"seed": seed, "model": name, **{key: metrics[key] for key in
                            ("accuracy", "macro_f1_9_labels", "log_loss")}})
            for label, values in zip(paper.limix.LABELS, probs.T, strict=True):
                frame[f"{name}_{label}"] = values
        frame.to_parquet(OUT / f"predictions_{seed}.parquet", index=False)
    scores = pd.DataFrame(results)
    scores.to_csv(OUT / "scores_by_seed.csv", index=False)
    assert scores.groupby("model").size().eq(5).all()
    summary = scores.groupby("model", as_index=False)[["accuracy", "macro_f1_9_labels", "log_loss"]].mean()
    summary.to_csv(OUT / "scores_summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
