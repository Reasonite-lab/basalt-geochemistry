"""Hold the 22 common measurements fixed and randomize the remaining 14."""

import numpy as np
import pandas as pd

import run_limix2m_geology_paper as paper
from run_limix2m_geology_isotopes import baseline


def main():
    args, model = paper.args_and_model()
    assert args.out.resolve() != paper.OUT.resolve()
    sample, x, _, _ = paper.load_cohort()
    ids = sample[paper.ID].to_numpy()
    y = pd.Categorical(sample[paper.LABEL], categories=paper.limix.LABELS).codes
    split = pd.read_csv(paper.FROZEN / "split_manifest.csv.gz").set_index(["seed", paper.ID])
    raw = np.power(10, x[:, :36].astype(float))
    all_rows, histories = [], []
    for seed in args.seeds:
        roles = split.loc[seed].loc[ids].role.to_numpy()
        train, test = (np.flatnonzero(roles == role) for role in ("train", "test"))
        context = dict(paper.context_rows(train, y, seed))[2048]
        idx, donors, _ = paper.test_masks(raw, train, test, sample.D36_complete.to_numpy(), seed)
        allowed = np.isfinite(raw[donors, 22:])
        permutation = np.random.default_rng(seed).random(allowed.shape).argsort(axis=1)
        random = np.take_along_axis(allowed, permutation, axis=1)
        np.testing.assert_array_equal(random.sum(1), allowed.sum(1))
        q = raw[idx].copy()
        q[:, 22:] = np.where(random, q[:, 22:], np.nan)
        q = paper.build_d36_74(q, allow_missing=True)[0]
        history = {}
        tree, meta = paper.tuned_tree(x, y, context, seed, history)
        for name, values in history.items():
            for step, (loss, error) in enumerate(zip(values["multi_logloss"], values["multi_error"], strict=True), 1):
                histories.append(dict(seed=seed, subset=name, round=step, loss=loss, accuracy=1-error,
                                      selected_round=meta["rounds"]))
        for scenario in paper.MASKS:
            old = baseline(seed, scenario, context, idx, donors, ids, y)
            if scenario == "完整实测":
                np.testing.assert_allclose(tree.predict(paper.build_d36_74(raw[idx], allow_missing=True)[0]),
                                           old["LightGBM"], atol=1e-7, rtol=1e-7)
            for method, probs in old.items():
                stats = paper.probability_metrics(y[idx], probs)
                all_rows.append(dict(seed=seed, scenario=scenario, model=method, n=len(idx),
                                     accuracy=stats["accuracy"], loss=stats["log_loss"]))
        probs = {"LimiX-2M": paper.limix_predict(model, x, y, context, q, args.batch),
                 "LightGBM": tree.predict(q, num_threads=4)}
        frame = pd.DataFrame({paper.ID: ids[idx], "true": paper.limix.LABELS[y[idx]]})
        for method, p in probs.items():
            stats = paper.probability_metrics(y[idx], p)
            all_rows.append(dict(seed=seed, scenario="扩展14项随机遮挡", model=method, n=len(idx),
                                 accuracy=stats["accuracy"], loss=stats["log_loss"]))
            for label, values in zip(paper.limix.LABELS, p.T, strict=True):
                frame[f"{method}_{label}"] = values
        args.out.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.out / f"predictions_{seed}.parquet", index=False)
        print(seed, all_rows[-2:], flush=True)
    pd.DataFrame(all_rows).to_csv(args.out / "paired_mask_scores.csv", index=False)
    pd.DataFrame(histories).to_csv(args.out / "tree_training_history.csv", index=False)


if __name__ == "__main__":
    main()
