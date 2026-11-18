"""Add measured Pb isotope ratios to the frozen D36/Sr–Nd random experiment."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from d22_process_common_2026_10_19 import BASE_PARAMS, CLASSES, dump, sha256
from expand_local_isotope_cohorts_2026_11_13 import QA
from run_d22_srnd_pattern_submodels_2026_11_13 import ACCEPTED_GRADES, fit_predict
from run_d36_srnd_partial_chemistry_2026_11_13 import (
    FROZEN, ID, KEYS, LABEL, OUT as BASELINE, PAIR, ROOT, SEEDS, load_cohort,
)


OUT = ROOT / 'reports/modeling/d36_srnd_pb_2026-11-13'
PB = ['PB206_PB204', 'PB207_PB204', 'PB208_PB204']
GRADES = [name + '_grade' for name in PB]
MODELS = ['D36_SrNd', 'D36_SrNd_Pb_flags', 'D36_SrNd_Pb']


def load_pb(cohort):
    qa = pd.read_parquet(QA, columns=[ID, LABEL, *PB, *GRADES])
    pb = cohort[[ID, LABEL, 'Reference', 'Province']].merge(qa, on=[ID, LABEL], how='left', validate='one_to_one')
    np.testing.assert_array_equal(pb[ID], cohort[ID])
    accepted = pb[GRADES].isin(ACCEPTED_GRADES).to_numpy()
    pb_values = pb[PB].to_numpy(float)
    assert np.isfinite(pb_values[accepted]).all() and (pb_values[accepted] > 0).all()
    pb[PB] = np.where(accepted, pb_values, np.nan)
    pb['Pb_available_n'] = accepted.sum(axis=1)
    pb['Pb_A_any'] = pb[GRADES].eq('A_TRAIN_CORE').any(axis=1)
    assert pb.Pb_available_n.gt(0).sum() == 5965 and pb.Pb_available_n.eq(3).sum() == 5922
    triplets = pb.loc[pb.Pb_available_n.eq(3), PB]
    duplicate = triplets.loc[triplets.duplicated(keep=False)]
    pb['duplicate_triplet'] = pb.index.isin(duplicate.index)
    return pb


def run_seed(seed, ids, labels, x_flags, x_values, roles):
    train, test = train_test_split(np.arange(len(labels)), stratify=labels, test_size=.2, random_state=seed)
    np.testing.assert_array_equal(np.sort(train), np.flatnonzero(roles == 'train'))
    np.testing.assert_array_equal(np.sort(test), np.flatnonzero(roles == 'test'))
    return pd.DataFrame({'seed': seed, ID: ids[test],
                         MODELS[1]: fit_predict(x_flags[train], labels[train], x_flags[test], seed),
                         MODELS[2]: fit_predict(x_values[train], labels[train], x_values[test], seed)})


def summarize():
    pred = pd.read_csv(OUT / 'predictions.csv.gz')
    assert not pred.duplicated(['seed', ID]).any() and pred.groupby('seed').size().eq(4851).all()
    assert pred[MODELS].isin(CLASSES).all().all()
    detail = []
    classes = []
    for seed, fold in pred.groupby('seed'):
        groups = {'all': fold, 'Pb_available': fold[fold.Pb_available_n.gt(0)],
                  'Pb_unavailable': fold[fold.Pb_available_n.eq(0)],
                  'all_excluding_duplicate_triplets': fold[~fold.duplicate_triplet],
                  'Pb_available_excluding_duplicate_triplets': fold[fold.Pb_available_n.gt(0) & ~fold.duplicate_triplet]}
        for name, group in groups.items():
            row = {'seed': seed, 'subset': name, 'n_test': len(group)}
            for model in MODELS:
                row[f'{model}_accuracy'] = group[model].eq(group.true).mean()
                row[f'{model}_macro_f1'] = f1_score(group.true, group[model], labels=CLASSES, average='macro', zero_division=0)
            row['Pb_values_vs_baseline_pp'] = 100 * (row[f'{MODELS[2]}_accuracy'] - row[f'{MODELS[0]}_accuracy'])
            row['Pb_flags_vs_baseline_pp'] = 100 * (row[f'{MODELS[1]}_accuracy'] - row[f'{MODELS[0]}_accuracy'])
            row['Pb_values_vs_flags_pp'] = 100 * (row[f'{MODELS[2]}_accuracy'] - row[f'{MODELS[1]}_accuracy'])
            detail.append(row)
        for label, group in fold.groupby('true'):
            classes.append({'seed': seed, 'class': label, 'n_test': len(group),
                            **{model: group[model].eq(group.true).mean() for model in MODELS}})
    metrics = pd.DataFrame(detail)
    metrics.to_csv(OUT / 'metrics_by_seed.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT / 'class_recall_by_seed.csv', index=False)
    grouped = metrics.groupby('subset')
    summary = grouped[metrics.columns[3:]].mean().join(grouped.n_test.agg(n_test_min='min', n_test_max='max'))
    summary.to_csv(OUT / 'metrics_summary.csv')
    print(summary.round(5).to_string(), flush=True)


def main():
    cohort, chemistry, chemical_features, _ = load_cohort()
    pb = load_pb(cohort)
    split = pd.read_csv(FROZEN / 'split_manifest.csv.gz').set_index(['seed', ID], verify_integrity=True)
    baseline_path = BASELINE / 'predictions.csv.gz'
    baseline = pd.read_csv(baseline_path, usecols=[*KEYS, 'mixed_D36_74_SrNd'])
    baseline = baseline.rename(columns={'mixed_D36_74_SrNd': MODELS[0]})
    labels = cohort[LABEL].to_numpy(dtype=str)
    base_x = np.column_stack([chemistry, cohort[PAIR].to_numpy(float)])
    values = pb[PB].to_numpy(float)
    flags = np.isfinite(values)
    x_flags = np.column_stack([base_x, flags])
    x_values = np.column_stack([base_x, values])
    print(f'{len(cohort)} samples; {flags.any(axis=1).sum()} with Pb; {pb.duplicate_triplet.sum()} duplicate-triplet rows.', flush=True)
    predictions = []
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        futures = [pool.submit(run_seed, seed, cohort[ID].to_numpy(), labels, x_flags, x_values,
                               split.loc[seed].loc[cohort[ID], 'role'].to_numpy()) for seed in SEEDS]
        for seed, future in zip(SEEDS, futures):
            predictions.append(future.result())
            print(f'{seed}: {len(predictions[-1])} test rows', flush=True)

    pred = pd.concat(predictions, ignore_index=True).merge(baseline, on=['seed', ID], validate='one_to_one')
    expected = split.loc[split.role.eq('test')].reset_index()[KEYS].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(pred[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    pred = pred.merge(pb[[ID, 'Pb_available_n', 'Pb_A_any', 'duplicate_triplet']], on=ID, validate='many_to_one')
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT / 'predictions.csv.gz', index=False)
    pb[[ID, LABEL, *PB, *GRADES, 'Pb_available_n', 'Pb_A_any', 'duplicate_triplet']].to_csv(OUT / 'pb_manifest.csv.gz', index=False)
    summarize()
    inputs = (QA, FROZEN / 'cohort_manifest.csv.gz', FROZEN / 'split_manifest.csv.gz', baseline_path)
    dump({'cohort': len(cohort), 'seeds': SEEDS, 'baseline_features': chemical_features + PAIR, 'Pb_features': PB,
          'accepted_grades': sorted(ACCEPTED_GRADES), 'Pb_available_n': int(flags.any(axis=1).sum()),
          'duplicate_complete_triplet_rows': int(pb.duplicate_triplet.sum()), 'lightgbm_version': lightgbm.__version__,
          'model_params': {**BASE_PARAMS, 'class_weight': None, 'n_jobs': 3},
          'input_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in inputs}, 'code_sha256': sha256(Path(__file__)),
          'output_sha256': {p.name: sha256(p) for p in OUT.glob('*.csv*')}}, OUT / 'audit.json')


if __name__ == '__main__':
    main()
