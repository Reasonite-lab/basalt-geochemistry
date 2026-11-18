"""Add partially observed D36 chemistry to the frozen D22/Sr–Nd experiment."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from clean_and_rerun_d36_2026_10_08 import E36, build_d36_74
from d22_process_common_2026_10_19 import BASE_PARAMS, E22, dump, exact_vector_hashes, sha256
from run_d22_srnd_native_nan_2026_11_13 import KEYS, OUT as D22_RUN
from run_d22_srnd_pattern_submodels_2026_11_13 import (
    ID, LABEL, OUT as FROZEN, PAIR, PHYSICAL, PILOT, ROOT, SEEDS, fit_predict,
)


OUT = ROOT / 'reports/modeling/d36_srnd_partial_chemistry_2026-11-13'
D22_MODELS = ['D22_42', 'D22_42_SrNd']
ALL_MODELS = ['mixed_D36_74', 'mixed_D36_74_SrNd']
COMPLETE_MODELS = ['complete_D36_74', 'complete_D36_74_SrNd']


def load_cohort():
    cohort = pd.read_csv(FROZEN / 'cohort_manifest.csv.gz', float_precision='round_trip').sort_values(ID).reset_index(drop=True)
    assert len(cohort) == 24251 and cohort[ID].is_unique and E36[:22] == E22
    columns = [ID, LABEL, *E36]
    sources = {
        'current_D36': pd.read_parquet(PILOT, columns=columns),
        'historical_D22_extension': pd.read_parquet(PHYSICAL, columns=columns),
    }
    raw = np.full((len(cohort), len(E36)), np.nan)
    assert set(cohort.cohort) == set(sources)
    for name, source in sources.items():
        rows = cohort.cohort.eq(name)
        selected = source.set_index(ID, verify_integrity=True).loc[cohort.loc[rows, ID]]
        np.testing.assert_array_equal(selected[LABEL], cohort.loc[rows, LABEL])
        raw[rows] = selected[E36].to_numpy(float)
    np.testing.assert_array_equal(exact_vector_hashes(raw[:, :22]), cohort.chemistry_sha256)
    assert not np.isinf(raw).any() and (raw[np.isfinite(raw)] > 0).all()
    complete = np.isfinite(raw).all(axis=1)
    np.testing.assert_array_equal(complete, cohort.cohort.eq('current_D36'))
    chemistry, names = build_d36_74(raw, allow_missing=True)
    np.testing.assert_array_equal(chemistry[complete], build_d36_74(raw[complete])[0])
    cohort['n_chem_observed'] = np.isfinite(raw).sum(axis=1)
    cohort['D36_complete'] = complete
    return cohort, chemistry, names, len(sources['current_D36'])


def run_seed(seed, ids, labels, complete, chemistry, isotope, roles):
    train, test = train_test_split(np.arange(len(labels)), stratify=labels, test_size=.2, random_state=seed)
    np.testing.assert_array_equal(np.sort(train), np.flatnonzero(roles == 'train'))
    np.testing.assert_array_equal(np.sort(test), np.flatnonzero(roles == 'test'))

    result = pd.DataFrame({'seed': seed, ID: ids[test]})
    full_train = train[complete[train]]
    full_test = test[complete[test]]
    assert len(np.unique(labels[full_train])) == len(np.unique(labels[full_test])) == 9
    designs = [chemistry, np.column_stack([chemistry, isotope])]
    for features, mixed, restricted in zip(designs, ALL_MODELS, COMPLETE_MODELS):
        result[mixed] = fit_predict(features[train], labels[train], features[test], seed)
        result[restricted] = pd.Series(pd.NA, index=result.index, dtype='string')
        result.loc[complete[test], restricted] = fit_predict(features[full_train], labels[full_train], features[full_test], seed)
    return result


def summarize():
    pred = pd.read_csv(OUT / 'predictions.csv.gz')
    models = [*D22_MODELS, *ALL_MODELS, *COMPLETE_MODELS]
    assert not pred.duplicated(['seed', ID]).any() and pred.groupby('seed').size().eq(4851).all()
    assert pred[D22_MODELS + ALL_MODELS].isin(pred.true.unique()).all().all()
    assert pred.loc[pred.D36_complete, COMPLETE_MODELS].isin(pred.true.unique()).all().all()
    assert pred.loc[~pred.D36_complete, COMPLETE_MODELS].isna().all().all()

    rows = []
    for seed, fold in pred.groupby('seed'):
        cohorts = [('all', fold), ('D36_complete', fold[fold.D36_complete]), ('partial_D36', fold[~fold.D36_complete])]
        for cohort_name, group in cohorts:
            eligible_models = models if cohort_name == 'D36_complete' else D22_MODELS + ALL_MODELS
            for pattern, part in [('all', group), *group.groupby('pattern')]:
                for model in eligible_models:
                    correct = int(part[model].eq(part.true).sum())
                    rows.append({'seed': seed, 'chemistry_coverage': cohort_name, 'isotope_pattern': pattern,
                                 'model': model, 'n_test': len(part), 'n_correct': correct, 'accuracy': correct / len(part)})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'accuracy_by_seed.csv', index=False)
    group_keys = ['chemistry_coverage', 'isotope_pattern', 'model']
    stats = detail.groupby(group_keys)
    summary = stats.accuracy.agg(mean='mean', sd='std').join(stats.n_test.agg(n_test_min='min', n_test_max='max'))
    summary.to_csv(OUT / 'accuracy_summary.csv')
    print((100 * summary.xs('all', level='isotope_pattern')['mean'].unstack()).round(3).to_string(), flush=True)


def main():
    cohort, chemistry, names, original_d36_n = load_cohort()
    roles = pd.read_csv(FROZEN / 'split_manifest.csv.gz').set_index(['seed', ID], verify_integrity=True)
    baseline = pd.read_csv(D22_RUN / 'predictions.csv.gz', usecols=[*KEYS, 'chemistry', 'native_nan'])
    baseline = baseline.rename(columns=dict(zip(['chemistry', 'native_nan'], D22_MODELS)))
    print(f'{len(cohort)} samples: {cohort.D36_complete.sum()} D36-complete; {(~cohort.D36_complete).sum()} D36-partial', flush=True)
    args = (cohort[ID].to_numpy(), cohort[LABEL].to_numpy(dtype=str), cohort.D36_complete.to_numpy(), chemistry, cohort[PAIR].to_numpy(float))
    predictions = []
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as workers:
        futures = [workers.submit(run_seed, seed, *args, roles.loc[seed].loc[cohort[ID], 'role'].to_numpy()) for seed in SEEDS]
        for seed, future in zip(SEEDS, futures):
            predictions.append(future.result())
            print(f'{seed}: {len(predictions[-1])} test rows', flush=True)

    pred = pd.concat(predictions, ignore_index=True).merge(baseline, on=['seed', ID], validate='one_to_one')
    expected_test = roles.loc[roles.role.eq('test')].reset_index()[KEYS].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(pred[KEYS].sort_values(KEYS).reset_index(drop=True), expected_test)
    pred = pred.merge(cohort[[ID, 'D36_complete', 'n_chem_observed']], on=ID, validate='many_to_one')
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT / 'predictions.csv.gz', index=False)
    cohort.groupby(['D36_complete', 'n_chem_observed', 'pattern']).size().rename('n').reset_index().to_csv(OUT / 'coverage.csv', index=False)
    summarize()
    inputs = [PILOT, PHYSICAL, FROZEN / 'cohort_manifest.csv.gz', FROZEN / 'split_manifest.csv.gz', D22_RUN / 'predictions.csv.gz']
    dump({'cohort': len(cohort), 'D36_original': original_d36_n, 'D36_complete': int(cohort.D36_complete.sum()),
          'seeds': list(SEEDS), 'chemistry_features': names, 'isotope_features': PAIR,
          'model_params': {**BASE_PARAMS, 'class_weight': None, 'n_jobs': 3}, 'lightgbm_version': lightgbm.__version__,
          'input_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
          'code_sha256': sha256(Path(__file__)), 'output_sha256': {path.name: sha256(path) for path in OUT.glob('*.csv*')}},
         OUT / 'audit.json')


if __name__ == '__main__':
    main()
