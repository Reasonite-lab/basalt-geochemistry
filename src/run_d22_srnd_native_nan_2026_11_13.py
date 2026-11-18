"""Replay the native-missing LightGBM experiment using the frozen D22 cohort."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from d22_process_common_2026_10_19 import BASE_PARAMS, E22, build_d22_42, dump, exact_vector_hashes, sha256
from run_d22_srnd_pattern_submodels_2026_11_13 import (
    ID, LABEL, OUT as ORIGINAL, PAIR, PATTERNS, PHYSICAL, PILOT, ROOT, SEEDS, fit_predict,
)


OUT = ROOT / 'reports/modeling/d22_srnd_native_nan_2026-11-13'
MODELS = ('chemistry', 'availability_only', 'native_nan')
KEYS = ['seed', ID, 'true', 'pattern']


def load_inputs():
    cohort = pd.read_csv(ORIGINAL / 'cohort_manifest.csv.gz', float_precision='round_trip').sort_values(ID).reset_index(drop=True)
    assert len(cohort) == 24251 and cohort[ID].is_unique and cohort.chemistry_sha256.is_unique
    columns = [ID, LABEL, *E22]
    sources = {
        'current_D36': pd.read_parquet(PILOT, columns=columns),
        'historical_D22_extension': pd.read_parquet(PHYSICAL, columns=columns),
    }
    raw = np.empty((len(cohort), len(E22)), dtype=float)
    assert set(cohort.cohort) == set(sources)
    for name, samples in sources.items():
        rows = cohort.cohort.eq(name)
        joined = samples.set_index(ID, verify_integrity=True).loc[cohort.loc[rows, ID]]
        np.testing.assert_array_equal(joined[LABEL], cohort.loc[rows, LABEL])
        raw[rows] = joined[E22].to_numpy(float)
    np.testing.assert_array_equal(exact_vector_hashes(raw), cohort.chemistry_sha256)

    chemistry, _ = build_d22_42(raw)
    isotope = cohort[PAIR].to_numpy(float)
    observed = np.isfinite(isotope)
    assert np.isfinite(isotope[observed]).all() and np.isnan(isotope[~observed]).all()
    np.testing.assert_array_equal(np.asarray(PATTERNS)[observed @ [1, 2]], cohort.pattern)
    designs = {
        'chemistry': chemistry,
        'availability_only': np.column_stack([chemistry, observed]),
        'native_nan': np.column_stack([chemistry, isotope]),
    }
    return cohort, designs


def run_seed(seed, cohort, roles, designs):
    labels = cohort[LABEL].to_numpy(dtype=str)
    train, test = train_test_split(np.arange(len(cohort)), stratify=labels, test_size=.2, random_state=seed)
    np.testing.assert_array_equal(np.sort(train), np.flatnonzero(roles == 'train'))
    np.testing.assert_array_equal(np.sort(test), np.flatnonzero(roles == 'test'))

    result = cohort.iloc[test][[ID, LABEL, 'pattern']].rename(columns={LABEL: 'true'}).copy()
    result.insert(0, 'seed', seed)
    for model, x in designs.items():
        result[model] = fit_predict(x[train], labels[train], x[test], seed)
    return result


def write_results(predictions):
    predictions.to_csv(OUT / 'predictions.csv.gz', index=False)
    replay = pd.read_csv(OUT / 'predictions.csv.gz').sort_values(KEYS).reset_index(drop=True)
    original = pd.read_csv(ORIGINAL / 'predictions.csv.gz', usecols=[*KEYS, *MODELS]).sort_values(KEYS).reset_index(drop=True)
    assert not replay.duplicated(['seed', ID]).any()
    assert replay[list(MODELS)].isin(replay.true.unique()).all().all()
    pd.testing.assert_frame_equal(replay, original[replay.columns])

    correct = replay[list(MODELS)].eq(replay.true, axis=0)
    group_keys = replay[['seed', 'pattern']]
    detail = pd.concat([pd.concat([group_keys, correct], axis=1), pd.concat([group_keys.assign(pattern='all'), correct], axis=1)])
    grouped = detail.groupby(['seed', 'pattern'])[list(MODELS)]
    accuracy = grouped.mean().join(grouped.size().rename('n_test')).reset_index()
    accuracy['native_vs_chemistry_pp'] = (accuracy.native_nan - accuracy.chemistry) * 100
    accuracy['native_vs_availability_pp'] = (accuracy.native_nan - accuracy.availability_only) * 100
    accuracy.to_csv(OUT / 'accuracy_by_seed.csv', index=False)
    summary = accuracy.groupby('pattern')[[*MODELS, 'native_vs_chemistry_pp', 'native_vs_availability_pp']].mean()
    summary.join(accuracy.groupby('pattern').n_test.agg(n_test_min='min', n_test_max='max')).to_csv(OUT / 'accuracy_summary.csv')
    pd.concat([replay[['seed', 'true']], correct], axis=1).groupby(['seed', 'true'])[list(MODELS)].mean().to_csv(OUT / 'class_recall_by_seed.csv')
    return summary


def main():
    cohort, designs = load_inputs()
    splits = pd.read_csv(ORIGINAL / 'split_manifest.csv.gz').set_index(['seed', ID], verify_integrity=True)
    print(f'Frozen cohort verified: {len(cohort)} physical samples', flush=True)
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as workers:
        futures = [workers.submit(run_seed, seed, cohort, splits.loc[seed].loc[cohort[ID], 'role'].to_numpy(), designs) for seed in SEEDS]
        predictions = []
        for seed, future in zip(SEEDS, futures):
            predictions.append(future.result())
            print(f'{seed}: {len(predictions[-1])} test predictions per model', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = write_results(pd.concat(predictions, ignore_index=True))
    paths = [PILOT, PHYSICAL, *(ORIGINAL / name for name in ('cohort_manifest.csv.gz', 'split_manifest.csv.gz', 'predictions.csv.gz'))]
    dump({'cohort': len(cohort), 'seeds': list(SEEDS), 'same_hard_labels_as_original': True,
          'lightgbm_version': lightgbm.__version__, 'missing_value': 'NaN', 'use_missing': True, 'zero_as_missing': False,
          'model_params': {**BASE_PARAMS, 'class_weight': None, 'n_jobs': 3},
          'input_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in paths},
          'code_sha256': sha256(Path(__file__)), 'output_sha256': {path.name: sha256(path) for path in OUT.glob('*.csv*')}},
         OUT / 'audit.json')
    print(summary.round(5).to_string(), flush=True)


if __name__ == '__main__':
    main()
