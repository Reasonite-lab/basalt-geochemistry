"""Offline D22 adaptation of Mercaldo and Blume's pattern submodels (kxy040)."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from d22_process_common_2026_10_19 import BASE_PARAMS, E22, build_d22_42, dump, exact_vector_hashes, sha256
from expand_local_isotope_cohorts_2026_11_13 import PAIR, PHYSICAL, PILOT, QA, QUARANTINE, ROOT


OUT = ROOT / 'reports/modeling/d22_srnd_pattern_submodels_2026-11-13'
ID = 'Physical_sample_ID'
LABEL = 'Secondary_label'
SOURCE = ['Database', 'Source_record_ID']
META = [ID, LABEL, *SOURCE, 'component_member_ids', 'Reference', 'Province']
GRADES = [field + '_grade' for field in PAIR]
ACCEPTED_GRADES = {'A_TRAIN_CORE', 'B_SENSITIVITY'}
SEEDS = (20261113, 20261114, 20261115, 20261116, 20261117)
PATTERNS = ('neither', 'Sr_only', 'Nd_only', 'both')
MODELS = ('chemistry', 'availability_only', 'native_nan', 'pattern_submodels', 'dual_routing')
PAPER = Path('C:/Users/ASUS/Desktop/_kxy040.pdf')


def build_cohort():
    pilot = pd.read_parquet(PILOT, columns=[*META, *E22])
    qa = pd.read_parquet(QA, columns=[ID, LABEL, 'pool', *PAIR, *GRADES])
    physical = pd.read_parquet(PHYSICAL, columns=[*META, *E22, 'training_eligible_provisional', 'component_label_conflict'])
    quarantined = pd.read_csv(QUARANTINE, usecols=[ID])
    assert all(frame[ID].is_unique for frame in (pilot, qa, physical))
    pilot_labels = pilot[[ID, LABEL]].merge(qa[[ID, LABEL]], on=ID, suffixes=('_pilot', '_qa'), validate='one_to_one')
    assert pilot_labels[f'{LABEL}_pilot'].eq(pilot_labels[f'{LABEL}_qa']).all()

    historical = qa.loc[qa.pool.eq('incomplete_extra'), [ID, LABEL]].merge(physical, on=[ID, LABEL], validate='one_to_one')
    assert len(historical) == int(qa.pool.eq('incomplete_extra').sum())
    historical_eligible = historical.training_eligible_provisional & ~historical.component_label_conflict
    historical_eligible &= ~historical[ID].isin(pilot[ID]) & ~historical[ID].isin(quarantined[ID])
    extension = historical.loc[historical_eligible, [*META, *E22]]
    frame = pd.concat([pilot.assign(cohort='current_D36'), extension.assign(cohort='historical_D22_extension')], ignore_index=True)
    frame = frame.merge(qa[[ID, LABEL, *PAIR, *GRADES]], how='left', on=[ID, LABEL], validate='one_to_one')
    assert frame[ID].is_unique
    for value, grade in zip(PAIR, GRADES):
        accepted = frame[grade].isin(ACCEPTED_GRADES)
        assert np.isfinite(frame.loc[accepted, value]).all() and frame.loc[accepted, value].gt(0).all()
        frame[value] = frame[value].where(accepted)

    raw = frame[E22].to_numpy(float)
    complete = np.isfinite(raw).all(axis=1) & (raw > 0).all(axis=1)
    eligible = frame.loc[complete].copy()
    eligible['chemistry_sha256'] = exact_vector_hashes(raw[complete])
    eligible['pattern'] = np.asarray(PATTERNS)[eligible[PAIR[0]].notna().astype(int) + 2 * eligible[PAIR[1]].notna().astype(int)]

    members = eligible[[ID, 'component_member_ids']].assign(member=lambda f: f.component_member_ids.str.split(' || ', regex=False)).explode('member')
    members = members[[ID, 'member']].dropna().drop_duplicates()
    members = members[members.member.ne('')]
    shared_ids = members.loc[members.member.duplicated(keep=False), ID]
    source_keys = eligible[SOURCE].notna().all(axis=1) & eligible.Source_record_ID.astype(str).str.strip().ne('')
    paired = eligible[PAIR].notna().all(axis=1)
    flags = pd.DataFrame({
        'shared_source_record': source_keys & eligible.duplicated(SOURCE, keep=False),
        'shared_component_member': eligible[ID].isin(shared_ids),
        'duplicate_D22_vector': eligible.chemistry_sha256.duplicated(keep=False),
        'duplicate_measured_pair': paired & eligible.duplicated(PAIR, keep=False),
    })
    exclude = flags.any(axis=1)
    eligible.loc[exclude, [ID, LABEL, 'cohort', 'pattern', 'chemistry_sha256']].join(flags).to_csv(OUT / 'excluded_identity_review.csv', index=False)
    cohort = eligible.loc[~exclude].sort_values(ID).reset_index(drop=True)
    assert cohort[LABEL].nunique() == 9 and cohort[ID].is_unique and cohort.chemistry_sha256.is_unique
    assert not cohort.loc[cohort.pattern.eq('both')].duplicated(PAIR).any()
    manifest_cols = [*META, 'cohort', 'pattern', 'chemistry_sha256', *PAIR, *GRADES]
    cohort[manifest_cols].to_csv(OUT / 'cohort_manifest.csv.gz', index=False)
    cohort.groupby(['cohort', 'pattern', LABEL]).size().rename('n').reset_index().to_csv(OUT / 'cohort_coverage.csv', index=False)
    return cohort, {
        'current_D36_before_D22_screen': len(pilot), 'eligible_historical_extension_before_D22_screen': len(extension),
        'total_complete_D22_before_identity_review': len(eligible), 'excluded_identity_review': int(exclude.sum()),
        'overlapping_identity_flags': {name: int(count) for name, count in flags.sum().items()},
        'final_cohort': len(cohort), 'patterns': cohort.pattern.value_counts().to_dict(),
        'missing_QA_link_count_in_final_cohort': int((~cohort[ID].isin(qa[ID])).sum()),
    }


def fit_predict(train_x, train_y, test_x, seed):
    model = LGBMClassifier(**{**BASE_PARAMS, 'class_weight': None, 'n_jobs': 3}, random_state=seed)
    model.fit(train_x, train_y)
    return model.classes_[model.booster_.predict(test_x, num_threads=3).argmax(axis=1)]


def run_split(seed, ids, labels, patterns, x, iso):
    train, test = train_test_split(np.arange(len(labels)), stratify=labels, test_size=.2, random_state=seed)
    assert len(np.intersect1d(train, test)) == 0 and len(train) + len(test) == len(labels)
    assert len(np.unique(labels[train])) == len(np.unique(labels[test])) == 9

    observed = np.isfinite(iso)
    inputs = {'chemistry': x, 'availability_only': np.column_stack([x, observed]), 'native_nan': np.column_stack([x, iso])}
    predictions = {name: fit_predict(values[train], labels[train], values[test], seed) for name, values in inputs.items()}
    predictions['pattern_submodels'] = np.empty(len(test), dtype=labels.dtype)
    predictions['dual_routing'] = predictions['chemistry'].copy()
    pattern_fit = []
    for code, pattern in enumerate(PATTERNS):
        selected = [index for index in range(len(PAIR)) if code & (1 << index)]
        features = np.column_stack([x, iso[:, selected]])
        fit = train[patterns[train] == pattern]
        cases = patterns[test] == pattern
        # The paper uses 2p as a heuristic. Also require training coverage of all nine classes.
        use_ps = len(fit) >= 2 * features.shape[1] and len(np.unique(labels[fit])) == 9
        # CCS uses training rows with every predictor needed by this pattern observed.
        if not use_ps:
            fit = train[observed[train][:, selected].all(axis=1)]
        assert np.isfinite(features[fit]).all() and len(np.unique(labels[fit])) == 9
        if cases.any():
            result = fit_predict(features[fit], labels[fit], features[test[cases]], seed)
            predictions['pattern_submodels'][cases] = result
            if pattern == 'both':
                predictions['dual_routing'][cases] = result
        pattern_fit.append({'seed': seed, 'pattern': pattern, 'strategy': 'PS' if use_ps else 'CCS',
                            'n_train_same_pattern': int((patterns[train] == pattern).sum()),
                            'n_fit': len(fit), 'n_test': int(cases.sum()), 'n_predictors': features.shape[1],
                            **{f'train_{label}': int((labels[fit] == label).sum()) for label in np.unique(labels)}})

    split = pd.DataFrame({'seed': seed, ID: ids, 'true': labels, 'pattern': patterns, 'role': 'train'})
    split.loc[test, 'role'] = 'test'
    pred = pd.DataFrame({'seed': seed, ID: ids[test], 'true': labels[test], 'pattern': patterns[test], **predictions})
    assert all(pred[model].isin(np.unique(labels)).all() for model in MODELS)
    return split, pred, pattern_fit


def verify_and_summarize():
    pred = pd.read_csv(OUT / 'predictions.csv.gz')
    split = pd.read_csv(OUT / 'split_manifest.csv.gz')
    keys = ['seed', ID, 'true', 'pattern']
    assert not split.duplicated(['seed', ID]).any() and not pred.duplicated(['seed', ID]).any()
    expected = split.loc[split.role.eq('test'), keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, pred[keys].sort_values(keys).reset_index(drop=True))
    assert pred[list(MODELS)].isin(split.true.unique()).all().all()
    assert pred.loc[pred.pattern.eq('both'), 'dual_routing'].equals(pred.loc[pred.pattern.eq('both'), 'pattern_submodels'])
    assert pred.loc[pred.pattern.ne('both'), 'dual_routing'].equals(pred.loc[pred.pattern.ne('both'), 'chemistry'])

    rows = []
    for seed, fold in pred.groupby('seed'):
        for pattern, part in [('all', fold), *fold.groupby('pattern')]:
            for model in MODELS:
                n_correct = sum(predicted == actual for predicted, actual in zip(part[model], part.true))
                rows.append({'seed': seed, 'pattern': pattern, 'model': model, 'n_test': len(part),
                             'n_correct': n_correct, 'accuracy': n_correct / len(part)})
    accuracy = pd.DataFrame(rows)
    accuracy.to_csv(OUT / 'accuracy_by_seed.csv', index=False)
    summary = accuracy.groupby(['pattern', 'model']).accuracy.agg(mean='mean', sd='std', min='min', max='max').reset_index()
    baseline = summary.loc[summary.model.eq('chemistry')].set_index('pattern')['mean']
    summary['delta_pp_vs_chemistry'] = 100 * (summary['mean'] - summary.pattern.map(baseline))
    summary.to_csv(OUT / 'accuracy_summary.csv', index=False)

    long = pred.melt(id_vars=keys, value_vars=list(MODELS), var_name='model', value_name='prediction')
    long['correct'] = long.true.eq(long.prediction)
    long.groupby(['seed', 'model', 'true']).correct.agg(n_test='size', n_correct='sum', recall='mean').reset_index().to_csv(OUT / 'class_recall_by_seed.csv', index=False)
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, audit = build_cohort()
    print(audit, flush=True)
    x, names = build_d22_42(cohort[E22].to_numpy(float))
    data = (cohort[ID].to_numpy(dtype=str), cohort[LABEL].to_numpy(dtype=str), cohort.pattern.to_numpy(dtype=str),
            x, cohort[PAIR].to_numpy(float))
    splits, predictions, fit_rows = [], [], []
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as workers:
        futures = [workers.submit(run_split, seed, *data) for seed in SEEDS]
        for seed, future in zip(SEEDS, futures):
            split, pred, fit = future.result()
            splits.append(split)
            predictions.append(pred)
            fit_rows.extend(fit)
            print(f'{seed}: {len(pred)} test samples, ' + ', '.join(f'{row["pattern"]}={row["strategy"]}' for row in fit), flush=True)
    pd.concat(splits, ignore_index=True).to_csv(OUT / 'split_manifest.csv.gz', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / 'predictions.csv.gz', index=False)
    pd.DataFrame(fit_rows).to_csv(OUT / 'pattern_training_support.csv', index=False)
    summary = verify_and_summarize()
    dump({**audit, 'split': '80/20 by physical sample, stratified by class', 'seeds': list(SEEDS),
          'D22_42_features': names, 'isotopes': PAIR, 'accepted_grades': sorted(ACCEPTED_GRADES),
          'paper_doi': '10.1093/biostatistics/kxy040', 'paper_pdf_sha256': sha256(PAPER) if PAPER.exists() else None,
          'input_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in (QA, PHYSICAL, PILOT, QUARANTINE)},
          'code_sha256': sha256(Path(__file__)), 'files_sha256': {p.name: sha256(p) for p in OUT.glob('*.csv*')}}, OUT / 'audit.json')
    print(summary.round(5).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
