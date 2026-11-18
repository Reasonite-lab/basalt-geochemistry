"""Expand the local measured-isotope candidate inventory and audit two primary workbooks."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from build_d36_isotope_pilot_2026_11_13 import BOUNDS, IDS, PANEL, TOL, digest, key
from incomplete_pool_fixed_test_2026_10_01 import ELEMENTS

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/modeling'
OUT = REPORTS / 'isotope_cohort_expansion_2026-11-13'
PILOT = REPORTS / 'd36_isotope_geologic_units_pilot_2026-11-13/D36_isotope_pilot.parquet'
QA = REPORTS / 'isotope_quality_verification_2026-10-24/isotope_sample_training_table.parquet'
PHYSICAL = ROOT / 'data/exports/all_classes_physical_sample_element_table_prospective_v9_2026-07-30.parquet'
QUARANTINE = ROOT / 'reports/data_quality/d36_identity_clean_audit_2026-10-08/quarantined_rows.csv'
SO255 = ROOT / 'data/raw/earthchem_expansion/all_dataset_1723/1723-4_Hauff_et_alSO255_Isotopes_BulkSample_ECL_Template_2021.xlsx'
CIR = ROOT / 'data/raw/earthchem_d36_expansion_2026_07_28_round4/2112_extracted/2112-2_Vincent_CIR-isotopes-2022.xlsx'
PAIR = ['SR87_SR86', 'ND143_ND144']


def coordinate_distance(a, b, c, d):
    a, b, c, d = np.radians(np.asarray([a, b, c, d], dtype=float))
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(np.sin((a-c)/2)**2 + np.cos(a)*np.cos(c)*np.sin((b-d)/2)**2, 0, 1)))


def method_code(value):
    return str(int(float(value)))


def source_records(path, name, doi, scope, pilot):
    data = pd.read_excel(path, '3 Data', header=None)
    samples = pd.read_excel(path, '2 Samples', header=None)
    metadata = pd.read_excel(path, '4 Primary Analytical Metadata', header=None)
    loc_col = list(samples.iloc[4]).index('LOCATION KEYWORDS')
    locations = samples.iloc[7:, [0, 2, 3, loc_col]].dropna(subset=[0]).copy()
    locations.columns = ['sample', 'latitude', 'longitude', 'source_location']
    locations['sample_key'] = locations['sample'].map(key)
    ambiguous = set(locations.loc[locations.sample_key.duplicated(False), 'sample_key'])
    locations = locations[~locations.sample_key.isin(ambiguous)].set_index('sample_key')
    methods = {}
    for _, row in metadata.iloc[6:].dropna(subset=[0, 1, 3, 5]).iterrows():
        field = {'87Sr/86Sr': 'SR87_SR86'}.get(str(row[1]).strip(), str(row[1]).strip().upper())
        methods[method_code(row[0]), field] = (row[3], row[4], row[5])
    anchor = pilot[scope].copy()
    anchor['sample_key'] = anchor.Sample_name.map(key)
    anchor = anchor[~anchor.sample_key.duplicated(False)].set_index('sample_key')
    target_cols = {i: str(v).strip().upper() for i, v in enumerate(data.iloc[1]) if str(v).strip().upper() in BOUNDS}
    records = []
    for i, row in data.iloc[6:].dropna(subset=[0]).iterrows():
        k = key(row[0]); material = str(row[2]).strip()
        loc = locations.loc[k] if k in locations.index else None
        target = anchor.loc[k] if k in anchor.index else None
        distance = np.nan if loc is None or target is None else coordinate_distance(loc.latitude, loc.longitude, target.Latitude, target.Longitude)
        location = '' if loc is None else str(loc.source_location)
        flag = ''
        if target is not None and name == 'SO255':
            if ('volcanic front' in location.lower() and target.Secondary_label == 'BAB') or ('rear arc' in location.lower() and target.Secondary_label == 'IOA'):
                flag = 'REVIEW_LOCATION_LABEL'
        for col, field in target_cols.items():
            value = pd.to_numeric(row[col], errors='coerce')
            if not np.isfinite(value):
                continue
            code = method_code(data.iloc[2, col]); method, instrument, lab = methods.get((code, field), ('', '', ''))
            uncertainty = row[col+1] if col+1 < len(row) and data.iloc[1, col+1] == '2SE' else np.nan
            low, high = BOUNDS[field]
            status = 'LINKED_METHOD_DOCUMENTED'
            if k in ambiguous or loc is None or target is None or not np.isfinite(distance) or distance > 1:
                status = 'REVIEW_IDENTITY'
            elif flag:
                status = flag
            elif not (low <= value <= high and material.lower() in ('whole rock', 'volcanic glass') and method and lab):
                status = 'REVIEW_ANALYTICAL'
            records.append({'dataset': name, 'doi': doi, 'sheet': '3 Data', 'excel_row': int(i)+1,
                            'sample_name': row[0], 'source_location': location, 'material': material,
                            'isotope': field, 'value': float(value), 'uncertainty_2SE': uncertainty,
                            'method': method, 'instrument': instrument, 'laboratory': lab,
                            'ratio_basis': 'as labeled in original workbook', 'distance_km': distance,
                            IDS: None if target is None else target[IDS],
                            'class': None if target is None else target.Secondary_label, 'status': status})
    return pd.DataFrame(records)


def source_pairs(records, pilot):
    ok = records.status.eq('LINKED_METHOD_DOCUMENTED')
    for (identity, field), part in records[ok].groupby([IDS, 'isotope']):
        previous = pilot.at[identity, field] if field in PANEL else np.nan
        published = pilot.at[identity, 'source_'+field]
        values = np.r_[part.value, previous, published]
        values = values[np.isfinite(values)]
        if np.ptp(values) > TOL.get(field, 2e-5):
            records.loc[part.index, 'status'] = 'REVIEW_RATIO_DISAGREEMENT'
    accepted = records[records.status.eq('LINKED_METHOD_DOCUMENTED')]
    wide = accepted.groupby([IDS, 'isotope']).value.median().unstack().add_prefix('workbook_')
    return accepted, wide


def broader_cohort(pilot):
    qa = pd.read_parquet(QA)
    physical = pd.read_parquet(PHYSICAL)
    quarantined = pd.read_csv(QUARANTINE, usecols=[IDS])
    assert physical[IDS].is_unique and qa[IDS].is_unique
    assert set(qa[IDS]).issubset(physical[IDS])
    extension = qa[qa.pool.eq('incomplete_extra') & qa.Sr_Nd_usable].merge(physical, on=[IDS, 'Secondary_label'], validate='one_to_one')
    assert len(extension) == int((qa.pool.eq('incomplete_extra') & qa.Sr_Nd_usable).sum())
    extension = extension[~extension[IDS].isin(pilot[IDS]) & ~extension[IDS].isin(quarantined[IDS]) & ~extension.component_label_conflict]
    source_cols = ['Database', 'Source_record_ID']
    source_collision = pd.MultiIndex.from_frame(extension[source_cols]).isin(pd.MultiIndex.from_frame(pilot[source_cols]))
    extension['same_source_id_as_current_D36'] = source_collision
    extension.loc[source_collision, [IDS, *source_cols, 'Sample_name', 'Secondary_label', 'Reference', 'Province']].to_csv(OUT / 'cross_cohort_source_identity_review.csv', index=False)
    extension = extension.assign(cohort='historical_non_D36_candidate')
    d36 = pilot[pilot.Sr_Nd_usable.eq(True).fillna(False)].assign(cohort='current_D36', same_source_id_as_current_D36=False)
    combined = pd.concat([d36, extension], ignore_index=True)
    assert combined[IDS].is_unique
    data = combined[ELEMENTS].to_numpy(float)
    output = []
    for width in (18, 22, 36):
        elements = ELEMENTS[:width]
        present = np.isfinite(data[:, :width]).all(1) & (data[:, :width] > 0).all(1)
        subset = combined.loc[present].copy()
        subset['same_chemical_vector'] = subset.duplicated(elements, keep=False)
        subset['same_isotope_pair'] = subset.duplicated(PAIR, keep=False)
        subset['review_identity_before_freeze'] = subset.same_chemical_vector | subset.same_isotope_pair | subset.same_source_id_as_current_D36
        clean = subset[~subset.review_identity_before_freeze]
        metadata = [IDS, 'Sample_name', 'Secondary_label', 'Database', 'Source_record_ID', 'Source_URL', 'Reference', 'Province', 'Location', 'Latitude', 'Longitude', 'Age', 'Label_evidence', 'Review_status', 'cohort', 'Sr_Nd_A', 'isotope_reference_guard_key', 'same_chemical_vector', 'same_isotope_pair', 'same_source_id_as_current_D36', 'review_identity_before_freeze']
        observation = [f'{field}{suffix}' for field in PANEL for suffix in ('', '_grade', '_references', '_methods')]
        subset[metadata + elements + observation].to_parquet(OUT / f'D{width}_SrNd_candidates.parquet', index=False)
        for (cohort, label), part in subset.groupby(['cohort','Secondary_label'], observed=True):
            good = clean[(clean.cohort == cohort) & (clean.Secondary_label == label)]
            output.append({'panel': f'D{width}', 'cohort': cohort, 'class': label, 'paired_n': len(part),
                           'A_pair_n': int(part.Sr_Nd_A.eq(True).sum()), 'province_count': part.Province.nunique(),
                           'reference_count': part.Reference.nunique(), 'n_after_duplicate_flags': len(good),
                           'provinces_after_flags': good.Province.nunique()})
    pd.DataFrame(output).to_csv(OUT / 'coverage_by_panel_class.csv', index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_parquet(PILOT)
    so = source_records(SO255, 'SO255', '10.26022/IEDA/111723', pilot.Reference.str.contains('111723', na=False), pilot)
    cir = source_records(CIR, 'Central Indian Ridge', '10.26022/IEDA/112112', pilot.Source_URL.str.contains('id=2112$', na=False), pilot)
    records = pd.concat([so, cir], ignore_index=True)
    accepted, wide = source_pairs(records, pilot.set_index(IDS))
    records.to_csv(OUT / 'primary_workbook_observations.csv', index=False, encoding='utf-8-sig')
    result = pilot.merge(wide, on=IDS, how='left', validate='one_to_one')
    result['workbook_Sr_Nd_pair'] = result[['workbook_'+f for f in PAIR]].notna().all(1)
    result['workbook_five_panel'] = result[['workbook_'+f for f in PANEL]].notna().all(1)
    result.to_parquet(OUT / 'D36_with_workbook_isotopes.parquet', index=False)
    broader_cohort(pilot)
    more_pairs = result.workbook_Sr_Nd_pair & ~result.Sr_Nd_usable.eq(True).fillna(False) & ~result.source_Sr_Nd_pair
    method = records.groupby(['dataset', 'status']).size().rename('n').reset_index()
    method.to_csv(OUT / 'workbook_screen_counts.csv', index=False)
    stats = {'workbook_observations_linked': len(accepted), 'workbook_physical_samples': int(accepted[IDS].nunique()),
             'workbook_sr_nd_pairs': int(result.workbook_Sr_Nd_pair.sum()),
             'workbook_five_panel': int(result.workbook_five_panel.sum()),
             'workbook_pairs_additional_to_previous': int(more_pairs.sum()),
             'workbook_additional_by_class': result.loc[more_pairs, 'Secondary_label'].value_counts().to_dict(),
             'workbook_hf_samples': int(result.workbook_HF176_HF177.notna().sum()),
             'linked_samples_by_workbook': accepted.groupby('dataset')[IDS].nunique().to_dict(),
             'inputs': {str(p.relative_to(ROOT)): digest(p) for p in (PILOT, QA, PHYSICAL, QUARANTINE, SO255, CIR)},
             'outputs': {p.name: digest(p) for p in OUT.iterdir() if p.suffix in ('.parquet', '.csv')}}
    (OUT / 'audit.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k not in ('inputs', 'outputs')}, indent=2))


if __name__ == '__main__':
    main()
