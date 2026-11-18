"""Link local source-reported isotopes to frozen D36 without changing either source."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / 'reports/modeling'
QUALITY = ROOT / 'reports/data_quality'
FROZEN = ROOT / 'data/frozen'
OUT = MODELING / 'd36_isotope_geologic_units_pilot_2026-11-13'
FIELDS = {'87Sr/86Sr': 'SR87_SR86', '143Nd/144Nd': 'ND143_ND144', '176Hf/177Hf': 'HF176_HF177',
          '206Pb/204Pb': 'PB206_PB204', '207Pb/204Pb': 'PB207_PB204', '208Pb/204Pb': 'PB208_PB204',
          'epsNd': 'epsNd', 'd53Cr': 'd53Cr'}
PANEL = ['SR87_SR86', 'ND143_ND144', 'PB206_PB204', 'PB207_PB204', 'PB208_PB204']
BOUNDS = {'SR87_SR86': (.69, .80), 'ND143_ND144': (.510, .514), 'HF176_HF177': (.280, .284),
          'PB206_PB204': (14, 30), 'PB207_PB204': (14, 18), 'PB208_PB204': (32, 50)}
TOL = dict(zip(PANEL, (1e-4, 2e-5, .2, .05, .2)))
IDS = 'Physical_sample_ID'


def read(path):
    inputs.append(path)
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def singleton(folder, pattern):
    return next(folder.glob(pattern))


def key(value):
    return re.sub(r'\s+', ' ', str(value).strip()).upper()


def harmonize_wide(raw, source, uncertainty=None):
    columns = [col for col in FIELDS if col in raw]
    data = raw.melt(id_vars=['Sample_ID'], value_vars=columns, var_name='source_field', value_name='value')
    data['isotope_field'] = data.source_field.map(FIELDS)
    data['source_sample_id'] = data.Sample_ID
    data['uncertainty'] = np.nan
    if uncertainty:
        data.loc[data.source_field == 'd53Cr', 'uncertainty'] = data.Sample_ID.map(raw.set_index('Sample_ID')[uncertainty])
    data['uncertainty_kind'] = np.where(data.uncertainty.notna(), '2SD', 'not available in QA table')
    data['source'] = source
    return data.drop(columns='Sample_ID')


def harmonize_long(raw, source, field_name, uncertainty, kind):
    data = raw.rename(columns={field_name: 'source_field', uncertainty: 'uncertainty'}).copy()
    radiogenic = data.isotope_field.str.match(r'^(Sr|Nd|Pb|Hf)\d+_')
    data.loc[radiogenic, 'isotope_field'] = data.loc[radiogenic, 'isotope_field'].str.upper()
    data['source'] = source
    data['uncertainty_kind'] = kind
    return data


def from_manifest(manifest, name='Sample_name'):
    pairs = manifest[[name, IDS]].drop_duplicates().copy()
    pairs[name] = pairs[name].map(key)
    assert pairs[name].is_unique
    return dict(zip(pairs[name], pairs[IDS]))


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = singleton(FROZEN, '*expansion13_2026-10-26/D36*.parquet')
    linked = read(singleton(MODELING, '*d36_expansion13*/*linked.parquet'))
    current = read(frozen)
    pd.testing.assert_frame_equal(linked[current.columns], current, check_dtype=False)
    assert current[IDS].is_unique and len(current) == 16459
    isotope_sources = {
        'Northeast Lau': 'https://doi.org/10.1594/PANGAEA.941306',
        'Kamchatka': 'https://doi.org/10.17632/c2kvn6kgph.1',
        'Northern Mariana': 'https://doi.org/10.17632/7xftfwrc42.1',
        'North Fiji': 'https://doi.org/10.17632/n3w3z7w55c.1',
    }
    lau = read(singleton(MODELING, '*isotope_quality_verification_pangaea*/*quality.csv'))
    kam = read(singleton(MODELING, '*isotope_quality_verification_kamchatka*/*quality.csv'))
    mar = read(singleton(MODELING, '*isotope_quality_verification_mariana*/*sidecar.csv'))
    fiji = read(singleton(MODELING, '*isotope_quality_verification_north_fiji*/*sidecar.csv'))
    mar_x = read(singleton(QUALITY, '*d36_mariana*/*crosswalk.csv'))
    assert mar_x.vector_exact_atol_1e10.all() and mar_x.name_match_n.eq(1).all()
    mar_new = current[current.Source_record_ID.str.startswith('LIT:MARIANA_NMTB:TableS1:', na=False)]
    mar_map = from_manifest(mar_x.rename(columns={'current_physical_sample_id': IDS}), 'source_sample')
    assert not set(mar_map) & set(from_manifest(mar_new))
    fiji_m = read(singleton(FROZEN, '*expansion03*/accepted_north_fiji_bab_manifest.csv'))
    fiji_a = read(singleton(QUALITY, '*north_fiji*duplicate*/full_audit.csv'))
    fiji_ok = fiji_a.loc[fiji_a.exact_sample_name_match & fiji_a.exact_vector_match & fiji_a.coordinate_match_within_1km, 'source_sample_id'].map(key)
    mappings = {
        'Northeast Lau': from_manifest(read(singleton(FROZEN, '*expansion11*/accepted_pangaea*'))),
        'Kamchatka': from_manifest(read(singleton(FROZEN, '*expansion12*/accepted_kamchatka*'))),
        'Northern Mariana': {**mar_map, **from_manifest(mar_new)},
        'North Fiji': {k: v for k, v in from_manifest(fiji_m).items() if k in set(fiji_ok)},
    }
    frames = [harmonize_wide(lau, 'Northeast Lau'), harmonize_wide(kam, 'Kamchatka', '2SD_d53Cr'),
              harmonize_long(mar, 'Northern Mariana', 'source_field', 'uncertainty', 'as reported in Table S1'),
              harmonize_long(fiji, 'North Fiji', 'source_row_label', 'uncertainty_2s', '2s as reported')]
    data = pd.concat(frames, ignore_index=True)
    data[IDS] = [mappings[s].get(key(k)) for s, k in zip(data.source, data.source_sample_id)]
    assert set(data[IDS].dropna()).issubset(current[IDS])
    data['source_doi'] = data.source.map(isotope_sources)
    data['value'] = pd.to_numeric(data.value, errors='raise')
    data['ratio_basis'] = 'source reported; measured versus initial not independently reviewed'
    data['status'] = np.select([data[IDS].isna(), ~np.isfinite(data.value)], ['UNLINKED', 'NOT_REPORTED'], default='LINKED_SOURCE_OBSERVED')
    for field, (low, high) in BOUNDS.items():
        outside = data.isotope_field.eq(field) & data.value.notna() & ~data.value.between(low, high)
        data.loc[outside & data[IDS].notna(), 'status'] = 'REVIEW_RANGE'
    approved = data[data.status.eq('LINKED_SOURCE_OBSERVED')].copy()
    assert not approved.duplicated([IDS, 'isotope_field']).any()
    lookup = linked.set_index(IDS)
    approved['legacy_value'] = [lookup.at[i, f] if f in PANEL else np.nan for i, f in zip(approved[IDS], approved.isotope_field)]
    approved['difference'] = (approved.value - approved.legacy_value).abs()
    approved['comparison_status'] = np.select([approved.legacy_value.isna(), approved.difference.gt(approved.isotope_field.map(TOL))],
                                              ['NO_LEGACY_VALUE', 'REVIEW_DISAGREEMENT'], default='WITHIN_LEGACY_TOLERANCE')
    data = data.merge(approved[[IDS, 'isotope_field', 'legacy_value', 'difference', 'comparison_status']],
                      on=[IDS, 'isotope_field'], how='left', validate='many_to_one')
    data = data.merge(current[[IDS, 'Secondary_label', 'Province']], on=IDS, how='left', validate='many_to_one')
    data.to_csv(OUT / 'source_isotope_observations.csv', index=False, encoding='utf-8-sig')
    source_values = approved.pivot(index=IDS, columns='isotope_field', values='value').add_prefix('source_')
    result = linked.merge(source_values, on=IDS, how='left', validate='one_to_one')
    result['source_observations'] = result[IDS].map(approved.groupby(IDS).size()).fillna(0).astype(int)
    result['source_Sr_Nd_pair'] = result[['source_SR87_SR86', 'source_ND143_ND144']].notna().all(axis=1)
    result['source_five_panel'] = result[['source_' + field for field in PANEL]].notna().all(axis=1)
    result['source_review_disagreement'] = result[IDS].isin(approved.loc[approved.comparison_status.eq('REVIEW_DISAGREEMENT'), IDS])
    result.to_parquet(OUT / 'D36_isotope_pilot.parquet', index=False)
    assert len(pd.read_parquet(OUT / 'D36_isotope_pilot.parquet')) == len(current)

    geo = []
    for source, system, rule in [('Northeast Lau', 'Lau Basin', 'accepted source manifest'),
                                 ('Northern Mariana', 'Izu-Bonin-Mariana system', 'source manifest or exact analysis crosswalk'),
                                 ('North Fiji', 'North Fiji Basin', 'named CSR segments and identity audit'),
                                 ('Kamchatka', 'Kamchatka Arc', 'accepted source manifest')]:
        geo.extend({IDS: i, 'named_system': system, 'source_subset': source, 'mapping_evidence': rule}
                   for i in mappings[source].values())
    geo = pd.DataFrame(geo).merge(current[[IDS, 'Sample_name', 'Reference', 'Province', 'Location', 'Secondary_label']],
                                 on=IDS, validate='one_to_one')
    geo.to_csv(OUT / 'named_geologic_units_seed.csv', index=False, encoding='utf-8-sig')
    support = result.assign(legacy_pair_A=result.Sr_Nd_A.eq(True).fillna(False),
                            legacy_pair_AB=result.Sr_Nd_usable.eq(True).fillna(False),
                            prior_qa_missing=result.isotope_qc_link_status.eq('NOT_IN_QA_INPUT'))
    support = support.groupby(['Secondary_label', 'Province', 'Reference', 'Database', 'Source_URL'], observed=True)
    support = support.agg(n=(IDS, 'size'), legacy_pair_A=('legacy_pair_A', 'sum'),
                          legacy_pair_AB=('legacy_pair_AB', 'sum'), source_pair=('source_Sr_Nd_pair', 'sum'),
                          prior_qa_missing=('prior_qa_missing', 'sum')).reset_index()
    support['n_without_legacy_A_pair'] = support.n - support.legacy_pair_A
    support.sort_values(['Secondary_label', 'n_without_legacy_A_pair', 'prior_qa_missing'],
                        ascending=[True, False, False]).to_csv(OUT / 'source_review_queue.csv', index=False, encoding='utf-8-sig')
    counts = []
    for label, part in result.groupby('Secondary_label', observed=True):
        counts.append({'class': label, 'd36_n': len(part), 'legacy_Sr_Nd_A': int(part.Sr_Nd_A.eq(True).sum()),
                       'legacy_Sr_Nd_AB': int(part.Sr_Nd_usable.eq(True).sum()), 'legacy_five_A': int(part.all_five_A.eq(True).sum()),
                       'legacy_five_AB': int(part.all_five_usable.eq(True).sum()), 'source_pairs': int(part.source_Sr_Nd_pair.sum()),
                       'source_pair_without_legacy_AB_pair': int((part.source_Sr_Nd_pair & ~part.Sr_Nd_usable.eq(True).fillna(False)).sum()),
                       'source_five_panel': int(part.source_five_panel.sum())})
    pd.DataFrame(counts).to_csv(OUT / 'coverage_by_class.csv', index=False)
    stats = {'d36_n': len(current), 'frozen_sha256': digest(frozen), 'newly_linked_observations': len(approved),
             'source_samples_linked_to_D36': int(data[IDS].nunique()), 'source_samples_with_observations': int(approved[IDS].nunique()),
             'source_sr_nd_pairs': int(result.source_Sr_Nd_pair.sum()), 'source_five_panel': int(result.source_five_panel.sum()),
             'source_pairs_without_legacy_usable_pair': int((result.source_Sr_Nd_pair & ~result.Sr_Nd_usable.eq(True).fillna(False)).sum()),
             'source_five_without_legacy_usable_panel': int((result.source_five_panel & ~result.all_five_usable.eq(True).fillna(False)).sum()),
             'old_QA_unreviewed_samples_with_linked_observations': int((result.source_observations.gt(0) & result.isotope_qc_link_status.eq('NOT_IN_QA_INPUT')).sum()),
             'overlapping_legacy_values': int(approved.legacy_value.notna().sum()),
             'legacy_disagreements': int(approved.comparison_status.eq('REVIEW_DISAGREEMENT').sum()),
             'named_system_seeds': geo.groupby('named_system').size().to_dict(),
             'source_observation_counts': approved.groupby('source').size().to_dict(),
             'inputs': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
             'outputs': {p.name: digest(p) for p in OUT.iterdir() if p.suffix in ('.csv', '.parquet')}}
    (OUT / 'audit.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k not in ('inputs', 'outputs')}, ensure_ascii=False, indent=2))


inputs = []


if __name__ == '__main__':
    main()
