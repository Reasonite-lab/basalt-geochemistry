"""Fixed-D36-test experiment: expand training with real incomplete analyses.

Chemical definitions are inserted by the notebook builder. No geography is a predictor.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, Booster
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from scipy.sparse import csr_matrix

CLASSES = ['BAB', 'CA', 'CFB', 'CR', 'IA', 'IOA', 'MOR', 'OI', 'OP']
ELEMENTS = ['SIO2(WT%)', 'TIO2(WT%)', 'AL2O3(WT%)', 'FEOT(WT%)', 'CAO(WT%)', 'MGO(WT%)', 'MNO(WT%)', 'K2O(WT%)', 'NA2O(WT%)', 'P2O5(WT%)', 'V(PPM)', 'CR(PPM)', 'NI(PPM)', 'RB(PPM)', 'SR(PPM)', 'Y(PPM)', 'ZR(PPM)', 'NB(PPM)', 'LA(PPM)', 'CE(PPM)', 'ND(PPM)', 'TH(PPM)', 'BA(PPM)', 'HF(PPM)', 'TA(PPM)', 'PR(PPM)', 'SM(PPM)', 'EU(PPM)', 'GD(PPM)', 'TB(PPM)', 'DY(PPM)', 'HO(PPM)', 'ER(PPM)', 'TM(PPM)', 'YB(PPM)', 'LU(PPM)']
SYMBOLS = ['Si', 'Ti', 'Al', 'Fe', 'Ca', 'Mg', 'Mn', 'K', 'Na', 'P', 'V', 'Cr', 'Ni', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'La', 'Ce', 'Nd', 'Th', 'Ba', 'Hf', 'Ta', 'Pr', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
PROXIES = [{'group': 0, 'name': 'Ba_Nb', 'kind': 'ratio', 'elements': ['BA(PPM)', 'NB(PPM)']}, {'group': 0, 'name': 'Ba_Th', 'kind': 'ratio', 'elements': ['BA(PPM)', 'TH(PPM)']}, {'group': 0, 'name': 'Rb_Nb', 'kind': 'ratio', 'elements': ['RB(PPM)', 'NB(PPM)']}, {'group': 0, 'name': 'Sr_Nd', 'kind': 'ratio', 'elements': ['SR(PPM)', 'ND(PPM)']}, {'group': 0, 'name': 'Ba_La', 'kind': 'ratio', 'elements': ['BA(PPM)', 'LA(PPM)']}, {'group': 0, 'name': 'K2O_Na2O', 'kind': 'ratio', 'elements': ['K2O(WT%)', 'NA2O(WT%)']}, {'group': 1, 'name': 'Th_Nb', 'kind': 'ratio', 'elements': ['TH(PPM)', 'NB(PPM)']}, {'group': 1, 'name': 'La_Nb', 'kind': 'ratio', 'elements': ['LA(PPM)', 'NB(PPM)']}, {'group': 1, 'name': 'Th_Yb', 'kind': 'ratio', 'elements': ['TH(PPM)', 'YB(PPM)']}, {'group': 1, 'name': 'Th_Ta', 'kind': 'ratio', 'elements': ['TH(PPM)', 'TA(PPM)']}, {'group': 1, 'name': 'La_Ta', 'kind': 'ratio', 'elements': ['LA(PPM)', 'TA(PPM)']}, {'group': 2, 'name': 'Nb_Yb', 'kind': 'ratio', 'elements': ['NB(PPM)', 'YB(PPM)']}, {'group': 2, 'name': 'La_Sm', 'kind': 'ratio', 'elements': ['LA(PPM)', 'SM(PPM)']}, {'group': 2, 'name': 'Zr_Nb', 'kind': 'ratio', 'elements': ['ZR(PPM)', 'NB(PPM)']}, {'group': 2, 'name': 'Nb_Ta', 'kind': 'ratio', 'elements': ['NB(PPM)', 'TA(PPM)']}, {'group': 2, 'name': 'Nb_Y', 'kind': 'ratio', 'elements': ['NB(PPM)', 'Y(PPM)']}, {'group': 2, 'name': 'Zr_Y', 'kind': 'ratio', 'elements': ['ZR(PPM)', 'Y(PPM)']}, {'group': 3, 'name': 'La_Yb', 'kind': 'ratio', 'elements': ['LA(PPM)', 'YB(PPM)']}, {'group': 3, 'name': 'Sm_Yb', 'kind': 'ratio', 'elements': ['SM(PPM)', 'YB(PPM)']}, {'group': 3, 'name': 'Dy_Yb', 'kind': 'ratio', 'elements': ['DY(PPM)', 'YB(PPM)']}, {'group': 3, 'name': 'Gd_Yb', 'kind': 'ratio', 'elements': ['GD(PPM)', 'YB(PPM)']}, {'group': 3, 'name': 'Ce_Yb', 'kind': 'ratio', 'elements': ['CE(PPM)', 'YB(PPM)']}, {'group': 4, 'name': 'FeOT_MgO', 'kind': 'ratio', 'elements': ['FEOT(WT%)', 'MGO(WT%)']}, {'group': 4, 'name': 'CaO_Al2O3', 'kind': 'ratio', 'elements': ['CAO(WT%)', 'AL2O3(WT%)']}, {'group': 4, 'name': 'Ni_MgO_ppm', 'kind': 'trace_oxide_ratio', 'elements': ['NI(PPM)', 'MGO(WT%)']}, {'group': 4, 'name': 'Cr_MgO_ppm', 'kind': 'trace_oxide_ratio', 'elements': ['CR(PPM)', 'MGO(WT%)']}, {'group': 4, 'name': 'Mg_number_total_Fe_proxy', 'kind': 'mg_number', 'elements': ['MGO(WT%)', 'FEOT(WT%)']}, {'group': 4, 'name': 'total_alkalies', 'kind': 'alkalies', 'elements': ['K2O(WT%)', 'NA2O(WT%)']}, {'group': 5, 'name': 'Eu_anomaly_chondrite', 'kind': 'eu_anomaly', 'elements': ['EU(PPM)', 'SM(PPM)', 'GD(PPM)']}, {'group': 5, 'name': 'Ce_anomaly_chondrite', 'kind': 'ce_anomaly', 'elements': ['CE(PPM)', 'LA(PPM)', 'PR(PPM)']}, {'group': 5, 'name': 'Ti_element_ppm_V', 'kind': 'ti_v', 'elements': ['TIO2(WT%)', 'V(PPM)']}, {'group': 5, 'name': 'P2O5_TiO2', 'kind': 'ratio', 'elements': ['P2O5(WT%)', 'TIO2(WT%)']}, {'group': 5, 'name': 'Pr_Sm', 'kind': 'ratio', 'elements': ['PR(PPM)', 'SM(PPM)']}, {'group': 5, 'name': 'Sr_Eu', 'kind': 'ratio', 'elements': ['SR(PPM)', 'EU(PPM)']}]
CHONDRITE = {'La': 0.237, 'Ce': 0.613, 'Pr': 0.0928, 'Sm': 0.148, 'Eu': 0.0563, 'Gd': 0.199}
POS = {e:i for i,e in enumerate(ELEMENTS)}


SPLITS = ['province_10fold', 'reference_10fold', 'random_10fold']
MODELS = ['complete_D36', 'complete_D22', 'complete_D18',
          'expanded_native', 'expanded_duplicate_control', 'expanded_masked']
PANELS = {'D36': list(range(36)), 'D22': list(range(22)), 'D18': list(range(18))}
FEATURE_NAMES = ['log_' + s for s in SYMBOLS] + ['clr_' + s for s in SYMBOLS[:10]] + [p['name'] for p in PROXIES]
OUT_REL = 'reports/modeling/incomplete_pool_fixed_test_2026-10-01'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def resolve_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p/'data/frozen/experiment_v5_2026-07-30/D36_v9.parquet').exists(): return p
    raise FileNotFoundError('请在 D:/AI地学 或 notebooks 目录运行。')

def clean_raw(raw):
    x = np.asarray(raw, dtype=np.float64).copy()
    x[~np.isfinite(x) | (x <= 0)] = np.nan
    return x

def features(raw):
    """Only compute a derived feature when all its measured constituents exist."""
    raw = clean_raw(raw)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.log10(raw)
        clr = z[:, :10] - z[:, :10].mean(1, keepdims=True)
        out = []
        for p in PROXIES:
            ix = [POS[e] for e in p['elements']]
            k = p['kind']
            if k == 'ratio': v = z[:, ix[0]] - z[:, ix[1]]
            elif k == 'trace_oxide_ratio': v = z[:, ix[0]] - z[:, ix[1]] - 4.
            elif k == 'mg_number':
                mg, fe = raw[:, ix[0]] / 40.304, raw[:, ix[1]] / 71.844
                v = 100 * mg / (mg + fe)
            elif k == 'alkalies': v = np.log10(raw[:, ix[0]] + raw[:, ix[1]])
            elif k == 'ti_v': v = z[:, ix[0]] + math.log10(10000 * 47.867 / 79.865) - z[:, ix[1]]
            elif k in ['eu_anomaly', 'ce_anomaly']:
                ss = ['Eu','Sm','Gd'] if k == 'eu_anomaly' else ['Ce','La','Pr']
                zz = z[:, ix] - np.log10([CHONDRITE[s] for s in ss])
                v = zz[:,0] - .5 * (zz[:,1] + zz[:,2])
            else: raise ValueError(k)
            out.append(v)
        x = np.column_stack([z, clr, *out]).astype(np.float32)
    assert x.shape[1] == 80 and not np.isinf(x).any()
    return x

def panel_raw(raw, panel):
    x = np.full_like(raw, np.nan, dtype=float)
    x[:, PANELS[panel]] = raw[:, PANELS[panel]]
    return x

class UnionFind:
    def __init__(self, n): self.p = np.arange(n, dtype=np.int64)
    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = int(self.p[i])
        return int(i)
    def union(self, a, b):
        a,b = self.find(int(a)),self.find(int(b))
        if a != b: self.p[max(a,b)] = min(a,b)
    def roots(self): return np.array([self.find(i) for i in range(len(self.p))])

def unite_codes(uf, codes):
    seen = {}
    for i, c in enumerate(codes):
        if c in seen: uf.union(i, seen[c])
        else: seen[c] = i

def reference_tokens(value):
    value = str(value).strip()
    tokens = ['georoc_ref:' + v for v in re.findall(r'\[(\d+)\]', value)]
    tokens += ['doi:' + v.lower().rstrip('.,;)') for v in re.findall(r'10\.\d{4,9}/[^\s\]<>]+', value)]
    # Full reference is retained as an additional conservative alias.
    if value and value.lower() not in ['nan','unknown','none']: tokens += ['reference:' + re.sub(r'\s+',' ',value).lower()]
    return set(tokens)

def load_folds(root, anchor):
    fp = root/'data/frozen/experiment_v5_2026-07-30/D36_v9_split_manifest.csv.gz'
    ten = root/'reports/modeling/ten_fold_all_validation_2026-09-11/oof_predictions.csv'
    prov = root/'reports/modeling/privileged_distillation_nested_panels_2026-09-15/outer_fold_manifest.csv.gz'
    frozen = pd.read_csv(fp).set_index('Physical_sample_ID').loc[anchor.Physical_sample_ID].reset_index()
    pred = pd.read_csv(ten, usecols=['split','fold','row_index','y_true'])
    folds = {}
    for s in ['random_10fold','reference_10fold']:
        b = pred.loc[pred.split.eq(s)].drop_duplicates().set_index('row_index')
        assert b.index.is_unique and len(b)==len(anchor)
        b = b.reindex(np.arange(len(anchor)))
        assert np.array_equal(b.y_true, anchor.Secondary_label)
        folds[s] = b.fold.to_numpy(int)
    p = pd.read_csv(prov).set_index('Physical_sample_ID').loc[anchor.Physical_sample_ID]
    folds['province_10fold'] = p.province_10fold.to_numpy(int)
    for s in SPLITS:
        assert set(folds[s]) == set(range(10)), s
        t = pd.DataFrame({'fold':folds[s], 'group':frozen[s.replace('_10fold','_component')]})
        assert t.groupby('group').fold.nunique().max()==1
    return folds, frozen, [fp,ten,prov]

def prepare(root, out):
    """Persist data-only leakage groups before fitting any model."""
    out.mkdir(parents=True, exist_ok=True)
    anchor_path = root/'data/frozen/experiment_v5_2026-07-30/D36_v9.parquet'
    pool_path = root/'data/exports/all_classes_physical_sample_element_table_prospective_v9_2026-07-30.parquet'
    cat_path = root/'data/interim/reconstruction_candidate_catalog_prospective_v9_2026-07-30.parquet'
    anchor = pd.read_parquet(anchor_path).reset_index(drop=True)
    pool = pd.read_parquet(pool_path)
    assert anchor.Physical_sample_ID.is_unique and pool.Physical_sample_ID.is_unique
    raw_all = clean_raw(pool[ELEMENTS].to_numpy(float))
    obs = np.isfinite(raw_all)
    qc = (pool.training_eligible_provisional.fillna(False) & ~pool.component_label_conflict.fillna(True)
          & pool.Confidence.isin(['A','B']) & pool.Secondary_label.isin(CLASSES)
          & obs[:,:10].all(1) & pd.Series(raw_all[:,:10].sum(1),index=pool.index).between(90,105)
          & (obs.sum(1)>=18))
    pool = pool.loc[qc].reset_index(drop=True)
    lookup = pool.set_index('Physical_sample_ID')
    same = lookup.loc[anchor.Physical_sample_ID]
    assert np.array_equal(same.Secondary_label,anchor.Secondary_label)
    assert np.array_equal(same[ELEMENTS].to_numpy(float),anchor[ELEMENTS].to_numpy(float))
    extra = pool.loc[~pool.Physical_sample_ID.isin(anchor.Physical_sample_ID)].copy()
    # Anchor rows always stay first, in the exact original order.
    pool = pd.concat([anchor,extra],ignore_index=True)
    raw = clean_raw(pool[ELEMENTS].to_numpy(float)); obs = np.isfinite(raw)
    n0,n = len(anchor),len(pool)
    assert n0==16383 and obs[:n0].all() and not obs[n0:].all(1).any()
    folds,frozen,fold_paths = load_folds(root,anchor)
    paths = [anchor_path,pool_path,cat_path,*fold_paths]
    source_hashes = {str(p.relative_to(root)):sha(p) for p in paths}
    signature = hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    cache_meta = out/'data_audit.json'
    cache_np = out/'prepared_indices.npz'
    if cache_meta.exists() and cache_np.exists():
        meta = json.loads(cache_meta.read_text(encoding='utf-8'))
        assert meta['source_signature']==signature
        d = np.load(cache_np)
        groups = {s:d[s] for s in [*SPLITS,'chemical_guard']}
        print('PREPARE RESUME',n,'samples',flush=True)
        return pool,raw,folds,groups,meta
    print('PREPARE',n,'eligible samples; checking observed-vector matches...',flush=True)
    uf = UnionFind(n)
    complete_hash = pd.util.hash_pandas_object(pd.DataFrame(raw[:n0]),index=False).to_numpy()
    unite_codes(uf, list(complete_hash) + ['partial:'+str(i) for i in range(n0,n)])
    patterns,inverse = np.unique(np.packbits(obs[n0:],axis=1),axis=0,return_inverse=True)
    partial_anchor_edges = []
    for k, packed in enumerate(patterns):
        inds = np.flatnonzero(inverse==k)+n0
        ix = np.flatnonzero(np.unpackbits(packed)[:36])
        projected = raw[:n0,ix]
        h0 = pd.util.hash_pandas_object(pd.DataFrame(projected),index=False).to_numpy()
        h1 = pd.util.hash_pandas_object(pd.DataFrame(raw[inds][:,ix]),index=False).to_numpy()
        buckets = {}
        for j,h in enumerate(h0): buckets.setdefault(int(h),[]).append(j)
        local = {}
        for row,h in zip(inds,h1):
            hh=int(h)
            if hh in local and np.array_equal(raw[row,ix],raw[local[hh],ix]): uf.union(row,local[hh])
            else: local[hh]=row
            for a in buckets.get(hh,[]):
                if np.array_equal(raw[row,ix],raw[a,ix]):
                    uf.union(row,a); partial_anchor_edges.append((int(row),int(a)))
        if (k+1)%300==0: print('PREPARE missing patterns',k+1,'/',len(patterns),flush=True)
    # Expand metadata through every source record in each physical component.
    catalog = pd.read_parquet(cat_path,columns=['Sample_ID','Reference','Province']).set_index('Sample_ID')
    member_rows=[]
    for i,value in enumerate(pool.component_member_ids):
        member_rows.extend((i,v) for v in str(value).split(' || ') if v)
    members=pd.DataFrame(member_rows,columns=['row','member'])
    assert members.member.is_unique and members.member.isin(catalog.index).all()
    members=members.join(catalog,on='member',validate='m:1')
    roots = uf.roots()
    groups={'random_10fold':roots.copy(),'chemical_guard':roots.copy()}
    for s,col in [('reference_10fold','Reference'),('province_10fold','Province')]:
        # A chemical copy guards sample leakage, but is not evidence that
        # two entire geographic provinces/papers constitute one geological unit.
        g=UnionFind(n)
        for _,part in frozen.groupby(s.replace('_10fold','_component'),sort=False):
            ii=part.index.to_numpy();
            for j in ii[1:]:g.union(ii[0],j)
        seen={}
        for row, value in zip(members.row,members[col]):
            if col=='Reference': tokens=reference_tokens(value)
            else:
                v=re.sub(r'\s+',' ',str(value).strip()).lower()
                tokens={v} if v and v not in ['unknown','nan','none'] else set()
            for token in tokens:
                if token in seen:g.union(int(row),seen[token])
                else:seen[token]=int(row)
        groups[s]=g.roots()
    plan=[]
    for s in SPLITS:
        for f in range(10):
            te=np.flatnonzero(folds[s]==f)
            keep=(~np.isin(groups[s],np.unique(groups[s][te]))
                  & ~np.isin(groups['chemical_guard'],np.unique(groups['chemical_guard'][te])))
            b=np.flatnonzero(keep[:n0]);e=np.flatnonzero(keep[n0:])+n0
            assert len(np.unique(pool.Secondary_label.iloc[b]))==9,(s,f,'baseline missing class')
            assert set(groups[s][b]).isdisjoint(groups[s][te])
            assert set(groups[s][e]).isdisjoint(groups[s][te])
            assert set(groups['chemical_guard'][np.concatenate([b,e])]).isdisjoint(groups['chemical_guard'][te])
            plan.append({'split':s,'fold':f,'n_test':len(te),'baseline_train_n':len(b),
                         'extra_train_n':len(e),'expanded_train_n':len(b)+len(e),
                         'baseline_purged_n':int((folds[s]!=f).sum()-len(b)),
                         'extra_purged_n':len(extra)-len(e),
                         'new_provinces_in_training':int(len(set(pool.Province.iloc[e])-set(pool.Province.iloc[b]))),
                         'new_references_in_training':int(len(set(pool.Reference.iloc[e])-set(pool.Reference.iloc[b])))})
    pd.DataFrame(plan).to_csv(out/'fold_training_pool_audit.csv',index=False,encoding='utf-8-sig')
    counts = pool.assign(pool=np.where(np.arange(n)<n0,'D36_anchor','incomplete_extra'),observed_elements=obs.sum(1))
    counts.groupby(['pool','Secondary_label']).agg(n=('Physical_sample_ID','size'),provinces=('Province','nunique'),references=('Reference','nunique'),mean_observed=('observed_elements','mean')).reset_index().to_csv(out/'pool_class_coverage.csv',index=False,encoding='utf-8-sig')
    counts[['Physical_sample_ID','Secondary_label','Confidence','Province','Reference','pool','observed_elements']].to_csv(out/'eligible_pool_manifest.csv.gz',index=False,compression='gzip')
    pd.DataFrame(partial_anchor_edges,columns=['partial_row','anchor_row']).to_csv(out/'exact_observed_match_edges.csv.gz',index=False,compression='gzip')
    np.savez_compressed(cache_np,**groups)
    meta={'status':'PREPARATION_PASSED','anchor_n':n0,'eligible_pool_n':n,'incomplete_extra_n':len(extra),
          'minimum_observed_elements':18,'major_total_range':[90,105],'missing_patterns':len(patterns),
          'exact_observed_partial_anchor_edges':len(partial_anchor_edges),
          'groups':{s:int(len(np.unique(groups[s]))) for s in SPLITS},
          'source_signature':signature,'source_hashes':source_hashes,
          'test_policy':'Original D36 10-fold test identities fixed; purge held-out metadata groups AND matching observed-chemistry sample components separately.',
          'metadata_are_predictors':False,'chemistry_match_policy':'Exact equality on all >=18 observed measurements is a conservative leakage group, not a claim of physical identity.',
          'unresolved_reference_aliases':'Only available source references, GEOROC bracket IDs, DOI strings, and frozen components can be linked.',
          'confidence_policy':'A/B labels retained as in D36; B includes province-supported labels and is not equivalent to independent sample-level confirmation.'}
    dump(meta,cache_meta)
    return pool,raw,folds,groups,meta

def score(y,p):
    assert p.shape==(len(y),9) and np.isfinite(p).all() and (p>=0).all()
    assert np.allclose(p.sum(1),1,atol=1e-5)
    h=p.argmax(1)
    pr,re,f1,sp=precision_recall_fscore_support(y,h,labels=np.arange(9),zero_division=0)
    return {'accuracy':float((h==y).mean()),'macro_f1':float(f1.mean()),
            'balanced_accuracy_supported':float(re[sp>0].mean()),'BAB_f1':float(f1[0]) if sp[0] else None,
            'n_test':len(y),'n_present_classes':int((sp>0).sum()),
            'log_loss':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-7,1)).mean())}

def cm_stats(cm):
    d=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-2)+cm.sum(-1)
    f=np.divide(2*d,den,out=np.zeros_like(d,dtype=float),where=den>0)
    return np.stack([d.sum(-1)/cm.sum((-2,-1)),f.mean(-1)],-1)

def bootstrap_pair(base,cand,repeats,seed):
    keys=['Physical_sample_ID','fold']
    d=base[keys+['true','pred','bootstrap_group']].merge(cand[keys+['true','pred']],on=keys,validate='1:1',suffixes=('_b','_c'))
    assert len(d)==len(base)==len(cand) and (d.true_b==d.true_c).all()
    g,uni=pd.factorize(d.bootstrap_group);yi=d.true_b.map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
    gb=np.zeros((len(uni),81),float);gc=np.zeros_like(gb)
    hb=d.pred_b.map({c:i for i,c in enumerate(CLASSES)}).to_numpy();hc=d.pred_c.map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
    np.add.at(gb,(g,yi*9+hb),1);np.add.at(gc,(g,yi*9+hc),1)
    rng=np.random.default_rng(seed);delta=[]
    sb,sc=csr_matrix(gb),csr_matrix(gc)
    for k in range(0,repeats,50):
        ix=rng.integers(0,len(uni),size=(min(50,repeats-k),len(uni)))
        weights=np.stack([np.bincount(v,minlength=len(uni)) for v in ix]).astype(float)
        cb=np.asarray(sb.T@weights.T).T.reshape(-1,9,9)
        cc=np.asarray(sc.T@weights.T).T.reshape(-1,9,9)
        delta.append(cm_stats(cc)-cm_stats(cb))
    dd=cm_stats(gc.sum(0).reshape(9,9))-cm_stats(gb.sum(0).reshape(9,9))
    lo,hi=np.quantile(np.concatenate(delta),[.025,.975],axis=0)
    return {'n_groups':len(uni),'delta_accuracy':float(dd[0]),'delta_macro_f1':float(dd[1]),
            'accuracy_ci_low':float(lo[0]),'accuracy_ci_high':float(hi[0]),
            'macro_f1_ci_low':float(lo[1]),'macro_f1_ci_high':float(hi[1])}

def summarize(run,cfg,groups):
    all_metrics=[];blocks=[]
    for s in cfg['splits']:
        for f in cfg['folds']:
            for name in MODELS:
                folder=run/s/f'fold_{f}'/name
                mm=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
                all_metrics.extend(mm)
                for path in folder.glob('predictions_*.csv.gz'):
                    d=pd.read_csv(path);d['split']=s;d['model']=name;d['panel']=path.name.split('_')[1].split('.')[0]
                    blocks.append(d)
    per_fold=pd.DataFrame(all_metrics);per_fold.to_csv(run/'per_fold_metrics.csv',index=False,encoding='utf-8-sig')
    oof=pd.concat(blocks,ignore_index=True);pooled=[];per_class=[];contrasts=[]
    for (s,name,panel),b in oof.groupby(['split','model','panel'],sort=False):
        b=b.sort_values(['fold','anchor_row'])
        if cfg['folds']==list(range(10)): assert len(b)==16383 and b.Physical_sample_ID.is_unique
        yy=b.true.map({c:i for i,c in enumerate(CLASSES)}).to_numpy();p=b[[f'p_{c}' for c in CLASSES]].to_numpy()
        pooled.append({'split':s,'model':name,'panel':panel,**score(yy,p)})
        pr,re,f1,sp=precision_recall_fscore_support(yy,p.argmax(1),labels=np.arange(9),zero_division=0)
        per_class.extend({'split':s,'model':name,'panel':panel,'class':c,'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'support':int(sp[i])} for i,c in enumerate(CLASSES))
    pooled=pd.DataFrame(pooled);pooled.to_csv(run/'pooled_oof_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(per_class).to_csv(run/'per_class_oof_metrics.csv',index=False,encoding='utf-8-sig')
    per_fold.groupby(['split','model','panel'])[['accuracy','macro_f1','BAB_f1']].agg(['mean','std']).to_csv(run/'fold_mean_std_metrics.csv',encoding='utf-8-sig')
    if cfg['bootstrap_repeats']:
        for s in cfg['splits']:
            for panel in PANELS:
                baseline='complete_'+panel
                pairs=[(baseline,'expanded_native'),(baseline,'expanded_masked'),
                       ('expanded_native','expanded_duplicate_control'),('expanded_duplicate_control','expanded_masked')]
                for j,(b,c) in enumerate(pairs):
                    a=oof.loc[oof.split.eq(s)&oof.model.eq(b)&oof.panel.eq(panel)].sort_values(['fold','anchor_row'])
                    d=oof.loc[oof.split.eq(s)&oof.model.eq(c)&oof.panel.eq(panel)].sort_values(['fold','anchor_row'])
                    contrasts.append({'split':s,'panel':panel,'baseline':b,'candidate':c,**bootstrap_pair(a,d,cfg['bootstrap_repeats'],cfg['seed']+j)})
        pd.DataFrame(contrasts).to_csv(run/'paired_group_bootstrap.csv',index=False,encoding='utf-8-sig')
    status='COMPLETED_FULL_10FOLD' if cfg['folds']==list(range(10)) and cfg['splits']==SPLITS else 'COMPLETED_SUBSET'
    dump({'status':status,'fitted_jobs':len(cfg['splits'])*len(cfg['folds'])*len(MODELS),
          'prediction_conditions':len(per_fold),'pooled_metrics':pooled.to_dict('records')},run/'summary.json')
    return pooled

def run_experiment(root,implementation_sha,cfg=None):
    root=Path(root)
    cfg=cfg or {'splits':SPLITS,'folds':list(range(10)),'seed':20261001,
                'tree_estimators':600,'bootstrap_repeats':1000,'augmentation_weight':.5,
                'protocol_version':'partial_pool_v1_20261001'}
    prepared=root/OUT_REL/'prepared_v2'
    pool,raw,folds,groups,data_audit=prepare(root,prepared)
    versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','lightgbm','scipy']}
    signature={'config':cfg,'implementation_sha':implementation_sha,'data_signature':data_audit['source_signature'],
               'package_versions':versions}
    code=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:12]
    run=root/OUT_REL/f'run_{code}';run.mkdir(parents=True,exist_ok=True)
    dump(signature,run/'run_config.json')
    n0=16383; yy=pool.Secondary_label.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
    xx=features(raw);test_features={p:features(panel_raw(raw[:n0],p)) for p in PANELS}
    total=len(cfg['splits'])*len(cfg['folds'])*len(MODELS);done=0;t0=time.time()
    print('OUTPUT',run,flush=True)
    for s in cfg['splits']:
        for f in cfg['folds']:
            te=np.flatnonzero(folds[s]==f)
            keep=(~np.isin(groups[s],np.unique(groups[s][te]))
                  & ~np.isin(groups['chemical_guard'],np.unique(groups['chemical_guard'][te])))
            base=np.flatnonzero(keep[:n0]);extra=np.flatnonzero(keep[n0:])+n0
            tr=np.concatenate([base,extra]);assert set(tr).isdisjoint(te)
            assert set(groups[s][tr]).isdisjoint(groups[s][te])
            assert set(groups['chemical_guard'][tr]).isdisjoint(groups['chemical_guard'][te])
            seed=cfg['seed']+f+100*SPLITS.index(s)
            rng=np.random.default_rng(seed)
            # Only outer-training incomplete patterns are eligible donors.
            donor=rng.choice(extra,size=len(base),replace=True)
            masked=raw[base].copy();masked[~np.isfinite(raw[donor])]=np.nan
            xm=features(masked)
            for name in MODELS:
                folder=run/s/f'fold_{f}'/name;folder.mkdir(parents=True,exist_ok=True)
                mp=folder/'metrics.json'
                if mp.exists():
                    meta=json.loads((folder/'training_audit.json').read_text(encoding='utf-8'))
                    assert meta['run_hash']==code
                    done+=1;print('RESUME',s,f,name,done,'/',total,flush=True);continue
                print('START',s,f,name,'base',len(base),'extra',len(extra),flush=True);ts=time.time()
                panels=list(PANELS) if name.startswith('expanded_') else [name.split('_')[1]]
                if name.startswith('complete_'):
                    panel=panels[0];xtrain=test_features[panel][base];train_rows=base
                    yt=yy[base];sw=np.ones(len(base))
                else:
                    xtrain=xx[tr];train_rows=tr;yt=yy[tr];sw=np.ones(len(tr))
                    if name in ['expanded_duplicate_control','expanded_masked']:
                        xadd=xx[base] if name=='expanded_duplicate_control' else xm
                        xtrain=np.vstack([xtrain,xadd]);yt=np.concatenate([yt,yy[base]])
                        sw=np.concatenate([sw,np.full(len(base),cfg['augmentation_weight'])])
                model=LGBMClassifier(n_estimators=cfg['tree_estimators'],learning_rate=.05,num_leaves=31,
                                     colsample_bytree=.8,subsample=.8,subsample_freq=1,class_weight='balanced',
                                     n_jobs=8,random_state=seed,verbosity=-1)
                model.fit(pd.DataFrame(xtrain,columns=FEATURE_NAMES),yt,sample_weight=sw)
                assert np.array_equal(model.classes_,np.arange(9))
                (folder/'model.txt').write_text(model.booster_.model_to_string(),encoding='utf-8')
                np.savez_compressed(folder/'training_indices.npz',train_rows=train_rows,test_rows=te,
                                    complete_base=base,extras=extra,pattern_donor=donor if name=='expanded_masked' else np.array([],int),
                                    sample_weight=sw)
                mm=[]
                for panel in panels:
                    p=model.predict_proba(pd.DataFrame(test_features[panel][te],columns=FEATURE_NAMES)).astype(np.float32)
                    pp=pd.DataFrame({'Physical_sample_ID':pool.Physical_sample_ID.iloc[te].to_numpy(),
                                     'anchor_row':te,'fold':f,'true':pool.Secondary_label.iloc[te].to_numpy(),
                                     'pred':np.asarray(CLASSES)[p.argmax(1)],'bootstrap_group':groups[s][te]})
                    for j,c in enumerate(CLASSES):pp[f'p_{c}']=p[:,j]
                    pp.to_csv(folder/f'predictions_{panel}.csv.gz',index=False,compression='gzip')
                    mm.append({'split':s,'fold':f,'model':name,'panel':panel,**score(yy[te],p)})
                meta={'run_hash':code,'seed':seed,'physical_train_n':len(train_rows),'fitted_rows_n':len(yt),
                      'base_n':len(base),'extra_n':len(extra) if name.startswith('expanded_') else 0,
                      'augmentation_n':len(base) if name in ['expanded_duplicate_control','expanded_masked'] else 0,
                      'feature_columns':FEATURE_NAMES,'nan_fraction':float(np.isnan(xtrain).mean()),
                      'train_class_counts':np.bincount(yt,minlength=9).tolist(),'elapsed_seconds':time.time()-ts,
                      'training_group_overlap':0,'test_label_used_for_selection':False,'reference_or_geography_as_predictor':False}
                dump(meta,folder/'training_audit.json');dump(mm,mp)
                done+=1;dump({'status':'RUNNING','completed_jobs':done,'total_jobs':total,'elapsed_seconds':time.time()-t0},run/'status.json')
                print('DONE',s,f,name,round(time.time()-ts,1),'s',done,'/',total,flush=True)
                del model,xtrain
    pooled=summarize(run,cfg,groups)
    dump({'status':'COMPLETED','completed_jobs':done,'total_jobs':total,'elapsed_seconds':time.time()-t0},run/'status.json')
    return run,pooled

if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--prepare-only',action='store_true');args.add_argument('--smoke',action='store_true');a=args.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.prepare_only: prepare(root,root/OUT_REL/'prepared_v2')
    else:
        cfg=None
        if a.smoke:cfg={'splits':[SPLITS[0]],'folds':[0],'seed':20261001,'tree_estimators':20,'bootstrap_repeats':0,'augmentation_weight':.5,'protocol_version':'smoke_partial_pool_v1_20261001'}
        run,pooled=run_experiment(root,sha(__file__),cfg)
        print(pooled.to_string(index=False),flush=True)
