"""Audit D36 identity conflicts, freeze a clean cohort, and rerun D36_74.

The original frozen dataset is never modified. Ambiguous records are quarantined
using provenance and identity evidence only; chemistry is never used to choose a
tectonic label or location. The clean experiment fits the fixed LightGBM baseline
and the previously tested five-class conditional local expert under random,
Reference, and Province five-fold validation with three seeds.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data/frozen/experiment_v5_2026-07-30/D36_v9.parquet"
SOURCE_MANIFEST = ROOT / "data/frozen/experiment_v5_2026-07-30/D36_v9_split_manifest.csv.gz"
OLD_RUN = ROOT / "reports/modeling/d36_transition_local_expert_70_74_2026-10-07/run_7fec97c76696"

FREEZE_DIR = ROOT / "data/frozen/experiment_v5_clean_identity_2026-10-08"
CLEAN_DATA = FREEZE_DIR / "D36_v9_clean.parquet"
CLEAN_MANIFEST = FREEZE_DIR / "D36_v9_clean_split_manifest.csv.gz"
FREEZE_SUMMARY = FREEZE_DIR / "freeze_summary.json"

AUDIT_DIR = ROOT / "reports/data_quality/d36_identity_clean_audit_2026-10-08"
MODEL_ROOT = ROOT / "reports/modeling/d36_clean_identity_rerun_2026-10-08"

CLASSES = ["BAB", "CA", "CFB", "CR", "IA", "IOA", "MOR", "OI", "OP"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
LOCAL_CLASSES = np.array([0, 1, 4, 5, 6], dtype=int)
SPLITS = ["random_fold", "reference_fold", "province_fold"]
SEEDS = [20261005, 20261006, 20261007]
FOLDS = list(range(5))
FOLD_SEED = 20261008

E36 = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)", "CAO(WT%)",
    "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
    "V(PPM)", "CR(PPM)", "NI(PPM)", "RB(PPM)", "SR(PPM)", "Y(PPM)",
    "ZR(PPM)", "NB(PPM)", "LA(PPM)", "CE(PPM)", "ND(PPM)", "TH(PPM)",
    "BA(PPM)", "HF(PPM)", "TA(PPM)", "PR(PPM)", "SM(PPM)", "EU(PPM)",
    "GD(PPM)", "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "TM(PPM)",
    "YB(PPM)", "LU(PPM)",
]

RATIO_PAIRS = [
    ("BA(PPM)", "NB(PPM)"), ("BA(PPM)", "LA(PPM)"),
    ("BA(PPM)", "TH(PPM)"), ("TH(PPM)", "NB(PPM)"),
    ("TH(PPM)", "YB(PPM)"), ("LA(PPM)", "NB(PPM)"),
    ("SR(PPM)", "ND(PPM)"), ("RB(PPM)", "NB(PPM)"),
    ("LA(PPM)", "YB(PPM)"), ("LA(PPM)", "SM(PPM)"),
    ("SM(PPM)", "YB(PPM)"), ("GD(PPM)", "YB(PPM)"),
    ("DY(PPM)", "YB(PPM)"), ("CE(PPM)", "YB(PPM)"),
    ("NB(PPM)", "YB(PPM)"), ("ZR(PPM)", "Y(PPM)"),
    ("NB(PPM)", "Y(PPM)"), ("ZR(PPM)", "NB(PPM)"),
    ("HF(PPM)", "NB(PPM)"), ("TA(PPM)", "YB(PPM)"),
    ("TIO2(WT%)", "V(PPM)"), ("FEOT(WT%)", "MGO(WT%)"),
    ("CAO(WT%)", "AL2O3(WT%)"), ("K2O(WT%)", "NA2O(WT%)"),
    ("P2O5(WT%)", "TIO2(WT%)"), ("NI(PPM)", "MGO(WT%)"),
    ("CR(PPM)", "MGO(WT%)"),
]

BASE_PARAMS = dict(
    objective="multiclass", n_estimators=600, learning_rate=0.05,
    num_leaves=31, min_child_samples=20, reg_lambda=0.0, reg_alpha=0.0,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
    class_weight="balanced", n_jobs=4, deterministic=True,
    force_col_wise=True, verbosity=-1,
)
LOCAL_PARAMS = dict(
    objective="multiclass", n_estimators=500, learning_rate=0.05,
    num_leaves=15, min_child_samples=30, reg_lambda=5.0, reg_alpha=0.1,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
    class_weight="balanced", n_jobs=4, deterministic=True,
    force_col_wise=True, verbosity=-1,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ref_tokens(value: object) -> set[str]:
    return set(re.findall(r"\d+", str(value)))


def sample_tokens(value: object) -> set[str]:
    return {
        re.sub(r"\s+", " ", part.strip().upper())
        for part in str(value).split(" / ") if part.strip()
    }


def joined(values: pd.Series) -> str:
    return " | ".join(sorted(set(values.dropna().astype(str))))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(p).argmax(axis=1)
    return {
        "n_test": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(9), average="macro", zero_division=0)),
    }


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, np.inf)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def fuse(base: np.ndarray, local: np.ndarray) -> np.ndarray:
    p = np.asarray(base, float).copy()
    q = normalize(local)
    mass = p[:, LOCAL_CLASSES].sum(axis=1)
    p[:, LOCAL_CLASSES] = mass[:, None] * q
    return normalize(p)


def helmert_basis(d: int = 10) -> np.ndarray:
    h = np.zeros((d - 1, d), float)
    for j in range(1, d):
        s = np.sqrt(j * (j + 1))
        h[j - 1, :j] = 1.0 / s
        h[j - 1, j] = -j / s
    return h


def build_d36_74(raw: np.ndarray, *, allow_missing: bool = False) -> tuple[np.ndarray, list[str]]:
    log36 = np.log10(raw)
    pos = {e: i for i, e in enumerate(E36)}
    parts = [log36, log36[:, :10] @ helmert_basis(10).T]
    names = ["log_" + e for e in E36] + [f"ilr_major_{i + 1}" for i in range(9)]
    for a, b in RATIO_PAIRS:
        parts.append((log36[:, pos[a]] - log36[:, pos[b]])[:, None])
        names.append(f"logratio_{a}_over_{b}")
    for num, left, right, name in [
        ("EU(PPM)", "SM(PPM)", "GD(PPM)", "Eu_anomaly"),
        ("CE(PPM)", "LA(PPM)", "PR(PPM)", "Ce_anomaly"),
    ]:
        parts.append((log36[:, pos[num]] - 0.5 * (
            log36[:, pos[left]] + log36[:, pos[right]]
        ))[:, None])
        names.append(name)
    x = np.column_stack(parts).astype(np.float32)
    assert x.shape[1] == 74 and len(names) == 74
    assert (not np.isinf(x).any()) if allow_missing else np.isfinite(x).all()
    return x, names


def audit_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    df = pd.read_parquet(SOURCE_DATA).reset_index(drop=True)
    assert len(df) == 16383 and df["Physical_sample_ID"].is_unique
    assert list(c for c in df.columns if c.endswith("(WT%)") or c.endswith("(PPM)")) == E36
    raw = df[E36].to_numpy(float)
    assert np.isfinite(raw).all() and (raw > 0).all()

    fp = pd.util.hash_pandas_object(df[E36], index=False).astype(str)
    df["_exact36_hash"] = fp
    exact_mask = df.duplicated(E36, keep=False)
    exact_groups = df.loc[exact_mask].groupby("_exact36_hash", sort=False)
    exact_gid = pd.Series(pd.NA, index=df.index, dtype="Int64")
    exact_summary = []
    for gid, (_, x) in enumerate(exact_groups, start=1):
        exact_gid.loc[x.index] = gid
        exact_summary.append({
            "exact_group_id": gid,
            "n_rows": len(x),
            "n_labels": x.Secondary_label.nunique(),
            "n_primary_labels": x.Primary_label.nunique(),
            "n_provinces": x.Province.nunique(),
            "n_references": x.Reference.nunique(),
            "n_databases": x.Database.nunique(),
            "labels": joined(x.Secondary_label),
            "provinces": joined(x.Province),
            "references": joined(x.Reference),
            "physical_ids": joined(x.Physical_sample_ID),
        })
    df["exact_group_id"] = exact_gid

    quarantine_reasons: dict[int, set[str]] = defaultdict(set)
    for i in df.index[exact_mask]:
        quarantine_reasons[int(i)].add("exact_36vector_duplicate")

    source_key = df.Database.astype(str) + "||" + df.Source_record_ID.astype(str)
    df["_source_key"] = source_key
    source_collision_summary = []
    suspicious_source_groups = 0
    for key, x in df.loc[source_key.duplicated(keep=False)].groupby("_source_key", sort=False):
        refs = [ref_tokens(v) for v in x.Reference]
        names = [sample_tokens(v) for v in x.Sample_name]
        shared_ref = bool(set.intersection(*refs)) if refs else False
        shared_name = bool(set.intersection(*names)) if names else False
        # A repeated source identifier inside the same cited source is not an
        # independent physical identity, even when names differ only by a
        # replicate marker (for example, ``MT-18 L3`` versus ``MT-18 L3*``).
        # File-local GEOROC identifier collisions with disjoint references are
        # retained for review rather than treated as the same record.
        suspicious = shared_ref
        if suspicious:
            suspicious_source_groups += 1
            for i in x.index:
                quarantine_reasons[int(i)].add("source_id_with_shared_reference_identity_conflict")
        source_collision_summary.append({
            "source_key": key,
            "n_rows": len(x),
            "n_vectors": x._exact36_hash.nunique(),
            "n_labels": x.Secondary_label.nunique(),
            "n_provinces": x.Province.nunique(),
            "shared_reference_token": shared_ref,
            "shared_sample_token": shared_name,
            "audit_action": "QUARANTINE" if suspicious else "REVIEW_KEEP",
            "labels": joined(x.Secondary_label),
            "provinces": joined(x.Province),
            "sample_names": joined(x.Sample_name),
            "references": joined(x.Reference),
            "physical_ids": joined(x.Physical_sample_ID),
        })

    sample_norm = df.Sample_name.astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    reference_norm = df.Reference.astype(str).str.replace(r"\s+", "", regex=True)
    sample_ref_key = df.Database.astype(str) + "||" + sample_norm + "||" + reference_norm
    df["_sample_reference_key"] = sample_ref_key
    sample_ref_mask = sample_ref_key.duplicated(keep=False)
    sample_ref_summary = []
    for key, x in df.loc[sample_ref_mask].groupby("_sample_reference_key", sort=False):
        for i in x.index:
            quarantine_reasons[int(i)].add("duplicate_normalized_sample_and_reference")
        sample_ref_summary.append({
            "sample_reference_key": key,
            "n_rows": len(x),
            "n_vectors": x._exact36_hash.nunique(),
            "n_labels": x.Secondary_label.nunique(),
            "n_provinces": x.Province.nunique(),
            "labels": joined(x.Secondary_label),
            "provinces": joined(x.Province),
            "source_ids": joined(x.Source_record_ID),
            "physical_ids": joined(x.Physical_sample_ID),
        })

    audit_rows = []
    for i, reasons in sorted(quarantine_reasons.items()):
        row = df.loc[i]
        audit_rows.append({
            "source_row_index": i,
            "action": "QUARANTINE_UNRESOLVED_IDENTITY",
            "reason": "|".join(sorted(reasons)),
            "exact_group_id": row.exact_group_id,
            "exact_36vector_hash": row._exact36_hash,
            "Physical_sample_ID": row.Physical_sample_ID,
            "Sample_name": row.Sample_name,
            "Database": row.Database,
            "Source_record_ID": row.Source_record_ID,
            "Reference": row.Reference,
            "Province": row.Province,
            "Primary_label": row.Primary_label,
            "Secondary_label": row.Secondary_label,
            "Latitude": row.Latitude,
            "Longitude": row.Longitude,
            "Review_status": row.Review_status,
        })
    audit = pd.DataFrame(audit_rows)
    remove = set(audit.source_row_index.astype(int))
    clean = df.drop(index=sorted(remove)).copy().reset_index(drop=True)
    clean = clean.drop(columns=[c for c in clean.columns if c.startswith("_") or c == "exact_group_id"])

    assert len(clean) + len(audit) == len(df)
    assert clean.Physical_sample_ID.is_unique
    assert not clean.duplicated(E36, keep=False).any()
    assert clean.Secondary_label.isin(CLASSES).all()
    assert set(clean.Secondary_label) == set(CLASSES)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exact_summary).to_csv(AUDIT_DIR / "exact_36vector_groups.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(source_collision_summary).to_csv(AUDIT_DIR / "source_id_collision_groups.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sample_ref_summary).to_csv(AUDIT_DIR / "sample_reference_collision_groups.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(AUDIT_DIR / "quarantined_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "policy": {
            "label_selection_from_chemistry": False,
            "quarantine_exact_36vector_duplicates": True,
            "quarantine_source_id_collisions_when_reference_identity_overlaps": True,
            "quarantine_exact_normalized_sample_reference_duplicates": True,
            "unresolved_rows_are_excluded_from_training_and_evaluation": True,
            "original_frozen_data_modified": False,
        },
        "source_rows": len(df),
        "clean_rows": len(clean),
        "quarantined_rows": len(audit),
        "exact_36vector_groups": len(exact_summary),
        "exact_36vector_rows": int(exact_mask.sum()),
        "exact_36vector_secondary_conflict_groups": int(sum(x["n_labels"] > 1 for x in exact_summary)),
        "source_id_collision_groups": len(source_collision_summary),
        "suspicious_source_id_groups_quarantined": suspicious_source_groups,
        "sample_reference_collision_groups": len(sample_ref_summary),
        "clean_exact_36vector_duplicate_rows": int(clean.duplicated(E36, keep=False).sum()),
        "class_counts_before": {str(k): int(v) for k, v in df.Secondary_label.value_counts().sort_index().items()},
        "class_counts_after": {str(k): int(v) for k, v in clean.Secondary_label.value_counts().sort_index().items()},
    }
    dump(summary, AUDIT_DIR / "audit_summary.json")
    return clean, audit, summary


def make_manifest(clean: pd.DataFrame) -> pd.DataFrame:
    y = clean.Secondary_label.to_numpy()
    manifest = clean[[
        "Physical_sample_ID", "Sample_name", "Database", "Source_record_ID",
        "Reference", "Province", "Primary_label", "Secondary_label",
    ]].copy()
    manifest["exact_36vector_hash"] = pd.util.hash_pandas_object(clean[E36], index=False).astype(str)
    manifest["random_component"] = "sample:" + clean.Physical_sample_ID.astype(str)
    manifest["reference_component"] = "Reference:" + clean.Reference.fillna("UNKNOWN").astype(str)
    manifest["province_component"] = "Province:" + clean.Province.fillna("UNKNOWN").astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    random_fold = np.full(len(clean), -1, int)
    for fold, (_, test) in enumerate(skf.split(np.zeros(len(clean)), y)):
        random_fold[test] = fold
    manifest["random_fold"] = random_fold

    for name, component in [("reference", "reference_component"), ("province", "province_component")]:
        fold_values = np.full(len(clean), -1, int)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
        for fold, (_, test) in enumerate(splitter.split(np.zeros(len(clean)), y, manifest[component])):
            fold_values[test] = fold
        manifest[f"{name}_fold"] = fold_values

    for split in SPLITS:
        assert set(manifest[split]) == set(FOLDS)
        table = pd.crosstab(manifest[split], y).reindex(index=FOLDS, columns=CLASSES, fill_value=0)
        assert not (table == 0).any(axis=None), (split, table)
        component = split.replace("_fold", "_component")
        assert manifest.groupby(component)[split].nunique().max() == 1
    assert manifest.exact_36vector_hash.is_unique
    return manifest


def freeze(clean: pd.DataFrame, manifest: pd.DataFrame, audit_summary: dict[str, object]) -> None:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_DATA, index=False)
    manifest.to_csv(CLEAN_MANIFEST, index=False, compression="gzip", encoding="utf-8-sig")
    fold_counts = {}
    for split in SPLITS:
        fold_counts[split] = pd.crosstab(manifest[split], manifest.Secondary_label).reindex(
            index=FOLDS, columns=CLASSES, fill_value=0
        ).reset_index().to_dict(orient="records")
    summary = {
        "status": "PASS",
        "source_data": str(SOURCE_DATA),
        "source_data_sha256": sha256(SOURCE_DATA),
        "clean_data": str(CLEAN_DATA),
        "clean_data_sha256": sha256(CLEAN_DATA),
        "manifest": str(CLEAN_MANIFEST),
        "manifest_sha256": sha256(CLEAN_MANIFEST),
        "rows": len(clean),
        "quarantined_rows": audit_summary["quarantined_rows"],
        "exact_vector_duplicate_rows": 0,
        "physical_id_unique": bool(clean.Physical_sample_ID.is_unique),
        "fold_seed": FOLD_SEED,
        "fold_counts": fold_counts,
        "leakage_checks": {
            "physical_id_cross_fold": 0,
            "exact_vector_cross_fold": 0,
            "declared_group_cross_fold": 0,
        },
    }
    dump(summary, FREEZE_SUMMARY)


def paired_group_bootstrap(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    deltas_a, deltas_f = [], []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[g] for g in sampled])
        mr = metrics(y[take], reference[take])
        mc = metrics(y[take], candidate[take])
        deltas_a.append(mc["accuracy"] - mr["accuracy"])
        deltas_f.append(mc["macro_f1"] - mr["macro_f1"])
    point_r, point_c = metrics(y, reference), metrics(y, candidate)
    return {
        "delta_accuracy": float(point_c["accuracy"] - point_r["accuracy"]),
        "delta_accuracy_lo": float(np.quantile(deltas_a, 0.025)),
        "delta_accuracy_hi": float(np.quantile(deltas_a, 0.975)),
        "delta_macro_f1": float(point_c["macro_f1"] - point_r["macro_f1"]),
        "delta_macro_f1_lo": float(np.quantile(deltas_f, 0.025)),
        "delta_macro_f1_hi": float(np.quantile(deltas_f, 0.975)),
    }


def experiment(clean: pd.DataFrame, manifest: pd.DataFrame, audit_summary: dict[str, object]) -> Path:
    raw = clean[E36].to_numpy(float)
    x, names = build_d36_74(raw)
    y = clean.Secondary_label.map(CLASS_TO_INT).to_numpy(int)
    folds = {s: manifest[s].to_numpy(int) for s in SPLITS}
    groups = {
        "random_fold": manifest.random_component.astype(str).to_numpy(),
        "reference_fold": manifest.reference_component.astype(str).to_numpy(),
        "province_fold": manifest.province_component.astype(str).to_numpy(),
    }
    config = {
        "version": "d36_clean_identity_rerun_v1",
        "cohort": "D36 v9 clean identity quarantine",
        "cohort_n": len(clean),
        "features": "D36_74: 36 log + 9 Helmert ILR + 27 log-ratios + Eu/Ce anomalies",
        "splits": SPLITS,
        "seeds": SEEDS,
        "folds": FOLDS,
        "fold_seed": FOLD_SEED,
        "classes": CLASSES,
        "baseline": BASE_PARAMS,
        "local": LOCAL_PARAMS,
        "local_classes": [CLASSES[i] for i in LOCAL_CLASSES],
        "test_rejection": False,
        "chemistry_used_to_select_labels": False,
    }
    provenance = {
        "config": config,
        "source_hashes": {
            str(SOURCE_DATA.relative_to(ROOT)): sha256(SOURCE_DATA),
            str(SOURCE_MANIFEST.relative_to(ROOT)): sha256(SOURCE_MANIFEST),
            str(CLEAN_DATA.relative_to(ROOT)): sha256(CLEAN_DATA),
            str(CLEAN_MANIFEST.relative_to(ROOT)): sha256(CLEAN_MANIFEST),
        },
        "script_sha256": sha256(Path(__file__)),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    run_id = hashlib.sha256(json.dumps(provenance, sort_keys=True, default=str).encode()).hexdigest()[:12]
    run = MODEL_ROOT / f"run_{run_id}"
    run.mkdir(parents=True, exist_ok=True)
    dump({"run_id": run_id, "run_dir": str(run)}, MODEL_ROOT / "latest_run.json")
    dump({
        "provenance": provenance,
        "audit_policy": audit_summary["policy"],
        "validation_policy": "New clean five-fold partitions; three fixed seeds; no outer-test tuning.",
        "inference_policy": "Chemistry only; every test sample receives nine probabilities and one argmax class; no rejection.",
        "comparison_policy": "Original full-cohort and original-model clean-subset scores are descriptive; the definitive result is the clean refit.",
    }, run / "protocol.json")
    (run / "frozen_implementation.py").write_bytes(Path(__file__).read_bytes())

    predictions: dict[tuple[str, int, str], np.ndarray] = {}
    fit_rows = []
    for split in SPLITS:
        for seed in SEEDS:
            out_base = np.full((len(clean), 9), np.nan, np.float32)
            out_local = np.full((len(clean), 9), np.nan, np.float32)
            for fold in FOLDS:
                test = np.flatnonzero(folds[split] == fold)
                train = np.flatnonzero(folds[split] != fold)
                assert set(groups[split][train]).isdisjoint(set(groups[split][test]))
                fold_dir = run / "models" / split / f"seed_{seed}" / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)

                start = time.time()
                base = LGBMClassifier(**BASE_PARAMS, random_state=seed + 100 * fold)
                base.fit(pd.DataFrame(x[train], columns=names), y[train])
                assert np.array_equal(base.classes_, np.arange(9))
                p_base = normalize(base.predict_proba(pd.DataFrame(x[test], columns=names)))
                out_base[test] = p_base
                (fold_dir / "baseline_model.txt").write_text(base.booster_.model_to_string(), encoding="utf-8")
                fit_rows.append({
                    "split": split, "seed": seed, "fold": fold, "model": "baseline",
                    "train_n": len(train), "test_n": len(test), "elapsed_seconds": time.time() - start,
                })

                start = time.time()
                local_rows = train[np.isin(y[train], LOCAL_CLASSES)]
                local = LGBMClassifier(**LOCAL_PARAMS, random_state=seed + 100 * fold)
                local.fit(pd.DataFrame(x[local_rows], columns=names), y[local_rows])
                assert np.array_equal(local.classes_, LOCAL_CLASSES)
                q = local.predict_proba(pd.DataFrame(x[test], columns=names))
                out_local[test] = fuse(p_base, q)
                (fold_dir / "local_expert_model.txt").write_text(local.booster_.model_to_string(), encoding="utf-8")
                fit_rows.append({
                    "split": split, "seed": seed, "fold": fold, "model": "local_expert",
                    "train_n": len(local_rows), "test_n": len(test), "elapsed_seconds": time.time() - start,
                })
                print("DONE", split, seed, fold, flush=True)

            assert np.isfinite(out_base).all() and np.isfinite(out_local).all()
            assert np.allclose(out_base.sum(1), 1, atol=2e-5)
            assert np.allclose(out_local.sum(1), 1, atol=2e-5)
            predictions[(split, seed, "baseline")] = out_base
            predictions[(split, seed, "local_expert")] = out_local
            np.savez_compressed(
                run / f"oof_D36_74_{split}_seed_{seed}.npz",
                true=y, folds=folds[split], groups=groups[split],
                baseline=out_base, local_expert=out_local,
            )

    pd.DataFrame(fit_rows).to_csv(run / "fit_log.csv", index=False, encoding="utf-8-sig")

    pooled, class_rows, effects, prediction_rows = [], [], [], []
    ensembles: dict[tuple[str, str], np.ndarray] = {}
    for split in SPLITS:
        for seed in SEEDS:
            for model in ["baseline", "local_expert"]:
                pooled.append({
                    "level": "seed", "split": split, "seed": seed, "model": model,
                    **metrics(y, predictions[(split, seed, model)]),
                })
        for model in ["baseline", "local_expert"]:
            p = np.mean([predictions[(split, seed, model)] for seed in SEEDS], axis=0).astype(np.float32)
            ensembles[(split, model)] = p
            pooled.append({
                "level": "seed_ensemble", "split": split, "seed": "ensemble", "model": model,
                **metrics(y, p),
            })
            pred = p.argmax(1)
            pr, rc, f1, sup = precision_recall_fscore_support(
                y, pred, labels=np.arange(9), zero_division=0
            )
            for c, name in enumerate(CLASSES):
                class_rows.append({
                    "split": split, "model": model, "class_name": name,
                    "precision": pr[c], "recall": rc[c], "f1": f1[c], "support": int(sup[c]),
                })
            cm = confusion_matrix(y, pred, labels=np.arange(9))
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
                run / f"confusion_{split}_{model}.csv", encoding="utf-8-sig"
            )
        effects.append({
            "split": split, "candidate": "local_expert", "reference": "baseline",
            **paired_group_bootstrap(
                y, ensembles[(split, "baseline")], ensembles[(split, "local_expert")],
                groups[split], 2000, FOLD_SEED + 1000 * SPLITS.index(split),
            ),
        })
        for i in range(len(clean)):
            prediction_rows.append({
                "Physical_sample_ID": clean.at[i, "Physical_sample_ID"],
                "Secondary_label": clean.at[i, "Secondary_label"],
                "Province": clean.at[i, "Province"],
                "Reference": clean.at[i, "Reference"],
                "split": split,
                "fold": int(folds[split][i]),
                "baseline_pred": CLASSES[int(ensembles[(split, "baseline")][i].argmax())],
                "baseline_confidence": float(ensembles[(split, "baseline")][i].max()),
                "local_expert_pred": CLASSES[int(ensembles[(split, "local_expert")][i].argmax())],
                "local_expert_confidence": float(ensembles[(split, "local_expert")][i].max()),
            })
        np.savez_compressed(
            run / f"oof_D36_74_{split}_ensemble.npz",
            true=y, folds=folds[split], groups=groups[split],
            baseline=ensembles[(split, "baseline")],
            local_expert=ensembles[(split, "local_expert")],
        )

    pooled_df = pd.DataFrame(pooled)
    class_df = pd.DataFrame(class_rows)
    effects_df = pd.DataFrame(effects)
    pooled_df.to_csv(run / "pooled_metrics.csv", index=False, encoding="utf-8-sig")
    class_df.to_csv(run / "class_metrics.csv", index=False, encoding="utf-8-sig")
    effects_df.to_csv(run / "paired_group_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(run / "oof_hard_predictions.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")

    comparison_rows = []
    keep_original = ~pd.read_parquet(SOURCE_DATA).Physical_sample_ID.isin(
        pd.read_csv(AUDIT_DIR / "quarantined_rows.csv").Physical_sample_ID
    ).to_numpy()
    original_y = pd.read_parquet(SOURCE_DATA).Secondary_label.map(CLASS_TO_INT).to_numpy(int)
    old_metrics_file = OLD_RUN / "pooled_metrics.csv"
    if old_metrics_file.exists():
        old_metrics = pd.read_csv(old_metrics_file)
        for split in ["reference_fold", "province_fold"]:
            old_npz = OLD_RUN / f"oof_D36_74_{split}.npz"
            with np.load(old_npz, allow_pickle=False) as z:
                for model in ["baseline", "local_expert"]:
                    p = z[f"{model}_ensemble"]
                    comparison_rows.append({
                        "cohort": "original_model_full_cohort", "split": split, "model": model,
                        **metrics(original_y, p),
                    })
                    comparison_rows.append({
                        "cohort": "original_model_clean_subset_no_refit", "split": split, "model": model,
                        **metrics(original_y[keep_original], p[keep_original]),
                    })
                    comparison_rows.append({
                        "cohort": "clean_refit_new_folds", "split": split, "model": model,
                        **metrics(y, ensembles[(split, model)]),
                    })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(run / "old_new_comparison.csv", index=False, encoding="utf-8-sig")

    write_report(run, audit_summary, pooled_df, class_df, effects_df, comparison_df)
    dump({
        "status": "COMPLETED",
        "cohort_n": len(clean),
        "quarantined_n": audit_summary["quarantined_rows"],
        "models_fitted": len(fit_rows),
        "expected_models": len(SPLITS) * len(SEEDS) * len(FOLDS) * 2,
        "every_test_sample_hard_labeled": True,
        "test_rejection": False,
        "clean_exact_vector_duplicates": 0,
    }, run / "summary.json")
    return run


def write_report(
    run: Path,
    audit: dict[str, object],
    pooled: pd.DataFrame,
    classes: pd.DataFrame,
    effects: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    ens = pooled.query("level == 'seed_ensemble'")
    split_names = {
        "random_fold": "随机五折", "reference_fold": "Reference五折", "province_fold": "Province五折"
    }
    lines = [
        "# D36身份清查与D36_74重跑报告", "",
        "## 数据清查", "",
        f"原始D36共{audit['source_rows']:,}条；隔离{audit['quarantined_rows']:,}条身份或来源归属无法可靠裁决的记录；清理后{audit['clean_rows']:,}条。原冻结文件未被覆盖。", "",
        f"原数据有{audit['exact_36vector_groups']}组完全重复36元素向量，涉及{audit['exact_36vector_rows']}条；其中{audit['exact_36vector_secondary_conflict_groups']}组二级标签冲突。清理集完全重复向量为0。", "",
        "处理原则：仅依据来源编号、样品名、文献和重复向量识别身份冲突；不根据化学组成选择地点或类别。无法核实的冲突记录从训练和评价中同时隔离。", "",
        "## 清理后D36_74三种子概率集成", "",
        "| 验证 | 模型 | Accuracy | Macro-F1 |", "| --- | --- | ---: | ---: |",
    ]
    for split in SPLITS:
        for model in ["baseline", "local_expert"]:
            r = ens[(ens.split == split) & (ens.model == model)].iloc[0]
            lines.append(f"| {split_names[split]} | {model} | {r.accuracy:.2%} | {r.macro_f1:.2%} |")
    lines += ["", "## 局部专家相对基础模型", "", "| 验证 | ΔAccuracy [95%区间] | ΔMacro-F1 [95%区间] |", "| --- | ---: | ---: |"]
    for _, r in effects.iterrows():
        lines.append(
            f"| {split_names[r.split]} | {r.delta_accuracy:+.2%} [{r.delta_accuracy_lo:+.2%}, {r.delta_accuracy_hi:+.2%}] | "
            f"{r.delta_macro_f1:+.2%} [{r.delta_macro_f1_lo:+.2%}, {r.delta_macro_f1_hi:+.2%}] |"
        )
    if len(comparison):
        lines += ["", "## 与原D36_74结果的描述性比较", "", "| 队列/训练 | 验证 | 模型 | Accuracy | Macro-F1 |", "| --- | --- | --- | ---: | ---: |"]
        for _, r in comparison.iterrows():
            lines.append(f"| {r.cohort} | {split_names[r.split]} | {r.model} | {r.accuracy:.2%} | {r.macro_f1:.2%} |")
    lines += [
        "", "## 结论边界", "",
        "- 本轮解决的是可确认的身份重复和来源元数据冲突，不等于已经核实每一条Province标签。",
        "- 新旧重训练使用不同的清理队列和重新生成的五折，因此分数差是整体管线变化，不能全部归因于删除记录。",
        "- 所有测试样品均输出九类概率和唯一类别，没有测试时拒识。",
        "- 仍无独立外部数据，不能把内部随机、Reference或Province分组结果表述为国际领先。",
        "", "## 产物", "",
        f"- 清理数据：`{CLEAN_DATA}`",
        f"- 新划分：`{CLEAN_MANIFEST}`",
        f"- 隔离记录：`{AUDIT_DIR / 'quarantined_rows.csv'}`",
        f"- 完整结果：`{run}`",
    ]
    (run / "实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    clean, audit, audit_summary = audit_and_clean()
    manifest = make_manifest(clean)
    freeze(clean, manifest, audit_summary)
    run = experiment(clean, manifest, audit_summary)
    print("COMPLETED", run, flush=True)


if __name__ == "__main__":
    main()
