"""Shared, offline-only utilities for the cleaned D22_42 replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/frozen/experiment_v2_2026-07-23/D22_QC.parquet"
OUT_ROOT = ROOT / "reports/modeling/d22_42_process_context_replay_2026-10-19"

CLASSES = ["BAB", "CA", "CFB", "CR", "IA", "IOA", "MOR", "OI", "OP"]
CLASS_TO_INT = {name: index for index, name in enumerate(CLASSES)}
E22 = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)", "CAO(WT%)",
    "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
    "V(PPM)", "CR(PPM)", "NI(PPM)", "RB(PPM)", "SR(PPM)", "Y(PPM)",
    "ZR(PPM)", "NB(PPM)", "LA(PPM)", "CE(PPM)", "ND(PPM)", "TH(PPM)",
]
RATIO_PAIRS = [
    ("TH(PPM)", "NB(PPM)"),
    ("LA(PPM)", "NB(PPM)"),
    ("ZR(PPM)", "Y(PPM)"),
    ("ZR(PPM)", "NB(PPM)"),
    ("TIO2(WT%)", "V(PPM)"),
    ("FEOT(WT%)", "MGO(WT%)"),
    ("CAO(WT%)", "AL2O3(WT%)"),
    ("K2O(WT%)", "NA2O(WT%)"),
    ("P2O5(WT%)", "TIO2(WT%)"),
    ("NI(PPM)", "MGO(WT%)"),
    ("CR(PPM)", "MGO(WT%)"),
]
BASE_PARAMS = dict(
    objective="multiclass",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    reg_lambda=0.0,
    reg_alpha=0.0,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    class_weight="balanced",
    n_jobs=8,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
)
EPS = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), EPS, np.inf)
    return probability / probability.sum(axis=1, keepdims=True)


def helmert_basis(dimension: int = 10) -> np.ndarray:
    basis = np.zeros((dimension - 1, dimension), dtype=float)
    for column in range(1, dimension):
        scale = np.sqrt(column * (column + 1))
        basis[column - 1, :column] = 1.0 / scale
        basis[column - 1, column] = -column / scale
    return basis


def build_d22_42(raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if raw.shape[1] != 22 or not np.isfinite(raw).all() or not (raw > 0).all():
        raise ValueError("D22_42 requires 22 finite, strictly positive measurements")
    log22 = np.log10(raw)
    positions = {name: index for index, name in enumerate(E22)}
    parts: list[np.ndarray] = [log22, log22[:, :10] @ helmert_basis(10).T]
    names = [f"log_{name}" for name in E22]
    names += [f"ilr_major_{index + 1}" for index in range(9)]
    for left, right in RATIO_PAIRS:
        parts.append((log22[:, positions[left]] - log22[:, positions[right]])[:, None])
        names.append(f"logratio_{left}_over_{right}")
    features = np.column_stack(parts).astype(np.float32)
    if features.shape[1] != 42 or len(names) != 42 or not np.isfinite(features).all():
        raise RuntimeError("Invalid D22_42 construction")
    return features, names


def exact_vector_hashes(raw: np.ndarray) -> np.ndarray:
    canonical = np.asarray(raw, dtype="<f8", order="C")
    return np.asarray(
        [hashlib.sha256(row.tobytes()).hexdigest() for row in canonical], dtype=str
    )


def load_clean_development() -> tuple[pd.DataFrame, np.ndarray, dict[str, object], pd.DataFrame]:
    original = pd.read_parquet(DATA).reset_index(drop=True)
    development = original.loc[original["split_role"].eq("development")].copy()
    development["source_row"] = development.index.astype(int)
    development.reset_index(drop=True, inplace=True)
    raw = development[E22].to_numpy(float)
    hashes = exact_vector_hashes(raw)
    development["exact_22vector_hash"] = hashes
    group_label_count = development.groupby("exact_22vector_hash")["Secondary_label"].transform("nunique")
    conflict = group_label_count.gt(1)
    candidates = development.loc[~conflict].copy()
    candidates.sort_values(
        ["exact_22vector_hash", "Physical_sample_ID", "source_row"],
        kind="stable",
        inplace=True,
    )
    keep = ~candidates.duplicated("exact_22vector_hash", keep="first")
    retained = candidates.loc[keep].sort_values("source_row", kind="stable").reset_index(drop=True)

    conflict_rows = development.loc[conflict].copy()
    conflict_rows["exclusion_reason"] = "exact_D22_vector_has_multiple_labels"
    duplicate_rows = candidates.loc[~keep].copy()
    duplicate_rows["exclusion_reason"] = "same_label_exact_D22_duplicate"
    excluded = pd.concat([conflict_rows, duplicate_rows], ignore_index=True)
    excluded = excluded[
        ["Physical_sample_ID", "source_row", "Secondary_label", "Province", "Reference",
         "exact_22vector_hash", "exclusion_reason"]
    ].sort_values(["exclusion_reason", "exact_22vector_hash", "Physical_sample_ID"], kind="stable")

    duplicate_groups = development.groupby("exact_22vector_hash").size()
    conflicting_groups = development.groupby("exact_22vector_hash")["Secondary_label"].nunique()
    audit = {
        "all_D22_rows": int(len(original)),
        "development_rows_before_cleaning": int(len(development)),
        "exact_duplicate_groups_size_gt_1": int((duplicate_groups > 1).sum()),
        "rows_in_exact_duplicate_groups": int(duplicate_groups[duplicate_groups > 1].sum()),
        "conflicting_exact_groups": int((conflicting_groups > 1).sum()),
        "rows_removed_for_label_conflict": int(conflict.sum()),
        "same_label_duplicate_rows_removed": int((~keep).sum()),
        "development_rows_after_cleaning": int(len(retained)),
        "retained_unique_physical_sample_ids": int(retained["Physical_sample_ID"].nunique()),
        "retained_unique_exact_vectors": int(retained["exact_22vector_hash"].nunique()),
        "all_22_values_finite_positive": bool(
            np.isfinite(retained[E22].to_numpy(float)).all()
            and (retained[E22].to_numpy(float) > 0).all()
        ),
    }
    if audit["retained_unique_physical_sample_ids"] != len(retained):
        raise RuntimeError("Physical_sample_ID is not unique after cleaning")
    if audit["retained_unique_exact_vectors"] != len(retained):
        raise RuntimeError("Exact D22 vectors remain duplicated after cleaning")
    return retained, retained["exact_22vector_hash"].to_numpy(str), audit, excluded


def score(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(probability).argmax(1)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(
            f1_score(y, prediction, labels=np.arange(len(CLASSES)), average="macro", zero_division=0)
        ),
    }


def per_class(y: np.ndarray, probability: np.ndarray) -> list[dict[str, object]]:
    prediction = np.asarray(probability).argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=np.arange(len(CLASSES)), zero_division=0
    )
    return [
        {
            "class": CLASSES[index],
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index in range(len(CLASSES))
    ]


def class_f1(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return precision_recall_fscore_support(
        y,
        np.asarray(probability).argmax(1),
        labels=np.arange(len(CLASSES)),
        zero_division=0,
    )[2]


def clustered_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    unique = pd.unique(groups)
    group_index = {group: index for index, group in enumerate(unique)}
    encoded_group = np.asarray([group_index[group] for group in groups], dtype=int)
    baseline_prediction = np.asarray(baseline).argmax(1)
    candidate_prediction = np.asarray(candidate).argmax(1)
    baseline_confusion = np.zeros((len(unique), len(CLASSES), len(CLASSES)), dtype=np.int64)
    candidate_confusion = np.zeros_like(baseline_confusion)
    np.add.at(baseline_confusion, (encoded_group, y, baseline_prediction), 1)
    np.add.at(candidate_confusion, (encoded_group, y, candidate_prediction), 1)
    draw_counts = rng.multinomial(
        len(unique), np.full(len(unique), 1.0 / len(unique)), size=repeats
    )
    baseline_draw = np.einsum("rg,gij->rij", draw_counts, baseline_confusion, optimize=True)
    candidate_draw = np.einsum("rg,gij->rij", draw_counts, candidate_confusion, optimize=True)

    def draw_scores(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total = confusion.sum(axis=(1, 2))
        accuracy = np.diagonal(confusion, axis1=1, axis2=2).sum(axis=1) / total
        true_count = confusion.sum(axis=2)
        predicted_count = confusion.sum(axis=1)
        true_positive = np.diagonal(confusion, axis1=1, axis2=2)
        denominator = true_count + predicted_count
        f1 = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros_like(true_positive, dtype=float),
            where=denominator > 0,
        )
        return accuracy, f1.mean(axis=1)

    baseline_accuracy, baseline_macro = draw_scores(baseline_draw)
    candidate_accuracy, candidate_macro = draw_scores(candidate_draw)
    accuracy_delta = candidate_accuracy - baseline_accuracy
    macro_delta = candidate_macro - baseline_macro
    before, after = score(y, baseline), score(y, candidate)
    return {
        "repeats": repeats,
        "group_count": int(len(unique)),
        "delta_accuracy": float(after["accuracy"] - before["accuracy"]),
        "delta_accuracy_95": [float(np.quantile(accuracy_delta, 0.025)), float(np.quantile(accuracy_delta, 0.975))],
        "delta_macro_f1": float(after["macro_f1"] - before["macro_f1"]),
        "delta_macro_f1_95": [float(np.quantile(macro_delta, 0.025)), float(np.quantile(macro_delta, 0.975))],
    }
