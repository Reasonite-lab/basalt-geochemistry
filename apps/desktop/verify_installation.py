"""Mechanical verification for the installed D36 desktop application."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from d36_predictor_core import (
    BUNDLE_DIR,
    CLASSES,
    D36Engine,
    E36,
    export_result,
)


def main() -> None:
    engine = D36Engine()
    example_input = pd.read_csv(BUNDLE_DIR / "example_input.csv")
    expected = pd.read_csv(BUNDLE_DIR / "example_predictions.csv")
    canonical = example_input.rename(columns={"Physical_sample_ID": "Sample_ID"})
    result = engine.analyze(canonical, use_spatial_prior=True, aggregate_top_n=20)
    actual = result.probabilities[[f"P_{name}" for name in CLASSES]].to_numpy(float)
    target = expected[[f"prob_{name}" for name in CLASSES]].to_numpy(float)
    replay_error = float(np.max(np.abs(actual - target)))
    if replay_error > 1e-12:
        raise RuntimeError(f"Frozen probability replay mismatch: {replay_error}")
    if not np.allclose(actual.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("Probability rows do not sum to one")
    if len(result.region_ranking) != 20 or len(result.reference_ranking) != 20:
        raise RuntimeError("Ranking row count mismatch")
    if not result.region_ranking["相似指数"].is_monotonic_decreasing:
        raise RuntimeError("Region ranking is not descending")
    if not result.reference_ranking["相似指数"].is_monotonic_decreasing:
        raise RuntimeError("Reference ranking is not descending")

    chemistry_only = engine.analyze(canonical, use_spatial_prior=False, aggregate_top_n=5)
    if not chemistry_only.probabilities["推理路径"].str.contains("纯化学").all():
        raise RuntimeError("Chemistry-only route was not preserved")
    no_coords = canonical.drop(columns=["Latitude", "Longitude"], errors="ignore")
    fallback = engine.analyze(no_coords, use_spatial_prior=True, aggregate_top_n=5)
    if not fallback.probabilities["推理路径"].str.contains("回退").all():
        raise RuntimeError("Missing-coordinate fallback was not preserved")

    with tempfile.TemporaryDirectory(prefix="d36_app_verify_") as temporary:
        output = Path(temporary) / "verification.xlsx"
        export_result(result, output)
        if not output.exists() or not output.with_suffix(".audit.json").exists():
            raise RuntimeError("Excel/audit export failed")
        required_sheets = {
            "九分类概率",
            "地区相似度_总体",
            "文献相似度_总体",
            "地区相似度_逐样品",
            "文献相似度_逐样品",
            "规范化输入",
        }
        with pd.ExcelFile(output) as workbook:
            if not required_sheets.issubset(set(workbook.sheet_names)):
                raise RuntimeError("Exported workbook sheets are incomplete")

    report = {
        "status": "PASS",
        "example_rows": int(len(canonical)),
        "D36_columns": int(len(E36)),
        "classes": CLASSES,
        "frozen_probability_replay_max_abs_error": replay_error,
        "region_ranking_descending": True,
        "reference_ranking_descending": True,
        "chemistry_only_route": True,
        "missing_coordinate_exact_fallback_route": True,
        "excel_export": True,
        "test_rejection": False,
    }
    path = Path(__file__).resolve().parent / "installation_verification.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

