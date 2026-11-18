"""Requirement-level acceptance for the complete desktop deliverable."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from d36_predictor_core import BUNDLE_DIR, CLASSES, D36Engine, export_result


APP_DIR = Path(__file__).resolve().parent
DESKTOP_SHORTCUT = Path.home() / "Desktop" / "D36构造环境判别工具.lnk"


def main() -> None:
    engine = D36Engine()
    example = pd.read_csv(BUNDLE_DIR / "example_input.csv").head(3)
    result = engine.analyze(example, use_spatial_prior=False)
    probability = result.probabilities[[f"P_{name}" for name in CLASSES]].to_numpy(float)
    checks = {
        "nine_probabilities_for_every_row": bool(probability.shape == (3, 9)),
        "probabilities_finite": bool(np.isfinite(probability).all()),
        "probabilities_sum_to_one": bool(np.allclose(probability.sum(1), 1.0, atol=1e-12)),
        "every_row_has_hard_label": bool(result.probabilities["预测环境"].notna().all()),
        "all_234_regions_ranked": bool(len(result.region_ranking) == 234),
        "region_similarity_descending": bool(result.region_ranking["相似指数"].is_monotonic_decreasing),
        "all_1689_references_ranked": bool(len(result.reference_ranking) == 1689),
        "reference_similarity_descending": bool(result.reference_ranking["相似指数"].is_monotonic_decreasing),
        "region_nearest_evidence_present": bool(result.region_ranking["最近证据样品"].astype(str).str.len().gt(0).all()),
        "reference_nearest_evidence_present": bool(result.reference_ranking["最近证据样品"].astype(str).str.len().gt(0).all()),
        "desktop_shortcut_exists": DESKTOP_SHORTCUT.exists(),
    }
    acceptance_output = APP_DIR / "outputs" / "final_acceptance_example.xlsx"
    export_result(result, acceptance_output)
    checks["complete_excel_exists"] = acceptance_output.exists()
    checks["audit_json_exists"] = acceptance_output.with_suffix(".audit.json").exists()
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checked_at": datetime.now().astimezone().isoformat(),
        "checks": checks,
        "model_bundle_status": engine.bundle["status"],
        "model_probability_columns": [f"P_{name}" for name in CLASSES],
        "region_rows": int(len(result.region_ranking)),
        "reference_rows": int(len(result.reference_ranking)),
        "example_output": str(acceptance_output),
        "desktop_shortcut": str(DESKTOP_SHORTCUT),
    }
    path = APP_DIR / "final_acceptance.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

