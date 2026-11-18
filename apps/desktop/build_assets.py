"""Build the immutable local assets used by the D36 desktop application."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from d36_predictor_core import APP_DIR, ASSET_DIR, E36, sha256


ROOT = Path(__file__).resolve().parents[2]
SOURCE_BUNDLE = ROOT / "reports/modeling/geosingle_prior_d36_v1_bundle_2026-10-17"
SOURCE_DATA = (
    ROOT
    / "data/frozen/experiment_v5_clean_identity_expansion06_2026-10-22"
    / "D36_v9_clean_expansion06.parquet"
)
TARGET_BUNDLE = ASSET_DIR / "geosingle_bundle"
TARGET_INDEX = ASSET_DIR / "retrieval_library_v1.npz"
TARGET_METADATA = ASSET_DIR / "retrieval_library_v1.json"


def safe_strings(series: pd.Series) -> np.ndarray:
    return series.fillna("").astype(str).to_numpy(dtype=str)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_BUNDLE.exists():
        shutil.rmtree(TARGET_BUNDLE)
    shutil.copytree(SOURCE_BUNDLE, TARGET_BUNDLE)

    frame = pd.read_parquet(SOURCE_DATA)
    values = frame[E36].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    if not valid.all():
        raise RuntimeError(f"Formal D36 retrieval rows are invalid: {(~valid).sum()}")
    np.savez_compressed(
        TARGET_INDEX,
        log36=np.log10(values).astype(np.float32),
        physical_id=safe_strings(frame["Physical_sample_ID"]),
        sample_name=safe_strings(frame["Sample_name"]),
        province=safe_strings(frame["Province"]),
        reference=safe_strings(frame["Reference"]),
        label=safe_strings(frame["Secondary_label"]),
        database=safe_strings(frame["Database"]),
        source_url=safe_strings(frame["Source_URL"]),
    )
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_data": str(SOURCE_DATA),
        "source_data_sha256": sha256(SOURCE_DATA),
        "rows": int(len(frame)),
        "provinces": int(frame["Province"].nunique()),
        "references": int(frame["Reference"].nunique()),
        "elements": E36,
        "index_sha256": sha256(TARGET_INDEX),
        "distance": "RMS Euclidean distance across 36 log10 element concentrations",
        "group_aggregation": "for each query, mean of closest up to 3 samples in group; then mean across queries",
        "similarity": "100 * 10^(-group distance)",
    }
    TARGET_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

