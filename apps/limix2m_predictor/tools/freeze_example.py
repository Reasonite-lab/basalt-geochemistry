"""Freeze the bundled example input and example predictions for the app."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\AI地学\apps\d36_desktop_predictor")
from d36_predictor_core import E36  # reuse the canonical element list source

sys.path.insert(0, r"D:\AI地学\repos\basalt-geochemistry\apps\limix2m_predictor")
from limix_core import ASSET_DIR, CLASSES, LimiX2MEngine  # noqa: E402

SOURCE = r"D:\AI地学\data\frozen\experiment_v5_clean_identity_expansion13_2026-10-26\D36_v9_clean_expansion13.parquet"

data = pd.read_parquet(SOURCE)
values = data[E36].apply(pd.to_numeric, errors="coerce").to_numpy(float)
valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
data = data.loc[valid].reset_index(drop=True)

# Same five samples as the v2 D36 example for cross-tool comparability.
ids = ["PHY:d2ab7b42a3f7122d216b", "PHY:443ce3e6445f09808ad1",
       "PHY:0fa0f5c9c91425653850", "PHY:888d22b04a4833590892",
       "PHY:bd4fe8f009d99d9c4c7f"]
index = data.index[data.Physical_sample_ID.astype(str).isin(ids)]
example = data.loc[index, ["Physical_sample_ID", "Latitude", "Longitude", *E36]].copy()
example = example.rename(columns={"Physical_sample_ID": "Sample_ID"})
input_path = ASSET_DIR / "example_input.csv"
example.to_csv(input_path, index=False, encoding="utf-8-sig")
print("wrote", input_path, len(example), "rows")

engine = LimiX2MEngine()
result = engine.analyze(example)
probability = result.probabilities[[f"P_{c}" for c in CLASSES]].to_numpy(float)
pred = pd.DataFrame(
    {"sample_id": result.probabilities["样品编号"].astype(str),
     **{f"prob_{c}": probability[:, i] for i, c in enumerate(CLASSES)}}
)
pred_path = ASSET_DIR / "example_predictions.csv"
pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
print("wrote", pred_path)

import json
(ASSET_DIR / "example_predictions.csv.audit.json").write_text(json.dumps(
    {"generated_by": "tools/freeze_example.py", "model": "LimiX-2M",
     "rows": int(len(example)),
     "note": "frozen example probabilities for portable self-test replay"},
    ensure_ascii=False, indent=2), encoding="utf-8")
print("done")
