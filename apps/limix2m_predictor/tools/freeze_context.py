"""Freeze the fixed LimiX-2M reference context and bundled model assets.

Context: 1024 complete D36 rows stratified by class from frozen expansion13,
seed 20261118. Saved as float32 log10 concentrations (NaN = unmeasured).
Also copies the official LimiX-2M inference code and checkpoint into the
application assets folder for offline desktop use.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
ASSETS = APP / "assets"
ROOT = Path(r"D:\AI地学")
SOURCE_DATA = ROOT / "data/frozen/experiment_v5_clean_identity_expansion13_2026-10-26/D36_v9_clean_expansion13.parquet"
UPSTREAM = ROOT / ".codex_tmp/limix2m_official"
DEPENDENCIES = ROOT / ".codex_tmp/limix2_dependencies"
CHECKPOINT = ROOT / ".codex_tmp/limix2_weights/LimiX-2M.ckpt"
WEIGHTS_SHA = "16f385d5201dc17cd6a8eb18becb0b4f8b3603838423b9f06062106acc965513"
CLASSES = ["BAB", "CA", "CFB", "CR", "IA", "IOA", "MOR", "OI", "OP"]
E36 = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)", "CAO(WT%)",
    "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
    "V(PPM)", "CR(PPM)", "NI(PPM)", "RB(PPM)", "SR(PPM)", "Y(PPM)",
    "ZR(PPM)", "NB(PPM)", "LA(PPM)", "CE(PPM)", "ND(PPM)", "TH(PPM)",
    "BA(PPM)", "HF(PPM)", "TA(PPM)", "PR(PPM)", "SM(PPM)", "EU(PPM)",
    "GD(PPM)", "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "TM(PPM)",
    "YB(PPM)", "LU(PPM)",
]
CONTEXT_BUDGET = 1024
SEED = 20261118


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data = pd.read_parquet(SOURCE_DATA)
    values = data[E36].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    data = data.loc[valid].reset_index(drop=True)
    values = values[valid]
    y = pd.Categorical(data.Secondary_label, categories=CLASSES).codes
    assert (y >= 0).all() and len(data) == 16455

    rng = np.random.default_rng(SEED)
    picked = []
    base, remainder = divmod(CONTEXT_BUDGET, len(CLASSES))
    for code in range(len(CLASSES)):
        indices = np.flatnonzero(y == code)
        take = base + (1 if code < remainder else 0)
        picked.append(rng.choice(indices, size=min(take, len(indices)), replace=False))
    context_rows = np.sort(np.concatenate(picked))
    assert len(context_rows) == CONTEXT_BUDGET

    ASSETS.mkdir(parents=True, exist_ok=True)
    context_x = np.log10(values[context_rows]).astype(np.float32)
    context_y = y[context_rows].astype(np.int64)
    np.savez_compressed(
        ASSETS / "context_v1.npz",
        x=context_x,
        y=context_y,
        sample_id=data.Physical_sample_ID.astype(str).to_numpy()[context_rows],
        classes=np.asarray(CLASSES),
        elements=np.asarray(E36),
    )
    meta = {
        "context_budget": int(len(context_rows)),
        "seed": SEED,
        "source_data": str(SOURCE_DATA),
        "source_data_sha256": sha256(SOURCE_DATA),
        "source_rows_total": 16459,
        "source_rows_valid": int(len(data)),
        "class_counts": {CLASSES[c]: int((context_y == c).sum()) for c in range(len(CLASSES))},
        "transformation": "log10, unmeasured retained as NaN",
        "protocol": "fixed stratified reference context for in-context classification",
        "model": "LimiX-2M",
        "checkpoint_sha256": sha256(CHECKPOINT),
        "upstream_license": "Apache-2.0 (LimiX-2M, stable-ai)",
    }
    assert meta["checkpoint_sha256"] == WEIGHTS_SHA
    (ASSETS / "context_v1.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Bundle official inference code and checkpoint for offline use.
    code_target = ASSETS / "limix_upstream"
    if code_target.exists():
        shutil.rmtree(code_target)
    for name in ("inference", "model", "utils", "config"):
        shutil.copytree(UPSTREAM / name, code_target / name)
    for extra in ("LICENSE.txt",):
        shutil.copyfile(UPSTREAM / extra, code_target / extra)
    shutil.copytree(DEPENDENCIES, ASSETS / "limix_dependencies")
    shutil.copyfile(CHECKPOINT, ASSETS / "LimiX-2M.ckpt")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
