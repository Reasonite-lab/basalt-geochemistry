"""Core inference service for the LimiX-2M desktop app.

LimiX-2M is an in-context tabular foundation model: probabilities for unknown
samples are produced against a frozen labeled reference context (1024 complete
D36 rows from expansion13). Missing element measurements are retained as
NaN, which LimiX-2M handles natively (joint modeling of variables and
missing values).

Input chemistry: log10-transformed D36 concentrations; unmeasured = NaN.
Output: nine-class probabilities with the same class order as the D36 tool.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_DIR))
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_DIR
ASSET_DIR = RESOURCE_DIR / "assets"
CONTEXT_PATH = ASSET_DIR / "context_v1.npz"
CONTEXT_META = ASSET_DIR / "context_v1.json"
CHECKPOINT_PATH = ASSET_DIR / "LimiX-2M.ckpt"
RETRIEVAL_INDEX = ASSET_DIR / "retrieval_library_v1.npz"
UPSTREAM_DIR = ASSET_DIR / "limix_upstream"
DEPENDENCIES_DIR = ASSET_DIR / "limix_dependencies"
CONFIG_PATH = UPSTREAM_DIR / "config" / "cls_default_noretrieval.json"
OUTPUT_DIR = APP_DIR / "outputs"

CLASSES = ["BAB", "CA", "CFB", "CR", "IA", "IOA", "MOR", "OI", "OP"]
CLASS_CN = {
    "BAB": "弧后盆地", "CA": "大陆弧", "CFB": "大陆溢流玄武岩", "CR": "大陆裂谷",
    "IA": "岛弧", "IOA": "洋内弧", "MOR": "洋中脊", "OI": "洋岛", "OP": "大洋高原",
}
E36 = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)", "CAO(WT%)",
    "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
    "V(PPM)", "CR(PPM)", "NI(PPM)", "RB(PPM)", "SR(PPM)", "Y(PPM)",
    "ZR(PPM)", "NB(PPM)", "LA(PPM)", "CE(PPM)", "ND(PPM)", "TH(PPM)",
    "BA(PPM)", "HF(PPM)", "TA(PPM)", "PR(PPM)", "SM(PPM)", "EU(PPM)",
    "GD(PPM)", "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "TM(PPM)",
    "YB(PPM)", "LU(PPM)",
]
WEIGHTS_SHA = "16f385d5201dc17cd6a8eb18becb0b4f8b3603838423b9f06062106acc965513"
MAX_INPUT_ROWS = 500
MIN_MEASURED = 18           # 全部10项主量 + 至少8项微量（D18 下限）
MIN_TRACES_MEASURED = 8
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _column_key(name: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    by_key = {_column_key(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if _column_key(candidate) in by_key:
            return by_key[_column_key(candidate)]
    return None


def canonicalize_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical input table with missing-aware validation (raises actionable errors)."""
    if frame.empty:
        raise ValueError("输入表没有数据行。")
    if len(frame) > MAX_INPUT_ROWS:
        raise ValueError(f"一次最多分析{MAX_INPUT_ROWS}个样品；当前为{len(frame)}个。请分批导入。")

    source = frame.copy()
    output = pd.DataFrame(index=source.index)
    id_column = _find_column(
        source, ["Physical_sample_ID", "Sample_ID", "Sample_name", "样品编号", "样品名"]
    )
    if id_column is None:
        output["Sample_ID"] = [f"unknown_{i + 1:04d}" for i in range(len(source))]
    else:
        identifiers = source[id_column].astype(str).str.strip()
        blank = identifiers.eq("") | identifiers.str.lower().isin(["nan", "none"])
        identifiers.loc[blank] = [f"unknown_{i + 1:04d}" for i in np.flatnonzero(blank)]
        output["Sample_ID"] = identifiers.to_numpy()

    for canonical in E36:
        short = canonical.split("(")[0]
        column = _find_column(source, [canonical, short])
        if column is None:
            output[canonical] = np.nan
            continue
        output[canonical] = pd.to_numeric(source[column], errors="coerce")

    values = output[E36].to_numpy(float)
    invalid = np.isfinite(values) & (values <= 0)
    if invalid.any():
        row_index, column_index = np.argwhere(invalid)[0]
        raise ValueError(
            f"样品 {output.iloc[int(row_index)]['Sample_ID']} 的 "
            f"{E36[int(column_index)]}={output.iloc[int(row_index)][E36[int(column_index)]]!r} 无效；"
            "已测量的元素必须为有限正数，未测量请留空。"
        )

    majors_ok = np.isfinite(values[:, :10]).all(axis=1)
    n_traces = np.isfinite(values[:, 10:]).sum(axis=1)
    n_measured = np.isfinite(values).sum(axis=1)
    coverage_ok = majors_ok & (n_traces >= MIN_TRACES_MEASURED)
    if not coverage_ok.all():
        row_index = int(np.flatnonzero(~coverage_ok)[0])
        sample = output.iloc[row_index]["Sample_ID"]
        lack_major = [E36[i] for i in range(10) if not np.isfinite(values[row_index, i])]
        detail = []
        if lack_major:
            detail.append("缺少主量：" + "、".join(lack_major))
        if n_traces[row_index] < MIN_TRACES_MEASURED:
            lack_traces = [
                E36[i] for i in range(10, len(E36))
                if not np.isfinite(values[row_index, i])
            ]
            detail.append(
                f"微量仅实测{int(n_traces[row_index])}项（需≥{MIN_TRACES_MEASURED}），"
                "未测量：" + "、".join(lack_traces[:12])
                + ("……" if len(lack_traces) > 12 else "")
            )
        raise ValueError(
            f"样品 {sample} 覆盖不足（实测{n_measured[row_index]}/36项）："
            + "；".join(detail)
            + f"。最低要求：全部10项主量加≥{MIN_TRACES_MEASURED}项微量。"
        )

    lat_column = _find_column(source, ["Latitude", "Lat", "纬度"])
    lon_column = _find_column(source, ["Longitude", "Lon", "Lng", "经度"])
    output["Latitude"] = (
        pd.to_numeric(source[lat_column], errors="coerce") if lat_column else np.nan
    )
    output["Longitude"] = (
        pd.to_numeric(source[lon_column], errors="coerce") if lon_column else np.nan
    )
    return output.reset_index(drop=True)


def read_input_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        frame = pd.read_excel(path)
    elif suffix in {".csv", ".txt"}:
        errors: list[Exception] = []
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                frame = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError as error:
                errors.append(error)
        else:
            raise errors[-1]
    else:
        raise ValueError("只支持CSV、TXT或XLSX输入文件。")
    return canonicalize_input(frame)


@dataclass
class LimiXResult:
    input_table: pd.DataFrame
    probabilities: pd.DataFrame
    nearest_samples: pd.DataFrame
    region_ranking: pd.DataFrame
    reference_ranking: pd.DataFrame
    audit: dict[str, Any]


class LimiX2MEngine:
    """Load the frozen LimiX-2M predictor once and serve repeated analyses."""

    def __init__(self, asset_dir: str | Path = ASSET_DIR) -> None:
        self.asset_dir = Path(asset_dir)
        self.context_meta = json.loads(CONTEXT_META.read_text(encoding="utf-8"))
        if sha256(CHECKPOINT_PATH) != WEIGHTS_SHA:
            raise ValueError("LimiX-2M 检查点校验失败。")
        with np.load(CONTEXT_PATH, allow_pickle=False) as saved:
            self.context_x = saved["x"].astype(np.float32)
            self.context_y = saved["y"].astype(np.int64)
            self.context_elements = saved["elements"].astype(str).tolist()
            self.context_classes = saved["classes"].astype(str).tolist()
        if self.context_elements != E36 or self.context_classes != CLASSES:
            raise ValueError("冻结上下文与元素/类别顺序不一致。")

        if not RETRIEVAL_INDEX.exists():
            raise FileNotFoundError(f"化学检索库不存在：{RETRIEVAL_INDEX}")
        with np.load(RETRIEVAL_INDEX, allow_pickle=False) as saved:
            self.train_log36 = saved["log36"].astype(np.float64)
            self.train_sample_id = saved["physical_id"].astype(str)
            self.train_sample_name = saved["sample_name"].astype(str)
            self.train_province = saved["province"].astype(str)
            self.train_reference = saved["reference"].astype(str)
            self.train_label = saved["label"].astype(str)
            self.train_database = saved["database"].astype(str)
            self.train_source_url = saved["source_url"].astype(str)
        if self.train_log36.shape != (len(self.train_sample_id), len(E36)):
            raise ValueError("化学检索库维数错误。")
        self.index_sha256 = sha256(RETRIEVAL_INDEX)

        sys.path[:0] = [str(DEPENDENCIES_DIR.resolve()), str(UPSTREAM_DIR.resolve())]
        import torch

        from inference.predictor import LimiXPredictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(4)
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.predictor = LimiXPredictor(
            device=torch.device(self.device),
            model_path=str(CHECKPOINT_PATH.resolve()),
            inference_config=config,
            mix_precision=self.device == "cuda",
            seed=20261118,
        )

    def predict_log10(self, query_log10: np.ndarray, query_batch: int = 128) -> np.ndarray:
        """In-context classification for log10-transformed queries (NaN = missing)."""
        query = np.asarray(query_log10, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        parts = []
        for start in range(0, len(query), query_batch):
            block = query[start:start + query_batch]
            parts.append(
                self.predictor.predict(
                    self.context_x, self.context_y, block, task_type="Classification"
                )
            )
        probability = np.concatenate(parts, axis=0)
        probability = np.clip(np.asarray(probability, dtype=float), EPS, None)
        return probability / probability.sum(axis=1, keepdims=True)

    def _distance_matrix(self, raw: np.ndarray) -> np.ndarray:
        """Per-query masked log10-RMS distance to every retrieval-library row."""
        raw = np.asarray(raw, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            query = np.where(
                np.isfinite(raw), np.log10(np.where(np.isfinite(raw), raw, 1.0)), np.nan
            ).astype(np.float64)
        library = self.train_log36
        observed = np.isfinite(query)
        query_zeroed = np.where(observed, query, 0.0)
        squared = (
            np.sum(query_zeroed * query_zeroed, axis=1)[:, None]
            + observed.astype(np.float64) @ (library * library).T
            - 2.0 * query_zeroed @ library.T
        )
        np.maximum(squared, 0.0, out=squared)
        count = np.maximum(observed.sum(axis=1), 1)
        return np.sqrt(squared) / np.sqrt(count[:, None])

    @staticmethod
    def _similarity(rms: float) -> float:
        return 100.0 * (10.0 ** (-float(rms)))

    def _sample_ranking(self, distance: np.ndarray, top_n: int = 20) -> pd.DataFrame:
        aggregate = distance.mean(axis=0)
        order = np.argsort(aggregate, kind="stable")[:top_n]
        rows = []
        for rank, index in enumerate(order, start=1):
            index = int(index)
            rows.append({
                "排名": rank,
                "样品编号": self.train_sample_id[index],
                "样品名": self.train_sample_name[index],
                "类别": self.train_label[index],
                "类别中文": CLASS_CN.get(self.train_label[index], ""),
                "地区": self.train_province[index],
                "文献": self.train_reference[index],
                "来源库": self.train_database[index],
                "相似指数": self._similarity(aggregate[index]),
                "log10_RMS距离": float(aggregate[index]),
                "来源链接": self.train_source_url[index],
            })
        return pd.DataFrame(rows)

    def _group_ranking(
        self, distance: np.ndarray, values: np.ndarray, top_n: int = 20
    ) -> pd.DataFrame:
        series = pd.Series(values, dtype=str)
        groups = [(name, np.asarray(list(idx), dtype=int))
                  for name, idx in series.groupby(series, sort=True).groups.items()]
        group_distance = np.empty((distance.shape[0], len(groups)))
        nearest_index = np.empty(len(groups), dtype=int)
        group_sizes = np.empty(len(groups), dtype=int)
        for g, (_, indices) in enumerate(groups):
            block = distance[:, indices]
            k = min(3, len(indices))
            group_distance[:, g] = (
                block[:, 0] if k == 1
                else np.partition(block, kth=k - 1, axis=1)[:, :k].mean(axis=1)
            )
            nearest_index[g] = indices[int(np.argmin(block.mean(axis=0)))]
            group_sizes[g] = len(indices)
        aggregate = group_distance.mean(axis=0)
        order = np.argsort(aggregate, kind="stable")[:top_n]
        rows = []
        for rank, g in enumerate(order, start=1):
            g = int(g)
            evidence = int(nearest_index[g])
            rows.append({
                "排名": rank,
                "名称": groups[g][0],
                "相似指数": self._similarity(aggregate[g]),
                "log10_RMS距离": float(aggregate[g]),
                "检索库样品数": int(group_sizes[g]),
                "最近证据样品": self.train_sample_name[evidence],
                "最近证据类别": self.train_label[evidence],
                "来源链接": self.train_source_url[evidence],
            })
        return pd.DataFrame(rows)

    def analyze(self, input_table: pd.DataFrame) -> LimiXResult:
        table = canonicalize_input(input_table)
        raw = table[E36].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            query = np.where(
                np.isfinite(raw), np.log10(np.where(np.isfinite(raw), raw, 1.0)), np.nan
            ).astype(np.float32)
        started = time.perf_counter()
        probability = self.predict_log10(query)
        elapsed = time.perf_counter() - started

        labels = np.asarray(CLASSES, dtype=str)[probability.argmax(axis=1)]
        descending = np.argsort(-probability, axis=1)
        second = np.asarray(CLASSES, dtype=str)[descending[:, 1]]
        entropy = -np.sum(
            probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1
        ) / math.log(len(CLASSES))
        measured = np.isfinite(raw).sum(axis=1)

        probability_table = pd.DataFrame(
            {
                "样品编号": table["Sample_ID"].astype(str),
                "预测环境": labels,
                "预测环境中文": [CLASS_CN[label] for label in labels],
                "实测元素数": measured.astype(int),
                "最高概率": probability.max(axis=1),
                "次高环境": second,
                "前两类概率差": probability[np.arange(len(table)), descending[:, 0]]
                - probability[np.arange(len(table)), descending[:, 1]],
                "归一化熵": entropy,
            }
        )
        for class_index, class_name in enumerate(CLASSES):
            probability_table[f"P_{class_name}"] = probability[:, class_index]

        distance = self._distance_matrix(raw)
        nearest_samples = self._sample_ranking(distance)
        region_ranking = self._group_ranking(distance, self.train_province)
        reference_ranking = self._group_ranking(distance, self.train_reference)

        audit = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "application": "LimiX-2M构造环境判别工具-v1.0",
            "model": "LimiX-2M (in-context, frozen reference context)",
            "device": self.device,
            "context_budget": int(len(self.context_y)),
            "inference_seconds": float(elapsed),
            "input_rows": int(len(table)),
            "minimum_coverage": "all 10 major oxides + at least 8 trace elements (>= 18 measured)",
            "probability_rows_sum_to_one": bool(
                np.allclose(probability.sum(axis=1), 1.0, atol=1e-9)
            ),
            "classes": CLASSES,
            "context_source_sha256": self.context_meta["source_data_sha256"],
            "checkpoint_sha256": WEIGHTS_SHA,
            "retrieval_library_rows": int(len(self.train_sample_id)),
            "retrieval_library_sha256": self.index_sha256,
            "similarity_definition": "100 * 10^(-mean per-query masked log10-RMS distance over measured elements)",
            "similarity_scope": "chemical-space retrieval only; not probability of geographic or bibliographic origin",
            "scope": "research software; probabilities should be interpreted with sample context",
        }
        return LimiXResult(
            input_table=table,
            probabilities=probability_table,
            nearest_samples=nearest_samples,
            region_ranking=region_ranking,
            reference_ranking=reference_ranking,
            audit=audit,
        )


def export_result(result: LimiXResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.probabilities.to_excel(writer, sheet_name="九分类概率", index=False)
        result.nearest_samples.to_excel(writer, sheet_name="相似样品Top20", index=False)
        result.region_ranking.to_excel(writer, sheet_name="地区相似排名", index=False)
        result.reference_ranking.to_excel(writer, sheet_name="文献相似排名", index=False)
        result.input_table.to_excel(writer, sheet_name="规范化输入", index=False)
        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells[:100]]
                width = min(max(max(map(len, values), default=8) + 2, 10), 52)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
        sheet = workbook["九分类概率"]
        headers = {cell.value: cell.column for cell in sheet[1]}
        for name in ["最高概率", "前两类概率差", *[f"P_{c}" for c in CLASSES]]:
            if name in headers:
                for row in range(2, sheet.max_row + 1):
                    sheet.cell(row, headers[name]).number_format = "0.00%"
    output_path.with_suffix(".audit.json").write_text(
        json.dumps(result.audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def default_output_path() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"LimiX2M未知样品分析_{stamp}.xlsx"


def write_blank_template(path: str | Path) -> Path:
    path = Path(path)
    columns = ["Sample_ID", "Latitude", "Longitude", *E36]
    pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
    return path
