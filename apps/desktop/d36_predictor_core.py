"""Core inference and chemical-retrieval services for the D36 desktop app.

The classifier and the retrieval library are deliberately separated:

* probabilities come from the frozen three-member D36_74 bundle;
* the optional coordinate prior is only used when the caller explicitly asks;
* Province and Reference rankings are descriptive chemical-space retrievals.

Similarity is not a probability of origin.  It is a monotonic presentation of
the D36 log10-RMS distance: ``100 * 10 ** (-distance)``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import Booster


# PyInstaller unpacks bundled read-only resources under ``sys._MEIPASS``.
# User-created outputs must instead stay beside the portable executable.
SOURCE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_DIR))
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_DIR
ASSET_DIR = RESOURCE_DIR / "assets"
BUNDLE_DIR = ASSET_DIR / "geosingle_bundle"
RETRIEVAL_INDEX = ASSET_DIR / "retrieval_library_v1.npz"
RETRIEVAL_METADATA = ASSET_DIR / "retrieval_library_v1.json"
OUTPUT_DIR = APP_DIR / "outputs"

CLASSES = ["BAB", "CA", "CFB", "CR", "IA", "IOA", "MOR", "OI", "OP"]
CLASS_CN = {
    "BAB": "弧后盆地",
    "CA": "大陆弧",
    "CFB": "大陆溢流玄武岩",
    "CR": "大陆裂谷",
    "IA": "岛弧",
    "IOA": "洋内弧",
    "MOR": "洋中脊",
    "OI": "洋岛",
    "OP": "大洋高原",
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
EPS = 1e-9
EARTH_RADIUS_KM = 6371.0088
MAX_INPUT_ROWS = 500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), EPS, np.inf)
    return probability / probability.sum(axis=1, keepdims=True)


def helmert_basis(dimension: int = 10) -> np.ndarray:
    basis = np.zeros((dimension - 1, dimension), float)
    for index in range(1, dimension):
        scale = math.sqrt(index * (index + 1))
        basis[index - 1, :index] = 1.0 / scale
        basis[index - 1, index] = -index / scale
    return basis


def build_d36_74(raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(E36):
        raise ValueError(f"D36矩阵必须为 n×{len(E36)}。")
    if not np.isfinite(raw).all() or not (raw > 0).all():
        raise ValueError("36个元素必须全部为有限正数。")
    log36 = np.log10(raw)
    positions = {element: index for index, element in enumerate(E36)}
    parts: list[np.ndarray] = [log36, log36[:, :10] @ helmert_basis(10).T]
    names = ["log_" + element for element in E36]
    names += [f"ilr_major_{index + 1}" for index in range(9)]
    for left, right in RATIO_PAIRS:
        parts.append(
            (log36[:, positions[left]] - log36[:, positions[right]])[:, None]
        )
        names.append(f"logratio_{left}_over_{right}")
    for numerator, left, right, name in [
        ("EU(PPM)", "SM(PPM)", "GD(PPM)", "Eu_anomaly"),
        ("CE(PPM)", "LA(PPM)", "PR(PPM)", "Ce_anomaly"),
    ]:
        anomaly = log36[:, positions[numerator]] - 0.5 * (
            log36[:, positions[left]] + log36[:, positions[right]]
        )
        parts.append(anomaly[:, None])
        names.append(name)
    features = np.column_stack(parts).astype(np.float32)
    if features.shape[1] != 74:
        raise RuntimeError("D36_74特征维数错误。")
    return features, names


def spherical_xyz(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)
    cos_lat = np.cos(lat)
    return np.column_stack(
        [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)]
    )


def _column_key(name: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    by_key = {_column_key(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if _column_key(candidate) in by_key:
            return by_key[_column_key(candidate)]
    return None


def canonicalize_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical input table and raise an actionable validation error."""
    if frame.empty:
        raise ValueError("输入表没有数据行。")
    if len(frame) > MAX_INPUT_ROWS:
        raise ValueError(
            f"一次最多分析{MAX_INPUT_ROWS}个样品；当前为{len(frame)}个。请分批导入。"
        )

    source = frame.copy()
    output = pd.DataFrame(index=source.index)
    id_column = _find_column(
        source, ["Physical_sample_ID", "Sample_ID", "Sample_name", "样品编号", "样品名"]
    )
    if id_column is None:
        output["Sample_ID"] = [f"unknown_{index + 1:04d}" for index in range(len(source))]
    else:
        identifiers = source[id_column].astype(str).str.strip()
        blank = identifiers.eq("") | identifiers.str.lower().isin(["nan", "none"])
        identifiers.loc[blank] = [
            f"unknown_{index + 1:04d}" for index in np.flatnonzero(blank)
        ]
        output["Sample_ID"] = identifiers.to_numpy()

    missing: list[str] = []
    for canonical in E36:
        short = canonical.split("(")[0]
        column = _find_column(source, [canonical, short])
        if column is None:
            missing.append(canonical)
            continue
        output[canonical] = pd.to_numeric(source[column], errors="coerce")
    if missing:
        raise ValueError("输入表缺少D36列：\n" + "、".join(missing))

    invalid = ~np.isfinite(output[E36].to_numpy(float)) | (
        output[E36].to_numpy(float) <= 0
    )
    if invalid.any():
        row_index, column_index = np.argwhere(invalid)[0]
        sample = output.iloc[int(row_index)]["Sample_ID"]
        element = E36[int(column_index)]
        value = output.iloc[int(row_index)][element]
        raise ValueError(
            f"样品 {sample} 的 {element}={value!r} 无效；D36要求36列全部为有限正数。"
        )

    lat_column = _find_column(source, ["Latitude", "Lat", "纬度"])
    lon_column = _find_column(source, ["Longitude", "Lon", "Lng", "经度"])
    output["Latitude"] = (
        pd.to_numeric(source[lat_column], errors="coerce")
        if lat_column is not None
        else np.nan
    )
    output["Longitude"] = (
        pd.to_numeric(source[lon_column], errors="coerce")
        if lon_column is not None
        else np.nan
    )
    latitude = output["Latitude"].to_numpy(float)
    longitude = output["Longitude"].to_numpy(float)
    if np.any(np.isfinite(latitude) & ((latitude < -90) | (latitude > 90))):
        raise ValueError("纬度必须位于[-90, 90]。")
    if np.any(np.isfinite(longitude) & ((longitude < -180) | (longitude > 180))):
        raise ValueError("经度必须位于[-180, 180]。")
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
class AnalysisResult:
    input_table: pd.DataFrame
    probabilities: pd.DataFrame
    region_ranking: pd.DataFrame
    reference_ranking: pd.DataFrame
    region_by_sample: pd.DataFrame
    reference_by_sample: pd.DataFrame
    audit: dict[str, Any]


class D36Engine:
    """Load the frozen model once and serve repeated GUI analyses."""

    def __init__(
        self,
        bundle_dir: str | Path = BUNDLE_DIR,
        retrieval_index: str | Path = RETRIEVAL_INDEX,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.retrieval_index_path = Path(retrieval_index)
        self.models, self.spatial_library, self.bundle = self._load_bundle()
        self._load_retrieval_library()

    def _load_bundle(self) -> tuple[list[Booster], dict[str, np.ndarray], dict[str, Any]]:
        metadata_path = self.bundle_dir / "bundle.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"模型包不存在：{metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["classes"] != CLASSES or metadata["elements"] != E36:
            raise ValueError("冻结模型包的类别或元素顺序不一致。")
        if canonical_hash(metadata["bundle_content_manifest"]) != metadata["bundle_content_sha256"]:
            raise ValueError("冻结模型包内容清单哈希不一致。")
        manifest = metadata["bundle_content_manifest"]
        for filename in metadata["model_files"]:
            if sha256(self.bundle_dir / filename) != manifest["model_sha256"][filename]:
                raise ValueError(f"模型文件校验失败：{filename}")
        library_path = self.bundle_dir / metadata["spatial_library_file"]
        if sha256(library_path) != manifest["spatial_library_sha256"]:
            raise ValueError("空间先验库校验失败。")

        models: list[Booster] = []
        with tempfile.TemporaryDirectory(prefix="d36_app_model_") as temporary_dir:
            temporary = Path(temporary_dir)
            for index, filename in enumerate(metadata["model_files"]):
                local_path = temporary / f"member_{index}.txt"
                shutil.copyfile(self.bundle_dir / filename, local_path)
                models.append(Booster(model_file=str(local_path)))
        with np.load(library_path, allow_pickle=False) as saved:
            spatial_library = {
                "names": saved["names"].astype(str),
                "xyz": saved["xyz"].astype(float),
                "distributions": saved["distributions"].astype(float),
            }
        return models, spatial_library, metadata

    def _load_retrieval_library(self) -> None:
        if not self.retrieval_index_path.exists():
            raise FileNotFoundError(f"化学检索库不存在：{self.retrieval_index_path}")
        with np.load(self.retrieval_index_path, allow_pickle=False) as saved:
            self.train_log36 = saved["log36"].astype(np.float32)
            self.train_sample_id = saved["physical_id"].astype(str)
            self.train_sample_name = saved["sample_name"].astype(str)
            self.train_province = saved["province"].astype(str)
            self.train_reference = saved["reference"].astype(str)
            self.train_label = saved["label"].astype(str)
            self.train_database = saved["database"].astype(str)
            self.train_source_url = saved["source_url"].astype(str)
        if self.train_log36.shape != (len(self.train_sample_id), len(E36)):
            raise ValueError("化学检索库维数错误。")
        self.index_sha256 = sha256(self.retrieval_index_path)
        self.region_groups = self._build_groups(self.train_province)
        self.reference_groups = self._build_groups(self.train_reference)

    @staticmethod
    def _build_groups(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
        series = pd.Series(values, dtype=str)
        groups = []
        for name, indices in series.groupby(series, sort=True).groups.items():
            groups.append((str(name), np.asarray(list(indices), dtype=int)))
        return groups

    def _predict_base(self, chemistry: np.ndarray) -> np.ndarray:
        features, feature_names = build_d36_74(chemistry)
        if feature_names != self.bundle["feature_names"]:
            raise RuntimeError("D36_74特征名与冻结模型不一致。")
        members = []
        for model in self.models:
            probability = np.asarray(model.predict(features), dtype=float)
            members.append(normalize_probability(probability))
        return normalize_probability(np.mean(members, axis=0))

    def _apply_spatial_prior(
        self,
        base: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & (latitude >= -90)
            & (latitude <= 90)
            & (longitude >= -180)
            & (longitude <= 180)
        )
        library_xyz = self.spatial_library["xyz"]
        distributions = self.spatial_library["distributions"]
        distances = np.full((len(base), len(library_xyz)), np.inf, dtype=float)
        if valid.any():
            query_xyz = spherical_xyz(latitude[valid], longitude[valid])
            cosine = np.clip(query_xyz @ library_xyz.T, -1.0, 1.0)
            distances[valid] = np.arccos(cosine) * EARTH_RADIUS_KM
        order = np.argsort(distances, axis=1, kind="stable")
        ordered_distance = np.take_along_axis(distances, order, axis=1)
        usable = valid & np.isfinite(ordered_distance[:, 0])
        members = []
        for setting in self.bundle["parameter_sets"]:
            k = min(int(setting["k_provinces"]), order.shape[1])
            take = order[:, :k]
            distance = ordered_distance[:, :k]
            weight = np.exp(-np.minimum(distance, 1e6) / float(setting["bandwidth_km"]))
            weight[~np.isfinite(distance)] = 0.0
            numerator = (distributions[take] * weight[:, :, None]).sum(axis=1)
            denominator = weight.sum(axis=1, keepdims=True)
            prior = np.full_like(base, 1.0 / len(CLASSES))
            current = usable & (denominator[:, 0] > 0)
            prior[current] = numerator[current] / denominator[current]
            prior = normalize_probability(prior)
            alpha = float(setting["alpha"])
            blended = base.copy()
            blended[current] = normalize_probability(
                np.exp(
                    (1.0 - alpha) * np.log(np.clip(base[current], 1e-12, 1.0))
                    + alpha * np.log(np.clip(prior[current], 1e-12, 1.0))
                )
            )
            members.append(blended)
        probability = normalize_probability(np.mean(members, axis=0))
        probability[~usable] = base[~usable]
        return probability, usable

    def _distance_matrix(self, chemistry: np.ndarray) -> np.ndarray:
        query = np.log10(np.asarray(chemistry, dtype=float)).astype(np.float64)
        library = self.train_log36.astype(np.float64, copy=False)
        squared = (
            np.sum(query * query, axis=1)[:, None]
            + np.sum(library * library, axis=1)[None, :]
            - 2.0 * query @ library.T
        )
        np.maximum(squared, 0.0, out=squared)
        return np.sqrt(squared) / math.sqrt(len(E36))

    def _group_rankings(
        self,
        distance: np.ndarray,
        groups: list[tuple[str, np.ndarray]],
        group_kind: str,
        sample_ids: np.ndarray,
        aggregate_top_n: int | None,
        per_sample_top_n: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        n_query = distance.shape[0]
        n_groups = len(groups)
        group_distance = np.empty((n_query, n_groups), dtype=np.float32)
        nearest_index = np.empty((n_query, n_groups), dtype=np.int32)
        group_sizes = np.empty(n_groups, dtype=np.int32)
        used_neighbors = np.empty(n_groups, dtype=np.int8)

        for group_index, (_, indices) in enumerate(groups):
            block = distance[:, indices]
            k = min(3, len(indices))
            if k == 1:
                nearest_mean = block[:, 0]
            else:
                nearest_mean = np.partition(block, kth=k - 1, axis=1)[:, :k].mean(axis=1)
            local_nearest = np.argmin(block, axis=1)
            group_distance[:, group_index] = nearest_mean.astype(np.float32)
            nearest_index[:, group_index] = indices[local_nearest]
            group_sizes[group_index] = len(indices)
            used_neighbors[group_index] = k

        aggregate_distance = group_distance.mean(axis=0)
        full_order = np.argsort(aggregate_distance, kind="stable")
        aggregate_order = full_order if aggregate_top_n is None else full_order[:aggregate_top_n]
        aggregate_rows = []
        for rank, group_index in enumerate(aggregate_order, start=1):
            name, _ = groups[int(group_index)]
            query_index = int(np.argmin(group_distance[:, group_index]))
            evidence_index = int(nearest_index[query_index, group_index])
            rms = float(aggregate_distance[group_index])
            row: dict[str, Any] = {
                "排名": rank,
                "地区" if group_kind == "region" else "文献": name,
                "相似指数": 100.0 * (10.0 ** (-rms)),
                "log10_RMS距离": rms,
                "典型倍数差": 10.0 ** rms,
                "检索库样品数": int(group_sizes[group_index]),
                "每个未知样品采用近邻数": int(used_neighbors[group_index]),
                "最近证据样品": self.train_sample_name[evidence_index],
                "最近证据类别": self.train_label[evidence_index],
                "最近证据类别中文": CLASS_CN.get(self.train_label[evidence_index], ""),
            }
            if group_kind == "region":
                row["最近证据文献"] = self.train_reference[evidence_index]
                row["来源链接"] = self.train_source_url[evidence_index]
            else:
                row["最近证据地区"] = self.train_province[evidence_index]
                row["来源链接"] = self.train_source_url[evidence_index]
            aggregate_rows.append(row)

        per_sample_rows = []
        for query_index, sample_id in enumerate(sample_ids):
            order = np.argsort(group_distance[query_index], kind="stable")[:per_sample_top_n]
            for rank, group_index in enumerate(order, start=1):
                name, _ = groups[int(group_index)]
                evidence_index = int(nearest_index[query_index, group_index])
                rms = float(group_distance[query_index, group_index])
                row = {
                    "样品编号": str(sample_id),
                    "排名": rank,
                    "地区" if group_kind == "region" else "文献": name,
                    "相似指数": 100.0 * (10.0 ** (-rms)),
                    "log10_RMS距离": rms,
                    "典型倍数差": 10.0 ** rms,
                    "检索库样品数": int(group_sizes[group_index]),
                    "最近证据样品": self.train_sample_name[evidence_index],
                    "最近证据类别": self.train_label[evidence_index],
                }
                if group_kind == "region":
                    row["最近证据文献"] = self.train_reference[evidence_index]
                else:
                    row["最近证据地区"] = self.train_province[evidence_index]
                row["来源链接"] = self.train_source_url[evidence_index]
                per_sample_rows.append(row)
        return pd.DataFrame(aggregate_rows), pd.DataFrame(per_sample_rows)

    def analyze(
        self,
        input_table: pd.DataFrame,
        *,
        use_spatial_prior: bool = False,
        aggregate_top_n: int | None = None,
        per_sample_top_n: int = 20,
    ) -> AnalysisResult:
        table = canonicalize_input(input_table)
        chemistry = table[E36].to_numpy(float)
        base = self._predict_base(chemistry)
        latitude = table["Latitude"].to_numpy(float)
        longitude = table["Longitude"].to_numpy(float)
        if use_spatial_prior:
            probability, spatial_used = self._apply_spatial_prior(base, latitude, longitude)
        else:
            probability = base
            spatial_used = np.zeros(len(table), dtype=bool)
        labels = np.asarray(CLASSES, dtype=str)[probability.argmax(axis=1)]
        descending = np.argsort(-probability, axis=1)
        second = np.asarray(CLASSES, dtype=str)[descending[:, 1]]
        entropy = -np.sum(
            probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1
        ) / math.log(len(CLASSES))

        probability_table = pd.DataFrame(
            {
                "样品编号": table["Sample_ID"].astype(str),
                "预测环境": labels,
                "预测环境中文": [CLASS_CN[label] for label in labels],
                "最高概率": probability.max(axis=1),
                "次高环境": second,
                "前两类概率差": probability[np.arange(len(table)), descending[:, 0]]
                - probability[np.arange(len(table)), descending[:, 1]],
                "归一化熵": entropy,
                "推理路径": np.where(
                    spatial_used,
                    "GeoSinglePrior-D36-v1（D36_74+坐标先验）",
                    np.where(
                        use_spatial_prior,
                        "D36_74（三成员；坐标缺失回退）",
                        "D36_74（三成员；纯化学）",
                    ),
                ),
            }
        )
        for class_index, class_name in enumerate(CLASSES):
            probability_table[f"P_{class_name}"] = probability[:, class_index]

        distance = self._distance_matrix(chemistry)
        sample_ids = table["Sample_ID"].astype(str).to_numpy()
        region, region_by_sample = self._group_rankings(
            distance,
            self.region_groups,
            "region",
            sample_ids,
            aggregate_top_n,
            per_sample_top_n,
        )
        reference, reference_by_sample = self._group_rankings(
            distance,
            self.reference_groups,
            "reference",
            sample_ids,
            aggregate_top_n,
            per_sample_top_n,
        )
        audit = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "application": "D36未知样品构造环境判别工具-v1.0",
            "input_rows": int(len(table)),
            "classes": CLASSES,
            "probability_rows_sum_to_one": bool(
                np.allclose(probability.sum(axis=1), 1.0, atol=1e-12)
            ),
            "every_valid_sample_hard_labeled": bool(len(labels) == len(table)),
            "test_rejection": False,
            "model": "frozen three-member D36_74 ensemble",
            "bundle_status": self.bundle["status"],
            "bundle_sha256": self.bundle["bundle_content_sha256"],
            "spatial_prior_requested": bool(use_spatial_prior),
            "spatial_prior_rows": int(spatial_used.sum()),
            "retrieval_library_rows": int(len(self.train_sample_id)),
            "retrieval_library_provinces": int(len(self.region_groups)),
            "retrieval_library_references": int(len(self.reference_groups)),
            "region_ranking_rows": int(len(region)),
            "reference_ranking_rows": int(len(reference)),
            "retrieval_library_sha256": self.index_sha256,
            "similarity_definition": "100 * 10^(-mean per-query nearest-3 log10-RMS36 distance)",
            "similarity_scope": "chemical-space retrieval only; not probability of geographic or bibliographic origin",
        }
        return AnalysisResult(
            input_table=table,
            probabilities=probability_table,
            region_ranking=region,
            reference_ranking=reference,
            region_by_sample=region_by_sample,
            reference_by_sample=reference_by_sample,
            audit=audit,
        )


def export_result(result: AnalysisResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.probabilities.to_excel(writer, sheet_name="九分类概率", index=False)
        result.region_ranking.to_excel(writer, sheet_name="地区相似度_总体", index=False)
        result.reference_ranking.to_excel(writer, sheet_name="文献相似度_总体", index=False)
        result.region_by_sample.to_excel(writer, sheet_name="地区相似度_逐样品", index=False)
        result.reference_by_sample.to_excel(writer, sheet_name="文献相似度_逐样品", index=False)
        result.input_table.to_excel(writer, sheet_name="规范化输入", index=False)
        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells[:100]]
                width = min(max(max(map(len, values), default=8) + 2, 10), 52)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
        probability_sheet = workbook["九分类概率"]
        headers = {cell.value: cell.column for cell in probability_sheet[1]}
        for column_name in ["最高概率", "前两类概率差", *[f"P_{name}" for name in CLASSES]]:
            if column_name in headers:
                for row in range(2, probability_sheet.max_row + 1):
                    probability_sheet.cell(row, headers[column_name]).number_format = "0.00%"
    audit_path = output_path.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(result.audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def default_output_path() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"D36未知样品分析_{stamp}.xlsx"


def write_blank_template(path: str | Path) -> Path:
    path = Path(path)
    columns = ["Sample_ID", "Latitude", "Longitude", *E36]
    pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
    return path

