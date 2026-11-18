"""Local web demo for the LimiX-2M basalt tectonic-setting classifier.

Serves a single-page UI and a JSON API that reuses the frozen desktop
inference core (same model, same retrieval library, same validation).
"""
from __future__ import annotations

import argparse
import io
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from limix_core import (  # noqa: E402
    ASSET_DIR,
    LimiX2MEngine,
    LimiXResult,
    canonicalize_input,
    export_result,
    read_input_table,
    write_blank_template,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATE_PATH = Path(tempfile.gettempdir()) / "limix2m_web_template.csv"

app = FastAPI(title="LimiX-2M 构造环境判别工具 Web Demo")
engine: LimiX2MEngine | None = None
engine_error: str | None = None
results: dict[str, LimiXResult] = {}


def get_engine() -> LimiX2MEngine:
    global engine, engine_error
    if engine is None:
        if engine_error:
            raise HTTPException(status_code=503, detail=f"模型加载失败：{engine_error}")
        try:
            engine = LimiX2MEngine()
        except Exception as error:  # noqa: BLE001
            engine_error = str(error)
            raise HTTPException(status_code=503, detail=f"模型加载失败：{error}") from error
    return engine


@app.on_event("startup")
def _preload_engine() -> None:
    try:
        get_engine()
    except HTTPException:
        pass


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/template")
def template() -> FileResponse:
    write_blank_template(TEMPLATE_PATH)
    return FileResponse(TEMPLATE_PATH, media_type="text/csv",
                        filename="LimiX2M未知样品输入模板.csv")


@app.get("/api/example")
def example() -> JSONResponse:
    frame = pd.read_csv(ASSET_DIR / "example_input.csv")
    return JSONResponse({"rows": frame.where(pd.notna(frame), None).to_dict("records")})


def _json_safe(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_float_dtype(clean[column]):
            clean[column] = clean[column].astype(float).round(6)
    return clean.where(pd.notna(clean), None).to_dict("records")


def _package(result: LimiXResult) -> dict:
    audit = dict(result.audit)
    return {
        "id": audit.get("result_id"),
        "probabilities": _json_safe(result.probabilities),
        "nearest_samples": _json_safe(result.nearest_samples),
        "region_ranking": _json_safe(result.region_ranking),
        "reference_ranking": _json_safe(result.reference_ranking),
        "audit": {key: value for key, value in audit.items()
                  if key not in {"classes"}},
    }


def _run_analysis(table: pd.DataFrame) -> dict:
    started = time.perf_counter()
    result = get_engine().analyze(table)
    result.audit["result_id"] = uuid.uuid4().hex[:12]
    result.audit["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    results[result.audit["result_id"]] = result
    return _package(result)


@app.post("/api/analyze/file")
async def analyze_file(file: UploadFile) -> JSONResponse:
    raw = await file.read()
    suffix = Path(file.filename or "input.csv").suffix.lower()
    try:
        if suffix == ".xlsx":
            frame = pd.read_excel(io.BytesIO(raw))
        elif suffix in {".csv", ".txt"}:
            last_error: Exception | None = None
            frame = None
            for encoding in ("utf-8-sig", "utf-8", "gb18030"):
                try:
                    frame = pd.read_csv(io.BytesIO(raw), encoding=encoding)
                    break
                except UnicodeDecodeError as error:
                    last_error = error
            if frame is None:
                raise ValueError(f"无法解码 CSV（尝试了 utf-8/gb18030）：{last_error}")
        else:
            raise ValueError("只支持 CSV、TXT 或 XLSX 文件。")
        return JSONResponse(_run_analysis(frame))
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/analyze/json")
def analyze_json(payload: dict) -> JSONResponse:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="请求体需要非空的 rows 数组。")
    try:
        table = canonicalize_input(pd.DataFrame(rows))
        return JSONResponse(_run_analysis(table))
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/export/{result_id}")
def export(result_id: str) -> FileResponse:
    result = results.get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="结果不存在或已过期，请重新分析。")
    path = Path(tempfile.gettempdir()) / f"limix2m_web_export_{result_id}.xlsx"
    export_result(result, path)
    return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename="LimiX2M未知样品分析.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7100)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
