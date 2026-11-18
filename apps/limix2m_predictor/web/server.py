"""Web app for the LimiX-2M basalt tectonic-setting classifier (multi-user).

Reuses the frozen desktop inference core (same model, same retrieval
library, same validation). Configuration via environment variables
(see .env.example): host/port, shared access token, result TTL, data dir.

Concurrency: inference is serialized with a lock (engine is not
thread-safe); FastAPI runs sync endpoints in a threadpool so other
requests (pages, exports) are not blocked during analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from limix_core import (  # noqa: E402
    ASSET_DIR,
    LimiX2MEngine,
    LimiXResult,
    canonicalize_input,
    export_result,
    write_blank_template,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
WEB_DIR = Path(__file__).resolve().parent

CONFIG = {
    "host": os.environ.get("LIMIX_HOST", "127.0.0.1"),
    "port": int(os.environ.get("LIMIX_PORT", "7100")),
    "access_token": os.environ.get("LIMIX_ACCESS_TOKEN", ""),
    "result_ttl_hours": float(os.environ.get("LIMIX_RESULT_TTL_HOURS", "24")),
    "data_dir": Path(os.environ.get("LIMIX_DATA_DIR", str(WEB_DIR / "web_data"))),
}
RESULTS_DIR = CONFIG["data_dir"] / "results"
TEMPLATE_PATH = CONFIG["data_dir"] / "template.csv"

app = FastAPI(title="LimiX-2M 构造环境判别工具")
engine: LimiX2MEngine | None = None
engine_error: str | None = None
analyze_lock = threading.Lock()
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
def _startup() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        get_engine()
    except HTTPException as error:
        print(f"[startup] engine preload failed: {error.detail}", flush=True)


def require_token(x_access_token: str | None = Header(default=None)) -> None:
    expected = CONFIG["access_token"]
    if expected:
        provided_hash = hashlib.sha256((x_access_token or "").encode()).hexdigest()
        if provided_hash != hashlib.sha256(expected.encode()).hexdigest():
            raise HTTPException(status_code=401, detail="需要有效的访问口令。")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/template")
def template(_token: None = Depends(require_token)) -> FileResponse:
    write_blank_template(TEMPLATE_PATH)
    return FileResponse(TEMPLATE_PATH, media_type="text/csv",
                        filename="LimiX2M未知样品输入模板.csv")


@app.get("/api/example")
def example(_token: None = Depends(require_token)) -> JSONResponse:
    frame = pd.read_csv(ASSET_DIR / "example_input.csv")
    return JSONResponse({"rows": frame.where(pd.notna(frame), None).to_dict("records")})


def _json_safe(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_float_dtype(clean[column]):
            clean[column] = clean[column].astype(float).round(6)
    return clean.where(pd.notna(clean), None).to_dict("records")


def _cleanup_results() -> None:
    cutoff = time.time() - CONFIG["result_ttl_hours"] * 3600
    for folder in RESULTS_DIR.glob("*"):
        try:
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


def _persist(payload: dict) -> None:
    folder = RESULTS_DIR / payload["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_payload(result_id: str) -> dict | None:
    path = RESULTS_DIR / result_id / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _run_analysis(table: pd.DataFrame) -> dict:
    started = time.perf_counter()
    with analyze_lock:  # engine is not thread-safe; serialize inference
        result = get_engine().analyze(table)
    result.audit["result_id"] = uuid.uuid4().hex[:12]
    result.audit["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    results[result.audit["result_id"]] = result
    audit = {key: value for key, value in result.audit.items() if key != "classes"}
    payload = {
        "id": result.audit["result_id"],
        "probabilities": _json_safe(result.probabilities),
        "nearest_samples": _json_safe(result.nearest_samples),
        "region_ranking": _json_safe(result.region_ranking),
        "reference_ranking": _json_safe(result.reference_ranking),
        "input_table": _json_safe(result.input_table),
        "audit": audit,
    }
    try:
        _persist(payload)
        _cleanup_results()
    except OSError as error:
        print(f"[persist] failed: {error}", flush=True)
    return payload


@app.post("/api/analyze/file")
def analyze_file(file: UploadFile, _token: None = Depends(require_token)) -> JSONResponse:
    raw = file.file.read()
    suffix = Path(file.filename or "input.csv").suffix.lower()
    try:
        if suffix == ".xlsx":
            frame = pd.read_excel(pd.io.common.BytesIO(raw))
        elif suffix in {".csv", ".txt"}:
            last_error: Exception | None = None
            frame = None
            for encoding in ("utf-8-sig", "utf-8", "gb18030"):
                try:
                    frame = pd.read_csv(pd.io.common.BytesIO(raw), encoding=encoding)
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
def analyze_json(payload: dict, _token: None = Depends(require_token)) -> JSONResponse:
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
def export(result_id: str, _token: None = Depends(require_token)) -> FileResponse:
    result = results.get(result_id)
    if result is None:
        payload = _load_payload(result_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="结果不存在或已过期，请重新分析。")
        result = LimiXResult(
            input_table=pd.DataFrame(payload["input_table"]),
            probabilities=pd.DataFrame(payload["probabilities"]),
            nearest_samples=pd.DataFrame(payload["nearest_samples"]),
            region_ranking=pd.DataFrame(payload["region_ranking"]),
            reference_ranking=pd.DataFrame(payload["reference_ranking"]),
            audit=payload["audit"],
        )
    export_path = RESULTS_DIR / result_id / "export.xlsx"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_result(result, export_path)
    return FileResponse(
        export_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="LimiX2M未知样品分析.xlsx")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "engine_loaded": engine is not None,
            "auth_required": bool(CONFIG["access_token"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=CONFIG["host"])
    parser.add_argument("--port", type=int, default=CONFIG["port"])
    args = parser.parse_args()
    print(f"config: host={args.host} port={args.port} auth={'on' if CONFIG['access_token'] else 'off'} "
          f"ttl={CONFIG['result_ttl_hours']}h data_dir={CONFIG['data_dir']}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
