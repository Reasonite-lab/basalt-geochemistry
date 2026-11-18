"""Build the CPU-only LimiX-2M desktop app (onedir) and verify the executable.

Uses the CPU venv's PyInstaller, so the bundled torch has no CUDA runtime
(portable size ~1 GB instead of ~8 GB; works on any Windows x64 machine).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]
DIST_PARENT = Path(r"D:\AI地学\delivery")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_licenses(stage: Path) -> None:
    target = stage / "third_party_licenses"
    target.mkdir()
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "unknown")
        versions[name] = dist.version
        for relative in dist.files or []:
            if ".." in relative.parts:
                continue
            if any(term in str(relative).lower() for term in ("license", "copying", "notice")):
                source = Path(dist.locate_file(relative))
                if source.is_file():
                    destination = target / name / str(relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    for relative in ("LICENSE.txt", "tcl/tcl8.6/license.terms", "tcl/tk8.6/license.terms"):
        source = Path(sys.base_prefix) / relative
        if source.exists():
            destination = target / "Python-Tcl-Tk" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    (target / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")


def main() -> None:
    import torch
    if sys.platform != "win32" or torch.version.cuda is not None:
        raise SystemExit("Build on Windows with the CPU-only torch environment.")
    build_root = ROOT / "build"
    build_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="limix2m-cpu-app-", dir=build_root))
    target = DIST_PARENT / "LimiX2MPredictor_CPU_v1.1.1"
    if target.exists():
        raise FileExistsError(f"Delivery already exists: {target}")
    assets = work / "assets"
    shutil.copytree(APP / "assets", assets, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", "triton", "triton_windows-*", "nvtx", "nvtx-*"
    ))
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--windowed", "--name", "LimiX2MPredictor_CPU",
        "--icon", str(APP / "assets/app_icon.ico"),
        "--paths", str(assets / "limix_upstream"),
        "--paths", str(assets / "limix_dependencies"),
        "--add-data", f"{assets};assets",
        "--exclude-module", "triton", "--exclude-module", "nvtx",
        "--collect-submodules", "inference", "--collect-submodules", "utils",
        "--collect-submodules", "model", "--collect-submodules", "kditransform",
        "--collect-submodules", "sklearn", "--collect-submodules", "tqdm",
        "--collect-submodules", "einops", "--collect-submodules", "numba",
        "--collect-submodules", "llvmlite",
        "--collect-all", "torch", "--hidden-import", "openpyxl",
        "--distpath", str(work), "--workpath", str(work / "pyinstaller"),
        "--specpath", str(work),
        str(APP / "limix_gui.pyw"),
    ], check=True, cwd=APP)

    stage = work / "LimiX2MPredictor_CPU"
    verification = work / "verification"
    verification.mkdir()
    executable = stage / "LimiX2MPredictor_CPU.exe"
    started = time.perf_counter()
    for option, filename in (("--self-test", "portable_self_test_report.json"),
                             ("--gui-smoke-test", "gui_smoke_report.json")):
        subprocess.run([str(executable), option, "--output-dir", str(verification)],
                       check=True, timeout=1800, cwd=work)
        report = json.loads((verification / filename).read_text(encoding="utf-8"))
        if report["status"] != "PASS" or not report["frozen_executable"]:
            raise RuntimeError(f"Executable verification failed: {filename}")
    print(f"verification done in {time.perf_counter() - started:.0f}s")

    for source, name in ((APP / "README.md", "使用说明.md"),
                         (ROOT / "LICENSE", "LICENSE"),
                         (APP / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
                         (APP / "assets/limix_upstream/LICENSE.txt", "LimiX_LICENSE.txt")):
        if source.exists():
            shutil.copyfile(source, stage / name)
    # Retain the established, tested input examples without copying previous user outputs.
    for name in ("输入模板.csv", "批量示例_63样品.csv", "缺测示例_36样品_D22.csv"):
        shutil.copy2(DIST_PARENT / "LimiX2MPredictor_v1.1.1" / name, stage / name)
    shutil.copytree(verification, stage / "verification")
    copy_licenses(stage)
    release_info = {
        "application": "LimiX-2M 构造环境判别工具（CPU 版）",
        "version": "1.1.1",
        "model": "LimiX-2M (stable-ai, Apache-2.0)",
        "platform": "Windows x64 (CPU, no GPU required)",
        "executable_sha256": digest(executable),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch_version": torch.__version__,
        "checkpoint_sha256": digest(APP / "assets/LimiX-2M.ckpt"),
        "context_sha256": digest(APP / "assets/context_v1.npz"),
        "retrieval_library_sha256": digest(APP / "assets/retrieval_library_v1.npz"),
        "executable_self_test": "PASS",
        "executable_gui_smoke_test": "PASS",
    }
    (stage / "release-info.json").write_text(
        json.dumps(release_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    shutil.copytree(stage, target)
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(json.dumps({
        "delivered": str(target),
        "size_bytes": size,
        "size_gb": round(size / (1 << 30), 2),
        "executable_sha256": release_info["executable_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
