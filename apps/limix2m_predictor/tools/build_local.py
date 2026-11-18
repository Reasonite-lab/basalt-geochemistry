"""Build the local LimiX-2M desktop app (onedir) and verify the executable.

Local build only: bundles torch with CUDA, the official LimiX-2M checkpoint,
the frozen reference context and the upstream inference code. Not for upload.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[2]
DIST_PARENT = Path(r"D:\AI地学\delivery")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    build_root = ROOT / "build"
    build_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="limix2m-app-", dir=build_root))
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--windowed", "--name", "LimiX2MPredictor",
        "--paths", str(APP / "assets/limix_upstream"),
        "--paths", str(APP / "assets/limix_dependencies"),
        "--add-data", f"{APP / 'assets'};assets",
        "--collect-submodules", "inference", "--collect-submodules", "utils",
        "--collect-submodules", "model", "--collect-submodules", "kditransform",
        "--collect-submodules", "sklearn", "--collect-submodules", "tqdm",
        "--collect-submodules", "einops",
        "--collect-all", "torch", "--hidden-import", "openpyxl",
        "--distpath", str(work), "--workpath", str(work / "pyinstaller"),
        "--specpath", str(work),
        str(APP / "limix_gui.pyw"),
    ], check=True, cwd=APP)

    stage = work / "LimiX2MPredictor"
    verification = work / "verification"
    verification.mkdir()
    executable = stage / "LimiX2MPredictor.exe"
    started = time.perf_counter()
    for option, filename in (("--self-test", "portable_self_test_report.json"),
                             ("--gui-smoke-test", "gui_smoke_report.json")):
        subprocess.run([str(executable), option, "--output-dir", str(verification)],
                       check=True, timeout=900, cwd=work)
        report = json.loads((verification / filename).read_text(encoding="utf-8"))
        if report["status"] != "PASS" or not report["frozen_executable"]:
            raise RuntimeError(f"Executable verification failed: {filename}")
    print(f"verification done in {time.perf_counter() - started:.0f}s")

    for source, name in ((APP / "README.md", "使用说明.md"),
                         (APP / "assets/limix_upstream/LICENSE.txt", "LimiX_LICENSE.txt")):
        if source.exists():
            shutil.copyfile(source, stage / name)
    release_info = {
        "application": "LimiX-2M 构造环境判别工具",
        "version": "1.1.1",
        "model": "LimiX-2M (stable-ai, Apache-2.0)",
        "platform": "Windows x64",
        "device": "CUDA GPU (falls back to CPU)",
        "executable_sha256": digest(executable),
    }
    (stage / "release-info.json").write_text(
        json.dumps(release_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    target = DIST_PARENT / "LimiX2MPredictor_v1.1.1"
    if target.exists():
        shutil.rmtree(target)
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
