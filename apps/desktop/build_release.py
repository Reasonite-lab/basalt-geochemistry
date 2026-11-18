"""Build the offline Windows portable release and test the actual executable."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from version import APP_VERSION

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESOURCES = HERE / "resources"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_assets(target: Path) -> dict:
    manifest = json.loads((RESOURCES / "manifest.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    for entry in manifest["archives"]:
        archive = RESOURCES / entry["file"]
        if digest(archive) != entry["sha256"]:
            raise RuntimeError(f"Resource archive checksum mismatch: {archive.name}")
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                destination = (target / member.filename).resolve()
                if not destination.is_relative_to(target.resolve()):
                    raise RuntimeError("Resource archive contains an unsafe path")
            package.extractall(target)
    for entry in manifest["files"]:
        if digest(target / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Extracted resource checksum mismatch: {entry['path']}")
    return manifest


def collect_licenses(target: Path) -> None:
    target.mkdir(parents=True)
    versions = {}
    for name in ("numpy", "pandas", "scipy", "lightgbm", "openpyxl", "python-dateutil",
                 "pytz", "tzdata", "et-xmlfile", "six", "pyinstaller"):
        dist = importlib.metadata.distribution(name)
        versions[name] = dist.version
        for filename in dist.files or []:
            if any(token in str(filename).lower() for token in ("license", "copying", "notice")):
                source = Path(dist.locate_file(filename))
                if source.is_file():
                    destination = target / name / Path(str(filename))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    prefix = Path(sys.base_prefix)
    for relative in ("LICENSE.txt", "tcl/tcl8.6/license.terms", "tcl/tk8.6/license.terms"):
        source = prefix / relative
        if source.exists():
            destination = target / "Python-Tcl-Tk" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    (target / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("Build on 64-bit Windows with Python 3.12.")
    build_parent = ROOT / "build"
    build_parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"desktop-{APP_VERSION}-", dir=build_parent))
    assets = work / "assets"
    manifest = prepare_assets(assets)
    stage = work / f"D36Predictor-{APP_VERSION}-Windows-x64"
    stage.mkdir()
    verification = work / "verification"
    verification.mkdir()
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
        "--windowed", "--name", "D36Predictor", "--icon", str(HERE / "D36工具.ico"),
        "--paths", str(HERE), "--add-data", f"{assets};assets",
        "--collect-binaries", "lightgbm", "--hidden-import", "openpyxl",
        "--distpath", str(stage), "--workpath", str(work / "pyinstaller"),
        "--specpath", str(work), str(HERE / "d36_predictor_gui.pyw"),
    ], check=True, cwd=ROOT)
    executable = stage / "D36Predictor.exe"
    for option, filename in (("--self-test", "portable_self_test_report.json"),
                             ("--gui-smoke-test", "gui_smoke_report.json")):
        subprocess.run([str(executable), option, "--output-dir", str(verification)],
                       check=True, timeout=180, cwd=work)
        report = json.loads((verification / filename).read_text(encoding="utf-8"))
        if report["status"] != "PASS" or not report["frozen_executable"]:
            raise RuntimeError(f"Executable verification failed: {filename}")
    # GUI smoke exports to its usual output folder; keep it as build evidence.
    if (stage / "outputs").exists():
        shutil.move(str(stage / "outputs"), str(verification / "gui_outputs"))
    for source, name in ((HERE / "README.md", "使用说明.md"),
                         (ROOT / "LICENSE", "LICENSE"),
                         (HERE / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
                         (HERE / "D36未知样品输入模板.csv", "D36未知样品输入模板.csv"),
                         (assets / "geosingle_bundle/example_input.csv", "示例输入.csv")):
        shutil.copy2(source, stage / name)
    collect_licenses(stage / "third_party_licenses")
    # Package only reproducible summaries: omit runner/user absolute paths.
    self_test = json.loads((verification / "portable_self_test_report.json").read_text(encoding="utf-8"))
    release = {
        "version": APP_VERSION, "model": self_test["model"], "platform": "Windows x64",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "executable_sha256": digest(executable),
        "model_resource_manifest_sha256": digest(RESOURCES / "manifest.json"),
        "frozen_probability_replay_max_abs_error": self_test["frozen_probability_replay_max_abs_error"],
        "executable_self_test": "PASS", "executable_gui_smoke_test": "PASS",
        "model_training_rows": manifest["model_training_rows"],
        "retrieval_library_rows": manifest["retrieval_library_rows"],
    }
    (stage / "release-info.json").write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    archive_path = dist / f"D36Predictor-{APP_VERSION}-Windows-x64.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for source in sorted(stage.rglob("*")):
            if source.is_file():
                package.write(source, source.relative_to(stage.parent).as_posix())
    checksums = dist / "SHA256SUMS.txt"
    checksums.write_text(f"{digest(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    info = {"tag": f"v{APP_VERSION}", "title": f"D36 构造环境判别工具 v{APP_VERSION}",
            "archive": archive_path.relative_to(ROOT).as_posix(),
            "checksums": checksums.relative_to(ROOT).as_posix()}
    (dist / "release.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**info, "verification": str(verification), "sha256": digest(archive_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
