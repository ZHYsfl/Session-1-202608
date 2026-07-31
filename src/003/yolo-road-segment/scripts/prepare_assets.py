from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

from ultralytics import YOLO, settings

from src.common import LOGGER, ROOT, ensure_project_dirs, setup_logging

COCO8_SEG_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-seg.zip"
)


@contextmanager
def working_directory(path: Path):
    import os

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe ZIP member: {member.filename}")
    archive.extractall(destination)


def configure_ultralytics() -> None:
    settings.update(
        {
            "datasets_dir": str((ROOT / "datasets").resolve()),
            "weights_dir": str((ROOT / "models").resolve()),
            "runs_dir": str((ROOT / "outputs").resolve()),
        }
    )
    LOGGER.info("Ultralytics project directories configured.")


def download_model(model_name: str, force: bool = False) -> Path:
    target = ROOT / "models" / model_name
    if target.exists() and not force:
        LOGGER.info("Model already exists: %s", target)
        return target
    if target.exists():
        target.unlink()

    with working_directory(target.parent):
        model = YOLO(model_name, task="segment")
        model.info(verbose=False)
    if not target.exists():
        downloaded = target.parent / Path(model_name).name
        if downloaded.exists() and downloaded != target:
            shutil.move(str(downloaded), str(target))
    if not target.exists():
        raise RuntimeError(f"Model download did not create the expected file: {target}")
    LOGGER.info("Model ready: %s", target)
    return target


def download_coco8(force: bool = False) -> Path:
    target = ROOT / "datasets" / "coco8-seg"
    if target.exists() and not force:
        LOGGER.info("COCO8-Seg already exists: %s", target)
        return target
    if target.exists():
        shutil.rmtree(target)

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "coco8-seg.zip"
        LOGGER.info("Downloading COCO8-Seg...")
        urllib.request.urlretrieve(COCO8_SEG_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, ROOT / "datasets")
    if not target.exists():
        raise RuntimeError(f"Dataset extraction did not create: {target}")
    LOGGER.info("COCO8-Seg ready: %s", target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare model weights and smoke-test data.")
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-coco8", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    ensure_project_dirs()
    configure_ultralytics()
    if not args.skip_model:
        download_model(args.model, force=args.force)
    if not args.skip_coco8:
        download_coco8(force=args.force)


if __name__ == "__main__":
    main()
