"""Download and verify official RoboMimic v1.5 PH low-dimensional datasets."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise SystemExit(
        "Missing huggingface_hub. Run setup_official_benchmark.ps1 first."
    ) from exc


REPO_ID = "robomimic/robomimic_datasets"
DATA_DIR = ROOT / "data" / "official"
FILES = {
    "lift": (
        "v1.5/lift/ph/low_dim_v15.hdf5",
        "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540",
    ),
    "can": (
        "v1.5/can/ph/low_dim_v15.hdf5",
        "3f2eb92e0a5025d0095e866ac16cc8092d6a762abe27dec90dbaff9027282962",
    ),
    "square": (
        "v1.5/square/ph/low_dim_v15.hdf5",
        "45d8cabb6d57a4c03e839aa5e4b3e58fb60fe8bd20e1951fec563a2abfd14951",
    ),
    "transport": (
        "v1.5/transport/ph/low_dim_v15.hdf5",
        "260515618f4c8b660e54171497ecceb08c7c6463691a9ce862b158bcacd950b4",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(tasks: list[str], verify_only: bool) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_cache = ROOT / "data" / "hf_download"
    for task in tasks:
        remote_path, expected_hash = FILES[task]
        destination = DATA_DIR / f"{task}_ph_low_dim_v15.hdf5"
        if not destination.exists() and not verify_only:
            downloaded = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=remote_path,
                    local_dir=download_cache,
                )
            )
            shutil.copy2(downloaded, destination)
        if not destination.exists():
            raise FileNotFoundError(destination)
        actual_hash = sha256_file(destination)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{task}: SHA-256 mismatch: {actual_hash} != {expected_hash}"
            )
        print(
            f"{task:<9} verified bytes={destination.stat().st_size} "
            f"sha256={actual_hash}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(FILES),
        default=list(FILES),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    main(args.tasks, args.verify_only)
