from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and print it for visibility."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, cwd=cwd)


def which(name: str) -> str | None:
    """Cross-platform `which`."""
    path = shutil.which(name)
    return path if path else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the uv-managed virtual environment and download assets."
    )
    parser.add_argument("--python-version", default="3.11")
    parser.add_argument("--torch-backend", default="auto")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--force-assets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    uv = which("uv")
    if uv is None:
        print(
            "uv was not found. Install it first:\n"
            "  powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"\n"
            "Or visit https://docs.astral.sh/uv/getting-started/installation/",
            file=sys.stderr,
        )
        return 1

    print(f"[1/5] Installing Python {args.python_version}...")
    run([uv, "python", "install", args.python_version])

    venv_dir = root / ".venv"
    if args.recreate and venv_dir.exists():
        print("Removing existing .venv...")
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print("[2/5] Creating virtual environment...")
        run([uv, "venv", "--python", args.python_version])
    else:
        print("[2/5] Reusing existing .venv.")

    python = venv_dir / "Scripts" / "python.exe" if sys.platform == "win32" else venv_dir / "bin" / "python"
    if not python.exists():
        print(f"Virtual-environment Python was not created: {python}", file=sys.stderr)
        return 1

    print(f"[3/5] Installing PyTorch backend '{args.torch_backend}'...")
    run([uv, "pip", "install", "--python", str(python), "torch", "torchvision", f"--torch-backend={args.torch_backend}"])

    print("[4/5] Installing project dependencies...")
    run([uv, "pip", "install", "--python", str(python), "-e", str(root)])

    if not args.skip_assets:
        print("[5/5] Downloading model and COCO8-Seg assets...")
        asset_cmd = [str(python), "-m", "scripts.prepare_assets"]
        if args.force_assets:
            asset_cmd.append("--force")
        run(asset_cmd)
    else:
        print("[5/5] Asset download skipped.")

    print("Locking environment...")
    with open("environment.lock.txt", "w", encoding="utf-8") as lock_file:
        subprocess.run(
            [uv, "pip", "freeze", "--python", str(python)],
            stdout=lock_file,
            check=True,
        )

    print("\nEnvironment ready.")
    run([
        str(python), "-c",
        "import torch, ultralytics; "
        "print('PyTorch:', torch.__version__); "
        "print('Ultralytics:', ultralytics.__version__); "
        "print('CUDA available:', torch.cuda.is_available()); "
        "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')",
    ])
    print("\nNext: python -m scripts.smoke_test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
