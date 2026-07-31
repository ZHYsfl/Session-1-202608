from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command and print it for visibility."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate check -> train -> validate -> predict."
    )
    parser.add_argument(
        "--stage",
        choices=["check", "train", "val", "predict", "all"],
        default="all",
    )
    parser.add_argument("--run-name", default="road_yolo26n_seg")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source", default="datasets/road/images/test")
    parser.add_argument("--clean-run", action="store_true")
    return parser.parse_args()


def find_python(root: Path) -> Path:
    """Locate the virtual-environment Python interpreter."""
    if sys.platform == "win32":
        candidate = root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = root / ".venv" / "bin" / "python"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Environment not found at {candidate}. Run 'python scripts/bootstrap.py' first."
        )
    return candidate


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    python = find_python(root)

    train_root = root / "outputs" / "train"

    def latest_train_dir() -> Path | None:
        """Find the most recent training directory with a best.pt checkpoint."""
        if not train_root.exists():
            return None
        candidates = sorted(
            train_root.glob(f"{args.run_name}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "weights" / "best.pt").exists():
                return candidate
        return None

    def invoke_check() -> None:
        run([str(python), "-m", "scripts.check_dataset", "--data", "configs/road_seg.yaml"])

    def invoke_train() -> None:
        if args.clean_run and train_root.exists():
            for old_dir in train_root.glob(f"{args.run_name}_*"):
                if old_dir.is_dir():
                    print(f"Removing existing training directory: {old_dir}")
                    shutil.rmtree(old_dir)
        run([
            str(python), "-m", "src.train",
            "--name", args.run_name,
            "--device", args.device,
            "--exist-ok",
        ])

    def invoke_val() -> None:
        train_dir = latest_train_dir()
        if train_dir is None:
            raise FileNotFoundError(f"Best model not found for run '{args.run_name}'")
        best_model = train_dir / "weights" / "best.pt"
        run([
            str(python), "-m", "src.validate",
            "--model", str(best_model),
            "--device", args.device,
            "--name", f"{args.run_name}_val",
            "--exist-ok",
        ])

    def invoke_predict() -> None:
        train_dir = latest_train_dir()
        if train_dir is None:
            raise FileNotFoundError(f"Best model not found for run '{args.run_name}'")
        best_model = train_dir / "weights" / "best.pt"
        run([
            str(python), "-m", "src.predict",
            "--model", str(best_model),
            "--source", args.source,
            "--device", args.device,
            "--name", f"{args.run_name}_predict",
            "--exist-ok",
        ])

    stages = {
        "check": [invoke_check],
        "train": [invoke_check, invoke_train],
        "val": [invoke_val],
        "predict": [invoke_predict],
        "all": [invoke_check, invoke_train, invoke_val, invoke_predict],
    }

    for step in stages[args.stage]:
        step()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
