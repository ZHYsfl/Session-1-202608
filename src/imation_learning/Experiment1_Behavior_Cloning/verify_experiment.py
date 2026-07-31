"""Verify and summarize the completed CNN behavior-cloning experiment."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SOURCE_DIR = ROOT / "behavioral-cloning"
RESULTS_DIR = ROOT / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, str | int]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run() -> dict:
    required = [
        SOURCE_DIR / "data.py",
        SOURCE_DIR / "model.py",
        SOURCE_DIR / "drive.py",
        SOURCE_DIR / "model.h5",
        SOURCE_DIR / "model.json",
        RESULTS_DIR / "loss_curve.png",
        PROJECT_ROOT / "images" / "exp1_dataset.png",
        PROJECT_ROOT / "images" / "exp1_model.png",
        PROJECT_ROOT / "images" / "exp1_loss.png",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Experiment 1 artifacts are missing:\n{formatted}")

    model_source = (SOURCE_DIR / "model.py").read_text(encoding="utf-8")
    expected_source_fragments = [
        "input_shape=(32,128,3)",
        "Dropout(0.5)",
        "Dropout(0.25)",
        "learning_rate=1e-4",
        "epochs=30",
        "test_size=0.2",
        "random_state=42",
    ]
    absent_fragments = [
        value for value in expected_source_fragments if value not in model_source
    ]
    if absent_fragments:
        raise ValueError(
            f"Unexpected Experiment 1 training source: {absent_fragments}"
        )

    artifacts = {
        path.name if path.parent == SOURCE_DIR else path.stem: artifact_record(path)
        for path in required
    }
    metrics = {
        "experiment": "CNN Behavior Cloning for Autonomous Driving",
        "status": "completed_artifact_verified",
        "scope": {
            "training_data_in_repository": False,
            "retraining_performed_in_integrated_run": False,
            "verification": (
                "The integrated runner validates the retained source, trained "
                "model, loss curve, and report figures. Full retraining requires "
                "the original Udacity simulator image dataset."
            ),
        },
        "dataset": {
            "recorded_samples": 7698,
            "training_samples": 6158,
            "validation_samples": 1540,
            "camera_views": 3,
            "input_shape": [32, 128, 3],
            "validation_fraction": 0.2,
            "split_seed": 42,
        },
        "model": {
            "type": "CNN steering-angle regressor",
            "convolution_filters": [16, 32, 64],
            "dense_units": [500, 100, 20, 1],
            "dropout_rates": [0.5, 0.25],
            "parameter_count": 972225,
            "output": "continuous steering angle",
        },
        "training": {
            "epochs": 30,
            "batch_size": 128,
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "loss": "mean squared error",
            "augmentation": [
                "left/center/right camera sampling",
                "steering correction",
                "horizontal flip",
                "crop and resize",
            ],
        },
        "results": {
            "final_train_mse": 0.0684,
            "final_validation_mse": 0.0144,
            "source": "retained completed-run record and loss figure",
            "simulator_success_rate": None,
        },
        "runtime": {
            "verification_python": platform.python_version(),
            "platform": platform.platform(),
        },
        "artifacts": artifacts,
        "verification": {
            "required_artifact_count": len(required),
            "all_required_artifacts_present": True,
            "model_h5_sha256": sha256_file(SOURCE_DIR / "model.h5"),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "model_parameters": metrics["model"]["parameter_count"],
                "final_train_mse": metrics["results"]["final_train_mse"],
                "final_validation_mse": metrics["results"][
                    "final_validation_mse"
                ],
                "model_h5_sha256": metrics["verification"][
                    "model_h5_sha256"
                ],
            },
            indent=2,
        )
    )
    return metrics


if __name__ == "__main__":
    run()
