from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.common import LOGGER, ROOT, setup_logging

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None


def collect_samples(images_dir: Path, labels_dir: Path, allow_missing_labels: bool) -> list[Sample]:
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No images found in: {images_dir}")

    stems: dict[str, Path] = {}
    samples: list[Sample] = []
    for image in images:
        if image.stem in stems:
            raise RuntimeError(
                f"Duplicate image stem '{image.stem}': {stems[image.stem]} and {image}"
            )
        stems[image.stem] = image
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            if not allow_missing_labels:
                raise FileNotFoundError(f"Missing label for image: {image}")
            label = None
        samples.append(Sample(image=image, label=label))
    return samples


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    items = list(samples)
    random.Random(seed).shuffle(items)
    total = len(items)
    train_end = round(total * train_ratio)
    val_end = train_end + round(total * val_ratio)
    train_end = min(train_end, total)
    val_end = min(val_end, total)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def copy_split(
    splits: dict[str, list[Sample]],
    output: Path,
    mode: str,
    clean: bool,
) -> Path:
    if clean and output.exists():
        shutil.rmtree(output)
    for split in splits:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = output / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "image", "label"])
        for split, samples in splits.items():
            for sample in samples:
                image_target = output / "images" / split / sample.image.name
                label_target = output / "labels" / split / f"{sample.image.stem}.txt"
                operation = shutil.copy2 if mode == "copy" else shutil.move
                operation(str(sample.image), str(image_target))
                if sample.label is None:
                    label_target.touch()
                else:
                    operation(str(sample.label), str(label_target))
                writer.writerow(
                    [split, image_target.relative_to(output), label_target.relative_to(output)]
                )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split YOLO segmentation images and labels.")
    parser.add_argument("--images", default="datasets/road_raw/images")
    parser.add_argument("--labels", default="datasets/road_raw/labels")
    parser.add_argument("--output", default="datasets/road")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    parser.add_argument("--allow-missing-labels", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    if any(ratio < 0 for ratio in (args.train, args.val, args.test)):
        raise ValueError("Split ratios must be non-negative.")
    if abs(args.train + args.val + args.test - 1.0) > 1e-9:
        raise ValueError("--train, --val, and --test must sum to 1.0.")

    samples = collect_samples(
        project_path(args.images),
        project_path(args.labels),
        args.allow_missing_labels,
    )
    splits = split_samples(samples, args.train, args.val, args.seed)
    manifest = copy_split(splits, project_path(args.output), args.mode, args.clean)
    LOGGER.info(
        "Dataset split complete: train=%d, val=%d, test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    LOGGER.info("Manifest: %s", manifest)


if __name__ == "__main__":
    main()
