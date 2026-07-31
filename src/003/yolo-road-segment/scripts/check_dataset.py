from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2

from src.common import LOGGER, ROOT, load_yaml, resolve_local_path, setup_logging, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Finding:
    level: str
    split: str
    path: str
    message: str


@dataclass
class Report:
    dataset: str
    class_names: dict[int, str]
    split_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    class_instances: dict[int, int] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return sum(item.level == "error" for item in self.findings)

    @property
    def warnings(self) -> int:
        return sum(item.level == "warning" for item in self.findings)


def class_names_from_yaml(data: dict[str, Any]) -> dict[int, str]:
    names = data.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError("Dataset YAML must contain 'names' as a list or mapping.")


def polygon_area(coords: list[float]) -> float:
    points = list(zip(coords[0::2], coords[1::2], strict=True))
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True)
        )
    ) / 2.0


def validate_label(
    label_path: Path,
    split: str,
    class_names: dict[int, str],
    findings: list[Finding],
    counts: Counter[int],
) -> None:
    text = label_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = line.split()
        location = f"{label_path}:{line_number}"
        if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
            findings.append(
                Finding(
                    "error",
                    split,
                    location,
                    "Expected: class_id followed by at least three x/y point pairs.",
                )
            )
            continue
        try:
            class_value = float(tokens[0])
            if not class_value.is_integer():
                raise ValueError
            class_id = int(class_value)
            coords = [float(token) for token in tokens[1:]]
        except ValueError:
            findings.append(Finding("error", split, location, "Non-numeric class or coordinate."))
            continue

        if class_id not in class_names:
            findings.append(
                Finding("error", split, location, f"Class id {class_id} is not declared in YAML.")
            )
        if not all(math.isfinite(value) for value in coords):
            findings.append(
                Finding("error", split, location, "Coordinate contains NaN or infinity.")
            )
            continue
        if not all(0.0 <= value <= 1.0 for value in coords):
            findings.append(
                Finding("error", split, location, "Coordinates must be normalized to [0, 1].")
            )
        if polygon_area(coords) <= 1e-8:
            findings.append(
                Finding("warning", split, location, "Polygon area is zero or very small.")
            )
        counts[class_id] += 1


def check_dataset(config_path: str, output_path: str) -> Report:
    config_file = resolve_local_path(config_path, must_exist=True)
    data = load_yaml(config_file)
    class_names = class_names_from_yaml(data)
    dataset_path = Path(str(data.get("path", ""))).expanduser()
    if not dataset_path.is_absolute():
        candidate = (ROOT / dataset_path).resolve()
        if candidate.exists():
            dataset_path = candidate
        else:
            dataset_path = (config_file.parent / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_path}")

    report = Report(dataset=str(dataset_path), class_names=class_names)
    class_counter: Counter[int] = Counter()

    for split in ("train", "val", "test"):
        split_value = data.get(split)
        if not split_value:
            continue
        image_dir = Path(str(split_value))
        if not image_dir.is_absolute():
            image_dir = dataset_path / image_dir
        label_dir = dataset_path / "labels" / split

        if not image_dir.exists():
            report.findings.append(
                Finding("error", split, str(image_dir), "Image directory missing.")
            )
            continue
        if not label_dir.exists():
            report.findings.append(
                Finding("error", split, str(label_dir), "Label directory missing.")
            )
            continue

        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}

        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        for stem in missing_labels:
            report.findings.append(
                Finding("error", split, str(image_dir / stem), "Image has no matching label file.")
            )
        for stem in orphan_labels:
            report.findings.append(
                Finding("warning", split, str(label_dir / f"{stem}.txt"), "Orphan label file.")
            )

        unreadable = 0
        empty_labels = 0
        for image in images:
            if cv2.imread(str(image)) is None:
                unreadable += 1
                report.findings.append(Finding("error", split, str(image), "Image is unreadable."))
            label = label_dir / f"{image.stem}.txt"
            if label.exists():
                if not label.read_text(encoding="utf-8-sig").strip():
                    empty_labels += 1
                validate_label(label, split, class_names, report.findings, class_counter)

        report.split_counts[split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_labels": empty_labels,
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "unreadable_images": unreadable,
        }
        if split in {"train", "val"} and not images:
            report.findings.append(
                Finding("error", split, str(image_dir), "Split contains no images.")
            )

    report.class_instances = dict(sorted(class_counter.items()))
    payload = asdict(report) | {"errors": report.errors, "warnings": report.warnings}
    write_json(output_path, payload)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate YOLO polygon segmentation data.")
    parser.add_argument("--data", default="configs/road_seg.yaml")
    parser.add_argument("--output", default="outputs/dataset_check.json")
    parser.add_argument("--warnings-as-errors", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    report = check_dataset(args.data, args.output)
    for split, counts in report.split_counts.items():
        LOGGER.info("%s: %s", split, counts)
    LOGGER.info("Class instances: %s", report.class_instances)
    LOGGER.info("Dataset check: %d error(s), %d warning(s)", report.errors, report.warnings)
    LOGGER.info("Report written to: %s", resolve_local_path(args.output))
    if report.errors or (args.warnings_as_errors and report.warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
