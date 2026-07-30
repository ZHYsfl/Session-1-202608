"""使用已知距离和尺寸的测量数据估算相机像素焦距。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from distance import estimate_focal_length


def load_measurements(
    csv_path: Path,
) -> dict[str, list[tuple[float, float, float]]]:
    """读取标定 CSV，并按照水平轴 x、垂直轴 y 分组。"""

    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "axis",
            "pixel_size_px",
            "known_distance_m",
            "known_size_m",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV columns must include: {', '.join(sorted(required))}")

        for line_number, row in enumerate(reader, start=2):
            axis = row["axis"].strip().lower()
            if axis not in {"x", "y"}:
                raise ValueError(f"line {line_number}: axis must be x or y")
            try:
                measurement = (
                    float(row["pixel_size_px"]),
                    float(row["known_distance_m"]),
                    float(row["known_size_m"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"line {line_number}: measurement values must be numeric"
                ) from exc
            grouped[axis].append(measurement)
    return dict(grouped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV containing calibration rows")
    parser.add_argument("--pixel-size", type=float, help="box width or height in px")
    parser.add_argument("--known-distance", type=float, help="measured distance in m")
    parser.add_argument("--known-size", type=float, help="physical width or height in m")
    parser.add_argument(
        "--axis",
        choices=("x", "y"),
        default="x",
        help="x for width/fx, y for height/fy",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grouped: dict[str, list[tuple[float, float, float]]]

    # 支持批量 CSV 标定，也支持命令行传入一组测量值快速验证。
    if args.csv:
        grouped = load_measurements(args.csv)
    else:
        values = (args.pixel_size, args.known_distance, args.known_size)
        if any(value is None for value in values):
            raise SystemExit(
                "Provide --csv or all of --pixel-size, --known-distance, "
                "and --known-size."
            )
        grouped = {args.axis: [values]}  # type: ignore[list-item]

    report: dict[str, float | int] = {}
    for axis, measurements in sorted(grouped.items()):
        # x 轴对应 fx，y 轴对应 fy。
        report[f"f{axis}_px"] = round(estimate_focal_length(measurements), 3)
        report[f"{axis}_measurement_count"] = len(measurements)

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
