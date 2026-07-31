from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.common import LOGGER, ROOT, setup_logging

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def collect_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    raise FileNotFoundError(f"Video source does not exist: {source}")


def extract_video(
    video_path: Path,
    output_dir: Path,
    interval: float,
    start: float,
    end: float | None,
    image_ext: str,
    jpeg_quality: int,
    overwrite: bool,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS reported by video: {video_path}")

    start_frame = max(0, round(start * fps))
    end_frame = frame_count if end is None else min(frame_count, round(end * fps))
    step = max(1, round(interval * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_index = start_frame
    total_samples = max(0, (end_frame - start_frame + step - 1) // step)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if image_ext in {"jpg", "jpeg"} else []

    with tqdm(total=total_samples, desc=video_path.name, unit="frame") as progress:
        while frame_index < end_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break
            filename = f"{video_path.stem}_{frame_index:08d}.{image_ext}"
            target = output_dir / filename
            if overwrite or not target.exists():
                if not cv2.imwrite(str(target), frame, params):
                    capture.release()
                    raise RuntimeError(f"Failed to write image: {target}")
                saved += 1
            frame_index += step
            progress.update(1)

    capture.release()
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract regularly spaced frames from videos.")
    parser.add_argument("source", help="Video file or directory containing videos.")
    parser.add_argument("--output", default="datasets/road_raw/images")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between frames.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--end", type=float, help="Optional end time in seconds.")
    parser.add_argument("--image-ext", choices=["jpg", "jpeg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0.")
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser()
    output = output.resolve() if output.is_absolute() else (ROOT / output).resolve()
    videos = collect_videos(source)
    if not videos:
        raise RuntimeError(f"No supported video files found under: {source}")

    total = 0
    for video in videos:
        total += extract_video(
            video,
            output,
            args.interval,
            args.start,
            args.end,
            args.image_ext,
            args.jpeg_quality,
            args.overwrite,
        )
    LOGGER.info("Extracted %d frame(s) to %s", total, output)


if __name__ == "__main__":
    main()
