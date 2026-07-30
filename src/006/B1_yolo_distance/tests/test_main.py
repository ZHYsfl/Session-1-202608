from pathlib import Path

import main


def test_parse_repository_relative_source(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "src" / "sample.avi"
    video.parent.mkdir()
    video.touch()
    monkeypatch.chdir(tmp_path)

    assert main.parse_source("src/sample.avi") == str(video.resolve())


def test_parse_project_relative_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = (main.PROJECT_DIR / "data/sample.avi").resolve()

    assert main.parse_source("data/sample.avi") == str(expected)


def test_parse_camera_index() -> None:
    assert main.parse_source("0") == 0
    assert main.parse_source(1) == 1


def test_command_line_focal_length_overrides_config() -> None:
    config = {
        "camera": {"fx_px": None, "fy_px": None},
        "objects": {
            "person": {"real_size_m": 1.7, "dimension": "height"}
        },
    }
    estimator = main.build_estimator(config, fy_override=680)

    assert estimator.fy_px == 680
