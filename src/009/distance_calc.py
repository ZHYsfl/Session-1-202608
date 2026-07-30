import numpy as np
from typing import Optional, Union
import config

CAR_REAL_HEIGHT = config.CAR_REAL_HEIGHT
PERSON_REAL_HEIGHT = config.PERSON_REAL_HEIGHT
FOCAL_LENGTH = config.FOCAL_LENGTH
ENABLE_WH_DOUBLE_CAL = config.ENABLE_WH_DOUBLE_CAL
ENABLE_FRAME_SMOOTH = config.ENABLE_FRAME_SMOOTH
SMOOTH_FRAME_NUM = config.SMOOTH_FRAME_NUM

distance_history = []

def _to_pixels(start: Union[int, float, str], end: Union[int, float, str]) -> int:
    try:
        value = round(abs(float(end) - float(start)))
        return max(0, int(value))
    except (ValueError, TypeError):
        return 0

def calc_distance_by_height(real_h: float, pixel_h: int, focal: int) -> Optional[float]:
    if pixel_h <= 0:
        return None
    return round((real_h * focal) / pixel_h, 2)

def calc_distance_by_width(real_w: float, pixel_w: int, focal: int) -> Optional[float]:
    if pixel_w <= 0:
        return None
    return round((real_w * focal) / pixel_w, 2)

def get_final_distance(cls: str, x1, y1, x2, y2) -> Optional[float]:
    global distance_history
    pixel_h = _to_pixels(y1, y2)
    pixel_w = _to_pixels(x1, x2)
    if pixel_h == 0:
        return None

    if cls == "car":
        h_true = CAR_REAL_HEIGHT
        w_true = 1.8
    elif cls == "person":
        h_true = PERSON_REAL_HEIGHT
        w_true = 0.45
    else:
        return None

    d_h = calc_distance_by_height(h_true, pixel_h, FOCAL_LENGTH)
    if d_h is None:
        return None

    if ENABLE_WH_DOUBLE_CAL:
        d_w = calc_distance_by_width(w_true, pixel_w, FOCAL_LENGTH)
        dist_raw = (d_h + d_w) / 2 if d_w is not None else d_h
    else:
        dist_raw = d_h

    if ENABLE_FRAME_SMOOTH:
        distance_history.append(dist_raw)
        if len(distance_history) > SMOOTH_FRAME_NUM:
            distance_history.pop(0)
        return round(float(np.mean(distance_history)), 2)

    return dist_raw

def clear_history() -> None:
    global distance_history
    distance_history = []