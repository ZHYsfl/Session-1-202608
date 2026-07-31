"""Mathematical utility functions."""

import numpy as np
from typing import Tuple


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi].

    Args:
        angle: Angle in radians.

    Returns:
        Normalized angle in [-pi, pi].
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def rotation_matrix_x(angle: float) -> np.ndarray:
    """Rotation matrix around X axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle: float) -> np.ndarray:
    """Rotation matrix around Y axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Rotation matrix around Z axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to ZYX Euler angles.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Tuple of (yaw, pitch, roll) in radians.
    """
    # Check for gimbal lock
    if abs(R[2, 0]) >= 1:
        yaw = 0
        if R[2, 0] < 0:  # R[2,0] = -1
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:  # R[2,0] = 1
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))

    return yaw, pitch, roll


def euler_zyx_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Convert ZYX Euler angles to rotation matrix.

    Args:
        yaw: Rotation around Z axis in radians.
        pitch: Rotation around Y axis in radians.
        roll: Rotation around X axis in radians.

    Returns:
        3x3 rotation matrix.
    """
    return rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)


def check_joint_limits(joint_angles: np.ndarray,
                       lower_limits: np.ndarray,
                       upper_limits: np.ndarray,
                       tolerance: float = 1e-6) -> bool:
    """Check if joint angles are within limits.

    Args:
        joint_angles: Array of joint angles.
        lower_limits: Array of lower limits.
        upper_limits: Array of upper limits.
        tolerance: Tolerance for limit checking.

    Returns:
        True if all joints are within limits, False otherwise.
    """
    return np.all(joint_angles >= lower_limits - tolerance) and \
           np.all(joint_angles <= upper_limits + tolerance)


def clamp_joint_angles(joint_angles: np.ndarray,
                       lower_limits: np.ndarray,
                       upper_limits: np.ndarray) -> np.ndarray:
    """Clamp joint angles to limits.

    Args:
        joint_angles: Array of joint angles.
        lower_limits: Array of lower limits.
        upper_limits: Array of upper limits.

    Returns:
        Clamped joint angles.
    """
    return np.clip(joint_angles, lower_limits, upper_limits)
