"""Analytical inverse kinematics for SO-101 (simplified model)."""

import numpy as np
from typing import Tuple, List, Optional
from ..utils.math_utils import normalize_angle, check_joint_limits


class AnalyticalIK:
    """Simplified analytical inverse kinematics for SO-101.

    This implements a simplified IK solution assuming:
    - Base rotation + planar shoulder-elbow-wrist structure
    - Known DH parameters or link lengths
    """

    def __init__(self,
                 shoulder_height: float = 0.08,
                 radial_offset: float = 0.02,
                 upper_arm_length: float = 0.15,
                 forearm_length: float = 0.15,
                 wrist_to_tcp: float = 0.10,
                 joint_limits: Optional[Tuple[List[float], List[float]]] = None):
        """Initialize analytical IK solver.

        Args:
            shoulder_height: Height of shoulder joint (h).
            radial_offset: Radial offset from base (d0).
            upper_arm_length: Length from shoulder to elbow (L1).
            forearm_length: Length from elbow to wrist (L2).
            wrist_to_tcp: Length from wrist to tool center point (L3).
            joint_limits: Tuple of (lower_limits, upper_limits) for 5 joints.
        """
        self.h = shoulder_height
        self.d0 = radial_offset
        self.L1 = upper_arm_length
        self.L2 = forearm_length
        self.L3 = wrist_to_tcp

        if joint_limits is not None:
            self.lower_limits = np.array(joint_limits[0])
            self.upper_limits = np.array(joint_limits[1])
        else:
            # Default limits (will be overridden by config)
            self.lower_limits = np.array([-3.14, -1.57, -2.35, -1.57, -3.14])
            self.upper_limits = np.array([3.14, 1.57, 2.35, 1.57, 3.14])

    def solve(self,
              target_pos: np.ndarray,
              target_alpha: float,
              target_psi: float,
              prefer_elbow_up: bool = True) -> Optional[np.ndarray]:
        """Solve inverse kinematics for target pose.

        Args:
            target_pos: Target position [x, y, z].
            target_alpha: Target pitch angle (gripper pitch).
            target_psi: Target roll angle (wrist roll).
            prefer_elbow_up: If True, prefer elbow-up solution.

        Returns:
            Joint angles [q1, q2, q3, q4, q5] or None if no solution.
        """
        x, y, z = target_pos

        # q1: Base rotation
        rho = np.sqrt(x**2 + y**2)
        if rho < 1e-6:
            return None  # Singular at origin

        q1 = np.arctan2(y, x)

        # Compute wrist position
        r_w = rho - self.d0 - self.L3 * np.cos(target_alpha)
        z_w = z - self.h - self.L3 * np.sin(target_alpha)

        # Check reachability
        dist_squared = r_w**2 + z_w**2
        reach_max = (self.L1 + self.L2)**2
        reach_min = (self.L1 - self.L2)**2

        if dist_squared > reach_max or dist_squared < reach_min:
            return None  # Target out of reach

        # q3: Elbow angle (law of cosines)
        D = (r_w**2 + z_w**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)

        if abs(D) > 1.0:
            return None  # Numerical error or unreachable

        # Two solutions: elbow up (+) and elbow down (-)
        sqrt_term = np.sqrt(1 - D**2)
        q3 = np.arctan2(sqrt_term if prefer_elbow_up else -sqrt_term, D)

        # q2: Shoulder angle
        beta = np.arctan2(z_w, r_w)
        gamma = np.arctan2(self.L2 * np.sin(q3),
                          self.L1 + self.L2 * np.cos(q3))
        q2 = beta - gamma

        # q4: Wrist angle
        q4 = target_alpha - q2 - q3

        # q5: Wrist roll
        q5 = target_psi

        # Normalize angles
        joint_angles = np.array([
            normalize_angle(q1),
            normalize_angle(q2),
            normalize_angle(q3),
            normalize_angle(q4),
            normalize_angle(q5)
        ])

        # Check joint limits
        if not check_joint_limits(joint_angles, self.lower_limits, self.upper_limits):
            return None

        return joint_angles

    def solve_both_branches(self,
                           target_pos: np.ndarray,
                           target_alpha: float,
                           target_psi: float) -> List[np.ndarray]:
        """Solve IK and return both elbow-up and elbow-down solutions if valid.

        Args:
            target_pos: Target position [x, y, z].
            target_alpha: Target pitch angle.
            target_psi: Target roll angle.

        Returns:
            List of valid solutions (may contain 0, 1, or 2 solutions).
        """
        solutions = []

        for prefer_elbow_up in [True, False]:
            sol = self.solve(target_pos, target_alpha, target_psi, prefer_elbow_up)
            if sol is not None:
                solutions.append(sol)

        return solutions

    def choose_closest_solution(self,
                               target_pos: np.ndarray,
                               target_alpha: float,
                               target_psi: float,
                               current_joint_angles: np.ndarray) -> Optional[np.ndarray]:
        """Choose the IK solution closest to current configuration.

        Args:
            target_pos: Target position.
            target_alpha: Target pitch.
            target_psi: Target roll.
            current_joint_angles: Current joint configuration.

        Returns:
            Closest valid solution or None.
        """
        solutions = self.solve_both_branches(target_pos, target_alpha, target_psi)

        if not solutions:
            return None

        if len(solutions) == 1:
            return solutions[0]

        # Choose solution with minimum joint distance
        distances = [np.linalg.norm(sol - current_joint_angles) for sol in solutions]
        return solutions[np.argmin(distances)]
