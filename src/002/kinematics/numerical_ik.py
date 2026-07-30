"""Numerical inverse kinematics using damped least squares."""

import numpy as np
from typing import Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from kinematics.forward_kinematics import ForwardKinematics
from utils.math_utils import check_joint_limits, clamp_joint_angles


class NumericalIK:
    """Damped Least Squares inverse kinematics solver.

    This implements iterative numerical IK using the damped least squares
    (Levenberg-Marquardt) method to handle singularities.
    """

    def __init__(self,
                 fk: ForwardKinematics,
                 joint_limits: Tuple[np.ndarray, np.ndarray],
                 max_iterations: int = 50,
                 position_tolerance: float = 0.001,
                 orientation_tolerance: float = 0.01,
                 damping_min: float = 0.001,
                 damping_max: float = 0.1,
                 max_joint_step: float = 0.1):
        """Initialize numerical IK solver.

        Args:
            fk: Forward kinematics instance.
            joint_limits: Tuple of (lower_limits, upper_limits).
            max_iterations: Maximum number of iterations.
            position_tolerance: Position error tolerance in meters.
            orientation_tolerance: Orientation error tolerance in radians.
            damping_min: Minimum damping factor.
            damping_max: Maximum damping factor (used near singularities).
            max_joint_step: Maximum joint change per iteration in radians.
        """
        self.fk = fk
        self.lower_limits = np.array(joint_limits[0])
        self.upper_limits = np.array(joint_limits[1])
        self.max_iterations = max_iterations
        self.pos_tol = position_tolerance
        self.ori_tol = orientation_tolerance
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.max_joint_step = max_joint_step

    def solve(self,
              target_pos: np.ndarray,
              target_alpha: float,
              target_psi: float,
              initial_guess: Optional[np.ndarray] = None,
              return_iterations: bool = False) -> Optional[np.ndarray]:
        """Solve inverse kinematics using damped least squares.

        Args:
            target_pos: Target position [x, y, z].
            target_alpha: Target pitch angle.
            target_psi: Target roll angle.
            initial_guess: Initial joint configuration. If None, uses zeros.
            return_iterations: If True, return (solution, num_iterations).

        Returns:
            Solution joint angles or None if failed.
            If return_iterations=True, returns (solution, iterations) or (None, iterations).
        """
        # Initial guess
        if initial_guess is None:
            q = np.zeros(5)
        else:
            q = initial_guess.copy()

        # Ensure initial guess is within limits
        q = clamp_joint_angles(q, self.lower_limits, self.upper_limits)

        # Target task space
        target_task = np.array([target_pos[0], target_pos[1], target_pos[2],
                               target_alpha, target_psi])

        for iteration in range(self.max_iterations):
            # Current task space
            pos, alpha, psi = self.fk.compute_task_space(q)
            current_task = np.array([pos[0], pos[1], pos[2], alpha, psi])

            # Error
            error = target_task - current_task

            # Check convergence
            pos_error = np.linalg.norm(error[:3])
            ori_error = np.linalg.norm(error[3:])

            if pos_error < self.pos_tol and ori_error < self.ori_tol:
                if return_iterations:
                    return q, iteration + 1
                return q

            # Compute Jacobian
            J = self.fk.compute_jacobian(q)

            # Adaptive damping based on manipulability
            manipulability = np.sqrt(np.linalg.det(J @ J.T))
            if manipulability < 0.01:  # Near singularity
                damping = self.damping_max
            else:
                damping = self.damping_min

            # Damped least squares update
            # Δq = J^T (JJ^T + λ²I)^(-1) e
            JJT = J @ J.T
            damped_matrix = JJT + (damping**2) * np.eye(5)

            try:
                delta_q = J.T @ np.linalg.solve(damped_matrix, error)
            except np.linalg.LinAlgError:
                # Singular matrix, increase damping
                damped_matrix = JJT + (self.damping_max**2) * np.eye(5)
                try:
                    delta_q = J.T @ np.linalg.solve(damped_matrix, error)
                except np.linalg.LinAlgError:
                    if return_iterations:
                        return None, iteration + 1
                    return None

            # Limit step size
            step_norm = np.linalg.norm(delta_q)
            if step_norm > self.max_joint_step:
                delta_q = delta_q * (self.max_joint_step / step_norm)

            # Update joints
            q = q + delta_q

            # Clamp to limits
            q = clamp_joint_angles(q, self.lower_limits, self.upper_limits)

        # Failed to converge
        if return_iterations:
            return None, self.max_iterations
        return None

    def refine_solution(self,
                       target_pos: np.ndarray,
                       target_alpha: float,
                       target_psi: float,
                       initial_solution: np.ndarray,
                       num_steps: int = 3) -> Tuple[np.ndarray, int]:
        """Refine an approximate solution using limited iterations.

        This is used for hybrid methods where NN provides initial guess.

        Args:
            target_pos: Target position.
            target_alpha: Target pitch.
            target_psi: Target roll.
            initial_solution: Initial joint angles from another method (e.g., NN).
            num_steps: Number of refinement iterations.

        Returns:
            Tuple of (refined_solution, iterations_used).
        """
        original_max_iter = self.max_iterations
        self.max_iterations = num_steps

        result = self.solve(target_pos, target_alpha, target_psi,
                          initial_guess=initial_solution,
                          return_iterations=True)

        self.max_iterations = original_max_iter

        if result[0] is None:
            # Refinement failed, return original
            return initial_solution, result[1]

        return result
