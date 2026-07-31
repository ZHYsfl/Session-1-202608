"""Unit tests for analytical inverse kinematics."""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinematics import AnalyticalIK


class TestAnalyticalIK(unittest.TestCase):
    """Test cases for analytical IK."""

    def setUp(self):
        """Set up test fixtures."""
        # Default SO-101 approximate parameters
        self.ik = AnalyticalIK(
            shoulder_height=0.08,
            radial_offset=0.02,
            upper_arm_length=0.15,
            forearm_length=0.15,
            wrist_to_tcp=0.10
        )

    def test_reachable_point(self):
        """Test IK for a reachable point.

        Target verified reachable under the simplified model (respects the
        +/-1.57 rad wrist limit). Note: with these nominal link lengths the
        reachable set is small -- the central finding of the paper.
        """
        target_pos = np.array([0.24, 0.0, 0.08])
        target_alpha = 0.0
        target_psi = 0.0

        solution = self.ik.solve(target_pos, target_alpha, target_psi)

        self.assertIsNotNone(solution)
        self.assertEqual(solution.shape, (5,))

    def test_unreachable_point(self):
        """Test IK for an unreachable point."""
        # Point beyond maximum reach
        target_pos = np.array([1.0, 0.0, 0.5])
        target_alpha = 0.0
        target_psi = 0.0

        solution = self.ik.solve(target_pos, target_alpha, target_psi)

        self.assertIsNone(solution)

    def test_both_branches(self):
        """Test that both elbow-up and elbow-down solutions are found."""
        target_pos = np.array([0.24, 0.0, 0.08])
        target_alpha = 0.0
        target_psi = 0.0

        solutions = self.ik.solve_both_branches(target_pos, target_alpha, target_psi)

        # Should have at least one solution
        self.assertGreater(len(solutions), 0)

        # For most positions, should have two solutions
        if len(solutions) == 2:
            # Solutions should be different
            self.assertGreater(
                np.linalg.norm(solutions[0] - solutions[1]),
                0.01
            )

    def test_joint_limits(self):
        """Test that solutions respect joint limits."""
        # Generate random reachable points
        for _ in range(10):
            # Random point in approximate workspace
            target_pos = np.random.uniform([0.1, -0.2, 0.05], [0.3, 0.2, 0.25])
            target_alpha = np.random.uniform(-0.5, 0.5)
            target_psi = np.random.uniform(-1.0, 1.0)

            solution = self.ik.solve(target_pos, target_alpha, target_psi)

            if solution is not None:
                # Check all joints are within limits
                self.assertTrue(np.all(solution >= self.ik.lower_limits))
                self.assertTrue(np.all(solution <= self.ik.upper_limits))

    def test_closest_solution(self):
        """Test that closest solution is selected."""
        target_pos = np.array([0.24, 0.0, 0.08])
        target_alpha = 0.0
        target_psi = 0.0

        # Current configuration favoring elbow-up
        current_up = np.array([0.0, 0.5, 0.5, 0.0, 0.0])

        # Current configuration favoring elbow-down
        current_down = np.array([0.0, 0.5, -0.5, 0.0, 0.0])

        solution_up = self.ik.choose_closest_solution(
            target_pos, target_alpha, target_psi, current_up
        )
        solution_down = self.ik.choose_closest_solution(
            target_pos, target_alpha, target_psi, current_down
        )

        if solution_up is not None and solution_down is not None:
            # Solutions should differ (selected different branches)
            self.assertGreater(
                np.linalg.norm(solution_up - solution_down),
                0.01
            )


if __name__ == '__main__':
    unittest.main()
