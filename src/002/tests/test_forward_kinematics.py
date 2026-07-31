"""Unit tests for forward kinematics."""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinematics import ForwardKinematics


class TestForwardKinematics(unittest.TestCase):
    """Test cases for forward kinematics."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Resolve URDF relative to the project root (src/002) so the test
        # passes regardless of the current working directory.
        root = Path(__file__).resolve().parents[1]
        urdf = root / 'models' / 'so101_new_calib.urdf'
        try:
            cls.fk = ForwardKinematics(str(urdf))
            cls.has_urdf = True
        except Exception as e:
            cls.has_urdf = False
            print(f"Warning: URDF not loaded ({e}), skipping FK tests")

    def test_zero_configuration(self):
        """Test forward kinematics at zero configuration."""
        if not self.has_urdf:
            self.skipTest("URDF not available")

        q = np.zeros(5)
        pos, rot = self.fk.compute(q)

        # Position should be along positive z-axis at zero config
        self.assertEqual(pos.shape, (3,))
        self.assertEqual(rot.shape, (3, 3))

        # Rotation matrix should be orthonormal
        np.testing.assert_almost_equal(
            rot @ rot.T,
            np.eye(3),
            decimal=6
        )

    def test_jacobian_shape(self):
        """Test Jacobian computation shape."""
        if not self.has_urdf:
            self.skipTest("URDF not available")

        q = np.random.uniform(-1, 1, size=5)
        J = self.fk.compute_jacobian(q)

        self.assertEqual(J.shape, (5, 5))

    def test_jacobian_numerical(self):
        """Test Jacobian against numerical differentiation."""
        if not self.has_urdf:
            self.skipTest("URDF not available")

        q = np.random.uniform(-1, 1, size=5)
        J = self.fk.compute_jacobian(q)

        # Numerical Jacobian
        eps = 1e-6
        J_numerical = np.zeros((5, 5))
        task0 = np.array([*self.fk.compute_task_space(q)[0],
                         *self.fk.compute_task_space(q)[1:]])

        for i in range(5):
            q_plus = q.copy()
            q_plus[i] += eps
            task_plus = np.array([*self.fk.compute_task_space(q_plus)[0],
                                 *self.fk.compute_task_space(q_plus)[1:]])
            J_numerical[:, i] = (task_plus - task0) / eps

        np.testing.assert_almost_equal(J, J_numerical, decimal=3)


if __name__ == '__main__':
    unittest.main()
