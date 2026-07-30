"""Forward kinematics implementation using URDF."""

import numpy as np
from typing import Tuple, Optional
import urdfpy


class ForwardKinematics:
    """Forward kinematics for SO-101 robot using URDF model.

    This class provides forward kinematics computation from joint angles
    to end-effector pose (position and orientation).
    """

    def __init__(self, urdf_path: str):
        """Initialize forward kinematics with URDF model.

        Args:
            urdf_path: Path to the URDF file.
        """
        self.robot = urdfpy.URDF.load(urdf_path)
        self.joint_names = self._get_actuated_joint_names()

    def _get_actuated_joint_names(self) -> list:
        """Get names of actuated joints (excluding gripper).

        Returns:
            List of joint names for the 5 DOF arm.
        """
        # Expected joint names for SO-101 (adjust based on actual URDF)
        expected_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll'
        ]

        # Extract from URDF and validate
        actuated_joints = []
        for joint in self.robot.joints:
            if joint.joint_type != 'fixed' and 'gripper' not in joint.name.lower():
                actuated_joints.append(joint.name)

        # If URDF uses different naming, we'll need to map it
        # For now, return first 5 non-fixed joints
        return actuated_joints[:5]

    def compute(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics.

        Args:
            joint_angles: Array of 5 joint angles in radians [q1, q2, q3, q4, q5].

        Returns:
            Tuple of:
                - position: 3D position [x, y, z] in meters
                - rotation_matrix: 3x3 rotation matrix
        """
        if len(joint_angles) != 5:
            raise ValueError(f"Expected 5 joint angles, got {len(joint_angles)}")

        # Create joint configuration dictionary
        joint_cfg = {}
        for name, angle in zip(self.joint_names, joint_angles):
            joint_cfg[name] = angle

        # Compute forward kinematics to end-effector link
        # Note: Need to identify the correct end-effector link name from URDF
        fk_result = self.robot.link_fk(cfg=joint_cfg)

        # Get the end-effector transform (typically the last link)
        # This needs to be adjusted based on actual URDF structure
        ee_link_name = self._get_end_effector_link_name()
        transform = fk_result[ee_link_name]

        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]

        return position, rotation_matrix

    def _get_end_effector_link_name(self) -> str:
        """Get the name of the end-effector link.

        Returns:
            Name of the end-effector link.
        """
        # This should be determined from the URDF structure
        # Common names: 'gripper_link', 'tool_link', 'end_effector', etc.
        # For now, return the last link in the chain

        # Find the leaf link (link with no children)
        all_links = set(link.name for link in self.robot.links)
        parent_links = set()

        for joint in self.robot.joints:
            if joint.parent:
                parent_links.add(joint.parent)

        # Leaf links are those that are not parents
        leaf_links = all_links - parent_links

        # If multiple leaf links, prefer one with 'gripper' or 'tool' in name
        for link_name in leaf_links:
            if any(keyword in link_name.lower() for keyword in ['gripper', 'tool', 'end']):
                return link_name

        # Otherwise return any leaf link
        return list(leaf_links)[0] if leaf_links else list(all_links)[-1]

    def compute_task_space(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Compute task space representation (x, y, z, α, ψ).

        Args:
            joint_angles: Array of 5 joint angles [q1, q2, q3, q4, q5].

        Returns:
            Tuple of:
                - position: [x, y, z]
                - alpha: pitch angle (gripper pitch) in radians
                - psi: roll angle (wrist roll) in radians
        """
        position, rotation = self.compute(joint_angles)

        # Extract pitch and roll from rotation matrix
        # Assuming the gripper points along local Z-axis
        # and we want pitch relative to horizontal

        # Local Z-axis direction in world frame
        z_axis = rotation[:, 2]

        # Pitch: angle from horizontal plane
        # alpha = arcsin(-z_axis[2]) for standard orientation
        # This may need adjustment based on actual gripper mounting
        alpha = np.arctan2(-z_axis[2], np.sqrt(z_axis[0]**2 + z_axis[1]**2))

        # Roll: wrist roll (often just q5 for simple wrist)
        # For more complex cases, extract from rotation matrix
        psi = joint_angles[4]  # Wrist roll is typically the 5th joint

        return position, alpha, psi

    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix for numerical IK.

        Args:
            joint_angles: Current joint angles [q1, q2, q3, q4, q5].

        Returns:
            5x5 Jacobian matrix relating joint velocities to task space velocities.
        """
        epsilon = 1e-6
        jacobian = np.zeros((5, 5))

        # Current task space configuration
        pos0, alpha0, psi0 = self.compute_task_space(joint_angles)
        task0 = np.array([pos0[0], pos0[1], pos0[2], alpha0, psi0])

        # Numerical differentiation
        for i in range(5):
            q_plus = joint_angles.copy()
            q_plus[i] += epsilon

            pos_plus, alpha_plus, psi_plus = self.compute_task_space(q_plus)
            task_plus = np.array([pos_plus[0], pos_plus[1], pos_plus[2],
                                  alpha_plus, psi_plus])

            jacobian[:, i] = (task_plus - task0) / epsilon

        return jacobian
