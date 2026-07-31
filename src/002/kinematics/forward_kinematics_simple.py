"""Simplified forward kinematics using URDF (without mesh loading)."""

import numpy as np
from typing import Tuple
import xml.etree.ElementTree as ET


class ForwardKinematics:
    """Forward kinematics for SO-101 robot using URDF (simplified).

    This version parses URDF for joint/link structure without loading meshes.
    """

    def __init__(self, urdf_path: str):
        """Initialize forward kinematics with URDF model.

        Args:
            urdf_path: Path to the URDF file.
        """
        self.urdf_path = urdf_path
        self.joints_info = {}
        self.links_info = {}

        # Parse URDF manually
        self._parse_urdf(urdf_path)

        # Kinematic chain order from base_link to end-effector.
        # Derived from URDF parent/child links:
        #   base_link -> shoulder_pan -> shoulder_lift -> elbow_flex
        #             -> wrist_flex -> wrist_roll -> (gripper, not part of IK)
        # NOTE: dict insertion order in the URDF is reversed and includes the
        # gripper, so we must NOT use list(self.joints_info.keys())[:5].
        self.joint_names = self._build_chain_order()

        # 末端工具偏移: gripper_frame_joint (fixed) 把坐标系从 gripper_link
        # 原点延伸到真正的把手尖端。这段长度约98mm, 之前漏算导致把手砸桌!
        # 从URDF解析该fixed joint的origin。
        self.tool_offset_xyz, self.tool_offset_rpy = self._parse_tool_offset(urdf_path)

        print(f"✓ 加载URDF: {urdf_path}")
        print(f"  发现{len(self.joint_names)}个关节 (基座→末端): {self.joint_names}")

    def _parse_tool_offset(self, urdf_path):
        """Parse the fixed gripper_frame_joint to get the tool-tip offset.

        This offset (~98mm along the gripper) extends the end-effector frame
        from the gripper_link origin to the physical tool tip. Missing this
        caused the tool to hit the table.

        Returns:
            (xyz, rpy) offset as numpy arrays. Zero offset if not found.
        """
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for joint in root.findall('joint'):
                if joint.get('name') == 'gripper_frame_joint':
                    origin = joint.find('origin')
                    if origin is not None:
                        if origin.get('xyz'):
                            xyz = np.array([float(v) for v in origin.get('xyz').split()])
                        if origin.get('rpy'):
                            rpy = np.array([float(v) for v in origin.get('rpy').split()])
                    break
        except Exception:
            pass
        print(f"  末端工具偏移 (把手尖端): {xyz} (长度 {np.linalg.norm(xyz)*1000:.0f}mm)")
        return xyz, rpy

    def _build_chain_order(self):
        """Build the ordered list of 5 IK joints from base to end-effector.

        Follows parent/child links starting at ``base_link`` and stops before
        the gripper joint (which is not part of the 5-DOF arm).
        """
        # Map child_link -> joint that produces it, and parent link per joint
        child_to_joint = {info['child']: name
                          for name, info in self.joints_info.items()}
        joint_parent = {name: info['parent']
                        for name, info in self.joints_info.items()}

        # Walk from base_link downward
        chain = []
        current_link = 'base_link'
        while True:
            # Find the joint whose parent is current_link
            next_joint = None
            for name, parent in joint_parent.items():
                if parent == current_link:
                    next_joint = name
                    break
            if next_joint is None:
                break
            # Stop before the gripper (it drives the moving jaw, not the arm)
            if next_joint == 'gripper':
                break
            chain.append(next_joint)
            current_link = self.joints_info[next_joint]['child']
            if len(chain) >= 5:
                break

        # Fallback: if the walk failed (unexpected URDF), use a hard-coded order
        if len(chain) != 5:
            expected = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                        'wrist_flex', 'wrist_roll']
            chain = [j for j in expected if j in self.joints_info]

        return chain

    def _parse_urdf(self, urdf_path: str):
        """Parse URDF file to extract joint and link information."""
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Parse joints
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')

            if joint_type in ['revolute', 'continuous']:
                # Get origin
                origin = joint.find('origin')
                xyz = [0, 0, 0]
                rpy = [0, 0, 0]
                if origin is not None:
                    if origin.get('xyz'):
                        xyz = [float(x) for x in origin.get('xyz').split()]
                    if origin.get('rpy'):
                        rpy = [float(x) for x in origin.get('rpy').split()]

                # Get axis
                axis_elem = joint.find('axis')
                axis = [1, 0, 0]  # default
                if axis_elem is not None and axis_elem.get('xyz'):
                    axis = [float(x) for x in axis_elem.get('xyz').split()]

                # Get limits
                limit = joint.find('limit')
                lower = -3.14
                upper = 3.14
                if limit is not None:
                    lower = float(limit.get('lower', -3.14))
                    upper = float(limit.get('upper', 3.14))

                self.joints_info[joint_name] = {
                    'type': joint_type,
                    'xyz': xyz,
                    'rpy': rpy,
                    'axis': axis,
                    'lower': lower,
                    'upper': upper,
                    'parent': joint.find('parent').get('link') if joint.find('parent') is not None else None,
                    'child': joint.find('child').get('link') if joint.find('child') is not None else None,
                }

    def compute(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics using DH-like approach.

        Args:
            joint_angles: Array of 5 joint angles in radians [q1, q2, q3, q4, q5].

        Returns:
            Tuple of:
                - position: 3D position [x, y, z] in meters
                - rotation_matrix: 3x3 rotation matrix
        """
        if len(joint_angles) != 5:
            raise ValueError(f"Expected 5 joint angles, got {len(joint_angles)}")

        # Build transformation matrices
        T = np.eye(4)

        for i, (joint_name, angle) in enumerate(zip(self.joint_names, joint_angles)):
            joint_info = self.joints_info[joint_name]

            # Translation from origin
            xyz = joint_info['xyz']
            trans = np.eye(4)
            trans[:3, 3] = xyz

            # Rotation from RPY
            rpy = joint_info['rpy']
            R_fixed = self._rpy_to_matrix(rpy)
            rot_fixed = np.eye(4)
            rot_fixed[:3, :3] = R_fixed

            # Rotation around axis by joint angle
            axis = joint_info['axis']
            R_joint = self._axis_angle_to_matrix(axis, angle)
            rot_joint = np.eye(4)
            rot_joint[:3, :3] = R_joint

            # Combine transformations
            T = T @ trans @ rot_fixed @ rot_joint

        # 追加末端工具偏移: 从gripper_link原点延伸到真正的把手尖端(~98mm)
        # 这一步至关重要, 否则FK算的是关节位置而非把手实际位置, 会导致撞桌
        tool_trans = np.eye(4)
        tool_trans[:3, :3] = self._rpy_to_matrix(self.tool_offset_rpy)
        tool_trans[:3, 3] = self.tool_offset_xyz
        T = T @ tool_trans

        position = T[:3, 3]
        rotation_matrix = T[:3, :3]

        return position, rotation_matrix

    def _rpy_to_matrix(self, rpy):
        """Convert roll-pitch-yaw to rotation matrix."""
        r, p, y = rpy

        # Roll (X)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)]
        ])

        # Pitch (Y)
        Ry = np.array([
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)]
        ])

        # Yaw (Z)
        Rz = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    def _axis_angle_to_matrix(self, axis, angle):
        """Convert axis-angle to rotation matrix using Rodrigues' formula."""
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)  # normalize

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

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

        # Extract pitch from rotation matrix
        # Assuming z-axis is the tool direction
        z_axis = rotation[:, 2]
        alpha = np.arctan2(-z_axis[2], np.sqrt(z_axis[0]**2 + z_axis[1]**2))

        # Roll is typically just the last joint
        psi = joint_angles[4]

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
