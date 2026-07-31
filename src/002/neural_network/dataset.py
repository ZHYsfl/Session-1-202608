"""Dataset class for IK training."""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from typing import Dict, Optional


class IKDataset(Dataset):
    """Dataset for inverse kinematics training.

    Data format:
        - input: [x, y, z, sin(α), cos(α), sin(ψ), cos(ψ), q_current_1:5]
        - target_joints: [q_target_1:5]
        - position: [x, y, z] (for evaluation)
        - orientation: [α, ψ] (for evaluation)
    """

    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 transform: Optional[callable] = None):
        """Initialize dataset.

        Args:
            data_path: Path to HDF5 data file.
            split: Dataset split ('train', 'val', 'test').
            transform: Optional transform to apply to data.
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform

        # Load data
        with h5py.File(data_path, 'r') as f:
            split_group = f[split]

            self.inputs = split_group['inputs'][:]
            self.target_joints = split_group['target_joints'][:]
            self.positions = split_group['positions'][:]
            self.orientations = split_group['orientations'][:]

        print(f"Loaded {split} dataset: {len(self)} samples")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - input: Network input tensor
                - target_joints: Target joint angles
                - position: Target position
                - orientation: Target orientation
        """
        sample = {
            'input': torch.tensor(self.inputs[idx], dtype=torch.float32),
            'target_joints': torch.tensor(self.target_joints[idx], dtype=torch.float32),
            'position': torch.tensor(self.positions[idx], dtype=torch.float32),
            'orientation': torch.tensor(self.orientations[idx], dtype=torch.float32)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class IKDatasetGenerator:
    """Generate training dataset for IK neural network."""

    def __init__(self,
                 fk_solver,
                 ik_solver,
                 joint_limits: tuple,
                 num_samples: int = 100000):
        """Initialize dataset generator.

        Args:
            fk_solver: Forward kinematics solver.
            ik_solver: Inverse kinematics solver (for label generation).
            joint_limits: Tuple of (lower_limits, upper_limits).
            num_samples: Number of samples to generate.
        """
        self.fk = fk_solver
        self.ik = ik_solver
        self.lower_limits = np.array(joint_limits[0])
        self.upper_limits = np.array(joint_limits[1])
        self.num_samples = num_samples

    def generate(self,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 current_joint_noise: float = 0.05,
                 seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate dataset.

        Args:
            train_ratio: Ratio of training samples.
            val_ratio: Ratio of validation samples.
            test_ratio: Ratio of test samples.
            current_joint_noise: Noise level for current joint angles.
            seed: Random seed.

        Returns:
            Dictionary with 'train', 'val', 'test' splits, each containing
            'inputs', 'target_joints', 'positions', 'orientations'.
        """
        np.random.seed(seed)

        print(f"Generating {self.num_samples} samples...")

        inputs = []
        target_joints = []
        positions = []
        orientations = []

        num_generated = 0
        num_attempts = 0
        max_attempts = self.num_samples * 10

        while num_generated < self.num_samples and num_attempts < max_attempts:
            num_attempts += 1

            # Random joint configuration
            q_sample = np.random.uniform(self.lower_limits, self.upper_limits)

            # Compute forward kinematics
            try:
                pos, alpha, psi = self.fk.compute_task_space(q_sample)
            except Exception as e:
                continue

            # Generate perturbed current joint angles
            q_current = q_sample + np.random.normal(0, current_joint_noise, size=5)
            q_current = np.clip(q_current, self.lower_limits, self.upper_limits)

            # Solve IK from perturbed state to find target
            q_target = self.ik.choose_closest_solution(pos, alpha, psi, q_current)

            if q_target is None:
                continue

            # Prepare input: [x, y, z, sin(α), cos(α), sin(ψ), cos(ψ), q_current]
            input_vec = np.array([
                pos[0], pos[1], pos[2],
                np.sin(alpha), np.cos(alpha),
                np.sin(psi), np.cos(psi),
                q_current[0], q_current[1], q_current[2],
                q_current[3], q_current[4]
            ])

            inputs.append(input_vec)
            target_joints.append(q_target)
            positions.append(pos)
            orientations.append([alpha, psi])

            num_generated += 1

            if num_generated % 10000 == 0:
                print(f"  Generated {num_generated}/{self.num_samples} samples")

        print(f"Dataset generation completed: {num_generated} samples "
              f"({num_attempts} attempts)")

        # Convert to arrays
        inputs = np.array(inputs)
        target_joints = np.array(target_joints)
        positions = np.array(positions)
        orientations = np.array(orientations)

        # Split dataset
        n = len(inputs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        dataset = {
            'train': {
                'inputs': inputs[train_idx],
                'target_joints': target_joints[train_idx],
                'positions': positions[train_idx],
                'orientations': orientations[train_idx]
            },
            'val': {
                'inputs': inputs[val_idx],
                'target_joints': target_joints[val_idx],
                'positions': positions[val_idx],
                'orientations': orientations[val_idx]
            },
            'test': {
                'inputs': inputs[test_idx],
                'target_joints': target_joints[test_idx],
                'positions': positions[test_idx],
                'orientations': orientations[test_idx]
            }
        }

        return dataset

    def save_to_hdf5(self, dataset: Dict, output_path: str):
        """Save dataset to HDF5 file.

        Args:
            dataset: Dataset dictionary from generate().
            output_path: Path to output HDF5 file.
        """
        with h5py.File(output_path, 'w') as f:
            for split_name, split_data in dataset.items():
                split_group = f.create_group(split_name)

                for key, value in split_data.items():
                    split_group.create_dataset(key, data=value,
                                              compression='gzip')

        print(f"Dataset saved to {output_path}")
