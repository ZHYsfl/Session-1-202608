"""Script to generate training dataset for neural network IK."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from kinematics import ForwardKinematics, AnalyticalIK
from neural_network import IKDatasetGenerator
from utils.config_loader import load_config, get_joint_limits, get_link_lengths


def main():
    parser = argparse.ArgumentParser(description='Generate IK training dataset')
    parser.add_argument('--config', type=str, default='configs/robot_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--urdf', type=str, default='models/so101.urdf',
                       help='Path to URDF file')
    parser.add_argument('--output', type=str, default='data/ik_dataset.h5',
                       help='Output path for dataset')
    parser.add_argument('--num_samples', type=int, default=150000,
                       help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    joint_limits = get_joint_limits(config, use_soft_limits=True)
    link_lengths = get_link_lengths(config)

    # Initialize forward kinematics
    print(f"Loading URDF from {args.urdf}...")
    try:
        fk = ForwardKinematics(args.urdf)
    except Exception as e:
        print(f"Error loading URDF: {e}")
        return

    # Initialize analytical IK for label generation
    print("Initializing analytical IK for label generation...")
    analytical_ik = AnalyticalIK(
        shoulder_height=link_lengths['shoulder_height'],
        radial_offset=link_lengths['radial_offset'],
        upper_arm_length=link_lengths['upper_arm'],
        forearm_length=link_lengths['forearm'],
        wrist_to_tcp=link_lengths['wrist_to_tcp'],
        joint_limits=joint_limits
    )

    # Initialize dataset generator
    generator = IKDatasetGenerator(
        fk_solver=fk,
        ik_solver=analytical_ik,
        joint_limits=joint_limits,
        num_samples=args.num_samples
    )

    # Generate dataset
    dataset_config = config['training']['dataset']
    dataset = generator.generate(
        train_ratio=dataset_config['train_ratio'],
        val_ratio=dataset_config['val_ratio'],
        test_ratio=dataset_config['test_ratio'],
        current_joint_noise=dataset_config['augmentation']['current_joint_noise'],
        seed=args.seed
    )

    # Save to HDF5
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_to_hdf5(dataset, str(output_path))

    print("\nDataset generation completed successfully!")
    print(f"Dataset saved to: {args.output}")
    print(f"Train samples: {len(dataset['train']['inputs'])}")
    print(f"Val samples: {len(dataset['val']['inputs'])}")
    print(f"Test samples: {len(dataset['test']['inputs'])}")


if __name__ == '__main__':
    main()
