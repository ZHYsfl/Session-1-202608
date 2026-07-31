"""Script to train neural network for inverse kinematics."""

import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

# project root src/002 is three levels up: scripts/training/<this file>
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from neural_network import IKNet, IKDataset, IKTrainer
from utils.config_loader import load_config, get_joint_limits


def main():
    parser = argparse.ArgumentParser(description='Train IK neural network')
    parser.add_argument('--config', type=str, default='configs/robot_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='data/ik_dataset.h5',
                       help='Path to dataset file')
    parser.add_argument('--output_dir', type=str, default='models/',
                       help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    joint_limits = get_joint_limits(config, use_soft_limits=True)

    # Load datasets
    print(f"Loading dataset from {args.dataset}...")
    train_dataset = IKDataset(args.dataset, split='train')
    val_dataset = IKDataset(args.dataset, split='val')

    # Create data loaders
    batch_size = config['training']['hyperparameters']['batch_size']
    # Windows下num_workers=0避免多进程DataLoader卡死
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(args.device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device == 'cuda')
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")

    # Create model
    print("Creating neural network model...")
    model_config = config['training']['model']
    model = IKNet(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        output_dim=model_config['output_dim'],
        activation=model_config['activation'],
        use_batch_norm=model_config['use_batch_norm'],
        dropout=model_config['dropout'],
        joint_limits=joint_limits
    )

    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = IKTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )

    # Train
    num_epochs = args.epochs if args.epochs else config['training']['hyperparameters']['epochs']
    print(f"\nStarting training for {num_epochs} epochs on {args.device}...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.train(num_epochs=num_epochs, save_dir=str(output_dir))

    print("\nTraining completed!")
    print(f"Best model saved to: {output_dir / 'ik_mlp_best.pth'}")


if __name__ == '__main__':
    main()
