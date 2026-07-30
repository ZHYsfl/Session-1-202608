"""Training script for IK neural network."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
from tqdm import tqdm

from .model import IKNet, IKLoss
from .dataset import IKDataset


class IKTrainer:
    """Trainer for IK neural network."""

    def __init__(self,
                 model: IKNet,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize trainer.

        Args:
            model: IK neural network.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration dictionary.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        loss_config = config['training']['loss']
        self.criterion = IKLoss(
            joint_weight=loss_config['joint_weight'],
            position_weight=loss_config['position_weight'],
            orientation_weight=loss_config['orientation_weight'],
            joint_limit_margin_weight=loss_config['joint_limit_margin_weight']
        )

        # Optimizer
        hp = config['training']['hyperparameters']
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=hp['learning_rate']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.early_stopping_patience = hp['early_stopping_patience']

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}'):
            # Move to device
            inputs = batch['input'].to(self.device)
            target_joints = batch['target_joints'].to(self.device)

            # Forward pass
            pred_joints = self.model(inputs)

            # Compute loss
            loss = self.criterion(
                pred_joints=pred_joints,
                target_joints=target_joints,
                joint_limits=(self.model.lower_limits, self.model.upper_limits)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate the model.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                target_joints = batch['target_joints'].to(self.device)

                pred_joints = self.model(inputs)

                loss = self.criterion(
                    pred_joints=pred_joints,
                    target_joints=target_joints,
                    joint_limits=(self.model.lower_limits, self.model.upper_limits)
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, num_epochs: int, save_dir: str):
        """Train the model.

        Args:
            num_epochs: Number of epochs to train.
            save_dir: Directory to save checkpoints.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"Training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, save_path / 'ik_mlp_best.pth')

                print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

            else:
                self.early_stopping_counter += 1

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path / f'checkpoint_epoch_{epoch+1}.pth')

            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")


def load_model(checkpoint_path: str, device: str = 'cpu') -> IKNet:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Loaded IK model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Create model
    model_config = config['training']['model']
    joint_limits = None  # Will be set from config if available

    model = IKNet(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        output_dim=model_config['output_dim'],
        activation=model_config['activation'],
        use_batch_norm=model_config['use_batch_norm'],
        dropout=model_config['dropout'],
        joint_limits=joint_limits
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
