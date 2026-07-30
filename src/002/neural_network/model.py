"""Neural network model for inverse kinematics."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class IKNet(nn.Module):
    """MLP network for inverse kinematics prediction.

    Input: [x, y, z, sin(α), cos(α), sin(ψ), cos(ψ), q_current_1:5]
    Output: [q_target_1:5]
    """

    def __init__(self,
                 input_dim: int = 12,
                 hidden_dims: list = [256, 256, 128],
                 output_dim: int = 5,
                 activation: str = "relu",
                 use_batch_norm: bool = True,
                 dropout: float = 0.1,
                 joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Initialize IK neural network.

        Args:
            input_dim: Input dimension (default 12).
            hidden_dims: List of hidden layer dimensions.
            output_dim: Output dimension (5 joints).
            activation: Activation function ('relu', 'tanh', 'elu').
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout rate.
            joint_limits: Tuple of (lower_limits, upper_limits) for output scaling.
        """
        super(IKNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Store joint limits for output scaling
        if joint_limits is not None:
            self.register_buffer('lower_limits',
                               torch.tensor(joint_limits[0], dtype=torch.float32))
            self.register_buffer('upper_limits',
                               torch.tensor(joint_limits[1], dtype=torch.float32))
        else:
            # Default limits
            self.register_buffer('lower_limits',
                               torch.tensor([-3.14, -1.57, -2.35, -1.57, -3.14]))
            self.register_buffer('upper_limits',
                               torch.tensor([3.14, 1.57, 2.35, 1.57, 3.14]))

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output joint angles of shape (batch_size, output_dim).
        """
        # Network output in [-1, 1]
        output = self.network(x)

        # Scale to joint limits
        # q = (upper + lower)/2 + (upper - lower)/2 * tanh_output
        mid = (self.upper_limits + self.lower_limits) / 2
        scale = (self.upper_limits - self.lower_limits) / 2

        joint_angles = mid + scale * output

        return joint_angles

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict joint angles from numpy input.

        Args:
            x: Input array of shape (input_dim,) or (batch_size, input_dim).

        Returns:
            Predicted joint angles as numpy array.
        """
        self.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x = x.reshape(1, -1)

            x_tensor = torch.tensor(x, dtype=torch.float32)
            if next(self.parameters()).is_cuda:
                x_tensor = x_tensor.cuda()

            output = self.forward(x_tensor)
            return output.cpu().numpy()


def prepare_input(position: np.ndarray,
                 alpha: float,
                 psi: float,
                 current_joints: np.ndarray) -> np.ndarray:
    """Prepare input vector for the neural network.

    Args:
        position: [x, y, z]
        alpha: Pitch angle in radians.
        psi: Roll angle in radians.
        current_joints: Current joint angles [q1, q2, q3, q4, q5].

    Returns:
        Input vector of shape (12,).
    """
    return np.array([
        position[0], position[1], position[2],
        np.sin(alpha), np.cos(alpha),
        np.sin(psi), np.cos(psi),
        current_joints[0], current_joints[1], current_joints[2],
        current_joints[4], current_joints[4]
    ])


class IKLoss(nn.Module):
    """Multi-component loss function for IK training."""

    def __init__(self,
                 joint_weight: float = 1.0,
                 position_weight: float = 10.0,
                 orientation_weight: float = 5.0,
                 joint_limit_margin_weight: float = 0.1):
        """Initialize loss function.

        Args:
            joint_weight: Weight for joint angle loss.
            position_weight: Weight for position error (from FK).
            orientation_weight: Weight for orientation error.
            joint_limit_margin_weight: Weight for joint limit margin penalty.
        """
        super(IKLoss, self).__init__()
        self.joint_weight = joint_weight
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.joint_limit_margin_weight = joint_limit_margin_weight

    def forward(self,
                pred_joints: torch.Tensor,
                target_joints: torch.Tensor,
                pred_pos: Optional[torch.Tensor] = None,
                target_pos: Optional[torch.Tensor] = None,
                pred_orientation: Optional[torch.Tensor] = None,
                target_orientation: Optional[torch.Tensor] = None,
                joint_limits: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """Compute multi-component loss.

        Args:
            pred_joints: Predicted joint angles (batch_size, 5).
            target_joints: Target joint angles (batch_size, 5).
            pred_pos: Predicted position from FK (batch_size, 3).
            target_pos: Target position (batch_size, 3).
            pred_orientation: Predicted orientation (batch_size, 2) [alpha, psi].
            target_orientation: Target orientation (batch_size, 2).
            joint_limits: Tuple of (lower, upper) limits.

        Returns:
            Total loss.
        """
        # Joint angle loss
        joint_loss = nn.functional.mse_loss(pred_joints, target_joints)
        total_loss = self.joint_weight * joint_loss

        # Position loss (if FK is available)
        if pred_pos is not None and target_pos is not None:
            pos_loss = nn.functional.mse_loss(pred_pos, target_pos)
            total_loss += self.position_weight * pos_loss

        # Orientation loss
        if pred_orientation is not None and target_orientation is not None:
            ori_loss = nn.functional.mse_loss(pred_orientation, target_orientation)
            total_loss += self.orientation_weight * ori_loss

        # Joint limit margin penalty
        if joint_limits is not None and self.joint_limit_margin_weight > 0:
            lower, upper = joint_limits
            margin = 0.1  # 0.1 radian margin

            # Penalty for being too close to limits
            lower_violation = torch.relu(lower + margin - pred_joints)
            upper_violation = torch.relu(pred_joints - (upper - margin))

            limit_penalty = torch.mean(lower_violation**2 + upper_violation**2)
            total_loss += self.joint_limit_margin_weight * limit_penalty

        return total_loss
