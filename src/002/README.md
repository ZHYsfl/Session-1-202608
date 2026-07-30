# SO-101 Inverse Kinematics Project

## Project Overview
Implementation and comparison of inverse kinematics methods for the LeRobot SO-101 robot arm:
- Analytical IK (simplified)
- Damped Least Squares numerical IK
- Neural Network IK
- Hybrid (NN + numerical refinement)

## Robot Structure
SO-101 has 6 motors:
1. Shoulder Pan (base rotation)
2. Shoulder Lift
3. Elbow Flex
4. Wrist Flex
5. Wrist Roll
6. Gripper (not included in IK)

Task space: (x, y, z, α, ψ) - position + pitch + roll (5 DOF)

## Directory Structure
```
src/002/
├── kinematics/          # Forward and inverse kinematics
├── neural_network/      # MLP implementation and training
├── data/               # Dataset generation and storage
├── simulation/         # MuJoCo simulation
├── robot_control/      # Real robot interface
├── utils/              # Helper functions
├── tests/              # Unit tests
├── models/             # Trained models
├── configs/            # Configuration files
└── results/            # Experiment results
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage
See individual module documentation.
