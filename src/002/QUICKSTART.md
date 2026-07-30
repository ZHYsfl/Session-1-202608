# SO-101 Inverse Kinematics - Quick Start Guide

## Installation

1. **Navigate to project directory:**
```bash
cd E:\exp\src\002
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Workflow

### Step 1: Obtain URDF File
Place the SO-101 URDF file at `models/so101.urdf`

You can obtain it from:
- LeRobot repository: `https://github.com/huggingface/lerobot`
- Or check existing robot configuration files

### Step 2: Generate Training Dataset
```bash
python generate_dataset.py \
    --config configs/robot_config.yaml \
    --urdf models/so101.urdf \
    --output data/ik_dataset.h5 \
    --num_samples 150000
```

This will:
- Sample 150K random joint configurations
- Compute forward kinematics
- Generate training labels using analytical IK
- Split into train/val/test sets (70%/15%/15%)
- Save to HDF5 file

**Expected time:** 1-2 hours depending on CPU

### Step 3: Train Neural Network
```bash
python train_network.py \
    --config configs/robot_config.yaml \
    --dataset data/ik_dataset.h5 \
    --output_dir models/ \
    --device cuda \
    --epochs 200
```

Training features:
- Batch size: 256
- Learning rate: 0.001 with ReduceLROnPlateau
- Early stopping (patience=20)
- Multi-component loss (joint + position + orientation)
- Checkpoints every 10 epochs
- Best model saved as `models/ik_mlp_best.pth`

**Expected time:** 2-4 hours on GPU, 12-24 hours on CPU

### Step 4: Run Evaluation
```bash
python run_evaluation.py \
    --config configs/robot_config.yaml \
    --urdf models/so101.urdf \
    --model models/ik_mlp_best.pth \
    --num_test 1000 \
    --output results/ik_comparison.json
```

This benchmarks:
- Analytical IK
- Numerical IK (damped least squares)
- Neural Network IK
- Hybrid IK (NN + 1/3/5 refinement steps)

Test sets:
- Normal: Central workspace
- Boundary: Near reach limits
- Singular: Near singularities
- Noisy: Perturbed inputs

**Expected time:** 30-60 minutes

## Evaluation Metrics

Results include:
- **Success Rate**: % solutions with error < 5mm and < 5°
- **Position Error**: Mean ± std (mm)
- **Orientation Error**: Mean ± std (degrees)
- **Solve Time**: Mean and 95th percentile (ms)
- **Iterations**: For numerical methods
- **Joint Jump**: Max angular change from current pose
- **Limit Violations**: % solutions violating joint limits

## Example Usage

### Using Analytical IK
```python
from kinematics import AnalyticalIK
from utils.config_loader import load_config, get_joint_limits, get_link_lengths
import numpy as np

# Load config
config = load_config('configs/robot_config.yaml')
joint_limits = get_joint_limits(config)
link_lengths = get_link_lengths(config)

# Create solver
ik = AnalyticalIK(
    shoulder_height=link_lengths['shoulder_height'],
    radial_offset=link_lengths['radial_offset'],
    upper_arm_length=link_lengths['upper_arm'],
    forearm_length=link_lengths['forearm'],
    wrist_to_tcp=link_lengths['wrist_to_tcp'],
    joint_limits=joint_limits
)

# Solve IK
target_pos = np.array([0.2, 0.1, 0.15])  # meters
target_alpha = 0.5  # radians (pitch)
target_psi = 0.0    # radians (roll)

solution = ik.solve(target_pos, target_alpha, target_psi)
if solution is not None:
    print("Joint angles:", np.rad2deg(solution))
else:
    print("No solution found")
```

### Using Neural Network IK
```python
from neural_network import load_model, prepare_input
import numpy as np

# Load trained model
model = load_model('models/ik_mlp_best.pth', device='cpu')

# Prepare input
target_pos = np.array([0.2, 0.1, 0.15])
target_alpha = 0.5
target_psi = 0.0
current_joints = np.array([0.0, 0.5, -1.0, 0.5, 0.0])

input_vec = prepare_input(target_pos, target_alpha, target_psi, current_joints)

# Predict
solution = model.predict(input_vec).flatten()
print("Predicted joint angles:", np.rad2deg(solution))
```

### Using Hybrid IK
```python
from kinematics import ForwardKinematics, NumericalIK
from neural_network import load_model, prepare_input
import numpy as np

# Initialize
fk = ForwardKinematics('models/so101.urdf')
numerical_ik = NumericalIK(fk, joint_limits, max_iterations=50)
model = load_model('models/ik_mlp_best.pth')

# Step 1: Neural network prediction
input_vec = prepare_input(target_pos, target_alpha, target_psi, current_joints)
nn_solution = model.predict(input_vec).flatten()

# Step 2: Numerical refinement (3 steps)
refined_solution, iterations = numerical_ik.refine_solution(
    target_pos, target_alpha, target_psi, 
    initial_solution=nn_solution,
    num_steps=3
)

print("Refined joint angles:", np.rad2deg(refined_solution))
print("Refinement iterations:", iterations)
```

## Configuration

Edit `configs/robot_config.yaml` to customize:

### Robot Parameters
```yaml
robot:
  link_lengths:
    shoulder_height: 0.08
    upper_arm: 0.15
    # ... adjust based on actual URDF
  
  joint_limits:
    shoulder_pan: [-3.14, 3.14]
    # ... adjust for safety
```

### Neural Network
```yaml
training:
  model:
    hidden_dims: [256, 256, 128]  # Network size
    dropout: 0.1                   # Regularization
  
  hyperparameters:
    batch_size: 256
    learning_rate: 0.001
    epochs: 200
```

### IK Solvers
```yaml
ik:
  numerical:
    max_iterations: 50
    damping_min: 0.001
    damping_max: 0.1
    position_tolerance: 0.001  # 1mm
  
  hybrid:
    numerical_refinement_steps: 3
```

## Troubleshooting

### URDF Loading Fails
- Check file path is correct
- Verify URDF is well-formed XML
- Check joint names match expected format

### Training Loss Not Decreasing
- Reduce learning rate
- Increase batch size
- Check dataset quality
- Try different random seed

### Low Success Rate in Evaluation
- Verify FK is correct (test with known configurations)
- Check joint limits are properly set
- Increase IK solver iterations
- Refine link length parameters

### Out of Memory During Training
- Reduce batch size
- Use smaller network (e.g., [128, 128])
- Enable gradient checkpointing

## Paper Compilation

After experiments are complete:

```bash
cd E:\exp\paper\002
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX editor (Overleaf, TeXworks, etc.)

## Directory Structure Reference
```
src/002/
├── configs/robot_config.yaml    # Configuration
├── generate_dataset.py          # Step 1 script
├── train_network.py             # Step 2 script
├── run_evaluation.py            # Step 3 script
├── kinematics/                  # IK implementations
├── neural_network/              # NN training
├── utils/                       # Helper functions
├── data/                        # Generated datasets
├── models/                      # Trained models
└── results/                     # Evaluation results
```

## Questions?

- Check PROJECT_STATUS.md for detailed progress
- Review code comments in source files
- Consult SO-101 documentation for hardware details
- Reference the LaTeX paper template for expected results format

---
Last updated: 2026-07-30
