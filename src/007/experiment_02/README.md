$experiment = "D:\\暑期科研实践2026.7\\robot\_collision\_prediction\\github\_007\_upload\\src\\007\\experiment\_02"



@'

\# Experiment 02: Robot Collision Prediction and Avoidance



\## Overview



This experiment implements robot collision prediction and collision avoidance in a Pygame simulation environment.



The project includes:



\- Multi-directional distance sensors

\- Collision data collection

\- MLP-based future collision prediction

\- DQN-based collision avoidance

\- Deceleration, specular reflection, and acceleration control

\- Blind-test evaluation and visualization



\## Main files



\- `main.py`: Pygame simulation and data collection

\- `make\_labels.py`: generates future collision labels

\- `train\_model.py`: trains the MLP collision prediction model

\- `robot\_env\_v3.py`: final V3 reinforcement-learning environment

\- `train\_dqn\_v3.py`: trains the final V3 DQN model

\- `evaluate\_final\_models.py`: evaluates the trained policies

\- `show\_dqn\_v3.py`: visualizes the final collision-avoidance policy



\## Final model



The final collision-avoidance model is the V3 DQN policy.



\- Observation dimension: 14

\- Action space: 2

\- Action 0: maintain the current direction

\- Action 1: decelerate, reflect, and accelerate

\- Best checkpoint: `best\_v3\_at\_30000\_steps`



Blind-test results over 500 episodes:



\- Collision rate: 1.20%

\- Survival rate: 98.80%

\- Mean speed: 148.45

\- Mean episode length: 597.63 / 600



\## Environment setup



Create a virtual environment:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

