# Experiment 2: DAgger on CartPole

This experiment compares offline Behavior Cloning (BC) with Dataset
Aggregation (DAgger) under a widened CartPole reset distribution. The dynamics
are included in the script, so Gymnasium is not required.

The full run includes:

- a fixed BC baseline and a six-iteration DAgger learning curve;
- beta-decay ablations for `beta_k = lambda^k`, with
  `lambda in {0.9, 0.5, 0.1}`;
- action-probability MSE and action disagreement on independent
  expert-, BC-, and DAgger-visited states;
- state-distribution and matched-seed rollout visualizations.

## Run

```powershell
python dagger_cartpole.py
```

Outputs:

- `results/metrics.json`: configuration, curves, ablations, and error metrics;
- `results/bc_policy.joblib` and `results/dagger_policy.joblib`;
- `results/state_coverage.npz`;
- `results/policy_state_distributions.npz`;
- `../images/exp2_*.png`.

All random seeds and evaluation episode seeds are fixed.
