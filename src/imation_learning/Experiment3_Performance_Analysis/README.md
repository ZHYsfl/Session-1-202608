# Experiment 3: Performance Analysis

This experiment reads measured artifacts from Experiments 1, 2, and 4. It does
not train a separate policy.

The analysis produces:

- a cross-experiment summary;
- CartPole state-coverage gains using 1st-to-99th percentile widths;
- an `(x, theta)` policy-state plot with expert, BC, and DAgger samples;
- BC vs DAgger action-probability MSE and action disagreement;
- a four-task summary of the official RoboMimic offline benchmark;
- `results/analysis_summary.json` with all derived values.

Experiment 1 is read from
`../Experiment1_Behavior_Cloning/results/metrics.json`. Run its artifact
verification before this analysis.

## Run

Run Experiments 2 and 4 first, then:

```powershell
python analyze_experiments.py
```
