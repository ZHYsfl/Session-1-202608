# Experiment 4: Official RoboMimic Low-Dimensional Benchmark

This experiment uses the official RoboMimic v1.5 proficient-human
low-dimensional HDF5 datasets for:

- Lift;
- Can;
- Square;
- Transport.

It compares three policy classes under one controlled training budget:

- feed-forward BC;
- BC-RNN with an LSTM;
- BC-Transformer with causal sequence context.

The script uses the official HDF5 train and validation trajectory masks,
verifies every dataset by SHA-256, and reports sample-level and
trajectory-balanced action errors. It also reports bootstrap 95% confidence
intervals, phase/component error, parameter count, training time, and CPU
inference latency.

This is an **offline action-prediction benchmark on official data**. It does
not infer simulator rollout success from validation MSE. Exact official paper
scores require the RoboMimic / robosuite / MuJoCo rollout stack and the official
training configurations. The compact models here provide a reproducible,
controlled comparison on Windows.

## Setup

```powershell
.\setup_official_benchmark.ps1
python download_official_datasets.py
```

The delivered directory already contains the four verified HDF5 files. To
verify them without downloading:

```powershell
python download_official_datasets.py --verify-only
```

## Run

```powershell
python official_benchmark.py
```

Outputs:

- `results/official_benchmark_metrics.json`;
- `results/official_training_histories.json`;
- `results/official_checkpoints/*.pt`;
- `../images/exp4_official_*.png`.

`pickplace_bc.py` and its old NPZ artifacts are retained only as the superseded
lightweight prototype; they are not used by the report or the default runner.
