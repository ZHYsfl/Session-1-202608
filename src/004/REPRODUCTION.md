# Four-Experiment Reproduction Guide

## Integrated Scope

The repository now contains four connected experiments:

1. Experiment 1 validates the retained CNN behavior-cloning source, trained
   `model.h5`, recorded metrics, and figures. The original driving images are
   not stored in the repository, so the integrated command does not claim to
   retrain this model.
2. Experiment 2 retrains BC and DAgger on CartPole, including beta ablations.
3. Experiment 3 reads machine-readable outputs from Experiments 1, 2, and 4.
4. Experiment 4 retrains BC, BC-RNN, and BC-Transformer on four official
   RoboMimic low-dimensional datasets.

## Environment

The completed integrated run used:

- Windows 11;
- Python 3.12.13 for artifact verification and Experiments 2-4;
- NumPy 2.3.5, Matplotlib 3.11.1, scikit-learn 1.9.0;
- PyTorch 2.12.1 CPU and h5py 3.15.1;
- fixed Experiment 2/4 random seed `20260730`.

The retained Experiment 1 model was trained with TensorFlow 2.15.0. Its full
dependency snapshot is in `Experiment1_Behavior_Cloning/requirements.txt`.

Install Experiment 2 and 3 dependencies:

```powershell
python -m pip install -r Experiment2_DAgger\requirements.txt
```

Install the isolated Experiment 4 dependencies:

```powershell
.\Experiment4_RoboMimic\setup_official_benchmark.ps1
```

The official HDF5 files are delivered under
`Experiment4_RoboMimic\data\official`. If absent, download and verify them:

```powershell
python Experiment4_RoboMimic\download_official_datasets.py
```

## One-Command Run

From the project root:

```powershell
.\run_all_experiments.ps1
```

The runner validates Experiment 1, executes Experiment 2, executes the official
Experiment 4 offline benchmark, then regenerates Experiment 3. It prefers the
bundled Python 3.12 runtime on this machine so the delivered PyTorch
dependencies use the matching ABI.

A smoke test is available:

```powershell
.\run_all_experiments.ps1 -Quick
```

The previous `run_experiments_2_3_4.ps1` remains available for retraining only
Experiments 2-4.

## Experiment 1 Full Retraining

Full retraining requires `driving_log.csv` and the simulator `IMG` directory
under `Experiment1_Behavior_Cloning/data`. With that data restored:

```powershell
cd Experiment1_Behavior_Cloning\behavioral-cloning
python model.py
```

## Artifact Policy

- `Experiment1_Behavior_Cloning/results/metrics.json` records the verified
  completed Experiment 1.
- Other `results/*.json` files are machine-readable experiment outputs.
- `results/*.joblib`, `behavioral-cloning/model.h5`, and
  `results/official_checkpoints/*.pt` are trained models.
- `data/official/*.hdf5` are official RoboMimic v1.5 PH datasets.
- `images/exp1_*.png` through `images/exp4_official_*.png` are report figures.
- The original `reports.md` is intentionally not edited.
- `reports_completed.md` is the integrated four-experiment report.
