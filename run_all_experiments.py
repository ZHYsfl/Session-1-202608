"""Validate or run all four imitation-learning experiments in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    (ROOT / "Experiment1_Behavior_Cloning" / "verify_experiment.py", False),
    (ROOT / "Experiment2_DAgger" / "dagger_cartpole.py", True),
    (ROOT / "Experiment4_RoboMimic" / "official_benchmark.py", True),
    (
        ROOT
        / "Experiment3_Performance_Analysis"
        / "analyze_experiments.py",
        False,
    ),
]


def main(quick: bool) -> None:
    for script, supports_quick in SCRIPTS:
        print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
        command = [sys.executable, str(script)]
        if quick and supports_quick:
            command.append("--quick")
        subprocess.run(command, cwd=script.parent, check=True)
    print("\nAll four experiments completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    main(arguments.quick)
