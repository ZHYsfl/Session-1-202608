"""Run Experiments 2, 4, and 3 in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "Experiment2_DAgger" / "dagger_cartpole.py",
    ROOT / "Experiment4_RoboMimic" / "official_benchmark.py",
    ROOT / "Experiment3_Performance_Analysis" / "analyze_experiments.py",
]


def main(quick: bool) -> None:
    for script in SCRIPTS:
        print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
        command = [sys.executable, str(script)]
        if quick and script.parent.name != "Experiment3_Performance_Analysis":
            command.append("--quick")
        subprocess.run(command, cwd=script.parent, check=True)
    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    main(arguments.quick)
