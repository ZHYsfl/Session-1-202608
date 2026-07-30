"""Visualization utilities for IK evaluation results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List


def plot_comparison_metrics(results_path: str, output_dir: str = 'results/figures'):
    """Plot comparison of IK methods across metrics.

    Args:
        results_path: Path to JSON results file.
        output_dir: Directory to save figures.
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style('whitegrid')
    sns.set_palette('husl')

    # Extract method names and metrics
    methods = list(set([r['method_name'] for r in results]))
    test_types = list(set([r.get('test_type', 'normal') for r in results]))

    # Plot 1: Success Rate by Test Type
    fig, ax = plt.subplots(figsize=(10, 6))

    for test_type in test_types:
        test_results = [r for r in results if r.get('test_type') == test_type]
        method_names = [r['method_name'].split('(')[0].strip() for r in test_results]
        success_rates = [r['success_rate'] * 100 for r in test_results]

        x = np.arange(len(method_names))
        ax.bar(x + test_types.index(test_type) * 0.2, success_rates,
               width=0.2, label=test_type)

    ax.set_xlabel('Method')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('IK Success Rate Comparison')
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(output_path / 'success_rate_comparison.png', dpi=300)
    plt.close()

    # Plot 2: Position Error
    fig, ax = plt.subplots(figsize=(10, 6))

    for test_type in test_types:
        test_results = [r for r in results if r.get('test_type') == test_type
                       and r['success_rate'] > 0]
        method_names = [r['method_name'].split('(')[0].strip() for r in test_results]
        pos_errors = [r['position_error_mean'] * 1000 for r in test_results]  # to mm

        x = np.arange(len(method_names))
        ax.bar(x + test_types.index(test_type) * 0.2, pos_errors,
               width=0.2, label=test_type)

    ax.set_xlabel('Method')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('IK Position Error Comparison')
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'position_error_comparison.png', dpi=300)
    plt.close()

    # Plot 3: Solve Time
    fig, ax = plt.subplots(figsize=(10, 6))

    method_times = {}
    for r in results:
        method = r['method_name'].split('(')[0].strip()
        if method not in method_times:
            method_times[method] = []
        method_times[method].append(r['solve_time_mean'] * 1000)  # to ms

    methods = list(method_times.keys())
    mean_times = [np.mean(method_times[m]) for m in methods]

    ax.bar(methods, mean_times)
    ax.set_xlabel('Method')
    ax.set_ylabel('Mean Solve Time (ms)')
    ax.set_title('IK Computation Time Comparison')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'solve_time_comparison.png', dpi=300)
    plt.close()

    # Plot 4: Accuracy vs Speed Tradeoff
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in results:
        if r['success_rate'] > 0:
            method = r['method_name'].split('(')[0].strip()
            test_type = r.get('test_type', 'normal')

            time_ms = r['solve_time_mean'] * 1000
            error_mm = r['position_error_mean'] * 1000

            marker = 'o' if test_type == 'normal' else '^'
            ax.scatter(time_ms, error_mm, s=100, label=f"{method} ({test_type})",
                      marker=marker, alpha=0.7)

    ax.set_xlabel('Solve Time (ms)')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('IK Accuracy vs Speed Tradeoff')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_speed_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {output_path}/")


def plot_training_history(history_path: str, output_path: str = 'results/figures/training_history.png'):
    """Plot training history curves.

    Args:
        history_path: Path to training history JSON file.
        output_path: Output figure path.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    axes[1].plot(epochs, history['learning_rate'], color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Training history plot saved to {output_path}")


def create_results_table(results_path: str, output_path: str = 'results/results_table.txt'):
    """Create formatted text table of results.

    Args:
        results_path: Path to JSON results file.
        output_path: Output table path.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Group by test type
    test_types = list(set([r.get('test_type', 'normal') for r in results]))

    with open(output_path, 'w') as f:
        for test_type in test_types:
            f.write(f"\n{'='*80}\n")
            f.write(f"Test Type: {test_type.upper()}\n")
            f.write(f"{'='*80}\n\n")

            test_results = [r for r in results if r.get('test_type') == test_type]

            # Table header
            f.write(f"{'Method':<30} {'Success':<10} {'Pos Error':<15} {'Time (ms)':<12}\n")
            f.write(f"{'-'*30} {'-'*10} {'-'*15} {'-'*12}\n")

            for r in test_results:
                method = r['method_name'].split('(')[0].strip()
                success = f"{r['success_rate']*100:.1f}%"

                if r['success_rate'] > 0:
                    pos_err = f"{r['position_error_mean']*1000:.2f} mm"
                else:
                    pos_err = "N/A"

                time_ms = f"{r['solve_time_mean']*1000:.2f}"

                f.write(f"{method:<30} {success:<10} {pos_err:<15} {time_ms:<12}\n")

            f.write("\n")

    print(f"Results table saved to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize IK evaluation results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='results/figures',
                       help='Output directory for figures')
    parser.add_argument('--training_history', type=str,
                       help='Path to training history JSON (optional)')

    args = parser.parse_args()

    # Plot comparison metrics
    plot_comparison_metrics(args.results, args.output_dir)

    # Create results table
    create_results_table(args.results)

    # Plot training history if available
    if args.training_history:
        plot_training_history(args.training_history,
                            f"{args.output_dir}/training_history.png")

    print("\nVisualization complete!")
