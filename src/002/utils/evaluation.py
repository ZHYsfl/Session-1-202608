"""Evaluation metrics and benchmarking for IK methods."""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class IKSolution:
    """Container for IK solution results."""
    joint_angles: Optional[np.ndarray]
    success: bool
    solve_time: float
    num_iterations: int = 0
    position_error: float = float('inf')
    orientation_error: float = float('inf')
    joint_limit_violation: bool = False
    max_joint_jump: float = 0.0


class IKEvaluator:
    """Evaluator for comparing IK methods."""

    def __init__(self, fk_solver, joint_limits: Tuple[np.ndarray, np.ndarray]):
        """Initialize evaluator.

        Args:
            fk_solver: Forward kinematics solver for error computation.
            joint_limits: Tuple of (lower_limits, upper_limits).
        """
        self.fk = fk_solver
        self.lower_limits = np.array(joint_limits[0])
        self.upper_limits = np.array(joint_limits[1])

    def evaluate_solution(self,
                         solution: Optional[np.ndarray],
                         target_pos: np.ndarray,
                         target_alpha: float,
                         target_psi: float,
                         current_joints: Optional[np.ndarray] = None,
                         solve_time: float = 0.0,
                         num_iterations: int = 0) -> IKSolution:
        """Evaluate a single IK solution.

        Args:
            solution: Computed joint angles or None if failed.
            target_pos: Target position.
            target_alpha: Target pitch.
            target_psi: Target roll.
            current_joints: Current joint configuration (for jump detection).
            solve_time: Time taken to compute solution.
            num_iterations: Number of iterations (for numerical methods).

        Returns:
            IKSolution object with all metrics.
        """
        if solution is None:
            return IKSolution(
                joint_angles=None,
                success=False,
                solve_time=solve_time,
                num_iterations=num_iterations
            )

        # Check joint limits
        limit_violation = not (np.all(solution >= self.lower_limits) and
                              np.all(solution <= self.upper_limits))

        # Compute forward kinematics
        try:
            achieved_pos, achieved_alpha, achieved_psi = self.fk.compute_task_space(solution)

            # Position error
            pos_error = np.linalg.norm(achieved_pos - target_pos)

            # Orientation error
            ori_error = np.sqrt((achieved_alpha - target_alpha)**2 +
                               (achieved_psi - target_psi)**2)

            # Success criteria: < 5mm position error and < 5 degrees orientation error
            success = (pos_error < 0.005) and (ori_error < np.deg2rad(5)) and not limit_violation

        except Exception as e:
            pos_error = float('inf')
            ori_error = float('inf')
            success = False

        # Joint jump (if current state provided)
        max_joint_jump = 0.0
        if current_joints is not None:
            max_joint_jump = np.max(np.abs(solution - current_joints))

        return IKSolution(
            joint_angles=solution,
            success=success,
            solve_time=solve_time,
            num_iterations=num_iterations,
            position_error=pos_error,
            orientation_error=ori_error,
            joint_limit_violation=limit_violation,
            max_joint_jump=max_joint_jump
        )

    def benchmark_method(self,
                        ik_method: Callable,
                        test_cases: List[Dict],
                        method_name: str = "IK Method") -> Dict:
        """Benchmark an IK method on a set of test cases.

        Args:
            ik_method: Callable that takes (pos, alpha, psi, current_joints) and
                      returns (solution, num_iterations) or just solution.
            test_cases: List of dicts with 'position', 'alpha', 'psi', 'current_joints'.
            method_name: Name of the method for reporting.

        Returns:
            Dictionary with aggregated metrics.
        """
        print(f"\nEvaluating {method_name} on {len(test_cases)} test cases...")

        results = []

        for i, test_case in enumerate(test_cases):
            target_pos = test_case['position']
            target_alpha = test_case['alpha']
            target_psi = test_case['psi']
            current_joints = test_case.get('current_joints', None)

            # Time the method
            start_time = time.time()
            try:
                result = ik_method(target_pos, target_alpha, target_psi, current_joints)

                # Handle different return types
                if isinstance(result, tuple):
                    solution, num_iter = result
                else:
                    solution = result
                    num_iter = 0

            except Exception as e:
                solution = None
                num_iter = 0
                print(f"  Error on case {i}: {e}")

            solve_time = time.time() - start_time

            # Evaluate
            ik_result = self.evaluate_solution(
                solution, target_pos, target_alpha, target_psi,
                current_joints, solve_time, num_iter
            )

            results.append(ik_result)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(test_cases)} cases")

        # Aggregate metrics
        metrics = self._aggregate_results(results)
        metrics['method_name'] = method_name

        self._print_metrics(metrics)

        return metrics

    def _aggregate_results(self, results: List[IKSolution]) -> Dict:
        """Aggregate individual results into summary metrics.

        Args:
            results: List of IKSolution objects.

        Returns:
            Dictionary of aggregated metrics.
        """
        num_total = len(results)
        num_success = sum(r.success for r in results)

        # Filter successful solutions for error metrics
        successful = [r for r in results if r.success]

        if successful:
            pos_errors = [r.position_error for r in successful]
            ori_errors = [r.orientation_error for r in successful]
            solve_times = [r.solve_time for r in results]
            iterations = [r.num_iterations for r in results if r.num_iterations > 0]
            joint_jumps = [r.max_joint_jump for r in results if r.max_joint_jump > 0]

            metrics = {
                'success_rate': num_success / num_total,
                'num_success': num_success,
                'num_total': num_total,
                'position_error_mean': np.mean(pos_errors),
                'position_error_std': np.std(pos_errors),
                'position_error_max': np.max(pos_errors),
                'orientation_error_mean': np.mean(ori_errors),
                'orientation_error_std': np.std(ori_errors),
                'solve_time_mean': np.mean(solve_times),
                'solve_time_p95': np.percentile(solve_times, 95),
                'solve_time_max': np.max(solve_times),
            }

            if iterations:
                metrics['iterations_mean'] = np.mean(iterations)
                metrics['iterations_max'] = np.max(iterations)

            if joint_jumps:
                metrics['max_joint_jump_mean'] = np.mean(joint_jumps)
                metrics['max_joint_jump_max'] = np.max(joint_jumps)

        else:
            metrics = {
                'success_rate': 0.0,
                'num_success': 0,
                'num_total': num_total,
                'position_error_mean': float('inf'),
                'solve_time_mean': np.mean([r.solve_time for r in results]),
            }

        # Limit violations
        num_limit_violations = sum(r.joint_limit_violation for r in results)
        metrics['joint_limit_violation_rate'] = num_limit_violations / num_total

        return metrics

    def _print_metrics(self, metrics: Dict):
        """Print formatted metrics.

        Args:
            metrics: Dictionary of metrics.
        """
        print(f"\n{'='*60}")
        print(f"Method: {metrics['method_name']}")
        print(f"{'='*60}")
        print(f"Success Rate: {metrics['success_rate']*100:.2f}% "
              f"({metrics['num_success']}/{metrics['num_total']})")

        if metrics['success_rate'] > 0:
            print(f"Position Error: {metrics['position_error_mean']*1000:.3f} ± "
                  f"{metrics['position_error_std']*1000:.3f} mm "
                  f"(max: {metrics['position_error_max']*1000:.3f} mm)")
            print(f"Orientation Error: {np.rad2deg(metrics['orientation_error_mean']):.3f} ± "
                  f"{np.rad2deg(metrics['orientation_error_std']):.3f} deg")

        print(f"Solve Time: {metrics['solve_time_mean']*1000:.3f} ms "
              f"(p95: {metrics['solve_time_p95']*1000:.3f} ms, "
              f"max: {metrics['solve_time_max']*1000:.3f} ms)")

        if 'iterations_mean' in metrics:
            print(f"Iterations: {metrics['iterations_mean']:.1f} "
                  f"(max: {metrics['iterations_max']})")

        if 'max_joint_jump_mean' in metrics:
            print(f"Max Joint Jump: {np.rad2deg(metrics['max_joint_jump_mean']):.2f} deg "
                  f"(max: {np.rad2deg(metrics['max_joint_jump_max']):.2f} deg)")

        print(f"Joint Limit Violations: {metrics['joint_limit_violation_rate']*100:.2f}%")
        print(f"{'='*60}\n")

    def save_results(self, all_metrics: List[Dict], output_path: str):
        """Save evaluation results to JSON file.

        Args:
            all_metrics: List of metric dictionaries from different methods.
            output_path: Path to output JSON file.
        """
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_metrics = []
        for metrics in all_metrics:
            serializable = {k: convert(v) for k, v in metrics.items()}
            serializable_metrics.append(serializable)

        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"Results saved to {output_path}")


def generate_test_cases(fk_solver,
                       joint_limits: Tuple[np.ndarray, np.ndarray],
                       num_samples: int = 1000,
                       test_type: str = 'normal',
                       seed: int = 42) -> List[Dict]:
    """Generate test cases for evaluation.

    Args:
        fk_solver: Forward kinematics solver.
        joint_limits: Joint limits tuple.
        num_samples: Number of test cases.
        test_type: Type of test ('normal', 'boundary', 'singular', 'noisy').
        seed: Random seed.

    Returns:
        List of test case dictionaries.
    """
    np.random.seed(seed)
    lower_limits, upper_limits = joint_limits

    test_cases = []
    attempts = 0
    max_attempts = num_samples * 10

    while len(test_cases) < num_samples and attempts < max_attempts:
        attempts += 1

        # Sample joint configuration based on test type
        if test_type == 'boundary':
            # Sample near joint limits
            q = np.random.uniform(lower_limits, upper_limits)
            # Push some joints near limits
            for i in range(5):
                if np.random.rand() < 0.3:
                    if np.random.rand() < 0.5:
                        q[i] = lower_limits[i] + np.random.uniform(0, 0.1)
                    else:
                        q[i] = upper_limits[i] - np.random.uniform(0, 0.1)
        else:
            # Normal sampling
            margin = 0.1  # Stay away from limits
            q = np.random.uniform(lower_limits + margin, upper_limits - margin)

        try:
            pos, alpha, psi = fk_solver.compute_task_space(q)

            # Add noise for 'noisy' test type
            if test_type == 'noisy':
                pos += np.random.normal(0, 0.005, size=3)  # 5mm noise

            # Generate current joints (slight perturbation)
            q_current = q + np.random.normal(0, 0.1, size=5)
            q_current = np.clip(q_current, lower_limits, upper_limits)

            test_cases.append({
                'position': pos,
                'alpha': alpha,
                'psi': psi,
                'current_joints': q_current,
                'ground_truth_joints': q
            })

        except Exception:
            continue

    print(f"Generated {len(test_cases)} test cases of type '{test_type}'")
    return test_cases
