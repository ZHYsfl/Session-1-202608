"""Main script to run all IK methods and generate comparison results."""

import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from kinematics import ForwardKinematics, AnalyticalIK, NumericalIK
from neural_network import load_model, prepare_input
from utils.config_loader import load_config, get_joint_limits, get_link_lengths
from utils.evaluation import IKEvaluator, generate_test_cases


def analytical_ik_wrapper(ik_solver):
    """Wrapper for analytical IK to match evaluation interface."""
    def solve(pos, alpha, psi, current_joints):
        if current_joints is not None:
            return ik_solver.choose_closest_solution(pos, alpha, psi, current_joints)
        else:
            return ik_solver.solve(pos, alpha, psi, prefer_elbow_up=True)
    return solve


def numerical_ik_wrapper(ik_solver):
    """Wrapper for numerical IK to match evaluation interface."""
    def solve(pos, alpha, psi, current_joints):
        if current_joints is None:
            current_joints = np.zeros(5)
        return ik_solver.solve(pos, alpha, psi, initial_guess=current_joints,
                              return_iterations=True)
    return solve


def neural_ik_wrapper(model):
    """Wrapper for neural network IK."""
    def solve(pos, alpha, psi, current_joints):
        if current_joints is None:
            current_joints = np.zeros(5)

        input_vec = prepare_input(pos, alpha, psi, current_joints)
        solution = model.predict(input_vec).flatten()
        return solution
    return solve


def hybrid_ik_wrapper(model, numerical_solver, num_steps=3):
    """Wrapper for hybrid IK (NN + numerical refinement)."""
    def solve(pos, alpha, psi, current_joints):
        if current_joints is None:
            current_joints = np.zeros(5)

        # Neural network prediction
        input_vec = prepare_input(pos, alpha, psi, current_joints)
        nn_solution = model.predict(input_vec).flatten()

        # Numerical refinement
        refined_solution, iterations = numerical_solver.refine_solution(
            pos, alpha, psi, nn_solution, num_steps=num_steps
        )

        return refined_solution, iterations
    return solve


def main():
    parser = argparse.ArgumentParser(description='Evaluate IK methods for SO-101')
    parser.add_argument('--config', type=str, default='configs/robot_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--urdf', type=str, default='models/so101.urdf',
                       help='Path to URDF file')
    parser.add_argument('--model', type=str, default='models/ik_mlp_best.pth',
                       help='Path to trained neural network model')
    parser.add_argument('--num_test', type=int, default=1000,
                       help='Number of test cases per test set')
    parser.add_argument('--output', type=str, default='results/ik_comparison.json',
                       help='Output path for results')
    parser.add_argument('--test_types', nargs='+',
                       default=['normal', 'boundary', 'singular', 'noisy'],
                       help='Types of test sets to evaluate')

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    joint_limits = get_joint_limits(config, use_soft_limits=True)
    link_lengths = get_link_lengths(config)

    # Initialize forward kinematics
    print(f"Loading URDF from {args.urdf}...")
    try:
        fk = ForwardKinematics(args.urdf)
    except Exception as e:
        print(f"Error loading URDF: {e}")
        print("Using simplified forward kinematics...")
        # Fallback or simplified FK
        return

    # Initialize IK solvers
    print("Initializing IK solvers...")

    # Analytical IK
    analytical_ik = AnalyticalIK(
        shoulder_height=link_lengths['shoulder_height'],
        radial_offset=link_lengths['radial_offset'],
        upper_arm_length=link_lengths['upper_arm'],
        forearm_length=link_lengths['forearm'],
        wrist_to_tcp=link_lengths['wrist_to_tcp'],
        joint_limits=joint_limits
    )

    # Numerical IK
    ik_config = config['ik']['numerical']
    numerical_ik = NumericalIK(
        fk=fk,
        joint_limits=joint_limits,
        max_iterations=ik_config['max_iterations'],
        position_tolerance=ik_config['position_tolerance'],
        orientation_tolerance=ik_config['orientation_tolerance'],
        damping_min=ik_config['damping_min'],
        damping_max=ik_config['damping_max'],
        max_joint_step=ik_config['max_joint_step']
    )

    # Neural network IK
    print(f"Loading neural network from {args.model}...")
    try:
        nn_model = load_model(args.model)
        has_nn = True
    except Exception as e:
        print(f"Could not load neural network: {e}")
        print("Skipping neural network and hybrid methods.")
        has_nn = False

    # Initialize evaluator
    evaluator = IKEvaluator(fk, joint_limits)

    # Run evaluations
    all_results = []

    for test_type in args.test_types:
        print(f"\n{'='*70}")
        print(f"Test Set: {test_type.upper()}")
        print(f"{'='*70}")

        # Generate test cases
        test_cases = generate_test_cases(
            fk, joint_limits,
            num_samples=args.num_test,
            test_type=test_type
        )

        # Evaluate analytical IK
        metrics_analytical = evaluator.benchmark_method(
            analytical_ik_wrapper(analytical_ik),
            test_cases,
            method_name=f"Analytical IK ({test_type})"
        )
        metrics_analytical['test_type'] = test_type
        all_results.append(metrics_analytical)

        # Evaluate numerical IK
        metrics_numerical = evaluator.benchmark_method(
            numerical_ik_wrapper(numerical_ik),
            test_cases,
            method_name=f"Numerical IK ({test_type})"
        )
        metrics_numerical['test_type'] = test_type
        all_results.append(metrics_numerical)

        if has_nn:
            # Evaluate neural network IK
            metrics_nn = evaluator.benchmark_method(
                neural_ik_wrapper(nn_model),
                test_cases,
                method_name=f"Neural Network IK ({test_type})"
            )
            metrics_nn['test_type'] = test_type
            all_results.append(metrics_nn)

            # Evaluate hybrid methods with different refinement steps
            for num_steps in [1, 3, 5]:
                metrics_hybrid = evaluator.benchmark_method(
                    hybrid_ik_wrapper(nn_model, numerical_ik, num_steps),
                    test_cases,
                    method_name=f"Hybrid IK (NN + {num_steps} steps) ({test_type})"
                )
                metrics_hybrid['test_type'] = test_type
                metrics_hybrid['refinement_steps'] = num_steps
                all_results.append(metrics_hybrid)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(all_results, str(output_path))

    print("\n" + "="*70)
    print("Evaluation completed!")
    print(f"Results saved to {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()
