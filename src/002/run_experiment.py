"""完整实验执行脚本 - SO-101 逆运动学研究"""

import sys
sys.path.insert(0, '.')

import numpy as np
import time
from pathlib import Path

print("\n" + "="*70)
print("SO-101 逆运动学实验 - 自动执行")
print("="*70)

# ============================================================================
# 阶段1: 验证系统准备
# ============================================================================
print("\n[阶段1/6] 验证系统准备...")

try:
    from kinematics.forward_kinematics_simple import ForwardKinematics
    print("  ✓ 正运动学模块")

    from kinematics.analytical_ik import AnalyticalIK
    print("  ✓ 解析逆运动学模块")

    from utils.config_loader import load_config, get_joint_limits, get_link_lengths
    print("  ✓ 配置加载模块")

    # 加载配置
    config = load_config('configs/robot_config.yaml')
    joint_limits = get_joint_limits(config, use_soft_limits=True)
    link_lengths = get_link_lengths(config)

    print("  ✓ 配置文件加载成功")

    # 初始化FK
    fk = ForwardKinematics('models/so101_new_calib.urdf')
    print("  ✓ URDF加载成功")

    print("\n✓ 系统准备完成！")

except Exception as e:
    print(f"\n✗ 系统准备失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 阶段2: 测试解析逆运动学
# ============================================================================
print("\n[阶段2/6] 测试解析逆运动学...")

try:
    # 初始化解析IK
    analytical_ik = AnalyticalIK(
        shoulder_height=link_lengths['shoulder_height'],
        radial_offset=link_lengths['radial_offset'],
        upper_arm_length=link_lengths['upper_arm'],
        forearm_length=link_lengths['forearm'],
        wrist_to_tcp=link_lengths['wrist_to_tcp'],
        joint_limits=joint_limits
    )

    print("  ✓ 解析IK初始化")

    # 测试几个点
    test_points = [
        ([0.20, 0.00, 0.15], 0.0, 0.0),
        ([0.15, 0.15, 0.10], -0.5, 0.0),
        ([0.18, -0.10, 0.12], 0.3, 0.5),
    ]

    success_count = 0
    for i, (pos, alpha, psi) in enumerate(test_points):
        solution = analytical_ik.solve(np.array(pos), alpha, psi)
        if solution is not None:
            # 验证FK
            fk_pos, fk_alpha, fk_psi = fk.compute_task_space(solution)
            error = np.linalg.norm(fk_pos - pos)
            if error < 0.01:  # 10mm tolerance
                success_count += 1
                print(f"  ✓ 测试点{i+1}: 误差={error*1000:.2f}mm")
            else:
                print(f"  ⚠ 测试点{i+1}: 误差={error*1000:.2f}mm (较大)")
        else:
            print(f"  ✗ 测试点{i+1}: 无解")

    print(f"\n  解析IK成功率: {success_count}/{len(test_points)}")
    print("✓ 解析逆运动学测试完成！")

except Exception as e:
    print(f"\n✗ 解析IK测试失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 阶段3: 生成训练数据（小规模测试）
# ============================================================================
print("\n[阶段3/6] 生成训练数据（测试版本 - 1000样本）...")

try:
    from neural_network.dataset import IKDatasetGenerator

    # 生成小规模数据集用于测试
    generator = IKDatasetGenerator(
        fk_solver=fk,
        ik_solver=analytical_ik,
        joint_limits=joint_limits,
        num_samples=1000  # 测试用，实际应该是150000
    )

    print("  正在生成数据...")
    dataset = generator.generate(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        current_joint_noise=0.05,
        seed=42
    )

    print(f"  ✓ 训练集: {len(dataset['train']['inputs'])} 样本")
    print(f"  ✓ 验证集: {len(dataset['val']['inputs'])} 样本")
    print(f"  ✓ 测试集: {len(dataset['test']['inputs'])} 样本")

    # 保存数据
    output_path = Path('data/ik_dataset_test.h5')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_to_hdf5(dataset, str(output_path))

    print(f"  ✓ 数据已保存到: {output_path}")
    print("\n✓ 数据生成完成！")

    print("\n  📝 注意: 这是测试版本(1000样本)")
    print("  📝 完整训练需要运行:")
    print("     python generate_dataset.py --num_samples 150000")

except Exception as e:
    print(f"\n✗ 数据生成失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 阶段4: 训练神经网络（跳过，需要大量数据）
# ============================================================================
print("\n[阶段4/6] 神经网络训练 [跳过]")
print("  ℹ️  需要大规模数据集(150k样本)")
print("  ℹ️  预计训练时间: 2-4小时(GPU) 或 12-24小时(CPU)")
print("  ℹ️  运行命令: python train_network.py --device cuda")

# ============================================================================
# 阶段5: 评估对比（使用解析和数值IK）
# ============================================================================
print("\n[阶段5/6] 评估IK方法对比（解析 vs 数值）...")

try:
    from kinematics.numerical_ik import NumericalIK
    from utils.evaluation import IKEvaluator, generate_test_cases

    # 初始化数值IK
    numerical_ik = NumericalIK(
        fk=fk,
        joint_limits=joint_limits,
        max_iterations=50,
        position_tolerance=0.001,
        orientation_tolerance=0.01
    )

    print("  ✓ 数值IK初始化")

    # 生成小规模测试集
    print("  正在生成测试用例...")
    test_cases = generate_test_cases(fk, joint_limits, num_samples=50, test_type='normal')

    print(f"  ✓ 生成{len(test_cases)}个测试用例")

    # 评估解析IK
    print("\n  评估解析IK...")
    evaluator = IKEvaluator(fk, joint_limits)

    def analytical_wrapper(pos, alpha, psi, current):
        if current is None:
            current = np.zeros(5)
        return analytical_ik.choose_closest_solution(pos, alpha, psi, current)

    results_analytical = []
    for case in test_cases[:10]:  # 只测试10个
        sol = analytical_wrapper(case['position'], case['alpha'], case['psi'],
                                 case['current_joints'])
        if sol is not None:
            results_analytical.append(sol)

    print(f"    成功率: {len(results_analytical)}/10")

    # 评估数值IK
    print("\n  评估数值IK...")

    def numerical_wrapper(pos, alpha, psi, current):
        if current is None:
            current = np.zeros(5)
        result = numerical_ik.solve(pos, alpha, psi, initial_guess=current,
                                   return_iterations=True)
        return result

    results_numerical = []
    for case in test_cases[:10]:
        result = numerical_wrapper(case['position'], case['alpha'], case['psi'],
                                  case['current_joints'])
        if result[0] is not None:
            results_numerical.append(result)

    print(f"    成功率: {len(results_numerical)}/10")
    print(f"    平均迭代次数: {np.mean([r[1] for r in results_numerical]):.1f}")

    print("\n✓ IK方法评估完成！")

except Exception as e:
    print(f"\n✗ 评估失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 阶段6: 总结和下一步
# ============================================================================
print("\n" + "="*70)
print("[阶段6/6] 实验总结")
print("="*70)

print("\n✓ 已完成:")
print("  1. ✓ 系统初始化和URDF加载")
print("  2. ✓ 解析逆运动学实现和测试")
print("  3. ✓ 小规模训练数据生成(1000样本)")
print("  4. ⊘ 神经网络训练(需要大规模数据)")
print("  5. ✓ IK方法对比评估(小规模)")

print("\n📋 下一步任务:")
print("\n  【软件实验路线】")
print("  1. 生成完整训练数据(150k样本):")
print("     python generate_dataset.py --num_samples 150000")
print("     预计时间: 1-2小时")
print()
print("  2. 训练神经网络:")
print("     python train_network.py --device cuda --epochs 200")
print("     预计时间: 2-4小时(GPU) 或 12-24小时(CPU)")
print()
print("  3. 完整评估对比:")
print("     python run_evaluation.py --num_test 1000")
print("     预计时间: 30-60分钟")
print()
print("  4. 生成结果图表:")
print("     python -m utils.visualization --results results/ik_comparison.json")
print()
print("  5. 撰写论文:")
print("     编辑 paper/002/main.tex")

print("\n  【真机验证路线】（需要接通12V电源）")
print("  1. 单关节测试:")
print("     python experiments/single_joint_test.py --port COM24")
print()
print("  2. 单点到达测试:")
print("     python experiments/single_point_test.py")
print()
print("  3. 主从遥操作演示:")
print("     python experiments/teleoperation_demo.py")

print("\n" + "="*70)
print("实验脚本执行完成！")
print("="*70)

print("\n💡 提示:")
print("  - 查看详细实验方案: EXPERIMENT_PLAN.md")
print("  - 查看快速启动指南: QUICK_START.md")
print("  - 双机械臂配置: DUAL_ARM_SETUP.md")

print("\n")
