"""把真机采集的原始数据转换为神经网络训练格式

采集格式 (collect_real_data.py 输出):
    joint_angles      (N,5)  实际到达的关节角(弧度)
    end_positions     (N,3)  末端把手尖端位置(米)
    end_orientations  (N,3)  [roll, pitch, yaw](弧度)
    servo_positions   (N,5)  舵机原始值

训练格式 (IKDataset 期望):
    train/val/test 三组, 每组:
        inputs        (n,12) [x,y,z,sin α,cos α,sin ψ,cos ψ, q_current 1:5]
        target_joints (n,5)  目标关节角
        positions     (n,3)  目标位置
        orientations  (n,2)  [alpha, psi]

映射说明:
    - target_joints = 采集到的实际关节角(这是真机"标准答案")
    - alpha(pitch) 用 end_orientations 的 pitch; psi(roll) 用关节5角度
      (与 forward_kinematics_simple.compute_task_space 的定义保持一致)
    - q_current = target + 高斯噪声, 模拟"从附近某姿态出发求解到目标"
      这让网络学会: 给定目标位姿和一个近似当前姿态, 预测精确目标关节角
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/002

import argparse
import numpy as np
import h5py


def build_task_space(fk, joint_angles):
    """用FK的compute_task_space重新计算(alpha, psi), 保证与求解器一致。"""
    positions = []
    alphas = []
    psis = []
    for q in joint_angles:
        pos, alpha, psi = fk.compute_task_space(q)
        positions.append(pos)
        alphas.append(alpha)
        psis.append(psi)
    return np.array(positions), np.array(alphas), np.array(psis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='real_robot_dataset.h5',
                    help='采集的原始数据')
    ap.add_argument('--output', default='data/ik_dataset.h5',
                    help='训练格式输出')
    ap.add_argument('--urdf', default='models/so101_new_calib.urdf')
    ap.add_argument('--noise', type=float, default=0.08,
                    help='q_current高斯噪声(弧度), 仅noise模式用')
    ap.add_argument('--qcurrent', default='random',
                    choices=['random', 'noise', 'home'],
                    help='q_current构造方式: random(推荐,防捷径)/noise(旧)/home')
    ap.add_argument('--augment', type=int, default=3,
                    help='每个真实样本生成几个不同噪声的训练样本')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from kinematics.forward_kinematics_simple import ForwardKinematics
    fk = ForwardKinematics(args.urdf)

    # 读采集数据
    print(f"读取采集数据: {args.input}")
    with h5py.File(args.input, 'r') as f:
        joint_angles = f['joint_angles'][:]        # (N,5)
        print(f"  原始样本数: {len(joint_angles)}")

    np.random.seed(args.seed)

    # 用FK一致地重算task-space (alpha, psi) 和位置
    print("用FK重算task-space位姿...")
    positions, alphas, psis = build_task_space(fk, joint_angles)

    # 数据增强: 每个真实样本配多个带噪的q_current
    inputs = []
    target_joints = []
    out_positions = []
    out_orientations = []

    lower = joint_angles.min(axis=0) - 0.2
    upper = joint_angles.max(axis=0) + 0.2
    # 所有真实关节角池, 供random模式采样无关的q_current
    all_q = joint_angles.copy()

    for i in range(len(joint_angles)):
        q_target = joint_angles[i]
        pos = positions[i]
        alpha = alphas[i]
        psi = psis[i]

        for _ in range(args.augment):
            # q_current 构造模式:
            #   random(默认): 从其他真实姿态随机取, 与目标无关 → 逼网络从位姿学映射,
            #                 避免"抄初值"捷径(旧noise模式导致真机112mm误差)
            #   noise: 目标+小噪声(旧行为, 有捷径陷阱)
            #   home: 固定HOME常量
            if args.qcurrent == 'random':
                q_current = all_q[np.random.randint(len(all_q))].copy()
            elif args.qcurrent == 'home':
                q_current = np.zeros(5)  # HOME=舵机2048=0弧度
            else:  # noise
                q_current = q_target + np.random.normal(0, args.noise, size=5)
                q_current = np.clip(q_current, lower, upper)

            input_vec = np.array([
                pos[0], pos[1], pos[2],
                np.sin(alpha), np.cos(alpha),
                np.sin(psi), np.cos(psi),
                q_current[0], q_current[1], q_current[2],
                q_current[3], q_current[4]
            ])
            inputs.append(input_vec)
            target_joints.append(q_target)
            out_positions.append(pos)
            out_orientations.append([alpha, psi])

    inputs = np.array(inputs, dtype=np.float32)
    target_joints = np.array(target_joints, dtype=np.float32)
    out_positions = np.array(out_positions, dtype=np.float32)
    out_orientations = np.array(out_orientations, dtype=np.float32)

    print(f"  增强后样本数: {len(inputs)} (增强倍数={args.augment})")

    # 切分 train/val/test = 70/15/15
    n = len(inputs)
    idx = np.random.permutation(n)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    print(f"  切分: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # 保存为训练格式
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write_split(f, name, sel):
        g = f.create_group(name)
        g.create_dataset('inputs', data=inputs[sel], compression='gzip')
        g.create_dataset('target_joints', data=target_joints[sel], compression='gzip')
        g.create_dataset('positions', data=out_positions[sel], compression='gzip')
        g.create_dataset('orientations', data=out_orientations[sel], compression='gzip')

    with h5py.File(out_path, 'w') as f:
        write_split(f, 'train', train_idx)
        write_split(f, 'val', val_idx)
        write_split(f, 'test', test_idx)

    print(f"\n✓ 训练数据已保存: {out_path}")
    print(f"  下一步: python train_network.py --dataset {out_path} --device cpu")


if __name__ == '__main__':
    main()
