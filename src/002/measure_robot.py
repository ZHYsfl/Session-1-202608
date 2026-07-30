"""Measure SO-101 robot arm link lengths interactively."""

import sys
sys.path.insert(0, '.')

from robot_control.so101_interface import SO101Robot
import numpy as np
import time


def measure_link_lengths():
    """Interactive link length measurement guide."""

    print("\n" + "="*70)
    print("SO-101 机械臂连杆长度测量向导")
    print("="*70)

    print("\n本向导将帮助您测量SO-101的关键尺寸，用于生成URDF文件")
    print("\n需要工具：")
    print("  - 尺子或卷尺（精度1mm）")
    print("  - 机械臂（可以不通电）")

    input("\n准备好后按Enter继续...")

    measurements = {}

    # Measurement 1: Base height
    print("\n" + "-"*70)
    print("测量 1/5: 底座高度 (h)")
    print("-"*70)
    print("说明：")
    print("  从机械臂安装底面到肩关节（第2个关节）旋转轴心的垂直距离")
    print("\n如何测量：")
    print("  1. 将机械臂放在平面上")
    print("  2. 测量底座底面到肩关节轴的高度")
    print("\n示意图：")
    print("     [肩关节]")
    print("        |")
    print("        | ← 测量这段距离")
    print("        |")
    print("     [底座]")
    print("    ----------")

    while True:
        try:
            h = float(input("\n请输入底座高度 (单位：米，例如 0.08): "))
            if 0.03 < h < 0.20:
                measurements['shoulder_height'] = h
                print(f"✓ 记录：底座高度 = {h*1000:.1f} mm")
                break
            else:
                print("⚠️ 数值异常，通常在30-200mm之间，请重新输入")
        except ValueError:
            print("✗ 输入无效，请输入数字")

    # Measurement 2: Upper arm length
    print("\n" + "-"*70)
    print("测量 2/5: 上臂长度 (L1)")
    print("-"*70)
    print("说明：")
    print("  从肩关节（第2个关节）到肘关节（第3个关节）的直线距离")
    print("\n如何测量：")
    print("  1. 将肩部和肘部关节伸直")
    print("  2. 测量两个关节旋转轴心之间的距离")
    print("\n示意图：")
    print("  [肩] ----L1---- [肘]")

    while True:
        try:
            L1 = float(input("\n请输入上臂长度 (单位：米，例如 0.15): "))
            if 0.08 < L1 < 0.30:
                measurements['upper_arm'] = L1
                print(f"✓ 记录：上臂长度 = {L1*1000:.1f} mm")
                break
            else:
                print("⚠️ 数值异常，通常在80-300mm之间，请重新输入")
        except ValueError:
            print("✗ 输入无效，请输入数字")

    # Measurement 3: Forearm length
    print("\n" + "-"*70)
    print("测量 3/5: 前臂长度 (L2)")
    print("-"*70)
    print("说明：")
    print("  从肘关节（第3个关节）到腕关节（第4个关节）的直线距离")
    print("\n如何测量：")
    print("  1. 将肘部和腕部关节伸直")
    print("  2. 测量两个关节旋转轴心之间的距离")
    print("\n示意图：")
    print("  [肘] ----L2---- [腕]")

    while True:
        try:
            L2 = float(input("\n请输入前臂长度 (单位：米，例如 0.15): "))
            if 0.08 < L2 < 0.30:
                measurements['forearm'] = L2
                print(f"✓ 记录：前臂长度 = {L2*1000:.1f} mm")
                break
            else:
                print("⚠️ 数值异常，通常在80-300mm之间，请重新输入")
        except ValueError:
            print("✗ 输入无效，请输入数字")

    # Measurement 4: Wrist to TCP
    print("\n" + "-"*70)
    print("测量 4/5: 腕到工具中心点距离 (L3)")
    print("-"*70)
    print("说明：")
    print("  从腕关节（第4个关节）到夹爪中心点的距离")
    print("\n如何测量：")
    print("  1. 将夹爪闭合")
    print("  2. 测量腕关节轴到夹爪指尖中点的距离")
    print("\n示意图：")
    print("  [腕] ----L3---- [夹爪中心]")
    print("                      \\/")

    while True:
        try:
            L3 = float(input("\n请输入腕到TCP距离 (单位：米，例如 0.10): "))
            if 0.05 < L3 < 0.20:
                measurements['wrist_to_tcp'] = L3
                print(f"✓ 记录：腕到TCP距离 = {L3*1000:.1f} mm")
                break
            else:
                print("⚠️ 数值异常，通常在50-200mm之间，请重新输入")
        except ValueError:
            print("✗ 输入无效，请输入数字")

    # Measurement 5: Radial offset (optional)
    print("\n" + "-"*70)
    print("测量 5/5: 径向偏置 (d0) [可选]")
    print("-"*70)
    print("说明：")
    print("  底座旋转轴到肩关节的水平距离（通常很小或为0）")
    print("\n如果不确定，直接按Enter跳过（使用默认值0.02m）")

    d0_input = input("\n请输入径向偏置 (单位：米，或按Enter使用默认值): ").strip()
    if d0_input:
        try:
            d0 = float(d0_input)
            measurements['radial_offset'] = d0
            print(f"✓ 记录：径向偏置 = {d0*1000:.1f} mm")
        except ValueError:
            measurements['radial_offset'] = 0.02
            print("✓ 使用默认值：径向偏置 = 20 mm")
    else:
        measurements['radial_offset'] = 0.02
        print("✓ 使用默认值：径向偏置 = 20 mm")

    # Summary
    print("\n" + "="*70)
    print("测量完成！")
    print("="*70)
    print("\n您的SO-101机械臂尺寸：")
    print(f"  底座高度 (h):      {measurements['shoulder_height']*1000:.1f} mm")
    print(f"  上臂长度 (L1):     {measurements['upper_arm']*1000:.1f} mm")
    print(f"  前臂长度 (L2):     {measurements['forearm']*1000:.1f} mm")
    print(f"  腕到TCP (L3):      {measurements['wrist_to_tcp']*1000:.1f} mm")
    print(f"  径向偏置 (d0):     {measurements['radial_offset']*1000:.1f} mm")

    total_reach = measurements['upper_arm'] + measurements['forearm'] + measurements['wrist_to_tcp']
    print(f"\n理论最大伸展距离: {total_reach*1000:.1f} mm")

    # Save to file
    import yaml
    output_file = "configs/measured_dimensions.yaml"

    with open(output_file, 'w') as f:
        yaml.dump({
            'robot': {
                'link_lengths': measurements,
                'measured_date': time.strftime('%Y-%m-%d'),
                'notes': 'Manually measured dimensions'
            }
        }, f)

    print(f"\n✓ 尺寸已保存到: {output_file}")

    # Offer to generate URDF
    print("\n" + "="*70)
    response = input("\n是否立即生成URDF文件? (y/n): ").strip().lower()

    if response == 'y':
        print("\n正在生成URDF...")
        generate_urdf_from_measurements(measurements)
    else:
        print("\n您可以稍后运行以下命令生成URDF:")
        print("  python generate_urdf.py")

    return measurements


def generate_urdf_from_measurements(measurements):
    """Generate URDF file from measurements."""

    urdf_content = f"""<?xml version="1.0"?>
<robot name="so101">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="{measurements['shoulder_height']}" radius="0.04"/>
      </geometry>
      <origin xyz="0 0 {measurements['shoulder_height']/2}" rpy="0 0 0"/>
    </visual>
  </link>

  <!-- Shoulder Pan Joint (Joint 1) -->
  <joint name="shoulder_pan" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 {measurements['shoulder_height']}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2.0"/>
  </joint>

  <link name="shoulder_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </visual>
  </link>

  <!-- Shoulder Lift Joint (Joint 2) -->
  <joint name="shoulder_lift" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2.0"/>
  </joint>

  <link name="upper_arm_link">
    <visual>
      <geometry>
        <cylinder length="{measurements['upper_arm']}" radius="0.02"/>
      </geometry>
      <origin xyz="{measurements['upper_arm']/2} 0 0" rpy="0 1.5708 0"/>
    </visual>
  </link>

  <!-- Elbow Flex Joint (Joint 3) -->
  <joint name="elbow_flex" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="forearm_link"/>
    <origin xyz="{measurements['upper_arm']} 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.35" upper="2.35" effort="10" velocity="2.0"/>
  </joint>

  <link name="forearm_link">
    <visual>
      <geometry>
        <cylinder length="{measurements['forearm']}" radius="0.018"/>
      </geometry>
      <origin xyz="{measurements['forearm']/2} 0 0" rpy="0 1.5708 0"/>
    </visual>
  </link>

  <!-- Wrist Flex Joint (Joint 4) -->
  <joint name="wrist_flex" type="revolute">
    <parent link="forearm_link"/>
    <child link="wrist_link"/>
    <origin xyz="{measurements['forearm']} 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="5" velocity="2.0"/>
  </joint>

  <link name="wrist_link">
    <visual>
      <geometry>
        <box size="0.04 0.04 0.04"/>
      </geometry>
    </visual>
  </link>

  <!-- Wrist Roll Joint (Joint 5) -->
  <joint name="wrist_roll" type="revolute">
    <parent link="wrist_link"/>
    <child link="gripper_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="5" velocity="2.0"/>
  </joint>

  <link name="gripper_link">
    <visual>
      <geometry>
        <cylinder length="{measurements['wrist_to_tcp']}" radius="0.015"/>
      </geometry>
      <origin xyz="{measurements['wrist_to_tcp']/2} 0 0" rpy="0 1.5708 0"/>
    </visual>
  </link>

  <!-- Tool Center Point -->
  <link name="tool_link"/>

  <joint name="tool_joint" type="fixed">
    <parent link="gripper_link"/>
    <child link="tool_link"/>
    <origin xyz="{measurements['wrist_to_tcp']} 0 0" rpy="0 0 0"/>
  </joint>

</robot>
"""

    urdf_file = "models/so101_measured.urdf"
    with open(urdf_file, 'w') as f:
        f.write(urdf_content)

    print(f"\n✓ URDF文件已生成: {urdf_file}")
    print("\n下一步：")
    print("  1. 测试正运动学:")
    print("     python tests/test_forward_kinematics.py")
    print("  2. 开始生成训练数据:")
    print("     python generate_dataset.py --urdf models/so101_measured.urdf")


if __name__ == "__main__":
    try:
        measurements = measure_link_lengths()
    except KeyboardInterrupt:
        print("\n\n测量被取消")
        sys.exit(0)
