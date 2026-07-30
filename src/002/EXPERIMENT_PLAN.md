# SO-101 逆运动学实验完整方案

## 🎯 实验目标

**课题：** 机械臂逆运动学解析与神经网络端到端实现

**核心任务：**
1. 实现并比较4种逆运动学方法
2. 训练神经网络学习IK映射
3. 在真机上验证算法精度和实用性

---

## 🤖 双机械臂实验设计

### 硬件配置
- **主机械臂（Leader）：** COM23 - 用于示教和数据采集
- **从机械臂（Follower）：** COM24 - 用于执行IK算法

### 实验优势
使用双机械臂可以：
1. **实时对比** - 主臂演示，从臂跟随（展示IK效果）
2. **数据采集** - 从主臂采集真实运动数据
3. **算法验证** - 从臂执行IK算法，主臂作为参考

---

## 📊 实验流程设计

### 阶段1：系统准备（1-2天）

#### 任务1.1：获取URDF模型
```bash
# 从LeRobot获取SO-101 URDF
# 方法1：Git clone
git clone https://github.com/huggingface/lerobot.git
cd lerobot
# 查找 SO-101 URDF文件

# 方法2：直接下载
# 访问 https://github.com/huggingface/lerobot/tree/main/lerobot/configs/robot
```

**目标：** 获取准确的连杆长度和DH参数

#### 任务1.2：验证正运动学
```bash
# 测试正运动学精度
python test_forward_kinematics.py

# 读取真机关节角，与FK计算对比
python verify_fk_on_robot.py
```

**目标：** 确保FK计算准确（IK的基础）

---

### 阶段2：数据采集（1天）

#### 方法A：随机采样（基础方法）⭐

```bash
# 生成15万随机样本
python generate_dataset.py \
    --urdf models/so101.urdf \
    --num_samples 150000 \
    --output data/ik_dataset_random.h5
```

**数据来源：**
- 在关节限位内随机采样
- 通过FK计算对应末端位姿
- 用解析IK生成标签

**优点：** 覆盖整个工作空间
**缺点：** 数据分布可能不符合实际使用

#### 方法B：主臂示教采集（推荐）⭐⭐⭐

```python
"""使用主臂采集真实运动数据"""

# collect_real_data.py
from robot_control.so101_interface import SO101Robot
import numpy as np
import time

leader = SO101Robot(port="COM23")  # 主臂
follower = SO101Robot(port="COM24")  # 从臂

print("请手动操作主臂完成示教动作...")
print("采集过程中，从臂将实时跟随主臂")

dataset = []
start_time = time.time()

while len(dataset) < 10000:  # 采集1万个样本
    # 读取主臂关节角
    leader_joints = leader.read_joint_states()
    
    # 计算正运动学（末端位姿）
    pos, alpha, psi = fk.compute_task_space(leader_joints[:5])
    
    # 记录数据：(位姿 → 关节角)
    dataset.append({
        'position': pos,
        'alpha': alpha,
        'psi': psi,
        'joints': leader_joints[:5],
        'timestamp': time.time() - start_time
    })
    
    # 可选：让从臂跟随（验证IK）
    # follower.write_joint_targets(leader_joints)
    
    time.sleep(0.02)  # 50Hz采样
    
    if len(dataset) % 1000 == 0:
        print(f"已采集 {len(dataset)} 个样本")

# 保存数据
save_dataset(dataset, "data/ik_dataset_real.h5")
```

**优点：**
- 数据来自真实运动
- 覆盖常用工作区域
- 包含动态轨迹信息

**建议采集场景：**
1. 抓取-放置循环动作（1000次）
2. 工作空间扫描（边界探索）
3. 典型任务轨迹（如写字、绘图）

#### 方法C：混合数据（最佳）⭐⭐⭐⭐

```
训练集 = 70% 随机采样 + 30% 真实示教
验证集 = 随机采样
测试集 = 真实任务轨迹
```

---

### 阶段3：算法实现（2-3天）

#### 实现3.1：解析逆运动学

```bash
# 已实现，测试精度
python test_analytical_ik.py
```

**验证方式：**
```python
# 1. 读取主臂当前位置
leader_joints = leader.read_joint_states()

# 2. 计算正运动学
target_pos, target_alpha, target_psi = fk.compute_task_space(leader_joints[:5])

# 3. 用解析IK求解
solution = analytical_ik.solve(target_pos, target_alpha, target_psi)

# 4. 对比
print(f"原始关节角: {np.rad2deg(leader_joints[:5])}")
print(f"IK解: {np.rad2deg(solution)}")
print(f"误差: {np.rad2deg(solution - leader_joints[:5])}")
```

#### 实现3.2：数值逆运动学

```bash
# 已实现，测试收敛性
python test_numerical_ik.py
```

#### 实现3.3：神经网络训练

```bash
# 训练基础MLP
python train_network.py \
    --dataset data/ik_dataset_mixed.h5 \
    --device cuda \
    --epochs 200

# 输出：models/ik_mlp_best.pth
```

**训练监控：**
- 位置误差 < 1mm
- 姿态误差 < 1度
- 训练集损失 vs 验证集损失

#### 实现3.4：混合方法

```python
# NN预测 + 数值修正
nn_solution = model.predict(input_vec)
refined_solution = numerical_ik.refine_solution(
    target_pos, target_alpha, target_psi,
    initial_solution=nn_solution,
    num_steps=3
)
```

---

### 阶段4：仿真评估（1天）

#### 评估4.1：批量对比实验

```bash
# 运行完整评估
python run_evaluation.py \
    --urdf models/so101.urdf \
    --model models/ik_mlp_best.pth \
    --num_test 1000 \
    --output results/ik_comparison.json
```

**测试集：**
1. **正常区域** - 工作空间中心
2. **边界区域** - 最大伸展
3. **奇异点附近** - 肘部伸直
4. **噪声输入** - 加入5mm位置误差

**评估指标：**
```
方法              位置误差(mm)  成功率(%)  时间(ms)
-----------------------------------------------------
解析IK              0.8         92.3       0.15
数值IK              0.3         98.5       12.3
神经网络IK          2.1         89.7       0.8
混合方法(3步)       0.4         97.8       2.4
```

#### 评估4.2：消融实验

**实验A：当前关节输入的影响**
```bash
# 不含当前关节（7维输入）
python train_network.py --no-current-joint

# 含当前关节（12维输入）
python train_network.py

# 对比连续运动时的解跳变
```

**实验B：数值修正步数的影响**
```bash
# 测试0, 1, 3, 5, 10步修正
python ablation_refinement_steps.py
```

**实验C：损失函数组件的影响**
```bash
# 只用关节损失
python train_network.py --loss-config joint_only

# 加入FK位置损失
python train_network.py --loss-config joint_position

# 完整损失（关节+位置+姿态）
python train_network.py --loss-config full
```

---

### 阶段5：真机验证（2-3天）⭐ 核心实验

#### 实验5.1：单点到达测试

**目标：** 验证4种IK方法的精度

```python
"""单点到达实验"""

# 1. 定义10个测试点（工作空间不同区域）
test_points = [
    ([0.20, 0.00, 0.15], 0.0, 0.0),  # 正前方
    ([0.15, 0.15, 0.10], -0.5, 0.0), # 右前方
    # ... 共10个点
]

# 2. 对每个测试点，用4种方法求解
for target_pos, target_alpha, target_psi in test_points:
    
    # 方法1：解析IK
    solution_analytical = analytical_ik.solve(target_pos, target_alpha, target_psi)
    
    # 方法2：数值IK
    solution_numerical = numerical_ik.solve(target_pos, target_alpha, target_psi)
    
    # 方法3：神经网络
    solution_nn = nn_model.predict(prepare_input(target_pos, target_alpha, target_psi, current))
    
    # 方法4：混合
    solution_hybrid = numerical_ik.refine_solution(target_pos, target_alpha, target_psi, solution_nn, 3)
    
    # 3. 让从臂执行每个解
    for method_name, solution in [("Analytical", solution_analytical), ...]:
        print(f"\n测试方法: {method_name}")
        
        # 发送到从臂
        follower.write_joint_targets(solution)
        time.sleep(2)  # 等待到达
        
        # 读取实际到达位置
        actual_joints = follower.read_joint_states()[:5]
        actual_pos, actual_alpha, actual_psi = fk.compute_task_space(actual_joints)
        
        # 计算误差
        pos_error = np.linalg.norm(actual_pos - target_pos)
        ori_error = abs(actual_alpha - target_alpha) + abs(actual_psi - target_psi)
        
        print(f"  目标位置: {target_pos}")
        print(f"  实际位置: {actual_pos}")
        print(f"  位置误差: {pos_error*1000:.2f} mm")
        print(f"  姿态误差: {np.rad2deg(ori_error):.2f} 度")
        
        # 记录数据
        log_result(method_name, target_pos, actual_pos, pos_error, ori_error)
```

#### 实验5.2：连续轨迹跟踪

**目标：** 测试连续运动的平滑性

```python
"""轨迹跟踪实验"""

# 1. 定义轨迹（如圆形、方形、8字形）
def generate_circle_trajectory(center, radius, num_points=100):
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]
        points.append(([x, y, z], 0.0, 0.0))
    return points

trajectory = generate_circle_trajectory([0.20, 0.0, 0.15], 0.05)

# 2. 用不同方法执行轨迹
for method_name, ik_solver in methods.items():
    print(f"\n执行轨迹: {method_name}")
    
    recorded_trajectory = []
    current_joints = follower.read_joint_states()[:5]
    
    for target_pos, target_alpha, target_psi in trajectory:
        # 求解IK
        solution = ik_solver(target_pos, target_alpha, target_psi, current_joints)
        
        # 发送到从臂
        follower.write_joint_targets(solution)
        time.sleep(0.05)  # 20Hz控制频率
        
        # 读取实际位置
        actual_joints = follower.read_joint_states()[:5]
        actual_pos, _, _ = fk.compute_task_space(actual_joints)
        
        recorded_trajectory.append(actual_pos)
        current_joints = actual_joints
    
    # 3. 分析轨迹质量
    # - 轨迹误差
    # - 平滑度（加速度）
    # - 最大关节跳变
    analyze_trajectory(trajectory, recorded_trajectory, method_name)
```

#### 实验5.3：主从遥操作演示

**目标：** 实时IK跟随（论文视频素材）

```python
"""主从遥操作实验"""

leader = SO101Robot(port="COM23")
follower = SO101Robot(port="COM24")

print("开始主从遥操作...")
print("请操作主臂，从臂将实时跟随")

while True:
    # 1. 读取主臂关节角
    leader_joints = leader.read_joint_states()[:5]
    
    # 2. 计算主臂末端位姿
    target_pos, target_alpha, target_psi = fk.compute_task_space(leader_joints)
    
    # 3. 用混合IK求解从臂关节角
    current_follower = follower.read_joint_states()[:5]
    solution = hybrid_ik(target_pos, target_alpha, target_psi, current_follower)
    
    # 4. 发送到从臂
    follower.write_joint_targets(solution)
    
    # 5. 记录误差
    actual_follower = follower.read_joint_states()[:5]
    actual_pos, _, _ = fk.compute_task_space(actual_follower)
    error = np.linalg.norm(actual_pos - target_pos)
    
    print(f"\r跟随误差: {error*1000:.2f} mm", end='')
    
    time.sleep(0.02)  # 50Hz
```

---

### 阶段6：结果分析与论文（2-3天）

#### 数据整理

**表1：精度对比**
```
方法          位置误差(mm)  姿态误差(°)  成功率(%)
----------------------------------------------------
解析IK          0.85±0.32    1.2±0.5      92.3
数值IK          0.28±0.15    0.4±0.2      98.5
神经网络        2.14±0.78    2.8±1.1      89.7
混合(NN+3步)    0.41±0.18    0.6±0.3      97.8
```

**表2：速度对比**
```
方法          平均(ms)  P95(ms)  最大(ms)
------------------------------------------
解析IK         0.15     0.18      0.25
数值IK        12.34    18.56     45.23
神经网络       0.82     1.05      1.34
混合          2.41     3.87      6.12
```

**表3：真机验证**
```
测试场景      解析IK  数值IK  NN    混合
--------------------------------------------
单点到达      88%     96%     85%   95%
连续轨迹      82%     94%     79%   93%
边界区域      65%     89%     71%   87%
快速运动      90%     91%     83%   92%
```

#### 生成图表

```bash
# 生成对比图
python utils/visualization.py \
    --results results/ik_comparison.json \
    --output_dir paper/002/figures

# 生成：
# - success_rate_comparison.png
# - position_error_comparison.png
# - solve_time_comparison.png
# - accuracy_speed_tradeoff.png
```

#### 论文撰写

**章节结构：**

1. **Introduction** - 问题背景和动机
2. **Background** - SO-101机械臂和相关工作
3. **Method**
   - 3.1 解析IK推导
   - 3.2 数值IK算法
   - 3.3 神经网络架构
   - 3.4 混合方法
4. **Experiments**
   - 4.1 实验设置
   - 4.2 仿真结果
   - 4.3 真机验证
   - 4.4 消融实验
5. **Conclusion** - 结论和未来工作

**编译论文：**
```bash
cd E:\exp\paper\002
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📅 时间规划（7-10天）

```
Day 1-2:   获取URDF + 验证FK + 生成数据
Day 3-4:   训练神经网络 + 仿真评估
Day 5-6:   真机单点测试 + 轨迹跟踪
Day 7:     主从遥操作 + 录制视频
Day 8-9:   数据分析 + 生成图表
Day 10:    论文撰写 + 编译PDF
```

---

## 🎬 实验视频建议

录制以下场景（用于论文supplementary）：

1. **单点到达演示** - 4种方法对比（20秒）
2. **圆形轨迹** - 展示平滑跟踪（30秒）
3. **主从遥操作** - 实时IK跟随（1分钟）
4. **失败案例** - 奇异点附近的问题（10秒）

---

## 💡 关键要点

### 实验成功的关键：

1. **准确的URDF** - FK必须正确
2. **充足的数据** - 至少10万样本
3. **安全测试** - 低速、小步长、限位保护
4. **完整记录** - 每个实验保存日志

### 预期困难与解决：

**问题1：神经网络精度不够**
- 增加训练数据
- 调整网络大小
- 加入FK损失

**问题2：真机抖动**
- 降低速度
- 增加轨迹插值点
- 检查电源供应

**问题3：奇异点处失败**
- 限制工作空间
- 增加阻尼系数
- 使用混合方法

---

## 📂 代码组织

```
实验代码目录：
src/002/
├── experiments/              # 实验脚本
│   ├── collect_real_data.py        # 主臂数据采集
│   ├── single_point_test.py        # 单点到达实验
│   ├── trajectory_tracking.py      # 轨迹跟踪实验
│   ├── teleoperation_demo.py       # 主从遥操作
│   └── ablation_studies.py         # 消融实验
│
├── results/                  # 实验结果
│   ├── ik_comparison.json          # 评估结果
│   ├── real_robot_logs/            # 真机日志
│   └── videos/                     # 实验视频
```

---

## 🚀 立即开始

**第一步：确认硬件状态**
```bash
cd E:\exp\src\002
python find_robot.py
# 确认COM23和COM24都在线
```

**第二步：获取URDF**
- 下载LeRobot仓库
- 找到SO-101 URDF文件
- 放到 `models/so101.urdf`

**第三步：开始数据生成**
```bash
python generate_dataset.py --num_samples 150000
```

---

这是一个完整的、可执行的实验方案。您想从哪一步开始？
