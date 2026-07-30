# SO-101 逆运动学实验 - 快速启动指南

## ✅ 当前状态

- ✅ 项目框架搭建完成
- ✅ 双机械臂连接成功（COM23, COM24）
- ✅ 舵机扫描完成（6个舵机，ID 1-6）
- ✅ URDF文件已放置
- ⏳ 正在安装依赖包（urdfpy, pyyaml）

---

## 🚀 实验启动步骤

### 第1步：验证URDF和正运动学（5分钟）

```bash
cd E:\exp\src\002

# 1. 测试URDF加载
python -c "
from kinematics import ForwardKinematics
fk = ForwardKinematics('models/so101.urdf')
print('✓ URDF加载成功')
"

# 2. 测试正运动学
python tests/test_forward_kinematics.py
```

**预期结果：**
- URDF加载成功
- FK计算正常
- Jacobian矩阵正确

---

### 第2步：测试解析逆运动学（5分钟）

```bash
# 测试解析IK
python tests/test_analytical_ik.py
```

**预期结果：**
- 能够求解可达点
- 拒绝不可达点
- 两个分支解（肘上/肘下）

---

### 第3步：生成训练数据（1-2小时）

```bash
# 生成15万训练样本
python generate_dataset.py \
    --config configs/robot_config.yaml \
    --urdf models/so101.urdf \
    --num_samples 150000 \
    --output data/ik_dataset.h5
```

**数据生成内容：**
- 随机采样关节配置
- 计算正运动学得到末端位姿
- 用解析IK生成标签
- 分为训练集（70%）、验证集（15%）、测试集（15%）

**进度监控：**
```
已生成 10000/150000 样本...
已生成 20000/150000 样本...
...
数据集生成完成: 150000 samples
```

---

### 第4步：训练神经网络（2-4小时 GPU / 12-24小时 CPU）

```bash
# 训练神经网络IK
python train_network.py \
    --config configs/robot_config.yaml \
    --dataset data/ik_dataset.h5 \
    --output_dir models/ \
    --device cuda \
    --epochs 200
```

**训练监控：**
```
Epoch 1: Train Loss = 0.0523, Val Loss = 0.0487
Epoch 2: Train Loss = 0.0412, Val Loss = 0.0398
...
Epoch 50: Train Loss = 0.0089, Val Loss = 0.0092
  -> Saved best model (val_loss: 0.0092)
```

**目标指标：**
- 位置误差 < 2mm
- 姿态误差 < 2度
- 成功率 > 85%

---

### 第5步：运行评估对比（30-60分钟）

```bash
# 完整评估4种IK方法
python run_evaluation.py \
    --config configs/robot_config.yaml \
    --urdf models/so101.urdf \
    --model models/ik_mlp_best.pth \
    --num_test 1000 \
    --output results/ik_comparison.json
```

**评估内容：**
- 解析IK
- 数值IK（阻尼最小二乘）
- 神经网络IK
- 混合方法（NN + 1/3/5步修正）

**测试场景：**
- 正常区域
- 工作空间边界
- 近奇异点
- 噪声输入

---

### 第6步：生成结果图表（5分钟）

```bash
# 可视化结果
python -m utils.visualization \
    --results results/ik_comparison.json \
    --output_dir paper/002/figures
```

**生成图表：**
- success_rate_comparison.png
- position_error_comparison.png
- solve_time_comparison.png
- accuracy_speed_tradeoff.png

---

### 第7步：真机验证（可选，1-2天）

⚠️ **需要接通12V电源，确保安全！**

```bash
# 单点到达测试
python experiments/single_point_test.py \
    --leader_port COM23 \
    --follower_port COM24

# 轨迹跟踪测试
python experiments/trajectory_tracking.py \
    --follower_port COM24

# 主从遥操作演示
python experiments/teleoperation_demo.py \
    --leader_port COM23 \
    --follower_port COM24
```

---

## 📊 预期实验结果

### 表1：精度对比（仿真）

| 方法 | 位置误差(mm) | 姿态误差(°) | 成功率(%) |
|------|--------------|-------------|-----------|
| 解析IK | 0.85±0.32 | 1.2±0.5 | 92.3 |
| 数值IK | 0.28±0.15 | 0.4±0.2 | 98.5 |
| 神经网络 | 2.14±0.78 | 2.8±1.1 | 89.7 |
| 混合(3步) | 0.41±0.18 | 0.6±0.3 | 97.8 |

### 表2：速度对比

| 方法 | 平均时间(ms) | P95时间(ms) |
|------|--------------|-------------|
| 解析IK | 0.15 | 0.18 |
| 数值IK | 12.34 | 18.56 |
| 神经网络 | 0.82 | 1.05 |
| 混合 | 2.41 | 3.87 |

---

## 🎓 论文撰写（2-3天）

### 论文结构

**文件位置：** `paper/002/main.tex`

**章节内容：**
1. **Introduction** - 研究动机和贡献
2. **Background** - SO-101介绍和相关工作
3. **Method**
   - 3.1 解析IK推导
   - 3.2 数值IK算法
   - 3.3 神经网络架构
   - 3.4 混合方法
4. **Experiments**
   - 4.1 实验设置
   - 4.2 仿真结果
   - 4.3 真机验证（可选）
   - 4.4 消融实验
5. **Conclusion**

### 编译论文

```bash
cd E:\exp\paper\002

# Windows: 使用LaTeX编辑器（如TeXworks, Overleaf）
# 或命令行（需安装MikTeX）
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## ⏱️ 时间规划

### 方案A：仿真优先（7天）

```
Day 1:   验证FK + 测试解析IK + 开始生成数据
Day 2:   完成数据生成 + 开始训练
Day 3:   完成训练 + 运行评估
Day 4:   消融实验 + 生成图表
Day 5-7: 论文撰写 + 编译
```

### 方案B：包含真机（10天）

```
Day 1-2:  验证FK + 生成数据
Day 3-4:  训练网络 + 仿真评估
Day 5-6:  真机单点测试 + 轨迹跟踪
Day 7:    主从遥操作演示
Day 8-10: 数据分析 + 论文撰写
```

---

## 📋 检查清单

### 软件准备
- [x] 项目框架
- [x] 双机械臂连接
- [x] URDF文件
- [ ] 依赖包安装（进行中）
- [ ] FK验证
- [ ] IK测试

### 实验执行
- [ ] 数据生成
- [ ] 网络训练
- [ ] 仿真评估
- [ ] 结果可视化
- [ ] （可选）真机验证

### 论文撰写
- [ ] 填充实验数据
- [ ] 生成图表
- [ ] 撰写各章节
- [ ] 编译PDF

---

## 🛠️ 故障排除

### 问题1：URDF加载失败
**解决：**
```bash
pip install urdfpy trimesh networkx
```

### 问题2：训练速度慢
**解决：**
- 使用GPU：`--device cuda`
- 减少样本：`--num_samples 50000`
- 减小网络：修改 `hidden_dims: [128, 128]`

### 问题3：内存不足
**解决：**
- 减小batch_size：`batch_size: 128`
- 使用更少数据
- 关闭其他程序

### 问题4：精度不够
**解决：**
- 增加训练数据
- 增加训练轮数：`--epochs 300`
- 调整损失权重
- 使用混合方法

---

## 📞 下一步

**现在立即执行：**

1. **等待依赖安装完成**（后台进行中）
2. **验证URDF**
   ```bash
   cd E:\exp\src\002
   python -c "from kinematics import ForwardKinematics; fk = ForwardKinematics('models/so101.urdf'); print('OK')"
   ```
3. **开始数据生成**

---

## 💡 提示

- **数据生成**可以在后台运行（1-2小时）
- **网络训练**推荐使用GPU
- **真机测试**记得接通12V电源并确保安全
- **论文撰写**可以在训练期间同时进行

---

**准备好了吗？告诉我依赖安装完成后继续！** 🚀
