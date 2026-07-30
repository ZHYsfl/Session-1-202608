# SO-101 机械臂连接快速指南

## 🚀 5分钟快速开始

### 第一步：硬件连接（5分钟）

1. **检查组件**
   - [ ] SO-101机械臂
   - [ ] USB转TTL适配器
   - [ ] 12V电源适配器（5A+）
   - [ ] USB数据线

2. **接线**
   ```
   电脑USB → USB转TTL适配器 → 控制板 → 舵机链
                                 ↑
                            12V电源
   ```
   
3. **重要提醒**
   - ⚠️ 先连接USB和舵机线，**最后**接12V电源
   - ⚠️ 确保机械臂周围1米内无障碍物
   - ⚠️ 准备好随时拔掉电源（紧急停止）

### 第二步：软件测试（3分钟）

1. **测试串口连接**
   ```bash
   cd E:\exp\src\002
   python robot_control/test_connection.py
   ```
   - 会列出所有可用COM端口
   - 自动测试连接是否正常

2. **扫描舵机**
   ```bash
   python robot_control/scan_servos.py
   ```
   - 输入COM端口号（如 COM3）
   - 选择协议（Feetech或Dynamixel）
   - 应该发现6个舵机（ID 1-6）

### 第三步：问题诊断

**如果找不到COM端口：**
- 检查设备管理器（Win+X → 设备管理器）
- 安装USB转TTL驱动程序

**如果扫描不到舵机：**
- 确认12V电源已接通（LED应亮起）
- 尝试不同波特率（1000000 或 115200）
- 检查DATA线是否连接正确

**如果舵机抖动：**
- 检查电源供电能力（电流够不够）
- 降低运动速度
- 检查有无ID冲突

---

## 📖 详细文档索引

### 基础文档
- **HARDWARE_SETUP.md** - 完整硬件连接教程（包含安全注意事项）
- **QUICKSTART.md** - 软件使用快速指南
- **PROJECT_STATUS.md** - 项目进度和待办事项
- **README.md** - 项目总览

### 测试脚本
```bash
# 1. 测试串口连接
python robot_control/test_connection.py

# 2. 扫描舵机ID
python robot_control/scan_servos.py

# 3. 运行单元测试（URDF加载后）
python tests/run_tests.py
```

### 数据流程脚本
```bash
# 1. 生成训练数据（1-2小时）
python generate_dataset.py --num_samples 150000

# 2. 训练神经网络（2-4小时GPU）
python train_network.py --device cuda --epochs 200

# 3. 运行评估对比（30-60分钟）
python run_evaluation.py --num_test 1000
```

---

## ⚠️ 安全第一

### 启动前检查清单
- [ ] 机械臂周围至少1米范围清空
- [ ] 所有人员远离运动范围
- [ ] 电源插头易于拔出（紧急停止）
- [ ] 关节未处于极限位置

### 禁止操作
- ❌ 不要在通电时强行移动关节
- ❌ 不要发送超过±30°的单步指令
- ❌ 不要在机械臂运动时触摸
- ❌ 不要超出关节限位运行

### 紧急情况处理
1. **立即拔掉12V电源插头**
2. 检查代码错误
3. 降低速度重新测试

---

## 📂 项目结构速查

```
src/002/
├── robot_control/              # 真机控制
│   ├── test_connection.py     # [第一步] 测试串口
│   ├── scan_servos.py          # [第二步] 扫描舵机
│   └── so101_interface.py      # 机器人接口（待完善）
│
├── kinematics/                 # 运动学
│   ├── forward_kinematics.py   # 正运动学（需URDF）
│   ├── analytical_ik.py        # 解析逆运动学
│   └── numerical_ik.py         # 数值逆运动学
│
├── neural_network/             # 神经网络
│   ├── model.py                # 网络架构
│   ├── dataset.py              # 数据集
│   └── trainer.py              # 训练器
│
├── configs/
│   └── robot_config.yaml       # 配置文件
│
├── generate_dataset.py         # 生成训练数据
├── train_network.py            # 训练网络
└── run_evaluation.py           # 评估对比
```

---

## 🎯 实验流程总览

### 阶段1：硬件连接（今天）
- [x] 组装机械臂
- [ ] 连接电脑并测试
- [ ] 扫描并确认6个舵机
- [ ] 配置舵机ID（如需要）

### 阶段2：获取URDF（明天）
- [ ] 从LeRobot获取SO-101 URDF
- [ ] 验证正运动学计算
- [ ] 更新配置文件中的连杆长度

### 阶段3：数据生成（1天）
- [ ] 生成15万训练样本
- [ ] 验证数据分布
- [ ] 划分训练/验证/测试集

### 阶段4：训练神经网络（1天）
- [ ] 训练基础MLP
- [ ] 添加当前关节状态输入
- [ ] 添加FK损失
- [ ] 监控训练曲线

### 阶段5：评估实验（1天）
- [ ] 对比4种IK方法
- [ ] 运行消融实验
- [ ] 生成对比图表
- [ ] 记录量化指标

### 阶段6：真机验证（1-2天）
- [ ] 单关节低速测试
- [ ] 单点到达测试
- [ ] 连续轨迹测试
- [ ] 安全性验证

### 阶段7：论文撰写（2-3天）
- [ ] 填充实验数据到表格
- [ ] 生成结果图表
- [ ] 撰写方法和实验章节
- [ ] 编译LaTeX生成PDF

**总计预估时间：7-10天**

---

## 🔗 快速链接

### 在线资源
- LeRobot GitHub: https://github.com/huggingface/lerobot
- Feetech舵机文档: https://www.feetechrc.com/
- Dynamixel文档: https://emanual.robotis.com/

### 社区支持
- LeRobot Discord
- ROS Robotics论坛
- Stack Exchange Robotics

### 驱动下载
- CH340: http://www.wch.cn/downloads/CH341SER_EXE.html
- CP2102: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
- FT232: https://ftdichip.com/drivers/vcp-drivers/

---

## 💡 常见问题

**Q: 需要什么Python版本？**
A: Python 3.8 或更高版本

**Q: 必须使用GPU吗？**
A: 不是必须，但GPU可以大幅加速训练（2-4小时 vs 12-24小时）

**Q: 机械臂可以直接使用NN输出吗？**
A: 不可以！必须经过安全检查和数值修正，参考混合方法

**Q: 如何备份实验数据？**
A: 定期commit到Git，重要结果保存多份

**Q: 论文必须用LaTeX吗？**
A: 是的，IEEE格式要求LaTeX（已提供模板）

---

## 📞 遇到问题？

1. **查看详细文档：** HARDWARE_SETUP.md
2. **检查项目状态：** PROJECT_STATUS.md
3. **运行测试脚本：** tests/run_tests.py
4. **查看代码注释：** 所有模块都有详细注释

---

**祝实验顺利！🎉**

创建时间：2026-07-30  
学生：002  
项目：SO-101 逆运动学研究
