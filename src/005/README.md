
# SLAM Mapping and Analysis

  Simultaneous Localization and Mapping (SLAM) 实验项目，基于 BreezySLAM 库在真实 LiDAR
  数据集上实现建图与轨迹分析。

  ## 项目概述

  本项目使用巴黎矿业大学（Paris Mines Tech）的 CoreSLAM 公开数据集，对 BreezySLAM
  算法在不同条件下的建图性能进行了系统对比实验。项目包含自实现的占用栅格建图算法（Occupancy Grid
  Mapping）以及完整的轨迹分析工具。

  ### 核心功能

  - **BreezySLAM 建图**：基于粒子滤波的 Rao-Blackwellized SLAM 算法
  - **里程计融合**：支持纯激光 SLAM 与激光 + 里程计融合两种模式
  - **轨迹分析**：自动计算路径长度、覆盖面积等定量指标
  - **对比可视化**：生成 2×2 对比拼接图，直观展示不同参数效果
  - **占用栅格建图**：自定义实现的占用栅格地图算法（inverse sensor model + Bayesian log-odds update）

  ## 项目结构

  005_project/
  ├── src/
  │   ├── occupancy_grid.py       # 自定义占用栅格建图算法
  │   └── slam_analysis.py        # 轨迹与地图质量分析工具
  ├── scripts/
  │   └── run_comparison.py       # 主实验脚本（运行所有对比实验）
  ├── results/
  │   ├── exp1.png                # 办公楼场景 - 无里程计
  │   ├── exp1_odom.png           # 办公楼场景 - 含里程计
  │   ├── exp2.png                # 实验室场景 - 无里程计
  │   ├── exp2_odom.png           # 实验室场景 - 含里程计
  │   ├── comparison_grid.png     # 四图对比拼接
  │   └── comparison.json         # 定量实验结果
  ├── README.md
  └── .gitignore

  ## 数据集

  | 数据集 | 环境 | 扫描数 | 描述 |
  |--------|------|--------|------|
  | exp1 | 办公楼（Office） | 756 | 走廊 + 房间，较空旷 |
  | exp2 | 实验室（Laboratory） | 641 | 障碍物较多，环境复杂 |

  数据来源：Paris Mines Tech CoreSLAM dataset（Steux & Hamzaoui, ICARCV 2010）

  ## 环境要求

  - Python 3.8+
  - NumPy
  - Pillow (PIL)
  - BreezySLAM

  ## 安装与使用

  ### 1. 安装依赖

  ```bash
  pip install numpy pillow

  2. 安装 BreezySLAM

  git clone https://github.com/simondlevy/BreezySLAM.git
  cd BreezySLAM/python && sudo python3 setup.py install

  3. 运行对比实验

  cd 005_project
  python3 scripts/run_comparison.py

  算法说明

  占用栅格地图（Occupancy Grid Map）

  地图被离散化为均匀网格，每个格子存储该位置被占用的概率。使用逆传感器模型（Inverse Sensor Model）和贝叶斯
  log-odds 更新公式：

  $$l_{t,i} = l_{t-1,i} + \text{inverse_sensor_model}(z_t, x_t, m_i)$$

  其中：
  - $l_{t,i}$ 是格子 $i$ 在时刻 $t$ 的 log-odds 值
  - $z_t$ 是激光观测数据
  - $x_t$ 是机器人位姿
  - $m_i$ 是地图格子

  Bresenham 射线追踪

  使用 Bresenham 直线算法确定激光束穿过的所有格子，将其标记为"空闲"；将激光终点标记为"占用"。

  实验结果

  定量结果

  ┌──────┬────────┬────────┬────────┬──────────┬──────────────┬───────────────┐
  │ 实验 │ 数据集 │ 里程计 │ 扫描数 │ 用时 (s) │ 路径长度 (m) │ 覆盖面积 (m²) │
  ├──────┼────────┼────────┼────────┼──────────┼──────────────┼───────────────┤
  │ 1    │ exp1   │ 否     │ 756    │ 2.69     │ 45.18        │ 153.17        │
  ├──────┼────────┼────────┼────────┼──────────┼──────────────┼───────────────┤
  │ 2    │ exp1   │ 是     │ 756    │ 2.32     │ 46.43        │ 160.35        │
  ├──────┼────────┼────────┼────────┼──────────┼──────────────┼───────────────┤
  │ 3    │ exp2   │ 否     │ 641    │ 2.13     │ 43.10        │ 146.73        │
  ├──────┼────────┼────────┼────────┼──────────┼──────────────┼───────────────┤
  │ 4    │ exp2   │ 是     │ 641    │ 2.05     │ 45.13        │ 160.46        │
  └──────┴────────┴────────┴────────┴──────────┴──────────────┴───────────────┘

  主要发现

  1. 里程计的作用：融合里程计数据后，机器人的轨迹路径更长（覆盖率更高），建图质量更稳定，说明里程计信息能
  有效修正纯激光 SLAM 的累积误差
  2. 处理速度：算法处理速度约为 270–320 scans/s，完全满足实时应用需求（典型激光雷达扫描频率为 10Hz）
  3. 数据集差异：办公楼场景（exp1）空间较开阔，建图效果更规整；实验室场景（exp2）障碍物更密集，对 SLAM
  算法的鲁棒性要求更高
  4. 定量对比：有里程计时，exp1 的路径长度从 45.18m 增加到 46.43m（+2.8%），覆盖面积从 153.17m² 增加到
  160.35m²（+4.7%），表明里程计能有效扩展探索范围

  参考文献

  1. BreezySLAM 仓库：https://github.com/simondlevy/BreezySLAM
  2. CoreSLAM: Steux, B. & El Hamzaoui, O. (2010). "CoreSLAM: a SLAM Algorithm in less than 200 lines of C
  code". ICARCV 2010.
  3. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press.
  4. Grisetti, G., Stachniss, C., & Burgard, W. (2007). "Improved Techniques for Grid Mapping with
  Rao-Blackwellized Particle Filters". IEEE Transactions on Robotics.

  许可证

  本项目基于 BreezySLAM（LGPL-3.0）进行实验与分析。