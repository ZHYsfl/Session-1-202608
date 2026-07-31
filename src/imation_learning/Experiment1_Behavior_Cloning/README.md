# 实验一：基于 CNN 的自动驾驶行为克隆

本目录保存已经完成的 Udacity 模拟器行为克隆实验，包括：

- `behavioral-cloning/model.py`：TensorFlow/Keras 训练入口；
- `behavioral-cloning/data.py`：图像预处理和数据增强；
- `behavioral-cloning/drive.py`：模拟器闭环控制接口；
- `behavioral-cloning/model.h5`：已训练 CNN；
- `results/loss_curve.png`：已完成训练的 Loss 曲线；
- `results/metrics.json`：供实验三和总报告读取的结构化指标。

## 集成验证

```powershell
python verify_experiment.py
```

该命令校验训练源码、模型、Loss 曲线和报告图片，并刷新
`results/metrics.json`。它不会假装重新训练模型。

## 完整重训边界

原始摄像头数据没有保存在当前仓库。完整重训需要将 Udacity 模拟器
`driving_log.csv` 和 `IMG` 图像目录放到：

```text
Experiment1_Behavior_Cloning/data/
```

然后执行：

```powershell
cd behavioral-cloning
python model.py
```

当前集成版本将实验一视为“已完成模型的可验证归档”；实验二和实验四则
由一键脚本实际重新训练。
