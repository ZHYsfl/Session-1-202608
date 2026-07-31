# Robot Collision Prediction and Avoidance

本项目使用 Pygame 构建移动机器人运动环境，并实现：

1. 基于多角度测距传感器的碰撞数据采集；
2. 基于 PyTorch MLP 的未来30帧碰撞预测；
3. 基于 DQN 的机器人碰撞规避；
4. 减速、镜面反射和重新加速控制；
5. 不同强化学习模型的训练与评价。

## 最终DQN模型

最终采用 V3 最佳模型：

- 状态维数：14
- 动作数量：2
- 动作0：保持当前方向
- 动作1：减速—镜面反射—重新加速
- 最佳模型：best_v3_at_30000_steps

## 安装依赖

```powershell
python -m pip install -r requirements.txt


