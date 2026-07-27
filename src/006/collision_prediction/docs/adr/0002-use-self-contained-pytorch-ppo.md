# 使用自包含的 PyTorch PPO

强化学习训练使用项目内实现的 PyTorch PPO，而不依赖 Stable-Baselines3。这样可让批量环境、rollout、GAE 和策略更新保持为 GPU 张量，减少 CPU 与 GPU 间的数据搬运，并便于记录实验细节；代价是项目必须通过单元测试和训练诊断自行保证 PPO 实现正确。
