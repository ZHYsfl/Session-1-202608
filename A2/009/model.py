"""
model.py — 模型侧（009）神经网络定义与权重管理

VOA (Vision-Other-Action) 多模态感知防碰撞网络架构：
  - PolicyNet: VOA Action 分支，SAC 随机策略网络，输出小车控制动作
  - QNet: ANN 碰撞预测双 Q 网络，评估状态-动作对的碰撞风险与价值
  - build_models(): 统一初始化全部网络（actor、critic、target_critic）
  - save_checkpoint() / load_checkpoint(): 权重持久化，含版本元信息

纯 PyTorch 实现，不引入第三方库以外的依赖。
仅定义网络结构与存取接口；损失计算、梯度反向传播、训练循环由 003 算法侧完成。

协议版本: 1.0
作者: 009
"""

import os
import math
import argparse
import tempfile
from datetime import datetime, timezone
from typing import Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 — 预留给 003 算法侧使用

# ============================================================================
# 全局常量（§7.1）
# 后续切换真机硬件：仅需修改本节常量，网络推理逻辑不动。
# 真机若更换雷达/超声波阵列，修改 OBS_DIM 对应的 lidar_count 即可。
# ============================================================================
OBS_DIM: int = 68         # 网络输入维度 = 64(lidar) + 2(goal: dist,bearing) + 2(vel: v,w)
ACT_DIM: int = 2          # 网络输出动作维度 = (v, w)
LOG_STD_MIN: float = -5.0 # log 标准差下界（防止方差坍缩为 0）
LOG_STD_MAX: float = 2.0  # log 标准差上界（防止方差爆炸）


# ============================================================================
# PolicyNet — SAC 随机策略网络（VOA Action 分支）
# ============================================================================
class PolicyNet(nn.Module):
    """SAC 高斯策略网络。

    输入归一化后的 68 维观测向量，输出动作分布参数 (mean, log_std)。
    sample() 通过重参数化采样 + tanh 压缩，保证动作天然落在 (-1, 1)² 有界空间内。

    架构：MLP 共享躯干 → 双头输出 (mean_head, log_std_head)
    """

    def __init__(self,
                 obs_dim: int = OBS_DIM,
                 act_dim: int = ACT_DIM,
                 hidden: tuple = (256, 256)) -> None:
        """初始化策略网络。

        Args:
            obs_dim: 观测向量维度，默认 68
            act_dim: 动作空间维度，默认 2
            hidden: 隐藏层各层神经元数量元组，默认 (256, 256)
                    支持任意长度元组（如 (128, 128)、(512, 256, 128)）
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        # ---- 共享特征提取层（MLP 躯干） ----
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature_net = nn.Sequential(*layers)

        # ---- 输出双头：均值 (mean) 和对数标准差 (log_std) ----
        last_hidden = hidden[-1] if hidden else 256
        self.mean_head = nn.Linear(last_hidden, act_dim)
        self.log_std_head = nn.Linear(last_hidden, act_dim)

        # 初始化：log_std_head 偏置设为较小负值，使初始探索偏向保守
        nn.init.uniform_(self.log_std_head.bias, -1.0, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播：给定观测，输出动作分布参数。

        Args:
            obs: 归一化观测张量，shape [B, obs_dim]，dtype float32

        Returns:
            mean:    动作均值，shape [B, act_dim]
            log_std: 动作对数标准差，shape [B, act_dim]，
                     已 clamp 到 [LOG_STD_MIN, LOG_STD_MAX]
        """
        feat = self.feature_net(obs)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        # clamp 保证数值稳定，防止梯度爆炸/坍缩
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """重参数化采样 + tanh 压缩，生成 SAC 动作。

        流程：
        1. forward() 获取 (mean, log_std)
        2. 重参数化：a_raw = μ + σ·ε,  ε ~ N(0,I)
        3. tanh 压缩至 (-1, 1)
        4. 对数概率修正（tanh 的变密度效应）

        Args:
            obs: 归一化观测张量，shape [B, obs_dim]，dtype float32

        Returns:
            action:   采样动作，shape [B, act_dim]，每个分量 ∈ (-1, 1)
            log_prob: 对数概率密度，shape [B]（已含 tanh 修正项）
            mean:     动作均值，shape [B, act_dim]（评估时可直接用作确定性策略）
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # ---- 重参数化采样 ----
        # a_raw = μ + σ ⊙ ε,  ε ~ N(0, I)
        eps = torch.randn_like(mean)
        action_raw = mean + std * eps

        # ---- 原始高斯对数概率 ----
        # log π(a_raw|s) = -½ Σ [ ln(2π) + 2·ln σ + (a_raw - μ)²/σ² ]
        log_prob_raw = -0.5 * (
            ((action_raw - mean) / (std + 1e-8)).pow(2)
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        log_prob_raw = log_prob_raw.sum(dim=-1)  # → [B]

        # ---- tanh 压缩 ----
        # a = tanh(a_raw) ∈ (-1, 1)
        action = torch.tanh(action_raw)

        # ---- tanh 对数概率修正 ----
        # log π(a|s) = log π(a_raw|s) − Σ log(1 − tanh²(a_raw) + ε)
        log_prob = log_prob_raw - torch.sum(
            torch.log(1.0 - action.pow(2) + 1e-6), dim=-1
        )

        return action, log_prob, mean


# ============================================================================
# QNet — 双 Q 价值网络（ANN 碰撞预测分支）
# ============================================================================
class QNet(nn.Module):
    """SAC 双 Q 网络。

    对状态-动作拼接对 (obs, act) 评估 Q 值。
    003 算法侧取两网输出的较小值缓解 Q 值高估偏差（clipped double-Q trick）。

    架构：MLP（obs⊕act 拼接输入）→ 单一 Q 值输出
    """

    def __init__(self,
                 obs_dim: int = OBS_DIM,
                 act_dim: int = ACT_DIM,
                 hidden: tuple = (256, 256)) -> None:
        """初始化 Q 网络。

        Args:
            obs_dim: 观测向量维度，默认 68
            act_dim: 动作空间维度，默认 2
            hidden: 隐藏层各层神经元数量元组，默认 (256, 256)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        # ---- 特征提取层（输入：obs ⊕ act） ----
        layers = []
        in_dim = obs_dim + act_dim  # 拼接状态与动作
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.net = nn.Sequential(*layers)

        # ---- 输出层：单一标量 Q 值 ----
        last_hidden = hidden[-1] if hidden else 256
        self.q_head = nn.Linear(last_hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """前向传播：评估 (obs, act) 的 Q 值。

        Args:
            obs: 归一化观测张量，shape [B, obs_dim]，dtype float32
            act: 动作张量，shape [B, act_dim]，dtype float32

        Returns:
            Q 值，shape [B, 1]，dtype float32
        """
        # 在最后一维拼接状态与动作
        x = torch.cat([obs, act], dim=-1)
        x = self.net(x)
        q = self.q_head(x)
        return q


# ============================================================================
# build_models — 统一初始化全部网络
# ============================================================================
def build_models(device) -> dict:
    """初始化 SAC 所需全部网络并移动到指定设备。

    创建：
      - actor (PolicyNet)
      - critic1, critic2 (QNet — 双 Q)
      - critic1_target, critic2_target (QNet 副本，冻结梯度)

    target 网络初始权重拷贝自对应 critic，requires_grad 置为 False。
    003 算法侧通过 soft update (EMA) 手动更新 target 权重。

    Args:
        device: torch.device 对象或字符串，如 torch.device("cuda:0") 或 "cpu"

    Returns:
        dict，固定键名：
            "actor"              — PolicyNet 实例
            "critic1"            — QNet 实例
            "critic2"            — QNet 实例
            "critic1_target"     — QNet 实例（冻结）
            "critic2_target"     — QNet 实例（冻结）
    """
    actor = PolicyNet().to(device)
    critic1 = QNet().to(device)
    critic2 = QNet().to(device)
    critic1_target = QNet().to(device)
    critic2_target = QNet().to(device)

    # target 初始权重与对应 critic 完全一致
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    # 冻结 target 网络梯度（参数不参与反向传播，仅通过 soft update 更新）
    for param in critic1_target.parameters():
        param.requires_grad_(False)
    for param in critic2_target.parameters():
        param.requires_grad_(False)

    return {
        "actor": actor,
        "critic1": critic1,
        "critic2": critic2,
        "critic1_target": critic1_target,
        "critic2_target": critic2_target,
    }


# ============================================================================
# save_checkpoint — 模型权重保存
# ============================================================================
def save_checkpoint(dir_path: str,
                    tag: str,
                    models: dict,
                    optimizers: Optional[dict] = None,
                    meta: Optional[dict] = None) -> str:
    """保存模型权重、可选优化器状态及训练元信息。

    产出文件: {dir_path}/ckpt_{tag}.pt

    Args:
        dir_path:   保存目录路径（不存在自动创建）
        tag:        检查点标签字符串，如 "ep_500", "best", "final"
        models:     模型 dict，由 build_models() 返回或部分子集
        optimizers: 优化器状态 dict，如 {"actor": Adam, "critic1": Adam, ...}，可选
        meta:       训练元信息 dict，至少包含:
                        episode          — int, 当前 episode 编号
                        update_step      — int, 累计梯度更新次数
                        obs_dim          — int, 观测维度（=68）
                        act_dim          — int, 动作维度（=2）
                        protocol_version — str, 协议版本号
                    saved_at 字段会自动补入 ISO 8601 UTC 时间戳

    Returns:
        str: 保存文件的完整绝对路径
    """
    os.makedirs(dir_path, exist_ok=True)

    ckpt: dict[str, Any] = {}

    # 保存各网络 state_dict
    for name, net in models.items():
        ckpt[name] = net.state_dict()

    # 保存优化器状态
    if optimizers is not None:
        ckpt["optimizers"] = {
            name: opt.state_dict() for name, opt in optimizers.items()
        }
    else:
        ckpt["optimizers"] = None

    # 元信息（自动补时间戳）
    if meta is None:
        meta = {}
    meta.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    ckpt["meta"] = meta

    filepath = os.path.join(dir_path, f"ckpt_{tag}.pt")
    torch.save(ckpt, filepath)
    return os.path.abspath(filepath)


# ============================================================================
# load_checkpoint — 模型权重加载
# ============================================================================
def load_checkpoint(path: str,
                    models: dict,
                    optimizers: Optional[dict] = None,
                    map_location: str = "cpu") -> dict:
    """从文件加载模型权重（及可选的优化器状态），就地恢复。

    Args:
        path:         检查点文件路径，如 "checkpoints/ckpt_best.pt"
        models:       模型 dict（键名须与保存时一致），权重就地加载
        optimizers:   优化器 dict（可选），状态就地加载
        map_location: torch.load 的设备映射参数，默认 "cpu"

    Returns:
        dict: 训练元信息（episode, update_step, protocol_version, saved_at 等）

    Raises:
        KeyError: 当 models 中某个键在 checkpoint 中不存在时
        FileNotFoundError: 当 path 指定的文件不存在时
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    # 加载网络权重（就地）
    for name in models:
        if name not in ckpt:
            raise KeyError(
                f"模型键 '{name}' 不在 checkpoint 中。"
                f"checkpoint 内可用键: {[k for k in ckpt if k not in ('optimizers', 'meta')]}"
            )
        models[name].load_state_dict(ckpt[name])

    # 加载优化器状态（就地）
    if optimizers is not None:
        opt_ckpt = ckpt.get("optimizers", None)
        if opt_ckpt is not None:
            for name in optimizers:
                if name in opt_ckpt:
                    optimizers[name].load_state_dict(opt_ckpt[name])

    return ckpt.get("meta", {})


# ============================================================================
# 自测程序（§7.4）
# 执行: python model.py --selftest
# 全绿打印 SELFTEST OK
# ============================================================================
def _selftest() -> bool:
    """内置自测：

    1. PolicyNet forward / sample 维度、值域、log_prob 有限性
    2. QNet forward 输出维度
    3. build_models 完整性（5 个网络、target 冻结）
    4. save → load 权重一致性 + meta 回读
    5. 双 Q 网络初始一致性

    Returns:
        bool: 全部通过返回 True
    """
    device = torch.device("cpu")
    B = 4  # batch size

    print("=" * 60)
    print("009 model.py 自测程序")
    print("=" * 60)

    # ---- [1/5] PolicyNet forward / sample 校验 ----
    print("[1/5] PolicyNet forward & sample 校验 …")
    actor = PolicyNet(obs_dim=OBS_DIM, act_dim=ACT_DIM).to(device)
    obs = torch.randn(B, OBS_DIM, device=device)

    # forward
    mean, log_std = actor.forward(obs)
    assert mean.shape == (B, ACT_DIM), \
        f"mean 维度错误: 期望 {(B, ACT_DIM)}, 实际 {mean.shape}"
    assert log_std.shape == (B, ACT_DIM), \
        f"log_std 维度错误: 期望 {(B, ACT_DIM)}, 实际 {log_std.shape}"
    assert torch.all(log_std >= LOG_STD_MIN - 1e-6), \
        f"log_std 低于 LOG_STD_MIN={LOG_STD_MIN}"
    assert torch.all(log_std <= LOG_STD_MAX + 1e-6), \
        f"log_std 超过 LOG_STD_MAX={LOG_STD_MAX}"
    print("    [OK] forward: mean/log_std 维度正确，log_std clamp 有效")

    # sample
    action, log_prob, mean2 = actor.sample(obs)
    assert action.shape == (B, ACT_DIM), \
        f"action 维度错误: 期望 {(B, ACT_DIM)}, 实际 {action.shape}"
    assert log_prob.shape == (B,), \
        f"log_prob 维度错误: 期望 {(B,)}, 实际 {log_prob.shape}"
    assert mean2.shape == (B, ACT_DIM), \
        f"mean 维度错误: 期望 {(B, ACT_DIM)}, 实际 {mean2.shape}"

    # action 值域必须在 (-1, 1) 内
    assert torch.all(action > -1.0) and torch.all(action < 1.0), \
        f"action 越界: min={action.min().item():.6f}, max={action.max().item():.6f}"
    # log_prob 必须全部有限（无 inf / nan）
    assert torch.all(torch.isfinite(log_prob)), \
        "log_prob 包含 inf 或 nan"
    print("    [OK] sample: action 维度/值域/有限性全部通过")

    # ---- [2/5] QNet forward 校验 ----
    print("[2/5] QNet forward 校验 …")
    qnet = QNet(obs_dim=OBS_DIM, act_dim=ACT_DIM).to(device)
    q_val = qnet.forward(obs, action)
    assert q_val.shape == (B, 1), \
        f"Q 值维度错误: 期望 {(B, 1)}, 实际 {q_val.shape}"
    assert torch.all(torch.isfinite(q_val)), "Q 值包含 inf 或 nan"
    print("    [OK] QNet: 输出 [B,1] 正确，数值有限")

    # ---- [3/5] build_models 完整性校验 ----
    print("[3/5] build_models 完整性校验 …")
    models = build_models(device)
    expected_keys = {"actor", "critic1", "critic2", "critic1_target", "critic2_target"}
    assert set(models.keys()) == expected_keys, \
        f"模型键集合不匹配: 期望 {expected_keys}, 实际 {set(models.keys())}"

    # 验证 target 网络参数梯度已冻结
    for name in ["critic1_target", "critic2_target"]:
        for p_name, p in models[name].named_parameters():
            assert not p.requires_grad, \
                f"{name}.{p_name} requires_grad 应为 False，实际为 True"
    print("    [OK] build_models: 5 个网络全部创建，target 梯度已冻结")

    # ---- [4/5] save / load checkpoint 一致性校验 ----
    print("[4/5] save → load checkpoint 一致性校验 …")
    meta_in = {
        "episode": 500,
        "update_step": 12345,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "protocol_version": "1.0",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # 保存
        saved_path = save_checkpoint(tmpdir, "selftest", models,
                                     optimizers=None, meta=meta_in)
        assert os.path.isfile(saved_path), f"checkpoint 文件未创建: {saved_path}"
        print(f"    [OK] checkpoint 已保存: {saved_path}")

        # 重建新网络并加载
        models2 = build_models(device)
        meta_out = load_checkpoint(saved_path, models2,
                                   optimizers=None, map_location="cpu")

        # 逐参数比对权重一致性
        for name in expected_keys:
            sd1 = models[name].state_dict()
            sd2 = models2[name].state_dict()
            assert sd1.keys() == sd2.keys(), \
                f"{name} state_dict 键集合不匹配"
            for key in sd1:
                assert torch.allclose(sd1[key], sd2[key], atol=1e-7), \
                    f"{name}.{key} 权重在 save/load 前后不一致"
        print("    [OK] 权重存取一致性通过")

        # 校验 meta 回读正确性
        for k in meta_in:
            assert meta_out[k] == meta_in[k], \
                f"meta['{k}'] 不匹配: 存 {meta_in[k]}, 读 {meta_out[k]}"
        assert "saved_at" in meta_out, "meta 缺少自动补入的 saved_at 时间戳"
        print("    [OK] meta 回读一致，saved_at 已自动补入")

    # ---- [5/5] 双 Q 网络 target 拷贝一致性校验 ----
    print("[5/5] 双 Q 网络 target 拷贝一致性校验 …")
    c1 = models["critic1"](obs, action)
    c2 = models["critic2"](obs, action)
    t1 = models["critic1_target"](obs, action)
    t2 = models["critic2_target"](obs, action)
    # critic1 与 critic2 应有独立初始化 → 输出不应相同
    assert not torch.allclose(c1, c2, atol=1e-6), \
        "critic1 与 critic2 初始输出相同（应独立随机初始化）"
    # target 是 critic 的精确拷贝 → 输出应完全一致
    assert torch.allclose(c1, t1, atol=1e-7), \
        "critic1_target 与 critic1 初始输出不一致（应为精确拷贝）"
    assert torch.allclose(c2, t2, atol=1e-7), \
        "critic2_target 与 critic2 初始输出不一致（应为精确拷贝）"
    print("    [OK] critic 独立初始化 + target 精确拷贝通过")

    # ---- 完成 ----
    print("=" * 60)
    print("SELFTEST OK — 全部校验通过")
    print("=" * 60)
    return True


# ============================================================================
# 模块入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="009 模型侧 — VOA 网络定义与自测"
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="执行内置自测，校验张量维度、数值范围、权重存取正确性"
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    else:
        print("009 模型侧模块 (model.py)")
        print("用法: python model.py --selftest")
        print("      运行自测程序，校验模型实现正确性。")
