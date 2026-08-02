# -*- coding: utf-8 -*-
"""
model_stub.py — 009 model.py 的接口兼容桩（开发期临时文件）

009 侧 model.py 未交付前，003 用它把 SAC 主循环跑通（api.md §8.2 并行开发流程）。
接口严格对齐 api_doc.md §7（类签名、函数签名、checkpoint 格式、--selftest 均一致），
009 交付 model.py 后，003 通过 models.py 自动优先加载外部 model.py，本文件可删除。

自测：python tools/model_stub.py --selftest
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 常量（api.md §7.1） =================
OBS_DIM: int = 68
ACT_DIM: int = 2
LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0
_SELFTEST = "selftest"  # 保持与 009 实现相同的 flag 语义


def _init_log_std_head(m: nn.Module) -> None:
    """log_std_head.bias 用 Uniform(-1, 0) 初始化：初始探索偏保守（σ ≈ 0.37~1.0）。"""
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.bias, -1.0, 0.0)


class PolicyNet(nn.Module):
    """MLP 共享躯干 → 双头输出 (mean_head, log_std_head)（api_doc.md §2）。"""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: tuple = (256, 256)) -> None:
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, act_dim)
        self.log_std_head = nn.Linear(in_dim, act_dim)
        self.log_std_head.apply(_init_log_std_head)

    def forward(self, obs):
        """→ (mean [B,act_dim], log_std [B,act_dim])，log_std 已 clamp。"""
        x = self.trunk(obs)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        """重参数化采样 + tanh 压缩 → (action, log_prob, mean)。"""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        a_raw = mean + std * eps
        # 原始高斯对数概率（按维度求和）
        log_prob_raw = (-0.5 * (((a_raw - mean) / (std + 1e-8)) ** 2).sum(dim=-1)
                        - log_std.sum(dim=-1)
                        - 0.5 * ACT_DIM * torch.log(torch.tensor(2.0 * torch.pi,
                                                                 device=obs.device)))
        action = torch.tanh(a_raw)
        # tanh 修正：log(1 − tanh²(a_raw)) 的数值稳定形式
        log_prob = log_prob_raw - 2.0 * (torch.log(torch.tensor(2.0, device=obs.device))
                                         - a_raw - F.softplus(-2.0 * a_raw)).sum(dim=-1)
        return action, log_prob, mean


class QNet(nn.Module):
    """obs⊕act 拼接 → MLP → 单一 Q 值（api_doc.md §3）。"""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: tuple = (256, 256)) -> None:
        super().__init__()
        layers = []
        in_dim = obs_dim + act_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        """→ Q [B, 1]。"""
        return self.net(torch.cat([obs, act], dim=-1))


def build_models(device) -> dict:
    """返回 {"actor", "critic1", "critic2", "critic1_target", "critic2_target"}。
    target 与对应 critic 同权重初始化并 requires_grad_(False)。"""
    actor = PolicyNet().to(device)
    critic1 = QNet().to(device)
    critic2 = QNet().to(device)
    critic1_target = QNet().to(device)
    critic2_target = QNet().to(device)

    for t, c in ((critic1_target, critic1), (critic2_target, critic2)):
        t.load_state_dict(c.state_dict())
        for p in t.parameters():
            p.requires_grad_(False)

    return {"actor": actor, "critic1": critic1, "critic2": critic2,
            "critic1_target": critic1_target, "critic2_target": critic2_target}


def save_checkpoint(dir_path: str, tag: str, models: dict,
                    optimizers: Optional[dict] = None,
                    meta: Optional[dict] = None) -> str:
    """保存为 {dir_path}/ckpt_{tag}.pt，返回完整路径（api_doc.md §5）。"""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ckpt_{tag}.pt"

    m = dict(meta or {})
    if "saved_at" not in m:  # 自动补全 ISO 8601 UTC 时间戳
        m["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                      time.gmtime())
    payload = {
        "actor": models["actor"].state_dict(),
        "critic1": models["critic1"].state_dict(),
        "critic2": models["critic2"].state_dict(),
        "critic1_target": models["critic1_target"].state_dict(),
        "critic2_target": models["critic2_target"].state_dict(),
        "optimizers": None if optimizers is None
                      else {k: v.state_dict() for k, v in optimizers.items()},
        "meta": m,
    }
    torch.save(payload, path)
    return str(path)


def load_checkpoint(path: str, models: dict,
                    optimizers: Optional[dict] = None,
                    map_location: str = "cpu") -> dict:
    """加载权重进 models（和可选 optimizers），返回 meta（api_doc.md §6）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    payload = torch.load(p, map_location=map_location)

    missing = [k for k in models if k not in payload]
    if missing:
        raise KeyError(f"models 中的键在 checkpoint 中缺失: {missing}；"
                       f"checkpoint 可用键: {list(payload.keys())}")
    for k in models:
        models[k].load_state_dict(payload[k])
    if optimizers is not None:
        saved_opts = payload.get("optimizers") or {}
        for k in optimizers:
            if k in saved_opts:
                optimizers[k].load_state_dict(saved_opts[k])
    return payload.get("meta", {})


# ================= 自测（api_doc.md §9，与 009 交付标准一致） =================
def _selftest() -> int:
    print("=" * 60)
    print("009 model.py 自测程序（model_stub 兼容桩）")
    print("=" * 60)
    torch.manual_seed(0)
    device = torch.device("cpu")
    B = 4
    obs = torch.randn(B, OBS_DIM, device=device)

    # [1/5] PolicyNet
    print("[1/5] PolicyNet forward & sample 校验 …")
    net = PolicyNet().to(device)
    mean, log_std = net(obs)
    assert mean.shape == (B, ACT_DIM) and log_std.shape == (B, ACT_DIM), "维度错"
    assert bool((log_std >= LOG_STD_MIN).all() and (log_std <= LOG_STD_MAX).all()), "clamp 失效"
    action, log_prob, m2 = net.sample(obs)
    assert action.shape == (B, ACT_DIM) and log_prob.shape == (B,) and m2.shape == (B, ACT_DIM)
    assert bool((action > -1.0).all() and (action < 1.0).all()), "action 越界"
    assert bool(torch.isfinite(log_prob).all()), "log_prob 非有限"
    print("    [OK] forward: mean/log_std 维度正确，log_std clamp 有效")
    print("    [OK] sample: action 维度/值域/有限性全部通过")

    # [2/5] QNet
    print("[2/5] QNet forward 校验 …")
    q = QNet().to(device)
    qv = q(obs, action)
    assert qv.shape == (B, 1) and bool(torch.isfinite(qv).all())
    print("    [OK] QNet: 输出 [B,1] 正确，数值有限")

    # [3/5] build_models
    print("[3/5] build_models 完整性校验 …")
    models = build_models(device)
    assert set(models) == {"actor", "critic1", "critic2",
                           "critic1_target", "critic2_target"}
    assert all(not p.requires_grad for p in
               list(models["critic1_target"].parameters())
               + list(models["critic2_target"].parameters()))
    print("    [OK] build_models: 5 个网络全部创建，target 梯度已冻结")

    # [4/5] save/load
    print("[4/5] save → load checkpoint 一致性校验 …")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        meta_in = {"episode": 7, "update_step": 100, "obs_dim": OBS_DIM,
                   "act_dim": ACT_DIM, "protocol_version": "1.1"}
        p = save_checkpoint(td, "t", models, None, meta_in)
        models2 = build_models(device)
        meta_out = load_checkpoint(p, models2)
        for k in models:
            for pa, pb in zip(models[k].parameters(), models2[k].parameters()):
                assert torch.allclose(pa, pb, atol=1e-7), f"{k} 权重不一致"
        assert meta_out["episode"] == 7 and "saved_at" in meta_out, "meta 回读不一致"
        print(f"    [OK] checkpoint 已保存: {p}")
        print("    [OK] 权重存取一致性通过")
        print("    [OK] meta 回读一致，saved_at 已自动补入")

    # [5/5] 双 Q
    print("[5/5] 双 Q 网络 target 拷贝一致性校验 …")
    o = torch.randn(1, OBS_DIM)
    assert not torch.allclose(models["critic1"](o, torch.randn(1, ACT_DIM)),
                              models["critic2"](o, torch.randn(1, ACT_DIM))), "双Q未独立初始化"
    for t, c in ((models["critic1_target"], models["critic1"]),
                 (models["critic2_target"], models["critic2"])):
        for pa, pb in zip(t.parameters(), c.parameters()):
            assert torch.allclose(pa, pb), "target 非精确拷贝"
    print("    [OK] critic 独立初始化 + target 精确拷贝通过")

    print("=" * 60)
    print("SELFTEST OK — 全部校验通过")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    sys.exit("model_stub.py 无 --selftest 参数时不执行任何操作")
