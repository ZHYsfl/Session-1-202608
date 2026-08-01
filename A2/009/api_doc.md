# 009 模型侧接口文档

> 项目：基于强化学习防碰撞的线控底盘模型设计 (VOA)
> 协议版本：1.0
> 作者：009
> 产出文件：[model.py](model.py)

---

## 目录

1. [全局常量](#1-全局常量)
2. [PolicyNet 策略网络](#2-policynet-策略网络)
3. [QNet 价值网络](#3-qnet-价值网络)
4. [build_models() 统一初始化](#4-build_models-统一初始化)
5. [save_checkpoint() 权重保存](#5-save_checkpoint-权重保存)
6. [load_checkpoint() 权重加载](#6-load_checkpoint-权重加载)
7. [checkpoint 文件结构](#7-checkpoint-文件结构)
8. [003 算法侧调用示例](#8-003-算法侧调用示例)
9. [自测程序](#9-自测程序)

---

## 1. 全局常量

```python
OBS_DIM: int = 68
ACT_DIM: int = 2
LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0
```

| 常量 | 值 | 说明 |
|---|---|---|
| `OBS_DIM` | 68 | 观测向量维度 = 64(lidar) + 2(goal: dist, bearing) + 2(vel: v, w) |
| `ACT_DIM` | 2 | 动作空间维度 = (v, w)，对应线速度和角速度 |
| `LOG_STD_MIN` | -5.0 | log_std 下界，防止方差坍缩为 0（σ_min ≈ 0.0067） |
| `LOG_STD_MAX` | 2.0 | log_std 上界，防止方差爆炸（σ_max ≈ 7.39） |

**兼容性设计**：切换真机硬件时，仅需修改本节常量。例如真机雷达线数从 64 变为 32，只需将 `OBS_DIM` 改为 `32 + 2 + 2 = 36`，网络推理逻辑无需任何改动。

---

## 2. PolicyNet 策略网络

### 类签名

```python
class PolicyNet(nn.Module):
    def __init__(self,
                 obs_dim: int = OBS_DIM,      # 默认 68
                 act_dim: int = ACT_DIM,      # 默认 2
                 hidden: tuple = (256, 256)) -> None
```

**架构**：`MLP 共享躯干 → 双头输出 (mean_head, log_std_head)`

```
obs [B,68] → Linear(68→256) → ReLU → Linear(256→256) → ReLU
                  ├─ mean_head:   Linear(256→2) → mean [B,2]
                  └─ log_std_head: Linear(256→2) → clamp → log_std [B,2]
```

**构造函数参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `obs_dim` | int | 68 | 观测向量维度 |
| `act_dim` | int | 2 | 动作空间维度 |
| `hidden` | tuple | (256, 256) | 隐藏层神经元数量元组，支持任意长度，如 `(128, 128)`、`(512, 256, 128)` |

**初始化细节**：`log_std_head.bias` 使用 `Uniform(-1.0, 0.0)` 初始化，使初始探索偏保守（σ ≈ 0.37~1.0）。

---

### forward()

```python
def forward(self, obs: torch.Tensor)
    -> Tuple[torch.Tensor, torch.Tensor]:
```

给定观测，输出动作高斯分布的参数。

| 项目 | 说明 |
|---|---|
| **输入** | `obs`: float32 张量，shape `[B, obs_dim]`（已由 003 归一化，各分量范围见 api.md §4） |
| **输出** | `mean`: float32 张量，shape `[B, act_dim]`，动作均值 |
| | `log_std`: float32 张量，shape `[B, act_dim]`，对数标准差，已 clamp 到 `[LOG_STD_MIN, LOG_STD_MAX]` |

---

### sample()

```python
def sample(self, obs: torch.Tensor)
    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

重参数化采样 + tanh 压缩，生成 SAC 训练/推理用动作。

**内部流程**：

1. `forward(obs)` → `(mean, log_std)`
2. 重参数化：`a_raw = mean + exp(log_std) ⊙ ε`，`ε ~ N(0, I)`
3. 计算原始高斯对数概率（按维度求和）
4. `action = tanh(a_raw)` → 压缩至 `(-1, 1)²`
5. tanh 对数概率修正：`log_prob = log_prob_raw − Σ log(1 − tanh²(a_raw) + ε)`

| 项目 | 说明 |
|---|---|
| **输入** | `obs`: float32 张量，shape `[B, obs_dim]` |
| **输出** | `action`: float32 张量，shape `[B, act_dim]`，**每个分量 ∈ (-1, 1)**（tanh 压缩，不含端点） |
| | `log_prob`: float32 张量，shape `[B]`，采样动作的对数概率密度（已含 tanh 修正项，每个元素为各维度求和后的标量） |
| | `mean`: float32 张量，shape `[B, act_dim]`，动作均值（**评估时直接取 mean 作为确定性策略**，不采样） |

**关键约定**：

- 训练时用 `action, log_prob, _ = actor.sample(obs)`
- 评估时用 `_, _, mean = actor.sample(obs)` 或直接用 `mean, _ = actor.forward(obs)` 取确定性策略
- action 天然落在 (-1,1)，003 负责放大到物理单位：`v = a[0] * v_max`，`w = a[1] * w_max`

---

## 3. QNet 价值网络

### 类签名

```python
class QNet(nn.Module):
    def __init__(self,
                 obs_dim: int = OBS_DIM,      # 默认 68
                 act_dim: int = ACT_DIM,      # 默认 2
                 hidden: tuple = (256, 256)) -> None
```

**架构**：`obs⊕act 拼接 → MLP → 单一 Q 值`

```
obs [B,68], act [B,2] → concat [B,70] → Linear(70→256) → ReLU
    → Linear(256→256) → ReLU → Linear(256→1) → Q [B,1]
```

**双 Q 设计**：003 创建两个独立的 `QNet` 实例（critic1, critic2），取 `min(c1, c2)` 作为 target Q 值，缓解高估偏差。

---

### forward()

```python
def forward(self, obs: torch.Tensor, act: torch.Tensor)
    -> torch.Tensor:
```

| 项目 | 说明 |
|---|---|
| **输入** | `obs`: float32 张量，shape `[B, obs_dim]` |
| | `act`: float32 张量，shape `[B, act_dim]` |
| **输出** | float32 张量，shape `[B, 1]`，状态-动作对 (obs, act) 的 Q 值 |

---

## 4. build_models() 统一初始化

```python
def build_models(device) -> dict:
```

| 项目 | 说明 |
|---|---|
| **参数** | `device`: `torch.device` 对象或字符串，如 `torch.device("cuda:0")` 或 `"cpu"` |
| **返回** | `dict`，固定键名如下表 |

**返回字典结构**：

| 键 | 类型 | requires_grad | 说明 |
|---|---|---|---|
| `"actor"` | `PolicyNet` | True | 策略网络，输出动作分布 |
| `"critic1"` | `QNet` | True | 双 Q 网络 1 |
| `"critic2"` | `QNet` | True | 双 Q 网络 2（独立随机初始化） |
| `"critic1_target"` | `QNet` | **False** | critic1 的 target 副本，初始权重精确拷贝 |
| `"critic2_target"` | `QNet` | **False** | critic2 的 target 副本，初始权重精确拷贝 |

**关键行为**：

- critic1 与 critic2 **独立随机初始化**（输出值不同，这是 clipped double-Q 的基础）
- target 网络权重是对应 critic 的**精确拷贝**（`load_state_dict`），`requires_grad_(False)` 冻结
- 003 通过 soft update 更新 target：`θ_target ← τ·θ_critic + (1−τ)·θ_target`（τ=0.005）

---

## 5. save_checkpoint() 权重保存

```python
def save_checkpoint(dir_path: str,
                    tag: str,
                    models: dict,
                    optimizers: Optional[dict] = None,
                    meta: Optional[dict] = None) -> str:
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `dir_path` | str | 保存目录路径（不存在自动创建） |
| `tag` | str | 检查点标签，如 `"ep_500"`, `"best"`, `"final"` |
| `models` | dict | 模型 dict，由 `build_models()` 返回或子集 |
| `optimizers` | dict 或 None | 优化器状态 dict，如 `{"actor": Adam, "critic1": Adam, ...}` |
| `meta` | dict 或 None | 元信息，须含 `episode`, `update_step`, `obs_dim`, `act_dim`, `protocol_version` |

| 返回值 | 类型 | 说明 |
|---|---|---|
| 文件路径 | str | 保存文件的完整绝对路径，如 `checkpoints/ckpt_ep_500.pt` |

**元信息自动补全**：`saved_at` 字段自动填入 ISO 8601 UTC 时间戳（如 `"2026-07-31T12:00:00+00:00"`），调用方无需手动填写。

---

## 6. load_checkpoint() 权重加载

```python
def load_checkpoint(path: str,
                    models: dict,
                    optimizers: Optional[dict] = None,
                    map_location: str = "cpu") -> dict:
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `path` | str | 检查点文件路径 |
| `models` | dict | 模型 dict（键名须与保存时一致），**就地加载** |
| `optimizers` | dict 或 None | 优化器 dict（可选），**就地加载** |
| `map_location` | str | 设备映射参数，默认 `"cpu"` |

| 返回值 | 类型 | 说明 |
|---|---|---|
| meta | dict | 保存时的元信息 dict |

**异常**：

- `FileNotFoundError` — 文件不存在
- `KeyError` — models 中的某个键在 checkpoint 中不存在（附可用键列表）

---

## 7. Checkpoint 文件结构

文件 `ckpt_{tag}.pt` 由 `torch.save` 序列化，是一个 dict：

```python
{
    # 各网络 state_dict（OrderedDict）
    "actor":             OrderedDict,  # PolicyNet 权重
    "critic1":           OrderedDict,  # QNet1 权重
    "critic2":           OrderedDict,  # QNet2 权重
    "critic1_target":    OrderedDict,  # QNet1 target 权重
    "critic2_target":    OrderedDict,  # QNet2 target 权重

    # 优化器状态（可选）
    "optimizers": {
        "actor":    dict,  # Adam optimizer state_dict
        "critic1":  dict,
        "critic2":  dict,
    } | None,

    # 训练元信息
    "meta": {
        "episode":          500,           # int, 当前 episode 编号
        "update_step":      12345,         # int, 累计梯度更新次数
        "obs_dim":          68,            # int
        "act_dim":          2,             # int
        "protocol_version": "1.0",         # str
        "saved_at":         "2026-07-31T12:00:00+00:00",  # ISO 8601 UTC
    },
}
```

**文件大小估算**（默认 (256,256) 隐藏层）：

| 网络 | 参数量 | 权重大小 (float32) |
|---|---|---|
| PolicyNet | ≈ 84K | ≈ 0.34 MB |
| QNet ×4 | ≈ 73K ×4 | ≈ 1.17 MB |
| **总计** | ≈ 376K | ≈ **1.5 MB** |

---

## 8. 003 算法侧调用示例

### 8.1 初始化

```python
import torch
from model import build_models, OBS_DIM, ACT_DIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = build_models(device)

actor = models["actor"]
critic1 = models["critic1"]
critic2 = models["critic2"]
critic1_target = models["critic1_target"]
critic2_target = models["critic2_target"]
```

### 8.2 训练时采样动作

```python
# obs_vec: 003 按 api.md §4 归一化后的 [B, 68] 张量
obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 68]

with torch.no_grad():
    action, log_prob, _ = actor.sample(obs_tensor)
# action[0] ∈ (-1, 1)² → 003 放大: v = action[0,0]*v_max, w = action[0,1]*w_max
```

### 8.3 评估时确定性推理

```python
with torch.no_grad():
    # 方式 1: 直接用 forward 取 mean
    mean, _ = actor.forward(obs_tensor)
    # 方式 2: 用 sample 取 mean（forward 内部也用一样的 mean）
    _, _, mean = actor.sample(obs_tensor)

action_det = mean  # 确定性动作，不采样
# 003 放大: v = action_det[0,0]*v_max, w = action_det[0,1]*v_max
```

### 8.4 SAC Q 值计算（003 实现损失时）

```python
# 从 buffer 采样的 batch
obs_batch = ...    # [256, 68]
act_batch = ...    # [256, 2]

q1 = critic1(obs_batch, act_batch)  # [256, 1]
q2 = critic2(obs_batch, act_batch)  # [256, 1]

# clipped double-Q: 取较小值作为 target
with torch.no_grad():
    next_action, next_log_prob, _ = actor.sample(next_obs_batch)
    q1_next = critic1_target(next_obs_batch, next_action)
    q2_next = critic2_target(next_obs_batch, next_action)
    q_next_min = torch.min(q1_next, q2_next)
    q_target = reward_batch + gamma * (1 - done_mask) * (q_next_min - alpha * next_log_prob.unsqueeze(-1))
```

### 8.5 Soft update（003 训练循环中）

```python
tau = 0.005
with torch.no_grad():
    for target_param, param in zip(critic1_target.parameters(), critic1.parameters()):
        target_param.copy_(tau * param + (1 - tau) * target_param)
    for target_param, param in zip(critic2_target.parameters(), critic2.parameters()):
        target_param.copy_(tau * param + (1 - tau) * target_param)
```

### 8.6 保存与加载

```python
from model import save_checkpoint, load_checkpoint

# 保存
meta = {
    "episode": episode,
    "update_step": total_updates,
    "obs_dim": OBS_DIM,
    "act_dim": ACT_DIM,
    "protocol_version": "1.0",
}
optimizers = {"actor": opt_actor, "critic1": opt_c1, "critic2": opt_c2}
save_path = save_checkpoint("checkpoints", "ep_500", models, optimizers, meta)

# 加载（恢复训练）
meta = load_checkpoint("checkpoints/ckpt_best.pt", models, optimizers, map_location="cpu")
start_episode = meta["episode"]
```

---

## 9. 自测程序

```bash
python model.py --selftest
```

**测试覆盖**：

| 测试项 | 校验内容 |
|---|---|
| [1/5] PolicyNet | forward 输出维度；log_std clamp 有效；sample 输出维度、action ∈ (-1,1)、log_prob 有限性 |
| [2/5] QNet | forward 输出 shape [B,1]、数值有限 |
| [3/5] build_models | 5 个网络全部创建；target 参数 `requires_grad=False` |
| [4/5] save/load | 权重存取前后逐参数比对（`torch.allclose(atol=1e-7)`）；meta 回读一致性；saved_at 自动补入 |
| [5/5] 双 Q 一致性 | critic1 ≠ critic2（独立初始化）；target 精确拷贝自对应 critic |

**预期输出**：

```
============================================================
009 model.py 自测程序
============================================================
[1/5] PolicyNet forward & sample 校验 …
    [OK] forward: mean/log_std 维度正确，log_std clamp 有效
    [OK] sample: action 维度/值域/有限性全部通过
[2/5] QNet forward 校验 …
    [OK] QNet: 输出 [B,1] 正确，数值有限
[3/5] build_models 完整性校验 …
    [OK] build_models: 5 个网络全部创建，target 梯度已冻结
[4/5] save → load checkpoint 一致性校验 …
    [OK] checkpoint 已保存: ...
    [OK] 权重存取一致性通过
    [OK] meta 回读一致，saved_at 已自动补入
[5/5] 双 Q 网络 target 拷贝一致性校验 …
    [OK] critic 独立初始化 + target 精确拷贝通过
============================================================
SELFTEST OK — 全部校验通过
============================================================
```
