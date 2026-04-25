# SAC（Soft Actor-Critic）学习文档

## 1. 算法基础认知

SAC 由 Tuomas Haarnoja 等于 2018 年提出，是基于最大熵框架的 off-policy Actor-Critic 算法。它在最大化期望回报的同时最大化策略熵，兼顾探索与利用，是当前连续控制最强的 RL 算法之一。在广告系统中用于连续出价调整和多目标权重调权。

## 2. 核心原理

### 最大熵目标

$$\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{T}\gamma^t\left(r_t + \alpha H(\pi(\cdot|s_t))\right)\right]$$

其中 α 是温度参数，H 是策略熵。SAC 同时学习：
- **两个 Q 网络**（减少过估计）：Q₁, Q₂
- **SAC v2 取消独立 V 网络，直接用双 Q 目标网络取 min**
- **一个策略网络** π(a|s)
- **自动温度调节** α

### 关键技术

1. **双 Q 网络**：取 min(Q₁, Q₂) 作为目标，减少过估计
2. **重参数化技巧**：a = tanh(μ + σ·ε)，ε ~ N(0,1)，使策略梯度可微
3. **自动温度调节**：通过约束优化自动调整 α

## 3. 数学公式与推导

Soft Bellman 方程：

$$Q(s,a) = r + \gamma\mathbb{E}_{s'}\left[V(s')\right]$$

$$V(s) = \mathbb{E}_{a\sim\pi}\left[Q(s,a) - \alpha\log\pi(a|s)\right]$$

Q 网络损失：

$$L_Q(\theta_i) = \mathbb{E}_{(s,a,r,s')}\left[\left(Q_i(s,a) - y\right)^2\right]$$

$$y = r + \gamma\left(\min_{j=1,2}Q_j(s',a') - \alpha\log\pi(a'|s')\right)$$

策略损失：

$$L_\pi(\phi) = \mathbb{E}_{s}\left[\mathbb{E}_{a\sim\pi}\left[\alpha\log\pi(a|s) - Q(s,a)\right]\right]$$

温度损失：

$$L(\alpha) = \mathbb{E}_{a\sim\pi}\left[-\alpha\log\pi(a|s) - \alpha\bar{H}\right]$$

## 4. 训练过程讲解

1. 初始化 Q₁, Q₂, π 及其目标网络 Q̄₁, Q̄₂（SAC v2，无 V 网络）
2. 从回放池采样 (s,a,r,s')
3. 用重参数化技巧从 π(·|s') 采样 a'
4. 计算软目标值 y = r + γ(min(Q̄₁,Q̄₂)(s',a') - α log π(a'|s'))
5. 更新 Q₁, Q₂：最小化 (Q_i(s,a) - y)²
6. 更新 π：最小化 E[α log π(a|s) - min(Q₁,Q₂)(s,a)]
7. 更新 α：自动调节温度
8. 软更新 Q̄₁, Q̄₂

## 5. 应用场景

- 广告连续出价调整
- 多目标权重动态调权
- 机器人灵巧操作
- 自动驾驶控制
- 能源管理系统

## 6. 优缺点分析

**优点**：
- 最大熵框架天然鼓励探索
- 双 Q 网络减少过估计
- 自动温度调节免去手动调参
- 样本效率高（off-policy）
- 训练稳定性强

**缺点**：
- 实现复杂度高
- 仅适用于连续动作空间（标准版）
- 计算开销较大（多个网络）
- 调参空间比 PPO 大

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import copy

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, deterministic=False):
        feat = self.net(x)
        mu = self.mu_head(feat)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mu)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        a = mu + std * eps
        log_prob = -0.5 * (((a - mu) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob -= torch.log(1 - torch.tanh(a) ** 2 + 1e-6)
        return torch.tanh(a), log_prob.sum(-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))
    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.q1 = Critic(state_dim, action_dim)
        self.q2 = Critic(state_dim, action_dim)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=3e-4)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.buffer = deque(maxlen=100000)

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s, deterministic)[0]
        return a.squeeze(0).numpy()

    def update(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_n, d = map(lambda x: torch.FloatTensor(np.array(x)), zip(*batch))
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            a_n, log_p_n = self.actor(s_n)
            q_target = torch.min(self.q1_target(s_n, a_n), self.q2_target(s_n, a_n))
            y = r + self.gamma * (q_target - alpha * log_p_n) * (1 - d)
        q1_loss = nn.MSELoss()(self.q1(s, a), y)
        q2_loss = nn.MSELoss()(self.q2(s, a), y)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()
        a_new, log_p = self.actor(s)
        q_new = torch.min(self.q1(s, a_new), self.q2(s, a_new))
        actor_loss = (alpha * log_p - q_new).mean()
        self.a_opt.zero_grad(); actor_loss.backward(); self.a_opt.step()
        alpha_loss = -(self.log_alpha * (log_p + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        for t, s_ in zip(self.q1_target.parameters(), self.q1.parameters()):
            t.data.copy_(self.tau * s_.data + (1 - self.tau) * t.data)
        for t, s_ in zip(self.q2_target.parameters(), self.q2.parameters()):
            t.data.copy_(self.tau * s_.data + (1 - self.tau) * t.data)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class SimpleSAC:
    def __init__(self, state_dim, action_dim, lr=0.0003):
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = lr
        self.log_alpha = 0.0
        self.target_entropy = -action_dim
        self.q1_W = np.random.randn(state_dim + action_dim, 32) * 0.1
        self.q1_b = np.zeros(32)
        self.q1_W2 = np.random.randn(32, 1) * 0.1
        self.q1_b2 = np.zeros(1)
        self.q1t_W = self.q1_W.copy()
        self.q1t_b = self.q1_b.copy()
        self.q1t_W2 = self.q1_W2.copy()
        self.q1t_b2 = self.q1_b2.copy()

    def predict_q(self, s, a, W, b, W2, b2):
        x = np.concatenate([s, a])
        h = np.maximum(0, x @ W + b)
        return h, h @ W2 + b2

    def soft_update(self):
        self.q1t_W = self.tau * self.q1_W + (1 - self.tau) * self.q1t_W
        self.q1t_b = self.tau * self.q1_b + (1 - self.tau) * self.q1t_b
        self.q1t_W2 = self.tau * self.q1_W2 + (1 - self.tau) * self.q1t_W2
        self.q1t_b2 = self.tau * self.q1_b2 + (1 - self.tau) * self.q1t_b2

    def update_alpha(self, log_prob):
        alpha = np.exp(self.log_alpha)
        alpha_grad = -(log_prob + self.target_entropy)
        self.log_alpha -= self.lr * alpha_grad
        return alpha
```

## 9. 可视化与结果理解

- **Q 值曲线**：双 Q 值应接近且逐步增长
- **策略熵曲线**：自动温度使熵维持在目标水平
- **温度 α 变化**：应逐步调整到合适值
- **Actor 损失**：α·log π - Q，应为正逐步减小

## 10. 模型评估

- **平均回合奖励**：核心指标
- **策略熵**：是否维持在合理水平
- **Q 值准确性**：Q 估计与实际回报的误差
- **温度 α 收敛性**：是否自动调节到稳定值

## 11. 常见问题与易错点

- **忘记 clamp log_std**：标准差过大或过小导致数值问题
- **重参数化技巧遗漏**：直接采样不可微，无法反向传播
- **目标熵设置不当**：通常取 -dim(action)
- **双 Q 网络忘记取 min**：单独使用一个 Q 网络会导致过估计
- **tanh 的 log_prob 修正遗漏**：需减去 log(1 - tanh²(a))

## 12. 学习总结

SAC 通过最大熵框架实现了探索与利用的自动平衡，双 Q 网络和自动温度调节使其成为当前最稳定的连续控制 RL 算法。在广告系统中适用于连续出价和多目标调权场景。相比 DDPG 更稳定，相比 TD3 探索更充分。

## 13. 练习题与思考题（含答案）

**Q1**：SAC 中最大熵项的作用是什么？

A1：最大熵项 α·H(π) 鼓励策略在最大化回报的同时保持随机性，防止过早收敛到次优策略，增强鲁棒性和探索能力。

**Q2**：为什么 SAC 需要重参数化技巧？

A2：直接从策略采样 a ~ π(·|s) 不可微（采样操作不可导）。重参数化 a = tanh(μ + σ·ε) 将随机性转移到 ε，使梯度可以通过 μ 和 σ 反向传播。

**Q3**：SAC 与 TD3 的区别是什么？

A3：SAC 使用最大熵框架和随机策略，TD3 使用确定性策略。SAC 通过自动温度调节鼓励探索，TD3 通过目标策略平滑减少过拟合。SAC 通常在探索密集型任务上更优。

## 14. 学习路径建议

前置知识：DDPG → Actor-Critic → 熵正则化
进阶方向：SAC → SAC+AE（辅助任务）→ 离散 SAC → 多智能体 SAC
