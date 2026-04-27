# TRPO 学习文档

> 信任区域策略优化——用 KL 约束保证策略更新安全。

> 来源线索：本节内容根据原书中关于"TRPO"的相关章节（第13章13.5.3节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** TRPO 通过在每次策略更新中约束新旧策略的 KL 散度不超过阈值，保证策略单调递增改进，是 PPO 的理论前身。

**直觉类比：** 普通策略梯度像"没有刹车的车"——可能一步冲太远导致性能崩溃。TRPO 像"装了限速器的车"——每次更新被限制在一个"信任区域"内，确保不会改得太猛。PPO 则是 TRPO 的简化版——用裁剪替代 KL 约束。

**历史背景：** TRPO 由 Schulman 等人于 2015 年提出（论文 "Trust Region Policy Optimization"），提供了策略单调改进的理论保证。PPO（2017）是其工程化简化。

**算法定位：** 策略梯度方法、on-policy、信任区域方法。

**前置知识：** 策略梯度、KL 散度、Actor-Critic、共轭梯度法。

---

## 2-3. 核心原理与数学公式

### 替代目标

$$\max_\theta \mathbb{E}_{s \sim \pi_{old}, a \sim \pi_{old}}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A_{\pi_{old}}(s,a)\right]$$

### KL 约束

$$\bar{D}_{KL}(\pi_{old} \| \pi_\theta) \leq \delta$$

即新旧策略的平均 KL 散度不超过 $\delta$（通常 0.01）。

### 近似求解

用泰勒展开近似目标（二阶）和约束（一阶）：

$$\max_\theta g^T(\theta - \theta_{old}) \quad \text{s.t.} \quad \frac{1}{2}(\theta - \theta_{old})^T H (\theta - \theta_{old}) \leq \delta$$

其中 $g$ 是策略梯度，$H$ 是 Fisher 信息矩阵。解析解：

$$\theta = \theta_{old} + \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g$$

实际用**共轭梯度法**近似 $H^{-1}g$，避免直接计算 $H$。

---

## 4-8. 代码实现

```python
import torch
import torch.nn as nn
import numpy as np

class TRPOAgent:
    def __init__(self, state_dim=4, action_dim=2, hidden=64,
                 gamma=0.99, kl_target=0.01, damping=0.1):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.gamma = gamma
        self.kl_target = kl_target
        self.damping = damping

    def get_action(self, state):
        probs = self.actor(torch.FloatTensor(state).unsqueeze(0))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_kl(self, old_probs, new_probs):
        return (old_probs * (torch.log(old_probs + 1e-8) - torch.log(new_probs + 1e-8))).sum(-1).mean()

    def flat_params(self):
        return torch.cat([p.data.view(-1) for p in self.actor.parameters()])

    def update_critic(self, states, returns, lr=1e-3):
        values = self.critic(torch.FloatTensor(states)).squeeze()
        loss = nn.MSELoss()(values, torch.FloatTensor(returns))
        return loss.item()

# 测试
agent = TRPOAgent(state_dim=4, action_dim=2)
state = np.random.randn(4)
action, log_prob = agent.get_action(state)
print(f"动作: {action}, log_prob: {log_prob.item():.4f}")
print(f"KL target: {agent.kl_target}")
print(f"策略参数量: {sum(p.numel() for p in agent.actor.parameters()):,}")
```

---

## 9-14. 练习与路径

**题1：** TRPO 与 PPO 的核心区别？

**参考答案：** TRPO 用硬 KL 约束（二阶优化，共轭梯度法），PPO 用裁剪目标（一阶优化）。TRPO 理论更严谨但实现复杂，PPO 实现简单且实践中效果相当。PPO 因此成为更流行的选择。

### 学习路径
- 前置：策略梯度、Actor-Critic
- 平行：PPO（TRPO 的简化版）
- 进阶：自然策略梯度、ACKTR
- 推荐：Schulman et al., "Trust Region Policy Optimization" (2015)
