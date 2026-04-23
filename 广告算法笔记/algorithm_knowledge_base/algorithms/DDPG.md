# DDPG（Deep Deterministic Policy Gradient）学习文档

## 1. 算法基础认知

DDPG 适用于连续动作空间（如出价调整幅度）。它结合了 DQN 的经验回放和目标网络技术，以及 Actor-Critic 架构，能够在连续动作空间中学习确定性策略。

## 2. 核心原理

DDPG 使用两个网络：
- **Actor 网络**：输出确定性动作 a = μ(s|θ^μ)
- **Critic 网络**：评估动作价值 Q(s, a|θ^Q)

### 更新规则

Critic 网络损失：
$$
L = \frac{1}{N} \sum_i (y_i - Q(s_i, a_i|\theta^Q))^2
$$

其中目标值：
$$
y_i = r_i + \gamma Q'(s_{i+1}, \mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})
$$

Actor 网络策略梯度：
$$
\nabla_{\theta^\mu} J \approx \frac{1}{N} \sum_i \nabla_a Q(s, a|\theta^Q)|_{s=s_i, a=\mu(s_i)} \nabla_{\theta^\mu} \mu(s|\theta^\mu)|_{s_i}
$$

## 3. 在广告出价中的应用

- 连续动作空间：出价调整幅度 λ ∈ [0.3, 3.0]
- 阿里在智能出价中有大量实践
- 动作 = 出价调整因子，Bid = BaseBid × λ

## 4. 代码实现（简化版）

```python
import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, max_action=3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Sigmoid()
        )
        self.max_action = max_action
        self.min_action = 0.3

    def forward(self, x):
        out = self.net(x)
        return self.min_action + out * (self.max_action - self.min_action)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))
```

## 5. 学习总结

DDPG 是广告出价中处理连续动作空间的主流 RL 算法。与 DQN（离散）互补。后续改进算法包括 TD3 和 SAC。
