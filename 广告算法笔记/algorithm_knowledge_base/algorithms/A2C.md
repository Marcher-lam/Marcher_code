# A2C（Advantage Actor-Critic）学习文档

## 1. 算法基础认知

A2C 是策略梯度方法，与 PPO 类似直接优化策略的期望回报。它同时学习策略（Actor）和价值函数（Critic），使用优势函数来降低方差。

## 2. 核心原理

### 优势函数

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t) = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### 策略梯度

$$
\nabla_\theta J(\theta) = \mathbb{E}_t \left[ A(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$

### 损失函数

$$
L = L_{policy} + c_1 L_{value} - c_2 H(\pi)
$$

其中 H(π) 是策略熵，用于鼓励探索。

## 3. 在广告中的应用

- 广告冷启动：PPO / A2C 直接优化探索策略的期望回报
- 多目标调控：可使用 A2C 训练策略网络

## 4. 代码实现

```python
import torch
import torch.nn as nn

class A2CNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)
```

## 5. 学习总结

A2C 是 PPO 的前身，结构更简单但稳定性稍差。在广告系统中，PPO 通常是首选，A2C 作为轻量级替代。
