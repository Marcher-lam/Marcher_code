# Actor-Critic 学习文档

> Actor 学策略，Critic 学价值——策略梯度与值函数的完美结合。

> 来源线索：本节内容根据原书中关于"Actor-Critic"的相关章节（第13章13.5.3节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** Actor-Critic 同时训练策略网络（Actor）和价值网络（Critic），Critic 用 TD 误差为 Actor 提供低方差的梯度信号。

**直觉类比：** 策略梯度（REINFORCE）像"只有运动员"——只能通过最终比分判断表现好坏。Actor-Critic 加了一个"教练"（Critic）——每一步都给出即时反馈"这一步做得怎么样"。教练的评估比最终比分更有指导性，大大加快了学习速度。

**历史背景：** Actor-Critic 框架由 Barto 等人于 1983 年提出。A2C/A3C（Mnih et al., 2016）使其在大规模问题上实用化。

**算法定位：** 强化学习、策略优化+值函数估计。

**前置知识：** 策略梯度、TD 学习、价值函数。

---

## 2-3. 核心原理与数学公式

### 两个网络

- **Actor $\pi_\theta(a|s)$**：策略网络，输出动作概率
- **Critic $V_\phi(s)$**：价值网络，输出状态价值估计

### 更新规则

**Critic 更新**（TD 误差）：

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

$$\mathcal{L}_{Critic} = \delta_t^2$$

**Actor 更新**（策略梯度，用 TD 误差替代回报）：

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$$

### 优势函数

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \approx r_t + \gamma V(s_{t+1}) - V(s_t) = \delta_t$$

---

## 4-8. 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(hidden, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

class A2CAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=1e-3, gamma=0.99):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def step(self, state):
        probs, value = self.model(torch.FloatTensor(state).unsqueeze(0))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(), dist.entropy()

    def update(self, rewards, log_probs, values, entropies):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        advantages = returns - torch.stack(values)
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -torch.stack(entropies).mean() * 0.01

        loss = actor_loss + 0.5 * critic_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

agent = A2CAgent(state_dim=4, action_dim=2)
state = np.random.randn(4)
action, lp, val, ent = agent.step(state)
print(f"动作:{action} 值估计:{val.item():.3f} 熵:{ent.item():.3f}")
```

---

## 9-14. 练习与路径

**题1：** Actor-Critic 相比 REINFORCE 的优势？

**参考答案：** REINFORCE 用完整回合回报 $G_t$ 作为梯度权重，方差大。Actor-Critic 用 Critic 的 TD 误差 $\delta_t$ 替代，方差显著降低，同时保持无偏（或低偏差）。此外 Actor-Critic 可以在线学习（无需等回合结束）。

### 学习路径
- 前置：策略梯度、TD 学习
- 进阶：A3C（异步）、PPO、SAC
- 推荐：Sutton & Barto, "Reinforcement Learning" 第13章
