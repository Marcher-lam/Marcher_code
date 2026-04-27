# 策略梯度与 REINFORCE 学习文档

> 直接优化策略——用梯度上升让好动作更可能发生。

> 来源线索：本节内容根据原书中关于"策略梯度"的相关章节（第13章13.5.3节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** 策略梯度方法直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升最大化期望累积奖励，REINFORCE 是最基础的策略梯度算法。

**直觉类比：** 价值方法（Q-Learning）像"学会评价每一步"，策略梯度像"直接学会怎么走"。你不需要知道每步的具体价值，只需要知道"这条路走得好，以后多走这条路；那条路走得差，以后少走"。

**历史背景：** REINFORCE 由 Williams 于 1992 年提出。策略梯度定理由 Sutton 等人于 1999 年证明。后续发展为 Actor-Critic、PPO 等现代算法。

**算法定位：** 强化学习、策略优化、on-policy。

**前置知识：** MDP、梯度下降、期望、策略。

---

## 2-3. 核心原理与数学公式

### 策略梯度定理

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\right]$$

### REINFORCE 更新

用回合回报 $G_t$ 替代 $Q$：

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

### 带基线的 REINFORCE

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))$$

基线 $b(s_t)$（通常用 $V(s_t)$）可以减小方差而不引入偏差。

---

## 4-8. 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class REINFORCE(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.net(x)

    def select_action(self, state):
        probs = self.forward(torch.FloatTensor(state))
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rewards, log_probs, gamma=0.99):
        # 计算折扣回报
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -sum(lp * G for lp, G in zip(log_probs, returns))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

agent = REINFORCE(state_dim=4, action_dim=2)
state = np.random.randn(4)
action, log_prob = agent.select_action(state)
print(f"状态: {state}, 动作: {action}, log_prob: {log_prob.item():.4f}")
print(f"参数量: {sum(p.numel() for p in agent.parameters()):,}")
```

---

## 9-14. 练习与路径

**题1：** 策略梯度方法与值函数方法的核心区别？

**参考答案：** 值函数方法（Q-Learning）先学值函数再间接推导策略，策略梯度直接参数化策略并优化。策略梯度的优势：(1) 天然处理连续动作空间；(2) 能学到随机策略；(3) 不需要 argmax 操作。缺点：方差大、样本效率低。

### 学习路径
- 前置：MDP、Q-Learning
- 进阶：Actor-Critic、PPO、SAC
- 推荐：Sutton & Barto, "Reinforcement Learning: An Introduction" 第13章
