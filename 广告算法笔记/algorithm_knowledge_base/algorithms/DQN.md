# DQN（Deep Q-Network）学习文档

## 1. 算法基础认知

DQN 通过神经网络拟合 Q 函数，能够处理高维状态空间。适用于离散动作空间（如探索档位选择）。

在广告系统中，DQN 被用于：
- 自动出价策略
- 冷启动探索档位选择
- 离散动作空间的竞价决策

## 2. 核心原理

### Q-Learning 基础

Q 函数定义：

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=t}^{T} r_k \mid s_t = s, a_t = a \right]
$$

贝尔曼方程：

$$
Q^*(s, a) = \mathbb{E}_{s'} \left[ r + \max_{a'} Q^*(s', a') \mid s, a \right]
$$

最优策略：$\pi^*(s) = \arg\max_a Q^*(s, a)$

### DQN 关键技术

- **经验回放（Experience Replay）**：存储 (s, a, r, s') 转移样本，随机采样训练
- **目标网络（Target Network）**：定期更新目标 Q 网络，稳定训练
- **ε-greedy 探索**：以概率 ε 随机选择动作

## 3. 在广告出价中的建模

| 组件 | 符号 | 描述 | 典型示例 |
|------|------|------|---------|
| 状态 | s_t | 描述系统当前状况 | 预算消耗率、时间进度、成本表现 |
| 动作 | a_t | 智能体可执行的操作 | 出价调整因子（乘数） |
| 奖励 | r_t | 动作后的即时反馈 | 转化价值 - 成本违反惩罚 |
| 策略 | π(s) | 从状态到动作的映射 | 贪婪策略：选择最大化 Q 值的动作 |

### 动作空间

离散动作空间：[0.5, 0.8, 1.0, 1.2, 1.5]，分别代表大幅降价、小幅降价、维持、小幅提价、大幅提价。

### 奖励函数

$$
r_t = \text{ScalingFactor} \times \text{Value}_t - \text{Penalty} \times \text{CostViolation}_t
$$

## 4. 代码实现（广告出价场景）

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=0.3, buffer_size=10000, batch_size=64):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state))
            return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
```

## 5. 学习总结

DQN 适用于离散动作空间的广告出价场景。阿里在智能出价中有大量实践。对于连续动作空间（如出价调整幅度），通常使用 DDPG。
