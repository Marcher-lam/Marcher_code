# Q-learning 学习文档

## 1. 算法基础认知

Q-learning 是一种无模型（model-free）的离线策略（off-policy）强化学习算法。适用于离散动作空间，通过学习 Q 函数来找到最优策略。

## 2. 核心原理

### Q 值更新规则

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

### 最优策略

$$
\pi^*(s) = \arg\max_a Q^*(s, a)
$$

## 3. 在广告出价中的应用

Q-learning / DQN 适用于离散动作空间。DQN 通过神经网络拟合 Q 函数，能够处理高维状态空间。

### 广告出价的 Q-learning 实现

在广告系统中，状态包括：
- budget_ratio: 剩余预算 / 总预算
- time_ratio: 已过去时间 / 总时间
- current_cpa: 最近平均转化成本
- cost_violation: 成本违反程度
- conversion_rate: 最近转化率

动作空间（离散）：[0.5, 0.8, 1.0, 1.2, 1.5]

奖励函数：
$$
reward = conversions - \lambda \times cost\_violation
$$

## 4. 代码实现（广告出价场景简化版）

```python
import numpy as np
from collections import defaultdict

class QLearningBiddingAgent:
    def __init__(self, n_actions=5, learning_rate=0.1, gamma=0.99, epsilon=0.3):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def discretize_state(self, state):
        return tuple((state * 10).astype(int))

    def select_action(self, state):
        s = self.discretize_state(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[s])

    def update(self, state, action, reward, next_state, done):
        s = self.discretize_state(state)
        s_next = self.discretize_state(next_state)
        best_next = np.max(self.q_table[s_next])
        target = reward + (1 - done) * self.gamma * best_next
        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])
```

## 5. 学习总结

Q-learning 是强化学习出价的基础算法，从 Q-learning 到 DQN（深度网络拟合 Q 函数），再到 PPO/SAC（策略梯度），广告出价 RL 不断演进。
