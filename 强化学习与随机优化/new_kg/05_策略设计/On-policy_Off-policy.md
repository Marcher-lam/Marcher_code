# On-policy与Off-policy学习 学习文档

> 在策略学习用自己经验更新，离策略学习从他人经验学习。

> 来源线索：本节内容根据原书中关于"On-policy vs Off-policy"的相关章节(Ch 17.5)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：On-policy评估和改进同一策略；Off-policy评估目标策略$\pi$但用行为策略$b$的数据。

**On-policy（在策略）**：
- 策略：执行$\pi$，更新$\pi$
- 代表：SARSA、REINFORCE
- 特点：安全、保守，探索影响学习

**Off-policy（离策略）**：
- 目标策略$\pi$（要学习的），行为策略$b$（收集数据的）
- 代表：Q-Learning、DQN
- 特点：高效利用数据，但可能不稳定

**Q-Learning的离策略性**：

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]$$

更新用$\max$（目标策略），但数据来自$\epsilon$-greedy（行为策略）。

**重要性采样比**：

$$\rho_{t:T} = \prod_{k=t}^T \frac{\pi(a_k|s_k)}{b(a_k|s_k)}$$

用于修正分布偏差。

## 4-8. 核心实现

```python
"""On-policy vs Off-policy 对比"""
import numpy as np

class SARSAAgent:  # On-policy
    def __init__(self, nS, nA, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.Q = np.zeros((nS, nA))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.nA = nA

    def select(self, s):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.nA)
        return np.argmax(self.Q[s])

    def update(self, s, a, r, s_next, a_next, done):
        target = r + self.gamma * self.Q[s_next, a_next] * (1-done)
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

class QLearningAgent:  # Off-policy
    def __init__(self, nS, nA, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.Q = np.zeros((nS, nA))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.nA = nA

    def select(self, s):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.nA)
        return np.argmax(self.Q[s])

    def update(self, s, a, r, s_next, done):
        target = r + self.gamma * np.max(self.Q[s_next]) * (1-done)
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

if __name__ == "__main__":
    np.random.seed(42)
    nS, nA = 16, 4
    sarsa = SARSAAgent(nS, nA)
    ql = QLearningAgent(nS, nA)
    for ep in range(1000):
        s = np.random.randint(nS)
        a_s = sarsa.select(s)
        a_q = ql.select(s)
        for step in range(30):
            s_next = np.random.randint(nS)
            r = 1.0 if s_next == 15 else -0.01
            done = s_next == 15
            a_s_next = sarsa.select(s_next)
            sarsa.update(s, a_s, r, s_next, a_s_next, done)
            ql.update(s, a_q, r, s_next, done)
            a_q = ql.select(s_next)
            s, a_s, a_q = s_next, a_s_next, a_q
            if done: break
    print(f"SARSA Q值范围: [{sarsa.Q.min():.2f}, {sarsa.Q.max():.2f}]")
    print(f"Q-Learning Q值范围: [{ql.Q.min():.2f}, {ql.Q.max():.2f}]")
```

## 9-14. 简要

### 12. 学习总结
On-policy：评估=执行策略（SARSA），安全但慢。Off-policy：评估$\pi$，执行$b$（Q-Learning），高效但可能不稳定。

### 13. 练习题
**Q1**：为什么Off-policy需要重要性采样？
**A1**：行为策略$b$产生的样本分布与目标策略$\pi$不同。IS比$\pi(a|s)/b(a|s)$修正分布偏差，确保估计无偏。

### 14. 学习路径
**前置**：SARSA、Q-Learning | **进阶**：重要性采样、DQN
**资源**：原书Ch 17.5、Sutton & Barto Ch 5.5-5.7
