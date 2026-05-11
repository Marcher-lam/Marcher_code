# Epsilon-Greedy策略 学习文档

> 以概率ε随机探索，1-ε贪心利用，最简单也最常用的探索策略。

> 来源线索：本节内容根据原书中关于"Epsilon-Greedy"的相关章节(Ch 11-17)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：ε-greedy以概率ε随机选动作（探索），1-ε选当前最优动作（利用）。

**公式**：

$$a_t = \begin{cases} \arg\max_a Q(s,a) & \text{概率 } 1-\epsilon \\ \text{random} & \text{概率 } \epsilon \end{cases}$$

**变体**：
- **衰减ε**：$\epsilon_t = \epsilon_0 / (1 + \beta t)$，初期探索后期利用
- **ε-first**：前$N$步纯探索，之后纯利用
- **自适应ε**：根据学习进度调整

**理论**：固定ε的遗憾$O(T)$（线性增长，非最优）。衰减$\epsilon_t = c/t$可达$O(\ln T)$。

## 4-8. 核心实现

```python
"""Epsilon-Greedy策略"""
import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.K = n_arms
        self.epsilon = epsilon
        self.Q = np.zeros(n_arms)
        self.n = np.zeros(n_arms)

    def select(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.K)
        return np.argmax(self.Q)

    def update(self, k, r):
        self.n[k] += 1
        self.Q[k] += (r - self.Q[k]) / self.n[k]

class DecayEpsilonGreedy(EpsilonGreedy):
    def __init__(self, n_arms, eps_start=1.0, eps_min=0.01, decay=0.999):
        super().__init__(n_arms, eps_start)
        self.eps_min = eps_min
        self.decay = decay

    def select(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.decay)
        return super().select()

if __name__ == "__main__":
    np.random.seed(42)
    means = [0.9, 0.6, 0.3]
    for name, agent in [
        ("ε=0.1固定", EpsilonGreedy(3, 0.1)),
        ("ε衰减", DecayEpsilonGreedy(3, 1.0, 0.01, 0.995)),
    ]:
        total_r = 0
        for t in range(1000):
            k = agent.select()
            r = float(np.random.random() < means[k])
            agent.update(k, r)
            total_r += r
        print(f"{name}: 累积奖励={total_r}, Q={agent.Q.round(3)}")
```

## 9-14. 简要

### 12. 学习总结
ε-greedy：概率ε随机，1-ε贪心。最简单的探索策略，衰减ε性能更优。固定ε遗憾$O(T)$，衰减ε遗憾$O(\ln T)$。

### 13. 练习题
**Q1**：ε=0.1意味着什么？
**A1**：每10步约有1步随机探索。对于K=10个动作，最优动作被选概率≈0.91，每个其他动作≈0.01。

### 14. 学习路径
**前置**：探索利用困境 | **进阶**：Boltzmann、UCB、Thompson采样
**资源**：原书Ch 11-17、Sutton & Barto Ch 2.2
