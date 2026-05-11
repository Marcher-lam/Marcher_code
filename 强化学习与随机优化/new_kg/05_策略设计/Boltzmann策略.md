# Boltzmann策略 学习文档

> 按Q值的指数比例选择动作，高温探索、低温利用。

> 来源线索：本节内容根据原书中关于"Boltzmann Policy"的相关章节(Ch 12.2.2)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：Boltzmann策略按$\mathbb{P}(a|s) \propto e^{Q(s,a)/\tau}$选动作，温度$\tau$控制探索-利用权衡。

**公式**：

$$\pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$

- $\tau \to \infty$：均匀随机（纯探索）
- $\tau \to 0$：贪心$\arg\max$（纯利用）
- $\tau$介于两者：按Q值比例随机化

**与ε-greedy的区别**：ε-greedy以概率ε随机选，1-ε贪心选。Boltzmann按Q值连续调整概率——好的动作更可能被选中，但不是确定性的。

**退火**：$\tau$从大值逐渐减小，先探索后利用。

## 4-8. 核心实现

```python
"""Boltzmann策略"""
import numpy as np

class BoltzmannPolicy:
    def __init__(self, n_actions, tau=1.0):
        self.K = n_actions
        self.tau = tau
        self.Q = np.zeros(n_actions)

    def select(self):
        exp_Q = np.exp(self.Q / self.tau)
        probs = exp_Q / exp_Q.sum()
        return np.random.choice(self.K, p=probs)

    def anneal(self, step, tau_min=0.1, decay=0.995):
        self.tau = max(tau_min, self.tau * decay)

if __name__ == "__main__":
    np.random.seed(42)
    policy = BoltzmannPolicy(4, tau=2.0)
    policy.Q = np.array([1, 3, 2, 0.5])  # 动作1最优
    print(f"τ=2.0时动作概率: {(np.exp(policy.Q/2.0)/np.exp(policy.Q/2.0).sum()).round(3)}")
    policy.tau = 0.5
    print(f"τ=0.5时动作概率: {(np.exp(policy.Q/0.5)/np.exp(policy.Q/0.5).sum()).round(3)}")
    policy.tau = 0.1
    print(f"τ=0.1时动作概率: {(np.exp(policy.Q/0.1)/np.exp(policy.Q/0.1).sum()).round(3)}")
```

## 9-14. 简要

### 12. 学习总结
Boltzmann：$\pi(a) \propto e^{Q(a)/\tau}$。温度$\tau$连续控制探索，$\tau$大=探索，$\tau$小=利用。是ε-greedy的光滑替代。

### 13. 练习题
**Q1**：Q值差异很大时Boltzmann有什么问题？
**A1**：指数$e^{Q/\tau}$可能溢出（数值不稳定）。解决方案：减去最大值$e^{(Q-\max Q)/\tau}$（不改变概率分布）。

### 14. 学习路径
**前置**：ε-greedy | **进阶**：策略梯度、Actor-Critic
**资源**：原书Ch 12.2.2
