# Boltzmann 探索 学习文档

> 基于 softmax 概率分布的探索策略。

## 1. 算法基础认知

Boltzmann探索（或 Softmax 探索）是强化学习中另一种平衡探索与利用的策略，它使用 softmax 分布来选择动作，概率与动作的Q值指数相关。

**直觉类比**：在餐厅点菜，不是完全随机（ε-greedy），而是对喜欢的菜概率高，对其他的菜概率低。喜欢程度由"温度"控制：温度高时更随机，温度低时更确定性。

**前置知识**：ε-greedy、Q-Learning

## 2. 核心原理

动作选择概率：
$$P(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{b} e^{Q(s,b)/\tau}}$$

其中τ是温度参数。

## 3. 数学公式与推导

温度τ的作用：
- τ → ∞：接近均匀分布（完全随机）
- τ → 0：接近贪心策略（完全利用）

## 4. 训练过程讲解

- τ：温度参数
- τ_decay：温度衰减率

## 5. 应用场景

- 动作空间较小的问题
- 需要平滑策略变化的问题

## 6. 优缺点分析

**优点**：策略平滑过渡
**缺点**：需要选择合适的温度τ

## 7. 调库实现

```python
"""
Boltzmann探索实现
"""
import numpy as np

class BoltzmannExploration:
    """Boltzmann探索策略"""
    def __init__(self, tau=1.0, tau_min=0.01, decay_rate=0.99):
        self.tau = tau
        self.tau_min = tau_min
        self.decay_rate = decay_rate
    
    def select_action(self, Q_values):
        """Softmax动作选择"""
        exp_q = np.exp(Q_values / self.tau)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(Q_values), p=probs)
    
    def decay(self):
        """温度衰减"""
        self.tau = max(self.tau_min, self.tau * self.decay_rate)

# 测试
np.random.seed(42)
bl = BoltzmannExploration(tau=2.0, decay_rate=0.95)
Q = np.array([0.5, 0.8, 0.3, 1.0])

for i in range(10):
    action = bl.select_action(Q)
    print(f"τ={bl.tau:.3f}, 选择动作: {action}")
    bl.decay()
```

## 8. 手工代码实现

```python
def boltzmann_select(Q, tau):
    """Boltzmann选择"""
    exp_q = np.exp(Q / tau)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(len(Q), p=probs)

# 示例
print("结果:", boltzmann_select(np.array([1.0, 2.0, 0.5]), 1.0))
```

## 9-14. 其他章节

核心公式：
$$P(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_b e^{Q(s,b)/\tau}}$$

> 来源线索：本节内容根据原书中关于"Boltzmann distribution"的相关章节整理。