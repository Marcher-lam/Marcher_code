# ε-greedy 学习文档

> 强化学习中平衡探索与利用的经典动作选择策略。

## 1. 算法基础认知

ε-greedy（Epsilon-Greedy）是强化学习中用于平衡探索（Exploration）与利用（Exploitation）的经典策略。它以概率ε进行随机探索，以概率1-ε选择当前认为最优的动作。

**直觉类比**：想象你在一家餐厅点菜。如果你总是点之前吃过的菜（利用），你不会发现新的美味；如果你总是随机点菜（探索），你可能吃到难吃的菜。ε-greedy的策略是：80%的时候点常吃的菜（利用），20%的时候随机尝试新菜（探索）。

**前置知识**：强化学习基础、Q-Learning

## 2. 核心原理

ε-greedy在探索和利用之间取得平衡：
- 随机探索：防止陷入局部最优
- 利用：最大化即时奖励

## 3. 数学公式与推导

动作选择概率：
$$P(a|s) = \begin{cases} epsilon / |A| + (1 - epsilon) & text{if } a = argmax_{a'} Q(s,a') \ epsilon / |A| & text{otherwise} end{cases}$$

## 4. 训练过程讲解

参数：
- ε：探索率，通常随训练衰减
- ε_min：最小探索率
- decay_rate：衰减率

## 5. 应用场景

- 所有探索-利用权衡问题
-  bandits
- 强化学习

## 6. 优缺点分析

**优点**：简单易实现、有效
**缺点**：ε固定时不高效

## 7. 调库实现

```python
"""
ε-greedy策略实现
"""
import numpy as np

class EpsilonGreedy:
    """ε-greedy动作选择策略"""
    def __init__(self, n_actions, epsilon=0.1, epsilon_min=0.01, decay_rate=0.995):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
    
    def select_action(self, Q_values, training=True):
        """选择动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(Q_values)
    
    def decay(self):
        """衰减ε"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

# 测试
np.random.seed(42)
eg = EpsilonGreedy(4, epsilon=0.2)

Q = np.array([0.5, 0.8, 0.3, 0.6])
for i in range(10):
    action = eg.select_action(Q)
    print(f"ε={eg.epsilon:.3f}, 选择动作: {action}, Q值: {Q[action]:.3f}")
    eg.decay()
```

## 8. 手工代码实现

```python
def epsilon_greedy(Q, n_actions, epsilon):
    """ε-greedy动作选择"""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q)

# 示例
Q_values = np.array([1.0, 2.0, 0.5, 1.5])
print("ε-greedy结果:", epsilon_greedy(Q_values, 4, 0.1))
```

## 9-14. 其他章节

代码已包含以上内容。学习总结：ε-greedy是探索-利用平衡的基础策略。

> 来源线索：本节内容根据原书中关于"ε-greedy transition policy"的相关章节整理。