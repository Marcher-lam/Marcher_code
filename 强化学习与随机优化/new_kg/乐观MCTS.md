# 乐观MCTS(Optimistic MCTS) 学习文档

> 在MCTS中用乐观值驱动探索，加速找到最优分支。

> 来源线索：本节内容根据原书中关于"Optimistic Monte Carlo Tree Search"的相关章节(Ch 19.8.4)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：乐观MCTS在UCB公式中加入乐观偏差，倾向于探索看起来有潜力但访问少的节点。

**UCB公式（标准UCT）**：

$$UCB(j) = \bar{X}_j + c\sqrt{\frac{\ln n}{n_j}}$$

**乐观版本**：增大探索系数$c$或用最大值替代均值：

$$UCB^{opt}(j) = \max(\bar{X}_j, \hat{X}_j^{max}) + c_{opt}\sqrt{\frac{\ln n}{n_j}}$$

**效果**：在搜索初期更积极地探索，快速排除差分支。适合搜索空间大但好解稀疏的问题。

## 4-8. 核心实现

```python
"""乐观MCTS"""
import numpy as np
import math

class OptimisticMCTSNode:
    def __init__(self, action=None, parent=None):
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.max_reward = -np.inf

    def ucb(self, c=2.0, optimistic_weight=0.3):
        if self.visits == 0: return np.inf
        mean = self.total_reward / self.visits
        optimistic = optimistic_weight * self.max_reward + (1-optimistic_weight) * mean
        return optimistic + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda c: c.ucb())

if __name__ == "__main__":
    np.random.seed(42)
    root = OptimisticMCTSNode()
    for _ in range(1000):
        node = root
        # Selection
        while node.children:
            node = node.best_child()
        # Expansion + Simulation
        for a in range(3):
            child = OptimisticMCTSNode(action=a, parent=node)
            reward = np.random.random()
            child.total_reward = reward
            child.max_reward = reward
            child.visits = 1
            node.children.append(child)
        node.visits += 1
    best = max(root.children, key=lambda c: c.total_reward/c.visits if c.visits else 0)
    print(f"最优动作: {best.action}, 均值奖励: {best.total_reward/best.visits:.3f}")
```

## 9-14. 简要

### 12. 学习总结
乐观MCTS：$UCB^{opt} = w\max + (1-w)\bar{X} + c\sqrt{\ln n/n_j}$。用乐观值加速探索，适合大搜索空间。

### 13. 练习题
**Q1**：乐观MCTS在什么情况下比标准UCT好？
**A1**：当最优解稀疏且奖励方差大时。乐观偏差帮助快速发现高奖励区域，但可能导致对差分支过度探索（过度乐观）。

### 14. 学习路径
**前置**：MCTS、UCB | **进阶**：AlphaGo搜索策略
**资源**：原书Ch 19.8.4
