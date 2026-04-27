# 马尔可夫决策过程 (MDP) 学习文档

> 强化学习的数学框架——状态、动作、奖励的决策模型。

> 来源线索：本节内容根据原书中关于"MDP"的相关章节（第13章13.5.1节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** MDP 用元组 $(S, A, P, R, \gamma)$ 建模序贯决策问题，目标是找到使累积奖励最大化的策略。

**直觉类比：** 下棋就是一个 MDP：棋盘状态是 $S$，合法走棋是 $A$，对手的应对是转移概率 $P$，胜负是奖励 $R$。你的目标（策略 $\pi$）是在每个状态下选择最优动作。

**算法定位：** 强化学习数学框架、序贯决策。

**前置知识：** 马尔可夫性质、概率论、期望。

---

## 2-3. 核心原理与数学公式

### MDP 五元组

$(S, A, P, R, \gamma)$
- $S$：状态集合
- $A$：动作集合
- $P(s'|s,a)$：状态转移概率
- $R(s,a)$：奖励函数
- $\gamma \in [0,1)$：折扣因子

### 状态价值函数

$$V^\pi(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s, \pi\right]$$

### Bellman 方程

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a) + \gamma V^\pi(s')]$$

### 最优价值函数

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a) + \gamma V^*(s')]$$

### Q 函数

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')$$

---

## 4-8. 代码实现

```python
import numpy as np

class GridWorldMDP:
    """4×4 网格世界 MDP"""
    def __init__(self, size=4, gamma=0.9):
        self.size = size
        self.gamma = gamma
        self.n_states = size * size
        self.n_actions = 4  # 上下左右
        self.goal = size * size - 1

    def step(self, state, action):
        row, col = state // self.size, state % self.size
        if action == 0: row = max(0, row - 1)   # 上
        elif action == 1: row = min(self.size-1, row + 1)  # 下
        elif action == 2: col = max(0, col - 1)   # 左
        elif action == 3: col = min(self.size-1, col + 1)  # 右
        next_state = row * self.size + col
        reward = 1.0 if next_state == self.goal else -0.01
        return next_state, reward

    def value_iteration(self, theta=1e-6):
        V = np.zeros(self.n_states)
        while True:
            delta = 0
            for s in range(self.n_states):
                if s == self.goal: continue
                v = V[s]
                V[s] = max(sum(self.gamma * V[s2] + r
                              for a in range(self.n_actions)
                              for s2, r in [self.step(s, a)])
                          / self.n_actions for _ in [0])
                delta = max(delta, abs(v - V[s]))
            if delta < theta: break
        return V

mdp = GridWorldMDP(size=4)
V = mdp.value_iteration()
print(f"状态价值函数:\n{V.reshape(4,4).round(3)}")
```

---

## 9-14. 练习与路径

**题1：** 马尔可夫性质为什么重要？

**参考答案：** 马尔可夫性质（$p(s_{t+1}|s_t, a_t) = p(s_{t+1}|s_0, a_0, \ldots, s_t, a_t)$）意味着未来只取决于当前状态，与历史无关。这极大简化了问题——策略只需基于当前状态，不需要记忆完整历史。

### 学习路径
- 前置：概率论、马尔可夫链
- 进阶：POMDP（部分可观测 MDP）、动态规划
