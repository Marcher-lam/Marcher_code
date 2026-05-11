# 有限与无限Horizon动态规划 学习文档

> 有限horizon有终止时刻、策略与时间有关；无限horizon无终止、策略平稳——两种设定的计算方法各有特点。

> 来源线索：本节内容根据原书中关于"Finite/Infinite Horizon Dynamic Programming"的相关章节(Ch 14.2-14.5)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：有限horizon DP在固定时间T内求解最优策略（策略一般非平稳）；无限horizon DP求解长期运行的最优平稳策略。

**前置知识**：Bellman方程、值迭代、策略迭代

## 2. 核心原理

**有限Horizon DP**：
- 时间范围 $t=0,1,...,T$
- 终端条件 $V_T(s_T)$ 已知
- 逆向递推：$V_t(s) = \max_a [r(s,a) + \gamma \sum_{s'} P(s'|s,a) V_{t+1}(s')]$
- 最优策略 $\pi^*_t(s)$ 一般与时间有关（非平稳）

**无限Horizon DP**：
- 无时间限制，值函数收敛到固定点
- Bellman方程：$V(s) = \max_a [r(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')]$
- 最优策略是平稳的（不依赖时间）
- 收敛条件：$\gamma < 1$（折扣）或特殊结构（如最短路径）

**关键区别**：

| 特性 | 有限Horizon | 无限Horizon |
|---|---|---|
| 时间长度 | 固定T | ∞ |
| 策略类型 | 非平稳 | 平稳 |
| 求解方向 | 逆向递推 | 迭代逼近 |
| 终端条件 | 需要 | 不需要 |
| 存储需求 | O(T·|S|) | O(|S|) |

## 3. Python 实现

```python
import numpy as np

def finite_horizon_dp(P, R, T, gamma=0.9):
    """有限horizon动态规划"""
    n_states, n_actions = R.shape
    V = np.zeros((T+1, n_states))
    policy = np.zeros((T, n_states), dtype=int)

    for t in range(T-1, -1, -1):
        for s in range(n_states):
            Q = np.zeros(n_actions)
            for a in range(n_actions):
                Q[a] = R[s, a] + gamma * P[a][s] @ V[t+1]
            V[t, s] = Q.max()
            policy[t, s] = Q.argmax()
    return V, policy

def infinite_horizon_dp(P, R, gamma=0.9, epsilon=1e-6):
    """无限horizon值迭代"""
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    while True:
        V_new = np.zeros(n_states)
        for s in range(n_states):
            Q = np.array([R[s, a] + gamma * P[a][s] @ V for a in range(n_actions)])
            V_new[s] = Q.max()
        if np.max(np.abs(V_new - V)) < epsilon:
            break
        V = V_new
    return V
```

## 4. 与其他方法的关系

- **值迭代/策略迭代**：无限horizon的两种求解方法
- **有限horizon近似**：用有限T近似无限horizon问题

## 5. 参考文献

- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization*, Ch 14.2-14.5
