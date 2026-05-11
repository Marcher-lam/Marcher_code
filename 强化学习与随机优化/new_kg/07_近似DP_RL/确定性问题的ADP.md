# 确定性问题的ADP(ADP for Deterministic Problems) 学习文档

> 当问题没有随机性时，ADP退化为确定性优化——理解这一特殊情形有助于认识ADP的本质。

> 来源线索：本节内容根据原书中关于"Approximate Dynamic Programming for Deterministic Problems"的相关章节(Ch 17.6.3)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：在确定性设定下，ADP简化为确定性优化问题——值函数近似退化为函数拟合，不存在探索-利用困境。

## 2. 核心原理

**确定性ADP的特点**：
- 无随机噪声 → 不需要探索
- 转移完全确定 → 轨迹可精确计算
- 纯利用策略就足够
- 值函数的凹性确保纯利用收敛

**为什么纯利用就够了**：
- 凹值函数的近似会将次优值推向正确解
- 不需要epsilon-greedy或Boltzmann探索
- 梯度下降可以直接找到最优

**与随机ADP的对比**：
- 随机ADP需要探索-利用平衡
- 随机ADP的步长需要处理噪声
- 确定性ADP更简单，是理解ADP的良好起点

## 3. Python 实现

```python
import numpy as np

def deterministic_adp(transition, reward, n_states, n_actions, gamma=0.9, n_iter=100):
    """确定性ADP"""
    V = np.zeros(n_states)

    for _ in range(n_iter):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            Q = np.zeros(n_actions)
            for a in range(n_actions):
                s2 = transition(s, a)  # 确定性转移
                Q[a] = reward(s, a) + gamma * V[s2]
            V_new[s] = Q.max()
        V = V_new
    return V

# 确定性GridWorld
def grid_transition(s, a):
    row, col = s // 4, s % 4
    moves = [(0,1),(0,-1),(1,0),(-1,0)]
    dr, dc = moves[a]
    nr, nc = row + dr, col + dc
    if 0 <= nr < 4 and 0 <= nc < 4:
        return nr * 4 + nc
    return s

def grid_reward(s, a):
    return 1.0 if s == 15 else -0.1

V = deterministic_adp(grid_transition, grid_reward, 16, 4)
print("值函数:")
print(np.round(V.reshape(4,4), 2))
```

## 4. 与其他方法的关系

- **确定性优化**：ADP在无噪声时的退化
- **值迭代**：确定性ADP等价于值迭代
- **最短路径**：经典的确定性DP问题

## 5. 参考文献

- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization*, Ch 17.6.3
