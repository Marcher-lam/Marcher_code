# Gauss-Seidel值迭代 学习文档

> 用最新估计值做异步更新，收敛速度显著快于标准值迭代。

> 来源线索：本节内容根据原书中关于"Gauss-Seidel Value Iteration"的相关章节(Ch 14.6)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：Gauss-Seidel值迭代在更新状态$s$的值时，使用已经更新过的邻居状态的新值，而非等待全部更新完再使用。

**标准VI**：$V^{n+1}(s) = \min_a [c(s,a) + \gamma \sum_{s'} P(s'|s,a) V^n(s')]$，所有状态用旧值$V^n$。

**Gauss-Seidel VI**：按固定顺序遍历状态，更新$s$时使用**最新的**$V$值（部分是$V^{n+1}$，部分是$V^n$）。

**收敛性**：因为使用了更新信息，每步的信息量更大，收敛通常比标准VI快2-3倍。

## 4-8. 核心实现

```python
"""Gauss-Seidel值迭代"""
import numpy as np

def gauss_seidel_vi(P, R, gamma=0.95, max_iter=500, tol=1e-6):
    """P: 转移概率 [nS, nA, nS], R: 奖励 [nS, nA]"""
    nS, nA = R.shape
    V = np.zeros(nS)
    for it in range(max_iter):
        delta = 0
        for s in range(nS):  # 按顺序遍历
            q_values = [R[s, a] + gamma * P[s, a] @ V for a in range(nA)]
            new_v = min(q_values)  # 代价最小化
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v  # 立即更新（GS关键）
        if delta < tol:
            return V, it + 1
    return V, max_iter

if __name__ == "__main__":
    np.random.seed(42)
    nS, nA = 20, 4
    P = np.random.dirichlet(np.ones(nS), (nS, nA))
    R = np.random.uniform(-1, 1, (nS, nA))
    R[:, -1] = 0  # 终止动作
    V, iters = gauss_seidel_vi(P, R)
    print(f"Gauss-Seidel VI在{iters}次迭代后收敛")
    print(f"值函数范围: [{V.min():.3f}, {V.max():.3f}]")
```

## 9-14. 简要

### 12. 学习总结
GS-VI：按顺序更新状态，立即使用新值。比标准VI更快收敛（通常2-3倍），实现简单。

### 13. 练习题
**Q1**：为什么GS-VI不改变收敛到的最优解？
**A1**：不动点（Bellman最优方程的解）是唯一的。GS只改变到达不动点的路径（用更即时的信息），不改变最终目标。

### 14. 学习路径
**前置**：值迭代 | **进阶**：异步DP、优先级扫描
**资源**：原书Ch 14.6
