# Lagrangian松弛(Lagrangian Relaxation) 学习文档

> 将难约束移到目标函数中，用Lagrange乘子处理耦合约束。

> 来源线索：本节内容根据原书中关于"Lagrangian Relaxation"的相关章节(Ch 7.6.4, Ch 14)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：Lagrangian松弛将复杂约束通过乘子$\lambda$移到目标函数中，将约束问题变为无约束（或简单约束）问题。

**方法**：

原问题：$\min f(x)$ s.t. $g_i(x) \leq 0$

松弛：$L(x, \lambda) = f(x) + \sum_i \lambda_i g_i(x)$

对偶问题：$\max_\lambda \min_x L(x, \lambda)$，$\lambda \geq 0$

**性质**：
- 弱对偶：$\max_\lambda \min_x L \leq \min_{g(x)\leq 0} f(x)$
- 强对偶：凸问题+Slater条件时等号成立
- 对偶间隙：非凸问题的松弛误差

**在赌博机中的应用**：Gittins指数可通过对偶分解推导——每条臂独立，耦合约束通过$\lambda$处理。

## 4-8. 核心实现

```python
"""Lagrangian松弛：资源分配"""
import numpy as np

def lagrangian_relaxation(values, weights, capacity, n_iter=100, lr=0.1):
    """背包问题的Lagrangian松弛"""
    n = len(values)
    lam = 1.0  # 初始乘子
    for it in range(n_iter):
        # 松弛后：每项独立决策
        ratios = values - lam * weights
        x = (ratios > 0).astype(float)
        # 对偶上升：调整乘子
        total_weight = x @ weights
        lam = max(0, lam + lr * (total_weight - capacity))
    return x, lam

if __name__ == "__main__":
    np.random.seed(42)
    n = 20
    values = np.random.uniform(5, 20, n)
    weights = np.random.uniform(1, 10, n)
    capacity = 50
    x, lam = lagrangian_relaxation(values, weights, capacity)
    print(f"选中物品: {np.where(x==1)[0]}")
    print(f"总价值: {(x*values).sum():.1f}, 总重量: {(x*weights).sum():.1f}/{capacity}")
    print(f"最优乘子λ: {lam:.3f}")
```

## 9-14. 简要

### 12. 学习总结
Lagrangian松弛：$L(x,\lambda) = f(x) + \lambda^Tg(x)$。约束→目标，复杂问题→简单子问题。对偶上升更新$\lambda$。

### 13. 练习题
**Q1**：什么条件下强对偶成立？
**A1**：凸目标+凸约束+Slater条件（存在严格可行点）。线性规划天然满足，整数规划不满足。

### 14. 学习路径
**前置**：线性规划、对偶理论 | **进阶**：Benders分解、列生成
**资源**：原书Ch 7.6.4, Ch 14、Fisher (1981)
