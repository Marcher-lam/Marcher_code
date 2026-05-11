# Benders分解 学习文档

> Benders分解将大规模随机规划问题拆成主问题和子问题迭代求解，是两阶段随机规划的经典算法。

> 来源线索：本节内容根据原书中关于"Benders Decomposition"的相关章节(Ch 18.6)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：Benders分解将复杂的随机优化问题分解为一个主问题（确定第一阶段决策）和多个子问题（评估第二阶段成本），通过迭代添加割平面收敛到最优解。

**直觉类比**：你要规划一周的食材采购（第一阶段决策），但不确定每天有多少客人（随机性）。你先做一个初步采购方案，然后让厨房厨师（子问题）评估："按这个采购量，如果客人多/少，还要花多少额外成本？"厨师反馈一个"教训"（割平面），帮你改进采购方案。反复迭代直到采购方案足够好。

**历史背景**：由Jacques Benders在1962年提出。Van Slyke & Wets (1969)将其扩展到随机规划（L-shaped method）。是大规模优化和随机规划的核心算法。

**算法定位**：优化/分解方法。在原书Ch 18.6中用于求解资源分配问题的值函数近似。

**前置知识**：线性规划、对偶理论、随机规划基础。

## 2. 核心原理

**核心思想**：对于两阶段随机规划：第一阶段决策$x$，第二阶段对每个情景$\omega$求解子问题$Q(x,\omega)$。Benders将$x$和$Q$分离：主问题决定$x$并估计$Q$，子问题精确计算$Q(x,\omega)$并返回割平面（改进估计）。

**工作流程**：

1. 求解主问题（当前对第二阶段成本的近似），得到$x^n$
2. 对每个情景$\omega$，求解子问题得到$Q(x^n,\omega)$和对偶解
3. 生成割平面$\theta \geq (h-Tx)^T\pi$（Benders割）
4. 将割平面加入主问题
5. 重复直到收敛

## 3. 数学公式与推导

### 两阶段随机规划

$$\min_x \left[c^Tx + \mathbb{E}_\omega[Q(x,\omega)]\right]$$

其中$Q(x,\omega) = \min_y \{q(\omega)^Ty : W(\omega)y = h(\omega)-T(\omega)x, y\geq 0\}$

### Benders割平面

对偶子问题：$\max_\pi \{(h-Tx)^T\pi : W^T\pi \leq q\}$

在极点$\pi^k$处的割平面：$\theta \geq (h^k-T^kx)^T\pi^k$

### 多割vs单割

- 单割（L-shaped）：所有情景生成一个平均割
- 多割：每个情景独立割，收敛更快但规模更大

### 收敛性

有限情景数下，Benders分解有限步收敛。

## 4-6. 简要

### 超参数

| 参数 | 含义 |
|------|------|
| 情景数 | 采样情景数量 |
| 收敛阈值 | $\epsilon$ |

### 应用
1. 能源系统规划
2. 供应链网络设计
3. 交通规划

### 优缺点
**优点**：可分解大规模问题、支持并行求解子问题
**缺点**：收敛可能较慢（需要正则化改进）、需要线性/凸结构

## 7-8. 实现

```python
"""Benders分解求解简单两阶段随机规划"""
import numpy as np
from scipy.optimize import linprog

def benders_decomposition(c, Q_senarios, T_senarios, h_senarios, n_senarios, max_iter=50):
    """
    两阶段随机规划的Benders分解
    min c'x + E[Q(x,ω)]
    """
    n_x = len(c)
    cuts = []  # 割平面集合

    for iteration in range(max_iter):
        # 主问题：min c'x + θ  s.t. 割平面约束
        n_vars = n_x + 1
        c_master = np.zeros(n_vars)
        c_master[:n_x] = c
        c_master[-1] = 1.0  # θ的系数

        A_ub = []
        b_ub = []
        for alpha, beta in cuts:
            row = np.zeros(n_vars)
            row[:n_x] = -beta
            row[-1] = -1.0  # θ ≥ α + β'x → -β'x - θ ≤ -α
            A_ub.append(row)
            b_ub.append(-alpha)

        bounds = [(None, None)] * n_x + [(0, None)]  # θ ≥ 0
        result = linprog(c_master, A_ub=np.array(A_ub) if A_ub else None,
                        b_ub=np.array(b_ub) if b_ub else None, bounds=bounds)
        x_star = result.x[:n_x]

        # 子问题：对每个情景求解
        theta_est = 0
        new_alpha, new_beta = 0, np.zeros(n_x)
        for s in range(n_senarios):
            rhs = h_senarios[s] - T_senarios[s] @ x_star
            theta_est += rhs @ np.linalg.solve(Q_senarios[s] @ Q_senarios[s].T + np.eye(len(rhs))*0.01, rhs) / n_senarios

        # 生成割平面（简化：用二次近似）
        # θ ≥ α + β'x
        new_alpha = theta_est + new_beta @ x_star
        cuts.append((new_alpha, new_beta))

        if iteration > 0 and abs(theta_est - result.x[-1]) < 1e-4:
            print(f"Benders分解在第{iteration+1}轮收敛")
            break

    return x_star

if __name__ == "__main__":
    np.random.seed(42)
    c = np.array([1.0, 2.0])
    K = 5  # 情景数
    Q_senarios = [np.eye(2) for _ in range(K)]
    T_senarios = [np.random.randn(2, 2) for _ in range(K)]
    h_senarios = [np.random.randn(2) for _ in range(K)]
    x = benders_decomposition(c, Q_senarios, T_senarios, h_senarios, K)
    print(f"最优第一阶段决策: {x.round(3)}")
```

## 9-14. 简要补充

### 9. 可视化
绘制目标函数值随迭代次数的收敛曲线。

### 10. 评估
比较Benders分解与直接求解的精度和计算时间。

### 11. 常见问题
1. **收敛慢**：加入正则化项（原书18.6.3）
2. **子问题不可行**：添加可行性割
3. **情景数过多**：用采样Benders（SAA）

### 12. 学习总结
Benders分解将$\min_x c'x + E[Q(x,ω)]$拆为主问题（定$x$）和子问题（算$Q$），通过割平面迭代改进。

### 13. 练习题
**Q1**：Benders分解中的割平面对应优化理论中的什么概念？
**A1**：割平面是目标函数的次梯度线性支撑，对应凸分析中的支撑超平面。每次迭代逐步逼近凸函数的epigraph。

### 14. 学习路径
**前置**：线性规划对偶 | **进阶**：随机分解(SD)、Level decomposition、整数Benders
**资源**：原书Ch 18.6、Birge & Louveaux "Introduction to Stochastic Programming"
