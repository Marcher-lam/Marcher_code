# 正则化Benders分解(Benders with Regularization) 学习文档

> 在标准Benders分解中加入正则项，稳定迭代过程并加速收敛。

> 来源线索：本节内容根据原书中关于"Benders with Regularization"的相关章节(Ch 18.6.3)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：正则化Benders分解在标准Benders切割平面上添加 proximal 正则项，约束每步迭代不偏离当前解太远，避免振荡和发散。

**算法定位**：凸资源分配 / 分解方法。是标准Benders分解的改进变体。

**前置知识**：Benders分解、凸优化、proximal点算法。

## 2. 核心原理

**核心思想**：标准Benders分解通过迭代添加切割平面逐步逼近最优解，但在某些情况下（尤其是非光滑或病态问题时）会产生剧烈振荡。正则化方法引入一个**稳定中心**$\bar{x}$和正则项$\frac{1}{2\lambda}\|x - \bar{x}\|^2$，确保每次迭代不会偏离稳定中心太远。

**与标准Benders的区别**：
- 标准Benders：每次求解主问题得到新的$x$，可能跳很远
- 正则化Benders：主问题加入$\frac{1}{2\lambda}\|x - \bar{x}\|^2$，约束$x$靠近$\bar{x}$
- 收到足够的改进后再更新稳定中心$\bar{x}$

**收敛优势**：正则化保证了目标函数的单调改进，避免切割平面方法常见的"zig-zag"振荡。

## 3. 数学公式

### 标准Benders主问题

$$\min_x \quad c^T x + \hat{Q}(x)$$

其中$\hat{Q}(x) = \max_k \{ \alpha_k + \beta_k^T x \}$是近似的前景函数（由切割平面构成）。

### 正则化Benders主问题

$$\min_x \quad c^T x + \hat{Q}(x) + \frac{1}{2\lambda}\|x - \bar{x}\|^2$$

其中：
- $\bar{x}$是当前稳定中心（anchor point）
- $\lambda > 0$是正则化参数（越小越约束在$\bar{x}$附近）
- 每次迭代后，如果目标改进足够大，更新$\bar{x} \leftarrow x^*$

### 切割平面（与标准Benders相同）

第$k$次迭代产生切割：

$$\alpha_k + \beta_k^T x \leq Q(x)$$

### 收敛条件

当$\|x^* - \bar{x}\| < \epsilon$且改进量$< \epsilon$时，算法收敛。

## 4. 训练/迭代过程

1. 初始化：选择初始$\bar{x}$，设$\lambda$，切割平面集合为空
2. 求解正则化主问题，得到$x^*$
3. 对$x^*$求解子问题，得到对偶解$\pi$
4. 生成新切割$\alpha_k + \beta_k^T x$，加入切割集
5. 计算改进量：若目标值显著下降，更新$\bar{x} \leftarrow x^*$
6. 检查收敛：若$\|x^* - \bar{x}\| < \epsilon$，停止；否则回到步骤2

## 5. 应用场景

- 多期资源分配问题（原书Ch 18核心应用）
- 两阶段随机规划（场景数很多时）
- 电力系统调度（机组组合）
- 供应链网络设计
- 交通流量分配

## 6. 优缺点

**优点**：
- 比标准Benders更稳定，减少振荡
- 保证目标单调改进
- 对病态问题收敛更快

**缺点**：
- 需要调节正则化参数$\lambda$
- 每步主问题多了一个二次项（但仍为凸QP）
- 参数选择对收敛速度影响大

## 7. 库实现

Python中可用 `scipy.optimize` 或 `cvxpy` 实现正则化主问题，子问题为LP。

## 8. 从零实现

```python
"""正则化Benders分解"""
import numpy as np
from scipy.optimize import linprog

class RegularizedBenders:
    def __init__(self, c, n_scenarios, lam=1.0, tol=1e-4):
        self.c = c                # 一阶段成本向量
        self.n_s = n_scenarios
        self.lam = lam            # 正则化参数
        self.tol = tol
        self.cuts_alpha = []      # 切割常数项
        self.cuts_beta = []       # 切割斜率向量
        self.n_x = len(c)

    def solve_subproblem(self, x, scenario):
        """子问题：给定x和场景，返回对偶变量和目标值"""
        # 简化示例：二次子问题 Q_s(x) = max{0, (x - scenario)**2}
        q_val = max(0, (x[0] - scenario) ** 2)
        # 对偶/梯度：dQ/dx = 2*(x - s) if x > s
        grad = np.zeros(self.n_x)
        if x[0] > scenario:
            grad[0] = 2 * (x[0] - scenario)
        alpha = q_val - grad @ x
        return q_val, alpha, grad

    def solve_master(self, x_anchor):
        """正则化主问题：线性规划 + 二次正则项的近似"""
        best_x = x_anchor.copy()
        best_val = float('inf')
        # 枚举切割的极点（简化：取切割中最紧的那个 + 正则项）
        for _ in range(200):
            # 当前目标：c^T x + max_cuts(alpha + beta^T x) + 1/(2*lam)*||x - anchor||^2
            # 梯度下降近似求解
            grad = self.c.copy()
            # 最紧切割的梯度
            if self.cuts_alpha:
                q_vals = [a + b @ best_x for a, b in zip(self.cuts_alpha, self.cuts_beta)]
                k = np.argmax(q_vals)
                grad += self.cuts_beta[k]
            # 正则项梯度
            grad += (best_x - x_anchor) / self.lam
            # 更新
            lr = 0.1
            best_x_new = best_x - lr * grad
            val = self.c @ best_x_new
            if self.cuts_alpha:
                val += max(a + b @ best_x_new for a, b in zip(self.cuts_alpha, self.cuts_beta))
            val += np.sum((best_x_new - x_anchor)**2) / (2 * self.lam)
            if abs(val - best_val) < 1e-8:
                break
            best_x = best_x_new
            best_val = val
        return best_x, best_val

    def solve(self, scenarios):
        x_anchor = np.zeros(self.n_x)
        for iteration in range(50):
            x_star, master_val = self.solve_master(x_anchor)
            # 所有场景的子问题
            total_q = 0
            for s in scenarios:
                q_val, alpha, beta = self.solve_subproblem(x_star, s)
                total_q += q_val / len(scenarios)
                self.cuts_alpha.append(alpha)
                self.cuts_beta.append(beta)
            # 更新稳定中心
            old_obj = self.c @ x_anchor + total_q
            new_obj = self.c @ x_star + total_q
            if new_obj < old_obj - self.tol:
                x_anchor = x_star.copy()
            # 收敛检查
            if np.linalg.norm(x_star - x_anchor) < self.tol:
                break
        return x_star

if __name__ == "__main__":
    np.random.seed(42)
    c = np.array([1.0])
    scenarios = np.random.randn(20) * 2 + 5
    rb = RegularizedBenders(c, n_scenarios=len(scenarios), lam=0.5)
    x_opt = rb.solve(scenarios)
    print(f"正则化Benders最优解: x* = {x_opt[0]:.4f}")
    print(f"场景均值: {scenarios.mean():.4f}")
```

## 9. 可视化

```python
import matplotlib.pyplot as plt
# 绘制切割平面逐步逼近Q(x)的过程
# x轴：x值，y轴：Q(x)
# 每条虚线是一个切割，粗实线是它们的逐点最大值
# 正则化使迭代路径更平滑
```

## 10. 评估方法

- 收敛迭代次数（vs 标准Benders）
- 目标值改进曲线（应单调下降）
- 解的质量（与精确解对比）

## 11. 常见错误

- $\lambda$太小：过度约束，收敛极慢
- $\lambda$太大：退化为标准Benders，失去稳定性
- 稳定中心更新策略不当：过早或过晚更新

## 12. 学习总结

正则化Benders：主问题加$\frac{1}{2\lambda}\|x - \bar{x}\|^2$正则项，约束迭代不偏离稳定中心。比标准Benders更稳定，减少振荡，适合病态或大规模问题。

## 13. 练习题

**Q1**：正则化参数$\lambda$如何影响收敛行为？
**A1**：$\lambda$小→强约束→每步改进小但稳定→收敛慢但不易发散；$\lambda$大→弱约束→接近标准Benders→收敛快但可能振荡。实践中常从小$\lambda$开始，逐步增大（warm-start策略）。

**Q2**：何时应该使用正则化而非标准Benders？
**A2**：当标准Benders出现明显振荡（目标值上下波动而非单调下降），或切割平面非常密集导致主问题解空间窄时。非光滑前景函数尤其受益。

## 14. 学习路径

**前置**：Benders分解、凸优化 | **进阶**：Level方法、Bundle方法
**资源**：原书Ch 18.6.3、Ruszczyński (2006) Chap 3
