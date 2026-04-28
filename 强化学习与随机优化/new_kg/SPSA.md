# SPSA(同时扰动随机近似) 学习文档

> SPSA只需两次函数评估即可估计梯度，是高维不可微优化问题的利器。

> 来源线索：本节内容根据原书中关于"SPSA"的相关章节(Ch 5.4.4)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：SPSA（Simultaneous Perturbation Stochastic Approximation）通过同时扰动所有参数来估计梯度，每步只需两次函数评估，无论参数维度多高。

**直觉类比**：你要调节收音机的100个旋钮找到最佳信号。传统方法（有限差分）需要逐个旋钮旋转试听——每个旋钮转一次，需要200次试听。SPSA的策略是：同时随机转动所有旋钮一个小角度，听一次；再同时反向转一次，听一次。只需要2次试听就能估计出"所有旋钮该怎么调"的方向。虽然方向不精确，但大量迭代后能收敛。

**历史背景**：SPSA由James Spall在1992年提出。它的核心洞察是：虽然同时扰动所有维度的梯度估计方差比逐维扰动大，但在高维问题中只需要$O(1)$次（而非$O(p)$次）函数评估，总计算量大幅降低。

**算法定位**：无导数/随机优化方法。在原书Ch 5中，SPSA是处理不可微目标函数的核心工具。

**前置知识**：SGD、梯度估计、概率论。

## 2. 核心原理

**核心思想**：对于目标函数$L(\theta) = \mathbb{E}[F(\theta, W)]$，如果无法直接计算梯度，可以用有限差分或SPSA来估计。有限差分需要$2p$次函数评估（$p$是参数维度），SPSA只需要2次。

**工作流程**：

1. 生成随机扰动向量$\Delta_n \in \mathbb{R}^p$（各分量独立对称分布，如$\pm1$伯努利）
2. 评估两个点：$y_n^+ = F(\theta_n + c_n\Delta_n, W^+)$和$y_n^- = F(\theta_n - c_n\Delta_n, W^-)$
3. 梯度估计：$\hat{g}_n = \frac{y_n^+ - y_n^-}{2c_n} \begin{pmatrix} 1/\Delta_{n1} \\ 1/\Delta_{n2} \\ \vdots \\ 1/\Delta_{np} \end{pmatrix}$
4. 参数更新：$\theta_{n+1} = \theta_n - a_n \hat{g}_n$

**关键概念**：

- **同时扰动**：所有参数同时变化，而非逐个变化
- **扰动向量$\Delta$**：通常取$\pm1$伯努利分布（对称条件）
- **梯度估计**：利用差分商$\frac{y^+-y^-}{2c}$乘以$1/\Delta_i$
- **效率**：每步只需2次函数评估，与维度$p$无关

```
有限差分（p=100）：
  需要 2×100 = 200 次函数评估
  θ₁+ε → y₁⁺, θ₁-ε → y₁⁻  （估计 ∂L/∂θ₁）
  θ₂+ε → y₂⁺, θ₂-ε → y₂⁻  （估计 ∂L/∂θ₂）
  ... ×100

SPSA（p=100）：
  只需要 2 次函数评估！
  θ + cΔ → y⁺  （所有参数同时扰动）
  θ - cΔ → y⁻  （反向扰动）
```

## 3. 数学公式与推导

### 梯度估计

SPSA的梯度估计：

$$\hat{g}_{ni} = \frac{y_n^+ - y_n^-}{2c_n \Delta_{ni}}$$

对所有维度$i=1,...,p$使用**同一个**差分$(y^+-y^-)$，只是除以不同的$\Delta_{ni}$。

### 为什么有效

对$L(\theta)$在$\theta_n$处泰勒展开：

$$L(\theta_n + c_n\Delta_n) - L(\theta_n - c_n\Delta_n) \approx 2c_n \sum_i \frac{\partial L}{\partial \theta_i}\Delta_{ni}$$

两边除以$2c_n\Delta_{nj}$取期望：

$$\mathbb{E}\left[\frac{L(\theta_n+c_n\Delta_n)-L(\theta_n-c_n\Delta_n)}{2c_n\Delta_{nj}}\right] \approx \frac{\partial L}{\partial \theta_j} + \sum_{i\neq j}\frac{\partial L}{\partial \theta_i}\mathbb{E}\left[\frac{\Delta_{ni}}{\Delta_{nj}}\right]$$

当$\Delta$各分量独立时，$\mathbb{E}[\Delta_{ni}/\Delta_{nj}]=0$（$i\neq j$），所以估计是无偏的！

### 收敛条件

- $a_n, c_n \to 0$，$\sum a_n = \infty$
- $\Delta_n$对称分布、有界、均方差非零
- $L(\theta)$充分光滑

### 与有限差分的对比

| 特性 | 有限差分 | SPSA |
|------|---------|------|
| 每步评估次数 | $2p$ | $2$ |
| 估计方差 | 低 | 高（但可接受） |
| 适用高维 | 差 | 好 |
| 实现复杂度 | 低 | 低 |

## 4. 训练过程讲解

### 超参数表

| 参数 | 含义 | 推荐范围 | 默认值 |
|------|------|----------|--------|
| $a_n$ | 步长序列 | 衰减 | $a/(n+A+1)^\alpha$ |
| $c_n$ | 扰动幅度 | 衰减 | $c/(n+1)^\gamma$ |
| $A$ | 稳定常数 | $[100,1000]$ | 100 |
| $\alpha$ | 步长衰减率 | 0.602 | 0.602 |
| $\gamma$ | 扰动衰减率 | 0.101 | 0.101 |

## 5. 应用场景

1. **策略搜索**：优化策略参数（原书Ch 12.5-12.6）
2. **仿真优化**：目标函数是黑箱仿真
3. **神经网络训练**：目标函数不可微或梯度难以计算
4. **工程优化**：复杂系统的参数调优

## 6. 优缺点分析

### 优点
1. **评估次数少**：每步只需2次，与维度无关
2. **无需梯度**：只需要函数评估
3. **实现简单**：核心代码不到30行
4. **高维适用**：参数维度增加不增加计算量

### 缺点
1. **收敛较慢**：方差比有限差分高
2. **步长敏感**：$a_n, c_n$需要仔细调整
3. **不如基于梯度的方法**：如果梯度可用，直接用SGD更好

## 7. 调库实现

```python
"""SPSA优化器"""
import numpy as np

class SPSAOptimizer:
    def __init__(self, dim, a=0.1, c=0.1, A=100, alpha=0.602, gamma=0.101):
        self.dim = dim
        self.a, self.c, self.A = a, c, A
        self.alpha, self.gamma = alpha, gamma
        self.n = 0

    def get_gains(self):
        n = self.n + 1
        a_n = self.a / (n + self.A) ** self.alpha
        c_n = self.c / n ** self.gamma
        return a_n, c_n

    def step(self, theta, eval_fn):
        """一步SPSA更新"""
        a_n, c_n = self.get_gains()

        # 生成对称扰动（±1伯努利）
        delta = 2 * np.random.binomial(1, 0.5, self.dim) - 1

        # 两次函数评估
        y_plus = eval_fn(theta + c_n * delta)
        y_minus = eval_fn(theta - c_n * delta)

        # 梯度估计
        grad = (y_plus - y_minus) / (2 * c_n * delta)

        # 更新
        theta_new = theta - a_n * grad
        self.n += 1
        return theta_new

# 测试：优化Rosenbrock函数
if __name__ == "__main__":
    np.random.seed(42)

    def rosenbrock(x):
        return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

    dim = 2
    theta = np.array([-1.5, 2.0])
    spsa = SPSAOptimizer(dim, a=0.2, c=0.1)

    for i in range(2000):
        theta = spsa.step(theta, rosenbrock)
        if (i+1) % 500 == 0:
            print(f"Step {i+1}: f={rosenbrock(theta):.4f}, θ={theta.round(3)}")

    print(f"\n最终解: {theta.round(4)}, f={rosenbrock(theta):.6f}")
    print(f"最优解: [1.0, 1.0], f=0")
```

## 8. 手工代码实现

```python
"""从零实现SPSA（含自适应步长）"""
import numpy as np

class SPSA:
    def __init__(self, dim, lr=0.1, perturbation=0.1):
        self.dim = dim
        self.lr = lr
        self.c = perturbation
        self.iter = 0

    def optimize(self, f, x0, n_iters=1000):
        x = x0.copy()
        history = [f(x)]

        for k in range(n_iters):
            self.iter = k + 1
            # 衰减步长和扰动
            ak = self.lr / (k + 1) ** 0.602
            ck = self.c / (k + 1) ** 0.101

            # ±1 伯努利扰动
            delta = 2 * (np.random.random(self.dim) > 0.5).astype(float) - 1

            # 两次评估
            fp = f(x + ck * delta)
            fm = f(x - ck * delta)

            # 梯度估计
            ghat = (fp - fm) / (2 * ck * delta)

            # 更新
            x = x - ak * ghat
            history.append(f(x))

        return x, history

if __name__ == "__main__":
    np.random.seed(42)
    # 二次函数优化
    A = np.array([[3, 1], [1, 2]])
    b = np.array([1, -1])
    f = lambda x: 0.5 * x @ A @ x - b @ x + 2
    x_opt = np.linalg.solve(A, b)

    spsa = SPSA(2, lr=0.3, perturbation=0.2)
    x, hist = spsa.optimize(f, np.array([5.0, -3.0]), n_iters=2000)
    print(f"SPSA解: {x.round(3)}, 精确解: {x_opt.round(3)}")
    print(f"最终值: {f(x):.4f}, 最优值: {f(x_opt):.4f}")
```

## 9-14. 简要补充

### 9. 可视化
绘制SPSA vs SGD的收敛曲线对比，SPSA前期波动大但能收敛。

### 10. 评估
比较SPSA与有限差分和SGD的收敛速度和最终精度。

### 11. 常见问题
1. **$\Delta$不能用正态分布**：需要对称有界分布（用$\pm1$伯努利）
2. **扰动幅度太大**：梯度估计不准 → 按Spall推荐公式设置$c_n$
3. **噪声过大**：多次平均或使用自适应步长

### 12. 学习总结
SPSA用2次函数评估估计$p$维梯度：$\hat{g}_i = (y^+-y^-)/(2c\Delta_i)$。适用于不可微、高维、黑箱优化。

### 13. 练习题
**Q1**：为什么SPSA的扰动$\Delta$必须满足对称分布？
**A1**：无偏性要求$\mathbb{E}[\Delta_i/\Delta_j]=0$（$i\neq j$）。对称分布保证此条件成立。

### 14. 学习路径
**前置**：SGD | **进阶**：SPSA二阶方法（2SPSA）、策略搜索中的SPSA（Ch 12.5）
**资源**：原书Ch 5.4.4、Spall "Introduction to Stochastic Search and Optimization" (2003)
