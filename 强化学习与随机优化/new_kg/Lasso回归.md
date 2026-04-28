# Lasso回归 学习文档

> Lasso通过L1正则化实现特征选择和稀疏解，在保留重要特征的同时自动压缩无关特征。

> 来源线索：本节内容根据原书中关于"Sparse Additive Models and Lasso"的相关章节(Ch 3.7.2)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：Lasso（Least Absolute Shrinkage and Selection Operator）在最小化MSE的同时对参数施加L1惩罚，使部分系数精确为0，实现自动特征选择。

**直觉类比**：你有100个特征预测房价，但不确定哪些有用。普通回归会给每个特征一个系数（哪怕很小）。Lasso像一个严厉的编辑：只保留对预测最有帮助的特征，把其他特征的系数直接设为0，既简化了模型又避免了过拟合。

**历史背景**：Lasso由Robert Tibshirani在1996年提出。它结合了子集选择（特征选择）和Ridge回归（系数收缩）的优点。后续发展出Elastic Net（L1+L2混合）、Group Lasso等变体。

**算法定位**：监督学习/正则化回归。在原书Ch 3.7.2中用于构建稀疏参数模型。

**前置知识**：线性回归、正则化、优化。

## 2. 核心原理

**目标函数**：

$$\hat{\theta}^{lasso} = \arg\min_\theta \left\{\frac{1}{2N}\sum_{i=1}^N(y_i - x_i^T\theta)^2 + \lambda \sum_{j=1}^p |\theta_j|\right\}$$

- 第一项：MSE损失
- 第二项：L1惩罚（$\lambda$控制稀疏程度）
- $\lambda$越大，越多系数被压缩为0

**为什么L1产生稀疏？**

L1惩罚的约束区域是菱形（有尖角），MSE的等值线最先接触菱形的顶点（某个坐标为0），所以自然产生稀疏解。而L2的约束区域是圆形（光滑），通常不会接触坐标轴。

## 3. 数学公式

### 软阈值算子

Lasso的解可以用软阈值（soft thresholding）表示：

$$\hat{\theta}_j = \text{sign}(z_j)(|z_j| - \lambda)_+$$

其中$z_j$是无正则化时的解，$(x)_+ = \max(0, x)$。

### 坐标下降法

Lasso没有解析解，通常用坐标下降法：每次只优化一个$\theta_j$，固定其他。

### 与Ridge对比

| 特性 | Lasso(L1) | Ridge(L2) |
|------|-----------|-----------|
| 稀疏性 | 是（自动特征选择） | 否 |
| 解析解 | 无 | 有 |
| 相关特征 | 选一个 | 平均分配 |

## 4-6. 简要

### 超参数

| 参数 | 含义 | 选择 |
|------|------|------|
| $\lambda$ | 正则化强度 | 交叉验证 |

### 应用
1. 高维特征选择（基因数据）
2. 稀疏信号恢复
3. 可解释模型

## 7-8. 实现

```python
"""Lasso回归：手工坐标下降实现"""
import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _soft_threshold(self, rho, lam):
        """软阈值算子：Lasso的核心"""
        if rho > lam: return rho - lam
        elif rho < -lam: return rho + lam
        else: return 0.0

    def fit(self, X, y):
        n, p = X.shape
        self.coef_ = np.zeros(p)
        self.intercept_ = np.mean(y)
        y_centered = y - self.intercept_

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(p):
                # 残差（去掉第j个特征的贡献）
                r_j = y_centered - X @ self.coef_ + X[:, j] * self.coef_[j]
                # 坐标下降更新
                rho_j = X[:, j] @ r_j / n
                self.coef_[j] = self._soft_threshold(rho_j, self.alpha)

            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

if __name__ == "__main__":
    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    true_coef = np.zeros(p)
    true_coef[:5] = [3, -2, 1.5, -1, 0.5]  # 只有前5个有用
    y = X @ true_coef + 0.5 * np.random.randn(n)

    model = LassoRegression(alpha=0.1)
    model.fit(X, y)

    print(f"真实系数: {true_coef}")
    print(f"估计系数: {model.coef_.round(3)}")
    print(f"非零系数个数: {np.sum(np.abs(model.coef_) > 0.01)}/{p}")
```

## 9-14. 简要

### 12. 学习总结
Lasso：$\min_\theta MSE + \lambda\|\theta\|_1$。L1惩罚自动选择特征，软阈值$\hat{\theta}_j = S(z_j, \lambda)$。

### 13. 练习题
**Q1**：$\lambda=0$和$\lambda\to\infty$时Lasso分别退化为什么？
**A1**：$\lambda=0$：普通最小二乘。$\lambda\to\infty$：所有系数为0（只有截距）。

### 14. 学习路径
**前置**：线性回归 | **进阶**：Elastic Net、Group Lasso、自适应Lasso
**资源**：原书Ch 3.7.2、Tibshirani (1996)原文
