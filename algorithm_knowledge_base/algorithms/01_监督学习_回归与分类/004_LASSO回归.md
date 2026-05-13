# LASSO回归 学习文档

> LASSO (Least Absolute Shrinkage and Selection Operator) 回归是一种使用L1正则化的线性回归方法,能够实现特征选择和稀疏解。

---

## 1. 算法基础认知

### 一句话定义
LASSO回归通过在损失函数中加入L1正则项 $\lambda\|\beta\|_1$,使得部分系数精确变为0,从而实现特征选择。

### 直觉类比
想象整理衣柜:
- **岭回归**:把衣服都叠小一点(系数缩小)
- **LASSO**:直接把不常穿的衣服扔掉(稀疏/置零)

### 历史背景
- 1996年,Tibshirani在JRSS提出
- 与岭回归并驾齐驱的特征选择方法
- 成为压缩感知理论的基础

### 算法定位
- **类型**:线性回归/正则化
- **输出**:稀疏系数 + 预测
- **模型类型**:线性模型 + L1正则

### 前置知识
- 线性回归
- 梯度下降
- 矩阵范数

---

## 2. 核心原理

### 2.1 核心思想
LASSO核心是**L1正则化**:
$$\min_{\beta} \|y - X\beta\|^2 + \lambda\|\beta\|_1$$

L1范数的**尖点**特性会使得解偏向稀疏!

### 2.2 工作流程
```
X, y, λ
    ↓
初始化 β = 0
    ↓
迭代:
    计算梯度(次梯度)
    软阈值更新
    ↓
稀疏解 ← 收敛
```

### 2.3 关键概念
- **L1范数**: $\|\beta\|_1 = \sum_j |\beta_j|$
- **软阈值**: $S_\lambda(z) = \text{sign}(z)(|z| - \lambda)_+$
- **稀疏性**: 某些系数精确为0
- **LARSEN**: 加速实现

### 2.4 几何直观
```
┌─────────────────────────────────────────────┐
│          岭回归 vs LASSO                    │
│                                             │
│    β2      LASSO           岭回归          │
│     ↑        \   /             \            │
│     │         \ /               \           │
│     │          \/                 \         │
│     │    ┌────┼────┐         ┌──┼──┐     │
│     │    │    │    │         │  │  │       │
│     │    │         \ /       \ /  \ /      │
│     │    │          \     \ /    \ /        │
│     │    │           \  __/__/___         │
│     │    │           \ ////            │
│     └────┼─────────-●●●---------------→ β1
│                    \/                    │
│               最优点                    │
│                                             │
│ LASSO:菱形约束 → 常接触点 → 稀疏解        │
│ 岭回归:圆形约束 → 圆滑接触 → 无稀疏   │
└─────────────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $X$ | 特征矩阵 $(n, p)$ |
| $y$ | 目标向量 $(n,)$ |
| $\beta$ | 系数向量 $(p,)$ |
| $\lambda$ | 正则化参数 |
| $\\|\beta\|_1$ | L1范数 |

### 3.2 问题形式化

$$\min_{\beta} \frac{1}{2n} \|y - X\beta\|^2_2 + \lambda \|\beta\|_1$$

或等价:
$$\min_{\beta} \|y - X\beta\|^2_2 \quad \text{s.t.} \sum_j |\beta_j| \leq t$$

### 3.3 软阈值推导

**一维情况**:
$$\min_\beta \frac{1}{2}(y - \beta)^2 + \lambda |\beta|$$

解为软阈值:
$$\beta^* = S_\lambda(y) = \begin{cases} y - \lambda & y > \lambda \\ 0 & |y| \leq \lambda \\ y + \lambda & y < -\lambda \end{cases}$$

### 3.4 坐标下降法

对每个坐标j:
$$\beta_j = \frac{1}{X_j^T X_j} S_\lambda(r_j - \sum_{k \neq j} X_j X_k \beta_k)$$

其中 $r_j = X_j^T(y - X\beta_{-j})$ 是残差。

### 3.5 算法步骤

```python
# LASSO 伪代码

def LASSO(X, y, lambda):
    beta = zeros(p)
    
    for iteration in max_iter:
        for j in range(p):
            # 计算残差
            r = y - X @ beta + X[:, j] * beta[j]
            
            # 软阈值更新
            rho = X[:, j] @ r
            beta[j] = soft_threshold(rho, lambda) / (X[:, j] @ X[:, j])
        
        # 检查收敛
        if converged: break
    
    return beta


def soft_threshold(z, lambda):
    if z > lambda: return z - lambda
    elif z < -lambda: return z + lambda
    else: return 0
```

---

## 4. 训练过程

### 4.1 实现代码

```python
"""
LASSO 回归完整实现
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

class LASSORegression(BaseEstimator, RegressorMixin):
    """LASSO回归"""
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, warm_start=False):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.coef_ = None
        self.n_iter_ = None
    
    def _soft_threshold(self, x, threshold):
        """软阈值函数"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _coordinate_descent(self, X, y, lambda_):
        """坐标下降法"""
        n, p = X.shape
        beta = np.zeros(p)
        
        # 预计算 XTX
        XTX = np.sum(X ** 2, axis=0)
        
        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            
            for j in range(p):
                # 计算残差
                residual = y - X @ beta + X[:, j] * beta[j]
                
                # 计算rho
                rho = X[:, j] @ residual
                
                # 软阈值更新
                if XTX[j] > 0:
                    beta[j] = self._soft_threshold(rho, lambda_ * n) / XTX[j]
            
            # 检查收敛
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        return beta
    
    def _least_angle_regression(self, X, y, lambda_):
        """LARS实现(更快)"""
        n, p = X.shape
        
        # 标准化
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_norm = (X - X_mean) / X_std
        
        y_mean = y.mean()
        y_norm = y - y_mean
        
        # 初始化
        active = []
        beta = np.zeros(p)
        corr = X_norm.T @ y_norm
        lambda_max = np.max(np.abs(corr))
        
        if lambda_ >= lambda_max:
            return beta
        
        # LARS迭代
        while lambda_ < lambda_max:
            # 选择最相关的未选特征
            remaining = np.setdiff1d(np.arange(p), active)
            if len(remaining) == 0:
                break
            
            corr_remaining = corr[remaining]
            new = remaining[np.argmax(np.abs(corr_remaining))]
            
            if abs(corr[new]) < lambda_:
                break
            
            active.append(new)
            
            if len(active) == 1:
                beta[new] = corr[new] / (X_norm[:, new] ** 2).sum()
            else:
                # 简化的LASSO更新
                X_active = X_norm[:, active]
                beta_active = np.linalg.lstsq(X_active, y_norm, rcond=None)[0]
                
                # LASSO修正
                for i, idx in enumerate(active):
                    if np.abs(beta_active[i]) < lambda_:
                        beta_active[i] = 0
                
                beta[active] = beta_active
            
            # 更新相关性
            residual = y_norm - X_norm @ beta
            corr = X_norm.T @ residual
            lambda_max = np.max(np.abs(corr))
        
        return beta
    
    def fit(self, X, y):
        """训练"""
        n = len(y)
        
        if self.alpha <= 0:
            # 无正则,标准最小二乘
            self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            # 坐标下降法
            self.coef_ = self._coordinate_descent(X, y, self.alpha)
        
        return self
    
    def predict(self, X):
        """预测"""
        return X @ self.coef_
    
    def score(self, X, y):
        """R²分数"""
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


class ElasticNetRegression:
    """弹性网回归(L1+L2)"""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        n, p = X.shape
        
        # 分离L1和L2
        lambda_1 = self.alpha * self.l1_ratio
        lambda_2 = self.alpha * (1 - self.l1_ratio)
        
        beta = np.zeros(p)
        
        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            
            for j in range(p):
                residual = y - X @ beta + X[:, j] * beta[j]
                rho = X[:, j] @ residual
                
                # 弹性网: L2 + 软阈值
                denom = (X[:, j] ** 2).sum() + lambda_2 * n
                
                if rho > lambda_1:
                    beta[j] = (rho - lambda_1) / denom
                elif rho < -lambda_1:
                    beta[j] = (rho + lambda_1) / denom
                else:
                    beta[j] = 0
            
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break
        
        self.coef_ = beta
        return self
```

### 4.2 收敛条件
- 系数变化 < tol
- 或达到最大迭代

### 4.3 超参数

| 参数 | 说明 | 推荐 |
|------|------|------|
| alpha (lambda) | 正则强度 | 0.1~10,交叉验证 |
| l1_ratio | L1比例 | 弹性网用 |
| max_iter | 最大迭代 | 1000 |

---

## 5. 应用场景

### 5.1 典型应用
- **特征选择**:基因筛选、变量选择
- **预测**:带稀疏性的回归
- **压缩感知**:信号重建

### 5.2 适用数据
- 高维数据($p > n$)
- 稀疏真实系数
- 需要可解释性

---

## 6. 优缺点

### 6.1 优点
| 优点 | 说明 |
|------|------|
| 稀疏解 | 自动特征选择 |
| 高维可用 | $p > n$ |
| 可解释 | 非零系数少 |

### 6.2 缺点
| 缺点 | 缓解 |
|------|------|
| 慢 | LARS加速 |
| 不稳定 | 弹性网 |
| 特征相关 | 组LASSO |

---

## 7. 调库实现

```python
"""
sklearn LASSO
"""
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

# LASSO
lasso = Lasso(alpha=1.0, max_iter=10000)
lasso.fit(X, y)
predictions = lasso.predict(X_test)

# 弹性网
enet = ElasticNet(alpha=1.0, l1_ratio=0.5)
enet.fit(X, y)
```

---

## 8. 手工实现

```python
"""
LASSO 核心简化版
"""

import numpy as np

class SimpleLASSO:
    """简化LASSO"""
    
    def __init__(self, lambda_=1.0, max_iter=1000):
        self.lambda_ = lambda_
        self.max_iter = max_iter
    
    def fit(self, X, y):
        n, p = X.shape
        
        # 标准化
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        X = (X - self.X_mean) / self.X_std
        
        self.y_mean = y.mean()
        y = y - self.y_mean
        
        # 坐标下降
        self.beta = np.zeros(p)
        
        for _ in range(self.max_iter):
            for j in range(p):
                # 残差
                r = y - X @ self.beta + X[:, j] * self.beta[j]
                
                # 软阈值
                rho = X[:, j] @ r
                
                if rho > self.lambda_:
                    self.beta[j] = rho - self.lambda_
                elif rho < -self.lambda_:
                    self.beta[j] = rho + self.lambda_
                else:
                    self.beta[j] = 0
        
        return self
    
    def predict(self, X):
        X = (X - self.X_mean) / self.X_std
        return X @ self.beta + self.y_mean
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_lasso_path(coefs, alphas, save_path='lasso_path.png'):
    """LASSO路径"""
    plt.figure(figsize=(10, 6))
    for coef in coefs.T:
        plt.plot(alphas, coef)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient')
    plt.title('LASSO Path')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.show()


def plot_coefficients(beta, names, save_path='beta.png'):
    """系数柱状图"""
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(beta)), beta)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient')
    plt.title('LASSO Coefficients')
    plt.savefig(save_path)
    plt.show()
```

---

## 10. 评估

```python
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'n_nonzero': np.sum(beta != 0)
    }
```

---

## 11. 常见问题

### 11.1 特征相关
- 使用弹性网
- 组LASSO

### 11.2 alpha选择
- 交叉验证
- 信息准则(AIC/BIC)

---

## 12. 总结

### 核心要点
1. **L1正则**: 稀疏解
2. **软阈值**: 闭式更新
3. **坐标下降**: 逐坐标优化
4. **特征选择**: 自动置零

### 算法链
```
LASSO → 弹性网 → 组LASSO → Fused LASSO
```

---

## 13. 练习题

**习题1**: 软阈值公式

<details>
<summary>答案</summary>

$$S_\lambda(y) = \text{sign}(y)(|y| - \lambda)_+$$

</details>

**习题2**: LASSO vs 岭回归

<details>
<summary>答案</summary>

LASSO: L1正则 → 稀疏解
岭回归: L2正则 → 小系数

</details>

---

## 14. 学习路径

- **初级**: L1/L2理解,k折验证
- **中级**: LARS,弹性网
- **高级**: 组LASSO,GLM

### 推荐资源
- **论文**: Tibshirani "Regression Shrinkage and Selection" (1996)
- **书籍**: "Elements of Statistical Learning"

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import numpy as np

class LASSO回归Scratch:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr, self.n_iter, self.losses = lr, n_iter, []
    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.n_iter):
            err = X @ self.w + self.b - y
            self.losses.append(np.mean(err**2))
            self.w -= self.lr * (2/n) * X.T @ err
            self.b -= self.lr * (2/n) * np.sum(err)
        return self
    def predict(self, X): return X @ self.w + self.b

np.random.seed(42)
X = np.random.randn(200, 3)
y = 2*X[:,0] - X[:,1] + 0.5*X[:,2] + np.random.randn(200)*0.1
m = LASSO回归Scratch().fit(X, y)
print(f"Loss: {m.losses[-1]:.6f}")
```
