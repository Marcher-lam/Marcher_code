# LASSO回归 学习文档

## 1. 算法基础认知

LASSO（Least Absolute Shrinkage and Selection Operator）回归是一种带 L1 正则化的线性回归方法。与岭回归的 L2 正则化不同，LASSO 使用权重绝对值之和作为惩罚项，能够将部分权重压缩到精确的 0，从而实现自动特征选择。

## 2. 核心原理

LASSO 在损失函数中加入 L1 惩罚项 $\lambda\|w\|_1 = \lambda\sum|w_j|$。从几何角度看，L1 约束区域是菱形（而非 L2 的圆形），菱形的顶点在坐标轴上，等高线更容易在顶点处与约束区域相切，使得部分权重恰好为 0。

这就是 LASSO 能产生稀疏解的根本原因——L1 范数的"棱角"使得最优解倾向于落在坐标轴上。

## 3. 数学公式与推导

**损失函数：**

$$J(w) = \frac{1}{2m}\sum_{i=1}^{m}(y_i - w^Tx_i)^2 + \lambda\sum_{j=1}^{n}|w_j|$$

**等价约束形式：**

$$\min_w \frac{1}{2m}\|y - Xw\|^2 \quad \text{s.t.} \quad \sum_{j=1}^{n}|w_j| \leq t$$

**为什么没有解析解？**

L1 范数在 $w_j = 0$ 处不可导，因此无法像岭回归那样直接求导令其为零得到闭式解。LASSO 通常使用坐标下降法求解。

**坐标下降法：**

固定其他所有 $w_k$（$k \neq j$），只对 $w_j$ 优化：

$$w_j := S\left(\frac{1}{m}\sum_{i=1}^{m}x_{ij}(y_i - \sum_{k\neq j}w_kx_{ik}), \lambda\right) / \left(\frac{1}{m}\sum_{i=1}^{m}x_{ij}^2\right)$$

其中 $S(z, \lambda)$ 为软阈值函数：

$$S(z, \lambda) = \text{sign}(z) \cdot \max(|z| - \lambda, 0)$$

## 4. 训练过程讲解

1. **数据标准化**：LASSO 对特征尺度敏感，必须标准化
2. **初始化权重**：全部设为 0
3. **坐标下降迭代**：对每个特征 $j$，固定其余权重，利用软阈值函数更新 $w_j$
4. **收敛判断**：权重变化小于阈值或达到最大迭代次数时停止
5. **选择 $\lambda$**：通常通过交叉验证选择最优正则化参数

## 5. 应用场景

- 高维数据的特征选择（如基因数据、文本特征）
- 广告系统中的稀疏特征建模
- 信号处理中的压缩感知
- 需要模型可解释性的场景（只保留关键特征）

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 自动特征选择，产生稀疏解 | 最多选择 $n$ 个特征（样本数限制） |
| 模型可解释性强 | 特征间高度相关时选择不稳定 |
| 减少过拟合 | $\lambda$ 较大时可能欠拟合 |
| 计算效率较高（坐标下降） | 没有闭式解，需要迭代求解 |

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=30, n_informative=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_cv = LassoCV(alphas=np.logspace(-3, 2, 50), cv=5)
lasso_cv.fit(X_train_scaled, y_train)

y_pred = lasso_cv.predict(X_test_scaled)

print(f"最优alpha: {lasso_cv.alpha_:.4f}")
print(f"非零权重数: {np.sum(lasso_cv.coef_ != 0)}")
print(f"总特征数: {X.shape[1]}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class LassoManual:
    def __init__(self, alpha=1.0, n_iters=1000, tol=1e-6):
        self.alpha = alpha
        self.n_iters = n_iters
        self.tol = tol
        self.weights = None
        self.bias = None

    def _soft_threshold(self, rho, lam):
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        else:
            return 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.mean(y)

        for _ in range(self.n_iters):
            weights_old = self.weights.copy()

            for j in range(n_features):
                residual = y - np.dot(X, self.weights) - self.bias + self.weights[j] * X[:, j]
                rho = np.dot(X[:, j], residual) / n_samples
                z = np.sum(X[:, j] ** 2) / n_samples
                self.weights[j] = self._soft_threshold(rho, self.alpha) / z

            self.bias = np.mean(y - np.dot(X, self.weights))

            if np.max(np.abs(self.weights - weights_old)) < self.tol:
                break

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

alphas = np.logspace(-3, 2, 50)
coefs = []
nonzero_counts = []

for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coefs.append(lasso.coef_)
    nonzero_counts.append(np.sum(lasso.coef_ != 0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(alphas, coefs)
ax1.set_xscale('log')
ax1.set_xlabel('alpha')
ax1.set_ylabel('权重系数')
ax1.set_title('LASSO: 权重路径')

ax2.plot(alphas, nonzero_counts)
ax2.set_xscale('log')
ax2.set_xlabel('alpha')
ax2.set_ylabel('非零特征数')
ax2.set_title('LASSO: 特征数量变化')

plt.tight_layout()
plt.savefig("lasso_path.png", dpi=150)
plt.show()
```

随着 $\alpha$ 增大，权重逐个变为精确的 0，这就是 L1 正则化的稀疏性。

## 10. 模型评估

除了常规回归指标（MSE、R²），还应关注：
- **非零权重数量**：衡量模型的稀疏程度
- **交叉验证误差曲线**：帮助选择最优 $\alpha$
- **被选中的特征**：检查是否符合业务直觉

## 11. 常见问题与易错点

- **忘记标准化**：不同尺度的特征受到不均匀的正则化
- **$\alpha$ 过大导致欠拟合**：重要特征也被压缩为 0
- **相关特征的选择不稳定**：高度相关的特征中 LASSO 可能随机选一个
- **迭代不收敛**：增大 `max_iter` 或调整 `tol`
- **样本数少于特征数时**：LASSO 最多选 $n$ 个特征，可考虑 ElasticNet

## 12. 学习总结

LASSO 通过 L1 正则化实现稀疏解和自动特征选择，核心算法是坐标下降法配合软阈值函数。与岭回归相比，LASSO 的优势在于特征选择能力，劣势是对相关特征不稳定。实际中常使用 LassoCV 自动选择正则化参数。

## 13. 练习题与思考题（含答案）

**Q1**: 为什么 L1 正则化能产生稀疏解而 L2 不能？

> A: L1 约束区域是菱形（有棱角），顶点在坐标轴上。等高线与菱形更容易在顶点相切，使部分权重为 0。L2 约束区域是光滑圆形，相切点很少在坐标轴上。

**Q2**: 当特征数大于样本数时，LASSO 有什么限制？

> A: LASSO 最多选择 $\min(n, p)$ 个非零特征。当 $p > n$ 时，最多保留 $n$ 个特征。

**Q3**: 软阈值函数 $S(z, \lambda)$ 的物理含义是什么？

> A: 它将绝对值小于 $\lambda$ 的值压缩为 0，大于 $\lambda$ 的值向 0 方向缩减 $\lambda$，这正是产生稀疏性的关键操作。

## 14. 学习路径建议

```
线性回归 → 岭回归 → LASSO回归 → ElasticNet → 稀疏学习理论
```

建议对比岭回归和 LASSO 的正则化路径图，直观理解 L1/L2 的差异，再学习 ElasticNet（L1+L2 混合正则化）。
