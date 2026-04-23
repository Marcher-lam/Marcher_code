# 岭回归与LASSO回归

> 正则化线性回归：防止过拟合的核心技术

---

## 1. 算法基础认知

### 1.1 什么是岭回归

**岭回归（Ridge Regression）是在线性回归的基础上添加L2正则化的回归方法。**

```
普通线性回归目标：
min Σ(yᵢ - ŷᵢ)²

岭回归目标：
min Σ(yᵢ - ŷᵢ)² + λ Σwⱼ²
                  ↑↑↑
            L2正则化项
```

### 1.2 什么是LASSO回归

**LASSO回归（Least Absolute Shrinkage and Selection Operator）是在线性回归的基础上添加L1正则化的回归方法。**

```
LASSO目标：
min Σ(yᵢ - ŷᵢ)² + λ Σ|wⱼ|
                ↑↑↑
            L1正则化项
```

### 1.3 为什么需要正则化

```
过拟合问题：
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   过拟合：模型记住了训练数据的噪声                          │
│                                                            │
│   数据点  *   *                                            │
│              *                                              │
│         *      *                                            │
│              *        *                                     │
│         *            *    *                                │
│              *            *                                │
│                                                            │
│   理想：找到一条"刚刚好"的线                               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 2. 核心原理

### 2.1 岭回归（L2正则化）

```
岭回归的损失函数：
J(w) = ||Xw - y||² + λ||w||²

解析解：
w = (XᵀX + λI)⁻¹Xᵀy

关键点：
- λI 是一个对角矩阵，确保 XᵀX + λI 满秩
- 所有特征权重都会缩小，但不会变为零
- λ 越大，权重越小
```

### 2.2 LASSO回归（L1正则化）

```
LASSO的损失函数：
J(w) = ||Xw - y||² + λ||w||₁

关键特性：
- L1正则化会导致稀疏解
- 许多权重会变为精确的零
- 本质上是特征选择
```

### 2.3 L1 vs L2 正则化对比

```
┌────────────────────────────────────────────────────────────┐
│                  L1 vs L2 正则化                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   L1 (LASSO):           L2 (Ridge):                       │
│   ┌──────────┐          ┌──────────┐                      │
│   │ *        │          │     *    │                      │
│   │   *      │          │   *   *  │                      │
│   │     *    │          │ *     *  │                      │
│   │       *  │          │ *     *  │                      │
│   └──────────┘          └──────────┘                      │
│                                                            │
│   产生稀疏解           权重均匀缩小                        │
│   自动特征选择         保留所有特征                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 岭回归推导

```
目标函数：
J(w) = (Xw - y)ᵀ(Xw - y) + λwᵀw

对w求导并令导数为零：
∂J/∂w = 2Xᵀ(Xw - y) + 2λw = 0

整理：
XᵀXw + λIw = Xᵀy
(XᵀX + λI)w = Xᵀy

解得：
w = (XᵀX + λI)⁻¹Xᵀy
```

### 3.2 LASSO的特性

```
LASSO的优化问题是凸的，但不是光滑的（不可导）

由于L1范数在0处有"角"，最优解经常落在坐标轴上
这使得LASSO能够进行特征选择

几何解释：
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   约束条件 ||w||₁ ≤ t  是一个菱形区域                      │
│                                                            │
│   Loss的等高线与约束区域的交点                              │
│   菱形的"角"更容易与等高线相交                             │
│   而"角"正好对应某个权重为0                                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.3 ElasticNet（弹性网络）

```
结合L1和L2正则化：
J(w) = ||Xw - y||² + λ₁||w||₁ + λ₂||w||²

同时获得L1和L2的优点：
- 稀疏性（L1）
- 稳定性（L2）
```

---

## 4. 应用场景

### 4.1 岭回归应用

| 场景 | 说明 |
|-----|------|
| **多重共线性** | 特征之间高度相关时 |
| **特征数量多** | 特征数量大于样本数量 |
| **防止过拟合** | 当简单线性回归过拟合时 |

### 4.2 LASSO应用

| 场景 | 说明 |
|-----|------|
| **特征选择** | 确定哪些特征重要 |
| **稀疏建模** | 需要稀疏解 |
| **可解释性** | 需要简化的模型 |

---

## 5. 优缺点分析

### 5.1 岭回归

| 优点 | 缺点 |
|-----|------|
| 解决多重共线性 | 不进行特征选择 |
| 数值稳定 | 超参数需要调优 |
| 防止过拟合 | 不产生稀疏解 |

### 5.2 LASSO

| 优点 | 缺点 |
|-----|------|
| 自动特征选择 | 特征选择不稳定 |
| 产生稀疏解 | 处理多重共线性不如岭回归 |
| 可解释性强 | 可能过度稀疏化 |

---

## 6. 调库实现

### 6.1 岭回归实现

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# ==================== 1. 生成数据 ====================
np.random.seed(42)

# 生成有噪声的多重共线性数据
n_samples = 100
n_features = 10

# 创建高度相关的特征
X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=5,  # 只有5个特征真正有用
    noise=10,
    coef=True,
    random_state=42
)

# 添加噪声特征（高度相关）
X[:, 5:] = np.random.randn(n_samples, n_features - 5) * 0.5
X[:, 5:] += X[:, :5] * 0.8  # 与有用特征高度相关

# ==================== 2. 划分数据 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征标准化（重要！）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 3. 比较不同alpha值 ====================
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

print("=" * 60)
print("不同alpha值的岭回归效果")
print("=" * 60)

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    train_score = ridge.score(X_train_scaled, y_train)
    test_score = ridge.score(X_test_scaled, y_test)
    
    # 计算权重范数
    weight_norm = np.linalg.norm(ridge.coef_)
    
    print(f"alpha={alpha:6.3f} | 训练R²={train_score:.4f} | 测试R²={test_score:.4f} | 权重范数={weight_norm:.4f}")

# ==================== 4. 交叉验证选择最优alpha ====================
from sklearn.linear_model import RidgeCV

# 自动选择最优alpha
alphas_to_try = np.logspace(-4, 4, 50)
ridge_cv = RidgeCV(alphas=alphas_to_try, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"\n最优alpha: {ridge_cv.alpha_:.6f}")
print(f"对应的R²: {ridge_cv.score(X_test_scaled, y_test):.4f}")

# ==================== 5. 可视化权重衰减 ====================
def plot_ridge_coefficients(X, y, alphas):
    """可视化不同alpha下权重系数的变���"""
    coefs = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)
    
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlabel('alpha')
    ax.set_ylabel('权重系数')
    ax.set_title('岭回归：权重系数随alpha变化')
    ax.grid(True, alpha=0.3)
    plt.show()

plot_ridge_coefficients(X_train_scaled, y_train, np.logspace(-4, 4, 50))
```

### 6.2 LASSO实现

```python
from sklearn.linear_model import LassoCV

# ==================== LASSO特征选择 ====================
print("=" * 60)
print("LASSO特征选择示例")
print("=" * 60)

# 使用LASSO进行特征选择
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# 查看哪些特征被保留
print("\n各特征的权重系数：")
for i, coef in enumerate(lasso.coef_):
    status = "保留" if abs(coef) > 0.01 else "剔除"
    print(f"  特征{i+1}: {coef:8.4f} ({status})")

# 保留的特征数量
n_features_kept = np.sum(np.abs(lasso.coef_) > 0.01)
print(f"\nLASSO保留了 {n_features_kept}/{n_features} 个特征")

# ==================== ElasticNet实现 ====================
print("\n" + "=" * 60)
print("ElasticNet（结合L1和L2）")
print("=" * 60)

# l1_ratio: L1占比
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)

print(f"测试R²: {elastic.score(X_test_scaled, y_test):.4f}")
print(f"保留特征数: {np.sum(np.abs(elastic.coef_) > 0.01)}")
```

---

## 7. 手工代码实现

### 7.1 岭回归手工实现

```python
import numpy as np

class RidgeRegressionManual:
    """
    手工实现的岭回归
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        训练岭回归模型
        
        使用正规方程：
        w = (XᵀX + λI)⁻¹Xᵀy
        """
        m, n = X.shape
        
        # 添加偏置列（方法1：使用增广矩阵）
        # X_b = np.c_[np.ones((m, 1)), X]
        # I = np.eye(n + 1)
        # I[0, 0] = 0  # 不对偏置正则化
        # self.weights = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        
        # 方法2：分开计算
        X_centered = X - np.mean(X, axis=0)
        y_centered = y - np.mean(y)
        
        # 岭回归解析解
        I = np.eye(n)
        self.weights = np.linalg.inv(X_centered.T @ X_centered + self.alpha * I) @ X_centered.T @ y_centered
        self.bias = np.mean(y) - np.mean(X, axis=0) @ self.weights
        
        return self
    
    def predict(self, X):
        """预测"""
        return X @ self.weights + self.bias


class LassoRegressionManual:
    """
    手工实现的LASSO回归
    使用坐标下降法
    """
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
    
    def _soft_threshold(self, x, threshold):
        """软阈值函数"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        """使用坐标下降法求解"""
        m, n = X.shape
        
        # 标准化
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)
        
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean
        
        # 初始化权重
        self.weights = np.zeros(n)
        
        # 坐标下降
        for iteration in range(self.max_iter):
            old_weights = self.weights.copy()
            
            for j in range(n):
                # 计算剩余（不含当前特征j的预测）
                residual = y_centered - X_scaled @ self.weights + X_scaled[:, j] * self.weights[j]
                
                # 更新权重（软阈值）
                rho_j = X_scaled[:, j] @ residual
                self.weights[j] = self._soft_threshold(rho_j, self.alpha * m) / (np.sum(X_scaled[:, j] ** 2) + 1e-8)
            
            # 检查收敛
            if np.max(np.abs(self.weights - old_weights)) < self.tol:
                break
        
        # 计算偏置
        self.bias = self.y_mean - np.sum(self.weights * self.X_mean / (self.X_std + 1e-8))
        
        return self
    
    def predict(self, X):
        """预测"""
        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)
        return X_scaled @ self.weights + self.bias


# ==================== 测试 ====================
np.random.seed(42)

# 生成数据
X, y = make_regression(n_samples=100, n_features=10, noise=10, random_state=42)

# 划分
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练岭回归
ridge = RidgeRegressionManual(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print(f"岭回归测试R²: {ridge.score(X_test_scaled, y_test):.4f}")

# 训练LASSO
lasso = LassoRegressionManual(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
print(f"LASSO测试R²: {lasso.predict(X_test_scaled) @ (y_test - y_test.mean()) / np.std(y_test):.4f}")
```

---

## 8. 可视化与结果理解

### 8.1 正则化路径

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_regularization_path():
    """绘制正则化路径"""
    from sklearn.linear_model import lasso_path
    
    # 计算LASSO路径
    alphas_lasso, coefs_lasso, _ = lasso_path(X_train_scaled, y_train, eps=0.001, n_alphas=100)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制系数路径
    for i in range(coefs_lasso.shape[0]):
        plt.plot(alphas_lasso, coefs_lasso[i], label=f'特征{i+1}', alpha=0.7)
    
    plt.xlabel('Alpha')
    plt.ylabel('权重系数')
    plt.title('LASSO正则化路径：权重随alpha变化')
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.show()

plot_regularization_path()
```

---

## 9. 模型评估

| 指标 | 说明 |
|-----|------|
| **R²** | 决定系数，衡量模型拟合程度 |
| **MSE** | 均方误差 |
| **非零权重数量** | LASSO特有，反映特征选择结果 |
| **权重分布** | 岭回归权重衰减情况 |

---

## 10. 常见问题与易错点

### 10.1 特征未标准化

```python
# ❌ 错误：不标准化
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)  # 结果不稳定

# ✅ 正确：标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ridge.fit(X_scaled, y)
```

### 10.2 alpha选择不当

```python
# alpha太小：过拟合
ridge = Ridge(alpha=0.0001)

# alpha太大：欠拟合
ridge = Ridge(alpha=10000)

# ✅ 建议：使用交叉验证选择
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
ridge_cv.fit(X, y)
```

---

## 11. 学习总结

### 11.1 核心要点

1. **岭回归** = 线性回归 + L2正则化
2. **LASSO** = 线性回归 + L1正则化
3. **正则化** 防止过拟合
4. **L1** 会产生稀疏解（特征选择）
5. **L2** 权重均匀缩小

### 11.2 记忆口诀

```
岭回归L2，收缩不剔除
LASSO用L1，稀疏解最美
特征要标准，否则白费劲
调参用CV，效果才能行
```

---

## 12. 练习题

1. 岭回归和LASSO回归的区别是什么？
2. 为什么正则化需要特征标准化？
3. 什么场景下应该使用LASSO而不是岭回归？

---

## 13. 学习路径建议

```
线性回归 → 岭回归/LASSO → 逻辑回归 → FM → DeepFM
```

**下一章**：[28_KNN](./28_KMeans.md) - K近邻算法