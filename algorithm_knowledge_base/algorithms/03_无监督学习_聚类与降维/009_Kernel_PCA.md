# Kernel PCA 核主成分分析 学习文档

> 非线性降维，将PCA扩展到核空间

---

## 1. 算法基础认知

### 1.1 一句话定义

Kernel PCA是PCA的非线性扩展，通过核函数将数据映射到高维特征空间后再做PCA，能够捕捉数据中的非线性结构。

### 1.2 直觉类比

Kernel PCA就像"升级版的PCA"。普通的PCA只能发现线性的"主方向"，但很多数据结构是弯曲的——就像一个S形曲线。Kernel PCA先用核函数（如高斯核）把数据"升维"到更高维空间，在那个空间里原本弯曲的结构就变成线性的了！

这就像把一根弯曲的吸管从侧面看是曲线，但从吸管的一端看进去——在那个二维切面上，它就是一条直线！

### 1.3 发展背景

- 1998年，Schölkopif等人在论文"Nonlinear Component Analysis as a Kernel Eigenvalue Problem"中提出
- 基于核方法的PCA扩展，解决非线性降维问题

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 降维 → 非线性PCA |
| 输出 | 主成分映射 |
| 方法 | 核技巧 |
| 范围 | 无监督 |

---

## 2. 核心原理

### 2.1 为什么需要非线性？

**线性PCA的局限**：普通的PCA只能发现线性结构。

```
举例：S形分布的数据
     ▲
     │    ●
     │  ●   ●
     │ ●  ●●
     │● ●●  ●
     │●●●   ●
     │ ●     ●
     └────────
     
这种数据用线性PCA效果很差，
但用高斯核Kernel PCA效果很好！
```

### 2.2 核函数原理

核函数的核心思想是"不用显式计算高维映射"：

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

其中 $\phi(x)$ 是映射函数，$K$ 是核函数。

### 2.3 常用核函数

| 核类型 | 公式 | 特点 |
|--------|------|------|
| 线性核 | $K(x, y) = x^T y$ | 退化为普通PCA |
| 多项式核 | $K(x, y) = (x^T y + c)^d$ | d阶多项式 |
| 高斯核(RBF) | $K(x, y) = \exp(-\frac{\|x-y\|^2}{2\sigma^2})$ | 局部映射 |
| sigmoid核 | $K(x, y) = \tanh(\alpha x^T y + c)$ | 类似神经网络 |

### 2.4 算法流程

```
Step 1: 输入数据 X ∈ R^(n×d)，n样本，d维
Step 2: 计算核矩阵 K，其中 K_ij = K(x_i, x_j)
Step 3: 中心化核矩阵：
        K_centered = K - 1_n K - K 1_n + 1_n K 1_n
        其中 1_n 是全1矩阵
Step 4: 求特征值分解：K_centered V = V Λ
Step 5: 取前k个特征向量 V_k
Step 6: 对新样本 x，计算投影：
        y = K(x, X) · V_k
```

---

## 3. 数学公式与推导

### 3.1 优化目标

在特征空间 $\mathcal{F}$ 中，PCA的优化目标是：

$$\max_v \frac{1}{n}\sum_{i=1}^n (v^T\phi(x_i) - \bar{\mu})^2$$

其中 $\bar{\mu} = \frac{1}{n}\sum_i v^T\phi(x_i)$。

### 3.2 对偶问题

通过核函数技巧，转化为：

$$\max_\alpha \quad \alpha^T K \alpha$$
$$\text{s.t.} \quad \|\alpha\| = 1$$

### 3.3 特征值问题

$$K \alpha = \lambda \alpha$$

求解特征值 $\lambda$ 和特征向量 $\alpha$。

### 3.4 投影公式

对于新样本 $x_{new}$，其在主成分上的投影为：

$$y = \sum_{i=1}^n \alpha_i K(x_{new}, x_i)$$

---

## 4. 训练过程讲解

### 4.1 训练步骤

```python
# Step 1: 计算核矩阵
K = compute_kernel(X, X, kernel_type='rbf', gamma=0.1)

# Step 2: 中心化
K_centered = center_kernel(K)

# Step 3: 特征值分解
eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

# Step 4: 取 top-k 主成分
idx = np.argsort(eigenvalues)[::-1][:k]
V = eigenvectors[:, idx]

# Step 5: 投影
X_new = K_test @ V
```

### 4.2 参数选择

| 参数 | 说明 | 建议 |
|------|------|------|
| kernel | 核类型 | rbf最常用 |
| gamma | RBF核带宽 | 1/(d·var(X)) |
| n_components | 降维维度 | 2-10 |

---

## 5. 应用场景

### 5.1 故障检测

Kernel PCA可以建立正常数据的"正常流形"，异常点会偏离这个流形，从而检测故障。

### 5.2 图像降维

人脸图像往往是非线性的，Kernel PCA能更好地捕捉人脸变化。

### 5.3 过程监控

化工、金融等过程监控，找出异常模式。

### 5.4 对比选择

| 场景 | 推荐方法 |
|------|----------|
| 线性结构 | PCA |
| 非线性结构 | Kernel PCA |
| 大数据 | t-SNE/UMAP |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 非线性处理 | 能捕捉复杂非线性结构 |
| 无显式映射 | 不需要计算高维 $\phi(x)$ |
| 泛化能力强 | 核函数可迁移到新数据 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 核选择敏感 | 不同核效果差异大 |
| 计算复杂度高 | O(n²) 核矩阵存储 |
| 可解释性差 | 难以理解映射后的空间 |

### 6.3 过拟合 vs 欠拟合

- **过拟合**：gamma太大 → 过拟合，记住训练数据
- **欠拟合**：gamma太小 → 欠拟合，信息丢失

---

## 7. 调库实现（Python + scikit-learn）

### 7.1 基本用法

```python
import numpy as np
from sklearn.decomposition import KernelPCA

# 创建模拟数据：非线性S形
np.random.seed(42)
t = np.linspace(0, 2*np.pi, 200)
X = np.column_stack([
    t + 0.1*np.random.randn(200),
    np.sin(t) + 0.1*np.random.randn(200)
])

# Kernel PCA 降维
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.5)
X_kpca = kpca.fit_transform(X)

print("原始形状:", X.shape)
print("降维后:", X_kpca.shape)
```

### 7.2 使用不同核

```python
# 线性核
kpca_linear = KernelPCA(n_components=2, kernel='linear')
X_linear = kpca_linear.fit_transform(X)

# 多项式核
kpca_poly = KernelPCA(n_components=2, kernel='poly', degree=3, coef0=1)
X_poly = kpca_poly.fit_transform(X)

# RBF核（高斯核）
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.5)
X_rbf = kpca_rbf.fit_transform(X)
```

### 7.3 参数调优

```python
from sklearn.model_selection import GridSearchCV

# 网格搜索最优gamma
param_grid = {'gamma': [0.01, 0.1, 0.5, 1, 2, 5]}
kpca = KernelPCA(n_components=2, kernel='rbf')
grid = GridSearchCV(kpca, param_grid, cv=5)
grid.fit(X)

print("最优gamma:", grid.best_params_)
```

---

## 8. 手工代码实现（核心算法手写）

```python
import numpy as np

class KernelPCA:
    """核主成分分析 - 手工实现"""
    
    def __init__(self, n_components=2, kernel='rbf', gamma=0.1):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.X_train = None
        self.lambdas = None
    
    def _compute_kernel(self, X1, X2):
        """计算核矩阵"""
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            # 高斯核
            pairwise_sq_dists = (
                np.sum(X1**2, axis=1, keepdims=True) + 
                np.sum(X2**2, axis=1) - 
                2 * (X1 @ X2.T)
            )
            return np.exp(-pairwise_sq_dists / (2 * self.gamma**2))
        elif self.kernel == 'poly':
            return (X1 @ X2.T + 1) ** 2
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _center_kernel(self, K):
        """中心化核矩阵"""
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    def fit(self, X):
        """训练"""
        self.X_train = X
        
        # 计算核矩阵
        K = self._compute_kernel(X, X)
        
        # 中心化
        K_centered = self._center_kernel(K)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        
        # 取前n_components个特征向量（倒序）
        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.lambdas = eigenvalues[idx]
        self.alpha = eigenvectors[:, idx]
        
        return self
    
    def transform(self, X):
        """投影到主成分"""
        K_test = self._compute_kernel(X, self.X_train)
        return K_test @ self.alpha
    
    def fit_transform(self, X):
        """训练+投影"""
        self.fit(X)
        return self.transform(self.X_train)
    
    def predict(self, X):
        """新样本投影"""
        return self.transform(X)


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    
    # 创建S形非线性数据
    t = np.linspace(0, 2*np.pi, 200)
    X = np.column_stack([
        t + 0.1*np.random.randn(200),
        np.sin(t) + 0.1*np.random.randn(200)
    ])
    
    # 手工实现
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.5)
    X_manual = kpca.fit_transform(X)
    
    # sklearn实现
    from sklearn.decomposition import KernelPCA
    kpca_sklearn = KernelPCA(n_components=2, kernel='rbf', gamma=0.5)
    X_sklearn = kpca_sklearn.fit_transform(X)
    
    print("手工实现形状:", X_manual.shape)
    print("sklearn形状:", X_sklearn.shape)
    print("前5个样本(手工):", X_manual[:5, :2])
    print("前5个样本(sklearn):", X_sklearn[:5, :2])
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

# 创建数据
np.random.seed(42)
t = np.linspace(0, 2*np.pi, 300)
X = np.column_stack([
    t + 0.15*np.random.randn(300),
    np.sin(t) + 0.15*np.random.randn(300)
])

# Kernel PCA 降维
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.3)
X_kpca = kpca.fit_transform(X)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 原始数据
axes[0].scatter(X[:, 0], X[:, 1], c=t, cmap='viridis', alpha=0.6)
axes[0].set_title('原始数据 (非线性S形)')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')

# Kernel PCA后
scatter = axes[1].scatter(X_kpca[:, 0], X_kpca[:, 1], c=t, cmap='viridis', alpha=0.6)
axes[1].set_title('Kernel PCA 降维后')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
plt.colorbar(scatter, ax=axes[1], label='t')

plt.tight_layout()
plt.savefig('kernel_pca_demo.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 核函数选择

| 核函数 | 适用场景 |
|--------|----------|
| linear | 线性可分数据 |
| poly | 多项式关系数据 |
| rbf | 通用非线性数据 |

### 10.2 gamma参数影响

```python
gammas = [0.01, 0.1, 0.5, 1, 2, 5]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, gamma in enumerate(gammas):
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
    X_kpca = kpca.fit_transform(X)
    
    axes[i].scatter(X_kpca[:, 0], X_kpca[:, 1], c=t, cmap='viridis', alpha=0.6)
    axes[i].set_title(f'gamma={gamma}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('kernel_pca_gamma.png', dpi=100)
plt.show()
```

**结论**：
- gamma太小 → 欠拟合，点聚在一起
- gamma太大 → 过拟合，点分散

---

## 11. 常见问题与易错点

### Q1: 如何选择核函数？

**答案**：先试RBF核（最常用），效果不好再试其他。

### Q2: 核矩阵太大怎么办？

**答��**：可用随机近似（RBFSampler）。

### Q3: 如何确定降维维度？

**答案**：看特征值谱，选择肘部位置。

### Q4: 和普通PCA的区别？

**答案**：普通PCA是线性的，Kernel PCA是非线性的。

### Q5: 为什么需要中心化？

**答案**：类比普通PCA，中心化使方差计算更准确。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核技巧 | 隐式高维映射 |
| 核函数 | RBF/Poly/Linear |
| 中心化 | 核矩阵的中心化 |
| 优缺点 | 非线性强但计算大 |

### 12.2 公式汇总

核函数：
$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

特征值问题：
$$K \alpha = \lambda \alpha$$

投影：
$$y = \sum_i \alpha_i K(x_{new}, x_i)$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. Kernel PCA 使用 RBF 核时，若 gamma 趋向无穷大，则：
   - A) 退化为线性核
   - B) 退化为多项式核
   - C) 每个点变成自己的簇

2. Kernel PCA 的时间复杂度是：
   - A) O(d)
   - B) O(n²)
   - C) O(n³)

### 13.2 简答题

1. 为什么 Kernel PCA 不需要显式计算映射函数 φ(x)？
2. 比较 Kernel PCA 和 t-SNE 的适用场景。

### 13.3 编程题

1. 用 Kernel PCA 对 sklearn 的 make_moons 数据集降维并可视化。
2. 实现基于 Nystroem 近似的 Kernel PCA。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
线性代数基础
    ↓
PCA原理
    ↓
核方法
    ↓
Kernel PCA
    ↓
其他核方法(SVM等)
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| PCA | 线性版本 |
| t-SNE | 非线性+概率 |
| UMAP | 更快非线降维 |
| Isomap | 流形学习 |

### 14.3 扩展阅读

- 论文：Schölkopf et al., 1998
- 书籍：《Pattern Classification》

---

## 附录

### 参考

1. Schölkopf, B., et al. (1998). Nonlinear Component Analysis as a Kernel Eigenvalue Problem. Neural Computation.
2. sklearn.decomposition.KernelPCA 文档

---

**文档结束**