# SVD（奇异值分解） 学习文档

## 1. 算法基础认知

### 1.1 什么是 SVD？

SVD（Singular Value Decomposition，奇异值分解）是线性代数中最重要、最优美的矩阵分解方法之一。它将任意矩阵分解为三个矩阵的乘积。

对于一个 m×n 的矩阵 R，SVD 将其分解为：

$$R = U \Sigma V^T$$

其中：
- **U**（m×m）：左奇异向量矩阵，正交矩阵
- **Σ**（m×n）：奇异值对角矩阵，对角元素非负且递减
- **V^T**（n×n）：右奇异向量矩阵的转置，正交矩阵

### 1.2 SVD 与推荐系统的关系

在推荐系统中：
- R 是用户-物品评分矩阵
- U 的每一行代表一个用户的特征
- V 的每一行代表一个物品的特征
- Σ 的对角元素表示各隐因子的重要性

### 1.3 为什么学习 SVD？

1. **理论基础**：矩阵分解方法的理论根基
2. **降维压缩**：可以用来压缩和降维
3. **去噪**：小奇异值对应噪声，可以截断
4. **数据填充**：可以用于预测缺失的评分

### 1.4 SVD vs 矩阵分解（MF）

| 特性 | 经典 SVD | 矩阵分解（MF） |
|------|----------|----------------|
| 分解形式 | R = UΣV^T | R ≈ PQ^T |
| 适用条件 | 要求矩阵完整 | 可处理稀疏矩阵 |
| 正交性 | U、V 正交 | P、Q 不一定正交 |
| 奇异值 | 显式计算奇异值 | 隐式学习 |
| 计算复杂度 | O(min(m³n, mn³)) | O(K × 非零元素数 × 迭代次数) |

## 2. 核心原理

### 2.1 数学定义

对于任意 m×n 实矩阵 R，存在分解：

$$R = U \Sigma V^T$$

其中：
- U = [u₁, u₂, ..., uₘ] 是 m×m 正交矩阵
- V = [v₁, v₂, ..., vₙ] 是 n×n 正交矩阵
- Σ 是 m×n 对角矩阵，对角元素 σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0

### 2.2 几何解释

SVD 可以理解为三个线性变换的组合：

1. **V^T**：旋转变换（在行空间）
2. **Σ**：缩放变换（沿坐标轴）
3. **U**：旋转变换（在列空间）

```
原始空间 → V^T 旋转 → Σ 缩放 → U 旋转 → 目标空间
```

### 2.3 与特征值分解的关系

对于方阵 A 的特征值分解：$A = Q \Lambda Q^{-1}$

SVD 与特征值分解的关系：
- $R^T R$ 的特征向量是 V 的列
- $R R^T$ 的特征向量是 U 的列
- 奇异值 σᵢ = √λᵢ（特征值的平方根）

### 2.4 截断 SVD（Truncated SVD）

只保留前 k 个最大的奇异值：

$$R \approx U_k \Sigma_k V_k^T$$

这是低秩近似，在推荐系统中用于降维。

## 3. 数学公式与推导

### 3.1 SVD 存在性证明（思路）

**定理**：对于任意 m×n 实矩阵 R，存在 SVD 分解。

**证明思路**：
1. 考虑 $R^T R$，这是一个 n×n 对称正定矩阵
2. 对称矩阵可以正交对角化：$R^T R = V \Lambda V^T$
3. 定义 Σ 的对角元素为 $\sigma_i = \sqrt{\lambda_i}$
4. 利用 $R v_i$ 构造 U 的列向量

### 3.2 Eckart-Young 定理

**定理**：截断 SVD 给出最佳低秩近似。

$$\min_{rank(A_k) = k} ||R - A_k||_F = ||R - U_k \Sigma_k V_k^T||_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

这说明 SVD 在 Frobenius 范数意义下是最优的低秩近似。

### 3.3 推荐系统中的预测公式

使用截断 SVD 预测用户 u 对物品 i 的评分：

$$\hat{r}_{ui} = \sum_{j=1}^{k} \sigma_j \cdot U_{uj} \cdot V_{ij}$$

或等价地：

$$\hat{r}_{ui} = (U_k \Sigma_k^{1/2})_u \cdot (V_k \Sigma_k^{1/2})_i^T$$

令 $P = U_k \Sigma_k^{1/2}$，$Q = V_k \Sigma_k^{1/2}$，则：

$$\hat{r}_{ui} = p_u \cdot q_i^T$$

这与一般的矩阵分解形式一致。

## 4. 训练过程讲解

### 4.1 完整 SVD 的计算

```python
import numpy as np

# 完整 SVD 分解
U, sigma, Vt = np.linalg.svd(R, full_matrices=False)

# 重构原矩阵
R_reconstructed = U @ np.diag(sigma) @ Vt
```

### 4.2 SVD 的计算复杂度

- **完整 SVD**：O(min(m³n, mn³))
- **截断 SVD**：O(mnk)，使用随机化算法可达 O(mn log k)

### 4.3 SVD 在推荐中的问题

**问题**：经典 SVD 要求矩阵是完整的，但推荐系统的评分矩阵非常稀疏。

**解决方案**：
1. **填充法**：用均值、中位数等填充缺失值
2. **迭代 SVD**：交替填充和分解
3. **FunkSVD**：只对已知评分建模（最常用）

## 5. 应用场景

### 5.1 直接应用

| 应用 | 说明 |
|------|------|
| 数据降维 | PCA 的核心算法 |
| 图像压缩 | 保留主要特征 |
| 噪声去除 | 截断小奇异值 |
| 推荐系统 | 用户/物品特征提取 |

### 5.2 推荐系统中的使用

```python
# 1. 填充缺失值（简单方法）
R_filled = np.nan_to_num(R, nan=np.nanmean(R))

# 2. 执行 SVD
U, sigma, Vt = np.linalg.svd(R_filled, full_matrices=False)

# 3. 截断
k = 20
U_k = U[:, :k]
sigma_k = sigma[:k]
Vt_k = Vt[:k, :]

# 4. 预测
R_pred = U_k @ np.diag(sigma_k) @ Vt_k
```

## 6. 优缺点分析

### 6.1 优点

1. **数学完美**：有严格的理论保证
2. **最优近似**：Eckart-Young 定理保证
3. **正交性**：U、V 正交，特征不相关
4. **奇异值排序**：自动给出特征重要性

### 6.2 缺点

1. **要求完整矩阵**：无法直接处理稀疏矩阵
2. **计算复杂度高**：O(min(m³n, mn³))
3. **不可解释**：奇异向量难以解释
4. **实时性差**：新数据需要重新计算

### 6.3 与 FunkSVD 对比

| 特性 | SVD | FunkSVD |
|------|-----|---------|
| 稀疏矩阵 | 需要填充 | 直接处理 |
| 计算复杂度 | 高 | 低（只处理非零元素） |
| 理论保证 | 最优近似 | 无严格保证 |
| 正交性 | 有 | 无 |
| 工业实用性 | 低 | 高 |

## 7. 调库实现

### 7.1 NumPy 实现

```python
import numpy as np

def svd_recommendation(R, k=20):
    """
    使用 SVD 进行推荐

    参数:
        R: 用户-物品评分矩阵
        k: 保留的奇异值数量

    返回:
        预测矩阵
    """
    # 填充缺失值（使用列均值）
    R_filled = R.copy()
    col_mean = np.nanmean(R_filled, axis=0)
    for i in range(R_filled.shape[1]):
        mask = np.isnan(R_filled[:, i])
        R_filled[mask, i] = col_mean[i]

    # SVD 分解
    U, sigma, Vt = np.linalg.svd(R_filled, full_matrices=False)

    # 截断
    U_k = U[:, :k]
    sigma_k = sigma[:k]
    Vt_k = Vt[:k, :]

    # 预测
    R_pred = U_k @ np.diag(sigma_k) @ Vt_k

    return R_pred, U_k, sigma_k, Vt_k


# 使用示例
if __name__ == "__main__":
    # 模拟评分矩阵（NaN 表示未评分）
    R = np.array([
        [5, 3, np.nan, 1],
        [4, np.nan, np.nan, 1],
        [1, 1, np.nan, 5],
        [np.nan, np.nan, 5, 4],
        [np.nan, 1, 5, 4],
    ])

    R_pred, U, sigma, Vt = svd_recommendation(R, k=2)

    print("原始矩阵:")
    print(R)
    print("\n预测矩阵:")
    print(R_pred.round(2))
```

### 7.2 SciPy 稀疏 SVD

```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def sparse_svd_recommendation(user_item_matrix, k=20):
    """
    使用稀疏 SVD 进行推荐

    参数:
        user_item_matrix: 稀疏的用户-物品矩阵
        k: 保留的奇异值数量

    返回:
        用户因子矩阵, 物品因子矩阵
    """
    # 转换为稀疏矩阵
    if not isinstance(user_item_matrix, csr_matrix):
        R = csr_matrix(user_item_matrix)
    else:
        R = user_item_matrix

    # 稀疏 SVD（只计算前 k 个奇异值）
    U, sigma, Vt = svds(R, k=k)

    # 注意：svds 返回的奇异值是升序的，需要反转
    idx = np.argsort(sigma)[::-1]
    U = U[:, idx]
    sigma = sigma[idx]
    Vt = Vt[idx, :]

    # 构建因子矩阵
    # 用户因子：U * sqrt(Sigma)
    # 物品因子：V * sqrt(Sigma)
    sqrt_sigma = np.sqrt(sigma)
    user_factors = U * sqrt_sigma
    item_factors = Vt.T * sqrt_sigma

    return user_factors, item_factors, sigma


# 使用示例
if __name__ == "__main__":
    # 稀疏矩阵
    R_sparse = csr_matrix([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [0, 0, 5, 4],
        [0, 1, 5, 4],
    ])

    user_factors, item_factors, sigma = sparse_svd_recommendation(R_sparse, k=2)

    print("奇异值:", sigma)
    print("\n用户因子矩阵 shape:", user_factors.shape)
    print("物品因子矩阵 shape:", item_factors.shape)

    # 预测
    predictions = user_factors @ item_factors.T
    print("\n预测矩阵:")
    print(predictions.round(2))
```

### 7.3 Scikit-learn TruncatedSVD

```python
from sklearn.decomposition import TruncatedSVD

def sklearn_svd_recommendation(R, n_components=20):
    """
    使用 sklearn 的 TruncatedSVD
    适用于稀疏矩阵
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    # fit_transform 返回 U * Sigma
    # 注意：这是对 R 的行（用户）进行降维
    user_features = svd.fit_transform(R)

    # 物品特征是 components_（V^T）
    item_features = svd.components_.T

    # 预测矩阵
    R_pred = user_features @ item_features.T

    return R_pred, user_features, item_features, svd


# 使用示例
if __name__ == "__main__":
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [0, 0, 5, 4],
        [0, 1, 5, 4],
    ])

    R_pred, user_features, item_features, svd = sklearn_svd_recommendation(R, n_components=2)

    print("解释方差比:", svd.explained_variance_ratio_)
    print("总解释方差:", sum(svd.explained_variance_ratio_))
```

## 8. 手工代码实现

### 8.1 完整的 SVD 实现（幂迭代法）

```python
import numpy as np

class SVDFromScratch:
    """
    从零实现 SVD（使用幂迭代法）
    """

    def __init__(self, n_components=None, n_iterations=100, tol=1e-6):
        """
        参数:
            n_components: 要计算的奇异值数量
            n_iterations: 幂迭代最大次数
            tol: 收敛阈值
        """
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.tol = tol

        self.U = None
        self.sigma = None
        self.Vt = None

    def _power_iteration(self, A, n_iterations, tol):
        """
        幂迭代法求最大特征值和特征向量

        参数:
            A: 对称矩阵
            n_iterations: 最大迭代次数
            tol: 收敛阈值

        返回:
            eigenvalue, eigenvector
        """
        n = A.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)

        for _ in range(n_iterations):
            v_new = A @ v
            v_new_norm = np.linalg.norm(v_new)
            v_new = v_new / v_new_norm

            # 检查收敛
            if np.abs(np.abs(v @ v_new) - 1) < tol:
                break

            v = v_new

        # 特征值
        eigenvalue = v.T @ A @ v

        return eigenvalue, v

    def fit(self, R):
        """
        对矩阵 R 进行 SVD 分解

        参数:
            R: m x n 矩阵

        返回:
            self
        """
        m, n = R.shape
        k = self.n_components or min(m, n)

        U_list = []
        sigma_list = []
        V_list = []

        R_deflated = R.copy()

        for i in range(k):
            # 计算 R^T R 和 R R^T
            RtR = R_deflated.T @ R_deflated
            RRt = R_deflated @ R_deflated.T

            # 幂迭代求最大特征值和特征向量
            # 方法1：从 R^T R 求 v_i
            _, v = self._power_iteration(RtR, self.n_iterations, self.tol)

            # 计算 u_i = R v_i / ||R v_i||
            Rv = R_deflated @ v
            sigma = np.linalg.norm(Rv)
            if sigma > 1e-10:
                u = Rv / sigma
            else:
                u = np.zeros(m)

            # 保存结果
            U_list.append(u)
            sigma_list.append(sigma)
            V_list.append(v)

            # 紧缩矩阵
            R_deflated = R_deflated - sigma * np.outer(u, v)

        self.U = np.column_stack(U_list)
        self.sigma = np.array(sigma_list)
        self.Vt = np.row_stack(V_list)

        return self

    def transform(self, R=None):
        """
        返回低维表示
        """
        if R is not None:
            return self.U @ np.diag(self.sigma)
        return self.U @ np.diag(self.sigma)

    def fit_transform(self, R):
        """
        拟合并转换
        """
        self.fit(R)
        return self.transform()

    def reconstruct(self, k=None):
        """
        重构矩阵

        参数:
            k: 使用前 k 个奇异值，None 表示使用全部
        """
        if k is None:
            k = len(self.sigma)

        U_k = self.U[:, :k]
        sigma_k = self.sigma[:k]
        Vt_k = self.Vt[:k, :]

        return U_k @ np.diag(sigma_k) @ Vt_k


# 使用示例
if __name__ == "__main__":
    # 测试矩阵
    R = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ], dtype=float)

    print("原始矩阵:")
    print(R)

    # 我们的实现
    svd_ours = SVDFromScratch(n_components=2)
    svd_ours.fit(R)

    print("\n奇异值:", svd_ours.sigma)
    print("\n重构矩阵 (k=2):")
    print(svd_ours.reconstruct(k=2).round(4))

    # 与 NumPy 对比
    U_np, sigma_np, Vt_np = np.linalg.svd(R, full_matrices=False)
    print("\nNumPy 奇异值:", sigma_np[:2])

    print("\n重构矩阵 (NumPy, k=2):")
    print((U_np[:, :2] @ np.diag(sigma_np[:2]) @ Vt_np[:2, :]).round(4))
```

### 8.2 带缺失值处理的迭代 SVD

```python
import numpy as np

class IterativeSVD:
    """
    迭代 SVD：处理缺失值的 SVD
    """

    def __init__(self, n_components=20, n_iterations=10, tol=1e-4):
        """
        参数:
            n_components: 隐因子数量
            n_iterations: 最大迭代次数
            tol: 收敛阈值
        """
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.tol = tol

        self.U = None
        self.sigma = None
        self.Vt = None

    def fit(self, R):
        """
        训练模型

        参数:
            R: 带缺失值的矩阵（NaN 表示缺失）
        """
        # 记录缺失值位置
        mask = ~np.isnan(R)

        # 初始化：用全局均值填充
        global_mean = np.nanmean(R)
        X = np.nan_to_num(R, nan=global_mean)

        for iteration in range(self.n_iterations):
            # SVD 分解
            U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

            # 截断
            U_k = U[:, :self.n_components]
            sigma_k = sigma[:self.n_components]
            Vt_k = Vt[:self.n_components, :]

            # 重构
            X_new = U_k @ np.diag(sigma_k) @ Vt_k

            # 只更新缺失值位置
            X_old = X.copy()
            X[~mask] = X_new[~mask]

            # 检查收敛
            diff = np.linalg.norm(X - X_old) / (np.linalg.norm(X_old) + 1e-10)
            if diff < self.tol:
                print(f"收敛于第 {iteration + 1} 次迭代")
                break

        self.U = U_k
        self.sigma = sigma_k
        self.Vt = Vt_k

        return self

    def predict(self, user_idx, item_idx):
        """预测单个评分"""
        return self.U[user_idx] @ np.diag(self.sigma) @ self.Vt[:, item_idx]

    def predict_all(self):
        """预测完整矩阵"""
        return self.U @ np.diag(self.sigma) @ self.Vt


# 使用示例
if __name__ == "__main__":
    # 带缺失值的评分矩阵
    R = np.array([
        [5, 3, np.nan, 1],
        [4, np.nan, np.nan, 1],
        [1, 1, np.nan, 5],
        [np.nan, np.nan, 5, 4],
        [np.nan, 1, 5, 4],
    ])

    print("原始矩阵（NaN 表示未评分）:")
    print(R)

    # 训练
    model = IterativeSVD(n_components=2, n_iterations=20)
    model.fit(R)

    # 预测
    R_pred = model.predict_all()
    print("\n预测矩阵:")
    print(R_pred.round(2))

    # 查看填充的值
    mask = np.isnan(R)
    print("\n预测的缺失值:")
    for i, j in zip(*np.where(mask)):
        print(f"  用户 {i} 对物品 {j} 的预测评分: {R_pred[i, j]:.2f}")
```

## 9. 可视化与结果理解

### 9.1 奇异值衰减可视化

```python
import matplotlib.pyplot as plt

def visualize_singular_values(sigma):
    """
    可视化奇异值的衰减
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 奇异值分布
    axes[0].bar(range(len(sigma)), sigma)
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Singular Value')
    axes[0].set_title('Singular Values Distribution')

    # 累积方差贡献
    variance_explained = sigma ** 2 / np.sum(sigma ** 2)
    cumulative_variance = np.cumsum(variance_explained)

    axes[1].plot(range(len(cumulative_variance)), cumulative_variance, 'b-o')
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印信息
    print(f"前 10 个奇异值: {sigma[:10]}")
    print(f"90% 方差需要的组件数: {np.argmax(cumulative_variance >= 0.9) + 1}")
```

### 9.2 用户/物品嵌入可视化

```python
from sklearn.decomposition import PCA

def visualize_embeddings(U, V, n_users_to_show=20, n_items_to_show=20):
    """
    可视化用户和物品的嵌入
    """
    # 如果维度大于2，使用 PCA 降维
    if U.shape[1] > 2:
        pca = PCA(n_components=2)
        U_2d = pca.fit_transform(U)
        V_2d = pca.transform(V)
    else:
        U_2d = U
        V_2d = V

    plt.figure(figsize=(12, 10))

    # 绘制用户
    plt.scatter(U_2d[:n_users_to_show, 0], U_2d[:n_users_to_show, 1],
                c='blue', alpha=0.6, s=100, label='Users', marker='o')

    # 绘制物品
    plt.scatter(V_2d[:n_items_to_show, 0], V_2d[:n_items_to_show, 1],
                c='red', alpha=0.6, s=100, label='Items', marker='^')

    # 添加标签
    for i in range(min(n_users_to_show, len(U_2d))):
        plt.annotate(f'U{i}', (U_2d[i, 0], U_2d[i, 1]))

    for i in range(min(n_items_to_show, len(V_2d))):
        plt.annotate(f'I{i}', (V_2d[i, 0], V_2d[i, 1]))

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('User and Item Embeddings in 2D Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### 9.3 重构误差可视化

```python
def visualize_reconstruction_error(R, max_k=50):
    """
    可视化不同 k 值下的重构误差
    """
    # 先填充缺失值
    global_mean = np.nanmean(R)
    R_filled = np.nan_to_num(R, nan=global_mean)

    errors = []
    k_values = range(1, min(max_k, min(R.shape)) + 1)

    for k in k_values:
        U, sigma, Vt = np.linalg.svd(R_filled, full_matrices=False)
        R_reconstructed = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]

        # 只计算非缺失位置的误差
        mask = ~np.isnan(R)
        error = np.sqrt(np.mean((R[mask] - R_reconstructed[mask]) ** 2))
        errors.append(error)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, errors, 'b-o')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('RMSE')
    plt.title('Reconstruction Error vs Number of Components')
    plt.grid(True, alpha=0.3)

    # 标记最优点
    optimal_k = k_values[np.argmin(errors)]
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.legend()

    plt.show()

    return optimal_k
```

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

def evaluate_svd(R, k=20, n_folds=5):
    """
    交叉验证评估 SVD 模型

    参数:
        R: 评分矩阵
        k: 隐因子数量
        n_folds: 交叉验证折数

    返回:
        评估指标
    """
    # 获取非缺失值的索引
    rows, cols = np.where(~np.isnan(R))
    ratings = R[rows, cols]

    # 打乱数据
    indices = np.random.permutation(len(ratings))
    rows, cols, ratings = rows[indices], cols[indices], ratings[indices]

    kf = KFold(n_splits=n_folds)
    rmse_scores = []
    mae_scores = []

    for train_idx, test_idx in kf.split(ratings):
        # 构建训练矩阵
        R_train = np.full_like(R, np.nan)
        R_train[rows[train_idx], cols[train_idx]] = ratings[train_idx]

        # 训练
        model = IterativeSVD(n_components=k, n_iterations=20)
        model.fit(R_train)

        # 预测
        R_pred = model.predict_all()

        # 评估
        actual = ratings[test_idx]
        predicted = R_pred[rows[test_idx], cols[test_idx]]

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

    return {
        'RMSE': np.mean(rmse_scores),
        'RMSE_std': np.std(rmse_scores),
        'MAE': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores)
    }
```

## 11. 常见问题与易错点

### 11.1 常见问题

**Q1：SVD 和 PCA 有什么关系？**

A：PCA 本质上就是对中心化后的数据进行 SVD。PCA 的主成分就是 SVD 的右奇异向量。

**Q2：为什么推荐系统不用经典 SVD？**

A：因为推荐系统的评分矩阵非常稀疏，经典 SVD 要求完整矩阵。实际使用的是 FunkSVD（只对已知评分建模）。

**Q3：奇异值大小代表什么？**

A：奇异值大小表示对应隐因子的"重要性"或"信息量"。大的奇异值对应主要的模式，小的奇异值可能是噪声。

### 11.2 易错点

1. **混淆 SVD 和 FunkSVD**：SVD 要求完整矩阵，FunkSVD 只对已知评分建模
2. **忘记奇异值排序**：不同库返回的奇异值顺序可能不同
3. **截断位置错误**：注意是 U 的列、Σ 的对角元素、V^T 的行
4. **填充方法不当**：用 0 填充可能引入偏差
5. **维度混淆**：U 是 m×m，V^T 是 n×n，Σ 是 m×n

## 12. 学习总结

### 12.1 核心要点

1. **SVD 定义**：R = UΣV^T，三个矩阵的乘积
2. **正交性**：U 和 V 是正交矩阵
3. **奇异值**：对角元素递减，表示特征重要性
4. **截断近似**：保留前 k 个奇异值实现低秩近似
5. **最优性**：Eckart-Young 定理保证截断 SVD 是最优低秩近似

### 12.2 SVD 家族

```
矩阵分解方法
├── 经典 SVD
│   ├── 完整 SVD
│   ├── 截断 SVD
│   └── 稀疏 SVD
├── FunkSVD
│   ├── 基本形式
│   └── 带偏置
├── BiasSVD
│   └── 加入用户/物品偏置
└── SVD++
    └── 考虑隐式反馈
```

## 13. 练习题与思考题

### 13.1 基础题

1. **（填空）** SVD 将矩阵 R 分解为三个矩阵的乘积：R = ______。

2. **（判断）** SVD 中的 U 和 V 矩阵都是正交矩阵。（ ）

3. **（简答）** 奇异值从大到小排列有什么意义？

### 13.2 进阶题

4. **（推导）** 证明：如果 R = UΣV^T 是 SVD 分解，那么 R^T R = VΣ^2V^T。

5. **（编程）** 实现一个函数，计算保留前 k 个奇异值后的重构误差。

6. **（分析）** 比较 SVD 和 FunkSVD 在处理稀疏矩阵时的区别。

### 13.3 思考题

7. 为什么 SVD 在推荐系统中逐渐被 FunkSVD 等方法取代？

8. 如何选择合适的截断 k 值？有哪些方法？

9. SVD 与深度学习中的 Embedding 有什么联系？

### 参考答案

1. U Σ V^T

2. 正确

3. 奇异值从大到小排列，使得我们可以通过截断来获得最优的低秩近似。大的奇异值对应主要的模式和信息。

4. R^T R = (UΣV^T)^T(UΣV^T) = VΣ^T U^T U Σ V^T = VΣ^2 V^T（因为 U^T U = I）

5. 提示：计算 ||R - U_k Σ_k V_k^T||_F

6. SVD 需要完整矩阵，需填充缺失值；FunkSVD 直接对已知评分建模，更自然地处理稀疏性。

## 14. 学习路径建议

### 14.1 前置知识

- [ ] 线性代数（矩阵运算、特征值）
- [ ] NumPy 基础
- [ ] 优化基础

### 14.2 学习顺序

1. **理论基础** → 学习 SVD 的数学定义和性质
2. **动手实现** → 用 NumPy 实现完整 SVD
3. **理解截断** → 实验不同 k 值的效果
4. **学习变体** → 学习 FunkSVD 等实用变体
5. **工程实践** → 学习大规模 SVD 的计算方法

### 14.3 推荐资源

- **教材**：《线性代数导论》- SVD 章节
- **视频**：3Blue1Brown - 线性代数的本质
- **论文**：Matrix Factorization Techniques for Recommender Systems
- **代码**：SciPy, scikit-learn 源码

### 14.4 下一步学习

完成 SVD 学习后，建议继续学习：
- **FunkSVD**：实用的矩阵分解方法
- **BiasSVD**：带偏置的扩展
- **SVD++**：考虑隐式反馈的扩展
- **深度学习推荐**：Neural CF 等
