# 线性代数基础（推荐系统方向）学习文档

## 1. 为什么推荐系统需要线性代数？

推荐系统本质上是在处理**高维向量**和**矩阵运算**：
- 用户和物品的嵌入向量
- 协同过滤的矩阵分解
- 深度学习的矩阵乘法

## 2. 向量基础

### 2.1 向量定义

向量是一个有序的数列，可以表示为列向量或行向量：

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

```python
import numpy as np

# 创建向量
v = np.array([1, 2, 3, 4, 5])
print(f"向量: {v}")
print(f"维度: {v.shape}")
print(f"向量范数: {np.linalg.norm(v)}")
```

### 2.2 向量运算

```python
import numpy as np

# 向量加法
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"加法: {v1 + v2}")

# 向量数乘
print(f"数乘: {2 * v1}")

# 内积（点积）
dot_product = np.dot(v1, v2)
print(f"内积: {dot_product}")

# 外积
outer_product = np.outer(v1, v2)
print(f"外积:\n{outer_product}")

# 余弦相似度
cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"余弦相似度: {cosine_sim:.4f}")
```

### 2.3 推荐系统中的向量

```python
# 用户嵌入向量
user_embedding = np.array([0.1, -0.3, 0.5, 0.2, -0.1])

# 物品嵌入向量
item_embedding = np.array([0.2, -0.1, 0.4, 0.3, 0.0])

# 计算用户对物品的偏好分数（点积）
score = np.dot(user_embedding, item_embedding)
print(f"偏好分数: {score:.4f}")

# 计算相似度
similarity = np.dot(user_embedding, item_embedding) / (
    np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
)
print(f"相似度: {similarity:.4f}")
```

## 3. 矩阵基础

### 3.1 矩阵定义

矩阵是二维数组：

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

```python
# 创建矩阵
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"矩阵:\n{A}")
print(f"形状: {A.shape}")

# 特殊矩阵
zeros = np.zeros((3, 3))        # 零矩阵
ones = np.ones((3, 3))          # 全1矩阵
identity = np.eye(3)            # 单位矩阵
random = np.random.randn(3, 3)  # 随机矩阵
```

### 3.2 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵加法
print(f"加法:\n{A + B}")

# 矩阵乘法
print(f"矩阵乘法:\n{A @ B}")  # 或 np.matmul(A, B)
print(f"逐元素乘法:\n{A * B}")

# 矩阵转置
print(f"转置:\n{A.T}")

# 矩阵求逆
A_inv = np.linalg.inv(A)
print(f"逆矩阵:\n{A_inv}")
print(f"验证 A @ A_inv:\n{A @ A_inv}")
```

### 3.3 推荐系统中的矩阵

```python
# 用户-物品交互矩阵
# 行: 用户, 列: 物品
R = np.array([
    [5, 3, 0, 1],  # 用户1
    [4, 0, 0, 1],  # 用户2
    [1, 1, 0, 5],  # 用户3
    [0, 0, 0, 4],  # 用户4
    [0, 1, 5, 4],  # 用户5
])

print("用户-物品交互矩阵:")
print(R)

# 矩阵分解: R ≈ U @ V^T
n_users, n_items = R.shape
n_factors = 2

# 用户矩阵 U (n_users × n_factors)
U = np.random.randn(n_users, n_factors)

# 物品矩阵 V (n_items × n_factors)
V = np.random.randn(n_items, n_factors)

# 预测
R_pred = U @ V.T
print(f"\n预测矩阵:\n{R_pred}")
```

## 4. 特征值与特征向量

### 4.1 定义

对于方阵 A，如果存在标量 λ 和非零向量 v 满足：

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

则 λ 是特征值，v 是对应的特征向量。

```python
# 计算特征值和特征向量
A = np.array([
    [2, 1],
    [1, 2]
])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证: A @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_v = eigenvalues[i] * v
    Av = A @ v
    print(f"\n验证 λ={eigenvalues[i]:.2f}:")
    print(f"A @ v = {Av}")
    print(f"λ * v = {lambda_v}")
```

### 4.2 在推荐中的应用

```python
# SVD 分解用于推荐
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 0, 4],
    [0, 1, 5, 4],
], dtype=float)

# SVD: R = U @ Σ @ V^T
U, sigma, Vt = np.linalg.svd(R, full_matrices=False)

print(f"U 形状: {U.shape}")      # 用户特征
print(f"Σ 形状: {sigma.shape}")  # 奇异值
print(f"V^T 形状: {Vt.shape}")   # 物品特征

# 使用前 k 个奇异值降维
k = 2
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]

# 重构
R_reconstructed = U_k @ sigma_k @ Vt_k
print(f"\n重构矩阵:\n{R_reconstructed.round(2)}")
```

## 5. 范数

### 5.1 向量范数

```python
v = np.array([3, 4])

# L1 范数（曼哈顿距离）
l1 = np.linalg.norm(v, ord=1)
print(f"L1 范数: {l1}")

# L2 范数（欧几里得距离）
l2 = np.linalg.norm(v, ord=2)
print(f"L2 范数: {l2}")

# 无穷范数
linf = np.linalg.norm(v, ord=np.inf)
print(f"无穷范数: {linf}")
```

### 5.2 矩阵范数

```python
A = np.array([[1, 2], [3, 4]])

# Frobenius 范数
frobenius = np.linalg.norm(A, 'fro')
print(f"Frobenius 范数: {frobenius}")

# 在正则化中的应用
def l2_regularization(W, lambda_reg):
    """L2 正则化"""
    return lambda_reg * np.sum(W ** 2)

def l1_regularization(W, lambda_reg):
    """L1 正则化"""
    return lambda_reg * np.sum(np.abs(W))
```

## 6. 常用矩阵分解

### 6.1 LU 分解

```python
from scipy.linalg import lu

A = np.array([
    [2, 1, 1],
    [4, 3, 3],
    [8, 7, 9]
])

P, L, U = lu(A)
print(f"P (置换矩阵):\n{P}")
print(f"L (下三角):\n{L}")
print(f"U (上三角):\n{U}")
print(f"验证 P @ L @ U:\n{P @ L @ U}")
```

### 6.2 QR 分解

```python
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

Q, R = np.linalg.qr(A)
print(f"Q (正交矩阵):\n{Q}")
print(f"R (上三角):\n{R}")
print(f"验证 Q @ R:\n{Q @ R}")
```

### 6.3 Cholesky 分解

```python
# Cholesky 分解要求正定矩阵
A = np.array([
    [4, 2, 2],
    [2, 5, 1],
    [2, 1, 6]
])

L = np.linalg.cholesky(A)
print(f"L (下三角):\n{L}")
print(f"验证 L @ L^T:\n{L @ L.T}")
```

## 7. 求解线性方程组

### 7.1 直接求解

```python
# Ax = b
A = np.array([
    [3, 1],
    [1, 2]
])
b = np.array([9, 8])

# 直接求解
x = np.linalg.solve(A, b)
print(f"解 x: {x}")
print(f"验证 A @ x: {A @ x}")
```

### 7.2 最小二乘法

```python
# 超定方程组（方程数 > 未知数）
A = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4]
])
b = np.array([6, 5, 7, 10])

# 最小二乘解
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print(f"最小二乘解: {x}")
print(f"残差: {residuals}")
```

## 8. 学习总结

### 8.1 核心要点

1. **向量**：推荐系统中的基本表示单元
2. **矩阵**：用户-物品交互的核心数据结构
3. **特征值分解**：理解矩阵性质的关键工具
4. **范数**：用于正则化和距离度量

### 8.2 推荐系统应用

| 线性代数概念 | 推荐系统应用 |
|------------|-------------|
| 向量点积 | 相似度计算 |
| 矩阵分解 | 协同过滤 |
| 特征值 | PCA 降维 |
| 范数 | 正则化 |
| SVD | 矩阵补全 |

## 9. 练习题

1. 实现一个完整的 SVD 推荐算法。

2. 比较不同范数对正则化的影响。

3. 使用 QR 分解求解最小二乘问题。
