# LLE（局部线性嵌入）学习文档

> 流形学习经典算法，通过保持局部邻域关系进行非线性降维

---

## 1. 算法基础认知

**一句话定义**：LLE（Locally Linear Embedding，局部线性嵌入）是一种无监督的流形学习算法，由Roweis和Saul于2000年提出。它通过保持数据在原始空间中的局部线性关系，将高维数据映射到低维空间，常用于非线性降维、数据可视化和特征提取。

**直觉类比**：LLE就像一张揉皱的纸重新展开。想象你把一张纸（高维数据）揉成一团（流形结构），现在要把它铺平但又不能撕破。LLE的思路是：只保持每个小区域内（邻域）的纸张完整，然后慢慢把所有小块连接起来铺平。关键洞察是"局部保持"——每个点只和它附近的点保持相对位置，不管远处怎么样。这样就能在保持局部结构的同时实现降维。

**历史背景**：
- 2000年，Sam Roweis和Lawrence Saul在论文"Nonlinear Dimensionality Reduction by Locally Linear Embedding"中首次提出
- 同期另一篇重要论文由Tenenbaum等提出Isomap
- 2003年，Zhang等提出LLE的改进版LLE-Symmetric
- 现在是scikit-learn内置的流行降维方法

**算法定位**：
- 类型：流形学习 → 非线性降维
- 输出：低维嵌入向量
- 模型类型：无监督流形学习

**前置知识**：
- [必备]：线性代数基础（矩阵运算、特征值分解）
- [必备]：机器学习基础（降维、聚类）
- [推荐]：PCA、Isomap等降维方法

---

## 2. 核心原理

### 2.1 流形假设

LLE的核心假设是**流形假设**：真实数据往往分布在高维空间中的低维流形上。

```
例如：
- 人脸图像 → 即使是4096维（64×64灰度），但人脸的变化只需要更少的自由度（光照、角度、表情）
- 手写数字 → 784维像素，但数字结构可以用更低维度描述
```

### 2.2 局部线性嵌入的核心思想

**核心洞察**：在流形的局部区域，数据可以近似用线性组合表示！

**三步走策略**：

```
步骤1：找邻居 → 对每个点找到k个最近邻
         ↓
步骤2：算权重 → 用邻居的线性组合表示当前点
         ↓
步骤3：嵌 入 → 在低维空间保持同样的线性组合关系
```

### 2.3 整体流程

```
                         高维空间                          低维空间
    ┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐
    │          原始高维数据                │    │          降维嵌入结果                │
    │         (如 4096维)                │    │           (如 2维)                   │
    │                                     │    │                                     │
    │   • • • •                    • • •   │    │         ●  ●  ●                           │
    │   •  ●  •   ←找邻居        • ● •    │    │         ●  ●  ●   ←保持局部结构       │
    │   • • • •                    • • •   │    │         ●  ●  ●                           │
    └─────────────────────────────────────┘    └─────────────────────────────────────┘
    
    步骤1：找k近邻      步骤2：算权重      步骤3：嵌入
    对每个点找k个        用邻居线性        在低维空间
    最近的点            组合表示          保持权重
```

---

## 3. 数学公式与推导

### 3.1 找邻居

**k近邻搜索**：对每个数据点 $x_i$，找到欧氏距离最近的k个邻居。

**距离度量**：

$$d(x_i, x_j) = \|x_i - x_j\|_2$$

设邻居集合为 $\mathcal{N}(i)$，包含k个最近邻的索引。

### 3.2 计算局部线性权重

**核心约束**：每个点可以用它的k个邻居的线性组合表示：

$$x_i \approx \sum_{j \in \mathcal{N}(i)} w_{ij} x_j$$

**约束条件**：权重之和为1（归一化）

$$\sum_j w_{ij} = 1$$

**最小化重建误差**：

$$\min_{w} \sum_i \|x_i - \sum_{j \in \mathcal{N}(i)} w_{ij} x_j\|^2$$

subject to $\sum_j w_{ij} = 1$

这可以转换为**最小二乘问题**求解！

**解析解**：

令 $X_i$ 为邻居构成的矩阵，目标函数：

$$\|X_i^T w - x_i\|^2$$

使用拉格朗日乘数法，得到闭式解：

$$w_{ij} = \frac{\sum_k C_{ik}^{-1}}{\sum_{l,m} C_{lm}^{-1}}$$

其中 $C_{ik} = (x_i - x_k)^T (x_i - x_m)$ 是局部协方差矩阵。

### 3.3 计算低维嵌入

**核心思想**：在低维空间保持同样的线性组合权重！

**目标函数**：

$$\min_{Y} \sum_i \|y_i - \sum_j w_{ij} y_j\|^2$$

写成矩阵形式：

$$\min_Y \|Y - WY\|^2 = \min_Y \|(I-W)Y\|^2$$

其中 $W$ 是权重矩阵。

**约束**：防止嵌入坍缩

$$\sum_i y_i = 0, \quad \frac{1}{N}YY^T = I$$

### 3.4 特征值分解

**最终求解**：变成广义特征值问题！

$$M = (I-W)^T (I-W)$$

求最小的d+1个特征值（最小的非零特征值），对应的特征向量构成嵌入。

**注意**：最小的特征值接近0（接近零），舍弃第1个，最小的d个对应嵌入。

---

## 4. 训练过程讲解

### 4.1 训练流程

```
       输入数据
           │
           ▼
    ┌───────────────┐
    │  找k近邻   │ ← 对每个点找k个最近邻
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │  计算权重   │ ← 最小二乘求W
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 构建矩阵 M  │ ← M = (I-W)'(I-W)
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 特征值分解  │ ← 求最小d+1个特征值
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │   输出嵌入   │ ← 舍弃最小，取下d个
    └───────────────┘
```

### 4.2 关键超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| n_neighbors (k) | 10-30 | 邻居数量 |
| n_components (d) | 2-3 | 嵌入维度 |
| method | 'standard' | 方法变体 |

### 4.3 权重计算详解

```python
# 伪代码
def compute_weights(X, k):
    N = X.shape[0]
    W = np.zeros((N, N))
    
    for i in range(N):
        # 找k近邻
        distances = pairwise_distances(X[i], X)
        neighbors = np.argsort(distances)[:k]
        
        # 构建局部矩阵
        X_neighbor = X[neighbors]  # k × D
        x_i = X[i]  # D,
        
        # 中心化
        X_centered = X_neighbor - x_i
        
        # 局部协方差
        C = X_centered @ X_centered.T + eps * np.eye(k)
        
        # 求权重（归一化）
        w = np.linalg.solve(C, np.ones(k))
        w = w / w.sum()
        
        W[i, neighbors] = w
    
    return W
```

### 4.4 时间复杂度

| 步骤 | 复杂度 | 说明 |
|------|--------|------|
| ��邻居 | O(N²D) | 需要两两距离 |
| 权重计算 | O(Nk³) | 每个点求解k×k线性系统 |
| 特征分解 | O(N³) | 稠密情况 |
| **总复杂度** | O(N³) | 瓶颈在特征分解 |

针对大数据集，使用稀疏近似（如sklearn的LocallyLinearEmbedding）。

---

## 5. 应用场景

### 5.1 数据可视化

最经典的应用——将高维数据降到2D/3D进行可视化：

```python
# 手写数字可视化
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

# 降到2维
lle = LocallyLinearEmbedding(n_neighbors=30, n_components=2)
X_embed = lle.fit_transform(digits.data)

# 可视化
plt.scatter(X_embed[:, 0], X_embed[:, 1], c=digits.target)
plt.colorbar()
plt.show()
```

### 5.2 图像降维

- 人脸识别（ Eigenfaces）
- 图像检索
- 图像聚类

### 5.3 文本聚类

- 文档向量化后的降维
- 主题模型可视化

### 5.4 生物信息学

- 基因表达数据降维
- 单细胞数据分析

### 5.5 预处理

作为其他模型的预处理步骤——先用LLE降维，再用分类器。

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **非线性降维** | 能处理弯曲流形 |
| **全局保持** | 保持局部结构 |
| **无需迭代优化** | 闭式解 |
| **计算高效** | 特征值分解 |
| **可处理outliers** | 对噪声较鲁棒 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **k敏感** | 邻居数k影响结果 |
| **流形不连通** | 断开区域效果差 |
| **out-of-sample** | 需要重新训练嵌入新点 |
| **计算复杂度** | O(N³)不适用于大数据 |
| **对噪声敏感** | k太小时 |

### 6.3 改进方案

| 改进 | 方法 | 年份 |
|------|------|------|
| LLE-Symmetric | 对称邻居权重 | 2003 |
| Modified LLE | 邻居选择 | 2001 |
| Hessian LLE | Hessian特征值 | 2003 |
| 稀疏近邻 | 加速近似 | 2004 |

---

## 7. 调库实现

### 7.1 scikit-learn实现（推荐）

```python
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np

# 生成弯月数据
np.random.seed(42)
t = np.linspace(0, np.pi, 100)
X1 = np.column_stack([np.cos(t) + 0.1*np.random.randn(100), 
                     np.sin(t) + 0.1*np.random.randn(100)])
X2 = np.column_stack([1 + 0.1*np.random.randn(100), 
                     0.5 + 0.1*np.random.randn(100)])
X = np.vstack([X1, X2])

# LLE降维
lle = LocallyLinearEmbedding(
    n_neighbors=10,      # 邻居数
    n_components=2,       # 嵌入维度
    method='standard'      # 方法
)

X_embed = lle.fit_transform(X)

print(f"原始维度: {X.shape}")
print(f"嵌入维度: {X_embed.shape}")
```

### 7.2 使用自定义数据

```python
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import load_digits

# 加载手写数字数据
digits = load_digits()
print(f"数据形状: {digits.data.shape}")  # (1797, 64)

# 降维到2维用于可视化
lle = LocallyLinearEmbedding(
    n_neighbors=30,
    n_components=2,
    method='standard'
)

X_embed = lle.fit_transform(digits.data)

print(f"嵌入形状: {X_embed.shape}")  # (1797, 2)
```

### 7.3 不同方法变体

```python
from sklearn.manifold import LocallyLinearEmbedding

# 方法对比
methods = ['standard', 'hessian', 'modified', 'ltsa']

for method in methods:
    try:
        lle = LocallyLinearEmbedding(
            n_neighbors=15,
            n_components=2,
            method=method
        )
        X_embed = lle.fit_transform(X)
        print(f"{method}: 成功")
    except Exception as e:
        print(f"{method}: 失败 - {e}")
```

---

## 8. 手工代码实现

### 8.1 完整LLE实现

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh


class LLE:
    """局部线性嵌入完整实现"""
    
    def __init__(self, n_neighbors=10, n_components=2):
        self.k = n_neighbors
        self.d = n_components
        
    def _find_neighbors(self, X):
        """找k近邻"""
        distances = cdist(X, X, metric='euclidean')
        
        # 对每行排序，取前k个
        neighbors = np.argsort(distances, axis=1)[:, 1:self.k+1]  # 排除自己
        
        return neighbors
    
    def _compute_weights(self, X, neighbors):
        """计算局部线性权重"""
        N = X.shape[0]
        W = np.zeros((N, N))
        
        eps = 1e-5
        
        for i in range(N):
            # 当前点和邻居
            x_i = X[i]
            neighbor_idx = neighbors[i]
            X_neighbor = X[neighbor_idx]  # (k, D)
            
            # 中心化
            X_centered = X_neighbor - x_i
            
            # 局部协方差矩阵
            C = X_centered @ X_centered.T  # (k, k)
            C += eps * np.eye(self.k)  # 正则化
            
            # 求解 (C @ w = 1)
            w = np.linalg.solve(C, np.ones(self.k))
            
            # 归一化
            w = w / w.sum()
            
            # 填充权重矩阵
            W[i, neighbor_idx] = w
        
        return W
    
    def _embed(self, W):
        """计算低维嵌入"""
        N = W.shape[0]
        
        # 构建 (I-W)
        I = np.eye(N)
        M = I - W
        
        # M = (I-W)T @ (I-W)
        M = M.T @ M
        
        # 特征值分解
        eigenvalues, eigenvectors = eigh(M)
        
        # 排序（小到大）
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 舍弃最小的（接近0），取d个
        embedding = eigenvectors[:, 1:self.d+1]
        
        return embedding
    
    def fit_transform(self, X):
        """拟合并转换"""
        print(f"原始数据: {X.shape}")
        
        # 1. 找邻居
        print("步骤1: 找k近邻...")
        neighbors = self._find_neighbors(X)
        
        # 2. 计算权重
        print("步骤2: 计算局部线性权重...")
        W = self._compute_weights(X, neighbors)
        
        # 3. 计算嵌入
        print("步骤3: 计算低维嵌入...")
        embedding = self._embed(W)
        
        print(f"嵌入结果: {embedding.shape}")
        
        return embedding


def demo():
    """演示LLE"""
    print("=== LLE演示 ===\n")
    
    # 生成S形（S-curve）数据
    np.random.seed(42)
    n_samples = 500
    
    # S形采样
    t = np.linspace(0, np.pi, n_samples//2)
    X1 = np.column_stack([
        t * np.cos(t) + 0.1*np.random.randn(n_samples//2),
        t * np.sin(t) + 0.1*np.random.randn(n_samples//2)
    ])
    X2 = np.column_stack([
        (t + np.pi) * np.cos(t + np.pi) + 0.1*np.random.randn(n_samples//2),
        (t + np.pi) * np.sin(t + np.pi) + 0.1*np.random.randn(n_samples//2)
    ])
    X = np.vstack([X1, X2])
    
    print(f"生成数据形状: {X.shape}")
    
    # LLE降维
    lle = LLE(n_neighbors=15, n_components=2)
    X_embed = lle.fit_transform(X)
    
    # 可视化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始数据
    axes[0].scatter(X[:, 0], X[:, 1], c=np.arange(len(X)), 
                   cmap='viridis', s=10)
    axes[0].set_title('原始数据（S形流形）')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    
    # 嵌入结果
    axes[1].scatter(X_embed[:, 0], X_embed[:, 1], c=np.arange(len(X)), 
                    cmap='viridis', s=10)
    axes[1].set_title('LLE嵌入（2维）')
    axes[1].set_xlabel('y1')
    axes[1].set_ylabel('y2')
    
    plt.tight_layout()
    plt.savefig('lle_demo.png')
    plt.show()


if __name__ == "__main__":
    demo()
```

### 8.2 优化版本（处理稀疏矩阵）

```python
import numpy as np
from scipy.sparse.csr_matrix
from scipy.sparse.linalg import eigsh


class SparseLLE:
    """稀疏近似LLE，适合大数据"""
    
    def __init__(self, n_neighbors=10, n_components=2):
        self.k = n_neighbors
        self.d = n_components
        
    def fit_transform(self, X):
        """稀疏近似实现"""
        N = X.shape[0]
        
        # 找邻居
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=self.k+1).fit(X)
        _, neighbors = nbrs.kneighbors(X)
        neighbors = neighbors[:, 1:]  # 排除自己
        
        # 计算权重（稀疏版本）
        W = np.zeros((N, self.k))
        
        for i in range(N):
            # 局部约束最小二乘
            x_i = X[i]
            x_neighbors = X[neighbors[i]]
            
            # 中心化
            z = x_neighbors - x_i
            
            # 局部PCA（简化）
            C = z @ z.T + 1e-5 * np.eye(self.k)
            w = np.linalg.solve(C, np.ones(self.k))
            W[i] = w / w.sum()
        
        # 构建稀疏权重矩阵
        W_sparse = np.zeros((N, N))
        for i in range(N):
            W_sparse[i, neighbors[i]] = W[i]
        
        # 特征值分解（使用稀疏近似）
        M = (np.eye(N) - W_sparse)
        M = M.T @ M
        
        # 取最小的d+1个特征值
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        idx = np.argsort(eigenvalues)
        
        embedding = eigenvectors[:, idx[1:self.d+1]]
        
        return embedding
```

---

## 9. 可视化与结果理解

### 9.1 流形可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import make_s_curve, make_swiss_roll


def visualize_manifolds():
    """可视化各种流形"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 1. S曲线
    X, color = make_s_curve(n_samples=1000, random_state=42)
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=color, cmap='viridis', s=10)
    axes[0, 0].set_title('S曲线（原始3维）')
    
    lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2)
    X_embed = lle.fit_transform(X)
    axes[0, 1].scatter(X_embed[:, 0], X_embed[:, 1], c=color, cmap='viridis', s=10)
    axes[0, 1].set_title('LLE嵌入（2维）')
    
    # 2. Swiss Roll
    X, color = make_swiss_roll(n_samples=1000, random_state=42)
    axes[1, 0].scatter(X[:, 0], X[:, 1], c=color, cmap='viridis', s=10)
    axes[1, 0].set_title('Swiss Roll（原始3维）')
    
    lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2)
    X_embed = lle.fit_transform(X)
    axes[1, 1].scatter(X_embed[:, 0], X_embed[:, 1], c=color, cmap='viridis', s=10)
    axes[1, 1].set_title('LLE嵌入（2维）')
    
    # 3. 双月亮
    from sklearn.datasets import make_moons
    X, color = make_moons(n_samples=500, random_state=42)
    X = X * 10  # 放大
    
    axes[0, 2].scatter(X[:, 0], X[:, 1], c=color, cmap='viridis', s=10)
    axes[0, 2].set_title('双月亮（原始2维）')
    
    lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2)
    X_embed = lle.fit_transform(X)
    axes[0, 3].scatter(X_embed[:, 0], X_embed[:, 1], c=color, cmap='viridis', s=10)
    axes[0, 3].set_title('LLE嵌入（2维）')
    
    # 4. 圆环
    t = np.linspace(0, 2*np.pi, 500)
    X = np.column_stack([np.cos(t), np.sin(t)]) * 5
    color = t
    axes[1, 2].scatter(X[:, 0], X[:, 1], c=color, cmap='viridis', s=10)
    axes[1, 2].set_title('圆环（原始2维）')
    
    lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2)
    X_embed = lle.fit_transform(X)
    axes[1, 3].scatter(X_embed[:, 0], X_embed[:, 1], c=color, cmap='viridis', s=10)
    axes[1, 3].set_title('LLE嵌入（2维）')
    
    plt.tight_layout()
    plt.savefig('lle_manifolds.png')
    plt.show()


if __name__ == "__main__":
    visualize_manifolds()
```

### 9.2 k值影响

```python
def visualize_k_effect():
    """展示k值对结果的影响"""
    
    from sklearn.datasets import make_s_curve
    
    X, color = make_s_curve(n_samples=800, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    k_values = [5, 10, 20, 30, 50, 100]
    
    for i, k in enumerate(k_values):
        ax = axes[i//3, i%3]
        
        lle = LocallyLinearEmbedding(n_neighbors=k, n_components=2)
        X_embed = lle.fit_transform(X)
        
        ax.scatter(X_embed[:, 0], X_embed[:, 1], c=color, cmap='viridis', s=10)
        ax.set_title(f'k={k}')
    
    plt.tight_layout()
    plt.savefig('lle_k_effect.png')
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| 重建误差 | $\sum \|x_i - \sum w_{ij}x_j\|^2$ |
| 保持局部结构 | 近邻在嵌入后的距离 |
| 可视化质量 | 流形展开效果 |

### 10.2 重建误差计算

```python
def reconstruction_error(X, neighbors, W):
    """计算重建误差"""
    N = X.shape[0]
    error = 0
    
    for i in range(N):
        x_i = X[i]
        x_neighbors = X[neighbors[i]]
        w = W[i, neighbors[i]]
        
        # 重建
        x_reconstruct = np.sum(w[:, np.newaxis] * x_neighbors, axis=0)
        
        error += np.sum((x_i - x_reconstruct) ** 2)
    
    return error / N
```

### 10.3 方法对比

| 方法 | 重建误差 | 计算时间 | 效果 |
|------|---------|--------|------|
| standard | 低 | 快 | 好 |
| hessian | 中 | 中 | 最好 |
| modified | 中 | 快 | 较好 |

---

## 11. 常见问题与易错点

### 11.1 k值选择

**问题**：k太小导致流形不连通，k太大导致短路。

**解决**：
- 默认k=10-30
- 画k-误差曲线选择
- 多次实验取最优

### 11.2 流形不连通

**问题**：数据所在流形不连通导致效果差

**解决**：
- 分别对每个连通分支做LLE
- 增加数据确保连通
- 换用其他方法

### 11.3 新数据嵌入

**问题**：训练完需要对新数据嵌入

**解决**：
- 只能用transform，不推荐
- 建议和新数据一起重新训练

### 11.4 对稀疏数据效果差

**问题**：数据太稀疏导致邻居选择不稳定

**解决**：
- 增加k值
- 使用其他降维方法

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | 保持局部线性关系的降维 |
| 核心 | 步骤1找k邻、步骤2算权重、步骤3嵌入 |
| 优点 | 非线性、计算快、闭式解 |
| 缺点 | 对k敏感、大数据慢 |

### 12.2 公式记忆

**权重求解**：
$$\min \sum_i \|x_i - \sum_j w_{ij} x_j\|^2 \quad s.t. \sum_j w_{ij}=1$$

**嵌入目标**：
$$\min \sum_i \|y_i - \sum_j w_{ij} y_j\|^2$$

### 12.3 扩展阅读

| 方法 | 特点 | 年份 |
|------|------|------|
| Isomap | 使用测地距离 | 2000 |
| t-SNE | 概率保持 | 2008 |
| UMAP | 拓扑保持 | 2018 |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：LLE的核心假设是什么？

**答案**：流形假设——真实数据分布在高维空间的低维流形上，且在局部区域内可以用线性���合���似。

**练习2**：k值对LLE有什么影响？

**答案**：k太小会导致流形不连通（每个点邻居太少）；k太大会导致短路（远距离点被错误地当作邻居）。需要选择合适的k。

**练习3**：为什么LLE能处理非线性结构？

**答案**：LLE不假设全局线性，而是保持每个点与其k个邻居的局部线性关系，这种局部性使得能捕获弯曲的流形结构。

### 13.2 进阶思考

**思考1**：LLE和PCA的区别？

**答案**：PCA是线性降维，寻找全局最大化方差的方向；LLE是非线性降维，保持局部邻居关系。PCA只能处理平面流形，LLE可以处理弯曲流形。

**思考2**：LLE对噪声数据为何敏感？

**答案**：噪声会影响k近邻的选择，进而影响权重计算和嵌入结果。可以用鲁棒邻居选择或数据预处理应对。

---

## 14. 学习路径建议

### 14.1 入门（1周）

| 天 | 内容 | 目标 |
|----|------|------|
| 1-2 | 降维基础 | 理解PCA |
| 3-4 | 流形假设 | 理解流形概念 |
| 5-6 | LLE原理 | 理解三步骤 |
| 7 | 代码 | 跑通demo |

### 14.2 进阶（2周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | LLE实现 | 完整代码 |
| 2 | 参数调优 | k值选择 |

### 14.3 实战（3周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | 数据准备 | 真实数据 |
| 2 | 可视化 | 效果展示 |
| 3 | 项目 | 端到端应用 |

---

## 附录

### A. 重要参考

| 参考 | 链接 |
|------|------|
| LLE原始论文 | https://science.sciencemag.org/content/290/5500/2323 |
| scikit-learn | https://scikit-learn.org/stable/modules/manifold.html#locally-linear-embedding |

### B. 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_neighbors | 10 | 邻居数 |
| n_components | 2 | 嵌入维度 |
| method | 'standard' | 方法 |

---

**文档结束**