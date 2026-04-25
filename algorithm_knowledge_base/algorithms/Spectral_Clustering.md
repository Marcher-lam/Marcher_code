# Spectral Clustering 谱聚类 学习文档

> 基于图论的高级聚类算法

---

## 1. 算法基础认知

### 1.1 一句话定义

Spectral Clustering（谱聚类）是基于图论的高级聚类算法，通过对数据的相似度矩阵进行特征分解，在谱空间中找到最优划分，是当前最流行的聚类算法之一。

### 1.2 直觉类比

谱聚类就像"切分网络"。把每个数据点看成一个节点，点与点之间的相似度看成连边的权重。要做的就是在保持"同类联系紧密"的前提下，把这个网络切成几个部分——这通过图的拉普拉斯矩阵的特征向量来实现！

想象你有一堆互相连接的灯泡，连接亮度代表相似度。要把这些灯泡分成几组，使得同组内的灯泡连接紧密，不同组之间连接稀疏。谱聚类的方法是：先把这些连接关系写成矩阵，求出特殊的"振动模式"（特征向量），然后在这个模式下重新排列灯泡——此时只需要简单的K-Means就能分开了！

### 1.3 发展背景

- 2002年，Ng, Jordan, Weiss提出（经典论文"On Spectral Clustering: Analysis and an Algorithm"）
- 2007年，Zelnik-Manor和Perona提出个性化PageRank
- 相比传统聚类，能发现任意形状的簇

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 聚类 → 谱方法 |
| 输出 | 聚类标签 |
| 方法 | 图分割 |
| 复杂度 | O(n³) |

---

## 2. 核心原理

### 2.1 为什么需要谱聚类？

**传统聚类的局限**：K-Means假设簇是凸形的，假设数据可以通过中心来描述。

```
K-Means擅长：
    ▲           不擅长：
    ●●●●           ●
    ●●●●           ●
    ●●●●           ●●●
                           ●●
```

而谱聚类不假设簇的形状，可以发现任意连接的簇。

### 2.2 核心思想

谱聚类的核心是"图分割"思想：

1. **建图**：把数据点看成图的节点，相似度是边的权重
2. **切图**：把图切成k个部分，使得内部紧密，外部稀疏
3. **映射**：用特征向量做新的表示
4. **聚类**：在新的表示空间用K-Means

### 2.3 算法流程

```
输入: 数据X, 聚类数k

Step 1: 构建相似度矩阵
       W_ij = exp(-||x_i - x_j||² / 2σ²)

Step 2: 构建度矩阵
       D_ii = Σ_j W_ij

Step 3: 计算拉普拉斯矩阵
       L = D - W (未归一化)
       或 L_norm = D^(-1/2) W D^(-1/2) (归一化)

Step 4: 求L最小的k个特征向量
       v₁, v₂, ..., v_k

Step 5: 构建特征向量矩阵V
       V = [v₁, v₂, ..., v_k] ∈ R^(n×k)

Step 6: 归一化行
       Y_i = V_i / ||V_i||

Step 7: 对Y用K-Means聚类

输出: 聚类标签
```

### 2.4 拉普拉斯矩阵性质

- L = D - W（未归一化）
- L是对称半正定矩阵
- 最小特征值0，对应特征向量[1,1,...,1]
- L的前k个特征向量构成k路划分的基

---

## 3. 数学公式与推导

### 3.1 相似度矩阵

**高斯核相似度**：
$$W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$

σ是带宽参数，控制邻域大小。

### 3.2 度矩阵

$$D_{ii} = \sum_{j=1}^n W_{ij}$$

### 3.3 拉普拉斯矩阵

**未归一化（Combinatorial）**：
$$L = D - W$$

**归一化（Symmetric）**：
$$L_{sym} = D^{-1/2} W D^{-1/2}$$

**归一化（Random Walk）**：
$$L_{rw} = D^{-1} L = I - D^{-1}W$$

### 3.4 优化目标

**RatioCut（未归一化）**：
$$\min_A \frac{\sum_{i,j \in A} W_{ij}}{\sum_{i \in A} |A|}$$

**Normalized Cut（归一化）**：
$$\min_A \frac{\sum_{i,j \in A} W_{ij}}{\sum_{i \in A} D_{ii}}$$

### 3.5 近似求解

由于精确分割是NP难问题，谱聚类通过松弛到连续问题求解：

$$\min_V V^T L V \quad \text{s.t.} V^T D V = I$$

解为L的最小特征向量。

---

## 4. 训练过程讲解

### 4.1 相似度计算

```python
def compute_affinity(X, sigma=1.0):
    """计算高斯核相似度矩阵"""
    n = len(X)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = np.sum((X[i] - X[j])**2)
            W[i, j] = np.exp(-dist / (2 * sigma**2))
            W[j, i] = W[i, j]
    return W
```

### 4.2 特征分解

```python
def spectral_clustering(X, k, sigma=1.0, mode='normalized'):
    # Step 1: 相似度矩阵
    W = compute_affinity(X, sigma)
    
    # Step 2: 度矩阵
    D = np.diag(W.sum(axis=1))
    
    # Step 3: 拉普拉斯矩阵
    if mode == 'unnormalized':
        L = D - W
    else:  # normalized
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        L = np.eye(len(X)) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # Step 4: 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Step 5: 取最小的k个
    V = eigenvectors[:, :k]
    
    # Step 6: 归一化行
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-10)
    
    # Step 7: K-Means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(V_norm)
    
    return labels
```

### 4.3 参数选择

| 参数 | 说明 | 选择方法 |
|------|------|----------|
| sigma | 核带宽 | 经验：数据方差的均值 |
| n_clusters | 聚类数 | elbow/silhouette |
| affinity | 相似度类型 | rbf/knn |

### 4.4 sigma选择技巧

```python
# 自动选择sigma
# 方法1：取所有对距离的中位数
dists = []
for i in range(n):
    for j in range(i+1, n):
        dists.append(np.sum((X[i]-X[j])**2))
sigma = np.median(dists)

# 方法2：取k近邻的距离均值
dists = []
for i in range(n):
    dist_to_others = np.sum((X - X[i])**2, axis=1)
    dist_to_others[i] = np.inf
    nearest_k_dists = np.sort(dist_to_others)[:10]
    dists.extend(nearest_k_dists)
sigma = np.mean(dists)
```

---

## 5. 应用场景

### 5.1 图像分割

谱聚类在计算机视觉中应用广泛：

```python
# 简单图像分割示例
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=200, noise=0.05)
spectral = SpectralClustering(n_clusters=2, affinity='rbf', gamma=1.0)
labels = spectral.fit_predict(X)
```

### 5.2 文本聚类

对文档相似度矩阵聚类：

```python
# 文本用TF-IDF + 余弦相似度
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF向量化
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(documents)

# 余弦相似度
W = cosine_similarity(X_tfidf)

# 谱聚类
spectral = SpectralClustering(n_clusters=k, affinity='precomputed')
labels = spectral.fit_predict(W)
```

### 5.3 社区发现

社交网络中的社区检测：

```python
# 图的邻接矩阵作为相似度
spectral = SpectralClustering(n_clusters=k, affinity='precomputed')
labels = spectral.fit_predict(adjacency_matrix)
```

### 5.4 对比选择

| 场景 | 推荐算法 |
|------|----------|
| 凸形簇 | K-Means |
| 非凸任意形状 | 谱聚类 |
| 大规模数据 | Mini-Batch K-Means |
| 密度簇 | DBSCAN |
| 层次结构 | 层次聚类 |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 任意形状 | 能发现非凸复杂形状 |
| 不假设分布 | 不需要高斯假设 |
| 理论基础 | 有图论保证 |
| 效果好 | 聚类质量高 |
| 全局最优 | 求近似解 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| O(n³)复杂度 | 不适合大规模数据 |
| 参数敏感 | sigma/gamma敏感 |
| 需要K-Means | 两步方法 |
| 内存消耗 | 存n×n矩阵 |

### 6.3 注意事项

1. **数据尺度**：先标准化，否则相似度不准
2. **sigma选择**：太大则全连通，太小则稀疏
3. **结果不稳定**：特征向量方向可能反转
4. **大规模数据**：用Nystroem近似

---

## 7. 调库实现（Python + scikit-learn）

### 7.1 基本用法

```python
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

# 生成测试数据（两弯月形）
np.random.seed(42)
X, _ = make_moons(n_samples=200, noise=0.05)

# 谱聚类
spectral = SpectralClustering(
    n_clusters=2,
    affinity='rbf',        # 相似度类型
    gamma=1.0,             # RBF核参数
    random_state=42
)
labels = spectral.fit_predict(X)

print("聚类分布:", np.bincount(labels))
```

### 7.2 不同相似度

```python
# RBF核（高斯核）
spectral_rbf = SpectralClustering(n_clusters=2, affinity='rbf', gamma=1.0)
labels_rbf = spectral_rbf.fit_predict(X)

# knn近邻
spectral_knn = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', 
                                 n_neighbors=10)
labels_knn = spectral_knn.fit_predict(X)

# 预计算相似度
from sklearn.metrics.pairwise import rbf_kernel
W = rbf_kernel(X, gamma=1.0)
spectral_pre = SpectralClustering(n_clusters=2, affinity='precomputed')
labels_pre = spectral_pre.fit_predict(W)
```

### 7.3 归一化对比

```python
# 未归一化拉普拉斯
spectral_un = SpectralClustering(n_clusters=2, affinity='rbf', 
                                assign_labels='discretize', random_state=42)
labels_un = spectral_un.fit_predict(X)

# 归一化拉普拉斯（默认）
spectral_norm = SpectralClustering(n_clusters=2, affinity='rbf',
                                  assign_labels='kmeans', random_state=42)
labels_norm = spectral_norm.fit_predict(X)
```

### 7.4 参数调优

```python
from sklearn.metrics import silhouette_score

# 网格搜索最优参数
best_score = -1
best_params = {}

for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
    for n_neighbors in [5, 10, 15, 20]:
        spectral = SpectralClustering(
            n_clusters=2,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            gamma=gamma,
            random_state=42
        )
        labels = spectral.fit_predict(X)
        score = silhouette_score(X, labels)
        
        if score > best_score:
            best_score = score
            best_params = {'gamma': gamma, 'n_neighbors': n_neighbors}

print(f"最优参数: {best_params}")
print(f"最优轮廓系数: {best_score:.3f}")
```

---

## 8. 手工代码实现（核心算法手写）

```python
import numpy as np

class SpectralClustering:
    """谱聚类 - 手工实现"""
    
    def __init__(self, n_clusters=2, sigma='auto', affinity='rbf', 
                 n_neighbors=10, random_state=None):
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.labels_ = None
        
    def _compute_affinity(self, X):
        """计算相似度矩阵"""
        if self.affinity == 'rbf':
            if self.sigma == 'auto':
                # 自动选择sigma：取k近邻距离均值
                dists = []
                for i in range(len(X)):
                    d = np.sum((X - X[i])**2, axis=1)
                    d[i] = np.inf
                    dists.extend(np.sort(d)[:self.n_neighbors])
                self.sigma = np.mean(dists)
            
            # 高斯核
            n = len(X)
            sq_dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2*X@X.T
            W = np.exp(-sq_dists / (2 * self.sigma**2))
            
        elif self.affinity == 'nearest_neighbors':
            n = len(X)
            W = np.zeros((n, n))
            for i in range(n):
                dists = np.sum((X - X[i])**2, axis=1)
                nearest = np.argsort(dists)[1:self.n_neighbors+1]
                W[i, nearest] = 1
            W = (W + W.T) / 2
            
        np.fill_diagonal(W, 0)
        return W
    
    def _compute_laplacian(self, W, mode='symmetric'):
        """计算拉普拉斯矩阵"""
        D = np.diag(W.sum(axis=1))
        
        if mode == 'unnormalized':
            L = D - W
        elif mode == 'symmetric':
            D_inv_sqrt = np.diag(1.0 / (np.diag(D)**0.5 + 1e-10))
            L = np.eye(len(X)) - D_inv_sqrt @ W @ D_inv_sqrt
        elif mode == 'random_walk':
            D_inv = np.diag(1.0 / (np.diag(D) + 1e-10))
            L = np.eye(len(X)) - D_inv @ W
            
        return L
    
    def _kmeans(self, X, k):
        """简单的K-Means实现"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n = len(X)
        
        # 随机初始化中心
        centroids = X[np.random.choice(n, k, replace=False)]
        
        for _ in range(100):
            # 分配
            dists = np.array([[np.sum((x - c)**2) for c in centroids] for x in X])
            labels = np.argmin(dists, axis=1)
            
            # 更新
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).sum() > 0 else centroids[j]
                for j in range(k)
            ])
            
            # 收敛检查
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        return labels
    
    def fit(self, X):
        """拟合"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Step 1: 相似度矩阵
        W = self._compute_affinity(X)
        
        # Step 2: 拉普拉斯矩阵
        L = self._compute_laplacian(W, mode='symmetric')
        
        # Step 3: 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Step 4: 取前k个特征向量
        V = eigenvectors[:, :self.n_clusters]
        
        # Step 5: 行归一化
        V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-10)
        
        # Step 6: K-Means聚类
        self.labels_ = self._kmeans(V_norm, self.n_clusters)
        
        return self
    
    def fit_predict(self, X):
        """拟合+预测"""
        self.fit(X)
        return self.labels_


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成测试数据
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=200, noise=0.05)
    
    # 手工实现
    spectral = SpectralClustering(n_clusters=2, affinity='rbf', gamma=2.0, random_state=42)
    labels = spectral.fit_predict(X)
    
    # sklearn实现
    from sklearn.cluster import SpectralClustering
    spectral_sklearn = SpectralClustering(n_clusters=2, affinity='rbf', gamma=2.0, random_state=42)
    labels_sklearn = spectral_sklearn.fit_predict(X)
    
    print("手工实现分布:", np.bincount(labels))
    print("sklearn分布:", np.bincount(labels_sklearn))
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles

# 测试不同数据
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

datasets = [
    ('moons', make_moons(n_samples=200, noise=0.05)),
    ('circles', make_circles(n_samples=200, noise=0.05, factor=0.5)),
    ('blobs', None),
]

# 生成blobs
from sklearn.datasets import make_blobs
np.random.seed(42)
X_blobs, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0)
datasets = [
    ('moons', make_moons(n_samples=200, noise=0.05)),
    ('circles', make_circles(n_samples=200, noise=0.05, factor=0.5)),
    ('blobs', (X_blobs, None)),
]

for col, (name, data) in enumerate(datasets):
    if data is None:
        continue
    X, _ = data
    
    # 谱聚类
    spectral = SpectralClustering(n_clusters=2 if name != 'blobs' else 3, 
                               affinity='rbf', gamma=2.0, random_state=42)
    labels = spectral.fit_predict(X)
    
    # 可视化
    axes[0, col].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    axes[0, col].set_title(f'Spectral: {name}')
    
    # 对比：K-Means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2 if name != 'blobs' else 3, random_state=42)
    labels_km = kmeans.fit_predict(X)
    
    axes[1, col].scatter(X[:, 0], X[:, 1], c=labels_km, cmap='viridis', alpha=0.7)
    axes[1, col].set_title(f'K-Means: {name}')

plt.tight_layout()
plt.savefig('spectral_clustering_demo.png', dpi=100)
plt.show()
```

**结果解读**：谱聚类可以正确分开月亮形和环形数据，而K-Means会失败。

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 轮廓系数 | $(b-a)/max(a,b)$ | 越大越好 |
| CH指数 | $BGS/(k-1)/(n-k)/WGS$ | 越大越好 |
| NMI | 归一化互信息 | 越大越好 |

### 10.2 评估代码

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, normalized_mutual_info_score

# 真实标签（如果有）
y_true = ...

# 预测标签
y_pred = labels

# 轮廓系数
silhouette = silhouette_score(X, y_pred)
print(f"轮廓系数: {silhouette:.3f}")

# CH指数
ch = calinski_harabasz_score(X, y_pred)
print(f"CH指数: {ch:.1f}")

# NMI
nmi = normalized_mutual_info_score(y_true, y_pred)
print(f"NMI: {nmi:.3f}")
```

---

## 11. 常见问题与易错点

### Q1: 如何选择sigma？

**答案**：太大则过度连接，太小则稀疏。用k近邻距离均值作为初值。

### Q2: 结果不稳定？

**答案**：特征向量可能有符号反转。用多启动取最优。

### Q3: 复杂度太高？

**答案**：用Nystroem近似或随机化特征分解。

### Q4: 和K-Means的区别？

**答案**：谱聚类能在特征空间分开非凸数据，K-Means不能。

### Q5: 需要多少数据？

**答案**：建议n>50，太少则特征分解不稳定。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心思想 | 图分割 |
| 拉普拉斯矩阵 | L = D - W |
| 特征向量 | 最小k个 |
| 优势 | 任意形状 |

### 12.2 公式汇总

相似度：
$$W_{ij} = \exp(-\|x_i-x_j\|^2/2\sigma^2)$$

度：
$$D_{ii} = \sum_j W_{ij}$$

拉普拉斯：
$$L = D - W$$

归一化：
$$L_{sym} = D^{-1/2} W D^{-1/2}$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. 谱聚类的复杂度是：
   - A) O(n)
   - B) O(n²)
   - C) O(n³)

2. 拉普拉斯矩阵的最小特征值是：
   - A) -1
   - B) 0
   - C) 1

3. 谱聚类适合的数据是：
   - A) 凸形簇
   - B) 任意形状
   - C) 球形簇

### 13.2 简答题

1. 为什么谱聚类需要两步（特征分解+K-Means）？
2. 比较归一化和未归一化拉普拉斯。

### 13.3 编程题

1. 实现基于Nystroem近似的谱聚类。
2. 用谱聚类对实际图像做分割。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
线性代数
    ↓
图论基础
    ↓
矩阵分解
    ↓
谱聚类原理
    ↓
应用实践
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| K-Means | 简单版本 |
| DBSCAN | 密度方法 |
| 层次聚类 | 树状结构 |
| 图卷积网络 | 深度学习版 |

### 14.3 扩展阅读

- Ng, Jordan, Weiss (2002). On Spectral Clustering
- von Luxburg (2007). A Tutorial on Spectral Clustering

---

## 附录

### 参考

1. Ng, A.Y., Jordan, M.I., Weiss, Y. (2002). On Spectral Clustering: Analysis and an Algorithm. NIPS.
2. von Luxburg, U. (2007). A Tutorial on Spectral Clustering. Statistics and Computing.
3. sklearn.cluster.SpectralClustering 文档

---

**文档结束**