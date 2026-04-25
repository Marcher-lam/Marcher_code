# UMAP 学习文档

> 更快速更高效的流形学习，比t-SNE快且保持更多结构

---

## 1. 算法基础认知

### 1.1 一句话定义

UMAP（Uniform Manifold Approximation and Projection，统一流形近似和投影）是由McInnes等人在2018年提出的非线性降维算法，基于代数拓扑，比t-SNE更快、保留更多全局结构。

### 1.2 直觉类比

UMAP就像"升级版t-SNE"。t-SNE像用橡皮泥把弯曲的表面慢慢压扁——虽然局部形状保持，但整体结构可能扭曲。UMAP的思路更聪明：它先找到数据的"拓扑骨架"（近似流形），再把这个骨架展开——这样既快（不用迭代计算），又保持全局结构！

想象你有一张皱巴巴的纸（高维数据）：
- t-SNE：一点一点把纸抚平，每次只关注局部，结果可能还是皱的
- UMAP：先找到纸的整体框架（拓扑骨架），然后一次性展开——既快又不皱！

### 1.3 发展背景

- 2018年，McInnes和Healy在论文"UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"中提出
- Python库umap-learn自2018年开源
- 2020年后成为流形学习主流算法

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 流形学习 → 非线性降维 |
| 输出 | 2D/3D嵌入 |
| 方法 | 拓扑+优化 |
| 速度 | O(n log n) |

---

## 2. 核心原理

### 2.1 为什么需要UMAP？

**t-SNE的局限**：
- O(n²)时间复杂度，太慢
- 丢失全局结构
- 随机性，需要多次运行
- 只适合2-3维嵌入

**UMAP的优势**：
- O(n log n)，快10倍+
- 保持全局结构
- 确定性结果
- 可用于任意维度

### 2.2 vs t-SNE对比

| 方面 | t-SNE | UMAP |
|------|-------|------|
| 速度 | O(n²) | **O(n log n)** |
| 全局结构 | 丢失 | **保持** |
| 随机性 | 需多次 | **确定性** |
| 理论基础 | 概率 | **拓扑** |
| 维度 | 2-3D | **任意** |
| 内存 | O(n²) | **O(n)** |

### 2.3 核心流程

```
高维数据
    │
    ▼
构建k近邻图
    │
    ▼
模糊拓扑结构
    │
    ▼
交叉熵优化
    │
    ▼
2D嵌入
```

### 2.4 核心思想：拓扑近似

UMAP假设数据分布在局部流形上，通过两点来近似：

1. **局部一致性**：数据局部近似为欧几里得空间
2. **全局连通性**：所有局部流形通过共享点连通

---

## 3. 数学公式与推导

### 3.1 距离度量

**k近邻距离**：
$$d(x_i, x_j) = \|x_i - x_j\|$$

**标准化距离**（自适应）：
$$d^{(c)}(x_i, x_j) = \max\left(0, d(x_i, x_j) - \rho_i\right)$$

其中 $\rho_i$ 是到第i个最近邻的距离。

### 3.2 模糊集合隶属度

对每个点i，其到点j的隶属度：
$$\mu_{ij} = \exp\left(\frac{-d(x_i, x_j)}{\sigma_i}\right)$$

标准化后：
$$\mu_{ij}' = \frac{\mu_{ij}}{\sum_k \mu_{ik}}$$

$\sigma_i$ 是局部调整参数，确保 $\sum_j \mu_{ij} = \log_2(k)$

### 3.3 高维图权重

边的权重（边的概率）：
$$p_{i|j} = \frac{\mu_{ij}'}{\sum_k \mu_{ik}'}$$

对称版本：
$$p_{ij} = \frac{p_{i|j} + p_{j|i}}{2}$$

### 3.4 低维图

在嵌入空间Y中：
$$q_{ij} = \frac{1}{1 + d(y_i, y_j)^2}$$

对称版本：
$$q_{ij} = \frac{q_{i|j} + q_{j|i}}{2}$$

### 3.5 交叉熵损失

$$C = \sum_{i \neq j} p_{ij} \log\frac{p_{ij}}{q_{ij}} + (1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}$$

这个损失同时优化局部和全局结构。

### 3.6 min_dist参数

控制嵌入点之间的最小距离：
$$q_{ij}' = \frac{q_{ij}}{1 + \frac{d(y_i, y_j)^2}{min\_dist}}$$

---

## 4. 训练过程讲解

### 4.1 参数配置

```python
# UMAP参数
config = {
    'n_neighbors': 15,        # 邻居数（影响局部vs全局）
    'min_dist': 0.1,         # 聚集程度
    'n_components': 2,        # 目标维度
    'metric': 'euclidean',     # 距离度量
    'learning_rate': 1.0,     # 学习率
    'n_epochs': 200,         # 训练轮数
}
```

### 4.2 参数影响

| 参数 | 影响 | 建议 |
|------|------|------|
| n_neighbors大 | 更多全局结构 | 15-50 |
| n_neighbors小 | 更多局部结构 | 5-15 |
| min_dist大 | 聚合 | 0.0-0.5 |
| min_dist小 | 分散 | 0.0-0.5 |

### 4.3 距离度量

```python
# 可用度量
metrics = [
    'euclidean',      # 欧几里得
    'manhattan',    # 曼哈顿
    'cosine',       # 余弦
    'correlation', # 相关性
    'hamming',     # 汉明（离散）
]
```

---

## 5. 应用场景

### 5.1 单细胞分析

单细胞RNA-seq降维：

```python
import scanpy as sc
import umap

# PCA预处理
pca = sc.pp.pca(adata)

# UMAP降维
reducer = umap.UMAP(n_components=2)
adata.obsm['X_umap'] = reducer.fit_transform(pca)

# 可视化
sc.pl.umap(adata, color=['cell_type', 'gene'])
```

### 5.2 可视化

```python
import matplotlib.pyplot as plt
import umap

# 降维
reducer = umap.UMAP()
embedding = reducer.fit_transform(X)

# 按类着色
for i in range(num_classes):
    mask = labels == i
    plt.scatter(embedding[mask, 0], embedding[mask, 1], label=f'Class {i}')

plt.legend()
plt.title('UMAP Visualization')
plt.savefig('umap_visualization.png', dpi=100)
plt.show()
```

### 5.3 特征提取

```python
# 作为特征提取器
reducer = umap.UMAP(n_components=10)
X_umap = reducer.fit_transform(X)

# 用于下游分类
clf = RandomForestClassifier()
clf.fit(X_umap, y)
```

### 5.4 对比

| 场景 | 方法 | 效果 |
|------|------|------|
| MNIST | t-SNE | 局部清晰 |
| 单细胞 | **UMAP** | **全局保持+快10x** |
| 文本 | t-SNE | 慢 |
| UMAP | **可用+d维** | **高效** |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **快** | O(n log n)，比t-SNE快10倍+ |
| **全局结构** | 保持数据整体结构 |
| **确定性** | 每次运行结果相同 |
| **可扩展** | 可用于任意维度 |
| **少参数** | 主要2个参数 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 参数敏感 | n_neighbors影响大 |
| 内存 | 大数据集需要更多内存 |
| 可解释性 | 不如PCA直观 |

### 6.3 注意事项

- n_neighbors：太小会丢失全局结构，太大会混乱
- min_dist：影响点的聚集程度
- 适合连续数据，离散数据效果可能差

---

## 7. 调库实现（Python）

### 7.1 基础用法

```python
import numpy as np
import umap

# 生成模拟数据
np.random.seed(42)
X = np.random.randn(1000, 50)

# 混合多类数据
class1 = X[:300] + np.random.randn(300, 50) * 0.1
class2 = X[300:600] + np.random.randn(300, 50) * 0.5 + 3
class3 = X[600:] + np.random.randn(400, 50) * 0.3 + 6

X_mixed = np.vstack([class1, class2, class3])
labels = np.array([0]*300 + [1]*300 + [2]*400)

# UMAP降维
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(X_mixed)

print(f"输入形状: {X_mixed.shape}")
print(f"输出形状: {embedding.shape}")
```

### 7.2 参数调优

```python
# 不同参数对比
results = []

for n_neighbors in [5, 15, 30, 50]:
    for min_dist in [0.0, 0.1, 0.3, 0.5]:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
        embedding = reducer.fit_transform(X)
        
        # 简单评估：聚类分离度
        from sklearn.metrics import silhouette_score
        score = silhouette_score(embedding, labels)
        
        results.append({
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'score': score
        })

# 打印最优
best = max(results, key=lambda x: x['score'])
print(f"最优: n_neighbors={best['n_neighbors']}, min_dist={best['min_dist']}, score={best['score']:.3f}")
```

### 7.3 大规模数据

```python
# 对大数据集使用采样
def umap_sample(X, sample_size=10000):
    """大数据集采样UMAP"""
    if len(X) <= sample_size:
        return umap.UMAP().fit_transform(X)
    
    # 随机采样
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[idx]
    
    # 训练
    reducer = umap.UMAP()
    embedding_sample = reducer.fit_transform(X_sample)
    
    # 转换剩余数据
    embedding = reducer.transform(X)
    
    return embedding

# 使用
embedding = umap_sample(large_dataset)
```

### 7.4 自定义度量

```python
import umap.utils as umap_utils

# 使用自定义距离函数
def my_distance(x, y):
    return np.linalg.norm(x - y)

# 传入custom_metric
reducer = umap.UMAP(metric=my_distance)
embedding = reducer.fit_transform(X)
```

---

## 8. 手工代码实现（理解原理）

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SimpleUMAP:
    """简化版UMAP - 理解原理"""
    
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
    
    def _find_neighbors(self, X):
        """找到k近邻"""
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        return distances, indices
    
    def _compute_memberships(self, X, distances, indices):
        """计算模糊集合隶属度"""
        n = len(X)
        
        # 计算sigma
        sigma = np.zeros(n)
        for i in range(n):
            d = distances[i]
            sigma[i] = d[-1] / np.log(self.n_neighbors)
        
        # 计算隶属度
        memberships = np.exp(-distances / sigma[:, np.newaxis])
        memberships = memberships / memberships.sum(axis=1, keepdims=True)
        
        return memberships
    
    def fit_transform(self, X):
        """简化版UMAP"""
        n = len(X)
        
        # 1. 找k近邻
        distances, indices = self._find_neighbors(X)
        
        # 2. 计算隶属度
        memberships = self._compute_memberships(X, distances, indices)
        
        # 3. 初始化低维嵌入
        np.random.seed(42)
        Y = np.random.randn(n, self.n_components) * 0.01
        
        # 4. 简化优化
        lr = 1.0
        n_epochs = 100
        
        for epoch in range(n_epochs):
            # 计算低维距离
            dY = np.linalg.norm(Y[:, np.newaxis] - Y[np.newaxis, :], axis=2)
            
            # 简化的吸引-排斥
            q = 1 / (1 + dY**2)
            q = q / q.sum(axis=1, keepdims=True)
            
            # 梯度（简化）
            grad = memberships.sum(axis=1, keepdims=True) @ Y - Y @ q.T
            
            Y += lr * grad * 0.01
        
        return Y


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成测试数据
    X = np.random.randn(500, 20)
    class1 = X[:200] + 5
    class2 = X[200:400] - 5
    class3 = X[400:]
    X = np.vstack([class1, class2, class3])
    
    # 简化版UMAP
    umap_simple = SimpleUMAP(n_components=2)
    embedding = umap_simple.fit_transform(X)
    
    print(f"输出形状: {embedding.shape}")
    
    # sklearn UMAP对比
    import umap
    reducer = umap.UMAP()
    embedding_sklearn = reducer.fit_transform(X)
    
    print(f"sklearn输出形状: {embedding_sklearn.shape}")
```

---

## 9. 可视化与结果理解

### 9.1 聚类可视化

```python
import matplotlib.pyplot as plt

def plot_umap_results(embedding, labels, title='UMAP'):
    """可视化UMAP结果"""
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=[colors[i]], label=f'Class {label}', 
                  alpha=0.6, s=10)
    
    plt.legend()
    plt.title(title)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(f'{title.lower()}.png', dpi=100)
    plt.show()


# 使用
plot_umap_results(embedding, labels, 'UMAP_clusters')
```

### 9.2 参数影响可视化

```python
# 可视化不同参数的影响
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

configs = [
    (5, 0.1, 'n=5, d=0.1'),
    (15, 0.1, 'n=15, d=0.1'),
    (50, 0.1, 'n=50, d=0.1'),
    (15, 0.0, 'n=15, d=0.0'),
    (15, 0.3, 'n=15, d=0.3'),
    (15, 0.5, 'n=15, d=0.5'),
]

for i, (n, d, title) in enumerate(configs):
    reducer = umap.UMAP(n_components=2, n_neighbors=n, min_dist=d)
    emb = reducer.fit_transform(X)
    
    axes[i].scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10', s=5)
    axes[i].set_title(title)

plt.tight_layout()
plt.savefig('umap_params.png', dpi=100)
plt.show()
```

### 9.3 t-SNE对比

```python
# UMAP vs t-SNE对比
from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(n_components=2, perplexity=30)
embedding_tsne = tsne.fit_transform(X)

# UMAP  
embedding_umap = umap.UMAP(n_components=2).fit_transform(X)

# 绘制对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels, cmap='tab10', s=5)
axes[0].set_title('t-SNE')

axes[1].scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=labels, cmap='tab10', s=5)
axes[1].set_title('UMAP')

plt.tight_layout()
plt.savefig('umap_vs_tsne.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| 运行时间 | 降维速度 |
| 轮廓系数 | 聚类质量 |
| 邻域保持 | 局部结构保持程度 |
| 全局结构 | 整体簇关系 |

### 10.2 评估代码

```python
import time
from sklearn.metrics import silhouette_score

def evaluate_umap(X, embedding, labels):
    """评估UMAP效果"""
    
    # 运行时间
    start = time.time()
    reducer = umap.UMAP()
    reducer.fit_transform(X)
    runtime = time.time() - start
    
    # 轮廓系数
    silhouette = silhouette_score(embedding, labels)
    
    return {
        'runtime': runtime,
        'silhouette': silhouette
    }

# 评估
results = evaluate_umap(X, embedding, labels)
print(f"运行时间: {results['runtime']:.2f}s")
print(f"轮廓系数: {results['silhouette']:.3f}")
```

---

## 11. 常见问题与易错点

### Q1: n_neighbors如何选择？

**答案**：
- 15（默认）- 平衡局部和全局
- 5-15- 更关注局部结构
- 30-50- 更关注全局结构

### Q2: 为什么结果不稳定？

**答案**：UMAP是确定性的！如果结果变化，检查是否用了随机初始化（如pca_helper随机种子）。

### Q3: 大数据集太慢？

**答案**：
- 使用采样（先对大样本UMAP，再transform小样本）
- 减少n_neighbors
- 减少n_epochs

### Q4: 嵌入成一团？

**答案**：min_dist太小。试试增加到0.3-0.5。

### Q5: 如何选择目标维度？

**答案**：
- 可视化：2D或3D
- 特征提取：10-50

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心 | 拓扑流形近似 |
| 公式 | 模糊集合+交叉熵 |
| 参数 | n_neighbors, min_dist |
| 优势 | 快+全局结构 |

### 12.2 公式汇总

隶属度：
$$\mu_{ij} = \exp(-d(x_i, x_j) / \sigma_i)$$

交叉熵：
$$C = \sum p_{ij} \log(p_{ij}/q_{ij})$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. UMAP相比t-SNE的优势是：
   - A) 更精确
   - B) 更快
   - C) 更简单

2. n_neighbors过大的影响是：
   - A) 更局部
   - B) 更全局
   - C) 无影响

### 13.2 简答题

1. 解释UMAP的拓扑近似原理。
2. 比较t-SNE和UMAP的适用场景。

### 13.3 编程题

1. 在实际数据集上比较不同参数。
2. 实现基于UMAP的特征提取+分类。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
PCA基础
    ↓
流形学习
    ↓
t-SNE
    ↓
UMAP原理
    ↓
实际应用
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| t-SNE | 前辈 |
| Isomap | 流形版PCA |
| LLE | 局部线性 |
| | |
| | |

### 14.3 扩展阅读

- McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction

---

## 附录

### 参考

1. McInnes, L., Healy, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction
2. https://github.com/lmcinnes/umap-learn
3. https://umap-learn.readthedocs.io/

---

**文档结束**