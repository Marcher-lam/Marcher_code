# Mini-Batch K-Means 小批量K均值 学习文档

> 大规模数据的快速聚类算法

---

## 1. 算法基础认知

### 1.1 一句话定义

Mini-Batch K-Means是K-Means的优化版本，每次只使用小批量数据进行聚类更新，大幅减少计算量的同时保持近似精度。

### 1.2 直觉类比

Mini-Batch K-Means就像从"全城人口普查"变成"抽样调查"。普通K-Means每次都要用全部数据计算新的聚类中心，Mini-Batch K-Means每次只随机抽取一部分人做调查——这样速度当然快很多，虽然偶尔会有小偏差但整体结果差不多！

想象你要找三个餐厅最集中的区域。如果你问遍全城所有人当然最准确，但太慢了。Mini-Batch K-Means的方法是：每次随机问100个人，根据他们的回答更新猜测位置，重复几次——这样虽然可能稍微有些波动，但结果差不多，而且快多了！

### 1.3 发展背景

- 2010年，由Sculley在Google提出（论文"Web-scale K-Means clustering"）
- 用于解决大规模数据的聚类问题

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 聚类 → K-Means变体 |
| 输出 | 聚类标签+中心 |
| 方法 | 随机小批量 |
| 复杂度 | O(batch_size × k × iter) |

---

## 2. 核心原理

### 2.1 为什么需要Mini-Batch？

**K-Means的问题**：每次迭代需要计算所有样本到中心的距离

$$O(n \cdot k \cdot d \cdot I)$$

其中：
- n = 样本数
- k = 聚类数
- d = 维度
- I = 迭代次数

当n很大时（如百万级），这个计算量无法接受。

### 2.2 vs K-Means对比

| 方面 | K-Means | Mini-Batch K-Means |
|------|---------|-------------------|
| 数据量 | 全部样本 | 小批量样本(batch_size) |
| 计算复杂度 | O(n·k·d·I) | O(b·k·d·I) |
| 速度 | 慢 | 快10-100倍 |
| 精度 | 高(~最优) | 略低(~相似) |
| 内存 | O(n·d) | O(batch·d) |
| 收敛 | 稳定 | 有噪动 |

### 2.3 算法流程

```
输入: 数据X, 聚类数k, batch_size, 迭代次数max_iter
初始化: 随机选择k个样本作为初始中心

repeat for iter in 1..max_iter:
    1. 随机采样batch_size个样本
    2. 对每个样本，找最近中心
    3. 累计每个簇的样本和
    4. 更新中心 = 累计和 / 该簇样本数
    5. 可选：重新计算抖动中心
    
until 收敛或达到max_iter

输出: 聚类中心 和 标签
```

### 2.4 更新公式

对于第j个簇，新中心为：

$$c_j = \frac{sum_{x_i \in S_j} x_j}{|S_j|}$$

其中 $S_j$ 是第j簇的样本集合。

---

## 3. 数学公式与推导

### 3.1 目标函数

同标准K-Means，最小化簇内平方误差（SSE）：

$$J = \sum_{j=1}^{k} \sum_{x_i \in c_j} \|x_i - c_j\|^2$$

### 3.2 批量更新 vs 在线更新

标准K-Means（批量）：
$$c_j^{(t+1)} = \frac{1}{|S_j^{(t)}|} \sum_{x_i \in S_j^{(t)}} x_i$$

Mini-Batch使用滑动平均更新：
$$c_j^{(t+1)} = (1 - \alpha) c_j^{(t)} + \alpha \cdot c_j^{new}$$

其中 $\alpha$ 是学习率，通常取0.001。

### 3.3 收敛性保证

使用适当的batch采样和学习率，可以证明算法以概率收敛到局部最优。

---

## 4. 训练过程讲解

### 4.1 初始化方法

| 方法 | 说明 |
|------|------|
| random | 随机选k个样本 |
| k-means++ | 用概率距离加权 |
| scaling | 缩放到标准正态 |

**推荐**：使用k-means++初始化，效果更好。

### 4.2 batch_size选择

| 数据规模 | batch_size |
|----------|-------------|
| < 10k | 100-500 |
| 10k-100k | 500-1000 |
| > 100k | 1000-3000 |

经验法则：batch_size = 3 × k 是个不错的选择。

### 4.3 迭代停止条件

- 达到max_iter
- 中心变化小于tol
- 标签不再变化

### 4.4 调优参数

```python
MiniBatchKMeans(
    n_clusters=8,           # 聚类数
    batch_size=100,          # 批量大小
    max_iter=300,            # 最大迭代
    tol=1e-4,               # 收敛阈值
    init='k-means++',       # 初始化方法
    max_no_change=10,      # 无改善最大轮数
    reassignment_ratio=0.01 # 再分配比例
)
```

---

## 5. 应用场景

### 5.1 大规模数据聚类

- 用户分群（百万级用户）
- 文档聚类（百万级文档）
- 图像聚类

### 5.2 实时聚类

- 流数据聚类
- 在线学习
- 边缘设备部署

### 5.3 对比选择

| 场景 | 推荐 |
|------|------|
| 小数据(<10k) | K-Means |
| 大数据(>10k) | Mini-Batch K-Means |
| 需要精确 | K-Means |
| 需要速度 | Mini-Batch |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 速度快 | batch采样减少计算 |
| 内存友好 | 不需要存全部数据 |
| 大数据适用 | 可处理百万级数据 |
| 在线学习 | 支持增量更新 |
| 可证明收敛 | 理论保证 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 精度略低 | 比标准K-Means略差 |
| 需要调参 | batch_size敏感 |
| 有随机性 | 结果有小幅波动 |
| 不适合高维 | 维度灾难 |

### 6.3 注意事项

- 不要用太小的batch_size（如<10）
- 建议使用k-means++初始化
- 需要更多的迭代次数

---

## 7. 调库实现（Python + scikit-learn）

### 7.1 基本用法

```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# 生成模拟数据
np.random.seed(42)
n_samples = 10000
X = np.vstack([
    np.random.randn(n_samples, 2) + [0, 0],
    np.random.randn(n_samples, 2) + [5, 5],
    np.random.randn(n_samples, 2) + [0, 5]
])

# Mini-Batch K-Means聚类
kmeans = MiniBatchKMeans(
    n_clusters=3,
    batch_size=500,
    random_state=42,
    max_iter=100
)
labels = kmeans.fit_predict(X)

# 中心
centers = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"样本数: {X.shape[0]}")
print(f"聚类数: {3}")
print(f"中心:\n{centers}")
print(f"惯性(SSE): {inertia:.2f}")
```

### 7.2 elbow法确定k

```python
from sklearn.metrics import silhouette_score

inertias = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=500, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(k_range, inertias, 'bo-')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Inertia (SSE)')
axes[0].set_title('Elbow Method')

axes[1].plot(k_range, silhouettes, 'ro-')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')

plt.tight_layout()
plt.savefig('mini_batch_kmeans_selection.png', dpi=100)
plt.show()
```

### 7.3 增量学习

```python
# 模拟流数据
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=100, random_state=42)

# 逐步增量更新
for chunk in np.array_split(X, 100):
    kmeans.partial_fit(chunk)

labels = kmeans.predict(X)
```

### 7.4 性能对比

```python
import time
from sklearn.cluster import KMeans

# 数据
np.random.seed(42)
X = np.random.randn(50000, 10)

# 标准K-Means
start = time.time()
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans.fit(X)
time_kmeans = time.time() - start

# Mini-Batch K-Means
start = time.time()
mb_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=1000, random_state=42, max_iter=100)
mb_kmeans.fit(X)
time_mb = time.time() - start

print(f"K-Means: {time_kmeans:.3f}s")
print(f"Mini-Batch K-Means: {time_mb:.3f}s")
print(f"加速比: {time_kmeans/time_mb:.1f}x")
```

---

## 8. 手工代码实现（核心算法手写）

```python
import numpy as np

class MiniBatchKMeans:
    """小批量K均值聚类 - 手工实现"""
    
    def __init__(self, n_clusters=3, batch_size=100, max_iter=100, 
                 tol=1e-4, init='random', random_state=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        
    def _init_centers(self, X):
        """初始化中心"""
        if self.init == 'random':
            idx = np.random.choice(len(X), self.n_clusters, replace=False)
            return X[idx].copy()
        elif self.init == 'k-means++':
            # 第一个中心随机
            centers = [X[np.random.randint(len(X))]]
            
            # 选择后续中心
            for _ in range(1, self.n_clusters):
                dists = np.array([
                    min(np.sum((x - c)**2) for c in centers)
                    for x in X
                ])
                probs = dists / dists.sum()
                idx = np.random.choice(len(X), p=probs)
                centers.append(X[idx])
            return np.array(centers)
            
    def _compute_distances(self, X, centers):
        """计算样本到中心的距离"""
        # [n_samples, n_centers]
        dists = np.zeros((len(X), self.n_clusters))
        for j, c in enumerate(centers):
            dists[:, j] = np.sum((X - c)**2, axis=1)
        return dists
    
    def fit(self, X):
        """训练"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = len(X)
        
        # 初始化中心
        self.cluster_centers_ = self._init_centers(X)
        
        for iteration in range(self.max_iter):
            # 随机采样batch
            batch_idx = np.random.choice(
                n_samples, 
                min(self.batch_size, n_samples), 
                replace=False
            )
            X_batch = X[batch_idx]
            
            # 分配到最近中心
            dists = self._compute_distances(X_batch, self.cluster_centers_)
            labels_batch = np.argmin(dists, axis=1)
            
            # 更新中心
            new_centers = np.zeros_like(self.cluster_centers_)
            for j in range(self.n_clusters):
                mask = labels_batch == j
                if mask.sum() > 0:
                    new_centers[j] = X_batch[mask].mean(axis=0)
                else:
                    new_centers[j] = self.cluster_centers_[j]
            
            # 检查收敛
            change = np.sum(np.abs(new_centers - self.cluster_centers_))
            self.cluster_centers_ = new_centers
            
            if change < self.tol:
                print(f"迭代 {iteration+1} 收敛")
                break
        
        # 计算最终标签
        dists = self._compute_distances(X, self.cluster_centers_)
        self.labels_ = np.argmin(dists, axis=1)
        self.inertia_ = np.min(dists, axis=1).sum()
        
        return self
    
    def predict(self, X):
        """预测"""
        dists = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(dists, axis=1)
    
    def fit_predict(self, X):
        """训练+预测"""
        self.fit(X)
        return self.labels_
    
    def partial_fit(self, X):
        """增量学习"""
        if self.cluster_centers_ is None:
            self.cluster_centers_ = self._init_centers(X)
        
        # 对当前batch分配
        dists = self._compute_distances(X, self.cluster_centers_)
        labels = np.argmin(dists, axis=1)
        
        # 更新中心
        for j in range(self.n_clusters):
            mask = labels == j
            if mask.sum() > 0:
                self.cluster_centers_[j] = (
                    0.999 * self.cluster_centers_[j] + 
                    0.001 * X[mask].mean(axis=0)
                )
        
        return self


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成测试数据
    X = np.vstack([
        np.random.randn(1000, 2) + [0, 0],
        np.random.randn(1000, 2) + [5, 5],
        np.random.randn(1000, 2) + [0, 5]
    ])
    
    # 手工实现
    kmeans = MiniBatchKMeans(n_clusters=3, batch_size=500, max_iter=100, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # sklearn实现
    from sklearn.cluster import MiniBatchKMeans
    kmeans_sklearn = MiniBatchKMeans(n_clusters=3, batch_size=500, random_state=42)
    labels_sklearn = kmeans_sklearn.fit_predict(X)
    
    print("手工实现标签分布:", np.bincount(labels))
    print("sklearn标签分布:", np.bincount(labels_sklearn))
    print("手工中心:\n", kmeans.cluster_centers_)
    print("sklearn中心:\n", kmeans_sklearn.cluster_centers_)
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
X = np.vstack([
    np.random.randn(500, 2) + [0, 0],
    np.random.randn(500, 2) + [5, 3],
    np.random.randn(500, 2) + [2, 5]
])

# 聚类
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=200, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 聚类结果
for j in range(3):
    mask = labels == j
    axes[0].scatter(X[mask, 0], X[mask, 1], c=colors[j], alpha=0.6, s=20)
    axes[0].scatter(centers[j, 0], centers[j, 1], c=colors[j], s=200, marker='x', linewidths=3)

axes[0].set_title('Mini-Batch K-Means 聚类结果')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')

# 迭代过程
inertias = []
for batch_size in [50, 100, 200, 500]:
    kmeans = MiniBatchKMeans(n_clusters=3, batch_size=batch_size, max_iter=50, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)

axes[1].bar(['50', '100', '200', '500'], inertias, color='steelblue')
axes[1].set_title('不同batch_size的SSE')
axes[1].set_xlabel('batch_size')
axes[1].set_ylabel('SSE (Inertia)')

plt.tight_layout()
plt.savefig('mini_batch_kmeans_demo.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 公式 | 范围 |
|------|------|------|
| SSE | $\sum \|x-c\|^2$ | 越小越好 |
| Silhouette | $(b-a)/max(a,b)$ | [-1,1],越大越好 |
| Calinski-Harabasz | $BGS/(k-1)/(n-k)/WGS$ | 越大越好 |

### 10.2 如何评估聚类质量？

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 轮廓系数
silhouette = silhouette_score(X, labels)
print(f"轮廓系数: {silhouette:.3f}")

# CH指数
ch = calinski_harabasz_score(X, labels)
print(f"CH指数: {ch:.1f}")
```

---

## 11. 常见问题与易错点

### Q1: 如何选择batch_size？

**答案**：batch_size = 3×k 是好起点，然后根据效果调整。

### Q2: 结果不稳定怎么办？

**答案**：设置random_state，多跑几次取最优。

### Q3: 比标准K-Means差多少？

**答案**：通常SSE高5-10%，但速度快10-100倍。

### Q4: 如何判断收敛？

**答案**：中心变化小于tol，或标签稳定。

### Q5: 适合高维数据吗？

**答案**：可以，但建议先用PCA降维。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心思想 | 随机小批量更新 |
| 速度优势 | batch_size决定 |
| 精度损失 | 通常<10% |
| 参数调优 | batch_size, n_clusters |

### 12.2 公式汇总

每批更新公式：
$$c_j^{(new)} = \frac{1}{|B_j|} \sum_{x_i \in B_j} x_i$$

目标函数（SSE）：
$$J = \sum_{j=1}^{k} \sum_{x_i \in c_j} \|x_i - c_j\|^2$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. 增大batch_size会：
   - A) 速度变慢，精度变高
   - B) 速度变快，精度变低
   - C) 速度不变，精度变高
   - D) 速度变慢，精度变低

2. Mini-Batch K-Means适用场景是：
   - A) 小数据集
   - B) 大规模数据
   - C) 需要精确结果
   - D) 高维稀疏数据

### 13.2 简答题

1. 为什么Mini-Batch K-Means比标准K-Means快？
2. batch_size过小有什么问题？

### 13.3 编程题

1. 实现elbow法自动选择最优k。
2. 比较不同batch_size对聚类质量的影响。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
统计基础
    ↓
K-Means原理
    ↓
Mini-Batch优化
    ↓
大规模应用
    ↓
流数据聚类
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| K-Means | 标准版本 |
| K-Means++ | 改进初始化 |
| DBSCAN | 密度聚类 |
| 层次聚类 | 树状结构 |

### 14.3 扩展阅读

- 论文：Sculley, 2010 - Web-scale K-Means clustering

---

## 附录

### 参考

1. Sculley, D. (2010). Web-scale K-Means clustering. WWW.
2. sklearn.cluster.MiniBatchKMeans 文档

---

**文档结束**