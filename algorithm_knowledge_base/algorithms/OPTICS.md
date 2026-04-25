# OPTICS 学习文档

> 密度聚类的改进，密度层次聚类

---

## 1. 算法基础认知

### 1.1 一句话定义

OPTICS（Ordering Points To Identify the Clustering Structure）是Ankerst等人在1999年提出的密度聚类算法，作为DBSCAN的改进，能自动发现不同密度的聚类结构，无需预设邻域半径。

### 1.2 直觉类比

OPTICS就像"用等高线画地形图"。想象你在看一张包含不同山丘的地图：
- DBSCAN需要预设一个固定的"海拔高度"（eps半径）来看哪些区域是山
- 但不同山的高度不同！你预设1500米，可能有的山只有800米高，就看不见了

OPTICS的做法是：先画出所有等高线（密度等高线），然后你想看哪个高度的"山"（聚类）都可以！

### 1.3 发展背景

- 1999年，Ankerst等人在SIGMOD发表论文"OPTICS: ordering points to identify the clustering structure"
- 是DBSCAN的改进，解决DBSCAN对参数敏感的问题
- 核心贡献：提出"密度等高线"概念

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 聚类 → 密度聚类 |
| 输出 | 有序点 + 可达距离 |
| 方法 | 密度层次 |
| 特点 | 自动发现聚类边界 |

---

## 2. 核心原理

### 2.1 为什么需要OPTICS？

**DBSCAN的问题**：固定eps半径

```
场景：数据有高密度区域和低密度区域
    ▲ 稀疏区域需要大eps才能聚类
    │●●
    │●●●●  密集区域
    │●●●●●●
    └─────────
    
DBSCAN: eps=小 → 只发现密集区
       eps=大 → 稀疏区连成一片

OPTICS: 自动找到每个区域的合适密度
```

### 2.2 vs DBSCAN对比

| 方面 | DBSCAN | OPTICS |
|------|--------|-------|
| 半径eps | 固定需预设 | 自动计算 |
| 聚类发现 | 直接输出 | 后处理提取 |
| 密度变化 | 敏感 | 不敏感 |
| 参数依赖 | 高 | 低 |
| 输出 | 聚类标签 | 有序+可达图 |

### 2.3 核心概念

**核心点**：在邻域ε内至少有MinPts个邻居的点

**核心距离**：使点p成为核心点的最小邻域距离
$$core\_dist_{\epsilon}(p) = d(p, N^{MinPts}(p))$$

**可达距离**：从点o到点p的可达距离
$$reachability\_dist_{\epsilon}(p) = \max(core\_dist_{\epsilon}(p), dist(p, o))$$

**可达图**：可达距离随点顺序的曲线

### 2.4 算法流程

```
Step 1: 对每个点，找到其MinPts最近邻
Step 2: 计算核心距离
Step 3: 按密度排序点（OPTICS排序）
Step 4: 输出可达图
Step 5: 后处理提取聚类（在可达图上找谷）
```

---

## 3. 数学公式与推导

### 3.1 核心距离

定义：使点p成为核心点的最小ε：

$$core\_dist_{\epsilon}(p) = \begin{cases} \text{undefined} & \text{if } |N_{\epsilon}(p)| < MinPts \\ d(p, N^{MinPts}(p)) & \text{otherwise} \end{cases}$$

其中$N^{MinPts}(p)$是p的第MinPts个最近邻。

### 3.2 可达距离

从种子点o到点p的可达距离：

$$reachability\_dist_{\epsilon}(p) = \begin{cases} \text{undefined} & \text{if } core\_dist_{\epsilon}(p) \text{ is undefined} \\ \max(core\_dist_{\epsilon}(p), dist(p, o)) & \text{otherwise} \end{cases}$$

### 3.3 OPTICS排序

算法保持一个优先队列（ seeds），按可达距离排序：

```
1. 随机选一个未访问点作为种子
2. 将其可达距离设为undefined
3. 对于其邻域内的每个点：
   - 如果未处理，计算可达距离
   - 加入seeds队列
4. 取出seeds中可达距离最小的点
5. 重复直到所有点处理完
```

### 3.4 聚类提取

在可达图上找"���"（波谷）：

```
xi方法：切割可达距离 > xi的点
       形成独立聚类

dbscan方法：切割eps内的连续区域
           形成聚类
```

---

## 4. 训练过程讲解

### 4.1 参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| min_samples | 核心点数 | 5-10 |
| max_eps | 最大可达距离 | 数据决定 |
| xi | 聚类边界敏感度 | 0.05 |
| metric | 距离类型 | euclidean |

### 4.2 xi vs eps

```python
# xi方法（推荐）
optics = OPTICS(min_samples=5, xi=0.05)
labels = optics.fit_predict(X)

# 等价DBSCAN方法
optics = OPTICS(min_samples=5, max_eps=0.5)
labels = optics.fit_predict(X)
```

### 4.3 sklearn使用

```python
from sklearn.cluster import OPTICS
import numpy as np

# 生成测试数据
np.random.seed(42)
X = np.concatenate([
    np.random.randn(100, 2) + [0, 0],
    np.random.randn(100, 2) + [5, 5],
    np.random.randn(50, 2) + [2, 5]
])

# OPTICS聚类
clustering = OPTICS(min_samples=10, xi=0.05)
labels = clustering.fit_predict(X)

# 获取可达距离
reachability = clustering.reachability_
ordering = clustering.ordering_

print(f"聚类标签: {np.unique(labels)}")
print(f"可达距离范围: [{reachability.min():.3f}, {reachability.max():.3f}]")
```

---

## 5. 应用场景

### 5.1 密度不同的数据

```python
# 不同密度的聚类
import numpy as np

# 高密度簇
cluster1 = np.random.randn(200, 2) * 0.3 + [0, 0]

# 低密度簇
cluster2 = np.random.randn(50, 2) * 1.5 + [10, 10]

# 噪声
noise = np.random.randn(30, 2) * 3 + [5, 5]

X = np.vstack([cluster1, cluster2, noise])

# OPTICS可以发现不同密度的簇
optics = OPTICS(min_samples=10, xi=0.03)
labels = optics.fit_predict(X)
```

### 5.2 层次聚类

OPTICS天然支持层次聚类：

```python
# 不同阈值提取不同层次
for xi in [0.01, 0.03, 0.05, 0.1]:
    optics = OPTICS(min_samples=10, xi=xi)
    labels = optics.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"xi={xi}: {n_clusters} clusters")
```

### 5.3 对比选择

| 场景 | 推荐 |
|------|------|
| 密度均匀 | DBSCAN |
| 密度不均 | OPTICS |
| 层次结构 | OPTICS |
| 高维数据 | HDBSCAN |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 自动发现参数 | 无需预设eps |
| 密度不变 | 适应不同密度 |
| 层次结构 | 可提取多层次 |
| 鲁棒 | 对参数不敏感 |
| 可视化 | 可达图直观 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 计算复杂 | O(n log n) |
| 内存消耗 | 需存储k近邻 |
| 解释复杂 | 需理解可达图 |
| 不适合高维 | 维数灾难 |

### 6.3 注意事项

- min_samples越大，核心距离越大
- xi越小，聚类越多
- 需要先看可达图

---

## 7. 调库实现（Python + scikit-learn）

### 7.1 基本用法

```python
import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt

# 生成多密度数据
np.random.seed(42)
X1 = np.random.randn(300, 2) * 0.5 + [0, 0]
X2 = np.random.randn(50, 2) * 2 + [8, 8]
X = np.vstack([X1, X2])

# OPTICS聚类
optics = OPTICS(min_samples=10, xi=0.05)
labels = optics.fit_predict(X)

# 可达图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
plt.title('OPTICS聚类结果')

plt.subplot(1, 2, 2)
plt.plot(optics.ordering_, optics.reachability_, 'b-', linewidth=0.5)
plt.xlabel('Point Index')
plt.ylabel('Reachability Distance')
plt.title('可达图')

plt.tight_layout()
plt.savefig('optics_demo.png', dpi=100)
plt.show()
```

### 7.2 聚类提取方法

```python
# 方法1: xi方法（默认）
optics_xi = OPTICS(min_samples=10, xi=0.05)
labels_xi = optics_xi.fit_predict(X)

# 方法2: max_eps方法
optics_eps = OPTICS(min_samples=10, max_eps=0.8)
labels_eps = optics_eps.fit_predict(X)

# 方法3: DBSCAN-like
# 设置max_eps为DBSCAN的eps
optics_dbscan = OPTICS(min_samples=5, max_eps=0.5)
labels_dbscan = optics_dbscan.fit_predict(X)

print(f"xi方法: {len(set(labels_xi)) - (1 if -1 in labels_xi else 0)} clusters")
print(f"eps方法: {len(set(labels_eps)) - (1 if -1 in labels_eps else 0)} clusters")
```

### 7.3 参数调优

```python
from sklearn.metrics import silhouette_score

# 网格搜索
results = []
for min_samples in [5, 10, 15, 20]:
    for xi in [0.01, 0.03, 0.05, 0.1]:
        optics = OPTICS(min_samples=min_samples, xi=xi)
        labels = optics.fit_predict(X)
        
        # 排除噪声
        mask = labels != -1
        if mask.sum() > 0 and len(set(labels[mask])) > 1:
            score = silhouette_score(X[mask], labels[mask])
            results.append({
                'min_samples': min_samples,
                'xi': xi,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'score': score
            })

# 结果
import pandas as pd
df = pd.DataFrame(results)
print(df.sort_values('score', ascending=False).head(10))
```

---

## 8. 手工代码实现（核心算法手写）

```python
import numpy as np
from heapq import heappush, heappop

class OPTICS:
    """OPTICS聚类 - 手工实现"""
    
    def __init__(self, min_samples=5, xi=0.05, max_eps=np.inf):
        self.min_samples = min_samples
        self.xi = xi
        self.max_eps = max_eps
        self.reachability_ = None
        self.ordering_ = None
        self.labels_ = None
    
    def _compute_core_dist(self, X):
        """计算每个点的核心距离"""
        n = len(X)
        k = self.min_samples
        
        # 计算k近邻距离
        core_dists = np.zeros(n)
        
        for i in range(n):
            dists = np.sum((X - X[i])**2, axis=1)
            dists[i] = np.inf
            kth_dist = np.partition(dists, k-1)[k-1]
            core_dists[i] = np.sqrt(kth_dist)
        
        return core_dists
    
    def fit(self, X):
        """OPTICS排序"""
        n = len(X)
        
        # 核心距离
        self.core_dists_ = self._compute_core_dist(X)
        
        # 初始化
        processed = np.zeros(n, dtype=bool)
        self.reachability_ = np.full(n, np.inf)
        self.ordering_ = []
        
        # 优先队列: (可达距离, 点索引)
        seeds = []
        
        for i in range(n):
            if not processed[i]:
                # 设为undefined
                self.reachability_[i] = np.inf
                
                # 扩展 cluster-order
                self.ordering_.append(i)
                processed[i] = True
                
                # 找到ε邻域
                dists = np.sum((X - X[i])**2, axis=1)
                neighbors = np.where(dists <= self.core_dists_[i])[0]
                
                # 处理邻域
                for j in neighbors:
                    if not processed[j]:
                        # 计算可达距离
                        reach_dist = max(self.core_dists_[i], np.sqrt(dists[j]))
                        
                        if reach_dist <= self.max_eps:
                            if self.reachability_[j] == np.inf:
                                # 首次加入
                                heappush(seeds, (reach_dist, j))
                            elif reach_dist < self.reachability_[j]:
                                # 更新
                                self.reachability_[j] = reach_dist
                                heappush(seeds, (reach_dist, j))
                
                # 处理seeds队列
                while seeds:
                    reach_dist, idx = heappop(seeds)
                    
                    if not processed[idx]:
                        self.ordering_.append(idx)
                        processed[idx] = True
                        
                        # 更新可达距离
                        dists = np.sum((X - X[idx])**2, axis=1)
                        neighbors = np.where(dists <= self.core_dists_[idx])[0]
                        
                        for j in neighbors:
                            if not processed[j]:
                                r = max(self.core_dists_[idx], np.sqrt(dists[j]))
                                if r <= self.max_eps:
                                    if self.reachability_[j] == np.inf:
                                        heappush(seeds, (r, j))
                                    elif r < self.reachability_[j]:
                                        self.reachability_[j] = r
                                        heappush(seeds, (r, j))
        
        self.ordering_ = np.array(self.ordering_)
        return self
    
    def _extract_clusters(self, X):
        """从可达图提取聚类"""
        if self.xi is None or self.xi == 0:
            return
        
        n = len(X)
        labels = np.full(n, -1)
        
        # 可达距离阈值
        threshold = self.xi * np.max(self.reachability_[np.isfinite(self.reachability_)])
        
        cluster_id = 0
        current_cluster = []
        
        for i, idx in enumerate(self.ordering_):
            if self.reachability_[i] > threshold or self.reachability_[i] == np.inf:
                # 聚类边界
                if len(current_cluster) > self.min_samples:
                    labels[np.array(current_cluster)] = cluster_id
                    cluster_id += 1
                current_cluster = []
            else:
                current_cluster.append(idx)
        
        # 最后一个聚类
        if len(current_cluster) > self.min_samples:
            labels[np.array(current_cluster)] = cluster_id
        
        self.labels_ = labels
        return labels
    
    def fit_predict(self, X):
        """拟合+提取"""
        self.fit(X)
        self._extract_clusters(X)
        return self.labels_
    
    def predict(self, X):
        """预测"""
        return self.labels_


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成数据
    X1 = np.random.randn(200, 2) + [0, 0]
    X2 = np.random.randn(100, 2) + [5, 5]
    X = np.vstack([X1, X2])
    
    # 手工实现
    optics = OPTICS(min_samples=10, xi=0.05)
    labels = optics.fit_predict(X)
    
    print(f"聚类标签: {np.unique(labels)}")
    print(f"可达距离: {optics.reachability_[:10]}")
```

---

## 9. 可视化与结果理解

### 9.1 可达图解读

```python
import matplotlib.pyplot as plt
import numpy as np

# 可达图解读
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 数据
np.random.seed(42)
X = np.vstack([
    np.random.randn(200, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 5],
    np.random.randn(50, 2) * 3 + [5, 0]
])

# OPTICS
from sklearn.cluster import OPTICS
optics = OPTICS(min_samples=10, xi=0.05)
labels = optics.fit_predict(X)

# 可达图
axes[0, 0].plot(optics.ordering_, optics.reachability_, 'b-', linewidth=0.5)
axes[0, 0].set_xlabel('Point Index')
axes[0, 0].set_ylabel('Reachability Distance')
axes[0, 0].set_title('OPTICS可达图')

# 聚类结果
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'gray']
for i in range(len(set(labels)) - (1 if -1 in labels else 0)):
    mask = labels == i
    axes[0, 1].scatter(X[mask, 0], X[mask, 1], c=colors[i], s=20, alpha=0.7)

# 噪声
noise = labels == -1
axes[0, 1].scatter(X[noise, 0], X[noise, 1], c='gray', s=10, alpha=0.3)
axes[0, 1].set_title('聚类结果')

# 不同xi对比
for j, xi in enumerate([0.01, 0.05]):
    optics = OPTICS(min_samples=10, xi=xi)
    labels = optics.fit_predict(X)
    axes[1, j].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
    axes[1, j].set_title(f'xi={xi}')

plt.tight_layout()
plt.savefig('optics_analysis.png', dpi=100)
plt.show()
```

### 9.2 不同min_samples

```python
# min_samples影响
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, ms in enumerate([5, 10, 15, 20]):
    optics = OPTICS(min_samples=ms, xi=0.05)
    labels = optics.fit_predict(X)
    
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
    axes[i].set_title(f'min_samples={ms}')

plt.tight_layout()
plt.savefig('optics_min_samples.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| 轮廓系数 | 聚类质量 |
| CH指数 | 聚类分离度 |
| Dav-Bouldin | 聚类紧密度 |

### 10.2 评估代码

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

mask = labels != -1
if mask.sum() > 0:
    silhouette = silhouette_score(X[mask], labels[mask])
    ch = calinski_harabasz_score(X[mask], labels[mask])
    
    print(f"轮廓系数: {silhouette:.3f}")
    print(f"CH指数: {ch:.1f}")
```

---

## 11. 常见问题与易错点

### Q1: xi如何选择？

**答案**：xi越大，聚类越少。通常0.01-0.1。

### Q2: 可达图怎么看聚类？

**答案**：找"谷"——可达距离突然下降的位置。

### Q3: 为什么有些点没标签？

**答案**：那些点是噪声，可在可达图上看成很高或undefined。

### Q4: OPTICS和HDBSCAN关系？

**答案**：HDBSCAN是OPTICS的更快实现。

### Q5: 参数和DBSCAN的关系？

**答案**：OPTICS的max_eps类似DBSCAN的eps。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心思想 | 密���等���线 |
| 公式 | 可达距离 |
| 输出 | 可达图+聚类 |
| 参数 | min_samples, xi |

### 12.2 公式汇总

核心距离：
$$core\_dist(p) = d(p, N^{MinPts}(p))$$

可达距离：
$$reachability(p) = \max(core\_dist(p), dist(p, seed))$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. OPTICS和DBSCAN的主要区别是：
   - A) OPTICS更快
   - B) OPTICS不需要预设eps
   - C) OPTICS需要更多参数

2. 可达图上的"谷"表示：
   - A) 聚类边界
   - B) 噪声
   - C) 中心

### 13.2 简答题

1. 为什么OPTICS能发现不同密度的聚类？
2. 如何从可达图提取聚类？

### 13.3 编程题

1. 实现基于OPTICS的层次聚类。
2. 比较OPTICS和HDBSCAN的效果。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
统计基础
    ↓
密度概念
    ↓
DBSCAN
    ↓
OPTICS
    ↓
HDBSCAN
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| DBSCAN | 固定参数版 |
| HDBSCAN | 快速版 |
| 层次聚类 | 树状版 |

### 14.3 扩展阅读

- Ankerst et al. (1999). OPTICS: ordering points to identify the clustering structure. SIGMOD.

---

## 附录

### 参考

1. Ankerst, M., et al. (1999). OPTICS: ordering points to identify the clustering structure. SIGMOD.
2. sklearn.cluster.OPTICS 文档

---

**文档结束**