# K-Means 学习文档

## 1. 算法基础认知

K-Means 是最经典的无监督聚类算法，由 Stuart Lloyd 于 1957 年提出。其目标是将 $n$ 个样本划分到 $K$ 个簇中，使得每个样本到其所属簇中心的距离之和最小。K-Means 属于划分式聚类方法，简单高效，是数据挖掘和广告算法中最常用的聚类工具之一。

## 2. 核心原理

K-Means 基于一个核心假设：**每个簇可以用其质心（centroid）代表**。算法通过交替执行两个步骤来优化：

1. **分配步骤（Assignment）**：将每个样本分配到最近的质心所属的簇
2. **更新步骤（Update）**：重新计算每个簇的质心（取簇内样本均值）

这两个步骤不断交替，直到收敛（质心不再变化或变化极小）。这正是 **Lloyd 算法** 的核心流程。

## 3. 数学公式与推导

**目标函数（惯性/Inertia）**：

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$

其中 $C_k$ 表示第 $k$ 个簇，$\mu_k$ 为其质心。

**最优分配**（固定 $\mu_k$，优化分配）：

$$c_i = \arg\min_k \| x_i - \mu_k \|^2$$

**最优质心**（固定分配，优化 $\mu_k$）：

$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

可以证明，每个更新步骤都会使 $J$ 单调递减，因此算法必然收敛。

## 4. 训练过程讲解

**k-means++ 初始化**（避免随机初始化导致收敛到差的局部最优）：

1. 随机选择第一个质心 $\mu_1$
2. 对每个样本 $x$，计算其到最近已选质心的距离 $D(x)$
3. 以概率 $P(x) \propto D(x)^2$ 选择下一个质心
4. 重复直到选出 $K$ 个质心

**完整训练流程**：
1. 用 k-means++ 初始化 $K$ 个质心
2. 分配每个样本到最近质心
3. 重新计算质心
4. 重复 2-3 直到收敛或达到最大迭代次数

## 5. 应用场景

- 用户画像分群（广告系统中对受众聚类）
- 广告点击行为聚类，发现用户兴趣模式
- 图像压缩（颜色量化）
- 文本聚类与主题发现
- 异常检测（远离所有簇中心的样本）

## 6. 优缺点分析

**优点**：
- 算法简单，易于实现和理解
- 计算效率高，时间复杂度 $O(nKt)$，$t$ 为迭代次数
- 适合大规模数据集

**缺点**：
- 需要预先指定 $K$ 值
- 对初始化敏感，可能收敛到局部最优
- 只能发现凸形簇，不适合非球形分布
- 对噪声和异常值敏感

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def _init_centroids(self, X):
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[idx].copy()

    def _assign(self, X, centroids):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        return np.argmin(dists, axis=1)

    def _update(self, X, labels):
        return np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

    def fit(self, X):
        self.centroids = self._init_centroids(X)
        for _ in range(self.max_iter):
            self.labels = self._assign(X, self.centroids)
            new_centroids = self._update(X, self.labels)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return self

    def predict(self, X):
        return self._assign(X, self.centroids)

km = KMeansScratch(n_clusters=4)
km.fit(X)
print("Hand-written K-Means centroids:\n", km.centroids)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, edgecolors='black', linewidths=1.5)
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig('kmeans_result.png', dpi=150)
plt.show()
```

关键可视化内容：样本点按簇着色，红色叉号为簇中心。

## 10. 模型评估

**肘部法则（Elbow Method）**：绘制不同 $K$ 值下的 Inertia 曲线，选择拐点处的 $K$。

```python
inertias = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

**轮廓系数（Silhouette Score）**：取值 $[-1, 1]$，越接近 1 表示聚类效果越好。

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

其中 $a(i)$ 是样本 $i$ 到同簇其他样本的平均距离，$b(i)$ 是到最近其他簇的平均距离。

## 11. 常见问题与易错点

- **K 值选择不当**：K 过小导致簇过于粗糙，K 过大导致过拟合。应结合肘部法则和业务经验
- **未做特征标准化**：K-Means 基于欧氏距离，特征尺度不同会严重影响结果，务必先做 StandardScaler
- **数据存在异常值**：异常值会严重拉偏质心，可考虑使用 K-Medoids
- **非凸形数据**：K-Means 无法处理月牙形等非凸分布，可使用 DBSCAN 或谱聚类

## 12. 学习总结

K-Means 是聚类分析的入门必学算法。其核心思想简单——交替优化分配和质心，但背后涉及 EM 思想的雏形。掌握 K-Means 后，应延伸学习 k-means++ 初始化、Mini-Batch K-Means（大规模数据）以及 DBSCAN（密度聚类）等变体。

## 13. 练习题与思考题（含答案）

**Q1**：K-Means 的时间复杂度是多少？若数据量增大 10 倍，运行时间大约增加多少？

> 答：$O(nKt)$。数据量增大 10 倍，运行时间大约增加 10 倍（线性增长）。

**Q2**：为什么 k-means++ 比随机初始化更好？

> 答：k-means++ 让初始质心尽量分散，避免多个质心初始化在同一个簇中，从而降低收敛到差局部最优的风险，通常能以更少迭代达到更好的结果。

**Q3**：K-Means 与 EM 算法有什么联系？

> 答：K-Means 可视为 EM 算法的特例——硬分配版本的 EM。分配步骤对应 E 步（计算隐变量的期望），更新步骤对应 M 步（最大化似然）。

## 14. 学习路径建议

- **前置知识**：欧氏距离、均值计算、基础优化思想
- **下一步学习**：高斯混合模型（GMM + EM）、DBSCAN、层次聚类、谱聚类
- **进阶方向**：Mini-Batch K-Means（流式数据）、K-Means 在推荐系统中的用户分群应用
