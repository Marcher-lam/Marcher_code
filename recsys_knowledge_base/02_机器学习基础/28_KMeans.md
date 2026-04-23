# KMeans 聚类 学习文档

## 1. 算法基础认知

### 1.1 什么是 KMeans？

KMeans 是一种**无监督学习**算法，将数据划分为 K 个不同的簇（cluster），使得同一簇内的数据点尽可能相似，不同簇的数据点尽可能不同。

### 1.2 应用场景

- 用户分群
- 物品聚类
- 推荐系统中的用户/物品分组
- 异常检测

### 1.3 核心思想

```
目标：最小化簇内平方和（Within-Cluster Sum of Squares, WCSS）

WCSS = Σ Σ ||x - μ_k||²
       k i∈C_k
```

## 2. 算法原理

### 2.1 算法流程

```
1. 随机初始化 K 个中心点
2. 重复直到收敛:
   a. 分配：将每个点分配到最近的中心点
   b. 更新：重新计算每个簇的中心点
```

### 2.2 完整实现

```python
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class KMeans:
    """
    KMeans 聚类算法
    """

    def __init__(self, n_clusters: int = 3, max_iters: int = 100,
                 tol: float = 1e-4, n_init: int = 10, random_state: int = None):
        """
        参数:
            n_clusters: 簇数量 K
            max_iters: 最大迭代次数
            tol: 收敛阈值
            n_init: 随机初始化次数（选最优）
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.centroids = None
        self.labels = None
        self.inertia_ = None  # WCSS

    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        训练 KMeans

        参数:
            X: 数据矩阵 (n_samples, n_features)
        """
        if self.random_state:
            np.random.seed(self.random_state)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            # 单次 KMeans
            centroids, labels, inertia = self._fit_single(X)

            # 保留最优结果
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia_ = best_inertia

        return self

    def _fit_single(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """单次 KMeans 训练"""
        n_samples = X.shape[0]

        # 随机初始化中心点
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[idx].copy()

        for _ in range(self.max_iters):
            # 分配点到最近的中心
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            # 更新中心点
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # 空簇：随机重新初始化
                    new_centroids[k] = X[np.random.randint(n_samples)]

            # 检查收敛
            if np.max(np.abs(new_centroids - centroids)) < self.tol:
                break

            centroids = new_centroids

        # 计算最终 WCSS
        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = sum(
            np.sum((X[labels == k] - centroids[k]) ** 2)
            for k in range(self.n_clusters)
        )

        return centroids, labels, inertia

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        计算所有点到所有中心点的距离

        参数:
            X: (n_samples, n_features)
            centroids: (n_clusters, n_features)

        返回:
            distances: (n_samples, n_clusters)
        """
        # 向量化计算
        distances = np.zeros((X.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)

        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测簇标签

        参数:
            X: 数据矩阵

        返回:
            labels: 簇标签
        """
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换为距离矩阵

        参数:
            X: 数据矩阵

        返回:
            distances: 到各中心的距离
        """
        return np.sqrt(self._compute_distances(X, self.centroids))

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """训练并预测"""
        self.fit(X)
        return self.labels


class KMeansPlusPlus(KMeans):
    """
    KMeans++ 初始化

    使用更智能的初始化方法，选择距离已选中心较远的点
    """

    def _fit_single(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """使用 KMeans++ 初始化的单次训练"""
        n_samples = X.shape[0]

        # KMeans++ 初始化
        centroids = self._kmeans_plus_plus_init(X)

        # 后续迭代
        for _ in range(self.max_iters):
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = X[np.random.randint(n_samples)]

            if np.max(np.abs(new_centroids - centroids)) < self.tol:
                break

            centroids = new_centroids

        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = sum(
            np.sum((X[labels == k] - centroids[k]) ** 2)
            for k in range(self.n_clusters)
        )

        return centroids, labels, inertia

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """KMeans++ 初始化"""
        n_samples = X.shape[0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        # 随机选择第一个中心
        centroids[0] = X[np.random.randint(n_samples)]

        # 选择后续中心
        for k in range(1, self.n_clusters):
            # 计算每个点到最近中心的距离
            distances = self._compute_distances(X, centroids[:k])
            min_distances = np.min(distances, axis=1)

            # 按距离平方的概率选择
            probs = min_distances / min_distances.sum()
            centroids[k] = X[np.random.choice(n_samples, p=probs)]

        return centroids


# 使用示例
def demo_kmeans():
    """KMeans 示例"""
    # 生成数据
    np.random.seed(42)

    # 3 个簇
    cluster1 = np.random.randn(100, 2) + [2, 2]
    cluster2 = np.random.randn(100, 2) + [-2, -2]
    cluster3 = np.random.randn(100, 2) + [2, -2]

    X = np.vstack([cluster1, cluster2, cluster3])

    # 训练 KMeans
    print("训练 KMeans...")
    kmeans = KMeansPlusPlus(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    print(f"WCSS: {kmeans.inertia_:.4f}")

    # 可视化
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
    plt.title('原始数据')

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
               c='red', marker='x', s=200, linewidths=3)
    plt.title('KMeans 聚类结果')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_kmeans()
```

## 3. 确定最优 K 值

### 3.1 肘部法则

```python
def elbow_method(X: np.ndarray, k_range: range = range(1, 11)):
    """
    肘部法则确定 K

    参数:
        X: 数据
        k_range: K 值范围
    """
    inertias = []

    for k in k_range:
        kmeans = KMeansPlusPlus(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('K')
    plt.ylabel('WCSS')
    plt.title('肘部法则')
    plt.show()

    return inertias
```

### 3.2 轮廓系数

```python
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    计算轮廓系数

    参数:
        X: 数据
        labels: 簇标签

    返回:
        平均轮廓系数
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1 or n_clusters == n_samples:
        return 0

    scores = np.zeros(n_samples)

    for i in range(n_samples):
        # 同簇平均距离 (a)
        same_cluster = X[labels == labels[i]]
        a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))

        # 最近其他簇平均距离 (b)
        b = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = X[labels == label]
                dist = np.mean(np.linalg.norm(other_cluster - X[i], axis=1))
                b = min(b, dist)

        # 轮廓系数
        scores[i] = (b - a) / max(a, b)

    return np.mean(scores)


def find_optimal_k_silhouette(X: np.ndarray, k_range: range = range(2, 11)):
    """使用轮廓系数找最优 K"""
    scores = []

    for k in k_range:
        kmeans = KMeansPlusPlus(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"K={k}, Silhouette={score:.4f}")

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, scores, 'bo-')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('轮廓系数')
    plt.show()

    optimal_k = k_range[np.argmax(scores)]
    print(f"最优 K: {optimal_k}")

    return optimal_k, scores
```

## 4. 推荐系统应用

### 4.1 用户聚类

```python
class UserClustering:
    """
    用户聚类

    将用户按行为特征聚类
    """

    def __init__(self, n_clusters: int = 5):
        self.kmeans = KMeansPlusPlus(n_clusters=n_clusters)
        self.user_features = None

    def fit(self, user_features: np.ndarray):
        """
        训练用户聚类

        参数:
            user_features: 用户特征矩阵 (n_users, n_features)
        """
        self.user_features = user_features
        self.kmeans.fit(user_features)
        return self

    def get_user_cluster(self, user_features: np.ndarray) -> np.ndarray:
        """获取用户的簇"""
        return self.kmeans.predict(user_features)

    def get_cluster_centroids(self) -> np.ndarray:
        """获取簇中心"""
        return self.kmeans.centroids

    def get_similar_users(self, user_idx: int, top_k: int = 10) -> List[int]:
        """获取同簇的相似用户"""
        user_cluster = self.kmeans.labels[user_idx]
        same_cluster_users = np.where(self.kmeans.labels == user_cluster)[0]
        same_cluster_users = same_cluster_users[same_cluster_users != user_idx]

        return same_cluster_users[:top_k].tolist()
```

### 4.2 物品聚类

```python
class ItemClustering:
    """
    物品聚类

    将物品按特征聚类
    """

    def __init__(self, n_clusters: int = 10):
        self.kmeans = KMeansPlusPlus(n_clusters=n_clusters)
        self.item_features = None
        self.item_ids = None

    def fit(self, item_features: np.ndarray, item_ids: List):
        """训练"""
        self.item_features = item_features
        self.item_ids = item_ids
        self.kmeans.fit(item_features)
        return self

    def get_similar_items(self, item_idx: int, top_k: int = 10) -> List[Tuple]:
        """获取同簇的相似物品"""
        item_cluster = self.kmeans.labels[item_idx]
        same_cluster = np.where(self.kmeans.labels == item_cluster)[0]
        same_cluster = same_cluster[same_cluster != item_idx]

        # 计算与同簇物品的距离
        target_feature = self.item_features[item_idx]
        distances = np.linalg.norm(
            self.item_features[same_cluster] - target_feature,
            axis=1
        )

        # 按距离排序
        sorted_indices = np.argsort(distances)[:top_k]
        result = [
            (self.item_ids[same_cluster[i]], distances[i])
            for i in sorted_indices
        ]

        return result
```

## 5. Mini-Batch KMeans

```python
class MiniBatchKMeans:
    """
    Mini-Batch KMeans

    适用于大规模数据
    """

    def __init__(self, n_clusters: int = 8, max_iters: int = 100,
                 batch_size: int = 100, random_state: int = None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.random_state = random_state

        self.centroids = None
        self.labels = None

    def fit(self, X: np.ndarray):
        """训练"""
        if self.random_state:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]

        # 初始化中心
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()

        for _ in range(self.max_iters):
            # 随机采样 mini-batch
            batch_idx = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[batch_idx]

            # 分配
            distances = np.zeros((self.batch_size, self.n_clusters))
            for k in range(self.n_clusters):
                distances[:, k] = np.sum((X_batch - self.centroids[k]) ** 2, axis=1)
            labels = np.argmin(distances, axis=1)

            # 更新
            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    # 增量更新
                    self.centroids[k] = (
                        self.centroids[k] * 0.9 + X_batch[mask].mean(axis=0) * 0.1
                    )

        # 最终标签
        distances = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)
        self.labels = np.argmin(distances, axis=1)

        return self
```

## 6. Sklearn 使用

```python
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score
import numpy as np


def sklearn_demo():
    """Sklearn KMeans 示例"""
    # 生成数据
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(100, 2) + [2, 2],
        np.random.randn(100, 2) + [-2, -2],
        np.random.randn(100, 2) + [2, -2]
    ])

    # 训练
    kmeans = SklearnKMeans(n_clusters=3, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X)

    print(f"轮廓系数: {silhouette_score(X, labels):.4f}")
    print(f"WCSS: {kmeans.inertia_:.4f}")
```

## 7. 学习总结

### 7.1 核心要点

1. **迭代优化**：分配 + 更新
2. **KMeans++**：更好的初始化
3. **K 值选择**：肘部法则、轮廓系数

### 7.2 优缺点

**优点：**
- 简单易实现
- 可扩展性好
- 适合球状簇

**缺点：**
- 需要指定 K
- 对异常值敏感
- 不适合非凸簇

## 8. 练习题

1. 实现 KMeans++ 的初始化过程。

2. 比较不同 K 值对聚类效果的影响。

3. 将 KMeans 应用于推荐系统的用户分群。
