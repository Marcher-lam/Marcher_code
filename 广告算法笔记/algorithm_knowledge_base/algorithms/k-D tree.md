# k-D tree 学习文档

## 1. 算法基础认知

k-D tree（k-dimensional tree）是一种对 k 维空间数据进行划分的数据结构，本质上是一棵二叉搜索树。它由 Bentley 于 1975 年提出，主要用于高效的多维空间最近邻搜索和范围查询，是 KNN 算法中加速搜索的核心工具。

k-D tree 中的 "k" 指的是数据的空间维度，而非 KNN 中的邻居数。

## 2. 核心原理

k-D tree 的构建过程：每次选择一个维度，以该维度的中位数作为切分点，将空间一分为二，递归构建左右子树。

**切分维度选择策略**：通常按维度轮流切分（第 $l$ 层使用第 $l \mod k$ 个维度），也可选择方差最大的维度。

**最近邻搜索**：利用树的结构进行剪枝——如果当前最近距离小于查询点到另一侧子空间的最小可能距离，则可以跳过该子树的搜索。

## 3. 数学公式与推导

**构建过程（递归）：**

给定数据集 $S = \{x_1, x_2, \ldots, x_n\}$，每个样本 $x_i \in \mathbb{R}^k$：

1. 选择切分维度 $d$（通常取 $d = \text{depth} \mod k$）
2. 找到维度 $d$ 上的中位数样本 $x_{\text{median}}$
3. 以 $x_{\text{median}}$ 为当前节点
4. 左子树：维度 $d$ 上小于中位数的样本
5. 右子树：维度 $d$ 上大于中位数的样本
6. 递归直到子集为空

**最近邻搜索剪枝条件：**

设当前最近距离为 $r$，查询点为 $q$，当前节点切分维度为 $d$，切分值为 $v$：

$$|q_d - v| \geq r \implies \text{可剪枝对应的子树}$$

**时间复杂度：**

| 操作 | 平均 | 最坏 |
|------|------|------|
| 构建 | $O(n\log n)$ | $O(n\log n)$ |
| 最近邻查询 | $O(\log n)$ | $O(n)$ |

## 4. 训练过程讲解

**构建过程（训练）**：

1. **选择切分维度**：第 0 层切分维度 0，第 1 层切分维度 1，循环往复
2. **找中位数**：对当前数据在切分维度上排序，取中位数作为节点
3. **分割数据**：小于中位数的归入左子树，大于中位数的归入右子树
4. **递归**：对左右子树重复上述过程

**查询过程（预测）**：

1. **向下搜索**：从根节点开始，按切分维度比较，找到叶节点
2. **回溯检查**：从叶节点向上回溯，检查另一侧子树是否有更近的点
3. **剪枝**：若切分超平面到查询点的距离已大于当前最近距离，跳过该子树

## 5. 应用场景

- KNN 的加速搜索（替代暴力搜索）
- 多维空间范围查询
- 图像特征匹配（SIFT/SURF 描述子匹配）
- 数据库索引（空间数据）
- 推荐系统中的近邻搜索

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 查询效率高（低维） | 高维数据效率急剧下降 |
| 构建简单 | 动态更新困难（插入/删除代价高） |
| 空间划分直观 | 对数据分布敏感 |
| 支持最近邻和范围查询 | 维度 > 20 时通常不如暴力搜索 |

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs
import time

np.random.seed(42)
X, _ = make_blobs(n_samples=10000, n_features=3, centers=5, random_state=42)
query = np.array([[0.0, 0.0, 0.0]])

tree = KDTree(X, leaf_size=40, metric='euclidean')

dist_sk, idx_sk = tree.query(query, k=5)
print("KDTree最近邻索引:", idx_sk[0])
print("KDTree最近邻距离:", dist_sk[0])

start = time.time()
for _ in range(1000):
    tree.query(query, k=5)
print(f"KDTree 1000次查询耗时: {time.time() - start:.4f}s")

start = time.time()
for _ in range(1000):
    dists = np.sqrt(np.sum((X - query) ** 2, axis=1))
    np.argsort(dists)[:5]
print(f"暴力搜索 1000次查询耗时: {time.time() - start:.4f}s")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class KDNode:
    def __init__(self, point, split_dim, left=None, right=None):
        self.point = point
        self.split_dim = split_dim
        self.left = left
        self.right = right

class KDTreeManual:
    def __init__(self, data):
        self.root = self._build(data.tolist(), depth=0)

    def _build(self, points, depth):
        if not points:
            return None

        k = len(points[0])
        axis = depth % k

        points.sort(key=lambda x: x[axis])
        median = len(points) // 2

        return KDNode(
            point=points[median],
            split_dim=axis,
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1:], depth + 1)
        )

    def nearest_neighbor(self, query, k=1):
        import heapq
        self.best = []
        self.k = k
        self._search(self.root, np.array(query), depth=0)
        return [(dist, node.point) for dist, node in sorted(self.best)]

    def _search(self, node, query, depth):
        if node is None:
            return

        dist = np.sqrt(np.sum((np.array(node.point) - query) ** 2))

        if len(self.best) < self.k:
            heapq.heappush(self.best, (-dist, node))
        elif dist < -self.best[0][0]:
            heapq.heapreplace(self.best, (-dist, node))

        axis = node.split_dim
        diff = query[axis] - node.point[axis]

        close_branch = node.left if diff < 0 else node.right
        far_branch = node.right if diff < 0 else node.left

        self._search(close_branch, query, depth + 1)

        if len(self.best) < self.k or abs(diff) < -self.best[0][0]:
            self._search(far_branch, query, depth + 1)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(42)
points_2d = np.random.rand(20, 2) * 10

tree = KDTreeManual(points_2d)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', s=30)

query = [5.0, 5.0]
ax.scatter(query[0], query[1], c='red', s=100, marker='*', label='查询点')

results = tree.nearest_neighbor(query, k=3)
for dist, pt in results:
    circle = plt.Circle((query[0], query[1]), dist, fill=False, color='red', linestyle='--')
    ax.add_patch(circle)
    ax.scatter(pt[0], pt[1], c='green', s=60, marker='s')

ax.set_title("KD-Tree 最近邻搜索可视化")
ax.legend()
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
plt.tight_layout()
plt.savefig("kdtree_nn_search.png", dpi=150)
plt.show()
```

红色圆圈表示搜索半径，绿色方块为找到的最近邻。

## 10. 模型评估

- **构建时间**：$O(n\log n)$，衡量建树效率
- **查询时间**：与暴力搜索对比加速比
- **查询精度**：验证找到的最近邻是否正确
- **维度影响**：观察不同维度下查询时间的变化

## 11. 常见问题与易错点

- **高维失效**：维度 > 20 时剪枝效果差，可能退化为线性扫描。此时应使用近似方法（如 LSH、HNSW）
- **中位数选择**：直接排序 $O(n\log n)$，可用 `nth_element` 优化到 $O(n)$
- **叶节点大小**：`leaf_size` 影响查询效率，通常 10-40
- **动态更新**：KD-Tree 不支持高效插入删除，需要重建或使用动态版本
- **数据分布不均**：某些区域点密集导致树不平衡

## 12. 学习总结

k-D tree 通过交替选择维度进行空间划分，将最近邻搜索从 $O(n)$ 降到平均 $O(\log n)$。它是 KNN 的核心加速结构，但在高维空间中效果有限。实际应用中，scipy/sklearn 内置的 KDTree 实现已经足够高效。

## 13. 练习题与思考题（含答案）

**Q1**: 为什么 k-D tree 在高维空间中效率下降？

> A: 高维空间中，切分超平面将空间分成两半后，查询点到另一侧的最小距离（$|q_d - v|$）通常较小，导致剪枝条件很少满足，几乎需要搜索所有节点，退化为 $O(n)$。

**Q2**: k-D tree 的平衡性取决于什么？

> A: 取决于每次切分时中位数的选择。如果总是选择中位数，则树是平衡的（左右子树大小差不超过 1）。如果数据分布极度偏斜，可能导致树不平衡。

**Q3**: 如果需要在 k-D tree 中频繁插入新数据，应该怎么做？

> A: k-D tree 不支持高效动态插入。替代方案：(1) 定期重建整棵树；(2) 使用动态 KD-tree（如 relaxed KD-tree）；(3) 使用 Ball Tree 或 R-tree 等支持动态更新的结构。

## 14. 学习路径建议

```
KNN → k-D tree → Ball Tree → 局部敏感哈希(LSH) → HNSW → 向量数据库
```

掌握 KD-Tree 后，建议学习 Ball Tree（适合高维）、LSH（近似搜索），再深入现代向量搜索技术如 HNSW 和 FAISS。
