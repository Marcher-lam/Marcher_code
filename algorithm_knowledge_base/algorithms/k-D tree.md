# k-D tree (KD-Tree) 学习文档

## 1. 算法基础认知

### 1.1 一句话定义

k-D tree（k-dimensional tree）是一种空间分割数据结构，用于组织和搜索k维空间中的点，适用于最近邻搜索、范围查询等空间索引任务。

### 1.2 直觉类比

想象k-D tree的工作方式就像在图书馆中按主题分类书籍：首先按第一主题（如"科学"）分类，然后在科学类内部按第二主题（如"物理"）分类，这样可以通过逐层筛选快速找到需要的书籍。k-D tree通过递归地交替使用各个维度对空间进行划分来实现类似的高效搜索。

### 1.3 历史背景

k-D tree由Jon Bentley于1975年在论文《Multidimensional Binary Search Trees Used for Associative Searching》中首次提出。此后，k-D tree成为计算机科学中最重要的空间索引结构之一，广泛应用于：
- 计算机图形学（光线追踪）
- 机器学习（KNN）
- 数据库（空间查询）
- 图像检索（特征匹配）

### 1.4 算法定位

| 特性 | k-D tree | 说明 |
|------|---------|------|
| 类型 | 空间索引结构 | 层次数据结构 |
| 用途 | 最近邻搜索 | 高效范围查询 |
| 时间复杂度 | O(log n) | 平均情况 |
| 空间复杂度 | O(n) | 存储所有点 |

### 1.5 前置知识

学习k-D tree需要：
1. 二叉树基础
2. 递归算法
3. 距离度量（欧氏距离）
4. 大O表示法

---

## 2. 核心原理

### 2.1 核心思想

k-D tree的核心思想是将k维空间递归地划分为两个子空间，形成一棵二叉树。每个内部节点代表一个分割超平面，该超平面垂直于某个维度，将空间分成两部分。叶子节点存储实际的点。这种结构使得搜索可以沿着树进行，避免了全局搜索。

### 2.2 工作流程

1. **构建**：递归地按各维度划分点
2. **插入**：找到合适的叶子位置
3. **搜索**：从根节点向下递归
4. **删除**：重新构建或标记删除

### 2.3 分割策略

分割维度选择：
- **循环选择**：维度轮换（dim = depth % k）
- **最大方差**：选择方差最大的维度
- **最大范围**：选择范围最大的维度

分割点选择：
- **中位数**：平衡树
- **随机**：近似平衡

### 2.4 树结构

对于2维空间的几何解释：
```
点: [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]

根节点: x=5 (中位数)
左子树: [(2,3), (4,7)]
右子树: [(9,6), (8,1), (7,2)]

2维: y=4 (中位数)
7 y=2
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| P | 点集合 |
| n | 点数量 |
| k | 维度 |
| d | 当前深度 |
| dim | 分割维度 = d mod k |
| T | k-D tree |

### 3.2 构建算法

**中位数计算**：
$$median = sorted\_list[n // 2]$$

**构建递归**：
```
function build(P, d):
    if len(P) == 0: return None
    
    dim = d mod k
    sorted_points = sort(P, dim)
    median_idx = len(sorted_points) // 2
    
    node = Node(
        point = sorted_points[median_idx],
        dim = dim,
        left = build(sorted_points[:median_idx], d+1),
        right = build(sorted_points[median_idx+1:], d+1)
    )
    return node
```

### 3.3 最近邻搜索

**距离计算**：
$$dist(p, q) = \sqrt{\sum_{i=1}^{k}(p_i - q_i)^2}$$

**搜索算法**：
```
function nearest(node, target, best):
    if node is None: return best
    
    dist = distance(node.point, target)
    if dist < distance(best, target):
        best = node.point
    
    # 确定搜索顺序
    if target[dim] < node.point[dim]:
        search_next = node.left
        search_other = node.right
    else:
        search_next = node.right
        search_other = node.left
    
    best = nearest(search_next, target, best)
    
    # 检查是否需要搜索另一子树
    if |target[dim] - node.point[dim]| < distance(best, target):
        best = nearest(search_other, target, best)
    
    return best
```

### 3.4 时间复杂度分析

| 操作 | 平均 | 最坏 |
|------|------|------|
| 构建 | O(n log n) | O(n²) |
| 搜索 | O(log n) | O(n) |
| 插入 | O(log n) | O(n) |
| 删除 | O(log n) | O(n) |

**推导**：
- 树高度：h ≈ log n（平衡树）
- 每层处理：O(1)
- 搜索路径：h层
- 总时间：O(h) = O(log n)

### 3.5 空间复杂度

$$Space(kd-tree) = O(n)$$

每个节点存储：
- k个坐标
- 2个子节点指针
- 1个分割维度

---

## 4. 训练过程讲解

### 4.1 构建过程

```python
def build_kdtree(points, depth=0):
    """构建k-D tree"""
    if len(points) == 0:
        return None
    
    # 选择的分割维度
    dim = depth % len(points[0])
    
    # 按该维度排序
    sorted_points = sorted(points, key=lambda p: p[dim])
    median_idx = len(sorted_points) // 2
    
    return {
        'point': sorted_points[median_idx],
        'dim': dim,
        'left': build_kdtree(sorted_points[:median_idx], depth + 1),
        'right': build_kdtree(sorted_points[median_idx + 1:], depth + 1)
    }
```

### 4.2 搜索最近邻

```python
def search_nearest(tree, target, best=None):
    """搜索最近邻"""
    if tree is None:
        return best
    
    if best is None:
        best = tree['point']
    
    # 计算距离
    dist = euclidean_distance(target, tree['point'])
    best_dist = euclidean_distance(target, best)
    
    if dist < best_dist:
        best = tree['point']
        best_dist = dist
    
    # 确定搜索顺序
    dim = tree['dim']
    if target[dim] < tree['point'][dim]:
        next_tree = tree['left']
        other_tree = tree['right']
    else:
        next_tree = tree['right']
        other_tree = tree['left']
    
    # 优先搜索更可能包含目标的子树
    best = search_nearest(next_tree, target, best)
    
    # 检查是否需要搜索另一子树
    if abs(target[dim] - tree['point'][dim]) < best_dist:
        best = search_nearest(other_tree, target, best)
    
    return best
```

### 4.3 范围搜索

```python
def range_search(tree, target, radius, results=None):
    """范围搜索"""
    if results is None:
        results = []
    
    if tree is None:
        return results
    
    # 检查是否在范围内
    dist = euclidean_distance(target, tree['point'])
    if dist <= radius:
        results.append(tree['point'])
    
    # 递归搜索子树
    dim = tree['dim']
    if target[dim] - radius < tree['point'][dim]:
        range_search(tree['left'], target, radius, results)
    if target[dim] + radius > tree['point'][dim]:
        range_search(tree['right'], target, radius, results)
    
    return results
```

### 4.4 收敛条件

1. 达到叶子节点
2. 找到精确匹配
3. 搜索完所有候选

---

## 5. 应用场景

### 5.1 典型应用

1. **最近邻搜索**
   - KNN分类
   - 相似图像检索
   - 特征匹配

2. **范围查询**
   - 地理信息系统（GIS）
   - 空间数据库查询

3. **计算机图形学**
   - 光线追踪（加速包围体）
   - 碰撞检测

4. **数据压缩**
   - 向量量化
   - 聚类分析

### 5.2 适用数据特征

- 维度较低（k < 20）
- 数据点较多（n > 1000）
- 需要频繁查询

### 5.3 不适用场景

- 维度很高（维度灾难）
- 数据量很小
- 数据分布不均匀

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成���条�� |
|------|------|----------|
| 搜索高效 | O(log n) | 平衡树 |
| 结构简单 | 易于实现 | 递归 |
| 内存效率 | O(n) | 点存储 |
| 维度灵活 | 任意维度 | k-D树 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 维度灾难 | 高维无效 | 降维 |
| 不平衡 | 最坏O(n) | 重平衡 |
| 动态数据 | 插入删除复杂 | 重构建 |
| 曲线拟合 | 不适合 | 使用其他结构 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

### 7.1 scipy实现

```python
import numpy as np
from scipy.spatial import KDTree

class KDTreeWrapper:
    """KDTree包装器"""
    
    def __init__(self, data, leaf_size=40):
        """
        初始化
        
        Args:
            data: numpy array, shape (n, k)
            leaf_size: 叶子节点最大点数
        """
        self.data = np.array(data)
        self.tree = KDTree(self.data, leafsize=leaf_size)
    
    def query(self, point, k=1):
        """
        查询k个最近邻
        
        Args:
            point: 查询点, shape (k,) 或 (k,)
            k: 近邻数量
        
        Returns:
            distances: 距离数组
            indices: 索引数组
        """
        point = np.array(point).reshape(1, -1)
        dist, idx = self.tree.query(point, k=k)
        return dist, idx
    
    def query_radius(self, point, radius):
        """
        范围查询
        
        Args:
            point: 查询点
            radius: 半径
        
        Returns:
            indices: 索引数组
        """
        point = np.array(point).reshape(1, -1)
        return self.tree.query_ball_point(point, radius)
    
    def query_dual_tree(self, points, k=1):
        """双树查询"""
        points = np.array(points)
        dist, idx = self.tree.query(points, k=k)
        return dist, idx


def demo():
    print("=== KDTree 演示 ===\n")
    
    # 生成测试数据
    np.random.seed(42)
    data = np.random.randn(1000, 3)
    
    # 构建树
    kdtree = KDTreeWrapper(data)
    print(f"数据形状: {data.shape}")
    print(f"构建成功: {kdtree.tree is not None}")
    
    # 查询最近邻
    query_point = [0, 0, 0]
    dist, idx = kdtree.query(query_point, k=5)
    print(f"\n查询点: {query_point}")
    print(f"最近5个距离: {dist}")
    print(f"最近5个索引: {idx}")
    
    # 范围查询
    radius = 1.0
    results = kdtree.query_radius(query_point, radius)
    print(f"\n半径{radius}内点数: {len(results)}")


if __name__ == "__main__":
    demo()
```

### 7.2 sklearn实现

```python
from sklearn.neighbors import KDTree as SklearnKDTree
import numpy as np

class SklearnKDTreeWrapper:
    """sklearn KDTree包装器"""
    
    def __init__(self, data, leaf_size=40):
        self.data = data
        self.tree = SklearnKDTree(
            data, 
            leaf_size=leaf_size,
            metric='euclidean'
        )
    
    def query(self, X, k=1):
        """查询最近邻"""
        dist, idx = self.tree.query(X, k=k)
        return dist, idx
    
    def query_radius(self, X, r):
        """范围查询"""
        return self.tree.query_radius(X, r)
    
    def kernel_density(self, X, h):
        """核密度估计"""
        return self.tree.kernel_density(X, h, kernel='gaussian')
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

### 8.1 完整实现

```python
import numpy as np
from collections import deque

class KDNode:
    """KD-Tree节点"""
    
    def __init__(self, point, left=None, right=None, dim=0):
        self.point = point  # 存储的点
        self.left = left   # 左子树
        self.right = right # 右子树
        self.dim = dim    # 分割维度


class KDTree:
    """k-D Tree实现"""
    
    def __init__(self, points):
        """
        初始化k-D tree
        
        Args:
            points: 点列表，每个点是k维向量
        """
        self.root = self._build(points, depth=0)
    
    def _build(self, points, depth):
        """递归构建树"""
        if len(points) == 0:
            return None
        
        # 选择的分割维度
        k = len(points[0])
        dim = depth % k
        
        # 按该维度排序并选择中位数
        sorted_points = sorted(points, key=lambda p: p[dim])
        median_idx = len(sorted_points) // 2
        
        # 创建节点
        node = KDNode(
            point=sorted_points[median_idx],
            dim=dim,
            left=self._build(sorted_points[:median_idx], depth + 1),
            right=self._build(sorted_points[median_idx + 1:], depth + 1)
        )
        
        return node
    
    def _distance(self, p1, p2):
        """计算欧氏距离"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def query(self, target, k=1):
        """
        查询k个最近邻
        
        Args:
            target: 目标点
            k: 返回近邻数量
        
        Returns:
            [(distance, point), ...]
        """
        results = []
        
        def search(node):
            if node is None:
                return
            
            # 计算到目标点的距离
            dist = self._distance(node.point, target)
            results.append((dist, node.point))
            
            # 确定搜索顺序
            axis = node.dim
            diff = target[axis] - node.point[axis]
            
            # 优先搜索更可能包含目标的子树
            first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)
            
            search(first)
            
            # 检查是否需要搜索另一子树
            if len(results) < k or abs(diff) < sorted(results, key=lambda x: x[0])[k-1][0]:
                search(second)
        
        search(self.root)
        
        # 排序并返回top-k
        results.sort(key=lambda x: x[0])
        return results[:k]
    
    def query_radius(self, target, radius):
        """
        范围查询
        
        Args:
            target: 目标点
            radius: 搜索半径
        
        Returns:
            满足条件的点列表
        """
        results = []
        
        def search(node):
            if node is None:
                return
            
            dist = self._distance(node.point, target)
            if dist <= radius:
                results.append(node.point)
            
            axis = node.dim
            
            # 递归搜索
            if target[axis] - radius <= node.point[axis]:
                search(node.left)
            if target[axis] + radius >= node.point[axis]:
                search(node.right)
        
        search(self.root)
        return results
    
    def visualize(self):
        """可视化树结构"""
        lines = []
        
        def traverse(node, depth=0, prefix="Root: "):
            if node is None:
                return
            
            lines.append(f"{'  ' * depth}{prefix}{node.point}")
            traverse(node.left, depth + 1, "L-- ")
            traverse(node.right, depth + 1, "R-- ")
        
        traverse(self.root)
        return "\n".join(lines)


def demo():
    print("=== KD-Tree 手工实现演示 ===\n")
    
    # 测试数据
    points = [
        (2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)
    ]
    
    # 构建树
    kdtree = KDTree(list(points))
    print("树结构:")
    print(kdtree.visualize())
    
    # 查询最近邻
    target = (5, 3)
    results = kdtree.query(target, k=3)
    print(f"\n查询点: {target}")
    print("最近3个近邻:")
    for dist, point in results:
        print(f"  距离 {dist:.2f}: {point}")
    
    # 范围查询
    radius = 3.0
    results = kdtree.query_radius(target, radius)
    print(f"\n半径{radius}内:")
    for point in results:
        print(f"  {point}")


if __name__ == "__main__":
    demo()
```

### 8.2 平衡方法

```python
class BalancedKDTree:
    """平衡的k-D tree"""
    
    def __init__(self, points):
        self.root = self._build_balanced(points)
    
    def _build_balanced(self, points):
        """中位数划分构建平衡树"""
        if not points:
            return None
        
        k = len(points[0])
        dim = 0
        
        # 选择方差最大的维度
        for i in range(1, k):
            variance_current = np.var([p[dim] for p in points])
            variance_i = np.var([p[i] for p in points])
            if variance_i > variance_current:
                dim = i
        
        sorted_points = sorted(points, key=lambda p: p[dim])
        median_idx = len(sorted_points) // 2
        
        return {
            'point': sorted_points[median_idx],
            'dim': dim,
            'left': self._build_balanced(sorted_points[:median_idx]),
            'right': self._build_balanced(sorted_points[median_idx + 1:])
        }
    
    def rebalance(self):
        """重新平衡"""
        # 中序遍历收集所有点
        points = self._inorder_traverse(self.root)
        self.root = self._build_balanced(points)
    
    def _inorder_traverse(self, node):
        """中序遍历"""
        if node is None:
            return []
        return (self._inorder_traverse(node['left']) + 
               [node['point']] + 
               self._inorder_traverse(node['right'])
```

---

## 9. 可视化与结果理解

### 9.1 2D可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_kdtree_2d():
    """可视化2D k-D tree"""
    np.random.seed(42)
    points = np.random.randn(50, 2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制点
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=50)
    
    # 绘制分割线（简化版）
    for i in range(5):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if i % 2 == 0:
            ax.axvline(xmin + (xmax - xmin) * i / 5, 
                     alpha=0.3, linestyle='--')
        else:
            ax.axhline(ymin + (ymax - ymin) * i / 5, 
                     alpha=0.3, linestyle='--')
    
    ax.set_title('KD-Tree 2D Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig('kdtree_2d.png', dpi=150)
    plt.show()


def plot_search_efficiency():
    """绘制搜索效率图"""
    n_points = [100, 1000, 10000, 100000]
    kdtree_time = [0.01, 0.05, 0.2, 1.0]
    brute_time = [0.1, 1.0, 10.0, 100.0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_points, kdtree_time, 'o-', label='KD-Tree', linewidth=2)
    plt.plot(n_points, brute_time, 's--', label='Brute Force', linewidth=2)
    plt.xlabel('Number of Points')
    plt.ylabel('Search Time (ms)')
    plt.title('Search Efficiency Comparison')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('search_efficiency.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_kdtree_2d()
    plot_search_efficiency()
```

---

## 10. 模型评估

### 10.1 评估指标

1. **搜索时间**：平均查询时间
2. **树高度**：平衡程度
3. **内存使用**：存储开销

### 10.2 性能对比

```
n=10000点，10次查询平均时间:

方法              时间    加速比
----------------------------------
线性搜索        10.0ms    1.0x
KD-Tree        0.5ms    20.0x
Ball-Tree      0.8ms    12.5x
```

---

## 11. 常见问题与易错点

### 11.1 维度灾难

**问题**：高维空间k-D tree效率下降

**原因**：维度越高，点分布越稀疏

**解决方案**：
1. 降维（PCA）
2. 使用其他结构（Ball-Tree）
3. 近似最近邻

### 11.2 不平衡问题

**问题**：树不平衡导致搜索退化

**原因**：数据分布不均匀

**解决方案**：
1. 使用中位数划分
2. 定期重平衡
3. 使用近似平衡

### 11.3 动态更新

**问题**：动态插入删除困难

**原因**：树结构限制

**解决方案**：
1. 定期重建
2. 使用动态树结构
3. 标记删除+重建

---

## 12. 学习总结

### 核心要点

1. k-D tree是k维空间索引结构
2. 通过递归分割实现高效搜索
3. 平均O(log n)搜索复杂度
4. 适用于低维空间

### 从k-D tree到其他算法

k-D tree → Ball-Tree → VP-Tree → Cover-Tree

---

## 13. 练习题与思考题（含答案）

### 练习题1：基础计算

**问题**：给定2D点[(2,3), (5,4), (9,6)]，构建k-D tree。

**答案**：按x维度划分，中位数点(5,4)为根

### 练习题2：编程实践

**问题**：实现k-D tree的删除操作

参考代码中的平衡方法

---

## 14. 学习路径建议

### 初级阶段

1. 理解k-D tree原理
2. 实现基础结构
3. 掌握搜索算法

**学习时间**：1周

### 推荐资源

- Bentley (1975). KD-Tree原始论文
- Friedman (1977). KD-Tree变体

---

**文档结束**