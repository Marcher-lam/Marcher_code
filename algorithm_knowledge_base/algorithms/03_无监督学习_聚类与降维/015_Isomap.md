# Isomap（等度量映射）学习文档

> 流形学习的经典算法，通过测地距离实现非线性降维

---

## 1. 算法基础认知

**一句话定义**：Isomap（Isometric Mapping，等度量映射）是由Tenenbaum等人在2000年提出的非线性降维算法，通过保持数据的"测地距离"（沿流形的最短路径），将高维数据映射到低维空间，是流形学习的里程碑方法。

**直觉类比**：Isomap就像"地图上的最短路径"。想象你在爬一座山（弯曲的流形），要从A点走到B点，直线距离（欧氏距离）会穿过山体，但实际最短路径是"盘山公路"的距离。Isomap计算的不是直线距离，而是在这个弯曲表面上走的实际距离，用这个"真距离"来做降维，就能保持数据的内在几何结构。

**历史背景**：
- 2000年，Tenenbaum等人在Science论文"A global geometric framework for nonlinear dimensionality reduction"中首次提出
- 与LLE并列为流形学习两大支柱
- 后续发展出Kernel PCA、Highway PCA等

**算法定位**：
- 类型：流形学习 → 非线性降维
- 输出：低维嵌入
- 模型类型：基于测地距离

**前置知识**：
- [必备]：线性降维（PCA、MDS）
- [必备]：图论基础
- [推荐]：流形概念

---

## 2. 核心原理

### 2.1 线性降维的局限

PCA和MDS只能处理线性结构：

- 直线/平面可以处理 ✓
- 弯曲的流形无法处理 ✗

例如：S形曲线（Swiss Roll），用PCA会切断，用Isomap可以展开。

### 2.2 Isomap的核心创新

**核心思想**：把数据看成流形，用图上的最短路径近似测地距离！

步骤：
1. 构建k近邻图
2. 计算所有点对的最短路径
3. MDS降维

### 2.3 整体流程

```
              高维数据
                 │
                 ▼
        ┌────────────────┐
        │  构建k近邻图   │
        │ (k-NN graph)   │
        └───────┬────────┘
                 │
                 ▼
        ┌────────────────┐
        │  计算测地距离   │
        │ (Floyd/Dijkstra)│
        └───────┬────────┘
                 │
                 ▼
        ┌────────────────┐
        │  MDS降维      │
        │ (保持距离)    │
        └───────┬────────┘
                 │
                 ▼
              低维嵌入
```

---

## 3. 数学公式与推导

### 3.1 构建邻接图

对每个点xi，找k个最近邻：

$$d(x_i, x_j) = \| x_i - x_j \|_2$$

如果xj是xi的k近邻或相反，连边。

### 3.2 计算测地距离

**最短路径近似**：

使用Dijkstra或Floyd-Warshall算法：

$$D_{ geodesic}(i,j) = \min_p \sum_{t=1}^{len(p)-1} d(x_{p_t}, x_{p_{t+1}})$$

其中p是路径。

### 3.3 中心化

$$B = -\frac{1}{2} H D^2 H$$

其中 $H = I - \frac{1}{n}11^T$，centering矩阵。

### 3.4 特征分解

$$B = V \Lambda V^T$$

取最大的d个特征值，构成嵌入：

$$Y = V_d \sqrt{\Lambda_d}$$

---

## 4. 训练过程讲解

### 4.1 算法

```
1. 对每个点xi，找k个最近邻，构建图G
2. 计算G中所有点对的最短路径D
3. 用D构建距离矩阵
4. 中心化: B = -1/2 * H*D^2*H
5. 特征分解: 取top-d特征向量
6. 输出: Y = V_d * sqrt(Lambda_d)
```

### 4.2 参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| n_neighbors | 10-30 | 邻居数 |
| n_components | 2-3 | ��维维度 |

### 4.3 时间复杂度

| 步骤 | 复杂度 |
|------|--------|
| k-NN | O(n² log n) |
| 最短路径 | O(n³) 或 O(n² log n) |
| 特征分解 | O(n³) |

n大时用稀疏近似。

---

## 5. 应用场景

### 5.1 数据可视化

- 人脸数据集
- 手写数字
- S曲线展开

### 5.2 预处理

作为其他模型的预处理步骤。

### 5.3 图像检索

特征提取后的相似度搜索。

---

## 6. 优缺点分析

### 6.1 优点

| 优点 |
|------|
| 非线性保持 |
| 全局最优 |
| 理论保证 |

### 6.2 缺点

| 缺点 |
|------|
| 计算重 |
| 断路敏感 |
| 需要连通 |

### 6.3 改进

| 方法 | 改进点 |
|------|--------|
| Landmark Isomap | 采样加速 |
| C-Isomap | 测地保持 |

---

## 7. 调库实现

### 7.1 sklearn（推荐）

```python
from sklearn.manifold import Isomap
from sklearn.datasets import make_s_curve

# 生成数据
X, color = make_s_curve(n_samples=1000, random_state=42)

# Isomap降维
isomap = Isomap(n_neighbors=15, n_components=2)
X_embed = isomap.fit_transform(X)

print(f"训练集大小: {X_embed.shape}")
```

### 7.2 不同数据

```python
# 人脸数据
from sklearn.datasets import fetch_lfw_people
lfw = fetch_lfw_people(min_faces_per_person=70)

isomap = Isomap(n_components=2)
embeddings = isomap.fit_transform(lfw.data)
```

---

## 8. 手工实现

### 8.1 完整实现

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra


class Isomap:
    def __init__(self, n_neighbors=10, n_components=2):
        self.k = n_neighbors
        self.d = n_components
        
    def fit_transform(self, X):
        X = np.array(X)
        
        # 1. 构建k近邻图
        distances = cdist(X, X)
        
        # 找k近邻
        sorted_idx = np.argsort(distances, axis=1)
        neighbors = sorted_idx[:, 1:self.k+1]  # 排除自己
        
        # 构建稀疏邻接矩阵
        n = len(X)
        row = []
        col = []
        data = []
        
        for i in range(n):
            for j in neighbors[i]:
                row.extend([i, j])
                col.extend([j, i])
                data.extend([distances[i,j], distances[i,j]])
        
        adj = np.zeros((n, n))
        for r, c, d in zip(row, col, data):
            adj[r, c] = d
            
        # 2. 最短路径
        D, _ = dijkstra(adj, directed=False, return_predecessors=True)
        
        # 3. MDS
        D = D ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D @ H
        
        # 4. 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # 排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 取top-d
        embedding = eigenvectors[:, :self.d] * np.sqrt(eigenvalues[:self.d])
        
        return embedding


# 使用
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成S形数据
    t = np.linspace(0, np.pi, 500)
    X1 = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
    X1 += np.random.randn(500, 2) * 0.1
    
    X2 = np.stack([(t + np.pi) * np.cos(t + np.pi), 
                   (t + np.pi) * np.sin(t + np.pi)], axis=1)
    X2 += np.random.randn(500, 2) * 0.1
    
    X = np.vstack([X1, X2])
    
    isomap = Isomap(n_neighbors=15, n_components=2)
    embedding = isomap.fit_transform(X)
    
    print(f"嵌入形状: {embedding.shape}")
```

---

## 9. 可视化

### 9.1 数据可视化

```python
import matplotlib.pyplot as plt

def plot_isomap(X, label):
    isomap = Isomap(n_neighbors=15, n_components=2)
    embed = isomap.fit_transform(X)
    
    plt.scatter(embed[:, 0], embed[:, 1], c=label)
    plt.colorbar()
    plt.show()
```

---

## 10. 评估

### 10.1 指标

| 指标 | 说明 |
|------|------|
| stress | 保持距离程度 |
| MSE | 均方误差 |

### 10.2 参数选择

k太大会短路，k太小不连通。

---

## 11. 常见问题与技巧

### 11.1 断路问题

解决：增加k或用多个初始点

### 11.2 稀疏数据

解决：landmark采样

---

## 12. 学习总结

### 12.1 核心要点

- 测地距离
- 最短路径
- MDS

### 12.2 扩展

- Landmark Isomap
- Kernel Isomap

---

## 13. 练习题

1. Isomap和PCA的区别？
2. 为什么需要最短路径？

---

## 14. 学习路径

1. MDS基础
2. 图论最短路径
3. Isomap
4. 实战

---

## 附录

### 参考

- 论文：Tenenbaum et al., 2000
- 库：sklearn

---

**文档结束**

---

## 补充材料：Isomap变体与扩展

### A1. Landmark Isomap

Landmark Isomap通过采样landmarks加速：

```python
class LandmarkIsomap:
    def __init__(self, n_neighbors=10, n_components=2, n_landmarks=200):
        self.k = n_neighbors
        self.d = n_components
        self.n_landmarks = n_landmarks
    
    def fit_transform(self, X):
        X = np.array(X)
        n = len(X)
        
        # 1. 随机采样landmarks
        indices = np.random.choice(n, self.n_landmarks, replace=False)
        landmarks = X[indices]
        
        # 2. 计算到landmarks的距离
        from scipy.spatial.distance import cdist
        D_all = cdist(X, landmarks)
        D_land = cdist(landmarks, landmarks)
        
        # 3. 用Dijkstra计算测地距离近似
        from scipy.sparse.csgraph import dijkstra
        D_land_geo, _ = dijkstra(D_land, directed=False, return_predecessors=True)
        
        # 4. 插值得到所有点对距离
        for i in range(n):
            for j in range(self.n_landmarks):
                if D_all[i, j] < D_land_geo[indices[i], indices[j]]:
                    D_all[i, j] = D_land_geo[indices[i], indices[j]]
        
        # 5. MDS降维
        D = D_all ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D @ H
        
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1]
        
        return eigenvectors[:, idx[:self.d]] * np.sqrt(eigenvalues[idx[:self.d]])
```

### A2. Kernel PCA与Isomap的结合

```python
def kernel_isomap(X, n_components=2, n_neighbors=10, gamma=0.1):
    """Kernel Isomap实现"""
    
    # 1. 计算核矩阵
    from scipy.spatial.distance import cdist
    K = np.exp(-gamma * cdist(X, X) ** 2)
    
    # 2. 中心化核矩阵
    n = len(X)
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # 3. Isomap降维
    from sklearn.manifold import Isomap
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    
    return isomap.fit_transform(X)
```

### A3. Isomap的全局最优性证明

**定理**：Isomap的解是最小化测地距离保持误差的全局最优解。

**证明**：
Isomap等价于在测地距离矩阵D上做MDS。MDS的解是对D²做特征分解，因此是全局最优的。

### A4. Isomap可视化进阶

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_swiss_roll():
    """可视化Swiss Roll的Isomap展开"""
    np.random.seed(42)
    
    # 生成Swiss Roll数据
    n = 2000
    t = np.linspace(0, 4 * np.pi, n)
    x = t * np.cos(t)
    y = np.random.randn(n) * 0.5
    z = t * np.sin(t)
    
    X_3d = np.stack([x, y, z], axis=1)
    colors = t
    
    # Isomap降维
    from sklearn.manifold import Isomap
    isomap = Isomap(n_neighbors=15, n_components=2)
    X_2d = isomap.fit_transform(X_3d)
    
    # 可视化
    fig = plt.figure(figsize=(16, 6))
    
    # 3D原始数据
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    scatter = ax.scatter(x, y, z, c=t, cmap='viridis', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Swiss Roll (3D)')
    plt.colorbar(scatter, ax=ax, shrink=0.5)
    
    # 2D降维结果
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=t, cmap='viridis', s=5)
    ax.set_xlabel('Isomap 1')
    ax.set_ylabel('Isomap 2')
    ax.set_title('Isomap (2D)')
    
    # 测地距离 vs 欧氏距离
    ax = fig.add_subplot(1, 3, 3)
    from scipy.spatial.distance import pdist, squareform
    euclidean = pdist(X_3d[:100])
    ax.hist(euclidean, bins=30, alpha=0.5, label='Euclidean')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('Distance Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('isomap_swiss_roll.png', dpi=150)
    plt.show()


def compare_manifold_learning_methods():
    """比较不同的流形学习方法"""
    np.random.seed(42)
    
    # 生成测试数据
    n = 1000
    t = np.linspace(0, 2*np.pi, n)
    x = np.sin(t) + np.random.randn(n) * 0.05
    y = np.cos(t) + np.random.randn(n) * 0.05
    z = t * 0.5 + np.random.randn(n) * 0.1
    
    X = np.stack([x, y, z], axis=1)
    
    # 应用不同方法
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS
    
    methods = {
        'Isomap': Isomap(n_neighbors=15, n_components=2),
        'LLE': LocallyLinearEmbedding(n_neighbors=15, n_components=2),
        'MDS': MDS(n_components=2)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (name, method) in zip(axes, methods.items()):
        try:
            X_embedded = method.fit_transform(X)
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=t, cmap='viridis', s=10)
            ax.set_title(f'{name}')
        except Exception as e:
            ax.set_title(f'{name} Failed')
        
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
    
    plt.tight_layout()
    plt.savefig('manifold_comparison.png', dpi=150)
    plt.show()


def analyze_parameter_sensitivity():
    """分析参数敏感性"""
    np.random.seed(42)
    
    # 生成数据
    n = 1000
    t = np.linspace(0, np.pi, n)
    X = np.stack([t * np.cos(t), t * np.sin(t)], axis=1) + np.random.randn(n, 2) * 0.1
    
    k_values = [5, 10, 15, 20, 30, 50]
    reconstruction_errors = []
    
    from sklearn.manifold import Isomap
    from sklearn.metrics import pairwise_distances
    
    for k in k_values:
        isomap = Isomap(n_neighbors=k, n_components=2)
        X_embedded = isomap.fit_transform(X)
        
        # 计算重构误差
        X_reconstructed = isomap reconstruct(X_embedded)
        error = np.mean((X - X_reconstructed) ** 2)
        reconstruction_errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, reconstruction_errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Reconstruction Error')
    plt.title('Parameter Sensitivity: k in Isomap')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('isomap_parameter_sensitivity.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_swiss_roll()
    compare_manifold_learning_methods()
    analyze_parameter_sensitivity()
```

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import numpy as np

class IsomapScratch:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr, self.n_iter, self.losses = lr, n_iter, []
    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.n_iter):
            err = X @ self.w + self.b - y
            self.losses.append(np.mean(err**2))
            self.w -= self.lr * (2/n) * X.T @ err
            self.b -= self.lr * (2/n) * np.sum(err)
        return self
    def predict(self, X): return X @ self.w + self.b

np.random.seed(42)
X = np.random.randn(200, 3)
y = 2*X[:,0] - X[:,1] + 0.5*X[:,2] + np.random.randn(200)*0.1
m = IsomapScratch().fit(X, y)
print(f"Loss: {m.losses[-1]:.6f}")
```

## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估

