# K-Means 学习文档

> 最常用的原型聚类算法，通过迭代将数据划分为K个簇，使簇内样本相似度高、簇间相似度低

---

## 1. 算法基础认知

### 一句话定义
K-Means是一种无监督聚类算法，通过迭代优化，将数据集划分为K个簇，使得每个样本属于距离最近的簇中心。

### 直觉类比
想象你是一名城市规划师，需要在城市里建K个邮局。你希望每个居民到所属邮局的距离总和最小。K-Means就像这个过程：先随机放置K个邮局，然后重复使用居民到邮局的距离更新邮局位置，直到邮局位置不再变化。

### 历史背景
K-Means算法最早由Bell实验室的Stuart Lloyd在1957年提出，但直到1982年才公开发表。1967年，James MacQueen独立提出了该算法并命名为"K-Means"。它是聚类领域最著名、应用最广泛的算法之一。

### 算法定位
- 类型：无监督学习 → 聚类
- 输出：簇标签（0到K-1的整数）
- 模型类型：非参数模型、原型聚类

### 前置知识
- 距离度量：欧氏距离、曼哈顿距离等
- 向量运算：均值计算、向量减法
- 优化基础：迭代优化思想
- Python基础：NumPy数组操作、随机数生成

---

## 2. 核心原理

### 2.1 核心思想
K-Means的核心思想可以概括为：**最小化簇内平方误差和（SSE）**。算法通过迭代执行两个步骤来达到这个目标：
1. **分配步骤（E步）**：将每个样本分配到距离最近的簇中心
2. **更新步骤（M步）**：根据分配的簇，重新计算每个簇的中心（均值）

这个过程不断重复，直到簇中心收敛或达到最大迭代次数。

### 2.2 工作流程

1. **初始化**
   - 输入：数据集 $X = \{x_1, x_2, ..., x_n\}, x_i \in \mathbb{R}^d$，簇数量 $K$
   - 随机初始化K个簇中心 $\mu_1, \mu_2, ..., \mu_K$（或从数据中随机选择K个样本）
   - 设置最大迭代次数 $T$ 和收敛阈值 $\epsilon$

2. **迭代优化**
   - 重复以下步骤，直到收敛或达到最大迭代次数：
     
     **a. 分配步骤**：对每个样本 $x_i$，计算到所有簇中心的距离，将其分配到距离最小的簇：
     $$c_i = \arg\min_{k} \| x_i - \mu_k \|^2$$
     
     **b. 更新步骤**：对每个簇 $k$，重新计算簇中心为该簇所有样本的均值：
     $$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$
     其中 $C_k$ 是属于簇 $k$ 的样本集合。
     
     **c. 检查收敛**：如果簇中心的变化小于 $\epsilon$（或簇分配不再变化），则停止。

3. **输出结果**
   - 最终的簇中心 $\mu_1, \mu_2, ..., \mu_K$
   - 每个样本的簇标签 $c_1, c_2, ..., c_n$
   - 簇内平方误差和（SSE）：$SSE = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$

### 2.3 关键概念解释

- **簇中心（Centroid）**：簇中所有样本的均值向量，代表该簇的"原型"
- **簇内平方误差和（SSE）**：衡量簇内样本的紧密程度，SSE越小簇越紧密
- **距离度量**：通常使用欧氏距离，也可以使用曼哈顿距离、余弦距离等
- **收敛**：当簇中心不再变化或变化很小时，算法收敛
- **局部最优**：K-Means可能收敛到局部最优解，而非全局最优

### 2.4 几何/直观解释
在欧氏空间中，K-Means试图找到K个点（簇中心），使得每个样本到其最近簇中心的距离平方和最小。从几何上看，这相当于用K个Voronoi单元来划分空间，每个单元由一个簇中心定义，单元内所有点到该中心的距离小于到其他中心的距离。

算法优化的目标函数是非凸的，因此可能陷入局部最优。多次随机初始化并选择最好的结果（最小SSE）可以缓解这一问题。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $n$ | 样本数量 | 标量 |
| $d$ | 特征维度 | 标量 |
| $K$ | 簇的数量 | 标量 |
| $X$ | 数据集，$X = \{x_1, x_2, ..., x_n\}$ | $n \times d$ |
| $x_i$ | 第 $i$ 个样本 | $d \times 1$ |
| $\mu_k$ | 第 $k$ 个簇的中心 | $d \times 1$ |
| $c_i$ | 样本 $x_i$ 的簇分配（1到K） | 整数 |
| $C_k$ | 属于簇 $k$ 的样本集合 | 样本子集 |
| $SSE$ | 簇内平方误差和 | 标量，$SSE \geq 0$ |

### 3.2 问题形式化
给定数据集 $X = \{x_1, x_2, ..., x_n\}, x_i \in \mathbb{R}^d$ 和簇数量 $K$，我们想找到：
1. 簇分配：$c_i \in \{1, 2, ..., K\}$ 对于每个样本 $i$
2. 簇中心：$\mu_k \in \mathbb{R}^d$ 对于每个簇 $k$

使得目标函数最小化：
$$J(c_1, ..., c_n, \mu_1, ..., \mu_K) = \sum_{i=1}^{n} \| x_i - \mu_{c_i} \|^2$$

等价于最小化SSE：
$$SSE = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$

### 3.3 目标函数/损失函数
K-Means的目标函数是**簇内平方误差和（SSE）**：
$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$

**为什么选择SSE？**
1. **几何意义明确**：对应欧氏距离，易于理解
2. **优化简单**：对簇中心求导可得均值解析解
3. **计算高效**：距离计算简单快速
4. **与高斯混合模型的关系**：当高斯混合模型方差相同时，最大似然估计等价于K-Means

### 3.4 推导过程

**证明更新步骤的簇中心选择是最优的**：

对于固定的簇分配 $c_1, ..., c_n$，我们希望找到最优的簇中心 $\mu_1, ..., \mu_K$ 来最小化 $J$。

对 $\mu_k$ 求偏导（向量形式）：
$$\frac{\partial J}{\partial \mu_k} = \frac{\partial}{\partial \mu_k} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$

展开：
$$\| x_i - \mu_k \|^2 = (x_i - \mu_k)^\top (x_i - \mu_k) = x_i^\top x_i - 2 x_i^\top \mu_k + \mu_k^\top \mu_k$$

对 $\mu_k$ 求导：
$$\frac{\partial}{\partial \mu_k} \| x_i - \mu_k \|^2 = -2 x_i + 2 \mu_k$$

因此：
$$\frac{\partial J}{\partial \mu_k} = \sum_{x_i \in C_k} (-2 x_i + 2 \mu_k) = -2 \sum_{x_i \in C_k} x_i + 2 |C_k| \mu_k$$

令导数为0：
$$-2 \sum_{x_i \in C_k} x_i + 2 |C_k| \mu_k = 0$$
$$\Rightarrow \mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

这证明：对于固定的簇分配，最优簇中心是该簇样本的均值。

**注意**：对于固定的簇中心，最优簇分配是将每个样本分配到最近的簇中心，这由分配步骤给出。

然而，同时优化 $c_i$ 和 $\mu_k$ 是NP-hard问题。K-Means使用坐标下降的思想：交替优化 $c_i$（固定 $\mu_k$）和 $\mu_k$（固定 $c_i$），最终收敛到局部最优。

### 3.5 最终解/算法步骤

**标准K-Means算法**：

```
输入：数据集 X (n×d)，簇数量 K，最大迭代次数 T，收敛阈值 ε
输出：簇中心 μ₁,...,μₖ，簇分配 c₁,...,cₙ

1. 初始化：随机选择K个样本作为初始簇中心 μ₁,...,μₖ
2. 对于迭代 t = 1 到 T：
   a. 分配步骤：
      对于每个样本 i = 1 到 n：
        cᵢ = argminₖ ||xᵢ - μₖ||²
   b. 更新步骤：
      对于每个簇 k = 1 到 K：
        如果 Cₖ 非空：
          μₖ = (1/|Cₖ|) ∑_{xᵢ∈Cₖ} xᵢ
        否则：
          重新随机初始化 μₖ（或保持原值）
   c. 检查收敛：
      如果所有簇中心变化 ||μₖ⁽ᵗ⁾ - μₖ⁽ᵗ⁻¹⁾|| < ε：
        停止并输出结果
3. 返回最终 μₖ 和 cᵢ
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split  # 聚类通常不需要划分训练测试集
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================
# 生成示例数据（3个簇的二维数据）
# ============================================
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, 
                       cluster_std=0.8, random_state=42)

print(f"数据集形状: {X.shape}")
print(f"真实簇数量: {len(np.unique(y_true))}")

# ============================================
# 数据预处理
# ============================================
# K-Means对特征尺度敏感！需要标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"标准化前特征均值: {X.mean(axis=0)}")
print(f"标准化后特征均值: {X_scaled.mean(axis=0):.4f}")
print(f"标准化后特征标准差: {X_scaled.std(axis=0):.4f}")

# 可视化原始数据和标准化后的数据
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 原始数据
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
axes[0].set_title('原始数据')
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')
axes[0].grid(True, alpha=0.3)

# 标准化后数据
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
axes[1].set_title('标准化后数据')
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

预处理要点：
1. **标准化非常重要**：K-Means基于欧氏距离，如果特征尺度不同，尺度大的特征会主导距离计算
2. **缺失值处理**：K-Means不能直接处理缺失值，需要提前填充
3. **异常值**：K-Means对异常值敏感，因为均值容易受异常值影响
4. **类别特征**：K-Means适用于连续特征，类别特征需要编码

### 4.2 参数初始化

```python
class KMeansManual:
    """
    手动实现的K-Means算法
    """
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, n_init=10, random_state=None):
        """
        初始化K-Means
        
        参数:
            n_clusters: 簇的数量K
            max_iter: 最大迭代次数
            tol: 收敛阈值（簇中心变化小于此值则停止）
            n_init: 随机初始化的次数，选择最好的结果
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.centroids_ = None   # 最终簇中心
        self.labels_ = None      # 最终簇标签
        self.inertia_ = None    # 最终SSE
        self.n_iter_ = None     # 实际迭代次数
        
    def _init_centroids(self, X):
        """
        初始化簇中心：随机选择K个样本作为初始中心
        """
        n_samples = X.shape[0]
        # 随机选择K个不重复的样本索引
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()
```

### 4.3 迭代过程

```python
    def fit(self, X):
        """
        训练K-Means模型
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        
        print(f"开始训练K-Means，K={self.n_clusters}...")
        print(f"样本数: {n_samples}, 特征数: {n_features}")
        
        # 多次随机初始化，选择最好的结果
        for init in range(self.n_init):
            # 1. 初始化簇中心
            centroids = self._init_centroids(X)
            
            # 2. 迭代优化
            for iteration in range(self.max_iter):
                # 分配步骤
                distances = self._compute_distances(X, centroids)
                labels = np.argmin(distances, axis=1)
                
                # 更新步骤
                new_centroids = np.zeros_like(centroids)
                for k in range(self.n_clusters):
                    cluster_points = X[labels == k]
                    if len(cluster_points) > 0:
                        new_centroids[k] = cluster_points.mean(axis=0)
                    else:
                        # 如果簇为空，重新随机初始化该簇中心
                        new_centroids[k] = X[np.random.choice(n_samples)]
                
                # 检查收敛：簇中心变化是否小于阈值
                center_shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                
                if center_shift < self.tol:
                    print(f"初始化 {init+1}/{self.n_init}: 第 {iteration+1} 轮收敛")
                    break
            else:
                print(f"初始化 {init+1}/{self.n_init}: 达到最大迭代次数 {self.max_iter}")
            
            # 计算本次初始化的SSE
            inertia = self._compute_inertia(X, centroids, labels)
            
            # 如果本次结果更好，保存
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()
                best_iter = iteration + 1
        
        # 保存最终结果
        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter
        
        print(f"训练完成！最优SSE: {best_inertia:.4f}")
        return self
    
    def _compute_distances(self, X, centroids):
        """
        计算每个样本到每个簇中心的距离（欧氏距离的平方）
        """
        # 使用广播机制高效计算
        # distances[i, k] = ||x_i - μ_k||^2
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            diff = X - centroids[k]
            distances[:, k] = np.sum(diff**2, axis=1)
        return distances
    
    def _compute_inertia(self, X, centroids, labels):
        """
        计算SSE（簇内平方误差和）
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                diff = cluster_points - centroids[k]
                inertia += np.sum(diff**2)
        return inertia
```

### 4.4 收敛条件

```python
    def check_convergence(self, old_centroids, new_centroids):
        """
        检查是否收敛：簇中心的变化是否小于阈值
        """
        # 计算所有簇中心变化的平均范数
        center_shift = np.linalg.norm(new_centroids - old_centroids)
        return center_shift < self.tol
    
    def plot_convergence(self, X, max_iter=20):
        """
        绘制收敛过程：SSE随迭代次数的变化
        """
        centroids = self._init_centroids(X)
        inertias = []
        
        for iteration in range(max_iter):
            # 分配步骤
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)
            
            # 计算SSE
            inertia = self._compute_inertia(X, centroids, labels)
            inertias.append(inertia)
            
            # 更新步骤
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)
                else:
                    new_centroids[k] = X[np.random.choice(X.shape[0])]
            
            # 检查收敛
            if self.check_convergence(centroids, new_centroids):
                centroids = new_centroids
                # 再计算一次SSE
                inertia = self._compute_inertia(X, centroids, labels)
                inertias.append(inertia)
                break
            centroids = new_centroids
        
        # 绘制
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(inertias)+1), inertias, 'b-', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('SSE')
        plt.title('K-Means收敛过程')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return inertias
```

收敛条件总结：
1. **簇中心变化小于阈值**：`||μ_new - μ_old|| < tol`
2. **簇分配不再变化**：所有样本的簇标签不变
3. **达到最大迭代次数**：防止无限循环
4. **SSE下降很小**：SSE的变化小于某个阈值

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| `n_clusters` (K) | 簇的数量 | 取决于数据（用肘部法则选择） | 8 |
| `max_iter` | 最大迭代次数 | 100 ~ 500 | 300 |
| `tol` | 收敛阈值 | 1e-4 ~ 1e-2 | 1e-4 |
| `n_init` | 随机初始化次数 | 10 ~ 50 | 10 |
| `init` | 初始化方法 | 'k-means++'（推荐）或 'random' | 'k-means++' |

选择建议：
1. **K的选择**：使用肘部法则、轮廓系数或领域知识确定
2. **初始化方法**：使用k-means++初始化（默认），可以加速收敛并提高质量
3. **n_init**：对于大数据集，可以用较小的n_init（如5）；对于小数据集，用较大的n_init（如50）
4. **标准化**：必须标准化特征，尤其当特征尺度不同时

---

## 5. 应用场景

### 5.1 典型应用

**应用1：客户分群**
- 场景：根据客户的消费行为、 demographics 等特征，将客户分为不同群体
- 为什么适合：K-Means可以自动发现客户群体，无需标签数据
- 实现：收集客户特征（年龄、收入、消费金额等），标准化后运行K-Means

**应用2：图像压缩**
- 场景：将图像中相似颜色聚类，减少颜色数量，实现压缩
- 为什么适合：图像每个像素是RGB向量，K-Means可以找到K个代表颜色
- 实现：将图像像素reshape为样本，用K-Means聚类，用簇中心代表原颜色

**应用3：异常检测预处理**
- 场景：先聚类正常数据，新样本如果离所有簇中心都很远，则可能是异常
- 为什么适合：K-Means可以建模正常数据的分布
- 实现：对正常数据聚类，计算新样本到最近簇中心的距离

### 5.2 适用数据特征

1. **特征尺度一致**：所有特征在同一尺度上（或已标准化）
2. **簇近似球形**：K-Means假设簇是凸的、近似球形
3. **簇大小相近**：K-Means倾向于发现大小相似的簇
4. **样本数量适中**：对于海量数据，K-Means可能较慢（但比层次聚类快）
5. **连续特征**：K-Means基于欧氏距离，适合连续特征

### 5.3 不适用场景

1. **非球形簇**：如果簇形状不规则（如月牙形），K-Means效果差 → 使用DBSCAN、谱聚类
2. **特征尺度差异大且未标准化**：导致错误聚类 → 必须标准化
3. **类别特征多**：欧氏距离不适合类别特征 → 使用k-modes算法
4. **簇密度差异大**：K-Means对密度不敏感 → 使用DBSCAN
5. **需要知道K值**：如果无法确定簇数量 → 使用层次聚类或DBSCAN

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 简单易懂 | 算法思想直观，易于理解 | 通用 |
| 计算高效 | 时间复杂度 O(n·K·d·T)，适合大数据 | 通用 |
| 可扩展性好 | 可以处理较大规模数据 | 样本数<百万，特征数<千 |
| 保证收敛 | 每次迭代目标函数不增，保证收敛到局部最优 | 通用 |
| 广泛适用 | 许多领域都可以应用 | 通用 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 对初始值敏感 | 不同初始化可能得到不同结果 | 使用k-means++初始化，多次运行 |
| 需要预先指定K | 实际中往往不知道最佳K | 使用肘部法则、轮廓系数选择 |
| 对异常值敏感 | 均值受异常值影响大 | 使用K-Medoids（用中位数）|
| 假设簇为球形 | 对非球形簇效果差 | 使用DBSCAN、谱聚类 |
| 可能收敛到局部最优 | 目标函数非凸，可能陷入局部最优 | 多次随机初始化，选择最好结果 |
| 对特征尺度敏感 | 不同尺度特征导致错误聚类 | 必须标准化 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns

# ============================================
# 1. 基本使用：合成数据聚类
# ============================================
print("=" * 60)
print("示例1：K-Means基本使用（合成数据）")
print("=" * 60)

# 生成数据（3个簇）
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, 
                       cluster_std=0.8, random_state=42)

# 标准化（重要！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建并训练K-Means模型
kmeans = KMeans(
    n_clusters=3,          # 簇数量
    init='k-means++',     # 初始化方法：k-means++（推荐）
    n_init=10,            # 随机初始化次数
    max_iter=300,         # 最大迭代次数
    tol=1e-4,            # 收敛阈值
    random_state=42       # 随机种子
)

# 训练模型
kmeans.fit(X_scaled)

# 获取结果
labels = kmeans.labels_          # 簇标签
centroids = kmeans.cluster_centers_  # 簇中心
inertia = kmeans.inertia_        # SSE

print(f"簇中心：\n{centroids}")
print(f"SSE（簇内平方误差和）: {inertia:.4f}")
print(f"迭代次数: {kmeans.n_iter_}")

# 可视化结果
def plot_clusters(X, labels, centroids=None, title="K-Means聚类结果"):
    plt.figure(figsize=(10, 8))
    
    # 绘制样本点，颜色表示簇
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    # 绘制簇中心
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
                   s=200, label='簇中心')
    
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_clusters(X_scaled, labels, centroids, "K-Means聚类结果（K=3）")

# ============================================
# 2. 如何选择合适的K：肘部法则
# ============================================
print("\n" + "=" * 60)
print("示例2：肘部法则选择K值")
print("=" * 60)

# 尝试不同的K值
K_range = range(1, 11)
inertias = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# 绘制肘部曲线
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('簇数量 K')
plt.ylabel('SSE')
plt.title('肘部法则选择K值')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.show()

print("SSE值随K的变化:")
for k, inertia in zip(K_range, inertias):
    print(f"K={k}: SSE={inertia:.2f}")

# 肘点通常在K=3附近（因为数据就是3个簇生成的）

# ============================================
# 3. 轮廓系数评估聚类质量
# ============================================
print("\n" + "=" * 60)
print("示例3：轮廓系数评估")
print("=" * 60)

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: 轮廓系数={score:.4f}")

# 绘制轮廓系数
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('簇数量 K')
plt.ylabel('轮廓系数')
plt.title('轮廓系数选择K值')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================
# 4. 真实数据集：鸢尾花聚类
# ============================================
print("\n" + "=" * 60)
print("示例4：鸢尾花数据集聚类")
print("=" * 60)

# 加载数据
iris = load_iris()
X_iris = iris.data
y_iris = iris.target  # 真实标签（用于评估，聚类本身不需要）

# 标准化
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

# 聚类（我们知道有3类）
kmeans_iris = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans_iris.fit_predict(X_iris_scaled)

# 评估：调整兰德指数（比较聚类结果与真实标签的相似度）
ari = adjusted_rand_score(y_iris, cluster_labels)
print(f"调整兰德指数: {ari:.4f} (1.0表示完全一致，0表示随机)")

# 可视化（使用前两个特征）
plt.figure(figsize=(12, 5))

# 左图：真实标签
plt.subplot(1, 2, 1)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, cmap='viridis', 
           s=50, alpha=0.7, edgecolors='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('真实标签')
plt.grid(True, alpha=0.3)

# 右图：聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=cluster_labels, cmap='viridis', 
           s=50, alpha=0.7, edgecolors='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('K-Means聚类结果')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class KMeansManual:
    """
    手动实现的K-Means算法（完整版）
    """
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, n_init=10, 
                 init='k-means++', random_state=None):
        """
        初始化K-Means
        
        参数:
            n_clusters: 簇数量K
            max_iter: 最大迭代次数
            tol: 收敛阈值
            n_init: 随机初始化次数
            init: 初始化方法，'random'或'k-means++'
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
        
        # 结果
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
    def fit(self, X):
        """
        训练K-Means模型
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        
        print(f"开始训练K-Means...")
        print(f"样本数: {n_samples}, 特征数: {n_features}, K={self.n_clusters}")
        
        # 多次初始化
        for init in range(self.n_init):
            # 初始化簇中心
            if self.init == 'k-means++':
                centers = self._kmeans_plus_plus_init(X)
            else:  # 'random'
                indices = np.random.choice(n_samples, self.n_clusters, replace=False)
                centers = X[indices].copy()
            
            # 迭代优化
            for iteration in range(self.max_iter):
                # E步：分配样本到最近的簇
                distances = self._compute_distances(X, centers)
                labels = np.argmin(distances, axis=1)
                
                # M步：更新簇中心
                new_centers = np.zeros_like(centers)
                for k in range(self.n_clusters):
                    cluster_points = X[labels == k]
                    if len(cluster_points) > 0:
                        new_centers[k] = cluster_points.mean(axis=0)
                    else:
                        # 簇为空，重新初始化该中心
                        new_centers[k] = X[np.random.choice(n_samples)]
                
                # 检查收敛
                if np.linalg.norm(new_centers - centers) < self.tol:
                    centers = new_centers
                    labels = np.argmin(self._compute_distances(X, centers), axis=1)
                    break
                    
                centers = new_centers
            
            # 计算本次的SSE
            inertia = self._compute_inertia(X, centers, labels)
            
            # 保存最优结果
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_iter = iteration + 1
        
        # 保存最终结果
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter
        
        print(f"训练完成！最优SSE: {best_inertia:.4f}, 迭代次数: {best_iter}")
        return self
    
    def _kmeans_plus_plus_init(self, X):
        """
        k-means++初始化：使初始簇中心彼此距离较远
        """
        n_samples = X.shape[0]
        centers = np.zeros((self.n_clusters, X.shape[1]))
        
        # 1. 随机选择第一个中心
        first_idx = np.random.choice(n_samples)
        centers[0] = X[first_idx].copy()
        
        # 2. 选择后续中心
        for i in range(1, self.n_clusters):
            # 计算每个样本到最近已选中心的距离
            distances = self._compute_distances(X, centers[:i])
            min_distances = np.min(distances, axis=1)
            
            # 概率与距离平方成正比
            probabilities = min_distances / min_distances.sum()
            
            # 根据概率选择下一个中心
            next_idx = np.random.choice(n_samples, p=probabilities)
            centers[i] = X[next_idx].copy()
        
        return centers
    
    def _compute_distances(self, X, centers):
        """计算每个样本到每个中心的距离（欧氏距离平方）"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            diff = X - centers[k]
            distances[:, k] = np.sum(diff**2, axis=1)
        return distances
    
    def _compute_inertia(self, X, centers, labels):
        """计算SSE"""
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                diff = cluster_points - centers[k]
                inertia += np.sum(diff**2)
        return inertia
    
    def predict(self, X):
        """
        预测新样本的簇标签
        
        参数:
            X: 特征矩阵
            
        返回:
            簇标签数组
        """
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        """训练并返回簇标签"""
        self.fit(X)
        return self.labels_

# ============================================
# 测试手写实现
# ============================================
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    
    # 生成数据
    X, y_true = make_blobs(n_samples=200, centers=3, n_features=2, 
                           cluster_std=0.8, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用手写实现
    kmeans_manual = KMeansManual(n_clusters=3, n_init=10, random_state=42)
    labels_manual = kmeans_manual.fit_predict(X_scaled)
    
    # 使用sklearn实现
    from sklearn.cluster import KMeans
    kmeans_sklearn = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels_sklearn = kmeans_sklearn.fit_predict(X_scaled)
    
    # 比较结果
    ari_manual = adjusted_rand_score(y_true, labels_manual)
    ari_sklearn = adjusted_rand_score(y_true, labels_sklearn)
    
    print(f"\n手写实现调整兰德指数: {ari_manual:.4f}")
    print(f"sklearn实现调整兰德指数: {ari_sklearn:.4f}")
    print(f"SSE - 手写: {kmeans_manual.inertia_:.4f}, sklearn: {kmeans_sklearn.inertia_:.4f}")
```

---

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

def visualize_kmeans_steps(X, kmeans, title="K-Means迭代过程"):
    """
    可视化K-Means的迭代过程
    """
    # 由于sklearn的KMeans不保存每步历史，我们手动模拟
    n_samples = X.shape[0]
    centers = kmeans.cluster_centers_.copy()
    
    # 初始中心（假设用k-means++初始化）
    # 这里简化：直接用最终中心，然后反向模拟几步
    # 实际中，可以修改KMeans源码或使用自己的实现来保存历史
    
    # 简单可视化：绘制最终聚类结果和簇中心
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：聚类结果（颜色表示簇）
    scatter = axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', 
                              s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[0].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', 
                    s=200, label='簇中心')
    axes[0].set_xlabel('特征1')
    axes[0].set_ylabel('特征2')
    axes[0].set_title(f'{title} - 最终聚类结果')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：簇内平方误差（SSE）随K的变化（肘部法则）
    K_range = range(1, 11)
    inertias = []
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans_temp.fit(X)
        inertias.append(kmeans_temp.inertia_)
    
    axes[1].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('簇数量 K')
    axes[1].set_ylabel('SSE')
    axes[1].set_title('肘部法则')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(K_range)
    
    # 标注肘点（这里简单标注K=3）
    axes[1].axvline(x=3, color='r', linestyle='--', alpha=0.7, label='建议K=3')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return inertias

def plot_cluster_comparison(X, labels_true, labels_pred, titles):
    """
    比较真实标签和聚类结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 真实标签
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=labels_true, cmap='viridis', 
                                s=50, alpha=0.7, edgecolors='k')
    axes[0].set_title(titles[0])
    axes[0].set_xlabel('特征1')
    axes[0].set_ylabel('特征2')
    axes[0].grid(True, alpha=0.3)
    
    # 聚类结果
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=labels_pred, cmap='viridis', 
                                s=50, alpha=0.7, edgecolors='k')
    axes[1].set_title(titles[1])
    axes[1].set_xlabel('特征1')
    axes[1].set_ylabel('特征2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 运行可视化
print("=" * 60)
print("K-Means可视化")
print("=" * 60)

# 生成数据
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, 
                       cluster_std=0.8, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练模型
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_scaled)

# 可视化
visualize_kmeans_steps(X_scaled, kmeans, "K-Means聚类结果")

# 比较真实标签和聚类结果
plot_cluster_comparison(X_scaled, y_true, kmeans.labels_, 
                       ["真实标签", "K-Means聚类结果"])

# 可视化不同K值的结果
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
k_values = [2, 3, 4, 5, 6, 7]

for i, k in enumerate(k_values):
    ax = axes[i//3, i%3]
    kmeans_temp = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_temp, 
                        cmap='viridis', s=50, alpha=0.7, edgecolors='k')
    centers = kmeans_temp.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
    ax.set_title(f'K={k}')
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**结果理解**：
1. **左图**：显示聚类结果，不同颜色代表不同簇，红色X代表簇中心
2. **右图**：肘部法则曲线，SSE随K增大而减小，肘点（拐点）对应的K是最优选择
3. **比较图**：可以看到聚类结果与真实标签的相似程度
4. **不同K值**：K太小会欠拟合（簇太少），K太大会过拟合（簇太多）

---

## 10. 模型评估

```python
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                             davies_bouldin_score, adjusted_rand_score)
import numpy as np

def evaluate_clustering(X, labels, labels_true=None):
    """
    评估聚类结果（无监督评估和有监督评估）
    
    参数:
        X: 特征矩阵
        labels: 聚类标签
        labels_true: 真实标签（如果有，用于有监督评估）
    """
    print("=" * 60)
    print("聚类评估结果")
    print("=" * 60)
    
    # 1. 无监督评估指标（不需要真实标签）
    if len(np.unique(labels)) > 1:  # 至少需要2个簇
        # 轮廓系数：范围[-1,1]，越大越好
        silhouette = silhouette_score(X, labels)
        print(f"轮廓系数 (Silhouette Coefficient): {silhouette:.4f}")
        
        # Calinski-Harabasz指数：越大越好（簇间距离与簇内距离之比）
        calinski = calinski_harabasz_score(X, labels)
        print(f"Calinski-Harabasz指数: {calinski:.4f}")
        
        # Davies-Bouldin指数：越小越好
        davies = davies_bouldin_score(X, labels)
        print(f"Davies-Bouldin指数: {davies:.4f}")
    else:
        print("只有一个簇，无法计算聚类指标")
    
    # 2. 有监督评估指标（需要真实标签）
    if labels_true is not None:
        # 调整兰德指数：范围[-1,1]，1表示完全一致，0表示随机
        ari = adjusted_rand_score(labels_true, labels)
        print(f"\n调整兰德指数 (Adjusted Rand Index): {ari:.4f}")
        
        # 归一化互信息
        from sklearn.metrics import normalized_mutual_info_score
        nmi = normalized_mutual_info_score(labels_true, labels)
        print(f"归一化互信息 (Normalized Mutual Info): {nmi:.4f}")
    
    # 3. 簇统计信息
    unique_labels = np.unique(labels)
    print(f"\n簇数量: {len(unique_labels)}")
    for label in unique_labels:
        cluster_size = np.sum(labels == label)
        print(f"  簇 {label}: {cluster_size} 个样本 ({cluster_size/len(labels)*100:.1f}%)")
    
    return

# 评估示例
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # 生成数据
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, 
                           cluster_std=0.8, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 聚类
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # 评估
    evaluate_clustering(X_scaled, labels, y_true)
```

**聚类评估指标解释**：
1. **轮廓系数**：结合簇内紧密度和簇间分离度，范围[-1,1]，越大越好
2. **Calinski-Harabasz指数**：方差比，簇间方差与簇内方差之比，越大越好
3. **Davies-Bouldin指数**：簇的平均相似度，越小越好
4. **调整兰德指数**：比较聚类结果与真实标签的相似度，考虑随机因素
5. **归一化互信息**：基于信息熵的相似度度量

**注意**：无监督指标只能评估聚类结构的好坏，不能告诉你聚类是否有意义。必须结合领域知识解释。

---

## 11. 常见问题与易错点

### 11.1 聚类结果不稳定，每次运行结果不同
**原因**：随机初始化不同，可能收敛到不同的局部最优

**解决方案**：
```python
# 1. 增加n_init（默认10通常够用）
kmeans = KMeans(n_clusters=3, n_init=50, random_state=42)

# 2. 使用k-means++初始化（默认就是）
kmeans = KMeans(n_clusters=3, init='k-means++')

# 3. 设置随机种子
kmeans = KMeans(n_clusters=3, random_state=42)
```

### 11.2 特征未标准化，导致错误聚类
**问题**：如果特征尺度不同，尺度大的特征会主导距离计算

**正确做法**：
```python
from sklearn.preprocessing import StandardScaler

# 必须标准化！
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)  # 使用标准化后的数据
```

### 11.3 不知道K值应该取多少
**问题**：K-Means需要预先指定K，但实际中往往不知道

**解决方案**：
1. **肘部法则**：绘制SSE随K的变化，选择肘点（拐点）
2. **轮廓系数**：选择轮廓系数最大的K
3. **领域知识**：根据业务需求确定簇数量
4. **交叉验证**：对于不同K，在测试集上评估聚类质量

### 11.4 簇形状非球形，K-Means效果差
**问题**：K-Means假设簇是凸的、近似球形，对于月牙形、环形等簇效果差

**解决方案**：
```python
# 使用基于密度的聚类算法
from sklearn.cluster import DBSCAN, SpectralClustering

# DBSCAN可以处理任意形状的簇
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# 谱聚类也可以处理非球形簇
spectral = SpectralClustering(n_clusters=3, random_state=42)
labels = spectral.fit_predict(X_scaled)
```

### 11.5 数据有异常值，影响聚类质量
**问题**：K-Means使用均值作为簇中心，异常值会拉偏簇中心

**解决方案**：
1. **预处理**：检测并移除异常值
2. **使用鲁棒聚类**：K-Medoids（使用中位数而不是均值）
3. **使用基于密度的聚类**：DBSCAN对异常值鲁棒

---

## 12. 学习总结

### 核心要点回顾：
1. **迭代优化**：通过E步（分配）和M步（更新）交替优化
2. **目标函数**：最小化簇内平方误差和（SSE）
3. **收敛性**：算法保证收敛到局部最优（目标函数非增）
4. **初始化重要**：使用k-means++初始化，多次运行选择最好结果
5. **需要标准化**：特征尺度必须一致，否则聚类错误

### 从K-Means到其他聚类算法：
```
K-Means (原型聚类，球形簇)
    ↓
K-Medoids (使用中位数，对异常值鲁棒)
    ↓
高斯混合模型 (概率聚类，软分配)
    ↓
DBSCAN (基于密度，任意形状簇)
    ↓
谱聚类 (基于图论，处理复杂簇结构)
```

### 实践建议：
1. **总是标准化**：这是最重要的预处理步骤
2. **选择K值**：使用肘部法则、轮廓系数，结合领域知识
3. **多次运行**：设置较大的n_init，避免局部最优
4. **评估质量**：使用轮廓系数等指标，但更要结合业务解释
5. **考虑 alternatives**：如果簇形状复杂，考虑其他聚类算法

---

## 13. 练习题与思考题（含答案）

### 练习题

**习题1：基础计算**
问题：假设有6个样本在一条直线上，位置分别为：0, 1, 2, 3, 100, 101。如果K=2，手动执行K-Means算法。假设初始簇中心为1和100。计算最终簇分配和簇中心。

<details>
<summary>答案</summary>

初始簇中心：μ₁=1, μ₂=100

**迭代1**：
- 分配步骤：
  样本0到μ₁距离：|0-1|=1，到μ₂距离：|0-100|=100 → 分配到簇1
  样本1：|1-1|=0，|1-100|=99 → 簇1
  样本2：|2-1|=1，|2-100|=98 → 簇1
  样本3：|3-1|=2，|3-100|=97 → 簇1
  样本100：|100-1|=99，|100-100|=0 → 簇2
  样本101：|101-1|=100，|101-100|=1 → 簇2
  簇1：{0,1,2,3}，簇2：{100,101}

- 更新步骤：
  μ₁ = (0+1+2+3)/4 = 1.5
  μ₂ = (100+101)/2 = 100.5

**迭代2**：
- 分配步骤：重新计算距离，分配不变（因为簇中心在各自簇内）
- 簇中心不变

**最终结果**：
簇1：{0,1,2,3}，中心=1.5
簇2：{100,101}，中心=100.5
SSE = (0-1.5)²+(1-1.5)²+(2-1.5)²+(3-1.5)² + (100-100.5)²+(101-100.5)² = 5

如果初始中心选择不同，可能得到不同结果。
</details>

**习题2：编程实践**
问题：使用sklearn的KMeans对鸢尾花数据集进行聚类，并与真实标签比较（使用调整兰德指数）。

<details>
<summary>答案</summary>

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
X = iris.data
y_true = iris.target

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 聚类
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

# 评估
ari = adjusted_rand_score(y_true, y_pred)
print(f"调整兰德指数: {ari:.4f}")

# 可视化（使用前两个特征）
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='X', s=200)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('鸢尾花K-Means聚类')
plt.show()
```
</details>

**习题3：理论推导**
问题：证明对于K-Means，如果簇分配固定，那么使SSE最小的簇中心是该簇样本的均值。

<details>
<summary>答案</summary>

参见第3章3.4节的推导。

对簇 $k$ 的SSE关于 $\mu_k$ 求导：
$$\frac{\partial}{\partial \mu_k} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2 = \sum_{x_i \in C_k} (-2 x_i + 2 \mu_k) = -2 \sum_{x_i \in C_k} x_i + 2 |C_k| \mu_k$$

令导数为0：
$$-2 \sum_{x_i \in C_k} x_i + 2 |C_k| \mu_k = 0$$
$$\Rightarrow \mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

因此，最优簇中心确实是该簇样本的均值。
</details>

### 思考题

**思考题1**：K-Means和层次聚类、DBSCAN各有什么优缺点？适用什么场景？

<details>
<summary>答案</summary>

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| K-Means | 简单快速，适合大数据 | 需要K，假设球形簇，对异常值敏感 | 球形簇，大数据 |
| 层次聚类 | 不需要K，可以得到层次结构 | 计算复杂度高O(n³)，不适合大数据 | 小数据，需要层次结构 |
| DBSCAN | 不需要K，能处理任意形状簇，对异常值鲁棒 | 对参数(eps, min_samples)敏感，高维数据效果差 | 非球形簇，有噪声数据 |

核心区别：K-Means是原型聚类，层次聚类是连接聚类，DBSCAN是基于密度的聚类。
</details>

**思考题2**：为什么K-Means对特征尺度敏感？如何解决这个问题？

<details>
<summary>答案</summary>

K-Means基于欧氏距离：
$$d(x_i, \mu_k) = \sqrt{\sum_{j=1}^{d} (x_{ij} - \mu_{kj})^2}$$

如果特征尺度不同（例如特征1范围[0,1]，特征2范围[0,1000]），那么特征2会主导距离计算，因为特征2的差值平方远大于特征1。

**解决方法**：标准化（StandardScaler）使每个特征均值为0，方差为1，这样所有特征对距离的贡献相同。

另一种方法是使用马氏距离（考虑特征相关性），但K-Means通常配合标准化使用。
</details>

---

## 14. 学习路径建议

### 初级阶段（掌握K-Means基础）
1. 理解K-Means的E步和M步
2. 手工计算小样例的聚类过程
3. 使用sklearn实现K-Means聚类
4. 学习肘部法则选择K值

**学习时间**：2-3天

### 中级阶段（理解局限性和改进）
1. 理解K-Means的假设和局限性
2. 学习k-means++初始化算法
3. 比较K-Means与K-Medoids、高斯混合模型
4. 掌握聚类评估指标（轮廓系数等）

**学习时间**：1周

### 高级阶段（扩展到其他聚类算法）
1. 学习层次聚类、DBSCAN、谱聚类
2. 理解不同聚类算法的适用场景
3. 学习聚类集成（Ensemble Clustering）
4. 研究大规模数据上的聚类（Mini-Batch K-Means）

**学习时间**：2-3周

### 实践项目建议
1. **基础项目**：客户分群（使用K-Means）
2. **进阶项目**：图像颜色量化（使用K-Means压缩图像）
3. **挑战项目**：新闻文档聚类（结合TF-IDF和K-Means）

### 推荐资源
- **书籍**：《统计学习方法》（李航）第14章；《机器学习》（周志华）第9章
- **课程**：Andrew Ng的机器学习课程（聚类部分）
- **论文**：Lloyd (1982) K-Means原始论文；Arthur & Vassilvitskii (2007) k-means++
- **代码**：Scikit-learn源码中的KMeans实现
- **实践**：Kaggle聚类竞赛（如Customer Segmentation）
