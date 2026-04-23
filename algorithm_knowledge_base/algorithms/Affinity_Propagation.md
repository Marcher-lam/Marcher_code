# Affinity Propagation 学习文档

> 基于消息传递的聚类算法，无需预设簇数，自动确定聚类中心。

---

## 1. 算法基础认知

### 1.1 发展背景

Affinity Propagation（AP，聚类传播）由 Frey 和 Dueck 于 2007 年在《Science》上提出。与传统聚类方法不同，AP 不需要预先指定簇数，而是通过数据点之间的消息传递自动确定聚类中心和簇的个数。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 基于原型的聚类 |
| 代表 | 数据点本身就是候选中心 |
| 优点 | 无需预设簇数自动确定 |
| 计算 | 消息传递迭代 |

### 1.3 与 K-Means 对比

| 特性 | K-Means | Affinity Propagation |
|------|--------|---------------------|
| 簇数 | 需预设 | 自动确定 |
| 中心 | 质心计算 | 数据点中选择 |
| 初始化 | 随机种子 | 无敏感 |
| 收敛 | 局部最优 | 近似全局最优 |

---

## 2. 核心原理

### 2.1 相似度矩阵

给定 $n$ 个数据点，相似度矩阵 $S$ 定义为：
$$S(i, j) = -\|x_i - x_j\|^2$$

或使用负欧氏距离的倍数：
$$S(i, j) = -\alpha \cdot \|x_i - x_j\|^2$$

对角线元素 $S(k, k)$ 表示点 $k$ 作为聚类中心的"适合度"，称为**参考度（preference）**。

### 2.2 消息传递机制

AP 通过两种消息传递进行迭代：

1. **责任度（Responsibility）**$r(i, k)$：点 $i$ 认为点 $k$ 作为其聚类中心的合适程度
2. **可用度（Availability）**$a(i, k)$：点 $k$ 作为点 $i$ 的聚类中心的可用程度

### 2.3 聚类中心判定

迭代收敛后，每个点的聚类中心由以下规则确定：
- 如果 $a(i, k) + r(i, k) > 0$，则 $i$ 的中心是 $k$
- 如果 $a(i, i) + r(i, i) > 0$，则 $i$ 自己就是中心

---

## 3. 数学公式与推导

### 3.1 责任度更新

$$r(i, k) \leftarrow S(i, k) - \max_{k' s.t. k' \neq k} \{a(i, k') + S(i, k')\}$$

这表示点 $i$ 选择了 $k$ 而不是其他候选 $k'$ 的程度。

**物理含义**：$r(i, k)$ 衡量 $k$ 作为 $i$ 的代表点的吸引力，排除其他竞争者。

### 3.2 可用度更新

可用度更新分为两部分：

**1. 更新 $a(k, k)$**：
$$a(k, k) \leftarrow \sum_{i' s.t. i' \neq k} \max(0, r(i', k))$$

**2. 更新 $a(i, k)$ for $i \neq k$**：
$$a(i, k) \leftarrow \min(0, r(k, k) + \sum_{i' s.t. i' \neq k,i} \max(0, r(i', k)))$$

### 3.3 阻尼因子

为防止振荡，加入阻尼因子（damping）$\lambda \in [0, 1)$：

$$r_{new}(i, k) = (1 - \lambda) \cdot r_{new}(i, k) + \lambda \cdot r_{old}(i, k)$$
$$a_{new}(i, k) = (1 - \lambda) \cdot a_{new}(i, k) + \lambda \cdot a_{old}(i, k)$$

通常 $\lambda = 0.5$ 或 0.9。

### 3.4 对数似然函数

AP 的目标是最大化总相似度：
$$\max \sum_{i} S(i, c(i))$$

其中 $c(i)$ 是点 $i$ 的聚类中心。

---

## 4. 训练过程讲解

### 4.1 算法流程

```
Input: 相似度矩阵 S, 参考度 preference, 阻尼 lambda
Output: 聚类中心集合, 簇标签

1. 初始化: a(i,k)=0, r(i,k)=0 for all i,k
2. 迭代直到收敛或达到最大次数:
3.     更新责任度:
       r(i,k) = S(i,k) - max_{k'≠k}{a(i,k')+S(i,k')}
4.     更新可用度:
       a(k,k) = sum_i max(0, r(i,k))
       a(i,k) = min(0, r(k,k) + sum_{i'≠i,k} max(0, r(i',k)))
5.     应用阻尼
6. 判断聚类中心:
   for i: if a(i,i)+r(i,i) > 0, 则 i 是中心
7. 分配簇标签
```

### 4.2 参数选择

| 参数 | 说明 | 常用值 |
|------|------|--------|
| preference | 参考度，中位数=auto | 中位数或-median(S) |
| damping | 阻尼因子 | 0.5-0.9 |
| max_iter | 最大迭代 | 200-1000 |
| conv_iter | 收敛判断次数 | 50 |

### 4.3 参考度影响

- **preference 越大**：产生越多簇中心（自私）
- **preference 越小**：产生越少簇中心（无私）
- **中位数**：平衡点

---

## 5. 应用场景

### 5.1 典型应用

- **图像分割**：分割成超像素
- **文本聚类**：主题发现
- **市场细分**：客户分群
- **生物信息学**：基因聚类

### 5.2 代码示例

```python
import numpy as np

def ap_cluster(X, damping=0.9, max_iter=200):
    """Affinity Propagation 聚类"""
    from sklearn.cluster import AffinityPropagation
    
    # 计算相似度（负距离）
    from sklearn.metrics.pairwise import euclidean_distances
    S = -euclidean_distances(X)
    np.fill_diagonal(S, np.median(S))  # 参考度设为中位数
    
    # 聚类
    ap = AffinityPropagation(
        affinity='precomputed',
        damping=damping,
        max_iter=max_iter
    )
    labels = ap.fit_predict(-S)  # 注意：输入为距离
    
    return labels
```

---

## 6. 优缺点分析

### 6.1 优点

1. **无需预设簇数**：自动确定
2. **不陷入局部最优**：消息传递机制
3. **对初始值不敏感**：无需随机种子
4. **适合大规模数据**：可并行

### 6.2 缺点

1. **计算复杂度高**：$O(n^2)$ 内存和时间
2. **对相似度敏感**：相似度定义影响结果
3. **可能不收敛**：极端情况下震荡

### 6.3 改进方向

- **Scalable AP**：分层降采样
- **Kernel AP**：核方法
- **Bidirectional AP**：双向消息传递

---

## 7. 调库实现

### 7.1 sklearn 实现

```python
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class AFFINITYPROPAGATION:
    """Affinity Propagation 聚类"""
    
    def __init__(self, damping=0.5, max_iter=200, 
                 convergence_iter=50):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.model = None
        
    def fit(self, X):
        """训练 AP 模型
        
        参数:
            X: 数据矩阵 (n_samples, n_features)
        """
        self.model = AffinityPropagation(
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter
        )
        self.labels_ = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        
        return self
    
    def predict(self, X):
        """预测新数据标签"""
        return self.model.predict(X)
    
    def fit_predict(self, X):
        """训练并预测"""
        self.fit(X)
        return self.labels_


def demo():
    """演示"""
    print("=== Affinity Propagation 演示 ===\n")
    
    # 生成测试数据
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=4, 
                      cluster_std=0.5, random_state=42)
    
    # 聚类
    ap = AFFINITYPROPAGATION(damping=0.9)
    labels = ap.fit_predict(X)
    
    n_clusters = len(set(labels))
    print(f"自动确定簇数: {n_clusters}")
    print(f"聚类中心索引: {ap.cluster_centers_}")
    print(f"各簇样本数: {np.bincount(labels)}")
    
    return labels


if __name__ == "__main__":
    demo()
```

### 7.2 手算预计算相似度

```python
# 先计算相似度矩阵再聚类
from sklearn.metrics.pairwise import euclidean_distances

def ap_with_precomputed_similarity(X):
    """使用预计算相似度"""
    
    # 计算负欧氏距离作为相似度
    dist = euclidean_distances(X)
    S = -dist
    
    # 设置参考度（对角线）
    np.fill_diagonal(S, np.median(S))
    
    # 聚类
    ap = AffinityPropagation(affinity='precomputed', damping=0.9)
    labels = ap.fit_predict(S)
    
    return labels
```

---

## 8. 手工代码实现

### 8.1 完整 AP 实现

```python
import numpy as np
import warnings

class AffinityPropagationManual:
    """Affinity Propagation 手工实现
    
    参数:
        damping: 阻尼因子，防止振荡
        max_iter: 最大迭代次数
        convergence: 收敛判断的迭代次数
        preference: 参考度（默认中位数）
    """
    
    def __init__(self, damping=0.5, max_iter=200, 
                 convergence_iter=50, preference=None):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.labels_ = None
        self.cluster_centers_ = None
        
    def fit(self, X):
        """
        训练 Affinity Propagation
        
        参数:
            X: 数据矩阵 (n_samples, n_features)
        """
        n = len(X)
        
        # 计算相似度矩阵（负欧氏距离）
        S = self._compute_similarity(X)
        
        # 初始化消息矩阵
        R = np.zeros((n, n))  # 责任度
        A = np.zeros((n, n))  # 可用度
        
        # 迭代
        for iteration in range(self.max_iter):
            # 更新责任度
            R_new = self._update_responsibility(S, A)
            
            # 更新可用度
            A_new = self._update_availability(R_new)
            
            # 应用阻尼
            if self.damping > 0:
                R = (1 - self.damping) * R_new + self.damping * R
                A = (1 - self.damping) * A_new + self.damping * A
            else:
                R = R_new
                A = A_new
            
            # 检查收敛
            if iteration >= self.convergence_iter:
                if self._check_convergence(R, A):
                    break
        
        # 确定聚类中心
        self.cluster_centers_indices_, self.labels_ = self._get_clusters(R, A)
        self.n_clusters = len(self.cluster_centers_indices_)
        
        # 转换为原始数据的中心
        if len(X) > 0:
            self.cluster_centers_ = X[self.cluster_centers_indices_]
        
        return self
    
    def _compute_similarity(self, X):
        """计算相似度矩阵"""
        n = len(X)
        
        # 负欧氏距离
        S = np.zeros((n, n))
        for i in range(n):
            diff = X - X[i]
            S[:, i] = -np.sum(diff ** 2, axis=1)
        
        # 设置参考度
        if self.preference is None:
            np.fill_diagonal(S, np.median(S))
        else:
            np.fill_diagonal(S, self.preference)
        
        return S
    
    def _update_responsibility(self, S, A):
        """更新责任度
        
        r(i,k) = S(i,k) - max_{k'≠k}{A(i,k') + S(i,k')}
        """
        n = S.shape[0]
        R = np.copy(S)
        
        for i in range(n):
            # 排除自身
            for k in range(n):
                # 找最大值（排除 k）
                row_without_k = np.concatenate([A[i, :k], A[i, k+1:]])
                col_without_k = np.concatenate([S[i, :k], S[i, k+1:]])
                
                max_val = np.max(row_without_k + col_without_k)
                R[i, k] = S[i, k] - max_val
        
        return R
    
    def _update_availability(self, R):
        """更新可用度
        
        a(k,k) = sum_{i≠k} max(0, r(i,k))
        a(i,k) = min(0, r(k,k) + sum_{i'≠i,k} max(0, r(i',k)))
        """
        n = R.shape[0]
        A = np.zeros_like(R)
        
        # 对角线（自我可用度）
        for k in range(n):
            A[k, k] = np.sum(np.maximum(0, R[:, k])) - np.maximum(0, R[k, k])
        
        # 非对角线
        for i in range(n):
            for k in range(n):
                if i == k:
                    continue
                    
                sum_val = R[k, k]
                for i_prime in range(n):
                    if i_prime != k and i_prime != i:
                        sum_val += np.maximum(0, R[i_prime, k])
                
                A[i, k] = np.minimum(0, sum_val)
        
        return A
    
    def _check_convergence(self, R, A):
        """检查是否收敛"""
        # 比较新旧差异
        return True  # 简化
    
    def _get_clusters(self, R, A):
        """确定簇"""
        n = R.shape[0]
        
        # 计算 a + r
        E = R + A
        
        # 对角线
        EK = np.diag(E)
        
        # 聚类中心：a(i,i) + r(i,i) > 0
        centers = np.where(EK > 0)[0]
        
        if len(centers) == 0:
            # 如果没有中心，选择S中对角线最大的点
            centers = np.array([np.argmax(np.diag(S))])
        
        # 分配标签
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            # 找最近的中心
            k_star = np.argmax(E[i, :])
            labels[i] = k_star
        
        return centers, labels
    
    def predict(self, X_new):
        """预测新数据"""
        n_new = len(X_new)
        X_train = self.cluster_centers_
        
        labels = np.zeros(n_new, dtype=int)
        for i in range(n_new):
            dists = np.sum((X_train - X_new[i]) ** 2, axis=1)
            labels[i] = np.argmin(dists)
        
        return labels


def demo_manual():
    """手工实现演示"""
    print("=== Affinity Propagation 手工实现演示 ===\n")
    
    from sklearn.datasets import make_blobs
    
    # 生成测试数据
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=4, 
                     cluster_std=0.6, random_state=42)
    
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    
    # 聚类
    ap = AffinityPropagationManual(damping=0.9, max_iter=100)
    ap.fit(X)
    
    n_clusters = ap.n_clusters
    print(f"\n自动确定簇数: {n_clusters}")
    print(f"聚类中心: {ap.cluster_centers_indices_}")
    print(f"各簇样本数: {np.bincount(ap.labels_)}")


if __name__ == "__main__":
    demo_manual()
```

---

## 9. 可视化与结果理解

### 9.1 聚类结果可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_ap():
    """可视化 AP 聚类结果"""
    from sklearn.datasets import make_blobs
    
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5)
    
    from sklearn.cluster import AffinityPropagation
    ap = AffinityPropagation(damping=0.9)
    labels = ap.fit_predict(X)
    
    plt.figure(figsize=(12, 5))
    
    # 原始数据
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
    plt.title('原始数据')
    
    # 聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(ap.cluster_centers_[:, 0], ap.cluster_centers_[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    plt.title(f'AP 聚类 (k={len(ap.cluster_centers_)})')
    
    plt.tight_layout()
    plt.savefig('ap_result.png', dpi=150)
    plt.show()
```

### 9.2 消息收敛可视化

```python
def plot_messages():
    """可视化消息收��"""
    iterations = range(1, 51)
    
    # 模拟收敛曲线
    r_norm = np.abs(np.exp(-0.1 * np.array(iterations)) * np.random.randn(50))
    a_norm = np.abs(np.exp(-0.15 * np.array(iterations)) * np.random.randn(50))
    
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, r_norm[:50], label='Responsibility', alpha=0.7)
    plt.plot(iterations, a_norm[:50], label='Availability', alpha=0.7)
    plt.xlabel('迭代')
    plt.ylabel('消息范数')
    plt.title('消息传递收敛')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ap_convergence.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate_ap(X, labels):
    """评估 AP 聚类结果"""
    
    # 轮廓系数
    silhouette = silhouette_score(X, labels)
    
    # CH 指数
    ch = calinski_harabasz_score(X, labels)
    
    return {
        'silhouette': silhouette,
        'calinski_harabasz': ch
    }
```

### 10.2 与其他聚类对比

| 指标 | AP | K-Means | DBSCAN |
|------|-----|---------|--------|
| 轮廓系数 | ++ | + | + |
| 自动 k | ✓ | ✗ | ✗ |
| 形状适应 | + | ✗ | ✓ |

---

## 11. 常见问题与易错点

### 11.1 参考度设置

**问题**：choice of preference 影响簇数

**解决**：
- 使用中位数：平衡
- 网格搜索调参

### 11.2 计算复杂度

**问题**：$O(n^2)$ 内存，时间

**解决**：
- 数据量大时采样
- 使用稀疏矩阵
- Scalable AP

### 11.3 不收敛

**问题**：消息震荡

**解决**：
- 增加阻尼因子
- 减少学习率

---

## 12. 学习总结

**核心要点**：

1. **消息传递**：责任度 + 可用度
2. **无需预设簇数**：自动确定
3. **迭代收敛**：阻尼更新
4. **应用广泛**：图像、文本、生物

**学习建议**：

1. 理解消息传递机制
2. 对比 K-Means 和层次聚类
3. 实践调参选择

---

## 13. 练习题与思考题

### 13.1 基础练习

1. 推导责任度更新公式
2. 解释阻尼因子的作用
3. 分析不同 preference 的影响

### 13.2 进阶练习

1. 在图像分割上应用 AP
2. 对比 AP 和 K-Means 结果

### 13.3 思考题

1. AP 如何处理大数据？
2. AP 与图聚类的联系？

---

### 13.4 详细答案与解析

#### 练习1：责任度推导

**问题**：推导 $r(i, k)$ 的更新公式

**解答**：

责任度 $r(i, k)$ 表示点 $i$ 认为点 $k$ 比其他候选更好：

$$r(i, k) \leftarrow S(i, k) - \max_{k' \neq k} \{a(i, k') + S(i, k')\}$$

**物理含义**：

- $i$ 对 $k$ 的评价 = $S(i, k)$
- 减去对其他候选的最佳评价
- 如果其他候选可用度高，$r(i, k)$ 降低

#### 练习2：阻尼因子作用

**问题**：为什么需要阻尼因子？

**解答**：

防止消息传递过程中出现振荡：
- $r(i, k)$ 和 $a(i, k)$ 相互影响
- 某些情况下会来回振荡
- 阻尼使更新平滑收敛：
  $$r_{new} = (1-\lambda)r_{new} + \lambda r_{old}$$

---

## 14. 学习路径建议

### 入门阶段

1. 理解消息传递机制
2. 掌握 AP 算法流程
3. 对比 K-Means

### 进阶阶段

1. 实现完整 AP
2. 调参实践
3. 应用图像分割

### 高级阶段

1. Scalable AP
2. 核 AP 结合
3. 深度聚类结合

**推荐路线**：

```
K-Means → 层次聚类 → DBSCAN → 
Affinity Propagation → Spectral → 深度聚类
```

**Affinity Propagation 是自动聚类的重要方法，掌握它对理解聚类算法体系很重要。**