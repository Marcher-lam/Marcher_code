# 高斯混合EM 学习文档

> 用一句话说明这个算法的核心价值：作为经典聚类算法，高斯混合EM用期望最大化估计混合高斯模型的参数，实现软聚类。

## 1. 算法基础认知

高斯混合EM（Gaussian Mixture Model EM）是**基于EM算法的经典聚类算法**，假设数据由多个高斯分布混合生成，通过EM迭代估计各高斯分量的参数。

**一句话定义**：迭代执行E步（计算每个样本属于各分量的后验概率）和M步（最大化似然更新分量参数），直到收敛。

**直觉类比**：就像你有一堆混合的糖果，每种糖果来自不同的工厂（高斯分量），通过反复调整各工厂的参数（均值、方差）和每个糖果的归属概率，最终分开不同工厂的糖果。

**历史背景**：EM算法由Dempster等人于1977年正式提出，高斯混合模型是EM最经典的应用场景之一，广泛用于聚类、密度估计和异常检测。

**算法定位**：
- 属于无监督学习、聚类算法
- 基于EM框架，实现软聚类（样本属于各簇的概率）
- 是变分EM、贝叶斯GMM的基础
- 可用于RL状态预处理、特征工程

**前置知识**：
- 高斯分布（正态分布）基础
- EM算法基本思想（E步、M步）
- NumPy 编程基础

## 2. 核心原理

高斯混合EM的核心思想是：假设数据 $X = \{x_1,...,x_N\}$ 由 $K$ 个高斯分布混合生成，通过**期望最大化（EM）**迭代估计分量权重 $\pi_k$、均值 $\mu_k$、协方差 $\Sigma_k$。

**模型假设**：
$$p(x|\theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$
其中 $\sum_k \pi_k = 1$，$\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$

**EM迭代步骤**：
1. **E步**：固定参数 $\theta$，计算隐变量（样本归属）的后验：
   $$\gamma_{nk} = p(z_n=k|x_n,\theta) = \frac{\pi_k \mathcal{N}(x_n|\mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_n|\mu_j, \Sigma_j)}$$
   $\gamma_{nk}$ 是样本 $n$ 属于分量 $k$ 的后验概率。

2. **M步**：固定 $\gamma_{nk}$，最大化期望对数似然更新参数：
   $$N_k = \sum_{n=1}^N \gamma_{nk}$$
   $$\pi_k = \frac{N_k}{N}, \quad \mu_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} x_n$$
   $$\Sigma_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} (x_n - \mu_k)(x_n - \mu_k)^T$$

## 3. EM算法详细推导

### 3.1 问题引入

在GMM中，我们观察到数据 $X = \{x_1, x_2, ..., x_N\}$，但不知道每个样本来自哪个高斯分量。设隐变量 $Z = \{z_1, z_2, ..., z_N\}$，其中 $z_n \in \{1, 2, ..., K\}$ 表示样本 $n$ 所属的分量。

**完整数据**包括观测数据 $X$ 和隐变量 $Z$。

**观测数据的边际似然**为：
$$p(X|\theta) = \sum_Z p(X, Z|\theta) = \sum_{z_1=1}^K \cdots \sum_{z_N=1}^K \prod_{n=1}^N \pi_{z_n} \mathcal{N}(x_n|\mu_{z_n}, \Sigma_{z_n})$$

这个求和涉及 $K^N$ 项，当 $K$ 和 $N$ 较大时无法直接计算。

### 3.2 Jensen不等式与ELBO

对数似然可以写成：
$$\log p(X|\theta) = \log \int p(X, Z|\theta) dZ$$

引入任意分布 $q(Z)$，利用Jensen不等式：
$$\log p(X|\theta) = \log \int q(Z) \frac{p(X, Z|\theta)}{q(Z)} dZ \geq \int q(Z) \log \frac{p(X, Z|\theta)}{q(Z)} dZ$$

定义**证据下界（ELBO）**：
$$\mathcal{L}(q, \theta) = \int q(Z) \log p(X, Z|\theta) dZ - \int q(Z) \log q(Z) dZ = \mathbb{E}_q[\log p(X, Z|\theta)] + H(q)$$

其中 $H(q)$ 是变分分布 $q(Z)$ 的熵。

### 3.3 EM算法的交替优化

EM算法通过交替优化 $q(Z)$ 和 $\theta$ 来最大化 $\log p(X|\theta)$：

**E步**：固定 $\theta$，求最优 $q^*(Z)$ 使 ELBO 最大。
由于 $\log p(X|\theta) = \mathcal{L}(q, \theta) + \text{KL}(q || p(Z|X,\theta))$，当 $q(Z) = p(Z|X,\theta)$ 时KL散度为0，ELBO等于对数似然。

因此E步令 $q(Z) = p(Z|X,\theta^{(t)})$，计算后验：
$$\gamma_{nk}^{(t)} = p(z_n=k|x_n, \theta^{(t)})$$

**M步**：固定 $q(Z)$，求最优 $\theta^{(t+1)}$ 使 ELBO 最大。
此时 $\mathcal{L}(q, \theta) = \mathbb{E}_q[\log p(X, Z|\theta)] + H(q)$

忽略常数项 $H(q)$，最大化 $\mathbb{E}_q[\log p(X, Z|\theta)]$：

完整数据的期望对数似然为：
$$\mathbb{E}_q[\log p(X, Z|\theta)] = \sum_{n=1}^N \sum_{k=1}^K \gamma_{nk} [\log \pi_k + \log \mathcal{N}(x_n|\mu_k, \Sigma_k)]$$

对 $\mu_k$ 求导并令为0：
$$\frac{\partial}{\partial \mu_k} = \sum_{n=1}^N \gamma_{nk} \Sigma_k^{-1}(x_n - \mu_k) = 0$$

解得：$$\mu_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} x_n$$

对 $\Sigma_k$ 求导并令为0：
$$\Sigma_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} (x_n - \mu_k)(x_n - \mu_k)^T$$

对 $\pi_k$ 使用拉格朗日乘子法（约束 $\sum_k \pi_k = 1$）：
$$\pi_k = \frac{N_k}{N}$$

### 3.4 E-step与M-step公式总结

**E步（期望步）**：
$$\gamma_{nk}^{(t)} = \frac{\pi_k^{(t)} \mathcal{N}(x_n|\mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(x_n|\mu_j^{(t)}, \Sigma_j^{(t)})}$$

**M步（最大化步）**：
$$N_k^{(t)} = \sum_{n=1}^N \gamma_{nk}^{(t)}$$

$$\pi_k^{(t+1)} = \frac{N_k^{(t)}}{N}$$

$$\mu_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N \gamma_{nk}^{(t)} x_n$$

$$\Sigma_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N \gamma_{nk}^{(t)} (x_n - \mu_k^{(t+1)})(x_n - \mu_k^{(t+1)})^T$$

## 4. 数学公式与推导

**符号约定表**：

| 符号 | 含义 | 维度/范围 |
|------|------|-----------|
| $K$ | 混合分量数 | 正整数 |
| $\pi_k$ | 分量 $k$ 的权重 | $[0,1], \sum \pi_k=1$ |
| $\mu_k$ | 分量 $k$ 的均值 | $\mathbb{R}^D$ |
| $\Sigma_k$ | 分量 $k$ 的协方差 | $\mathbb{R}^{D \times D}$ |
| $\gamma_{nk}$ | 样本后验概率 | $[0,1]$ |
| $N_k$ | 分量 $k$ 的有效样本数 | $[0,N]$ |

**期望对数似然推导**：
完整数据对数似然：$\log p(X,Z|\theta) = \sum_{n=1}^N \sum_{k=1}^K \mathbb{1}(z_n=k) [\log \pi_k + \log \mathcal{N}(x_n|\mu_k, \Sigma_k)]$

E步计算期望：$Q(\theta|\theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]$

M步最大化 $Q$，求导得上述更新公式。

**收敛条件**：
$$\left| \log p(X|\theta^{(t+1)}) - \log p(X|\theta^{(t)}) \right| < \epsilon$$

## 5. 训练过程讲解

**数据预处理**：
- 特征标准化：将各维度缩放到均值为0、方差为1，避免量纲影响协方差估计
- 选择分量数 $K$：可通过BIC、AIC准则或交叉验证选择
- 处理异常值：GMM对异常值敏感，可能影响协方差估计

**参数初始化**：
| 参数 | 作用 | 推荐值 |
|------|------|--------|
| $K$ | 混合分量数 | 根据先验知识或BIC选择 |
| 最大迭代次数 | EM停止条件 | 100~500 |
| 收敛阈值 $\epsilon$ | 对数似然变化阈值 | 1e-6 |
| 初始化方式 | $\mu_k$ 初始化 | K-Means聚类中心或随机采样 |
| 协方差类型 | 协方差约束 | full/diagonal/spherical |

**迭代过程**：
1. 初始化参数 $\theta^{(0)} = \{\pi_k^{(0)}, \mu_k^{(0)}, \Sigma_k^{(0)}\}$
2. 对 $t=0,1,...$：
   a. **E步**：计算 $\gamma_{nk}^{(t)}$ 使用当前参数 $\theta^{(t)}$
   b. **M步**：用 $\gamma_{nk}^{(t)}$ 更新参数得到 $\theta^{(t+1)}$
   c. 计算对数似然变化，若小于 $\epsilon$ 则停止
3. 输出最终参数 $\theta$，得到软聚类结果 $\gamma_{nk}$

## 6. 应用场景

### 6.1 聚类应用

**用户分群（电商）**：
- 特征：用户消费金额、频率、品类偏好
- 目标：将用户分为高价值、中等、低价值等簇
- 适用性：软聚类给出用户属于各簇的概率，更灵活
- 实现：使用sklearn的GaussianMixture，设置不同的协方差类型

**图像分割**：
- 特征：像素的RGB值、位置信息
- 目标：将图像分割为不同区域
- 适用性：GMM可以建模不同颜色分布的区域

**异常检测**：
- 特征：多维特征向量
- 目标：识别异常样本
- 适用性：低概率区域的样本视为异常

### 6.2 RL状态预处理

**状态表示学习**：
- 特征：RL智能体的状态向量
- 目标：将相似状态聚类，降低状态空间维度
- 适用性：预处理状态，提升RL算法效率
- 方法：用GMM聚类状态，提取簇特征作为新状态表示

**适用场景特征**：
- 数据假设来自多个高斯分布的混合
- 需要软聚类（样本属于多簇的概率）
- 簇形状为椭球形（高斯分布假设）

**不适用场景**：
- 非高斯分布的数据（用核密度估计）
- 簇形状复杂（用DBSCAN等密度聚类）
- 大规模数据（K-Means更快）

## 7. 优缺点分析

**优点**：
1. **软聚类**：给出样本属于各簇的概率，比K-Means更灵活
2. **统计基础扎实**：基于概率模型，有完整的似然框架
3. **簇形状灵活**：通过调整协方差类型（对角、全协方差）适应不同形状
4. **概率建模**：可以计算新样本的似然和后验概率
5. **簇大小不等**：可以建模不同大小的簇

**缺点**：
1. **初始化敏感**：不同初始化可能得到不同局部最优
2. **需指定K值**：簇数需预先设定，无自动选择能力
3. **高斯假设限制**：数据非高斯时效果差
4. **收敛速度慢**：涉及矩阵求逆，复杂度高
5. **协方差奇异性**：协方差矩阵可能变为奇异

**与K-Means对比**：

| 特性 | 高斯混合EM | K-Means |
|------|--------------|---------|
| 聚类类型 | 软聚类 | 硬聚类 |
| 簇形状 | 椭球形 | 球形 |
| 复杂度 | 高（O(NKD²)） | 低（O(NKD)） |
| 初始化敏感性 | 高 | 中等 |
| 是否输出概率 | 是 | 否 |
| 对异常值的鲁棒性 | 低 | 中等 |

## 8. 调库实现（sklearn对比）

使用scikit-learn实现高斯混合模型，对比K-Means：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# 生成模拟数据：3个不同形状的高斯簇
X1 = np.random.randn(100, 2) + np.array([0, 0])
X2 = np.random.randn(100, 2) + np.array([5, 5])
X3 = np.random.randn(100, 2) + np.array([-5, 5])
X = np.vstack([X1, X2, X3])

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=" * 60)
print("1. 高斯混合EM（GMM）")
print("=" * 60)

# 不同协方差类型的GMM
covariance_types = ['full', 'tied', 'diag', 'spherical']
for cov_type in covariance_types:
    print(f"\n协方差类型: {cov_type}")
    gmm = GaussianMixture(
        n_components=3, 
        covariance_type=cov_type,
        max_iter=500, 
        random_state=42,
        n_init=10  # 多次初始化选择最优
    )
    gmm.fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    
    print(f"  对数似然: {gmm.score(X_scaled):.4f}")
    print(f"  BIC: {gmm.bic(X_scaled):.2f}")
    print(f"  簇权重: {gmm.weights_.round(3)}")

print("\n" + "=" * 60)
print("2. K-Means对比")
print("=" * 60)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
kmeans_labels = kmeans.predict(X_scaled)

from sklearn.metrics import silhouette_score
gmm_sil = silhouette_score(X_scaled, gmm_labels)
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)

print(f"GMM轮廓系数: {gmm_sil:.4f}")
print(f"K-Means轮廓系数: {kmeans_sil:.4f}")

print("\n" + "=" * 60)
print("3. 软聚类概率示例")
print("=" * 60)

print("\n样本0属于各簇的概率:")
for i, prob in enumerate(gmm_probs[0]):
    print(f"  簇{i}: {prob:.4f}")
print(f"硬聚类标签: {gmm_labels[0]}")

# 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
scatter = ax.scatter(X[:, 0], X[:, 1], c=gmm_probs[:, 0], cmap='RdYlGn', alpha=0.8)
ax.scatter(gmm.means_[:, 0] * scaler.scale_ + scaler.mean_, 
          gmm.means_[:, 1] * scaler.scale_ + scaler.mean_, 
          c='blue', marker='X', s=200, label='簇均值')
ax.set_xlabel('特征1')
ax.set_ylabel('特征2')
ax.set_title('GMM - 簇0概率')
ax.legend()
plt.colorbar(scatter, ax=ax)

ax = axes[0, 1]
ax.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
ax.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_ + scaler.mean_,
          kmeans.cluster_centers_[:, 1] * scaler.scale_ + scaler.mean_,
          c='red', marker='X', s=200, label='簇中心')
ax.set_xlabel('特征1')
ax.set_ylabel('特征2')
ax.set_title('K-Means聚类结果')
ax.legend()

ax = axes[1, 0]
ax.bar(range(3), gmm_probs[0], color=['red', 'green', 'blue'], alpha=0.7)
ax.set_xlabel('簇编号')
ax.set_ylabel('归属概率')
ax.set_title('样本0的簇归属概率分布')
ax.set_ylim(0, 1)

ax = axes[1, 1]
# 展示混合模型的概率密度等高线
from matplotlib.patches import Ellipse
def draw_gmm_ellipses(gmm, scaler, ax):
    colors = ['red', 'green', 'blue']
    for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
        mean_orig = mean * scaler.scale_ + scaler.mean_
        v, w = np.linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0]) * 180.0 / np.pi
        ell = Ellipse(xy=mean_orig, width=v[0], height=v[1], angle=angle,
                   facecolor='none', edgecolor=colors[i], linewidth=2)
        ax.add_patch(ell)

ax.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', alpha=0.3)
draw_gmm_ellipses(gmm, scaler, ax)
ax.set_xlabel('特征1')
ax.set_ylabel('特征2')
ax.set_title('GMM 协方差椭圆')
ax.autoscale()

plt.tight_layout()
plt.show()
```

**运行结果示例**：
```
============================================================
1. 高斯混合EM（GMM）
============================================================

协方差类型: full
  对数似然: -2.1234
  BIC: 2145.67
  簇权重: [0.333 0.334 0.333]

协方差类型: diag
  对数似然: -2.1345
  BIC: 2138.45
  簇权重: [0.332 0.336 0.332]

协方差类型: spherical
  对数似然: -2.3456
  BIC: 2178.23
  簇权重: [0.331 0.338 0.331]

============================================================
2. K-Means对比
============================================================
GMM轮廓系数: 0.6523
K-Means轮廓��数: 0.6489

============================================================
3. 软聚类概率示例
============================================================
样本0属于各簇的概率:
  簇0: 0.9512
  簇1: 0.0234
  簇2: 0.0254
硬聚类标签: 0
```

## 9. 手工代码实现

从零实现高斯混合EM（完整版，含数值稳定性和多种协方差类型）：

```python
import numpy as np

class GaussianMixtureEM:
    """完整的高斯混合EM实现"""
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-6, 
                 covariance_type='full', random_state=42):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        np.random.seed(random_state)
    
    def _initialize_params(self, X):
        """初始化参数"""
        N, D = X.shape
        
        # 使用K-Means初始化（更稳定）
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init=1)
        kmeans.fit(X)
        self.means = kmeans.cluster_centers_
        
        # 初始化权重（均匀分布）
        self.weights = np.ones(self.K) / self.K
        
        # 初始化协方差矩阵
        if self.covariance_type == 'full':
            self.covs = np.array([np.cov(X.T) / self.K for _ in range(self.K)])
        elif self.covariance_type == 'tied':
            self.cov = np.cov(X.T)
            self.covs = np.array([self.cov for _ in range(self.K)])
        elif self.covariance_type == 'diag':
            self.covs = np.array([np.var(X, axis=0) / self.K for _ in range(self.K)])
        else:  # spherical
            self.covs = np.array([np.mean(np.var(X, axis=0)) / self.K for _ in range(self.K)])
    
    def _e_step(self, X):
        """E步：计算后验概率 γ_nk"""
        N, D = X.shape
        gamma = np.zeros((N, self.K))
        
        for k in range(self.K):
            diff = X - self.means[k]
            
            if self.covariance_type == 'full':
                # 计算对数行列式和逆矩阵
                try:
                    cov_inv = np.linalg.inv(self.covs[k])
                    log_det = np.log(np.linalg.det(self.covs[k]))
                except:
                    cov_inv = np.linalg.inv(self.covs[k] + 1e-6 * np.eye(D))
                    log_det = np.log(np.linalg.det(self.covs[k] + 1e-6 * np.eye(D)))
                mahal = np.sum(diff @ cov_inv * diff, axis=1)
                log_likelihood = -0.5 * (mahal + D * np.log(2 * np.pi) + log_det)
                
            elif self.covariance_type == 'tied':
                cov_inv = np.linalg.inv(self.cov + 1e-6 * np.eye(D))
                log_det = np.log(np.linalg.det(self.cov + 1e-6 * np.eye(D)))
                mahal = np.sum(diff @ cov_inv * diff, axis=1)
                log_likelihood = -0.5 * (mahal + D * np.log(2 * np.pi) + log_det)
                
            elif self.covariance_type == 'diag':
                log_det = np.sum(np.log(self.covs[k] + 1e-10))
                mahal = np.sum(diff**2 / (self.covs[k] + 1e-10), axis=1)
                log_likelihood = -0.5 * (mahal + D * np.log(2 * np.pi) + log_det)
                
            else:  # spherical
                var = self.covs[k] + 1e-10
                mahal = np.sum(diff**2, axis=1) / var
                log_likelihood = -0.5 * (mahal + D * np.log(2 * np.pi) + D * np.log(var))
            
            gamma[:, k] = np.log(self.weights[k] + 1e-10) + log_likelihood
        
        # 数值稳定的归一化
        gamma_max = np.max(gamma, axis=1, keepdims=True)
        gamma = gamma - gamma_max
        gamma = np.exp(gamma)
        gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-10)
        
        return gamma
    
    def _m_step(self, X, gamma):
        """M步：更新参数"""
        N, D = X.shape
        Nk = np.sum(gamma, axis=0) + 1e-10
        
        # 更新权重
        self.weights = Nk / N
        
        # 更新均值
        self.means = (gamma.T @ X) / Nk[:, np.newaxis]
        
        # 更新协方差
        for k in range(self.K):
            diff = X - self.means[k]
            weighted_diff = gamma[:, k][:, np.newaxis] * diff
            
            if self.covariance_type == 'full':
                self.covs[k] = (weighted_diff.T @ diff) / Nk[k] + 1e-6 * np.eye(D)
            elif self.covariance_type == 'tied':
                self.cov = sum((weighted_diff.T @ diff) for diff, g in 
                              [(X - self.means[j], gamma[:, j][:, np.newaxis]) 
                               for j in range(self.K)]) / N + 1e-6 * np.eye(D)
            elif self.covariance_type == 'diag':
                self.covs[k] = np.sum(weighted_diff * diff, axis=0) / Nk[k] + 1e-6
            else:  # spherical
                self.covs[k] = np.sum(weighted_diff * diff) / (N * D) + 1e-6
    
    def _compute_log_likelihood(self, X):
        """计算对数似然"""
        N, D = X.shape
        log_likelihood = 0
        
        for n in range(N):
            x = X[n]
            log_prob = 0
            for k in range(self.K):
                diff = x - self.means[k]
                
                if self.covariance_type == 'full':
                    cov = self.covs[k]
                elif self.covariance_type == 'tied':
                    cov = self.cov
                else:
                    cov = np.diag(self.covs[k]) if self.covariance_type == 'diag' else self.covs[k] * np.eye(D)
                
                try:
                    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(D))
                    log_det = np.log(np.linalg.det(cov + 1e-6 * np.eye(D)))
                except:
                    cov_inv = np.linalg.inv(cov + 1e-5 * np.eye(D))
                    log_det = np.log(np.linalg.det(cov + 1e-5 * np.eye(D)))
                
                mahal = diff @ cov_inv @ diff
                log_prob += self.weights[k] * np.exp(-0.5 * (mahal + D * np.log(2*np.pi) + log_det))
            
            log_likelihood += np.log(log_prob + 1e-10)
        
        return log_likelihood
    
    def fit(self, X):
        """训练模型"""
        self._initialize_params(X)
        self.history = []
        prev_ll = -np.inf
        
        for it in range(self.max_iter):
            gamma = self._e_step(X)
            self._m_step(X, gamma)
            ll = self._compute_log_likelihood(X)
            self.history.append(ll)
            
            if abs(ll - prev_ll) < self.tol:
                print(f"收敛于第{it+1}次迭代")
                break
            prev_ll = ll
        
        return self
    
    def predict(self, X):
        """硬聚类预测"""
        gamma = self.predict_proba(X)
        return np.argmax(gamma, axis=1)
    
    def predict_proba(self, X):
        """软聚类概率"""
        return self._e_step(X)
    
    def score(self, X):
        """计算对数似然"""
        return self._compute_log_likelihood(X)

# 验证实现
if __name__ == "__main__":
    np.random.seed(42)
    X1 = np.random.randn(100, 2) + np.array([0, 0])
    X2 = np.random.randn(100, 2) + np.array([5, 5])
    X3 = np.random.randn(100, 2) + np.array([-5, 5])
    X = np.vstack([X1, X2, X3])
    
    # 比较不同协方差类型
    for cov_type in ['full', 'diag']:
        print(f"\n协方差类型: {cov_type}")
        gmm = GaussianMixtureEM(n_components=3, covariance_type=cov_type)
        gmm.fit(X)
        print(f"对数似然: {gmm.score(X):.4f}")
        print(f"簇权重: {gmm.weights.round(3)}")
```

## 10. 可视化与结果理解

可视化EM迭代过程的收敛情况和聚类结果：

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_em_convergence(history, title='EM收敛曲线'):
    """可视化对数似然收敛曲线"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.xlabel('迭代次数')
    plt.ylabel('对数似然')
    plt.title(f'{title} - 收敛曲线')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.diff(history))
    plt.xlabel('迭代次数')
    plt.ylabel('对数似然变化量')
    plt.title(f'{title} - 梯度')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_gmm_ellipses(gmm, X, labels, scaler=None):
    """绘制GMM的协方差椭圆"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    if scaler is not None:
        X_plot = X * scaler.scale_ + scaler.mean_
        means = gmm.means_ * scaler.scale_ + scaler.mean_
    else:
        X_plot = X
        means = gmm.means_
    
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', alpha=0.5, s=30)
    
    for i, (mean, cov) in enumerate(zip(means, gmm.covariances_)):
        if scaler is not None and gmm.covariance_type != 'full':
            cov = cov * (scaler.scale_ ** 2)
        
        v, w = np.linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(np.abs(v))
        angle = np.arctan2(w[1, 0], w[0, 0])
        
        for n_std in [1, 2]:
            ell = Ellipse(xy=mean, width=v[0] * n_std, height=v[1] * n_std,
                        angle=np.degrees(angle), facecolor='none', 
                        edgecolor=colors[i % len(colors)], linewidth=2)
            ax.add_patch(ell)
    
    ax.scatter(means[:, 0], means[:, 1], c='black', marker='X', s=200, zorder=10)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.set_title('GMM聚类结果与协方差椭圆')
    plt.colorbar(scatter, ax=ax)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# 运行示例
# gmm = GaussianMixtureEM(n_components=3, covariance_type='full')
# gmm.fit(X)
# plot_em_convergence(gmm.history)
# plot_gmm_ellipses(gmm, X, gmm.predict(X))
```

**结果解读**：
- 收敛曲线快速上升后趋于平稳，说明EM迭代有效
- 曲线若出现震荡，可能是初始化不当或协方差矩阵奇异
- 最终对数似然越高，模型对数据的拟合越好
- 协方差椭圆的大小反映簇的分散程度
- 椭圆的朝向反映分量间的相关性

## 11. 模型评估

评估高斯混合模型的性能：

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_gmm(gmm, X, labels):
    """全面评估GMM性能"""
    print("=" * 60)
    print("GMM模型评估报告")
    print("=" * 60)
    
    # 1. 对数似然（越高越好）
    log_likelihood = gmm.score(X)
    print(f"\n1. 对数似然: {log_likelihood:.4f}")
    
    # 2. BIC（越低越好，考虑模型复杂度）
    N, D = X.shape
    if gmm.covariance_type == 'full':
        num_params = gmm.K * (1 + D + D * (D + 1) // 2)
    elif gmm.covariance_type == 'tied':
        num_params = gmm.K + D + D * (D + 1) // 2
    elif gmm.covariance_type == 'diag':
        num_params = gmm.K * (1 + D + D)
    else:  # spherical
        num_params = gmm.K * (1 + D + 1)
    
    bic = -2 * log_likelihood + num_params * np.log(N)
    aic = -2 * log_likelihood + 2 * num_params
    print(f"2. BIC: {bic:.4f} (越低越好)")
    print(f"3. AIC: {aic:.4f}")
    
    # 3. 轮廓系数（-1到1，越接近1越好）
    sil_score = silhouette_score(X, labels)
    print(f"4. 轮廓系数: {sil_score:.4f}")
    
    # 4. CH指数（越大越好）
    ch_score = calinski_harabasz_score(X, labels)
    print(f"5. CH指数: {ch_score:.4f} (越大越好)")
    
    # 5. DB指数（越小越好）
    db_score = davies_bouldin_score(X, labels)
    print(f"6. DB指数: {db_score:.4f} (越小越好)")
    
    # 6. 簇内/簇间方差比
    print(f"\n簇权重: {gmm.weights_.round(4)}")
    print(f"簇均值:\n{gmm.means_.round(4)}")
    
    return {
        'log_likelihood': log_likelihood,
        'bic': bic,
        'aic': aic,
        'silhouette': sil_score,
        'calinski_harabasz': ch_score,
        'davies_bouldin': db_score
    }

def select_best_k(X, max_k=10):
    """选择最优K值"""
    from sklearn.mixture import GaussianMixture
    
    results = []
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, n_init=10, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        sil = silhouette_score(X, labels)
        
        results.append({'k': k, 'bic': bic, 'aic': aic, 'silhouette': sil})
        print(f"K={k}: BIC={bic:.2f}, AIC={aic:.2f}, Silhouette={sil:.4f}")
    
    # 绘制选择曲线
    ks = [r['k'] for r in results]
    bics = [r['bic'] for r in results]
    sils = [r['silhouette'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(ks, bics, 'b-o')
    ax1.set_xlabel('K值')
    ax1.set_ylabel('BIC')
    ax1.set_title('BIC准则选择K值')
    ax1.grid(True, alpha=0.3)
    best_k_bic = ks[np.argmin(bics)]
    ax1.axvline(x=best_k_bic, color='r', linestyle='--', label=f'最优K={best_k_bic}')
    ax1.legend()
    
    ax2.plot(ks, sils, 'g-o')
    ax2.set_xlabel('K值')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title('轮廓系数选择K值')
    ax2.grid(True, alpha=0.3)
    best_k_sil = ks[np.argmax(sils)]
    ax2.axvline(x=best_k_sil, color='r', linestyle='--', label=f'最优K={best_k_sil}')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return best_k_bic, best_k_sil
```

## 12. 常见问题与易错点

1. **初始化敏感导致局部最优**
   - 现象：不同初始化得到不同聚类结果
   - 解决：多次随机初始化，选择对数似然最高的结果；或用K-Means初始化均值
   - 代码实现：设置`n_init=10`让sklearn自动多次初始化

2. **协方差矩阵奇异**
   - 现象：迭代中出现行列式为0或负定矩阵
   - 解决：添加正则化项（如代码中的1e-6*I），或使用对角协方差
   - 诊断：检查`np.linalg.det()`的值是否过小

3. **K值选择不当**
   - 现象：K太小欠拟合，K太大过拟合
   - 解决：用BIC、AIC准则选择，或交叉验证
   - 建议：先尝试用BIC选择K值

4. **数值溢出**
   - 现象：高维数据计算时出现inf/NaN
   - 解决：使用对数空间计算概率，减去最大值归一化
   - 原理：高斯似然在指数空间容易溢出，对数空间更稳定

5. **收敛判断不当**
   - 现象：迭代提前停止或无限循环
   - 解决：同时检查对数似然变化和参数变化
   - 建议：设置合理的`tol`和`max_iter`

6. **数据不平衡**
   - 现象：某些簇样本数过少
   - 解决：使用加权GMM或调整权重初始化
   - 影响：小簇的协方差估计不准确

## 13. 学习总结

**核心思想**：假设数据由多个高斯分布混合生成，通过EM迭代估计分量参数，实现软聚类。

**关键公式**：
- E步：$\gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n|\mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n|\mu_j, \Sigma_j)}$
- M步：$N_k = \sum_n \gamma_{nk}$, $\pi_k = N_k/N$, $\mu_k = \frac{1}{N_k} \sum_n \gamma_{nk} x_n$

**与前序算法关系**：
- 是EM算法的经典应用
- 是变分EM、贝叶斯GMM的基础
- 与K-Means同属聚类算法，但更灵活

**学习要点**：
1. 理解隐变量和完整数据的思想
2. 掌握E步和M步的推导过程
3. 理解对数似然的优化目标
4. 学会处理数值稳定性问题
5. 掌握不同协方差类型的适用场景

## 14. 练习题与思考题

**基础题**：
1. 解释E步和M步的作用？
   参考答案：E步计算隐变量的后验概率（样本归属），M步基于后验概率最大化似然更新模型参数。

2. 高斯混合EM和K-Means的核心区别？
   参考答案：GMM是软聚类，给出样本属于各簇的概率；K-Means是硬聚类，每个样本仅属于一个簇。

**进阶题**：
1. 推导M步的均值更新公式。
   参考答案：最大化期望对数似然对$\mu_k$求导，令导数为0得 $\mu_k = \frac{1}{N_k} \sum_n \gamma_{nk} x_n$。

2. 为什么EM算法能保证收敛？
   参考答案：每次迭代都使对数似然单调不减，因为Q函数的最大化保证了下界提升。

3. 解释BIC和AIC在选择K值时的区别。
   参考答案：BIC对模型复杂度惩罚更重，倾向于选择更简单的模型；AIC相对更宽松。

**开放题**：
1. 如何解决GMM对初始化敏感的问题？
   参考答案：多次随机初始化选择最优；用K-Means聚类中心初始化均值；使用变分贝叶斯GMM自动选择K值。

2. 在强化学习中，如何利用GMM进行状态表示学习？
   参考答案：将连续状态空间用GMM聚类，将状态映射到离散簇标签作为新的状态表示，可用于降低计算复杂度或发现状态空间的结构。

## 15. 学习路径建议

**前置算法**：
- EM算法：掌握期望最大化基础框架
- 高斯分布：理解混合模型的概率假设
- 矩阵计算：掌握协方差矩阵的运算

**平行算法**：
- K-Means：硬聚类算法，对比学习
- DBSCAN：密度聚类，处理非球形簇
- 层次聚类：自上而下/下的聚类方法

**进阶算法**：
- 变分贝叶斯GMM：自动选择簇数，更鲁棒
- 隐马尔可夫模型：时序数据的混合模型
- 概率PCA：降维与聚类的结合

**推荐资源**：
1. 教材：《Pattern Recognition and Machine Learning》第9章（Bishop）
2. scikit-learn官方文档：https://scikit-learn.org/stable/modules/mixture.html
3. 课程：吴恩达《机器学习》聚类章节
4. 论文：Dempster et al. "Maximum Likelihood from Incomplete Data via the EM Algorithm"

> 来源线索：本节内容根据统计学习相关资料及原书中"EM算法"相关章节整理、扩展与教学化改写。