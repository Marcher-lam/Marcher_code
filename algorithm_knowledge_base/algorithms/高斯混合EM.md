# 高斯混合EM 学习文档

> 期望最大化算法在混合高斯模型中的应用

---

## 1. 算法基础认知

### 1.1 一句话定义

高斯混合模型（Gaussian Mixture Model, GMM）和期望最大化（Expectation-Maximization, EM）算法是聚类和概率密度估计的核心方法，通过迭代E步（计算隶属度）和M步（更新参数）来拟合数据。

### 1.2 直觉类比

想象你在一个昏暗的房间里看一群人：

- **K-means**：强行把每人归为一类（硬划分）
- **GMM**：每人以不同概率属于多个类别（软划分）

GMM EM就像"猜猜这堆点有几个簇"——通过迭代猜测和完善，最终找到最佳拟合。

### 1.3 发展背景

- 1977年：Dempster, Laird, Rubin 提出EM算法
- 2000年代：GMM成为语音识别标配
- 2010年后：深度学习时代仍是重要基线

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 聚类/密度估计 |
| 模型 | 混合高斯分布 |
| 方法 | EM迭代优化 |
| 输出 | 软硬聚类结果 |

---

## 2. 核心原理

### 2.1 问题定义

给定数据 {x₁, x₂, ..., xₙ}，假设由K个高斯混合生成：

$$P(x) = \sum_{k=1}^{K} \pi_k \cdot N(x | \mu_k, \Sigma_k)$$

其中：
- πk：混合系数（权重），∑πk = 1
- μk：第k个高斯的均值
- Σk：第k个高斯的协方差矩阵

### 2.2 EM算法流程

```
输入: 数据X, K个分量
初始化: πk, μk, Σk

重复 until 收敛:
    E步: 计算每个数据点属于每个分量的后验概率
         γ(zik) = πk * N(xi | μk, Σk) / Σj πj * N(xi | μj, Σj)
    
    M步: 更新参数
         πk = Nk / N  (Nk = Σi γ(zik))
         μk = Σi γ(zik) * xi / Nk
         Σk = Σi γ(zik) * (xi-μk)(xi-μk)T / Nk

输出: 聚类结果和模型参数
```

### 2.3 关键概念

**E步（期望步）**：
计算"数据点i属于簇k的概率"

$$\gamma_{ik} = \frac{\pi_k N(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j N(x_j | \mu_j, \Sigma_j)}$$

**M步（最大化步）**：
更新参数使期望-likelihood最大化

$$Q(\theta) = \sum_{i} \sum_{k} \gamma_{ik} \log P(x_i, z_i=k | \theta)$$

---

## 3. 数学公式与推导

### 3.1 完整数据似然

完整数据的对数似然：

$$\log P(X, Z | \theta) = \sum_{i=1}^{N} \sum_{k=1}^{K} z_{ik} [\log \pi_k + \log N(x_i | \mu_k, \Sigma_k)]$$

其中 zik 是隐变量，表示xi属于簇k。

### 3.2 E步推导

给定当前参数θ^t，E步计算后验概率：

$$\gamma_{ik}^{(t)} = P(z_{ik}=1 | x_i, \theta^{(t)})$$

$$= \frac{\pi_k^{(t)} N(x_i | \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} N(x_j | \mu_j^{(t)}, \Sigma_j^{(t)})}$$

这是贝叶斯定理的直接应用。

### 3.3 M步推导

M步最大化Q函数：

对μk求导设为0：

$$\mu_k^{(t+1)} = \frac{\sum_i \gamma_{ik}^{(t)} x_i}{\sum_i \gamma_{ik}^{(t)}}$$

对Σk求导（简化 spherical）：

$$\sigma_k^{(t+1)} = \frac{\sum_i \gamma_{ik}^{(t)} ||x_i - \mu_k||^2}{d \sum_i \gamma_{ik}^{(t)}}$$

其中d是数据维度。

对πk用拉格朗日乘数法：

$$\pi_k^{(t+1)} = \frac{\sum_i \gamma_{ik}^{(t)}}{N}$$

### 3.4 收敛性证明

EM算法保证：
- 每次迭代：log P(X | θ^(t+1)) ≥ log P(X | θ^t)
- 收敛到局部最优（全局最优需要多次初始化）

---

## 4. PyTorch实现

### 4.1 基础GMM EM

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class GaussianMixtureEM(nn.Module):
    """GMM with EM algorithm"""
    
    def __init__(self, n_features, n_clusters, eps=1e-8):
        super().__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.eps = eps
        
        # 混合系数 (K,)
        self.logits = nn.Parameter(torch.randn(n_clusters))
        
        # 均值 (K, D)
        self.means = nn.Parameter(torch.randn(n_clusters, n_features))
        
        # 协方差 (K, D, D) - 下三角矩阵
        self.raw_cov = nn.Parameter(torch.eye(n_features).unsqueeze(0).repeat(n_clusters, 1, 1))
    
    @property
    def weights(self):
        return F.softmax(self.logits, dim=0)
    
    @property
    def covs(self):
        # 简化为对角协方差
        return torch.exp(self.raw_cov)
    
    def mvn_sample(self, mean, cov):
        """从多元正态分布采样"""
        return MultivariateNormal(mean, cov).sample()
    
    def e_step(self, x):
        """E步：计算后验概率"""
        # (batch,) + (K,) -> (K, batch)
        log_weights = torch.log(self.weights + self.eps)
        
        log_probs = []
        for k in range(self.n_clusters):
            mvn = MultivariateNormal(self.means[k], torch.diag(self.covs[k]))
            log_prob = mvn.log_prob(x)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=0)  # (K, batch)
        
        # 归一化
        log_probs_full = log_weights.unsqueeze(1) + log_probs
        log_probs_norm = torch.logsumexp(log_probs_full, dim=0)
        
        # 后验概率
        gamma = torch.exp(log_probs_full - log_probs_norm.unsqueeze(0))
        
        return gamma.T  # (batch, K)
    
    def m_step(self, x, gamma):
        """M步：更新参数"""
        N = x.size(0)
        Nk = gamma.sum(dim=0)  # (K,)
        
        # 更新权重
        new_logits = torch.log(Nk + self.eps)
        self.logits.data = new_logits
        
        # 更新均值
        new_means = (gamma.T @ x) / (Nk.unsqueeze(1) + self.eps)
        self.means.data = new_means
        
        # 更新协方差
        new_covs = []
        for k in range(self.n_clusters):
            if Nk[k] < 1:
                new_covs.append(torch.eye(self.n_features))
                continue
            
            diff = x - self.means[k]
            cov = (gamma[:, k:k+1] * diff).T @ diff / (Nk[k] + self.eps)
            cov = torch.diag(cov).clamp(min=self.eps)
            new_covs.append(cov)
        
        self.raw_cov.data = torch.log(torch.stack(new_covs) + self.eps)
    
    def fit(self, x, n_iterations=100):
        """训练"""
        self.train()
        
        for i in range(n_iterations):
            # E步
            gamma = self.e_step(x)
            
            # 计算loss
            loss = -torch.mean(gamma)
            
            # M步
            self.m_step(x, gamma)
            
            if (i + 1) % 20 == 0:
                print(f"Iter {i+1}, Loss: {loss.item():.4f}")
        
        return self
    
    def predict(self, x):
        """预测聚类标签"""
        self.eval()
        with torch.no_grad():
            gamma = self.e_step(x)
            return torch.argmax(gamma, dim=1)
    
    def predict_proba(self, x):
        """预测后验概率"""
        self.eval()
        with torch.no_grad():
            return self.e_step(x)
```

### 4.2 完整实现（带初始化）

```python
class GMM:
    """完整的GMM EM实现"""
    
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.labels_ = None
    
    def _initialize(self, X):
        """初始化：使用K-means结果"""
        torch.manual_seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # 使用torch.kmeans初始化
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        km_labels = kmeans.fit_predict(X.numpy())
        
        # 初始化参数
        self.weights_ = torch.ones(self.n_clusters) / self.n_clusters
        self.means_ = torch.tensor(kmeans.cluster_centers_)
        
        # 计算协方差
        self.covariances_ = []
        for k in range(self.n_clusters):
            mask = torch.tensor(km_labels == k)
            if mask.sum() < 1:
                self.covariances_.append(torch.eye(n_features))
            else:
                Xk = X[mask]
                cov = torch.cov(Xk.T) + torch.eye(n_features) * 1e-4
                self.covariances_.append(cov)
        self.covariances_ = torch.stack(self.covariances_)
    
    def _e_step(self, X):
        """E步"""
        log_weights = torch.log(self.weights_ + 1e-8)
        
        log_probs = []
        for k in range(self.n_clusters):
            mvn = MultivariateNormal(self.means_[k], self.covariances_[k])
            log_probs.append(mvn.log_prob(X))
        
        log_probs = torch.stack(log_probs, dim=0)
        log_probs_full = log_weights.unsqueeze(1) + log_probs
        log_probs_norm = torch.logsumexp(log_probs_full, dim=0)
        
        return torch.exp(log_probs_full - log_probs_norm.unsqueeze(0))
    
    def _m_step(self, X, gamma):
        """M步"""
        N = X.size(0)
        
        Nk = gamma.sum(dim=0)
        
        # 更新权重
        self.weights_ = Nk / N
        
        # 更新均值
        for k in range(self.n_clusters):
            if Nk[k] < 1:
                continue
            self.means_[k] = (gamma[:, k:k+1] * X).sum(dim=0) / Nk[k]
        
        # 更新协方差
        for k in range(self.n_clusters):
            if Nk[k] < 1:
                continue
            diff = X - self.means_[k]
            cov = (gamma[:, k:k+1] * diff).T @ diff / Nk[k]
            cov = torch.diag(torch.diag(cov)) + torch.eye(X.size(1)) * 1e-4
            self.covariances_[k] = cov
    
    def fit(self, X):
        """训练"""
        self._initialize(X)
        
        prev_log_likelihood = float('-inf')
        
        for iteration in range(self.max_iter):
            # E步
            gamma = self._e_step(X)
            
            # M步
            self._m_step(X, gamma)
            
            # 计算对数似然
            log_probs = []
            for k in range(self.n_clusters):
                mvn = MultivariateNormal(self.means_[k], self.covariances_[k])
                log_probs.append(mvn.log_prob(X))
            log_probs = torch.stack(log_probs)
            log_likelihood = torch.sum(
                torch.logsumexp(torch.log(self.weights_[:, None]) + log_probs, dim=0)
            )
            
            if iteration % 10 == 0:
                print(f"Iter {iteration}: Log-likelihood: {log_likelihood.item():.2f}")
            
            # 收敛检查
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged at iteration {iteration}")
                break
            
            prev_log_likelihood = log_likelihood.item()
        
        self.labels_ = torch.argmax(gamma, dim=1)
        
        return self
    
    def predict(self, X):
        """预测"""
        gamma = self._e_step(X)
        return torch.argmax(gamma, dim=1)
    
    def predict_proba(self, X):
        """预测概率"""
        return self._e_step(X)
```

---

## 5. 代码示例

### 5.1 聚类实验

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def demo_gmm():
    print("=== GMM EM 聚类演示 ===\n")
    
    # 生成数据
    X, y = make_blobs(
        n_samples=300, 
        centers=3, 
        n_features=2,
        cluster_std=0.8,
        random_state=42
    )
    X = torch.tensor(X, dtype=torch.float32)
    
    # 训练GMM
    gmm = GMM(n_clusters=3, max_iter=100)
    gmm.fit(X)
    
    # 预测
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 原始数据
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5)
    axes[0].set_title('Ground Truth')
    
    # GMM聚类
    axes[1].scatter(X[:, 0], X[:, 1], c=labels.numpy(), cmap='viridis', alpha=0.5)
    axes[1].scatter(
        gmm.means_[:, 0], gmm.means_[:, 1], 
        c='red', marker='x', s=100
    )
    axes[1].set_title('GMM Clustering')
    
    # 置信椭圆
    for k in range(3):
        cov = gmm.covariances_[k].numpy()
        mean = gmm.means_[k].numpy()
        
        # 绘制椭圆
        from matplotlib.patches import Ellipse
        import numpy as np
        
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(eigenvals)
        
        ellipse = Ellipse(
            mean, width, height, angle,
            fill=False, edgecolor='red'
        )
        axes[1].add_patch(ellipse)
    
    axes[2].imshow(probs.numpy().T, aspect='auto', cmap='viridis')
    axes[2].set_title('Cluster Probabilities')
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Cluster')
    
    plt.tight_layout()
    plt.savefig('gmm_demo.png', dpi=100)
    print(f"图表已保存到 gmm_demo.png")


if __name__ == "__main__":
    demo_gmm()
```

### 5.2 密度估计

```python
def demo_density_estimation():
    print("=== 密度估计演示 ===\n")
    
    # 加载真实数据
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = torch.tensor(iris.data[:, :2], dtype=torch.float32)  # 只用前两个特征
    
    # 训练GMM进行密度估计
    gmm = GMM(n_clusters=5, max_iter=100)
    gmm.fit(X)
    
    # 生成新数据样本
    from torch.distributions import MixtureSameFamily, Categorical
    
    mix = Categorical(gmm.weights_)
    comp = MultivariateNormal(gmm.means_, gmm.covariances_)
    gmm_dist = MixtureSameFamily(mix, comp)
    
    # 采样
    samples = gmm_dist.sample((100,))
    
    print(f"采样形状: {samples.shape}")
    print(f"原始数据范围: [{X.min():.2f}, {X.max():.2f}]")
    print(f"采样数据范围: [{samples.min():.2f}, {samples.max():.2f}]")
    
    # 计算对数似然（密度估计质量）
    log_prob = gmm_dist.log_prob(X)
    print(f"平均对数似然: {log_prob.mean():.2f}")


if __name__ == "__main__":
    demo_density_estimation()
```

### 5.3 scikit-learn版本

```python
from sklearn.mixture import GaussianMixture

def demo_sklearn():
    print("=== scikit-learn GMM ===\n")
    
    # 数据
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=200, centers=3, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    
    # 使用scikit-learn
    gmm = GaussianMixture(
        n_components=3, 
        covariance_type='full',
        max_iter=100,
        random_state=42
    )
    gmm.fit(X.numpy())
    
    # 预测
    labels = gmm.predict(X.numpy())
    probs = gmm.predict_proba(X.numpy())
    
    print(f"收敛: {gmm.converged_}")
    print(f"迭代次数: {gmm.n_iter_}")
    print(f"对数似然: {gmm.score(X.numpy()) * X.shape[0]:.2f}")


if __name__ == "__main__":
    demo_sklearn()
```

---

## 6. 变体

### 6.1 变分EM（Variational EM）

```python
class VariationalGMM:
    """变分GMM - 使用变分推断"""
    
    def __init__(self, n_clusters, alpha=1.0):
        self.n_clusters = n_clusters
        self.alpha = alpha  # Dirichlet先验参数
    
    def fit(self, X, max_iter=100):
        """变分EM"""
        N, D = X.shape
        
        # 初始化变分参数
        phi = torch.rand(N, self.n_clusters)
        phi = phi / phi.sum(dim=1, keepdim=True)
        
        # Dirichlet参数
        alpha_k = torch.ones(self.n_clusters) * self.alpha
        
        for iteration in range(max_iter):
            # E步：更新phi
            for k in range(self.n_clusters):
                # 更新phi
                pass
            
            # M步
            pass
        
        return self
```

### 6.2 贝叶斯GMM

```python
class BayesianGMM:
    """贝叶斯GMM - 使用采样"""
    
    def __init__(self, n_clusters, n_samples=100):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
    
    def fit(self, X):
        """Gibbs采样"""
        N, D = X.shape
        
        # 初始化
        assignments = torch.randint(0, self.n_clusters, (N,))
        
        for sample in range(self.n_samples):
            # 采样分配
            for i in range(N):
                pass
        
        return self
```

### 6.3 在线GMM

```python
class OnlineGMM:
    """在线GMM - 适合流数据"""
    
    def __init__(self, n_clusters, decay=0.99):
        self.n_clusters = n_clusters
        self.decay = decay
    
    def update(self, x):
        """增量更新"""
        # E步
        gamma = self._e_step(x.unsqueeze(0))
        
        # M步（滑动平均）
        self.weights_ = self.decay * self.weights_ + (1 - self.decay) * gamma
        # 更新均值和协方差
        pass
```

---

## 7. 常见问题

### Q1: GMM vs K-means 区别？

- GMM：软聚类（输出概率）
- K-means：硬聚类（确定标签）
- GMM：适合椭圆形簇
- K-means：适合球形簇

### Q2: 如何选择K？

- 使用BIC/AIC准则
- 使用交叉验证
- 观察对数似然拐点

### Q3: 陷入局部最优？

- 多次随机初始化
- 使用K-means初始化
- 尝试不同初始位置

### Q4: 协方差类型选择？

- 'full'：完整协方差（灵活但参数多）
- 'tied'：共享协方差
- 'diag'：对角协方差
- 'spherical'：单方差

---

## 8. 练习题

### 选择题

1. E步计算什么？
   - A) 参数更新   B) 后验概率   C) 距离
   - **答案：B**

2. K-means和GMM的主要区别？
   - A) K不同   B) 软/硬聚类   C) 维度
   - **答案：B**

3. EM算法保证收敛吗？
   - A) 全局最优   B) 局部最优   C) 不收敛
   - **答案：B**

### 简答题

1. 解释EM算法的E步和M步？

   **答案**：E步计算隐变量的后验概率，M步更新参数最大化期望似然

2. 为什么需要多次初始化？

   **答案**：EM只保证局部最优，不同初始化可能得到不同结果

### 编程题

1. 实现带BIC准则的K选择：

```python
def select_k(X, max_k=10):
    """使用BIC选择最优K"""
    bic_scores = []
    
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k)
        gmm.fit(X)
        
        # BIC = -2 * log_likelihood + k * log(n) * d
        log_likelihood = gmm.score(X) * X.shape[0]
        n_params = k * (X.shape[1] + 1) - 1
        bic = -2 * log_likelihood + n_params * np.log(X.shape[0])
        bic_scores.append(bic)
    
    return np.argmin(bic_scores) + 1
```

---

## 9. 学习路径

### 9.1 算法关系

```
EM算法
    ↓
├─ GMM → 聚类
├─ HMM → 序列
└─ LDA → 主题模型
```

### 9.2 扩展

| 算法 | 场景 |
|------|------|
| K-means++ | 更好初始化 |
| Mini-batch GMM | 大数据 |
| 变分EM | 大规模 |

---

## 10. 附录

### A. scikit-learn参数

| 参数 | 说明 |
|------|------|
| n_components | K |
| covariance_type | 协方差类型 |
| max_iter | 最大迭代 |
| n_init | 初始化次数 |

### B. 参考

- Dempster et al. (1977). "Maximum Likelihood from Incomplete Data"
- sklearn.mixture.GaussianMixture

---

**文档结束**