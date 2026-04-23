# 高斯混合EM 学习文档

## 1. 算法基础认知

高斯混合模型（Gaussian Mixture Model, GMM）是一种概率聚类模型，它假设数据由若干个高斯分布混合生成。GMM 的参数估计通常使用 EM 算法，这便是"高斯混合EM"。与 K-Means 的硬分配不同，GMM 为每个样本给出属于各簇的概率（软分配），能捕捉更丰富的数据结构。

## 2. 核心原理

GMM 假设数据 $X$ 的生成过程为：

1. 以概率 $\pi_k$（混合权重，$\sum_k \pi_k = 1$）选择第 $k$ 个高斯分量
2. 从该高斯分布 $\mathcal{N}(\mu_k, \Sigma_k)$ 中采样生成样本

因此数据的概率密度为：

$$P(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x | \mu_k, \Sigma_k)$$

隐变量 $z_i$ 表示样本 $x_i$ 来自哪个分量，EM 算法通过迭代估计 $\{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$。

## 3. 数学公式与推导

**多元高斯分布**：

$$\mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right)$$

**完全数据对数似然**：

$$\log P(X, Z | \theta) = \sum_{i=1}^{n} \sum_{k=1}^{K} z_{ik} \left[\log \pi_k + \log \mathcal{N}(x_i | \mu_k, \Sigma_k)\right]$$

**E 步——计算响应度（Responsibility）**：

$$\gamma_{ik} = P(z_i = k | x_i, \theta) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

$\gamma_{ik}$ 表示样本 $i$ 属于分量 $k$ 的后验概率。

**M 步——更新参数**：

$$N_k = \sum_{i=1}^{n} \gamma_{ik}$$

$$\mu_k^{new} = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} x_i$$

$$\Sigma_k^{new} = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} (x_i - \mu_k^{new})(x_i - \mu_k^{new})^T$$

$$\pi_k^{new} = \frac{N_k}{n}$$

## 4. 训练过程讲解

1. **初始化**：用 K-Means 结果或随机方式初始化 $\mu_k, \Sigma_k, \pi_k$
2. **E 步**：计算每个样本属于每个高斯分量的响应度 $\gamma_{ik}$
3. **M 步**：用响应度作为"软权重"更新均值、协方差和混合权重
4. **计算对数似然**：$\ell(\theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)$
5. **判断收敛**：对数似然变化小于阈值则停止

## 5. 应用场景

- 客户分群（比 K-Means 更灵活，能处理不同大小/形状的簇）
- 异常检测（低概率样本视为异常）
- 图像分割
- 语音识别中的声学建模
- 广告系统中用户行为的多模态建模

## 6. 优缺点分析

**优点**：
- 软分配，提供每个样本属于各簇的概率
- 能拟合不同形状、大小的簇（通过协方差矩阵）
- 有坚实的概率论基础
- 可通过 BIC/AIC 选择簇数 $K$

**缺点**：
- 假设各分量是高斯分布，对非高斯数据效果差
- 对初始化敏感，可能收敛到差的局部最优
- 协方差矩阵参数量大，高维时容易过拟合
- 计算复杂度高于 K-Means

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=[0.5, 1.0, 1.5], random_state=42)

gmm = GaussianMixture(n_components=3, covariance_type='full', max_iter=200,
                       n_init=5, random_state=42, reg_covar=1e-6)
gmm.fit(X)

labels = gmm.predict(X)
probs = gmm.predict_proba(X)

print(f"Means:\n{gmm.means_.round(2)}")
print(f"Weights: {gmm.weights_.round(3)}")
print(f"Converged: {gmm.converged_}, Iterations: {gmm.n_iter_}")
print(f"Sample probabilities (first 5):\n{probs[:5].round(3)}")

bic = gmm.bic(X)
aic = gmm.aic(X)
print(f"BIC: {bic:.2f}, AIC: {aic:.2f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from scipy.stats import multivariate_normal

class GMM_EM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_params(self, X):
        np.random.seed(self.random_state)
        n, d = X.shape
        K = self.n_components
        idx = np.random.choice(n, K, replace=False)
        self.means_ = X[idx].copy()
        self.covariances_ = np.array([np.cov(X.T) + 1e-6 * np.eye(d) for _ in range(K)])
        self.weights_ = np.ones(K) / K

    def fit(self, X):
        n, d = X.shape
        K = self.n_components
        self._init_params(X)
        self.log_likelihoods = []

        for it in range(self.max_iter):
            resp = np.zeros((n, K))
            for k in range(K):
                resp[:, k] = self.weights_[k] * multivariate_normal.pdf(X, mean=self.means_[k], cov=self.covariances_[k])
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp /= resp_sum + 1e-300

            Nk = resp.sum(axis=0)
            self.weights_ = Nk / n
            for k in range(K):
                self.means_[k] = (resp[:, k:k+1].T @ X).flatten() / Nk[k]
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k:k+1] * diff).T @ diff / Nk[k] + 1e-6 * np.eye(d)

            ll = np.sum(np.log(resp_sum.flatten() + 1e-300))
            self.log_likelihoods.append(ll)
            if it > 0 and abs(ll - self.log_likelihoods[-2]) < self.tol:
                break

        self.converged_ = it < self.max_iter - 1
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        resp = np.zeros((len(X), self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * multivariate_normal.pdf(X, mean=self.means_[k], cov=self.covariances_[k])
        return np.argmax(resp, axis=1)

gmm_s = GMM_EM(n_components=3)
gmm_s.fit(X)
print(f"Hand-written GMM converged: {gmm_s.converged_}, iters: {gmm_s.n_iter_}")
print(f"Means:\n{gmm_s.means_.round(2)}")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_gmm(X, labels, means, covariances, title='GMM Result'):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20, alpha=0.6)
    plt.scatter(means[:, 0], means[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidths=1.5)
    ax = plt.gca()
    for k in range(len(means)):
        eigenvalues, eigenvectors = np.linalg.eigh(covariances[k])
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=means[k], width=width, height=height, angle=angle, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(ellipse)
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_gmm(X, labels, gmm.means_, gmm.covariances_, 'GMM Clustering with Confidence Ellipses')
```

红色椭圆表示各高斯分量的 95% 置信区域，直观展示簇的形状和大小。

## 10. 模型评估

**选择最优 $K$**：

```python
bics = []
for k in range(1, 8):
    gm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gm.fit(X)
    bics.append(gm.bic(X))

best_k = np.argmin(bics) + 1
print(f"Best K by BIC: {best_k}")
```

- **BIC**：$-2\ell + p\log n$，惩罚更重，倾向于更简单的模型
- **AIC**：$-2\ell + 2p$，惩罚较轻
- **Silhouette Score**：也可用于评估聚类质量

## 11. 常见问题与易错点

- **协方差类型选择**：`covariance_type` 可选 `full`（完全）、`tied`（共享）、`diag`（对角）、`spherical`（球形），影响模型复杂度
- **正则化不足**：高维小样本时协方差矩阵容易奇异，需加大 `reg_covar`
- **分量数选择**：不要盲目增加 $K$，用 BIC 选择
- **与 K-Means 的关系**：GMM 更通用；当协方差为 $\sigma^2 I$ 且 $\sigma \to 0$ 时退化为 K-Means

## 12. 学习总结

高斯混合模型 + EM 算法是概率聚类的标准方法。它将"聚类"问题转化为"密度估计"问题，用概率而非硬标签描述不确定性。GMM-EM 是理解 EM 算法思想（E 步求期望、M 步最大化）的最佳实例。

## 13. 练习题与思考题（含答案）

**Q1**：GMM 中，当所有协方差矩阵约束为 $\sigma^2 I$ 且 $\sigma^2 \to 0$ 时，EM 的 E 步会退化成什么？

> 答：响应度 $\gamma_{ik}$ 会趋向 one-hot（只有最近分量的响应度趋近 1），即退化为 K-Means 的硬分配。

**Q2**：GMM 的 `covariance_type='diag'` 意味着什么？有什么优缺点？

> 答：协方差矩阵为对角阵，即各维度独立但方差不同。优点是参数少、计算快、不易过拟合；缺点是不能捕捉特征间的相关性。

**Q3**：如何用 GMM 做异常检测？

> 答：用 GMM 拟合正常数据的分布，对新样本计算 $P(x)$，若概率低于阈值则判定为异常。关键在于选择合适的阈值（如用正常数据的第 1 百分位作为阈值）。

## 14. 学习路径建议

- **前置知识**：EM 算法、多元高斯分布、贝叶斯基础
- **下一步学习**：变分贝叶斯 GMM（自动确定 $K$）、狄利克雷过程 GMM（非参数贝叶斯）
- **进阶方向**：VAE（变分自编码器）、Normalizing Flows、深度生成模型
