# EM 学习文档

## 1. 算法基础认知

EM（Expectation-Maximization）算法是一种用于含隐变量的概率模型参数估计的迭代算法，由 Dempster、Laird 和 Rubin 于 1977 年正式提出。EM 算法不是某一个具体模型，而是一个通用的优化框架，广泛应用于高斯混合模型、HMM、PLSA 等含有隐变量的模型中。

## 2. 核心原理

当模型中存在隐变量 $Z$ 时，直接最大化观测数据 $X$ 的对数似然 $\log P(X|\theta)$ 通常很困难。EM 算法巧妙地将这个困难的优化问题分解为两个可计算的步骤：

- **E 步（Expectation）**：固定参数 $\theta$，计算隐变量 $Z$ 的后验分布，并求完全数据对数似然的期望（Q 函数）
- **M 步（Maximization）**：固定隐变量分布，最大化 Q 函数来更新参数 $\theta$

## 3. 数学公式与推导

**观测数据的对数似然**：

$$\ell(\theta) = \log P(X|\theta) = \log \sum_Z P(X, Z|\theta)$$

由于对数内有求和，直接优化困难。

**引入 Q 函数**（E 步）：

$$Q(\theta | \theta^{(t)}) = \mathbb{E}_{Z|X, \theta^{(t)}} \left[ \log P(X, Z | \theta) \right]$$

**M 步**：

$$\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})$$

**收敛性保证（基于 Jensen 不等式）**：

由 Jensen 不等式可知：

$$\log P(X|\theta) \geq Q(\theta | \theta^{(t)}) + H(\theta^{(t)})$$

其中 $H$ 是与 $\theta$ 无关的熵项。因此每步 M 步都保证 $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$，对数似然单调递增，算法收敛到局部最优。

**ELBO（证据下界）**：

$$\log P(X|\theta) \geq \mathbb{E}_q[\log P(X,Z|\theta)] - \mathbb{E}_q[\log q(Z)] = \text{ELBO}(q, \theta)$$

## 4. 训练过程讲解

1. **初始化**：设定参数初值 $\theta^{(0)}$
2. **E 步**：用当前参数 $\theta^{(t)}$ 计算隐变量的后验分布 $P(Z|X, \theta^{(t)})$，进而计算 Q 函数
3. **M 步**：对 Q 函数关于 $\theta$ 求导令其为零（或直接求极值），得到 $\theta^{(t+1)}$
4. **判断收敛**：若 $|\ell(\theta^{(t+1)}) - \ell(\theta^{(t)})| < \epsilon$ 则停止，否则回到步骤 2

## 5. 应用场景

- 高斯混合模型（GMM）的参数估计
- 隐马尔可夫模型（HMM）的 Baum-Welch 算法
- PLSA 主题模型
- 缺失数据填补
- 广告系统中的用户兴趣聚类与画像推断

## 6. 优缺点分析

**优点**：
- 通用框架，适用于任何含隐变量的模型
- 每步保证对数似然单调递增
- 实现相对简单，E 步和 M 步通常有闭式解

**缺点**：
- 只能保证收敛到局部最优，对初始化敏感
- 收敛速度可能较慢（线性收敛率）
- M 步可能没有闭式解，需要数值优化

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

gmm = GaussianMixture(n_components=3, covariance_type='full', max_iter=100, random_state=42)
gmm.fit(X)

labels = gmm.predict(X)
print(f"Means:\n{gmm.means_}")
print(f"Weights: {gmm.weights_}")
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"Log-likelihood: {gmm.score(X):.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from scipy.stats import multivariate_normal

def em_gmm(X, K, max_iter=100, tol=1e-6):
    n, d = X.shape
    np.random.seed(42)
    mu = X[np.random.choice(n, K, replace=False)]
    cov = np.array([np.eye(d) for _ in range(K)])
    pi = np.ones(K) / K
    log_likelihoods = []

    for iteration in range(max_iter):
        resp = np.zeros((n, K))
        for k in range(K):
            resp[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=cov[k])
        resp /= resp.sum(axis=1, keepdims=True)

        Nk = resp.sum(axis=0)
        pi = Nk / n
        mu = (resp.T @ X) / Nk[:, None]
        for k in range(K):
            diff = X - mu[k]
            cov[k] = (resp[:, k:k+1] * diff).T @ diff / Nk[k]

        ll = sum(np.log(sum(pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=cov[k])
                          for k in range(K))))
        log_likelihoods.append(ll)
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return mu, cov, pi, resp, log_likelihoods

mu, cov, pi, resp, lls = em_gmm(X, K=3)
print(f"Converged in {len(lls)} iterations")
print(f"Means:\n{mu}")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(lls)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.title('EM Convergence')

plt.subplot(1, 2, 2)
labels = np.argmax(resp, axis=1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(mu[:, 0], mu[:, 1], c='red', marker='X', s=200)
plt.title('EM-GMM Clustering')
plt.tight_layout()
plt.show()
```

对数似然曲线应单调递增并趋于平稳，验证 EM 的收敛性质。

## 10. 模型评估

- **对数似然**：$\ell(\theta)$ 越大越好，反映模型对数据的拟合程度
- **BIC/AIC**：考虑模型复杂度的信息准则，用于选择隐变量数量 $K$
- **下游任务性能**：用 EM 学到的特征或聚类结果评估下游分类/聚类效果

## 11. 常见问题与易错点

- **初始值敏感**：不同的初始化可能收敛到不同的局部最优，建议多次运行取最优
- **奇异协方差**：某簇只有一个样本时协方差矩阵奇异，需加正则项
- **EM 不等于 K-Means**：K-Means 是硬分配，EM 是软分配（输出概率）
- **收敛不等于全局最优**：EM 只保证收敛到局部最优

## 12. 学习总结

EM 算法是含隐变量模型的通用参数估计框架。其核心思想——通过不断交替计算隐变量的期望和最大化参数——贯穿了 GMM、HMM、PLSA 等重要模型。理解 EM 的关键是掌握 Jensen 不等式推导的 ELBO 下界。

## 13. 练习题与思考题（含答案）

**Q1**：为什么说 EM 算法中的 E 步本质是在最大化 ELBO 关于 $q(Z)$ 的部分？

> 答：E 步将 $q(Z)$ 设为 $P(Z|X,\theta^{(t)})$，此时 KL 散度 $D_{KL}(q \| P(Z|X,\theta)) = 0$（当 $\theta = \theta^{(t)}$），ELBO 达到最大等于 $\log P(X|\theta^{(t)})$。

**Q2**：EM 算法的收敛速度与什么有关？

> 答：与"信息矩阵的条件数"有关。当隐变量的后验分布接近确定性时（信息量大），收敛快；当后验很不确定时，收敛慢。

**Q3**：K-Means 可以看作 EM 的特例吗？

> 答：可以。K-Means 等价于假设各簇协方差为单位矩阵 $\sigma^2 I$ 且 $\sigma^2 \to 0$ 时的 GMM-EM，此时软分配退化为硬分配。

## 14. 学习路径建议

- **前置知识**：极大似然估计、Jensen 不等式、条件概率
- **下一步学习**：高斯混合模型（GMM）、HMM 的 Baum-Welch 算法、变分 EM
- **进阶方向**：变分推断、随机 EM、在线 EM
