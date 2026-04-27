# EM 算法 学习文档

> 期望最大化：含隐变量概率模型参数估计的经典迭代方法。

> 来源线索：本节内容根据原书中关于"EM算法"的相关章节（第13章13.4.3节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** EM 算法通过交替执行"求期望"和"求极大值"两步，迭代估计含隐变量模型的参数。

**直觉类比：** 想象一个厨师要改良两种酱料的配方，但不知道每道菜用的是哪种酱料（隐变量）。厨师先猜一种配方，根据现有数据推断每道菜最可能用了哪种酱料（E步），然后根据这个推断更新配方（M步）。反复迭代，配方越来越精准。

**历史背景：** EM 算法由 Dempster、Laird 和 Rubin 于 1977 年正式提出，是含隐变量概率模型参数估计的标准方法。广泛应用于高斯混合模型（GMM）、隐马尔可夫模型（HMM）等。

**算法定位：** 参数估计/优化方法，用于含隐变量的概率模型的极大似然估计。

**前置知识：** 极大似然估计、Jensen 不等式、条件概率、高斯分布、Python/NumPy。

---

## 2. 核心原理

### 核心思想

当模型含有隐变量时，似然函数 $p(X|\theta) = \sum_Z p(X,Z|\theta)$ 中含有对隐变量的求和，无法直接求解析解。EM 算法将这个困难问题分解为两步交替迭代：

- **E 步（Expectation）**：固定参数 $\theta$，计算隐变量的后验分布 $p(Z|X,\theta)$，求完全数据对数似然的期望（Q 函数）
- **M 步（Maximization）**：固定隐变量分布，最大化 Q 函数，更新参数 $\theta$

### 工作流程

1. 初始化参数 $\theta^{(0)}$
2. **E 步**：计算 $Q(\theta, \theta^{(t)}) = E_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]$
3. **M 步**：$\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$
4. 重复 2-3 直到收敛

### 关键概念

- **隐变量（Latent Variable）**：不可直接观测的随机变量，如 GMM 中样本的类别归属
- **Q 函数**：完全数据对数似然在隐变量后验分布下的期望
- **Jensen 不等式**：证明 EM 算法单调递增似然下界的数学工具

---

## 3. 数学公式与推导

### 问题形式化

给定观测数据 $X = \{x_1, \ldots, x_n\}$，隐变量 $Z = \{z_1, \ldots, z_n\}$，参数 $\theta$，目标是最大化：

$$\mathcal{L}(\theta) = \log p(X|\theta) = \log \sum_Z p(X,Z|\theta)$$

由于对数内有对 $Z$ 的求和，无法直接优化。

### ELBO 推导

引入隐变量的任意分布 $q(Z)$：

$$\log p(X|\theta) = \log \sum_Z q(Z) \frac{p(X,Z|\theta)}{q(Z)}$$

由 Jensen 不等式（$\log$ 是凹函数）：

$$\log p(X|\theta) \geq \sum_Z q(Z) \log \frac{p(X,Z|\theta)}{q(Z)} = \text{ELBO}(q, \theta)$$

右边就是证据下界（ELBO）。当 $q(Z) = p(Z|X,\theta)$ 时取等号。

### Q 函数定义

E 步计算 Q 函数（取 $q(Z) = p(Z|X,\theta^{(t)})$）：

$$Q(\theta, \theta^{(t)}) = E_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)] = \sum_Z p(Z|X,\theta^{(t)}) \log p(X,Z|\theta)$$

### EM 算法保证单调递增

可以证明：$\mathcal{L}(\theta^{(t+1)}) \geq \mathcal{L}(\theta^{(t)})$。因为 M 步最大化了 ELBO，而 E 步使 ELBO 等于似然。

---

## 4. 训练过程讲解

### 以高斯混合模型（GMM）为例

**参数**：$\theta = \{\pi_k, \mu_k, \sigma_k^2\}_{k=1}^K$

**E 步**（计算每个样本属于每个高斯分量的后验概率）：

$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \sigma_j^2)}$$

**M 步**（用后验概率更新参数）：

$$\mu_k = \frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}}, \quad \sigma_k^2 = \frac{\sum_{i=1}^n \gamma_{ik}(x_i - \mu_k)^2}{\sum_{i=1}^n \gamma_{ik}}, \quad \pi_k = \frac{\sum_{i=1}^n \gamma_{ik}}{n}$$

### 超参数表

| 超参数 | 作用 | 建议 |
|--------|------|------|
| K | 聚类/分量数 | 用 BIC/AIC 选择 |
| max_iter | 最大迭代数 | 100-500 |
| tol | 收敛阈值 | 1e-4 |
| init_method | 初始化方式 | k-means++ |

---

## 5. 应用场景

1. **高斯混合模型（GMM）**：聚类分析、密度估计
2. **隐马尔可夫模型（HMM）**：语音识别、NLP
3. **缺失数据填补**：数据清洗
4. **图像分割**：将像素聚类为不同区域

---

## 6. 优缺点分析

### 优点
1. **理论保证**：似然函数单调递增，不会变差
2. **实现简单**：E 步和 M 步有明确的计算公式
3. **通用性强**：适用于任何含隐变量的概率模型

### 缺点
1. **局部最优**：依赖初始化，可能陷入局部最优。缓解：多次随机初始化取最优
2. **收敛速度**：比牛顿法慢，尤其是接近极值点时
3. **需要知道隐变量数量**：如 GMM 中的 K 值需预先设定

---

## 7. 调库实现

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成测试数据：3 个高斯簇
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# 使用 scikit-learn 的 GMM（内部使用 EM 算法）
gmm = GaussianMixture(n_components=3, max_iter=100, random_state=42)
gmm.fit(X)

# 预测每个样本的聚类标签
labels = gmm.predict(X)
probs = gmm.predict_proba(X)  # 每个样本属于各簇的概率（即 E 步的后验）

print(f"均值:\n{gmm.means_}")
print(f"协方差:\n{gmm.covariances_}")
print(f"混合权重: {gmm.weights_}")
print(f"对数似然: {gmm.score(X):.4f}")
```

---

## 8. 手工代码实现

```python
import numpy as np

class GMM_EM:
    """手工实现 EM 算法求解高斯混合模型"""

    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def _gaussian_pdf(self, X, mu, sigma):
        """计算多元高斯概率密度"""
        d = X.shape[1]
        diff = X - mu
        # sigma^{-1} 和行列式
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det)
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        return norm_const * np.exp(exponent)

    def fit(self, X):
        """EM 算法训练 GMM"""
        n, d = X.shape
        K = self.K

        # 初始化参数
        np.random.seed(42)
        self.means_ = X[np.random.choice(n, K, replace=False)]  # 随机选 K 个样本作为初始均值
        self.covariances_ = np.array([np.eye(d) for _ in range(K)])  # 初始协方差为单位矩阵
        self.weights_ = np.ones(K) / K  # 初始混合权重均匀

        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            # === E 步：计算后验概率 gamma_ik ===
            gamma = np.zeros((n, K))
            for k in range(K):
                gamma[:, k] = self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])
            # 归一化：gamma_ik = pi_k * N(x_i|mu_k,sigma_k) / sum_j(pi_j * N(x_i|mu_j,sigma_j))
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma = gamma / (gamma_sum + 1e-10)

            # === M 步：更新参数 ===
            N_k = gamma.sum(axis=0)  # 每个分量的有效样本数
            for k in range(K):
                # 更新均值：mu_k = sum_i(gamma_ik * x_i) / N_k
                self.means_[k] = (gamma[:, k:k+1] * X).sum(axis=0) / N_k[k]
                # 更新协方差：sigma_k = sum_i(gamma_ik * (x_i - mu_k)(x_i - mu_k)^T) / N_k
                diff = X - self.means_[k]
                self.covariances_[k] = (gamma[:, k:k+1] * diff).T @ diff / N_k[k]
            # 更新混合权重：pi_k = N_k / n
            self.weights_ = N_k / n

            # 计算对数似然
            log_likelihood = np.sum(np.log(gamma_sum + 1e-10))
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

        return self

    def predict(self, X):
        """预测聚类标签"""
        gamma = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            gamma[:, k] = self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])
        return np.argmax(gamma, axis=1)

# 测试
if __name__ == '__main__':
    np.random.seed(42)
    # 生成 3 个高斯簇
    X = np.vstack([
        np.random.randn(100, 2) + [2, 2],
        np.random.randn(100, 2) + [-2, 2],
        np.random.randn(100, 2) + [0, -2]
    ])
    gmm = GMM_EM(n_components=3)
    gmm.fit(X)
    labels = gmm.predict(X)
    print(f"学到的均值:\n{gmm.means_}")
    print(f"混合权重: {gmm.weights_}")
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_em_convergence():
    """可视化 EM 算法的收敛过程"""
    # 模拟对数似然的收敛曲线
    iterations = np.arange(1, 31)
    ll_values = -500 + 200 * (1 - np.exp(-iterations/8)) + np.random.randn(30) * 5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(iterations, ll_values, 'b-o', markersize=4)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Log-Likelihood', fontsize=11)
    ax1.set_title('EM 算法对数似然收敛曲线', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 可视化 GMM 聚类结果
    np.random.seed(42)
    X = np.vstack([np.random.randn(100,2)+[2,2], np.random.randn(100,2)+[-2,2], np.random.randn(100,2)+[0,-2]])
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3, random_state=42).fit(X)
    labels = gmm.predict(X)
    scatter = ax2.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', alpha=0.6, s=20)
    ax2.scatter(gmm.means_[:,0], gmm.means_[:,1], c='red', marker='x', s=200, linewidths=3)
    ax2.set_title('GMM 聚类结果', fontsize=12)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    plt.tight_layout()
    plt.savefig('em_convergence.png', dpi=100, bbox_inches='tight')
    plt.show()

visualize_em_convergence()
```

---

## 10. 模型评估

```python
from sklearn.metrics import silhouette_score, adjusted_rand_score

def evaluate_gmm(X, labels, y_true=None):
    """评估 GMM 聚类质量"""
    sil = silhouette_score(X, labels)
    print(f"轮廓系数: {sil:.4f}")  # 越接近 1 越好

    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
        print(f"调整兰德指数: {ari:.4f}")  # 越接近 1 越好
```

---

## 11. 常见问题与易错点

### 数据层面
1. **未标准化导致协方差矩阵奇异**
   - 现象：LinAlgError: singular matrix
   - 解决：对数据做标准化，或给协方差矩阵加正则化项

### 模型层面
1. **K 值选择不当**
   - 解决：用 BIC（贝叶斯信息准则）选择最优 K 值
2. **初始化不好导致局部最优**
   - 解决：使用 k-means 初始化，或多次随机初始化取最优

### 调参层面
1. **收敛判据太严格导致无限循环**
   - 解决：设 max_iter 上限 + 合理的 tol（如 1e-4）

---

## 12. 学习总结

EM 算法通过 Jensen 不等式将对数似然的直接优化转化为对 ELBO 的交替优化。E 步计算隐变量的后验分布，M 步更新模型参数。核心公式：$Q(\theta, \theta^{(t)}) = \sum_Z p(Z|X,\theta^{(t)}) \log p(X,Z|\theta)$。EM 保证似然单调递增，但可能陷入局部最优。

---

## 13. 练习题与思考题

**题1：** 为什么 EM 算法能保证似然函数单调递增？

**参考答案：** E 步取 $q(Z) = p(Z|X,\theta^{(t)})$ 使 ELBO 等于当前似然值。M 步最大化 ELBO，而 ELBO 是似然的下界，所以新参数对应的似然值至少不小于旧值。

**题2：** EM 算法与 K-Means 的关系是什么？

**参考答案：** K-Means 可看作 EM 算法的特例——硬分配版本的 GMM（后验概率退化为 one-hot，协方差矩阵趋于 0）。K-Means 的"分配"对应 E 步，"更新中心"对应 M 步。

**题3（开放）：** EM 算法能否用于深度学习？有什么变体？

**参考答案思路：** EM 可用于含隐变量的深度模型（如 VAE 的训练可视为一种变分 EM）。变分 EM 使用可学习的推断网络近似后验分布，适合高维隐变量。深度 EM 在语音识别（HMM-DNN）和半监督学习中有应用。

---

## 14. 学习路径建议

### 前置算法
- 极大似然估计（MLE）
- 贝叶斯定理、条件概率
- Jensen 不等式

### 平行算法
- K-Means（EM 的特例/简化版）
- 变分推断（EM 的推广）

### 进阶算法
- 变分 EM（使用神经网络近似后验）
- 变分自编码器（VAE）
- 隐马尔可夫模型（HMM）的 Baum-Welch 算法

### 推荐资源
1. **教材**：Bishop, "Pattern Recognition and Machine Learning" 第9章
2. **论文**：Dempster et al., "Maximum Likelihood from Incomplete Data via the EM Algorithm" (1977)
3. **博客**：Sean Borman 的 "The Expectation Maximization Algorithm" 教程
