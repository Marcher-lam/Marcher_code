# 变分EM 学习文档

## 1. 算法基础认知

变分EM（Variational EM）是将变分推断（Variational Inference）与 EM 框架结合的方法。当经典 EM 中 E 步的后验分布 $P(Z|X,\theta)$ 难以计算（不可解或计算代价过高）时，变分EM 用一个简单的近似分布 $q(Z)$ 来逼近真实后验，使 EM 框架仍然可用。它是现代贝叶斯机器学习的核心方法之一。

## 2. 核心原理

经典 EM 中 E 步需要计算 $\mathbb{E}_{Z|X,\theta}[\log P(X,Z|\theta)]$，但当隐变量维度高、后验分布复杂时，这个期望难以计算。

变分EM 的解决思路：**用一个可处理的分布族 $\mathcal{Q}$ 中的 $q(Z)$ 来近似真实后验 $P(Z|X,\theta)$**。

核心等式：

$$\log P(X|\theta) = \text{ELBO}(q, \theta) + D_{KL}(q(Z) \| P(Z|X,\theta))$$

由于 KL 散度非负，最大化 ELBO 等价于最小化 $q$ 与真实后验的 KL 散度。

## 3. 数学公式与推导

**ELBO 展开**：

$$\text{ELBO}(q, \theta) = \mathbb{E}_q[\log P(X,Z|\theta)] - \mathbb{E}_q[\log q(Z)]$$

$$= \mathbb{E}_q[\log P(X|Z,\theta)] + \mathbb{E}_q[\log P(Z|\theta)] - \mathbb{E}_q[\log q(Z)]$$

$$= \mathbb{E}_q[\log P(X|Z,\theta)] - D_{KL}(q(Z) \| P(Z|\theta))$$

**平均场近似（Mean-Field Approximation）**：

假设隐变量可分解为独立因子：

$$q(Z) = \prod_{j=1}^{M} q_j(Z_j)$$

在平均场假设下，最优 $q_j^*$ 满足：

$$\log q_j^*(Z_j) = \mathbb{E}_{q_{-j}}[\log P(X, Z|\theta)] + \text{const}$$

其中 $q_{-j}$ 表示除 $j$ 外所有其他隐变量的变分分布。

**变分EM 的两步迭代**：

- **变分 E 步**：固定 $\theta$，优化 $q$ 以最大化 ELBO（即最小化 $D_{KL}(q\|P(Z|X,\theta))$）
- **M 步**：固定 $q$，优化 $\theta$ 以最大化 ELBO

## 4. 训练过程讲解

1. **初始化**：设定变分参数和模型参数的初值
2. **变分 E 步**：
   - 在平均场假设下，对每个 $q_j$，计算 $\mathbb{E}_{q_{-j}}[\log P(X,Z|\theta)]$
   - 更新 $q_j$ 使其匹配该期望的形式
   - 坐标上升法交替更新各 $q_j$ 直到收敛
3. **M 步**：用变分分布下的期望替代真实后验期望，最大化 ELBO 更新 $\theta$
4. 重复 2-3 直到 ELBO 收敛

## 5. 应用场景

- LDA 主题模型（隐变量为文档-主题分配和主题-词分布）
- 深度生成模型（VAE 的理论基础）
- 贝叶斯神经网络的后验近似
- 大规模数据中的快速近似推断
- 广告推荐中的协同过滤隐因子模型

## 6. 优缺点分析

**优点**：
- 使不可解的后验推断变为可计算
- 收敛通常比 MCMC 快得多
- 可扩展到大规模数据（随机变分推断）
- 目标函数 ELBO 明确，便于监控收敛

**缺点**：
- 平均场假设忽略了变量间的相关性，近似精度有限
- 倾向于低估后验方差（过于集中）
- 不同变分族的选择影响近似质量
- 仍然只有局部最优保证

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "广告点击率预估 机器学习 深度学习",
    "推荐系统 协同过滤 用户画像",
    "CTR预估 特征工程 广告算法",
    "深度学习 神经网络 自然语言处理",
    "广告算法 推荐系统 点击率 用户画像",
    "自然语言处理 文本分类 深度学习"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

lda = LatentDirichletAllocation(n_components=2, max_iter=50, learning_method='batch', random_state=42)
lda.fit(X)

print("Topic-word distribution:")
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-5:]]
    print(f"  Topic {topic_idx}: {top_words}")
print(f"Perplexity: {lda.perplexity(X):.2f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from scipy.special import digamma, softmax

def variational_em_gmm(X, K, max_iter=100, tol=1e-6):
    n, d = X.shape
    np.random.seed(42)
    mu = X[np.random.choice(n, K, replace=False)]
    cov = np.array([np.eye(d) for _ in range(K)])
    pi_k = np.ones(K) / K

    for iteration in range(max_iter):
        phi = np.zeros((n, K))
        for k in range(K):
            diff = X - mu[k]
            phi[:, k] = np.log(pi_k[k] + 1e-10) - 0.5 * np.sum(diff @ np.linalg.inv(cov[k]) * diff, axis=1) - 0.5 * np.log(np.linalg.det(cov[k]) + 1e-10)
        phi = softmax(phi, axis=1)

        Nk = phi.sum(axis=0)
        pi_k = Nk / n
        for k in range(K):
            mu[k] = (phi[:, k:k+1].T @ X).flatten() / Nk[k]
            diff = X - mu[k]
            cov[k] = (phi[:, k:k+1] * diff).T @ diff / Nk[k] + 1e-6 * np.eye(d)

        if iteration > 0 and iteration % 10 == 0:
            elbo = _compute_elbo(X, phi, mu, cov, pi_k, K)
            print(f"Iter {iteration}, ELBO: {elbo:.4f}")

    return mu, cov, pi_k, phi

def _compute_elbo(X, phi, mu, cov, pi_k, K):
    n, d = X.shape
    elbo = 0
    for k in range(K):
        diff = X - mu[k]
        log_p_xz = -0.5 * np.sum(diff @ np.linalg.inv(cov[k]) * diff) - 0.5 * n * d * np.log(2*np.pi) - 0.5 * n * np.log(np.linalg.det(cov[k]) + 1e-10)
        elbo += np.sum(phi[:, k] * np.log(pi_k[k] + 1e-10)) + np.sum(phi[:, k] * log_p_xz / n)
    elbo -= np.sum(phi * np.log(phi + 1e-10))
    return elbo

mu_v, cov_v, pi_v, phi_v = variational_em_gmm(X[:100], K=3, max_iter=50)
print("Variational EM means:\n", mu_v)
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

elbo_history = []
for i in range(1, 50):
    mu_t, cov_t, pi_t, phi_t = variational_em_gmm(X[:100], K=3, max_iter=i)
    elbo_history.append(_compute_elbo(X[:100], phi_t, mu_t, cov_t, pi_t, 3))

plt.figure(figsize=(8, 4))
plt.plot(elbo_history)
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('Variational EM: ELBO Convergence')
plt.tight_layout()
plt.show()
```

ELBO 应单调递增并趋于平稳，验证变分 EM 的收敛性。

## 10. 模型评估

- **ELBO 值**：越大表示近似越好，是变分 EM 的直接优化目标
- **重构质量**：用学到的模型参数重构数据的误差
- **Perplexity**：语言模型/主题模型中常用的评估指标
- **与真实后验的 KL 散度**（若真实后验已知）

## 11. 常见问题与易错点

- **平均场假设过强**：隐变量间有强相关时，平均场近似质量差
- **ELBO 非凸**：变分 EM 仍可能收敛到局部最优
- **数值稳定性**：计算 ELBO 时注意 log(0) 问题，务必加 epsilon
- **与经典 EM 的区别**：经典 EM 在 E 步计算精确后验，变分 EM 计算近似后验

## 12. 学习总结

变分 EM 是经典 EM 的推广，通过引入变分近似，使不可解的后验推断变得可计算。它是 LDA、VAE 等重要模型的理论基础。理解变分 EM 的核心在于理解 ELBO 和平均场近似——前者是优化的目标函数，后者是实现可计算的近似手段。

## 13. 练习题与思考题（含答案）

**Q1**：变分 EM 与经典 EM 的根本区别是什么？

> 答：经典 EM 的 E 步计算精确后验 $P(Z|X,\theta)$；变分 EM 的 E 步用变分分布 $q(Z)$ 近似后验，适用于精确后验不可解的情况。

**Q2**：平均场近似的核心假设是什么？它有什么局限？

> 答：核心假设是隐变量相互独立：$q(Z) = \prod_j q_j(Z_j)$。局限在于忽略了变量间的后验相关性，导致近似精度有限，且倾向于低估后验方差。

**Q3**：为什么 ELBO 可以作为收敛的判据？

> 答：因为 $\log P(X|\theta) = \text{ELBO}(q,\theta) + D_{KL}(q\|P(Z|X,\theta))$，左边固定 $X$ 后在迭代中是只与 $\theta$ 相关的量，ELBO 的增加意味着要么 $\log P(X|\theta)$ 增加，要么 KL 散度减小，或者两者兼有。因此 ELBO 单调递增可以作为收敛判据。

## 14. 学习路径建议

- **前置知识**：EM 算法、KL 散度、Jensen 不等式
- **下一步学习**：高斯混合EM（具体实例）、LDA 主题模型、VAE
- **进阶方向**：随机变分推断（SVI）、归一化流（Normalizing Flows）、Stein 变分梯度下降
