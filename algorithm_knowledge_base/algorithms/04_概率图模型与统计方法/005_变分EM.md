# 变分EM 学习文档

> 用一句话说明这个算法的核心价值：变分EM（Variational EM）将EM算法与变分推断相结合，通过优化证据下界（ELBO）来处理含有隐变量的概率模型，兼具EM的迭代优化特性和变分的近似推断能力。

## 1. 算法基础认知

### 1.1 发展历史
变分EM是**期望最大化（EM）算法**与**变分推断（Variational Inference）**的交叉产物。EM算法由Arthur Dempster等人于1977年系统化提出，用于含有隐变量的参数估计；而变分推断的思想可追溯至1990年代中期（Jordan et al. 1999），将贝叶斯推断转化为优化问题。变分EM的融合使得在复杂隐变量模型中既保持EM的收敛性，又获得变分的灵活性，广泛应用于主题模型、混合模型、深度学习等领域。

**关键里程碑**：
- 1977：EM算法正式提出（Dempster, Laird, Rubin）
- 1999：变分推断系统化（Jordan, Becker, McLachlan）
- 2003：变分贝叶斯方法成熟应用于混合模型
- 2010s：变分自编码器（VAE）将变分EM推广到深度生成模型

**相关人物**：David Blei, Michael Jordan, Zoubin Ghahramani

### 1.2 类比理解
| 类比场景 | 对应变分EM逻辑 |
|---------|--------------|
| 猜谜游戏逐步逼近 | E步：根据当前猜测推断隐变量分布；M步：根据隐变量更新模型参数 |
| 模糊照片逐渐清晰 | 变分推断用简单分布近似复杂后验，EM迭代优化近似质量 |
| 团队合作分工 | E步成员汇报隐变量信息，M步负责人综合更新策略 |
| 导航寻路 | ELBO作为“指南针”，指引参数向最优解移动 |

### 1.3 算法定位

| 属性 | 取值 | 说明 |
|------|------|------|
| 模型类型 | 有模型（Model-based） | 假设数据生成过程已知（带隐变量） |
| 算法类别 | 近似推断 + 最大似然估计 | EM框架 + 变分近似 |
| 隐变量处理 | 近似推断 | 不精确计算后验，用变分分布替代 |
| 优化目标 | 证据下界（ELBO） | 最大化 $\mathbb{E}_{q(z)}[\log p(x,z)] - \mathbb{E}_{q(z)}[\log q(z)]$ |
| 输出 | 近似后验 $q(z)$、参数 $\theta$ | 概率模型参数与隐变量分布 |

### 1.4 前置知识清单

#### 数学基础
- [ ] 概率论基础：条件概率、贝叶斯定理、联合分布
- [ ] 期望与对数似然
- [ ] 拉格朗日乘数法（用于ELBO优化约束）
- [ ] KL散度及其性质：$D_{KL}(q\|p) = \mathbb{E}_q[\log q - \log p]$

#### 编程基础
- [ ] Python NumPy/SciPy 基础
- [ ] 数值优化（梯度上升/下降）
- [ ] 矩阵运算与分布采样

#### 强化学习前置
- [ ] 概率图模型（有向/无向图）
- [ ] 隐马尔可夫模型（HMM）
- [ ] 高斯混合模型（GMM）

### 1.5 相关算法对比

| 算法 | 核心思想 | 优点 | 缺点 |
|------|---------|------|------|
| 标准EM | 精确最大化似然 | 保证似然非降，收敛稳定 | 隐变量后验需精确计算，往往不可行 |
| 变分EM | 用变分分布近似后验 | 可处理复杂隐变量，灵活 | 近似误差，ELBO可能非紧 |
| 梯度下降 | 直接优化似然 | 通用性强 | 隐变量导致梯度不可导 |
| MCMC采样 | 马尔可夫链近似后验 | 无偏估计 | 收敛慢，计算成本高 |
| VAE | 变分自编码器（基于变分EM） | 端到端学习，生成模型 | 需要大量数据，训练不稳定 |

**关键区别**：标准EM要求 $p(x,z)$ 可积且能精确计算后验；变分EM放宽此要求，通过优化界来逼近后验。

> 来源线索：本节内容根据原书中关于"第14章 高斯混合EM"和"第15章 隐马尔可夫"的相关章节整理、扩展与教学化改写。

## 2. 核心原理

### 2.1 变分推断基础

**目标**：对后验分布 $p(z|x)$ 的推断，其中 $z$ 为隐变量，$x$ 为观测数据。精确计算通常不可行（尤其高维时）。

**变分近似**：引入一个简单分布族 $q(z)$，通过最小化 KL 散度来逼近后验：
$$
q^*(z) = \arg\min_{q\in Q} D_{KL}(q(z)\|p(z|x))
$$

利用贝叶斯公式 $p(z|x)=\frac{p(x,z)}{p(x)}$，展开 KL 散度：
$$
D_{KL}(q\|p(z|x)) = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(x,z)] + \log p(x)
$$

注意到 $\log p(x)$ 与 $q$ 无关，最小化 KL 等价于最大化 $\mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z)]$，即 **证据下界（ELBO）**：
$$
\mathcal{L}(q) = \mathbb{E}_{q(z)}[\log p(x,z)] - \mathbb{E}_{q(z)}[\log q(z)]
$$

最大化 ELBO 同时最小化 KL 散度，且 $\mathcal{L}(q) = \log p(x) - D_{KL}(q\|p(z|x))$，因此优化 ELBO 也间接优化了对数证据。

### 2.2 EM 与变分 EM 的融合

标准 EM 的 E 步计算完全后验 $p(z|x)$，M 步最大化完全数据的对数似然期望。变分 EM 将 E 步改为优化变分分布 $q(z)$ 以最大化 ELBO，M 步则基于 $q(z)$ 的期望更新参数。

**变分 EM 迭代**：
- **E 步**：固定参数 $\theta$，更新 $q(z)$ 以最大化 $\mathcal{L}(q,\theta)$
- **M 步**：固定变分分布 $q(z)$，更新参数 $\theta$ 以最大化 $\mathbb{E}_{q(z)}[\log p(x,z)]$

这实际上是将 EM 推广到变分推断框架，允许使用近似推断。

### 2.3 ELBO 的分解与优化

ELBO 可写为：
$$
\mathcal{L}(q,\theta) = \mathbb{E}_{q(z)}[\log p(x|z)] - D_{KL}(q(z)\|p(z)) + \mathbb{E}_{q(z)}[\log p(\theta)] - D_{KL}(q(z)\|p(\theta))
$$
（若采用先验 $p(\theta)$ 并将 $q$ 分解为 $q(z,\theta)=q(z)q(\theta)$）

## 3. 数学公式与推导

### 3.1 ELBO 的变分优化

最大化 ELBO 关于变分分布 $q$ 的优化问题可通过**变分自由能**（即 KL 散度 $D_{KL}(q\|p(z|x))$）理解。根据 Jensen 不等式：
$$
\log p(x) = \log \int p(x,z) dz = \log \int \frac{p(x,z)}{q(z)} q(z) dz \geq \int q(z) \log \frac{p(x,z)}{q(z)} dz = \mathcal{L}(q)
$$
其中等号成立当且仅当 $q(z) = p(z|x)$。

### 3.2 共轭指数族情形

当隐变量 $z$ 和变分分布 $q(z)$ 属于指数族时，ELBO 的最大化有解析解。例如，若 $z$ 是多项式分布，$q(z)$ 可更新为：
$$
q(z_k) \propto \exp\left(\mathbb{E}_{i\neq k}[\log p(x,z_i, z_k)]\right)
$$

### 3.3 平均场变分推断

假设 $q(z)$ 可分解为各隐变量子集的乘积：$q(z) = \prod_{i=1}^m q_i(z_i)$。对每个 $q_i$，最优解为：
$$
q_i^*(z_i) \propto \exp\left(\mathbb{E}_{j\neq i}[\log p(x, z)]\right)
$$
迭代更新直至收敛。

## 4. 训练过程讲解

变分EM的训练过程包括以下步骤：

1. **初始化**：给定参数初始值 $\theta^{(0)}$ 和变分分布初始 $q^{(0)}(z)$
2. **E步（变分推断）**：固定 $\theta^{(t)}$，通过优化算法（如梯度上升、坐标上升）最大化 $\mathcal{L}(q,\theta^{(t)})$ 得到 $q^{(t+1)}(z)$
3. **M步（参数更新）**：固定 $q^{(t+1)}(z)$，最大化 $\mathcal{L}(q^{(t+1)},\theta)$ 得到 $\theta^{(t+1)}$
4. **收敛检查**：若 $\mathcal{L}(q^{(t+1)},\theta^{(t+1)}) - \mathcal{L}(q^{(t)},\theta^{(t)}) < \epsilon$ 或达到最大迭代次数，停止；否则返回步骤2

**实用技巧**：
- 使用自然梯度上升更新变分参数
- 采用随机变分推断处理大规模数据
- 监控ELBO变化趋势判断收敛

## 5. 应用场景

- 主题建模（如LDA）
- 高斯混合模型参数估计
- 隐马尔可夫模型（HMM）
- 变分自编码器（VAE）
- 图像分割与聚类
- 自然语言处理中的词义消歧

## 6. 优缺点分析

**优点**：
- 计算效率高于MCMC方法
- 可扩展到大规模数据集
- 提供理论保证的下界
- 灵活性高，可结合深度学习

**缺点**
- 结果依赖于变分族的选择
- 可能陷入局部最优
- 对模型假设敏感
- ELBO最大化不保证完全逼近真实后验

## 7. 调库实现

import numpy as np
from scipy.special import digamma, gammaln

def variational_em(data, max_iter=100, tol=1e-4):
    """变分EM算法的简单实现示例（高斯混合模型）"""
    n_samples, n_features = data.shape
    k_components = 3
    
    # 初始化变分参数
    alpha = np.ones(k_components)  # 狄利克雷分布参数
    mu = data[np.random.choice(n_samples, k_components, replace=False)]  # 均值
    sigma = np.array([np.eye(n_features)] * k_components)  # 协方差
    
    elbo_history = []
    
    for iteration in range(max_iter):
        # E步：更新变分分布 q(z) 的参数（权重）
        log_likelihood = np.zeros((n_samples, k_components))
        for c in range(k_components):
            log_likelihood[:, c] = (
                -0.5 * n_features * np.log(2 * np.pi)
                - 0.5 * np.linalg.slogdet(sigma[c])[1]
                - 0.5 * ((data - mu[c]) @ np.linalg.inv(sigma[c]) * (data - mu[c])).sum(axis=1)
            )
            log_likelihood[:, c] += digamma(alpha[c]) - digamma(alpha.sum())
        
        log_weights = log_likelihood + digamma(alpha) - digamma(alpha.sum())
        log_sum = np.logaddexp.reduce(log_weights, axis=1, keepdims=True)
        w = np.exp(log_weights - log_sum)
        
        # M步：更新变分参数
        n_k = w.sum(axis=0)
        alpha = n_k + 1.0  # 假设对称狄利克雷先验
        
        for c in range(k_components):
            weighted_data = data.T @ w[:, c]
            mu[c] = weighted_data / n_k[c]
            centered = data - mu[c]
            sigma[c] = (w[:, c][:, None] * centered).T @ centered / n_k[c] + np.eye(n_features) * 1e-6
        
        # 计算ELBO
        elbo = (n_k * (digamma(alpha) - digamma(alpha.sum()))).sum()
        elbo += (w * (log_likelihood + log_sum)).sum()
        elbo -= (w * np.log(w)).sum()
        elbo -= 0.5 * ((sigma ** 2).sum() for sigma in sigma)
        elbo_history.append(elbo)
        
        if iteration > 0 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            break
    
    return mu, sigma, w, elbo_history

## 8. 手工代码实现

import numpy as np
from scipy.special import digamma

class SimpleVariationalEM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def _initialize(self, X):
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covs = np.array([np.eye(n_features)] * self.n_components)
        self.elbo_history = []
    
    def _e_step(self, X):
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            log_resp[:, k] = self._log_gaussian(X, self.means[k], self.covs[k])
            log_resp[:, k] += digamma(self.weights[k]) - digamma(self.weights.sum())
        
        # Log-sum-exp trick for numerical stability
        log_sum = np.logaddexp.reduce(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_sum)
        return resp, log_sum.flatten()
    
    def _log_gaussian(self, X, mean, cov):
        d = X.shape[1]
        centered = X - mean
        try:
            inv_cov = np.linalg.inv(cov)
            log_det = np.linalg.slogdet(cov)[1]
            return -0.5 * (d * np.log(2 * np.pi) + log_det + 
                          (centered @ inv_cov * centered).sum(axis=1))
        except np.linalg.LinAlgError:
            return -1e10 * np.ones(X.shape[0])
    
    def _m_step(self, X, resp):
        n_samples = X.shape[0]
        nk = resp.sum(axis=0)
        
        # Update weights
        self.weights = (nk + 1) / (n_samples + self.n_components)
        
        # Update means and covariances
        for k in range(self.n_components):
            if nk[k] > 0:
                self.means[k] = (resp[:, k][:, None] * X).sum(axis=0) / nk[k]
                centered = X - self.means[k]
                self.covs[k] = (resp[:, k][:, None] * centered).T @ centered / nk[k]
                self.covs[k] += np.eye(X.shape[1]) * 1e-6
    
    def _compute_elbo(self, X, resp, log_sum):
        elbo = 0
        
        # Entropy term
        elbo += (resp * log_sum).sum()
        
        # Expected complete log likelihood
        for k in range(self.n_components):
            elbo += (resp[:, k] * self._log_gaussian(X, self.means[k], self.covs[k])).sum()
            elbo += digamma(self.weights[k]) * resp[:, k].sum() - digamma(self.weights.sum()) * resp[:, k].sum()
        
        # KL divergence between q and p (assuming symmetric Dirichlet prior)
        elbo -= (resp * np.log(resp + 1e-10)).sum()
        
        return elbo
    
    def fit(self, X):
        """拟合变分EM模型"""
        self._initialize(X)
        
        for iteration in range(self.max_iter):
            # E步
            resp, log_sum = self._e_step(X)
            
            # M步
            self._m_step(X, resp)
            
            # 计算ELBO
            elbo = self._compute_elbo(X, resp, log_sum)
            self.elbo_history.append(elbo)
            
            # 检查收敛
            if iteration > 0 and abs(self.elbo_history[-1] - self.elbo_history[-2]) < self.tol:
                break
        
        return self
    
    def predict(self, X):
        """预测隐变量后验概率"""
        resp, _ = self._e_step(X)
        return resp

## 9. 可视化与结果理解

import matplotlib.pyplot as plt

# 可视化ELBO收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(elbo_history, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('Variational EM Convergence')
plt.grid(True, alpha=0.3)
plt.show()

# 可视化聚类结果
plt.figure(figsize=(10, 6))
for k in range(n_components):
    plt.scatter(X[resp[:, k] > 0.5, 0], X[resp[:, k] > 0.5, 1], 
                label=f'Component {k+1}', alpha=0.6)
plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='Means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Variational EM Clustering Results')
plt.legend()
plt.show()

## 10. 模型评估

评估变分EM性能的指标：
- ELBO收敛值
- 预测对数似然
- 聚类纯度（Purity）
- 调整兰德指数（Adjusted Rand Index）

## 11. 常见问题与易错点

- **ELBO不增**：可能由于学习率过高或数值不稳定
  - 解决方法：使用更小的学习率或自然梯度
- **数值下溢**：高维数据导致概率计算溢出
  - 解决方法：使用对数空间和log-sum-exp技巧
- **局部最优**：变分推断易陷入局部最优
  - 解决方法：多次随机初始化
- **变分族选择不当**：平均场假设可能过强
  - 解决方法：使用更复杂的变分族结构

## 12. 学习总结

变分EM通过结合EM算法和变分推断，为含有隐变量的复杂模型提供了一种可扩展的近似推断方法。它在保证计算效率的同时，能够处理传统EM难以应对的高维和复杂结构问题。

## 13. 练习题与思考题

1. 平均场变分推断的假设是什么？在什么情况下会失效？
2. 如何选择变分分布族（variational family）？
3. ELBO的下降意味着什么？如何解释这种现象？
4. 变分EM与标准EM的主要区别是什么？

## 14. 学习路径建议

建议按以下顺序学习：
1. 贝叶斯推断基础
2. 期望最大化（EM）算法
3. 变分推断基础
4. KL散度与ELBO
5. 平均场变分推断
6. 变分EM算法
7. 收敛性与优化技巧
8. 在生成模型中的应用（如VAE）
9. 高级变分推断技术（如自然梯度）