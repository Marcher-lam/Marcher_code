# EM 学习文档

> EM (Expectation Maximization) 期望最大化算法是一种用于处理隐变量的迭代优化方法,广泛用于聚类、密度估计、参数学习等场景。

---

## 1. 算法基础认知

### 一句话定义
EM算法通过交替执行"E步"(求期望)和"M步"(最大化)两个步骤,迭代优化包含隐变量的概率模型参数。

### 直觉类比
**相亲配对问题**:
- 你知道每个人的偏好(模型参数),但不知道他们实际上更偏好谁(隐变量)
- **E步**:根据当前参数,猜测每个人最可能的配对
- **M步**:根据当前配对猜测,更新大家的偏好参数
- 重复直到收敛

### 历史背景
- 1977年,Dempster, Laird, Rubin在JRSS提出EM算法
- 解决"数据不完整"情况下的参数估计
- 成为统计学习基石算法

### 算法定位
- **类型**:参数估计/迭代优化
- **输出**:模型参数 $\theta$
- **模型类型**:概率生成模型

### 前置知识
- 概率论基础(条件概率、贝叶斯)
- 极大似然估计
- 凸优化基础

---

## 2. 核心原理

### 2.1 核心思想
EM算法核心是**处理缺失数据**:
- 观测数据 $X$ :我们能看到的数据
- 隐数据 $Z$ :缺失/隐藏的数据
- 完全数据: $(X, Z)$

**迭代**:
1. **E步**: 计算 $Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log L(\theta; X, Z)]$
2. **M步**: $\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$

### 2.2 工作流程
```
初始化 θ^(0)
    ↓
E步: Q(θ, θ^(t)) = E[log L(θ;X,Z)|X,θ^(t)]
    ↓
M步: θ^(t+1) = argmax Q(θ, θ^(t))
    ↓
收敛? → 是→结束,否→返回E步
```

### 2.3 关键概念
- **对数似然函数**: $\log L(\theta; X)$
- **Q函数**: 隐变量条件期望
- **Jensen不等式**: 保证收敛

### 2.4 几何直观
```
┌─────────────────────────────────────────────┐
│          EM迭代过程                         │
│                                             │
│   log L(θ)  ↑                               │
│            │      /\    /\                   │
│            │     /  \/  \                    │
│            │    /   /\   \                   │
│            │   /    /\    \                  │
│            │  /____/__\____\____            │
│              θ^0  θ^1  θ^2 θ*               │
│                           ↑                │
│                      收敛到最优              │
└─────────────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $X$ | 观测数据 |
| $Z$ | 隐数据 |
| $\theta$ | 模型参数 |
| $L(\theta;X)$ | 似然函数 |
| $Q(\theta,\theta^{(t)})$ | Q函数 |
| $p(z\|x,\theta)$ | 后验概率 |

### 3.2 问题形式化

**目标**: 最大化观测数据的对数似然
$$\hat{\theta} = \arg\max_\theta \log L(\theta; X)$$

**困难**: $\log L(\theta; X)$ 可能难以直接优化（隐变量求和/积分）

### 3.3 EM推导

**Step 1: 构造函数**
$$\log L(\theta; X) = \log \sum_Z P(X,Z|\theta)$$

**Step 2: Jensen不等式下界**
对任意分布 $q(Z)$:
$$\log L(\theta; X) = \log \sum_Z q(Z)\frac{P(X,Z|\theta)}{q(Z)}$$
$$\geq \sum_Z q(Z) \log \frac{P(X,Z|\theta)}{q(Z)}$$

**Step 3: 选择最优下界**
令 $q(Z) = P(Z|X,\theta^{(t)})$, 即取��验分布:
$$\log L(\theta; X) \geq B(\theta,\theta^{(t)})$$
其中 $B$ 是下界, 在 $\theta^{(t)}$ 处取等。

**Step 4: 迭代优化**
- **E步**: 计算 $Q(\theta,\theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log P(X,Z|\theta)]$
- **M步**: $\theta^{(t+1)} = \arg\max_\theta Q(\theta,\theta^{(t)})$

### 3.4 收敛性证明
$$\log L(\theta^{(t+1)}) \geq \log L(\theta^{(t)})$$

Jensen不等式保证单调收敛。

---

## 4. 训练过程

### 4.1 实现代码

```python
"""
EM 算法完整实现
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, List

class GaussianMixtureEM:
    """高斯混合模型的EM算法"""
    
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0
    
    def _initialize(self, X):
        """初始化参数"""
        n_samples, n_features = X.shape
        
        # 随机选择K个样本作为初始均值
        idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx]
        
        # 初始化权重为均匀
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # 初始化协方差为单位矩阵
        self.covariances_ = np.array([np.eye(n_features)] * self.n_components)
    
    def _compute_weights(self, X):
        """计算每个分量的权重"""
        n_samples = X.shape[0]
        weights = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            weights[:, k] = self.weights_[k] * self._pdf(X, self.means_[k], self.covariances_[k])
        
        return weights
    
    def _pdf(self, X, mean, cov):
        """高斯分布概率密度"""
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)
        coefficient = 1.0 / np.sqrt(np.linalg.det(cov) * (2 * np.pi) ** X.shape[1])
        return coefficient * np.exp(exponent)
    
    def fit(self, X):
        """拟合数据"""
        self._initialize(X)
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iter):
            # ===== E步: 计算后验概率 =====
            weights = self._compute_weights(X)
            responsibilities = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
            
            # ===== M步: 更新参数 =====
            Nk = responsibilities.sum(axis=0)  # 各分量负责的样本数
            
            # 更新权重
            self.weights_ = Nk / n_samples
            
            # 更新均值
            self.means_ = (responsibilities.T @ X) / (Nk[:, np.newaxis] + 1e-10)
            
            # 更新协方差
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff = responsibilities[:, k:k+1] * diff
                self.covariances_[k] = (weighted_diff.T @ diff) / (Nk[k] + 1e-10)
            
            # ===== 检查收敛 =====
            log_likelihood = np.log(weights.sum(axis=1) + 1e-10).sum()
            
            if iteration > 0 and abs(log_likelihood - self.log_likelihood_prev_) < self.tol:
                self.converged_ = True
                break
            
            self.log_likelihood_prev_ = log_likelihood
            self.n_iter_ = iteration + 1
        
        return self
    
    def predict_proba(self, X):
        """预测后验概率"""
        weights = self._compute_weights(X)
        return weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
    
    def predict(self, X):
        """预测类别"""
        return self.predict_proba(X).argmax(axis=1)
    
    def score(self, X):
        """对数似然"""
        weights = self._compute_weights(X)
        return np.log(weights.sum(axis=1) + 1e-10).sum()


class EMforGMM:
    """简化版GMM-EM实现"""
    
    def __init__(self, K, max_iter=100, tol=1e-6):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X):
        n, d = X.shape
        
        # 1. 初始化参数
        np.random.seed(42)
        self.mu = X[np.random.choice(n, self.K, False)]
        self.sigma = np.array([np.eye(d)] * self.K)
        self.pi = np.ones(self.K) / self.K
        
        for _ in range(self.max_iter):
            # 2. E步: 计算责任度
            responsibilities = self._e_step(X)
            
            # 3. M步: 更新参数
            Nk = responsibilities.sum(0)
            new_pi = Nk / n
            new_mu = (responsibilities.T @ X) / (Nk[:, np.newaxis] + 1e-10)
            
            # 更新协方差
            new_sigma = np.zeros_like(self.sigma)
            for k in range(self.K):
                diff = X - new_mu[k]
                new_sigma[k] = (responsibilities[:, k:k+1] * diff).T @ diff / (Nk[k] + 1e-10)
            
            # 4. 检查收敛
            ll = self._log_likelihood(X)
            self._update_params(new_pi, new_mu, new_sigma)
            
            if ll < self.ll_prev and self.ll_prev - ll < self.tol:
                break
            
            self.ll_prev = ll
        
        return self
    
    def _e_step(self, X):
        """E步"""
        log_resp = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            log_resp[:, k] = np.log(self.pi[k] + 1e-10) + self._log_gaussian(X, self.mu[k], self.sigma[k])
        
        log_resp_max = log_resp.max(1, keepdims=True)
        resp = np.exp(log_resp - log_resp_max)
        return resp / (resp.sum(1, keepdims=True) + 1e-10)
    
    def _log_gaussian(self, X, mu, sigma):
        """对数高斯密度"""
        diff = X - mu
        return -0.5 * (diff @ np.linalg.inv(sigma) * diff).sum(1) - 0.5 * (np.log(np.linalg.det(sigma)) + X.shape[1] * np.log(2 * np.pi))
    
    def _log_likelihood(self, X):
        """对数似然"""
        ll = 0
        for k in range(self.K):
            ll += self.pi[k] * np.exp(self._log_gaussian(X, self.mu[k], self.sigma[k]))
        return np.log(ll + 1e-10).sum()
    
    def _update_params(self, pi, mu, sigma):
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
```

---

## 5. 应用场景

### 5.1 典型应用
- **高斯混合模型聚类**:GMM-EM
- **隐马尔可夫模型**:HMM参数学习
- **缺失数据填补**
- **密度估计**

### 5.2 适用数据
- 有隐变量/缺失数据
- 生成模型参数学习
- 聚类

---

## 6. 优缺点

### 6.1 优点
| 优点 | 说明 |
|------|------|
| 收敛稳定 | 单调不增 |
| 普适性强 | 各种模型 |
| 简单实现 | 迭代框架 |

### 6.2 缺点
| 缺点 | 缓解 |
|------|------|
| 局部最优 | 多次初始化 |
| 收敛慢 | 加速技巧 |
| 需要E步可解析 | 变分近似 |

---

## 7. 调库实现

```python
"""
sklearn 实现
"""
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',
    max_iter=100,
    n_init=5
)
gmm.fit(X)

# 预测
labels = gmm.predict(X)
probs = gmm.predict_proba(X)
```

---

## 8. 手工实现

```python
"""
EM 核心简化版 - 二维GMM
"""

import numpy as np

class SimpleEM:
    """简化EM"""
    
    def __init__(self, K, n_init=3):
        self.K = K
        self.n_init = n_init
    
    def fit(self, X):
        best_ll = -np.inf
        best_params = None
        
        for _ in range(self.n_init):
            params = self._fit_once(X)
            ll = self._ll(X, params)
            if ll > best_ll:
                best_ll = ll
                best_params = params
        
        self.params_ = best_params
        return self
    
    def _fit_once(self, X):
        n, d = X.shape
        
        # 初始化
        mu = X[np.random.choice(n, self.K, False)]
        sigma = np.array([np.eye(d)] * self.K)
        pi = np.ones(self.K) / self.K
        
        for _ in range(100):
            # E步
            gamma = self._e_step(X, mu, sigma, pi)
            
            # M步
            Nk = gamma.sum(0)
            pi = Nk / n
            mu = (gamma.T @ X) / (Nk[:, np.newaxis] + 1e-10)
            
            for k in range(self.K):
                diff = X - mu[k]
                sigma[k] = (gamma[:, k:k+1] * diff).T @ diff / (Nk[k] + 1e-10)
        
        return {'mu': mu, 'sigma': sigma, 'pi': pi}
    
    def _e_step(self, X, mu, sigma, pi):
        log_gamma = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            log_gamma[:, k] = np.log(pi[k] + 1e-10) + self._log_pdf(X, mu[k], sigma[k])
        
        log_gamma_max = log_gamma.max(1, keepdims=True)
        gamma = np.exp(log_gamma - log_gamma_max)
        return gamma / (gamma.sum(1, keepdims=True) + 1e-10)
    
    def _log_pdf(self, X, mu, sigma):
        diff = X - mu
        return -0.5 * (diff @ np.linalg.inv(sigma) * diff).sum(1) - 0.5 * np.log(np.linalg.det(sigma))
    
    def _ll(self, X, params):
        ll = 0
        for k in range(self.K):
            ll += params['pi'][k] * np.exp(self._log_pdf(X, params['mu'][k], params['sigma'][k]))
        return np.log(ll + 1e-10).sum()
    
    def predict(self, X):
        gamma = self._e_step(X, self.params_['mu'], self.params_['sigma'], self.params_['pi'])
        return gamma.argmax(1)
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_gmm_clusters(X, labels, means, save_path='gmm.png'):
    """可视化聚类结果"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100)
    plt.savefig(save_path)
    plt.show()


def plot_convergence(log_likelihoods, save_path='convergence.png'):
    """收敛曲线"""
    plt.figure(figsize=(8, 4))
    plt.plot(log_likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('EM Convergence')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.show()
```

---

## 10. 评估

```python
from sklearn.metrics import silhouette_score

def evaluate_gmm(X, labels):
    """评估"""
    return {
        'silhouette': silhouette_score(X, labels),
        'log_likelihood': ll
    }
```

---

## 11. 常见问题

### 11.1 局部最优
- 多次随机初始化
- 选择最优结果

### 11.2 奇异性
- 协方差加小正则
- 限制协方差类型

---

## 12. 总结

### 核心要点
1. **E步**: 计算后验/责任度
2. **M步**: 更新参数
3. **单调收敛**: Jensen下界
4. **局部最优**: 多次初始化

### 算法链
```
EM → EM for GMM → 变分EM → VAE
```

---

## 13. 练习题

**习题1**: EM的单调性

<details>
<summary>答案</summary>

$$\log L(\theta^{(t+1)}) \geq \log L(\theta^{(t)})$$

由Jensen不等式保证。

</details>

**习题2**: 为什么叫"期望最���化"?

<details>
<summary>答案</summary>

E步: 计算关于隐变量的期望(Q函数)
M步: 最大化Q函数更新参数

</details>

---

## 14. 学习路径

- **初级**: 理解E-M步骤,运行GMM
- **中级**: 推导收敛性,处理问题
- **高级**: 变分EM,VAE

### 推荐资源
- **论文**: Dempster et al. "Maximum Likelihood from Incomplete Data" (1977)
- **书籍**: "Pattern Recognition and Machine Learning" - Bishop