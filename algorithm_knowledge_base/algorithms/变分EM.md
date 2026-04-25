# 变分EM 学习文档

> 使用变分推断近似求解隐变量模型的后验分布，是EM算法的变体

---

## 1. 算法基础认知

### 一句话定义
变分EM（Variational EM）是一种用于隐变量模型的参数估计方法，它使用变分推断来近似不可解的后验分布，然后最大化证据下界（ELBO）。

### 直觉类比
想象你要理解一个复杂的概率模型（如高斯混合模型），其中有一些隐藏变量（哪些样本属于哪个高斯分布）。变分EM就像先用一个简单的分布（变分分布）去近似复杂的真实后验，然后调整模型参数使得这个近似尽可能好。

### 历史背景
变分推断的历史可以追溯到1970年代，但变分EM作为EM的扩展，在1990年代随着机器学习复兴而流行。Attias在1999年的工作将变分方法系统化地应用于学习和推断。如今，变分推断是变分自编码器（VAE）等生成模型的基础。

### 算法定位
- 类型：无监督学习 → 参数估计、隐变量推断
- 输出：模型参数、隐变量的近似后验分布
- 模型类型：生成模型、概率图模型

### 前置知识
- EM算法：理解E步和M步
- 变分推断：理解变分分布、KL散度、证据下界（ELBO）
- 概率论：指数族分布、共轭先验
- 优化基础：梯度下降、坐标上升
- Python基础：NumPy、概率编程库（如PyMC3、PyTorch）

---

## 2. 核心原理

### 2.1 核心思想
变分EM的核心思想是：**将复杂的后验分布 $P(Z|X;\theta)$ 用简单的变分分布 $Q(Z)$ 来近似，通过最大化证据下界（ELBO）来同时优化变分参数和模型参数**。

与标准EM不同，变分EM的E步不是计算后验期望，而是找到最佳变分近似 $Q^*(Z)$，然后M步优化模型参数。

### 2.2 工作流程

1. **初始化**
   - 输入：观测数据 $X = (x_1, ..., x_n)$，隐变量 $Z = (z_1, ..., z_n)$
   - 初始化模型参数 $\theta^{(0)}$ 和变分参数 $\phi^{(0)}$
   - 设置收敛阈值 $\epsilon$ 和最大迭代次数 $T$

2. **迭代优化（对 $t=1$ 到 $T$）**
   - **E步（变分）**：固定 $\theta^{(t-1)}$，优化变分参数 $\phi$ 以最大化ELBO：
     $$Q^{(t)} = \arg\max_Q \mathcal{L}(\theta^{(t-1)}, Q)$$
     其中 $\mathcal{L}(\theta, Q) = \mathbb{E}_Q[\log P(X,Z;\theta)] - \mathbb{E}_Q[\log Q]$ 是ELBO。
   
   - **M步**：固定 $Q^{(t)}$，优化模型参数 $\theta$ 以最大化ELBO：
     $$\theta^{(t)} = \arg\max_\theta \mathcal{L}(\theta, Q^{(t)})$$
   
   - **检查收敛**：如果 $|\mathcal{L}^{(t)} - \mathcal{L}^{(t-1)}| < \epsilon$，则停止。

3. **输出结果**
   - 模型参数 $\theta^{(T)}$
   - 隐变量的近似后验分布 $Q^{(T)}(Z)$

### 2.3 关键概念解释

- **变分分布** $Q(Z)$：简单的概率分布族（如均值场近似：$Q(Z) = \prod_i Q_i(z_i)$）
- **证据下界（ELBO）**：$\mathcal{L}(\theta, Q) = \log P(X;\theta) - \text{KL}(Q || P(Z|X;\theta))$，是 $\log P(X;\theta)$ 的下界
- **均值场近似**：假设隐变量之间相互独立，将联合分布分解为边缘分布的乘积
- **坐标上升**：交替优化变分参数和模型参数，保证ELBO单调上升
- **KL散度**：$\text{KL}(Q || P)$ 衡量两个分布之间的差异，非负

### 2.4 几何/直观解释
从几何角度看，变分EM在分布空间中寻找一个简单分布 $Q$，使其与真实后验 $P(Z|X;\theta)$ 的KL散度最小。ELBO是 $\log P(X)$ 减去这个KL散度，所以最大化ELBO等价于最小化KL散度。

对于均值场近似，$Q(Z) = \prod_i Q_i(z_i)$，我们可以想象在分布空间中，通过坐标上升更新每个 $Q_i$，直到收敛到局部最优。

---

## 3. 数学公式与推导$

### 3.1 符号约定

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $X$ | 观测数据 | $n \times d$ 或序列 |
| $Z$ | 隐变量 | 随机变量向量 |
| $\theta$ | 模型参数 | 参数向量 |
| $Q(Z)$ | 变分分布（近似后验） | 概率分布 |
| $\phi$ | 变分参数（$\mathcal{Q}$ 的参数） | 参数向量 |
| $\mathcal{L}$ | 证据下界（ELBO） | 标量 |
| $\text{KL}$ | KL散度 | 标量，非负 |

### 3.2 问题形式化
给定观测数据 $X$，隐变量模型 $P(X,Z;\theta)$，我们希望：
1. **推断**：计算后验分布 $P(Z|X;\theta)$（通常不可解）
2. **学习**：估计模型参数 $\theta$ 使得 $\log P(X;\theta)$ 最大

变分方法将推断问题转化为优化问题：找到变分分布 $Q(Z)$ 最小化 $\text{KL}(Q || P(Z|X;\theta))$。

### 3.3 目标函数/损失函数
变分EM使用**证据下界（ELBO）**作为目标函数：

$$\mathcal{L}(\theta, Q) = \mathbb{E}_Q[\log P(X,Z;\theta)] - \mathbb{E}_Q[\log Q(Z)]$$

**为什么使用ELBO？**
1. **与对数似然的关系**：$\log P(X;\theta) = \mathcal{L}(\theta, Q) + \text{KL}(Q || P(Z|X;\theta))$，由于KL散度非负，ELBO是对数似然的下界
2. **优化简单**：ELBO是 $\theta$ 和 $\phi$ 的函数，可以通过坐标上升优化
3. **泛化性强**：适用于各种隐变量模型
4. **与变分自编码器（VAE）的联系**：VAE的变分下界就是ELBO

### 3.4 推导过程

**Step 1: 分解对数边际似然**

$$\log P(X;\theta) = \log \int P(X,Z;\theta) dZ$$

引入变分分布 $Q(Z)$：

$$\log P(X;\theta) = \log \int Q(Z) \frac{P(X,Z;\theta)}{Q(Z)} dZ$$

利用Jensen不等式（因为 $\log$ 是凹函数）：

$$\log P(X;\theta) \geq \int Q(Z) \log \frac{P(X,Z;\theta)}{Q(Z)} dZ = \mathcal{L}(\theta, Q)$$

这就是ELBO。

**Step 2: ELBO的另一种形式**

$$\mathcal{L}(\theta, Q) = \mathbb{E}_Q[\log P(X,Z;\theta)] - \mathbb{E}_Q[\log Q(Z)]$$

$$= \mathbb{E}_Q[\log P(X,Z;\theta)] + H(Q)$$

其中 $H(Q)$ 是变分分布的熵。

也可以写成：

$$\mathcal{L}(\theta, Q) = \log P(X;\theta) - \text{KL}(Q || P(Z|X;\theta))$$

因为：

$$\text{KL}(Q || P) = \mathbb{E}_Q[\log Q] - \mathbb{E}_Q[\log P(Z|X)]$$
$$= \mathbb{E}_Q[\log Q] - \mathbb{E}_Q[\log P(X,Z)] + \log P(X)$$

所以 $\log P(X) - \text{KL} = \mathcal{L}$。

**Step 3: 坐标上升更新**

对于均值场近似 $Q(Z) = \prod_i Q_i(z_i)$，可以证明最优的 $Q_j(z_j)$ 满足：

$$\log Q_j^*(z_j) = \mathbb{E}_{i \neq j}[\log P(X,Z;\theta)] + \text{const}$$

这意味着每个 $Q_j$ 正比于联合分布在其他变量上的期望的指数。

**Step 4: M步更新**

固定 $Q$，ELBO关于 $\theta$ 的最大化通常就是最大化期望对数联合概率：

$$\theta^{(t)} = \arg\max_\theta \mathbb{E}_Q[\log P(X,Z;\theta)]$$

对于指数族模型，这通常有解析解（类似标准EM的M步）。

### 3.5 最终解/算法步骤$

**变分EM算法（均值场近似）**：

```
输入：数据 X，模型 P(X,Z;θ)，变分族 Q（如均值场）
输出：参数 θ，变分参数 ϕ

1. 初始化 θ⁽⁰⁾, ϕ⁽⁰⁾，计算 ℒ⁽⁰⁾ = ℒ(θ⁽⁰⁾, Q⁽⁰⁾)
2. 对 t=1 到 T：
   a. E步（变分）：固定 θ⁽ᵗ⁻¹⁾，更新每个 Qⱼ：
      log Qⱼ⁽ᵗ⁾(zⱼ) = Eᵢ≠ⱼ [log P(X,Z;θ⁽ᵗ⁻¹⁾)] + const
      （对每个隐变量 j 更新，重复直到变分分布收敛）
   b. M步：固定 Q⁽ᵗ⁾，更新 θ：
      θ⁽ᵗ⁾ = arg max_θ E_Q⁽ᵗ⁾ [log P(X,Z;θ)]
   c. 计算 ℒ⁽ᵗ⁾ = ℒ(θ⁽ᵗ⁾, Q⁽ᵗ⁾)
   d. 如果 |ℒ⁽ᵗ⁾ - ℒ⁽ᵗ⁻¹⁾| < ε，则停止
3. 返回 θ⁽ᵀ⁾, Q⁽ᵀ⁾
```

---

## 4. 训练过程讲解$

### 4.1 数据预处理$

```python
import numpy as np
from scipy.special import digamma  # 用于狄利克雷分布

# ============================================
# 示例：高斯混合模型（GMM）的变分推断
# ============================================
# 生成模拟数据（来自两个高斯分布）
np.random.seed(42)
n_samples = 300
# 真实参数
true_mu1 = [0, 0]
true_mu2 = [3, 3]
true_sigma = [[1, 0], [0, 1]]

# 生成数据
X1 = np.random.multivariate_normal(true_mu1, true_sigma, size=n_samples//2)
X2 = np.random.multivariate_normal(true_mu2, true_sigma, size=n_samples//2)
X = np.vstack([X1, X2])

print(f"数据形状: {X.shape}")
print(f"前5个样本:\n{X[:5]}")

# 在变分推断中，数据需要是NumPy数组
# 不需要特别的预处理，但可能需要标准化（取决于模型）
```

预处理要点：
1. **数据格式**：根据隐变量模型，数据可能是向量、序列等
2. **标准化**：对于高斯混合等模型，标准化可能有助于收敛
3. **初始化**：需要初始化模型参数和变分参数
4. **变分族选择**：通常使用均值场近似（完全分解）

### 4.2 参数初始化$

```python
def initialize_gmm_variational(n_samples, n_components=2, n_features=2):
    """
    初始化高斯混合模型的变分参数
    
    返回:
        mu: 均值参数 (n_components, n_features)
        sigma: 协方差矩阵 (n_components, n_features, n_features) - 对角
        pi: 混合系数 (n_components,) - 狄利克雷共轭先验参数
        resp: 责任矩阵 (n_samples, n_components) - 变分分布 q(z_i)
    """
    # 1. 初始化混合系数（狄利克雷参数）
    alpha = np.ones(n_components)  # 共轭先验参数
    
    # 2. 初始化均值（从数据中随机选）
    idx = np.random.choice(n_samples, n_components, replace=False)
    mu = X[idx].copy()
    
    # 3. 初始化协方差（单位矩阵）
    sigma = np.array([np.eye(n_features) for _ in range(n_components)])
    
    # 4. 初始化责任（每个样本属于每个成分的后验概率近似）
    resp = np.random.dirichlet(np.ones(n_components), size=n_samples)
    
    return mu, sigma, pi, resp, alpha

# 初始化
mu, sigma, pi, resp, alpha = initialize_gmm_variational(X.shape[0], n_components=2, n_features=2)
print(f"初始化均值:\n{mu}")
print(f"责任矩阵形状: {resp.shape}")
```

初始化建议：
1. **K-Means初始化**：用K-Means初始化均值，通常更稳定
2. **随机初始化**：多次随机初始化，选择ELBO最大的结果
3. **先验参数**：狄利克雷先验通常用对称参数（如全1）
4. **责任矩阵**：可以初始化为均匀的或基于距离的软分配$

### 4.3 迭代过程（变分E步和M步）$

```python
def variational_e_step(X, mu, sigma, alpha, resp, n_iter_variational=10):
    """
    变分E步：更新责任矩阵resp（即q(z_i)）
    对于高斯混合，最优q(z_i)是多项分布，参数更新有解析形式
    """
    n_samples, n_features = X.shape
    n_components = len(mu)
    
    for iter_v in range(n_iter_variational):
        # 对于高斯混合，变分E步简化为计算新的责任
        # 计算对数似然每个样本属于每个成分
        log_prob = np.zeros((n_samples, n_components))
        
        for k in range(n_components):
            # 多元高斯对数概率密度（简化：假设对角协方差）
            diff = X - mu[k]
            # 简化：只考虑对角元素
            log_det = -0.5 * np.sum(np.log(np.diag(sigma[k])))
            exponent = -0.5 * np.sum((diff ** 2) / np.diag(sigma[k]), axis=1)
            log_prob[:, k] = np.log(alpha[k] + 1e-10) + log_det + exponent
        
        # 转换为责任（softmax）
        # 为了防止数值溢出，减去最大值
        log_prob_max = np.max(log_prob, axis=1, keepdims=True)
        prob = np.exp(log_prob - log_prob_max)
        resp_new = prob / np.sum(prob, axis=1, keepdims=True)
        
        # 检查收敛
        delta = np.mean(np.abs(resp_new - resp))
        resp = resp_new
        
        if delta < 1e-6:
            # print(f"变分E步收敛于第 {iter_v+1} 次迭代")
            break
    
    return resp

def variational_m_step(X, resp, mu, sigma, alpha):
    """
    变分M步：更新模型参数 mu, sigma, alpha
    对于高斯混合，有解析解
    """
    n_samples, n_features = X.shape
    n_components = len(mu)
    
    # 1. 更新混合系数（狄利克雷参数）
    N_k = np.sum(resp, axis=0)  # 每个成分的有效样本数
    alpha_new = alpha + N_k  # 狄利克雷的后验参数
    
    # 2. 更新均值
    for k in range(n_components):
        mu[k] = np.sum(resp[:, k:k+1] * X, axis=0) / (N_k[k] + 1e-10)
    
    # 3. 更新协方差（对角）
    for k in range(n_components):
        diff = X - mu[k]
        # 对角协方差：每个特征独立
        sigma[k] = np.diag(np.sum(resp[:, k:k+1] * (diff ** 2), axis=0) / (N_k[k] + 1e-10))
    
    return mu, sigma, alpha_new

def compute_elbo(X, resp, mu, sigma, alpha):
    """
    计算证据下界（ELBO）
    """
    n_samples, n_features = X.shape
    n_components = len(mu)
    elbo = 0.0
    
    for k in range(n_components):
        # E[log P(X|Z=k)]
        diff = X - mu[k]
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * np.diag(sigma[k]))) \
                  -0.5 * np.sum((diff ** 2) / np.diag(sigma[k]), axis=1)
        elbo += np.sum(resp[:, k] * log_prob)
        
        # E[log pi_k] - 使用狄利克雷的期望
        # 对于狄利克雷(alpha)，E[log pi_k] = digamma(alpha_k) - digamma(sum(alpha))
        if k < len(alpha):
            elbo += np.sum(resp[:, k]) * (digamma(alpha[k]) - digamma(np.sum(alpha)))
    
    # -E[log Q(Z)] = -sum_i sum_k resp_ik * log resp_ik
    elbo -= np.sum(resp * np.log(resp + 1e-10))
    
    return elbo

# 运行变分EM
print("\n" + "="*60)
print("变分EM训练过程（高斯混合模型）")
print("="*60)

n_iterations = 50
elbos = []

for t in range(n_iterations):
    # E步（变分）
    resp = variational_e_step(X, mu, sigma, alpha, resp, n_iter_variational=10)
    
    # M步
    mu, sigma, alpha = variational_m_step(X, resp, mu, sigma, alpha)
    
    # 计算ELBO
    elbo = compute_elbo(X, resp, mu, sigma, alpha)
    elbos.append(elbo)
    
    if (t+1) % 10 == 0:
        print(f"Iteration {t+1}/{n_iterations}, ELBO: {elbo:.4f}")

print(f"训练完成！最终ELBO: {elbos[-1]:.6f}")
```

### 4.4 收敛条件$

变分EM的收敛条件：
1. **ELBO变化很小**：$|\mathcal{L}^{(t)} - \mathcal{L}^{(t-1)}| < \epsilon$
2. **参数变化很小**：$\|\theta^{(t)} - \theta^{(t-1)} \| < \epsilon$
3. **变分参数收敛**：责任矩阵的变化很小
4. **达到最大迭代次数**

```python
def check_variational_convergence(elbos, tol=1e-6):
    """检查是否收敛"""
    if len(elbos) < 2:
        return False
    return abs(elbos[-1] - elbos[-2]) < tol
```

### 4.5 超参数及推荐范围$

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| `n_iterations` | 变分EM迭代次数 | 50 ~ 1000 | 100 |
| `n_iter_variational` | 每个E步内部变分推断迭代次数 | 10 ~ 100 | 10 |
| `tol` | 收敛阈值 | 1e-6 ~ 1e-4 | 1e-6 |
| `n_components` | 隐变量个数（如高斯混合的成分数） | 根据问题确定 | 必须指定 |
| `init` | 初始化方法 | 'kmeans' 或 'random' | 'kmeans' |

选择建议：
1. **成分数**：使用BIC、AIC或领域知识确定
2. **初始化**：使用K-Means通常更稳定
3. **收敛阈值**：太大会导致欠拟合，太小会过拟合（对初始化敏感）
4. **多次初始化**：变分EM可能陷入局部最优，多次运行选最好$

---

## 5. 应用场景$

### 5.1 典型应用$

**应用1：高斯混合模型（GMM）的密度估计**
- 场景：给定数据，估计其概率密度（可能多峰）
- 为什么适合：变分GMM可以快速近似后验，比吉布斯采样快
- 实现：使用变分推断估计GMM参数和样本责任$

**应用2：主题模型（LDA）的推断**
- 场景：从文档集合中提取主题（隐变量是主题分配）
- 为什么适合：LDA的精确推断是NP-hard，变分推断提供近似解
- 实现：使用变分EM推断每个文档的主题分布和主题的词分布$

**应用3：变分自编码器（VAE）**
- 场景：学习生成模型，隐空间是连续向量
- 为什么适合：VAE的目标函数就是变分下界（ELBO）
- 实现：使用神经网络参数化变分分布 $Q(z|x)$ 和生成模型 $P(x|z)$

### 5.2 适用数据特征$

1. **隐变量模型**：存在不可观测的隐变量，需要推断
2. **复杂后验**：精确推断不可解，需要近似
3. **大规模数据**：变分推断通常比MCMC方法快
4. **需要点估计**：变分推断给出近似后验，可用于后续任务$
5. **指数族模型**：通常假设模型是指数族，有共轭性质$

### 5.3 不适用场景$

1. **需要精确后验**：变分推断给出的只是近似 → 使用MCMC方法
2. **小数据且需要不确定性估计**：变分方法可能低估方差 → 使用吉布斯采样等
3. **非指数族模型**：变分更新可能没有解析形式 → 使用黑盒变分推断
4. **实时应用**：变分EM可能太慢 → 使用在线变分推断$
5. **模型选择**：变分EM需要指定隐变量个数 → 使用贝叶斯非参方法（如狄利克雷过程）

---

## 6. 优缺点分析$

### 6.1 优点$

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 近似推断 | 为复杂后验提供近似解 | 模型适当指定 |
| 速度较快 | 比MCMC方法通常快很多 | 变分族选择合理 |
| 易于实现 | 对于指数族模型，更新常是解析的 | 共轭先验 |
| 可扩展 | 可以处理大规模数据 | 使用随机变分推断 |
| 与深度学习结合 | VAE等现代生成模型的基础 | 使用神经网络参数化 |

### 6.2 缺点$

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 局部最优 | 变分下界优化是非凸的 | 多次初始化，选最好 |
| 近似误差 | 变分分布与真实后验有差距 | 使用更灵活的变分族 |
| 均值场假设 | 假设隐变量独立，可能不成立 | 使用结构化变分 |
| 需要指定隐变量个数 | 实际中往往不知道 | 使用贝叶斯非参方法 |
| 对初始化敏感 | 不同初始化可能得到不同结果 | 多次初始化，或使用K-Means初始化 |

---

## 7. 调库实现（Python + 完整代码 + 注释）$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ============================================
# 1. 基本使用：高斯混合模型的变分推断
# ============================================
print("="*60)
print("示例1：使用scikit-learn的GMM（基于变分EM）")
print("="*60)

# 生成数据（同前）
np.random.seed(42)
n_samples = 300
X1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=n_samples//2)
X2 = np.random.multivariate_normal([3,3], [[1,0],[0,1]], size=n_samples//2)
X = np.vstack([X1, X2])

print(f"数据形状: {X.shape}")

# 创建GMM模型（scikit-learn使用变分EM）
gmm = GaussianMixture(
    n_components=2,        # 成分数
    covariance_type='diag',  # 协方差类型：对角
    max_iter=100,          # 最大迭代次数
    tol=1e-3,             # 收敛阈值
    random_state=42
)

# 训练模型（变分EM）
gmm.fit(X)

# 预测责任（后验概率近似）
resp = gmm.predict_proba(X)  # 每个样本属于每个成分的概率

# 预测标签
labels = gmm.predict(X)

print(f"\n训练完成！")
print(f"权重: {gmm.weights_}")
print(f"均值:\n{gmm.means_}")
print(f"协方差（对角）:\n{gmm.covariances_}")
print(f"收敛: {gmm.converged_}")
print(f"迭代次数: {gmm.n_iter_}")

# 计算对数似然（近似）
log_likelihood = gmm.score(X)  # 返回平均对数似然
print(f"\n平均对数似然: {log_likelihood:.4f}")

# ============================================
# 2. 可视化聚类结果
# ============================================
def plot_gmm_results(X, labels, means, covariances, title="GMM聚类结果"):
    plt.figure(figsize=(10, 8))
    
    # 绘制样本点，颜色表示成分
    for k in range(2):
        idx = (labels == k)
        plt.scatter(X[idx, 0], X[idx, 1], 
                   c='red' if k==0 else 'blue', 
                   label=f'成分 {k}', s=50, alpha=0.7)
    
    # 绘制均值
    plt.scatter(means[:,0], means[:,1], c='black', marker='X', 
                s=200, label='均值')
    
    # 绘制协方差椭圆（简化：只画1倍标准差）
    from matplotlib.patches import Ellipse
    for k in range(2):
        # 对角协方差：轴长=标准差
        width = 2 * np.sqrt(covariances[k, 0])
        height = 2 * np.sqrt(covariances[k, 1])
        ell = Ellipse(means[k], width, height, alpha=0.3, 
                        color='red' if k==0 else 'blue')
        plt.gca().add_patch(ell)
    
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_gmm_results(X, labels, gmm.means_, gmm.covariances_, "GMM聚类结果（变分EM）")

# ============================================
# 3. 变分下界（ELBO）的可视化
# ============================================
print("\n" + "="*60)
print("示例2：ELBO随迭代的变化（概念性）")
print("="*60)

# 注意：scikit-learn的GMM不保存ELBO历史
# 这里我们用简单的模拟来展示ELBO的变化

def simulate_elbo_history():
    """模拟ELBO随迭代的变化（通常单调上升）"""
    n_iter = 50
    elbo = np.zeros(n_iter)
    base = -500  # 起始ELBO（假设值）
    for i in range(n_iter):
        # ELBO通常单调上升，趋于收敛
        elbo[i] = base + 100 * (1 - np.exp(-i/10)) + np.random.randn() * 2
    return elbo

elbo_sim = simulate_elbo_history()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(elbo_sim)+1), elbo_sim, 'b-', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('ELBO')
plt.title('变分EM：ELBO随迭代的变化（模拟）')
plt.grid(True, alpha=0.3)
plt.show()

print(f"模拟最终ELBO: {elbo_sim[-1]:.4f}")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）$

```python
import numpy as np
from scipy.special import digamma

class VariationalGMM:
    """
    手动实现的高斯混合模型变分EM
    简化版：假设对角协方差，使用均值场近似
    """
    
    def __init__(self, n_components=2, max_iter=100, tol=1e-6, random_state=None):
        """
        初始化变分GMM
        
        参数:
            n_components: 高斯成分数
            max_iter: 最大迭代次数
            tol: 收敛阈值
            random_state: 随机种子
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
        self.weights_ = None      # 混合系数
        self.means_ = None        # 均值
        self.covariances_ = None  # 协方差（对角）
        self.converged_ = False
        self.n_iter_ = 0
        self.elbos_ = []          # ELBO历史
        
    def _initialize(self, X):
        """初始化参数（使用K-Means）"""
        n_samples, n_features = X.shape
        
        # 使用K-Means初始化均值
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_components, random_state=42)
        labels = kmeans.fit_predict(X)
        self.means_ = kmeans.cluster_centers_.copy()
        
        # 初始化协方差（每个成分的对角协方差）
        self.covariances_ = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            X_k = X[labels == k]
            if len(X_k) > 1:
                self.covariances_[k] = np.var(X_k, axis=0)
            else:
                self.covariances_[k] = np.ones(n_features)
        
        # 初始化混合系数（均匀）
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # 初始化狄利克雷参数（先验+数据）
        self.alpha_ = np.ones(self.n_components) + np.bincount(labels, minlength=self.n_components)
        
    def _e_step(self, X, resp, n_iter_v=10):
        """变分E步：更新责任矩阵resp"""
        n_samples, n_features = X.shape
        
        for _ in range(n_iter_v):
            resp_old = resp.copy()
            
            # 计算对数似然每个样本属于每个成分
            log_prob = np.zeros((n_samples, self.n_components))
            
            for k in range(self.n_components):
                # 多元高斯（对角协方差）对数概率密度
                diff = X - self.means_[k]
                # 对数行列式
                log_det = -0.5 * np.sum(np.log(2 * np.pi * self.covariances_[k]))
                # 指数部分
                exponent = -0.5 * np.sum((diff ** 2) / self.covariances_[k], axis=1)
                log_prob[:, k] = np.log(self.weights_[k] + 1e-10) + log_det + exponent
            
            # Softmax转换为责任
            log_prob_max = np.max(log_prob, axis=1, keepdims=True)
            prob = np.exp(log_prob - log_prob_max)
            resp_new = prob / np.sum(prob, axis=1, keepdims=True)
            
            # 检查收敛
            if np.mean(np.abs(resp_new - resp)) < 1e-6:
                resp = resp_new
                break
            resp = resp_new
        
        return resp
    
    def _m_step(self, X, resp):
        """变分M步：更新模型参数"""
        n_samples, n_features = X.shape
        N_k = np.sum(resp, axis=0)  # 每个成分的有效样本数
        
        # 1. 更新混合系数（狄利克雷参数）
        self.alpha_ = np.ones(self.n_components) + N_k
        
        # 2. 更新权重（期望）
        self.weights_ = N_k / n_samples
        
        # 3. 更新均值
        for k in range(self.n_components):
            self.means_[k] = np.sum(resp[:, k:k+1] * X, axis=0) / (N_k[k] + 1e-10)
        
        # 4. 更新协方差（对角）
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.sum(resp[:, k:k+1] * (diff ** 2), axis=0) / (N_k[k] + 1e-10)
        
    def _compute_elbo(self, X, resp):
        """计算ELBO"""
        n_samples, n_features = X.shape
        elbo = 0.0
        
        for k in range(self.n_components):
            # E[log P(X|Z=k)]
            diff = X - self.means_[k]
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * self.covariances_[k])) \
                      -0.5 * np.sum((diff ** 2) / self.covariances_[k]), axis=1)
            elbo += np.sum(resp[:, k] * log_prob)
            
            # E[log pi_k] （狄利克雷期望）
            # 简化：使用alpha的期望
            if len(self.alpha_) > k:
                digamma_alpha = digamma(self.alpha_[k])
                digamma_sum = digamma(np.sum(self.alpha_))
                elbo += np.sum(resp[:, k]) * (digamma_alpha - digamma_sum)
        
        # -E[log Q(Z)]
        elbo -= np.sum(resp * np.log(resp + 1e-10))
        
        return elbo
    
    def fit(self, X):
        """训练变分GMM"""
        n_samples, n_features = X.shape
        
        # 初始化
        self._initialize(X)
        resp = np.random.dirichlet(np.ones(self.n_components), size=n_samples)
        
        print(f"开始训练变分GMM...")
        print(f"样本数: {n_samples}, 特征数: {n_features}")
        print(f"成分数: {self.n_components}")
        
        # 迭代优化
        for t in range(self.max_iter):
            # E步（变分）
            resp = self._e_step(X, resp, n_iter_v=10)
            
            # M步
            self._m_step(X, resp)
            
            # 计算ELBO
            elbo = self._compute_elbo(X, resp)
            self.elbos_.append(elbo)
            
            # 检查收敛
            if t > 0 and abs(elbo - self.elbos_[-2]) < self.tol:
                self.converged_ = True
                print(f"第 {t+1} 轮收敛！ELBO变化 < {self.tol}")
                break
            
            if (t+1) % 10 == 0:
                print(f"Iteration {t+1}/{self.max_iter}, ELBO: {elbo:.4f}")
        
        self.n_iter_ = t + 1
        print(f"训练完成！最终ELBO: {self.elbos_[-1]:.4f}")
        return self
    
    def predict(self, X):
        """预测样本成分"""
        # 计算责任，选择最大责任的成分
        resp = self._e_step(X, np.ones((X.shape[0], self.n_components)) / self.n_components
        return np.argmax(resp, axis=1)
    
    def predict_proba(self, X):
        """返回责任矩阵（后验概率近似）"""
        resp = self._e_step(X, np.ones((X.shape[0], self.n_components)) / self.n_components
        return resp

# ============================================
# 测试手写实现
# ============================================
if __name__ == "__main__":
    # 生成数据
    np.random.seed(42)
    n_samples = 300
    X1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=n_samples//2)
    X2 = np.random.multivariate_normal([3,3], [[1,0],[0,1]], size=n_samples//2)
    X = np.vstack([X1, X2])
    
    # 训练手写模型
    vgmm = VariationalGMM(n_components=2, max_iter=50, random_state=42)
    vgmm.fit(X)
    
    # 预测
    labels = vgmm.predict(X)
    
    # 评估
    print(f"\n手写变分GMM结果:")
    print(f"权重: {vgmm.weights_}")
    print(f"均值:\n{vgmm.means_}")
    print(f"是否收敛: {vgmm.converged_}")
    print(f"迭代次数: {vgmm.n_iter_}")
    
    # 与sklearn比较
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    
    print(f"\nsklearn GMM结果:")
    print(f"权重: {gmm.weights_}")
    print(f"均值:\n{gmm.means_}")
    print(f"是否收敛: {gmm.converged_}")
    print(f"迭代次数: {gmm.n_iter_}")
```

---

## 9. 可视化与结果理解$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def visualize_variational_gmm(X, labels, means, covariances, elbos=None, title="变分GMM结果"):
    """
    可视化变分GMM的结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 聚类结果
    ax = axes[0]
    for k in range(2):
        idx = (labels == k)
        ax.scatter(X[idx, 0], X[idx, 1], 
                  c='red' if k==0 else 'blue', 
                  label=f'成分 {k}', s=50, alpha=0.7)
    
    # 绘制均值和协方差椭圆
    from matplotlib.patches import Ellipse
    for k in range(2):
        ax.scatter(means[k, 0], means[k, 1], c='black', marker='X', 
                    s=200, label='均值' if k==0 else None)
        
        width = 2 * np.sqrt(covariances[k, 0])
        height = 2 * np.sqrt(covariances[k, 1])
        ell = Ellipse(means[k], width, height, alpha=0.3, 
                        color='red' if k==0 else 'blue')
        ax.add_patch(ell)
    
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.set_title(f'{title} - 聚类结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. ELBO变化（如果有）
    ax = axes[1]
    if elbos is not None:
        ax.plot(range(1, len(elbos)+1), elbos, 'b-', linewidth=2)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('ELBO')
        ax.set_title('ELBO随迭代的变化')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'ELBO历史不可用\n（sklearn不保存）', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('ELBO变化（模拟）')
    
    plt.tight_layout()
    plt.show()

# 运行可视化
print("="*60)
print("变分GMM可视化")
print("="*60)

# 训练sklearn的GMM
np.random.seed(42)
X1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=150)
X2 = np.random.multivariate_normal([3,3], [[1,0],[0,1]], size=150)
X = np.vstack([X1, X2])

gmm = GaussianMixture(n_components=2, max_iter=100, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# 模拟ELBO历史
elbo_sim = np.linspace(-500, -400, 100) + np.random.randn(100) * 5

visualize_variational_gmm(X, labels, gmm.means_, gmm.covariances_, elbo_sim, "变分GMM结果")

# 绘制责任矩阵热图
resp = gmm.predict_proba(X)
plt.figure(figsize=(10, 6))
plt.imshow(resp[:50], aspect='auto', cmap='Blues')  # 只显示前50个样本
plt.xlabel('成分')
plt.ylabel('样本索引')
plt.title('责任矩阵（前50个样本）')
plt.colorbar(label='责任（后验概率近似）')
plt.xticks([0,1], ['成分0', '成分1'])
plt.show()
```

**结果理解**：
1. **聚类结果图**：显示样本根据责任分配到的成分，椭圆表示每个成分的协方差
2. **ELBO变化图**：ELBO应该单调上升（除了数值误差），直到收敛
3. **责任矩阵热图**：每行表示样本属于每个成分的概率，应该每行和为1

---

## 10. 模型评估$

```python
import numpy as np
from sklearn.metrics import adjusted_rand_score

def evaluate_variational_gmm(model, X, labels_true=None):
    """
    评估变分GMM模型
    """
    print("="*60)
    print("变分GMM模型评估报告")
    print("="*60)
    
    # 1. 对数似然（模型拟合程度）
    log_likelihood = model.score(X)
    print(f"平均对数似然: {log_likelihood:.4f}")
    
    # 2. 预测成分
    labels_pred = model.predict(X)
    
    # 3. 如果有真实标签，计算调整兰德指数
    if labels_true is not None:
        ari = adjusted_rand_score(labels_true, labels_pred)
        print(f"\n调整兰德指数: {ari:.4f} (1.0表示完全一致）")
    
    # 4. 模型参数
    print(f"\n模型参数:")
    print(f"权重: {model.weights_}")
    print(f"均值:\n{model.means_}")
    print(f"协方差（对角）:\n{model.covariances_}")
    
    # 5. 收敛信息
    print(f"\n收敛状态: {model.converged_}")
    print(f"迭代次数: {model.n_iter_}")
    
    # 6. ELBO（如果可用）
    if hasattr(model, 'elbos_') and len(model.elbos_) > 0:
        print(f"最终ELBO: {model.elbos_[-1]:.4f}")
    
    return log_likelihood

# 评估示例
# evaluate_variational_gmm(gmm, X)
```

**变分EM的特殊评估点**：
1. **ELBO**：证据下界，越大表示模型对数据的拟合越好（但不一定防止过拟合）
2. **对数似然**：比较不同模型（如不同成分数）时使用
3. **责任矩阵**：检查每个样本的责任是否合理（应该明确分配或不确定）
4. **模型选择**：使用BIC、AIC选择成分数：
   $$BIC = -2 \cdot \log P(X;\hat{\theta}) + k \log n$$
   其中 $k$ 是参数个数
5. **与真实后验比较**：如果可能，比较变分近似与吉布斯采样的结果$

---

## 11. 常见问题与易错点$

### 11.1 模型不收敛，ELBO震荡或不上升
**原因**：
- 学习率（步长）太大（对于梯度上升）
- 初始化不好，陷入坏的局部最优
- 数据尺度差异大，导致数值不稳定

**解决方案**：
```python
# 1. 使用K-Means初始化均值
gmm = GaussianMixture(n_components=2, init_params='kmeans')

# 2. 标准化数据
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# 3. 增加迭代次数或调整收敛阈值
gmm = GaussianMixture(max_iter=1000, tol=1e-4)

# 4. 多次初始化，选择ELBO最大的结果
best_gmm = None
best_elbo = -np.inf
for i in range(10):
    gmm = GaussianMixture(n_components=2, random_state=i)
    gmm.fit(X)
    elbo = gmm.score(X)
    if elbo > best_elbo:
        best_elbo = elbo
        best_gmm = gmm
```

### 11.2 成分数选择困难，不知道该用多少
**问题**：变分EM需要指定成分数 $K$，但实际中往往未知。

**解决方案**：
```python
# 使用BIC选择成分数
import numpy as np

bics = []
ks = range(1, 10)

for k in ks:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    bic = gmm.bic(X)
    bics.append(bic)
    print(f"K={k}: BIC={bic:.4f}")

# 选择BIC最小的K
best_k = ks[np.argmin(bics)]
print(f"\n最佳成分数: {best_k}")

# 绘制BIC曲线
plt.figure(figsize=(10, 6))
plt.plot(ks, bics, 'b-', linewidth=2)
plt.xlabel('成分数 K')
plt.ylabel('BIC')
plt.title('使用BIC选择成分数')
plt.grid(True, alpha=0.3)
plt.show()
```

### 11.3 协方差矩阵奇异或接近奇异
**问题**：数据维度高或样本少，导致协方差矩阵不可逆。

**解决方案**：
```python
# 1. 使用对角协方差（假设特征独立）
gmm = GaussianMixture(covariance_type='diag')

# 2. 使用球面协方差（所有特征方差相同）
gmm = GaussianMixture(covariance_type='spherical')

# 3. 添加正则化（小常数到协方差对角）
# 注意：scikit-learn的GMM有reg_covar参数
gmm = GaussianMixture(reg_covar=1e-6)  # 添加到协方差对角

# 4. 减少特征数（降维）或使用PCA预处理
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
```

### 11.4 变分近似误差大，结果不可靠
**原因**：均值场假设太强（假设隐变量独立），变分分布与真实后验差距大。

**解决方案**：
1. **使用更灵活的变分族**：如结构化变分（考虑变量间依赖）
2. **使用MCMC方法**：如吉布斯采样，给出更准确后验但慢
3. **检查责任**：如果责任都在0.5附近，说明模型不确定，近似可能不好
4. **比较不同变分族**：尝试不同的变分分布族，看ELBO差异$

### 11.5 数值不稳定，出现log(0)或溢出
**原因**：概率值太小，对数计算溢出。

**解决方案**：
```python
# 1. 在计算对数概率时添加小常数
log_prob = np.log(prob + 1e-10)

# 2. 使用对数空间计算，避免中间概率值
# 在E步中，使用log-sum-exp技巧
def log_sum_exp(arr, axis=1):
    """稳定计算log(sum(exp(arr)))"""
    max_arr = np.max(arr, axis=axis, keepdims=True)
    return max_arr + np.log(np.sum(np.exp(arr - max_arr), axis=axis, keepdims=True))

# 3. 避免协方差行列式为零
covariance += 1e-6 * np.eye(n_features)
```

---

## 12. 学习总结$

### 核心要点回顾：
1. **变分推断**：用简单分布 $Q(Z)$ 近似复杂后验 $P(Z|X)$
2. **ELBO**：证据下界 $\mathcal{L}(\theta, Q) = \log P(X) - \text{KL}(Q || P)$
3. **坐标上升**：交替优化变分参数（E步）和模型参数（M步）
4. **均值场近似**：$Q(Z) = \prod_i Q_i(z_i)$，假设变量独立
5. **与VAE关系**：变分自编码器的目标函数就是ELBO$

### 从变分EM到其他推断方法：
```
变分EM (坐标上升，均值场)
    ↓
在线变分推断 - 处理数据流
    ↓
黑盒变分推断 (BBVI) - 使用梯度方法，更通用
    ↓
MCMC方法 (吉布斯采样) - 精确但慢
    ↓
变分自编码器 (VAE) - 深度学习+变分推断
```

### 实践建议：
1. **初始化**：使用K-Means，多次随机初始化选最好
2. **模型选择**：使用BIC、AIC选择成分数或隐变量维度
3. **检查收敛**：监控ELBO，确保单调上升
4. **变分族选择**：根据问题复杂度选择适当的变分近似
5. **与基准比较**：对于简单问题，与吉布斯采样结果比较$

---

## 13. 练习题与思考题（含答案）$

### 练习题$

**习题1：基础概念**
问题：解释ELBO的含义，为什么它是 $\log P(X)$ 的下界？

<details>
<summary>答案</summary>

ELBO是证据下界（Evidence Lower Bound），定义为：
$$\mathcal{L}(\theta, Q) = \mathbb{E}_Q[\log P(X,Z;\theta)] - \mathbb{E}_Q[\log Q(Z)]$$

它与对数边际似然的关系：
$$\log P(X;\theta) = \mathcal{L}(\theta, Q) + \text{KL}(Q || P(Z|X;\theta))$$

因为KL散度非负，所以 $\mathcal{L}(\theta, Q) \leq \log P(X;\theta)$。因此，最大化ELBO等价于最小化KL散度，使得变分分布 $Q$ 尽可能接近真实后验。

简单说，ELBO是下界，我们通过最大化它来间接最大化对数似然。
</details>

**习题2：编程实践**
问题：使用scikit-learn的GMM对手写数字进行聚类（使用降维后的数据）。

<details>
<summary>答案</summary>

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target

# 降维到2D（便于可视化）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 训练GMM（变分EM）
gmm = GaussianMixture(n_components=10, random_state=42)  # 10个数字
gmm.fit(X_pca)

# 预测
labels = gmm.predict(X_pca)

# 评估（与真实标签比较）
ari = adjusted_rand_score(y, labels)
print(f"调整兰德指数: {ari:.4f}")

# 可视化（前300个样本）
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:300, 0], X_pca[:300, 1], c=labels[:300], 
                         cmap='tab10', s=50, alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('GMM聚类结果（手写数字，PCA降维后）')
plt.colorbar(scatter, label='预测的digit')
plt.show()
```
</details>

**习题3：理论推导**
问题：证明对于高斯混合模型，变分E步的最优责任是：
$$r_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

<details>
<summary>答案</summary>

对于高斯混合，联合分布为：
$$P(X,Z;\theta) = \prod_{i=1}^n \prod_{k=1}^K \left( \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k) \right)^{z_{ik}}$$

变分分布为：$Q(Z) = \prod_i Q_i(z_i)$，其中 $Q_i$ 是多项分布，参数 $r_{ik}$。

ELBO为：
$$\mathcal{L} = \sum_{i=1}^n \sum_{k=1}^K r_{ik} \left( \log \pi_k + \log \mathcal{N}(x_i | \mu_k, \Sigma_k) \right) - \sum_{i=1}^n \sum_{k=1}^K r_{ik} \log r_{ik} + \text{const}$$

约束 $\sum_k r_{ik} = 1$。使用拉格朗日乘子，对 $r_{ik}$ 求导：

$$\frac{\partial}{\partial r_{ik}} \left[ \mathcal{L} + \lambda_i \left(1 - \sum_k r_{ik}) \right] = \log \pi_k + \log \mathcal{N}(x_i | \mu_k, \Sigma_k) - \log r_{ik} - 1 - \lambda_i = 0$$

解得：
$$r_{ik} \propto \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)$$

归一化后得到公式。
</details>

### 思考题$

**思考题1**：变分EM和标准EM有什么区别？

<details>
<summary>答案</summary>

| 方面 | 标准EM | 变分EM |
|------|----------|----------|
| E步 | 计算精确后验期望 $P(Z|X;\theta)$ | 用变分分布 $Q(Z)$ 近似后验 |
| M步 | 最大化期望对数联合概率 | 最大化ELBO（$\theta$ 和 $Q$ 的坐标上升） |
| 后验 | 精确计算（对于指数族） | 近似（均值场等） |
| 速度 | 通常较快（如果E步有解析解） | 可能较慢（需要迭代优化变分参数） |
| 通用性 | 仅限于共轭指数族 | 更通用，可结合变分推断 |

核心区别：标准EM的E步计算精确后验（如果可能），变分EM用近似后验。
</details>

**思考题2**：变分推断和MCMC方法（如吉布斯采样）各有什么优缺点？

<details>
<summary>答案</summary>

| 方面 | 变分推断 | MAVC方法 |
|------|----------|----------|
| 速度 | 通常较快 | 较慢（需要大量采样） |
| 准确性 | 近似，可能低估方差 | 渐近精确 |
| 收敛诊断 | ELBO单调上升，易检查 | 需要复杂诊断（如Rhat） |
| 适用模型 | 需要推导变分更新 | 更通用，任何模型均可采样 |
| 不确定性估计 | 可能不准（近似误差） | 准确（给出后验分布） |
| 大规模数据 | 可以扩展（随机变分） | 困难，计算量大 |

核心区别：变分推断给出近似但快速，MCMC给出精确但慢。
</details>

---

## 14. 学习路径建议$

### 初级阶段（掌握变分EM基础）
1. 理解EM算法和标准E步、M步
2. 掌握变分推断、ELBO、KL散度
3. 手动计算高斯混合的变分E步和M步
4. 使用scikit-learn实现GMM聚类

**学习时间**：2-3周**

### 中级阶段（理解原理和扩展）
1. 理解均值场近似和变分坐标上升
2. 学习变分推断与VAE的关系
3. 掌握模型选择准则（BIC、AIC）
4. 比较变分EM与吉布斯采样的优劣

**学习时间**：3-4周**

### 高级阶段（扩展到现代生成模型）
1. 学习变分自编码器（VAE）
2. 掌握黑盒变分推断（BBVI）
3. 理解变分推断在贝叶斯神经网络中的应用
4. 研究在线变分推断、随机变分推断

**学习时间**：4-6周**

### 实践项目建议
1. **基础项目**：高斯混合模型聚类（使用变分EM）
2. **进阶项目**：主题模型LDA的变分推断实现
3. **挑战项目**：变分自编码器（VAE）实现图像生成

### 推荐资源
- **书籍**：《机器学习》（周志华）第14章；《Pattern Recognition and Machine Learning》（Bishop）第10章
- **课程**：David Blei的变分推断课程；Hugo Larochelle的深度学习课程（VAE部分）
- **论文**：Attias (1999) Variational Bayesian Framework；Kingma & Welling (2013) VAE原始论文
- **代码**：Scikit-learn源码中的GMM实现；PyTorch实现VAE
- **实践**：Kaggle聚类竞赛；生成模型应用（如图像生成）
