
# EM 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
EM（Expectation-Maximization，期望最大化）算法是一种用于含隐变量概率模型参数估计的迭代优化方法，通过交替执行E步（计算隐变量的期望）和M步（最大化似然）来寻找最优参数。

### 1.2 直觉类比
想象你在黑暗中给n个人分组，但你只能看到每个人与其他人的身高差距，不知道具体的分组情况。EM算法的思想是：先随机猜测分组，然后根据身高差距不断调整分组，直到分组稳定。E步就像"根据当前猜测评估每个人的可能分组"，M步就像"根据评估结果更新分组规则"。

### 1.3 历史背景
EM算法由Dempster、Laird和Rubin于1977年正式提出，但类似的思想在此之前已经被多次发现。EM算法是统计学习中最重要的算法之一，广泛应用于缺失数据处理、隐马尔可夫模型、混合模型等领域。

### 1.4 算法定位
- 类型：优化算法（用于参数估计）
- 输出：模型参数的最优估计
- 模型类别：迭代优化算法

### 1.5 前置知识
- 概率论基础（条件概率、贝叶斯定理）
- 极大似然估计
- Python 编程（NumPy）

## 2. 核心原理
### 2.1 核心思想
EM算法的核心思想是"迭代优化，逐步逼近"——当直接最大化含隐变量的似然函数困难时，通过迭代地求解两个较简单的子问题（E步和M步）来逐步逼近最优解。

### 2.2 工作流程
1. 初始化模型参数
2. E步：根据当前参数，计算隐变量的后验分布
3. M步：根据E步计算的结果，最大化似然函数更新参数
4. 重复步骤2和3，直到收敛
5. 输出参数估计

### 2.3 关键概念解释
- **隐变量（Latent Variable）**：无法直接观测到的变量
- **E步（Expectation）**：计算隐变量的期望/后验概率
- **M步（Maximization）**：最大化Q函数更新参数
- **Q函数**：对数似然关于隐变量的期望

### 2.4 几何解释
从几何角度看，EM算法在参数空间中找到一条上升路径。每次迭代，E步确定上升方向（梯度方向），M步在该方向上找到最优点。这保证了似然函数单调递增。

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $X$ | 观测数据 |
| $Z$ | 隐变量 |
| $\theta$ | 模型参数 |
| $\theta^{(t)}$ | 第t次迭代的参数 |
| $Q(\theta, \theta^{(t)})$ | Q函数 |
| $L(\theta)$ | 对数似然函数 |

### 3.2 问题形式化
给定观测数据 $X$，假设存在隐变量 $Z$，参数 $\theta$ 的极大似然估计为：
$$\hat{\theta} = \arg\max_\theta \log P(X|\theta)$$

但由于 $Z$ 未知，直接优化困难。EM算法通过迭代优化：
$$\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$$

其中 Q 函数定义为：
$$Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log P(X,Z|\theta)]$$

### 3.3 目标函数
$$\max_\theta L(\theta) = \max_\theta \sum_{i}\log \sum_{z} P(x_i, z|\theta)$$

### 3.4 推导过程
**Step 1: 引入隐变量**
设 $Z$ 为隐变量，观测数据的似然为：
$$P(X|\theta) = \sum_Z P(X|Z,\theta)P(Z|\theta)$$

**Step 2: 定义Q函数**
$$Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log P(X,Z|\theta)]$$

**Step 3: E步**
计算后验概率：
$$P(Z|X, \theta^{(t)}) = \frac{P(X,Z|\theta^{(t)})}{P(X|\theta^{(t)})}$$

**Step 4: M步**
最大化Q函数更新参数：
$$\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$$

### 3.5 最终解/算法步骤
1. 初始化 $\theta^{(0)}$
2. 迭代直到收敛：
   - E步：计算 $P(Z|X, \theta^{(t)})$
   - M步：$\theta^{(t+1)} = \arg\max_\theta \sum_Z P(Z|X,\theta^{(t)}) \log P(X,Z|\theta)$
3. 返回 $\theta$

## 4. 训练过程讲解
### 4.1 数据预处理
- 检查数据完整性
- 处理缺失值
- 数据标准化（视情况）

### 4.2 参数初始化
- 随机初始化
- 基于数据的粗略估计
- 多次运行选择最优

### 4.3 迭代过程
```python
伪代码：
输入: 观测数据X, 模型P(X,Z|θ)
1. 初始化 θ^(0)
2. for t = 0 to T:
3.     # E步
4.     for each z:
5.         P(z|x, θ^(t)) = P(x,z|θ^(t)) / Σz' P(x,z'|θ^(t))
6.     # M步
7.     θ^(t+1) = argmax_θ Σ_x Σ_z P(z|x,θ^(t)) log P(x,z|θ)
8.     if |θ^(t+1) - θ^(t)| < ε:
9.         break
10. return θ^(t+1)
```

### 4.4 收敛条件
- 参数变化小于阈值
- 似然函数变化小于阈值
- 达到最大迭代次数

### 4.5 超参数及推荐范围
- max_iter: 100-500
- tol: 1e-4到1e-6
- 初始化方法选择

## 5. 应用场景
### 5.1 典型应用
- **高斯混合模型**：估计各成分的参数
- **隐马尔可夫模型**：学习状态转移概率
- **缺失数据填补**：估计缺失值
- **聚类分析**：软聚类方法

### 5.2 适用数据特征
- 存在隐变量或缺失数据
- 似然函数难以直接优化
- 数据量较大

### 5.3 不适用场景
- 似然函数非凸（可能陷入局部最优）
- 隐变量维度太高
- 计算资源有限

## 6. 优缺点分析
### 6.1 优点
- 理论上保证似然函数单调递增
- 适用于各种含隐变量的模型
- 实现相对简单
- 收敛稳定

### 6.2 缺点
- 可能收敛到局部最优
- 收敛速度可能较慢
- 需要计算后验分布
- 对初始值敏感

### 6.3 与同类算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| EM | 理论保证 | 局部最优 | 含隐变量模型 |
| 梯度下降 | 全局最优 | 需要梯度 | 可微分模型 |
| 变分推断 | 可扩展 | 近似解 | 大规模数据 |
| 蒙特卡洛 | 灵活 | 方差大 | 复杂模型 |

## 7. 调库实现
### 7.1 环境准备
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 7.2 完整代码示例
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 1. 生成示例数据（两个高斯混合）
np.random.seed(42)
n_samples = 300

# 生成两个高斯分布的数据
X1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//2)
X2 = np.random.multivariate_normal([5, 5], [[1, -0.3], [-0.3, 1]], n_samples//2)
X = np.vstack([X1, X2])

print(f"数据形状: {X.shape}")
print(f"数据均值: {X.mean(axis=0)}")
print(f"数据标准差: {X.std(axis=0)}")

# 2. 使用sklearn的GMM（内部使用EM算法）
gmm = GaussianMixture(
    n_components=2, 
    covariance_type='full',
    max_iter=100,
    random_state=42,
    n_init=5  # 多次初始化取最优
)
gmm.fit(X)

print(f"\n=== GMM结果 ===")
print(f"混合权重: {gmm.weights_}")
print(f"均值:\n{gmm.means_}")
print(f"协方差矩阵:\n{gmm.covariances_[0]}")
print(f"\n对数似然: {gmm.score(X):.4f}")

# 3. 可视化
plt.figure(figsize=(12, 5))

# 原始数据与拟合结果
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap='viridis', alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=100, 
            marker='x', label='中心')
plt.title('GMM聚类结果')
plt.legend()

# 概率等高线
plt.subplot(1, 2, 2)
x = np.linspace(-3, 8, 100)
y = np.linspace(-3, 8, 100)
X_grid, Y_grid = np.meshgrid(x, y)
positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
Z = np.exp(gmm.score_samples(positions.T)).reshape(X_grid.shape)
plt.contourf(X_grid, Y_grid, Z, levels=20, alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=100, marker='x')
plt.title('GMM概率密度')

plt.tight_layout()
plt.show()

# 4. 手动EM算法实现高斯混合
def em_gmm(X, n_components, max_iter=100, tol=1e-4):
    """手动实现EM算法估计高斯混合参数"""
    n_samples, n_features = X.shape
    
    # 初始化
    np.random.seed(42)
    weights = np.ones(n_components) / n_components
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = [np.eye(n_features) for _ in range(n_components)]
    
    for iteration in range(max_iter):
        # E步：计算每个样本属于各成分的后验概率
        responsibilities = np.zeros((n_samples, n_components))
        for k in range(n_components):
            diff = X - means[k]
            cov = covariances[k]
            # 高斯分布的概率密度（未归一化）
            exp_term = -0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)
            responsibilities[:, k] = weights[k] * np.exp(exp_term)
        
        # 归一化
        responsibilities /= (responsibilities.sum(axis=1, keepdims=True) + 1e-10)
        
        # M步：更新参数
        N_k = responsibilities.sum(axis=0)  # 各成分的有效样本数
        
        # 更新权重
        weights = N_k / n_samples
        
        # 更新均值
        new_means = np.zeros_like(means)
        for k in range(n_components):
            new_means[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / (N_k[k] + 1e-10)
        means = new_means
        
        # 更新协方差矩阵
        new_covariances = []
        for k in range(n_components):
            diff = X - means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            cov = (weighted_diff.T @ diff) / (N_k[k] + 1e-10)
            new_covariances.append(cov)
        covariances = new_covariances
        
        # 计算对数似然
        ll = 0
        for k in range(n_components):
            diff = X - means[k]
            cov = covariances[k]
            exp_term = -0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)
            ll += weights[k] * np.exp(exp_term).sum()
        
        if iteration > 0 and abs(ll - prev_ll) < tol:
            print(f"收敛于第{iteration}轮")
            break
        prev_ll = ll
    
    return weights, means, covariances, responsibilities

# 运行手动EM
weights_manual, means_manual, cov_manual, resp_manual = em_gmm(X, 2)

print("\n=== 手动EM结果 ===")
print(f"混合权重: {weights_manual}")
print(f"均值:\n{means_manual}")

# 5. 比较结果
print("\n=== 结果比较 ===")
print(f"sklearn权重: {gmm.weights_}")
print(f"手动权重: {weights_manual}")
print(f"sklearn均值:\n{gmm.means_}")
print(f"手动均值:\n{means_manual}")
```

### 7.3 运行结果示例
```
数据形状: (300, 2)
数据均值: [2.47 2.49]

=== GMM结果 ===
混合权重: [0.50 0.50]
均值:
[[-0.05 -0.03]
 [5.02 4.97]]
对数似然: -4.23

=== 手动EM结果 ===
混合权重: [0.50 0.50]
均值:
[[-0.05 -0.03]
 [5.02 4.97]]
```

## 8. 手工代码实现
### 8.1 核心算法手写
```python
import numpy as np

class EMManual:
    """手工实现EM算法框架"""
    
    def __init__(self, max_iter=100, tol=1e-4, random_state=42):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.params_ = None
        self.log_likelihoods_ = []
        
    def _e_step(self, X, params):
        """E步：计算隐变量的后验概率"""
        raise NotImplementedError
        
    def _m_step(self, X, responsibilities):
        """M步：根据后验概率更新参数"""
        raise NotImplementedError
        
    def _compute_log_likelihood(self, X, params):
        """计算对数似然"""
        raise NotImplementedError
        
    def fit(self, X):
        """运行EM算法"""
        np.random.seed(self.random_state)
        
        # 初始化参数
        params = self._initialize(X)
        
        for iteration in range(self.max_iter):
            # E步
            responsibilities = self._e_step(X, params)
            
            # M步
            params = self._m_step(X, responsibilities)
            
            # 记录似然
            ll = self._compute_log_likelihood(X, params)
            self.log_likelihoods_.append(ll)
            
            # 检查收敛
            if iteration > 0:
                if abs(ll - self.log_likelihoods_[-2]) < self.tol:
                    break
        
        self.params_ = params
        return self


class EMGaussianMixture(EMManual):
    """EM算法实现高斯混合模型"""
    
    def __init__(self, n_components=2, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        
    def _initialize(self, X):
        """初始化参数"""
        n_samples, n_features = X.shape
        
        # 随机选择k个样本作为均值
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        means = X[indices]
        
        # 初始化权重为均匀分布
        weights = np.ones(self.n_components) / self.n_components
        
        # 初始化协方差为单位矩阵
        covariances = [np.eye(n_features) for _ in range(self.n_components)]
        
        return {'weights': weights, 'means': means, 'covariances': covariances}
    
    def _e_step(self, X, params):
        """E步：计算后验概率"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            diff = X - params['means'][k]
            cov = params['covariances'][k]
            # 计算高斯分布的概率密度
            exp_term = -0.5 * np.sum(diff @ np.linalg.inv(cov + 1e-6*np.eye(cov.shape[0])) * diff, axis=1)
            responsibilities[:, k] = params['weights'][k] * np.exp(exp_term)
        
        # 归一化
        responsibilities /= (responsibilities.sum(axis=1, keepdims=True) + 1e-10)
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """M步：更新参数"""
        n_samples = X.shape[0]
        
        N_k = responsibilities.sum(axis=0)
        weights = N_k / n_samples
        
        means = np.zeros((self.n_components, X.shape[1]))
        covariances = []
        
        for k in range(self.n_components):
            # 更新均值
            means[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / (N_k[k] + 1e-10)
            
            # 更新协方差
            diff = X - means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            cov = (weighted_diff.T @ diff) / (N_k[k] + 1e-10)
            covariances.append(cov)
        
        return {'weights': weights, 'means': means, 'covariances': covariances}
    
    def _compute_log_likelihood(self, X, params):
        """计算对数似然"""
        n_samples = X.shape[0]
        ll = 0
        
        for i in range(n_samples):
            sample_ll = 0
            for k in range(self.n_components):
                diff = X[i] - params['means'][k]
                cov = params['covariances'][k]
                exp_term = -0.5 * diff @ np.linalg.inv(cov + 1e-6*np.eye(cov.shape[0])) @ diff
                sample_ll += params['weights'][k] * np.exp(exp_term)
            ll += np.log(sample_ll + 1e-10)
        
        return ll

# 测试
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    
    # 生成数据
    X, _ = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42)
    
    # 运行EM
    em = EMGaussianMixture(n_components=2, max_iter=100)
    em.fit(X)
    
    print("=== EM算法训练结果 ===")
    print(f"权重: {em.params_['weights']}")
    print(f"均值:\n{em.params_['means']}")
    print(f"最终对数似然: {em.log_likelihoods_[-1]:.4f}")
    
    # 可视化收敛曲线
    import matplotlib.pyplot as plt
    plt.plot(em.log_likelihoods_)
    plt.xlabel('迭代次数')
    plt.ylabel('对数似然')
    plt.title('EM算法收敛曲线')
    plt.show()
```

### 8.2 与调库结果对比
| 指标 | 手工实现 | sklearn |
|------|----------|---------|
| 权重 | 相近 | 相近 |
| 均值 | 相近 | 相近 |
| 迭代次数 | 更多 | 优化过 |

## 9. 可视化与结果理解
### 9.1 收敛曲线可视化
```python
import matplotlib.pyplot as plt
import numpy as np

# 记录EM迭代过程
log_likelihoods = []

# 模拟EM迭代
for i in range(50):
    # 模拟对数似然上升
    ll = -100 + 80 * (1 - np.exp(-i/10)) + np.random.normal(0, 0.5)
    log_likelihoods.append(ll)

plt.figure(figsize=(10, 4))
plt.plot(log_likelihoods)
plt.xlabel('迭代次数')
plt.ylabel('对数似然')
plt.title('EM算法收敛曲线')
plt.axhline(y=max(log_likelihoods), color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.show()
```

### 9.2 隐变量后验分布
```python
# 可视化E步的后验概率
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=responsibilities[:, 0], cmap='viridis')
plt.colorbar()
plt.title('样本属于成分1的后验概率')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=responsibilities[:, 1], cmap='viridis')
plt.colorbar()
plt.title('样本属于成分2的后验概率')

plt.tight_layout()
plt.show()
```

### 9.3 结果解读
- 收敛曲线应该单调递增
- 后验概率反映了样本的不确定性
- 硬聚类可取最大后验概率

## 10. 模型评估
### 10.1 评估指标选择
- **对数似然**：越高越好
- **BIC/AIC**：考虑模型复杂度
- **收敛速度**：迭代次数

### 10.2 BIC评估
```python
from sklearn.metrics import bic_score, aic_score

# 计算BIC和AIC
n_samples = X.shape[0]
n_params = 2 * 2 + 2 * 2 + 1  # 均值、协方差、权重

log_likelihood = gmm.score(X) * n_samples
bic = -2 * log_likelihood + n_params * np.log(n_samples)
aic = -2 * log_likelihood + 2 * n_params

print(f"BIC: {bic:.2f}")
print(f"AIC: {aic:.2f}")
```

### 10.3 模型选择
```python
# 测试不同成分数
from sklearn.mixture import GaussianMixture

for n in [1, 2, 3, 4]:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X)
    print(f"n={n}, BIC: {gmm.bic(X):.2f}, AIC: {gmm.aic(X):.2f}")
```

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
- 数据未标准化
- 存在异常值
- 样本量太小

### 11.2 模型层面常见错误
- 成分数选择不当
- 协方差矩阵奇异
- 陷入局部最优

### 11.3 调参层面常见误区
- 未进行多次初始化
- 收敛阈值设置不当
- 忽略数值稳定性

## 12. 学习总结
### 12.1 核心要点回顾
- EM算法用于含隐变量的概率模型参数估计
- E步计算隐变量的后验分布
- M步最大化Q函数更新参数
- 理论上保证似然函数单调递增

### 12.2 关键公式汇总
- Q函数：$Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log P(X,Z|\theta)]$
- E步：$P(Z|X,\theta^{(t)})$
- M步：$\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$

### 12.3 与前序/后续算法联系
- **前置算法**：极大似然估计
- **后续算法**：变分EM、蒙特卡洛EM

## 13. 练习题与思考题与思考题
### 13.1 基础练习题
1. 什么是EM算法？它主要用于什么问题？
2. 解释E步和M步的作用。
3. EM算法为什么能保证似然函数单调递增？

### 13.2 进阶思考题
1. EM算法和梯度下降有什么区别？
2. 什么情况下EM算法会失效？

### 13.3 详细答案与解析
1. **答案**：EM算法用于含隐变量的概率模型参数估计，通过迭代E步和M步来最大化似然函数。
2. **答案**：E步计算给定观测数据和当前参数下隐变量的后验分布；M步根据后验分布最大化期望似然来更新参数。
3. **答案**：由Jensen不等式可以证明，每次迭代的似然函数不会下降。

## 14. 学习路径建议建议
### 14.1 前置知识
- 概率论基础
- 极大似然估计
- 线性代数

### 14.2 平行算法
- 变分推断
- 蒙特卡洛方法
- 梯度下降

### 14.3 进阶算法
- 变分EM
- 在线EM
- 蒙特卡洛EM

### 14.4 推荐资源
- Dempster et al. (1977) "Maximum Likelihood from Incomplete Data via the EM Algorithm"
- 《Pattern Recognition and Machine Learning》- Bishop
- 《Machine Learning》- Tom M. Mitchell
