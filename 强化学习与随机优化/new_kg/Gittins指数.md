# Gittins指数 学习文档

> 为每条臂计算一个指数代表其未来价值，选择指数最大的臂。

> 来源线索：本节内容根据原书中关于"Gittins Index"的相关章节(Ch 7.6.4)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：Gittins指数将每条臂的"继续探索价值"量化为一个标量指数，选择指数最大的臂即为贝叶斯最优策略。

**公式**：对于臂$k$，Gittins指数$\nu_k$定义为：

$$\nu_k = \sup_\tau \frac{\mathbb{E}\left[\sum_{t=0}^{\tau-1}\gamma^t R_k^{t+1}\right]}{\mathbb{E}\left[\sum_{t=0}^{\tau-1}\gamma^t\right]}$$

其中$\tau$是最优停止时间，$\gamma$是折扣因子。

**关键性质**：
- 只适用于折扣无限horizon的赌博机问题
- 贝叶斯最优（在适当条件下）
- 计算复杂，实际中常用UCB或Thompson采样近似

**与Lagrangian松弛的关系**：Gittins指数可通过将臂间的耦合约束松弛得到。

## 4-8. 核心实现

```python
"""Gittins指数近似计算"""
import numpy as np
from scipy.stats import norm

def gittins_index_normal(mu, sigma, n, gamma=0.9):
    """正态分布信念下的Gittins指数近似"""
    # 简化近似：均值 + 探索奖金
    exploration_bonus = sigma * norm.ppf(gamma**n)
    return mu + exploration_bonus

if __name__ == "__main__":
    np.random.seed(42)
    K = 5
    means = np.random.normal(0, 1, K)
    sigma = 1.0
    n_pulls = np.ones(K)
    empirical_means = np.zeros(K)

    for step in range(200):
        indices = [gittins_index_normal(empirical_means[k], sigma/np.sqrt(n_pulls[k]),
                                          n_pulls[k]) for k in range(K)]
        k = np.argmax(indices)
        reward = np.random.normal(means[k], 0.5)
        empirical_means[k] = (empirical_means[k]*n_pulls[k] + reward)/(n_pulls[k]+1)
        n_pulls[k] += 1

    print(f"真实均值: {means.round(2)}")
    print(f"估计均值: {empirical_means.round(2)}")
    print(f"拉取次数: {n_pulls}")
```

## 9-14. 简要

### 12. 学习总结
Gittins指数：$\nu_k = \sup_\tau \frac{E[\sum\gamma^t R_k]}{E[\sum\gamma^t]}$。每臂一个标量，选最大。理论最优但计算困难，实际中多用UCB/Thompson替代。

### 13. 练习题
**Q1**：Gittins指数为什么只适用于折扣情形？
**A1**：无折扣（$\gamma=1$）时，未来无限收益不可比，指数无法定义。折扣保证未来奖励的"现值"有限且可比。

### 14. 学习路径
**前置**：多臂赌博机 | **进阶**：UCB、Thompson采样、Lagrangian松弛
**资源**：原书Ch 7.6.4、Gittins (1979)
