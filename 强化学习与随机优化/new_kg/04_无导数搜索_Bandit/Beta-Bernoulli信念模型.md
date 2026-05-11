# Beta-Bernoulli信念模型 学习文档

> 用Beta分布建模伯努利过程的成功概率，是贝叶斯赌博机的数学基础。

> 来源线索：本节内容根据原书中关于"Beta-Bernoulli Belief Model"的相关章节(Ch 7.4)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：Beta-Bernoulli模型用$\text{Beta}(\alpha, \beta)$作为伯努利参数$p$的先验，每次观测后更新$\alpha$或$\beta$，是Thompson采样的数学基础。

**模型**：
- 成功概率$p \sim \text{Beta}(\alpha, \beta)$
- 观测$Y \sim \text{Bernoulli}(p)$
- 成功：$\alpha \leftarrow \alpha + 1$，失败：$\beta \leftarrow \beta + 1$

**Beta分布性质**：

$$\mathbb{E}[p] = \frac{\alpha}{\alpha + \beta}, \quad \text{Var}[p] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

先验选择：$\text{Beta}(1,1)$=均匀分布（无信息先验）。

**为什么共轭**：Beta是Bernoulli的共轭先验——后验仍是Beta分布，形式简洁。

## 4-8. 核心实现

```python
"""Beta-Bernoulli信念模型"""
import numpy as np
from scipy.stats import beta as beta_dist

class BetaBernoulliBelief:
    def __init__(self, n_arms, alpha0=1.0, beta0=1.0):
        self.K = n_arms
        self.alpha = np.full(n_arms, alpha0)
        self.beta = np.full(n_arms, beta0)

    def posterior_mean(self):
        return self.alpha / (self.alpha + self.beta)

    def posterior_var(self):
        ab = self.alpha + self.beta
        return self.alpha * self.beta / (ab**2 * (ab + 1))

    def update(self, k, success):
        if success:
            self.alpha[k] += 1
        else:
            self.beta[k] += 1

    def sample(self):
        """从后验采样（用于Thompson采样）"""
        return np.random.beta(self.alpha, self.beta)

if __name__ == "__main__":
    np.random.seed(42)
    true_probs = [0.9, 0.6, 0.3]
    belief = BetaBernoulliBelief(3)
    for t in range(500):
        # Thompson采样
        samples = belief.sample()
        k = np.argmax(samples)
        result = float(np.random.random() < true_probs[k])
        belief.update(k, result)
    print(f"真实概率: {true_probs}")
    print(f"后验均值: {belief.posterior_mean().round(3)}")
    print(f"后验方差: {belief.posterior_var().round(4)}")
    print(f"α参数:   {belief.alpha.astype(int)}")
    print(f"β参数:   {belief.beta.astype(int)}")
```

## 9-14. 简要

### 12. 学习总结
Beta-Bernoulli：$p \sim \text{Beta}(\alpha,\beta)$，成功$\alpha$+1，失败$\beta$+1。Bernoulli的共轭先验，Thompson采样的核心。

### 13. 练习题
**Q1**：$\alpha$和$\beta$都很大时意味着什么？
**A1**：观测数据多，后验方差小，对$p$的估计很确定。$\alpha/(\alpha+\beta)$接近真实概率。

### 14. 学习路径
**前置**：贝叶斯更新、Thompson采样 | **进阶**：正态信念模型、知识梯度
**资源**：原书Ch 7.4
