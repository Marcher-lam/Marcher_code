# Thompson采样(Thompson Sampling) 学习文档

> 从后验分布中采样参数，按采样结果选择动作，是最优雅的贝叶斯探索策略。

> 来源线索：本节内容根据原书中关于"Thompson Sampling"的相关章节(Ch 7)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：Thompson采样从当前后验分布中为每臂采样一个参数，选择采样值最大的臂——不确定性越大，采样值波动越大，自然实现探索。

**算法**：
1. 为每臂$k$维护后验$p(\mu_k | H^n)$
2. 每步从后验采样$\hat{\mu}_k \sim p(\mu_k | H^n)$
3. 选择$k = \arg\max_k \hat{\mu}_k$
4. 观测奖励，用贝叶斯更新后验

**Beta-Bernoulli赌博机**（最常用）：
- 先验：$\mu_k \sim \text{Beta}(\alpha_k, \beta_k)$
- 观测成功：$\alpha_k \leftarrow \alpha_k + 1$
- 观测失败：$\beta_k \leftarrow \beta_k + 1$
- 采样：$\hat{\mu}_k \sim \text{Beta}(\alpha_k, \beta_k)$

**理论**：Thompson采样的遗憾与UCB同阶$O(\sqrt{Kn\ln n})$，且在贝叶斯遗憾意义下渐近最优。

## 4-8. 核心实现

```python
"""Thompson采样：Beta-Bernoulli赌博机"""
import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.K = n_arms
        self.alpha = np.ones(n_arms)  # Beta先验α=1
        self.beta = np.ones(n_arms)   # Beta先验β=1

    def select(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, k, reward):
        if reward > 0.5:
            self.alpha[k] += 1
        else:
            self.beta[k] += 1

if __name__ == "__main__":
    np.random.seed(42)
    K = 5
    true_means = [0.9, 0.7, 0.5, 0.3, 0.1]
    ts = ThompsonSampling(K)
    for t in range(2000):
        k = ts.select()
        r = float(np.random.random() < true_means[k])
        ts.update(k, r)
    print(f"Thompson采样 (2000步):")
    print(f"  后验均值: {(ts.alpha/(ts.alpha+ts.beta)).round(3)}")
    print(f"  拉取次数: {(ts.alpha+ts.beta-2).astype(int)}")
    print(f"  真实均值: {true_means}")
```

## 9-14. 简要

### 12. 学习总结
Thompson采样：从后验采样$\hat{\mu}_k \sim p(\mu_k|H^n)$，选$\arg\max\hat{\mu}_k$。贝叶斯最优的探索策略，实现简洁，效果优异。

### 13. 练习题
**Q1**：Thompson采样和UCB的核心区别？
**A1**：UCB是确定性策略（确定性上界），Thompson是随机策略（从后验采样）。TS天然表达参数不确定性，在贝叶斯框架下更自然。实践中TS通常表现更好且更易扩展到复杂模型。

### 14. 学习路径
**前置**：贝叶斯更新、多臂赌博机 | **进阶**：上下文Thompson采样
**资源**：原书Ch 7、Thompson (1933)、Russo et al. (2018)
