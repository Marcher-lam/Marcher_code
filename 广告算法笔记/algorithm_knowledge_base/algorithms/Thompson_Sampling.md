# Thompson Sampling 学习文档

## 1. 算法基础认知

Thompson Sampling 是一种基于贝叶斯推断的 Bandit 算法，通过对 pCTR/pCVR 的后验分布进行采样来决策。数据少时方差大，自然倾向探索。

## 2. 核心原理

### 贝叶斯更新

对于 Beta-Bernoulli 模型：

$$
w_k \sim \text{Beta}(\alpha_k, \beta_k)
$$

- 每次获得正向反馈：α_k += 1
- 每次获得负向反馈：β_k += 1
- 选择采样值最大的臂

### 在广告冷启动中的应用

对 pCTR/pCVR 的后验分布进行采样，数据少时方差大，自然倾向探索。样本数增多后分布收敛，趋向利用。

## 3. 优缺点

- 实现简单
- 自然平衡探索与利用
- 适合在线学习
- 响应速度中等
- 稳定性中等

## 4. 代码实现

```python
import numpy as np

class ThompsonSamplingBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

## 5. 学习总结

Thompson Sampling 是广告系统中常用的探索策略，与 UCB 互为补充。在 Explore & Exploit 中，Thompson Sampling 和 ε-greedy 也是新广告冷启动的主流方案。
