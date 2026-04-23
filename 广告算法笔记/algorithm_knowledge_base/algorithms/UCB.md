# UCB（Upper Confidence Bound）学习文档

## 1. 算法基础认知

UCB（置信上界）是一种经典的 Bandit 算法，通过为每个臂计算置信上界来平衡探索与利用。数据越少，探索奖励越大。

## 2. 核心原理

### UCB 公式

$$
a^* = \arg\max_a \left[ \hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a(t)}} \right]
$$

其中：
- $\hat{\mu}_a$：臂 a 的平均奖励
- $c$：探索参数
- $t$：总尝试次数
- $N_a(t)$：臂 a 被选择的次数

### 在广告冷启动中的应用

$$
\text{score} = \text{mean\_reward} + q \times \sqrt{\frac{\ln N}{n}}
$$

数据越少，探索奖励越大，从而鼓励系统对新广告进行探索。

## 3. 优缺点

- 确定性的探索-利用平衡
- 响应速度中等
- 复杂度低
- 稳定性高

## 4. 代码实现

```python
import numpy as np

class UCBBandit:
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.total_count) / self.counts
        )
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_count += 1
        n = self.counts[arm]
        self.values[arm] = (1 - 1.0/n) * self.values[arm] + (1.0/n) * reward
```

## 5. 学习总结

UCB 提供确定性的探索-利用平衡，是广告冷启动中最常用的探索策略之一。几乎所有大厂都在使用。
