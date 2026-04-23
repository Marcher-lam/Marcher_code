# Bandit（多臂老虎机）学习文档

## 1. 算法基础认知

Bandit（多臂老虎机，Multi-Armed Bandit, MAB）是一类在线学习问题，核心是在"探索"（尝试新选项以获取信息）和"利用"（选择当前已知最优选项）之间取得平衡。

在广告系统中，Bandit 方法被广泛应用于：
- 冷启动探索策略
- 多目标权重动态调权
- 新广告/新素材效果探索
- 创意优选

## 2. 核心原理

将权重选择建模为多臂老虎机（MAB）问题：

$$
w_k \sim \text{Beta}(\alpha_k, \beta_k)
$$

综合 Reward 定义：

$$
\text{Reward} = \Delta\text{Rev} + \lambda_1 \cdot \Delta\text{UX} + \lambda_2 \cdot \Delta\text{Eco} + \lambda_3 \cdot \Delta\text{ROI}
$$

### Contextual Bandit

将冷启动视为上下文多臂老虎机问题，根据广告特征选择最优探索策略。相比传统 Bandit，引入了广告侧特征作为上下文。计算高效，适合在线部署。

字节跳动利用 Contextual Bandit 动态调整新广告的探索加权系数，根据实时反馈数据自适应调整探索力度。

## 3. 应用场景

- 广告冷启动探索策略
- 多目标权重动态调权（与 PID 配合）
- 新广告/新素材效果探索
- 广告创意动态优选
- eCPM 加权：boosted_eCPM = eCPM × (1 + boost_factor)，boost_factor 随数据积累逐渐衰减

## 4. 工业实践

- 工业最佳实践：大多数广告系统采用"PID为主 + Bandit辅助"的混合策略。PID 负责快速响应偏差，Bandit 负责探索更优的权重组合。
- 字节跳动：Contextual Bandit 动态调整探索加权系数

## 5. 常见 Bandit 算法

| 方法 | 适用场景 | 响应速度 | 复杂度 | 稳定性 |
|------|---------|---------|--------|--------|
| Thompson Sampling | 离散权重空间 | ★★中 | 低 | ★★中 |
| UCB | 离散权重空间 | ★★中 | 低 | ★★★高 |
| ε-greedy | 简单探索 | ★★★快 | 低 | ★★中 |

## 6. 代码实现

```python
import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = (1 - 1.0/n) * self.values[arm] + (1.0/n) * reward
```

## 7. 学习总结

Bandit 是广告系统中探索-利用权衡的核心方法，常与 PID 混合使用。主要变体包括 UCB、Thompson Sampling、ε-greedy 和 Contextual Bandit。
