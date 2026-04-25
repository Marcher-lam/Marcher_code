# UCB（Upper Confidence Bound）学习文档

## 1. 算法基础认知

UCB（置信上界）是一种经典的 Bandit 算法，通过为每个臂的期望奖励添加不确定性bonus来平衡探索与利用。核心思想：选择奖励估计值加上置信区间上界最大的臂。在广告冷启动中，UCB 是最常用的确定性探索策略。

## 2. 核心原理

UCB 基于 optimism in face of uncertainty 原则。对于每个臂，我们维护其平均奖励估计 $\hat{\mu}_a$，并加上一个随尝试次数递减的不确定性项。数据越少的臂不确定性越大，越容易被选中探索。随着数据积累，不确定性减小，决策趋向利用。

## 3. 数学公式与推导

**UCB1 公式**：

$$
a^* = \arg\max_a \left[ \hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a(t)}} \right]
$$

其中：
- $\hat{\mu}_a = \frac{1}{N_a(t)}\sum_{i=1}^{N_a(t)} r_i$：臂 $a$ 的经验平均奖励
- $c$：探索参数（通常 $c=\sqrt{2}$）
- $t$：总尝试次数
- $N_a(t)$：臂 $a$ 被选择的次数

**不确定性项推导**：由 Hoeffding 不等式，$\hat{\mu}_a$ 与真实均值 $\mu_a$ 的偏差以概率至少 $1 - 2t^{-2c^2}$ 满足：

$$
\mu_a \leq \hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a(t)}}
$$

**广告冷启动中的形式**：

$$
\text{score} = \text{mean\_reward} + q \times \sqrt{\frac{\ln N}{n}}
$$

## 4. 训练过程讲解

1. 初始化：每个臂尝试一次（或设置初始计数为 0）
2. 选择阶段：计算每个臂的 UCB 分数，选最大的
3. 执行选择的臂，观察奖励 $r$
4. 更新该臂的平均奖励和选择计数
5. 重复步骤 2-4
6. 随着交互轮次增加，不确定性项递减，选择趋向最优臂

## 5. 应用场景

- 广告冷启动探索（新广告/新创意的流量分配）
- 推荐冷启动（新 item 的曝光策略）
- A/B 测试加速（多方案快速比较）
- 在线学习与实时决策

## 6. 优缺点分析

**优点：**
- 确定性的探索-利用平衡（相同输入相同输出）
- 理论保证：UCB1 的累计遗憾上界为 $O(\sqrt{KT\ln T})$
- 实现简单，计算量小
- 稳定性高，适合工业部署

**缺点：**
- 假设奖励分布有界，对异常值敏感
- 探索参数 $c$ 需要调优
- 不支持上下文信息（需升级为 LinUCB）
- 初始阶段探索成本较高

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
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
        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_count += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

bandit = UCBBandit(n_arms=5, c=2.0)
for _ in range(100):
    arm = bandit.select_arm()
    reward = np.random.binomial(1, [0.3, 0.5, 0.7, 0.4, 0.6][arm])
    bandit.update(arm, reward)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import math
import random

class UCBBanditScratch:
    def __init__(self, n_arms, c=1.414):
        self.c = c
        self.counts = [0] * n_arms
        self.rewards = [0.0] * n_arms
        self.total = 0

    def select_arm(self):
        for a in range(len(self.counts)):
            if self.counts[a] == 0:
                return a
        ucb_scores = []
        for a in range(len(self.counts)):
            mean = self.rewards[a] / self.counts[a]
            bonus = self.c * math.sqrt(math.log(self.total) / self.counts[a])
            ucb_scores.append(mean + bonus)
        return ucb_scores.index(max(ucb_scores))

    def update(self, arm, reward):
        self.total += 1
        self.counts[arm] += 1
        self.rewards[arm] += reward
```

## 9. 可视化与结果理解

- 绘制各臂的 UCB 分数随时间的变化曲线
- 累计遗憾曲线：UCB vs 随机策略 vs 纯利用策略
- 各臂被选择的频率随时间的收敛过程
- 探索参数 $c$ 对收敛速度的影响

## 10. 模型评估

- 累计遗憾（Cumulative Regret）：$\sum_{t=1}^{T}(\mu^* - \mu_{a_t})$
- 收敛速度：多快收敛到最优臂
- 冷启动场景：新广告获得足够曝光所需轮次
- A/B 对比：UCB vs $\epsilon$-greedy vs Thompson Sampling

## 11. 常见问题与易错点

- **探索参数 $c$**：$c=\sqrt{2}$ 是理论最优，实际常取 1~3 之间调优
- **初始探索**：确保每个臂至少被尝试一次再使用 UCB 公式
- **奖励尺度**：不同量纲的奖励需归一化，否则 UCB 分数不可比
- **上下文信息**：标准 UCB 不考虑用户/广告特征，需用 LinUCB
- **非平稳环境**：奖励分布随时间变化时，需引入滑动窗口或衰减因子

## 12. 学习总结

UCB 通过"均值 + 不确定性"的策略实现确定性的探索-利用平衡，是广告冷启动中最简单可靠的探索方案。理解 UCB 是学习 LinUCB（上下文 Bandit）和 Thompson Sampling 的基础。

## 13. 练习题与思考题（含答案）

**Q1：UCB 的不确定性项为什么随 $\sqrt{\ln t / N_a(t)}$ 递减？**
A1：由 Hoeffding 不等式，置信区间宽度与 $1/\sqrt{N_a}$ 成正比，$\ln t$ 项保证置信度随总轮次增长。

**Q2：UCB 和 Thompson Sampling 的核心区别？**
A2：UCB 是确定性算法（相同状态相同决策），Thompson Sampling 是随机采样（同状态可能不同决策）。

**Q3：为什么 UCB 适合广告冷启动？**
A3：新广告数据少 → 不确定性大 → UCB 分数高 → 自然获得更多曝光 → 数据快速积累。

**Q4：如何将 UCB 扩展为支持上下文的版本？**
A4：LinUCB 用线性模型 $\hat{\mu}_a = \theta_a^T x$ 替代简单均值，不确定性用 $\sqrt{x^T A_a^{-1} x}$ 计算。

## 14. 学习路径建议

1. 理解 Multi-Armed Bandit 问题和探索-利用困境
2. 掌握 UCB 公式及其理论推导（Hoeffding 不等式）
3. 学习 Thompson Sampling（贝叶斯替代方案）
4. 进阶 LinUCB（上下文 Bandit）
5. 应用到广告冷启动系统设计
