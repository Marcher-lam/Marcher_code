# Bandit（多臂老虎机）学习文档

## 1. 算法基础认知

Bandit（多臂老虎机，Multi-Armed Bandit, MAB）是一类在线学习问题，核心是在"探索"（Exploration，尝试新选项以获取信息）和"利用"（Exploitation，选择当前已知最优选项）之间取得平衡。

在广告系统中，Bandit 方法被广泛应用于冷启动探索、多目标权重动态调权、新广告/新素材效果探索和创意优选等场景。

## 2. 核心原理

### 问题建模

将每个候选动作（广告、策略、权重组合）视为一个"臂"，目标是在有限次尝试内最大化累积收益。

### Beta 分布采样（Thompson Sampling）

$$
w_k \sim \text{Beta}(\alpha_k, \beta_k)
$$

每次采样得到每个臂的估计值，选择采样值最大的臂。$\alpha_k$ 和 $\beta_k$ 分别记录第 $k$ 个臂的成功和失败次数。

### 综合奖励定义（广告场景）

$$
\text{Reward} = \Delta\text{Rev} + \lambda_1 \cdot \Delta\text{UX} + \lambda_2 \cdot \Delta\text{Eco} + \lambda_3 \cdot \Delta\text{ROI}
$$

### Contextual Bandit

将冷启动视为上下文多臂老虎机问题，根据广告特征选择最优探索策略。引入广告侧特征作为上下文，计算高效，适合在线部署。

## 3. 数学公式与推导

**UCB（Upper Confidence Bound）**：

$$
a_t = \arg\max_{k} \left[ \hat{\mu}_k + c \sqrt{\frac{\ln t}{N_k(t)}} \right]
$$

其中 $\hat{\mu}_k$ 为第 $k$ 臂的经验均值，$N_k(t)$ 为到时刻 $t$ 为止第 $k$ 臂被拉的次数，$c$ 控制探索力度。

**Thompson Sampling 后验更新**：

$$
\alpha_k \leftarrow \alpha_k + r_t, \quad \beta_k \leftarrow \beta_k + (1 - r_t)
$$

**遗憾界（Regret Bound）**：UCB 的累积遗憾为 $O(\sqrt{KT \ln T})$，Thompson Sampling 同阶。

## 4. 训练/运行过程讲解

1. 初始化每个臂的参数（计数、均值或 Beta 参数）
2. 每轮根据策略选择一个臂（探索或利用）
3. 观察该臂的收益 $r_t \in \{0, 1\}$
4. 更新该臂的后验参数
5. 重复直至收敛或达到预算

**工业实践**：PID 为主 + Bandit 辅助。PID 负责快速响应偏差，Bandit 负责探索更优的权重组合。字节跳动利用 Contextual Bandit 动态调整新广告的探索加权系数。

## 5. 应用场景

- 广告冷启动探索策略（新广告 eCPM 加权）
- 多目标权重动态调权（与 PID 配合）
- 新广告/新素材效果探索
- 广告创意动态优选
- eCPM 加权：$boosted\_eCPM = eCPM \times (1 + boost\_factor)$，boost_factor 随数据积累逐渐衰减

## 6. 优缺点分析

### 常见算法对比

| 方法 | 适用场景 | 响应速度 | 复杂度 | 稳定性 |
|------|---------|---------|--------|--------|
| Thompson Sampling | 离散权重空间 | 中 | 低 | 中 |
| UCB | 离散权重空间 | 中 | 低 | 高 |
| ε-greedy | 简单探索 | 快 | 低 | 中 |

**优点**：
- 在线学习，无需离线训练数据
- 自动平衡探索与利用
- 适合非平稳环境（广告效果随时间变化）

**缺点**：
- 探索阶段有流量损失
- 标准算法不利用上下文信息（需 Contextual Bandit）
- 收敛速度受臂数量影响

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np

class ThompsonSamplingBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        samples = [np.random.beta(self.alpha[k], self.beta[k]) for k in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)

class UCBBandit:
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select_arm(self):
        self.t += 1
        for k in range(self.n_arms):
            if self.counts[k] == 0:
                return k
        ucb = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = (1 - 1.0 / n) * self.values[arm] + (1.0 / n) * reward

np.random.seed(42)
bandit = ThompsonSamplingBandit(n_arms=5)
true_probs = [0.1, 0.3, 0.5, 0.7, 0.4]
for t in range(1000):
    arm = bandit.select_arm()
    reward = np.random.random() < true_probs[arm]
    bandit.update(arm, reward)
print(f"Estimated: {[a / (a + b) for a, b in zip(bandit.alpha, bandit.beta)]}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
import math

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
        self.values[arm] = (1 - 1.0 / n) * self.values[arm] + (1.0 / n) * reward

class UCBBanditSimple:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.t = 0

    def select_arm(self):
        self.t += 1
        for k in range(self.n_arms):
            if self.counts[k] == 0:
                return k
        means = self.sums / self.counts
        ucb = means + np.sqrt(2 * math.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sums[arm] += reward
```

## 9. 可视化与结果理解

- 绘制各臂的累积选择次数随时间变化（收敛到最优臂）
- 绘制累积遗憾（Regret）随时间的变化曲线
- 对比 ε-greedy、UCB、Thompson Sampling 三者的收敛速度
- 可视化 Beta 分布的先验→后验更新过程

## 10. 模型评估

- **累积遗憾**：$R_T = \sum_{t=1}^{T} (\mu^* - \mu_{a_t})$，越小越好
- **收敛速度**：找到最优臂所需的轮数
- **广告场景指标**：新广告冷启动 CTR 达标时间、探索流量占比

## 11. 常见问题与易错点

- ε-greedy 的 $\epsilon$ 固定不衰减会导致始终有 $\epsilon$ 比例的流量浪费，建议衰减
- UCB 在非平稳环境下表现不佳，需要滑动窗口或衰减因子
- Thompson Sampling 的 Beta 先验参数初始化影响冷启动探索
- 工业场景需考虑预算约束（不能无限探索）

## 12. 学习总结

Bandit 问题的核心贡献在于将"探索与利用的权衡"形式化为一个可严格分析的数学框架，并给出了累积遗憾的理论下界和达到该下界的最优算法（如 UCB、Thompson Sampling）。这些算法证明了有限次尝试内最大化累积收益是可实现的，且收敛速度有理论保证。

Bandit 的关键优势是轻量、在线、自适应，不需要离线训练即可实时学习最优策略。它最适合广告冷启动（新广告探索）、创意优选、多目标权重动态调权等需要在不确定性下快速决策的场景。Thompson Sampling 因其实现简单且收敛性能优异，通常是实际应用中的首选算法。

在知识体系中，Bandit 是本库中 MDP 和强化学习（DQN、PPO、SAC 等）的简化特例——当状态空间退化为单一状态时，强化学习问题就退化为 Bandit 问题。Contextual Bandit 则是 Bandit 向完整 RL 过渡的中间形态，引入了上下文特征。

工业最佳实践是"PID 为主 + Bandit 辅助"的混合策略：PID 负责快速响应偏差，Bandit 负责探索更优的权重组合。实际部署时需注意设置探索预算上限以控制流量损失，并结合 eCPM 加权（$boosted\_eCPM = eCPM \times (1 + boost\_factor)$）实现平滑过渡。

## 13. 练习题与思考题（含答案）

**Q1**: 什么是探索-利用困境？
> A1: 利用当前最优选项获得即时收益 vs 探索新选项获取信息以获得长期更大收益之间的权衡。

**Q2**: Thompson Sampling 为什么比 ε-greedy 更好？
> A2: TS 通过后验分布采样自然平衡探索与利用，不确定性高的臂有更大被选中概率，而 ε-greedy 的探索是随机的。

**Q3**: 在广告冷启动中，如何用 Bandit 提升 eCPM？
> A3: $boosted\_eCPM = eCPM \times (1 + boost\_factor)$，Bandit 动态调整每个新广告的 boost_factor，数据不足时加大探索，数据积累后逐渐衰减。

## 14. 学习路径建议

1. 先理解探索-利用困境的概念
2. 学习 ε-greedy、UCB、Thompson Sampling 三种基础算法
3. 学习 Contextual Bandit（LinUCB）
4. 进阶：学习 PID + Bandit 混合策略、强化学习（MDP → Q-learning → DQN）
