# Multi-Armed Bandits 学习文档

## 1. 算法基础认知

Multi-Armed Bandits（多臂老虎机）是强化学习中最基础的序贯决策问题。名字来源于赌场中的老虎机：面前有 K 台老虎机（臂），每台的期望收益未知，玩家需要在有限次数内最大化总收益。

核心矛盾是 **探索（Exploration）与利用（Exploitation）的权衡**：
- **利用**：选择当前认为最好的臂，获取短期收益
- **探索**：尝试不确定的臂，获取信息以获得长期更高收益

形式化地，设 K 个臂，每个臂 $a$ 的奖励服从未知分布 $R_a$，期望为 $\mu_a$。最优臂的期望为 $\mu^* = \max_a \mu_a$。T 轮交互后的**累积遗憾（Regret）**定义为：

$$\text{Regret}(T) = T\mu^* - \sum_{t=1}^T \mu_{a_t}$$

目标是设计策略使 Regret 增长尽可能慢。

## 2. 核心原理

### 2.1 $\epsilon$-Greedy

以概率 $\epsilon$ 随机探索，以概率 $1-\epsilon$ 选择当前最优臂。简单直观，但遗憾为 $O(T)$ 线性增长。

### 2.2 UCB（Upper Confidence Bound）

对每个臂维护置信上界：

$$\text{UCB}_a(t) = \hat{\mu}_a + \sqrt{\frac{2\ln t}{N_a(t)}}$$

其中 $\hat{\mu}_a$ 是经验均值，$N_a(t)$ 是到时刻 $t$ 为止拉臂 $a$ 的次数。选择 UCB 值最大的臂。第二项随拉动次数递减，天然实现了探索-利用平衡。遗憾为 $O(\sqrt{T\ln T})$。

### 2.3 Thompson Sampling

基于贝叶斯思想。假设每个臂的奖励分布参数有先验（如 Beta 分布），每次采样后更新后验：

$$\alpha_a \leftarrow \alpha_a + r_t, \quad \beta_a \leftarrow \beta_a + (1 - r_t)$$

每轮从每个臂的后验分布中采样，选采样值最大的臂。经验表现优异，渐近最优。

## 3. 数学公式与推导

### UCB 遗憾界推导

由 Hoeffding 不等式，对臂 $a$ 做 $n$ 次观测后：

$$P(|\hat{\mu}_a - \mu_a| \geq \epsilon) \leq 2\exp(-2n\epsilon^2)$$

令 $\epsilon = \sqrt{\frac{2\ln t}{N_a(t)}}$，并集界保证所有臂同时置信的概率至少 $1 - 2/T$。对次优臂 $a$（$\Delta_a = \mu^* - \mu_a$），它被选中的条件是置信上界超过最优臂，由此可得总遗憾：

$$\mathbb{E}[\text{Regret}] \leq \sum_{a: \Delta_a > 0} \left( \frac{8\ln T}{\Delta_a} + \frac{\pi^2}{3}\Delta_a \right)$$

## 4. 训练过程讲解

1. **初始化**：每个臂各拉一次，初始化经验均值 $\hat{\mu}_a$ 和计数 $N_a = 1$
2. **循环 t = K+1, ..., T**：
   - 计算 UCB：$\text{UCB}_a = \hat{\mu}_a + c\sqrt{\ln t / N_a}$
   - 选择 $a_t = \arg\max_a \text{UCB}_a$
   - 观测奖励 $r_t$
   - 更新：$\hat{\mu}_{a_t} \leftarrow (\hat{\mu}_{a_t} \cdot N_{a_t} + r_t) / (N_{a_t} + 1)$，$N_{a_t} \leftarrow N_{a_t} + 1$

## 5. 应用场景

- **在线广告投放**：每个广告是一个臂，点击率是奖励
- **推荐系统**：推荐候选物品的选择
- **临床试验**：不同治疗方案的选择
- **超参数调优**：不同超参数配置是不同的臂
- **路由选择**：网络路径的最优选择

## 6. 优缺点分析

| 方法 | 优点 | 缺点 |
|------|------|------|
| $\epsilon$-Greedy | 实现简单，易理解 | 遗憾线性增长，$\epsilon$ 需手动调 |
| UCB | 次线性遗憾，无需调参 | 对非平稳环境敏感 |
| Thompson Sampling | 贝叶斯最优，适应性强 | 需要指定先验分布 |

## 7. 调库实现（Python）

```python
import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedy:
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
        self.values[arm] = ((n - 1) / n) * self.values[arm] + reward / n

class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0

    def select_arm(self):
        self.total_count += 1
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = self.values + np.sqrt(2 * np.log(self.total_count) / self.counts)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + reward / n

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def select_arm(self):
        samples = [np.random.beta(self.successes[a], self.failures[a]) for a in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
```

## 8. 手工代码实现

```python
class BernoulliBandit:
    def __init__(self, probs):
        self.probs = probs
        self.n_arms = len(probs)
        self.best_prob = max(probs)

    def pull(self, arm):
        return 1.0 if np.random.random() < self.probs[arm] else 0.0

def simulate(bandit, strategy, n_steps=5000):
    regrets = []
    cumulative_regret = 0.0
    for t in range(n_steps):
        arm = strategy.select_arm()
        reward = bandit.pull(arm)
        strategy.update(arm, reward)
        cumulative_regret += bandit.best_prob - bandit.probs[arm]
        regrets.append(cumulative_regret)
    return regrets

probs = [0.2, 0.4, 0.6, 0.8, 0.5]
bandit = BernoulliBandit(probs)

regrets_eg = simulate(bandit, EpsilonGreedy(len(probs), epsilon=0.1))
regrets_ucb = simulate(bandit, UCB1(len(probs)))
regrets_ts = simulate(bandit, ThompsonSampling(len(probs)))

plt.figure(figsize=(10, 6))
plt.plot(regrets_eg, label='epsilon-Greedy')
plt.plot(regrets_ucb, label='UCB1')
plt.plot(regrets_ts, label='Thompson Sampling')
plt.xlabel('Steps')
plt.ylabel('Cumulative Regret')
plt.title('Multi-Armed Bandit Comparison')
plt.legend()
plt.grid(True)
plt.savefig('bandit_comparison.png', dpi=150)
plt.show()
```

## 9. 可视化与结果理解

累积遗憾曲线是核心评估指标：
- **$\epsilon$-Greedy**：线性增长（始终以固定概率探索次优臂）
- **UCB1**：亚线性增长（$O(\sqrt{T\ln T})$），探索逐渐减少
- **Thompson Sampling**：对数增长，收敛最快

Thompson Sampling 通常在实验中表现最好，因为它根据后验不确定性自适应调整探索力度。

## 10. 模型评估

- **累积遗憾**：$\text{Regret}(T) = \sum_{t=1}^T (\mu^* - \mu_{a_t})$
- **平均奖励**：$\bar{r} = \frac{1}{T}\sum_{t=1}^T r_t$，越接近 $\mu^*$ 越好
- **次优臂选择比例**：$\frac{1}{T}\sum_{t=1}^T \mathbf{1}[a_t \neq a^*]$
- **收敛速度**：遗憾从线性变为对数所需的时间步数

## 11. 常见问题与易错点

- **忘记初始化**：UCB 中每个臂至少拉一次，否则除零错误
- **$\epsilon$ 固定不衰减**：导致遗憾线性增长，实践中可用 $\epsilon_t = \min(1, cK/t)$
- **混淆 bandit 与 MDP**：bandit 无状态转移，是 MDP 的退化特例
- **非平稳环境**：奖励分布随时间变化时，需要滑动窗口或衰减因子
- **Thompson Sampling 先验选择**：伯努利奖励用 Beta 分布，高斯奖励用正态-逆Gamma

## 12. 学习总结

Multi-Armed Bandits 是理解探索-利用权衡的入门问题。三种经典策略从不同角度解决该问题：$\epsilon$-Greedy 从行为层面、UCB 从频率统计层面、Thompson Sampling 从贝叶斯层面。掌握 Bandit 是学习完整强化学习算法（如 MDP、策略梯度）的基础。

## 13. 练习题与思考题

**Q1**：证明 UCB1 的遗憾上界为 $O(\sqrt{KT\ln T})$。

> **提示**：利用 Hoeffding 不等式和并集界，分别分析最优臂被低估和次优臂被高估的概率。

**Q2**：为什么 Thompson Sampling 在实践中通常优于 UCB？

> **答案**：Thompson Sampling 从后验分布采样，天然考虑了不确定性；在信息充足时几乎不探索，而 UCB 的置信上界对已充分探索的臂仍然偏大，导致过度探索。

**Q3**：如果奖励不是伯努利分布而是高斯分布，Thompson Sampling 如何修改？

> **答案**：使用正态-逆Gamma先验（或简化为已知方差下的正态先验），后验更新用共轭性完成。

## 14. 学习路径建议

1. **前置知识**：概率论基础（期望、方差、贝叶斯定理）
2. **本节掌握**：$\epsilon$-Greedy → UCB → Thompson Sampling
3. **进阶方向**：
   - Contextual Bandit（上下文老虎机）：LinUCB、LinThompson
   - Adversarial Bandit（对抗老虎机）：EXP3
   - 组合优化与 Bandit：CMAB
4. **后续学习**：MDP、蒙特卡洛方法、时序差分学习、策略梯度
