# Thompson Sampling 学习文档

## 1. 算法基础认知

Thompson Sampling 是一种基于贝叶斯推断的 Bandit 算法，通过对奖励的后验分布进行采样来决策。核心思想：数据少时后验方差大，采样值波动大，自然倾向探索；数据多时后验收敛，采样值趋近真实值，转向利用。在广告冷启动中与 UCB 互为补充。

## 2. 核心原理

Thompson Sampling 的本质是概率匹配（Probability Matching）：以每个臂是最优臂的后验概率来选择该臂。对每个臂维护奖励的后验分布，每次决策时从各后验分布采样一个值，选择采样值最大的臂。贝叶斯更新使后验随数据不断收敛。

## 3. 数学公式与推导

**Beta-Bernoulli 模型**（伯努利奖励，如点击/不点击）：

先验分布：

$$
w_k \sim \text{Beta}(\alpha_k, \beta_k)
$$

初始时 $\alpha_k = \beta_k = 1$（均匀先验）。

贝叶斯更新规则：
- 获得正向反馈（点击）：$\alpha_k \leftarrow \alpha_k + 1$
- 获得负向反馈（未点击）：$\beta_k \leftarrow \beta_k + 1$

选择策略：

$$
a^* = \arg\max_k \tilde{w}_k, \quad \tilde{w}_k \sim \text{Beta}(\alpha_k, \beta_k)
$$

后验均值与方差：

$$
\mathbb{E}[w_k] = \frac{\alpha_k}{\alpha_k + \beta_k}, \quad \text{Var}[w_k] = \frac{\alpha_k \beta_k}{(\alpha_k + \beta_k)^2(\alpha_k + \beta_k + 1)}
$$

数据少时方差大（探索），数据多时方差小（利用）。

**高斯奖励扩展**：

$$
\mu_k \sim \mathcal{N}\left(\frac{\sum r_i}{n_k + 1}, \frac{1}{n_k + 1}\right)
$$

## 4. 训练过程讲解

1. 初始化：每个臂的 $\alpha_k = 1, \beta_k = 1$
2. 采样：从每个臂的 $\text{Beta}(\alpha_k, \beta_k)$ 中采样一个值 $\tilde{w}_k$
3. 选择：选采样值最大的臂 $a^* = \arg\max_k \tilde{w}_k$
4. 执行并观察奖励 $r \in \{0, 1\}$
5. 更新：$r=1$ 则 $\alpha_{a^*} += 1$，否则 $\beta_{a^*} += 1$
6. 重复步骤 2-5

## 5. 应用场景

- 广告冷启动探索（新广告/新创意 pCTR 采样）
- 推荐 cold start（新 item 曝光策略）
- 在线 A/B 测试
- 多目标优化中的探索

## 6. 优缺点分析

**优点：**
- 实现极其简单
- 自然平衡探索与利用（无需手动调参）
- 适合在线学习和非平稳环境
- 贝叶斯框架天然支持不确定性量化

**缺点：**
- 随机性导致相同状态可能不同决策（不利于调试）
- 需要指定先验分布（选择不当会影响初期表现）
- 高维上下文扩展比 LinUCB 更复杂
- 收敛速度理论分析比 UCB 更难

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import numpy as np

class ThompsonSamplingBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

bandit = ThompsonSamplingBandit(n_arms=5)
true_rates = [0.3, 0.5, 0.7, 0.4, 0.6]
for _ in range(100):
    arm = bandit.select_arm()
    reward = np.random.binomial(1, true_rates[arm])
    bandit.update(arm, reward)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import random
import math

class ThompsonSamplingScratch:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = [1.0] * n_arms
        self.beta = [1.0] * n_arms

    def _sample_beta(self, a, b):
        u1 = random.random()
        u2 = random.random()
        while u1 == 0:
            u1 = random.random()
        while u2 == 0:
            u2 = random.random()
        x = (-2.0 * math.log(u1)) ** 0.5 * math.cos(2.0 * math.pi * u2)
        y = (-2.0 * math.log(u1)) ** 0.5 * math.sin(2.0 * math.pi * u2)
        return x

    def select_arm(self):
        samples = []
        for a in range(self.n_arms):
            s = random.betavariate(self.alpha[a], self.beta[a])
            samples.append(s)
        return samples.index(max(samples))

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

## 9. 可视化与结果理解

- 绘制各臂的 Beta 后验分布随时间的演变（从均匀分布到尖峰）
- 累计遗憾曲线：Thompson Sampling vs UCB vs 随机策略
- 后验均值 $\alpha_k / (\alpha_k + \beta_k)$ 的收敛过程
- 探索程度可视化：后验分布的方差随时间的衰减曲线

## 10. 模型评估

- 累计遗憾（Cumulative Regret）：$\sum_{t=1}^{T}(\mu^* - r_t)$
- 后验收敛速度：后验均值与真实值的偏差
- 冷启动效率：新广告获得充分曝光所需的轮次
- A/B 对比：Thompson Sampling vs UCB vs $\epsilon$-greedy

## 11. 常见问题与易错点

- **先验选择**：$\alpha=\beta=1$ 是均匀先验（无信息），$\alpha=\beta=0.5$ 是 Jeffreys 先验
- **连续奖励**：Bernoulli 奖励用 Beta 分布，Gaussian 奖励用 Normal-Inverse-Gamma 分布
- **非平稳环境**：引入衰减因子 $\gamma$，定期缩小 $\alpha, \beta$
- **与 UCB 对比**：Thompson Sampling 随机性强（适合多样化场景），UCB 确定性强（适合需要稳定性的场景）
- **冷启动合并**：实际系统中常将 Thompson Sampling 与 pCTR 模型结合，对模型预测的后验进行采样

## 12. 学习总结

Thompson Sampling 通过贝叶斯后验采样实现自然的探索-利用平衡，是广告冷启动中与 UCB 并列的主流探索策略。其实现简单、无需手动调参的优势使其在工业界广泛使用。理解 Thompson Sampling 是构建广告冷启动系统的基础。

## 13. 练习题与思考题（含答案）

**Q1：为什么 Thompson Sampling 能自然平衡探索与利用？**
A1：数据少时后验方差大，采样值波动范围大，低概率臂也有机会被选中（探索）；数据多时方差小，采样值集中在均值附近（利用）。

**Q2：Thompson Sampling 的随机性是好是坏？**
A2：是双刃剑。好处是天然实现多样化探索，适合广告场景；坏处是相同状态可能产生不同决策，不利于调试和复现。

**Q3：如何将 Thompson Sampling 与 pCTR 模型结合？**
A3：用 pCTR 模型预测作为 Beta 分布的 $\alpha/(alpha+\beta)$，结合历史点击数据更新后验，对后验采样得到最终分数。

**Q4：Beta(1,1) 和 Beta(0.5,0.5) 先验的区别？**
A4：Beta(1,1) 是均匀分布，Beta(0.5,0.5) 是 Jeffreys 先验（两端高、中间低），后者更鼓励初期探索极端值。

## 14. 学习路径建议

1. 理解贝叶斯推断和后验分布的概念
2. 掌握 Beta-Bernoulli 模型及其更新规则
3. 对比学习 UCB（确定性 vs 随机性探索）
4. 进阶 LinUCB 和 Contextual Thompson Sampling
5. 学习广告冷启动系统中的 E&E 策略组合设计
