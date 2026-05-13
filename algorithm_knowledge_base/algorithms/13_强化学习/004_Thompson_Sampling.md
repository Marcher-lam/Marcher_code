# Thompson Sampling 学习文档

> 基于贝叶斯的多臂老虎机算法，平衡探索与利用

---

## 1. 算法基础认知

### 1.1 一句话定义

Thompson Sampling（汤普森采样）是多臂老虎机（Multi-Armed Bandit）问题的一种求解算法，核心思想是**利用贝叶斯后验概率进行采样来平衡探索与利用**：对每个臂的奖励分布维护先验信念，每轮根据后验采样一个估计值，选择采样值最大的臂，然后根据实际奖励更新后验。

### 1.2 直觉类比

想象你在一家有很多台老虎机的赌场，每台老虎机给出奖励的概率不同，但你不知道哪台最好。你不能一直只玩同一台（可能错过更好的），也不能每台都均匀试一遍（太浪费时间）。Thompson Sampling的策略是：**每轮根据你 bisher 对各台机器的了解，随机"应该玩哪台"**——你越确定的机器，被选中的概率越接近你的估计；你不确定的机器，也有可能被选中来收集信息。这种"带随机性的按概率选择"就是探索与利用的平衡。

### 1.3 历史背景

Thompson Sampling由W.R. Thompson于1933年在论文《On the likelihood that one arm beats the other》中首次提出，最初用于临床试验中的患者分配问题。1990年代在增强学习领域被重新发现并推广。2010年代在推荐系统、在线广告等领域广泛应用，是EE（Explore-Exploit）问题的标准解法。Google的AdTimes系统和Microsoft的Explore机制都使用Thompson Sampling的变体。

### 1.4 算法定位

| 特性 | 说明 |
|------|------|
| 类型 | 强化学习 / 在线学习 |
| 输出 | 动作/臂的选择策略 |
| 模型类型 | 贝叶斯概率模型 |
| 时间复杂度 | O(K)，K为臂数 |

### 1.5 前置知识

- [必备]：概率分布（Beta分布、Gaussian分布）
- [必备]：贝叶斯统计（先验、后验）
- [扩展]：多臂老虎机基础
- [扩展]：UCB算法

---

## 2. 核心原理

### 2.1 核心思想

Thompson Sampling的核心思想是**贝叶斯采样**：维护每个臂的奖励分布后验，每轮从后验分布中采样一个值，选择采样值最大的臂。这相当于：不是机械地选择期望最高的臂（利用），也不是均匀随机选择（探索），而是按照后验概率"软选择"——某臂的期望越高，被选中的概率越大，但不是100%。

### 2.2 工作流程

```
初始化：为每个臂设置先验分布（如Beta(1,1)）
每轮循环：
    1. 对每个臂，从其后验分布采样一个值 theta_i ~ posterior_i
    2. 选择采样值最大的臂：a = argmax_i theta_i
    3. 拉取该臂，获得奖励 r
    4. 根据奖励更新该臂的后验分布
```

### 2.3 关键概念解释

- **先验分布**：选择臂之前对奖励分布的假设。通常Beta(1,1)（均匀）或Beta(α,β)。

- **后验分布**：观察到奖励数据后更新的分布。Beta-Bernoulli模型下，后验仍是Beta分布，共轭性使得更新简单。

- **探索（Exploration）**：尝试不确定的臂，获取信息。

- **利用（Exploitation）**：选择已知期望最高的臂，获取已知的高奖励。

- **后悔（Regret）**：选择最优臂获得的期望奖励与实际获得的奖励之差的期望。

### 2.4 几何/直观解释

将每个臂的奖励分布看作一条概率曲线。Thompson Sampling每轮在这条曲线上随机采一个点，然后选择最高的点对应的臂。随着数据增加，曲线越来越窄（不确定性降低），采样值越来越接近真实均值。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| K | 臂的数量 | 标量 |
| a_k | 第k个臂 | 标量 |
| r | 奖励 | 标量 |
| t | 当前轮次 | 标量 |
| α_k | 第k个臂的Beta分布参数（成功次数） | 标量 |
| β_k | 第k个臂的Beta分布参数（失败次数） | 标量 |
| μ_k | 第k个臂的真实成功概率 | 标量 |
| θ_k | 第k个臂的采样值 | 标量 |
| T_k(t) | 到时刻t拉取臂k的次数 | 标量 |
| S_k(t) | 到时刻t臂k获得奖励为1的次数 | 标量 |

### 3.2 问题形式化

**多臂老虎机形式化**：

K个臂，第k个臂的奖励服从Bernoulli(μ_k)分布。目标是一轮接一轮地选择臂，最大化累计奖励。

每轮的决策：
$$a_t = \arg\max_{k} \theta_k$$

其中θ_k从后验分布采样：
$$\theta_k \sim Beta(\alpha_k + S_k(t-1), \beta_k + T_k(t-1) - S_k(t-1))$$

### 3.3 目标函数/损失函数

累计奖励（无显式损失函数）：
$$G = \sum_{t=1}^{T} r_t$$

或最小化累计后悔：
$$Regret(T) = \sum_{t=1}^{T} (\mu^* - \mu_{a_t})$$

其中μ*是最优臂的期望奖励。

Thompson Sampling的期望累计后悔为O(log T)，与UCB相当，且常数更小。

### 3.4 推导过程

**步骤1：Beta-Bernoulli共轭**

假设每个臂的成功概率μ_k ∈ [0,1]，奖励r∈{0,1}（伯努利分布）。

对μ_k设置先验：Beta(α, β)

观察到奖励r后，后验：
$$P(\mu_k | r) \propto P(r | \mu_k) P(\mu_k)$$

由于伯努利分布是二项分布的共轭先验，后验仍是Beta分布：
$$\mu_k | data \sim Beta(\alpha + r, \beta + (1-r))$$

**步骤2：采样决策**

每轮，从每个臂的后验采样：
$$\theta_k \sim Beta(\alpha_k + S_k, \beta_k + F_k)$$

选择：
$$a = \arg\max_k \theta_k$$

**步骤3：更新**

获得奖励r后，更新参数：
- 如果r=1：α_k ← α_k + 1
- 如果r=0：β_k ← β_k + 1

### 3.5 算法步骤

```
输入：臂数K，总轮数T
输出：选择序列

1. 初始化：每个臂k，设 alpha_k = 1, beta_k = 1

2. for t = 1 to T:
    a. for each arm k:
        采样 theta_k ~ Beta(alpha_k, beta_k)
    
    b. 选择臂: a_t = argmax_k theta_k
    
    c. 拉取臂a_t，获得奖励r
    
    d. 更新:
       if r == 1: alpha_{a_t} += 1
       else: beta_{a_t} += 1
```

---

## 4. 训练过程讲解

### 4.1 参数初始化

- **乐观初始化**：设alpha=1, beta=1，即先验为均匀分布
- **保守初始化**：可以设alpha=α0, beta=β0，根据历史数据
- **多臂初始化**：对不同臂可以设不同的先验

### 4.2 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| prior_alpha | 先验成功次数 | 0.5-2 | 1 |
| prior_beta | 先验失败次数 | 0.5-2 | 1 |
| epsilon | 探索率 | 0-0.1 | 0 |

### 4.3 收敛条件

- **固定轮数**：跑T轮后停止
- **累计后悔阈值**：累计后悔小于某值后停止
- **平稳性检测**：选择分布变化小于阈值

---

## 5. 应用场景

### 5.1 典型应用

**在线广告投放**：每展示一次广告是"拉取一个臂"，用户点击是"奖励=1"。

**推荐系统**：每次推荐是选择一个"臂"，用户是否接受是奖励。

**临床试验**：给患者分配治疗方案，患者的恢复是奖励。

**A/B测试**：动态调整流量分配，快速学习最佳版本。

**新闻推荐**：推荐不同类别的新闻，学习用户偏好。

### 5.2 适用数据特征

- **二元奖励**：点击/未点击、购买/未购买
- **在线学习**：数据流式到达
- **探索必要**：对未知有不确定性
- **成本敏感**：探索有成本，不能均匀探索

### 5.3 不适用场景

- **离线评估**：有完整数据的批量学习
- **连续奖励**：需要其他分布假设
- **对抗环境**：对手会适应性变化

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 自然平衡 | 自动探索与利用平衡 | 贝叶斯假设合理 |
| 易于实现 | 后验更新简单 | 共轭分布 |
| 期望后悔低 | O(log T)级别 | 奖励分布已知 |
| 可扩展 | 支持 contextual | 扩展到线性模型 |
| 样本高效 | 比epsilon-greedy高效 | 短期在线环境 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 需要先验 | 先验影响结果 | 数据驱动初始化 |
| 计算成本 | 每轮要采样K次 | 批量采样 |
| 共轭限制 | 只能用共轭分布 | 变分近似 |
| 离散臂 | 不适合连续 | 分箱近似 |

### 6.3 与同类算法对比

| 算法 | 复杂度 | 期望后悔 | 特点 |
|------|--------|----------|------|
| Epsilon-Greedy | O(1) | O(√T) | 简单但差 |
| UCB1 | O(K log T) | O(log T) | 无贝叶斯假设 |
| Thompson Sampling | O(K) | O(log T) | 贝叶斯方法 |
| LinUCB | O(K d) | O(d log T) | 连续臂 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
"""
Thompson Sampling 调库实现
"""

import numpy as np
from typing import List, Tuple, Optional

class ThompsonSamplingBandit:
    """
    Thompson Sampling 多臂老虎机
    
    使用Beta-Bernoulli模型，适合二元奖励
    """
    
    def __init__(self, n_arms: int, prior_alpha: float = 1.0, 
                 prior_beta: float = 1.0):
        """
        初始化
        
        参数:
            n_arms: 臂的数量
            prior_alpha: 先验Alpha参数
            prior_beta: 先验Beta参数
        """
        self.n_arms = n_arms
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # 初始化后验参数
        self.alpha = np.full(n_arms, prior_alpha)
        self.beta = np.full(n_arms, prior_beta)
        
        # 统计
        self.counts = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)
        
    def select_arm(self) -> int:
        """
        选择臂
        
        返回:
            选中的臂索引
        """
        # 从每个臂的后验分布采样
        samples = np.random.beta(self.alpha, self.beta)
        
        # 选择采样值最大的臂
        arm = np.argmax(samples)
        return arm
    
    def update(self, arm: int, reward: float):
        """
        更新后验
        
        参数:
            arm: 被选中的臂
            reward: 获得的奖励（0或1）
        """
        # 更新后验参数
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        
        # 更新统计
        self.counts[arm] += 1
        self.rewards[arm] += reward
    
    def get_success_rate(self, arm: int) -> float:
        """
        获取估计成功率
        
        参数:
            arm: 臂索引
            
        返回:
            估计的成功概率
        """
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total_counts = self.counts.sum()
        return {
            'counts': self.counts,
            'rewards': self.rewards,
            'success_rates': self.alpha / (self.alpha + self.beta),
            'total_pulls': total_counts
        }


class GaussianThompsonSampling:
    """
    高斯奖励的Thompson Sampling
    
    适合连续奖励（如点击次数、金额）
    """
    
    def __init__(self, n_arms: int, prior_mu: float = 0.0,
                 prior_sigma: float = 1.0, noise_sigma: float = 1.0):
        """
        初始化
        
        ��数:
            n_arms: 臂数
            prior_mu: 先验均值
            prior_sigma: 先验标准差
            noise_sigma: 噪声标准差
        """
        self.n_arms = n_arms
        
        # 使用正态-正态共轭
        self.prior_mu = prior_mu
        self.prior_precision = 1.0 / (prior_sigma ** 2)
        self.noise_precision = 1.0 / (noise_sigma ** 2)
        
        # 后验参数
        self.posterior_mu = np.full(n_arms, prior_mu)
        self.posterior_precision = np.full(n_arms, self.prior_precision)
        
        self.counts = np.zeros(n_arms)
        self.sum_rewards = np.zeros(n_arms)
        
    def select_arm(self) -> int:
        """选择臂"""
        # 从每个臂的后验采样
        samples = np.random.normal(
            self.posterior_mu, 
            1.0 / np.sqrt(self.posterior_precision)
        )
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        """更新后验（在线贝叶斯更新）"""
        # 更新后验精度
        self.posterior_precision[arm] += self.noise_precision
        # 更新后验均值
        self.posterior_mu[arm] = (
            (self.prior_precision * self.prior_mu + self.noise_precision * reward +
             self.posterior_precision[arm] * self.posterior_mu[arm] - 
             self.noise_precision * reward) / self.posterior_precision[arm]
        )
        
        # 精确更新
        precision_sum = self.prior_precision + self.counts[arm] * self.noise_precision
        reward_sum = self.prior_precision * self.prior_mu + self.noise_precision * self.sum_rewards[arm]
        
        self.posterior_mu[arm] = reward_sum / precision_sum
        self.posterior_precision[arm] = precision_sum
        
        # 更新统计
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward


def simulate_bandit(n_arms: int = 5, n_rounds: int = 1000, 
                  true_probs: List[float] = None) -> dict:
    """
    模拟Thompson Sampling
    
    参数:
        n_arms: 臂数
        n_rounds: 总轮数
        true_probs: 各臂的真实成功概率
        
    返回:
        统计结果
    """
    if true_probs is None:
        np.random.seed(42)
        true_probs = np.random.random(n_arms)
        true_probs = true_probs / true_probs.sum() * 0.6
        true_probs[0] = 0.4
    
    # 创建bandit
    bandit = ThompsonSamplingBandit(n_arms)
    
    # 记录
    selected_arms = []
    rewards_list = []
    cumulative_rewards = []
    
    # 模拟
    for t in range(n_rounds):
        # 选择臂
        arm = bandit.select_arm()
        
        # 模拟奖励
        reward = 1 if np.random.random() < true_probs[arm] else 0
        
        # 更新
        bandit.update(arm, reward)
        
        # 记录
        selected_arms.append(arm)
        rewards_list.append(reward)
        cumulative_rewards.append(sum(rewards_list))
    
    return {
        'selected_arms': selected_arms,
        'rewards': rewards_list,
        'cumulative_rewards': cumulative_rewards,
        'estimated_probs': [bandit.get_success_rate(i) for i in range(n_arms)],
        'true_probs': true_probs.tolist(),
        'counts': bandit.counts
    }


def demo():
    """演示"""
    print("=" * 50)
    print("Thompson Sampling 演示")
    print("=" * 50)
    
    # 模拟
    results = simulate_bandit(n_arms=5, n_rounds=1000)
    
    print(f"\n真实成功率: {results['true_probs']}")
    print(f"估计成功率: {results['estimated_probs']}")
    print(f"各臂被选择次数: {results['counts']}")
    print(f"累计奖励: {results['cumulative_rewards'][-1]}")
    
    # 最优臂
    best_arm = np.argmax(results['true_probs'])
    print(f"\n理论最优臂: {best_arm}")
    print(f"最优臂被选择次数: {results['counts'][best_arm]}")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
"""
Thompson Sampling 手工实现
"""

import numpy as np
import random
from collections import defaultdict

class ManualThompsonSampling:
    """
    手工实现的Thompson Sampling
    
    不依赖numpy，仅用标准库
    """
    
    def __init__(self, n_arms: int):
        """
        初始化
        
        参数:
            n_arms: 臂数
        """
        self.n_arms = n_arms
        
        # Beta分布参数：alpha（成功），beta（失败）
        self.alpha = [1] * n_arms
        self.beta = [1] * n_arms
        
        # 统计
        self.counts = [0] * n_arms
        self.successes = [0] * n_arms
    
    def _sample_beta(self, alpha: float, beta: float) -> float:
        """
        手动采样Beta分布
        
        使用Box-Muller变换近似
        """
        # 使用Gamma分布采样
        # Gamma(α, 1) / (Gamma(α, 1) + Gamma(β, 1))
        
        def sample_gamma(shape):
            """Gamma分布采样（近似）"""
            if shape < 1:
                # 使用拒绝采样
                while True:
                    u = random.random()
                    v = random.random()
                    if u < 1:
                        return v ** (1 / shape)
            else:
                # 近似正态
                return max(0.01, random.gauss(shape, shape ** 0.5))
        
        x = sample_gamma(alpha)
        y = sample_gamma(beta)
        return x / (x + y)
    
    def select_arm(self) -> int:
        """
        选择臂
        
        返回:
            选中的臂索引
        """
        samples = []
        for i in range(self.n_arms):
            # 采样
            theta = self._sample_beta(self.alpha[i], self.beta[i])
            samples.append(theta)
        
        # 选择最大的
        return samples.index(max(samples))
    
    def update(self, arm: int, reward: int):
        """
        更新后验
        
        参数:
            arm: 被选中的臂
            reward: 奖励（0或1）
        """
        if reward == 1:
            self.alpha[arm] += 1
            self.successes[arm] += 1
        else:
            self.beta[arm] += 1
        
        self.counts[arm] += 1
    
    def get_estimates(self):
        """获取估计"""
        return [self.alpha[i] / (self.alpha[i] + self.beta[i]) 
                for i in range(self.n_arms)]


class SimplifiedThompsonBandit:
    """
    简化版：直接使用随机选择近似
    
    适用于不需要精确后验的场景
    """
    
    def __init__(self, n_arms: int, explore_factor: float = 0.1):
        """
        初始化
        
        参数:
            n_arms: 臂数
            explore_factor: 探索因子
        """
        self.n_arms = n_arms
        self.explore_factor = explore_factor
        
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
    
    def select_arm(self) -> int:
        """选择臂"""
        # 添加探索噪声
        values = []
        for i in range(self.n_arms):
            noise = random.gauss(0, self.explore_factor / (self.counts[i] + 1))
            values.append(self.values[i] + noise)
        
        return values.index(max(values))
    
    def update(self, arm: int, reward: float):
        """更新"""
        n = self.counts[arm]
        self.values[arm] = (n * self.values[arm] + reward) / (n + 1)
        self.counts[arm] += 1


def manual_demo():
    """手工实现演示"""
    print("=" * 50)
    print("Thompson Sampling 手工实现演示")
    print("=" * 50)
    
    # 真实成功率
    np.random.seed(42)
    true_probs = np.random.random(5)
    true_probs = true_probs / true_probs.sum() * 0.5
    
    print(f"\n真实成功率: {true_probs.tolist()}")
    
    # 模拟
    bandit = ManualThompsonSampling(n_arms=5)
    
    for t in range(500):
        arm = bandit.select_arm()
        reward = 1 if np.random.random() < true_probs[arm] else 0
        bandit.update(arm, reward)
    
    estimates = bandit.get_estimates()
    print(f"估计成功率: {estimates}")
    print(f"选择次数: {bandit.counts}")


if __name__ == "__main__":
    manual_demo()
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_thompson_sampling():
    """可视化Thompson Sampling"""
    
    np.random.seed(42)
    
    # 设置
    n_arms = 3
    n_rounds = 500
    true_probs = [0.3, 0.5, 0.7]
    
    # 模拟
    results = simulate_bandit(n_arms=n_arms, n_rounds=n_rounds, 
                               true_probs=true_probs)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 各臂选择次数
    axes[0, 0].bar(range(n_arms), results['counts'], color='steelblue')
    axes[0, 0].set_xlabel('Arm')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Arm Selection Counts')
    axes[0, 0].set_xticks(range(n_arms))
    axes[0, 0].set_xticklabels([f'Arm {i}' for i in range(n_arms)])
    
    # 2. 真实vs估计
    x = np.arange(n_arms)
    width = 0.35
    axes[0, 1].bar(x - width/2, results['true_probs'], width, label='True')
    axes[0, 1].bar(x + width/2, results['estimated_probs'], width, label='Estimated')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('True vs Estimated')
    axes[0, 1].legend()
    
    # 3. 累计奖励
    axes[1, 0].plot(results['cumulative_rewards'], 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Cumulative Reward')
    axes[1, 0].set_title('Cumulative Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 后验分布
    # 简化：只画Beta分布曲线
    from scipy.stats import beta as beta_dist
    x = np.linspace(0, 1, 100)
    colors = ['red', 'green', 'blue']
    
    for i in range(n_arms):
        a, b = results['estimated_probs'][i] * 100 + 1, (1 - results['estimated_probs'][i]) * 100 + 1
        y = beta_dist.pdf(x, a, b)
        axes[1, 1].plot(x, y, color=colors[i], label=f'Arm {i}')
    
    axes[1, 1].set_xlabel('Success Rate')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Posterior Distributions')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('thompson_sampling.png', dpi=150)
    plt.show()


def plot_regret():
    """可视化Regret"""
    
    np.random.seed(42)
    
    # 模拟不同算法的Regret
    n_rounds = 1000
    true_probs = [0.4, 0.6, 0.8]
    
    # Thompson Sampling
    ts_results = simulate_bandit(n_arms=3, n_rounds=n_rounds, true_probs=true_probs)
    
    # 计算Regret
    optimal_prob = max(true_probs)
    ts_regret = np.cumsum([optimal_prob - true_probs[r] for r in ts_results['selected_arms']])
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(ts_regret, 'b-', linewidth=2, label='Thompson Sampling')
    plt.plot([np.log(t+1) for t in range(n_rounds)], 'r--', 
            linewidth=2, label='O(log t)')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.title('Regret Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regret.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    visualize_thompson_sampling()
    plot_regret()
```

**结果解读**：

1. **选择次数**：真实最优臂（概率0.7）被选次数最多，符合预期。
2. **估计vs真实**：估计值逐渐逼近真实值，初期波动大，后期稳定。
3. **后验分布**：随着数据增加，分布越来越窄（不确定性降低）。

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 公式 |
|------|------|------|
| 累计奖励 | 总奖励 | Σr_t |
| 平均奖励 | 每轮平均 | Σr_t / T |
| 后悔 | 与最优差 | Σ(μ* - μ_a) |
| 期望后悔 | 后悔的期望 | E[Σ(μ* - μ_a)] |
| 伪后悔 | 样本估计后悔 | 基于累计奖励 |

### 10.2 后悔上界

Thompson Sampling的期望累计后悔：
$$E[Regret(T)] = O\left(\frac{K \log T}{\Delta_{min}}\right)$$

其中Δ_min是最小次优间隙。

---

## 11. 常见问题与易错点

### 11.1 问题1：先验选择影响结果

**原因**：先验Beta(1,1)可能在数据少时导致过度探索。

**解决方案**：用数据驱动先验。

```python
# 数据初始化
prior_alpha = 1 + success_history
prior_beta = 1 + failure_history
```

### 11.2 问题2：臂数量大时计算慢

**原因**：每轮采样K次，K大时慢。

**解决方案**：使用Top-K采样或批处理。

### 11.3 问题3：非共轭分布

**原因**：无法用Beta分布模型。

**解决方案**：变分近似或MCMC。

---

## 12. 学习总结

### 核心要点回顾：

1. **贝叶斯框架**：维护后验分布，每轮采样选择
2. **探索-利用平衡**：通过随机性自然平衡
3. **共轭更新**：Beta-Bernoulli后验仍是Beta
4. **O(log T)后悔**：理论保证

### 从Thompson Sampling到其他算法：

- Thompson Sampling → UCB（无贝叶斯版本）
- Thompson Sampling → LinUCB（线性扩展）
- Thompson Sampling → Contextual Bandit（加入上下文）
- Thompson Sampling → EXP3（对抗性扩展）

### 实践建议：

1. 默认用Beta-Bernoulli，适合二元奖励
2. 先验用Beta(1,1)即均匀分布
3. 实时要求高用批量采样或简化版

---

## 13. 练习题与思考题（含答案）

### 练习题

**习题1：基础计算**

问题：假设两臂Bandit，A的真实成功概率0.8，B是0.5。用Thompson Sampling，初始先验Beta(1,1)，第一轮A被选中并获得奖励=1。更新后的后验是什么？

<details>
<summary>答案</summary>

初始：Beta(1,1)

A被选中，奖励=1：
- α_A = 1 + 1 = 2
- β_A = 1 + 0 = 1

B保持：
- α_B = 1
- β_B = 1

所以A的后验是Beta(2,1)，B是Beta(1,1)。

</details>

**习题2：编程实践**

问题：用Python实现Gaussian Thompson Sampling。

<details>
<summary>答案</summary>

```python
import random
import numpy as np

class GaussianTS:
    def __init__(self, n_arms):
        self.n = n_arms
        self.counts = [0] * n
        self.sums = [0.0] * n
        self.prior_var = 1.0
    
    def select_arm(self):
        # 采样
        samples = []
        for i in range(self.n):
            mu = self.sums[i] / max(1, self.counts[i])
            var = self.prior_var / (1 + self.counts[i])
            sample = random.gauss(mu, var**0.5)
            samples.append(sample)
        return samples.index(max(samples))
    
    def update(self, arm, reward):
        self.sums[arm] += reward
        self.counts[arm] += 1
```

</details>

**习题3：理论推导**

问题：Thompson Sampling比Epsilon-Greedy好的原因？

<details>
<summary>答案</summary>

1. Epsilon-Greedy固定概率探索，Thompson Sampling根据后验动态调整。
2. Thompson Sampling在数据少时自然探索多，数据多��自��利用多。
3. 期望后悔Thompson Sampling是O(log T)，Epsilon-Greedy是O(√T)。

</details>

### 思考题

**思考题1**：如何扩展到Contextual Bandit？

<details>
<summary>答案</summary>

1. 假设奖励与上下文线性：r = w^T x + noise
2. 使用Linear Regression建模E[r|x]
3. 每轮从后验采样w
4. 按采样w计算各臂的期望，选择最高的

</details>

**思考题2**：Thompson Sampling在实际系统中如何部署？

<details>
<summary>答案</summary>

1. 离线预训练：用历史数据建立先验
2. 在线服务：每请求一次采样+更新
3. A/B测试：与现有方法对比
4. 监控：监控选择分布和累计奖励

</details>

---

## 14. 学习路径建议

### 初级阶段（掌握基础）

1. 理解多臂老虎机问题
2. 理解Epsilon-Greedy
3. 掌握Thompson Sampling原理
4. 实现简单版本

**学习时间**：1-2天

### 中级阶段（理解扩展）

1. 学习UCB算法
2. 学习贝叶斯推导
3. 理解后悔理论
4. 实践广告场景

**学习时间**：1周

### 高级阶段（扩展应用）

1. Contextual Bandit
2. LinUCB
3. 变分近似
4. 生产系统部署

**学习时间**：2-3周

### 实践项目建议

1. **基础项目**：模拟老虎机实验
2. **进阶项目**：广告投放系统
3. **挑战项目**：推荐系统EE问题

### 推荐资源

- **论文**：Thompson原始论文（1933）
- **书籍**：《Bandit Algorithms》- Bubeck
- **课程**：Berkeley CS294-134

---

**文档结束**