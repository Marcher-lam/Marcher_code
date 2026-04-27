# Multi-Armed Bandits 学习文档

> Multi-Armed Bandits（多臂老虎机）是强化学习中一类经典的单步决策问题，完美体现了"探索-利用权衡"（Exploration-Exploitation Trade-off）的核心挑战

---

## 1. 算法基础认知

### 1.1 一句话定义

**Multi-Armed Bandits（多臂老虎机，MAB）** 是一种序贯决策框架，智能体在每个时刻从K个"臂"（动作）中选择一个，根据该臂的奖励分布获得随机奖励，目标是最大化累计奖励。这一框架优雅地形式化了"应该继续选择已知的好选择，还是尝试新的选择？"这一核心困境。

### 1.2 直觉类比

想象你进入一家有K台老虎机的赌场：

| 情形 | 决策 | 代价 |
|------|------|------|
| 一直投币到之前赢过的那台 | 利用已知信息 | 可能错过更好的机器 |
| 每台机器都尝试几次 | 探索新可能性 | 浪费在表现差的机器上 |

这就是**探索-利用困境**（Exploration-Exploitation Dilemma）。MAB算法就是要找到最佳平衡点。

### 1.3 历史背景

| 年份 | 里程碑 |
|------|--------|
| 1933 | Thompson (1933) 提出最早的最优臂识别方案 |
| 1952 | Robbins 正式提出多臂老虎机问题 |
| 1979 | Lai & Robbins 给出UCB的下界分析 |
| 1985 | Agrawal 提出UCB1算法 |
| 1998 | Auer 等证明ε-Greedy的遗憾上界 |
| 2011 | Chapelle & Li 提出Thompson Sampling的现代分析 |

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 问题类型 | 在线学习/序贯决策 |
| 学习范式 | 强化学习（单状态、KMDP） |
| 核心挑战 | 探索vs利用的权衡 |
| 优化目标 | 最小化累计遗憾 |

### 1.5 前置知识

- 概率论基础（期望、方差、分布）
- 统计学基础（置信区间、假设检验）
- Python编程（NumPy、Matplotlib）

---

## 2. 核心原理

### 2.1 问题形式化

**K臂老虎机**可形式化为：

- **臂集合**：$\mathcal{A} = \{1, 2, ..., K\}$
- **奖励分布**：$r_t \sim P(\cdot|a_t)$，对于每个臂$a$，奖励服从某个未知分布
- **期望奖励**：$\mu_a = \mathbb{E}[r|a]$，目标是找到期望奖励最高的臂

**累计遗憾**（Cumulative Regret）：
$$R_T = T\mu^* - \sum_{t=1}^{T}\mu_{a_t}$$

其中$\mu^* = \max_a \mu_a$是最优臂的期望奖励。

### 2.2 核心算法

#### 2.2.1 ε-Greedy（ε-贪心）

以概率$1-\varepsilon$选择当前估计最好的臂，以概率$\varepsilon$随机选择任意臂：

```python
import numpy as np

class EpsilonGreedy:
    """ε-Greedy算法"""
    
    def __init__(self, n_arms, epsilon=0.1, seed=42):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)      # 每个臂被选中的次数
        self.values = np.zeros(n_arms)      # 每个臂的估计价值
        np.random.seed(seed)
    
    def select_arm(self):
        """选择臂"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(self.n_arms)
        else:
            # 利用：选择估计价值最高的臂
            return np.argmax(self.values)
    
    def update(self, arm, reward):
        """更新臂的价值估计"""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        # 增量更新：Q_{n+1} = Q_n + (r - Q_n) / n
        self.values[arm] += (reward - value) / n
```

#### 2.2.2 UCB1（上置信界算法）

使用置信上界平衡探索与利用，公式为：
$$UCB_a = \bar{\mu}_a + \sqrt{\frac{2\ln t}{n_a}}$$

其中$\bar{\mu}_a$是臂$a$的平均奖励，$n_a$是被选中的次数，$t$是总时间步。

```python
import math

class UCB1:
    """UCB1算法（Upper Confidence Bound）"""
    
    def __init__(self, n_arms, c=2.0, seed=42):
        self.n_arms = n_arms
        self.c = c  # 探索参数
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0
        np.random.seed(seed)
    
    def select_arm(self):
        """选择UCB值最大的臂"""
        self.total_counts += 1
        
        # 首先确保每个臂都被选择过至少一次
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # 计算UCB值
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            exploitation = self.values[arm]
            exploration = math.sqrt(
                2 * math.log(self.total_counts) / self.counts[arm]
            )
            ucb_values[arm] = exploitation + self.c * exploration
        
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        """更新臂的价值估计"""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] += (reward - value) / n
```

#### 2.2.3 Thompson Sampling（汤普森采样）

基于贝叶斯后验进行采样：假设奖励服从伯努利分布，使用Beta分布作为共轭先验：

```python
from scipy.stats import beta as beta_dist

class ThompsonSampling:
    """Thompson Sampling算法（伯努利奖励）"""
    
    def __init__(self, n_arms, alpha_prior=1, beta_prior=1, seed=42):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * alpha_prior  # 成功次数 + 1
        self.beta = np.ones(n_arms) * beta_prior    # 失败次数 + 1
        np.random.seed(seed)
    
    def select_arm(self):
        """从后验分布采样并选择最大期望的臂"""
        samples = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            # 从Beta分布采样
            samples[arm] = beta_dist.rvs(
                self.alpha[arm], self.beta[arm]
            )
        return np.argmax(samples)
    
    def update(self, arm, reward):
        """更新Beta分布参数"""
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

### 2.3 工作流程

```
┌─────────────────────────────────────┐
│         初始化                      │
│   - 设置臂数量K、算法参数           │
│   - 初始化Q值、计数器等            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      每个时间步t=1,2,...,T          │
│   1. 选择臂 a_t (基于策略π)        │
│   2. 获得奖励 r_t ~ P(·|a_t)       │
│   3. 更新价值估计 Q(a_t)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      计算累计 regret                │
│   R_T = Tμ* - Σμ(a_t)             │
└─────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含�� | 维度 |
|------|------|------|
| $K$ | 臂的数量 | 标量 |
| $T$ | 总时间步 | 标量 |
| $a_t$ | 时间步t选择的臂 | 标量 |
| $r_t$ | 时间步t获得的奖励 | 标量 |
| $\mu_a$ | 臂$a$的真实期望奖励 | 标量 |
| $\hat{\mu}_a$ | 臂$a$的估计期望奖励 | 标量 |
| $n_a$ | 臂$a$被选中的次数 | 标量 |

### 3.2 累积遗憾上界

#### 3.2.1 ε-Greedy的遗憾

假设所有非最优臂的期望奖励与最优臂的差距为$\Delta_a = \mu^* - \mu_a$，则：

$$\mathbb{E}[R_T] \leq \varepsilon T \max_a \Delta_a + \sum_a \frac{\ln T}{\Delta_a}$$

#### 3.2.2 UCB1的遗憾

$$\mathbb{E}[R_T] \leq \sum_a \frac{8\ln T}{\Delta_a} + K$$

#### 3.2.3 Thompson Sampling的遗憾

$$\mathbb{E}[R_T] \leq \sum_a \frac{2\ln T}{\Delta_a} + K$$

### 3.3 伯努利老虎机的形式化

对于伯努利老虎机（奖励为0或1）：

**概率形式**：$P(r=1|a) = \mu_a$

**后验更新**：
$$P(\mu_a | \mathcal{D}) \propto P(\mathcal{D} | \mu_a) P(\mu_a)$$

使用Beta-Bernoulli共轭：
$$P(\mu_a | \mathcal{D}) = \text{Beta}(\alpha_a + S_a, \beta_a + F_a)$$

其中$S_a$、$F_a$分别是成功和失败的次数。

### 3.4 期望遗憾下界

Lai & Robbins (1985) 证明，对于任何算法：

$$\liminf_{T \to \infty} \frac{\mathbb{E}[R_T]}{\ln T} \geq \sum_a \frac{1}{\text{KL}(P_a || P^*)}$$

其中$\text{KL}$是KL散度，$P^*$是最优臂的奖励分布。

### 3.5 软最大化（Softmax）变体

使用Gumbel-Softmax近似：

$$P(a) = \frac{\exp(\beta Q_a)}{\sum_{a'} \exp(\beta Q_{a'})}$$

其中$\beta$是温度参数。

---

## 4. 训练过程讲解

### 4.1 环境设置

```python
import numpy as np
import matplotlib.pyplot as plt

class BanditEnvironment:
    """老虎机环境"""
    
    def __init__(self, probs, seed=42):
        """
        Args:
            probs: 每个臂获得奖励1的概率
        """
        self.probs = np.array(probs)
        self.n_arms = len(probs)
        self.best_arm = np.argmax(probs)
        self.best_prob = max(probs)
        np.random.seed(seed)
    
    def step(self, arm):
        """执行动作，返回奖励"""
        return 1 if np.random.random() < self.probs[arm] else 0
    
    def get_optimal_regret(self, n_steps):
        """理论最优的累计regret"""
        return n_steps * (1 - self.best_prob) if self.best_prob < 1 else 0
```

### 4.2 对比实验

```python
def run_comparison(n_arms=10, n_steps=1000, n_runs=100):
    """对比不同算法的性能"""
    
    # 设置真实奖励概率（第一个臂最优）
    true_probs = np.random.rand(n_arms)
    true_probs[0] = max(true_probs) + 0.1  # 确保第一个臂最优
    
    results = {
        'epsilon_greedy': [],
        'ucb1': [],
        'thompson_sampling': []
    }
    
    for run in range(n_runs):
        env = BanditEnvironment(true_probs, seed=run)
        
        eg = EpsilonGreedy(n_arms, epsilon=0.1, seed=run)
        ucb = UCB1(n_arms, c=2.0, seed=run)
        ts = ThompsonSampling(n_arms, seed=run)
        
        for t in range(n_steps):
            # ε-Greedy
            arm = eg.select_arm()
            reward = env.step(arm)
            eg.update(arm, reward)
            
            # UCB1
            arm = ucb.select_arm()
            reward = env.step(arm)
            ucb.update(arm, reward)
            
            # Thompson Sampling
            arm = ts.select_arm()
            reward = env.step(arm)
            ts.update(arm, reward)
        
        # 记录最终regret
        results['epsilon_greedy'].append(eg.total_reward)
        results['ucb1'].append(ucb.total_reward)
        results['thompson_sampling'].append(ts.total_reward)
    
    return results
```

### 4.3 可视化

```python
def plot_regret_curves(n_arms=5, n_steps=2000, n_runs=50):
    """绘制累积regret曲线"""
    
    # 5个臂，真实概率[0.1, 0.2, 0.3, 0.4, 0.5]
    true_probs = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.15] * (n_arms - 5)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    algorithms = [
        ('ε-Greedy', 'blue'),
        ('UCB1', 'green'),
        ('Thompson Sampling', 'red')
    ]
    
    for ax, (name, color) in zip(axes, algorithms):
        cumulative_regrets = []
        
        for run in range(n_runs):
            env = BanditEnvironment(true_probs, seed=run)
            
            if name == 'ε-Greedy':
                agent = EpsilonGreedy(n_arms, epsilon=0.1, seed=run)
            elif name == 'UCB1':
                agent = UCB1(n_arms, c=2.0, seed=run)
            else:
                agent = ThompsonSampling(n_arms, seed=run)
            
            regrets = []
            optimal_prob = max(true_probs)
            
            for t in range(n_steps):
                arm = agent.select_arm()
                reward = env.step(arm)
                agent.update(arm, reward)
                
                # 累积regret
                regret = regrets[-1] + (optimal_prob - true_probs[arm]) if regrets else 0
                regrets.append(regret)
            
            cumulative_regrets.append(regrets)
        
        mean_regret = np.mean(cumulative_regrets, axis=0)
        std_regret = np.std(cumulative_regrets, axis=0)
        
        ax.plot(mean_regret, color=color, label='Mean Regret')
        ax.fill_between(
            range(n_steps),
            mean_regret - std_regret,
            mean_regret + std_regret,
            alpha=0.3, color=color
        )
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('bandit_comparison.png', dpi=150)
    plt.show()
```

### 4.4 超参数推荐

| 算法 | 超参数 | 推荐范围 | 默认值 |
|------|--------|----------|--------|
| ε-Greedy | $\varepsilon$ | 0.01-0.3 | 0.1 |
| UCB1 | $c$ (探索系数) | 1-4 | 2 |
| Thompson Sampling | $\alpha, \beta$ (先验) | 1,1 | 1,1 |

---

## 5. 应用场景

### 5.1 典型应用

| 场景 | 臂 | 奖励 | 说明 |
|------|-----|------|------|
| **广告投放** | 不同广告创意 | 点击率 | 最大化点击 |
| **临床试验** | 不同药物 | 治愈率 | 平衡疗效和患者安全 |
| **推荐系统** | 不同物品 | 用户交互 | 最大化参与度 |
| **A/B测试** | 不同版本 | 转化率 | 快速找到最优版本 |
| **网络路由** | 不同路径 | 延迟 | 最小化延迟 |

### 5.2 适用数据特征

- 奖励是随机的（不确定性）
- 需要在线学习（不能离线分析）
- 样本昂贵（每次反馈有限）
- 分布可能随时间变化（非平稳性）

### 5.3 不适用场景

- 离线评估可行（可以用监督学习）
- 奖励确定无噪声
- 样本充足（可以充分探索）
- 系统不允许探索（零容忍）

---

## 6. 优缺点分析

### 6.1 优点

| 算法 | 优点 | 说明 |
|------|------|------|
| ε-Greedy | 简单易实现 | 只需维护Q值 |
| ε-Greedy | 理论保证 | 简单但有效 |
| UCB1 | 有PAC保证 | 可识别最优臂 |
| UCB1 | 无参数需要调节 | 自适应探索 |
| Thompson Sampling | 概率最优 | 接近下界 |
| Thompson Sampling | 适合高方差 | 后验处理不确定性 |

### 6.2 缺点

| 算法 | 缺点 | 缓解方法 |
|------|------|----------|
| ε-Greedy | 需要调节ε | 衰减ε |
| ε-Greedy | 探索随机 | 利用确定性 |
| UCB1 | 对稀疏奖励敏感 | 使用KL-UCB |
| Thompson Sampling | 需要分布假设 | 扩展到其他分布 |

### 6.3 算法对比

| 属性 | ε-Greedy | UCB1 | Thompson Sampling |
|------|----------|------|-------------------|
| 累积regret | $O(\sqrt{T})$ | $O(\ln T)$ | $O(\ln T)$ |
| 实现难度 | 易 | 中 | 中 |
| 理论基础 | 弱 | 强 | 强 |
| 适用场景 | 初学者 | 一般 | 高性能 |

---

## 7. 调库实现

### 7.1 环境准备

```bash
pip install numpy scipy matplotlib
```

### 7.2 使用BanditLib（如果可用）

```python
# 注意：以下使用模拟的调库实现
# 实际中可以使用专门的库如 gumrocket, pybandit 等

import numpy as np
from scipy.stats import beta as beta_dist

class MultiArmedBandit:
    """多臂老虎机通用接口"""
    
    def __init__(self, n_arms, algorithm='ucb1'):
        self.n_arms = n_arms
        self.algorithm = algorithm
        
        if algorithm == 'ucb1':
            self.agent = UCB1(n_arms)
        elif algorithm == 'thompson':
            self.agent = ThompsonSampling(n_arms)
        else:
            self.agent = EpsilonGreedy(n_arms)
    
    def select_action(self):
        """选择臂"""
        return self.agent.select_arm()
    
    def update(self, arm, reward):
        """更新"""
        self.agent.update(arm, reward)
    
    def run(self, env, n_steps):
        """运行实验"""
        rewards = []
        for t in range(n_steps):
            arm = self.select_action()
            reward = env.step(arm)
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

# 示例使用
if __name__ == "__main__":
    # 创建环境
    probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    env = BanditEnvironment(probs)
    
    # 运行算法
    for algo_name in ['epsilon_greedy', 'ucb1', 'thompson']:
        agent = MultiArmedBandit(5, algorithm=algo_name)
        rewards = agent.run(env, n_steps=1000)
        
        print(f"{algo_name}: 总奖励={sum(rewards)}, 平均={np.mean(rewards):.3f}")
```

---

## 8. 手工代码实现

### 8.1 完整实现

```python
import numpy as np
import math
from scipy.stats import beta as beta_dist

class BanditAlgorithms:
    """完整的多臂老虎机算法实现"""
    
    @staticmethod
    def epsilon_greedy(probs, epsilon, n_steps, seed=42):
        """ε-Greedy算法"""
        np.random.seed(seed)
        n_arms = len(probs)
        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)
        total_reward = 0
        
        for t in range(n_steps):
            if np.random.random() < epsilon:
                arm = np.random.randint(n_arms)
            else:
                arm = np.argmax(values)
            
            reward = 1 if np.random.random() < probs[arm] else 0
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]
            total_reward += reward
        
        return total_reward
    
    @staticmethod
    def ucb1(probs, c, n_steps, seed=42):
        """UCB1算法"""
        np.random.seed(seed)
        n_arms = len(probs)
        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)
        total_reward = 0
        
        for t in range(n_arms):
            reward = 1 if np.random.random() < probs[t] else 0
            counts[t] = 1
            values[t] = reward
            total_reward += reward
        
        for t in range(n_arms, n_steps):
           .ucb_values = values + c * np.sqrt(
                2 * math.log(t) / counts
            )
            arm = np.argmax(ucb_values)
            
            reward = 1 if np.random.random() < probs[arm] else 0
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]
            total_reward += reward
        
        return total_reward
    
    @staticmethod
    def thompson_sampling(probs, n_steps, seed=42):
        """Thompson Sampling算法"""
        np.random.seed(seed)
        n_arms = len(probs)
        alpha = np.ones(n_arms)
        beta_params = np.ones(n_arms)
        total_reward = 0
        
        for t in range(n_steps):
            samples = [beta_dist.rvs(alpha[i], beta_params[i]) 
                      for i in range(n_arms)]
            arm = np.argmax(samples)
            
            reward = 1 if np.random.random() < probs[arm] else 0
            if reward > 0:
                alpha[arm] += 1
            else:
                beta_params[arm] += 1
            total_reward += reward
        
        return total_reward

# 测试
if __name__ == "__main__":
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25, 0.35]
    n_steps = 1000
    
    for algo in ['epsilon_greedy', 'ucb1', 'thompson_sampling']:
        reward = getattr(BanditAlgorithms, algo)(probs, 0.1, n_steps)
        print(f"{algo}: 总奖励 = {reward}")
```

### 8.2 与理论最优对比

```python
def compare_with_optimal():
    """与理论最优对比"""
    probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_steps = 1000
    n_runs = 100
    
    optimal_reward = n_steps * max(probs)
    
    results = {
        'epsilon_greedy': [],
        'ucb1': [],
        'thompson_sampling': []
    }
    
    for run in range(n_runs):
        for algo in results.keys():
            if algo == 'epsilon_greedy':
                reward = BanditAlgorithms.epsilon_greedy(
                    probs, 0.1, n_steps, seed=run
                )
            elif algo == 'ucb1':
                reward = BanditAlgorithms.ucb1(
                    probs, 2.0, n_steps, seed=run
                )
            else:
                reward = BanditAlgorithms.thompson_sampling(
                    probs, n_steps, seed=run
                )
            results[algo].append(optimal_reward - reward)
    
    print("平均regret对比：")
    for algo, regrets in results.items():
        print(f"  {algo}: {np.mean(regrets):.2f}")
```

---

## 9. 可视化与结果理解

### 9.1 臂价值估计的收敛

```python
def plot_value_convergence():
    """可视化价值估计的收敛过程"""
    np.random.seed(42)
    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_steps = 5000
    
    eg = EpsilonGreedy(len(probs), epsilon=0.1)
    ucb = UCB1(len(probs), c=2.0)
    ts = ThompsonSampling(len(probs))
    
    for algo in [eg, ucb, ts]:
        algo.values_history = []
        for t in range(n_steps):
            arm = algo.select_arm()
            reward = 1 if np.random.random() < probs[arm] else 0
            algo.update(arm, reward)
            algo.values_history.append(algo.values.copy())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, algo, name in zip(
        axes, [eg, ucb, ts],
        ['ε-Greedy', 'UCB1', 'Thompson Sampling']
    ):
        values = np.array(algo.values_history)
        for i in range(len(probs)):
            ax.plot(values[:, i], label=f'Arm {i}', alpha=0.5)
        ax.axhline(y=probs[0], color='blue', linestyle='--', alpha=0.3)
        ax.axhline(y=probs[4], color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Estimated Value')
        ax.set_title(name)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('value_convergence.png')
    plt.show()
```

### 9.2 探索-利用可视化

```python
def plot_exploration_exploitation():
    """可视化探索-利用过程"""
    np.random.seed(42)
    probs = [0.15, 0.25, 0.35, 0.45, 0.55]
    n_steps = 2000
    
    ucb = UCB1(len(probs), c=2.0)
    
    # 记录每次选择的臂和真实概率
    choices = []
    for t in range(n_steps):
        arm = ucb.select_arm()
        reward = 1 if np.random.random() < probs[arm] else 0
        ucb.update(arm, reward)
        choices.append((arm, probs[arm]))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 臂的选择分布
    arm_counts = np.bincount([c[0] for c in choices], minlength=len(probs))
    axes[0].bar(range(len(probs)), arm_counts)
    axes[0].set_xlabel('Arm')
    axes[0].set_ylabel('Times Chosen')
    arms[0].set_title('Arm Selection Distribution')
    
    # 真实概率 vs 选择次数
    axes[1].scatter(probs, arm_counts)
    for i, (p, c) in enumerate(zip(probs, arm_counts)):
        axes[1].annotate(f'Arm {i}', (p, c))
    axes[1].set_xlabel('True Probability')
    axes[1].set_ylabel('Times Chosen')
    axes[1].set_title('Selection vs Probability')
    
    plt.tight_layout()
    plt.savefig('exploration_exploitation.png')
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 累积regret | $R_T = T\mu^* - \sum_t r_t$ | 核心指标 |
| 平均奖励 | $\frac{1}{T}\sum_t r_t$ | 越高越好 |
| 最优选择率 | $\frac{1}{T}\sum_t \mathbb{1}\{a_t=a^*\}$ | 收敛到1 |
| 公平性 | $\sum_a \frac{|n_a - n_b|}{\max n_a - \min n_a}$ | 探索是否均匀 |

### 10.2 交叉验证

```python
def cross_validate():
    """K折交叉验证"""
    probs_sets = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.15, 0.25, 0.35, 0.45, 0.55],
        [0.2, 0.3, 0.4, 0.5, 0.6],
    ]
    
    results = {algo: [] for algo in ['eg', 'ucb', 'ts']}
    
    for probs in probs_sets:
        for algo in results.keys():
            runs = 50
            rewards = []
            for _ in range(runs):
                if algo == 'eg':
                    r = BanditAlgorithms.epsilon_greedy(
                        probs, 0.1, 1000
                    )
                elif algo == 'ucb':
                    r = BanditAlgorithms.ucb1(
                        probs, 2.0, 1000
                    )
                else:
                    r = BanditAlgorithms.thompson_sampling(
                        probs, 1000
                    )
                rewards.append(r / 1000)
            results[algo].append(np.mean(rewards))
    
    print("各算法平均奖励率：")
    for algo, vals in results.items():
        print(f"  {algo}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
```

---

## 11. 常见问题与易错点

### 11.1 问题1：ε设置不当

**表现**：regret持续线性增长

**原因**：ε太大导致过度探索，或太小导致陷入次优

**解决方案**：使用衰减ε
```python
class EpsilonDecay:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.01, decay_steps=1000):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.step = 0
    
    def get_epsilon(self):
        t = min(self.step, self.decay_steps)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  (1 - t / self.decay_steps)
        self.step += 1
        return epsilon
```

### 11.2 问题2：UCB的数值不稳定

**原因**：$\ln t$在早期增长太快

**解决方案**：使用修正的UCB
```python
def ucb_safe(t, n_a, value, c=2.0):
    if n_a == 0:
        return float('inf')
    return value + c * math.sqrt(math.log(t + 1) / n_a)
```

### 11.3 问题3：非平稳环境

**原因**：奖励分布随时间变化

**解决方案**：使用折扣Thompson Sampling
```python
class DiscountedThompsonSampling:
    def __init__(self, n_arms, gamma=0.95):
        self.n_arms = n_arms
        self.gamma = gamma
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
    
    def update(self, arm, reward):
        self.alpha *= self.gamma
        self.beta *= self.gamma
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

---

## 12. 学习总结

### 12.1 核心要点

1. **探索-利用权衡**是MAB的核心挑战
2. **累计regret**是核心评估指标
3. 三种经典算法：ε-Greedy（简单）、UCB1（理论保证）、Thompson Sampling（接近最优）
4. 无免费午餐：没有绝对最优的算法，只有适合场景的算法

### 12.2 关键公式

- **UCB1**: $UCB_a = \hat{\mu}_a + \sqrt{2\ln t / n_a}$
- **Thompson Sampling**: $\mu_a \sim \text{Beta}(\alpha_a, \beta_a)$
- **累积regret上界**: $O(K\ln T)$（对于UCB和TS）

### 12.3 学习路径

```
基础（了解概念）
   ↓
ε-Greedy（入门）
   ↓
UCB1（理解置信界）
   ↓
Thompson Sampling（贝叶斯角度）
   ↓
Contextual Bandits（扩展到上下文）
   ↓
Dueling Bandits（比较臂）
```

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1：计算UCB值**
> 给定Q=[0.5, 0.6, 0.7], n=[10, 5, 2], t=100, 计算各个臂的UCB值（c=2）。

<details>
<summary>答案</summary>

UCB公式：$UCB_a = Q_a + c\sqrt{2\ln t / n_a}$

$UCB_1 = 0.5 + 2\sqrt{2\ln 100 / 10} = 0.5 + 2\sqrt{4.605/10} = 0.5 + 2\sqrt{0.4605} = 0.5 + 2 * 0.6788 = 1.8576$

$UCB_2 = 0.6 + 2\sqrt{4.605/5} = 0.6 + 2\sqrt{0.921} = 0.6 + 2 * 0.9599 = 2.5198$

$UCB_3 = 0.7 + 2\sqrt{4.605/2} = 0.7 + 2\sqrt{2.3025} = 0.7 + 2 * 1.5174 = 3.7348$

因此选择Arm 3。

</details>

**练习2：实现衰减ε-Greedy**
> 实现ε从1.0衰减到0.01的ε-Greedy变体。

<details>
<summary>答案</summary>

```python
class EpsilonDecayGreedy:
    def __init__(self, n_arms, epsilon_start=1.0, epsilon_end=0.01, decay_steps=1000):
        self.n_arms = n_arms
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.step = 0
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def get_epsilon(self):
        if self.step >= self.decay_steps:
            return self.epsilon_end
        ratio = self.step / self.decay_steps
        return self.epsilon_start * (1 - ratio) + self.epsilon_end * ratio
    
    def select_arm(self):
        epsilon = self.get_epsilon()
        if np.random.random() < epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)
    
    def update(self, arm, reward):
        self.step += 1
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
```
</details>

### 13.2 进阶思考

**思考题：Contextual Bandits**
> 如果每个臂的奖励不仅依赖于选择，还依赖于上下文（如用户特征），如何修改算法？

<details>
<summary>提示和答案</summary>

Contextual Bandits（ contextual bandits）的扩展方向：

1. **线性模型**：假设$\mu_a(x) = x^T \theta_a$，使用LinUCB
2. **非线性模型**：使用神经网络估计上下文-奖励关系
3. **元学习**：学习不同上下文之间的共享表示

LinUCB的更新：
$$\theta_a = (X_a^TX_a + \lambda I)^{-1} X_a^T y_a$$

核心思想是将上下文信息加入到臂的价值估计中。

</details>

---

## 14. 学习路径建议

### 14.1 第一阶段（1-2周）

- 理解MAB问题形式化
- 实现ε-Greedy
- 理解探索-利用困境

### 14.2 第二阶段（2-3周）

- 实现UCB1
- 推导regret上界
- 实现Thompson Sampling

### 14.3 第三阶段（3-4周）

- 扩展到Contextual Bandits
- 实现LinUCB
- 项目实践（广告投放模拟）

### 14.4 实践项目

1. **广告投放模拟**：模拟真实广告系统
2. **推荐系统**：电影推荐场景
3. **A/B测试优化**：在线实验设计

### 14.5 推荐资源

- **论文**：Lai & Robbins (1985), "Asymptotically efficient adaptive allocation rules"
- **书籍**："Bandit Algorithms" by Tor Lattimore and Csaba Szepesvari
- **课程**：UCL RL Course by David Silver

---

**文档结束**

*参考：Sutton & Barto (2018), "Reinforcement Learning: An Introduction"*