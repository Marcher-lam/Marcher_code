# Multi-Armed Bandits 学习文档

> Multi-Armed Bandits（多臂老虎机）是强化学习中一类经典的序贯决策问题，完美体现了"探索-利用权衡"（Exploration-Exploitation Trade-off）的核心挑战，是理解更复杂强化学习算法的基础

---

## 1. 算法基础认知

### 1.1 一句话定义

**Multi-Armed Bandits（多臂老虎机，MAB）** 是序贯决策理论中的基础框架，智能体在每个离散时间步从K个可选动作（"臂"）中选择一个，根据该臂对应的随机奖励分布获得即时奖励，目标是最大化T步累计奖励。这一框架形式化了一个根本性困境：**应该继续选择已知的好选项，还是尝试新的可能性？**

### 1.2 经典类比

想象你进入一个有K台不同老虎机的赌场：

| 策略 | 描述 | 风险 |
|------|------|------|
| 总是投币到之前赢过的那台 | 利用已知信息 | 可能永远不知道更好的机器 |
| 每台机器都轮流尝试 | 探索新可能性 | 浪费在差的机器上 |
| 平衡探索与利用 | 智能分配尝试 | 找到最优但需要时间 |

这就是MAB研究的核心：**探索-利用困境**（Exploration-Exploitation Dilemma）

### 1.3 历史背景

| 年份 | 重要进展 |
|------|----------|
| 1933 | Thompson提出最早的序贯分配方法 |
| 1952 | Robbins证明问题可追溯到最优Stopping |
| 1979 | Lai & Robbins建立信息论下界 |
| 1985 | Agrawal提出UCB1算法 |
| 1998 | Auer证明ε-Greedy的界限 |
| 2011 | Chapelle & Li给出TS的现代分析 |

### 1.4 问题定位

| 维度 | 描述 |
|------|------|
| 学习范式 | 在线学习/序贯决策 |
| 状态空间 | 单状态（ stateless） |
| 动作空间 | K个离散动作 |
| 奖励模型 | 随机，服从各臂的独立分布 |
| 优化目标 | 最小化累计遗憾 |

### 1.5 前置知识

- 基础概率论（期望、方差）
- 统计学基础（点估计、置信区间）
- Python编程（NumPy实现）

---

## 2. 核心原理

### 2.1 问题形式化

**K臂老虎机**数学模型：

- **动作集**：$\mathcal{A} = \{1, 2, \ldots, K\}$
- **奖励分布**：$r_t \sim P(\cdot|a_t)$，假设各臂独立
- **期望奖励**：$\mu_a = \mathbb{E}[r|a]$，真实但未知
- **最优动作**：$a^* = \arg\max_a \mu_a$

**累计遗憾**（Cumulative Regret）：
$$R_T = T\mu^* - \sum_{t=1}^{T} r_t = sum_{t=1}^{T} (\mu^* - \mu_{a_t})$$

其中$\mu^* = \max_a \mu_a$是最优臂期望奖励。

### 2.2 核心算法

#### 2.2.1 ε-Greedy（ε-贪心）

以概率$1-\varepsilon$利用（选最好），以概率$\varepsilon$探索（随机选）：

```python
import numpy as np

class EpsilonGreedy:
    """ε-Greedy 多臂老虎机算法"""
    
    def __init__(self, n_arms, epsilon=0.1, seed=42):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)      # 每个臂被选中次数
        self.values = np.zeros(n_arms)   # 每个臂的估计价值
        self.total_reward = 0
        np.random.seed(seed)
    
    def select_arm(self):
        """基于ε-Greedy策略选择臂"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)
    
    def update(self, arm, reward):
        """增量更新价值估计"""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.total_reward += reward
```

#### 2.2.2 UCB1（上置信界算法）

使用**上置信界**平衡探索利用：
$$UCB_a = hat{mu}_a + sqrt{frac{2 ln t}{n_a}}$$

第一项鼓励利用，第二项鼓励探索：

```python
import math

class UCB1:
    """UCB1 (Upper Confidence Bound) 算法"""
    
    def __init__(self, n_arms, c=2.0, seed=42):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0
        self.total_reward = 0
        np.random.seed(seed)
    
    def select_arm(self):
        """选择UCB值最大的臂"""
        self.total_counts += 1
        
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        ucb = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            exploitation = self.values[arm]
            exploration = self.c * math.sqrt(
                2 * math.log(self.total_counts) / self.counts[arm]
            )
            ucb[arm] = exploitation + exploration
        
        return np.argmax(ucb)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.total_reward += reward
```

#### 2.2.3 Thompson Sampling（汤普森采样）

使用贝叶斯后验采样，假设伯努利奖励：

```python
from scipy.stats import beta as beta_dist

class ThompsonSampling:
    """Thompson Sampling (伯努利老虎机)"""
    
    def __init__(self, n_arms, alpha_prior=1, beta_prior=1, seed=42):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * alpha_prior
        self.beta = np.ones(n_arms) * beta_prior
        self.total_reward = 0
        np.random.seed(seed)
    
    def select_arm(self):
        """从Beta后验采样并选择"""
        samples = [beta_dist.rvs(self.alpha[i], self.beta[i]) 
                  for i in range(self.n_arms)]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        self.total_reward += reward
```

### 2.3 算法流程图

```
初始化: 
  - counts[arm] = 0
  - values[arm] = 0

for t = 1 to T:
  | 策略选择臂
  |   ├─ ε-Greedy: 随机 (prob=ε) 或 最优 (prob=1-ε)
  |   ├─ UCB: argmax(Q_a + c√(2lnt/n_a))
  |   └─ TS: 采样后验，选择采样值最大的
  |
  | 获得奖励 r ~ P(.|arm)
  |
  | 更新估计
  |   Q(a) ← Q(a) + (r-Q(a))/n_a
  |
  | 记录: 累积奖励，regret
```

---

## 3. 数学公式与推导

### 3.1 符号表

| 符号 | 含义 | 类型 |
|------|------|------|
| $K$ | 臂数量 | 标量 |
| $T$ | 总时间步 | 标量 |
| $a_t$ | 第t步选择的臂 | 标量 |
| $r_t$ | 第t步获得的奖励 | 标量 |
| $n_a$ | 臂a被选次数 | 标量 |
| $\mu_a$ | 臂a真实期望 | 标量 |
| $\hat{\mu}_a$ | 臂a估计期望 | 标量 |
| $\Delta_a$ | $\mu^* - \mu_a$ | 标量 |
| $R_T$ | T步累积regret | 标量 |

### 3.2 Regret上界

#### ε-Greedy:
$$mathbb{E}[R_T] leq varepsilon T max_a Delta_a + sum_a frac{ln T}{Delta_a}$$

#### UCB1:
$$mathbb{E}[R_T] leq sum_a frac{8 ln T}{Delta_a} + K$$

#### Thompson Sampling:
$$mathbb{E}[R_T] leq sum_a frac{2 ln T}{Delta_a} + K$$

### 3.3 Beta-Bernoulli共轭

假设奖励$r sim Bernoulli(mu_a)$，使用Beta先验：

后验概率：
$$P(mu_a | D) = Beta(alpha_a + S_a, beta_a + F_a)$$

其中$S_a$, $F_a$是成功和失败次数。

采样期望：
$$mathbb{E}[mu_a | D] = frac{alpha_a}{alpha_a + beta_a}$$

### 3.4 下界

Lai & Robbins (1985) 证明任何算法满足：
$$liminf_{T oinfty} frac{mathbb{E}[R_T]}{ln T} geq sum_a KL(P_a || P^*)$$

这是**最优可达下界**，是算法比较的理论基准。

### 3.5 KL-UCB

更紧的界使用KL散度：
$$UCB_a^{KL} = sup { mu : KL(P_a || P_{hat{mu}_a}) leq frac{ln t}{n_a} }$$

---

## 4. 训练过程讲解

### 4.1 环境类

```python
class BanditEnv:
    """老虎机环境"""
    
    def __init__(self, probs, seed=42):
        self.probs = np.array(probs)
        self.n_arms = len(probs)
        self.best_arm = np.argmax(probs)
        self.best_prob = max(probs)
        np.random.seed(seed)
    
    def step(self, arm):
        """执行动作，获得伯努利奖励"""
        return 1 if np.random.random() < self.probs[arm] else 0
    
    def optimal_reward(self, T):
        """理论最优累计奖励"""
        return T * self.best_prob
```

### 4.2 训练循环

```python
def train_agent(agent, env, n_steps, verbose=True):
    """训练agent并记录曲线"""
    regrets = []
    rewards = []
    optimal_counts = []
    optimal = env.best_arm
    
    for t in range(n_steps):
        arm = agent.select_arm()
        reward = env.step(arm)
        agent.update(arm, reward)
        
        rewards.append(reward)
        regrets.append(
            regrets[-1] + (env.best_prob - env.probs[arm])
            if regrets else env.best_prob - env.probs[arm]
        )
        optimal_counts.append(
            optimal_counts[-1] + (arm == optimal)
            if optimal_counts else (arm == optimal)
        )
    
    if verbose:
        print(f"总奖励: {sum(rewards)}")
        print(f"累积regret: {regrets[-1]:.2f}")
        print(f"最优选择率: {optimal_counts[-1]/n_steps:.2%}")
    
    return {
        'rewards': rewards,
        'regrets': regrets,
        'optimal_counts': optimal_counts
    }
```

### 4.3 参数调优

```python
def tune_epsilon():
    """调优ε-Greedy的ε值"""
    probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_steps = 1000
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    results = {}
    for eps in epsilons:
        total_rewards = []
        for run in range(50):
            env = BanditEnv(probs, seed=run)
            agent = EpsilonGreedy(5, epsilon=eps, seed=run)
            train_agent(agent, env, n_steps, verbose=False)
            total_rewards.append(agent.total_reward)
        results[eps] = np.mean(total_rewards)
    
    best_eps = max(results, key=results.get)
    print(f"最佳ε: {best_eps}")
    return best_eps, results
```

---

## 5. 应用场景

### 5.1 典型工业应用

| 场景 | 臂定义 | 奖励 | 业务目标 |
|------|--------|------|---------|
| **在线广告** | 广告创意 | 点击 | CPM最大化 |
| **推荐系统** | 物品 | 点击/购买 | 转化率 |
| **临床试验** | 药物 | 治愈 | 有效性+安全 |
| **A/B测试** | 版本 | 转化 | 快速收敛 |
| **网络路由** | 路径 | 延迟 | 低延迟 |

### 5.2 新兴应用

| 领域 | 应用 |
|------|------|
| 大模型 | Prompt选择 |
| 对话系统 | 响应策略 |
| 自动驾驶 | 动作规划 |
| 机器人 | 技能选择 |

### 5.3 适用条件

- 在线学习（无法离线批处理）
- 样本昂贵（探索成本高）
- 奖励随机（有噪声）
- 分布可能变化（非平稳）

---

## 6. 优缺点分析

### 6.1 算法对比

| 算法 | Regret | 实现 | 调参 | 适用场景 |
|------|-------|------|------|----------|
| ε-Greedy | $O(sqrt{T})$ | 最简 | ε | 初学/基线 |
| UCB1 | $O(log T)$ | 易 | c | 一般场景 |
| TS | $O(log T)$ | 中 | 无 | 高性能 |

### 6.2 优缺点对比

| 算法 | 优点 | 缺点 |
|------|------|------|
| ε-Greedy | 简单稳定 | 渐近regret差 |
| UCB1 | 有PAC保证 | 对稀疏敏感 |
| TS | 接近下界 | 需要分布假设 |

---

## 7. 调库实现（Python）

### 7.1 环境配置

```bash
pip install numpy scipy matplotlib
```

### 7.2 库使用示例

```python
import numpy as np

class BanditFramework:
    """统一的多臂老虎机框架"""
    
    def __init__(self, n_arms, algo='ucb1', **kwargs):
        self.n_arms = n_arms
        self.algo_name = algo
        
        if algo == 'ucb1':
            self.agent = UCB1(n_arms, c=kwargs.get('c', 2.0))
        elif algo == 'ts':
            self.agent = ThompsonSampling(n_arms)
        else:
            self.agent = EpsilonGreedy(
                n_arms, epsilon=kwargs.get('epsilon', 0.1)
            )
    
    def run(self, env, n_steps):
        """运行实验"""
        return train_agent(self.agent, env, n_steps, verbose=False)

# 使用示例
if __name__ == "__main__":
    probs = [0.15, 0.25, 0.35, 0.45, 0.55, 0.2]
    env = BanditEnv(probs)
    
    for algo in ['eg', 'ucb1', 'ts']:
        framework = BanditFramework(6, algo)
        result = framework.run(env, 1000)
        print(f"{algo}: {result['regrets'][-1]:.2f}")
```

---

## 8. 手工代码实现

### 8.1 完整实现

```python
import numpy as np
import math
from scipy.stats import beta

class BanditAlgorithms:
    """完整的多臂老虎机算法实现"""
    
    @staticmethod
    def epsilon_greedy(probs, epsilon, n_steps, seed=42):
        np.random.seed(seed)
        n = len(probs)
        counts = np.zeros(n)
        values = np.zeros(n)
        
        for _ in range(n_steps):
            if np.random.random() < epsilon:
                arm = np.random.randint(n)
            else:
                arm = np.argmax(values)
            reward = 1 if np.random.random() < probs[arm] else 0
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]
        
        return sum(probs) * n_steps - np.sum(counts * (max(probs) - probs))
    
    @staticmethod
    def ucb1(probs, c, n_steps, seed=42):
        np.random.seed(seed)
        n = len(probs)
        counts = np.zeros(n)
        values = np.zeros(n)
        
        for t in range(n):
            reward = 1 if np.random.random() < probs[t] else 0
            counts[t] = 1
            values[t] = reward
        
        for t in range(n, n_steps):
            ucb = values + c * np.sqrt(2 * np.log(t) / counts)
            arm = np.argmax(ucb)
            reward = 1 if np.random.random() < probs[arm] else 0
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]
        
        return sum(probs) * n_steps - np.sum(counts * (max(probs) - probs))
    
    @staticmethod
    def thompson_sampling(probs, n_steps, seed=42):
        np.random.seed(seed)
        n = len(probs)
        alpha = np.ones(n)
        beta_params = np.ones(n)
        
        for _ in range(n_steps):
            samples = [beta.rvs(alpha[i], beta_params[i]) for i in range(n)]
            arm = np.argmax(samples)
            reward = 1 if np.random.random() < probs[arm] else 0
            if reward > 0:
                alpha[arm] += 1
            else:
                beta_params[arm] += 1
        
        return sum(probs) * n_steps - (alpha + beta_params - 2).dot(probs)

# 测试
if __name__ == "__main__":
    probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("Regret对比：")
    for algo in ['epsilon_greedy', 'ucb1', 'thompson_sampling']:
        r = getattr(BanditAlgorithms, algo)(probs, 0.1, 1000)
        print(f"  {algo}: {r:.2f}")
```

---

## 9. 可视化与结果理解

### 9.1 绘制Regret曲线

```python
import matplotlib.pyplot as plt

def plot_regret_curves():
    """绘制累积regret比较图"""
    np.random.seed(42)
    probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_steps = 2000
    n_runs = 30
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    algos = [
        ('Epsilon-Greedy', 'epsilon_greedy', 'blue'),
        ('UCB1', 'ucb1', 'green'),
        ('Thompson Sampling', 'thompson_sampling', 'red')
    ]
    
    for ax, (name, algo, color) in zip(axes, algos):
        regrets = []
        optimal = max(probs)
        
        for run in range(n_runs):
            np.random.seed(run)
            r = getattr(BanditAlgorithms, algo)(probs, 0.1, n_steps)
            regrets.append(r)
        
        ax.plot(regrets, color=color, alpha=0.7)
        ax.axhline(y=np.mean(regrets), color=color, linestyle='--')
        ax.set_title(name)
        ax.set_xlabel('Run')
        ax.set_ylabel('Regret')
    
    plt.tight_layout()
    plt.savefig('regret_comparison.png', dpi=150)
    plt.show()
```

### 9.2 价值收敛

```python
def plot_value_convergence():
    """可视化价值估计的收敛"""
    np.random.seed(42)
    probs = [0.15, 0.3, 0.45, 0.6, 0.75]
    n_steps = 3000
    
    agent = UCB1(5)
    counts_history = []
    values_history = []
    
    for t in range(n_steps):
        arm = agent.select_arm()
        reward = 1 if np.random.random() < probs[arm] else 0
        agent.update(arm, reward)
        counts_history.append(agent.counts.copy())
        values_history.append(agent.values.copy())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    counts = np.array(counts_history)
    for i in range(5):
        axes[0].plot(counts[:, i], label=f'Arm {i}')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Selection Count')
    axes[0].set_title('Arm Selection Count over Time')
    axes[0].legend()
    
    values = np.array(values_history)
    for i in range(5):
        axes[1].axhline(y=probs[i], linestyle='--', alpha=0.3)
        axes[1].plot(values[:, i], label=f'Arm {i}')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Estimated Value')
    axes[1].set_title('Value Estimation Convergence')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 公式 | 意义 |
|------|------|------|
| 累积regret | $R_T$ | 越小越好 |
| 平均奖励 | $mean(r_t)$ | 越大越好 |
| 最优率 | $frac{1}{T} sum mathbb{1}(a_t=a^*)$ | 越大越好 |
| 收敛速度 | 达到稳定所需步数 | 越快越好 |

### 10.2 统计评估

```python
def statistical_evaluation():
    """统计评估"""
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.15]
    n_steps = 1000
    n_runs = 50
    
    results = {a: [] for a in ['eg', 'ucb', 'ts']}
    
    for run in range(n_runs):
        for algo in results.keys():
            if algo == 'eg':
                r = BanditAlgorithms.epsilon_greedy(probs, 0.1, n_steps, seed=run)
            elif algo == 'ucb':
                r = BanditAlgorithms.ucb1(probs, 2.0, n_steps, seed=run)
            else:
                r = BanditAlgorithms.thompson_sampling(probs, n_steps, seed=run)
            results[algo].append(r)
    
    print("Regret统计:")
    for algo, vals in results.items():
        print(f"  {algo}: mean={np.mean(vals):.1f}, std={np.std(vals):.1f}")
```

---

## 11. 常见问题与易错点

### 11.1 问题：ε设置

**症状**：regret线性增长

**原因**：ε太大（过度探索）或太小（欠探索）

**解**：使用衰减ε
```python
def get_epsilon(t, decay_steps=1000):
    return max(0.01, 1 - t / decay_steps)
```

### 11.2 问题：UCB数值

**症状**：早期不稳定

**原因**：ln t增长太快

**解**：使用ln(t+1)

### 11.3 问题：非平稳

**症状**：无法适应变化

**解**：使用滑动窗口或折扣
```python
class SlidingWindowTS:
    def __init__(self, n_arms, window=100):
        self.window = window
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
    
    def update(self, arm, reward):
        self.alpha *= 0.99
        self.beta *= 0.99
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

---

## 12. 学习总结

### 12.1 核心要点

1. **探索-利用权衡**是核心挑战
2. **累积regret**是核心指标
3. ε-Greedy（简单）、UCB1（理论保证）、TS（接近最优）
4. 没有通用最优，只有适合场景的选择

### 12.2 关键公式

- UCB: $\hat{\mu}_a + \sqrt{2\ln t / n_a}$
- TS: $\mu_a \sim Beta(\alpha_a, \beta_a)$
- Regret上界: $O(\ln T)$

### 12.3 进阶方向

```
MAB → Contextual Bandits → LinUCB
  ↓
Dueling Bandits
  ↓
Combinatorial Bandits
  ↓
Non-stationary Bandits
```

---

## 13. 练习题

### 练习1

**问题**：推导UCB1在两臂情况下的regret上界

<details>
<summary>答案</summary>

对于两臂，设$\Delta = |\mu_1 - \mu_2|$，假设$\mu_1 > \mu_2$：

$$R_T leq frac{8 ln T}{Delta} + 2$$

证明思路：
- 错误臂被选中的次数上界由$8 ln T /Delta^2$决定
- 每次错误损失的regret为$\Delta$
- 加上常数项$K=2$
</details>

### 练习2

**问题**：实现衰减ε的ε-Greedy

<details>
<summary>答案</summary>

```python
class EpsilonDecayGreedy:
    def __init__(self, n_arms, eps_start=1.0, eps_end=0.01):
        self.n_arms = n_arms
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.t = 0
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def get_epsilon(self):
        ratio = min(1.0, self.t / 10000)
        return self.eps_end + (self.eps_start - self.eps_end) * (1 - ratio)
    
    def select_arm(self):
        if np.random.random() < self.get_epsilon():
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)
```
</details>

### ���习3

**问题**：比较三种算法在5个不同概率分布下的性能

<details>
<summary>提示</summary>

1. 定义5组不同的probs
2. 运行100次取平均
3. 绘制箱线图比较
</details>

---

## 14. 学习路径建议

### 第一阶段（1周）

- 理解MAB问题
- 实现ε-Greedy
- 运行实验观察

### 第二阶段（1周）

- 实现UCB1
- 理解置信界
- 对比regret曲线

### 第三阶段（1周）

- 实现Thompson Sampling
- 理解贝叶斯角度
- 扩展到Contextual Bandits

### 实践项目

1. 广告投放模拟系统
2. 推荐系统冷启动
3. A/B测试优化框架

### 推荐资源

- **书籍**："Bandit Algorithms" by Lattimore & Szepesvari
- **论文**：Lai & Robbins (1985), "Asymptotically Efficient Adaptive Allocation Rules"
- **课程**：UCL RL Course

---

**文档结束**

*参考文献：Sutton & Barto (2018); Lattimore & Szepesvari (2020)*