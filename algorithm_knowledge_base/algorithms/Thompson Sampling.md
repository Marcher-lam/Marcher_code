# Thompson Sampling 学习文档

> 贝叶斯方法解决多臂老虎机问题的经典算法

---

## 1. 算法基础认知

### 1.1 一句话定义

Thompson Sampling（汤普森采样）是一种基于贝叶斯推断的多臂老虎机（Multi-Armed Bandit）算法，通过维护每个臂的回报概率分布并进行后验采样来平衡探索与利用，是解决探索-利用权衡问题的经典方法。

### 1.2 直觉类比

想象你在一家赌场面对K台老虎机，每台机器有不同的未知获胜概率：
- 传统贪心：只玩目前胜率最高的
- ε-greedy：偶尔随机探索
- Thompson Sampling：想象你有一个"信念球"，根据当前信念随机选择

就像玩德州扑克时，你会根据对手的下注模式（后验分布）来估计他们的手牌范围，然后做出最优决策。

### 1.3 发展背景

- 1933年：Thompson提出原始算法
- 2010年代：深度学习时代的重要应用
- 2018年后：与强化学习结合（Meta-learning等）

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 在线学习/决策算法 |
| 问题 | 多臂老虎机（MAB） |
| 核心 | 贝叶斯后验采样 |
| 目标 | 最小化累积 regret |

---

## 2. 核心原理

### 2.1 问题定义

多臂老虎机：有K个臂，每个臂i有未知获胜概率θi。目标是在T轮内最大化累积奖励，或最小化regret：

$$Regret(T) = \sum_{t=1}^{T} (\theta^* - theta_{a_t})$$

其中θ*是最佳臂的真实获胜概率。

### 2.2 Thompson Sampling 算法

```
对于每轮 t=1,2,...,T:
    1. 对每个臂i，从后验分布 P(θi | 数据i) 采样 θi~
    2. 选择采样值最大的臂: a_t = argmax_i θi~
    3. 获得奖励 r_t ∈ {0,1}
    4. 更新臂i的后验分布
```

### 2.3 后验分布

对于伯努利老虎机（奖励为0/1），使用Beta-共轭分布：

- 先验：Beta(α, β)
- 观察：成功s次，失败f次
- 后验：Beta(α+s, β+f)

采样公式：
$$\theta \sim Beta(\alpha_i + successes_i, beta_i + failures_i)$$

### 2.4 vs 其他算法对比

| 算法 | 策略 | 特点 |
|------|------|------|
| Epsilon-Greedy | ε概率随机 | 简单，但非最优 |
| UCB | 置信上界 | 确定性 |
| **Thompson Sampling** | **贝叶斯采样** | **自适应探索** |
| LinUCB | 线性UCB | 适合上下文 |

---

## 3. 数学公式与推导

### 3.1 Beta分布

Beta分布的概率密度函数：

$$f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, beta)}$$

其中B是Beta函数：

$$B(\alpha, beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$$

### 3.2 后验更新

设先验 Beta(α, β)，观察n次，其中成功s次，失败f次：

$$\alpha_{new} = \alpha + s$$
$$\beta_{new} = beta + f$$

后验预测分布：
$$P(x | data) = Beta(\alpha+s, \beta+f)$$

### 3.3 期望regret上界

对于K个臂的伯努利老虎机，Thompson Sampling的期望regret：

$$E[Regret(T)] \leq O(\sqrt{K \cdot T \cdot log T})$$

这与UCB的regret上界相同，实际中通常更好。

### 3.4 扩展：高斯老虎机

对于连续奖励，假设奖励~N(θ, σ²)，使用正态共轭：

- 先验：θ ~ N(μ0, σ0²)
- 后验：θ | data ~ N(μn, σn²)

其中：
$$\frac{1}{\sigma_n^2} = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}$$
$$\mu_n = \sigma_n^2 (\frac{\mu_0}{\sigma_0^2} + \frac{\sum r_i}{\sigma^2})$$

---

## 4. PyTorch实现

### 4.1 基础Thompson Sampling

```python
import torch
import numpy as np
from scipy.stats import beta

class ThompsonSampling:
    """Thompson Sampling for Bernoulli Bandit"""
    
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # 每臂的Beta先验参数 (alpha, beta)
        self.alpha = np.ones(n_arms)  # 成功计数 + 1
        self.beta = np.ones(n_arms)  # 失败计数 + 1
    
    def select_arm(self):
        """从后验分布采样并选择最大值的臂"""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        """更新后验分布"""
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def get_mean(self):
        """获取每臂的后验均值"""
        return self.alpha / (self.alpha + self.beta)


class ThompsonSamplingTorch:
    """PyTorch版本的Thompson Sampling"""
    
    def __init__(self, n_arms, device='cpu'):
        self.n_arms = n_arms
        self.device = device
        # Alpha: 成功次数 + 1, Beta: 失败次数 + 1
        self.alpha = torch.ones(n_arms, device=device)
        self.beta = torch.ones(n_arms, device=device)
    
    def select_arm(self):
        """采样并选择臂"""
        samples = torch.distributions.Beta(self.alpha, self.beta).sample()
        return torch.argmax(samples).item()
    
    def update(self, arm, reward):
        """更新分布"""
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def expected_values(self):
        """后验均值"""
        return self.alpha / (self.alpha + self.beta)
```

### 4.2 高斯Thompson Sampling

```python
class GaussianThompsonSampling:
    """高斯老虎机的Thompson Sampling"""
    
    def __init__(self, n_arms, prior_std=1.0, noise_std=1.0):
        self.n_arms = n_arms
        self.noise_std = noise_std
        
        # 先验：均值=0, 标准差=prior_std
        self.prior_mean = torch.zeros(n_arms)
        self.prior_precision = torch.ones(n_arms) / (prior_std ** 2)
        
        # 观测计数
        self.n = torch.zeros(n_arms)
        
        # 后验均值和精度
        self.posterior_mean = self.prior_mean.clone()
        self.posterior_precision = self.prior_precision.clone()
    
    def select_arm(self):
        """从后验采样并选择"""
        # 采样
        std = 1.0 / torch.sqrt(self.posterior_precision)
        samples = torch.normal(self.posterior_mean, std)
        return torch.argmax(samples).item()
    
    def update(self, arm, reward):
        """更新后验"""
        self.n[arm] += 1
        self.posterior_precision[arm] += 1.0 / (self.noise_std ** 2)
        
        # 精度加权平均更新均值
        old_precision = self.posterior_precision[arm] - 1.0 / (self.noise_std ** 2)
        self.posterior_mean[arm] = (
            old_precision * self.posterior_mean[arm] + 
            reward / (self.noise_std ** 2)
        ) / self.posterior_precision[arm]
```

### 4.3 上下文Thompson Sampling（LinTS）

```python
class LinearThompsonSampling:
    """线性上下文Thompson Sampling"""
    
    def __init__(self, n_arms, context_dim, prior_variance=1.0, noise_variance=0.1):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.noise_variance = noise_variance
        
        # 每臂的贝叶斯线性回归参数
        self.A = [
            torch.eye(context_dim) * 1.0 / prior_variance 
            for _ in range(n_arms)
        ]
        self.b = [
            torch.zeros(context_dim) 
            for _ in range(n_arms)
        ]
    
    def select_arm(self, context):
        """
        context: (n_arms, context_dim) 或 (context_dim,)
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        samples = []
        for arm in range(self.n_arms):
            # 后验采样
            A_inv = torch.inverse(self.A[arm])
            mean = A_inv @ self.b[arm]
            cov = A_inv * self.noise_variance
            
            # 采样权重
            weight = torch.multivariate_normal(mean, cov)
            # 预测奖励
            reward = context[arm] @ weight
            samples.append(reward)
        
        return torch.argmax(torch.stack(samples)).item()
    
    def update(self, arm, context, reward):
        """更新后验"""
        self.A[arm] += context.unsqueeze(1) @ context.unsqueeze(0)
        self.b[arm] += reward * context
```

---

## 5. 代码示例

### 5.1 实验对比

```python
import matplotlib.pyplot as plt

def run_experiment(n_arms=5, n_steps=1000, n_runs=100):
    """运行对比实验"""
    
    # 真实获胜概率
    true_probs = np.random.beta(2, 2, n_arms)
    print(f"真实获胜概率: {true_probs}")
    print(f"最佳臂: {np.argmax(true_probs)}")
    
    results = {
        'TS': [],
        'greedy': [],
        'UCB': []
    }
    
    for run in range(n_runs):
        # Thompson Sampling
        ts = ThompsonSampling(n_arms)
        
        # 贪心
        greedy = ThompsonSampling(n_arms)
        
        # UCB
        ucb_counts = np.zeros(n_arms)
        
        total_reward_ts = 0
        total_reward_greedy = 0
        total_reward_ucb = 0
        
        for t in range(n_steps):
            # Thompson Sampling
            arm_ts = ts.select_arm()
            reward = np.random.binomial(1, true_probs[arm_ts])
            ts.update(arm_ts, reward)
            total_reward_ts += reward
            
            # Greedy (总是选择当前最好)
            arm_greedy = np.argmax(greedy.get_mean())
            reward = np.random.binomial(1, true_probs[arm_greedy])
            greedy.update(arm_greedy, reward)
            total_reward_greedy += reward
            
            # UCB
            ucb_value = ts.get_mean() + np.sqrt(2*np.log(t+1) / (ucb_counts + 0.1))
            arm_ucb = np.argmax(ucb_value)
            reward = np.random.binomial(1, true_probs[arm_ucb])
            ts.update(arm_ucb, reward)
            ucb_counts[arm_ucb] += 1
            total_reward_ucb += reward
        
        results['TS'].append(total_reward_ts)
        results['greedy'].append(total_reward_greedy)
        results['UCB'].append(total_reward_ucb)
    
    # 输出结果
    for algo, rewards in results.items():
        print(f"{algo}: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    
    return results


def demo():
    print("=== Thompson Sampling 实验 ===\n")
    results = run_experiment(n_arms=5, n_steps=1000, n_runs=50)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    for algo, rewards in results.items():
        plt.hist(rewards, alpha=0.5, label=algo, bins=20)
    plt.xlabel('Total Reward')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Thompson Sampling vs Other Algorithms')
    plt.savefig('ts_experiment.png', dpi=100)
    print(f"\n图表已保存到 ts_experiment.png")


if __name__ == "__main__":
    demo()
```

### 5.2 在线服务

```python
class BanditService:
    """在线AB测试服务"""
    
    def __init__(self, n_variants):
        self.ts = ThompsonSampling(n_variants)
        self.n_requests = 0
    
    def select_variant(self):
        """选择测试变体"""
        variant = self.ts.select_arm()
        self.n_requests += 1
        return variant
    
    def report_reward(self, variant, reward):
        """报告奖励（转化=1，未转化=0）"""
        self.ts.update(variant, reward)
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'total_requests': self.n_requests,
            'expected_rates': self.ts.get_mean().tolist(),
            'alpha': self.ts.alpha.tolist(),
            'beta': self.ts.beta.tolist()
        }


# Web服务示例
def demo_service():
    print("=== Bandit服务演示 ===\n")
    
    service = BanditService(n_variants=3)
    
    # 模拟1000个请求
    true_rates = [0.1, 0.15, 0.12]  # 真实转化率
    
    for i in range(1000):
        variant = service.select_variant()
        # 模拟真实转化
        reward = np.random.binomial(1, true_rates[variant])
        service.report_reward(variant, reward)
    
    stats = service.get_stats()
    print(f"总请求: {stats['total_requests']}")
    print(f"估计转化率: {[f'{r:.3f}' for r in stats['expected_rates']]}")
    print(f"真实转化率: {true_rates}")


if __name__ == "__main__":
    demo_service()
```

---

## 6. 变体

### 6.1 粒子Thompson Sampling

使用粒子表示后验分布：

```python
class ParticleThompsonSampling:
    """粒子版本的Thompson Sampling"""
    
    def __init__(self, n_arms, n_particles=100):
        self.n_arms = n_arms
        self.n_particles = n_particles
        
        # 粒子权重
        self.weights = [
            np.ones(n_particles) / n_particles 
            for _ in range(n_arms)
        ]
    
    def select_arm(self):
        """加权采样选择"""
        samples = []
        for arm in range(self.n_arms):
            # 重采样
            indices = np.random.choice(
                self.n_particles, 
                size=self.n_particles, 
                p=self.weights[arm]
            )
            # 随机选择
            sample = np.random.choice(indices)
            samples.append(sample / self.n_particles)
        
        return np.argmax(samples)
```

### 6.2 分布式Thompson Sampling

```python
class DistributedThompsonSampling:
    """分布式Thompson Sampling"""
    
    def __init__(self, n_arms, n_workers):
        self.n_arms = n_arms
        self.n_workers = n_workers
        
        # 全局参数
        self.global_alpha = np.ones(n_arms)
        self.global_beta = np.ones(n_arms)
    
    def local_sample(self, worker_id):
        """本地采样"""
        return np.random.beta(self.global_alpha, self.global_beta)
    
    def aggregate(self, worker_id, local_updates):
        """聚合本地更新"""
        for arm, (alpha, beta) in enumerate(local_updates):
            self.global_alpha[arm] += alpha - 1
            self.global_beta[arm] += beta - 1
```

### 6.3 Bootstrap Thompson Sampling

```python
class BootstrapThompsonSampling:
    """Bootstrap Thompson Sampling"""
    
    def __init__(self, n_arms, n_bootstrap=50):
        self.n_arms = n_arms
        self.n_bootstrap = n_bootstrap
        
        # 存储历史数据
        self.data = {i: [] for i in range(n_arms)}
    
    def select_arm(self):
        """Bootstrap采样选择"""
        samples = []
        for arm in range(self.n_arms):
            if len(self.data[arm]) == 0:
                samples.append(0.5)
                continue
            
            # Bootstrap重采样
            bootstrap_data = np.random.choice(
                self.data[arm], 
                size=len(self.data[arm]), 
                replace=True
            )
            sample = np.mean(bootstrap_data)
            samples.append(sample)
        
        return np.argmax(samples)
```

---

## 7. 常见问题

### Q1: 何时使用Thompson Sampling？

- 当需要自适应探索时
- 当臂的奖励分布已知（伯努利/高斯）时
- 当需要平衡探索与利用时

### Q2: 与UCB相比的优势？

- 更自然地处理不确定性
- 更适合非平稳环境
- 实际中regret通常更低

### Q3: 如何处理非伯努利奖励？

- 连续奖励：使用高斯共轭
- 其他分布：使用拉普拉斯近似或粒子方法

### Q4: 计算复杂度高吗？

- 每次需要从K个Beta分布采样
- O(K)时间复杂度，可接受
- 大量臂时考虑使用LinUCB

---

## 8. 练习题

### 选择题

1. Thompson Sampling使用什么分布作为伯努利老虎机的后验？
   - A) 正态分布   B) Beta分布   C) Gamma分布
   - **答案：B**

2. Thompson Sampling的核心思想是？
   - A) 贪心选择   B) 随机采样选择   C) UCB上界
   - **答案：B**

3. 对于连续奖励，应该使用什么分布？
   - A) Beta   B) 正态   C) 泊松
   - **答案：B**

### 简答题

1. 解释Thompson Sampling如何平衡探索与利用？

   **答案**：通过从后验分布采样，高概率选择更好的臂（利用），同时有概率选择其他臂（探索）

2. 为什么Thompson Sampling比贪心更好？

   **答案**：贪心容易陷入局部最优，Thompson Sampling的自适应探索可以发现更好的臂

### 编程题

1. 实现非平稳环境的Thompson Sampling：

```python
# 解决方案：使用衰减先验或滑动窗口
class NonStationaryTS:
    def __init__(self, n_arms, decay=0.99):
        self.decay = decay
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
    
    def update(self, arm, reward):
        self.alpha *= self.decay
        self.beta *= self.decay
        
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

---

## 9. 学习路径

### 9.1 算法对比

```
多臂老虎机
    ↓
├─ UCB → 确定性
├─ ε-Greedy → 简单
└��� Thompson Sampling → 贝叶斯（推荐）
```

### 9.2 扩展方向

| 场景 | 方法 |
|------|------|
| 上下文信息 | LinTS |
| 非平稳环境 | 衰减先验 |
| 深度强化学习 | DQN + TS |

---

## 10. 附录

### A. 参考论文

- Thompson (1933). "On the likelihood that a given experiment will lead to a decision"
- Chapel et al. (2010). "A Bayesian Approach to Thompson Sampling"

### B. 实现对比

| 实现 | 特点 |
|------|------|
| scipy.stats.beta | 基础实现 |
| PyTorch | GPU加速 |
| River | 在线学习库 |

---

**文档结束**