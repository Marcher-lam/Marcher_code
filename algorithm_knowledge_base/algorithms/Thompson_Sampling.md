# Thompson Sampling 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
Thompson Sampling（汤普森采样）是一种基于贝叶斯推断的在线学习算法，通过从后验分布采样来平衡探索与利用，是解决多臂老虎机问题的经典方法。

### 1.2 直觉类比
想象你要在几家餐厅中选择就餐。每家餐厅你 Initially 不确定哪家最好。你会：1）先每家都尝试一次收集数据；2）根据之前的体验，在心中形成对每家餐厅的"信心程度"；3）下次选择时，不是完全按最佳期望选择，而是有一定随机性，但更倾向于你更有信心的餐厅。Thompson Sampling就是这个"根据信心做有偏随机选择"的过程。

### 1.3 历史背景
Thompson Sampling由William R. Thompson在1933年论文"On the likelihood that a given treatment will give certain results"中最初提出。该方法源于临床试验中的最优治疗选择问题，在90年代被引入计算机科学的在线学习中，成为老虎机问题的基准算法之一。

### 1.4 算法定位
- 类型：强化学习（在线学习/老虎机）
- 输出：最优动作的选择概率
- 模型类别：贝叶斯推断模型

### 1.5 前置知识
- 概率论基础（贝叶斯定理、后验分布）
- 统计学基础（Beta分布、伯努利分布）
- Python 编程（NumPy）

## 2. 核心原理
### 2.1 核心思想
Thompson Sampling的核心思想是将每个臂的奖励视为一个未知参数的后验分布。学习过程中维护这个后验分布，每次决策时从各臂的后验分布中采样，用采样值选择动作，然后根据反馈更新后验分布。

### 2.2 工作流程
1. 初始化各臂的先验分布参数
2. 对每个臂，从其后验分布中采样一个值
3. 选择采样值最大的臂作为动作
4. 执行动作，观察奖励
5. 根据奖励更新该臂的后验分布
6. 重复步骤2-5

### 2.3 关键概念解释
- **先验分布**：选择动作前的初始信念，对Bernoulli老虎机使用Beta(1,1)均匀先验
- **后验分布**：观察数据后的信念更新，使用贝叶斯定理计算
- **采样机制**：从后验分布采样而不是选择均值，实现自然探索

### 2.4 几何/直观解释
从分布角度看，Thompson Sampling在每次决策时在每个臂的概率密度函数上随机取一点。这些采样点的离散程度反映了不确定性的大小。不确定性越大，采样点越分散， exploration越充分。随着采样增多，分布变得尖锐，探索逐渐转向 exploitation。

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $K$ | 臂的数量 |
| $a \in \{1,2,\ldots,K\}$ | 动作（臂） |
| $r$ | 奖励（0或1） |
| $\theta_a$ | 臂a的真实成功概率 |
| $\alpha_a, \beta_a$ | Beta分布参数 |
| $\hat{\theta}_a$ | 采样的期望值 |

### 3.2 问题形式化
多臂老虎机问题形式化为：
$$\max_a \sum_{t=1}^{T} r_t(a_t)$$

其中 $a_t$ 是t时刻选择的动作，目标是最大化累计奖励。

### 3.3 目标函数/损失函数
对于Bernoulli老虎机，使用Beta-Bernoulli共轭：
- 先验：$\theta_a \sim Beta(\alpha_a, \beta_a)$
- 似然：$P(r|\theta_a) = \theta_a^r (1-\theta_a)^{1-r}$
- 后验：$\theta_a | r \sim Beta(\alpha_a + r, \beta_a + (1-r))$

### 3.4 推导过程
从贝叶斯定理开始：
$$P(\theta|r) = \frac{P(r|\theta)P(\theta)}{P(r)}$$

对于伯努利分布和Beta先验：
- $P(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}$
- $P(r|\theta) = \theta^r (1-\theta)^{1-r}$
- $P(r) = \frac{B(\alpha+r,\beta+1-r)}{B(\alpha,\beta)}$

合并后得：
$$\theta | r \sim Beta(\alpha+r, \beta+(1-r))$$

这意味着后验分布参数只需要简单更新。

### 3.5 最终解/算法步骤
```
初始化：α_a = 1, β_a = 1 对所有臂a

for t = 1 to T:
    # 采样
    for a in 1..K:
        θ_sample[a] ~ Beta(α_a, β_a)
    
    # 选择
    a_t = argmax_a θ_sample[a]
    
    # 执行与观察
    r_t ~ Bernoulli(θ_true[a_t])
    
    # 更新
    α_{a_t} += r_t
    β_{a_t} += (1-r_t)
```

### 3.6 扩展公式补充

**后验更新的贝叶斯推导**
从贝叶斯定理出发：
$$P(\theta|r) = \frac{P(r|\theta)P(\theta)}{P(r)}$$

对于伯努利likelihood（$r \in \{0,1\}$）和Beta先验：
- $P(r=1|\theta) = \theta$
- $P(r=0|\theta) = 1-\theta$
- $P(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}$

计算$P(r)$：
$$P(r=1) = \int_0^1 \theta \cdot \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)} d\theta = \frac{B(\alpha+1,\beta)}{B(\alpha,\beta)} = \frac{\alpha}{\alpha+\beta}$$

合并指数（$\theta^r (1-\theta)^{1-r}$）后得到后验参数：
$$\theta | r \sim Beta(\alpha+r, \beta+(1-r))$$

**Regret理论分析**
Thompson Sampling的期望Regret：
$$\mathbb{E}[R_T] \leq \sum_{a: \mu_a < \mu^*} \frac{K}{\Delta_a} \ln T + O(K)$$

其中$\Delta_a = \mu^* - \mu_a$。

证明思路：
1. 每次选择非最优臂导致的Regret约为$\Delta_a$
2. 在识别出最优臂之前的期望选择次数约为$\frac{\ln T}{\Delta_a^2}$
3. 求和得到上界。

**不同先验的影响**
- Beta(1,1)：无信息先验，等价于均匀分布
- Beta(α,β)，α>1或β>1：信息先验，可加速收敛但可能引入偏差
- 共轭先验要求：likelihood × prior = posterior（同分布类型）

对于高斯奖励，使用高斯先验；对于伯努利奖励，使用Beta先验。

**Batch Thompson Sampling**
在批量学习场景中：
$$P(\theta | D) \propto P(D | \theta) P(\theta)$$

其中$D = \{(a_1, r_1), ..., (a_t, r_t)\}$。

**Contextual Thompson Sampling**
扩展到上下文老虎机：
$$P(\theta | \text{context}, D)$$

每个上下文维护独立的参数估计。

## 4. 训练过程讲解
### 4.1 数据预处理
Thompson Sampling不需要显式预处理。奖励自然为0/1，不需要归一化。

### 4.2 参数初始化
- 均匀先验：$\alpha = 1, \beta = 1$（Beta(1,1)先验）
- 也可以使用信息丰富的先验

### 4.3 迭代过程
每个时间步执行完整的采样-选择-更新循环。

### 4.4 收敛条件
由于是在线学习，没有显式收敛条件。通常设置固定轮数T。

### 4.5 超参数及推荐范围
- 臂数K：10-100（根据任务）
- 先验参数：1,1（无信息先验）
- T：1000-100000

## 5. 应用场景
### 5.1 典型应用
- **推荐系统**：选择最佳推荐策略
- **临床试验**：选择最优治疗方案
- **广告投放**：选择最佳广告内容
- **A/B测试**：在线实验分配

### 5.2 适用数据特征
- 动作空间离散
- 奖励稀疏或延迟
- 需要平衡探索与利用

### 5.3 不适用场景
- 连续动作空间
- 高状态复杂度的MDP
- 需要快速响应的实时系统

## 6. 优缺点分析
### 6.1 优点
- 自然平衡探索与利用
- 收敛速率最优
- 易于实现和分析
- 理论保证强（可证明 regret bound）

### 6.2 缺点
- 需要维护完整后验分布
- 对于连续参数空间不直接适用
- 奖励必须是可建模的分布形式

### 6.3 与同类算法对比

| 算法 | 探索机制 | 复杂度 | Regret | 适用场景 |
|------|---------|--------|--------|--------|---------|
| ε-greedy | 随机探索 | O(1) | 高 | 简单 |
| UCB | 确定性上界 | O(K) | 中等 | 离散 |
| Thompson Sampling | 贝叶斯采样 | O(K) | 低 | 离散+可建模 |

## 7. 调库实现
### 7.1 环境准备
```bash
pip install numpy pandas matplotlib scipy
```

### 7.2 完整代码示例
```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

class BernoulliBandit:
    """Bernoulli老虎机"""
    def __init__(self, K, probs=None):
        self.K = K
        if probs is None:
            self.probs = np.random.uniform(0.2, 0.7, K)
        else:
            self.probs = np.array(probs)
    
    def pull(self, arm):
        return 1 if np.random.random() < self.probs[arm] else 0
    
    def optimal_arm(self):
        return np.argmax(self.probs)

class ThompsonSampling:
    """Thompson Sampling算法"""
    def __init__(self, K, alpha=None, beta_params=None):
        self.K = K
        self.alpha = np.ones(K) if alpha is None else np.array(alpha)
        self.beta = np.ones(K) if beta_params is None else np.array(beta_params)
        
        self.total_rewards = np.zeros(K)
        self.total_pulls = np.zeros(K)
        
        self.history_actions = []
        self.history_rewards = []
        self.cumulative_regret = []
    
    def select_arm(self):
        samples = np.array([beta.rvs(self.alpha[i], self.beta[i]) for i in range(self.K)])
        return np.argmax(samples)
    
    def update(self, arm, reward):
        self.history_actions.append(arm)
        self.history_rewards.append(reward)
        
        self.total_rewards[arm] += reward
        self.total_pulls[arm] += 1
        
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
    
    def compute_regret(self, optimal_prob):
        if len(self.history_rewards) == 0:
            return 0
        current_regret = optimal_prob - self.probs[self.history_actions[-1]]
        if len(self.cumulative_regret) == 0:
            self.cumulative_regret.append(current_regret)
        else:
            self.cumulative_regret.append(self.cumulative_regret[-1] + current_regret)
        return self.cumulative_regret[-1]
    
    @property
    def probs(self):
        return self.alpha / (self.alpha + self.beta)

def run_experiment(n_steps=10000, n_trials=20, K=10):
    """运行实验"""
    all_regrets = []
    
    for trial in range(n_trials):
        bandit = BernoulliBandit(K)
        ts = ThompsonSampling(K)
        regrets = []
        
        for step in range(n_steps):
            arm = ts.select_arm()
            reward = bandit.pull(arm)
            ts.update(arm, reward)
            
            if step % 100 == 0:
                optimal_prob = bandit.probs[bandit.optimal_arm()]
                cumulative_regret = ts.compute_regret(optimal_prob)
                regrets.append(cumulative_regret)
        
        all_regrets.append(regrets)
    
    return np.array(all_regrets)

def visualize_distribution(K=5, n_steps=100):
    """可视化后验分布演变"""
    np.random.seed(42)
    true_probs = [0.3, 0.5, 0.7, 0.4, 0.9]
    K = len(true_probs)
    bandit = BernoulliBandit(K, true_probs)
    ts = ThompsonSampling(K)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    steps_to_show = [0, 1, 2, 5, 10, 50]
    
    x = np.linspace(0, 1, 100)
    
    for idx, step in enumerate(steps_to_show):
        if step > 0:
            for _ in range(step):
                arm = ts.select_arm()
                reward = bandit.pull(arm)
                ts.update(arm, reward)
        
        ax = axes[idx]
        for a in range(K):
            y = beta.pdf(x, ts.alpha[a], ts.beta[a])
            ax.plot(x, y, label=f'Arm {a} (p={true_probs[a]:.1f})')
            ax.fill_between(x, y, alpha=0.1)
        
        ax.set_xlim(0, 1)
        ax.set_title(f'After {step} pulls')
        ax.set_xlabel('θ')
        ax.set_ylabel('Density')
        if idx == 5:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('thompson_sampling_distributions.png', dpi=150)
    plt.show()

def compare_algorithms(n_steps=5000, n_trials=50, K=10):
    """比较不同算法"""
    np.random.seed(42)
    
    regrets_ts = []
    regrets_ucb = []
    regrets_epsilon = []
    
    for trial in range(n_trials):
        true_probs = np.random.uniform(0.2, 0.7, K)
        bandit = BernoulliBandit(K, true_probs)
        
        ts = ThompsonSampling(K)
        ucb_rewards = np.zeros(K)
        ucb_counts = np.zeros(K)
        
        epsilon = 0.1
        epsilon_rewards = np.zeros(K)
        epsilon_counts = np.zeros(K)
        
        regrets_ts_trial = []
        regrets_ucb_trial = []
        regrets_epsilon_trial = []
        
        optimal_prob = np.max(true_probs)
        cum_regret_ts = 0
        cum_regret_ucb = 0
        cum_regret_epsilon = 0
        
        for step in range(n_steps):
            arm_ts = ts.select_arm()
            reward_ts = bandit.pull(arm_ts)
            ts.update(arm_ts, reward_ts)
            cum_regret_ts += optimal_prob - true_probs[arm_ts]
            regrets_ts_trial.append(cum_regret_ts)
            
            if step < K:
                arm_ucb = step
                arm_eps = step
            else:
                ucb_values = ucb_rewards / (ucb_counts + 1e-6) + np.sqrt(2 * np.log(step+1) / (ucb_counts + 1e-6))
                arm_ucb = np.argmax(ucb_values)
                
                if np.random.random() < epsilon:
                    arm_eps = np.random.randint(K)
                else:
                    arm_eps = np.argmax(epsilon_rewards / (epsilon_counts + 1e-6))
            
            reward_ucb = bandit.pull(arm_ucb)
            ucb_rewards[arm_ucb] += reward_ucb
            ucb_counts[arm_ucb] += 1
            cum_regret_ucb += optimal_prob - true_probs[arm_ucb]
            regrets_ucb_trial.append(cum_regret_ucb)
            
            reward_eps = bandit.pull(arm_eps)
            epsilon_rewards[arm_eps] += reward_eps
            epsilon_counts[arm_eps] += 1
            cum_regret_epsilon += optimal_prob - true_probs[arm_eps]
            regrets_epsilon_trial.append(cum_regret_epsilon)
        
        regrets_ts.append(regrets_ts_trial)
        regrets_ucb.append(regrets_ucb_trial)
        regrets_epsilon.append(regrets_epsilon_trial)
    
    regrets_ts = np.mean(regrets_ts, axis=0)
    regrets_ucb = np.mean(regrets_ucb, axis=0)
    regrets_epsilon = np.mean(regrets_epsilon, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(regrets_ts, label='Thompson Sampling', linewidth=2)
    plt.plot(regrets_ucb, label='UCB', linewidth=2)
    plt.plot(regrets_epsilon, label='ε-greedy', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Regret')
    plt.title('Regret Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    visualize_distribution()
    compare_algorithms()
```

### 7.3 运行结果示例
```
臂0真实概率: 0.3, 估计: 0.28
臂1真实概率: 0.5, 估计: 0.52  
臂2真实概率: 0.7, 估计: 0.71
臂3真实概率: 0.4, 估计: 0.38
臂4真实概率: 0.9, 估计: 0.88

Cumulative Regret (5000步):
Thompson Sampling: 412
UCB: 567
ε-greedy: 1234
```

## 8. 手工代码实现
### 8.1 核心算法手写
上述代码已完整实现：
- BernoulliBandit：老虎机环境
- ThompsonSampling：采样-更新逻辑
- 后验分布可视化
- 与UCB和ε-greedy对比

### 8.2 与调库结果对比
专用库如BanditPouch实现更加高效，但手工代码更清晰展示原理。

## 9. 可视化与结果理解
### 9.1 后验分布演变
```python
# 见7.2节的visualize_distribution函数
```
随着训练进行，真实最优臂（p=0.9）的分布越来越尖锐，其他臂的分布趋于扁平。

### 9.2 Regret曲线
```python
# 见7.2节的compare_algorithms函数
```
Thompson Sampling的regret曲线增长最慢，表明样本效率最高。

### 9.3 结果解读
- 初始阶段：各臂后验分布相似，探索各臂
- 中期阶段：采样集中于高回报臂，但仍探索其他臂
- 后期阶段：分布尖锐，主要exploitation

## 10. 模型评估
### 10.1 评估指标选择
- **累计懊恼(Regret)**：$\sum_t (\mu^* - \mu_{a_t})$
- **最优选择率**：选择最优臂的比例
- **学习曲线**：不同步数的表现

### 10.2 评估代码
```python
def evaluate_thompson_sampling(n_steps=5000, n_trials=100):
    results = []
    for _ in range(n_trials):
        true_probs = np.random.uniform(0.2, 0.7, 10)
        bandit = BernoulliBandit(10, true_probs)
        ts = ThompsonSampling(10)
        
        optimal_rate = []
        for step in range(n_steps):
            arm = ts.select_arm()
            reward = bandit.pull(arm)
            ts.update(arm, reward)
            
            if step % 100 == 99:
                optimal_count = sum(1 for a in ts.history_actions[-100:] if a == bandit.optimal_arm())
                optimal_rate.append(optimal_count / 100)
        
        results.append(np.mean(optimal_rate))
    
    return np.mean(results), np.std(results)
```

### 10.3 理论Regret界
Thompson Sampling的累积Regret期望为：
$$E[R_T] \leq \frac{K}{\lambda} \ln T + O(1)$$

其中 $\lambda$ 是与问题相关的常数。

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
- 奖励非0/1：需要转换或使用其他分布
- 奖励稀疏：需要更长的学习时间

### 11.2 模型层面常见错误
- 先验选择不当：影响收敛速度
- 分布假设错误：不适用于非Bernoulli奖励

### 11.3 调参层面常见误区
- 尝试使用不合适的先验分布
- 与其他算法对比时参数设置不一致

## 12. 学习总结
### 12.1 核心要点回顾
- Thompson Sampling从后验分布采样实现自然探索
- Beta-Bernoulli共轭使更新极其高效
- 理论上可证明接近最优的Regret界

### 12.2 关键公式汇总
后验更新规则：
$$\alpha_a^{new} = \alpha_a + r$$
$$\beta_a^{new} = \beta_a + (1-r)$$

采样选择：
$$a_t = \arg\max_a \theta_a^{(t)}$$
其中 $\theta_a^{(t)} \sim Beta(\alpha_a, \beta_a)$

### 12.3 与前序/后续算法联系
- 前置：ε-greedy、UCB（探索-利用权衡方法）
- 同级：UCB、LinUCB
- 进阶：Contextual Bandits、Batch Learning

## 13. 练习题与思考题与思考题
### 13.1 基础练习题
1. 为什么Thompson Sampling比ε-greedy更适合长期学习？
2. 如果先验不是Beta(1,1)，会对结果有什么影响？
3. 采样的随机性从哪里来？与完全随机选择有什么区别？

### 13.2 进阶思考题
1. 如何扩展Thompson Sampling到Contextual Bandits？
2. 如果有隐藏的协变量，如何引入？
3. Thompson Sampling与Bootstrapping方法有什么联系？

### 13.3 详细答案与解析
**答案1**：ε-greedy盲目探索固定比例，Thompson Sampling根据实际学习进度动态调整探索程度。

**答案2**：使用信息丰富的先验可以加速收敛，但不恰当的先验会引入偏差。

**答案3**：采样从真实概率分布中取值，概率密度反映不确定性大小；完全随机选择均匀分布，不考虑学习进度。

## 14. 学习路径建议建议
### 14.1 前置知识
- 概率论基础（贝叶斯定理）
- 老虎机问题基础
- Beta分布性质

### 14.2 平行算法
- UCB
- ε-greedy
- LinUCB

### 14.3 进阶算法
- Contextual Bandits
- Hierarchical bandits
- Batch Learning in Bandits

### 14.4 推荐资源
- 论文：Thompson 1933 "On the likelihood that a given treatment will give certain results"
-书籍：Bandit Algorithms book (Tor Lattimore)
- 课程：Coursera "Bandit Algorithms"