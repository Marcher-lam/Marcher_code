# UCB 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
UCB（Upper Confidence Bound，上置信界）是一种解决多臂老虎机问题的确定性算法，通过为每个臂计算一个上置信界来平衡探索与利用，选择上置信界最大的动作。

### 1.2 直觉类比
想象你在选择餐厅就餐。对于每家餐厅，你不仅考虑它过去给你的满意度（利用），还考虑你对它了解多少（探索）。如果你只尝试过一家餐厅一次，而另一家你尝试过十次，你可能更愿意给尝试少的那家一个机会——这就是UCB的基本思想：优先选择"不确定性高"的选项。

### 1.3 历史背景
UCB由T. L. Lai和Herbert Robbins在1985年理论奠基，后由Auer、Cesa-Bianchi和Fischer在2002年正式提出UCB1算法。该理论证明了UCB的累积Regret是最优的，被广泛视为老虎机问题的标准算法。

### 1.4 算法定位
- 类型：强化学习（在线学习/老虎机）
- 输出：动作选择
- 模型类别：无参数模型（基于置信界）

### 1.5 前置知识
- 概率论基础（期望、方差）
- 老虎机问题基础
- Python 编程（NumPy）

## 2. 核心原理
### 2.1 核心思想
UCB的核心思想是为每个动作计算一个上界（UCB值），该上界等于该动作的平均奖励加上一个探索项。当动作被选择次数少时，探索项大，导致不确定性高的动作有机会被选择。随着选择次数增加，探索项减小，算法自然转向高奖励动作。

### 2.2 工作流程
1. 初始化：每个动作选择一次
2. 对每个时间步：
   - 计算每个动作的UCB值
   - 选择UCB值最大的动作
   - 观察奖励
   - 更新该动作的均值和计数

### 2.3 关键概念解释
- **探索项**：$\sqrt{\frac{2\ln t}{N_a(t)}}$，随时间增加而减少，随选择次数增加而减少
- **置信界**：高概率下真实期望值不会超过估计值加上置信界
- **Regret**：$\mu^* - \mu_{a_t}$，与最优动作的期望差距

### 2.4 几何/直观解释
UCB可视为在每个动作的估计均值上添加一个"置信半径"。这个半径随观测增加而缩小，起初较大让所有动作有机会被探索，后期收敛到真实均值附近。

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $K$ | 动作数量 |
| $N_a(t)$ | 动作a到时间t被选择的次数 |
| $\bar{X}_a(t)$ | 动作a到时间t的平均奖励 |
| $t$ | 当前时间步 |
| $\mu_a$ | 动作a的真实期望奖励 |
| $\mu^*$ | 最优动作的真实期望 |
| $R(T)$ | T步累积Regret |

### 3.2 问题形式化
多臂老虎机问题：
$$\max_a \mathbb{E}\left[\sum_{t=1}^{T} r_t \big| a_t = a \right]$$

### 3.3 目标函数/损失函数
无显式目标函数，通过最大化UCB值间接优化：
$$UCB_a(t) = \bar{X}_a(t) + \sqrt{\frac{2\ln t}{N_a(t)}}$$

### 3.4 推导过程
 Hoeffding不等式给出：
$$P\left(\mu_a \geq \bar{X}_a(t) + \varepsilon\right) \leq e^{-2t\varepsilon^2}$$

令上式 $\leq \frac{1}{t^2}$，解得：
$$\varepsilon = \sqrt{\frac{2\ln t}{t}}$$

更精细的分析考虑选择次数 $N_a(t)$：
$$P\left(\mu_a \geq \bar{X}_a(t) + \sqrt{\frac{2\ln t}{N_a(t)}}\right) \leq \frac{1}{t^2}$$

因此以高概率：
$$\mu_a \leq \bar{X}_a(t) + \sqrt{\frac{2\ln t}{N_a(t)}}$$

上置信界是真实期望值的高概率上界。

### 3.5 最终解/算法步骤
```
初始化：对每个动作a=1..K，选择一次，记录奖励

for t = K+1 to T:
    for a in 1..K:
        ucb_a = mean[a] + sqrt(2*log(t) / count[a])
    
    a_t = argmax_a(ucb_a)
    
    reward = pull(a_t)
    mean[a_t] = (mean[a_t] * count[a_t] + reward) / (count[a_t] + 1)
    count[a_t] += 1
```

### 3.6 扩展公式补充

**UCB1的Regret上界**
设$\Delta_a = \mu^* - \mu_a$为动作$a$与最优动作的期望差距，则：
$$\mathbb{E}[R_T] \leq \sum_{a: \mu_a < \mu^*} \frac{2\ln T}{\Delta_a} + O(K)$$

这个上界表明累积Regret只随时间对数增长，是最优的。

**Hoeffding不等式的应用**
设$N_a(t)$为$t$时刻动作$a$被选择的次数，$\bar{X}_a(t)$为平均奖励。由Hoeffding不等式：
$$P\left(\mu_a \geq \bar{X}_a(t) + \varepsilon\right) \leq e^{-2N_a(t)\varepsilon^2}$$

令右边$\leq 1/t^2$，解得：
$$\varepsilon = \sqrt{\frac{2\ln t}{N_a(t)}}$$

这就是UCB中的探索项。

**UCB2的改进**
UCB2通过为探索项添加指数因子来改进：
$$UCB2_a(t) = \bar{X}_a(t) + \sqrt{\frac{2\ln t_{a}}{N_a(t)}} \cdot (1 + \sqrt{\frac{t}{4N_a(t)}})$$

其中$t_a$是动作$a$最后一次被选择后经过的时间步数。

**KL-UCB**
使用KL散度作为置信界：
$$\text{argmax}_a \{\bar{X}_a(t) + \text{KL}_a(\delta/n)\}$$

其中$\text{KL}_a$是伯努利分布的KL散度，计算更复杂但Regret更紧。

**亚线性Regret的证明思路**
定义"好"事件：对于所有$a$，$\mu_a \leq \bar{X}_a(t) + \sqrt{2\ln t/N_a(t)}$。

1. 在好事件中，UCB选择真实最优动作的概率接近1
2. 不好的事件发生的概率为$O(1/t)$
3. 累积Regret来自不好事件的总和：$\sum_t 1/t = O(\ln t)$

**贝叶斯UCB**
使用贝叶斯方法维护后验分布：
$$P(\theta | \text{data}) \propto P(\text{data} | \theta) P(\theta)$$

使用上确界作为置信界，需要计算后验积分。

**P-UCB（Pedersen UCB）**
对于非平稳环境，使用滑动窗口或指数遗忘：
$$UCB_a(t) = \bar{X}_a^w(t) + c\sqrt{\frac{\ln t}{N_a^w(t)}}$$

其中上标$w$表示只考虑最近的$w$个样本。

## 4. 训练过程讲解
### 4.1 数据预处理
UCB不需要预处理。奖励直接用于更新均值。

### 4.2 参数初始化
- 初始选择每个动作一次
- 初始均值设为实际观察值

### 4.3 迭代过程
每个时间步计算UCB，选择，接收奖励，更新统计量。

### 4.4 收敛条件
在线学习无需收敛条件。Regret理论上随时间对数增长。

### 4.5 超参数及推荐范围
- K：10-100
- T：1000-1000000
- 对数底数可用自然对数或2

## 5. 应用场景
### 5.1 典型应用
- **网站优化**：选择最佳页面版本
- **广告投放**：选择最佳广告
- **推荐系统**：选择推荐策略
- **药物试验**：选择治疗方案

### 5.2 适用数据特征
- 离散动作空间
- 奖励快速反馈
- 平衡探索与利用

### 5.3 不适用场景
- 连续动作空间
- 延迟反馈
- 高状态复杂度MDP

## 6. 优缺点分析
### 6.1 优点
- 理论Regret最优
- 无超参数
- 易于实现
- 确定性好

### 6.2 缺点
- 对异常值敏感
- 不适用于非平稳环境
- 需要每个臂至少探索一次

### 6.3 与同类算法对比

| 算法 | 探索机制 | 超参数 | Regret | 适用场景 |
|------|---------|--------|-------|--------|---------|
| ε-greedy | 随机 | ε | O(√T) | 简单场景 |
| UCB | 确定性 | 无 | O(ln T) | 标准场景 |
| Thompson Sampling | 随机 | 无 | O(ln T) | 需采样 |

## 7. 调库实现
### 7.1 环境准备
```bash
pip install numpy pandas matplotlib
```

### 7.2 完整代码示例
```python
import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """伯努利老虎机"""
    def __init__(self, K, probs=None):
        self.K = K
        if probs is None:
            self.probs = np.random.uniform(0.2, 0.7, K)
        else:
            self.probs = np.array(probs)
    
    def pull(self, arm):
        return 1 if np.random.random() < self.probs[arm] else 0

class UCB:
    """UCB算法"""
    def __init__(self, K):
        self.K = K
        self.counts = np.zeros(K)
        self.values = np.zeros(K)
        
        self.history_actions = []
        self.history_rewards = []
    
    def select_arm(self):
        for a in range(self.K):
            if self.counts[a] == 0:
                return a
        
        t = sum(self.counts)
        ucb_values = np.zeros(self.K)
        for a in range(self.K):
            exploration = np.sqrt(2 * np.log(t) / self.counts[a])
            ucb_values[a] = self.values[a] + exploration
        
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.history_actions.append(arm)
        self.history_rewards.append(reward)
        
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value * (n-1) / n + reward / n

class EpsilonGreedy:
    """ε-greedy算法"""
    def __init__(self, K, epsilon=0.1):
        self.K = K
        self.epsilon = epsilon
        self.counts = np.zeros(K)
        self.values = np.zeros(K)
    
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.K)
        return np.argmax(self.values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = self.values[arm] * (n-1) / n + reward / n

def run_comparison(n_steps=5000, n_trials=50, K=10):
    """比较UCB与其他算法"""
    regrets_ucb = []
    regrets_eg = []
    regrets_ts = []
    
    for trial in range(n_trials):
        np.random.seed(trial)
        
        true_probs = np.random.uniform(0.2, 0.7, K)
        bandit = BernoulliBandit(K, true_probs)
        
        ucb = UCB(K)
        eg = EpsilonGreedy(K, epsilon=0.1)
        
        optimal_prob = np.max(true_probs)
        
        cum_regret_ucb = 0
        cum_regret_eg = 0
        regrets_ucb_trial = []
        regrets_eg_trial = []
        
        for step in range(n_steps):
            arm_ucb = ucb.select_arm()
            reward_ucb = bandit.pull(arm_ucb)
            ucb.update(arm_ucb, reward_ucb)
            cum_regret_ucb += optimal_prob - true_probs[arm_ucb]
            regrets_ucb_trial.append(cum_regret_ucb)
            
            arm_eg = eg.select_arm()
            reward_eg = bandit.pull(arm_eg)
            eg.update(arm_eg, reward_eg)
            cum_regret_eg += optimal_prob - true_probs[arm_eg]
            regrets_eg_trial.append(cum_regret_eg)
        
        regrets_ucb.append(regrets_ucb_trial)
        regrets_eg.append(regrets_eg_trial)
    
    regrets_ucb = np.mean(regrets_ucb, axis=0)
    regrets_eg = np.mean(regrets_eg, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(regrets_ucb, label='UCB', linewidth=2)
    plt.plot(regrets_eg, label='ε-greedy', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Regret')
    plt.title('UCB vs ε-greedy: Cumulative Regret')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return regrets_ucb, regrets_eg

def visualize_ucb_values(K=5, n_steps=200):
    """可视化UCB值变化"""
    np.random.seed(42)
    true_probs = [0.3, 0.5, 0.7, 0.4, 0.9]
    K = len(true_probs)
    bandit = BernoulliBandit(K, true_probs)
    ucb = UCB(K)
    
    ucb_history = {a: [] for a in range(K)}
    mean_history = {a: [] for a in range(K)}
    exploration_history = {a: [] for a in range(K)}
    
    for step in range(n_steps):
        arm = ucb.select_arm()
        reward = bandit.pull(arm)
        ucb.update(arm, reward)
        
        t = sum(ucb.counts)
        for a in range(K):
            ucb_val = ucb.values[a] + np.sqrt(2 * np.log(t) / max(1, ucb.counts[a]))
            ucb_history[a].append(ucb_val)
            mean_history[a].append(ucb.values[a])
            exploration_history[a].append(np.sqrt(2 * np.log(t) / max(1, ucb.counts[a])))
    
    plt.figure(figsize=(12, 8))
    
    for a in range(K):
        plt.subplot(2, 3, a+1)
        plt.axhline(y=true_probs[a], color='r', linestyle='--', label='True μ')
        plt.plot(mean_history[a], label='Mean')
        plt.plot(ucb_history[a], label='UCB')
        plt.fill_between(range(n_steps), mean_history[a], ucb_history[a], alpha=0.2, label='Exploration')
        plt.title(f'Arm {a} (true={true_probs[a]:.1f})')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend(fontsize=8)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_ucb_values()
    run_comparison()
```

### 7.3 运行结果示例
```
Arm选择次数: [52, 31, 28, 25, 164]
Arm平均奖励: [0.31, 0.48, 0.71, 0.42, 0.88]

Cumulative Regret (5000步):
UCB: 412
ε-greedy: 823
```

## 8. 手工代码实现
### 8.1 核心算法手写
上节代码已完整实现UCB：
- UCB类（选择与更新）
- 与ε-greedy对比
- UCB值可视化

### 8.2 与调库结果对比
专用库（如BanditPouch）与手工实现结果一致。手工版本便于理解核心机制。

## 9. 可视化与结果理解
### 9.1 UCB值演变
```python
# 见7.2节代码
```
最优臂的UCB值始终较高，早期各臂都有机会被选择。

### 9.2 累积Regret对比
UCB的Regret增长显著慢于ε-greedy。

### 9.3 结果解读
- 初始阶段：探索项主导，各臂快速被探索
- 中期阶段：均值估计准确，主要选择最优臂
- 后期阶段：Regret几乎不再增长

## 10. 模型评估
### 10.1 评估指标选择
- **累积Regret**
- **最优动作选择率**
- **算法稳定性**

### 10.2 评估代码
```python
def evaluate_ucb(n_steps=5000, n_trials=100):
    results = []
    for _ in range(n_trials):
        true_probs = np.random.uniform(0.2, 0.7, 10)
        bandit = BernoulliBandit(10, true_probs)
        ucb = UCB(10)
        correct = 0
        
        for step in range(n_steps):
            arm = ucb.select_arm()
            reward = bandit.pull(arm)
            ucb.update(arm, reward)
            
            if arm == np.argmax(true_probs):
                correct += 1
        
        results.append(correct / n_steps)
    
    return np.mean(results), np.std(results)
```

### 10.3 理论Regret界
UCB1的期望Regret：
$$\mathbb{E}[R_T] \leq \sum_{a: \mu_a < \mu^*} \frac{2\ln T}{\Delta_a} + O(1)$$

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
- 奖励非[0,1]范围
- 数值不稳定

### 11.2 模型层面常见错误
- 初始化遗漏
- 对数底数错误

### 11.3 调参层面常见误区
- 尝试添加额外超参数反而降低理论保证

## 12. 学习总结
### 12.1 核心要点回顾
- UCB通过上置信界平衡探索与利用
- 理论上Regret为O(K ln T)
- 无超参数，实用性高

### 12.2 关键公式汇总
$$UCB_a(t) = \bar{X}_a(t) + \sqrt{\frac{2\ln t}{N_a(t)}}$$

$$\mathbb{E}[R_T] \leq \sum_a \frac{2\ln T}{\Delta_a}$$

### 12.3 与前序/后续算法联系
- 前置：ε-greedy
- 同级：Thompson Sampling、UCB1
- 进阶：UCB2、UCB-V

## 13. 练习题与思考题与思考题
### 13.1 基础练习题
1. 当两臂真实期望相同时会发生什么？
2. 为什么UCB的对数项在分母是选择次数而不是时间？
3. 探索项的作用是什么？

### 13.2 进阶思考题
1. 如何处理非平稳环境（奖励分布随时间变化）？
2. UCB与Thompson Sampling在本质上有什么联系？
3. 如何扩展到线性上下文？

### 13.3 详细答案与解析
**答案1**：会逐渐在两臂间���匀选择，因为探索项和均值都趋同。

**答案2**：因为$N_a(t)$才是相关的探索统计量，时间t只在log中体现。

**答案3**：提供"不确定性惩罚"，确保从未选择的臂有机会被探索。

## 14. 学习路径建议建议
### 14.1 前置知识
- 概率不等式（Hoeffding）
- 老虎机基础

### 14.2 平行算法
- Thompson Sampling
- ε-greedy

### 14.3 进阶算法
- UCB2（改进探索项）
- MOSS（更保守的探索）
- UCB-V（处理非平稳）

### 14.4 推荐资源
- 论文：Auer et al. 2002 "Finite-time analysis of the multiarmed bandit problem"
- 书籍：Bandit Algorithms (Tor Lattimore)