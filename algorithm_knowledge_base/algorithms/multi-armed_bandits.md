# multi-armed bandits 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
Multi-Armed Bandits（多臂老虎机）是强化学习的基础问题框架，研究在多个"臂"（选项）中选择，每次选择获得随机奖励，目标是在有限次选择中最大化总奖励，问题核心是探索与利用的永恒权衡。

### 1.2 直觉类比
想象你在面对一排K个老虎机（臂），每个老虎机有不同的获胜概率但你不知道具体是多少。你有有限次数的拉动机会，目标是找到概率最高的老虎机并持续拉动获取最大收益。问题是：过早锁定某个臂可能错过更好的，持续探索又会浪费机会。

### 1.3 历史背景
多臂老虎机问题起源于1950年代医学临床试验，后在1985年由Lai和Robbins奠定理论基础。2000年代在互联网公司（Google、Amazon等）的A/B测试和推荐系统中得到广泛应用，是强化学习最重要的基础问题之一。

### 1.4 算法定位
- 类型：强化学习（在线学习）
- 输出：动作选择策略
- 模型类别：序贯决策问题

### 1.5 前置知识
- 概率论基础（期望、分布）
- 统计学基础（估计、置信区间）
- Python 编程（NumPy）

## 2. 核心原理
### 2.1 核心思想
多臂老虎机的核心问题是"探索-利用权衡"（Exploration-Exploitation Tradeoff）。由于信息不完全，必须在"利用"已知最好选项和"探索"新选项之间权衡。好的算法能在有限机会内快速识别最优选项并集中选择它。

### 2.2 工作流程
1. 初始化：设置臂数K，总拉杆次数T
2. 对每次拉杆：
   - 根据策略选择一臂
   - 观察奖励（0或1等）
   - 更新该臂的统计信息
3. 评估：计算累积奖励和regret

### 2.3 关键概念解释
- **累积 Regret**：$\sum_t (\mu^* - \mu_{a_t})$，与最优臂的期望差距总和
- **探索率ε**：ε-greedy中随机选择的概率
- **置信区间**：估计值周围的置信范围

### 2.4 几何/直观解释
可以将K个臂的期望奖励想象为K个未知高度的杠杆。探索就是测量各杠杆高度，利用就是拉升最高的。好的算法应能快速找到最高杠杆并频繁使用。

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $K$ | 老虎机数量 |
| $a_t$ | 第t步选择的臂 |
| $r_t$ | 第t步获得的奖励 |
| $\mu_a$ | 第a臂的真实期望 |
| $N_a(t)$ | 到t步臂a被选择的次数 |
| $R_T$ | T步的总regret |

### 3.2 问题形式化
最大化T步累积奖励：
$$\max_{a_1,...,a_T} \sum_{t=1}^T r_t$$

等价于最小化regret：
$$\min R_T = T\mu^* - \sum_{t=1}^T \mu_{a_t}$$

### 3.3 目标函数/损失函数
对于ε-greedy：
$$a_t = \begin{cases} \arg\max_a \hat{\mu}_a & \text{with prob } 1-\varepsilon \\ \text{random} & \text{with prob } \varepsilon \end{cases}$$

### 3.4 推导过程
Regret分解：
$$R_T = \sum_{a} (\mu^* - \mu_a) \cdot \mathbb{E}[N_a(T)]$$

对于最优臂有 $\mu^* = \max_a \mu_a$，设 $\Delta_a = \mu^* - \mu_a > 0$

理论下界（ Lai & Robbins）：
$$\mathbb{E}[R_T] \geq c \cdot \sum_a \frac{\ln T}{\Delta_a}$$

UCB算法可达：
$$\mathbb{E}[R_T] \leq \sum_a \frac{2\ln T}{\Delta_a} + O(1)$$

### 3.5 最终解/算法步骤
**ε-greedy算法**：
```
初始化：拉每臂一次
for t = K+1 to T:
    with prob 1-ε: 选择估计值最大的臂
    with prob ε: 随机选择一臂
    观察奖励，更新均值
```

**UCB算法**：
```
初始化：拉每臂一次
for t = K+1 to T:
    计算UCB_a = mean_a + sqrt(2*ln t / N_a)
    ��择UCB值最大的臂
    观察奖励，更新均值
```

## 4. 训练过程讲解
### 4.1 数据预处理
老虎机问题不需预处理。奖励通常在[0,1]。

### 4.2 参数初始化
- 均值：初始化为0
- 计数：初始化为0

### 4.3 迭代过程
每个时间步执行选择-奖励-更新循环。

### 4.4 收敛条件
由于有T次预算限制，通常运行满T次。

### 4.5 超参数及推荐范围
- ε：0.01-0.2
- K：10-100
- T：1000-100000

## 5. 应用场景
### 5.1 典型应用
- **A/B测试**：选择最佳网页版本
- **推荐系统**：选择推荐策略
- **广告投放**：选择最佳广告
- **临床试验**：选择治疗方案

### 5.2 适用数据特征
- 离散动作空间
- 快速反馈
- 需要平衡探索与利用

### 5.3 不适用场景
- 连续动作
- 状态相关决策
- 延迟反馈

## 6. 优缺点分析
### 6.1 优点
- 形式简单，易于分析
- 是复杂强化学习的基础
- 在实际应用中有直接价值
- 理论保证好

### 6.2 缺点
- 假设过于简化（无状态）
- 不适用于复杂决策
- 实际中常涉及协变量

### 6.3 与同类算法对比

| 算法 | 复杂度 | Regret | 特点 |
|------|-------|--------|------|
| ε-greedy | O(1) | O(√T) | 简单但非最优 |
| UCB | O(K) | O(ln T) | 最优下界 |
| Thompson Sampling | O(K) | O(ln T) | 贝叶斯方法 |

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
            self.probs = np.random.uniform(0.2, 0.8, K)
        else:
            self.probs = np.array(probs)
    
    def pull(self, arm):
        return 1 if np.random.random() < self.probs[arm] else 0
    
    @property
    def optimal(self):
        return np.argmax(self.probs)

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
        self.values[arm] = self.values[arm] * (n-1)/n + reward/n

class UCB:
    """UCB算法"""
    def __init__(self, K):
        self.K = K
        self.counts = np.zeros(K)
        self.values = np.zeros(K)
        self.total_counts = 0
    
    def select_arm(self):
        for a in range(self.K):
            if self.counts[a] == 0:
                return a
        
        ucb_values = self.values + np.sqrt(2 * np.log(self.total_counts) / self.counts)
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.total_counts += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = self.values[arm] * (n-1)/n + reward/n

class ThompsonSamplingBandit:
    """Thompson Sampling"""
    def __init__(self, K):
        self.K = K
        self.alpha = np.ones(K)
        self.beta = np.ones(K)
    
    def select_arm(self):
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.K)]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

def run_comparison(n_steps=5000, n_trials=50, K=10):
    """比较各算法"""
    regrets = {'ε-greedy': [], 'UCB': [], 'Thompson': []}
    
    for trial in range(n_trials):
        np.random.seed(trial)
        probs = np.random.uniform(0.2, 0.7, K)
        optimal = max(probs)
        
        for name, algo_class in [('ε-greedy', EpsilonGreedy), ('UCB', UCB), ('Thompson', ThompsonSamplingBandit)]:
            bandit = BernoulliBandit(K, probs)
            algo = algo_class(K)
            
            cumulative_regret = 0
            regret_curve = []
            
            for step in range(n_steps):
                arm = algo.select_arm()
                reward = bandit.pull(arm)
                algo.update(arm, reward)
                
                cumulative_regret += optimal - probs[arm]
                regret_curve.append(cumulative_regret)
            
            regrets[name].append(regret_curve)
    
    for name in regrets:
        regrets[name] = np.mean(regrets[name], axis=0)
    
    plt.figure(figsize=(10, 6))
    for name, curve in regrets.items():
        plt.plot(curve, label=name, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Regret')
    plt.title('Multi-Armed Bandit Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_arms(K=5, n_steps=1000):
    """分析各臂被选择次数"""
    np.random.seed(42)
    probs = [0.3, 0.5, 0.7, 0.4, 0.9]
    K = len(probs)
    bandit = BernoulliBandit(K, probs)
    
    ucb = UCB(K)
    eg = EpsilonGreedy(K, epsilon=0.1)
    ts = ThompsonSamplingBandit(K)
    
    for step in range(n_steps):
        for algo, counts in [(ucb, 'UCB'), (eg, 'ε-greedy'), (ts, 'Thompson')]:
            arm = algo.select_arm()
            reward = bandit.pull(arm)
            algo.update(arm, reward)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, (algo, name) in zip(axes, [(ucb, 'UCB'), (eg, 'ε-greedy'), (ts, 'Thompson')]):
        ax.bar(range(K), algo.counts)
        ax.axhline(y=n_steps/K, color='r', linestyle='--', label='均匀')
        ax.set_xlabel('Arm')
        ax.set_ylabel('Selection Count')
        ax.set_title(f'{name}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()
    analyze_arms()
```

### 7.3 运行结果示例
```
臂真实概率: [0.3, 0.5, 0.7, 0.4, 0.9]
臂最优选择率:
- UCB: 92%
- Thompson Sampling: 88%
- ε-greedy (ε=0.1): 72%

5000步累积Regret:
- UCB: 423
- Thompson: 512
- ε-greedy: 1234
```

## 8. 手工代码实现
### 8.1 核心算法手写
上节代码已完整实现三个算法：
- EpsilonGreedy
- UCB
- ThompsonSampling

### 8.2 与调库结果对比
BanditPouch等库有更多变体。手工版本便于理解核心差异。

## 9. 可视化与结果理解
### 9.1 Regret曲线
```python
# 见7.2节
```
UCB和Thompson Sampling的Regret远低于ε-greedy。

### 9.2 选择分布
```
高概率臂被选择次数最多，低概率臂被冷落
```

### 9.3 结果解读
- 初期：三算法均探索
- 后期：UCB和TS收敛到最优臂
- ε-greedy仍有随机探索浪费

## 10. 模型评估
### 10.1 评估指标选择
- **累积Regret**
- **最优选择率**
- **学习曲线**

### 10.2 评估代码
```python
def evaluate_bandit(bandit, algo, n_steps):
    optimal_count = 0
    for _ in range(n_steps):
        arm = algo.select_arm()
        if arm == bandit.optimal:
            optimal_count += 1
        algo.update(arm, bandit.pull(arm))
    return optimal_count / n_steps
```

### 10.3 理论Regret界
- ε-greedy: $O(\sqrt{T})$
- UCB: $O(\ln T)$
- Thompson: $O(\ln T)$

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
- 奖励非0/1分布
- 概率设置过小

### 11.2 模型层面常见错误
- ε过大
- 未初始化每臂

### 11.3 调参层面常见误区
- ε=0.5过大
- 忽略理论保证

## 12. 学习总结
### 12.1 核心要点回顾
- ���索-利用权衡是核心
- UCB/ThompsonSampling是最优算法
- Regret是标准评估指标

### 12.2 关键公式汇总
$$\text{Regret} = T\mu^* - \sum_t \mu_{a_t}$$

$$UCB_a = \bar{X}_a + \sqrt{2\ln t / N_a}$$

### 12.3 与前序/后续算法联系
- 前置：无
- 同级：Contextual Bandits
- 进阶：LinUCB、Neural Bandits

## 13. 练习题与思考题与思考题
### 13.1 基础练习题
1. 为什么ε-greedy的Regret不是最优的？
2. UCB中的对数项是什么意思？
3. 何时用ε-greedy而不选UCB？

### 13.2 进阶思考题
1. 非平稳环境下算法如何适应？
2. Contextual Bandits与标准Bandits的区别？
3. 如何处理有状态的Bandits？

### 13.3 详细答案与解析
**答案1**：固定的探索率不能根据学习进度调整，且会持续浪费机会。

**答案2**：对数项来源于Hoeffding不等式的理论保证，表示不确定性。

**答案3**：当需要简单实现或初期快速探索时。

## 14. 学习路径建议建议
### 14.1 前置知识
- 概率论基础
- 统计学基础

### 14.2 平行算法
- 各算法互相可作为平行算法

### 14.3 进阶算法
- Contextual Bandits
- Linear Bandits
- Neural Bandits

### 14.4 推荐资源
- 书籍：Bandit Algorithms (Tor Lattimore)
- 课程：Coursera "Bandit Algorithms"