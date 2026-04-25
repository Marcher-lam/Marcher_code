# UCB 学习文档

> 上置信界算法，多臂老虎机中的经典探索-利用平衡方法

---

## 1. 算法基础认知

### 1.1 一句话定义

UCB（Upper Confidence Bound，上置信界）是多臂老虎机问题的一种求解算法，核心思想是**同时考虑每个臂的期望奖励和不确定性来选择臂**：选择"已知收益高"且"不确定性大"的臂，从而在探索与利用之间取得平衡。

### 1.2 直觉类比

想象你在选择餐厅吃饭。A餐厅你去过很多次，味道不错（利用）；B餐厅你只去过一次，还没确定（探索）。UCB的策略是：不仅考虑哪个餐厅"很可能好吃"，还要考虑我们对它的"不确定程度"。新的没去过的餐厅"不确定性"最高，即使期望稍低也值得尝试。数学上，"不确定性"用置信区间的上界表示，选择期望+上界总和最大的餐厅。

### 1.3 历史背景

UCB由Auer, Cesa-Bianchi和Fischer于2002年在论文《Finite-time Analysis of the Multiarmed Bandit Problem》中正式提出给出了O(sqrt(KT log T))的期望后悔上界。之前1995年的Hoeffding bound已有类似思想。UCB1是最经典的UCB变体，后续有UCB2、UCB-V、KL-UCB等多个改进版本。UCB与Thompson Sampling并列为多臂老虎机的标准解法。

### 1.4 算法定位

| 特性 | 说明 |
|------|------|
| 类型 | 强化学习 / 在线学习 |
| 输出 | 臂的选择策略 |
| 模型类型 | 确定性算法 |
| 时间复杂度 | O(log T) per round |

### 1.5 前置知识

- [必备]：概率基础（Chernoff bound）
- [必备]：对数运算
- [扩展]：多臂老虎机基础
- [扩展]：Thompson Sampling

---

## 2. 核心原理

### 2.1 核心思想

UCB的核心思想是**置信界**：用统计方法估计每个臂的期望奖励的置信区间，选择上界最大的臂。这样：
- 期望高的臂→上界高（利用）
- 不确定性大的臂→上界高（探索）
- 随着数据增加，不确定性缩小，上界收敛

### 2.2 工作流程

```
初始化：每个臂先，拉一次，获得初始奖励
每轮循环：
    1. 对每个臂，计算UCB值 = 平均奖励 + 置信上界
    2. 选择UCB值最大的臂
    3. 拉取该臂，获得奖励
    4. 更新该臂的统计
```

### 2.3 关键概念解释

- **置信界**：以一定概率保证的真实期望的上界。

- ** Hoeffding不等式**：可用于推导UCB，给出概率上界。

- **探索项**：`sqrt(2 log t / n)`项，随着拉取次数增加而减少。

- **利用项**：平均奖励项，反映已知信息。

- **后悔（Regret）**：选择非最优臂带来的期望损失。

### 2.4 几何/直观解释

将每个臂的估计看作一条带误差棒的点。UCB值=点+误差棒。初期误差棒长，什么臂都有可能；后期误差棒短，只选均值最高的。随着轮数增加，探索减少，利用增多。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| K | 臂的数量 | 标量 |
| T | 总轮数 | 标量 |
| t | 当前轮数 | 标量 |
| μ_k | 第k个臂的真实期望 | 标量 |
| μ̂_k | 估计的平均奖励 | 标量 |
| n_k | 臂k被拉取的次数 | 标量 |
| r_t | 第t轮的奖励 | 标量 |
| UCB_k(t) | 臂k在时刻t的UCB值 | 标量 |
| Δ_k | 臂k与最优臂的差距 | 标量 |

### 3.2 问题形式化

多臂老虎机：K个臂，每个臂有固定但未知的期望奖励μ_k。

目标是一轮接一轮选择臂，最大化累计奖励：
$$\max \sum_{t=1}^{T} r_t$$

UCB1的决策规则：
$$a_t = \arg\max_{k} \left( \hat{\mu}_k(t-1) + \sqrt{\frac{2 \log t}{n_k(t-1)}} \right)$$

其中：
- $\hat{\mu}_k(t-1)$：到t-1时刻的平均奖励
- $n_k(t-1)$：到t-1时刻拉取k的次数

### 3.3 目标函数/损失函数

累计奖励：
$$G = \sum_{t=1}^{T} r_t$$

累计后悔：
$$Regret(T) = \sum_{t=1}^{T} (\mu^* - \mu_{a_t})$$

其中μ* = max_k μ_k是最优臂的期望。

UCB1的期望后悔上界：
$$E[Regret(T)] \leq \sum_{k: \Delta_k > 0} \left( \frac{8 \log T}{\Delta_k} + \Delta_k \right)$$

### 3.4 推导过程

**步骤1：Hoeffding不等式**

设X_i是独立同分布的[0,1]随机变量，E[X_i] = μ，则：
$$P(|\bar{X} - \mu| \geq \epsilon) \leq 2e^{-2T\epsilon^2}$$

转化为UCB形式：
$$P(\mu > \hat{\mu}_k + u) \leq 2e^{-2T u^2}$$

设右边等于δ，解出u：
$$u = \sqrt{\frac{1}{2T} \log\frac{2}{\delta}}$$

**步骤2：合并时间项**

设整体置信水平δ=1/t²，则：
$$u_k(t) = \sqrt{\frac{2 \log t}{n_k(t)}}$$

加上安全常数sqrt(2)得到标准形式。

**步骤3：选择规则**

UCB_k(t) = μ̂_k + u_k(t)

选择最大化UCB_k(t)的臂。

### 3.5 算法步骤

```
输入：臂数K，总轮数T
输出：选择序列

1. 初始化：对每个臂k=1到K
       拉取一次，记录奖励
       n_k = 1, sum_k = r_k

2. for t = K+1 to T:
    a. for each arm k:
        计算 UCB_k(t) = sum_k/n_k + sqrt(2*log(t) / n_k)
    
    b. 选择臂: a_t = argmax_k UCB_k(t)
    
    c. 拉取臂a_t，获得奖励r
    
    d. 更新:
       n_{a_t} += 1
       sum_{a_t} += r
```

---

## 4. 训练过程讲解

### 4.1 初始化

- **每个臂先拉一次**：保证n_k > 0，避免除零
- **也可以用历史数据初始化**：用已有数据计算初始值

### 4.2 参数调整

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| c | 探索常数 | 1-2 | sqrt(2) |
| alpha | 指数 | 0.5-1 | 1（UCB1）|
| initbonus | 初始奖励 | 0.5-1 | 0 |

### 4.3 收敛条件

- **固定轮数**：跑T轮后停止
- **平稳性检测**：选择变化率小于阈值
- **累计后悔阈值**：后悔小于某值

---

## 5. 应用场景

### 5.1 典型应用

**网站优化**：选择不同的页面布局，看哪个转化率高。

**广告投放**：测试不同广告创意，选择点击率高的。

**超参数调优**：选择不同的参数组合，快速找到最优。

**药物试验**：分配患者到不同治疗组。

**推荐系统**：平衡新内容推荐和热门推荐。

### 5.2 适用数据特征

- **固定奖励分布**：每个臂的奖励分布不随时间变化
- **在线学习**：数据顺序到达，不能预先知道
- **探索必要**：对某些臂不了解或了解少
- **成本敏感**：探索有成本，不能穷举

### 5.3 不适用场景

- **对抗环境**：臂的奖励会适应变化
- **离线评估**：所有数据已知
- **连续动作**：臂太多或连续
- **延迟反馈**：奖励获得有延迟

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 有理论保证 | O(log T)后悔上界 | 奖励有界 |
| 确定性 | 无随机性，结果可复现 | 参数固定 |
| 简单高效 | O(K log T)复杂度 | K不太大 |
| 无参数 | 只有一个c常数 | 奖励[0,1] |
| 易于扩展 | 可加先验等 | 基础形式 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 需要知道T | 探索项依赖t | 用模糊上界 |
| 对边界噪声敏感 | 过度探索小差异 | 增大c |
| 不适合变分布 | 奖励分布变化时差 | PHE等方法 |
| 离散臂 | 连续动作困难 | 离散化 |

### 6.3 与同类算法对比

| 算法 | 复杂度 | 期望后悔 | 特点 |
|------|--------|----------|------|
| Epsilon-Greedy | O(1) | O(��T) | 简单但差 |
| UCB1 | O(K log T) | O(log T) | 标准版本 |
| KL-UCB | O(K log T) | O(log T) | 更紧 |
| Thompson Sampling | O(K) | O(log T) | 贝叶斯 |
| EXP3 | O(K log T) | O(√KT) | 对抗性 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
"""
UCB 调库实现
"""

import numpy as np
from typing import List, Tuple

class UCBBandit:
    """
    UCB1 多臂老虎机
    
    标准UCB算法，期望后悔O(log T)
    """
    
    def __init__(self, n_arms: int, c: float = 1.414):
        """
        初始化
        
        参数:
            n_arms: 臂的数量
            c: 探索常数，默认sqrt(2)
        """
        self.n_arms = n_arms
        self.c = c
        
        # 统计
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        
        # 初始化后拉取次数
        self.t = 0
        
    def _compute_ucb(self, arm: int) -> float:
        """
        计算UCB值
        
        参数:
            arm: 臂索引
            
        返回:
            UCB值
        """
        if self.counts[arm] == 0:
            # 未拉取过的臂返回无穷大
            return float('inf')
        
        # 平均奖励
        avg = self.sums[arm] / self.counts[arm]
        
        # 置信上界
        exploration = self.c * np.sqrt(2 * np.log(self.t) / self.counts[arm])
        
        return avg + exploration
    
    def select_arm(self) -> int:
        """
        选择臂
        
        返回:
            选中的臂索引
        """
        self.t += 1
        
        # 计算每个臂的UCB
        ucb_values = [self._compute_ucb(i) for i in range(self.n_arms)]
        
        # 选择最大的
        arm = np.argmax(ucb_values)
        return arm
    
    def update(self, arm: int, reward: float):
        """
        更新统计
        
        参数:
            arm: 被选中的臂
            reward: 获得的奖励
        """
        self.counts[arm] += 1
        self.sums[arm] += reward
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        total_counts = self.counts.sum()
        avg_rewards = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.counts[i] > 0:
                avg_rewards[i] = self.sums[i] / self.counts[i]
        
        return {
            'counts': self.counts,
            'avg_rewards': avg_rewards,
            'total_pulls': total_counts
        }


class UCBV(UCBBandit):
    """
    UCB-V 算法
    
    使用方差调整的UCB，适合噪声大的奖励
    """
    
    def __init__(self, n_arms: int, c: float = 1.414, v: float = 1.0):
        """
        初始化
        
        参数:
            n_arms: 臂数
            c: 探索常数
            v: 方差参数
        """
        super().__init__(n_arms, c)
        self.v = v
        
        # 方差相关
        self.squares = np.zeros(n_arms)
        
    def _compute_ucb_v(self, arm: int) -> float:
        """UCB-V公式"""
        if self.counts[arm] == 0:
            return float('inf')
        
        avg = self.sums[arm] / self.counts[arm]
        
        # 方差估计
        if self.counts[arm] > 1:
            var = max(0, self.squares[arm] / self.counts[arm] - avg ** 2)
        else:
            var = self.v
        
        # V-UCB项
        exploration = np.sqrt(self.v * 2 * np.log(self.t) * np.log(self.t) / 
                             self.counts[arm])
        
        avg + exploration
    
    def select_arm(self) -> int:
        """选择臂"""
        self.t += 1
        ucb_values = [self._compute_ucb_v(i) for i in range(self.n_arms)]
        self.t -= 1  # 下面的update会+1
        
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """更新"""
        self.t += 1
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.squares[arm] += reward ** 2


class KLUCB(UCBBandit):
    """
    KL-UCB算法
    
    使用KL散度推导更紧的界
    """
    
    def __init__(self, n_arms: int, c: float = 1.0):
        super().__init__(n_arms, c)
        
    def _kl(self, p: float, q: float) -> float:
        """KL散度"""
        if p <= 0 or p >= 1:
            return float('inf')
        return p * np.log(p / q) + (1-p) * np.log((1-p) / (1-q))
    
    def _solve_kl_ucb(self, arm: int) -> float:
        """解KL-UCB不等式"""
        if self.counts[arm] == 0:
            return float('inf')
        
        avg = self.sums[arm] / self.counts[arm]
        target = (np.log(self.t) + self.c * np.log(np.log(self.t))) / self.counts[arm]
        
        # 二分搜索
        lo, hi = avg, 1.0
        for _ in range(20):
            mid = (lo + hi) / 2
            if self._kl(avg, mid) < target:
                lo = mid
            else:
                hi = mid
        
        return hi
    
    def select_arm(self) -> int:
        self.t += 1
        
        kl_ucb = [self._solve_kl_ucb(i) if self.counts[i] > 0 
                  else float('inf') for i in range(self.n_arms)]
        
        return np.argmax(kl_ucb)


def simulate_ucb(n_arms: int = 5, n_rounds: int = 1000,
                true_probs: List[float] = None) -> dict:
    """
    模拟UCB
    
    参数:
        n_arms: 臂数
        n_rounds: 总轮数
        true_probs: 各臂的真实成功概率
        
    返回:
        模拟结果
    """
    if true_probs is None:
        np.random.seed(42)
        true_probs = np.random.random(n_arms)
        true_probs = true_probs / true_probs.sum() * 0.5
    
    # 创建bandit
    bandit = UCBBandit(n_arms)
    
    # 初始化：每个臂拉一次
    for arm in range(n_arms):
        reward = 1 if np.random.random() < true_probs[arm] else 0
        bandit.update(arm, reward)
    
    # 记录
    selected_arms = []
    rewards_list = []
    cumulative_rewards = []
    
    # 模拟
    for t in range(n_rounds - n_arms):
        arm = bandit.select_arm()
        reward = 1 if np.random.random() < true_probs[arm] else 0
        
        bandit.update(arm, reward)
        
        selected_arms.append(arm)
        rewards_list.append(reward)
        cumulative_rewards.append(sum(rewards_list))
    
    return {
        'selected_arms': selected_arms,
        'rewards': rewards_list,
        'cumulative_rewards': cumulative_rewards,
        'estimated_probs': bandit.sums / bandit.counts,
        'true_probs': true_probs,
        'counts': bandit.counts
    }


def demo():
    """演示"""
    print("=" * 50)
    print("UCB 演示")
    print("=" * 50)
    
    # 真实成功率
    true_probs = [0.2, 0.4, 0.6, 0.8]
    print(f"真实成功率: {true_probs}")
    
    # 模拟
    results = simulate_ucb(n_arms=4, n_rounds=1000, true_probs=true_probs)
    
    print(f"\n估计成功率: {results['estimated_probs']}")
    print(f"各臂选择次数: {results['counts']}")
    print(f"累计奖励: {results['cumulative_rewards'][-1]}")
    
    # 最优臂
    best_arm = np.argmax(true_probs)
    print(f"\n理论最优臂: {best_arm} (概率{true_probs[best_arm]})")
    print(f"最优臂被选择比例: {results['counts'][best_arm] / results['counts'].sum():.2%}")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
"""
UCB 手工实现
"""

import math
import random

class ManualUCB:
    """
    UCB手工实现
    
    不依赖numpy，仅用标准库
    """
    
    def __init__(self, n_arms: int, c: float = 1.414):
        """
        初始化
        
        参数:
            n_arms: 臂数
            c: 探索常数
        """
        self.n_arms = n_arms
        self.c = c
        
        # 统计：拉取次数、总奖励
        self.counts = [0] * n_arms
        self.sums = [0.0] * n_arms
        
        # 总轮数
        self.t = 0
    
    def _compute_ucb(self, arm: int) -> float:
        """计算UCB值"""
        if self.counts[arm] == 0:
            return float('inf')
        
        # 平均奖励
        avg = self.sums[arm] / self.counts[arm]
        
        # 置信上界
        exploration = self.c * math.sqrt(2 * math.log(self.t) / self.counts[arm])
        
        return avg + exploration
    
    def select_arm(self) -> int:
        """选择臂"""
        self.t += 1
        
        # 计算每个臂的UCB
        ucb_values = [self._compute_ucb(i) for i in range(self.n_arms)]
        
        # 选择最大的
        arm = ucb_values.index(max(ucb_values))
        return arm
    
    def update(self, arm: int, reward: float):
        """更新统计"""
        self.counts[arm] += 1
        self.sums[arm] += reward
    
    def get_estimates(self):
        """获取估计"""
        return [self.sums[i] / max(1, self.counts[i]) for i in range(self.n_arms)]


class SimpleUCB:
    """
    简化版UCB（用于理解原理）
    """
    
    def __init__(self, n_arms: int):
        self.n = n_arms
        self.counts = [1] * n  # 初始化为1避免除零
        self.sums = [0.5] * n  # 假设初始成功率0.5
        self.t = n
    
    def select(self):
        """选择"""
        values = []
        for i in range(self.n):
            avg = self.sums[i] / self.counts[i]
            explore = math.sqrt(2 * math.log(self.t) / self.counts[i])
            values.append(avg + explore)
        
        return values.index(max(values))
    
    def update(self, arm, reward):
        self.t += 1
        self.counts[arm] += 1
        self.sums[arm] += reward


def manual_demo():
    """手工实现演示"""
    print("=" * 50)
    print("UCB 手工实现演示")
    print("=" * 50)
    
    # 真实成功率
    random.seed(42)
    true_probs = [0.2, 0.4, 0.6, 0.8]
    print(f"真实成功率: {true_probs}")
    
    # 模拟
    ucb = ManualUCB(n_arms=4)
    
    # 初始化
    for arm in range(4):
        reward = 1 if random.random() < true_probs[arm] else 0
        ucb.update(arm, reward)
    
    # 模拟
    for _ in range(500):
        arm = ucb.select_arm()
        reward = 1 if random.random() < true_probs[arm] else 0
        ucb.update(arm, reward)
    
    print(f"\n估计成功率: {ucb.get_estimates()}")
    print(f"各臂选择次数: {ucb.counts}")


if __name__ == "__main__":
    manual_demo()
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_ucb():
    """UCB可视化"""
    
    np.random.seed(42)
    
    # 设置
    n_arms = 4
    n_rounds = 500
    true_probs = [0.15, 0.35, 0.55, 0.75]
    
    # 模拟
    results = simulate_ucb(n_arms=n_arms, n_rounds=n_rounds, true_probs=true_probs)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 各臂选择次数
    ax = axes[0, 0]
    ax.bar(range(n_arms), results['counts'], color='steelblue')
    ax.set_xlabel('Arm')
    ax.set_ylabel('Count')
    ax.set_title('Arm Selection Counts')
    ax.set_xticks(range(n_arms))
    
    # 2. 真实vs估计
    ax = axes[0, 1]
    x = np.arange(n_arms)
    width = 0.35
    ax.bar(x - width/2, results['true_probs'], width, label='True')
    ax.bar(x + width/2, results['estimated_probs'], width, label='Estimated')
    ax.set_ylabel('Success Rate')
    ax.set_title('True vs Estimated')
    ax.legend()
    
    # 3. 累计奖励
    ax = axes[1, 0]
    ax.plot(results['cumulative_rewards'], 'b-', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward')
    ax.grid(True, alpha=0.3)
    
    # 4. 探索项衰减
    ax = axes[1, 1]
    n_samples = np.arange(1, 101)
    exploration = np.sqrt(2 * np.log(n_samples) / n_samples)
    ax.plot(n_samples, exploration, 'r-', linewidth=2)
    ax.set_xlabel('Number of pulls')
    ax.set_ylabel('Exploration term')
    ax.set_title('Exploration term decay')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ucb_visualization.png', dpi=150)
    plt.show()


def compare_algorithms():
    """比较UCB和Thompson Sampling"""
    
    np.random.seed(42)
    
    # 设置
    n_arms = 3
    n_rounds = 500
    true_probs = [0.3, 0.5, 0.7]
    optimal = max(true_probs)
    
    # UCB
    ucb_results = simulate_ucb(n_arms=n_arms, n_rounds=n_rounds, true_probs=true_probs)
    
    # 计算Regret
    ucb_regret = np.cumsum([optimal - true_probs[r] for r in ucb_results['selected_arms']])
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(ucb_regret, 'b-', linewidth=2, label='UCB1')
    plt.plot([np.log(t+1) for t in range(n_rounds)], 'r--', 
            linewidth=2, label='O(log t)')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.title('UCB Regret')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ucb_regret.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    visualize_ucb()
    compare_algorithms()
```

**结果解读**：

1. **选择次数**：最优臂（0.75）被选最多，符合预期。
2. **估计vs真实**：估计值逐渐逼近真实值。
3. **探索项**：随拉取次数增加而指数衰减。

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 公式 |
|------|------|------|
| 累计奖励 | 总奖励 | Σr_t |
| 累计后悔 | 与最优差 | Σ(μ* - μ_a) |
| 期望后悔上界 | 理论保证 | O(K log T / Δ)|
| 臂选择分布 | 各臂被选比例 | count_k / T |

### 10.2 后悔上界

UCB1的期望后悔上界：
$$E[Regret(T)] \leq \sum_{k: \Delta_k > 0} \left( \frac{8 \log T}{\Delta_k} + \Delta_k \right)$$

---

## 11. 常见问题与易错点

### 11.1 问题1：探索常数c的选择

**原因**：c太大过度探索，c太小可能错过最优臂。

**解决方案**：c=sqrt(2)适用于标准设置，可调整。

```python
# 噪声大时增大c
bandit = UCBBandit(n_arms=5, c=2.0)
```

### 11.2 问题2：需要预先知道T

**原因**：UCB公式用log(t)，t未知时无法计算。

**解决方案**：用log(min(T, T_max) + 1)或用模糊上界。

### 11.3 问题3：初始阶段不稳定

**原因**：初期探索项大，可能选择次优臂。

**解决方案**：增大初始探索轮数或用小c。

---

## 12. 学习总结

### 核心要点回顾：

1. **UCB = 利用项 + 探索项**：利用历史信息+不确定性补偿
2. **探索项随拉取次数衰减**：用log(t)保证O(log T)后悔
3. **有理论保证**：期望后悔上界O(log T)
4. **确定性**：无随机性，结果可复现

### 从UCB到其他算法：

- UCB1 → KL-UCB（更紧的界）
- UCB1 → UCB-V（方差调整）
- UCB1 → UCB2（指数探索）
- UCB1 → Thompson Sampling（贝叶斯版本）

### 实践建议：

1. 默认用UCB1，c=sqrt(2)
2. 噪声大用UCB-V
3. 初始先每个臂拉一次

---

## 13. 练习题与思考题（含答案）

### 练习题

**习题1：基础计算**

问题：两臂Bandit，A平均奖励0.8（拉取10次），B平均奖励0.5（拉取10次），当前第21轮。计算两臂的UCB值（c=sqrt(2)）。

<details>
<summary>答案</summary>

t=20（已初始化）

A：
- 利用项 = 0.8
- 探索项 = sqrt(2) * sqrt(2*log(20) / 10) = 1.414 * sqrt(4*log(20)/10)
  = 1.414 * sqrt(1.297 / 10) = 1.414 * 0.36 = 0.51
- UCB = 0.8 + 0.51 = 1.31

B：
- 利用项 = 0.5
- 探索项 = 0.51（同样）
- UCB = 0.5 + 0.51 = 1.01

选择A。

</details>

**习题2：编程实践**

问题：用Python实现UCB1。

<details>
<summary>答案</summary>

```python
import math

class UCB1:
    def __init__(self, n_arms):
        self.n = n_arms
        self.counts = [0] * n
        self.sums = [0.0] * n
        self.t = 0
    
    def select(self):
        self.t += 1
        ucb = []
        for i in range(self.n):
            if self.counts[i] == 0:
                ucb.append(float('inf'))
            else:
                avg = self.sums[i] / self.counts[i]
                explore = math.sqrt(2 * math.log(self.t) / self.counts[i])
                ucb.append(avg + explore)
        return ucb.index(max(ucb))
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sums[arm] += reward
```

</details>

**习题3：理论推导**

问题：为什么UCB的探索项用log(t)而不是t？

<details>
<summary>答案</summary>

1. Hoeffding不等式：P(偏差>ε) ≤ 2exp(-2nε²)
2. 要使概率<1/t²，需要ε = sqrt(log(t²)/n) = sqrt(2log t / n)
3. 所以用log(t)：随时间增加但比t慢
4. 如果用log log t，探索太少；用log²t，探索太多

</details>

### 思考题

**思考题1**：UCB和Thompson Sampling哪个更好？

<details>
<summary>答案</summary>

1. 理论：期望后悔都是O(log T)，常数相近
2. 实现：UCB确定，TS随机
3. 扩展：TS更易扩展到 contextual
4. 实践：通常TS稍好但差异不大，视情况选择

</details>

**思考题2**：如何处理奖励分布随时间变化？

<details>
<summary>答案</summary>

1. 滑动窗口UCB：只使用最近N轮数据
2. 指数加权UCB：使用指数加权平均
3. discounted Thompson Sampling
4. 探索-利用切换：先探索后利用

</details>

---

## 14. 学习路径建议

### 初级阶段（掌握基础）

1. 理解多臂老虎机问题
2. 掌握UCB原理和公式
3. 手工计算简单例子
4. 实现代码

**学习时间**：1-2天

### 中级阶段（理解扩展）

1. 学习不同UCB变体
2. 理解后悔理论
3. 比较UCB和TS
4. 实践应用

**学习时间**：1周

### 高级阶段（扩展应用）

1. Contextual Bandit
2. 线性UCB
3. 生产系统部署
4. 最新研究

**学习时间**：2-3周

### 实践项目建议

1. **基础项目**：模拟老虎机实验
2. **进阶项目**：广告A/B测试系统
3. **挑战项目**：推荐系统EE问题

### 推荐资源

- **论文**：Auer et al. (2002). UCB
- **书籍**：《Bandit Algorithms》- Bubeck
- **课程**：Berkeley CS294-134

---

**文档结束**