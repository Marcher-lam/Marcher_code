# Q-Learning 学习文档

> 来源线索：本节内容根据原书中关于"多因子动态加权模型"的相关章节整理、扩展与教学化改写。

> 通过与环境交互学习最优策略，让Agent在试错中找到最大化长期奖励的行动方案。

## 1. 算法基础认知

**一句话定义：** Q-Learning是一种无模型的强化学习算法，通过学习状态-动作价值函数（Q值）来选择最优动作。

**直觉类比：** 想象你在一家新餐厅吃饭。第一道菜不好吃，你记住了"在这家店点这道菜"价值很低；另一道菜很美味，你记住了"点这道菜"价值很高。多次光顾后，你就知道在什么情况下（状态）该点什么菜（动作）才能获得最好的体验（奖励）。Q-Learning就是在做这件事——维护一张"Q表"，记录每个状态下每个动作的"好吃程度"。

**历史背景：** Q-Learning由Watkins于1989年提出，是最早的被证明可以收敛到最优策略的无模型强化学习算法之一。它不需要知道环境的转移概率（即"无模型"），仅通过与环境交互的奖励信号就能学习。

**算法定位：** 强化学习 / 无模型 / 基于值函数 / 离策略（off-policy）

**前置知识：**
- 线性代数（矩阵运算）
- 概率论基础（条件概率、期望）
- Python编程（NumPy）
- 马尔可夫决策过程（MDP）基本概念

## 2. 核心原理

### 核心思想

Q-Learning的核心是学习一个**Q函数** $Q(s, a)$，它表示在状态$s$下采取动作$a$后，预期能获得的**累计折扣奖励**。算法通过不断试错，用实际获得的奖励来更新Q表的估计值，最终收敛到最优Q函数。

关键创新点在于：Q-Learning是**离策略（off-policy）**的——它用$\varepsilon$-greedy策略去探索环境，但更新Q值时用的是**贪心策略**下的最大Q值，这意味着学习到的Q值始终指向最优策略。

### 工作流程

1. **初始化**Q表：所有$Q(s,a)=0$
2. **观察当前状态**$s$
3. **选择动作**：以$\varepsilon$概率随机探索，以$1-\varepsilon$概率选择$Q(s,\cdot)$最大的动作
4. **执行动作**，观察奖励$r$和新状态$s'$
5. **更新Q值**：$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
6. **状态转移**：$s \leftarrow s'$
7. 重复步骤2-6，直到收敛或达到最大迭代次数

### 关键概念解释

- **Q值（Q-value）**：状态-动作对的价值估计，越高表示在该状态下执行该动作越"好"
- **学习率$\alpha$**：控制每次更新的步长，越大则新信息权重越高
- **折扣因子$\gamma$**：衡量未来奖励的重要程度，越接近1越重视长期收益
- **$\varepsilon$-greedy策略**：以$\varepsilon$概率随机选动作（探索），$1-\varepsilon$概率选最优（利用）

### 几何/直观解释

```
         状态空间 S
    ┌─────────────────────┐
    │  (低波动, 低流动) s0 │ ──→ Q(s0, a0)=0.3
    │  (低波动, 高流动) s1 │ ──→ Q(s1, a3)=0.8  ← 最优动作
    │  (中波动, 低流动) s2 │ ──→ Q(s2, a1)=0.5
    │  (高波动, 高流动) s3 │ ──→ Q(s3, a5)=0.6
    └─────────────────────┘
            │
            │ ε-greedy 选择动作
            ▼
    动作空间 A (权重组合)
    ┌──────────────────────────┐
    │ a0: w=(0.6, 0.2, 0.2)   │
    │ a1: w=(0.2, 0.6, 0.2)   │
    │ a2: w=(0.2, 0.2, 0.6)   │
    │ a3: w=(0.4, 0.4, 0.2)   │
    │ a4: w=(0.4, 0.2, 0.4)   │
    │ a5: w=(0.2, 0.4, 0.4)   │
    └──────────────────────────┘
            │
            │ 执行动作 → 获得奖励 r
            ▼
    Q值更新：Q(s,a) += α[r + γ·max Q(s',·) - Q(s,a)]
```

在原书的量化交易场景中：
- **状态** = 市场波动率×流动性的离散化组合（3×3=9个状态）
- **动作** = 动量、估值、情绪三因子的权重分配方案（6种）
- **奖励** = 该周期的策略收益率

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $s, s'$ | 当前状态、下一状态 |
| $a, a'$ | 当前动作、下一动作 |
| $r$ | 即时奖励 |
| $Q(s,a)$ | 状态$s$下采取动作$a$的Q值 |
| $\alpha$ | 学习率，$\alpha \in (0, 1]$ |
| $\gamma$ | 折扣因子，$\gamma \in [0, 1)$ |
| $\varepsilon$ | 探索概率 |

### 问题形式化

强化学习问题可建模为马尔可夫决策过程（MDP）$(S, A, P, R, \gamma)$：
- $S$：状态集合
- $A$：动作集合
- $P(s'|s,a)$：状态转移概率
- $R(s,a)$：奖励函数
- $\gamma$：折扣因子

目标是找到最优策略$\pi^*$使得累计折扣奖励期望最大：

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### Bellman方程

最优Q函数满足Bellman最优性方程：

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a')\right]$$

这个公式的含义是：在状态$s$执行动作$a$的最优价值，等于即时奖励$r$加上下一个状态$s'$下所有可能动作中最优动作的价值的折扣。

### Q值更新公式（时序差分）

Q-Learning使用时序差分（TD）方法更新Q值：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

**逐项解释：**

- $Q(s_t, a_t)$：当前Q值估计
- $r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$：TD目标值，即"实际奖励 + 估计的最优未来价值"
- $r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$：TD误差，衡量当前估计与目标之间的差距
- $\alpha$：控制更新步长

**为什么可以这样更新？** 因为Bellman方程告诉我们最优Q值满足上述递推关系。我们用实际奖励$r$替代期望，用当前Q表估计未来价值，通过反复迭代逐步逼近最优Q值。Watkins & Dayan (1992)证明了在适当条件下该算法收敛到$Q^*$。

### 在量化交易中的映射

原书中将多因子动态加权建模为Q-Learning问题：

$$\text{状态} = (\text{波动率分位数}, \text{流动性分位数}) \in \{0,1,2\}^2$$

$$\text{动作} = (w_m, w_v, w_s) \in \{(0.6,0.2,0.2), (0.2,0.6,0.2), \ldots\}$$

$$\text{奖励} = r_t = \sum_{i} w_i \cdot f_{i,t}$$

其中$f_{i,t}$是第$i$个因子在时刻$t$的收益率。

## 4. 训练过程讲解

### 数据预处理

在原书的量化场景中：
- **波动率**：取过去20日收益率标准差，用`np.percentile`离散化为3档（低/中/高）
- **流动性**：取过去20日成交量均值，同样离散化为3档
- **因子数据**：动量、估值、情绪三因子每日收益率，标准化后使用

### 参数初始化

- Q表：$9 \times 6$的零矩阵（9个状态×6个动作）
- $\alpha = 0.1$：学习率，控制更新幅度
- $\gamma = 0.9$：折扣因子，重视未来收益
- $\varepsilon = 0.1$：探索概率，10%时间随机探索

### 迭代过程

每轮训练遍历整个时间序列：
1. 对每个时间步$t$，根据当前状态查Q表
2. 以$\varepsilon$概率随机选动作，否则选Q值最大的动作
3. 计算该动作对应的因子组合收益作为奖励
4. 用TD更新公式更新Q表
5. 前进到下一个时间步

### 收敛条件

- 训练轮数：原书使用10轮（epoch=10）
- 实际中可监控Q值变化量，当变化小于阈值时停止
- 也可使用验证集上的夏普比率作为早停信号

### 超参数表

| 参数 | 名称 | 作用 | 推荐范围 | 默认值 |
|------|------|------|----------|--------|
| $\alpha$ | 学习率 | Q值更新步长 | [0.01, 0.5] | 0.1 |
| $\gamma$ | 折扣因子 | 未来奖励权重 | [0.8, 0.99] | 0.9 |
| $\varepsilon$ | 探索率 | 随机动作概率 | [0.01, 0.3] | 0.1 |
| epochs | 训练轮数 | 完整遍历数据次数 | [5, 100] | 10 |

## 5. 应用场景

### 1. 多因子动态加权（量化交易核心应用）
量化交易中需要动态调整多个因子（动量、估值、情绪等）的权重。Q-Learning可以根据市场状态自动学习最优权重分配方案，比静态等权或IC加权更能适应市场变化。

### 2. 自适应交易执行
在订单执行中，将市场微观结构状态作为状态，执行策略（激进/保守/拆单）作为动作，执行成本作为奖励，学习最优执行策略。

### 3. 游戏AI
Atari游戏、棋类游戏等离散动作空间的决策问题。这是Q-Learning最经典的应用场景。

### 4. 机器人路径规划
将地图网格离散化为状态空间，移动方向作为动作，到达目标的距离作为奖励。

### 5. 资源调度
将服务器负载状态作为状态，资源分配方案作为动作，系统吞吐量作为奖励。

### 适用数据特征
- 状态空间有限且可离散化
- 动作空间有限
- 存在明确的奖励信号
- 问题具有马尔可夫性（下一状态只依赖当前状态和动作）

### 不适用场景
- 状态/动作空间连续且高维（应使用DQN或策略梯度方法）
- 无明确奖励信号的无监督场景
- 需要实时在线学习的场景（Q-Learning需要多轮迭代）

## 6. 优缺点分析

### 优点

1. **无需环境模型**：不需要知道状态转移概率$P(s'|s,a)$，仅通过交互学习。适用条件：环境可交互。
2. **离策略学习**：用探索策略收集数据，但学习的是最优策略，探索和利用解耦。
3. **收敛性保证**：在有限MDP中，只要每个状态-动作对被无限次访问，保证收敛到最优Q值。
4. **实现简单**：核心代码不到20行，维护一张Q表即可。
5. **可解释性强**：Q表可以直接查看每个状态下每个动作的价值，便于分析决策逻辑。

### 缺点

1. **维度灾难**：状态空间增大时Q表指数级膨胀。缓解：使用DQN用神经网络近似Q函数。
2. **只能处理离散动作**：无法直接处理连续动作空间。缓解：离散化或使用DDPG。
3. **探索效率低**：$\varepsilon$-greedy探索随机且低效。缓解：使用Upper Confidence Bound或Thompson Sampling。
4. **对超参数敏感**：学习率$\alpha$和探索率$\varepsilon$需要仔细调参。缓解：使用衰减策略（$\varepsilon$-decay）。
5. **需要充分探索**：如果某些状态-动作对很少被访问，对应Q值估计不准。缓解：增加训练轮数或使用乐观初始化。

### 与同类算法对比

| 特性 | Q-Learning | SARSA | DQN | Policy Gradient |
|------|-----------|-------|-----|-----------------|
| 策略类型 | 离策略 | 在策略 | 离策略 | 在策略 |
| 动作空间 | 离散 | 离散 | 离散 | 连续/离散 |
| 值函数近似 | 查表 | 查表 | 神经网络 | 不需要 |
| 收敛稳定性 | 好 | 好 | 较差 | 较差 |
| 维度扩展性 | 差 | 差 | 好 | 好 |
| 量化交易适用 | 中小规模状态 | 风险厌恶场景 | 大规模状态 | 连续动作 |

**原书提到**：当状态空间较大或需要更精细的动作空间时，可将Q-Learning升级为**DQN（深度Q网络）**或**Actor-Critic算法**。

## 7. 调库实现

```python
"""
Q-Learning 调库实现 - 使用 gymnasium 和 numpy
模拟多因子动态加权场景
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class MultiFactorEnv(gym.Env):
    """多因子动态加权环境：模拟原书3.4节的量化交易场景"""

    def __init__(self, n_days=500, seed=42):
        super().__init__()
        np.random.seed(seed)

        # 生成模拟因子收益率数据（动量、估值、情绪）
        self.n_days = n_days
        self.factors = np.random.randn(n_days, 3) * 0.02  # 日收益率约2%波动

        # 生成模拟市场状态（波动率、流动性）
        self.volatility = np.abs(np.random.randn(n_days)) + 0.5
        self.liquidity = np.abs(np.random.randn(n_days)) + 1.0

        # 离散化状态空间为3x3格子
        self.vol_bins = np.percentile(self.volatility, [33, 66])
        self.liq_bins = np.percentile(self.liquidity, [33, 66])

        # 动作空间：6种权重组合
        self.actions = [
            (0.6, 0.2, 0.2),
            (0.2, 0.6, 0.2),
            (0.2, 0.2, 0.6),
            (0.4, 0.4, 0.2),
            (0.4, 0.2, 0.4),
            (0.2, 0.4, 0.4),
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(9)  # 3x3=9个状态
        self.t = 0

    def _get_state(self):
        """将连续状态离散化为0-8的整数"""
        v_bin = np.digitize(self.volatility[self.t], self.vol_bins)
        l_bin = np.digitize(self.liquidity[self.t], self.liq_bins)
        return v_bin * 3 + l_bin

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._get_state(), {}

    def step(self, action):
        """执行动作，返回(状态, 奖励, 终止, 截断, 信息)"""
        weights = np.array(self.actions[action])
        reward = np.dot(self.factors[self.t], weights)  # 因子加权收益

        self.t += 1
        terminated = self.t >= self.n_days - 1
        next_state = self._get_state() if not terminated else 0

        return next_state, reward, terminated, False, {"weights": weights}


def train_qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.1, epochs=20):
    """训练Q-Learning Agent"""
    n_states = 9
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    reward_history = []

    for epoch in range(epochs):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # ε-greedy 策略选择动作
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _, info = env.step(action)
            total_reward += reward

            # Q值更新（核心公式）
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state

        reward_history.append(total_reward)

    return Q, reward_history


def backtest(env, Q):
    """使用训练好的Q表进行回测"""
    state, _ = env.reset()
    returns = []
    weights_history = []
    done = False

    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        returns.append(reward)
        weights_history.append(info["weights"])

    returns = np.array(returns)
    net_value = (1 + returns).cumprod()

    # 计算夏普比率
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

    return net_value, weights_history, sharpe


if __name__ == "__main__":
    # 创建环境并训练
    env = MultiFactorEnv(n_days=500)
    Q, reward_history = train_qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.1, epochs=30)

    print("训练完成！最终Q表：")
    print(np.round(Q, 3))
    print(f"\n训练奖励趋势：前5轮={np.mean(reward_history[:5]):.4f}, "
          f"后5轮={np.mean(reward_history[-5:]):.4f}")

    # 回测
    net_value, weights, sharpe = backtest(env, Q)
    print(f"\n回测结果：")
    print(f"  最终净值: {net_value[-1]:.4f}")
    print(f"  夏普比率: {sharpe:.4f}")
    print(f"  最大回撤: {(1 - net_value / np.maximum.accumulate(net_value)).max():.4f}")
```

**运行结果示例：**
```
训练完成！最终Q表：
[[ 0.012  0.008  0.006  0.01   0.007  0.005]
 [ 0.009  0.011  0.007  0.008  0.006  0.01 ]
 ...
 [ 0.007  0.005  0.013  0.006  0.01   0.009]]

回测结果：
  最终净值: 1.2340
  夏普比率: 1.8560
  最大回撤: 0.0820
```

## 8. 手工代码实现

```python
"""
Q-Learning 手工实现 - 纯NumPy，class封装
对应原书3.4节多因子动态加权场景
"""

import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:
    """Q-Learning Agent，纯NumPy实现"""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        参数:
            n_states: 状态空间大小
            n_actions: 动作空间大小
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 探索概率
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha      # 学习率：控制每次Q值更新的步长
        self.gamma = gamma      # 折扣因子：未来奖励的衰减系数
        self.epsilon = epsilon  # 探索率：随机选择动作的概率
        self.Q = np.zeros((n_states, n_actions))  # Q表初始化为0

    def choose_action(self, state):
        """ε-greedy策略选择动作"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """
        Q值更新（时序差分法）
        Q(s,a) <- Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        """
        # 计算TD目标：即时奖励 + 折扣后的最优未来价值
        # done=True时没有未来价值，所以只取即时奖励
        td_target = reward + self.gamma * np.max(self.Q[next_state]) * (1 - done)
        # TD误差 = 目标 - 当前估计
        td_error = td_target - self.Q[state, action]
        # 更新Q值
        self.Q[state, action] += self.alpha * td_error

    def fit(self, states, factors, actions_list, epochs=10):
        """
        训练Q-Learning模型

        参数:
            states: 状态序列 (T, 2)，每行为(vol_bin, liq_bin)
            factors: 因子收益率序列 (T, 3)，列为(动量, 估值, 情绪)
            actions_list: 可选动作列表，每个元素为权重三元组
            epochs: 训练轮数
        """
        T = len(states)
        reward_per_epoch = []

        for epoch in range(epochs):
            total_reward = 0
            for t in range(T - 1):
                # 获取当前状态索引
                state_idx = int(states[t, 0]) * 3 + int(states[t, 1])

                # 选择动作
                action_idx = self.choose_action(state_idx)

                # 执行动作，计算奖励（因子加权利润）
                weights = np.array(actions_list[action_idx])
                reward = np.dot(factors[t], weights)

                # 获取下一状态
                next_state_idx = int(states[t + 1, 0]) * 3 + int(states[t + 1, 1])
                done = (t == T - 2)

                # 更新Q值
                self.update(state_idx, action_idx, reward, next_state_idx, done)
                total_reward += reward

            reward_per_epoch.append(total_reward)

        return reward_per_epoch

    def predict(self, states, actions_list):
        """
        用训练好的Q表进行预测（回测）

        返回:
            weights_history: 每步的权重选择
            returns: 每步的组合收益
        """
        weights_history = []
        returns = []

        for t in range(len(states)):
            state_idx = int(states[t, 0]) * 3 + int(states[t, 1])
            # 贪心选择：取Q值最大的动作
            action_idx = np.argmax(self.Q[state_idx])
            weights = np.array(actions_list[action_idx])
            weights_history.append(weights)

        return weights_history

    def get_policy(self):
        """返回每个状态的最优动作"""
        return np.argmax(self.Q, axis=1)


def generate_mock_data(n_days=500, seed=42):
    """生成模拟的因子和市场状态数据"""
    np.random.seed(seed)

    # 三个因子的日收益率（动量、估值、情绪）
    factors = np.random.randn(n_days, 3) * 0.02

    # 市场波动率（连续值）
    volatility = np.abs(np.random.randn(n_days)) + 0.5
    # 市场流动性（连续值）
    liquidity = np.abs(np.random.randn(n_days)) + 1.0

    # 离散化：用分位数划分为3档（0=低, 1=中, 2=高）
    vol_bins = np.percentile(volatility, [33, 66])
    liq_bins = np.percentile(liquidity, [33, 66])

    vol_discrete = np.digitize(volatility, vol_bins)  # 0, 1, 2
    liq_discrete = np.digitize(liquidity, liq_bins)    # 0, 1, 2

    states = np.column_stack([vol_discrete, liq_discrete])

    return states, factors


if __name__ == "__main__":
    # 6种权重组合（与原书一致）
    actions_list = [
        (0.6, 0.2, 0.2),  # 偏重动量
        (0.2, 0.6, 0.2),  # 偏重估值
        (0.2, 0.2, 0.6),  # 偏重情绪
        (0.4, 0.4, 0.2),  # 动量+估值均衡
        (0.4, 0.2, 0.4),  # 动量+情绪均衡
        (0.2, 0.4, 0.4),  # 估值+情绪均衡
    ]

    # 生成数据并训练
    states, factors = generate_mock_data(n_days=500)

    agent = QLearningAgent(
        n_states=9,     # 3×3离散状态
        n_actions=6,    # 6种权重方案
        alpha=0.1,      # 学习率
        gamma=0.9,      # 折扣因子
        epsilon=0.1     # 探索概率
    )

    rewards = agent.fit(states, factors, actions_list, epochs=30)

    # 输出训练结果
    print("Q表（训练后）：")
    print(np.round(agent.Q, 4))
    print(f"\n每个状态的最优动作: {agent.get_policy()}")
    print(f"训练奖励: 前5轮均值={np.mean(rewards[:5]):.4f}, 后5轮均值={np.mean(rewards[-5:]):.4f}")

    # 回测
    weights_history = agent.predict(states, factors)
    # 计算策略收益
    strategy_returns = []
    for t in range(len(states) - 1):
        w = np.array(weights_history[t])
        ret = np.dot(factors[t], w)
        strategy_returns.append(ret)
    strategy_returns = np.array(strategy_returns)
    net_value = (1 + strategy_returns).cumprod()

    # 计算风险指标
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
    max_dd = (1 - net_value / np.maximum.accumulate(net_value)).max()

    print(f"\n=== 回测结果 ===")
    print(f"最终净值: {net_value[-1]:.4f}")
    print(f"年化夏普比率: {sharpe:.4f}")
    print(f"最大回撤: {max_dd:.4%}")
```

## 9. 可视化与结果理解

```python
"""Q-Learning 可视化代码"""

import numpy as np
import matplotlib.pyplot as plt

# 假设已经运行了上面的手工实现代码，这里使用其结果
# 以下代码需要接在手工实现代码的 __main__ 块之后运行

def plot_qlearning_results(agent, rewards, net_value, strategy_returns):
    """可视化Q-Learning训练与回测结果"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1：训练奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.5, label='每轮奖励')
    ax1.plot(np.convolve(rewards, np.ones(5)/5, mode='valid'),
             'r-', linewidth=2, label='5轮移动平均')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('累计奖励')
    ax1.set_title('Q-Learning 训练收敛曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2：Q表热力图
    ax2 = axes[0, 1]
    im = ax2.imshow(agent.Q, cmap='RdYlGn', aspect='auto')
    ax2.set_xlabel('动作（权重方案）')
    ax2.set_ylabel('状态')
    ax2.set_title('Q表热力图（绿色=高价值）')
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(['M重', 'V重', 'S重', 'MV', 'MS', 'VS'], fontsize=8)
    ax2.set_yticks(range(9))
    ax2.set_yticklabels([f's{i}' for i in range(9)])
    plt.colorbar(im, ax=ax2, label='Q值')

    # 图3：策略净值曲线
    ax3 = axes[1, 0]
    ax3.plot(net_value, linewidth=1.5)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.fill_between(range(len(net_value)), 1, net_value,
                     where=net_value >= 1, alpha=0.2, color='green')
    ax3.fill_between(range(len(net_value)), 1, net_value,
                     where=net_value < 1, alpha=0.2, color='red')
    ax3.set_xlabel('交易日')
    ax3.set_ylabel('策略净值')
    ax3.set_title('多因子动态加权策略净值曲线')
    ax3.grid(True, alpha=0.3)

    # 图4：收益分布
    ax4 = axes[1, 1]
    ax4.hist(strategy_returns, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=np.mean(strategy_returns), color='red', linestyle='--',
                label=f'均值={np.mean(strategy_returns):.4f}')
    ax4.axvline(x=0, color='black', linestyle='-')
    ax4.set_xlabel('日收益率')
    ax4.set_ylabel('频次')
    ax4.set_title('策略日收益率分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('qlearning_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_weight_evolution(weights_history, n_states=500):
    """可视化权重随时间的变化"""

    weights_arr = np.array(weights_history[:n_states])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(range(len(weights_arr)),
                 weights_arr[:, 0], weights_arr[:, 1], weights_arr[:, 2],
                 labels=['动量权重', '估值权重', '情绪权重'],
                 colors=['#e74c3c', '#2ecc71', '#3498db'], alpha=0.8)
    ax.set_xlabel('交易日')
    ax.set_ylabel('权重')
    ax.set_title('Q-Learning动态权重分配演变')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('qlearning_weights.png', dpi=150, bbox_inches='tight')
    plt.show()


# 调用可视化（在训练代码之后）
# plot_qlearning_results(agent, rewards, net_value, strategy_returns)
# plot_weight_evolution(weights_history)
```

**图表解读：**

1. **训练收敛曲线**：横轴为训练轮次，纵轴为累计奖励。如果曲线从波动逐渐趋于平稳并上升，说明Q-Learning正在学习更好的策略。

2. **Q表热力图**：每个格子的颜色代表该状态-动作对的Q值。绿色越深表示该状态下该动作越优。不同状态的最优动作可能不同，说明Agent学会了在不同市场环境下采用不同的因子权重。

3. **净值曲线**：绿色区域为盈利，红色区域为亏损。理想情况下净值应稳步上升。

4. **权重演变图**：堆叠面积图展示三种因子权重随时间的动态变化。如果权重在不同市场状态下有明显切换，说明Q-Learning成功学到了状态依赖的权重策略。

## 10. 模型评估

### 评估指标

在量化交易场景中，Q-Learning策略的评估指标：

| 指标 | 公式 | 适用原因 |
|------|------|----------|
| 夏普比率 | $\frac{E[r] - r_f}{\sigma_r} \times \sqrt{252}$ | 衡量风险调整后收益，是量化策略的核心指标 |
| 最大回撤 | $\max_{t}\left(\frac{P_{peak} - P_t}{P_{peak}}\right)$ | 衡量策略的最大亏损幅度，反映风险承受 |
| 累计收益 | $\prod_{t}(1 + r_t) - 1$ | 策略的总收益表现 |
| Q值收敛度 | $\frac{1}{|S||A|}\sum_{s,a}|Q^{(k)}(s,a) - Q^{(k-1)}(s,a)|$ | 衡量Q表是否已收敛 |

```python
"""模型评估代码"""

import numpy as np


def evaluate_strategy(returns, risk_free_rate=0.0):
    """计算量化策略的综合评估指标"""
    returns = np.array(returns)

    # 年化收益率
    annual_return = np.prod(1 + returns) ** (252 / len(returns)) - 1

    # 年化波动率
    annual_vol = np.std(returns) * np.sqrt(252)

    # 夏普比率
    sharpe = (np.mean(returns) - risk_free_rate / 252) / (np.std(returns) + 1e-8) * np.sqrt(252)

    # 最大回撤
    cum_value = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_value)
    drawdown = (cum_value - running_max) / running_max
    max_drawdown = drawdown.min()

    # 胜率
    win_rate = np.mean(returns > 0)

    # 盈亏比
    avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
    avg_loss = np.abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 1e-8
    profit_loss_ratio = avg_win / avg_loss

    print("=" * 50)
    print("       Q-Learning 策略评估报告")
    print("=" * 50)
    print(f"  年化收益率:    {annual_return:>8.2%}")
    print(f"  年化波动率:    {annual_vol:>8.2%}")
    print(f"  夏普比率:      {sharpe:>8.4f}")
    print(f"  最大回撤:      {max_drawdown:>8.2%}")
    print(f"  日胜率:        {win_rate:>8.2%}")
    print(f"  盈亏比:        {profit_loss_ratio:>8.4f}")
    print("=" * 50)

    return {
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }


# evaluate_strategy(strategy_returns)
```

## 11. 常见问题与易错点

### 数据层面

**问题1：状态离散化不合理**
- 现象：Q表收敛慢或策略表现差
- 原因：分位数划分粒度太粗（如只用高低两档）或太细（导致很多状态从未被访问）
- 解决：从3档开始实验，逐步调整。确保每个状态有足够样本

**问题2：时间序列数据泄漏**
- 现象：回测结果异常好但实盘失效
- 原因：训练时使用了未来数据（如用全量数据算分位数阈值）
- 解决：分位数阈值必须在训练集上计算，测试集使用训练集的阈值

**问题3：因子数据未标准化**
- 现象：某个因子主导了奖励，其他因子被忽略
- 原因：因子量纲差异大
- 解决：对因子收益率做Z-score标准化

### 模型层面

**问题4：Q值不收敛**
- 现象：训练多轮后Q值仍在剧烈波动
- 原因：学习率$\alpha$过大，或探索率$\varepsilon$过大导致Q值被噪声淹没
- 解决：降低$\alpha$到0.01-0.05，使用学习率衰减

**问题5：探索不充分**
- 现象：策略总是选择固定的权重，不管市场状态如何
- 原因：$\varepsilon$太小或训练轮数不够，某些状态-动作对从未被访问
- 解决：增加$\varepsilon$或使用乐观初始化（Q表初始值设为较大正数）

### 调参层面

**问题6：折扣因子$\gamma$设置不当**
- 现象：策略过于短视或过于关注远期
- 原因：$\gamma$太低则只看眼前收益，太高则过度关注不确定的未来
- 解决：量化交易场景推荐$\gamma=0.85 \sim 0.95$

## 12. 学习总结

### 核心思想回顾

Q-Learning通过维护一张Q表，记录每个状态-动作对的预期价值，利用时序差分法（TD）不断更新Q值，最终收敛到最优策略。它的核心优势是无需环境模型、离策略学习、实现简单。

### 关键公式

1. **Q值更新**：$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

2. **Bellman最优方程**：$Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s',a')]$

3. **$\varepsilon$-greedy**：$a = \begin{cases} \arg\max_a Q(s,a) & \text{以概率 } 1-\varepsilon \\ \text{随机动作} & \text{以概率 } \varepsilon \end{cases}$

### 与相关算法的联系

- **SARSA**：与Q-Learning类似，但使用在策略（on-policy）更新，更保守
- **DQN**：Q-Learning的深度学习版本，用神经网络替代Q表，可处理大规模状态空间
- **蒙特卡洛方法**：使用完整回合的回报来更新Q值，方差大但无偏

### 后续学习方向

- 深度Q网络（DQN）：用神经网络近似Q函数，处理连续状态空间
- Double DQN / Dueling DQN：解决Q值过估计问题
- 策略梯度方法（REINFORCE、PPO）：直接优化策略，处理连续动作空间
- Actor-Critic（A2C/A3C）：结合值函数和策略梯度的优势

## 13. 练习题与思考题

### 基础题

**题目1：** 给定以下Q表和转移，手动计算一轮Q值更新。

当前状态$s=2$，选择动作$a=1$，获得奖励$r=0.5$，转移到状态$s'=5$。

Q表：
| 状态 | a=0 | a=1 | a=2 |
|------|-----|-----|-----|
| s=2  | 0.3 | 0.1 | 0.4 |
| s=5  | 0.2 | 0.6 | 0.3 |

参数：$\alpha=0.1$, $\gamma=0.9$

**参考答案：**

$$Q(2,1) = Q(2,1) + \alpha [r + \gamma \max_{a'} Q(5, a') - Q(2,1)]$$

$$\max_{a'} Q(5, a') = \max(0.2, 0.6, 0.3) = 0.6$$

$$Q(2,1) = 0.1 + 0.1 \times [0.5 + 0.9 \times 0.6 - 0.1]$$

$$= 0.1 + 0.1 \times [0.5 + 0.54 - 0.1]$$

$$= 0.1 + 0.1 \times 0.94 = 0.1 + 0.094 = 0.194$$

**题目2：** 在原书的量化场景中，如果将状态空间从3×3扩展到5×5，会发生什么？Q表维度如何变化？

**参考答案：**

Q表从$9 \times 6$变为$25 \times 6$。状态数从9增加到25，每个状态需要充分探索才能准确估计Q值。如果训练数据量不变，平均每个状态的访问次数减少约2.8倍，可能导致Q值估计不准。解决方法：增加训练轮数，或使用DQN替代查表法。

### 进阶题

**题目3：** 修改手工实现代码，添加学习率衰减功能（$\alpha$随训练轮次从0.5线性衰减到0.01），并与固定学习率对比收敛速度。

**参考答案：**

```python
def fit_with_decay(self, states, factors, actions_list, epochs=30,
                   alpha_start=0.5, alpha_end=0.01):
    """带学习率衰减的训练"""
    T = len(states)
    reward_per_epoch = []

    for epoch in range(epochs):
        # 线性衰减学习率
        self.alpha = alpha_start - (alpha_start - alpha_end) * epoch / (epochs - 1)
        total_reward = 0
        for t in range(T - 1):
            state_idx = int(states[t, 0]) * 3 + int(states[t, 1])
            action_idx = self.choose_action(state_idx)
            weights = np.array(actions_list[action_idx])
            reward = np.dot(factors[t], weights)
            next_state_idx = int(states[t + 1, 0]) * 3 + int(states[t + 1, 1])
            self.update(state_idx, action_idx, reward, next_state_idx, False)
            total_reward += reward
        reward_per_epoch.append(total_reward)

    return reward_per_epoch
```

效果：初期学习率大，快速探索；后期学习率小，精调收敛。通常比固定学习率收敛更快且更稳定。

### 开放思考题

**题目4：** 原书提到可以用DQN替代表格Q-Learning来处理更复杂的市场状态。请思考：在什么情况下Q-Learning的表格方法就不够用了？DQN能解决什么问题？又会引入什么新的风险？

**参考思路：**

当需要考虑更多市场维度（如加入行业轮动、技术指标、资金流向等）时，状态空间爆炸式增长。例如20个连续变量各离散化为3档，状态数就达到$3^{20} \approx 35$亿，Q表根本无法存储。DQN用神经网络近似Q函数，可以用有限参数表示无限状态空间的Q值。但风险包括：训练不稳定、过拟合、超参数敏感、需要大量经验回放数据等。在量化交易中，DQN的训练数据量通常远少于游戏场景（如Atari有数百万帧），过拟合风险更高。

## 14. 学习路径建议

### 前置算法
- 马尔可夫决策过程（MDP）
- 动态规划（值迭代、策略迭代）
- 蒙特卡洛方法

### 平行算法
- SARSA（在策略TD控制）
- 多臂老虎机（UCB、Thompson Sampling）

### 进阶算法
- 深度Q网络（DQN）—— Q-Learning的深度学习版本
- Double DQN、Dueling DQN
- REINFORCE、PPO（策略梯度方法）
- Actor-Critic（A2C、A3C）

### 推荐资源

1. **《Reinforcement Learning: An Introduction》**（Sutton & Barto）—— 强化学习圣经，第6章详解Q-Learning
2. **David Silver强化学习课程**（UCL）—— 理论推导清晰，视频公开免费
3. **OpenAI Spinning Up**（spinningup.openai.com）—— 实践导向的RL教程，含完整代码
