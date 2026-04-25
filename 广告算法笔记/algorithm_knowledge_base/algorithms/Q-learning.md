# Q-learning 学习文档

## 1. 算法基础认知

Q-learning 是无模型（model-free）的离线策略（off-policy）强化学习算法，由 Watkins 于 1989 年提出。它直接学习最优 Q 函数而不需要环境模型，是强化学习最基础的算法之一。在广告系统中用于离散出价档位选择。

## 2. 核心原理

Q 函数表示在状态 s 执行动作 a 后，最优策略下的期望累计折扣回报：

$$Q^*(s, a) = \mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k+1} \mid s_t=s, a_t=a\right]$$

Q-learning 的更新规则：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[r_{t+1} + \gamma\max_{a'}Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

其中：
- α 是学习率
- γ 是折扣因子
- r_{t+1} + γ max Q(s',a') 是 TD 目标
- r_{t+1} + γ max Q(s',a') - Q(s,a) 是 TD 误差 δ

最优策略从 Q 表直接得到：

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

## 3. 数学公式与推导

### 收敛性

Q-learning 的 TD 误差：

$$\delta_t = r_{t+1} + \gamma\max_{a'}Q(s_{t+1}, a') - Q(s_t, a_t)$$

在满足以下条件时 Q-learning 收敛到 Q*：
1. 所有 (s,a) 被无限次访问
2. 学习率 Σα_t = ∞, Σα_t² < ∞
3. γ < 1 或回合有限

### ε-greedy 探索

$$a_t = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_a Q(s_t, a) & \text{with probability } 1-\epsilon \end{cases}$$

广告出价的奖励设计：

$$r_t = \alpha \cdot \text{Conversions}_t - \beta \cdot \max(0, \text{CPA}_t - \text{CPA}_{target})$$

## 4. 训练过程讲解

1. 初始化 Q 表：Q(s,a) = 0 对所有 s,a
2. 每个回合：
   - 初始化状态 s
   - 对每步：用 ε-greedy 选动作 a
   - 执行动作，观察 r, s'
   - 更新：Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)]
   - s ← s'
3. 衰减 ε：ε ← ε × decay
4. 重复直到收敛

## 5. 应用场景

- 广告离散出价档位选择
- 路径规划与迷宫
- 资源调度
- 任何离散状态/动作空间的决策问题
- 作为 DQN 的理论基础

## 6. 优缺点分析

**优点**：
- 无模型，不需要环境动力学
- 离线策略，可利用历史数据
- 理论保证收敛到最优
- 实现简单直观

**缺点**：
- 只能处理离散小规模状态空间
- 状态/动作空间大时 Q 表爆炸
- 收敛速度慢
- 不适用于连续状态/动作

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions, lr=0.1, gamma=0.99, epsilon=0.3):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def discretize_state(self, state):
        return tuple((np.array(state) * 10).astype(int))

    def select_action(self, state):
        s = self.discretize_state(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[s]))

    def update(self, state, action, reward, next_state, done):
        s = self.discretize_state(state)
        s_next = self.discretize_state(next_state)
        best_next = np.max(self.q_table[s_next])
        td_target = reward + (1 - done) * self.gamma * best_next
        td_error = td_target - self.q_table[s][action]
        self.q_table[s][action] += self.lr * td_error

    def decay_epsilon(self, decay=0.995, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay)

def train_bidding(env, agent, n_episodes=1000):
    rewards_history = []
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        rewards_history.append(total_reward)
    return rewards_history
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class QLearningFromScratch:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 0.3
        self.n_actions = n_actions

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, done):
        best_q_next = np.max(self.Q[s_next]) if not done else 0
        td_target = r + self.gamma * best_q_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.lr * td_error
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

    def get_value(self):
        return np.max(self.Q, axis=1)
```

## 9. 可视化与结果理解

- **Q 值热力图**：横轴动作，纵轴状态，颜色表示 Q 值大小
- **学习曲线**：回合奖励随训练轮次的增长
- **ε 衰减曲线**：从高探索到高利用的过渡
- **最优策略可视化**：每个状态的最优动作
- **TD 误差分布**：应逐步趋近于零

## 10. 模型评估

- **平均回合奖励**：策略整体性能
- **最优 Q 值收敛**：Q 值变化量应逐步趋近于零
- **策略稳定性**：策略在足够多轮次后不再变化
- **ε=0 时的性能**：纯贪婪策略的表现

## 11. 常见问题与易错点

- **状态离散化粒度不当**：太粗丢失信息，太细导致 Q 表过大
- **学习率 α 不衰减**：后期应减小学习率以稳定收敛
- **ε 衰减过快**：探索不足，Q 值不完整
- **γ 设为 1**：可能导致 Q 值不收敛（无限回合时）
- **max 操作导致过估计**：这是 Q-learning 的固有偏差，Double Q-learning 可缓解

## 12. 学习总结

Q-learning 是强化学习最基础的算法，通过 TD 学习直接优化 Q 函数。它理论简洁、实现简单，但只能处理离散小规模状态空间。理解 Q-learning 是学习 DQN、Double DQN 等深度强化学习算法的基础。

## 13. 练习题与思考题（含答案）

**Q1**：Q-learning 为什么是 off-policy 的？

A1：Q-learning 的更新使用 max Q(s',a')，与行为策略（ε-greedy）无关。它学习的是最优策略的 Q 值，而行为策略仅用于收集数据。

**Q2**：广告出价中如何将连续状态离散化？

A2：将每个状态维度分桶。例如预算消耗率 [0,0.2,0.4,0.6,0.8,1.0] 分 5 档，时间进度分 10 档，CPA 分 5 档，组合后状态数 = 5×10×5 = 250。

**Q3**：Q-learning 与 SARSA 的区别是什么？

A3：Q-learning 用 max Q(s',a') 作为目标（off-policy），SARSA 用 Q(s',a')（a' 由行为策略选出，on-policy）。SARSA 更保守但更稳定，Q-learning 更激进但可能过估计。

## 14. 学习路径建议

前置知识：MDP → 动态规划 → TD 学习
进阶方向：SARSA → Double Q-learning → DQN → Rainbow
