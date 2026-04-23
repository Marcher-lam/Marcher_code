# SARSA 学习文档

## 1. 算法基础认知

SARSA 是一种**同策略（on-policy）**的时序差分控制算法，名字来源于其更新所需的五元组 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。由 Rummery 和 Niranjan 于 1994 年提出。

与 Q-learning（异策略）不同，SARSA 使用**实际执行的下一个动作**来更新 Q 值，因此它的行为策略和目标策略相同。这使得 SARSA 更"保守"——它会考虑探索带来的风险。

## 2. 核心原理

### 2.1 SARSA 更新规则

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

关键点：$A_{t+1}$ 是由**当前策略**在 $S_{t+1}$ 处选择的动作，而非最优动作。

### 2.2 $\epsilon$-Greedy 策略改进

SARSA 通常配合 $\epsilon$-greedy 策略使用：

$$\pi(a|s) = \begin{cases} 1 - \epsilon + \epsilon/|A| & \text{if } a = \arg\max_{a'} Q(s, a') \\ \epsilon/|A| & \text{otherwise} \end{cases}$$

根据策略改进定理，基于 $Q^\pi$ 的 $\epsilon$-greedy 策略保证不差于原策略。

### 2.3 GLIE 条件

SARSA 收敛到最优策略需要满足 GLIE（Greedy in the Limit with Infinite Exploration）条件：
1. 所有状态-动作对被无限次访问
2. 策略最终收敛到贪心策略（$\epsilon_t \to 0$）

常用设置：$\epsilon_t = 1/t$ 或 $\epsilon_t = \epsilon_0 / t$。

## 3. 数学公式与推导

### SARSA 的收敛性

在表格情况下，如果满足以下条件，SARSA 收敛到 $Q^*$：

1. 步长 $\alpha_t(s,a)$ 满足 Robbins-Monro 条件
2. 策略满足 GLIE 条件
3. 步长与策略无关：$\sum_t \alpha_t(s,a) \mathbf{1}[S_t=s, A_t=a]$ 收敛

### SARSA vs Q-learning 的对比

| 特性 | SARSA | Q-learning |
|------|-------|------------|
| 策略类型 | On-policy | Off-policy |
| 更新目标 | $R + \gamma Q(S', A')$ | $R + \gamma \max_{a'} Q(S', a')$ |
| 收敛目标 | 最优 $\epsilon$-greedy 策略 | 最优策略 |
| 风险意识 | 考虑探索风险 | 乐观估计 |

SARSA 的期望更新目标为：

$$\mathbb{E}_\pi[R + \gamma Q(S', A')] = R + \gamma \sum_{a'} \pi(a'|S') Q(S', a')$$

## 4. 训练过程讲解

1. **初始化**：$Q(s,a) = 0$，$\forall s, a$；选择初始 $\epsilon$
2. **对每个 episode**：
   - $S \leftarrow$ 初始状态
   - $A \leftarrow \epsilon\text{-greedy}(S, Q)$
   - **循环**：
     - 执行 $A$，观测 $R, S'$
     - $A' \leftarrow \epsilon\text{-greedy}(S', Q)$
     - $Q(S, A) \leftarrow Q(S, A) + \alpha[R + \gamma Q(S', A') - Q(S, A)]$
     - $S \leftarrow S'$，$A \leftarrow A'$
   - 直到 $S$ 为终止状态
3. **衰减 $\epsilon$**：$\epsilon \leftarrow \epsilon \times \text{decay}$

注意：$A'$ 在 $Q$ 更新**之前**就选定，这是 on-policy 的关键。

## 5. 应用场景

- **机器人导航**：在障碍物环境中学习安全路径（SARSA 的保守性有利于避开危险）
- **游戏 AI**：Atari 游戏中的实时策略学习
- **自动驾驶**：考虑安全约束的驾驶策略
- **资源调度**：云计算任务分配
- **广告竞价**：考虑探索成本的实时竞价策略

## 6. 优缺点分析

**优点**：
- 同策略学习，安全性更高（考虑探索风险）
- 收敛更稳定
- 实现简单直观
- 在有噪声/随机性的环境中表现更好

**缺点**：
- 学习速度慢于 Q-learning（受探索影响）
- 收敛到最优 $\epsilon$-greedy 策略而非最优策略
- 需要配合 $\epsilon$ 衰减策略
- 样本效率低于 off-policy 方法

## 7. 调库实现（Python）

```python
import numpy as np
import gym
from collections import defaultdict

class SARSAAgent:
    def __init__(self, n_actions, epsilon=0.1, alpha=0.5, gamma=1.0):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.Q[next_state][next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

def train_sarsa(env, num_episodes=2000):
    agent = SARSAAgent(
        n_actions=env.action_space.n,
        epsilon=0.1,
        alpha=0.5,
        gamma=1.0
    )
    rewards_history = []
    for ep in range(num_episodes):
        state = env.reset()
        action = agent.select_action(state)
        total_reward = 0.0
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            total_reward += reward
        rewards_history.append(total_reward)
        if (ep + 1) % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.95)
    return agent, rewards_history
```

## 8. 手工代码实现

```python
import numpy as np

class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def step(self, s, a, r, s_next, a_next, done):
        td_target = r + self.gamma * self.Q[s_next, a_next] * (1 - done)
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

    def train_episode(self, env):
        s = env.reset()
        a = self.epsilon_greedy(s)
        total_r = 0.0
        done = False
        while not done:
            s_next, r, done, _ = env.step(a)
            a_next = self.epsilon_greedy(s_next)
            self.step(s, a, r, s_next, a_next, done)
            s, a = s_next, a_next
            total_r += r
        return total_r
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

def plot_sarsa_results(rewards_history):
    window = 100
    smoothed = [np.mean(rewards_history[max(0, i-window):i+1]) for i in range(len(rewards_history))]
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (100-ep window)')
    plt.title('SARSA Training Progress')
    plt.grid(True)
    plt.savefig('sarsa_training.png', dpi=150)
```

经典 Cliff Walking 实验结果：SARSA 学到的路径会避开悬崖边缘（更安全），而 Q-learning 学到的最优路径紧贴悬崖（不考虑探索时掉下悬崖的风险）。

## 10. 模型评估

- **累计奖励曲线**：滑动平均奖励随 episode 的变化
- **策略收敛性**：最优动作选择比例是否趋于 1
- **与 Q-learning 对比**：在确定性/随机性环境中的表现差异
- **样本效率**：达到指定奖励水平所需的 episode 数

## 11. 常见问题与易错点

- **$A'$ 的选择时机**：必须在更新 $Q$ 之前用当前策略选择 $A'$，不能先更新再选
- **混淆 on-policy 和 off-policy**：SARSA 用的是实际选择的 $A'$，不是 $\arg\max$
- **$\epsilon$ 衰减过快**：导致探索不足，陷入局部最优
- **$\alpha$ 过大**：在函数近似下可能导致发散
- **done 状态的 Q 值**：终止状态的 Q 值应为 0，注意 `(1 - done)` 处理

## 12. 学习总结

SARSA 是最经典的 on-policy TD 控制算法。它通过使用实际执行的动作进行更新，天然地考虑了探索策略的影响，使得学到的策略更安全稳健。理解 SARSA 与 Q-learning 的区别（on-policy vs off-policy）是掌握强化学习算法设计的关键一步。

## 13. 练习题与思考题

**Q1**：在 Cliff Walking 环境中，为什么 SARSA 的在线表现优于 Q-learning？

> **答案**：因为 Q-learning 学习最优路径（紧贴悬崖），但执行时 $\epsilon$-greedy 探索可能掉下悬崖。SARSA 考虑了探索动作的影响，学到避开悬崖的路径，实际执行时更安全。

**Q2**：SARSA 的 TD 目标 $\mathbb{E}[R + \gamma Q(S', A')]$ 展开后是什么？

> **答案**：$R + \gamma \sum_a \pi(a|S') Q(S', a)$，即当前策略下 $S'$ 处的期望 Q 值，而非最大 Q 值。

**Q3**：如何将 SARSA 扩展到多步版本（n-step SARSA）？

> **答案**：用 n-step 回报 $G_{t:t+n} = \sum_{k=0}^{n-1}\gamma^k R_{t+k+1} + \gamma^n Q(S_{t+n}, A_{t+n})$ 替代 1-step 目标，多步 SARSA 可以结合 $\lambda$ 构成 SARSA($\lambda$)。

## 14. 学习路径建议

1. **前置知识**：TD 预测、Q-learning、$\epsilon$-greedy 策略
2. **本节掌握**：SARSA 更新规则、on-policy 控制流程、GLIE 条件
3. **进阶方向**：
   - SARSA($\lambda$)：结合资格迹
   - Expected SARSA：用期望代替采样
   - N-step SARSA
4. **后续学习**：DQN、策略梯度方法、Actor-Critic
