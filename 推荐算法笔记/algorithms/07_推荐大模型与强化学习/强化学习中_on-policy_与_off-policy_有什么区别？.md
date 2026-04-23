# 面试题：强化学习中 on-policy 与 off-policy 有什么区别？

面试题：强化学习中 on-policy 与 off-policy 有什么区别？

强化学习中 on-policy 与 off-policy 的核心区别在于行为策略（生成数据的策略）与目标策略（被优化的策略）是否一致。

# 1. 基本定义

On-policy：

行为策略与目标策略完全一致，即智能体通过当前策略与环境交互生成数据，并直接使用这些数据更新同一策略。

示例：SARSA 算法中，下一动作 a′ 由当前策略选择，更新时使用 $Q ( s ^ { \prime } , \bar { a } ^ { \prime } )$

 Off-policy：行为策略与目标策略分离，即智能体通过其他策略（如历史策略、随机策略）生成数据，但用这些数据优化不同的目标策略。

示例：Q-learning 算法中，更新时采用最大 Q 值对应的动作（目标策略为贪婪策略），而数据可能来自 ε-greedy 策略（行为策略）。

# 2. 技术原理对比

<table><tr><td>维度</td><td>On-policy</td><td>Off-policy</td></tr><tr><td>策略更新</td><td>使用当前策略生成的轨迹 (s,a,r,s&#x27;,a&#x27;) 更新，如 SARSA</td><td>允许使用不同策略的轨迹，如 Q-learning</td></tr><tr><td>数学条件</td><td>策略生成的轨迹分布与目标策略分布一致</td><td>需满足覆盖性条件：目标策略的动作在行为策略中出现的概率非 0</td></tr><tr><td>重要性采样</td><td>无需调整数据分布</td><td>需通过重要性权重修正不同策略的分布差异</td></tr></table>

# 3. 优缺点分析

On-policy

优点：

 稳定性高：策略更新与数据生成同步，避免策略偏移（Policy Shift）；  
 实时适应性强：适合动态环境（如机器人实时控制）。

缺点：

 数据利用率低：旧数据因策略更新失效，需频繁重新采样；  
 探索受限：依赖当前策略，可能陷入局部最优。

Off-policy

 优点：

 数据复用性强：支持历史数据（如离线强化学习）与多策略数据融合（如经验回放）；  
 探索性更优：允许行为策略独立设计（如高风险探索）。

 缺点：

 训练不稳定：策略差异可能导致 Q 值高估或低估；  
 计算复杂度高：需处理重要性权重等额外计算。

# 4. 典型应用场景

On-policy：

 实时交互场景：如机器人导航、游戏实时对战（需快速适应环境变化）；  
 高安全要求任务：如自动驾驶（需避免策略突变带来的风险）。  
 典型算法：SARSA、A2C（Advantage Actor-Critic）、PPO（近端策略优化）。

Off-policy：

 离线学习：利用历史日志数据训练（如广告竞价策略优化）；  
 多策略协同：结合专家示范与随机探索（如机器人模仿学习）；  
 典型算法：Q-learning、DDPG（深度确定性策略梯度）、DQN（深度 Q 网络

# 形式化定义

## 马尔可夫决策过程（MDP）基础

强化学习通常建模为马尔可夫决策过程 $\langle S, A, P, R, \gamma \rangle$：

- $S$：状态空间
- $A$：动作空间
- $P(s'|s,a)$：状态转移概率
- $R(s,a)$：奖励函数
- $\gamma \in [0,1]$：折扣因子

策略 $\pi(a|s)$ 定义了在状态 $s$ 下选择动作 $a$ 的概率。强化学习的目标是找到最优策略 $\pi^*$ 最大化累积奖励的期望：

$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

## On-Policy 的形式化定义

在 on-policy 设置中，行为策略 $\mu$ 和目标策略 $\pi$ 相同（$\mu = \pi$）。策略梯度为：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

其中 $G_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ 是从时刻 $t$ 开始的累积回报，轨迹 $\tau$ 必须由当前策略 $\pi_\theta$ 生成。

## Off-Policy 的形式化定义

在 off-policy 设置中，行为策略 $\mu \neq \pi$。为了使用 $\mu$ 生成的数据来优化 $\pi$，需要引入重要性采样比率：

$$\rho_t = \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}$$

修正后的策略梯度为：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \mu}\left[\sum_{t=0}^{T} \rho_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

重要性权重 $\rho_t$ 修正了分布差异，但也引入了高方差问题。

## 覆盖性条件（Coverage Assumption）

Off-policy 学习有效的前提条件是：对于所有 $\pi(a|s) > 0$ 的 $(s,a)$ 对，都有 $\mu(a|s) > 0$。即行为策略必须"覆盖"目标策略可能采取的所有动作。如果某个动作在行为策略中概率为零，就无法从数据中学习该动作的价值。

## 算法对比表

| 算法 | 类型 | 策略类型 | 值函数 | 经验回放 | 重要性采样 |
|------|------|---------|--------|---------|-----------|
| SARSA | On-policy | 随机策略 | Q(s,a) | 否 | 否 |
| REINFORCE | On-policy | 随机策略 | 无 | 否 | 否 |
| A2C/A3C | On-policy | 随机策略 | V(s) | 否 | 否 |
| PPO | On-policy | 随机策略 | V(s) | 否 | 限幅IS（clip） |
| Q-Learning | Off-policy | 贪婪策略 | Q(s,a) | 可选 | 否（TD学习） |
| DQN | Off-policy | 贪婪策略 | Q(s,a) | 是 | 否 |
| DDPG | Off-policy | 确定性策略 | Q(s,a) | 是 | 否 |
| SAC | Off-policy | 随机策略 | 双Q网络 | 是 | 重参数化 |
| TD3 | Off-policy | 确定性策略 | 双Q网络 | 是 | 否 |
| ACER | Off-policy | 随机策略 | Q(s,a) | 是 | 截断IS |

## PPO 的特殊地位

PPO（Proximal Policy Optimization）是一个 on-policy 算法，但它通过重要性采样限幅技术，在一定程度上获得了 off-policy 的数据复用能力：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中 $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$。通过限幅，PPO 允许策略在一定范围内偏离数据生成策略，但不会偏离太远，兼顾了稳定性和数据利用率。

## 代码实现对比

### On-Policy：SARSA 实现

```python
import numpy as np


class SARSAAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, s, a, r, s_next, a_next):
        td_target = r + self.gamma * self.q_table[s_next, a_next]
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.lr * td_error

    def train_episode(self, env):
        state = env.reset()
        action = self.choose_action(state)
        total_reward = 0
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = self.choose_action(next_state)
            self.update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            total_reward += reward
        return total_reward
```

### Off-Policy：Q-Learning 实现

```python
import random
from collections import deque


class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1, buffer_size=10000, batch_size=32):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def store_transition(self, s, a, r, s_next, done):
        self.replay_buffer.append((s, a, r, s_next, done))

    def update_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        for s, a, r, s_next, done in batch:
            td_target = r if done else r + self.gamma * np.max(self.q_table[s_next])
            self.q_table[s, a] += self.lr * (td_target - self.q_table[s, a])

    def train_episode(self, env):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            self.store_transition(state, action, reward, next_state, done)
            self.update_batch()
            state = next_state
            total_reward += reward
        return total_reward
```

### 关键区别在代码中的体现

```python
sarsa_td_target = r + gamma * Q(s_next, a_next)
qlearn_td_target = r + gamma * max_a Q(s_next, a)
```

SARSA 使用**当前策略选择的下一个动作** $a'$ 来计算 TD 目标（on-policy），Q-Learning 使用**最优动作**来计算 TD 目标（off-policy，目标策略是贪婪策略）。

## 在推荐系统中的应用

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| 实时推荐策略调整 | On-policy（PPO） | 需要稳定地在线更新，避免推荐策略剧烈波动 |
| 离线推荐模型训练 | Off-policy（DQN/SAC） | 可以利用海量历史日志数据 |
| 推荐系统模拟器 | Off-policy | 使用历史数据训练评估模型 |
| 广告竞价优化 | Off-policy（BCQ/CQL） | 离线学习最优出价策略 |
| 信息茧房打破 | On-policy | 需要实时探索用户未知兴趣 |

## 何时选择 On-Policy vs Off-Policy

| 决策因素 | 选择 On-Policy | 选择 Off-Policy |
|---------|---------------|----------------|
| 数据获取成本 | 低（可频繁交互） | 高（需复用历史数据） |
| 安全性要求 | 高（策略变化需可控） | 低（允许离线大胆探索） |
| 环境动态性 | 高（环境快速变化） | 低（环境相对稳定） |
| 计算资源 | 有限（不能存储大量数据） | 充足（可维护经验回放池） |
| 样本效率要求 | 不高 | 高（数据稀缺） |

## 常见问题

1. **Q: 为什么 PPO 被归类为 on-policy 但用了重要性采样？**
   A: PPO 的重要性采样比率是在同一轮采样的数据上计算的（$\pi_{\theta_{old}}$ 就是采样时的策略），并且通过 clip 限制策略偏离幅度。本质上数据仍然来自当前策略，只是在微小的参数更新范围内做了一点"off-policy"的扩展。

2. **Q: Off-policy 中经验回放池大小如何设置？**
   A: 通常设置为 $10^5 \sim 10^6$ 条转换。过小会导致数据多样性不足，过大会导致旧数据与当前策略差距过大（分布偏移严重）。对于推荐系统场景，通常设置更大的回放池（$10^7 \sim 10^8$）。

3. **Q: 能否将 on-policy 算法改为 off-policy？**
   A: 理论上可以通过重要性采样将任何 on-policy 算法改为 off-policy，但重要性权重的高方差问题会导致训练不稳定。ACER（Actor-Critic with Experience Replay）就是这种思路的代表。

# 总结：

本质差异：数据生成与策略优化的耦合性。  
 选择依据：若需高稳定性与实时性，选 On-policy；若需数据复用与灵活探索，选 Off-policy。  
 趋势：工业场景（如广告、推荐系统）更倾向 Off-policy（尤其是离线强化学习），因其能复用历史数据并降低在线探索成本。
