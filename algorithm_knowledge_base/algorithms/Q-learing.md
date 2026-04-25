# Q-learning 学习文档

## 1. 算法基础认知

Q-learning是强化学习中**最经典的算法**之一，由Watkins于1989年提出，是无模型（model-free）离策略（off-policy）学习的代表。Q-learning的核心思想是学习一个动作价值函数Q(s,a)，表示在状态s下采取动作a能够获得的累积奖励期望。通过迭代更新Q表，Q-learning能够找到最优的策略，无需环境模型。

Q-learning的无模型特性使其特别适合于状态和动作空间较小、离散的问题。在这些问题中，Q表可以显式存储，算法简单高效。Q-learning是后续众多强化学习算法（如Deep Q-Network、DQN等）的基础，理解Q-learning对于学习强化学习至关重要。

Q-learning的"离策略"特性意味着：它使用ε-greedy等行为策略探索环境，但学习的是最优策略的价值。这允许使用经验回放（experience replay）提高数据效率。

## 2. 核心原理

Q-learning的核心原理是**通过时间差分（TD）更新学习最优动作价值函数**。给定四元组(s, a, r, s')，Q-learning的更新规则为：

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

其中α是学习率，γ是折扣因子。TD目标 = r + γ × max Q(s', a')是当前估计的最优价值，TD误差 = TD目标 - 当前Q值。

Q-learning是离策略算法：使用当前Q值选择动作（行为策略），但更新时使用最优动作的值（目标策略）。这使得探索和利用可以同时进行。

## 3. 数学公式与推导

### 3.1 Q函数定义

$$Q^*(s,a) = \mathbb{E}[R_t | S_t=s, A_t=a] = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} | S_t=s, A_t=a\right]$$

Q*(s,a)是在状态s下采取动作a，然后遵循最优策略能够获得的期望累积折扣奖励。

### 3.2 最优性方程（Bellman最优方程）

$$Q^*(s,a) = \mathbb{E}[R_t | S_t=s, A_t=a] + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')$$

这是Q-learning试图满足的递归关系。右边的期望包含即时奖励和后续最优价值的折扣期望。

### 3.3 更新公式

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

- α：学习率，控制新信息的权重
- γ：折扣因子，考虑未来奖励的重要性
- TD目标：$r + \gamma \max_{a'} Q(s', a')$
- TD误差：TD目标 - 当前Q值

### 3.4 动作选择

$$\pi(s) = \arg\max_a Q(s,a)$$

最优策略是选择Q值最大的动作。行为策略常用ε-greedy：
- 以概率1-ε选择argmax Q(s,a)
- 以概率ε随机选择动作

### 3.5 收敛条件

Q-learning收敛的条件：
1. 所有状态-动作对被访问无限次
2. 学习率满足：∑α_t = ∞, ∑α_t^2 < ∞
3. 折扣因子γ ∈ [0, 1)

## 4. 训练过程讲解

Q-learning的训练过程非常简洁：

```
初始化Q(s,a)为小值
for episode in range(num_episodes):
    初始化状态s
    for step in range(max_steps):
        # 选择动作（ε-greedy）
        a = choose_action(s, Q, epsilon)
        
        # 执行动作，获得奖励和下一状态
        r, s' = env.step(a)
        
        # 计算TD目标
        target = r + gamma * max(Q(s'))
        
        # 更新Q值
        Q(s,a) = Q(s,a) + alpha * (target - Q(s,a))
        
        # 更新状态
        s = s'
        
        if s是终止状态:
            break
```

关键超参数：
| 超参数 | 作用 | 典型值 |
|--------|------|--------|
| α | 学习率 | 0.1-0.5 |
| γ | 折扣因子 | 0.9-0.99 |
| ε | 探索率 | 0.1-0.5 |
| episodes | 训练轮数 | 1000+ |

## 5. 应用场景

Q-learning主要应用场景包括：**网格世界游戏**，如迷宫、格子世界等经典强化学习环境；**控制问题**，如倒立摆、悬崖行走等简单控制任务；**库存管理**，优化库存决策；**调度问题**，任务调度、资源分配。近年来，Q-learning的深度版本（DQN）被应用于Atari游戏、围棋等复杂任务。

经典应用示例：
1. Grid World：最基本的强化学习环境
2. Frozen Lake：OpenAI Gym环境
3. Taxi-v3：出租车调度问题

## 6. 优缺点分析

Q-learning的优点包括：**简单实现**，核心只需要几行代码；**离策略**，可以高效利用历史数据；**理论保证**，在合适条件下收敛到最优策略；**无模型**，不需要环境动力学模型。缺点包括：**状态空间限制**，只适用于离散小状态空间；**维度灾难**，状态多时Q表爆炸；**探索不足**，ε-greedy可能探索不充分。

| 优点 | 说明 | 适用场景 |
|------|------|----------|
| 简单 | 实现简洁 | 学习入门 |
| 高效 | 离策略 | 数据受限 |
| 收敛 | 理论保证 | 小状态空间 |

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 维度灾难 | Q表爆炸 | 使用函数逼近 |
| 离散限制 | 不适用连续 | 离散化或DQN |

## 7. 调库实现（Python完整代码）

```python
import numpy as np
import gym
from collections import defaultdict


class QLearningAgent:
    """Q-Learning智能体"""
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount_factor=0.99, epsilon=0.1, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q表
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        """ε-greedy动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Q-learning更新"""
        current_Q = self.Q[state, action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD更新
        self.Q[state, action] = current_Q + self.lr * (target - current_Q)
        
        # 探索衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, num_episodes=10000, max_steps=100):
        """训练智能体"""
        rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.learn(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Epsilon = {self.epsilon:.4f}")
        
        return rewards
    
    def evaluate(self, env, num_episodes=100, max_steps=100):
        """评估智能体"""
        total_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = np.argmax(self.Q[state])
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)


class LinearQlearning:
    """线性函数逼近的Q-learning"""
    def __init__(self, n_actions, feature_dim, learning_rate=0.01, gamma=0.99):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.weights = np.random.randn(feature_dim, n_actions) * 0.01
    
    def predict(self, features):
        """预测Q值"""
        return features @ self.weights
    
    def choose_action(self, features):
        """动作选择"""
        Q_values = self.predict(features)
        return np.argmax(Q_values)
    
    def learn(self, features, action, reward, next_features, done):
        """更新"""
        current_Q = self.predict(features)[action]
        
        if done:
            target = reward
        else:
            next_Q = self.predict(next_features)
            target = reward + self.gamma * np.max(next_Q)
        
        # 梯度更新
        error = target - current_Q
        gradient = np.zeros_like(self.weights)
        gradient[:, action] = features
        self.weights += self.lr * error * gradient


def run_frozen_lake():
    """运行FrozenLake环境"""
    env = gym.make('FrozenLake-v1')
    
    print("=== Q-Learning on FrozenLake ===")
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.2
    )
    
    rewards = agent.train(env, num_episodes=5000)
    
    eval_reward = agent.evaluate(env, num_episodes=1000)
    print(f"\nEvaluation Average Reward: {eval_reward:.2f}")


def run_cliff_walking():
    """运行Cliff Walking环境"""
    env = gym.make('CliffWalking-v0')
    
    print("\n=== Q-Learning on CliffWalking ===")
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1
    )
    
    rewards = agent.train(env, num_episodes=5000)
    
    eval_reward = agent.evaluate(env, num_episodes=1000)
    print(f"\nEvaluation Average Reward: {eval_reward:.2f}")


if __name__ == '__main__':
    import gym
    run_frozen_lake()
    run_cliff_walking()
```

## 8. 手工代码实现

```python
import numpy as np
import matplotlib.pyplot as plt


def simple_q_learning():
    """简化版Q-learning"""
    n_states = 4
    n_actions = 2
    Q = np.zeros((n_states, n_actions))
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 1000
    
    for episode in range(num_episodes):
        s = 0
        while s != n_states - 1:
            if np.random.random() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = np.argmax(Q[s])
            
            if a == 0:
                s_new = max(0, s - 1)
            else:
                s_new = min(n_states - 1, s + 1)
            
            r = -1 if s_new == n_states - 1 else 0
            
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_new]) - Q[s, a]
            s = s_new
    
    return Q


def grid_world_q_learning():
    """网格世界Q-learning"""
    grid_size = 4
    n_states = grid_size * grid_size
    n_actions = 4
    
    Q = np.zeros((n_states, n_actions))
    
    transitions = {
        0: {0: 0, 1: 4, 2: 0, 3: 0},
        1: {0: 1, 1: 5, 2: 0, 3: 1},
        2: {0: 2, 1: 6, 2: 0, 3: 2},
        3: {0: 3, 1: 7, 2: 0, 3: 3},
        4: {0: 0, 1: 8, 2: 0, 3: 5},
        15: {0: 15, 1: 15, 2: 11, 3: 14}
    }
    
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    
    for _ in range(10000):
        s = 0
        
        while s != 15:
            if np.random.random() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = np.argmax(Q[s])
            
            s_new = transitions.get(s, {a: s}).get(a, s)
            
            r = 1 if s_new == 15 else 0
            
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_new]) - Q[s, a])
            s = s_new
    
    return Q


if __name__ == '__main__':
    Q = simple_q_learning()
    print("Q-table:")
    print(Q)
```

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_q_learning_curves():
    """绘制Q-learning训练曲线"""
    episodes = np.arange(1000)
    
    rewards_early = np.random.randn(1000).cumsum()
    rewards_mid = np.random.randn(900, 1).mean(axis=1).cumsum()
    rewards_late = np.full(100, 10).cumsum()
    
    rewards = np.concatenate([rewards_early[:100], rewards_mid, rewards_late])
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title('Q-Learning Training Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q_learning_curve.png', dpi=150)
    plt.show()


def plot_q_table_heatmap():
    """绘制Q表热力图"""
    Q = np.random.randn(16, 4) * 10
    
    plt.figure(figsize=(10, 6))
    plt.imshow(Q, aspect='auto', cmap='RdYlGn')
    plt.colorbar(label='Q Value')
    plt.title('Q-Table Heatmap', fontsize=14)
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig('q_table_heatmap.png', dpi=150)
    plt.show()


def plot_convergence():
    """绘制收敛过程"""
    episodes = np.arange(1, 10001)
    errors = np.exp(-0.0005 * episodes) + 0.1 * np.random.randn(len(episodes))
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(episodes, errors)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('TD Error', fontsize=12)
    plt.title('Q-Learning Convergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q_convergence.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    plot_q_learning_curves()
    plot_q_table_heatmap()
    plot_convergence()
```

结果分析：Q-learning通常在几百到几千个episode后收敛。训练曲线显示奖励逐渐增加，最终稳定在较高的值。TD误差逐渐减小到0，表示Q函数逐渐收敛到最优值。

## 10. 模型评估

Q-learning的评估关注：
1. **收敛速度**：多少episode后收敛
2. **最终性能**：收敛后的平均奖励
3. **稳定性**：不同随机种子下的表现
4. **样本效率**：达到性能所需的交互次数

评估指标：
1. Average Reward：平均episode奖励
2. TD Error：TD误差
3. Steps per Episode：每episode步数
4. Success Rate：成功完成任务的比例

## 11. 常见问题与易错点

常见问题：**Q值不收敛**，学习率过大或过小；**探索不足**，ε过小导致局部最优；**状态表示不当**，连续状态离散化不合理。使用时的易错点：**忘记探索��减**，固定ε导致性能下降；**折扣因子不当**，γ过大或过小；**终止状态处理错误**，忘记设置done标志。

## 12. 学习总结

Q-learning是强化学习的基础算法，通过TD学习最优动作价值函数。核心简单，效果显著。学习要点：TD学习、ε-greedy探索、离策略特性。

## 13. 练习题与思考题（含答案）

**练习题1**：写出Q-learning的更新公式。

答案：Q(s,a) ← Q(s,a) + α[r + γmax Q(s',a') - Q(s,a)]

**练习题2**：为什么Q-learning是离策略算法？

答案：使用ε-greedy探索，但学习的是最优策略的价值。

**思考题1**：Q-learning的局限性？

答案：1.只适用于离散小状态空间 2.维度灾难

### 13.3 详细答案与解析

#### 练习：计算

**问题**：状态0，执行动作1，到达状态1，获得奖励0，再执行动作1到达状态2，奖励1，γ=0.9，α=0.1，Q(0,1)=1, Q(1,1)=2, Q(2,1)=3，更新Q(0,1)。

**答案**：
```
target = r + γ * max(Q(s',:)) = 0 + 0.9 * 2 = 1.8
Q(0,1) = 1 + 0.1 * (1.8 - 1) = 1.08
```

## 14. 学习路径建议

学习Q-learning：
1. 强化学习基础
2. MDP和Bellman方程
3. TD学习
4. Q-learning实现
5. 扩展到DQN

### 14.1 扩展资源

**论文**：
1. Watkins (1989). "Learning from Delayed Rewards"
2. "Q-learning original paper"