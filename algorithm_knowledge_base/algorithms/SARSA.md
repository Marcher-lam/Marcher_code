# SARSA 学习文档

## 1. 算法基础认知

SARSA（State-Action-Reward-State-Action）是强化学习中**经典的时序差分算法**，由Rummery和Niranjan于1994年提出，是**在策略（on-policy）**学习的代表。与Q-learning不同，SARSA学习的是当前策略的动作价值函数，因此在训练过程中会考虑探索的影响。SARSA的核心思想是在线学习一个动作价值函数Q(s,a)，每次在状态下采取实际动作后立即更新Q值。

理解SARSA需要先理解在策略学习的概念：SARSA学习的行为策略和目标策略是同一个策略（通常是ε-greedy），这意味着SARSA会学习到在考虑 Exploration 情况下的实际策略价值，因此更适合于需要避免危险动作的实际应用。SARSA的名称来源于其更新使用的序列：(State, Action, Reward, Next State, Next Action)。

SARSA是Q-learning的"在策略"版本，两者形成对比。Q-learning是离策略（off-policy），学习最优策略价值；SARSA是在策略（on-policy），学习当前策略价值。

## 2. 核心原理

SARSA的核心原理是**在策略学习：学习当前行为策略下的动作价值函数**。给定转换(s, a, r, s', a')，SARSA的更新规则为：

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$$

其中α是学习率，γ是折扣因子，a'是下一个状态s'下根据当前策略选择的动作。关键区别于Q-learning的是：Q-learning使用max Q(s',a')作为TD目标，而SARSA使用实际的Q(s',a')。这使得SARSA学习的是"如果继续遵循当前策略"的真实价值，而不是假设选择最优动作的价值。

SARSA的特性使其更适合于以下情况：1）需要避免探索的损失，例如危险环境；2）实际部署时使用与训练相同的策略；3）需要学习探索成本的情况。

## 3. 数学公式与推导

### 3.1 SARSA更新规则

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

- α：学习率，控制更新步长
- γ：折扣因子，考虑未来奖励
- a_{t+1}：下一个状态下的实际动作（根据ε-greedy选择）

### 3.2 在策略性

SARSA学习的是当前行为策略π的价值：

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} | S_t=s, A_t=a\right]$$

这与离策略Q-learning的Q*对应对应但不同。

### 3.3 策略改进（重要性采样）

SARSA可以直接从当前策略产生的经验中学习，不需要行为策略和目标策略的分离：

$$a_t \sim \pi(\cdot | s_t)$$
$$a_{t+1} \sim \pi(\cdot | s_{t+1})$$

### 3.4 收敛条件

类似Q-learning，SARSA收敛需要：
1. 所有状态-动作对访问无限次
2. 学习率满足随机近似条件
3. 策略是逐渐贪婪的（ε逐渐衰减）

## 4. 训练过程讲解

SARSA的训练过程与Q-learning类似，但在选择动作时：

```
初始化Q(s,a)为小值
for episode in range(num_episodes):
    初始化状态s
    a = 选择动作（ε-greedy from Q）
    
    for step in range(max_steps):
        执行动作a，获得r, s'
        
        # 关键区别：选择下一个动作a'
        a' = 选择动作（ε-greedy from Q）
        
        # 使用(s', a')更新，而非max
        Q(s,a) += α * [r + γQ(s',a') - Q(s,a)]
        
        s = s'
        a = a'
        
        if s是终止状态:
            break
```

与Q-learning对比：
| Q-learning | SARSA |
|-----------|------|
| off-policy | on-policy |
| a' = argmax Q(s',:) | a' = ε-greedy(s') |
| 学习Q* | 学习Q^π |
| 探索成本低 | 探索计入成本 |

## 5. 应用场景

SARSA主要应用场景包括：**格子世界导航**，经典的强化学习示例；**在策略任务**，需要学习探索成本的情况；**安全关键系统**，如机器人控制，避免训练时不安全的动作。更典型的应用是：
1. 机器人路径规划
2. 游戏AI（需要考虑探索的场景）
3. 自动驾驶决策
4. 资源管理

实际应用示例：
1. Cliff Walking问题：学习绕过悬崖的最安全路径
2. 出租车调度：在策略场景下优化
3. 网格导航

## 6. 优缺点分析

SARSA的优点：**在策略学习**，学习当前策略的真实价值；**安全**，避免探索危险动作；**收敛稳定**，收敛条件宽松。缺点：**需要探索**，训练前期需要足够探索；**可能次优**，不学习最优策略价值；**对ε敏感**，ε-greedy参数影响性能。

| 优点 | 说明 | 适用场景 |
|------|------|----------|
| 在策略 | 学习真实Q值 | 部署相同策略 |
| 安全 | 避免危险动作 | 机器人 |
| 稳定 | 收敛条件宽松 | 训练 |

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 次优 | 非最优策略 | 降低ε |
| 探索 | 需要足够探索 | ε衰减 |

## 7. 调库实现（Python完整代码）

```python
import numpy as np
import gym
from collections import defaultdict


class SARSAAgent:
    """SARSA智能体"""
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
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def choose_action(self, state):
        """ε-greedy动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def learn(self, state, action, reward, next_state, next_action, done):
        """SARSA更新"""
        current_Q = self.Q[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state][next_action]
        
        # SARSA更新（使用真实的next_action）
        self.Q[state][action] += self.lr * (target - current_Q)
        
        # 探索衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, num_episodes=10000, max_steps=100):
        """训练智能体"""
        rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            action = self.choose_action(state)
            total_reward = 0
            
            for step in range(max_steps):
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                next_action = self.choose_action(next_state)
                
                self.learn(state, action, reward, next_state, next_action, done)
                
                state = next_state
                action = next_action
                
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


class SARSALambdaAgent:
    """带 eligibility trace 的SARSAL"""
    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.99, epsilon=0.1, lambda_=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.lambda_ = lambda_
        
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.eligibility = defaultdict(lambda: np.zeros(n_actions))
    
    def learn(self, state, action, reward, next_state, next_action, done):
        """SARSAL更新"""
        # 计算TD误差
        if done:
            td_error = reward - self.Q[state][action]
        else:
            td_error = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
        
        # 更新eligibility trace
        self.eligibility[state][action] += 1
        
        # 更新所有状态-动作对
        for s in self.eligibility:
            for a in range(self.n_actions):
                self.Q[s][a] += self.lr * td_error * self.eligibility[s][a]
                self.eligibility[s][a] *= self.gamma * self.lambda_


class QLearningAgent:
    """Q-learning对比"""
    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def learn(self, state, action, reward, next_state, done):
        current_Q = self.Q[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.Q[next_state])
        
        # Q-learning更新（使用max）
        self.Q[state][action] += self.lr * (target - current_Q)


def run_cliff_walking():
    """运行Cliff Walking环境"""
    env = gym.make('CliffWalking-v0')
    
    print("=== SARSA on CliffWalking ===")
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    
    # SARSA
    sarsa_agent = SARSAAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1
    )
    
    sarsa_rewards = sarsa_agent.train(env, num_episodes=5000)
    
    # Q-learning对照
    ql_agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1
    )
    
    ql_env = gym.make('CliffWalking-v0')
    for _ in range(500):
        state = ql_env.reset()
        for __ in range(100):
            action = ql_agent.choose_action(state)
            next_state, reward, done, _ = ql_env.step(action)
            ql_agent.learn(state, action, reward - 100 if next_state < 47 else reward, next_state, done)
            state = next_state
            if done:
                break
    
    sarsa_eval = sarsa_agent.evaluate(env, num_episodes=1000)
    ql_eval = ql_agent.evaluate(ql_env, num_episodes=1000)
    
    print(f"\nSARSA Average Reward: {sarsa_eval:.2f}")
    print(f"Q-learning Average Reward: {ql_eval:.2f}")


def run_frozen_lake():
    """运行FrozenLake环境"""
    env = gym.make('FrozenLake-v1')
    
    print("\n=== SARSA on FrozenLake ===")
    
    agent = SARSAAgent(
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
    run_cliff_walking()
    run_frozen_lake()
```

## 8. 手工代码实现

```python
import numpy as np


def basic_sarsa():
    """基础SARSA实现"""
    n_states = 4
    n_actions = 2
    Q = np.zeros((n_states, n_actions))
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 1000
    
    for _ in range(num_episodes):
        s = 0
        a = np.argmax(Q[s]) if np.random.random() > epsilon else np.random.randint(n_actions)
        
        while s != n_states - 1:
            s_new = min(n_states - 1, s + (1 if a == 1 else 0))
            r = 1 if s_new == n_states - 1 else -0.01
            
            a_new = np.argmax(Q[s_new]) if np.random.random() > epsilon else np.random.randint(n_actions)
            
            Q[s, a] += alpha * (r + gamma * Q[s_new, a_new] - Q[s, a])
            
            s = s_new
            a = a_new
    
    return Q


def cliffwalking_sarsa():
    """Cliff Walking SARSA"""
    n_rows = 4
    n_cols = 12
    n_states = n_rows * n_cols
    n_actions = 4
    
    Q = np.zeros((n_states, n_actions))
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    
    for _ in range(10000):
        s = 0
        a = np.random.randint(n_actions)
        
        while True:
            if np.random.random() < epsilon:
                a_new = np.random.randint(n_actions)
            else:
                a_new = np.argmax(Q[s])
            
            s_old = s
            
            if a == 0 and s >= n_cols:
                s -= n_cols
            elif a == 1 and s < n_states - n_cols:
                s += n_cols
            elif a == 2 and s % n_cols > 0:
                s -= 1
            elif a == 3 and s % n_cols < n_cols - 1:
                s += 1
            
            if 0 < s < n_states - n_cols:
                r = -1
            else:
                r = -100
            
            Q[s_old, a] += alpha * (r + gamma * Q[s, a_new] - Q[s_old, a])
            
            if s == n_states - 1 or r == -100:
                break
    
    return Q


if __name__ == '__main__':
    Q = basic_sarsa()
    print("Q-table:")
    print(Q)
```

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_sarsa_vs_qlearning():
    """对比SARSA vs Q-learning"""
    episodes = np.arange(1, 10001)
    
    sarsa_reward = -50 + 20 * (1 - np.exp(-0.0003 * episodes)) + 5 * np.random.randn(10000)
    ql_reward = -100 + 50 * (1 - np.exp(-0.0004 * episodes)) + 10 * np.random.randn(10000)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, sarsa_reward, label='SARSA', alpha=0.7)
    plt.plot(episodes, ql_reward, label='Q-learning', alpha=0.7)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('SARSA vs Q-learning', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sarsa_vs_ql.png', dpi=150)
    plt.show()


def plot_q_table():
    """Q表可视化"""
    Q = np.random.randn(16, 4) * 10
    
    plt.figure(figsize=(10, 6))
    plt.imshow(Q.max(axis=1).reshape(4, 4), cmap='RdYlGn')
    plt.colorbar(label='Max Q Value')
    plt.title('State Values from Q-table', fontsize=14)
    plt.tight_layout()
    plt.savefig('sarsa_q.png', dpi=150)
    plt.show()


def plot_convergence():
    """收敛曲线"""
    steps = np.arange(1, 10001)
    errors = np.exp(-0.0005 * steps) + 0.1 * np.random.randn(len(steps))
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(steps, errors)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('TD Error', fontsize=12)
    plt.title('SARSA Convergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sarsa_conv.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    plot_sarsa_vs_qlearning()
    plot_q_table()
    plot_convergence()
```

结果分析：SARSA在Cliff Walking等任务上学习到更安全的策略（虽然可能不是最优的），因为它将探索的成本计入了价值函数。Q-learning学习到最短路径但可能会“撞墙”。

## 10. 模型评估

SARSA评估：
1. **Average Reward**：平均episode奖励
2. **Steps per Episode**：完成任务所需步数
3. **Success Rate**：成功率
4. **TD Error**：收敛过程监控

## 11. 常见问题与易错点

问题：
1. 忘记选择next_action
2. ε不衰减导致不收敛
3. 混淆SARSA和Q-learning

解决：
1. 明确是on-policy学习
2. 设置ε的衰减
3. 对比理解

## 12. 学习总结

SARSA是on-policy TD学习算法，与Q-learning形成对比。核心：使用真实动作a'更新，学习当前策略的价值。

学习要点：
1. on-policy vs off-policy
2. TD学习
3. 安全策略学习

## 13. 练习题与思考题（含答案）

**练习题1**：写出SARSA的更新公式。

答案：Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]

**练习题2**：SARSA vs Q-learning的区别。

答案：SARSA使用真实a'，学习当前策略；Q-learning使用max a'，学习最优策略。

**思考题1**：什么时候使用SARSA？

答案：当部署时使用与训练相同的策略，或需要考虑探索成本时。

### 13.3 详细答案与解析

#### 练习：计算

**问题**：s=0, a=0, r=-1, s'=1, a'=1, γ=0.9, α=0.1, Q(0,0)=1, Q(1,1)=2，更新Q(0,0)。

**答案**：
```
target = -1 + 0.9 * 2 = 0.8
Q(0,0) = 1 + 0.1 * (0.8 - 1) = 0.98
```

## 14. 学习路径建议

学习SARSA：
1. 强化学习基础
2. TD学习
3. on-policy vs off-policy
4. SARSA vs Q-learning对比

### 14.1 扩展资源

**论文**：
1. Rummery & Niranjan (1994). "On-line Q-learning using connectionist systems"
2. "<cite>Original SARSA paper</cite>"

**框架**：
1.OpenAI Gym
2. Stable Baselines

**问题对比**：
| 问题 | SARSA策略 | Q-learning策略 |
|------|----------|----------------|
| Cliff Walking | 安全但绕远 | 最优但风险 |
| 机器人 | 安全探索 | 可能危险 |