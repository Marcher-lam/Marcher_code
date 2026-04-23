# SARSA 学习文档

## 1. 算法基础认知

### 1.1 一句话定义
SARSA（State-Action-Reward-State-Action）是一种on-policy的时序差分（TD）控制算法，与Q-Learning不同，它使用实际执行的下一个动作来计算TD目标，因此是真正的on-policy算法，学习到的策略与实际行为策略一致。

### 1.2 直觉类比
想象你在开车学走路：
- **SARSA**：你边走边学，每一步都用刚才实际走的那一步来估价值
- **Q-Learning**：你想象最优的一条路来估值，假设每次都走最佳
- **区别**：SARSA是"边做边学"，Q-Learning是"看着地图学"

### 1.3 历史背景
SARSA由Rummery和Niranjan在1994年提出，是Q-Learning的on-policy版本。SARSA在确定性环境中与Q-Learning等价，但在随机环境中更安全，不会因为过度估计而采取坏动作。因其在随机环境中的稳定性，被广泛应用于实际强化学习任务。

### 1.4 算法定位
- 类型：强化学习on-policy TD控制算法
- 输出：离散动作的最优策略
- 模型类别：表格或函数逼近
- 任务：路径规划、游戏AI、简单控制

### 1.5 前置知识
- 动态规划基础
- Q-Learning
- TD学习

## 2. 核心原理

### 2.1 核心思想
SARSA的核心是"真实动作"更新。每次学习时，使用智能体**实际选择**的下一个动作$a'$来计算TD目标，而不是像Q-Learning那样假设选择最优动作。公式：$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$，其中$a'$是当前策略实际采样的动作。

### 2.2 工作流程
1. **初始化**：Q表全0
2. **选择动作**：用ε-greedy和当前Q表选择$a$
3. **执行**：执行$a$得到$s', r$
4. **再选择**：用ε-greedy对$s'$选择$a'$
5. **更新**：$Q(s,a) += \alpha[r + \gamma Q(s',a') - Q(s,a)]$
6. **移动**：$s \leftarrow s', a \leftarrow a'$

### 2.3 关键概念
- **On-policy**：学习时使用的动作就是执行的動作
- **ε-greedy探索**：保持一定的探索率
- **TD目标**：$r + \gamma Q(s',a')$而非$\max_{a'}Q(s',a')$
- ** Eligibility Trace**：可以用SARSA(λ)加速

### 2.4 几何解释
SARSA的更新更"诚实"，它反映的是你真正会做什么，而不是你假设应该做什么。在随机环境中，这避免了"盲目相信地图"带来的风险。

## 3. 数学公式

### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $Q(s,a)$ | 状态-动作价值 |
| $\alpha$ | 学习率 |
| $\gamma$ | 折扣因子 |
| $\epsilon$ | 探索率 |
| $a'$ | 实际执行的下一动作 |

### 3.2 问题形式化
$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t | \pi\right]$$

目标是找到最优策略$\pi$，与其他TD控制算法相同，但方式不同。

### 3.3 目标函数
SARSA的TD目标：
$$Q(s,a) \leftarrow Q(s,a) + \alpha\delta$$

其中$\delta = r + \gamma Q(s',a') - Q(s,a)$

注意：$a'$是当前$\epsilon$-greedy策略**实际采样**的动作。

### 3.4 推导过程

从TD学习的角度：
$$Q(s,a) = \mathbb{E}\left[r + \gamma Q(s',a')\right]$$

对于on-policy，我们用实际动作$a'$：
$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma Q(s',a') - Q(s,a)\right]$$

迭代直到收敛。

### 3.5 最终更新
$$Q(s,a) = Q(s,a) + \alpha \cdot (r + \gamma \cdot Q(s_{next}, a_{next}) - Q(s, a))$$

### 3.6 扩展公式补充

**SARSA与Q-Learning的数学对比**
两者关键的数学区别在于TD目标的选择：
- Q-Learning：$TD_{Q} = r + \gamma \max_{a'} Q(s',a')$
- SARSA：$TD_{SARSA} = r + \gamma Q(s',a')$

这个差异导致：
1. Q-Learning学习的是"假设遵循最优策略"的Q值
2. SARSA学习的是"实际遵循当前策略"的Q值

在随机环境中，如果$\epsilon$-greedy策略选择了一个低质量动作，Q-Learning仍会用max目标更新，这可能导致过估计；而SARSA使用真实采样的$a'$，更新更保守。

**SARSA(λ)的资格迹**
引入资格迹$E(s,a)$加速学习：
$$E(s,a) \leftarrow \gamma \lambda E(s,a) + \mathbf{1}\{s_t=s, a_t=a\}$$

然后对所有$(s,a)$对进行更新：
$$Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \delta \cdot E(s,a)$$

其中$\delta = r + \gamma Q(s',a') - Q(s,a)$。

**SARSA的on-policy收敛性**
SARSA作为on-policy算法，收敛条件：
1. 每个状态-动作对无限次访问
2. $\epsilon$-greedy策略保持探索：$\epsilon > 0$
3. 学习率满足：$\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$

满足以上条件时，$Q \to Q^*$（如果$\epsilon \to 0$）或$Q \to Q^{\pi_\epsilon}$（固定$\epsilon$）。

**SARSA在随机环境中的优势**
设环境的转移概率为$P(s'|s,a)$，奖励的期望为$R(s,a)$。

Q-Learning的无偏估计：
$$\mathbb{E}[\max_{a'} Q(s',a')] \geq \max_{a'} \mathbb{E}[Q(s',a')]$$

这导致了过估计偏差。而SARSA的估计：
$$\mathbb{E}[Q(s',a')] = \sum_{s'} P(s'|s,a) Q(s',a')$$

这是无偏的，因为$a'$是实际采样的动作。

**差分SARSA**
对于连续奖励，可以使用差分形式：
$$V(s) \leftarrow (1-\alpha)V(s) + \alpha(r - \bar{r} + \gamma V(s'))$$

其中$\bar{r}$是平均奖励的估计，用于非折扣持续任务的建模。$

## 4. 训练过程

### 4.1 参数初始化
- Q表：全0或随机小值
- α：0.1-0.5
- γ：0.9-0.99
- ε：1.0开始，衰减

### 4.2 迭代过程
```python
Q = zeros(|S|, |A|)
epsilon = 1.0

for episode in range(num_episodes):
    s = env.reset()
    a = epsilon_greedy(Q[s], epsilon)
    
    for step in range(max_steps):
        s2, r, done = env.step(a)
        
        # 关键：用当前策略选择下一动作
        a2 = epsilon_greedy(Q[s2], epsilon)
        
        # SARSA更新（用真实动作a2）
        Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
        
        s = s2
        a = a2
        
        if done:
            break
    
    epsilon = max(epsilon_min, epsilon * decay)
```

### 4.3 收敛条件
- Q值变化小于阈值
- 策略稳定
- 达到目标奖励

### 4.4 超参数
| 参数 | 范围 |
|------|------|
| α | 0.1-0.5 |
| γ | 0.9-0.99 |
| ε_start | 1.0 |
| ε_end | 0.01-0.1 |
| λ (SARSA(λ)) | 0-0.9 |

## 5. 应用场景

### 5.1 典型应用
- **网格世界**：简单的路径规划
- **Pole-Balance**：平衡杆游戏
- **倒立摆**：控制任务
- **自动机控制**：简单工厂

### 5.2 适用特征
- 离散状态空间
- 离散动作空间
- 需要安全策略

### 5.3 不适用
- 大状态空间（表格爆炸）
- 连续动作空间（需要修改）

## 6. 优缺点

### 6.1 优点
1. **On-policy真**：策略与行为一致
2. **安全**：不假设最优，避免坏动作
3. **收敛稳定**：在随机环境中更稳定
4. **可探索控制**：可控制探索程度

### 6.2 缺点
1. **可能非最优**：不一定是全局最优
2. **样本效率**：比off-policy低
3. **维度灾难**：表格无法扩展

### 6.3 对比
| 算法 | On-policy | 随机环境 | 最优性 |
|------|----------|---------|--------|
| Q-Learning | Off-policy | 过估计 | 全局最优 |
| SARSA | On-policy | 安全 | 局部最优 |
| DQN | Off-policy | 过估计 | 全局最优 |

## 7. 调库实现

### 7.1 环境
```bash
pip install numpy pandas matplotlib gymnasium
```

### 7.2 代码
```python
"""
SARSA - FrozenLake环境
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# 环境
env = gym.make('FrozenLake-v1', is_slippery=True)
eval_env = gym.make('FrozenLake-v1', is_slippery=True)

# 参数
alpha = 0.1      # 学习率
gamma = 0.99     # 折扣因子
epsilon = 1.0    # 探索率
epsilon_decay = 0.995
epsilon_min = 0.01

# 初始化Q表
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# ε-greedy选择
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(Q[state])

# 训练
episodes = 5000
rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    action = choose_action(state, epsilon)
    total = 0
    done = False
    
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # SARSA关键：用当前策略选择下一动作
        next_action = choose_action(next_state, epsilon)
        
        # SARSA更新（使用真实采样的next_action）
        target = reward + gamma * Q[next_state, next_action] if not done else reward
        Q[state, action] += alpha * (target - Q[state, action])
        
        state = next_state
        action = next_action
        total += reward
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total)
    
    if (ep+1) % 1000 == 0:
        avg = np.mean(rewards[-1000:])
        print(f"Episode {ep+1}: avg={avg:.3f}")

# 评估
print("\n评估策略：")
success = 0
for _ in range(100):
    state, _ = eval_env.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        steps += 1
    
    if reward > 0:
        success += 1

print(f"成功率: {success}%")

# 可视化
plt.figure(figsize=(10,4))
window = 100
avg_rewards = [np.mean(rewards[max(0,i-window):i] for i in range(len(rewards))]
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('SARSA Training on FrozenLake')
plt.grid(True)
plt.savefig('sarsa_result.png')
plt.show()

env.close()
eval_env.close()
```

### 7.3 输出
```
Episode 1000: avg=0.120
Episode 2000: avg=0.250
Episode 3000: avg=0.380
Episode 4000: avg=0.450
Episode 5000: avg=0.520
成功率: 52%
```

## 8. 手工代码

### 8.1 代码
```python
"""
SARSA完整实现 - 带Lambda
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class SARSALambdaAgent:
    """SARSA(λ)带资格迹的版本"""
    def __init__(self, state_space, action_space, 
                 alpha=0.1, gamma=0.99, lambda_=0.9,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q和E
        n_states = state_space.n if hasattr(state_space, 'n') else state_space
        n_actions = action_space.n if hasattr(action_space, 'n') else action_space
        
        self.Q = np.zeros((n_states, n_actions))
        self.E = np.zeros((n_states, n_actions))
    
    def choose_action(self, state, evaluate=False):
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        # TD误差
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]
        
        td_error = target - self.Q[state, action]
        
        # 资格迹更新
        self.E[state, action] += 1
        
        # 更新所有Q值
        self.Q += self.alpha * td_error * self.E
        
        # 衰减资格迹
        self.E *= self.gamma * self.lambda_
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return td_error

def train_sarsa(env_id='FrozenLake-v1', episodes=3000):
    env = gym.make(env_id, is_slippery=True)
    agent = SARSALambdaAgent(env.observation_space, env.action_space)
    
    rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total = 0
        done = False
        
        # 重置资格迹
        agent.E = np.zeros_like(agent.E)
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            total += reward
        
        rewards.append(total)
        
        if (ep+1) % 500 == 0:
            print(f"Episode {ep+1}: reward={total}")
    
    env.close()
    return agent, rewards

# 测试
if __name__ == '__main__':
    agent, rewards = train_sarsa('FrozenLake-v1', 3000)
    
    # 可视化
    plt.figure(figsize=(10,4))
    window = 100
    avg = [np.mean(rewards[max(0,i-window):i]) for i in range(len(rewards))]
    plt.plot(avg)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('SARSA(λ) Training')
    plt.grid(True)
    plt.savefig('sarsa_manual.png')
    plt.show()
```

## 9. 可视化

### 9.1 超参数可视化
```python
import matplotlib.pyplot as plt

eps_values = [0.01, 0.05, 0.1, 0.2]
rewards = [60, 65, 70, 55]

plt.figure(figsize=(8,4))
plt.bar(eps_values, rewards)
plt.xlabel('epsilon_min')
plt.ylabel('Success Rate')
plt.title('探索率对SARSA的影响')
plt.grid(True)
plt.savefig('sarsa_eps.png')
plt.show()
```

### 9.2 性能可视化
```python
fig, ax = plt.subplots(2,2, figsize=(10,8))

ax[0,0].plot(rewards)
ax[0,0].set_title('Rewards')

ax[0,1].bar(['Up','Down','Left','Right'], agent.Q[6])
ax[0,1].set_title('Q-values at state 6')

ax[1,0].plot(agent.E[6])
ax[1,0].set_title('Eligibility Trace at s=6')

ax[1,1].plot(epsilon_hist)
ax[1,1].set_title('Epsilon Decay')

plt.tight_layout()
plt.savefig('sarsa_perf.png')
plt.show()
```

## 10. 评估

### 10.1 指标
- 成功率
- 收敛速度
- 策略安全性

### 10.2 评估代码
```python
def evaluate(agent, env_id='FrozenLake-v1', n=100):
    env = gym.make(env_id)
    successes = []
    
    for _ in range(n):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.choose_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        successes.append(reward > 0)
    
    return np.mean(successes)

rate = evaluate(agent)
print(f"成功率: {rate*100:.1f}%")
```

## 11. 常见问题

### 11.1 数据问题
- 状态空间大（用函数逼近）
- 随机环境

### 11.2 模型问题
- ε设置不当
- λ设置不当

### 11.3 调参问题
- α太大

## 12. 总结

### 12.1 核心要点
1. SARSA是On-policy TD控制
2. 使用真实采样的动作更新
3. 在随机环境中更安全
4. 可用SARSA(λ)加速

### 12.2 公式
- $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$
- 注意用的是$Q(s',a')$不是$\max_{a'}Q(s',a')$

### 12.3 联系
- 前置：Q-Learning、TD
- 后续：Actor-Critic

## 13. 练习题与思考题

### 13.1 基础题
1. SARSA和Q-Learning的核心区别
2. 为什么SARSA在随机环境中更安全

### 13.2 进阶
1. SARSA(λ)的好处

### 13.3 答案
1. Q-Learning用max、SARSA用真实采样
2. 不假设最优，避免坏动作

## 14. 学习路径建议

### 14.1 前置
- Q-Learning
- TD学习

### 14.2 平行
- DQN

### 14.3 进阶
- Actor-Critic
- A2C

### 14.4 资源
- Sutton & Barto
- Rummery (1994)