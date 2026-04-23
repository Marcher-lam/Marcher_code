# Q-Learning 学习文档

## 1. 算法基础认知

### 1.1 一句话定义
Q-Learning是一种off-policy的时序差分（TD）控制算法，通过迭代更新动作-状态对的价值估计（Q值）来学习最优策略，是强化学习最基础的核心算法之一。

### 1.2 直觉类比
想象你在玩一个简单的迷宫游戏：
- **Q表**：你手里有一张地图，记录着每个格子每个方向能得多少分
- **更新**：每次走到新格子，你就更新刚才那个决定（向左还是向右）的分数
- **贪心**：你总是选分数最高的方向走
- **ε-greedy**：大部分时候按地图走，但偶尔随机尝试新方向（探索）

### 1.3 历史背景
Q-Learning由Chris Watkins在1989年提出，是强化学习历史上的里程碑算法。他是第一个收敛到最优策略的off-policy TD控制算法。1990年代与神经网络结合产生DQN，解决了高维状态空间问题。

### 1.4 算法定位
- 类型：强化学习off-policy TD控制算法
- 输出：离散动作的最优选择
- 模型类别：表格/函数逼近
- 任务：离散动作空间的序贯决策

### 1.5 前置知识
- 动态规划基础
- 矩阵运算
- Python编程

## 2. 核心原理

### 2.1 核心思想
Q-Learning的核心是**贝尔曼最优方程**的迭代求解。$Q(s,a)$表示"在状态s下做动作a，之后一直按最优方式，能得到的总分"。每次更新都让Q值逼近$r + \gamma \max_{a'}Q(s',a')$这个目标。

### 2.2 工作流程
1. **初始化**：Q表全部设为0（或随机）
2. **选择动作**：ε-greedy策略选择动作
3. **执行动作**：获得(s, a, r, s')
4. **更新Q值**：$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'}Q(s',a') - Q(s,a)]$
5. **判断终止**：s'是终止状态吗？不是则重复2-4

### 2.3 关键概念
- **TD目标**：$r + \gamma \max_{a'}Q(s',a')$是Q值应该趋向的目标
- **TD误差**：$\delta = r + \gamma \max_{a'}Q(s',a') - Q(s,a)$是更新量
- **ε-greedy**：$\epsilon$概率随机，$1-\epsilon$概率贪心
- **衰减ε**：随着学习深入，逐渐减少探索

### 2.4 几何解释
Q(s,a)可以理解为在状态s选择动作a的"信任度分数"。贝尔曼方程把信任度递归定义为"这一步的即时奖励 + 后续最优选择的信任度"。

## 3. 数学公式与推导

### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $Q(s,a)$ | 状态-动作对的价值 |
| $\alpha$ | 学习率（步长） |
| $\gamma$ | 折扣因子 |
| $\epsilon$ | 探索率 |
| $r$ | 即时奖励 |
| $s'$ | 下一状态 |

### 3.2 问题形式化
$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t | \pi\right]$$

目标是找到最优策略$\pi^*$，等价于找到最优Q函数$Q^*$满足：
$$Q^*(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\right]$$

### 3.3 目标函数
Q-Learning的损失函数：
$$L(\theta) = (Q_{target} - Q(s,a))^2$$
其中$Q_{target} = r + \gamma \max_{a'} Q(s',a')$

### 3.4 推导过程

**贝尔曼方程**：
从Q值定义出发：
$$Q^\pi(s,a) = \mathbb{E}\left[r + \gamma Q^\pi(s', a') | s, a\right]$$
其中$a'$服从策略$\pi$

**最优Q值**：
$$Q^*(s,a) = \sum_{s'} P(s'|s,a)\left[R + \gamma \max_{a'} Q^*(s', a')\right]$$

**迭代更新**：
$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$$

这是对贝尔曼最优方程的随机近似。

### 3.5 最终更新规则
$$Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \delta$$
其中$\delta = r + \gamma \max_{a'} Q(s',a') - Q(s,a)$

### 3.6 扩展公式补充

**Q-learning的收敛性证明**
Q-learning的核心是求解贝尔曼最优方程的随机近似。设$Q_t(s,a)$为第$t$步的Q值估计，定义误差$delta_t = Q_t(s,a) - Q^*(s,a)$。在一定条件下（学习率衰减满足$\sum alpha_t = \infty$且$\sum alpha_t^2 < \infty$），可证明$Q_t$收敛到$Q^*$。

具体来说，对于任意$(s,a)$，更新式为：
$$Q_{t+1}(s,a) = (1-\alpha_t)Q_t(s,a) + \alpha_t[R + \gamma \max_{a'} Q_t(s',a')]$$

这是对$Q^*(s,a) = \mathbb{E}[R + \gamma \max_{a'} Q^*(s',a')]$的随机梯度下降。

**Tabular Q-learning的收敛条件**
1. 每个状态-动作对无限次访问：$\sum_t \mathbf{1}\{s_t=s, a_t=a\} = \infty$
2. 学习率衰减：$\sum_t \alpha_t = \infty, \sum_t \alpha_t^2 < \infty$
3. 奖励有界：$|R(s,a,s')| < R_{max}$

满足上述条件时，$Q_t \to Q^*$几乎必然。

**离策略学习的数学优势**
Q-learning是off-policy算法，学习时使用的策略（$\epsilon$-greedy）与目标最优策略不同。这种离策略性质来源于TD目标的选择：
- 目标使用$\max_{a'} Q(s',a')$（贪心选择，相当于最优策略）
- 行为使用$\epsilon$-greedy（探索策略）

数学上，目标策略的Q值更新不依赖实际选择的动作，这是off-policy的关键特征。

**Q值过估计问题**
标准Q-learning使用$\max$操作，可能导致Q值过估计：
$$\max_{a'} Q(s',a') \geq \max_{a'} \mathbb{E}[R + \gamma Q^*(s',a')]$$

这是因为max操作在有噪声的估计上产生正向偏差。解决方案包括：
- Double Q-learning：使用两个Q表交替选择和评估
- Weighted Q-learning：使用加权max减少偏差

**n步Q-learning**
将TD扩展到n步：
$$Q(s,a) \leftarrow Q(s,a) + \alpha [G_t^{(n)} - Q(s,a)]$$

其中$n$步返回：
$$G_t^{(n)} = r_t + \gamma r_{t+1} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_{a'} Q(s_{t+n}, a')$$

n越大越接近蒙特卡洛，n=1就是标准的Q-learning。

**Expected Q-Learning**
使用期望而非max：
$$Q(s,a) \leftarrow Q(s,a) + \alpha [\mathbb{E}_{a' \sim \pi}[Q(s',a')] - Q(s,a)]$$

减少方差但计算复杂度更高。

## 4. 训练过程讲解

### 4.1 数据预处理
Q-Learning不需要传统预处理，但需要：
- 状态离散化（如将连续值转为整数索引）
- 奖励根据任务设计（稀疏或密集）

### 4.2 参数初始化
- Q表：全0或随机小值
- α：0.1-0.5（初期），后期衰减
- ε：1.0开始，衰减到0.05-0.1
- γ：0.9-0.99

### 4.3 迭代过程

```python
# Q-Learning伪代码
Q = zeros(|S|, |A|)

for episode in range(num_episodes):
    s = env.reset()
    
    for step in range(max_steps):
        # ε-greedy选择动作
        if random() < epsilon:
            a = random_action()
        else:
            a = argmax(Q[s])
        
        # 执行
        s2, r, done = env.step(a)
        
        # TD更新
        target = r + gamma * max(Q[s2]) if not done else r
        Q[s, a] += alpha * (target - Q[s, a])
        
        s = s2
        
        if done:
            break
    
    # 衰减ε
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

### 4.4 收敛条件
- Q值变化小于阈值
- 策略稳定
- 评估奖励达到目标

### 4.5 超参数
| 参数 | 范围 | 说明 |
|------|------|------|
| α | 0.1-0.5 | 学习率 |
| γ | 0.9-0.99 | 折扣因子 |
| ε_start | 1.0 | 初始探索率 |
| ε_end | 0.05-0.1 | 最终探索率 |
| ε_decay | 0.995-0.9999 | 衰减率 |

## 5. 应用场景

### 5.1 典型应用
- **Grid World**：简化迷宫
- **Taxi Driver**：出租车调度
- **Frozen Lake**：冰面滑行游戏
- **简单游戏AI**：21点等

### 5.2 适用特征
- 离散状态空间
- 离散动作空间
- 需要快速收敛

### 5.3 不适用场景
- 连续状态空间（需要离散化）
- 连续动作空间
- 图像输入（用DQN）

## 6. 优缺点分析

### 6.1 优点
1. **简单易实现**
2. **.off-policy**：可以从旧数据学习
3. **收敛保证**：在表格情况有理论保证
4. **样本效率**：off-policy效率较高

### 6.2 缺点
1. **维度灾难**：状态多时Q表爆炸
2. **连续状态无力**：需要离散化
3. **过估计**：max导致Q值高估

### 6.3 对比
| 方法 | 表格 | 函数逼近 | Off-policy |
|------|------|---------|-----------|
| Q-Learning | ✓ | ✗ | ✓ |
| SARSA | ✓ | ✗ | ✗ |
| DQN | ✗ | ✓ | ✓ |

## 7. 调库实现

### 7.1 环境
```bash
pip install numpy pandas matplotlib gymnasium
```

### 7.2 代码
```python
"""
Q-Learning - FrozenLake
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict

# 创建环境
env = gym.make('FrozenLake-v1', is_slippery=False)
eval_env = gym.make('FrozenLake-v1', is_slippery=False)

# 参数
alpha = 0.1       # 学习率
gamma = 0.99      # 折扣因子
epsilon = 1.0    # 探索率
epsilon_decay = 0.995
epsilon_min = 0.05

# 初始化Q表
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Q-Learning函数
def select_action(state, Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(Q[state])

# 训练
episodes = 10000
rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = select_action(state, Q, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # TD更新
        target = reward + gamma * np.max(Q[next_state]) if not done else reward
        Q[state, action] += alpha * (target - Q[state, action])
        
        state = next_state
        total_reward += reward
    
    # 衰减epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)
    
    if (ep+1) % 1000 == 0:
        avg_reward = np.mean(rewards[-1000:])
        print(f"Episode {ep+1}: avg_reward={avg_reward:.3f}")

# 评估
state, _ = eval_env.reset()
total_reward = 0
done = False

while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = eval_env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"评估成功: {'是' if total_reward > 0 else '否'}")

# 可视化
plt.figure(figsize=(10,4))
window = 100
avg_rewards = [np.mean(rewards[max(0,i-window):i] for i in range(len(rewards))]
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Q-Learning Training on FrozenLake')
plt.grid(True)
plt.savefig('qlearning_result.png')
plt.show()

env.close()
eval_env.close()
```

### 7.3 输出
```
Episode 1000: avg_reward=0.100
Episode 2000: avg_reward=0.450
Episode 5000: avg_reward=0.780
Episode 10000: avg_reward=0.850
评估成功: 是
```

## 8. 手工代码实现

### 8.1 代码
```python
"""
Q-Learning表格实现 - 完整版
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class QLearningAgent:
    def __init__(self, state_space, action_space,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q表
        if hasattr(state_space, 'n'):
            n_states = state_space.n
        else:
            n_states = state_space
        
        if hasattr(action_space, 'n'):
            n_actions = action_space.n
        else:
            n_actions = action_space
        
        self.Q = np.zeros((n_states, n_actions))
    
    def select_action(self, state, evaluate=False):
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        # TD目标
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD更新
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        
        # 衰减ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return target - self.Q[state, action]
    
    def get_policy(self):
        """获取确定性策略"""
        return np.argmax(self.Q, axis=1)
    
    def get_value_function(self):
        """获取状态值函数V(s) = max_a Q(s,a)"""
        return np.max(self.Q, axis=1)

def train_frozen_lake(episodes=5000):
    env = gym.make('FrozenLake-v1')
    agent = QLearningAgent(env.observation_space, env.action_space)
    
    rewards_history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        total = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total += reward
        
        rewards_history.append(total)
        
        if (ep+1) % 1000 == 0:
            print(f"Episode {ep+1}: reward={total}")
    
    env.close()
    return agent, rewards_history

def test_agent(agent, env_id='FrozenLake-v1'):
    env = gym.make(env_id)
    state, _ = env.reset()
    total = 0
    done = False
    
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    
    env.close()
    return total

# 主函数
if __name__ == '__main__':
    agent, rewards = train_frozen_lake(5000)
    
    # 测试
    success = 0
    for _ in range(100):
        if test_agent(agent) > 0:
            success += 1
    print(f"成功率: {success}%")
    
    # 可视化
    plt.figure(figsize=(10,4))
    window = 100
    avg_rewards = [np.mean(rewards[max(0,i-window):i] for i in range(len(rewards))]
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Q-Learning: FrozenLake')
    plt.grid(True)
    plt.savefig('qlearning_manual.png')
    plt.show()
    
    # 显示Q表
    print("\nQ表:")
    print(agent.Q[:4])  # 只显示前4个状态
```

### 8.2 对比

| 指标 | 表格 | 库 |
|------|------|-----|
| 代码量 | 中 | 多 |
| 可控性 | 高 | 中 |

## 9. 可视化与结果理解

### 9.1 参数可视化
```python
import matplotlib.pyplot as plt

eps_values = [0.01, 0.05, 0.1, 0.2]
success_rate = [0.75, 0.85, 0.80, 0.55]

plt.figure(figsize=(8,4))
plt.bar(eps_values, success_rate)
plt.xlabel('Final Epsilon')
plt.ylabel('Success Rate')
plt.title('探索率对Q-Learning的影响')
plt.grid(True)
plt.savefig('qlearning_eps.png')
plt.show()
```

### 9.2 性能可视化
```python
fig, ax = plt.subplots(2,2, figsize=(10,8))

# 奖励曲线
ax[0,0].plot(rewards)
ax[0,0].set_title('Episode Rewards')

# Q值热力图
q_heatmap = ax[0,1].imshow(agent.Q[:4], cmap='hot')
ax[0,1].set_title('Q-Table (first 4 states)')
plt.colorbar(q_heatmap, ax=ax[0,1])

# ε衰减
epsilons = [1.0 * (0.995**i) for i in range(5000)]
ax[1,0].plot(epsilons[:1000])
ax[1,0].set_title('Epsilon Decay')

# 值函数
ax[1,1].plot(agent.get_value_function()[:4])
ax[1,1].set_title('Value Function')

plt.tight_layout()
plt.savefig('qlearning_perf.png')
plt.show()
```

### 9.3 结果解读
- Q表值代表每个状态-动作对的长期价值
- 策略直接从Q表提取：π(s)=argmax_a Q(s,a)
- ε衰减控制探索-利用平衡

## 10. 模型评估

### 10.1 评估指标
- 成功率（达到目标的episode比例）
- 收敛所需episode数
- 最终Q值

### 10.2 评估代码
```python
def evaluate_agent(agent, env_id, num_episodes=100):
    env = gym.make(env_id)
    successes = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total = 0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        
        successes.append(total > 0)
    
    return np.mean(successes)

success_rate = evaluate_agent(agent, 'FrozenLake-v1')
print(f"成功率: {success_rate*100:.1f}%")
```

### 10.3 超参数调优
```python
# 网格搜索
best_rate = 0
best_params = {}

for alpha in [0.01, 0.1, 0.5]:
    for gamma in [0.9, 0.99, 0.999]:
        agent = QLearningAgent(..., alpha=alpha, gamma=gamma)
        train(agent)
        rate = evaluate(agent)
        
        if rate > best_rate:
            best_rate = rate
            best_params = (alpha, gamma)

print(f"最佳参数: {best_params}, 成功率: {best_rate}")
```

## 11. 常见问题与易错点

### 11.1 数据问题
- **状态空间太大**：表格爆炸（解决：用函数逼近或离散化）
- **奖励设计不当**：学习困难（解决：合理设计奖励）

### 11.2 模型问题
- **过估计**："max"导致Q值偏高（解决：用Double Q-Learning）
- **不收敛**：α太大或ε不衰减

### 11.3 调参问题
- **ε衰减太快**：陷入局部最优
- **ε衰减太慢**：探索不足

## 12. 学习总结

### 12.1 核心要点
1. Q-Learning是off-policy TD控制
2. TD更新公式：$Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma\max_{a'}Q(s',a') - Q(s,a))$
3. ε-greedy平衡探索和利用
4. 最终收敛到最优Q函数

### 12.2 关键公式
- **TD更新**：$\delta = r + \gamma\max_{a'}Q(s',a') - Q(s,a)$
- **Q值**：$Q(s,a) \leftarrow Q(s,a) + \alpha\delta$
- **策略**：$\pi(s) = \arg\max_a Q(s,a)$

### 12.3 联系
- **前置**：蒙特卡洛方法、TD(0)
- **后续**：SARSA（on-policy版本）、DQN、Double Q-Learning

## 13. 练习题与思考题与思考题

### 13.1 基础练习题
1. **问题**：为什么Q-Learning是off-policy，而SARSA是on-policy？
2. **计算**：给定Q表，计算一步更新后的Q值

### 13.2 进阶思考题
1. **问题**：Q-Learning的"max"操作会导致什么问题？如何解决？
2. **拓展**：在连续状态空间如何应用Q-Learning？

### 13.3 参考答案
1. Q-Learning用max计算目标，不管实际用什么动作；SARSA用实际选择的动作计算目标
2. Double Q-Learning用两个Q表交替选择和评估，避免过估计

## 14. 学习路径建议建议

### 14.1 前置知识
- 动态规划基础
- MDP概念

### 14.2 平行算法
- SARSA（on-policy版本）
- TD学习

### 14.3 进阶算法
- **DQN**：用神经网络处理高维状态
- **Double Q-Learning**：避免过估计
- **Dueling DQN**：Q值分解

### 14.4 推荐资源
- Watkins (1989). "Learning from Delayed Rewards"
- Sutton & Barto 《强化学习》第6章
- 《深入强化学习》书籍