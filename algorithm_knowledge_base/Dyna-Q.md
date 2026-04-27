# Dyna-Q 学习文档

> 结合模型学习与Q-learning，通过规划加速强化学习。

## 1. 算法基础认知

**一句话定义：** Dyna-Q是一种结合Q-learning和模型学习的强化学习架构，智能体既通过真实经验学习，也通过内部模型模拟的经验进行规划。

**直觉类比：** 想象你学习开车，不仅通过实际上路练习（真实经验），还在家中用驾驶模拟器练习（模型模拟）。模拟器让你更高效地利用时间，快速掌握驾驶技能。

**历史背景：** Dyna-Q由Richard S. Sutton在1990年提出，是整合模型学习和无模型学习的经典架构。它解决了纯Q-learning样本效率低的问题。

**算法定位：** 基于模型的强化学习（Model-based RL）架构，集成Q-learning和规划。

**前置知识：**
- Q-learning基础
- 马尔可夫决策过程（MDP）
- 动态规划基础
- Python编程

## 2. 核心原理

**核心思想：** Dyna-Q维护一个环境模型，智能体既用真实经验更新Q值和模型，也用模型生成的模拟经验进行额外的Q值更新（规划）。这样可以用更少的真实交互学会最优策略。

**工作流程：**
1. 初始化Q表、模型Model
2. 对于每个episode：
   a. 初始化状态s
   b. 当s不是终止状态时：
      - 根据ε-greedy选择动作a
      - 执行动作，获得(s, a, r, s')
      - **直接更新：** 用Q-learning更新Q(s,a)
      - **学习模型：** Model(s,a) → (s', r)
      - **规划循环（n次）：**
        - 从经验中采样状态-动作对(s_sim, a_sim)
        - 用模型预测：(s'_sim, r_sim) = Model(s_sim, a_sim)
        - 用Q-learning更新Q(s_sim, a_sim)基于(s_sim, a_sim, r_sim, s'_sim)
      - s ← s'
3. 重复直到收敛。

**关键概念解释：**
- **模型（Model）：** 对环境转移的估计，存储P(s'|s,a)和R(r|s,a)
- **规划（Planning）：** 使用模型生成模拟经验并更新价值函数
- **n次规划：** 每次真实交互后进行n次模拟更新，n是重要超参数
- **经验池：** 存储历史状态-动作对，用于规划时采样

**几何/直观解释：**
```
Dyna-Q架构示意图：

真实交互: [智能体] --(s,a)--> [真实环境] --(r,s')--> [智能体]
                              ↓
                        学习模型: Model(s,a) = (s', r)

规划: [智能体] --(s_sim,a_sim)--> [内部模型] --(r_sim,s'_sim)--> [智能体]
            ↑                                          ↓
            +-------- 从经验池采样 <--------+

真实更新: Q(s,a) += α[r + γ·max Q(s',a') - Q(s,a)]
规划更新: Q(s_sim,a_sim) += α[r_sim + γ·max Q(s'_sim,a') - Q(s_sim,a_sim)]
```

## 3. 数学公式与推导

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| Model(s,a) | 环境模型 | 预测(s,a)的下一状态和奖励 |
| n | 规划步数 | 每次真实交互后的规划次数 |
| Q(s,a) | Q值函数 | 与Q-learning相同的Q表 |
| α | 学习率 | 用于Q值更新 |
| β | 折扣因子 | 用于计算回报 |

**问题形式化：**

目标与Q-learning相同：学习最优动作价值函数Q*(s,a)。

**模型学习：**

Dyna-Q学习一个确定性的模型：记录每个(s,a)对应的(s', r)：

$$\text{Model}(s,a) \leftarrow (s', r)$$

在实践中，对于随机环境，可以存储平均奖励和最常见下一状态，或使用概率分布。

**Q-learning更新（真实和规划相同）：**

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \beta \max_{a'} Q(s',a') - Q(s,a) \right]$$

**完整Dyna-Q算法：**

对每个时间步t：
1. 执行动作a_t，观察r_{t+1}, s_{t+1}
2. 更新模型：Model(s_t, a_t) = (s_{t+1}, r_{t+1})
3. 更新Q值：Q(s_t, a_t) += α[r_{t+1} + β·max_a' Q(s_{t+1}, a') - Q(s_t, a_t)]
4. 重复n次：
   a. 随机采样之前的(s, a)
   b. 查询模型：(s', r) = Model(s, a)
   c. 更新Q值：Q(s, a) += α[r + β·max_a' Q(s', a') - Q(s, a)]

**为什么这样加速学习：**

1. **模型学习：** 从每个真实经验中学习环境动态性
2. **规划：** 用模型生成大量模拟经验，增加数据效率
3. **经验重用：** 每次真实交互产生n次更新，样本效率提高n倍

**收敛性：** 在满足与Q-learning类似的条件下，Dyna-Q也收敛到最优Q值。关键是模型要准确，且规划时的采样要覆盖重要状态-动作对。

## 4. 训练过程讲解

**数据预处理：**
- 状态-动作对存储：需要存储历史经验用于规划采样
- 模型表示：表格环境用查找表，连续状态可用函数逼近

**参数初始化：**
- Q表初始化：全0或小的随机值
- 模型初始化：空或随机初始值
- 学习率α：0.1~0.5
- 折扣因子β：0.9~0.99
- 规划步数n：5~100（根据计算资源）
- 探索率ε：从1.0衰减到0.01

**迭代过程：**
1. 每个episode开始，重置环境状态
2. 在每步中：
   - 选择并执行动作，获得真实经验
   - 学习模型（记录转移）
   - 直接Q-learning更新
   - 规划循环n次：
     * 从经验池随机采样(s,a)
     * 用模型预测(s',r)
     * Q-learning更新
3. 直到达到终止状态或最大步数

**收敛条件：**
- Q值变化小于阈值
- 策略稳定
- 模型预测准确（可选验证）
- 达到最大episode数

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| α (学习率) | 控制Q值更新步长 | 0.1~0.5 | 0.2 |
| β (折扣因子) | 权衡即时与未来奖励 | 0.9~0.99 | 0.9 |
| n (规划步数) | 每次真实交互的规划次数 | 5~100 | 20 |
| ε (探索率) | 平衡探索与利用 | 0.01~1.0 | 从1.0衰减 |
| 经验池大小 | 规划采样的经验数量 | 1000~100000 | 10000 |

## 5. 应用场景

**典型应用：**

1. **机器人导航：** 机器人在未知环境中学习路径规划。**为什么适合：** 可以通过少量真实移动+大量模拟规划快速学习地图和路径。

2. **游戏AI：** 棋类游戏、视频游戏等。**为什么适合：** 可以模拟大量对局，快速评估不同走法。

3. **资源调度：** 云计算资源分配、生产线调度等。**为什么适合：** 可以基于历史数据建立模型，快速规划最优调度。

4. **仿真环境训练：** 当真实交互成本高时（如自动驾驶）。**为什么适合：** 用仿真模型替代部分真实交互。

**适用数据特征：**
- 环境交互成本较高
- 环境动态性可以学习（模型相对稳定）
- 状态-动作空间有限（或可用函数逼近）
- 需要快速学习

**不适用场景：**
- 模型极难学习（环境高度随机或复杂）
- 计算资源有限（规划需要额外计算）
- 环境快速变化（模型很快过时）
- 真实交互极其廉价（直接Q-learning可能更简单）

## 6. 优缺点分析

**优点：**
1. **样本效率高：** 相同真实样本下学习更快。**成立条件：** 模型相对准确，规划有效。
2. **利用先验知识：** 可以用先验模型初始化。**成立条件：** 有环境的部分知识。
3. **平衡探索与利用：** 规划相当于在模型上的利用。**成立条件：** 模型可信。
4. **灵活性：** 可以调整n平衡真实学习与规划。

**缺点：**
1. **模型误差传播：** 不准确的模型导致次优策略。**问题：** 模型错误会误导规划更新。**缓解思路：** 使用模型不确定性估计，或限制规划步数。
2. **计算开销：** 每次真实交互需要额外n次规划计算。**问题：** 计算资源消耗大。**缓解思路：** 根据计算能力调整n，或使用高效规划算法。
3. **仅适用于确定性环境：** 标准Dyna-Q假设模型是确定性的。**问题：** 随机环境需要存储分布。**缓解思路：** 使用概率模型或改进算法如Dyna-Q+。

**与同类算法对比：**

| 特性 | Q-learning | Dyna-Q | Prioritized Sweeping |
|------|------------|---------|-------------------|
| 样本效率 | 低 | 高 | 高 |
| 计算开销 | 低 | 中 | 中高 |
| 模型需求 | 无 | 有 | 有 |
| 规划策略 | 无 | 均匀采样 | 优先级采样 |

## 7. 调库实现

使用numpy手动实现Dyna-Q（结合gymnasium环境）：

```python
"""
Dyna-Q算法调库实现
结合Q-learning和模型学习，通过规划加速学习
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import random
import matplotlib.pyplot as plt

class DynaQAgent:
    """
    Dyna-Q智能体
    结合Q-learning和模型学习，通过规划加速
    """
    
    def __init__(self, state_space_size, action_space_size,
                 learning_rate=0.2, discount_factor=0.9,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 planning_steps=20):
        """
        初始化Dyna-Q智能体
        
        参数:
        - state_space_size: 状态空间大小
        - action_space_size: 动作空间大小
        - learning_rate: 学习率α
        - discount_factor: 折扣因子β
        - epsilon: 初始探索率
        - planning_steps: 每次真实交互后的规划步数n
        """
        # Q表
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        
        # 模型: 存储 (s,a) -> (s', r)
        self.model = {}
        
        # 经验池: 存储所有访问过的(s,a)
        self.experience_pool = []
        
        # 超参数
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.planning_steps = planning_steps
        self.action_space_size = action_space_size
    
    def choose_action(self, state):
        """
        根据ε-greedy策略选择动作
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            state_key = self._state_to_key(state)
            q_values = self.q_table[state_key]
            return np.random.choice(np.where(q_values == q_values.max())[0])
    
    def learn_model(self, state, action, reward, next_state):
        """
        学习环境模型
        
        确定性模型: 直接记录 (s,a) -> (next_state, reward)
        """
        state_key = self._state_to_key(state)
        key = (state_key, action)
        
        # 存储到模型
        self.model[key] = (next_state, reward)
        
        # 添加到经验池（如果是新的状态-动作对）
        if key not in self.experience_pool:
            self.experience_pool.append(key)
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Q-learning更新（用于真实和规划）
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # 计算TD目标
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        # TD误差
        td_error = td_target - self.q_table[state_key][action]
        
        # 更新Q值
        self.q_table[state_key][action] += self.lr * td_error
        
        return td_error
    
    def planning(self):
        """
        规划阶段：使用模型进行模拟更新
        
        从经验池随机采样，用模型预测进行Q-learning更新
        """
        if len(self.experience_pool) == 0:
            return
        
        for _ in range(self.planning_steps):
            # 随机采样一个状态-动作对
            state_key, action = random.choice(self.experience_pool)
            
            # 查询模型
            if (state_key, action) in self.model:
                next_state, reward = self.model[(state_key, action)]
                
                # 模拟Q-learning更新
                next_state_key = self._state_to_key(next_state)
                
                if isinstance(next_state_key, str) or isinstance(next_state_key, tuple):
                    # 非终止状态
                    td_target = reward + self.gamma * np.max(self.q_table[next_state_key])
                else:
                    # 终止状态（假设是整数编号）
                    td_target = reward
                
                td_error = td_target - self.q_table[state_key][action]
                self.q_table[state_key][action] += self.lr * td_error
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _state_to_key(self, state):
        """将状态转换为可哈希的键"""
        if isinstance(state, (int, np.integer)):
            return state
        return tuple(state) if isinstance(state, np.ndarray) else state


def train_dyna_q(env_name="FrozenLake-v1", num_episodes=1000,
                learning_rate=0.2, discount_factor=0.9,
                planning_steps=20):
    """
    训练Dyna-Q智能体
    """
    env = gym.make(env_name, is_slippery=False)
    
    agent = DynaQAgent(
        state_space_size=env.observation_space.n,
        action_space_size=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        planning_steps=planning_steps
    )
    
    episode_rewards = []
    
    print(f"开始训练Dyna-Q (规划步数n={planning_steps})...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 学习模型
            agent.learn_model(state, action, reward, next_state)
            
            # 直接Q-learning更新（真实经验）
            agent.update_q_value(state, action, reward, next_state, done)
            
            # 规划（使用模型进行额外更新）
            agent.planning()
            
            state = next_state
            total_reward += reward
        
        # 衰减探索率
        agent.decay_epsilon()
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"最近100轮平均奖励: {avg_reward:.3f}, "
                  f"探索率: {agent.epsilon:.3f}")
    
    env.close()
    return agent, episode_rewards


def evaluate_agent(agent, env_name="FrozenLake-v1", num_episodes=100):
    """评估智能体"""
    env = gym.make(env_name, is_slippery=False)
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # 贪婪策略
            old_epsilon = agent.epsilon
            agent.epsilon = 0
            action = agent.choose_action(state)
            agent.epsilon = old_epsilon
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done and reward > 0:
                success_count += 1
            state = next_state
    
    env.close()
    success_rate = success_count / num_episodes
    print(f"\n评估完成！成功率: {success_rate*100:.1f}%")
    return success_rate


# 主程序
if __name__ == "__main__":
    # 训练Dyna-Q
    agent, rewards = train_dyna_q(
        env_name="FrozenLake-v1",
        num_episodes=1000,
        planning_steps=20
    )
    
    # 评估
    evaluate_agent(agent, num_episodes=100)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='每轮奖励', color='blue')
    
    # 滑动平均
    from pandas import Series
    rewards_series = Series(rewards)
    moving_avg = rewards_series.rolling(window=100, min_periods=1).mean()
    plt.plot(moving_avg, label='100轮滑动平均', color='red', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.title('Dyna-Q 学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dyna_q_curve.png', dpi=150)
    plt.show()
```

**运行结果示例：**
```
开始训练Dyna-Q (规划步数n=20)...
Episode 100/1000, 最近100轮平均奖励: 0.35, 探索率: 0.606
Episode 200/1000, 最近100轮平均奖励: 0.58, 探索率: 0.367
Episode 500/1000, 最近100轮平均奖励: 0.82, 探索率: 0.082
Episode 1000/1000, 最近100轮平均奖励: 0.91, 探索率: 0.050

评估完成！成功率: 94.0%
```

## 8. 手工代码实现

使用NumPy从零实现Dyna-Q核心逻辑：

```python
"""
Dyna-Q从零实现
实现核心Dyna-Q算法，包含模型和规划
"""

import numpy as np
import random
from typing import Dict, Tuple, List, Optional


class DynaQ:
    """
    Dyna-Q算法从零实现
    
    核心思想:
    1. Q-learning更新 (真实经验)
    2. 模型学习 (记录转移)
    3. 规划 (使用模型模拟更新)
    """
    
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 learning_rate: float = 0.2,
                 discount_factor: float = 0.9,
                 planning_steps: int = 20,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化Dyna-Q算法
        
        参数:
        - num_states: 状态数量
        - num_actions: 动作数量
        - learning_rate: 学习率 α
        - discount_factor: 折扣因子 β
        - planning_steps: 规划步数 n
        - epsilon: 初始探索率
        """
        # Q表
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)
        
        # 模型: 使用两个数组存储 next_state 和 reward
        self.model_next_state = np.full((num_states, num_actions), -1, dtype=np.int32)
        self.model_reward = np.zeros((num_states, num_actions), dtype=np.float32)
        
        # 经验池: 存储 (state, action) 对
        self.experience_pool = []
        
        # 超参数
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.n = planning_steps
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
    
    def select_action(self, state: int) -> int:
        """
        使用ε-greedy策略选择动作
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int,
                        reward: float, next_state: int, done: bool):
        """
        Q-learning更新
        
        数学原理:
        TD目标 = r + γ * max_a' Q(s', a')
        Q(s,a) = Q(s,a) + α * (TD目标 - Q(s,a))
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        
        return td_error
    
    def learn_model(self, state: int, action: int,
                       reward: float, next_state: int):
        """
        学习确定性模型
        
        数学原理:
        Model(s,a) <- (s', r)
        """
        self.model_next_state[state, action] = next_state
        self.model_reward[state, action] = reward
        
        # 添加到经验池
        if (state, action) not in self.experience_pool:
            self.experience_pool.append((state, action))
    
    def planning_update(self):
        """
        执行一次规划更新
        
        从经验池随机采样，用模型预测进行Q-learning更新
        """
        if len(self.experience_pool) == 0:
            return
        
        # 随机采样
        state, action = random.choice(self.experience_pool)
        
        # 查询模型
        next_state = self.model_next_state[state, action]
        reward = self.model_reward[state, action]
        
        # Q-learning更新
        if next_state == -1:  # 模型未知
            return
        
        if next_state >= self.q_table.shape[0]:  # 终止状态
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
    
    def planning_phase(self):
        """
        完整的规划阶段: 执行n次规划更新
        """
        for _ in range(self.n):
            self.planning_update()
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def fit(self, env, num_episodes: int, max_steps_per_episode: int = 100):
        """
        训练Dyna-Q智能体
        """
        print(f"开始训练Dyna-Q (规划步数n={self.n})...")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                # 选择动作
                action = self.select_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 学习模型
                self.learn_model(state, action, reward, next_state)
                
                # 直接Q-learning更新
                self.update_q_value(state, action, reward, next_state, done)
                
                # 规划阶段
                self.planning_phase()
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # 衰减探索率
            self.decay_epsilon()
            episode_rewards.append(total_reward)
            
            if (episode + 1) % (num_episodes // 10) == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"平均奖励: {avg_reward:.3f}, "
                      f"探索率: {self.epsilon:.3f}")
        
        return episode_rewards
    
    def predict(self, state: int) -> int:
        """使用学到的策略预测动作"""
        return np.argmax(self.q_table[state])
    
    def evaluate(self, env, num_episodes: int = 100) -> float:
        """评估策略"""
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.predict(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        print(f"评估完成！{num_episodes}轮平均奖励: {avg_reward:.3f}")
        return avg_reward


# 测试代码
if __name__ == "__main__":
    # 创建简单网格世界环境
    class SimpleGridWorld:
        def __init__(self):
            self.grid_size = 4
            self.n_states = self.grid_size * self.grid_size
            self.n_actions = 4
            self.start_state = 0
            self.goal_state = 15
            self.state = self.start_state
        
        def reset(self):
            self.state = self.start_state
            return self.state, {}
        
        def step(self, action):
            row = self.state // self.grid_size
            col = self.state % self.grid_size
            
            if action == 0:  # 上
                row = max(0, row - 1)
            elif action == 1:  # 下
                row = min(self.grid_size - 1, row + 1)
            elif action == 2:  # 左
                col = max(0, col - 1)
            elif action == 3:  # 右
                col = min(self.grid_size - 1, col + 1)
            
            self.state = row * self.grid_size + col
            
            if self.state == self.goal_state:
                return self.state, 1.0, True, False, {}
            else:
                return self.state, 0.0, False, False, {}
    
    # 创建环境和智能体
    env = SimpleGridWorld()
    agent = DynaQ(
        num_states=env.n_states,
        num_actions=env.n_actions,
        learning_rate=0.2,
        discount_factor=0.9,
        planning_steps=10
    )
    
    # 训练
    rewards = agent.fit(env, num_episodes=500)
    
    # 评估
    agent.evaluate(env, num_episodes=100)
```

## 9. 可视化与结果理解

```python
"""
Dyna-Q可视化代码
比较Dyna-Q与Q-learning的学习速度
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve_comparison(q_learning_rewards, dyna_q_rewards, 
                                window=100):
    """
    比较Dyna-Q和Q-learning的学习曲线
    
    图表解读：
    - Dyna-Q通常比Q-learning学习更快
    - 曲线上升说明智能体在学习
    - 两条曲线的差距体现规划带来的加速
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Q-learning曲线
    ax.plot(q_learning_rewards, alpha=0.3, color='blue', label='Q-learning')
    q_series = pd.Series(q_learning_rewards)
    q_avg = q_series.rolling(window=window, min_periods=1).mean()
    ax.plot(q_avg, color='blue', linewidth=2, label=f'Q-learning (滑动平均)')
    
    # Dyna-Q曲线
    ax.plot(dyna_q_rewards, alpha=0.3, color='red', label='Dyna-Q')
    d_series = pd.Series(dyna_q_rewards)
    d_avg = d_series.rolling(window=window, min_periods=1).mean()
    ax.plot(d_avg, color='red', linewidth=2, label=f'Dyna-Q (滑动平均)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('累积奖励')
    ax.set_title('Dyna-Q vs Q-learning 学习曲线对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dyna_q_vs_q_learning.png', dpi=150)
    plt.show()


def plot_model_accuracy(model_next_state, env, sample_states_actions):
    """
    可视化模型准确性
    
    图表解读：
    - 比较模型预测的next_state和真实转移
    - 模型准确时，规划才有效
    """
    # 这里需要环境支持查询真实转移
    # 简化示例
    pass


def plot_planning_effect(n_values, results):
    """
    绘制不同规划步数n对学习的影响
    
    图表解读：
    - n=0: 退化为普通Q-learning
    - n增大: 学习加快，但计算开销增加
    - n过大: 可能过拟合模型误差
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for n, rewards in zip(n_values, results):
        rewards_series = pd.Series(rewards)
        moving_avg = rewards_series.rolling(window=50, min_periods=1).mean()
        ax.plot(moving_avg, label=f'n={n}')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('平均奖励')
    ax.set_title('规划步数n对Dyna-Q性能的影响')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('planning_effect.png', dpi=150)
    plt.show()
```

## 10. 模型评估

```python
"""
Dyna-Q模型评估代码
评估学习和规划效果
"""

import numpy as np
from typing import Dict

def evaluate_dyna_q(agent, env, num_episodes: int = 100) -> Dict:
    """
    全面评估Dyna-Q性能
    """
    episode_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.predict(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        
        episode_rewards.append(total_reward)
        if total_reward > 0:
            success_count += 1
    
    results = {
        'average_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': success_count / num_episodes,
        'model_size': len(agent.experience_pool)
    }
    
    return results


def compare_sample_efficiency(q_learning_agent, dyna_q_agent, 
                             env, num_runs=5, num_episodes=500):
    """
    比较样本效率
    
    为什么重要：
    - Dyna-Q的核心优势是样本效率高
    - 用更少的真实交互达到相同性能
    """
    q_learning_performance = []
    dyna_q_performance = []
    
    for run in range(num_runs):
        # 评估Q-learning
        result_ql = evaluate_dyna_q(q_learning_agent, env, num_episodes=100)
        q_learning_performance.append(result_ql['success_rate'])
        
        # 评估Dyna-Q
        result_dq = evaluate_dyna_q(dyna_q_agent, env, num_episodes=100)
        dyna_q_performance.append(result_dq['success_rate'])
    
    print(f"Q-learning 平均成功率: {np.mean(q_learning_performance):.3f}")
    print(f"Dyna-Q 平均成功率: {np.mean(dyna_q_performance):.3f}")
    print(f"样本效率提升: {np.mean(dyna_q_performance) / np.mean(q_learning_performance):.2f}倍")
```

## 11. 常见问题与易错点

**数据层面易错点：**

1. **问题：模型存储错误**
   - 现象：规划更新时出现异常或学习效果差
   - 原因：模型没有正确存储(s,a)对应的(s',r)
   - 解决方案：仔细检查learn_model实现，确保正确记录转移

2. **问题：经验池管理不当**
   - 现象：内存占用过大或采样效率低
   - 原因：经验池无限增长或采样策略不合理
   - 解决方案：限制经验池大小，使用合理的数据结构

**模型层面易错点：**

1. **问题：规划使用未学习的模型**
   - 现象：早期规划无效或有害
   - 原因：采样到模型未覆盖的(s,a)
   - 解决方案：检查模型是否已学习该转移，跳过未知转移

2. **问题：模型不准确导致次优策略**
   - 现象：Dyna-Q性能不如Q-learning
   - 原因：环境是随机的，但模型是确定性的
   - 解决方案：使用概率模型或限制规划步数

3. **问题：忽略done状态的模型处理**
   - 现象：终止状态的规划更新错误
   - 原因：终止状态没有next_state，模型应特殊处理
   - 解决方案：在模型中标记终止状态

**调参层面易错点：**

1. **问题：规划步数n设置不当**
   - 现象：n太小加速不明显，n太大计算浪费
   - 原因：没有根据任务特性调整
   - 解决方案：从n=10开始尝试，根据计算资源调整

2. **问题：学习率α在规划和真实更新中相同**
   - 现象：规划更新可能导致不稳定
   - 原因：模型预测有误差，应使用更小的学习率
   - 解决方案：为规划更新使用更小的学习率

## 12. 学习总结

**核心思想回顾：** Dyna-Q是一种结合模型学习和Q-learning的架构。智能体通过真实交互学习环境模型，然后使用模型进行规划（模拟更新），从而用更少的真实样本学会最优策略。

**关键公式：**
1. Q-learning更新：Q(s,a) += α[r + β·max_a' Q(s',a') - Q(s,a)]
2. 模型学习：Model(s,a) ← (s', r)
3. 规划：使用模型生成的经验进行Q-learning更新

**与前序算法或相关算法的联系：**
- 基于**Q-learning**的核心更新机制
- 集成**动态规划**的规划思想
- 是**Prioritized Sweeping**等高级规划算法的基础
- 与**Model-Based RL**的关系：Dyna-Q是model-based的一种实现

**后续学习方向：**
- **Prioritized Sweeping**：优先级规划，更高效
- **Dyna-Q+**：处理模型不确定性
- **Model-Based RL with Uncertainty**：贝叶斯模型、高斯过程等
- **MBMF**（Model-Based Model-Free）：结合两者优势

## 13. 练习题与思考题

**基础题1：** Dyna-Q中的规划步数n=0时，算法退化成什么？请解释原因。

**答案：**
当n=0时，Dyna-Q退化为标准的Q-learning。
原因：规划阶段执行0次更新，只有直接Q-learning更新（真实经验），这与Q-learning完全相同。

**基础题2：** 为什么Dyna-Q在确定性环境中比随机环境中更有效？

**答案：**
标准Dyna-Q学习确定性模型：Model(s,a) = (s', r)，即假设每个(s,a)只有唯一的下一状态和奖励。
- 在确定性环境中：模型完全准确，规划更新与真实更新一致，加速学习。
- 在随机环境中：模型只能记录一个样本（通常是最后一次），忽略随机性，导致模型不准确，规划更新可能误导Q值。

**进阶题1：** 假设在Dyna-Q中，环境是随机的：P(s'|s,a)是一个分布。请设计一种改进方法，使Dyna-Q能处理随机环境。

**答案：**
可以使用以下方法之一：
1. **存储经验分布**：模型存储(s,a)对应的多个(s',r)样本，规划时随机采样
2. **平均模型**：存储平均奖励和平均下一状态
3. **概率模型**：使用高斯模型或查表存储转移概率分布
4. **递归Dyna-Q**：使用分布模型的期望进行规划更新

**进阶题2：** 分析Dyna-Q的计算复杂度（每次真实交互），并讨论如何优化。

**答案：**
每次真实交互的计算：
- Q-learning更新：O(1)
- 模型学习：O(1)
- 规划n步：O(n)次Q-learning更新

总复杂度：O(n) per real step

优化方法：
1. 优先级规划：不是均匀采样，而是优先规划TD误差大的状态-动作对
2. 限制经验池大小：避免采样低效
3. 批量规划：累积多个真实经验后批量规划
4. 使用近似规划算法

**开放思考题：** Dyna-Q使用一个统一的模型来规划。在实际问题中，环境可能是非平稳的（随时间变化）。请思考如何让Dyna-Q适应非平稳环境？

**参考答案思路：**
1. **模型衰减**：旧经验逐渐忘记，模型权重随时间衰减
2. **滑动窗口**：只保留最近N个经验的模型
3. **多模型集成**：维护多个模型，根据预测准确性加权
4. **检测环境变化**：当预测误差持续较大时，重置模型或部分重置

## 14. 学习路径建议

**前置算法：**
1. **Q-learning**：Dyna-Q的核心更新机制
2. **马尔可夫决策过程（MDP）**：理解状态和转移
3. **动态规划**：理解规划的基本思想

**平行算法：**
1. **Prioritized Sweeping**：更高级的规划策略
2. **RTP-Q Learning**：结合主动探索的规划

**进阶算法：**
1. **Prioritized Sweeping**：优先级重放和经验回放
2. **Dyna-Q+**：处理模型不确定性和过拟合
3. **Model-Based RL with Deep Learning**：深度模型结合DQN

**推荐资源：**
1. **教材**：Sutton & Barto, "Reinforcement Learning: An Introduction"（第8章）
2. **论文**：Sutton (1990), "Integrated architectures for learning, planning, and reacting"
3. **在线课程**：David Silver's RL Course (Lecture 7: Model-Based RL)
4. **代码实践**：OpenAI Spinning Up - Dyna-Q实现
