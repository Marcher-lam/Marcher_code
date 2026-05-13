# RTP-Q Learning 学习文档#

> 结合实时动态规划和主动探索规划，加速强化学习。

## 1. 算法基础认知#

**一句话定义：** RTP-Q（Real-Time Planning Q-learning）是一种结合模型学习、优先级规划和主动探索的强化学习架构，通过时间约束探索规划加速学习。

**直觉类比：** 想象你在一个新城市找路，不仅用地图模拟路线（规划），还会特意去探索尚未访问的区域（主动探索），这样既高效又不会遗漏重要地点。

**历史背景：** RTP-Q由赵刚等人在1999年提出，是Dyna-Q的改进版本。它解决了Dyna-Q中缺乏主动探索机制的问题，通过引入"子目标"概念加速学习。

**算法定位：** 基于模型的强化学习架构，集成Q-learning、模型学习和主动探索规划。

**前置知识：**
- Q-learning基础
- Dyna-Q
- Prioritized Sweeping
- Python编程#

## 2. 核心原理#

**核心思想：** RTP-Q在Dyna-Q基础上增加了主动探索规划（Active Exploration Planning, AEP）。当智能体发现未充分探索的状态时，将其设为"子目标"，并利用模型规划到达该子目标的路径，从而更高效地探索环境。

**工作流程：**
1. 初始化Q表、模型、子目标集
2. 对于每个时间步：
   a. **主动探索检查：** 如果当前状态有满足子目标条件的规则(s,a)，随机执行一个
   b. 否则按常规选择动作（ε-greedy）
   c. 执行动作，获得(s, a, r, s')
   d. **直接更新：** Q-learning更新Q(s,a)
   e. **学习模型：** Model(s,a) ← (s', r)
   f. **规划阶段（n次）：**
      - 从经验池采样或选择子目标相关状态
      - 用模型预测并Q-learning更新
   g. **子目标更新：** 检查是否需要设置新子目标

**关键概念解释：**
- **子目标（Sub-goal）：** 满足特定条件的状态，如"未充分访问"或"Q值为0"
- **主动探索规划（AEP）：** 主动寻找并规划到达未探索区域的路径
- **子奖励（Sub-reward）：** 到达子目标时给予的额外奖励，用于传播学习信号
- **时间约束：** RTP-Q考虑规划的时间成本，保证实时性

**几何/直观解释：**
```
RTP-Q架构示意图：

[智能体] --(s,a)--> [真实环境] --(r,s')--> [智能体]
     ↓                                           ↑
[主动探索规划器] -- 设置子目标                   
     ↑              ↓
[内部模型] <-- 学习模型 -- [智能体]

工作流程：
1. 发现Q(s,a)=0且未访问的规则 → 设为子目标
2. 使用模型规划到达子目标的路径
3. 执行规划路径，获得子奖励
4. 子奖励通过Q值反向传播
```

## 3. 数学公式与推导#

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| SG | 子目标集 | 满足特定条件的状态-动作对 |
| r_sub | 子奖励 | 到达子目标时的奖励 |
| H(s,a) | H值 | 子奖励的折扣传播值 |
| β₂ | 探索折扣因子 | 用于主动探索，通常<β |

**子目标定义：**

满足以下任一条件的状态-动作对(s,a)可设为子目标：

1. **Q值为0且未访问：** Q(s,a) = 0 且 flag(s,a) = 0
2. **未来价值更高：** β²·max Q(s',a') - Q(s,a) > 0

数学表达：
$$(s,a) \in SG \iff \left\{ \begin{array}{l} Q(s,a) = 0 \land \text{flag}(s,a) = 0 \\ \text{or} \\ \beta^2 \max_{a'} Q(s',a') - Q(s,a) > 0 \end{array} \right.$$

**H值（子奖励传播）：**

当设置子目标(s,a)时，计算H值用于向回传播：

$$H(s,a) = \beta_2 \cdot \max_{a'} Q(s',a')$$

其中β₂ ≤β，用于折扣子奖励的传播。

**Q-learning更新（同Dyna-Q）：**

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \beta \max_{a'} Q(s',a') - Q(s,a) \right]$$

**完整RTP-Q算法：**

对每个时间步t：
1. **检查子目标：** 如果当前状态有(s,a) ∈ SG，随机执行一个
2. 否则使用ε-greedy选择动作
3. 执行动作，获得r_{t+1}, s_{t+1}
4. **直接更新：** Q(s_t, a_t) += α[r_{t+1} + β·max Q(s_{t+1},a') - Q(s_t,a_t)]
5. **学习模型：** Model(s_t, a_t) = (s_{t+1}, r_{t+1})
6. **规划阶段：**
   - 对i = 1到n：
     a. 选择(s,a)：要么是子目标，要么从经验池采样
     b. 查询模型：(s',r) = Model(s,a)
     c. Q(s,a) += α[r + β·max Q(s',a') - Q(s,a)]
     d. 如果(s,a)是子目标，更新H值并传播
7. **子目标管理：** 移除已充分学习的子目标，添加新发现的

**为什么加速学习：**
1. **主动探索：** 不等待随机探索，主动寻找未学习区域
2. **子奖励传播：** 通过H值将学习信号快速传播
3. **模型利用：** 用模型规划，减少真实交互需求

## 4. 训练过程讲解#

**数据预处理：**
- 状态-动作对存储：记录访问次数、Q值等
- 子目标管理：维护子目标集合及其H值
- 模型表示：表格或函数逼近

**参数初始化：**
- Q表：全0
- 模型：空
- 子目标集：空
- 学习率α：0.1~0.5
- 折扣因子β：0.9~0.99，β₂：0.7~0.9
- 规划步数n：10~100
- 子奖励r_sub：通常设为1.0

**迭代过程：**
1. 每个episode开始，重置环境
2. 在每步中：
   - **子目标检查：** 如果当前状态有子目标规则，随机选一个执行
   - 否则常规选择动作
   - 执行动作，获得经验
   - Q-learning更新
   - 学习模型
   - **规划：**
     * 优先处理子目标相关的更新
     * 从经验池采样补充
   - **子目标更新：**
     * 检查新状态是否应设为子目标
     * 移除已学习的子目标（Q值已收敛）
3. 直到终止

**收敛条件：**
- Q值稳定
- 子目标集空（所有重要状态已学习）
- 达到最大episode数

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| α | 学习率 | 0.1~0.5 | 0.2 |
| β | 折扣因子 | 0.9~0.99 | 0.9 |
| β₂ | 探索折扣因子 | 0.7~0.9 | 0.8 |
| n | 规划步数 | 10~100 | 20 |
| r_sub | 子奖励 | 0.5~2.0 | 1.0 |
| θ | 子目标阈值 | 0.01~0.1 | 0.05 |

## 5. 应用场景#

**典型应用：**

1. **迷宫求解：** 需要快速探索未知区域。**为什么适合：** 主动探索机制能快速找到未访问区域。
2. **机器人导航：** 未知环境中的路径规划。**为什么适合：** 结合模型和主动探索，减少真实移动成本。
3. **游戏AI：** 需要快速学习游戏机制。**为什么适合：** 子目标帮助智能体理解关键游戏状态。
4. **大规模MDP：** 状态空间巨大，随机探索低效。**为什么适合：** 主动探索针对性更强。

**适用数据特征：**
- 环境交互成本较高
- 需要系统性探索
- 有环境模型（或可学习）
- 存在"关键状态"需要优先学习

**不适用场景：**
- 完全随机环境（子目标不稳定）
- 计算资源极有限（规划+探索开销大）
- 实时性要求极高（规划需要时间）

## 6. 优缺点分析#

**优点：**
1. **探索效率高：** 主动寻找未学习区域。**成立条件：** 子目标定义合理。
2. **学习速度快：** 结合规划和主动探索。**成立条件：** 模型相对准确。
3. **系统性探索：** 不会遗漏重要状态。**成立条件：** 子目标覆盖关键区域。
4. **可结合优先级：** 可与Prioritized Sweeping结合。

**缺点：**
1. **子目标定义复杂：** 需要精心设计条件。**问题：** 不当定义导致探索低效。**缓解思路：** 根据任务特性调整子目标条件。
2. **计算开销更大：** 比Dyna-Q额外需要子目标管理。**问题：** 计算资源消耗增加。**缓解思路：** 限制子目标数量。
3. **对模型依赖更强：** 主动探索基于模型规划。**问题：** 模型错误会误导探索。**缓解思路：** 使用模型不确定性估计。

**与同类算法对比：**

| 特性 | Dyna-Q | RTP-Q | Prioritized Sweeping |
|------|---------|--------|---------------------|
| 主动探索 | 无 | 有 | 无 |
| 子目标 | 无 | 有 | 无 |
| 计算开销 | 中 | 中高 | 中高 |
| 探索效率 | 中 | 高 | 中 |

## 7. 调库实现#

```python
"""
RTP-Q算法调库实现
结合主动探索和规划
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import random
import heapq
import matplotlib.pyplot as plt

class RTPQAgent:
    """
    RTP-Q智能体
    结合Q-learning、模型学习、规划和主动探索
    """
    
    def __init__(self, state_space_size, action_space_size,
                 learning_rate=0.2, discount_factor=0.9,
                 explore_discount=0.8,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 planning_steps=20, subgoal_reward=1.0):
        """
        初始化RTP-Q智能体
        
        参数:
        - explore_discount: 探索折扣因子β₂
        """
        # Q表
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        
        # 模型
        self.model = {}
        
        # 子目标管理
        self.subgoals = {}  # (state_key, action) -> H_value
        
        # 访问计数
        self.visit_count = defaultdict(int)
        
        # 超参数
        self.lr = learning_rate
        self.gamma = discount_factor
        self.gamma2 = explore_discount  # β₂
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.planning_steps = planning_steps
        self.subgoal_reward = subgoal_reward
        self.action_space_size = action_space_size
        
        # 经验池
        self.experience_pool = []
    
    def choose_action(self, state):
        """
        选择动作：优先执行子目标规则
        """
        state_key = self._state_to_key(state)
        
        # 检查是否有子目标规则
        subgoal_actions = []
        for action in range(self.action_space_size):
            if (state_key, action) in self.subgoals:
                subgoal_actions.append(action)
        
        if subgoal_actions:
            # 随机执行一个子目标规则
            return random.choice(subgoal_actions)
        else:
            # 常规ε-greedy
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_space_size)
            else:
                q_values = self.q_table[state_key]
                return np.random.choice(np.where(q_values == q_values.max())[0])
    
    def check_subgoal(self, state, action, next_state):
        """
        检查是否需要设为子目标
        
        条件:
        1. Q(s,a)=0且未访问
        2. β²·max Q(s',a') - Q(s,a) > 0
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        q_value = self.q_table[state_key][action]
        visits = self.visit_count[(state_key, action)]
        
        # 条件1: Q=0且未充分访问
        if q_value == 0 and visits < 2:
            self.subgoals[(state_key, action)] = self.subgoal_reward
            return True
        
        # 条件2: 未来价值更高
        if not isinstance(next_state_key, str):
            max_next_q = np.max(self.q_table[next_state_key])
            if self.gamma**2 * max_next_q - q_value > 0:
                h_value = self.gamma2 * max_next_q
                self.subgoals[(state_key, action)] = h_value
                return True
        
        return False
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Q-learning更新"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error
        
        return abs(td_error)
    
    def planning_with_subgoals(self):
        """结合子目标的规划"""
        if not self.subgoals and not self.experience_pool:
            return
        
        steps = 0
        # 优先处理子目标
        subgoal_keys = list(self.subgoals.keys())
        
        for (state_key, action) in subgoal_keys:
            if steps >= self.planning_steps:
                break
            
            # 查询模型
            if (state_key, action) in self.model:
                next_state, reward = self.model[(state_key, action)]
                
                # 使用子奖励（如果是子目标）
                if (state_key, action) in self.subgoals:
                    reward = self.subgoal_reward
                
                # Q-learning更新
                self.update_q_value(state_key, action, reward, next_state, False)
                
                # 如果Q值已收敛，移除子目标
                if self.q_table[state_key][action] > 0.5:  # 简化判断
                    del self.subgoals[(state_key, action)]
            
            steps += 1
        
        # 补充：从经验池采样
        while steps < self.planning_steps and self.experience_pool:
            state_key, action = random.choice(self.experience_pool)
            
            if (state_key, action) in self.model:
                next_state, reward = self.model[(state_key, action)]
                self.update_q_value(state_key, action, reward, next_state, False)
            
            steps += 1
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _state_to_key(self, state):
        """状态转键"""
        if isinstance(state, (int, np.integer)):
            return state
        return tuple(state) if isinstance(state, np.ndarray) else state


def train_rtpq(env_name="FrozenLake-v1", num_episodes=1000,
               learning_rate=0.2, discount_factor=0.9,
               planning_steps=20):
    """训练RTP-Q智能体"""
    env = gym.make(env_name, is_slippery=False)
    
    agent = RTPQAgent(
        state_space_size=env.observation_space.n,
        action_space_size=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        planning_steps=planning_steps
    )
    
    episode_rewards = []
    
    print(f"开始训练RTP-Q (规划步数n={planning_steps})...")
    
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
            state_key = agent._state_to_key(state)
            agent.model[(state_key, action)] = (next_state, reward)
            
            # 更新访问计数
            agent.visit_count[(state_key, action)] += 1
            
            # 检查子目标
            agent.check_subgoal(state, action, next_state)
            
            # 直接Q-learning更新
            agent.update_q_value(state, action, reward, next_state, done)
            
            # 添加到经验池
            if (state_key, action) not in agent.experience_pool:
                agent.experience_pool.append((state_key, action))
            
            # 规划阶段
            agent.planning_with_subgoals()
            
            state = next_state
            total_reward += reward
        
        # 衰减探索率
        agent.decay_epsilon()
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"平均奖励: {avg_reward:.3f}, "
                  f"子目标数: {len(agent.subgoals)}, "
                  f"探索率: {agent.epsilon:.3f}")
    
    env.close()
    return agent, episode_rewards


# 主程序
if __name__ == "__main__":
    agent, rewards = train_rtpq(
        env_name="FrozenLake-v1",
        num_episodes=1000,
        planning_steps=20
    )
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='每轮奖励', color='blue')
    
    from pandas import Series
    rewards_series = Series(rewards)
    moving_avg = rewards_series.rolling(window=100, min_periods=1).mean()
    plt.plot(moving_avg, label='100轮滑动平均', color='red', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.title('RTP-Q 学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rtpq_curve.png', dpi=150)
    plt.show()
```

## 8. 手工代码实现#

```python
"""
RTP-Q从零实现
实现主动探索和子目标机制
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional


class RTPQ:
    """
    RTP-Q算法从零实现
    """
    
    def __init__(self, 
                 num_states: int,
                 num_actions: int,
                 learning_rate: float = 0.2,
                 discount_factor: float = 0.9,
                 explore_discount: float = 0.8,
                 planning_steps: int = 20):
        """初始化"""
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)
        self.model_next_state = np.full((num_states, num_actions), -1, dtype=np.int32)
        self.model_reward = np.zeros((num_states, num_actions), dtype=np.float32)
        
        # 子目标: 存储 (state, action) -> h_value
        self.subgoals = {}
        
        # 访问计数
        self.visit_count = np.zeros((num_states, num_actions), dtype=np.int32)
        
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.gamma2 = explore_discount
        self.n = planning_steps
        self.num_actions = num_actions
        
        self.experience_pool = []
    
    def select_action(self, state: int) -> int:
        """选择动作，优先子目标"""
        # 检查子目标
        subgoal_actions = []
        for action in range(self.num_actions):
            if (state, action) in self.subgoals:
                subgoal_actions.append(action)
        
        if subgoal_actions:
            return random.choice(subgoal_actions)
        
        # ε-greedy
        if random.random() < 0.1:
            return random.randint(0, self.num_actions - 1)
        return np.argmax(self.q_table[state])
    
    def check_and_set_subgoal(self, state: int, action: int, next_state: int):
        """检查并设置子目标"""
        q_value = self.q_table[state, action]
        visits = self.visit_count[state, action]
        
        # 条件1
        if q_value == 0 and visits < 2:
            self.subgoals[(state, action)] = self.subgoal_reward
            return True
        
        # 条件2
        if next_state >= 0 and next_state < self.q_table.shape[0]:
            max_next_q = np.max(self.q_table[next_state])
            if self.gamma**2 * max_next_q - q_value > 0:
                h_value = self.gamma2 * max_next_q
                self.subgoals[(state, action)] = h_value
                return True
        
        return False
    
    def update_q_value(self, state: int, action: int,
                        reward: float, next_state: int, done: bool):
        """Q-learning更新"""
        if done or next_state < 0:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return abs(td_error)
    
    def planning(self):
        """规划阶段"""
        steps = 0
        
        # 处理子目标
        subgoal_items = list(self.subgoals.items())
        for (state, action), h_value in subgoal_items:
            if steps >= self.n:
                break
            
            if self.model_next_state[state, action] >= 0:
                next_state = self.model_next_state[state, action]
                reward = self.subgoal_reward  # 使用子奖励
                
                self.update_q_value(state, action, reward, next_state, False)
                
                # 检查是否移除子目标
                if self.q_table[state, action] > 0.5:
                    del self.subgoals[(state, action)]
            
            steps += 1
        
        # 从经验池补充
        while steps < self.n and self.experience_pool:
            state, action = random.choice(self.experience_pool)
            
            if self.model_next_state[state, action] >= 0:
                next_state = self.model_next_state[state, action]
                reward = self.model_reward[state, action]
                self.update_q_value(state, action, reward, next_state, False)
            
            steps += 1
```

## 9. 可视化与结果理解#

```python
"""
RTP-Q可视化
比较RTP-Q、Dyna-Q、Q-learning的学习曲线
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_comparison(q_learning_rewards, dyna_q_rewards, rtpq_rewards):
    """比较三种算法的学习曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Q-learning
    q_series = pd.Series(q_learning_rewards)
    ax.plot(q_learning_rewards, alpha=0.3, color='blue', label='Q-learning')
    ax.plot(q_series.rolling(window=100).mean(), color='blue', linewidth=2, label='Q-learning (平均)')
    
    # Dyna-Q
    d_series = pd.Series(dyna_q_rewards)
    ax.plot(dyna_q_rewards, alpha=0.3, color='green', label='Dyna-Q')
    ax.plot(d_series.rolling(window=100).mean(), color='green', linewidth=2, label='Dyna-Q (平均)')
    
    # RTP-Q
    r_series = pd.Series(rtpq_rewards)
    ax.plot(rtpq_rewards, alpha=0.3, color='red', label='RTP-Q')
    ax.plot(r_series.rolling(window=100).mean(), color='red', linewidth=2, label='RTP-Q (平均)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('累积奖励')
    ax.set_title('Q-learning vs Dyna-Q vs RTP-Q')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rtpq_comparison.png', dpi=150)
    plt.show()


def plot_subgoals_over_time(subgoal_counts):
    """绘制子目标数量变化"""
    plt.figure(figsize=(10, 6))
    plt.plot(subgoal_counts, color='purple', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('子目标数量')
    plt.title('RTP-Q 子目标数量变化')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rtpq_subgoals.png', dpi=150)
    plt.show()
```

## 10. 模型评估#

```python
"""
RTP-Q评估代码
"""

import numpy as np
from typing import Dict

def evaluate_rtpq(agent, env, num_episodes: int = 100) -> Dict:
    """评估RTP-Q性能"""
    episode_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = np.argmax(agent.q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        if episode_reward > 0:
            success_count += 1
    
    results = {
        'average_reward': np.mean(episode_rewards),
        'success_rate': success_count / num_episodes,
        'num_subgoals': len(agent.subgoals)
    }
    return results
```

## 11. 常见问题与易错点#

**数据层面易错点：**

1. **问题：子目标条件设计不当**
   - 现象：子目标过多或过少
   - 原因：阈值设置不合理
   - 解决方案：根据任务调整Q值和访问次数阈值

**模型层面易错点：**

1. **问题：子目标未及时移除**
   - 现象：已学习的状态仍被视为子目标
   - 原因：未检查Q值收敛
   - 解决方案：定期检查并移除已收敛的子目标

**调参层面易错点：**

1. **问题：β₂设置不当**
   - 现象：子奖励传播过强或过弱
   - 原因：β₂与β关系不合理
   - 解决方案：β₂应小于β，通常β₂=0.8*β

## 12. 学习总结#

**核心思想回顾：** RTP-Q在Dyna-Q基础上增加主动探索规划。通过子目标机制，智能体主动寻找未充分学习的区域，并用模型规划到达路径，从而加速学习。

**关键公式：**
1. 子目标条件：Q(s,a)=0且未访问，或β²·max Q(s',a') - Q(s,a) > 0
2. H值：H(s,a) = β₂·max Q(s',a')

**后续学习方向：**
- **Q-ae Learning**：本书提出的另一种主动探索方法
- **结合Prioritized Sweeping**：优先级+主动探索

## 13. 练习题与思考题#

**基础题1：** RTP-Q中的β₂（探索折扣因子）与β（折扣因子）有何不同？为何β₂通常小于β？

**答案：**
- β用于常规Q-learning更新，衡量未来奖励的一般价值
- β₂专门用于子奖励传播，通常β₂ < β
- 原因：子目标相关的探索是短期行为，不需要长期折扣

**开放思考题：** 在什么情况下RTP-Q可能不如Dyna-Q？为什么？

**参考答案：** 
- 环境高度随机：子目标不稳定，频繁变化
- 计算资源严格受限：管理子目标需要额外开销
- 环境很小：主动探索的优势不明显

## 14. 学习路径建议#

**前置算法：**
1. **Q-learning**
2. **Dyna-Q**
3. **Prioritized Sweeping**

**进阶算法：**
1. **Q-ae Learning**（本书第2章）：另一种主动探索
2. **结合Prioritized Sweeping**：更高效的规划

**推荐资源：**
1. **论文**：Zhao, Tatsumi & Sun (1999), "RTP-Q: A reinforcement learning system with time constraints exploration planning"
2. **相关章节**：本书第4章


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述RTP-Q_Learning的核心思想及适用场景。
<details><summary>参考答案</summary>
RTP-Q_Learning通过数据驱动学习输入到输出的映射，适用于强化学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出RTP-Q_Learning的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现RTP-Q_Learning核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. RTP-Q_Learning在什么情况下会失效？
2. 训练数据很少时，RTP-Q_Learning还能有效工作吗？
3. 如何将RTP-Q_Learning与其他方法结合？

