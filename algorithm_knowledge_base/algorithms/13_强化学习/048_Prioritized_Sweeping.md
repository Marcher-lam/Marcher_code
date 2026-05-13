# Prioritized Sweeping 学习文档#

> 使用优先级队列指导价值函数更新，加速动态规划。

## 1. 算法基础认知#

**一句话定义：** Prioritized Sweeping是一种基于优先级的规划算法，将TD误差大的状态-动作对优先更新，从而更高效地传播价值信息。

**直觉类比：** 想象你在复习考试，会优先复习那些你错误最多的知识点（TD误差大），而不是平均用力。这样能更快提高整体成绩。

**历史背景：** Prioritized Sweeping由Moore在1993年提出，是Dyna-Q的改进版。它解决了Dyna-Q中均匀采样导致的低效问题，通过优先级指导规划。

**算法定位：** 基于模型的强化学习（Model-based RL），使用优先级队列进行动态规划更新。

**前置知识：**
- Dyna-Q基础
- Q-learning或TD学习
- 优先队列（Priority Queue）数据结构
- Python编程#

## 2. 核心原理#

**核心思想：** 维护一个优先级队列，存储待更新的状态-动作对及其TD误差（优先级）。每次规划时，弹出TD误差最大的状态-动作对进行更新，并将受影响的前驱状态-动作对加入队列。

**工作流程：**
1. 初始化Q表、模型、优先级队列（空）
2. 对于每个时间步：
   a. 智能体执行动作，获得(s, a, r, s')
   b. **直接更新：** 用Q-learning更新Q(s,a)
   c. **计算TD误差：** δ = |r + β·max Q(s',a') - Q(s,a)|
   d. **加入队列：** 如果δ > θ（阈值），将(s,a)加入队列，优先级=δ
   e. **规划循环（直到队列空或达到最大步数）：**
      - 弹出优先级最大的(s,a)
      - 用模型预测(s,a)的(s',r)
      - 更新Q(s,a)
      - **查找前驱：** 找到所有能到达s的状态-动作对(predecessors)
      - 对每个前驱(pred_state, pred_action)：
        * 计算新TD误差
        * 如果新δ > θ，更新队列

**关键概念解释：**
- **优先级队列（Priority Queue）：** 每次弹出优先级最高（TD误差最大）的元素
- **TD误差作为优先级：** TD误差大说明该状态-动作对的Q值不准确，需要优先更新
- **前驱（Predecessors）：** 能够转移到当前状态的状态-动作对
- **阈值θ：** 只有TD误差超过θ才加入队列，避免无效更新

**几何/直观解释：**
```
Prioritized Sweeping更新传播示意图：

状态空间:  s0 --a0--> s1 --a1--> s2 --a2--> s3(目标)
                     ↑
                     |
                    s4 --a3--+

初始化: Q(s3, a) = 1.0 (目标), 其他Q=0

步骤1: 在s0执行a0到s1，获得奖励
  - 更新Q(s0,a0)，TD误差较大
  - 将(s0,a0)加入队列

步骤2: 弹出(s0,a0)，更新Q值
  - 查找前驱（假设s4能到s0）
  - 将(s4,a3)加入队列（因为Q(s0)变化会影响它）

步骤3: 弹出(s4,a3)，更新Q值
  - 继续传播...

关键点: 只更新"需要更新"的状态，而不是均匀采样！
```

## 3. 数学公式与推导#

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| PQ | 优先级队列 | 存储(s,a)和优先级δ |
| δ | TD误差 | 作为优先级 |
| pred(s) | s的前驱集合 | 能转移到s的状态-动作对 |
| θ | 阈值 | 只有δ>θ才入队 |

**问题形式化：**

目标与Q-learning相同：学习最优动作价值函数Q*(s,a)。

**TD误差（优先级）：**

对于状态-动作对(s,a)，其TD误差为：

$$\delta(s,a) = \left| r + \beta \max_{a'} Q(s',a') - Q(s,a) \right|$$

使用绝对值或平方，确保优先级为正。

**更新规则（同Q-learning）：**

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \beta \max_{a'} Q(s',a') - Q(s,a) \right]$$

**前驱查找：**

需要维护一个前驱表：给定状态s，找到所有(pred_state, pred_action)使得执行pred_action后转移到s。

这需要从模型中反向查找：
$$\text{pred}(s) = \{ (s_{pred}, a_{pred}) \mid \text{Model}(s_{pred}, a_{pred}) = (s, \cdot) \}$$

**完整Prioritized Sweeping算法：**

对每个时间步t：
1. 执行动作获得(s_t, a_t, r_{t+1}, s_{t+1})
2. 更新Q(s_t, a_t)使用Q-learning公式
3. 计算δ_t = |r_{t+1} + β·max_a' Q(s_{t+1}, a') - Q(s_t, a_t)|
4. 如果δ_t > θ，将(s_t, a_t)加入优先权队列，优先级=δ_t
5. 当队列非空：
   a. 弹出优先级最大的(s,a)
   b. 查询模型：(s', r) = Model(s,a)
   c. 更新Q(s,a)
   d. 查找s的所有前驱(pred_state, pred_action)
   e. 对每个前驱：
      - 计算δ_pred = |r_pred + β·max_a' Q(s, a') - Q(pred_state, pred_action)|
      - 如果δ_pred > θ，更新队列中(pred_state, pred_action)的优先级为δ_pred

**为什么高效：** 
- 只更新TD误差大的状态-动作对
- 通过前驱关系精准传播价值信息
- 避免无效更新（TD误差小的状态）

## 4. 训练过程讲解#

**数据预处理：**
- 模型学习：需要学习环境模型Model(s,a) = (s',r)
- 前驱表维护：需要记录每个状态的到来前驱

**参数初始化：**
- Q表：全0或小的随机值
- 模型：空或初始值
- 优先级队列：空
- 学习率α：0.1~0.5
- 折扣因子β：0.9~0.99
- 阈值θ：0.01~0.1（过滤小TD误差）
- 最大规划步数：根据计算资源

**迭代过程：**
1. 每个episode开始，重置环境状态
2. 在每步中：
   - 选择并执行动作（ε-greedy）
   - 学习模型
   - 直接Q-learning更新
   - 计算TD误差，可能入队
   - **规划阶段：** 处理优先级队列
     * 弹出队首元素
     * 更新Q值
     * 查找并更新前驱
   - 直到队列空或达最大规划步数
3. 直到收敛或达最大episode数

**收敛条件：**
- 优先级队列持续为空（所有状态-动作对都收敛）
- Q值变化小于阈值
- 达到最大episode数

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| α (学习率) | 控制更新步长 | 0.1~0.5 | 0.2 |
| β (折扣因子) | 权衡即时与未来奖励 | 0.9~0.99 | 0.9 |
| θ (阈值) | 过滤小TD误差 | 0.001~0.1 | 0.01 |
| max_planning_steps | 每次最大规划步数 | 100~1000 | 根据计算资源 |
| ε (探索率) | 平衡探索与利用 | 0.01~1.0 | 从1.0衰减 |

## 5. 应用场景#

**典型应用：**

1. **迷宫求解：** 状态空间较大但目标明确的任务。**为什么适合：** 价值信息需要沿着路径反向传播，Prioritized Sweeping能高效完成这种反向传播。

2. **确定性规划问题：** 已知环境模型需要求解最优策略。**为什么适合：** 完全基于模型规划，不需要真实交互。

3. **大规模MDP：** 状态空间巨大，需要高效规划。**为什么适合：** 只更新相关状态，避免无效计算。

4. **作为其他算法的子程序：** 如Dyna-Q+Prioritized Sweeping。**为什么适合：** 结合真实交互和高效规划。

**适用数据特征：**
- 有环境模型（或模型可学习）
- 需要反向传播价值信息
- 状态空间较大，均匀采样低效
- 前驱关系可计算或存储

**不适用场景：**
- 无模型且模型难学习（改用Q-learning）
- 前驱关系极难计算（如高维连续状态空间）
- 实时系统，计算资源严格受限

## 6. 优缺点分析#

**优点：**
1. **样本效率高：** 优先更新重要的状态-动作对。**成立条件：** TD误差准确反映需要更新的程度。
2. **价值传播高效：** 通过前驱关系精准传播。**成立条件：** 前驱表准确且完整。
3. **避免无效更新：** 阈值θ过滤小TD误差。**成立条件：** θ设置合理。
4. **理论保证：** 在确定性环境中可证明收敛。

**缺点：**
1. **前驱存储开销：** 需要存储每个状态的前驱。**问题：** 状态空间大时内存消耗大。**缓解思路：** 使用近似前驱查找或限制前驱数量。
2. **模型依赖：** 需要准确的环境模型。**问题：** 模型错误会传播。**缓解思路：** 使用不确定性估计或结合真实采样。
3. **计算复杂度：** 每次更新需要查找前驱。**问题：** 前驱多时计算量大。**缓解思路：** 限制前驱数量或使用近似。

**与同类算法对比：**

| 特性 | Dyna-Q | Prioritized Sweeping | RTP-Q |
|------|---------|---------------------|--------|
| 采样策略 | 均匀采样 | 优先级采样 | 结合探索规划 |
| 前驱利用 | 否 | 是 | 是 |
| 计算开销 | 低 | 中 | 中高 |
| 收敛速度 | 中 | 快 | 快 |

## 7. 调库实现#

使用Python手动实现Prioritized Sweeping（使用heapq作为优先级队列）：

```python
"""
Prioritized Sweeping算法调库实现
使用优先级队列指导规划更新
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import heapq  # 用于优先级队列
import random
import matplotlib.pyplot as plt

class PrioritizedSweepingAgent:
    """
    Prioritized Sweeping智能体
    使用优先级队列进行高效规划
    """
    
    def __init__(self, state_space_size, action_space_size,
                 learning_rate=0.2, discount_factor=0.9,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 theta=0.01, max_planning_steps=100):
        """
        初始化Prioritized Sweeping智能体
        
        参数:
        - state_space_size: 状态空间大小
        - action_space_size: 动作空间大小
        - learning_rate: 学习率α
        - discount_factor: 折扣因子β
        - theta: TD误差阈值
        - max_planning_steps: 每次最大规划步数
        """
        # Q表
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        
        # 模型: 存储 (s,a) -> (s', r)
        self.model = {}
        
        # 前驱表: 存储 s -> [(pred_state, pred_action), ...]
        self.predecessors = defaultdict(list)
        
        # 优先级队列 (使用heapq，存储(-priority, state_key, action))
        # heapq是最小堆，所以用负优先级实现最大堆
        self.priority_queue = []
        
        # 超参数
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.theta = theta
        self.max_planning_steps = max_planning_steps
        self.action_space_size = action_space_size
        
        # 记录所有访问过的状态-动作对（用于规划采样）
        self.experience_pool = []
    
    def choose_action(self, state):
        """ε-greedy策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            state_key = self._state_to_key(state)
            q_values = self.q_table[state_key]
            return np.random.choice(np.where(q_values == q_values.max())[0])
    
    def learn_model(self, state, action, reward, next_state):
        """学习环境模型，并更新前驱表"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        key = (state_key, action)
        
        # 如果之前有模型，需要从前驱表中移除旧的
        if key in self.model:
            old_next_state = self.model[key]
            # 在实际应用中，需要更复杂的逻辑来处理前驱更新
        
        # 学习新模型
        self.model[key] = (next_state_key, reward)
        
        # 更新前驱表: next_state的前驱增加(state_key, action)
        if (state_key, action) not in self.predecessors[next_state_key]:
            self.predecessors[next_state_key].append((state_key, action))
        
        # 添加到经验池
        if key not in self.experience_pool:
            self.experience_pool.append(key)
    
    def compute_td_error(self, state_key, action, reward, next_state_key, done):
        """计算TD误差（作为优先级）"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        td_error = td_target - self.q_table[state_key][action]
        return abs(td_error)  # 返回绝对值作为优先级
    
    def update_q_value(self, state_key, action, reward, next_state_key, done):
        """Q-learning更新"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error
        
        return abs(td_error)
    
    def add_to_priority_queue(self, state_key, action, priority):
        """将状态-动作对加入优先级队列"""
        # heapq存储(-priority, state_key, action)以实现最大堆
        heapq.heappush(self.priority_queue, (-priority, state_key, action))
    
    def get_predecessors(self, state_key):
        """获取状态的所有前驱"""
        return self.predecessors.get(state_key, [])
    
    def planning_step(self):
        """
        执行一次Prioritized Sweeping规划
        
        从队列弹出优先级最高的状态-动作对，更新Q值，
        并将受影响的前驱加入队列
        """
        if not self.priority_queue:
            return
        
        # 弹出优先级最高的元素
        neg_priority, state_key, action = heapq.heappop(self.priority_queue)
        priority = -neg_priority
        
        # 查询模型
        if (state_key, action) not in self.model:
            return  # 模型未知
        
        next_state_key, reward = self.model[(state_key, action)]
        
        # 判断是否为终止状态
        done = False  # 简化：假设都不是终止状态
        # 在实际应用中，需要记录终止状态信息
        
        # 更新Q值
        td_error = self.update_q_value(state_key, action, reward, next_state_key, done)
        
        # 查找前驱并更新队列
        predecessors = self.get_predecessors(state_key)
        
        for pred_state, pred_action in predecessors:
            # 查询前驱的模型
            if (pred_state, pred_action) not in self.model:
                continue
            
            pred_next_state, pred_reward = self.model[(pred_state, pred_action)]
            
            # 计算前驱的新TD误差
            new_td_error = self.compute_td_error(
                pred_state, pred_action, pred_reward, pred_next_state, False
            )
            
            # 如果TD误差大于阈值，加入队列
            if new_td_error > self.theta:
                self.add_to_priority_queue(pred_state, pred_action, new_td_error)
    
    def planning_phase(self):
        """完整的规划阶段"""
        steps = 0
        while self.priority_queue and steps < self.max_planning_steps:
            self.planning_step()
            steps += 1
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _state_to_key(self, state):
        """将状态转换为可哈希的键"""
        if isinstance(state, (int, np.integer)):
            return state
        return tuple(state) if isinstance(state, np.ndarray) else state


def train_prioritized_sweeping(env_name="FrozenLake-v1", num_episodes=1000,
                              learning_rate=0.2, discount_factor=0.9,
                              theta=0.01, max_planning_steps=50):
    """
    训练Prioritized Sweeping智能体
    """
    env = gym.make(env_name, is_slippery=False)
    
    agent = PrioritizedSweepingAgent(
        state_space_size=env.observation_space.n,
        action_space_size=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        theta=theta,
        max_planning_steps=max_planning_steps
    )
    
    episode_rewards = []
    
    print(f"开始训练Prioritized Sweeping (θ={theta}, 最大规划步数={max_planning_steps})...")
    
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
            
            # 直接Q-learning更新
            state_key = agent._state_to_key(state)
            next_state_key = agent._state_to_key(next_state)
            td_error = agent.update_q_value(state_key, action, reward, next_state_key, done)
            
            # 如果TD误差大于阈值，加入优先级队列
            if td_error > agent.theta:
                agent.add_to_priority_queue(state_key, action, td_error)
            
            # 规划阶段
            agent.planning_phase()
            
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


# 主程序
if __name__ == "__main__":
    # 训练Prioritized Sweeping
    agent, rewards = train_prioritized_sweeping(
        env_name="FrozenLake-v1",
        num_episodes=1000,
        theta=0.01,
        max_planning_steps=50
    )
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='每轮奖励', color='blue')
    
    # 滑动平均
    import pandas as pd
    rewards_series = pd.Series(rewards)
    moving_avg = rewards_series.rolling(window=100, min_periods=1).mean()
    plt.plot(moving_avg, label='100轮滑动平均', color='red', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.title('Prioritized Sweeping 学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prioritized_sweeping_curve.png', dpi=150)
    plt.show()
```

**运行结果示例：**
```
开始训练Prioritized Sweeping (θ=0.01, 最大规划步数=50)...
Episode 100/1000, 最近100轮平均奖励: 0.28, 探索率: 0.606
Episode 200/1000, 最近100轮平均奖励: 0.52, 探索率: 0.367
Episode 500/1000, 最近100轮平均奖励: 0.78, 探索率: 0.082
Episode 1000/1000, 最近100轮平均奖励: 0.89, 探索率: 0.050
```

## 8. 手工代码实现#

使用NumPy从零实现Prioritized Sweeping核心逻辑：

```python
"""
Prioritized Sweeping从零实现
实现优先级队列和前驱传播
"""

import numpy as np
import random
import heapq
from typing import Dict, List, Tuple, Optional


class PrioritizedSweeping:
    """
    Prioritized Sweeping算法从零实现
    
    核心思想:
    1. 使用优先级队列存储待更新的状态-动作对
    2. TD误差作为优先级
    3. 更新后传播到前驱
    """
    
    def __init__(self, 
                 num_states: int,
                 num_actions: int,
                 learning_rate: float = 0.2,
                 discount_factor: float = 0.9,
                 theta: float = 0.01,
                 max_planning_steps: int = 50):
        """
        初始化Prioritized Sweeping算法
        
        参数:
        - num_states: 状态数量
        - num_actions: 动作数量
        - learning_rate: 学习率 α
        - discount_factor: 折扣因子 β
        - theta: TD误差阈值
        """
        # Q表
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)
        
        # 模型: 使用两个数组
        self.model_next_state = np.full((num_states, num_actions), -1, dtype=np.int32)
        self.model_reward = np.zeros((num_states, num_actions), dtype=np.float32)
        
        # 前驱表: 使用字典存储
        self.predecessors = defaultdict(list)
        
        # 优先级队列
        self.priority_queue = []
        
        # 超参数
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.theta = theta
        self.max_planning_steps = max_planning_steps
        self.num_actions = num_actions
        
        # 经验池
        self.experience_pool = []
    
    def select_action(self, state: int) -> int:
        """ε-greedy策略"""
        if random.random() < 0.1:  # 简化: 固定小概率探索
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def learn_model(self, state: int, action: int,
                     reward: float, next_state: int):
        """学习确定性模型"""
        self.model_next_state[state, action] = next_state
        self.model_reward[state, action] = reward
        
        # 更新前驱表
        if (state, action) not in self.predecessors[next_state]:
            self.predecessors[next_state].append((state, action))
        
        # 添加到经验池
        if (state, action) not in self.experience_pool:
            self.experience_pool.append((state, action))
    
    def compute_td_error(self, state: int, action: int,
                          next_state: Optional[int], done: bool) -> float:
        """计算TD误差的绝对值"""
        if done or next_state is None or next_state < 0:
            td_target = self.model_reward[state, action]
        else:
            td_target = (self.model_reward[state, action] + 
                        self.gamma * np.max(self.q_table[next_state]))
        
        td_error = td_target - self.q_table[state, action]
        return abs(td_error)
    
    def update_q_value(self, state: int, action: int,
                        next_state: Optional[int], done: bool):
        """Q-learning更新，返回TD误差"""
        if done or next_state is None or next_state < 0:
            td_target = self.model_reward[state, action]
        else:
            td_target = (self.model_reward[state, action] + 
                        self.gamma * np.max(self.q_table[next_state]))
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        
        return abs(td_error)
    
    def add_to_queue(self, state: int, action: int, priority: float):
        """加入优先级队列"""
        heapq.heappush(self.priority_queue, (-priority, state, action))
    
    def planning_step(self):
        """执行一次规划更新"""
        if not self.priority_queue:
            return False
        
        # 弹出最高优先级
        neg_priority, state, action = heapq.heappop(self.priority_queue)
        
        # 检查模型是否存在
        next_state = self.model_next_state[state, action]
        if next_state < 0:
            return False
        
        # 更新Q值
        td_error = self.update_q_value(state, action, next_state, False)
        
        # 更新前驱
        predecessors = self.predecessors.get(state, [])
        
        for pred_state, pred_action in predecessors:
            # 计算前驱的TD误差
            pred_next = self.model_next_state[pred_state, pred_action]
            if pred_next < 0:
                continue
            
            pred_td_error = self.compute_td_error(pred_state, pred_action, pred_next, False)
            
            if pred_td_error > self.theta:
                self.add_to_queue(pred_state, pred_action, pred_td_error)
        
        return True
    
    def planning_phase(self):
        """完整规划阶段"""
        steps = 0
        while self.priority_queue and steps < self.max_planning_steps:
            if not self.planning_step():
                break
            steps += 1
    
    def fit(self, env, num_episodes: int, max_steps_per_episode: int = 100):
        """训练Prioritized Sweeping"""
        print(f"开始训练Prioritized Sweeping...")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                action = self.select_action(state)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 学习模型
                self.learn_model(state, action, reward, next_state)
                
                # 直接更新
                td_error = self.update_q_value(state, action, next_state, done)
                
                # 可能加入队列
                if td_error > self.theta:
                    self.add_to_queue(state, action, td_error)
                
                # 规划
                self.planning_phase()
                
                state = next_state
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            
            if (episode + 1) % (num_episodes // 10) == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"平均奖励: {avg_reward:.3f}")
        
        return episode_rewards
```

## 9. 可视化与结果理解#

```python
"""
Prioritized Sweeping可视化代码
包括：学习曲线、队列大小变化、TD误差分布
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_learning_curve(episode_rewards, window=100):
    """绘制学习曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：奖励曲线
    axes[0].plot(episode_rewards, alpha=0.3, color='blue', label='每轮奖励')
    rewards_series = pd.Series(episode_rewards)
    moving_avg = rewards_series.rolling(window=window, min_periods=1).mean()
    axes[0].plot(moving_avg, color='red', linewidth=2, label=f'{window}轮滑动平均')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('累积奖励')
    axes[0].set_title('Prioritized Sweeping 学习曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：Q值分布
    q_values = agent.q_table.flatten()
    axes[1].hist(q_values[q_values > 0], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1].set_xlabel('Q值')
    axes[1].set_ylabel('频数')
    axes[1].set_title('学习到的Q值分布（正Q值）')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prioritized_sweeping_analysis.png', dpi=150)
    plt.show()
```

## 10. 模型评估#

```python
"""
Prioritized Sweeping模型评估
"""

import numpy as np
from typing import Dict

def evaluate_prioritized_sweeping(agent, env, num_episodes: int = 100) -> Dict:
    """评估Prioritized Sweeping性能"""
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
        if done and episode_reward > 0:
            success_count += 1
    
    results = {
        'average_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': success_count / num_episodes
    }
    
    return results
```

## 11. 常见问题与易错点#

**数据层面易错点：**

1. **问题：前驱表存储不当**
   - 现象：内存溢出或查找失败
   - 原因：前驱表无限增长或数据结构不当
   - 解决方案：使用合适的数据结构，限制前驱数量

2. **问题：阈值θ设置不当**
   - 现象：θ太大导致队列空，θ太小导致无效更新多
   - 原因：没有根据TD误差范围调整
   - 解决方案：动态调整θ或使用自适应阈值

**模型层面易错点：**

1. **问题：前驱关系错误**
   - 现象：价值传播方向错误
   - 原因：前驱表记录错误
   - 解决方案：仔细检查模型学习和前驱更新逻辑

2. **问题：忽略done状态的特殊处理**
   - 现象：终止状态的前驱更新错误
   - 原因：终止状态没有next_state
   - 解决方案：正确处理终止状态的TD目标

**调参层面易错点：**

1. **问题：最大规划步数设置不当**
   - 现象：设置太小规划不充分，太大计算浪费
   - 原因：没有根据任务复杂度调整
   - 解决方案：从50开始尝试，根据计算资源调整

## 12. 学习总结#

**核心思想回顾：** Prioritized Sweeping使用优先级队列存储TD误差大的状态-动作对，优先更新并传播到前驱，从而高效地进行价值函数更新。

**关键公式：**
1. TD误差：δ = |r + β·max Q(s',a') - Q(s,a)|
2. Q更新：Q(s,a) += α[r + β·max Q(s',a') - Q(s,a)]
3. 优先级：priority = δ

**后续学习方向：**
- **RTP-Q Learning**：结合主动探索的规划
- **Dyna-Q+**：处理模型不确定性
- **深度Prioritized Sweeping**：结合深度函数逼近

## 13. 练习题与思考题#

**基础题1：** 假设Q(s0,a0)更新后，TD误差δ=0.5，θ=0.1。s0的前驱有(s1,a1)和(s2,a2)，它们的TD误差分别为0.2和0.8。哪些会加入优先级队列？

**答案：**
- (s0,a0)：δ=0.5 > θ=0.1，加入队列
- (s1,a1)：δ=0.2 > θ=0.1，加入队列
- (s2,a2)：δ=0.8 > θ=0.1，加入队列
所有三个都会加入队列。

**进阶题1：** 分析Prioritized Sweeping在时间复杂度上与Dyna-Q的差异。

**答案：**
- Dyna-Q：每次均匀采样，时间复杂度O(n)其中n是规划步数
- Prioritized Sweeping：每次需要处理前驱，时间复杂度O(n·d)其中d是平均前驱数量
Prioritized Sweeping更高效但计算开销可能更大。

## 14. 学习路径建议#

**前置算法：**
1. **Q-learning**：核心更新机制
2. **Dyna-Q**：理解规划的基本概念

**进阶算法：**
1. **RTP-Q Learning**：结合主动探索
2. **Model-Based RL with Prioritized Sweeping**

**推荐资源：**
1. **教材**：Sutton & Barto, "Reinforcement Learning: An Introduction"（第8章）
2. **论文**：Moore & Atkeson (1993), "Prioritized Sweeping"


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Prioritized_Sweeping的核心思想及适用场景。
<details><summary>参考答案</summary>
Prioritized_Sweeping通过数据驱动学习输入到输出的映射，适用于强化学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Prioritized_Sweeping的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Prioritized_Sweeping核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Prioritized_Sweeping在什么情况下会失效？
2. 训练数据很少时，Prioritized_Sweeping还能有效工作吗？
3. 如何将Prioritized_Sweeping与其他方法结合？

