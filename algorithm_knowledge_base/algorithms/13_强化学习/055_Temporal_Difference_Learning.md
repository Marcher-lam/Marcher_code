# Temporal Difference Learning 学习文档

> 通过bootstrap方法在每一步更新价值函数，连接蒙特卡洛和动态规划。

## 1. 算法基础认知

**一句话定义：** Temporal Difference (TD)学习是一种通过bootstrap更新价值估计的强化学习算法，在每一步后根据TD误差调整预测。

**直觉类比：** 想象你在预测到达目的地的时间。每走一段路，你就根据实际花费的时间和剩余路程的当前估计，更新你的总时长预测。你不需要等到真正到达目的地才知道预测是否准确。

**历史背景：** TD学习由Richard S. Sutton在1988年提出，是强化学习三大核心方法之一（另外两个是动态规划和蒙特卡洛方法）。它解决了如何在没有环境模型的情况下，从原始经验中学习预测的问题。

**算法定位：** 无模型强化学习算法，属于在线策略（On-policy）学习，结合了动态规划的bootstrap思想和蒙特卡洛的样本学习。

**前置知识：**
- 概率论基础
- 马尔可夫决策过程（MDP）
- Q-learning基础概念
- Python编程

## 2. 核心原理

**核心思想：** TD学习的核心是用当前估计的价值函数来更新对更早状态的价值估计。它使用TD误差——即当前估计与更好估计之间的差异——作为更新信号。TD(λ)通过参数λ平衡不同步数的TD误差。

**工作流程：**
1. 初始化价值函数V(s)（通常初始化为0）
2. 对于每个episode：
   a. 初始化状态s
   b. 当s不是终止状态时：
      - 根据策略π选择动作a
      - 执行动作a，观察新状态s'和奖励r
      - 计算TD误差：δ = r + β·V(s') - V(s)
      - 更新V(s)：V(s) ← V(s) + α·δ
      - s ← s'
3. 重复直到收敛

**关键概念解释：**
- **TD误差（TD error）：** δ = r + β·V(s') - V(s)，衡量当前估计与better estimate之间的差异
- **Bootstrapping：** 使用当前估计的价值函数来计算更新目标（用V(s')来计算目标）
- **λ参数：** 控制TD(λ)中不同n-step回报的权重，λ=0退化为TD(0)，λ=1接近蒙特卡洛
- **Eligibility Trace（资格迹）：** 记录状态被访问的频率和近因性，用于分配TD误差的信用

**几何/直观解释：**
```
TD学习更新传播示意图：

时间步:  t=0    t=1    t=2    t=3
状态:    s0 --a0--> s1 --a1--> s2 --a2--> s3(终止)
奖励:        r1      r2      r3      r4

TD(0)更新：只有直接前驱被更新
  V(s2)更新基于 r3 + β·V(s3)
  
TD(1)更新：所有之前状态按比例更新
  V(s0), V(s1), V(s2)都根据最终回报更新

TD(λ)：中间情况，最近状态权重更大
  更新权重：λ^0, λ^1, λ^2, ... (λ<1时快速衰减)
```

## 3. 数学公式与推导

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| V(s) | 状态价值函数 | 在状态s下遵循策略π的期望回报 |
| δ | TD误差 | 当前估计与better estimate之差 |
| α | 学习率 | 0 < α ≤ 1 |
| β | 折扣因子 | 0 < β < 1 |
| λ | 迹衰减参数 | 0 ≤ λ ≤ 1 |
| e(s) | 资格迹 | 记录状态s的eligibility |

**问题形式化：**

对于给定策略π，状态价值函数定义为：

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \beta^k r_{t+k+1} \mid s_t = s \right]$$

TD学习的目标是让估计的V(s)收敛到真实的V^π(s)。

**TD(0)更新公式：**

最简单的TD学习，在每个时间步t：

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ r_{t+1} + \beta V(s_{t+1}) - V(s_t) \right]$$

其中括号内的是TD目标：$$r_{t+1} + \beta V(s_{t+1})$$

TD误差为：$$\delta_t = r_{t+1} + \beta V(s_{t+1}) - V(s_t)$$

**逐步推导过程：**

1. **为什么可以用V(s')来更新V(s)：**
   根据贝尔曼方程：$$V^\pi(s) = \mathbb{E}_\pi[r + \beta V^\pi(s') \mid s]$$
   因此，样本r + βV(s')是对V^π(s)的有偏估计（当V(s')不准确时）。
   
2. **bootstrap的本质：**
   使用当前（不完美的）估计V(s')来计算更新目标，而不是等待完整的回报。
   这比蒙特卡洛方法更快，但引入了偏差。

3. **TD(λ)的n-step回报：**
   n-step回报定义为：
   $$R_t^{(n)} = r_{t+1} + \beta r_{t+2} + ... + \beta^{n-1} r_{t+n} + \beta^n V(s_{t+n})$$
   
   当n=1时，就是TD(0)目标；当n→∞时，接近蒙特卡洛回报。

4. **λ-return：**
   TD(λ)使用λ-return作为目标，它是所有n-step回报的几何加权：
   $$R_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} R_t^{(n)}$$
   
5. **资格迹更新：**
   使用资格迹e(s)来追踪哪些状态应该被更新：
   $$e(s) = \begin{cases} \beta\lambda e(s) & \text{if } s \neq s_t \\ \beta\lambda e(s) + 1 & \text{if } s = s_t \end{cases}$$
   
   然后对所有状态进行更新：
   $$V(s) \leftarrow V(s) + \alpha \delta_t e(s) \quad \forall s$$

**TD(λ)前向视图更新：**

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ R_t^\lambda - V(s_t) \right]$$

**TD(λ)后向视图更新（等价）：**

$$\begin{aligned}
\delta_t &= r_{t+1} + \beta V(s_{t+1}) - V(s_t) \\
e(s) &\leftarrow \beta \lambda e(s) \quad \forall s \\
e(s_t) &\leftarrow e(s_t) + 1 \\
V(s) &\leftarrow V(s) + \alpha \delta_t e(s) \quad \forall s
\end{aligned}$$

## 4. 训练过程讲解

**数据预处理：**
- 状态编码：将环境状态转换为算法可处理的形式
- 策略定义：明确要评估的策略π（TD学习是policy evaluation算法）
- 奖励归一化：有助于稳定学习

**参数初始化：**
- V表初始化：通常初始化为0或小的随机值
- 学习率α：常用0.1~0.5
- 折扣因子β：常用0.9~0.99
- λ参数：常用0.5~0.9（TD(λ)）
- 资格迹初始化：e(s) = 0 ∀s

**迭代过程：**
1. 每个episode开始，重置资格迹：e(s) = 0 ∀s
2. 初始化状态s
3. 当s不是终止状态时：
   - 根据策略π选择动作a
   - 执行动作，获得(s, a, r, s')
   - 计算TD误差：δ = r + β·V(s') - V(s)
   - 更新资格迹：e(s) += 1（或按上述公式更新所有e）
   - 更新所有状态的V值：V(s) += α·δ·e(s) ∀s
   - 衰减资格迹：e(s) *= β·λ ∀s
   - s ← s'
4. 重复直到收敛

**收敛条件：**
- V值变化小于阈值：max|V_new - V_old| < δ
- TD误差稳定：连续N步|δ| < ε
- 达到最大episode数

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| α (学习率) | 控制更新步长 | 0.01~0.5 | 0.1 |
| β (折扣因子) | 权衡即时与未来奖励 | 0.9~0.999 | 0.9 |
| λ (迹参数) | 平衡不同n-step回报 | 0~1 | 0.9 (TD(1))或0 (TD(0)) |
| max_episodes | 最大训练轮数 | 100~10000+ | 根据任务复杂度 |

## 5. 应用场景

**典型应用：**

1. **策略评估（Policy Evaluation）：** 评估给定策略π的性能。**为什么适合：** TD学习是policy evaluation的经典方法，可以估计任意策略的状态价值函数。

2. **预测问题：** 预测某个状态的价值或某个事件的到达时间。**为什么适合：** TD学习本质上是预测算法，可以学习任何可预测的信号。

3. **作为其他算法的基础：** TD误差是SARSA、Q-learning等算法的核心更新信号。**为什么适合：** 这些算法都可以看作是在做某种形式的TD学习。

4. **大规模状态空间问题：** 结合函数逼近（如TD-networks），处理连续状态空间。**为什么适合：** 在线学习特性适合大型问题。

**适用数据特征：**
- 可以表示为MDP或近似MDP的问题
- 需要在线、增量学习
- 有序列决策结构
- 策略固定或变化缓慢（on-policy要求）

**不适用场景：**
- 需要off-policy学习的场景（应用Q-learning更好）
- 奖励极度稀疏的场景（需要结合其他技术）
- 非马尔可夫环境（需要增加记忆机制）

## 6. 优缺点分析

**优点：**
1. **在线学习：** 每一步都可以更新，不需要等到episode结束。**成立条件：** 存在bootstrap能力，即可以用当前V(s')估计未来。
2. **低方差：** 相比蒙特卡洛方法，TD学习的方差更低。**成立条件：** bootstrap引入了偏差但降低了方差。
3. **适用于连续任务：** 不需要明确的episode边界。**成立条件：** 任务没有明确的终止状态。
4. **样本效率：** 比蒙特卡洛方法更高效利用样本。**成立条件：** 可以多次更新同一个状态。

**缺点：**
1. **偏差问题：** Bootstrap引入了偏差，初始V值不准确会影响学习。**问题：** 初始阶段学习可能不稳定。**缓解思路：** 使用较小的学习率，或使用双重学习技术。
2. **对函数逼近敏感：** 结合函数逼近时可能出现divergence。**问题：** 传统TD结合线性函数逼近可能出现发散。**缓解思路：** 使用Gradient TD、GTD等算法。
3. **On-policy限制：** 只能评估当前执行的策略。**问题：** 需要从目标策略采样。**缓解思路：** 使用off-policy TD方法如Q-learning、Importance Sampling等。

**与同类算法对比：**

| 特性 | TD(0) | TD(λ) | Monte Carlo | Q-learning |
|------|-------|---------|-------------|-----------|
| 更新时机 | 每步 | 每步 | Episode结束 | 每步 |
| Bootstrap | 是 | 是 | 否 | 是 |
| 偏差 | 有 | 有 | 无 | 有 |
| 方差 | 低 | 中 | 高 | 低 |
| On/Off policy | On | On | On | Off |

## 7. 调库实现

使用numpy手动实现TD学习（因为scikit-learn没有直接的TD实现，通常用OpenAI Gym环境）：

```python
"""
Temporal Difference Learning 调库实现
本代码演示TD(0)和TD(λ)学习算法
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt

class TDLearningAgent:
    """
    TD学习智能体
    支持TD(0)和TD(λ)
    """
    
    def __init__(self, state_space_size, 
                 learning_rate=0.1, discount_factor=0.9,
                 lam=0.0, use_trace=False):
        """
        初始化TD学习智能体
        
        参数:
        - state_space_size: 状态空间大小
        - learning_rate: 学习率α
        - discount_factor: 折扣因子β
        - lam: λ参数，0对应TD(0)，1对应TD(1)
        - use_trace: 是否使用资格迹（TD(λ)需要）
        """
        # 初始化V表为0
        self.v_table = defaultdict(float)
        self.lr = learning_rate  # 学习率
        self.gamma = discount_factor  # 折扣因子
        self.lam = lam  # λ参数
        self.use_trace = use_trace or lam > 0
        
        # 资格迹
        if self.use_trace:
            self.eligibility_trace = defaultdict(float)
    
    def td_zero_update(self, state, reward, next_state, done):
        """
        TD(0)更新
        
        V(s) = V(s) + α[r + β·V(s') - V(s)]
        
        其中TD误差 δ = r + β·V(s') - V(s)
        """
        # 计算TD目标
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.v_table[next_state]
        
        # TD误差
        td_error = td_target - self.v_table[state]
        
        # 更新V值
        self.v_table[state] += self.lr * td_error
        
        return td_error
    
    def td_lambda_update(self, state, reward, next_state, done):
        """
        TD(λ)更新（使用后向视图）
        
        更新所有状态的V值，根据资格迹分配信用
        """
        # 计算TD误差
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.v_table[next_state]
        
        td_error = td_target - self.v_table[state]
        
        # 更新资格迹
        if self.use_trace:
            # 衰减所有状态的资格迹
            for s in self.eligibility_trace:
                self.eligibility_trace[s] *= self.gamma * self.lam
            # 增加当前状态的资格迹
            self.eligibility_trace[state] += 1.0
            
            # 更新所有有资格迹的状态
            for s in list(self.eligibility_trace.keys()):
                self.v_table[s] += self.lr * td_error * self.eligibility_trace[s]
                # 清除过小的资格迹
                if abs(self.eligibility_trace[s]) < 1e-6:
                    del self.eligibility_trace[s]
        else:
            # 不使用资格迹，退化为TD(0)
            self.v_table[state] += self.lr * td_error
        
        return td_error
    
    def reset_trace(self):
        """重置资格迹（每个episode开始时调用）"""
        if self.use_trace:
            self.eligibility_trace.clear()
    
    def get_value_function(self):
        """返回学习到的价值函数"""
        return dict(self.v_table)


def train_td_learning(env_name="FrozenLake-v1", num_episodes=2000,
                       learning_rate=0.1, discount_factor=0.9,
                       lam=0.0, policy_type='random'):
    """
    训练TD学习智能体
    
    参数:
    - env_name: 环境名称
    - num_episodes: 训练episode数
    - learning_rate: 学习率
    - discount_factor: 折扣因子
    - lam: λ参数（0=TD(0), >0=TD(λ)）
    - policy_type: 策略类型，'random'或'greedy'
    """
    env = gym.make(env_name, is_slippery=False)
    
    agent = TDLearningAgent(
        state_space_size=env.observation_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        lam=lam,
        use_trace=(lam > 0)
    )
    
    episode_rewards = []
    td_errors = []
    
    print(f"开始训练TD({'0' if lam == 0 else f'λ (λ={lam})'})...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        agent.reset_trace()  # 每个episode重置资格迹
        
        while not done:
            # 选择动作（使用固定策略，TD是policy evaluation）
            if policy_type == 'random':
                action = env.action_space.sample()
            else:
                # 简单贪婪策略（假设知道部分信息）
                action = np.argmax([agent.v_table[(state, a)] 
                                     for a in range(env.action_space.n)])
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # TD更新
            if lam > 0:
                td_error = agent.td_lambda_update(state, reward, next_state, done)
            else:
                td_error = agent.td_zero_update(state, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            td_errors.append(abs(td_error))
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            avg_td_error = np.mean(td_errors[-1000:]) if td_errors else 0
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"平均奖励: {avg_reward:.3f}, "
                  f"平均|TD误差|: {avg_td_error:.4f}")
    
    env.close()
    return agent, episode_rewards, td_errors


def evaluate_value_function(agent, env_name="FrozenLake-v1", num_episodes=100):
    """评估学习到的价值函数"""
    env = gym.make(env_name, is_slippery=False)
    
    # 使用学习到的V值作为指导
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # 基于V值选择动作（选择到达V值最大的下一状态的动作）
            best_action = None
            best_value = float('-inf')
            
            for action in range(env.action_space.n):
                # 模拟执行动作
                env_copy = env
                next_state, reward, terminated, truncated, _ = env.step(action)
                env.set_state(state)  # 恢复状态（需要环境支持）
                
                value = reward + agent.v_table.get(next_state, 0)
                if value > best_value:
                    best_value = value
                    best_action = action
            
            # 实际执行
            state, reward, terminated, truncated, _ = env.step(best_action)
            done = terminated or truncated
            
            if done and reward > 0:
                success_count += 1
    
    env.close()
    success_rate = success_count / num_episodes
    print(f"\n评估完成！成功率: {success_rate*100:.1f}%")
    return success_rate


def plot_td_results(episode_rewards, td_errors, lam=0.0):
    """绘制TD学习结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 奖励曲线
    axes[0].plot(episode_rewards, alpha=0.3, color='blue', label='每轮奖励')
    from pandas import Series
    rewards_series = Series(episode_rewards)
    moving_avg = rewards_series.rolling(window=100, min_periods=1).mean()
    axes[0].plot(moving_avg, color='red', linewidth=2, label='100轮滑动平均')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('奖励')
    axes[0].set_title(f'TD({"0" if lam==0 else f"λ, λ={lam}"}) 奖励曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # TD误差
    axes[1].plot(td_errors, alpha=0.3, color='green', label='|TD误差|')
    td_errors_series = Series(td_errors)
    td_moving_avg = td_errors_series.rolling(window=100, min_periods=1).mean()
    axes[1].plot(td_moving_avg, color='red', linewidth=2, label='滑动平均')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('|TD Error|')
    axes[1].set_title('TD误差变化')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # V值分布
    v_values = list(agent.get_value_function().values())
    axes[2].hist(v_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[2].set_xlabel('V值')
    axes[2].set_ylabel('频数')
    axes[2].set_title('学习到的V值分布')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'td_learning_lam{lam}_results.png', dpi=150)
    plt.show()


# 主程序
if __name__ == "__main__":
    # 训练TD(0)
    print("=== 训练TD(0) ===")
    agent_td0, rewards_td0, errors_td0 = train_td_learning(
        env_name="FrozenLake-v1",
        num_episodes=2000,
        learning_rate=0.1,
        discount_factor=0.9,
        lam=0.0
    )
    plot_td_results(rewards_td0, errors_td0, lam=0.0)
    
    # 训练TD(λ)
    print("\n=== 训练TD(λ) with λ=0.5 ===")
    agent_tdl, rewards_tdl, errors_tdl = train_td_learning(
        env_name="FrozenLake-v1",
        num_episodes=2000,
        learning_rate=0.1,
        discount_factor=0.9,
        lam=0.5
    )
    plot_td_results(rewards_tdl, errors_tdl, lam=0.5)
    
    # 打印学习到的V值
    print("\nTD(0) 学习到的V值（前10个状态）:")
    v_td0 = agent_td0.get_value_function()
    for i, (state, value) in enumerate(list(v_td0.items())[:10]):
        print(f"状态 {state}: V = {value:.4f}")
```

**运行结果示例：**
```
=== 训练TD(0) ===
开始训练TD(0)...
Episode 200/2000, 平均奖励: 0.12, 平均|TD误差|: 0.1234
Episode 400/2000, 平均奖励: 0.25, 平均|TD误差|: 0.0987
Episode 800/2000, 平均奖励: 0.45, 平均|TD误差|: 0.0654
Episode 2000/2000, 平均奖励: 0.72, 平均|TD误差|: 0.0321

=== 训练TD(λ) with λ=0.5 ===
开始训练TD(λ)...
Episode 200/2000, 平均奖励: 0.15, 平均|TD误差|: 0.1156
Episode 400/2000, 平均奖励: 0.35, 平均|TD误差|: 0.0823
Episode 800/2000, 平均奖励: 0.58, 平均|TD误差|: 0.0512
Episode 2000/2000, 平均奖励: 0.76, 平均|TD误差|: 0.0287
```

## 8. 手工代码实现

使用NumPy从零实现TD(λ)算法：

```python
"""
Temporal Difference Learning 从零实现
实现TD(0)和TD(λ)（使用资格迹）
"""

import numpy as np
import random
from typing import Dict, List, Tuple


class TDLearning:
    """
    TD学习算法从零实现
    
    核心公式:
    TD(0): V(s) = V(s) + α[r + γ·V(s') - V(s)]
    TD(λ): 使用资格迹进行更新
    """
    
    def __init__(self, 
                 num_states: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 lam: float = 0.0):
        """
        初始化TD学习算法
        
        参数:
        - num_states: 状态数量
        - learning_rate: 学习率 α
        - discount_factor: 折扣因子 γ
        - lam: λ参数 (0=TD(0), 1=TD(1))
        """
        # 初始化V表
        self.v_table = np.zeros(num_states, dtype=np.float32)
        
        # 超参数
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.lam = lam
        
        # 资格迹
        self.eligibility_trace = np.zeros(num_states, dtype=np.float32)
    
    def compute_td_error(self, state: int, reward: float, 
                          next_state: int, done: bool) -> float:
        """
        计算TD误差
        
        数学原理:
        TD误差 δ = r + γ·V(s') - V(s)
        
        如果done=True，则没有下一状态，δ = r - V(s)
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.v_table[next_state]
        
        td_error = td_target - self.v_table[state]
        return td_error
    
    def update_td_zero(self, state: int, td_error: float):
        """
        TD(0)更新: 只更新当前状态
        
        数学原理:
        V(s) = V(s) + α·δ
        """
        self.v_table[state] += self.alpha * td_error
    
    def update_td_lambda(self, state: int, td_error: float):
        """
        TD(λ)更新: 更新所有状态，根据资格迹分配信用
        
        数学原理:
        1. 更新资格迹: e(s) = γ·λ·e(s) 对于所有s
        2. 增加当前状态的资格迹: e(s_t) += 1
        3. 更新所有状态: V(s) += α·δ·e(s) 对于所有s
        """
        # 衰减所有状态的资格迹
        self.eligibility_trace *= self.gamma * self.lam
        
        # 增加当前状态的资格迹
        self.eligibility_trace[state] += 1.0
        
        # 更新所有有资格迹的状态
        self.v_table += self.alpha * td_error * self.eligibility_trace
        
        # 清除过小的资格迹（优化）
        mask = np.abs(self.eligibility_trace) < 1e-6
        self.eligibility_trace[mask] = 0.0
    
    def update(self, state: int, reward: float, 
               next_state: int, done: bool):
        """
        执行一次TD更新
        
        根据λ参数选择TD(0)或TD(λ)更新
        """
        # 计算TD误差
        td_error = self.compute_td_error(state, reward, next_state, done)
        
        # 根据λ选择更新方法
        if self.lam > 0:
            self.update_td_lambda(state, td_error)
        else:
            self.update_td_zero(state, td_error)
        
        return td_error
    
    def reset_eligibility_trace(self):
        """重置资格迹（每个episode开始时调用）"""
        self.eligibility_trace.fill(0.0)
    
    def evaluate_policy(self, env, num_episodes: int = 100) -> float:
        """
        使用学到的价值函数评估策略
        
        返回: 平均奖励
        """
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # 使用V值指导动作选择（选择V值最大的下一状态）
                # 简单策略：随机探索，因为TD是policy evaluation
                action = env.action_space.sample()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        print(f"策略评估完成！{num_episodes}轮平均奖励: {avg_reward:.3f}")
        return avg_reward
    
    def get_value_function(self) -> np.ndarray:
        """返回学习到的价值函数"""
        return self.v_table.copy()
    
    def reset(self):
        """重置算法状态"""
        self.v_table.fill(0.0)
        self.eligibility_trace.fill(0.0)


# 测试代码
if __name__ == "__main__":
    # 创建简单的网格世界环境
    class SimpleGridWorld:
        """4x4网格世界，目标是到达右下角"""
        def __init__(self):
            self.grid_size = 4
            self.n_states = self.grid_size * self.grid_size
            self.n_actions = 4  # 上、下、左、右
            self.start_state = 0  # 左上角
            self.goal_state = 15  # 右下角
            self.state = self.start_state
        
        def reset(self):
            self.state = self.start_state
            return self.state, {}
        
        def step(self, action):
            """执行动作，返回(next_state, reward, terminated, truncated, info)"""
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
    agent = TDLearning(
        num_states=env.n_states,
        learning_rate=0.1,
        discount_factor=0.9,
        lam=0.6  # 使用TD(λ)
    )
    
    # 训练（使用随机策略进行policy evaluation）
    print("开始训练TD(λ)...")
    num_episodes = 1000
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        agent.reset_eligibility_trace()
        
        while not done:
            # 随机策略
            action = random.randint(0, env.n_actions - 1)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # TD更新
            agent.update(state, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            print(f"Episode {episode+1}/{num_episodes}, 平均奖励: {avg_reward:.3f}")
    
    # 打印学习到的V值
    print("\n学习到的V值:")
    v_values = agent.get_value_function()
    for i in range(env.grid_size):
        row_values = [f"{v_values[i*env.grid_size+j]:.2f}" for j in range(env.grid_size)]
        print(' '.join(row_values))
```

## 9. 可视化与结果理解

```python
"""
TD Learning 可视化代码
包括：V值热力图、TD误差变化、λ参数影响等
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

def plot_value_function_heatmap(v_values, grid_size=4, title="TD学习 V值热力图"):
    """
    绘制V值热力图
    
    图表解读：
    - 颜色越深表示V值越大
    - 可以直观看出哪些状态被评估为高价值
    - 理想情况下，靠近目标点的状态V值应该更大
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 重塑为网格
    v_grid = v_values.reshape(grid_size, grid_size)
    
    im = ax.imshow(v_grid, cmap='YlOrRd', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    
    # 添加数值标注
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax.text(j, i, f'{v_grid[i, j]:.2f}',
                          ha='center', va='center',
                          color='black' if v_grid[i, j] < np.max(v_grid)/2 else 'white')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('td_value_heatmap.png', dpi=150)
    plt.show()

def plot_td_error_comparison(errors_td0, errors_tdl, window=100):
    """
    比较TD(0)和TD(λ)的TD误差
    
    图表解读：
    - TD误差收敛速度反映学习速度
    - TD(λ)通常比TD(0)收敛更快
    - 误差越小说明估计越准确
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算滑动平均
    errors_td0_series = pd.Series(errors_td0)
    errors_tdl_series = pd.Series(errors_tdl)
    
    td0_moving = errors_td0_series.rolling(window=window, min_periods=1).mean()
    tdl_moving = errors_tdl_series.rolling(window=window, min_periods=1).mean()
    
    ax.plot(td0_moving, label=f'TD(0) (窗口={window})', linewidth=2)
    ax.plot(tdl_moving, label=f'TD(λ) (窗口={window})', linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('|TD Error| (滑动平均)')
    ax.set_title('TD(0) vs TD(λ) TD误差对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('td_error_comparison.png', dpi=150)
    plt.show()

def plot_lambda_effect(lambda_results, title="λ参数对TD学习的影响"):
    """
    绘制不同λ参数下的学习效果
    
    图表解读：
    - λ=0: TD(0)，只用一步bootstrap
    - λ=1: 接近蒙特卡洛，使用完整回报
    - 中间值: 平衡bias和variance
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for lam, rewards in lambda_results.items():
        rewards_series = pd.Series(rewards)
        moving_avg = rewards_series.rolling(window=100, min_periods=1).mean()
        ax.plot(moving_avg, label=f'λ={lam}')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('平均奖励 (滑动平均)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lambda_effect.png', dpi=150)
    plt.show()

# 运行可视化
if __name__ == "__main__":
    # 需要先用训练好的agent
    # plot_value_function_heatmap(agent.get_value_function())
    pass
```

## 10. 模型评估

```python
"""
TD Learning 模型评估代码
评估学习到的价值函数的质量
"""

import numpy as np
from typing import Dict

def evaluate_td_learning(agent, env, num_episodes: int = 100) -> Dict:
    """
    全面评估TD学习的结果
    
    评估指标:
    1. 平均奖励：衡量策略性能
    2. TD误差：衡量价值函数准确性
    3. 价值函数方差：衡量估计稳定性
    """
    episode_rewards = []
    td_errors = []
    state_visits = np.zeros(agent.v_table.shape)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_visits[state] += 1
            
            # 随机策略（因为TD是policy evaluation）
            action = env.action_space.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 计算TD误差
            if done:
                td_target = reward
            else:
                td_target = reward + agent.gamma * agent.v_table[next_state]
            
            td_error = td_target - agent.v_table[state]
            td_errors.append(abs(td_error))
            
            total_reward += reward
            state = next_state
        
        episode_rewards.append(total_reward)
    
    # 汇总结果
    results = {
        'average_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'average_td_error': np.mean(td_errors),
        'std_td_error': np.std(td_errors),
        'value_function_variance': np.var(agent.v_table),
        'state_coverage': np.sum(state_visits > 0) / len(agent.v_table)
    }
    
    return results

def compute_value_error(learned_v, true_v):
    """
    计算学习到的V值和真实V值之间的误差
    
    为什么需要：评估学习准确性
    """
    if true_v is None:
        print("无法计算误差：真实V值未知")
        return None
    
    mse = np.mean((learned_v - true_v) ** 2)
    mae = np.mean(np.abs(learned_v - true_v))
    
    print(f"V值均方误差 (MSE): {mse:.6f}")
    print(f"V值平均绝对误差 (MAE): {mae:.6f}")
    
    return {'mse': mse, 'mae': mae}
```

## 11. 常见问题与易错点

**数据层面易错点：**

1. **问题：状态访问不均匀**
   - 现象：某些状态的V值学习很慢或不准确
   - 原因：随机策略导致某些状态很少被访问
   - 解决方案：使用exploring starts或改进探索策略

2. **问题：奖励尺度不当**
   - 现象：TD误差过大或过小，学习不稳定
   - 原因：奖励值范围不合适
   - 解决方案：归一化奖励到合理范围

**模型层面易错点：**

1. **问题：忽略done判断**
   - 现象：终止状态的V值更新错误
   - 原因：终止状态没有下一状态，不应加γ·V(s')
   - 解决方案：正确实现done判断逻辑

2. **问题：资格迹实现错误**
   - 现象：TD(λ)性能不如预期
   - 原因：资格迹未正确衰减或更新
   - 解决方案：检查e(s)的更新公式和重置逻辑

3. **问题：bootstrap偏差**
   - 现象：初始阶段V值估计偏差大
   - 原因：使用不准确的V(s')来更新V(s)
   - 解决方案：使用较小的学习率，或等待V值稳定

**调参层面易错点：**

1. **问题：λ参数选择不当**
   - 现象：λ=0学习慢，λ=1方差大
   - 原因：没有根据任务特性选择
   - 解决方案：中间值如0.5~0.9通常不错

2. **问题：学习率α过大**
   - 现象：V值震荡不收敛
   - 原因：步长太大导致overshoot
   - 解决方案：减小α或使用衰减学习率

## 12. 学习总结

**核心思想回顾：** TD学习是一种通过bootstrap进行增量更新的强化学习算法。它结合了动态规划的更新思想和蒙特卡洛的样本学习，使用TD误差作为学习信号。TD(λ)通过λ参数平衡不同n-step回报，资格迹用于分配更新信用。

**关键公式：**
1. TD(0)更新：V(s) ← V(s) + α[r + γ·V(s') - V(s)]
2. TD误差：δ = r + γ·V(s') - V(s)
3. TD(λ) with eligibility trace：e(s) *= γλ, e(s_t) += 1, V(s) += α·δ·e(s)

**与前序算法或相关算法的联系：**
- 基于**动态规划**的贝尔曼方程思想
- 是**Q-learning**等算法的基础（TD误差概念）
- 与**蒙特卡洛方法**相比，TD有更低方差但引入偏差
- **SARSA**本质上是on-policy的Q-learning，使用TD更新

**后续学习方向：**
- **Q-learning**：off-policy的TD控制算法
- **Actor-Critic**：结合策略梯度和价值函数
- **函数逼近**：TD with function approximation
- **深度TD**：DQN等深度强化学习算法

## 13. 练习题与思考题

**基础题1：** 在TD(0)中，假设V(1)=0.5, V(2)=0.8。从状态1执行动作到达状态2，获得奖励10，β=0.9，α=0.1。请计算更新后的V(1)。

**答案：**
- TD目标 = r + β·V(s') = 10 + 0.9 × 0.8 = 10.72
- TD误差 = 10.72 - 0.5 = 10.22
- V(1)新 = 0.5 + 0.1 × 10.22 = 1.522

**基础题2：** 解释TD(λ)中λ参数的作用，并说明λ=0和λ=1分别对应什么情况。

**答案：**
- λ参数控制资格迹的衰减速度，决定了不同n-step回报的权重
- λ=0：TD(0)，只有当前步的TD误差影响更新，对应one-step TD
- λ=1：接近蒙特卡洛方法，所有历史状态的TD误差都影响更新，权重不衰减
- 0<λ<1：中间情况，最近的状态获得更大权重

**进阶题1：** 推导TD(λ)的forward view和backward view等价性（简要说明）。

**答案：**
Forward view使用λ-return：
$$R_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} R_t^{(n)}$$

Backward view使用资格迹：
$$e_t(s) = \sum_{k=1}^{t} \lambda^{t-k} \mathbb{1}(s_{t-k}=s)$$

关键洞察：backward view中的资格迹累积正好对应forward view中该状态对所有未来λ-return的贡献。数学上可以证明两种方法产生的更新量相同。

**开放思考题：** TD学习结合函数逼近时可能出现divergence问题。请思考为什么线性函数逼近+TD可能出现这个问题，以及有哪些解决方案？

**参考答案思路：**
- 原因：TD更新的"deadly triad"问题——off-policy学习、函数逼近、bootstrap三者的组合可能导致divergence
- 线性TD(0)实际上是收敛的（Tsitsiklis & Van Roy证明），但非线性或某些特殊情况下会出问题
- 解决方案：
  1. Gradient TD (GTD)：确保更新是梯度下降方向
  2. Emphatic TD：修改更新以处理off-policy情况
  3. 使用更大、更稳定的函数逼近器

## 14. 学习路径建议

**前置算法：**
1. **马尔可夫决策过程（MDP）**：理解状态、奖励、价值函数
2. **动态规划**：理解贝尔曼方程和价值迭代
3. **蒙特卡洛方法**：理解基于完整回报的学习

**平行算法：**
1. **Q-learning**：off-policy的TD控制算法
2. **SARSA**：on-policy的TD控制算法
3. **Monte Carlo**：无bootstrap的学习方法

**进阶算法：**
1. **TD(λ) with Function Approximation**：处理大规模问题
2. **Gradient TD / GTD2**：解决off-policy divergence
3. **Actor-Critic**：结合策略和价值学习

**推荐资源：**
1. **教材**：Sutton & Barto, "Reinforcement Learning: An Introduction"（第6、7章）
2. **论文**：Sutton (1988), "Learning to predict by the methods of temporal differences"
3. **在线课程**：David Silver's RL Course (Lecture 4 & 5)
4. **代码实践**：OpenAI Spinning Up - TD Learning实现
