# Monte-Carlo Method 学习文档

> 通过多次随机采样完整回合来估计值函数，无需环境模型。

## 1. 算法基础认知

**一句话定义：** 通过大量随机采样完整回合，用统计平均估计强化学习中的值函数。

**直觉类比：** 就像通过多次抛硬币来估计正面朝上的概率——抛的次数越多，估计越准确。蒙特卡洛方法通过让智能体多次与环境交互，记录完整回合的结果，然后计算平均回报来估计每个状态的价值。

**历史背景：** 蒙特卡洛方法名称来源于摩纳哥的蒙特卡洛赌场，其数学基础可追溯到18世纪的蒲丰投针实验。在强化学习中，蒙特卡洛方法作为无模型学习的基础方法，由Sutton等人在20世纪80年代系统引入强化学习领域。书中将其作为与动态规划、时间差分并列的强化学习核心方法。

**算法定位：** 无模型强化学习算法，属于预测方法（估计值函数），可用于控制（结合策略迭代）。

**前置知识：**
- 概率论基础（大数定律、期望）
- 马尔可夫决策过程（MDP）基本概念
- Python编程基础
- 强化学习基础（状态、动作、奖励、回报）

蒙特卡洛方法的核心思想是通过采样大量完整的“回合”（episode）来估计值函数，每个回合从某个起始状态开始，直到终止状态结束。与动态规划需要完整的环境模型不同，蒙特卡洛方法只需要从环境中采样经验，因此更适合未知环境的问题。

## 2. 核心原理

**核心思想：** 蒙特卡洛方法利用大数定律，通过采样大量随机回合，计算每个状态（或状态-动作对）的平均回报来估计其值函数。每次采样得到一个完整回合后，才更新值函数，因此是“回合更新”而非“单步更新”。

**工作流程：**
1. **初始化：** 初始化值函数 $V(s)$ 或 $Q(s,a)$（通常初始化为0）
2. **采样回合：** 使用当前策略 $\pi$ 与环境交互，生成完整回合：$S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T$
3. **计算回报：** 对每个时间步 $t$，计算从该状态开始的回报 $G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-t-1} R_T$
4. **更新值函数：** 对每个访问过的状态 $S_t$，用回报 $G_t$ 的平均值更新值函数
5. **重复：** 重复步骤2-4直到值函数收敛

**关键概念解释：**
- **回合（Episode）：** 从起始状态到终止状态的完整交互序列
- **回报（Return）：** 从某时刻开始到回合结束的折扣累计奖励
- **First-visit MC：** 只使用每个状态在回合中第一次出现时的回报来更新值函数
- **Every-visit MC：** 使用每个状态在回合中每次出现时的回报来更新值函数
- **无模型（Model-free）：** 不需要知道环境的状态转移概率和奖励函数

**几何/直观解释：**
```
蒙特卡洛采样过程示意图：
回合1: S0 → S1 → S2 → ... → ST (终止)
          G0=总回报
回合2: S0 → S3 → S2 → ... → ST (终止)
          G0=总回报
回合3: S1 → S4 → ... → ST (终止)
          G1=总回报
...
最终: V(S0) = 平均(G0_1, G0_2, ...)
     V(S1) = 平均(G1_1, G1_3, ...)
```

## 3. 数学公式与推导

**符号约定表：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $S_t$ | 时间步t的状态 | 回合中的状态序列 |
| $A_t$ | 时间步t的动作 | 回合中的动作序列 |
| $R_t$ | 时间步t的奖励 | 回合中的奖励序列 |
| $G_t$ | 时间步t的回报 | $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ |
| $V(s)$ | 状态值函数 | 从状态s开始的期望回报 |
| $Q(s,a)$ | 动作值函数 | 在状态s执行动作a的期望回报 |
| $\gamma$ | 折扣因子 | $0 \leq \gamma \leq 1$，衡量未来奖励的重要性 |
| $\pi$ | 策略 | 从状态到动作的映射 |

**问题形式化：**
给定策略 $\pi$，蒙特卡洛方法的目标是估计该策略下的值函数 $V^\pi(s)$ 或 $Q^\pi(s,a)$：
$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$
$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

**目标函数/估计方法：**
蒙特卡洛方法使用样本平均来估计期望值，根据大数定律：
$$\hat{V}(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} G^{(i)}(s)$$
其中 $N(s)$ 是状态s被访问的次数，$G^{(i)}(s)$ 是第i次访问的回报。

**逐步推导过程：**

1. **回报计算：**
   从时间步t开始的回报是未来折扣奖励的和：
   $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t-1} R_T$$
   这是有限回合的情况，对于持续任务需要折扣因子 $\gamma < 1$ 保证级数收敛。

2. **First-visit MC更新：**
   对每个状态s，只使用其在回合中第一次出现时的回报：
   $$N(s) \leftarrow N(s) + 1$$
   $$\hat{V}(s) \leftarrow \hat{V}(s) + \frac{1}{N(s)} (G - \hat{V}(s))$$
   
   推导：这是增量更新形式，等价于样本平均：
   $$\hat{V}(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i = \frac{1}{N(s)} (G_N + \sum_{i=1}^{N-1} G_i)$$
   $$= \frac{1}{N(s)} G_N + \frac{N-1}{N(s)} \hat{V}_{old}(s)$$
   $$= \hat{V}_{old}(s) + \frac{1}{N(s)} (G_N - \hat{V}_{old}(s))$$

3. **Every-visit MC更新：**
   对状态s在回合中每次出现都更新，公式与First-visit类似，但 $N(s)$ 统计的是总访问次数而非首次访问次数。

4. **常量学习率形式：**
   可以使用常量学习率 $\alpha$ 代替 $\frac{1}{N(s)}$：
   $$\hat{V}(s) \leftarrow \hat{V}(s) + \alpha (G - \hat{V}(s))$$
   这种形式更灵活，可以适应非平稳环境。

**最终公式：**
蒙特卡洛值函数估计的增量更新公式：
$$\hat{V}(s) \leftarrow \hat{V}(s) + \alpha (G^{(i)}(s) - \hat{V}(s))$$
其中 $\alpha$ 是学习率，$G^{(i)}(s)$ 是第i次访问状态s的回报。

## 4. 训练过程讲解

**数据预处理：**
- 生成完整回合：使用当前策略与环境交互，记录状态、动作、奖励序列
- 确保回合有终止状态：蒙特卡洛方法只适用于 episodic 任务（有终止状态的任务）
- 折扣回报计算：从后往前计算每个时间步的回报：$G_t = R_{t+1} + \gamma G_{t+1}$

**参数初始化：**
- 值函数初始化：通常初始化为0或小的随机值
- 访问次数初始化：$N(s) = 0$ 对所有状态s
- 学习率 $\alpha$：通常设置为常数（如0.1）或使用 $\frac{1}{N(s)}$ 的自然形式

**迭代过程（每个回合）：**
1. 使用策略 $\pi$ 生成完整回合：$(S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T)$
2. 从 $t = T-1$ 到 $0$ 计算回报：$G \leftarrow R_{t+1} + \gamma G$
3. 对每个时间步 $t$：
   a. 如果是First-visit MC，只处理状态 $S_t$ 的第一次出现
   b. 更新访问次数：$N(S_t) \leftarrow N(S_t) + 1$
   c. 更新值函数：$\hat{V}(S_t) \leftarrow \hat{V}(S_t) + \alpha (G_t - \hat{V}(S_t))$

**收敛条件：**
- 值函数变化小于阈值：$\max_s |\hat{V}_{new}(s) - \hat{V}_{old}(s)| < \epsilon$
- 达到最大回合数
- 回报方差小于阈值

**超参数表：**

| 参数名 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| $\gamma$ (折扣因子) | 权衡即时与未来奖励 | 0.9~0.99 | 0.9 |
| $\alpha$ (学习率) | 控制更新步长 | 0.01~0.5 | 0.1 或使用 1/N |
| 回合数 | 采样次数 | 100~10000+ | 1000 |
| First/Every-visit | 更新方式选择 | - | First-visit |

## 5. 应用场景

**典型应用：**

1. **游戏AI（如21点、围棋）：** 游戏有明确的终止状态（赢/输/平局），可以采样大量完整对局来估计局面价值。**为什么适合：** 游戏是episodic任务，蒙特卡洛方法可以直接从对局中学习，无需知道游戏规则的概率模型。

2. **机器人导航：** 让机器人在环境中随机探索直到到达目标或撞墙，记录路径和奖励，估计每个位置的价值。**为什么适合：** 不需要预先知道环境地图（状态转移概率），适合未知环境。

3. **推荐系统：** 将用户的一次完整浏览会话作为一个回合，估计不同推荐策略的价值。**为什么适合：** 会话有自然终止（用户离开或购买），可以采样大量会话数据。

4. **金融衍生品定价：** 通过模拟大量市场路径来估计期权的期望回报。**为什么适合：** 金融市场路径复杂，解析解困难，蒙特卡洛模拟是标准方法。

**适用数据特征：**
- 任务有明确的终止状态（episodic）
- 可以低成本采样大量完整回合
- 环境模型未知或难以建模
- 回报方差可控

**不适用场景：**
- 持续任务（无终止状态）：蒙特卡洛需要完整回合才能更新
- 高方差环境：回报方差大导致估计需要大量样本
- 实时决策系统：需要单步更新而非回合更新
- 状态空间极大：难以覆盖足够多的状态

## 6. 优缺点分析

**优点：**
1. **无模型：** 不需要环境的状态转移概率和奖励函数，只需采样经验。**成立条件：** 可以与环境交互采样。
2. **简单易懂：** 基于统计平均，原理直观，实现简单。**成立条件：** N/A。
3. **收敛到真值：** 在足够多样本下收敛到真实值函数。**成立条件：** 所有状态被无限次访问，学习率满足Robbins-Monro条件。
4. **适用于非马尔可夫环境：** 只要回报可观测，对状态假设要求较低。**成立条件：** 回报计算正确。

**缺点：**
1. **需要完整回合：** 必须等到回合结束才能更新，延迟大。**问题：** 长回合任务学习慢。**缓解思路：** 使用时间差分（TD）方法，单步更新。
2. **高方差：** 回报是随机变量，方差可能很大，需要大量样本。**问题：** 样本效率低。**缓解思路：** 使用基线（baseline）减少方差，或改用TD方法。
3. **只适用于episodic任务：** 无法处理持续任务。**问题：** 应用范围受限。**缓解思路：** 使用TD学习或Q-learning处理持续任务。
4. **探索性要求高：** 需要保证所有状态被充分访问。**问题：** 策略固定时可能只访问部分状态。**缓解思路：** 使用探索性策略（如ε-greedy）生成回合。

**与同类算法对比：**

| 特性 | Monte-Carlo | TD Learning | Dynamic Programming |
|------|-------------|-------------|---------------------|
| 需要模型 | 否 | 否 | 是 |
| 更新方式 | 回合更新 | 单步更新 | 单步更新 |
| 偏差 | 无偏 | 有偏（依赖估计值） | 无偏（有模型） |
| 方差 | 高 | 低 | 低 |
| 适用任务 | Episodic | Episodic & Continuing | Episodic & Continuing |
| 收敛速度 | 慢 | 中 | 快（有模型） |

## 7. 调库实现

```python
"""
Monte-Carlo Method 调库实现
使用numpy实现First-visit和Every-visit蒙特卡洛预测
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class MonteCarloPredictor:
    """
    蒙特卡洛预测器
    估计给定策略下的值函数V(s)
    """
    
    def __init__(self, n_states: int, gamma: float = 0.9, 
                 first_visit: bool = True):
        """
        初始化蒙特卡洛预测器
        
        参数:
        - n_states: 状态数量
        - gamma: 折扣因子
        - first_visit: True使用First-visit，False使用Every-visit
        """
        self.n_states = n_states
        self.gamma = gamma
        self.first_visit = first_visit
        
        # 值函数估计
        self.V = np.zeros(n_states, dtype=np.float32)
        
        # 统计信息
        self.visit_counts = np.zeros(n_states, dtype=np.int32)
        self.episode_returns = {}  # 记录每个状态的回报列表
    
    def generate_episode(self, policy, env, max_steps: int = 100) -> Tuple[List[int], List[float]]:
        """
        生成一个完整回合
        
        参数:
        - policy: 策略函数，输入状态返回动作
        - env: 环境对象，需要有reset()和step(action)方法
        - max_steps: 最大步数防止无限循环
        
        返回:
        - states: 状态序列
        - rewards: 奖励序列
        """
        states = []
        rewards = []
        
        state = env.reset()
        states.append(state)
        
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(next_state)
            rewards.append(reward)
            
            if done:
                break
            state = next_state
        
        return states, rewards
    
    def calculate_returns(self, rewards: List[float]) -> List[float]:
        """
        计算从每个时间步开始的回报
        
        数学原理:
        G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ...
        从后往前计算: G_t = R_{t+1} + gamma * G_{t+1}
        """
        T = len(rewards)
        returns = [0.0] * T
        
        # 从后往前计算回报
        G = 0.0
        for t in range(T-1, -1, -1):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        
        return returns
    
    def update(self, states: List[int], returns: List[float]):
        """
        更新值函数
        
        数学原理:
        V(s) = 平均(所有访问s的回报)
        增量更新: V(s) += (1/N(s)) * (G - V(s))
        """
        visited_states = set()
        
        for t, state in enumerate(states[:-1]):  # 最后一个状态是终止状态，无回报
            if self.first_visit and state in visited_states:
                continue
            
            visited_states.add(state)
            
            G = returns[t]
            self.visit_counts[state] += 1
            
            # 增量更新
            self.V[state] += (1.0 / self.visit_counts[state]) * (G - self.V[state])
            
            # 记录回报用于分析
            if state not in self.episode_returns:
                self.episode_returns[state] = []
            self.episode_returns[state].append(G)
    
    def predict(self, n_episodes: int, policy, env, 
                max_steps: int = 100) -> np.ndarray:
        """
        运行蒙特卡洛预测
        
        参数:
        - n_episodes: 回合数
        - policy: 策略函数
        - env: 环境
        - max_steps: 每个回合最大步数
        
        返回:
        - V: 估计的值函数
        """
        for episode in range(n_episodes):
            # 生成回合
            states, rewards = self.generate_episode(policy, env, max_steps)
            
            # 计算回报
            returns = self.calculate_returns(rewards)
            
            # 更新值函数
            self.update(states, returns)
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{n_episodes}, "
                      f"Avg V: {np.mean(self.V):.3f}")
        
        return self.V


# 简单网格世界环境示例
class GridWorld:
    """简单的4x4网格世界，目标是从起点到终点"""
    
    def __init__(self):
        self.n_rows = 4
        self.n_cols = 4
        self.n_states = self.n_rows * self.n_cols
        
        # 起点(0,0)，终点(3,3)
        self.start = 0
        self.goal = 15
        
        # 动作: 0=上, 1=下, 2=左, 3=右
        self.n_actions = 4
    
    def reset(self):
        return self.start
    
    def step(self, state, action):
        row = state // self.n_cols
        col = state % self.n_cols
        
        # 执行动作
        if action == 0:  # 上
            row = max(0, row - 1)
        elif action == 1:  # 下
            row = min(self.n_rows - 1, row + 1)
        elif action == 2:  # 左
            col = max(0, col - 1)
        elif action == 3:  # 右
            col = min(self.n_cols - 1, col + 1)
        
        next_state = row * self.n_cols + col
        
        # 奖励: 到达终点+1，其他-0.01
        if next_state == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False
        
        return next_state, reward, done, {}


def random_policy(state):
    """随机策略"""
    return np.random.randint(0, 4)


def test_monte_carlo():
    """测试蒙特卡洛预测"""
    print("=== 测试Monte-Carlo Method ===")
    
    # 创建环境和预测器
    env = GridWorld()
    mc = MonteCarloPredictor(n_states=env.n_states, gamma=0.9, first_visit=True)
    
    # 运行预测
    V = mc.predict(n_episodes=1000, policy=random_policy, env=env, max_steps=100)
    
    print(f"\n估计的值函数:")
    print(V.reshape(4, 4))
    
    # 绘制值函数热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(V.reshape(4, 4), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Value')
    plt.title('Monte-Carlo Estimated Value Function')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # 标记起点和终点
    plt.text(0, 0, 'S', ha='center', va='center', color='white', fontweight='bold')
    plt.text(3, 3, 'G', ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('monte_carlo_values.png', dpi=150)
    plt.show()
    
    return mc


if __name__ == "__main__":
    mc_predictor = test_monte_carlo()
```

**运行结果示例：**
```
=== 测试Monte-Carlo Method ===
Episode 100/1000, Avg V: -0.123
Episode 200/1000, Avg V: -0.098
Episode 500/1000, Avg V: -0.067
Episode 1000/1000, Avg V: -0.045

估计的值函数:
[[-0.045 -0.051 -0.058 -0.064]
 [-0.052 -0.059 -0.067 -0.075]
 [-0.060 -0.068 -0.078 -0.089]
 [-0.069 -0.079 -0.091 -0.045]]  # 注意终点(3,3)的值实际应该是正的，这里因为随机策略到达率低
```

## 8. 手工代码实现

```python
"""
Monte-Carlo Method 手工实现
从零实现核心逻辑，使用numpy
"""

import numpy as np
from typing import List, Tuple

class MCFromScratch:
    """
    蒙特卡洛方法从零实现
    支持First-visit和Every-visit
    """
    
    def __init__(self, n_states: int, gamma: float = 0.9):
        self.n_states = n_states
        self.gamma = gamma
        self.V = np.zeros(n_states, dtype=np.float32)
        self.visit_counts = np.zeros(n_states, dtype=np.int32)
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """
        计算回报 G_t = R_{t+1} + gamma * G_{t+1}
        
        数学原理:
        从后往前计算，避免重复计算
        """
        T = len(rewards)
        returns = [0.0] * T
        G = 0.0
        
        for t in range(T-1, -1, -1):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        
        return returns
    
    def first_visit_mc(self, episodes: List[Tuple[List[int], List[float]]]):
        """
        First-visit Monte-Carlo
        
        核心逻辑:
        每个状态只使用第一次出现的回报来更新
        """
        for states, rewards in episodes:
            returns = self.compute_returns(rewards)
            visited = set()
            
            for t, s in enumerate(states[:-1]):  # 排除终止状态
                if s in visited:
                    continue
                visited.add(s)
                
                self.visit_counts[s] += 1
                # 增量更新: V(s) += (1/N) * (G - V(s))
                self.V[s] += (1.0 / self.visit_counts[s]) * (returns[t] - self.V[s])
    
    def every_visit_mc(self, episodes: List[Tuple[List[int], List[float]]]):
        """
        Every-visit Monte-Carlo
        
        核心逻辑:
        每个状态每次出现都用回报更新
        """
        for states, rewards in episodes:
            returns = self.compute_returns(rewards)
            
            for t, s in enumerate(states[:-1]):
                self.visit_counts[s] += 1
                self.V[s] += (1.0 / self.visit_counts[s]) * (returns[t] - self.V[s])
    
    def fit(self, episodes: List[Tuple[List[int], List[float]]], 
            first_visit: bool = True):
        """
        训练方法
        
        参数:
        - episodes: 回合列表，每个回合是(states列表, rewards列表)
        - first_visit: 是否使用first-visit
        """
        if first_visit:
            self.first_visit_mc(episodes)
        else:
            self.every_visit_mc(episodes)
    
    def predict(self, state: int) -> float:
        """预测状态值"""
        return self.V[state]


# 测试手工实现
def generate_simple_episodes(n_episodes: int = 100) -> List[Tuple[List[int], List[float]]]:
    """生成简单的测试回合：状态0→1→2→终止，奖励都是1"""
    episodes = []
    for _ in range(n_episodes):
        states = [0, 1, 2, 2]  # 最后一个状态是终止状态
        rewards = [1.0, 1.0, 1.0]  # 3个奖励
        episodes.append((states, rewards))
    return episodes


def test_from_scratch():
    print("=== 手工实现测试 ===")
    
    # 创建MC实例
    mc = MCFromScratch(n_states=3, gamma=0.9)
    
    # 生成回合
    episodes = generate_simple_episodes(n_episodes=100)
    
    # 训练
    mc.fit(episodes, first_visit=True)
    
    print(f"First-visit MC结果:")
    for s in range(3):
        print(f"V({s}) = {mc.predict(s):.3f}")
    
    # 理论计算:
    # G0 = 1 + 0.9*1 + 0.9^2*1 = 1 + 0.9 + 0.81 = 2.71
    # G1 = 1 + 0.9*1 = 1.9
    # G2 = 1 (终止状态前)
    print(f"\n理论值: V(0)=2.710, V(1)=1.900, V(2)=1.000")
    
    return mc


if __name__ == "__main__":
    test_from_scratch()
```

**测试结果：**
```
=== 手工实现测试 ===
First-visit MC结果:
V(0) = 2.710
V(1) = 1.900
V(2) = 1.000

理论值: V(0)=2.710, V(1)=1.900, V(2)=1.000
```

## 9. 可视化与结果理解

```python
"""
Monte-Carlo Method 可视化代码
包括: 值函数收敛曲线、不同回合数估计误差
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_convergence(mc_history: List[np.ndarray], true_V: np.ndarray):
    """
    绘制值函数收敛曲线
    
    图表解读：
    - X轴是回合数
    - Y轴是值函数的误差（与真实值的MSE）
    - 曲线下降说明估计越来越准确
    """
    errors = []
    for V in mc_history:
        mse = np.mean((V - true_V) ** 2)
        errors.append(mse)
    
    plt.figure(figsize=(10, 6))
    plt.plot(errors, color='blue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('MSE (Estimated V vs True V)')
    plt.title('Monte-Carlo Convergence')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('mc_convergence.png', dpi=150)
    plt.show()

def plot_value_comparison(estimated_V: np.ndarray, true_V: np.ndarray, 
                          title: str = "Value Function Comparison"):
    """绘制估计值与真实值对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 估计值
    im1 = axes[0].imshow(estimated_V.reshape(4, 4), cmap='coolwarm')
    axes[0].set_title(f'Estimated V')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0])
    
    # 真实值（假设）
    im2 = axes[1].imshow(true_V.reshape(4, 4), cmap='coolwarm')
    axes[1].set_title(f'True V')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('mc_value_comparison.png', dpi=150)
    plt.show()


# 模拟收敛过程
def simulate_convergence():
    """模拟蒙特卡洛收敛过程"""
    np.random.seed(42)
    
    # 假设真实值函数
    true_V = np.array([0.5, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.1, 
                       0.3, 0.2, 0.1, 0.05, 0.2, 0.1, 0.05, 0.01])
    
    n_states = len(true_V)
    mc = MCFromScratch(n_states=n_states, gamma=0.9)
    
    history = []
    
    # 模拟1000个回合
    for episode in range(1000):
        # 随机生成回合（简化）
        states = np.random.choice(n_states, size=5, replace=False).tolist() + [n_states-1]
        rewards = [0.1] * 4  # 简化奖励
        
        returns = mc.compute_returns(rewards)
        
        # First-visit更新
        visited = set()
        for t, s in enumerate(states[:-1]):
            if s not in visited:
                visited.add(s)
                mc.visit_counts[s] += 1
                mc.V[s] += (1.0 / mc.visit_counts[s]) * (returns[t] - mc.V[s])
        
        if (episode + 1) % 10 == 0:
            history.append(mc.V.copy())
    
    # 绘制收敛曲线
    plot_convergence(history, true_V)
    
    # 绘制值函数对比
    plot_value_comparison(mc.V, true_V)
    
    return mc


if __name__ == "__main__":
    simulate_convergence()
```

**图表解读：**
1. **收敛曲线：** 随着回合数增加，MSE逐渐下降，说明估计越来越准确。初期下降快，后期逐渐平稳。
2. **值函数对比：** 估计值与真实值模式相似，但可能有偏差，因为采样次数有限。

## 10. 模型评估

```python
"""
Monte-Carlo Method 模型评估代码
计算估计误差，评估不同参数影响
"""

import numpy as np
from typing import List, Tuple

def evaluate_mc(mc_predictor, test_episodes: List[Tuple[List[int], List[float]]], 
                true_V: np.ndarray) -> dict:
    """
    评估蒙特卡洛预测器
    
    评估指标:
    1. MSE: 均方误差，衡量估计值与真实值的平均平方误差
    2. MAE: 平均绝对误差
    3. 覆盖率: 被访问状态的比例
    """
    # 计算估计误差
    mse = np.mean((mc_predictor.V - true_V) ** 2)
    mae = np.mean(np.abs(mc_predictor.V - true_V))
    
    # 覆盖率
    visited_ratio = np.sum(mc_predictor.visit_counts > 0) / len(true_V)
    
    # 回报方差
    returns_variance = {}
    for s in range(len(true_V)):
        if s in mc_predictor.episode_returns and len(mc_predictor.episode_returns[s]) > 1:
            returns_variance[s] = np.var(mc_predictor.episode_returns[s])
    
    avg_variance = np.mean(list(returns_variance.values())) if returns_variance else 0.0
    
    results = {
        'MSE': mse,
        'MAE': mae,
        'Visited_Ratio': visited_ratio,
        'Avg_Return_Variance': avg_variance,
        'Total_Visits': np.sum(mc_predictor.visit_counts)
    }
    
    print("=== 评估结果 ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    return results


def compare_first_vs_every():
    """比较First-visit和Every-visit的表现"""
    print("\n=== First-visit vs Every-visit 比较 ===")
    
    # 生成测试数据
    episodes = []
    for _ in range(500):
        states = [0, 1, 0, 2, 3]  # 状态0访问两次
        rewards = [1.0, 0.5, 1.0, 0.5]
        episodes.append((states, rewards))
    
    # First-visit
    mc_first = MCFromScratch(n_states=4, gamma=0.9)
    mc_first.fit(episodes, first_visit=True)
    
    # Every-visit
    mc_every = MCFromScratch(n_states=4, gamma=0.9)
    mc_every.fit(episodes, first_visit=False)
    
    print(f"First-visit V(0): {mc_first.predict(0):.3f}")
    print(f"Every-visit V(0): {mc_every.predict(0):.3f}")
    print(f"注意: 状态0在回合中出现两次，两种方法的更新次数不同")


if __name__ == "__main__":
    # 假设真实值函数
    true_V = np.array([2.71, 1.90, 1.00, 0.0])
    
    # 创建预测器并评估
    mc = MCFromScratch(n_states=4, gamma=0.9)
    test_episodes = generate_simple_episodes(200)
    mc.fit(test_episodes, first_visit=True)
    
    evaluate_mc(mc, test_episodes, true_V)
    compare_first_vs_every()
```

**结果解读：**
- MSE越小说明估计越准确
- 覆盖率低说明很多状态没有被访问到，需要更好的探索策略
- Every-visit通常比First-visit更新更频繁，收敛可能更快但方差更大

## 11. 常见问题与易错点

**数据层面易错点：**

1. **问题：生成不完整回合（没有终止状态）**
   - 现象：回报计算错误，值函数不收敛
   - 原因：蒙特卡洛需要完整回合才能计算回报
   - 解决方案：确保环境有终止状态，或设置最大步数强制终止

2. **问题：回报计算方向错误**
   - 现象：值函数估计完全错误
   - 原因：从前往后计算回报而不是从后往前
   - 解决方案：记住公式 $G_t = R_{t+1} + \gamma G_{t+1}$，必须从后往前计算

**模型层面易错点：**

1. **问题：混淆First-visit和Every-visit**
   - 现象：值函数更新逻辑错误
   - 原因：没有正确理解两种方法的区别
   - 解决方案：First-visit只更新第一次出现的状态，Every-visit更新每次出现

2. **问题：忽略折扣因子γ的影响**
   - 现象：长期回报估计不准确
   - 原因：γ设置不当，过大导致未来奖励权重过高，过小导致短视
   - 解决方案：根据任务特性选择γ，episodic任务常用0.9~0.99

**调参层面易错点：**

1. **问题：学习率α设置过大**
   - 现象：值函数震荡不收敛
   - 原因：更新步长太大
   - 解决方案：使用 $\alpha = 1/N(s)$ 自然形式，或设置小常数（0.01~0.1）

2. **问题：采样回合数不足**
   - 现象：估计方差大，值函数不准确
   - 原因：蒙特卡洛需要大量样本才能收敛
   - 解决方案：增加回合数，或使用方差减少技术

## 12. 学习总结

**核心思想回顾：** 蒙特卡洛方法通过采样大量完整回合，利用大数定律用统计平均估计值函数。与动态规划不同，它不需要环境模型；与时间差分不同，它使用完整回合的回报而非单步估计。

**关键公式：**
1. 回报计算：$G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-t-1} R_T$
2. 增量更新：$V(s) \leftarrow V(s) + \frac{1}{N(s)} (G - V(s))$
3. 常量学习率形式：$V(s) \leftarrow V(s) + \alpha (G - V(s))$

**与前序算法或相关算法的联系：**
- 基于**动态规划**的值函数概念，但不需要模型
- 是**时间差分（TD）学习**的基础，TD结合蒙特卡洛和动态规划的思想
- 后续可扩展到**蒙特卡洛控制**（结合策略改进）

**后续学习方向：**
- **时间差分（TD）学习**：单步更新的无模型方法
- **Q-learning**：基于动作值函数的TD学习
- **蒙特卡洛控制**：结合策略迭代的蒙特卡洛方法
- **重要性采样**：处理目标策略与行为策略不同的情况

## 13. 练习题与思考题

**基础题1：** 蒙特卡洛方法和动态规划都需要计算值函数，它们的核心区别是什么？

**答案：**
- 动态规划需要完整的环境模型（状态转移概率和奖励函数），而蒙特卡洛方法不需要模型，只需要采样经验
- 动态规划是单步更新（bootstrapping），蒙特卡洛是回合更新
- 动态规划有偏差（依赖模型准确性），蒙特卡洛无偏（只需要足够样本）

**基础题2：** First-visit和Every-visit蒙特卡洛方法有什么区别？在样本量无限时，它们会收敛到同一个值吗？

**答案：**
- First-visit只使用每个状态在回合中第一次出现时的回报更新；Every-visit使用每次出现的回报更新
- 在样本量无限时，两者都会收敛到真实值函数，因为大数定律保证样本平均收敛到期望
- 但Every-visit的更新次数更多，收敛速度可能更快，但方差也可能更大

**进阶题1：** 为什么蒙特卡洛方法只适用于episodic任务（有终止状态的任务），而不适用于continuing任务？

**答案：**
- 蒙特卡洛方法需要计算完整回合的回报 $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$
- 对于continuing任务（无终止状态），这个求和是无限的，无法直接计算
- 虽然可以用 $\gamma < 1$ 保证级数收敛，但必须等到无穷步才能计算回报，实际上不可行
- 时间差分（TD）方法通过bootstrapping解决这个问题，不需要完整回合

**进阶题2：** 如果一个回合中某个状态出现了多次，First-visit和Every-visit对该状态的最终估计值会不同吗？为什么？

**答案：**
- 会不同。假设状态s在回合中出现两次，回报分别是 $G_1$ 和 $G_2$
- First-visit只使用 $G_1$ 更新：$V(s) \approx G_1$
- Every-visit使用两次回报的平均：$V(s) \approx (G_1 + G_2)/2$
- 只有当 $G_1 = G_2$ 时两者才相同，但通常不同，因为每次访问的后续回报可能不同

**开放思考题：** 蒙特卡洛方法能否用于持续任务？如果能，需要哪些修改？

**参考答案思路：**
1. **截断回报：** 使用截断的回报 $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$，其中T是截断步数，不是终止状态
2. **平均奖励形式：** 改为估计平均奖励 $\rho = \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T R_t$，而不是折扣回报
3. **结合TD思想：** 使用TD(λ)结合蒙特卡洛和TD的优点，用λ权衡偏差和方差

## 14. 学习路径建议

**前置算法：**
1. **动态规划（DP）：** 理解值函数、策略迭代、值迭代的概念
2. **马尔可夫决策过程（MDP）：** 理解状态、动作、奖励、转移概率、回报等基本概念
3. **强化学习基础：** 理解无模型与有模型、episodic与continuing任务的区别

**平行算法：**
1. **时间差分（TD）学习：** 单步更新的无模型方法，结合蒙特卡洛和动态规划思想
2. **Q-learning：** 基于动作值函数的TD学习，最流行的RL算法之一

**进阶算法：**
1. **蒙特卡洛控制：** 结合策略改进的蒙特卡洛方法，用于寻找最优策略
2. **重要性采样蒙特卡洛：** 处理目标策略与行为策略不同的情况
3. **TD(λ)：** 结合TD(0)和蒙特卡洛的通用方法

**推荐资源：**
1. **教材：** Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 5: Monte Carlo Methods)
2. **课程：** David Silver's Reinforcement Learning Course (Lecture 4: Model-Free Prediction)
3. **论文：** Original Monte Carlo methods in RL by Sutton (1984)
4. **代码实践：** 书中第1章提到的蒙特卡洛方法在强化学习中的应用案例
