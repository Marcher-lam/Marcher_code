# TD 学习文档

> 强化学习中的核心算法，通过自举（bootstrapping）进行更新的无模型学习方法

---

## 1. 算法基础认知

### 一句话定义
TD（Temporal Difference）学习是强化学习中的一种无模型方法，通过自举（用当前估计更新当前估计）来学习状态值函数或动作值函数。

### 直觉类比
想象你在学习骑自行车，每次尝试后你根据当前的距离目标（如保持平衡）来调整你的行为。不需要等待一次完整的骑行结束（蒙特卡洛），也不需要一个精确的自行车动力学模型（有模型方法）。你边骑边学，用当前的估计来更新之前对“好”的判断。

### 历史背景
TD学习由Rich Sutton在1988年提出，是无模型强化学习的基石。TD与蒙特卡洛方法和动态规划一起，构成了强化学习的三大方法支柱。Q-learing就是TD学习的一个著名应用。

### 算法定位
- 类型：强化学习 → 无模型学习
- 输出：状态值函数 $V(s)$ 或动作值函数 $Q(s,a)$
- 模型类型：无模型、在线学习

### 前置知识
- 马尔可夫决策过程（MDP）：状态、动作、奖励、转移概率
- 动态规划：策略迭代、值迭代
- 蒙特卡洛方法：通过完整序列更新
- 基本微积分：学习率、梯度概念
- Python基础：NumPy、循环、类

---

## 2. 核心原理

### 2.1 核心思想
TD学习的核心理念是：**通过自举（bootstrapping）进行更新，即用当前对下一个状态的估计来代替完整的回报**。

与蒙特卡洛不同，TD不需要等待一个完整序列结束；与动态规划不同，TD不需要环境模型（转移概率）。TD更新目标：$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

### 2.2 工作流程

1. **初始化**
   - 输入：MDP环境（或交互数据），折扣因子 $\gamma \in [0,1]$，学习率 $\alpha$
   - 初始化值函数：$V(s)$（如全部为0）或 $Q(s,a)$
   - 设置回合数 $N$ 和每回合最大步数 $T$

2. **TD(0) 算法（预测）**
   - 对回合 $n=1$ 到 $N$：
     a. 初始化状态 $S_0$
     b. 对 $t=0$ 到 $T$（直到终止）：
        i. 采取行动 $A_t$（如 $\epsilon$-贪婪策略）
        ii. 观察奖励 $R_{t+1}$ 和下一个状态 $S_{t+1}$
        iii. **TD更新**：
            $$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$
            其中 $R_{t+1} + \gamma V(S_{t+1})$ 是TD目标，$R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 是TD误差。
        iv. 如果 $S_{t+1}$ 是终止状态，则跳出循环

3. **输出结果**
   - 学习到的值函数 $V(s)$ 或 $Q(s,a)$

### 2.3 关键概念解释

- **自举（Bootstrapping）**：使用当前估计值来更新自身，而不是等待完整回报
- **TD误差**：$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$，衡量当前估计的误差
- **TD目标**：$R_{t+1} + \gamma V(S_{t+1})$，是下一个奖励加折扣的下一个状态值
- **资格迹（Eligibility Traces）**：用于TD($\lambda$)等更高级算法
- **TD($\lambda$)**：结合蒙特卡洛和TD，使用$\lambda$参数平衡

### 2.4 几何/直观解释
在状态空间中，TD学习通过局部更新来塑造值函数。每次从状态 $s$ 转移到 $s'$，我们用奖励 $r$ 加折扣的 $V(s')$ 来更新 $V(s)$。可以想象成在状态图上，每个转移都传递一些值信息。

与蒙特卡洛对比：蒙特卡洛需要等到回合结束，用实际回报更新所有访问过的状态；而TD是每一步都更新，使用当前估计的 $V(s')$ 来代替剩余回报。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $S_t$ | 时刻 $t$ 的状态 | 状态索引或向量 |
| $A_t$ | 时刻 $t$ 采取的动作 | 动作索引 |
| $R_t$ | 时刻 $t$ 获得的奖励 | 标量 |
| $V(s)$ | 状态值函数 | 函数：$\mathcal{S} \rightarrow \mathbb{R}$ |
| $Q(s,a)$ | 动作值函数 | 函数：$\mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ |
| $\pi(a|s)$ | 策略（动作概率） | $\pi: \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$ |
| $\gamma$ | 折扣因子 | 标量，$0 \leq \gamma \leq 1$ |
| $\alpha$ | 学习率 | 标量，$0 < \alpha \leq 1$ |
| $\delta_t$ | TD误差 | 标量 |

### 3.2 问题形式化
给定MDP，我们希望学习值函数：
- **状态值函数**：$V^\pi(s) = \mathbb{E}_\pi [G_t | S_t = s]$，其中 $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ 是回报。
- **动作值函数**：$Q^\pi(s,a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a]$

TD学习通过样本数据（交互或离线数据）来估计这些值函数。

### 3.3 目标函数/损失函数
TD学习可以使用**均方TD误差**作为损失函数：

$$J(\theta) = \mathbb{E} \left[ \left( R_{t+1} + \gamma V_\theta(S_{t+1}) - V_\theta(S_t) \right)^2 \right]$$

其中 $\theta$ 是值函数的参数（如线性函数近似时的权重）。

**为什么使用TD误差？**
1. **无偏估计**：在给定策略下，$R_{t+1} + \gamma V(S_{t+1})$ 是 $V(S_t)$ 的无偏估计（如果 $V$ 准确）
2. **自举更新**：不需要完整回报，可以每一步更新
3. **与动态规划连接**：TD(0) 等价于策略评估的实时版本
4. **TD($\lambda$)**：结合多步TD误差，通常效果更好

### 3.4 推导过程

**Step 1: TD(0) 更新公式**

从值函数定义：$V^\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s]$

这意味着 $V^\pi(s)$ 应满足Bellman方程。TD(0) 使用样本估计来更新：

$$\Delta V(s) = \alpha (R_{t+1} + \gamma V(s') - V(s))$$

其中 $s' = S_{t+1}$。

**Step 2: 为什么称为“误差”？**

定义TD误差：
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

那么更新可以写成：$V(S_t) \leftarrow V(S_t) + \alpha \delta_t$

如果 $V$ 准确，那么 $\mathbb{E}[\delta_t] = 0$。当 $V$ 不准时，$\delta_t$ 指示了估计误差的方向。

**Step 3: TD($\lambda$) 推导**

TD($\lambda$) 使用资格迹 $e_t(s)$ 来加权不同时间步的TD误差。

更新规则：
$$e_t(s) = \gamma \lambda e_{t-1}(s) + \mathbb{I}(S_t = s)$$
$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)$$

其中 $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 是TD误差。

$\lambda=0$ 时退化为TD(0)；$\lambda=1$ 时接近蒙特卡洛（使用累积奖励）。

### 3.5 最终解/算法步骤

**TD(0) 算法（预测，评估策略 $\pi$）**：

```
输入：策略 π，折扣因子 γ，学习率 α，回合数 N
输出：状态值函数 V(s)

1. 初始化 V(s) 任意（如全0）
2. 对 n=1 到 N：
   a. 初始化状态 S
   b. 对 t=0 到 T（直到终止）：
      i. 根据 π 采取动作 A = π(S)
      ii. 观察奖励 R 和下一个状态 S'
      iii. TD更新：V(S) ← V(S) + α [R + γ V(S') - V(S)]
      iv. S ← S'
      v. 如果 S 是终止状态，则跳出
3. 返回 V
```

**Sarsa（TD控制）**：

```
输入：折扣因子 γ，学习率 α，探索率 ε，回合数 N
输出：动作值函数 Q(s,a)

1. 初始化 Q(s,a) 任意
2. 对 n=1 到 N：
   a. 初始化状态 S，选择动作 A（如 ε-贪婪）
   b. 对 t=0 到 T（直到终止）：
      i. 观察奖励 R 和下一个状态 S'
      ii. 选择下一个动作 A'（如 ε-贪婪）
      iii. Sarsa更新：Q(S,A) ← Q(S,A) + α [R + γ Q(S',A') - Q(S,A)]
      iv. S ← S', A ← A'
      v. 如果 S 是终止状态，则跳出
3. 返回 Q
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================
# 示例环境：简单网格世界（简化为1D）
# ============================================
# 我们构建一个简单的线性世界：状态0-4，目标是状态4
# 动作：左(0) 或 右(1)
# 奖励：到达状态4得+10，其他转移得0

class SimpleGridworld:
    def __init__(self, n_states=5, goal_state=4):
        self.n_states = n_states
        self.goal_state = goal_state
        self.state = 0  # 当前状态
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        # 执行动作
        if action == 0:  # 左
            next_state = max(0, self.state - 1)
        else:  # 右
            next_state = min(self.n_states - 1, self.state + 1)
        
        # 奖励
        reward = 10.0 if next_state == self.goal_state else 0.0
        
        # 是否终止
        done = (next_state == self.goal_state)
        
        self.state = next_state
        return next_state, reward, done
    
    def render(self):
        print(f"当前状态: {self.state}/{self.n_states-1}")

print("简单网格世界环境:")
env = SimpleGridworld()
print(f"状态数: {env.n_states}")
print(f"目标状态: {env.goal_state}")
```

预处理要点：
1. **环境获取**：TD学习需要与环境交互（在线学习）或离线数据（离线TD）
2. **状态表示**：可以是离散索引或特征向量（函数近似时）
3. **奖励设计**：奖励塑造（reward shaping）可以加速学习
4. **折扣因子**：$\gamma$ 选择影响长期回报的权衡

### 4.2 参数初始化

```python
def initialize_td(n_states, method='td0'):
    """
    初始化TD学习参数
    
    参数:
        n_states: 状态数
        method: 'td0' 或 'sarsa' 或 'q-learning'
    
    返回:
        值函数字典或数组
    """
    if method in ['td0', 'sarsa', 'q-learning']:
        # 初始化值函数为0
        V = np.zeros(n_states)  # 对于TD预测
        return {'V': V, 'method': method}
    else:
        raise ValueError("方法必须是 td0, sarsa, 或 q-learning")
```

初始化建议：
1. **值函数**：通常初始化为0，或乐观初始化（如Q-learning中初始化为很高值）
2. **学习率**：$\alpha \in (0,1]$，通常0.1-0.5
3. **折扣因子**：$\gamma \in [0,1]$，通常0.9-0.99
4. **$\epsilon$-贪婪参数**：$\epsilon$ 从1.0衰减到0.01

### 4.3 迭代过程（TD(0) 预测）

```python
def td0_prediction(env, n_episodes=1000, alpha=0.1, gamma=0.9):
    """
    TD(0) 预测：学习给定策略的值函数 V(s)
    
    参数:
        env: 环境（有 reset() 和 step() 方法）
        n_episodes: 回合数
        alpha: 学习率
        gamma: 折扣因子
    
    返回:
        学习到的 V(s)
    """
    n_states = env.n_states
    V = np.zeros(n_states)  # 初始化值函数
    
    print(f"开始TD(0) 预测，状态数: {n_states}")
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:  # 防止无限循环
            # 使用随机策略（或固定策略）
            action = np.random.randint(0, 2)  # 随机动作
            
            next_state, reward, done = env.step(action)
            
            # TD(0) 更新
            td_target = reward + gamma * V[next_state]
            td_error = td_target - V[state]
            V[state] += alpha * td_error
            
            state = next_state
            steps += 1
        
        if (episode+1) % 200 == 0:
            print(f"Episode {episode+1}/{n_episodes}, V(0)={V[0]:.4f}")
    
    print(f"训练完成！最终V: {V}")
    return V

# 运行TD(0) 预测
V_learned = td0_prediction(env, n_episodes=2000, alpha=0.1, gamma=0.9)

# 绘制学习到的值函数
plt.figure(figsize=(10, 6))
plt.plot(range(env.n_states), V_learned, 'bo-', linewidth=2, markersize=8)
plt.xlabel('状态')
plt.ylabel('值函数 V(s)')
plt.title('TD(0) 学习到的值函数')
plt.grid(True, alpha=0.3)
plt.show()
```

### 4.4 收敛条件

TD学习通常训练固定的回合数 $N$，但可以监控收敛：

```python
def check_td_convergence(V_history, tol=1e-4):
    """
    检查TD学习是否收敛
    """
    if len(V_history) < 2:
        return False
    V_change = np.max(np.abs(V_history[-1] - V_history[-2]))
    return V_change < tol
```

收敛相关要点：
1. **值函数稳定**：$\| V^{(k+1)} - V^{(k)} \| < \epsilon$
2. **TD误差接近0**：平均 $|\delta_t|$ 很小
3. **达到最大回合数**：实际中通常用固定回合数
4. **策略收敛**：对于控制算法，策略应稳定

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| `alpha` (学习率) | 控制更新步长 | 0.01 ~ 0.5 | 0.1 |
| `gamma` (折扣因子) | 权衡当前与未来奖励 | 0.9 ~ 0.999 | 0.9 |
| `epsilon` (探索率) | $\epsilon$-贪婪策略的探索率 | 1.0 ~ 0.01 (衰减) | 0.1 |
| `lambda` (TD($\lambda$)) | 混合多步TD误差 | 0.0 ~ 1.0 | 0.9 |
| `n_episodes` | 回合数 | 100 ~ 10000 | 1000 |

选择建议：
1. **学习率**：从0.1开始，如果振荡则减小，如果过慢则增大
2. **折扣因子**：任务周期短用较大$\gamma$，长周期用小$\gamma$
3. **$\epsilon$-贪婪**：通常从1.0衰减到0.01，保证充分探索
4. **TD($\lambda$)**：$\lambda=0$ 是TD(0)，$\lambda=1$ 接近蒙特卡洛

---

## 5. 应用场景

### 5.1 典型应用

**应用1：游戏AI（如TD-Gammon）**
- 场景：学习西洋双陆棋（Backgammon）的玩法
- 为什么适合：TD学习可以直接从交互中学习，不需要模型
- 实现：使用TD($\lambda$) 训练神经网络近似值函数

**应用2：机器人导航**
- 场景：机器人学习在未知环境中导航到目标
- 为什么适合：TD学习可以在线学习，无需环境模型
- 实现：使用Sarsa或Q-learning学习状态-动作值

**应用3：推荐系统（强化学习版）**
- 场景：根据用户反馈（点击/购买）调整推荐策略
- 为什么适合：TD学习可以处理延迟奖励（最终购买）
- 实现：将用户状态、物品作为状态-动作对，使用Q-learning

### 5.2 适用数据特征

1. **序贯决策问题**：问题可以分解为多个时间步决策
2. **延迟奖励**：奖励可能在多步后才获得
3. **无需环境模型**：TD是无模型方法，直接从交互学习
4. **在线学习**：可以与环境实时交互学习
5. **高维状态空间**：结合函数近似（如神经网络）

### 5.3 不适用场景

1. **需要完整模型**：如果环境模型已知，动态规划通常更高效 → 使用值迭代、策略迭代
2. **回合内需要完整回报**：蒙特卡洛可能更好 → 使用蒙特卡洛控制
3. **连续状态和动作**：需要连续控制 → 使用策略梯度（如REINFORCE）
4. **多智能体**：TD需要扩展 → 使用MADDPG等
5. **离线数据且行为策略不同**：需要重要性采样 → 使用Q-learning off-policy

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 在线学习 | 可以与环境交互实时学习 | 环境可交互 |
| 无模型 | 不需要转移概率模型 | 通用 |
| 比蒙特卡洛快 | 每一步都更新，不需要等到回合结束 | 通用 |
| 可以离线学习 | 也可以从固定数据集中学习 | 有离线数据 |
| 与函数近似结合 | 可以处理高维状态空间 | 使用神经网络等 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 对参数敏感 | 学习率、折扣因子等需要调整 | 使用网格搜索、自动调参 |
| 收敛速度 | 可能收敛慢，尤其在高维空间 | 使用更好的函数近似、调整参数 |
| 探索与利用的平衡 | 需要$\epsilon$-贪婪等探索策略 | 使用衰减$\epsilon$、UCB等 |
| 离线数据偏差 | Off-policy数据可能偏差大 | 使用重要性采样、Q-learning的off-policy特性 |
| 收敛性保证 | 线性函数近似下有保证，非线性不一定 | 使用线性近似或谨慎设计 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================
# 1. Q-learning 实现（TD控制）
# ============================================
print("=" * 60)
print("示例1：Q-learning（TD控制）")
print("=" * 60)

class QLearningAgent:
    """
    Q-learning智能体（Off-policy TD控制）
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 初始化Q表
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)  # 探索
        else:
            return np.argmax(self.Q[state])  # 利用
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning更新规则"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def train(self, env, n_episodes=1000):
        """训练Q-learning智能体"""
        scores = []
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 100:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            scores.append(total_reward)
            
            if (episode+1) % 200 == 0:
                avg_score = np.mean(scores[-100:]) if episode >= 100 else np.mean(scores)
                print(f"Episode {episode+1}/{n_episodes}, 平均奖励: {avg_score:.2f}")
        
        return scores

# 训练Q-learning智能体
env = SimpleGridworld()
agent = QLearningAgent(n_states=env.n_states, n_actions=2, alpha=0.1, gamma=0.9, epsilon=0.1)
scores = agent.train(env, n_episodes=2000)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(scores, 'b-', alpha=0.7, label='每回合奖励')
plt.plot(np.convolve(scores, np.ones(100)/100, mode='valid'), 'r-', linewidth=2, label='滑动平均(100回合)')
plt.xlabel('回合')
plt.ylabel('总奖励')
plt.title('Q-learning 学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n学习到的Q表:\n{agent.Q}")
print(f"\n最优策略: {np.argmax(agent.Q, axis=1)}")

# ============================================
# 2. Sarsa 实现（On-Policy TD控制）
# ============================================
print("\n" + "=" * 60)
print("示例2：Sarsa（On-Policy TD控制）")
print("=" * 60)

class SarsaAgent:
    """Sarsa智能体（On-Policy TD控制）"""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """Sarsa更新：使用下一个动作"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q[next_state, next_action]
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def train(self, env, n_episodes=1000):
        scores = []
        
        for episode in range(n_episodes):
            state = env.reset()
            action = self.choose_action(state)
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 100:
                next_state, reward, done = env.step(action)
                next_action = self.choose_action(next_state)
                
                self.update(state, action, reward, next_state, next_action, done)
                
                state = next_state
                action = next_action
                total_reward += reward
                steps += 1
            
            scores.append(total_reward)
            
            if (episode+1) % 200 == 0:
                avg = np.mean(scores[-100:]) if episode >= 100 else np.mean(scores)
                print(f"Episode {episode+1}/{n_episodes}, 平均奖励: {avg:.2f}")
        
        return scores

# 训练Sarsa智能体
agent_sarsa = SarsaAgent(n_states=env.n_states, n_actions=2, alpha=0.1, gamma=0.9, epsilon=0.1)
scores_sarsa = agent_sarsa.train(env, n_episodes=2000)

# 比较Q-learning和Sarsa
plt.figure(figsize=(10, 6))
plt.plot(scores, 'b-', alpha=0.5, label='Q-learning')
plt.plot(scores_sarsa, 'r-', alpha=0.5, label='Sarsa')
plt.xlabel('回合')
plt.ylabel('总奖励')
plt.title('Q-learning vs Sarsa 学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
import matplotlib.pyplot as plt

class TD0Manual:
    """
    手动实现的TD(0) 预测（表格值函数）
    """
    def __init__(self, n_states, alpha=0.1, gamma=0.9):
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros(n_states)  # 值函数
        self.V_history_ = []      # V的历史（用于收敛检查）
    
    def td0_update(self, state, reward, next_state, done):
        """
        TD(0) 更新规则
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.V[next_state]
        
        td_error = td_target - self.V[state]
        self.V[state] += self.alpha * td_error
    
    def train(self, env, policy, n_episodes=1000):
        """
        训练TD(0) 预测
        
        参数:
            env: 环境
            policy: 策略函数，输入状态，返回动作
            n_episodes: 回合数
        """
        V_history = []
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 100:
                action = policy(state)
                next_state, reward, done = env.step(action)
                
                self.td0_update(state, reward, next_state, done)
                
                state = next_state
                steps += 1
            
            V_history.append(self.V.copy())
            
            if (episode+1) % 200 == 0:
                print(f"Episode {episode+1}/{n_episodes}, V(0)={self.V[0]:.4f}")
        
        self.V_history_ = V_history
        print(f"训练完成！")
        return self.V
    
    def plot_learning_curve(self):
        """绘制值函数的学习曲线"""
        V_history = np.array(self.V_history_)
        
        plt.figure(figsize=(10, 6))
        for s in range(self.n_states):
            plt.plot(V_history[:, s], label=f'状态 {s}')
        
        plt.xlabel('回合')
        plt.ylabel('值函数 V(s)')
        plt.title('TD(0) 值函数学习曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ============================================
# 测试手写实现
# ============================================
if __name__ == "__main__":
    # 环境
    env = SimpleGridworld()
    
    # 随机策略
    def random_policy(state):
        return np.random.randint(0, env.n_actions)
    
    # 创建并训练TD(0) 智能体
    td_agent = TD0Manual(n_states=env.n_states, alpha=0.1, gamma=0.9)
    V_learned = td_agent.train(env, random_policy, n_episodes=2000)
    
    # 绘制学习曲线
    td_agent.plot_learning_curve()
    
    # 打印结果
    print(f"\n学习到的值函数: {V_learned}")
    
    # 与理论值比较（如果已知）
    # 对于简单环境，可以计算真实值函数
    print("\nTD(0) 学习完成！")
```

---

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_td_results(env, V, Q=None, title="TD学习结果"):
    """
    可视化TD学习的结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 值函数条形图
    ax = axes[0, 0]
    ax.bar(range(env.n_states), V, color='skyblue', alpha=0.7)
    ax.set_xlabel('状态')
    ax.set_ylabel('值函数 V(s)')
    ax.set_title('学习到的值函数 V(s)')
    ax.grid(True, alpha=0.3)
    
    # 2. 最优策略（如果有Q表）
    if Q is not None:
        optimal_policy = np.argmax(Q, axis=1)
        ax = axes[0, 1]
        ax.bar(range(env.n_states), optimal_policy, color='lightgreen', alpha=0.7)
        ax.set_xlabel('状态')
        ax.set_ylabel('最优动作')
        ax.set_title('最优策略（动作: 0=左, 1=右)')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['左', '右'])
        ax.grid(True, alpha=0.3)
    else:
        ax = axes[0, 1]
        ax.text(0.5, 0.5, '需要Q表\n（TD控制）', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('最优策略（不可用）')
    
    # 3. 值函数收敛曲线（状态0）
    if hasattr(env, 'V_history_') and len(env.V_history_) > 0:
        V_history = np.array(env.V_history_)
        ax = axes[1, 0]
        ax.plot(V_history[:, 0], 'b-', linewidth=2, label='状态 0')
        ax.set_xlabel('回合')
        ax.set_ylabel('V(0)')
        ax.set_title('值函数 V(0) 收敛曲线')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax = axes[1, 0]
        ax.text(0.5, 0.5, '需要保存V历史\n在训练期间', 
                ha='center', va='center', transform=ax.transAxes)
    
    # 4. TD误差分布（模拟）
    # 这里我们模拟TD误差的分布
    np.random.seed(42)
    td_errors = np.random.randn(1000) * 0.1 + 0.01  # 模拟TD误差
    
    ax = axes[1, 1]
    ax.hist(td_errors, bins=50, alpha=0.7, color='orange')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('TD误差')
    ax.set_ylabel('频数')
    ax.set_title('TD误差分布（模拟）')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 运行可视化
print("=" * 60)
print("TD学习可视化")
print("=" * 60)

# 使用之前训练的Q-learning智能体
visualize_td_results(env, V_learned, agent.Q, "TD学习结果（Q-learning）")
```

**结果理解**：
1. **值函数条形图**：显示每个状态的值，通常目标状态值最高
2. **最优策略**：显示每个状态下选择的最优动作（左或右）
3. **收敛曲线**：值函数随时间（回合）稳定到真实值
4. **TD误差分布**：TD误差应集中0附近，表明学习稳定

---

## 10. 模型评估

```python
import numpy as np

def evaluate_td_agent(agent, env, n_episodes=100, render=False):
    """
    评估TD智能体（策略）的性能
    """
    scores = []
    steps_list = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 100:
            # 贪婪策略（无探索）
            action = np.argmax(agent.Q[state]) if hasattr(agent, 'Q') else np.argmax(agent.V)
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            
            if render and episode == 0:
                env.render()
        
        scores.append(total_reward)
        steps_list.append(steps)
    
    avg_score = np.mean(scores)
    avg_steps = np.mean(steps_list)
    
    print("=" * 60)
    print("TD智能体评估报告")
    print("=" * 60)
    print(f"评估回合数: {n_episodes}")
    print(f"平均总奖励: {avg_score:.4f}")
    print(f"平均步数: {avg_steps:.2f}")
    print(f"最高奖励: {np.max(scores)}")
    print(f"最低奖励: {np.min(scores)}")
    
    return avg_score, avg_steps

# 评估示例
# evaluate_td_agent(agent, env, n_episodes=100)
```

**TD学习的特殊评估点**：
1. **平均奖励**：评估学习到的策略质量
2. **步数**：达到目标所需步数，越少越好
3. **值函数误差**：与真实值函数的均方误差（如果已知）
4. **稳定性**：多次运行，检查策略的一致性
5. **探索与利用权衡**：评估时通常关闭探索（贪婪策略）

---

## 11. 常见问题与易错点

### 11.1 智能体学习很慢，收敛困难
**原因**：
- 学习率太小或太大
- 探索不足（$\epsilon$太小）或探索过度（$\epsilon$太大）
- 奖励稀疏，学习信号弱

**解决方案**：
```python
# 1. 调整学习率
agent = QLearningAgent(alpha=0.5)  # 增大学习率

# 2. 使用衰减的epsilon
eps = 1.0
for episode in range(n_episodes):
    eps = max(0.01, eps * 0.995)  # 指数衰减
    agent.epsilon = eps

# 3. 奖励塑造（reward shaping）
# 添加中间奖励引导学习
def shaped_reward(state, next_state, original_reward):
    if next_state == env.goal_state:
        return original_reward
    else:
        # 根据距离目标的远近给与中间奖励
        return original_reward - 0.1 * abs(next_state - env.goal_state)
```

### 11.2 策略震荡，不收敛
**原因**：
- 学习率太大，更新步长过大
- $\epsilon$ 一直很大，没有衰减
- 环境随机性大

**解决方案**：
```python
# 1. 减小学习率
agent = QLearningAgent(alpha=0.01)

# 2. 更激进的epsilon衰减
agent.epsilon = 0.01  # 固定小值，或快速衰减

# 3. 使用经验回放（Experience Replay）
# 存储过往经验，随机回放学习
replay_buffer = []
def store_experience(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))
    if len(replay_buffer) > 10000:
        replay_buffer.pop(0)

def replay_update(batch_size=32):
    # 从buffer中随机采样batch
    batch = np.random.choice(len(replay_buffer), batch_size)
    for idx in batch:
        s, a, r, s_, d = replay_buffer[idx]
        agent.update(s, a, r, s_, d)
```

### 11.3 Off-Policy与On-Policy混淆
**问题**：Q-learning是off-policy，Sarsa是on-policy，更新规则不同。

**解决方案**：
```python
# Q-learning (Off-Policy)：
td_target = r + gamma * max_a Q(s', a)  # 使用下一个状态的最大Q值

# Sarsa (On-Policy]：
next_action = policy(s')  # 使用当前策略选择下一个动作
td_target = r + gamma * Q(s', next_action)  # 使用实际选择的动作
```

### 11.4 高维状态空间，表格方法不适用
**问题**：状态数太大，无法用表格存储。

**解决方案**：
```python
# 使用函数近似（如线性函数、神经网络）
class LinearVFunction:
    def __init__(self, n_features, alpha=0.1):
        self.alpha = alpha
        self.w = np.random.randn(n_features) * 0.01
    
    def features(self, state):
        # 状态到特征向量的映射
        return np.array([1.0, state / 10.0])  # 简单线性特征
    
    def predict(self, state):
        features = self.features(state)
        return np.dot(features, self.w)
    
    def update(self, state, reward, next_state, done, gamma=0.9):
        # TD更新
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * self.predict(next_state)
        
        td_error = td_target - self.predict(state)
        features = self.features(state)
        self.w += self.alpha * td_error * features
```

### 11.5 折扣因子$\gamma$选择困难
**问题**：$\gamma$ 影响未来奖励的权重，选择不当影响学习。

**解决方案**：
1. **任务周期短**：用较大$\gamma$（如0.9-0.99）
2. **任务周期长**：用小$\gamma$（如0.5-0.9）
3. **实验确定**：尝试不同值，观察性能
```python
gammas = [0.5, 0.9, 0.99]
for gamma in gammas:
    agent = QLearningAgent(n_states, n_actions, gamma=gamma)
    # 训练并评估
    # 选择性能最好的gamma
```

---

## 12. 学习总结

### 核心要点回顾：
1. **自举更新**：TD使用当前估计 $V(s')$ 来更新 $V(s)$，无需等待完整回报
2. **TD误差**：$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$，指导学习方向
3. **TD(0) vs TD($\lambda$)**：$\lambda$ 控制多步TD的权衡
4. **On-Policy vs Off-Policy**：Sarsa是on-policy，Q-learning是off-policy
5. **与蒙特卡洛比较**：TD更快（每一步更新），但可能有偏差

### 从TD到其他强化学习：
```
TD(0) 预测
    ↓
Sarsa (On-Policy TD控制)
    ↓
Q-learning (Off-Policy TD控制) - 最著名应用
    ↓
DQN (Deep Q-Network) - 结合深度学习
    ↓
更高级算法 (A3C, PPO, SAC等)
```

### 实践建议：
1. **默认使用**：Q-learning（off-policy，通常更稳定）
2. **调整超参数**：$\alpha=0.1, \gamma=0.9, \epsilon$ 从1.0衰减到0.01
3. **监控收敛**：绘制值函数或TD误差曲线
4. **探索策略**：$\epsilon$-贪婪最常用，可结合衰减
5. **函数近似**：高维状态空间使用线性函数或神经网络

---

## 13. 练习题与思考题（含答案）

### 练习题

**习题1：基础计算**
问题：在简单网格世界中，状态转移：从状态0到状态1，奖励 $r=0$。已知 $V(0)=2.0, V(1)=3.0, \gamma=0.9$。计算TD(0) 更新后 $V(0)$ 的值（学习率 $\alpha=0.1$）。

<details>
<summary>答案</summary>

TD(0) 更新公式：
$$V(s) \leftarrow V(s) + \alpha [R + \gamma V(s') - V(s)]$$

代入：
$TD误差 = r + \gamma V(s') - V(s) = 0 + 0.9 \times 3.0 - 2.0 = 2.7 - 2.0 = 0.7$

更新：
$V(0)_{new} = 2.0 + 0.1 \times 0.7 = 2.0 + 0.07 = 2.07$

因此，更新后 $V(0) = 2.07$。
</details>

**习题2：编程实践**
问题：使用TD(0) 预测在简单网格世界上学习值函数，并绘制学习曲线。

<details>
<summary>答案</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

# 环境
env = SimpleGridworld()

# TD(0) 预测
V = td0_prediction(env, n_episodes=2000, alpha=0.1, gamma=0.9)

# 绘制值函数
plt.bar(range(env.n_states), V, color='skyblue')
plt.xlabel('状态')
plt.ylabel('值函数 V(s)')
plt.title('TD(0) 学习到的值函数')
plt.grid(True, alpha=0.3)
plt.show()

# 绘制学习曲线（需要修改td0_prediction保存历史）
print(f"最终值函数: {V}")
```
</details>

**习题3：理论推导**
问题：证明在给定策略 $\pi$ 下，$R_{t+1} + \gamma V^\pi(S_{t+1})$ 是 $V^\pi(S_t)$ 的无偏估计。

<details>
<summary>答案</summary>

根据Bellman方程：
$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s', r|s,a) (r + \gamma V^\pi(s'))$$

对于给定策略，$\mathbb{E}_\pi [R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s, A_t = a] = \sum_{s', r} P(s', r|s,a) (r + \gamma V^\pi(s')) = V^\pi(s)$

因此，在给定策略和准确值函数下，$R_{t+1} + \gamma V^\pi(S_{t+1})$ 是 $V^\pi(S_t)$ 的无偏估计。

TD(0) 用样本估计这个期望，所以更新是朝着减少TD误差的方向。
</details>

### 思考题

**思考题1**：TD学习与蒙特卡洛方法有什么区别？

<details>
<summary>答案</summary>

| 方面 | TD学习 | 蒙特卡洛方法 |
|------|----------|----------|
| 更新时机 | 每一步都更新 | 需要等到回合结束 |
| 需要模型 | 不需要（无模型） | 不需要（无模型） |
| 偏差与方差 | 有偏差（自举更新）但方差小 | 无偏差但方差大（需要完整回报） |
| 在线学习 | 适合（每一步更新） | 不适合（需要完整序列） |
| TD($\lambda$) | 可以平衡两者 | 蒙特卡洛就是$\lambda=1$的特例 |

核心区别：TD使用自举更新，蒙特卡洛使用完整回报。TD通常方差更小，蒙特卡洛无偏差。
</details>

**思考题2**：Q-learning和Sarsa有什么区别？各适用什么场景？

<details>
<summary>答案</summary>

| 方面 | Q-learning | Sarsa |
|------|----------|----------|
| 策略类型 | Off-Policy | On-Policy |
| 更新目标 | $r + \gamma \max_{a'} Q(s',a')$ | $r + \gamma Q(s', a')$，其中 $a'$ 是按当前策略选择 |
| 探索影响 | 不影响目标（off-policy） | 探索影响目标（on-policy） |
| 收敛性 | 通常收敛到最优Q* | 收敛到当前策略的Q值 |
| 适用场景 | 学习最优策略，不管探索策略 | 学习当前策略的改进，更安全 |

核心区别：Q-learning学习最优策略（off-policy），Sarsa学习当前策略（on-policy）。在探索性强时，Sarsa可能更安全（学到一个考虑探索的策略）。
</details>

---

## 14. 学习路径建议

### 初级阶段（掌握TD基础）
1. 理解MDP和值函数（$V$ 和 $Q$）
2. 掌握TD(0) 更新公式和TD误差
3. 手动计算小样例的TD更新
4. 实现表格TD( 0) 预测

**学习时间**：1-2周**

### 中级阶段（理解TD控制）
1. 理解Sarsa和Q-learning的区别
2. 掌握$\epsilon$-贪婪探索策略
3. 学习TD($\lambda$) 和资格迹
4. 理解Off-Policy与On-Policy的对比

**学习时间**：2-3周**

### 高级阶段（扩展到深度强化学习）
1. 学习DQN（Deep Q-Network）和experience replay
2. 理解策略梯度方法（REINFORCE）
3. 掌握Actor-Critic方法（A2C, PPO）
4. 研究现代TD算法（如TD3, SAC）

**学习时间**：3-4周**

### 实践项目建议
1. **基础项目**：网格世界导航（Q-learning）
2. **进阶项目**：CartPole平衡（使用DQN）
3. **挑战项目**：Atari游戏（使用DQN或A3C）

### 推荐资源
- **书籍**：《Reinforcement Learning: An Introduction》（Sutton & Barto）第6章
- **课程**：David Silver的强化学习课程（YouTube）
- **论文**：Sutton (1988) TD学习原始论文；Watkins (1989) Q-learning论文
- **代码**：OpenAI Gym、Stable-Baselines3
- **实践**：强化学习经典环境（Gridworld, CartPole, Atari）
