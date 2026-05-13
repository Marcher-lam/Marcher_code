# 时序差分方法(TD) 学习文档

> 核心价值：结合蒙特卡洛采样与动态规划自举的在线学习方法，每步即可更新价值估计。
> 来源线索：本节内容根据原书第4章4.4-4.6节"时序差分方法"相关内容整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：TD方法在每步交互后立即用"即时奖励+下一状态估计"来更新当前状态的价值，兼顾MC的无模型特性和DP的自举效率。

**直觉类比**：MC像一个学期结束才看总成绩的学生；TD像每次小考后立刻更新对总成绩预期的学生——虽然每次小考的预期不完美，但更新频繁，最终反而更快收敛。更具体地说，MC 要等到整个学期结束才能知道自己的真实水平（需要完整回合），而 TD 每次考试后都基于当前信息调整预期（每步更新），虽然单次调整可能不准确，但累积多次后收敛更快。

**历史背景**：TD学习由Arthur Samuel（1959年）在跳棋程序中首次使用，后由Richard Sutton（1988年）系统化。它是强化学习中最核心的学习机制之一。TD(λ) 算法统一了 TD(0) 和 MC 方法，是理解 GAE（广义优势估计）的基础。现代深度强化学习中的 DQN、SAC、TD3 等算法的核心更新规则本质上都是 TD 学习。

**算法定位**：免模型(model-free)预测/控制方法。是Q-Learning、SARSA、DQN等算法的基础。TD 方法处于蒙特卡洛方法（纯采样，无自举）和动态规划（纯自举，需要模型）之间的最佳平衡点——既不需要环境模型（免模型），也不需要等待完整回合（在线学习）。

**前置知识**：MDP、贝尔曼方程、蒙特卡洛方法。理解 TD 方法需要先掌握三个核心概念：(1) 贝尔曼方程建立了值函数的递推关系 $V(s) = \mathbb{E}[r + \gamma V(s')]$；(2) 蒙特卡洛方法用采样回报的均值估计值函数；(3) TD 方法将两者结合——用采样代替期望（来自MC），用自举代替完整回报（来自DP）。

**TD方法的三大变体概述**：TD方法家族包含三个核心成员，它们共享"自举"的基本思想但应用方式不同。(1) TD预测：在固定策略下估计 $V(s)$，是纯粹的值函数估计，不涉及策略优化。(2) SARSA（同策略TD控制）：同时学习Q值和改进策略，更新公式为 $Q(s,a) \leftarrow Q(s,a) + lpha[r + \gamma Q(s',a') - Q(s,a)]$，其中 $a'$ 是当前策略在 $s'$ 下实际选择的动作。SARSA的"保守"特性使其在危险环境中更安全——因为它在更新时考虑了探索的风险。(3) Q-Learning（异策略TD控制）：更新公式为 $Q(s,a) \leftarrow Q(s,a) + lpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$，直接使用最优动作的Q值，不依赖当前策略。Q-Learning更"激进"，追求最优策略但可能走危险路径。这三种方法分别对应不同的应用场景，理解它们的区别是掌握强化学习控制方法的基础。

## 2. 核心原理

### 核心思想

TD的核心创新是**自举(bootstrap)**：用一个估计值来更新另一个估计值。具体来说，TD目标 = 即时奖励 + 对下一状态价值的当前估计。

### 工作流程

1. 在状态$s_t$执行动作$a_t$
2. 观察奖励$r_{t+1}$和下一状态$s_{t+1}$
3. 计算TD目标：$r_{t+1} + \gamma V(s_{t+1})$
4. 计算TD误差：$\delta = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$
5. 更新价值：$V(s_t) \leftarrow V(s_t) + \alpha \delta$
6. 重复

### 关键概念

- **TD目标**：$r_{t+1} + \gamma V(s_{t+1})$，对真实回报的估计
- **TD误差(TD Error)**：$\delta = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$，衡量预测与目标的差距
- **自举(Bootstrap)**：用估计值更新估计值
- **TD(0)**：单步TD，只用一步的实际转移
- **n步TD**：用n步的实际奖励再自举
- **TD(λ)**：通过λ参数在TD(0)和MC之间连续插值

**深入理解**：理解核心原理的关键是把握为什么这样设计而非仅仅怎么实现。每一个设计决策背后都有明确的数学动机或实践经验支撑。建议在学习时多问自己如果不用这个设计会怎样，通过反面思考加深理解。

### n步TD与TD(λ)的详细原理

**n步TD的直觉**：TD(0)只看一步就"猜测"未来，而MC看完整个回合才给出判断。n步TD折中——看n步真实的奖励后再用估计值自举。n越大，偏差越小（更接近真实回报）但方差越大（更多随机因素被累积）。实验表明，n=3~5通常是偏差和方差的最佳折中点。

**TD(λ)的统一框架**：TD(λ)不用选择固定的n，而是通过参数λ在所有n步回报之间做指数加权平均。直觉上，TD(λ)同时考虑了1步、2步、3步...的所有回报，但权重随n指数衰减（$\lambda^{n-1}$）。λ=0时只有1步TD有权重（退化为TD(0)），λ=1时所有步的权重相等（退化为MC）。

**资格迹（Eligibility Traces）**：TD(λ)的高效实现使用资格迹 $E_t(s)$ 来记录每个状态在最近被访问的"新鲜度"。更新公式为：$E_t(s) = \gamma\lambda E_{t-1}(s) + \mathbf{1}(S_t=s)$，然后 $V(s) \leftarrow V(s) + lpha \delta_t E_t(s)$。资格迹使得一个TD误差可以"反向传播"到之前访问过的多个状态，而不需要显式地计算多步回报。这种机制与神经网络中反向传播的梯度流有异曲同工之妙。

## 3. 数学公式与推导

### TD(0)更新公式

$$V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)] \tag{4.5}$$

**推导**：这是一个随机梯度下降步骤。令损失函数$L = \frac{1}{2}(V(s_t) - y_t)^2$，其中$y_t = r_{t+1} + \gamma V(s_{t+1})$是TD目标。梯度为$\nabla L = V(s_t) - y_t$，因此更新为$V \leftarrow V - \alpha(V - y) = V + \alpha(y - V)$。

注意：严格来说这不是真正的梯度下降（因为目标$y_t$中也包含$V$），但实践证明这种"半梯度"方法效果很好。

### 终止状态处理

终止状态没有下一步，$\gamma V(s_{t+1})$无意义：

$$\begin{cases} V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1} - V(s_t)], & \text{终止状态} \\ V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)], & \text{非终止状态} \end{cases} \tag{4.6}$$

### n步TD

TD(0)只向前看一步。n步TD用n步实际奖励后再自举：

$$n=1(\text{TD}): \quad G_t^{(1)} = r_{t+1} + \gamma V(s_{t+1})$$
$$n=2: \quad G_t^{(2)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2})$$
$$n=\infty(\text{MC}): \quad G_t^{\infty} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{T-t-1} r_T$$

当$n \to \infty$时，n步TD退化为蒙特卡洛方法。因此n步TD统一了TD和MC。

### TD(λ)方法

通过参数$\lambda \in [0,1]$在TD(0)和MC之间平滑插值：

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

- $\lambda=0$：TD(0)
- $\lambda=1$：MC

## 4. 训练过程讲解

### 数据预处理
- 无特殊预处理，直接与环境交互。TD 方法不需要预先收集数据集，而是边交互边学习。
- TD 适用于有终止或无终止的环境。对于有终止状态的回合制任务，在终止状态处 TD 目标退化为 $r_{t+1}$（因为 $V(s_{terminal})=0$）；对于持续性任务，TD 可以持续更新，不需要等待回合结束。

### 参数初始化
- **$V(s) = 0$**（或小的随机值）：初始价值函数通常设为全零。初始值的选择会影响收敛速度但不影响最终结果——如果初始值接近真实值，收敛更快。
- **学习率 $\alpha = 0.1$**：控制每步更新的幅度。$\alpha$ 太大（如 0.5）会导致值函数震荡不收敛，$\alpha$ 太小（如 0.001）会导致收敛极慢。通常从 0.1 开始尝试。
- **折扣因子 $\gamma = 0.9$**：决定算法对未来奖励的关注程度。$\gamma$ 接近 1 时算法更关注长期回报，接近 0 时算法更短视。

### 迭代过程详解

TD(0) 预测的每次迭代包含以下关键步骤：

**第一步：状态初始化**。重置环境，获取初始状态 $s$。如果是新回合的开始，不需要额外操作；如果是回合中间，继续从上次的状态开始。

**第二步：动作选择与执行**。根据当前策略选择动作 $a$（在纯预测任务中策略是固定的）。执行动作 $a$ 后，环境返回即时奖励 $r$ 和下一状态 $s'$。这一步是 TD 方法与环境的唯一交互窗口。

**第三步：计算 TD 目标与 TD 误差**。TD 目标是 $r + \gamma V(s')$，它是对"当前状态应该有多少价值"的估计。TD 误差 $\delta = r + \gamma V(s') - V(s)$ 衡量了当前估计与新信息之间的差距。如果 $\delta > 0$，说明状态比预期好，应上调 $V(s)$；如果 $\delta < 0$，说明比预期差，应下调。

**第四步：更新价值函数**。$V(s) \leftarrow V(s) + \alpha \cdot \delta$。更新幅度由学习率 $\alpha$ 控制。然后 $s \leftarrow s'$，回到第二步继续。

### 超参数表

| 名称 | 作用 | 推荐范围 | 默认 | 调参建议 |
|------|------|----------|------|---------|
| $\alpha$ | 学习率 | [0.01, 0.5] | 0.1 | 从 0.1 开始，不收敛则降至 0.05 |
| $\gamma$ | 折扣因子 | [0.9, 0.99] | 0.9 | 短回合 0.99，长回合 0.9~0.95 |
| $n$ (n步TD) | 自举步数 | [1, 10] | 1 | n=1 即 TD(0)，n=3~5 通常是好的折中 |
| $\lambda$ (TD(λ)) | 偏差-方差权衡 | [0, 1] | 0.5 | 0.3~0.7 范围内调试，兼顾效率和准确性 |

## 5. 应用场景

### 1. 在线机器人控制
机器人每走一步就更新价值估计，不需要等回合结束。TD 的在线特性使其成为实时控制系统的理想选择。例如扫地机器人在探索房间时，每移动一步就能更新对"这个位置价值多高"的判断。这使得机器人可以在探索的同时学习，边干边学，大幅提升了效率。

### 2. 游戏 AI 实时训练
在游戏进行中持续学习策略。TD 不需要等游戏结束就能更新，适合实时对战和在线博弈。AlphaGo 的策略评估就使用了 TD 思想——它通过自我对弈的 TD 学习来估计棋盘局面的胜率，每下一步棋就更新对当前局面的评估。DQN（Deep Q-Network）在 Atari 游戏上的突破本质上也是 TD 方法——用神经网络替代 Q 表格，但核心更新规则仍然是 TD 误差。

### 3. 持续性任务（无终止状态）
资源调度、网络路由等持续性任务没有自然的"回合结束"点，MC 方法完全不适用（MC 需要完整回合），但 TD 可以正常工作。例如服务器集群的负载均衡，每分配一个请求就能更新策略，不需要等到"一天结束"才能评估效果。

### 4. 自动驾驶实时决策
车辆在行驶过程中需要实时做出决策（变道、加减速），TD 方法可以在每一步感知后立即更新对当前状态的价值评估，不需要等到整个行程结束。这种"边行驶边学习"的能力对自动驾驶系统至关重要。

### 5. 作为高级算法的基础组件
Q-Learning、SARSA、DQN、Actor-Critic 本质上都是在 TD 更新的基础上发展而来。理解 TD 是掌握整个强化学习算法体系的底层逻辑。DQN 的核心是"用神经网络近似 TD 目标"，PPO 的 Critic 训练也是基于 TD 误差。几乎所有现代深度强化学习算法都包含 TD 学习作为核心组件。

### 不适用场景
- 非马尔可夫环境（TD 假设未来只依赖当前状态，如果历史信息重要则 TD 表现差，可考虑 RNN/LSTM 处理）
- 需要无偏估计的场景（如策略梯度中用作基线，此时 MC 更合适，因为 MC 的回报估计是无偏的）

**应用选择指南**：选择算法时，首先判断动作空间类型（离散用DQN系列，连续用DDPG/TD3/SAC），其次判断样本效率需求（高用异策略方法，低用同策略方法），最后判断稳定性需求（高用PPO/TD3）。

在实际应用中，TD方法的在线更新特性使其成为几乎所有现代深度强化学习算法的核心更新机制。

## 6. 优缺点分析

### 优点
1. **在线学习**：TD 方法最大的优势是每步更新，不需要等待回合结束。这意味着智能体在与环境交互的同时就在学习，非常适合实时控制系统和在线部署场景。相比之下，MC 方法必须等到回合结束才能进行一次更新，在长回合或持续性任务中效率极低。
2. **不完整序列学习**：TD 可从不完整的轨迹片段中学习，即使在交互过程中被打断也能利用已有信息更新值函数。这一特性使 TD 在实际应用中更具鲁棒性。
3. **适用于持续任务**：对于没有终止状态的任务（如服务器负载均衡、网络路由），MC 方法完全不适用（无法获得完整回报），但 TD 可以正常工作，因为它的更新只依赖单步转移。
4. **利用马尔可夫性**：在马尔可夫环境中，TD 能利用状态转移的概率结构加速学习。当环境确实满足马尔可夫性质时，TD 的自举机制实际上是一种有效的信息利用方式。

### 缺点
1. **有偏估计**：自举（用估计值 $V(s')$ 更新另一个估计值 $V(s)$）引入偏差。初始时 $V(s')$ 可能与真实值相差很大，导致 TD 目标 $r + \gamma V(s')$ 系统性地偏离真实回报。偏差会随着训练逐渐减小，但在训练初期可能影响学习质量。
2. **依赖初始值**：$V$ 的初始值不仅影响收敛速度，在极端情况下可能导致收敛到错误的值。虽然理论上 TD(0) 在适当条件下保证收敛，但实践中如果初始值设置不当可能需要更多训练。
3. **函数近似时不稳定**：当 TD 方法与神经网络等函数近似器结合时（如 DQN），可能出现训练不稳定甚至发散的情况。这是因为"致命三要素"同时出现：自举、离策略学习、函数近似。解决这一问题需要引入目标网络、经验回放等工程技巧。

### 偏差-方差权衡

TD 方法在偏差和方差之间提供了一个可调节的光谱。TD(0) 偏差最大但方差最低，MC 无偏但方差最高，n 步 TD 和 TD(λ) 则在两者之间插值。实践中通常选择 $n=3\sim5$ 或 $\lambda=0.5\sim0.7$ 作为折中。

### 与MC/DP对比

| 特性 | DP | MC | TD |
|------|-----|-----|-----|
| 需要模型 | 是 | 否 | 否 |
| 自举 | 是 | 否 | 是 |
| 在线更新 | N/A | 否 | 是 |
| 偏差 | 高 | 无 | 中 |
| 方差 | 低 | 高 | 中 |
| 数据效率 | 最高（全量利用） | 低（只用一次） | 中（逐步利用） |

## 7. 调库实现

```python
"""
TD(0) 预测示例
使用 gymnasium 的 FrozenLake 环境
"""
import gymnasium as gym
import numpy as np
from collections import defaultdict

def td0_prediction(env, policy, num_episodes=10000, alpha=0.1, gamma=0.9):
    """TD(0) 预测状态价值函数
    
    参数:
        env: gymnasium 环境
        policy: 策略函数
        num_episodes: 训练回合数
        alpha: 学习率
        gamma: 折扣因子
    """
    V = defaultdict(float)
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        while True:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # TD目标：r + γ * V(s')（终止状态V=0）
            td_target = reward + (1 - terminated) * gamma * V.get(next_state, 0.0)
            # TD误差
            td_error = td_target - V.get(state, 0.0)
            # 更新
            V[state] = V.get(state, 0.0) + alpha * td_error
            
            state = next_state
            if terminated or truncated:
                break
    
    return V

# 测试
env = gym.make('FrozenLake-v1', is_slippery=True, map_name="4x4")
policy = lambda s: env.action_space.sample()  # 随机策略

print("=== TD(0) 预测 (FrozenLake-v1) ===")
V = td0_prediction(env, policy, num_episodes=50000, alpha=0.05, gamma=0.9)

print("\n状态价值函数 (4x4网格):")
for row in range(4):
    vals = [f"{V.get(row*4+col, 0.0):.4f}" for col in range(4)]
    print(f"  {' '.join(vals)}")

env.close()
```

## 8. 手工代码实现

```python
"""
从零实现 TD(0)、n步TD 和 TD(λ)
"""
import numpy as np
from collections import defaultdict

class TDPrediction:
    """时序差分预测器"""
    
    def __init__(self, gamma=0.9, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.V = defaultdict(float)
    
    def td0_update(self, state, reward, next_state, terminated):
        """TD(0) 更新"""
        td_target = reward + (1 - terminated) * self.gamma * self.V[next_state]
        td_error = td_target - self.V[state]
        self.V[state] += self.alpha * td_error
        return td_error
    
    def n_step_td(self, trajectory, n):
        """n步TD更新
        
        参数:
            trajectory: [(s, r, done), ...] 完整轨迹
            n: 步数
        """
        T = len(trajectory)
        for t in range(T):
            # 计算n步回报
            G = 0
            for i in range(n):
                if t + i < T:
                    G += (self.gamma ** i) * trajectory[t + i][1]
            # 加上自举项
            if t + n < T:
                G += (self.gamma ** n) * self.V[trajectory[t + n][0]]
            # 更新
            td_error = G - self.V[trajectory[t][0]]
            self.V[trajectory[t][0]] += self.alpha * td_error
    
    def train_td0(self, env, policy, n_episodes=1000):
        """TD(0) 训练"""
        for _ in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            while True:
                action = policy(state)
                result = env.step(action)
                next_state, reward = result[0], result[1]
                done = result[2] if len(result) > 3 else result[2]
                self.td0_update(state, reward, next_state, done)
                state = next_state
                if done:
                    break
        return dict(self.V)


# 简单环境测试
class SimpleChain:
    """链式MDP：0→1→2→3(终止)，每步奖励-1"""
    def __init__(self):
        self.n_states = 4
        self.state = 0
    
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        reward = -1
        self.state += 1
        done = (self.state == 3)
        if done:
            reward = 0
        return self.state, reward, done

if __name__ == "__main__":
    env = SimpleChain()
    policy = lambda s: 0  # 固定策略
    
    td = TDPrediction(gamma=0.9, alpha=0.1)
    V = td.train_td0(env, policy, n_episodes=5000)
    
    print("=== TD(0) 预测结果 ===")
    print(f"V(0)={V.get(0,0):.4f}, V(1)={V.get(1,0):.4f}, V(2)={V.get(2,0):.4f}, V(3)={V.get(3,0):.4f}")
    print("真实值: V(0)≈-2.71, V(1)≈-1.9, V(2)≈-1.0, V(3)=0.0")
```

## 9. 可视化与结果理解

```python
"""可视化TD vs MC收敛对比"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class ChainEnv:
    def __init__(self, n_states=5):
        self.n = n_states
        self.state = 0
    def reset(self):
        self.state = 0; return self.state
    def step(self, a):
        self.state += 1
        r = -1 if self.state < self.n - 1 else 0
        return self.state, r, self.state == self.n - 1

def compare_td_mc():
    gamma = 0.9
    n_states = 5
    true_V = np.array([sum([gamma**k * (-1) for k in range(s)]) for s in range(n_states-1)] + [0.0])
    # 修正：从状态s出发，走n_states-1-s步到终点
    true_V = np.zeros(n_states)
    for s in range(n_states):
        steps = n_states - 1 - s
        G = 0
        for k in range(steps):
            G += gamma**k * (-1)
        true_V[s] = G
    
    n_runs = 100
    n_episodes = 200
    
    td_errors = np.zeros(n_episodes)
    mc_errors = np.zeros(n_episodes)
    
    for run in range(n_runs):
        V_td = np.zeros(n_states)
        V_mc = np.zeros(n_states)
        mc_returns = {s: [] for s in range(n_states)}
        
        for ep in range(n_episodes):
            env = ChainEnv(n_states)
            state = env.reset()
            states, rewards = [state], []
            
            while True:
                ns, r, done = env.step(0)
                rewards.append(r)
                # TD更新
                td_target = r + gamma * (1-done) * V_td[ns]
                V_td[state] += 0.1 * (td_target - V_td[state])
                states.append(ns)
                state = ns
                if done: break
            
            # MC更新
            G = 0
            visited = set()
            for t in reversed(range(len(rewards))):
                G = rewards[t] + gamma * G
                s = states[t]
                if s not in visited:
                    mc_returns[s].append(G)
                    V_mc[s] = np.mean(mc_returns[s])
                    visited.add(s)
            
            td_errors[ep] += np.mean(np.abs(V_td - true_V))
            mc_errors[ep] += np.mean(np.abs(V_mc - true_V))
    
    td_errors /= n_runs
    mc_errors /= n_runs
    
    plt.figure(figsize=(8, 5))
    plt.plot(td_errors, label='TD(0)', linewidth=2)
    plt.plot(mc_errors, label='MC (首次访问)', linewidth=2)
    plt.xlabel('训练回合数', fontsize=12)
    plt.ylabel('平均绝对误差', fontsize=12)
    plt.title('TD(0) vs MC 收敛速度对比', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('td_vs_mc.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_td_mc()
```

**结果解读**：TD(0)通常比MC更快收敛（尤其是在线场景下），因为TD每步更新而MC需等回合结束。但MC的最终估计可能更准确（无偏）。

## 10. 模型评估

### 评估指标

| 指标 | 说明 | 为什么适合 TD |
|------|------|--------------|
| 值函数 RMSE | 估计值与真值的均方根误差 | 直接衡量 TD 预测的准确性 |
| 收敛速度 | 达到目标精度所需步数 | TD 的优势就是比 MC 更快收敛 |
| TD 误差分布 | TD 误差 $\delta_t$ 的统计特征 | 误差均值应趋近 0，方差应逐渐减小 |
| 学习曲线稳定性 | 值函数估计随时间的波动 | 反映 TD 学习的稳定性 |

```python
"""TD预测质量评估"""
import numpy as np

def evaluate_td(V_estimated, V_true, n_states):
    error = np.array([abs(V_estimated.get(s, 0) - V_true[s]) for s in range(n_states)])
    print("=== TD预测质量 ===")
    print(f"最大误差: {np.max(error):.6f}")
    print(f"RMSE: {np.sqrt(np.mean(error**2)):.6f}")
    for s in range(n_states):
        print(f"  状态{s}: V={V_estimated.get(s,0):.4f}, 真值={V_true[s]:.4f}")
```

### TD 与 MC 的评估对比
评估 TD 方法时，建议同时运行 MC 方法作为对照。TD 应该比 MC 更快收敛（更少步数达到目标精度），但最终精度可能略低于 MC（因为 TD 有偏差）。在 FrozenLake 等小环境中，可以先用动态规划计算出真值 $V^*$，然后对比 TD 和 MC 的收敛曲线。

### TD评估的完整代码示例

```python
import numpy as np

def comprehensive_td_evaluation(env, policy, true_V, gamma=0.9, 
                                 alpha=0.1, n_episodes=10000, n_repeats=20):
    results = []
    for _ in range(n_repeats):
        V = defaultdict(float)
        for ep in range(n_episodes):
            state, _ = env.reset()
            while True:
                action = policy(state)
                ns, r, terminated, truncated, _ = env.step(action)
                td_target = r + (1 - terminated) * gamma * V.get(ns, 0.0)
                td_error = td_target - V.get(state, 0.0)
                V[state] = V.get(state, 0.0) + alpha * td_error
                state = ns
                if terminated or truncated:
                    break
        est = np.array([V.get(s, 0.0) for s in range(len(true_V))])
        results.append(est)
    results = np.array(results)
    return {
        'mean_rmse': np.sqrt(np.mean((results.mean(axis=0) - true_V)**2)),
        'max_bias': np.max(np.abs(results.mean(axis=0) - true_V)),
        'std_across_runs': np.mean(results.std(axis=0)),
    }
```

TD评估的关键指标：(1) 偏差（bias）——多次运行的平均值与真值之差，反映自举引入的系统性偏差；(2) 跨运行标准差——同一算法多次运行的方差，反映TD方法的稳定性。通常TD的偏差较小但方差也较小，MC偏差为零但方差较大。

## 11. 常见问题与易错点

### 数据层面
1. **终止状态未置零**：$\gamma V(s_{t+1})$ 在终止状态时无意义——终止状态没有"未来"。如果忘记处理终止状态，TD 目标会包含一个无意义的估计值，导致值函数学习结果错误。**解决方案**：使用 `td_target = reward + (1-terminated) * gamma * V[next_state]`，其中 `terminated` 是布尔值，终止时为 1，使自举项归零。
2. **学习率过大**：值函数震荡不收敛。具体表现为训练过程中价值估计忽大忽小，无法稳定下来。在极端情况下，值函数甚至可能发散到无穷大。**解决方案**：$\alpha$ 从 0.1 开始，如果观察到震荡则降到 0.05 或 0.01。对于使用函数近似的场景，建议使用 Adam 优化器配合更小的学习率（如 1e-3）。
3. **状态表示不一致**：在表格型 TD 中，相同的状态必须映射到相同的索引。如果状态表示不一致（例如同一个位置有时用 (1,2) 有时用 (2,1)），则值函数查找会出错。**解决方案**：确保状态到索引的映射是一对一的。

### 模型层面
4. **混淆 TD 预测与 TD 控制**：TD 预测（TD Prediction）是在给定策略下估计 $V(s)$ 或 $Q(s,a)$，策略本身不改变；TD 控制（如 SARSA、Q-Learning）同时估计值函数并优化策略。常见错误是在预测任务中意外改变了策略，或在控制任务中使用了固定策略。**解决方案**：明确当前是预测还是控制任务，预测时策略保持不变。
5. **自举偏差被忽视**：自举使得初始值影响收敛路径。如果 $V$ 的初始值全部为 0，而真实值函数为负数（如每步惩罚 -1 的环境），TD 需要较长时间才能从 0 调整到正确范围。**解决方案**：合理的初始化（如用小的随机值或先跑几轮 MC 获取初始估计）和较小的学习率可以帮助缓解。
6. **n 步 TD 的回报计算错误**：在实现 n 步 TD 时，需要缓存 n 步的奖励和状态，然后在第 n 步之后才能进行更新。常见错误是在回合结束前没有正确处理剩余不足 n 步的情况。**解决方案**：回合结束时，对剩余不足 n 步的状态使用 MC 方式更新（即不用自举项）。

### 调参层面
7. **n 步 TD 的 n 选择不当**：n 太小（如 n=1 即 TD(0)）方差低但偏差大，学习初期可能走弯路；n 太大（如 n=100）会像 MC 一样高方差，且需要缓存大量数据。**解决方案**：n=3~5 通常是不错的折中，既利用了多步信息又保持了合理的方差。
8. **$\lambda$ 选择不当（TD(λ)）**：$\lambda$ 太小偏差大，$\lambda$ 太小方差大。**解决方案**：通常在 0.5~0.7 范围内调试。如果环境马尔可夫性强，可以用较小的 $\lambda$；如果环境噪声大，可以用较大的 $\lambda$。

## 12. 学习总结

TD 方法是强化学习最核心的学习机制，它结合了 MC 的采样特性和 DP 的自举特性：
- **TD 目标**：$r + \gamma V(s')$——用一步真实奖励加上估计值作为更新目标
- **TD 误差**：$\delta = r + \gamma V(s') - V(s)$——衡量当前估计与目标之间的差距，是整个 TD 学习的驱动力
- **n 步 TD**：统一 TD 和 MC 的框架，n=1 是 TD(0)，n=∞ 是 MC。通过调节 n 可以在偏差和方差之间灵活权衡

**TD 在强化学习算法体系中的枢纽地位**：TD 不仅是一种独立的预测方法，更是几乎所有现代强化学习算法的核心更新机制。从 TD(0) 预测出发，将状态价值 $V(s)$ 扩展为动作价值 $Q(s,a)$，就得到 SARSA（同策略）和 Q-Learning（异策略）两种控制算法。将 TD 与神经网络结合，就得到 DQN 及其各种改进（Double DQN、Dueling DQN）。将 TD 思想应用于策略梯度的优势函数估计，就得到 GAE（广义优势估计），这是 PPO 等高级策略优化算法的关键组件。

**与前序算法的联系**：贝尔曼方程提供了值函数的递推关系，蒙特卡洛方法展示了如何通过采样估计期望，TD 方法则是将两者结合——用采样的单步转移来近似贝尔曼方程中的期望。TD 目标 $r + \gamma V(s')$ 本质上就是贝尔曼方程右边的一个单样本估计。

**后续算法的发展方向**：(1) 从预测到控制：SARSA 和 Q-Learning 将 TD 扩展到控制问题；(2) 从表格到函数近似：DQN 用神经网络替代 Q 表格；(3) 从单步到多步：n 步 TD 和 GAE 提供更灵活的回报估计；(4) 从值函数到策略梯度：TD 误差作为优势函数估计器，连接了值函数方法和策略方法。

**总结要点**：学习本节后，你应该能回答三个核心问题：(1) 这个算法解决了什么问题？(2) 它的核心创新点是什么？(3) 它与前置和后续算法的区别和联系是什么？如果这三个问题都能清晰回答，说明你真正理解了这个算法。

**TD与MC在实际项目中的选择**：在大多数现代深度强化学习项目中，TD方法是默认选择。原因是：(1) 深度学习训练通常需要大量数据，TD的在线更新可以更高效地利用每一个样本；(2) 经验回放机制（DQN的核心组件）本质上就是存储TD转移 $(s,a,r,s')$ 并反复学习；(3) GAE（广义优势估计）将TD(λ)思想应用于优势函数估计，是PPO等高级算法的标准配置。MC方法在策略梯度的基线估计中仍有价值，但通常被TD方法的优势函数估计所替代。

## 13. 练习题与思考题

### 基础题

**题目1**：TD误差$\delta = r + \gamma V(s') - V(s)$的每个部分分别代表什么含义？

**参考答案**：
- $r + \gamma V(s')$：TD目标，是对当前状态"应该"有多少价值的估计（即时奖励+未来折扣）
- $V(s)$：当前对状态价值的估计
- $\delta$：两者之差，即"预测误差"。$\delta > 0$说明状态比预期好，应上调；$\delta < 0$说明比预期差，应下调

**题目2**：为什么说TD方法是MC和DP的结合？

**参考答案**：
- **像MC**：TD从实际经验（采样）中学习，不需要环境模型
- **像DP**：TD使用自举（用估计值$V(s')$更新$V(s)$），不需要完整回合
- TD本质上是在MC的目标上做了一个近似——用$r + \gamma V(s')$替代完整的$G_t$

### 进阶题

**题目3**：为什么TD(0)的估计是有偏的？偏差的来源是什么？

**参考答案**：
偏差来源于**自举(bootstrap)**。TD目标$r + \gamma V(s')$中使用了$V(s')$——这是当前的估计值，不是真实值。如果$V(s')$估计偏高，TD目标也会偏高，导致$V(s)$被高估。

具体来说：
- MC的目标$G_t$是回报的真实值（无偏），但方差大
- TD的目标$r + \gamma V(s')$方差小（只依赖一步），但因为$V(s')$是估计值所以有偏
- 随着训练进行，$V$越来越准确，偏差逐渐减小

### 开放思考题

**题目4**：在什么情况下MC会优于TD？反过来，什么情况下TD会远优于MC？

**参考答案**：

MC优于TD的情况：
1. **非马尔可夫环境**：MC不假设马尔可夫性，TD依赖它
2. **需要无偏估计**：如策略梯度中用作基线
3. **回合很短**：MC不需要等太久，且无偏的优势体现出来

TD远优于MC的情况：
1. **持续任务（无终止）**：MC完全不适用
2. **回合很长**：MC要等很久才能更新，TD在线更新效率高得多
3. **在线实时学习**：TD每步更新，MC不行
4. **马尔可夫性强的环境**：TD能充分利用马尔可夫性质加速学习

### 开放思考题（补充）

**题目5**：在深度强化学习中，"致命三要素"（函数近似、自举、离策略学习）同时出现会导致训练不稳定。请思考：DQN通过哪些工程技巧来缓解这个问题？这些技巧分别解决了三要素中的哪一个？

**参考答案**：DQN引入了三个关键技巧来缓解致命三要素：(1) **经验回放**——打乱数据的相关性，使训练数据近似独立同分布，缓解了函数近似+自举带来的不稳定；(2) **目标网络**——延迟更新TD目标中的 $V(s')$ 或 $Q(s',a')$，减少自举带来的"追逐移动目标"问题；(3) **梯度裁剪/奖励裁剪**——限制更新幅度，防止函数近似器的发散。其中经验回放主要解决函数近似的问题（数据相关性），目标网络主要解决自举的问题（目标不稳定），两者结合有效缓解了致命三要素的负面影响。

## 14. 学习路径建议

### 前置知识
- **贝尔曼方程**：TD 更新公式 $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$ 本质上是贝尔曼期望方程的采样近似，理解贝尔曼方程是理解 TD 的前提。
- **蒙特卡洛方法**：MC 方法是 TD 的一极（n=∞ 的 n 步 TD），理解 MC 的无偏性和高方差特性有助于理解 TD 的设计动机。

### 平行学习
- **Q-Learning / SARSA**：TD 在控制问题上的直接应用。SARSA 是同策略 TD 控制（更新用实际执行的下一个动作），Q-Learning 是异策略 TD 控制（更新用 max 操作选择最优动作）。建议对比学习，理解"同策略 vs 异策略"的核心差异。

### 进阶学习
1. **SARSA**（第5章）：同策略 TD 控制。理解它为什么在 CliffWalking 中学到安全路径（因为更新考虑了探索风险），这是理解 on-policy 算法特性的经典案例。
2. **Q-Learning**（第5章）：异策略 TD 控制。理解它为什么在 CliffWalking 中学到最优但危险的路径，以及 max 操作导致的 Q 值过估计问题。
3. **DQN**（第7章）：用神经网络替代 Q 表格，将 TD 方法从表格型扩展到函数近似型。理解经验回放和目标网络如何解决 TD + 神经网络的训练不稳定问题。
4. **GAE（广义优势估计）**：TD(λ) 思想在优势函数估计中的应用，是 PPO 等策略优化算法的关键组件。理解 GAE 的指数加权机制 $\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$ 如何在偏差和方差之间权衡。

### 推荐资源
1. **《强化学习：一种介绍》（Sutton & Barto）第6章**：TD 学习的权威论述，包含 TD(0)、Sarsa、Q-Learning 的理论分析和实验对比。书中关于随机行走和 CliffWalking 的实验非常经典，建议亲手复现。
2. **《Joy RL：强化学习实践教程》第4-5章**：本书的 TD 方法章节，包含完整的代码实现和实验说明，适合边读边练。
3. **David Silver 强化学习课程第4-5讲（UCL）**：视频讲解 TD 方法和模型无关的控制方法，直观清晰，适合辅助理解理论推导。
4. **Spinning Up (OpenAI) 的 Key Papers 列表**：列出了 TD 学习到 DQN 的发展脉络中的关键论文，适合深入研究。
