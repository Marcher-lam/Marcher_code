# 面试题：DQN、Double DQN 和 Dueling DQN，三者原理与区别

# 1 DQN

深度 Q 网络（Deep Q-Network, DQN）是深度强化学习的基础算法，其核心思想是用神经网络近似 Q-learning 中的动作价值函数（Q 函数），从而处理高维状态空间（如图像输入）的问题。

传统 Q-learning 在状态空间过大或连续时，无法通过表格方式存储 Q 值，DQN 通过参数化的函数 $Q _ { \theta }$ 来拟合最优 Q 值函数。

# 1.1 基本原理

 在 Q-learning 中，需要优化的目标函数为：

$$
\min  _ {\theta} J (\theta) = \mathbb {E} \left[ \left(R + \gamma \max  _ {a} Q \left(S ^ {\prime}, a; \theta\right) - Q (S, A; \theta)\right) \right]
$$

其中 R 表示即时奖励， $\gamma$ 为折扣因子，S 和 A 分别表示当前状态和动作， $S ^ { \prime }$ 表示下一状态。

 DQN 的 TD 目标（Temporal Difference Target）为：

$$
Y _ {t} ^ {D Q N} = R _ {t + 1} + \gamma \max  _ {a} Q \left(S _ {t + 1}, a; \theta^ {-}\right)
$$

其中 $\theta$ 是训练网络的参数， $\theta ^ { - }$ 是目标网络的参数。

# 1.2 主要创新

DQN引入了两个关键技术创新：

 经验回放（Experience Replay）：智能体与环境交互的经验 $( s , a , r , s ^ { \prime } , \mathrm { d o n e } )$ 被存储到经验池中，训练时从池中随机采样。这解决了数据间相关性带来的训练不稳定性问题，同时提高了样本利用率。  
 目标网络（Target Network）：DQN 使用两套网络——训练网络（参数 $\theta$ ）和目标网络（参数 $\theta ^ { - }$ ）。TD 目标的计算基于目标网络，定期将训练网络参数复制给目标网络（通常每 $\tau$ 步一次），极大提升了训练稳定性。

# 2 Double DQN

Double DQN（DDQN）是针对 DQN 存在的 Q 值过高估计（overestimation）问题提出的改进算法。传统 DQN 的 max 操 作会使 Q 值的估计越来越高于真实值，导致策略次优和训练不稳定。

# 2.1 过高估计问题及其解决

在传统 DQN 中，TD 目标为：

$$
Y _ {t} ^ {D Q N} = R _ {t + 1} + \gamma Q \left(S _ {t + 1}, \arg \max  _ {a} Q \left(S _ {t + 1}, a; \theta^ {-}\right); \theta^ {-}\right)
$$

这相当于使用同一套目标网络 θ−同时选择动作（argmax 操作）和评估价值（Q 值计算），导致估计偏差累积。

Double DQN 通过解耦动作选择与价值评估来解决这个问题：

$$
Y _ {t} ^ {D D Q N} = R _ {t + 1} + \gamma Q \left(S _ {t + 1}, \arg \max  _ {a} Q \left(S _ {t + 1}, a; \theta\right); \theta^ {-}\right)
$$

即利用训练网络 θ 选择动作（argmax），然后用目标网络 θ−评估该动作的价值。

# 2.2 数学推导

Double DQN 的优化目标函数变为：

$$
\min  _ {\theta} J (\theta) = \mathbb {E} \left[ \left(R + \gamma Q \left(S ^ {\prime}, \arg \max  _ {a ^ {\prime}} Q \left(S ^ {\prime}, a ^ {\prime}; \theta\right); \theta^ {-}\right) - Q (S, A; \theta)\right) \right]
$$

这样即使训练网络 θ 对某个动作存在过高估计，目标网络 $\theta ^ { - }$ 的评估也能抵消部分偏差，使 Q 值估计更接近真实值，提高算法稳定性和收敛性。

# 3 Dueling DQN

Dueling DQN 采用了网络结构创新，通过分解 Q 值函数为状态价值和动作优势两个部分，来更有效地评估状态和动作的价值。

![](images/7eb7bf7252e2200651959d4b21f6ec58d9f086b8ad4dcc7482243e00ecbe6af3.jpg)  
Figure 1. A popular single stream $Q$ -network (top) and the dueling $Q$ -network (bottom). The dueling network has two streams to separately estimate (scalar) state-value and the advantages for each action; the green output module implements equation (9) to combine them. Both networks output $Q$ -values for each action.

# 3.1 价值函数与优势函数

Dueling DQN 的核心思想来源于优势函数（Advantage Function）的概念：

 状态价值函数 V(s)：衡量处于状态 s 的好坏程度  
 动作价值函数 Q(s,a)：衡量在状态 s 下选择动作 a 的长期回报  
优势函数 A(s,a)：定义为 A(s,a)=Q(s,a)−V(s)，表示动作 a 相对于平均水平的优势程度对优势函数取期望 $\mathbb { E } _ { a \sim \pi } [ A ( s , a ) ] = 0$ ，即优势函数在所有动作上的平均值为零。

# 3.2 网络架构与公式

Dueling DQN 将传统 DQN 的单一 Q 网络输出层分为两个分支：

 价值流（Value Stream）：输出标量 $V ( s ; \theta , \beta )$ ，表示状态价值

 优势流（Advantage Stream）：输出向量 $A ( s , a ; \theta , \alpha )$ ，表示每个动作的优势值

最终 Q 值的计算方式为：

$$
Q (s, a; \theta , \alpha , \beta) = V (s; \theta , \beta) + \left(A (s, a; \theta , \alpha) - \max  _ {a ^ {\prime} \in A} A (s, a ^ {\prime}; \theta , \alpha)\right)
$$

实践中也常使用均值形式：

$$
Q (s, a; \theta , \alpha , \beta) = V (s; \theta , \beta) + \left(A (s, a; \theta , \alpha) - \frac {1}{\mathcal {A}} \sum_ {a ^ {\prime}} A (s, a ^ {\prime}; \theta , \alpha)\right)
$$

这种结构强制优势函数零中心化，解决了辨识性问题（V和A的相对尺度不确定），同时使网络能更高效地学习状态价值表示。

# 4 三者对比与适用场景

<table><tr><td>特性</td><td>DQN</td><td>Double DQN</td><td>Dueling DQN</td></tr><tr><td>核心创新</td><td>基础算法：神经网络近似Q函数+经验回放+目标网络</td><td>解耦动作选择与价值评估</td><td>网络结构分离：
状态价值V+动作优势A</td></tr><tr><td>TD目标公式</td><td>Yt=r+γmaxaQ(s&#x27;,a;θ-)</td><td>Yt=r+γQ(s&#x27;,arg maxaQ(s&#x27;,a;θ);θ-)</td><td>与DQN或Double DQN相同，但Q网络结构不同</td></tr><tr><td>解决的问题</td><td>处理高维状态空间，稳定训练</td><td>减轻Q值过高估计</td><td>更好评估状态价值，尤其动作影响较小时</td></tr><tr><td>训练稳定性</td><td>相对较低，存在过高估计</td><td>较高，减轻了过高估计</td><td>较高，学习更鲁棒的状态表征</td></tr><tr><td>计算复杂度</td><td>较低</td><td>略高于DQN（需两次前向传播）</td><td>与DQN相当（分支结构增加参数不多）</td></tr><tr><td>适用动作空间</td><td>离散动作空间</td><td>离散动作空间</td><td>离散动作空间（尤其是动作数量较多时）</td></tr></table>

#  DQN 适用场景：

适用于中等复杂度环境、离散动作空间、作为基础学习算法。例如简单的 Atari游戏（如 Pong）、低维状态空间的决策问题。作为基础算法，适合初学者理解和实现深度强化学习的基本原理。

#  Double DQN 适用场景：

适用于需要减少 Q 值过高估计的环境，特别是那些奖励稀疏或需要长时间规划的任务。在许多 Atari 游戏（如 SpaceInvaders）中，Double DQN 相比 DQN 能取得更好的性能和稳定性。也适用于医疗诊断、金融交易等对估计准确性要求较高的领域。

#  Dueling DQN 适用场景：

适用于状态价值至关重要而单个动作影响相对较小的环境。例如自动驾驶中，环境状态（道路、交通情况）比具体动作（微小转向调整）更重要；或者资源分配问题中，状态（资源总量）比具体分配动作更关键。在动作空间较大的环境中，Dueling 结构能显著提高学习效率。

# 5 PyTorch 实现对比

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """基础 DQN 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN 网络：分离价值流与优势流"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class DQNAgent:
    """DQN / Double DQN / Dueling DQN 统一实现"""
    def __init__(self, state_dim, action_dim, mode="dqn", hidden_dim=128, lr=1e-3, gamma=0.99, tau=10):
        self.mode = mode
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mode == "dueling":
            self.policy_net = DuelingDQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net = DuelingDQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.step_count = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state_t).argmax(dim=1).item()

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.mode == "double":
                best_actions = self.policy_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.tau == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()


def train_and_compare():
    """训练循环示例（以 CartPole 为例）"""
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {}
    for mode in ["dqn", "double", "dueling"]:
        agent = DQNAgent(state_dim, action_dim, mode=mode)
        episode_rewards = []

        for ep in range(100):
            state, _ = env.reset()
            total_reward = 0
            epsilon = max(0.01, 1.0 - ep / 80)

            while True:
                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.buffer.push(state, action, reward, next_state, float(done))
                agent.update(64)
                state = next_state
                total_reward += reward
                if done:
                    break

            episode_rewards.append(total_reward)
            if (ep + 1) % 20 == 0:
                avg = np.mean(episode_rewards[-20:])
                print(f"[{mode.upper():>8}] Episode {ep+1:3d} | 平均奖励: {avg:.1f}")

        results[mode] = episode_rewards

    print("\n三种算法最终 20 回合平均奖励对比:")
    for mode, rewards in results.items():
        print(f"  {mode.upper():>8}: {np.mean(rewards[-20:]):.1f}")


if __name__ == "__main__":
    train_and_compare()
```

## 常见问题与易错点

1. **目标网络更新频率**：$\tau$ 过大则训练不稳定，过小则收敛慢，通常设为 100-1000 步
2. **Double DQN 的动作选择网络**：必须使用训练网络（policy_net）选动作，用目标网络（target_net）评估价值，两个网络不能搞反
3. **Dueling 的均值归一化**：用 `advantage.mean(dim=-1, keepdim=True)` 而非 `max`，保证梯度稳定传播到价值流
4. **经验回放池大小**：过小会导致样本相关性高，过大会占用过多内存，通常设为 $10^4$ 到 $10^6$

## 学习总结

DQN 系列的演进逻辑清晰：DQN 解决"高维状态空间"问题，Double DQN 解决"Q 值过高估计"问题，Dueling DQN 解决"状态价值与动作优势混叠"问题。三者可组合使用（如 Dueling Double DQN），在实际应用中推荐以 Double DQN 为基础配置。
