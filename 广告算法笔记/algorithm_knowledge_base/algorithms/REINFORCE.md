# REINFORCE 学习文档

## 1. 算法基础认知

REINFORCE 是最基础的**策略梯度（Policy Gradient）**算法，由 Williams 于 1992 年提出。它直接对策略参数进行优化，通过**蒙特卡洛采样**估计梯度并更新策略，无需价值函数（但可引入基线降低方差）。

与基于价值的方法（Q-learning、SARSA）不同，REINFORCE 直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升最大化期望回报。

## 2. 核心原理

### 2.1 策略梯度定理

目标函数 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$，策略梯度定理表明：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

其中 $G_t = \sum_{k=t}^{T}\gamma^{k-t}r_k$ 是从时刻 $t$ 开始的折扣回报。

### 2.2 REINFORCE 更新

用蒙特卡洛采样近似期望：

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

直觉：增大导致高回报动作的概率，减小导致低回报动作的概率。

### 2.3 带基线的 REINFORCE

引入基线 $b(s)$ 减少方差：

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))$$

基线不引入偏差（$\mathbb{E}[\nabla \log \pi \cdot b] = 0$），但显著降低方差。常用基线为状态价值函数 $V(s)$。

## 3. 数学公式与推导

### 策略梯度定理推导

$$\nabla_\theta J(\theta) = \nabla_\theta \int \pi_\theta(\tau) R(\tau) d\tau = \int \nabla_\theta \pi_\theta(\tau) R(\tau) d\tau$$

利用 $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$（log-likelihood trick）：

$$= \int \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau) R(\tau) d\tau = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(\tau) R(\tau)]$$

展开轨迹概率 $\pi_\theta(\tau) = \prod_{t} \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)$，环境转移项对 $\theta$ 梯度为 0：

$$\nabla_\theta \log \pi_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

### 基线不引入偏差

$$\mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)] = \sum_{s} d(s) \sum_a \nabla_\theta \pi_\theta(a|s) \cdot b(s) = \sum_s d(s) b(s) \nabla_\theta \sum_a \pi_\theta(a|s) = 0$$

因为 $\sum_a \pi_\theta(a|s) = 1$，其梯度为 0。

## 4. 训练过程讲解

1. **初始化**：策略网络参数 $\theta$
2. **对每个 episode**：
   - 用当前策略 $\pi_\theta$ 采样完整轨迹 $(s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T)$
   - 计算每步回报 $G_t = \sum_{k=t}^{T}\gamma^{k-t}r_{k+1}$（反向递推）
   - 计算策略梯度损失：$\mathcal{L} = -\sum_{t} \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))$
   - 梯度下降更新 $\theta$
3. **重复**直到收敛

## 5. 应用场景

- **游戏 AI**：Atari、棋类等离散动作空间
- **机器人控制**：连续动作空间的运动控制
- **自然语言处理**：文本生成（RLHF 的基础组件）
- **推荐系统**：列表推荐策略优化
- **广告投放**：创意选择、出价策略

## 6. 优缺点分析

**优点**：
- 直接优化策略，可以学习随机策略
- 适用于连续动作空间
- 理论优雅，收敛性有保证
- 无需价值函数近似（基础版本）

**缺点**：
- 高方差（蒙特卡洛估计）
- 样本效率低（on-policy，数据只能用一次）
- 需要完整 episode
- 训练不稳定，对超参数敏感

## 7. 调库实现（Python）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update(self):
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.stack([-log_prob * G for log_prob, G in zip(self.log_probs, returns)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

def train(env_name='CartPole-v1', num_episodes=1000):
    env = gym.make(env_name)
    agent = REINFORCE(env.observation_space.shape[0], env.action_space.n)
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward
        agent.update()
        if (ep + 1) % 50 == 0:
            print(f'Episode {ep+1}, Reward: {total_reward}')
    return agent
```

## 8. 手工代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class REINFORCEBaseline(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.shared = nn.Linear(state_dim, hidden)
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.shared(x))
        probs = F.softmax(self.policy_head(h), dim=-1)
        value = self.value_head(h)
        return probs, value

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.forward(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

def plot_training(rewards, window=50):
    smoothed = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.plot(smoothed, label=f'{window}-ep average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('reinforce_training.png', dpi=150)
```

关键观察：
- 训练曲线波动大（高方差特征）
- 带基线版本收敛更快更稳
- CartPole 上通常 300-500 episode 收敛到满分

## 10. 模型评估

- **平均回报**：最近 N 个 episode 的平均总奖励
- **成功率**：达到目标条件的 episode 比例
- **策略熵**：$H = -\sum_a \pi(a|s)\log\pi(a|s)$，监控策略是否过早收敛
- **梯度方差**：监控梯度方差评估训练稳定性

## 11. 常见问题与易错点

- **忘记标准化回报**：不标准化导致梯度尺度不稳定
- **遗漏负号**：梯度上升对应损失函数的负号，`loss = -log_prob * return`
- **数据只用一次**：REINFORCE 是 on-policy，旧数据不能重复使用
- **熵正则化缺失**：策略可能过早退化为确定性策略，加 $-\beta H(\pi)$ 防止
- **基线选择不当**：坏基线可能增大方差，应确保 $b(s)$ 与 $G_t$ 相关

## 12. 学习总结

REINFORCE 是策略梯度方法的基础。它直接优化策略参数，通过 log-likelihood trick 将期望回报的梯度转化为对数策略概率与回报的乘积。引入基线是减少方差的关键技巧。虽然简单，REINFORCE 的思想贯穿所有高级策略梯度算法（PPO、A2C、SAC 等）。

## 13. 练习题与思考题

**Q1**：证明 $\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = 0$ 对任意 $b(s)$ 成立。

> **答案**：$\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \sum_a \nabla_\theta \pi_\theta(a|s) = b(s) \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \nabla_\theta 1 = 0$

**Q2**：为什么回报标准化不引入偏差？

> **答案**：标准化是对同一个 episode 内的回报做仿射变换（减均值除标准差），这在 batch 内等价于乘一个正数加一个常数。常数部分相当于基线（不引入偏差），正数部分只影响梯度大小不影响方向。

**Q3**：REINFORCE 和 Actor-Critic 的本质区别是什么？

> **答案**：REINFORCE 用蒙特卡洛回报 $G_t$ 作为策略梯度的权重，Actor-Critic 用学习到的价值函数 $V(s)$ 或 $Q(s,a)$ 替代。后者用 bootstrapping 减少方差但引入偏差。

## 14. 学习路径建议

1. **前置知识**：策略梯度定理、蒙特卡洛方法
2. **本节掌握**：REINFORCE 更新规则、基线技巧
3. **进阶方向**：
   - Actor-Critic（A2C/A3C）
   - PPO（Proximal Policy Optimization）
   - TRPO（Trust Region Policy Optimization）
4. **后续学习**：DDPG、SAC、TD3（连续动作空间的高级方法）
