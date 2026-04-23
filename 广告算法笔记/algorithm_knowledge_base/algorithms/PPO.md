# PPO（Proximal Policy Optimization）学习文档

## 1. 算法基础认知

PPO 是一种策略梯度方法，适用于连续动作空间，直接优化探索策略的期望回报。在广告系统中被广泛用于：
- 出价策略优化
- 多目标权重动态调权
- 冷启动策略学习

## 2. 核心原理

### 策略优化目标

$$
\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

### PPO Clipped 目标函数

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中：
$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

- ε 通常设为 0.1~0.2
- 优势函数 Â_t 可以通过 GAE（Generalized Advantage Estimation）计算

### 在广告多目标调控中的 MDP 建模

| 要素 | 定义 | 具体内容 |
|------|------|---------|
| State | 当前指标状态 | (Rev_t, UX_t, Eco_t, Context_t) |
| Action | 排序权重 | (w_1^t, w_2^t, ..., w_M^t) |
| Reward | 多目标加权和 | r_t = Σ λ_k · f_k(s_t, a_t) |

## 3. PPO 训练流程

1. **Step 1：收集轨迹**：用当前策略在环境中交互，收集 (s, a, r, s') 序列
2. **Step 2：计算优势**：使用 GAE 计算每个时间步的优势函数
3. **Step 3：多轮更新（PPO 特有）**：用同一批数据更新多个 epoch（通常 3~10 个）
4. **Step 4：重复**：回到 Step 1

## 4. 代码实现（简化版）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=4):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.mse_loss = nn.MSELoss()

    def update(self, memories):
        states = torch.FloatTensor(np.array(memories['states']))
        actions = torch.LongTensor(memories['actions'])
        old_log_probs = torch.FloatTensor(memories['log_probs'])
        rewards = memories['rewards']
        dones = memories['dones']

        returns = []
        discounted_reward = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            discounted_reward = r + self.gamma * discounted_reward * (1 - d)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.k_epochs):
            action_probs, values = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            advantages = returns - values.detach().squeeze()
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

## 5. 学习总结

PPO 是当前最常用的 RL 算法之一，适用于广告出价的连续/离散动作空间。在广告多目标调控中，PPO 用于动态调整排序权重；在冷启动中，PPO/A2C 用于直接优化探索策略的期望回报。
