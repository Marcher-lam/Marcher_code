# DDPG 学习文档

> DDPG (Deep Deterministic Policy Gradient) 是一种用于连续动作空间的深度强化学习算法，结合了深度Q网络和确定性策略梯度，能够处理高维连续动作空间的任务。

---

## 1. 算法基础认知

### 一句话定义
DDPG 是一种off-policy深度强化学习算法，通过Actor-Critic架构和目标网络，能够在连续动作空间中学习确定性的最优策略。

### 直觉类比
想象学习驾驶汽车：
- **传统方法（离散）**：只能选择"左转"、"右转"、"直行"等离散动作
- **DDPG**：可以直接学习"方向盘转动15度"这样的连续动作
- 想象有一个"教练"(Critic)评估你的动作好坏，一个"驾驶员"(Actor)执行动作

### 历史背景
- 2015年，Lillicrap等人在ICML提出DDPG
- 将DQN的成功经验（目标网络、经验回放）引入连续控制
- 是连续控制任务的基准算法

### 算法定位
- **类型**：深度强化学习 / Off-policy rl
- **输出**：连续动作向量 $\mathbf{a} \in \mathbb{R}^n$
- **模型类型**：Actor-Critic + 目标网络

### 前置知识
- 强化学习基础（MDP、Q函数）
- 深度学习（神经网络训练）
- 梯度下降优化

---

## 2. 核心原理

### 2.1 核心思想
DDPG的核心思想是**将DQN扩展到连续动作空间**：

1. **Actor（策略网络）**：直接输出连续动作 $\mu(s|\theta^\mu)$
2. **Critic（Q网络）**：评估状态-动作对的价值 $Q(s,a|\theta^Q)$
3. **目标网络**：提供稳定的训练目标
4. **经验回放**：复用历史数据

关键创新：**确定性策略** $a = \mu(s)$ 而非随机策略 $\pi(a|s)$。

### 2.2 工作流程
```
环境 → 状态s → Actor → 动作a → 环境
                    ↓
                Critic → Q值评估
                    ↓
             经验回放 ← 存储(s,a,r,s')
                    ↓
            随机采样更新网络
```

### 2.3 关键概念解释
- **确定性策略**：$a = \mu(s|\theta)$，每个状态对应唯一动作
- **目标Q值**：$Y_i = r_i + \gamma Q'(s'_i, \mu'(s'_i))$
- **策略梯度**：$\nabla_\theta J \approx \mathbb{E}[\nabla_a Q(s,a) \nabla_\theta \mu(s|\theta)]$
- **Ornstein-Uhlenbeck噪声**：用于探索的时序相关噪声

### 2.4 几何/直观解释
```
┌─────────────────────────────────────────────────┐
│              DDPG 架构图                          │
│                                                 │
│   ┌─────────┐        ┌─────────┐               │
│   │ Actor  │  a=μ(s)│ Critic │               │
│   │ μ(s|θ)│───────→│ Q(s,a) │               │
│   └─────────┘        └─────────┘               │
│        ↑                   ↑                  │
│    参数θ              参数ϕ                   │
│                                         │
│   ┌─────────────────────────────────┐        │
│   │         目标网络                  │        │
│   │  μ'(s|θ')   Q'(s,a|ϕ')        │        │
│   └─────────────────────────────────┘        │
│                                         │
│   ┌─────────────────────────────────┐        │
│   │       经验回放缓冲区              │        │
│   │    [s,a,r,s',done]             │        │
│   └─────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|----------|
| $s$ | 状态 | $\mathbb{R}^n$ |
| $a$ | 动作 | $\mathbb{R}^m$ |
| $r$ | 奖励 | $\mathbb{R}$ |
| $\mu(s|\theta)$ | 确定性策略 | $\mathbb{R}^m$ |
| $Q(s,a\|\theta^Q)$ | Q函数 | $\mathbb{R}$ |
| $\gamma$ | 折扣因子 | scalar |
| $\theta$ | Actor参数 | - |
| $\theta^Q$ | Critic参数 | - |

### 3.2 问题形式化
**目标**：最大化期望累积奖励
$$J(\theta) = \mathbb{E}_{s_0}[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)]$$

其中 $a_t = \mu(s_t|\theta)$。

### 3.3 目标函数/损失函数

**Critic Loss (MSE)**：
$$\mathcal{L}_{critic} = \mathbb{E}[(Y_i - Q(s_i, a_i|\theta^Q))^2]$$
$$Y_i = r_i + \gamma (1-done_i) Q'(s'_i, \mu'(s'_i|\theta^{\mu'})|\theta^{Q'})$$

**Actor Loss**：
$$\mathcal{L}_{actor} = -\mathbb{E}[Q(s, \mu(s|\theta)|\theta^Q)]$$

简化形式：
$$\nabla_\theta \mathcal{L}_{actor} \approx \mathbb{E}[\nabla_a Q(s,a|\theta^Q) \nabla_\theta \mu(s|\theta)]$$

### 3.4 推导过程

**Step 1: 确定性策略梯度**
从DPG (Deterministic Policy Gradient) 理论：
$$\nabla_\theta J \approx \mathbb{E}_{s \sim D}[\nabla_a Q(s,a|\theta^Q) \nabla_\theta \mu(s|\theta)]$$

**Step 2: 结合深度学习**
用神经网络近似Q函数和策略：
- Actor: $\mu(s|\theta)$ 是一个神经网络
- Critic: $Q(s,a|\theta^Q)$ 是另一个神经网络

**Step 3: 引入目标网络**
使用"软更新"更新目标网络：
$$\theta' \leftarrow \tau \theta + (1-\tau) \theta'$$
其中 $\tau \ll 1$。

**Step 4: 探索噪声**
确定性策略需要添加噪声进行探索：
$$a = \mu(s|\theta) + \mathcal{N}$$
通常使用Ornstein-Uhlenbeck过程。

### 3.5 最终解/算法步骤

```python
# DDPG 伪代码

# 1. 初始化
Actor: μ(s|θ)  (网络输出连续动作)
Critic: Q(s,a|θ^Q)
目标网络: μ', Q'
经验回放: D

# 2. 主循环
for episode in episodes:
    s = env.reset()
    for step in steps:
        # 2.1 选择动作 (加探索噪声)
        a = μ(s|θ) + noise
        a = clip(a, env.action_space)
        
        # 2.2 执行
        s', r, done = env.step(a)
        D.push(s, a, r, s', done)
        
        # 2.3 更新
        if len(D) > batch_size:
            minibatch = sample(D, batch_size)
            
            # 更新Critic
            y = r + γ * Q'(s', μ'(s'))
            loss_critic = (y - Q(s,a))^2
            optimize(loss_critic)
            
            # 更新Actor
            loss_actor = -Q(s, μ(s))
            optimize(loss_actor)
            
            # 更新目标网络
            μ' = τ*μ + (1-τ)*μ'
            Q' = τ*Q + (1-τ)*Q'
        
        s = s'
```

---

## 4. 训练过程讲解

### 4.1 数据预处理
- 状态归一化
- 动作归一化到 [-1,1]
- 奖励缩放

### 4.2 参数初始化
- 权重初始化：Xavier
- 目标网络：与主网络相同

### 4.3 迭代过程

```python
"""
DDPG 完整实现 (PyTorch)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """经验回放"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """Ornstein-Uhlenbeck噪声"""
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = np.zeros_like(mu)
    
    def reset(self):
        self.x = np.zeros_like(self.mu)
    
    def sample(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(len(self.x))
        self.x += dx
        return self.x


class Actor(nn.Module):
    """Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出[-1,1]
        )
    
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DDPG:
    """DDPG算法"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 复制参数到目标网络
        self._hard_update_target(self.actor_target, self.actor)
        self._hard_update_target(self.critic_target, self.critic)
        
        # 噪声
        self.noise = OUNoise(mu=np.zeros(action_dim))
        
        # 经验回放
        self.replay_buffer = ReplayBuffer()
    
    def select_action(self, state, noise=0.0):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).cpu().numpy().squeeze(0)
        
        if noise > 0:
            action += noise * self.noise.sample()
        
        return np.clip(action, -1, 1)
    
    def update(self, batch_size=64):
        """更新网络"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # ===== Critic更新 =====
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== Actor更新 =====
        # 重新计算当前状态的Q值
        new_actions = self.actor(states)
        actor_loss = -self.critic(states, new_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ===== 目标网络更新 =====
        self._soft_update_target(self.actor_target, self.actor)
        self._soft_update_target(self.critic_target, self.critic)
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
    
    def _soft_update_target(self, target, source):
        """软更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def _hard_update_target(self, target, source):
        """硬更新"""
        target.load_state_dict(source.state_dict())


def train_ddpg(env_fn, num_episodes=500, batch_size=64):
    """训练DDPG"""
    env = env_fn()
    agent = DDPG(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        agent.noise.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            # 选择动作 (探索噪声衰减)
            noise = max(0.1, 1.0 - episode/num_episodes)
            action = agent.select_action(state, noise=noise)
            
            # 执行
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新
            agent.update(batch_size)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        if episode % 50 == 0:
            print(f"Episode {episode}: reward={episode_reward:.1f}")
    
    return rewards
```

### 4.4 收敛条件
- 平均episode奖励持续上升
- Q值趋于稳定
- 策略变化趋于平缓

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|--------|--------|
| $\gamma$ | 折扣因子 | 0.99~0.999 | 0.99 |
| $\tau$ | 软更新率 | 0.001~0.01 | 0.005 |
| batch_size | 批量大小 | 32~256 | 64 |
| buffer_size | 回放大小 | $10^5$~$10^6$ | 100000 |
| lr_actor | Actor学习率 | $10^{-5}$~$10^{-3}$ | 1e-4 |
| lr_critic | Critic学习率 | $10^{-4}$~$10^{-3}$ | 1e-3 |

---

## 5. 应用场景

### 5.1 典型应用
- **机器人控制**：机械臂、腿式机器人
- **自动驾驶**：车辆转向控制
- **游戏AI**：连续动作游戏
- **工业控制**：过程控制

### 5.2 适用数据特征
- 连续动作空间
- 低维度状态
- 有明确奖励信号

### 5.3 不适用场景
- 离散动作空间（用DQN更好）
- 稀疏奖励环境
- 高维视觉输入

---

## 6. 优缺点分析

### 6.1 优点
| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 连续动作 | 处理连续空间 | 动作连续 |
| 样本高效 | 经验回放 | 数据收集难 |
| 稳定训练 | 目标网络 | 超参数合适 |
| 简单实现 | 架构清晰 | 理解了原理 |

### 6.2 缺点
| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| Q值过估计 | 对Q值过于乐观 | TD3算法 |
| 对超参数敏感 | 学习率等 | 调参实践 |
| 探索困难 | 确定性+噪声 | 调整噪声参数 |
| 不稳定 | 高方差 | PPO/SAC |

---

## 7. 调库实现（Python + 完整代码 + 注释）

使用Stable-Baselines3：
```python
"""
DDPG调库实现 - 使用Stable-Baselines3
"""

import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# 创建环境
env = gym.make('Pendulum-v1')

# 动作噪声
action_noise = OrnsteinUhlenbeckActionNoise(
    mu=np.zeros(1),
    theta=0.15,
    sigma=0.2
)

# 创建DDPG模型
model = DDPG(
    'MlpPolicy',
    env,
    learning_rate=1e-3,
    buffer_size=100000,
    gamma=0.99,
    tau=0.005,
    action_noise=action_noise,
    verbose=1
)

# 训练
model.learn(total_timesteps=100000)

# 评估
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

简化版DDPG：
```python
"""
DDPG 核心简化实现
包含: Actor, Critic, 目标网络, 经验回放, OU噪声
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class SimpleDDPG:
    """简化DDPG"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # 复制参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 超参数
        self.gamma = 0.99
        self.tau = 0.005
        
        # 回放缓冲区
        self.replay = deque(maxlen=50000)
        
        # 噪声
        self.noise = np.zeros(action_dim)
    
    def get_action(self, state, noise=0.1):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).numpy().squeeze(0)
        
        # 添加噪声
        self.noise = 0.15 * (0 - self.noise) + 0.2 * np.random.randn(self.action_dim)
        action += noise * self.noise
        
        return np.clip(action, -1, 1)
    
    def store(self, s, a, r, s_next, done):
        self.replay.append((s, a, r, s_next, done))
    
    def update(self, batch_size=32):
        if len(self.replay) < batch_size:
            return
        
        # 采样
        batch = random.sample(self.replay, batch_size)
        s, a, r, s_next, done = map(np.array, zip(*batch))
        
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_next = torch.FloatTensor(s_next)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Critic更新
        with torch.no_grad():
            next_action = self.actor_target(s_next)
            target_q = r + self.gamma * (1 - done) * self.critic_target(s_next, next_action)
        
        current_q = self.critic(s, a)
        critic_loss = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.critic_opt.zero_grad()
        critic_loss_fn = nn.MSELoss()(current_q, target_q)
        critic_loss_fn.backward()
        self.critic_opt.step()
        
        # Actor更新
        new_action = self.actor(s)
        actor_loss = -self.critic(s, new_action).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # 目标网络更新
        self._update_target(self.actor_target, self.actor)
        self._update_target(self.critic_target, self.critic)
        
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss_fn.item()}
    
    def _update_target(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
```

---

## 9. 可视化与结果理解

```python
"""
DDPG结果可视化
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_ddpg_results(rewards, save_path='ddpg_reward.png'):
    """绘制训练结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 奖励曲线
    ax1.plot(rewards, alpha=0.3)
    window = 50
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, label=f'MA({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 最终分布
    ax2.hist(rewards[-100:], bins=20, edgecolor='black')
    ax2.axvline(np.mean(rewards[-100:]), color='r', linestyle='--', 
                label=f'Mean: {np.mean(rewards[-100:]):.1f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Final Performance')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
```

---

## 10. 模型评估

```python
def evaluate_ddpg(agent, env, num_episodes=10):
    """评估"""
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        episode_r = 0
        done = False
        
        while not done:
            action = agent.get_action(state, noise=0)
            state, r, done, _ = env.step(action)
            episode_r += r
        
        rewards.append(episode_r)
    
    return {'mean': np.mean(rewards), 'std': np.std(rewards)}
```

---

## 11. 常见问题与易错点

### 11.1 训练不收敛
**原因**：学习率过大、探索噪声不合适

**解决**：降低学习率、调整OU噪声参数

### 11.2 Q值爆炸
**原因**：目标网络更新过快

**解决**：减小 $\tau$、使用TD3

### 11.3 策略无法探索
**原因**：噪声衰减过快

**解决**：保持探索噪声

---

## 12. 学习总结

### 核心要点：
1. Actor-Critic架构处理连续动作
2. 确定性策略简化问题
3. 经验回放+目标网络稳定训练
4. OU噪声探索

### 算法链：
```
DDPG → TD3 (解决Q值过估计)
     → SAC (最大熵)
     → PPO (on-policy稳定)
```

---

## 13. 练习题

**习题1**：计算DDPG的目标Q值

<details>
<summary>答案</summary>

$$Y_i = r_i + \gamma \cdot Q'(s'_i, \mu'(s'_i))$$

其中 $\mu'$ 和 $Q'$ 是目标网络。

</details>

**习题2**：为什么DDPG需要OU噪声而不是高斯噪声？

<details>
<summary>答案</summary>

OU噪声具有时间相关性，能更好地探索连续空间（类似"惯性"），而高斯噪声每步独立，容易导致策略在原点附近抖动。

</details>

---

## 14. 学习路径建议

- **初级**：理解Actor-Critic、运行demo
- **中级**：推导策略梯度、调参
- **高级**：TD3、SAC对比研究

### 推荐资源
- 论文：Lillicrap et al. "Continuous Control with Deep RL" (2015)
- 代码：https://github.com/DLR-RM/stable-baselines3