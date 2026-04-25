# TD3 (Twin Delayed DDPG) 双延迟深度确定性策略梯度 学习文档

> TD3是强化学习中用于连续控制的actor-critic算法，是DDPG的改进版本，解决了Q值过估计问题

---

## 1. 算法基础认知

### 1.1 一句话定义

**TD3 (Twin Delayed Deep Deterministic Policy Gradient)** 是一种基于深度强化学习的连续控制算法，通过引入双Q网络、延迟策略更新和目标策略平滑三个关键技术，显著提升了DDPG算法的稳定性和性能，是目前最流行的连续控制算法之一。

### 1.2 直觉类比

想象你在学习打网球：1）你有一个教练（Critic）评估你的每个动作，但教练有时会过度乐观（Q值过估计）；2）你同时找两个教练取平均变得更可靠（双Q网络）；3）你不会每次击球都请教教练，而是每隔几次才学习新动作（延迟更新）；4）教练偶尔会给你一些随机建议让你不要过于死板（目标策略平滑）。这就是TD3的核心思想！

### 1.3 历史背景

| 年份 | 里程碑 |
|------|--------|
| 2014 | DDPG - 深度确定性策略梯度 |
| 2017 | TD3 - 解决DDPG过估计问题 |
| 2018 | SAC - 最大熵actor-critic |
| 2019 | PPO - 策略梯度改进 |
| 2020 | C51, QR-DQN - 值函数分布 |

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 连续控制 / Actor-Critic |
| 核心 | 双Q网络 + 延迟更新 |
| 状态 | On-policy / 离线策略 |
| 优点 | 稳定、高样本效率 |

### 1.5 前置知识

- 强化学习基础（MDP, Bellman方程）
- 深度学习（神经网络、反向传播）
- Python + PyTorch

---

## 2. 核心原理

### 2.1 DDPG的问题

**DDPG的Q值过估计**：

1. **贝尔曼更新**使用max操作：
$$Q(s,a) \leftarrow r + \gamma \cdot \max_{a'} Q(s', a')$$

2. **过估计原因**：
   - 最大化包含噪声估计误差
   - 误差会累积并传播
   - 导致策略退化

3. **过估计的负面影响**：
   - 价值被高估
   - 策略被误导
   - 最终崩溃

### 2.2 TD3的三个核心技术

**1. 双Q网络（Twin Q-networks）**：

使用两个独立的Q网络，取较小的值作为目标：
$$Y = r + \gamma \cdot \min(Q_1(s', a'), Q_2(s', a'))$$

这有效缓解了过估计。

**2. 延迟策略更新（Delayed Policy Updates）**：

Actor不必每步更新，而是每隔 $d$ 步更新一次：
```python
if step % d == 0:
    update_actor()
    update_target()
```

**3. 目标策略平滑（Target Policy Smoothing）**：

在目标动作添加噪声：
$$a' = \pi(s') + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma)$$

这防止策略过度拟合到特定值函数尖峰。

### 2.3 整体框架

```python
# TD3伪代码
for state in buffer:
    # 1. 从策略采样动作
    action = actor(state) + noise
    
    # 2. 环境交互
    next_state, reward = env.step(action)
    
    # 3. 存储到replay buffer
    
    # 4. 每步更新Q网络（每步）
    for _ in gradient_steps:
        # Twin Q targets
        target_Q = r + gamma * min(Q1_target, Q2_target)
        critic_loss = MSE(Q1(s,a), target_Q) + MSE(Q2(s,a), target_Q)
    
    # 5. 每d步更新策略和目标
    if step % delay == 0:
        actor_loss = -Q1(s, actor(s))
        update_actor()
        soft_update_targets()
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $s \in \mathcal{S}$ | 状态空间 |
| $a \in \mathcal{A}$ | 动作空间 |
| $\pi_\theta(a|s)$ | 策略网络 |
| $Q_{\phi}(s,a)$ | Q网络 |
| $Q_1, Q_2$ | 双Q网络 |
| $\mu_{\theta}(s)$ | 确定性策略 |
| $\gamma$ | 折扣因子 |

### 3.2 Q网络目标

**标准Bellman算子**：
$$\mathcal{T} Q(s,a) = r(s,a) + \gamma \cdot \mathbb{E}_{s'}[V(s')]$$

其中 $V(s') = \max_{a'} Q(s', a')$

**TD3目标**（使用双Q取小）：
$$Y = r + \gamma \cdot \min(Q_1(s', a'), Q_2(s', a'))$$

### 3.3 Critic损失

$$L_C = \mathbb{E}[(Q(s,a) - Y)^2]$$

对于双Q网络：
$$L_{C1} = \mathbb{E}[(Q_1(s,a) - Y)^2]$$
$$L_{C2} = \mathbb{E}[(Q_2(s,a) - Y)^2]$$

### 3.4 Actor损失

策略梯度（最大化Q）：
$$\nabla_\theta L = \mathbb{E}[\nabla_a Q_1(s,a)|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)]$$

简化为：
$$L_A = -\mathbb{E}[Q_1(s, \mu_\theta(s))]$$

### 3.5 目标网络更新

**软更新**（Polyak更新）：
$$\phi_{target} = \tau \cdot \phi + (1-\tau) \cdot \phi_{target}$$

典型值：$\tau = 0.005$

**硬更新**（每N步复制）：
$$\theta_{target} = \theta$$

### 3.6 策略梯度推导

**确定性策略梯度（DPG）**：
$$\nabla_\theta J = \mathbb{E}[\nabla_a Q(s,a)|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)]$$

**证明**：从性能目标出发
$$J(\theta) = \mathbb{E}_{s \sim \rho^\pi}[R(s, \mu_\theta(s))]$$

对 $\theta$ 求导：
$$\nabla_\theta J = \mathbb{E}_{s \sim \rho^\pi}[\nabla_a Q(s,a) \cdot \nabla_\theta \mu_\theta(s)]$$

---

## 4. PyTorch实现

### 4.1 核心网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    """Actor网络：连续动作的确定性策略"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.max_action * self.net(state)
    
    def action(self, state):
        """确定性动作"""
        return self.forward(state)


class Critic(nn.Module):
    """Critic网络：Q函数"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2（双Q）
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def q1(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)
```

### 4.2 TD3算法

```python
class TD3:
    """Twin Delayed DDPG"""
    
    def __init__(self, state_dim, action_dim, max_action=1.0,
                 hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005,
                 policy_delay=2, expl_noise=0.1, policy_noise=0.2,
                 noise_clip=0.5):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # 双Critic
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 初始化目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 训练统计
        self.total_it = 0
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        state = torch.FloatTensor(state.reshape(1, -1))
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        
        if not deterministic:
            # 探索噪声
            noise = np.random.normal(0, self.expl_noise)
            action = action + noise
        
        return np.clip(action, -self.max_action, self.max_action)
    
    def train(self, replay_buffer, batch_size=256):
        """单步训练"""
        self.total_it += 1
        
        # 从buffer采样
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # 目标策略平滑噪声
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            # 目标动作
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # 双Q目标（取小）
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Bellman目标
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        # Critic损失
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟策略更新
        if self.total_it % self.policy_delay == 0:
            
            # Actor损失（用Q1）
            actor_action = self.actor(state)
            current_Q1 = self.critic.q1(state, actor_action)
            actor_loss = -current_Q1.mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if self.total_it % self.policy_delay == 0 else 0,
        }
    
    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1 - self.tau)
            )
```

### 4.3 Replay Buffer

```python
class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))
    
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        index = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.state[index]),
            torch.FloatTensor(self.action[index]),
            torch.FloatTensor(self.reward[index]),
            torch.FloatTensor(self.next_state[index]),
            torch.FloatTensor(self.done[index])
        )
```

### 4.4 训练循环

```python
def train_td3(env, num_episodes=1000, max_steps=1000):
    """TD3完整训练"""
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    
    # 创建TD3
    agent = TD3(state_dim, action_dim, max_action)
    buffer = ReplayBuffer(state_dim, action_dim)
    
    # 训练统计
    rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            
            # 环境交互
            next_state, reward, done, _ = env.step(action)
            
            # 存储
            buffer.add(state, action, reward, next_state, done)
            
            # 训练
            if buffer.size > batch_size:
                agent.train(buffer, batch_size)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")
    
    return agent, rewards
```

---

## 5. 代码示例

### 5.1 完整示例

```python
import gym
import numpy as np
import matplotlib.pyplot as plt


def demo_td3_cartpole():
    """TD3在CartPole上的演示"""
    
    print("=" * 60)
    print("TD3 (Twin Delayed DDPG) 演示")
    print("=" * 60)
    
    env = gym.make('Pendulum-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"最大动作: {max_action}")
    
    # 创建TD3
    agent = TD3(state_dim, action_dim, max_action)
    buffer = ReplayBuffer(state_dim, action_dim)
    
    # 预填充buffer
    print("\n预填充经验池...")
    state = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()
    
    # 训练
    print("\n训练中...")
    rewards_history = []
    
    for episode in range(500):
        state = env.reset()
        episode_reward = 0
        
        for step in range(200):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            buffer.add(state, action, reward, next_state, done)
            
            # 训练更新
            agent.train(buffer, batch_size=256)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards_history.append(episode_reward)
        
        if episode % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"Episode {episode}: Avg Reward = {avg:.2f}")
    
    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('TD3 Training Curve')
    plt.savefig('td3_training.png', dpi=150)
    plt.close()
    
    return agent, rewards_history


def compare_ddpg_td3():
    """对比DDPG和TD3"""
    
    # DDPG参数
    ddpg_config = {
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'tau': 0.001,
    }
    
    # TD3参数
    td3_config = {
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'tau': 0.005,
        'policy_delay': 2,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
    }
    
    print("\n超参数对比:")
    print("-" * 40)
    print(f"{'参数':<20} {'DDPG':<15} {'TD3':<15}")
    print("-" * 40)
    for key in ddpg_config:
        print(f"{key:<20} {ddpg_config[key]:<15} {td3_config[key]:<15}")
    
    print("\n核心差异：")
    print("- DDPG: 单Q网络，直接策略更新")
    print("- TD3: 双Q网络，取min目标")
    print("- TD3: 延迟策略更新")
    print("- TD3: 目标策略平滑")


if __name__ == "__main__":
    agent, history = demo_td3_cartpole()
    compare_ddpg_td3()
```

---

## 6. 应用场景

### 6.1 连续控制

| 环境 | 说明 |
|------|------|
| **MuJoCo** | HalfCheetah, Hopper, Walker |
| **PyBullet** | Ant, Humanoid |
| **Robotics** | 机��臂控制 |

### 6.2 实际应用

| 应用 | 说明 |
|------|------|
| **机器人** | 运动控制 |
| **自动驾驶** | 转向/速度控制 |
| **游戏AI** | 连续动作游戏 |

### 6.3 代码

```python
# 使用gymnasium
import gymnasium as gym

env = gym.make('HalfCheetah-v4')
agent = TD3(env)

for episode in range(1000):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(1000):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: {total_reward:.2f}")
```

---

## 7. 优缺点分析

### 7.1 优点

| 优点 | 说明 |
|------|------|
| **稳定** | 解决DDPG的过估计 |
| **高效** | 离线策略，样本效率高 |
| **简单** | 实现相对简单 |
| **灵活** | 可与其他技术结合 |

### 7.2 缺点

| 缺点 | 说明 | 缓解 |
|------|------|------|
| **超参敏感** | 需要调参 | Grid Search |
| **探索** | 需要噪声策略 | 参数调节 |
| **收敛** | 训练可能不稳定 | 降低学习率 |

### 7.3 对比

| 算法 | Q过估计 | 策略延迟 | 目标平滑 | 稳定性 |
|------|--------|----------|----------|--------|
| DDPG | 有 | 无 | 无 | 差 |
| TD3 | 使用min | 有 | 有 | 好 |
| SAC | 有 | 有 | 无 | 好 |
| PPO | 无 | 无 | 无 | 好 |

---

## 8. 常见问题与易错点

### 8.1 问题1：训练不收敛

**可能原因**：
1. 学习率过高
2. Replay buffer太小
3. 探索噪声过大

**解决方案**：
```python
# 降低学习率
agent.actor_optimizer = optim.Adam(agent.actor.parameters(), lr=1e-4)

# 增加buffer大小
buffer = ReplayBuffer(state_dim, action_dim, max_size=1e7)

# 减少探索噪声
expl_noise = 0.05
```

### 8.2 问题2：Q值爆炸

**可能原因**：
1. 没有target网络
2. 目标更新太快
3. 奖励缩放不对

**解决方案**：
```python
# 使用target网络
target_Q = reward + gamma * min(Q1_target, Q2_target)

# 减小tau
tau = 0.001

# 缩放奖励
reward = reward / 100
```

### 8.3 问题3：策略退化

**可能原因**：
1. Q过估计
2. 探索不足
3. 局部最优

**解决方案**：
```python
# 使用TD3的双Q和延迟更新
policy_noise = 0.1
policy_delay = 2

# 增加探索
expl_noise = 0.2
```

---

## 9. 学习总结

### 9.1 核心要点

1. **双Q网络**：取min缓解过估计
2. **延迟更新**：每隔几步更新策略
3. **目标平滑**：添加噪声防止过拟合

### 9.2 关键公式

$$Y = r + \gamma \cdot \min(Q_1(s', a'), Q_2(s', a'))$$

$$L_C = (Q_1 - Y)^2 + (Q_2 - Y)^2$$

$$L_A = -Q_1(s, \mu_\theta(s))$$

### 9.3 学习路径

强化学习基础 → DDPG → TD3 → SAC → PPO

---

## 10. 练习题

### 10.1 基础题

1. 解释为什么TD3使用双Q网络
2. 目标策略平滑的作用是什么

### 10.2 进阶题

3. 实现自己的TD3变体
4. 比较TD3和SAC的性能

### 10.3 答案

<details>
<summary>答案1</summary>

因为最大化操作会放大Q值的估计误差。通过取两个Q网络的最小值，可以减少这种放大效应，从而缓解过估计问题。

</details>

<details>
<summary>答案2</summary>

目标策略平滑在目标动作上添加高斯噪声，使策略不会过度拟合到值函数的特定尖峰，提高泛化能力和稳定性。

</details>

---

## 11. 学习路径建议

### 11.1 第一阶段

1. 理解强化学习基础
2. 理解DDPG原理
3. 实现基础TD3

### 11.2 第二阶段

1. 调参实践
2. MuJoCo环境实验
3. 理解理论

### 11.3 第三阶段

1. 学习SAC、PPO
2. 论文阅读
3. 项目实践

---

## 12. 可视化与结果理解

```python
def visualize_q_values():
    """可视化Q值学习"""
    
    # 记录Q值
    q_values = []
    
    for episode in range(100):
        # 训练
        ...
        q_values.append(current_q)
    
    plt.figure(figsize=(10, 5))
    plt.plot(q_values)
    plt.xlabel('Episode')
    plt.ylabel('Q Value')
    plt.title('Critic Q Value Learning')
    plt.show()


def visualize_policy():
    """可视化策略"""
    
    # 可视化连续动作
    states = np.linspace(-2, 2, 20)
    actions = [agent.select_action(s, deterministic=True) for s in states]
    
    plt.figure(figsize=(10, 5))
    plt.plot(states, actions)
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.title('Learned Policy')
    plt.show()
```

---

## 13. 模型评估

### 13.1 评估指标

| 指标 | 说明 |
|------|------|
| **Episode Reward** | 累积奖励 |
| **Average Return** | 平均回报 |
| **Sample Efficiency** | 样本效率 |

### 13.2 代码

```python
def evaluate_agent(agent, env, num_episodes=10):
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
    
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
    }
```

---

## 14. 进阶内容

### 14.1 与其他算法对比

| 算法 | 类型 | 样本效率 | 稳定性 | 适用场景 |
|------|------|----------|--------|----------|
| TD3 | Off-policy | 高 | 中 | 连续控制 |
| SAC | Off-policy | 高 | 高 | 连续控制 |
| PPO | On-policy | 低 | 高 | 通用 |
| DDPG | Off-policy | 高 | 低 | 连续控制 |

### 14.2 扩展方向

1. **最大熵TD3**：结合熵正则
2. **分布式TD3**：使用C51
3. **离线TD3**：CQL

### 14.3 推荐资源

- Addressing Function Approximation Error in Actor-Critic Methods (TD3原始论文)
- OpenAI Spinning Up
- CleanRL

---

**文档结束**

*参考论文：Addressing Function Approximation Error in Actor-Critic Methods (Fujita et al., 2018)*