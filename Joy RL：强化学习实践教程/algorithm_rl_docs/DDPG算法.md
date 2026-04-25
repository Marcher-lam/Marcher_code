# DDPG（深度确定性策略梯度）学习文档

> 适用于连续动作空间的深度强化学习算法

---

## 1. 算法基础认知

**一句话定义**：DDPG（Deep Deterministic Policy Gradient）结合了DQN的成功经验与确定性策略梯度，用于处理连续动作空间的控制问题。

**直觉类比**：DQN只能输出每个动作的得分，但DDPG可以直接输出"最好的那个动作"——就像从选择题变成填空题。

**历史背景**：由Lillicrap等人在2015年提出，是连续控制的经典算法。

**算法定位**：
- 类型：深度强化学习 → 连续控制
- 输出：确定性动作
- 模型类型：Actor-Critic + DQN

---

## 2. 核心原理

### 2.1 核心思想

- **Actor**：输出确定性动作μ(s)
- **Critic**：评估Q(s, μ(s))
- **目标网络**：稳定训练
- **经验回放**：像DQN一样

### 2.2 网络结构

```python
# Actor: s -> a
Actor: state -> action

# Critic: (s, a) -> Q
Critic: (state, action) -> Q-value
```

### 2.3 更新公式

**Actor更新**（最大化Q）：
$$\nabla_\theta J = \mathbb{E}[\nabla_a Q(s, a)|\_{a=\mu(s)} \nabla_\theta \mu(s)]$$

**Critic更新**（DQN风格）：
$$L = [r + \gamma Q'(s', \mu'(s')) - Q(s, a)]^2$$

---

## 3. 应用场景

### 3.1 典型应用

- 机器人控制（机械臂）
- 连续动作游戏
- 自动驾驶

### 3.2 适用条件

✓ 连续动作空间
✓ 高维状态

---

## 4. 调库实现

```python
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        
        # 复制参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
    
    def select_action(self, state):
        with torch.no_grad():
            return self.actor(state).numpy()
    
    def update(self, batch):
        # Critic更新
        Q = self.critic(batch.s, batch.a)
        target_Q = batch.r + 0.99 * self.target_critic(batch.s2, self.target_actor(batch.s2))
        
        critic_loss = nn.MSELoss()(Q, target_Q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Actor更新
        actor_loss = -self.critic(batch.s, self.actor(batch.s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # 软更新
        for target, main in zip([self.target_actor, self.target_critic], 
                               [self.actor, self.critic]):
            for t, m in zip(target.parameters(), main.parameters()):
                t.data.copy_(0.005 * m.data + 0.995 * t.data)

if __name__ == "__main__":
    import gymnasium as gym
    
    env = gym.make('Pendulum-v1')
    agent = DDPGAgent(3, 1)
    
    print("=" * 50)
    print("DDPG测试")
    print("=" * 50)
    
    for ep in range(100):
        s, _ = env.reset()
        total_r = 0
        
        for _ in range(200):
            a = agent.select_action(torch.FloatTensor(s).unsqueeze(0))
            s2, r, d, t, _ = env.step(a)
            total_r += r
            d = d or t
        
        if ep % 20 == 0:
            print(f"回合{ep}: 奖励={total_r:.1f}")
```

---

## 5. 算法特点

### 5.1 优点

✓ 连续动作空间
✓ 端到端学习
✓ 样本高效

### 5.2 缺点

✗ 超参数敏感
✗ 可能不稳定
✗ 需要大量调参

---

## 6. 总结

✓ DDPG = DQN + 策略梯度
✓ 适用于连续控制
✓ 是很多先进算法的基础