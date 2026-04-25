# Actor-Critic 算法学习文档

> 结合策略梯度与值函数学习的混合算法框架

---

## 1. 算法基础认知

**一句话定义**：Actor-Critic（演员-评论家）框架结合了策略梯度（Actor）和值函数学习（Critic），用Critic评估来指导Actor的更新，是现代强化学习的核心框架。

**直觉类比**：就像你在看一部电影，Actor是演员的表演（策略），Critic是影评人的评价。演员需要影评人的反馈来改进表演——这就是AC的思想。

**历史背景**：由Barto、Sutton等人在1983年提出，是现代深度强化学习的基础。

**算法定位**：
- 类型：强化学习 → 策略梯度 + 值函数
- 输出：随机策略
- 模型类型：Actor-Critic混合

**前置知识**：
- [必备] 策略梯度基础
- [必备] 值函数基础

---

## 2. 核心原理

### 2.1 核心思想

- **Actor**：学习策略π(a|s)，输出动作
- **Critic**：评估V(s)或Q(s,a)，指导更新

### 2.2 工作流程

1. **Actor选择动作**：根据当前策略
2. **执行并获得奖励**
3. **Critic评估**：计算TD误差
4. **更新Actor**：用Critic的评估改进策略
5. **更新Critic**：改进值函数估计

### 2.3 关键公式

**Actor更新**（策略梯度）：
$$\nabla_\theta J \approx E[nabla_\theta \log \pi_\theta(a|s) \cdot A]$$

其中A是Advantage（优势函数）:
$$A = R + \gamma V(s') - V(s)$$

**Critic更新**：
$$L = [R + \gamma V(s') - V(s)]^2$$

---

## 3. 应用场景

### 3.1 典型应用

- **A2C/A3C**：优势Actor-Critic
- **DDPG**：确定性Actor-Critic
- **PPO**：近端策略Actor-Critic

### 3.2 适用条件

✓ 连续/离散动作
✓ 需要稳定训练

---

## 4. 调库实现

```python
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.gamma = 0.99
    
    def select_action(self, state):
        with torch.no_grad():
            probs = self.actor(torch.FloatTensor(state))
            return torch.multinomial(probs, 1).item()
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Critic更新
        values = self.critic(states).squeeze()
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            targets = rewards + self.gamma * (1 - dones) * next_values
        
        critic_loss = nn.MSELoss()(values, targets)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Actor更新
        probs = self.actor(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            advantages = targets - values
        
        actor_loss = -(log_probs * advantages).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

# ===============================
# 测试
# ===============================
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = ActorCriticAgent(4, 2)
    
    print("=" * 50)
    print("Actor-Critic测试")
    print("=" * 50)
    
    for ep in range(200):
        s, _ = env.reset()
        total_r = 0
        
        for _ in range(500):
            a = agent.select_action(s)
            s2, r, d, t, _ = env.step(a)
            total_r += r
            d = d or t
            
            # 简单的单步更新演示
            if ep % 10 == 0:
                agent.update([s], [a], [r], [s2], [float(d)])
            
            s = s2
        
        if ep % 50 == 0:
            print(f"回合{ep}: 奖励={total_r}")
```

---

## 5. 算法特点

### 5.1 优点

1. **方差 reduction**：Critic评估减少波动
2. **在线学习**：每步都可更新
3. **通用框架**：很多算法的基��

### 5.2 缺点

1. **超参数多**：需要调节
2. **可能不稳定**：需要小心设置

---

## 6. 学习路径

### 6.1 后续进阶

- [x] **A2C**：优势Actor-Critic
- [x] **DDPG**：深度确定性AC
- [x] **PPO**：近端策略AC

---

## 7. 总结

✓ AC = 策略梯度 + 值函数
✓ 用Critic的评估指导Actor
✓ 现代强化学习的基础框架