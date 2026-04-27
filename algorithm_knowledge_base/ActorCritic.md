# Actor-Critic 学习文档

> 结合策略梯度与值函数近似的强化学习方法。

## 1. 算法基础认知

Actor-Critic（演员-评论家）结合了策略梯度（Actor）和值函数近似（Critic）的优点。Actor负责选择动作，Critic负责评估动作的价值，两者协作学习。

**直觉类比**：像学打网球，Actor像教练教你在不同局面用什么动作，Critic像解说员评论这一球的得失。两者配合，Actor越来越会打球，Critic越来越会评价。

**前置知识**：REINFORCE、Q-Learning

## 2. 核心原理

**两个组件**：
- Actor：策略网络，输出动作
- Critic：值网络，评估价值

## 3. 数学公式与推导

**Advantage函数**：
$$A(s,a) = Q(s,a) - V(s)$$

**更新规则**：
- Actor：$\nabla_\theta J = \nabla_\theta \log \pi_\theta(a|s) A(s,a)$
- Critic：$L = (R + \gamma V(s') - V(s))^2$

## 4. 训练过程讲解

**参数**：
| 参数 | 作用 |
|------|------|
| actor_lr | Actor学习率 |
| critic_lr | Critic学习率 |

## 5. 应用场景

- 连续控制
- 机器人
- 游戏AI

## 6. 优缺点分析

**优点**：低方差、可以直接学习策略
**缺点**：需要两个网络

## 7. 调库实现

```python
"""
Actor-Critic算法
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    """Actor策略网络"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    """Critic值网络"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class A2CAgent:
    """A2C智能体 (同步版本)"""
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.001, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def choose_action(self, state):
        """选择动作"""
        with torch.no_grad():
            probs = self.actor(torch.FloatTensor(state))
            action = np.random.choice(self.action_dim, p=probs.numpy())
        return action
    
    def update(self, state, action, reward, next_state, done):
        """更新"""
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([1 if done else 0])
        
        # Critic更新
        V = self.critic(state)
        with torch.no_grad():
            V_next = self.critic(next_state)
            target = reward + (1 - done) * self.gamma * V_next
        
        critic_loss = (target - V).pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor更新
        probs = self.actor(state)
        log_prob = torch.log(probs[0, action] + 1e-8)
        
        advantage = (target - V).detach()
        actor_loss = -log_prob * advantage
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# 测试
np.random.seed(42)
torch.manual_seed(42)

agent = A2CAgent(4, 4)

for step in range(500):
    state = np.random.randn(4)
    action = agent.choose_action(state)
    next_state = np.random.randn(4)
    reward = np.random.randn()
    done = np.random.choice([True, False])
    
    agent.update(state, action, reward, next_state, done)

print("Actor-Critic训练完成")
print(f"Actor网络结构: {agent.actor.network}")
```

## 8-14. 其他章节

**学习总结**：Actor-Critic结合了策略梯度和值函数，是现代强化学习的核心架构。

**核心公式**：
- Actor: $\nabla_\theta J = E[\nabla_\theta \log \pi_\theta(a|s) A(s,a)]$
- Critic: $L = (R + \gamma V(s') - V(s))^2$

> 来源线索：本节内容根据原书中关于"Actor-Critic"的相关章节整理。