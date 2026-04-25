# A2C（Advantage Actor-Critic）学习文档

> 优势Actor-Critic，用Advantage函数减少方差的策略梯度算法

---

## 1. 算法基础认知

**一句话定义**：A2C（Advantage Actor-Critic）使用Advantage函数A(s,a) = Q(s,a) - V(s)来替代原来的回报，显著降低策略梯度的方差，同时保留无偏性。

**直觉类比**：REINFORCE是"凭感觉"做决定，A2C是"冷静分析后做决定"——它知道这一步比平均水平好多少，这就是Advantage的含义。

**历史背景**：由Mnih等人在2016年提出，用于Atari游戏的异步actor-critic算法。

**算法定位**：
- 类型：强化学习 → 策略梯度
- 输出：随机策略 + 状态价值
- 模型类型：Actor-Critic with Advantage

---

## 2. 核心原理

### 2.1 Advantage函数

**定义**：
$$A(s,a) = Q(s,a) - V(s)$$

**含义**：执行动作a相对于平均水平的好坏程度

### 2.2 优势估计

**使用TD误差作为Advantage近似**：
$$\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

或者使用n步回报：
$$\hat{A}_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t)$$

### 2.3 更新公式

**Actor更新**：
$$\nabla_\theta J \approx E[\hat{A}_t \cdot \nabla_\theta \ln \pi_\theta(a_t|s_t)]$$

**Critic更新**：
$$L = [r + \gamma V(s_{t+1}) - V(s_t)]^2$$

---

## 3. 数学推导

### 3.1 为什么用Advantage？

**原始REINFORCE**：
$$\nabla_\theta J = E[G_t \cdot \nabla_\theta \ln \pi_\theta(a|s)]$$

**A2C**：
$$\nabla_\theta J = E[A(s,a) \cdot \nabla_\theta \ln \pi_\theta(a|s)]$$

因为 $E[G_t|s_t] = V(s_t)$，所以：
$$E[G_t - V(s_t)] = 0$$

这意味着减去V不会改变期望，但会大大降低方差！

### 3.2 实际实现

```python
def update(states, actions, rewards, next_states, dones):
    # Critic计算当前V
    values = critic(states).squeeze()
    
    # TD目标作为Advantage近似
    with torch.no_grad():
        next_values = critic(next_states).squeeze()
        targets = rewards + gamma * (1 - dones) * next_values
    
    advantages = targets - values
    
    # Actor: 用Advantage更新
    log_probs = actor.evaluate_log_probs(states, actions)
    actor_loss = -(advantages.detach() * log_probs).mean()
    
    # Critic: MSE损失
    critic_loss = nn.MSELoss()(values, targets)
    
    return actor_loss + 0.5 * critic_loss
```

---

## 4. 调库实现

```python
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor头
        self.actor = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic头
        self.critic = nn.Linear(256, 1)
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def act(self, state):
        probs, _ = self(torch.FloatTensor(state).unsqueeze(0))
        return torch.multinomial(probs, 1).item()

class A2CAgent:
    """A2C智能体"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, ent_coef=0.01, vf_coef=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    
    def select_action(self, state):
        with torch.no_grad():
            probs, _ = self.network(torch.FloatTensor(state).unsqueeze(0))
            return torch.multinomial(probs, 1).item()
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 前向传播
        action_probs, values = self.network(states)
        
        # Critic loss (TD目标)
        with torch.no_grad():
            _, next_values = self.network(next_states)
            targets = rewards + self.gamma * (1 - dones) * next_values.squeeze()
        
        advantages = targets - values.squeeze()
        
        # Actor loss (策略梯度)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8
        actor_loss = -(advantages.detach() * log_probs).mean()
        
        # 熵奖励（鼓励探索）
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values.squeeze(), targets)
        
        # 总损失
        total_loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

# ===============================
# 测试
# ===============================
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = A2CAgent(4, 2, lr=3e-4, gamma=0.99)
    
    print("=" * 50)
    print("A2C算法测试")
    print("=" * 50)
    
    for episode in range(200):
        # 收集经验
        states, actions, rewards, next_states, dones = [], [], [], [], []
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(200):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 更新
        agent.update(states, actions, rewards, next_states, dones)
        
        if episode % 50 == 0:
            print(f"回合{episode}: 奖励={total_reward}")
```

---

## 5. 并行版本：A3C

### 5.1 异步A3C

```python
class A3CAgent:
    """异步A2C - 使用多个worker并行收集经验"""
    
    def __init__(self, state_dim, action_dim, n_workers=4):
        self.n_workers = n_workers
        self.global_agent = A2CAgent(state_dim, action_dim)
        self.workers = [A2CAgent(state_dim, action_dim) for _ in range(n_workers)]
    
    def train_async(self):
        # 并行收集
        trajectories = [worker.collect() for worker in self.workers]
        
        # 聚合梯度
        for traj in trajectories:
            self.global_agent.update(traj)
        
        # 同步参数
        for worker in self.workers:
            worker.sync_from(self.global_agent)
```

---

## 6. 与其他算法对比

| 算法 | Advantage | 方差 | 并行 | 稳定性 |
|------|----------|------|------|------|------|
| REINFORCE | ❌ | 高 | 否 | 低 |
| A2C | ✓ | 中 | 可 | 中 |
| A3C | ✓ | 低 | ✓ | 高 |
| PPO | ✓ | 低 | 可 | 高 |

---

## 7. 总结

✓ A2C = REINFORCE + Advantage
✓ 用V(s)作为基线，显著降低方差
✓ 保持无偏性
✓ 是现代策略梯度方法的基础