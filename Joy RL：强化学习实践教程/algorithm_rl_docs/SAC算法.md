# SAC（Soft Actor-Critic）学习文档

> 基于最大熵的Actor-Critic，在稳定性和探索之间取得最佳平衡

---

## 1. 算法基础认知

**一句话定义**：SAC（Soft Actor-Critic）使用最大熵正则化，让策略在最大化回报的同时保持随机性，从而在连续控制任务中达到最佳性能。

**直觉类比**：SAC教代理"不要把所有鸡蛋放在一个篮子里"——它鼓励探索多种可能的策略，不像DDPG只学一个确定性动作。这让它更robust，不容易过拟合到某个特定解。

**历史背景**：由Haarnoja等人在2018年提出，是目前连续控制任务中表现最好的算法之一。

**算法定位**：
- 类型：强化学习 → 最大熵策略梯度
- 输出：随机策略（带熵奖励）
- 模型类型：双Critic + 软策略

---

## 2. 核心原理

### 2.1 最大熵目标

$$J(\pi) = E_{\tau \sim \pi}[\sum_t r_t + \alpha \cdot H(\pi(\cdot|s_t))]$$

**熵项**：$H(\pi) = E_{a \sim \pi}[-\log \pi(a)]$

### 2.2 软Q函数

$$Q_{soft}(s,a) = E_{\tau \sim \pi}[G_t|s_0=s,a_0=a]$$

**软Bellman方程**：
$$Q_{soft}(s,a) = r + \gamma \cdot E_{s'\sim p, a'\sim \pi}[Q_{soft}(s',a') - \alpha \cdot \log \pi(a'|s')]$$

### 2.3 三个网络

1. **Actor**：策略网络，输出动作分布
2. **Critic1, Critic2**：双Q网络（防止过估计）
3. **Target**：目标V网络

---

## 3. 数学公式

### 3.1 策略更新

$$\pi_{\phi}(\cdot|s) = \text{softmax}(f_{\phi}(s) / \alpha)$$

### 3.2 熵温度

$$\alpha = \text{自动调节}$$

或固定一个较优的α值（如0.2）

### 3.3 Critic更新（双Q取最小）

$$L = \min_i [r + \gamma \cdot V_{target}(s') - Q_i(s,a)]^2$$

其中：
$$V_{target}(s') = \min_j Q_j(s',\tilde{a}') - \log \pi_{\phi}(\tilde{a}'|s')$$

---

## 4. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

class GaussianPolicy(nn.Module):
    """高斯策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 动作均值和方差
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        h = self.net(x)
        return self.mean(h), self.log_std(h)
    
    def sample(self, state):
        mean, log_std = self(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()  # reparameterization trick
        action = torch.tanh(x_t)
        
        # 计算log概率（带tanh变换）
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        
        return action, log_prob.sum(dim=-1)

class QNetwork(nn.Module):
    """Q网络"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class SACAgent:
    """SAC智能体"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, alpha=0.2):
        self.gamma = gamma
        self.alpha = alpha
        
        # 网络
        self.policy = GaussianPolicy(state_dim, action_dim)
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        self.target_v = QNetwork(state_dim, action_dim)
        
        # 优化器
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), 
            lr=lr
        )
        
        # 初始化目标网络
        self.target_v.load_state_dict(self.q1.state_dict())
    
    def select_action(self, state):
        with torch.no_grad():
            action, _ = self.policy(torch.FloatTensor(state))
            return action.numpy()
    
    def update(self, state, action, reward, next_state, done):
        s = torch.FloatTensor(state)
        a = torch.FloatTensor(action)
        r = torch.FloatTensor(reward)
        ns = torch.FloatTensor(next_state)
        d = torch.FloatTensor(done)
        
        # ===== Critic更新 =====
        # 当前Q
        q1 = self.q1(s, a.unsqueeze(0)).squeeze()
        q2 = self.q2(s, a.unsqueeze(0)).squeeze()
        
        # 目标V
        with torch.no_grad():
            na, log_prob = self.policy(ns.unsqueeze(0))
            target_v = torch.min(self.target_v(ns, na.squeeze(0)))
            target_q = r + self.gamma * (1 - d) * (target_v - self.alpha * log_prob)
        
        # Q损失
        q1_loss = ((q1 - target_q) ** 2).mean()
        q2_loss = ((q2 - target_q) ** 2).mean()
        q_loss = q1_loss + q2_loss
        
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        
        # ===== Actor更新 =====
        new_action, log_prob = self.policy(s.unsqueeze(0))
        q_new = torch.min(self.q1(s, new_action.squeeze(0)), 
                      self.q2(s, new_action.squeeze(0)))
        policy_loss = (self.alpha * log_prob - q_new).mean()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        # ===== 软更新目标网络 =====
        for target, main in zip(self.target_v.parameters(), self.q1.parameters()):
            target.data.copy_(0.005 * main.data + 0.995 * target.data)
        
        return q_loss.item() + policy_loss.item()

# ===============================
# 测试
# ===============================
if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    agent = SACAgent(3, 1)
    
    print("=" * 50)
    print("SAC算法测试")
    print("=" * 50)
    
    for episode in range(200):
        s, _ = env.reset()
        total_reward = 0
        
        for _ in range(200):
            a = agent.select_action(s)
            s2, r, d, t, _ = env.step(a)
            total_reward += r
            
            agent.update([s], [a], [r], [s2], [float(d or t)])
            
            s = s2
        
        if episode % 50 == 0:
            print(f"回合{episode}: 奖励={total_reward:.1f}")
```

---

## 5. 与其他算法对比

| 算法 | 熵奖励 | 动作类型 | 稳定性 | 样本效率 |
|------|---------|----------|---------|----------|----------|
| DDPG | ❌ | 确定性 | 低 | 中 |
| TD3 | ❌ | 确定性 | 中 | 中 |
| SAC | ✓ | 随机 | **高** | **高** |

---

## 6. 总结

✓ SAC = 双Critic + 随机策略 + 熵奖励
✓ 最大熵 → 更好的探索
✓ 自动熵温度调节
✓ 连续控制任务的SOTA之一