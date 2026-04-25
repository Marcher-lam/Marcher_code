# REINFORCE算法 学习文档

> 策略梯度的原始基线算法，通过Monte Carlo采样估计梯度

---

## 1. 算法基础认知

**一句话定义**：REINFORCE是最基础的策略梯度算法，通过完整回合的回报来估计策略梯度并更新策略网络，是一种完全基于 Monte Carlo 的方法。

**直觉类比**：就像你打完一把游戏后，回顾整个过程的得失，然后根据自己的"感觉"来调整下次怎么打——这就是REINFORCE的核心："凭整体感觉做决定"。

**历史背景**：由Williams在1992年提出，是所有策略梯度方法的鼻祖。

**算法定位**：
- 类型：强化学习 → 策略梯度
- 输出：随机策略网络
- 模型类型：纯策略梯度（无Critic）

---

## 2. 核心原理

### 2.1 核心思想

**策略梯度公式**：
$$\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta}[G_t \cdot \nabla_\theta \ln \pi_\theta(a_t|s_t)]$$

**更新规则**：
$$\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \ln \pi_\theta(a_t|s_t)$$

### 2.2 工作流程

1. **采样**：用当前策略生成完整回合
2. **计算回报**：计算每个时刻的折扣回报G_t
3. **计算梯度**：对每个(s,a,r)计算∇lnπ(a|s)·G_t
4. **更新**：梯度上升更新参数

---

## 3. 数学公式推导

### 3.1 目标函数

$$J(\theta) = E_{\tau \sim \pi_\theta}[\sum_{t=0}^T \gamma^t r_t]$$

### 3.2 梯度推导

**Step 1: 对数策略导数**

$$\nabla_\theta \ln \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

**Step 2: 链式法则**

$$\nabla_\theta J = \sum_t E[G_t \cdot \nabla_\theta \ln \pi_\theta(a_t|s_t)]$$

**Step 3: 实际实现**

对于离散动作：用 softmax + log_softmax
对于连续动作：用均值方差分布

---

## 4. 调库实现

```python
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

class PolicyNet(nn.Module):
    """策略网络"""
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
    
    def act(self, state):
        probs = self(torch.FloatTensor(state))
        return torch.multinomial(probs, 1).item()
    
    def evaluate_actions(self, states, actions):
        probs = self(states)
        return torch.log(probs.gather(1, actions.unsqueeze(1)))

class REINFORCEAgent:
    """REINFORCE智能体"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        
        self.policy = PolicyNet(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
    
    def collect_episode(self, env, max_steps=500):
        """收集一个完整回合"""
        trajectory = []
        state, _ = env.reset()
        
        for _ in range(max_steps):
            action = self.policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            trajectory.append((state, action, reward))
            state = next_state
            
            if done:
                break
        
        return trajectory
    
    def update(self, trajectory):
        """REINFORCE更新"""
        states, actions, rewards = [], [], []
        
        for state, action, reward in trajectory:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        
        # 计算折扣回报
        G = 0
        returns = []
        for r in reversed(rewards):
            G = self.gamma * G + r
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略梯度损失
        log_probs = self.policy.evaluate_actions(states, actions).squeeze()
        loss = -(log_probs * returns).mean()
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# ===============================
# 测试
# ===============================
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = REINFORCEAgent(4, 2, lr=3e-4, gamma=0.99)
    
    print("=" * 50)
    print("REINFORCE算法测试")
    print("=" * 50)
    
    for episode in range(300):
        # 收集回合
        trajectory = agent.collect_episode(env)
        
        # 更新
        agent.update(trajectory)
        
        if episode % 50 == 0:
            # 评估
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.policy.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            print(f"回合{episode}: 奖励={total_reward}")
```

---

## 5. 与其他算法对比

| 算法 | 是否用Critic | 方差 | 偏差 | 样本效率 |
|------|-----------|------|------|----------|
| REINFORCE | ❌ | 高 | 无 | 低 |
| Actor-Critic | ✓ | 中 | 有 | 中 |
| A2C | ✓ | 低 | 有 | 高 |

---

## 6. 常见问题与改进

### 6.1 高方差问题

REINFORCE的主要问题是方差很高：
- 一个回合的随机性太大
- 梯度估计不稳定

**解决**：加入基线（baseline）来减少方差
$$A_t = G_t - b(s_t)$$

### 6.2 改进版：带基线的REINFORCE

```python
# 加上一个值函数作为基线
class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNet(state_dim, action_dim)
        self.value_net = nn.Linear(state_dim, 1)  # 基线网络
        
    def update(self, trajectory):
        # 计算回报
        G = 0
        for _, _, r in reversed(trajectory):
            G = gamma * G + r
        
        # 计算Advantage
        for state, action, reward in trajectory:
            baseline = self.value_net(state)
            advantage = G - baseline
            
            # 用advantage来更新
            log_prob = self.policy.log_prob(state, action)
            policy_loss = -advantage * log_prob
            value_loss = (baseline - G) ** 2
            
            # 更新...
```

---

## 7. 总结

✓ REINFORCE = 最基础的策略梯度算法
✓ 优点：实现简单、无偏
✓ 缺点：方差高
✓ 改进：加入Critic作为基线（变成Actor-Critic）