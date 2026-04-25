# Double DQN 学习文档

> 解决DQN中Q值过估计问题的改进算法

---

## 1. 算法基础认知

**一句话定义**：Double DQN（Double Deep Q-Network）通过使用两个Q网络来分别选择动作和评估动作，解决DQN中max操作导致的Q值过估计问题。

**直觉类比**：DQN就像让一个人同时做"提议"和"裁判"，容易"打分虚高"；Double DQN让两个人分工——一个提议，一个评分，这样更公正。

**历史背景**：由van Hasselt等人在2015年提出，是DQN最重要的改进之一。

**算法定位**：
- 类型：DQN改进 → 过估计问题
- 输出：更准确的Q值

---

## 2. 核心原理

### 2.1 过估计问题

在DQN中：
$$Q(s',a') \leftarrow \max_a Q(s',a)$$

max操作会选择"最乐观"的Q值，导致整体Q值被高估。

### 2.2 Double DQN的核心

**选择**：用*当前网络*选择最好动作
$$a^* = \arg\max_a Q(s',a; \theta)$$

**评估**：用*目标网络*评估该动作
$$Q_{target}(s,a) = r + \gamma \cdot Q(s', a^*; \theta^-)$$

这样分工明确，减少过估计！

---

## 3. 调库实现

```python
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DoubleDQNAgent:
    """Double DQN智能体"""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        # 双Q网络
        self.q_online = QNetwork(state_dim, action_dim)
        self.q_target = QNetwork(state_dim, action_dim)
        
        # 复制参数
        self.q_target.load_state_dict(self.q_online.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.replay_buffer = []
        self.buffer_size = 100000
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        
        with torch.no_grad():
            return self.q_online(torch.FloatTensor(state)).argmax().item()
    
    def store(self, s, a, r, ns, d):
        self.replay_buffer.append((s, a, r, ns, d))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        # 采样
        batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        s, a, r, ns, d = zip(*[self.replay_buffer[i] for i in batch])
        
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        ns = torch.FloatTensor(ns)
        d = torch.FloatTensor(d)
        
        # ===== Double DQN更新 =====
        # 用在线网络选择动作
        with torch.no_grad():
            next_actions = self.q_online(ns).argmax(1)
            # 用目标网络评估
            next_q = self.q_target(ns).gather(1, next_actions.unsqueeze(1)).squeeze()
            target = r + self.gamma * (1 - d) * next_q
        
        # 当前Q
        current_q = self.q_online(s).gather(1, a.unsqueeze(1)).squeeze()
        
        loss = nn.MSELoss()(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        if np.random.random() < 0.01:
            self.q_target.load_state_dict(self.q_online.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = DoubleDQNAgent(4, 2)
    
    print("=" * 50)
    print("Double DQN测试")
    print("=" * 50)
    
    for ep in range(200):
        s, _ = env.reset()
        total = 0
        
        for _ in range(500):
            a = agent.select_action(s)
            ns, r, d, t, _ = env.step(a)
            agent.store(s, a, r, ns, d)
            agent.update()
            
            s = ns
            total += r
            if d or t:
                break
        
        if ep % 50 == 0:
            print(f"回合{ep}: 奖励={total}")
```

---

## 4. 对比

| 算法 | 解决 | 过估计 | 实现复杂度 |
|------|------|------|----------|
| DQN | - | 有 | 基线 |
| Double DQN | ✓ | 缓解 | +10% |
| Dueling DQN | - | - | +网络结构 |
| Double Dueling | ✓ | ✓✓✓ | 组合 |

---

## 5. 总结

✓ Double DQN = 分工合作
✓ 选择用Online，评估用Target
✓ 简单有效的改进
✓ 减少Q值过估计