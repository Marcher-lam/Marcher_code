# Experience Replay 学习文档

> 强化学习中用于提高样本效率的经验回放技术。

## 1. 算法基础认知

Experience Replay（经验回放）是强化学习中的一种核心技术，用于提高样本效率。它将智能体与环境的交互经验存储在回放缓冲区中，然后在训练时随机采样这些经验进行学习。

**直觉类比**：想象你在学习打网球。你不会每次打完球就立刻反思那一次的动作，而是把自己所有打球的经验（好的和坏的）都记录下来，然后随机抽取一些来分析动作的正确性。这就是经验回放的思想。

**历史背景**：由Lin在1992年提出，最初用于加速学习。

**前置知识**：Q-Learning、TD学习

## 2. 核心原理

核心思想：
1. 存储交互经验 (s, a, r, s', done)
2. 从缓冲区随机采样
3. 使用采样经验更新

## 3. 数学公式与推导

**回放缓冲区**：
$$D = \{e_1, e_2, ..., e_t\}$$

其中 $e_t = (s_t, a_t, r_t, s_{t+1}, done)$

**随机采样更新**：
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + gamma * max_{a'} Q(s',a') - Q(s,a)]$$

## 4. 训练过程讲解

**超参数**：
| 参数 | 作用 |
|------|------|
| buffer_size | 回放缓冲区大小 |
| batch_size | 每次采样数量 |
| learning_starts | 开始学习前的步数 |

## 5. 应用场景

- DQN
- DDPG
- 其他深度RL算法

## 6. 优缺点分析

**优点**：
1. 提高样本效率
2. 打破时间相关性
3. 支持离策略学习

**缺点**：
1. 需要额外内存
2. 增加计算量

## 7. 调库实现

```python
"""
Experience Replay实现
"""
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, buffer_size, batch_size, seed=42):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = seed
        random.seed(seed)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        """随机采样"""
        batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), 
                np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN智能体 with Experience Replay"""
    def __init__(self, state_size, action_size, buffer_size=10000, 
                 batch_size=64, gamma=0.95, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        
        self.Q = np.zeros((state_size, action_size))
        self.target_Q = np.zeros((state_size, action_size))
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.push(state, action, reward, next_state, done)
    
    def replay(self):
        """经验回放学习"""
        if len(self.buffer) < self.buffer.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        for i in range(len(states)):
            s, a, r, s_next, done = states[i], actions[i], rewards[i], next_states[i], dones[i]
            
            if done:
                target = r
            else:
                target = r + self.gamma * np.max(self.target_Q[s_next])
            
            self.Q[s, a] += self.lr * (target - self.Q[s, a])
    
    def update_target(self, tau=0.001):
        """更新目标网络"""
        self.target_Q = (1 - tau) * self.target_Q + tau * self.Q

# 测试
np.random.seed(42)
agent = DQNAgent(16, 4)

for step in range(1000):
    s = np.random.randint(16)
    a = np.random.randint(4)
    s_next = np.random.randint(16)
    r = np.random.randn()
    done = np.random.choice([True, False])
    
    agent.remember(s, a, r, s_next, done)
    
    if step > 64:
        agent.replay()

print(f"回放缓冲区大小: {len(agent.buffer)}")
print("Q值:")
print(agent.Q[:5])
```

## 8-14. 其他章节

**学习总结**：经验回放是提高样本效率的核心技术。

**核心公式**：
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'}Q(s',a') - Q(s,a)]$$

> 来源线索：本节内容根据原书中关于"experience-replay"的相关章节整理。