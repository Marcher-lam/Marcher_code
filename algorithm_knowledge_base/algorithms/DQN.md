# DQN 学习文档

> DQN (Deep Q-Network) 是一种将深度学习与强化学习结合的算法,使用经验回放和目标网络来解决不稳定的Q学习问题,是深度强化学习的里程碑。

---

## 1. 算法基础认知

### 一句话定义
DQN 通过深度神经网络近似Q函数,结合经验回放缓冲区和目标网络,能够从高维状态空间中学习最优策略。

### 直觉类比
想象一个学生在玩电子游戏:
- **Q表**:记忆每个状态-动作对的得分
- **DQN**:像一个经验丰富的"游戏教练",能够评估在任何游戏画面(状态)下,采取某个动作的好坏

### 历史背景
- 2013年,Mnih等人在NIPS提出DQN(Atari游戏)
- 2015年,Nature论文发表
- 奠定了深度强化学习的基础

### 算法定位
- **类型**:值函数近似/深度强化学习
- **输出**:每个动作的Q值
- **模型类型**:卷积神经网络/Q网络

### 前置知识
- Q学习基础
- 神经网络训练
- 经验回放概念

---

## 2. 核心原理

### 2.1 核心思想
DQN的核心是**用深度学习近似Q函数**:

1. **神经网络近似**: $Q(s,a|\theta) \approx Q^*(s,a)$
2. **经验回放**:存储历史$(s,a,r,s')$随机采样打破时间相关性
3. **目标网络**:固定参数计算目标Q值,稳定训练

### 2.2 工作流程
```
环境 → 状态s → Q网络 → 动作a (ε-greedy)
                  ↓
              存储(s,a,r,s')
                  ↓
        随机小批量采样
                  ↓
        最小化TD误差更新网络
```

### 2.3 关键概念
- **TD误差**: $y_j - Q(s_i,a_i|\theta)$
- **目标Q值**: $y_j = r_j + \gamma \max_{a'}Q'(s'_j,a')$
- **ε-greedy**: $\epsilon$概率随机,其余贪心

### 2.4 架构图
```
┌─────────────────────────────────────┐
│          DQN 架构                   │
│  ┌─────────┐   ┌──────────────┐     │
│  │ 输入s   │→  │  CNN/MLP   │→ Q值    │
│  │(84x84)  │   │  θ         │ (actions)│
│  └─────────┘   └──────────────┘     │
│                                    │
│ ┌───────────────────────────────┐   │
│ │    经验回放缓冲区D            │   │
│ │ [s,a,r,s',done] × M         │   │
│ └───────────────────────────────┘   │
│                                    │
│ ┌───────────────────────────────┐   │
│ │    目标网络Q' (θ-)             │   │
│ │ 定期从Q复制参数               │   │
│ └───────────────────────────────┘   │
└─────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $s$ | 状态 |
| $a$ | 动作 |
| $r$ | 奖励 |
| $Q(s,a;\theta)$ | 参数化Q函数 |
| $y$ | TD目标 |
| $\theta$ | 在线网络参数 |
| $\theta^-$ | 目标网络参数 |
| $\gamma$ | 折扣因子 |

### 3.2 Q学习目标
$$L(\theta) = \mathbb{E}_{(s,a,r,s',done)}[(y - Q(s,a;\theta))^2]$$

其中目标:
$$y = r + \gamma \max_{a'} Q(s',a';\theta^-)$$

### 3.3 训练目标
$$\min_\theta \mathcal{L}_{TD} = \min_\theta \mathbb{E}[(r + \gamma \max_{a'}Q(s',a';\theta^-) - Q(s,a;\theta))^2]$$

### 3.4 推导过程

**Q函数近似**:
用神经网络 $Q(s,a;\theta)$ 逼近真实最优 $Q^*(s,a)$

**经验回放益处**:
- 打破样本间时间相关性
- 提高数据利用率

**目标网络作用**:
- 目标 $y_j$ 固定 $\theta^-$ 一段时间
- 避免训练振荡和发散

### 3.5 算法步骤

```python
# DQN 伪代码
# 1. 初始化
Q网络: Q(s,a|θ)
目标网络: Q(s,a|θ-) = Q(s,a|θ)
经验回放: D

# 2. 主循环
for episode in episodes:
    s = env.reset()
    
    for step in steps:
        # 探索
        if random < ε:
            a = random_action()
        else:
            a = argmax_a Q(s,a)
        
        # 执行
        s', r, done = env.step(a)
        D.push(s,a,r,s',done)
        
        # 更新
        if len(D) > batch_size:
            minibatch = sample(D)
            
            # 目标Q值
            y = r
            if not done:
                y += γ * max_a Q(s',a|θ-)
            
            # TD误差
            loss = (y - Q(s,a|θ))^2
            
            # 更新
            gradient_descent(loss)
        
        # 目标网络更新
        if step % C == 0:
            θ- = θ
        
        s = s'
```

---

## 4. 训练过程

### 4.1 实现代码

```python
"""
DQN 完整实现 (PyTorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), 
                np.array(reward), np.array(next_state), 
                np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 图像输入: (4,84,84) -> 扁平化
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        # 处理图像输入
        x = self.conv(state)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQN:
    """DQN算法"""
    
    def __init__(self, state_dim, action_dim, 
                 hidden_dim=256, gamma=0.99, lr=1e-4,
                 buffer_size=100000, target_update_freq=1000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        # Q网络
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 复制参数到目标网络
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 步数计数
        self.steps = 0
    
    def select_action(self, state, epsilon=0.0):
        """ε-greedy动作选择"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state).unsqueeze(0))
            return q_values.argmax(dim=-1).item()
    
    def update(self, batch_size=32):
        """更新Q网络"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # TD误差
        loss = F.mse_loss(current_q, target_q)
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 目标网络更新
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return {'loss': loss.item()}


# 训练函数
def train_dqn(env, agent, num_episodes=500, batch_size=32,
              epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=50000):
    """训练DQN"""
    rewards = []
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            # ε衰减
            epsilon = max(epsilon_end, epsilon_start - episode * (epsilon_start - epsilon_end) / epsilon_decay)
            
            # 选择动作
            action = agent.select_action(state, epsilon)
            
            # 执行
            next_state, reward, done, _ = env.step(action)
            
            # 存储
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            
            # 更新
            agent.update(batch_size)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        if episode % 50 == 0:
            avg_reward = np.mean(rewards[-50:])
            print(f"Episode {episode}: avg_reward={avg_reward:.1f}, ε={epsilon:.3f}")
    
    return rewards
```

---

## 5. 超参数

| 超参数 | 作用 | 推荐范围 |
|--------|------|----------|
| $\gamma$ | 折扣因子 | 0.99 |
| $\epsilon_{start}$ | 初始探索率 | 1.0 |
| $\epsilon_{end}$ | 最终探索率 | 0.1 |
| $\epsilon_{decay}$ | 探索衰减步数 | 50000 |
| buffer_size | 回放缓冲区大小 | 100000 |
| target_update_freq | 目标网络更新频率 | 1000 |

---

## 6. 应用场景

### 6.1 典型应用
- Atari游戏
- 机器人控制
- 资源调度

### 6.2 适用场景
- 离散动作空间
- 高维状态(图像)
- 需要样本高效

---

## 7. 优缺点

### 7.1 优点
| 优点 | 说明 |
|------|------|
| 端到端学习 | 直接从图像学习 |
| 经验回放 | 数据高效 |
| 稳定训练 | 目标网络 |

### 7.2 缺点
| 缺点 | 缓解 |
|------|------|
| 过估计 | Double DQN |
| 离散动作 | DDPG(连续) |

---

## 8. 调库实现

```python
"""
使用Stable-Baselines3
"""
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

model = DQN('CnnPolicy', 'BreakoutNoFrameskip-v4')
model.learn(total_timesteps=100000)

# 评估
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
```

---

## 9. 手工实现

```python
"""
DQN 核心简化版
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class SimpleDQN:
    """简化DQN"""
    
    def __init__(self, state_dim, action_dim, hidden=128):
        self.action_dim = action_dim
        
        # Q网络
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        
        # 目标网络
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.target.load_state_dict(self.net.state_dict())
        
        self.replay = deque(maxlen=10000)
        self.gamma = 0.99
        self.steps = 0
    
    def act(self, s, eps=0.1):
        if random.random() < eps:
            return random.randint(0, self.action_dim-1)
        with torch.no_grad():
            return self.net(torch.FloatTensor(s)).argmax().item()
    
    def push(self, *args):
        self.replay.append(args)
    
    def update(self, batch=32):
        if len(self.replay) < batch:
            return
        
        batch = random.sample(self.replay, batch)
        s, a, r, s2, done = map(np.array, zip(*batch))
        
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s2 = torch.FloatTensor(s2)
        done = torch.FloatTensor(done)
        
        # 当前Q
        q = self.net(s).gather(1, a.unsqueeze(1))
        
        # 目标Q
        with torch.no_grad():
            target_q = r + self.gamma * self.target(s2).max(1)[0] * (1-done)
        
        loss = nn.MSELoss()(q.squeeze(), target_q)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        self.steps += 1
        if self.steps % 100 == 0:
            self.target.load_state_dict(self.net.state_dict())
        
        return loss.item()
```

---

## 10. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_dqn_results(rewards, save_path='dqn.png'):
    plt.figure(figsize=(10,4))
    plt.plot(rewards, alpha=0.3)
    smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
    plt.plot(smoothed, label='MA(50)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.show()
```

---

## 11. 评估

```python
def evaluate(agent, env, n_episodes=10):
    """评估"""
    rewards = []
    for _ in range(n_episodes):
        s = env.reset()
        r = 0
        done = False
        while not done:
            a = agent.act(s, 0)
            s, reward, done, _ = env.step(a)
            r += reward
        rewards.append(r)
    return {'mean': np.mean(rewards), 'std': np.std(rewards)}
```

---

## 12. 常见问题

### 12.1 Q值过估计
- 原因: $\max$ 操作导致过估计
- 缓解: Double DQN

### 12.2 训练不稳定
- 使用目标网络
- 梯度裁剪

---

## 13. 总结

### 核心要点
1. **CNN近似Q函数**
2. **经验回放**
3. **��标网络**稳定训练
4. **ε-greedy**探索

### 算法链
```
DQN → Double DQN → Dueling DQN → Rainbow DQN
```

---

## 14. 练习题

**习题1**: TD误差定义

<details>
<summary>答案</summary>

$$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'}Q(s',a';\theta^-) - Q(s,a;\theta))^2]$$

</details>

**习题2**: 目标网络作用

<details>
<summary>答案</summary>

固定目标值,避免训练时目标不断变化导致不稳定。

</details>

---

## 15. 学习路径

- **初级**: Q学习基础,运行Atari demo
- **中级**: 理解CNN架构,调参
- **高级**: Double DQN, Rainbow

### 推荐资源
- **论文**: Mnih et al. "Playing Atari with Deep RL" (2013)
- **代码**: https://github.com/DLR-RM/stable-baselines3