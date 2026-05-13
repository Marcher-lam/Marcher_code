# PER（优先经验回放）学习文档

> 强化学习中基于优先级的经验回放技术，让重要经验被更多学习

---

## 1. 算法基础认知

**一句话定义**：PER（Prioritized Experience Replay，优先经验回放）是由Schaul等人在2015年提出的强化学习技术，通过让TD误差大的经验（"意外"的经验）有更高的概率被回放，加速学习过程。

**直觉类比**：PER就像一个"智能错题本"。传统经验回放就像把所有做过的题目（经验）都放在一个盒子里，每次复习时随机抽一道。但有时候你最需要复习的是那些错得很离谱的题（TD误差大），因为这些题暴露了你的知识盲区。PER就是按"错题程度"排序，优先复习那些错得最离谱的题目（TD误差大的经验），这样学习效率大大提高。

**历史背景**：
- 2015年，Schaul等人在论文"Prioritized Experience Replay"中提出
- 成为DQN系列算法的标配
- 后续发展出SumTree、Hindsight Experience Replay等

**算法定位**：
- 类型：强化学习 → 经验回放
- 作用：加速学习、提高样本效率
- 模型类型：优先级采样

**前置知识**：
- [必备]：强化学习基础（DQN）
- [必备]：经验回放（Experience Replay）
- [推荐]：TD学习

---

## 2. 核心原理

### 2.1 传统经验回放的问题

传统经验回放（Experience Replay）：

```python
# 随机采样
for i in range(batch_size):
    experience = random.sample(replay_buffer)
    learn(experience)
```

**问题**：随机采样意味着所有经验被选中概率相同，即使是"不重要"的经验也会被重复学习，而重要的经验（误差大的）可能被忽略。

### 2.2 PER的核心思想

**核心洞察**：TD误差越大的经验越值得学习！

| 经验类型 | TD误差 | 优先级 |
|----------|--------|--------|
| 意外的结果（大惊喜/大失望） | 高 | 高 |
| 预期内的结果 | 低 | 低 |

**优先级定义**：

$$p_i = |TD_i|^\alpha + \epsilon$$

其中 $\alpha$ 控制优先级程度（$\alpha=0$ 等价于均匀采样）。

### 2.3 整体流程

```
              经验缓冲区
                │
        ┌───────┴───────┐
        ▼             ▼
    计算TD误差      按优先级排序
        │             │
        └───────┬───────┘
                ▼
    优先级采样（高TD→高概率）
                │
                ▼
         DQN更新
```

---

## 3. 数学公式与推导

### 3.1 TD误差

**TD误差定义**：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

在深度Q学习中：

$$TD_i = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

### 3.2 优先级

**抽样概率**（Softmax形式）：

$$P(i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha}$$

### 3.3 重要性采样权重

由于优先级采样改变了采样分布，需要用重要性采样（IS）校正：

$$w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta$$

其中 $\beta$ 控制校正程度（$\beta=1$ 完全校正）。

**最终的梯度权重**：

$$w_i = \frac{w_i}{\max_j w_j}$$  # 归一化

### 3.4 优先级更新

每次学习后更新优先级：

$$p_i = |TD_i| + \epsilon$$

---

## 4. 训练过程讲解

### 4.1 训练流程

```
       初始化经验池
           │
           ▼
    ┌───────────────┐
    │ 采样批量    │ ← 优先级采样
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 计算TD误差  │
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 重要性加权  │
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ DQN更新   │ ← 加权梯度
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 更新优先级 │
    └───────────────┘
           │
           └───→ 循环
```

### 4.2 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| alpha | 0.6 | 优先级程度 |
| beta | 0.4 | IS校正程度 |
| epsilon | 1e-6 | 防止零优先级 |
| capacity | 100000 | 经验池大小 |

### 4.3 实现细节

| 技巧 | 说明 |
|------|------|
| SumTree | 高效优先级采样 |
| 均匀起始 | beta从低到高 |
| 截断 | 截断最大权重 |

---

## 5. 应用场景

### 5.1 Atari游戏

```python
# Atari游戏
# PER + DQN
# 性能提升明显
```

### 5.2 连续控制

```python
# MuJoCo环境
# PER + SAC/PPO
```

### 5.3 机器人学习

```python
# 真实机器人
# 样本珍贵，需要PER提高效率
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **加速学习** | 优先学习重要经验 |
| **样本效率** | 减少所需的回合数 |
| **稳定收敛** | 减少方差 |
| **简单实现** | 只需改采样 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **偏差** | 需要IS校正 |
| **额外计算** | 优先级排序 |
| **敏感** | alpha/beta需要调 |
| **不适用于Offline** | 需要仔细校正 |

### 6.3 改进方案

| 改进 | 方法 |
|------|------|
| Rank-based PER | 用排名代替绝对值 |
| 分布式PER | 多线程采样 |
| HER | Hindsight Experience Replay |

---

## 7. 调库实现

### 7.1 多种库实现

```python
# 1. Tianshou
from tianshou.experience import PrioritizedReplayBuffer

buffer = PriorizedReplayBuffer(
    size=100000,
    alpha=0.6,
    beta=0.4
)

# 2. 手动实现
import numpy as np
import random


class SumTree:
    """SumTree实现优先级采样"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0
        
    def _propagate(self, idx, change):
        """向上传播变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx, s):
        """向下查找"""
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])
    
    def update(self, idx, value):
        """更新优先级"""
        change = value - self.tree[idx]
        self.tree[idx] = value
        self._propagate(idx, change)
        
    def add(self, priority, data):
        """添加经验"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def sample(self, batch_size):
        """采样"""
        indices = []
        batch = []
        
        for _ in range(batch_size):
            s = random.uniform(0, self.tree[0])
            idx = self._retrieve(0, s)
            batch.append(self.data[self.write])
            indices.append(idx)
            
        return indices, batch


class PEReplayBuffer:
    """PER回放缓冲区"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        self.buffer = []
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        # 计算优先级（初始用最大）
        max_priority = 1.0 if len(self.buffer) > 0 else max_priority
        
        experience = (state, action, reward, next_state, done)
        
        # 添加到树
        self.tree.add(max_priority ** self.alpha, experience)
        self.buffer.append(experience)
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """按优先级采样"""
        indices, batch = self.tree.sample(batch_size)
        
        # 计算IS权重
        priorities = np.array([self.tree.tree[i + self.capacity - 1] 
                         for i in indices])
        probs = priorities / self.tree.tree[0]
        weights = (1 / (len(self.buffer) * probs)) ** self.beta
        weights = weights / weights.max()
        
        # 解包经验
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return (states, actions, rewards, next_states, dones, 
                weights, indices)
    
    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)


def demo_per():
    """演示PER"""
    import gym
    
    buffer = PEReplayBuffer(10000, alpha=0.6, beta=0.4)
    env = gym.make('CartPole-v1')
    
    # 收集经验
    for episode in range(100):
        state = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            
            buffer.add(state, action, reward, next_state, done)
            state = next_state
    
    # 采样
    if len(buffer.buffer) >= 32:
        batch = buffer.sample(32)
        print(f"采样成功: {len(batch)} 个样本")


if __name__ == "__main__":
    demo_per()
```

---

## 8. 手工代码实现

### 8.1 完整PER+DQN实现

```python
import numpy as np
import random
import gym
from collections import deque


class SumTree:
    """SumTree优先级采样"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n = 0
        
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])
    
    def update(self, idx, value):
        change = value - self.tree[idx]
        self.tree[idx] = value
        self._propagate(idx, change)
        
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n < self.capacity:
            self.n += 1
            
    def sample(self, batch_size):
        batch = []
        indices = []
        
        for _ in range(batch_size):
            s = random.uniform(0, self.tree[0])
            idx = self._retrieve(0, s)
            batch.append(self.data[idx - self.capacity + 1])
            indices.append(idx)
            
        return indices, batch


class PrioritizedReplay:
    """优先经验回放"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        
        self.tree = SumTree(capacity)
        self.buffer = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.tree.write] = experience
            
        self.tree.add(self.max_priority ** self.alpha, experience)
        
    def sample(self, batch_size):
        indices, batch = self.tree.sample(batch_size)
        
        # IS权重
        priorities = [self.tree.tree[i] for i in indices]
        probs = np.array(priorities) / self.tree.tree[0]
        weights = (1 / (len(self.buffer) * probs)) ** self.beta
        weights = weights / (weights.max() + 1e-8)
        
        # 转换
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class DQNPER:
    """DQN with PER"""
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.replay = PrioritizedReplay(100000, alpha=0.6, beta=0.4)
        
        self.q_net = self._build_net()
        self.target_net = self._build_net()
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = None  # 需导入torch
        self.gamma = 0.99
        
    def _build_net(self):
        # 简化：需要实际PyTorch网络
        return None
        
    def store(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)
        
    def train_step(self, batch_size=32):
        if len(self.replay.buffer) < batch_size:
            return
            
        states, actions, rewards, ns, dones, weights, indices = self.replay.sample(batch_size)
        
        # 简化的DQN更新
        # 实际需要对torch计算TD误差
        td_errors = np.random.randn(batch_size)
        
        # 更新优先级
        self.replay.update_priorities(indices, td_errors)


# 使用示例
def demo():
    """演示"""
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agentDQN = DQNPER(state_dim, action_dim)
    
    print("PER初始化完成")
    
    for episode in range(100):
        state = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            ns, reward, done, _ = env.step(action)
            agent.store(state, action, reward, ns, done)
            state = ns
        
        if episode % 10 == 0:
            print(f"Episode {episode}: 缓存 {len(agent.replay.buffer)}")


if __name__ == "__main__":
    demo()
```

---

## 9. 可视化与结果理解

### 9.1 采样分布可视化

```python
def visualize_priorities():
    """可视化优先级分布"""
    
    priorities = ...
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(priorities)), priorities)
    plt.xlabel('Experience Index')
    plt.ylabel('Priority')
    plt.title('Priority Distribution')
    plt.show()
```

---

## 10. 模型评估

### 10.1 性能对比

| 方法 | 样本效率 | 最终性能 |
|------|---------|---------|
| DQN | 1x | 基准 |
| DQN+PER | 1.5-2x | 相近或更好 |

### 10.2 超参数影响

| alpha | 效果 |
|--------|------|
| 0 | 退化为随机采样 |
| 0.6 | 推荐值 |
| 1 | 过度专注高TD |

---

## 11. 常见问题与易错点

### 11.1 偏差

问题：优先级采样引入偏差

解决：使用IS校正（beta参数）

### 11.2 初始化

问题：初始优先级低，采样不到

解决：初始用最大优先级

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | 优先级采样 |
| 关键 | TD误差→优先级 |
| 校正 | IS权重 |
| 优势 | 样本效率 |

### 12.2 扩展

- Rank-based PER
- Distributed PER
- HER

---

## 13. 练习题

### 13.1 基础

1. 为什么要优先学习TD误差大的经验？
2. IS校正的作用？

### 13.2 进阶

1. PER和普通回放的区别？
2. alpha=0 vs alpha=1的区别？

---

## 14. 学习路径

1. 强化学习基础
2. 经验回放
3. PER原理
4. 实现与调参

---

## 附录

### 参考

- 论文：Schaul et al., 2015
- 库：Tianshou, baselines

---

**文档结束**