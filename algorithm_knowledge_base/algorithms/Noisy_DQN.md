# Noisy DQN 学习文档

## 1. 算法基础认知

### 1.1 定义

Noisy DQN（噪声深度 Q 网络）是由 Fortunato 等人于 2017 年提出的算法，其核心思想是：**在网络参数上添加可学习的噪声**，从而实现更高效的探索。与传统的 $\epsilon$-greedy 探索不同，Noisy DQN 将噪声作为可学习参数，通过梯度下降自动调整探索程度。

数学定义为：

$$
Q(s, a) = f(s, a; \theta) + f(s, a; \theta_{\mu}) \cdot \epsilon
$$

其中：
- $f(s, a; \theta)$：主网络参数
- $f(s, a; \theta_{\mu})$：噪声分支参数
- $\epsilon$：从噪声分布采样的随机变量

### 1.2 直观类比

将 Noisy DQN 想象为**带随机性的弹簧**：普通 DQN 像是固定的弹簧（确定性），而 Noisy DQN 像是可调节松紧度的弹簧（探索程度可学习），随着训练进行，弹簧会自动调整到合适的松紧度。

### 1.3 历史背景

- **DQN**（2013）：深度 Q 网络
- **Noisy DQN**（2017）：加入参数噪声
- **NoisyNet**（2018）：推广到其他网络

---

## 2. 核心原理

### 2.1 噪声机制

Noisy DQN 使用两种噪声分布：

1. **独立高斯噪声**：
   $$
   \epsilon_i \sim \mathcal{N}(0, 1)
   $$

2. **因式高斯噪声**：
   $$
   \epsilon_i = \sum_j \epsilon_{i,j}
   $$
   其中 $\epsilon_{i,j} \sim \mathcal{N}(0, 1)$

### 2.2 参数化噪声

线性层变为：
$$
y = (W + \sigma_W \odot \epsilon_W) x + (b + \sigma_b \odot \epsilon_b)
$$

其中 $\odot$ 表示逐元素乘法。

### 2.3 与 $\epsilon$-greedy 对比

| 方面 | $\epsilon$-greedy | Noisy DQN |
|------|-------------------|-----------|
| 探索方式 | 随机动作 | 随机参数 |
| 调整方式 | 超参数 | 梯度学习 |
| 时间尺度 | 短时 | 持续 |
| 效果 | 固定 | 自适应 |

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $Q(s,a)$ | 动作价值函数 |
| $\theta$ | 主网络参数 |
| $\theta_{\mu}$ | 噪声参数 |
| $\sigma$ | 噪声标准差 |
| $\epsilon$ | 噪声样本 |

### 3.2 Q 网络

$$
Q(s, a; \theta, \theta_{\mu}) = \text{MLP}(s, a; \theta) + \text{MLP}(s, a; \theta_{\mu}) \odot \epsilon
$$

### 3.3 目标函数

$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 \right]
$$

### 3.4 噪声梯度

噪声采样时，梯度通过采样点反向传播（使用 reparameterization trick）。

---

## 4. 训练过程讲解

### 4.1 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoisyLinear(nn.Module):
    """噪声线性层"""
    
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 主权重
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        
        # 噪声权重
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # 注册噪声缓冲区
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters(sigma_init)
        self.reset_noise()
    
    def reset_parameters(self, sigma_init):
        """初始化参数"""
        mu_range = 1 / np.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(sigma_init / np.sqrt(self.out_features))
        self.bias_sigma.data.fill_(sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声"""
        # 因式分解噪声
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        """生成噪声"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x):
        """前向传播"""
        if self.training:
            # 训练时采样噪声
            self.reset_noise()
        
        # 计算权重
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        
        return F.linear(x, weight, bias)
```

### 4.2 Noisy DQN 网络

```python
class NoisyDQN(nn.Module):
    """Noisy DQN 网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # 特征提取
        self.fc1 = NoisyLinear(state_dim, hidden_dim)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim)
        
        # Q 值输出
        self.value = NoisyLinear(hidden_dim, 1)
        
        # 优势函数（用于多动作）
        self.advantage = NoisyLinear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.value(x)
        advantage = self.advantage(x)
        
        # Dueling DQN 组合
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q
```

### 4.3 完整训练循环

```python
import torch.optim as optim
from collections import deque
import random

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

def train_noisy_dqn():
    """训练 Noisy DQN"""
    
    # 环境
    env = ...  # gym 环境
    
    # 网络
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    q_network = NoisyDQN(state_dim, action_dim)
    target_network = NoisyDQN(state_dim, action_dim)
    target_network.load_state_dict(q_network.state_dict())
    
    # 优化器
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    
    # 回放缓冲区
    buffer = ReplayBuffer(100000)
    
    # 训练
    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        
        for step in range(500):
            # 选择动作（Noisy DQN 直接从网络采样）
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()
            
            # 执行
            next_state, reward, done, _ = env.step(action)
            
            # 存储
            buffer.push(state, action, reward, next_state, done)
            
            # 训练
            if len(buffer) > 32:
                batch = buffer.sample(32)
                states, actions, rewards, next_states, dones = batch
                
                # 目标 Q 值
                with torch.no_grad():
                    next_q = target_network(next_states).max(1)[0]
                    target_q = rewards + (1 - dones) * 0.99 * next_q
                
                # 当前 Q 值
                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                # 损失
                loss = F.mse_loss(current_q, target_q)
                
                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 更新目标网络
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}")
    
    return q_network

train_noisy_dqn()
```

---

## 5. 应用场景

### 5.1 游戏 AI

Noisy DQN 的主要应用：
- Atari 游戏
- 棋类游戏
- 电子竞技

### 5.2 机器人控制

- 连续动作空间
- 探索学习

### 5.3 推荐系统

- 探索-利用平衡
- 冷启动问题

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 自适应探索 | 探索程度自动学习 |
| 端到端 | 噪声与策略联合优化 |
| 持续探索 | 始终保持探索 |
| 简单实现 | 不需要复杂架构 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 收敛慢 | 噪声增加方差 |
| 调试难 | 探索行为不直观 |
| 超参数 | 噪声规模需调 |

---

## 7. 调库实现

### 7.1 使用现有库

```python
# 使用 ptan（PyTorch 强化学习库）
import ptan
import torch

def use_ptan():
    """使用 ptan 库"""
    
    # Noisy DQN
    dqn = ptan.agent.DQNAgent(
        net=NoisyDQN(state_dim, action_dim),
        optimizer=optimizer,
        action_selector=ptan.actions.EpsilonGreedyActionSelector(epsilon=0.1)
    )
    
    print("Noisy DQN 创建成功")

use_ptan()
```

### 7.2 完整示例

```python
def complete_example():
    """完整示例"""
    
    import gym
    
    # 环境
    env = gym.make('CartPole-v1')
    
    # 状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 网络
    net = NoisyDQN(state_dim, action_dim, hidden_dim=128)
    target_net = NoisyDQN(state_dim, action_dim, hidden_dim=128)
    target_net.load_state_dict(net.state_dict())
    
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 训练（简化版）
    print("环境就绪")
    
    return net, optimizer

complete_example()
```

---

## 8. 手工代码实现

### 8.1 完整 Noisy DQN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualNoisyDQN:
    """手动实现 Noisy DQN"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, sigma_init=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 主网络参数
        self.W1_mu = torch.randn(hidden_dim, state_dim) * np.sqrt(2/state_dim)
        self.b1_mu = torch.zeros(hidden_dim)
        
        self.W2_mu = torch.randn(action_dim, hidden_dim) * np.sqrt(2/hidden_dim)
        self.b2_mu = torch.zeros(action_dim)
        
        # 噪声参数
        self.W1_sigma = torch.full((hidden_dim, state_dim), sigma_init/np.sqrt(hidden_dim))
        self.b1_sigma = torch.full((hidden_dim,), sigma_init/np.sqrt(hidden_dim))
        
        self.W2_sigma = torch.full((action_dim, hidden_dim), sigma_init/np.sqrt(action_dim))
        self.b2_sigma = torch.full((action_dim,), sigma_init/np.sqrt(action_dim))
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.get_params(), lr=0.001)
        
        # 噪声缓冲区
        self.eps_W1 = torch.randn_like(self.W1_mu)
        self.eps_b1 = torch.randn_like(self.b1_mu)
        self.eps_W2 = torch.randn_like(self.W2_mu)
        self.eps_b2 = torch.randn_like(self.b2_mu)
    
    def get_params(self):
        """获取所有参数"""
        return [self.W1_mu, self.b1_mu, self.W2_mu, self.b2_mu,
                self.W1_sigma, self.b1_sigma, self.W2_sigma, self.b2_sigma]
    
    def reset_noise(self):
        """重置噪声"""
        self.eps_W1 = torch.randn_like(self.W1_mu)
        self.eps_b1 = torch.randn_like(self.b1_mu)
        self.eps_W2 = torch.randn_like(self.W2_mu)
        self.eps_b2 = torch.randn_like(self.b2_mu)
    
    def forward(self, x):
        """前向传播"""
        # 噪声权重
        W1 = self.W1_mu + self.W1_sigma * self.eps_W1
        b1 = self.b1_mu + self.b1_sigma * self.eps_b1
        
        W2 = self.W2_mu + self.W2_sigma * self.eps_W2
        b2 = self.b2_mu + self.b2_sigma * self.eps_b2
        
        # 前向
        h = F.relu(x @ W1.t() + b1)
        q = h @ W2.t() + b2
        
        return q
    
    def act(self, state):
        """选择动作"""
        with torch.no_grad():
            q = self.forward(state)
            return q.argmax().item()
    
    def update(self, batch):
        """更新网络"""
        states, actions, rewards, next_states, dones = batch
        
        # 当前 Q
        q = self.forward(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 目标 Q
        with torch.no_grad():
            next_q = self.forward(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q
        
        # 损失
        loss = F.mse_loss(q, target_q)
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 重置噪声
        self.reset_noise()
        
        return loss.item()

# 测试
dqn = ManualNoisyDQN(state_dim=4, action_dim=2)
x = torch.randn(1, 4)
q = dqn.forward(x)
print(f"Q 值: {q}")
```

### 8.2 验证实现

```python
def verify_implementation():
    """验证实现"""
    
    # PyTorch 版本
    net = NoisyDQN(4, 2, hidden_dim=64)
    x = torch.randn(1, 4)
    
    # 多次前向
    outputs = []
    for _ in range(5):
        net.reset_noise() if hasattr(net, 'reset_noise') else None
        with torch.no_grad():
            outputs.append(net(x).numpy())
    
    # 检查多样性
    std = np.std(outputs)
    print(f"输出标准差: {std:.4f}")
    print(f"不同前向有不同输出: {std > 0}")

verify_implementation()
```

---

## 9. 可视化与结果理解

### 9.1 探索曲线

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_exploration():
    """可视化探索"""
    
    # 模拟探索奖励
    episodes = np.arange(200)
    rewards = 10 * (1 - np.exp(-episodes/50)) + np.random.randn(200) * 5
    
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, rewards, 'b-', alpha=0.6)
    plt.plot(episodes, np.convolve(rewards, np.ones(10)/10, mode='valid'), 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Exploration Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('exploration.png', dpi=150)
    plt.show()

plot_exploration()
```

### 9.2 Q 值分布

```python
def plot_q_distribution():
    """Q 值分布"""
    
    q_values = np.random.randn(1000)
    
    plt.figure(figsize=(8, 4))
    plt.hist(q_values, bins=30, edgecolor='black')
    plt.xlabel('Q Value')
    plt.ylabel('Count')
    plt.title('Q Value Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig('q_dist.png', dpi=150)
    plt.show()

plot_q_distribution()
```

---

## 10. 模型评估

### 10.1 评估指标

```python
def evaluate_noisy_dqn():
    """评估 Noisy DQN"""
    
    rewards = [10, 15, 20, 25, 30, 50, 80, 100]
    episodes = list(range(len(rewards)))
    
    print("=== Noisy DQN Evaluation ===")
    print(f"Max Reward: {max(rewards):.2f}")
    print(f"Mean Reward: {np.mean(rewards[-10:]):.2f}")
    print(f"Convergence: {rewards[-1] > rewards[0]}")
    
    return {
        'max_reward': max(rewards),
        'mean_reward': np.mean(rewards[-10:]),
        'converged': rewards[-1] > rewards[0]
    }

evaluate_noisy_dqn()
```

---

## 11. 常见问题与易错点

### 11.1 噪声规模

**问题**：噪声太大导致不稳定？

**解决**：调整 sigma_init 初始值。

### 11.2 探索-利用

**问题**：探索不足？

**解决**：增加噪声参数学习率。

---

## 12. 学习总结

### 12.1 核心要点

1. **参数噪声**：在权重上添加噪声
2. **可学习**：探索程度自动学习
3. **端到端**：与 DQN 相同训练方式
4. **替代 $\epsilon$-greedy**：更自适应的探索

### 12.2 变体

| 方法 | 特点 |
|------|------|
| NoisyNet | 推广到其他网络 |
| Noisy Prior | 贝叶斯方法 |
| RND | 随机网络蒸馏 |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：实现 NoisyLinear 层。

### 13.2 思考题

**思考题**：Noisy DQN 何时比 $\epsilon$-greedy 更好？

---

## 14. 学习路径建议

### 14.1 第一阶段

1. 理解 DQN
2. 理解噪声机制

### 14.2 第二阶段

1. 实现 NoisyLinear
2. 实现完整 DQN

### 14.3 第三阶段

1. 调参优化
2. 对比其他方法

### 14.4 推荐资源

- **论文**：《Noisy Networks for Exploration》
- **代码**：Ptan 库

---

*Noisy DQN 是一种简单而有效的探索方法，它将噪声作为可学习参数，实现了自适应的探索策略。*