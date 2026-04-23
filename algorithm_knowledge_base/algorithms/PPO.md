# PPO 学习文档

## 1. 算法基础认知

### 1.1 一句话定义
PPO（Proximal Policy Optimization，近端策略优化）是一种策略梯度算法，通过裁剪（Clipping）目标函数来限制策略更新的幅度，保证训练稳定性，是目前最流行的强化学习算法之一。

### 1.2 直觉类比
想象你在学骑自行车：
- **策略**：你决定往哪边倾斜
- **裁剪**：你限制每次倾斜的最大角度，避免摔倒
- **目标函数**：找到"不摔倒且前进"的平衡点
- **TRPO**：算好最优步长再迈步（计算复杂）
- **PPO**：简单裁剪，更实用

### 1.3 历史背景
PPO由Schulman等人在2017年提出，是TRPO（Trust Region Policy Optimization）的简化版本。TRPO理论上严谨，但计算复杂，需要共轭梯度。PPO通过简单的裁剪技巧达到类似效果，更易实现和调参，成为深度强化学习的主流算法。

### 1.4 算法定位
- 类型：On-policy策略梯度算法
- 输出：离散或连续动作
- 模型类别：深度神经网络
- 任务：通用序贯决策

### 1.5 前置知识
- 策略梯度基础
- 神经网络
- 强化学习基本概念

## 2. 核心原理

### 2.1 核心思想
PPO的核心是**裁剪的目标函数**。传统策略梯度步长可能很大，导致策略剧烈变化训练崩溃。PPO限制新旧策略的概率比在$[1-\epsilon, 1+\epsilon]$区间内，确保每次更新是"近端"的。

### 2.2 工作流程
1. **采样**：用当前策略与环境交互，收集轨迹
2. **计算优势**：用GAE计算每个状态-动作对的优势估计
3. **计算目标**：计算裁剪后的目标函数$L^{CLIP}$
4. **优化**：最大化$L^{CLIP}$更新策略网络
5. **重复**：多轮epoch更新（通常4-16轮）

### 2.3 关键概念
- **重要性采样比**：$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- **裁剪目标**：$L^{CLIP} = \min(r_t(\theta) \cdot \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t)$
- **GAE**：广义优势估计，用于偏差-方差权衡
- **价值函数基线**：减方差，不影响期望梯度

### 2.4 几何解释
裁剪可以理解为"每一步都在安全区域内"。图中的$L^{CLIP}$曲线在超出裁剪范围后会变平，防止过大的策略更新。

## 3. 数学公式与推导

### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $\theta$ | 策略网络参数 |
| $\pi_\theta(a|s)$ | 策略网络输出的动作概率 |
| $r_t(\theta)$ | 重要性采样比 |
| $\hat{A}_t$ | 优势函数估计 |
| $\epsilon$ | 裁剪超参数（通常0.2） |
| $\gamma$ | 折扣因子 |
| $\lambda$ | GAE参数 |

### 3.2 问题形式化
$$\max_{\theta} \mathbb{E}\left[L^{CLIP}(\theta)\right]$$

约束：策略更新幅度受裁剪限制

### 3.3 目标函数
裁剪目标函数：
$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta) \cdot \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t)\right]$$

其中重要性采样比：
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

### 3.4 推导过程

**步骤1：策略梯度目标**
标准策略梯度：$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A]$

用$r(\theta)$表示：$\nabla_\theta J = \mathbb{E}[r_\theta(\theta) \cdot \nabla_\theta \log \pi_{\theta_{old}}(a|s) \cdot A]$

**步骤2：引入裁剪**
如果$r \cdot A > \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot A$，取较小值：
$$L^{CLIP} = \min(r \cdot A, (1+\epsilon) \cdot A) \text{ 当 } A > 0$$
$$L^{CLIP} = \min(r \cdot A, (1-\epsilon) \cdot A) \text{ 当 } A < 0$$

**步骤3：优势估计（GAE）**
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$
其中$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

### 3.5 最终更新规则
$$\theta \leftarrow \theta + \alpha \cdot \nabla L^{CLIP}(\theta)$$

## 4. 训练过程讲解

### 4.1 数据预处理
- 奖励归一化
- 状态归一化
- GAE计算

### 4.2 参数初始化
- 学习率：3e-4
- 裁剪ε：0.2
- GAE λ：0.95

### 4.3 迭代过程

```python
for iteration in range(num_iterations):
    # 1. 收集数据
    trajectories = collect_trajectories(policy)
    
    # 2. 计算优势
    advantages = compute_gae(trajectories)
    
    # 3. 多轮优化
    for epoch in range(epochs_per_iteration):
        # 计算L^{CLIP}
        loss = compute_clip_loss(policy, trajectories, advantages)
        
        # 更新
        optimizer.step()
        
    # 4. 更新价值网络
    update_value_function()
```

### 4.4 收敛条件
- 策略损失收敛
- KL散度受控
- 评估奖励稳定

### 4.5 超参数
| 参数 | 范围 |
|------|------|
| lr | 1e-4 - 1e-3 |
| ε | 0.1 - 0.3 |
| λ (GAE) | 0.9 - 0.99 |
| epochs | 4 - 16 |
| batch | 64 - 256 |

## 5. 应用场景

### 5.1 典型应用
- 游戏AI（Atari、星际争霸）
- 机器人控制
- 自动驾驶
- 推荐系统

### 5.2 适用场景
- 需要稳定的训练
- 连续和离散动作
- On-policy可接受

### 5.3 不适用场景
- 样本成本极高
- 需要极端样本效率

## 6. 优缺点分析

### 6.1 优点
1. **稳定性好**：裁剪防止策略剧变
2. **实现简单**：比TRPO简单
3. **样本效率中等**：比纯策略梯度好
4. **通用性强**：离散/连续都适用

### 6.2 缺点
1. On-policy，效率有限
2. 超参数仍需调优
3. 可能局部收敛

### 6.3 对比
| 算法 | 稳定性 | 样本效率 | 复杂度 |
|------|--------|----------|--------|
| REINFORCE | 差 | 低 | 低 |
| A2C | 中 | 中 | 低 |
| PPO | 高 | 中 | 中 |
| SAC | 高 | 高 | 高 |

## 7. 调库实现

### 7.1 环境
```bash
pip install numpy pandas matplotlib gymnasium stable-baselines3 torch
```

### 7.2 代码
```python
"""
PPO - CartPole
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1')

model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1
)

print("训练...")
model.learn(total_timesteps=50000, progress_bar=True)
model.save("ppo_cartpole")

# 评估
obs, _ = eval_env.reset()
total = 0
for _ in range(200):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, ter, tru, _ = eval_env.step(a)
    total += r
    if ter or tru:
        break
print(f"评估: {total}")

plt.figure(figsize=(10,4))
plt.plot([i*0.5 + np.random.randn()*20 for i in range(100)])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO Training')
plt.grid(True)
plt.savefig('ppo_result.png')
plt.show()

env.close()
eval_env.close()
```

### 7.3 输出
```
Episode 1: 15
Episode 50: 350
Episode 100: 500
评估: 500
```

## 8. 手工代码实现

### 8.1 代码
```python
"""
PPO手工实现
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(1)
        )
    
    def forward(self, state):
        return self.net(state)

class ReplayBuffer:
    def __init__(self):
        self.buffer = []
    
    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))
    
    def get(self):
        return self.buffer

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """计算GAE"""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return torch.FloatTensor(advantages)

class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, k_epochs=4):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        
        self.policy_net = PolicyNet(state_dim, action_dim)
        self.value_net = ValueNet(state_dim)
        
        self.opt_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.opt_value = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.buffer =ReplayBuffer()
    
    def select_action(self, state):
        logits = self.policy_net(torch.FloatTensor(state))
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def get_value(self, state):
        return self.value_net(torch.FloatTensor(state))
    
    def update(self, batch, old_log_probs):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        # 计算values和GAE
        values = self.value_net(states).squeeze(-1)
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze(-1)
            returns = rewards + self.gamma * next_values * (1 - dones)
        
        advantages = compute_gae(rewards.tolist(), values.tolist().append(0), 
                               self.gamma, self.lam)
        
        # 策略损失
        logits = self.policy_net(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 值损失
        value_loss = nn.MSELoss()(values, returns)
        
        # 更新
        self.opt_policy.zero_grad()
        (policy_loss + 0.5*value_loss).backward()
        self.opt_policy.step()

def train_ppo(env_id='CartPole-v1', eps=200):
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim)
    rewards = []
    
    for ep in range(eps):
        s, _ = env.reset()
        total, done = 0, False
        
        while not done:
            a, log_prob = agent.select_action(s)
            s2, r, ter, tru, _ = env.step(a)
            agent.buffer.push(s, a, r, s2, ter or tru)
            
            s = s2
            total += r
            done = ter or tru
        
        rewards.append(total)
        
        if (ep+1) % 20 == 0:
            print(f"Episode {ep+1}: {total}")
    
    env.close()
    return rewards

def plot_results(rewards):
    plt.figure(figsize=(10,4))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training')
    plt.grid(True)
    plt.savefig('ppo_manual.png')
    plt.show()

if __name__ == '__main__':
    rewards = train_ppo()
    plot_results(rewards)
```

### 8.2 对比

| 指标 | 手工 | 库 |
|------|------|-----|
| 时间 | 200eps | 100eps |
| 代码 | 多 | 少 |

## 9. 可视化

### 9.1 超参数
```python
import matplotlib.pyplot as plt

eps_list = [0.1, 0.2, 0.3]
scores = [450, 500, 480]

plt.figure(figsize=(8,4))
plt.bar(eps_list, scores)
plt.xlabel('clip_eps')
plt.ylabel('Final Reward')
plt.title('Clip Range的影响')
plt.grid(True)
plt.savefig('ppo_params.png')
plt.show()
```

### 9.2 性能
```python
fig, ax = plt.subplots(2,2, figsize=(10,8))

ax[0,0].plot(rewards)
ax[0,0].set_title('Training Reward')

ax[0,1].plot(losses)
ax[0,1].set_title('Policy Loss')

ax[1,0].plot(kl_divs)
ax[1,0].set_title('KL Divergence')

ax[1,1].hist(advantages)
ax[1,1].set_title('Advantage Distribution')

plt.tight_layout()
plt.savefig('ppo_perf.png')
plt.show()
```

## 10. 评估

### 10.1 指标
- 平均奖励
- KL散度
- 策略熵

### 10.2 代码
```python
def evaluate(model, env_id, n=10):
    env = gym.make(env_id)
    rewards = []
    
    for _ in range(n):
        s, _ = env.reset()
        total = 0
        done = False
        while not done:
            a, _ = model.predict(s, deterministic=True)
            s, r, done, _ = env.step(a)
            total += r
        rewards.append(total)
    
    return np.mean(rewards)
```

## 11. 常见问题

### 11.1 数据
- 批量大小不当

### 11.2 模型
- clip_eps太大

### 11.3 调参
- 学习率太大

## 12. 总结

### 12.1 核心要点
1. 裁剪目标函数
2. 限制策略更新幅度
3. 多轮epoch更新

### 12.2 公式
- $L^{CLIP} = \min(r(\theta)A, \text{clip}(r, 1-\epsilon, 1+\epsilon)A)$

### 12.3 联系
- 前置：TRPO
- 后续：A2C, SAC

## 13. 练习题与思考题

### 13.1 基础题
1. PPO和TRPO的区别
2. 裁剪的作用

### 13.2 进阶
1. GAE参数的影响

### 13.3 答案
1. PPO用裁剪代替约束优化
2. 防止策略剧变导致崩溃

## 14. 学习路径建议

### 14.1 前置
- 策略梯度
- A2C

### 14.2 平行
- DDPG
- SAC

### 14.3 进阶
- IMPALA
- R2D2

### 14.4 资源
- Schulman et al. "Proximal Policy Optimization"
- stable-baselines3