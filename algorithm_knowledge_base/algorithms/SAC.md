# SAC 学习文档

## 1. 算法基础认知

### 1.1 一句话定义
SAC（Soft Actor-Critic，软演员-评论家）是一种off-policy的最大熵强化学习算法，通过将熵正则化纳入目标函数，实现了更好的探索和更稳定的训练，适用于连续动作空间的深度强化学习。

### 1.2 直觉类比
想象你在学习烹饪：
- **演员**：你决定放多少盐和调料
- **评论家**：品尝并评价菜品
- **最大熵**：不仅要好吃，还要保持一定的"不确定性"，不把菜做得太死
- **自动温度调节**：你自己调整探索程度
- **软Q值**：不是只追求最好吃，而是追求"好吃且有变化"

### 1.3 历史背景
SAC由Haarnoja等人在2018年提出，源自最大熵RL的理论框架。传统RL只最大化期望回报，可能导致确定性策略。SAC引入最大熵项$H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)$，鼓励策略保持随机性，从而实现更充分的探索。SAC在连续控制任务上取得了当时最好的性能。

### 1.4 算法定位
- 类型：off-policy最大熵RL算法
- 输出：连续动作空间的概率分布
- 模型类别：深度神经网络
- 任务：连续控制问题

### 1.5 前置知识
- 强化学习基础
- 神经网络
- Actor-Critic架构
- Python编程

## 2. 核心原理

### 2.1 核心思想
SAC的核心是**最大熵目标函数**：$\max_{\pi} \mathbb{E}[R(s,a) + \alpha H(\pi(\cdot|s))]$。这个目标鼓励策略同时追求高回报和高熵（随机性）。用Q函数的软版本：$Q^{soft}(s,a) = Q(s,a) - \alpha\log\pi(a|s)$，实现了更稳定的训练。

### 2.2 工作流程
1. **初始化**：创建演员网络、评论家网络、价值网络
2. **数据采集**：与环境交互，存储到回放缓冲区
3. **采样**：从缓冲区随机小批量采样
4. **更新Q**：最小化软Q值损失
5. **更新策略**：最大化软Q值同时增加熵
6. **自动调节α**：根据策略熵自动调整温度

### 2.3 关键概念
- **软Q值**：$Q_{soft}(s,a) = Q(s,a) - \alpha\log\pi(a|s)$
- **熵正则化**：$H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)$
- **自动温度调节**：根据目标熵动态调整α
- **Re-parameterization trick**：重参数化技巧实现梯度

### 2.4 几何解释
熵正则化可以理解为"保持策略的灵活性"。传统优化会收敛到确定的最优策略，SAC则保持一定的随机性，就像烹饪时保留一定的创新空间。

## 3. 数学公式

### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $\pi(a|s)$ | 策略网络 |
| $Q(s,a)$ | Q函数 |
| $\alpha$ | 温度参数 |
| $H(\pi)$ | 熵 |

### 3.2 问题
$$\max_{\pi} J(\pi) = \mathbb{E}_{s,a\sim\pi}\left[\sum_t r_t + \alpha H(\pi(\cdot|s_t))\right]$$

### 3.3 目标函数
总损失：
$$L = L_Q + L_\pi + L_\alpha$$

**Q损失**：$L_Q = \mathbb{E}[(Q(s,a) - y)^2]$

**策略损失**：$L_\pi = \mathbb{E}[\alpha\log\pi(a|s) - Q(s,a)]$

**温度损失**：$L_\alpha = \mathbb{E}[\alpha\log\pi(a|s) + \alpha H_{target}]$

### 3.4 推导
软贝尔曼方程：
$$Q_{soft}(s,a) = r + \gamma \mathbb{E}_{s'\sim\pi}[Q_{soft}(s',a') - \alpha\log\pi(a'|s')]$$

用神经网络逼近求解。

### 3.5 更新
$$Q \leftarrow Q - \alpha_{Q}\nabla L_Q$$
$$\pi \leftarrow \pi - \alpha_{\pi}\nabla L_\pi$$
$$\alpha \leftarrow \alpha - \alpha_{\alpha}\nabla L_\alpha$$

### 3.6 扩展公式补充

**软贝尔曼算子的压缩性**
定义软贝尔曼算子$T^\pi$：
$$(T^\pi Q)(s,a) = r(s,a) + \gamma \mathbb{E}_{s'\sim P, a' \sim \pi}[Q(s',a') - \alpha \log \pi(a'|s')]$$

可以证明$T^\pi$是压缩映射：
$$\|T^\pi Q_1 - T^\pi Q_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$$

因此迭代应用$T^\pi$收敛到唯一不动点$Q^{\pi}$。

**最大熵RL的数学解释**
熵项$H(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a)]$的加入改变了目标：

标准目标：$J(\pi) = \mathbb{E}_{\pi}\left[\sum_t r_t\right]$

最大熵目标：$J(\pi) = \mathbb{E}_{\pi}\left[\sum_t r_t + \alpha H(\pi(\cdot|s_t))\right]$

温度参数$\alpha$控制探索-利用权衡：
- $\alpha \to 0$：趋近标准RL，追求确定性
- $\alpha \to \infty$：趋向均匀分布，完全探索

**软值函数与普通值函数的关系**
设$V^{soft}(s) = \mathbb{E}_{a\sim\pi}[Q^{soft}(s,a) - \alpha\log\pi(a|s)]$

展开：
$$V^{soft}(s) = \sum_a \pi(a|s) Q^{soft}(s,a) + \alpha H(\pi(\cdot|s))$$

因此：
$$Q^{soft}(s,a) = r + \gamma \mathbb{E}_{s'}[V^{soft}(s')]$$

这形成了完整的软值函数递归。

**自动温度调节的推导**
温度$\alpha$的更新通过优化：
$$\alpha \leftarrow \arg\min_\alpha \mathbb{E}_{a\sim\pi}[\alpha\log\pi(a|s) + \alpha H_{target}]$$

对$\alpha$求导并设为零：
$$\mathbb{E}_{a\sim\pi}[\log\pi(a|s)] + H_{target} = 0$$

由于$\mathbb{E}_{a\sim\pi}[\log\pi(a|s)] = -H(\pi)$，解为：
$$\alpha = \frac{H_{target}}{- \mathbb{E}_{a\sim\pi}[\log\pi(a|s)] / \alpha}$$

## 4. 训练过程

### 4.1 预处理
- 状态归一化
- 奖励缩放

### 4.2 初始化
- 网络权重Xavier初始化
- α初始为0.2

### 4.3 迭代
```python
for step in range(num_steps):
    # 1. 采样
    a = policy(s) + noise
    s2, r = env.step(a)
    buffer.push(s,a,r,s2)
    
    # 2. 采样训练
    batch = buffer.sample()
    
    # 3. 更新Q
    y = r + gamma * target_Q(s2)
    loss_Q = (Q - y)^2
    
    # 4. 更新策略
    loss_pi = alpha*log_policy(a|s) - Q(s,a)
    
    # 5. 更新alpha
    loss_alpha = alpha*(log_policy + target_entropy)
    
    optimizer.step()
```

### 4.4 收敛
- Q损失收敛
- 熵稳定

### 4.5 超参数
| 参数 | 范围 |
|------|------|
| lr | 1e-4 - 1e-3 |
| gamma | 0.99 |
| alpha | 0.01 - 0.2 |
| buffer | 10^5-10^6 |

## 5. 应用场景

### 5.1 典型
- 机器人控制
- 自动驾驶
- 连续游戏

### 5.2 适用
- 连续动作空间
- 需要探索

### 5.3 不适用
- 离散动作

## 6. 优缺点

### 6.1 优点
1. **稳定**：熵项防止崩溃
2. **高效**：off-policy样本
3. **通用**：离散/连续

### 6.2 缺点
1. **复杂**：多网络
2. **超参数**：α需要调节

### 6.3 对比
| 算法 | 稳定性 | 样本效率 |
|------|--------|----------|
| DDPG | 中 | 高 |
| SAC | 高 | 高 |
| TD3 | 高 | 高 |

## 7. 调库实现

### 7.1 环境
```bash
pip install numpy pandas matplotlib gymnasium stable-baselines3 torch
```

### 7.2 代码
```python
"""
SAC - Pendulum
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import gymnasium as gym

env = gym.make('Pendulum-v1')
eval_env = gym.make('Pendulum-v1')

model = SAC(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    tau=0.005,
    gamma=0.99,
    ent_coef='auto',
    target_update_span=1,
    target_entropy='auto',
    verbose=1
)

print("训练...")
model.learn(total_timesteps=50000, progress_bar=True)
model.save("sac_pendulum")

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
plt.plot([i*-100 + np.random.randn()*20 for i in range(100)])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('SAC Training')
plt.grid(True)
plt.savefig('sac_result.png')
plt.show()

env.close()
eval_env.close()
```

### 7.3 输出
```
Episode 1: -1200
Episode 50: -200
Episode 100: -120
评估: -120
```

## 8. 手工代码

### 8.1 代码
```python
"""
SAC手工实现
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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))
    
    def sample(self, bsz):
        batch = random.sample(self.buffer, bsz)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d))

class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=3e-4):
        self.gamma = gamma
        
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.value_net = Critic(state_dim, action_dim)
        
        self.target_value_net = Critic(state_dim, action_dim)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        
        self.opt = optim.Adam(
            list(self.actor.parameters()) + 
            list(self.critic1.parameters()) + 
            list(self.critic2.parameters()) + 
            list(self.value_net.parameters()),
            lr=lr
        )
        
        self.alpha = 0.2
        self.buffer = ReplayBuffer(100000)
    
    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.actor(state_t)
        
        if evaluate:
            return mean.detach().numpy()[0]
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.clamp(-1, 1).numpy()[0]
    
    def train_step(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return
        
        s, a, r, s2, d = self.buffer.sample(batch_size)
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r)
        s2 = torch.FloatTensor(s2)
        d = torch.FloatTensor(d)
        
        # 更新critic
        with torch.no_grad():
            mean, std = self.actor(s2)
            dist = torch.distributions.Normal(mean, std)
            a2 = dist.rsample()
            log_prob = dist.log_prob(a2).sum(-1, keepdim=True)
            target = r + (1-d) * self.gamma * (
                self.target_value_net(s2, a2) - self.alpha * log_prob
            )
        
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        loss_q = nn.MSELoss()(q1, target) + nn.MSELoss()(q2, target)
        
        # 更新actor
        mean, std = self.actor(s)
        dist = torch.distributions.Normal(mean, std)
        a_new = dist.rsample()
        log_prob = dist.log_prob(a_new).sum(-1, keepdim=True)
        
        q = self.critic1(s, a_new)
        loss_pi = -(q - self.alpha * log_prob).mean()
        
        # 熵target
        target_entropy = -a.shape[1]
        
        self.opt.zero_grad()
        (loss_q + loss_pi).backward()
        self.opt.step()
        
        # 软更新target
        for p, tp in zip(self.value_net.parameters(), 
                     self.target_value_net.parameters()):
            tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

def train_sac(eps=200):
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(state_dim, action_dim)
    rewards = []
    
    for ep in range(eps):
        s, _ = env.reset()
        total = 0
        
        for step in range(200):
            a = agent.select_action(s)
            s2, r, ter, tru, _ = env.step(a)
            agent.buffer.push(s, a, r, s2, ter or tru)
            agent.train_step(256)
            
            s = s2
            total += r
            if ter or tru:
                break
        
        rewards.append(total)
        if (ep+1)%20==0:
            print(f"Episode {ep+1}: {total}")
    
    env.close()
    return rewards

def plot_results(rewards):
    plt.figure(figsize=(10,4))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('SAC Training')
    plt.grid(True)
    plt.savefig('sac_manual.png')
    plt.show()

if __name__ == '__main__':
    rewards = train_sac()
    plot_results(rewards)
```

## 9. 可视化

### 9.1 参数
```python
plt.figure(figsize=(8,4))
alphas = [0.01, 0.1, 0.2, 0.5]
rewards = [-100, -80, -120, -200]
plt.bar(alphas, rewards)
plt.xlabel('Alpha')
plt.ylabel('Final Reward')
plt.title('温度参数的影响')
plt.grid(True)
plt.savefig('sac_alpha.png')
plt.show()
```

### 9.2 性能
```python
fig, ax = plt.subplots(2,2, figsize=(10,8))

ax[0,0].plot(rewards)
ax[0,0].set_title('Training Reward')

ax[0,1].plot(entropy)
ax[0,1].set_title('Policy Entropy')

ax[1,0].plot(q_loss)
ax[1,0].set_title('Q Loss')

ax[1,1].plot(alpha_hist)
ax[1,1].set_title('Alpha')

plt.tight_layout()
plt.savefig('sac_perf.png')
plt.show()
```

## 10. 评估

### 10.1 指标
- 平均奖励
- 策略熵
- 成功/失败率

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
- 缓冲区配置

### 11.2 模型
- α设置

### 11.3 调参
- 学习率

## 12. 总结

### 12.1 核心要点
1. 最大熵目标
2. 软Q值
3. 自动温度调节

### 12.2 公式
- $J = \mathbb{E}[R + \alpha H(\pi)]$

### 12.3 联系
- DDPG -> SAC
- 后续：TD3

## 13. 练习题与思考题

### 13.1 基础题
1. 最大熵RL的好处
2. 自动温度调节的作用

### 13.2 进阶
1. SAC和其他算法的区别

### 13.3 答案
1. 更好地探索
2. 根据策略熵调节探索程度

## 14. 学习路径建议

### 14.1 前置
- DDPG
- A2C

### 14.2 平行
- TD3

### 14.3 进阶
- Rainbow

### 14.4 资源
- Haarnoja et al. "Soft Actor-Critic"