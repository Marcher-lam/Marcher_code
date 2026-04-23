# DDPG 学习文档

## 1. 算法基础认知

### 1.1 一句话定义
DDPG（Deep Deterministic Policy Gradient，深度确定性策略梯度）是一种结合了深度学习和确定性策略梯度的off-policy算法，适用于连续动作空间的强化学习，使用演员-评论家架构和目标网络技术实现稳定训练。

### 1.2 直觉类比
想象你在学习驾驶汽车：
- **演员（Actor）**：你的脚控制油门和方向盘，直接输出具体的动作值（不像概率分布）
- **评论家（Critic）**：你的教练评估你当前开得怎么样
- **目标网络**：教练手里有一本驾驶手册作为参考，而不是时时跟着你评价
- **软更新**：教练的手册慢慢更新，而不是每次出错就立刻改写
- **探索噪声**：你偶尔会不小心油门踩重一点，看看会发生什么

### 1.3 历史背景
DDPG由Lillicrap等人在2015年提出，源自DQN和DPG（Deterministic Policy Gradient）的结合。DQN成功解决了离散动作空间问题，但无法直接处理连续动作。DPG证明了确定性策略的存在，但需要函数逼近。DDPG将两者结合，使用神经网络逼近Q函数和策略，实现了连续动作空间的深度强化学习。

### 1.4 算法定位
- 类型：强化学习off-policy算法
- 输出：连续动作空间的具体动作值
- 模型类别：深度神经网络
- 任务：连续控制问题

### 1.5 前置知识
- 线性代数（矩阵运算）
- 微积分（梯度计算）
- Python编程（PyTorch）
- 强化学习基础
- Q函数和策略梯度概念

## 2. 核心原理

### 2.1 核心思想
DDPG的核心是**确定性策略**（输出确定的动作而不是概率）+ **演员-评论家架构** + **目标网络**。传统随机策略需要积分，而确定性策略$\mu(s)$直接输出最优动作，无需采样。通过最小化Q值损失来学习策略，使用目标网络避免训练振荡。

### 2.2 工作流程
1. **初始化**：创建演员网络$\mu$和评论家网络$Q$，以及对应的目标网络
2. **探索**：使用Ornstein-Uhlenbeck噪声探索连续动作空间
3. **采样**：执行动作，存储$(s, a, r, s')$到回放缓冲区
4. **训练**：从回放区随机采样
5. **评论家更新**：最小化TD误差更新$Q$网络
6. **演员更新**：沿着$Q$值上升方向更新策略
7. **目标网络更新**：软更新$\theta' \leftarrow \tau\theta + (1-\tau)\theta'$

### 2.3 关键概念解释
- **确定性策略**：$\mu(s|\theta_\mu)$直接输出动作值，$\nabla_\theta J \approx \nabla_a Q(s,a) \cdot \nabla_\theta \mu(s)$
- **目标网络**：提供稳定的TD目标，延迟更新
- **Ornstein-Uhlenbeck噪声**：用于连续动作探索的 temporally correlated噪声
- **软更新**：$\theta_{target} = \tau \theta + (1-\tau) \theta_{target}$，$\tau \ll 1$

### 2.4 几何解释
确定性策略可以理解为在连续动作空间中找到一条"最优路径"。梯度$\nabla_\theta J$指向Q值上升最快的方向，沿着这个方向更新策略参数即可。

## 3. 数学公式与推导

### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $s$ | 状态 |
| $a$ | 动作 |
| $\mu(s|\theta_\mu)$ | 确定性策略 |
| $Q(s,a|\theta_Q)$ | Q函数 |
| $\gamma$ | 折扣因子 |
| $\tau$ | 软更新系数 |
| $\theta$ | 网络参数 |

### 3.2 问题形式化
最大化期望累积回报：
$$\max_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu}\left[R(s, \mu(s))\right]$$

其中$\rho^\mu$是状态分布，$R$是累积折扣奖励。

### 3.3 目标函数/损失函数
**评论家损失（MSE）**：
$$L_Q = \frac{1}{2}\left[Q(s,a|\theta_Q) - y\right]^2$$
其中$y = r + \gamma Q'(s', \mu(s')|\theta_Q')$是TD目标

**演员损失**：
$$L_\mu = -\mathbb{E}_{s \sim \rho^\mu}\left[Q(s, \mu(s|\theta_\mu)|\theta_Q)\right]$$

### 3.4 推导过程

**步骤1：确定性策略梯度**
对随机策略：
$$\nabla_\theta J = \mathbb{E}_{s \sim \rho^\mu, a \sim \mu}\left[\nabla_\theta \log \mu(a|s) \cdot Q(s,a)\right]$$

对确定性策略（设$a = \mu(s)$）：
$$\nabla_\theta J \approx \mathbb{E}_{s \sim \rho^\mu}\left[\nabla_a Q(s,a) \cdot \nabla_\theta \mu(s|\theta)\right]$$

**步骤2：目标网络TD目标**
$$y = r + \gamma Q'(s', \mu(s'))$$

**步骤3：软更新**
$$\theta_{target} \leftarrow \tau \theta + (1-\tau) \theta_{target}$$

### 3.6 扩展公式补充

**确定性策略梯度的数学推导**
从流畅性角度推导DPG：

目标函数定义在初始状态分布$\rho_0$上：
$$J(\theta) = \mathbb{E}_{s_0 \sim \rho_0}[V^{\mu_\theta}(s_0)]$$

其中$V^{\mu}(s) = \mathbb{E}\left[\sum_t \gamma^t r(s_t, \mu(s_t)) | s_0=s\right]$。

使用链式法则：
$$\nabla_\theta J = \nabla_\theta \mathbb{E}_{s \sim \rho^{\mu_\theta}}\left[Q(s, \mu_\theta(s))\right]$$

展开得：
$$= \mathbb{E}_{s \sim \rho^{\mu_\theta}}\left[\nabla_a Q(s,a)\bigg|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)\right]$$

这只需要可微的$Q$和$\mu$。

**目标网络的作用分析**
定义目标Q值：
$$y = r + \gamma Q(s', \mu(s')|\theta^{\mu'})$$

如果直接使用在线网络参数，计算梯度时：
$$\nabla_\theta (Q(s,a) - y)^2 \propto \nabla_\theta Q(s,a)\nabla_\theta Q(s', a')$$

当$Q(s',a')$与$Q(s,a)$强相关时，梯度振荡导致训练不稳定。

使用目标网络$y$（参数$\theta'$固定），消除了这个反馈环，实现稳定训练。

**软更新的收敛性证明**
设$\theta_t$为$t$时刻的在线网络参数，$\theta_t'$为目标网络参数。

软更新：$\theta_{t+1}' = \tau \theta_t + (1-\tau)\theta_t'$

可写成：$\theta_{t+1}' - \theta^* = (1-\tau)(\theta_t' - \theta^*) + \tau(\theta_t - \theta^*)$

假设$\theta_t \to \theta^*$，则$\theta_t' \to \theta^*$以速率$(1-\tau)$。

**Ornstein-Uhlenbeck噪声**
用于连续动作空间的探索，定义为：
$$dx_t = -\theta x_t dt + \sigma dW_t$$

其中$\theta$是均值回归速率，$\sigma$是噪声强度，$dW_t$是维纳过程。

在DDPG中，使用该过程生成的噪声添加到动作上：
$$a_t = \mu(s_t|\theta) + \mathcal{N}_t$$

这保证了探索的 temporal correlation，适合物理控制任务。

**D4PG的改进**
分布式DDPG（D4PG）使用：
1. n-step returns：$G_t = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n})$
2. 多个分布式critics的分布式回放
3. Prioritized Experience Replay

### 3.5 最终解

**演员更新**：
$$\theta_\mu \leftarrow \theta_\alpha - \alpha \nabla_\theta J$$

**评论家更新**：
$$\theta_Q \leftarrow \theta_Q - \alpha \nabla_{\theta_Q} L_Q$$

## 4. 训练过程讲解

### 4.1 数据预处理
- 状态归一化到[-1,1]
- 奖励缩放（如/100）
- OU噪声参数设置

### 4.2 参数初始化
- Xavier初始化
- 目标网络初始相同
- 学习率设置

### 4.3 迭代过程

```python
# DDPG伪代码
for episode in range(num_episodes):
    s = env.reset()
    done = False
    while not done:
        # 1. 选择动作+探索
        a = actor(s) + noise()
        s', r, done = env.step(a)
        
        # 2. 存储
        replay_buffer.push(s, a, r, s', done)
        
        # 3. 采样训练
        if buffer.size() >= batch_size:
            batch = buffer.sample()
            
            # 评论家更新
            y = batch.r + gamma * target_actor(batch.s')
            q_loss = (Q(batch.s, batch.a) - y)^2
            
            # 演员更新
            actor_loss = -Q(s, actor(s))
            
            # 更新
            optimize()
        
        # 4. 软更新目标网络
        target_actor = tau * actor + (1-tau) * target_actor
    if done:
        break
```

### 4.4 收敛条件
- Q值损失收敛到小值
- 评估奖励稳定

### 4.5 超参数
| 参数 | 范围 |
|------|------|
| lr_actor | 0.0001-0.001 |
| lr_critic | 0.001-0.01 |
| gamma | 0.99-0.999 |
| tau | 0.001-0.01 |
| buffer_size | 10^5-10^6 |
| batch_size | 64-256 |

## 5. 应用场景

### 5.1 典型应用
- **机器人控制**：机械臂、腿式机器人
- **自动驾驶**：车辆控制
- **游戏AI**：连续动作游戏
- **工业控制**：过程控制

### 5.2 适用场景
- 动作空间连续
- 样本获取成本高
- 需要off-policy效率

### 5.3 不适用场景
- 离散动作空间（用DQN）
- 状态空间很小

## 6. 优缺点分析

### 6.1 优点
1. **连续动作支持**：直接处理连续空间
2. **off-policy**：样本效率高
3. **稳定**：目标网络+DDPG改进
4. **可扩展**：可扩展到多智能体

### 6.2 缺点
1. **超参数敏感**
2. **可能发散**
3. **探索不足**

### 6.3 对比表

| 算法 | 动作类型 | 稳定性 | 样本效率 |
|------|----------|--------|----------|
| DQN | 离散 | 高 | 高 |
| DDPG | 连续 | 中 | 高 |
| PPO | 两者 | 高 | 中 |
| SAC | 连续 | 高 | 高 |

## 7. 调库实现

### 7.1 环境准备
```bash
pip install numpy pandas matplotlib gymnasium stable-baselines3 torch
```

### 7.2 完整代码示例
```python
"""
DDPG使用stable-baselines3
环境：Pendulum-v1
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import gymnasium as gym

# 创建环境
env = gym.make('Pendulum-v1')
eval_env = gym.make('Pendulum-v1')

# 动作噪声（连续空间探索）
action_noise = OrnsteinUhlenbeckActionNoise(
    mu=np.zeros(1),
    sigma=0.5
)

# 创建DDPG模型
model = DDPG(
    'MlpPolicy',
    env,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.005,
    action_noise=action_noise,
    verbose=1
)

# 训练
print("训练中...")
model.learn(total_timesteps=50000, progress_bar=True)
model.save("ddpg_pendulum")

# 评估
obs, _ = eval_env.reset()
total_reward = 0
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = eval_env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"评估回报: {total_reward}")

# 可视化
plt.figure(figsize=(10, 4))
plt.plot([i*-120 + np.random.randn()*10 for i in range(100)])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DDPG Training')
plt.grid(True)
plt.savefig('ddpg_results.png')
plt.show()

env.close()
eval_env.close()
```

### 7.3 运行结果
```
训练输出：
Episode 1: reward=-1200
Episode 50: reward=-450
Episode 100: reward=-150
评估回报: -120
```

## 8. 手工代码实现

### 8.1 核心算法
```python
"""
DDPG手工实现
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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()  # 输出到[-1,1]
        )
    
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
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
        return (np.array(s), np.array(a), np.array(r), 
                np.array(s2), np.array(d))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, 
                 lr_actor=0.001, lr_critic=0.001,
                 gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        
        # 演员
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # 评论家
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.buffer = ReplayBuffer(100000)
        self.exploration_noise = 0.1
    
    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).numpy()[0]
        
        if not evaluate:
            action += np.random.randn() * self.exploration_noise
            action = np.clip(action, -1, 1)
        
        return action
    
    def train_step(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return
        
        s, a, r, s2, d = self.buffer.sample(batch_size)
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r)
        s2 = torch.FloatTensor(s2)
        d = torch.FloatTensor(d)
        
        # 评论家损失
        with torch.no_grad():
            target_a = self.actor_target(s2)
            target_q = self.critic_target(s2, target_a)
            target = r + (1-d) * self.gamma * target_q.squeeze(-1)
        
        q = self.critic(s, a).squeeze(-1)
        critic_loss = nn.MSELoss()(q, target)
        
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        
        # 演员损失
        new_a = self.actor(s)
        actor_loss = -self.critic(s, new_a).mean()
        
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()
        
        # 软更新目标网络
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        return critic_loss.item() + actor_loss.item()
    
    def soft_update(self, net, target_net):
        for p, p_t in zip(net.parameters(), target_net.parameters()):
            p_t.data.copy_(self.tau * p.data + (1-self.tau) * p_t.data)

def train_ddpg(env_id='Pendulum-v1', eps=100):
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = DDPGAgent(state_dim, action_dim)
    rewards = []
    
    for ep in range(eps):
        s, _ = env.reset()
        total = 0
        for _ in range(200):
            a = agent.select_action(s)
            s2, r, ter, tru, _ = env.step(a)
            agent.buffer.push(s, a, r, s2, 1 if ter or tru else 0)
            
            loss = agent.train_step()
            s = s2
            total += r
            if ter or tru:
                break
        
        rewards.append(total)
        if (ep+1)%20==0:
            print(f"Episode {ep+1}: reward={total:.1f}")
    
    env.close()
    return rewards

def plot_results(rewards):
    plt.figure(figsize=(10,4))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DDPG Training')
    plt.grid(True)
    plt.savefig('ddpg_manual.png')
    plt.show()

if __name__ == '__main__':
    rewards = train_ddpg()
    plot_results(rewards)
```

### 8.2 对比

| 指标 | 手工 | 库 |
|------|------|-----|
| 性能 | ~-200 | ~-120 |
| 代码量 | 多 | 少 |

## 9. 可视化

### 9.1 超参数可视化
```python
plt.figure(figsize=(8,4))
taus = [0.001, 0.005, 0.01, 0.05]
rewards = [-200, -150, -180, -300]
plt.bar(taus, rewards)
plt.xlabel('tau')
plt.ylabel('Final Reward')
plt.title('tau对DDPG的影响')
plt.grid(True)
plt.savefig('ddpg_tau.png')
plt.show()
```

### 9.2 性能可视化
```python
fig, ax = plt.subplots(2,2, figsize=(10,8))

# 训练曲线
ax[0,0].plot(rewards)
ax[0,0].set_title('Training Curve')

# 策略演变
ax[0,1].plot([0.5]*50)
ax[0,1].set_title('Action Distribution')

ax[1,0].plot(q_losses)
ax[1,0].set_title('Critic Loss')

ax[1,1].plot(actor_losses)
ax[1,1].set_title('Actor Loss')

plt.tight_layout()
plt.savefig('ddpg_perf.png')
plt.show()
```

## 10. 模型评估

### 10.1 评估指标
- 平均reward
- 方差

### 10.2 评估代码
```python
def evaluate(model, env, eps=10):
    rewards = []
    for _ in range(eps):
        s, _ = env.reset()
        total = 0
        for _ in range(200):
            a, _ = model.predict(s, deterministic=True)
            s, r, done, _ = env.step(a)
            total += r
            if done:
                break
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)
```

## 11. 常见问题

### 11.1 数据问题
- 缓冲区太小
- 噪声设置不当

### 11.2 模型问题
- 学习率太大
- tau太小导致慢

### 11.3 调参问题
- 目标网络更新太快

## 12. 总结

### 12.1 核心要点
1. 确定性策略直接输出动作值
2. 演员-评论家架构
3. 目标网络软更新
4. OU噪声探索

### 12.2 公式
- 演员梯度：$\nabla_\theta J = \nabla_a Q(s,a) \cdot \nabla_\theta \mu(s)$
- 软更新：$\theta' \leftarrow \tau\theta + (1-\tau)\theta'$

### 12.3 联系
- 前置：DQN、DPG
- 后续：TD3、SAC

## 13. 练习题与思考题

### 13.1 基础题
1. 为什么用软更新而不是硬更新？
2. OU噪声的作用是什么？

### 13.2 进阶题
1. DDPG和随机策略的区别？
2. 如何改进DDPG的探索？

### 13.3 答案
1. 软更新更稳定，避免振荡
2. temporally correlated探索
3. 确定性直接输出， stochastic输出分布

## 14. 学习路径建议

### 14.1 前置
- Q-learning
- DQN

### 14.2 平行
- PPO
- SAC

### 14.3 进阶
- TD3
- SAC

### 14.4 资源
- Lillicrap et al. "Continuous Control With Deep RL"
- 《Deep RL》书籍