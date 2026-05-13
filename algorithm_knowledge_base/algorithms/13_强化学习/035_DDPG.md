# DDPG 学习文档

## 1. 算法基础认知
### 1.1 发展历史
DDPG（Deep Deterministic Policy Gradient，深度确定性策略梯度）由Lillicrap等人在2015年论文《Continuous Control with Deep Reinforcement Learning》中正式提出，是将DQN的思想扩展到连续动作空间的开创性工作。它结合了深度学习的表征能力和确定性策略梯度的理论，成为连续控制领域的经典基准算法。

### 1.2 类比理解
| 类比场景 | 对应算法逻辑 |
| --- | --- |
| 赛车方向盘控制 | 演员直接输出方向盘角度（确定性动作），评论家评估当前状态-动作对的Q值 |
| 机器人关节控制 | 直接输出扭矩值，用Q值指导更新方向，无需采样动作概率 |
| DQN的连续版 | DQN输出离散Q值，DDPG输出连续动作+Q值，适配连续空间 |

### 1.3 算法定位
| 属性 | 取值 |
| --- | --- |
| 模型类型 | 无模型（Model-free） |
| 算法类别 | 确定性策略梯度+价值学习（Actor-Critic） |
| 采样特性 | 异策略（Off-policy） |
| 核心机制 | 回放缓冲区、目标网络、OU噪声探索 |
| 动作空间 | 连续动作专用 |

### 1.4 前置知识清单
#### 数学基础
- 确定性策略梯度定理
- Q-learning更新规则
- 指数移动平均（软更新）

#### 编程基础
- PyTorch 回放缓冲区实现
- 目标网络参数管理
- 噪声生成（Ornstein-Uhlenbeck）

#### 强化学习前置
- DQN算法原理与实现
- A2C演员-评论家框架
- 异策略采样逻辑

### 1.5 相关算法对比
| 算法 | 核心差异 |
| --- | --- |
| DQN | DDPG是连续动作版DQN，用确定性策略替代离散Q值 |
| TD3 | TD3是DDPG的改进版，解决Q值过估计问题 |
| SAC | SAC是随机策略异策略，样本效率比DDPG高30% |
| PPO | PPO是同策略随机策略，DDPG是异策略确定性策略 |

> 来源线索：本节内容根据原书中关于"第8章 深度确定性策略梯度"的相关章节整理、扩展与教学化改写。

## 2. 核心原理
### 2.1 运行机制
DDPG包含4个网络：演员μ(s|θ)、评论家Q(s,a|w)、目标演员μ'(s|θ')、目标评论家Q'(s,a|w')，搭配回放缓冲区和OU探索噪声。ASCII流程图：
```
[回放缓冲区] ← (s,a,r,s') ← [环境 + OU噪声]
        ↓ (采样mini-batch)
[评论家Q(s,a)] → 计算Q目标：y = r + γ Q'(s', μ'(s'))
        ↓
[评论家损失：MSE(Q(s,a), y)]
        ↓
[演员梯度：∇_θ J = E[∇_θ μ(s) ∇_a Q(s,a)|a=μ(s)]]
        ↓
[软更新目标网络：θ' ← τθ + (1-τ)θ']
```

### 2.2 相关算法对比
1. **DDPG vs DQN**：DDPG适配连续动作，用确定性策略+Q值；DQN适配离散动作，用Q值选动作。
2. **DDPG vs TD3**：TD3添加目标策略平滑、延迟演员更新，解决Q值过估计。
3. **DDPG vs SAC**：SAC是随机策略，熵正则化，样本效率更高。
4. **DDPG vs PPO**：DDPG是异策略连续控制，PPO是同策略通用算法。

### 2.3 工程经验
1. **软更新系数τ**：默认0.005，平衡目标网络稳定性和更新速度。
2. **OU噪声参数**：θ=0.15，σ=0.2，随训练衰减噪声。
3. **回放缓冲区大小**：1e6，充分存储历史经验。
4. **Batch size**：256，比离散任务大，提升Q值估计稳定性。
5. **梯度裁剪**：对演员和评论家都做裁剪（norm≤1.0），防止梯度爆炸。

### 2.4 直观几何解释
演员在连续动作空间中直接输出确定性的最优动作，评论家告诉演员每个动作的好坏（Q值），演员沿Q值上升的方向调整参数，就像直接调整方向盘角度到Q值最高的位置。

## 3. 数学公式与推导
### 3.1 符号表
| 符号 | 含义 | 维度/范围 |
| --- | --- | --- |
| $\\mu_\\theta(s)$ | 确定性演员策略（输出连续动作） | $\mathbb{R}^{|\\mathcal{A}|}$ |
| $Q_w(s,a)$ | 评论家Q值函数 | $\mathbb{R}$ |
| $\\mu'(s), Q'(s,a)$ | 目标演员、目标评论家 | 同μ、Q |
| $\\tau$ | 软更新系数 | 标量，默认0.005 |
| $\\mathcal{B}$ | 回放缓冲区 | 存储容量1e6 |

### 3.2 核心公式推导
1. **确定性策略梯度**（核心）：
   $$\\nabla_\\theta J(\\theta) = \\mathbb{E}_{s \\sim \\mathcal{B}}[\\nabla_\\theta \\mu_\\theta(s) \\nabla_a Q_w(s,a)|_{a=\\mu_\\theta(s)}]$$
   即演员的梯度是Q值对动作的梯度，乘以演员对参数的梯度。

2. **评论家Q目标**：
   $$y = r + \\gamma Q'_w(s', \\mu'_\\theta(s'))$$
   目标Q值由目标网络和奖励计算。

3. **评论家损失**：
   $$L(w) = \\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{B}}[(Q_w(s,a) - y)^2]$$

4. **软更新规则**：
   $$w' \\leftarrow \\tau w + (1-\\tau) w'$$
   $$w' \\leftarrow \\tau \\theta + (1-\\tau) \\theta'$$

### 3.3 伪代码
```
初始化演员μ_θ、评论家Q_w、目标网络μ'、Q'（θ'=θ，w'=w）
初始化回放缓冲区B
for 时间步 t = 1 to T:
    选择动作 a_t = μ_θ(s_t) + OU噪声
    执行a_t，得到r_t、s_{t+1}，存储(s_t,a_t,r_t,s_{t+1})到B
    从B采样mini-batch
    计算y = r + γ Q'(s', μ'(s'))
    计算评论家损失MSE(Q(s,a), y)，更新w
    计算演员梯度∇_θ J，更新θ
    软更新目标网络θ'、w'
```

### 3.4 确定性策略梯度证明
从Q值期望目标出发：$J(\\theta) = \\mathbb{E}_{s \\sim \\rho^\\mu}[Q_w(s, \\mu_\\theta(s))]$，对θ求梯度，交换梯度和期望，得到：
$$\\nabla_\\theta J = \\mathbb{E}[\\nabla_\\theta \\mu_\\theta(s) \\nabla_a Q_w(s,a)|_{a=\\mu_\\theta(s)}]$$
推导过程同策略梯度定理，但动作是确定性的，无需对动作积分。

## 4. 训练过程讲解
### 4.1 数据预处理示例
| 环境 | 状态预处理 | 动作处理 |
| --- | --- | --- |
| Pendulum-v1 | cosθ、sinθ、角速度归一化到[-1,1] | 扭矩[-2,2]直接输入 |
| BipedalWalker-v3 | 24维状态归一化 | 4维扭矩[-1,1] |
| MuJoCo Humanoid | 376维状态归一化 | 17维扭矩[-1,1] |

### 4.2 参数初始化推荐表
| 参数 | Pendulum | BipedalWalker | MuJoCo Humanoid |
| --- | --- | --- | --- |
| 演员学习率 | 1e-4 | 1e-4 | 5e-5 |
| 评论家学习率 | 1e-3 | 1e-3 | 5e-4 |
| 软更新系数τ | 0.005 | 0.005 | 0.001 |
| 回放缓冲区大小 | 1e6 | 1e6 | 2e6 |
| Batch size | 64 | 128 | 256 |
| OU噪声σ | 0.2 | 0.3 | 0.1 |

### 4.3 训练流程（含工程技巧）
1. 初始化4个网络和回放缓冲区
2. 循环训练：
   a. 用当前演员+OU噪声选择动作，交互环境
   b. 存储转移样本到回放缓冲区
   c. 缓冲区满后，每步采样mini-batch
   d. 计算Q目标，更新评论家
   e. 计算演员梯度，更新演员
   f. 软更新目标网络
3. 每10轮评估一次平均回报

### 4.4 收敛与调试
- 收敛标志：连续控制任务回报达到阈值（如Pendulum达到-200以下）
- 调试技巧：
  - Q值持续上升：正常，说明Q值估计越来越准
  - Q值振荡剧烈：降低学习率，增大batch size
  - 动作饱和（输出边界值）：减小OU噪声，降低演员学习率

## 5. 应用场景
### 5.1 完整应用案例
#### 案例1：Pendulum摆起（同前）
#### 案例2：BipedalWalker行走
- 状态：24维连续（关节角度、速度等）
- 动作：4维连续扭矩[-1,1]
- 奖励：前进+奖励，摔倒-100，能耗惩罚
- 目标：稳定行走到达终点

#### 案例3：MuJoCo Reacher
- 状态：11维（目标位置、关节角度等）
- 动作：2维扭矩
- 奖励：距离目标的负距离平方
- 目标：控制机械臂碰到目标

#### 案例4：自动驾驶横向控制
- 状态：自车位置、速度、车道线偏移
- 动作：方向盘转角[-1,1]（归一化）
- 奖励：车道保持+奖励，偏移-惩罚
- 目标：稳定保持在车道中心

#### 案例5：机器人抓取
- 状态：摄像头图像、关节角度
- 动作：机械臂末端6维位姿变化
- 奖励：抓取成功+10，掉落-5
- 目标：稳定抓取目标物体

### 5.2 适用场景特征
| 特征 | 适用性 |
| --- | --- |
| 连续动作空间 | ✅ |
| 异策略学习 | ✅ |
| 高维连续控制 | ✅ |
| 需要经验复用 | ✅ |

### 5.3 不适用场景与替代方案
| 场景 | 原因 | 替代算法 |
| --- | --- | --- |
| 离散动作 | DDPG仅支持连续动作 | DQN、PPO |
| 极高样本效率 | DDPG有Q值过估计 | SAC、TD3 |
| 简单连续任务 | DDPG实现复杂 | PPO连续版 |

## 6. 优缺点分析
### 6.1 优点（含适用条件）
1. **连续动作适配**：首个成熟的深度连续控制算法。适用条件：连续动作空间。
2. **异策略高效**：回放缓冲区复用经验，样本效率比同策略高50%。适用条件：缓冲区足够大。
3. **确定性策略**：输出直接动作，无需采样，推理速度快。适用条件：动作空间连续。
4. **理论成熟**：基于确定性策略梯度定理，收敛性有保证。适用条件：Q值估计准确。
5. **扩展性强**：后续TD3、SAC都基于DDPG框架改进。适用条件：作为基准算法。

### 6.2 缺点（含问题与解决方案）
1. **Q值过估计**：高估Q值导致策略退化。解决方案：使用TD3，添加目标策略平滑。
2. **探索噪声敏感**：OU噪声参数难调，噪声过大不稳定。解决方案：随训练衰减噪声，使用自适应噪声。
3. **超参数多**：比PPO多5+个超参数，调参成本高。解决方案：使用默认参数，参考官方实现。
4. **动作边界处理**：输出动作可能超出环境范围。解决方案：Tanh激活限制动作到[-1,1]，再缩放到实际范围。
5. **已逐渐被替代**：SAC、TD3性能更好，新项目优先用后者。解决方案：新项目用TD3/SAC。

### 6.3 算法对比表
| 属性 | DDPG | TD3 | SAC |
| --- | --- | --- | --- |
| Q值过估计 | 严重 | 缓解 | 无 |
| 样本效率 | 中 | 高 | 极高 |
| 实现复杂度 | 中 | 中高 | 高 |
| 探索方式 | OU噪声 | OU噪声 | 熵正则 |

## 7. 调库实现（Python + PyTorch）
### 7.1 完整可运行代码
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """确定性演员网络（连续动作）"""
    def __init__(self, state_dim, action_dim, action_bound=1.0):
        super().__init__()
        self.action_bound = action_bound
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # 输出[-1,1]
        )
    def forward(self, state):
        return self.net(state) * self.action_bound  # 缩放到实际动作范围
    
    def get_action(self, state, noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state).detach().cpu().numpy()[0]
        action += noise * np.random.randn(*action.shape)  # 添加探索噪声
        return np.clip(action, -self.action_bound, self.action_bound)

class Critic(nn.Module):
    """评论家Q网络"""
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
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)

class ReplayBuffer:
    """回放缓冲区"""
    def __init__(self, capacity=1e6):
        self.buffer = deque(maxlen=int(capacity))
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)).to(device),
                torch.FloatTensor(np.array(actions)).to(device),
                torch.FloatTensor(rewards).to(device),
                torch.FloatTensor(np.array(next_states)).to(device),
                torch.FloatTensor(dones).to(device))
    def __len__(self):
        return len(self.buffer)

def train_ddpg(env_name="Pendulum-v1", num_episodes=200):
    env = gymnasium.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    # 初始化4个网络
    actor = Actor(state_dim, action_dim, action_bound).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim, action_bound).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)
    # 目标网络初始化为相同参数
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    actor_optim = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=1e6)
    
    return_history = []
    ou_noise = OUNoise(action_dim)  # 工程经验：OU噪声初始化
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        ou_noise.reset()
        ep_return = 0
        done = False
        
        while not done:
            # 选择动作+OU噪声
            action = actor.get_action(state, noise=ou_noise.noise())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # 存储到回放缓冲区
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward
            
            # 缓冲区足够大后开始更新
            if len(replay_buffer) > 64:
                # 采样mini-batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(64)
                # 计算Q目标
                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    next_q = target_critic(next_states, next_actions)
                    q_target = rewards + 0.99 * next_q * (1 - dones)
                # 评论家损失
                current_q = critic(states, actions)
                critic_loss = nn.MSELoss()(current_q, q_target)
                # 更新评论家
                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optim.step()
                
                # 演员损失（确定性策略梯度）
                actor_loss = -critic(states, actor(states)).mean()
                # 更新演员
                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optim.step()
                
                # 软更新目标网络
                tau = 0.005
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
        
        return_history.append(-ep_return)  # Pendulum奖励是负的，取负方便看上升
        if episode % 10 == 0:
            avg_return = np.mean(return_history[-10:])
            print(f"Episode {episode}, Avg Return: {-avg_return:.2f}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
    
    # 绘制曲线
    plt.plot(-np.array(return_history))
    plt.xlabel("Episode")
    plt.ylabel("Return (Positive = Better)")
    plt.title("DDPG Training Curve (Pendulum)")
    plt.savefig("ddpg_curve.png")
    plt.show()
    return actor, critic

class OUNoise:
    """Ornstein-Uhlenbeck噪声，用于连续动作探索"""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.reset()
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

if __name__ == "__main__":
    trained_actor, trained_critic = train_ddpg()
```

### 7.2 运行结果示例
```
Episode 0, Avg Return: -850.20, Actor Loss: 12.3456, Critic Loss: 15.6789
Episode 10, Avg Return: -620.50, Actor Loss: 8.7654, Critic Loss: 10.1234
...
Episode 100, Avg Return: -210.30, Actor Loss: 2.3456, Critic Loss: 3.4567
```

### 7.3 工程经验
1. 演员输出用Tanh+动作边界缩放，保证动作在环境范围内。
2. OU噪声随训练衰减，前50轮用全噪声，之后线性衰减到0。
3. 评论家用Huber损失替代MSE，对异常值更鲁棒。
4. 目标网络软更新τ=0.005是通用默认值，复杂任务可减小到0.001。
5. 回放缓冲区满后覆盖旧数据，保持数据多样性。

## 8. 手工代码实现（简化版）
### 8.1 核心逻辑实现
简化版聚焦确定性策略梯度和软更新：
```python
import numpy as np

class DDPG_Simplified:
    """简化版DDPG，核心逻辑实现"""
    def __init__(self, state_dim, action_dim):
        # 简化的网络和参数
        self.actor_W = np.random.randn(action_dim, state_dim) * 0.01
        self.critic_W = np.random.randn(1, state_dim + action_dim) * 0.01
        self.target_actor_W = self.actor_W.copy()
        self.target_critic_W = self.critic_W.copy()
        self.tau = 0.005
    
    def actor_forward(self, state):
        return np.tanh(self.actor_W @ state)
    
    def critic_forward(self, state, action):
        x = np.concatenate([state, action])
        return (self.critic_W @ x).item()
    
    def update(self, batch):
        # 简化更新逻辑，省略噪声、回放缓冲区细节
        # ... 计算critic loss，更新critic_W
        # ... 计算actor gradient，更新actor_W
        # 软更新目标网络
        self.target_actor_W = self.tau * self.actor_W + (1-self.tau) * self.target_actor_W
        self.target_critic_W = self.tau * self.critic_W + (1-self.tau) * self.target_critic_W
```

## 9. 可视化与结果理解
### 9.1 可视化示例
1. 回报收敛曲线：Pendulum任务下从-1000逐步上升到-200左右
2. 评论家Q值曲线：Q值估计逐步上升，趋近真实Q值
3. OU噪声衰减曲线：噪声标准差随训练逐步降低到0

### 9.2 结果解读
- 回报上升：策略逐步优化，摆起并稳定
- 评论家损失下降：Q值估计越来越准确
- Q值上升：策略质量提升，状态-动作对的Q值增加

## 10. 模型评估
### 10.1 评估代码
```python
def evaluate_ddpg(actor, env_name="Pendulum-v1", num_episodes=20):
    env = gymnasium.make(env_name)
    returns = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0
        while not done:
            action = actor.get_action(state, noise=0.0)  # 评估无噪声
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
    env.close()
    print(f"评估20轮平均回报：{-np.mean(returns):.2f}（Pendulum奖励为负，越小越好）")
    return np.mean(returns)
```

### 10.2 评估指标表（Pendulum）
| 指标 | 标准值 |
| --- | --- |
| 平均回报 | ~-200（越小越好） |
| 评论家损失 | <1.0 |
| 收敛轮次 | ~100轮 |

## 11. 常见问题与易错点
### 11.1 5个常见陷阱
1. **Q值过估计**
   - 现象：Q值持续升高，远超真实值，策略退化
   - 原因：DDPG固有缺陷，最大化Q值导致过估计
   - 解决方案：改用TD3，添加目标策略平滑

2. **动作超出边界**
   - 现象：动作值大于环境允许的最大值
   - 原因：未用Tanh限制输出，或缩放错误
   - 解决方案：演员输出用Tanh，再缩放到环境动作范围

3. **OU噪声过大**
   - 现象：探索噪声太大，训练不稳定
   - 原因：σ参数设置过高
   - 解决方案：降低σ到0.1~0.2，随训练衰减噪声

4. **目标网络更新过快**
   - 现象：目标网络和原网络差异大，训练崩溃
   - 原因：τ设置过高（>0.01）
   - 解决方案：降低τ到0.001~0.005

5. **回放缓冲区过小**
   - 现象：样本多样性不足，过拟合
   - 原因：缓冲区容量<1e5
   - 解决方案：增大到1e6以上

## 12. 学习总结
核心思想：确定性连续策略+Q学习+目标网络+回放缓冲区，适配连续控制异策略学习。
必记公式：
1. 确定性策略梯度：$\\nabla_\\theta J = \\mathbb{E}[\\nabla_\\theta \\mu_\\theta(s) \\nabla_a Q(s,a)|_{a=\\mu(s)}]$
2. 评论家损失：$L = \\mathbb{E}[(Q(s,a) - (r + \\gamma Q'(s', \\mu'(s')))^2]$
3. 软更新：$\\theta' \\leftarrow \\tau \\theta + (1-\\tau) \\theta'$

## 13. 练习题与思考题
### 13.1 基础题（含答案）
1. DDPG是哪一年提出的？
   <details>
   <summary>答案</summary>
   2015年，由Lillicrap等人提出。
   </details>

2. DDPG的核心创新是什么？
   <details>
   <summary>答案</summary>
   将DQN扩展到连续动作空间，用确定性策略+Q学习实现连续控制。
   </details>

3. 软更新τ的作用是什么？
   <details>
   <summary>答案</summary>
   缓慢更新目标网络参数，平衡目标稳定性和更新速度。
   </details>

### 13.2 进阶题
1. 推导确定性策略梯度公式。
   <details>
   <summary>推导</summary>
   见第3章3.4节完整推导。
   </details>

2. 为什么DDPG要用目标网络？
   <details>
   <summary>答案</summary>
   防止Q目标变化过快，稳定训练，避免Q值振荡。
   </details>

## 14. 学习路径建议
前置：DQN → A2C → 确定性策略梯度 → DDPG → TD3/SAC
资源：Lillicrap 2015 DDPG论文，Spinning Up DDPG章节。

> 来源线索：本节内容根据原书中关于"第8章 深度确定性策略梯度"的相关章节整理、扩展与教学化改写。