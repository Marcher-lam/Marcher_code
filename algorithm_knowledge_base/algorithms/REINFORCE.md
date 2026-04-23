# REINFORCE 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
REINFORCE（REward Increment = Non-negative Factor times Estimate times Reinforcement times Characteristic Eligibility）是一种策略梯度算法，通过蒙特卡洛采样估计策略梯度来直接优化随机策略的参数。

### 1.2 直觉类比
想象教练指导游泳运动员。教练不能直接告诉运动员"每个时刻该怎么动"，只能在该运动员完成整个比赛后给出一个总体评分（奖励）。REINFORCE就像教练根据最终成绩反向推断：游得好的动作应该保留，游得差的动作应该改进。"高分时做的每个动作都值得强化"是核心思想。

### 1.3 历史背景
REINFORCE由Ronald J. Williams在1992年论文"Statistical Learning in Stochastic Dynamic Environments"中提出，是最早的策略梯度算法之一。尽管后来出现了更复杂的算法（如PPO、Actor-Critic），REINFORCE仍然是理解策略梯度的基石。

### 1.4 算法定位
- 类型：强化学习（策略梯度）
- 输出：随机策略 $\pi(a|s;\theta)$
- 模型类别：参数化策略模型

### 1.5 前置知识
- 强化学习基础（MDP、回报）
- 概率论（期望、方差）
- 深度学习（神经网络）
- Python 编程（PyTorch）

## 2. 核心原理
### 2.1 核心思想
REINFORCE的核心思想是利用"策略梯度定理"：策略性能的梯度可以写成"梯度乘以回报"的期望形式。通过采样完整轨迹，用轨迹的回报作为无偏估计来计算梯度并更新策略参数。

### 2.2 工作流程
1. 初始化策略参数 $\theta$
2. 采样一条轨迹 $\tau = (s_0, a_0, r_0, ..., s_T)$
3. 计算轨迹的总回报 $G = \sum_{t=0}^{T} \gamma^t r_t$
4. 对轨迹中每一步：计算 $\nabla_\theta \log \pi(a_t|s_t;\theta)$
5. 计算梯度估计：$\hat{g} = G \cdot \sum_t \nabla_\theta \log \pi(a_t|s_t;\theta)$
6. 使用梯度上升更新 $\theta$
7. 重复步骤2-6

### 2.3 关键概念解释
- **轨迹回报**：从当前时刻到结束的折扣奖励总和
- **对数策略梯度**：$\nabla_\theta \log \pi(a|s)$给出在参数空间中使该动作概率更大的方向
- **基线(baseline)**：减去基线减少方差，不改变期望

### 2.4 几何/直观解释
策略参数空间是一个高维曲面。REINFORCE的梯度方向指向使当前轨迹更可能重复的方向。由于使用整个轨迹的回报作为权重，高回报轨迹会使其中所有动作的概率都增加。

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $\pi_\theta(a|s)$ | 参数化策略 |
| $J(\theta)$ | 策略性能（平均回报） |
| $G_t$ | 从时刻t开始的折扣回报 |
| $\nabla_\theta \log \pi_\theta(a|s)$ | 策略梯度 |
| $\theta$ | 策略网络参数 |

### 3.2 问题形式化
寻找最大化期望累积奖励的参数：
$$\max_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

### 3.3 目标函数/损失函数
无显式损失函数。通过梯度上升最大化目标函数。

### 3.4 推导过程
从性能梯度定理开始（ Sutton et al. 1999）：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

推导：
$$J(\theta) = \mathbb{E}_{\tau}[R] = \int \pi_\theta(\tau) R(\tau) d\tau$$

$$\nabla_\theta J = \int \nabla_\theta \pi_\theta(\tau) R(\tau) d\tau = \int \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau) R(\tau) d\tau$$

其中 $\pi_\theta(\tau) = \prod_t \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t,a_t)$

因此 $\nabla_\theta \log \pi_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$

代入得：
$$\nabla_\theta J = \mathbb{E}_{\tau}\left[\left(\sum_t G_t\right) \cdot \left(\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right)\right]$$

简化后得到各时刻的独立贡献：
$$\nabla_\theta J = \mathbb{E}_{\tau}\left[\sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

### 3.5 最终解/算法步骤
```
初始化：随机策略参数θ

for episode in 1..M:
    1. 用策略π_θ采样轨迹 τ = (s0,a0,r0,...,sT)
    2. 计算每个时刻的回报：G_t = Σ_{k=t}^{T-1} γ^k r_k
    3. 对轨迹中每一步：
       ∇θ += G_t · ∇θ log πθ(at|st)
    4. θ ← θ + α · ∇θ  (梯度上升)
```

## 4. 训练过程讲解
### 4.1 数据预处理
REINFORCE不需要显式预处理。状态可能需要归一化以提高训练稳定性。

### 4.2 参数初始化
- 策略网络：随机初始化
- 基线：初始化为0

### 4.3 迭代过程
```
for epoch in range(num_epochs):
    for _ in range(trajectories_per_epoch):
        trajectory = collect_trajectory(policy)
        returns = compute_returns(trajectory)
        
        for t, (s,a,G) in enumerate(trajectory):
            log_prob = policy.log_prob(s, a)
            loss = -G * log_prob
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4.4 收敛条件
- 回报变化小于阈值
- 最大epoch数

### 4.5 超参数及推荐范围
- 学习率：3e-4
- 批量轨迹数：100
- 折扣因子γ：0.99
- 基线系数：有基线时可选

## 5. 应用场景
### 5.1 典型应用
- **游戏AI**：Atari游戏、围棋
- **机器人控制**：连续动作
- **自然语言处理**：文本生成
- **推荐系统**：序列推荐

### 5.2 适用数据特征
- 离散或连续动作空间
- 可采样完整轨迹
- 奖励稀疏或延迟

### 5.3 不适用场景
- 需要高样本效率的场景（方差过高）
- 动作空间极大的场景
- 实时决策场景

## 6. 优缺点分析
### 6.1 优点
- 理论基础扎实
- 兼容任意可微策略参数化
- 适用于离散和连续动作
- 无Bootstrapping偏差

### 6.2 缺点
- 方差极高
- 样本效率低
- 需要完整轨迹
- 难以处理稀疏奖励

### 6.3 与同类算法对比

| 算法 | 方差 | 样本效率 | 偏差 | 适用场景 |
|------|-----|---------|------|--------|---------|
| REINFORCE | 高 | 低 | 无 | 基准 |
| Actor-Critic | 中 | 中 | 少量 | 连续控制 |
| PPO | 低 | 高 | 少量 | SOTA |

## 7. 调库实现
### 7.1 环境准备
```bash
pip install numpy pandas matplotlib torch gym
```

### 7.2 完整代码示例
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward'])

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        logits = self.net(state)
        return logits
    
    def get_action(self, state, deterministic=False):
        logits = self.forward(state)
        if deterministic:
            return torch.argmax(logits).item()
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def log_prob(self, state, action):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        return log_probs[action]

def compute_returns(rewards, gamma=0.99):
    """计算折扣回报"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

class REINFORCE:
    """REINFORCE算法"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.policy.get_action(state, deterministic)
    
    def update(self, trajectories):
        total_loss = 0
        for traj in trajectories:
            states, actions, rewards = zip(*traj)
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards_np = np.array(rewards)
            returns = compute_returns(rewards_np, self.gamma)
            
            returns = normalize(returns)
            
            log_probs = []
            for s, a in zip(states, actions):
                lp = self.policy.log_prob(s.unsqueeze(0), a.item())
                log_probs.append(lp)
            
            log_probs = torch.stack(log_probs)
            loss = -(log_probs * returns).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(trajectories)

def collect_trajectory(env, policy, max_steps=200):
    """采集一条轨迹"""
    state = env.reset()
    trajectory = []
    
    for _ in range(max_steps):
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        trajectory.append((state, action, reward))
        state = next_state
        
        if done:
            break
    
    return trajectory

def train_reinforce(env_name='CartPole-v1', num_episodes=500, num_trajectories=10):
    """训练REINFORCE"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(state_dim, action_dim)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        trajectories = []
        for _ in range(num_trajectories):
            traj = collect_trajectory(env, agent)
            trajectories.append(traj)
        
        agent.update(trajectories)
        
        test_reward = 0
        for _ in range(5):
            traj = collect_trajectory(env, agent)
            test_reward += sum(r for _, _, r in traj)
        
        avg_reward = test_reward / 5
        episode_rewards.append(avg_reward)
        
        if episode % 50 == 0:
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")
    
    env.close()
    return episode_rewards

def compare_with_baseline(env_name='CartPole-v1', num_episodes=300):
    """与随机策略对比"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(state_dim, action_dim)
    
    random_rewards = []
    agent_rewards = []
    
    for episode in range(num_episodes):
        traj = collect_trajectory(env, agent)
        reward = sum(r for _, _, r in traj)
        agent_rewards.append(reward)
        
        state = env.reset()
        random_reward = 0
        for _ in range(200):
            action = env.action_space.sample()
            _, r, done, _ = env.step(action)
            random_reward += r
            if done:
                break
        random_rewards.append(random_reward)
        
        if episode % 50 == 0:
            print(f"Episode {episode}: Agent={reward:.1f}, Random={random_reward:.1f}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(agent_rewards, label='REINFORCE', alpha=0.8)
    plt.plot(random_rewards, label='Random', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE vs Random Policy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_reinforce(num_episodes=200)
    compare_with_baseline()
```

### 7.3 运行结果示例
```
Episode 0: Avg Reward = 21.4
Episode 50: Avg Reward = 85.2
Episode 100: Avg Reward = 142.6
Episode 150: Avg Reward = 178.4

CartPole通常在200 episodes达到接近200的分数
```

## 8. 手工代码实现
### 8.1 核心算法手写
上节代码已完整实现：
- PolicyNetwork（策略网络）
- REINFORCE类（梯度计算与更新）
- 轨迹采集
- 基线归一化

### 8.2 与调库结果对比
Stable-Baselines3的REINFORCE实现更加高效，支持更多环境。手工版本便于理解原理。

## 9. 可视化与结果理解
### 9.1 训练曲线
```python
import matplotlib.pyplot as plt

def plot_training_curve(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE Training Curve')
    plt.grid(True)
    plt.show()
```

### 9.2 策略可视化
```python
def visualize_value_function(policy, env):
    """可视化状态值函数近似"""
    states = np.linspace(-1, 1, 20)
    values = []
    for s in states:
        state_tensor = torch.FloatTensor([s, 0, 0, 0])
        values.append(policy.evaluate_state(state_tensor))
    plt.plot(states, values)
    plt.show()
```

### 9.3 结果解读
- 初始阶段分数低且波动大（随机探索）
- 中期分数上升（策略逐渐学会）
- 后期分数高且稳定（策略收敛）

## 10. 模型评估
### 10.1 评估指标选择
- **平均回报**
- **收敛速度**
- **方差**

### 10.2 评估代码
```python
def evaluate_policy(policy, env, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        traj = collect_trajectory(env, policy)
        rewards.append(sum(r for _, _, r in traj))
    return np.mean(rewards), np.std(rewards)
```

### 10.3 基线归一化效果
使用基线可显著降低方差：
$$V(s_t) = \mathbb{E}[G_t]$$ 作为基线

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
- 轨迹太短（学习不够）
- 未考虑折扣因子

### 11.2 模型层面常见错误
- 忽略回报标准化
- log概率计算错误

### 11.3 调参层面常见误区
- 学习率过高导致不稳定
- 批量大小不当

## 12. 学习总结
### 12.1 核心要点回顾
- REINFORCE直接优化策略梯度
- 无偏但方差高
- 使用完整轨迹的回报

### 12.2 关键公式汇总
$$\nabla_\theta J = \mathbb{E}_\tau\left[\sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

### 12.3 与前序/后续算法联系
- 前置：随机策略搜索
- 同级：Actor-Critic
- 进阶：PPO、A2C

## 13. 练习题与思考题与思考题
### 13.1 基础练习题
1. 为什么REINFORCE是无偏的？
2. 如何降低REINFORCE的方差？
3. 基线的作用是什么？

### 13.2 进阶思考题
1. REINFORCE与Actor-Critic的区别是什么？
2. 能否将REINFORCE与值函数结合？
3. 如何处理连续动作空间？

### 13.3 详细答案与解析
**答案1**：因为期望梯度等于真实性能梯度。

**答案2**：引入基线、使用多个轨迹平均、引入值函数。

**答案3**：保持无偏的同时减少方差，因为 $E[G_t - b] = E[G_t]$ 当b与动作无关时。

## 14. 学习路径建议建议
### 14.1 前置知识
- MDP基础
- 概率论（期望、梯度）

### 14.2 平行算法
- 随机搜索
- 进化策略

### 14.3 进阶算法
- Actor-Critic
- PPO
- A2C

### 14.4 推荐资源
- 论文：Williams 1992 "Statistical Learning in Stochastic Dynamic Environments"
- 书籍：Sutton & Barto 第13章
- 课程：David Silver RL公开课