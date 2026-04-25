# TRPO（信任域策略优化）学习文档

> 强化学习中保证策略单调改进的里程碑算法

---

## 1. 算法基础认知

**一句话定义**：TRPO（Trust Region Policy Optimization，信任域策略优化）是由Schulman等人于2015年提出的强化学习算法，通过在策略更新的步长上施加约束，确保策略的单调改进（保证性能不下降），是PPO等现代算法的理论基础。

**直觉类比**：TRPO就像学骑车时的"渐进式"学习方法。当你要学会一个新动作（比如单脚站立）时，你不会突然完全改变骑车方式，而是每次只移动一点点——比如先把重心稍微前移一点点，感受一下是否稳定。如果稳定，再继续；如果不稳，就退回来。TRPO的核心思想就是"每次只改变一点点"，通过限制新策略和旧策略的KL散度（分布差异），确保每次更新都是安全的，不会让性能突然下降。

**历史背景**：
- 2015年，John Schulman等人在论文"Trust Region Policy Optimization"中提出
- 解决了策略梯度方法学习率敏感的问题
- 是PPO的理论基础
- 后续发展出PPO2、PPO-Max等

**算法定位**：
- 类型：强化学习 → 策略优化
- 输出：策略参数更新
- 模型类型：基于置信域的方法

**前置知识**：
- [必备]：强化学习基础（MDP、回报）
- [必备]：策略梯度（REINFORCE、Actor-Critic）
- [推荐]：马尔可夫决策过程

---

## 2. 核心原理

### 2.1 策略梯子的问题

传统策略梯度方法（如REINFORCE）面临核心问题：

| 问题 | 表现 | 原因 |
|------|------|------|
| 学习率敏感 | 性能波动大 | 步长难以确定 |
| 不稳定 | 训练发散 | 可能跨度过大 |
| 单调性 | 无法保证改进 | 可能变差 |

**根本原因**：策略更新步长不合适！

### 2.2 TRPO的核心创新

**信任域约束**：限制新旧策略的差异！

$$\max_{\theta} \mathbb{E}_{s \sim \rho_{\theta_{old}}, a \sim \pi_{\theta_{old}}} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)} \hat{A}(s,a) \right]$$

s.t. 

$$\mathbb{E}_{s \sim \rho_{\theta_{old}}} \left[ KL(\pi_{\theta_{old}}(\cdot|s) || \pi_{\theta}(\cdot|s)) \right] \leq \delta$$

其中：
- $\hat{A}$：优势函数估计
- $\delta$：信任域半径（通常0.01-0.02）
- $\rho_{\theta_{old}}$：旧策略的状态访问分布

### 2.3 整体流程

```
                    旧策略 π_θold
                          │
            ┌──────────────┴──────────────┐
            ▼                           ▼
    收集轨迹 D               计算优势函数 Â
            │                           │
            └──────────────┬──────────────┘
                           ▼
    ┌─────────────────────────────────────┐
    │       约束优化问题                 │
    │  max E[ratio · Â]                  │
    │  s.t. KL(π_new || π_old) ≤ δ      │
    └──────────────┬─────────────────────┘
                   ▼
            新策略 π_θnew
```

---

## 3. 数学公式与推导

### 3.1 目标函数

**策略梯度目标**（从策略梯度定理推导）：

$$L^{\pi}(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim \pi_{\theta}} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{\theta_{old}}}(s,a) \right]$$

其中 $d^{\pi}$ 是策略 $\pi$ 的折扣状态访问分布。

### 3.2 重要性采样比率

$$r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)$$

这个比率衡量新旧策略的差异。

### 3.3 KL散度约束

**KL散度**（多项式分布）：

$$KL(\pi_{old} || \pi) = \sum_a \pi_{old}(a) \log \frac{\pi_{old}(a)}{\pi(a)}$$

近似形式（二阶展开）：

$$KL \approx \frac{1}{2} \mathbb{E}_{s} \left[ (\theta - \theta_{old})^T F (theta - \theta_{old}) \right]$$

其中 $F$ 是Fisher信息矩阵。

### 3.4 共轭梯度求解

由于约束优化直接求解困难，TRPO使用**共轭梯度法**：

```python
# 伪代码：共轭梯度求解
def conjugate_gradient(A, b, x0, max_iter=10):
    x = x0
    r = b - A @ x
    p = r
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        if np.linalg.norm(r_new) < 1e-8:
            break
            
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
    
    return x
```

### 3.5 线性近似

使用一阶泰勒展开近似目标函数：

$$\hat{L}(\theta) \approx L(\theta_{old}) + \nabla_{\theta} L |_{\theta_{old}} \cdot (\theta - \theta_{old})$$

梯度：

$$\nabla_{\theta} L = \mathbb{E}_{s,a \sim \pi_{old}} \left[ r_t(\theta) \nabla_{\theta} \log \pi_{\theta}(a|s) \hat{A}(s,a) \right]$$

---

## 4. 训练过程讲解

### 4.1 整体训练流程

```
       初始化策略参数 θ
           │
           ▼
    ┌───────────────┐
    │ 收集轨迹数据  │ ← 使用当前策略
    │  (s, a, r)  │
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 计算回报    │ ← GAE或TD
    │  G(t)       │
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 计算优势函数 │ ← advantage估计
    │  A(s,a)    │
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 约束优化    │ ← 共轭梯度
    │  max L s.t.KL≤δ│
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 更新策略   │ ← 新参数 θ
    └───────┬───────┘
           │
           └───→ 返回循环
```

### 4.2 优势函数估计

**使用GAE（Generalized Advantage Estimation）**：

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

### 4.3 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| max_kl | 0.01-0.02 | KL散度上限 |
| cg_iters | 10 | 共轭梯度迭代 |
| gamma | 0.99 | 折扣因子 |
| lam | 0.95 | GAE参数 |
| ent_coef | 0.01 | 熵系数 |

### 4.4 实现技巧

| 技巧 | 说明 |
|------|------|
| 线搜索 | 确保约束满足 |
| 阻尼 | 改善数值稳定性 |
| 熵奖励 | 鼓励探索 |

---

## 5. 应用场景

### 5.1 连续控制

TRPO在连续控制任务中表现优异：

```python
# MuJoCo环境
# HalfCheetan, Walker2d, Hopper, Swimmer
```

### 5.2 机器人控制

实际机器人任务的策略学习：

```python
# 双足行走
# 机械臂操作
```

### 5.3 游戏AI

 Atari游戏等：

```python
# Pong, Breakout
# 需要精确动作控制
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **稳定性** | 保证单调改进 |
| **无超参敏感** | 对学习率不敏感 |
| **样本效率** | 比策略梯度好 |
| **理论保证** | 有收敛证明 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **计算重** | 需要Fisher矩阵 |
| **内存大** | 存储大量轨迹 |
| **实现复杂** | 共轭梯度难实现 |
| **不太适合Online** | 不如PPO灵活 |

### 6.3 改进方向

| 改进 | 方法 |
|------|------|
| PPO | 简化约束为裁剪 |
| ACER | 加入回忆录 |
| SAC | 自动调整温度 |

---

## 7. 调库实现

### 7.1 rllab实现

```python
# 安装
# rllab库

from rllab.algos.trpo import TRPO
from rllab.envs.box2d CartPoleEnv import CartPoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.q_functions.mlp_q_function import MLPQFunction

env = CartPoleEnv()

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(32, 32)
)

algo = TRPO(
    env=env,
    policy=policy,
    n_itr=500,
    batch_size=5000,
    max_kl=0.01,
   -gi_iters=10
)

algo.train()
```

### 7.2/stable-baselines3

```python
# 安装
# pip install stable-baselines3

import gym
from stable_baselines3 import PPO, TRPO

env = gym.make('Hopper-v3')

# PPO (简化版TRPO，推荐使用)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# 评估
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break
```

### 7.3 手动实现PYTORCH

```python
import torch
import torch.nn as nn
import numpy as np


class TRPOPolicy(nn.Module):
    """TRPO策略网络"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=(64, 64)):
        super().__init__()
        
        layers = []
        dims = [obs_dim] + list(hidden_dims)
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
            
        self.feature = nn.Sequential(*layers)
        
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs):
        features = self.feature(obs)
        mean = self.mean(features)
        return mean, self.log_std.exp()
    
    def get_action(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class ValueFunction(nn.Module):
    """价值函数"""
    
    def __init__(self, obs_dim, hidden_dims=(64, 64)):
        super().__init__()
        
        layers = []
        dims = [obs_dim] + list(hidden_dims)
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, obs):
        return self.net(obs)


def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """GAE优势估计"""
    
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t+1]
            
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages)


class TRPO:
    """TRPO实现"""
    
    def __init__(self, env, policy, value_fn, lr=1e-3, gamma=0.99, 
                 lam=0.95, max_kl=0.01, cg_iters=10):
        self.env = env
        self.policy = policy
        self.value_fn = value_fn
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        
        self.policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
        
    def collect_trajectories(self, num_steps):
        """收集轨迹"""
        
        obs = self.env.reset()
        obss = []
        actions = []
        rewards = []
        log_probs = []
        
        for _ in range(num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            with torch.no_grad():
                action, log_prob = self.policy.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()
            obs, reward, done, _ = self.env.step(action_np)
            
            obss.append(obs_tensor)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            if done:
                obs = self.env.reset()
        
        return (torch.stack(obss), torch.stack(actions), 
                torch.tensor(rewards, dtype=torch.float32),
                torch.stack(log_probs))
    
    def update(self, obss, actions, rewards, old_log_probs):
        """策略更新"""
        
        # 计算价值
        values = self.value_fn(obss).squeeze()
        
        # 计算优势
        advantages = compute_advantages(rewards, values.detach(), 
                                     self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 策略梯度
        mean = self.policy.get_action(obss)[0]
        dist = torch.distributions.Normal(mean, self.policy.log_std.exp())
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        ratio = (new_log_probs - old_log_probs).exp()
        surr_loss = -(ratio * advantages).mean()
        
        # 反向传播策略梯度
        self.policy_opt.zero_grad()
        surr_loss.backward()
        self.policy_opt.step()
        
        # 注意：实际TRPO会用约束优化
        # 这里是简化版
        
    def train(self, num_iterations=1000, steps_per_iter=2048):
        """训练循环"""
        
        for i in range(num_iterations):
            obss, actions, rewards, log_probs = self.collect_trajectories(steps_per_iter)
            self.update(obss, actions, rewards, log_probs)
            
            if i % 10 == 0:
                total_reward = rewards.sum().item()
                print(f"Iter {i}: Reward = {total_reward:.2f}")


def demo():
    """演示"""
    import gym
    
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = TRPOPolicy(obs_dim, action_dim)
    value_fn = ValueFunction(obs_dim)
    
    trpo = TRPO(env, policy, value_fn, max_kl=0.01)
    trpo.train(num_iterations=50, steps_per_iter=1000)


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

### 8.1 完整TRPO实现

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        mean = self.net(state)
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            
        log_prob = Normal(mean, std).log_prob(action).sum(dim=-1)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.net(state)


def conjugate_gradient(A_func, b, x_init, max_iter=10, eps=1e-8):
    """共轭梯度求解 Ax = b"""
    
    x = x_init.clone()
    r = b.clone()
    p = r.clone()
    rsold = (r * r).sum()
    
    for _ in range(max_iter):
        Ap = A_func(p)
        pAp = (p * Ap).sum()
        
        if pAp < eps:
            break
            
        alpha = rsold / (pAp + eps)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = (r * r).sum()
        
        if rsnew < eps:
            break
            
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x


def flat_grad(f, params):
    """计算梯度"""
    return torch.autograd.grad(f, params, 
                            create_graph=True,
                            allow_unused=True)


def flat_hessian(L, params):
    """计算Hessian"""
    grads = flat_grad(L, params)
    flat_grads = torch.cat([g.view(-1) if g is not None else 
                          torch.zeros(p.numel()) 
                          for p, g in zip(params, grads)])
    
    grads.requires_grad_(True)
    
    g2 = (grads * grads).sum()
    h = flat_grad(g2, grads)
    
    hessian = torch.cat([g.view(-1) if g is not None else 
                       torch.zeros(p.numel()) 
                       for p, g in zip(params, h)])
    
    return hessian


class TRPOAgent:
    """TRPO智能体"""
    
    def __init__(self, state_dim, action_dim, max_kl=0.01, 
                 gamma=0.99, lam=0.95):
        
        self.max_kl = max_kl
        self.gamma = gamma
        self.lam = lam
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_fn = ValueNetwork(state_dim)
        
        self.policy_params = list(self.policy.parameters())
        
    def get_trajectory(self, env, max_steps=1000):
        """收集轨迹"""
        
        state = env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []
        
        total_reward = 0
        done = False
        
        while len(states) < max_steps and not done:
            state_t = torch.FloatTensor(state)
            
            with torch.no_grad():
                action, log_prob = self.policy.get_action(state_t)
            
            action_np = action.cpu().numpy()
            
            state, reward, done, _ = env.step(action_np)
            
            states.append(state_t)
            actions.append(torch.FloatTensor(action_np))
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            
            total_reward += reward
            
        return states, actions, rewards, log_probs, total_reward
    
    def compute_advantages(self, rewards, values, dones):
        """GAE计算"""
        
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages)
        
        return advantages - advantages.mean(), advantages.std()
    
    def update(self, states, actions, rewards, old_log_probs):
        """TRPO更新"""
        
        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        
        # 计算价值
        values = self.value_fn(states_t).squeeze()
        
        # 计算优势
        with torch.no_grad():
            advantages, _ = self.compute_advantages(
                rewards, values.tolist(), [False] * len(rewards)
            )
        
        # 策略loss
        mean, std = self.policy(states_t)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
        
        ratio = (new_log_probs - torch.stack(old_log_probs)).exp()
        surr_loss = -(ratio * advantages).mean()
        
        # 价值loss
        value_loss = F.mse_loss(values, 
                              torch.stack(rewards[:-1]).flip(0).cumsum(0).flip(0) 
                              if len(rewards) > 1 else values * 0)
        
        # 策略梯度
        self.policy.optimizer.zero_grad()
        surr_loss.backward()
        
        # 简化实现：直接用梯度上升
        # 实际应该用共轭梯度+约束优化
        
        self.policy.optimizer.step()


def demo_trpo():
    """TRPO演示"""
    import gym
    
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = TRPOAgent(state_dim, action_dim)
    
    for episode in range(20):
        states, actions, rewards, log_probs, total_reward = agent.get_trajectory(env, 200)
        
        print(f"Episode {episode}: Reward = {total_reward}")
        
        if len(states) > 10:
            agent.update(states, actions, rewards, log_probs)


if __name__ == "__main__":
    demo_trpo()
```

---

## 9. 优缺点与评估

### 9.1 性能评估

| 环境 | 回报 |
|------|------|
| HalfCheetan | ~2000 |
| Hopper | ~2000 |
| Walker2d | ~1500 |

### 9.2 与PPO对比

| 方法 | 稳定性 | 计算量 | 实现 |
|------|--------|--------|------|
| TRPO | 很高 | 中 | 复杂 |
| PPO | 高 | 低 | 简单 |

---

## 10. 学习总结

### 10.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | 约束策略更新确保单调改进 |
| 约束 | KL散度 ≤ max_kl |
| 求解 | 共轭梯度 |
| 优势 | 稳定性高 |

### 10.2公式记忆

**目标**：
$$\max_{\theta} \mathbb{E}[r_t(\theta) \hat{A}]$$

**约束**：
$$KL(\pi_{old} || \pi_{\theta}) \leq \delta$$

### 10.3 扩展

- PPO：简化版（裁剪）
- SAC：最大熵
- A3C：异步

---

## 11. 练习题与思考题

### 11.1 选择题

1. TRPO的核心约束是：
   - A) 学习率
   - B) KL散度
   - C) 熵

2. TRPO使用什么求解器？
   - A) SGD
   - B) 共轭梯度
   - C) Adam

3. PPO相比TRPO：
   - A) 更稳定
   - B) 更简单
   - C) 更复杂

### 11.2 简答题

1. 解释"信任域"的概念？
2. 为什么限制KL散度能保证稳定性？
3. 比较TRPO和PPO的优缺点？

### 11.3 编程题

1. 实现共轭梯度求解器
2. 计算Fisher矩阵
3. 在CartPole上测试TRPO

---

## 12. 学习路径建议

### 12.1 进阶路径

```
强化学习基础
    ↓
策略梯度
    ↓
TRPO理论
    ↓
共轭梯度
    ↓
PPO实现
    ↓
SAC
```

### 12.2 相关算法

| 算法 | 关系 |
|------|------|
| PPO | TRPO简化版 |
| A2C | 同步版 |
| SAC | 最大熵版 |
| DDPG | 连续动作 |

### 12.3 扩展阅读

1. Schulman et al. (2015). Trust Region Policy Optimization
2. Schulman et al. (2017). Proximal Policy Optimization

---

## 13. 常见问题与易错点

### Q1: 共轭梯度怎么实现？

**答案**：用scipy.sparse.linalg.cg或手动实现。

### Q2: Fisher矩阵怎么计算？

**答案**：通过对数概率的Hessian近似。

### Q3: max_kl设置多少？

**答案**：常用0.01-0.02。

### Q4: PPO和TRPO选哪个？

**答案**：PPO更简单常用，TRPO更稳定。

### Q5: 支持离散动作吗？

**答案**：支持，需要修改策略输出。

---

## 14. 学习总结

### 14.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | 约束策略更新确保单调改进 |
| 约束 | KL散度 ≤ max_kl |
| 求解 | 共轭梯度 |
| 优势 | 稳定性高 |

### 14.2 公式汇总

目标函数：
$$\max_{\theta} \mathbb{E}[r_t(\theta) \hat{A}]$$

约束：
$$KL(\pi_{old} || \pi_{\theta}) \leq \delta$$

TRPO损失：
$$\mathcal{L}_{TRPO} = \mathcal{L}_{PG} - \beta \cdot KL$$

---

## 附录

### A. 参数速查

| 参数 | 推荐值 |
|------|--------|
| max_kl | 0.01 |
| discount | 0.99 |
| gae_lambda | 0.95 |
| cg_iters | 10 |

### B. 参考

1. Schulman et al. (2015). TRPO. arXiv:1502.05477
2. stable-baselines3库

---

**文档结束**