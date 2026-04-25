# TD3（Twin Delayed DDPG）学习文档

## 1. 算法基础认知

TD3 由 Scott Fujimoto 等于 2018 年提出，是 DDPG 的改进版本。它通过三个关键技巧解决 DDPG 中的 Q 值过估计和训练不稳定问题，是连续动作空间中性能最稳定的算法之一。在广告系统中用于连续出价调整。

## 2. 核心原理

TD3 在 DDPG 基础上增加三个技巧：

### 1. 裁剪双 Q 学习（Clipped Double-Q）

使用两个 Q 网络，目标值取较小者：

$$y = r + \gamma \min_{i=1,2} Q'_{\theta_i'}(s', \tilde{a}')$$

### 2. 延迟策略更新（Delayed Policy Updates）

Critic 每步更新，Actor 每 d 步更新一次（通常 d=2），减少 Actor 更新对 Critic 的影响。

### 3. 目标策略平滑（Target Policy Smoothing）

在目标动作上添加截断噪声：

$$\tilde{a}' = \mu'(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma)$$

这使 Q 值在动作空间上更平滑，防止过拟合窄峰。

## 3. 数学公式与推导

Critic 损失（两个 Q 网络分别计算）：

$$L(\theta_i) = \frac{1}{N}\sum_j\left(y_j - Q_{\theta_i}(s_j, a_j)\right)^2, \quad i=1,2$$

目标值：

$$y = r + \gamma \min(Q'_{\theta_1'}(s', \tilde{a}'), Q'_{\theta_2'}(s', \tilde{a}')) \text{ where } \tilde{a}' = \text{clip}\left(\mu'(s') + \text{clip}(\mathcal{N}(0,\sigma),-c,c), a_{low}, a_{high}\right)$$

Actor 损失（每 d 步更新）：

$$\nabla_\phi J \approx \frac{1}{N}\sum_j \nabla_a Q_{\theta_1}(s_j, a)|_{a=\mu_\phi(s_j)} \nabla_\phi \mu_\phi(s_j)$$

注意：Actor 只用 Q₁（而非 min）来计算梯度，因为双 Q 的目的只是减少目标值过估计。

## 4. 训练过程讲解

1. 初始化 Actor μ、双 Critic Q₁/Q₂ 及其目标网络
2. 每步：用 Actor 选动作 + 噪声探索，存 (s,a,r,s')
3. 采样 mini-batch，计算目标值 y（含平滑噪声 + 双 Q 取 min）
4. 更新 Q₁ 和 Q₂
5. 每 d 步：更新 Actor，软更新所有目标网络
6. 重复步骤 2-5

## 5. 应用场景

- 广告连续出价调整（DDPG 升级版）
- 机器人连续控制
- 自动驾驶
- 工业过程控制
- 能源调度优化

## 6. 优缺点分析

**优点**：
- 解决 DDPG 的 Q 值过估计问题
- 训练稳定性显著提升
- 目标平滑减少 Q 值窄峰
- 延迟更新减少 Actor-Critic 耦合

**缺点**：
- 仅适用于连续动作空间
- 实现复杂度高于 DDPG
- 确定性策略，探索能力不如 SAC
- 超参数较多

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, action_dim), nn.Tanh())
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.ReLU(),
                                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.ReLU(),
                                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, a):
        inp = torch.cat([x, a], dim=-1)
        return self.q1(inp), self.q2(inp)

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action=1.0):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=100000)
        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_delay = 2
        self.total_it = 0

    def select_action(self, state, noise=0.1):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s).squeeze(0).numpy()
        if noise > 0:
            a = a + np.random.normal(0, noise, size=a.shape)
        return np.clip(a, -self.max_action, self.max_action)

    def store(self, *transition):
        self.buffer.append(transition)

    def _soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def update(self, batch_size=100):
        if len(self.buffer) < batch_size:
            return
        self.total_it += 1
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_n, d = map(lambda x: torch.FloatTensor(np.array(x)), zip(*batch))
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a_n = (self.actor_target(s_n) + noise).clamp(-self.max_action, self.max_action)
            q1_t, q2_t = self.critic_target(s_n, a_n)
            target_q = r + self.gamma * torch.min(q1_t, q2_t) * (1 - d)
        q1, q2 = self.critic(s, a)
        c_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
        if self.total_it % self.policy_delay == 0:
            a_loss = -self.critic.q1(torch.cat([s, self.actor(s)], dim=-1)).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class SimpleTD3:
    def __init__(self, state_dim, action_dim, lr_c=0.001, lr_a=0.0001):
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.total_it = 0
        self.q1 = {'W': np.random.randn(state_dim + action_dim, 32) * 0.1,
                    'b': np.zeros(32), 'W2': np.random.randn(32, 1) * 0.1, 'b2': np.zeros(1)}
        self.q2 = {'W': np.random.randn(state_dim + action_dim, 32) * 0.1,
                    'b': np.zeros(32), 'W2': np.random.randn(32, 1) * 0.1, 'b2': np.zeros(1)}
        self.actor = {'W': np.random.randn(state_dim, 32) * 0.1,
                      'b': np.zeros(32), 'W2': np.random.randn(32, action_dim) * 0.1, 'b2': np.zeros(action_dim)}
        self.target_q1 = {k: v.copy() for k, v in self.q1.items()}
        self.target_q2 = {k: v.copy() for k, v in self.q2.items()}
        self.target_actor = {k: v.copy() for k, v in self.actor.items()}

    def predict_q(self, s, a, params):
        x = np.concatenate([s, a])
        h = np.maximum(0, x @ params['W'] + params['b'])
        return h @ params['W2'] + params['b2']

    def predict_actor(self, s, params):
        h = np.maximum(0, s @ params['W'] + params['b'])
        return np.tanh(h @ params['W2'] + params['b2'])

    def soft_update(self, target, source):
        for k in target:
            target[k] = self.tau * source[k] + (1 - self.tau) * target[k]

    def update(self, s, a, r, s_n, done):
        self.total_it += 1
        a_n = self.predict_actor(s_n, self.target_actor)
        noise = np.clip(np.random.normal(0, self.policy_noise, a_n.shape), -self.noise_clip, self.noise_clip)
        a_n = np.clip(a_n + noise, -1, 1)
        q1_target = self.predict_q(s_n, a_n, self.target_q1)
        q2_target = self.predict_q(s_n, a_n, self.target_q2)
        target_q = r + self.gamma * min(q1_target, q2_target) * (1 - done)
        if self.total_it % self.policy_delay == 0:
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_q1, self.q1)
            self.soft_update(self.target_q2, self.q2)
```

## 9. 可视化与结果理解

- **双 Q 值对比**：Q₁ 和 Q₂ 应接近，取 min 后更保守
- **Q 值 vs 实际回报**：应逐步对齐
- **策略平滑效果**：Q 值曲面应更平滑
- **延迟更新效果**：Actor 损失更新频率明显低于 Critic

## 10. 模型评估

- **平均回合奖励**：核心性能指标
- **Q 值过估计程度**：Q 估计与实际回报的差距
- **训练稳定性**：奖励曲线的方差
- **与 DDPG 对比**：同等条件下 TD3 应更稳定

## 11. 常见问题与易错点

- **忘记策略延迟**：d=2 即每 2 次 Critic 更新才更新 1 次 Actor
- **目标平滑噪声过大**：σ 应远小于动作范围
- **忘记 clamp 目标动作**：加噪后必须裁剪到合法范围
- **两个 Q 网络用同一个优化器**：应分别用独立优化器
- **Actor 更新时用 min(Q₁,Q₂)**：Actor 应用 Q₁ 的梯度，min 只用于计算目标值

## 12. 学习总结

TD3 通过三个简洁有效的技巧显著提升了 DDPG 的稳定性：双 Q 取 min 减少过估计、延迟更新解耦 Actor-Critic、目标平滑防止 Q 值窄峰。在广告连续出价场景中，TD3 是比 DDPG 更可靠的选择。

## 13. 练习题与思考题（含答案）

**Q1**：为什么 TD3 在计算 Actor 梯度时只用 Q₁ 而不是 min(Q₁,Q₂)？

A1：min 操作在目标值计算中使用是为了减少过估计。Actor 梯度需要通过 Q 值反向传播，使用 Q₁ 的梯度方向更明确。若用 min，梯度可能来自不同的 Q 网络，导致方向不稳定。

**Q2**：目标策略平滑为什么能提升性能？

A2：添加小噪声使目标值在动作邻域内平均化，防止 Q 网络过拟合到特定动作值的窄峰，提高泛化能力。这类似于一种正则化。

**Q3**：TD3 与 SAC 的选择标准是什么？

A3：需要强探索和鲁棒性选 SAC（最大熵+随机策略），需要确定性行为和更快收敛选 TD3。广告出价通常选 TD3 或 SAC，取决于对确定性的需求。

## 14. 学习路径建议

前置知识：DQN → DDPG → Q 值过估计问题
进阶方向：TD3 → SAC（最大熵）→ 离散动作空间扩展 → 广告出价实战
