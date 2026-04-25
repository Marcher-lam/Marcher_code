# DDPG（Deep Deterministic Policy Gradient）学习文档

## 1. 算法基础认知

DDPG 结合了 DQN 的经验回放与目标网络机制，以及 Actor-Critic 架构，能够在连续动作空间中学习确定性策略。由 Lillicrap 等于 2015 年提出，是 DQN 向连续动作空间的扩展。

在广告系统中，DDPG 被用于连续出价调整（如 Bid = BaseBid × λ，λ ∈ [0.3, 3.0]），阿里在智能出价中有大量实践。

## 2. 核心原理

DDPG 使用四个网络：
- **Actor μ(s|θ^μ)**：输入状态，输出确定性动作
- **Critic Q(s,a|θ^Q)**：输入状态和动作，输出 Q 值
- **目标 Actor μ'(s|θ^μ')** 和 **目标 Critic Q'(s,a|θ^Q')**：缓慢跟踪主网络

Actor 通过 Critic 的梯度更新方向：

$$\nabla_{\theta^\mu} J \approx \frac{1}{N} \sum_i \nabla_a Q(s,a|\theta^Q)|_{s=s_i,a=\mu(s_i)} \cdot \nabla_{\theta^\mu} \mu(s|\theta^\mu)|_{s_i}$$

Critic 损失函数：

$$L = \frac{1}{N}\sum_i \left(y_i - Q(s_i, a_i|\theta^Q)\right)^2$$

目标值使用软更新：

$$y_i = r_i + \gamma Q'(s_{i+1}, \mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})$$

## 3. 数学公式与推导

软更新规则：

$$\theta' \leftarrow \tau \theta + (1 - \tau)\theta', \quad \tau \ll 1$$

Critic 梯度推导：

$$\frac{\partial L}{\partial \theta^Q} = -\frac{2}{N}\sum_i \left(y_i - Q(s_i,a_i|\theta^Q)\right) \nabla_{\theta^Q} Q(s_i,a_i|\theta^Q)$$

Actor 策略梯度：链式法则展开，先对动作求 Q 值梯度，再对 Actor 参数求动作梯度。

广告出价的奖励函数：

$$r_t = \alpha \cdot \text{Value}_t - \beta \cdot \max(0, \text{Cost}_t - \text{Budget}_t)$$

## 4. 训练过程讲解

1. 初始化 Actor、Critic 及其目标网络，经验回放池
2. 每步：用 Actor 选动作 + OU 噪声探索，执行后存 (s,a,r,s')
3. 从回放池采样 mini-batch
4. 用目标网络计算 TD 目标 y
5. 更新 Critic：最小化 (y - Q(s,a))²
6. 更新 Actor：沿 ∇_a Q · ∇_θ μ 方向
7. 软更新目标网络

## 5. 应用场景

- 广告连续出价调整（λ 连续调节）
- 机器人连续控制
- 自动驾驶控制
- 资源分配与调度
- 金融交易信号

## 6. 优缺点分析

**优点**：
- 适用于连续动作空间
- 确定性策略，样本效率较高
- 结合 DQN 稳定技巧

**缺点**：
- Q 值过估计导致策略次优
- 对超参数敏感（τ、噪声等）
- 训练不稳定，容易发散
- 仅适用于连续动作空间

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Sigmoid()
        )
        self.min_a, self.max_a = 0.3, max_action

    def forward(self, x):
        return self.min_a + self.net(x) * (self.max_a - self.min_a)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=3.0):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.a_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=100000)
        self.gamma = 0.99
        self.tau = 0.005
        self.max_action = max_action

    def select_action(self, state, noise=0.1):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s).squeeze(0).numpy()
        a = a + np.random.normal(0, noise, size=a.shape)
        return np.clip(a, 0.3, self.max_action)

    def store(self, *transition):
        self.buffer.append(transition)

    def _soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_n, d = map(lambda x: torch.FloatTensor(np.array(x)), zip(*batch))
        with torch.no_grad():
            target_q = r + self.gamma * self.critic_target(s_n, self.actor_target(s_n)) * (1 - d)
        critic_loss = nn.MSELoss()(self.critic(s, a), target_q)
        self.c_opt.zero_grad()
        critic_loss.backward()
        self.c_opt.step()
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.a_opt.zero_grad()
        actor_loss.backward()
        self.a_opt.step()
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class SimpleDDPG:
    def __init__(self, state_dim, action_dim, lr_a=0.0001, lr_c=0.001, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.params = {}
        for prefix in ['a', 'c']:
            self.params[f'{prefix}W1'] = np.random.randn(state_dim, 32) * 0.1
            self.params[f'{prefix}b1'] = np.zeros(32)
            self.params[f'{prefix}W2'] = np.random.randn(32, 32) * 0.1
            self.params[f'{prefix}b2'] = np.zeros(32)
        self.params['aW3'] = np.random.randn(32, action_dim) * 0.1
        self.params['ab3'] = np.zeros(action_dim)
        self.params['cW3'] = np.random.randn(32 + action_dim, 1) * 0.1
        self.params['cb3'] = np.zeros(1)
        self.target_params = {k: v.copy() for k, v in self.params.items()}

    def relu(self, x):
        return np.maximum(0, x)

    def actor_forward(self, s, p=None):
        p = p or self.params
        h1 = self.relu(s @ p['aW1'] + p['ab1'])
        h2 = self.relu(h1 @ p['aW2'] + p['ab2'])
        out = h2 @ p['aW3'] + p['ab3']
        return h1, h2, 0.3 + 2.7 / (1.0 + np.exp(-out))

    def critic_forward(self, s, a, p=None):
        p = p or self.params
        h1 = self.relu(s @ p['cW1'] + p['cb1'])
        h2 = self.relu(h1 @ p['cW2'] + p['cb2'])
        inp = np.concatenate([h2, a])
        return h1, h2, inp @ p['cW3'] + p['cb3']

    def soft_update(self):
        for k in self.params:
            self.target_params[k] = self.tau * self.params[k] + (1 - self.tau) * self.target_params[k]
```

## 9. 可视化与结果理解

- **出价因子变化曲线**：观察 λ 是否在合理范围内波动并逐步收敛
- **Q 值曲线**：Critic 估计值应逐步贴近真实回报
- **奖励曲线**：应逐步上升
- **Actor 损失**：应为负且逐步减小（更负表示 Critic 评估的动作价值更高）

## 10. 模型评估

- **平均回合奖励**：衡量策略整体表现
- **出价效率**：转化量 / 总花费
- **CPA 达标率**：最终 CPA 在目标范围内的比例
- **策略稳定性**：连续回合奖励的方差

## 11. 常见问题与易错点

- **Q 值过估计**：Critic 过度乐观导致 Actor 学到次优策略，TD3 通过双 Q 网络缓解
- **噪声衰减不当**：探索噪声过大导致策略不收敛
- **τ 设置过大**：目标网络跟踪过快导致不稳定
- **Actor 与 Critic 学习率不匹配**：通常 Critic 学习率大于 Actor
- **忘记梯度裁剪**：可能导致梯度爆炸

## 12. 学习总结

DDPG 是广告连续出价的主流 RL 算法，通过 Actor-Critic 架构处理连续动作空间。它继承了 DQN 的经验回放和目标网络，并用软更新替代硬更新。但 DDPG 存在 Q 值过估计和训练不稳定问题，TD3 和 SAC 是其改进版本。

## 13. 练习题与思考题（含答案）

**Q1**：DDPG 与 DQN 的核心区别是什么？

A1：DQN 只适用于离散动作空间，输出各动作的 Q 值；DDPG 通过 Actor 网络直接输出连续动作，Critic 评估连续动作的 Q 值。

**Q2**：为什么 DDPG 使用软更新而不是硬更新？

A2：软更新 θ' ← τθ + (1-τ)θ' 使目标网络缓慢跟踪主网络，保证目标值平稳变化，避免剧烈震荡。

**Q3**：广告出价中 DDPG 的动作空间如何设计？

A3：动作 = 出价调整因子 λ ∈ [0.3, 3.0]，最终出价 = BaseBid × λ。Actor 输出经过 Sigmoid 映射到 [0.3, 3.0]。

## 14. 学习路径建议

前置知识：DQN → Actor-Critic → 确定性策略梯度
进阶方向：TD3（双Q+延迟更新）→ SAC（最大熵）→ 广告出价实际应用
