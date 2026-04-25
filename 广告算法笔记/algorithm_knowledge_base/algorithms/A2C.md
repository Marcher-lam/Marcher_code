# A2C（Advantage Actor-Critic）学习文档

## 1. 算法基础认知

A2C 是策略梯度方法的经典框架，同时学习策略（Actor）和价值函数（Critic），用优势函数降低梯度方差。A3C 是其异步版本，A2C 是同步版本（更简单且效率相当）。在广告系统中可作为 PPO 的轻量替代方案。

## 2. 核心原理

### 优势函数

$$A(s,a) = Q(s,a) - V(s)$$

实际计算中用 TD 误差近似：

$$A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 策略梯度

$$\nabla_\theta J(\theta) = \mathbb{E}_t\left[A(s_t, a_t) \nabla_\theta \log\pi_\theta(a_t|s_t)\right]$$

### 总损失函数

$$L = L_{policy} + c_1 L_{value} - c_2 H(\pi)$$

其中：
- $L_{policy} = -\frac{1}{N}\sum_t A(s_t,a_t)\log\pi(a_t|s_t)$
- $L_{value} = \frac{1}{N}\sum_t (G_t - V(s_t))^2$
- $H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)$（熵正则化，鼓励探索）

## 3. 数学公式与推导

Critic 的价值函数用 TD(0) 更新：

$$V(s_t) \leftarrow V(s_t) + \alpha_c \left[r_t + \gamma V(s_{t+1}) - V(s_t)\right]$$

优势函数也可用 n-step 或 GAE：

$$\hat{A}_t^{(n)} = \sum_{l=0}^{n-1}\gamma^l r_{t+l} + \gamma^n V(s_{t+n}) - V(s_t)$$

策略梯度无偏性证明：$\mathbb{E}[A(s,a)\nabla\log\pi(a|s)] = \nabla J(\pi)$，因为 Critic 提供的基线不引入偏差但降低方差。

## 4. 训练过程讲解

1. 初始化 Actor 网络 π(a|s;θ) 和 Critic 网络 V(s;φ)
2. 并行 N 个 worker，每个 worker：
   - 用当前策略交互收集 (s,a,r,s')
   - 计算优势 A = r + γV(s') - V(s)
3. 汇总所有 worker 数据
4. 更新 Actor：沿 ∇_θ Σ A·log π(a|s) 上升
5. 更新 Critic：min Σ (G_t - V(s))²
6. 重复步骤 2-5

## 5. 应用场景

- 广告冷启动探索策略（轻量替代 PPO）
- 多目标调控的策略训练
- 游戏AI（Atari 等）
- 连续控制（连续版本 A2C）
- 快速原型验证 RL 方案

## 6. 优缺点分析

**优点**：
- 结构简单，易于实现和调试
- 同步并行，GPU 利用率高
- 优势函数降低梯度方差

**缺点**：
- 样本效率低（on-policy）
- 训练不如 PPO 稳定（无裁剪保护）
- 策略更新幅度不可控
- 对超参数敏感

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=7e-4, gamma=0.99, entropy_coef=0.01):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        s = torch.FloatTensor(state)
        probs, value = self.model(s)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns

    def update(self, states, actions, log_probs, returns, values):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)
        values = torch.stack(values)

        advantages = returns - values.detach()
        probs, new_values = self.model(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        actor_loss = -(new_log_probs * advantages).mean()
        critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
        entropy_loss = -entropy.mean()

        loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class SimpleA2C:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.lr = lr
        self.gamma = 0.99
        self.aW1 = np.random.randn(state_dim, 32) * 0.1
        self.aW2 = np.random.randn(32, action_dim) * 0.1
        self.cW1 = np.random.randn(state_dim, 32) * 0.1
        self.cW2 = np.random.randn(32, 1) * 0.1

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, state):
        a_h = np.maximum(0, state @ self.aW1)
        probs = self.softmax(a_h @ self.aW2)
        c_h = np.maximum(0, state @ self.cW1)
        value = (c_h @ self.cW2).item()
        return probs, value, a_h, c_h

    def update(self, state, action, reward, next_state, done):
        probs, value, a_h, c_h = self.forward(state)
        _, next_value, _, _ = self.forward(next_state)
        advantage = reward + self.gamma * next_value * (1 - done) - value
        grad_logp = probs.copy()
        grad_logp[action] -= 1.0
        self.aW2 -= self.lr * advantage * (a_h.reshape(-1, 1) @ grad_logp.reshape(1, -1))
        td_error = value - (reward + self.gamma * next_value * (1 - done))
        grad_cW2 = c_h.reshape(-1, 1) @ np.array([[td_error]])
        self.cW2 -= self.lr * grad_cW2
```

## 9. 可视化与结果理解

- **优势函数分布**：正优势表示动作优于平均，负优势表示劣于平均
- **策略熵**：应逐步降低，从探索到利用
- **价值函数**：应逐步逼近真实回报
- **Actor/Critic 损失**：应分别收敛

## 10. 模型评估

- **平均回合奖励**：衡量整体策略性能
- **策略熵**：探索程度，过低说明过早收敛
- **价值函数误差**：Critic 预测准确性
- **学习曲线稳定性**：奖励方差应逐步减小

## 11. 常见问题与易错点

- **忘记梯度裁剪**：A2C 容易出现梯度爆炸
- **熵系数设置不当**：过大使策略随机，过小导致过早收敛
- **优势函数未归一化**：不同状态的优势尺度差异大
- **并行 worker 数不够**：数据多样性不足
- **学习率过大**：策略震荡不收敛

## 12. 学习总结

A2C 的核心贡献在于将 Actor-Critic 框架以最简洁的形式实现：用优势函数 $A(s,a)=Q-V$ 作为策略梯度的权重，在不引入偏差的前提下显著降低了方差。同步并行（vs A3C 的异步）的设计使得 GPU 利用率更高，训练过程更易复现。

A2C 的关键优势是结构简单、代码量小、调试直观，适合作为强化学习方案的快速验证 baseline。当需要在广告系统中快速上线一个策略梯度方案时，A2C 是首选的轻量替代方案，比 PPO 实现成本低但性能相当。

在知识体系中，A2C 向上衔接 REINFORCE 和策略梯度理论，向下是 PPO（裁剪保护）、SAC（最大熵 off-policy）、DDPG（连续动作）等高级算法的直接基础。掌握 A2C 中优势函数和熵正则化的用法，是理解后续所有 Actor-Critic 变体的关键。

工业实践中需要注意梯度裁剪（clip_grad_norm）和熵系数的调优，前者防止训练崩溃，后者控制探索-利用的平衡。建议先用 A2C 验证环境和 reward 设计是否合理，再切换到 PPO 等更稳定的算法。

## 13. 练习题与思考题（含答案）

**Q1**：为什么使用优势函数 A(s,a) 而不是 Q(s,a) 作为策略梯度的权重？

A1：优势函数 A = Q - V 减去了状态价值的基线，降低了梯度方差。基线不影响梯度无偏性（因为 ∇E[V(s)] = 0），但能显著减小方差。

**Q2**：A2C 与 A3C 的区别是什么？

A2：A3C 使用异步更新（各 worker 独立更新全局网络），A2C 使用同步更新（等所有 worker 完成后统一更新）。实践中 A2C 效率相当且更易实现。

**Q3**：熵正则化的作用是什么？

A3：熵正则化在损失中加入策略熵（取负号即鼓励高熵），防止策略过早收敛到次优确定性策略，保持探索能力。

## 14. 学习路径建议

前置知识：策略梯度 → 价值函数 → TD 学习
进阶方向：A2C → PPO（裁剪保护）→ SAC（最大熵）→ 多智能体 RL
