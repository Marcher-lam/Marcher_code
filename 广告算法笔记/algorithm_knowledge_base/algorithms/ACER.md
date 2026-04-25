# ACER 学习文档

## 1. 算法基础认知

ACER（Actor-Critic with Experience Replay）由 Wang 等人于 2017 年提出，是一种**离策略（off-policy）的 Actor-Critic 算法**。它解决了策略梯度方法中的核心难题：如何高效利用经验回放中的旧数据进行学习。

核心创新点：
- **截断重要性采样**：控制 off-policy 修正的方差
- **Retrace($\lambda$)**：用于 Critic 的稳定多步 off-policy 回报估计
- **信任域更新**：通过 KL 散度约束限制策略更新幅度

## 2. 核心原理

### 2.1 Off-Policy 策略梯度

行为策略 $\mu$ 采样数据，目标策略 $\pi$ 需要学习。标准重要性采样修正：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\mu}\left[\rho_t \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{Q}^\pi(s_t, a_t)\right]$$

其中 $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\mu(a_t|s_t)}$ 是重要性权重。

### 2.2 截断重要性权重

标准重要性权重方差极大。ACER 使用截断形式：

$$\bar{\rho}_t = \min(c, \rho_t)$$

其中 $c$ 是截断常数（通常 $c=10$）。截断保证了梯度估计的方差有界，同时仍是无偏的（当 $\hat{Q}$ 正确时）。

### 2.3 Retrace($\lambda$) 回报

用于 Critic 更新的多步 off-policy 回报估计：

$$Q^{\text{ret}}(s_t, a_t) = r_t + \gamma \bar{\rho}_{t+1}\left(Q^{\text{ret}}(s_{t+1}, a_{t+1}) - Q(s_{t+1}, a_{t+1})\right) + \gamma V(s_{t+1})$$

其中 $\bar{\rho}_t = \min(1, \rho_t)$。Retrace 是安全的重要性采样方法，保证收缩性（不会发散）。

### 2.4 信任域策略更新

ACER 使用平均策略 $\Phi$（策略的运行平均）作为参考：

$$\tilde{\theta} = \arg\min_\theta \bar{\mathbb{E}}_t\left[\text{KL}(\Phi(\cdot|s_t) \| \pi_\theta(\cdot|s_t))\right]$$

然后用 TRPO 式的约束更新：$\theta \leftarrow \theta + z$，其中 $z$ 由以下优化得到：

$$\max_z \nabla_\theta J(\theta)^T z, \quad \text{s.t.} \quad \frac{1}{2}z^T F z \leq \delta$$

$F$ 是 Fisher 信息矩阵。

## 3. 数学公式与推导

### 离策略策略梯度的推导

从同策略梯度出发：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi}\left[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)\right]$$

引入重要性采样：

$$= \mathbb{E}_{\mu}\left[\frac{\pi_\theta(a|s)}{\mu(a|s)} \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)\right]$$

使用 $\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$ 和截断技巧，ACER 的梯度估计为：

$$\hat{g}_t = \bar{\rho}_t \nabla_\theta \log \pi_\theta(a_t|s_t) \left(Q^{\text{ret}}(s_t, a_t) - V_\phi(s_t)\right)$$

减去 $V_\phi(s_t)$ 作为基线进一步降低方差。

### Retrace 收缩性

Retrace 的关键是 $\bar{\rho}_t = \min(1, \rho_t)$。这个截断保证了：

$$\mathbb{E}_\mu[\bar{\rho}_t | s_t] = \sum_a \min(\pi(a|s_t), \mu(a|s_t)) \leq 1$$

这是保证 $Q^{\text{ret}}$ 收敛到 $Q^\pi$ 的充分条件。

## 4. 训练过程讲解

1. **初始化**：策略网络 $\pi_\theta$、价值网络 $V_\phi$、平均策略 $\Phi$、经验回放缓冲区
2. **与环境交互**（on-policy 部分）：
   - 按 $\pi_\theta$ 选择动作，存储转移 $(s, a, r, s')$
3. **从回放缓冲区采样** mini-batch
4. **计算 Retrace 回报** $Q^{\text{ret}}$：从后向前递推
5. **更新 Critic**：$L_\phi = (V_\phi(s_t) - Q^{\text{ret}}(s_t, a_t))^2$
6. **计算策略梯度**：截断重要性权重 + 基线
7. **信任域更新 Actor**：解约束优化问题
8. **更新平均策略**：$\Phi \leftarrow \alpha \Phi + (1-\alpha)\pi_\theta$
9. **重复 2-8**

## 5. 应用场景

- **连续控制**：MuJoCo、机器人操作
- **离散决策**：Atari 游戏的高效训练
- **推荐系统**：利用历史交互数据优化推荐策略
- **广告系统**：利用旧日志数据进行策略优化（off-policy 天然契合）
- **自动驾驶**：利用模拟器历史数据

## 6. 优缺点分析

**优点**：
- 样本效率高（利用经验回放）
- 训练稳定（信任域 + Retrace）
- 同时支持离散和连续动作空间
- Off-policy 能力可利用历史数据

**缺点**：
- 实现复杂（Retrace、信任域、截断重要性采样）
- 超参数多（$c$、$\lambda$、$\delta$、$\alpha$ 等）
- 计算开销较大（需要维护多个网络和平均策略）
- 对回放缓冲区大小敏感

## 7. 调库实现（Python）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ACERNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mu_logprob):
        self.buffer.append((state, action, reward, next_state, done, mu_logprob))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, mu_logprobs = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), np.array(mu_logprobs))

    def __len__(self):
        return len(self.buffer)
```

## 8. 手工代码实现

```python
class ACERAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 c=10.0, lam=0.95, trust_region_delta=1.0):
        self.net = ACERNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.avg_net = ACERNetwork(state_dim, action_dim)
        self.avg_net.load_state_dict(self.net.state_dict())
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.c = c
        self.lam = lam
        self.delta = trust_region_delta
        self.action_dim = action_dim

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = self.net(state_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def compute_retrace(self, rewards, values, ratios, dones):
        Q_ret = values[-1]
        Q_rets = []
        for t in reversed(range(len(rewards))):
            Q_ret = rewards[t] + self.gamma * Q_ret * (1 - dones[t])
            rho_bar = min(1.0, ratios[t])
            Q_ret = values[t] + rho_bar * (Q_ret - values[t])
            Q_rets.insert(0, Q_ret)
        return Q_rets

    def update(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones, mu_lps = self.buffer.sample(batch_size)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        dones_t = torch.FloatTensor(dones)

        logits, values = self.net(states_t)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            avg_logits, _ = self.avg_net(states_t)
            avg_probs = F.softmax(avg_logits, dim=-1)
            mu_probs_t = torch.FloatTensor(np.exp(mu_lps))
            pi_selected = torch.gather(probs, 1, actions_t.unsqueeze(1)).squeeze(1)
            ratios = pi_selected / (mu_probs_t + 1e-8)
            trunc_ratios = torch.clamp(ratios, max=self.c)

        values = values.squeeze()
        selected_log_probs = torch.gather(log_probs, 1, actions_t.unsqueeze(1)).squeeze(1)

        Q_rets = self.compute_retrace(
            rewards.tolist(), values.detach().tolist(),
            ratios.detach().tolist(), dones.tolist()
        )
        Q_rets_t = torch.FloatTensor(Q_rets)

        advantage = Q_rets_t - values.detach()
        policy_loss = -(trunc_ratios * selected_log_probs * advantage).mean()

        kl = (avg_probs * (torch.log(avg_probs + 1e-8) - log_probs)).sum(dim=-1).mean()
        value_loss = F.mse_loss(values, Q_rets_t.detach())

        loss = policy_loss + 0.5 * value_loss + 10.0 * kl
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        for p, avg_p in zip(self.net.parameters(), self.avg_net.parameters()):
            avg_p.data.copy_(0.99 * avg_p.data + 0.01 * p.data)
```

## 9. 可视化与结果理解

ACER 的关键可视化：
- **训练回报曲线**：应比 on-policy 方法（REINFORCE、A2C）收敛更快
- **重要性权重分布**：$\rho_t$ 的直方图，截断后应集中在 $[0, c]$ 区间
- **KL 散度**：策略更新前后的 KL，应在信任域 $\delta$ 以内
- **Retrace Q 值**：与真实 Q 值的对比，验证 off-policy 修正的准确性

## 10. 模型评估

- **平均回报**：每 100 episode 的滑动平均
- **样本效率**：达到指定回报所需的环境交互步数（ACER 的核心优势）
- **KL 散度**：策略更新幅度，监控信任域约束是否有效
- **Q 值误差**：$|Q^{\text{ret}} - Q_{\text{true}}|$，衡量 Critic 质量

## 11. 常见问题与易错点

- **Retrace 递推方向**：必须从后向前递推，不能正向
- **重要性权重计算**：$\rho = \pi/\mu$，分母是行为策略而非目标策略
- **截断值 $c$ 的选择**：太大则方差高，太小则偏差大，通常 $c \in [5, 20]$
- **平均策略更新**：$\Phi$ 的更新速率影响信任域的松紧
- **回放缓冲区大小**：太大会导致行为策略与目标策略差距过大
- **数值稳定性**：重要性权重可能导致除零或溢出，需要加 epsilon

## 12. 学习总结

ACER 的核心贡献在于系统性地解决了策略梯度方法中"如何安全高效地利用旧数据"这一难题。它通过三项关键技术创新——截断重要性权重（控制方差）、Retrace($\lambda$) 回报（保证 Critic 收敛的 off-policy 多步估计）、信任域策略更新（限制策略变化幅度）——将经验回放与 on-policy 策略梯度无缝融合。

ACER 最突出的优势是样本效率高，能反复利用历史交互数据，特别适合数据获取成本高的场景（如广告系统中的真实流量实验）。当需要在已有日志数据上训练或微调策略时，ACER 的 off-policy 能力使其比 PPO、A2C 等 on-policy 方法更有优势。

在知识体系中，ACER 可视为 DQN（经验回放）和 PPO（信任域）思想的统一，同时与 SARSA、Q-learning 等时序差分方法有深厚的理论联系。理解 ACER 后再学习 SAC、TD3 等现代 off-policy 算法会更加顺畅。

工业实践中 ACER 的实现复杂度较高，需同时维护策略网络、价值网络和平均策略网络。如果项目资源有限，可优先考虑 SAC 作为更简洁的 off-policy 替代方案。

## 13. 练习题与思考题

**Q1**：为什么 ACER 使用 $\min(c, \rho_t)$ 而非直接使用 $\rho_t$？

> **答案**：当 $\rho_t$ 很大时（行为策略与目标策略差异大），梯度估计方差极大，可能导致训练不稳定。截断保证了方差有界。截断引入的偏差是安全的，因为截断等价于忽略那些 $\rho > c$ 的样本的贡献。

**Q2**：Retrace 中为什么使用 $\bar{\rho}_t = \min(1, \rho_t)$ 而非 $\min(c, \rho_t)$？

> **答案**：Critic 更新需要更强的收缩性保证。$\bar{\rho}_t \leq 1$ 保证了 $\mathbb{E}_\mu[\bar{\rho}_t] \leq 1$，这是 Retrace 收敛到 $Q^\pi$ 的必要条件。使用 $c > 1$ 会破坏这个收缩性。

**Q3**：ACER 与 PPO 都限制了策略更新幅度，方式有何不同？

> **答案**：PPO 通过 clipped surrogate objective 直接在损失函数中限制 $\rho_t$ 的范围（$[1-\epsilon, 1+\epsilon]$）。ACER 通过信任域（KL 约束）间接限制，并维护一个平均策略 $\Phi$ 作为参考点。

## 14. 学习路径建议

1. **前置知识**：策略梯度（REINFORCE）、Actor-Critic、重要性采样
2. **本节掌握**：截断重要性权重、Retrace、信任域更新
3. **进阶方向**：
   - SAC（Soft Actor-Critic）：基于最大熵的 off-policy 方法
   - TD3：Twin Delayed DDPG
   - Off-Policy PPO 变体
4. **后续学习**：将 ACER 的 off-policy 技巧应用于实际广告/推荐系统
