# PPO（Proximal Policy Optimization）学习文档

## 1. 算法基础认知

PPO 由 OpenAI 于 2017 年提出，是当前最广泛使用的策略梯度算法。它通过裁剪目标函数限制策略更新幅度，兼顾了实现简单性和训练稳定性。在广告系统中被广泛用于出价策略优化、多目标权重调权和冷启动策略学习。

## 2. 核心原理

PPO 的核心思想：每次策略更新不要走太远。引入概率比：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

裁剪目标函数：

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中 ε 通常取 0.1~0.2，Â_t 是优势函数估计，通过 GAE 计算：

$$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

总损失 = Actor 损失 + Critic 损失 - 熵奖励：

$$L = -L^{CLIP} + c_1 L^{VF} - c_2 H(\pi)$$

## 3. 数学公式与推导

策略梯度的基本形式：

$$\nabla_\theta J(\theta) = \mathbb{E}_t\left[\hat{A}_t \nabla_\theta \log\pi_\theta(a_t|s_t)\right]$$

PPO 的裁剪保证了 r_t(θ) 被限制在 [1-ε, 1+ε] 内。当 r_t > 1+ε 且 Â > 0 时，梯度被截断；当 r_t < 1-ε 且 Â < 0 时，梯度也被截断。这防止了过大的策略更新。

广告多目标调控的 MDP 建模：
- 状态：(收入指标, 用户体验指标, 生态指标, 上下文)
- 动作：排序权重 (w₁, w₂, ..., w_M)
- 奖励：r_t = Σ λ_k · f_k(s_t, a_t)

## 4. 训练过程讲解

1. 用当前策略收集一批轨迹 {(s,a,r,s')}
2. 用 GAE 计算每个时间步的优势 Â_t 和回报 G_t
3. 对多个 epoch（通常 3~10 个）重复：
   - 计算 r_t(θ) 和裁剪目标
   - 更新 Actor（最大化 L^CLIP）
   - 更新 Critic（最小化 (V(s) - G_t)²）
4. 丢弃旧数据，收集新轨迹

## 5. 应用场景

- 广告出价策略优化（连续/离散动作空间）
- 多目标排序权重动态调权
- 冷启动探索策略
- 大语言模型 RLHF 对齐
- 机器人控制

## 6. 优缺点分析

**优点**：
- 训练稳定，超参数不太敏感
- 实现简单，调参容易
- 同时支持连续和离散动作空间
- 样本效率优于 A2C

**缺点**：
- 在线策略（on-policy），样本只用一次
- 比 DDPG/SAC 样本效率低
- 多 epoch 更新可能引入偏差
- 训练速度受限于环境交互

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat = self.shared(x)
        probs = torch.softmax(self.actor(feat), dim=-1)
        value = self.critic(feat)
        return probs, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=4):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.mse = nn.MSELoss()

    def select_action(self, state):
        s = torch.FloatTensor(state)
        probs, _ = self.policy(s)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_gae(self, rewards, values, dones, lam=0.95):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (values[t+1] if t+1 < len(values) else 0) * (1-dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1-dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        for _ in range(self.k_epochs):
            probs, values = self.policy(torch.FloatTensor(np.array(states)))
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(torch.LongTensor(actions))
            entropy = dist.entropy()
            ratio = torch.exp(log_probs - torch.FloatTensor(old_log_probs))
            adv = torch.FloatTensor(advantages)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse(values.squeeze(), torch.FloatTensor(returns))
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class SimplePPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, eps_clip=0.2):
        self.eps_clip = eps_clip
        self.lr = lr
        self.W1 = np.random.randn(state_dim, 32) * 0.1
        self.W_actor = np.random.randn(32, action_dim) * 0.1
        self.W_critic = np.random.randn(32, 1) * 0.1

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, state):
        h = np.maximum(0, state @ self.W1)
        probs = self.softmax(h @ self.W_actor)
        value = (h @ self.W_critic).item()
        return probs, value, h

    def ppo_loss(self, ratio, advantage):
        surr1 = ratio * advantage
        surr2 = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        return -min(surr1, surr2)

    def update_step(self, state, action, old_log_prob, advantage, return_g):
        probs, value, h = self.forward(state)
        new_log_prob = np.log(probs[action] + 1e-8)
        ratio = np.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantage
        surr2 = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        ppo_advantage = min(surr1, surr2)
        critic_loss = value - return_g
        grad_logp = probs.copy()
        grad_logp[action] -= 1.0
        grad_W_actor = h.reshape(-1, 1) @ grad_logp.reshape(1, -1)
        self.W_actor -= self.lr * ppo_advantage * grad_W_actor
        grad_critic = h.reshape(-1, 1) @ np.array([[critic_loss]])
        self.W_critic -= self.lr * grad_critic
```

## 9. 可视化与结果理解

- **策略损失曲线**：应为正（因为取负），逐步下降
- **价值损失曲线**：应逐步下降
- **熵曲线**：应逐步降低（从探索到利用）
- **奖励曲线**：应逐步上升
- **概率比 r_t(θ) 分布**：应集中在 1 附近

## 10. 模型评估

- **平均回合奖励**：策略整体性能
- **策略熵**：衡量探索程度
- **KL 散度**：新旧策略的差异，衡量更新幅度
- **价值函数误差**：Critic 预测的准确性

## 11. 常见问题与易错点

- **k_epochs 过大**：同一批数据重复训练过多导致过拟合
- **ε 设置不当**：过小限制过强，过大导致不稳定
- **GAE 的 λ 被忽略**：λ 控制偏差-方差权衡，通常取 0.95
- **忘记标准化优势**：advantages 应做零均值单位方差归一化
- **奖励尺度不当**：大奖励导致策略更新过大

## 12. 学习总结

PPO 是当前最实用的策略梯度算法，通过裁剪机制在性能与稳定性之间取得平衡。它在广告出价、多目标调控中广泛应用，也是 RLHF 的标准算法。相比 A2C 更稳定，相比 SAC/DDPG 更易实现。

## 13. 练习题与思考题（含答案）

**Q1**：PPO 的裁剪机制为什么能防止策略崩溃？

A1：当策略偏离旧策略过多时（r_t 超出 [1-ε,1+ε]），梯度被截断为零，防止一次性更新过大导致策略性能骤降。

**Q2**：GAE 中 λ 的作用是什么？

A2：λ 控制优势估计的偏差-方差权衡。λ=0 时 Â_t = δ_t（低方差高偏差），λ=1 时 Â_t = Σγ^l δ_{t+l}（高方差低偏差）。通常取 0.95。

**Q3**：为什么 PPO 适合广告多目标调权？

A3：广告排序权重是连续变量，且策略需要稳定更新（线上 A/B 测试要求），PPO 的裁剪保证了每次更新幅度可控。

## 14. 学习路径建议

前置知识：策略梯度 → Actor-Critic → TRPO
进阶方向：PPO → PPO + GAE → 多智能体 PPO → RLHF 应用
