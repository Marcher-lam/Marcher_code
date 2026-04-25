# DQN（Deep Q-Network）学习文档

## 1. 算法基础认知

DQN 将深度神经网络与 Q-learning 结合，用神经网络近似 Q 函数，从而处理高维连续状态空间。它是强化学习从表格方法走向深度学习的里程碑，由 DeepMind 于 2013 年提出。

在广告系统中，DQN 被用于自动出价策略、冷启动探索档位选择、离散动作空间的竞价决策等场景。

## 2. 核心原理

Q 函数表示在状态 s 执行动作 a 后，遵循策略 π 所获得的期望累计回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t=s, a_t=a\right]$$

最优 Q 函数满足贝尔曼方程：

$$Q^*(s, a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

DQN 的三项关键技术：
- **经验回放**：存储转移样本 (s,a,r,s')，随机采样打破相关性
- **目标网络**：独立的目标 Q 网络，定期同步，稳定训练
- **ε-greedy 探索**：以概率 ε 随机选动作，其余选 Q 值最大动作

## 3. 数学公式与推导

DQN 的损失函数：

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

其中 θ 是在线网络参数，θ⁻ 是目标网络参数，每 C 步同步：θ⁻ ← θ。

梯度：

$$\nabla_\theta L = -2\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right) \nabla_\theta Q(s, a; \theta)$$

广告出价的奖励设计：

$$r_t = \alpha \cdot \text{Conversions}_t - \beta \cdot \max(0, \text{CPA}_t - \text{CPA}_{target})$$

## 4. 训练过程讲解

1. 初始化 Q 网络 Q(s,a;θ) 和目标网络 Q(s,a;θ⁻)，经验池 D
2. 每个回合：用 ε-greedy 选动作，执行后存 (s,a,r,s') 到 D
3. 从 D 随机采样 mini-batch，计算目标值 y = r + γ max Q(s',·;θ⁻)
4. 对 Q 网络做梯度下降：min (y - Q(s,a;θ))²
5. 每 C 步更新目标网络：θ⁻ ← θ
6. 逐步衰减 ε

## 5. 应用场景

- 广告自动出价（离散出价档位选择）
- 冷启动探索策略
- 推荐系统中的列表排序
- 游戏AI（Atari 等）
- 任何高维状态 + 离散动作空间的决策问题

## 6. 优缺点分析

**优点**：
- 能处理高维连续状态空间
- 经验回放提高样本效率
- 目标网络稳定训练

**缺点**：
- 仅适用于离散动作空间
- Q 值过估计问题
- 对超参数敏感（学习率、ε 衰减等）
- 训练不够稳定，可能发散

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.3
        self.batch_size = 64
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(state)).argmax().item()

    def store(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_n, d = zip(*batch)
        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s_n = torch.FloatTensor(np.array(s_n))
        d = torch.FloatTensor(d)
        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_target = r + self.gamma * self.target_net(s_n).max(1)[0] * (1 - d)
        loss = nn.MSELoss()(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class DQNFromScratch:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        np.random.seed(42)
        self.d_in, self.d_out = state_dim, action_dim
        self.gamma = gamma
        self.lr = lr
        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / 64)
        self.W1 = np.random.randn(state_dim, 64) * scale1
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 64) * scale2
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, action_dim) * scale2
        self.b3 = np.zeros(action_dim)

    def _forward(self, x, W1, b1, W2, b2, W3, b3):
        h1 = np.maximum(0, x @ W1 + b1)
        h2 = np.maximum(0, h1 @ W2 + b2)
        return h1, h2, h2 @ W3 + b3

    def predict(self, x):
        _, _, q = self._forward(x, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        return q

    def update(self, state, action, reward, next_state, done):
        h1, h2, q = self._forward(state, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        next_q = self.predict(next_state)
        target = reward + self.gamma * np.max(next_q) * (1 - done)
        error = q[action] - target
        grad_q = np.zeros(self.d_out)
        grad_q[action] = error
        dW3 = h2.reshape(-1, 1) @ grad_q.reshape(1, -1)
        db3 = grad_q
        grad_h2 = grad_q @ self.W3.T * (h2 > 0)
        dW2 = h1.reshape(-1, 1) @ grad_h2.reshape(1, -1)
        db2 = grad_h2
        grad_h1 = grad_h2 @ self.W2.T * (h1 > 0)
        dW1 = state.reshape(-1, 1) @ grad_h1.reshape(1, -1)
        db1 = grad_h1
        for param, grad in [(self.W3, dW3), (self.b3, db3), (self.W2, dW2),
                            (self.b2, db2), (self.W1, dW1), (self.b1, db1)]:
            param -= self.lr * grad
```

## 9. 可视化与结果理解

训练过程中应关注：
- **累计奖励曲线**：应逐步上升并趋于稳定
- **Q 值分布**：观察不同动作的 Q 值变化
- **ε 衰减曲线**：从探索到利用的过渡
- **损失曲线**：应逐渐下降

典型广告出价场景：初始 ε=0.3，训练 1000 回合后 ε 衰减至 0.05，出价策略趋于最优。

## 10. 模型评估

- **累计奖励**：回合总奖励的平均值与方差
- **出价 ROI**：转化价值 / 出价成本
- **CPA 达标率**：实际 CPA 在目标 CPA 以下的比例
- **收敛速度**：奖励稳定所需的训练回合数

## 11. 常见问题与易错点

- **忘记更新目标网络**：目标网络必须定期同步，否则训练不稳定
- **ε 衰减过快**：导致探索不足，陷入局部最优
- **经验池太小**：样本多样性不足；太大则旧样本过时
- **奖励缩放不当**：奖励量级差异大时网络难以学习
- **状态归一化遗漏**：输入特征量级差异大时影响收敛

## 12. 学习总结

DQN 是深度强化学习的基础算法，通过神经网络近似 Q 函数，结合经验回放和目标网络实现稳定训练。它适用于离散动作空间，是广告出价 RL 应用的起点。进阶方向包括 Double DQN、Dueling DQN，以及连续动作空间的 DDPG/TD3/SAC。

## 13. 练习题与思考题（含答案）

**Q1**：DQN 为什么要使用目标网络？

A1：目标网络提供稳定的目标值 r + γ max Q(s',·;θ⁻)，避免在线网络"追着自己跑"导致的震荡和发散。

**Q2**：广告出价中，状态、动作、奖励分别如何设计？

A2：状态 = (预算消耗率, 时间进度, 当前CPA, 转化率)；动作 = 出价调整因子档位 [0.5,0.8,1.0,1.2,1.5]；奖励 = α·转化量 - β·成本超标惩罚。

**Q3**：DQN 与 Q-learning 的核心区别是什么？

A3：Q-learning 用表格存储 Q 值，DQN 用神经网络近似 Q 值。DQN 能处理高维/连续状态空间，Q-learning 只能处理离散小规模状态。

## 14. 学习路径建议

前置知识：Q-learning → 神经网络基础 → MDP
进阶方向：Double DQN → Dueling DQN → Rainbow → DDPG（连续动作空间）→ TD3/SAC
