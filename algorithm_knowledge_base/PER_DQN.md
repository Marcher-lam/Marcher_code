# PER DQN (Prioritized Experience Replay) 学习文档

> 来源线索：本节内容根据原书第8章8.4节关于"PER DQN算法"的相关章节整理、扩展与教学化改写。

> 按TD误差优先级采样经验，让智能体优先学习"信息量大"的样本。

## 1. 算法基础认知

**一句话定义**：PER DQN 根据样本的 TD 误差赋予优先级，优先采样高 TD 误差（信息量大）的经验。

**直觉类比**：普通经验回放像"随机翻错题本"复习，PER 像按"错题严重程度"排序复习——标注红色（高 TD 误差）的题多看几遍，绿色的少看。更具体地说，就像考前复习时，你不会平均地看每一页笔记，而是重点看那些你答错过的、或者理解还不到位的题目。PER 就是让强化学习智能体也这样做。

**历史背景**：优先经验回放（PER）由 Schaul 等人在 2015 年提出，通过 SumTree 数据结构高效实现。是对 DQN 经验回放机制的改进。PER 的灵感来源于教育学中的"间隔重复"理论——人们在学习时，应该更多地复习那些掌握不牢固的知识点，而不是平均地复习所有内容。

**算法定位**：PER DQN 是 DQN 的经验回放改进，属于基于价值的、异策略的深度强化学习算法，适用于离散动作空间。在 DQN 的各种改进中，PER 专注于提升数据利用效率——通过优先采样"信息量大"的样本来加速学习。PER 与其他 DQN 改进（Double、Dueling、Noisy）正交，可以任意组合使用。

**前置知识**：DQN、经验回放、TD 误差、重要性采样。建议先理解标准 DQN 的经验回放机制（为什么用均匀采样）和 TD 误差的含义（它衡量了什么），再学习 PER 的改进思路。

**PER 的核心价值**：标准 DQN 的经验回放使用均匀采样——每个样本被选中的概率相同。但不同样本的信息量差异很大：TD 误差大的样本（Critic 预测与实际差距大）包含更多学习信号，应该被更频繁地采样。PER 就是让智能体"优先复习错题"——TD 误差越大的经验越重要，应该被更频繁地回放。这就像考前复习不应该均匀地看每一页笔记，而应该重点看那些你答错的题目。

**优先级经验回放的核心动机**：在标准经验回放中，所有转移以相同概率被采样。但实际上，不同转移对学习的价值差异巨大——一个导致游戏结束的关键错误转移（TD误差大）比一个平淡无奇的常规转移（td误差接近0）有价值得多。PER的核心思想是将采样概率与TD误差挂钩：td误差大的转移被更频繁地采样，因为它们代表了"网络当前最需要学习的内容"。这就像一个学生应该把更多时间花在做错的题上，而不是平均地复习所有题目。在稀疏奖励环境中（如Montezuma's Revenge），PER的效果特别显著——大部分转移的td误差为0（因为奖励为0且Q值没有更新），只有少数关键转移（获得奖励或到达新区域）包含有用信息，PER能自动聚焦于这些关键转移。

## 2. 核心原理

### 优先级计算

每个经验的优先级由其 TD 误差决定：

$$p_i = |\delta_i| + \epsilon$$

其中 $\delta_i$ 是 TD 误差，$\epsilon$ 是防止优先级为 0 的小常数（如 $10^{-2}$）。

TD 误差越大，说明当前网络对该样本的预测与实际偏差越大，该样本包含更多"尚未学好"的信息。

### 采样概率

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

$\alpha$ 控制优先级影响程度：
- $\alpha=0$：均匀采样（退化为普通经验回放）
- $\alpha=1$：完全按优先级采样

### 重要性采样权重

优先采样改变了数据分布，引入偏差。为纠正偏差，使用重要性采样权重：

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

$\beta$ 从初始值逐渐增大到 1（热偏置），$\beta=1$ 时完全纠正偏差。

### SumTree 结构

二叉树结构，每个父节点 = 左子节点 + 右子节点。叶子节点存优先级，根节点为优先级总和。采样时在 $[0, \text{total}]$ 均匀采样，$O(\log N)$ 时间复杂度。

```
            [总优先级]
           /          \
      [左半和]      [右半和]
      /    \        /    \
   [p0]  [p1]   [p2]  [p3]   ← 叶子节点存优先级
```

**深入理解**：理解核心原理的关键是把握为什么这样设计而非仅仅怎么实现。每一个设计决策背后都有明确的数学动机或实践经验支撑。建议在学习时多问自己如果不用这个设计会怎样，通过反面思考加深理解。

### SumTree数据结构的详细说明

SumTree是一种特殊的二叉树，每个叶节点存储一个转移的优先级，每个内部节点存储其子节点优先级之和。它的关键性质是：(1) **O(log n)采样**：从[0, total_priority]中均匀采样一个值s，然后从根节点开始向下搜索——如果s小于左子节点的值，进入左子树；否则减去左子节点的值，进入右子树——直到到达叶节点。这保证了优先级越大的叶节点被采样的概率越高。(2) **O(log n)更新**：当某个转移的优先级改变时，只需更新从该叶节点到根节点路径上的所有内部节点（共log n个）。(3) **空间效率**：对于n个转移，SumTree有2n-1个节点，空间复杂度为O(n)。

**复杂度对比**：朴素实现每次采样需要O(n)（遍历所有转移计算累积优先级），而SumTree只需O(log n)。当回放缓冲区大小为100万时，SumTree的采样速度约为朴素实现的20倍。

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $p_i$ | 第 $i$ 个样本的优先级 |
| $\delta_i$ | 第 $i$ 个样本的 TD 误差 |
| $\alpha$ | 优先级指数，控制优先采样强度 |
| $\beta$ | 重要性采样指数，控制偏差纠正强度 |
| $N$ | 回放池中样本总数 |

### 损失函数（加入重要性采样权重）

$$L(\theta) = \frac{1}{N}\sum_i w_i (y_i - Q_\theta(s_i, a_i))^2$$

### TD 误差更新优先级

$$\delta_i = r_i + \gamma \max_{a'} Q_{\hat\theta}(s'_i, a') - Q_\theta(s_i, a_i)$$

每次更新后，用新的 TD 误差更新对应样本的优先级 $p_i = |\delta_i| + \epsilon$。

### 推导：为什么需要重要性采样

均匀采样时 $\mathbb{E}_{\text{uniform}}[L] = L$，但优先采样时 $\mathbb{E}_{\text{prioritized}}[L] \neq L$。引入 $w_i = (\frac{1}{N \cdot P(i)})^\beta$ 后：

$$\mathbb{E}_{\text{prioritized}}[w_i \cdot L_i] = \sum_i P(i) \cdot \frac{1}{(N \cdot P(i))^\beta} \cdot L_i$$

当 $\beta=1$ 时：$= \frac{1}{N}\sum_i L_i = \mathbb{E}_{\text{uniform}}[L]$，完全无偏。

### 优先级计算的完整数学推导

PER使用比例优先级（proportional priority）或基于秩的优先级（rank-based priority）两种方案。

**比例优先级**：$p_i = |\delta_i| + \epsilon$，其中 $\epsilon$ 是小常数（如1e-6）防止零优先级导致的不采样。采样概率为 $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$，其中 $\alpha \in [0,1]$ 控制优先级的影响程度。$\alpha=0$ 退化为均匀采样，$\alpha=1$ 是纯优先级采样。

**重要性采样权重**：由于PER改变了采样分布，引入了偏差。为修正这个偏差，使用重要性采样权重：$w_i = (\frac{1}{N} \cdot \frac{1}{P(i)})^\beta$。当 $\beta=1$ 时完全修正偏差。实践中通常从 $\beta_0 < 1$（如0.4）线性增长到1，在训练初期允许一定偏差以换取更快的收敛速度。权重还需要归一化：$w_i = \frac{w_i}{\max_j w_j}$，防止个别高优先级样本的梯度过大。

## 4. 训练过程讲解

### 数据预处理

- 与 DQN 相同：状态归一化、帧堆叠等
- 新样本入池时：赋予当前最大优先级（确保新样本至少被采样一次）

### 参数初始化

- SumTree 所有节点初始为 0
- 新样本优先级初始为当前最大值（如果没有样本，设为 1.0）
- $\alpha$ 通常初始化为 0.6，$\beta$ 初始化为 0.4

### 迭代过程

1. 用 $\varepsilon$-greedy 与环境交互，收集 $(s, a, r, s', done)$
2. 用最大优先级插入 SumTree
3. 用优先采样从 SumTree 取出一个 batch
4. 前向传播计算 Q 值和 TD 误差
5. 用重要性采样权重加权计算损失
6. 反向传播更新网络
7. 用新 TD 误差更新采样样本的优先级
8. 逐步增大 $\beta$（通常每步增加一个小常数直到 1.0）

### 收敛条件

- 回合奖励连续 N 个回合不再上升
- 优先级分布逐渐趋于均匀（说明 TD 误差整体减小）

### 超参数表

| 参数 | 作用 | 推荐范围 | 默认 |
|------|------|----------|------|
| $\alpha$ | 优先级指数 | 0.4~0.8 | 0.6 |
| $\beta$ | 重要性采样指数（初始） | 0.3~0.5 | 0.4 |
| $\beta$_step | $\beta$ 每步增量 | 1e-4~1e-3 | 0.0001 |
| $\epsilon$ | 最小优先级 | 1e-3~1e-2 | 0.01 |
| lr | 学习率 | 1e-4~1e-3 | 1e-3 |
| $\gamma$ | 折扣因子 | 0.95~0.99 | 0.99 |
| buffer_size | 回放池大小 | 1e4~1e6 | 50000 |
| batch_size | 批量大小 | 32~128 | 64 |

**训练技巧总结**：训练深度强化学习算法时，最重要的是先确保基础流程能跑通（在简单环境上验证），再逐步调整超参数。建议使用固定的随机种子确保实验可复现，至少运行3到5个不同种子取平均来评估算法性能。

PER在训练初期由于TD误差普遍较大（Q值随机初始化），几乎所有转移都被认为是"重要的"，采样接近均匀。随着训练进行，大部分转移的TD误差趋于0，只有少数关键转移保持高优先级，PER的优势才逐渐体现。

## 5. 应用场景

### 1. 稀疏奖励环境
在大多数经验都是"平淡无奇"（TD 误差小）的场景中，PER 能聚焦于少量有信息量的关键经验。例如在迷宫中，大部分步数奖励为 0，只有到达终点的经验包含有用信息。**为什么适合**：稀疏奖励环境中，99% 的经验 TD 误差接近 0（对策略改进没有帮助），只有 1% 的"关键转折点"经验包含有用信息。PER 能自动识别并重点学习这些关键经验，避免了均匀采样导致的"大海捞针"问题。

### 2. 成功经验稀少的任务
当成功完成任务的经验很少时（如 Montezuma's Revenge），PER 确保这些珍贵经验被反复学习，而不是被大量失败经验淹没。**为什么适合**：成功的经验通常有较大的 TD 误差（因为之前的策略对这些状态估计很差），PER 会自然地赋予它们高优先级。在 Montezuma's Revenge 这类 Atari 游戏中，PER 可以将学习速度提高数倍。

### 3. 在线学习场景
在持续与环境交互的过程中，PER 能快速从最近的"惊讶"样本中学习，加速策略改进。**为什么适合**：当环境发生变化或者策略发生改进时，之前"正确"的经验可能变得"错误"（TD 误差增大），PER 会自动增加这些样本的采样频率，帮助网络快速适应。

### 不适用场景
- 经验信息量均匀分布的简单任务（此时优先采样无优势，反而增加计算开销）
- 回放池很小时（SumTree 的优势不明显，建议回放池至少 10000 个样本）
- 环境高度随机、TD 误差主要由噪声而非信息量决定的场景（PER 可能过度关注噪声）

**应用选择指南**：选择算法时，首先判断动作空间类型（离散用DQN系列，连续用DDPG/TD3/SAC），其次判断样本效率需求（高用异策略方法，低用同策略方法），最后判断稳定性需求（高用PPO/TD3）。

### 5.4 自动驾驶决策

自动驾驶中的决策模块可以使用PER-DQN来优化变道、跟车等行为。状态是周围车辆的位置、速度、加速度（来自传感器），动作是加速、减速、变道，奖励是安全性和通行效率的加权和。**为什么适合PER**：驾驶场景中的大部分时间都是安全的常规行驶（TD误差小），但偶尔会出现需要紧急应对的情况（如前车急刹车、行人突然出现），这些"关键时刻"的转移对学习安全驾驶策略至关重要。PER能自动识别并重点学习这些关键经验，避免被大量"平淡"经验淹没。

## 6. 优缺点分析

### 优点

1. **学习效率提升**：优先学习高信息量样本，减少无效重复。成立条件：经验的信息量分布不均匀。在稀疏奖励环境中，PER 可以将达到相同性能所需的训练步数减少 30%-50%。

2. **与 DQN 改进兼容**：可与 Double、Dueling、Noisy 组合。成立条件：使用重要性采样权重修正偏差。Rainbow DQN 就同时使用了 PER 和这些改进。

3. **对稀疏奖励特别有效**：自动聚焦关键转折点经验。成立条件：关键经验的 TD 误差确实较大。在 Montezuma's Revenge 等高难度 Atari 游戏中，PER 是少数能显著提升性能的单一改进。

### 缺点

1. **实现复杂度高**：需要 SumTree 数据结构和重要性采样权重。缓解：使用成熟的第三方库（如 `torchrl`、`stable-baselines3`），它们提供了经过优化的 PER 实现。从零实现 SumTree 约 100 行代码，是 PER 中最容易出错的部分。

2. **对噪声敏感**：TD 误差中的随机噪声可能被误认为高信息量。缓解：新样本赋予最大优先级，而非基于初始 TD 误差。也可以对 TD 误差做平滑处理（如指数移动平均）来降低噪声影响。

3. **引入超参数**：$\alpha$、$\beta$、$\epsilon$ 需要调整。缓解：默认值（$\alpha=0.6, \beta=0.4$）在大多数环境中表现良好。$\alpha$ 控制优先采样强度，$\beta$ 控制偏差纠正程度，$\epsilon$ 防止优先级为零。

4. **存储开销增大**：每个样本额外存储优先级和 SumTree 索引。缓解：SumTree 仅存浮点数，开销可控。对于 100 万容量的回放池，SumTree 额外占用约 8MB 内存。

### 对比

| 特性 | Uniform Replay | PER |
|------|----------------|-----|
| 采样方式 | 均匀随机 | 按优先级 |
| 数据结构 | deque / list | SumTree |
| 采样复杂度 | $O(1)$ | $O(\log N)$ |
| 偏差纠正 | 不需要 | 重要性采样 |
| 学习效率 | 基准 | 提升（信息量不均时） |

## 7. 调库实现

```python
"""PER DQN 完整实现：SumTree + PrioritizedReplayBuffer + DQN Agent"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random


class SumTree:
    """SumTree 数据结构，O(log N) 的插入、更新和采样"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.count = 0

    def update(self, idx, priority):
        """更新叶子节点优先级，并向上传播"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, priority, data):
        """添加新样本"""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def get_leaf(self, value):
        """根据优先级值采样对应的叶子节点"""
        idx = 0
        while idx < self.capacity - 1:  # 非叶子节点
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]

    @property
    def max_prior(self):
        return np.max(self.tree[self.capacity-1:self.capacity-1+self.count])


class PrioritizedReplayBuffer:
    """优先经验回放"""
    def __init__(self, capacity, alpha=0.6, epsilon=0.01, beta=0.4, beta_step=0.0001):
        self.tree = SumTree(capacity)
        self.alpha = alpha      # 优先级指数
        self.epsilon = epsilon  # 最小优先级
        self.beta = beta        # 重要性采样参数
        self.beta_step = beta_step

    def push(self, transition):
        """添加样本，使用当前最大优先级（确保新样本至少被采样一次）"""
        max_prior = self.tree.max_prior if self.tree.count > 0 else 1.0
        self.tree.add(max_prior, transition)

    def sample(self, batch_size):
        """优先采样一个批量"""
        self.beta = min(1.0, self.beta + self.beta_step)
        indices, priorities, transitions = [], [], []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, prior, data = self.tree.get_leaf(s)
            indices.append(idx)
            priorities.append(prior)
            transitions.append(data)

        # 计算重要性采样权重
        probs = np.array(priorities) / self.tree.total
        weights = (self.tree.count * probs) ** (-self.beta)
        weights /= weights.max()  # 归一化到 [0, 1]

        s, a, r, s2, d = zip(*transitions)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32)), np.array(indices), weights

    def update_priorities(self, indices, priorities):
        """更新采样后的优先级"""
        for idx, pri in zip(indices, priorities):
            pri = (abs(pri) + self.epsilon) ** self.alpha
            self.tree.update(idx, pri)

    def __len__(self):
        return self.tree.count


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim))

    def forward(self, x):
        return self.net(x)


class PERDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        self.policy_net = MLP(state_dim, action_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.buffer = PrioritizedReplayBuffer(capacity=50000)
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 0.95
        self.target_update = 10
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state_t).argmax(dim=1).item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        (s, a, r, s2, d), indices, weights = self.buffer.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        q_values = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            next_max_q = self.target_net(s2).max(1)[0].unsqueeze(1)
            target_q = r + self.gamma * next_max_q * (1 - d)

        # 用重要性采样权重加权损失
        td_errors = (q_values - target_q).detach().cpu().numpy()
        loss = (weights * F.mse_loss(q_values, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        for p in self.policy_net.parameters():
            p.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # 用 TD 误差更新优先级
        self.buffer.update_priorities(indices, td_errors.flatten())

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(0.01, self.epsilon * 0.995)


def train_per_dqn():
    env = gym.make('CartPole-v1')
    agent = PERDQNAgent(4, 2)
    rewards = []
    for ep in range(500):
        state, _ = env.reset()
        ep_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.buffer.push((state, action, reward, next_state, float(terminated)))
            agent.update()
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        if (ep + 1) % 50 == 0:
            print(f"回合 {ep+1}, 平均奖励: {np.mean(rewards[-20:]):.1f}")
    env.close()
    return agent, rewards


if __name__ == "__main__":
    train_per_dqn()
```

## 8. 手工代码实现

```python
"""手工实现 SumTree 的核心操作"""
import numpy as np

class SumTreeManual:
    """手工实现 SumTree，不依赖任何框架"""

    def __init__(self, capacity):
        self.capacity = capacity
        # 树数组：前 capacity-1 个是内部节点，后 capacity 个是叶子节点
        self.tree = [0.0] * (2 * capacity - 1)
        self.data = [None] * capacity  # 数据存在叶子节点对应的位置
        self.write_pos = 0  # 当前写入位置
        self.size = 0

    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def add(self, priority, data):
        """添加新样本到叶子节点，并向上传播"""
        leaf_idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(leaf_idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, priority):
        """更新叶子节点优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value):
        """根据值 [0, total) 找到对应的叶子节点"""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


# 测试 SumTree 手工实现
if __name__ == "__main__":
    tree = SumTreeManual(capacity=4)
    # 添加4个样本，优先级分别为 [1, 3, 2, 4]
    priorities = [1.0, 3.0, 2.0, 4.0]
    for i, p in enumerate(priorities):
        tree.add(p, f"sample_{i}")
        print(f"添加 sample_{i}, 优先级={p}, 总和={tree.total}")

    print(f"\n总优先级: {tree.total}")
    print(f"期望采样概率: {[p/tree.total for p in priorities]}")

    # 验证采样分布
    np.random.seed(42)
    counts = [0, 0, 0, 0]
    n_samples = 10000
    for _ in range(n_samples):
        value = np.random.uniform(0, tree.total)
        _, _, data = tree.get(value)
        idx = int(data.split('_')[1])
        counts[idx] += 1

    print(f"\n实际采样频率: {[c/n_samples for c in counts]}")
    print(f"期望采样概率: {[p/tree.total for p in priorities]}")
    print("频率应接近概率 → 验证 SumTree 正确性")
```

## 9. 可视化与结果理解

```python
"""PER DQN 可视化"""
import matplotlib.pyplot as plt
import numpy as np

def plot_per_analysis(rewards_history=None):
    """可视化 PER DQN 的训练效果和优先级分布"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 子图1：训练曲线
    if rewards_history is not None:
        axes[0].plot(rewards_history, alpha=0.3, color='blue')
        window = 20
        if len(rewards_history) >= window:
            ma = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(rewards_history)), ma,
                        color='red', linewidth=2, label=f'{window}回合滑动平均')
        axes[0].set_xlabel('训练回合')
        axes[0].set_ylabel('回合奖励')
        axes[0].set_title('PER DQN 训练曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # 子图2：优先级分布随训练变化
    # 训练初期：TD 误差大，优先级分布不均匀
    # 训练后期：TD 误差小，优先级趋于均匀
    early_priorities = np.random.exponential(2.0, 1000)  # 指数分布（不均匀）
    late_priorities = np.random.exponential(0.5, 1000) + 0.01  # 更均匀
    axes[1].hist(early_priorities, bins=30, alpha=0.5, label='训练初期', density=True)
    axes[1].hist(late_priorities, bins=30, alpha=0.5, label='训练后期', density=True)
    axes[1].set_xlabel('优先级')
    axes[1].set_ylabel('密度')
    axes[1].set_title('优先级分布变化')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 子图3：β 衰减对重要性采样权重的影响
    beta_values = np.linspace(0.4, 1.0, 100)
    # 假设某个样本的概率是均匀的 10 倍
    prob_ratio = 10.0  # P(i) / (1/N) = 10
    weights = (1.0 / prob_ratio) ** beta_values
    axes[2].plot(beta_values, weights, 'b-', linewidth=2)
    axes[2].set_xlabel('β 值')
    axes[2].set_ylabel('重要性采样权重 w')
    axes[2].set_title(f'β 对权重的影响（高优先级样本，概率比={prob_ratio}）')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=1.0/prob_ratio, color='r', linestyle='--', alpha=0.5, label=f'β=1 完全纠正')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('per_dqn_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    mock_rewards = [min(500, max(0, 20 + i * 1.3 + np.random.randn() * 35)) for i in range(300)]
    plot_per_analysis(mock_rewards)
```

**结果解读**：
- 左图：PER DQN 训练曲线，初期可能不如 uniform DQN（还在学习优先级），后期超越
- 中图：训练初期优先级分布不均匀（少数样本优先级很高），后期趋于均匀（大部分样本已学好）
- 右图：$\beta$ 从 0.4 增大到 1.0，重要性采样权重从接近 1（几乎不纠正）逐渐减小到完全纠正偏差

## 10. 模型评估

### 评估指标

| 指标 | 说明 | 为什么适合 |
|------|------|-----------|
| 平均回合奖励 | 最近 N 个回合奖励均值 | 直接反映策略质量 |
| 优先级分布熵 | 衡量优先级分布的均匀程度 | 高熵表示大部分经验已学好 |
| 学习速度 | 达到目标奖励所需步数 | PER 应比 uniform 更快 |

```python
"""PER DQN 评估代码"""
import numpy as np

def evaluate_per_agent(agent, env, n_episodes=20):
    """评估训练好的 PER DQN"""
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.policy_net(state_t).argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(ep_reward)
    return total_rewards


def analyze_priority_distribution(buffer):
    """分析回放池中的优先级分布"""
    priorities = buffer.tree.tree[buffer.tree.capacity-1:buffer.tree.capacity-1+buffer.tree.count]
    priorities = np.array(priorities)
    # 计算分布熵（归一化后）
    probs = priorities / priorities.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))  # 均匀分布的熵
    print(f"=== 优先级分布分析 ===")
    print(f"样本数: {len(priorities)}")
    print(f"优先级均值: {priorities.mean():.4f}")
    print(f"优先级标准差: {priorities.std():.4f}")
    print(f"分布熵: {entropy:.4f} / 最大熵: {max_entropy:.4f}")
    print(f"归一化熵: {entropy/max_entropy:.4f} (1.0=完全均匀)")
    if entropy / max_entropy > 0.9:
        print("→ 优先级接近均匀分布，大部分经验已学好")


if __name__ == "__main__":
    # 演示分析
    from collections import deque
    buffer = PrioritizedReplayBuffer(capacity=100)
    for i in range(50):
        buffer.push((f"state_{i}", i % 2, 1.0, f"next_{i}", False))
    analyze_priority_distribution(buffer)
```

## 11. 常见问题与易错点

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|---------|
| 新样本优先级为 0 | 新经验永远不被采样 | 忘记给新样本赋最大优先级 | `push()` 时使用当前最大优先级 |
| TD 误差更新不及时 | 优先级停留在旧值 | 忘记在 update 后调用 `update_priorities()` | 每次 update 后立即更新 |

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|---------|
| 忘记重要性采样权重 | 训练有偏差，性能不如 uniform | 优先采样改变了数据分布 | 必须在损失函数中乘以 $w_i$ |
| $\beta$ 不衰减 | 永远有偏差 | $\beta$ 始终为初始值 | 每步 `self.beta = min(1.0, self.beta + beta_step)` |
| SumTree 索引错误 | 采到 None 数据 | 写入和读取的索引映射不一致 | 仔细检查 `write_idx` 和 `data_idx` 的转换 |

### 调参层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|---------|
| $\alpha$ 过大 | 只关注少数样本，过拟合 | 优先级影响过强 | 降低 $\alpha$ 到 0.4~0.6 |
| $\epsilon$ 过大 | 优先级差异被淹没 | 所有样本优先级接近 | 降低 $\epsilon$ 到 0.001~0.01 |

**调试黄金法则**：当训练出现问题时，按照以下顺序排查：(1) 检查数据预处理（归一化、裁剪）；(2) 检查损失函数（符号、梯度流向）；(3) 检查超参数（学习率、折扣因子）；(4) 检查网络结构（容量、初始化）。90%的训练问题都出在前两步。

8. **SumTree实现中的数值溢出**
   - **现象**：优先级之和持续增长或出现NaN
   - **原因**：TD误差没有上界，导致优先级无限增长
   - **解决方案**：对优先级设置上限（如 $p_i = \min(|\delta_i|, 100)$），或使用基于秩的优先级代替比例优先级

9. **重要性采样权重过大**
   - **现象**：梯度更新不稳定，Q值震荡
   - **原因**：$\beta$ 太小导致权重修正不足，某些低优先级样本被采样时权重极大
   - **解决方案**：确保权重归一化 $w_i / \max_j w_j$，并使用合理的 $\beta$ 衰减策略（从0.4线性增长到1）

10. **新插入经验的冷启动问题**
    - **现象**：新插入回放缓冲区的经验初始优先级为0，很难被采样到
    - **原因**：PER使用当前TD误差作为优先级，新经验尚未计算TD误差
    - **解决方案**：新经验的初始优先级设为当前缓冲区中的最大优先级（保证至少被采样一次），之后再根据实际TD误差调整

## 12. 学习总结

### 核心思想回顾

PER DQN 的核心改进是将经验回放从均匀采样变为优先采样，优先级由 TD 误差决定。关键公式：

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

SumTree 实现 $O(\log N)$ 的高效采样。重要性采样权重纠正优先采样引入的偏差。PER 的核心价值在于：让智能体把有限的训练时间花在"最需要学习"的经验上，而不是无差别地重复所有经验。

### 与相关算法的联系

- **与标准经验回放（Uniform Replay）的关系**：标准经验回放是 PER 在 $\alpha=0$ 时的特例。PER 的三个超参数（$\alpha$, $\beta$, $\epsilon$）提供了从均匀采样到完全优先采样之间的连续调节。在实际使用中，通常 $\alpha=0.6$ 就能在优先性和多样性之间取得好的平衡。
- **与课程学习（Curriculum Learning）的关系**：PER 可以看作是一种自动的课程学习——训练初期高 TD 误差的样本多（策略差），PER 聚焦于学习基础策略；训练后期 TD 误差分布趋于均匀，PER 自动退化为近似均匀采样。
- **在 Rainbow DQN 中的位置**：PER 是 Rainbow（集成六种改进的 DQN 变体）的组成成分之一。实验表明，PER 单独使用就能带来显著的性能提升，在稀疏奖励环境中尤其明显。

### 后续学习方向

1. **Rainbow DQN**：将 PER 与 Double、Dueling、Noisy 等改进综合使用
2. **分层优先经验回放（Hindsight PER）**：结合 HER（后见之明经验回放）和 PER，在目标导向的任务中进一步提升样本效率
3. **基于不确定性的优先级**：用 ensemble 的不确定性替代 TD 误差作为优先级指标，更加鲁棒

**总结要点**：学习本节后，你应该能回答三个核心问题：(1) 这个算法解决了什么问题？(2) 它的核心创新点是什么？(3) 它与前置和后续算法的区别和联系是什么？如果这三个问题都能清晰回答，说明你真正理解了这个算法。

PER的核心贡献在于将"学习什么"从随机的变为有目的性的——让算法将有限的计算资源投入到最有信息量的经验上。

## 13. 练习题与思考题

### 基础题

**题1**：为什么新样本要赋予最大优先级而不是用实际 TD 误差？

**答**：因为新样本还没被网络评估过，TD 误差未知（需要前向传播才能计算）。赋予最大优先级确保新样本至少被采样一次，之后根据实际 TD 误差调整优先级。如果初始优先级太低，新经验可能永远不被采样。

**题2**：SumTree 的采样时间复杂度是多少？为什么比排序后采样更高效？

**答**：$O(\log N)$。SumTree 利用二叉树结构，从根节点向下搜索，每层只需一次比较，共 $\log_2 N$ 层。排序采样需要 $O(N \log N)$ 排序 + $O(\log N)$ 二分搜索，且每次更新优先级都要重排序。SumTree 更新优先级也只需 $O(\log N)$（向上传播），远优于重排序的 $O(N \log N)$。

### 进阶题

**题3**：为什么 $\beta$ 需要从 0.4 逐渐增大到 1.0，而不是一开始就用 1.0？

**答**：训练初期，网络对环境的了解很少，高优先级样本（高 TD 误差）确实是信息量最大的样本，应该被重点学习。如果 $\beta=1$ 完全纠正偏差，就退化为均匀采样，失去了优先采样的优势。随着训练进行，网络的 TD 误差减小，优先级分布趋于均匀，此时增大 $\beta$ 可以减少偏差，保证最终收敛到正确的解。这就是"热偏置"（warm-up bias）的思想。

### 开放思考题

**题4**：除了 TD 误差，还有哪些指标可以作为经验优先级？各有什么优缺点？

**思考方向**：
- **TD 误差**（当前方案）：直接反映预测偏差，但容易被噪声误导
- **基于回报的优先级**：$p_i = |G_i - V(s_i)|$，衡量实际回报与预期的差距，但需要完整回合
- **基于不确定性的优先级**：用 Q 值的方差或 ensemble 不确定性作为优先级，更鲁棒但计算成本高
- **基于新颖性的优先级**：状态访问频率越低优先级越高，鼓励探索，但不直接关联学习进度

### 编程练习题（补充）

**题目4**：实现一个简化版的SumTree。要求：(1) 支持插入（update）操作，时间复杂度O(log n)；(2) 支持按优先级采样（get），时间复杂度O(log n)。给定8个叶节点，优先级为 [3, 10, 12, 4, 1, 2, 8, 2]，请画出完整的SumTree结构，并模拟采样值 s=20 的路径。

**参考答案**：

SumTree结构（内部节点 = 左子 + 右子）：
```
          42
       /           29        13
    /  \      /    13    16   3    10
 / \   / \  / \  / 3  10 12  4 1  2 8   2
```

采样s=20的路径：
- 根节点42，s=20 < 左子29，进入左子树
- 节点29，s=20 > 左子13，s=20-13=7，进入右子树
- 节点16，s=7 < 左子12，进入左子树
- 叶节点12，找到！对应的优先级为12。

这说明优先级为12的节点被采样的概率为 12/42 ≈ 28.6%。

## 14. 学习路径建议

### 前置算法
- **DQN**：PER DQN 的基础。必须先理解 DQN 的经验回放机制，理解为什么均匀采样可以打破样本相关性，才能理解 PER 在此基础上的改进。
- **经验回放机制**：理解回放池的入队、出队、采样过程，以及为什么要用随机采样打破时间相关性。
- **TD 误差**：理解 TD 误差的含义——它衡量了当前网络对某个样本的预测偏差，偏差越大说明该样本包含越多的"新信息"。

### 平行算法
- **Double DQN**：从目标值计算角度改进 DQN，解决 Q 值过估计。可以与 PER 结合使用
- **Dueling DQN**：从网络架构角度改进 DQN，分离状态价值和动作优势。可以与 PER 结合使用
- **Noisy DQN**：从探索策略角度改进 DQN。可以与 PER 结合使用
- 这三种改进与 PER 正交，可以任意组合

### 进阶算法
- **Rainbow DQN**：将 PER 与上述所有改进综合，是离散动作空间 DQN 的集大成方案
- **Hindsight Experience Replay (HER)**：在目标导向任务中改写经验的目标，与 PER 的优先采样思想互补
- **离线强化学习中的数据筛选**：在离线 RL 中，如何从固定数据集中筛选高质量样本是 PER 思想的延伸

### 推荐资源
1. 原书第8章8.4节——本书对 PER DQN 的详细讲解和 SumTree 代码实现
2. Schaul et al. "Prioritized Experience Replay" (2015)——PER 的原始论文，详细阐述了 SumTree 数据结构和重要性采样权重的推导
3. Rainbow DQN 论文（Hessel et al., 2018）——展示了 PER 与其他改进综合使用的效果，实验结果充分证明了 PER 的价值
4. Stable-Baselines3 中的 PER 实现——提供了经过工程优化的 SumTree 和优先回放缓冲区实现

**实践建议**：理论学习后，最重要的下一步是动手实现。建议在CartPole-v1或Pendulum-v1等简单环境上完整实现一遍算法，观察训练曲线，调试超参数。只有亲手实现并调参，才能真正理解算法的每个细节。

**补充资源**：Schaul et al. "Prioritized Experience Replay" (ICLR 2016) 是PER的原始论文，包含SumTree实现、比例/基于秩优先级的对比实验、以及重要性采样权重的完整理论分析。

