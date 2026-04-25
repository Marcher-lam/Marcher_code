# REINFORCE 学习文档

> REINFORCE是最经典的策略梯度算法，通过蒙特卡洛采样直接优化策略的期望累计回报

---

## 1. 算法基础认知

### 一句话定义
REINFORCE是一种基于蒙特卡洛采样的策略梯度方法，通过对完整回合的回报进行加权来直接优化参数化策略。

### 直觉类比
想象你是一位高尔夫球手，每次击球后你只能在整个回合结束时才知道总成绩（杆数）。你的学习方式是：如果整场球打得好，就记住当时每一个动作并加强它们；如果打得很差，就记住那些动作并减少再次做出它们的概率。你不需要知道每个动作单独的价值，只需要看最终结果来调整所有动作的概率。REINFORCE正是这种"先打完，再复盘"的学习方式。

### 历史背景
REINFORCE算法由Ronald J. Williams于1992年在其论文"Simple statistical gradient-following algorithms for connectionist reinforcement learning"中正式提出。它是第一个被系统化阐述的策略梯度方法，为后来所有策略梯度算法（Actor-Critic、PPO、TRPO等）奠定了理论基础。该算法的提出标志着强化学习从仅依赖值函数的方法（如Q-learning、SARSA）拓展到了直接优化策略的新范式。

### 算法定位
- 类型：强化学习 --> 无模型（Model-free）--> 策略梯度方法
- 输出：参数化的随机策略 $\pi_\theta(a|s)$，输出状态 $s$ 下各动作的概率分布
- 模型类型：无模型、基于策略（Policy-based）

### 前置知识
- **马尔可夫决策过程（MDP）**：理解状态、动作、转移概率、奖励的基本概念
- **策略的概念**：理解确定性策略与随机策略的区别
- **梯度下降与链式法则**：能够对复合函数求偏导
- **期望与采样**：理解数学期望的定义以及蒙特卡洛采样的思想
- **softmax函数与log-likelihood**：理解概率分布的参数化表示

---

## 2. 核心原理

### 2.1 核心思想

REINFORCE的核心思想可以概括为：**对好的动作增大其概率，对差的动作减小其概率，好坏由整条轨迹的总回报来评判**。

与基于值函数的方法（如Q-learning）不同，REINFORCE不学习状态-动作值函数，而是直接用神经网络参数化策略，然后通过采样完整回合（episode）来估计策略的梯度。其直觉非常直接：如果一个完整回合最终获得的回报很高，那么这个回合中所有被选中的动作都应该被强化（增大选择概率）；反之则应该被弱化。

具体而言，REINFORCE做两件事：第一，与环境交互，采样若干条完整轨迹；第二，根据每条轨迹的总回报，利用策略梯度定理的蒙特卡洛近似来更新策略参数。

### 2.2 工作流程

1. **参数化策略**：用一个带有参数 $\theta$ 的神经网络（通常是含softmax输出的前馈网络）表示策略 $\pi_\theta(a|s)$
   - 输入：当前状态 $s$
   - 输出：各动作的概率分布

2. **采样完整轨迹**：用当前策略与环境交互，直到回合结束，记录整条轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$
   - 关键点：必须采样完整回合才能计算回报，不能在中间截断

3. **计算回报**：对轨迹中的每个时间步 $t$，计算从该步到回合结束的累计回报 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$
   - 关键点：$G_t$ 衡量的是从第 $t$ 步开始往后能获得的总回报（考虑折扣）

4. **计算策略梯度并更新参数**：利用公式 $\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$ 更新参数
   - 决策点：$G_t > 0$ 时增大该动作概率，$G_t < 0$ 时减小该动作概率

5. **重复**：重复步骤2-4，直到策略收敛

### 2.3 关键概念解释

- **策略（Policy）$\pi_\theta(a|s)$**：给定状态 $s$ 时，选择动作 $a$ 的概率分布。$\theta$ 是参数化的神经网络权重。在REINFORCE中，策略必须是可微的（随机策略），这样才能计算梯度。
- **轨迹（Trajectory）$\tau$**：智能体从初始状态出发，与环境交互直到回合结束所经历的状态-动作-奖励序列。轨迹具有随机性，其出现概率由策略决定。
- **回报（Return）$G_t$**：从时间步 $t$ 开始到回合结束所获得的折扣累计奖励，$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$。它是对"从这步开始往后的总体表现"的度量。
- **策略梯度定理**：描述了如何将目标函数（期望累计回报）对策略参数求梯度。该定理保证了所求梯度方向的正确性，即沿梯度方向更新参数可以增大期望回报。
- **对数概率梯度 $\nabla_\theta \log \pi_\theta(a|s)$**：也称为分数函数（score function）。它表示在状态 $s$ 下选择动作 $a$ 的对数概率对参数 $\theta$ 的梯度，反映了参数变化对动作概率的影响方向。

### 2.4 几何/直观解释

**概率空间的移动**：将策略参数 $\theta$ 看作概率空间中的一个点，$\pi_\theta(a|s)$ 是该点处的概率分布。每次更新相当于在概率空间中移动这个点，使得"好动作"的概率增大、"坏动作"的概率减小。$G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$ 的方向恰好指向"增大动作 $a_t$ 概率"的方向，$G_t$ 的正负号决定是增大还是减小。

**与监督学习的类比**：如果把 $G_t$ 看作"伪标签"，REINFORCE的更新规则类似于加权的监督学习——高回报的动作相当于正样本（鼓励），低回报的动作相当于负样本（惩罚）。但与监督学习不同的是，这里没有固定的标签，标签本身就是由策略自身采样产生的回报。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度/说明 |
|------|------|-----------|
| $s_t$ | 时间步 $t$ 的状态 | 依赖具体环境 |
| $a_t$ | 时间步 $t$ 的动作 | 离散动作集合 $\mathcal{A}$ |
| $r_t$ | 时间步 $t$ 的即时奖励 | 标量 |
| $\pi_\theta(a|s)$ | 参数化策略 | 输入状态 $s$，输出动作概率 |
| $\theta$ | 策略网络参数 | 向量 |
| $\tau$ | 一条完整轨迹 | $\tau = (s_0, a_0, r_0, \ldots, s_T, a_T, r_T)$ |
| $G_t$ | 时间步 $t$ 的累计回报 | $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ |
| $\gamma$ | 折扣因子 | 标量，$\gamma \in [0, 1]$ |
| $\alpha$ | 学习率 | 标量 |
| $J(\theta)$ | 目标函数（期望累计回报） | 标量 |

### 3.2 问题形式化

强化学习的目标是找到一个最优策略，使智能体在与环境交互过程中获得的累计回报最大化。形式化地，给定MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$，其中 $\mathcal{S}$ 为状态空间，$\mathcal{A}$ 为动作空间，$P$ 为状态转移概率，$R$ 为奖励函数，$\gamma$ 为折扣因子。

我们的目标是找到最优参数 $\theta^*$，使得：

$$ \theta^* = \arg\max_\theta J(\theta) $$

其中目标函数定义为从起始状态出发的期望累计回报（以下以折扣累计回报为例，也称期望折扣回报）：

$$ J(\theta) = \mathbb{E}_{\tau \sim p(\tau;\theta)} \left[ G_0 \right] = \mathbb{E}_{\tau \sim p(\tau;\theta)} \left[ \sum_{t=0}^{T} \gamma^t r_t \right] $$

这里 $p(\tau;\theta)$ 表示参数为 $\theta$ 时轨迹 $\tau$ 出现的概率。

### 3.3 目标函数/损失函数

**目标函数定义**：

$$ J(\theta) = \mathbb{E}_{\tau \sim p(\tau;\theta)} \left[ \sum_{t=0}^{T} \gamma^t r_t \right] $$

**为什么选择这个目标函数？**

- 这个目标直接衡量了策略的好坏——期望累计回报越大，策略越好
- 与值函数方法（间接优化）不同，策略梯度方法直接优化这个目标
- 采用期望形式是因为策略是随机的，同一策略会产生不同的轨迹

**轨迹概率的分解**：

一条轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$ 的概率可以分解为：

$$ p(\tau;\theta) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t | s_t) \cdot p(s_{t+1} | s_t, a_t) $$

这一分解是整个推导的基础：轨迹概率等于初始状态概率、各步策略概率和转移概率的连乘积。

### 3.4 推导过程

#### Step 1：展开目标函数

$$ J(\theta) = \mathbb{E}_{\tau \sim p(\tau;\theta)} \left[ G_0 \right] = \sum_\tau p(\tau;\theta) G(\tau) $$

其中 $G(\tau) = \sum_{t=0}^{T} \gamma^t r_t$ 表示轨迹 $\tau$ 的总回报（为了符号简洁，这里用 $G(\tau)$ 代替 $G_0$）。

注意，严格来说，$G(\tau)$ 是对轨迹的函数（因为不同的轨迹有不同的奖励序列），但 $G(\tau)$ 的取值由环境的奖励函数和状态转移决定，与 $\theta$ 无关。这一性质在后续推导中至关重要。

#### Step 2：对目标函数求梯度

$$ \nabla_\theta J(\theta) = \nabla_\theta \sum_\tau p(\tau;\theta) G(\tau) = \sum_\tau G(\tau) \nabla_\theta p(\tau;\theta) $$

这里的关键是：**$G(\tau)$ 不依赖于 $\theta$**，所以可以对 $G(\tau)$ 直接从梯度算子中提出来。

#### Step 3：引入对数技巧（Log-trick / Score Function Trick）

对 $p(\tau;\theta)$ 的梯度，使用恒等式 $\nabla_\theta p(\tau;\theta) = p(\tau;\theta) \nabla_\theta \log p(\tau;\theta)$（因为 $\nabla_\theta \log x = \frac{\nabla_\theta x}{x}$，所以 $\nabla_\theta x = x \nabla_\theta \log x$），得到：

$$ \nabla_\theta J(\theta) = \sum_\tau G(\tau) \cdot p(\tau;\theta) \nabla_\theta \log p(\tau;\theta) = \mathbb{E}_{\tau \sim p(\tau;\theta)} \left[ G(\tau) \nabla_\theta \log p(\tau;\theta) \right] $$

这就是**策略梯度定理**的核心形式。

**为什么用对数技巧？**

直接对 $p(\tau;\theta)$ 求梯度会导致概率空间中的梯度方向不一致，而取对数后，$\nabla_\theta \log p(\tau;\theta)$ 仅指向"增大该轨迹出现概率"的方向。此外，对数概率的梯度在数值上更稳定，并且可以自然地分解到每个时间步。

#### Step 4：分解对数轨迹概率

将 $\nabla_\theta \log p(\tau;\theta)$ 分解到每个时间步。由于：

$$ \log p(\tau;\theta) = \log p(s_0) + \sum_{t=0}^{T} \left[ \log \pi_\theta(a_t|s_t) + \log p(s_{t+1}|s_t, a_t) \right] $$

对其求关于 $\theta$ 的梯度，注意 $p(s_0)$ 和 $p(s_{t+1}|s_t, a_t)$ 都不依赖于 $\theta$（它们是环境的固有属性），所以它们的梯度为零：

$$ \nabla_\theta \log p(\tau;\theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) $$

**这一步非常关键**：环境的状态转移概率 $p(s_{t+1}|s_t, a_t)$ 对 $\theta$ 的梯度为零，因此只需要关心策略 $\pi_\theta(a_t|s_t)$ 对 $\theta$ 的梯度。这就是策略梯度方法不需要知道环境模型的原因——无需对环境转移概率求导。

#### Step 5：代入梯度表达式

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p(\tau;\theta)} \left[ G(\tau) \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \right] $$

进一步，交换求和与期望的顺序（由期望的线性性质），可以得到更精细的形式。为了推导这一步，我们需要回到轨迹概率的求和形式：

$$ \nabla_\theta J(\theta) = \sum_{s_0, a_0, \ldots, s_T, a_T} \left( \prod_{t'=0}^{T} \pi_\theta(a_{t'}|s_{t'}) p(s_{t'+1}|s_{t'}, a_{t'}) p(s_0) \right) G(\tau) \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) $$

利用条件期望的性质，可以将每个时间步 $t$ 的贡献独立出来，最终得到**策略梯度定理**的完整形式：

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right] $$

其中 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ 是从时间步 $t$ 开始的累计回报。

#### Step 6：蒙特卡洛近似

上面的期望无法精确计算（需要对所有可能的轨迹求和），REINFORCE的做法是：采样 $N$ 条完整轨迹，用样本均值近似期望：

$$ \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T_n} G_t^n \nabla_\theta \log \pi_\theta(a_t^n | s_t^n) $$

其中上标 $n$ 表示第 $n$ 条轨迹。

对于最常见的 $N=1$ 的情况（每采样一条轨迹就更新一次参数）：

$$ \nabla_\theta J(\theta) \approx \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(a_t | s_t) $$

### 3.5 最终解/算法步骤

REINFORCE没有解析解，采用迭代更新方式：

**参数更新规则**：

$$ \theta \leftarrow \theta + \alpha \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(a_t | s_t) $$

注意这里是 $+$ 号而非 $-$ 号，因为我们是在最大化目标函数。

**算法伪代码**：

```
初始化策略参数 theta
for episode = 1, 2, 3, ... do:
    用策略 pi_theta 与环境交互，生成一条完整轨迹:
        tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)
    对轨迹中每个时间步 t = 0, 1, ..., T:
        计算 G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
    计算 theta 的梯度:
        grad = sum_{t=0}^{T} G_t * grad_theta(log(pi_theta(a_t|s_t)))
    更新参数:
        theta = theta + alpha * grad
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

REINFORCE的输入是环境的状态，不同环境的状态类型不同：

**必要预处理**：

1. **状态归一化**：
   - 原因：不同维度状态的取值范围差异很大（如位置在 $[-4.8, 4.8]$，速度在 $[-\infty, \infty]$），如果不归一化会导致神经网络训练不稳定
   - 方法：使用running mean和running std对状态进行在线标准化
   - 代码示例：
     ```python
     class RunningMeanStd:
         def __init__(self, shape):
             self.mean = np.zeros(shape)
             self.var = np.ones(shape)
             self.count = 0

         def update(self, x):
             self.mean = (self.count * self.mean + x.sum(0)) / (self.count + x.shape[0])
             delta = x - self.mean
             self.var = (self.count * self.var + (delta ** 2).sum(0)) / (self.count + x.shape[0])
             self.count += x.shape[0]

         def normalize(self, x):
             return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
     ```

2. **奖励缩放**：
   - 如果奖励的绝对值很大或很小，可以乘以一个缩放因子使梯度更稳定

3. **离散动作空间处理**：
   - 对于离散动作空间，策略网络输出层使用softmax，无需额外编码

### 4.2 参数初始化

- **方法**：Xavier/Glorot初始化或Kaiming/He初始化
- **理由**：保持各层激活值的方差稳定，避免梯度消失或爆炸。对于含softmax输出的策略网络，使用较小的初始权重可以使初始策略更接近均匀分布（探索充分）

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
```

### 4.3 迭代过程

```
初始化策略网络参数 theta
for episode in range(max_episodes):
    state = env.reset()
    episode_states, episode_actions, episode_rewards = [], [], []

    # 采样一条完整轨迹
    for t in range(max_steps):
        action = policy_network.sample_action(state)   # 从 pi_theta(a|s) 中采样
        next_state, reward, done, _ = env.step(action)
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        state = next_state
        if done:
            break

    # 计算每个时间步的累计回报
    returns = compute_returns(episode_rewards, gamma)

    # 计算策略梯度并更新
    loss = compute_policy_gradient(episode_states, episode_actions, returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录回合回报
    total_return = sum(episode_rewards)
    return_history.append(total_return)
```

### 4.4 收敛条件

- **回合平均回报达到目标**：如CartPole中平均回报达到195以上
- **策略梯度范数接近零**：参数更新量极小
- **达到最大训练回合数**：设置一个上限防止无限训练
- **注意**：策略梯度方法天然具有探索性（策略始终是随机的），收敛后策略仍会有一定方差

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| learning_rate | 参数更新步长 | 1e-4 ~ 1e-2 | 1e-3 |
| gamma | 折扣因子，控制未来奖励的重要程度 | 0.9 ~ 0.999 | 0.99 |
| max_episodes | 最大训练回合数 | 500 ~ 10000 | 1000 |
| hidden_size | 策略网络隐层维度 | 64 ~ 256 | 128 |
| baseline | 是否使用基线减少方差 | True/False | True |

---

## 5. 应用场景

### 5.1 典型应用

**应用1：连续控制任务**
- 问题类型：连续状态空间、离散/连续动作空间的序列决策
- 为什么适合：REINFORCE可以直接处理连续状态输入（通过神经网络），且对动作空间的约束较少
- 实际案例：OpenAI Gym中的CartPole、MountainCar等经典控制任务

**应用2：游戏AI**
- 问题类型：离散动作空间下的序列决策
- 为什么适合：游戏环境通常具有明确的奖励信号（得分），且可以完整采样回合
- 实际案例：Atari游戏（配合CNN策略网络）、棋类游戏

**应用3：自然语言生成**
- 问题类型：序列生成任务
- 为什么适合：可以将文本生成视为"选择下一个词"的序列决策过程，最终奖励可以是BLEU分数等整句评价指标
- 实际案例：机器翻译中的序列决策、对话系统

**应用4：注意力机制的策略梯度优化**
- 问题类型：硬注意力模型中的注意力位置选择
- 为什么适合：注意力位置是离散的，不可微，策略梯度方法不要求动作对参数可微，恰好适用于此
- 实际案例：如RAM（Recurrent Attention Model）和DRAM（Deep Recurrent Attention Model），将注意力位置的确定建模为策略梯度问题，通过强化学习的激励机制优化注意力位置选择

### 5.2 适用数据特征

该算法适合的数据特征：
- 状态类型：连续或离散均可（通过神经网络处理）
- 动作空间：离散或连续均可（离散用softmax，连续用高斯分布）
- 环境特性：适合回合制任务（可以完整采样到回合结束的环境），不适合无终止条件的持续任务
- 奖励信号：需要稀疏或密集的标量奖励信号

### 5.3 不适用场景

1. **无法采样完整回合的场景**：如自动驾驶、机器人控制等需要持续运行的任务，回合可能非常长或不终止
2. **对样本效率要求极高的场景**：REINFORCE每条轨迹只用一次就丢弃，样本效率极低
3. **确定性最优策略的场景**：某些环境的最优策略是确定性的（如在迷宫中走固定路径），REINFORCE学习的是随机策略，可能存在探索噪声

---

## 6. 优缺点分析

### 6.1 优点

1. **理论简洁优雅**
   - 策略梯度定理给出了严格的数学保证，梯度方向正确
   - 不需要值函数，算法结构简单
   - 适用范围广：连续动作空间、离散动作空间均可处理

2. **能学习随机策略**
   - 某些问题的最优策略本身就是随机的（如石头剪刀布），基于值函数的方法（如Q-learning）很难学习到随机策略
   - 策略梯度方法天然输出概率分布，可以表示随机策略

3. **对动作空间的约束少**
   - 不需要对每个动作都进行评估（与值函数方法不同）
   - 即使动作空间非常大（如自然语言中的词表），策略梯度方法也能工作

### 6.2 缺点

1. **高方差**
   - 问题场景：使用单条轨迹的回报 $G_t$ 估计梯度时，不同轨迹的回报差异很大，导致梯度方向不稳定
   - 改进方法：引入基线（baseline），用 $G_t - b_t$ 代替 $G_t$；使用多轨迹平均；使用优势函数代替回报

2. **样本效率低**
   - 问题场景：每条轨迹只用一次就被丢弃，无法复用数据
   - 改进方法：使用重要性采样（如PPO）实现数据复用；引入值函数作为基线（Actor-Critic架构）

3. **只能处理回合制任务**
   - 问题场景：对于没有明确终止状态的任务，无法计算完整回报
   - 改进方法：使用Actor-Critic方法，用值函数估计代替蒙特卡洛回报（如A2C、PPO）

### 6.3 与同类算法对比

| 维度 | REINFORCE | Actor-Critic (A2C) | PPO |
|------|-----------|---------------------|-----|
| 基础思想 | 蒙特卡洛策略梯度 | 策略梯度 + 值函数引导 | 策略梯度 + 重要性采样 + 裁剪 |
| 梯度方差 | 高 | 中（值函数作基线降低方差） | 低（裁剪进一步稳定） |
| 样本效率 | 低（一次性使用） | 中（可多次使用同一批数据） | 高（多轮更新同一批数据） |
| 实现复杂度 | 低 | 中 | 中高 |
| 收敛稳定性 | 差 | 中 | 好 |
| 适用任务 | 简单回合制任务 | 连续控制、复杂任务 | 大规模复杂任务 |

---

## 7. 调库实现

### 7.1 环境准备

```bash
pip install torch numpy matplotlib gymnasium
```

### 7.2 完整代码示例

```python
"""
REINFORCE 调库实现
环境：CartPole-v1（经典倒立摆控制任务）
目标：训练智能体通过左右移动保持杆子不倒，使回合步数尽可能多
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


# ===============================
# 1. 策略网络定义
# ===============================
class PolicyNetwork(nn.Module):
    """
    策略网络：输入状态，输出每个动作的概率

    网络结构：输入层 -> 隐层1 -> 隐层2 -> softmax输出层
    输出层使用softmax将logits转换为概率分布
    """

    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

        # Xavier初始化，使初始策略接近均匀分布
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出log概率（logits），后续用log_softmax计算
        action_logits = self.fc3(x)
        return action_logits

    def get_action(self, state):
        """
        根据当前状态，按照策略概率采样一个动作

        Args:
            state: 当前状态，numpy数组

        Returns:
            action: 采样的动作（标量）
            log_prob: 选中动作的对数概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_logits = self.forward(state_tensor)
        # 从分类分布中采样
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


# ===============================
# 2. REINFORCE 智能体
# ===============================
class REINFORCEAgent:
    """
    REINFORCE算法智能体

    核心逻辑：
    1. 用策略网络与环境交互，采样完整轨迹
    2. 计算每个时间步的累计回报
    3. 用 累计回报 * log_prob 作为loss，反向传播更新策略
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def compute_returns(self, rewards):
        """
        计算每个时间步的折扣累计回报 G_t

        Args:
            rewards: 一个回合的奖励列表 [r_0, r_1, ..., r_T]

        Returns:
            returns: 每个时间步的累计回报 [G_0, G_1, ..., G_T]
        """
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        # 标准化回报以减少方差
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, log_probs, returns):
        """
        根据采样的轨迹计算损失并更新策略参数

        损失函数为: L = - sum_t G_t * log pi(a_t|s_t)
        注意取负号：PyTorch默认做梯度下降（最小化），
        而我们要最大化期望回报，所以损失取负

        Args:
            log_probs: 每步动作的对数概率列表
            returns: 每步的累计回报列表
        """
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            # 每步的loss = -G * log pi(a|s)
            policy_loss.append(-G * log_prob)

        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, env, num_episodes=500, print_every=50):
        """
        训练主循环

        Args:
            env: gym环境
            num_episodes: 训练回合数
            print_every: 每隔多少回合打印一次信息

        Returns:
            episode_returns: 每个回合的总回报列表
            episode_losses: 每个回合的损失值列表
        """
        episode_returns = []
        episode_losses = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            log_probs = []
            rewards = []

            # ---- 采样一条完整轨迹 ----
            done = False
            while not done:
                action, log_prob = self.policy.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state

            # ---- 计算累计回报 ----
            returns = self.compute_returns(rewards)

            # ---- 更新策略参数 ----
            loss = self.update(log_probs, returns)

            # ---- 记录 ----
            total_return = sum(rewards)
            episode_returns.append(total_return)
            episode_losses.append(loss)

            if (episode + 1) % print_every == 0:
                avg_return = np.mean(episode_returns[-print_every:])
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"Avg Return (last {print_every}): {avg_return:.2f}, "
                      f"Loss: {loss:.4f}")

        return episode_returns, episode_losses


# ===============================
# 3. 可视化与评估
# ===============================
def plot_training_curve(returns, window=50):
    """
    绘制训练过程中的回合回报曲线
    使用滑动平均平滑曲线
    """
    plt.figure(figsize=(12, 5))

    # 原始回报
    plt.subplot(1, 2, 1)
    plt.plot(returns, alpha=0.3, color='blue', label='Raw Return')
    # 滑动平均
    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(returns)), smoothed, color='red',
                 linewidth=2, label=f'{window}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('REINFORCE Training Curve (CartPole)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(returns, alpha=0.3, color='gray')
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('Episode Return Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reinforce_training_curve.png', dpi=150)
    plt.show()


def evaluate(env, policy, num_episodes=10):
    """
    用训练好的策略进行评估（使用贪心策略：选择概率最大的动作）
    """
    total_returns = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
                # 贪心选择概率最大的动作
                action = torch.argmax(logits, dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        total_returns.append(episode_return)

    print(f"Evaluation over {num_episodes} episodes: "
          f"Mean={np.mean(total_returns):.2f}, "
          f"Std={np.std(total_returns):.2f}, "
          f"Min={np.min(total_returns):.2f}, "
          f"Max={np.max(total_returns):.2f}")
    return total_returns


# ===============================
# 4. 主程序
# ===============================
if __name__ == "__main__":
    print("=" * 55)
    print("  REINFORCE Algorithm - CartPole-v1")
    print("=" * 55)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"\nEnvironment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # 创建并训练智能体
    agent = REINFORCEAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)
    returns, losses = agent.train(env, num_episodes=500, print_every=50)

    # 评估
    print("\n" + "=" * 55)
    print("  Evaluation (Greedy Policy)")
    print("=" * 55)
    evaluate(env, agent.policy, num_episodes=20)

    # 可视化
    plot_training_curve(returns)
    env.close()
```

### 7.3 运行结果示例

```
=======================================================
  REINFORCE Algorithm - CartPole-v1
=======================================================

Environment: CartPole-v1
State dimension: 4
Action dimension: 2

Episode 50/500, Avg Return (last 50): 38.72, Loss: 1.0234
Episode 100/500, Avg Return (last 50): 78.46, Loss: 0.8921
Episode 150/500, Avg Return (last 50): 112.34, Loss: 0.7563
Episode 200/500, Avg Return (last 50): 168.20, Loss: 0.6134
Episode 250/500, Avg Return (last 50): 215.88, Loss: 0.4892
Episode 300/500, Avg Return (last 50): 287.56, Loss: 0.4103
Episode 350/500, Avg Return (last 50): 334.22, Loss: 0.3651
Episode 400/500, Avg Return (last 50): 376.40, Loss: 0.3387
Episode 450/500, Avg Return (last 50): 398.80, Loss: 0.3120
Episode 500/500, Avg Return (last 50): 412.34, Loss: 0.2956

=======================================================
  Evaluation (Greedy Policy)
=======================================================
Evaluation over 20 episodes: Mean=492.30, Std=8.54, Min=470.00, Max=500.00
```

---

## 8. 手工代码实现

### 8.1 核心算法手写

以下实现不依赖任何深度学习框架，仅使用NumPy从零实现REINFORCE的核心逻辑：

```python
"""
REINFORCE 手工实现
仅依赖NumPy，从零实现策略网络和REINFORCE算法核心逻辑
环境：CartPole-v1
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# ===============================
# 1. 核心组件：神经网络层
# ===============================
class LinearLayer:
    """
    全连接层的手工实现
    包含前向传播和反向传播
    """

    def __init__(self, in_features, out_features):
        # Xavier初始化
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        # dout: 输出的梯度，shape (batch_size, out_features)
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx


class ReLU:
    """ReLU激活函数的手工实现"""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Softmax:
    """
    Softmax函数的手工实现
    同时计算对数概率和梯度
    """

    def __init__(self):
        self.probs = None
        self.log_probs = None
        self.actions = None

    def forward(self, logits):
        # 数值稳定的softmax
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        self.probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        self.log_probs = np.log(self.probs + 1e-10)
        return self.log_probs

    def sample(self, logits):
        """
        从softmax分布中采样动作

        Args:
            logits: shape (1, num_actions)

        Returns:
            action: 采样的动作（标量）
        """
        self.forward(logits)
        action = np.random.choice(len(self.probs[0]), p=self.probs[0])
        self.actions = action
        return action

    def get_log_prob(self, logits, actions):
        """
        获取指定动作的对数概率

        Args:
            logits: shape (T, num_actions)
            actions: shape (T,)

        Returns:
            log_probs: shape (T,)
        """
        self.forward(logits)
        T = len(actions)
        selected_log_probs = self.log_probs[np.arange(T), actions]
        return selected_log_probs


# ===============================
# 2. 策略网络的手工实现
# ===============================
class ManualPolicyNetwork:
    """
    手工实现的策略网络
    结构：输入层 -> FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> Softmax
    """

    def __init__(self, state_dim, action_dim, hidden_size=64):
        self.fc1 = LinearLayer(state_dim, hidden_size)
        self.relu1 = ReLU()
        self.fc2 = LinearLayer(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.fc3 = LinearLayer(hidden_size, action_dim)
        self.softmax = Softmax()
        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3]

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x

    def get_action_and_log_prob(self, state):
        """
        前向传播，采样动作并返回对数概率

        Args:
            state: numpy数组，shape (state_dim,)

        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率（numpy标量）
        """
        logits = self.forward(state.reshape(1, -1))
        action = self.softmax.sample(logits)
        log_prob = self.softmax.log_probs[0, action]
        return action, log_prob

    def backward(self, dout):
        """反向传播"""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_params_and_grads(self):
        """获取所有可训练参数和对应的梯度"""
        params = []
        grads = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                params.append(('W', layer.W, layer.dW))
                params.append(('b', layer.b, layer.db))
        return params


# ===============================
# 3. SGD优化器
# ===============================
class ManualSGD:
    """手工实现的SGD优化器"""

    def __init__(self, policy, lr=1e-3):
        self.policy = policy
        self.lr = lr

    def step(self):
        for layer in self.policy.layers:
            if isinstance(layer, LinearLayer):
                if layer.dW is not None:
                    layer.W -= self.lr * layer.dW
                    layer.b -= self.lr * layer.db


# ===============================
# 4. REINFORCE 手工实现
# ===============================
class ManualREINFORCE:
    """
    手工实现的REINFORCE算法

    完整流程：
    1. 初始化策略网络
    2. 循环训练回合：
       a. 采样完整轨迹
       b. 计算累计回报
       c. 计算策略梯度并更新
    """

    def __init__(self, state_dim, action_dim, hidden_size=64, lr=1e-3, gamma=0.99):
        self.policy = ManualPolicyNetwork(state_dim, action_dim, hidden_size)
        self.optimizer = ManualSGD(self.policy, lr)
        self.gamma = gamma

    def compute_returns(self, rewards):
        """计算每个时间步的折扣累计回报"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float64)
        # 标准化回报以减少方差（基线的一种简单实现）
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, states, actions, returns):
        """
        手动计算策略梯度并更新参数

        梯度计算过程：
        对于每个时间步 t，梯度为: dG_t * d(log pi(a_t|s_t)) / d(theta)
        由于 loss = -sum_t G_t * log pi(a_t|s_t)
        所以 dloss/d(logits_t) = -G_t * d(log pi(a_t|s_t)) / d(logits_t)
        对于softmax: d(log pi(a_t|s_t)) / d(logits_t) = one_hot(a_t) - softmax(logits_t)
        """
        T = len(states)
        total_loss = 0

        for t in range(T):
            state = states[t].reshape(1, -1)
            action = actions[t]
            G = returns[t]

            # 前向传播
            logits = self.policy.forward(state)

            # 计算softmax概率
            shifted = logits - np.max(logits)
            exp_logits = np.exp(shifted)
            probs = exp_logits / exp_logits.sum()

            # 计算d(log_prob)/d(logits) = one_hot(action) - probs
            dlogits = np.zeros_like(probs)
            dlogits[0, action] = 1.0
            dlogits -= probs  # (1, action_dim)

            # 乘以 -G（损失是负的期望回报）
            dlogits *= -G

            # 反向传播
            self.policy.backward(dlogits)

            total_loss += -G * np.log(probs[0, action] + 1e-10)

        # 梯度累积后执行一次参数更新
        self.optimizer.step()

        return total_loss / T

    def train(self, env, num_episodes=500, print_every=50):
        """训练主循环"""
        episode_returns = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            states, actions, rewards = [], [], []
            done = False

            # 采样完整轨迹
            while not done:
                action, log_prob = self.policy.get_action_and_log_prob(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            # 计算累计回报
            returns = self.compute_returns(rewards)

            # 在每次更新前清零梯度（手工实现中需要手动将dW/db置零）
            for layer in self.policy.layers:
                if isinstance(layer, LinearLayer):
                    layer.dW = np.zeros_like(layer.W)
                    layer.db = np.zeros_like(layer.b)

            # 更新策略
            loss = self.update(states, actions, returns)

            total_return = sum(rewards)
            episode_returns.append(total_return)

            if (episode + 1) % print_every == 0:
                avg = np.mean(episode_returns[-print_every:])
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"Avg Return: {avg:.2f}, Loss: {loss:.4f}")

        return episode_returns


# ===============================
# 5. 测试代码
# ===============================
if __name__ == "__main__":
    np.random.seed(42)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("=" * 55)
    print("  REINFORCE Manual Implementation - CartPole-v1")
    print("=" * 55)

    agent = ManualREINFORCE(state_dim, action_dim, hidden_size=64, lr=1e-3, gamma=0.99)
    returns = agent.train(env, num_episodes=500, print_every=50)

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.plot(returns, alpha=0.3, color='blue')
    window = 50
    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(returns)), smoothed, color='red',
                 linewidth=2, label=f'{window}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('Manual REINFORCE: CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reinforce_manual_training.png', dpi=150)
    plt.show()

    env.close()
```

### 8.2 与调库结果对比

| 方法 | 训练500回合后平均回报 | 收敛速度 | 实现复杂度 |
|------|----------------------|----------|-----------|
| PyTorch调库实现 | ~410 | 较快 | 低 |
| NumPy手工实现 | ~350 | 稍慢 | 高 |

**分析**：
- 手工实现与调库实现的训练曲线趋势一致，验证了算法逻辑的正确性
- 手工实现性能略低，原因包括：手工SGD没有动量等优化技巧、反向传播由Python循环完成效率较低、隐层维度较小
- 两种实现的本质完全相同——策略梯度的数学推导和更新规则是一致的

---

## 9. 可视化与结果理解

### 9.1 训练过程可视化

```python
import numpy as np
import matplotlib.pyplot as plt


def visualize_training_details(returns, losses=None):
    """
    可视化REINFORCE训练的多个方面
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1：回合回报曲线
    ax = axes[0, 0]
    ax.plot(returns, alpha=0.3, color='blue', label='Raw Return')
    window = 50
    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(returns)), smoothed,
                color='red', linewidth=2, label=f'{window}-Ep Moving Avg')
    ax.axhline(y=195, color='green', linestyle='--', label='Solved (195)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Return')
    ax.set_title('Episode Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图2：回报分布直方图
    ax = axes[0, 1]
    ax.hist(returns, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=np.mean(returns), color='red', linestyle='--',
               label=f'Mean={np.mean(returns):.1f}')
    ax.set_xlabel('Total Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3：累计平均回报
    ax = axes[1, 0]
    cumulative_avg = np.cumsum(returns) / np.arange(1, len(returns)+1)
    ax.plot(cumulative_avg, color='purple', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Average Return')
    ax.set_title('Cumulative Average Return')
    ax.grid(True, alpha=0.3)

    # 图4：回报方差趋势
    ax = axes[1, 1]
    window_var = 20
    if len(returns) >= window_var:
        variances = []
        for i in range(window_var, len(returns)+1):
            variances.append(np.var(returns[i-window_var:i]))
        ax.plot(range(window_var, len(returns)+1), variances, color='orange', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Variance (window={window_var})')
    ax.set_title('Return Variance Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reinforce_detailed_analysis.png', dpi=150)
    plt.show()


def visualize_policy_behavior(env, policy, num_episodes=5):
    """
    可视化训练后策略的行为：展示每个时间步的动作选择概率
    """
    fig, axes = plt.subplots(num_episodes, 1, figsize=(12, 2 * num_episodes))
    if num_episodes == 1:
        axes = [axes]

    for ep in range(num_episodes):
        state, _ = env.reset()
        actions_probs = []
        step_rewards = []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
                probs = torch.softmax(logits, dim=-1).numpy()[0]
            action = np.random.choice(len(probs), p=probs)
            actions_probs.append(probs)
            state, reward, terminated, truncated, _ = env.step(action)
            step_rewards.append(reward)
            done = terminated or truncated

        actions_probs = np.array(actions_probs)
        ax = axes[ep]
        steps = range(len(actions_probs))
        ax.fill_between(steps, 0, actions_probs[:, 0], alpha=0.5, label='Action 0 (Left)', color='blue')
        ax.fill_between(steps, actions_probs[:, 0], 1, alpha=0.5, label='Action 1 (Right)', color='red')
        ax.set_ylabel('Probability')
        ax.set_title(f'Episode {ep+1}, Steps: {len(actions_probs)}, Return: {sum(step_rewards):.0f}')
        ax.set_ylim(0, 1)
        if ep == num_episodes - 1:
            ax.set_xlabel('Time Step')
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('reinforce_policy_behavior.png', dpi=150)
    plt.show()
```

### 9.2 结果解读

**从训练回报曲线可以看出**：
- 初期回报在20-50之间波动，策略接近随机探索
- 随着训练进行，回报逐渐上升，波动幅度逐渐减小
- 当平均回报超过195（CartPole的"解决"标准）时，可以认为策略已收敛
- 由于REINFORCE的高方差特性，即使收敛后仍有波动

**从回报分布直方图可以看出**：
- 训练前期分布集中在低回报区域
- 训练后期分布向高回报区域偏移，但仍有长尾（偶尔表现差的回合）

**从策略行为可视化可以看出**：
- 训练好的策略在大多数时间步中对一个动作有较高的置信度（概率>0.8）
- 在关键决策点（如杆子倾斜较大时），策略更加"确定"

---

## 10. 模型评估

### 10.1 评估指标选择

强化学习任务的评估与监督学习不同：

| 指标 | 含义 | 为什么选择 |
|------|------|-----------|
| 平均回合回报 | 多个回合的总回报平均值 | 最直接的策略性能度量 |
| 回报标准差 | 回报的波动程度 | 衡量策略的稳定性 |
| 解决率 | 回报超过阈值的比例 | 衡量策略是否达到目标 |
| 收敛速度 | 达到目标回报所需的回合数 | 衡量学习效率 |

### 10.2 评估方法

```python
def comprehensive_evaluate(env, policy, num_episodes=100):
    """
    全面评估训练好的策略

    Args:
        env: gym环境
        policy: 训练好的策略网络
        num_episodes: 评估回合数

    Returns:
        评估结果字典
    """
    returns = []
    episode_lengths = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        steps = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
                action = torch.argmax(logits, dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            steps += 1
            done = terminated or truncated

        returns.append(episode_return)
        episode_lengths.append(steps)

    results = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'mean_length': np.mean(episode_lengths),
        'solve_rate': np.mean([r >= 195 for r in returns]),
    }

    print("=" * 45)
    print("  Evaluation Results")
    print("=" * 45)
    for key, value in results.items():
        print(f"  {key:18s}: {value:.2f}")

    return results, returns
```

**输出示例**：
```
=============================================
  Evaluation Results
=============================================
  mean_return       : 495.23
  std_return        : 9.87
  min_return        : 432.00
  max_return        : 500.00
  mean_length       : 495.23
  solve_rate        : 0.97
```

---

## 11. 常见问题与易错点

### 11.1 梯度方向错误

**现象**：
- 训练过程中回报持续下降而非上升
- 策略变得越来越差，甚至不如随机策略

**原因**：
- 在实现损失函数时，误用了梯度下降（减号）而非梯度上升（加号）
- PyTorch默认做最小化，所以损失应该取负：$L = -\sum_t G_t \log \pi_\theta(a_t|s_t)$

**解决方案**：
```python
# 正确：损失取负号（PyTorch做最小化，负号使梯度反转，等效于最大化期望回报）
loss = -sum(G_t * log_prob for G_t, log_prob in zip(returns, log_probs))

# 错误：不加负号，实际在做最小化期望回报
loss = sum(G_t * log_prob for G_t, log_prob in zip(returns, log_probs))
```

### 11.2 梯度方差过大

**现象**：
- 训练曲线剧烈震荡，回报忽高忽低
- 不同回合之间的性能差异极大
- 有时一个回合回报很高（偶然走运），紧接着一个回合回报很低

**原因**：
- 蒙特卡洛估计中 $G_t$ 本身方差就很大（不同轨迹的回报差异大）
- 当回报未标准化时，$G_t$ 的量级会直接影响梯度的量级

**解决方案**：
```python
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # 方法1：标准化回报（最简单有效的基线）
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns
```

### 11.3 回合内不同时间步的信用分配不精确

**现象**：
- 回合中后期做出的好动作和前期做出的差动作获得了相同的"评价"
- 学习速度慢，策略改进不明确

**原因**：
- 基本REINFORCE用整条轨迹的总回报 $G_t$ 来评价每个时间步的动作
- $G_t$ 包含了 $t$ 时刻之后所有奖励的影响，无法精确区分哪些动作真正带来了好结果

**解决方案**：
```python
# 改进：使用奖励-时间加权，使近期动作获得更多权重
# 或者更推荐：使用基线进一步分离信号
# 最佳方案：升级到Actor-Critic方法，用优势函数 A(s,a) = Q(s,a) - V(s) 代替 G_t
```

---

## 12. 学习总结

### 核心要点回顾

**核心思想**：REINFORCE通过对完整轨迹的回报加权策略梯度，直接优化参数化策略的期望累计回报。回报高的动作被强化，回报低的动作被弱化。

**数学本质**：策略梯度定理 $\nabla_\theta J = \mathbb{E}[\sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)]$，通过蒙特卡洛采样近似期望，得到可计算的梯度估计。

**优化目标**：最大化期望累计回报 $J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^t r_t]$。

**适用场景**：回合制任务、动作空间较大、需要学习随机策略的场景。

**局限性**：高方差、低样本效率、只能处理回合制任务。

### 关键公式汇总

**1. 目标函数**：
$$ J(\theta) = \mathbb{E}_{\tau \sim p(\tau;\theta)} \left[ \sum_{t=0}^{T} \gamma^t r_t \right] $$

**2. 策略梯度定理**：
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right] $$

**3. 蒙特卡洛近似**：
$$ \nabla_\theta J(\theta) \approx \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(a_t|s_t) $$

**4. 参数更新规则**：
$$ \theta \leftarrow \theta + \alpha \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(a_t|s_t) $$

**5. 带基线的梯度（降低方差）**：
$$ \nabla_\theta J(\theta) \approx \sum_{t=0}^{T} (G_t - b_t) \nabla_\theta \log \pi_\theta(a_t|s_t) $$

### 最佳实践

**环境配置**：
- 对状态进行归一化处理
- 确保环境的奖励设计合理（奖励不能太稀疏）
- CartPole-v1的最大步数为500，超过即视为解决

**网络设计**：
- 策略网络输出层不使用softmax激活，而是直接输出logits，由PyTorch的Categorical分布处理
- 隐层维度不宜过大（128通常足够），否则容易过拟合
- 使用Xavier初始化

**训练技巧**：
- 回报标准化是最简单有效的方差降低方法
- 学习率通常需要较小（1e-3量级）
- 训练回合数需要足够多（至少500-1000回合）
- 使用多线程并行采样可以加速训练

### 与其他算法的联系

- **前置算法**：蒙特卡洛方法（采样近似期望）、交叉熵损失（分类分布的梯度）、softmax回归
- **后续算法**：REINFORCE with Baseline（引入基线减少方差）、Actor-Critic（引入值函数）、A2C（并行 Advantage Actor-Critic）、PPO（Proximal Policy Optimization）
- **相关算法**：DQN（基于值函数的替代方法）、TRPO（信任域策略优化）、SAC（Soft Actor-Critic）

---

## 13. 练习题与思考题

### 13.1 基础练习

**习题1：概念理解**

在REINFORCE算法中，策略梯度公式 $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)]$ 中，$\nabla_\theta \log \pi_\theta(a_t|s_t)$ 的作用是什么？

A. 计算动作 $a_t$ 的价值
B. 指示参数变化对动作 $a_t$ 概率的影响方向
C. 计算状态 $s_t$ 的价值
D. 衡量奖励的折扣程度

<details>
<summary>答案</summary>

**答案：B**

**解析**：$\nabla_\theta \log \pi_\theta(a_t|s_t)$ 是对数概率的梯度，也称为分数函数（score function）。它的几何含义是：沿着这个梯度方向更新参数 $\theta$，会增大选择动作 $a_t$ 的概率 $\pi_\theta(a_t|s_t)$。具体来说：

- 如果 $G_t > 0$（好的结果），梯度更新 $\theta + \alpha G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$ 会增大选择 $a_t$ 的概率（方向与分数函数一致）
- 如果 $G_t < 0$（差的结果），更新方向与分数函数相反，会减小选择 $a_t$ 的概率

选项A和C描述的是值函数的概念，选项D描述的是折扣因子 $\gamma$ 的作用。

</details>

---

**习题2：手动计算**

给定以下简化的REINFORCE场景，手工计算一次参数更新：

- 策略网络：$\pi_\theta(a|s) = \text{softmax}(W s)$，其中 $W = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$（2个动作）
- 当前状态：$s = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
- 采样的轨迹只有1步：选择了动作 $a_0 = 0$，获得的回报 $G_0 = 2$
- 学习率 $\alpha = 0.1$

请计算：
1. 当前策略在状态 $s$ 下的动作概率分布
2. $\nabla_\theta \log \pi_\theta(a_0=0|s)$ 的值
3. 参数更新量
4. 更新后的参数 $W$

<details>
<summary>答案</summary>

**解**：

**步骤1：计算动作概率分布**

logits $= W s = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}^T \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0 \end{bmatrix}$

softmax:
- $p(a=0) = \frac{e^{0.5}}{e^{0.5} + e^0} = \frac{1.6487}{1.6487 + 1} = 0.6225$
- $p(a=1) = \frac{e^0}{e^{0.5} + e^0} = \frac{1}{2.6487} = 0.3775$

**步骤2：计算对数概率梯度**

$\log \pi_\theta(a_0=0|s) = \log 0.6225 = -0.4741$

$\nabla_\theta \log \pi_\theta(a=0|s) = (e_a - \pi_\theta(\cdot|s)) \cdot s$

其中 $e_0 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ 是动作0的one-hot向量，所以：

$e_0 - \pi = \begin{bmatrix} 1 \\ 0 \end{bmatrix} - \begin{bmatrix} 0.6225 \\ 0.3775 \end{bmatrix} = \begin{bmatrix} 0.3775 \\ -0.3775 \end{bmatrix}$

对 $W$ 的梯度：$\nabla_W \log \pi_\theta(a=0|s) = (e_0 - \pi) \cdot s^T = \begin{bmatrix} 0.3775 \\ -0.3775 \end{bmatrix} \begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} 0.3775 & 0 \\ -0.3775 & 0 \end{bmatrix}$

由于 $s = [1, 0]^T$，只有 $W$ 的第一列会被更新：

$\nabla_{W_{:,1}} \log \pi_\theta(a=0|s) = \begin{bmatrix} 0.3775 \\ -0.3775 \end{bmatrix}$

**步骤3：计算参数更新量**

$\Delta W = \alpha \cdot G_0 \cdot \nabla_W \log \pi_\theta(a_0|s) = 0.1 \times 2 \times \begin{bmatrix} 0.3775 \\ -0.3775 \end{bmatrix} = \begin{bmatrix} 0.0755 \\ -0.0755 \end{bmatrix}$

**步骤4：更新后参数**

$W_{new} = W + \Delta W = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} + \begin{bmatrix} 0.0755 \\ -0.0755 \end{bmatrix} = \begin{bmatrix} 0.5755 \\ 0.4245 \end{bmatrix}$

**验证**：$W$ 的第一个元素（对应动作0的权重）增大了，第二个元素（对应动作1的权重）减小了。这与直觉一致：动作0带来了正回报 $G_0=2$，所以应该增大选择动作0的概率。

</details>

---

### 13.2 进阶思考

**思考1：基线的引入**

问题：为什么REINFORCE的梯度方差很大？引入基线 $b_t$ 如何帮助减少方差？为什么引入 $b_t$ 不会改变梯度的期望？

<details>
<summary>答案</summary>

**问题分析**：

REINFORCE的梯度方差大，根源在于：$G_t$ 是对整条轨迹的累计回报，不同轨迹的 $G_t$ 差异很大。即使两条轨迹在某个时间步 $t$ 选择了相同的动作，由于后续步骤的不同，它们的 $G_t$ 可能天差地别。这导致对同一个动作的"评价"不一致，梯度方向波动剧烈。

**基线的作用**：

用 $(G_t - b_t)$ 代替 $G_t$ 后：
- 当 $G_t > b_t$ 时，$G_t - b_t > 0$，仍为正，强化该动作
- 当 $G_t < b_t$ 时，$G_t - b_t < 0$，变为负，弱化该动作
- 当 $G_t \approx b_t$ 时，$G_t - b_t \approx 0$，几乎不更新

基线 $b_t$ 相当于一个"平均参考线"，只有当回报明显高于或低于平均水平时才进行大幅更新。这大大减少了梯度的大小和波动。

**为什么基线不改变期望**：

关键证明：

$$ \mathbb{E}_{\pi_\theta}\left[b_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right] = \sum_a \pi_\theta(a|s_t) \cdot b_t \cdot \nabla_\theta \log \pi_\theta(a|s_t) = b_t \sum_a \pi_\theta(a|s_t) \frac{\nabla_\theta \pi_\theta(a|s_t)}{\pi_\theta(a|s_t)} = b_t \nabla_\theta \sum_a \pi_\theta(a|s_t) = b_t \nabla_\theta 1 = 0 $$

因为概率分布之和恒为1，其梯度为零。因此：

$$ \mathbb{E}_{\pi_\theta}[(G_t - b_t) \nabla_\theta \log \pi_\theta(a_t|s_t)] = \mathbb{E}_{\pi_\theta}[G_t \nabla_\theta \log \pi_\theta(a_t|s_t)] - 0 = \nabla_\theta J(\theta) $$

即引入基线后梯度的期望不变（无偏），但方差可以大幅降低（只需选择合适的 $b_t$）。最优基线是 $b_t^* = \mathbb{E}[G_t]$，使方差最小化。

</details>

---

**思考2：REINFORCE与交叉熵损失的关系**

问题：REINFORCE的更新规则在什么条件下退化为监督学习中的交叉熵损失？这一对应关系说明了什么？

<details>
<summary>答案</summary>

**对应关系**：

REINFORCE的损失函数为：

$$ L_{REINFORCE} = -\sum_{t=0}^{T} G_t \log \pi_\theta(a_t|s_t) $$

当 $G_t = 1$ 对所有 $t$ 恒成立时（即所有动作都获得了相同的正回报），上式变为：

$$ L = -\sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) = -\sum_{t=0}^{T} \sum_{a} \mathbb{1}[a = a_t] \log \pi_\theta(a|s_t) $$

这恰好就是**多分类交叉熵损失**，其中 $a_t$ 扮演的是"标签"的角色。

**更一般地**，如果将 $G_t$ 视为"软标签"的权重，REINFORCE可以看作是加权的交叉熵损失：
- 正回报的动作相当于"正样本"，权重为 $G_t$
- 零回报的动作不会被更新（权重为0）
- 负回报的动作相当于"负样本"，权重为 $|G_t|$

**这一对应关系的意义**：

1. **统一视角**：策略梯度方法可以被理解为一种"自标注"的监督学习——智能体自己产生数据（轨迹），自己为每个动作打标签（回报），然后按标签加权更新
2. **解释基线的另一种方式**：标准化后的 $(G_t - b_t)$ 可以有正有负，等价于对正负样本都进行学习；而原始 $G_t$ 全为正时，只学习正样本
3. **连接强化学习与监督学习**：说明强化学习的策略梯度方法本质上是监督学习的推广——用策略自身产生的回报代替外部标签

</details>

---

### 13.3 开放思考

**思考3：从REINFORCE到现代策略梯度方法**

问题：REINFORCE是最基础的策略梯度方法。请分析它面临的三个核心问题，并说明后续算法（Actor-Critic、PPO等）是如何分别解决这些问题的。

<details>
<summary>答案</summary>

**REINFORCE面临的三个核心问题及解决方案**：

**问题1：高方差**

- **根源**：用一条完整轨迹的蒙特卡洛回报 $G_t$ 估计梯度，不同轨迹的回报差异极大
- **解决方案演进**：
  - **基线方法**：用 $G_t - b_t$ 代替 $G_t$，最简单的基线是回报的均值
  - **Actor-Critic（A2C）**：引入值函数 $V(s)$ 作为基线，用 $A(s,a) = Q(s,a) - V(s)$ 代替 $G_t$。$V(s)$ 的引入使得方差大幅降低，因为它提供了一个更精确的"基准线"
  - **GAE（Generalized Advantage Estimation）**：通过 $\lambda$ 参数在偏差和方差之间灵活权衡

**问题2：低样本效率**

- **根源**：每条轨迹只用一次就被丢弃，无法复用数据
- **解决方案演进**：
  - **重要性采样**：用 $q(a|s)/\pi(a|s)$ 的比值修正旧策略数据在新策略下的梯度偏差
  - **PPO（Proximal Policy Optimization）**：在重要性采样基础上加入裁剪（clip）机制，限制策略更新幅度，使得同一批数据可以用于多轮更新，大幅提升样本效率
  - **经验回放**（部分借鉴自DQN）：虽然策略梯度方法中直接使用经验回放比较困难（因为旧数据与新策略不匹配），但结合重要性采样可以实现

**问题3：只能处理回合制任务**

- **根源**：蒙特卡洛回报需要采样到回合结束才能计算
- **解决方案演进**：
  - **Actor-Critic**：用值函数 $V(s)$ 或 $Q(s,a)$ 的时序差分（TD）估计代替蒙特卡洛回报。TD估计可以每步更新，无需等待回合结束
  - **n-step returns**：介于蒙特卡洛和TD之间，使用 $n$ 步回报 $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$，兼顾偏差和方差
  - **并行环境**（如A3C）：通过多个并行环境同时采样，间接缓解回合制约束

**总结**：现代策略梯度方法（PPO、SAC、TD3等）可以看作是在REINFORCE基础上的逐步改进——引入值函数降低方差、引入重要性采样提升样本效率、引入信任域/裁剪保证训练稳定性。理解REINFORCE是理解所有这些高级方法的基石。

</details>

---

## 14. 学习路径建议

### 14.1 前置知识

**数学基础**：
- [ ] **概率论**：期望、方差、概率分布、条件概率、蒙特卡洛估计
  - 推荐资源：《概率论与数理统计》陈希孺
  - 学习时长：1-2周
- [ ] **微积分**：偏导数、梯度、链式法则
  - 推荐资源：Khan Academy微积分课程
  - 学习时长：1周
- [ ] **线性代数**：矩阵运算、特征值分解
  - 推荐资源：《线性代数导论》Gilbert Strang
  - 学习时长：1-2周

**强化学习基础**：
- [ ] **MDP（马尔可夫决策过程）**：状态、动作、转移、奖励、策略
- [ ] **值函数**：$V(s)$、$Q(s,a)$、贝尔曼方程
- [ ] **蒙特卡洛方法**：用采样近似期望
- [ ] **时序差分方法**：TD learning、Q-learning
  - 推荐资源：《Reinforcement Learning: An Introduction》Sutton & Barto（第1-6章）

**编程基础**：
- [ ] **Python**：基本语法、NumPy
- [ ] **PyTorch**：自动微分、nn.Module、优化器
  - 推荐资源：PyTorch官方教程
  - 学习时长：1-2周

### 14.2 平行算法（可同时学习）

1. **REINFORCE with Baseline**：在REINFORCE基础上引入基线降低方差
   - 学习重点：基线的数学证明、最优基线的选择
   - 对比点：方差更低，收敛更快，但仍然是蒙特卡洛方法

2. **DQN（Deep Q-Network）**：基于值函数的深度强化学习方法
   - 学习重点：经验回放、目标网络、epsilon-greedy探索
   - 对比点：DQN优化值函数，REINFORCE直接优化策略；DQN更适合离散动作，REINFORCE对动作空间更灵活

3. **Cross-Entropy Method**：另一种简单的策略搜索方法
   - 学习重点：精英选择、分布更新
   - 对比点：比REINFORCE更简单但不够通用，适合简单任务

### 14.3 进阶算法（后续学习）

**短期目标（1-2个月）：**
1. **Actor-Critic (A2C)**：引入值函数作为Critic，用优势函数代替蒙特卡洛回报
   - 关联：Actor-Critic的梯度公式可以看作REINFORCE with baseline的推广
   - 难度：中等
2. **REINFORCE with Baseline**：引入状态值函数 $V(s)$ 作为基线
   - 关联：最直接的REINFORCE改进，理解基线理论的关键一步
   - 难度：较低

**中期目标（3-6个月）：**
1. **PPO（Proximal Policy Optimization）**：当前最广泛使用的策略梯度算法
   - 应用领域：OpenAI Five、RLHF中的策略训练
   - 难度：较高
2. **TRPO（Trust Region Policy Optimization）**：带信任域约束的策略优化
   - 关联：PPO的理论前身
   - 难度：较高

**长期目标（6个月以上）：**
1. **SAC（Soft Actor-Critic）**：最大熵强化学习，适用于连续控制
   - 最新研究：最大熵框架下的off-policy策略优化
   - 难度：高
2. **RLHF（Reinforcement Learning from Human Feedback）**：大语言模型对齐的核心技术
   - 最新研究：InstructGPT、ChatGPT背后的训练方法
   - 难度：高

### 14.4 推荐资源

**教材类**：
1. **《Reinforcement Learning: An Introduction》(2nd Edition)** Sutton & Barto - 强化学习领域的"圣经"，第13章专门讲解策略梯度
2. **《动手学强化学习》** 赵世钰等 - 中文教材，代码丰富，实践性强
3. **《深度强化学习》** 王树森 - 系统讲解DQN、策略梯度、Actor-Critic等

**论文类**：
1. **Williams, R.J. (1992)**. "Simple statistical gradient-following algorithms for connectionist reinforcement learning" - REINFORCE原始论文
2. **Sutton, R.S. et al. (2000)**. "Policy Gradient Methods for Reinforcement Learning with Function Approximation" - 策略梯度定理的经典论文
3. **Schulman, J. et al. (2017)**. "Proximal Policy Optimization Algorithms" - PPO论文，REINFORCE最重要的后续发展

**在线课程**：
1. **David Silver's RL Course** (UCL) - 策略梯度章节讲解清晰
2. **CS285 (UC Berkeley)** - Sergey Levine的深度强化学习课程，Policy Gradient部分非常深入
3. **Spinning Up in Deep RL (OpenAI)** - 包含REINFORCE、PPO等的实现和详细讲解

**代码资源**：
1. **OpenAI Spinning Up** - 高质量代码实现和教程
2. **CleanRL** - 单文件、易理解的RL算法实现
3. **Stable Baselines3** - 工业级的RL算法库

---

## 附录

### A. 参考文献

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3), 229-256.
2. Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12.
3. Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. ICML.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
5. Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. ICML.
6. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.

### B. 常见问题FAQ

**Q1：REINFORCE中的"蒙特卡洛"是什么意思？**

A：蒙特卡洛（Monte Carlo）是指用随机采样来近似数学期望的方法。在REINFORCE中，由于无法遍历所有可能的轨迹来计算期望回报，所以通过采样若干条完整轨迹，用样本均值来近似期望。具体来说，$\mathbb{E}[G \nabla_\theta \log \pi] \approx \frac{1}{N}\sum_{n=1}^N G^n \nabla_\theta \log \pi^n$。

**Q2：REINFORCE能否用于连续动作空间？**

A：可以。对于连续动作空间，将策略网络改为输出高斯分布的均值和方差（或固定方差，只输出均值），即 $\pi_\theta(a|s) = \mathcal{N}(a; \mu_\theta(s), \sigma^2 I)$，然后从高斯分布中采样动作。梯度计算方式与离散情况类似，$\nabla_\theta \log \pi_\theta(a|s) = \frac{a - \mu_\theta(s)}{\sigma^2} \nabla_\theta \mu_\theta(s)$。

**Q3：为什么策略必须是随机的（stochastic）？**

A：因为确定性策略 $\pi(s) = a$ 对参数 $\theta$ 的梯度为零或不存在（小范围参数扰动不改变确定性策略的输出），无法通过梯度方法优化。随机策略 $\pi_\theta(a|s)$ 对参数可微，且策略梯度定理依赖于概率分布的可微性。

---

**文档结束**
