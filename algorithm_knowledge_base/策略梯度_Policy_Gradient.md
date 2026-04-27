# 策略梯度 (Policy Gradient) 学习文档

> 来源线索：本节内容根据原书第9章关于"策略梯度算法"的相关章节整理、扩展与教学化改写。

> 直接优化策略函数，用对数微分技巧将累乘变累加，从采样中学习最优行为。

## 1. 算法基础认知

**一句话定义**：策略梯度直接对参数化策略进行优化，通过采样轨迹的回报来估计梯度方向。

**直觉类比**：想象你在练习射箭。基于价值的方法（如 DQN）会告诉你"这个位置瞄准目标中心的期望得分是多少"，你根据得分表选最高分的动作。而策略梯度方法直接让你形成"在这个位置应该怎么射"的肌肉记忆——不关心具体分数，只根据每次射击结果的好坏来调整你的射箭动作概率。如果某次射箭结果好（回报高），就增加这个动作的概率；如果结果差（回报低），就减少这个动作的概率。经过大量练习后，好的射箭动作被强化，差的被淘汰。

**历史背景**：策略梯度定理的现代形式由 Sutton 等人在 1999 年确立，但核心思想可追溯到 Williams 的 REINFORCE 算法（1992）。REINFORCE 证明了策略梯度可以用采样的回报来无偏估计，这为策略梯度方法奠定了理论和实践基础。此后，Actor-Critic（2000年代）、TRPO（2015）、PPO（2017）等算法都是在策略梯度框架上的改进。

**算法定位**：策略梯度是一类算法的统称，属于基于策略的（policy-based）方法，与基于价值的（value-based）方法互补。在强化学习的算法谱系中，策略梯度方法占据了"直接优化策略"这一重要位置——它不通过价值函数间接推导策略，而是直接搜索最优策略的参数。REINFORCE 是策略梯度的最简实现，Actor-Critic 是引入价值函数的增强版，PPO 是加上约束优化的工业级版本。

**前置知识**：
- **马尔可夫决策过程（MDP）**：理解状态 $s$、动作 $a$、奖励 $r$、转移概率 $P(s'|s,a)$ 和折扣因子 $\gamma$ 的概念。策略梯度在 MDP 框架下推导，需要知道什么是轨迹、什么是回报。
- **梯度下降/梯度上升**：理解梯度的含义（函数增长最快的方向），以及为什么沿梯度方向更新参数可以优化目标函数。策略梯度方法用的是梯度上升（最大化期望回报）。
- **概率论基础**：条件概率 $\pi(a|s)$（在状态 $s$ 下选动作 $a$ 的概率）、期望 $\mathbb{E}$（平均值的数学表示）、对数概率 $\log\pi$（将对数微分技巧转化为可计算的梯度）。

**为什么需要策略梯度**：基于价值的方法（如 DQN）有两个根本性限制——(1) 只能处理离散动作空间（需要用 argmax 选动作，连续空间中无法枚举）；(2) 只能学习确定性策略（总是选 Q 值最高的动作，无法表示混合策略）。策略梯度方法通过直接参数化策略函数 $\pi_\theta(a|s)$ 绕过了这两个限制，使得连续动作空间和随机策略成为可能。

## 2. 核心原理

### 核心思想

不通过价值函数间接得到策略，而是直接参数化策略函数 $\pi_\theta(a|s)$，通过梯度上升最大化策略的期望回报。这种"直接优化"的思路与基于价值的方法有本质区别：DQN 先学习 Q 值表，然后从中推导最优策略（取 argmax）；策略梯度则跳过 Q 值表，直接在策略空间中搜索最优参数。

### 工作流程

1. 初始化策略参数 $\theta$（通常用神经网络参数化）
2. 用当前策略 $\pi_\theta$ 与环境交互，采集轨迹（一组完整的 $(s, a, r)$ 序列）
3. 利用策略梯度定理计算梯度估计：$\nabla_\theta J \approx \sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$
4. 梯度上升更新参数：$\theta \leftarrow \theta + \alpha \nabla_\theta J$
5. 重复步骤 2~4

这个流程与监督学习的训练循环非常相似——采集数据、计算梯度、更新参数。关键区别在于：监督学习的梯度来自标注数据的损失函数，策略梯度的梯度来自环境交互的回报信号。

### 关键概念

- **随机性策略 $\pi_\theta(a|s)$**：输出动作概率分布，而非确定性动作。随机策略有两个好处：一是自然支持探索（概率分配使所有动作都有机会被尝试），二是适用于需要混合策略的场景（如博弈论中的纳什均衡）。
- **对数微分技巧**：$\nabla_\theta P_\theta(\tau) = P_\theta(\tau) \nabla_\theta \log P_\theta(\tau)$，将累乘变累加。这是策略梯度推导的核心数学工具，没有它，轨迹概率的连乘形式无法有效求导。
- **轨迹**：一组状态-动作序列 $\tau = \{s_0, a_0, r_1, s_1, a_1, \ldots\}$，是策略梯度方法的基本数据单元。一条轨迹对应一次完整的环境交互。
- **平稳分布**：马尔可夫链长期运行后的稳定状态分布 $d^\pi(s)$。它描述了在策略 $\pi$ 下，各状态被访问的频率，用于将轨迹级梯度转化为时步级梯度。

### 策略函数设计

**离散动作空间**：用 softmax 函数

$$\pi_\theta(s, a) = \frac{e^{\phi(s,a)^T\theta}}{\sum_b e^{\phi(s,b)^T\theta}}$$

其中 $\phi(s,a)$ 是状态-动作对的特征向量。softmax 保证输出是合法的概率分布（所有动作概率之和为 1）。

**连续动作空间**：用高斯分布

$$a \sim \mathcal{N}(\phi(s)^T\theta, \sigma^2)$$

策略网络输出动作的均值 $\mu(s)$，方差 $\sigma^2$ 可以是固定的或由网络输出。连续策略通过采样从高斯分布中获得具体动作值。

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $\pi_\theta(a\|s)$ | 参数化策略，给定状态$s$下选择动作$a$的概率 |
| $J(\theta)$ | 策略的目标函数（期望回报） |
| $\nabla_\theta J$ | 策略梯度，目标函数对策略参数的梯度 |
| $Q^{\pi}(s,a)$ | 策略$\pi$下的动作价值函数 |
| $V^{\pi}(s)$ | 策略$\pi$下的状态价值函数 |
| $A^{\pi}(s,a)$ | 优势函数，$A = Q - V$ |
| $G_t$ | 从时步$t$开始的回报 |

### 策略梯度定理

策略梯度定理是整个策略梯度方法的理论基础，由Sutton等人在1999年证明：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\right] \tag{9.1}$$

**推导思路**（简化版）：

1. 目标函数：$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$，即策略$\pi_\theta$下轨迹回报的期望
2. 展开：$J(\theta) = \int p_\theta(\tau) R(\tau) d\tau$
3. 对$\theta$求梯度：$\nabla_\theta J = \int \nabla_\theta p_\theta(\tau) R(\tau) d\tau$
4. 利用 $\nabla_\theta p = p \cdot \nabla_\theta \log p$（对数导数技巧）：
   $\nabla_\theta J = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) d\tau = \mathbb{E}[\nabla_\theta \log p_\theta(\tau) R(\tau)]$
5. 展开 $\log p_\theta(\tau) = \sum_t \log \pi_\theta(a_t|s_t) + \text{(与$\theta$无关的项)}$
6. 最终得到策略梯度定理的形式

### 对数导数技巧的直觉

对数导数技巧 $\nabla_\theta p = p \cdot \nabla_\theta \log p$ 是策略梯度的数学核心。它的直觉含义是：我们不直接对概率$p$求导（这很困难，因为$p$必须满足归一化约束），而是对$\log p$求导（这更简单，因为$\log$把乘法变加法，且消除了归一化常数）。

### 策略梯度的不同变体

将策略梯度定理中的 $Q^{\pi}(s,a)$ 替换为不同的估计，可以得到不同的算法：

| 替换量 | 算法 | 偏差 | 方差 |
|--------|------|------|------|
| $G_t$（MC回报） | REINFORCE | 无偏 | 高 |
| $G_t - b$（带基线） | REINFORCE+baseline | 无偏 | 中 |
| $r + \gamma V(s') - V(s)$（TD误差） | Actor-Critic | 有偏 | 低 |
| $\sum (\gamma\lambda)^l \delta_{t+l}$（GAE） | PPO+GAE | 有偏 | 较低 |

**重要性质**：替换量只需要满足 $\mathbb{E}[\hat{A}(s,a) \nabla_\theta \log \pi(a|s)] = \nabla_\theta J$ 即可保证策略梯度方向正确。这就是为什么可以用基线（减去$V(s)$）来降低方差而不引入偏差——因为基线不依赖于动作$a$。
## 4. 训练过程讲解

数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 参数化策略 |
| $\tau$ | 轨迹 $(s_0, a_0, r_1, s_1, \ldots)$ |
| $P_\theta(\tau)$ | 轨迹概率 |
| $R(\tau)$ | 轨迹回报 |
| $d^\pi(s)$ | 策略 $\pi$ 下的状态平稳分布 |
| $G_t$ | 从 $t$ 时刻开始的折扣回报 |

### 轨迹概率

$$P_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)$$

### 目标函数

$$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int_\tau P_\theta(\tau) R(\tau) d\tau$$

### 策略梯度推导

**利用对数微分技巧**：

$$\nabla_\theta J = \int_\tau \nabla_\theta P_\theta(\tau) R(\tau) = \int_\tau P_\theta(\tau) \nabla_\theta \log P_\theta(\tau) R(\tau)$$

**展开 $\log P_\theta(\tau)$**：

$$\log P_\theta(\tau) = \underbrace{\log p(s_0)}_{\text{与}\theta\text{无关}} + \sum_t \left[\underbrace{\log \pi_\theta(a_t|s_t)}_{\text{与}\theta\text{有关}} + \underbrace{\log p(s_{t+1}|s_t,a_t)}_{\text{与}\theta\text{无关}}\right]$$

因此：

$$\nabla_\theta \log P_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**最终策略梯度**：

$$\nabla_\theta J = \mathbb{E}_{\tau}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

### 基于平稳分布的推导

$$J(\theta) = \sum_{s} d^\pi(s) \sum_a \pi_\theta(a|s) Q^\pi(s, a)$$

$$\nabla_\theta J = \mathbb{E}_{\pi_\theta}[Q^\pi(s,a) \nabla_\theta \log \pi_\theta(a|s)]$$

这个更通用的形式将 $Q^\pi(s,a)$ 替换为任何优势估计（如 $G_t$ 或 $A(s,a)$），是 Actor-Critic 算法的理论基础。

### 引入基线

$$\nabla_\theta J = \mathbb{E}_{\pi_\theta}[(Q^\pi(s,a) - b(s)) \nabla_\theta \log \pi_\theta(a|s)]$$

基线 $b(s)$（通常取 $V(s)$）不改变期望但降低方差。

### 引入基线的数学证明
策略梯度引入基线 $b(s)$ 后，关键需要证明 $\mathbb{E}_{\pi}[b(s) \nabla_\theta \log\pi(a|s)] = 0$：

$$\mathbb{E}_{\pi}[b(s) \nabla_\theta \log\pi(a|s)] = \sum_s d^\pi(s) b(s) \sum_a \pi(a|s) \nabla_\theta \log\pi(a|s)$$

由于 $\nabla_\theta \log\pi(a|s) = \frac{\nabla_\theta \pi(a|s)}{\pi(a|s)}$，所以 $\pi(a|s) \nabla_\theta \log\pi(a|s) = \nabla_\theta \pi(a|s)$。对动作求和：$\sum_a \nabla_\theta \pi(a|s) = \nabla_\theta \sum_a \pi(a|s) = \nabla_\theta 1 = 0$。因此基线项为 0，不改变梯度的期望。

这个证明揭示了策略梯度的一个重要性质：任何只依赖状态（不依赖动作）的函数都可以作为基线。最优基线是 $V(s)$，它使梯度方差最小化。## 4. 训练过程讲解

### 数据预处理

- 状态通常直接使用，或做标准化（减均值除标准差），确保输入数值范围合理
- 回报计算：从后往前折扣累加 $G_t = r_t + \gamma G_{t+1}$，注意初始值 $G_T = r_T$
- 回报标准化：$(G - \mu) / \sigma$，这是策略梯度中稳定梯度最关键的一步。不做标准化时，长回合的回报远大于短回合，导致梯度被长回合主导，训练极不稳定。

### 参数初始化

- 策略网络参数：Xavier 或默认初始化。不建议用太大的初始值，否则初始策略可能偏向某个动作
- 学习率是策略梯度中最敏感的超参数。过大导致策略崩溃（一次更新就破坏了好不容易学到的策略），过小导致学习缓慢

### 迭代过程

1. 用当前策略采集一个或多个完整轨迹。REINFORCE 通常每回合采集一条轨迹，但也可以一次采多条取平均降低方差
2. 计算每步的折扣回报 $G_t$（从后往前累加）
3. 标准化回报：`returns = (returns - mean) / (std + 1e-8)`
4. 计算策略梯度：$\hat{g} = \sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$
5. 梯度上升：$\theta \leftarrow \theta + \alpha \hat{g}$（在 PyTorch 中通过对负损失做梯度下降实现）
6. 清空缓冲区，准备下一回合
7. 重复直到收敛

### 收敛条件

- 策略性能（回合奖励）连续 N 个回合不再上升（N 取 20~50）
- 策略熵降到很低（策略接近确定性，表示已经"确信"最优动作）

### 超参数表

| 参数 | 作用 | 推荐范围 | 默认 |
|------|------|----------|------|
| lr | 学习率 | 1e-4~1e-2 | 1e-3 |
| $\gamma$ | 折扣因子 | 0.95~0.99 | 0.99 |
| hidden_dim | 隐藏层维度 | 64~256 | 128 |

### 训练技巧

- **使用基线**：引入 $V(s)$ 作为基线，用 $G_t - V(s)$ 替代 $G_t$，方差可降低 50% 以上。这是从 REINFORCE 进化到 Actor-Critic 的核心改进。
- **批量采集**：每次采集多条轨迹（如 10 条），取梯度平均，进一步降低方差。
- **学习率调度**：训练初期用较大学习率快速探索，后期逐渐降低进行精细调整。

## 5. 应用场景

### 1. 连续动作控制（机器人、自动驾驶）
策略梯度可直接输出连续动作（高斯策略），无需离散化。在机器人关节控制中，动作空间是连续的角度值（如机械臂 7 个关节角度），如果用 DQN 需要将每个角度离散化为几十个级别，精度损失严重且动作空间指数爆炸。策略梯度通过高斯分布直接输出连续值，完全避免了离散化问题。

### 2. 随机策略需求（博弈对抗）
在博弈问题中需要混合策略（如石头剪刀布中的随机出拳），策略梯度自然输出概率分布。纯价值方法（如 DQN）倾向于学习确定性策略（总是出石头），在对抗中容易被对手利用。策略梯度的随机性使得学习到的策略天然具有不可预测性。

### 3. 大动作空间或不可微的动作空间
当动作空间很大（如词汇表有数万个候选词）或动作本身不可微（如离散的组合优化决策）时，基于价值的方法需要为每个动作计算 Q 值，计算量巨大。策略梯度直接在策略空间中搜索，只需要对采样到的动作计算梯度。

### 4. 自然语言处理的 RLHF
在大语言模型的对齐训练中，RLHF 使用策略梯度（PPO）优化文本生成策略。策略网络就是语言模型本身，动作是生成的 token 序列，奖励来自人类偏好模型。这是策略梯度方法在工业界最成功的应用之一。

### 5. 自动化交易与资源调度
在金融交易、云计算资源分配等场景中，动作空间通常是连续的（买入/卖出数量、资源分配比例），且环境具有非平稳性。策略梯度的在线学习能力使其能够适应环境变化。

#

### 5.3 自然语言处理（RLHF）

ChatGPT等大语言模型的对齐训练使用了策略梯度方法。在RLHF（Reinforcement Learning from Human Feedback）流程中：(1) 首先训练一个奖励模型，学习人类对回复质量的偏好；(2) 然后用PPO（策略梯度的一种）优化语言模型的输出，使其生成的回复最大化奖励模型的评分。这里语言模型是策略网络，prompt是状态，生成的token是动作，奖励模型提供反馈信号。策略梯度的核心优势在于它可以处理连续的、高维的动作空间（词汇表通常有数万到数十万个token），而Q-Learning等方法在这种动作空间下几乎不可行。

### 5.4 游戏AI（星际争霸）

DeepMind的AlphaStar使用了策略梯度方法（具体是V-trace + PPO的变体）来训练星际争霸II的AI。AlphaStar的策略网络直接输出动作概率分布（建筑选择、单位控制、技能使用等），动作空间极大且包含连续成分（如小地图点击位置）。策略梯度方法的可扩展性使其能处理这种复杂任务——AlphaStar在完整游戏中达到了 Grandmaster 级别，排名前0.2%的人类玩家。

### 5.5 机器人运动控制
策略梯度方法在机器人步态学习、机械臂操作等领域有广泛应用。高斯策略可以直接输出连续的关节角度或力矩，而且策略梯度能学到平滑、自然的运动轨迹。Boston Dynamics 和 OpenAI 的机械手解魔方项目都使用了策略梯度方法的变体。

### 不适用场景
- 简单离散任务（Q-learning 更高效，实现更简单）
- 需要极高样本效率的场景（策略梯度是同策略方法，数据只用一次）
- 纯表格型小状态空间（用动态规划更直接）
- 需要完全离线学习的场景（策略梯度需要与环境交互采集数据）

**应用选择指南**：选择算法时，首先判断动作空间类型（离散用DQN系列，连续用DDPG/TD3/SAC），其次判断样本效率需求（高用异策略方法，低用同策略方法），最后判断稳定性需求（高用PPO/TD3）。

## 6. 优缺点分析

### 优点

1. **适用于连续动作空间**：不受离散动作限制，可以直接输出连续值（如高斯策略的均值和方差）。成立条件：策略函数支持连续输出。这是策略梯度相比 DQN 最大的优势——DQN 只能处理离散动作，而策略梯度天然支持连续动作。

2. **无偏梯度估计**：策略梯度定理保证 $\hat{g} = \sum_t G_t \nabla_\theta \log\pi_\theta(a_t|s_t)$ 是真实梯度的无偏估计。成立条件：采样数量足够（单条轨迹的梯度估计虽然无偏但方差很大，需要多条轨迹取平均）。无偏性意味着只要采样足够多，梯度估计会收敛到真实梯度方向。

3. **能学习随机策略**：自然输出概率分布，适用于需要混合策略的博弈场景。成立条件：环境需要随机性。在确定性策略下（如 DQN），对手可以完全预测你的行为；在随机策略下，对手无法预测你的具体动作，只能应对你的概率分布。

4. **更好的收敛保证**：在某些条件下保证收敛到局部最优（至少不会像 DQN 那样因为 Q 值过估计而发散）。成立条件：学习率调度合适（满足 Robbins-Monro 条件 $\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$）。

### 缺点

1. **高方差**：梯度估计方差大，需要大量样本才能得到可靠的梯度方向。一条轨迹的回报 $G_t$ 包含了所有后续步骤的随机性（后续动作的随机采样、状态转移的随机性），这些噪声直接传递到梯度估计中。缓解方案：引入基线 $V(s)$（如 Actor-Critic），使用 GAE，增加批量大小。

2. **收敛到局部最优**：策略梯度沿梯度方向更新，只保证收敛到局部最优，不保证全局最优。缓解方案：多随机种子取最佳结果，使用熵正则化鼓励探索以跳出局部最优。

3. **采样效率低**：同策略方法，采集的数据只用一次就丢弃。每一轮参数更新后，旧策略采集的数据就不适用了（因为梯度公式中的 $\pi_\theta$ 已经改变）。缓解方案：PPO 通过重要性采样实现数据的有限重用（通常 3~4 轮）。

### 对比

| 特性 | 策略梯度 | DQN | Actor-Critic |
|------|---------|-----|-------------|
| 动作空间 | 离散+连续 | 离散 | 离散+连续 |
| 方差 | 高 | 中 | 低 |
| 偏差 | 无偏 | 有偏 | 有偏 |
| 收敛性 | 局部最优 | 较好 | 较好 |
| 样本效率 | 低 | 高 | 中 |

从对比中可以看出，策略梯度在动作空间灵活性上占优，但在方差和样本效率上处于劣势。后续的 Actor-Critic 和 PPO 正是为了弥补这些劣势而提出的。

## 7. 调库实现

```python
"""REINFORCE 策略梯度实现 - PyTorch + Gymnasium (CartPole-v1)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np


class PolicyNetwork(nn.Module):
    """策略网络：输入状态，输出动作概率"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)  # softmax 输出动作概率


class REINFORCEAgent:
    """REINFORCE 策略梯度 Agent"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """采样动作并记录 log 概率"""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update(self):
        """策略梯度更新"""
        # 计算折扣回报
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        # 标准化回报，降低方差
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # 策略梯度损失
        log_probs = torch.stack(self.log_probs)
        loss = -(log_probs * returns).mean()  # 负号：梯度上升

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空缓冲区
        self.log_probs, self.rewards = [], []


def train_reinforce():
    env = gym.make('CartPole-v1')
    agent = REINFORCEAgent(4, 2, lr=1e-3, gamma=0.99)
    rewards_history = []

    for ep in range(500):
        state, _ = env.reset()
        ep_reward = 0
        while True:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            ep_reward += reward
            if terminated or truncated:
                break
        agent.update()
        rewards_history.append(ep_reward)
        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_history[-20:])
            print(f"回合 {ep+1}, 平均奖励: {avg:.1f}")

    env.close()
    return agent, rewards_history


if __name__ == "__main__":
    train_reinforce()
```

## 8. 手工代码实现

```python
"""策略梯度核心逻辑手工实现 - NumPy"""
import numpy as np


class SimplePolicy:
    """手工实现 softmax 策略"""
    def __init__(self, state_dim, action_dim):
        self.W = np.random.randn(state_dim, action_dim) * 0.01
        self.action_dim = action_dim

    def get_probs(self, state):
        """softmax 输出动作概率"""
        logits = state @ self.W
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定
        return exp_logits / exp_logits.sum()

    def sample(self, state):
        """根据概率采样动作"""
        probs = self.get_probs(state)
        return np.random.choice(self.action_dim, p=probs)

    def log_prob(self, state, action):
        """计算 log π(a|s)"""
        probs = self.get_probs(state)
        return np.log(probs[action] + 1e-10)

    def gradient_log_prob(self, state, action):
        """手工计算 ∇_θ log π(a|s)
        softmax 策略的梯度 = φ(s,a) - E_π[φ(s,·)]
        其中 φ(s,a) 是特征向量（这里简化为 one-hot × state）
        """
        probs = self.get_probs(state)
        # ∇_θ log π(a|s) = x(s) ⊗ (e_a - π(·|s))
        one_hot = np.zeros(self.action_dim)
        one_hot[action] = 1
        grad = np.outer(state, one_hot - probs)
        return grad


def policy_gradient_update(policy, trajectories, lr=0.01, gamma=0.99):
    """手工策略梯度更新"""
    grad_accum = np.zeros_like(policy.W)

    for states, actions, rewards in trajectories:
        # 计算折扣回报
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        # 标准化
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 累积梯度
        for t in range(len(states)):
            grad = policy.gradient_log_prob(states[t], actions[t])
            grad_accum += grad * returns[t]

    # 梯度上升
    policy.W += lr * grad_accum / len(trajectories)
    return grad_accum


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    policy = SimplePolicy(4, 2)
    state = np.random.randn(4)

    probs = policy.get_probs(state)
    print(f"动作概率: {probs}")
    action = policy.sample(state)
    print(f"采样动作: {action}")
    print(f"log π(a|s): {policy.log_prob(state, action):.4f}")

    grad = policy.gradient_log_prob(state, action)
    print(f"梯度形状: {grad.shape}")
    print(f"梯度范数: {np.linalg.norm(grad):.4f}")
```

## 9. 可视化与结果理解

```python
"""策略梯度训练可视化"""
import matplotlib.pyplot as plt
import numpy as np


def plot_policy_gradient(rewards_history=None):
    """可视化策略梯度的训练效果和高方差特征"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 子图1：训练曲线（展示高方差特征）
    if rewards_history is not None:
        axes[0].plot(rewards_history, alpha=0.3, color='blue')
        window = 20
        if len(rewards_history) >= window:
            ma = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(rewards_history)), ma,
                        color='red', linewidth=2, label=f'{window}回合滑动平均')
    else:
        np.random.seed(42)
        eps = np.arange(300)
        # 策略梯度特有的高方差训练曲线
        sim = np.minimum(500, 30 + eps * 1.2 + np.random.randn(300) * 60)
        axes[0].plot(eps, sim, alpha=0.3, color='blue')
        ma = np.convolve(sim, np.ones(20)/20, mode='valid')
        axes[0].plot(range(19, 300), ma, 'r-', linewidth=2, label='滑动平均')
    axes[0].set_xlabel('训练回合')
    axes[0].set_ylabel('回合奖励')
    axes[0].set_title('REINFORCE 训练曲线（高方差特征）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2：策略演化（动作概率随训练变化）
    # 模拟：动作0的概率从0.5逐渐变为接近1
    episodes = np.arange(100)
    prob_action0 = 0.5 + 0.48 * (1 - np.exp(-episodes / 30))
    prob_action1 = 1 - prob_action0
    axes[1].fill_between(episodes, 0, prob_action0, alpha=0.5, color='blue', label='动作 0')
    axes[1].fill_between(episodes, prob_action0, 1, alpha=0.5, color='red', label='动作 1')
    axes[1].set_xlabel('训练回合')
    axes[1].set_ylabel('动作概率')
    axes[1].set_title('策略演化（动作概率随训练变化）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 子图3：梯度方差对比（有/无基线）
    np.random.seed(42)
    no_baseline = np.random.randn(200) * 3.0  # 无基线高方差
    with_baseline = np.random.randn(200) * 1.5  # 有基线低方差
    axes[2].hist(no_baseline, bins=30, alpha=0.5, label='无基线', density=True)
    axes[2].hist(with_baseline, bins=30, alpha=0.5, label='有基线 V(s)', density=True)
    axes[2].set_xlabel('梯度估计值')
    axes[2].set_ylabel('密度')
    axes[2].set_title('基线对梯度方差的影响')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('policy_gradient_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_policy_gradient()
```

**结果解读**：
- 左图：策略梯度训练曲线波动大（高方差特征），需要滑动平均才能看清趋势。与 DQN 平滑的上升曲线不同，REINFORCE 的训练曲线即使在收敛后仍有较大波动，这是蒙特卡罗回报 $G_t$ 高方差的直接体现。
- 中图：随着训练进行，最优动作的概率逐渐增大，策略从随机变为确定。理想的学习过程应该是"概率平滑转移"——如果一个动作的概率突然从 0.9 跳到 0.1，说明策略崩溃了。
- 右图：引入基线 $V(s)$ 后梯度方差显著降低（橙色比蓝色更集中）。这是 Actor-Critic 方法的核心动机——通过减去基线降低梯度方差，从而稳定训练。

**高方差的具体表现**：策略梯度的训练曲线有一个非常明显的特征——即使整体趋势是上升的，每个回合之间的奖励波动非常大（可能一个回合 500 分，下一个回合 100 分）。这不是 bug，而是蒙特卡罗回报 $G_t$ 高方差的直接结果。引入基线后（Actor-Critic），这个波动会显著减小。## 10. 模型评估### ## 10. 模型评估

评估指标

| 指标 | 说明 | 为什么适合 |
|------|------|-----------|
| 平均回合奖励 | 最近 N 个回合奖励均值 | 直接反映策略质量，是最核心的评估指标 |
| 策略熵 | 动作分布的熵值 $H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)$ | 过低=过早收敛到次优解，过高=未学到东西 |
| 学习曲线斜率 | 奖励增长率（可用线性回归拟合） | 衡量学习效率，对比不同算法的收敛速度 |
| 梯度方差 | 策略梯度估计值的方差 | 直接反映策略梯度的"噪声水平"，是评估改进效果的关键指标 |

## 10. 模型评估

### 评估方法

评估策略梯度方法时，建议关注以下信号：

1. **训练曲线的整体趋势**：由于策略梯度的高方差特性，单看原始训练曲线可能看不出趋势，建议使用滑动平均（窗口 20~50）或指数移动平均来平滑曲线。
2. **策略熵的变化**：正常的学习过程中，策略熵应从高（接近 $\log|\mathcal{A}|$）逐渐降低。如果熵始终很高，说明策略没有学到东西；如果熵突然降到 0，可能发生了策略崩溃。
3. **最终性能的稳定性**：与 DQN 不同，策略梯度的最终性能可能不稳定（即使收敛后仍有波动）。建议用最后 50~100 个回合的平均奖励作为最终性能指标。

```python
"""策略梯度评估代码"""
import torch
import numpy as np
from torch.distributions import Categorical


def evaluate_policy(agent, env, n_episodes=20):
    """评估训练好的策略"""
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = agent.policy(state_t)
            action = probs.argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(ep_reward)
    return total_rewards


def compute_policy_entropy(agent, states):
    """计算策略熵"""
    entropies = []
    for state in states:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = agent.policy(state_t)
        entropy = Categorical(probs).entropy().item()
        entropies.append(entropy)
    return np.mean(entropies)


if __name__ == "__main__":
    print("评估说明：")
    print("1. 使用 evaluate_policy() 评估策略性能")
    print("2. 使用 compute_policy_entropy() 检查策略多样性")
    print("3. 策略熵从高到低是正常的学习过程")
    print("4. 如果熵始终很高 → 策略没有学到东西")
```

### 与 DQN 的评估差异
策略梯度方法的评估与 DQN 有重要区别：(1) 策略梯度不需要"取最优动作"（因为策略本身就是概率分布），可以直接用 `probs.argmax()` 作为确定性策略评估，也可以用采样策略评估（更接近实际训练行为）；(2) 策略梯度通常需要更多回合才能达到稳定性能（高方差导致需要更多样本），建议评估至少 50 个回合取平均。## 11. 常见问题与易错点

## 11. 常见问题与易错点

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|---------|
| 回报未标准化 | 训练不稳定，梯度忽大忽小 | 不同回合回报量级差异大（长回合回报远大于短回合） | $(G - \mu) / \sigma$，每回合独立标准化 |
| 轨迹太短 | 梯度信号弱，学习缓慢 | 短回合提供的信息量不足 | 增大 $\gamma$ 让回报传播更远，或用 n-step 方法 |

数据层面最常见的错误是忘记标准化回报。建议在训练开始时打印一个 batch 的回报分布，确认均值在 0 附近、标准差在 1 附近。如果回报均值为 100、标准差为 500，说明标准化缺失。

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|---------|
| 梯度方差过大 | 训练曲线剧烈波动，策略忽好忽坏 | 策略梯度固有高方差（单条轨迹的噪声太大） | 引入 baseline $V(s)$，增加批量大小 |
| 学习率过大 | 策略崩溃（奖励骤降至初始水平） | 策略参数更新步长过大，一次更新破坏了好不容易学到的策略 | 降低学习率，或用 PPO 的 clip 机制限制更新幅度 |
| softmax 数值溢出 | NaN 损失，训练完全失败 | logits 值过大导致 $e^{logits}$ 溢出 | `logits - max(logits)` 数值稳定化技巧 |
| log 概率未记录 | 无法计算策略梯度 | 只采样了动作但没有保存 $\log\pi(a_t|s_t)$ | 在采样时同时保存 log_prob |

模型层面最严重的 bug 是策略崩溃。如果训练中奖励突然从 500 降到 20，说明学习率太大。建议立即降低学习率到原来的 1/10，并用最近一个好的 checkpoint 恢复训练。

### 调参层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|---------|
| 学习率选择困难 | 对学习率极其敏感，差 10 倍结果完全不同 | 策略梯度方法共性问题 | 从 1e-3 开始，指数调整（1e-2, 1e-3, 1e-4） |
| $\gamma$ 过低 | 策略短视，不关注长期回报 | 远期奖励衰减太快 | 增大 $\gamma$ 到 0.99 |
| 隐藏层太小 | 策略表达能力不足 | 网络容量不足以编码复杂策略 | hidden_dim 至少 128，复杂任务用 256 |

调参建议：策略梯度对学习率最敏感，建议从 1e-3 开始尝试。如果训练 100 个回合后奖励没有明显上升趋势，说明学习率可能太小，增大到 5e-3；如果训练不稳定或奖励骤降，说明学习率太大，降低到 1e-4。

## 12. 学习总结

策略梯度的核心贡献是提出了直接优化策略的框架：

$$\nabla_\theta J = \mathbb{E}_{\pi_\theta}[Q^\pi(s,a) \nabla_\theta \log \pi_\theta(a|s)]$$

**统一视角**：将 $Q^\pi$ 替换为不同估计即得到不同算法，策略梯度定理是这些算法的共同理论基础：
- $G_t$（蒙特卡罗回报）→ REINFORCE，最简单但方差最高
- $V(s)$ + 优势函数 → A2C，引入 Critic 降低方差
- GAE 加权优势 → PPO 中的标准优势估计方法
- 重要性采样 + clip → PPO，支持异策略数据重用并约束更新幅度

**与前序知识的关系**：策略梯度是 Q-learning 和 TD 方法之后的范式转换。Q-learning 间接通过价值函数推导策略（先学 Q 值，再取 argmax），策略梯度直接在策略空间中搜索最优参数。两种范式各有优劣：基于价值的方法样本效率高但不支持连续动作，基于策略的方法支持连续动作但样本效率低。Actor-Critic 正是两者的融合——用价值函数（Critic）来降低策略梯度（Actor）的方差。

**与后续算法的演进路线**：REINFORCE（策略梯度的最简实现）→ Actor-Critic（引入 Critic 作基线）→ A2C/A3C（多进程并行训练）→ PPO（clip 约束 + 重要性采样）。这条路线的核心驱动力是不断降低方差、提高样本效率、增强训练稳定性，同时保持策略梯度的核心优势（连续动作、随机策略）。

**核心限制与改进方向**：策略梯度的两个核心限制是高方差和低样本效率。高方差通过引入基线（Actor-Critic）、GAE、批量归一化来缓解；低样本效率通过重要性采样（PPO）或经验回放（异策略方法）来缓解。理解了这些限制和改进方向，就能理解为什么 PPO、SAC 等现代算法要设计得那么"复杂"——每一层复杂度都是为了解决策略梯度的一个具体问题。

**策略梯度定理的统一视角**：策略梯度定理 $\nabla_\theta J = \mathbb{E}[Q^\pi(s,a) \nabla_\theta \log \pi_\theta(a|s)]$ 是一个极其通用的公式。将 $Q^\pi$ 替换为不同的估计，就得到不同的算法：用蒙特卡罗回报 $G_t$ 得到 REINFORCE；用优势函数 $A(s,a) = Q - V$ 得到 A2C；用 GAE 加权优势得到 PPO 的标准配置。这个统一视角说明，策略梯度方法的发展本质上就是"如何更好地估计 $Q^\pi$"的历史。

## 13. 练习题与思考题

### 基础题

**题1**：为什么策略梯度使用 $\log \pi_\theta$ 而非 $\pi_\theta$？

**答**：对数微分技巧将累乘变累加，简化计算。轨迹概率 $P_\theta(\tau) = \prod_t \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t,a_t)$ 是连乘形式，直接求导非常复杂。取对数后 $\log P_\theta(\tau) = \sum_t \log \pi_\theta(a_t|s_t) + \text{const}$，梯度变为 $\nabla_\theta \log P_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$，简洁且可计算。

**题2**：基于平稳分布的策略梯度公式中，$d^\pi(s)$ 是什么？为什么需要它？

**答**：$d^\pi(s)$ 是策略 $\pi$ 下的状态平稳分布，表示在长期运行中状态 $s$ 被访问的频率。它将"回合级"的轨迹概率转化为"时步级"的期望，使得可以用单步的 $(s, a)$ 而非完整轨迹来计算梯度。这是从 REINFORCE（需要完整轨迹）过渡到 Actor-Critic（每步更新）的理论基础。

### 进阶题

**题3**：证明引入基线 $b(s)$ 不改变策略梯度的期望。

**答**：需要证明 $\mathbb{E}_{\pi}[(Q(s,a) - b(s)) \nabla \log\pi(a|s)] = \mathbb{E}_{\pi}[Q(s,a) \nabla \log\pi(a|s)]$，即 $\mathbb{E}_{\pi}[b(s) \nabla \log\pi(a|s)] = 0$。

$$\mathbb{E}_{\pi}[b(s) \nabla \log\pi(a|s)] = \sum_s d^\pi(s) b(s) \sum_a \pi(a|s) \nabla \log\pi(a|s)$$

由于 $\sum_a \pi(a|s) \nabla \log\pi(a|s) = \sum_a \nabla \pi(a|s) = \nabla \sum_a \pi(a|s) = \nabla 1 = 0$，因此基线项为 0。但 $b(s)$ 改变了梯度的方差，选择 $b(s) = V(s)$ 可以最小化方差。

### 开放思考题

**题4**：策略梯度方法的"同策略"限制是否可以被打破？如何打破？

**思考方向**：
- **重要性采样**：用旧策略采样的数据估计新策略的梯度，$w = \frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)}$。PPO 限制 $w$ 的范围来保证稳定性。
- **离线 RL**：从任意策略收集的数据中学习，完全打破同策略限制（如 CQL、IQL）。
- **代价比对**：重要性采样在 $w$ 很大时方差极高，实际中通常限制重用次数（PPO 只重用 3-4 轮）。

**题目1**：策略梯度方法相比值函数方法（如 DQN）有什么优势和劣势？

**参考答案**：优势：(1) 天然处理连续动作空间——直接输出连续值，无需离散化；(2) 可以学习随机策略——输出动作概率分布，在博弈等需要混合策略的场景中不可替代；(3) 策略更平滑——值函数方法学到的策略可能在相似状态间跳变，策略梯度的策略参数化使其更连续。劣势：(1) 方差高——策略梯度的方差通常比值函数方法的更新方差大；(2) 样本效率低——同策略方法（如 REINFORCE）不能复用旧数据；(3) 训练不稳定——策略更新过大可能导致性能骤降。

**题目2**：策略梯度定理 $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi(a|s) Q^\pi(s,a)]$ 中的 $Q^\pi$ 可以用什么替代而不改变梯度的正确性？

**参考答案**：可以替换为任何满足 $\mathbb{E}_{a \sim \pi}[f(s,a) \nabla_\theta \log \pi(a|s)] = \nabla_\theta J$ 的函数。具体来说：(1) $G_t$（MC 回报）——REINFORCE 算法；(2) $G_t - b(s)$（带基线）——不引入偏差因为 $b$ 不依赖 $a$；(3) $r + \gamma V(s') - V(s)$（TD 误差）——Actor-Critic；(4) $\sum (\gamma\lambda)^l \delta_{t+l}$（GAE）——PPO 等现代算法。只要替换量在期望意义下等价于 $Q^\pi(s,a)$ 或 $A^\pi(s,a)$，梯度方向就是正确的。

**题目3**：为什么策略梯度方法通常采用同策略（on-policy）设定？异策略策略梯度有什么困难？

**参考答案**：策略梯度的期望 $\mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta \cdot Q]$ 是在当前策略 $\pi_\theta$ 下定义的。如果用旧策略 $\pi_{\theta_{old}}$ 收集的数据来估计梯度，需要用重要性采样修正：乘以 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$。当新旧策略差异较大时，重要性采样比的方差极大（可能为 0 或 1000+），导致梯度估计不可用。PPO 通过 clip 机制限制策略变化幅度，TRPO 通过 KL 散度约束策略更新，都是为了安全地使用（近似的）异策略数据。

## 14. 学习路径建议

**前置**：Q-learning、TD 方法。在学习策略梯度之前，建议先理解 DQN 的完整流程（经验回放、目标网络、$\varepsilon$-greedy），因为策略梯度与基于价值的方法是两种截然不同的范式。通过对比 DQN 和 REINFORCE 的差异，可以更深刻地理解"为什么需要策略梯度"——不是为了替代 DQN，而是为了解决 DQN 无法处理的连续动作空间和随机策略需求。

**平行**：REINFORCE（策略梯度的蒙特卡罗实现）。建议在读完本章后，用 PyTorch 实现一个完整的 REINFORCE 算法在 CartPole 上训练。这是理解策略梯度的最佳方式——亲自观察高方差的训练曲线，亲手调试学习率，亲身体验"策略崩溃"。REINFORCE 代码不超过 100 行，是最简策略梯度的最佳学习材料。

**进阶路线**：
- **Actor-Critic → A2C → PPO**：这是策略梯度方法的主流演进路线。每一步都是在前一步基础上做增量改进：Actor-Critic 加了 Critic 作基线，A2C 加了优势函数，PPO 加了 clip 约束和重要性采样。建议逐步学习，理解每一步解决了前一步的什么问题。
- **TRPO**：如果对理论感兴趣，可以学习 TRPO（信任域策略优化），它是 PPO 的理论基础，用 KL 散度约束策略更新幅度。

**推荐资源**：
1. **原书第9章**：系统讲解策略梯度定理的推导和 REINFORCE 的实现细节，包含完整的伪代码和练习题。
2. **Sutton & Barto《Reinforcement Learning: An Introduction》第13章**：策略梯度方法的理论基础，从策略梯度定理到自然策略梯度的完整推导。第 13.1~13.4 节是必读内容。
3. **Williams "Simple statistical gradient-following algorithms for connectionist reinforcement learning" (1992)**：REINFORCE 的原始论文，只有 8 页，但包含了策略梯度的核心思想。建议精读，理解 Williams 是如何用对数微分技巧推导策略梯度的。
4. **Karpathy "Deep Reinforcement Learning: Pong from Pixels"（博客）**：用简单的例子和生动的语言讲解策略梯度，是最好的入门材料之一。读完这篇博客再看原书，会有豁然开朗的感觉。
5. **Lil'Log (lilianweng.github.io) "Policy Gradient Algorithms"**：系统总结了策略梯度方法的发展历程，从 REINFORCE 到 PPO，配有清晰的公式推导和代码示例。
