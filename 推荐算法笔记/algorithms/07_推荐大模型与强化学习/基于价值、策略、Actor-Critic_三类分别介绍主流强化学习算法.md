# 面试题：基于价值、策略、Actor-Critic 三类分别介绍主流强化学习算法

面试题：基于价值、策略、Actor-Critic 三类分别介绍主流强化学习算法

下面按照基于价值、基于策略和 Actor-Critic 这三类主流强化学习方法进行介绍。

# 基于价值的方法

基于价值的方法的核心思想是先学习一个价值函数（通常是动作价值函数Q-function），然后通过选择能够最大化价值的动作来间接地推导出最优策略。这类方法通常适用于离散动作空间。

# 1 Q-Learning

 核心公式：其核心是时序差分更新，通过不断迭代来逼近最优动作价值函数 $Q ^ { * } ( s , a )$ ：

$$
Q \left(s _ {t}, a _ {t}\right) \leftarrow Q \left(s _ {t}, a _ {t}\right) + \alpha \left[ r _ {t + 1} + \gamma \max  _ {a ^ {\prime}} Q \left(s _ {t + 1}, a ^ {\prime}\right) - Q \left(s _ {t}, a _ {t}\right) \right]
$$

其中， $_ \alpha$ 是学习率， 是折扣因子。目标 $r _ { t + 1 } + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( s _ { t + 1 } , a ^ { \prime } )$ 包含了当前奖励和对下一状态最大 Q 值的估计。

 场景：经典 Q-Learning 是表格型方法，适用于状态和动作空间小、可枚举的场景，如简单的网格世界。其思想是深度 Q网络等算法的基础。

# 2 SARSA

 核心公式：SARSA 的更新公式与 Q-Learning 相似但关键区别在于目标值的计算：

$$
Q \left(s _ {t}, a _ {t}\right) \leftarrow Q \left(s _ {t}, a _ {t}\right) + \alpha \left[ r _ {t + 1} + \gamma Q \left(s _ {t + 1}, a _ {t + 1}\right) - Q \left(s _ {t}, a _ {t}\right) \right]
$$

它使用当前策略（通常包含探索，如ε-greedy）实际选择的下一个动作 $a _ { t + 1 }$ 来计算目标，而不是直接使用最大Q值。

 场景：由于更新依赖于当前策略实际执行的动作，SARSA 更注重策略的安全性，适合需要考虑探索风险和高交互成本的场景，如机器人导航。

# 3 深度Q 网络及其变种

当状态空间是高维时（如图像），需要用神经网络来近似 Q 函数。

 DQN: 引入经验回放（打破数据相关性）和目标网络（稳定训练）。损失函数为：

$$
L (\theta) = \mathbb {E} _ {(s, a, r, s ^ {\prime}) \sim D} \left[ \left(r + \gamma \max  _ {a ^ {\prime}} Q _ {\text {t a r g e t}} (s ^ {\prime}, a ^ {\prime}; \theta^ {-}) - Q (s, a; \theta)\right) ^ {2} \right]
$$

 Double DQN: 解决 DQN 对 Q 值过高估计的问题，通过解耦动作选择与价值评估。   
 Dueling DQN: 将 Q 网络分解为状态价值函数 V 和优势函数 A，即

$$
Q (s, a) = V (s) + A (s, a) - \frac {1}{A} \sum_ {a ^ {\prime}} A (s, a ^ {\prime}) \quad , \text {使 网 络 能 更 高 效 地 学 习 状 态 的 价 值 。}
$$

场景：DQN 系列算法特别适合处理高维状态观测（如玩 Atari 游戏），但动作空间仍需是离散的。

# 基于策略的方法

基于策略的方法不依赖价值函数，而是直接参数化并优化策略函数 $\pi _ { \boldsymbol { \theta } } ( a | s )$ 。这种方法特别适用于连续动作空间，并能自

然地学习随机策略。

# 1 REINFORCE

 核心公式：REINFORCE是一种蒙特卡洛策略梯度算法，使用完整轨迹的回报 $G _ { t }$ 来估计梯度。其策略梯度更新公式为：$\nabla _ { \theta } J ( \theta ) = \mathbb { E } _ { \pi _ { \theta } } [ \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } | s _ { t } ) G _ { t } ]$ 参数更新为： $\theta  \theta + \alpha \nabla _ { \theta } J ( \theta ) ,$ 。  
 场景：REINFORCE是策略梯度的基础算法，实现简单，能直接处理连续动作空间。但由于使用完整回合的回报，估计的方差较高，收敛性可能较慢，更适合回合制任务。

# ① Actor-Critic 方法

Actor-Critic 框架结合了基于价值和基于策略方法的优点，通过两个组件进行学习：Actor（执行者，负责根据策略选择动作）和 Critic（评论者，负责评估当前策略的价值）。

# 1 A2C / A3C（异步优势 Actor-Critic）

 核心公式：该算法使用优势函数 A(s,a) 来替代 REINFORCE 中的回报 $G _ { t }$ ，从而减少方差。

优势函数衡量的是在状态 s 下采取动作 a 相对于平均情况有多好，表示为 $A ( s , a ) = Q ( s , a ) - V ( s ) _ { , }$ 。在实际中，常用时序差分误差来近似优势函数，即 $\delta _ { t } = r _ { t + 1 } + \gamma V ( s _ { t + 1 } ) - V ( s _ { t } ) ,$ 。策略梯度更新为：

$$
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {\pi_ {\theta}} [ \nabla_ {\theta} \log \pi_ {\theta} (a _ {t} | s _ {t}) \delta_ {t} ]
$$

同时，Critic 网络会通过最小化时序差分误差来更新价值函数参数。

 场景：A3C 支持异步并行训练，效率较高。A2C 是其同步版本。这类算法在需要持续学习且探索性较强的环境中表现良好。

# 2 深度确定性策略梯度DDPG

 核心公式：DDPG 适用于连续动作空间。Actor 网络 $\mu ( s )$ 输出确定性动作，Critic 网络 $Q ( s , a )$ 评估动作价值。Critic 的更新类似 DQN，Actor 的更新则是最大化 Critic 评估的 Q 值：

$$
\nabla_ {\theta} J (\theta) \approx \mathbb {E} \left[ \left. \nabla_ {a} Q (s, a) \right| _ {a = \mu (s)} \nabla_ {\theta} \mu (s) \right]
$$

它也采用经验回放和目标网络来稳定训练。

 场景：适用于需要连续控制的任务，如机器人控制、自动驾驶中的方向盘控制等。

# 3 近端策略优化 PPO

 核心公式：PPO 通过限制策略更新的步长来确保训练稳定性。其核心目标函数（CLIP 版本）为：

$$
L ^ {\mathrm {C L I P}} (\theta) = \mathbb {E} _ {t} [ \min  (r _ {t} (\theta) A _ {t}, \operatorname {c l i p} (r _ {t} (\theta), 1 - \epsilon , 1 + \epsilon) A _ {t}) ]
$$

rt()= π(atst)其中 是新旧策略的概率比， $A _ { t }$ 是优势函数估计。clip 操作防止 过分偏离 1.0，从而约束更新幅度。

 场景：PPO 因其出色的稳定性、相对简单的实现和良好的性能，已成为目前强化学习实践中的首选算法之一，广泛应用于机器人、游戏 AI 等多种连续控制场景。

# 4 软演员-评论家 SAC

 核心公式：SAC 在标准的最大化累积奖励目标基础上，增加了一个熵正则项，以鼓励策略的探索性。其目标函数为：

$$
J (\pi) = \mathbb {E} _ {(s, a) \sim \pi} \left[ \sum_ {t} \gamma^ {t} \left(r \left(s _ {t}, a _ {t}\right) + \alpha \mathcal {H} \left(\pi \left(\cdot \mid s _ {t}\right)\right)\right) \right]
$$

其中 $\mathcal { H }$ 是策略的熵， $\alpha$ 是温度参数，用于平衡奖励和熵的重要性。

 场景：SAC 是一种离线策略算法，样本效率高，其鼓励探索的特性使其在需要大量探索的复杂连续控制任务中表现非常出色，但训练时间可能相对较长。

① 综合对比与选型指南  

<table><tr><td>算法</td><td>主要类型</td><td>核心思想</td><td>关键特征</td><td>典型适用场景</td></tr><tr><td>Q-Learning</td><td>价值</td><td>通过学习最优动作价值函数选择动作</td><td>离线策略，表格法</td><td>离散、低维状态/动作空间（如网格世界）</td></tr><tr><td>SARSA</td><td>价值</td><td>通过当前策略选择的动作更新动作价值函数</td><td>在线策略，更稳健</td><td>动态或高风险场景，强调安全性</td></tr><tr><td>DQN系列</td><td>价值</td><td>用神经网络近似Q函数，处理高维状态</td><td>经验回放，目标网络，离散动作</td><td>高维状态空间、离散动作空间（如Atari游戏）</td></tr><tr><td>REINFORCE</td><td>策略</td><td>直接优化策略，使用蒙特卡洛回报估计梯度</td><td>在线策略，高方差，实现简单</td><td>连续动作空间，回合制任务</td></tr><tr><td>A2C/A3C</td><td>Actor-Critic</td><td>使用优势函数降低策略梯度方差</td><td>在线策略，并行训练，降低方差</td><td>并行环境，需要高效探索的持续任务</td></tr><tr><td>DDPG</td><td>Actor-Critic</td><td>将DQN思想扩展至连续动作空间</td><td>离线策略，确定性策略，经验回放</td><td>高维状态和连续动作空间（如机器人连续控制）</td></tr><tr><td>PPO</td><td>Actor-Critic</td><td>在优化策略时限制更新幅度以保持稳定</td><td>在线策略，剪辑目标函数，稳定易用</td><td>大规模连续控制（机器人、游戏AI），实践首选</td></tr><tr><td>SAC</td><td>Actor-Critic</td><td>在最大化累积奖励的同时最大化策略熵</td><td>离线策略，随机策略，鼓励探索，样本效率高</td><td>复杂连续控制，需大量探索的任务</td></tr></table>

# 算法选型考量要点：

 动作空间类型：这是首要考量点。离散动作可选 Q-Learning、DQN 系列等；连续动作则优先考虑 DDPG、PPO、SAC、REINFORCE 等。  
 样本效率与稳定性：离线策略算法（如 DDPG, SAC, DQN）能重复利用历史数据，通常样本效率更高。PPO 等通过约束更新策略在稳定性和易用性上表现突出。  
 探索性需求：在需要智能体充分探索未知环境时，SAC的熵正则化或具有随机策略的算法更具优势。  
 问题复杂度与计算资源：对于简单、低维问题，表格法（如Q-Learning）或基础策略梯度可能足够。面对复杂、高维问题，深度强化学习算法（DQN, PPO, SAC）是更可行的选择，但同时需要更多的计算资源。

# 1. 策略梯度直观理解

策略梯度方法的核心思想非常直观：如果一个动作能够获得更高的回报，那么就增加这个动作被选择的概率；反之，如果一个动作带来的回报较低，就减少其概率。

这与基于价值的算法（如 Q-learning）不同。基于价值的算法先学习价值函数，再根据价值函数选择动作；而策略梯度方法直接基于参数化策略（例如用一个神经网络表示策略 $\pi _ { \boldsymbol { \theta } } ( a | s )$ ），并通过梯度上升来优化策略参数 $\theta$ ，以最大化期望回报。

# 2. 目标函数

强化学习的目标是最大化智能体在与环境交互中获得的期望累积回报。目标函数通常定义为：

$$
J (\theta) = \mathbb {E} _ {\tau \sim \pi_ {\theta}} [ R (\tau) ],
$$

其中 表示一条轨迹（trajectory）， $\tau = ( s _ { 0 } , a _ { 0 } , s _ { 1 } , a _ { 1 } , \dots , s _ { T } )$ $R ( \tau ) = { \sum _ { t = 0 } ^ { T } r ( s _ { t } , a _ { t } ) }$ 是轨迹 $\tau$ 的总回报。我们的目标是找到最优参数 $\theta ^ { * }$ ，使得 $J ( \theta )$ 最大： $\theta ^ { * } = \arg \operatorname* { m a x } _ { \theta } J ( \theta )$

# 3. 策略梯度定理推导

策略梯度定理告诉我们，目标函数 $J ( \theta )$ 关于参数 $\theta$ 的梯度可以表示为：

$$
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {\tau \sim \pi_ {\theta}} \left[ \sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} \left(a _ {t} \mid s _ {t}\right) \cdot R (\tau) \right]
$$

# 推导过程主要步骤如下：

1. 梯度表达：首先，写出梯度的表达式：

$$
\nabla_ {\theta} J (\theta) = \nabla_ {\theta} \mathbb {E} _ {\tau \sim \pi_ {\theta}} [ R (\tau) ] = \nabla_ {\theta} \int p _ {\theta} (\tau) R (\tau) d \tau , \text {其 中} p _ {\theta} (\tau) \text {为 轨 迹} \tau \text {的 概 率}
$$

2. 似然比技巧：将梯度运算符移入积分，并应用似然比技巧 （Likelihood Ratio Trick），即使用恒等式$\nabla _ { \boldsymbol { \theta } } p _ { \boldsymbol { \theta } } ( \tau ) = p _ { \boldsymbol { \theta } } ( \tau ) \nabla _ { \boldsymbol { \theta } } \log { p _ { \boldsymbol { \theta } } ( \tau ) }$ ，那么：

$$
\begin{array}{l} \nabla_ {\theta} J (\theta) = \int \nabla_ {\theta} p _ {\theta} (\tau) R (\tau) d \tau \\ = \int p _ {\theta} (\tau) \nabla_ {\theta} \log p _ {\theta} (\tau) R (\tau) d \tau \\ = \mathbb {E} _ {\tau \sim \pi_ {\theta}} [ \nabla_ {\theta} \log p _ {\theta} (\tau) \cdot R (\tau) ] \\ \end{array}
$$

3. 分解轨迹概率：一条轨迹 $\tau$ 的概率 $p _ { \theta } ( \tau )$ 可以分解为：

$$
p _ {\theta} (\tau) = p \left(s _ {0}\right) \prod_ {t = 0} ^ {T} \pi_ {\theta} \left(a _ {t} \mid s _ {t}\right) p \left(s _ {t + 1} \mid s _ {t}, a _ {t}\right)
$$

其中 $p ( s _ { 0 } )$ 是初始状态分布， $p ( s _ { t + 1 } | s _ { t } , a _ { t } )$ 是环境的状态转移概率。

4. 取对数化简：对 $p _ { \theta } ( \tau )$ 取对数：

$$
\log p _ {\theta} (\tau) = \log p (s _ {0}) + \sum_ {t = 0} ^ {T} \left(\log \pi_ {\theta} \left(a _ {t} \mid s _ {t}\right) + \log p \left(s _ {t + 1} \mid s _ {t}, a _ {t}\right)\right)
$$

再对 $\theta$ 求梯度。注意 $\log p ( s _ { 0 } )$ 和 $\log p \big ( s _ { t + 1 } \big | s _ { t } , a _ { t } \big )$ 与策略参数 $\theta$ 无关，因此它们的梯度为零。于是：

$$
\nabla_ {\theta} \log p _ {\theta} (\tau) = \sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} (a _ {t} | s _ {t})
$$

5. 得到最终形式：将上式代回第2步的梯度表达式：

$$
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {\tau \sim \pi_ {\theta}} \left[ \left(\sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} \left(a _ {t} \mid s _ {t}\right)\right) \cdot R (\tau) \right]
$$

在实际应用中，我们通常通过采样来近似这个期望。假设我们采样了 N 条轨迹，那么梯度可以近似为：

$$
\nabla_ {\theta} J (\theta) \approx \frac {1}{N} \sum_ {i = 1} ^ {N} \left[ \left(\sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} \left(a _ {t} ^ {(i)} \mid s _ {t} ^ {(i)}\right)\right) \cdot R \left(\tau^ {(i)}\right) \right]
$$

# ① 4. 减少方差：引入基线（Baseline）与奖励变换

原始的策略梯度的方差（Variance）较高，会导致训练不稳定。一些常见的改进如下：

引入基线：策略梯度定理可以推广为：

$$
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {\tau \sim \pi_ {\theta}} \left[ \sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} (a _ {t} | s _ {t}) \cdot (R (\tau) - b) \right]
$$

其中 $b$ 是一个基线，通常选择为平均回报 $b = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } R ( \tau ^ { ( i ) } )$ 。理论证明，减去一个基线不会改变梯度的期望值（无偏），但能有效降低方差。

 Advantage Function ： 一 个 更 精 细 的 方 法 是 使 用 优 势 函 数 （ Advantage Function ）$A ^ { \pi } ( s _ { t } , a _ { t } ) = Q ^ { \pi } ( s _ { t } , a _ { t } ) - V ^ { \pi } ( s _ { t } )$

优势函数衡量了在状态 $s _ { t }$ 下采取动作 $a _ { t }$ 比平均情况好多少。此时的梯度变为：

$$
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {\tau \sim \pi_ {\theta}} \left[ \sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} \left(a _ {t} \mid s _ {t}\right) \cdot A ^ {\pi} \left(s _ {t}, a _ {t}\right) \right]
$$

使用优势函数可以显著降低方差，是 Actor-Critic 算法的基础。

 奖励变换：在原始公式中，轨迹上每个时刻的动作都用整个轨迹的总回报 $R ( \tau )$ 来加权，这并不合理，因为 时刻之后的动作不会影响 $t$ 时刻之前的回报。

因此，我们通常用从当前时刻到结束的累积奖励（Reward-to-go） $\begin{array} { r } { \hat { R } _ { t } = \sum _ { t ^ { \prime } = t } ^ { T } r \big ( s _ { t ^ { \prime } } , a _ { t ^ { \prime } } \big ) _ { \neq / \neq / \neq \pm } R ( \tau ) _ { \circ } } \end{array}$

结合以上两点，一个更优的梯度估计式为：

$$
\nabla_ {\theta} J (\theta) \approx \frac {1}{N} \sum_ {i = 1} ^ {N} \sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} \left(a _ {t} ^ {(i)} \mid s _ {t} ^ {(i)}\right) \cdot \left(\hat {R} _ {t} ^ {(i)} - b (s _ {t})\right)
$$

其中基线 $b ( s _ { t } )$ 也可以是状态相关的，例如常用状态价值函数 $V ^ { \pi } ( s _ { t } )$ 作为基线。

# 5. 与最大似然估计的比较

通过比较可以更好地理解策略梯度：

$$
\nabla_ {\theta} J _ {M L} (\theta) \approx \frac {1}{N} \sum_ {i = 1} ^ {N} \sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} \left(a _ {t} ^ {(i)} \mid s _ {t} ^ {(i)}\right)
$$

最大似然估计（MLE）的梯度为

，目标是最大化观察到动作的似然。

$$
\nabla_ {\theta} J (\theta) \approx \frac {1}{N} \sum_ {i = 1} ^ {N} \sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} \left(a _ {t} ^ {(i)} \mid s _ {t} ^ {(i)}\right) \cdot R \left(\tau^ {(i)}\right)
$$

策略梯度的表达式为：

可以看出，策略梯度相当于用累积回报 $R ( \tau )$ 给最大似然估计的梯度加了个权重。回报高的轨迹权重更大，模型会更大程度地增加这些轨迹中动作的概率；回报为负的轨迹则其动作概率会被抑制。

# 6. 策略梯度中的技巧与改进

<table><tr><td>改进方法</td><td>目的</td><td>说明</td></tr><tr><td>基线 (Baseline)</td><td>降低梯度估计的方差</td><td>常用平均回报或价值函数 V(s)作为基线。减去基线后变为 ∇θ log πθ(at st) · (R(τ) - b)，不影响无偏性。</td></tr><tr><td>Advantage 函数</td><td>更有效地衡量动作的相对好坏</td><td>A(st, at) = Q(st, at) - V(st)。梯度形式变为 ∇θ log πθ(at st) · A(st, at)，方差更低。</td></tr><tr><td>折扣因子</td><td>强调近期奖励，降低远期不确定性</td><td>在计算累积奖励时引入折扣因子γ，\(\hat{R}_{t}=\sum_{t&#x27;=t}^{T}\gamma^{t&#x27;-t}r\left(st&#x27;,at&#x27;\right)\)。</td></tr></table>

小结：Policy Gradient策略梯度定理是许多现代强化学习算法（如 Actor-Critic、PPO、TRPO）的基石。掌握其推导和理解其背后的直观含义（增加高回报动作的概率，减少低回报动作的概率），以及如何通过基线 Baseline、优势函数等方法降低方差，对于应对技术面试和深入理解强化学习都至关重要。

