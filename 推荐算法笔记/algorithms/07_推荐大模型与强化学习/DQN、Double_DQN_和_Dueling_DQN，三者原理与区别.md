# 面试题：DQN、Double DQN 和 Dueling DQN，三者原理与区别

# 面试题：DQN、Double DQN 和 Dueling DQN，三者原理与区别

# 1 DQN

深度 Q 网络（Deep Q-Network, DQN）是深度强化学习的基础算法，其核心思想是用神经网络近似 Q-learning 中的动作价值函数（Q 函数），从而处理高维状态空间（如图像输入）的问题。

传统 Q-learning 在状态空间过大或连续时，无法通过表格方式存储 Q 值，DQN 通过参数化的函数 $Q _ { \theta }$ 来拟合最优 Q 值函数。

# 1.1 基本原理

 在 Q-learning 中，需要优化的目标函数为：

$$
\min  _ {\theta} J (\theta) = \mathbb {E} \left[ \left(R + \gamma \max  _ {a} Q \left(S ^ {\prime}, a; \theta\right) - Q (S, A; \theta)\right) \right]
$$

其中 R 表示即时奖励， $\gamma$ 为折扣因子，S 和 A 分别表示当前状态和动作， $S ^ { \prime }$ 表示下一状态。

 DQN 的 TD 目标（Temporal Difference Target）为：

$$
Y _ {t} ^ {D Q N} = R _ {t + 1} + \gamma \max  _ {a} Q \left(S _ {t + 1}, a; \theta^ {-}\right)
$$

其中 $\theta$ 是训练网络的参数， $\theta ^ { - }$ 是目标网络的参数。

# 1.2 主要创新

DQN引入了两个关键技术创新：

 经验回放（Experience Replay）：智能体与环境交互的经验 $( s , a , r , s ^ { \prime } , \mathrm { d o n e } )$ 被存储到经验池中，训练时从池中随机采样。这解决了数据间相关性带来的训练不稳定性问题，同时提高了样本利用率。  
 目标网络（Target Network）：DQN 使用两套网络——训练网络（参数 $\theta$ ）和目标网络（参数 $\theta ^ { - }$ ）。TD 目标的计算基于目标网络，定期将训练网络参数复制给目标网络（通常每 $\tau$ 步一次），极大提升了训练稳定性。

# 2 Double DQN

Double DQN（DDQN）是针对 DQN 存在的 Q 值过高估计（overestimation）问题提出的改进算法。传统 DQN 的 max 操 作会使 Q 值的估计越来越高于真实值，导致策略次优和训练不稳定。

# 2.1 过高估计问题及其解决

在传统 DQN 中，TD 目标为：

$$
Y _ {t} ^ {D Q N} = R _ {t + 1} + \gamma Q \left(S _ {t + 1}, \arg \max  _ {a} Q \left(S _ {t + 1}, a; \theta^ {-}\right); \theta^ {-}\right)
$$

这相当于使用同一套目标网络 θ−同时选择动作（argmax 操作）和评估价值（Q 值计算），导致估计偏差累积。

Double DQN 通过解耦动作选择与价值评估来解决这个问题：

$$
Y _ {t} ^ {D D Q N} = R _ {t + 1} + \gamma Q \left(S _ {t + 1}, \arg \max  _ {a} Q \left(S _ {t + 1}, a; \theta\right); \theta^ {-}\right)
$$

即利用训练网络 θ 选择动作（argmax），然后用目标网络 θ−评估该动作的价值。

# 2.2 数学推导

Double DQN 的优化目标函数变为：

$$
\min  _ {\theta} J (\theta) = \mathbb {E} \left[ \left(R + \gamma Q \left(S ^ {\prime}, \arg \max  _ {a ^ {\prime}} Q \left(S ^ {\prime}, a ^ {\prime}; \theta\right); \theta^ {-}\right) - Q (S, A; \theta)\right) \right]
$$

这样即使训练网络 θ 对某个动作存在过高估计，目标网络 $\theta ^ { - }$ 的评估也能抵消部分偏差，使 Q 值估计更接近真实值，提高算法稳定性和收敛性。

# 3 Dueling DQN

Dueling DQN 采用了网络结构创新，通过分解 Q 值函数为状态价值和动作优势两个部分，来更有效地评估状态和动作的价值。

![](images/7eb7bf7252e2200651959d4b21f6ec58d9f086b8ad4dcc7482243e00ecbe6af3.jpg)  
Figure 1. A popular single stream $Q$ -network (top) and the dueling $Q$ -network (bottom). The dueling network has two streams to separately estimate (scalar) state-value and the advantages for each action; the green output module implements equation (9) to combine them. Both networks output $Q$ -values for each action.

# 3.1 价值函数与优势函数

Dueling DQN 的核心思想来源于优势函数（Advantage Function）的概念：

 状态价值函数 V(s)：衡量处于状态 s 的好坏程度  
 动作价值函数 Q(s,a)：衡量在状态 s 下选择动作 a 的长期回报  
优势函数 A(s,a)：定义为 A(s,a)=Q(s,a)−V(s)，表示动作 a 相对于平均水平的优势程度对优势函数取期望 $\mathbb { E } _ { a \sim \pi } [ A ( s , a ) ] = 0$ ，即优势函数在所有动作上的平均值为零。

# 3.2 网络架构与公式

Dueling DQN 将传统 DQN 的单一 Q 网络输出层分为两个分支：

 价值流（Value Stream）：输出标量 $V ( s ; \theta , \beta )$ ，表示状态价值

 优势流（Advantage Stream）：输出向量 $A ( s , a ; \theta , \alpha )$ ，表示每个动作的优势值

最终 Q 值的计算方式为：

$$
Q (s, a; \theta , \alpha , \beta) = V (s; \theta , \beta) + \left(A (s, a; \theta , \alpha) - \max  _ {a ^ {\prime} \in A} A (s, a ^ {\prime}; \theta , \alpha)\right)
$$

实践中也常使用均值形式：

$$
Q (s, a; \theta , \alpha , \beta) = V (s; \theta , \beta) + \left(A (s, a; \theta , \alpha) - \frac {1}{\mathcal {A}} \sum_ {a ^ {\prime}} A (s, a ^ {\prime}; \theta , \alpha)\right)
$$

这种结构强制优势函数零中心化，解决了辨识性问题（V和A的相对尺度不确定），同时使网络能更高效地学习状态价值表示。

# 4 三者对比与适用场景

<table><tr><td>特性</td><td>DQN</td><td>Double DQN</td><td>Dueling DQN</td></tr><tr><td>核心创新</td><td>基础算法：神经网络近似Q函数+经验回放+目标网络</td><td>解耦动作选择与价值评估</td><td>网络结构分离：
状态价值V+动作优势A</td></tr><tr><td>TD目标公式</td><td>Yt=r+γmaxaQ(s&#x27;,a;θ-)</td><td>Yt=r+γQ(s&#x27;,arg maxaQ(s&#x27;,a;θ);θ-)</td><td>与DQN或Double DQN相同，但Q网络结构不同</td></tr><tr><td>解决的问题</td><td>处理高维状态空间，稳定训练</td><td>减轻Q值过高估计</td><td>更好评估状态价值，尤其动作影响较小时</td></tr><tr><td>训练稳定性</td><td>相对较低，存在过高估计</td><td>较高，减轻了过高估计</td><td>较高，学习更鲁棒的状态表征</td></tr><tr><td>计算复杂度</td><td>较低</td><td>略高于DQN（需两次前向传播）</td><td>与DQN相当（分支结构增加参数不多）</td></tr><tr><td>适用动作空间</td><td>离散动作空间</td><td>离散动作空间</td><td>离散动作空间（尤其是动作数量较多时）</td></tr></table>

#  DQN 适用场景：

适用于中等复杂度环境、离散动作空间、作为基础学习算法。例如简单的 Atari游戏（如 Pong）、低维状态空间的决策问题。作为基础算法，适合初学者理解和实现深度强化学习的基本原理。

#  Double DQN 适用场景：

适用于需要减少 Q 值过高估计的环境，特别是那些奖励稀疏或需要长时间规划的任务。在许多 Atari 游戏（如 SpaceInvaders）中，Double DQN 相比 DQN 能取得更好的性能和稳定性。也适用于医疗诊断、金融交易等对估计准确性要求较高的领域。

#  Dueling DQN 适用场景：

适用于状态价值至关重要而单个动作影响相对较小的环境。例如自动驾驶中，环境状态（道路、交通情况）比具体动作（微小转向调整）更重要；或者资源分配问题中，状态（资源总量）比具体分配动作更关键。在动作空间较大的环境中，Dueling 结构能显著提高学习效率。

# 回答总结：

 PPO 是 on-policy 算法：其数据采集与优化策略严格一致，且无长期经验存储机制。  
 通过重要性采样提升效率：在单批次数据上多次更新（K-step），模拟 off-policy 的样本复用，但本质仍是 on-policy 框架。  
 工业应用定位：PPO 在 RLHF 等场景中作为 on-policy 优化器，依赖实时数据生成（如 GPT 对齐任务）。

PPO（Proximal Policy Optimization）算法本质上是 on-policy（同策略）方法，但通过重要性采样（Importance Sampling）技术实现了部分数据复用，使其在训练效率上接近 off-policy 方法。

# 1. 核心性质：On-Policy

 数据来源：PPO 使用当前策略 （当前参数化的策略网络）与环境交互收集数据，每次策略更新后需重新采样新数据。旧数据无法跨轮次复用，符合 on-policy 的定义。  
 策略一致性：训练优化的策略（Actor）与数据采集的策略是同一个，即“自己生成数据、自己学习”。

# 2. 重要性采样的作用：模拟 Off-Policy 效率

PPO 通过重要性采样在单次迭代内复用当前批次的数据，实现类似 off-policy 的样本效率：

#  技术原理：

用旧策略 $\pi _ { \theta _ { \mathrm { o l d } } }$ 采集的数据，计算新策略 $\pi _ { \theta }$ 的更新梯度：

$$
\nabla J (\theta) \approx \mathbb {E} _ {s, a \sim \pi_ {\mathrm {o l d}}} \left[ \frac {\pi_ {\theta} (a | s)}{\pi_ {\mathrm {o l d}} (a | s)} A ^ {\pi_ {\mathrm {o l d}}} (s, a) \right], \quad \text {其 中} \quad \frac {\pi_ {\theta}}{\pi_ {\mathrm {o l d}}} \quad \text {为 重 要 性 权 重}, \text {修 正 策 略 差 异}.
$$

 数据复用限制：重要性采样仅在单次迭代的 K 次小批量更新中复用数据（如 ${ \sf K } = 3 \sim 1 0$ 次），之后必须丢弃旧数据并重新采样， 无法长期存储经验。

# 3. 与典型 Off-Policy 方法的对比

<table><tr><td>特性</td><td>PPO</td><td>Off-Policy（如DDPG、SAC）</td></tr><tr><td>数据来源</td><td>当前策略采样，每次更新后丢弃</td><td>历史策略数据存储在经验回放池</td></tr><tr><td>数据复用</td><td>仅单批次内K次更新</td><td>长期复用任意历史数据</td></tr><tr><td>策略一致性</td><td>训练策略=采样策略</td><td>训练策略≠采样策略（如旧策略）</td></tr><tr><td>典型组件</td><td>无经验回放池</td><td>必需经验回放池</td></tr><tr><td>样本效率</td><td>中（依赖重复采样）</td><td>高（数据可复用）</td></tr></table>

⋅ 关键区别 ：PPO 的“伪 off-policy”特性仅限于单批次内的短期数据复用，而真正 off-policy 方法（如 DDPG）通过经验回放池长期跨轮次复用数据。

# 4. 设计动机：平衡稳定性与效率

 On-Policy的稳定性：直接使用当前策略数据，避免因策略差异导致的价值估计偏差（如 DDPG 需目标网络稳定训练）。  
 Clip 机制进一步约束 ：限制重要性权重 $r _ { t } ( \theta )$ 在 $\left[ 1 - \epsilon , 1 + \epsilon \right]$ 之间，防止新旧策略差异过大导致梯度失效，增强 on-policy 训练的稳定性。

# 1. 论文信息

 论文标题：Decision Transformer: Reinforcement Learning via Sequence Modeling   
 论文链接：https://arxiv.org/abs/2106.01345  
 官方代码：https://github.com/kzl/decision-transformer  
 作者机构：UC Berkeley, Facebook AI Research (FAIR), Google Brain

# 2. 提出背景

Decision Transformer（DT）的提出源于传统强化学习（RL）方法的几个固有挑战：

 长期信用分配困难：传统 RL 算法（如 DQN、PPO）在长时序任务中，由于依赖贝尔曼方程（Bellman equation）的迭代更新，对稀疏奖励或延迟奖励的处理效率较低，信用分配（Credit Assignment）效果不佳。  
 离线 RL 的稳定性问题：离线强化学习（Offline RL）中，智能体仅从固定数据集中学习，传统方法如 Q-learning 容易因价值函数高估（value overestimation）或分布外（OOD）动作导致训练不稳定。  
计算效率与框架复杂性：传统 RL 需设计复杂的价值函数或策略梯度优化框架，而 Transformer 在自然语言处理（NLP）领域已证明能有效建模长序列数据。DT 试图将 RL 问题重新定义为序列建模任务，利用 Transformer 的并行化能力简化流程。

DT 的核心目标是：通过序列建模替代动态规划，避免传统 RL 的"致命三要素"（函数逼近、自举、离线学习），同时实现更稳定的离线策略学习。

# 3. 主要创新点

 范式转变：将 RL 问题转化为条件序列生成任务，使用 Transformer 架构直接预测动作，而非依赖价值函数优化或策略梯度。  
 Return-to-Go 条件化：引入"剩余回报"（Return-to-Go）作为条件信号，使策略能根据目标回报调整行为（例如，高目标回报触发激进动作，低目标回报触发保守动作）。  
 完全监督学习框架：采用离线数据集进行监督训练，通过最大似然估计预测动作，避免传统 RL的在线交互探索。  
 长程依赖建模：利用 Transformer 的自注意力机制直接捕捉状态-动作-回报间的长期依赖，替代贝尔曼方程的逐步更新。

# 4. 数学原理与模型架构

# 4.1 轨迹表示

将轨迹表示为三元组序列，每个时间步包含：

$$
\hat {R} _ {t} = \sum_ {t ^ {\prime} = t} ^ {T} r _ {t ^ {\prime}}
$$

 剩余回报（Return-to-Go）： ，表示从时刻 $t$ 到轨迹结束的累积奖励。

状态（State）： $s _ { t } \in { \mathcal { S } } .$ 。  
动作（Action）： $a _ { t } \in \mathcal A _ { c }$ 。

轨迹形式为： $\tau = ( \hat { R } _ { 1 } , s _ { 1 } , a _ { 1 } , \hat { R } _ { 2 } , s _ { 2 } , a _ { 2 } , \dots , \hat { R } _ { T } , s _ { T } , a _ { T } )$

# 4.2 模型架构

![](images/208d791464be29c2e27d3c901975fa80962b930e782e861d7e15ef5f694e5e27.jpg)

 输入编码：对每个模态（剩余回报、状态、动作）使用独立的线性嵌入层，将原始输入投影到向量空间。添加时间步编码（非标准位置编码）以保留序列顺序。  
 Transformer backbone：采用 GPT 风格的因果 Transformer 解码器，确保自回归生成时仅关注历史信息。  
 注意力机制：通过查询（Query）、键（Key）、值（Value）计算注意力权重，公式为：

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right) V
$$

# 4.3 训练目标

通过最小化预测动作与真实动作的差异进行训练：

$$
\mathcal {L} = - \sum_ {t} \log P \left(a _ {t} \mid \hat {R} _ {\leq t}, s _ {\leq t}, a _ {<   t}\right)
$$

离散动作：交叉熵损失

$$
\mathcal {L} = \frac {1}{T} \sum_ {t} | | a _ {t} - \hat {a} _ {t} | | ^ {2}
$$

连续动作：均方误差（MSE）损失

# 5. 算法步骤

# 训练阶段

1. 数据准备：从离线数据集中采样轨迹片段，计算每个时间步的 $\hat { R } _ { t _ { \circ } }$   
2. 输入构建：将最近的 K 个三元组 $( \hat { R } _ { i } , s _ { i } , a _ { i } )$ 作为输入，生成 3K 个令牌。  
3. 模型优化：使用梯度下降最小化动作预测损失，仅对动作输出计算损失。

# 推理阶段

1. 初始化：设定目标回报 $\hat { R } _ { \mathrm { t a r g e t } }$ （如专家级回报），获取初始状态 $s _ { 1 }$ 。  
2. 自回归生成：

a. 输入当前序列 $[ \hat { R } _ { t } , s _ { t } , a _ { < t } ]$ 到 Transformer。  
b. 模型输出动作 $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi $ ，并与环境交互得到奖励 $r _ { t }$ 和下一状态 $s _ { t + 1 }$ 。  
c. 更新剩余回报： $\hat { R } _ { t + 1 } = \hat { R } _ { t } - r _ { t }$ 。  
d. 重复直至轨迹终止。

# 6. 与传统方法的比较

<table><tr><td>特性</td><td>Decision Transformer</td><td>传统 RL（如 CQL、PPO）</td></tr><tr><td>问题建模</td><td>序列生成（监督学习）</td><td>动态规划/策略优化</td></tr><tr><td>回报处理</td><td>目标回报作为条件输入</td><td>通过价值函数隐式建模</td></tr><tr><td>长期依赖</td><td>自注意力机制直接捕捉</td><td>依赖折扣因子或循环网络</td></tr><tr><td>离线学习</td><td>直接利用轨迹数据，无需交互</td><td>需重要性采样或约束优化</td></tr><tr><td>探索机制</td><td>依赖数据分布，无显式探索</td><td>ε-greedy、随机策略</td></tr></table>

# 7. 总结

Decision Transformer 通过将强化学习重构为条件序列建模问题，提供了一种简化且高效的替代方案。

其核心优势在于：

 规避了传统 RL 的稳定性问题（如"致命三要素"）。  
 在稀疏奖励和长程依赖任务中表现显著优于传统方法。  
 为融合大规模预训练模型（如 GPT）与决策任务奠定了基础。

不过也存在一定的局限性：

 计算开销：序列长度增加时，注意力机制计算复杂度呈平方增长。  
 外推能力有限：若最优动作未在数据集中出现，DT 难以生成超越数据质量的策略。  
随机性建模弱：Transformer 输出多为确定性动作，难以建模随机策略。
