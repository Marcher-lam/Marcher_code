# 面试题：介绍 RLHF 算法 PPO、DPO、GRPO，写下损失函数

面试题：介绍 RLHF 算法 PPO、DPO、GRPO，写下损失函数

在大模型的 RLHF（基于人类反馈的强化学习）训练中，主流的强化学习算法包括 PPO（Proximal Policy

Optimization）、DPO（Direct Preference Optimization）和 GRPO（Group Relative Policy Optimization）。以下是详细说明：

# 1. PPO（近端策略优化）

# PPO (Proximal Policy Optimization) 2017 OpenAI

![](images/a600a4ba34c673d3a409517d64bc5e89d9eef2c23c1c3948956511e93cd1e381.jpg)

![](images/fcfb6f6dc9e91e28630dbcae876c4c4adefe6f8a015c90ac1a7359bf8ee61a65.jpg)

核心思想：通过约束策略更新的步长（避免突变），稳定训练过程。PPO 是 RLHF 中最广泛应用的算法（如InstructGPT/ChatGPT）。

# 损失函数表达式 ：

$$
\mathcal {L} _ {\mathrm {P P O}} (\theta) = \mathbb {E} _ {(x, y) \sim D _ {\pi_ {\theta}}} \left[ \min  \left(r _ {t} (\theta) \hat {A} _ {t}, \operatorname {c l i p} (r _ {t} (\theta), 1 - \epsilon , 1 + \epsilon) \hat {A} _ {t}\right) - \beta \cdot D _ {\mathrm {K L}} \left(\pi_ {\theta} (y | x) \| \pi_ {\text {b a s e}} (y | x)\right) \right]
$$

# 参数说明 ：

$r _ { t } ( \theta ) = { \frac { \pi _ { \theta } { \bigl ( } y | x { \bigr ) } } { \pi _ { \mathrm { o l d } } { \bigl ( } y | x { \bigr ) } } }$ ：新旧策略的概率比，用于衡量策略变化。

 ：优势函数（估计当前动作优于平均水平的程度）。 $\hat { A } _ { t }$   
 ：将概率比限制在 [1−ϵ, 1+ϵ] 内（ϵ ≈ 0.1），防止策略突变。 $[ 1 - \epsilon , 1 + \epsilon ]$ $\epsilon \approx 0 . 1$   
 $\beta \cdot D _ { \mathrm { K L } }$ ：KL 散度惩罚项，约束微调模型（ $\pi _ { \theta }$ ）与初始监督微调模型（ $\pi _ { \mathrm { b a s e } }$ ）的分布差异。

# 训练流程 ：

1. 采样生成回复 $y \sim \pi _ { \theta } ( \cdot | x )$ ；  
2. 奖励模型 RM 计算奖励 $r _ { \theta } ( x , y )$ ；  
3. 结合 KL 惩罚更新策略，确保生成文本的连贯性。

# 2. DPO（直接偏好优化）

# DPO (Direct Preference Optimization) 2023 Stanford

![](images/42303d90361b7103af75730c2de30204ada20952c9c85728652d2db501e75ff9.jpg)

=算法架构图

![](images/856fa2327959b9cf267cad35c2a9cd9e1bee498eeb04414d24588f23c0ff4d11.jpg)

![](images/a01e3ff8f150fafb09f1a73b7d5e422bd701670e664f94c4a7cda942393ba29c.jpg)

核心公式

$$
1. B r a d l e y - T e r r y \text {偏 好 模 型}
$$

$$
\mathrm {P} \left(\mathrm {y} _ {\mathrm {w}} > \mathrm {y} _ {1} \mid \mathrm {x}\right) = \sigma \left(\mathrm {r} \left(\mathrm {x}, \mathrm {y} _ {\mathrm {w}}\right) - \mathrm {r} \left(\mathrm {x}, \mathrm {y} _ {1}\right)\right)
$$

2.隐式奖励函数

$$
\mathrm {r} (\mathrm {x}, \mathrm {y}) = \beta \log \left(\pi_ {\theta} (\mathrm {y} | \mathrm {x}) / \pi_ {\text {r e f}} (\mathrm {y} | \mathrm {x})\right) + \beta \log Z (\mathrm {x})
$$

$$
\mathrm {L} _ {\mathrm {D P O}} = - \mathbb {E} _ {\left(\mathrm {x}, \mathrm {y} _ {\mathrm {w}}, \mathrm {y} _ {1}\right)} [ \log \sigma (\beta \cdot (\log (\pi_ {\theta} (\mathrm {y} _ {\mathrm {w}} | \mathrm {x}) / \pi_ {\text {r e f}} (\mathrm {y} _ {\mathrm {w}} | \mathrm {x})) - \log (\pi_ {\theta} (\mathrm {y} _ {1} | \mathrm {x}) / \pi_ {\text {r e f}} (\mathrm {y} _ {1} | \mathrm {x}))) ]
$$

$$
\mathrm {L} _ {\mathrm {D P O}} = - \mathbb {E} [ \log \sigma (\beta \cdot \Delta \mathrm {r}) ]
$$

$$
\text {其 中} \Delta \mathrm {r} = \mathrm {r} _ {\theta} (\mathrm {x}, \mathrm {y} _ {\mathrm {w}}) - \mathrm {r} _ {\theta} (\mathrm {x}, \mathrm {y} _ {\mathrm {f}})
$$

![](images/8933548fee40aa9d5ffc42b6757a21c9c765bffedad8c0be2222b68673ce24c7.jpg)

√无需训练独立奖励模型  
√无需强化学习采样  
√直接从偏好数据学习  
√β控制偏离参考策略程度  
√本质是监督学习问题

核心思想：省去奖励模型（RM）训练环节，直接用人类偏好数据优化策略，降低训练复杂度。

# 损失函数表达式 ：

$$
\mathcal {L} _ {\mathrm {D P O}} (\theta) = - \mathbb {E} _ {(x, y _ {w}, y _ {l}) \sim D} \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} (y _ {w} | x)}{\pi_ {\mathrm {r e f}} (y _ {w} | x)} - \beta \log \frac {\pi_ {\theta} (y _ {l} | x)}{\pi_ {\mathrm {r e f}} (y _ {l} | x)}\right) \right]
$$

# 参数说明：

 $y _ { w } , y _ { l }$ ：人类标注的优/劣回复样本。  
 $\pi _ { \mathrm { r e f } }$ ：参考策略（通常为监督微调后的模型）。  
 β：温度系数，控制策略更新幅度。

# 优势 ：

 无需单独训练 RM，直接通过偏好数据驱动策略优化。  
 训练速度更快，资源消耗更低。

# 3. GRPO（群组相对策略优化）

![](images/93ee178a4f83846420437577ce649d5a48390cf1161f2366bacded40cd6440c8.jpg)  
GRPO (Group Relative Policy Optimization) 2024 DeepSeek

# 核心公式

2.组内相对优势(核心创新）

5.KL散度正则

#

√无需Critic/Value网络  
√组内相对奖励归一化  
√结合 PPO Clip 机制  
√适合大语言模型训练  
√显著降低内存占用

核心思想：通过组内样本的奖励归一化计算相对优势，替代传统价值网络。

# 损失函数表达式：

$$
\mathcal {L} _ {\mathrm {G R P O}} (\theta) = \mathbb {E} _ {q \sim Q} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \min  \left(r _ {i, t} \hat {A} _ {i, t}, \operatorname {c l i p} \left(r _ {i, t}, 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {i, t}\right) - \beta D _ {\mathrm {K L}} \left(\pi_ {\theta} \| \pi_ {\text {r e f}}\right) \right]
$$

# 参数说明 ：

$\hat { A } _ { i , t } = \frac { r _ { i } - \mu } { \sigma }$ ：组内归一化优势（ $\mu , \sigma$ 为组内均值和标准差）；  
 $\beta D _ { \mathrm { K L } }$ ：KL 散度约束（防止偏离参考策略）。

特点：显存占用降低 $40 \%$ ，专精可验证任务（如数学推理、代码生成）。

# 二、算法对比分析

<table><tr><td>维度</td><td>PPO</td><td>DPO</td><td>GRPO</td></tr><tr><td>损失函数核心</td><td>裁剪概率比 + KL 惩罚 + 价值函数损失</td><td>偏好对的概率比对数差</td><td>组内归一化优势 + KL 约束</td></tr><tr><td>需奖励模型</td><td>是</td><td>否</td><td>否</td></tr><tr><td>计算复杂度</td><td>高（需 Actor-Critic 双网络）</td><td>低（监督学习式优化）</td><td>中（组采样增加开销）</td></tr><tr><td>训练效率</td><td>慢（两阶段训练）</td><td>快（提速 30%-50%）</td><td>中（依赖组大小 G）</td></tr><tr><td>稳定性</td><td>中等（依赖 RM 质量）</td><td>高（直接约束策略）</td><td>高（KL 显式约束）</td></tr><tr><td>适用场景</td><td>通用对齐（对话、创意生成）</td><td>快速迭代的偏好学习</td><td>可验证任务（数学、代码）</td></tr></table>

# 1. GRPO 算法的核心思想

GRPO（Group Relative Policy Optimization，群体相对策略优化） 是 DeepSeek 团队为提升大语言模型（如数学推理、复杂任务处理能力）训练效率而设计的强化学习算法。其核心思想是通过群组采样和相对奖励归一化，替代传统 PPO 算法中的价值网络（Critic），从而降低计算复杂度并提升训练稳定性。

# 关键特点：

 无需价值网络：直接通过组内样本的奖励对比计算优势函数，省去了价值模型的训练开销。  
群组采样：针对同一输入问题，生成多个输出序列，基于组内奖励分布进行归一化处理，作为优势估计的基准。  
 动态稳定性控制：结合裁剪机制（Clipping）和 KL 散度惩罚，防止策略更新偏离参考策略过远。

# 2. GRPO 的优势函数

GRPO 的优势函数通过以下步骤计算：

1. 群组采样：对每个输入问题，使用旧策略生成 G 个不同的输出序列（如 $\scriptstyle { \mathsf { G } } = 4 \sim 8$ ）。  
2. 奖励计算：对每个输出序列计算累积奖励（例如数学问题的答案正确性、格式规范性）。  
3. 奖励归一化：将组内奖励标准化（例如减去均值、除以标准差），得到归一化后的奖励值作为优势估计。

4. 优势函数：归一化后的奖励直接作为优势值，即：

$$
A _ {t} = \frac {\text {奖 励} - \mu_ {g r o u p}}{\sigma_ {g r o u p}}
$$

其中，μ_group 和 σ_group 分别为组内奖励的均值和标准差。

# 与传统 PPO 的对比：

 PPO 需通过价值网络估计优势函数，而 GRPO 仅依赖组内样本的统计特性，降低了计算复杂度。  
 归一化处理减少了奖励的绝对数值波动对策略更新的影响，提升了训练稳定性。

# 3. GRPO 的优化目标函数

GRPO 的优化目标函数由三部分组成：

$$
\mathcal {L} _ {\mathrm {G R P O}} (\theta) = \mathbb {E} _ {q \sim Q} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \min  \left(r _ {i, t} \hat {A} _ {i, t}, \operatorname {c l i p} \left(r _ {i, t}, 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {i, t}\right) - \beta D _ {\mathrm {K L}} \left(\pi_ {\theta} \| \pi_ {\text {r e f}}\right) \right]
$$

 策略梯度项：鼓励模型生成高奖励的输出序列，基于归一化后的优势值计算。  
 裁剪项：限制新旧策略的概率比变化幅度（如裁剪范围 0.8,1.2），防止策略突变。  
 KL 散度惩罚项：约束新策略与参考策略（如 SFT 模型）的偏离程度，提升训练稳定性。

# 4. GRPO 的优势与局限性

# .  优势：

 高效性：省去价值网络，内存和计算开销降低约 $40 \%$ 。  
 稳定性：组内归一化和 KL 散度约束使训练中断率从 PPO 的 $1 7 \%$ 降至 $2 . 3 \%$ 。  
 适用性：特别适合数学推理、编程等需要精确答案的任务（如 DeepSeek-Math、DeepSeek-R1）。

#  局限性：

 依赖参考策略质量：初始参考策略（如 SFT 模型）需具备一定性能，否则影响优化效果。  
 超参数敏感：裁剪范围、KL 系数等需精细调参。

