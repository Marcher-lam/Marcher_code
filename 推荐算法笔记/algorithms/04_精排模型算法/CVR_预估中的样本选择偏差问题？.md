# 面试题：CVR 预估中的样本选择偏差问题？

面试题：CVR 预估中的样本选择偏差问题？

# 1. 样本选择偏差定义

在 CVR（转化率）预估中， 样本选择偏差（Sample Selection Bias, SSB）指训练数据与推理数据分布不一致的现象。CVR 模型通常基于点击后样本 （即用户点击了广告/商品后的行为数据）训练，而实际推理时需对所有曝光样本 （无论是否被点击）进行预测。

由于点击行为本身具有低概率特性（通常不足 $1 \%$ ），训练样本仅覆盖了整体曝光样本的极小子集，导致模型在训练阶段学习的分布与实际在线预测阶段的分布存在显著差异。

# 2. 具体表现

 分布偏移：点击样本的特征空间（如用户兴趣、商品属性）可能与未点击样本差异较大，模型无法泛化到未点击样本。   
 数据稀疏：点击样本量远小于曝光样本量，导致模型难以学习未点击样本的特征表示。   
 反事实偏差：未点击样本的真实转化行为未知，直接将其视为负样本会引入噪声。

# 3. 缓解样本选择偏差的算法模型

ESMM（Entire Space Multi-Task Model）

核心思想：通过多任务学习同时建模 CTR（点击率）和CTCVR（曝光后点击且转化的概率），间接推导 CVR，使训练数据覆盖全曝光样本空间。

![](images/e7f00ce1fedffece853680ca803512717d37c772079b9d99af440234875b0ada.jpg)

$$
\underbrace {p (y = 1 , z = 1 | \boldsymbol {x})} _ {p C T C V R} = \underbrace {p (y = 1 | \boldsymbol {x})} _ {p C T R} \times \underbrace {p (z = 1 | y = 1 , \boldsymbol {x})} _ {p C V R}.
$$

模型公式：

# 优点：

 全空间训练，消除 SSB。共享 Embedding 参数，缓解数据稀疏（DS）问题。

# 缺点：

 假设 CTR 和 CTCVR 独立，可能低估 CVR（PIP 问题）。  
 Potential Independance Priority（潜在独立先验），ESMM 分别建模 CTR 和 CVR，会忽视"转化"依赖于"点击"这一因果关系，即：ESMM 模型结构上 CVR 的预测是不依赖于 click 的，但真实情况是发生点击后，才会发生转化，是有依赖关系的，导致 CTCVR预估不准。

ESCM²（Entire Space Counterfactual Multi-Task Model）

论文链接：ESCM2: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation

# 核心思想：

 ESMM 主要解决样本选择偏差和数据稀疏问题，但存在固有估计偏差（IEB）和潜在独立性优先（PIP）问题；

 固有估计偏差（IEB）：CTCVR=CTR×CVR 的乘法假设在非独立场景下失效。  
 潜在独立性优先（PIP）：CTR 与 CVR 的联合优化可能掩盖因果关系。

 引入因果推断中的反事实学习，通过调整样本权重（如逆倾向加权 IPW）或双重稳健估计（ Doubly Robust，DR）解决SSB。

![](images/0f9663f92a8b847be23596f2611112c09fff66d5a2c1f9043d1d04b1d63baea4.jpg)

# 模型公式：

# 1. 逆倾向加权（Inverse Propensity Weighting, IPW）

通过 CTR预估值（倾向分）调整点击样本权重，消除选择偏差：

$$
\begin{array}{l} \mathcal {R} _ {\mathrm {I P S}} \left(\phi_ {\mathrm {C T R}}, \phi_ {\mathrm {C V R}}\right) = \mathbb {E} _ {(u, i) \in \mathcal {D}} \left[ \frac {\hat {o} _ {u , i} \delta \left(r _ {u , i} , \hat {r} _ {u , i} \left(x _ {u , i} ; \phi_ {\mathrm {C V R}}\right)\right)}{\hat {o} _ {u , i} \left(x _ {u , i} ; \phi_ {\mathrm {C T R}}\right)} \right] \\ = \frac {1}{| \mathcal {D} |} \sum_ {(u, i) \in \mathcal {D}} \frac {\mathcal {O} _ {u , i} \delta (r _ {u , i} , \hat {r} _ {u , i} (x _ {u , i} ; \phi_ {\mathrm {C V R}}))}{\hat {\sigma} _ {u , i} (x _ {u , i} ; \phi_ {\mathrm {C T R}})}, \\ \end{array}
$$

其中， $o _ { u , i }$ 表示是否点击， $\hat { o } _ { u , i }$ 表示 CTR 预估值， $\delta ( r _ { u , i } , \hat { r } _ { u , i } )$ 表示 CVR 预估值的 loss 误差(交叉熵)。

# 2. 双重稳健估计（Doubly Robust, DR）

 结合 IPW 与误差纠正模型（Imputation Model，IM），降低方差并提升稳健性，若倾向分或误差模型之一正确，则估计无偏。

$$
\begin{array}{l} \mathcal {R} _ {\mathrm {D R}} ^ {\text {e r r}} \left(\phi_ {\mathrm {C T R}}, \phi_ {\mathrm {C V R}}, \phi_ {\mathrm {I M P}}\right) \\ = \mathbb {E} _ {(u, i) \in \mathcal {D}} \left[ \hat {\delta} _ {u, i} (x _ {u, i}; \phi_ {\mathrm {I M P}}) + \frac {o _ {u , i} \hat {e} _ {u , i} (x _ {u , i} ; \phi_ {\mathrm {C V R}} , \phi_ {\mathrm {I M P}})}{\hat {\sigma} _ {u , i} (x _ {u , i} ; \phi_ {\mathrm {C T R}})} \right] \\ \end{array}
$$

其中， $\hat { e } _ { u , i } = \delta _ { u , i } - \hat { \delta } _ { u , i }$ 表示两者的差值，

$\hat { \delta } _ { u , i }$ 表示 Imputation Model 预估 CVR 误差（其 label $\boldsymbol { \mathfrak { H } } ^ { \delta ( r _ { u , i } , \hat { r } _ { u , i } ) }$ ，这点比较绕）。

 上述的 DR Loss，需加上如下 mse loss，减少真实 cvr loss 与 imputed cvr loss 之间的距离，保证 准确性：

$$
\begin{array}{l} \mathcal {R} _ {\mathrm {D R}} ^ {\mathrm {i m p}} \left(\phi_ {\mathrm {C T R}}, \phi_ {\mathrm {C V R}}, \phi_ {\mathrm {I M P}}\right) \\ = \mathbb {E} _ {(u, i) \in \mathcal {D}} \left[ \frac {o _ {u , i} \hat {e} _ {u , i} ^ {2} \left(x _ {u , i} ; \phi_ {\mathrm {C V R}} , \phi_ {\mathrm {I M P}}\right)}{\delta_ {u , i} \left(x _ {u , i} ; \phi_ {\mathrm {C T R}}\right)} \right] \\ \end{array}
$$

# 3. 损失函数设计

$$
\mathcal {L} = \lambda_ {C T R} \mathcal {L} _ {C T R} + \lambda_ {D R} \mathcal {R} _ {D R} + \lambda_ {C T C V R} \mathcal {L} _ {C T C V R}
$$

# 4. 模型优缺点：

优点：

 通过 IPW 或 DR 调整样本权重，减少 MNAR（缺失非随机）偏差。  
 结合误差修正 Imputation Model 模型，提升鲁棒性，IPW 和 Imputation Model 两者只要有一个准确，即可保证 CVRSSB 纠偏。

缺点：

 依赖 CTR 预测准确性，若 CTR 偏差较大，修正效果受限。  
 DR 方法需额外训练误差模型，增加 $30 \%$ 以上计算开销。

UKD（Uncertainty-regularized Knowledge Distillation）

论文：UKD: Debiasing Conversion Rate Estimation via Uncertainty-regularized Knowledge Distillation

# ESMM 类方法的局限性：

 乘法假设偏差：CTCVR=CTR×CVR 的独立性假设在非独立场景下失效。  
 未点击样本的梯度误导：ESMM 会将未点击样本的 CVR 向 0 优化（因 CTCVR 任务梯度恒正），但真实情况中未点击样本的转化标签应为未知而非 0。

# UKD 框架：基于知识蒸馏的去偏方法

UKD 通过教师-学生模型框架，结合对抗学习和不确定性正则化，解决样本选择偏差问题

![](images/2d0767f550bde915d3b0b6912fc9cd02f7b5d2120cde75e32640c541974539c3.jpg)

# 1. 教师模型：点击自适应表征与伪标签生成

目标：为未点击样本生成可靠的伪转化标签，使其能够参与全空间训练。

# 模型结构：

特征提取器：将输入特征映射为表征 $h _ { \circ }$   
 CVR 预测器：输出 CVR 预估值 p_conv 。  
域判别器：区分点击与未点击样本的表征分布。

关键思想：通过对抗训练混淆域判别器，使特征表征 $h$ 无法区分点击/未点击样本，从而生成点击自适应伪标签。

# 2. 学生模型：不确定性正则化知识蒸馏

目标：利用教师生成的伪标签训练学生模型，同时通过不确定性建模抑制噪声影响。

# 模型结构：

共享特征层：与 CTR 任务共享 Embedding。  
 双 CVR 预测器：独立预测 p_conv(1) 和 p_conv(2) ，通过 Dropout 增强差异性。  
 不确定性估计：KL 散度衡量两个预测器的不一致性： $u = D _ { K L } ( p _ { c o n v } ^ { ( 1 ) } | | p _ { c o n v } ^ { ( 2 ) } )$

# 动态权重调整：

未点击样本的 CVR损失根据不确定性动态加权：

$$
L _ {C V R - u n c l i c k} = \sum_ {i} \frac {1}{1 + \beta u _ {i}} \cdot \mathcal {L} _ {C V R} \left(p _ {c o n v}, \hat {y} _ {c o n v}\right)
$$

$\beta$ 为超参数，高不确定性样本权重降低，减少噪声干扰。

# 总损失函数：

$$
\mathcal {L} _ {\text {s t u d e n t}} = \lambda_ {C T R} \mathcal {L} _ {C T R} + \lambda_ {C V R} \left(\mathcal {L} _ {C V R - c l i c k} + \mathcal {L} _ {C V R - u n c l i c k}\right)
$$

其中 LCVR-click 为点击样本的真实 CVR 标签损失，LCVR-unclick 为未点击样本的 CVR 伪标签损失

# 关键创新点

领域自适应与对抗学习：教师模型通过对抗训练消除点击/未点击样本的表征差异，生成更可靠的伪标签。  
 不确定性正则化：双预测器设计量化伪标签噪声，动态调整损失权重，避免过拟合噪声样本。  
 全空间训练：学生模型同时利用点击样本（真实标签）和未点击样本（伪标签），直接优化全空间 CVR 预估

三者综合对比  

<table><tr><td>模型</td><td>核心问题</td><td>建模思想</td><td>理论支撑</td></tr><tr><td>ESMM</td><td>样本选择偏差（SSB）和数据稀疏性</td><td>通过多任务隐式建模全空间，利用CTR和CTCVR任务的乘积关系间接学习CVR，避免直接在点击空间训练CVR。</td><td>无理论无偏性证明，依赖乘法假设</td></tr><tr><td>ESCM²</td><td>ESMM的固有估计偏差（IEB）、潜在独立性假设失效（PIP）</td><td>引入因果推断中的反事实学习（IPW/DR），直接在全曝光空间建模CVR，通过逆倾向加权和双重稳健估计消除偏差。</td><td>双重稳健性理论（倾向分或误差模型准确即可无偏）</td></tr><tr><td>UKD</td><td>伪标签噪声与未点击样本利用不足</td><td>基于知识蒸馏框架，教师模型生成未点击样本的伪标签，学生模型通过不确定性正则化抑制噪声，实现全空间训练。</td><td>领域自适应与KL散度不确定性建模，无严格无偏证明</td></tr></table>

<table><tr><td>维度</td><td>ESMM</td><td>ESCM²</td><td>UKD</td></tr><tr><td>训练空间</td><td>隐式全空间（乘法假设）</td><td>显式全空间（IPW/DR纠偏）</td><td>显式全空间（伪标签蒸馏）</td></tr><tr><td>偏差处理</td><td>无法解决IEB和PIP</td><td>通过因果推断消除IEB和PIP</td><td>通过伪标签与噪声抑制缓解SSB</td></tr><tr><td>数据利用率</td><td>仅间接利用未点击样本</td><td>间接利用（倾向分加权）</td><td>直接利用未点击样本（伪标签）</td></tr><tr><td>计算复杂度</td><td>低</td><td>高（需误差模型和倾向分动态更新）</td><td>中等（对抗训练和双塔预测）</td></tr><tr><td>适用场景</td><td>粗排或低偏差场景（如内容推荐）</td><td>高精度CVR需求场景（如电商广告）</td><td>高噪声或未点击样本丰富场景</td></tr></table>

在 CVR 预估中，延迟反馈问题（Delayed Feedback）的经典解决方案是延迟反馈模型（Delayed Feedback Model, DFM）。

# 一、DFM 核心思想

DFM的核心是通过联合建模转化率（CVR）和转化延迟时间分布，解决因延迟反馈导致的假负样本问题。以下是其核心公式和推导过程：

# 1. 变量定义

 特征： $X$ （用户、广告特征）  
 隐变量： $C \in \{ 0 , 1 \}$ （最终是否转化）  
 观测变量： $Y \in \{ 0 , 1 \}$ （当前是否已观测到转化）  
 延迟时间： $D$ （点击到转化的时间间隔，若未转化则未定义）  
经过时间： $E$ （从点击到当前观测的时间）

# 2. 概率模型

 CVR 模型：预估最终转化概率

$$
p (C = 1 | X) = \sigma (w ^ {T} X) _ {(\text {逻 辑 回 归 形 式})}
$$

其中， $\sigma$ 为 sigmoid 函数， $w$ 为模型参数

 延迟时间模型：假设延迟时间 $D$ 服从指数分布

$$
p (D | X) = \lambda (X) e ^ {- \lambda (X) D}, \lambda (X) = e ^ {v ^ {T} X} \quad \text {其 中 ,} \lambda (X) \text {为 与 特 征 相 关 的 指 数 分 布 参 数}
$$

# 3. 联合概率分布

观测到转化（ $\forall = 1$ ）： $p ( Y = 1 , D | X ) = p ( C = 1 | X ) \cdot p ( D | X )$   
未观测到转化（ $\forall = \pmb { 0 }$ ，包含观测窗口以外的真实转化）：

$$
p (Y = 0 | X, E) = p (C = 0 | X) + p (C = 1 | X) \cdot p (D > E | X)
$$

其中，p(D>E|X)=e-(X)E $p ( D > E | X ) = e ^ { - \lambda ( X ) E }$ 为延迟时间超过观测窗口的概率。

# 4. 损失函数

基于最大似然估计（MLE），损失函数为负对数似然：

$$
\mathcal {L} = - \sum_ {Y = 1} \log p (Y = 1, D | X) - \sum_ {Y = 0} \log p (Y = 0 | X, E)
$$

具体展开后：

$$
\mathcal {L} = - \sum_ {Y = 1} [ \log \sigma (w ^ {T} X) + \log \lambda (X) - \lambda (X) D ] - \sum_ {Y = 0} \log [ 1 - \sigma (w ^ {T} X) + \sigma (w ^ {T} X) e ^ {- \lambda (X) E} ]
$$

该损失函数需同时对 $w$ （CVR 参数）和 $v$ （延迟参数）进行优化。

# 二、实现方法与训练流程

# 1. EM 算法迭代

DFM 通常通过 EM 算法交替优化 CVR 模型和延迟模型：

E 步：计算隐变量 C 的后验概率

$$
p (C = 1 | Y = 0, X, E) = \frac {\sigma \left(w ^ {T} X\right) e ^ {- \lambda (X) E}}{1 - \sigma \left(w ^ {T} X\right) + \sigma \left(w ^ {T} X\right) e ^ {- \lambda (X) E}}
$$

 M 步：固定隐变量后验分布，分别优化 w 和 $v$ 参数。

# 2. 梯度下降优化

实际工程中，常直接使用梯度下降联合优化：

import torch   
class DFM(torch(nnModule): def__init__(self，input_dim): super().__init_(） self.cvr_layer $=$ torch.mm.Linear(input_dim,1)#CVR模型 self.delay_layer $=$ torch.mm.Linear(input_dim,1)#延迟参数模型

```python
def forward(self, X, Y, D, E):
    cvr_logit = self.cvr_layer(X)
    lambda_logit = self.delay_layer(X)
    lambda_ = torch.exp( lambda_logit)
# 计算损失
loss_pos = -torch.log(torch.sigmoid(cvr_logit)) - \torch.log( lambda_ + lambda_ * D
loss_neg = -torch.log(1 - torch.sigmoid(cvr_logit) + \torch.sigmoid(cvr_logit) * torch.exp(-lambda_ * E))
total_loss = torch.sum(Y * loss_pos + (1 - Y) * loss_neg)
return total_loss 
```

# 三、优化与变体

# 1. 非参数延迟分布（NPDFM）

原始 DFM 假设延迟时间服从指数分布，但实际场景可能更复杂。非参数模型（如分位数回归或生存分析）可替代指数分布假设。

# 2. 在线学习（ES-DFM）

结合流式数据动态调整样本权重，缓解分布偏移问题：

$$
\mathcal {L} _ {E S - D F M} = \sum_ {i} \frac {1}{p \left(e _ {i} \mid X _ {i}\right)} \cdot \mathcal {L} \left(X _ {i}, Y _ {i}, D _ {i}, E _ {i}\right)
$$

其中， $p ( e | X )$ 为动态采样分布。

# 四、工程实践建议

# 1 数据预处理：

1. 对延迟时间 $D$ 进行归一化，避免数值不稳定。  
2. 对未转化样本（ $\mathsf { Y } { = } 0$ ）记录最大观测时间 E。

# 2 模型校准：

使用 Platt Scaling 或 Isotonic Regression 校准 CVR 预估值，减少因延迟假设引入的偏差。

# 3 线上部署：

3. 仅部署 CVR 模型，延迟模型仅用于训练阶段。  
4. 实时更新模型参数，适应延迟分布变化。

