# 面试题：DIN 原理介绍&带时间衰减的 DIN 代码实现

# 面试题：DIN 原理介绍&带时间衰减的 DIN 代码实现

论文链接：Deep Interest Network

# 一、 DIN 核心原理

DIN 是阿里巴巴提出的动态兴趣建模网络，核心思想是通过注意力机制捕捉用户历史行为与当前候选商品的动态相关性。

# 1. 动态兴趣表征

 传统模型缺陷：Embedding+MLP 结构对用户历史行为进行平均池化或求和池化，导致兴趣表示过于静态（例如用户购买过泳衣和奶粉，推荐泳镜时无法区分两者的重要性）。  
 注意力机制：通过候选广告与历史行为的交互生成注意力权重，加权求和后得到候选 item 相关的用户兴趣向量，实现"千物千面"的个性化表征。公式如下：

$$
v _ {u} = \sum_ {i = 1} ^ {T} \alpha_ {i} e _ {i}, \quad \alpha_ {i} = \operatorname {M L P} \left(e _ {a} \oplus e _ {i} \oplus \left(e _ {a} \odot e _ {i}\right)\right)
$$

其中， $e _ { a }$ 为候选广告 emb， $e _ { i }$ 为历史行为 emb，⋅表示拼接，⋅表示哈达玛积。

# 2. 训练优化技巧

 Dice 激活函数：根据输入分布动态调整激活阈值，公式为：

$$
f (s) = p (s) \cdot s + (1 - p (s)) \cdot \alpha s, \quad p (s) = \frac {1}{1 + e ^ {- \frac {s - E [ s ]}{\sqrt {V a r [ s ]} + c}}}
$$

 小批量感知正则化：仅对当前 mini-batch 中出现的稀疏特征参数计算 L2 正则化，降低计算开销。

![](images/2401d842cb8ca902d8172c001b6449ebc6d373a6328a293b4937ad6766c38224.jpg)

# 二、时间动态衰减的 DIN 改进设计

原始DIN未显式考虑用户兴趣随时间衰减的特性，可通过以下方式引入时间动态衰减：

# 1. 时间衰减因子

对用户历史行为的时间戳 $t _ { i }$ 计算衰减权重 $\beta _ { i }$ ：

$$
\beta_ {i} = \exp (- \lambda \cdot (t _ {\text {c u r r e n t}} - t _ {i}))
$$

其中 为衰减系数， $t _ { c u r r e n t }$ 为当前时间。

# 2. 改进注意力机制

将时间衰减因子融入注意力权重计算：

$$
\alpha_ {i} ^ {\prime} = \alpha_ {i} \cdot \beta_ {i} = \mathrm {M L P} \left(e _ {a} \oplus e _ {i} \oplus \left(e _ {a} \odot e _ {i}\right)\right) \cdot \exp (- \lambda \cdot \Delta t _ {i})
$$

此设计使近期行为对候选广告的权重更高，同时保留原始 DIN 的相关性建模能力。

# 3. 动态衰减的物理意义

 短期兴趣强化：近期点击/购买行为对当前推荐影响更大（如用户昨天浏览的手机比上月浏览的书籍更相关）。  
 长期兴趣保留：通过可学习参数 $\lambda$ 控制衰减速度，避免完全丢弃长期兴趣（如季节性购物习惯）。

# 三、代码实现（PyTorch）

```python
import torch
import torch.nn as nn
import numpy as np
class TimeDecayDIN(nnModule):
    def __init__(self, emb_dim=10000, feat_dim=64, hidden_dim=128):
        super().__init()
        #嵌入层：用户行为、候选广告等特征
        self_embedding = nn.Embedding(emb_dim, feat_dim)
        self.attn_net = nn.Sequential(
            nn.Linear(3*feat_dim, hidden_dim),
            nn.ReLU(),  #替换Dice激活函数
            nn.Linear(hidden_dim, 1))
        self.lambda Decay = nn.Parameter(torch.tensor(0.1))  #时间衰减系数
def forward(self, user_behaviors, candidate_ad, time_deltas):
    #嵌入转换
    e_a = selfembedding(candidate_ad)  #候选广告嵌入
    e_i = selfembedding(user_behaviors)  #历史行为嵌入
    #时间衰减计算
    beta = torch.exp(-self.lambda Decay * time_deltas)  #[bs, len]
    #注意力得分
    batch_size, seq_len, _ = e_i.shape
    e_a Expand = e_a unsqueeze(1).expand(-1, seq_len, -1)
    interaction = torch.cat([e_a Expand, e_i, e_a Expand * e_i], dim=-1)
    alpha = self.attn_net(interaction).squeeze(-1)  #[bs, s_len]
    alpha = alpha * beta  #融入时间衰减
    #动态兴趣向量
    v_u = (alpha softmax(dim=-1).unsqueeze(-1) * e_i).sum(dim=1)
    output = torch.cat([v_u, e_a], dim=-1)  #拼接其他特征并预测
    return output 
```

# 示例用法

```txt
model = TimeDecayDIN()
user_behaviors = torch.randint(0, 1000, (32, 50)) # bs=32, seq_len=50
candidate_ad = torch.randint(0, 1000, (32,))
time_deltas = torch Rand(32, 50) * 30 # 模拟时间差（天）
output = model(user_behaviors, candidate_ad, time_deltas)
print(output.shape) 
```

论文地址：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

# 一、 背景：为什么需要 GQA？

传统 Transformer 中的多头注意力（MHA）每个头独立计算键（Key）、值（Value）和查询（Query），导致：

 高计算开销：参数量为 O(num_heads $\times$ d_model)，推理时需缓存所有头的 KV，显存占用随序列长度线性增长。  
 解码延迟：自回归生成时，MHA 需重复计算 KV，拖累吞吐量。

多查询注意力（MQA）：通过所有头共享一组 KV，显著降低计算量，但牺牲表达能力，尤其对复杂任务精度下降明显。

分组查询注意力（GQA）的提出：在 MHA 和 MQA 间取得平衡——分组共享 KV，减少计算量同时保留多样性。

![](images/969cecaedb6c6d656b2420dca9c1fc955ec8bcaa073fc9c22a5af4c03948699a.jpg)  
Multi-head

![](images/9fe609318a7e98be972c9a213e68a669eab6adab4a8f757994bf525f27b43ece.jpg)  
Grouped-query

![](images/25fb1c10a1a94ccc4920087d36541568589e77c937e20c5ae1f0c2c2bfbf447e.jpg)  
Multi-query

# 二、核心原理与数学表达

# 1. 分组策略

 将 num_heads 个查询头分为 num_groups 组（每组含 num_heads $/$ num_groups 个头）。  
 每组共享一组键（K）和值（V） ，独立计算查询（Q）。

# 2. 计算流程

设输入序列长度 T，隐藏维度 d_model，每组注意力计算为：

$$
\operatorname {A t t e n t i o n} _ {g} \left(Q _ {g}, K _ {g}, V _ {g}\right) = \operatorname {s o f t m a x} \left(\frac {Q _ {g} K _ {g} ^ {T}}{\sqrt {d _ {k}}}\right) V _ {g}
$$

最终输出为各组输出的拼接：

$$
\text {O u t p u t} = \operatorname {C o n c a t} \left(\operatorname {A t t e n t i o n} _ {1}, \dots , \operatorname {A t t e n t i o n} _ {G}\right) W ^ {O}
$$

# 复杂度分析 ：

 计算量：从 MHA 的 O( ${ \mathsf { T } } ^ { \wedge } 2 \times$ num_heads) 降至 O(T^2 × num_groups)。  
 KV 缓存：缓存大小从 $2 \times \top \times$ num_heads $\times$ d_head 压缩至 $2 \times \top \times$ num_groups $\times$ d_head。

# 三、与其他注意力机制对比

<table><tr><td>特性</td><td>MHA (多头注意力)</td><td>MQA (多查询注意力)</td><td>GQA (分组查询注意力)</td></tr><tr><td>KV头数量</td><td>num_heads</td><td>1</td><td>num_groups (可配置)</td></tr><tr><td>计算效率</td><td>低(计算/显存开销大)</td><td>高</td><td>中高(接近 MQA)</td></tr><tr><td>模型质量</td><td>高(强表达能力)</td><td>低(共享 KV 导致信息损失)</td><td>接近 MHA (组内多样性保留)</td></tr><tr><td>适用场景</td><td>短文本、高精度任务</td><td>实时推理、低资源场景</td><td>长文本生成、大规模 LLM</td></tr><tr><td>代表模型</td><td>BERT, GPT-3</td><td>PaLM, StarCoder</td><td>LLaMA-2, Claude, Qwen</td></tr></table>

# ① 论文基本信息

 论文标题：Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free   
 论文地址：https://arxiv.org/abs/2505.06708  
 代码地址：https://github.com/qiuzh20/gated_attention   
 获奖情况：NeurIPS 2025 最佳论文奖 (NeurIPS 2025 Best Paper Award)

该论文通过一个精巧的 Gate 设计，显著提升了大型语言模型的性能、稳定性和长上下文处理能力。以下是该论文的介绍。

# 标注注意力机制的问题：

表达能力的低秩瓶颈：在标准的多头注意力机制中，值（Value）和输出（Output）是两个连续的线性变换。当每个注意力头的维度（d_head）小于模型隐藏层维度（d_model）时，这两个线性层的复合会形成一个低秩线性映射，限制了模型的表达能力，使其难以拟合更复杂的函数。  
 注意力汇聚（Attention Sink）：由于 Softmax 函数要求所有注意力权重之和为 1，模型会倾向于将大量无关的注意力权重（平均高达 $4 6 . 7 \%$ ）分配给序列的第一个 token（如 BOS），挤压对其他相关上下文 token 的关注。  
 训练不稳定性：上述问题常伴随着训练过程中的“损失尖峰”（Loss Spike）和隐藏层的“巨量激活”（数值超 1000），在低精度（如 BF16）训练下易引发数值误差，导致训练崩溃。

# 创新方案：在注意力机制的关键位置加一扇“门”

![](images/d075c909fb5ddd87faf093025527d9c921a5dcf4ebe301b9bc636300bec4bcc5.jpg)

![](images/ec8319deba33121744490f33ef183c2daaa50b0e9bdabfc3db7585675b3ce492.jpg)

研究团队的核心创新在于系统性地探索了在标准注意力机制中引入简单门控的最佳位置和形式 。

# 1. 门控位置探索：

在注意力层的 5 个关键位置测试了门控操作（如上图所示），包括查询（Q）、键（K）、值（V）投影后，缩放点积注意力（SDPA）输出后（G1），以及最终输出投影后（G5）。

# 2. 最优方案发现：

大量实验证明，在 SDPA 输出后（G1 位置）施加一个头部专属、逐元素、基于 Sigmoid 函数的乘性门控，效果最为显著。其数学形式为：

$$
\mathbf {y} _ {i} = \operatorname {S o f t m a x} \left(\frac {\mathbf {q} _ {i} \mathbf {K} ^ {T}}{\sqrt {d _ {k}}}\right) \mathbf {V}
$$

其中， ${ \bf q } _ { i }$ 是当前第 i 个 token 的查询向量，K 和 V 分

标准缩放点积注意力（SDPA）：

别是键和值矩阵。

Gated Attention 在 SDPA 输出后引入门控，其核心公式为： $\mathbf { o } _ { i } = \mathbf { y } _ { i } \odot \sigma ( \mathbf { W } _ { g } \mathbf { x } _ { i } )$ ，

其中：

 $\mathbf { y } _ { i }$ 是 SDPA 的输出。  
 $\mathbf { x } _ { i }$ 是当前 token 在注意力层之前的隐藏状态（通常是 Pre-Norm 后的输出）。  
是一个可学习的线性投影权重。 $\mathbf { W } _ { g }$   
 $\sigma$ 是 Sigmoid 激活函数，将门控分数压缩到(0,1)区间。  
 $\odot$ 表示逐元素相乘（Hadamard 积）。

# 关键分析：门控为何如此有效？

#  引入非线性，突破表达能力瓶颈：

 标准的注意力机制中，Value 投影和输出投影是两个连续的线性变换，构成了一个低秩线性映射，限制了模型的表达能力。  
 G1 位置的门控恰好处在这两个线性层之间，引入了一个非线性操作，极大地增强了模型的表达能力。

#  引入查询依赖的稀疏性，主动过滤信息：

 G1 门控的分数由当前查询 token 计算得出，因此是查询依赖的。研究发现，这些门控分数具有高度稀疏性（平均值仅为0.116），使模型能动态判断并“忽略”对当前 token 无关的历史信息。  
 主动过滤机制从根本上解决了“注意力汇聚”问题，将首 token 的注意力占比从 $4 6 . 7 \%$ 降至 $4 . 8 \%$ ，让注意力分配更均匀合理。

# 效果与应用

该研究提出的门控注意力机制在实践中展现出多重优势：

<table><tr><td>维度</td><td>具体效果</td></tr><tr><td>性能提升</td><td>在多项基准测试（如MMLU、GSM8K）中，仅增加约1%的参数，即可实现困惑度（PPL）稳定降低0.2以上，MMLU得分提升约2分。</td></tr><tr><td>训练稳定性</td><td>门控机制显著平滑了训练损失曲线，减少了损失尖峰，使模型能够承受更大的学习率（如8e-3）和批量大小，从而可能加快训练速度并扩展超参空间。</td></tr><tr><td>长上下文泛化</td><td>由于消除了注意力汇聚，门控模型在长上下文外推任务中表现卓越。在使用YaRN方法将上下文扩展至128k时，其性能衰减远小于基线模型（仅下降6.89% vs 41.56%），展现了强大的长度外推能力。</td></tr><tr><td>实际应用</td><td>该技术已成功应用于Qwen3-Next系列模型。其在长文档处理（如法律、学术文本）和高效训练方面具有显著的工业落地潜力。</td></tr></table>

# 总结

Gated Attention 这项研究的意义在于，它通过严谨、大规模的实验，揭示了一个简单而深刻的道理：大模型的提升不一定需要复杂的架构革命，有时在关键位置添加一个精巧的“开关”，就能显著优化模型的核心行为。

三者均为针对传统 MHA（多头注意力）的优化方案，核心目标是解决长文本场景下 KV缓存显存占用过高、推理速度慢、计算复杂度平方级增长的痛点，但优化路径和适用场景有本质差异。

# 一、核心对比表

<table><tr><td>对比维度</td><td>GQA(分组查询注意力)</td><td>MLA(多头潜变量注意力)</td><td>DSA(DeepSeek 稀疏注意力)</td></tr><tr><td>核心设计思路</td><td>分组共享KV头,平衡MHA精度与MQA效率</td><td>低秩联合压缩KV到潜空间,极致降低KV缓存</td><td>基于MLA,动态筛选Top-K关键Token 做注意力,从稠密计算转为稀疏计算</td></tr><tr><td>核心优化对象</td><td>KV头的数量(减少KV头总数)</td><td>KV的特征维度(压缩单组KV的维度)</td><td>注意力计算的Token数量(减少参与计算的Token总数)</td></tr><tr><td>KV缓存开销</td><td>中等,约为MHA的1/4~1/8(取决于分组数)</td><td>极低,约为MHA的6%~10%</td><td>极致低,200K上下文下较MLA再降75%</td></tr><tr><td>计算复杂度</td><td>O(n²·g·d)(g为分组数,远小于总头数h)</td><td>O(n²·d_c)(d_c为压缩后的潜变量维度,远小于原维度d)</td><td>O(n·k·d)(k为选中的Top-K Token数,远小于序列长度n)</td></tr><tr><td>精度表现</td><td>接近MHA,差距&lt;1%,显著优于MQA</td><td>持平甚至超越MHA,无明显精度损失</td><td>长文本下与MLA基本持平,精度损失&lt;0.5%</td></tr><tr><td>长文本适配上限</td><td>支持128K以内,超过后KV缓存压力仍显著</td><td>支持128K~200K,显存压力大幅缓解</td><td>原生适配200K+超长上下文,推理成本断崖式下降</td></tr><tr><td>核心优势</td><td>实现简单,训练/推理兼容好,通用场景性价比最高</td><td>压缩比高,精度无损,推理速度快,长文本适配性优于GQA</td><td>彻底解决长文本O(n²)计算瓶颈,推理成本极低,不丢失跨全文关键信息</td></tr><tr><td>核心短板</td><td>分组数需手动调优,超长上下文场景收益有限</td><td>对训练优化要求高,算子适配有一定门槛</td><td>稀疏计算需定制算子,训练需两阶段适配,工程复杂度最高</td></tr><tr><td>代表落地模型</td><td>Llama 2/3、GPT-4、Qwen系列</td><td>DeepSeek V2/V3、GLM-5</td><td>DeepSeek V3.2、GLM-5</td></tr></table>

# 二、三者核心原理详解

1. GQA（Grouped-Query Attention，分组查询注意力）

GQA 是目前工业界最通用的折中方案，本质是 MHA（全多头）与 MQA（单 KV 头）的平衡产物。

核心逻辑：将所有 Query 头划分为 G 个组，每组内的所有 Query 头共享同一组 Key 和 Value 头。例如 32个 Query 头分为 4 组，每组 8 个 Query 头共享 1 组 KV 头，最终仅需存储 4 组 KV，KV 缓存直接降至MHA 的 1/8。  
核心特点：实现极简，无需修改注意力计算的核心逻辑，仅需调整 KV 头的数量；即使是用 MHA 预训练的模型，也可通过少量微调适配 GQA，兼容性极强。

2. MLA（Multi-Latent Attention，多头潜变量注意力）

MLA是 DeepSeek 提出的 KV压缩方案，解决了GQA/MQA“减少 KV头数会损失表达能力”的核心缺陷。

核心逻辑：与 GQA“减少 KV 头数”的思路完全不同，MLA 不减少 KV 头数，而是压缩单组 KV 的特征维度：将 KV 的表示拆分为两部分——用于计算注意力分数的低维潜变量（如将 512 维的 K 压缩到 128 维），和用于输出聚合的高维特征。KV 缓存仅需存储低维潜变量，在保留头间独立性的同时，极致降低显存占用。  
关键表现：官方测试中，MLA的 KV缓存仅为MHA的 $6 \% { \sim } 1 0 \%$ ，同时在主流基准测试上的表现全面优于原生 MHA；GLM-5 采用的 MLA-256 变体，将头维度从 192 调整至 256，头数减少 1/3，参数不变的前提下进一步提升了推理速度。

# 3. DSA（DeepSeek Sparse Attention，深度求索稀疏注意力）

DSA 是基于 MLA 的超长上下文优化方案，从根本上解决了注意力计算 ${ \mathsf { O } } ( { \mathsf { n } } ^ { 2 } )$ 的瓶颈。

核心逻辑：传统稠密注意力（MHA/GQA/MLA）要求每个 Token 与全序列所有历史 Token 计算注意力，序列越长，计算量平方级增长。DSA 的核心创新是基于内容动态筛选关键 Token：新增一个轻量级的“闪电索引器（Lightning Indexer）”，先快速扫描全序列历史 Token，选出与当前 Token 最相关的 Top-K 个（如 2048个），仅对这部分Token执行完整的MLA注意力计算，其余Token直接跳过。  
核心优势：与固定滑动窗口（仅看最近 N 个 Token）不同，DSA 是内容感知的动态选择，无论 Token 在序列的开头还是结尾，只要与当前任务相关就会被选中，不会丢失长距离关键信息（如合同核心条款、文档开头的指令）；通过“稠密预热 $\mapsto$ 稀疏过渡”的两阶段训练，可实现精度几乎无损的适配。

# 三、小结

三者是清晰的技术演进路径，适配不同的业务场景：

 通用场景首选GQA：实现简单、兼容性强，在 8K~32K常规上下文场景下，是精度与效率的最优平衡。  
 长文本场景首选 MLA：在 128K 左右的长文本场景下，相比 GQA 能大幅降低显存占用，同时不损失模型精度，适配长文档理解、代码库分析等任务。  
 超长文本场景首选 DSA+MLA：200K+超长上下文场景的最优解，从根本上降低计算与显存成本，适配书籍阅读、全量合同审核、Agent 长链路思考等场景，也是 GLM-5 的核心选型。

# 4.3 序列建模

