# 面试题：字节 OneTrans 模型介绍，高效整合序列建模和特征交互的大一统模型

# 面试题：字节 OneTrans 模型介绍，高效整合序列建模和特征交互的大一统模型

字节跳动提出的 OneTrans 模型，通过一个统一的 Transformer 架构，有效地将推荐系统中两个核心任务——用户行为序列建模和非序列特征交互——进行了整合。

<table><tr><td></td><td>内容</td></tr><tr><td>论文标题</td><td>OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender</td></tr><tr><td>论文链接</td><td>https://arxiv.org/abs/2510.26104</td></tr><tr><td>背景问题</td><td>传统推荐系统排序模型将序列建模（如DIN）和特征交互（如DCNv2）作为独立模块，限制了双向信息流动，且不利于统一优化和扩展，存在以下局限：
·信息流动受阻：序列特征和非序列特征之间的信息无法进行双向、充分的交互。例如，用户的静态画像（如年龄）难以直接影响对其行为序列的解读。
·优化与扩展困难：分离的模块导致模型结构碎片化，难以应用大语言模型（LLM）中成熟的优化技术（如KV缓存），也阻碍了模型的统一扩展。</td></tr><tr><td>核心目标</td><td>提出一个统一的Transformer骨干网络，同时处理序列建模和特征交互，促进信息双向交换，并借鉴大语言模型（LLM）的优化技术实现高效训练和推理。</td></tr><tr><td>关键创新点</td><td>1. 统一Tokenizer 处理多源特征
2. 混合参数化（序列Token共享参数，非序列Token独有参数）
3. 金字塔堆叠结构渐进式压缩信息
4. 跨请求KV缓存等LLM优化技术</td></tr><tr><td>实验效果</td><td>离线实验：CTR预测AUC提升1.53%，CVR预测UAUC提升3.23%。
线上A/B：在TikTok电商场景下，用户人均订单数提升4.35%，人均GMV提升5.68%，同时推理延迟有所降低。</td></tr></table>

# ① 1 背景：从"分治"到"统一"的架构演进

在推荐系统的精排阶段，理解用户兴趣主要依赖两方面信息：

 一是用户的历史行为序列（如点击、购买记录），  
 二是非序列特征（如用户画像、商品属性、上下文信息）。

传统方法采用"先编码后交互"的范式：先用一个模块（如DIN）从行为序列中学习用户兴趣表示，再将这个表示与非序列特征拼接，送入另一个模块（如 DCNv2）进行高阶特征交叉。

这种"分治"策略存在明显瓶颈：

 信息流动壁垒：序列建模模块无法利用用户画像、当前场景等非序列特征来辅助理解历史行为；反之，特征交互模块也难以在早期获得序列信息的滋养。  
 系统效率低下：模块分立导致计算图碎片化，无法应用 LLM的高效优化技术（如 KV缓存），增加了推理时延，也阻碍了模型的统一缩放。

![](images/b1a4d695523a229d088d8cba9994dd9c99bbd4234af06cb04718e165c37ee53a.jpg)  
(a) Conventional Approach

![](images/2febb1af5f11df2935ea277e20f11fc29ba730671540016d79044c4bd2412aa9.jpg)  
(b) OneTrans

OneTrans 的核心思想就是拆掉这堵"模块墙"，用一个统一的 Transformer 模型来协同完成这两项任务。

# ① 2 模型原理

OneTrans 的框架主要包含以下几个关键设计：

![](images/4d8a73e3201d7505451999595464a7a7b7ae9b6448411541863f5614517043cb.jpg)

# 1. 统一特征 Token 化

模型首先将异构的输入特征映射到统一的 Token 空间。

 非序列特征 Token 化：对于用户画像、商品属性、上下文等上百个非序列特征，OneTrans 采用了 Auto-Split Tokenizer。该方法将所有特征拼接后通过一个共享的 MLP，再分割成固定数量的 Token。这种方法相比按语义分组处理的 Group-wiseTokenizer 更直接高效。  
 序列特征 Token 化：对于多种类型的行为序列（点击、加购等），先将每个行为项通过 MLP 投影，然后融合。融合策略上，时间戳感知融合（按真实发生时间交错混合所有行为）被证明优于按行为重要性排序的策略。

# 2. OneTrans 块与混合参数化

统一的Token序列（序列Token在前，非序列Token在后）被送入堆叠的 OneTrans块中。这是模型最具创新性的部分，它采用了混合参数化策略来应对Token的异质性：

 序列 Token：所有代表历史行为的序列 Token 共享一套 Q、K、V 投影矩阵和 FFN 的权重。这种共享机制提升了计算效率，并促进了跨时间步的泛化。  
 非序列 Token：每个代表特定静态特征的非序列Token都拥有自己专属的Q、K、V和FFN权重。这保留了非序列特征的独特语义，使模型能精细学习特征间的交叉。

在注意力机制上，采用因果注意力掩码：序列 Token 只能关注其之前的序列 Token，而非序列 Token 可以关注所有序列Token及它之前的非序列Token，从而实现了两类特征间的双向、受控交互。

# 3. 金字塔堆叠与信息蒸馏

为了高效处理长序列，OneTrans 引入了金字塔式结构。随着网络层数的加深，每一层只保留最近的一部分序列 Token 作为Query，而 Key 和 Value 则基于完整的序列计算。这样做有两个好处：

信息蒸馏：迫使模型将长序列中的信息逐步浓缩、提炼到后续的 Token 和非序列 Token 中。  
 计算效率：显著减少了需要计算的 Query 数量，降低了注意力机制的计算复杂度，节约了内存和计算资源。

# 4. 借鉴 LLM 的优化技术

OneTrans 巧妙地借鉴了LLM的成熟优化技术，这对于工业部署至关重要：

 跨请求 KV缓存：在一个请求内，用户的行为序列（序列Token）对于所有候选商品是共享的。OneTrans 采用两阶段计算：先计算并缓存序列 Token的键值对；对于每个候选商品，只需计算其非序列 Token，再与缓存的历史序列信息进行交叉注意力计算。这使序列计算复杂度从 O(L)降至O(ΔL)（ΔL是新行为数量）。  
其他优化：同时集成 FlashAttention-2 和混合精度训练，进一步降低了训练内存消耗并提升了推理速度。

# ① 实验效果与性能表现

# 离线实验

在字节跳动的大规模工业数据集上，OneTrans 与多种强基线模型进行了对比。

 OneTrans-S（91M 参数）：在 CTR 任务上 AUC 相对提升 $1 . 1 3 \%$ ，CVR 任务上 AUC 相对提升 $0 . 9 0 \%$ 。  
 OneTrans-L（330M 参数）：提升更为显著，CTR AUC 相对提升 $1 . 5 3 \%$ ，CVR 的用户级 AUC 相对提升 $3 . 2 3 \%$

消融实验验证了其关键设计的有效性：Auto-Split Tokenizer 优于分组方式，时间戳感知融合最优，为非序列 Token 分配特定参数至关重要等。

<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="2">CTR</td><td colspan="2">CVR (order)</td><td colspan="2">Efficiency</td></tr><tr><td>AUC↑</td><td>UAUC↑</td><td>AUC↑</td><td>UAUC↑</td><td>Params (M)</td><td>TFLOPs</td></tr><tr><td>(1) Base model</td><td>DCNv2 + DIN (base)*</td><td>0.79623</td><td>0.71927</td><td>0.90361</td><td>0.71955</td><td>10</td><td>0.06</td></tr><tr><td rowspan="3">(2) Feature-interaction</td><td>Wukong + DIN</td><td>+0.08%</td><td>+0.11%</td><td>+0.14%</td><td>+0.11%</td><td>28</td><td>0.54</td></tr><tr><td>HiFormer + DIN</td><td>+0.11%</td><td>+0.18%</td><td>+0.23%</td><td>-0.20%</td><td>108</td><td>1.35</td></tr><tr><td>RankMixer + DIN*</td><td>+0.27%</td><td>+0.36%</td><td>+0.43%</td><td>+0.19%</td><td>107</td><td>1.31</</tr></tr></table>

# 线上 A/B 测试

在 TikTok 电商的真实场景中，OneTrans-L 与参数量约 100M 的先进基线（RankMixer+Transformer）进行对比，取得了显著的业务增长：

 信息流场景：人均订单数提升 $4 . 3 5 \%$ ，人均 GMV 提升 $5 . 6 8 \%$ 。  
 商城场景：人均订单数提升 $2 . 5 8 \%$ ，人均 GMV提升 $3 . 6 7 \%$ 。  
系统效率：在取得效果提升的同时，模型推理延迟还降低了约 $3 \%$ ，展示其优异的工程优化水平。

# 总结

 OneTrans 模型的核心贡献在于，它成功地将推荐系统中的【序列建模】和【特征交互】两个关键任务统一到了一个简洁、强大的 Transformer 架构中。  
 它通过混合参数化策略巧妙解决了特征异质性难题，并通过金字塔堆叠和跨请求 KV 缓存等设计，在保证模型性能的同时，极大地提升了计算效率，满足了工业应用对低延迟和高吞吐的严苛要求。  
 该工作不仅提升了推荐效果，更重要的是为推荐模型的设计提供了一个新的、可扩展的范式，标志着推荐系统向"大一统"的架构演进迈出了关键一步。

 论文标题：《TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders》   
 论文链接：https://arxiv.org/abs/2602.06563   
 发表单位&年份：字节跳动，2026   
 关键词：大模型 Scaling Up、精排模型、推荐系统、TokenMixer、混合专家 (MoE)、工业部署

# 一、 研究背景

推荐系统是互联网生态的核心，但其深度学习模型在扩展时面临瓶颈。早期的扩展尝试往往只增加模型宽度或参数，缺乏对架构的深思熟虑。后续一些工作（如Wukong、HiFormer、DHEN）改进了结构，但常忽视硬件协同设计，导致硬件利用率不足、性能不优。

此前提出的 TokenMixer 架构（即 RankMixer）用轻量级的 Token 混合算子替代 Transformer 中的自注意力，平衡了效果与效率，但在更深的配置中遇到了瓶颈：

# RankMixer 存在的问题：

 次优的残差设计：残差连接中，混合前后的Token 维度与语义可能不匹配，阻碍信息传播。  
 模型架构不"纯"：历史遗留了许多碎片化算子，计算强度低但内存开销高，拉低整体硬件利用率。  
 深层模型梯度更新不足：原TokenMixer 通常较浅（如 2层），增加深度后难以稳定训练和获得增益。  
 MoE 稀疏化不充分：RankMixer 使用"稠密训练，稀疏推理"的 MoE 范式，无法降低训练成本，且激活的专家数动态变化，对推理不友好。  
 扩展探索有限：受框架和训练效率限制，参数规模仅达到约 10 亿。

TokenMixer-Large 的目标就是通过系统性的架构演进，设计一个面向极大规模推荐的模型，解决上述问题。

# 二、 TokenMixer-Large 核心技术

模型整体架构包含三部分：Token 化、TokenMixer-Large 模块、稀疏化 Per-token 混合专家。

# 1. Token 化

 将高维稀疏特征（用户、物品、行为序列、交叉特征等）通过嵌入层映射为低维稠密向量。  
 考虑到特征异构性，模型按语义对嵌入分组，每组分别用不同的 MLP 压缩对齐为固定维度的语义 Token。  
 此外，引入一个全局 Token 来聚合全局信息（类似 BERT 的[CLS]），并与各语义 Token 拼接，形成模型的输入。

# 2. TokenMixer-Large 模块

![](images/4b9febcbd6ceb4c192f9d41044e5e9f1e675d26087c867e4889c66f7192514ce.jpg)

这是模型的核心，采用堆叠结构。每个模块包含三个关键部分：

#  混合与还原：

 这是解决原 TokenMixer 维度不匹配问题的核心。原方法在一次混合后 Token 数量会变化，导致残差连接断裂。  
 TokenMixer-Large 采用对称的两层结构：第一层混合原始 Token 间信息，第二层专门将混合后的 Token 还原回原始维度。这确保了输入输出维度一致，建立了稳定的残差通路。

#  Per-token SwiGLU：

 将 RankMixer 中的 Per-token FFN 升级为 Per-token SwiGLU 激活函数。  
 pSwiGLU(x) $=$ FC_down(Swish(FC_gate(x)) ⋅ FC_up(x))，其中权重矩阵是每个 Token 独立的，以建模 Token 间的特征异质性。

#  残差连接与归一化：

 标准残差：采用 Pre-Norm 设计（将 LayerNorm 置于残差分支计算前）替换原有的 Post-Norm，以提升训练稳定性。同时，用更轻量的 RMSNorm 替代 LayerNorm。  
 层间残差与辅助损失：除了标准残差，每隔几层添加层间残差连接，将底层特征直接传到高层，缓解梯度消失。同时，计算底层输出与高层输出的联合损失，形成辅助损失，迫使底层学习"预测高层特征的偏差"，增强其表征能力，确保深层网络中所有参数都得到充分训练。

![](images/ed1fb30e3390e237ca7f73050160f22b62c9d81199fe2f36690444eb0e040f19.jpg)  
Internal Residual

![](images/9dab53b34f3904af257937291c015187953cc538ac654911b2d73d0a5edfa1a7.jpg)  
Auxiliary Loss

# 3. 稀疏化 Per-token 混合专家

为了在扩大规模时保持高性价比，设计了 Sparse-Pertoken MoE。

 策略：采用"先扩大，后稀疏"的迭代策略。先设计出性能最佳的全激活稠密模型，再将每个Token的SwiGLU精细化为多个子专家，并进行稀疏激活，实现"稀疏训练，稀疏服务"，大幅降低训练和推理成本。

#  关键设计：

 共享专家：引入一个始终被激活的共享专家，以提高训练稳定性和效果。  
 门控值缩放：在路由器 g(·) 的输出前乘以一个常数缩放因子 $\mathtt { a _ { \circ } }$ 。由于稀疏激活，每个专家被更新的概率降低，此操作可放大激活专家的梯度，使其更新更充分。研究发现最佳 α值与稀疏率成反比。  
 下行矩阵小初始化：将 SwiGLU 中最后的下投影矩阵 FC_down 的初始化标准差设为 FC_up/FC_gate 的 1/100（如0.01）。这使得训练早期 $\mathsf { F } ( \mathsf { x } ) { + } \mathsf { x }$ 更接近恒等映射，提升了深层模型的训练稳定性。

# 三、 工程优化

为了支持超大规模模型的高效训练和服务，论文提出了一系列工程优化：

 高性能自定义算子：开发了 MoEPermute、MoEGroupedFFN、MoEUnpermute 等一系列融合算子，减少调度开销，提高设备利用率。  
 FP8 量化：推理时使用 FP8 E4M3 进行后训练量化，在几乎无损精度的情况下实现了 1.7 倍加速。  
 Token 并行：一种专为 TokenMixer-Large 架构设计的模型并行策略。它将模型参数和计算按 Token 维度划分到多个设备，通过对计算流的精心设计，将每层的通信次数从 4次减少到 2次，显著提高了训练和推理吞吐量。

# 四、 实验结果

论文在抖音的电商、信息流广告、直播等多个真实业务场景进行了大规模实验。

# 1. 效果与效率对比：

a. 在参数量约 5 亿的模型对比中，TokenMixer-Large 在 CTCVR 任务上相对 DLRM-MLP 基线取得了 $+ 0 . 9 4 \%$ 的 AUC提升，优于所有基线模型（如 Wukong, DHEN, RankMixer 等）。  
b. 稀疏化Per-token MoE在仅激活一半参数的情况下，性能与稠密模型相当，显著提升了模型的投资回报率。

# 2. Scaling Law 验证：

a. TokenMixer-Large 的性能随参数/FLOPs 增加而提升，且其收益曲线比 RankMixer 更陡峭。  
b. 超越 10 亿参数后，需要平衡地增加模型宽度、深度和缩放因子，才能获得更好回报。模型越大，需要更多训练数据才能完全收敛。  
c. 在离线实验中，模型在广告、电商、直播场景分别成功扩展至 150 亿、70 亿、40 亿参数。

# 3. 消融实验：

a. 验证了"混合与还原"、Per-token SwiGLU、残差连接、层间残差与辅助损失等核心组件的有效性。其中"混合与还原"和 Per-token SwiGLU 贡献最大。  
b. 验证了Sparse-Pertoken MoE中共享专家、门控值缩放、下行矩阵小初始化等设计的正向作用。

# 4. 在线性能：

模型已在字节跳动多个场景上线，服务数亿用户，取得了显著的线上业务指标提升：

 电商：订单量 $+ 1 . 6 6 \%$ ，人均预览支付 $G M V { + } 2 . 9 8 \%$ 。  
 广告：广告主满意度得分 $+ 2 . 0 \%$ 。  
直播：营收 $+ 1 . 4 \%$

# 五、 小结

TokenMixer-Large 是对原有 TokenMixer 架构的一次系统性升级。

 它通过混合与还原操作解决了深层模型的残差传播问题；  
 通过层间残差与辅助损失保障了深层网络的训练稳定性；  
 通过稀疏化 Per-token MoE 及配套的工程优化实现了极大规模下的高效扩展。

该工作不仅在多个业务场景中验证了其卓越的离线效果和线上收益，也为工业级推荐系统模型的架构设计与工程实现提供了重要参考。

# OneTrans 架构代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoSplitTokenizer(nn.Module):
    """非序列特征 Token 化：拼接后通过 MLP 分割为固定数量 Token"""
    def __init__(self, feat_dim, n_tokens, d_token):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, n_tokens * d_token),
            nn.ReLU(),
        )
        self.n_tokens = n_tokens
        self.d_token = d_token

    def forward(self, dense_features):
        projected = self.mlp(dense_features)
        return projected.view(-1, self.n_tokens, self.d_token)


class SequenceTokenizer(nn.Module):
    """序列特征 Token 化：MLP 投影 + 时间戳感知融合"""
    def __init__(self, item_dim, d_token):
        super().__init__()
        self.proj = nn.Linear(item_dim, d_token)

    def forward(self, seq_items):
        return self.proj(seq_items)


class OneTransBlock(nn.Module):
    """OneTrans 块：混合参数化注意力 + FFN"""
    def __init__(self, d_model, n_heads_seq, n_heads_nonseq, d_ff):
        super().__init__()
        self.attn_seq = nn.MultiheadAttention(d_model, n_heads_seq, batch_first=True)
        self.attn_cross = nn.MultiheadAttention(d_model, n_heads_nonseq, batch_first=True)
        self.ffn_seq = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.ffn_nonseq = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
            for _ in range(4)
        ])
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, seq_tokens, nonseq_tokens, attn_mask=None):
        seq_out, _ = self.attn_seq(seq_tokens, seq_tokens, seq_tokens, attn_mask=attn_mask)
        seq_tokens = self.norm1(seq_tokens + seq_out)

        combined = torch.cat([seq_tokens, nonseq_tokens], dim=1)
        nonseq_out, _ = self.attn_cross(nonseq_tokens, combined, combined)
        nonseq_tokens = self.norm2(nonseq_tokens + nonseq_out)

        seq_tokens = seq_tokens + self.ffn_seq(seq_tokens)
        for i in range(nonseq_tokens.size(1)):
            nonseq_tokens[:, i] = nonseq_tokens[:, i] + self.ffn_nonseq[i](nonseq_tokens[:, i:i+1]).squeeze(1)

        return seq_tokens, nonseq_tokens


class OneTrans(nn.Module):
    """OneTrans 完整模型"""
    def __init__(self, feat_dim, item_dim, n_seq_tokens, n_nonseq_tokens, d_model, d_ff, n_layers, n_heads):
        super().__init__()
        self.seq_tokenizer = SequenceTokenizer(item_dim, d_model)
        self.nonseq_tokenizer = AutoSplitTokenizer(feat_dim, n_nonseq_tokens, d_model)
        self.blocks = nn.ModuleList([
            OneTransBlock(d_model, n_heads, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.pred_head = nn.Sequential(
            nn.Linear(d_model * n_nonseq_tokens, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, seq_items, dense_features):
        seq_tokens = self.seq_tokenizer(seq_items)
        nonseq_tokens = self.nonseq_tokenizer(dense_features)

        for block in self.blocks:
            seq_tokens, nonseq_tokens = block(seq_tokens, nonseq_tokens)

        pred = nonseq_tokens.view(nonseq_tokens.size(0), -1)
        return self.pred_head(pred).squeeze(-1)
```

## OneTrans vs 传统分治方案对比

| 维度 | 传统方案 (DIN + DCNv2) | OneTrans |
|------|----------------------|----------|
| 信息流向 | 单向（序列→特征交互） | 双向（统一注意力） |
| 参数效率 | 模块独立，参数冗余 | 共享+独有混合参数化 |
| 推理优化 | 难以应用 KV Cache | 跨请求 KV Cache |
| 扩展性 | 受限于模块碎片化 | 金字塔堆叠统一扩展 |
| 部署复杂度 | 多模块独立部署 | 单一模型端到端 |

## 常见问题与易错点

1. **混合参数化的区分**：序列 Token 共享参数（因为语义同构），非序列 Token 独有参数（因为语义异构），两者不能混淆
2. **因果注意力掩码**：序列部分用因果掩码，非序列部分用全连接掩码，需在 attention 计算时正确设置
3. **KV Cache 的边界**：跨请求缓存仅适用于同一用户的序列 Token，非序列 Token 随候选商品变化需重新计算

## 学习总结

OneTrans 代表了推荐系统精排模型从"分治"到"统一"的架构演进方向。其核心创新在于混合参数化策略（共享+独有）和跨请求 KV Cache，使一个 Transformer 同时高效处理序列建模和特征交互。这种统一架构不仅提升了模型效果，还为后续的模型扩展（如 TokenMixer-Large 的千亿参数）奠定了基础。

# 核心数学公式

## 1. 统一注意力公式

OneTrans 将序列建模和特征交互统一到一个注意力框架中。设序列 Token 集合为 $S = \{s_1, \ldots, s_L\}$，非序列 Token 集合为 $N = \{n_1, \ldots, n_M\}$，统一注意力计算为：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{mask}\right) V
$$

其中混合参数化体现在 $Q$、$K$、$V$ 的投影矩阵上：

- 序列 Token $s_i$：$Q_{s_i} = s_i W_Q^{seq}$，$K_{s_i} = s_i W_K^{seq}$，$V_{s_i} = s_i W_V^{seq}$（共享参数）
- 非序列 Token $n_j$：$Q_{n_j} = n_j W_Q^{n_j}$，$K_{n_j} = n_j W_K^{n_j}$，$V_{n_j} = n_j W_V^{n_j}$（独有参数）

## 2. 因果注意力掩码

OneTrans 的注意力掩码矩阵 $M_{mask}$ 定义如下：

$$
M_{mask}[i][j] = \begin{cases} 0 & \text{if } i, j \in S \text{ and } j \leq i \quad \text{(序列内因果)} \\ 0 & \text{if } i \in N, j \in S \quad \text{(非序列可看全部序列)} \\ 0 & \text{if } i, j \in N \text{ and } j \leq i \quad \text{(非序列间因果)} \\ -\infty & \text{otherwise} \quad \text{(屏蔽)} \end{cases}
$$

这种设计确保了：序列 Token 之间遵循因果性（不能"看到未来"）；非序列 Token 可以看到所有序列 Token（实现双向信息流）；非序列 Token 之间同样遵循因果性。

## 3. 跨特征注意力计算

非序列 Token $n_j$ 对序列信息和非序列信息的注意力聚合：

$$
n_j' = \sum_{k=1}^{L} \alpha_{jk}^{seq} V_{s_k} + \sum_{k=1}^{j} \alpha_{jk}^{nonseq} V_{n_k}
$$

其中注意力权重：

$$
\alpha_{jk}^{seq} = \frac{\exp(q_{n_j}^T k_{s_k} / \sqrt{d_k})}{\sum_{m=1}^{L}\exp(q_{n_j}^T k_{s_m} / \sqrt{d_k}) + \sum_{m=1}^{j}\exp(q_{n_j}^T k_{n_m} / \sqrt{d_k})}
$$

## 4. 金字塔压缩的 Query 选择

第 $l$ 层的金字塔压缩仅保留最近 $L_l$ 个序列 Token 作为 Query：

$$
L_l = \max\left(L_{min}, \left\lfloor L \cdot r^l \right\rfloor\right)
$$

其中 $r \in (0, 1)$ 是压缩率，$L_{min}$ 是最小保留长度。Key 和 Value 仍然基于完整序列计算，确保信息不丢失。

## 5. 跨请求 KV 缓存的复杂度分析

传统方案中，每个候选商品都需要独立计算序列注意力，复杂度为 $O(N_{cand} \cdot L \cdot d)$。

OneTrans 的跨请求 KV 缓存将序列部分计算分摊：

$$
\text{总复杂度} = \underbrace{O(L \cdot d)}_{\text{序列 KV 缓存（一次）}} + N_{cand} \cdot \underbrace{O(M \cdot (L + M) \cdot d)}_{\text{非序列交叉注意力}}
$$

当 $M \ll L$ 时，每个候选商品的计算量从 $O(L^2 d)$ 降至 $O(MLd)$，实现了数量级的加速。
