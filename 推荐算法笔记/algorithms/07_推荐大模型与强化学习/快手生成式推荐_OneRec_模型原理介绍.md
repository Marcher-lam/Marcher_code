# 面试题：快手生成式推荐 OneRec 模型原理介绍

# 面试题：快手生成式推荐 OneRec 模型原理介绍

快手OneRec 是一种突破性的生成式推荐模型，其核心原理在于通过统一的端到端架构替代传统多阶段推荐流程，结合会话级生成与偏好对齐技术实现推荐系统的范式革新。

链接 OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment

![](images/0b5c28c2caaf95857c3be1ea014315063b52b7e365d40fed97d5a0bc4a0173b8.jpg)

# 一、核心技术原理

# 1. 端到端生成架构

OneRec 采用 Encoder-Decoder 结构，直接输入用户历史行为序列（如观看、点赞记录），一次性输出完整推荐列表（Session）。

相比传统"召回→粗排 精排"级联架构，省去多阶段候选集筛选过程，消除信息传递损耗。

![](images/de80bc1e196b05050ecfd1b33aca839a962f78b27e017717d21a1fc4e5b565b6.jpg)

# 2. 语义 ID 表征体系

 通过残差量化编码将多模态视频特征转化为离散语义 ID，通过 Balanced K-means 算法避免传统 K-means 的"沙漏现象"。  
 视频特征经过层次化残差量化后生成形如[153,4092,7215]的语义 ID，分别对应【粗粒度类别 内容主题 $\xrightarrow { }$ 细粒度特

征】。

 输入序列组织为[BOS]分隔的多层级 token，增强上下文建模能力。

Algorithm 1: Balanced K-means Clustering  
Input: Item set $\mathcal{V}$ , number of clusters $K$ 1 Compute $w \gets |\mathcal{V}| / K$ 2 Initialize centroids $C_l = \{c_1^l, \dots, c_K^l\}$ with random selection;  
3 repeat  
4 Initialize unassigned set $\mathcal{U} \gets \mathcal{V}$ 5 for each cluster $k \in \{1, \dots, K\}$ do  
6 Sort $\mathcal{U}$ by ascending distance from centroid $c_k^l$ ;  
7 Assign $\mathcal{V}_k \gets \mathcal{U}[0 : w - 1]$ ;  
8 Update centroid $c_k^l \gets \frac{1}{w} \sum_{r^l \in \mathcal{V}_k} r^l$ ;  
9 Remove assigned items $\mathcal{U} \gets \mathcal{U} \setminus \mathcal{V}_k$ ;  
10 end  
11 until Assignment convergence;  
Output: Optimized codebook $C_l = \{c_1^l, \dots, c_K^l\}$

# 3. 混合专家扩展（MoE）

 在 Decoder 层引入稀疏 MoE 机制：前馈网络替换为 $_ { \mathsf { N } } = 2 4$ 个专家网络，每个 token 仅激活 Top-2 专家（计算量仅线性增加）。  
 通过负载均衡损失防止专家坍缩。

# 二、生成机制创新

# 1. 会话级生成策略

 定义标准：生成 5-10 个视频组成的 Session，需满足观看数≥5、总时长超阈值、存在互动行为  
 解码控制：

 温度采样策略：首视频温度系数 $\scriptstyle \mathtt { T } = 0 . 8$ （确定性高），末视频 $\tau { = } 1 . 2$ （探索性强）  
 多样性掩码：限制同类型视频重复出现概率

# 2. 迭代偏好对齐（Iteative Preference Alignment, IPA）

![](images/7fe5e5b7e137e874039fc77f857b5394a3a3cf45578f560b02b819860de99b0f.jpg)

分两阶段优化生成质量：

 基础训练：最小化会话级 NTP (next token prediction)损失

$$
L _ {N T P} = - \sum_ {t = 1} ^ {T} \log P (x _ {t} | x _ {<   t})
$$

#  DPO 微调：

 奖励模型设计：多目标预测观看时长、完播率、点赞率等，结构采用 Self-Attention 融合会话特征  
 硬负采样：通过 Beam Search 生成候选，选择相似度 0.4-0.6 区间样本构建偏好对( $S _ { u } ^ { w } , S _ { u } ^ { l } )$   
 偏好优化公式：

$$
L _ {D P O} = - \log \mathfrak {Q} (\mathfrak {F} (\log \frac {\pi_ {\mathfrak {f}} (S ^ {w})}{\pi_ {\mathfrak {r e f}} (S ^ {w})} - \log \frac {\pi_ {\mathfrak {f}} (S ^ {l})}{\pi_ {\mathfrak {r e f}} (S ^ {l})}))
$$

# 三、工程优化策略

# 1. 训练体系

 混合精度训练：采用 bfloat16 格式，GradScaler 损失缩放系数初始值 8192  
 分阶段解冻：先训练语义 ID 层 解冻 MoE 层 联合优化 DPO 目标

# 2. 在线部署推理优化

 KV 缓存分块：内存占用降低 $6 3 \%$   
 MoE路由引擎：TensorRT 实现专用推理加速  
 动态早停机制：设置置信度阈值提前终止低质量候选

![](images/db7b356f5e23b89e5f259a47e8fb67fd21f67289cd1c480e01d73c981d3d41e3.jpg)  
Figure 3: Framework of Online Deployment of OneRec.

# 四、实验效果验证

Table 2: The absolute improvement of OneRec compared to the current multi-stage system in the online A/B testing setting.   

<table><tr><td>Model</td><td>Total Watch Time</td><td>Average View Duration</td></tr><tr><td>OneRec-0.1B</td><td>+0.57%</td><td>+4.26%</td></tr><tr><td>OneRec-1B</td><td>+1.21%</td><td>+5.01%</td></tr><tr><td>OneRec-1B+IPA</td><td>+1.68%</td><td>+6.56%</td></tr></table>

该模型在快手在线 AB 测试中，参数规模达 1B 时推理成本仅增加 $7 \%$ ，验证了工业级可行性。

当前局限主要在于低活跃用户场景表现不足，未来计划引入多模态特征增强冷启动

# 五、OneRec 与传统多阶段推荐对比

| 对比维度 | 传统多阶段推荐 | OneRec 生成式推荐 |
|---------|-------------|-----------------|
| 架构模式 | 召回→粗排→精排→重排级联 | 端到端 Encoder-Decoder |
| 候选集处理 | 各阶段独立筛选，信息逐级衰减 | 一次生成完整 Session |
| 语义 ID | 物品 ID 或 embedding 查表 | RQ-VAE 层次化离散编码 |
| 训练目标 | 各阶段独立优化（CTR/CVR等） | 会话级 NTP + DPO 偏好对齐 |
| 推理延迟 | 多阶段串行累加 | 单次前向 + 投机解码 |
| 系统复杂度 | 多模块维护成本高 | 统一模型，部署简洁 |
| 多样性控制 | 重排阶段后处理 | 解码时温度采样+多样性掩码 |

# 六、代码实现：语义 ID 编码与 Session 生成

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
class ResidualQuantizer(nn.Module):
    def __init__(self, input_dim, num_levels=3, codebook_size=4096):
        super().__init__()
        self.num_levels = num_levels
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, input_dim) * 0.1)
            for _ in range(num_levels)
        ])
        self.output_proj = nn.Linear(input_dim, codebook_size, bias=False)

    def quantize_level(self, residual, codebook):
        distances = torch.cdist(residual.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
        indices = distances.argmin(dim=-1)
        quantized = codebook[indices]
        return indices, quantized

    def forward(self, x):
        residual = x
        all_indices = []
        for level in range(self.num_levels):
            indices, quantized = self.quantize_level(residual, self.codebooks[level])
            all_indices.append(indices)
            residual = residual - quantized
        return torch.stack(all_indices, dim=-1)
class OneRecSessionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, num_experts=4, top_k=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.temperature_start = 0.8
        self.temperature_end = 1.2

    def forward(self, encoder_output, target_ids=None, session_length=8):
        batch_size = encoder_output.size(0)
        memory = encoder_output.unsqueeze(1)
        if target_ids is not None:
            embeds = self.token_embedding(target_ids)
            seq_len = target_ids.size(1)
            embeds = embeds + self.pos_encoding[:, :seq_len, :]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            output = self.decoder(embeds, memory.expand(-1, seq_len, -1), tgt_mask=causal_mask)
            logits = self.output_head(output)
            return logits
        else:
            generated = []
            current_token = torch.zeros(batch_size, 1, dtype=torch.long, device=encoder_output.device)
            for step in range(session_length * 3):
                temperature = self.temperature_start + (self.temperature_end - self.temperature_start) * step / (session_length * 3)
                embeds = self.token_embedding(current_token)
                seq_len = embeds.size(1)
                embeds = embeds + self.pos_encoding[:, :seq_len, :]
                causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=embeds.device)
                output = self.decoder(embeds, memory.expand(-1, seq_len, -1), tgt_mask=causal_mask)
                logits = self.output_head(output[:, -1:, :]) / temperature
                next_token = torch.multinomial(F.softmax(logits.squeeze(1), dim=-1), 1)
                generated.append(next_token)
                current_token = torch.cat([current_token, next_token], dim=1)
            return torch.cat(generated, dim=1)
d_model, vocab_size = 256, 8192
rq = ResidualQuantizer(d_model, num_levels=3, codebook_size=4096)
video_features = torch.randn(100, d_model)
semantic_ids = rq(video_features)
print(f"Semantic IDs shape: {semantic_ids.shape}")
encoder_out = torch.randn(2, d_model)
decoder = OneRecSessionDecoder(vocab_size, d_model=d_model, num_layers=4)
target = torch.randint(0, vocab_size, (2, 24))
logits = decoder(encoder_out, target_ids=target)
print(f"Training logits shape: {logits.shape}")
```

# 七、常见问题与易错点

| 问题 | 说明 | 建议 |
|------|------|------|
| 语义 ID 碰撞 | 不同视频映射到相同语义 ID，导致推荐重复 | 增加量化层级或扩大 codebook 大小，使用 Balanced K-means |
| 冷启动用户 | 低活跃用户历史行为少，生成质量差 | 引入多模态特征（封面图、标题）作为辅助输入 |
| MoE 负载不均衡 | 部分专家不被激活，资源浪费 | 增加负载均衡损失，使用 auxiliary loss 约束 |
| DPO 偏好对质量 | 硬负采样区间过窄导致偏好区分度不足 | 相似度区间 0.4-0.6 为经验值，需根据数据分布调整 |
| Session 连贯性 | 生成视频之间缺乏主题连贯性 | 引入会话级奖励模型，优化多视频协同吸引力 |
| 推理延迟 | 自回归生成多 token 延迟高 | 使用投机解码 + KV 缓存分块优化 |

# 八、学习总结

1. OneRec 通过端到端生成架构替代传统多阶段推荐，消除了级联信息损失，是推荐系统向生成式范式转变的代表作
2. 语义 ID 表征体系（RQ-VAE + Balanced K-means）解决了大规模物品库的离散化表征问题，内存占用降低 99%
3. IPA（迭代偏好对齐）结合 NTP 预训练和 DPO 微调，解决了生成式推荐的偏好优化问题
4. MoE 稀疏激活机制在 1B 参数规模下仅增加 7% 推理成本，验证了工业级部署可行性
5. 在线 AB 测试观看时长 +1.68%，证明了生成式推荐在短视频场景的实际价值

# 九、思考题

1. OneRec 的语义 ID 编码与传统 embedding 查表各有什么优劣？在什么场景下语义 ID 更有优势？
2. 如果将 OneRec 从短视频推广到电商推荐，需要做哪些核心修改？

**参考答案：**

1. 语义 ID 优势：内存友好（离散编码 vs 浮点向量）、层次化结构天然支持层次化检索、冷启动物品可通过内容特征直接获得 ID。劣势：量化损失导致细粒度特征丢失、需要额外的量化训练流程。语义 ID 更适合物品库极大（亿级以上）且更新频繁的场景，因为新增物品无需维护庞大的 embedding 表。

2. 核心修改包括：将视频多模态特征替换为商品图文特征；Session 定义从视频观看序列改为购物车/浏览序列；奖励模型从观看时长/完播率改为转化率/客单价；多样性控制从视频类型改为商品品类；Decoder 输出需要适配更结构化的商品 ID 体系。
