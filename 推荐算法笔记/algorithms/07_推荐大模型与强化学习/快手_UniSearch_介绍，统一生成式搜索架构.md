# 面试题：快手 UniSearch 介绍，统一生成式搜索架构

# 面试题：快手 UniSearch 介绍，统一生成式搜索架构

UniSearch 是快手在2025年提出的统一生成式搜索架构。它旨在用端到端的生成式模型，重构传统搜索"召回-粗排-精排"的级联链路，尤其在直播这类高动态场景中，实现更精准、更实时的搜索体验。

论文：https://arxiv.org/pdf/2509.06887  

<table><tr><td>方面</td><td>核心内容</td></tr><tr><td>提出背景</td><td>解决传统搜索在直播等高动态场景下语义理解不足、级联链路优化目标不一致、响应慢的问题。</td></tr><tr><td>核心创新</td><td>●真端到端联合训练：将视频编码器(Encoder)和搜索生成器(Generator)置于同一框架联合优化，解决目标不一致问题。
●残差渐进式语义ID：通过多层级语义ID（如动画类→儿童向→熊出没IP）模拟传统搜索的"召回→粗排→精排"漏斗结构。
●动态Trie树约束：实时维护在线内容的有效路径，确保生成结果必然存在，有效生成率达99.8%。
●在线偏好优化(SPO)：根据用户实时反馈持续优化模型，应用搜索业务感知的强化学习优化 Search Preference Optimization (SPO) 来进一步提升生成性能。</td></tr><tr><td>架构核心</td><td>●Search Generator(理解Query，生成语义ID序列)
●Video Encoder(将视频内容编码为语义ID)，通过联合损失函数统一优化。</td></tr><tr><td>关键原理</td><td>联合损失函数：
L = λ□·L Contrast (语义对齐) + λ□·L_Codebook (码本质量) + λ□·L_NTP (生成准确性)</td></tr><tr><td>效果</td><td>主要应用于快手直播搜索。上线后带来直播进间次数提升3.31%（近两年单实验最大收益），新用户贡献了近58.73%的增长，同时降低了系统资源消耗。</td></tr></table>

![](images/73df6c7ffcf1d0619fbc1e6af5b4cec2afc365cbef31368f436dbed1bfc93e42.jpg)

![](images/167e13ddd085315fbd42a21f726984a0362613abdf8fbb6f649f934baddb1a97e.jpg)

# 模型架构与原理

UniSearch 的架构设计，其核心在于将一个复杂的多阶段系统统一为一个可端到端学习和优化的整体。

# 核心组件：搜索生成器 $^ +$ 视频编码器协同

![](images/b8314073238aed4edfcb5937850dae4b731096cf6f7ada89b99f06db279d927c.jpg)  
(a) Model Architecture and Unified Pre-training

1 Search Generator（搜索生成器）：基于 Encoder-Decoder 的模型。  
 Encoder（编码器）：负责理解用户的搜索词（Query）、用户的历史行为以及搜索时的上下文信息，形成一个综合的意图表征。  
 Decoder（解码器）：以上述意图表征为条件，自回归地生成目标视频或直播间的语义 ID 序列。它不再是"检索"已有内容，而是直接"创造"一串指向理想结果的语义ID代码。

2 Video Encoder（视频编码器）：负责为平台上的每个视频/直播间创建独特的"身份证"——语义 ID。

 它利用 VQ-VAE 技术，将视频的标题、封面、画面内容等多模态信息编码成一个连续的向量，再通过"量化"过程，将其映射到一个离散的"码本"上，从而产生语义 ID。这相当于为视频内容生成了一个离散的、机器可读的语义 ID 摘要。

# 关键机制：动态Trie树+在线偏好优化

![](images/9a8f80676ace86e7fdc3448cb85c90b5da4ec6e82f41c66d2fef63b4be5b3173.jpg)  
(b) UniSearch Deployment and Online Post-training

# 1 动态 Trie 树约束：

 这是 UniSearch 能应用于直播等高动态场景的基石。直播内容瞬息万变，直播间随时开播、下播。  
 Trie 树是一种数据结构，可以实时监听所有在线直播间的语义 ID，形成一个不断更新的"有效路径地图"。  
 当 Search Generator 的 Decoder 一步步生成语义 ID 时，每一步都需向 Trie 树"咨询"下一步有哪些有效的选择。通过Beam Search算法，模型能在所有合法路径中找出最优的几个，从根本上杜绝了生成无效内容ID的可能性。

# 2 在线偏好优化（SPO）：

 UniSearch 不是一个静止的系统，而是一个能够持续进化的智能体。它会实时收集两方面的反馈信号：一是系统内部的评分（如精排模型的相关性判断），二是用户的真实行为（如点击、观看时长、关注等）。  
 这些信号被合成为一个奖励（Reward），然后通过类似于 GRPO 的强化学习算法，对模型参数进行微调。这意味着，如果系统发现用户普遍更喜欢"玩具开箱"类的直播，它就会在后续的生成中提高此类内容语义ID的生成概率。

# 联合损失函数：

UniSearch 的"真端到端"特性，数学上体现在其联合损失函数（Joint Loss Function）上：

$$
L = \lambda_ {1} \cdot L _ {\text {c o n t r a s t}} + \lambda_ {2} \cdot L _ {\text {c o d e b o o k}} + \lambda_ {3} \cdot L _ {\text {N T P}}
$$

这个公式将三个关键目标统一优化：

 $L _ { c o n t r a s t }$ （对比损失）：确保查询（Query）的语义和视频（Video）的语义在向量空间中对齐，即语义上相近的 Query 和Video，其向量表示也应接近。  
 $L _ { c o d e b o o k }$ （码本损失）：这是 VQ-VAE 特有的损失，用于优化码本的质量，让量化过程更精确，避免码本"坍塌"。  
 （Next Token 预估损失）：即生成模型标准的"下一个 Token 预测"损失，确保 Generator 能够准确地生成下一个语义 ID。

通过调整 $\lambda$ 超参数来平衡这三项损失，模型得以学习如何更好地理解内容、理解用户意图并生成准确的结果。

# 架构深度解析

## 残差渐进式语义 ID 的设计

UniSearch 的语义 ID 采用层级结构，例如一个直播间的语义 ID 可能是 `[动画类, 儿童向, 熊出没IP]`。这种设计有双重优势：

1. **模拟传统搜索的漏斗结构**：第一层相当于粗排（从所有直播中筛选类别），第二层相当于精排（在类别内筛选），第三层相当于最终排序（精准匹配）
2. **提高生成效率**：Trie 树的层级越深，可选的分支越少，Beam Search 的效率越高

语义 ID 的生成过程使用残差量化（Residual Quantization）：

$$\mathbf{z} = \text{Encode}(x), \quad \text{ID}_1 = \text{Q}_1(\mathbf{z}), \quad \text{ID}_2 = \text{Q}_2(\mathbf{z} - \mathbf{c}_{\text{ID}_1}), \quad \ldots$$

每一层量化残差（前一层无法编码的细节），逐层逼近原始向量。

## VQ-VAE 的量化过程

VQ-VAE 将连续向量 $\mathbf{z}$ 映射到离散码本 $\mathbf{C} = \{\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_K\}$ 中最近的码字：

$$\text{ID} = \arg\min_k \|\mathbf{z} - \mathbf{c}_k\|^2$$

码本损失包含两部分：

$$L_{\text{codebook}} = \|\text{sg}(\mathbf{z}) - \mathbf{c}\|^2 + \beta \|\mathbf{z} - \text{sg}(\mathbf{c})\|^2$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient 操作，$\beta$ 是承诺损失权重。第一项推动码字靠近编码器输出，第二项推动编码器输出靠近码字。

## 代码实现

### 核心架构简化实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, code_dim, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, code_dim)
        self.beta = beta

    def forward(self, z):
        z_flat = z.reshape(-1, z.shape[-1])
        dist = (z_flat.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)
        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices).reshape(z.shape)
        commitment_loss = F.mse_loss(z, z_q.detach()) * self.beta
        codebook_loss = F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()
        return z_q, indices.reshape(z.shape[:-1]), commitment_loss + codebook_loss


class ResidualQuantizer(nn.Module):
    def __init__(self, n_levels, n_codes, code_dim):
        super().__init__()
        self.levels = nn.ModuleList([
            VectorQuantizer(n_codes, code_dim) for _ in range(n_levels)
        ])
        self.n_levels = n_levels

    def forward(self, z):
        residual = z
        all_ids = []
        total_loss = 0
        for level in self.levels:
            z_q, ids, loss = level(residual)
            all_ids.append(ids)
            total_loss += loss
            residual = residual - z_q
        return torch.stack(all_ids, dim=-1), total_loss


class VideoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_levels=3, n_codes=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.quantizer = ResidualQuantizer(n_levels, n_codes, hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        semantic_ids, vq_loss = self.quantizer(z)
        return z, semantic_ids, vq_loss


class SearchGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, n_semantic_levels, n_codes):
        super().__init__()
        self.query_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True),
            num_layers=n_layers
        )
        self.id_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True),
            num_layers=n_layers
        )
        self.semantic_embeddings = nn.ModuleList([
            nn.Embedding(n_codes, d_model) for _ in range(n_semantic_levels)
        )
        self.query_embedding = nn.Embedding(vocab_size, d_model)
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, n_codes) for _ in range(n_semantic_levels)
        ])
        self.n_levels = n_semantic_levels

    def forward(self, query_ids, target_ids=None):
        query_emb = self.query_embedding(query_ids)
        query_repr = self.query_encoder(query_emb)
        if target_ids is not None:
            target_emb = torch.zeros(query_repr.size(0), self.n_levels, query_repr.size(-1), device=query_repr.device)
            for level in range(self.n_levels):
                target_emb[:, level] = self.semantic_embeddings[level](target_ids[:, level])
            causal_mask = nn.Transformer.generate_square_subsequent_mask(self.n_levels, device=query_repr.device)
            decoded = self.id_decoder(target_emb, query_repr, tgt_mask=causal_mask)
            logits = []
            for level in range(self.n_levels):
                logits.append(self.output_heads[level](decoded[:, level]))
            return logits
        return self._generate(query_repr)

    def _generate(self, query_repr, beam_size=5):
        batch_size = query_repr.size(0)
        current_emb = torch.zeros(batch_size, 1, query_repr.size(-1), device=query_repr.device)
        generated_ids = []
        for level in range(self.n_levels):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(level + 1, device=query_repr.device)
            if level > 0:
                level_embs = [self.semantic_embeddings[l](generated_ids[l]) for l in range(level)]
                current_emb = torch.stack(level_embs, dim=1)
            decoded = self.id_decoder(current_emb, query_repr, tgt_mask=tgt_mask)
            logit = self.output_heads[level](decoded[:, -1])
            next_id = logit.argmax(dim=-1)
            generated_ids.append(next_id)
        return torch.stack(generated_ids, dim=-1)


class UniSearch(nn.Module):
    def __init__(self, vocab_size, video_input_dim, d_model=256, n_heads=8,
                 n_layers=4, n_semantic_levels=3, n_codes=1024):
        super().__init__()
        self.video_encoder = VideoEncoder(video_input_dim, d_model, n_semantic_levels, n_codes)
        self.search_generator = SearchGenerator(vocab_size, d_model, n_heads, n_layers, n_semantic_levels, n_codes)
        self.temp = nn.Parameter(torch.tensor(0.07))

    def contrastive_loss(self, query_repr, video_repr):
        query_repr = F.normalize(query_repr, dim=-1)
        video_repr = F.normalize(video_repr, dim=-1)
        logits = torch.matmul(query_repr, video_repr.T) / self.temp
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_q2v = F.cross_entropy(logits, labels)
        loss_v2q = F.cross_entropy(logits.T, labels)
        return (loss_q2v + loss_v2q) / 2

    def forward(self, query_ids, video_features, target_ids=None):
        video_repr, semantic_ids, vq_loss = self.video_encoder(video_features)
        if target_ids is not None:
            logits = self.search_generator(query_ids, target_ids)
            ntp_loss = sum(F.cross_entropy(logits[l], target_ids[:, l]) for l in range(len(logits)))
        else:
            ntp_loss = torch.tensor(0.0, device=query_ids.device)
        return vq_loss, ntp_loss


model = UniSearch(vocab_size=5000, video_input_dim=128)
query = torch.randint(0, 5000, (4, 8))
video_feat = torch.randn(4, 128)
target = torch.randint(0, 1024, (4, 3))
vq_loss, ntp_loss = model(query, video_feat, target)
print(f"VQ Loss: {vq_loss.item():.4f}, NTP Loss: {ntp_loss.item():.4f}")
```

### 动态 Trie 树实现

```python
class DynamicTrie:
    def __init__(self, n_levels, n_codes):
        self.n_levels = n_levels
        self.n_codes = n_codes
        self.trie = {}
        self.active_ids = set()

    def insert(self, semantic_ids):
        node = self.trie
        self.active_ids.add(tuple(semantic_ids))
        for level, code in enumerate(semantic_ids):
            if code not in node:
                node[code] = {"children": {}, "is_end": level == len(semantic_ids) - 1}
            node = node[code]["children"]
            if level == len(semantic_ids) - 1:
                node.get("_end", True)

    def remove(self, semantic_ids):
        self.active_ids.discard(tuple(semantic_ids))

    def get_valid_next(self, prefix):
        node = self.trie
        for code in prefix:
            if code not in node:
                return []
            node = node[code]["children"]
        return list(node.keys())

    def beam_search(self, generator_fn, beam_size=5):
        beams = [([], 0.0)]
        for level in range(self.n_levels):
            candidates = []
            for prefix, score in beams:
                valid_next = self.get_valid_next(prefix)
                if not valid_next:
                    candidates.append((prefix, score))
                    continue
                for code in valid_next:
                    new_prefix = prefix + [code]
                    new_score = score + generator_fn(level, code)
                    candidates.append((new_prefix, new_score))
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        return beams


trie = DynamicTrie(n_levels=3, n_codes=1024)
trie.insert([10, 200, 3000])
trie.insert([10, 200, 3001])
trie.insert([10, 201, 3000])
trie.insert([15, 300, 4000])
print(f"前缀 [10, 200] 的有效后续: {trie.get_valid_next([10, 200])}")
```

## 与传统搜索系统对比

| 维度 | 传统搜索 | UniSearch |
|------|---------|-----------|
| 架构 | 召回→粗排→精排（级联） | 端到端生成式 |
| 优化目标 | 各阶段独立优化 | 联合优化 |
| 语义理解 | 依赖各阶段模型 | 统一语义空间 |
| 实时性 | 需要更新各阶段索引 | Trie 树动态维护 |
| 新内容处理 | 需要入索引延迟 | 实时编码为语义 ID |
| 部署复杂度 | 多服务级联 | 单服务（但模型更大） |
| 可扩展性 | 受级联瓶颈限制 | 模型缩放定律 |
| 冷启动 | 依赖物品特征 | 语义 ID 自动编码 |

## 部署注意事项

1. **Trie 树的实时性维护**：直播间的开播/下播需要在秒级更新 Trie 树，否则会生成已失效的语义 ID
2. **Beam Search 的延迟控制**：Beam Size 越大结果越好但延迟越高，需要根据 QPS 要求权衡
3. **码本的容量规划**：码本大小决定了可区分的语义粒度，过小会导致语义碰撞，过大会导致码本利用率低
4. **在线学习的安全性**：SPO 的在线更新需要做好灰度和回滚机制，防止模型退化

## 常见问题

1. **Q: 语义 ID 的层级深度如何确定？**
   A: 层级深度与码本大小的乘积决定了可表示的语义空间大小。例如 3 层、每层 1024 个码字，理论上可表示 $1024^3 \approx 10^9$ 个不同的语义 ID。实际中需要根据平台内容量确定。

2. **Q: 如果 Trie 树为空怎么办？**
   A: 这通常发生在系统初始化或全部直播间下播时。此时应降级到传统的关键词搜索或返回热门直播。

3. **Q: UniSearch 能否用于非直播场景？**
   A: 可以。对于短视频等相对静态的内容，Trie 树的更新频率可以降低，但核心架构不变。不过对于静态内容，传统搜索的级联架构可能仍然足够高效。

# 7.2 大模型面试题
