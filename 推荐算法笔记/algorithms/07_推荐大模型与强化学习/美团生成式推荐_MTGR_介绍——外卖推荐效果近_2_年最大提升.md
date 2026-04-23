# 面试题：美团生成式推荐 MTGR 介绍——外卖推荐效果近 2 年最大提升

# 面试题：美团生成式推荐 MTGR 介绍——外卖推荐效果近 2 年最大提升

美团 MTGR（Meituan Generative Recommendation）是一个工业级的生成式推荐框架，它成功地将 LLM 的缩放定律（ScalingLaw）应用于推荐系统，在美团核心的外卖推荐场景中取得了显著的效果提升和成本优化。

参考链接：https://tech.meituan.com/2025/05/19/meituan-generative-recommendation.html

<table><tr><td>特性维度</td><td>具体说明</td></tr><tr><td>提出背景</td><td>●传统DLRM模型在外卖场景遭遇效果瓶颈；●生成式推荐（GR）虽具扩展性但舍弃关键交叉特征。</td></tr><tr><td>核心目标</td><td>构建混合架构，兼顾生成式推荐的扩展性优势和传统DLRM的交叉特征优势</td></tr><tr><td>关键创新</td><td>数据组织方式：●用户粒度样本压缩●保留全部交叉特征模型架构组件：●Group LayerNorm●动态混合掩码策略</td></tr><tr><td>核心架构</td><td>基于改进的HSTU（分层序列转导单元）架构，使用Transformer编码器统一建模多种类型特征</td></tr><tr><td>业务收益</td><td>离线CTCVR GAUC提升+2.88pp，线上订单量+1.22%，PV_CTR+1.31%，推理资源节省12%</td></tr></table>

# ① 提出背景与待解难题

美团外卖推荐场景经过近十年的迭代，基于传统 DLRM模型进一步提升转化率变得十分困难，主要面临两大挑战 ：

 扩展瓶颈：传统方法通过增加模型复杂度来提升效果，但存在天花板。其推理成本会随候选物品数量线性增长，扩展性差。  
 特征取舍困境：纯粹的生成式推荐模型（如 Meta 的 GR）为追求扩展性而舍弃了人工交叉特征，但这对于外卖这类强依赖"用户-商家"交叉信息（如距离、历史点击率）的业务会造成严重的性能损失，且无法通过单纯扩大模型规模来弥补。

MTGR 的目标正是在于解决这一困境，探索一条融合之路。

# ① 核心模型架构

# 数据组织与特征处理

MTGR 对齐了 DLRM 的全部特征体系，包括用户画像、上下文环境、用户历史行为序列以及候选物品特征。其核心创新在于数据组织方式：

![](images/3165bb46ee805270d2d2893373fed6c7faa3a858d641c32ba939d6ef4b19e7e8.jpg)

 用户粒度样本压缩：传统 DLRM 为每个（用户, 候选物品）对创建一行样本，导致同一用户特征被重复编码。MTGR 将同一用户在一个时间窗口内的所有曝光候选物品聚合为一个样本，极大减少了数据冗余，为后续的计算复用打下基础。  
 保留交叉特征：MTGR 将交叉特征作为候选物品的一部分进行嵌入和编码，确保了关键信息不丢失。消融实验表明，移除交叉特征会导致性能大幅下降，甚至抵消模型扩大带来的收益。

![](images/006d33f24e045edc4b61af9ebeea3d84d50fa61bd2e5f2027b4b38a0bf0e11a0.jpg)

![](images/77864fb9f9e7e2c4884214090f2b2345c1b93beb715044cd6f9271ccc5ca9b20.jpg)

# 模型架构与关键组件

MTGR 采用改进的 HSTU 架构作为主干网络，并引入了两项核心创新 ：

 Group LayerNorm：针对推荐数据中不同特征（用户画像、历史行为、候选物品）语义空间不同的问题，MTGR 对不同类型的特征分组进行 LayerNorm，使用不同的参数进行归一化，这促进了不同语义空间下 Token 的对齐，提升了模型的表示能力。  
动态混合掩码策略：为防止信息泄露，MTGR 设计了精细的掩码策略。

 静态历史特征（用户画像、长期行为序列）：全局可见。  
 当日实时行为：遵循因果关系，每个 Token 仅对出现在其之后的 Token 可见。  
 候选物品：仅对自身可见。

这种策略确保了在复杂的外卖 Feed 流场景下建模的因果正确性。

# ① 工程实现：训练与推理优化

为了支撑千亿参数级别模型的大规模分布式训练与高效部署，美团构建了 MTGR-Training 和 MTGR-Inference 引擎。

 MTGR-Training 训练引擎：基于 TorchRec 构建，并进行了多项深度优化。

 动态哈希表：解决了流式训练中不断涌现的新用户和新物品的嵌入分配问题，避免了固定嵌入表的内存浪费或溢出风险。  
 变长序列负载均衡：根据用户序列的实际长度动态调整每个 GPU 的 batch size，使计算负载均衡，避免了因序列长尾分布导致的计算等待。  
 定制化计算内核：借鉴 Flash-Attention 思想，手写了融合的 HSTU 计算内核，支持变长序列输入且无需 padding，显著提升了计算效率。

 MTGR-Inference 推理引擎：尽管单样本计算量（FLOPs）增加了 65 倍，但凭借用户粒度样本压缩带来的计算复用，MTGR 通过 TensorRT 图优化、FP16 量化、合并传输（H2D）等技术，最终实现了推理资源节省 $12 \%$ 的效果。

# ① 实际业务收益

MTGR 在美团业务中取得了显著成效，并验证了推荐系统中的缩放定律 ：

 效果提升：离线核心指标CTCVR GAUC 提升2.88个百分点。在线 A/B测试显示，外卖首页列表订单量提升 $1 . 2 2 \%$ ，这是近两年单次迭代的最大收益。  
 验证缩放定律：通过设计 Small、Middle、Large 三种规模的模型，MTGR 清晰地展示了随着模型参数和计算量的增加，推荐效果持续提升的幂律关系，为后续发展指明了方向。

# MTGR 架构深度解析

## 与传统 DLRM 的数据组织对比

传统 DLRM 的数据组织方式：

```
用户A, 商品1, 距离=2km, 历史CTR=0.05 → 样本1
用户A, 商品2, 距离=3km, 历史CTR=0.03 → 样本2  
用户A, 商品3, 距离=1km, 历史CTR=0.08 → 样本3
```

MTGR 的数据组织方式：

```
用户A + [商品1(距离=2km, CTR=0.05), 商品2(距离=3km, CTR=0.03), 商品3(距离=1km, CTR=0.08)] → 单个样本
```

这种组织方式的好处：
1. 用户画像特征只编码一次，通过自注意力机制与所有候选物品交互
2. 候选物品之间也可以通过注意力机制互相比较，相当于隐式地做了 list-wise 排序
3. 训练数据量压缩为原来的 $1/N$（$N$ 为平均候选物品数），但信息量不减少

## Group LayerNorm 的数学表达

标准 LayerNorm 对所有 token 使用相同的归一化参数：$\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$

Group LayerNorm 对不同类型的 token 使用不同的归一化参数：

$$\text{GroupLN}(x_i) = \frac{x_i - \mu_{g(i)}}{\sigma_{g(i)}} \cdot \gamma_{g(i)} + \beta_{g(i)}$$

其中 $g(i)$ 表示 token $i$ 所属的组（用户画像组、历史行为组、候选物品组）。这样做的原因是：

- 用户画像 token 的分布与候选物品 token 的分布差异很大
- 如果使用统一的 LayerNorm，分布差异大的 token 会被过度压缩或拉伸
- 分组归一化保持了各组内部分布的一致性

## 动态混合掩码策略详解

MTGR 的掩码策略可以形式化为一个注意力掩码矩阵 $M$，其中 $M_{ij} \in \{0, 1\}$ 表示 token $j$ 是否对 token $i$ 可见：

| 特征类型 | 对用户画像可见 | 对历史行为可见 | 对当日行为可见 | 对候选物品可见 |
|---------|-------------|-------------|-------------|-------------|
| 用户画像 | 1 | 1 | 1 | 1 |
| 历史行为 | 1 | 因果掩码 | 1 | 1 |
| 当日行为 | 1 | 1 | 因果掩码 | 1 |
| 候选物品 i | 1 | 1 | 1 | 仅物品 i |

## 代码实现

### MTGR 核心架构简化实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupLayerNorm(nn.Module):
    def __init__(self, d_model, n_groups):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_groups)])

    def forward(self, x, group_ids):
        output = torch.zeros_like(x)
        for g_id in range(len(self.norms)):
            mask = (group_ids == g_id)
            if mask.any():
                output[mask] = self.norms[g_id](x[mask])
        return output


class MTGRBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_groups=3):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.group_norm_attn = GroupLayerNorm(d_model, n_groups)
        self.group_norm_ffn = GroupLayerNorm(d_model, n_groups)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, group_ids, attn_mask=None):
        normed = self.group_norm_attn(x, group_ids)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out
        normed = self.group_norm_ffn(x, group_ids)
        x = x + self.ffn(normed)
        return x


class MTGRModel(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_layers, n_candidates=10):
        super().__init__()
        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(1000, d_model) for _ in range(n_features)
        ])
        self.type_embedding = nn.Embedding(3, d_model)
        self.blocks = nn.ModuleList([
            MTGRBlock(d_model, n_heads, n_groups=3) for _ in range(n_layers)
        ])
        self.score_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.n_candidates = n_candidates

    def build_mask(self, seq_len, group_ids):
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                if group_ids[j] == 2:
                    if i != j:
                        mask[i, j] = 0
        return mask.unsqueeze(0)

    def forward(self, feature_ids, group_ids):
        batch_size = feature_ids.size(0)
        embeddings = []
        for i, emb_layer in enumerate(self.feature_embeddings):
            embeddings.append(emb_layer(feature_ids[:, i]))
        x = torch.stack(embeddings, dim=1)
        x = x + self.type_embedding(group_ids.unsqueeze(0).expand(batch_size, -1))
        attn_mask = self.build_mask(x.size(1), group_ids)
        for block in self.blocks:
            x = block(x, group_ids, attn_mask=attn_mask)
        candidate_mask = (group_ids == 2)
        candidate_repr = x[:, candidate_mask, :]
        scores = self.score_head(candidate_repr).squeeze(-1)
        return scores


model = MTGRModel(n_features=20, d_model=64, n_heads=4, n_layers=2)
feature_ids = torch.randint(0, 1000, (4, 20))
group_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2])
scores = model(feature_ids, group_ids)
print(f"Output scores shape: {scores.shape}")
```

## 与其他生成式推荐模型对比

| 维度 | MTGR（美团） | GR（Meta） | TIGER | UniSearch（快手） |
|------|-------------|-----------|-------|-----------------|
| 核心架构 | 改进 HSTU | HSTU | Transformer + 语义ID | Encoder-Decoder |
| 是否保留交叉特征 | 是（关键创新） | 否 | 否 | 否 |
| 数据组织 | 用户粒度压缩 | 用户粒度压缩 | 标准序列 | 标准序列 |
| 语义ID | 无 | 无 | 多层级RQ-VAE | 残差渐进式VQ-VAE |
| 推理优化 | 计算复用+TensorRT | M-FALCON | 标准推理 | Trie树约束 |
| 适用场景 | 外卖/本地生活 | 社交媒体推荐 | 通用推荐 | 搜索/直播 |
| 缩放定律验证 | 是（3种规模） | 是 | 部分 | 部分 |

## 关键经验总结

1. **交叉特征不可轻易丢弃**：在外卖等本地生活场景中，用户-商家距离、历史点击率等交叉特征对预测至关重要。MTGR 通过将交叉特征作为候选物品 token 的一部分来保留这些信息
2. **计算复用是推理效率的关键**：虽然单样本计算量增加了 65 倍，但用户粒度样本压缩使得同一用户的多次请求只需编码一次用户特征
3. **Group LayerNorm 是必要的**：实验表明，使用标准 LayerNorm 替换 Group LayerNorm 会导致 GAUC 下降约 0.5 个百分点
4. **掩码策略的因果性**：候选物品之间的隔离掩码防止了信息泄露，确保了在线推理时的公平性

## 常见问题

1. **Q: MTGR 和 Meta 的 GR 有什么核心区别？**
   A: 最核心的区别是 MTGR 保留了交叉特征。Meta 的 GR 为了追求扩展性放弃了交叉特征，但这在外卖场景中损失过大。MTGR 通过将交叉特征编码到候选物品 token 中，实现了两者的融合。

2. **Q: 用户粒度样本压缩会不会导致训练数据量减少？**
   A: 从样本行数来看确实减少了，但信息量没有减少。同一用户的所有候选物品在同一个样本中通过注意力机制交互，实际上信息利用更充分了。

3. **Q: Group LayerNorm 和标准 LayerNorm 的参数量差异有多大？**
   A: Group LayerNorm 的参数量是标准 LayerNorm 的 $G$ 倍（$G$ 为组数，通常为 3），但 LayerNorm 参数本身很小（$2d$），所以绝对增加量可以忽略不计。

## 学习总结

MTGR 是生成式推荐在工业界落地的标杆案例。它的核心贡献不在于提出全新的模型架构，而在于解决了生成式推荐在实际业务中面临的关键问题：如何在享受扩展性优势的同时不丢失业务关键的交叉特征。其用户粒度样本压缩、Group LayerNorm、动态混合掩码三大创新点，都是针对外卖推荐场景的具体痛点设计的，体现了"工程创新服务于业务需求"的思路。
