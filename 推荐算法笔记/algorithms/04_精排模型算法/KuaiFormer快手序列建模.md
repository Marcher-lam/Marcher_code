# KuaiFormer 快手基于 Transformer 的召回模型

## 1. 算法基础认知

KuaiFormer 是快手提出的基于 Transformer 架构的用户行为序列建模模型，用于推荐系统中的召回阶段。其核心创新在于将推荐问题重新定义为 **Next Action Prediction（下一动作预测）** 序列生成任务，使召回目标与排序目标保持一致，同时通过层次化压缩和多兴趣提取实现高效的工业级部署。

## 2. 详细原理

### 2.1 Next Action Prediction 范式

传统召回方法（如双塔、DSSM）将用户兴趣压缩为单个向量，存在信息瓶颈。KuaiFormer 借鉴语言建模思路，将用户历史行为视为序列，预测下一个交互 item：

$$P(item_{t+1} | item_1, item_2, \ldots, item_t)$$

这种范式天然适合 Transformer 的自回归/自注意力结构，且目标函数与精排 CTR 优化目标一致。

### 2.2 层次化序列压缩

用户行为序列可能长达数千，直接输入 Transformer 计算量巨大（$O(n^2)$）。KuaiFormer 提出层次化压缩策略：

1. **时间段划分**：将序列按时间分为"早期、中期、近期"三段
2. **粒度分组**：早期用粗粒度（64 个 token 一组），近期用细粒度（16 个 token 一组）
3. **组内聚合**：每组通过平均池化或注意力池化压缩为单个 token

计算资源降低至原方案的 **10%**，同时保留近期行为的细粒度信息。

### 2.3 多兴趣提取

引入多个可学习的 Query Token（类似 BERT 的 [CLS]），通过交叉注意力从压缩后的序列中提取多维兴趣：

$$Q_{interest} = [q_1, q_2, \ldots, q_K]$$
$$Attention(Q_{interest}, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

每个 Query Token 负责捕获用户某一维度的兴趣（如"科技"、"娱乐"、"体育"），形成多兴趣表征矩阵。

### 2.4 高效训练优化

- **In-batch Softmax**：用批次内其他样本作为负样本，替代全局 Softmax，训练效率提升数十倍
- **LogQ 校正**：修正采样偏差，$score_{corrected} = score - \log(freq_i)$，高频 item 被适当惩罚
- **标签平滑**：将硬标签 $y \in \{0,1\}$ 软化为 $\tilde{y} = (1-\epsilon)y + \epsilon/K$，处理行为模糊性

## 3. 模型架构

```
用户行为序列 [i1, i2, ..., iN]
        │
   层次化压缩模块
   ┌─────┼─────┐
  早期  中期  近期
  (64)  (32)  (16)
        │
  压缩 Token 序列
        │
   Transformer Encoder (L 层)
        │
   多 Query Token 交叉注意力
   ┌───┼───┐
  Q1  Q2  Q3  (多维兴趣)
        │
   兴趣表征矩阵 [K × d]
        │
   In-batch Softmax 匹配
```

## 4. 训练过程

1. 将用户行为序列按时间排序并分段压缩
2. 压缩后序列输入 Transformer Encoder
3. 多 Query Token 通过交叉注意力生成多维兴趣向量
4. 正样本 item 与兴趣向量计算点积得分
5. In-batch Softmax + LogQ 校正计算损失
6. 反向传播更新所有参数

损失函数：

$$\mathcal{L} = -\log \frac{\exp(s(u, i^+) / \tau)}{\sum_{j \in \mathcal{B}} \exp(s(u, j) / \tau)}$$

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 短视频召回 | 快手核心业务，海量用户行为序列 |
| 电商多兴趣召回 | 用户同时关注多个品类 |
| 新闻推荐 | 兴趣多样且随时间变化 |
| 音乐播放列表 | 用户听歌行为具有时序性 |

## 6. 优缺点分析

**优点**：
- 召回与排序目标一致，减少链路损耗
- 多兴趣表征缓解单向量信息瓶颈
- 层次压缩兼顾效率与精度
- 自注意力捕获长距离依赖

**缺点**：
- Transformer 推理延迟高于双塔模型
- 层次压缩可能丢失早期重要信息
- 多兴趣数量 K 需手动调参
- 训练资源消耗较大

## 7. 与相关方法对比

| 方法 | 序列建模方式 | 兴趣维度 | 计算效率 | 长序列支持 |
|------|------------|---------|---------|-----------|
| DSSM | 无（静态向量） | 单维 | 极高 | 不适用 |
| SASRec | 自注意力 | 单维 | 中 | 有限（~500） |
| BST | Transformer+位置编码 | 单维 | 中 | 有限 |
| MIND | 动态路由 | 多维 | 高 | 中等 |
| KuaiFormer | 层次化Transformer | 多维 | 中高 | 强（10K+） |

## 8. PyTorch 代码实现

```python
import torch
import torch.nn as nn
import math

class HierarchicalCompression(nn.Module):
    def __init__(self, embed_dim, early_grain=64, mid_grain=32, recent_grain=16):
        super().__init__()
        self.early_grain = early_grain
        self.mid_grain = mid_grain
        self.recent_grain = recent_grain
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, seq_emb, seq_len):
        early_end = seq_len * 2 // 5
        mid_end = seq_len * 4 // 5
        recent = seq_emb[mid_end:]
        
        def compress(tokens, grain):
            if tokens.size(0) == 0:
                return tokens.unsqueeze(0)
            pad_len = (grain - tokens.size(0) % grain) % grain
            if pad_len > 0:
                tokens = torch.cat([tokens, tokens[-1:].expand(pad_len, -1)])
            n_groups = tokens.size(0) // grain
            return tokens.view(n_groups, grain, -1).mean(dim=1)
        
        early_comp = compress(seq_emb[:early_end], self.early_grain)
        mid_comp = compress(seq_emb[early_end:mid_end], self.mid_grain)
        compressed = torch.cat([early_comp, mid_comp, recent], dim=0)
        return self.proj(compressed)

class KuaiFormer(nn.Module):
    def __init__(self, num_items, embed_dim=64, num_heads=4, 
                 num_layers=2, num_interests=3, max_seq_len=200):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.compression = HierarchicalCompression(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.interest_queries = nn.Parameter(torch.randn(num_interests, embed_dim))
        self.temperature = nn.Parameter(torch.ones(1))
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, seq_ids, seq_len):
        seq_emb = self.item_embedding(seq_ids)
        compressed = self.compression(seq_emb, seq_len)
        compressed = compressed.unsqueeze(0)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            compressed.size(1)
        ).to(compressed.device)
        encoded = self.transformer(compressed, mask=causal_mask)
        
        queries = self.interest_queries.unsqueeze(0).expand(encoded.size(0), -1, -1)
        scores = torch.matmul(queries, encoded.transpose(-2, -1)) / math.sqrt(encoded.size(-1))
        attn_weights = torch.softmax(scores, dim=-1)
        interest_emb = torch.matmul(attn_weights, encoded)
        interest_emb = self.layer_norm(interest_emb)
        return interest_emb
    
    def compute_loss(self, interest_emb, target_item_emb, batch_items_emb):
        scores = torch.matmul(interest_emb, target_item_emb.unsqueeze(-1)).squeeze(-1)
        pos_score = scores.max(dim=1).values
        neg_scores = torch.matmul(interest_emb, batch_items_emb.T)
        neg_scores = neg_scores.max(dim=1).values
        logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return nn.functional.cross_entropy(logits, labels)
```

## 9. 可视化与结果理解

- **注意力热力图**：可视化各 Query Token 对序列的关注分布，验证多兴趣是否分化
- **压缩前后对比**：观察层次化压缩是否保留关键行为模式
- **兴趣聚类**：将多兴趣向量降维可视化，检查是否形成语义清晰的兴趣簇

## 10. 常见问题与易错点

1. **序列填充处理**：压缩时需忽略 padding 位置，否则引入噪声
2. **兴趣数量选择**：K 过小退化为单兴趣，K 过大增加推理成本，一般 3-8
3. **In-batch 负采样偏差**：热门 item 被过度当作负样本，需 LogQ 修正
4. **冷启动问题**：新用户行为序列过短，层次压缩失效，需设最小序列长度阈值

## 11. 学习总结

KuaiFormer 的核心贡献在于将推荐召回重新定义为序列预测任务，通过层次化压缩和多兴趣提取实现高效的 Transformer 召回。其设计思想——"目标一致性"和"多粒度建模"——对推荐系统架构设计有重要启发。

## 12. 学习路径建议

- **前置知识**：Transformer、注意力机制、推荐系统召回基础
- **进阶方向**：SASRec、BST、TIGER（生成式召回）
- **推荐论文**：KuaiFormer (KDD 2024)、SASRec (ICDM 2018)、MIND (CIKM 2019)
