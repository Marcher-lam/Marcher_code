# TIN（Temporal Interest Network）腾讯时间兴趣网络

## 1. 算法基础认知

TIN（Temporal Interest Network）是腾讯提出的用于用户行为序列建模的推荐模型。核心创新在于提出**语义-时间四向交互**机制，通过联合建模行为与目标的语义关联及动态时间衰减，精准捕获用户兴趣的时序演化规律。该模型在微信朋友圈广告场景落地，支持高达 54,000 长度的行为序列。

## 2. 详细原理

### 2.1 动机

现有序列模型（DIN、DIEN、BST 等）将时间信息简单拼接或加性融合，未充分挖掘"行为语义 × 目标语义 × 行为时间 × 目标时间"的高阶交叉信息。TIN 认为用户兴趣的演化具有以下特征：

1. **时间衰减**：近期行为比远期行为更重要，但衰减速率因语义类别而异
2. **目标感知**：时间衰减应与目标 item 相关（同类衰减慢，异类衰减快）
3. **语义-时间耦合**：时间效果不能脱离语义独立建模

### 2.2 四向交互

TIN 的核心是同时捕捉四个维度的交叉信号：

$$\text{行为语义} \times \text{目标语义} \times \text{行为时间} \times \text{目标时间}$$

传统方法仅建模前两个（语义交互），TIN 引入时间维度形成四向交叉。

## 3. 关键模块

### 3.1 目标感知时间编码（TTE）

TIN 提出两种时间编码方式，分别捕获不同粒度的时间信息：

**TTE-P（相对位置编码）**：

$$TTE\text{-}P_i = PE(pos_i)$$

编码行为在序列中的相对位置，捕获兴趣的序数演化。

**TTE-T（时间间隔编码）**：

$$TTE\text{-}T_i = TE(\Delta t_i) = TE(t_{target} - t_i)$$

编码行为发生时间与当前目标的时间间隔，捕获绝对时间衰减。

两种编码都经过**目标感知变换**：

$$\tilde{T}_i = MLP(T_i \oplus e_{target})$$

其中 $\oplus$ 表示拼接，$e_{target}$ 是目标 item 的嵌入。

### 3.2 目标感知注意力（TA）

在缩放点积注意力中融入时间编码：

$$\alpha_i = \frac{(W_Q e_{target})^T \cdot (W_K (e_i + \tilde{T}_i))}{\sqrt{d_k}}$$

时间编码直接参与注意力权重的计算，而非仅作为加性偏置。

### 3.3 目标感知表示（TR）

通过元素级乘法融合语义和时间信号：

$$\hat{e}_i = e_i \odot \sigma(W_{tr} \tilde{T}_i)$$

时间编码通过门控机制调制行为嵌入的每个维度。

### 3.4 综合输出

$$Output = \sum_i \alpha_i \cdot \hat{e}_i$$

## 4. 数学推导

### 4.1 时间衰减函数推导

假设用户兴趣按指数衰减：

$$I(t) = I_0 \cdot e^{-\lambda \cdot \Delta t}$$

不同类别有不同的衰减率 $\lambda_c$。TTE-T 通过学习时间间隔的嵌入隐式建模这种类别依赖的指数衰减。

### 4.2 四向交互展开

以 $\alpha_i$ 为例，展开四向交互：

$$\alpha_i \propto f(\underbrace{e_{target}}_{\text{目标语义}}, \underbrace{e_i}_{\text{行为语义}}, \underbrace{t_{target}}_{\text{目标时间}}, \underbrace{t_i}_{\text{行为时间}})$$

具体为：

$$\alpha_i = softmax\left(\frac{Q(e_{target})^T \cdot K(e_i, TTE(t_i, t_{target}))}{\sqrt{d}}\right)$$

其中 TTE 编码同时依赖行为时间和目标时间。

## 5. 训练过程

1. 将用户行为序列编码为嵌入向量 $[e_1, e_2, \ldots, e_n]$
2. 计算每个行为与目标 item 的 TTE-P 和 TTE-T 编码
3. 通过目标感知变换融合时间和目标信息
4. 计算 TA 注意力权重和 TR 门控表示
5. 加权聚合得到用户兴趣向量
6. 兴趣向量与目标 item 嵌入交互，预测 CTR
7. Binary Cross-Entropy 损失优化

## 6. 支持超长序列

TIN 支持 54,000 长度行为序列，关键优化：

- **分段聚合**：将长序列按时间窗口预聚合
- **稀疏注意力**：只计算与目标语义相关的 Top-K 行为
- **时间索引**：用时间戳建立索引，快速过滤过期行为

## 7. 应用场景

| 场景 | 说明 | 效果 |
|------|------|------|
| 微信朋友圈广告 | 用户长期行为序列建模 | GMV +1.93% |
| 电商推荐 | 用户跨类目兴趣演化 | CTR 提升 |
| 内容推荐 | 兴趣随时间自然变化 | 时效性更好 |
| 在线时延 | 54K 长序列在线推理 | 仅增 5ms |

## 8. 优缺点分析

**优点**：
- 四向交互充分挖掘语义-时间耦合信息
- 目标感知时间编码比固定位置编码更灵活
- 支持超长行为序列（54K）
- 在线推理延迟极低（仅增 5ms）
- 可作为插件集成到现有 CTR 模型

**缺点**：
- 时间编码需要额外存储时间戳信息
- 四向交互增加计算复杂度（相对 DIN）
- 时间间隔编码依赖精确的时间戳
- 短序列场景优势不明显

## 9. 与相关方法对比

| 方法 | 时间建模 | 目标感知 | 序列长度 | 交互维度 |
|------|---------|---------|---------|---------|
| DIN | 无 | 语义注意力 | 短（~100） | 2向（语义×语义） |
| DIEN | GRU 隐状态 | 无 | 中（~200） | 2向 |
| BST | 位置编码+Transformer | 语义注意力 | 中（~200） | 2向 |
| TIN | TTE-P + TTE-T | 语义+时间联合 | 超长（54K） | 4向 |

## 10. PyTorch 代码实现

```python
import torch
import torch.nn as nn
import math

class TargetAwareTemporalEncoding(nn.Module):
    def __init__(self, embed_dim, time_embed_dim=64):
        super().__init__()
        self.pos_embedding = nn.Linear(1, time_embed_dim)
        self.time_embedding = nn.Linear(1, time_embed_dim)
        self.target_proj = nn.Linear(embed_dim, time_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(time_embed_dim * 3, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, embed_dim)
        )
    
    def forward(self, positions, time_deltas, target_emb):
        pos_enc = self.pos_embedding(positions.unsqueeze(-1))
        time_enc = self.time_embedding(time_deltas.unsqueeze(-1))
        target_enc = self.target_proj(target_emb).unsqueeze(1)
        target_enc = target_enc.expand_as(pos_enc)
        fused = self.fusion(torch.cat([pos_enc, time_enc, target_enc], dim=-1))
        return fused

class TIN(nn.Module):
    def __init__(self, num_items, embed_dim=64, time_embed_dim=64,
                 max_seq_len=1000, num_heads=4):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.tte = TargetAwareTemporalEncoding(embed_dim, time_embed_dim)
        
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.time_gate = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, seq_ids, positions, time_deltas, target_id):
        seq_emb = self.item_embedding(seq_ids)
        target_emb = self.item_embedding(target_id)
        
        temporal_enc = self.tte(positions, time_deltas, target_emb)
        
        gated_seq = seq_emb * torch.sigmoid(self.time_gate(temporal_enc))
        
        Q = self.W_q(target_emb).unsqueeze(1)
        K = self.W_k(gated_seq + temporal_enc)
        V = self.W_v(gated_seq)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        interest = torch.matmul(attn_weights, V).squeeze(1)
        
        combined = interest * target_emb
        return self.output_proj(combined).squeeze(-1)


class TINLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, pred, label):
        return self.bce(pred, label.float())
```

## 11. 可视化与结果理解

- **注意力权重热力图**：观察不同时间行为的注意力分布，验证时间衰减效果
- **门控值分布**：可视化 TR 门控值，观察哪些维度被时间信息调制
- **兴趣演化曲线**：对同一用户不同时刻的兴趣向量做 t-SNE，观察兴趣漂移

## 12. 常见问题与易错点

1. **时间戳精度**：行为时间戳需要足够精度（秒级），过粗（天级）会损失信息
2. **时间间隔归一化**：$\Delta t$ 需做 log 变换或归一化，否则数值范围过大
3. **序列长度与效率**：54K 序列需配合稀疏注意力，全量注意力不现实
4. **冷启动用户**：新用户行为极少，TIN 退化为简单 embedding 查找
5. **TTE-P 和 TTE-T 权重**：两种时间编码的贡献可能不同，建议用注意力学习权重

## 13. 学习总结

TIN 的核心贡献在于将时间信息从"辅助特征"提升为"一等公民"，通过四向交互和目标感知时间编码，实现了语义与时间的深度耦合建模。其对超长序列的支持（54K）和极低的在线延迟（+5ms）展现了优秀的工程能力。设计思想——"时间衰减应是目标感知的"——对后续序列模型设计有重要启发。

## 14. 学习路径建议

- **前置知识**：DIN、DIEN、注意力机制、位置编码
- **进阶方向**：BERT4Rec、SASRec、ETA、SDIM
- **推荐论文**：TIN (腾讯)、DIN (KDD 2018)、DIEN (AAAI 2019)
