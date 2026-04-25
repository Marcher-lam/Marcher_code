# Transformer 学习文档

## 1. 算法基础认知

Transformer 是基于 Self-Attention 的序列建模架构，摒弃了 RNN 的循环结构，完全通过注意力机制建模序列依赖。在广告系统中，Transformer 被用于 BST（行为序列建模）、OneTrans（统一排序）、Decision Transformer（自动出价）等核心场景。

## 2. 核心原理

Transformer 由编码器和解码器堆叠而成，核心组件包括：Multi-Head Attention、Position Encoding、Feed-Forward Network、Layer Normalization、残差连接。Self-Attention 使每个位置能直接关注序列中任意其他位置，配合位置编码提供顺序信息。

## 3. 数学公式与推导

**Scaled Dot-Product Attention**（最基础的 Attention 公式）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $Q, K, V$ 分别为查询、键、值矩阵，$d_k$ 为键的维度。除以 $\sqrt{d_k}$ 防止点积过大导致 softmax 梯度消失。

**Multi-Head Attention**：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**位置编码（Positional Encoding）**：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

**Feed-Forward Network**：

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

**Layer Normalization + 残差连接**：

$$
\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

## 4. 训练过程讲解

1. 输入序列经 embedding 层映射为 $d_{model}$ 维向量
2. 加上位置编码 $PE$ 得到位置感知的输入表示
3. 编码器：Multi-Head Attention → Add & Norm → FFN → Add & Norm
4. 解码器：带 mask 的 Self-Attention → Cross-Attention → FFN
5. 线性层 + Softmax 输出预测
6. 交叉熵损失，Adam 优化器，学习率 warmup + cosine decay

## 5. 应用场景

- **BST**：Transformer 编码用户行为序列，用于 CTR 预估
- **OneTrans**：单个 Transformer 统一处理序列建模和特征交互
- **Decision Transformer**：将出价建模为序列决策问题
- **RankMixer**：无参数特征交互的排序模型
- 超长序列建模（LONGER，token 压缩降低复杂度）

## 6. 优缺点分析

**优点：**
- 并行计算，训练效率远高于 RNN
- 直接建模任意距离的依赖关系
- 灵活且可扩展（参数量大时效果持续提升）
- 架构统一，适用于多种任务

**缺点：**
- 计算复杂度 $O(n^2)$，序列长度受限
- 位置编码不如 RNN 的自然顺序建模
- 参数量大，小数据集易过拟合
- 推理时 KV 缓存内存消耗大

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(500, d_model) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_enc[:seq_len]
        out = self.encoder(x)
        return self.fc(out[:, -1, :])

model = TransformerModel(input_dim=64, d_model=128, nhead=4, num_layers=2, output_dim=1)
x = torch.randn(32, 10, 64)
pred = model(x)
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)

mha = MultiHeadAttention(d_model=128, nhead=4)
x = torch.randn(2, 10, 128)
out = mha(x)
```

## 9. 可视化与结果理解

- 多头注意力权重热力图：不同 head 关注不同模式（局部、全局、周期）
- 位置编码相似度矩阵：相邻位置编码相似，远距离差异大
- BST 中 Transformer 层输出 t-SNE 可视化：行为序列的语义聚类
- 学习率 warmup 曲线与训练 loss 的关系

## 10. 模型评估

- 广告 CTR/CVR：AUC + LogLoss + NDCG
- 消融实验：对比不同 head 数、层数、d_model 的影响
- 序列长度测试：短序列 vs 长序列的性能差异
- 推理延迟：对比 RNN/GRU baseline 的在线耗时

## 11. 常见问题与易错点

- **位置编码选择**：正弦编码适合外推，可学习编码适合固定长度
- **Pre-LN vs Post-LN**：Pre-LN（先 LayerNorm 再 Attention）训练更稳定
- **Causal Mask**：自回归生成任务必须 mask 未来位置
- **Flash Attention**：IO-aware 的 tiling 算法，2-4x 加速，建议线上使用
- **GQA**：Grouped Query Attention 减少 KV 头数，降低推理内存

## 12. 学习总结

Transformer 是现代广告推荐系统的核心架构，从 BST 的行为序列建模到 OneTrans 的统一排序模型，再到 Decision Transformer 的出价优化，其应用范围不断扩展。理解 Transformer 是掌握前沿广告算法的必要前提。

## 13. 练习题与思考题（含答案）

**Q1：Transformer 为什么比 RNN 更适合并行训练？**
A1：RNN 必须顺序计算 $h_t$ 依赖 $h_{t-1}$，Transformer 的 Self-Attention 可同时计算所有位置，无顺序依赖。

**Q2：位置编码为什么是必要的？**
A2：Self-Attention 本身是置换不变的（与顺序无关），位置编码注入位置信息使模型能区分不同位置的相同内容。

**Q3：Multi-Head Attention 相比 Single-Head 的优势？**
A3：不同 head 可关注不同子空间的模式（如局部依赖、全局依赖、语法关系），多头提供更丰富的表示。

**Q4：BST 中 Transformer 编码行为序列的关键设计是什么？**
A4：将用户行为序列中的每个行为（item embedding + 位置/时间编码）作为 Transformer 输入，用 Self-Attention 捕捉行为间依赖。

## 14. 学习路径建议

1. 先掌握 Attention 机制（QKV 框架）
2. 理解 Multi-Head Attention 和位置编码
3. 完整理解 Encoder-Decoder 架构
4. 学习 BST 论文，理解 Transformer 在广告序列建模中的应用
5. 进阶 OneTrans（统一注意力）、Flash Attention（工程优化）
