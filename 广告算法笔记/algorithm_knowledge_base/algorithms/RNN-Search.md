# RNN-Search 学习文档

## 1. 算法基础认知

RNN-Search 是 Bahdanau 等人在 2014 年提出的基于注意力机制的序列到序列（Seq2Seq）模型，首次在神经机器翻译（NMT）中引入了**注意力机制**。传统 Seq2Seq 模型将整个源序列压缩为固定长度的上下文向量，信息瓶颈严重。RNN-Search 让解码器在每个生成步动态"关注"源序列的不同部分，突破了这一瓶颈。

RNN-Search 由三个核心组件构成：编码器（双向 RNN）、解码器（单向 RNN）和注意力机制（对齐模型）。它是 Attention 机制的开山之作，直接启发了后来的 Transformer 架构。

## 2. 核心原理

**传统 Seq2Seq 的问题**：编码器将变长源序列编码为固定维度的上下文向量 $c$，解码器所有时刻都使用同一个 $c$。当源序列较长时，$c$ 无法保留所有信息，导致翻译质量下降。

**RNN-Search 的解决方案**：在解码的每个时刻 $t$，动态计算一个上下文向量 $c_t$，它是编码器所有隐藏状态的加权平均，权重由注意力机制决定：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} \cdot h_i$$

其中 $\alpha_{t,i}$ 表示在生成第 $t$ 个目标词时对源序列第 $i$ 个位置的注意力权重。

**Beam Search 解码**：在推理阶段使用 Beam Search 代替贪心解码，同时维护 $k$ 个候选序列，选择累计概率最高的路径，平衡搜索质量与效率。

## 3. 数学公式与推导

**编码器**（双向 RNN）：

$$\overrightarrow{h}_i = \overrightarrow{f}(x_i, \overrightarrow{h}_{i-1}), \quad \overleftarrow{h}_i = \overleftarrow{f}(x_i, \overleftarrow{h}_{i+1})$$

$$h_i = [\overrightarrow{h}_i; \overleftarrow{h}_i]$$

**对齐模型（注意力得分）**：

$$e_{t,i} = v_a^\top \tanh(W_a s_{t-1} + U_a h_i)$$

其中 $s_{t-1}$ 是解码器上一时刻的隐藏状态，$v_a, W_a, U_a$ 是可学习参数。

**注意力权重**：

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x} \exp(e_{t,j})}$$

**上下文向量**：

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i$$

**解码器**：

$$s_t = \text{RNN}(y_{t-1}, s_{t-1}, c_t)$$

$$P(y_t | y_{<t}, x) = \text{softmax}(W_o \cdot [s_t; c_t])$$

## 4. 训练过程讲解

1. **编码**：源序列通过双向 RNN 编码，得到所有位置的隐藏状态 $\{h_1, ..., h_{T_x}\}$。
2. **逐步解码**：对每个时刻 $t$：
   - 用上一隐藏状态 $s_{t-1}$ 与所有 $h_i$ 计算注意力得分 $e_{t,i}$
   - Softmax 得到注意力权重 $\alpha_{t,i}$
   - 加权求和得到上下文向量 $c_t$
   - 更新解码器状态 $s_t$ 并预测 $y_t$
3. **损失计算**：$L = -\sum_{t=1}^{T_y} \log P(y_t^* | y_{<t}^*, x)$，即所有时刻交叉熵之和。
4. **反向传播**：注意力权重的梯度会引导模型学习正确的对齐关系。

## 5. 应用场景

- 机器翻译（RNN-Search 最初的应用场景）
- 文本摘要（源文档编码，逐步生成摘要）
- 对话系统（对话历史编码，生成回复）
- 广告系统中的标题生成、query 改写
- 语音识别（声学特征序列到文本）
- 图像描述生成（CNN 编码图像特征，RNN 解码为文字）

## 6. 优缺点分析

**优点：**
- 解决了固定长度上下文向量的信息瓶颈
- 注意力权重提供了可解释的对齐关系（可视化源-目标对齐）
- 对长序列的翻译质量显著优于传统 Seq2Seq

**缺点：**
- 注意力计算需要 $O(T_x \times T_y)$ 的存储和计算，序列越长开销越大
- RNN 的串行性限制了训练和推理速度
- 被 Transformer 架构全面超越（Transformer 并行化程度更高）

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RNNSearch(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder_cell = nn.LSTMCell(embed_dim + hidden_dim * 2, hidden_dim)
        self.attn_W = nn.Linear(hidden_dim, hidden_dim * 2)
        self.attn_v = nn.Linear(hidden_dim * 2, 1)
        self.out = nn.Linear(hidden_dim + hidden_dim * 2, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.src_embed(src)
        enc_out, _ = self.encoder(src_emb)
        tgt_emb = self.tgt_embed(tgt)
        batch_size, tgt_len, _ = tgt_emb.shape
        s = torch.zeros(batch_size, self.hidden_dim)
        cell = torch.zeros(batch_size, self.hidden_dim)
        logits = []
        for t in range(tgt_len):
            attn_scores = self.attn_v(torch.tanh(self.attn_W(s).unsqueeze(1) + enc_out))
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (attn_weights * enc_out).sum(dim=1)
            inp = torch.cat([tgt_emb[:, t], context], dim=1)
            s, cell = self.decoder_cell(inp, (s, cell))
            logits.append(self.out(torch.cat([s, context], dim=1)))
        return torch.stack(logits, dim=1)

src_vocab, tgt_vocab, embed_d, hidden_d = 8000, 6000, 128, 256
model = RNNSearch(src_vocab, tgt_vocab, embed_d, hidden_d)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

src_batch = torch.randint(1, src_vocab, (16, 20))
tgt_batch = torch.randint(1, tgt_vocab, (16, 15))

for epoch in range(3):
    optimizer.zero_grad()
    output = model(src_batch, tgt_batch)
    loss = criterion(output.view(-1, tgt_vocab), tgt_batch.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def attention_weights(s_prev, h_all, Wa, va):
    scores = []
    for h in h_all:
        score = va.T @ np.tanh(Wa @ s_prev + h)
        scores.append(score[0])
    scores = np.array(scores)
    exp_scores = np.exp(scores - scores.max())
    return exp_scores / exp_scores.sum()

def beam_search_decode(model_encode_fn, decode_step_fn, bos_id, eos_id, beam_width=5, max_len=30):
    beams = [([], 0.0)]
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq and seq[-1] == eos_id:
                new_beams.append((seq, score))
                continue
            next_probs = decode_step_fn(seq)
            top_k = np.argsort(next_probs)[-beam_width:]
            for idx in top_k:
                new_seq = seq + [idx]
                new_score = score + np.log(next_probs[idx] + 1e-10)
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq and seq[-1] == eos_id for seq, _ in beams):
            break
    return beams[0][0]
```

## 9. 可视化与结果理解

- **注意力对齐矩阵**：将 $\alpha_{t,i}$ 绘制为热力图（横轴源序列、纵轴目标序列），对角线上的高亮表示正确的对齐关系
- **源-目标词对齐**：观察注意力权重是否正确地将翻译后的词对齐到源词
- **Beam Search 路径**：可视化 beam 搜索树，展示候选序列的探索过程
- **长序列 vs 短序列**：对比 RNN-Search 和传统 Seq2Seq 在不同源序列长度下的 BLEU 分数，证明注意力的优势

## 10. 模型评估

- **BLEU 分数**：机器翻译的标准评估指标，衡量生成序列与参考序列的 n-gram 重合度
$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$
- **ROUGE**：文本摘要任务的评估指标
- **Attention 对齐准确率**：评估注意力权重是否正确对齐源-目标词
- **解码速度**：Beam Search 的 beam width 对推理延迟的影响

## 11. 常见问题与易错点

- **Teacher Forcing**：训练时使用真实目标序列作为解码器输入，推理时使用模型自身预测，两者分布不一致（Exposure Bias）
- **Beam Search 的 beam width**：过小（1）退化为贪心，过大（50+）计算成本高且可能降低多样性，通常取 5-10
- **长度惩罚**：Beam Search 偏好短序列，需要加入长度归一化
- **注意力维度匹配**：编码器双向输出维度是 2×hidden_dim，解码器 hidden_dim 可能不同，需用投影层对齐
- **EOS token 处理**：解码时需正确处理序列结束符，否则可能无限生成

## 12. 学习总结

RNN-Search 是注意力机制的开山之作，核心创新是让解码器在每步动态关注源序列的不同位置。这不仅解决了 Seq2Seq 的信息瓶颈问题，更催生了整个注意力机制的研究方向。虽然已被 Transformer 取代，但 RNN-Search 中的注意力思想仍是现代 NLP 的基石。

## 13. 练习题与思考题（含答案）

**Q1：为什么 RNN-Search 比传统 Seq2Seq 更擅长处理长序列？**

A1：传统 Seq2Seq 将源序列压缩为固定长度向量，长序列信息必然丢失。RNN-Search 在每步通过注意力机制直接访问源序列的所有隐藏状态，无需压缩，信息利用更充分。

**Q2：Beam Search 中 beam width=1 和 beam width=10 有什么区别？**

A2：beam width=1 即贪心搜索，每步选概率最高的一个词，容易陷入局部最优。beam width=10 同时维护 10 个候选序列，有更大机会找到全局最优路径，但计算量约为 10 倍。

**Q3：注意力权重矩阵 $\alpha_{t,i}$ 为什么可以被视为"对齐"关系的软估计？**

A3：当 $\alpha_{t,i}$ 较大时，说明生成第 $t$ 个目标词时模型主要关注源序列第 $i$ 个位置，这恰好对应了源-目标词之间的翻译对齐关系。在机器翻译中，注意力矩阵通常呈现清晰的对角线模式。

## 14. 学习路径建议

1. 先掌握基础 Seq2Seq 模型（无注意力）
2. 理解注意力机制的直觉——"在正确的时间关注正确的信息"
3. 实现 RNN-Search 并可视化注意力对齐矩阵
4. 学习 Beam Search 解码策略
5. 阅读 Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate"
6. 进阶：学习 Luong Attention（全局/局部注意力）、Self-Attention，直至 Transformer
