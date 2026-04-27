# 编码器-解码器（Encoder-Decoder）架构学习文档

> 通用序列转换架构，编码器处理输入，解码器生成输出。

## 1. 算法基础认知

### 一句话定义

编码器-解码器是一种将输入序列转换为输出序列的架构，编码器压缩输入信息为上下文表示，解码器基于该表示逐步生成输出。

### 直觉类比

就像翻译员工作时——先完整聆听说话者的内容（编码），理解其含义，然后用自己的语言表达出来（解码）。

### 历史背景

- **2014年**：Seq2Seq论文提出LSTM编码器-解码器
- **2015年**：Bahdanau引入注意力机制增强解码器
- **2017年**：Transformer完全基于注意力
- **2020年**：BART、T5等预训练编码器-解码器模型兴起

### 算法定位

编码器-解码器是**序列转换任务的通用框架**，广泛用于机器翻译、语音识别、文本摘要等。

## 2. 核心原理

### 2.1 编码器

编码器将输入序列 $x = (x_1, ..., x_n)$ 映射为上下文表示：

$$h_t = \text{Encoder}(x_t, h_{t-1})$$

最终输出 $c = h_n$（或所有隐藏状态的函数）。

### 2.2 解码器

解码器基于上下文和已生成token逐步预测输出：

$$y_t = \text{Decoder}(y_{t-1}, s_{t-1}, c)$$

### 2.3 注意力增强

解码器可访问编码器所有隐藏状态（而非仅最后一个）：

$$c_t = \sum_{i=1}^n \alpha_{ti} h_i$$

$$\alpha_{ti} = \frac{\exp(\text{score}(s_{t-1}, h_i))}{\sum_j \exp(\text{score}(s_{t-1}, h_j))}$$

这解决了固定上下文向量的信息瓶颈问题。

### 2.4 Transformer编码器-解码器

Transformer版本使用自注意力代替RNN：
- 编码器：自注意力 + 前馈网络（可并行）
- 解码器：掩码自注意力 + 交叉注意力 + 前馈网络

## 3. 数学公式与推导

### 3.1 RNN编码器

$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$

所有隐藏状态：$H = [h_1, h_2, ..., h_n]$

### 3.2 RNN解码器

第 $t$ 步：
$$s_t = \tanh(W_{is} y_{t-1} + b_{is} + W_{hs} s_{t-1} + b_{hs})$$

带注意力：
$$s_t = \tanh(W_{is} y_{t-1} + W_{hs} s_{t-1} + W_{cs} c_t)$$

输出概率：
$$p(y_t | y_{<t}, x) = \text{softmax}(W_s s_t + b_s)$$

### 3.3 Transformer交叉注意力

$$Q = s_{t-1} W^Q, \quad K = H W^K, \quad V = H W^V$$

$$c_t = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

## 4. 训练过程

### 4.1 训练流程
1. 编码器处理完整输入序列
2. 解码器使用teacher forcing逐个生成输出
3. 计算交叉熵损失（只计算输出部分）
4. 反向传播更新整个模型

### 4.2 推理流程
1. 编码器处理输入（只需一次）
2. 解码器从 `<sos>` 开始，逐个生成token
3. 使用beam search选择最佳序列
4. 遇到 `<eos>` 停止

## 5. 应用场景

1. **机器翻译**：源语言→目标语言
2. **文本摘要**：长文本→短摘要
3. **语音识别**：声学特征→文本
4. **图像描述**：图像→文本描述
5. **代码生成**：自然语言→代码

## 6. 优缺点

### 优点
1. **通用框架**：适用于各种seq2seq任务
2. **端到端**：无需中间特征工程
3. **注意力增强**：解决了信息瓶颈

### 缺点
1. **顺序生成**：解码器逐个生成，无法并行
2. **错误累积**：早期错误影响后续生成
3. **训练-推理不一致**：teacher forcing vs 自回归

## 7. 调库实现

```python
"""
编码器-解码器架构的完整PyTorch实现
包含RNN和Transformer两种版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RNNEncoder(nn.Module):
    """RNN编码器"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class RNNDecoder(nn.Module):
    """带注意力的RNN解码器"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim,
                          num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))

        # 注意力
        attn_scores = (hidden[-1].unsqueeze(1) @ encoder_outputs.transpose(1, 2))
        attn_weights = F.softmax(attn_scores.squeeze(1), dim=1)
        context = (attn_weights.unsqueeze(1) @ encoder_outputs)

        # RNN
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        # 输出
        output = self.fc(torch.cat([output.squeeze(1), context.squeeze(1)], dim=1))
        return output, hidden, attn_weights


class RNNSeq2Seq(nn.Module):
    """RNN编码器-解码器"""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        dec_input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = trg[:, t] if teacher_force else output.argmax(1)

        return outputs


class TransformerEncoderDecoder(nn.Module):
    """Transformer编码器-解码器"""

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_len=512):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._positional_encoding(max_len, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def _positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _generate_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model) + \
                  self.pos_encoding[:, :src.size(1)].to(src.device)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model) + \
                  self.pos_encoding[:, :tgt.size(1)].to(tgt.device)

        if tgt_mask is None:
            tgt_mask = self._generate_mask(tgt.size(1), tgt.device)

        memory = self.encoder(src_emb, mask=src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc(output)


def demo():
    print("=== RNN编码器-解码器 ===")
    encoder = RNNEncoder(10000, 256, 512)
    decoder = RNNDecoder(10000, 256, 512)
    model = RNNSeq2Seq(encoder, decoder, 'cpu')

    src = torch.randint(0, 10000, (4, 20))
    trg = torch.randint(0, 10000, (4, 25))
    out = model(src, trg)
    print(f"RNN Seq2Seq输出: {out.shape}")

    print("\n=== Transformer编码器-解码器 ===")
    tf_model = TransformerEncoderDecoder(10000, d_model=256, num_heads=4, num_layers=3)
    src = torch.randint(0, 10000, (4, 20))
    tgt = torch.randint(0, 10000, (4, 25))
    out_tf = tf_model(src, tgt[:, :-1])
    print(f"Transformer输出: {out_tf.shape}")
    print(f"参数量: {sum(p.numel() for p in tf_model.parameters()):,}")


if __name__ == "__main__":
    demo()
```

## 8. 手工实现

```python
"""编码器-解码器核心手工实现"""
import numpy as np

def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    """单步GRU手工实现"""
    r = 1 / (1 + np.exp(-(W_ih[0] @ x + b_ih[0] + W_hh[0] @ h + b_hh[0])))
    z = 1 / (1 + np.exp(-(W_ih[1] @ x + b_ih[1] + W_hh[1] @ h + b_hh[1])))
    n = np.tanh(W_ih[2] @ x + b_ih[2] + r * (W_hh[2] @ h + b_hh[2]))
    return (1 - z) * n + z * h

def test():
    D, H = 128, 256
    x = np.random.randn(D)
    h = np.zeros(H)
    W_ih = np.random.randn(3, H, D) * 0.1
    W_hh = np.random.randn(3, H, H) * 0.1
    h_next = gru_cell(x, h, W_ih, W_hh, np.zeros((3, H)), np.zeros((3, H)))
    print(f"GRU: ({D},) -> ({H},)")

if __name__ == "__main__":
    test()
```

## 9. 可视化

```python
"""编码器-解码器可视化"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_encoder_decoder(save_path='enc_dec_arch.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 编码器（左侧）
    enc_layers = 4
    for i in range(enc_layers):
        axes[0].barh(i, 1, height=0.6, color='steelblue', alpha=0.8)
        axes[0].text(1.05, i, f'编码器层 {i+1}', va='center')
    axes[0].set_xlim(0, 2.5)
    axes[0].set_ylim(-0.5, enc_layers-0.5)
    axes[0].set_title('编码器 (自底向上处理输入)')
    axes[0].axis('off')

    # 解码器（右侧）
    dec_layers = 4
    for i in range(dec_layers):
        axes[1].barh(i, 1, height=0.6, color='coral', alpha=0.8)
        axes[1].text(1.05, i, f'解码器层 {i+1}', va='center')
    # 交叉注意力箭头
    axes[1].annotate('', xy=(0.5, -0.3), xytext=(0.5, -1.5),
                     arrowprops=dict(arrowstyle='->', color='green', lw=2))
    axes[1].text(-0.3, -1.0, '编码器输出\n(交叉注意力)', fontsize=9, ha='center')
    axes[1].set_xlim(0, 2.5)
    axes[1].set_ylim(-1.5, dec_layers-0.5)
    axes[1].set_title('解码器 (自顶向下生成输出)')
    axes[1].axis('off')

    plt.suptitle('编码器-解码器架构', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"已保存到 {save_path}")

if __name__ == "__main__":
    visualize_encoder_decoder()
```

## 10. 模型评估

```python
"""编码器-解码器评估"""

def compute_perplexity(loss):
    """从交叉熵损失计算困惑度"""
    return np.exp(loss)

def evaluate_sequence(model, src, trg, criterion):
    model.eval()
    with torch.no_grad():
        output = model(src, trg)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        ppl = compute_perplexity(loss.item())
    return loss.item(), ppl

def demo_eval():
    model = TransformerEncoderDecoder(10000, d_model=128, num_heads=4, num_layers=2)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    src = torch.randint(1, 9999, (2, 10))
    tgt = torch.randint(1, 9999, (2, 12))
    loss, ppl = evaluate_sequence(model, src, tgt, criterion)
    print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    import torch.nn as nn
    demo_eval()
```

## 11. 常见问题与易错点

**Q1: 编码器-解码器的信息瓶颈在哪里？**
在使用RNN的版本中，信息瓶颈在"最后一个隐藏状态"——所有输入信息必须压缩到一个向量。注意力机制解决了这个问题。

**Q2: Transformer的编码器和解码器为什么结构不同？**
解码器有掩码自注意力（防止看到未来token）和交叉注意力（连接编码器输出），编码器只有自注意力。

**Q3: Teacher Forcing是什么？为什么需要？**
训练时将真实标签作为下一步输入（而非模型预测）。加速收敛，但导致训练-推理分布不匹配。

**Q4: 推理时为什么不能使用teacher forcing？**
因为推理时没有真实标签可用。

## 12. 学习总结

- 编码器-解码器是序列转换的通用框架
- RNN版本：顺序计算，信息瓶颈
- Transformer版本：并行计算，注意力解决信息瓶颈
- 核心组件：编码器（理解输入）+ 解码器（生成输出）+ 交叉注意力（连接两者）

## 13. 练习题

**基础题：**

1. 编码器和解码器的角色分别是什么？
> **答案：** 编码器将输入序列编码为上下文表示；解码器基于上下文表示逐步生成输出序列。

2. 交叉注意力在编码器-解码器中的作用是什么？
> **答案：** 连接编码器和解码器——解码器在每一步查询编码器的所有隐藏状态，获取与当前生成相关的输入信息。

**进阶题：**

3. 为什么RNN编码器-解码器比Transformer编码器-解码器慢？
> **答案：** RNN按时间步顺序计算，无法并行。Transformer的自注意力可以同时处理所有位置。

4. 编码器-解码器架构在不使用注意力时的主要缺陷是什么？
> **答案：** 信息瓶颈——所有输入信息压缩到固定维度的上下文向量，长序列信息丢失严重。

## 14. 学习路径

**前置：** RNN、LSTM、注意力机制
**平行：** Seq2Seq、Transformer
**进阶：** BART、T5、mT5