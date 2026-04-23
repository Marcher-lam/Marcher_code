# Encoder-Decoder 学习文档

## 1. 算法基础认知

Encoder-Decoder（编码器-解码器）是深度学习中处理序列到序列（Sequence-to-Sequence, Seq2Seq）任务的核心架构范式，由 Sutskever 等人在 2014 年提出。其核心思想是将变长输入序列编码为固定维度的语义表示，再从该表示解码出变长输出序列。

这种"先理解、再生成"的范式极其通用，不仅适用于机器翻译，还被广泛应用于文本摘要、对话系统、语音识别等领域。现代 Transformer 本质上也遵循 Encoder-Decoder 架构。

## 2. 核心原理

**编码器（Encoder）**：逐步读取输入序列 $x_1, x_2, ..., x_{T_x}$，通过 RNN（或 CNN、Transformer）将信息累积到隐藏状态中。最终隐藏状态 $h_{T_x}$（或上下文向量 $c$）被视为整个输入序列的语义压缩。

**上下文向量（Context Vector）**：编码器输出的语义表示，是输入序列的固定维度向量。在基础 Seq2Seq 中，它就是编码器的最后一个隐藏状态。

**解码器（Decoder）**：以上下文向量为初始状态，逐步生成输出序列 $y_1, y_2, ..., y_{T_y}$。每个时刻 $t$ 的输出依赖于上下文向量、已生成的词 $y_{<t}$ 和解码器当前状态。

**关键约束**：输入和输出序列的长度可以不同，这使得 Seq2Seq 天然适用于翻译（源语言和目标语言长度不同）、摘要（长文档→短摘要）等任务。

## 3. 数学公式与推导

**编码器**：

$$h_t = f(x_t, h_{t-1})$$

$$c = h_{T_x}$$

其中 $f$ 是 RNN 单元（如 LSTM），$c$ 是最终上下文向量。

**解码器**：

$$s_t = g(y_{t-1}, s_{t-1}, c)$$

$$P(y_t | y_{<t}) = \text{softmax}(W_o s_t + b_o)$$

其中 $g$ 是解码器 RNN 单元，$s_0 = c$（或通过线性变换初始化）。

**训练目标**（最大似然估计）：

$$L = -\sum_{t=1}^{T_y} \log P(y_t^* | y_{<t}^*, x)$$

**Teacher Forcing**：训练时解码器输入使用真实标签 $y_t^*$ 而非模型预测 $\hat{y}_t$，加速收敛但引入 Exposure Bias。

**推理时的自回归生成**：

$$\hat{y}_t = \arg\max_{y} P(y | \hat{y}_{<t}, x)$$

## 4. 训练过程讲解

1. **输入处理**：源序列经 embedding 后逐词输入编码器 RNN，最终得到上下文向量 $c$。
2. **解码训练**：目标序列前加 `<SOS>` 起始符、后加 `<EOS>` 结束符。解码器以 `<SOS>` 为起始输入，逐步预测下一个词。
3. **Teacher Forcing**：训练时每步输入真实目标词（即使上一步预测错误），确保训练稳定。
4. **损失计算**：对所有预测步计算交叉熵损失，求平均。
5. **推理生成**：从 `<SOS>` 开始，每步将上一步预测的词作为输入，直到生成 `<EOS>` 或达到最大长度。

## 5. 应用场景

- 机器翻译（源语言→目标语言）
- 文本摘要（长文档→短摘要）
- 对话生成（用户输入→系统回复）
- 广告系统中的 query 改写、标题自动生成
- 代码生成（自然语言描述→代码）
- 语音识别（声学特征序列→文字序列）
- 图像描述（CNN 提取图像特征→RNN 生成文字描述）

## 6. 优缺点分析

**优点：**
- 输入输出长度可不同，灵活性强
- 端到端训练，无需手工设计特征
- 架构通用，适用于多种 Seq2Seq 任务

**缺点：**
- 基础版本的信息瓶颈：固定维度上下文向量难以承载长序列全部信息
- 自回归解码速度慢（必须逐步生成，无法并行）
- Exposure Bias：训练用真实词、推理用预测词，分布不一致
- 对长距离依赖的建模能力有限（RNN 的固有问题）

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embed(x)
        _, (h, c) = self.rnn(embedded)
        return h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h, c):
        embedded = self.embed(x)
        output, (h, c) = self.rnn(embedded, (h, c))
        return self.fc(output), h, c

class Seq2Seq(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(enc_vocab, embed_dim, hidden_dim)
        self.decoder = Decoder(dec_vocab, embed_dim, hidden_dim)

    def forward(self, src, tgt):
        h, c = self.encoder(src)
        batch_size, tgt_len = tgt.shape
        outputs = []
        inp = tgt[:, 0:1]
        for t in range(tgt_len):
            out, h, c = self.decoder(inp, h, c)
            outputs.append(out)
            inp = tgt[:, t:t+1]
        return torch.cat(outputs, dim=1)

enc_vocab, dec_vocab, embed_d, hidden_d = 5000, 6000, 128, 256
model = Seq2Seq(enc_vocab, dec_vocab, embed_d, hidden_d)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

src = torch.randint(1, enc_vocab, (16, 20))
tgt = torch.randint(1, dec_vocab, (16, 15))

for epoch in range(3):
    optimizer.zero_grad()
    output = model(src, tgt)
    loss = criterion(output.view(-1, dec_vocab), tgt.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class SimpleEncoderDecoder:
    def __init__(self, src_vocab, tgt_vocab, embed_dim, hidden_dim):
        self.src_embed = np.random.randn(src_vocab, embed_dim) * 0.01
        self.tgt_embed = np.random.randn(tgt_vocab, embed_dim) * 0.01
        self.enc_W = np.random.randn(hidden_dim, embed_dim + hidden_dim) * 0.01
        self.enc_b = np.zeros(hidden_dim)
        self.dec_W = np.random.randn(hidden_dim, embed_dim + hidden_dim) * 0.01
        self.dec_b = np.zeros(hidden_dim)
        self.out_W = np.random.randn(tgt_vocab, hidden_dim) * 0.01
        self.out_b = np.zeros(tgt_vocab)

    def encode(self, src_ids):
        h = np.zeros(self.enc_W.shape[0])
        for idx in src_ids:
            x = self.src_embed[idx]
            h = np.tanh(self.enc_W @ np.concatenate([x, h]) + self.enc_b)
        return h

    def decode_step(self, tgt_id, h):
        x = self.tgt_embed[tgt_id]
        h = np.tanh(self.dec_W @ np.concatenate([x, h]) + self.dec_b)
        logits = self.out_W @ h + self.out_b
        return logits, h

    def generate(self, src_ids, sos_id, eos_id, max_len=20):
        h = self.encode(src_ids)
        current_id = sos_id
        output = []
        for _ in range(max_len):
            logits, h = self.decode_step(current_id, h)
            current_id = np.argmax(logits)
            if current_id == eos_id:
                break
            output.append(current_id)
        return output
```

## 9. 可视化与结果理解

- **编码器隐藏状态演变**：绘制编码过程中隐藏状态的 t-SNE 降维图，观察相似输入序列是否编码到相近区域
- **解码过程可视化**：展示逐步生成过程中概率分布的变化，观察模型如何在每一步聚焦到正确的词
- **上下文向量分析**：对上下文向量做 PCA 分析，检查不同语义的输入是否被编码到不同区域
- **序列长度 vs 翻译质量**：绘制源序列长度与 BLEU 分数的关系曲线，观察信息瓶颈

## 10. 模型评估

- **BLEU**：机器翻译标准指标，衡量 n-gram 精确率
- **ROUGE-L**：文本摘要指标，基于最长公共子序列
- **困惑度（Perplexity）**：语言模型质量指标
- **生成质量**：人工评估流畅性、准确性、完整性
- **延迟**：自回归解码的逐词生成速度

## 11. 常见问题与易错点

- **上下文向量瓶颈**：基础 Seq2Seq 中所有信息压缩到一个向量，长序列性能差。解决方案：加入注意力机制
- **Teacher Forcing 比例**：全程使用 Teacher Forcing 会导致 Exposure Bias。可用 Scheduled Sampling 逐步减少 Teacher Forcing
- **EOS 训练不足**：如果训练数据中 EOS 出现频率太低，模型可能学不会在合适时机停止生成
- **词汇表不一致**：编码器和解码器通常使用不同的词汇表（如不同语言），实现时需分别处理
- **梯度裁剪**：Seq2Seq 的 BPTT 梯度路径很长，必须做梯度裁剪防止梯度爆炸

## 12. 学习总结

Encoder-Decoder 是序列到序列建模的基础范式。编码器负责"理解"输入，解码器负责"生成"输出，通过固定维度的上下文向量桥接两者。虽然基础版本存在信息瓶颈，但引入注意力机制后（RNN-Search）得到了根本解决。理解 Encoder-Decoder 架构是掌握 Transformer、GPT、BERT 等现代模型的前提。

## 13. 练习题与思考题（含答案）

**Q1：为什么基础 Seq2Seq 的上下文向量是信息瓶颈？**

A1：无论输入序列多长，编码器都将其压缩为固定维度（如 256 维）的单一向量。当序列很长时，256 维无法保留所有必要信息，就像把一整本书压缩成一句话——必然丢失细节。

**Q2：Teacher Forcing 是什么？为什么要用 Scheduled Sampling？**

A2：Teacher Forcing 指训练时解码器输入真实标签而非模型预测。这加速了训练但导致 Exposure Bias——模型在推理时看不到真实标签，一旦预测错误，后续输入偏离训练分布。Scheduled Sampling 以一定概率使用模型预测替代真实标签，逐步过渡。

**Q3：Seq2Seq 的编码器和解码器可以使用不同的 RNN 类型吗？**

A3：可以。编码器常用双向 LSTM（利用完整上下文），解码器用单向 LSTM（自回归生成要求）。甚至编码器可以用 CNN、解码器用 RNN，架构非常灵活。

## 14. 学习路径建议

1. 理解 RNN 和 LSTM 的基础原理
2. 实现基础 Seq2Seq（无注意力）用于简单的序列复制任务
3. 观察上下文向量瓶颈：在长序列上测试性能下降
4. 加入注意力机制，升级为 RNN-Search
5. 学习 Transformer 的 Encoder-Decoder 架构
6. 进阶：了解非自回归翻译（NAT）、CVAE 等生成模型
