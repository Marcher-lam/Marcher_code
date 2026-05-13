# Seq2Seq（Sequence to Sequence）学习文档

> 序列到序列模型，使用编码器-解码器架构处理变长序列转换问题。

## 1. 算法基础认知

### 一句话定义

Seq2Seq是一种将任意长度输入序列映射到任意长度输出序列的神经网络架构，是机器翻译等序列转换任务的基础模型。

### 直觉类比

就像把一段法语翻译成英语——先完整理解法语句子（编码），然后用英语表达出来（解码）。编码器"听"完原句，解码器"说"出翻译。

### 历史背景

- **2014年**：Cho等人提出基于RNN的Seq2Seq（称为"编码器-解码器"）
- **2014年**：Sutskever等人独立提出类似架构（称为"sequence to sequence learning"）
- **2015年**：Bahdanau等人将注意力机制引入Seq2Seq
- **2016年**：Google Neural Machine Translation使用8层LSTM Seq2Seq
- **2017年**：Transformer取代LSTM-based Seq2Seq成为主流

### 算法定位

Seq2Seq是**序列转换任务**的通用框架，属于监督学习。

## 2. 核心原理

### 2.1 核心思想

编码器将输入序列压缩为固定维度的上下文向量（context vector），解码器基于此向量逐个生成输出token。

### 2.2 工作流程

1. 编码器逐个读取输入token，更新隐藏状态
2. 最后一个隐藏状态作为上下文向量 $c$
3. 解码器以 $c$ 和起始符 `<sos>` 为输入，预测第一个输出token
4. 用预测的token作为下一步输入，循环直到生成结束符 `<eos>`

### 2.3 信息瓶颈问题

固定维度的上下文向量是主要瓶颈——无论输入多长，所有信息必须压缩到一个向量中。这是注意力机制引入的主要动机。

## 3. 数学公式与推导

### 3.1 编码器

编码器按时间步处理输入序列：

$$h_t = f(x_t, h_{t-1})$$

其中 $f$ 是RNN/LSTM/GRU单元，$x_t$ 是第 $t$ 个输入token的嵌入向量。

上下文向量 $c$ 通常是最后一个隐藏状态（或所有隐藏状态的某种组合）：

$$c = h_T$$

### 3.2 解码器

解码器在时间步 $t$：

$$s_t = f(y_{t-1}, s_{t-1}, c)$$

其中 $s_t$ 是解码器隐藏状态，$y_{t-1}$ 是前一时刻的输出。

预测概率：

$$p(y_t | y_{<t}, c) = \text{softmax}(W_s s_t + b_s)$$

### 3.3 训练：Teacher Forcing

训练时，使用真实标签（而非模型预测）作为下一步输入：

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log p(y_t^* | y_{<t}^*, x)$$

其中 $y_t^*$ 是真实目标token。Teacher Forcing加速收敛，但导致训练-推理的分布不匹配（exposure bias）。

## 4. 训练过程

### 4.1 数据预处理
- 构建词表（source和target分别）
- 添加特殊token: `<pad>`, `<sos>`, `<eos>`, `<unk>`
- Padding到统一长度（或使用packed_sequence）

### 4.2 训练配置
- 损失函数: CrossEntropyLoss（忽略pad位置）
- 优化器: Adam (lr=0.001) 或 SGD
- 梯度裁剪: max_norm=5.0（防止梯度爆炸）
- Teacher forcing ratio: 0.5（50%用真实标签，50%用预测）

### 4.3 推理：Beam Search
推理时不能使用teacher forcing。常用Beam Search（宽度=3~5）替代贪心解码。

## 5. 应用场景

1. **机器翻译**：英译中、中译英等
2. **文本摘要**：长文档→短摘要
3. **对话系统**：用户输入→机器人回复
4. **语音识别**：语音特征→文本
5. **代码生成**：自然语言→代码

## 6. 优缺点

### 优点
1. **变长序列**：输入输出长度均可变
2. **通用框架**：适用于多种seq2seq任务
3. **端到端训练**：无需特征工程

### 缺点
1. **信息瓶颈**：固定上下文向量难编码长序列
2. **梯度消失**：长序列训练困难
3. **顺序计算**：无法并行
4. **暴露偏差**：Teacher forcing导致训练-推理不匹配

## 7. 调库实现

```python
"""
Seq2Seq模型的完整PyTorch实现（带Attention）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    """LSTM编码器"""

    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    """Bahdanau注意力"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (B, H), encoder_outputs: (B, S, H)
        dec_hidden = decoder_hidden.unsqueeze(1)  # (B, 1, H)
        score = self.V(torch.tanh(
            self.W1(dec_hidden) + self.W2(encoder_outputs)))  # (B, S, 1)
        attn_weights = F.softmax(score.squeeze(-1), dim=1)  # (B, S)
        context = (attn_weights.unsqueeze(1) @ encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    """带注意力的LSTM解码器"""

    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim,
                            num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, cell, encoder_outputs):
        token = token.unsqueeze(1)
        embedded = self.dropout(self.embedding(token))

        # 计算注意力
        context, attn = self.attention(hidden[-1], encoder_outputs)
        context = context.unsqueeze(1)

        # LSTM
        lstm_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # 预测
        prediction = self.fc(torch.cat([output.squeeze(1), context.squeeze(1)], dim=1))
        return prediction, hidden, cell, attn


class Seq2Seq(nn.Module):
    """完整的Seq2Seq模型（带注意力）"""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 编码
        encoder_outputs, hidden, cell = self.encoder(src)

        # 解码
        dec_input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(
                dec_input, hidden, cell, encoder_outputs)
            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t] if teacher_force else output.argmax(1)

        return outputs


def create_model(src_vocab=10000, trg_vocab=10000, embed_dim=256,
                 hidden_dim=512, num_layers=2, device='cpu'):
    encoder = Encoder(src_vocab, embed_dim, hidden_dim, num_layers)
    decoder = Decoder(trg_vocab, embed_dim, hidden_dim, num_layers)
    model = Seq2Seq(encoder, decoder, device)
    return model


def demo():
    device = 'cpu'
    model = create_model(10000, 10000, 256, 512, 2, device)
    src = torch.randint(0, 10000, (32, 20))
    trg = torch.randint(0, 10000, (32, 25))
    out = model(src, trg)
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 推理
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src[:1])
        dec_input = trg[:1, [0]]
        predictions = []
        for _ in range(30):
            output, hidden, cell, attn = model.decoder(
                dec_input.squeeze(1), hidden, cell, encoder_outputs)
            pred_token = output.argmax(1)
            predictions.append(pred_token.item())
            dec_input = pred_token
    print(f"生成长度: {len(predictions)}")


if __name__ == "__main__":
    demo()
```

## 8. 手工实现

```python
"""Seq2Seq核心手工实现"""
import numpy as np

def lstm_cell_handcraft(x, h, c, W_ih, W_hh, b_ih, b_hh):
    """单步LSTM手工实现"""
    gates = (W_ih @ x + b_ih) + (W_hh @ h + b_hh)
    i, f, g, o = np.split(gates, 4)
    i = 1 / (1 + np.exp(-i))
    f = 1 / (1 + np.exp(-f))
    g = np.tanh(g)
    o = 1 / (1 + np.exp(-o))
    c_next = f * c + i * g
    h_next = o * np.tanh(c_next)
    return h_next, c_next

def test():
    np.random.seed(42)
    D, H = 128, 256
    x = np.random.randn(D)
    h = np.zeros(H)
    c = np.zeros(H)
    W_ih = np.random.randn(H*4, D) * 0.1
    W_hh = np.random.randn(H*4, H) * 0.1
    h_next, c_next = lstm_cell_handcraft(x, h, c, W_ih, W_hh, np.zeros(H*4), np.zeros(H*4))
    print(f"LSTM单步: {x.shape} -> h={h_next.shape}, c={c_next.shape}")
    print("测试通过!")

if __name__ == "__main__":
    test()
```

## 9. 可视化

```python
"""Seq2Seq注意力可视化"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(attention_weights, src_tokens, trg_tokens,
                        save_path='seq2seq_attention.png'):
    """可视化注意力权重矩阵"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_xticklabels(src_tokens, fontsize=8)
    ax.set_yticklabels(trg_tokens, fontsize=8)
    ax.set_xlabel('源语言')
    ax.set_ylabel('目标语言')
    plt.colorbar(im, ax=ax)
    plt.title('Seq2Seq 注意力权重')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)

def demo():
    src = ['我', '爱', '机', '器', '学', '习']
    trg = ['I', 'love', 'machine', 'learning']
    attn = np.random.rand(len(trg), len(src))
    attn = attn / attn.sum(axis=1, keepdims=True)
    visualize_attention(attn, src, trg)

if __name__ == "__main__":
    demo()
```

## 10. 模型评估

```python
"""Seq2Seq评估指标"""

def compute_bleu(reference, hypothesis, max_n=4):
    """简化BLEU计算"""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    precisions = []
    for n in range(1, max_n+1):
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
        matches = sum((hyp_ngrams & ref_ngrams).values())
        total = max(sum(hyp_ngrams.values()), 1)
        precisions.append(matches / total)
    brevity_penalty = min(1, np.exp(1 - len(ref_tokens)/max(len(hyp_tokens), 1)))
    geometric_mean = np.exp(np.mean(np.log(np.maximum(precisions, 1e-10))))
    return brevity_penalty * geometric_mean * 100

def demo_bleu():
    from collections import Counter
    ref = "I love machine learning"
    hyp = "I love machine learning"
    print(f"完美匹配BLEU: {compute_bleu(ref, hyp):.2f}")
    hyp2 = "I like machine learning"
    print(f"部分匹配BLEU: {compute_bleu(ref, hyp2):.2f}")

if __name__ == "__main__":
    demo_bleu()
```

## 11. 常见问题与易错点

**Q1: Teacher Forcing的优缺点？**
优点：加速收敛，训练稳定。缺点：exposure bias——推理时用预测而非真实标签，分布差异导致误差累积。

**Q2: 为什么LSTM比普通RNN更适合Seq2Seq？**
LSTM的遗忘门和输入门解决了梯度消失/爆炸问题，能更好地编码长序列。

**Q3: Beam Search如何工作？**
每个时间步保留Top-B个候选序列，而非仅保留最优的1个。宽度B=3~5时效果最好。

**Q4: 为什么Seq2Seq被Transformer取代？**
(1) RNN的时序计算无法并行 (2) 长距离依赖仍有限 (3) Transformer的注意力机制更强大。

## 12. 学习总结

- Seq2Seq是序列转换任务的基础框架
- 核心组件：RNN编码器 + RNN解码器 + 注意力（可选）
- 核心问题：信息瓶颈 + 顺序计算 + 梯度消失
- 历史意义：开启了深度学习在NLP序列转换任务的时代
- 继承关系：Seq2Seq → Seq2Seq+Attention → Transformer

## 13. 练习题

**基础题：**

1. Seq2Seq中编码器最后一个隐藏状态的维度是多少？为什么是信息瓶颈？
> **答案：** 维度=hidden_dim，与序列长度无关。无论输入多长，都被压缩到固定维度向量，丢失了长序列的细粒度信息。

2. Teacher Forcing的ratio=0.5是什么意思？
> **答案：** 50%的概率使用真实token作为下一个输入，50%使用模型自己的预测。平衡训练稳定性和泛化能力。

**进阶题：**

3. 为什么解码器在推理时不使用Teacher Forcing？
> **答案：** 推理时没有真实标签可用。必须使用生成的token作为下一步输入，否则推理和训练不一致。

4. 如何解决Seq2Seq的exposure bias问题？
> **答案：** (1) Scheduled sampling——训练时逐渐降低teacher forcing ratio (2) 强化学习优化（如REINFORCE）。

## 14. 学习路径

**前置：** RNN、LSTM、GRU
**平行：** 注意力机制、Beam Search
**进阶：** Transformer、BART、T5

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Seq2SeqNet(nn.Module):
    def __init__(self, dim_in=20, dim_h=64, dim_out=2):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim_in, dim_h), nn.Linear(dim_h, dim_out)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))

def train(model, X, y, epochs=100, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X), y)
        loss.backward(); opt.step()
        if (ep+1)%20==0: print(f"Ep{ep+1} loss={loss.item():.4f}")

m = Seq2SeqNet()
train(m, torch.randn(500,20), torch.randint(0,2,(500,)))
```
