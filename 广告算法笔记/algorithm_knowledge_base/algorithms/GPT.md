# GPT 学习文档

## 1. 算法基础认知

GPT（Generative Pre-trained Transformer）是 OpenAI 于 2018 年提出的自回归语言模型。它使用 Transformer 的解码器架构，通过在大规模文本上进行无监督预训练，再在下游任务上微调，开创了"预训练+微调"的范式。

GPT 系列包括 GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023），参数量从 1.17 亿增长到万亿级别，是当前大语言模型（LLM）的基础架构。

## 2. 核心原理

GPT 是一个**自回归语言模型**，核心思想是：

$$P(\text{text}) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

即根据已生成的所有上文来预测下一个词。

### 架构：Transformer Decoder

GPT 使用 Transformer 的解码器部分（去掉编码器），关键组件：
- **掩码自注意力**：每个位置只能关注自身及之前的位置（因果注意力），确保自回归特性
- **前馈网络**：对每个位置独立做非线性变换
- **层归一化 + 残差连接**：稳定训练

### 两阶段范式

1. **预训练**：在大规模文本上做下一个词预测（无监督）
2. **微调**：在特定任务数据上调整模型参数（有监督）

## 3. 数学公式与推导

### 自注意力（带掩码）

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

其中 $M$ 是掩码矩阵：

$$M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$$

确保位置 $i$ 无法看到位置 $j > i$ 的信息。

### 预训练目标

给定语料 $\mathcal{U} = \{u_1, u_2, \ldots, u_n\}$，最大化对数似然：

$$\mathcal{L}_{\text{pretrain}} = \sum_{t=1}^{T} \log P(u_t \mid u_1, \ldots, u_{t-1}; \theta)$$

### 微调目标

对于带标签数据集 $\mathcal{D}$，任务损失为：

$$\mathcal{L}_{\text{finetune}} = \sum_{(x, y) \in \mathcal{D}} \log P(y \mid x_1, \ldots, x_m; \theta)$$

实际训练中通常结合语言模型损失：

$$\mathcal{L} = \mathcal{L}_{\text{finetune}} + \lambda \cdot \mathcal{L}_{\text{pretrain}}$$

## 4. 训练过程讲解

### 预训练阶段

1. **数据准备**：收集大规模文本（BooksCorpus 等），使用 BPE 分词
2. **输入编码**：文本 → token 序列 → token embedding + position embedding
3. **前向传播**：token 序列经过多层 Transformer Decoder
4. **损失计算**：每个位置预测下一个 token，计算交叉熵损失
5. **反向传播**：更新所有参数

### 微调阶段

1. 在输入序列末尾添加任务特定的 token（如 `[CLS]`）
2. 添加线性分类头
3. 在目标任务数据上联合优化任务损失和语言模型损失

## 5. 应用场景

- 文本生成（故事续写、代码生成）
- 机器翻译
- 文本摘要
- 问答系统
- ChatGPT 式对话系统
- 广告文案生成、创意写作

## 6. 优缺点分析

**优点**：
- 强大的文本生成能力（自回归天然适合生成任务）
- 预训练+微调范式，迁移能力强
- 架构统一，不同任务只需调整输入格式
- 规模化效应显著（更大模型效果更好）

**缺点**：
- 单向注意力，无法同时利用双向上下文（相比 BERT）
- 生成速度慢（逐 token 生成）
- 预训练计算资源需求大
- 可能生成不准确或有害内容（幻觉问题）

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_text = "Artificial intelligence is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7,
    top_k=50,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=128):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = attn.masked_fill(self.mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

vocab_size = 100
model = MiniGPT(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)
x = torch.randint(0, vocab_size, (2, 16))
logits = model(x)
print("输入形状:", x.shape)
print("输出形状:", logits.shape)

target = torch.randint(0, vocab_size, (2, 16))
loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
print("损失:", loss.item())
```

## 9. 可视化与结果理解

GPT 的核心可视化：**掩码注意力矩阵**。左下三角为可关注区域，右上三角被遮蔽。

```
位置:  1  2  3  4
1:    [✓  ✗  ✗  ✗]    ← 位置1只能看自己
2:    [✓  ✓  ✗  ✗]    ← 位置2看1,2
3:    [✓  ✓  ✓  ✗]    ← 位置3看1,2,3
4:    [✓  ✓  ✓  ✓]    ← 位置4看全部
```

这保证了 GPT 的自回归特性：生成第 $t$ 个词时只能看到前 $t-1$ 个词。

## 10. 模型评估

- **困惑度（Perplexity）**：语言模型的核心指标，$PPL = \exp(-\frac{1}{N}\sum \log P(x_t \mid x_{<t}))$
- **下游任务准确率**：GLUE、SuperGLUE 等 benchmark
- **人工评估**：流畅性、相关性、准确性
- **BLEU / ROUGE**：生成文本与参考文本的相似度

## 11. 常见问题与易错点

- **掩码方向**：因果掩码必须确保未来位置不被看到，实现时注意 `triu` 的方向
- **位置编码**：GPT 使用可学习的位置嵌入（非正弦），最大长度固定
- **生成策略**：贪心解码容易退化，实际使用 beam search / top-k / top-p / temperature
- **EOS token**：生成时需要设置停止条件，否则模型会持续生成

## 12. 学习总结

GPT 使用 Transformer Decoder 架构进行自回归语言建模，通过"预训练+微调"范式在各类 NLP 任务上取得了突破性成果。其核心是因果掩码的自注意力机制，使模型只能看到上文信息。GPT 的局限性在于单向注意力（相比 BERT 的双向），但在文本生成任务上天然适配。从 GPT 到 ChatGPT 的发展展示了规模化的力量。

## 13. 练习题与思考题（含答案）

**Q1**：GPT 为什么使用因果掩码（Causal Mask）？

**A1**：因为 GPT 是自回归模型，生成第 $t$ 个 token 时只能依赖前 $t-1$ 个 token。因果掩码将未来位置的注意力权重设为 $-\infty$（Softmax 后为 0），确保信息不泄露。

**Q2**：GPT 和 BERT 的预训练目标有什么区别？

**A2**：GPT 使用下一个词预测（从左到右），BERT 使用掩码语言模型（双向上下文预测被遮蔽的词）。GPT 擅长生成，BERT 擅长理解。

**Q3**：为什么 GPT 系列可以通过增大模型规模持续提升效果？

**A3**：语言建模是一个开放性问题，模型容量越大，能捕获的语言规律越丰富。同时，大规模数据中蕴含的模式需要更大的模型来表达，存在 scaling law（损失随模型大小、数据量、计算量幂律下降）。

## 14. 学习路径建议

1. 掌握 Transformer 架构 → 2. 理解 GPT 的自回归建模 → 3. 对比 GPT 与 BERT 的区别 → 4. 学习 GPT-2/GPT-3 的规模化策略 → 5. 了解 RLHF（人类反馈强化学习）对齐方法
