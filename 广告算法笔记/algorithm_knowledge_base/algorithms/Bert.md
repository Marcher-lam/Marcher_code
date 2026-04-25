# BERT 学习文档

## 1. 算法基础认知

BERT（Bidirectional Encoder Representations from Transformers）是 Google 于 2018 年提出的预训练语言模型。它使用 Transformer 编码器架构，通过双向上下文建模，在 11 项 NLP 基准测试上刷新了纪录，标志着 NLP 领域进入预训练时代。

BERT 的核心创新是两个预训练任务：**MLM（Masked Language Model）** 和 **NSP（Next Sentence Prediction）**，使得模型能够学习深层双向语言表示。

## 2. 核心原理

### 双向建模

与 GPT 的单向（左到右）不同，BERT 允许每个位置同时看到左右两侧的上下文信息，通过 Transformer 的完整自注意力实现真正的双向编码。

### 预训练任务

**MLM（掩码语言模型）**：随机遮蔽 15% 的输入 token，让模型根据上下文预测被遮蔽的词。这打破了自回归的限制，实现了双向信息流。

**NSP（下一句预测）**：给定句子对 (A, B)，预测 B 是否是 A 的下一句，帮助模型理解句子间的关系。

## 3. 数学公式与推导

### 输入表示

BERT 的输入 = Token Embedding + Segment Embedding + Position Embedding：

$$\mathbf{h}_i^{(0)} = \mathbf{e}_{\text{token}}^{(i)} + \mathbf{e}_{\text{segment}}^{(i)} + \mathbf{e}_{\text{position}}^{(i)}$$

### Transformer 编码器

每层编码器包含多头自注意力和前馈网络：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

注意 BERT 使用的是 **GELU** 激活函数而非 ReLU。

### MLM 损失

设被遮蔽的位置集合为 $\mathcal{M}$，MLM 损失为：

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x}; \theta)$$

其中 $\tilde{x}$ 是被遮蔽后的输入序列。

### NSP 损失

$$\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext} \mid \mathbf{h}_{\texttt{[CLS]}})$$

### 总损失

$$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

### MLM 的遮蔽策略

被选中的 15% token 中：
- 80% 替换为 `[MASK]`
- 10% 替换为随机词
- 10% 保持不变

这种策略是为了缓解预训练与微调之间的差异（微调时没有 `[MASK]` token）。

## 4. 训练过程讲解

### 预训练

1. **数据准备**：大规模文本语料（BooksCorpus + English Wikipedia）
2. **构建训练样本**：
   - MLM：随机遮蔽 15% 的 token
   - NSP：50% 真实句子对 + 50% 随机句子对
3. **前向传播**：输入经多层 Transformer 编码器
4. **计算损失**：MLM 损失 + NSP 损失
5. **优化**：Adam 优化器，线性学习率衰减 + warmup

### 微调

1. 在输入前添加 `[CLS]` token，句间插入 `[SEP]`
2. 取 `[CLS]` 对应的输出向量作为句子级表示
3. 添加任务特定的分类头（通常是单层线性层）
4. 在下游数据上端到端微调所有参数

## 5. 应用场景

- 文本分类（情感分析、新闻分类）
- 命名实体识别（NER）
- 问答系统（SQuAD）
- 自然语言推断（NLI / MNLI）
- 语义相似度计算
- 广告系统中的 CTR 预估特征提取
- 搜索排序

## 6. 优缺点分析

**优点**：
- 双向上下文建模，理解能力强
- 统一的预训练+微调范式，适配多种任务
- 预训练模型开源，社区生态完善（HuggingFace）
- 在理解类任务上表现优异

**缺点**：
- 不擅长文本生成（非自回归模型）
- 预训练计算成本高（BERT-base: 4 天 × 4 TPU）
- `[MASK]` token 在预训练与微调间存在差异
- NSP 任务被后续研究认为效果有限（RoBERTa 移除了 NSP）
- 固定长度限制（BERT-base: 512 tokens）

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

texts = ["这部电影非常好看", "这部电影太烂了"]
labels = torch.tensor([1, 0])

encodings = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
loader = DataLoader(dataset, batch_size=2)

optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()

for epoch in range(3):
    for input_ids, attention_mask, label in loader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
test_enc = tokenizer(["很棒的作品"], padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    out = model(**test_enc)
    pred = torch.argmax(out.logits, dim=1)
    print(f"预测标签: {pred.item()}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, C))

class BertEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ff(x))
        return x

class MiniBERT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.seg_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([BertEncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.mlm_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size))
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, token_type_ids=None):
        B, T = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.token_emb(input_ids) + self.seg_emb(token_type_ids) + self.pos_emb(torch.arange(T, device=input_ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        mlm_logits = self.mlm_head(x)
        nsp_logits = self.nsp_head(x[:, 0])
        return mlm_logits, nsp_logits

vocab_size = 100
model = MiniBERT(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)
input_ids = torch.randint(0, vocab_size, (2, 16))
mlm_out, nsp_out = model(input_ids)
print(f"MLM 输出形状: {mlm_out.shape}")
print(f"NSP 输出形状: {nsp_out.shape}")

mlm_target = torch.randint(0, vocab_size, (2, 16))
nsp_target = torch.tensor([1, 0])
loss_mlm = F.cross_entropy(mlm_out.view(-1, vocab_size), mlm_target.view(-1))
loss_nsp = F.cross_entropy(nsp_out, nsp_target)
print(f"MLM 损失: {loss_mlm.item():.4f}, NSP 损失: {loss_nsp.item():.4f}")
print(f"总损失: {(loss_mlm + loss_nsp).item():.4f}")
```

## 9. 可视化与结果理解

BERT 的注意力可视化可以展示模型关注了哪些词。例如在句子 "The cat sat on the [MASK]" 中：

```
预测 [MASK] 时，注意力权重分布：
  "The":    ██░░░░░  (10%)
  "cat":    █████░░  (50%)   ← 主语，提供语义
  "sat":    ███░░░░  (30%)   ← 动词，提供语法
  "on":     █░░░░░░  (5%)
  "the":    ░░░░░░░  (5%)
```

`[CLS]` token 的输出向量聚拢了整个句子的语义信息，常用于句子级分类。

## 10. 模型评估

- **GLUE benchmark**：涵盖 SST-2、MNLI、QQP 等 9 个任务的综合性评测
- **SQuAD**：阅读理解任务，使用 Exact Match（EM）和 F1 分数
- **NER**：CoNLL-2003，使用实体级 F1 分数
- **困惑度**：可以评估 MLM 的质量，但 BERT 的 MLM 损失与自回归困惑度不可直接比较

## 11. 常见问题与易错点

- **输入长度超限**：BERT 最大 512 tokens，超长文本需要截断或滑动窗口
- **微调学习率**：通常使用 2e-5 到 5e-5，过大会破坏预训练权重
- **[CLS] 的使用**：句子级任务用 `[CLS]` 的输出，token 级任务用每个位置的输出
- **attention_mask**：padding 位置必须设置 mask 为 0，否则会引入噪声
- **分词差异**：中文 BERT 使用字级别分词，不是词级别

## 12. 学习总结

BERT 的核心贡献在于通过 MLM（掩码语言模型）预训练任务打破了自回归语言模型只能单向建模的限制，实现了真正的双向上下文表示。其"预训练+微调"范式极大地降低了下游任务的标注数据需求，一个通用预训练模型即可通过少量微调适配到分类、匹配、序列标注等多种任务，标志着 NLP 进入预训练时代。

BERT 的关键优势是双向理解能力强、迁移学习效果好、社区生态完善（HuggingFace），适合几乎所有 NLP 理解类任务。在广告系统中，BERT 常被用于 query 意图理解、搜索相关性匹配和 CTR 模型的文本特征提取。但它不擅长文本生成，且 512 token 的长度限制在处理长文档时需要额外策略。

在知识体系中，BERT 是本库中 Transformer 编码器架构的直接应用，与 GPT（解码器）形成对比。理解 BERT 是掌握后续预训练模型（RoBERTa、ALBERT、ELECTRA）以及广告领域中基于预训练的 CTR 模型的基础。

工业实践中，BERT 的微调学习率通常设置为 2e-5 到 5e-5，过大会破坏预训练权重。大规模部署时需考虑知识蒸馏（DistilBERT）或模型剪枝以降低推理延迟，中文场景使用字级别分词需注意覆盖领域专用词汇。

## 13. 练习题与思考题（含答案）

**Q1**：BERT 的 MLM 中，为什么 15% 的被选中 token 不全部替换为 `[MASK]`？

**A1**：如果全部替换为 `[MASK]`，模型会学到"看到 [MASK] 就预测"的捷径，且微调时没有 `[MASK]` 导致预训练-微调不一致。混合替换策略（80% [MASK]，10% 随机词，10% 原词）缓解了这个问题。

**Q2**：BERT 为什么不适合文本生成任务？

**A2**：BERT 是编码器模型，没有因果掩码，每个位置能看到全部上下文。生成任务需要自回归地逐词预测，BERT 的双向注意力会导致信息泄露。生成任务应使用 GPT 等解码器模型。

**Q3**：RoBERTa 相比 BERT 做了哪些改进？

**A3**：主要改进包括：(1) 移除 NSP 任务；(2) 动态掩码（每个 epoch 重新生成 mask）；(3) 更大的 batch size 和训练数据；(4) 使用 Byte-Pair Encoding（BPE）分词；(5) 更长的训练时间。这些改进显著提升了模型效果。

## 14. 学习路径建议

1. 掌握 Transformer 编码器架构 → 2. 理解 BERT 的 MLM + NSP 预训练 → 3. 对比 BERT 与 GPT 的架构差异 → 4. 学习 BERT 变体（RoBERTa、ALBERT、DistilBERT）→ 5. 掌握 HuggingFace Transformers 框架 → 6. 探索 T5、BART 等编码器-解码器模型
