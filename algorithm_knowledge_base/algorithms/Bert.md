# Bert 学习文档

> 基于Transformer编码器的双向预训练语言模型，通过掩码语言建模和下一句预测任务学习深度双向表示

---

## 1. 算法基础认知

### 一句话定义
BERT（Bidirectional Encoder Representations from Transformers）是一种基于Transformer编码器架构的预训练语言模型，通过双向上下文学习深度语言表示，在下游任务上通过微调达到优异性能。

### 直觉类比
想象你在阅读句子"The animal didn't cross the street because it was too tired." 你需要理解"it"指的是"animal"。BERT可以同时看到句子的左右两侧，通过双向注意力直接建立"it"和"animal"之间的联系，无论它们距离多远。这就像你同时能看到整个句子，而不是像传统RNN那样一个词一个词地读。

### 历史背景
BERT由Google的Devlin等人于2018年提出，基于Vaswani等人2017年的Transformer架构。BERT的关键创新是：1）使用双向Transformer编码器；2）提出掩码语言建模（MLM）作为预训练任务；3）引入下一句预测（NSP）任务。BERT刷新了11个NLP任务的SOTA，引发了预训练-微调范式的革命。

### 算法定位
- 类型：自监督学习 → 语言模型（可微调到各种下游任务）
- 输出：句子或词的表示（用于分类、问答、NER等）
- 模型类型：双向语言模型、基于Transformer编码器

### 前置知识
- Transformer架构：自注意力、位置编码、编码器结构
- 语言模型基础：掩码语言建模、自监督学习
- 深度学习：预训练与微调、迁移学习
- 注意力机制：Query、Key、Value概念
- Python基础：PyTorch、Hugging Face Transformers库

---

## 2. 核心原理

### 2.1 核心思想
BERT的核心思想是**通过双向Transformer编码器，在大规模无标注文本上预训练，学习深度双向语言表示，然后通过微调适配到各种下游NLP任务**：

1. **双向编码器**：使用Transformer编码器，同时考虑左右上下文
2. **掩码语言建模（MLM）**：随机遮盖15%的词，让模型预测被遮盖的词
3. **下一句预测（NSP）**：判断两个句子是否连续，学习句子级关系
4. **微调**：在下游任务数据上继续训练，适配特定任务

### 2.2 工作流程

**预训练阶段**：
1. **输入处理**：词嵌入 + 段嵌入（区分句子A/B）+ 位置编码
2. **MLM任务**：随机遮盖15%的词，模型预测被遮盖的词
3. **NSP任务**：判断句子B是否是句子A的下一句
4. **联合训练**：两个任务的损失加权求和

**微调阶段**：
1. **任务适配**：根据下游任务添加适当的输出层
   - 分类任务：添加分类头（[CLS]标记的表示 → 分类器）
   - 问答任务：预测答案的起始和结束位置
   - NER任务：对每个词标记进行分类
2. **继续训练**：在任务数据上微调所有参数（或BERT部分参数）

### 2.3 关键概念解释

- **掩码语言建模（MLM）**：随机遮盖15%的词，让模型根据双向上下文预测被遮盖的词。这迫使模型学习深度双向表示。
- **下一句预测（NSP）**：判断句子B是否是句子A的真实下一句。帮助模型理解句子间关系，对QA和NLI任务有益。
- **[CLS]标记**：每个序列开头的特殊标记，其最终隐藏状态通常用于分类任务。
- **[SEP]标记**：分隔两个句子的特殊标记。
- **双向注意力**：BERT使用Transformer编码器，每个词都能关注到句子中所有位置（包括左右两侧）。
- **句子对输入**：BERT可以接受句子对（如问答中的问题和上下文），通过段嵌入区分。

### 2.4 几何/直观解释

**从表示学习角度看**：BERT的每一层都在学习不同层次的语言表示：
- **低层**：学习词性、句法结构
- **中层**：学习语义关系、句子结构
- **高层**：学习任务相关的抽象表示

**从注意力角度看**：BERT的双向注意力允许每个词直接"看到"所有其他词。例如对于"The animal didn't cross the street because it was too tired."，当处理"it"时，注意力可以直接聚焦于"animal"，无论它们距离多远。

**MLM任务**：就像填空练习，"今天天气真[MASK]，适合去公园。"模型需要根据上下文预测[MASK]应该是"好"或"晴朗"等。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|----------|
| $n$ | 序列长度（包括[CLS]和[SEP]） | 标量 |
| $d_{model}$ | 模型维度（嵌入维度） | 标量 |
| $L$ | Transformer编码器层数 | 标量 |
| $H$ | 注意力头数 | 标量 |
| $x$ | 输入序列（词ID） | $n \times 1$ |
| $E$ | 词嵌入矩阵 | $V \times d_{model}$ |
| $T$ | 段嵌入（句子A/B） | $2 \times d_{model}$ |
| $P$ | 位置编码 | $n \times d_{model}$ |
| $h_{[CLS]}$ | [CLS]标记的隐藏状态 | $d_{model} \times 1$ |

### 3.2 问题形式化

**预训练目标**：BERT在大规模文本语料上预训练，通过两个自监督任务：

1. **掩码语言建模（MLM）**：
   给定被随机遮盖15%词的序列 $\tilde{x}$，预测被遮盖的原始词：
   $$\max_\theta \log P(x_{masked} | \tilde{x}; \theta)$$

2. **下一句预测（NSP）**：
   给定句子对 $(A, B)$，判断B是否是A的下一句：
   $$\max_\theta \log P(\text{IsNext}(A,B) | A, B; \theta)$$

**联合目标**：
$$\mathcal{L}(\theta) = \mathcal{L}_{MLM}(\theta) + \mathcal{L}_{NSP}(\theta)$$

### 3.3 目标函数/损失函数

**1. MLM损失（掩码语言建模）**：

对于被遮盖的位置 $i \in \mathcal{M}$（占15%），计算交叉熵损失：

$$\mathcal{L}_{MLM}(\theta) = -\sum_{i \in \mathcal{M}} \log \text{softmax}(W h_i + b)_{x_i}$$

其中：
- $h_i$ 是位置 $i$ 的最终隐藏状态（来自Transformer编码器）
- $W$ 是输出权重矩阵（通常与词嵌入矩阵共享权重）
- $x_i$ 是被遮盖位置 $i$ 的真实词ID

**2. NSP损失（下一句预测）**：

使用[CLS]标记的最终隐藏状态 $h_{[CLS]}$ 进行二分类：

$$\mathcal{L}_{NSP}(\theta) = -\log P(\text{IsNext} | h_{[CLS]}; \theta) - \log P(\text{NotNext} | h_{[CLS]}; \theta)$$

实现为二分类的交叉熵损失：
$$\mathcal{L}_{NSP}(\theta) = -[y \log \sigma(W_{NSP} h_{[CLS]}) + (1-y) \log(1-\sigma(W_{NSP} h_{[CLS]}))]$$

其中 $y \in \{0, 1\}$ 是真实标签（1表示B是A的下一句）。

### 3.4 推导过程

**Step 1：输入表示**

对于输入序列（可能是句子对 $(A, B)$）：

$$x = [\text{[CLS]}, w_1^A, w_2^A, ..., \text{[SEP]}, w_1^B, w_2^B, ..., \text{[SEP]}]$$

输入表示是三种嵌入的和：

$$\text{InputEmbedding}(x_i) = E_{x_i} + T_{seg(i)} + P_i$$

其中：
- $E_{x_i}$：词嵌入（WordPiece嵌入）
- $T_{seg(i)}$：段嵌入（句子A所有词用 $T_A$，句子B所有词用 $T_B$）
- $P_i$：位置编码（学习的位置嵌入，不是正弦/余弦）

**Step 2：Transformer编码器**

输入表示通过 $L$ 层Transformer编码器：

$$h^{(0)} = \text{InputEmbedding}(x)$$

$$h^{(l)} = \text{TransformerEncoderLayer}^{(l)}(h^{(l-1)}), \quad l = 1, ..., L$$

每层包含：
- 多头自注意力（双向，看到整个序列）
- 残差连接 + 层归一化
- 前馈网络
- 残差连接 + 层归一化

**Step 3：输出表示**

最终层输出 $h^{(L)} = [h_{[CLS]}, h_1, h_2, ..., h_n]$ 是序列的上下文表示。

**Step 4：任务特定的输出层**

- **MLM**：对每个被遮盖位置 $i$，计算 $\text{softmax}(W h_i^{(L)})$，预测原词
- **NSP**：使用 $h_{[CLS]}^{(L)}$ 进行二分类

### 3.5 最终解/算法步骤

**BERT预训练算法**：
```
输入：大规模文本语料 D，BERT配置（L, H, d_model）
输出：预训练模型参数 θ

1. 初始化BERT参数 θ（Xavier初始化）
2. 对于每次迭代，直到收敛：
   a. 从D采样批次句子对 (A, B)
   b. 随机采样15%的词作为遮盖位置 M
      对 i ∈ M:
        以80%概率替换为[MASK]
        以10%概率替换为随机词
        以10%概率保持不变（保留原词）
   c. 构造输入：x = [CLS] + A + [SEP] + B + [SEP]
   d. 计算输入表示：h⁽⁰⁾ = E_x + T_seg + P
   e. 通过L层Transformer编码器：h⁽ᴸ⁾ = Encoder(h⁽⁰⁾)
   f. 计算MLM损失：L_MLM = -Σᵢ∈M log softmax(Whᵢ⁽ᴸ⁾)ₓᵢ
   g. 计算NSP损失：L_NSP = -log P(IsNext | h_{[CLS]}⁽ᴸ⁾)
   h. 总损失：L = L_MLM + L_NSP
   i. 反向传播更新参数：θ ← θ - α∇θL
3. 返回预训练模型参数 θ
```

**BERT微调算法**：
```
输入：预训练BERT参数 θ，下游任务数据 {(xᵢ, yᵢ)}ᵢ₌₁ᴺ
输出：微调后的模型参数 θ'

1. 根据任务构造输入（如分类任务：x = [CLS] + 文本 + [SEP]）
2. 添加任务特定的输出层（如分类头：W_task）
3. 对于每次迭代：
   a. 前向传播：h = BERT(x; θ)，logits = W_task · h_{[CLS]}
   b. 计算任务损失：L_task = TaskLoss(logits, y)
   c. 反向传播更新所有参数（θ和W_task）
4. 返回微调后的参数 θ'
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

```python
from transformers import BertTokenizer

# ============================================
# BERT数据预处理要点
# ============================================
print("=" * 60)
print("BERT数据预处理")
print("=" * 60)

# 1. 加载BERT分词器（使用WordPiece）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(f"词表大小: {tokenizer.vocab_size}")
print(f"特殊标记: {tokenizer.special_tokens_map}")

# 2. 示例句子对（用于NSP任务或QA任务）
sentence_a = "The cat sat on the mat."
sentence_b = "It was a sunny day."

# 3. 分词 + 添加特殊标记
# BERT的输入格式：[CLS] + A + [SEP] + B + [SEP]
encoding = tokenizer(
    sentence_a,
    sentence_b,
    add_special_tokens=True,  # 自动添加[CLS]和[SEP]
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

print(f"\n句子A: {sentence_a}")
print(f"句子B: {sentence_b}")
print(f"\n输入ID形状: {encoding['input_ids'].shape}")
print(f"注意力掩码形状: {encoding['attention_mask'].shape}")
print(f"段ID形状: {encoding['token_type_ids'].shape}")

# 4. 查看分词结果
tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
print(f"\n分词结果（前20个）: {tokens[:20]}")

# 5. MLM任务：创建遮盖标签
# 随机遮盖15%的词（除了特殊标记）
input_ids = encoding['input_ids'].clone()
labels = input_ids.clone()

# 创建遮盖掩码（不遮盖特殊标记）
special_tokens_mask = tokenizer.get_special_tokens_mask(
    encoding['input_ids'][0], already_has_special_tokens=True
)
attention_mask = encoding['attention_mask'][0]

# 随机选择15%的位置进行遮盖
probability_matrix = torch.full(input_ids.shape, 0.15)
mask = (torch.bernoulli(probability_matrix).bool() & 
        ~torch.tensor(special_tokens_mask, dtype=torch.bool).unsqueeze(0) & 
        (attention_mask.bool()).unsqueeze(0))

# 执行遮盖
masked_input = input_ids.clone()
masked_input[mask] = tokenizer.mask_token_id
labels[~mask] = -100  # 只计算被遮盖位置的损失

print(f"\n原始输入: {tokenizer.decode(input_ids[0][:20])}")
print(f"遮盖后输入: {tokenizer.decode(masked_input[0][:20])}")
print(f"标签（前20个）: {labels[0][:20].tolist()}")
```

**预处理要点**：
1. **WordPiece分词**：BERT使用WordPiece子词分词，词表大小通常是30k左右
2. **特殊标记**：`[CLS]`（分类标记）、`[SEP]`（分隔标记）、`[MASK]`（MLM任务）、`[PAD]`（填充）
3. **段ID（token_type_ids）**：区分句子A（0）和句子B（1）
4. **注意力掩码（attention_mask）**：标记哪些是真实词（1），哪些是填充（0）
5. **MLM的80-10-10规则**：80%替换为`[MASK]`，10%替换为随机词，10%保持不变

### 4.2 参数初始化

```python
from transformers import BertConfig, BertForMaskedLM

# ============================================
# BERT模型初始化
# ============================================
print("\n" + "=" * 60)
print("BERT模型初始化")
print("=" * 60)

# 1. BERT配置（BERT-Base为例）
config = BertConfig(
    vocab_size=30522,       # 词表大小
    hidden_size=768,         # 模型维度（d_model）
    num_hidden_layers=12,    # Transformer层数
    num_attention_heads=12,  # 注意力头数
    intermediate_size=3072,   # 前馈网络中间层维度（4*hidden_size）
    max_position_embeddings=512,  # 最大序列长度
    type_vocab_size=2,       # 段嵌入的种类数（句子A/B）
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

# 2. 初始化BERT模型（用于MLM任务）
model = BertForMaskedLM(config)

# 3. 查看模型结构
print(f"模型配置:")
print(f"  词表大小: {config.vocab_size}")
print(f"  模型维度 (hidden_size): {config.hidden_size}")
print(f"  Transformer层数: {config.num_hidden_layers}")
print(f"  注意力头数: {config.num_attention_heads}")
print(f"  最大序列长度: {config.max_position_embeddings}")

# 4. 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# 5. BERT使用学习的位置嵌入（不是正弦/余弦）
# 查看位置嵌入
print(f"\n位置嵌入形状: {model.bert.embeddings.position_embeddings.weight.shape}")
print(f"词嵌入形状: {model.bert.embeddings.word_embeddings.weight.shape}")
print(f"段嵌入形状: {model.bert.embeddings.token_type_embeddings.weight.shape}")
```

**初始化建议**：
1. **权重初始化**：使用Xavier初始化或正态分布 $N(0, 0.02^2)$
2. **位置嵌入**：BERT使用可学习的位置嵌入（不是正弦/余弦）
3. **激活函数**：BERT使用GELU激活（不像GPT-2）
4. **预训练模型**：通常直接使用预训练权重（`from_pretrained('bert-base-uncased')`），而不是从头初始化

### 4.3 迭代过程（训练循环）

```python
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AdamW

# ============================================
# BERT预训练循环（简化版）
# ============================================
print("\n" + "=" * 60)
print("BERT预训练循环（示例）")
print("=" * 60)

# 假设我们有数据加载器
# class MLMDataset(Dataset):
#     def __init__(self, texts, tokenizer, max_length=128):
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#     
#     def __len__(self):
#         return len(self.texts)
#     
#     def __getitem__(self, idx):
#         # 创建MLM任务的数据
#         text = self.texts[idx]
#         encoding = self.tokenizer(
#             text,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         # 创建遮盖（简化版）
#         input_ids = encoding['input_ids'][0]
#         labels = input_ids.clone()
#         # 随机选择15%位置遮盖（简化）
#         mask = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool()
#         input_ids[mask] = self.tokenizer.mask_token_id
#         labels[~mask] = -100  # 忽略非遮盖位置
#         
#         return {
#             'input_ids': input_ids,
#             'attention_mask': encoding['attention_mask'][0],
#             'token_type_ids': encoding['token_type_ids'][0],
#             'labels': labels
#         }

# 初始化模型
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 优化器（使用AdamW，带权重衰减）
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# 训练参数
num_epochs = 3

print(f"训练设备: {device}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 模拟一个训练batch
batch_size = 4
seq_len = 128

# 模拟输入（实际应从数据加载器获取）
input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
attention_mask = torch.ones(batch_size, seq_len).to(device)
token_type_ids = torch.zeros(batch_size, seq_len).to(device)  # 只有一个句子
labels = input_ids.clone()
# 创建遮盖标签（简化）
mask = torch.bernoulli(torch.full((batch_size, seq_len), 0.15)).bool()
labels[~mask] = -100  # 忽略非遮盖位置

# 训练模式
model.train()

for epoch in range(num_epochs):
    # 清零梯度
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels
    )
    
    loss = outputs.loss
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("\n预训练完成（示例batch）")
```

**训练要点**：
1. **学习率**：预训练通常使用很小的学习率（如5e-5），配合Warmup调度
2. **批次大小**：BERT预训练使用大批次（如256或更大），可能需要梯度累积
3. **梯度裁剪**：防止梯度爆炸，通常裁剪到范数1.0
4. **权重衰减**：BERT使用AdamW优化器，带有L2正则化（weight_decay=0.01）
5. **Warmup**：学习率预热（如前10%步线性增加到峰值）

### 4.4 收敛条件

BERT预训练通常在固定步数后停止（如1M步、3M步），但可以监控：

```python
def check_bert_convergence(losses, perplexities, window=1000):
    """检查BERT是否收敛"""
    if len(losses) < window:
        return False
    
    # 检查损失是否稳定
    recent_losses = losses[-window:]
    loss_std = np.std(recent_losses)
    
    # 检查困惑度（Perplexity）是否不再下降
    recent_ppl = perplexities[-window:]
    ppl_diff = recent_ppl[-1] - np.mean(recent_ppl[:-1])
    
    if loss_std < 0.01 and abs(ppl_diff) < 1.0:
        print(f"可能收敛: 损失标准差={loss_std:.4f}, 困惑度变化={ppl_diff:.2f}")
        return True
    return False
```

**收敛相关要点**：
1. **困惑度（Perplexity）**：BERT的MLM任务主要评估指标，$PPL = e^{loss}$
2. **训练/验证损失曲线**：应下降并趋于平稳
3. **NSP准确率**：二元分类准确率，监控是否过拟合
4. **下游任务性能**：定期在下游任务上评估，确保学到了有用的表示

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值（BERT-Base） |
|--------|------|----------|----------|
| `hidden_size` | 模型维度（d_model） | 768, 1024, 1536 | 768 |
| `num_hidden_layers` | Transformer层数 | 12, 24, 36 | 12 |
| `num_attention_heads` | 注意力头数 | 12, 16, 32 | 12 |
| `intermediate_size` | FFN中间层维度 | 4*hidden_size | 3072 |
| `max_position_embeddings` | 最大序列长度 | 512, 1024, 2048 | 512 |
| `learning_rate` | 学习率 | 5e-5 ~ 3e-4 | 5e-5 |
| `batch_size` | 批次大小 | 16, 32, 64 | 取决于GPU内存 |
| `warmup_steps` | 学习率预热步数 | 10k, 30k, 50k | 总步数的10% |
| `weight_decay` | L2正则化强度 | 0.01 ~ 0.1 | 0.01 |
| `dropout` | Dropout概率 | 0.1 ~ 0.2 | 0.1 |

**选择建议**：
1. **模型规模**：BERT-Base（110M参数）适合大多数任务；BERT-Large（340M参数）需要更多资源
2. **序列长度**：根据任务需求设置，BERT最多512个token（位置嵌入限制）
3. **学习率**：BERT对学习率敏感，通常使用5e-5（微调）或1e-4（预训练）
4. **批次大小**：受GPU内存限制，大批次有助于稳定训练

---

## 5. 应用场景

### 5.1 典型应用

**应用1：文本分类**
- 场景：情感分析（正面/负面）、主题分类、垃圾邮件检测
- 为什么适合：BERT的[CLS]标记表示整个句子的语义，适合分类
- 实现：在[CLS]标记上添加分类头，微调BERT

**应用2：问答系统（QA）**
- 场景：给定问题和包含答案的段落，预测答案的起始和结束位置
- 为什么适合：BERT可以处理句子对（问题和段落），学习它们的交互
- 实现：添加QA头，预测答案的start_logits和end_logits

**应用3：命名实体识别（NER）**
- 场景：识别文本中的实体（人名、地名、组织名等）
- 为什么适合：BERT对每个词输出上下文表示，适合序列标注
- 实现：对每个词（或子词）添加分类头，预测实体标签

### 5.2 适用数据特征

1. **需要双向上下文**：任务需要从左右两侧理解上下文（如问答、NER）
2. **中等规模数据**：微调通常在1k-100k样本上效果良好
3. **需要深度语言理解**：任务需要深层的语义表示
4. **可用预训练模型**：BERT的预训练模型广泛可用（Hugging Face）
5. **句子或文档级任务**：BERT可处理长达512个token的文本

### 5.3 不适用场景

1. **自回归生成**：BERT是编码器，不擅长自回归生成文本 → 使用GPT等解码器模型
2. **超大规模生成**：BERT不适合大规模文本生成 → 使用Transformer解码器或编码器-解码器
3. **实时推理（低延迟）**：BERT的前向传播需要计算所有层，延迟较高 → 使用蒸馏、量化
4. **文本长度>512**：BERT的位置嵌入限制为512 → 使用Longformer、BigBird等长文档模型
5. **计算资源有限**：BERT-Base需要~1GB显存进行推理 → 使用DistilBERT、MobileBERT等轻量模型

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 双向上下文 | 同时利用左右上下文，理解更深入 | 使用Transformer编码器 |
| 强大的迁移能力 | 预训练学到的表示可迁移到下游任务 | 有大量预训练数据 |
| 简单的微调 | 只需添加任务特定层，继续训练即可 | 下游任务数据适中 |
| 刷新多项SOTA | 在11个NLP任务上达到当时最佳 | 大规模预训练 |
| 广泛的可用性 | Hugging Face等平台提供预训练模型 | 开源社区支持 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 预训练计算昂贵 | 需要大量GPU/TPU训练数天 | 使用预训练模型，不必从头训练 |
| 最大序列长度限制 | 位置嵌入限制为512（BERT-Base） | 使用Longformer、BigBird等长文档模型 |
| 不适合生成任务 | BERT是编码器，不擅长自回归生成 | 使用GPT、BART等生成模型 |
| [MASK]不匹配 | 预训练使用[MASK]，微调时没有[MASK] | 使用ELECTRA等改进模型 |
| 计算资源需求 | 推理需要计算所有层，延迟较高 | 使用蒸馏（DistilBERT）、量化、剪枝 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================
# 使用Hugging Face Transformers实现BERT微调
# ============================================
print("=" * 60)
print("BERT调库实现（Hugging Face Transformers）")
print("=" * 60)

# ============================================
# 1. 加载预训练模型和分词器
# ============================================
print("\n加载BERT-Base模型和分词器...")

# 加载分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载预训练模型（用于序列分类）
# num_labels: 分类类别数（如2表示二分类）
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # 二分类（如情感分析）
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"模型加载完成，设备: {device}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 2. 准备示例数据（情感分析）
# ============================================
class SentimentDataset(Dataset):
    """情感分析数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 示例数据（正面情感=1，负面情感=0）
texts = [
    "This movie is fantastic! I loved it.",
    "Terrible film, waste of time.",
    "The book was amazing, couldn't put it down.",
    "Boring and poorly written."
]
labels = [1, 0, 1, 0]

# 创建数据集
dataset = SentimentDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"\n数据集大小: {len(dataset)}")

# ============================================
# 3. 训练循环（微调）
# ============================================
def train_bert(model, dataloader, epochs=3):
    """微调BERT模型"""
    model.train()
    
    # 优化器（使用AdamW）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # 学习率调度器（线性预热+衰减）
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10%预热
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

# 注意：这里不实际运行训练，仅展示代码
print("\n微调代码已准备（需要实际数据运行）")

# ============================================
# 4. 推理（预测）
# ============================================
def predict_sentiment(model, tokenizer, text):
    """预测文本情感"""
    model.eval()
    
    # 分词
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
    
    return prediction, probabilities[0][prediction].item()

# 测试推理（需要加载训练好的模型）
# test_text = "This movie is great!"
# pred, prob = predict_sentiment(model, tokenizer, test_text)
# print(f"\n测试文本: {test_text}")
# print(f"预测: {'正面' if pred==1 else '负面'}, 概率: {prob:.4f}")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================
# 手写实现BERT核心组件（简化版，用于教学）
# ============================================
print("=" * 60)
print("手写实现BERT核心组件")
print("=" * 60)

class BertEmbeddings(nn.Module):
    """BERT的嵌入层（词嵌入 + 段嵌入 + 位置嵌入）"""
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, token_type_ids=None):
        """
        input_ids: (batch, seq_len)
        token_type_ids: (batch, seq_len)，区分句子A/B
        """
        seq_len = input_ids.size(1)
        
        # 位置ID（0, 1, 2, ..., seq_len-1）
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 三种嵌入的和
        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        seg_emb = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_emb + pos_emb + seg_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertSelfAttention(nn.Module):
    """BERT的自注意力机制（双向）"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)，1表示真实词，0表示填充
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # 线性投影
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        # 应用注意力掩码（将填充位置设为很大的负数）
        if attention_mask is not None:
            # 扩展mask到(b
            mask = attention_mask[:, None, None, :]  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_size)
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return context

class BertLayer(nn.Module):
    """BERT编码器层"""
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_heads, dropout)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_LayerNorm = nn.LayerNorm(hidden_size)
        
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_LayerNorm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.attention_output(attn_output)
        attn_output = self.attention_dropout(attn_output)
        hidden_states = self.attention_LayerNorm(hidden_states + attn_output)
        
        # 前馈网络 + 残差连接 + 层归一化
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)  # BERT使用GELU
        output = self.output(intermediate_output)
        output = self.output_dropout(output)
        hidden_states = self.output_LayerNorm(hidden_states + output)
        
        return hidden_states

# ============================================
# 测试手写BERT组件
# ============================================
print("\n测试手写BERT组件...")

# 初始化组件
vocab_size = 30522
hidden_size = 768
num_heads = 12
intermediate_size = 3072
max_position_embeddings = 512
seq_len = 10
batch_size = 2

# 创建嵌入层
embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, 2)
print(f"嵌入层创建完成")

# 创建编码器层
bert_layer = BertLayer(hidden_size, num_heads, intermediate_size)
print(f"编码器层创建完成")

# 测试输入
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

# 前向传播
embedded = embeddings(input_ids, token_type_ids)
print(f"嵌入输出形状: {embedded.shape}")

output = bert_layer(embedded)
print(f"编码器层输出形状: {output.shape}")

print("BERT核心组件工作正常！")
```

---

## 9. 可视化与结果理解

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# BERT可视化：注意力权重
# ============================================
print("=" * 60)
print("BERT可视化：注意力权重")
print("=" * 60)

def visualize_bert_attention(model, tokenizer, text, layer_idx=-1, head_idx=0):
    """可视化BERT的注意力权重（简化版）"""
    model.eval()
    
    # 分词
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids]).to(next(model.parameters()).device)
    
    # 获取注意力权重（需要修改模型或使用hook，这里仅示意）
    # 实际中，可以使用BertModel并获取attentions参数
    
    # 示意：创建随机注意力权重进行可视化
    seq_len = len(tokens)
    attn_weights = np.random.rand(seq_len, seq_len)  # 实际应从模型获取
    attn_weights = attn_weights / attn_weights.sum(axis=1, keepdims=True)  # 归一化
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, 
                 xticklabels=tokens, 
                 yticklabels=tokens,
                 cmap='YlOrRd', 
                 annot=True, 
                 fmt='.2f')
    plt.title(f'BERT Self-Attention Weights (Layer {layer_idx+1}, Head {head_idx+1})')
    plt.xlabel('Key Position (attended to)')
    plt.ylabel('Query Position (attending)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    print("观察要点：")
    print("1. BERT是双向的，每个词可以关注到所有其他词")
    print("2. 不同头可能关注不同的关系（语法、语义等）")
    print("3. 训练后的模型会学习到有意义的关系")

# 示例（需要实际模型和分词器）
# text = "The cat sat on the mat because it was tired."
# visualize_bert_attention(model, tokenizer, text, layer_idx=-1, head_idx=0)
```

**结果理解**：
1. **注意力热力图**：显示每个词与其他词之间的注意力权重，颜色越深表示关注度越高
2. **多头差异**：不同头可能学习不同的关系模式（如语法关系、共指关系等）
3. **训练vs未训练**：未训练的模型注意力比较均匀，训练后的模型有清晰的模式

---

## 10. 模型评估

```python
import torch
import torch.nn as nn
import math

# ============================================
# BERT模型评估
# ============================================
print("=" * 60)
print("BERT模型评估")
print("=" * 60)

def evaluate_bert_classification(model, dataloader, device):
    """评估BERT分类模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item() * input_ids.size(0)
            
            _, predicted = torch.max(logits, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    perplexity = math.exp(avg_loss)
    
    return avg_loss, accuracy, perplexity

# 假设我们有验证集
# val_loss, val_acc, val_ppl = evaluate_bert_classification(model, val_dataloader, device)

# print("\n" + "="*50)
# print("BERT模型评估报告")
# print("="*50)
# print(f"验证损失: {val_loss:.4f}")
# print(f"准确率 (Accuracy): {val_acc:.4f}")
# print(f"困惑度 (Perplexity): {val_ppl:.4f}")
# print(f"较高的准确率表示分类性能越好")
# print(f"较低的困惑度表示MLM任务性能越好")

print("\nBERT特殊评估指标：")
print("1. MLM困惑度 (Perplexity): 掩码语言建模任务的主要指标")
print("2. NSP准确率: 下一句预测任务的准确率")
print("3. 下游任务指标: 根据任务而定（如准确率、F1分数、BLEU等）")
print("4. 推理速度: 每秒处理的样本数，或延迟（毫秒）")
```

**BERT特殊评估点**：
1. **MLM困惑度（Perplexity）**：语言建模任务的主要评估指标，$PPL = e^{loss}$
2. **NSP准确率**：下一句预测任务的二元分类准确率
3. **下游任务性能**：根据任务而定（如分类任务用准确率/F1，QA任务用Exact Match/F1）
4. **推理速度**：BERT的推理延迟，特别是对于长序列
5. **模型大小vs性能**：BERT-Base vs BERT-Large的权衡

---

## 11. 常见问题与易错点

### 11.1 [MASK]在预训练和微调之间的不匹配
**原因**：
BERT预训练时使用`[MASK]`标记，但微调时（如分类任务）没有`[MASK]`，导致预训练-微调不匹配。

**解决方案**：
```python
# BERT的解决方案：
# 在预训练中，只有80%的遮盖位置替换为[MASK]，
# 10%替换为随机词，10%保持不变。
# 这样模型学会在微调时处理没有[MASK]的输入。

# 改进的模型：ELECTRA
# 使用判别式预训练：用一个生成器替换[MASK]，然后判别器判断每个词是否被替换
# 避免了[MASK]的不匹配问题
```

### 11.2 序列长度超过512个token
**原因**：
BERT的位置嵌入最大为512，对于长文档处理不了。

**解决方案**：
```python
# 方法1：截断（简单但丢失信息）
encoding = tokenizer(text, max_length=512, truncation=True)

# 方法2：滑动窗口（对长文档）
def sliding_window_tokenization(text, tokenizer, window_size=510, stride=256):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+window_size]
        if len(chunk) < 10:  # 太短则跳过
            break
        chunks.append(tokenizer.convert_tokens_to_ids(chunk))
    return chunks

# 方法3：使用长文档模型
# 如Longformer、BigBird，支持更长的序列（如4096或更长）
from transformers import LongformerModel
```

### 11.3 微调时学习率设置不当
**原因**：
BERT预训练使用了较小的学习率（如1e-4），微调时如果学习率太大，可能破坏预训练学到的表示。

**解决方案**：
```python
# 微调时使用较小的学习率
# BERT-Base: 2e-5 ~ 5e-5
# BERT-Large: 1e-5
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# 使用学习率调度器（预热+线性衰减）
from transformers import get_linear_schedule_with_warmup

total_steps = len(dataloader) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10%预热

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 每个batch后调用 scheduler.step()
```

### 11.4 忽略填充位置的损失计算
**原因**：
BERT的输入通常包含填充（`[PAD]`），如果不处理，这些填充位置会影响损失计算。

**解决方案**：
```python
# 正确做法：使用attention_mask和labels
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels  # 对于填充位置，labels设为-100（PyTorch会忽略）
)

# 对于MLM任务，手动设置标签
labels = input_ids.clone()
labels[~mask] = -100  # 忽略非遮盖位置（或填充位置）

# Hugging Face的模型会自动处理attention_mask
```

---

## 12. 学习总结

### 核心要点回顾：
1. **双向编码器**：BERT使用Transformer编码器，同时考虑左右上下文
2. **MLM任务**：$\max_\theta \log P(x_{masked} | \tilde{x}; \theta)$
3. **NSP任务**：判断句子B是否是句子A的下一句
4. **微调**：在下游任务数据上继续训练，适配特定任务
5. **输入表示**：词嵌入 + 段嵌入 + 位置嵌入

### 从BERT到其他模型：
```
BERT（双向编码器，MLM+NSP预训练）
    ↓
RoBERTa（去掉NSP，更多数据，更大批次）
    ↓
ALBERT（参数共享，句子顺序预测）
    ↓
ELECTRA（判别器预训练，更样本高效）
    ↓
其他：DistilBERT（蒸馏）、MobileBERT（移动端）、BioBERT（领域适应）
```

### 实践建议：
1. **默认策略**：加载预训练BERT → 根据任务添加输出层 → 微调
2. **资源管理**：BERT-Base需要~1GB显存推理；微调需要更多
3. **学习率**：微调通常使用2e-5到5e-5
4. **序列长度**：BERT最长512；长文档使用Longformer等
5. **报告**：给出下游任务指标（准确率、F1）、MLM困惑度（如果有）

---

## 13. 练习题与思考题（含答案）

### 练习题

**习题1：基础计算**
问题：BERT-Base配置：hidden_size=768，num_heads=12。计算：
1. 每个头的维度是多少？
2. 如果输入序列长度n=128，计算单个自注意力层的参数量（只考虑Q、K、V、输出投影矩阵）。

<details>
<summary>答案</summary>

1. **每个头的维度**：
   $$\text{head\_size} = \frac{\text{hidden\_size}}{\text{num\_heads}} = \frac{768}{12} = 64$$

2. **参数量计算**：
   - Query投影矩阵 $W_Q$: $768 \times 768 = 589,824$ 参数
   - Key投影矩阵 $W_K$: $768 \times 768 = 589,824$ 参数
   - Value投影矩阵 $W_V$: $768 \times 768 = 589,824$ 参数
   - 输出投影矩阵 $W_O$: $768 \times 768 = 589,824$ 参数
   
   总参数量（仅这四个矩阵）：$4 \times 589,824 = 2,359,296$ 参数。
   
   注意：这里没有计算偏置项。BERT的某些实现可能省略偏置。
</details>

**习题2：编程实践**
问题：使用Hugging Face Transformers加载BERT-Base模型，对句子进行编码，查看各层的输出形状。

<details>
<summary>答案</summary>

```python
from transformers import BertTokenizer, BertModel
import torch

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # 评估模式

# 示例句子
text = "The cat sat on the mat."

# 分词
encoding = tokenizer(text, return_tensors='pt')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

print(f"输入ID形状: {input_ids.shape}")
print(f"注意力掩码形状: {attention_mask.shape}")

# 前向传播（获取所有隐藏状态）
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True  # 返回所有层的隐藏状态
    )

# 查看输出
print(f"\n最后一层的隐藏状态形状: {outputs.last_hidden_state.shape}")
print(f"  形状: (batch={input_ids.shape[0]}, seq_len={input_ids.shape[1]}, hidden_size=768)")

print(f"\n所有隐藏状态的数量: {len(outputs.hidden_states)}")
print(f"  包括：嵌入层输出 + 12层编码器输出 + 最后一层（=13个）")

# 查看第一层和最后一层
print(f"\n[CLS]标记的表示（第一层）: {outputs.hidden_states[0][0, 0, :5]}...")  # 前5个值
print(f"[CLS]标记的表示（最后一层）: {outputs.last_hidden_state[0, 0, :5]}...")

# 获取tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(f"\nTokens（前10个）: {tokens[:10]}")
```
</details>

**习题3：理论推导**
问题：解释BERT的MLM任务中80-10-10规则的动机。为什么不直接将所有遮盖位置替换为`[MASK]`？

<details>
<summary>答案</summary>

**80-10-10规则**：
对于选择的15%要遮盖的位置：
- **80%替换为`[MASK]`**：让模型学习根据上下文预测被遮盖的词
- **10%替换为随机词**：迫使模型关注上下文，而不是只依赖某个特定词
- **10%保持不变**：让模型学习表示真实的词（因为微调时没有`[MASK]`）

**为什么不直接全部替换为`[MASK]`？**

1. **预训练-微调不匹配**：
   - 预训练时输入包含`[MASK]`标记
   - 微调时（如分类任务）输入没有`[MASK]`
   - 这种不匹配可能损害微调性能

2. **解决方案**：
   - 通过10%保持不变，模型学会处理没有`[MASK]`的输入
   - 通过10%随机替换，模型不过度依赖特定词，增强鲁棒性

**效果**：
- 实验表明，80-10-10规则比全部替换为`[MASK]`有更好的下游任务性能
- 虽然这增加了预训练的难度（因为模型有时看到的是错误输入），但提高了泛化能力

**总结**：80-10-10规则是为了缓解预训练（有`[MASK]`）和微调（无`[MASK]`）之间的不匹配问题，提高模型的鲁棒性和泛化能力。
</details>

### 思考题

**思考题1**：BERT和GPT有什么核心区别？各适用于什么场景？

<details>
<summary>答案</summary>

| 方面 | BERT | GPT |
|------|------|------|
| **架构** | Transformer编码器（双向） | Transformer解码器（自回归） |
| **注意力** | 双向自注意力（看到整个句子） | 掩码自注意力（只能看到前面） |
| **预训练任务** | MLM（预测遮盖词）+ NSP（下一句预测） | 自回归语言建模（预测下一个词） |
| **上下文** | 双向，同时利用左右上下文 | 单向，只能利用上文 |
| **适用任务** | 理解任务（分类、QA、NER等） | 生成任务（文本生成、对话等） |
| **输入/输出** | 编码器：输入文本，输出表示 | 解码器：输入上文，生成下文 |

**适用场景**：

**BERT适合**：
- **文本理解**：情感分析、文本分类、自然语言推理
- **问答系统**：给定问题和上下文，找出答案
- **序列标注**：命名实体识别（NER）、词性标注
- **需要双向上下文的任务**：任何需要理解整个句子的任务

**GPT适合**：
- **文本生成**：故事创作、新闻写作、代码生成
- **对话系统**：ChatGPT类型的应用
- **自回归任务**：任何需要逐步生成的任务

**结合使用**：
- **编码器-解码器模型**（如BART、T5）：结合BERT的双向编码和GPT的生成能力，适合序列到序列任务（翻译、摘要）
</details>

**思考题2**：为什么BERT最大序列长度是512？如何扩展到更长文本？

<details>
<summary>答案</summary>

**为什么BERT最大序列长度是512？**

1. **位置嵌入限制**：BERT学习的位置嵌入矩阵大小为 $512 \times 768$。位置ID超过511就没有对应的嵌入了。
2. **计算复杂度**：自注意力的复杂度是 $O(n^2)$，其中 $n$ 是序列长度。超过512，计算量和内存消耗会急剧增加。
3. **预训练数据**：大多数预训练句子对长度在512以内。

**如何扩展到更长文本？**

1. **滑动窗口（Sliding Window）**：
   ```python
   def sliding_window(tokens, window_size=510, stride=256):
       chunks = []
       for i in range(0, len(tokens), stride):
           chunk = tokens[i:i+window_size]
           chunks.append(chunk)
           if len(chunk) < window_size:
               break
       return chunks
   ```
   对长文档分成多个512长度的chunk，分别处理，然后合并结果。

2. **长文档Transformer模型**：
   - **Longformer**：使用局部注意力+全局注意力，复杂度降为 $O(n)$
   - **BigBird**：类似Longformer，使用稀疏注意力模式
   - **LED（Longformer Encoder-Decoder）**：用于序列到序列任务
   
   这些模型可以处理长达4096或8192个token。

3. **层次化方法**：
   - 先对每个段落编码（512 token）
   - 然后对所有段落的表示进行聚合（如另一个Transformer层）
   
4. **记忆增强**：
   - 使用外部记忆模块存储长文档的信息
   - 如Memory Transformer等

**总结**：BERT的512限制来自位置嵌入和计算复杂度。对于长文档，可以使用滑动窗口或更长的模型（如Longformer）。
</details>

---

## 14. 学习路径建议

### 初级阶段（掌握BERT基础）
1. 理解BERT的双向编码器架构和Transformer基础
2. 掌握MLM和NSP预训练任务
3. 学会使用Hugging Face Transformers加载和微调BERT
4. 理解输入表示：词嵌入 + 段嵌入 + 位置嵌入

**学习时间**：3-4周**

### 中级阶段（深入理解原理）
1. 推导BERT的预训练目标和损失函数
2. 理解80-10-10规则和预训练-微调匹配问题
3. 掌握不同下游任务的微调技巧（分类、QA、NER）
4. 探索BERT的变体：RoBERTa、ALBERT、ELECTRA等

**学习时间**：4-6周**

### 高级阶段（前沿研究）
1. 研究更高效的预训练方法：ELECTRA、GCNN等
2. 了解长文档Transformer：Longformer、BigBird等
3. 探索领域适应：BioBERT、SciBERT等
4. 研究模型压缩：DistilBERT、MobileBERT、量化、剪枝
5. 探索多语言和多模态BERT变体

**学习时间**：6-8周**

### 实践项目建议
1. **基础项目**：情感分析（如IMDB电影评论），微调BERT-Base
2. **进阶项目**：问答系统（SQuAD数据集），实现span prediction
3. **挑战项目**：命名实体识别（CoNLL-2003），处理嵌套实体或跨句依赖

### 推荐资源
- **书籍**：《Natural Language Processing with Transformers》（Lewis Tunstall等）；《Speech and Language Processing》（Dan Jurafsky等）第部分
- **课程**：Stanford CS224N（NLP with Deep Learning）；Hugging Face课程（https://huggingface.co/learn）
- **论文**：Devlin et al. (2018) BERT论文；Liu et al. (2019) RoBERTa论文；Clark et al. (2020) ELECTRA论文
- **代码**：Hugging Face Transformers官方文档；BERT GitHub仓库（https://github.com/google-research/bert）
- **实践**：Hugging Face平台（https://huggingface.co/）；使用预训练BERT进行各种NLP任务；参与Kaggle NLP竞赛
