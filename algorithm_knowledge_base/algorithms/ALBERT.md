# ALBERT 学习文档

> 轻量级 BERT，参数效率提升 10 倍，效果超越 BERT。

---

## 1. 算法基础认知

### 1.1 发展背景

ALBERT（A Lite BERT）由 Google Research 于 2019 年提出，是 BERT 的轻量级改进版本。原始 BERT 虽然在各项 NLP 任务上取得了 SOTA 效果，但参数量巨大（BERT-Large 达 3.4 亿），训练和推理成本高昂。ALBERT 通过两项核心技术将参数量减少到原来的 1/10，同时保持甚至超越原版性能。

### 1.2 核心定位

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头数 |
|------|--------|------|----------|-------------|
| BERT-Base | 1.08 亿 | 12 | 768 | 12 |
| BERT-Large | 3.4 亿 | 24 | 1024 | 16 |
| ALBERT-Base | 1200 万 | 12 | 768 | 12 |
| ALBERT-Large | 6000 万 | 24 | 1024 | 16 |

### 1.3 三项核心改进

1. **因子分解嵌入参数**：将词嵌入矩阵分解为两个小矩阵
2. **跨层参数共享**：各层之间共享注意力机制和前馈网络参数
3. **句间连贯性预训练**：引入句间顺序预测任务

---

## 2. 核心原理

### 2.1 因子分解嵌入（Factorized Embedding Parameterization）

原始 BERT 中，词嵌入维度 $E$ 必须等于隐藏维度 $H$，这导致了大量的参数开销。ALBERT 将嵌入矩阵分解为两个更小的矩阵：

$$V \times H \rightarrow (V \times E) + (E \times H)$$

其中：
- $V$ 是词表大小（通常 30000）
- $H$ 是隐藏维度（768 或 1024）
- $E$ 是嵌入维度（通常 128）

**参数节省**：以 BERT-Base 为例：
- 原版：$30000 \times 768 = 2300$ 万参数
- ALBERT：$30000 \times 128 + 128 \times 768 = 384$ 万 + 10 万 = 390 万参数

### 2.2 跨层参数共享（Cross-Layer Parameter Sharing）

ALBERT 的所有 12 层（Base）或 24 层（Large）共享同一组参数。具体共享方式：

- **全部共享**：所有层使用完全相同的注意力机制和前馈网络
- **仅共享注意力**：各层共享自注意力机制，但前馈网络独立
- **仅共享前馈网络**：各层共享前馈网络，但注意力独立

实践中发现，共享全部参数效果最好，虽然有一定性能损失，但参数量大幅减少。

### 2.3 句间连贯性预测（Sentence Ordering Prediction）

ALBERT 提出了一个新的预训练任务——句间连贯性预测（SOP）：

- **原版 NSP（Next Sentence Prediction）**：预测 B 是否是 A 的下一句（二分类）
- **改进 SOP**：预测 A 和 B 的顺序是否正确（二分类）

SOP 任务要求模型理解句子之间的逻辑顺序和语义关系，比 NSP 更具挑战性，能更好地学习句间关系。

---

## 3. 数学公式与推导

### 3.1 因子分解嵌入

设词表大小为 $V$，隐藏维度为 $H$，因子分解维度为 $E$：

**原版嵌入层**：
$$W_{orig} \in \mathbb{R}^{V \times H}, \quad \text{参数量} = V \times H$$

**因子分解后**：
$$W_1 \in \mathbb{R}^{V \times E}, \quad W_2 \in \mathbb{R}^{E \times H}$$
$$\text{参数量} = V \times E + E \times H$$

参数节省比例：
$$\frac{V \times E + E \times H}{V \times H} = \frac{E}{H} + \frac{E}{V}$$

当 $E=128, H=768, V=30000$ 时：
$$\frac{128}{768} + \frac{128}{30000} \approx 0.167 + 0.004 = 17.1\%$$

### 3.2 自注意力机制

标准自注意力计算（以单头为例）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q = XW^Q$：查询矩阵
- $K = XW^K$：键矩阵  
- $V = XW^V$：值矩阵
- $d_k$：键维度

多层情况下，每层 $l$ 都需要独立的参数矩阵 $W^Q_l, W^K_l, W^V_l$。跨层参数共��后，所有层使用同一组参数。

### 3.3 预训练损失函数

**MLM 损失（Masked Language Modeling）**：
$$L_{MLM} = -\sum_{i \in M} \log P(x_i | x_{\setminus i})$$

**SOP 损失（Sentence Order Prediction）**：
$$L_{SOP} = -\sum \log P(y | x_A, x_B)$$

**总损失**：
$$L_{total} = L_{MLM} + \lambda \cdot L_{SOP}$$

其中 $\lambda$ 通常取 0.5。

---

## 4. 训练过程讲解

### 4.1 预训练任务

**任务一：MLM（遮蔽语言模型）**

```
Input: "The [MASK] runs fast"
Target: "The [MASK] runs fast" -> "cat"
```

1. 随机遮蔽 15% 的 token
2. 80% 替换为 [MASK]
3. 10% 替换为随机 token
4. 10% 保持不变
5. 预测被遮蔽的 token

**任务二：SOP（句间顺序预测）**

```
Input: Sentence A + [SEP] + Sentence B
Label: 正确顺序/错误顺序
```

1. 从文档中连续选取两个句子作为正样本
2. 交换顺序作为负样本
3. 二分类预测顺序是否正确

### 4.2 下游任务微调

**分类任务**：

```python
from transformers import AlbertForSequenceClassification

model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
outputs = model(input_ids, attention_mask, labels=labels)
```

**问答任务**：

```python
from transformers import AlbertForQuestionAnswering

model = AlbertForQuestionAnswering.from_pretrained('albert-base-v2')
outputs = model(input_ids, attention_mask, start_positions=start, end_positions=end)
```

### 4.3 训练配置

| 参数 | 小规模 | 中规模 | 大规模 |
|------|--------|--------|--------|
| Batch Size | 64 | 256 | 512 |
| Learning Rate | 1e-4 | 5e-5 | 3e-5 |
| Epochs | 3-5 | 3-5 | 3-5 |
| Warmup Steps | 10% | 10% | 10% |

---

## 5. 应用场景

### 5.1 典型应用

- **文本分类**：情感分析、主题分类
- **问答系统**：机器阅读理解
- **自然语言推理**：句对关系判断
- **序列标注**：命名实体识别、词性标注

### 5.2 代码示例

```python
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

# 加载模型
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')

# 文本分类
text = "This is a great movie!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=-1)
print(f"预测类别: {pred.item()}")
```

---

## 6. 优缺点分析

### 6.1 优点

1. **参数效率高**：参数量仅为 BERT 的约 1/10
2. **训练速度快**：约 3 倍速度提升
3. **内存占用小**：可在单卡训练
4. **性能接近**：在多项任务上性能接近 BERT-Large
5. **易于部署**：模型体积小，适合移动端

### 6.2 缺点

1. **共享参数限制**：跨层共享导致表达能力下降
2. **推理速度相近**：参数量减少但计算量不变
3. **微调效果不稳**：有时比 BERT 效果略差

### 6.3 改进方向

- **ALBERT-xxlarge**：更大隐藏维度的版本
- **ELECTRA**：替换预训练任务为替换检测
- **RoBERTa**：移除参数共享，更多训练数据

---

## 7. 调库实现

### 7.1 HuggingFace 实现

```python
from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertForSequenceClassification

# 加载 tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# 加载 MLM 模型
mlm_model = AlbertForMaskedLM.from_pretrained('albert-base-v2')

# 示例：完形填空
text = "The [MASK] is running on the street."
inputs = tokenizer(text, return_tensors='pt')
outputs = mlm_model(**inputs)
pred = torch.argmax(outputs.logits, dim=-1)
predicted_word = tokenizer.decode(pred[0])
print(f"预测词: {predicted_word}")

# 加载分类模型
clf_model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)

# 文本分类
texts = ["This is a great movie!", "This movie is terrible."]
labels = [1, 0]
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
inputs['labels'] = torch.tensor(labels)

outputs = clf_model(**inputs)
loss = outputs.loss
logits = outputs.logits
print(f"损失: {loss.item()}")
print(f"预测: {torch.argmax(logits, dim=-1).tolist()}")
```

### 7.2 下游任务微调

```python
# 文本分类微调
from transformers import AlbertForSequenceClassification, AdamW
from torch.utils.data import DataLoader

# 加载模型
model = AlbertForSequenceClassification.from_pretrained(
    'albert-base-v2',
    num_labels=2
)
model.train()

# 优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练循环
for epoch in range(3):
    for batch in dataloader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True)
        inputs['labels'] = batch['label']
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    print(f"Epoch {epoch}: Loss = {loss.item()}")
```

---

## 8. 手工代码实现

### 8.1 简化版 ALBERT 架构

```python
import torch
import torch.nn as nn
import numpy as np

class FactorizedEmbedding(nn.Module):
    """因子分解嵌入层"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed_proj = nn.Embedding(vocab_size, embed_dim)
        self.hidden_proj = nn.Linear(embed_dim, hidden_dim)
        
    def forward(self, x):
        """将词 ID 映射到隐藏空间"""
        x = self.embed_proj(x)  # (batch, seq, embed_dim)
        x = self.hidden_proj(x)  # (batch, seq, hidden_dim)
        return x


class SharedAttention(nn.Module):
    """共享自注意力机制"""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Q, K, V 投影（所有层共享）
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.output(attention_output)


class SharedFFN(nn.Module):
    """共享前馈网络"""
    
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, ffn_dim)
        self.dense2 = nn.Linear(ffn_dim, hidden_dim)
        
    def forward(self, x):
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        return x


class AlbertLayer(nn.Module):
    """ALBERT 单层（注意力 + 前馈）"""
    
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.attention = SharedAttention(hidden_dim, num_heads)
        self.ffn = SharedFFN(hidden_dim, ffn_dim)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差
        attn_output = self.attention(self.layernorm1(x), attention_mask)
        x = x + attn_output
        
        # 前馈网络 + 残差
        ffn_output = self.ffn(self.layernorm2(x))
        x = x + ffn_output
        
        return x


class AlbertLite(nn.Module):
    """简化版 ALBERT"""
    
    def __init__(self, vocab_size=30000, embed_dim=128, hidden_dim=768, 
                 num_layers=12, num_heads=12, ffn_dim=3072):
        super().__init__()
        
        # 因子分解嵌入
        self.embedding = FactorizedEmbedding(vocab_size, embed_dim, hidden_dim)
        
        # 共享的 ALBERT 层
        self.encoder_layer = AlbertLayer(hidden_dim, num_heads, ffn_dim)
        
        # 所有层共享同一个 encoder_layer
        self.layers = nn.ModuleList([self.encoder_layer] * num_layers)
        
        # 输出层
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 嵌入
        x = self.embedding(input_ids)
        
        # 编码器（所有层共享参数）
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        # 输出
        x = self.layernorm(x)
        logits = self.classifier(x[:, 0])  # [CLS] 位置
        
        return logits


def demo():
    print("=== ALBERT 手工实现演示 ===\n")
    
    model = AlbertLite(
        vocab_size=30000,
        embed_dim=128,
        hidden_dim=768,
        num_layers=12,
        num_heads=12
    )
    
    # 模拟输入
    input_ids = torch.randint(0, 30000, (2, 10))
    attention_mask = torch.ones(2, 10)
    
    # 前向传播
    logits = model(input_ids, attention_mask)
    
    print(f"输入 shape: {input_ids.shape}")
    print(f"输出 shape: {logits.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    
if __name__ == "__main__":
    demo()
```

---

## 9. 可视化与结果理解

### 9.1 参数分布可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_params():
    """可视化 ALBERT vs BERT 参数分布"""
    
    models = ['BERT-Base', 'BERT-Large', 'ALBERT-Base', 'ALBERT-Large']
    params = [108, 340, 12, 60]  # 百万参数
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, params, color=['steelblue', 'steelblue', 'coral', 'coral'])
    
    plt.ylabel('参数量 (百万)')
    plt.title('BERT vs ALBERT 参数量对比')
    
    for bar, param in zip(bars, params):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{param}M', ha='center')
    
    plt.tight_layout()
    plt.savefig('albert_params.png', dpi=150)
    plt.show()


def visualize_performance():
    """可视化性能对比"""
    
    tasks = ['SQuAD', 'MNLI', 'SST-2']
    bert_scores = [88.5, 84.5, 93.3]
    albert_scores = [88.1, 84.3, 93.2]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, bert_scores, width, label='BERT-Large', color='steelblue')
    plt.bar(x + width/2, albert_scores, width, label='ALBERT-Large', color='coral')
    
    plt.ylabel('准确率 (%)')
    plt.title('BERT vs ALBERT 性能对比')
    plt.xticks(x, tasks)
    plt.legend()
    plt.ylim(80, 95)
    
    plt.tight_layout()
    plt.savefig('albert_perf.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 下游任务评估

```python
from transformers import AlbertForSequenceClassification, AlbertTokenizer
import torch
from sklearn.metrics import accuracy_score, classification_report

# 加载模型
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')

# 评估函数
def evaluate(dataset):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for text, label in dataset:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)
            labels.append(label)
    
    accuracy = accuracy_score(labels, predictions)
    return accuracy, classification_report(labels, predictions)
```

### 10.2 GLUE 基准测试

| 任务 | BERT-Large | ALBERT-Large |
|------|------------|--------------|
| CoLA | 60.6 | 61.3 |
| SST-2 | 93.3 | 93.2 |
| MRPC | 90.2 | 90.5 |
| STS-B | 91.2 | 91.3 |
| MNLI | 86.3 | 86.5 |
| QNLI | 93.5 | 93.7 |

---

## 11. 常见问题与易错点

### 11.1 参数共享的影响

**问题**：跨层参数共享是否会影响模型表达能力？

**解答**：会有轻微影响。实验表明，参数共享会导致约 1-2% 的性能下降，但参数量减少 10 倍以上，是值得的权衡。

### 11.2 嵌入维度选择

**问题**：如何选择因子分解维度 E？

**解答**：经验公式 $E = \sqrt{H}$ 或取 128。E 太小会损失信息，太大则省参效果不明显。

### 11.3 训练技巧

1. **学习率**：使用较小学习率（2e-5 ~ 5e-5）
2. **Warmup**：前 10% steps 线性warmup
3. **梯度裁剪**：max_norm=1.0

---

## 12. 学习总结

**核心要点**：

1. **因子分解嵌入**：将 $V \times H$ 分解为 $V \times E + E \times H$
2. **跨层参数共享**：所有层共享同一组注意力参数和 FFN 参数
3. **SOP 预训练**：预测句子顺序而非简单预测下一句
4. **参数量对比**：ALBERT-Large 仅 6000 万，BERT-Large 达 3.4 亿

**学习建议**：

1. 先掌握 BERT 原理
2. 对比理解 ALBERT 的三项改进
3. 在下游任务上微调验证

---

## 13. 练习题与思考题

### 13.1 基础练习

1. 计算 BERT-Base 和 ALBERT-Base 的参数量差异
2. 因子分解嵌入的参数节省公式推导
3. SOP 与 NSP 的区别

### 13.2 进阶练习

1. 手动实现因子分解嵌入层
2. 实现共享注意力机制
3. 在 IMDB 数据集上微调 ALBERT

### 13.3 思考题

1. ALBERT 的参数共享策略有哪些变体？各有什么优劣？
2. 如何进一步改进 ALBERT？

---

### 13.4 详细答案与解析

#### 练习1：参数量计算

**问题**：计算 BERT-Base 和 ALBERT-Base 的参数量。

**答案**：

- BERT-Base：1.08 亿（108M）
- ALBERT-Base：1200 万（12M）

**解析**：

**BERT-Base 参数量**：
- 嵌入层：30000 × 768 = 2300 万
- 12 层 transformer：12 × (768×768×4 + 3072×768×2) ≈ 8500 万
- 总计：约 1.08 亿

**ALBERT-Base 参数量**：
- 因子分解嵌入：30000 × 128 + 128 × 768 = 390 万
- 共享层参数：768×768×4 + 3072×768×2 ≈ 700 万
- 应用 12 次：约 840 万
- 再加上输出层：约 1200 万

#### 练习2：因子分解公式

**问题**：推导因子分解的参数节省比例。

**答案**：节省比例 = $1 - \frac{E}{H} - \frac{E}{V}$

**解析**：

原��参��量：$V \times H$

因子分解后：$V \times E + E \times H = E(V + H)$

节省比例：
$$\frac{V \times H - E(V + H)}{V \times H} = 1 - \frac{E}{H} - \frac{E}{V}$$

当 $E=128, H=768, V=30000$：
$$1 - \frac{128}{768} - \frac{128}{30000} = 1 - 0.167 - 0.004 = 82.9\%$$

约节省 83% 参数。

#### 练习3：SOP vs NSP

**问题**：SOP 和 NSP 有什么区别？为什么 SOP 更好？

**答案**：

- NSP：二分类，判断 B 是否是 A 的下一个句子。正样本：A 的真实下一句；负样本：随机句子
- SOP：二分类，判断 A 和 B 的顺序是否正确。正样本：正确顺序；负样本：交换顺序

**SOP 更好的原因**：

1. 更难，需要理解句间逻辑关系
2. 避免将"主题预测"和"连贯性预测"混淆
3. 更接近下游任务的实际需求

---

## 14. 学习路径建议

**入门阶段（1-2周）**：

1. 学习 Transformer 架构 → 理解自注意力
2. 学习 BERT 原理 → 理解 MLM + NSP 预训练
3. 学习 ALBERT 改进 → 理解三项核心技术

**进阶阶段（2-3周）**：

1. 实践 HuggingFace ALBERT 微调
2. 对比不同参数共享策略
3. 在 GLUE 上评估性能

**高级阶段**：

1. 改进预训练任务（ELECTRA）
2. 知识蒸馏（DistilBERT）
3. 高效推理（ALBERT 量化）

**推荐学习路线**：

```
Transformer → BERT → ALBERT → RoBERTa → ELECTRA → SpanBERT
```