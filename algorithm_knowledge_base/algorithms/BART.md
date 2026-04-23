# BART 学习文档

> 序列到序列预训练的去噪自编码器，结合 BERT 和 GPT 的优势。

---

## 1. 算法基础认知

### 1.1 发展背景

BART（Bidirectional and Auto-Regressive Transformers）由 Facebook AI 于 2019 年提出，发表在论文《BART: Denoising Sequence-to-Sequence Pre-training》。BART 创新性地结合了 BERT 的双向编码器和 GPT 的自回归解码器，成为文本生成任务的主流预训练模型。

### 1.2 核心定位

| 模型 | 架构 | 预训练任务 | 适用场景 |
|------|------|-----------|----------|
| BERT | 双向 Encoder | MLM | 理解任务 |
| GPT | 单向 Decoder | AR LM | 生成任务 |
| BART | Encoder-Decoder | 去噪自编码 | sequence-to-sequence |

### 1.3 模型系列

| 模型 | 参数 | 层数 | 隐藏维度 | 注意力头数 |
|------|------|------|----------|-------------|
| BART-base | 139M | 12 | 768 | 16 |
| BART-large | 400M | 12 | 1024 | 16 |

---

## 2. 核心原理

### 2.1 编码器-解码器架构

BART 采用标准的 Encoder-Decoder 架构：

- **编码器**：使用双向自注意力，类似 BERT
- **解码器**：使用遮蔽自注意力，类似 GPT，但可以关注编码器输出

```
输入文本 → 编码器 (双向) → 中间表示 → 解码器 (自回归) → 输出文本
```

### 2.2 去噪预训练任务

BART 的核心创新是**去噪自编码**：

1. **文本损坏**：对输入进行各种变换
2. **编码**：通过编码器处理损坏文本
3. **重建**：解码器自回归恢复原始文本

### 2.3 五种文本损坏策略

| 策略 | 说明 | 示例 |
|------|------|------|
| Token Masking | 随机遮蔽 token | "BART is [M] AI tool" |
| Token Deletion | 删除随机 token | "BART is AI tool" |
| Text Infilling | 遮蔽一段文本 | "BART [M...M] tool" |
| Sentence Permutation | 打乱句子顺序 | "tool is BART AI" |
| Document Rotation | 旋转文档起始 | "is BART AI tool" |

### 2.4 与其他模型对比

- **BART vs BERT**：BART 更适合生成任务
- **BART vs GPT**：BART 可以利用双向上下文
- **BART vs T5**：结构类似，但预训练任务不同

---

## 3. 数学公式与推导

### 3.1 编码器前向传播

给定输入序列 $x$ 和损坏序列 $\tilde{x}$：

$$h_t = \text{Encoder}(\tilde{x})$$

编码器使用双向自注意力：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

### 3.2 解码器自回归生成

解码器逐 token 生成：
$$P(x_t | x_{<t}, h) = \text{softmax}(W \cdot h_t)$$

损失函数：
$$L = -\sum_t \log P(x_t | x_{<t}, h)$$

### 3.3 序列到序列损失

对于 sequence-to-sequence 任务：

$$L = -\sum_{i} \log P(y_i | y_{<i}, x)$$

其中 $x$ 是源序列，$y$ 是目标序列。

### 3.4 注意力掩码

**编码器**：无掩码（双向）
**解码器**：上三角掩码（防止看到未来）

$$\text{Mask}_{i,j} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$$

---

## 4. 训练过程讲解

### 4.1 预训练阶段

```
Input: 原始文本corpus
Output: 预训练BART模型

1. 文本损坏:
   - 随机选择损坏策略
   - 应用到batch中每个文本
2. 编码:
   - 损坏文本输入Encoder
   - 得到中间表示
3. 解码:
   - 解码器自回归生成
   - 计算交叉熵损失
4. 反向传播更新参数
5. 重复直到收敛
```

### 4.2 微调阶段

**序列生成任务**（如摘要、翻译）：

```python
# 微调BART
outputs = model(
    input_ids=source_ids,
    decoder_input_ids=target_ids
)
loss = outputs.loss
loss.backward()
```

**分类任务**（如问答）：

```python
# 使用Encoder输出做分类
encoder_hidden = model.encoder(input_ids)
logits = classifier(encoder_hidden)
```

### 4.3 推理

使用束搜索或贪婪解码：

```python
# 贪婪解码
generated = model.generate(input_ids, max_length=100)

# 束搜索
generated = model.generate(
    input_ids, 
    num_beams=5,
    max_length=100
)
```

---

## 5. 应用场景

### 5.1 典型应用

- **文本摘要**：生成文章摘要
- **机器翻译**：序列到序列翻译
- **问答系统**：阅读理解
- **文本风格转换**：改写生成

### 5.2 HuggingFace 使用

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# 加载模型
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# 文本生成
inputs = tokenizer("BART is a pre-trained model.", return_tensors="pt")
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

---

## 6. 优缺点分析

### 6.1 优点

1. **统一框架**：支持理解和生成任务
2. **去噪预训练**：强大的表示学习
3. **灵活架构**：Encoder-Decoder 分离
4. **多任务适应**：可以通过微调适应各种任务

### 6.2 缺点

1. **生成速度慢**：自回归解码
2. **训练复杂**：需要更多计算资源
3. **长度外推**：难以生成长序列

### 6.3 改进方向

- **BART-sum**：专门的摘要模型
- **mBART**：多语言版本
- **BART+DAPO**：强化学习微调

---

## 7. 调库实现

### 7.1 预训练模型使用

```python
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartModel

class BART:
    """BART 序列到序列模型
    
    参数:
        model_name: 模型名称
        max_length: 最大生成长度
    """
    
    def __init__(self, model_name='facebook/bart-base'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
        # 条件生成模型
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # 基础模型（无解码器）
        self.base_model = BartModel.from_pretrained(model_name)
        
        self.max_length = 100
        
    def generate(self, text, num_beams=1, early_stopping=True):
        """文本生成
        
        参数:
            text: 输入文本
            num_beams: 束搜索宽度
            early_stopping: 提前停止
        """
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        
        summary_ids = self.model.generate(
            inputs['input_ids'],
            num_beams=num_beams,
            max_length=self.max_length,
            early_stopping=early_stopping
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    def encode(self, text):
        """编码文本到表示"""
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.base_model.encoder(inputs['input_ids'])
        return outputs.last_hidden_state
    
    def finetune(self, source_texts, target_texts, epochs=3, lr=2e-5):
        """微调模型
        
        参数:
            source_texts: 源文本列表
            target_texts: 目标文本列表
            epochs: 训练轮数
            lr: 学习率
        """
        # 编码
        inputs = self.tokenizer(source_texts, return_tensors='pt', padding=True, truncation=True)
        targets = self.tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True)
        
        # 设置decoder_input_ids
        decoder_input_ids = targets['input_ids']
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        
        # 训练
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=decoder_input_ids,
                labels=targets['input_ids']
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        return self


def demo():
    """BART 演示"""
    print("=== BART 去噪自编码器演示 ===\n")
    
    # 加载模型
    bart = BART('facebook/bart-base')
    
    # 文本生成
    text = "BART is a pre-trained sequence-to-sequence model developed by Facebook AI."
    generated = bart.generate(text)
    print(f"输入: {text}")
    print(f"生成: {generated}")
    
    # 编码
    encoding = bart.encode(text)
    print(f"\n编码维度: {encoding.shape}")
    
    return bart


if __name__ == "__main__":
    demo()
```

### 7.2 文本摘要应用

```python
def summarize_article(article, model, max_length=142):
    """
    文章摘要生成
    
    参数:
        article: 文章文本
        model: BART模型
        max_length: 最大生成长度
    """
    inputs = model.tokenizer(article, return_tensors='pt', max_length=1024, truncation=True)
    
    summary_ids = model.model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=max_length,
        early_stopping=True
    )
    
    return model.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

---

## 8. 手工代码实现

### 8.1 简化 BART 架构

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class BARTEncoder(nn.Module):
    """BART 编码器"""
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(768)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
            
        return x


class BARTDecoder(nn.Module):
    """BART 解码器"""
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_model*4)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, memory):
        x = self.embedding(x) * math.sqrt(768)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, memory)
            
        return x


class SimpleBART(nn.Module):
    """简化版 BART
    
    参数:
        vocab_size: 词表大小
        d_model: 隐藏维度
        n_layers: 层数
        n_heads: 注意力头数
    """
    
    def __init__(self, vocab_size=50264, d_model=768, n_layers=6, n_heads=12):
        super().__init__()
        
        self.encoder = BARTEncoder(vocab_size, d_model, n_layers, n_heads)
        self.decoder = BARTDecoder(vocab_size, d_model, n_layers, n_heads)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt):
        """前向传播"""
        # 编码
        memory = self.encoder(src)
        
        # 解码
        output = self.decoder(tgt, memory)
        
        # 投影到词表
        logits = self.output_projection(output)
        
        return logits
    
    def generate(self, src, max_len=100):
        """自回归生成（简化）"""
        self.eval()
        
        with torch.no_grad():
            # 编码
            memory = self.encoder(src)
            
            # 解码
            generated = torch.zeros(1, 1, dtype=torch.long)
            
            for _ in range(max_len):
                output = self.decoder(generated, memory)
                next_token = output[:, -1:].argmax()
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == 2:  # EOS token
                    break
                    
        return generated


def demo_manual():
    """BART 手工实现演示"""
    print("=== BART 手工实现演示 ===\n")
    
    # 模型参数
    vocab_size = 10000
    d_model = 256
    n_layers = 3
    n_heads = 4
    
    # 创建模型
    bart = SimpleBART(vocab_size, d_model, n_layers, n_heads)
    
    # 模拟输入
    src = torch.randint(0, vocab_size, (1, 20))
    tgt = torch.randint(0, vocab_size, (1, 10))
    
    # 前向传播
    logits = bart(src, tgt)
    
    print(f"输入 shape: {src.shape}")
    print(f"目标 shape: {tgt.shape}")
    print(f"输出 shape: {logits.shape}")
    print(f"\n参数量: {sum(p.numel() for p in bart.parameters()):,}")


if __name__ == "__main__":
    demo_manual()
```

---

## 9. 可视化与结果理解

### 9.1 模型架构可视化

```python
def visualize_bart_architecture():
    """可视化 BART 架构"""
    
    print("""
    BART 架构:
    
    输入序列: [x1, x2, x3, x4, x5]
                    ↓
              ┌─────┴─────┐
              │ Encoder  │
              │(双向注意力)│
              └─────┬─────┘
                    ↓
              ┌─────┴─────┐
              │ Cross    │
              │Attention │
              └─────┬─────┘
                    ↓
              ┌─────┴─────┐
              │ Decoder  │
              │(单向注意力│
              └─────┬─────┘
                    ↓
              ┌─────┴─────┐
              │  Output │
              └────────┘
    """)
```

### 9.2 生成质量对比

```python
def compare_decoding_methods():
    """对比不同解码方法"""
    print("""
    解码方法对比:
    
    方法        速度    质量    多���性
    ──────────────────────────────
    Greedy     快      低     低
    Beam      中      中     中
    Top-K     中      中     高
    Nucleus   慢      高     高
    """)
```

---

## 10. 模型评估

### 10.1 生成质量评估

```python
from sklearn.metrics import precision_score, recall_score

def evaluate_generation(reference, hypothesis):
    """评估生成质量"""
    
    # BLEU 分数
    def compute_bleu(reference, hypothesis):
        """简化 BLEU 计算"""
        from collections import Counter
        
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if len(hyp_words) == 0:
            return 0.0
            
        common = Counter(ref_words) & Counter(hyp_words)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
            
        precision = num_same / len(hyp_words)
        
        return precision
    
    bleu = compute_bleu(reference, hypothesis)
    
    return {'BLEU': bleu}
```

### 10.2 下游任务基准

| 任务 | BART-base | BART-large | 其他模型 |
|------|-----------|------------|----------|
| CNN-DM | 19.5 | 21.5 | T5: 20.5 |
| XSum | 20.0 | 21.7 | T5: 21.0 |
| IMDb | 8.7 | 8.6 | BERT: 9.1 |

---

## 11. 常见问题与易错点

### 11.1 生成重复

**问题**：生成内容重复

**解决方案**：
- 使用 n-gram 惩罚
- 增加 Top-K 或 Nucleus 采样
- 使用重复惩罚

```python
# 重复惩罚
generated = model.generate(
    input_ids,
    repetition_penalty=1.2
)
```

### 11.2 长度控制

**问题**：生成长度难以控制

**解答**：
- 设置 max_length
- 使用长度惩罚
- 后处理截断

### 11.3 微调技巧

1. **学习率**：2e-5 ~ 5e-5
2. **Epochs**：3-10
3. **Batch**：根据显存

---

## 12. 学习总结

**核心要点**：

1. **编码器-解码器**：标准 Transformer 架构
2. **去噪预训练**：多种文本损坏策略
3. **序列生成**：自回归解码生成
4. **灵活应用**：微调适应各种任务

**学习建议**：

1. 掌握 Transformer 架构
2. 理解去噪预训练
3. 实践文本生成任务

---

## 13. 练习题与思考题

### 13.1 基础练习

1. BART 与 BERT 的区别
2. 去噪预训练的优势
3. 编码器-解码器架构

### 13.2 进阶练习

1. 实现 BART 文本摘要
2. 对比不同解码方法

### 13.3 思考题

1. BART 适合哪些任务？
2. 如何改进 BART？

---

### 13.4 详细答案与解析

#### 练习1：BART vs BERT

**问题**：BART 与 BERT 的核心区别

**答案**：

| 特性 | BERT | BART |
|------|------|------|
| 架构 | Encoder only | Encoder-Decoder |
| 预训练 | MLM | 去噪自编码 |
| 预训练目标 | 重建遮蔽token | 重建原始文本 |
| 生成能力 | 无 | 有 |
| 应用 | 理解任务 | 生成任务 |

#### 练习2：去噪优势

**问题**：为什么去噪预训练效果好？

**解答**：

1. **任务多样性**：5种不同损坏策略
2. **目标更难**：需要理解整体结构
3. **泛化强**：适应各种下游任务

---

## 14. 学习路径建议

### 入门阶段

1. 学习 Transformer 架构
2. 理解 BERT 和 GPT
3. 掌握 BART 原理

### 进阶阶段

1. 实践文本摘要
2. 微调 BART
3. 对比解码方法

### 高级阶段

1. 改进预训练任务
2. 多语言 BART
3. 强化学习微调

**推荐路线**：

```
Transformer → BERT → GPT → BART → T5 → ChatGPT
```

**BART 是序列到序列模型的基础，掌握它对学习文本生成很重要。**