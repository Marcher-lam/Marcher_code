# ALBERT（A Lite BERT）学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
ALBERT（A Lite BERT）是Google于2019年提出的轻量级BERT变体，通过**跨层参数共享**和**嵌入矩阵因子分解**，在大幅减少参数量的同时保持接近BERT的性能，使在有限计算资源下训练更大模型成为可能。

### 1.2 直觉类比
ALBERT就像一个"武林高手"的内功修炼方式——普通门派（BERT）给每个弟子（Transformer层）各发一本秘籍（独立参数），门派越大越占地方；ALBERT只发一本秘籍，所有弟子共用（参数共享），而且把秘籍分成上下册（因子分解），使得门派可以招收更多弟子（堆更多层）而仓库不爆炸（显存可控）。

### 1.3 历史背景
- **2019年9月**：Google提出ALBERT
- **2020年**：发布ALBERT-xxlarge（12层共享，参数量仅223M）
- **意义**：展示了参数共享的巨大潜力，推动了轻量化预训练模型发展

### 1.4 算法定位
ALBERT属于**轻量级自编码语言模型**，专注于参数效率优化。

---

## 2. 核心原理

### 2.1 嵌入矩阵因子分解
BERT的嵌入矩阵 $E \in \mathbb{R}^{V \times H}$ 中 $V$ 很大（~30000），$H$ 也大（~1024），导致嵌入层参数量巨大。ALBERT将其分解为两个小矩阵：

$$E \in \mathbb{R}^{V \times d}, \quad W \in \mathbb{R}^{d \times H}$$

其中 $d \ll H$（通常 $d=128$）。

参数量从 $V \times H$ 减少到 $V \times d + d \times H$。例如：
- BERT-base: $30000 \times 768 = 23M$ 参数
- ALBERT: $30000 \times 128 + 128 \times 768 = 3.9M + 0.1M = 4M$ 参数（减少~83%）

### 2.2 跨层参数共享
ALBERT共享所有Transformer层的参数：

**三种共享方案**：
1. 仅共享自注意力参数（效果最好）
2. 仅共享FFN参数
3. **全部共享**（ALBERT最终采用）

$$W_{layer1} = W_{layer2} = ... = W_{layerN}$$

这意味着12层Transformer共享同一套权重，相当于计算了12次但参数只有1层的量。

### 2.3 句序预测（SOP）替代NSP
ALBERT发现BERT的NSP任务太简单（主题+主题判断），用SOP（Sentence Order Prediction）替代：

- **正样本**：连续的句子对（A→B）
- **负样本**：交换顺序的句子对（B→A）

SOP迫使模型理解句子间的连贯性和逻辑顺序。

---

## 3. 数学公式与推导

### 3.1 因子分解的数学形式
输入词 $w_i$ 的嵌入向量：

$$e_i = W \cdot E[w_i]$$

其中 $E[w_i] \in \mathbb{R}^d$ 是词 $w_i$ 在低维空间的表示，$W \in \mathbb{R}^{d \times H}$ 将其投影到隐藏维度。

### 3.2 参数共享的效果
设Transformer层数为 $N$，单层参数为 $\theta$：
- BERT: $N \times \theta$ 个独立参数
- ALBERT: $\theta$ 个共享参数

参数缩减率：$\frac{1}{N}$。对于12层模型，参数减少12倍。

### 3.3 SOP损失函数
$$L_{SOP} = -\mathbb{E}_{(A,B)\sim D} \log P(\text{label} | A, B)$$

其中 $\text{label} = 1$ 表示 $B$ 是 $A$ 的下一句，$\text{label} = 0$ 表示顺序被交换。

### 3.4 MLM损失（与BERT一致）
$$L_{MLM} = -\mathbb{E}_{x\sim D} \sum_{i\in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})$$

ALBERT的MLM与BERT完全相同。

---

## 4. 训练过程讲解

### 4.1 训练步骤
1. **数据准备**：BookCorpus + Wikipedia
2. **因子分解**：将词映射到低维空间（d=128）再投影到高维（H=768/1024）
3. **MLM**：随机mask 15%的token，预测被mask的词
4. **SOP**：判断两个句子的顺序是否正确
5. **参数共享**：所有Transformer层使用相同参数，但每层独立计算
6. **联合优化**：$L = L_{MLM} + L_{SOP}$

### 4.2 与BERT训练过程对比
| 方面 | BERT | ALBERT |
|------|------|--------|
| 嵌入层 | V×H | V×d + d×H |
| Transformer层 | 每层独立 | 全部共享 |
| 句子关系 | NSP（主题判断） | SOP（顺序判断） |
| 内存占用 | 高 | 低（约1/3） |
| 训练速度 | 标准 | 略快（参数更少，通信量小） |

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 资源受限设备 | 移动端、边缘设备 |
| 大规模模型部署 | 数百个模型并行服务 |
| 快速实验 | 参数少，训练和调参快 |
| 大模型预训练 | ALBERT-xxlarge可在单卡训练 |

---

## 6. 优缺点分析

### 优点
1. **参数量大幅减少**：BERT-base的~1/10
2. **显存友好**：可在消费级GPU上训练
3. **可扩展性**：容易堆叠更多层（ALBERT-xxlarge 12层 vs BERT-large 24层参数更少）
4. **SOP更有效**：比NSP提供更有意义的训练信号

### 缺点
1. **性能略有下降**：参数共享导致表示能力降低
2. **推理速度未提升**：虽然参数少，但计算量不变（层数不变）
3. **大模型训练不稳定**：深层共享参数可能梯度异常
4. **嵌入瓶颈**：$d=128$ 可能不足以编码复杂语义

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel, AlbertTokenizer, AlbertForPreTraining
import math

class ALBERTClassifier(nn.Module):
    """ALBERT文本分类器"""
    def __init__(self, model_name='albert-base-v2', num_classes=2):
        super().__init__()
        self.albert = AlbertModel.from_pretrained(model_name)
        self.config = self.albert.config
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # ALBERT使用pooler_output
        pooled = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled)
        return logits


class ALBERTWithFactorizedEmbedding(nn.Module):
    """带嵌入因子分解的ALBERT"""
    def __init__(self, vocab_size=30000, embedding_dim=128, hidden_dim=768,
                 num_layers=12, num_heads=12):
        super().__init__()
        # 因子分解嵌入层
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # 位置嵌入（不可分解，因为位置数少）
        self.position_embedding = nn.Embedding(512, hidden_dim)
        
        # 共享的Transformer层
        self.shared_layer = TransformerLayer(hidden_dim, num_heads)
        self.num_layers = num_layers
        
        # 输出层
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),  # 先投影到低维
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, vocab_size),
        )
        
        # SOP分类头
        self.sop_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        B, L = input_ids.shape
        
        # 因子分解嵌入
        word_emb = self.word_embedding(input_ids)  # [B, L, d]
        word_emb = self.embedding_projection(word_emb)  # [B, L, H]
        
        pos_emb = self.position_embedding(
            torch.arange(L, device=input_ids.device).unsqueeze(0)
        )
        
        x = word_emb + pos_emb
        
        # 共享层多次计算
        for _ in range(self.num_layers):
            x = self.shared_layer(x, attention_mask)
        
        x = self.norm(x)
        
        mlm_logits = self.mlm_head(x)
        sop_logits = self.sop_head(x[:, 0])  # [CLS]
        
        return mlm_logits, sop_logits


class TransformerLayer(nn.Module):
    """单Transformer层（被ALBERT共享）"""
    def __init__(self, d_model, nhead, d_ff=3072, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class ALBERTFineTuner:
    """ALBERT微调器"""
    def __init__(self, model_name='albert-base-v2'):
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = AlbertForPreTraining.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def predict_sop(self, sentence_a, sentence_b):
        """句序预测"""
        inputs = self.tokenizer(
            sentence_a, sentence_b,
            return_tensors='pt',
            truncation=True,
            padding=True
        ).to(self.device)
        
        outputs = self.model(**inputs)
        sop_logits = outputs.sop_logits
        pred = sop_logits.argmax(dim=-1).item()
        
        return "正确顺序" if pred == 0 else "顺序错误"


def test_albert():
    """测试ALBERT"""
    model = ALBERTClassifier(num_classes=3)
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    
    texts = ["This is great!", "I don't like it.", "It's okay."]
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        logits = model(inputs.input_ids, inputs.attention_mask)
        pred = logits.argmax(dim=-1).item()
        print(f"'{text}' → {pred}")
    
    # 测试因子分解嵌入
    factorized_model = ALBERTWithFactorizedEmbedding()
    ids = torch.randint(0, 3000, (2, 16))
    mlm_out, sop_out = factorized_model(ids)
    print(f"MLM输出: {mlm_out.shape}, SOP输出: {sop_out.shape}")
    
    print("ALBERT测试通过！")

if __name__ == "__main__":
    test_albert()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandwrittenALBERT(nn.Module):
    """ALBERT核心逻辑手工实现"""
    def __init__(self, vocab_size=30000, embedding_dim=128, hidden_dim=768,
                 nhead=12, num_layers=12):
        super().__init__()
        # 因子分解嵌入
        self.word_emb = nn.Embedding(vocab_size, embedding_dim)
        self.emb_proj = nn.Linear(embedding_dim, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)
        
        # 共享Transformer层
        self.shared_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.shared_norm1 = nn.LayerNorm(hidden_dim)
        self.shared_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.shared_norm2 = nn.LayerNorm(hidden_dim)
        
        self.num_layers = num_layers
        
        # 预测头
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, vocab_size),
        )
        
    def forward(self, input_ids):
        B, L = input_ids.shape
        
        # 嵌入
        x = self.emb_proj(self.word_emb(input_ids))
        x = x + self.pos_emb(torch.arange(L, device=input_ids.device).unsqueeze(0))
        
        # 共享层多次前向
        for _ in range(self.num_layers):
            residual = x
            attn_out, _ = self.shared_attn(x, x, x)
            x = self.shared_norm1(residual + attn_out)
            residual = x
            ffn_out = self.shared_ffn(x)
            x = self.shared_norm2(residual + ffn_out)
        
        return self.mlm_head(x)


def test_handwritten():
    model = HandwrittenALBERT()
    ids = torch.randint(0, 3000, (2, 16))
    logits = model(ids)
    print(f"手工ALBERT输出: {logits.shape}")
    
    # 参数量对比
    bert_params = 30000 * 768 + 12 * (768*768*3 + 768*3072*2)
    albert_params = 30000*128 + 128*768 + 768*768*3 + 768*3072*2
    print(f"BERT-base估算参数量: {bert_params/1e6:.1f}M")
    print(f"ALBERT-base估算参数量: {albert_params/1e6:.1f}M")

if __name__ == "__main__":
    test_handwritten()
```

---

## 9. 可视化与结果理解

### 9.1 参数共享的效果
共享参数使得不同层的输出分布更加相似（各层"风格"一致），但语义抽象层次不同（浅层学语法，深层学语义）。

### 9.2 因子分解的效果
$d=128$ 的嵌入空间可视化显示t-SNE投影中语义相近的词仍能聚在一起，说明低维空间足够编码词义信息。

---

## 10. 模型评估

| 模型 | 参数量 | SQuAD 1.1 F1 | MNLI | SST-2 |
|------|--------|-------------|------|-------|
| BERT-base | 110M | 88.5 | 84.5 | 93.5 |
| ALBERT-base | 12M | 88.2 | 83.8 | 92.8 |
| ALBERT-xxlarge | 223M | 91.4 | 90.8 | 96.1 |

---

## 11. 常见问题

### Q1: ALBERT的推理速度比BERT快吗？
A: 不会。虽然参数少，但计算量相同（层数、隐藏维度不变）。推理速度取决于FLOPs而非参数量。

### Q2: ALBERT的嵌入维度d为什么选128？
A: 128在大多数任务上足够好，进一步增大收益递减。这是GLUE上的消融实验结果。

### Q3: ALBERT为什么不用LayerNorm共享？
A: LayerNorm的 $\gamma, \beta$ 参数很少（2×H），共享后严重影响训练稳定性。

---

## 12. 学习总结

ALBERT通过**嵌入因子分解**和**跨层参数共享**实现了~90%的参数量缩减，证明了参数效率优化在不显著牺牲性能前提下的可行性。

---

## 13. 练习题

### 习题1：计算ALBERT的嵌入因子分解节省了多少参数。
**答案**：BERT-base: 30000×768=23.04M。ALBERT: 30000×128+128×768=3.84M+0.098M=3.94M。节省 = (23.04-3.94)/23.04 ≈ 83%。

### 习题2：SOP比NSP好在哪？
**答案**：NSP判断两个句子是否属于同一主题（容易），SOP判断顺序是否正确（需要理解连贯性），SOP提供了更有意义的训练信号。

### 习题3：为什么ALBERT堆更多层效果不下降？
**答案**：参数共享天然具有正则化效果，防止过拟合。同时更多层提供更多计算（非线性变换次数），提升了模型容量。

### 习题4：ALBERT的softmax共享和BERT有什么不同？
**答案**：ALBERT的MLM头先投影到低维（embedding_dim）再投影回词汇表，比BERT直接H→V更参参数高效。

---

## 14. 学习路径建议

### 前置
- BERT、Transformer

### 平行
- RoBERTa、ELECTRA、DistilBERT、TinyBERT

### 进阶
- MobileBERT、MiniLM
