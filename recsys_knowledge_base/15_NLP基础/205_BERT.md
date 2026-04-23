# BERT 学习文档

> 双向编码的里程碑——理解而非生成

---

## 1. 算法基础认知

### 1.1 什么是BERT

**BERT（Bidirectional Encoder Representations from Transformers）** 是Google 2018年提出的预训练语言模型，使用Transformer Encoder，通过**双向**上下文理解文本。

```
GPT:  从左到右生成（单向）
BERT: 同时看左右两边（双向）→ 更好的理解能力

BERT = Bidirectional Encoder Representations from Transformers
```

### 1.2 预训练任务

| 任务 | 说明 |
|------|------|
| **MLM** (Masked Language Model) | 随机遮盖15%的词，预测被遮盖的词 |
| **NSP** (Next Sentence Prediction) | 判断两个句子是否相邻 |

### 1.3 在推荐系统中的应用

| 应用 | 说明 |
|------|------|
| **BERT4Rec** | 用BERT做序列推荐，预测被mask的行为 |
| **文本特征** | 用BERT编码商品标题、描述等文本 |
| **语义理解** | 理解用户query和物品描述的语义 |

---

## 3. 数学公式

### 3.1 MLM损失

$$\mathcal{L}_{MLM} = -\mathbb{E}_{mask}\log P(x_i | x_{\backslash i})$$

### 3.2 NSP损失

$$\mathcal{L}_{NSP} = -\log P(\text{IsNext} | A, B)$$

### 3.3 总损失

$$\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$$

---

## 7. 调库实现

```python
"""
BERT 在推荐系统中的应用: BERT4Rec
"""
import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    """
    BERT4Rec: 用BERT做序列推荐
    
    核心思想:
    - 用户行为序列 [A, B, C, D, E]
    - 随机mask: [A, [MASK], C, [MASK], E]
    - 预测被mask的位置的物品
    """
    
    def __init__(self, num_items, d_model=64, n_heads=2, n_layers=2, max_len=50):
        super().__init__()
        
        self.item_embedding = nn.Embedding(num_items + 2, d_model)  # +2: padding + mask
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        self.output = nn.Linear(d_model, num_items)
    
    def forward(self, item_seq):
        B, T = item_seq.shape
        pos = torch.arange(T, device=item_seq.device).unsqueeze(0)
        
        x = self.item_embedding(item_seq) + self.pos_embedding(pos)
        x = self.transformer(x)
        logits = self.output(x)
        
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = BERT4Rec(num_items=1000, d_model=64, n_heads=2, n_layers=2)
    item_seq = torch.randint(0, 1000, (4, 20))
    
    logits = model(item_seq)
    print(f"行为序列: {item_seq.shape}")
    print(f"预测输出: {logits.shape}")
```

---

## 12. 学习总结

1. **BERT = Transformer Encoder + MLM + NSP**
2. **双向理解**：同时利用左右上下文，比GPT单向更适合理解任务
3. **MLM预训练**：随机mask词并预测，强迫模型理解上下文
4. **BERT4Rec**：将MLM思想用于推荐，mask用户行为预测物品

---

## 14. 学习路径

```
GPT → [当前: BERT] → BERT4Rec → 大模型推荐
```
