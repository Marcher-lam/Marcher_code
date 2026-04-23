# TinyBERT 学习文档

> 更激进的 BERT 蒸馏，4 层 Transformer。

---

## 1. 算法基础认知

### 1.1 发展背景

TinyBERT 由华为诺亚实验室于 2019 年在论文《TinyBERT: Distilling BERT for Natural Language Understanding》中提出，采用两阶段蒸馏：通用领域预训练蒸馏 + 下游任务微调蒸馏。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 知识蒸馏 |
| 参数 | 14M（减少 87%） |
| 层数 | 4 层 |
| 性能 | 保留 96% |

### 1.3 对比

| 模型 | 参数 | GLUE |
|------|------|------|
| BERT-base | 110M | 100% |
| DistilBERT | 66M | 97% |
| TinyBERT | 14M | 96% |

---

## 2. 核心原理

### 2.1 两阶段蒸馏

1. **通用蒸馏**：在大规模语料上蒸馏
2. **任务蒸馏**：在下游任务数据上蒸馏

### 2.2 知识迁移

- **输出层**： logits 蒸馏
- **隐藏层**： hidden states 蒸馏
- **注意力**： attention 蒸馏

### 2.3 层级映射

| 教师层 | 学生层 |
|--------|--------|
| 12 层 | 4 层 |
| 768 维 | 312 维 |

---

## 3. 数学公式与推导

### 3.1 注意力蒸馏

$$L_{attn} = \frac{1}{h} \sum_{i=1}^{h} \text{MSE}(A_i^S, A_i^T)$$

### 3.2 隐藏层蒸馏

$$L_{hidn} = \sum_{j=1}^{L_S} \text{MSE}(W_j^S H_j^S, W^T H_{\phi(j)}^T)$$

### 3.3 总损失

$$L = \alpha L_{attn} + \beta L_{hidn} + \gamma L_{pred}$$

---

## 4. 训练过程讲解

### 4.1 蒸馏配置

| 阶段 | 批量大小 | 学习率 |
|------|---------|--------|
| 通用 | 512 | 1e-3 |
| 任务 | 64 | 5e-5 |

### 4.2 训练数据

- 通用：BookCorpus + Wikipedia
- 任务：下游任务数据

---

## 5. 应用场景

### 5.1 典型应用

- **移动端**：手机部署
- **嵌入式**：IoT 设备
- **实时系统**：低延迟需求

### 5.2 代码示例

```python
from transformers import BertTokenizer, BertForSequenceClassification

# 使用 TinyBERT
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
```

---

## 6. 调库实现

### 6.1 HuggingFace 实现

```python
import torch

class TinyBERTModel:
    """TinyBERT 轻量模型"""
    
    def __init__(self, model_name='huawei-noah/TinyBERT_General_4L_312D'):
        from transformers import BertTokenizer, BertForSequenceClassification
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        
    def encode(self, text):
        return self.tokenizer(text, return_tensors='pt')
    
    def classify(self, text):
        inputs = self.encode(text)
        outputs = self.model(**inputs)
        return outputs.logits.argmax(-1)


def demo():
    print("=== TinyBERT 演示 ===\n")
    
    model = TinyBERTModel()
    print(f"参数量: 14M")
    print(f"速度: 3x BERT")
    
    return model


if __name__ == "__main__":
    demo()
```

---

## 7. 手工代码实现

### 7.1 简化学生模型

```python
import torch
import torch.nn as nn

class TinyBERTStudent(nn.Module):
    """TinyBERT 学生模型（4层）"""
    
    def __init__(self, vocab_size=30522, hidden_dim=312, num_layers=4):
        super().__init__()
        
        # 嵌入
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 4 层 Transformer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=12,
                dim_feedforward=hidden_dim * 4
            )
            for _ in range(num_layers)
        ])
        
        # 输出
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        cls_output = x[:, 0]
        
        return self.classifier(cls_output)


def demo():
    print("=== TinyBERT 手工实现演示 ===\n")
    
    model = TinyBERTStudent(num_layers=4, hidden_dim=312)
    
    input_ids = torch.randint(0, 30522, (1, 20))
    output = model(input_ids)
    
    print(f"输入: {input_ids.shape}")
    print(f"输出: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    demo()
```

---

## 8. 优缺点分析

### 8.1 优点

1. **极小参数量**：14M 仅 BERT 13%
2. **速度快**：3 倍加速
3. **效果好**：保留 96% 性能

### 8.2 缺点

1. **性能损失**：仍有 4% 下降
2. **训练复杂**：两阶段蒸馏

---

## 9. 可视化与结果理解

### 9.1 性能对比

```python
def visualize():
    print("""
    轻量 BERT 对比:
    
    模型         参数    GLUE   速度
    ───────────────────────────────
    BERT       110M    100%   1.0x
    DistilBERT 66M     97%    1.6x
    TinyBERT   14M     96%    3.0x
    MiniLM    22M     95%    2.5x
    """)
```

---

## 10. 模型评估

### 10.1 GLUE 基准

| 任务 | BERT | TinyBERT |
|------|------|-----------|
| MNLI | 84.5% | 81.0% |
| SST-2 | 93.5% | 90.5% |

---

## 11. 学习总结

**核心要点**：

1. **两阶段蒸馏**：通用+任务
2. **多层蒸馏**：注意力+隐藏+输出
3. **极小模型**：4 层 312 维

**TinyBERT 核心优势**：
- 极小参数
- 速度快 3 倍

---

## 12. 练习题与思考题

### 12.1 基础练习

1. 蒸馏 vs 剪枝
2. 注意力蒸馏原理

### 12.2 思考题

1. 蒸馏极限

---

## 14. 学习路径建议

### 入门阶段

1. BERT 基础
2. 知识蒸馏

### 进阶阶段

1. TinyBERT 原理
2. 实践蒸馏

### 高级阶段

1. 其他轻量模型
2. 部署优化

**推荐路线**：

```
BERT → DistilBERT → TinyBERT → MobileBERT
```

**TinyBERT 是极轻量 BERT 的代表，熟练掌握它对端侧部署很重要。**