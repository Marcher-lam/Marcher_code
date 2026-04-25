# DistilBERT 学习文档

> BERT 的轻量级蒸馏版本，速度提升 60%，参数减少 40%。

---

## 1. 算法基础认知

### 1.1 发展背景

DistilBERT 由 HuggingFace 团队于 2019 年在论文《DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter》中提出，通过知识蒸馏技术将 110M 参数的 BERT-base 压缩到 66M，保留 97% 的性能。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 知识蒸馏 |
| 参数 | 66M（减少 40%） |
| 速度 | 提升 60% |
| 性能 | 保留 97% |

### 1.3 对比

| 模型 | 参数 | GLUE | 速度 |
|------|------|------|------|
| BERT-base | 110M | 100% | 1x |
| DistilBERT | 66M | 97% | 1.6x |

---

## 2. 核心原理

### 2.1 知识蒸馏

学生模型学习教师模型的输出分布：

$$L_{KD} = \sum_i p_T(y_i) \log p_S(y_i)$$

### 2.2 三重损失

1. **蒸馏损失**：学生模仿教师
2. **MLM 损失**：原始任务
3. **余弦嵌入**：隐藏层对齐

### 2.3 架构简化

- 移除 NSP 任务
- 层数从 12 → 6
- 隐藏维度保持 768

---

## 3. 数学公式与推导

### 3.1 蒸馏损失

$$L_{distill} = \sum_{i} \sum_{c} p_T^c(z_i) \log p_S^c(z_i)$$

其中 $z_i$ 是 logits，$c$ 是类别。

### 3.2 软标签

$$p_T^c(z) = \frac{\exp(z_c/T)}{\sum_k \exp(z_k/T)}$$

$T$ 是温度，通常取 2。

### 3.3 总损失

$$L = \alpha L_{MLM} + (1-\alpha) L_{KD} + \beta L_{cos}$$

---

## 4. 训练过程讲解

### 4.1 训练配置

| 参数 | 值 |
|------|-----|
| 温度 T | 2 |
| α | 0.3 |
| β | 1 |
| Batch | 512 |

### 4.2 训练数据

使用与 BERT 相同的语料：BookCorpus + English Wikipedia

---

## 5. 应用场景

### 5.1 典型应用

- **移动端部署**：手机、边缘设备
- **实时推理**：延迟敏感场景
- **低成本推理**：资源受限环境

### 5.2 代码示例

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

---

## 6. 调库实现

### 6.1 HuggingFace 实现

```python
import torch

class DistilBERTModel:
    """DistilBERT 轻量模型"""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        
    def encode(self, text):
        return self.tokenizer(text, return_tensors='pt')
    
    def classify(self, text):
        inputs = self.encode(text)
        outputs = self.model(**inputs)
        return outputs.logits.argmax(-1)
    
    def extract_features(self, text):
        inputs = self.encode(text)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state


def demo():
    print("=== DistilBERT 演示 ===\n")
    
    model = DistilBERTModel()
    
    # 分类
    result = model.classify("This is a great movie!")
    print(f"分类结果: {result}")
    
    return model


if __name__ == "__main__":
    demo()
```

---

## 7. 手工代码实现

### 7.1 简化实现

```python
import torch
import torch.nn as nn

class DistilBERTStudent(nn.Module):
    """DistilBERT 学生模型（简化）"""
    
    def __init__(self, vocab_size=30522, hidden_dim=768, num_layers=6):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer 层（6层）
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=12,
                dim_feedforward=hidden_dim*4
            )
            for _ in range(num_layers)
        ])
        
        # 输出
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        # [CLS] token
        cls_output = x[:, 0]
        
        return self.classifier(cls_output)


def demo():
    print("=== DistilBERT 手工实现演示 ===\n")
    
    model = DistilBERTStudent(num_layers=6)
    
    # 输入
    input_ids = torch.randint(0, 30522, (1, 20))
    
    # 输出
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

1. **速度快**：1.6x 推理加速
2. **体积小**：减少 40% 参数
3. **效果好**：保留 97% 性能

### 8.2 缺点

1. **性能损失**：仍有 3% 下降
2. **需要教师**：训练依赖 BERT

### 8.3 改进方向

- **TinyBERT**：更激进蒸馏
- **MiniLM**：层蒸馏

---

## 9. 可视化与结果理解

### 9.1 性能对比

```python
def visualize():
    print("""
    DistilBERT 性能对比:
    
    模型         参数    GLUE分数   推理速度
    ─────────────────────────────────────
    BERT-base  110M     100%       1.0x
    DistilBERT 66M      97%        1.6x
    TinyBERT   14M      85%        3.0x
    """)
```

---

## 10. 模型评估

### 10.1 GLUE 基准

| 任务 | BERT | DistilBERT |
|------|------|-------------|
| MNLI | 84.5% | 82.0% |
| SST-2 | 93.5% | 91.5% |
| CoLA | 60.5% | 57.0% |

---

## 11. 学习总结

**核心要点**：

1. **知识蒸馏**：学生学习教师
2. **三重损失**：MLM+KD+Cos
3. **层数减半**：6层 vs 12层

**DistilBERT 核心优势**：
- 速度快 60%
- 参数少 40%
- 效果好

---

## 12. 练习题与思考题

### 12.1 选择题

1. DistilBERT的层数是BERT的：
   - A) 相同
   - B) 一半
   - C) 两倍

2. 蒸馏损失不包含：
   - A) MLM
   - B) BERT的Embedding
   - C) 对抗损失

3. 训练时的教师模型需要：
   - A) 随机初始化
   - B) 预训练权重
   - C) 无所谓

### 12.2 简答题

1. 解释三重损失函数的作用？
2. 为什么层数减半能保持效果？

### 12.3 编程题

1. 使用transformers加载DistilBERT
2. 比较BERT和DistilBERT推理速度

---

## 13. 常见问题与易错点

### Q1: DistilBERT支持中文吗？

**答案**：支持，支持多语言版本distilbert-base-multilingual-cased。

### Q2: 如何进一步压缩？

**答案**：结合量化（INT8）和剪枝。

### Q3: 可以微调吗？

**答案**：可以，和BERT类似的方式微调。

### Q4: 精度损失多少？

**答案**：约1-3%，但在可接受范围内。

### Q5: 适用于哪些任务？

**答案**：文本分类、问答、序列标注等。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
BERT基础
    ↓
知识蒸馏
    ↓
DistilBERT原理
    ↓
实践蒸馏
    ↓
TinyBERT/MobileBERT
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| BERT | 教师模型 |
| TinyBERT | 蒸馏改进 |
| MobileBERT | 极致压缩 |
| ALBERT | 共享参数 |

### 14.3 扩展阅读

1. Sanh et al. (2019). DistilBERT
2. Wang et al. (2020). TinyBERT

---

## 15. 学习总结

### 15.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心创新 | 蒸馏学习 |
| 三重损失 | MLM+KD+Cosine |
| 压缩效果 | 60%速度, 40%参数 |

### 15.2 公式汇总

学生损失：
$$\mathcal{L}_S = \lambda \mathcal{L}_{MLM} + (1-\lambda) \mathcal{L}_{KD}$$

蒸馏损失：
$$\mathcal{L}_{KD} = \sum_i \text{KL}(p_i^T || p_i^S) = \sum_i \sum_j p_j^T \log \frac{p_j^T}{p_j^S}$$

---

## 附录

### A. 超参数速查

| 参数 | 推荐值 |
|------|--------|
| n_layers | 6 |
| dim | 768 |
| attention_heads | 12 |
| temperature | 2.0 |

### B. 参考

1. Sanh et al. (2019). DistilBERT. arXiv:1910.01008

---

**文档结束**