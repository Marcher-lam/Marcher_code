# BERT（Bidirectional Encoder Representation from Transformers）学习文档

> 双向编码器表示，使用Transformer编码器和掩码语言建模，刷新NLP多项任务记录。

## 1. 算法基础认知

### 一句话定义

BERT是一种基于Transformer编码器的双向预训练语言模型，通过掩码语言建模和下一句预测两个任务进行预训练，然后在下游任务上进行微调。

### 直觉类比

就像一个人阅读文章时，能够同时"瞻前顾后"地理解每个词的含义。BERT正是让模型具备这种双向理解能力的架构。

### 历史背景

- **2018年10月**：Google发布BERT论文
- **2019年**：BERT成为NLP领域基准
- **2020年**：RoBERTa、ALBERT等改进版本

### 算法定位

BERT是**预训练+微调范式**的里程碑模型，属于监督学习（预训练）/迁移学习。

### 前置知识

- Transformer编码器
- 注意力机制
- 语言模型基础

---

## 2. 核心原理

### 核心思想

BERT的核心创新是**双向**和**预训练-微调**。通过双向自注意力让每个词看到左右上下文，通过掩码机制实现无监督预训练。

### 工作流程

**预训练阶段**：
1. 15%的词被选中，其中80%被替换为[MASK]，10%随机替换，10%保持不变
2. 训练模型预测被掩码的词（MLM任务）
3. 训练模型判断两句话是否相邻（NSP任务）

**微调阶段**：
1. 在下游任务数据上微调所有参数
2. 添加任务相关层进行分类

### 架构图

```
输入: "The cat sat on the [MASK]"
       ↓
Token Embedding + Segment Embedding + Position Embedding
       ↓
Transformer Encoder × 12 (BERT-Base) / × 24 (BERT-Large)
       ↓
[CLS]向量 → 分类层 → 输出
```

---

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $x_i$ | 输入token |
| $h_i$ | 编码后的表示 |
| $M$ | 掩码位置集合 |
| $\theta$ | 模型参数 |

### 掩码语言建模（MLM）

$$L_{MLM} = -\sum_{i \in M} \log P(x_i | x_{\setminus M})$$

目标是最小化被掩码位置的对数损失。

### 下一句预测（NSP）

$$L_{NSP} = -\log P(label | h_{[CLS]})$$

判断句子B是否是句子A的下一句（二分类）。

### 预测概率

$$P(x_i | x_{\setminus M}) = \text{softmax}(h_i W)$$

---

## 4. 训练过程讲解

### 超参数表

| 参数 | BERT-Base | BERT-Large |
|------|-----------|-------------|
| 层数L | 12 | 24 |
| 隐藏维度H | 768 | 1024 |
| 注意力头数A | 12 | 16 |
| 参数量 | 110M | 340M |
| 序列长度 | 512 | 512 |
| 预训练轮数 | 1M steps | 1M steps |
| 批量大小 | 256 | 256 |
| 学习率 | 1e-4 | 1e-4 |

### 训练技巧

1. **学习率调度**：使用warmup（10%步数）+线性衰减
2. **Adam优化器**：$\beta_1=0.9, \beta_2=0.999$
3. **梯度裁剪**：max_norm=1.0
4. **数据增强**：动态掩码、随机替换

---

## 5. 应用场景

1. **文本分类**：情感分析、垃圾邮件检测
2. **问答系统**：SQuAD阅读理解
3. **命名实体识别**：NER任务
4. **句子匹配**：自然语言推断
5. **文本生成**：BERT-GPT组合

---

## 6. 优缺点分析

### 优点

1. **双向建模**：利用完整上下文
2. **预训练-微调**：减少标注数据需求
3. **通用性强**：一个模型可微调多个任务
4. **迁移学习**：大幅提升小数据集效果

### 缺点

1. **无法建模**：不适合生成任务
2. **计算量大**：训练和推理成本高
3. **序列长度**：最大512限制

---

## 7. 调库实现

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    """基于BERT的文本分类器"""
    def __init__(self, num_classes=2, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]位置的输出进行分类
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        return self.classifier(pooled_output)

# 微调示例
def fine_tune_bert():
    from transformers import BertForSequenceClassification
    from torch.utils.data import DataLoader
    from transformers import AdamW
    
    # 加载预训练模型
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=2
    )
    
    # 准备数据
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 编码文本
    inputs = tokenizer("这是一个正面情感的例子", return_tensors='pt')
    
    # 前向传播
    outputs = model(**inputs)
    loss = outputs.loss
    logits = outputs.logits
    
    # 反向传播
    loss.backward()
    
    return model

# 测试
if __name__ == "__main__":
    model = BertClassifier(num_classes=2)
    input_ids = torch.randint(0, 30000, (32, 128))
    attention_mask = torch.ones(32, 128)
    logits = model(input_ids, attention_mask)
    print(f"输出形状: {logits.shape}")  # (32, 2)
```

---

## 8. 手工代码实现

```python
import numpy as np

class SimplifiedBERT:
    """简化的BERT实现（仅演示思想）"""
    
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12):
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 简化的嵌入层
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.segment_embedding = np.zeros((2, d_model))
        self.position_embedding = np.random.randn(512, d_model) * 0.02
        
        # 简化 Transformer 层（省略）
        
    def forward(self, input_ids, segment_ids=None):
        # 词嵌入
        token_emb = np.array([[self.token_embedding[t] for t in seq] 
                             for seq in input_ids])
        
        # 位置编码
        seq_len = input_ids.shape[1]
        pos_emb = self.position_embedding[:seq_len]
        
        # 组合
        hidden = token_emb + pos_emb
        
        if segment_ids is not None:
            seg_emb = np.array([[self.segment_embedding[s] for s in seq] 
                               for seq in segment_ids])
            hidden += seg_emb
        
        # 简化的 Transformer 处理（实际需要完整实现）
        
        return hidden

# 测试
if __name__ == "__main__":
    np.random.seed(42)
    bert = SimplifiedBERT(30000)
    input_ids = np.random.randint(0, 30000, (2, 10))
    output = bert(input_ids)
    print(f"输出形状: {output.shape}")  # (2, 10, 768)
```

---

## 9. 可视化与评估

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_bert_attention():
    """可视化BERT注意力模式"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 模拟不同层的注意力模式
    patterns = [
        # 底层：更多关注邻近词
        np.random.rand(12, 12) * 0.3 + np.eye(12) * 0.7,
        # 中层：形成局部块
        np.random.rand(12, 12) * 0.2 + np.ones((12, 12)) * 0.1,
        # 高层：全局关注
        np.random.rand(12, 12) * 0.1 + 0.05,
    ]
    
    titles = ['底层（局部）', '中层（局部块）', '高层（全局）']
    
    for i, (ax, pattern) in enumerate(zip(axes.flatten()[:3], patterns)):
        im = ax.imshow(pattern, cmap='Blues', aspect='auto')
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.suptitle('BERT不同层级的注意力模式')
    plt.tight_layout()
    plt.savefig('bert_attention_layers.png', dpi=150)
    plt.show()

def evaluate_bert_performance():
    """评估BERT在常见任务上的性能"""
    results = {
        'SQuAD 1.1 (F1)': 93.2,
        'SQuAD 2.0 (F1)': 88.5,
        'MNLI (m/mm)': 86.7/84.9,
        'SST-2': 93.5,
        'CoLA': 60.5,
        'QQP': 71.2,
    }
    
    tasks = list(results.keys())
    scores = [v if isinstance(v, float) else v[0] for v in results.values()]
    
    plt.figure(figsize=(10, 6))
    plt.barh(tasks, scores, color='steelblue')
    plt.xlabel('F1/Accuracy Score')
    plt.title('BERT Performance on GLUE Benchmark')
    plt.tight_layout()
    plt.savefig('bert_glue_scores.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    visualize_bert_attention()
    evaluate_bert_performance()
```

---

## 10. 常见问题与易错点

1. **显存不足**：使用梯度累积或减少批量大小
2. **过拟合**：增加dropout，减少训练轮数
3. **数据格式错误**：注意[CLS]和[SEP]的使用
4. **序列过长**：使用滑动窗口或longformer

---

## 11. 学习总结

BERT的核心贡献是证明了**双向Transformer**+**预训练-微调**范式的强大能力。这一范式深刻影响了NLP领域，催生了RoBERTa、ALBERT、ELECTRA等众多后续模型。

---

## 12. 练习题

1. **基础**：BERT为什么使用掩码而不是直接训练？
2. **进阶**：为什么BERT使用[CLS]做分类而不是所有token的平均？
3. **开放**：如何用更小的模型达到接近BERT的效果？

---

## 13. 学习路径

- 前置：Transformer、位置编码
- 平行：GPT、RoBERTa
- 进阶：ALBERT、ELECTRA、DeBERTa