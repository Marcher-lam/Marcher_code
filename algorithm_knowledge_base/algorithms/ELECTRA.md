# ELECTRA 学习文档

> 替换 token 检测的预训练语言模型，效率超越 BERT。

---

## 1. 算法基础认知

### 1.1 发展背景

ELECTRA（Efficiently Learning an Encoder that Classifies Token Replacements Accurately）由 Clark 等人于 2020 年在论文《ELECTRA: Pre-training Text Encoders as Discriminators》中提出。与 BERT 的 MLM（Masked Language Modeling）不同，ELECTRA 使用替换 token 检测（Replaced Token Detection，RTD）作为预训练任务，训练效率显著提升。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 判别式预训练 |
| 任务 | 替换 token 检测（RTD） |
| 效率 | 比 BERT 高 3-4 倍 |
| 性能 | 小模型尤为显著 |

### 1.3 模型系列

| 模型 | 参数量 | GLUE 分数 |
|------|--------|----------|
| ELECTRA-Small | 14M | 75.8 |
| ELECTRA-Base | 110M | 92.0 |
| ELECTRA-Large | 340M | 94.7 |

---

## 2. 核心原理

### 2.1 预训练任务对比

| BERT（MLM） | ELECTRA（RTD） |
|-------------|---------------|
| 15% 遮蔽 | 15% 替换 |
| 预测遮蔽 token | 判别是否为替换 |
| 80% → [MASK] | 替换→-generator |
| 10% → random | 10% → random |
| 10% → original | 10% → original |

### 2.2 生成器-判别器

ELECTRA 使用两个模型：

1. **生成器（Generator）**：小规模 MLM 模型
   - 预测被替换位置的token
2. **判别器（Discriminator）**：RTD 任务
   - 预测每个 token 是否被替换

### 2.3 RTD 损失

$$\text{Loss}_{RTD} = -\sum_{i=1}^{n} [y_i = 1] \log D(x_i) + [y_i = 0] \log(1-D(x_i))$$

其中 $y_i=1$ 表示 token $i$ 被替换。

---

## 3. 数学公式与推导

### 3.1 替换 Token 生成

给定输入序列 $x$，随机选择 15% 位置：

- 80%：替换为 [MASK]
- 10%：替换为随机 token
- 10%：保持不变（训练判别器时需要）

生成器预测替换位置的 token。

### 3.2 判别器预测

$D(x_i)$ 预测 $x_i$ 是否被替换：

$$D(x) = \sigma(w^T \cdot h(x) + b)$$

### 3.3 联合损失

$$\text{Loss} = \text{Loss}_{Gen} + \lambda \cdot \text{Loss}_{RTD}$$

通常 $\lambda \in [1, 50]$。

---

## 4. 训练过程讲解

### 4.1 训练流程

```
Input: 原始文本corpus

1. 生成替换：使用 Generator 生成替换文本
2. 训练判别器：RTD 任务
3. 保留判别器权重，仅训练 Generator
4. 迭代直到收敛
```

### 4.2 参数设置

| 参数 | Small | Base | Large |
|------|-------|------|-------|
| batch_size | 256 | 2048 | 512 |
| learning_rate | 2e-4 | 1e-4 | 2e-4 |
| weight_decay | 0.01 | 0.01 | 0.01 |
| epochs | 3-5 | 1-3 | 1-3 |

### 4.3 推理

推理时只需要判别器，生成器用于训练。

---

## 5. 应用场景

### 5.1 典型应用

- **文本分类**：情感分析
- **问答系统**：机器阅读理解
- **序列标注**：命名实体识别
- **文本相似度**：语义匹配

### 5.2 HuggingFace 使用

```python
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# 加载模型
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator')

# 推理
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

---

## 6. 调库实现

### 6.1 PyTorch 实现

```python
import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer

class ELECTRA:
    """ELECTRA 预训练模型
    
    参数:
        model_name: 模型名称
    """
    
    def __init__(self, model_name='google/electra-base-discriminator'):
        self.model_name = model_name
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraModel.from_pretrained(model_name)
        self.model.eval()
        
    def encode(self, text):
        """编码文本"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return inputs
    
    def extract_features(self, text):
        """提取特征"""
        inputs = self.encode(text)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
    def classify(self, text, num_labels=2):
        """分类（需要微调）"""
        model = ElectraForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        return model


def demo():
    """ELECTRA 演示"""
    print("=== ELECTRA 演示 ===\n")
    
    model = ELECTRA('google/electra-small-discriminator')
    
    # 特征提取
    features = model.extract_features("This is a great movie!")
    print(f"输入: This is a great movie!")
    print(f"特征维度: {features.shape}")
    
    return model


if __name__ == "__main__":
    demo()
```

### 6.2 RTD 任务实现

```python
class ElectraDiscriminator(nn.Module):
    """ELECTRA 判别器"""
    
    def __init__(self, vocab_size, embed_dim, num_layers):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads=8),
            num_layers=num_layers
        )
        self.classifier = nn.Linear(embed_dim, 1)
        
    def forward(self, input_ids):
        """前向传播
        
        input_ids: (batch, seq_len)
        """
        x = self.embed(input_ids)
        hidden = self.encoder(x)
        logits = self.classifier(hidden).squeeze(-1)
        
        return torch.sigmoid(logits)
```

---

## 7. 手工代码实现

### 7.1 简化 ELECTRA

```python
import torch
import torch.nn as nn

class SimpleElectra:
    """简化 ELECTRA"""
    
    def __init__(self, vocab_size=30522, embed_dim=128, hidden_dim=256):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # 生成器
        self.generator = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # 判别器
        self.discriminator = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_dim, 4),
                num_layers=3
            ),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def predict_replaced(self, input_ids):
        """预测是否被替换"""
        return self.discriminator(input_ids)


def demo():
    """ELECTRA 演示"""
    print("=== ELECTRA 手工实现演示 ===\n")
    
    model = SimpleElectra()
    
    # 输入
    input_ids = torch.randint(0, 30522, (1, 20))
    
    # 判别
    output = model.predict_replaced(input_ids)
    
    print(f"输入: {input_ids.shape}")
    print(f"输出: {output.shape}")


if __name__ == "__main__":
    demo()
```

---

## 8. 优缺点分析

### 8.1 优点

1. **训练效率高**：比 BERT 快 3-4 倍
2. **小模型效果好**：Small 即可达到 Base 效果
3. **RTD 任务**：每个 token 都参与训练

### 8.2 缺点

1. **两阶段训练**：需要训练生成器
2. **推理成��**：��� BERT 相当
3. **调参**：敏感的超参数

### 8.3 改进方向

- **ALECTRA**：All-token RTD
- **ConvELECTRA**：卷积改进

---

## 9. 可视化与结果理解

### 9.1 BERT vs ELECTRA

```python
def plot_comparison():
    """性能对比"""
    import matplotlib.pyplot as plt
    
    models = ['BERT-Base', 'ELECTRA-Base', 'ELECTRA-Small']
    accuracy = [92.0, 92.0, 90.5]
    flops = [1.0, 0.25, 0.05]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(models, accuracy, color='steelblue')
    ax1.set_ylabel('GLUE 分数')
    ax1.set_title('准确率对比')
    ax1.set_ylim(85, 95)
    
    ax2.bar(models, flops, color='coral')
    ax2.set_ylabel('相对计算量')
    ax2.set_title('训练效率')
    
    plt.tight_layout()
    plt.savefig('electra_comparison.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 GLUE 基准

| 模型 | MNLI | SST-2 | QNLI | SQuAD |
|------|------|-------|------|-------|
| ELECTRA-S | 87.9 | 95.3 | 91.2 | 75.8 |
| ELECTRA-B | 91.3 | 96.9 | 94.8 | 85.2 |
| BERT-B | 86.6 | 94.9 | 92.7 | 81.3 |

### 10.2 评估指标

- **GLUE**：综合理解基准
- **SQuAD**：问答
- **SST-2**：情感分析

---

## 11. 常见问题与易错点

### 11.1 生成器训练

**问题**：生成器训练不足

**解决**：使用足够数据训练生成器

### 11.2 替换率

**问题**：替换率设置

**解决**：15% 是最佳实践

---

## 12. 学习总结

**核心要点**：

1. **RTD 任务**：替换 token 检测
2. **生成-判别**：两阶段训练
3. **效率高**：3-4 倍于 BERT
4. **小模型好**：Small 效果显著

**ELECTRA 核心优势**：
- 训练效率高
- 小模型效果好
- RTD 任务有效

**学习建议**：

1. 理解 RTD 原理
2. 对比 BERT
3. 实践微调

---

## 13. 练习题与思考题

### 13.1 基础练习

1. RTD vs MLM 区别
2. ELECTRA 训练流程
3. 判别器实现

### 13.2 进阶练习

1. ELECTRA 微调
2. 性能对比实验

### 13.3 思考题

1. ELECTRA 的局限
2. 改进方向

---

### 13.4 详细答案与解析

#### 练习1：vs BERT

**问题**：ELECTRA 相对 BERT 的优势

**解答**：

| 方面 | BERT | ELECTRA |
|------|------|--------|
| 预训练任务 | MLM | RTD |
| 训练效率 | 基准 | 3-4x |
| token 使用 | 15% | 100% |
| 小模型效果 | 一般 | 显著 |

---

## 14. 学习路径建议

### 入门阶段

1. BERT 基础
2. 预训练任务理解
3. ELECTRA 原理

### 进阶阶段

1. ELECTRA 微调
2. 对比实验

### 高级阶段

1. 改进 RTD
2. 多任务学习

**推荐路线**：

```
BERT → RoBERTa → ELECTRA → ALBERT → DeBERTa
```

**ELECTRA 是预训练效率的重要突破，熟练掌握它对学习预训练模型很重要。**