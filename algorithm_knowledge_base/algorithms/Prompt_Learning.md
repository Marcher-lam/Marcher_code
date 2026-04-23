# Prompt Learning 学习文档

> 通过设计提示模板引导预训练大模型，实现高效任务适配。

---

## 1. 算法基础认知

**Prompt Learning（提示学习）** 是一种利用预训练大模型能力的范式，通过设计合适的提示（prompts）来引导模型完成特定任务，无需 fine-tuning 模型参数。

### 1.1 核心思想

传统fine-tuning：
```
预训练模型 → 梯度更新 → 任务模型
```

Prompt Learning：
```
预训练模型 + 提示模板 → 任务输出
```

### 1.2 为什么需要Prompt Learning？

- 大模型参数巨大，fine-tuning成本高
- 少量样本即可触发模型能力
- 保留模型泛化能力

### 1.3 与Fine-tuning对比

| 方法 | 参数更新 | 样本需求 | 泛化能力 |
|------|--------|----------|----------|
| Fine-tuning | 全部/部分 | 大量 | 任务内 |
| Prompt Learning | 0 | 少量 | 通用 |

---

## 2. 核心原理

### 2.1 提示类型

**硬提示（Hard Prompt）**：
- 人工设计的离散文本
- 如："The sentiment of this movie is [MASK]"

**软提示（Soft Prompt）**：
- 可学习的连续向量
- 如：[v1, v2, v3, ..., vk]

### 2.2 模板设计

```python
# 分类任务
template = "This is a [SUBJ] . It was really [MASK] ."

# 抽取任务  
template = "[SUBJ] is located in [LOC] . [MASK] ."
```

### 2.3 答案空间

将标签映射到词：
- 正面 → "great"
- 负面 → "terrible"

---

## 3. 数学公式

### 3.1 提示推理

给定输入x和提示模板T：
```python
output = model(T.format(x))
```

### 3.2 软提示学习

$$\mathcal{L} = -\sum_i \log P(y_i | x_i, \theta)$$

其中$\theta$只包含软提示向量

### 3.3 P-Tuning

使用LSTM编码可学习提示：
$$h_i = LSTM(h_{i-1}, x)$$

---

## 4. 训练过程

### 4.1 人工提示设计

```python
# 情感分析
prompt = "The review is [ sentiment ] . The review was really [MASK] ."
# 填充[MASK]位置获得答案
```

### 4.2 自动提示优化

```python
# CoOp: 可学习提示
prompt_tokens = nn.Parameter(embeddings)

# 训练
loss = task_loss(model, prompt_tokens)
```

### 4.3 少样本学习

Few-shot设置：
```python
examples = [
    ("Great movie!", "positive"),
    ("Boring film", "negative"),
]
```

---

## 5. 应用场景

### 5.1 Text Classification

情感分析、主题分类、意图识别。

### 5.2 Question Answering

阅读理解、问答任务。

### 5.3 Text Generation

文本续写、代码生成。

### 5.4 Information Extraction

命名实体识别、关系抽取。

---

## 6. 调库实现（Hugging Face）

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class PromptLearner:
    """提示学习器"""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def predict(self, text, template, label_words):
        """预测"""
        # 填充模板
        input_text = template.replace("[SUBJ]", text)
        
        # 编码
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits
        
        # 获取[MASK]位置预测
        mask_idx = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero()
        # 取label_words中最高概率的词
        
        return label_words[top_idx]


class CoOp:
    """CoOp: 可学习提示"""
    
    def __init__(self, num_tokens=10, embed_dim=768):
        # 软提示
        self.prompt_embeddings = torch.randn(num_tokens, embed_dim)
    
    def forward(self, input_ids, token_type_ids):
        # 拼接提示和输入
        combined_input = torch.cat([self.prompt_embeddings, input_embeddings], dim=1)
        return self.model(combined_input)


def demo():
    print("=== Prompt Learning 演示 ===\n")
    model = PromptLearner()
    text = "This movie is amazing!"
    template = "The sentiment is [MASK] ."
    label_words = {"positive": "great", "negative": "terrible"}
    # result = model.predict(text, template, label_words)


if __name__ == "__main__":
    demo()
```

---

## 7. 手工代码实现

```python
import numpy as np

class SimplePromptLearning:
    """简化版提示学习"""
    
    def __init__(self, model):
        self.model = model
        self.prompt_template = "Text: {text}. Sentiment: [MASK]."
        self.label_words = ["positive", "negative"]
    
    def infer(self, text):
        """推理"""
        prompt = self.prompt_template.format(text=text)
        # 使用模型推理
        return self.model.complete(prompt, self.label_words)


if __name__ == "__main__":
    print("=== 提示学习实现 ===\n")
    print("1. 设计提示模板")
    print("2. 映射答案空间")
    print("3. 执行推理")
```

---

## 8. 可视化

```python
import matplotlib.pyplot as plt

def visualize():
    print("\n=== 提示学习流程 ===\n")
    print("""
输入 → 模板拼接 → 大模型推理 → 答案映射
 x     prompt       model         y
    
GPT-3示例:
Input:  "Review: I love this movie. Sentiment:"
Model:  "positive"
    """)


if __name__ == "__main__":
    visualize()
```

---

## 9. 主流方法

| 方法 | 类型 | 特点 |
|------|------|------|
| Manual Prompt | 人工 | 简单有效 |
| Auto Prompt | 自动搜索 | 更优性能 |
| P-Tuning | LSTM编码 | 连续提示 |
| CoOp | 连续提示 | 端到端可微 |
| Prefix Tuning | 前缀 | 冻结LM |

---

## 10. 评估

### 10.1 效果评估

- 准确率、F1
- 少样本性能

### 10.2 效率评估

- 推理延迟
- 内存占用

---

## 11. 常见问题

### 11.1 提示敏感性

不同提示效果差异大

### 11.2 答案空间选择

需要人工设计/搜索

### 11.3 多任务冲突

多任务学习需要统一提示设计

---

## 12. 学习总结

**Prompt Learning要点**：

1. **提示设计**：核心技巧
2. **硬提示**：人工设计文本
3. **软提示**：可学习向量
4. **答案映射**：标签→词
5. **优点**：零参数、高效

---

## 13. 练习题与思考题

1. 为什么大模型适合提示学习？
2. 硬提示和软提示的区别？

---

## 14. 学习路径建议

1. 理解GPT-3的few-shot能力
2. 学习各种提示技巧
3. 实践少样本分类

*Prompt Learning让大模型使用更加高效，是NLP的重大范式转变。*