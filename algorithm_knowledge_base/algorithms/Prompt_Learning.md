# Prompt Learning 学习文档

> 通过设计提示模板引导预训练大模型，实现高效任务适配

---

## 1. 算法基础认知

### 1.1 一句话定义

Prompt Learning（提示学习）是一种利用预训练大模型能力的范式，通过设计合适的提示（prompts）来引导模型完成特定任务，无需 fine-tuning 模型参数。这种方法在2020年GPT-3出现后变得极其重要，让单个模型能够执行无数任务。

### 1.2 直觉类比

想象你让一个知识渊博但没专门学习某个任务的人做事：
- **Fine-tuning**：手把手教他，每个细节都重新学
- **Prompt Learning**：给他一个提示/例子，他自己就能理解任务

就像你和一个经验丰富的厨师说"帮我做个生日蛋糕"，他不需要详细教就能理解意思！

### 1.3 发展历程

| 年份 | 里程碑 |
|------|---------|
| 2020 | GPT-3 few-shot能力 |
| 2021 | Prompt Learning概念 |
| 2021 | Auto-PT自动搜索 |
| 2021 | P-tuning v1/v2 |
| 2022 | InstructGPT (RLHF) |
| 2023 | LLM prompting |

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 少样本/零样本学习 |
| 核心 | 提示设计 |
| 参数 | 不更新模型 |
| 目标 | 任务适配 |

---

## 2. 核心原理

### 2.1 问题定义

给定预训练模型 M，输入提示 x，预测 y：

$$y = M(x)$$

目标是设计好的提示 x，使模型输出正确。

### 2.2 提示类型

**硬提示（Hard Prompt）**：
- 人工设计的离散文本
- 例子："The sentiment of this movie is [MASK]"
- 优点：可解释、直观
- 缺点：需要人工、可能非最优

**软提示（Soft Prompt）**：
- 可学习的连续向量 [v₁, v₂, ..., vₖ]
- 优点：自动优化、效果更好
- 缺点：不可解释、需要训练

### 2.3 提示设计

```python
# 模板设计示例
templates = {
    "情感分类": "Review: {text}. Sentiment: [MASK].",
    "问答": "Question: {question}? Answer: [MASK].",
    " summarization": "Article: {article}. Summary: [MASK].",
}

# 答案空间映射
label_words = {
    "positive": ["great", "excellent", "amazing"],
    "negative": ["terrible", "awful", "boring"],
}
```

---

## 3. 数学公式与推导

### 3.1 提示推理

给定输入 x 和提示模板 T：
```python
input_text = T.format(x)
output = model(input_text)
```

### 3.2 少样本学习

Few-shot 提示：
```python
prompt = """Example 1: Input: Great movie! Output: positive
Example 2: Input: Boring film Output: negative
Example 3: Input: {text} Output:"""
```

### 3.3 软提示学习

目标函数（只更新提示）：
$$\mathcal{L} = -\sum_i \log P(y_i | x_i, P; \theta)$$

其中 P 是提示向量，θ 是冻结的模型参数。

### 3.4 P-Tuning

使用LSTM/MLP编码可学习提示：

$$h_i = \text{MLP}(h_{i-1}, x)$$

---

## 4. PyTorch实现

### 4.1 基础实现

```python
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM


class PromptLearner:
    """提示学习器"""
    
    def __init__(self, model_name="bert-base-uncased", template="The sentiment is [MASK]."):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.template = template
        self device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def predict(self, text, label_words):
        """预测"""
        input_text = self.template.format(text=text)
        
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # 获取[MASK]位置
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (inputs['input_ids'] == mask_token_id).nonzero()
        
        if len(mask_positions) == 0:
            return None
        
        mask_pos = mask_positions[0, 1]
        
        # 获取label words的logit
        word_logits = []
        for label, words in label_words.items():
            for word in words:
                word_id = self.tokenizer.convert_tokens_to_ids(word)
                word_logits.append((label, logits[0, mask_pos, word_id].item()))
        
        # 返回最高概率的label
        best_label = max(word_logits, key=lambda x: x[1])[0]
        
        return best_label
    
    def batch_predict(self, texts, label_words):
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text, label_words)
            results.append(result)
        return results
```

### 4.2 P-Tuning实现

```python
class PTuningPromptLearner(nn.Module):
    """P-Tuning提示学习器"""
    
    def __init__(self, model_name, num_tokens, hidden_dim):
        super().__init__()
        
        from transformers import BertModel, BertTokenizer
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 可学习提示
        self.prompt_embeddings = nn.Embedding(num_tokens, hidden_dim)
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=2, 
            bidirectional=True,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, input_ids, attention_mask=None):
        # 输入嵌入
        input_embeddings = self.model.embeddings(input_ids)
        
        # 可学习提示嵌入
        prompt_emb = self.prompt_embeddings(
            torch.arange(
                self.prompt_embeddings.num_embeddings,
                device=input_ids.device
            ).unsqueeze(0).expand(input_ids.size(0), -1)
        )
        
        # LSTM编码
        prompt_encoded = self.lstm(prompt_emb)[0]
        prompt_encoded = self.mlp(prompt_encoded)
        
        # 拼接
        embeddings = torch.cat([prompt_encoded, input_embeddings], dim=1)
        
        # 通过模型
        outputs = self.model(inputs_embeds=embeddings)
        
        return outputs
```

### 4.3 CoOp实现

```python
class CoOp(nn.Module):
    """CoOp: 可学习的提示"""
    
    def __init__(self, model_name, num_tokens, hidden_dim):
        super().__init__()
        
        from transformers import BertModel
        self.model = BertModel.from_pretrained(model_name)
        
        # 可学习提示向量
        self.prompt_tokens = nn.Parameter(
            torch.randn(num_tokens, hidden_dim)
        )
    
    def forward(self, input_ids, attention_mask=None):
        # 嵌入输入
        input_emb = self.model.embeddings(input_ids)
        
        # 拼接提示
        prompt = self.prompt_tokens.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        
        embeddings = torch.cat([prompt, input_emb], dim=1)
        
        return self.model(inputs_embeds=embeddings, attention_mask=attention_mask)


class CoOpTrainer:
    """CoOp训练器"""
    
    def __init__(self, model_name, num_tokens=10, lr=0.001):
        self.model = CoOp(model_name, num_tokens, 768)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    
    def train_step(self, input_ids, labels):
        """训练步骤"""
        outputs = self.model(input_ids)
        logits = outputs.logits
        
        # 预测token的logit
        loss = nn.CrossEntropyLoss()(
            logits[:, :input_ids.size(1)], 
            labels
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## 5. 代码示例

### 5.1 完整示例

```python
def demo_prompt_learning():
    print("=== Prompt Learning 演示 ===\n")
    
    # 创建模型
    learner = PromptLearner("bert-base-uncased")
    
    # 测试数据
    texts = [
        "This movie is amazing! I love it!",
        "Terrible movie. Wasted my time.",
        "It was okay, nothing special."
    ]
    
    label_words = {
        "positive": ["great", "excellent", "amazing", "good"],
        "negative": ["terrible", "awful", "boring", "bad"]
    }
    
    # 预测
    results = learner.batch_predict(texts, label_words)
    
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Prediction: {result}\n")
    
    return results


if __name__ == "__main__":
    demo_prompt_learning()
```

### 5.2 GPT风格提示

```python
class GPTPromptLearner:
    """GPT风格的提示学习"""
    
    def __init__(self, model_name="gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt, max_length=50):
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def demo_gpt_prompt():
    print("=== GPT Prompt演示 ===\n")
    
    learner = GPTPromptLearner()
    
    # Few-shot提示
    prompt = """Capital of France is Paris
Capital of Japan is Tokyo
Capital of"""
    
    result = learner.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}\n")


if __name__ == "__main__":
    demo_gpt_prompt()
```

### 5.3 自动化提示设计

```python
class AutoPromptLearner:
    """自动提示搜索"""
    
    def __init__(self, model_name, candidate_words):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.candidate_words = candidate_words
    
    def evaluate(self, prompt, texts, labels):
        """评估提示效果"""
        correct = 0
        
        for text, label in zip(texts, labels):
            input_text = prompt.format(text=text)
            inputs = self.tokenizer(input_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # 预测
            pred = self.predict_from_logits(logits, inputs['input_ids'])
            
            if pred == label:
                correct += 1
        
        return correct / len(texts)
    
    def predict_from_logits(self, logits, input_ids):
        """从logits预测"""
        mask_pos = (input_ids == self.tokenizer.mask_token_id).nonzero()
        
        if len(mask_pos) == 0:
            return None
        
        mask_pos = mask_pos[0, 1]
        
        word_ids = []
        for words in self.candidate_words.values():
            for word in words:
                word_ids.append(self.tokenizer.convert_tokens_to_ids(word))
        
        best_word = max(word_ids, key=lambda w: logits[0, mask_pos, w].item())
        
        for label, words in self.candidate_words.items():
            if any(self.tokenizer.convert_tokens_to_ids(w) == best_word for w in words):
                return label
        
        return None
    
    def search_best_prompt(self, base_prompt, texts, labels, n_iterations=20):
        """搜索最佳提示"""
        templates = [
            "{text}. It was [MASK].",
            "The movie is [MASK]. {text}",
            "{text} - [MASK]",
        ]
        
        best_prompt = None
        best_acc = 0
        
        for template in templates:
            acc = self.evaluate(template, texts, labels)
            
            if acc > best_acc:
                best_acc = acc
                best_prompt = template
        
        return best_prompt, best_acc
```

---

## 6. 主流方法对比

| 方法 | 类型 | 可微 | 需要训练 | 效果 |
|------|------|------|---------|------|
| Manual Prompt | 硬提示 | 否 | 否 | 基线 |
| Auto-PT | 硬提示 | 近似 | 搜索 | 好 |
| P-Tuning | 软提示 | 是 | 是 | 很好 |
| CoOp | 软提示 | 是 | 是 | 最佳 |
| Prefix Tuning | 软提示 | 是 | 是 | 好 |

### 6.1 方法详解

**Manual Prompt**：
```python
# 人工设计
prompt = "The sentiment is [MASK]. {text}"
```

**P-Tuning**：
```python
# LSTM编码可学习提示
prompt_emb = lstm(learnable_prompt)
```

**CoOp**：
```python
# 可学习token
prompt = torch.nn.Parameter(num_tokens, dim)
```

**Prefix Tuning**：
```python
# 加在前馈网络的输出上
output = torch.cat([prefix, model(x)], dim=1)
```

---

## 7. 应用场景

### 7.1 文本分类

```python
# 情感分析
templates = {
     "positive": "I love this! [MASK]",
     "negative": "This is bad. [MASK]",
}

# 主题分类
topic_template = "This article is about [MASK]. {text}"

# 意图分类
intent_template = "User says: {text}. Intent: [MASK]"
```

### 7.2 问答

```python
# 问答
qa_template = "Question: {question}? Answer: [MASK]."

# 填空式 QA
cloze_template = "{context} [MASK] is the answer to {question}."
```

### 7.3 信息抽取

```python
# 命名实体识别
ner_template = "{text} [MASK] works at [ORG]."

# 关系抽取
rel_template = "{subject} and {object} have [MASK] relation."
```

### 7.4 代码生成

```python
# 代码完成任务
code_template = """# Write a function to {task}
def solution():"""
```

---

## 8. 常见问题与易错点

### Q1: 为什么大模型适合提示学习？

- 预训练已经学到了丰富知识
- Few-shot能力是涌现现象
- 可通过提示激活不同能力

### Q2: 硬提示和软提示的选择？

- 场景简单 → 硬提示
- 复杂任务 → 软提示
- 资源受限 → CoOp

### Q3: 提示敏感性？

- 同义不同词效果差异大
- 需要多次尝试
- 自动化搜索

### Q4: 如何处理类别映射？

```python
# 一对多映射
label_words = {
    "positive": ["good", "great", "positive"],
    "negative": ["bad", "negative", "terrible"],
}
# 取平均或最大
```

---

## 9. 练习题

### 选择题

1. Prompt Learning的核心优势？
   - A) 效果好   B) 参数少   C) 不需要微调
   - **答案：B**

2. 硬提示和软提示的区别？
   - A) 长度   B) 可学习型   C) 位置
   - **答案：B**

3. CoOp的功能？
   - A) 搜索   B) 生成   C) 可学习提示
   - **答案：C**

### 简答题

1. 为什么Few-shot有效？

   **答案**：预训练模型已经学习了语言模式，只需激活

2. P-Tuning vs CoOp？

   **答案**：P-Tuning用LSTM编码，CoOp直接学习token embedding

### 编程题

实现Auto-Prompt：

```python
def auto_prompt_search(model, templates, data, labels):
    """自动搜索最佳提示"""
    best_template = None
    best_accuracy = 0
    
    for template in templates:
        accuracy = evaluate(model, template, data, labels)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_template = template
    
    return best_template, best_accuracy
```

---

## 10. 学习路径

### 10.1 方法演进

```
Manual Prompt → Auto-PT → P-Tuning → CoOp → Prefix
```

### 10.2 扩展方向

```
单个任务 → 多任务 → 领域适应 → 指令学习
```

---

## 11. 附录

### A. 最佳实践

| 场景 | 推荐方法 |
|------|----------|
| 快速测试 | Manual Prompt |
| 少样本 | Few-shot |
| 高精度 | CoOp |
| 中文 | Chinese-PT |

### B. 参考论文

- GPT-3 (2020). "Language Models are Few-Shot Learners"
- P-Tuning (2021). "GPT Understands"
- CoOp (2021). "Learning to Learn Your Prompts"

---

**文档结束**