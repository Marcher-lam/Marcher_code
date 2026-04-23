# LLaMA 学习文档

> Meta 开源的大语言模型，70B 参数性能超越 GPT-3。

---

## 1. 算法基础认知

### 1.1 发展背景

LLaMA（Large Language Model Meta AI）由 Meta AI 于 2023 年发布，是一系列开源大语言模型。LLaMA-13B 在多数基准测试上超越了 GPT-3（175B），开启了开源大模型的新时代。

### 1.2 核心定位

| 模型 | 参数量 | 上下文 | 训练数据 |
|------|--------|--------|----------|
| LLaMA-7B | 7B | 2048 | 1T token |
| LLaMA-13B | 13B | 2048 | 1T token |
| LLaMA-33B | 33B | 2048 | 1.4T token |
| LLaMA-65B | 65B | 2048 | 1.4T token |

### 1.3 核心创新

- **SwiGLU 激活**：结合 Swish 和 GLU
- **RoPE 旋转位置编码**：高效的位置编码
- **RMSNorm**：更稳定的归一化

---

## 2. 核心原理

### 2.1 架构

Transformer Decodec + 改进：
- Pre-norm RMSNorm
- SwiGLU 激活函数
- RoPE 旋转位置编码
- 注意力机制优化

### 2.2 SwiGLU

```python
# Swish + GLU 组合
def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(x) * gate
```

### 2.3 RoPE

位置编码公式：
$$f_{q,k}(m) = \begin{pmatrix} \cos(m\theta_i) & \sin(m\theta_i) \\ -\sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

---

## 3. 数学公式与推导

### 3.1 RMSNorm

$$x = \frac{x}{\sqrt{mean(x^2) + \epsilon}} \cdot \gamma$$

### 3.2 Attention 计算

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 3.3 训练损失

$$\mathcal{L} = -\sum_{i=1}^{N} \log P(x_i | x_{<i})$$

---

## 4. 训练过程讲解

### 4.1 预训练数据

| 数据集 | 比例 |
|--------|------|
| CommonCrawl | 67% |
| GitHub | 15% |
| Wikipedia | 4.5% |
| Books | 4.5% |
| ArXiv | 4.5% |

### 4.2 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| Batch | 4000K tokens |
| 预热 | 2000 steps |

---

## 5. 应用场景

### 5.1 典型应用

- **对话系统**：智能助手
- **文本生成**：文章写作
- **代码生成**：编程辅助

### 5.2 本地部署

```python
# 使用 llama.cpp 本地部署
from llama_cpp import Llama

model = Llama(model_path="models/7B/ggml-model.bin")
output = model("Explain quantum computing")
```

---

## 6. 调库实现

### 6.1 HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/LLaMA-7B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-7B")

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
```

### 6.2 本地推理

```python
class LLaMA:
    """LLaMA 本地模型"""
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
        
    def generate(self, prompt, max_tokens=100):
        return self.model.complete(prompt, max_tokens)
```

---

## 7. 手工代码实现

### 7.1 简化架构

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """RMSNorm 归一化"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU 激活"""
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        x1 = self.w1(x)
        x3 = self.w3(x)
        return self.w2(F.silu(x1) * x3)


class LLaMAModel(nn.Module):
    """简化 LLaMA"""
    
    def __init__(self, vocab_size, dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
```

---

## 8. 优缺点分析

### 8.1 优点

1. **开源**：可自由使用
2. **高效**：小模型大性能
3. **多语言**：多语言训练

### 8.2 缺点

1. **推理慢**：需要 GPU
2. **显存**：需要大量显存

### 8.3 改进方向

- LLaMA 2: 更好指令微调
- Code LLaMA: 代码专用

---

## 9. 可视化与结果理解

### 9.1 性能对比

```python
def plot_performance():
    import matplotlib.pyplot as plt
    
    models = ['GPT-3', 'PaLM', 'LLaMA-13B', 'LLaMA-65B']
    scores = [56, 58, 62, 68]
    
    plt.figure(figsize=(8, 5))
    plt.bar(models, scores, color='steelblue')
    plt.ylabel('MMLU 分数')
    plt.title('大模型性能对比')
    plt.ylim(50, 75)
    plt.tight_layout()
    plt.savefig('llama_perf.png')
```

---

## 10. 模型评估

### 10.1 基准测试

| 模型 | MMLU | GSM8K |
|------|------|-------|
| GPT-3 | 56 | 35 |
| LLaMA-33B | 63 | 73 |
| LLaMA-65B | 68 | 81 |

---

## 11. 学习总结

**核心要点**：

1. **开源**：Meta 开源
2. **高效**：小模型高性能  
3. **改进**：SwiGLU + RoPE
4. **70B vs GPT-3**：性能超越

**LLaMA 核心优势**：
- 开源可商用
- 高效率
- 高性能

**学习建议**：

1. 理解 Transformer
2. 掌握改进点
3. 本地部署

---

## 12. 练习题与思考题

### 12.1 基础练习

1. LLaMA vs GPT 区别
2. SwiGLU 原理

### 12.2 思考题

1. LLaMA 的商业影响

---

## 14. 学习路径建议

### 入门阶段

1. Transformer 基础
2. 大模型概念

### 进阶阶段

1. LLaMA 架构
2. 本地部署

### 高级阶段

1. 微调 LLaMA
2. 领域应用

**推荐路线**：

```
GPT-2 → GPT-3 → LLaMA → Alpaca → Vicuna
```

**LLaMA 开启了大模型开源时代，熟练掌握它对学习和应用大模型很重要。**