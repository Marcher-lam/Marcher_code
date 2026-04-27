# GPT（Generative Pre-Training）学习文档

> 单向自回归语言模型，通过无监督预训练+有监督微调，引领大语言模型时代。

## 1. 算法基础认知

### 一句话定义

GPT是一种基于Transformer解码器的单向自回归语言模型，通过预测下一个token进行预训练，然后在下游任务上进行微调。

### 直觉类比

GPT就像一个"接话茬"高手——读了一段文字后，预测下一个最可能出现的词。它从左到右逐词生成，每次只利用前面的信息。

### 历史背景

- **2018年6月**：OpenAI发布GPT-1（1.17亿参数）
- **2019年2月**：GPT-2发布（15亿参数）
- **2020年6月**：GPT-3发布（1750亿参数）
- **2022年11月**：ChatGPT发布，引领AI浪潮
- **2024年**：GPT-4多模态模型

### 算法定位

GPT是**生成式预训练模型**，属于自监督学习/生成学习。

---

## 2. 核心原理

### 核心思想

GPT的核心是**单向自回归**——每个位置的预测只能看到左侧的所有token。这与BERT的双向建模形成对比。

### 工作流程

**预训练**：最大化语言模型的对数似然
$$\mathcal{L}_{LM} = \sum_i \log P(x_i | x_{<i}; \theta)$$

**微调**：
$$\mathcal{L}_{fine} = \mathcal{L}_{LM} + \lambda \mathcal{L}_{CLS}$$

### 架构对比

| 特性 | GPT | BERT |
|------|-----|------|
| 方向 | 单向（向左） | 双向 |
| 架构 | Transformer解码器 | Transformer编码器 |
| 预训练 | 语言建模 | MLM + NSP |
| 适用 | 文本生成 | 理解任务 |
| 参数利用率 | 高 | 中 |

---

## 3. 数学公式与推导

### 语言建模目标

给定token序列 $x = (x_1, x_2, ..., x_n)$，语言模型目标是：
$$\mathcal{L}(x) = \sum_{i=1}^{n} \log P(x_i | x_{<i}; \theta)$$

### GPT前向计算

$$h_0 = W_e x + W_p$$
$$h_l = \text{TransformerBlock}(h_{l-1})$$
$$P(x) = \text{softmax}(h_n W_e^T)$$

其中：
- $W_e$：词嵌入矩阵
- $W_p$：位置嵌入矩阵

---

## 4. 训练过程讲解

### 超参数演进

| 版本 | 参数规模 | 层数 | 维度 | 头数 | 上下文长度 |
|------|----------|------|------|------|------------|
| GPT-1 | 117M | 12 | 768 | 12 | 512 |
| GPT-2 | 1.5B | 48 | 1600 | 25 | 1024 |
| GPT-3 | 175B | 96 | 12288 | 96 | 4096 |
| GPT-4 | ~1.7T+ | - | - | - | 32K+ |

### 训练技巧

1. **大batch**：GPT-3使用3.2M tokens/batch
2. **学习率调度**：使用cosine decay
3. **混合精度**：FP16加速
4. **梯度检查点**：节省显存

---

## 5. 应用场景

1. **文本生成**：文章、代码、诗歌
2. **对话系统**：ChatGPT
3. **代码补全**：GitHub Copilot
4. **翻译**：零样本翻译
5. **推理**：Chain-of-Thought
6. **多模态**：GPT-4V

---

## 6. 优缺点分析

### 优点

1. **强大生成能力**：文本流畅自然
2. **零样本学习**：无需微调可做任务
3. **涌现能力**：大模型出现推理能力
4. **通用性**：一个模型处理多种任务

### 缺点

1. **单向建模**：无法利用右侧上下文
2. **训练成本**：GPT-3训练需数百万美元
3. **长序列限制**：无法处理超长文档
4. **幻觉问题**：可能生成错误信息

---

## 7. 调库实现

```python
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class GPT2Generator:
    """GPT-2文本生成器"""
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.model.eval()
        
    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50):
        """条件文本生成"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.last_hidden_state[:, -1, :] / temperature
            
            # Top-k采样
            top_k = min(top_k, logits.size(-1))
            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
            
            probs = torch.softmax(top_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 转换为token
            generated_id = top_indices.gather(-1, next_token)
            generated_text = self.tokenizer.decode(generated_id[0])
            
        return generated_text

# 测试
if __name__ == "__main__":
    generator = GPT2Generator()
    result = generator.generate("In a world where", max_length=20)
    print(f"生成文本: {result}")
```

---

## 8. 手工代码实现

```python
import numpy as np

class SimplifiedGPT:
    """简化的GPT实现"""
    
    def __init__(self, vocab_size, d_model=768, num_layers=12):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 简化的词嵌入和位置嵌入
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.position_embedding = np.random.randn(512, d_model) * 0.02
        
        # 输出层
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
        
    def forward(self, input_ids):
        """前向传播（单向自回归）"""
        seq_len = input_ids.shape[1]
        
        # 嵌入
        token_emb = np.array([[self.token_embedding[t] for t in seq] 
                             for seq in input_ids])
        pos_emb = self.position_embedding[:seq_len]
        
        hidden = token_emb + pos_emb
        
        # 简化Transformer处理
        # 实际需要完整的TransformerBlock
        
        # 语言模型头
        logits = np.dot(hidden, self.lm_head)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=20, temperature=1.0):
        """自回归生成"""
        for _ in range(max_new_tokens):
            # 只用左侧序列预测下一个
            logits = self.forward(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            probs = np.exp(next_token_logits - np.max(next_token_logits))
            probs = probs / np.sum(probs)
            
            next_token = np.random.choice(self.vocab_size, p=probs)
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
            
        return input_ids

# 测试
if __name__ == "__main__":
    np.random.seed(42)
    gpt = SimplifiedGPT(10000)
    input_ids = np.array([[1, 2, 3]])  # 假设1,2,3是特定token
    generated = gpt.generate(input_ids, max_new_tokens=10)
    print(f"生成token序列: {generated}")
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_gpt_architecture():
    """可视化GPT架构"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    components = [
        (5, 11.5, "输入序列", "lightblue"),
        (5, 10.5, "词嵌入 + 位置编码", "lightgreen"),
        (5, 9, "Masked Self-Attention × N", "lightyellow"),
        (5, 7, "残差连接 + LayerNorm", "lightcoral"),
        (5, 5, "FFN + 残差 + LayerNorm", "lightcoral"),
        (5, 3, "线性层 + Softmax", "lightgray"),
        (5, 1.5, "输出token预测", "lightblue"),
    ]
    
    for x, y, text, color in components:
        rect = plt.Rectangle((x-2, y-0.5), 4, 1, 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10)
    
    # 箭头
    for i in range(len(components)-1):
        ax.annotate('', 
                   xy=(components[i+1][0], components[i+1][1]+0.5), 
                   xytext=(components[i][0], components[i][1]-0.5),
                   arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.title("GPT 架构图", fontsize=14)
    plt.tight_layout()
    plt.savefig('gpt_architecture.png', dpi=150)
    plt.show()

def plot_model_scaling():
    """模型规模与性能关系"""
    params = [0.1, 0.3, 0.7, 1.5, 3, 6, 175]  # 十亿参数
    # 模拟各项任务性能趋势
    zero_shot = [15, 22, 35, 50, 65, 78, 95]
    
    plt.figure(figsize=(10, 6))
    plt.plot(params, zero_shot, 'o-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('参数量（十亿）')
    plt.ylabel('Zero-shot 准确率 (%)')
    plt.title('GPT模型规模 vs 性能')
    plt.grid(True, alpha=0.3)
    plt.savefig('gpt_scaling.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    visualize_gpt_architecture()
    plot_model_scaling()
```

---

## 10. 常见问题与易错点

1. **训练不收敛**：检查学习率和batch size
2. **生成重复**：使用temperature和top-p采样
3. **显存溢出**：使用梯度累积和混合精度
4. **长序列生成**：使用KV cache优化

---

## 11. 学习总结

GPT系列开创了**生成式预训练**范式。从GPT-1到GPT-4，展示了大模型"涌现"能力的可能。其核心思想——预测下一个token——简单而强大，已成为大语言模型的基础。

---

## 12. 练习题

1. **基础**：GPT和BERT的核心区别是什么？
2. **进阶**：为什么GPT需要masked attention而BERT不需要？
3. **开放**：如何解决大模型的"幻觉"问题？

---

## 13. 学习路径

- 前置：Transformer解码器、语言模型
- 平行：GPT-2、GPT-3、ChatGPT
- 进阶：LLaMA、Claude、PaLM