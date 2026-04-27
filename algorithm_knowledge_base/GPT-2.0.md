# GPT-2.0 学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
GPT-2.0是OpenAI于2019年发布的大规模自回归语言模型（15亿参数），首次展示了"零样本学习"能力——直接使用语言模型完成下游任务而无需微调，开启了大语言模型时代。

### 1.2 直觉类比
GPT-2就像一个"万能填空高手"——你给它一个开头，它就能续写下去；你给它"请把这句话翻译成中文：Hello → "，它就知道这是翻译任务；你给它"问题：... 答案："，它就知道这是在问答。所有任务都变成了"接龙"游戏。

### 1.3 历史背景
- **2018年6月**：GPT-1发布（1.1亿参数），证明生成式预训练有效
- **2019年2月**：GPT-2发布（15亿参数），发现零样本能力
- **2019年8月**：全部开源（原计划分阶段发布因安全顾虑）
- **2020年6月**：GPT-3发布（1750亿参数），提示学习范式成熟

### 1.4 算法定位
GPT-2.0是**自回归（Autoregressive）语言模型**，使用Transformer解码器架构，通过预测下一个token进行训练和推理。

---

## 2. 核心原理

### 2.1 自回归语言建模
GPT-2的核心是最大化序列的似然概率：

$$P(x) = \prod_{t=1}^{T} P(x_t | x_{<t})$$

训练时：给定前 $t-1$ 个token，预测第 $t$ 个token
推理时：逐个生成token，每次将新token拼接回输入

### 2.2 零样本任务学习
GPT-2的关键发现：一个训练好的语言模型可以"隐式"学习执行各种任务，只需要把任务描述放在输入中（称为"提示"或"prompt"）：

| 任务 | 提示格式 |
|------|----------|
| 翻译 | `Translate English to French: Hello => ` |
| 问答 | `Q: What is gravity? A: ` |
| 摘要 | `Text: ... TL;DR: ` |
| 阅读理解 | `Passage: ... Question: ... Answer: ` |

### 2.3 Transformer解码器架构
GPT-2使用只包含解码器的Transformer，但移除了编码器-解码器注意力：
- **掩码自注意力**：每个token只能看到自己和左侧token
- **层归一化**：在子层之前应用（Pre-LN）
- **激活函数**：GELU替代ReLU
- **词嵌入共享**：输入输出嵌入共享权重

### 2.4 模型规模与配置
| 版本 | 层数 | d_model | 头数 | 参数量 |
|------|------|---------|------|--------|
| Small | 12 | 768 | 12 | 1.17亿 |
| Medium | 24 | 1024 | 16 | 3.45亿 |
| Large | 36 | 1280 | 20 | 7.74亿 |
| XL | 48 | 1600 | 25 | 15亿 |

---

## 3. 数学公式与推导

### 3.1 自回归负对数似然
训练目标是最小化负对数似然：

$$L = -\sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P(x_{i,t} | x_{i,<t}; \theta)$$

其中 $N$ 是样本数，$T_i$ 是第 $i$ 个序列的长度。

### 3.2 掩码自注意力
GPT-2使用因果掩码的缩放点积注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M$ 是因果掩码矩阵：

$$M_{ij} = \begin{cases} 0, & i \geq j \\ -\infty, & i < j \end{cases}$$

这保证了位置 $i$ 只能看到位置 $\leq i$ 的信息。

### 3.3 多尺度注意力
使用多头注意力，每个头关注不同的表示子空间：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中 $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$

### 3.4 缩放规则
随着模型增大，学习率需调整。GPT-2使用余弦学习率衰减：

$$\text{lr}(t) = \text{lr}_{\max} \cdot \frac{1}{2}\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

---

## 4. 训练过程讲解

### 4.1 训练数据：WebText
- 来源：Reddit上获得至少3个karma的帖子链接
- 规模：4500万个网页链接，约40GB文本
- 特点：质量高于Common Crawl，多样性足够

### 4.2 训练过程
1. **数据预处理**：BPE分词（词汇表50257）
2. **序列构建**：将文本分割为1024 token的片段
3. **前向传播**：计算每个位置的下一个token预测概率
4. **损失计算**：交叉熵损失
5. **优化**：Adam优化器，cosine学习率衰减
6. **梯度裁剪**：全局梯度范数裁剪为1.0

### 4.3 训练细节
- **batch大小**：512个序列
- **学习率**：最大2.5e-4，余弦衰减到0
- **训练步数**：100万步
- **硬件**：32个TPU v3核心（XL版本）

---

## 5. 应用场景

| 场景 | 输入 | 输出 |
|------|------|------|
| 文本续写 | 给定开头 | 续写内容 |
| 机器翻译 | "Translate to French: Hello" | "Bonjour" |
| 问答 | "Q: What is the capital of France? A:" | "Paris" |
| 摘要 | "Text: ... TL;DR:" | 摘要结果 |
| 代码生成 | "def fibonacci(n):" | 完整函数 |
| 故事生成 | "Once upon a time" | 完整故事 |

---

## 6. 优缺点分析

### 优点
1. **零样本能力**：无需微调即可执行多种任务
2. **架构简洁**：仅解码器，比编码器-解码器更简单
3. **生成质量高**：文本流畅自然
4. **扩展性好**：模型越大，性能越好（Scaling Law）

### 缺点
1. **单向信息**：只能使用左侧上下文（不能像BERT一样用双向信息）
2. **参数量大**：训练和推理成本高
3. **可控性差**：难以控制生成内容的方向和质量
4. **偏见问题**：训练数据隐含的偏见会被放大

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Generator:
    """GPT-2文本生成器"""
    def __init__(self, model_name="gpt2-xl"):
        """初始化GPT-2模型和分词器"""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def generate(self, prompt, max_length=100, temperature=0.9, top_k=50, top_p=0.95, num_return_sequences=1):
        """
        文本生成
        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            temperature: 采样温度（越低越确定）
            top_k: Top-K采样
            top_p: Nucleus采样
            num_return_sequences: 返回序列数量
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_texts = []
        for i, output in enumerate(outputs):
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts
    
    def zero_shot_task(self, task_type, input_text):
        """
        零样本任务执行
        Args:
            task_type: 任务类型 (translate, qa, summarize)
            input_text: 输入文本
        """
        prompts = {
            "translate": f"Translate English to French: {input_text}\nFrench: ",
            "qa": f"Question: {input_text}\nAnswer: ",
            "summarize": f"{input_text}\nTL;DR: ",
        }
        
        prompt = prompts.get(task_type, input_text)
        result = self.generate(prompt, max_length=50, temperature=0.3)
        return result[0]


class GPT2FineTuner:
    """GPT-2微调器"""
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def prepare_dataset(self, texts, max_length=512):
        """准备训练数据"""
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors="pt"
        )
        return encodings
    
    def train_step(self, input_ids, attention_mask):
        """单步训练"""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        # GPT-2的LM训练：输入和标签相同（shifted）
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=input_ids.to(self.device)  # 自动做shift
        )
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def test_gpt2():
    """测试GPT-2基本功能"""
    generator = GPT2Generator("gpt2")  # 使用small版本（更轻量）
    
    # 文本续写
    result = generator.generate("The future of artificial intelligence", max_length=50)
    print(f"续写结果: {result[0][:200]}...")
    
    # 零样本翻译
    translation = generator.zero_shot_task("translate", "Hello, how are you?")
    print(f"翻译: {translation}")
    
    # 零样本问答
    answer = generator.zero_shot_task("qa", "What is the capital of Japan?")
    print(f"问答: {answer}")
    
    print("GPT-2测试通过！")

if __name__ == "__main__":
    test_gpt2()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class HandwrittenGPT2(nn.Module):
    """GPT-2核心逻辑手工实现（单层Transformer解码器）"""
    def __init__(self, vocab_size=50257, d_model=768, nhead=12, num_layers=12, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 词嵌入 + 位置嵌入（GPT-2使用可学习位置编码）
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化（Pre-LN风格）
        self.final_norm = nn.LayerNorm(d_model)
        
        # 输出投影（与输入嵌入共享权重）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # 权重绑定
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [B, L]
        """
        B, L = input_ids.shape
        
        # 嵌入
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x = token_emb + pos_emb
        
        # 因果掩码
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=input_ids.device), diagonal=1
        )
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.final_norm(x)
        
        # LM预测头
        logits = self.lm_head(x)  # [B, L, vocab_size]
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """
        自回归生成
        Args:
            input_ids: [B, L] 初始输入
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 如果超过最大长度，截取最后max_len个token
                if input_ids.shape[1] > self.max_len:
                    input_ids = input_ids[:, -self.max_len:]
                
                logits = self.forward(input_ids)
                next_logits = logits[:, -1, :] / temperature
                
                # 采样
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # 如果生成EOS token则停止
                if next_token.item() == 50256:  # GPT-2的EOS
                    break
                    
        return input_ids


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层（GPT-2风格）"""
    def __init__(self, d_model, nhead, d_ff=3072, dropout=0.1):
        super().__init__()
        # Pre-LN: 在子层之前做归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, causal_mask):
        # 1. 掩码自注意力（Pre-LN）
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = residual + self.dropout1(attn_out)
        
        # 2. 前馈网络（Pre-LN）
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout2(ffn_out)
        
        return x


def test_handwritten_gpt2():
    """测试手工GPT-2"""
    # 使用小配置
    model = HandwrittenGPT2(vocab_size=10000, d_model=256, nhead=4, num_layers=4, max_len=128)
    
    # 测试前向传播
    B, L = 2, 32
    input_ids = torch.randint(0, 10000, (B, L))
    logits = model(input_ids)
    print(f"前向输出: {logits.shape}")  # [2, 32, 10000]
    
    # 测试生成
    prompt = torch.randint(0, 10000, (1, 5))
    output = model.generate(prompt, max_new_tokens=10)
    print(f"生成输出长度: {output.shape[1]}")
    
    print("手工GPT-2测试通过！")

if __name__ == "__main__":
    test_handwritten_gpt2()
```

---

## 9. 可视化与结果理解

### 9.1 注意力模式可视化
GPT-2的注意力模式特点：
- **局部注意**：靠近的token互相注意较多
- **句法依赖**：动词对主语有较高注意力
- **长距离依赖**：某些头捕捉长距离依赖关系

### 9.2 生成质量与温度的关系
```python
def demo_temperature_effect(generator, prompt):
    """演示温度对生成质量的影响"""
    for temp in [0.1, 0.5, 1.0, 1.5]:
        result = generator.generate(prompt, temperature=temp, max_length=50)
        print(f"T={temp}: {result[0][:100]}")
```

温度低（0.1）：重复、保守
温度适中（0.7-1.0）：流畅、多样
温度高（1.5+）：随机、可能无意义

---

## 10. 模型评估

### 10.1 困惑度（Perplexity）
$$PPL = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(x_t | x_{<t})\right)$$

```python
def compute_perplexity(model, tokenizer, text):
    """计算困惑度"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss)
    return ppl.item()
```

### 10.2 评估结果
| 模型 | WikiText-2 PPL | 零样本效果 |
|------|---------------|-----------|
| GPT-2 Small | 35.8 | 一般 |
| GPT-2 Medium | 28.1 | 较好 |
| GPT-2 Large | 22.8 | 良好 |
| GPT-2 XL | 19.9 | 优秀 |

---

## 11. 常见问题与易错点

### Q1: GPT-2和BERT的主要区别是什么？
A: GPT-2是自回归（从左到右），使用因果注意力掩码；BERT是自编码（双向），使用MLM训练。GPT-2擅长生成，BERT擅长理解。

### Q2: 为什么GPT-2能零样本学习？
A: 大模型在大量数据上训练后，隐式学习了任务的模式。任务格式（如"Q: ... A: "）在训练数据中频繁出现，模型学会了遵循这种模式。

### Q3: GPT-2位置编码为什么用可学习的？
A: 可学习位置编码比正弦编码更灵活，可以适应不同长度的序列。

### Q4: 为什么GPT-2使用Pre-LN而不是Post-LN？
A: Pre-LN（在子层前归一化）训练更稳定，允许更大的学习率和更高的学习率，减少训练后期的不稳定性。

---

## 12. 学习总结

### 核心贡献
1. **零样本任务学习**：证明大语言模型可以直接从提示中理解任务
2. **Scaling Law的实证**：模型规模增大带来能力质变
3. **仅解码器架构**：简化了Transformer，验证了其强生成能力

### 关键技术点
- 因果掩码自注意力
- 可学习位置编码
- Pre-LN层归一化
- GELU激活函数

---

## 13. 练习题与思考题（含答案）

### 习题1：理解题
解释GPT-2如何用自回归方式实现"零样本翻译"。

**答案**：GPT-2将翻译任务转化为"提示 + 生成"格式。例如输入"Translate English to French: Hello\nFrench:"，GPT-2会在训练数据中学到这种翻译模式——看到"Translate ... to ..."格式后，它会继续生成对应的翻译。这个过程不需要微调，只是利用了语言模型的条件概率分布。

### 习题2：推导题
如果输入序列长度为 $L$，模型层数为 $N$，头数为 $H$，隐藏维度为 $D$，请计算GPT-2一次前向传播的FLOPs。

**答案**：每个自注意力层：$O(L^2 \cdot D + L \cdot D^2)$；每个FFN层：$O(L \cdot D^2)$。总FLOPs ≈ $N \cdot (L^2 \cdot D + 2L \cdot D^2)$。对于GPT-2 XL（N=48, D=1600, L=1024），约为 48 * (1024² * 1600 + 2 * 1024 * 1600²) ≈ 3.3e14 FLOPs。

### 习题3：编程题
实现GPT-2的因果注意力掩码。

**答案**：
```python
def create_causal_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask
```

### 习题4：思考题
GPT-2的零样本能力来自于哪些因素？为什么小模型没有这种能力？

**答案**：主要因素：1）**训练数据规模**：大量高质量数据让模型学到丰富的任务模式；2）**模型容量**：足够的参数可以存储更多"隐式知识"；3）**训练目标**：预测下一个token需要理解广泛的上下文依赖。小模型没有零样本能力是因为：参数量不足以存储足够的任务模式信息，训练数据中的稀疏模式无法被小模型捕获。

---

## 14. 学习路径建议

### 前置知识
- **Transformer**：理解自注意力机制
- **GPT-1**：生成式预训练的开端
- **自回归语言模型**：概率链式法则

### 进阶方向
1. **GPT-3**：1750亿参数，提示学习（Prompt Learning）
2. **InstructGPT**：基于人类反馈的强化学习（RLHF）
3. **ChatGPT**：对话优化的GPT模型
4. **LLaMA**：高效大型语言模型
5. **Chinchilla**：Scaling Laws的重新审视

### 学习路线
```
GPT-1 → GPT-2 → GPT-3 → InstructGPT → ChatGPT / LLaMA
```
