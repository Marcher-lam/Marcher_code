# XLNet 学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
XLNet是CMU和Google Brain于2019年提出的预训练语言模型，通过**置换语言建模（Permutation Language Modeling）**目标，结合自回归（AR）和自编码（AE）的优势，避免了BERT的[MASK]标记带来的预训练-微调差异，在多项NLP任务上超越BERT。

### 1.2 直觉类比
XLNet就像一场"乱序猜词游戏"——给定一句打乱顺序的话"The cat on mat the sat"，你需要按照原始顺序逐个推测每个位置应该是什么词。这既让你理解了全句的上下文（像BERT），又能完整地生成句子（像GPT），而且不需要对话句进行任何修改（没有[MASK]标记）。

### 1.3 历史背景
- **2019年6月**：CMU和Google Brain提出XLNet
- **2019年8月**：发布XLNet-Large，在20个NLP任务上达到SOTA
- **技术基础**：引入Transformer-XL的相对位置编码和片段递归

### 1.4 算法定位
XLNet是**置换语言模型**，结合了自回归和自编码的优点。

---

## 2. 核心原理

### 2.1 置换语言建模（PLM）
关键idea：对一个长度为 $T$ 的序列，考虑所有 $T!$ 种排列顺序。在每种排列下，模型按自回归方式预测每个token：

$$L = \mathbb{E}_{z \sim \mathcal{Z}_T} \left[ \sum_{t=1}^T \log P(x_{z_t} | x_{z_{<t}}) \right]$$

其中 $z$ 是一个排列，$z_{<t}$ 是排列中前 $t-1$ 个位置。

**关键洞察**：模型看到的"上下文"是排列中的前 $t-1$ 个token，而不是原始序列中的左侧token。通过改变排列，token $x_i$ 有时只能看左侧，有时只能看右侧，有时两侧都看——从而学到了双向表示。

### 2.2 双流注意力（Two-Stream Attention）
标准自回归无法直接实现PLM，因为位置编码会泄露目标信息。XLNet引入双流注意力：

**1) 内容流（Content Stream）**
- 传统自注意力
- 编码 $h_{z_t}$：包含位置 $z_t$ 的内容信息
- 查询、键、值都来自内容

**2) 查询流（Query Stream）**
- 只能看到位置信息，不能看到内容
- 编码 $g_{z_t}$：只包含位置 $z_t$ 的位置信息
- 查询来自查询流，键和值来自内容流

### 2.3 Transformer-XL的集成
XLNet复用Transformer-XL的两项技术：
- **相对位置编码**：更好地处理长距离依赖
- **片段递归**：在长文档上保持上下文连续性

---

## 3. 数学公式与推导

### 3.1 置换语言建模的目标函数
给定序列 $x = (x_1, ..., x_T)$，排列 $z$：

$$L = \max_\theta \mathbb{E}_{z \sim \mathcal{Z}_T} \sum_{t=1}^T \log P_\theta(x_{z_t} | x_{z_{<t}})$$

其中排列 $z$ 是随机采样，并非所有 $T!$ 种排列——通常只采样一部分。

### 3.2 双流注意力计算
初始状态：
$$h_i^{(0)} = e(x_i) \quad \text{(内容流：词嵌入)}$$
$$g_i^{(0)} = w \quad \text{(查询流：可学习向量)}$$

第 $m$ 层：
$$\text{内容流: } h_{z_t}^{(m)} = \text{Attention}(Q=h_{z_t}^{(m-1)}, KV=h_{z_{<t}}^{(m-1)})$$
$$\text{查询流: } g_{z_t}^{(m)} = \text{Attention}(Q=g_{z_t}^{(m-1)}, KV=h_{z_{<t}}^{(m-1)})$$

最终预测：
$$P(x_{z_t} | x_{z_{<t}}) = \text{softmax}(e(x)^T g_{z_t}^{(M)})$$

### 3.3 部分预测（Partial Prediction）
为降低计算量，只预测排列中最后 $1/K$ 的token：

$$L = \mathbb{E}_{z \sim \mathcal{Z}_T} \sum_{t=c+1}^T \log P(x_{z_t} | x_{z_{<t}})$$

其中 $c = \lfloor T/K \rfloor$，$K$ 通常设为7（预测后1/7的token）。

### 3.4 相对位置编码
XLNet使用Transformer-XL的相对位置编码：

$$A_{i,j} = q_i^T k_j + q_i^T W_{k,R} R_{i-j} + u^T k_j + v^T W_{k,R} R_{i-j}$$

其中 $R_{i-j}$ 是相对位置正弦编码，$u, v$ 是可学习参数。

---

## 4. 训练过程讲解

### 4.1 预训练步骤
1. 从语料中采样序列 $x$
2. 随机采样一个排列 $z$
3. 初始化双流注意力
4. 按排列顺序前向传播
5. 计算部分预测损失（只预测后1/K token）
6. 反向传播更新参数

### 4.2 双流注意力切换
- 训练时：查询流和内容流并行计算
- 微调时：只使用内容流（查询流不需要）
- 微调时的输入不需要排列，按原始顺序处理

### 4.3 训练效率优化
- 使用部分预测减少计算量
- 共享同一排列内 $z_{<c}$ 的表示

---

## 5. 应用场景

| 场景 | 说明 | XLNet优势 |
|------|------|-----------|
| 阅读理解 | SQuAD 2.0 | 双向上下文建模 |
| 文本分类 | GLUE | 无[MASK]差异 |
| 自然语言推理 | MNLI | 深度语义理解 |
| 文本生成 | 需要微调 | 自回归架构 |

---

## 6. 优缺点分析

### 优点
1. **双向+自回归**：同时拥有BERT的双向理解和GPT的生成能力
2. **无[MASK]差异**：不引入人工[MASK]标记，预训练-微调一致
3. **长文本处理**：继承Transformer-XL的长序列能力

### 缺点
1. **计算量大**：双流注意力需要2倍计算
2. **训练复杂**：排列采样和双流管理增加工程难度
3. **生成不如GPT**：单向自回归并非设计初衷
4. **实现困难**：代码复杂度高，不易复现

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetModel, XLNetTokenizer, XLNetLMHeadModel
import math

class XLNetClassifier(nn.Module):
    """XLNet文本分类器"""
    def __init__(self, num_classes=2, model_name='xlnet-base-cased'):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained(model_name)
        self.config = self.xlnet.config
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.d_model, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # XLNet使用last hidden state中的最后一个token
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 取最后一个token作为序列表示（XLNet约定）
        last_token = outputs.last_hidden_state[:, -1, :]
        pooled = self.dropout(last_token)
        logits = self.classifier(pooled)
        return logits


class XLNetGenerator:
    """XLNet文本生成器"""
    def __init__(self, model_name='xlnet-base-cased'):
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetLMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def generate(self, prompt, max_length=100, temperature=0.8, top_k=40):
        """文本生成（XLNet是双向模型，生成需要特殊处理）"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class XLNetTwoStreamAttention(nn.Module):
    """XLNet双流注意力手工实现（核心机制）"""
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # 内容流投影
        self.content_q = nn.Linear(d_model, d_model)
        self.content_k = nn.Linear(d_model, d_model)
        self.content_v = nn.Linear(d_model, d_model)
        
        # 查询流投影
        self.query_q = nn.Linear(d_model, d_model)
        
        # 输出
        self.output = nn.Linear(d_model, d_model)
        
    def forward(self, content, query, attn_mask=None, content_mask=None):
        """
        Args:
            content: [B, L, D] 内容流（含词信息）
            query: [B, L, D] 查询流（仅位置信息）
        """
        B, L, D = content.shape
        
        # 内容流注意力
        c_q = self.content_q(content).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        c_k = self.content_k(content).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        c_v = self.content_v(content).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        
        # 查询流注意力
        g_q = self.query_q(query).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        
        # 内容流输出（标准自注意力）
        c_scores = torch.matmul(c_q, c_k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            c_scores = c_scores + attn_mask.unsqueeze(0).unsqueeze(0)
        c_attn = F.softmax(c_scores, dim=-1)
        c_out = torch.matmul(c_attn, c_v)
        c_out = c_out.transpose(1, 2).contiguous().view(B, L, D)
        c_out = self.output(c_out)
        
        # 查询流输出（Q来自查询流，KV来自内容流）
        g_scores = torch.matmul(g_q, c_k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if content_mask is not None:
            g_scores = g_scores + content_mask.unsqueeze(0).unsqueeze(0)
        g_attn = F.softmax(g_scores, dim=-1)
        g_out = torch.matmul(g_attn, c_v)
        g_out = g_out.transpose(1, 2).contiguous().view(B, L, D)
        g_out = self.output(g_out)
        
        return c_out, g_out


def test_xlnet():
    """测试XLNet分类器"""
    classifier = XLNetClassifier(num_classes=3)
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    
    texts = ["I love this movie!", "This is terrible.", "It's okay."]
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        logits = classifier(inputs.input_ids, inputs.attention_mask)
        pred = logits.argmax(dim=-1).item()
        print(f"文本: '{text}' → 预测类别: {pred}")
    
    print("XLNet测试通过！")

if __name__ == "__main__":
    test_xlnet()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools

class HandwrittenXLNetPLM(nn.Module):
    """XLNet置换语言建模核心逻辑"""
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.word_embed = nn.Embedding(vocab_size, d_model)
        self.query_start = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            TwoStreamAttentionLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, perm_mask):
        """
        Args:
            input_ids: [B, L]
            perm_mask: [B, L, L] 排列掩码（1=不可见）
        """
        B, L = input_ids.shape
        
        # 内容流
        content = self.word_embed(input_ids)  # [B, L, D]
        
        # 查询流（初始为可学习向量）
        query = self.query_start.expand(B, L, -1)
        
        # 通过层
        for layer in self.layers:
            content, query = layer(content, query, perm_mask)
        
        # 最终预测（使用查询流）
        query = self.norm(query)
        logits = self.output(query)  # [B, L, vocab_size]
        
        return logits
    
    def generate_permutation_mask(self, batch_size, seq_len, pred_len=None):
        """
        生成置换语言建模的注意力掩码
        Args:
            pred_len: 每个排列中需要预测的token数（后1/K）
        """
        if pred_len is None:
            pred_len = seq_len // 7  # 预测后1/7
        
        masks = []
        for _ in range(batch_size):
            # 随机排列
            perm = torch.randperm(seq_len)
            
            # 掩码: 1表示不可见
            mask = torch.ones(seq_len, seq_len)
            for i in range(seq_len):
                # 可以看到排列中在它之前的token
                pos = perm[i]
                # perm[:i] 是在排列中排在pos之前的token
                mask[pos, perm[:i]] = 0  # 可见
            
            # 只预测最后pred_len个token
            non_pred = perm[:-pred_len] if pred_len > 0 else perm
            mask[non_pred, :] = 1  # 非预测位置所有内容不可见
            
            masks.append(mask)
        
        return torch.stack(masks).float()  # [B, L, L]


class TwoStreamAttentionLayer(nn.Module):
    """双流注意力层"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.content_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.query_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
    def forward(self, content, query, mask):
        # 内容流
        c_out, _ = self.content_attn(content, content, content, attn_mask=mask)
        content = self.norm(content + c_out)
        ffn_out = self.ffn(content)
        content = self.norm(content + ffn_out)
        
        # 查询流
        q_out, _ = self.query_attn(query, content, content, attn_mask=mask)
        query = self.norm(query + q_out)
        # 注意：查询流的FFN与内容流共享
        
        return content, query


def test_xlnet_handwritten():
    model = HandwrittenXLNetPLM(vocab_size=5000, d_model=256, nhead=4, num_layers=3)
    B, L = 2, 10
    
    ids = torch.randint(0, 5000, (B, L))
    mask = model.generate_permutation_mask(B, L)
    
    logits = model(ids, mask)
    print(f"手工XLNet输出: {logits.shape}")

if __name__ == "__main__":
    test_xlnet_handwritten()
```

---

## 9. 可视化与结果理解

### 9.1 排列示例
原始序列: [x1, x2, x3, x4, x5]
排列: [3, 1, 4, 2, 5]
预测顺序: x3 → x1 → x4 → x2 → x5
x3看到: []（第一个，没有上下文）
x1看到: [x3]
x4看到: [x3, x1]
x2看到: [x3, x1, x4]
x5看到: [x3, x1, x4, x2]

### 9.2 与BERT对比
- BERT中x3看到所有token（包括自身被mask）
- XLNet中x3在不同的排列中看到不同的上下文
- 通过多样化的排列，x3学会了利用双向信息

---

## 10. 模型评估

| 模型 | RACE | SQuAD 2.0 | MNLI | IMDB |
|------|------|-----------|------|------|
| BERT-Large | 72.0 | 83.2 | 86.6 | 95.7 |
| XLNet-Large | 83.2 | 88.8 | 89.7 | - |

---

## 11. 常见问题

### Q1: 为什么XLNet在RACE上提升巨大（72.0→83.2）？
A: RACE是长文本阅读理解，需要深度双向理解。XLNet通过PLM实现了真正的双向建模，而BERT的MLM仍然依赖[MASK]标记。

### Q2: 双流注意力和标准注意力的区别？
A: 标准注意力只有一个流（内容流）。双流增加查询流，在预测时只使用位置信息不使用词信息，防止信息泄露。

### Q3: XLNet在微调时如何使用？
A: 微调时不需要排列，也不需要双流注意力。只使用内容流，按原始序列顺序处理。

---

## 12. 学习总结

XLNet的核心贡献是**置换语言建模**——在不引入[MASK]的前提下实现了双向上下文建模。双流注意力机制是其工程实现的关键创新。

---

## 13. 练习题

### 习题1：为什么XLNet需要双流注意力？
**答案**：防止信息泄露。在自回归预测时，如果使用标准注意力，当前位置的词信息会泄露到预测中（模型只需复制当前词即可"预测"正确）。

### 习题2：部分预测为什么只需要预测后1/K的token？
**答案**：减少计算量。前面c个token有完整的上下文依赖，但仍被建模；只有最后的token在损失中计算。

### 习题3：推导XLNet的预测分布公式。
**答案**：
$$P(x_{z_t} | x_{z_{<t}}) = \frac{\exp(e(x)^T g_{z_t})}{\sum_{x'}\exp(e(x')^T g_{z_t})}$$
其中 $g_{z_t}$ 是查询流在位置 $z_t$ 的表示。

### 习题4：XLNet和BERT在预训练-微调一致性上有什么差异？
**答案**：BERT在预训练时使用[MASK]标记，但微调时没有。XLNet不使用[MASK]，预训练和微调的输入格式完全一致。

---

## 14. 学习路径建议

### 前置
- BERT、Transformer-XL、自回归语言模型

### 平行
- RoBERTa、ELECTRA、ALBERT

### 进阶
- DeBERTa、XLM-R
