# MASS（Masked Sequence-to-Sequence Pre-training）学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
MASS（Masked Sequence-to-Sequence Pre-training）是微软于2019年提出的Seq2Seq预训练框架，通过mask连续一段原文并让解码器预测被mask部分来学习语言表示，是BART同期且类似的独立工作。

### 1.2 直觉类比
MASS就像"填空式翻译训练"——把英语句子中连续的一段词涂掉，让模型看到前半句和后半句，然后在法语端（解码器侧）填出被涂掉的部分。这种训练迫使编码器理解上下文，解码器学会生成。

### 1.3 历史背景
- **2019年6月**：微软提出MASS（与BART几乎同期）
- **核心创新**：首次将mask策略系统地应用于Seq2Seq预训练
- **后续影响**：启发了mBART等跨语言Seq2Seq预训练模型

### 1.4 算法定位
MASS是**Seq2Seq去噪自编码模型**，专注提升文本生成（特别是机器翻译）能力。

---

## 2. 核心原理

### 2.1 连续mask策略
MASS的核心是mask一段连续的token序列（不是BERT那样离散地mask单个token）：

- **编码器输入**：带mask的源序列 $x^{\backslash u:v}$（从 $u$ 到 $v$ 被mask）
- **解码器输入**：仅包含连续的[MASK] token
- **解码器目标**：预测被mask的部分 $x^{u:v}$

### 2.2 mask长度 $k$ 的影响
$k = v - u + 1$ 是关键超参数：
- $k = 1$：退化为BERT的MLM
- $k = \text{seq\_len}$：退化为GPT的语言模型
- $k = \text{seq\_len}/2$：最优（平衡理解和生成）

### 2.3 编码器-解码器协同
MASS让编码器专注理解（编码unmasked部分），解码器专注生成（预测masked部分），实现了比BERT/GPT更好的协同。

---

## 3. 数学公式与推导

### 3.1 预训练目标
给定原始序列 $x = (x_1, ..., x_T)$，mask位置从 $u$ 到 $v$，长度为 $k = v-u+1$：

编码器输入 $x^{\backslash u:v}$ 将位置 $u$ 到 $v$ 替换为[MASK]。
解码器输入为 $k$ 个[MASK] token。

目标函数：

$$L(\theta; x) = -\log P(x^{u:v} | x^{\backslash u:v}; \theta)$$

$$= -\sum_{t=u}^{v} \log P(x_t | x_{<t}^{u:v}, x^{\backslash u:v}; \theta)$$

### 3.2 k=1的特殊情况
当 $k=1$ 时，MASS退化为BERT的MLM：
编码器mask一个token，解码器只有1步，实际只需要编码器的表示。

$$L_{k=1}(\theta; x) = -\sum_{i=1}^{T} \log P(x_i | x^{\backslash i}; \theta)$$

### 3.3 k=T的特殊情况
当 $k=T$ 时，编码器输入全部为[MASK]，解码器需要从零生成完整序列：

$$L_{k=T}(\theta; x) = -\sum_{t=1}^{T} \log P(x_t | x_{<t}, [MASK]_1^T; \theta)$$

这等价于GPT的语言模型训练（但需要额外的解码器）。

### 3.4 最优k的理论分析
MASS的 $k$ 控制了编码器和解码器之间的信息分配：
- 编码器获得的信息量 $= T - k$
- 解码器需要生成的信息量 $= k$
- 当 $k \approx T/2$ 时，编码器和解码器都获得充分的训练信号

---

## 4. 训练过程讲解

### 4.1 预训练步骤
1. **数据准备**：从大规模单语语料采样序列
2. **确定mask位置**：随机选择起始位置 $u$ 和长度 $k$
3. **构造编码器输入**：将 $x_{u...v}$ 替换为[MASK]
4. **构造解码器输入**：$k$ 个连续的[MASK]
5. **前向传播**：编码器编码带mask的序列，解码器预测被mask部分
6. **损失计算**：仅对被mask的token计算交叉熵
7. **反向传播**：更新编码器和解码器参数

### 4.2 机器翻译微调
MASS特别擅长机器翻译微调：
1. 加载预训练参数初始化NMT模型
2. 输入源语言句子（两端语言不同，但架构相同）
3. 解码器生成目标语言翻译
4. 微调所有参数

### 4.3 与BERT/GPT的训练对比
| 模型 | 编码器输入 | 解码器输入 | 预测目标 |
|------|-----------|-----------|----------|
| BERT | x with [MASK] | 无解码器 | 单个mask token |
| GPT | x (from left) | 无解码器 | 下一个token |
| MASS | x with span [MASK] | k个[MASK] | span内所有token |

---

## 5. 应用场景

| 场景 | 描述 |
|------|------|
| 机器翻译 | MASS的核心应用，显著提升NMT效果 |
| 文本摘要 | 微调后可用于生成式摘要 |
| 对话生成 | 适用于开放域对话系统 |
| 文本去噪 | 恢复被破坏的文本 |

---

## 6. 优缺点分析

### 优点
1. **统一框架**：将BERT和GPT统一在Seq2Seq中
2. **生成能力强**：k≈T/2时编码和解码能力均衡
3. **翻译提升显著**：在WMT翻译任务上提升+2~3 BLEU

### 缺点
1. **k值敏感**：需要调优mask长度
2. **训练复杂**：需要编码器-解码器同时训练
3. **单语言限制**：只在源语言进行mask

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MASSConfig, MASSModel
import math

class MASSPretraining(nn.Module):
    """MASS预训练模型"""
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = MASSConfig()
        self.model = MASSModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.shared.weight  # 权重绑定
        
    def forward(self, encoder_input_ids, decoder_input_ids):
        """
        Args:
            encoder_input_ids: 编码器输入（带mask）[B, src_len]
            decoder_input_ids: 解码器输入（连续[MASK]）[B, k]
        """
        outputs = self.model(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True
        )
        
        # 解码器输出 → 词汇表预测
        logits = self.lm_head(outputs.last_hidden_state)
        return logits


class MASSTextInfiller:
    """MASS文本填充器"""
    def __init__(self, model_name="microsoft/mass-base"):
        from transformers import MASSForConditionalGeneration, MASSTokenizer
        self.model = MASSForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = MASSTokenizer.from_pretrained(model_name)
        
    def infill(self, text_with_mask, max_length=50):
        """
        填充被mask的文本
        Example: "The [MASK] sat on the [MASK]"
        """
        inputs = self.tokenizer(text_with_mask, return_tensors="pt")
        
        # 由于MASS解码器也接受[MASK]，需要特殊处理
        # 这里使用条件生成的方式
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def translate(self, source_text, src_lang="en", tgt_lang="fr"):
        """机器翻译"""
        inputs = self.tokenizer(source_text, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class MASSMaskGenerator:
    """MASS连续mask生成器"""
    def __init__(self, mask_token_id, pad_token_id):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        
    def generate_mask(self, input_ids, mask_ratio=0.5):
        """
        生成MASS风格的连续mask
        Args:
            input_ids: [B, L]
            mask_ratio: mask比例（k/T ≈ 0.5）
        Returns:
            encoder_input: 编码器输入（带mask）
            decoder_input: 解码器输入（连续[MASK]）
            labels: 需要预测的token
        """
        B, L = input_ids.shape
        k = int(L * mask_ratio)
        
        # 随机选择起始位置
        u = torch.randint(0, L - k, (B, 1))
        
        encoder_input = input_ids.clone()
        decoder_input = []
        labels = []
        
        for b in range(B):
            start = u[b].item()
            # mask编码器输入
            encoder_input[b, start:start+k] = self.mask_token_id
            # 解码器输入：k个[MASK]
            decoder_input.append(torch.full((k,), self.mask_token_id))
            # 标签：原始token
            labels.append(input_ids[b, start:start+k])
        
        decoder_input = torch.stack(decoder_input)
        labels = torch.stack(labels)
        
        return encoder_input, decoder_input, labels


def test_mass():
    """测试MASS基本功能"""
    model = MASSPretraining()
    
    B, L, k = 2, 20, 10
    encoder_ids = torch.randint(0, 1000, (B, L))
    decoder_ids = torch.full((B, k), 1)  # [MASK] token id
    
    logits = model(encoder_ids, decoder_ids)
    print(f"MASS输出形状: {logits.shape}")  # [B, k, vocab_size]
    
    print("MASS测试通过！")

if __name__ == "__main__":
    test_mass()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class HandwrittenMASS(nn.Module):
    """MASS核心逻辑手工实现"""
    def __init__(self, vocab_size=30000, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 共享嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # 输出
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        
    def forward(self, src_ids, tgt_ids):
        """MASS前向传播"""
        # 编码器
        src_emb = self.embedding(src_ids) + self.pos_embedding(
            torch.arange(src_ids.size(1), device=src_ids.device).unsqueeze(0)
        )
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory)
        memory = self.encoder_norm(memory)
        
        # 解码器
        tgt_len = tgt_ids.size(1)
        tgt_emb = self.embedding(tgt_ids) + self.pos_embedding(
            torch.arange(tgt_len, device=tgt_ids.device).unsqueeze(0)
        )
        
        causal_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float('-inf'), device=tgt_ids.device), diagonal=1
        )
        
        x = tgt_emb
        for layer in self.decoder_layers:
            x = layer(x, memory, causal_mask)
        x = self.decoder_norm(x)
        
        logits = self.output(x)
        return logits
    
    def mass_loss(self, src_ids, tgt_ids, labels):
        """计算MASS的mask预测损失"""
        logits = self.forward(src_ids, tgt_ids)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
        return loss


def apply_continuous_mask(tokens, mask_id, mask_ratio=0.5):
    """对token序列应用连续mask"""
    L = len(tokens)
    k = int(L * mask_ratio)
    start = random.randint(0, max(0, L - k))
    
    encoder_input = tokens[:start] + [mask_id] * k + tokens[start+k:]
    decoder_input = [mask_id] * k
    labels = tokens[start:start+k]
    
    return encoder_input, decoder_input, labels


def test_handwritten():
    model = HandwrittenMASS(vocab_size=5000, d_model=256, nhead=4)
    B, S, k = 2, 20, 10
    src = torch.randint(0, 5000, (B, S))
    tgt = torch.full((B, k), 1)
    labels = torch.randint(0, 5000, (B, k))
    
    loss = model.mass_loss(src, tgt, labels)
    print(f"MASS损失: {loss.item():.4f}")

if __name__ == "__main__":
    test_handwritten()
```

---

## 9. 可视化与结果理解

MASS的mask策略效果示例：
```
原文:    The quick brown fox jumps over the lazy dog
k=10:    [MASK x10] over the lazy dog  (编码器)
         解码器: [MASK x10] → "The quick brown fox jumps"
```

---

## 10. 模型评估

| 任务 | 指标 | Base | Large |
|------|------|------|-------|
| WMT En-Fr | BLEU | 40.6 | 43.1 |
| WMT En-De | BLEU | 30.1 | 33.0 |

---

## 11. 常见问题

### Q1: MASS和BART的区别？
A: 两者非常相似，核心区别：MASS关注连续mask单一策略和最优k值；BART提供五种噪声策略。MASS在翻译上略优，BART在摘要上略优。

### Q2: 为什么MASS的k≈T/2最优？
A: 平衡了编码器的理解难度和解码器的生成难度，两者都获得充分训练。

---

## 12. 学习总结
MASS通过**连续mask**策略创新性地统一了BERT和GPT的训练范式，证明了Seq2Seq预训练对生成任务（特别是机器翻译）的巨大价值。

---

## 13. 练习题

### 习题1：MASS中k=1和k=T分别对应什么模型？
**答案**：k=1→BERT（MLM），k=T→纯语言模型（GPT）。

### 习题2：MASS为什么比BERT更适合机器翻译？
**答案**：MASS的编码器-解码器结构与翻译任务完全匹配，且连续mask训练了生成能力。

### 习题3：计算MASS的k=1和k=T时的损失函数等价形式。
**答案**：
k=1: $L = -\sum_i \log P(x_i | x^{\backslash i})$
k=T: $L = -\sum_t \log P(x_t | x_{<t}, [MASK]_1^T)$

---

## 14. 学习路径建议

### 前置
- BERT、GPT、Seq2Seq

### 平行
- BART、T5

### 进阶
- mBART、Pegasus
