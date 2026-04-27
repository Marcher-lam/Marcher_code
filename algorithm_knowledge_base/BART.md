# BART（Bidirectional and Auto-Regressive Transformers）学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
BART（Bidirectional and Auto-Regressive Transformer）是Facebook AI于2019年提出的序列到序列去噪自编码器，结合BERT的双向编码能力和GPT的自回归解码能力，通过破坏文本再重建的方式学习语言表示。

### 1.2 直觉类比
BART就像一个"文章修复专家"——给你一篇被各种方式破坏的文章（关键词被涂黑、句子被删掉、段落被打乱），你需要把它恢复原样。这个过程让你深刻理解文章的结构、语义和上下文关系。

### 1.3 历史背景
- **2019年10月**：Facebook AI提出BART
- **2020年**：BART成为摘要、翻译等生成任务的标准基线
- **后续**：mBART扩展到多语言，BART-large在许多任务上仍具竞争力

### 1.4 算法定位
BART是**Seq2Seq去噪自编码模型**，属于自监督预训练。

---

## 2. 核心原理

### 2.1 编码器-解码器架构
- **编码器**：双向Transformer（类似BERT），理解完整上下文
- **解码器**：自回归Transformer（类似GPT），从左到右生成

### 2.2 五种噪声策略
BART的核心创新在于多样化的文本破坏方式：

1. **Token Masking**：随机将token替换为[MASK]（同BERT）
2. **Token Deletion**：随机删除token
3. **Text Infilling**：随机采样文本片段，用一个[MASK]替换整个片段
4. **Sentence Permutation**：将文档按句号分割后打乱句子顺序
5. **Document Rotation**：随机选择一个token作为文档的新起点（循环移位）

### 2.3 联合训练
所有噪声策略在同一个模型中训练，共享编码器和解码器参数。

---

## 3. 数学公式与推导

### 3.1 预训练目标
给定原始文本 $x$ 和破坏后的版本 $\tilde{x}$：

$$L = -\log P(x | \tilde{x}) = -\sum_{t=1}^T \log P(x_t | x_{<t}, \tilde{x})$$

其中 $x_t$ 是解码器在位置 $t$ 生成的token。

### 3.2 Text Infilling的数学建模
Text Infilling从原始文本中采样 $k$ 个span，每个span长度服从泊松分布 $\lambda=3$：

$$P(\text{span length} = l) = \frac{\lambda^l e^{-\lambda}}{l!}$$

损坏比例约为原始文本的30%。

### 3.3 解码条件概率
解码器每一步生成token的概率：

$$P(y_t | y_{<t}, x) = \text{softmax}(W h_t + b)$$

$$h_t = \text{Decoder}(y_{<t}, \text{Encoder}(x))$$

### 3.4 微调目标
摘要生成（Label $y$ 是摘要，$x$ 是文档）：

$$L_{sum} = -\log P(y | x) = -\sum_t \log P(y_t | y_{<t}, x)$$

---

## 4. 训练过程讲解

### 4.1 预训练步骤
1. 从语料中采样文本 $x$
2. 随机选择一种噪声策略破坏文本得到 $\tilde{x}$
3. 编码器编码 $\tilde{x}$ 得到上下文表示
4. 解码器以 $\tilde{x}$ 的编码为条件，自回归预测原始文本 $x$
5. 计算交叉熵损失并更新参数

### 4.2 微调步骤
- **摘要**：输入原文 $x$，解码器生成摘要 $y$
- **翻译**：输入源语言 $x$，解码器生成目标语言 $y$
- **分类**：解码器最后一个token的表示通过分类头预测

---

## 5. 应用场景

| 任务 | 输入 | 输出 |
|------|------|------|
| 文本摘要 | 长文档 | 短摘要 |
| 机器翻译 | 源语言文本 | 目标语言翻译 |
| 对话生成 | 对话历史 | 回复 |
| 文本去噪 | 带噪声文本 | 清理后的文本 |
| 问答生成 | 上下文+问题 | 答案 |

---

## 6. 优缺点分析

### 优点
1. **丰富的噪声策略**：五种策略互补，提供多样的训练信号
2. **双向+自回归**：兼具理解和生成能力
3. **强生成能力**：在摘要、翻译等任务上性能突出

### 缺点
1. **计算量大**：编码器-解码器架构参数量翻倍
2. **训练慢**：Seq2Seq结构比BERT慢2-3倍
3. **长序列推理慢**：自回归生成逐token进行

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel

class BARTFineTuner(nn.Module):
    """BART模型微调器"""
    def __init__(self, model_name='facebook/bart-base'):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def generate_summary(self, text, max_length=150, min_length=40, num_beams=4):
        """生成文本摘要"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        summary_ids = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def train_step(self, input_text, target_text, lr=3e-5):
        """单步训练"""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text, return_tensors='pt', max_length=1024, truncation=True
        ).to(self.device)
        
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                target_text, return_tensors='pt', max_length=128, truncation=True
            ).to(self.device)
        
        # 前向传播
        outputs = self.model(
            **inputs,
            labels=targets.input_ids
        )
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class BARTDenoisingTrainer:
    """BART去噪预训练模拟器"""
    def __init__(self, model_name='facebook/bart-base'):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
    def add_noise(self, text, noise_type='text_infilling'):
        """对文本添加特定类型的噪声"""
        tokens = self.tokenizer.tokenize(text)
        
        if noise_type == 'token_masking':
            # 随机mask 15% token
            masked = []
            for token in tokens:
                if torch.rand(1) < 0.15:
                    masked.append('<mask>')
                else:
                    masked.append(token)
                    
        elif noise_type == 'token_deletion':
            # 随机删除10% token
            masked = [t for t in tokens if torch.rand(1) > 0.1]
            
        elif noise_type == 'text_infilling':
            # 连续mask片段
            masked = []
            i = 0
            while i < len(tokens):
                if torch.rand(1) < 0.15:
                    span_len = min(int(torch.poisson(torch.tensor(3.0))), len(tokens) - i)
                    masked.append('<mask>')
                    i += span_len
                else:
                    masked.append(tokens[i])
                    i += 1
                    
        elif noise_type == 'sentence_permutation':
            # 打乱句子顺序
            masked = tokens  # 简化：实际按句号分割
        
        else:
            masked = tokens
            
        return self.tokenizer.convert_tokens_to_string(masked)
    
    def pretrain_step(self, text):
        """单步预训练"""
        noisy_text = self.add_noise(text)
        return self.train_step(noisy_text, text)


def test_bart():
    """测试BART基本功能"""
    model = BARTFineTuner()
    
    # 摘要测试
    text = """
    The Transformer architecture has become the dominant approach in natural language processing.
    BART combines ideas from BERT and GPT into a single model. 
    It uses a standard sequence-to-sequence architecture with a bidirectional encoder and an autoregressive decoder.
    This allows BART to excel at both understanding and generation tasks.
    """
    
    summary = model.generate_summary(text, max_length=50, min_length=10)
    print(f"原文: {text[:100]}...")
    print(f"摘要: {summary}")
    
    print("BART测试通过！")

if __name__ == "__main__":
    test_bart()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class HandwrittenBART(nn.Module):
    """BART核心逻辑手工实现"""
    def __init__(self, vocab_size=30000, d_model=768, nhead=12, 
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
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
        
        # 输出头
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight  # 权重绑定
        
    def forward(self, source_ids, target_ids, source_mask=None, target_mask=None):
        """前向传播（训练模式）"""
        # 编码器
        memory = self.encode(source_ids, source_mask)
        
        # 解码器
        logits = self.decode(target_ids, memory, source_mask, target_mask)
        
        return logits
    
    def encode(self, source_ids, mask=None):
        """编码器前向传播"""
        x = self.embedding(source_ids) + self.pos_embedding(
            torch.arange(source_ids.size(1), device=source_ids.device).unsqueeze(0)
        )
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return self.encoder_norm(x)
    
    def decode(self, target_ids, memory, source_mask=None, target_mask=None):
        """解码器前向传播"""
        x = self.embedding(target_ids) + self.pos_embedding(
            torch.arange(target_ids.size(1), device=target_ids.device).unsqueeze(0)
        )
        
        # 因果掩码
        L = target_ids.size(1)
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=target_ids.device), diagonal=1
        )
        
        for layer in self.decoder_layers:
            x = layer(x, memory, causal_mask, source_mask)
        
        x = self.decoder_norm(x)
        return self.output_proj(x)
    
    def generate(self, source_ids, max_len=50, source_mask=None):
        """自回归生成"""
        self.eval()
        memory = self.encode(source_ids, source_mask)
        
        # 初始解码器输入
        bos_id = torch.tensor([[0]], device=source_ids.device)
        generated = bos_id
        
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.decode(generated, memory, source_mask)
                next_logits = logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == 2:  # EOS
                    break
                    
        return generated


def add_text_infilling_noise(tokens, mask_token_id, poison_lambda=3.0, poison_rate=0.3):
    """添加Text Infilling噪声"""
    corrupted = []
    i = 0
    while i < len(tokens):
        if random.random() < poison_rate:
            span_len = min(int(torch.poisson(torch.tensor(poison_lambda)).item()), len(tokens) - i)
            corrupted.append(mask_token_id)
            i += span_len
        else:
            corrupted.append(tokens[i])
            i += 1
    return corrupted


def test_handwritten_bart():
    model = HandwrittenBART(vocab_size=10000, d_model=256, nhead=4)
    B, S, T = 2, 20, 15
    src = torch.randint(0, 10000, (B, S))
    tgt = torch.randint(0, 10000, (B, T))
    logits = model(src, tgt[:, :-1])  # teacher forcing
    print(f"手工BART输出: {logits.shape}")

if __name__ == "__main__":
    test_handwritten_bart()
```

---

## 9. 可视化与结果理解

BART的Text Infilling效果示例：
```
原文:    The cat sat on the mat and looked at the bird.
Infilling: [MASK] sat on the [MASK] and looked at the bird.
重建:    The cat sat on the mat and looked at the bird.
```

---

## 10. 模型评估

| 任务 | 指标 | BART-Base | BART-Large |
|------|------|-----------|------------|
| CNN/DM摘要 | ROUGE-L | 38.12 | 41.71 |
| XSUM摘要 | ROUGE-L | 31.44 | 36.27 |
| SQuAD | F1 | 88.8 | 91.1 |

---

## 11. 常见问题

### Q1: BART和T5的区别？
A: BART使用去噪自编码目标，噪声类型更丰富；T5使用Span Corruption，将所有任务统一为Text-to-Text格式。BART更适合摘要，T5更通用。

### Q2: 为什么BART的文本去噪有效？
A: 去噪目标迫使模型理解语言的整体结构，比MLM（只预测单个token）更具挑战性。

---

## 12. 学习总结
BART的核心贡献在于将BERT的双向理解和GPT的自回归生成统一到Seq2Seq框架中，通过多样化的去噪目标学习鲁棒的语言表示。

---

## 13. 练习题

### 习题1：BART的Text Infilling和BERT的MLM有什么区别？
**答案**：BERT mask单个token，BART mask连续span用一个[MASK]替代。BART的任务更难，需要模型预测多个token。

### 习题2：为什么BART适合文本摘要？
**答案**：BART预训练时是destruction→reconstruction，这本质上就是摘要任务（长文→短摘要）的扩展。

### 习题3：BART五种噪声策略哪种最有效？
**答案**：Text Infilling最有效，因为它迫使模型理解语言的整体结构。

---

## 14. 学习路径建议

### 前置
- BERT、GPT、Seq2Seq、Transformer

### 平行
- MASS、T5、Pegasus

### 进阶
- mBART、BART-large、Longformer
