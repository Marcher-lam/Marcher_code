# UniLM（Unified Language Model）学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
UniLM（Unified Language Model）是微软于2019年提出的统一预训练语言模型，通过精心设计的注意力掩码机制同时支持单向语言模型、双向语言模型和序列到序列语言模型三种训练目标，实现"一个模型，三种能力"。

### 1.2 直觉类比
UniLM就像一把"瑞士军刀"——通过不同的"掩码开关"切换工作模式。当开关拨到"理解"模式，它像BERT一样看全文；拨到"生成"模式，它像GPT一样从左到右生成；拨到"翻译"模式，它像Seq2Seq一样编码→解码。

### 1.3 历史背景
- **2019年10月**：微软提出UniLM v1
- **2020年**：UniLM v2发布（引入预训练去噪）
- **意义**：首个统一理解与生成的预训练框架

### 1.4 算法定位
UniLM是**统一预训练语言模型**，同时支持NLU和NLG任务。

---

## 2. 核心原理

### 2.1 三种训练目标
UniLM通过三种不同的注意力掩码实现：

**1) 单向LM（Left-to-Right LM）**
- 类似GPT
- 每个token只能看到左侧token
- 适用于文本生成

**2) 双向LM（Bidirectional LM）**
- 类似BERT
- 每个token可以看到所有token
- 适用于文本理解

**3) Seq2Seq LM**
- 编码器部分双向，解码器部分单向
- 编码器token可以看到编码器全部token
- 解码器token可以看到编码器全部token和自身左侧token
- 适用于翻译、摘要等Seq2Seq任务

### 2.2 注意力掩码设计
核心创新在于可变的注意力掩码矩阵：

$$M_{ij} = \begin{cases} 
0, & \text{允许位置 $i$ 注意位置 $j$} \\
-\infty, & \text{禁止}
\end{cases}$$

三种任务的掩码矩阵：
- **双向**: $M = \mathbf{0}$ (全0)
- **单向**: $M_{ij} = 0 \text{ if } i \geq j, \text{ else } -\infty$
- **Seq2Seq**: 编码器内部全0，解码器内部单向，解码器可以看编码器

### 2.3 共享参数
三种训练目标共享完全相同的Transformer参数，通过注意力掩码实现任务切换。这使得模型在训练过程中同时获得理解和生成能力。

---

## 3. 数学公式与推导

### 3.1 联合训练目标
三个目标联合优化：

$$L_{UniLM} = L_{bi} + L_{left-to-right} + L_{seq2seq}$$

### 3.2 双向LM损失
$$L_{bi} = -\mathbb{E}_{x\sim D} \sum_{i\in M} \log P(x_i | x_{\backslash M})$$

对随机mask的token进行预测（同BERT的MLM）。

### 3.3 单向LM损失
$$L_{left-to-right} = -\mathbb{E}_{x\sim D} \sum_{t=1}^T \log P(x_t | x_{<t})$$

对序列每个token进行自回归预测。

### 3.4 Seq2Seq LM损失
对平行句对 $(x, y)$：

$$L_{seq2seq} = -\mathbb{E}_{(x,y)} \sum_{t=1}^{|y|} \log P(y_t | y_{<t}, x)$$

解码器以编码器输出和之前生成的token为条件。

### 3.5 注意力计算
三种任务统一的计算公式：

$$h_i = \sum_{j: M_{ij}=0} \frac{\exp(q_i k_j^T / \sqrt{d})}{\sum_{l: M_{il}=0} \exp(q_i k_l^T / \sqrt{d})} v_j$$

通过 $M$ 控制信息流。

---

## 4. 训练过程讲解

### 4.1 预训练步骤
1. 从大规模语料中采样文本
2. 随机选择训练目标（每种目标概率相等）
3. 根据目标创建对应的注意力掩码
4. 前向传播计算损失
5. 反向传播更新模型参数
6. 重复以上步骤直到收敛

### 4.2 任务切换策略
每个batch随机分配任务：
- 1/3 batch做双向LM（理解）
- 1/3 batch做单向LM（生成）
- 1/3 batch做Seq2Seq LM（翻译/摘要）

### 4.3 微调
- **分类任务**：使用双向LM，[CLS]表示+分类头
- **生成任务**：使用Seq2Seq LM，解码器自回归生成
- **抽取式QA**：使用双向LM，预测答案的起始和结束位置

---

## 5. 应用场景

| 场景 | 掩码模式 | 示例 |
|------|----------|------|
| 文本分类 | 双向 | 情感分析、意图识别 |
| 命名实体识别 | 双向 | 识别人名、地名 |
| 抽取式问答 | 双向 | 从文章中找答案 |
| 文本生成 | 单向 | 故事续写 |
| 机器翻译 | Seq2Seq | 英译中 |
| 文本摘要 | Seq2Seq | 长文→摘要 |

---

## 6. 优缺点分析

### 优点
1. **一模型三用**：理解、生成、翻译用一个模型搞定
2. **参数高效**：三种能力共享参数，总参数量小
3. **任务通用**：微调后可适配几乎所有NLP任务

### 缺点
1. **训练复杂**：需要三种任务均衡，避免互相干扰
2. **长文本受限**：Seq2Seq模式受最大位置编码限制
3. **生成速度慢**：Decoder需要自回归生成

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UniLMAttention(nn.Module):
    """UniLM注意力层（支持三种掩码模式）"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask_type='bidirectional', segment_ids=None):
        """
        Args:
            x: [B, L, D]
            mask_type: 'bidirectional', 'left-to-right', 'seq2seq'
            segment_ids: [B, L] 0=编码器, 1=解码器 (for seq2seq)
        """
        B, L, D = x.shape
        
        Q = self.w_q(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 创建注意力掩码
        mask = self._create_mask(L, mask_type, segment_ids).to(x.device)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)  # [B, nhead, L, L]
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.w_o(output)
        
        return output
    
    def _create_mask(self, L, mask_type, segment_ids=None):
        """
        创建注意力掩码
        Returns: [L, L] or [B, L, L]
        """
        if mask_type == 'bidirectional':
            return torch.zeros(L, L)
        
        elif mask_type == 'left-to-right':
            mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
            return mask
        
        elif mask_type == 'seq2seq':
            # segment_ids: 0=编码器部分, 1=解码器部分
            B, _ = segment_ids.shape
            mask = torch.full((B, L, L), float('-inf'))
            
            for b in range(B):
                # 找到编码器和解码器的分界
                encoder_end = (segment_ids[b] == 0).sum().item()
                
                # 编码器部分：全双向
                mask[b, :encoder_end, :encoder_end] = 0
                
                # 解码器部分：看编码器全部 + 自身左侧
                mask[b, encoder_end:, :encoder_end] = 0  # 看编码器
                for i in range(encoder_end, L):
                    mask[b, i, encoder_end:i+1] = 0  # 看自身左侧
                    
            return mask


class UniLMModel(nn.Module):
    """UniLM完整模型"""
    def __init__(self, vocab_size=30522, d_model=768, nhead=12, num_layers=12):
        super().__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # 编码器/解码器
        
        # Transformer层
        self.layers = nn.ModuleList([
            UniLMTransformerLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # MLM头
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
        # LM头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.word_embedding.weight
        
    def forward(self, input_ids, mask_type='bidirectional', segment_ids=None):
        """
        Args:
            input_ids: [B, L]
            mask_type: 训练模式
            segment_ids: [B, L] 用于seq2seq模式
        """
        B, L = input_ids.shape
        
        # 嵌入
        word_emb = self.word_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(L, device=input_ids.device).unsqueeze(0))
        
        if segment_ids is not None:
            seg_emb = self.segment_embedding(segment_ids)
        else:
            seg_emb = self.segment_embedding(torch.zeros(B, L, dtype=torch.long, device=input_ids.device))
        
        x = word_emb + pos_emb + seg_emb
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x, mask_type=mask_type, segment_ids=segment_ids)
        
        x = self.norm(x)
        
        if mask_type == 'bidirectional':
            # MLM预测
            logits = self.mlm_head(x)
        else:
            # LM预测
            logits = self.lm_head(x)
            
        return logits
    
    def generate(self, input_ids, max_length=50, segment_ids=None):
        """自回归生成"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(input_ids, mask_type='left-to-right', segment_ids=segment_ids)
                next_logits = logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


class UniLMTransformerLayer(nn.Module):
    """UniLM Transformer层"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = UniLMAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask_type='bidirectional', segment_ids=None):
        attn_out = self.attention(x, mask_type, segment_ids)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


def test_unilm():
    """测试UniLM三种模式"""
    model = UniLMModel(vocab_size=10000, d_model=256, nhead=4, num_layers=4)
    B, L = 2, 16
    
    input_ids = torch.randint(0, 10000, (B, L))
    
    # 双向模式
    out_bi = model(input_ids, 'bidirectional')
    print(f"双向模式输出: {out_bi.shape}")
    
    # 单向模式
    out_lr = model(input_ids, 'left-to-right')
    print(f"单向模式输出: {out_lr.shape}")
    
    # Seq2Seq模式
    seg_ids = torch.cat([
        torch.zeros(B, 8, dtype=torch.long),
        torch.ones(B, 8, dtype=torch.long)
    ], dim=1)
    out_s2s = model(input_ids, 'seq2seq', seg_ids)
    print(f"Seq2Seq模式输出: {out_s2s.shape}")
    
    print("UniLM测试通过！")

if __name__ == "__main__":
    test_unilm()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandwrittenUniLM(nn.Module):
    """UniLM核心逻辑手工实现"""
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            HandwrittenUniLMLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask):
        # mask: [B, L, L]
        for layer in self.layers:
            x = layer(x, mask)
        return self.output(self.norm(x))
    
    @staticmethod
    def make_bidirectional_mask(L):
        """双向掩码"""
        return torch.zeros(L, L)
    
    @staticmethod
    def make_causal_mask(L):
        """单向因果掩码"""
        mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
        return mask
    
    @staticmethod
    def make_seq2seq_mask(L1, L2):
        """Seq2Seq掩码：编码器L1，解码器L2"""
        L = L1 + L2
        mask = torch.full((L, L), float('-inf'))
        mask[:L1, :L1] = 0
        mask[L1:, :L1] = 0
        for i in range(L2):
            mask[L1+i, L1:L1+i+1] = 0
        return mask


def test_handwritten():
    model = HandwrittenUniLM()
    B, L = 2, 10
    x = torch.randint(0, 1000, (B, L))
    
    # 测试三种掩码模式
    for name, mask_fn in [
        ('bidirectional', model.make_bidirectional_mask),
        ('causal', model.make_causal_mask),
        ('seq2seq', lambda: model.make_seq2seq_mask(5, 5)),
    ]:
        mask = mask_fn(L).unsqueeze(0).expand(B, -1, -1)
        out = model(x, mask)
        print(f"{name} 输出: {out.shape}")

if __name__ == "__main__":
    test_handwritten()
```

---

## 9. 可视化与结果理解

三种掩码模式的注意力模式可视化：
- **双向**：注意力矩阵全亮（全连接）
- **单向**：对角线及下方亮，上方全黑（因果）
- **Seq2Seq**：编码器部分全亮，解码器部分左下亮右上暗

---

## 10. 模型评估

| 任务 | 指标 | BERT-Base | UniLM-Base |
|------|------|-----------|------------|
| SQuAD 1.1 | F1 | 88.5 | 90.1 |
| CoLA | MCC | 60.2 | 62.3 |
| CNN/DM摘要 | ROUGE-L | 36.5 (BART) | 38.2 (UniLM) |

---

## 11. 常见问题

### Q1: UniLM的三种目标如何平衡？
A: 通过均匀采样，每个batch 1/3做每种目标。实际中发现三种目标互相促进。

### Q2: UniLM和T5的区别？
A: UniLM是单一Transformer通过掩码切换任务；T5使用编码器-解码器架构，更重但性能更好。

### Q3: 为什么UniLM的生成任务需要Seq2Seq掩码而不是纯单向？
A: Seq2Seq掩码允许解码器看编码器全部信息，这为生成提供了更丰富的上下文。

---

## 12. 学习总结

UniLM的核心创新是通过**注意力掩码**统一了三种语言模型范式（双向/单向/Seq2Seq），证明了"共享参数+任务条件"的有效性，为后续Unified-IO、OFA等统一多任务模型奠定了基础。

---

## 13. 练习题

### 习题1：画出三种掩码模式的注意力矩阵。
**答案**：
- 双向：5×5全1矩阵
- 单向：下三角矩阵
- Seq2Seq：2+3编码，矩阵左上2×2全1，左下3×2全1，右下3×3下三角

### 习题2：UniLM如何同时处理理解和生成任务？
**答案**：通过注意力掩码切换。理解用双向，生成用单向或Seq2Seq，参数共享。

### 习题3：编写代码生成Seq2Seq注意力掩码。
**答案**：如上面 `make_seq2seq_mask` 函数。

### 习题4：思考：UniLM三种目标是否可能互相干扰？
**答案**：有可能。双向模式鼓励"偷看"，单向模式禁止"偷看"。但实验表明共享利大于弊，因为模型学会了根据掩码调整行为。

---

## 14. 学习路径建议

### 前置
- BERT、GPT、Transformer

### 平行
- T5、BART、MASS

### 进阶
- UniLM v2、GLM、UL2
