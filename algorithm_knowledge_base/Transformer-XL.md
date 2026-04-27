# Transformer-XL 学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
Transformer-XL（Extra Long）是Google Brain于2019年提出的长序列语言模型，通过**片段递归（Segment-Level Recurrence）**和**相对位置编码（Relative Positional Encoding）**，使Transformer能够处理超过固定长度限制的超长序列。

### 1.2 直觉类比
Transformer-XL就像给Transformer装了一个"工作记忆"——处理当前段落时，它能记住上一段落的"思路"（片段递归），并且能准确知道每个词在历史中的相对位置（相对位置编码）。这就像你读一本长篇小说时，虽然书被分成一页一页（片段），但你的大脑能记住前一页的内容，理解故事的连续性。

### 1.3 历史背景
- **2019年1月**：Google Brain团队提出Transformer-XL
- **2019年6月**：XLNet将其作为基础组件
- **2020年**：Compressive Transformer在此基础上扩展

### 1.4 算法定位
Transformer-XL是**长序列自回归语言模型**，解决了标准Transformer固定长度上下文的限制。

---

## 2. 核心原理

### 2.1 标准Transformer的局限
标准Transformer的固定长度编码导致两个问题：
1. **上下文碎片化**：序列被截断为独立片段，片段间无信息流通
2. **位置编码受限**：超出最大长度$L_{\max}$的token无法编码

### 2.2 片段递归（Segment-Level Recurrence）
处理当前片段时，保留并复用前一个片段的隐藏状态：

$$\tilde{h}^{(t-1)} = [\text{SG}(h^{(t-1)}_{\tau}), h^{(t-1)}_{\tau+1}]$$

$$h^{(t)}_{\tau+1} = f(\tilde{h}^{(t-1)}, h^{(t)}_{\tau+1})$$

其中 $\text{SG}(\cdot)$ 是stop-gradient（梯度不反向传播到前一片段）。

**效果**：理论上可以建模最长 $N \times L$ 的依赖，$N$ 是层数，$L$ 是片段长度。

### 2.3 相对位置编码
标准Transformer使用绝对位置编码，在片段递归时无法区分同一位置在不同片段中的语义。Transformer-XL使用相对位置编码：

$$A^{\text{rel}}_{i,j} = \underbrace{q_i^T k_j}_{(a)} + \underbrace{q_i^T W_{k,R} R_{i-j}}_{(b)} + \underbrace{u^T k_j}_{(c)} + \underbrace{v^T W_{k,R} R_{i-j}}_{(d)}$$

其中：
- (a) 标准内容-内容注意力
- (b) 查询 $q_i$ 与相对位置 $R_{i-j}$ 的注意力（基于内容的偏置）
- (c) 全局内容偏置 $u^T k_j$
- (d) 全局位置偏置 $v^T W_{k,R} R_{i-j}$

---

## 3. 数学公式与推导

### 3.1 片段递归的数学表达
第 $\tau$ 个片段的第 $n$ 层的隐藏状态：

$$h_{\tau+1}^{(n)} = \text{TransformerLayer}(h_{\tau+1}^{(n-1)}, \tilde{h}_{\tau}^{(n-1)})$$

其中 $\tilde{h}_{\tau}^{(n-1)} = \text{concat}[\text{SG}(h_{\tau}^{(n-1)}), h_{\tau+1}^{(n-1)}]$

梯度流动：
- $h_{\tau+1}^{(n)}$ 的梯度可以流向 $h_{\tau+1}^{(n-1)}$（当前片段）
- $h_{\tau+1}^{(n)}$ 的梯度**不**流向 $h_{\tau}^{(n-1)}$（前一片段被SG阻断）

### 3.2 相对位置编码详解
标准绝对位置编码的自注意力：

$$A_{i,j}^{\text{abs}} = (E_{x_i} + U_i)^T W_q^T W_k (E_{x_j} + U_j)$$

展开为四项：
$$= \underbrace{E_{x_i}^T W_q^T W_k E_{x_j}}_{(a)} + \underbrace{E_{x_i}^T W_q^T W_k U_j}_{(b)} + \underbrace{U_i^T W_q^T W_k E_{x_j}}_{(c)} + \underbrace{U_i^T W_q^T W_k U_j}_{(d)}$$

相对位置编码改造：
$$A_{i,j}^{\text{rel}} = \underbrace{E_{x_i}^T W_q^T W_{k,E} E_{x_j}}_{(a')} + \underbrace{E_{x_i}^T W_q^T W_{k,R} R_{i-j}}_{(b')} + \underbrace{u^T W_{k,E} E_{x_j}}_{(c')} + \underbrace{v^T W_{k,R} R_{i-j}}_{(d')}$$

主要改变：
- 用 $R_{i-j}$ 替代 $U_j$（相对位置替代绝对位置）
- 用 $u$ 替代 $U_i^T W_q^T$（统一查询位置偏置）
- 键分为内容键 $W_{k,E}$ 和位置键 $W_{k,R}$

### 3.3 计算效率优化
朴素实现 $O(L^2)$ 对相对位置编码来说计算量很大。Transformer-XL使用批矩阵乘法优化：

$$A = q_{\text{content}}^T k_{\text{content}}^T + \text{shift}(q_{\text{content}}^T R^T) + u^T k_{\text{content}}^T + \text{shift}(v^T R^T)$$

其中 $\text{shift}$ 操作将排序交换为相对位置。

---

## 4. 训练过程讲解

### 4.1 训练流程
1. 将长文档分割为重叠的片段（每个片段长度 $L$）
2. 按顺序处理片段
3. 处理第 $\tau+1$ 个片段时，复用第 $\tau$ 个片段的隐藏状态
4. 计算语言模型损失（预测下一个token）
5. 梯度反向传播只更新当前片段参数
6. 保存当前片段隐藏状态供下一个片段使用

### 4.2 与标准Transformer对比
| 步骤 | 标准Transformer | Transformer-XL |
|------|----------------|----------------|
| 片段1 | 处理片段1, 丢弃状态 | 处理片段1, 保存状态 |
| 片段2 | 重新初始化, 处理片段2 | 复用片段1状态, 处理片段2 |
| 片段3 | 重新初始化, 处理片段3 | 复用片段2状态, 处理片段3 |

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 长文档语言建模 | 书籍、论文、法律文档 |
| 音乐生成 | 长程旋律结构建模 |
| 代码生成 | 长函数/文件上下文 |
| 文档级情感分析 | 整本书的情感轨迹 |

---

## 6. 优缺点分析

### 优点
1. **超长上下文**：可建模数千到数万token的依赖
2. **评估加速**：推理时缓存历史状态，比重新计算快1800倍
3. **困惑度降低**：WikiText-103 PPL从标准Transformer的35.0降至24.0
4. **平滑过渡**：片段间信息连续，无截断损失

### 缺点
1. **内存开销**：需要缓存历史隐藏状态
2. **实现复杂**：相对位置编码的实现较复杂
3. **推理延迟**：缓存管理增加推理延迟

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TransfoXLModel, TransfoXLTokenizer, TransfoXLLMHeadModel
import math

class TransformerXLLM(nn.Module):
    """Transformer-XL语言模型"""
    def __init__(self, model_name='transfo-xl-wt103'):
        super().__init__()
        self.model = TransfoXLLMHeadModel.from_pretrained(model_name)
        self.tokenizer = TransfoXLTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        return outputs.loss, outputs.logits
    
    def generate(self, prompt, max_length=100, temperature=0.8):
        """文本生成"""
        self.model.eval()
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # 初始化记忆
        mems = None
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(
                    input_ids=inputs[:, -1:],
                    mems=mems,
                    labels=None
                )
                logits = outputs.logits[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)
                inputs = torch.cat([inputs, next_token], dim=1)
                mems = outputs.mems
        
        return self.tokenizer.decode(inputs[0], skip_special_tokens=True)


class SegmentRecurrentLayer(nn.Module):
    """Transformer-XL片段递归层手工实现"""
    def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 注意力
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # 相对位置编码参数
        self.u = nn.Parameter(torch.randn(1, self.nhead, self.d_k))
        self.v = nn.Parameter(torch.randn(1, self.nhead, self.d_k))
        self.pos_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def _rel_shift(self, x, zero_upper=True):
        """相对位置编码的特殊移位操作"""
        x_shape = x.shape
        x = x.view(-1, x_shape[-2], x_shape[-1])
        x_padded = F.pad(x, (0, 0, 1, 0))  # 上方补一行0
        x = x_padded.view(x_shape[0], x_shape[1] + 1, x_shape[-2], x_shape[-1])
        x = x[:, 1:, :, :].view_as(x_shape)
        
        if zero_upper:
            mask = torch.triu(torch.ones(x_shape[-2], x_shape[-1], device=x.device), diagonal=1)
            x = x * (1 - mask.unsqueeze(0).unsqueeze(0))
            
        return x
    
    def forward(self, x, pos_encoding, mem=None, mask=None):
        """
        Args:
            x: [B, L, D] 当前片段
            pos_encoding: [L+mem_len, D] 位置编码
            mem: [B, mem_len, D] 前一片段的隐藏状态
        """
        B, L, D = x.shape
        
        # 拼接记忆
        if mem is not None:
            kv = torch.cat([mem, x], dim=1)
        else:
            kv = x
        
        kv_len = kv.shape[1]
        
        # 投影到多头
        q = self.q_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(kv).view(B, kv_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(kv).view(B, kv_len, self.nhead, self.d_k).transpose(1, 2)
        
        # 相对位置编码
        pos_enc = self.pos_proj(pos_encoding).view(-1, self.nhead, self.d_k).transpose(0, 1)
        
        # 注意力分数计算
        # 1) 内容-内容: q @ k^T
        content_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 2) 内容-位置: q @ pos_enc^T + u @ k^T + v @ pos_enc^T
        pos_scores = torch.matmul(q, pos_enc.transpose(-2, -1).unsqueeze(0))
        pos_scores = self._rel_shift(pos_scores)
        
        # 3) 全局内容偏置 u
        u_scores = torch.matmul(self.u, k.transpose(-2, -1))  # [1, nhead, 1, kv_len]
        
        # 4) 全局位置偏置 v
        v_scores = torch.matmul(self.v, pos_enc.transpose(-2, -1).unsqueeze(0))
        v_scores = self._rel_shift(v_scores)
        
        # 总分数
        scores = content_scores + pos_scores + u_scores + v_scores
        
        # 因果掩码
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        
        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        
        # 残差+归一化
        x = self.norm1(x + self.dropout(out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


def test_transformer_xl():
    """测试Transformer-XL"""
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    
    text = "The history of natural language processing"
    inputs = tokenizer.encode(text, return_tensors='pt')
    
    outputs = model(input_ids=inputs, labels=inputs)
    ppl = torch.exp(outputs.loss).item()
    print(f"困惑度: {ppl:.4f}")
    
    print("Transformer-XL测试通过！")

if __name__ == "__main__":
    test_transformer_xl()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandwrittenTransformerXL(nn.Module):
    """Transformer-XL核心逻辑简化实现"""
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimplifiedXLLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, memory=None):
        """
        x: [B, L]
        memory: [num_layers, B, mem_len, D] 或 None
        """
        x = self.embedding(x)
        B, L, D = x.shape
        
        new_memory = []
        
        for i, layer in enumerate(self.layers):
            mem = memory[i] if memory is not None else None
            x, layer_mem = layer(x, mem)
            new_memory.append(x[:, -L:, :])
        
        x = self.norm(x)
        logits = self.output(x)
        
        return logits, torch.stack(new_memory)


def test_handwritten():
    model = HandwrittenTransformerXL(vocab_size=5000, d_model=256, nhead=4, num_layers=3)
    B, L = 2, 20
    ids = torch.randint(0, 5000, (B, L))
    logits, _ = model(ids)
    print(f"手工Transformer-XL输出: {logits.shape}")

if __name__ == "__main__":
    test_handwritten()
```

---

## 9. 可视化与结果理解

### 9.1 递归连接的可视化
Transformer-XL在层与层之间建立递归连接：第 $n$ 层的输出成为下一片段第 $n$ 层的额外输入。

### 9.2 困惑度对比
| 序列长度 | 标准Transformer | Transformer-XL |
|----------|----------------|----------------|
| 512 | 35.0 | 24.0 |
| 1024 | 无法处理（截断） | 22.1 |
| 3072 | 无法处理 | 19.8 |

---

## 10. 模型评估

| 数据集 | 指标 | LSTM | Transformer | Transformer-XL |
|--------|------|------|-------------|----------------|
| WikiText-103 | PPL | 40.8 | 35.0 | 24.0 |
| enwik8 | bpc | 1.46 | 1.20 | 1.06 |
| text8 | bpc | 1.45 | 1.18 | 1.03 |

---

## 11. 常见问题

### Q1: 片段递归中的stop-gradient为什么重要？
**答案**：防止梯度通过长距离递归路径爆炸/消失。如果梯度能流向所有历史片段，训练会变得非常不稳定。

### Q2: 相对位置编码如何处理序列长度超出训练长度？
**答案**：相对位置编码基于距离 $i-j$，不受绝对位置限制。虽然训练时最大距离受限于片段长度+记忆长度，但推理时可以泛化到更长的距离。

### Q3: Transformer-XL和Longformer的区别？
**答案**：Transformer-XL使用递归处理长序列；Longformer使用稀疏注意力（滑动窗口+全局token）。前者更精确但更慢，后者更快但可能丢失全局信息。

---

## 12. 学习总结

Transformer-XL通过**片段递归**+**相对位置编码**，将Transformer的上下文窗口从固定512扩展到数千token，为长序列建模提供了有效的解决方案。

---

## 13. 练习题

### 习题1：Transformer-XL如何解决上下文碎片化？
**答案**：通过片段递归复用前一片段的隐藏状态，使信息可以在片段间流动。

### 习题2：相对位置编码与绝对位置编码的核心区别？
**答案**：绝对位置编码给每个位置固定向量，在片段递归时无法区分两个片段中相同位置的语义差异；相对位置编码基于token间距离建模，天然适应递归。

### 习题3：假设片段长度L=128，记忆长度M=128，层数N=6，最大可建模的依赖距离是多少？
**答案**：理论上 $N \times (L+M) = 6 \times 256 = 1536$。但实际上因为梯度衰减，有效距离小于此值。

### 习题4：推理时Transformer-XL为什么比标准Transformer快？
**答案**：标准Transformer每次生成一个token需要重新计算所有token的隐藏状态（$O(L^2)$）；Transformer-XL缓存历史状态，每次只需计算新token（$O(L)$）。

---

## 14. 学习路径建议

### 前置
- Transformer、自注意力机制

### 平行
- XLNet、Compressive Transformer

### 进阶
- Longformer、Big Bird、Sparse Transformer
