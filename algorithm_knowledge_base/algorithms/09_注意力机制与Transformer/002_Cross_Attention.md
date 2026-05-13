# Cross Attention 交叉注意力 学习文档

> 跨序列交互的注意力机制，Transformer架构的核心组件。

---

## 1. 算法基础认知

**Cross Attention（交叉注意力）** 是Transformer架构中的一种关键注意力机制，它允许一个序列（Query）关注另一个完全不同的序列（Key和Value）。与自注意力（Self-Attention）不同，交叉注意力实现了编码器和解码器之间的信息传递，是序列到序列（Seq2Seq）任务的核心组件。

### 1.1 为什么需要Cross Attention？

在Seq2Seq任务中（如机器翻译），编码器处理源序列生成表示，解码器需要从这个表示中提取 信息来生成目标序列。Cross Attention正是这两者之间的"桥梁"：

- **编码器输出（Source）**：包含源序列的语义信息
- **解码器状态（Query）**：当前需要关注什么
- **Attention机制**：让解码器"看到"源序列的相关部分

### 1.2 Cross Attention vs Self-Attention

| 特性 | Self-Attention | Cross Attention |
|------|---------------|------------------|
| Query来源 | 同一序列 | 目标序列（解码器） |
| Key/Value来源 | 同一序列 | 源序列（编码器） |
| 应用场景 | 特征提取 | 跨序列信息传递 |
| 示例 | Transformer编码器层 | Transformer解码器层 |

---

## 2. 核心原理

### 2.1 注意力机制基础

Cross Attention本质是点积注意力（Multi-Head Attention的一个变体）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n_q \times d}$：Query矩阵（来自解码器）
- $K \in \mathbb{R}^{n_k \times d}$：Key矩阵（来自编码器）
- $V \in \mathbb{R}^{n_k \times d}$：Value矩阵（来自编码器）
- $d_k$：Key的维度，用于缩放

### 2.2 多头交叉注意力

通常使用多头（Multi-Head）版本增强表示能力：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 2.3 在Transformer中的位置

在标准Transformer解码器中，Cross Attention位于：

```
Decoder Layer:
├── Self-Attention (Masked)
├── Add & Norm
├── Cross Attention  ← 编码器信息注入点
├── Add & Norm  
├── Feed Forward
└── Add & Norm
```

---

## 3. 数学公式与推导

### 3.1 Q、K、V的来源

在Transformer解码器中：

$$Q = \text{Linear}_{query}(x) + \text{Pe}(pos)$$
$$K = \text{Linear}_{key}(encoder\_output)$$
$$V = \text{Linear}_{value}(encoder\_output)$$

其中：
- $x$：解码器当前输入
- $encoder\_output$：编码器的最终输出
- $pos$：位置编码

### 3.2 注意力计算

**步骤1：计算相似度分数**
$$S = QK^T \in \mathbb{R}^{n_q \times n_k}$$

**步骤2：缩放**
$$S_{scaled} = S / \sqrt{d_k}$$

**步骤3：Softmax**
$$A = \text{softmax}(S_{scaled}, \text{axis}=-1) \in \mathbb{R}^{n_q \times n_k}$$

**步骤4：加权求和**
$$\text{Output} = AV \in \mathbb{R}^{n_q \times d_v}$$

### 3.3 计算复杂度

- **时间复杂度**：$O(n_q \cdot n_k \cdot d)$
- **空间复杂度**：$O(n_q \cdot n_k)$

当$n_q = n_k = n$时，简化为$O(n^2 \cdot d)$

### 3.4 梯度流动

Cross Attention的梯度计算：
$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial O} \cdot (AK^T) / \sqrt{d_k}$$
$$\frac{\partial \mathcal{L}}{\partial K} = \frac{\partial \mathcal{L}}{\partial O}^T \cdot A^T \cdot Q / \sqrt{d_k}$$
$$\frac{\partial \mathcal{L}}{\partial V} = A^T \cdot \frac{\partial \mathcal{L}}{\partial O}$$

---

## 4. 训练过程讲解

### 4.1 训练时的特殊处理

在训练时，解码器可以看到完整的target序列（使用masked attention防止信息泄露），但Cross Attention仍然能看到完整的encoder输出。

### 4.2 Mask机制

在解码器中，Cross Attention需要mask来遮挡padding位置：

```python
# 创建padding mask
decoder_mask = (encoder_output != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, src_len]
```

### 4.3 训练流程

```
For each batch:
    1. encoder_output = encoder(input)
    2. decoder_input = shift_right(target)  # 右移
    3. self_attn_output = masked_self_attn(decoder_input)
    4. cross_attn_output = cross_attn(self_attn_output, encoder_output)
    5. output = feed_forward(cross_attn_output)
    6. loss = cross_entropy(output, target)
    7. backward()
```

### 4.4 参数更新

Cross Attention的可学习参数：
- 三个线性变换矩阵：$W_Q^o, W_K^o, W_V^o$（每个attention head）
- 输出映射矩阵：$W^O$
- 以及LayerNorm的参数

---

## 5. 应用场景

### 5.1 机器翻译

Cross Attention是神经机器翻译（NMT）的核心组件，负责将源语言信息传递到目标语言生成过程。

```python
# 示例：英语到中文翻译
# Input: "The cat sits on the mat"
# Encoder输出包含每个词的中文语义向量
# Cross Attention帮助解码器选择相关的中文语义
# Output: "猫坐在垫子上"
```

### 5.2 文本摘要

编码器处理原文，解码器通过Cross Attention选择关键信息生成摘要。

### 5.3 语音识别

编码器处理音频特征，Cross Attention帮助生成对应的文本转录。

### 5.4 视觉问答

编码器处理图像和问题，Cross Attention帮助理解问题并定位图像相关区域。

### 5.5 图像描述生成

编码器处理图像特征，解码器通过Cross Attention生成描述文本。

---

## 6. 优缺点分析

### 6.1 优点

1. **跨序列信息传递**：实现源序列到目标序列的信息流动
2. **灵活的长程依赖**：直接建立任意位置间的关注关系
3. **并行计算**：相比RNN，可以并行计算所有位置
4. **可解释性强**：注意力权重可以可视化

### 6.2 缺点

1. **二次复杂度**：$O(n^2)$的注意力计算
2. **内存消耗大**：需要存储注意力矩阵
3. **顺序敏感**：没有位置编码时无法区分顺序

### 6.3 改进方向

1. **稀疏注意力**：局部+全局的稀疏模式
2. **线性注意力**：核函数近似实现线性复杂度
3. **Flash Attention**：IO-aware的高效实现
4. **Informer**：针对长序列的改进
5. **Longformer**：滑动窗口+全局注意力

---

## 7. 调库实现（PyTorch完整代码）

### 7.1 标准Transformer中的Cross Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        初始化交叉注意力层
        
        Args:
            d_model: 模型隐藏维度
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)  # Query (来自解码器)
        self.W_K = nn.Linear(d_model, d_model)  # Key (来自编码器)
        self.W_V = nn.Linear(d_model, d_model)  # Value (��自编码器)
        self.W_O = nn.Linear(d_model, d_model)  # 输出
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, encoder_output, encoder_mask=None):
        """
        前向传播
        
        Args:
            query: 解码器输入 [batch, tgt_len, d_model]
            encoder_output: 编码器输出 [batch, src_len, d_model]
            encoder_mask: 编码器mask [batch, src_len]
            
        Returns:
            交叉注意力输出 [batch, tgt_len, d_model]
        """
        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = encoder_output.size(1)
        
        # 残差连接
        residual = query
        
        # Q, K, V投影
        Q = self.W_Q(query).view(batch_size, tgt_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(encoder_output).view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(encoder_output).view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Mask处理
        if encoder_mask is not None:
            # 创建attention mask [batch, 1, 1, src_len]
            mask = encoder_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        
        # 输出投影
        output = self.W_O(context)
        
        # 残差连接和LayerNorm
        output = self.layer_norm(output + residual)
        
        return output, attn_weights


class TransformerDecoderLayerWithCrossAttention(nn.Module):
    """带交叉注意力的Transformer解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 自注意力（带mask）
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, encoder_mask=None):
        """
        解码器层前向传播
        
        Args:
            x: 解码器输入
            encoder_output: 编码器输出
            tgt_mask: 目标序列mask
            encoder_mask: 编码器输出mask
        """
        # 自注意力
        x_norm = self.norm1(x)
        self_attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm, attn_mask=tgt_mask
        )
        x = x + self.dropout(self_attn_output)
        
        # 交叉注意力
        x_norm = self.norm2(x)
        cross_attn_output, attn_weights = self.cross_attn(
            x_norm, encoder_output, encoder_mask
        )
        x = x + cross_attn_output
        
        # 前馈网络
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + ff_output
        
        return x, attn_weights


# 示例使用
def demo_cross_attention():
    """演示Cross Attention"""
    print("=== Cross Attention 演示 ===\n")
    
    # 参数
    batch_size = 2
    src_len = 5  # 源序列长度
    tgt_len = 4   # 目标序列长度
    d_model = 8   # 模型维度
    num_heads = 2   # 注意力头数
    
    # 模拟数据
    encoder_output = torch.randn(batch_size, src_len, d_model)
    query = torch.randn(batch_size, tgt_len, d_model)
    encoder_mask = torch.ones(batch_size, src_len)  # 无padding
    
    # 创建模型
    attention = MultiHeadCrossAttention(d_model, num_heads)
    
    # 前向传播
    output, attn_weights = attention(query, encoder_output, encoder_mask)
    
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"Query形状: {query.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 可视化注意力权重
    print(f"\n注意力权重示例 (batch 0):")
    print(attn_weights[0].detach().numpy())


if __name__ == "__main__":
    demo_cross_attention()
```

### 7.2 使用Hugging Face Transformers

```python
from transformers import AutoModel, AutoTokenizer
import torch

class CrossAttentionModel:
    """使用Hugging Face Transformers实现带Cross Attention的模型"""
    
    def __init__(self, model_name="t5-small"):
        """
        初始化模型
        
        Args:
            model_name: 模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_cross_attention(self, source_text, target_text):
        """
        获取Cross Attention权重
        
        Args:
            source_text: 源文本
            target_text: 目标文本（部分）
            
        Returns:
            注意力权重
        """
        # 编码
        source_encoding = self.tokenizer(
            source_text, 
            return_tensors="pt",
            padding=True
        )
        
        # 仅对source做encoder
        with torch.no_grad():
            encoder_output = self.model.encoder(
                input_ids=source_encoding.input_ids,
                attention_mask=source_encoding.attention_mask
            ).last_hidden_state
        
        # 模拟decoder attention (实际使用需要完整的seq2seq模型)
        print(f"源文本编码长度: {source_encoding.input_ids.shape[1]}")
        print(f"编码器输出形状: {encoder_output.shape}")
        
        return encoder_output


# 示例
def demo_huggingface():
    print("=== Hugging Face Transformers示例 ===\n")
    # 注意：T5等模型内部实现了Cross Attention
    print("Cross Attention在Decoder层中自动使用")
    print("可以直接调用 model.generate() 生成文本")


if __name__ == "__main__":
    demo_huggingface()
```

---

## 8. 手工代码实现

```python
import numpy as np
import torch
import torch.nn as nn
import math

def softmax(x, axis=-1):
    """数值稳定的softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class SimpleCrossAttention:
    """纯PyTorch实现的Cross Attention（不含框架）"""
    
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 简化版：使用nn.Parameter
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_K = nn.Parameter(torch.randn(d_model, d_model))
        self.W_V = nn.Parameter(torch.randn(d_model, d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_model))
    
    def forward(self, query, key, value, mask=None):
        """
        Cross Attention前向传播
        
        Args:
            query: [batch, tgt_len, d_model]
            key: [batch, src_len, d_model]
            value: [batch, src_len, d_model]
            mask: [batch, src_len]
        """
        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)
        
        # 投影
        Q = torch.matmul(query, self.W_Q.view(self.d_model, self.d_model))
        K = torch.matmul(key, self.W_K.view(self.d_model, self.d_model))
        V = torch.matmul(value, self.W_V.view(self.d_model, self.d_model))
        
        # Reshape to multi-head
        Q = Q.view(batch_size, self.num_heads, tgt_len, self.d_k).transpose(1, 2)
        K = K.view(batch_size, self.num_heads, src_len, self.d_k).transpose(1, 2)
        V = V.view(batch_size, self.num_heads, src_len, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, tgt_len, self.d_model)
        
        # 输出投影
        output = torch.matmul(context, self.W_O.view(self.d_model, self.d_model))
        
        return output, attn_weights


def manual_cross_attention():
    """手动实现Cross Attention（纯Python逻辑）"""
    print("=== 手动实现Cross Attention ===\n")
    
    # 参数
    batch_size = 1
    src_len = 3
    tgt_len = 2
    d_model = 4
    num_heads = 2
    d_k = d_model // num_heads
    
    # 模拟数据
    Q = np.random.randn(batch_size, tgt_len, d_model)
    K = np.random.randn(batch_size, src_len, d_model)
    V = np.random.randn(batch_size, src_len, d_model)
    
    # 简化的线性投影（直接用随机矩阵）
    W_Q = np.random.randn(d_model, d_model)
    W_K = np.random.randn(d_model, d_model)
    W_V = np.random.randn(d_model, d_model)
    
    # 投影
    Q_proj = np.einsum('btd,dk->btk', Q, W_Q)
    K_proj = np.einsum('bsd,dk->bsk', K, W_K)
    V_proj = np.einsum('bsd,dk->bsk', V, W_V)
    
    # 计算注意力分数
    scores = np.einsum('bth,bsh->bts', Q_proj, K_proj) / np.sqrt(d_k)
    
    # Softmax
    attn = softmax(scores, axis=-1)
    
    # 加权求和
    output = np.einsum('bts,bsd->btd', attn, V_proj)
    
    print(f"输入Q形状: {Q.shape}")
    print(f"输入K形状: {K.shape}")
    print(f"输入V形状: {V.shape}")
    print(f"输出形状: {output.shape}")
    print(f"\n注意力权重 (batch 0):\n{attn[0]}")
    
    return output, attn


if __name__ == "__main__":
    manual_cross_attention()
    
    # PyTorch版本
    print("\n" + "="*50)
    print("PyTorch实现")
    print("="*50)
    
    # 测试
    query = torch.randn(1, 2, 8)
    key = torch.randn(1, 3, 8)
    value = torch.randn(1, 3, 8)
    
    attention = SimpleCrossAttention(d_model=8, num_heads=2)
    output, weights = attention(query, key, value)
    
    print(f"Query: {query.shape}")
    print(f"Key: {key.shape}")
    print(f"Value: {value.shape}")
    print(f"Output: {output.shape}")
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_cross_attention():
    """可视化Cross Attention"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross Attention 分析', fontsize=14, fontweight='bold')
    
    # 1. 注意力权重热力图
    ax1 = axes[0, 0]
    # 模拟：目标序列位置对源序列的注意力
    src_tokens = ['The', 'cat', 'sits', 'on', 'the', 'mat']
    tgt_tokens = ['猫', '坐在', '垫子', '上']
    
    # 创建模拟注意力矩阵
    np.random.seed(42)
    attn_matrix = np.random.rand(len(tgt_tokens), len(src_tokens))
    attn_matrix = attn_matrix / attn_matrix.sum(axis=1, keepdims=True)
    
    im = ax1.imshow(attn_matrix, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(len(src_tokens)))
    ax1.set_yticks(range(len(tgt_tokens)))
    ax1.set_xticklabels(src_tokens, rotation=45, ha='right')
    ax1.set_yticklabels(tgt_tokens)
    ax1.set_xlabel('Source (English)', fontsize=10)
    ax1.set_ylabel('Target (Chinese)', fontsize=10)
    ax1.set_title('注意力权重热力图', fontsize=11)
    plt.colorbar(im, ax=ax1)
    
    # 2. 不同层的注意力分布
    ax2 = axes[0, 1]
    layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6']
    avg_attn = [0.85, 0.72, 0.65, 0.58, 0.45, 0.32]  # 模拟：高层更分散
    
    ax2.bar(layers, avg_attn, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Decoder Layer', fontsize=10)
    ax2.set_ylabel('平均注意力熵', fontsize=10)
    ax2.set_title('各层注意力分散程度', fontsize=11)
    ax2.set_xticklabels(layers, rotation=30, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 头数对性能的影响
    ax3 = axes[1, 0]
    num_heads = [1, 2, 4, 8, 16]
    bleu_scores = [22.5, 24.8, 26.1, 27.3, 27.8]
    memory = [100, 180, 320, 580, 1050]
    
    ax3_twin = ax3.twinx()
    line1, = ax3.plot(num_heads, bleu_scores, 'b-o', linewidth=2, markersize=8, label='BLEU')
    line2, = ax3_twin.plot(num_heads, memory, 'r--s', linewidth=2, markersize=8, label='Memory (MB)')
    
    ax3.set_xlabel('Number of Heads', fontsize=10)
    ax3.set_ylabel('BLEU Score', fontsize=10, color='blue')
    ax3_twin.set_ylabel('Memory (MB)', fontsize=10, color='red')
    ax3.set_title('头数对性能的影响', fontsize=11)
    ax3.legend(handles=[line1, line2], loc='upper left')
    ax3.set_xticks(num_heads)
    ax3.grid(True, alpha=0.3)
    
    # 4. Query长度影响
    ax4 = axes[1, 1]
    q_lengths = [5, 10, 20, 40, 80]
    time_cost = [0.1, 0.4, 1.6, 6.4, 25.6]
    
    ax4.plot(q_lengths, time_cost, 'g-o', linewidth=2, markersize=8)
    ax4.fill_between(q_lengths, time_cost, alpha=0.3, color='green')
    ax4.set_xlabel('Target Sequence Length', fontsize=10)
    ax4.set_ylabel('Computation Time (ms)', fontsize=10)
    ax4.set_title('序列长度vs计算时间 (O(n²))', fontsize=11)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_attention_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n图表已保存到 cross_attention_analysis.png")


def explain_attention_weights():
    """解释注意力权重的含义"""
    print("="*50)
    print("注意力权重解释")
    print("="*50)
    
    # 示例：机器翻译
    print("\n【示例：英译中】")
    print("-" * 40)
    print("源: The cat sits on the mat")
    print("目: 猫坐在垫子上")
    print("\n注意力权重解读:")
    print("  - '猫' 关注 'cat' (权重 0.92)")
    print("  - '坐' 关注 'sits' (权重 0.88)")
    print("  - '在' 关注 'on' (权重 0.76)")
    print("  - '垫子' 关注 'mat' (权重 0.84)")
    print("  - '上' 关注 'the mat' (权重 0.65)")
    
    print("\n【规律】")
    print("  1. 大多数位置有清晰的对应关系")
    print("  2. 功能词(如介词)注意力较分散")
    print("  3. 高层注意力更全局化")
    print("  4. 注意力可解释性强")


if __name__ == "__main__":
    visualize_cross_attention()
    explain_attention_weights()
```

---

## 10. 模型评估

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

def evaluate_cross_attention(attn_weights, alignment=None):
    """
    评估Cross Attention
    
    Args:
        attn_weights: 注意力权重 [tgt_len, src_len]
        alignment: 真实对齐（可选）
        
    Returns:
        评估指标
    """
    results = {}
    
    # 1. 注意力熵（衡量分散程度）
    eps = 1e-10
    entropy = -np.sum(attn_weights * np.log(attn_weights + eps), axis=1)
    results['avg_entropy'] = np.mean(entropy)
    results['max_entropy'] = np.max(entropy)
    
    # 2. 稀疏性
    results['sparsity'] = np.mean(attn_weights < 0.01)
    
    # 3. 对齐质量（如果有ground truth）
    if alignment is not None:
        pred_align = np.argmax(attn_weights, axis=1)
        results['alignment_acc'] = np.mean(pred_align == alignment)
    
    return results


def compare_num_heads():
    """对比不同的头数"""
    print("=== 注意力头数对比实验 ===\n")
    
    configs = [
        {'heads': 1, 'name': 'Single Head'},
        {'heads': 4, 'name': '4 Heads'},
        {'heads': 8, 'name': '8 Heads'},
        {'heads': 16, 'name': '16 Heads'},
    ]
    
    print(f"{'配置':<15} {'BLEU':<10} {'Memory':<12} {'Speed':<10}")
    print("-" * 50)
    
    for config in configs:
        # 模拟结果
        bleu = 24.0 + 0.25 * config['heads']
        memory = 100 + 50 * config['heads']
        speed = 1.0 - 0.05 * config['heads']
        
        print(f"{config['name']:<15} {bleu:<10.2f} {memory:<12} {speed:<10.2f}x")
    
    print("\n结论：头数增加 → BLEU��高、内存增加、速度下降")
    print("建议：小模型用4-8头，大模型用16头")


def analyze_attention_patterns():
    """分析注意力模式"""
    print("\n=== 常见注意力模式 ===\n")
    
    patterns = [
        "对角线型（局部注意力）",
        "对齐型（跨序列对应）",
        "层次型（全局+局部）",
        "稀疏型（选择性注意）"
    ]
    
    for i, pattern in enumerate(patterns, 1):
        print(f"{i}. {pattern}")
    
    print("\n【实用建议】")
    print("  - 可视化检查注意力是否合理")
    print("  - 对齐型注意力对翻译任务重要")
    print("  - 异常模式可能表示模型问题")


if __name__ == "__main__":
    compare_num_heads()
    analyze_attention_patterns()
```

---

## 11. 常见问题与易错点

### 11.1 Query/Key/Value顺序混淆

**问题**：混淆哪个序列是Query，哪个是Key/Value

**解决**：记住——"我"（解码器当前状态）看"你"（编码器输出），所以：
- Query = 解码器
- Key/Value = 编码器

```python
# 正确理解
# Q: decoder_hidden_states  # 你想查询什么
# K: encoder_output   # 我（编码器）有什么Key
# V: encoder_output   # 我（编码器）有什么Value
output, _ = cross_attn(Q=decoder_state, K=encoder_output, V=encoder_output)
```

### 11.2 Mask应用错误

**问题**：Mask应用到错误位置

**解决**：Cross Attention的mask应该mask掉Key/Value中的padding位置

```python
# 正确
scores = Q @ K^T / sqrt(d_k)
scores = scores.masked_fill(key_padding_mask == 0, -inf)  # mask的是K中的位置
attn = softmax(scores, dim=-1)
output = attn @ V
```

### 11.3 维度不匹配

**问题**：d_model不能被num_heads整除

**解决**：确保num_heads是d_model的因数

```python
# 正确
assert d_model % num_heads == 0
d_k = d_model // num_heads
```

### 11.4 残差连接缺失

**问题**：没有残差连接导致训练不稳定

**解决**：添加残差连接和LayerNorm

```python
# 标准Transformer
output = self.layer_norm(x + attention_output + feed_forward_output)
```

### 11.5 区分不开Self和Cross Attention

**问题**：混淆Self Attention和Cross Attention的计算

**解决**：
```
Self-Attention: Q, K, V 都来自同一个序列
Cross-Attention: Q来自目标序列, K, V 来自源序列
```

---

## 12. 学习总结

**Cross Attention核心要点**：

1. **跨序列交互**：实现编码器-解码器信息传递
2. **Query/Key/Value分离**：Query来自解码器，K/V来自编码器
3. **注意力计算**：点积注意力 + softmax
4. **多头增强**：多个注意力头捕获不同关系
5. **位置敏感**：需要位置编码区分位置

**为什么Cross Attention有效**：
- 允许解码器按需访问编码器的任意位置信息
- 可学习的软对齐，不依赖显式对齐标注
- 并行计算，效率高

---

## 13. 练习题与思考题与思考题

### 13.1 选择题

1. Cross Attention中，Query来自哪里？
   - A) 编码器
   - B) 解码器
   - C) 嵌入层
   - D) 都是
   
   **答案：B**

2. 设d_model=512, num_heads=8，则d_k等于多少？
   - A) 64
   - B) 8
   - C) 512
   - D) 4096
   
   **答案：A**（d_k = 512/8 = 64）

3. Cross Attention的时间复杂度是？
   - O(n²) 
   - O(n³)
   - O(n)
   - O(1)
   
   **答案：A**（当源和目标长度相同时）

### 13.2 简答题

1. **问题**：简述Cross Attention vs Self-Attention的区别
   
   **答案**：
   - Self-Attention: Q, K, V来自同一序列，用于特征提取
   - Cross-Attention: Q来自目标序列，K, V来自源序列，用于跨序列信息传递

2. **问题**：为什么需要多��注��力？
   
   **答案**：多头可以同时关注不同位置的不同关系，一个头关注语法，一个头关注语义等。

3. **问题**：Cross Attention如何实现机器翻译的对齐？
   
   **答案**：通过学习注意力权重，自动建立源语言词到目标语言词的对应关系。

### 13.3 编程题

1. **题目**：实现一个简化版Cross Attention
   
   ```python
   import torch
   import torch.nn.functional as F
   
   def cross_attention(Q, K, V, mask=None):
       """
       简化版Cross Attention
       Q: [batch, tgt_len, d]
       K: [batch, src_len, d]
       V: [batch, src_len, d]
       """
       d_k = Q.size(-1)
       scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
       
       if mask is not None:
           scores = scores.masked_fill(mask == 0, float('-inf'))
       
       attn = F.softmax(scores, dim=-1)
       return torch.matmul(attn, V)
   ```

### 13.4 思考题

1. **问题**：如果编码器和解码器长度差异很大，注意力计算会怎样？
   
   **思考**：时间复杂度和空间复杂度取决于较短序列的平方。仍需要计算完整的注意力矩阵。

2. **问题**：为什么Cross Attention在最后一层解码器层特别重要？
   
   **思考**：因为需要最后一次"读取"编码器信息来生成输出。

---

## 14. 学习路径建议建议

### 14.1 入门路径

1. **注意力机制基础** → 理解Softmax Attention
2. **Self-Attention** → 理解Transformer编码器
3. **Cross Attention** → 理解Transformer解码器
4. **完整Seq2Seq** → 构建翻译系统

### 14.2 进阶路径

1. **Sparse Attention** → 理解高效变体
2. **Linear Attention** → 理解线性近似
3. **Flash Attention** → 理解IO优化
4. **Long Sequence** → 理解长序列处理

### 14.3 推荐资源

**论文**：
- "Attention Is All You Need" (Transformer原始论文)
- "Neural Machine Translation by Jointly Learning to Align and Translate"

**实践**：
- Hugging Face Transformers
- fairseq

---

*Cross Attention是现代序列到序列模型的核心，理解它是掌握Transformer架构的关键一步。*

## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Cross_Attention的核心思想及适用场景。
<details><summary>参考答案</summary>
Cross_Attention通过数据驱动学习输入到输出的映射，适用于深度学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Cross_Attention的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Cross_Attention核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Cross_Attention在什么情况下会失效？
2. 训练数据很少时，Cross_Attention还能有效工作吗？
3. 如何将Cross_Attention与其他方法结合？

