# 因果掩码与自回归生成 学习文档

> 来源线索：本节内容根据原书中关于"因果掩码"与"错位输入输出格式"（第4章 4.1.4节）的相关章节整理、扩展与教学化改写。

> 用三角掩码遮住未来信息，迫使模型自左向右逐个预测——让"生成"成为可能。

## 1. 算法基础认知

**一句话定义**：通过掩码禁止模型看到未来token，实现从左到右的逐token预测。

**直觉类比**：想象你在做英语完形填空——每次只能看到当前空格之前的内容，不能偷看后面的答案。你只能基于已知的上下文推测下一个词。因果掩码正是实现了这样的"只能往前看"的限制。

**历史背景**：因果掩码（Causal Masking）与自回归（Autoregressive）架构的概念源于序列建模的基本原则。在深度学习时代，GPT系列（OpenAI, 2018）和后续的LLaMA、DeepSeek等大模型都建立在自回归解码的基础上。这种设计使得模型天然适合文本生成任务——逐token预测的机制与人类"边说边想"的语言产生过程高度一致。

**算法定位**：深度学习 / 注意力机制 / 序列生成策略。是自回归模型的核心约束机制，属于训练策略和架构设计的交叉领域。

**前置知识**：
- 自注意力机制（理解Q、K、V和注意力分数矩阵）
- Softmax函数（理解掩码值设为负无穷后变为0的原理）
- 矩阵的下三角/上三角概念
- 自编码 vs 自回归架构的基本区别

## 2. 核心原理

### 核心思想

在自回归模型中，每个token的生成是按顺序逐个进行的。为了确保在预测第 $t$ 个token时，模型不会"偷看"第 $t+1$ 个及以后的token，我们用一个三角掩码矩阵将未来位置全部遮盖。

具体而言：对于注意力分数矩阵 $S \in \mathbb{R}^{N \times N}$，其中 $S_{ij}$ 表示第 $i$ 个位置对第 $j$ 个位置的关注程度。因果掩码将 $j > i$ 的所有位置设为 $-\infty$，经softmax归一化后这些位置的权重变为0。

### 工作流程

1. **计算注意力分数**：正常计算 $S = QK^T$，得到完整的 $N \times N$ 注意力分数矩阵
2. **构建因果掩码**：创建一个上三角矩阵（不包括对角线），上三角部分为需要掩盖的未来位置
3. **应用掩码**：将掩码位置替换为一个极大的负数（通常为 `float('-inf')` 或 `-1e9`）
4. **Softmax归一化**：对每一行做softmax——被掩码位置的 $\exp(-\infty)=0$，权重自然分配给可见位置
5. **加权求和**：只使用"合法"位置对应的V值

### 关键概念解释

- **因果性（Causality）**：当前输出只能依赖于过去的输入，不能依赖未来——这保证了模型生成的合法性
- **上三角掩码 / 下三角掩码**：取决于矩阵索引的约定，本质是"遮盖当前之后的所有位置"
- **错位输入输出**：训练时输入是 $[t_1, t_2, ..., t_{N-1}]$，输出目标是 $[t_2, t_3, ..., t_N]$——让模型学习"用前面的词预测下一个词"

### 几何/直观解释

```
注意力分数矩阵（N=5时）：
          到token:  t1  t2  t3  t4  t5
从token t1:      [v,  -∞, -∞, -∞, -∞]   ← 只看自己
      t2:      [v,   v, -∞, -∞, -∞]   ← 看t1和自己
      t3:      [v,   v,  v, -∞, -∞]   ← 看前3个
      t4:      [v,   v,  v,  v, -∞]   ← 看前4个
      t5:      [v,   v,  v,  v,  v]   ← 看全部

错位输入输出示意：
  输入:  [你,  好,  人,  工,  智能]
  输出:  [好,  人,  工,  智能, SEP]
         ↑模型用"你"预测"好"，用"你好"预测"人"...
```

这种下三角结构确保了训练时对所有位置可以**并行计算损失**——每个位置都独立地"用过去预测现在"。

## 3. 数学公式与推导

### 符号约定表

| 符号 | 含义 | 维度 |
|------|------|------|
| $N$ | 序列长度 | 标量 |
| $S$ | 注意力分数矩阵 | $(N, N)$ |
| $M$ | 因果掩码矩阵 | $(N, N)$ |
| $A$ | Softmax后的注意力权重 | $(N, N)$ |
| $X_{in}$ | 输入序列 | $(N, d_{model})$ |
| $Y_{target}$ | 输出目标（错位后的） | $(N, d_{model})$ |

### 数学推导

**步骤1：标准注意力分数**

$$S = \frac{QK^T}{\sqrt{d_k}}$$

$S_{ij}$ 表示位置 $i$ 对位置 $j$ 的原始注意分数。

**步骤2：构建因果掩码**

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

掩码矩阵是一个下三角为0、上三角为 $-\infty$ 的矩阵。

**步骤3：应用掩码**

$$\tilde{S}_{ij} = S_{ij} + M_{ij}$$

当 $j > i$ 时，$\tilde{S}_{ij} = S_{ij} + (-\infty) = -\infty$。

**步骤4：Softmax归一化**

$$A_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_{k=1}^{N} \exp(\tilde{S}_{ik})}$$

由于 $\exp(-\infty) = 0$，当 $j > i$ 时 $A_{ij} = 0$。并且由于每个合法位置至少包含对角线（$j=i$ 时 $M_{ii}=0$），softmax是良好定义的。

**步骤5：注意力输出**

$$Z_i = \sum_{j=1}^{N} A_{ij} V_j = \sum_{j=1}^{i} A_{ij} V_j$$

每个输出的注意力窗口被限制在 $[1, i]$ 范围内。

### 错位训练格式的数学形式

训练时，将输入序列 $X = [x_1, x_2, ..., x_N]$ 输入模型，模型输出 $\hat{Y} = [\hat{y}_1, ..., \hat{y}_N]$，损失函数为：

$$\mathcal{L} = \sum_{t=1}^{N-1} \text{CE}(\hat{y}_t, x_{t+1})$$

即：第 $t$ 个输出位置预测第 $t+1$ 个真实token。第 $N$ 个位置通常预测一个特殊的结束符（SEP/EOS）。这种设计使得一条序列就能提供 $N-1$ 个训练样本。

## 4. 训练过程讲解

### 数据预处理

- 文本需要先进行tokenization，转换为整数ID序列
- 添加特殊token：BOS（序列开始）和EOS（序列结束）
- 构造错位对：input = [BOS, t1, t2, ..., tN]，target = [t1, t2, ..., tN, EOS]
- padding到统一长度后用padding mask忽略填充位置

### 参数初始化

- 因果掩码本身没有可训练参数——它是纯粹的确定性操作
- 通常用 `torch.tril` 或 `torch.triu` 生成掩码，与模型参数无关

### 关键实现细节

```python
# 创建因果掩码（上三角掩码）
mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
# mask[i, j] = True 当 j > i 时（即未来位置）
# 在注意力计算中: scores.masked_fill(mask, float('-inf'))
```

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| mask填充值 | 遮盖位置的替代值 | -1e9 或 -inf | `float('-inf')` |
| diagonal偏移 | 是否允许self-attention | 0或1 | 1（上三角不含对角线） |

## 5. 应用场景

### 典型应用

1. **GPT系列语言模型**：GPT-1/2/3/4全部使用因果掩码+自回归架构进行文本生成。掩码的自然约束使模型能生成长篇连贯文本。

2. **DeepSeek等现代LLM**：DeepSeek-V2/V3的解码器部分使用因果掩码，确保在API对话和代码生成场景中逐token输出的合法性。结合MLA进一步优化了KV cache。

3. **语音/音乐生成**：WaveNet等自回归音频模型也使用因果掩码——当前采样点只能基于过去的波形预测。

4. **时序预测**：金融、天气等时序数据预测中，因果掩码确保不引入未来信息，符合预测任务的真实设定。

### 适用数据特征

- 序列生成任务（文本、音频、代码等）
- 严格需要时序因果关系的数据
- 需要流式（逐个）输出的场景

### 不适用场景

- 文本分类/理解任务——这时需要双向上下文（应用BERT式的双向注意力）
- 完形填空——需要同时看到前后文
- 需要全局理解的翻译任务——编码器部分通常不用因果掩码

## 6. 优缺点分析

### 优点

| 优点 | 成立条件 | 说明 |
|------|----------|------|
| 天然适合生成任务 | 生成方向从左到右（或其他固定方向） | 逐token预测与人类的语言产生过程高度一致 |
| 训练高效 | 使用错位输入输出格式 | 一次前向传播即可对整条序列计算损失（Teacher Forcing），无需逐步自回归 |
| 推理一致 | 掩码条件在训练和推理中保持一致 | 训练时看到的注意力模式和推理时完全一致，避免train-test mismatch |
| 数学简洁 | 仅需一个三角矩阵 | 实现简单，不需要额外参数 |

### 缺点

| 缺点 | 何时出问题 | 缓解思路 |
|------|-----------|----------|
| 单向视野受限 | 分类/理解任务需要双向上下文 | 使用编码器（双向）+解码器（单向）的Encoder-Decoder架构 |
| 曝光偏差（Exposure Bias） | 推理时预测错误会累积 | 使用Scheduled Sampling或强化学习训练 |
| 长序列不如双向 | 长文本中需要远距离前瞻 | 对生成任务这是必然约束；理解任务应使用双向模型 |

### 与双向注意力的对比

| 特性 | 因果掩码（单向） | 双向注意力 |
|------|-----------------|-----------|
| 视野 | 只能看过去 | 能看到全部上下文 |
| 适用任务 | 文本生成 | 文本理解/分类/NER |
| 代表模型 | GPT, DeepSeek解码器 | BERT, DeepSeek编码器 |
| 训练效率 | Teacher Forcing高效 | 同样可以并行 |
| 推理 | 逐个token生成 | 一次推理即可 |

## 7. 调库实现

```python
"""使用PyTorch实现带因果掩码的自注意力"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """带因果掩码的自注意力层"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q、K、V联合投影
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 预计算因果掩码（形状适配多头）
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1).bool()
        )
    
    def forward(self, x, max_len=None):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # 联合投影
        qkv = self.W_qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        
        # 重塑为多头: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, heads, seq, seq)
        
        # 应用因果掩码
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        # 输出投影
        output = self.W_o(context)
        
        return output, attn_weights


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = CausalSelfAttention(d_model=256, num_heads=8)
    x = torch.randn(2, 6, 256)
    
    output, weights = model(x)
    print("=== 带因果掩码的自注意力 ===")
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    print(f"权重: {weights.shape}")
    
    # 检查因果性
    seq = 6
    for i in range(seq):
        for j in range(seq):
            if j > i:
                assert weights[0, 0, i, j].item() == 0.0, f"因果性被破坏: ({i},{j})"
    print("因果性验证通过: 所有j>i位置的权重均为0")

    # 打印注意力权重示例
    print("\n注意力权重(头0, 样本0):")
    for i in range(seq):
        row = [f"{weights[0, 0, i, j].item():.3f}" for j in range(seq)]
        print(f"  位置{i}: {row}")
    print("  (上三角为零，符合因果掩码预期)")
```

**运行结果示例**：
```
=== 带因果掩码的自注意力 ===
输入: torch.Size([2, 6, 256])
输出: torch.Size([2, 6, 256])
权重: torch.Size([2, 8, 6, 6])
因果性验证通过: 所有j>i位置的权重均为0

注意力权重(头0, 样本0):
  位置0: ['1.000', '0.000', '0.000', '0.000', '0.000', '0.000']
  位置1: ['0.489', '0.511', '0.000', '0.000', '0.000', '0.000']
  位置2: ['0.322', '0.340', '0.338', '0.000', '0.000', '0.000']
  位置3: ['0.242', '0.256', '0.254', '0.248', '0.000', '0.000']
  位置4: ['0.195', '0.202', '0.201', '0.198', '0.204', '0.000']
  位置5: ['0.163', '0.169', '0.168', '0.165', '0.170', '0.165']
  (上三角为零，符合因果掩码预期)
```

## 8. 手工代码实现

```python
"""从零手写因果掩码自回归注意力——完全使用基础张量操作"""
import torch
import torch.nn as nn
import math


class AutoregressiveAttention(nn.Module):
    """手写因果掩码自注意力 + 错位输出头
    
    不使用 nn.MultiheadAttention，仅用基础 nn.Linear 和 张量运算。
    额外包含错位输入输出格式的实现。
    """
    
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        # 词嵌入和位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 自注意力投影
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # 输出头：预测下一个token
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.scale = math.sqrt(self.d_k)
    
    def create_causal_mask(self, seq_len, device):
        """创建因果掩码矩阵
        
        返回: (seq_len, seq_len) 布尔矩阵, True=被掩盖位置
        实现原理: torch.triu(diagonal=1)创建上三角(不含对角线)的矩阵
        上三角对应 j > i 的未来位置, 设为True后被mask掉
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask
    
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len), 整数token ID
        
        使用错位格式: 
        - 注意力输入: [t1, t2, ..., t_{N-1}]
        - 预测目标: [t2, t3, ..., t_N]
        
        返回:
            logits: (batch, seq_len-1, vocab_size) 每个位置的预测
        """
        batch, seq_len = input_ids.shape
        
        # 获取嵌入
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        # x: (batch, seq_len, d_model)
        
        # Q、K、V投影
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑为多头形式
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, num_heads, seq_len, seq_len)
        
        # 创建并应用因果掩码
        causal_mask = self.create_causal_mask(seq_len, input_ids.device)
        # 扩展维度以匹配scores的形状
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)
        # context: (batch, num_heads, seq_len, d_k)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        # 输出投影
        hidden = self.W_o(context)
        
        # --- 错位输出 ---
        # 用第t个位置的hidden state预测第t+1个token
        # hidden[:, :-1] 用于预测 input_ids[:, 1:]
        logits = self.lm_head(hidden[:, :-1, :])  # (batch, seq_len-1, vocab_size)
        
        return logits, attn_weights


# ========== 测试代码 ==========
if __name__ == "__main__":
    torch.manual_seed(42)
    
    vocab_size = 100
    model = AutoregressiveAttention(
        vocab_size=vocab_size, d_model=128, num_heads=4, max_seq_len=64
    )
    
    # 模拟输入：[BOS, t1, t2, t3, t4]
    input_ids = torch.randint(0, vocab_size, (2, 5))
    print("=== 手写自回归注意力 + 错位输出 ===")
    print(f"输入 token IDs 形状: {input_ids.shape}")
    
    logits, weights = model(input_ids)
    print(f"输出 logits 形状: {logits.shape}  # (batch, seq-1, vocab)")
    
    # 验证因果性
    seq = input_ids.shape[1]
    for i in range(seq):
        for j in range(seq):
            if j > i:
                assert weights[0, 0, i, j] == 0.0
    print("因果性验证通过 ✓")
    
    # 验证错位输出
    # 输入: [BOS, t1, t2, t3, t4] (5个token)
    # 输出: 预测[t1, t2, t3, t4]  (4个位置的logits)
    assert logits.shape[1] == seq - 1
    print("错位输出格式验证通过: seq_len-1 个预测 ✓")
    
    # 展示损失计算
    targets = input_ids[:, 1:]  # 真实目标: [t1, t2, t3, t4]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
    print(f"\n交叉熵损失: {loss.item():.4f}")
```

## 9. 可视化与结果理解

```python
"""因果掩码可视化"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ------ 图1: 因果掩码矩阵 ------
N = 8
mask = np.triu(np.ones((N, N)), k=1)  # 上三角=1（被掩盖）

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始掩码
im1 = axes[0].imshow(mask, cmap='Reds', vmin=0, vmax=1)
axes[0].set_title('因果掩码矩阵 (红色=被遮盖)', fontsize=14)
axes[0].set_xlabel('Key位置 j')
axes[0].set_ylabel('Query位置 i')
for i in range(N):
    for j in range(N):
        color = 'white' if mask[i, j] > 0.5 else 'black'
        axes[0].text(j, i, 'mask' if mask[i, j] > 0.5 else 'ok',
                    ha='center', va='center', color=color, fontsize=9)

# 可用位置数: 每个query可以关注多少个key
visible = [(i+1) for i in range(N)]  # 位置i可以看i+1个token
axes[1].bar(range(1, N+1), visible, color=['#ff6b6b' if i < N//2 else '#4ecdc4' for i in range(N)])
axes[1].set_title('每个Query位置可见的Key数量', fontsize=14)
axes[1].set_xlabel('Query位置')
axes[1].set_ylabel('可见Key数量')
for i, v in enumerate(visible):
    axes[1].text(i+1, v+0.1, str(v), ha='center')

# 错位格式示意
train_text = [f"t{i}" for i in range(1, 7)]
input_tokens = ['BOS'] + train_text
target_tokens = train_text + ['EOS']
colors_in = ['#a8e6cf'] + ['#dcedc1'] * 6
colors_out = ['#ffd3b6'] * 6 + ['#ff8b94']

axes[2].axis('off')
y_positions = [0.8, 0.4]
labels = ['输入\n序列', '目标\n序列']
for y, label, tokens, colors in zip(y_positions, labels, 
                                     [input_tokens, target_tokens],
                                     [colors_in, colors_out]):
    axes[2].text(-0.1, y, label, ha='right', va='center', fontsize=12, fontweight='bold')
    for i, (t, c) in enumerate(zip(tokens, colors)):
        axes[2].text(i * 0.15, y, t, ha='center', va='center',
                    fontsize=11, bbox=dict(boxstyle='round', facecolor=c, alpha=0.7))
    # 画箭头
    for i in range(len(tokens)-1):
        axes[2].annotate('', xy=((i+1)*0.15, y-0.05), xytext=(i*0.15, y-0.05),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1))

axes[2].set_xlim(-0.2, 1.2)
axes[2].set_ylim(0, 1)
axes[2].set_title('错位输入输出格式', fontsize=14)

plt.tight_layout()
plt.savefig('causal_mask_viz.png', dpi=100)
plt.show()

print("图1解读: 因果掩码矩阵中对角线及以上(红色)被遮盖")
print("图2解读: 随着位置向后, 可见的上下文越来越长(对角线优势)")
print("图3解读: 输入='BOS t1...t6', 目标='t1...t6 EOS' —— 用当前位置预测下一位置")
```

## 10. 模型评估

自回归模型的因果掩码效果主要通过**困惑度（Perplexity）**和**生成质量**评估。

```python
"""评估因果掩码自回归模型"""
def evaluate_autoregressive(model, dataloader, device='cpu'):
    """计算困惑度（Perplexity）"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for input_ids in dataloader:
            input_ids = input_ids.to(device)
            logits, _ = model(input_ids)  # (batch, seq-1, vocab)
            targets = input_ids[:, 1:]     # (batch, seq-1)
            
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"=== 自回归模型评估 ===")
    print(f"平均交叉熵损失: {avg_loss:.4f}")
    print(f"困惑度 (Perplexity): {perplexity:.2f}")
    print(f"  (困惑度越低模型越好, 接近vocab_size时相当于随机猜测)")
    
    return perplexity

# 使用示例:
# ppl = evaluate_autoregressive(model, test_loader)
```

**结果解读**：
- 困惑度衡量模型对测试数据的"惊讶程度"。值越低越好
- 困惑度等于 `vocab_size` 时 = 等效于均匀随机猜测
- 困惑度接近1时 = 模型对序列有完美的预测能力

## 11. 常见问题与易错点

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 错位格式搞反 | 模型loss不下降，输出全是EOS | 输入=target或方向颠倒 | 确保 input = [BOS, t1, ..., tN], target = [t1, ..., tN, EOS] |
| 忘记加BOS/EOS | 生成时无法停止 | 模型不知道何时开始/结束生成 | 训练和推理时都添加BOS和EOS token |

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| mask方向反了 | 模型能看到未来，训练loss异常低 | triu和tril搞混 | 标准做法用 `triu(diagonal=1)` 创建上三角掩码 |
| 推理时不加mask | 推理结果与训练不一致 | 训练时有mask，推理时忘记加 | 确保训练和推理使用完全相同的掩码逻辑（可用 `register_buffer` 保存） |
| diagonal设置错误 | 第一个token只能看自己或完全看不到 | k值设错 | 默认 `k=1`：上三角不含对角，即第i个位置可以看到第i个（含自己） |

### 调参层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| Teacher Forcing的曝光偏差 | 推理时生成质量远差于训练 | 训练用真实历史/推理用预测历史 | Scheduled Sampling(按概率使用预测值)/RL训练 |
| 序列太长 | 早期位置信息被稀释 | 三角掩码中早期位置的注意力被后面的位置"挤占" | 增加d_k或使用ALiBi等位置偏移方法来增强近距注意力 |

## 12. 学习总结

### 核心思想回顾

因果掩码 + 错位格式的本质是**将"生成"问题转化为"分类"问题来训练**——每次预测下一个token都是一个分类任务（类别=词表大小），通过三角掩码确保训练时不引入未来信息。

核心公式：
- 因果掩码：$M_{ij} = -\infty$ 当 $j > i$
- 错位训练：$\mathcal{L} = \text{CE}(f(x_{1:t}), x_{t+1})$
- 自回归生成：$P(x_1, ..., x_N) = \prod_{t=1}^{N} P(x_t | x_1, ..., x_{t-1})$

### 与前序/相关算法的联系

- 因果掩码是**自注意力**的约束形式，去掉了head mask中的"未来"部分
- 是GPT/DeepSeek等**自回归LLM**的基础
- 与**Cross-Attention**形成对比：后者是跨序列关注，前者是本序列内的单向关注
- **双向注意力**（BERT式）是不加因果掩码的特例

### 后续学习方向

- Top-K/Top-P采样（控制生成随机性）
- KV Cache（推理加速的关键）
- Beam Search（提升生成质量）
- 自回归 vs 非自回归生成的权衡

## 13. 练习题与思考题

### 基础题1：掩码矩阵构造

写出Python代码，用一行 `torch` 操作创建一个 5×5 的因果掩码张量，被掩盖的位置为 `True`。

**参考答案**：
```python
import torch
mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()
print(mask)
# 输出:
# tensor([[False,  True,  True,  True,  True],
#         [False, False,  True,  True,  True],
#         [False, False, False,  True,  True],
#         [False, False, False, False,  True],
#         [False, False, False, False, False]])
```

### 基础题2：错位格式

给定 token 序列 [你, 好, 世, 界]（索引后为 [10, 20, 30, 40]），请写出训练时的 input 和 target 序列（假设 BOS=1, EOS=2）。

**参考答案**：
- input: [1, 10, 20, 30, 40] → [BOS, 你, 好, 世, 界]
- target: [10, 20, 30, 40, 2] → [你, 好, 世, 界, EOS]
- 模型用input[:, :4]预测target[:4]，最后一个位置（EOS）的loss通常也参与训练

### 进阶题：因果掩码与padding

如果batch中有不同长度的序列，如何同时处理因果掩码和padding掩码？写出伪代码。

**参考答案**：
需要叠加两层掩码：
1. **因果掩码**：上三角（j > i），形状 (seq_len, seq_len)，扩展到 (1, 1, seq_len, seq_len)
2. **Padding掩码**：对key序列中padding位置，形状 (batch, 1, 1, seq_len)

叠加方式：
```python
# causal_mask: (1, 1, N, N), True=被掩盖(未来位置)
# pad_mask: (batch, 1, 1, N), True=被掩盖(padding的key)
# 合并: 任何一方为True则掩盖
combined_mask = causal_mask | pad_mask
# 或: scores = scores.masked_fill(causal_mask, -inf)
#      scores = scores.masked_fill(pad_mask, -inf)
```
注意padding掩码只需要遮盖Key维度（dim=-1被关注方），而因果掩码同时约束Query和Key的关系。

### 开放思考题

除了自左向右的因果生成，是否存在其他方向的"因果"生成？比如从外向内、从粗到精（hierarchical generation）？它们的掩码矩阵如何设计？各自适合什么场景？

**参考思路**：
- **从外向内**（Insertion Transformer）：先生成两端再填空。掩码不是三角而是"挖空"模式。适合需要全局规划的并行生成。
- **从粗到精**：先生成粗粒度的token序列，再逐步细化。掩码是分层的——不同层级的掩码约束不同。适合长序列生成和图像生成。
- **双向并行生成**（Mask-Predict，如BERT的生成变体）：每次mask掉一些位置，并行预测被mask的token，迭代多轮完成。掩码是随机块。适合需要质量控制的场景。
- 每种设计都是"因果关系"定义的扩展——核心思想是"确保预测方向与信息获取方向一致"。

## 14. 学习路径建议

### 前置算法
- 自注意力机制：理解QKV和softmax的加权机制
- 词嵌入/Tokenization：理解离散token如何转化为连续表示
- 交叉熵损失：理解语言模型训练的损失函数

### 平行算法
- **填充掩码（Padding Mask）**：处理不等长序列的配套技术
- **双向注意力（BERT式）**：不加因果掩码的对比方案
- **Encoder-Decoder Attention**：编码器是双向，解码器是单向的混合架构

### 进阶算法
- **Top-K / Top-P 采样**：控制生成多样性的解码策略
- **Beam Search**：保留多条候选路径的搜索策略
- **KV Cache**：推理时缓存已计算的K和V以加速生成
- **Speculative Decoding**：用草稿模型加速自回归生成

### 推荐资源
1. **论文**：《Attention is All You Need》Section 3.2.2 — 原始的掩码设计
2. **博客**：The Illustrated GPT-2 (Jay Alammar) — GPT中因果注意力的可视化
3. **代码**：Andrej Karpathy的 nanoGPT — 极简的因果掩码实现（约300行）
