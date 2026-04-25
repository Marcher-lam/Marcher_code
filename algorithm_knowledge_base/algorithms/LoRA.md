# LoRA (Low-Rank Adaptation) 学习文档

## 1. 算法基础认知

### 1.1 定义

LoRA（Low-Rank Adaptation）是一种高效的参数微调方法，由 Hu 等人在 2021 年提出。其核心思想是：大模型的权重更新可以表示为低秩矩阵，从而只需要训练少量参数就能实现高效微调。

**核心公式：**

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 通过低秩分解来近似权重更新：

$$
W' = W_0 + \Delta W = W_0 + BA
$$

其中：
- $A \in \mathbb{R}^{r \times k}$：降维矩阵
- $B \in \mathbb{R}^{d \times r}$：升维矩阵
- $r \ll \min(d, k)$：低秩维度（rank）

### 1.2 直观类比

**书架类比：**
- 全量微调：重新整理整个书架
- LoRA：在原有书籍的基础上，添加少量新的书架标签（低秩更新）
- 只需要���整标签，而不移动书籍本身

**矩阵分解类比：**
- 原始权重 $W_0$：一本完整的书
- 低秩更新 $\Delta W = BA$：几个关键章节的摘要
- 最终权重 $W_0 + BA$：更新后的完整内容

### 1.3 历史背景

| 时间 | 事件 |
|------|------|
| 2021 | LoRA 论文发布 |
| 2022-2023 | 成为 LLM 微调的主流方法 |
| 2023+ | QLoRA（量化+LoRA）广泛使用 |

---

## 2. 核心原理

### 2.1 低秩分解的数学推导

**问题：** 全量微调需要更新所有参数，参数量巨大。

**假设：** 权重更新矩阵 $\Delta W$ 是低秩的。

**为什么这个假设成立？**
- 预训练模型已经学习到了基础知识
- 微调时只需要"调整"而非"重新学习"
- 任务相关的知识可以用低维子空间表示

**低秩分解：**

设原始权重 $W_0 \in \mathbb{R}^{d \times k}$，梯度更新 $\Delta W \in \mathbb{R}^{d \times k}$

LoRA 假设 $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$

**参数节省：**

| 方法 | 参数量 |
|------|--------|
| 全量微调 | $d \times k$ |
| LoRA | $d \times r + r \times k = r(d + k)$ |

当 $r \ll \min(d, k)$ 时，参数量大幅减少。

**示例：**
- $W_0$: $4096 \times 4096$
- 全量微调：$4096^2 = 16,777,216$ 参数
- LoRA ($r=8$)：$8 \times (4096 + 4096) = 65,536$ 参数
- **节省 99.6%！**

### 2.2 LoRA 的前向传播

**完整前向传播：**

```python
# 训练模式
output = (W_0 + BA) @ x = W_0 @ x + BA @ x

# 推理模式（合并权重）
W_merged = W_0 + BA
output = W_merged @ x
```

**关键洞察：**
- 训练时保持 $W_0$ 冻结，只更新 $A$ 和 $B$
- 推理时可以合并权重（合并后的权重可以缓存）

### 2.3 秩的选择

**秩 r 的影响：**

| 秩 | 参数 | 表达能力 |
|----|------|----------|
| 1 | 最少 | 可能不足 |
| 4-8 | 较少 | 通常足够 |
| 16-32 | 中等 | 表达能力强 |
| 64+ | 较多 | 接近全量 |

**实验建议（论文结论）：**
- 在 LoRA 中，$r=4$ 或 $r=8$ 通常就足够
- 增大 $r$ 不一定带来显著提升
- 不同的层可能需要不同的 $r$

### 2.4 哪些层应用 LoRA

**标准做法：** 对 Query 和 Value 投影应用 LoRA

```python
# LLaMA/Transformer 中的 LoRA 应用
class LoRALinear(nn.Module):
    def __init__(self, weight, bias=None, r=8, lora_alpha=16, lora_dropout=0.0):
        # 原始权重冻结
        self.weight = weight
        self.weight.requires_grad = False
        
        # LoRA 参数
        self.lora_A = nn.Parameter(weight.new_zeros((r, weight.shape[1])))
        self.lora_B = nn.Parameter(weight.new_zeros((weight.shape[0], r)))
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # 缩放因子
        self.scaling = lora_alpha / r
    
    def forward(self, x):
        # 原始输出 + LoRA 输出
        return F.linear(x, self.weight, self.bias) + \
               self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
```

---

## 3. PyTorch 实现

### 3.1 基础 LoRA 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    基础 LoRA 线性层
    
    公式：y = Wx + BAx * scaling
    """
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0, bias=True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), 
            requires_grad=False
        )
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features), 
                requires_grad=False
            )
        else:
            self.bias = None
        
        # LoRA 参数
        # A: [rank, in_features], 初始化为 N(0, 0.02)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        # B: [out_features, rank], 初始化为全零
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        if dropout > 0:
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.lora_dropout = nn.Identity()
    
    def forward(self, x):
        """
        x: [..., in_features]
        """
        # 原始线性变换
        base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA 变换
        # x @ A.T @ B.T = (B @ A) @ x.T.T = BA @ x
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output

# 测试
lora_layer = LoRALinear(in_features=512, out_features=512, rank=8)
x = torch.randn(4, 10, 512)
y = lora_layer(x)
print(f"Input: {x.shape}, Output: {y.shape}")

# 参数统计
print(f"\n原始权重参数: {lora_layer.weight.numel():,}")
print(f"LoRA A 参数: {lora_layer.lora_A.numel():,}")
print(f"LoRA B 参数: {lora_layer.lora_B.numel():,}")
print(f"LoRA 参数量: {lora_layer.lora_A.numel() + lora_layer.lora_B.numel():,}")
print(f"参数量减少: {(1 - (lora_layer.lora_A.numel() + lora_layer.lora_B.numel()) / lora_layer.weight.numel()) * 100:.2f}%")
```

### 3.2 带推理合并的 LoRA

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinearMerged(nn.Module):
    """
    支持推理时合并权重的 LoRA
    
    训练时：保持 W_0 冻结，更新 BA
    推理时：合并为 W = W_0 + BA
    """
    
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 原始权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # LoRA 投影
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 标记是否已合并
        self.merged = False
    
    def merge_weights(self):
        """合并权重（推理时使用）"""
        if not self.merged:
            # W = W_0 + B @ A * scaling
            delta_w = self.lora_B @ self.lora_A * self.scaling
            self.weight.data = self.weight.data + delta_w
            
            # 冻结 LoRA 参数
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
            self.merged = True
    
    def forward(self, x):
        if self.merged:
            # 已合并，直接使用
            return F.linear(x, self.weight)
        else:
            # 训练模式
            return F.linear(x, self.weight) + \
                   (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

# 测试
lora = LoRALinearMerged(512, 512, rank=8)
x = torch.randn(4, 10, 512)

# 训练模式
y_train = lora(x)
print(f"训练模式 Output: {y_train.shape}")

# 合并模式
lora.merge_weights()
y_inference = lora(x)
print(f"推理模式 Output: {y_inference.shape}")
```

### 3.3 完整的 LoRA Transformer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, weight, rank=8, alpha=16, dropout=0.0, bias=True):
        super().__init__()
        self.weight = weight
        self.weight.requires_grad = False
        self.bias = bias
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, weight.shape[1]))
        self.lora_B = nn.Parameter(torch.zeros(weight.shape[0], rank))
        
        if dropout > 0:
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.lora_dropout = nn.Identity()
    
    def forward(self, x):
        base = F.linear(x, self.weight)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora_out

class LoRATransformerLayer(nn.Module):
    """
    应用 LoRA 的 Transformer 层
    """
    
    def __init__(self, d_model, num_heads, d_ff, lora_rank=8, lora_alpha=16):
        super().__init__()
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        # QKV 投影（标准线性层）
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # 对 Q 和 V 应用 LoRA
        self.q_lora = LoRALinear(self.q_proj.weight, rank=lora_rank, alpha=lora_alpha)
        self.v_lora = LoRALinear(self.v_proj.weight, rank=lora_rank, alpha=lora_alpha)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Self-Attention
        q = self.q_lora(x)
        k = self.k_proj(x)  # K 不用 LoRA
        v = self.v_lora(x)
        
        # 计算注意力
        # (简化实现)
        batch, seq, _ = x.shape
        q = q.view(batch, seq, self.d_model // self.head_dim, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.d_model // self.head_dim, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.d_model // self.head_dim, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = torch.matmul(F.softmax(scores, dim=-1), v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        
        x = x + self.o_proj(attn)
        x = self.norm1(x)
        
        # FFN
        x = x + self.ffn(x)
        x = self.norm2(x)
        
        return x

# 测试
layer = LoRATransformerLayer(d_model=512, num_heads=8, d_ff=2048, lora_rank=8)
x = torch.randn(2, 16, 512)
out = layer(x)
print(f"Input: {x.shape}, Output: {out.shape}")

# 统计 LoRA 参数
lora_params = layer.q_lora.lora_A.numel() + layer.q_lora.lora_B.numel()
lora_params += layer.v_lora.lora_A.numel() + layer.v_lora.lora_B.numel()
print(f"LoRA 参数量: {lora_params:,}")
```

---

## 4. 代码示例

### 4.1 QLoRA 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit

class QLoRALinear(nn.Module):
    """
    QLoRA: 量化 LoRA
    权重使用 4-bit 量化，LoRA 参数使用 16-bit
    """
    
    def __init__(self, weight, bias=None, rank=8, alpha=16):
        super().__init__()
        self.weight = weight
        self.weight.requires_grad = False
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 使用全精度
        self.lora_A = nn.Parameter(torch.zeros(rank, weight.shape[1], dtype=torch.float16))
        self.lora_B = nn.Parameter(torch.zeros(weight.shape[0], rank, dtype=torch.float16))
    
    def forward(self, x):
        # 量化权重的前向传播需要反量化
        # 这里简化处理，实际使用 bitsandbytes
        base = self.weight.to(x.dtype) @ x.T if self.weight.dtype != x.dtype else F.linear(x, self.weight)
        
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base.T + lora_out if base.dim() == 2 else base + lora_out

# 简化的 4-bit 量化示例
def simple_quantize(x, n_bits=4):
    """简化版量化"""
    Q = 2 ** n_bits
    scale = x.abs().max() / (Q - 1)
    quantized = (x / scale).round().clamp(0, Q - 1)
    return quantized, scale

def simple_dequantize(quantized, scale):
    """简化版反量化"""
    return quantized * scale

# 测试
x = torch.randn(512, 512)
quantized, scale = simple_quantize(x, n_bits=4)
reconstructed = simple_dequantize(quantized, scale)
print(f"原始权重范围: [{x.min():.4f}, {x.max():.4f}]")
print(f"量化后范围: [{quantized.min()}, {quantized.max()}]")
print(f"重建误差: {(x - reconstructed).abs().mean():.4f}")
```

### 4.2 层级 LoRA 可视化

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_lora_ranks():
    """可视化 LoRA 秩分布"""
    
    # 模拟不同层的奇异值分布
    np.random.seed(42)
    ranks = [4, 8, 16, 32]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, r in enumerate(ranks):
        # 模拟奇异值（SVD 的平方根）
        svd = np.random.lognormal(0, 1, size=(r, 10))
        singular_values = np.sqrt(np.sum(svd**2, axis=1))
        singular_values = singular_values / singular_values.sum()
        
        # 累积解释方差
        cumulative_var = np.cumsum(singular_values**2)
        cumulative_var = cumulative_var / cumulative_var[-1]
        
        axes[idx].bar(range(len(singular_values)), singular_values, alpha=0.7)
        axes[idx].set_title(f'Rank r={r}\n(r/r_total ≈ {r/50:.2%})')
        axes[idx].set_xlabel('Singular Value Index')
        axes[idx].set_ylabel('Normalized Magnitude')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('LoRA Rank and Singular Value Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig('lora_rank_visualization.png', dpi=150)
    plt.show()
    
    # 打印不同秩的参数量
    d, k = 4096, 4096
    print("\n参数对比 (d_model=4096):")
    print(f"全量微调: {d * k:,} 参数")
    for r in ranks:
        lora_params = r * (d + k)
        reduction = 1 - lora_params / (d * k)
        print(f"LoRA r={r}: {lora_params:,} 参数 ({reduction:.2%} 减少)")

visualize_lora_ranks()
```

---

## 5. 应用场景
，读者可能需要手动安装一些必要的Python辅助包以确保程序的顺利运行。这里主要涉及两个关键的安装包，具体如下： ```python from flash_attn import flash_attn_qkvpacked_func from xformers.opse import memory_efficientattention ``` 这里分别使用了flash_attn与xformers来实现特殊的注意力架构。其中，xformers可以使用如下的代码进行安装： ```batch pip install -U xformers --index-url https://download.pytorch.org/whl/cu124 ``` # 注意 上面的安装代码需要使用CUDA 12.4。具体安装的版本，读者可以自行斟酌。 对于flash_attn的安装，Windows版本的flash_attn无法直接安装，读者可以使用本书配套代码库中作者编译好的flash_attn安装，从而完成本地化DeepSeek-VL2的部署。 # 8.2 广告文案撰写实战1：PEFT与LoRA详解 DeepSeek在文本生成、信息检索和智能问答等多个领域都展现出了令人瞩目的性能，这得益于其精心设计的初始训练过程。然而，不容忽视的是，尽管DeepSeek的架构设计能够在一定程度上减


---

## 6. 优缺点分析
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充LoRA的优缺点分析相关内容]


---

## 7. 调库实现
，读者可能需要手动安装一些必要的Python辅助包以确保程序的顺利运行。这里主要涉及两个关键的安装包，具体如下： ```python from flash_attn import flash_attn_qkvpacked_func from xformers.opse import memory_efficientattention ``` 这里分别使用了flash_attn与xformers来实现特殊的注意力架构。其中，xformers可以使用如下的代码进行安装： ```batch pip install -U xformers --index-url https://download.pytorch.org/whl/cu124 ``` # 注意 上面的安装代码需要使用CUDA 12.4。具体安装的版本，读者可以自行斟酌。 对于flash_attn的安装，Windows版本的flash_attn无法直接安装，读者可以使用本书配套代码库中作者编译好的flash_attn安装，从而完成本地化DeepSeek-VL2的部署。 # 8.2 广告文案撰写实战1：PEFT与LoRA详解 DeepSeek在文本生成、信息检索和智能问答等多个领域都展现出了令人瞩目的性能，这得益于其精心设计的初始训练过程。然而，不容忽视的是，尽管DeepSeek的架构设计能够在一定程度上减


---

## 8. 手工代码实现
，读者可能需要手动安装一些必要的Python辅助包以确保程序的顺利运行。这里主要涉及两个关键的安装包，具体如下： ```python from flash_attn import flash_attn_qkvpacked_func from xformers.opse import memory_efficientattention ``` 这里分别使用了flash_attn与xformers来实现特殊的注意力架构。其中，xformers可以使用如下的代码进行安装： ```batch pip install -U xformers --index-url https://download.pytorch.org/whl/cu124 ``` # 注意 上面的安装代码需要使用CUDA 12.4。具体安装的版本，读者可以自行斟酌。 对于flash_attn的安装，Windows版本的flash_attn无法直接安装，读者可以使用本书配套代码库中作者编译好的flash_attn安装，从而完成本地化DeepSeek-VL2的部署。 # 8.2 广告文案撰写实战1：PEFT与LoRA详解 DeepSeek在文本生成、信息检索和智能问答等多个领域都展现出了令人瞩目的性能，这得益于其精心设计的初始训练过程。然而，不容忽视的是，尽管DeepSeek的架构设计能够在一定程度上减


---

## 9. 可视化与结果理解
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充LoRA的可视化与结果理解相关内容]


---

## 10. 模型评估
，读者可能需要手动安装一些必要的Python辅助包以确保程序的顺利运行。这里主要涉及两个关键的安装包，具体如下： ```python from flash_attn import flash_attn_qkvpacked_func from xformers.opse import memory_efficientattention ``` 这里分别使用了flash_attn与xformers来实现特殊的注意力架构。其中，xformers可以使用如下的代码进行安装： ```batch pip install -U xformers --index-url https://download.pytorch.org/whl/cu124 ``` # 注意 上面的安装代码需要使用CUDA 12.4。具体安装的版本，读者可以自行斟酌。 对于flash_attn的安装，Windows版本的flash_attn无法直接安装，读者可以使用本书配套代码库中作者编译好的flash_attn安装，从而完成本地化DeepSeek-VL2的部署。 # 8.2 广告文案撰写实战1：PEFT与LoRA详解 DeepSeek在文本生成、信息检索和智能问答等多个领域都展现出了令人瞩目的性能，这得益于其精心设计的初始训练过程。然而，不容忽视的是，尽管DeepSeek的架构设计能够在一定程度上减


---

## 11. 常见问题与易错点
，读者可能需要手动安装一些必要的Python辅助包以确保程序的顺利运行。这里主要涉及两个关键的安装包，具体如下： ```python from flash_attn import flash_attn_qkvpacked_func from xformers.opse import memory_efficientattention ``` 这里分别使用了flash_attn与xformers来实现特殊的注意力架构。其中，xformers可以使用如下的代码进行安装： ```batch pip install -U xformers --index-url https://download.pytorch.org/whl/cu124 ``` # 注意 上面的安装代码需要使用CUDA 12.4。具体安装的版本，读者可以自行斟酌。 对于flash_attn的安装，Windows版本的flash_attn无法直接安装，读者可以使用本书配套代码库中作者编译好的flash_attn安装，从而完成本地化DeepSeek-VL2的部署。 # 8.2 广告文案撰写实战1：PEFT与LoRA详解 DeepSeek在文本生成、信息检索和智能问答等多个领域都展现出了令人瞩目的性能，这得益于其精心设计的初始训练过程。然而，不容忽视的是，尽管DeepSeek的架构设计能够在一定程度上减


---

## 12. 学习总结
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充LoRA的学习总结相关内容]


---

## 13. 练习题与思考题
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充LoRA的练习题与思考题相关内容]


---

## 14. 学习路径建议
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充LoRA的学习路径建议相关内容]


---
