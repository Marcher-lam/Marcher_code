# Layer Normalization 学习文档

## 1. 算法基础认知

### 1.1 定义

Layer Normalization（层归一化）是由 Ba、Kiros 和 Hinton 在 2016 年提出的归一化技术。它在同一层的所有神经元之间进行归一化，而不是在 batch 维度上。

$$
\mathbf{y} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

其中：
- $\mu = \frac{1}{H} \sum_{i=1}^{H} x_i$：均值
- $\sigma^2 = \frac{1}{H} \sum_{i=1}^{H}(x_i - \mu)^2$：方差
- $\gamma, \beta$：可学习的缩放和偏移参数
- $\epsilon$：数值稳定项（通常为 $10^{-6}$）

### 1.2 直观类比

**班级考试成绩类比：**

- **Batch Norm**：将每个学生在不同科目中的表现与班级整体比较
- **Layer Norm**：将每门科目的分数在自己班级内进行比较

**关键区别：**
- Layer Norm 独立于 batch 大小
- 适用于序列模型（RNN、Transformer）

### 1.3 历史背景

| 时间 | 事件 |
|------|------|
| 2015 | Batch Normalization 被提出 |
| 2016 | Layer Normalization 被提出 |
| 2017 | Transformer 架构使用 Layer Norm |
| 2018+ | 成为 Transformer 系列模型的标准 |

---

## 2. 核心原理

### 2.1 归一化维度详解

**Layer Norm 的计算维度：**

对于输入 $\mathbf{x} \in \mathbb{R}^{B \times H}$（B=batch, H=hidden）：
- 计算均值：$\mu = \frac{1}{H}\sum_{i=1}^{H}x_i$（沿 H 维度）
- 计算方差：$\sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2$
- 归一化：$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
- 输出：$y_i = \gamma_i \hat{x}_i + \beta_i$

**对于 Transformer 中的 3D 张量 $\mathbf{x} \in \mathbb{R}^{B \times S \times H}$：**

```python
# PyTorch 实现（只看最后一个维度）
x: shape [B, S, H]
μ = x.mean(dim=-1)  # shape [B, S]
σ² = x.var(dim=-1)  # shape [B, S]
# broadcasting 自动处理
```

### 2.2 Layer Norm vs Batch Norm vs Instance Norm

| 归一化方法 | 归一化维度 | 应用场景 | batch 依赖 |
|-----------|-----------|---------|-----------|
| **Batch Norm** | $(B, C, H, W)$ → $(C)$ | CNN，CV | 强依赖 |
| **Layer Norm** | $(B, H)$ → $(H)$ | RNN，NLP | 无关 |
| **Instance Norm** | $(B, C, H, W)$ → $(B, C)$ | 风格迁移 | 无关 |
| **Group Norm** | $(B, C, H, W)$ → $(B, G)$ | 小 batch CV | 无关 |

**可视化：**

```
输入张量 [B, C, H, W]:

Batch Norm:     ┌─────────────┐
                │  对每个C    │
                │  所有B,H,W  │
                └─────────────┘

Layer Norm:     ┌─────────────┐
                │  对每个H    │
                │  所有B,C    │
                └─────────────┘

Instance Norm:  ┌─────────────┐
                │  对每个B,C  │
                │  所有H,W    │
                └─────────────┘
```

### 2.3 为什么要归一化

**问题：** 深层网络的内部协变量偏移（Internal Covariate Shift）

**解决方案：** 归一化使每一层的输入保持稳定的分布

**Layer Norm 的优势：**
1. **不依赖 batch**：适合在线学习和小 batch
2. **适合 RNN**：每个时间步独立归一化
3. **Transformer 标准**：几乎所有 Transformer 模型使用

---

## 3. PyTorch 实现

### 3.1 PyTorch 内置实现

```python
import torch
import torch.nn as nn

# 标准 Layer Norm
ln = nn.LayerNorm(normalized_shape=256)
x = torch.randn(4, 10, 256)  # batch=4, seq_len=10, hidden=256
y = ln(x)
print(f"Input: {x.shape}, Output: {y.shape}")

# 2D 输入的 Layer Norm
ln_2d = nn.LayerNorm(normalized_shape=128)
x_2d = torch.randn(16, 128)  # batch=16, features=128
y_2d = ln_2d(x_2d)
print(f"Input: {x_2d.shape}, Output: {y_2d.shape}")

# 可学习的参数
print(f"gamma (scale): {ln.weight.shape}")
print(f"beta (shift): {ln.bias.shape}")
```

### 3.2 手写实现

```python
import torch
import torch.nn as nn

def layer_norm(x, eps=1e-6):
    """
    手动实现 Layer Normalization
    
    参数：
        x: 输入张量，形状 [..., H]
        eps: 数值稳定项
    
    返回：
        归一化后的张量
    """
    # 获取特征维度
    normalized_shape = x.shape[-1]
    
    # 计算均值和方差（在最后一个维度上）
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    
    # 归一化
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # 可学习的缩放和偏移
    gamma = torch.ones(x.shape[-1])
    beta = torch.zeros(x.shape[-1])
    
    # 广播到所有维度
    gamma = gamma.view(*([1] * (len(x.shape) - 1)), -1)
    beta = beta.view(*([1] * (len(x.shape) - 1)), -1)
    
    return gamma * x_norm + beta

# 测试
x = torch.randn(4, 10, 256)
y = layer_norm(x)
print(f"Input mean: {x.mean(dim=-1).mean():.6f}, variance: {x.var(dim=-1).mean():.6f}")
print(f"Output mean: {y.mean(dim=-1).mean():.6f}, variance: {y.var(dim=-1).mean():.6f}")
```

### 3.3 Transformer 中的 Layer Norm

```python
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层（展示 Layer Norm 的使用）"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Pre-LN Transformer（现代常用）
        # 标准：残差连接在归一化之前 (Post-LN)
        # 现代：归一化在残差连接之前 (Pre-LN)
        
        # Pre-LN 风格（更稳定）
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out  # 残差连接
        
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  # 残差连接
        
        return x

# 测试
encoder = TransformerEncoderLayer(d_model=512, num_heads=8, d_ff=2048)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
out = encoder(x)
print(f"Input: {x.shape}, Output: {out.shape}")
```

---

## 4. 代码示例

### 4.1 不同归一化方法对比

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def compare_normalizations():
    """对比不同归一化方法"""
    
    torch.manual_seed(42)
    
    # 输入：batch=4, channel=8, height=4, width=4
    x = torch.randn(4, 8, 4, 4)
    
    results = {}
    
    # Batch Norm
    bn = nn.BatchNorm2d(8)
    results['Batch Norm'] = bn(x).clone()
    
    # Layer Norm（2D）
    ln = nn.LayerNorm(8)
    x_2d = x.permute(0, 2, 3, 1).reshape(-1, 8)  # [B*H*W, C]
    results['Layer Norm'] = ln(x_2d).reshape(4, 4, 4, 8).permute(0, 3, 1, 2)
    
    # Instance Norm
    in_ = nn.InstanceNorm2d(8)
    results['Instance Norm'] = in_(x)
    
    # Group Norm（4组）
    gn = nn.GroupNorm(num_groups=4, num_channels=8)
    results['Group Norm'] = gn(x)
    
    # 绘图对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, (name, out) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(out[0, 0].numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'{name}\nmean={out.mean():.4f}, std={out.std():.4f}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150)
    plt.show()
    
    # 打印统计信息
    print("\n各归一化方法的输出统计：")
    for name, out in results.items():
        print(f"  {name}: mean={out.mean().item():.4f}, std={out.std().item():.4f}")

compare_normalizations()
```

### 4.2 Pre-LN vs Post-LN Transformer

```python
import torch
import torch.nn as nn

class PostLNTransformerLayer(nn.Module):
    """Post-LN Transformer（原始架构）"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, x):
        # Post-LN: 归一化在残差之后
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)  # 先残差，后归一化
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class PreLNTransformerLayer(nn.Module):
    """Pre-LN Transformer（现代架构）"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, x):
        # Pre-LN: 归一化在残差之前
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # 先归一化，后残差
        
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x

# 梯度流对比
def test_gradient_flow():
    torch.manual_seed(42)
    
    post_ln = PostLNTransformerLayer(64, 4)
    pre_ln = PreLNTransformerLayer(64, 4)
    
    x = torch.randn(2, 8, 64)
    
    # 计算梯度
    out_post = post_ln(x)
    out_post.sum().backward()
    
    post_grads = [p.grad.abs().mean().item() for p in post_ln.parameters() if p.grad is not None]
    
    # 重置
    for p in post_ln.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    out_pre = pre_ln(x)
    out_pre.sum().backward()
    
    pre_grads = [p.grad.abs().mean().item() for p in pre_ln.parameters() if p.grad is not None]
    
    print("Post-LN 第一层梯度:", post_grads[0])
    print("Pre-LN 第一层梯度:", pre_grads[0])
    print("\nPre-LN 通常有更稳定的梯度流")

test_gradient_flow()
```

### 4.3 RMSNorm 实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMS Normalization
    论文：Root Mean Square Layer Normalization (2019)
    
    简化版：只计算 RMS，不减去均值
    """
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        # RMS = sqrt(E[x²])
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm

# 测试
rmsnorm = RMSNorm(256)
x = torch.randn(4, 10, 256)
y = rmsnorm(x)

print(f"RMSNorm Input: {x.shape}")
print(f"RMSNorm Output: {y.shape}")
print(f"Output RMS: {y.pow(2).mean(dim=-1).mean():.4f}")  # 应该接近1
```

---

## 5. 应用场景
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的应用场景相关内容]


---

## 6. 优缺点分析
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的优缺点分析相关内容]


---

## 7. 调库实现
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的调库实现相关内容]


---

## 8. 手工代码实现
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的手工代码实现相关内容]


---

## 9. 可视化与结果理解
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的可视化与结果理解相关内容]


---

## 10. 模型评估
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的模型评估相关内容]


---

## 11. 常见问题与易错点
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的常见问题与易错点相关内容]


---

## 12. 学习总结
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的学习总结相关内容]


---

## 13. 练习题与思考题
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的练习题与思考题相关内容]


---

## 14. 学习路径建议
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Layer归一化的学习路径建议相关内容]


---
