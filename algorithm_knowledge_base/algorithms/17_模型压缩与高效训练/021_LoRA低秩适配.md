# LoRA 低秩适配 学习文档

> 来源线索：本节内容根据原书中关于"LoRA低秩适配"（第8章 8.2.2节）的相关章节整理、扩展与教学化改写。

> 冻结大模型，只训练极少量参数——低秩分解让微调成本降低1000倍。

## 1. 算法基础认知

**一句话定义**：LoRA通过在预训练权重矩阵旁添加低秩分解矩阵来实现参数高效微调。

**直觉类比**：想象一本已经写好的百科全书（预训练模型）。传统微调是修改全书内容（成本极高）。LoRA则是在每页旁边贴一张小纸条（低秩矩阵），只写上修改内容。使用时，原书内容加上小纸条的修改就是最终答案。训练时只更新小纸条，原书不变。

**历史背景**：LoRA（Low-Rank Adaptation）由Hu等人在2021年提出，迅速成为大模型微调的主流方法。其核心洞察是：预训练模型的权重变化具有低秩特性，因此可以用低秩矩阵近似。

**算法定位**：深度学习 / 参数高效微调 / 低秩分解。是PEFT（Parameter-Efficient Fine-Tuning）的经典方法。

**前置知识**：
- 矩阵的秩和低秩分解
- 预训练-微调范式
- 线性代数基础

## 2. 核心原理

### 核心思想

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA将权重的变化量 $\Delta W$ 参数化为两个低秩矩阵的乘积：

$$\Delta W = B \cdot A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$

其中 $r \ll \min(d, k)$ 是远小于原矩阵维度的秩。

### 工作流程

1. 冻结原始权重 $W_0$
2. 初始化低秩矩阵：$A$ 使用随机高斯初始化，$B$ 初始化为零
3. 前向传播：$h = W_0 x + BAx = (W_0 + BA)x$
4. 训练时只更新 $A$ 和 $B$
5. 微调完成后，可以将 $\Delta W = BA$ 合并回 $W_0$

### 关键概念

- **秩 $r$**：控制LoRA的表达能力。$r$ 越大表达力越强但参数越多。通常 $r=4, 8, 16, 64$
- **缩放因子 $\alpha$**：$\Delta W = \frac{\alpha}{r} BA$，控制LoRA的贡献比例
- **目标模块**：通常应用于Q、K、V投影矩阵，也可以应用于FFN层
- **零初始化**：$B$ 初始化为零，确保训练开始时 $\Delta W = 0$，不破坏预训练权重

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $W_0$ | 预训练权重 | $(d, k)$ |
| $B$ | LoRA下投影 | $(d, r)$ |
| $A$ | LoRA上投影 | $(r, k)$ |
| $r$ | LoRA秩 | $\ll d, k$ |
| $\alpha$ | 缩放因子 | 标量 |

### LoRA前向传播

$$h = W_0 x + \frac{\alpha}{r} B A x$$

### 参数量对比

- 原始参数：$d \times k$
- LoRA参数：$d \times r + r \times k = r(d + k)$
- 压缩比：$\frac{d \times k}{r(d + k)}$

例如：$d = k = 4096, r = 8$：
- 原始：$4096 \times 4096 = 16.7M$
- LoRA：$8 \times (4096 + 4096) = 65.5K$
- 压缩比：$256\times$

### 梯度分析

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T \frac{\partial \mathcal{L}}{\partial h} x^T$$

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^T$$

$W_0$ 被冻结，不计算梯度。

## 4. 训练过程讲解

### 数据预处理

与标准微调相同：根据下游任务准备数据（分类数据、指令数据等）。

### 参数初始化

- $A$：使用Kaiming均匀初始化（或正态分布 $N(0, \sigma^2)$）
- $B$：初始化为零矩阵，确保初始时 $\Delta W = BA = 0$
- $W_0$：冻结，不更新

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| 秩 $r$ | LoRA表达能力 | 4-64 | 8 |
| $\alpha$ | 缩放因子 | 8-32 | 16 |
| 目标模块 | 应用LoRA的层 | Q/K/V/O | Q, V |
| dropout | LoRA内dropout | 0-0.1 | 0 |

## 5. 应用场景

1. **大语言模型微调**：对LLaMA、DeepSeek等模型进行指令微调或领域适配，只需训练原参数量0.1%-1%的参数。

2. **多模态模型适配**：对CLIP等模型进行下游任务适配，如图像描述生成。

3. **个性化生成**：对Stable Diffusion进行风格适配，训练LoRA权重来控制生成风格。

4. **多任务适配**：为不同任务训练不同的LoRA权重，共享基础模型。

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 训练参数极少（<1%原始参数） | 秩太小时表达力受限 |
| 不增加推理延迟（可合并回原权重） | 需要选择合适的秩r和目标模块 |
| 适合多任务：共享基础模型，切换LoRA | 不适合需要大幅修改模型的场景 |
| 节省存储（只保存LoRA权重） | 某些复杂任务可能不如全量微调 |

**与其他微调方法对比**：

| 方法 | 可训练参数 | 推理开销 | 效果 |
|------|-----------|---------|------|
| 全量微调 | 100% | 无 | 最优 |
| LoRA | ~0.5% | 无（可合并） | 接近全量 |
| Adapter | ~2% | 有（额外层） | 接近全量 |
| Prompt Tuning | ~0.01% | 有（额外token） | 较弱 |

## 7. 调库实现

```python
"""使用 PEFT 库实现 LoRA 微调"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA线性层实现"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 原始线性层（冻结）
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # LoRA低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        # 原始变换 + LoRA调整
        original = nn.functional.linear(x, self.weight, self.bias)
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original + lora
    
    def merge_weights(self):
        """将LoRA权重合并回原始权重"""
        self.weight.data += (self.lora_B @ self.lora_A * self.scaling).data
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()


class LoRAAttention(nn.Module):
    """在Q、V投影上应用LoRA的多头注意力"""
    
    def __init__(self, d_model, num_heads, rank=8, alpha=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # 使用LoRA的Q、V投影
        self.q_proj = LoRALinear(d_model, d_model, rank, alpha)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # K不使用LoRA
        self.v_proj = LoRALinear(d_model, d_model, rank, alpha)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    d_model = 256
    
    # 模拟预训练权重
    lora_linear = LoRALinear(d_model, d_model, rank=8, alpha=16)
    lora_linear.weight.data = torch.randn_like(lora_linear.weight) * 0.02  # 模拟预训练权重
    
    x = torch.randn(2, 10, d_model)
    
    print("=== LoRA 测试 ===")
    
    # 初始时LoRA贡献为零（B初始化为零）
    out = lora_linear(x)
    original = nn.functional.linear(x, lora_linear.weight, lora_linear.bias)
    diff = (out - original).abs().max()
    print(f"初始LoRA偏移: {diff:.10f} (应为0)")
    
    # 参数量对比
    total = lora_linear.weight.numel() + lora_linear.bias.numel()
    lora_params = lora_linear.lora_A.numel() + lora_linear.lora_B.numel()
    print(f"\n原始参数: {total}")
    print(f"LoRA参数: {lora_params}")
    print(f"压缩比: {total/lora_params:.1f}x")
    print(f"LoRA参数占比: {lora_params/(total+lora_params)*100:.2f}%")
    
    # 训练测试
    target = torch.randn_like(out)
    loss = nn.MSELoss()(out, target)
    loss.backward()
    print(f"\n损失: {loss.item():.4f}")
    print(f"LoRA A梯度范数: {lora_linear.lora_A.grad.norm():.4f}")
    print(f"LoRA B梯度范数: {lora_linear.lora_B.grad.norm():.4f}")
    print(f"原始权重梯度: {lora_linear.weight.grad} (应为None, 因为冻结)")
```

## 8. 手工代码实现

```python
"""从零实现LoRA微调流程"""
import torch
import torch.nn as nn
import math


class SimpleTransformerLayer(nn.Module):
    """简单的Transformer层（用于演示LoRA微调）"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.scale = math.sqrt(self.d_head)
    
    def forward(self, x):
        batch, seq, d = x.shape
        
        # 自注意力
        Q = self.q_proj(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        
        attn = torch.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(batch, seq, d)
        x = self.norm1(x + self.out_proj(out))
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


def apply_lora_to_model(model, rank=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    """手动为模型添加LoRA
    
    核心思路: 找到目标线性层, 用LoRA线性层替换
    """
    lora_params = []
    
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            # 获取原始权重
            d_out, d_in = module.weight.shape
            
            # 创建LoRA参数
            lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
            lora_B = nn.Parameter(torch.zeros(d_out, rank))
            
            # 存储到模块上
            module.lora_A = lora_A
            module.lora_B = lora_B
            module.lora_scaling = alpha / rank
            
            # 冻结原始权重
            module.weight.requires_grad = False
            
            # 修改forward方法
            original_forward = module.forward
            
            def make_lora_forward(orig_fwd, scaling):
                def lora_forward(x):
                    return orig_fwd(x) + x @ lora_A.T @ lora_B.T * scaling
                return lora_forward
            
            module.forward = make_lora_forward(original_forward, module.lora_scaling)
            lora_params.extend([lora_A, lora_B])
    
    return lora_params


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 创建"预训练"模型
    model = SimpleTransformerLayer(d_model=128, num_heads=4)
    
    print("=== LoRA微调演示 ===")
    
    # 统计原始参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"原始参数量: {total_params:,}")
    
    # 应用LoRA
    lora_params = apply_lora_to_model(model, rank=8, alpha=16)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"LoRA参数量: {trainable_params:,}")
    print(f"训练比例: {trainable_params/total_params*100:.2f}%")
    
    # 训练
    optimizer = torch.optim.Adam(lora_params, lr=1e-3)
    x = torch.randn(2, 10, 128)
    target = torch.randn(2, 10, 128)
    
    print("\n微调过程:")
    for step in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        optimizer.step()
        print(f"  Step {step+1}: loss = {loss.item():.4f}")
```

## 9. 可视化与结果理解

```python
"""LoRA可视化"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 不同秩r的参数量对比
d = 4096
ranks = [1, 2, 4, 8, 16, 32, 64, 128]
original = d * d
lora_params = [r * (d + d) for r in ranks]
ratios = [p / original * 100 for p in lora_params]

axes[0].bar(range(len(ranks)), ratios, color='#3498db', edgecolor='black')
axes[0].set_xticks(range(len(ranks)))
axes[0].set_xticklabels([str(r) for r in ranks])
axes[0].set_title(f'LoRA参数占原始参数比例 (d={d})', fontsize=13)
axes[0].set_xlabel('秩 r')
axes[0].set_ylabel('参数占比 (%)')
for i, v in enumerate(ratios):
    axes[0].text(i, v + 0.1, f'{v:.2f}%', ha='center', fontsize=8)

# 图2: 不同r下的微调效果
ranks_perf = [1, 2, 4, 8, 16, 32, 64]
# 模拟性能数据（参考公开实验）
full_finetune = 92.0
lora_perf = [85.2, 87.8, 89.5, 91.2, 91.8, 91.9, 92.0]

axes[1].plot(ranks_perf, lora_perf, 'o-', color='#2ecc71', linewidth=2, label='LoRA')
axes[1].axhline(y=full_finetune, color='red', linestyle='--', label='全量微调')
axes[1].set_title('不同秩r下的微调性能', fontsize=13)
axes[1].set_xlabel('秩 r')
axes[1].set_ylabel('准确率 (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图3: LoRA低秩结构示意
ax = axes[2]
ax.text(0.5, 0.8, 'W₀', fontsize=20, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black'))
ax.text(0.5, 0.5, '+', fontsize=20, ha='center')
ax.text(0.25, 0.25, 'B', fontsize=16, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
ax.text(0.45, 0.25, '×', fontsize=16, ha='center')
ax.text(0.65, 0.25, 'A', fontsize=16, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

# 标注维度
ax.text(0.5, 0.72, f'd×k = {d}×{d}', fontsize=10, ha='center', color='blue')
ax.text(0.25, 0.15, f'd×r', fontsize=9, ha='center', color='orange')
ax.text(0.65, 0.15, f'r×k', fontsize=9, ha='center', color='orange')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('LoRA: W₀ + BA', fontsize=13)
ax.axis('off')

plt.tight_layout()
plt.savefig('lora_viz.png', dpi=100)
plt.show()

print("图1解读: 秩r=8时参数仅占0.39%, r=64时也仅占3.1%")
print("图2解读: r=8-16时性能接近全量微调, 继续增大r收益递减")
print("图3解读: LoRA将大矩阵分解为两个小矩阵的乘积")
```

## 10. 模型评估

```python
"""评估LoRA微调效果"""
def evaluate_lora(model, val_loader):
    """评估LoRA微调后的模型"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
    
    avg_loss = total_loss / total_samples
    print(f"验证损失: {avg_loss:.4f}")
    return avg_loss

def merge_and_evaluate(model):
    """合并LoRA权重后评估（验证合并是否正确）"""
    model.eval()
    
    # 合并前
    x_test = torch.randn(1, 10, 128)
    with torch.no_grad():
        out_before = model(x_test)
    
    # 合并权重
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            module.weight.data += (module.lora_B @ module.lora_A * module.lora_scaling).data
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
    
    # 合并后
    with torch.no_grad():
        out_after = model(x_test)
    
    diff = (out_before - out_after).abs().max()
    print(f"合并前后差异: {diff:.10f} (应≈0)")
```

## 11. 常见问题与易错点

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 任务数据太少 | LoRA过拟合 | 低秩也足以在小数据上过拟合 | 增加正则化，或使用更小的r |

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| B未零初始化 | 训练初期性能下降 | 初始LoRA偏移破坏预训练权重 | 确保B初始化为零 |
| 目标模块选择 | 性能不如预期 | LoRA没应用到关键层 | 对Q、K、V、O都应用LoRA |

### 调参层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 秩r太小 | 精度不足 | 低秩不足以表达任务差异 | 增大r到16或32 |
| alpha/r比例不当 | 收敛慢或不稳定 | 缩放因子不合适 | alpha通常设为2×r |

## 12. 学习总结

LoRA的核心公式：

$$h = W_0 x + \frac{\alpha}{r} BAx$$

其中 $W_0 \in \mathbb{R}^{d \times k}$ 冻结，$B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$ 可训练，$r \ll d, k$。

LoRA的关键价值：以不到1%的参数量实现接近全量微调的效果，且可以合并回原权重不增加推理开销。

## 13. 练习题与思考题

### 基础题1：参数量计算

一个注意力层的Q投影矩阵形状为 (4096, 4096)。使用LoRA，秩r=16。计算原始参数量、LoRA参数量和压缩比。

**参考答案**：
- 原始参数 = 4096 × 4096 = 16,777,216
- LoRA参数 = 4096 × 16 + 16 × 4096 = 65,536 + 65,536 = 131,072
- 压缩比 = 16,777,216 / 131,072 = 128倍

### 基础题2：零初始化的作用

为什么 $B$ 初始化为零而不是随机初始化？

**参考答案**：
$B$ 初始化为零确保 $\Delta W = BA = 0$（零矩阵乘以任何矩阵都是零矩阵）。这意味着训练开始时，LoRA不改变模型行为，预训练权重的影响完全保留。如果随机初始化，训练开始时模型输出会被大幅扰动，可能破坏预训练知识。

### 进阶题：LoRA与QLoRA

QLoRA在LoRA基础上增加了4-bit量化。解释为什么量化基础模型不影响LoRA的训练效果。

**参考答案**：
QLoRA将基础模型权重 $W_0$ 量化到4-bit以节省显存，但LoRA计算时使用反量化（dequantize）到高精度：
$$h = \text{dequant}(W_0^{4bit}) x + \frac{\alpha}{r} BAx$$
LoRA本身（$A, B$）保持高精度（如BF16）。量化只影响 $W_0$ 的存储和前向传播，不影响梯度的精度。因此QLoRA几乎不损失精度。

### 开放思考题

LoRA假设权重变化具有低秩特性。什么情况下这个假设不成立？此时应该如何微调？

**参考思路**：
低秩假设在以下情况可能不成立：
1. **领域跨度大**：从通用文本到高度专业的代码生成，需要大幅修改权重
2. **多任务冲突**：同时适配多个差异很大的任务
3. **新语言/新模态**：添加模型从未见过的新语言或新模态

解决方案：
- 增大秩r（如r=256）
- 结合Adapter层
- 分阶段微调（先LoRA热身，再全量微调部分层）
- 使用DoRA（将权重分解为方向和幅度，分别用LoRA适配）

## 14. 学习路径建议

### 前置知识
- 矩阵的低秩分解
- 预训练-微调范式
- Transformer架构

### 平行学习
- PEFT其他方法（Adapter、Prefix-Tuning）
- 量化技术（INT8/INT4量化）

### 进阶方向
- QLoRA（量化 + LoRA）
- DoRA（方向和幅度分解）
- LoRA在多模态模型中的应用

### 推荐资源
1. **论文**：LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
2. **论文**：QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
3. **库**：Hugging Face PEFT库 — LoRA的工业级实现
