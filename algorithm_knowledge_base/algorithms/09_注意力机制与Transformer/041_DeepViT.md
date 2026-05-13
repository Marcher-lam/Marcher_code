# DeepViT (Deep Vision Transformer) 学习文档

## 1. 算法基础认知

### 1.1 算法要解决什么问题

直觉上，增加 Transformer 的深度应该能提升模型性能——深层网络能学习更复杂的特征表示。然而，实验发现，当 ViT 的深度增加到一定程度（如超过 16 层）后，模型性能不仅不再提升，反而会下降。

这是为什么？

研究人员发现，深层 ViT 存在一个被称为 **注意力崩溃（Attention Collapse）** 的问题：当 Transformer 加深时，不同层的注意力图（attention maps）变得越来越相似，最终几乎完全一致，丧失了多样性。这意味着深层的自注意力机制无法学到新的信息，深层网络退化成了浅层网络的重复堆叠。

DeepViT（由 Zhou 等人于 2021 年提出）的核心目标是：**如何避免深层 ViT 中的注意力崩溃，让 Transformer 的每一层都能学到多样化的注意力模式？**

### 1.2 核心思路概览

DeepViT 的答案是一个简单而优雅的机制——**Re-Attention（再注意力）**：

```
标准注意力: Attention(Q, K, V) = Softmax(QK^T/√d) V
Re-Attention: Re-Attention(attn) = λ · MLP(attn_reshaped) 的重排列
```

Re-Attention 在每层自注意力之后，对注意力图进行额外的线性变换，**重新生成**注意力权重。这种简单的操作有效地打破了注意力崩溃，增加了跨层注意力的多样性。

通过 Re-Attention，DeepViT 成功地将 ViT 的深度从 16 层扩展到了 32 层甚至更深，且性能持续提升。

### 1.3 整体架构

```
输入图像 → Patch Embedding → [Transformer 块 × N] → MLP Head
                                    ↓
                            Transformer 块内部:
                            LayerNorm → MHA → Re-Attention → LayerNorm → FFN
```

每个 Transformer 块在标准的 MHA（Multi-Head Attention）之后，额外插入一个 Re-Attention 模块，用于重新生成注意力权重。

## 2. 核心原理

### 2.1 注意力崩溃（Attention Collapse）

#### 什么是注意力崩溃

注意力崩溃是指随着 Transformer 深度增加，各层注意力图之间的相似度不断上升，最终几乎完全一致的现象。

设第 l 层的注意力图为 A^(l) ∈ ℝ^{H×N×N}（H 为注意力头数，N 为序列长度），注意力崩溃可以量化为相邻层注意力图之间的平均余弦相似度：

```
S(l) = cosine_similarity(A^(l), A^(l+1))
```

实验发现，对于标准 ViT：
- 浅层（1-6 层）：S(l) ≈ 0.3-0.5（注意力多样）
- 中层（7-12 层）：S(l) ≈ 0.7-0.9（注意力开始趋同）
- 深层（13+ 层）：S(l) ≈ 0.95+（注意力几乎完全一致）

#### 为什么注意力崩溃会发生？

注意力崩溃的根本原因在于 **softmax 函数的饱和特性**。

在自注意力中，注意力权重通过 softmax 计算：
```
A_ij = exp(Q_i · K_j / √d) / Σ_k exp(Q_i · K_k / √d)
```

当查询 Q 和键 K 的内积值较大时，softmax 的输出会趋近于 one-hot 分布（几乎所有注意力集中在一个 token 上）。当 Transformer 加深时，这种"过度集中"的趋势会逐步累积和放大，最终导致所有头的注意力图都变得非常相似。

数学上可以证明：在深层 Transformer 中，Q 和 K 的分布会趋向于某种稳定的不动点，使得不同层的注意力图收敛到相似的模式。

### 2.2 Re-Attention 机制

Re-Attention 的核心思想是：**在注意力计算之后，对注意力图进行可学习的重新变换，打破注意力崩溃的循环**。

具体流程：

1. **标准多头注意力**：计算注意力输出 O = Concat(head₁,...,head_H) · W_O
2. **注意力图重排列**：将多头注意力图 A ∈ ℝ^{H×N×N} 重排列为 A' ∈ ℝ^{N×H·N}
3. **线性变换**：通过可学习的 MLP 对 A' 进行变换，生成新的注意力图
4. **重新加权**：用新的注意力图对 V 进行加权求和

Re-Attention 的关键在于**引入了额外的可学习参数**来打破对称性。标准注意力中，注意力图完全由 Q 和 K 决定（二者来自同一组线性投影），而 Re-Attention 引入了独立于 Q、K 的变换矩阵，使注意力图有了新的自由度。

### 2.3 DeepViT 的整体设计

DeepViT 的完整 Transformer 块设计：

```
输入 x
    ↓
x' = LayerNorm(x)
    ↓
Q, K, V = x' · W_Q, x' · W_K, x' · W_V      # 生成 Q, K, V
    ↓
A = Softmax(QK^T/√d)                          # 注意力图 (H, N, N)
    ↓
A' = ReAttention(A)                            # Re-Attention 重新生成
    ↓
O = Concat(A'_1 · V_1, ..., A'_H · V_H) · W_O # 输出投影
    ↓
x = x + O                                      # 残差连接
    ↓
x = x + FFN(LayerNorm(x))                      # FFN + 残差
    ↓
输出
```

## 3. 数学公式与推导

### 3.1 标准多头注意力

设输入 x ∈ ℝ^{N×D}，多头注意力计算：

```
Q = x · W_Q ∈ ℝ^{N×d_k}      W_Q ∈ ℝ^{D×d_k}
K = x · W_K ∈ ℝ^{N×d_k}      W_K ∈ ℝ^{D×d_k}
V = x · W_V ∈ ℝ^{N×d_v}      W_V ∈ ℝ^{D×d_v}

对于第 h 头：
head_h = softmax(Q_h · K_h^T / √d_k) · V_h ∈ ℝ^{N×d_v}

拼接所有头：
MHA(x) = Concat(head_1, ..., head_H) · W_O ∈ ℝ^{N×D}
```

其中 H = D/d_k 为注意力头数。

### 3.2 Re-Attention 的数学定义

Re-Attention 对多头注意力图 A ∈ ℝ^{H×N×N} 进行变换。

**步骤 1：重排列**

将 A 从 (H, N, N) 重排列为 (N, H·N)：
```
A_flat = Reshape(A, (N, H·N))
```

或者等价地看作：
```
A_flat[i] = Concat(A[1][i,:], A[2][i,:], ..., A[H][i,:])
```
其中 A[h] 是第 h 个注意力头的注意力矩阵。

**步骤 2：线性变换**

通过一个可学习的 MLP（通常为单层线性层 + GELU 激活）：
```
A_new = MLP(A_flat) ∈ ℝ^{N×H·N}
```

MLP 的参数 W_re ∈ ℝ^{(H·N)×(H·N)} 和 b_re ∈ ℝ^{H·N} 是可学习的。

**步骤 3：重排列回原形状**

```
A' = Reshape(A_new, (H, N, N))
```

**步骤 4：Re-Attention 输出**

```
O_re = Concat(A'_1 · V_1, ..., A'_H · V_H) · W_O
```

### 3.3 简化的 Re-Attention 形式

原论文提出了一种更高效的实现方式，使用**注意力图的通道变换**：

```
O_re = λ · W_re · Reshape(A, (N, H·N)) · V
```

或者更简单地，将 Re-Attention 视为对注意力图的线性混合：

```
A'[h] = Σ_{i=1}^{H} α_{h,i} · A[i]  (h = 1,...,H)
```

其中 α_{h,i} 是可学习的混合系数。这相当于在注意力头之间进行信息的线性组合。

### 3.4 Re-Attention 的参数效率

Re-Attention 引入了额外的参数量：
- 当使用全连接层时：H²·N² 个参数（N 为序列长度，如 196）
- 当使用通道混合方式时：H² 个参数

通常使用后者（通道混合），因为它在参数效率和效果之间取得了更好的平衡。

### 3.5 注意力多样性的数学度量

论文使用以下指标度量注意力多样性：

**相邻层注意力余弦相似度**：
```
S_l = (1/H) · Σ_{h=1}^{H} cos(A^(l)_h, A^(l+1)_h)
```
其中 A^(l)_h 是第 l 层第 h 头的注意力图。

**头间注意力多样性**：
```
D_l = (2/(H(H-1))) · Σ_{i<j} (1 - cos(A^(l)_i, A^(l)_j))
```

通过 Re-Attention，DeepViT 成功地将 S_l 从 0.95+ 降至 0.3-0.5，D_l 从 0.1 以下提升至 0.5-0.7。

## 4. 训练过程讲解

### 4.1 训练配置

DeepViT 的训练配置与 DeiT（Data-efficient Image Transformers）相似：

- **优化器**：AdamW (β₁=0.9, β₂=0.999)
- **学习率调度**：Cosine annealing，初始学习率 5e-4
- **权重衰减**：0.05
- **热身阶段**：10 epoch 线性热身至目标学习率
- **批次大小**：1024
- **训练轮数**：300 epoch
- **标签平滑**：0.1
- **Dropout**：0.1
- **Stochastic Depth**：0.1（深层 Dropout 率更高）

### 4.2 数据增强

- RandomResizedCrop (224×224)
- RandomHorizontalFlip
- RandAugment (2, 15)
- Mixup (α=0.8)
- CutMix (α=1.0)
- ColorJitter

### 4.3 训练流程

```python
# 伪代码
for epoch in range(300):
    model.train()
    for images, labels in dataloader:
        # 数据增强
        if mixup_or_cutmix:
            images, labels_a, labels_b, lam = mixup_or_cutmix(images, labels)

        # 前向传播
        outputs = model(images)

        # 损失计算
        loss = cross_entropy(outputs, labels_a) * lam + \
               cross_entropy(outputs, labels_b) * (1 - lam)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 学习率更新
    scheduler.step()

    # 验证
    val_acc = validate(model, val_loader)
    print(f"Epoch {epoch}: val_acc = {val_acc:.2f}%")
```

### 4.4 蒸馏训练

DeepViT 也支持知识蒸馏：
- 教师模型：RegNetY-16GF 或预训练的 ViT-L
- 蒸馏损失：KL 散度 + 硬标签交叉熵
- 蒸馏温度：T = 3.0
- 蒸馏权重：0.5

## 5. 应用场景

### 5.1 图像分类

DeepViT 在 ImageNet 上取得了优异的性能：
- DeepViT-S：78.3% top-1（22M 参数）
- DeepViT-B：79.8% top-1（37M 参数）
- DeepViT-L：81.2% top-1（55M 参数）
- DeepViT-Deep（32 层）：81.8% top-1（48M 参数）

相比标准 ViT，DeepViT 在相同参数量下准确率提升约 1-2%。

### 5.2 注意力多样性分析

DeepViT 的一个核心应用是作为研究**注意力多样性**的框架。通过调整 Re-Attention 的强度，可以分析注意力多样性与模型性能之间的关系。

### 5.3 深层模型研究

DeepViT 证明了视觉 Transformer 可以像 CNN 一样通过增加深度来提升性能，为后续更深层 ViT 的研究（如 ViT-L、ViT-H）提供了理论基础。

## 6. 优缺点分析

### 6.1 优点

1. **解决了注意力崩溃问题**：Re-Attention 有效恢复了深层 Transfomer 的注意力多样性
2. **参数高效**：Re-Attention 仅增加少量参数（H²），计算开销小
3. **即插即用**：Re-Attention 可以作为插件，集成到任何 ViT 变体中
4. **深层模型可达**：使 ViT 可以扩展到 32 层甚至更深，性能持续提升

### 6.2 缺点

1. **仅缓解而非根治**：Re-Attention 缓解了注意力崩溃，但没有完全解决
2. **额外计算开销**：即使计算量小，Re-Attention 仍然增加了推理延迟
3. **超参数敏感**：Re-Attention 的变换方式（全连接 vs 通道混合）需要调优
4. **理论分析不足**：Re-Attention 为什么有效缺乏严格的理论证明

## 7. 调库实现

### 7.1 完整实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReAttention(nn.Module):
    """
    Re-Attention 模块：对多头注意力图进行重新变换
    打破深层 Transformer 中的注意力崩溃

    输入: 注意力图 A: (B, H, N, N)
    输出: 重新变换后的注意力图 A': (B, H, N, N)
    """
    def __init__(self, num_heads: int, seq_len: int, use_mlp: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.use_mlp = use_mlp

        if use_mlp:
            # 使用 MLP 对注意力图进行变换
            # 输入: (B, N, H*N)，输出: (B, N, H*N)
            self.re_attn = nn.Sequential(
                nn.Linear(num_heads * seq_len, num_heads * seq_len),
                nn.GELU(),
                nn.Linear(num_heads * seq_len, num_heads * seq_len),
            )
        else:
            # 使用简单的线性混合（通道混合方式）
            # 每个输出头是所有输入头的线性组合
            self.re_attn = nn.Parameter(
                torch.eye(num_heads) * 0.5 + 0.5 / num_heads
            )
            # 初始化为接近恒等映射但带有轻微混合

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        """
        attn: (B, H, N, N) - 多头注意力图
        返回: (B, H, N, N) - 重新变换后的注意力图
        """
        B, H, N, _ = attn.shape

        if self.use_mlp:
            # MLP 方式
            # 重排列: (B, H, N, N) -> (B, N, H*N)
            attn_flat = attn.permute(0, 2, 1, 3).reshape(B, N, H * N)
            # MLP 变换
            attn_new = self.re_attn(attn_flat)
            # 重排列回: (B, N, H*N) -> (B, H, N, N)
            attn_new = attn_new.reshape(B, N, H, N).permute(0, 2, 1, 3)
        else:
            # 通道混合方式
            # re_attn: (H, H) -> 每个输出头是输入头的加权组合
            # 注意力图 reshape: (B, H, N, N) -> (B, H, N*N)
            attn_flat = attn.reshape(B, H, N * N)
            # 头混合: (B, H, N*N) @ (H, H)^T -> (B, H, N*N)
            attn_mixed = attn_flat @ self.re_attn.T
            # reshape 回: (B, H, N, N)
            attn_new = attn_mixed.reshape(B, H, N, N)

        return attn_new


class DeepViTBlock(nn.Module):
    """
    DeepViT 的 Transformer 块
    在标准 MHA 之后添加 Re-Attention 模块
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_re_attn: bool = True,
    ):
        super().__init__()
        self.use_re_attn = use_re_attn

        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 多头注意力
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Re-Attention
        if use_re_attn:
            self.re_attn = ReAttention(num_heads, seq_len, use_mlp=False)

        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力 + Re-Attention
        residual = x
        x = self.norm1(x)

        # 标准多头注意力
        # nn.MultiheadAttention 返回 (attn_output, attn_weights)
        # 但我们无法直接从 nn.MultiheadAttention 获取注意力权重
        # 因此下面我们手动实现多头注意力以获取注意力图

        # 手动实现注意力以获取注意力图
        B, N, D = x.shape
        head_dim = D // num_heads

        # 生成 Q, K, V
        q = self.attn.in_proj_weight[:D, :]
        k = self.attn.in_proj_weight[D:2*D, :]
        v = self.attn.in_proj_weight[2*D:, :]

        Q = F.linear(x, q)  # (B, N, D)
        K = F.linear(x, k)  # (B, N, D)
        V = F.linear(x, v)  # (B, N, D)

        # 分头
        Q = Q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, N, Dh)
        K = K.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        # 注意力权重
        scale = head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale  # (B, H, N, N)
        attn_weights = F.softmax(attn, dim=-1)

        # Re-Attention
        if self.use_re_attn:
            attn_weights = self.re_attn(attn_weights)

        # 加权求和
        attn_out = attn_weights @ V  # (B, H, N, Dh)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        attn_out = self.attn.out_proj(attn_out)

        x = residual + attn_out

        # FFN + 残差
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class DeepViT(nn.Module):
    """
    DeepViT: Deep Vision Transformer
    使用 Re-Attention 解决深层注意力崩溃问题
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 384,
        depth: int = 16,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_re_attn: bool = True,
    ):
        super().__init__()

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.seq_len = num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # DeepViT blocks
        self.blocks = nn.ModuleList([
            DeepViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                seq_len=num_patches + 1,  # +1 for cls token
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_re_attn=use_re_attn,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # DeepViT blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x


def test_deepvit():
    """测试 DeepViT 前向传播"""
    model = DeepViT(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=384,
        depth=16,
        num_heads=6,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


if __name__ == "__main__":
    test_deepvit()
```

### 7.2 使用 timm 库调用

```python
import torch
from timm.models import create_model

# DeepViT 在 timm 中的名称
try:
    model = create_model(
        "deepvit_small_patch16_224",
        pretrained=True,
        num_classes=1000,
    )
    print(f"DeepViT-Small 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输出形状: {out.shape}")

except Exception as e:
    print(f"timm 中可能没有直接提供 DeepViT 模型: {e}")
    print("请使用上面的手工实现或升级 timm 版本")
```

## 8. 手工代码实现

### 8.1 注意力崩溃检测

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def compute_attention_similarity(model, x):
    """
    计算各层注意力图之间的余弦相似度
    用于检测注意力崩溃

    参数:
        model: DeepViT 模型
        x: 输入图像 (B, C, H, W)

    返回:
        similarities: 相邻层注意力图的余弦相似度列表
    """
    model.eval()
    attention_maps = []

    # Hook 函数：捕获每层的注意力图
    def get_attention(name):
        def hook(module, input, output):
            # 我们需要捕获手动实现的注意力图
            # 对于 DeepViTBlock，我们需要修改其 forward 以返回注意力图
            pass
        return hook

    # 这里简化处理，直接使用模型前向传播时存储的注意力图
    # 在实际使用中，需要修改模型以返回注意力图

    with torch.no_grad():
        _ = model(x)

    # 模拟数据（实际使用中替换为真实注意力图）
    num_layers = len(model.blocks)
    similarities = []

    for l in range(num_layers - 1):
        # A_l: (H, N, N), A_l1: (H, N, N)
        # 计算余弦相似度
        sim = np.random.uniform(0.3, 0.5)  # 模拟数据
        similarities.append(sim)

    return similarities


def analyze_attention_collapse():
    """
    对比标准 ViT 和 DeepViT 的注意力崩溃程度
    """
    # 创建标准 ViT（不使用 Re-Attention）
    vit_normal = DeepViT(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=384,
        depth=16,
        num_heads=6,
        use_re_attn=False,  # 不使用 Re-Attention
    )

    # 创建 DeepViT（使用 Re-Attention）
    vit_deep = DeepViT(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=384,
        depth=16,
        num_heads=6,
        use_re_attn=True,  # 使用 Re-Attention
    )

    x = torch.randn(1, 3, 224, 224)

    # 实际使用时，需要获取各层的注意力图进行对比
    print("标准 ViT: 期望相邻层注意力相似度较高 (>0.8)")
    print("DeepViT:  期望相邻层注意力相似度较低 (<0.5)")

    return vit_normal, vit_deep
```

### 8.2 Re-Attention 的变体实现

```python
class ReAttentionV2(nn.Module):
    """
    Re-Attention 的改进版本
    使用多个并行的线性变换，每头独立变换
    """
    def __init__(self, num_heads: int, seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len

        # 每头一个独立的线性变换
        # 每头的注意力图大小为 (N, N)
        self.head_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len, seq_len),
                nn.GELU(),
                nn.Linear(seq_len, seq_len),
            )
            for _ in range(num_heads)
        ])

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        """
        attn: (B, H, N, N)
        """
        B, H, N, _ = attn.shape
        outputs = []

        for h in range(H):
            # 第 h 头的注意力图: (B, N, N)
            head_attn = attn[:, h, :, :]
            # 独立变换
            head_out = self.head_transforms[h](head_attn)
            outputs.append(head_out.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class GatedReAttention(nn.Module):
    """
    门控 Re-Attention
    通过门控机制控制 Re-Attention 的强度
    """
    def __init__(self, num_heads: int, seq_len: int):
        super().__init__()
        self.num_heads = num_heads

        # 混合矩阵
        self.mix = nn.Parameter(torch.eye(num_heads) + 0.1 * torch.randn(num_heads, num_heads))

        # 门控参数
        self.gate = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 2.0)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        """
        attn: (B, H, N, N)
        使用门控控制 Re-Attention 与原始注意力的混合比例
        """
        # Re-Attention 变换
        B, H, N, _ = attn.shape
        attn_flat = attn.reshape(B, H, N * N)
        attn_mixed = attn_flat @ self.mix.T
        attn_new = attn_mixed.reshape(B, H, N, N)

        # 门控: 原始注意力 vs Re-Attention 的加权平均
        gate = torch.sigmoid(self.gate)
        attn_out = gate * attn_new + (1 - gate) * attn

        return attn_out
```

## 9. 可视化与结果理解

### 9.1 注意力崩溃可视化

```python
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_attention_collapse(normal_similarities, deep_similarities):
    """
    可视化注意力相似度对比
    """
    layers = np.arange(1, len(normal_similarities) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(layers, normal_similarities, 'r-o', label='Standard ViT (no Re-Attention)', linewidth=2)
    plt.plot(layers, deep_similarities, 'b-s', label='DeepViT (with Re-Attention)', linewidth=2)

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Cosine Similarity\n(between adjacent layers)', fontsize=12)
    plt.title('Attention Collapse: Layer-wise Attention Similarity', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)

    # 标注无注意力崩溃的区域
    plt.axhspan(0, 0.5, alpha=0.1, color='green', label='Diverse attention (healthy)')
    plt.axhspan(0.5, 1.0, alpha=0.1, color='red', label='Attention collapse')

    plt.tight_layout()
    plt.show()


def visualize_head_diversity_heatmap(model, x):
    """
    可视化多头注意力图的热力图
    对比标准 ViT 和 DeepViT 的头部多样性
    """
    # 实际使用中需要从模型提取注意力图
    # 这里创建示例数据
    num_heads = 6
    num_patches = 197  # 196 patches + 1 cls token

    # 模拟标准 ViT 的注意力图（头之间非常相似）
    vit_attn = np.zeros((num_heads, num_patches, num_patches))
    base_attn = np.random.rand(num_patches, num_patches)
    base_attn = base_attn / base_attn.sum(axis=-1, keepdims=True)

    for h in range(num_heads):
        vit_attn[h] = base_attn + 0.05 * np.random.randn(num_patches, num_patches)
        vit_attn[h] = np.abs(vit_attn[h])
        vit_attn[h] /= vit_attn[h].sum(axis=-1, keepdims=True)

    # 模拟 DeepViT 的注意力图（头之间多样化）
    deepvit_attn = np.zeros((num_heads, num_patches, num_patches))
    for h in range(num_heads):
        deepvit_attn[h] = np.random.rand(num_patches, num_patches)
        deepvit_attn[h] /= deepvit_attn[h].sum(axis=-1, keepdims=True)

    # 可视化 cls token 对其他 token 的注意力模式
    fig, axes = plt.subplots(2, num_heads, figsize=(4 * num_heads, 8))

    for h in range(num_heads):
        # 标准 ViT
        im1 = axes[0, h].imshow(vit_attn[h][0, 1:].reshape(14, 14), cmap='viridis')
        axes[0, h].set_title(f'ViT Head {h+1}')
        axes[0, h].axis('off')
        plt.colorbar(im1, ax=axes[0, h], fraction=0.046)

        # DeepViT
        im2 = axes[1, h].imshow(deepvit_attn[h][0, 1:].reshape(14, 14), cmap='viridis')
        axes[1, h].set_title(f'DeepViT Head {h+1}')
        axes[1, h].axis('off')
        plt.colorbar(im2, ax=axes[1, h], fraction=0.046)

    axes[0, 0].set_ylabel('Standard ViT', fontsize=12)
    axes[1, 0].set_ylabel('DeepViT', fontsize=12)
    plt.suptitle('Attention Map Comparison (cls→patch attention)', fontsize=14)
    plt.tight_layout()
    plt.show()
```

### 9.2 训练曲线可视化

```python
def plot_training_curves(vit_log, deepvit_log):
    """
    对比标准 ViT 和 DeepViT 的训练曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = np.arange(1, len(vit_log['val_acc']) + 1)

    # 准确率曲线
    axes[0].plot(epochs, vit_log['val_acc'], 'r-', label='Standard ViT', linewidth=2)
    axes[0].plot(epochs, deepvit_log['val_acc'], 'b-', label='DeepViT', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 损失曲线
    axes[1].plot(epochs, vit_log['val_loss'], 'r-', label='Standard ViT', linewidth=2)
    axes[1].plot(epochs, deepvit_log['val_loss'], 'b-', label='DeepViT', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


def evaluate_deepvit(model, dataloader, device):
    """
    评估 DeepViT 模型
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
```

### 10.2 注意力多样性指标

```python
def compute_attention_diversity(model, x):
    """
    计算注意力多样性指标
    """
    model.eval()
    all_attentions = []

    # Hook 收集注意力图
    def hook_fn(module, input, output):
        if hasattr(module, 're_attn'):
            pass  # 收集注意力图

    with torch.no_grad():
        _ = model(x)

    # 实际使用中需要从模型获取真实注意力图
    # 这里返回模拟指标
    diversity_metrics = {
        'mean_head_similarity': 0.35,  # 低=好（头之间有差异）
        'mean_layer_similarity': 0.42, # 低=好（层之间有差异）
        'attention_entropy': 2.5,      # 高=好（注意力分布更均匀）
    }

    return diversity_metrics
```

## 11. 常见问题与易错点

### 11.1 Re-Attention 的初始化

**问题**：Re-Attention 的初始值对训练效果有很大影响。

**原因**：如果初始化不当，Re-Attention 在训练初期可能完全破坏注意力图，导致训练不稳定。

**解决方法**：将 Re-Attention 的混合矩阵初始化为接近单位矩阵（identity matrix），使 Re-Attention 在训练初期接近恒等变换，然后逐步学习。

```python
# 正确的初始化方式
self.re_attn = nn.Parameter(torch.eye(num_heads) * 0.9 + 0.1 / num_heads)
```

### 11.2 注意力图的维度

**问题**：在实现 Re-Attention 时，注意力图的维度处理容易出错。

**正确维度**：
- 注意力图 A: (B, H, N, N)，其中 B = batch, H = num_heads, N = seq_len
- Re-Attention 的混合矩阵: (H, H)，用于在头之间混合信息

### 11.3 Re-Attention 的位置

**问题**：Re-Attention 应该放在注意力计算之前还是之后？

**答案**：放在注意力计算之后、加权求和之前。即：

```
A = Softmax(QK^T/√d)  →  A' = Re-Attention(A)  →  O = A'V
```

如果放在之前（对 Q 或 K 做变换），就变成了不同的操作。

### 11.4 序列长度的影响

**问题**：Re-Attention 的参数量与序列长度 N 相关（全连接版本）。

**解决方法**：使用通道混合版本（参数为 H×H），参数与序列长度无关，可以处理任意输入尺寸。

## 12. 学习总结

### 12.1 核心贡献

DeepViT 的主要贡献在于：

1. **发现并分析了注意力崩溃问题**：首次系统性地研究了 ViT 深层中的注意力坍缩现象
2. **提出了 Re-Attention 机制**：一个简单、高效、即插即用的解决方案
3. **验证了深层 ViT 的有效性**：证明了通过适当的机制设计，视觉 Transformer 可以像 CNN 一样受益于深度增加

### 12.2 关键洞察

- 注意力崩溃是深层 ViT 性能瓶颈的根本原因
- Re-Attention 通过引入额外的可学习自由度来打破注意力崩溃
- 有效的注意力多样性是深层 Transformer 性能的关键

### 12.3 与相关方法对比

| 方法 | 解决思路 | 参数开销 | 效果 |
|------|---------|---------|------|
| DeepViT (Re-Attention) | 变换注意力图 | 低 (H²) | 好 |
| Talk Heads | 头之间信息交换 | 中 (H·d) | 一般 |
| CSWin | 十字形窗口注意力 | 低 | 好（但改变架构） |
| Shunted Transformer | 多尺度注意力 | 中 | 好（但改变架构） |

## 13. 练习题与思考题

### 13.1 基础题

**题目 1**：什么是注意力崩溃？为什么它会在深层 ViT 中发生？

**答案**：注意力崩溃是指随着 Transformer 深度增加，各层的注意力图变得越来越相似，最终几乎完全一致的现象。其根本原因是 softmax 函数的饱和特性——在深层中，Q 和 K 的内积趋向于较大值，使 softmax 输出趋向于 one-hot 分布，导致注意力模式失去多样性。

**题目 2**：Re-Attention 的核心思想是什么？

**答案**：Re-Attention 在标准注意力计算之后，对注意力图进行可学习的线性变换，引入新的自由度来打破注意力崩溃。具体来说，它将多头注意力图进行重排列后通过 MLP 或线性混合，生成新的注意力权重。

**题目 3**：Re-Attention 引入了多少额外参数？为什么说它是参数高效的？

**答案**：在通道混合版本中，Re-Attention 只引入 H² 个参数（H 为注意力头数）。当 H=6 时，仅 36 个参数，几乎可以忽略不计。在 MLP 版本中，参数为 2(H·N)²，当 N=197, H=6 时约 2.8M 参数，但仍然相对高效。

### 13.2 进阶题

**题目 4**：Re-Attention 和 Talk Heads（Talking-Heads Attention）有何异同？

**答案**：
相同点：两者都在注意力头之间进行信息交换。
不同点：
- Talk Heads 在 softmax 之前进行头间混合（softmax(λ · QKT + bias)），而 Re-Attention 在 softmax 之后
- Talk Heads 对 logits 进行变换，Re-Attention 对概率进行变换
- Talk Heads 的目的是一般性的头间信息交换，Re-Attention 专门针对注意力崩溃

**题目 5**：为什么 Re-Attention 使用线性变换而不是非线性变换？

**答案**：实验发现线性混合已经足够有效。原因是注意力图本身已经是通过 softmax 归一化的概率分布，其多样性主要受限于头之间的相关性过高。线性混合就足以打破这种相关性——通过将不同头的注意力图线性组合，每个头可以接收到来自其他头的信息，从而产生差异化的注意力模式。引入非线性会额外增加计算开销，但收益不大。

### 13.3 思考题

**题目 6**：是否存在比 Re-Attention 更好的解决注意力崩溃的方法？请提出你的想法。

**答案**（开放题）：可能的方向包括：
1. **QK 解耦**：让 Q 和 K 的投影来自不同的输入表示，而非同一层输出
2. **注意力正则化**：在损失函数中加入注意力多样性的正则项
3. **随机注意力**：在深层中随机丢弃部分注意力连接
4. **动态深度**：根据注意力崩溃程度动态决定是否执行某层的注意力

**题目 7**：注意力崩溃是一个普遍现象还是 ViT 特有的？在 NLP Transformer 中是否存在？

**答案**：注意力崩溃在 NLP Transformer 中也存在，但在视觉 Transformer 中更为严重。原因是：
- 图像 patch 的语义相似度通常高于文本 token（相邻 patch 的像素高度相关），导致注意力更容易饱和
- 文本 Transformer 中，token 之间的语义差异更大（不同单词之间的差异远大于不同图像 patch）
- NLP Transformer 通常使用更大的模型（更多头数和隐层维度），这增加了注意力多样性

## 14. 学习路径建议

### 14.1 前置知识

1. **ViT (Vision Transformer)**：理解 patch embedding、class token、自注意力
2. **多头注意力机制**：理解 QKV 的计算和注意力头的拆分
3. **softmax 的特性**：理解 softmax 的饱和问题

### 14.2 学习步骤

1. **第一步**：精读原论文《DeepViT: Towards Deeper Vision Transformer》
2. **第二步**：理解注意力崩溃的概念和成因
3. **第三步**：实现 Re-Attention 模块，与标准 MHA 对比
4. **第四步**：在 CIFAR-10 上对比标准 ViT 和 DeepViT 的性能差异
5. **第五步**：可视化注意力图，观察 Re-Attention 的效果
6. **第六步**：尝试将 Re-Attention 集成到其他视觉 Transformer 中（如 Swin、PVT）

### 14.3 相关论文推荐

- DeepViT (Zhou et al., 2021)：原论文
- ViT (Dosovitskiy et al., 2020)：Vision Transformer 基础
- DeiT (Touvron et al., 2021)：数据高效的 ViT 训练
- Talking-Heads Attention (Shazeer et al., 2020)：头间信息交换
- CSWin Transformer (Dong et al., 2021)：十字形窗口注意力

### 14.4 实践建议

1. 先在 MNIST 或 CIFAR-10 等小数据集上验证 DeepViT
2. 对比不同深度（8, 16, 24, 32 层）下标准 ViT 和 DeepViT 的性能
3. 可视化注意力图，亲自验证注意力崩溃和 Re-Attention 的效果
4. 尝试将 Re-Attention 与其他注意力机制（如相对位置编码）结合
