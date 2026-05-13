# DANet双注意力网络 学习文档

> 空间与通道双重注意力并联——用自注意力同时建模空间和通道的长范围依赖。
>
> 来源线索：本节内容根据原书第2.3.3节"DANet：空间与通道并举的注意力模块"整理。

---

## 1. 算法基础认知

**一句话定义：** DANet（Dual Attention Network）由Fu Jun等人于2019年在CVPR上提出，以并联方式组织空间域和通道域两个自注意力子模块，分别建模特征间的空间依赖关系和通道依赖关系。

**核心思想：** 传统CNN的卷积操作受限于局部感受野，无法捕捉长距离依赖。DANet引入两个并行的自注意力模块：
1. **位置注意力模块（PAM）**：建模任意两个位置之间的空间依赖——无论距离多远，相似的特征互相增强
2. **通道注意力模块（CAM）**：建模任意两个通道之间的依赖——相关的通道互相增强

两个模块的输出通过求和融合，使每个位置的特征都聚合了全局空间上下文和全局通道上下文。

**为什么并行？** 空间注意力和通道注意力捕获的是不同维度的依赖关系。并行设计允许两者各自独立建模，然后通过求和融合实现互补，比串联更有效地保留各自的信息。

**DANet vs SENet vs Non-local：**

| 方法 | 注意力域 | 机制 | 特点 |
|------|---------|------|------|
| SENet | 通道 | 压缩-激励 | 轻量级、只建模通道 |
| Non-local | 空间 | 自注意力 | 重量级、只建模空间 |
| DANet | 空间+通道 | 双自注意力并联 | 全建模、效果好 |

---

## 2. 核心原理

### 2.1 位置注意力模块（PAM）

位置注意力计算特征图上任意两个位置之间的相互影响。

给定特征图 $A \in \mathbb{R}^{C \times H \times W}$，通过三个 $1\times1$ 卷积生成 $B, C, D \in \mathbb{R}^{C' \times H \times W}$：

$$
B = \text{Conv}_q(A), \quad C = \text{Conv}_k(A), \quad D = \text{Conv}_v(A)
$$

重塑为二维矩阵 $B, C, D \in \mathbb{R}^{C' \times N}$（$N = H \times W$）。

空间注意力图 $S \in \mathbb{R}^{N \times N}$：

$$
s_{ji} = \frac{\exp(B_i \cdot C_j)}{\sum_{i=1}^N \exp(B_i \cdot C_j)}
$$

$s_{ji}$ 表示位置 $i$ 对位置 $j$ 的影响程度。

输出：

$$
E_j = \alpha \sum_{i=1}^N s_{ji} \cdot D_i + A_j
$$

其中 $\alpha$ 是可学习尺度参数（初始化为0）。

### 2.2 通道注意力模块（CAM）

通道注意力计算任意两个通道之间的相互影响。

给定 $A \in \mathbb{R}^{C \times H \times W}$，重塑为 $A \in \mathbb{R}^{C \times N}$。

通道注意力图 $X \in \mathbb{R}^{C \times C}$：

$$
x_{ji} = \frac{\exp(A_i \cdot A_j)}{\sum_{i=1}^C \exp(A_i \cdot A_j)}
$$

$x_{ji}$ 表示通道 $i$ 对通道 $j$ 的影响程度。

输出：

$$
E_j = \beta \sum_{i=1}^C x_{ji} \cdot A_i + A_j
$$

其中 $\beta$ 是可学习尺度参数（初始化为0）。

### 2.3 特征融合

两个模块的输出通过逐元素求和融合：

$$
F_{out} = \text{Conv}(F_{PAM} + F_{CAM})
$$

然后通过分类头进行语义分割或分类。

---

## 3. 数学公式与推导

### 3.1 位置注意力的矩阵形式

$$
S = \text{softmax}_\text{row}(B^T \otimes C)
$$

其中 $B, C \in \mathbb{R}^{C' \times N}$，$\otimes$ 表示矩阵乘法。

$$
F_{PAM} = \alpha \cdot D \otimes S^T + A
$$

其中 $D \in \mathbb{R}^{C' \times N}$，$S \in \mathbb{R}^{N \times N}$。

### 3.2 通道注意力的矩阵形式

$$
X = \text{softmax}_\text{row}(A \otimes A^T)
$$

其中 $A \in \mathbb{R}^{C \times N}$。

$$
F_{CAM} = \beta \cdot X^T \otimes A + A
$$

### 3.3 计算复杂度分析

位置注意力：$O(N^2 \cdot C')$，其中 $N = H \times W$。
通道注意力：$O(C^2 \cdot N)$。

当 $H, W$ 很大时，PAM的计算量非常大（$O(H^2W^2)$）。通常需要降低特征图分辨率或减小通道数。

---

## 4. 训练过程讲解

DANet通过端到端反向传播训练。

**前向传播流程：**
1. backbone网络（如ResNet-101）提取特征 $A \in \mathbb{R}^{C \times H \times W}$
2. 位置注意力模块：
   - $1\times1$ 卷积生成 $B, C, D$（通道降维到 $C'=C/8$）
   - 矩阵乘法计算 $N \times N$ 注意力图
   - 注意力加权求和
   - 残差连接
3. 通道注意力模块：
   - 矩阵乘法计算 $C \times C$ 注意力图
   - 注意力加权求和
   - 残差连接
4. 融合：$F_{PAM} + F_{CAM} \to 1\times1$ 卷积
5. 分类/分割头

**反向传播：** 通过标准交叉熵损失回传。两个模块的 $\alpha, \beta$ 初始化为0，确保训练初期以原始特征为主，逐步加入注意力。

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 语义分割 | Cityscapes, PASCAL VOC 等场景分割 |
| 场景解析 | 需要长距离上下文的任务 |
| 目标检测 | 全局上下文增强检测特征 |
| 实例分割 | 长距离依赖建模 |
| 视频理解 | 时空维度的注意力 |
| 图像生成 | 全局一致的图像生成 |

---

## 6. 优缺点分析

**优点：**
- ✅ **空间+通道双建模**：捕获全维度的长距离依赖
- ✅ **残差连接**：训练稳定，易于优化
- ✅ **并联设计**：各自独立建模，互补性好
- ✅ **即插即用**：可插入任何骨干网络
- ✅ **效果好**：在语义分割任务上SOTA

**缺点：**
- ❌ **计算量大**：位置注意力 $O(H^2W^2)$ 复杂度
- ❌ **内存消耗大**：$N \times N$ 的注意力图消耗大量内存
- ❌ **高分辨率图像受限**：需要降采样特征图
- ❌ **两个模块可能冗余**：空间和通道注意力捕获的信息有重叠

---

## 7. 调库实现

```python
"""DANet双注意力网络 - PyTorch完整实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PositionAttention(nn.Module):
    """位置注意力模块 (PAM) - 建模空间依赖"""
    
    def __init__(self, in_channels, reduction=8):
        """
        参数:
            in_channels: 输入通道数
            reduction: 通道降维比
        """
        super().__init__()
        
        # 三个1x1卷积: query, key, value
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # 可学习尺度参数（初始化为0）
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # softmax
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        参数:
            x: 输入特征图 (batch, C, H, W)
        
        返回:
            out: 空间注意力增强特征 (batch, C, H, W)
        """
        batch, C, H, W = x.shape
        N = H * W
        
        # 生成 Q, K, V
        Q = self.query_conv(x).view(batch, -1, N).permute(0, 2, 1)  # (B, N, C')
        K = self.key_conv(x).view(batch, -1, N)                      # (B, C', N)
        V = self.value_conv(x).view(batch, -1, N)                    # (B, C, N)
        
        # 位置注意力图: S = softmax(Q @ K)
        energy = torch.bmm(Q, K)  # (B, N, N)
        attention = self.softmax(energy)
        
        # 注意力加权: E = V @ S^T
        out = torch.bmm(V, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch, C, H, W)
        
        # 残差连接: gamma * E + x
        out = self.gamma * out + x
        
        return out


class ChannelAttention(nn.Module):
    """通道注意力模块 (CAM) - 建模通道依赖"""
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        参数:
            x: 输入特征图 (batch, C, H, W)
        
        返回:
            out: 通道注意力增强特征 (batch, C, H, W)
        """
        batch, C, H, W = x.shape
        N = H * W
        
        # 重塑: (B, C, H, W) -> (B, C, N)
        A = x.view(batch, C, N)
        
        # 通道注意力图: X = softmax(A @ A^T)
        energy = torch.bmm(A, A.permute(0, 2, 1))  # (B, C, C)
        attention = self.softmax(energy)
        
        # 注意力加权: E = X^T @ A
        out = torch.bmm(attention.permute(0, 2, 1), A)  # (B, C, N)
        out = out.view(batch, C, H, W)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


class DANet(nn.Module):
    """双注意力网络 - 并联PAM和CAM"""
    
    def __init__(self, in_channels=2048, num_classes=21):
        """
        参数:
            in_channels: 输入特征图通道数
            num_classes: 分割类别数
        """
        super().__init__()
        
        # 双注意力模块（并联）
        self.position_attention = PositionAttention(in_channels, reduction=8)
        self.channel_attention = ChannelAttention(in_channels)
        
        # 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
        )
        
        # 分类头
        self.classifier = nn.Conv2d(in_channels, num_classes, 1)
    
    def forward(self, x):
        """
        参数:
            x: 输入特征图 (batch, C, H, W)
        
        返回:
            out: 分割结果 (batch, num_classes, H, W)
        """
        # 位置注意力
        feat_pa = self.position_attention(x)
        
        # 通道注意力
        feat_ca = self.channel_attention(x)
        
        # 并联融合
        feat = torch.cat([feat_pa, feat_ca], dim=1)
        feat = self.fusion(feat)
        
        # 分类
        out = self.classifier(feat)
        
        return out


class DANetWithBackbone(nn.Module):
    """完整DANet（含backbone）- 适配小尺寸输入"""
    
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        # 简化的backbone模拟
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        )
        
        self.danet = DANet(in_channels=512, num_classes=num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.danet(features)
        return out


def demo():
    """演示DANet前向传播"""
    # 测试位置注意力
    print("=== 位置注意力 ===")
    pam = PositionAttention(64, reduction=8)
    x = torch.randn(2, 64, 16, 16)
    y = pam(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")
    
    # 测试通道注意力
    print("\n=== 通道注意力 ===")
    cam = ChannelAttention(64)
    y = cam(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")
    
    # 测试完整DANet
    print("\n=== DANet ===")
    danet = DANet(in_channels=64, num_classes=21)
    y = danet(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")
    
    # 测试含backbone的模型
    print("\n=== DANet with Backbone ===")
    model = DANetWithBackbone(in_channels=3, num_classes=10)
    x2 = torch.randn(2, 3, 64, 64)
    y2 = model(x2)
    print(f"输入: {x2.shape} -> 输出: {y2.shape}")
    
    # 参数量
    total = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total:,}")
    
    # 验证PAM的 gamma 初始值
    print(f"PAM gamma 初始值: {pam.gamma.item():.4f}")
    print(f"CAM gamma 初始值: {cam.gamma.item():.4f}")


def visualize_attention():
    """可视化注意力图"""
    pam = PositionAttention(16, reduction=4)
    x = torch.randn(1, 16, 8, 8)
    
    with torch.no_grad():
        y = pam(x)
        
    # 获取注意力图
    batch, C, H, W = x.shape
    N = H * W
    Q = pam.query_conv(x).view(batch, -1, N).permute(0, 2, 1)
    K = pam.key_conv(x).view(batch, -1, N)
    energy = torch.bmm(Q, K)
    attn_map = F.softmax(energy, dim=-1)
    
    print(f"注意力图: {attn_map.shape}")
    print(f"注意力值范围: [{attn_map.min():.4f}, {attn_map.max():.4f}]")
    
    # 检查注意力是否集中在少数位置
    entropy = -(attn_map * torch.log(attn_map + 1e-8)).sum(dim=-1)
    print(f"平均注意力熵: {entropy.mean().item():.4f}")


if __name__ == "__main__":
    demo()
    print()
    visualize_attention()
```

---

## 8. 手工代码实现

```python
"""DANet - 手工自注意力实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dot_product_attention_manual(Q, K, V):
    """
    手工点积自注意力
    
    参数:
        Q: (batch, N, d_k)
        K: (batch, d_k, N)
        V: (batch, d_v, N)
    
    返回:
        out: (batch, d_v, N)
    """
    batch, N, d_k = Q.shape
    
    # Q @ K -> (batch, N, N)
    energy = torch.bmm(Q, K)
    
    # softmax
    max_energy = energy.max(dim=-1, keepdim=True)[0]
    exp = torch.exp(energy - max_energy)
    attention = exp / exp.sum(dim=-1, keepdim=True)
    
    # V @ attention^T -> (batch, d_v, N)
    out = torch.bmm(V, attention.permute(0, 2, 1))
    
    return out, attention


def position_attention_manual(x, W_q, W_k, W_v, gamma):
    """手工位置注意力"""
    batch, C, H, W = x.shape
    N = H * W
    
    # 线性投影
    Q = (x.view(batch, C, N).permute(0, 2, 1) @ W_q.t()).permute(0, 2, 1)
    Q = Q.view(batch, -1, N).permute(0, 2, 1)
    K = (x.view(batch, C, N).permute(0, 2, 1) @ W_k.t()).permute(0, 2, 1)
    V = (x.view(batch, C, N).permute(0, 2, 1) @ W_v.t()).permute(0, 2, 1)
    
    out, _ = dot_product_attention_manual(Q, K, V)
    out = out.view(batch, C, H, W)
    out = gamma * out + x
    
    return out


def channel_attention_manual(x, gamma):
    """手工通道注意力"""
    batch, C, H, W = x.shape
    N = H * W
    
    A = x.view(batch, C, N)
    _, attention = dot_product_attention_manual(
        A.permute(0, 2, 1), A, A
    )
    
    out = (attention.permute(0, 2, 1) @ A).view(batch, C, H, W)
    out = gamma * out + x
    
    return out


def test_danet_manual():
    batch, C, H, W = 1, 8, 4, 4
    x = torch.randn(batch, C, H, W)
    
    W_q = torch.randn(2, 8)
    W_k = torch.randn(2, 8)
    W_v = torch.randn(8, 8)
    gamma = nn.Parameter(torch.zeros(1))
    
    out_pa = position_attention_manual(x, W_q, W_k, W_v, gamma)
    out_ca = channel_attention_manual(x, gamma)
    
    print(f"手工PAM: {out_pa.shape}")
    print(f"手工CAM: {out_ca.shape}")
    print("测试通过")


if __name__ == "__main__":
    test_danet_manual()
```

---

## 9. 可视化与结果理解

### 9.1 位置注意力的空间效果

- 语义相似的区域相互增强：如"道路"区域彼此促进
- 长距离上下文传递：天空的蓝色信息传递到地面的水池区域

### 9.2 通道注意力的通道关系

- 相关通道相互增强：如"红色"通道和"交通标志"通道
- 某些通道成为"hub channel"与其他多个通道高度相关

### 9.3 两个注意力的互补性

- 位置注意力关注"哪里和哪里相关"
- 通道注意力关注"什么特征和什么特征相关"
- 两者结合 = 全维度的上下文建模

---

## 10. 模型评估

```python
"""DANet评估"""
import torch
import torch.nn as nn


def evaluate_danet_memory():
    """评估DANet的内存消耗"""
    batch, C, H, W = 1, 64, 32, 32
    x = torch.randn(batch, C, H, W)
    
    pam = PositionAttention(C, reduction=8)
    cam = ChannelAttention(C)
    
    # 计算PAM的注意力图大小
    N = H * W
    attn_memory = N * N * 4 / 1024 / 1024  # MB (float32)
    print(f"位置注意力图大小 (N={N}): {attn_memory:.2f} MB")
    
    # 计算CAM的注意力图大小
    attn_memory_c = C * C * 4 / 1024 / 1024
    print(f"通道注意力图大小 (C={C}): {attn_memory_c:.2f} MB")
    
    # 前向传播内存
    y_pam = pam(x)
    y_cam = cam(x)
    
    print(f"PAM输出: {y_pam.shape}")
    print(f"CAM输出: {y_cam.shape}")


if __name__ == "__main__":
    evaluate_danet_memory()
```

---

## 11. 常见问题与易错点

### Q1: DANet的PAM为什么使用 $1\times1$ 卷积降维？
**A:** 减少通道维度从 $C$ 到 $C/8$，降低 $QK^T$ 矩阵乘法的计算量。如果不降维，$C$ 通常为2048，$Q \in \mathbb{R}^{N \times 2048}$ 的计算量会非常大。

### Q2: 为什么 $\gamma$ 和 $\beta$ 初始化为0？
**A:** 训练初期注意力模块的预测不可靠，初始化为0相当于跳过注意力，以原始特征为主。随着训练逐渐增大，注意力模块逐步发挥作用，保证训练稳定性。

### Q3: PAM和CAM的计算复杂度差异？
**A:** PAM为 $O(N^2)$（$N$ 是点数），CAM为 $O(C^2)$（$C$ 是通道数）。当 $N > C$（高分辨率特征图）时PAM更耗时，当 $C > N$（深层窄特征图）时CAM更耗时。

### Q4: DANet与Non-local网络的区别？
**A:** Non-local只建模空间依赖（类似PAM），DANet同时建模空间和通道依赖。此外，DANet的两模块并联设计是其主要创新。

### Q5: DANet适用于实时场景吗？
**A:** 不适用。$N \times N$ 的注意力图对高分辨率特征图计算量极大（如 $64 \times 64$ 时需 $4096 \times 4096$ 矩阵）。通常需要下采样特征图到 $\leq 32 \times 32$。

---

## 12. 学习总结

**核心要点：**
1. 位置注意力：$N \times N$ 空间自注意力
2. 通道注意力：$C \times C$ 通道自注意力
3. 并联设计 + 求和融合
4. 可学习残差尺度参数 $\gamma, \beta$
5. 端到端训练

**DANet vs 其他注意力模块：**

| 模块 | 空间建模 | 通道建模 | 计算量 |
|------|---------|---------|--------|
| SENet | 无 | $C \times C$ GAP+FC | 低 |
| CBAM | $7\times7$ 卷积 | $C \times C$ GAP+FC | 低 |
| Non-local | $N \times N$ | 无 | 高 |
| DANet | $N \times N$ | $C \times C$ | 高 |

---

## 13. 练习题与思考题

### 基础题

**1.** DANet中PAM的注意力图 $S$ 的维度是多少？

<details>
<summary>答案</summary>
$S \in \mathbb{R}^{N \times N}$，其中 $N = H \times W$ 是特征图的像素数。$s_{ji}$ 表示位置 $i$ 对位置 $j$ 的注意力权重。
</details>

**2.** CAM中使用 $A \cdot A^T$ 的意义是什么？

<details>
<summary>答案</summary>
$A \in \mathbb{R}^{C \times N}$，$A \cdot A^T \in \mathbb{R}^{C \times C}$ 是通道间的相似度矩阵（非正式协方差），$(A A^T)_{ij}$ 表示通道 $i$ 和通道 $j$ 在所有空间位置上的响应相似度。
</details>

**3.** 为什么PAM要使用3个不同的 $1\times1$ 卷积？

<details>
<summary>答案</summary>
遵循自注意力机制的QKV设计：Query（查询位置的特征）、Key（被查询位置的特征）、Value（用于加权求和的值特征）。Q和K计算注意力权重，V提供加权的内容。
</details>

### 进阶题

**4.** 推导：如果 $\gamma = 0$ 且 $\beta = 0$，DANet的输出是什么？

<details>
<summary>答案</summary>
此时 $F_{PAM} = x$，$F_{CAM} = x$，融合后为两个 $x$ 的拼接 $2x$（$2C$ 通道）。卷积后还原为 $C$ 通道。实际上等于一个简单的 $1\times1$ 卷积操作，没有任何注意力效果。这验证了 $\gamma, \beta$ 初始化为0的合理性。
</details>

**5.** 如何改进DANet使其适用于高分辨率图像？

<details>
<summary>答案</summary>
(1) 金字塔注意力：在不同分辨率的特征图上分别计算后融合；(2) 稀疏注意力：只计算局部区域的自注意力；(3) 轴向注意力：将 $N \times N$ 分解为行注意力和列注意力 $(H \times H + W \times W)$；(4) 使用Cross-attention替代Self-attention。
</details>

**6.** DANet的两个注意力模块是否存在冗余？如何量化？

<details>
<summary>答案</summary>
存在一定程度冗余。可以计算PAM和CAM输出的互信息或余弦相似度来量化。如果高度冗余，可以只用其中一个。实验中两者联合使用性能最优，说明冗余小于互补。
</details>

---

## 14. 学习路径建议

### 预备知识
- 自注意力机制（Scaled Dot-Product Attention）
- Non-local网络
- SENet压缩-激励模块
- 语义分割基本概念

### 进阶方向
1. **DANet -> CCNet**：Criss-Cross注意力，降低复杂度
2. **DANet -> ANN**：Asymmetric Non-local，进一步优化
3. **DANet -> OCRNet**：目标上下文表示
4. **DANet -> SegFormer**：基于Transformer的分割架构

### 推荐阅读
- Fu et al. "Dual Attention Network for Scene Segmentation." CVPR 2019.
- Wang et al. "Non-local Neural Networks." CVPR 2018.
- Hu et al. "Squeeze-and-Excitation Networks." CVPR 2018.
- Huang et al. "CCNet: Criss-Cross Attention for Semantic Segmentation." ICCV 2019.

### 项目实践
1. 在Cityscapes数据集上实现DANet进行语义分割
2. 比较PAM单独、CAM单独和PAM+CAM的效果差异
3. 用DANet模块替换ResNet-50中的3x3卷积，观察性能变化
4. 分析不同 $\gamma, \beta$ 初始值对训练的影响
