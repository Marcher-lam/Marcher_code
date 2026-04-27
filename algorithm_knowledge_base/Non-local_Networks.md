# Non-local Neural Networks 学习文档

> 捕获长距离依赖的非局部注意力模块——让卷积网络"一眼看到全局"。

## 1. 算法基础认知

### 一句话定义

Non-local Networks通过非局部操作（Non-local Operation）捕获特征图中任意两个位置之间的依赖关系，不受空间距离的限制。这是自注意力机制在视觉领域的先驱工作。

### 直觉类比

就像你在看一张照片时，可以瞬间"注意到"画面任何角落之间的关联——不需要像卷积那样逐层传递信息。Non-local让神经网络也能"一眼看到全局"——图像左上角的云和右下角的水面反射可能相关，即使它们相隔很远。

### 历史背景

- **2018年**：Wang等人在CVPR 2018提出Non-local Neural Networks
- **动机**：CNN的局部感受野限制了长距离依赖建模
- **后续**：成为视频理解、目标检测、语义分割的重要组件
- **影响**：奠定了自注意力在视觉领域的应用基础，直接影响了后来的Transformer应用

### 算法定位

Non-local是**长距离依赖建模模块**，即插即用，可嵌入任何CNN架构。

## 2. 核心原理

### 2.1 核心思想

卷积操作只关注局部邻域（3×3感受野），堆叠多层虽然可以扩大感受野，但效率低且优化困难。Non-local操作直接计算任意两个位置之间的关联，一次性捕获全局依赖。

### 2.2 非局部操作

$$y_i = \frac{1}{C(x)} \sum_{\forall j} f(x_i, x_j) \cdot g(x_j)$$

其中：
- $i$ 是输出位置，$j$ 是所有位置的索引
- $x_i$, $x_j$ 是输入特征向量
- $f(x_i, x_j)$ 是成对关系函数（衡量 $i$ 和 $j$ 的相关性）
- $g(x_j)$ 是变换函数（对 $x_j$ 做线性变换）
- $C(x)$ 是归一化因子

### 2.3 四种$f$函数的选择

| 方法 | $f(x_i, x_j)$ | 特点 |
|------|---------------|------|
| Gaussian | $e^{x_i^T x_j}$ | 无参数，softmax归一化 |
| Embedded Gaussian | $e^{\theta(x_i)^T \phi(x_j)}$ | 可学习的嵌入变换 |
| Dot product | $x_i^T x_j$ | 无归一化，无参数 |
| Concatenation | $\text{ReLU}(w^T[x_i; x_j])$ | 拼接后线性变换 |

### 2.4 残差连接

Non-local块包含残差连接，方便嵌入任何网络：

$$z_i = W_z y_i + x_i$$

残差连接使得即使在初始阶段（注意力权重不可靠），信息流也不会被破坏。

## 3. 数学公式与推导

### 3.1 Embedded Gaussian

最常用的实现使用Embedded Gaussian作为 $f$：

$$f(x_i, x_j) = e^{\theta(x_i)^T \phi(x_j)}$$

$$\theta(x_i) = W_\theta x_i, \quad \phi(x_j) = W_\phi x_j$$

归一化因子：

$$C(x) = \sum_{\forall j} f(x_i, x_j)$$

### 3.2 具体计算过程

对于输入特征 $x \in \mathbb{R}^{C \times H \times W}$：

1. **线性变换**：$\theta(x) = W_\theta x$，$\phi(x) = W_\phi x$，$g(x) = W_g x$
   - 输出通道：$C' = C/2$（降维以减少计算量）

2. **矩阵乘法**（计算注意力图）：
   $$\theta(x) \in \mathbb{R}^{N \times C'}, \quad \phi(x) \in \mathbb{R}^{C' \times N}$$
   其中 $N = H \times W$
   $$A = \text{softmax}(\theta(x) \cdot \phi(x)) \in \mathbb{R}^{N \times N}$$

3. **加权聚合**：
   $$y = A \cdot g(x)^T \in \mathbb{R}^{N \times C'}$$

4. **输出变换**：
   $$z = W_z y + x$$

### 3.3 复杂度分析

Non-local的计算复杂度为 $O(N^2 C')$，其中 $N = H \times W$。这是主要的计算瓶颈。

## 4. 训练过程讲解

### 4.1 嵌入方式

Non-local块可以嵌入到CNN的任何位置：
- **早期层**：捕获低层特征间的长距离关系
- **后期层**：细化高层语义特征
- **通常做法**：在ResNet的res4和res5阶段各加一个Non-local块

### 4.2 初始化

为了使Non-local块在训练初期不破坏预训练权重，$W_z$ 初始化为0（残差分支输出为0，$z = x$）。

### 4.3 训练细节
- 与主干网络联合训练
- 使用标准SGD或Adam
- 无需特殊正则化

## 5. 应用场景

1. **视频分类**：在时间维度上捕获帧间依赖（Non-local的原始动机）
2. **目标检测**：在检测器中建模目标间的关系
3. **语义分割**：捕获全局上下文信息，改善分割一致性
4. **图像生成**：在生成器中建模长距离依赖
5. **姿态估计**：建模关节间的空间依赖关系

## 6. 优缺点分析

### 优点
1. **长距离依赖**：直接捕获任意距离的关系
2. **即插即用**：可嵌入任意CNN，无需修改主干
3. **效果显著**：在视频分类上提升约2-3%

### 缺点
1. **计算量大**：$O(N^2)$ 计算量和内存占用
2. **显存消耗大**：需要保存 $N \times N$ 的注意力矩阵
3. **小输入上效果不明显**：空间尺寸较小时，局部感受野已足够

## 7. 调库实现

```python
"""
Non-local Neural Networks 的完整PyTorch实现

论文: "Non-local Neural Networks" (CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    """Non-local注意力块

    捕获特征图中任意两点之间的依赖关系。

    参数:
        in_channels: 输入通道数
        inter_channels: 中间通道数（降维用）
        mode: 相关性计算模式 ('gaussian', 'dot', 'concat')
        sub_sample: 是否对空间进行下采样（降低计算量）
        bn_layer: 是否使用BatchNorm
    """

    def __init__(self, in_channels, inter_channels=None, mode='gaussian',
                 sub_sample=False, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        self.mode = mode

        # 变换函数 g
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        # 如果使用BN，对g的输出做BN
        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.W_z = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        # 初始化W_z为0（残差分支初始为0）
        nn.init.constant_(self.W_z[-1].weight, 0) if bn_layer else \
            nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z[-1].bias, 0) if bn_layer else \
            nn.init.constant_(self.W_z.bias, 0)

        # 根据模式设置theta和phi
        if mode == 'gaussian':
            self.theta = None
            self.phi = None
        elif mode == 'embedded':
            self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
            self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        elif mode == 'dot':
            self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
            self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        elif mode == 'concat':
            self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
            self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
            self.concat_proj = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 空间下采样（可选，降低计算量）
        if sub_sample:
            self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        else:
            self.max_pool = None

    def forward(self, x):
        batch_size, C, H, W = x.shape
        N = H * W

        # ---- 步骤1: g变换 ----
        g_x = self.g(x)
        if self.max_pool is not None:
            g_x = self.max_pool(g_x)
        g_x = g_x.view(batch_size, self.inter_channels, -1)  # (B, C', N')

        # ---- 步骤2: 计算相关性 f ----
        if self.mode == 'gaussian':
            # 直接计算点积
            theta_x = x.view(batch_size, C, N)
            phi_x = x.view(batch_size, C, N)
        elif self.mode in ('embedded', 'dot'):
            theta_x = self.theta(x)
            phi_x = self.phi(x)
            if self.max_pool is not None:
                theta_x = self.max_pool(theta_x)
                phi_x = self.max_pool(phi_x)
            theta_x = theta_x.view(batch_size, self.inter_channels, -1)
            phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        elif self.mode == 'concat':
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        # ---- 步骤3: 注意力矩阵 ----
        if self.mode in ('gaussian', 'embedded', 'dot'):
            # theta: (B, C', N) -> (B, N, C')
            # phi: (B, C', N')
            # f: (B, N, N')
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

            if self.mode in ('gaussian', 'embedded'):
                f_div_C = F.softmax(f, dim=-1)
            else:
                f_div_C = f / (f.shape[-1] + 1e-8)
        elif self.mode == 'concat':
            # theta: (B, C', N, 1), phi: (B, C', 1, N')
            # f: (B, 1, N, N')
            f = self.concat_proj(torch.relu(theta_x + phi_x))
            f_div_C = F.softmax(f.view(batch_size, 1, -1), dim=-1)

        # ---- 步骤4: 加权聚合 ----
        y = torch.matmul(f_div_C, g_x.permute(0, 2, 1))  # (B, N, C')
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, H, W)

        # ---- 步骤5: 输出变换 + 残差连接 ----
        z = self.W_z(y) + x

        return z


class NonLocalResNetBlock(nn.Module):
    """在ResNet瓶胫块中嵌入Non-local"""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        # Non-local块嵌入在残差分支内
        self.non_local = NonLocalBlock(planes * 4, inter_channels=planes * 2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes * 4, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 4)
        ) if stride != 1 or in_planes != planes * 4 else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.non_local(out)  # Non-local在残差分支
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def demo():
    """Non-local模块演示"""
    # 测试不同模式
    for mode in ['gaussian', 'embedded', 'dot']:
        block = NonLocalBlock(in_channels=256, inter_channels=128, mode=mode)
        x = torch.randn(1, 256, 28, 28)
        out = block(x)
        print(f"模式 {mode:>10}: 输入 {x.shape} -> 输出 {out.shape}")

    # 带下采样版本
    block_sub = NonLocalBlock(in_channels=256, inter_channels=64, mode='embedded', sub_sample=True)
    x = torch.randn(1, 256, 56, 56)
    out = block_sub(x)
    print(f"下采样模式: 输入 {x.shape} -> 输出 {out.shape}")

    # 参数量
    block = NonLocalBlock(256, 128, 'embedded')
    print(f"Non-local参数量: {sum(p.numel() for p in block.parameters()):,}")


if __name__ == "__main__":
    demo()
```

## 8. 手工实现

```python
"""Non-local核心手工实现"""
import numpy as np

def non_local_handcraft(x, W_theta, W_phi, W_g, W_z):
    """手工Non-local前向"""
    B, C, H, W = x.shape
    N = H * W

    # theta: 线性变换 + reshape
    theta = (x.reshape(B, C, N).transpose(0, 2, 1) @ W_theta.T).transpose(0, 2, 1)

    # phi: 线性变换 + reshape
    phi = x.reshape(B, C, N).transpose(0, 2, 1) @ W_phi.T

    # g: 线性变换 + reshape
    g = x.reshape(B, C, N).transpose(0, 2, 1) @ W_g.T

    # 注意力: theta(B,C',N) @ phi(B,C',N)^T = (B,N,N)
    f = theta @ phi.transpose(0, 2, 1)
    f = f - f.max(axis=-1, keepdims=True)
    f_div_C = np.exp(f) / np.exp(f).sum(axis=-1, keepdims=True)

    # 加权: (B,N,N) @ (B,N,C') = (B,N,C')
    y = f_div_C @ g
    y = y.transpose(0, 2, 1).reshape(B, -1, H, W)
    z = (y.transpose(0, 2, 3, 1) @ W_z.T).transpose(0, 3, 1, 2) + x
    return z

def test():
    np.random.seed(42)
    B, C, H, W = 1, 64, 16, 16
    C_half = C // 2
    x = np.random.randn(B, C, H, W)
    W_t = np.random.randn(C_half, C) * 0.1
    W_p = np.random.randn(C_half, C) * 0.1
    W_g = np.random.randn(C_half, C) * 0.1
    W_z = np.random.randn(C, C_half) * 0.1
    out = non_local_handcraft(x, W_t, W_p, W_g, W_z)
    print(f"Non-local手工: {x.shape} -> {out.shape}")
    print(f"输出范围: [{out.min():.4f}, {out.max():.4f}]")

if __name__ == "__main__":
    test()
```

## 9. 可视化

```python
"""Non-local注意力图可视化"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_non_local_attention(attention_map, save_path='nl_attention.png'):
    """可视化Non-local注意力图
    
    参数:
        attention_map: (N, N) 注意力矩阵, N=H*W
    """
    N = attention_map.shape[0]
    H = W = int(np.sqrt(N))
    attn_2d = attention_map.reshape(H, W, H, W)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    query_points = [(H//4, W//4), (H//2, W//2), (3*H//4, 3*W//4),
                    (H//4, 3*W//4), (3*H//4, W//4), (H//2, W//4),
                    (W//4, H//2), (H//2, 3*W//4), (3*H//4, 3*W//4)]

    for idx, (qy, qx) in enumerate(query_points):
        ax = axes[idx // 3, idx % 3]
        attn_map = attn_2d[qy, qx]
        im = ax.imshow(attn_map, cmap='viridis')
        ax.set_title(f'Query ({qx},{qy})', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Non-local 注意力图: 不同Query点的关注区域', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"已保存到 {save_path}")

def simulate_attention():
    np.random.seed(42)
    N = 64
    attn = np.zeros((N, N))
    center_idx = N // 2
    for i in range(N):
        dist = abs(i - center_idx) / N
        attn[i, :] = np.exp(-dist) + np.random.randn(N) * 0.1
    attn = np.exp(attn)
    attn = attn / attn.sum(axis=1, keepdims=True)
    visualize_non_local_attention(attn)

if __name__ == "__main__":
    simulate_attention()
```

## 10. 模型评估

```python
"""Non-local模块效果评估"""
import torch.nn as nn

class SimpleCNNWithNL(nn.Module):
    def __init__(self, use_nl=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.nl = NonLocalBlock(64, 32, 'embedded') if use_nl else nn.Identity()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.nl(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

def evaluate_nl_impact():
    model_plain = SimpleCNNWithNL(use_nl=False)
    model_nl = SimpleCNNWithNL(use_nl=True)
    p_plain = sum(p.numel() for p in model_plain.parameters())
    p_nl = sum(p.numel() for p in model_nl.parameters())
    print(f"无NL: {p_plain:,} 参数")
    print(f"有NL: {p_nl:,} 参数 (+{p_nl-p_plain:,})")
    x = torch.randn(8, 3, 32, 32)
    out_plain = model_plain(x)
    out_nl = model_nl(x)
    print(f"无NL输出分布: mean={out_plain.mean():.4f}, std={out_plain.std():.4f}")
    print(f"有NL输出分布: mean={out_nl.mean():.4f}, std={out_nl.std():.4f}")

if __name__ == "__main__":
    evaluate_nl_impact()
```

## 11. 常见问题与易错点

**Q1: Non-local和Self-Attention的关系？**
Non-local = Self-Attention在视觉领域的"前身"。两者核心公式完全相同。Non-local提出时（2018）早于ViT（2020）。

**Q2: 如何降低Non-local的计算量？**
三种策略：(1) 降维中间通道 $C'$ (2) 空间下采样 $\phi$ 和 $g$ (3) 使用稀疏注意力或轴向注意力。

**Q3: Non-local应该放在网络的什么位置？**
通常放在深层（如res4/res5），因为深层特征图尺寸小，$N^2$ 计算量可接受，且深层需要全局信息。

**Q4: 为什么W_z初始化为0？**
梯度下降初始阶段，Non-local的注意力权重不可靠。残差分支为0确保输出 $z=x$，不影响预训练模型。网络逐渐学习调整权重。

## 12. 学习总结

- **核心贡献**：将非局部操作引入CNN，开创了视觉中的自注意力范式
- **技术本质**：$y_i = \frac{1}{C}\sum_j f(x_i, x_j)g(x_j)$，即注意力机制的通用形式
- **继承关系**：Self-Attention → Non-local → ViT → Transformer视觉全家桶
- **局限性**：$O(N^2)$ 复杂度限制了在高分辨率输入上的应用
- **后续改进**：GCNet（简化版）、CCNet（十字交叉注意力）、OCRNet（目标上下文）

## 13. 练习题

**基础题：**

1. 写出Non-local操作的通用公式，解释每个符号的含义。
> **答案：** $y_i = \frac{1}{C(x)}\sum_j f(x_i, x_j)g(x_j)$，$y_i$是输出，$x_i,x_j$是输入特征，$f$度量相关性，$g$做特征变换，$C$归一化。

2. Non-local操作为什么需要残差连接？
> **答案：** 保证训练稳定。初始阶段注意力权重不可靠，残差连接确保信息流不中断。

**进阶题：**

3. 比较Non-local和深度可分离卷积在捕获长依赖上的优劣。
> **答案：** Non-local直接建模任意两点关系，但$O(N^2)$复杂度。深度可分离卷积通过大核（如7×7）扩大量受野，复杂度线性，但无法建模真正的全局依赖。

4. 如何将Non-local扩展到3D（视频）？
> **答案：** 将2D空间扩展为3D时空：$x \in \mathbb{R}^{T \times C \times H \times W}$，在$T \times H \times W$维度上计算全连接注意力。

## 14. 学习路径

**前置：** CNN、注意力机制、ResNet
**平行：** SE-Net（通道注意力）、CBAM（混合注意力）、GCNet（简化Non-local）
**进阶：** ViT、Swin Transformer、Masked Autoencoder (MAE)