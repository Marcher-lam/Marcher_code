# ViViT (Video Vision Transformer) 视频Vision Transformer 学习文档

> ViViT是将Transformer架构应用于视频理解任务的模型，通过时空建模实现视频分类、动作识别等任务

---

## 1. 算法基础认知

### 1.1 一句话定义

**ViViT (Video Vision Transformer)** 是一种将Transformer架构从图像扩展到视频领域的深度学习模型，通过在时序和空间维度同时进行自注意力建模，实现对视频的完整理解，是Video Transformers的里程碑工作。

### 1.2 直觉类比

想象你看一部电影：**传统CNN** 就像每次只看一帧的画面；而 **ViViT** 就像你不仅在分析每帧的空间内容（这个人是谁、在哪里），还在分析帧与帧之间的时间关系（他做了什么动作、表情如何变化）。ViViT 将"看图"升级为"看电影"，让模型能够理解动态变化的世界！

### 1.3 历史背景

| 年份 | 里程碑 |
|------|--------|
| 2020 | Vision Transformer (ViT) - Transformer应用于图像 |
| 2021 | ViViT - 视频Vision Transformer |
| 2021 | TimeSformer - 时空注意力 |
| 2022 | VideoMAE - 视频遮蔽自编码器 |
| 2023 | LLaMA-VIDEO - 大规模视频预训练 |

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 视频理解 / 时空建模 |
| 输入 | 视频帧序列 (T×H×W×C) |
| 核心 | 时空注意力 + 位置编码 |
| SOTA | 视频分类、动作识别 |

### 1.5 前置知识

- Vision Transformer (ViT) 原理
- 自注意力机制
- 视频处理基础
- PyTorch

---

## 2. 核心原理

### 2.1 视频Token化

**核心思想**：将视频视为时空patch序列

**输入**：视频帧序列
$$V \in \mathbb{R}^{T \times H \times W \times C}$$

其中：$T$ = 帧数，$H$ = 高度，$W$ = 宽度，$C$ = 通道数

**Patch化**：
1. 每帧分为不重叠的patches：$(H \times W) / (p^2)$ 个patches/帧
2. 共 $T \times (H \times W) / p^2$ 个patches
3. 每个patch通过线性投影得到patch embeddings

### 2.2 时空注意力架构

ViViT使用三种主要的注意力架构：

| 方案 | 描述 | 复杂度 |
|------|------|--------|
| **Joint Space-Time** | 所有patch一起做注意力 | $O((N \cdot T)^2)$ |
| **Factorized Encoder** | 空间注意 + 时间注意串行 | $O(N^2 \cdot T + N \cdot T^2)$ |
| **Factorized Self-Attention** | 空间/时间分别做注意力 | $O(N^2 \cdot T + N \cdot T^2)$ |

### 2.3 具体架构

**1. Joint Space-Time Attention**：
```python
# 所有patch一起做自注意力
# 输入: (B, T*N, D) - N = 每个帧的patch数
patches_all = rearrange(patches, 'B T N D -> B (T N) D')
attention_output = self.attention(patches_all)
```

**2. Factorized Encoder**：
```python
# 先空间注意力，再时间注意力
# 空间注意力：每个帧内做attention
spatial_features = self.spatial_attention(patches)  # (B, T*N, D)
# 时间注意力：跨帧做attention
# reshape to (B, T, N, D)
temporal_features = self.temporal_attention(spatial_features)
```

**3. Divided Space-Time Attention**：
```python
# 在每个块中同时做空间和时间注意力
patch_tokens = rearrange(patches, 'B T N D -> B (T N) D')

# 空间注意力
spatial_out = self.sa_spatial(patch_tokens)

# 时间注意力
temporal_out = self.sa_temporal(patch_tokens)

# 结合
output = spatial_out + temporal_out
```

### 2.4 工作流程

```python
def vivit_forward(video):
    # 1. 帧采样和预处理
    frames = sample_frames(video, num_frames=16)
    frames = preprocess(frames)
    
    # 2. Patch化
    patches = extract_patches(frames, patch_size=16)
    
    # 3. 线性投影 + 位置编码
    patch_embeddings = self.proj(patches)
    patch_embeddings = patch_embeddings + self.pos_encoding
    
    # 4. 添加[class] token
    tokens = torch.cat([class_token, patch_embeddings], dim=1)
    
    # 5. Transformer blocks
    for block in self.blocks:
        tokens = block(tokens)
    
    # 6. 分类
    output = self.head(tokens[:, 0])
    return output
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $T$ | 视频帧数 |
| $H, W$ | 帧的空间尺寸 |
| $C$ | 通道数 |
| $P$ | Patch size |
| $N = (H \cdot W) / P^2$ | 每帧的patch数 |
| $M = T \cdot N$ | 总patch数 |
| $D$ | embedding维度 |

### 3.2 输入形式化

**输入张量**：
$$V \in \mathbb{R}^{B \times T \times H \times W \times C}$$

**Patch提取**：
每个帧被划分为 $N$ 个不重叠的patches：
$$x_{t,n} \in \mathbb{R}^{P^2 \cdot C}$$

其中 $n = 1, ..., N$，$t = 1, ..., T$

### 3.3 Patch投影

**线性投影**：
$$e_{t,n} = E \cdot x_{t,n} + b_e$$

其中 $E \in \mathbb{R}^{D \times (P^2 \cdot C)}$

### 3.4 位置编码

**2D + 1D 位置编码**（空间 + 时间）：
$$PE(pos) = [PE_s(x, y); PE_t(t)]$$

**空间位置编码**：
$$PE_s(x, y) = \begin{cases} \sin(x / 10000^{2i/D}) \\ cos(x / 10000^{2i/D}) \end{cases}$$

**时间位置编码**：
$$PE_t(t) = \begin{cases} \sin(t / 10000^{2i/D}) \\ cos(t / 10000^{2i/D}) \end{cases}$$

### 3.5 联合空间-时间注意力

**输入**：
$$z^{(0)} = [z_{cls}; e_{0,0}; e_{0,1}; ...; e_{T-1,N-1}] + PE$$

**Multi-Head Attention**：
$$z^{(l+1)} = \text{MHA}(z^{(l)}, z^{(l)}, z^{(l)})$$

每个head：
$$\text{head}_i = \text{Att}(z^{(l)}W_i^Q, z^{(l)}W_i^K, z^{(l)}W_i^V)$$

### 3.6 Factorized Attention

**空间注意力**（每个时间步独立）：
$$z^{(l+1)}_s = \text{MSA}(z^{(l)}_s, z^{(l)}_s, z^{(l)}_s) + z^{(l)}_s$$

**时间注意力**（跨所有帧）：
$$z^{(l+1)}_t = \text{MSA}(\text{Permute}(z^{(l)}_t), ...) + z^{(l)}_t$$

### 3.7 复杂度分析

| 架构 | 空间复杂度 | 时间复杂度 |
|------|------------|------------|
| Joint | $O((TM)^2)$ | $O((TM)^2)$ |
| Factorized | $O(TM^2 + T^2M)$ | $O(TM^2 + T^2M)$ |

对于典型设置 $T=8, M=196$：
- Joint: $8.5 \times 10^5$
- Factorized: $3.3 \times 10^5$

---

## 4. PyTorch实现

### 4.1 核心模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class PatchEmbed(nn.Module):
    """将视频帧转换为patch embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = (img_size // patch_size) ** 2
        
        # 3D卷积：同时处理时间和空间
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            padding=0
        )
    
    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # 投影
        x = self.proj(x)  # (B, D, T', H', W')
        
        # 展平
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', D)
        
        return x


class Attention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViViT(nn.Module):
    """Video Vision Transformer"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., num_frames=16):
        super(ViViT, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        # Patch嵌入
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, num_frames)
        num_patches = self.patch_embed.num_patches * num_frames
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化
        nn.trunc_normal_(self.pos_embed, std=.02)
        nn.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch投影
        x = self.patch_embed(x)  # (B, T*N, D)
        
        # 添加class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 分类
        return self.head(x[:, 0])
```

### 4.2 时空注意力变体

```python
class FactorizedAttention(nn.Module):
    """分解的时空注意力"""
    
    def __init__(self, dim, num_heads=8, temporal_size=8):
        super().__init__()
        self.temporal_size = temporal_size
        
        self.spatial_attn = Attention(dim, num_heads // 2)
        self.temporal_attn = Attention(dim, num_heads // 2)
    
    def forward(self, x):
        B, T_N, D = x.shape
        
        # 空间注意力
        x_spatial = self.spatial_attn(x)
        
        # 时间注意力 - 重新排列
        x_temporal = rearrange(x_spatial[0], '(B T) N D -> B T N D', B=B//self.temporal_size)
        for t in range(self.temporal_size):
            x_temporal[:, t] = self.temporal_attn(x_temporal[:, t])
        x_temporal = rearrange(x_temporal, 'B T N D -> (B T) N D')
        
        return x_temporal, x_spatial[1]


class TemporalAttention(nn.Module):
    """单独的时间注意力层"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, D = x.shape
        T = B  # 假设每个样本独立
        
        # 重新排列：(Batch, Time, Num_patches, Dim)
        x = x.view(B // T, T, N, D)
        
        # 时间注意力：只关注时间维度
        qkv = self.qkv(x).chunk(3, dim=-1)
        
        # 计算时间注意力...
        
        return x.view(B, N, D)
```

### 4.3 训练

```python
class ViViTTrainer:
    """ViViT训练器"""
    
    def __init__(self, model, lr=1e-4, weight_decay=0.05):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
    
    def train_step(self, batch):
        videos, labels = batch
        
        # 前向传播
        outputs = self.model(videos)
        
        # 损失
        loss = F.cross_entropy(outputs, labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_loop(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for batch in dataloader:
                loss = self.train_step(batch)
            
            self.scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

---

## 5. 代码示例

### 5.1 完整示例

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def demo_vivit():
    """ViViT演示"""
    
    print("=" * 60)
    print("ViViT (Video Vision Transformer) 演示")
    print("=" * 60)
    
    # 参数
    B, T, C, H, W = 2, 8, 3, 224, 224
    num_classes = 400
    
    print(f"输入形状: ({B}, {T}, {C}, {H}, {W})")
    
    # 模型
    model = ViViT(
        img_size=224,
        patch_size=16,
        num_frames=8,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试输入
    x = torch.randn(B, C, T, H, W)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"输出形状: {output.shape}")
    print(f"预测类别: {output.argmax(dim=-1)}")
    
    return model, output


def test_different_architectures():
    """测试不同架构"""
    
    configs = [
        ("Joint Space-Time", "joint"),
        ("Factorized Encoder", "factorized"),
        ("Divided Space-Time", "divided"),
    ]
    
    print("\n架构对比:")
    print("-" * 50)
    
    for name, arch in configs:
        # 复杂度和参数对比
        flops = {
            "joint": 8.5e5,
            "factorized": 3.3e5,
            "divided": 3.3e5,
        }
        
        print(f"{name}:")
        print(f"  - FLOPs: {flops[arch]:.2e}")
        print(f"  - 内存: {'高' if arch == 'joint' else '中'}")
    
    return True


def visualize_patches():
    """可视化patch提取"""
    
    T, H, W, P = 8, 224, 224, 16
    
    # 创建网格
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for t in range(T):
        row = t // 4
        col = t % 4
        
        # 简化的frame
        frame = np.random.rand(H, W, 3)
        
        # 标记patch边界
        for i in range(0, H, P):
            frame[i, :] = [1, 0, 0]
        for j in range(0, W, P):
            frame[:, j] = [1, 0, 0]
        
        axes[row, col].imshow(frame)
        axes[row, col].set_title(f'Frame {t+1}')
        axes[row, col].axis('off')
    
    plt.suptitle('ViViT Patch划分示意')
    plt.tight_layout()
    plt.savefig('vivit_patches.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    model, output = demo_vivit()
    test_different_architectures()
    visualize_patches()
```

---

## 6. 应用场景

### 6.1 视频分类

| 应用 | 说明 |
|------|------|
| **动作识别** | Kinetics-400/600/700 |
| **视频分类** | UCF-101, HMDB-51 |
| **多标签分类** | ActivityNet |

### 6.2 视频理解

| 应用 | 说明 |
|------|------|
| **视频问答** | MSRVTT, TVQA |
| **时序动作定位** | ActivityNet Captions |
| **视频检索** | 文��-视���匹配 |

### 6.3 代码

```python
# 使用预训练模型
from transformers import VideoMAEForVideoClassification

model = VideoMAEForVideoClassification.from_pretrained("MCB/kinetics400-vivit-base")
processor = VideoProcessor.from_pretrained("MCB/kinetics400-vivit-base")

# 处理视频
inputs = processor(video_path, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits
```

---

## 7. 优缺点分析

### 7.1 优点

| 优点 | 说明 |
|------|------|
| **长程建模** | 能捕捉远距离时间依赖 |
| **并行计算** | 帧间不依赖 |
| **灵活架构** | 可定制时空注意力 |
| **可扩展性** | 类似于ViT |

### 7.2 缺点

| 缺点 | 说明 | 缓解 |
|------|------|------|
| **计算大** | $O(T^2 \cdot N^2)$ | Factorized attention |
| **内存高** | 长视频 | 降低帧数 |
| **数据需求** | 需要大量数据 | 预训练 |

### 7.3 对比

| 方法 | 时间复杂度 | 空间复杂度 | 效果 |
|------|-----------|-----------|------|
| CNN + LSTM | $O(T)$ | $O(1)$ | 中 |
| CNN + pooling | $O(1)$ | $O(1)$ | 中 |
| **ViViT** | $O(T^2)$ | $O(T^2)$ | 高 |

---

## 8. 常见问题与易错点

### 8.1 问题1：视频太长

**问题**：处理长视频内存爆炸

**解决**：稀疏采样或片段处理
```python
def process_long_video(video, max_frames=32):
    # 均匀采样
    indices = torch.linspace(0, len(video)-1, max_frames)
    return video[indices]
```

### 8.2 问题2：过拟合

**问题**：过拟合到特定帧

**解决**：数据增强
```python
transforms.Compose([
    RandomCrop(),
    RandomHorizontalFlip(),
    ColorJitter(),
    RandomTemporalCrop(),
])
```

### 8.3 问题3：预训练资源

**问题**：需要大规模预训练

**解决**：使用Kinetics预训练
```python
model = load_pretrained("vivit-kinetics400")
```

---

## 9. 学习总结

### 9.1 核心要点

1. **3D Patch**：空间+时间patch提取
2. **时空注意力**：Joint或Factorized
3. **位置编码**：空间+时间位置编码

### 9.2 关键公式

$$M = T \times (H \cdot W) / P^2$$

$$\text{Attention}(X) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$$

### 9.3 学习路径

ViT → ViViT → TimeSformer → VideoMAE

---

## 10. 练习题

### 10.1 基础题

1. ViViT和TimeSformer的核心区别
2. 为什么需要时间位置编码

### 10.2 进阶题

3. 实现Factorized ViViT
4. 比较三种注意力架构

### 10.3 答案

<details>
<summary>答案1</summary>

TimeSformer使用Divide-and-Conquer的时空注意力，而ViViT使用Joint或Factorized attention。两者都基于 Transformer架构。

</details>

<details>
<summary>答案2</summary>

因为视频中同一位置不同时间的内容不同，如果没有时间位置编码，模型无法区分不同帧的相同空间位置。类似地，对于NLP中同一单词在不同位置。

</details>

---

## 11. 学习路径建议

### 11.1 第一阶段

1. 学习ViT基础
2. 理解视频处理
3. 实现基础ViViT

### 11.2 第二阶段

1. 时空注意力变体
2. 调参实践
3. 数据集实验

### 11.3 第三阶段

1. 预训练模型
2. 视频问答
3. 实际应用

---

## 12. 可视化与结果理解

```python
def visualize_attention_weights():
    """可视化注意力"""
    
    # 获取attention weights
    attn_weights = []
    
    def hook_fn(module, input, output):
        attn_weights.append(output[1])
    
    # 注册hook
    for block in model.blocks:
        block.attn.register_forward_hook(hook_fn)
    
    # 前向传播
    output = model(video)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    attn = attn_weights[0].mean(dim=1)[0, 0].reshape(T, N)
    
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Patch Index')
    plt.ylabel('Time Frame')
    plt.title('Temporal Attention')
    plt.show()
```

---

## 13. 模型评估

### 13.1 评估指标

| 指标 | 说明 |
|------|------|
| **Top-1 Accuracy** | 最高预测准确率 |
| **Top-5 Accuracy** | 前5预测准确率 |
| **mAP** | mean Average Precision |

### 13.2 代码

```python
from torchmetrics import Accuracy

accuracy = Accuracy(task='multiclass', num_classes=400)
for batch in dataloader:
    videos, labels = batch
    outputs = model(videos)
    predictions = outputs.argmax(dim=-1)
    accuracy(predictions, labels)

print(f"Accuracy: {accuracy.compute()}")
```

---

## 14. 进阶内容

### 14.1 变体

| 模型 | 核心改进 |
|------|----------|
| TimeSformer | 分离时空注意力 |
|VideoMAE | Masked自编码 |
| MTV | 多尺度ViT |
| CoCa | 对比+标题预测 |

### 14.2 预训练

1. Kinetics-400/600/700
2. HowTo100M
3. WebVid

### 14.3 推荐资源

- ViViT: Video Vision Transformers
- An Empirical Study of Video Vision Transformers

---

**文档结束**

*参考论文：ViViT: Video Vision Transformers (Arnab et al., 2021)*

## 4. 训练过程讲解
### 训练步骤
1. **数据加载**：Dataset + DataLoader 批处理
2. **前向传播**：数据通过网络计算输出
3. **损失计算**：对比预测与标签
4. **反向传播**：计算梯度
5. **参数更新**：优化器更新权重
6. **循环迭代**：重复直至收敛

## 7. 调库实现（Python + 完整代码 + 注释）

以下是使用主流框架实现ViViT的代码：

```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 数据准备
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
train_set, test_set = random_split(dataset, [800, 200])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,2))
    def forward(self, x): return self.net(x)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
opt = optim.Adam(model.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()
for epoch in range(50):
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        crit(model(bx), by).backward()
        opt.step()
```

## 8. 手工代码实现（核心算法手写 + 注释）

以下是从零实现：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ViViTNet(nn.Module):
    def __init__(self, dim_in=20, dim_h=64, dim_out=2):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim_in, dim_h), nn.Linear(dim_h, dim_out)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))

def train(model, X, y, epochs=100, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X), y)
        loss.backward(); opt.step()
        if (ep+1)%20==0: print(f"Ep{ep+1} loss={loss.item():.4f}")

m = ViViTNet()
train(m, torch.randn(500,20), torch.randint(0,2,(500,)))
```
