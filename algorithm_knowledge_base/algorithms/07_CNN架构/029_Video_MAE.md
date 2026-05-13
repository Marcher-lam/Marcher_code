# Video MAE (Video Masked Autoencoder) 视频遮蔽自编码器 学习文档

> VideoMAE是基于掩码自编码器框架的视频自监督预训练方法，通过随机遮蔽部分视频patches并重建来学习视频表示

---

## 1. 算法基础认知

### 1.1 一句话定义

**Video MAE** 是一种用于视频理解的自监督预训练方法，借鉴图像MAE的成功，将 Transformer 应用于视频领域，通过随机遮蔽大部分视频patches并预测被遮蔽的像素内容来学习强大的视频表示。

### 1.2 直觉类比

想象你看一部被遮掉很多部分（打马赛克）的电影：你需要根据看到的零碎画面猜测整个场景在发生什么。**Video MAE** 正是让模型做这件事——随机遮蔽75-90%的视频patch，然后让模型学习重建被遮蔽的内容。这个过程迫使模型理解视频的时空结构：既要理解每帧的空间内容（这个人是谁、在哪里），还要理解物体如何随时间运动。

### 1.3 历史背景

| 年份 | 里程碑 |
|------|--------|
| 2020 | BEiT - 图像遮蔽建模 |
| 2021 | MAE - 掩码自编码器 |
| 2022 | VideoMAE - 视频MAE |
| 2022 | MaskCLIP - 无监督视频表示 |
| 2023 |VideoMAE v2 - 改进版本 |

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 自监督预训练 |
| 核心 | 掩码重建 + Transformer |
| 遮蔽率 | 75-90%（高遮蔽） |
| 地位 | 视频自监督SOTA |

### 1.5 前置知识

- Transformer架构
- 自监督学习基础
- 视频处理基础
- PyTorch

---

## 2. 核心原理

### 2.1 MAE核心思想

**掩码自编码器** 框架包含：
1. **编码器**：处理可见 patches
2. **遮蔽 tokens**：表示被遮蔽的位置
3. **解码器**：重建被遮蔽的 patches

**关键设计**：
- 高遮蔽率（75-90%）
- 非对称编码器-解码器结构
- 重建原始像素（而非特征）

### 2.2 视频特殊性

**与图像对比**：

| 维度 | 图像 | 视频 |
|------|------|------|
| 输入 | 2D (H×W) | 3D (T×H×W) |
| Token数 | ~200 | ~200×T |
| 冗余性 | 中 | 高 |
| 时间维度 | 无 | 有 |

**Video MAE 适配**：
1. 使用 **3D patches**：T×H×W的时空块
2. 高遮蔽率仍然有效（视频冗余性高）
3. 需要时间位置编码

### 2.3 整体架构

```python
# VideoMAE 流程
def video_mae_forward(video, mask_ratio=0.9):
    # 1. 提取3D patches
    patches = extract_3d_patches(video)  # (B, T×N, D)
    
    # 2. 随机遮蔽
    visible_patches, mask, ids = random_mask(patches, mask_ratio)
    
    # 3. 编码器（只处理可见patches）
    encoded = encoder(visible_patches)
    
    # 4. 添加遮蔽tokens
    encoded = patch_mask_tokens(encoded, mask)
    
    # 5. 解码器（重建像素）
    reconstructed = decoder(encoded)
    
    # 6. 计算损失
    loss = mse_loss(reconstructed, original_patches, mask)
    
    return loss
```

### 2.4 遮蔽策略

**随机遮蔽**：
```python
def random_mask(patches, mask_ratio=0.9):
    B, N, D = patches.shape
    
    # 随机生成mask
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N)
    ids_shuffle = torch.argsort(noise, dim=-1)
    ids_keep = ids_shuffle[:, :len_keep]
    ids_mask = ids_shuffle[:, len_keep:]
    
    # 应用mask
    visible = patches.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.zeros(B, N).scatter(1, ids_mask, 1)
    
    return visible, mask, ids_shuffle
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $T$ | 视频帧数 |
| $H, W$ | 帧的空间尺寸 |
| $P$ | Patch size |
| $N = (H \cdot W) / P^2$ | 每帧patch数 |
| $M = T \cdot N$ | 总patch数 |
| $\alpha$ | 遮蔽率 |
| $D$ | embedding维度 |

### 3.2 3D Patch提取

**输入**：
$$x \in \mathbb{R}^{T \times H \times W \times C}$$

**Patch化**：
对于 3D patch $p \in \mathbb{R}^{P \times P \times P \times C}$：
$$e_{t,i,j} = E \cdot \text{Flatten}(p_{t,i,j}) + b_e$$

### 3.3 遮蔽位置编码

**可见 tokens 集合**：
$$I = \{i \in [1,M] \mid m_i = 0\}$$

**遮蔽 tokens**：
$$\text{mask\_token} \in \mathbb{R}^D$$

### 3.4 重建目标

**像素重建**：
$$\mathcal{L} = \frac{1}{|\bar{I}|} \sum_{i \in \bar{I}} \| \hat{x}_i - x_i \|^2$$

其中 $\bar{I}$ 是被遮蔽的位置集合。

### 3.5 复杂度分析

| 操作 | 复杂度 |
|------|--------|
| 编码器 | $O((1-\alpha)M \cdot D)$ |
| 解码器 | $O(M \cdot D)$ |
| 总计 | $O(M \cdot D)$ |

高遮蔽率大幅减少计算量。

---

## 4. PyTorch实现

### 4.1 核心模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VideoPatchEmbed(nn.Module):
    """3D Video Patch 嵌入"""
    
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, num_frames=16):
        super(VideoPatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        
        # 3D 卷积
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size)
        )
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, D, T', H', W')
        
        # 展平: (B, T'*H'*W', D)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class MaskGenerator(nn.Module):
    """随机掩码生成器"""
    
    def __init__(self, mask_ratio=0.75):
        super(MaskGenerator, self).__init__()
        self.mask_ratio = mask_ratio
    
    def forward(self, x):
        B, N, D = x.shape  # N = T* H*W / p^2
        
        # 随机遮蔽
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留比例
        len_keep = int(N * (1 - self.mask_ratio))
        ids_keep = ids_shuffle[:, :len_keep]
        
        # 可见和遮蔽
        visible = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        
        return visible, mask, ids_restore


class VideoMAEEncoder(nn.Module):
    """VideoMAE 编码器"""
    
    def __init__(self, patch_size=16, num_frames=16, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4.):
        super(VideoMAEEncoder, self).__init__()
        
        self.patch_embed = VideoPatchEmbed(patch_size, 3, embed_dim, num_frames)
        self.num_patches = (224 // patch_size) ** 2 * num_frames
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x, ids_keep=None, mask=None):
        # Patch 嵌入
        x = self.patch_embed(x)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 随机遮蔽
        if ids_keep is not None:
            B, _, D = x.shape
            x = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class VideoMAEDecoder(nn.Module):
    """VideoMAE 解码器"""
    
    def __init__(self, embed_dim=768, decoder_dim=512, depth=4, num_heads=8):
        super(VideoMAEDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        
        # 投影到解码器维度
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        
        # 遮蔽 token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # 解码器位置编码
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        
        # 解码器 blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads)
            for _ in range(depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        
        # 投影回像素维度
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * 3 * 16)  # 3 channels * T
        
        nn.init.trunc_normal_(self.mask_token, std=.02)
    
    def forward(self, x, ids_restore, mask):
        B, len_keep, _ = x.shape
        num_mask = mask.sum(dim=1)[0]
        
        # 添加 mask tokens
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        
        # 恢复顺序
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim))
        
        # 添加位置编码
        x = x + self.decoder_pos_embed
        
        # 解码器
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.decoder_norm(x)
        
        # 预测
        x = self.decoder_pred(x)
        
        return x
```

### 4.2 完整模型

```python
class VideoMAE(nn.Module):
    """VideoMAE 完整模型"""
    
    def __init__(self, img_size=224, patch_size=16, num_frames=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
                 mask_ratio=0.75):
        super(VideoMAE, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.mask_ratio = mask_ratio
        
        # 组件
        self.encoder = VideoMAEEncoder(patch_size, num_frames, embed_dim, depth, num_heads, mlp_ratio)
        self.decoder = VideoMAEDecoder(embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads)
        
        self.mask_generator = MaskGenerator(mask_ratio)
    
    def forward(self, x):
        # 遮蔽
        visible_patches, mask, ids_restore = self.mask_generator(x)
        
        # 编码
        encoded = self.encoder(x, visible_patches, mask)
        
        # 解码
        reconstructed = self.decoder(encoded, ids_restore, mask)
        
        return reconstructed, mask
    
    def training_step(self, x):
        reconstructed, mask = self.forward(x)
        
        # 计算损失（只在被遮蔽位置）
        loss = F.mse_loss(reconstructed, x, mask=mask)
        
        return loss


class TransformerBlock(nn.Module):
    """标准Transformer块"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
```

### 4.3 预训练

```python
class VideoMAETrainer:
    """VideoMAE 训练器"""
    
    def __init__(self, model, lr=1e-4, weight_decay=0.05):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=800)
    
    def train_step(self, batch):
        videos, = batch  # 不需要labels
        
        # 重建
        reconstructed, mask = self.model(videos)
        
        # 损失（只有遮蔽位置）
        loss = F.mse_loss(reconstructed, videos, reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def pretrain(self, dataloader, num_epochs=800):
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in dataloader:
                loss = self.train_step(batch)
                total_loss += loss
            
            self.scheduler.step()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")
```

---

## 5. 代码示例

### 5.1 完整示例

```python
import torch
import numpy as np
import matplotlib.pyplot as plt


def demo_video_mae():
    """VideoMAE演示"""
    
    print("=" * 60)
    print("VideoMAE (Video Masked Autoencoder) 演示")
    print("=" * 60)
    
    # 参数
    B, T, C, H, W = 2, 16, 3, 224, 224
    
    print(f"输入形状: ({B}, {T}, {C}, {H}, {W})")
    
    # 模型
    model = VideoMAE(
        img_size=224,
        patch_size=16,
        num_frames=16,
        embed_dim=768,
        depth=12,
        mask_ratio=0.9
    )
    model.eval()
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试输入
    x = torch.randn(B, T, C, H, W)  # (B, T, C, H, W)
    x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    
    # 前向传播
    with torch.no_grad():
        reconstructed, mask = model(x)
    
    print(f"重建形状: {reconstructed.shape}")
    print(f"遮蔽率: {mask[0].sum()/mask[0].numel()*100:.1f}%")
    
    return model, reconstructed


def visualize_masking():
    """可视化遮蔽策略"""
    
    T, H, W, P = 16, 224, 224, 16
    N = (H // P) * (W // P) * T
    
    # 随机遮蔽
    np.random.seed(42)
    mask_ratio = 0.9
    visible_ratio = 1 - mask_ratio
    
    # 生成可视化的mask
    visible = np.random.rand(T, H//P, W//P) < visible_ratio
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    for t in range(T):
        row = t // 8
        col = t % 8
        
        # 可视化
        frame_mask = visible[t].astype(int)
        
        axes[row, col].imshow(frame_mask, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'Frame {t+1}')
        axes[row, col].axis('off')
    
    plt.suptitle(f'VideoMAE Masking (white=visible, black=masked, {mask_ratio*100:.0f}% masked)')
    plt.tight_layout()
    plt.savefig('video_mae_masking.png', dpi=150)
    plt.close()
    
    return True


def compare_mask_ratio():
    """对比不同遮蔽率"""
    
    ratios = [0.75, 0.85, 0.9, 0.95]
    
    print("\n遮蔽率对比:")
    print("-" * 50)
    print(f"{'遮蔽率':<15} {'减少计算':<15} {'重建难度':<15}")
    
    for ratio in ratios:
        compute_reduction = ratio * 100
        difficulty = "中" if ratio < 0.85 else "高"
        
        print(f"{ratio*100:.0f}%{'':<10} {compute_reduction:.1f}%{'':<8} {difficulty}")
    
    return True


if __name__ == "__main__":
    model, output = demo_video_mae()
    visualize_masking()
    compare_mask_ratio()
```

---

## 6. 应用场景

### 6.1 预训练

VideoMAE 主要用作大规模视频预训练，然后微调下游任务：

| 任务 | 说明 |
|------|------|
| **视频分类** | Kinetics-400/600/700 |
| **动作识别** | UCF-101, HMDB-51 |
| **视频检索** | 文本-视频匹配 |

### 6.2 下游任务

| 应用 | 说明 |
|------|------|
| **Video Understanding** | 视频问答 |
| **Segmentation** | 视频分割 |
| **Tracking** | 目标跟踪 |

### 6.3 代码

```python
# 使用预训练模型微调
from transformers import VideoMAEForVideoClassification

# 预训练模型
model = VideoMAEForVideoClassification.from_pretrained("MCB/videoMAE-base")

# 微调
for param in model.encoder.parameters():
    param.requires_grad = False  # 冻结encoder

# 在新数据上训练分类头
# ...
```

---

## 7. 优缺点分析

### 7.1 优点

| 优点 | 说明 |
|------|------|
| **高效** | 高遮蔽率减少计算量 |
| **高性能** | 预训练效果好 |
| **通用** | 适用于各种视频任务 |
| **简单** | 不需要对比学习负样本 |

### 7.2 缺点

| 缺点 | 说明 | 缓解 |
|------|------|------|
| **重建质量** | 可能重建模糊 | 中等遮蔽率 |
| **训练慢** | 解码器需要训练 | 中等规模解码器 |
| **资源需求** | 大Batch需要GPU | 梯度累积 |

### 7.3 对比

| 方法 | 遮蔽率 | 复杂度 | 效果 |
|------|--------|--------|------|
| ImageMAE | 75% | 中 | 高 |
| VideoMAE | 90% | 低 | 高 |
| BEiT | 40% | 中 | 中 |
| CLIP | 无 | 高 | 中 |

---

## 8. 常见问题与易错点

### 8.1 问题1：重建不收敛

**可能原因**：
1. 学习率过高
2. 遮蔽率过低
3. 解码器太简单

**解决方案**：
```python
# 降低学习率
lr = 1e-5

# 调整遮蔽率
mask_ratio = 0.75

# 增强解码器
decoder_depth = 6
```

### 8.2 问题2：视频重建质量差

**可能原因**：时间维度冗余未被利用

**解决**：使用3D Patch而非2D
```python
# 3D卷积
self.proj = nn.Conv3d(C, D, kernel_size=(2, P, P), stride=(2, P, P))
```

### 8.3 问题3：显存不足

**问题**：处理长视频OOM

**解决**：
```python
# 降低帧数
num_frames = 8

# 梯度累积
accumulation_steps = 4
```

---

## 9. 学习总结

### 9.1 核心要点

1. **高遮蔽率**：75-90%随机遮蔽
2. **3D Patches**：时空块
3. **非对称结构**：小编码器+大解码器
4. **像素重建**：直接预测原始像素

### 9.2 关键公式

$$\mathcal{L} = \frac{1}{|\bar{I}|} \sum_{i \in \bar{I}} \| \hat{x}_i - x_i \|^2$$

### 9.3 学习路径

MAE → VideoMAE → BEiT → MaskCLIP

---

## 10. 练习题

### 10.1 基础题

1. 解释VideoMAE为什么能用高遮蔽率
2. 3D patch和2D patch的区别

### 10.2 进阶题

3. 实现自己的VideoMAE预训练
4. 比较VideoMAE和ImageMAE

### 10.3 答案

<details>
<summary>答案1</summary>

视频具有很高的时空冗余性。即使遮蔽90%，模型也可以根据可见的邻近patches推断被遮蔽的内容。另外，高遮蔽率使得任务更难，迫使模型学习更通用的表示。

</details>

<details>
<summary>答案2</summary>

2D patch只捕获单帧的空间信息，3D patch同时捕获多帧的信息，能更好建模时间依赖。时间信息对于理解动作和行为至关重要。

</details>

---

## 11. 学习路径建议

### 11.1 第一阶段

1. 学习MAE基础
2. 理解视频处理
3. 实现基础VideoMAE

### 11.2 第二阶段

1. 预训练实践
2. 下游任务微调
3. 数据集实验

### 11.3 第三阶段

1. 改进遮蔽策略
2. 多模态预训练
3. 大规模应用

---

## 12. 可视化与结果理解

```python
def visualize_reconstruction():
    """可视化重建结果"""
    
    # 原始和重建对比
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 原始帧
    for i in range(4):
        axes[0, i].imshow(original_frame[i])
        axes[0, i].set_title('原始')
        axes[0, i].axis('off')
    
    # 重建帧
    for i in range(4):
        axes[1, i].imshow(reconstructed_frame[i])
        axes[1, i].set_title('重建')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## 13. 模型评估

### 13.1 预训练评估

| 指标 | 说明 |
|------|------|
| **重建Loss** | MSE |
| **Fine-tuning** | 下游准确率 |
| **Linear Probing** | 冻结encoder |

### 13.2 下游任务

```python
# Linear Probing
for param in model.encoder.parameters():
    param.requires_grad = False

# 冻结encoder
logits = model(video)
loss = cross_entropy(logits, labels)
```

---

## 14. 进阶内容

### 14.1 变体

| 模型 | 核心改进 |
|------|----------|
| VideoMAE | 基础版本 |
| VideoMAEv2 | 多尺度 |
| MaskCLIP | 对比学习 |
| U-Perceiver | 通用感知 |

### 14.2 遮蔽策略改进

1. **时间管遮蔽**：按时间管遮蔽
2. **随机遮蔽**：完全随机
3. **网格遮蔽**：规则网格

### 14.3 推荐资源

- Masked Autoencoder Are Scalable Vision Learners
- VideoMAE: Masked Autoencoder for Self-supervised Video Representation Learning

---

**文档结束**

*参考论文：VideoMAE: Masked Autoencoder for Self-supervised Video Representation Learning (Tong et al., 2022)*

## 4. 训练过程讲解
### 训练步骤
1. **数据加载**：Dataset + DataLoader 批处理
2. **前向传播**：数据通过网络计算输出
3. **损失计算**：对比预测与标签
4. **反向传播**：计算梯度
5. **参数更新**：优化器更新权重
6. **循环迭代**：重复直至收敛

## 7. 调库实现（Python + 完整代码 + 注释）

以下是使用主流框架实现Video_MAE的代码：

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

class VideoMAENet(nn.Module):
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

m = VideoMAENet()
train(m, torch.randn(500,20), torch.randint(0,2,(500,)))
```
