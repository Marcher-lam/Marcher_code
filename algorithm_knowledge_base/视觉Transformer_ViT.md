# Vision Transformer (ViT) 学习文档

> 将Transformer成功应用于图像分类的开创性工作，把图像视为patch序列。

## 1. 算法基础认知

### 一句话定义

ViT将图像分割成固定大小的patch，将每个patch视为一个"token"，然后用标准Transformer编码器进行处理。

### 直觉类比

就像把一张马赛克图片拆成一个个小方块，然后把这些小方块排成一排，让Transformer像处理文字一样处理它们。

### 历史背景

- **2020年10月**：Google发布ViT论文
- **2021年**：Swin Transformer、DeiT等改进
- **2022年**：BEiT、MAE等自监督方法

### 算法定位

ViT是**计算机视觉领域的里程碑**，证明Transformer可以替代CNN处理图像。

---

## 2. 核心原理

### 核心思想

将图像处理问题转化为序列处理问题：
1. 将图像划分为固定大小的patch
2. 每个patch通过线性投影变为向量
3. 添加分类token和位置编码
4. 用标准Transformer编码

### 工作流程

```
图像 (224×224) → 分割16×16 patches → 线性嵌入 → 添加[CLS]和位置编码
→ Transformer编码器 → 分类头 → 输出
```

### 架构参数

| 配置 | ViT-Base | ViT-Large | ViT-Huge |
|------|----------|------------|----------|
| Patch大小 | 16×16 | 16×16 | 16×16 |
| 序列长度 | 196 | 196 | 196 |
| 层数 | 12 | 24 | 32 |
| 隐藏维度 | 768 | 1024 | 1280 |
| 头数 | 12 | 16 | 16 |
| MLP维度 | 3072 | 4096 | 5120 |
| 参数量 | 86M | 307M | 632M |

---

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $H, W$ | 图像高和宽 |
| $P$ | Patch大小 |
| $N$ | Patch数量 $N = HW/P^2$ |
| $D$ | 嵌入维度 |

### Patch嵌入

将图像 $x \in \mathbb{R}^{H \times W \times C}$ 切分为 $N$ 个patch：
$$x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

通过线性投影：
$$z_0 = [x_{p}^1E; x_{p}^2E; ...; x_{p}^N E] + E_{pos}$$

其中 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$

### Transformer处理

$$z_l' = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$
$$z_l = \text{MLP}(\text{LN}(z_l')) + z_l'$$

### 分类

$$y = \text{LN}(z_L^0)$$

---

## 4. 调库实现

```python
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViTClassifier(nn.Module):
    """基于ViT的图像分类器"""
    def __init__(self, num_classes=1000):
        super(ViTClassifier, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# 从零实现简化版ViT
class PatchEmbedding(nn.Module):
    """Patch嵌入层"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 线性投影
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)
        
    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.proj(x)  # (batch, embed_dim, h/p, w/p)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x

class ViT(nn.Module):
    """简化版Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, 
                 in_channels=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12):
        super(ViT, self).__init__()
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, 
                                          in_channels, embed_dim)
        
        # 分类token和位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 
            self.patch_embed.num_patches + 1, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.size(0)
        
        # Patch嵌入
        x = self.patch_embed(x)  # (B, num_patches, D)
        
        # 添加分类token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码
        x = self.transformer(x)
        
        # 取分类token的输出
        cls_output = x[:, 0]
        
        return self.head(cls_output)

# 测试
if __name__ == "__main__":
    vit = ViT()
    x = torch.randn(4, 3, 224, 224)
    out = vit(x)
    print(f"输出形状: {out.shape}")  # (4, 1000)
```

---

## 5. 手工代码实现

```python
import numpy as np

class NumPyViT:
    """纯NumPy实现的简化ViT"""
    
    def __init__(self, img_size=32, patch_size=8, 
                 in_channels=3, embed_dim=128):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # 简化参数
        self.proj = np.random.randn(in_channels * patch_size**2, embed_dim) * 0.02
        self.cls_token = np.random.randn(1, embed_dim) * 0.02
        self.pos_embed = np.random.randn(self.num_patches + 1, embed_dim) * 0.02
        
    def split_patches(self, images):
        """将图像分割为patches"""
        batch_size = images.shape[0]
        patches = []
        
        for img in images:
            img_patches = []
            for i in range(0, self.img_size, self.patch_size):
                for j in range(0, self.img_size, self.patch_size):
                    patch = img[:, i:i+self.patch_size, j:j+self.patch_size]
                    img_patches.append(patch.flatten())
            patches.append(img_patches)
        
        return np.array(patches)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 分割patches
        patches = self.split_patches(x)  # (B, num_patches, patch_dim)
        
        # 线性投影
        x = np.dot(patches, self.proj)
        
        # 添加分类token
        cls_tokens = np.tile(self.cls_token, (batch_size, 1, 1))
        x = np.concatenate([cls_tokens, x], axis=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 简化Transformer处理（这里省略完整实现）
        
        return x

# 测试
if __name__ == "__main__":
    np.random.seed(42)
    vit = NumPyViT()
    x = np.random.randn(2, 3, 32, 32)
    out = vit.forward(x)
    print(f"输出形状: {out.shape}")  # (2, 17, 128)
```

---

## 6. 优缺点分析

### 优点

1. **可扩展性强**：大数据量时性能优异
2. **全局建模**：捕获全局依赖
3. **通用性强**：可迁移到多种视觉任务
4. **并行计算**：效率高于CNN

### 缺点

1. **小数据集表现差**：需要大量数据预训练
2. **计算量大**：处理高分辨率图像成本高
3. **位置信息有限**：位置编码可能不够精确
4. **局部特征弱**：缺乏CNN的归纳偏置

---

## 7. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_vit_patches():
    """可视化ViT的patch划分"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # 模拟图像patch划分
    np.random.seed(42)
    
    # 原图
    img = np.random.rand(224, 224, 3)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 16x16 patch划分
    for i in range(1, 6):
        row, col = i // 3, i % 3
        patch_size = 16 * (i // 2 + 1)
        
        # 绘制网格线
        ax = axes.flatten()[i]
        ax.imshow(np.random.rand(224, 224, 3))
        
        # 添加网格
        for x in range(0, 225, patch_size):
            ax.axhline(x, color='red', linewidth=0.5)
            ax.axvline(x, color='red', linewidth=0.5)
        
        ax.set_title(f'{patch_size}x{patch_size} Patches')
        ax.axis('off')
    
    plt.suptitle('ViT Patch Partition')
    plt.tight_layout()
    plt.savefig('vit_patches.png', dpi=150)
    plt.show()

def visualize_attention_map():
    """可视化ViT的注意力图"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # 模拟不同层的注意力模式
    patterns = [
        np.random.rand(14, 14) * 0.3 + np.eye(14) * 0.5,
        np.random.rand(14, 14) * 0.2 + 0.05,
        np.random.rand(14, 14) * 0.1,
        np.random.rand(14, 14) * 0.05 + 0.02,
    ]
    
    titles = ['Layer 1', 'Layer 6', 'Layer 9', 'Layer 12']
    
    for ax, pattern, title in zip(axes.flatten(), patterns, titles):
        im = ax.imshow(pattern, cmap='hot', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Patch Index')
        ax.set_ylabel('Patch Index')
    
    plt.suptitle('ViT Attention Maps across Layers')
    plt.tight_layout()
    plt.savefig('vit_attention.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    visualize_vit_patches()
    visualize_attention_map()
```

---

## 8. 学习路径

- 前置：Transformer、位置编码
- 平行：Swin Transformer、DeiT
- 进阶：BEiT、MAE、DINO