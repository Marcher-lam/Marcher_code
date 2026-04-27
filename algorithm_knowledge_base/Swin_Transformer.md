# Swin Transformer 学习文档

> 带移动窗口的分层视觉Transformer，在图像分类、检测、分割任务上取得SOTA。

## 1. 算法基础认知

### 一句话定义

Swin Transformer通过移动窗口划分实现局部注意力，并结合层级结构实现多尺度表示，成为视觉Transformer的里程碑模型。

### 直觉类比

就像用放大镜看地图——每次只关注一个窗口区域，然后移动放大镜查看下一个区域。通过窗口的滑动覆盖整个地图，同时保持计算效率。

### 历史背景

- **2021年3月**：Microsoft Research提出Swin Transformer
- **后续发展**：Swin-V2、Swin3D等

### 算法定位

Swin Transformer是**视觉基础模型**，可用于分类、检测、分割等任务。

---

## 2. 核心原理

### 核心创新

1. **分层结构**：像CNN一样产生多尺度特征
2. **移动窗口**：通过窗口移动实现全局建模
3. **局部注意**：每个窗口内计算注意力，降低复杂度

### 架构图

```
输入图像 → Patch分割 → 线性嵌入 → Swin Block × 2 → Patch合并
→ Swin Block × 2 → Patch合并 → Swin Block × 2 → 全局池化 → 分类
```

---

## 3. 数学公式

### 窗口注意力

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V$$

$O(M^2 \cdot d)$ 复杂度，$M$是窗口大小。

### 移动窗口

- 偶数层：规则窗口划分
- 奇数层：窗口偏移$\lfloor M/2 \rfloor$

---

## 4. 调库实现

```python
import torch
import torch.nn as nn

class SwinTransformer(nn.Module):
    """Swin Transformer实现"""
    def __init__(self, img_size=224, patch_size=4, 
                 in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24]):
        super(SwinTransformer, self).__init__()
        
        # 简化实现 - 完整版需要实现SwinBlock
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 
                                    kernel_size=patch_size, 
                                    stride=patch_size)
        
        self.layers = nn.ModuleList()
        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            layer = nn.ModuleList([
                SwinBlock(embed_dim * (2**i), num_head)
                for _ in range(depth)
            ])
            self.layers.append(layer)
            
        self.norm = nn.LayerNorm(embed_dim * 8)
        self.head = nn.Linear(embed_dim * 8, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        for layer in self.layers:
            for block in layer:
                x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

class SwinBlock(nn.Module):
    """Swin Transformer Block（简化版）"""
    def __init__(self, dim, num_heads):
        super(SwinBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# 使用timm库
def use_timm_swin():
    import timm
    model = timm.create_model('swin_base_patch4_window7_224', 
                              pretrained=True)
    return model

# 测试
if __name__ == "__main__":
    model = SwinTransformer()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"输出形状: {out.shape}")  # (1, 1000)
```

---

## 5. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_swin_windows():
    """可视化Swin窗口划分"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    h, w = 224, 224
    window_size = 7
    
    # 规则窗口
    axes[0].set_title('Regular Window Partition')
    for i in range(0, h, window_size):
        axes[0].axhline(i, color='red', linewidth=0.5)
    for j in range(0, w, window_size):
        axes[0].axvline(j, color='red', linewidth=0.5)
    axes[0].set_xlim(0, w)
    axes[0].set_ylim(h, 0)
    axes[0].axis('off')
    
    # 移动窗口
    axes[1].set_title('Shifted Window Partition')
    offset = window_size // 2
    for i in range(-offset, h, window_size):
        axes[1].axhline(i, color='blue', linewidth=0.5)
    for j in range(-offset, w, window_size):
        axes[1].axvline(j, color='blue', linewidth=0.5)
    axes[1].set_xlim(0, w)
    axes[1].set_ylim(h, 0)
    axes[1].axis('off')
    
    # 注意力连接
    axes[2].set_title('Window Connections')
    axes[2].plot([100, 100], [50, 150], 'o-', linewidth=2, markersize=5)
    axes[2].plot([100, 150], [100, 110], 'o-', linewidth=2, markersize=5)
    axes[2].set_xlim(0, 224)
    axes[2].set_ylim(224, 0)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('swin_windows.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    visualize_swin_windows()
```

---

## 6. 性能对比

| 模型 | ImageNet Top-1 | Params |
|------|----------------|--------|
| Swin-T | 81.2% | 28M |
| Swin-S | 83.2% | 50M |
| Swin-B | 85.2% | 88M |
| Swin-L | 87.3% | 197M |

---

## 7. 学习路径

- 前置：ViT、Transformer
- 平行： DeiT
- 进阶：Swin-V2、SegFormer