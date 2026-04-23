# MAE (Masked Autoencoders) 学习文档

> 可迁移自监督预训练，ViT 时代的最强表示学习。

---

## 1. 算法基础认知

### 1.1 发展背景

MAE 由 He 等人于 2022 年在论文《Masked Autoencoders Are Scalable Vision Transformers》中提出，通过随机遮蔽大部分图像 patch 进行自监督预训练，在 ImageNet 上达到了 87.8% 的 Top-1 准确率。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 自监督预训练 |
| 遮蔽率 | 75% |
| 编码器 | ViT |
| 解码器 | Transformer |

### 1.3 性能对比

| 方法 | ImageNet | 参数 |
|------|----------|------|
| supervised | 84.5% | 86M |
| BEiT | 86.3% | 86M |
| MAE | 87.8% | 86M |

---

## 2. 核心原理

### 2.1 遮蔽策略

随机遮蔽 75% 的图像 patch：

- 编码器：仅处理可见 patch（25%）
- 解码器：重建被遮蔽 patch

### 2.2 非对称编码器-解码器

```
编码器：ViT-B（处理 25% patch）
解码器：ViT-L（轻量级，重建 75%）
```

---

## 3. 数学公式与推导

### 3.1 重建目标

预测原始 patch 的归一化像素值：

$$\hat{x}_i = \text{Decoder}(E(I_{vis}))$$

### 3.2 损失函数

$$\mathcal{L} = \frac{1}{|m|} \sum \| \hat{x}_i - x_i \|^2$$

其中 $m$ 是遮蔽比例。

---

## 4. 训练过程讲解

### 4.1 训练配置

| 参数 | 值 |
|------|-----|
| 遮蔽率 | 75% |
| 批量大小 | 4096 |
| 预训练 | 800 epochs |
| 初始化 | ImageNet-1K |

### 4.2 重建目标

使用归一化像素值而非 Patch embeddings

---

## 5. 应用场景

### 5.1 典型应用

- **图像分类**：下游任务微调
- **目标检测**：DETR 系列
- **语义分割**：UperNet

### 5.2 代码示例

```python
import torch
import torchvision.transforms as T

# 数据增强
transform = T.Compose([
    T.RandomResizedCrop(224),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MAE 预训练模型
from mae import mae_vit_base_patch16

model = mae_vit_base_patch16(pretrained=True)
```

---

## 6. 调库实现

### 6.1 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAE(nn.Module):
    """Masked Autoencoder"""
    
    def __init__(self, embed_dim=768, patch_size=16, mask_ratio=0.75):
        super().__init__()
        
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # 编码器
        self.encoder = ViTEncoder(embed_dim)
        
        # 解码器
        self.decoder = ViTDecoder(embed_dim, num_patches=196)
        
    def forward(self, x):
        # 生成 patch
        x = self.patchify(x)
        
        # 随机遮蔽
        visible_indices, masked_indices = self.random_mask(x)
        visible_x = x[visible_indices]
        
        # 编码
        encoded = self.encoder(visible_x)
        
        # 解码
        reconstructed = self.decoder(encoded, masked_indices)
        
        return reconstructed
    
    def random_mask(self, x):
        N = len(x)
        num_visible = int(N * (1 - self.mask_ratio))
        indices = torch.randperm(N)[:num_visible]
        
        visible = indices.sort()[0]
        masked = torch.cat([i for i in range(N) if i not in visible])
        
        return visible, masked


def demo():
    print("=== MAE 演示 ===\n")
    model = MAE()
    print(f"遮蔽率: 75%")
    print(f"应用: 自监督预训练")


if __name__ == "__main__":
    demo()
```

### 6.2 预训练模型加载

```python
import mae_vit_base_patch16

model = mae_vit_base_patch16(pretrained=True)
```

---

## 7. 手工代码实现

### 7.1 简化 MAE

```python
import torch
import torch.nn as nn

class MAEEncoder(nn.Module):
    """MAE 编码器"""
    
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        
        # patchify
        x = self.proj(x)  # (B, C, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        
        if mask is not None:
            x = x[mask]
        
        return x


class MAEDecoder(nn.Module):
    """MAE 解码器"""
    
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.proj = nn.Linear(embed_dim, num_patches)
        
    def forward(self, x):
        
        # 重建
        x = self.proj(x)
        
        return x
```

---

## 8. 优缺点分析

### 8.1 优点

1. **简单有效**：随机遮蔽即可
2. **可扩展性好**：适用于任何 ViT
3. **性能 SOTA**：ImageNet 最高

### 8.2 缺点

1. **训练慢**：需要更多 epochs
2. **重建目标**：像素值

### 8.3 改进方向

- MAE v2：更好的归一化
- MixMAE：混合遮蔽

---

## 9. 可视化与结果理解

### 9.1 重建效果

```python
def visualize():
    print("""
    原图          遮蔽 (75%)     重建
    
    ┌───┬───┐            ? ? ? ?
    │ █ │ █ │      →      ? █ █ ?
    ├───┼───┤    MAE     ? █ █ ?
    │ █ │ █ │            ? ? ? ?
    └───┴───┘            ? ? ? ?
    """)
```

---

## 10. 模型评估

### 10.1 ImageNet 微调

| 预训练 | Top-1 | 精度 |
|--------|-------|-------|
| supervised | 86.5% | |
| BEiT | 87.4% | |
| MAE | 87.8% | |

---

## 11. 学习总结

**核心要点**：

1. **随机遮蔽**：75% 高遮蔽率
2. **非对称结构**：编码器轻量化
3. **像素重建**：简单有效

**MAE 核心优势**：
- 可扩展性强
- 性能 SOTA
- 简单实现

**学习建议**：

1. 理解 ViT
2. 掌握自监督
3. 实践预训练

---

## 12. 练习题与思考题

### 12.1 基础练习

1. MAE vs BERT
2. 遮蔽率选择

### 12.2 思考题

1. 为什么高遮蔽率有效

---

### 12.3 详细答案

**问题**：75% 遮蔽率

**解答**：

- 减少建模难度
- 增加重建难度
- 防止信息泄露

---

## 14. 学习路径建议

### 入门阶段

1. ViT 基础
2. 自监督学习

### 进阶阶段

1. MAE 原理
2. 对比 BEiT

### 高级阶段

1. 改进 MAE
2. 下游应用

**推荐路线**：

```
ViT → BEiT → MAE → MixMAE → DINO
```

**MAE 是视觉自监督的里程碑，熟练掌握它对学习视觉预训练很重要。**