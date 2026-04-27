# MAE（Masked Autoencoder）学习文档

> 何恺明团队提出的自编码CV预训练模型。

## 1. 算法基础认知

### 一句话定义

MAE是Facebook AI提出的掩码自编码器，使用高比例随机mask和非对称编码器-解码器架构。

### 历史背景

- **2021年11月**：MAE论文发布
- **核心创新**：75%高mask率 + 非对称架构

### 算法定位

MAE是**CV自监督预训练模型**，属于掩码图像建模（MIM）。

---

## 2. 核心原理

### 核心设计

1. **高mask率**：75% mask，远高于BERT的15%
2. **非对称架构**：轻量解码器 + 完整编码器
3. **像素重构**：直接预测原始像素

### 预训练流程

```
输入图像 → 分块 → 随机75% mask → 编码器(仅25%块) → 解码器 → 重构mask块
```

### 数学公式

$$L = \frac{1}{M} \sum_{i \in M} ||x_i - \hat{x}_i||^2$$

其中M为mask的块数量。

---

## 3. 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAE(nn.Module):
    """MAE模型实现"""
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 depth=12, num_heads=12, mask_ratio=0.75):
        super(MAE, self).__init__()
        self.mask_ratio = mask_ratio
        
        # 编码器
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        
        # 解码器（更轻量）
        decoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4)
        self.decoder = nn.TransformerEncoder(decoder_layer, 4)
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.head = nn.Linear(embed_dim, patch_size ** 2 * 3)
        
    def forward(self, x):
        B, C, H, W = x.shape
        patch_h = patch_w = H // self.patch_embed.kernel_size[0]
        
        # 分块 + 位置编码
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embed
        
        # 生成mask
        N = x.shape[1]
        num_keep = int(N * (1 - self.mask_ratio))
        keep_indices = torch.randperm(N)[:num_keep]
        mask_indices = torch.randperm(N)[num_keep:]
        
        x_keep = x[:, keep_indices]
        
        # 编码
        enc_out = self.encoder(x_keep)
        
        # 解码
        dec_input = torch.zeros(B, N, x.shape[2]).to(x.device)
        dec_input[:, keep_indices] = enc_out
        dec_input = dec_input + self.decoder_pos_embed
        
        dec_out = self.decoder(dec_input)
        
        # 重构
        pred = self.head(dec_out)
        
        return pred, keep_indices, mask_indices
    
    def patchify(self, imgs):
        """将图像转为patches"""
        p = self.patch_embed.kernel_size[0]
        return imgs.reshape(imgs.shape[0], 3, imgs.shape[2] // p, p, imgs.shape[3] // p, p).permute(0, 2, 4, 3, 5, 1).reshape(imgs.shape[0], -1, p * p * 3)

# 预训练示例
def pretrain_mae():
    model = MAE(img_size=224, patch_size=16, embed_dim=768, depth=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 模拟数据
    images = torch.randn(8, 3, 224, 224)
    
    model.train()
    pred, keep_idx, mask_idx = model(images)
    
    # 获取真实值
    target = model.patchify(images)
    
    # 计算损失（仅在mask位置）
    loss = F.mse_loss(pred[:, mask_idx], target[:, mask_idx])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"预训练损失: {loss.item():.4f}")

if __name__ == "__main__":
    pretrain_mae()
```

---

## 4. 性能对比

| 模型 | 数据集 | Top-1精度 |
|------|--------|-----------|
| MAE ViT-H | ImageNet | 87.8% |
| BEiT | ImageNet | 86.3% |
| MoCo v3 | ImageNet | 81.0% |

---

## 5. 学习路径

- 前置：ViT, BEiT
- 进阶：MAE变体, 视频MAE