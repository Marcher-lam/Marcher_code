# BEiT（Bidirectional Encoder Representation from Image Transformers）学习文档

> 微软亚研院提出的CV版BERT。

## 1. 算法基础认知

### 一句话定义

BEiT是微软亚洲研究院提出的"CV版BERT"，使用dVAE离散化图像并预测视觉token。

### 历史背景

- **2021年6月**：BEiT v1发布
- **2022年8月**：BEiT v2, v3发布
- **核心创新**：视觉token预测

### 算法定位

BEiT是**CV自监督预训练模型**，属于掩码图像建模（MIM）。

---

## 2. 核心原理

### 两阶段训练

1. **dVAE训练**：学习视觉码本（8192个token）
2. **ViT预训练**：预测mask块的视觉token

### 掩码策略

- 类似BERT，但预测离散token而非像素
- mask比例：40%

### 模型结构

- backbone: ViT
- 输出：预测视觉token分布

---

## 3. 代码实现

```python
import torch
import torch.nn as nn

class BEiT(nn.Module):
    """BEiT模型简化实现"""
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 depth=12, num_heads=12, num_tokens=8192):
        super(BEiT, self).__init__()
        self.num_tokens = num_tokens
        
        # 图像分块嵌入
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        
        # 可学习mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4)
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        # 预测头
        self.head = nn.Linear(embed_dim, num_tokens)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
    def forward(self, x, mask=None):
        B = x.shape[0]
        
        # 分块
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        
        # 添加位置编码
        x = x + self.pos_embed[:, 1:, :]
        
        # 应用mask
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            x = x * (1 - mask) + self.mask_token * mask
            
        # 添加cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer
        x = self.transformer(x)
        
        # 预测mask位置的token
        logits = self.head(x[:, 1:, :])  # 排除cls token
        
        return logits
    
    def patchify(self, x):
        """将图像分为patches"""
        p = self.patch_embed.kernel_size[0]
        return x.reshape(x.shape[0], 3, x.shape[2] // p, p, x.shape[3] // p, p).permute(0, 2, 4, 3, 5, 1).reshape(x.shape[0], -1, p * p * 3)

# 模拟dVAE token化
class Tokenizer:
    """离散tokenizer（简化版）"""
    def __init__(self, vocab_size=8192):
        self.vocab_size = vocab_size
        
    def encode(self, images):
        """将图像转为token IDs（模拟）"""
        B, C, H, W = images.shape
        # 模拟token化
        tokens = torch.randint(0, self.vocab_size, (B, (H//16)*(W//16)))
        return tokens
    
    def decode(self, tokens):
        """token解码为图像（模拟）"""
        # 实际使用dVAE解码器
        return torch.randn(tokens.shape[0], 3, 224, 224)

# 预训练损失
def pretrain_beit():
    model = BEiT(img_size=224, patch_size=16, embed_dim=768, depth=12)
    tokenizer = Tokenizer(vocab_size=8192)
    
    # 模拟输入
    images = torch.randn(4, 3, 224, 224)
    mask = torch.rand(4, 196) > 0.6  # 40% mask
    
    # 前向
    logits = model(images, mask)
    
    # 目标token
    target_tokens = tokenizer.encode(images)
    
    # 交叉熵损失
    loss = nn.functional.cross_entropy(logits[mask], target_tokens[mask])
    
    print(f"BEiT预训练损失: {loss.item():.4f}")

if __name__ == "__main__":
    pretrain_beit()
```

---

## 4. 性能对比

| 模型 | Top-1精度 | 参数量 |
|------|-----------|--------|
| BEiT-Base | 83.2% | 86M |
| BEiT-Large | 86.3% | 304M |
| Supervised ViT-L | 76.5% | 304M |

---

## 5. 学习路径

- 前置：BERT, ViT
- 进阶：BEiT v2/v3, MAE