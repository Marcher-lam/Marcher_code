# MoCo v1/v2/v3 学习文档

> 何恺明团队提出的动量对比学习自监督框架。

## 1. 算法基础认知

### 一句话定义

MoCo系列是基于动量对比的自监督视觉特征学习方法，通过队列机制构建大字典解决对比学习问题。

### 历史背景

- **2019年11月**：MoCo v1发布
- **2020年3月**：MoCo v2发布（改进版）
- **2021年4月**：MoCo v3发布（ViT版）

### 算法定位

MoCo是**自监督对比学习框架**，属于无监督表示学习。

---

## 2. 核心原理

### v1核心设计

1. **双编码器架构**：查询编码器 + 动量编码器
2. **队列机制**：维护大字典（65536个key）
3. **动量更新**：$m=0.999$，缓慢更新

### v3核心改进

- 使用ViT作为backbone
- 冻结patch embedding解决训练不稳定
- 更大的batch（4096+）

---

## 3. 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):
    """MoCo v2简化实现"""
    def __init__(self, feature_dim=128, queue_size=65536, momentum=0.999):
        super(MoCo, self).__init__()
        self.m = momentum
        self.queue_size = queue_size
        
        # 编码器 (查询)
        self.encoder_q = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj_q = nn.Linear(128, feature_dim)
        
        # 编码器 (键) - 动量更新
        self.encoder_k = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj_k = nn.Linear(128, feature_dim)
        
        # 队列
        self.register_buffer("queue", torch.randn(queue_size, feature_dim))
        self.queue_ptr = 0
        
    def forward(self, im_q, im_k):
        # 查询编码
        q = self.encoder_q(im_q).flatten(1)
        q = F.normalize(self.proj_q(q), dim=1)
        
        # 键编码（动量编码器）
        with torch.no_grad():
            # 动量更新
            self._momentum_update()
            
            k = self.encoder_k(im_k).flatten(1)
            k = F.normalize(self.proj_k(k), dim=1)
        
        # 对比损失
        loss = self._contrastive_loss(q, k)
        
        return loss
    
    def _momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.mul_(self.m).add_(p_q.data, alpha=1-self.m)
        for p_q, p_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            p_k.data.mul_(self.m).add_(p_q.data, alpha=1-self.m)
            
    def _contrastive_loss(self, q, k):
        # 正样本相似度
        pos = torch.sum(q * k, dim=1, keepdim=True)
        
        # 负样本相似度
        queue = self.queue.clone().detach()
        neg = torch.matmul(q, queue.T)
        
        # InfoNCE损失
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # 更新队列
        self._dequeue_and_enqueue(k)
        
        return loss
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = self.queue_ptr
        
        self.queue[ptr:ptr+batch_size] = keys
        self.queue_ptr = (ptr + batch_size) % self.queue_size

class MoCoV3(nn.Module):
    """MoCo v3 - 使用ViT"""
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, 
                 feature_dim=128, num_heads=12, depth=12):
        super(MoCoV3, self).__init__()
        
        # ViT编码器
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size//patch_size)**2 + 1, embed_dim))
        
        # 冻结patch embedding解决训练不稳定
        for p in self.patch_embed.parameters():
            p.requires_grad = False
            
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        # 投影头
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, feature_dim)
        )
        
        # 动量版本
        self.head_k = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, feature_dim)
        )
        self.head_k.load_state_dict(self.head.state_dict())
        for p in self.head_k.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        B = x.shape[0]
        
        # ViT编码
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        
        # cls token特征
        feat = x[:, 0]
        
        # 投影
        q = self.head(feat)
        q = F.normalize(q, dim=1)
        
        return q

if __name__ == "__main__":
    # 测试
    moco = MoCo(feature_dim=128)
    im_q = torch.randn(4, 3, 224, 224)
    im_k = torch.randn(4, 3, 224, 224)
    loss = moco(im_q, im_k)
    print(f"MoCo损失: {loss.item():.4f}")
```

---

## 4. 性能对比

| 模型 | ImageNet Top-1 | 参数量 |
|------|---------------|--------|
| MoCo v1 | 60.6% | - |
| MoCo v2 | 67.1% | - |
| MoCo v3 (ViT-B) | 76.7% | 86M |
| MoCo v3 (ViT-L) | 81.0% | 304M |

---

## 5. 学习路径

- 前置：对比学习, SimCLR
- 进阶：DINO, BYOL