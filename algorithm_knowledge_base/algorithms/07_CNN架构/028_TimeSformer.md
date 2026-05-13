# TimeSformer 时间空间Transformer 学习文档

> 首个纯Transformer视频理解模型

---

## 1. 算法基础认知

### 1.1 一句话定义

TimeSformer是Facebook AI于2021年提出的纯Transformer视频理解模型，首次将Transformer成功应用于视频分类，取代了传统的3D CNN！

### 1.2 直觉类比

TimeSformer就像一个"全局感知的视频分析员"。传统的CNN需要逐帧处理，看到的是局部；而TimeSformer能同时"看"整个视频的所有帧——既知道这一帧发生了什么（空间注意力），也知道帧与帧之间的变化（时间注意力）。

想象你要分析一段篮球比赛视频：
- 传统方法：逐帧看，看完第100帧才能理解整个动作
- TimeSformer：同时看全部100帧，通过注意力机制自动发现"投篮动作"跨越多帧的关系！

### 1.3 发展背景

- 2021年3月，Facebook AI的Bertasius等人在论文"Space-Time Transformer"中提出
- 首次将Transformer成功应用于视频理解
- 在Kinetics-400上达到SOTA水平
- 开创视频理解新范式！随后有Video ViT、Swin Transformer等

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 视频理解 → 动作识别 |
| 输出 | 视频分类 |
| 模型 | 纯Transformer |
| 特点 | 联合时空注意力 |
| 计算 | O(N²T²) → O(T×N²) |

---

## 2. 核心原理

### 2.1 为什么需要视频Transformer？

**传统3D CNN的问题**：
- 感受野受限于卷积核大小
- 需要大量3D卷积堆叠才能捕获长程依赖
- 计算密集，参数多

**Transformer的优势**：
- 全局自注意力，任意位置直接建模关系
- 更高效的参数利用
- 并行计算

### 2.2 视频表示

将视频看作**时空立方体**：

$$x \in \mathbb{R}^{C \times T \times H \times W}$$

其中：
- T = 时间帧数
- H, W = 空间分辨率
- C = 通道数

### 2.3 时空注意力机制

TimeSformer使用三种注意力：

| 类型 | 说明 | 计算复杂度 |
|------|------|-----------|
| 空间注意力 | 每帧内部自注意力 | O(N²) per frame |
| 时间注意力 | 跨帧自注意力 | O(T²) per position |
| 联合时空 | 同时考虑时空 | O(T²N²) |

### 2.4 架构流程

```
输入视频 [C, T, H, W]
    │
    ▼
线性投影 + 位置编码
    │
    ▼
分割为T帧 [T, N] (N=H×W)
    │
    ▼
┌─ 空间注意力 ─┐
│               │
├─ 时间注意力 ─┤
│               │
├─ 联合注意力 ─┘
    │
    ▼
MLP + 残差连接
    │
    ▼
分类头 → 输出类别
```

---

## 3. 数学公式与推导

### 3.1 位置编码

时空位置编码需要同时编码空间位置和时间位置：

$$PE_{(t,h,w),2i} = \sin(\frac{pos}{10000^{2i/d}})$$
$$PE_{(t,h,w),2i+1} = \cos(\frac{pos}{10000^{2i/d}})$$

其中 $pos = t \cdot H \cdot W + h \cdot W + w$

### 3.2 空间注意力

对于第t帧：

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

$Q, K, V \in \mathbb{R}^{N \times d}$，其中 $N = H \times W$

### 3.3 时间注意力

固定空间位置，跨帧计算：

$$Attention_{h,w}(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

$Q, K, V \in \mathbb{R}^{T \times d}$

### 3.4 联合时空注意力

同时考虑所有时空位置：

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

$Q, K, V \in \mathbb{R}^{T \cdot N \times d}$

### 3.5 复杂度分析

| 注意力类型 | 空间 | 时间 | 总复杂度 |
|-----------|------|------|--------|
| 空间 | O(T×N²) | - | O(T×N²) |
| 时间 | - | O(N×T²) | O(N×T²) |
| 联合 | O(T²×N²) | - | O(T²×N²) |

---

## 4. 训练过程讲解

### 4.1 模型配置

```python
# TimeSformer配置
config = {
    'img_size': 224,
    'num_frames': 8,
    'num_classes': 400,
    'attention_type': 'divided_space_time',  # 分开时空
    'drop_rate': 0.0,
    'attn_drop_rate': 0.0,
    'drop_path_rate': 0.1,
}
```

### 4.2 训练参数

| 参数 | 建议值 |
|------|--------|
| batch_size | 8-16 |
| lr | 1e-4 |
| weight_decay | 0.05 |
| epochs | 30-100 |
| warmup_epochs | 5 |

### 4.3 数据增强

```python
# 训练增强
train_aug = Compose([
    Resize(256),
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ColorJitter(),
    Normalize()
])
```

---

## 5. 应用场景

### 5.1 视频动作识别

主要应用场景：

```python
# 动作识别
video = torch.randn(1, 3, 8, 224, 224)  # T=8帧
output = model(video)
pred = output.argmax(dim=-1)
```

### 5.2 视频分类

| 数据集 | Top-1 |
|--------|-------|
| Kinetics-400 | 80.7% |
| Kinetics-600 | 77.8% |
| EPIC-KITCHENS | 47.5% |

### 5.3 行为检测

```python
# 时序动作检测
features = model(video)
start_logits = model.start_head(features)
end_logits = model.end_head(features)
segments = non_max_suppression(start_logits, end_logits)
```

### 5.4 对比其他方法

| 方法 | K400 | 计算量 |
|------|------|-------|
| I3D | 74.3% | 高 |
| SlowFast | 78.8% | 中 |
| Non-local | 77.7% | 高 |
| **TimeSformer** | **80.7%** | 高 |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 全局感受野 | 任意帧关系可直接建模 |
| 可解释 | 注意力可视化 |
| 并行计算 | GPU高效 |
| 精度高 | SOTA水平 |
| 灵活 | 可分离/联合注意力 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 显存消耗大 | O(N²T²) |
| 计算量大 | 长视频困难 |
| 长序列问题 | T增大时平方增长 |

### 6.3 注意事项

- 建议T≤16，否则显存爆炸
- 可用divided_space_time类型节省计算
- 需要强大的数据增强

---

## 7. 调库实现（Python）

### 7.1 安装

```bash
pip install timesformer
```

### 7.2 基本用法

```python
import torch
from timesformer.models.vit import TimeSformer

# 加载模型
model = TimeSformer(
    img_size=224,
    num_classes=400,
    num_frames=8,
    attention_type='divided_space_time',
    pretrained=True
)

# 输入：[B, C, T, H, W]
video = torch.randn(1, 3, 8, 224, 224)

# 前向传播
model.eval()
with torch.no_grad():
    output = model(video)

print(f"输出: {output.shape}")  # [1, 400]
```

### 7.3 训练示例

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# 模型
model = TimeSformer(
    img_size=224, 
    num_classes=400, 
    num_frames=8,
    attention_type='divided_space_time'
)

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# 损失
criterion = torch.nn.CrossEntropyLoss()

# 训练
for epoch in range(30):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        videos, labels = batch
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss={total_loss/len(dataloader):.4f}")
```

### 7.4 注意力可视化

```python
# 提取注意力权重
def get_attention(model, video):
    model.eval()
    
    # 中间层注意力
    attns = []
    
    def hook_fn(module, input, output):
        attns.append(output)
    
    # 注册hook
    for block in model.blocks:
        block.attn.attn_dropout.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(video)
    
    return attns
```

---

## 8. 手工代码实现（理解原理）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpaceTimeAttention(nn.Module):
    """时空注意力 - 简化版"""
    def __init__(self, dim, num_heads=8, attention_type='divided'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_type = attention_type
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, T=None):
        """
        x: [B, T*N, d] 或 [B, T, N, d]
        """
        B, L, d = x.shape
        
        # 空间注意力和时间注意力
        if self.attention_type == 'divided':
            # 分开计算
            # x shape: [B, T, N, d]
            x = x.view(B, T, -1, d)
            
            # 空间注意力: 对每个t帧
            x_space = x.view(B * T, -1, d)
            qkv = self.qkv(x_space).reshape(B * T, -1, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            attn_s = (q @ k.transpose(-2, -1)) * self.scale
            attn_s = attn_s.softmax(dim=-1)
            x_s = (attn_s @ v).reshape(B, T, -1, d)
            
            # 时间注意力: 对每个空间位置
            x_time = x.permute(0, 2, 1, 3).reshape(B * L, T, d)
            qkv = self.qkv(x_time).reshape(B * L, T, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            attn_t = (q @ k.transpose(-2, -1)) * self.scale
            attn_t = attn_t.softmax(dim=-1)
            x_t = (attn_t @ v).reshape(B, L, T, d).permute(0, 2, 1, 3)
            
            # 合并
            x = x_s + x_t
            x = x.reshape(B, -1, d)
        
        else:
            # 联合注意力
            qkv = x.reshape(B, -1, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).reshape(B, L, d)
        
        x = self.proj(x)
        return x


class TimeSformerBlock(nn.Module):
    """TimeSformer块"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpaceTimeAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TimeSformerSimple(nn.Module):
    """简化版TimeSformer"""
    def __init__(self, img_size=224, num_frames=8, num_classes=400, depth=12, dim=768, num_heads=12):
        super().__init__()
        
        # 嵌入层
        self.to_patch_embedding = nn.Conv3d(3, dim, kernel_size=(1, 16, 16), stride=(1, 16, 16))
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, (img_size//16)**2, dim))
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TimeSformerBlock(dim, num_heads)
            for _ in range(depth)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, video):
        # video: [B, C, T, H, W]
        B = video.shape[0]
        
        # 嵌入
        x = self.to_patch_embedding(video)  # [B, d, T, H', W']
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H', W', d]
        x = x.reshape(B, -1, x.shape[-1])  # [B, T*N, d]
        
        # 位置编码
        x = x + self.pos_embedding
        
        # Transformer
        for block in self.blocks:
            x = block(x)
        
        # 分类
        x = self.norm(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.head(x)
        
        return x


# 测试
if __name__ == "__main__":
    model = TimeSformerSimple(
        img_size=224,
        num_frames=8,
        num_classes=400,
        depth=12,
        dim=768,
        num_heads=12
    )
    
    video = torch.randn(1, 3, 8, 224, 224)
    output = model(video)
    
    print(f"输入: {video.shape}")
    print(f"输出: {output.shape}")
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数: {total_params/1e6:.1f}M")
```

---

## 9. 可视化与结果理解

### 9.1 注意力可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 可视化时空注意力
def visualize_attention(attn_weights, num_frames=8, img_size=224):
    """
    可视化注意力权重
    attn_weights: [T, N, N]
    """
    N = img_size // 16
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for t in range(num_frames):
        ax = axes[t // 4, t % 4]
        attn = attn_weights[t].reshape(N, N)
        im = ax.imshow(attn, cmap='viridis')
        ax.set_title(f'Frame {t}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('timesformer_attn.png', dpi=100)
    plt.show()
```

### 9.2 特征可视化

```python
# t-SNE可视化特征
from sklearn.manifold import TSNE

# 提取特征
features = []
labels_list = []

for video, label in dataloader:
    with torch.no_grad():
        feat = model(video)
        features.append(feat)
        labels_list.append(label)

features = torch.cat(features)
labels = torch.cat(labels_list)

# t-SNE
tsne = TSNE(n_components=2)
features_2d = tsne.fit_transform(features)

# 绘图
plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20')
plt.title('TimeSformer特征可视化')
plt.savefig('timesformer_tsne.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| Top-1 | 最高概率正确 |
| Top-5 | 前5正确 |
| FLOPs | 计算量 |
| 显存 | GPU显存 |

### 10.2 评估代码

```python
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in dataloader:
            outputs = model(videos)
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum()
            total += len(labels)
    
    return correct / total

accuracy = evaluate(model, test_loader)
print(f"Accuracy: {accuracy:.2%}")
```

---

## 11. 常见问题与易错点

### Q1: 显存不足？

**答案**：减少num_frames或img_size，或用divided_space_time类型。

### Q2: 训练不稳定？

**答案**：用较小的学习率，加warmup。

### Q3: 和SlowFast比较？

**答案**：TimeSformer精度更高，但计算量更大。

### Q4: 如何处理长视频？

**答案**：分段处理后拼接特征。

### Q5: 需要预训练？

**答案**：建议用ImageNet预训练的2D ViT初始化。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心 | 时空自注意力 |
| 类型 | divided/joint |
| 输入 | [C, T, H, W] |
| 输出 | 类别分数 |

### 12.2 公式汇总

空间注意力：
$$Attn_s = softmax(\frac{Q_s K_s^T}{\sqrt{d}})V_s$$

时间注意力：
$$Attn_t = softmax(\frac{Q_t K_t^T}{\sqrt{d}})V_t$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. TimeSformer的核心创新是：
   - A) 3D卷积
   - B) 时空注意力
   - C) 残差连接

2. 什么注意力类型节省计算：
   - A) joint
   - B) divided
   - C) both

### 13.2 简答题

1. 为什么TimeSformer比3D CNN精度高？
2. divided和joint attention的区别？

### 13.3 编程题

1. 实现时空位置编码。
2. 比较不同注意力类型的效果。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
Transformer基础
    ↓
ViT图像理解
    ↓
视频理解
    ↓
TimeSformer
    ↓
Swin Transformer
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| ViT | 空间版 |
| Video ViT | 视频版 |
| Swin | 分层版 |
| MViT | 轻量版 |

### 14.3 扩展阅读

- Bertasius et al. (2021). Space-Time Transformer. arXiv:2103.15691

---

## 附录

### 参考

1. Bertasius et al. (2021). Space-Time Transformer. arXiv:2103.15691
2. https://github.com/facebookresearch/TimeSformer

---

**文档结束**