# Vision Transformer (ViT) 学习文档

> 来源线索：本节内容根据原书中关于"ViT模型"和"Patch和Position Embedding"（第5章 5.5.1-5.5.2节）的相关章节整理、扩展与教学化改写。

> 用Transformer处理图像——将图片切成小块，像处理文本序列一样理解视觉信息。

## 1. 算法基础认知

**一句话定义**：ViT将图像分割为固定大小的patch序列，用标准Transformer编码器进行图像分类。

**直觉类比**：想象你在看一幅拼图。传统CNN是逐块扫描（局部感受野），而ViT是把所有拼图块铺在桌面上，同时看所有块之间的关系（全局注意力）。每个拼图块就是一个"token"，整个Transformer就像一个聪明的观察者，同时考虑所有拼图块来理解整幅画。

**历史背景**：ViT由Dosovitskiy等人在2020年的论文"An Image is Worth 16x16 Words"中提出。这是注意力机制在图像识别领域的开创性应用，证明了纯Transformer架构在大规模数据上可以超越CNN。

**算法定位**：深度学习 / 计算机视觉 / Transformer架构。ViT是视觉领域的Transformer基础模型。

**前置知识**：
- 自注意力机制
- Transformer编码器架构
- 图像基础（像素、通道、卷积）
- 位置编码

## 2. 核心原理

### 核心思想

ViT的核心创新是将图像转化为"视觉token序列"，然后用标准Transformer处理：

1. **Patch分割**：将图像切分为固定大小的patch（如16×16），每个patch相当于NLP中的一个"词"
2. **线性嵌入**：将每个patch展平并通过线性层投影为固定维度向量
3. **位置编码**：添加可学习的位置嵌入，保留空间信息
4. **分类token**：添加一个特殊的[CLS] token用于图像分类
5. **Transformer编码**：通过多层Transformer编码器处理patch序列

### 工作流程

```
输入图像 (224×224×3)
    ↓ Patch分割
14×14 = 196个patch (每个16×16×3)
    ↓ 展平 + 线性投影
196个token (每个768维)
    ↓ 添加[CLS] token + 位置编码
197个token (每个768维)
    ↓ L层Transformer Encoder
197个token (每个768维)
    ↓ 取[CLS] token
分类输出
```

### 关键概念

- **Patch Embedding**：通过卷积（kernel=stride=patch_size）实现高效分块和投影
- **[CLS] Token**：可学习的特殊token，其最终表示用于分类
- **位置编码**：可学习的1D位置嵌入，添加到patch token上
- **与CNN的区别**：ViT从第一层就有全局感受野，CNN需要多层堆叠才能获得全局视野

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $H, W, C$ | 图像高、宽、通道数 | 224, 224, 3 |
| $P$ | patch大小 | 16 |
| $N$ | patch数量 | $HW/P^2 = 196$ |
| $d$ | 嵌入维度 | 768 |
| $L$ | Transformer层数 | 12 |

### Patch Embedding

将图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ 分割为 $N = HW/P^2$ 个patch：

$$\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}, \quad i = 1, 2, ..., N$$

每个patch通过线性投影映射到 $d$ 维：

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; ...; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$$

其中 $\mathbf{E} \in \mathbb{R}^{P^2 C \times d}$ 是投影矩阵，$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times d}$ 是位置编码。

### Transformer编码器

$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}, \quad l = 1, ..., L$$

$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$

### 分类头

取[CLS] token的最终表示通过线性层分类：

$$\hat{y} = \text{Linear}(\mathbf{z}_L^0)$$

## 4. 训练过程讲解

### 数据预处理

- 图像resize到固定大小（通常224×224）
- 像素值归一化到[0,1]或标准化（ImageNet均值/方差）
- 数据增强：随机裁剪、水平翻转、颜色抖动等

### 参数初始化

- Patch投影层使用截断正态初始化
- 位置编码初始化为零（可学习）
- [CLS] token初始化为零
- Transformer层使用标准初始化

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| patch_size | 每个patch的大小 | 8/16/32 | 16 |
| embed_dim | 嵌入维度 | 384/768/1024 | 768 |
| depth | Transformer层数 | 6/12/24 | 12 |
| num_heads | 注意力头数 | 6/12/16 | 12 |
| MLP比例 | FFN扩展比 | 2-4 | 4 |

### ViT变体

| 模型 | embed_dim | depth | heads | 参数量 |
|------|-----------|-------|-------|--------|
| ViT-Tiny | 192 | 12 | 3 | 5.7M |
| ViT-Small | 384 | 12 | 6 | 22M |
| ViT-Base | 768 | 12 | 12 | 86M |
| ViT-Large | 1024 | 24 | 16 | 307M |

## 5. 应用场景

1. **图像分类**：ViT的原始用途。在大规模数据集（如ImageNet-21k或JFT-300M）上预训练后，迁移到下游分类任务。

2. **目标检测**：ViT作为骨干网络（如DETR、ViTDet），提供全局特征表示。

3. **多模态大模型**：DeepSeek-VL2等模型使用ViT作为视觉编码器，将图像转为token序列送入LLM。

4. **语义分割**：将ViT的patch级特征上采样恢复为像素级分割图（如SegViT）。

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 全局感受野，从第一层就能捕获长距离依赖 | 小数据集上容易过拟合（缺乏归纳偏置） |
| 架构统一，NLP和CV使用同一架构 | 计算量随patch数量平方增长 |
| 预训练后迁移学习效果好 | 需要大规模数据预训练才能超越CNN |
| 可扩展性强，模型越大性能越好 | 推理时显存占用大（attention map是 $N^2$） |

**与CNN对比**：

| 特性 | CNN | ViT |
|------|-----|-----|
| 归纳偏置 | 强（平移不变性、局部性） | 弱（几乎无先验） |
| 感受野 | 从局部到全局 | 全局 |
| 数据需求 | 较少 | 较多 |
| 扩展性 | 受限 | 优异 |
| 可解释性 | 特征图可视化 | 注意力图可视化 |

## 7. 调库实现

```python
"""使用 PyTorch 实现 Vision Transformer (ViT)"""
import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """将图像分割为patch并线性投影"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积实现分块+投影（高效且可微分）
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (batch, C, H, W)
        x = self.proj(x)           # (batch, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)      # (batch, num_patches, embed_dim)
        x = self.norm(x)
        return x


class ViT(nn.Module):
    """Vision Transformer 完整实现"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN（ViT使用Pre-LN）
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        batch = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # 添加[CLS] token
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, embed_dim)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer编码
        x = self.encoder(x)
        x = self.norm(x)
        
        # 取[CLS] token进行分类
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 创建ViT-Base模型
    model = ViT(
        img_size=224, patch_size=16, in_channels=3,
        num_classes=10, embed_dim=768, depth=12, num_heads=12
    )
    
    # 模拟输入
    images = torch.randn(2, 3, 224, 224)
    output = model(images)
    
    print("=== ViT 测试 ===")
    print(f"输入: {images.shape}")
    print(f"输出: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Patch数量: {model.patch_embed.num_patches}")
    print(f"序列长度（含CLS）: {model.patch_embed.num_patches + 1}")
```

## 8. 手工代码实现

```python
"""从零实现ViT（不使用nn.TransformerEncoder，手动实现所有组件）"""
import torch
import torch.nn as nn
import math


class ManualPatchEmbed(nn.Module):
    """手写Patch Embedding（不使用Conv2d，用基础张量操作）"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        
        # 线性投影
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (batch, C, H, W)
        batch, C, H, W = x.shape
        P = self.patch_size
        
        # 手动分块: (batch, C, H, W) -> (batch, num_patches, patch_dim)
        # 重塑为 (batch, C, H/P, P, W/P, P)
        x = x.reshape(batch, C, H // P, P, W // P, P)
        # 调换维度: (batch, H/P, W/P, P, P, C)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # 展平: (batch, H/P, W/P, P*P*C)
        x = x.reshape(batch, self.num_patches, -1)
        
        # 线性投影
        x = self.proj(x)
        x = self.norm(x)
        return x


class ManualMultiHeadAttention(nn.Module):
    """手写多头自注意力"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(out)


class ManualTransformerBlock(nn.Module):
    """手写Transformer编码器块（Pre-LN）"""
    
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ManualMultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ManualViT(nn.Module):
    """手写Vision Transformer"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=10, embed_dim=384, depth=6,
                 num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = ManualPatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # 堆叠Transformer块
        self.blocks = nn.ModuleList([
            ManualTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        batch = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 小型ViT用于测试
    model = ManualViT(
        img_size=64, patch_size=8, in_channels=3,
        num_classes=5, embed_dim=192, depth=4, num_heads=6
    )
    
    images = torch.randn(2, 3, 64, 64)
    output = model(images)
    
    print("=== 手写ViT测试 ===")
    print(f"输入: {images.shape}")
    print(f"Patch数量: {model.patch_embed.num_patches}")
    print(f"输出: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 验证训练可行性
    labels = torch.tensor([0, 3])
    loss = nn.CrossEntropyLoss()(output, labels)
    loss.backward()
    print(f"损失: {loss.item():.4f}")
    print("反向传播成功!")
```

## 9. 可视化与结果理解

```python
"""ViT可视化：Patch分割和注意力图"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: Patch分割示意
img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
patch_size = 32  # 用更大的patch便于可视化

axes[0].imshow(img)
# 绘制patch网格
for i in range(0, 224, patch_size):
    axes[0].axhline(y=i, color='white', linewidth=0.5)
    axes[0].axvline(x=i, color='white', linewidth=0.5)
axes[0].set_title(f'Patch分割 (P={patch_size}, 共{(224//patch_size)**2}个patch)', fontsize=13)
axes[0].set_xlabel('W')
axes[0].set_ylabel('H')

# 图2: [CLS] token的注意力图（模拟）
num_patches = 14 * 14
np.random.seed(42)
# 模拟CLS token对每个patch的注意力（中心区域更受关注）
attn_map = np.zeros((14, 14))
for i in range(14):
    for j in range(14):
        dist = np.sqrt((i-7)**2 + (j-7)**2)
        attn_map[i][j] = np.exp(-dist/5) + np.random.rand() * 0.1
attn_map = attn_map / attn_map.sum()

import seaborn as sns
sns.heatmap(attn_map, cmap='YlOrRd', ax=axes[1],
            xticklabels=[], yticklabels=[], cbar_kws={'label': '注意力权重'})
axes[1].set_title('[CLS] Token对各Patch的注意力', fontsize=13)

# 图3: ViT vs CNN感受野对比
# CNN: 感受野随层数增长
cnn_rf = [3, 5, 9, 17, 33, 65, 129]  # 3x3卷积堆叠的感受野
layers = list(range(1, len(cnn_rf) + 1))

axes[2].plot(layers, cnn_rf, 'b-o', label='CNN (3×3卷积)', linewidth=2)
axes[2].axhline(y=224, color='r', linestyle='--', alpha=0.5, label='ViT全局感受野')
axes[2].fill_between(layers, 0, cnn_rf, alpha=0.1, color='blue')
axes[2].set_title('感受野对比: CNN vs ViT', fontsize=13)
axes[2].set_xlabel('网络层数')
axes[2].set_ylabel('感受野大小（像素）')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vit_viz.png', dpi=100)
plt.show()

print("图1解读: 图像被分割为固定大小的patch, 每个patch相当于一个token")
print("图2解读: [CLS] token通过注意力机制聚合所有patch的信息, 中心区域通常获得更多关注")
print("图3解读: CNN需要多层堆叠才能获得全局感受野, ViT从第一层就是全局的")
```

## 10. 模型评估

```python
"""ViT模型评估"""
def evaluate_vit(model, dataloader, device='cpu'):
    """在测试集上评估ViT"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"测试准确率: {accuracy:.2f}%")
    return accuracy

def compute_attention_entropy(model, images):
    """计算注意力图的熵，评估注意力集中度"""
    # 注册hook获取注意力权重
    attn_weights = []
    
    def hook_fn(module, input, output):
        # 对于手写模型，可以在forward中返回注意力
        pass
    
    print("注意力熵越低 → 注意力越集中 → 模型更聚焦")
    print("ViT的浅层注意力通常更分散（学习局部特征）")
    print("深层注意力更集中（关注语义相关区域）")
```

## 11. 常见问题与易错点

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 图像大小与patch不匹配 | 运行时报错 | 图像尺寸不能被patch_size整除 | resize到匹配的尺寸（如224） |
| 小数据集过拟合 | 训练精度高但测试差 | ViT缺乏CNN的归纳偏置 | 使用预训练权重+微调，或增加数据增强 |

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 显存不足 | OOM错误 | 注意力是 $N^2$ 复杂度 | 减小图像分辨率或增大patch_size |
| 训练不稳定 | 损失震荡/NaN | 大学习率+无预训练 | 使用较小的学习率，加warmup |

### 调参层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| patch_size选择 | 精度和速度不理想 | 太小token多计算慢，太大信息丢失 | 分类用16，密集预测用8 |

## 12. 学习总结

ViT证明了纯Transformer架构可以应用于图像理解，核心公式：

$$\mathbf{z}_0 = [\mathbf{x}_{cls}; \mathbf{x}_p^1 \mathbf{E}; ...; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$$

$$\mathbf{z}_l = \text{MLP}(\text{LN}(\text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1})) + \mathbf{z}'_l$$

ViT的关键创新在于将视觉问题转化为序列问题，使得NLP和CV可以使用统一架构。后续的多模态大模型（如DeepSeek-VL2、CLIP）都以ViT作为视觉编码器。

## 13. 练习题与思考题

### 基础题1：Patch数量计算

输入图像大小为384×384×3，patch_size为16。计算：patch数量、每个patch展平后的维度、如果embed_dim=768，Patch Embedding层的参数量。

**参考答案**：
- patch数量 = (384/16) × (384/16) = 24 × 24 = 576
- 每个patch展平维度 = 16 × 16 × 3 = 768
- Patch Embedding参数量 = 768 × 768 = 589,824（线性投影矩阵）

### 基础题2：序列长度与计算量

ViT-Base（patch_size=16, img_size=224）的自注意力计算量是多少？如果img_size增加到448，计算量增加多少倍？

**参考答案**：
- 序列长度 N = (224/16)² = 196
- 自注意力计算量 ∝ N² = 196² = 38,416
- img_size=448时，N = (448/16)² = 784，计算量 = 784² = 614,656
- 增加倍数 = 614,656 / 38,416 = 16倍（分辨率翻倍，计算量增加16倍）

### 进阶题：ViT与多模态

在DeepSeek-VL2中，ViT的输出（patch token序列）如何与文本token结合？有哪些挑战？

**参考答案**：
1. ViT输出196个768维的视觉token
2. 通过线性层投影到与文本相同的维度
3. 与文本token拼接为一个序列
4. 挑战：
   - 视觉token数量远多于文本token（196 vs ~50），导致序列过长
   - 解决方案：使用Token压缩（如AvgPool将196压缩到36-64个）
   - 视觉和文本的语义空间不同，需要投影和对齐

### 开放思考题

ViT在patch内是完全局部处理的（只是展平+投影），这是否意味着ViT丢失了patch内部的像素级空间关系？如何改进？

**参考思路**：
- ViT确实丢失了patch内部的空间结构，每个patch被视为一个整体
- 改进方案：
  1. 使用更小的patch_size（如8或4），但会增加计算量
  2. 在patch内部使用小型CNN提取子特征
  3. 使用层次化Transformer（如Swin Transformer），在局部窗口内计算注意力
  4. 使用多尺度patch（不同大小的patch捕获不同粒度的信息）

## 14. 学习路径建议

### 前置算法
- 自注意力机制
- Transformer编码器
- 位置编码

### 平行学习
- CNN架构（ResNet等）—— 理解ViT相比CNN的优劣
- Swin Transformer —— 层次化视觉Transformer

### 进阶方向
- V-MoE（ViT + MoE）
- 多模态大模型中的视觉编码器（CLIP、DeepSeek-VL2）
- 视频ViT（时空注意力）

### 推荐资源
1. **论文**：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020)
2. **代码**：timm库（PyTorch Image Models）中的ViT实现
3. **论文**：Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Liu et al., 2021)
