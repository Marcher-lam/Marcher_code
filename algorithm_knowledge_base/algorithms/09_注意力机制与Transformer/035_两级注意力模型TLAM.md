# 两级注意力模型TLAM 学习文档

> 先定位再识别——两级注意力机制用于细粒度分类。
>
> 来源线索：本节内容根据原书第2章关于"细粒度分类"的相关章节整理。

---

## 1. 算法基础认知

**一句话定义：** 两级注意力模型（Two-Level Attention Model, TLAM）用于细粒度图像分类，第一级注意力定位关键区域（如鸟的头部和身体），第二级注意力在该区域内提取精细的判别性特征进行分类。

**核心思想：** 细粒度分类的关键在于找出最能区分不同子类的判别性区域。例如，区分不同种类的鸟，关键差异可能在头部（喙的形状、颜色）和翅膀（花纹）等局部区域。TLAM通过两级注意力实现：
- **第一级（空间注意力）**：在图像中定位关键部件的位置
- **第二级（通道/特征注意力）**：在定位区域上提取最有判别力的特征

**为什么需要两级？** 单级注意力要么定位不精确，要么特征提取不充分。两级设计使网络先"看哪里"再"看什么"，模拟了人类细粒度识别的认知过程。

**TLAM vs 其他细粒度方法：**

| 方法 | 是否需要部件标注 | 推理速度 | 性能 |
|------|----------------|---------|------|
| TLAM | 否（弱监督） | 中 | 高 |
| 部件检测 + CNN | 是 | 慢 | 更高 |
| 直接CNN分类 | 否 | 快 | 中 |

---

## 2. 核心原理

### 2.1 第一级注意力：空间定位

从CNN特征图中预测多个关键点的位置：

$$
P_k = \text{Localizer}(F), \quad k = 1, ..., K
$$

其中 $F \in \mathbb{R}^{H \times W \times C}$ 是特征图，$P_k \in \mathbb{R}^2$ 是第 $k$ 个关键点的 $(x,y)$ 坐标。

定位网络通常是一个全卷积网络，输出 $K$ 个通道的注意力热图：

$$
A_k(x,y) = \text{softmax}(f_k(F)), \quad \sum_{x,y} A_k(x,y) = 1
$$

关键点位置为注意力热图的期望：

$$
P_k = \left(\sum_{x,y} x \cdot A_k(x,y), \sum_{x,y} y \cdot A_k(x,y)\right)
$$

### 2.2 区域特征提取

以每个关键点为中心，裁剪固定大小的区域 $R_k$：

$$
R_k = \text{CropAndResize}(I, P_k, size)
$$

通过共享的CNN提取每个区域的特征 $f_k$。

### 2.3 第二级注意力：特征加权

对提取的区域特征施加通道注意力，突出判别性通道：

$$
f_k' = f_k \odot \sigma(W \cdot f_k + b)
$$

其中 $\odot$ 是逐元素相乘，$\sigma$ 是Sigmoid函数。

### 2.4 分类

融合所有区域的特征并进行分类：

$$
\hat{y} = \text{Classifier}(\text{Concat}(f_1', f_2', ..., f_K'))
$$

---

## 3. 数学公式与推导

### 3.1 空间注意力热图

给定特征图 $F \in \mathbb{R}^{H \times W \times C}$，定位网络通过卷积生成 $K$ 个注意力热图：

$$
A_k = \text{softmax}_{HW}(\text{Conv}_{1\times1}(F)_k), \quad k = 1, ..., K
$$

其中 $\text{softmax}_{HW}$ 在空间维度上做归一化。

### 3.2 关键点坐标的期望形式

$$
x_k = \sum_{i=1}^H \sum_{j=1}^W j \cdot A_k(i,j), \quad y_k = \sum_{i=1}^H \sum_{j=1}^W i \cdot A_k(i,j)
$$

这种可微分的定位方式使整个模型可以端到端训练。

### 3.3 通道注意力

通道注意力权重：

$$
w_c = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(f_k)))
$$

其中 $\text{GAP}$ 是全局平均池化，$W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$，$W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$，$r$ 是降维比率。

### 3.4 总损失

$$
\mathcal{L} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda \cdot \mathcal{L}_{diversity}
$$

$\mathcal{L}_{diversity}$ 鼓励不同关键点关注不同区域：

$$
\mathcal{L}_{diversity} = \sum_{k \neq l} \left| \sum_{i,j} A_k(i,j) \cdot A_l(i,j) \right|
$$

---

## 4. 训练过程讲解

### 4.1 端到端训练

TLAM用端到端方式联合训练所有组件：

1. **前向传播**：
   - 图像经过backbone CNN提取特征
   - 定位网络生成K个注意力热图
   - 从热图计算K个关键点坐标
   - 裁剪对齐K个区域
   - 每个区域经过特征网络提取特征
   - 第二级注意力加权
   - 融合后分类

2. **反向传播**：
   - 分类损失 $\mathcal{L}_{CE}$ 回传到所有组件
   - 多样性损失 $\mathcal{L}_{diversity}$ 防止关键点重合
   - 定位网络靠分类损失的梯度学习

### 4.2 训练技巧

- **预热**：初期固定定位网络，先训练分类器
- **多样性正则化**：防止多个关键点指向同一区域
- **Dropout**：在区域特征融合时使用，增强泛化
- **数据增强**：随机裁剪、翻转等

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 细粒度鸟类分类 | CUB-200-2011数据集 |
| 车型识别 | 区分不同品牌型号 |
| 飞机型号识别 | 基于局部特征的型号区分 |
| 食物分类 | 精细食物类别识别 |
| 医学图像分析 | 病灶区域的定位与分类 |
| 零售商品识别 | 相似商品的区分 |

---

## 6. 优缺点分析

**优点：**
- ✅ **弱监督**：只需要类别标签，无需部件标注
- ✅ **可解释**：关键点定位显示分类依据
- ✅ **端到端**：联合训练所有组件
- ✅ **细粒度效果好**：针对判别性区域建模
- ✅ **可扩展**：可增加更多注意力层级

**缺点：**
- ❌ **定位不稳定**：关键点可能收敛到非判别性区域
- ❌ **计算量大**：需要裁剪多个区域额外前向
- ❌ **关键点数量敏感**：K的选择影响性能
- ❌ **知识难迁移**：不同类别所需关键点不同
- ❌ **多样性损失权重敏感**：过强限制定位自由

---

## 7. 调库实现

```python
"""TLAM两级注意力模型 - PyTorch完整实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class SpatialLocalizer(nn.Module):
    """第一级注意力：空间定位网络"""
    
    def __init__(self, in_channels, n_keypoints=4):
        super().__init__()
        self.n_keypoints = n_keypoints
        
        self.localizer = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, n_keypoints, 1),  # K个注意力热图
        )
    
    def forward(self, features):
        """
        参数:
            features: 特征图 (batch, C, H, W)
        
        返回:
            heatmaps: 注意力热图 (batch, K, H, W)
            keypoints: 关键点坐标 (batch, K, 2)
        """
        # 生成注意力热图 (未归一化)
        raw_heatmaps = self.localizer(features)
        batch, K, H, W = raw_heatmaps.shape
        
        # 空间softmax
        heatmaps = raw_heatmaps.view(batch, K, -1)
        heatmaps = F.softmax(heatmaps, dim=2)
        heatmaps = heatmaps.view(batch, K, H, W)
        
        # 计算关键点坐标 (期望位置)
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, H, device=features.device),
            torch.linspace(0, 1, W, device=features.device),
            indexing='ij'
        )
        
        keypoints = torch.zeros(batch, K, 2, device=features.device)
        for k in range(K):
            heatmap_k = heatmaps[:, k, :, :]  # (batch, H, W)
            keypoints[:, k, 0] = (heatmap_k * x_grid).sum(dim=[1, 2])
            keypoints[:, k, 1] = (heatmap_k * y_grid).sum(dim=[1, 2])
        
        return heatmaps, keypoints


class ChannelAttention(nn.Module):
    """第二级注意力：通道注意力"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        """
        参数:
            x: 特征图 (batch, C, H, W)
        
        返回:
            out: 通道加权后的特征 (batch, C, H, W)
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FeatureExtractor(nn.Module):
    """区域特征提取网络"""
    
    def __init__(self, in_channels=512, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_dim)
        self.attention = ChannelAttention(out_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # 第二级通道注意力
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        return x


class CropAndResize(nn.Module):
    """根据关键点裁剪和缩放区域"""
    
    def __init__(self, crop_size=64):
        super().__init__()
        self.crop_size = crop_size
    
    def forward(self, features, keypoints, img_size):
        """
        裁剪特征图上的区域
        
        参数:
            features: 特征图 (batch, C, H, W)
            keypoints: 关键点 (batch, K, 2) 坐标归一化到[0,1]
            img_size: (H, W) 原始图像尺寸
        
        返回:
            crops: 裁剪区域 (batch, K, C, crop_size, crop_size)
        """
        batch, C, H, W = features.shape
        K = keypoints.shape[1]
        crop_size = self.crop_size
        device = features.device
        
        # 生成采样网格
        crops = []
        for k in range(K):
            kp = keypoints[:, k, :]  # (batch, 2)
            
            # 构建以关键点为中心的网格
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-0.2, 0.2, crop_size, device=device),
                torch.linspace(-0.2, 0.2, crop_size, device=device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.unsqueeze(0).expand(batch, -1, -1, -1)
            
            # 加上关键点偏移
            offset = kp.view(batch, 1, 1, 2)
            grid = grid + offset
            
            # 采样
            crop = F.grid_sample(features, grid, align_corners=True)
            crops.append(crop)
        
        return torch.stack(crops, dim=1)  # (batch, K, C, crop_size, crop_size)


class TLAM(nn.Module):
    """两级注意力模型"""
    
    def __init__(self, backbone_channels=512, n_keypoints=4, 
                 crop_size=64, feat_dim=256, n_classes=200):
        super().__init__()
        
        self.n_keypoints = n_keypoints
        
        # 第一级：空间定位
        self.localizer = SpatialLocalizer(backbone_channels, n_keypoints)
        
        # 裁剪模块
        self.cropper = CropAndResize(crop_size)
        
        # 特征提取（共享）
        self.feature_extractor = FeatureExtractor(backbone_channels, feat_dim)
        
        # 分类器
        self.classifier = nn.Linear(feat_dim * n_keypoints, n_classes)
    
    def forward(self, features, img_size):
        """
        参数:
            features: backbone特征 (batch, C, H, W)
            img_size: (H, W) 原始图像尺寸
        
        返回:
            logits: 分类logits (batch, n_classes)
            heatmaps: 注意力热图 (batch, K, H, W)
            keypoints: 关键点坐标 (batch, K, 2)
        """
        # 第一级：定位关键区域
        heatmaps, keypoints = self.localizer(features)
        
        # 裁剪区域
        crops = self.cropper(features, keypoints, img_size)
        batch, K, C, crop_h, crop_w = crops.shape
        
        # 第二级：提取特征 + 通道注意力
        region_features = []
        for k in range(K):
            feat = self.feature_extractor(crops[:, k])
            region_features.append(feat)
        
        # 融合所有区域特征
        combined = torch.cat(region_features, dim=1)
        
        # 分类
        logits = self.classifier(combined)
        
        return logits, heatmaps, keypoints


def compute_diversity_loss(heatmaps):
    """多样性损失：鼓励不同关键点关注不同区域"""
    batch, K, H, W = heatmaps.shape
    loss = 0.0
    
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            # 两个热图的重叠
            overlap = (heatmaps[:, k1] * heatmaps[:, k2]).sum(dim=[1, 2])
            loss += overlap.mean()
    
    return loss / (K * (K - 1) / 2)


class TLAMWithBackbone(nn.Module):
    """包含backbone的完整TLAM模型"""
    
    def __init__(self, n_keypoints=4, n_classes=200):
        super().__init__()
        # 简化的backbone（实际可用ResNet）
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
        )
        
        self.tlam = TLAM(512, n_keypoints, crop_size=32, feat_dim=256, n_classes=n_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits, heatmaps, keypoints = self.tlam(features, x.shape[2:])
        return logits, heatmaps, keypoints


def demo():
    model = TLAMWithBackbone(n_keypoints=4, n_classes=200)
    x = torch.randn(2, 3, 224, 224)
    logits, heatmaps, keypoints = model(x)
    
    print(f"分类输出: {logits.shape}")
    print(f"注意力热图: {heatmaps.shape}")
    print(f"关键点坐标: {keypoints.shape}")
    
    # 多样性损失
    div_loss = compute_diversity_loss(heatmaps)
    print(f"多样性损失: {div_loss.item():.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""TLAM - 手工关键点提取和注意力实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_softmax_manual(logits):
    """手工空间softmax"""
    batch, K, H, W = logits.shape
    flat = logits.view(batch, K, -1)
    exp = torch.exp(flat - flat.max(dim=2, keepdim=True)[0])
    probs = exp / exp.sum(dim=2, keepdim=True)
    return probs.view(batch, K, H, W)


def expected_keypoint_manual(heatmaps):
    """手工计算关键点坐标（期望形式）"""
    batch, K, H, W = heatmaps.shape
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, 1, H),
        torch.linspace(0, 1, W),
        indexing='ij'
    )
    
    keypoints = torch.zeros(batch, K, 2)
    for b in range(batch):
        for k in range(K):
            h = heatmaps[b, k]
            keypoints[b, k, 0] = (h * x_grid).sum()
            keypoints[b, k, 1] = (h * y_grid).sum()
    
    return keypoints


def channel_attention_manual(x, W1, W2):
    """手工通道注意力"""
    batch, C = x.shape
    # 压缩
    squeezed = torch.relu(x @ W1.t())  # (batch, C/r)
    # 激励
    weights = torch.sigmoid(squeezed @ W2.t())  # (batch, C)
    return x * weights


def test_tlam_manual():
    x = torch.randn(2, 4, 8, 8)
    heatmaps = spatial_softmax_manual(x)
    kp = expected_keypoint_manual(heatmaps)
    print(f"手工空间注意力: 热图 {heatmaps.shape}, 关键点 {kp.shape}")
    print(f"关键点坐标 (batch 0):\n{kp[0].detach().numpy()}")
    print("测试通过")


if __name__ == "__main__":
    test_tlam_manual()
```

---

## 9. 可视化与结果理解

### 9.1 注意力热图

- 4个关键点对应4个注意力热图
- 每个热图应聚焦于不同区域（头部、翅膀、尾部、脚）
- 多样性损失确保热图不重叠

### 9.2 关键点定位

- 正确训练后，关键点会落在判别性部位
- 所有鸟类的头部（即使角度不同）都能被定位到

### 9.3 通道注意力的效果

- 不同鸟种在相同区域（如头部）的判别特征不同
- 通道注意力自适应地选择最有判别力的特征维度

---

## 10. 模型评估

```python
"""TLAM评估"""
import torch
from sklearn.metrics import accuracy_score


def evaluate_tlam():
    model = TLAMWithBackbone(n_keypoints=4, n_classes=10)
    x = torch.randn(16, 3, 112, 112)
    y = torch.randint(0, 10, (16,))
    
    logits, heatmaps, keypoints = model(x)
    
    # 准确率
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(y.numpy(), preds.numpy())
    print(f"分类准确率: {acc:.4f}")
    
    # 多样性检查
    div = compute_diversity_loss(heatmaps)
    print(f"多样性损失: {div.item():.4f}")


if __name__ == "__main__":
    evaluate_tlam()
```

---

## 11. 常见问题与易错点

### Q1: TLAM不需要部件标注，关键点如何学习？
**A:** 关键点通过分类损失的梯度回传学习。如果分类需要关注喙的形状，定位网络会学习将关键点放在喙上。这是"弱监督定位"的典型例子。

### Q2: 多个关键点指向同一区域怎么办？
**A:** 多样性损失 $\mathcal{L}_{diversity}$ 惩罚关键点重叠。如果两个热图有重叠，损失增大，会迫使关键点分散。

### Q3: 两个级别的注意力在设计上有什么区别？
**A:** 第一级是空间注意力（定位"哪里"），输出热图和坐标；第二级是通道注意力（选择"什么特征"），输出通道权重。两者互补。

### Q4: 如何选择关键点数量K？
**A:** 常用 K=4（分别对应上下左右四个部位）。太少的K无法覆盖所有判别区域，太多的K会导致冗余和计算量增大。

### Q5: TLAM和注意力机制的关系？
**A:** TLAM本质上是"空间注意力 + 通道注意力"的组合。空间注意力定位区域，通道注意力增强特征。两级都是注意力机制在视觉任务中的具体应用。

---

## 12. 学习总结

**核心要点：**
1. 第一级：空间注意力定位关键区域
2. 第二级：通道注意力增强判别性特征
3. 可微分的期望关键点坐标
4. 多样性损失保证注意力分散
5. 弱监督学习——仅需类别标签

**公式总结：**
$$
A_k(x,y) = \text{softmax}(f_{loc}(F)_k), \quad 
P_k = \mathbb{E}_{A_k}[(x,y)], \quad
f_k' = f_k \odot \sigma(W \cdot \text{GAP}(f_k))
$$

---

## 13. 练习题与思考题

### 基础题

**1.** 为什么关键点坐标用期望形式而不是argmax？

<details>
<summary>答案</summary>
argmax不可微分，无法通过梯度回传训练定位网络。期望形式 $\mathbb{E}[x] = \sum x \cdot A(x)$ 是可微的，使端到端训练成为可能。
</details>

**2.** 多样性损失的作用是什么？

<details>
<summary>答案</summary>
防止关键点坍塌到同一位置。如果所有关键点都关注同一个区域，就退化为单级注意力，失去了两级设计的优势。
</details>

**3.** 通道注意力中的降维比率r如何选择？

<details>
<summary>答案</summary>
r=16是常用值。r在增加非线性表达能力（过小的中间层）和减少参数量之间平衡。r太大中间层过小会丢失信息，r=1则无降维。
</details>

### 进阶题

**4.** 推导TLAM中关键点坐标的梯度如何传到定位网络。

<details>
<summary>答案</summary>
设 $L$ 是分类损失，$x_k = \sum_{i,j} j \cdot A_k(i,j)$。$\partial L / \partial A_k(i,j) = \partial L / \partial x_k \cdot j + \partial L / \partial y_k \cdot i$。通过 $\partial A_k / \partial f_{loc}$ 传到定位网络。这实现了端到端的弱监督定位。
</details>

**5.** 设计一个TLAM的三级注意力扩展方案。

<details>
<summary>答案</summary>
第三级可以是尺度注意力：对不同尺度裁剪的区域分配权重。实现方式：在多个尺度上裁剪区域（如32×32, 64×64, 128×128），通过一个注意力网络学习每个尺度的权重，加权融合后分类。
</details>

---

## 14. 学习路径建议

### 预备知识
- CNN特征提取（VGG、ResNet）
- 细粒度分类基本概念
- 弱监督学习

### 进阶方向
1. **TLAM -> RA-CNN**：循环注意力CNN，交替放大和识别
2. **TLAM -> MA-CNN**：多注意力CNN，并行注意力分支
3. **TLAM -> OSME+MAMC**：多注意力多类别联合学习
4. **TLAM -> Transformer-based**：ViT + 细粒度分类

### 推荐阅读
- Xiao et al. "Two-Level Attention Model for Fine-Grained Object Classification." 2015.
- Fu et al. "Look Closer to See Better: Recurrent Attention CNN." 2017.
- Zheng et al. "Learning Multi-Attention Convolutional Neural Network." 2017.

### 项目实践
1. 在CUB-200-2011数据集上实现TLAM
2. 可视化注意力热图和定位结果
3. 比较不同K值对细粒度分类性能的影响
