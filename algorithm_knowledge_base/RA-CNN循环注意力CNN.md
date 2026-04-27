# RA-CNN循环注意力CNN 学习文档

> 不断"拉近镜头"——用循环注意力逐步聚焦细粒度分类的关键区域。

## 1. 算法基础认知

**一句话定义：** RA-CNN（Recurrent Attention CNN）由Jianlong Fu等人于2017年提出，在多个尺度上逐步聚焦注意力区域，每次聚焦后对更小区域进行更精细的分类。

**直觉类比：** 想象你在远处看到一只鸟——你先大致判断它是一只鸟（尺度1），然后走近看，注意到它翅膀的颜色（尺度2），再凑近看，发现喙的形状（尺度3）。RA-CNN模拟了这种"逐步放大"的注意力过程。

**核心思想：** "离得更近，看得更清"。在三个尺度上：每个尺度预测类别+注意力区域（通过APN），下一尺度用该区域放大后的子图作为输入，使分类逐步聚焦于判别性区域。

**关键组件：** APN（Attention Proposal Network）预测注意力正方形区域中心点和半边长。

## 2. 核心原理

### 2.1 工作流程

```
尺度1: 全图 → 分类 + APN预测注意力区域
          ↓ (裁剪放大)
尺度2: 注意力子图 → 分类 + APN预测更精细区域
          ↓ (裁剪放大)
尺度3: 最精细子图 → 分类
```

### 2.2 APN（Attention Proposal Network）

APN对每个尺度的特征图，输出3个参数：
- $t_x$：注意力区域中心x坐标（归一化[-1,1]）
- $t_y$：注意力区域中心y坐标（归一化[-1,1]）
- $t_l$：注意力区域半边长（归一化[-1,1]）

用双线性插值（而非硬裁剪）实现区域的裁剪，确保梯度可回传。

### 2.3 损失函数

$$\mathcal{L} = \sum_{s=1}^3 \mathcal{L}_{\text{cls}}^{(s)} + \lambda \sum_{s=1}^{2} \mathcal{L}_{\text{rank}}^{(s)}$$

其中排名损失鼓励更精细尺度的分类得分更高：
$$\mathcal{L}_{\text{rank}}^{(s)} = \max(0, p^{(s+1)} - p^{(s)} + \text{margin})$$

这里 $p^{(s)}$ 是尺度 $s$ 上正确类别的预测概率。

## 3. 数学公式与推导

### 3.1 注意力裁剪

APN预测三个坐标参数 $[t_x, t_y, t_l]$，其中 $t_l$ 控制注意力框的大小。裁剪区域的坐标：
- 中心：$(c_x, c_y) = ((t_x+1)/2 \cdot W, (t_y+1)/2 \cdot H)$
- 半边长：$L = (t_l+1)/2 \cdot \min(W, H)$

裁剪区域为：$[c_x - L, c_x + L] \times [c_y - L, c_y + L]$

### 3.2 双线性插值（可微裁剪）

由于裁剪坐标是连续的，不能直接索引。使用双线性插值从原图采样注意力区域：

$$V_{ij} = \sum_{u=1}^H \sum_{v=1}^W U_{uv} \cdot \max(0, 1 - |x_{ij} - v|) \cdot \max(0, 1 - |y_{ij} - u|)$$

其中 $x_{ij}, y_{ij}$ 是目标像素在源图中的采样坐标。这个操作是可微的，梯度可以回传到APN参数。

### 3.3 Ranking Loss的直觉

排名损失 $p^{(s+1)} > p^{(s)}$ 意味着：更精细尺度的分类应该比更粗尺度的分类更确信。这驱动APN找到真正有助于分类的判别性区域。

## 4. 训练过程

### 4.1 训练流程
1. 所有尺度共享CNN权重（权值绑定）
2. 前向传播：三个尺度依次处理
3. 计算总损失（3个分类损失 + 2个排名损失）
4. 端到端反向传播（梯度通过双线性采样回传到APN）

### 4.2 训练细节
- Backbone: VGG-16（预训练，共享权重）
- 尺度数: 3（最常用）
- 输入尺寸: 224×224 → 112×112 → 56×56
- 优化器: SGD, momentum=0.9
- 排名margin: 0.05

### 4.3 为什么共享权重？
三个尺度的分类子网络共享权重。这迫使网络学习"尺度不变的判别特征"，同时大幅减少参数量。

## 5. 应用场景

1. **细粒度分类**：鸟类、汽车、飞机等细粒度识别
2. **行人检测**：逐步关注行人局部特征
3. **医学图像分析**：从全图逐步聚焦到病灶区域

## 6. 优缺点分析

### 优点
1. **多尺度聚焦**：从粗到细逐步定位判别性区域
2. **端到端可微**：APN使用双线性采样，梯度可回传
3. **排名约束**：强制更精细尺度有更高置信度
4. **共享权重**：减少参数量

### 缺点
1. **固定尺度数**：预先设定了3个尺度
2. **顺序计算**：不能并行，推理速度慢
3. **边框假设**：假设注意力区域是正方形

## 7. 调库实现

```python
"""
RA-CNN（循环注意力CNN）完整PyTorch实现
用于细粒度图像分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionSubnet(nn.Module):
    """单尺度的注意力子网络"""

    def __init__(self, num_classes=200, base_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(base_channels*4, num_classes)

    def forward(self, x):
        features = self.features(x)
        pooled = features.mean(dim=[2, 3])
        logits = self.classifier(pooled)
        return logits, features


class APN(nn.Module):
    """注意力建议网络"""

    def __init__(self, in_channels=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, features):
        x = features.mean(dim=[2, 3])
        params = torch.tanh(self.fc(x))
        tx, ty, tl = params[:, 0], params[:, 1], params[:, 2]
        return tx, ty, tl


class RACNN(nn.Module):
    """循环注意力CNN"""

    def __init__(self, num_classes=200, n_scales=3, base_channels=64):
        super().__init__()
        self.n_scales = n_scales
        self.subnet = AttentionSubnet(num_classes, base_channels)
        self.apns = nn.ModuleList([
            APN(base_channels * 4) for _ in range(n_scales - 1)
        ])

    def _attention_crop(self, x, tx, ty, tl, output_size):
        """可微的注意力裁剪（使用双线性采样）"""
        batch = x.size(0)
        h, w = x.size(2), x.size(3)

        # 反归一化
        cx = (tx + 1) / 2 * w
        cy = (ty + 1) / 2 * h
        L = (tl + 1) / 2 * min(h, w)

        # 限制边界
        cx = torch.clamp(cx, 0, w-1)
        cy = torch.clamp(cy, 0, h-1)
        L = torch.clamp(L, 1, min(h, w)//2)

        # 生成采样网格
        theta = torch.zeros(batch, 2, 3, device=x.device)
        theta[:, 0, 0] = L / (w / 2)
        theta[:, 1, 1] = L / (h / 2)
        theta[:, 0, 2] = (cx - w/2) / (w / 2)
        theta[:, 1, 2] = (cy - h/2) / (h / 2)

        grid = F.affine_grid(theta, (batch, 1, output_size, output_size), align_corners=False)
        crop = F.grid_sample(x, grid, align_corners=False)
        return crop

    def forward(self, x):
        logits_list = []
        current_input = x
        size = x.size(2)

        for s in range(self.n_scales):
            logits, features = self.subnet(current_input)
            logits_list.append(logits)

            if s < self.n_scales - 1:
                tx, ty, tl = self.apns[s](features)
                size = size // 2
                current_input = self._attention_crop(
                    current_input, tx, ty, tl, size)

        return logits_list


class RACNNLoss(nn.Module):
    """RA-CNN损失：分类损失 + 排名损失"""

    def __init__(self, margin=0.05, rank_weight=1.0):
        super().__init__()
        self.margin = margin
        self.rank_weight = rank_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_list, target):
        cls_loss = sum(self.ce(logits, target) for logits in logits_list)
        rank_loss = 0
        for s in range(len(logits_list) - 1):
            p_s = F.softmax(logits_list[s], dim=1)
            p_s1 = F.softmax(logits_list[s+1], dim=1)
            target_prob_s = p_s.gather(1, target.unsqueeze(1)).squeeze()
            target_prob_s1 = p_s1.gather(1, target.unsqueeze(1)).squeeze()
            rank_loss += torch.clamp(
                target_prob_s1 - target_prob_s + self.margin, min=0).mean()
        return cls_loss + self.rank_weight * rank_loss


def demo():
    x = torch.randn(2, 3, 224, 224)
    model = RACNN(num_classes=200, n_scales=3)
    outputs = model(x)
    for i, out in enumerate(outputs):
        print(f"尺度{i+1}输出: {out.shape}")

    target = torch.randint(0, 200, (2,))
    loss_fn = RACNNLoss()
    loss = loss_fn(outputs, target)
    print(f"总损失: {loss.item():.4f}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    demo()
```

## 8. 手工实现

```python
"""RA-CNN核心手工实现"""
import numpy as np

def bilinear_attention_crop(image, tx, ty, tl, output_size):
    """手工双线性注意力裁剪"""
    h, w = image.shape[:2]
    cx = (tx + 1) / 2 * w
    cy = (ty + 1) / 2 * h
    L = (tl + 1) / 2 * min(h, w)
    crop = np.zeros((output_size, output_size, 3))
    for i in range(output_size):
        for j in range(output_size):
            src_x = cx + (j - output_size/2) * (2*L) / output_size
            src_y = cy + (i - output_size/2) * (2*L) / output_size
            x0, y0 = int(src_x), int(src_y)
            x1, y1 = min(x0+1, w-1), min(y0+1, h-1)
            dx, dy = src_x - x0, src_y - y0
            crop[i, j] = (1-dy)*(1-dx)*image[y0, x0] + (1-dy)*dx*image[y0, x1] \
                       + dy*(1-dx)*image[y1, x0] + dy*dx*image[y1, x1]
    return crop

def test():
    img = np.random.rand(224, 224, 3)
    crop = bilinear_attention_crop(img, 0.0, 0.0, 0.5, 112)
    print(f"裁剪: {img.shape} -> {crop.shape}")
    print("测试通过!")

if __name__ == "__main__":
    test()
```

## 9. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_ra_cnn_pipeline(image, crops_list, save_path='ra_cnn_pipeline.png'):
    """可视化RA-CNN的逐步聚焦过程"""
    n = len(crops_list) + 1
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    axes[0].imshow(image)
    axes[0].set_title(f'尺度1 (全图)\n分类得分: {crops_list[0]:.3f}' 
                       if len(crops_list) > 0 else '尺度1 (全图)')
    axes[0].axis('off')
    for i, (crop, score) in enumerate(crops_list):
        axes[i+1].imshow(crop)
        axes[i+1].set_title(f'尺度{i+2} (放大)\n分类得分: {score:.3f}')
        axes[i+1].axis('off')
    plt.suptitle('RA-CNN: 逐步聚焦注意力区域', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"已保存到 {save_path}")

def demo_vis():
    img = np.random.rand(224, 224, 3)
    crops = [
        (np.random.rand(112, 112, 3), np.random.rand()),
        (np.random.rand(56, 56, 3), np.random.rand() + 0.1),
    ]
    visualize_ra_cnn_pipeline(img, crops)

if __name__ == "__main__":
    demo_vis()
```

## 10. 模型评估

```python
"""RA-CNN评估"""
def evaluate_ra_cnn(model, test_loader):
    model.eval()
    correct_list = [0] * model.n_scales
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            logits_list = model(x)
            total += y.size(0)
            for s, logits in enumerate(logits_list):
                correct_list[s] += (logits.argmax(1) == y).sum().item()
    accs = [c/total for c in correct_list]
    for s, acc in enumerate(accs):
        print(f"尺度{s+1}准确率: {acc:.4f}")
    return accs

def demo_eval():
    model = RACNN(num_classes=200, n_scales=3)
    x = torch.randn(16, 3, 224, 224)
    y = torch.randint(0, 200, (16,))
    outputs = model(x)
    for s, out in enumerate(outputs):
        acc = (out.argmax(1) == y).float().mean()
        print(f"尺度{s+1}模拟准确率: {acc:.4f}")

if __name__ == "__main__":
    demo_eval()
```

## 11. 常见问题与易错点

**Q1: APN预测的tx, ty, tl为什么用tanh激活？**
tanh输出范围[-1,1]，方便表示归一化后的坐标和半边长。0表示中心/半尺寸，±1表示边界。

**Q2: 如果APN预测的注意力区域不在物体上，怎么办？**
排名损失会处理这种情况——如果裁剪区域没有包含判别性特征，更精细尺度的分类得分不会更高，排名损失就会惩罚APN。

**Q3: 共享权重有什么问题？**
三个尺度的输入分辨率不同（224, 112, 56），共享权重要求CNN对尺度具有一定的鲁棒性。实际中VGG-16的卷积层对尺度变化有一定容忍度。

## 12. 学习总结

- RA-CNN通过循环注意力实现"从粗到细"的逐步聚焦
- APN + 双线性采样的组合实现了可微的注意力裁剪
- 排名损失保证了更精细尺度的分类更有信心
- 与MA-CNN的对比：RA-CNN是"顺序多尺度"，MA-CNN是"并行多部件"

## 13. 练习题

**基础题：**

1. RA-CNN使用几个尺度？每个尺度的作用是什么？
> **答案：** 3个尺度。尺度1识别大致类别，尺度2定位判别性区域，尺度3精细识别局部特征。

2. Ranking Loss的目的是什么？
> **答案：** 确保更精细尺度的预测概率更高，驱动APN找到真正有助于分类的区域。

**进阶题：**

3. 如果APN总是预测中心区域（tx=0, ty=0），如何改进？
> **答案：** 增加注意力区域的多样性损失，或引入注意力区域的随机初始化。

4. RA-CNN能否扩展到>3个尺度？
> **答案：** 可以，但收益递减。图像分辨率每次减半，超过3个尺度后分辨率太低，信息损失严重。

5. 在RA-CNN中，APN为什么使用tanh激活函数而不是sigmoid？
> **答案：** tanh输出范围为[-1,1]，中心为0，适合表示"相对于图像中心的偏移"。sigmoid输出[0,1]只能表示正方向，不适合表示负方向偏移。

6. 如果训练时所有尺度的预测概率都很高（>0.9），排名损失是否还起作用？
> **答案：** 如果所有尺度概率都高但精细化程度不够，排名损失可能已经饱和（概率不能再高了）。此时可能需要调整margin或使用其他正则化手段。

## 14. 学习路径

**前置：** CNN基础、细粒度分类
**平行：** MA-CNN（多部件注意力）、NTS-Net
**进阶：** Vision Transformer、DINO（自监督ViT）

### 14.1 推荐学习顺序

1. **第1步**：掌握标准CNN分类（VGG/ResNet）
2. **第2步**：理解弱监督定位（CAM、Grad-CAM）
3. **第3步**：学习RA-CNN（递归多尺度）
4. **第4步**：学习MA-CNN（并行多部件）
5. **第5步**：进阶到DINO（自监督+注意力）