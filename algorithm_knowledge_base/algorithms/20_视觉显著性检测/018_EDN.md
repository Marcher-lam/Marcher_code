# EDN 学习文档

> 编码器-解码器显著性网络——用端到端深度学习实现注视点预测。

## 1. 算法基础认知

**一句话定义：** EDN（Encoder-Decoder Network）是一种基于编码器-解码器架构的视觉显著性检测模型，通过端到端的学习直接从原始图像预测人眼注视点分布图。

**直觉类比：** EDN像一位画家——先快速勾勒出场景的粗略轮廓（编码器下采样），再逐步填补细节（解码器上采样），过程中还会参考之前的草稿（跳连skip connection）。

**核心思想：** 使用编码器提取多尺度图像特征，使用解码器逐步恢复分辨率生成显著性图。编码器-解码器之间的跳连（skip connection）保留了低层细节信息。

**历史背景：** 受FCN和U-Net等语义分割架构启发，EDN将编码器-解码器架构引入显著性检测领域，标志着从手工特征到端到端学习的转变。

**算法定位：** 基于编码器-解码器架构的显著性检测模型，端到端训练。

## 2. 核心原理

### 2.1 网络架构

```
输入图像 → 编码器(VGG/ResNet) → 瓶颈层 → 解码器 → 显著性图
                ↓ 跳连 ↓           ↓ 跳连 ↓
            多尺度特征            细节恢复
```

### 2.2 关键设计

- **编码器：** 使用预训练CNN（VGG16/ResNet50）提取多尺度特征
- **解码器：** 逐步上采样恢复空间分辨率，融合编码器的跳连特征
- **损失函数：** 混合使用KL散度、CC（相关系数）、SIM（相似度）等多个显著性评估指标

### 2.3 为什么跳连重要？

编码器下采样过程中丢失了空间细节（边缘、纹理）。解码器虽然有高层语义信息，但缺少这些细节。跳连将编码器每个stage的细节特征直接传到解码器对应stage，实现语义+细节的互补。

## 3. 数学公式与推导

### 3.1 编码器-解码器

编码器特征提取：
$$F_l = f_l(F_{l-1}), \quad l=1,...,L$$

解码器上采样并融合：
$$D_{l-1} = g_l(\text{Up}(D_l) \oplus F_{l-1})$$

其中 $\oplus$ 是通道拼接，$g_l$ 是卷积块。

### 3.2 损失函数

EDN的混合损失：

$$\mathcal{L} = \lambda_{KL}\mathcal{L}_{KL} + \lambda_{CC}\mathcal{L}_{CC} + \lambda_{SIM}\mathcal{L}_{SIM}$$

KL散度：
$$\mathcal{L}_{KL} = \sum_i \hat{S}_i \log \frac{\hat{S}_i}{S_i + \epsilon}$$

CC（相关系数）：
$$\mathcal{L}_{CC} = 1 - \frac{\text{Cov}(S, \hat{S})}{\sigma_S \sigma_{\hat{S}}}$$

### 3.3 为什么需要混合损失？

KL散度关注分布匹配（鼓励全面覆盖注视点区域），CC关注线性相关性（对齐预测和真实值的相对强度），SIM关注直方图匹配（保持整体亮度一致）。三者互补。

## 4. 训练过程讲解

### 4.1 训练流程
1. 加载预训练编码器权重
2. 随机初始化解码器
3. 每个batch：前向传播→计算混合损失→反向传播
4. 学习率调整：余弦退火或step decay
5. 在验证集上监控CC和NSS指标

### 4.2 数据增强
- 随机水平翻转、随机旋转(±5°)、随机裁剪
- 色彩抖动（亮度±0.1, 对比度±0.1）
- Gaussian blur（模拟注视点模糊）

### 4.3 训练提示
- 使用ImageNet预训练权重显著加速收敛
- 解码器学习率可以比编码器大10倍
- 使用梯度裁剪防止梯度爆炸

## 5. 应用场景

1. **注视点预测**：预测人在自由观看任务中的眼动轨迹
2. **图像质量评估**：显著性引导的图像质量评价
3. **视频显著性**：扩展到视频帧序列的注视点预测

## 6. 优缺点分析

**优点：** 端到端训练、跳跃连接保留细节、多尺度特征融合
**缺点：** 需要大量标注数据、分辨率受限、计算量大

## 7. 调库实现

```python
"""
EDN（编码器-解码器显著性网络）的完整PyTorch实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class EDN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = ConvBlock(1024+512, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ConvBlock(512+256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(256+128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(128+64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class MixedSaliencyLoss(nn.Module):
    def __init__(self, w_kl=1.0, w_cc=1.0, w_sim=1.0):
        super().__init__()
        self.w_kl, self.w_cc, self.w_sim = w_kl, w_cc, w_sim

    def forward(self, pred, target):
        pred, target = pred.flatten(), target.flatten()
        target_norm = target / (target.sum() + 1e-8)
        pred_norm = pred / (pred.sum() + 1e-8)
        kl = (target_norm * torch.log(target_norm / (pred_norm + 1e-8) + 1e-8)).sum()
        cc = 1 - torch.corrcoef(torch.stack([pred, target]))[0, 1]
        hist_p = torch.histc(pred, bins=256, min=0, max=1)
        hist_t = torch.histc(target, bins=256, min=0, max=1)
        hist_p = hist_p / hist_p.sum()
        hist_t = hist_t / hist_t.sum()
        sim = -torch.min(hist_p, hist_t).sum()
        return self.w_kl*kl + self.w_cc*cc + self.w_sim*sim


def demo():
    model = EDN()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    criterion = MixedSaliencyLoss()
    gt = torch.rand(1, 1, 256, 256)
    loss = criterion(y, gt)
    print(f"输入: {x.shape}, 输出: {y.shape}, Loss: {loss.item():.4f}")
    print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    demo()
```

### 7.1 EDN训练循环

```python
"""
EDN完整训练循环
"""

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SaliencyDataset(Dataset):
    """合成显著性数据集"""

    def __init__(self, num_samples=100, size=64):
        self.data = []
        np.random.seed(42)
        for _ in range(num_samples):
            img = np.random.rand(3, size, size).astype(np.float32) * 0.3
            gt = np.zeros((1, size, size), dtype=np.float32)
            cx, cy = np.random.randint(size//4, 3*size//4, 2)
            r = np.random.randint(5, size//4)
            y, x = np.ogrid[:size, :size]
            mask = (x - cx)**2 + (y - cy)**2 < r**2
            img[:, mask] = 0.8
            gt[0, mask] = 1.0
            from scipy.ndimage import gaussian_filter
            gt[0] = gaussian_filter(gt[0], sigma=3)
            gt = gt / gt.max()
            self.data.append((img, gt))

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        return torch.FloatTensor(self.data[i][0]), torch.FloatTensor(self.data[i][1])


def train_edn():
    """训练EDN模型"""
    device = 'cpu'
    model = EDN().to(device)
    criterion = MixedSaliencyLoss(w_kl=1.0, w_cc=0.5, w_sim=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = SaliencyDataset(num_samples=30, size=64)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("=== EDN训练演示 ===")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for images, gts in loader:
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, gts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
    print("训练完成!")


if __name__ == "__main__":
    train_edn()
```

---

## 8. 手工代码实现

```python
"""EDN核心手工实现"""

import numpy as np

def conv2d_numpy(x, weight, bias=None, stride=1, padding=0):
    """手工2D卷积"""
    C, H, W = x.shape
    Cout, Cin, Kh, Kw = weight.shape
    H_out = (H + 2*padding - Kh) // stride + 1
    W_out = (W + 2*padding - Kw) // stride + 1
    x_pad = np.pad(x, ((0,0),(padding,padding),(padding,padding)), mode='constant')
    out = np.zeros((Cout, H_out, W_out))
    for co in range(Cout):
        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[:, i*stride:i*stride+Kh, j*stride:j*stride+Kw]
                out[co, i, j] = np.sum(patch * weight[co]) + (bias[co] if bias is not None else 0)
    return out

def maxpool2d_numpy(x, k=2, s=2):
    C, H, W = x.shape
    Ho, Wo = H//s, W//s
    out = np.zeros((C, Ho, Wo))
    for c in range(C):
        for i in range(Ho):
            for j in range(Wo):
                out[c,i,j] = x[c, i*s:i*s+k, j*s:j*s+k].max()
    return out

def test():
    x = np.random.randn(3, 64, 64)
    w = np.random.randn(64, 3, 3, 3) * 0.1
    b = np.zeros(64)
    out = conv2d_numpy(x, w, b, stride=1, padding=1)
    pooled = maxpool2d_numpy(out)
    print(f"Conv: {x.shape} -> {out.shape}")
    print(f"Pool: {out.shape} -> {pooled.shape}")

if __name__ == "__main__":
    test()
```

## 9. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_edn_pipeline():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    inputs = [np.random.rand(100,100) for _ in range(6)]
    titles = ['原始图像', '编码器特征', '瓶颈特征',
              '解码器第4层', '解码器第1层', '最终显著性图']
    for i, (ax, inp, title) in enumerate(zip(axes.flatten(), inputs, titles)):
        ax.imshow(inp, cmap='gray' if i!=0 else None)
        ax.set_title(f'({chr(97+i)}) {title}', fontsize=10)
        ax.axis('off')
    plt.suptitle('EDN编码器-解码器流水线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('edn_pipeline.png', dpi=150)
    print("EDN流水线可视化已保存")

if __name__ == "__main__":
    visualize_edn_pipeline()
```

## 10. 模型评估

```python
"""EDN模型评估"""

def evaluate_edn(pred, gt):
    from sklearn.metrics import roc_auc_score
    p, g = pred.flatten(), gt.flatten()
    auc = roc_auc_score((g > g.mean()).astype(int), p)
    cc = np.corrcoef(p, g)[0, 1]
    p_norm = (p - p.mean()) / (p.std() + 1e-8)
    nss = p_norm[g > g.mean()].mean()
    return {'AUC': auc, 'CC': cc, 'NSS': nss}

def benchmark():
    np.random.seed(42)
    pred = np.random.rand(256, 256)
    gt = np.zeros((256, 256))
    gt[80:180, 80:180] = 1
    from scipy.ndimage import gaussian_filter
    gt = gaussian_filter(gt, sigma=10)
    gt = gt / gt.max()
    metrics = evaluate_edn(pred, gt)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    benchmark()
```

## 11. 常见问题与易错点

**Q1: EDN和FCN的区别？**
EDN在FCN基础上增加了跳连（类似U-Net），且专门针对显著性检测优化了损失函数。

**Q2: 为什么EDN的输出是连续值（0~1）而不是二值？**
显著性检测得到的是注视点概率密度图（连续），不是分割掩膜（二值）。Sigmoid输出满足这一需求。

**Q3: 解码器性能不如编码器时怎么办？**
解码器可以增加通道数或使用更宽的解码器。也可以对解码器使用更高的学习率。

## 12. 学习总结

- EDN是编码器-解码器架构在显著性检测中的典型应用
- 核心设计：U-Net风格跳连 + 混合损失函数
- 本质：将分割架构迁移到显著性任务，配合特定损失函数

## 13. 练习题

**基础题：**

1. EDN编码器下采样了几次？输出分辨率是多少？
> **答案：** 4次，输出是输入的1/16分辨率。

2. 跳连的目的是什么？
> **答案：** 保留低层空间细节（边缘、纹理），补充解码器上采样丢失的信息。

**进阶题：**

3. 如果将EDN中的双线性上采样替换为转置卷积，会有什么影响？
> **答案：** 转置卷积可学习上采样参数（更灵活），但可能产生棋盘格伪影。双线性上采样无参数、更稳定。

4. 如何将EDN扩展到视频显著性检测？
> **答案：** 增加时序模块（Conv3D/LSTM/3D-UNet），在视频帧间传播显著性信息。

## 14. 学习路径

**前置：** FCN、U-Net、CNN基础
**平行：** DeepFix、SALICON
**进阶：** 基于Transformer的显著性检测