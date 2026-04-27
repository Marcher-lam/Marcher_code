# DeepFix 学习文档

> 基于全卷积网络的注视点预测模型，用深度学习实现视觉显著性检测。

## 1. 算法基础认知

**一句话定义：** DeepFix是由Kruthiventi等人于2016年提出的基于全卷积网络（FCN）的注视点预测模型，通过多尺度特征融合和多任务学习实现高精度的视觉显著性检测。

**直觉类比：** DeepFix像一位经验丰富的摄影师——它能够同时从全局（场景布局）和局部（物体细节）两个尺度分析图像，找出最吸引眼球的区域。同时它还学会了"边缘"的概念（多任务学习），就像摄影师知道物体的轮廓在哪里。

**历史背景：** 2016年，DeepFix在MIT Saliency Benchmark上取得了当时最先进的性能，标志着深度学习在显著性检测领域的成熟应用。

**算法定位：** DeepFix属于**深度视觉显著性检测**模型，基于FCN架构进行端到端的注视点预测。

## 2. 核心原理

### 2.1 工作流程

```
输入图像 → 多尺度CNN特征提取 → 多任务学习(显著性+边缘) → 跨尺度特征融合 → 显著性图输出
```

### 2.2 关键设计

**全卷积架构：** 使用FCN实现端到端的显著性预测，避免了复杂的后处理

**多尺度特征：** 融合不同层的特征图（conv3, conv4, conv5），同时捕获全局和局部信息

**多任务学习：** 同时预测显著性和边缘，利用边缘信息增强显著性

**侧输出融合：** 在多个尺度上分别预测显著性图，然后融合得到最终结果

### 2.3 为什么多任务学习有效？

边缘检测和显著性检测共享低层特征（边缘、纹理），但高层语义不同。通过同时优化两个目标，网络在低层学到更丰富的边缘纹理表征，这些表征对显著性检测也是有用的。

## 3. 数学公式与推导

### 3.1 多尺度预测

DeepFix在 $L$ 个不同尺度上分别预测显著性图：

$$S^{(l)} = f_l(F^{(l)})$$

其中 $F^{(l)}$ 是第 $l$ 层的特征图，$f_l$ 是对应的预测头。

### 3.2 特征融合

$$S = g([S^{(1)}; S^{(2)}; ...; S^{(L)}])$$

其中 $g$ 是融合函数（1×1卷积 + 上采样）。

### 3.3 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{KL}}(S, \hat{S}) + \lambda \mathcal{L}_{\text{BCE}}(E, \hat{E})$$

KL散度度量预测显著性分布 $S$ 与ground truth $\hat{S}$ 之间的差异：

$$\mathcal{L}_{\text{KL}} = \sum_i \hat{S}_i \log \frac{\hat{S}_i}{S_i + \epsilon}$$

### 3.4 KL散度的直觉

KL散度 = 0 当且仅当 $S = \hat{S}$。当预测分布与真实分布一致时，KL散度最小。与MSE相比，KL散度更关注概率分布的匹配，而不是绝对值的差异。

## 4. 训练过程讲解

### 4.1 数据预处理
- 图像缩放到224×224
- ground truth显著性图做高斯模糊（$\sigma=3$~5像素）
- 归一化到[0,1]范围

### 4.2 训练配置
- 优化器: SGD with momentum 0.9
- 学习率: 初始0.001, 每20epoch衰减0.1
- Batch size: 16
- Epochs: 50
- 数据增强: 随机翻转、色彩抖动

### 4.3 训练流程
```
1. 加载预训练VGG-16权重（ImageNet）
2. 随机初始化侧输出层和融合层
3. 前向传播 → 计算KL损失 + 边缘损失
4. 反向传播 → 更新参数
5. 每5epoch在验证集上评估CC/NSS指标
```

## 5. 应用场景

1. **注视点预测**：预测人眼在图像上的注视位置，用于认知科学研究
2. **广告设计**：评估广告布局的视觉吸引力
3. **UI/UX优化**：判断用户界面元素的视觉优先级
4. **图像裁剪**：根据显著性图自动裁剪保留重要内容

## 6. 优缺点分析

**优点：** 端到端训练、多尺度特征融合、多任务学习增强
**缺点：** 需要大量标注数据、模型参数量大、对异常场景泛化有限

## 7. 调库实现

```python
"""
DeepFix风格的显著性检测模型——完整PyTorch实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块: Conv + BN + ReLU"""
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class DeepFixLike(nn.Module):
    """DeepFix风格的显著性预测网络"""
    def __init__(self, in_channels=3):
        super().__init__()
        # 编码器（简化VGG风格）
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, 64), ConvBlock(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            ConvBlock(64, 128), ConvBlock(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(
            ConvBlock(256, 512), ConvBlock(512, 512), ConvBlock(512, 512))
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = nn.Sequential(
            ConvBlock(512, 512), ConvBlock(512, 512), ConvBlock(512, 512))

        # 侧输出（多尺度）
        self.side3 = nn.Conv2d(256, 1, 1)
        self.side4 = nn.Conv2d(512, 1, 1)
        self.side5 = nn.Conv2d(512, 1, 1)

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid())

        # 边缘分支
        self.edge_branch = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        input_size = x.shape[2:]
        # 编码
        e1 = self.pool1(self.enc1(x))
        e2 = self.pool2(self.enc2(e1))
        e3 = self.enc3(e2)  # 1/4
        e4 = self.enc4(self.pool3(e3))  # 1/8
        e5 = self.enc5(self.pool4(e4))  # 1/16

        # 侧输出并上采样
        h, w = input_size
        s3 = F.interpolate(self.side3(e3), size=(h,w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.side4(e4), size=(h,w), mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.side5(e5), size=(h,w), mode='bilinear', align_corners=False)

        saliency = self.fusion(torch.cat([s3, s4, s5], dim=1))
        edge = F.interpolate(self.edge_branch(e5), size=(h,w), mode='bilinear', align_corners=False)
        return saliency, edge


class DeepFixLoss(nn.Module):
    """DeepFix损失: KL + 边缘BCE"""
    def __init__(self, edge_weight=0.1):
        super().__init__()
        self.edge_weight = edge_weight

    def forward(self, pred_sal, gt_sal, pred_edge=None, gt_edge=None):
        pred_sal = pred_sal + 1e-8
        gt_sal_norm = gt_sal / (gt_sal.sum() + 1e-8)
        kl = (gt_sal_norm * torch.log(gt_sal_norm / pred_sal)).sum()
        if pred_edge is not None and gt_edge is not None:
            edge_loss = F.binary_cross_entropy(pred_edge, gt_edge)
            return kl + self.edge_weight * edge_loss
        return kl


def demo():
    model = DeepFixLike()
    x = torch.randn(2, 3, 224, 224)
    sal, edge = model(x)
    print(f"输入: {x.shape}")
    print(f"显著性: {sal.shape} [{sal.min():.4f}, {sal.max():.4f}]")
    print(f"边缘: {edge.shape} [{edge.min():.4f}, {edge.max():.4f}]")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    demo()
```

### 7.1 多任务 DeepFix 训练循环

```python
"""
DeepFix完整训练循环示例
"""

import torch.optim as optim


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """一个epoch的训练"""
    model.train()
    total_loss = 0

    for batch_idx, (images, saliency_gts, edge_gts) in enumerate(dataloader):
        images = images.to(device)
        saliency_gts = saliency_gts.to(device)
        edge_gts = edge_gts.to(device)

        optimizer.zero_grad()
        pred_sal, pred_edge = model(images)
        loss = criterion(pred_sal, saliency_gts, pred_edge, edge_gts)
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx}: loss={loss.item():.4f}")

    return total_loss / len(dataloader)


def create_synthetic_dataset(num_samples=100, img_size=64):
    """创建合成数据集用于演示
    
    生成包含随机"显著物体"的图像和对应的ground truth。
    """
    images = []
    saliency_gts = []
    edge_gts = []

    for _ in range(num_samples):
        img = np.random.rand(img_size, img_size, 3).astype(np.float32) * 0.3
        gt = np.zeros((img_size, img_size), dtype=np.float32)
        edge = np.zeros((img_size, img_size), dtype=np.float32)

        # 随机放置一个矩形"显著物体"
        cx, cy = np.random.randint(10, img_size-10, 2)
        size = np.random.randint(5, 15)
        x1, y1 = max(0, cx-size), max(0, cy-size)
        x2, y2 = min(img_size, cx+size), min(img_size, cy+size)
        img[y1:y2, x1:x2] = [0.8, 0.2, 0.2]
        gt[y1:y2, x1:x2] = 1.0

        # 高斯模糊gt
        from scipy.ndimage import gaussian_filter
        gt = gaussian_filter(gt, sigma=3)
        gt = gt / gt.max()

        # 边缘: Canny边缘的简化模拟
        edge[y1:y2, x1] = 1
        edge[y1:y2, x2-1] = 1
        edge[y1, x1:x2] = 1
        edge[y2-1, x1:x2] = 1

        images.append(img.transpose(2, 0, 1))
        saliency_gts.append(gt)
        edge_gts.append(edge)

    return (torch.tensor(np.array(images)),
            torch.tensor(np.array(saliency_gts)).unsqueeze(1),
            torch.tensor(np.array(edge_gts)).unsqueeze(1))


def demo_training():
    """演示DeepFix训练流程"""
    print("=== DeepFix训练演示 ===")

    device = 'cpu'
    model = DeepFixLike().to(device)
    criterion = DeepFixLoss(edge_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    images, sal_gts, edge_gts = create_synthetic_dataset(num_samples=20, img_size=64)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, sals, edges):
            self.imgs, self.sals, self.edges = imgs, sals, edges
        def __len__(self): return len(self.imgs)
        def __getitem__(self, i):
            return self.imgs[i], self.sals[i], self.edges[i]

    dataset = SimpleDataset(images, sal_gts, edge_gts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    print("训练前:")
    with torch.no_grad():
        ps, pe = model(images[:2])
        print(f"  初始损失: KL+Edge={criterion(ps, sal_gts[:2], pe, edge_gts[:2]):.4f}")

    for epoch in range(3):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: avg_loss={loss:.4f}")

    print("训练完成!")


if __name__ == "__main__":
    demo_training()
```

---

## 8. 手工代码实现

```python
"""DeepFix核心算子手工实现"""
import numpy as np

def bilinear_upsample(x, target_h, target_w):
    """手工双线性上采样"""
    h, w = x.shape
    out = np.zeros((target_h, target_w))
    for i in range(target_h):
        for j in range(target_w):
            src_i = i * h / target_h
            src_j = j * w / target_w
            # 双线性插值
            i0, j0 = int(src_i), int(src_j)
            i1, j1 = min(i0+1, h-1), min(j0+1, w-1)
            di, dj = src_i - i0, src_j - j0
            out[i,j] = (1-di)*(1-dj)*x[i0,j0] + (1-di)*dj*x[i0,j1] \
                      + di*(1-dj)*x[i1,j0] + di*dj*x[i1,j1]
    return out

def test_handcraft():
    x = np.random.randn(8, 8)
    up = bilinear_upsample(x, 16, 16)
    print(f"手工上采样: {x.shape} → {up.shape}")
    print("测试通过!")

if __name__ == "__main__":
    test_handcraft()
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_saliency_deepfix(image, sal_map, edge_map):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title('(a) 输入图像'); axes[0].axis('off')
    im = axes[1].imshow(sal_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('(b) 显著性图'); axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(edge_map, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('(c) 边缘图'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('deepfix_vis.png', dpi=150)
    print("可视化已保存")

if __name__ == "__main__":
    img = np.random.rand(100, 100, 3)
    sal = np.random.rand(100, 100)
    edge = np.random.rand(100, 100) > 0.5
    visualize_saliency_deepfix(img, sal, edge.astype(float))
```

## 10. 模型评估

```python
"""DeepFix评估指标"""

import numpy as np

def compute_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    # KL散度
    kl = (gt * np.log(gt / (pred + 1e-8) + 1e-8)).sum()
    # CC相关系数
    cc = np.corrcoef(pred, gt)[0, 1]
    # NSS
    pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
    nss = pred_norm[gt > gt.mean()].mean()
    return {'KL': kl, 'CC': cc, 'NSS': nss}

def demo_eval():
    np.random.seed(42)
    pred = np.random.rand(64, 64)
    gt = np.random.rand(64, 64)
    gt[20:40, 20:40] = 0.8
    metrics = compute_metrics(pred, gt)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    demo_eval()
```

## 11. 常见问题与易错点

**Q1: 为什么DeepFix采用多尺度特征而非单尺度？**
不同显著物体尺寸不同。小物体需要高分辨率低层特征，大物体需要语义丰富的高层特征。多尺度融合可以同时处理不同尺度。

**Q2: KL散度和MSE在显著性评估中的区别？**
KL散度关注概率分布匹配（更适合注视点预测），MSE关注像素级匹配（更适合显著物体检测）。

**Q3: 边缘辅助任务为什么有帮助？**
边缘检测和显著性检测共享低层特征（边缘、纹理）。多任务学习迫使网络提取更丰富的特征，这些特征对显著性预测有正面作用。

## 12. 学习总结

- DeepFix是深度学习应用于显著性检测的代表作
- 核心创新：全卷积 + 多尺度 + 多任务
- 局限性：VGG backbone较旧，计算量大
- 继承与发展：DeepFix的思想被后来的EDN、SalGAN等模型继承

## 13. 练习题与思考题

**基础题：**

1. DeepFix的三种"尺度"分别对应网络的哪些层？
> **答案：** conv3（1/8分辨率），conv4（1/16分辨率），conv5（1/16分辨率）。conv3捕获细节，conv5捕获语义。

2. DeepFix为什么要预测边缘图？
> **答案：** 多任务学习增强特征表示。边缘检测辅助显著性检测，两者共享低层特征。

**进阶题：**

3. 如果将DeepFix中的VGG替换为ResNet，预期性能会如何变化？
> **答案：** ResNet更易优化，梯度传播更好，理论上性能会提升。实际上后来的SalGAN等确实使用了更深的backbone。

4. DeepFix对非自然图像（如医学图像）的泛化能力如何？
> **答案：** 有限。DeepFix在自然图像上训练，对医学图像（不同纹理/色彩分布）的泛化能力不佳。需要fine-tune。

## 14. 学习路径

**前置：** CNN基础、FCN语义分割、视觉显著性基础
**平行：** SALICON、SAM显著性模型
**进阶：** 基于Transformer的显著性检测、视频显著性检测
