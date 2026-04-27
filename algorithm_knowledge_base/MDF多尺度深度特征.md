# MDF多尺度深度特征 学习文档

> 用CNN提取多尺度深度特征进行显著物体检测——预训练+微调模式。
> 来源线索：原书第2.2.2节"MDF：基于多尺度深度特征的显著物体检测"。

---

## 1. 算法基础认知

**一句话定义：** MDF（Multi-scale Deep Features）由Li等人于2015年提出，使用S-3CNN（三个共享权重的CNN）分别提取区域、邻域和整图的多尺度深度特征，融合后通过全连接网络预测显著性。

**核心思想：** 深度特征（CNN提取）远强于手工特征。MDF将三个尺度的深度特征融合，同时获得局部细节和全局语义信息。

**三个尺度：**
1. 区域尺度：围绕像素的小块(~64x64)，包含局部纹理和颜色信息
2. 邻域尺度：中等区域(~128x128)，包含局部上下文
3. 全局尺度：整幅图像(~224x224)，包含全局语义

**历史定位：** MDF是最早将CNN引入显著物体检测的工作之一。

---

## 2. 核心原理

### 2.1 网络架构

三个权值共享的CNN特征提取器(S-3CNN) + 特征融合层 + 显著性预测层。

### 2.2 S-3CNN结构

使用类似AlexNet的结构：5个卷积层 + 全局平均池化 + 全连接层(512维)。

三个尺度的图像共享相同的CNN权重。

### 2.3 特征融合

f = [f_region, f_neighbor, f_global] (1536维)
h1 = ReLU(W1*f + b1)  # 1536->300
h2 = ReLU(W2*h1 + b2) # 300->300
s = sigmoid(W3*h2 + b3) # 300->1

### 2.4 训练策略

预训练+微调：ImageNet预训练，SOD数据集端到端微调。

正负样本：显著区域内的像素为正，背景为负，1:1平衡采样。

---

## 3. 数学公式与推导

### 3.1 CNN特征提取

f_s = CNN(x_s; theta)，theta在三个尺度间共享。

每层：h = ReLU(Conv(x, W) + b)，再经MaxPool和AdaptiveAvgPool。

### 3.2 融合与预测

s = sigma(W_3 * ReLU(W_2 * ReLU(W_1 * f + b_1) + b_2) + b_3)

### 3.3 损失函数

L = -1/N * sum_i [y_i * log(s_i) + (1-y_i) * log(1-s_i)]

二值交叉熵损失。

---

## 4. 训练过程讲解

### 4.1 数据准备

对每个像素取三个尺度的图像块：区域64x64，邻域128x128，全局224x224。

### 4.2 训练步骤

1. 加载ImageNet预训练CNN权重
2. 替换最后分类层为512维特征层
3. 初始化融合网络
4. 端到端训练：SGD, lr=1e-4, momentum=0.9, 10 epochs

### 4.3 推理

逐像素取三个尺度图像块 -> CNN前向 -> 融合预测 -> 插值恢复 -> 高斯平滑。

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 显著物体检测 | 自然图像中定位显著物体 |
| 图像分割辅助 | 显著性作为分割先验 |
| 目标检测 | 区域提议的筛选 |
| 视觉跟踪 | 初始帧目标定位 |

---

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 深度特征远强于手工特征 | 逐像素推理速度慢 |
| 多尺度融合提供互补信息 | 三个CNN需三次前向 |
| ImageNet预训练提供良好初始化 | 全局尺度重复计算 |
| 端到端训练优化全局 | 训练数据采样开销大 |

---

## 7. 调库实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class SharedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))


class MDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = SharedCNN()
        self.fusion = nn.Sequential(
            nn.Linear(512*3, 300), nn.ReLU(inplace=True),
            nn.Linear(300, 300), nn.ReLU(inplace=True),
            nn.Linear(300, 1), nn.Sigmoid()
        )

    def forward(self, region, neighbor, full):
        f_r = self.cnn(region)
        f_n = self.cnn(neighbor)
        f_g = self.cnn(full)
        combined = torch.cat([f_r, f_n, f_g], dim=1)
        score = self.fusion(combined)
        return score


def demo_mdf():
    model = MDF()
    model.eval()
    region = torch.randn(2, 3, 64, 64)
    neighbor = torch.randn(2, 3, 128, 128)
    full = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        scores = model(region, neighbor, full)
    print(f"MDF输出: {scores.shape}, scores: {scores.numpy().flatten()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params/1e6:.2f}M")

    # 模拟全图预测
    dummy = torch.randn(1, 3, 224, 224)
    h, w = 224, 224; stride = 32
    sm = np.zeros((h,w)); cm = np.zeros((h,w))
    for y in range(0, h-64, stride):
        for x in range(0, w-64, stride):
            cy, cx = y+32, x+32
            r = dummy[:,:,max(0,cy-32):min(h,cy+32),max(0,cx-32):min(w,cx+32)]
            r = F.interpolate(r, (64,64), mode='bilinear', align_corners=False) if r.shape[2]!=64 else r
            n = dummy[:,:,max(0,cy-64):min(h,cy+64),max(0,cx-64):min(w,cx+64)]
            n = F.interpolate(n, (128,128), mode='bilinear', align_corners=False) if n.shape[2]!=128 else n
            with torch.no_grad(): s = model(r, n, dummy).item()
            sm[min(h-1,cy),min(w-1,cx)] = s; cm[min(h-1,cy),min(w-1,cx)] = 1
    sm = np.where(cm>0, sm, 0)
    print(f"显著图: [{sm.min():.3f}, {sm.max():.3f}]")
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].imshow(dummy[0].permute(1,2,0).numpy()*0.2+0.5)
    axes[0].set_title('Input'); axes[0].axis('off')
    im = axes[1].imshow(sm, cmap='jet'); axes[1].set_title('MDF'); axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout(); plt.savefig('mdf_demo.png', dpi=150); plt.show()

if __name__ == '__main__':
    demo_mdf()
```

---

## 8. 手工代码实现（NumPy）

```python
import numpy as np
from scipy.ndimage import gaussian_filter


class MDFNumpy:
    def __init__(self):
        np.random.seed(42)
        self.W1 = np.random.randn(10,32)*0.01; self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32,16)*0.01; self.b2 = np.zeros(16)
        self.W3 = np.random.randn(16,1)*0.01; self.b3 = np.zeros(1)

    def _patch_feat(self, patch):
        gray = np.mean(patch, axis=2) if patch.ndim==3 else patch
        f = [gray.mean(), gray.std(), np.abs(np.diff(gray,axis=1)).mean(), np.abs(np.diff(gray,axis=0)).mean()]
        if patch.ndim==3:
            for c in range(3): f.extend([patch[:,:,c].mean(), patch[:,:,c].std()])
        f = f[:10]
        while len(f)<10: f.append(0)
        return np.array(f)

    def _predict(self, feat):
        x = np.maximum(0, feat@self.W1+self.b1)
        x = np.maximum(0, x@self.W2+self.b2)
        return 1/(1+np.exp(-(x@self.W3+self.b3)))

    def compute_saliency(self, image, stride=8):
        if image.max()>1.0: image/=255.0
        h,w = image.shape[:2]
        sm = np.zeros((h,w)); cnt = np.zeros((h,w))
        full_gray = np.mean(image,axis=2)
        for y in range(0, h-16, stride):
            for x in range(0, w-16, stride):
                cy,cx = y+8, x+8
                r = image[max(0,cy-8):min(h,cy+8),max(0,cx-8):min(w,cx+8)]
                n = image[max(0,cy-16):min(h,cy+16),max(0,cx-16):min(w,cx+16)]
                fr = self._patch_feat(r); fn = self._patch_feat(n)
                fg = np.zeros(10); fg[:3]=[full_gray.mean(),full_gray.std(),np.abs(np.diff(full_gray)).mean()]
                feat = np.concatenate([fr,fn,fg])
                feat_pad = np.zeros(32); feat_pad[:len(feat)]=feat
                s = self._predict(feat_pad)
                sm[cy,cx]=s; cnt[cy,cx]=1
        sm = np.where(cnt>0, sm, 0)
        sm = gaussian_filter(sm, 3)
        return (sm-sm.min())/(sm.max()-sm.min()+1e-8)

def demo_numpy():
    np.random.seed(42)
    img = np.random.rand(48,48,3); img[15:33,15:33]=[0.8,0.2,0.2]
    m = MDFNumpy(); s = m.compute_saliency(img, stride=8)
    print(f"MDF手工: [{s.min():.3f}, {s.max():.3f}]")

if __name__ == '__main__':
    demo_numpy()
```

---

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt


def visualize_mdf_scales():
    np.random.seed(42)
    img = np.ones((128,128,3))*0.15; img[35:95,35:95]=[0.7,0.3,0.3]
    cy,cx = 65,65
    fig, axes = plt.subplots(1,4,figsize=(16,4))
    axes[0].imshow(img)
    axes[0].plot(cx,cy,'r+',markersize=15,linewidth=3)
    axes[0].set_title('(a) Input + center'); axes[0].axis('off')
    r = img[cy-32:cy+32,cx-32:cx+32]
    axes[1].imshow(r); axes[1].set_title('(b) Region 64x64'); axes[1].axis('off')
    n = img[cy-64:cy+64,cx-64:cx+64]
    axes[2].imshow(n); axes[2].set_title('(c) Neighbor 128x128'); axes[2].axis('off')
    axes[3].imshow(img); axes[3].set_title('(d) Global 224x224'); axes[3].axis('off')
    plt.suptitle('MDF三尺度特征', fontsize=14); plt.tight_layout(); plt.savefig('mdf_scales.png',dpi=150); plt.show()
    print("MDF可视化已保存")

if __name__ == '__main__':
    visualize_mdf_scales()
```

---

## 10. 模型评估

### 10.1 MDF在公开数据集上的性能
| 方法 | F-measure | MAE | 速度 |
| DRFI | 0.772 | 0.105 | 10s |
| MDF | 0.824 | 0.089 | 30s |

MDF在2015年达到SOTA。

### 10.2 评估代码
```python
def evaluate(saliency, gt_mask):
    ps, rs = [], []
    for t in np.linspace(0,1,256):
        b = (saliency>t).astype(np.int32)
        tp = np.sum((b==1)&(gt_mask>0.5))
        fp = np.sum((b==1)&(gt_mask<=0.5))
        fn = np.sum((b==0)&(gt_mask>0.5))
        ps.append(tp/(tp+fp+1e-8)); rs.append(tp/(tp+fn+1e-8))
    return np.array(ps), np.array(rs)
```

---

## 11. 常见问题与易错点

### Q1: 为什么三个CNN共享权值?
A: 共享权值使三个尺度的特征位于相同语义空间，便于融合。

### Q2: MDF为什么慢?
A: 逐像素处理，即使步长8，224x224图像仍需~441次前向。

### Q3: ImageNet预训练为什么重要?
A: SOD数据集小(~10K)，直接训练易过拟合。预训练提供通用视觉特征。

### Q4: MDF vs FCN?
A: MDF是逐像素分类，FCN是全卷积，FCN效率远高于MDF。

---

## 12. 学习总结

- MDF是CNN在SOD中的早期应用，证明深度特征优越性
- S-3CNN多尺度融合是核心设计
- 逐像素滑动窗口是效率瓶颈
- 预示了全卷积方法的发展方向

---

## 13. 练习题与思考题

### 练习1
题目：MDF的逐像素推理总计算量(224x224, 步长8)?
答案：ceil((224-64)/8+1)^2 = 441次CNN前向。

### 练习2
题目：为什么需要三尺度而非单尺度?
答案：单尺度缺乏上下文(仅有区域)或缺乏细节(仅有全局)。三尺度互补。

### 练习3：思考题
如何改进MDF的推理速度?
答案：1. 全卷积网络一次前向；2. 空洞卷积保持分辨率；3. 稀疏采样+插值恢复。

---

## 14. 学习路径建议

### 前置知识
1. CNN基础：卷积、池化、全连接
2. ImageNet分类与迁移学习
3. 图像多尺度分析

### 后续学习
1. 全卷积网络(FCN)
2. U-Net：编码器-解码器结构
3. FPN(特征金字塔网络)
4. BASNet：边界感知显著物体检测

### 推荐文献
1. Li G, Yu Y. "Visual saliency based on multiscale deep features." CVPR 2015.
2. Long J, et al. "Fully convolutional networks for semantic segmentation." CVPR 2015.
3. Ronneberger O, et al. "U-Net." MICCAI 2015.
