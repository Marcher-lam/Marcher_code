# SCHED模型 学习文档

> 带短连接的多尺度显著物体检测深度模型——用HED+短连接实现显著物体端到端预测。
> 来源线索：原书第2.2.2节"SCHED：带有短连接的多尺度显著物体检测深度模型"。

---

## 1. 算法基础认知

**一句话定义：** SCHED（Short-Connection Saliency）由Hou等人于2017年提出，基于HED网络结构，在侧边输出层之间添加短连接，让深层位置信息与浅层细节特征融合，实现多尺度显著物体检测的端到端训练。

**核心创新：** 在HED的侧边输出之间引入短连接(Short Connections)，使深层(高语义、低分辨率)特征可以指导浅层(低语义、高分辨率)特征的显著性预测。

**网络结构核心理念：** "一降二升三拼四融五概率"——下采样提取多尺度特征，上采样恢复分辨率，拼接融合多尺度输出，最终输出概率图。

---

## 2. 核心原理

### 2.1 基础网络：HED

HED使用VGG16作为骨干网络，在5个stage的输出上分别连接侧边输出层进行深度监督。每个侧边输出是1x1卷积将特征图映射为单通道。

### 2.2 SCHED的短连接

在HED基础上，添加从深层到浅层的短连接：
- Side5 -> Side4 (深层指导中层)
- Side4 -> Side3 + Side2 (中层指导浅层)

实现：深层侧边输出上采样到浅层分辨率，逐元素相加。

### 2.3 "一降二升三拼四融五概率"

1. 一降：5次下采样(VGG16的5个stage)
2. 二升：5个侧边输出都上采样到原图大小
3. 三拼：拼接5个上采样后的侧边输出
4. 四融：1x1卷积融合拼接的特征
5. 五概率：Sigmoid输出显著性概率图

### 2.4 深度监督

L_total = sum_{k=1}^5 w_k * L_side_k + w_fuse * L_fuse

每个损失都是二值交叉熵损失。

---

## 3. 数学公式与推导

### 3.1 侧边输出

S_k = sigma(Conv_1x1(F_k))

### 3.2 带短连接的侧边输出

S_5' = sigma(Conv_1x1(F_5))
S_4' = sigma(Conv_1x1(F_4) + upsample(S_5'))
S_3' = sigma(Conv_1x1(F_3) + upsample(S_4'))
S_2' = sigma(Conv_1x1(F_2) + upsample(S_3'))
S_1' = sigma(Conv_1x1(F_1) + upsample(S_2'))

### 3.3 融合输出

F = [S_1', S_2', S_3', S_4', S_5']; S_fuse = sigma(Conv_1x1(F))

### 3.4 加权交叉熵

L = -beta * sum_{i in Y_+} log(s_i) - (1-beta) * sum_{i in Y_-} log(1-s_i)

其中 beta = |Y_-|/|Y| 用于平衡正负样本。

---

## 4. 训练过程讲解

### 4.1 训练策略

1. 加载VGG16 ImageNet预训练权重
2. 初始化侧边输出层、短连接层、融合层(随机)
3. 端到端训练：SGD momentum=0.9, lr=1e-6/1e-5, batch=10, epochs=20
4. 数据增强：随机翻转、裁剪、颜色抖动、尺度扰动

### 4.2 推理过程

输入图像 -> 缩放到320x320 -> 前向传播 -> 使用融合输出 -> 恢复到原图大小 -> (可选CRF后处理)

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 显著物体检测 | 端到端预测，SOTA性能 |
| 边缘检测 | HED结构天然适合 |
| 语义分割 | 多尺度特征融合可迁移 |
| 实例分割 | 显著性作为前景提议 |

---

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 端到端训练，无需后处理 | 需VGG16预训练权重 |
| 短连接有效融合多尺度特征 | 参数量大 |
| 多侧边输出提供深度监督 | 训练需大量SOD数据 |
| 推理速度快(一次前向) | 对小物体检测有限 |

性能：ECSSD上F-measure=0.911, MAE=0.058。

---

## 7. 调库实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SCHED(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 512)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Conv2d(128, 1, 1)
        self.side3 = nn.Conv2d(256, 1, 1)
        self.side4 = nn.Conv2d(512, 1, 1)
        self.side5 = nn.Conv2d(512, 1, 1)

        self.short4 = nn.Conv2d(512, 1, 1)
        self.short3 = nn.Conv2d(256, 1, 1)
        self.short2 = nn.Conv2d(128, 1, 1)
        self.short1 = nn.Conv2d(64, 1, 1)

        self.fuse = nn.Conv2d(5, 1, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h, w = x.shape[2:]
        e1 = self.conv1(x); p1 = self.pool(e1)
        e2 = self.conv2(p1); p2 = self.pool(e2)
        e3 = self.conv3(p2); p3 = self.pool(e3)
        e4 = self.conv4(p3); p4 = self.pool(e4)
        e5 = self.conv5(p4)

        s1 = F.interpolate(self.side1(e1), (h,w), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.side2(e2), (h,w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.side3(e3), (h,w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.side4(e4), (h,w), mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.side5(e5), (h,w), mode='bilinear', align_corners=False)

        sh5 = s5
        sh4 = F.interpolate(self.short4(e4), (h,w), mode='bilinear', align_corners=False) + sh5
        sh3 = F.interpolate(self.short3(e3), (h,w), mode='bilinear', align_corners=False) + sh4
        sh2 = F.interpolate(self.short2(e2), (h,w), mode='bilinear', align_corners=False) + sh3
        sh1 = F.interpolate(self.short1(e1), (h,w), mode='bilinear', align_corners=False) + sh2

        fused = torch.cat([s1+sh1, s2+sh2, s3+sh3, s4+sh4, s5+sh5], dim=1)
        fused_out = torch.sigmoid(self.fuse(fused))
        sides = [torch.sigmoid(s1+sh1), torch.sigmoid(s2+sh2), torch.sigmoid(s3+sh3),
                 torch.sigmoid(s4+sh4), torch.sigmoid(s5+sh5)]
        return fused_out, sides


def demo_sched():
    model = SCHED()
    x = torch.randn(1, 3, 224, 224)
    y = (torch.rand(1, 1, 224, 224) > 0.5).float()
    fused, sides = model(x)
    loss = F.binary_cross_entropy(fused, y)
    for s in sides: loss += F.binary_cross_entropy(s, y)
    print(f"SCHED输出: fused={fused.shape}, loss={loss:.4f}")
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    fig, axes = plt.subplots(2,3,figsize=(15,10))
    axes[0,0].imshow(x[0].permute(1,2,0).numpy()*0.2+0.5)
    axes[0,0].set_title('Input'); axes[0,0].axis('off')
    axes[0,1].imshow(y[0,0].detach().numpy(),cmap='gray')
    axes[0,1].set_title('GT'); axes[0,1].axis('off')
    im = axes[0,2].imshow(fused[0,0].detach().numpy(),cmap='jet')
    axes[0,2].set_title('Fused'); axes[0,2].axis('off')
    plt.colorbar(im,ax=axes[0,2],fraction=0.046)
    for i in range(5):
        r,c = 1+i//3, i%3
        im = axes[r,c].imshow(sides[i][0,0].detach().numpy(),cmap='jet')
        axes[r,c].set_title(f'Side{i+1}'); axes[r,c].axis('off')
        plt.colorbar(im,ax=axes[r,c],fraction=0.046)
    plt.suptitle('SCHED多侧边输出',fontsize=14); plt.tight_layout()
    plt.savefig('sched_output.png',dpi=150); plt.show()

if __name__ == '__main__':
    demo_sched()
```

---

## 8. 手工代码实现（NumPy）

```python
import numpy as np
from scipy.ndimage import gaussian_filter, zoom


class SCHEDNumpy:
    def compute_saliency(self, image):
        if image.max() > 1.0: image /= 255.0
        gray = np.mean(image, axis=2)
        h, w = gray.shape
        features = []
        for i in range(5):
            sigma = 2**i
            b1 = gaussian_filter(gray, sigma)
            b2 = gaussian_filter(gray, sigma*2) if i>0 else gaussian_filter(gray, 0.5)
            features.append(np.abs(b1-b2))
        sides = []
        for feat in features:
            if feat.shape != (h,w):
                sides.append(zoom(feat, (h/feat.shape[0], w/feat.shape[1])))
            else:
                sides.append(feat)
        sh = [None]*5; sh[4] = sides[4]
        for i in range(3,-1,-1): sh[i] = sides[i] + sh[i+1]
        fused = np.mean(sh, axis=0)
        fused = gaussian_filter(fused, 1)
        return (fused-fused.min())/(fused.max()-fused.min()+1e-8)


def demo_numpy():
    np.random.seed(42)
    img = np.random.rand(64,64,3); img[20:45,20:45]=[0.8,0.2,0.2]
    m = SCHEDNumpy(); s = m.compute_saliency(img)
    print(f"SCHED手工: [{s.min():.3f}, {s.max():.3f}]")
    print(f"前景: {s[20:45,20:45].mean():.3f}, 背景: {s[:20,:20].mean():.3f}")

if __name__ == '__main__':
    demo_numpy()
```

---

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom


def visualize_sched_sc():
    np.random.seed(42)
    img = np.ones((96,96,3))*0.2; img[25:72,25:72]=[0.7,0.3,0.3]
    gray = np.mean(img, axis=2); h,w = gray.shape

    features = []
    for i in range(5):
        sigma = 2**i
        b1 = gaussian_filter(gray, sigma)
        b2 = gaussian_filter(gray, sigma*2) if i>0 else gaussian_filter(gray, 0.5)
        features.append(np.abs(b1-b2))
    sides = []
    for feat in features:
        if feat.shape != (h,w): sides.append(zoom(feat, (h/feat.shape[0],w/feat.shape[1])))
        else: sides.append(feat)

    no_sc = np.mean(sides, axis=0)
    sh = [None]*5; sh[4]=sides[4]
    for i in range(3,-1,-1): sh[i]=sides[i]+sh[i+1]
    with_sc = np.mean(sh, axis=0)

    fig, axes = plt.subplots(2,4,figsize=(16,8))
    axes[0,0].imshow(img); axes[0,0].set_title('(a) Input'); axes[0,0].axis('off')
    axes[0,1].imshow(sides[0],cmap='hot'); axes[0,1].set_title('(b) Side1'); axes[0,1].axis('off')
    axes[0,2].imshow(sides[2],cmap='hot'); axes[0,2].set_title('(c) Side3'); axes[0,2].axis('off')
    axes[0,3].imshow(sides[4],cmap='hot'); axes[0,3].set_title('(d) Side5'); axes[0,3].axis('off')
    axes[1,0].imshow(sh[0],cmap='jet'); axes[1,0].set_title('(e) Short Side1'); axes[1,0].axis('off')
    axes[1,1].imshow(sh[1],cmap='jet'); axes[1,1].set_title('(f) Short Side2'); axes[1,1].axis('off')
    no_sc = (no_sc-no_sc.min())/(no_sc.max()-no_sc.min()+1e-8)
    im = axes[1,2].imshow(no_sc,cmap='jet')
    axes[1,2].set_title('(g) w/o Short'); axes[1,2].axis('off')
    plt.colorbar(im,ax=axes[1,2],fraction=0.046)
    with_sc = (with_sc-with_sc.min())/(with_sc.max()-with_sc.min()+1e-8)
    im = axes[1,3].imshow(with_sc,cmap='jet')
    axes[1,3].set_title('(h) w/ Short'); axes[1,3].axis('off')
    plt.colorbar(im,ax=axes[1,3],fraction=0.046)
    plt.suptitle('SCHED短连接效果',fontsize=14); plt.tight_layout()
    plt.savefig('sched_sc.png',dpi=150); plt.show()
    print("SCHED可视化已保存")

if __name__ == '__main__':
    visualize_sched_sc()
```

---

## 10. 模型评估

### 10.1 SCHED性能
| 方法 | ECSSD(F) | ECSSD(MAE) | DUT-OMRON(F) |
| HED | 0.851 | 0.072 | 0.694 |
| MDF | 0.824 | 0.089 | 0.651 |
| SCHED | 0.911 | 0.058 | 0.754 |

短连接带来+6% F-measure相比HED。

### 10.2 评估代码
```python
def evaluate(saliency, gt_mask):
    T = 2*saliency.mean()
    b = (saliency>T).astype(np.int32)
    tp=np.sum((b==1)&(gt_mask>0.5)); fp=np.sum((b==1)&(gt_mask<=0.5)); fn=np.sum((b==0)&(gt_mask>0.5))
    prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8); f=1.3*prec*rec/(0.3*prec+rec+1e-8)
    mae=np.mean(np.abs(saliency-gt_mask))
    return prec,rec,f,mae
```

---

## 11. 常见问题与易错点

### Q1: 短连接 vs 残差连接(ResNet)?
A: ResNet的残差连接解决梯度消失。SCHED的短连接传播语义信息，解决多尺度融合。

### Q2: 为什么要5个侧边输出?
A: VGG16有5个stage，分辨率从1到1/16，覆盖细到粗的多尺度信息。

### Q3: SCHED vs U-Net?
A: U-Net用编码器-解码器+跳跃连接。SCHED用HED风格侧边输出+短连接，所有输出独立监督。

### Q4: 短连接为什么提升性能?
A: 深层语义信息通过短连接指导浅层，抑制背景噪声，突出显著物体。

### Q5: 类别平衡权重的作用?
A: 显著物体只占~20%，不加权网络偏向预测背景。

---

## 12. 学习总结

### 12.1 核心要点
- HED基础 + 短连接创新 = SCHED
- 短连接让深层语义信息指导浅层细节
- 多侧边输出+深度监督提供更好训练信号
- 端到端，一次前向得到全图显著图

### 12.2 贡献
- 证明多尺度融合中信息流动的重要性
- 短连接简洁有效

---

## 13. 练习题与思考题

### 练习1
题目：Stage4短连接的参数量?
答案：512通道输入，1通道输出，1x1卷积：512*1+1=513个参数。

### 练习2
题目：SCHED能否用于视频显著性检测?
答案：可以。加3D卷积、光流特征或时序一致性损失。

### 练习3：思考题
题目：如果将短连接改为双向(浅层也向深层传递)，会有帮助吗?
答案：可能没有。深层已有全局信息，浅层细节对深层帮助有限，且增加参数和计算量。

---

## 14. 学习路径建议

### 前置知识
1. VGGNet网络结构
2. HED边缘检测
3. 多尺度特征融合
4. 深度监督训练

### 后续学习
1. DSS：SCHED的改进版
2. BASNet：边界感知显著物体检测
3. U^2-Net：嵌套U结构
4. TransalNet：Transformer在SOD中的应用

### 推荐文献
1. Hou Q, et al. "Deeply supervised salient object detection with short connections." CVPR 2017.
2. Xie S, Tu Z. "Holistically-nested edge detection." ICCV 2015.
3. Qin X, et al. "BASNet: Boundary-aware salient object detection." CVPR 2019.
