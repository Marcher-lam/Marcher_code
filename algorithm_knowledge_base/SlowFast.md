# SlowFast 双流视频理解 学习文档

> Facebook AI提出的双路径3D CNN，高效捕捉时序信息

---

## 1. 算法基础认知

### 1.1 一句话定义

SlowFast是Facebook AI于2018年提出的视频理解架构，通过"慢路径"（低帧率捕捉空间语义）和"快路径"（高帧率捕捉运动信息）的双分支设计，在Kinetics-400/600数据集上取得SOTA！

### 1.2 直觉类比

SlowFast就像人脑处理视频的"两条通道"。一条是"慢思考"通道——低帧率但看得很清楚，理解整体是什么（"一个人在跑步"）；另一条是"快反应"通道——高帧率捕捉快速动作细节（"跑步的节奏和姿态"）。两者结合既知道"是什么"又知道"怎么动"！

想象看一部电影：
- Slow路径 = 每隔1秒看一帧，重点理解场景和人物
- Fast路径 = 连续看全部帧，捕捉快速动作
- 大脑结合两者 = 完整理解电影内容

### 1.3 发展背景

- 2018年，Facebook AI的Feichtenhofer等人在论文"SlowFast Networks for Action Recognition"中提出
- 获CVPR 2019 Best Paper Finalist
- 成为视频理解基础架构
- 后续有Slowfast R-CNN等扩展

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 视频理解 → 动作识别 |
| 输出 | 动作类别 |
| 模型 | 双分支3D CNN |
| 效率 | 高FLOPs效率 |

---

## 2. 核心原理

### 2.1 核心思想

视频中有两种信息：空间语义（是什么）和时间动态（怎么动）。传统3D CNN试图同时捕捉两者，但效率低。SlowFast分开处理，更高效！

### 2.2 双路径设计

```
视频帧序列 (T帧)
    │
    ├─→ Slow路径 ──┐ 低帧率，语义信息
    │            
    │            │  横向连接融合
    ├─→ Fast路径 ─┤ 高帧率，运动信息
    │
    └───────────┘
         │
         ▼
       分类输出
```

### 2.3 路径对比

| 路径 | 帧采样 | 通道数 | 感受野 | 作用 |
|------|--------|--------|--------|------|
| Slow | 1/8 (稀疏) | 多 (3/4) | 空间大 | 语义理解 |
| Fast | 1/32 (密集) | 少 (1/4) | 时间长 | 运动捕捉 |

### 2.4 横向连接

两条路径通过侧向连接融合：

```
Slow: [B, C, T, H, W]
Fast: [B, αC, αT, H, W]
    │
    ▼
融合 → [B, C', T, H, W]
```

---

## 3. 数学公式与推导

### 3.1 时间采样

Slow路径：每隔$\tau$帧取一帧
$$x_{slow} = x[:, :, ::\tau, :, :]$$

Fast路径：使用全部帧或每隔几帧
$$x_{fast} = x[:, :, ::\tau//8, :, :]$$

### 3.2 通道数设计

$\alpha$ = 8 常数

- Slow路径通道：$C$
- Fast路径通道：$\alpha C$

总通道比：$1 + \alpha = 9$ 即Slow:8/9，Fast:1/9

### 3.3 融合公式

$$y = F(y_{slow}, y_{fast}) = \text{Conv}([y_{slow}, y_{fast}])$$

可以用：
- 相加：$y = y_{slow} + y_{fast}$
- 拼接：$y = \text{Conv}([y_{slow}; y_{fast}])$

### 3.4 输出

$$\hat{y} = \text{Softmax}(GlobalAvgPool(I_{slow} + I_{fast}))$$

---

## 4. 训练过程讲解

### 4.1 预训练权重

常用ImageNet预训练的2D ResNet初始化Slow路径。

### 4.2 数据增强

| 方法 | 说明 |
|------|------|
| 随机缩放 | 短边256-320 |
| 随机裁剪 | 224×224 |
| 随机水平翻转 | 50% |
| 颜色抖动 | 增强 |

### 4.3 训练配置

```python
# 典型配置
epochs = 256
batch_size = 8
lr = 0.1
momentum = 0.9
weight_decay = 1e-4

# 学习率调度
scheduler = CosineAnnealing(lr, epochs)
```

---

## 5. 应用场景

### 5.1 动作识别

主要应用：视频动作分类

| 数据集 | Top-1 |
|--------|-------|
| Kinetics-400 | 78.8% |
| Kinetics-600 | 80.4% |
| Something-Something | 61.8% |

### 5.2 视频检测

SlowFast R-CNN用于视频目标检测：

```python
# 伪代码
features = slowfast_backbone(video)
proposals = rpn(features)
boxes = rcnn_head(features, proposals)
```

### 5.3 时序动作定位

动作检测+定位：

```python
# 伪代码
features = slowfast(video)
start_scores, end_scores = temporal_head(features)
segments = nms(start_scores, end_scores)
```

### 5.4 对比其他方法

| 方法 | K400精度 | 计算量 |
|------|----------|--------|
| C3D | 56.8% | 中 |
| I3D | 74.3% | 高 |
| Non-local | 77.7% | 高 |
| **SlowFast** | **78.8%** | 中 |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 高精度 | SOTA水平 |
| 高效率 | FLOPs低 |
| 可解释 | 双重路径 |
| 迁移好 | 预训练可用 |
| 扩展强 | 可加更多路径 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 显存高 | 需存多帧 |
| 调参 | 需要实验 |
| 延迟 | 实际部署 |

### 6.3 注意事项

- $\tau$ 通常为8或16
- $\alpha$ 通常为8
- 慢路径用预训练2D CNN初始化效果更好

---

## 7. 调库实现（Python + PyTorchVideo）

### 7.1 torchvision

```python
import torch
import pytorchvideo.models.hub as models

# 加载预训练模型
model = models.slowfast_r50(pretrained=True)

# 输入：[B, C, T, H, W] = [1, 3, 32, 224, 224]
video = torch.randn(1, 3, 32, 224, 224)

# 前向传播
model.eval()
with torch.no_grad():
    output = model(video)

print(f"输出类别: {output.argmax(dim=-1)}")
```

### 7.2 MMAction2

```python
from mmaction.models import build_backbone

# 构建模型
backbone = build_backbone(dict(
    type='SlowFast',
    backbone=dict(
        type='resnet3d_slowfast',
        depth=50,
        pretrained='torchvideo://r50_kinetics',
        alpha=8,
        tau=8,
    )
))

# 输入
data = dict(
    imgs=torch.randn(1, 3, 32, 224, 224),
    sampling_offsets=torch.randint(0, 8, (1, 32))
)

output = backbone(data)
```

### 7.3 训练示例

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# 模型
model = models.slowfast_r101(pretrained=False)

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 训练循环
for epoch in range(100):
    model.train()
    
    for batch in dataloader:
        videos, labels = batch
        
        optimizer.zero_grad()
        output = model(videos)
        
        loss = torch.nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss={loss.item():.4f}")
```

---

## 8. 手工代码实现（理解原理）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlowPath(nn.Module):
    """Slow路径 - 低帧率，空间语义"""
    def __init__(self, backbone='resnet50'):
        super().__init__()
        
        # 简化的3D ResNet
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        return x.squeeze(-1).squeeze(-1)


class FastPath(nn.Module):
    """Fast路径 - 高帧率，运动信息"""
    def __init__(self, in_channels=3, out_channels_ratio=8):
        super().__init__()
        
        # 更少的通道数
        mid_channels = in_channels * out_channels_ratio
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=(5, 7, 7), stride=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(mid_channels, mid_channels*2, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        return x.squeeze(-1).squeeze(-1)


class SlowFast(nn.Module):
    """SlowFast双流网络"""
    def __init__(self, num_classes=400, tau=8, alpha=8):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        
        # 双路径
        self.slow_path = SlowPath()
        self.fast_path = FastPath(out_channels_ratio=alpha)
        
        # 融合层
        # Fast输出是alpha通道，Slow输出是1通道，需要融合
        self.fusion = nn.Conv3d(512 * (1 + alpha), 2048, kernel_size=1)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        
        # Slow路径：每隔tau帧采样
        x_slow = x[:, :, ::self.tau, :, :]
        slow_feat = self.slow_path(x_slow)  # [B, 512, 1, 1, 1]
        
        # Fast路径：全部帧
        x_fast = x[:, :, ::self.tau//self.alpha, :, :]
        fast_feat = self.fast_path(x_fast)  # [B, 512*alpha, 1, 1, 1]
        
        # 融合
        fused = torch.cat([slow_feat, fast_feat], dim=1)  # [B, 512*(1+alpha), 1, 1, 1]
        fused = self.fusion(fused)  # [B, 2048, 1, 1, 1]
        
        # 分类
        out = self.avgpool(fused).squeeze(-1).squeeze(-1).squeeze(-1)
        out = self.fc(out)
        
        return out


# 测试
if __name__ == "__main__":
    model = SlowFast(num_classes=400, tau=8, alpha=8)
    
    # 输入视频 [B, C, T, H, W]
    video = torch.randn(1, 3, 32, 224, 224)
    
    output = model(video)
    print(f"输出: {output.shape}")  # [1, 400]
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数: {total_params/1e6:.1f}M")
```

---

## 9. 可视化与结果理解

### 9.1路径可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 模拟特征图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Slow路径特征
slow = np.random.randn(1, 64, 8, 28, 28)
axes[0].imshow(slow[0, 0].mean(axis=0))
axes[0].set_title('Slow路径特征')

# Fast路径特征
fast = np.random.randn(1, 512, 8, 28, 28)
axes[1].imshow(fast[0, 0].mean(axis=0))
axes[1].set_title('Fast路径特征')

# 融合特征
fused = slow + fast[:, :64]
axes[2].imshow(fused[0, 0].mean(axis=0))
axes[2].set_title('融合特征')

plt.tight_layout()
plt.savefig('slowfast_features.png', dpi=100)
plt.show()
```

### 9.2 帧采样可视化

```python
# 可视化帧采样
frames = np.random.randint(0, 255, (32, 224, 224, 3))

fig, axes = plt.subplots(2, 8, figsize=(16, 4))

# Slow: 每8帧取1帧
for i, idx in enumerate(range(0, 32, 8)):
    if i < 8:
        axes[0, i].imshow(frames[idx])
        if i == 0:
            axes[0, i].set_title('Slow')

# Fast: 每1帧
for i in range(8):
    axes[1, i].imshow(frames[i*4])
    if i == 0:
        axes[1, i].set_title('Fast')

plt.tight_layout()
plt.savefig('slowfast_sampling.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| Top-1准确率 | 最高概率正确 |
| Top-5准确率 | 前5正确 |
| 计算量 | FLOPs |
| 显存 | GPU显存 |

### 10.2 评估代码

```python
from sklearn.metrics import top_k_accuracy_score

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in dataloader:
            output = model(videos)
            preds = output.argmax(dim=-1)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Top-1
    top1 = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Top-5
    # ... 需要logits
    
    return {'top1': top1}
```

---

## 11. 常见问题与易错点

### Q1: tau和alpha如何选择？

**答案**：默认tau=8, alpha=8。可根据视频速度调整。

### Q2: 显存不够？

**答案**：减小输入分辨率或batch size，或用更小的backbone。

### Q3: 训练不稳定？

**答案**：学习率不要太高，用预训练权重初始化。

### Q4: 为什么叫SlowFast？

**答案**：Slow路径慢采样（低帧率），Fast路径快采样（高帧率）。

### Q5: 可以只用一条路径吗？

**答案**：可以，但效果会下降。双路径设计是关键。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 双路径 | Slow + Fast |
| 慢 | 空间语义 |
| 快 | 时间运动 |
| 融合 | 横向连接 |

### 12.2 公式汇总

Slow采样：
$$x_{slow} = x[:, :, ::\tau, :, :]$$

Fast采样：
$$x_{fast} = x[:, :, ::\tau/\alpha, :, :]$$

融合：
$$y = Conv([y_{slow}; y_{fast}])$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. Slow路径主要捕捉：
   - A) 运动信息
   - B) 空间语义
   - C) 颜色信息

2. tau通常取：
   - A) 4
   - B) 8
   - C) 16

### 13.2 简答题

1. 解释SlowFast如何实现高效低FLOPs？
2. 为什么需要两条路径而不是一条？

### 13.3 编程题

1. 实现SlowFast的横向连接。
2. 用SlowFast做动作检测。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
视频理解基础
    ↓
3D CNN
    ↓
双流网络
    ↓
SlowFast
    ↓
SlowFast R-CNN
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| 13D CNN | 单流版 |
| Non-local | 注意力版 |
| R(2+1)D | 分解版 |
| X3D | 轻量版 |

### 14.3 扩展阅读

- Feichtenhofer et al. (2018). SlowFast Networks for Action Recognition. CVPR.

---

## 附录

### 参考

1. Feichtenhofer et al. (2018). SlowFast Networks for Action Recognition. CVPR.
2. PyTorchVideo: https://pytorchvideo.org/
3. MMAction2: https://mmaction2.readthedocs.io/

---

**文档结束**