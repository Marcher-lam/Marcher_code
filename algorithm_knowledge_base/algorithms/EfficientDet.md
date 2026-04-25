# EfficientDet 学习文档

> 谷歌EfficientNet团队打造的高效目标检测，平衡精度与速度

---

## 1. 算法基础认知

### 1.1 一句话定义

EfficientDet是谷歌EfficientNet团队于2020年提出的目标检测模型，在保持高精度的同时大幅优化效率，使用BiFPN和compound scaling实现SOTA水平。

### 1.2 直觉类比

EfficientDet就像一辆"省油的赛车"。它不是简单地减少引擎功率（降低精度），而是通过精心设计的"空气动力学"（BiFPN让特征流通更顺畅）+ "涡轮增压"（更高效的缩放方法）实现既快又准！

传统目标检测模型就像一辆重型卡车——虽然能装更多货物（检测更多目标），但耗油严重（计算量大）。EfficientDet通过：
1. **双向高速公路**（BiFPN）：让信息双向流动，更高效
2. **智能调度**（compound scaling）：同时调整所有部件，而不是只改一个

### 1.3 发展背景

- 2020年，谷歌Tan和Le在论文"EfficientDet: Scalable and Efficient Object Detection"中提出
- 继承EfficientNet的缩放思想
- 在COCO数据集上达到新SOTA
- 从D0到D7，形成完整系列

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 目标检测 → one-stage |
| 输出 | 检测框+类别+置信度 |
| 模型 | EfficientNet + BiFPN |
| 特点 | 高效+高精度 |

---

## 2. 核心原理

### 2.1 为什么需要新FPN？

**传统FPN的问题**：
- 只有单向信息流（top-down）
- 特征融合方式简单
- 难以捕捉多尺度特征

**BiFPN的创新**：
- 双向流动（top-down + bottom-up）
- 可学习的融合权重
- 特征跨尺度连接

### 2.2 BiFPN详解

```
普通FPN:                    BiFPN:
                          ┌────P3────┐
    ┌───P3──┐               ↑  融合  ↓  ↑  融合  ↓
    │ 融合 ←───┤          ┌──P2──┐   ←─────→  ┌──P2──┐
    └───P2──┘             ↑  融合  ↓  ↑  融合  ↓  融合
                     P1 ─┘    └─────→  P1 ─┘
```

**核心特点**：
1. **双向流动**：P3→P2→P1（top-down）然后P1→P2→P3（bottom-up）
2. **可学习权重**每条边有权重 $W_i$，更灵活
3. **特征重利用**：输入直接连到输出

### 2.3 特征融合公式

$$O = \sum_i \frac{W_i \cdot I_i}{\sum_j W_j}$$

其中 $I_i$ 是输入特征，$W_i$ 是可学习权重。

### 2.4 Compound Scaling

同时缩放多个维度：

| 参数 | 缩放法则 |
|------|----------|
| Backbone | $2^{depth}$ |
| BiFPN | $1.35^{width}$ |
| Box/Class Head | 同 backbone |
| 分辨率 | $2^{resolution}$ |

**D0-D7配置**：

| 模型 | Backbone | BiFPN | 分辨率 | 参数量 | AP |
|------|---------|-------|-------|--------|-----|
| D0 | B0 | 64 | 512 | 3.9M | 33.8 |
| D1 | B1 | 88 | 640 | 6.5M | 38.2 |
| D2 | B2 | 112 | 768 | 8.1M | 40.4 |
| D3 | B3 | 160 | 896 | 12M | 45.1 |
| D4 | B4 | 224 | 1024 | 20M | 48.5 |
| D5 | B5 | 288 | 1280 | 34M | 50.7 |
| D6 | B6 | 384 | 1400 | 44M | 52.3 |
| D7 | B6 | 384 | 1536 | 54M | 53.7 |

---

## 3. 数学公式与推导

### 3.1 BiFPN单层计算

给定输入特征 $I^{(l)}_i$，输出特征 $O^{(l)}$：

$$O^{(l)} = \text{Conv}(\sum_i \frac{W^{(l)}_{i,j} \cdot I^{(l)}_i)$$

其中 $W^{(l)}_{i,j}$ 是可学习权重。

### 3.2 多尺度特征融合

多层特征 $\{P_3, P_4, P_5\}$ 融合为：

$$P^{out}_4 = \text{Conv}(\frac{W_1 \cdot P^{in}_4 + W_2 \cdot \text{Resize}(P^{in}_5)}{W_1 + W_2})$$

### 3.3 检测头

$$Box = \text{BoxHead}(P_i)$$
$$Class = \text{ClassHead}(P_i)$$

每个尺度的 $P_i$ 预测对应大小的物体。

### 3.4 损失函数

分类损失（Focal Loss）：
$$L_{cls} = -\sum(1-p_t)^\gamma \log(p_t)$$

回归损失（Box Loss）：
$$L_{reg} = \sum SmoothL1(|boxes - ground|])$$

---

## 4. 训练过程讲解

### 4.1 数据增强

```python
# 标准增强策略
train_augmentation = [
    AutoAugment(),      # 自动增强
    Mosaic(),         # 4图拼接
    CutMix(),        # 剪切粘贴
    Flip(),          # 翻转
]
```

### 4.2 训练配置

```python
# 训练参数
config = {
    'batch_size': 64,
    'learning_rate': 0.88,     # 对于SGD
    'momentum': 0.9,
    'weight_decay': 4e-5,
    'epochs': 300,
    'warmup_epochs': 5,
}
```

### 4.3 优化器

```python
# SGD配置
optimizer = SGD(
    lr=0.88,
    momentum=0.9,
    weight_decay=4e-5,
    nesterov=True
)

# 学习率调度
scheduler = CosineAnnealingLR(optimizer, T_max=300)
```

---

## 5. 应用场景

### 5.1 移动端部署

```python
# D0/D1 适合移动端
model = timm.create_model('efficientdet_d0', pretrained=True)

# 在移动设备上推理
result = model.detect(image, score_threshold=0.5)
for box, score, class_id in result:
    print(f"Class: {class_id}, Score: {score:.2f}, Box: {box}")
```

### 5.2 云端服务

```python
# D4/D7 适合服务端
model = timm.create_model('efficientdet_d4', pretrained=True)

# 批量检测
results = model.batch_detect(images)
```

### 5.3 对比其他模型

| 模型 | AP | FLOPs | 参数量 | 速度 |
|------|-----|-------|--------|------|
| YOLOv4 | 43.5 | 65B | 64M | 中 |
| RetinaNet | 39.1 | 27B | 30M | 快 |
| FCOS | 44.6 | 80B | 90M | 慢 |
| **EfficientDet D3** | **45.1** | **52B** | **12M** | **快** |
| **EfficientDet D7** | **53.7** | **380B** | **54M** | **中** |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 高效 | 比同类精度更高，FLOPs更低 |
| 可扩展 | D0-D7系列满足不同需求 |
| 效果好 | COCO新SOTA |
| 特征融合好 | BiFPN双向流通 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 复杂度高 | BiFPN比普通FPN复杂 |
| 显存需求 | 大模型需要大量显存 |
| 调参难 | compound scaling需要经验 |

### 6.3 注意事项

- D0-D2适合移动端
- D4-D7适合服务器
- 分辨率影响很大

---

## 7. 调库实现（Python）

### 7.1 timm库使用

```bash
pip install timm
```

```python
import timm
import torch

# 加载预训练模型
model = timm.create_model('efficientdet_d0', pretrained=True)
model.eval()

# 检测
with torch.no_grad():
    output = model(image)
    
# 解析结果
boxes = output[..., 0:4]   # [x1, y1, x2, y2]
scores = output[..., 4]       # 置信度
class_ids = output[..., 5]   # 类别
```

### 7.2 完整推理代码

```python
import timm
import cv2
import torch
import numpy as np

def detect_efficientdet(image_path, model_name='efficientdet_d0'):
    # 加载模型
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 预处理
    input_size = {'efficientdet_d0': 512,
                'efficientdet_d4': 1024}[model_name]
    
    img_resized = cv2.resize(img, (input_size, input_size))
    img_normalized = img_resized / 255.0
    img_tensor = torch.FloatTensor(img_normalized).permute(2, 0, 1).unsqueeze(0)
    
    # 检测
    with torch.no_grad():
        output = model(img_tensor)
    
    # 解析结果（简化）
    results = []
    for det in output[0]:
        if det[4] > 0.5:  # 置信度阈值
            results.append({
                'box': det[:4].numpy(),
                'score': det[4].item(),
                'class': det[5].item()
            })
    
    return results


# 使用
results = detect_efficientdet('test.jpg')
for r in results:
    print(f"Box: {r['box']}, Score: {r['score']:.2f}, Class: {r['class']}")
```

### 7.3 训练示例

```python
import timm
import torch
from torch.utils.data import DataLoader

# 创建模型
model = timm.create_model('efficientdet_d0', num_classes=80)

# 数据
class DetDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        return torch.randn(3, 512, 512), torch.tensor([0, 0, 0.5, 0.5, 1.0, 0])

loader = DataLoader(DetDataset(), batch_size=8)

# 训练循环
optimizer = torch.optim.SGD(model.parameters(), lr=0.88, momentum=0.9)

for epoch in range(10):
    model.train()
    total_loss = 0
    
    for images, targets in loader:
        optimizer.zero_grad()
        
        # 前向（简化）
        outputs = model(images)
        
        # 计算损失（简化）
        loss = sum(o.mean() for o in outputs if o.numel() > 0)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss={total_loss/len(loader):.4f}")
```

---

## 8. 手工代码实现（理解原理）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN(nn.Module):
    """BiFPN简化版"""
    def __init__(self, in_channels_list, out_channels, num_layers=3):
        super().__init__()
        
        # 特征适配层
        self.adapt = nn.ModuleDict({
            str(i): nn.Conv2d(c, out_channels, 1)
            for i, c in enumerate(in_channels_list)
        })
        
        # 可学习权重
        self.weights = nn.Parameter(torch.ones(num_layers * 2))
        
        # 融合层
        self.fusion = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, features):
        """features: [P3, P4, P5]
        
        BiFPN: top-down + bottom-up
        """
        # 适配通道
        adapted = [self.adapt[str(i)](f) for i, f in enumerate(features)]
        
        # Top-down融合
        td_out = [adapted[-1]]
        for i in range(len(adapted)-2, -1, -1):
            td_out.append(adapted[i] + F.interpolate(td_out[0], 
                                             size=adapted[i].shape[-2:]))
        
        # Bottom-up融合
        out = [td_out[-1]]
        for i in range(len(td_out)-1):
            out.append(td_out[i] + out[0])
        
        return [self.fusion(o) for o in out]


class EfficientDetModel(nn.Module):
    """简化版EfficientDet"""
    def __init__(self, num_classes=80):
        super().__init__()
        
        # Backbone (简化)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # BiFPN
        self.bifpn = BiFPN([128, 128, 128], 64)
        
        # 检测头
        self.head = nn.Conv2d(64, (4+1+num_classes), 1)
    
    def forward(self, x):
        # 特征提取
        features = []
        for layer in self.backbone:
            x = layer(x)
            if x.shape[-1] <= 32:  # 多尺度输出
                features.append(x)
        
        # BiFPN
        fused = self.bifpn(features)
        
        # 检测头
        outputs = [self.head(f) for f in fused]
        
        return outputs


# 测试
if __name__ == "__main__":
    model = EfficientDetModel(num_classes=80)
    
    # 输入
    x = torch.randn(1, 3, 512, 512)
    
    # 前向
    outputs = model(x)
    
    print("输出数量:", len(outputs))
    print("每个输出形状:", outputs[0].shape)
    
    # 参数量
    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total/1e6:.1f}M")
```

---

## 9. 可视化与结果理解

### 9.1 检测结果可视化

```python
import cv2
import matplotlib.pyplot as plt

def visualize_detections(image, detections, save_path='detections.jpg'):
    """可视化检测结果"""
    img = cv2.imread(image) if isinstance(image, str) else image.copy()
    
    for det in detections:
        box = det['box']
        score = det['score']
        class_id = det['class']
        
        # 画框
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 画标签
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(img, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(save_path, dpi=100)
    plt.show()


# 使用
visualize_detections('test.jpg', results)
```

### 9.2 模型对比可视化

```python
import matplotlib.pyplot as plt

# 模型性能数据
models = ['D0', 'D1', 'D2', 'D3', 'D4', 'D7']
aps = [33.8, 38.2, 40.4, 45.1, 48.5, 53.7]
flops = [3.9, 6.5, 8.1, 12, 20, 54]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# AP对比
ax1.bar(models, aps, color='steelblue')
ax1.set_ylabel('AP')
ax1.set_title('精度对比')

# FLOPs对比
ax2.bar(models, flops, color='coral')
ax2.set_ylabel('FLOPs (B)')
ax2.set_title('计算量对比')

plt.tight_layout()
plt.savefig('efficientdet_comparison.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| AP | COCO精度 |
| AP50 | IoU=0.5 |
| AP75 | IoU=0.75 |
| AP_s/m/l | 小/中/大物体 |
| FLOPs | 计算量 |
| 参数量 | 模型大小 |

### 10.2 评估代码

```python
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco(model, coco_gt, image_dir):
    """COCO数据集评估"""
    model.eval()
    
    results = []
    for img_id in coco_gt.getImgIds():
        # 加载图像
        img_info = coco_gt.loadImgs(img_id)[0]
        img = cv2.imread(f"{image_dir}/{img_info['file_name']}")
        
        # 检测
        with torch.no_grad():
            dets = model.detect(img)
        
        results.extend(dets)
    
    # 评估
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats


# 评估
stats = evaluate_coco(model, coco_gt, image_dir)
print(f"AP: {stats[0]:.2f}")
print(f"AP50: {stats[1]:.2f}")
```

---

## 11. 常见问题与易错点

### Q1: 如何选择模型版本？

**答案**：
- 移动端：D0-D2
- 服务器：D4-D7
- 追求精度：D7

### Q2: 为什么显存不够？

**答案**：大分辨率需要大显存。D7需要24GB+。

### Q3: 如何提升精度？

**答案**：增加数据增强、更大的预训练 backbone。

### Q4: 和YOLOv4比哪个好？

**答案**：精度差不多时EfficientDet FLOPs更低。

### Q5: 训练需要什么硬件？

**答案**：D0可用单卡3090，D7需要8卡A100。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心创新 | BiFPN双向融合 |
| 缩放方法 | Compound scaling |
| 模型系列 | D0-D7 |
| 优势 | 高效高精度 |

### 12.2 公式汇总

BiFPN融合：
$$O = \sum_i \frac{W_i \cdot I_i}{\sum_j W_j}$$

Compound scaling：
$$scale\_factor = \phi^{depth} \times \phi^{width}$$

---

## 13. 练习题

### 13.1 选择题

1. BiFPN的创新是：
   - A) 更深的网络
   - B) 双向特征流
   - C) 更大的分辨率

2. Compound scaling同时调整：
   - A) 深度和宽度
   - B) 只调整宽度
   - C) 只调整深度

### 13.2 简答题

1. 解释BiFPN和FPN的区别。
2. EfficientDet比YOLOv4好在哪？

### 13.3 编程题

1. 实现简化的BiFPN。
2. 在实际数据集上测试EfficientDet。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
目标检测基础
    ↓
FPN/RetinaNet
    ↓
EfficientNet
    ↓
BiFPN原理
    ↓
EfficientDet应用
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| EfficientNet | Backbone |
| YOLOv4 | 对手 |
| RetinaNet | 前辈 |
| FCOS | Anchor-free |

### 14.3 扩展阅读

- Tan et al. (2020). EfficientDet: Scalable and Efficient Object Detection

---

## 附录

### 参考

1. Tan et al. (2020). EfficientDet: Scalable and Efficient Object Detection
2. https://github.com/google/automl/tree/master/efficientdet
3. https://github.com/rwightman/pytorch-image-models

---

**文档结束**