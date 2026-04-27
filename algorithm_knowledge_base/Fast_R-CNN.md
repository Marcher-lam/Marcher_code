# Fast R-CNN 学习文档

## 1. 算法基础认知

### 1.1 定义

Fast R-CNN（Fast Region-based Convolutional Network）是 2015 年提出的目标检测算法，由 Ross Girshick 提出，是 R-CNN 和 SPPNet 的改进版本。其核心思想是**端到端训练**的目标检测，利用 RoI Pooling 将不同尺寸的区域提议映射为固定维度的特征。

### 1.2 直观类比

将 Fast R-CNN 想象为一个**智能质检员**：
1. 接收图片（整张图像）
2. 快速扫描定位可疑区域（区域提议）
3. 对每个区域进行特征提取和分类（RoI Pooling + 全连接层）
4. 输出物体类别和位置

### 1.3 历史背景

- **R-CNN**（2014）：首次将 CNN 应用于目标检测，但每个区域分别提取特征，速度很慢
- **SPPNet**（2014）：空间金字塔池化，但无法端到端训练
- **Fast R-CNN**（2015）：结合两者优点，实现端到端训练，速度提升 10 倍

---

## 2. 核心原理

### 2.1 网络结构

Fast R-CNN 的网络结构如下：

```
输入图像
    ↓
卷积特征提取（共享卷积）
    ↓
区域提议（Selective Search）
    ↓
RoI Pooling（映射为固定尺寸）
    ↓
全连接层（分类 + Bounding Box 回归）
    ↓
输出：类别概率 + 边界框偏移
```

### 2.2 核心创新

1. **RoI Pooling**：将不同尺寸的区域提议映射为固定维度（如 $7 \times 7 \times 512$）
2. **多任务损失**：同时优化分类和回归
3. **特征共享**：整张图像只提取一次特征

### 2.3 与 R-CNN 对比

| 方面 | R-CNN | Fast R-CNN |
|------|-------|-----------|
| 特征提取 | 每个区域分别提取 | 整张图共享一次 |
| 训练方式 | 分阶段 | 端到端 |
| 速度 | 慢（~1fps） | 快（~7fps） |
| 特征维度 | 固定 | 固定（RoI Pooling） |

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $I$ | 输入图像 |
| $F$ | 卷积特征图 |
| $R$ | 区域提议（RoI） |
| $x$ | RoI Pooling 后的特征 |
| $p$ | 类别概率分布 |
| $t$ | Bounding Box 回归参数 |

### 3.2 RoI Pooling

对于每个 RoI $r = (x, y, w, h)$，其中 $(x, y)$ 为左上角坐标，$(w, h)$ 为宽高：

1. 将 RoI 划分为 $H \times W$ 个网格
2. 每个网格进行最大池化

设网格大小为：
$$
h_{grid} = \frac{h}{H}, \quad w_{grid} = \frac{w}{W}
$$

池化后得到固定维度 $H \times W$ 的特征。

### 3.3 分类损失

对于 $K$ 个类别（含背景），使用 Softmax 交叉熵损失：

$$
L_{cls}(p, p^*) = -\sum_{c=1}^{K} \mathbb{1}\{c = p^*\} \log p_c = -\log p_{p^*}
$$

其中 $p^*$ 为真实类别，$p_c$ 为预测概率。

### 3.4 Bounding Box 回归

对于类别 $c$，预测回归参数 $t_c = (t_{cx}, t_{cy}, t_{cw}, t_{ch})$：

- $t_{cx}$：中心 x 偏移（log 空间）
- $t_{cy}$：中心 y 偏移（log 空间）
- $t_{cw}$：宽度缩放（log 空间）
- $t_{ch}$：高度缩放（log 空间）

真实偏移 $t^*$ 与 RoI $r$ 和真实框 $g$ 的关系：

$$
t_{cx}^* = \frac{g_x - r_x}{r_w}, \quad t_{cy}^* = \frac{g_y - r_y}{r_h}
$$
$$
t_{cw}^* = \log(\frac{g_w}{r_w}), \quad t_{ch}^* = \log(\frac{g_h}{r_h})
$$

回归损失（smooth L1）：

$$
L_{loc}(t, t^*) = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t_i - t_i^*)
$$

其中：
$$
\text{smooth}_{L_1}(x) = \begin{cases} 0.5 x^2 & |x| < 1 \\ |x| - 0.5 & |x| \geq 1 \end{cases}
$$

### 3.5 多任务损失

$$
L(p, p^*, t, t^*) = L_{cls}(p, p^*) + \mathbb{1}\{p^* > 0\} L_{loc}(t, t^*)
$$

其中 $\mathbb{1}\{p^* > 0\}$ 表示非背景类别才计算回归损失。

---

## 4. 训练过程讲解

### 4.1 数据准备

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FastRCNNDataset(Dataset):
    """Fast R-CNN 数据集"""
    
    def __init__(self, image_paths, annotations):
        self.image_paths = image_paths
        self.annotations = annotations
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image = self.load_image(self.image_paths[idx])
        boxes, labels = self.annotations[idx]
        
        return image, boxes, labels
    
    def load_image(self, path):
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert('RGB')
        return np.array(img)
```

### 4.2 网络定义（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16FeatureExtractor(nn.Module):
    """VGG16 特征提取器"""
    
    def __init__(self):
        super().__init__()
        # 前 4 个卷积块（去掉最后的 pool5）
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
    
    def forward(self, x):
        return self.features(x)

class RoIPooling(nn.Module):
    """RoI Pooling 层"""
    
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size
    
    def forward(self, features, rois):
        """
        features: [B, C, H, W]
        rois: [N, 5] (batch_idx, x1, y1, x2, y2)
        """
        batch_idx = rois[:, 0].long()
        rois = rois[:, 1:]
        
        # 将坐标映射到特征图尺寸
        # 假设特征图已经下采样 16 倍
        rois = rois / 16.0
        
        output = []
        for i in range(len(rois)):
            x1, y1, x2, y2 = rois[i]
            roi_feature = features[batch_idx[i], :, y1:y2, x1:x2]
            
            # 简单池化
            roi_pooled = F.adaptive_max_pool2d(
                roi_feature.unsqueeze(0), 
                output_size=(self.out_size, self.out_size)
            )
            output.append(roi_pooled.squeeze(0))
        
        return torch.stack(output)

class FastRCNN(nn.Module):
    """Fast R-CNN 网络"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.extractor = VGG16FeatureExtractor()
        self.roi_pool = RoIPooling(7)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        
        # 分类头
        self.classifier = nn.Linear(4096, num_classes)
        
        # 回归头
        self.bbox_regressor = nn.Linear(4096, 4 * num_classes)
    
    def forward(self, x, rois):
        """
        x: [B, 3, H, W]
        rois: [N, 5]
        """
        # 特征提取
        features = self.extractor(x)
        
        # RoI Pooling
        roi_features = self.roi_pool(features, rois)
        
        # 展平
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        # 全连接
        fc_features = self.fc(roi_features)
        
        # 分类和回归
        cls_scores = self.classifier(fc_features)
        bbox_preds = self.bbox_regressor(fc_features)
        
        return cls_scores, bbox_preds
```

### 4.3 损失函数定义

```python
class FastRCNNLoss(nn.Module):
    """Fast R-CNN 损失函数"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    def smooth_l1_loss(self, pred, target, beta=1.0):
        """Smooth L1 Loss"""
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
        return loss.sum(dim=1)
    
    def forward(self, cls_scores, bbox_preds, labels, bbox_targets, rois):
        """
        cls_scores: [N, num_classes]
        bbox_preds: [N, 4*num_classes]
        labels: [N]
        bbox_targets: [N, 4]
        rois: [N, 5]
        """
        # 分类损失
        cls_loss = F.cross_entropy(cls_scores, labels)
        
        # 正样本（不含背景）才计算回归损失
        pos_mask = labels > 0
        if pos_mask.sum() > 0:
            pos_labels = labels[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            pos_bbox_targets = bbox_targets[pos_mask]
            
            # 每个类别的回归
            bbox_loss = 0
            for c in range(1, self.num_classes):
                c_mask = pos_labels == c
                if c_mask.sum() > 0:
                    c_pred = pos_bbox_preds[c_mask, 4*c:4*(c+1)]
                    c_target = pos_bbox_targets[c_mask]
                    bbox_loss += self.smooth_l1_loss(c_pred, c_target).mean()
            
            bbox_loss = bbox_loss / (self.num_classes - 1)
        else:
            bbox_loss = torch.tensor(0.0, device=cls_scores.device)
        
        # 总损失
        total_loss = cls_loss + bbox_loss
        
        return total_loss, cls_loss, bbox_loss
```

### 4.4 训练循环

```python
def train_fast_rcnn():
    """训练 Fast R-CNN"""
    import torch.optim as optim
    
    # 创建模型
    model = FastRCNN(num_classes=21)  # VOC 20 类 + 背景
    model.train()
    
    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # 损失函数
    criterion = FastRCNNLoss(num_classes=21)
    
    # 训练循环
    for epoch in range(10):
        total_loss = 0
        
        for batch_idx in range(100):
            # 前向传播
            optimizer.zero_grad()
            
            # 模拟数据
            images = torch.randn(2, 3, 600, 1000)
            rois = torch.randint(0, 2, (10, 5)).float()
            labels = torch.randint(0, 21, (10,))
            bbox_targets = torch.randn(10, 4)
            
            # 前向传播
            cls_scores, bbox_preds = model(images, rois)
            
            # 计算损失
            loss, cls_loss, bbox_loss = criterion(
                cls_scores, bbox_preds, labels, bbox_targets, rois
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/100:.4f}")

train_fast_rcnn()
```

---

## 5. 应用场景

### 5.1 目标检测

Fast R-CNN 主要用于目标检测：
- 通用物体检测（PASCAL VOC、COCO）
- 人脸检测
- 行人检测

### 5.2 场景理解

- 自动驾驶中的障碍物检测
- 机器人视觉中的物体识别
- 医学图像中的病灶检测

### 5.3 实例分割

Fast R-CNN 的变体可以扩展为 Mask R-CNN，实现实例分割。

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 端到端训练 | 整个网络可以一起优化 |
| 特征共享 | 整张图像只提取一次特征 |
| 多任务损失 | 同时优化分类和定位 |
| 精度高 | 相对于 R-CNN 精度相当或更高 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 区域提议慢 | 仍依赖 Selective Search |
| 难以实时 | ~7fps，无法实时处理 |
| 内存占用 | 需要存储大量特征 |
| 固定感受野 | 对小物体效果差 |

---

## 7. 调库实现

### 7.1 使用 torchvision

```python
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def train_with_torchvision():
    """使用 torchvision 训练 Fast R-CNN"""
    
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 修改类别数
    num_classes = 21  # VOC
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        model.roi_heads.box_predictor.in_features,
        num_classes
    )
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print("模型加载成功")
    return model

train_with_torchvision()
```

### 7.2 推理

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

def infer_with_model():
    """使用模型推理"""
    
    # 加载模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # 加载图像
    img = Image.open('test.jpg').convert('RGB')
    img_tensor = F.to_tensor(img).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # 解析结果
    pred = predictions[0]
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']
    
    # 过滤低置信度
    keep = scores > 0.5
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    print(f"检测到 {len(boxes)} 个目标")
    return boxes, labels, scores

infer_with_model()
```

---

## 8. 手工代码实现

### 8.1 RoI Pooling 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualRoIPooling(nn.Module):
    """手动实现 RoI Pooling"""
    
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size
    
    def forward(self, features, rois):
        """
        features: [B, C, H, W]
        rois: [N, 5] (batch_idx, x1, y1, x2, y2)
        """
        batch_indices = rois[:, 0].long()
        rois_coords = rois[:, 1:]
        
        # 量化到特征图坐标
        # 假设下采样率为 16（典型 VGG 设置）
        scale = 16.0
        rois_coords = rois_coords / scale
        
        # 边界裁剪
        _, C, H, W = features.shape
        rois_coords[:, 0] = rois_coords[:, 0].clamp(0, W - 1)
        rois_coords[:, 1] = rois_coords[:, 1].clamp(0, H - 1)
        rois_coords[:, 2] = rois_coords[:, 2].clamp(0, W - 1)
        rois_coords[:, 3] = rois_coords[:, 3].clamp(0, H - 1)
        
        # 分箱池化
        output = []
        for i in range(len(rois_coords)):
            batch_idx = batch_indices[i]
            x1, y1, x2, y2 = rois_coords[i]
            
            # 提取 RoI 特征
            roi_feat = features[batch_idx, :, y1.int():y2.int(), x1.int():x2.int()]
            
            # 自适应池化到固定大小
            if roi_feat.numel() > 0:
                pooled = F.adaptive_max_pool2d(
                    roi_feat.unsqueeze(0),
                    output_size=(self.out_size, self.out_size)
                )
            else:
                pooled = torch.zeros(1, C, self.out_size, self.out_size)
            
            output.append(pooled.squeeze(0))
        
        return torch.stack(output)

# 验证
features = torch.randn(2, 512, 37, 62)  # 典型 VGG 特征图大小
rois = torch.tensor([
    [0, 100, 100, 200, 200],
    [1, 50, 50, 150, 150]
])

roi_pool = ManualRoIPooling(out_size=7)
output = roi_pool(features, rois)
print(f"输出形状: {output.shape}")  # [2, 512, 7, 7]
```

### 8.2 Bounding Box 回归

```python
import torch

def bbox_transform(boxes, deltas):
    """Bounding Box 变换
    
    boxes: [N, 4] (x1, y1, x2, y2)
    deltas: [N, 4] (dx, dy, dw, dh)
    """
    # 转换为中心-宽高形式
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    
    # 应用偏移
    pred_ctr_x = ctr_x + deltas[:, 0] * widths
    pred_ctr_y = ctr_y + deltas[:, 1] * heights
    pred_w = torch.exp(deltas[:, 2]) * widths
    pred_h = torch.exp(deltas[:, 3]) * heights
    
    # 转换回左上角-右下角形式
    pred_boxes = torch.zeros_like(boxes)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    
    return pred_boxes

def bbox_transform_inv(boxes, gt_boxes):
    """Bounding Box 逆变换
    
    计算从 boxes 到 gt_boxes 的偏移
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights
    
    # 计算偏移
    dx = (gt_ctr_x - ctr_x) / widths
    dy = (gt_ctr_y - ctr_y) / heights
    dw = torch.log(gt_widths / widths)
    dh = torch.log(gt_heights / heights)
    
    return torch.stack([dx, dy, dw, dh], dim=1)

# 验证
boxes = torch.tensor([[10, 10, 50, 50]])
gt_boxes = torch.tensor([[15, 15, 60, 60]])

deltas = bbox_transform_inv(boxes, gt_boxes)
pred_boxes = bbox_transform(boxes, deltas)

print(f"原始框: {boxes}")
print(f"GT框: {gt_boxes}")
print(f"预测框: {pred_boxes}")
```

---

## 9. 可视化与结果理解

### 9.1 检测结果可视化

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

def visualize_detections(image_path, boxes, labels, scores, class_names):
    """可视化检测结果"""
    
    # 加载图像
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # 绘制每个检测框
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(
            x1, y1 - 5,
            f'{class_names[label]}: {score:.2f}',
            fontsize=10, color='r',
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('detections.png', dpi=150)
    plt.show()

# 示例
boxes = np.array([[100, 100, 200, 200], [300, 150, 400, 300]])
labels = [1, 3]  # 类别索引
scores = [0.95, 0.87]
class_names = {1: 'person', 3: 'car'}

visualize_detections('test.jpg', boxes, labels, scores, class_names)
```

### 9.2 特征图可视化

```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def visualize_features(model, image_path):
    """可视化卷积特征"""
    
    # 加载图像
    from torchvision import transforms
    from PIL import Image
    
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # 提取特征
    features = model.extractor.features(img_tensor)
    
    # 可视化前 16 个通道
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        feature = features[0, i].detach().numpy()
        ax.imshow(feature, cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Channel {i}')
    
    plt.tight_layout()
    plt.savefig('features.png', dpi=150)
    plt.show()

visualize_features(FastRCNN(num_classes=21), 'test.jpg')
```

---

## 10. 模型评估

### 10.1 mAP 计算

```python
import numpy as np

def compute_ap(recalls, precisions):
    """计算 AP"""
    # 添加首尾点
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # 降序排列
    i = np.argsort(recalls)[::-1]
    recalls = recalls[i]
    precisions = precisions[i]
    
    # 计算包络
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 计算 AP
    ap = 0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    
    return ap

def compute_map(predictions, ground_truths, num_classes):
    """计算 mAP"""
    
    aps = []
    for c in range(1, num_classes):
        # 收集该类别的预测和 GT
        preds_c = []
        gts_c = []
        
        for pred in predictions:
            preds_c.extend(pred[c])
        
        # 按置信度排序
        preds_c = sorted(preds_c, key=lambda x: x[2], reverse=True)
        
        # 计算 TP/FP
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))
        
        for i, pred in enumerate(preds_c):
            if pred in ground_truths[c]:
                tp[i] = 1
            else:
                fp[i] = 1
        
        # 计算累计
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths[c])
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        aps.append(compute_ap(recalls, precisions))
    
    return np.mean(aps)

print("评估准备好了")
```

---

## 11. 常见问题与易错点

### 11.1 特征图坐标

**问题**：RoI 坐标是原图坐标还是特征图坐标？

**解决**：原图坐标，需要除以下采样率（如 16）得到特征图坐标。

### 11.2 背景类别

**问题**：背景类别的回归损失是否计算？

**解决**：不计算，$\mathbb{1}\{p^* > 0\}$ 表示只对非背景计算。

### 11.3 训练不稳定

**问题**：训练过程中loss 震荡？

**解决**：
- 减小学习率
- 使用预训练模型
- 调整 batch size

---

## 12. 学习总结

### 12.1 核心要点

1. **RoI Pooling**：将不同尺寸 RoI 映射为固定维度
2. **多任务损失**：同时优化分类和回归
3. **端到端训练**：整个网络一起优化
4. **特征共享**：减少计算量

### 12.2 与后续算法关系

- **Faster R-CNN**：用 RPN 替代 Selective Search
- **YOLO**：anchor-free 的单阶段检测
- **SSD**：多尺度特征融合

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：计算 RoI（100, 100, 200, 200）在 16x 下采样特征图上的坐标。

**答案**：
- $x_1 = 100 / 16 = 6.25$
- $y_1 = 100 / 16 = 6.25$
- $x_2 = 200 / 16 = 12.5$
- $y_2 = 200 / 16 = 12.5$

### 13.2 思考题

**思考题**：为什么 Fast R-CNN 的精度比 R-CNN 高？

**答案**：
1. 端到端训练可以联合优化特征提取和分类器
2. 多任务损失提供了更强的监督信号
3. 特征共享避免了重复计算

---

## 14. 学习路径建议

### 14.1 基础（1-2 天）

1. 理解 R-CNN 结构
2. 理解 Selective Search
3. 理解 RoI Pooling

### 14.2 进阶（2-3 天）

1. 实现 Fast R-CNN
2. 训练模型
3. 可视化结果

### 14.3 应用（3-5 天）

1. 使用预训练模型
2. 调参优化
3. 部署推理

### 14.4 推荐资源

- **论文**：Fast R-CNN
- **代码**：torchvision
- **数据集**：PASCAL VOC, COCO

---

*Fast R-CNN 是目标检测领域的重要里程碑，它的端到端训练思想深深影响了后续的 Faster R-CNN、Mask R-CNN 等算法。*