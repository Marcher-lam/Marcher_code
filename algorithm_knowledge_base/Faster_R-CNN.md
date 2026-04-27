# Faster R-CNN 学习文档

## 1. 算法基础认知

Faster R-CNN发表于2015年，由Shaoqing Ren等人提出，是R-CNN系列的第三代作品。它首次实现了真正端到端的目标检测，将候选区域生成也纳入深度神经网络，真正实现了"一体化"检测框架。

**核心创新**：用区域提议网络（Region Proposal Network, RPN）替代了耗时的Selective Search算法，实现了端到端训练。

**性能突破**：在GPU上实现约5fps的检测速度，PASCAL VOC 2012上mAP达到73.2%。

## 2. 核心原理

Faster R-CNN由两个核心模块组成：

**模块1：区域提议网络（RPN）**
- 输入：主干网络提取的特征图
- 输出：候选边界框（proposals）和前景/背景分数
- 核心概念：锚框（Anchor）
- 每张特征图上的位置生成k个不同尺度和长宽比的锚框

**模块2：检测网络（Fast R-CNN）**
- ROI Pooling：将不同尺寸的候选区域池化到固定尺寸
- 分类：预测目标类别
- 回归：精确调整边界框位置

**整体流程**：
1. 图像输入主干网络（如VGG16、ResNet），提取特征图
2. RPN基于特征图生成候选区域和分数
3. ROI Pooling从特征图中提取每个候选区域的特征
4. 全连接层分类和回归，输出最终检测结果
5. NMS后处理，去除重复框

## 3. 数学公式与推导

**锚框定义**：
每个特征图位置(i,j)对应原图中心点：
$$(x_c, y_c) = (i \times stride + centeroffset, j \times stride + centeroffset)$$

锚框尺度s和长宽比r：
$$size = s^2$$
$$aspect = [1/r, 1, r]$$
$$width = size \times \sqrt{aspect}$$
$$height = size / \sqrt{aspect}$$

**RPN分类损失**：
对于每个锚框，分类标签为：
$$l_i = \begin{cases} 1 & \text{与某GT的IoU}>0.7 \\ 0 & \text{与所有GT的IoU}<0.3 \end{cases}$$

分类损失（二分类交叉熵）：
$$L_{cls} = -\frac{1}{N_{cls}}\sum_i [l_i \log(p_i) + (1-l_i)\log(1-p_i)]$$

**RPN回归损失**：
回归目标（针对正样本锚框）：
$$t_x = (x - x_a) / w_a, \quad t_y = (y - y_a) / h_a$$
$$t_w = \log(w / w_a), \quad t_h = \log(h / h_a)$$

Smooth L1损失：
$$L_{reg}(t, t^*) = \sum_{i \in \{x,y,w,h\}} smooth_{L1}(t_i - t_i^*)$$

其中：
$$smooth_{L1}(x) = \begin{cases} 0.5x^2 & |x|<1 \\ |x|-0.5 & |x|\geq 1 \end{cases}$$

**总损失**：
$$L = L_{cls} + \lambda \cdot L_{reg}$$

其中λ通常设为10，平衡分类和回归损失。

**ROI Pooling**：
将候选区域划分为HxW网格：
$$h = H / h_{region}, \quad w = W / w_{region}$$
每个网格内做最大池化，得到固定尺寸的特征。

## 4. 训练过程讲解

Faster R-CNN采用"四步交替训练"策略：

**阶段1：训练RPN**
- 使用ImageNet预训练的CNN作为主干网络
- 初始化RPN层，随机权重
- 固定CNN参数，训练RPN
- 生成候选区域用于下一阶段

**阶段2：训练Fast R-CNN检测网络**
- 使用阶段1的RPN生成的候选区域
- 初始化Fast R-CNN检测头
- 固定CNN和RPN，训练检测网络

**阶段3：精修RPN**
- 使用阶段2的检测网络参数初始化RPN
- 固定检测网络，精修RPN

**阶段4：精修检测网络**
- 使用阶段3的RPN生成的候选区域
- 固定RPN和CNN，精修检测网络

**训练细节**：
- 批量大小：每张图像256个锚框，正负比例1:1
- 学习率：0.001（30k迭代），0.0001（10k迭代）
- 权重衰减：0.0005
- Momentum：0.9

## 5. 应用场景

Faster R-CNN广泛应用于：

- **通用目标检测**：PASCAL VOC、COCO比赛的基础算法
- **自动驾驶**：车辆、行人、交通标志检测
- **人脸检测**：人脸框定和识别
- **医学影像**：CT、MRI中的病变检测
- **无人机航拍**：大规模场景目标检测
- **视频分析**：实时目标跟踪和检测

## 6. 优缺点分析

**优点**：
- **真正端到端**：从图像到检测结果，全程可微分可训练
- **速度快**：RPN取代Selective Search，GPU加速显著
- **精度高**：两阶段设计，检测精度优于单阶段方法
- **统一框架**：分类和回归统一在一个网络中
- **可迁移**：骨干网络可替换为不同架构

**缺点**：
- **训练复杂**：四步交替训练耗时较长
- **推理速度**：仍受限于两阶段设计，不如YOLO快
- **小目标检测**：小目标容易漏检
- **锚框设计**：需要人工设计锚框参数
- **内存占用**：特征图缓存需要大量显存

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import feature_extraction

class AnchorGenerator:
    def __init__(self, base_size, scales, ratios):
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios
    
    def generate_anchors(self):
        py = self.base_size / 2.
        px = self.base_size / 2.
        anchors = []
        for ratio in self.ratios:
            for scale in self.scales:
                h = self.base_size * scale / (ratio ** 0.5)
                w = self.base_size * scale * (ratio ** 0.5)
                anchors.append([px - w/2, py - h/2, px + w/2, py + h/2])
        return torch.tensor(anchors)

class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.reg_layer = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        rpn_cls = self.cls_layer(x)
        rpn_reg = self.reg_layer(x)
        return rpn_cls, rpn_reg

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, num_anchors, feat_stride=16):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = AnchorGenerator(base_size=16, scales=[8,16,32], ratios=[0.5,1,2])
        self.rpn_head = RPNHead(in_channels, num_anchors)
        self.feat_stride = feat_stride
        self.num_anchors = num_anchors
    
    def forward(self, features, image_size):
        rpn_cls, rpn_reg = self.rpn_head(features)
        anchors = self._generate_anchors(features.shape[-2:])
        return rpn_cls, rpn_reg, anchors
    
    def _generate_anchors(self, feature_size):
        import numpy as np
        base_anchors = self.anchor_generator.generate_anchors()
        feature_stride = self.feat_stride
        shifts_x = torch.arange(feature_size[1]) * feature_stride
        shifts_y = torch.arange(feature_size[0]) * feature_stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.flatten()
        shift_y = shift_y.flatten()
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
        anchors = base_anchors.unsqueeze(0) + shifts.unsqueeze(1)
        return anchors.flatten(0, 1)

class RoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
    
    def forward(self, features, rois):
        batch_size = features.size(0)
        num_rois = rois.size(0)
        
        roi_indices = rois[:, 0].long()
        roi_bboxes = rois[:, 1:]
        
        output_h, output_w = self.output_size
        
        sub_h = roi_bboxes[:, 3] - roi_bboxes[:, 1] / output_h
        sub_w = roi_bboxes[:, 2] - roi_bboxes[:, 0] / output_w
        
        x = torch.arange(output_w, device=features.device).float()
        y = torch.arange(output_h, device=features.device).float()
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        x = (x.float() * sub_w.unsqueeze(1) + roi_bboxes[:, 0].unsqueeze(1)).flatten()
        y = (y.float() * sub_h.unsqueeze(1) + roi_bboxes[:, 1].unsqueeze(1)).flatten()
        
        x1 = x.floor().long()
        y1 = y.floor().long()
        x2 = (x + 1).clamp(max=features.shape[3]-1).long()
        y2 = (y + 1).clamp(max=features.shape[2]-1).long()
        
        indices = roi_indices.repeat_interleave(output_h * output_w)
        
        output = torch.zeros(num_rois, output_h, output_w, features.size(1), device=features.device)
        
        for b in range(batch_size):
            batch_mask = indices == b
            if batch_mask.sum() == 0:
                continue
            feat_batch = features[b]
            
            x1_b, y1_b = x1[batch_mask], y1[batch_mask]
            x2_b, y2_b = x2[batch_mask], y2[batch_mask]
            
            v11 = feat_batch[:, y1_b, x1_b]
            v12 = feat_batch[:, y1_b, x2_b]
            v21 = feat_batch[:, y2_b, x1_b]
            v22 = feat_batch[:, y2_b, x2_b]
            
            w11 = (x2_b.float() - x) * (y2_b.float() - y)
            w12 = (x - x1_b.float()) * (y2_b.float() - y)
            w21 = (x2_b.float() - x) * (y - y1_b.float())
            w22 = (x - x1_b.float()) * (y - y1_b.float())
            
            interpolated = (v11 * w11 + v12 * w12 + v21 * w21 + v22 * w22)
            interpolated = interpolated.view(output_h, output_w, -1, features.size(1))
            output[batch_mask] = interpolated.permate(2, 0, 1, 3)
        
        return output

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super(FasterRCNN, self).__init__()
        if backbone == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
            feature_channels = 1024
        else:
            self.backbone = torchvision.models.vgg16(pretrained=True)
            feature_channels = 512
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        self.rpn = RegionProposalNetwork(in_channels=feature_channels, num_anchors=9)
        self.roi_pool = RoIAlign(output_size=(7, 7), spatial_scale=1/16)
        
        self.fc1 = nn.Linear(feature_channels * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)
        
        self.num_classes = num_classes
    
    def forward(self, images, targets=None):
        features = self.backbone(images)
        
        rpn_cls, rpn_reg, anchors = self.rpn(features, (images.shape[2], images.shape[3]))
        
        proposals = self._generate_proposals(rpn_cls, rpn_reg, anchors, images.shape)
        
        roi_pooled = self.roi_pool(features, proposals)
        roi_pooled = roi_pooled.view(roi_pooled.size(0), -1)
        
        fc1_out = F.relu(self.fc1(roi_pooled))
        fc2_out = F.relu(self.fc2(fc1_out))
        
        cls_scores = self.cls_score(fc2_out)
        bbox_pred = self.bbox_pred(fc2_out)
        
        if self.training and targets is not None:
            loss = self._compute_loss(cls_scores, bbox_pred, targets, proposals)
            return loss
        else:
            return cls_scores, bbox_pred, proposals
    
    def _generate_proposals(self, rpn_cls, rpn_reg, anchors, image_shape):
        rpn_cls = rpn_cls.permute(0, 2, 3, 1).contiguous()
        rpn_cls = rpn_cls.view(rpn_cls.size(0), -1)
        rpn_cls = F.softmax(rpn_cls, dim=-1)[:, 1]
        
        rpn_reg = rpn_reg.permute(0, 2, 3, 1).contiguous()
        rpn_reg = rpn_reg.view(rpn_reg.size(0), -1, 4)
        
        batch_size = rpn_cls.shape[0]
        proposals = []
        for i in range(batch_size):
            scores = rpn_cls[i]
            deltas = rpn_reg[i]
            
            top_indices = torch.topk(scores, min(2000, scores.size(0))[1]
            
            deltas = deltas[top_indices]
            anchors_i = anchors[top_indices]
            
            proposals_i = self._apply_deltas(anchors_i, deltas)
            proposals_i = self._clip_boxes(proposals_i, image_shape[2:])
            proposals.append(proposals_i)
        
        return torch.stack(proposals)
    
    def _apply_deltas(self, boxes, deltas):
        x = (boxes[:, 0] + boxes[:, 2]) / 2
        y = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]
        
        x_new = x + dx * w
        y_new = y + dy * h
        w_new = w * torch.exp(dw)
        h_new = h * torch.exp(dh)
        
        return torch.stack([
            x_new - w_new / 2,
            y_new - h_new / 2,
            x_new + w_new / 2,
            y_new + h_new / 2
        ], dim=1)
    
    def _clip_boxes(self, boxes, image_size):
        boxes[:, 0] = boxes[:, 0].clamp(min=0, max=image_size[1])
        boxes[:, 1] = boxes[:, 1].clamp(min=0, max=image_size[0])
        boxes[:, 2] = boxes[:, 2].clamp(min=0, max=image_size[1])
        boxes[:, 3] = boxes[:, 3].clamp(min=0, max=image_size[0])
        return boxes
    
    def _compute_loss(self, cls_scores, bbox_pred, targets, proposals):
        classification_loss = F.cross_entropy(cls_scores, targets['labels'])
        
        bbox_pred = bbox_pred.view(-1, self.num_classes, 4)
        bbox_pred = bbox_pred[torch.arange(len(targets['labels'])), targets['labels']]
        
        regression_loss = F.smooth_l1_loss(
            bbox_pred, 
            targets['boxes'],
            reduction='sum'
        ) / len(targets['labels'])
        
        total_loss = classification_loss + 0.01 * regression_loss
        return total_loss

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        
        for key in targets:
            if isinstance(targets[key], torch.Tensor):
                targets[key] = targets[key].to(device)
        
        optimizer.zero_grad()
        loss = model(images, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FasterRCNN(num_classes=21).to(device)
print(f"Faster R-CNN model initialized on {device}")
```

## 8. 手工代码实现（PyTorch Tensor）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ManualRPN(nn.Module):
    def __init__(self, in_channels=512, num_anchors=9):
        super(ManualRPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(512, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feature_map):
        conv_out = F.relu(self.conv(feature_map))
        cls_scores = self.cls_logits(conv_out)
        bbox_deltas = self.bbox_pred(conv_out)
        return cls_scores, bbox_deltas

class ManualROIAlign(nn.Module):
    def __init__(self, output_size=(7, 7), spatial_scale=1/16):
        super(ManualROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
    
    def forward(self, feature_map, rois):
        batch_idx = rois[:, 0].long()
        coords = rois[:, 1:]
        
        scaled_coords = coords * self.spatial_scale
        
        output_h, output_w = self.output_size
        
        h_bins = torch.linspace(0, scaled_coords[:, 3] - scaled_coords[:, 1], output_h + 1)
        w_bins = torch.linspace(0, scaled_coords[:, 2] - scaled_coords[:, 0], output_w + 1)
        
        outputs = []
        
        for b in range(feature_map.size(0)):
            mask = batch_idx == b
            if mask.sum() == 0:
                continue
            
            rois_batch = scaled_coords[mask]
            feat_batch = feature_map[b]
            
            roi_h = rois_batch[:, 3] - rois_batch[:, 1]
            roi_w = rois_batch[:, 2] - rois_batch[:, 0]
            
            cell_h = roi_h / output_h
            cell_w = roi_w / output_w
            
            interpolated = torch.zeros(mask.sum(), output_h, output_w, feat_batch.size(0), device=feature_map.device)
            
            for row in range(output_h):
                for col in range(output_w):
                    y_center = rois_batch[:, 1] + (row + 0.5) * cell_h
                    x_center = rois_batch[:, 0] + (col + 0.5) * cell_w
                    
                    y_low = y_center.floor().long().clamp(0, feat_batch.size(1) - 2)
                    x_low = x_center.floor().long().clamp(0, feat_batch.size(2) - 2)
                    y_high = (y_low + 1).clamp(0, feat_batch.size(1) - 1)
                    x_high = (x_low + 1).clamp(0, feat_batch.size(2) - 1)
                    
                    v_tl = feat_batch[:, y_low, x_low]
                    v_tr = feat_batch[:, y_low, x_high]
                    v_bl = feat_batch[:, y_high, x_low]
                    v_br = feat_batch[:, y_high, x_high]
                    
                    w_tl = (x_high.float() - x_center) * (y_high.float() - y_center)
                    w_tr = (x_center - x_low.float()) * (y_high.float() - y_center)
                    w_bl = (x_high.float() - x_center) * (y_center - y_low.float())
                    w_br = (x_center - x_low.float()) * (y_center - y_low.float())
                    
                    interpolated[:, row, col] = (v_tl * w_tl + v_tr * w_tr + 
                                              v_bl * w_bl + v_br * w_br)
            
            outputs.append(interpolated)
        
        if len(outputs) == 0:
            return torch.zeros(0, output_h, output_w, feature_map.size(1))
        
        return torch.cat(outputs, dim=0)

class ManualDetectorHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ManualDetectorHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_scores = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_scores, bbox_pred

def generate_anchors(feature_shape, strides, sizes, ratios):
    anchors_list = []
    for feat_h, feat_w, stride in zip(feature_shape, strides):
        for i in range(feat_h):
            for j in range(feat_w):
                center_y = (i + 0.5) * stride
                center_x = (j + 0.5) * stride
                
                for size in sizes:
                    for ratio in ratios:
                        h = size * (ratio ** 0.5)
                        w = size / (ratio ** 0.5)
                        
                        y1 = center_y - h / 2
                        x1 = center_x - w / 2
                        y2 = center_y + h / 2
                        x2 = center_x + w / 2
                        
                        anchors_list.append([x1, y1, x2, y2])
    
    return torch.tensor(anchors_list)

def compute_iou(boxes1, boxes2):
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area1[:, None] + area2 - inter
    
    return inter / (union + 1e-6)

def nms(boxes, scores, iou_threshold=0.7):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return torch.stack(keep)

def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 
                       0.5 * diff ** 2 / beta,
                       diff - 0.5 * beta)
    return loss.mean()

rpn = ManualRPN(in_channels=512, num_anchors=9)
roi_align = ManualROIAlign()
detector_head = ManualDetectorHead(in_features=512*7*7, num_classes=21)
print("Manual Faster R-CNN components created")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def visualize_anchors(image, anchors, scores=None, top_k=100):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    if scores is not None:
        top_indices = torch.topk(scores, min(top_k, len(scores))[1]
        anchors = anchors[top_indices]
    
    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=0.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title(f'RPN Anchors (showing {len(anchors)} anchors)')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('faster_rcnn_anchors.png', dpi=150)
    plt.close()

def visualize_rpn_scores(feature_map, stride=16):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    feature_np = feature_map[0].cpu().detach().numpy()
    
    for i, ax in enumerate(axes.flat):
        if i < feature_np.shape[0]:
            ax.imshow(feature_np[i], cmap='viridis')
            ax.set_title(f'Channel {i}')
        ax.axis('off')
    
    plt.suptitle('RPN Feature Map Activations')
    plt.tight_layout()
    plt.savefig('faster_rcnn_features.png', dpi=150)
    plt.close()

def visualize_detection_results(image, boxes, labels, scores, 
                                  class_names, threshold=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = colors[label % len(colors)]
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=2, edgecolor=color,
                                 facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        
        text = f'{class_names[label]}: {score:.2f}'
        ax.text(x1, y1 - 5, text, fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax.set_title('Faster R-CNN Detection Results')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('faster_rcnn_detections.png', dpi=150)
    plt.close()

def visualize_roi_pooling(feature_map, rois, pooled_size=(7, 7)):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, ax in enumerate(axes.flat):
        if i < min(10, len(roi_pooled)):
            pooled = roi_pooled[i].cpu().detach().numpy()
            pooled_vis = pooled.mean(axis=2)
            ax.imshow(pooled_vis, cmap='viridis')
            ax.set_title(f'ROI {i}')
        ax.axis('off')
    
    plt.suptitle('ROI Pooled Features')
    plt.tight_layout()
    plt.savefig('faster_rcnn_roi_pooling.png', dpi=150)
    plt.close()

def plot_training_curves():
    epochs = list(range(1, 51))
    rpn_cls_loss = [np.random.random() * 0.5 for _ in epochs]
    rpn_reg_loss = [np.random.random() * 0.2 for _ in epochs]
    total_loss = [r + c for r, c in zip(rpn_cls_loss, rpn_reg_loss)]
    val_map = [0.3 + 0.4 * (1 - np.exp(-e/15)) for e in epochs]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, rpn_cls_loss, label='RPN Classification Loss')
    axes[0].plot(epochs, rpn_reg_loss, label='RPN Regression Loss')
    axes[0].plot(epochs, total_loss, label='Total Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs, val_map, label='Validation mAP', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    axes[1].set_title('Validation mAP')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('faster_rcnn_training.png', dpi=150)
    plt.close()

def visualize_proposal_pipeline(image, proposals, scores, final_detections):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    for prop in proposals[:50]:
        x1, y1, x2, y2 = prop
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=0.5, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_title('Stage 1: RPN Proposals')
    axes[0].axis('off')
    
    axes[1].imshow(image)
    for prop in proposals[:20]:
        x1, y1, x2, y2 = prop
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=1, edgecolor='g', facecolor='none')
        axes[1].add_patch(rect)
    axes[1].set_title('Stage 2: After NMS')
    axes[1].axis('off')
    
    axes[2].imshow(image)
    for det in final_detections:
        x1, y1, x2, y2, label, score = det
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='b', facecolor='none')
        axes[2].add_patch(rect)
    axes[2].set_title('Stage 3: Final Detections')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('faster_rcnn_pipeline.png', dpi=150)
    plt.close()

print("Visualization functions defined")
```

## 10. 模型评估

```python
import numpy as np
import torch

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def compute_ap(precisions, recalls):
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def evaluate_detector(predictions, ground_truths, iou_threshold=0.5):
    aps = []
    
    for class_id in range(num_classes):
        pred_mask = predictions[:, 5] == class_id
        pred_class = predictions[pred_mask]
        
        gt_mask = ground_truths[:, 4] == class_id
        gt_class = ground_truths[gt_mask]
        
        if len(gt_class) == 0:
            continue
        
        pred_class = pred_class[pred_class[:, 4].argsort()[::-1]]
        
        tp = np.zeros(len(pred_class))
        fp = np.zeros(len(pred_class))
        
        gt_matched = np.zeros(len(gt_class))
        
        for i, pred in enumerate(pred_class):
            max_iou = 0
            max_idx = -1
            
            for j, gt in enumerate(gt_class):
                if gt_matched[j]:
                    continue
                
                iou = calculate_iou(pred[:4], gt[:4])
                
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= iou_threshold:
                tp[i] = 1
                gt_matched[max_idx] = 1
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recalls = tp_cumsum / len(gt_class)
        
        ap = compute_ap(precisions, recalls)
        aps.append(ap)
    
    return np.mean(aps), aps

def compute_detection_metrics(pred_boxes, pred_scores, pred_labels,
                                 gt_boxes, gt_labels, num_classes):
    metrics = {}
    
    for class_id in range(1, num_classes):
        pred_mask = pred_labels == class_id
        gt_mask = gt_labels == class_id
        
        if gt_mask.sum() == 0:
            metrics[f'class_{class_id}_recall'] = 0.0
            continue
        
        pred_c = pred_scores[pred_mask]
        boxes_c = pred_boxes[pred_mask]
        
        sorted_idx = np.argsort(-pred_c)
        
        tp = np.zeros(len(pred_c))
        fp = np.zeros(len(pred_c))
        gt_matched = np.zeros(gt_mask.sum())
        
        for i, idx in enumerate(sorted_idx):
            box = boxes_c[idx]
            max_iou = 0
            max_j = -1
            
            for j, gt_box in enumerate(gt_boxes[gt_mask]):
                iou = calculate_iou(box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_j = j
            
            if max_iou >= 0.5 and gt_matched[max_j] == 0:
                tp[i] = 1
                gt_matched[max_j] = 1
            else:
                fp[i] = 1
        
        tp_sum = np.cumsum(tp)
        fp_sum = np.cumsum(fp)
        
        precision = tp_sum / (tp_sum + fp_sum + 1e-10)
        recall = tp_sum / gt_mask.sum()
        
        metrics[f'class_{class_id}_precision'] = precision[-1] if len(precision) > 0 else 0
        metrics[f'class_{class_id}_recall'] = recall[-1] if len(recall) > 0 else 0
    
    return metrics

print("Evaluation metrics defined")
```

## 11. 常见问题与易错点

**问题1：锚框设计不当**
锚框尺寸和比例需要适配数据集目标尺寸分布。不合适的锚框会导致正样本不足，训练不稳定。

**问题2：RPN与检测网络训练不平衡**
两类损失量级差异大（约1:10），需要调参λ平衡，否则RPN或检测网络一方训练不足。

**问题3：训练收敛慢**
学习率设置不当会导致训练不收敛，建议使用预训练模型初始化，并使用分段学习率。

**问题4：候选区域质量差**
如果RPN生成的候选区域覆盖不足，会影响最终检测精度。需要根据目标分布调整锚框。

**问题5：NMS阈值选择**
NMS阈值过高会导致重复检测，阈值过低会误删正确检测。通常在0.5-0.7之间调整。

**问题6：特征图与原图对齐**
特征图的步长需要与锚框生成对齐，否则会有位置偏移误差。

**问题7：batch内图像尺寸不一**
RPN可以处理不同尺寸图像，但ROI Pooling需要统一管理batch内的ROI。

**问题8：GPU显存不足**
Faster R-CNN显存占用大，需要合理设置batch size，或使用梯度累积。

## 12. 学习总结

Faster R-CNN的核心贡献：

1. **RPN首创**：用神经网络学习候选区域，替代传统Selective Search
2. **端到端训练**：整个检测流程可反向传播，梯度流畅
3. **共享特征**：RPN和检测网络共享CNN特征，计算高效
4. **锚框机制**：多尺度多比例锚框，无需图像金字塔
5. **高精度**：两阶段设计保证了检测精度优势

后续演进：
- Mask R-CNN：增加实例分割分支
- Cascade R-CNN：多级检测器级联
- RetinaNet：引入Focal Loss改进单阶段检测

## 13. 练习题与思考题与思考题（含答案）

**练习题1**：
为什么Faster R-CNN使用9个锚框而不是3个或27个？

**答案**：
3个太少，无法覆盖多尺度目标；27���过多，计算量大且正样本分散。9个（3尺度×3比例）是精度和速度的平衡点。

**练习题2**：
RPN的分类损失为什么要用二分类（前景/背景）而不是直接预测K类？

**答案**：
RPN只需要筛选可能包含目标的区域，不需要识别具体类别。类别预测由后续检测网络完成，保持了模块化。

**练习题3**：
为什么ROI Pooling改为ROI Align？

**答案**：
ROI Pooling的两次量化会造成定位误差。ROI Align使用双线性插值精确采样，保持了空间对齐。

**思考题**：
Faster R-CNN与YOLO的本质区别是什么？

**答案**：
Faster R-CNN是两阶段检测器，先生成候选区域再分类；YOLO是单阶段检测器，直接从特征图回归检测结果。两阶段精度高但速度慢，单阶段速度快但精度略低。

## 14. 学习路径建议建议

**入门阶段（1周）**：
- 理解目标检测基本概念
- 学习卷积神经网络基础
- 对比R-CNN系列演进

**进阶阶段（2周）**：
- 理解RPN工作原理
- 学习锚框设计方法
- 实现Faster R-CNN训练流程

**实战阶段（2周）**：
- 在自定义数据集上训练
- 调参与优化技巧
- 模型部署

**推荐资源**：
- 原始论文：Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- 代码：detectron2、mmdetection
- 课程：斯坦福CS231n