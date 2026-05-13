# R-CNN 学习文档

## 1. 算法基础认知

R-CNN（Regions with Convolutional Neural Networks）发表于2014年，由Ross Girshick等人提出，是深度学习时代目标检测的开创性工作。它首次将深度卷积神经网络用于目标检测，显著提升了PASCAL VOC数据集上的检测精度。

**核心问题**：如何在图像中定位和识别多种类别的目标物体？

与传统滑动窗口方法不同，R-CNN采用"候选区域+CNN特征+SVM分类"的流程，首先使用Selective Search生成约2000个候选区域，然后对每个区域提取CNN特征，最后用SVM进行分类。

## 2. 核心原理

R-CNN的工作流程包含四个关键步骤：

**步骤1：候选区域生成**
使用Selective Search算法从图像中提取约2000个可能包含目标的候选区域。Selective Search通过颜色、纹理、尺寸和空间重叠度等特征进行区域合并，生成多尺度、多形状的候选框。

**步骤2：特征提取**
对每个候选区域，通过CNN（通常使用AlexNet）提取固定长度的特征向量。由于CNN要求输入尺寸固定，候选区域被缩放（wrap）到227×227像素。

**步骤3：SVM分类**
使用支持向量机（SVM）对每个候选区域的特征进行分类。R-CNN为每个类别训练一个二分类SVM，判断候选区域是否包含该类目标。

**步骤4：边界框回归**
训练一个线性回归模型，对候选边界框进行微调，使其更准确地包围目标。回归模型预测框的位置偏移量（dx, dy, dw, dh）。

## 3. 数学公式与推导

**候选区域特征提取**：
给定输入图像I，候选区域R，CNN特征提取可表示为：
$$f_R = CNN(I_R; \theta)$$
其中I_R是候选区域R缩放后的图像，θ是CNN参数。

**SVM分类**：
对于类别c，分类器输出：
$$score_c(R) = w_c^T f_R + b_c$$
其中w_c和b_c是类别c的SVM参数。最终类别为：
$$c^* = \arg\max_c score_c(R)$$

**边界框回归**：
预测偏移量：
$$[\Delta x, \Delta y, \Delta w, \Delta h] = w_{reg}^T f_R + b_{reg}$$
更新后的边界框：
$$x' = x + \Delta x \cdot w, \quad y' = y + \Delta y \cdot h$$
$$w' = w \cdot e^{\Delta w}, \quad h' = h \cdot e^{\Delta h}$$

## 4. 训练过程讲解

R-CNN的训练是多阶段、分别进行的：

**阶段1：预训练CNN**
在大规模图像分类数据集（如ImageNet）上预训练CNN，学习图像特征表示。预训练网络作为特征提取器。

**阶段2：微调CNN**
将CNN的全连接层替换为随机初始化的层，在检测数据上进行微调。Softmax分类器输出类别数量+1（背景）。

**阶段3：训练SVM分类器**
使用正负样本训练SVM。正样本是IoU≥0.3的候选区域，负样本是其他区域。每个类别独立训练二分类SVM。

**阶段4：训练边界框回归器**
使用与正样本IoU≥0.6的候选区域，训练线性回归模型预测边界框偏移量。

## 5. 应用场景

R-CNN主要用于：

- **通用目标检测**：识别图像中的汽车、人、动物等常见目标
- **PASCAL VOC挑战赛**：2013年检测挑战赛冠军，mAP从59.4%提升至66.4%
- **场景理解**：为图像字幕生成、视觉问答提供目标检测基础
- **自动驾驶**：车辆、行人检测（后续改进版本）

## 6. 优缺点分析

**优点**：
- 首次将深度CNN用于目标检测，证明了深度特征的强大表示能力
- 与传统方法（如DPM）相比，检测精度显著提升
- 模块化设计，可灵活更换特征提取网络和分类器

**缺点**：
- 训练流程复杂，需要多个独立训练阶段
- 对每个候选区域单独提取CNN特征，速度极慢（每张图约50秒）
- 候��区域没有共享计算，大量重复计算
- 特征缩放导致信息损失
- 无法端到端训练，梯度无法反向传播

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from selective_search import selective_search

class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        self.backbone = torchvision.models.alexnet(pretrained=True)
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.num_classes = num_classes
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_bbox = nn.Linear(4096, 4)
    
    def forward(self, x):
        features = self.backbone(x)
        cls_scores = self.cls_score(features)
        bbox_deltas = self.bbox_bbox(features)
        return cls_scores, bbox_deltas

def selective_search_boxes(img, scale=500, sigma=0.9, min_size=20):
    boxes = selective_search(img, scale, sigma, min_size)
    return boxes[:2000]

def warp_image(img, bbox, size=227):
    x, y, w, h = bbox
    crop = img[y:y+h, x:x+w]
    resized = cv2.resize(crop, (size, size))
    return resized

def prepare_batch(image, boxes):
    batch = []
    for box in boxes:
        warped = warp_image(image, box)
        batch.append(warped)
    return np.stack(batch)

def extract_features(model, images, device='cuda'):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    batch_tensors = []
    for img in images:
        tensor = transform(img)
        batch_tensors.append(tensor)
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        features = model.backbone(batch)
    return features

def nms(boxes, scores, threshold=0.3):
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        iou = compute_iou(boxes[i], boxes[order[1:]])
        mask = iou <= threshold
        order = order[1:][mask]
    return keep

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[0]+box[2], boxes[:, 0]+boxes[:, 2])
    y2 = np.minimum(box[1]+box[3], boxes[:, 1]+boxes[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    union = box[2]*box[3] + boxes[:, 2]*boxes[:, 3] - inter
    return inter / (union + 1e-6)

def predict(model, image, device='cuda'):
    boxes = selective_search_boxes(image)
    batch = prepare_batch(image, boxes)
    features = extract_features(model, batch, device)
    scores = model.cls_score(features)
    deltas = model.bbox_bbox(features)
    scores_np = scores.cpu().numpy()
    for i in range(len(boxes)):
        bbox_deltas = deltas[i].cpu().numpy()
        box = boxes[i]
        dx, dy, dw, dh = bbox_deltas
        w, h = box[2], box[3]
        new_box = [
            box[0] + dx * w,
            box[1] + dy * h,
            box[2] * np.exp(dw),
            box[3] * np.exp(dh)
        ]
        boxes[i] = new_box
    final_boxes = []
    for c in range(1, model.num_classes):
        class_scores = scores_np[:, c]
        keep = nms(boxes, class_scores)
        for idx in keep:
            final_boxes.append((boxes[idx], c, class_scores[idx]))
    return final_boxes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RCNN(num_classes=21).to(device)
print("R-CNN model loaded successfully")
```

## 8. 手工代码实现（PyTorch Tensor）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 21)
        self.fc4 = nn.Linear(4096, 4)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=3, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_scores = self.fc3(x)
        bbox_deltas = self.fc4(x)
        return cls_scores, bbox_deltas

class SVMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SVMClassifier, self).__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
        self.bias = nn.Parameter(torch.randn(num_classes))
    
    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias

class BBoxRegressor(nn.Module):
    def __init__(self, input_dim):
        super(BBoxRegressor, self).__init__()
        self.weight = nn.Parameter(torch.randn(4, input_dim))
        self.bias = nn.Parameter(torch.randn(4))
    
    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias

def compute_classification_loss(scores, targets, num_classes):
    return F.cross_entropy(scores, targets)

def compute_bbox_loss(deltas, targets, bbox_weights):
    loss = F.smooth_l1_loss(deltas, targets, reduction='none')
    loss = loss * bbox_weights.unsqueeze(1)
    return loss.mean()

def apply_bbox_deltas(boxes, deltas):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    x_new = x + dx * w
    y_new = y + dy * h
    w_new = w * torch.exp(dw)
    h_new = h * torch.exp(dh)
    return torch.stack([x_new, y_new, w_new, h_new], dim=1)

def compute_iou_tensor(box1, box2):
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 0] + box1[:, 2], box2[:, 0] + box2[:, 2])
    y2 = torch.min(box1[:, 1] + box1[:, 3], box2[:, 1] + box2[:, 3])
    inter = torch.max(torch.zeros_like(x1), x2 - x1) * torch.max(torch.zeros_like(y1), y2 - y1)
    area1 = box1[:, 2] * box1[:, 3]
    area2 = box2[:, 2] * box2[:, 3]
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

cnn = SimpleCNN()
svm = SVMClassifier(input_dim=4096, num_classes=21)
bbox_reg = BBoxRegressor(input_dim=4096)
print("Custom R-CNN components initialized")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_detections(image, boxes, scores, labels, colors, class_names, threshold=0.5):
    img = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x, y, w, h = [int(v) for v in box]
        color = colors[label % len(colors)]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        text = f"{class_names[label]}: {score:.2f}"
        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def plot_detection_pipeline():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('R-CNN Detection Pipeline', fontsize=14)
    axes[0, 0].set_title('1. Original Image')
    axes[0, 1].set_title('2. Selective Search Regions')
    axes[0, 2].set_title('3. CNN Features')
    axes[1, 0].set_title('4. SVM Scores')
    axes[1, 1].set_title('5. Bounding Box Regression')
    axes[1, 2].set_title('6. Final Detection (NMS)')
    plt.tight_layout()
    plt.savefig('rcnn_pipeline.png', dpi=150)
    plt.close()

def plot_pr_curves(precisions, recalls, class_names):
    plt.figure(figsize=(10, 8))
    for i, (prec, rec, name) in enumerate(zip(precisions, recalls, class_names)):
        plt.plot(rec, prec, label=name, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('rcnn_pr_curves.png', dpi=150)
    plt.close()

def visualize_feature_maps(features, num_maps=16):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            feature_map = features[0, i].cpu().detach().numpy()
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Feature Map {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('rcnn_feature_maps.png', dpi=150)
    plt.close()

def plot_training_curves(losses_cls, losses_bbox, mAPs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses_cls, label='Classification Loss')
    axes[0].plot(losses_bbox, label='BBox Regression Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Losses')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(mAPs)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    axes[1].set_title('Validation mAP')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('rcnn_training.png', dpi=150)
    plt.close()

print("Visualization functions defined")
```

## 10. 模型评估

**主要评估指标**：

- **mAP（mean Average Precision）**：所有类别AP的平均值，是目标检测最重要的指标
- **Precision**：预测为正的样本中真正为正的比例
- **Recall**：所有正样本中被正确预测的比例
- **IoU（Intersection over Union）**：预测框与真实框的交并比

**计算方法**：
```python
def compute_ap(precision, recall):
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices+1] - recall[indices]) * precision[indices+1])
    return ap

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return inter / (union + 1e-6)

def evaluate_detections(pred_boxes, pred_scores, pred_labels, 
                        gt_boxes, gt_labels, iou_threshold=0.5):
    num_classes = max(max(pred_labels), max(gt_labels)) + 1
    aps = []
    for c in range(1, num_classes):
        pred_mask = pred_labels == c
        pred_c = pred_scores[pred_mask]
        boxes_c = [tuple(b) for b in pred_boxes[pred_mask]]
        gt_mask = gt_labels == c
        gt_c = gt_boxes[gt_mask]
        if len(gt_c) == 0:
            continue
        sorted_idx = np.argsort(-pred_c)
        tp = np.zeros(len(pred_c))
        fp = np.zeros(len(pred_c))
        gt_matched = np.zeros(len(gt_c))
        for i, idx in enumerate(sorted_idx):
            box = boxes_c[idx]
            max_iou = 0
            max_idx = -1
            for j, gt_box in enumerate(gt_c):
                iou = compute_iou(box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            if max_iou >= iou_threshold and gt_matched[max_idx] == 0:
                tp[i] = 1
                gt_matched[max_idx] = 1
            else:
                fp[i] = 1
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        rec = tp_cumsum / len(gt_c)
        ap = compute_ap(prec, rec)
        aps.append(ap)
    return np.mean(aps)

print("Evaluation metrics defined")
```

## 11. 常见问题与易错点

**问题1：训练速度极慢**
R-CNN需要对约2000个候选区域分别提取CNN特征，单张图像处理时间约50秒。解决：使用SPP-Net加速，或后续的Fast R-CNN、Faster R-CNN。

**问题2：候选区域特征缩放变形**
候选区域形状不一，强制缩放到227×227会导致变形。解决：使用SPP层实现空间金字塔池化。

**问题3：多阶段训练繁琐**
需要分别训练CNN、SVM、回归器，无法端到端优化。解决：后续Fast R-CNN实现了单阶段训练。

**问题4：候选框质量依赖Selective Search**
如果候选框不完整或不准确，后续检测效果会受影响。解决：使用更好的候选区域生成算法。

**问题5：正负样本定义**
IoU阈值选择（通常0.3）对结果影响很大。解决：调参或使用硬负挖掘。

**问题6：内存占用大**
4000维特征向量 × 2000候选区域 × batch，需要大量内存。解决：减小batch或使用特征共享。

## 12. 学习总结

R-CNN的核心贡献在于：
1. **开创性**：首次将深度CNN用于目标检测，奠定了两阶段检测器的基础
2. **模块化设计**：候选区域→特征提取→分类→回归的流程清晰
3. **显著提升**：在PASCAL VOC上mAP从传统方法的~40%提升到~66%

其局限性推动了后续改进：
- SPP-Net：引入空间金字塔池化加速特征提取
- Fast R-CNN：实现单阶段训练和ROI Pooling
- Faster R-CNN：用RPN替代Selective Search实现真正端到端

理解R-CNN对于理解目标检测的发展脉络至关重要，它是现代检测器的鼻祖。

## 13. 练习题与思考题与思考题（含答案）

**练习题1**：
如果Selective Search生成500个候选区域而不是2000个，检测精度会提升还是下降？为什么？

**答案**：
可能下降。候选区域数量减少会降低召回率，因为某些小目标或遮挡目标可能没有被候选框覆盖。但计算速度会提升约4倍。这是一个精度-速度的权衡。

**练习题2**：
为什么R-CNN中SVM分类器需要单独训练，而不是直接使用CNN的softmax输出？

**答案**：
因为CNN的softmax是在ImageNet预训练任务上学习的，类别可能不匹配检测任务。实验发现使用SVM单独训练可以得到更好的分类边界。另外，正负样本的定义（IoU>0.3为正）不同于ImageNet分类任务，需要重新训练分类器。

**练习题3**：
R-CNN的边界框回归为什么使用对数尺度变换？

**答案**：
因为边界框的宽度和高度应该是正数。使用指数变换e^{dw}确保预测的宽高总是正的，同时也使得相对误差在各个尺度上均匀分布。

**练习题4**：
如果目标很小（比如小于32像素），R-CNN能否检测到？会遇到什么问题？

**答案**：
很难检测到。问题包括：候选区域内缩放到227×27后，小目标信息几乎损失殆尽；CNN的感受野相对目标太大，没有足够的细粒度特征。

**思考题**：
R-CNN与传统的滑动窗口方法相比，本质区别是什么？

**答案**：
滑动窗口使用固定的窗口在图像上密集扫描，对于不同尺寸目标需要多尺度扫描，计算量大。R-CNN通过Selective Search生成候选区域，数量少且多尺度，减少了无效计算。但候选区域不共享特征仍是瓶颈。

## 14. 学习路径建议建议

**入门阶段**：
1. 理解目标检测的基本概念：分类 vs 检测，包围框，IoU
2. 学习Selective Search算法原理
3. 理解CNN特征提取的基本原理

**进阶阶段**：
1. 对比R-CNN、SPP-Net、Fast R-CNN、Faster R-CNN的演进
2. 理解ROI Pooling和空间金字塔池化的区别
3. 学习RPN（区域提议网络）的工作原理

**深入阶段**：
1. 实现完整的R-CNN训练流程
2. 分析消融实验，理解各组件的贡献
3. 对比两阶段检测器与单阶段检测器的优劣

**推荐学习资源**：
- 原始论文：Rich feature hierarchies for accurate object detection and semantic segmentation (2014)
- 斯坦福CS231nLecture 11：Object Detection
- 代码实现：detectron2、mmdetection

建议学习时间：2-3周

## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述R-CNN的核心思想及适用场景。
<details><summary>参考答案</summary>
R-CNN通过数据驱动学习输入到输出的映射，适用于深度学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出R-CNN的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现R-CNN核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. R-CNN在什么情况下会失效？
2. 训练数据很少时，R-CNN还能有效工作吗？
3. 如何将R-CNN与其他方法结合？

