# YOLO (You Only Look Once) 学习文档

> 实时目标检测的统一框架，单次前向传播完成检测。

---

## 1. 算法基础认知

### 1.1 发展背景

YOLO 由 Joseph Redmon 等人于 2015 年在论文《You Only Look Once: Unified, Real-Time Object Detection》中提出，实现了首个实时单阶段目标检测器，将检测问题转化为回归问题。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 单阶段检测器 |
| 速度 | 45-155 FPS |
| 精度 | mAP 63.4% |
| 特点 | 端到端 |

### 1.3 模型系列

| 模型 | 参数量 | FPS |
|------|--------|-----|
| YOLOv1 | 45M | 45 |
| YOLOv2 | 34M | 40 |
| YOLOv3 | 62M | 45 |
| YOLOv4 | 64M | 65 |
| YOLOv5 | 68M | 140 |
| YOLOv8 | 80M | 160 |

---

## 2. 核心原理

### 2.1 检测框架

将图像划分为 S×S 网格，每个网格预测：

- B 个边界框 (x, y, w, h)
- C 个类别概率
- 每个边界框包含置信度

### 2.2 输出张量

```
S × S × (B × 5 + C)

B: 边界框数量
5: x, y, w, h, conf
C: 类别数
```

### 2.3 非极大值抑制 (NMS)

去除重复检测：
1. 按置信度排序
2. 保留最高置信度框
3. 忽略与保留框重叠度过高的框

---

## 3. 数学公式与推导

### 3.1 边界框预测

$$b_x = \sigma(t_x) + c_x$$
$$b_y = \sigma(t_y) + c_y$$
$$b_w = p_w e^{t_w}$$
$$b_h = p_h e^{t_h}$$

### 3.2 置信度

$$P(\text{Object}) \times \text{IOU}(pred, truth)$$

### 3.3 损失函数

$$\mathcal{L} = \lambda_{coord} \mathcal{L}_{coord} + \lambda_{conf} \mathcal{L}_{conf} + \mathcal{L}_{class}$$

---

## 4. 训练过程讲解

### 4.1 训练配置

| 参数 | 值 |
|------|-----|
| Batch | 64 |
| LR | 0.001 |
| Epochs | 135 |
| Weight Decay | 0.0005 |

### 4.2 数据增强

- Random cropping
- Multi-scale training
- Color jittering

### 4.3 Anchor Boxes

使用 k-means 聚类生成先验框

---

## 5. 应用场景

### 5.1 典型应用

- **实时监控**：交通检测
- **自动驾驶**：车辆/行人检测
- **工业**：缺陷检测

### 5.2 代码示例

```python
import torch

# 加载 YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 检测
results = model('image.jpg')
results.show()
```

---

## 6. 调库实现

### 6.1 ultralytics 实现

```python
import torch

class YOLO:
    """YOLO 目标检测"""
    
    def __init__(self, model_name='yolov5s'):
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        
    def detect(self, image_path, conf=0.25):
        """检测目标"""
        results = self.model(image_path, conf=conf)
        
        # 获取结果
        boxes = results.xyxy[0]
        classes = results.names
        
        return boxes, classes
    
    def detect_batch(self, images):
        """批量检测"""
        results = self.model(images)
        return results


def demo():
    print("=== YOLO 演示 ===\n")
    
    # 创建模型
    yolo = YOLO('yolov5s')
    print(f"检测速度: 140 FPS")
    print(f"支持: 80 类检测")


if __name__ == "__main__":
    demo()
```

### 6.2 自定义实现

```python
import torch
import torch.nn as nn

class Conv(nn.Module):
    """卷积 + BN + LeakyReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOv1(nn.Module):
    """简化 YOLOv1"""
    
    def __init__(self, num_classes=20):
        super().__init__()
        
        # 特征提取
        self.backbone = nn.Sequential(
            Conv(3, 64, 7, stride=2),
            nn.MaxPool2d(2),
            Conv(64, 192, 3),
            nn.MaxPool2d(2),
            Conv(192, 512, 3),
            nn.MaxPool2d(2),
        )
        
        # 检测头
        self.head = Conv(512, 1024, 3)
        self.output = nn.Conv2d(1024, (5+num_classes)*5, 1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = self.output(x)
        
        return x
```

---

## 7. 手工代码实现

### 7.1 检测流程

```python
import numpy as np

class YOLODetector:
    """YOLO 检测器简化版"""
    
    def __init__(self, num_classes=80):
        self.num_classes = num_classes
        self.num_anchors = 3
        self.grid_size = 13
        
    def preprocess(self, image):
        """图像预处理"""
        # 调整大小
        image = cv2.resize(image, (416, 416))
        # 归一化
        image = image / 255.0
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        # 添加 batch 维度
        image = np.expand_dims(image, 0)
        
        return image
    
    def postprocess(self, output):
        """后处理"""
        # 解析输出
        # NMS
        boxes = []
        
        return boxes


def demo():
    print("=== YOLO 手工实现演示 ===\n")
    print(f"核心思想: 回归问题")
    print(f"速度: 实时检测")


if __name__ == "__main__":
    demo()
```

---

## 8. 优缺点分析

### 8.1 优点

1. **速度快**：单阶段检测
2. **端到端**：一体化
3. **泛化能力**：可迁移

### 8.2 缺点

1. **小目标**：召回率低
2. **密集检测**：效果差

### 8.3 改进方向

- YOLOv4: 引入 CSP, PAN
- YOLOv5: 工程优化

---

## 9. 可视化与结果理解

### 9.1 检测流程

```python
def visualize():
    print("""
    YOLO 检测流程:
    
    图片
      ↓
    特征提取 (CNN)
      ↓
    检测头 (1×1 conv)
      ↓
    解析输出 → NMS
      ↓
    边界框 + 类别
    
    总耗时: ~8ms (YOLOv5s)
    """)
```

---

## 10. 模型评估

### 10.1 COCO mAP

| 模型 | mAP |
|------|-----|
| YOLOv3 | 33.0% |
| YOLOv4 | 43.5% |
| YOLOv5s | 56.8% |
| YOLOv5l | 64.8% |

---

## 11. 学习总结

**核心要点**：

1. **回归问题**：端到端检测
2. **单次前向**：实时检测
3. **网格划分**：Spatial divide
4. **NMS**：去除重复

**YOLO 核心优势**：
- 速度快 155+ FPS
- 简单有效
- 工程化强

---

## 12. 练习题与思考题

### 12.1 选择题

1. YOLO是单阶段还是双阶段检测器？
   - A) 单阶段
   - B) 双阶段

2. YOLO的核心优势是：
   - A) 精度高
   - B) 速度快
   - C) 小目标检测

3. NMS的作用是：
   - A) 增加检测框
   - B) 去除重复
   - C) 提高置信度

### 12.2 简答题

1. YOLO vs Faster R-CNN的区别？
2. 检测流程是什么？
3. NMS原理是什么？

### 12.3 编程题

1. 实现NMS
2. 使用Ultralytics库检测图片
3. 比较YOLOv5和v8效果

---

## 13. 常见问题与易错点

### Q1: 小目标检测效果差？

**答案**：训练时增强小目标样本，或用FPN/PAN。

### Q2: 密集目标检测差？

**答案**：用Anchor-free或增加样本。

### Q3: 如何改进YOLO？

**答案**：YOLOv4引入CSP/PAN，YOLOv5工程优化。

### Q4: mAP怎么提高？

**答案**：更多训练数据、数据增强、更大模型。

### Q5: 部署选哪个版本？

**答案**：YOLOv5s最快，v5l精度最高。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
目标检测基础
    ↓
CNN理解
    ↓
YOLO原理
    ↓
Faster R-CNN
    ↓
YOLOv5/v8
    ↓
部署优化
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| R-CNN | 双阶段基础 |
| Fast R-CNN | 改进版 |
| Faster R-CNN | 区域建议 |
| YOLOv8 | 最新版 |

### 14.3 扩展阅读

1. Redmon et al. (2016). YOLO9000
2. Ultralytics (2023). YOLOv8

---

## 附录

### A. 参数速查

| 版本 | mAP | FPS |
|------|-----|-----|
| YOLOv5s | 56.8% | 155 |
| YOLOv5m | 64.0% | 123 |
| YOLOv5l | 64.8% | 99 |
| YOLOv8s | 52.5% | 180 |

### B. 参考

1. Redmon et al. (2016). You Only Look Once
2. Ultralytics YOLOv8文档

---

**文档结束**