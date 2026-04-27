# DETR（Detection Transformer）学习文档

> 端到端的目标检测Transformer，革新了传统检测范式。

## 1. 算法基础认知

### 一句话定义

DETR将目标检测建模为集合预测问题，使用Transformer解码器直接输出预测框，消除对NMS后处理步骤的依赖。

### 直觉类比

DETR就像让一个人看一张照片，然后直接说"这里有3辆车、2个人"——不需要先找出所有可能的候选区域（Selective Search），也不需要筛选重复的框（NMS）。Transformer直接给出最终答案。

### 历史背景

- **2020年5月**：Facebook AI提出DETR
- **2020年**：获得ECCV最佳论文奖

### 算法定位

DETR是**端到端目标检测模型**，属于Transformer在CV中的应用。

---

## 2. 核心原理

### 核心架构

1. CNN backbone提取特征
2. Transformer编码器处理特征序列
3. Transformer解码器接收object queries
4. FFN输出类别和框

### 关键创新

- **Object Queries**：可学习的查询向量
- **集合预测**：无NMS，直接输出
- **匈牙利匹配**：一对一匹配预测和GT

---

## 3. 调库实现

```python
import torch
import torch.nn as nn
from transformers import DetrModel, DetrImageProcessor

class DETRDetector:
    """DETR目标检测器"""
    def __init__(self, num_classes=91):
        self.model = DetrModel.from_pretrained('facebook/detr-resnet-50')
        self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
        
    def detect(self, image):
        inputs = self.processor(images=image, return_tensors='pt')
        outputs = self.model(**inputs)
        
        # 后处理
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=[(image.shape[0], image.shape[1])]
        )[0]
        
        return results

# 简化实现
class SimpleDETR(nn.Module):
    """简化版DETR"""
    def __init__(self, num_queries=100, num_classes=80, d_model=256):
        super(SimpleDETR, self).__init__()
        self.num_queries = num_queries
        
        # CNN backbone (简化的)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # 投影层
        self.input_proj = nn.Conv2d(64, d_model, 1)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, 8)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6)
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # 预测头
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)
        )
        
    def forward(self, x):
        # 特征提取
        x = self.backbone(x)
        x = self.input_proj(x)  # [B, C, H, W]
        
        # 展平为序列
        h, w = x.shape[2:]
        x = x.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        # 编码
        memory = self.encoder(x)
        
        # 解码
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(1), 1)
        hs = self.decoder(queries, memory)
        
        # 预测
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

# 测试
if __name__ == "__main__":
    model = SimpleDETR()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"预测类别形状: {out['pred_logits'].shape}")
    print(f"预测框形状: {out['pred_boxes'].shape}")
```

---

## 4. 优缺点

### 优点

1. **端到端**：无需NMS等后处理
2. **简洁**：架构统一
3. **长序列**：适合大物体检测

### 缺点

1. **收敛慢**：需要更多训练时间
2. **小物体**：性能不如传统方法
3. **计算量**：对高分辨率图像敏感

---

## 5. 性能对比

| 方法 | AP | FPS |
|------|-----|-----|
| Faster RCNN | 42.0 | 15 |
| RetinaNet | 39.1 | 32 |
| DETR | 42.0 | 28 |
| DETR-R101 | 43.5 | - |

---

## 6. 学习路径

- 前置：Transformer、目标检测基础
- 平行：Deformable DETR、CARAFE
- 进阶：Swin R-CNN、ViT-RCNN