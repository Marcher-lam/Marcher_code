# SAM（Segment Anything Model）学习文档

> Meta发布的"分割一切"模型，NLP提示学习引入CV的里程碑。

## 1. 算法基础认知

### 一句话定义

SAM是Meta于2023年4月发布的"分割一切"模型，首次将NLP的提示学习范式引入图像分割。

### 历史背景

- **2023年4月**：SAM论文发布
- **核心创新**：提示驱动的通用分割
- **数据集**：SA-1B（1100万张图像，10亿个mask）

### 算法定位

SAM是**通用图像分割模型**，基于Transformer编码器-解码器架构。

---

## 2. 核心原理

### 提示驱动

- 接收多种提示：点、框、文字、mask
- 输出对应的分割结果

### 三组件

1. **图像编码器**：ViT-Huge，提取图像特征
2. **提示编码器**：编码各种提示
3. **掩码解码器**：预测分割mask

### 数据引擎

- 阶段1：辅助标注
- 阶段2：半自动
- 阶段3：全自动

---

## 3. 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMModel(nn.Module):
    """Segment Anything Model简化实现"""
    def __init__(self, image_size=1024, embed_dim=1280):
        super(SAMModel, self).__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        
        # 图像编码器 (ViT-H)
        from transformers import ViTModel
        self.image_encoder = ViTModel.from_pretrained("google/vit-huge-patch14-224").encoder
        
        # 提示编码器
        self.point_embedding = nn.Embedding(4, embed_dim)
        self.box_embedding = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(512, embed_dim),  # 简化
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 掩码解码器
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        
    def forward(self, image, prompt):
        """
        image: (B, 3, H, W)
        prompt: 字典，包含'points', 'boxes', 'text'等
        """
        # 图像编码
        image_feat = self.image_encoder(image)
        
        # 提示编码
        prompt_emb = self.encode_prompt(prompt)
        
        # 掩码预测
        mask = self.predict_mask(image_feat, prompt_emb)
        
        return mask
    
    def encode_prompt(self, prompt):
        """编码各种提示"""
        prompt_emb = []
        
        if 'points' in prompt:
            points = prompt['points']  # (N, 2) xy坐标
            point_emb = self.point_embedding.weight[0:points.shape[0]]
            prompt_emb.append(point_emb)
            
        if 'boxes' in prompt:
            boxes = prompt['boxes']  # (N, 4) xyxy
            box_emb = self.box_embedding(boxes)
            prompt_emb.append(box_emb)
            
        if 'text' in prompt:
            text = prompt['text']  # (N, 512)
            text_emb = self.text_embedding(text)
            prompt_emb.append(text_emb)
            
        if len(prompt_emb) > 0:
            return torch.cat(prompt_emb, dim=1)
        return None
    
    def predict_mask(self, image_feat, prompt_emb):
        """预测分割掩码"""
        # 简化：直接使用全局特征预测
        B, N, D = image_feat.shape
        
        # 全局平均池化
        global_feat = image_feat.mean(dim=1)  # (B, D)
        
        # 添加prompt信息
        if prompt_emb is not None:
            global_feat = global_feat + prompt_emb.mean(dim=0, keepdim=True)
            
        # 重塑为特征图
        feat_map = global_feat.view(B, D, 1, 1)
        
        # 解码
        mask = self.mask_decoder(feat_map)
        mask = torch.sigmoid(mask)
        
        return mask

# 使用SAM官方实现
def use_sam_from_pretrained():
    """使用预训练SAM"""
    from segment_anything import sam_model_registry, SamPredictor
    
    # 加载模型
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    predictor = SamPredictor(sam)
    
    # 预测
    # image = ... # 加载图像
    # predictor.set_image(image)
    
    # 点提示
    # point_coords = np.array([[500, 500]])
    # point_labels = np.array([1])
    # masks, scores, logits = predictor.predict(
    #     point_coords=point_coords,
    #     point_labels=point_labels,
    #     multimask_output=True
    # )
    
    # 框提示
    # box = np.array([x1, y1, x2, y2])
    # masks, scores, logits = predictor.predict(
    #     point_coords=None,
    #     box=box,
    #     multimask_output=False
    # )
    
    return predictor

# 自动化分割
def auto_segment(image, points_per_side=32):
    """自动分割全图"""
    from segment_anything import SamPredictor
    
    predictor = SamPredictor(torch.load("sam_vit_h.pth"))
    predictor.set_image(image)
    
    # 生成网格点
    h, w = image.shape[:2]
    points = []
    for i in range(points_per_side):
        for j in range(points_per_side):
            x = int((j + 0.5) * w / points_per_side)
            y = int((i + 0.5) * h / points_per_side)
            points.append([x, y])
    
    # 批量预测
    all_masks = []
    for point in points:
        masks, _, _ = predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([1])
        )
        all_masks.append(masks)
        
    return all_masks

if __name__ == "__main__":
    print("SAM模型已定义")
```

---

## 4. 性能

- 在SA-1B上零样本训练
- 零样本迁移能力：
  - 边缘检测
  - 物体提议
  - 实例分割
  - 语义分割

---

## 5. 学习路径

- 前置：ViT, Transformer, 提示学习
- 进阶：SAM 2, 交互式分割