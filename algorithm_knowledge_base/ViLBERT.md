# ViLBERT 学习文档

> 视觉-语言BERT的开山之作，采用双塔Transformer架构分别处理图像和文本，通过协同注意力（Co-Attention）模块实现跨模态交互。

## 1. 算法基础认知

### 一句话定义

ViLBERT（Vision-and-Language BERT）是2019年提出的首个视觉-语言预训练模型，采用双塔结构分别处理图像和文本，通过协同注意力（Co-attentional Transformer）实现跨模态融合。

### 直觉类比

ViLBERT的工作方式类似于两个人分别用母语阅读同一份双语文件：
- 一个人专门读中文部分（文本塔）
- 另一个人专门读英文部分（图像塔）
- 当遇到不理解的地方，互相看一眼对方的文档（协同注意力）
- 最终两人都获得完整的理解

### 历史背景

- **2019年8月**：ViLBERT由Jiasen Lu等人在NeurIPS 2019发表
- **核心创新**：协同注意力机制（Co-Attention），让图像和文本在各自编码的同时互相"关注"
- **后续影响**：为后续大量多模态模型（LXMERT、UNITER、OSCAR等）奠定了双塔+交叉注意力的架构范式

### 算法定位

ViLBERT是**视觉语言预训练模型**，属于多模态理解模型，开创了"各自编码+交叉融合"的双塔架构范式。

---

## 2. 核心原理

### 双塔架构

ViLBERT包含两个独立的编码塔和一个协同注意力融合层：

```

图像区域(Faster R-CNN) → 图像塔(Transformer) → 视觉特征
                                                        → 协同注意力层 → 融合特征
文本(BERT Token) → 文本塔(Transformer) → 文本特征
```

### 图像塔

- 使用Faster R-CNN + ResNet-101提取图像区域特征
- 每个区域使用RoI Pooling得到2048维特征
- 加上5维位置编码（坐标+宽高比）
- 通过Transformer编码图像区域间的关系

### 文本塔

- 使用BERT-Base（12层Transformer）
- 处理文本token序列
- 输出768维的文本特征

### 协同注意力（Co-Attention）

协同注意力是ViLBERT的核心创新，其工作原理：

1. **图像注意文本**：图像特征作为query，文本特征作为key/value
2. **文本注意图像**：文本特征作为query，图像特征作为key/value

这种双向注意力让两种模态可以互相"查看"对方的信息。

### 预训练目标

ViLBERT使用三个预训练目标：

1. **掩膜语言建模（MLM）**：Mask部分文本token，根据图像和上下文预测
2. **掩膜视觉建模（MVM）**：Mask部分图像区域，预测其类别标签
3. **图文匹配（ITM）**：二分类判断图文是否匹配

---

## 3. 数学公式与推导

### 3.1 协同注意力计算

给定图像特征 $V \in \mathbb{R}^{N_v \times d}$ 和文本特征 $T \in \mathbb{R}^{N_t \times d}$：

图像到文本的注意力（图像关注文本）：

$$A_{v2t} = \text{Softmax}\left(\frac{(VW_q)(TW_k)^T}{\sqrt{d}}\right)$$
$$C_{v2t} = A_{v2t} \cdot (TW_v)$$

文本到图像的注意力（文本关注图像）：

$$A_{t2v} = \text{Softmax}\left(\frac{(TW_q)(VW_k)^T}{\sqrt{d}}\right)$$
$$C_{t2v} = A_{t2v} \cdot (VW_v)$$

其中 $W_q, W_k, W_v$ 是投影矩阵，$A$ 是注意力权重矩阵，$C$ 是加权后的上下文向量。

### 3.2 MLM损失

$$\mathcal{L}_{MLM} = -\mathbb{E}_{(v,w) \sim D} \log P(w_m | w_{\backslash m}, v)$$

其中 $w_m$ 是被Mask的token，$w_{\backslash m}$ 是未被Mask的token，$v$ 是图像特征。

### 3.3 MVM损失

$$\mathcal{L}_{MVM} = -\mathbb{E}_{(v,w) \sim D} \log P(o_m | v_{\backslash m}, w)$$

其中 $o_m$ 是被Mask区域的类别标签，$v_{\backslash m}$ 是未被Mask的区域。

### 3.4 ITM损失

$$\mathcal{L}_{ITM} = -\mathbb{E}_{(v,w) \sim D} [y \log p + (1-y) \log(1-p)]$$

其中 $y \in \{0,1\}$ 表示图文是否匹配，$p$ 是预测的匹配概率。

### 3.5 总损失

$$\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{MVM} + \mathcal{L}_{ITM}$$

---

## 4. 训练过程讲解

### 阶段一：特征提取

1. **图像区域提取**：使用Faster R-CNN检测图像中的物体区域
2. **区域特征编码**：每个区域提取2048维特征+5维位置（坐标+宽高）
3. **文本token化**：使用BERT tokenizer

### 阶段二：双塔编码

1. **图像塔**：区域特征经过位置编码后，通过Transformer学习区域间关系
2. **文本塔**：token通过BERT编码

### 阶段三：协同注意力融合

- 在多个Transformer层中交替使用自注意力和协同注意力
- 协同注意力让两个模态互相交换信息

### 阶段四：预训练任务

- MLM、MVM、ITM三个任务同时训练
- 共享编码器参数

### 训练细节

- 使用Conceptual Captions和COCO Captions数据集
- 优化器：Adam
- 学习率：4e-5
- batch size：512
- 训练步数：约100万步

---

## 5. 应用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| 视觉问答 | 根据图像回答问题 | "图中有几只猫？→ 2" |
| 视觉常识推理 | 图像+问题的推理 | 选择合理的答案 |
| 指代表达理解 | 理解"左边穿红衣服的人" | 定位到具体区域 |
| 文本到图像检索 | 用文字搜图 | "白猫在沙发上" |
| 图像到文本检索 | 用图搜文 | 找到描述该图片的文字 |
| 短语定位 | 定位文本中提到的物体 | "dog" → 图像中狗的位置 |

---

## 6. 优缺点分析

### 优点

1. **开创性工作**：首个视觉-语言BERT，奠定了双塔+协同注意力的范式
2. **协同注意力**：双向注意力让图文充分交互，理解更深入
3. **模块化设计**：图像塔和文本塔可以独立升级
4. **预训练有效**：预训练后在下游任务上显著提升

### 缺点

1. **依赖目标检测**：需要Faster R-CNN提取区域特征，计算量大
2. **双塔分离**：图文交互仅在协同注意力层发生，前期编码时没有交互
3. **推理速度慢**：目标检测+双塔编码+协同注意力，整体速度较慢
4. **模型规模大**：ResNet-101 + BERT + 协同注意力，参数量大

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class CoAttentionLayer(nn.Module):
    """
    协同注意力层（Co-Attention Layer）
    同时进行图像到文本 和 文本到图像的双向注意力
    """
    def __init__(self, hidden_dim=768, num_heads=12):
        super().__init__()
        # 图像到文本的交叉注意力
        self.v2t_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        # 文本到图像的交叉注意力
        self.t2v_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 自注意力
        self.self_attn_v = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # LayerNorm
        self.norm_v1 = nn.LayerNorm(hidden_dim)
        self.norm_t1 = nn.LayerNorm(hidden_dim)
        self.norm_v2 = nn.LayerNorm(hidden_dim)
        self.norm_t2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, v_feat, t_feat):
        """
        Args:
            v_feat: (B, N_v, D) 图像区域特征
            t_feat: (B, N_t, D) 文本特征
        Returns:
            v_out, t_out
        """
        # 1. 交叉注意力
        # 图像→文本
        v2t_out, _ = self.v2t_attn(v_feat, t_feat, t_feat)
        v_feat = self.norm_v1(v_feat + v2t_out)
        
        # 文本→图像
        t2v_out, _ = self.t2v_attn(t_feat, v_feat, v_feat)
        t_feat = self.norm_t1(t_feat + t2v_out)
        
        # 2. 自注意力
        v_self, _ = self.self_attn_v(v_feat, v_feat, v_feat)
        v_feat = self.norm_v2(v_feat + v_self)
        
        t_self, _ = self.self_attn_t(t_feat, t_feat, t_feat)
        t_feat = self.norm_t2(t_feat + t_self)
        
        # 3. FFN
        v_feat = v_feat + self.ffn_v(v_feat)
        t_feat = t_feat + self.ffn_t(t_feat)
        
        return v_feat, t_feat

class ViLBERT(nn.Module):
    """
    ViLBERT模型实现
    双塔架构：图像塔(Transformer) + 文本塔(BERT) + 协同注意力
    """
    def __init__(self, image_dim=2048, text_dim=768, hidden_dim=768,
                 num_heads=12, num_co_attention_layers=6, num_classes=3129):
        super().__init__()
        
        # 图像编码器
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        # 位置编码（x1,y1,x2,y2,w,h）
        self.image_position_encoding = nn.Linear(6, hidden_dim)
        
        # 图像塔Transformer（编码区域间关系）
        self.visual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, batch_first=True),
            num_layers=6
        )
        
        # 文本编码器（使用预训练BERT）
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        
        # 协同注意力层
        self.co_attention_layers = nn.ModuleList([
            CoAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_co_attention_layers)
        ])
        
        # ITM分类头
        self.itm_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # VQA分类头（可选项）
        self.vqa_head = nn.Linear(hidden_dim, num_classes)
        
        # MLM预测头
        self.mlm_head = nn.Linear(hidden_dim, self.text_encoder.config.vocab_size)
        
    def forward(self, image_features, image_boxes, input_ids, attention_mask,
                task='vqa'):
        """
        前向传播
        Args:
            image_features: (B, N_v, 2048) Faster R-CNN区域特征
            image_boxes: (B, N_v, 6) 区域位置 [x1,y1,x2,y2,w,h]
            input_ids: (B, N_t) 文本token IDs
            attention_mask: (B, N_t) 注意力掩码
            task: 'vqa', 'itm', 'mlm' 等
        """
        B = image_features.shape[0]
        
        # 1. 图像编码
        v_feat = self.image_projection(image_features)
        v_pos = self.image_position_encoding(image_boxes)
        v_feat = v_feat + v_pos
        v_feat = self.visual_transformer(v_feat)
        
        # 2. 文本编码
        t_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        t_feat = t_outputs.last_hidden_state
        
        # 3. 协同注意力融合
        for co_attn in self.co_attention_layers:
            v_feat, t_feat = co_attn(v_feat, t_feat)
        
        # 4. 任务特定输出
        if task == 'vqa':
            # 使用[CLS] token特征
            cls_feat = t_feat[:, 0]
            return self.vqa_head(cls_feat)
        
        elif task == 'itm':
            # 拼接图像[CLS]和文本[CLS]
            v_cls = v_feat[:, 0]
            t_cls = t_feat[:, 0]
            combined = torch.cat([v_cls, t_cls], dim=1)
            return self.itm_head(combined)
        
        elif task == 'mlm':
            return self.mlm_head(t_feat)
        
        else:
            return v_feat, t_feat

# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = ViLBERT()
    
    # 模拟输入
    B = 2
    N_v = 36  # 36个图像区域
    N_t = 20  # 20个文本token
    
    image_features = torch.randn(B, N_v, 2048)
    image_boxes = torch.randn(B, N_v, 6)
    input_ids = torch.randint(0, 30522, (B, N_t))
    attention_mask = torch.ones(B, N_t)
    
    # 测试不同任务
    vqa_out = model(image_features, image_boxes, input_ids, attention_mask, task='vqa')
    itm_out = model(image_features, image_boxes, input_ids, attention_mask, task='itm')
    
    print(f"VQA输出形状: {vqa_out.shape}")  # (2, 3129)
    print(f"ITM输出形状: {itm_out.shape}")  # (2, 2)
    print("ViLBERT前向传播成功!")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandcraftMultiHeadAttention(nn.Module):
    """手工多头注意力"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        B = query.shape[0]
        
        Q = self.W_q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        
        return self.W_o(out), attn

class HandcraftCoAttention(nn.Module):
    """
    手工协同注意力模块
    同时实现 图像→文本 和 文本→图像 的双向注意力
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        # 视觉到文本的交叉注意力
        self.v2t_attn = HandcraftMultiHeadAttention(d_model, n_heads)
        # 文本到视觉的交叉注意力
        self.t2v_attn = HandcraftMultiHeadAttention(d_model, n_heads)
        
        # 前馈网络
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1_v = nn.LayerNorm(d_model)
        self.norm1_t = nn.LayerNorm(d_model)
        self.norm2_v = nn.LayerNorm(d_model)
        self.norm2_t = nn.LayerNorm(d_model)
        
    def forward(self, v_feat, t_feat):
        """
        v_feat: (B, N_v, D) 图像区域特征
        t_feat: (B, N_t, D) 文本特征
        """
        # 图像→文本交叉注意力
        v_attn, v_attn_weights = self.v2t_attn(v_feat, t_feat, t_feat)
        v_feat = self.norm1_v(v_feat + v_attn)
        
        # 文本→图像交叉注意力
        t_attn, t_attn_weights = self.t2v_attn(t_feat, v_feat, v_feat)
        t_feat = self.norm1_t(t_feat + t_attn)
        
        # FFN
        v_feat = self.norm2_v(v_feat + self.ffn_v(v_feat))
        t_feat = self.norm2_t(t_feat + self.ffn_t(t_feat))
        
        return v_feat, t_feat, v_attn_weights, t_attn_weights

class HandcraftViLBERT(nn.Module):
    """
    手工实现的简化ViLBERT
    包含图像编码、文本编码和协同注意力
    """
    def __init__(self, d_model=768, n_heads=12, n_layers=6,
                 vocab_size=30522, max_img_regions=100):
        super().__init__()
        
        # 文本嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.text_pos_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        
        # 文本编码器层
        self.text_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
            for _ in range(6)
        ])
        
        # 图像编码
        self.image_proj = nn.Linear(2048, d_model)
        self.image_pos_embedding = nn.Linear(6, d_model)
        
        # 图像编码器层
        self.visual_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
            for _ in range(6)
        ])
        
        # 协同注意力层
        self.co_attention_layers = nn.ModuleList([
            HandcraftCoAttention(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
    def encode_text(self, input_ids):
        """文本编码"""
        B, L = input_ids.shape
        x = self.token_embedding(input_ids) + self.text_pos_embedding[:, :L, :]
        for layer in self.text_layers:
            x = layer(x)
        return x
    
    def encode_image(self, image_features, image_boxes):
        """图像编码"""
        v = self.image_proj(image_features) + self.image_pos_embedding(image_boxes)
        for layer in self.visual_layers:
            v = layer(v)
        return v
    
    def forward(self, image_features, image_boxes, input_ids):
        """前向传播"""
        # 独立编码
        v_feat = self.encode_image(image_features, image_boxes)
        t_feat = self.encode_text(input_ids)
        
        # 协同注意力融合
        all_v_attn_weights = []
        all_t_attn_weights = []
        
        for layer in self.co_attention_layers:
            v_feat, t_feat, v_w, t_w = layer(v_feat, t_feat)
            all_v_attn_weights.append(v_w)
            all_t_attn_weights.append(t_w)
        
        return {
            'visual_features': v_feat,
            'text_features': t_feat,
            'v_attn_weights': all_v_attn_weights,
            't_attn_weights': all_t_attn_weights
        }

# 测试手工实现
if __name__ == "__main__":
    model = HandcraftViLBERT()
    
    # 模拟输入
    image_features = torch.randn(2, 36, 2048)
    image_boxes = torch.randn(2, 36, 6)
    input_ids = torch.randint(0, 30522, (2, 20))
    
    outputs = model(image_features, image_boxes, input_ids)
    
    print(f"视觉特征形状: {outputs['visual_features'].shape}")  # (2, 36, 768)
    print(f"文本特征形状: {outputs['text_features'].shape}")    # (2, 20, 768)
    print(f"协同注意力层数: {len(outputs['v_attn_weights'])}")  # 6
    print("手工ViLBERT测试通过!")
```

---

## 9. 可视化与结果理解

### 9.1 协同注意力可视化

协同注意力的权重可以直观展示图文之间的对应关系：
- 文本中的"cat"关注图像中猫所在的区域（t2v注意力）
- 图像中的猫区域关注文本中的"cat"（v2t注意力）
- 这种双向注意力让模型可以理解"猫"这个词对应图像中的哪个部分

### 9.2 双塔特征可视化

使用t-SNE或PCA降维可视化：
- 图像塔输出的区域特征聚集在特定区域（物体类别）
- 文本塔输出的token特征按照语义聚集
- 协同注意力后，两者的分布更加接近

### 9.3 预训练损失曲线

- MLM损失：缓慢下降，类似BERT预训练
- MVM损失：下降较快
- ITM损失：稳定下降，说明模型学会了匹配判断

---

## 10. 模型评估

### 10.1 评估指标

| 任务 | 评估指标 | 典型结果 |
|------|---------|---------|
| VQA 2.0 | 准确率 | 70.9% |
| RefCOCO+ | 定位准确率 | 72.34% |
| Flickr30K ITR | Recall@1 | 58.2% |

### 10.2 消融实验

ViLBERT论文中的重要消融实验结论：
- 移除协同注意力 → 性能下降约10%
- 移除MVM预训练 → 性能下降约3%
- 使用随机初始化替换BERT → 性能下降约15%

---

## 11. 常见问题与易错点

### Q1: 协同注意力和交叉注意力的区别？

交叉注意力通常指单方向的注意力（如文本注意图像）。协同注意力是双向的——图像注意文本的同时文本也注意图像。ViLBERT中的协同注意力是双向交叉注意力的组合。

### Q2: ViLBERT和VisualBERT的区别？

ViLBERT是双塔架构（各自编码后融合），VisualBERT是单塔架构（图文拼接后一起编码）。ViLBERT的图文交互发生在协同注意力层，VisualBERT的图文交互发生在自注意力层。

### Q3: 为什么需要图像区域的"位置编码"？

Faster R-CNN提取的区域特征不包含位置信息。位置编码告诉模型每个区域在图像中的空间位置（坐标和大小），这对理解"左边""右边""上面"等空间关系至关重要。

### Q4: MVM（掩膜视觉建模）是如何工作的？

类似于BERT的MLM，MVM随机Mask一些图像区域的特征，让模型根据其他区域和文本信息预测被Mask区域的类别标签。这迫使模型学习"视觉完形填空"能力。

### Q5: ViLBERT为什么是"开山之作"？

因为ViLBERT是第一个将BERT-style的预训练成功应用到视觉-语言领域的模型。它证明了MLM+MVM+ITM的预训练范式可以迁移到多模态领域。

---

## 12. 学习总结

### 核心知识点

1. **ViLBERT = 图像塔 + 文本塔 + 协同注意力**
2. **协同注意力**：双向交叉注意力，让两个模态充分交互
3. **三大预训练目标**：MLM（文本掩码）、MVM（视觉掩码）、ITM（图文匹配）
4. **依赖目标检测**：使用Faster R-CNN提取区域特征

### 架构速记

ViLBERT = BERT文本编码 + ResNet视觉编码 + Co-Attention融合 + MLM/MVM/ITM预训练

### 关键历史地位

ViLBERT是多模态预训练的开山之作，深刻影响了LXMERT、UNITER、VLBERT等后续模型。

---

## 13. 练习题与思考题（含答案）

### 习题1：协同注意力

**问题**：在一个协同注意力层中，共有几次注意力计算？

**答案**：4次。图像→文本交叉注意力、文本→图像交叉注意力、图像自注意力、文本自注意力。

### 习题2：双塔 vs 单塔

**问题**：ViLBERT的双塔架构相比VisualBERT的单塔架构有什么优势？

**答案**：双塔架构可以分别对图像和文本进行独立编码，利用各自领域的预训练模型（如BERT做文本、Faster R-CNN做图像）。单塔架构虽然更简单，但在编码阶段无法利用领域专用预训练模型。

### 习题3：MVM任务

**问题**：MVM损失中的$o_m$是什么？

**答案**：$o_m$是被Mask图像区域的物体类别标签，来自Faster R-CNN的检测结果。

### 习题4：应用场景

**问题**：ViLBERT在处理"左边的猫"这样的指代表达时，协同注意力如何发挥作用？

**答案**：文本中的"左边"通过t2v注意力关注图像中左边的区域；"猫"通过t2v注意力关注猫所在的区域。两者的注意力叠加，模型定位到"左边的猫"对应的区域。

### 习题5：思考题

**问题**：如果移除视觉塔中的Transformer层，直接将Faster R-CNN特征送入协同注意力，会有什么影响？

**答案**：移除视觉塔Transformer意味着图像区域之间没有信息交互。例如，"轮子"和"车身"区域之间无法建立关系，"一辆车"的整体理解会下降。实验表明视觉塔的Transformer层对性能有显著贡献。

---

## 14. 学习路径建议

### 前置知识
- BERT / Transformer
- Faster R-CNN（目标检测）
- 注意力机制（自注意力、交叉注意力）
- 预训练-微调范式

### 平行模型
- **VisualBERT**：单塔架构的视觉语言模型（更简单）
- **LXMERT**：三编码器架构（对象关系+语言+跨模态）
- **UNITER**：统一Transformer的VL模型

### 进阶方向
- **OSCAR**：使用检测标签作为锚点
- **VinVL**：更好的视觉特征和预训练
- **VLMo**：混合专家架构的VL模型

### 学习顺序建议

```
① BERT/Transformer → ② 目标检测 → ③ 注意力机制 → ④ ViLBERT → ⑤ LXMERT/UNITER
```
