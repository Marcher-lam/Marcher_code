# ViLT 学习文档

> 无卷积无目标检测的极简视觉语言模型（Vision-and-Language Transformer），首次将ViT的图像分块方法直接应用于多模态学习，完全摒弃CNN和区域检测。

## 1. 算法基础认知

### 一句话定义

ViLT（Vision-and-Language Transformer）是首个完全放弃CNN和目标检测的视觉语言预训练模型，直接将图像分块（patch）并通过线性投影映射为token，与文本token拼接后送入Transformer统一编码。

### 直觉类比

ViLT的工作方式就像一个双语阅读者，但他不再需要"提取重点段落"（目标检测）或"看图解"（CNN），而是直接把整页书（图像分块）和翻译文本（文本token）交错排列，逐行阅读。这种"读原文"的方式虽然看起来更笨拙，但实际上避免了信息丢失。

### 架构类型分类

ViLT属于第五种视觉语言架构（Type 5）：
- VE（视觉编码器）：轻量级（仅线性投影，无CNN/检测器）
- TE（文本编码器）：轻量级（仅embedding）
- MI（模态交互）：重量级（深层Transformer）

### 历史背景

- **2021年2月**：ViLT由Kim等人发表在ICML 2021
- **核心创新**：首次证明可以完全去掉CNN和目标检测，只用patch embedding实现VL预训练
- **效率**：训练速度比VisualBERT快约65倍（因为不需要Faster R-CNN提取特征）

### 算法定位

ViLT是**极简视觉语言Transformer**，属于单塔（Single-Stream）架构，以"最轻量的编码器+最重量级的交互"为设计哲学。

---

## 2. 核心原理

### 极简架构

ViLT的架构极其简单：

```
图像 → 分块(16×16) → 线性投影 → 图像token序列
文本 → Tokenizer → Embedding → 文本token序列
                                          ↓
                                拼接为单一序列
                                          ↓
                                Transformer编码器
                                          ↓
                                统一的多模态特征
```

### 图像处理（无CNN）

ViLT处理图像的方式与ViT完全相同：
1. 将图像分割为固定大小的patches（如16×16）
2. 每个patch通过线性投影（Convolution-free）映射为embedding
3. 加上位置编码和[CLS] token

### 模型对比

| 维度 | ViLBERT/VisualBERT | ViLT |
|------|-------------------|------|
| 图像编码 | Faster R-CNN + ResNet | 线性投影（无CNN） |
| 特征数 | 36个区域 | 196个patches |
| 位置编码 | 5维坐标 | 1D位置编码 |
| 推理速度 | 慢（检测器慢） | 快（无检测器） |
| 参数量 | 大（检测器+编码器） | 小（仅Transformer） |

### 预训练目标

ViLT使用三个预训练目标：
1. **图文匹配（ITM）**：二分类判断图文是否匹配
2. **掩膜语言建模（MLM）**：预测Mask的文本token
3. **词补对齐（WPA）**：预测被Mask的图像patch与文本词的对齐关系（可选）

---

## 3. 数学公式与推导

### 3.1 图像分块与嵌入

输入图像 $I \in \mathbb{R}^{H \times W \times 3}$，分割为 $N = HW / P^2$ 个patches：

每个patch通过线性投影：

$$v_i = E \cdot \text{flatten}(I_i) + p_i$$

其中 $E \in \mathbb{R}^{D \times 3P^2}$ 是投影矩阵，$p_i$ 是位置编码。

### 3.2 输入序列

$$X = [t_{[CLS]}, t_1, ..., t_m, v_{[IMG]}, v_1, ..., v_n]$$

其中 $t$ 是文本token，$v$ 是图像patch token，$[CLS]$ 和 $[IMG]$ 是特殊token。

### 3.3 Transformer编码

$$H = \text{Transformer}(X)$$

其中 $H \in \mathbb{R}^{(m+n+2) \times D}$ 是编码后的多模态特征。

### 3.4 ITM损失

$$\mathcal{L}_{ITM} = -\log P(y | H_{[CLS]})$$

其中 $y \in \{0,1\}$ 表示图文是否匹配。

### 3.5 MLM损失

$$\mathcal{L}_{MLM} = -\sum_{i \in M} \log P(t_i | H_{\backslash i})$$

其中 $M$ 是被Mask的文本位置集合。

### 3.6 WPA损失（Word-Patch Alignment）

WPA是一种辅助损失，预测被Mask的图像patch与文本词的对齐：

将被Mask的patch特征与所有词特征计算相似度，然后用对比损失：

$$\mathcal{L}_{WPA} = -\log \frac{\exp(s(v_m, t_+)/\tau)}{\sum_j \exp(s(v_m, t_j)/\tau)}$$

其中 $v_m$ 是被Mask的patch，$t_+$ 是与之对齐的词。

---

## 4. 训练过程讲解

### 阶段一：输入准备

1. **图像**：调整为224×224 → 分割为14×14=196个patches（patch_size=16）
2. **图像嵌入**：每个patch通过线性层投影到768维 + 位置编码
3. **文本**：BERT tokenizer分词 + embedding

### 阶段二：序列拼接

将文本token和图像patch token拼接：
```
[CLS] text_tokens [SEP] [IMG] patch_tokens
```
使用segment embedding区分模态。

### 阶段三：Transformer编码

拼接序列通过标准Transformer编码器（12层），所有位置通过自注意力充分交互。

### 阶段四：预训练任务

- ITM：使用[CLS] token判断图文匹配
- MLM：随机Mask 15%文本token并预测
- WPA：随机Mask部分图像patch，预测与文本的对齐

### 训练细节

- 优化器：AdamW，学习率1e-4
- 数据：MS-COCO + Visual Genome + Conceptual Captions
- Batch size：256
- 图像分辨率：224×224
- patch size：32×32（基础版）或16×16（大版）

---

## 5. 应用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| 视觉问答 | 根据图像回答问题 | "图像中有几辆车？" |
| 图文匹配 | 判断图文是否匹配 | 验证图文一致性 |
| 图文检索 | 双向检索 | 用文搜图/用图搜文 |
| 视觉推理 | 多模态推理 | "杯子和桌子是什么关系？" |
| 零样本分类 | 无需训练的分类 | 用文本标签分类图像 |

---

## 6. 优缺点分析

### 优点

1. **极简架构**：无CNN、无目标检测，只用线性投影
2. **速度快**：训练和推理速度比基于检测的模型快数十倍
3. **端到端训练**：整个模型可以端到端训练
4. **参数少**：参数量远小于ViLBERT/LXMERT
5. **容易部署**：不需要Faster R-CNN等复杂组件

### 缺点

1. **细粒度理解弱**：没有目标检测，对精细物体关系的理解不如基于区域的模型
2. **patch数量多**：196个patches vs 36个区域，序列更长
3. **空间信息有限**：只有1D位置编码，缺少空间关系编码
4. **在大规模数据上表现好**：在小数据集上不如基于检测的方法

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层
    将图像分为patches并通过线性投影映射为token
    无卷积！仅使用线性层
    """
    def __init__(self, image_size=224, patch_size=32, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        
        # 线性投影（无卷积）
        self.projection = nn.Linear(
            in_channels * patch_size * patch_size, 
            embed_dim
        )
        
    def forward(self, x):
        """
        x: (B, 3, H, W)
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size
        
        # 将图像分割为patches
        x = x.unfold(2, self.patch_size, self.patch_size) \
             .unfold(3, self.patch_size, self.patch_size) \
             .contiguous()
        x = x.view(B, C, self.n_patches, -1)
        x = x.permute(0, 2, 3, 1)  # (B, n_patches, patch_size*patch_size*C)
        x = x.reshape(B, self.n_patches, -1)
        
        # 线性投影
        x = self.projection(x)  # (B, n_patches, embed_dim)
        return x

class ViLT(nn.Module):
    """
    ViLT模型：极简视觉语言Transformer
    无CNN，无目标检测，纯线性投影+Transformer
    """
    def __init__(self, image_size=224, patch_size=32, text_dim=768,
                 hidden_dim=768, num_heads=12, num_layers=12, 
                 vocab_size=30522, max_text_len=40):
        super().__init__()
        
        # 图像编码（极简：仅线性投影）
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, hidden_dim)
        num_patches = self.patch_embed.n_patches
        
        # 图像特殊token ([IMG])
        self.img_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        
        # 文本嵌入
        self.text_embed = nn.Embedding(vocab_size, text_dim)
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_text_len, text_dim))
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 片段嵌入
        self.segment_embed = nn.Embedding(3, hidden_dim)
        
        # 特殊token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, 
                                      dim_feedforward=hidden_dim*4, 
                                      batch_first=True),
            num_layers=num_layers
        )
        
        self.ln = nn.LayerNorm(hidden_dim)
        
        # 任务头部
        self.itm_head = nn.Linear(hidden_dim, 2)
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, pixel_values, input_ids, attention_mask=None, task='itm'):
        """
        前向传播
        Args:
            pixel_values: (B, 3, H, W) 图像
            input_ids: (B, L) 文本token
            attention_mask: (B, L) 文本注意力掩码
            task: 'itm', 'mlm'
        """
        B = pixel_values.shape[0]
        L = input_ids.shape[1]
        N = self.patch_embed.n_patches
        
        # 1. 图像patch嵌入
        img_tokens = self.patch_embed(pixel_values)  # (B, N, D)
        img_cls = self.img_cls_token.expand(B, -1, -1)
        img_tokens = torch.cat([img_cls, img_tokens], dim=1)  # (B, N+1, D)
        img_tokens = img_tokens + self.img_pos_embed
        
        # 图像segment ID = 1
        img_seg = self.segment_embed(
            torch.ones(B, N + 1, dtype=torch.long, device=pixel_values.device)
        )
        img_tokens = img_tokens + img_seg
        
        # 2. 文本嵌入
        txt_tokens = self.text_embed(input_ids)
        txt_tokens = txt_tokens + self.text_pos_embed[:, :L, :]
        txt_tokens = self.text_proj(txt_tokens)
        
        # 文本segment ID = 0
        txt_seg = self.segment_embed(
            torch.zeros(B, L, dtype=torch.long, device=pixel_values.device)
        )
        txt_tokens = txt_tokens + txt_seg
        
        # 3. 添加[CLS]和[SEP]
        cls = self.cls_token.expand(B, -1, -1)
        sep = self.sep_token.expand(B, -1, -1)
        cls_seg = self.segment_embed(
            torch.full((B, 1), 2, dtype=torch.long, device=pixel_values.device)
        )
        sep_seg = self.segment_embed(
            torch.full((B, 1), 2, dtype=torch.long, device=pixel_values.device)
        )
        
        cls = cls + cls_seg
        sep = sep + sep_seg
        
        # 4. 拼接序列: [CLS] + 文本 + [SEP] + 图像
        combined = torch.cat([cls, txt_tokens, sep, img_tokens], dim=1)
        
        # 5. Transformer编码
        combined = self.transformer(combined)
        combined = self.ln(combined)
        
        # 6. 任务输出
        cls_out = combined[:, 0]  # [CLS] token
        
        if task == 'itm':
            return self.itm_head(cls_out)
        elif task == 'mlm':
            # 文本部分（不包括CLS和SEP）
            txt_out = combined[:, 1:1+L]
            return self.mlm_head(txt_out)
        else:
            return combined

# 使用示例
if __name__ == "__main__":
    model = ViLT()
    
    # 模拟输入
    B = 2
    pixel_values = torch.randn(B, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (B, 20))
    
    # 测试ITM
    itm_out = model(pixel_values, input_ids, task='itm')
    print(f"ITM输出形状: {itm_out.shape}")  # (2, 2)
    
    # 测试MLM
    mlm_out = model(pixel_values, input_ids, task='mlm')
    print(f"MLM输出形状: {mlm_out.shape}")  # (2, 20, 30522)
    
    print("ViLT前向传播成功!")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandcraftPatchEmbed(nn.Module):
    """
    手工图像分块嵌入（无卷积）
    将图像展平并线性投影
    """
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 线性投影层（等价于conv2d with kernel_size=patch_size, stride=patch_size）
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        
        # 手工分割
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, self.n_patches, -1)
        
        return self.proj(patches)

class HandcraftViLT(nn.Module):
    """
    手工实现的简化ViLT
    极简：只有patch_embed + text_embed + Transformer
    """
    def __init__(self, img_size=224, patch_size=32, vocab_size=30522,
                 d_model=768, n_heads=12, n_layers=12, max_len=40):
        super().__init__()
        
        n_patches = (img_size // patch_size) ** 2
        
        # 图像嵌入
        self.patch_embed = HandcraftPatchEmbed(img_size, patch_size, 3, d_model)
        self.img_pos = nn.Parameter(torch.randn(1, n_patches + 1, d_model))
        self.img_cls = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 文本嵌入
        self.word_embed = nn.Embedding(vocab_size, d_model)
        self.text_pos = nn.Parameter(torch.randn(1, max_len, d_model))
        
        # 类型嵌入
        self.type_embed = nn.Embedding(3, d_model)
        
        # Transformer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        
    def forward(self, pixel_values, input_ids):
        B = pixel_values.shape[0]
        L = input_ids.shape[1]
        N = self.patch_embed.n_patches
        
        # 图像token
        img_tokens = self.patch_embed(pixel_values)
        img_cls = self.img_cls.expand(B, -1, -1)
        img_tokens = torch.cat([img_cls, img_tokens], dim=1)
        img_tokens = img_tokens + self.img_pos
        img_tokens = img_tokens + self.type_embed(
            torch.ones(B, N+1, dtype=torch.long)
        )
        
        # 文本token
        txt_tokens = self.word_embed(input_ids)
        txt_tokens = txt_tokens + self.text_pos[:, :L, :]
        txt_tokens = txt_tokens + self.type_embed(
            torch.zeros(B, L, dtype=torch.long)
        )
        
        # 拼接
        x = torch.cat([txt_tokens, img_tokens], dim=1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln(x)
        return self.head(x[:, 0])

# 测试
if __name__ == "__main__":
    model = HandcraftViLT()
    x = torch.randn(2, 3, 224, 224)
    txt = torch.randint(0, 30522, (2, 20))
    out = model(x, txt)
    print(f"输出形状: {out.shape}")  # (2, 2)
    print("手工ViLT测试通过!")
```

---

## 9. 可视化与结果理解

### 9.1 Patch注意力可视化

ViLT的每层自注意力可以可视化：
- 底层：图像patch关注相邻patch（局部空间关系）
- 中层：图像patch关注同语义区域，文本token关注相关词
- 高层：图文token交叉关注，形成多模态理解

### 9.2 特征分布

ViLT的图像patch特征和文本token特征在Transformer编码后：
- 初始阶段：两种模态的特征分布在各自区域
- 深层：特征混合，图文token的语义空间趋于一致

### 9.3 与基于检测的模型对比

- 质量：ViLT在简单任务上接近ViLBERT，复杂任务（如细粒度VQA）仍有差距
- 速度：训练速度快65倍，推理速度快100倍
- 这使得ViLT适合需要实时性的应用

---

## 10. 模型评估

### 10.1 评估指标

| 任务 | 指标 | ViLT-B/32 | ViLT-B/16 |
|------|------|-----------|-----------|
| VQAv2 | Accuracy | 71.26% | 73.54% |
| NLVR2 | Accuracy | 75.70% | 77.49% |
| COCO IR (R@1) | Recall | 52.9% | 56.1% |

### 10.2 效率对比

| 模型 | 训练速度 | 推理速度 | 参数量 |
|------|---------|---------|--------|
| ViLBERT | 1× | 1× | 220M |
| ViLT-B/32 | 65× | 100× | 87M |

---

## 11. 常见问题与易错点

### Q1: ViLT为什么可以去掉CNN和目标检测？

ViLT将图像分割为固定大小的patches，每个patch通过线性投影为token。这种方法虽然丢失了CNN的局部感受野先验，但在大规模数据上，Transformer可以自行学习到类似的特征。

### Q2: 32×32和16×16 patch的区别？

32×32：图像被分为7×7=49个patches，更少但信息更粗。16×16：14×14=196个patches，信息更细但序列更长。16×16版本效果更好但训练更慢。

### Q3: ViLT的位置编码为什么只需要1D？

ViLT使用类似ViT的1D位置编码，因为Transformer本身可以学习到空间关系。虽然2D位置编码理论上更好，但1D位置编码已经足够且更简单。

### Q4: ViLT的[IMG] token的作用？

[IMG] token类似于ViT的[CLS] token，汇聚图像的整体信息。它与文本[CLS] token一起用于ITM分类。

### Q5: ViLT为什么称"Type-5"架构？

VL模型按"视觉编码器复杂度"和"模态交互深度"分类。ViLT的视觉编码器最轻（Type-5中的极简VE），但交互深度最大（重型MI）。

---

## 12. 学习总结

### 核心知识点

1. **ViLT = 图像patch线性投影 + 文本embedding + 单塔Transformer**
2. **无CNN、无目标检测**：极简的视觉编码
3. **Type-5架构**：最轻编码器 + 最重交互
4. **速度优势**：比基于检测的模型快65-100倍

### 架构速记

ViLT = ViT式图像分块 + BERT式文本嵌入 + 拼接后Transformer编码

### 关键洞见

ViLT证明了视觉语言模型中"卷积/检测"不是必须的，线性投影+Transformer的组合在大规模数据上可以取得有竞争力的结果，同时大幅提升效率。

---

## 13. 练习题与思考题（含答案）

### 习题1：Patch计算

**问题**：224×224的图像，patch_size=32，共有多少个patches？

**答案**：(224/32)² = 7² = 49个patches。

### 习题2：序列长度

**问题**：文本20个token，图像49个patches，加上特殊token后ViLT的序列长度是多少？

**答案**：1([CLS]) + 20(文本) + 1([SEP]) + 1([IMG]) + 49(patches) = 72。

### 习题3：与ViLBER的对比

**问题**：ViLT相比ViLBERT的参数量为什么少那么多？

**答案**：ViLBERT需要Faster R-CNN（ResNet-101 + RPN）+ BERT + 协同注意力模块。ViLT只有一个Transformer + 线性投影层。

### 习题4：训练速度

**问题**：ViLT训练时不需要Faster R-CNN提取特征，这对训练流程有什么影响？

**答案**：Faster R-CNN在训练时需要单独推理图像来提取区域特征，这无法与模型训练并行，且需要额外的存储。ViLT直接输入原始图像，所有组件端到端训练。

### 习题5：思考题

**问题**：ViLT的分块方式放弃了空间先验（如CNN的局部连接），这在什么情况下会成为问题？

**答案**：当需要理解精细的空间关系时（如"左边的猫在右边的狗上面"），ViLT的1D位置编码可能不足以编码复杂的2D空间关系。基于区域的检测方法可以提供更精确的空间定位信息。

---

## 14. 学习路径建议

### 前置知识
- ViT（Vision Transformer）
- BERT
- Transformer自注意力

### 平行模型
- **ViLBERT**：双塔+检测（复杂但精度高）
- **VisualBERT**：单塔+检测（简单但依赖检测）
- **CLIP**：双塔对比（无融合）

### 进阶方向
- **ALBEF**：对比学习+单塔融合的混合
- **BLIP**：Vit+理解+生成的统一模型
- **BEiT-3**：多模态统一掩码建模

### 学习顺序建议

```
① ViT → ② BERT → ③ 单塔多模态(ViLT) → ④ 对比双塔(CLIP) → ⑤ 混合架构(ALBEF/BLIP)
```
