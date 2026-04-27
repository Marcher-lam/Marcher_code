# ALBEF 学习文档

> 对齐后融合（Align Before Fuse）的视觉语言模型，通过对比学习先对齐图像与文本特征，再通过多模态编码器深度融合。

## 1. 算法基础认知

### 一句话定义

ALBEF（Align Before Fuse）提出"先对齐再融合"的核心思想——在对图像和文本进行深度融合之前，先通过对比学习将两者的特征空间对齐，从而获得更好的多模态表示。

### 直觉类比

想象一个翻译小组协作：首先，两位翻译各自学习对方的语言词汇对照表（对齐阶段），然后再一起讨论句子结构和上下文（融合阶段）。如果一开始就没有建立词汇对照表，直接讨论复杂句法就会因为基础概念不匹配而效率低下。ALBEF正是用对比学习先建立"视觉-语言词汇对照表"。

### 历史背景

- **2021年7月**：Salesforce Research发布ALBEF论文
- **核心创新**：提出动量蒸馏（Momentum Distillation）处理噪声图文对数据
- **影响**：在VQAv2、NLVR2、Image Captioning等任务上达到SOTA，成为后续多模态模型（BLIP、BLIP-2等）的基础

### 算法定位

ALBEF是**视觉语言预训练模型**，属于多模态理解与生成模型，强调通过对比学习实现特征空间对齐。

---

## 2. 核心原理

### 三组件架构

ALBEF包含三个核心模块：

1. **图像编码器**：ViT-B/16（12层Transformer），将图像编码为patch序列特征
2. **文本编码器**：6层Transformer，对文本进行编码
3. **多模态编码器**：6层Transformer（带交叉注意力层），深度融合图像和文本特征

### 工作流程

```
图像 → ViT编码器 → 图像特征
                              → 对比学习(ITC) → 特征对齐
文本 → BERT编码器 → 文本特征
                              → 多模态编码器 → 融合特征 → [ITM/MLM任务]
```

### 预训练目标

ALBEF使用三个预训练目标：

1. **图文对比学习（ITC, Image-Text Contrastive）**：拉近匹配图文对的距离，推远不匹配图文对
2. **图文匹配（ITM, Image-Text Matching）**：二分类任务判断图文是否匹配
3. **掩膜语言建模（MLM, Masked Language Modeling）**：根据图像和上下文文本预测被Mask的词汇

### 动量蒸馏

ALBEF维护一个动量版本的模型（Momentum Model），通过EMA（指数移动平均）更新：

$$\theta_m \leftarrow m \cdot \theta_m + (1 - m) \cdot \theta$$

其中 $m$ 是动量系数（通常取0.995），$\theta$ 是当前模型参数，$\theta_m$ 是动量模型参数。

动量模型为ITC和MLM生成伪标签，作为额外的监督信号，特别适用于有噪声的图文对数据。

---

## 3. 数学公式与推导

### 3.1 图文对比学习（ITC）损失

令 $I$ 为图像特征，$T$ 为文本特征。在batch中，有N个图文对 $(I_i, T_i)$ 为正样本，其余 $N^2 - N$ 对为负样本。

图像到文本的对比损失：

$$\mathcal{L}_{i2t} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(s(I_i, T_i)/\tau)}{\sum_{j=1}^{N} \exp(s(I_i, T_j)/\tau)}$$

文本到图像的对比损失：

$$\mathcal{L}_{t2i} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(s(T_i, I_i)/\tau)}{\sum_{j=1}^{N} \exp(s(T_i, I_j)/\tau)}$$

ITC总损失为两者之和：

$$\mathcal{L}_{itc} = \frac{1}{2}(\mathcal{L}_{i2t} + \mathcal{L}_{t2i})$$

其中 $s(I,T) = \frac{I^T T}{\|I\|\|T\|}$ 是余弦相似度，$\tau$ 是温度系数。

### 3.2 图文匹配（ITM）损失

ITM是一个二分类问题，使用交叉熵损失：

$$\mathcal{L}_{itm} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

其中 $y_i \in \{0,1\}$ 是真实标签，$p_i$ 是模型预测的匹配概率。

### 3.3 掩膜语言建模（MLM）损失

MLM使用交叉熵损失预测被Mask的词汇：

$$\mathcal{L}_{mlm} = -\frac{1}{N_m} \sum_{i=1}^{N_m} \log P(w_i | I, \hat{T})$$

其中 $N_m$ 是被Mask的token总数，$w_i$ 是原始token，$\hat{T}$ 是被Mask后的文本, $I$ 是图像特征。

### 3.4 总损失

$$\mathcal{L} = \mathcal{L}_{itc} + \mathcal{L}_{itm} + \mathcal{L}_{mlm}$$

---

## 4. 训练过程讲解

### 阶段一：特征提取

- 图像通过ViT被分为16×16的patches，编码为序列特征
- 文本通过tokenizer转换为token IDs，再通过BERT编码

### 阶段二：对比学习对齐

- 计算图像特征和文本特征的余弦相似度矩阵
- 使用InfoNCE损失拉近匹配对、推远非匹配对
- 动量模型提供额外的伪标签监督

### 阶段三：多模态融合

- 将对齐后的图文特征送入多模态编码器
- 多模态编码器包含交叉注意力层
- 融合特征用于ITM和MLM任务

### 训练技巧

- **难负样本挖掘**：ITM任务使用ITC得分最高的负样本作为难负样本
- **动量蒸馏**：解决噪声图文对带来的错误监督问题
- **梯度队列**：维护一个特征队列以增加负样本数量

---

## 5. 应用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| 图文检索 | 图像搜索文本 / 文本搜索图像 | 根据"黑猫"搜索相关图片 |
| 视觉问答 | 根据图像回答问题 | "图中有什么动物？→ 猫" |
| 图像描述 | 为图像生成文字描述 | "一只黑猫趴在沙发上" |
| 视觉推理 | 涉及图像的多模态推理 | "猫是否在沙发上？" |
| 零样本分类 | 无需训练直接分类新类别 | 结合CLIP风格的prompt模板 |

---

## 6. 优缺点分析

### 优点

1. **先对齐后融合**：对比学习在融合前对齐特征，减少多模态编码器的负担
2. **动量蒸馏机制**：有效处理数据噪声，提升训练稳定性
3. **强大的表示能力**：在多个下游任务上达到SOTA
4. **高效的预训练**：相比从头训练多模态模型，预训练效率更高

### 缺点

1. **双阶段处理**：编码和解码流程较复杂，推理速度较慢
2. **依赖目标检测**：原始版本使用Faster R-CNN的特征（后续改进可用ViT替代）
3. **参数量大**：完整的ALBEF模型参数量较大
4. **对比学习的限制**：batch size大小对对比学习效果影响显著

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor
from PIL import Image
import requests

class ALBEFModel(nn.Module):
    """
    ALBEF模型的PyTorch实现
    使用HuggingFace预训练组件构建
    """
    def __init__(self, vision_model="google/vit-base-patch16-224",
                 text_model="bert-base-uncased", embed_dim=256):
        super().__init__()
        
        # 图像编码器：ViT
        self.visual_encoder = ViTModel.from_pretrained(vision_model)
        # 图像特征投影到统一维度
        self.vision_proj = nn.Linear(self.visual_encoder.config.hidden_size, embed_dim)
        
        # 文本编码器：BERT
        self.text_encoder = BertModel.from_pretrained(text_model)
        # 文本特征投影到统一维度
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        
        # 多模态编码器（交叉注意力层）
        self.cross_attention = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4,
                batch_first=True
            ) for _ in range(4)
        ])
        
        # ITM分类头
        self.itm_head = nn.Linear(embed_dim * 2, 2)
        
        # MLM分类头
        self.mlm_head = nn.Linear(embed_dim, self.text_encoder.config.vocab_size)
        
        # 温度参数（可学习）
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        
    def contrastive_loss(self, image_feat, text_feat):
        """图文对比学习损失（ITC）"""
        # 归一化
        image_feat = F.normalize(image_feat, dim=1)
        text_feat = F.normalize(text_feat, dim=1)
        
        # 相似度矩阵（N x N）
        sim = image_feat @ text_feat.t() / self.temp
        
        # 标签：对角线为正样本
        labels = torch.arange(sim.shape[0], device=sim.device)
        
        # 对称的对比损失
        loss_i2t = F.cross_entropy(sim, labels)
        loss_t2i = F.cross_entropy(sim.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def forward(self, pixel_values, input_ids, attention_mask, 
                labels=None, match_labels=None):
        """
        前向传播
        Args:
            pixel_values: 图像张量 (B, 3, H, W)
            input_ids: 文本token IDs (B, seq_len)
            attention_mask: 注意力掩码 (B, seq_len)
            labels: MLM标签 (B, seq_len)，-100表示忽略
            match_labels: ITM标签 (B,)
        """
        B = pixel_values.shape[0]
        
        # 1. 图像编码
        visual_outputs = self.visual_encoder(pixel_values)
        image_feat = visual_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        image_feat_proj = self.vision_proj(image_feat)
        
        # 2. 文本编码
        text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_feat = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_feat_proj = self.text_proj(text_feat)
        
        # 3. 对比学习损失
        loss_itc = self.contrastive_loss(image_feat_proj, text_feat_proj)
        
        # 4. 多模态编码
        # 将图文特征拼接并通过交叉注意力层
        multimodal_feat = torch.cat([
            image_feat.unsqueeze(1),
            text_feat.unsqueeze(1)
        ], dim=1)  # (B, 2, D)
        
        for layer in self.cross_attention:
            multimodal_feat = layer(multimodal_feat)
        
        img_out = multimodal_feat[:, 0]  # 图像侧输出
        txt_out = multimodal_feat[:, 1]  # 文本侧输出
        
        # 5. ITM损失
        if match_labels is not None:
            itm_input = torch.cat([img_out, txt_out], dim=1)
            itm_logits = self.itm_head(itm_input)
            loss_itm = F.cross_entropy(itm_logits, match_labels)
        else:
            loss_itm = torch.tensor(0.0)
        
        # 6. MLM损失
        if labels is not None:
            # 使用文本侧输出预测被Mask的词汇
            mlm_logits = self.mlm_head(txt_out.unsqueeze(1)).squeeze(1)
            loss_mlm = F.cross_entropy(mlm_logits, labels, ignore_index=-100)
        else:
            loss_mlm = torch.tensor(0.0)
        
        return {
            'loss': loss_itc + loss_itm + loss_mlm,
            'loss_itc': loss_itc,
            'loss_itm': loss_itm,
            'loss_mlm': loss_mlm,
            'image_feat': image_feat_proj,
            'text_feat': text_feat_proj
        }

# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = ALBEFModel()
    
    # 模拟输入
    pixel_values = torch.randn(4, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (4, 20))
    attention_mask = torch.ones(4, 20)
    
    # 前向传播
    outputs = model(pixel_values, input_ids, attention_mask)
    
    print(f"ITC损失: {outputs['loss_itc'].item():.4f}")
    print(f"图像特征形状: {outputs['image_feat'].shape}")
    print(f"文本特征形状: {outputs['text_feat'].shape}")
    print("ALBEF模型前向传播成功!")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)

class CrossAttentionLayer(nn.Module):
    """手工交叉注意力层（ALBEF核心组件）"""
    def __init__(self, d_model, n_heads, d_ff=2048):
        super().__init__()
        # 文本→图像的交叉注意力
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, text_feat, image_feat):
        # 交叉注意力：文本查询图像
        attn_out, _ = self.cross_attn(text_feat, image_feat, image_feat)
        text_feat = self.norm1(text_feat + attn_out)
        
        # FFN
        ffn_out = self.ffn(text_feat)
        text_feat = self.norm2(text_feat + ffn_out)
        
        return text_feat

class HandcraftALBEF(nn.Module):
    """
    手工实现的ALBEF核心模块
    包含：图像编码、文本编码、对比学习、多模态融合
    """
    def __init__(self, vocab_size=30522, d_model=768, n_heads=12, 
                 num_layers=6, max_seq_len=512):
        super().__init__()
        
        # 文本嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # 文本编码器（Transformer层）
        self.text_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4,
                                      batch_first=True)
            for _ in range(6)
        ])
        
        # 图像投影（假设输入已经是ViT提取的图像特征）
        self.image_projection = nn.Linear(768, d_model)
        
        # 多模态编码器（交叉注意力）
        self.multimodal_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads)
            for _ in range(num_layers)
        ])
        
        # 对比学习的投影头
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 256)
        )
        
        # ITM分类头
        self.itm_head = nn.Linear(d_model * 2, 2)
        
        # MLM预测头
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
    def encode_text(self, input_ids):
        """文本编码"""
        B, L = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :L, :]
        
        for layer in self.text_encoder_layers:
            x = layer(x)
        return x
    
    def contrastive_align(self, img_feat, txt_feat):
        """对比学习对齐"""
        img_proj = self.contrastive_head(img_feat.mean(dim=1))
        txt_proj = self.contrastive_head(txt_feat.mean(dim=1))
        
        img_proj = F.normalize(img_proj, dim=1)
        txt_proj = F.normalize(txt_proj, dim=1)
        
        return img_proj, txt_proj
    
    def forward(self, image_patches, input_ids, attention_mask=None):
        """
        前向传播
        Args:
            image_patches: 图像patch特征 (B, N_patches, 768)
            input_ids: 文本token IDs (B, L)
            attention_mask: 注意力掩码 (B, L)
        """
        # 1. 图像处理
        img_feat = self.image_projection(image_patches)
        img_feat = img_feat + self.pos_embedding[:, :image_patches.shape[1], :]
        
        # 2. 文本编码
        txt_feat = self.encode_text(input_ids)
        
        # 3. 对比学习对齐
        img_proj, txt_proj = self.contrastive_align(img_feat, txt_feat)
        
        # 4. 多模态融合（交叉注意力）
        # 文本特征作为query，图像特征作为key/value
        for layer in self.multimodal_layers:
            txt_feat = layer(txt_feat, img_feat)
        
        # 5. 汇聚特征
        img_cls = img_feat[:, 0]  # 图像[CLS]
        txt_cls = txt_feat[:, 0]  # 文本[CLS]
        
        # ITM预测
        itm_input = torch.cat([img_cls, txt_cls], dim=1)
        itm_logits = self.itm_head(itm_input)
        
        # MLM预测
        mlm_logits = self.mlm_head(txt_feat)
        
        return {
            'itm_logits': itm_logits,
            'mlm_logits': mlm_logits,
            'img_proj': img_proj,
            'txt_proj': txt_proj,
            'img_feat': img_feat,
            'txt_feat': txt_feat
        }

# 测试手工实现
if __name__ == "__main__":
    model = HandcraftALBEF()
    
    # 模拟输入：batch=2, 196个patches, 20个token
    image_patches = torch.randn(2, 196, 768)
    input_ids = torch.randint(0, 30522, (2, 20))
    
    outputs = model(image_patches, input_ids)
    
    print(f"ITM logits形状: {outputs['itm_logits'].shape}")  # (2, 2)
    print(f"MLM logits形状: {outputs['mlm_logits'].shape}")  # (2, 20, 30522)
    print(f"图像投影形状: {outputs['img_proj'].shape}")      # (2, 256)
    print("手工ALBEF实现测试通过!")
```

---

## 9. 可视化与结果理解

### 9.1 对比学习的特征对齐效果

ALBEF的对比学习将匹配的图文对在特征空间中拉近。经过训练后：
- 匹配的图像和文本特征余弦相似度趋近于1
- 不匹配的图像和文本特征余弦相似度趋近于0

### 9.2 注意力热力图可视化

多模态编码器的交叉注意力层可以可视化：
- 文本中的每个词关注图像的哪些区域
- 例如"猫"这个词会关注图像中猫所在的区域

### 9.3 训练损失曲线

典型训练过程中，三种损失的变化：
- ITC损失：快速下降（对比学习收敛较快）
- ITM损失：平稳下降
- MLM损失：缓慢下降（语言建模需要更多数据）

---

## 10. 模型评估

### 10.1 评估指标

| 任务 | 评估指标 | 说明 |
|------|---------|------|
| 图文检索 | R@1, R@5, R@10 | 检索召回率 |
| 视觉问答 | Accuracy | 答案准确率 |
| 图像描述 | BLEU, CIDEr, SPICE | 生成文本质量 |
| 视觉推理 | Accuracy | 推理正确率 |

### 10.2 评估流程

```python
def evaluate_retrieval(model, dataloader, device='cuda'):
    """评估图文检索性能"""
    model.eval()
    image_features = []
    text_features = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, texts = batch['image'].to(device), batch['text'].to(device)
            
            outputs = model(images, texts)
            image_features.append(outputs['image_feat'])
            text_features.append(outputs['text_feat'])
    
    # 计算所有图文对的相似度
    img_feats = torch.cat(image_features)
    txt_feats = torch.cat(text_features)
    
    sim_matrix = img_feats @ txt_feats.t()
    
    # 计算R@K
    batch_size = sim_matrix.shape[0]
    ranks = torch.argsort(sim_matrix, descending=True)
    
    r1 = (ranks == torch.arange(batch_size).unsqueeze(1)).any(dim=1).float().mean()
    
    return {'R@1': r1.item()}
```

---

## 11. 常见问题与易错点

### Q1: 为什么需要"先对齐再融合"？

如果不先对齐，图像和文本的特征处于不同的语义空间中，多模态编码器需要同时完成"对齐"和"融合"两个任务，学习效率低。先对齐后再融合，多模态编码器可以专注于学习跨模态的交互关系。

### Q2: 动量蒸馏解决了什么问题？

图文对数据通常含有噪声（如网页爬取的alt-text与图像不完全匹配）。动量蒸馏让模型从自身的动量版本中学习soft targets，而不是硬标签，提高了对噪声的鲁棒性。

### Q3: ITC和ITM的区别是什么？

ITC在全局级别对齐图文特征，计算的是整个图像和整个文本的相似度；ITM在实例级别判断图文是否匹配，通常使用更难的任务设定（如ITC得分高的负样本）。

### Q4: batch size对对比学习影响

对比学习依赖大量负样本。batch size太小会导致负样本不足，影响对齐质量。ALBEF使用队列机制（queue）来缓存历史batch的特征，扩大负样本池。

---

## 12. 学习总结

### 核心知识点

1. **ALBEF = 视觉编码器 + 文本编码器 + 多模态编码器**
2. **三大预训练目标**：ITC（对比学习对齐）、ITM（匹配判断）、MLM（掩码预测）
3. **先对齐后融合**是通过对比学习实现特征空间对齐后再进行跨模态交互
4. **动量蒸馏**通过EMA更新的动量模型产生软标签，提高噪声鲁棒性

### 关键启发

ALBEF证明了多模态学习中"对齐"和"融合"可以解耦为两个阶段，这种设计思想深刻影响了后续的BLIP、BLIP-2等模型。

---

## 13. 练习题与思考题（含答案）

### 习题1：理解ITC损失

**问题**：给定batch size=4的图文对，计算ITC损失时的相似度矩阵是什么形状？

**答案**：4×4的矩阵，其中对角线上的4个是正样本，其余12个是负样本。

### 习题2：动量蒸馏公式

**问题**：动量模型的更新公式为 $\theta_m \leftarrow m \cdot \theta_m + (1-m) \cdot \theta$。当m=0.995时，说明什么？

**答案**：动量模型更新非常缓慢，每个step只吸收0.5%的当前模型参数，保证了动量模型的稳定性。

### 习题3：多模态编码器的作用

**问题**：为什么ALBEF在多模态编码器中只让文本注意图像，而不是双向注意？

**答案**：ALBEF的设计以语言理解为中心，让文本特征通过交叉注意力从图像特征中获取视觉信息。这种非对称设计可以避免视觉和语言特征在浅层就过度耦合。

### 习题4：对比学习的温度系数

**问题**：温度系数 $\tau$ 对对比学习有什么影响？

**答案**：温度系数控制对负样本的惩罚力度。较小的 $\tau$ 使softmax分布更尖锐，加大难负样本的梯度；较大的 $\tau$ 使分布更平滑，对所有负样本一视同仁。ALBEF使用可学习的温度参数自动调整。

### 习题5：思考题

**问题**：如果去掉ITC损失，只使用ITM+MLM训练ALBEF，会有什么影响？

**答案**：没有ITC损失，图像和文本的特征空间无法在对齐。多模态编码器需要同时学习"如何对齐"和"如何融合"，训练难度增加，最终性能会下降。ITC相当于提供了一个良好的初始化，让后续的多模态融合更高效。

---

## 14. 学习路径建议

### 前置知识
- Transformer架构（自注意力、交叉注意力）
- ViT（Vision Transformer）
- 对比学习（SimCLR、MoCo）
- BERT（掩码语言建模）

### 平行模型
- **BLIP**：ALBEF的改进版，引入Captioning和Filtering机制
- **CLIP**：纯对比学习的图文对齐模型
- **ViLBERT**：双塔+协同注意力的VL模型先驱

### 进阶方向
- **BLIP-2**：引入Q-Former进一步优化图文对齐效率
- **CoCa**：对比学习+描述生成的统一框架
- **Flamingo**：少样本多模态理解模型

### 学习顺序建议

```
① Transformer → ② ViT/BERT → ③ 对比学习 → ④ ALBEF → ⑤ BLIP/BLIP-2
```
