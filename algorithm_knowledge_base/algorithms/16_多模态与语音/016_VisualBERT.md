# VisualBERT 学习文档

> 单塔Transformer的视觉语言统一模型，将图像和文本视为同一个序列进行一体化处理，以最简单的架构实现强大的跨模态理解。

## 1. 算法基础认知

### 一句话定义

VisualBERT是将图像区域特征和文本token特征拼接为同一个序列，送入单塔Transformer进行一体化编码的视觉语言模型，以极简架构实现跨模态理解。

### 直觉类比

想象一个人同时阅读一本中英对照的书籍，但不是分别看中文和英文再对照，而是将中文和英文交错排列成一行行，像"中文-英文-中文-英文"一样从头到尾阅读。每个词都能看到旁边的对应翻译——这就是单塔架构的精髓。

### 历史背景

- **2019年8月**：VisualBERT由Liunian Harold Li等人提出
- **核心创新**：极简的单塔架构，将图像区域当做"视觉token"与文本token一起编码
- **影响**：证明了简单的单塔+自注意力足以实现强大的跨模态理解

### 算法定位

VisualBERT是**视觉语言预训练模型**，属于单塔（Single-Stream）架构，以"最简架构实现最强效果"为设计理念。

---

## 2. 核心原理

### 单塔架构

VisualBERT的核心思想极其简单：

```
文本token: [CLS] a cat on the couch [SEP]
图像区域:  [IMG1] [IMG2] [IMG3] ... [IMG36]

合并序列: [CLS] a cat on the couch [SEP] [IMG1] [IMG2] ... [IMG36]
                                                        ↓
                                               BERT Transformer
                                                        ↓
                                              统一的多模态特征
```

### 关键组件

1. **文本嵌入**：使用BERT的token embedding + position embedding + segment embedding
2. **图像嵌入**：Faster R-CNN区域特征通过线性投影到BERT隐藏维度 + 可学习的位置编码
3. **片段嵌入**：使用不同的segment id区分文本（0）和图像（1）
4. **BERT backbone**：标准的BERT-base Transformer

### 自注意力的跨模态交互

在单塔架构中，图文交互通过Transformer的自注意力机制自动完成：
- 文本token可以关注所有图像区域
- 图像区域可以关注所有文本token
- 图文之间的注意力权重由Transformer自行学习

### 预训练目标

VisualBERT使用两个目标：

1. **掩膜语言建模（MLM）**：根据图像和上下文文本预测被Mask的token
2. **图文匹配（ITM）**：判断图文是否匹配

---

## 3. 数学公式与推导

### 3.1 输入表示

输入序列由文本和图像拼接：

$$X = [t_1, ..., t_n, v_1, ..., v_m]$$

其中 $t_i$ 是第 $i$ 个文本token的嵌入，$v_j$ 是第 $j$ 个图像区域的嵌入。

文本嵌入：

$$t_i = E_{token}(w_i) + E_{pos}(i) + E_{seg}(0)$$

图像嵌入：

$$v_j = W_v \cdot f_j + E_{pos}(n+j) + E_{seg}(1)$$

其中 $W_v$ 是图像特征的投影矩阵，$f_j$ 是Faster R-CNN提取的2048维特征。

### 3.2 自注意力跨模态交互

对于序列中的任意位置 $i$，自注意力计算：

$$h_i' = \sum_{j=1}^{n+m} \alpha_{ij} \cdot (h_j W_v)$$

$$\alpha_{ij} = \text{Softmax}\left(\frac{h_i W_q \cdot h_j W_k^T}{\sqrt{d}}\right)$$

由于自注意力机制，$h_i$ 可以关注所有其他位置，包括跨模态的位置。文本位置可以关注图像位置，反之亦然。

### 3.3 MLM损失

$$\mathcal{L}_{MLM} = -\mathbb{E}_{(v,w)} \sum_{i \in M} \log P(w_i | w_{\backslash M}, v)$$

其中 $M$ 是被Mask的文本位置集合。

### 3.4 ITM损失

$$\mathcal{L}_{ITM} = -\mathbb{E}_{(v,w)} [y \log p + (1-y) \log(1-p)]$$

其中使用[CLS] token的特征作为序列表示进行分类。

---

## 4. 训练过程讲解

### 阶段一：输入准备

1. **文本**：分词后得到token IDs + 片段ID（0）
2. **图像**：Faster R-CNN提取36个区域特征（2048维）+ 位置坐标
3. **投影**：图像特征通过线性层投影到768维
4. **拼接**：文本 + 图像组合为单一序列

### 阶段二：BERT编码

- 组合序列送入BERT-base
- 通过12层Transformer的自注意力机制
- 每个位置都能关注所有其他位置

### 阶段三：预训练任务

- MLM：随机Mask 15%的文本token并预测
- ITM：使用[CLS]进行图文匹配二分类

### 训练细节

- 预训练数据：MS-COCO + Visual Genome + Conceptual Captions
- 优化器：Adam，学习率5e-5
- Batch size：256
- 序列长度：文本≤50 + 图像≤50

---

## 5. 应用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| 视觉问答 | 根据图像回答问题 | "杯子里有什么？" |
| 视觉常识推理 | 图像+问题的推理选择 | 选择合理的答案 |
| 自然语言推理 | 图文匹配的二分类判断 | 文本是否描述图像 |
| 短语定位 | 定位文本中提到的物体 | "左边的人"定位 |
| 图文检索 | 双向检索 | 用图搜文/用文搜图 |

---

## 6. 优缺点分析

### 优点

1. **极简架构**：单塔架构极其简单，易于实现和理解
2. **自动跨模态交互**：自注意力机制天然支持跨模态交互，不需要专门设计交叉注意力
3. **参数共享**：所有模态共享同一组Transformer参数
4. **易于扩展**：可以简单加入其他模态（如音频）作为新的token

### 缺点

1. **序列长度问题**：图文拼接后序列变长，自注意力的计算复杂度是 $O((n+m)^2)$
2. **模态混淆**：所有token在同一个空间中嵌入，可能导致模态特异性信息的丢失
3. **依赖检测**：仍需要Faster R-CNN区域特征
4. **缺少独立预训练**：不能分别利用独立的视觉/语言预训练模型

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class VisualBERT(nn.Module):
    """
    VisualBERT模型
    单塔架构：将图像区域和文本token拼接为同一序列送入BERT
    """
    def __init__(self, bert_name="bert-base-uncased", image_dim=2048, 
                 hidden_dim=768, max_img_regions=50, num_classes=3129):
        super().__init__()
        
        # BERT backbone
        self.bert = BertModel.from_pretrained(bert_name)
        
        # 图像特征投影
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        
        # 图像位置嵌入（可学习）
        self.image_position_embedding = nn.Parameter(
            torch.zeros(1, max_img_regions, hidden_dim)
        )
        
        # 片段嵌入（区分文本和图像）
        # segment_ids: 0=文本, 1=图像
        self.segment_embedding = nn.Embedding(3, hidden_dim)
        
        # 任务头部
        self.itm_head = nn.Linear(hidden_dim, 2)
        self.mlm_head = nn.Linear(hidden_dim, self.bert.config.vocab_size)
        self.vqa_head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, text_ids, text_mask, image_features, 
                image_masks=None, task='vqa'):
        """
        前向传播
        Args:
            text_ids: (B, N_t) 文本token IDs
            text_mask: (B, N_t) 文本注意力掩码
            image_features: (B, N_v, 2048) 图像区域特征
            image_masks: (B, N_v) 图像区域掩码（可选）
            task: 任务类型
        """
        B = text_ids.shape[0]
        N_t = text_ids.shape[1]
        N_v = image_features.shape[1]
        
        # 1. 文本嵌入
        # BERT的word embedding
        text_emb = self.bert.embeddings.word_embeddings(text_ids)
        # BERT的position embedding
        text_pos = self.bert.embeddings.position_embeddings(
            torch.arange(N_t, device=text_ids.device).unsqueeze(0).expand(B, -1)
        )
        # 文本segment ID = 0
        text_seg = self.segment_embedding(
            torch.zeros(B, N_t, dtype=torch.long, device=text_ids.device)
        )
        text_emb = text_emb + text_pos + text_seg
        
        # 2. 图像嵌入
        img_emb = self.image_projection(image_features)
        img_pos = self.image_position_embedding[:, :N_v, :]
        img_seg = self.segment_embedding(
            torch.ones(B, N_v, dtype=torch.long, device=text_ids.device)
        )
        img_emb = img_emb + img_pos + img_seg
        
        # 3. 拼接序列
        combined_emb = torch.cat([text_emb, img_emb], dim=1)
        
        # 4. 组合注意力掩码
        if image_masks is None:
            image_masks = torch.ones(B, N_v, device=text_ids.device)
        combined_mask = torch.cat([text_mask, image_masks], dim=1)
        
        # 5. BERT编码
        outputs = self.bert(
            inputs_embeds=combined_emb,
            attention_mask=combined_mask
        )
        
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]  # [CLS] token
        
        # 6. 任务输出
        if task == 'vqa':
            return self.vqa_head(cls_output)
        elif task == 'itm':
            return self.itm_head(cls_output)
        elif task == 'mlm':
            # 只预测文本部分的MLM
            text_output = sequence_output[:, :N_t]
            return self.mlm_head(text_output)
        else:
            return {
                'sequence_output': sequence_output,
                'cls_output': cls_output,
                'text_output': sequence_output[:, :N_t],
                'image_output': sequence_output[:, N_t:]
            }

class VisualBERTForPretraining(nn.Module):
    """VisualBERT预训练包装器（MLM + ITM）"""
    def __init__(self, bert_name="bert-base-uncased", image_dim=2048, hidden_dim=768):
        super().__init__()
        self.visual_bert = VisualBERT(bert_name, image_dim, hidden_dim)
        
    def forward(self, text_ids, text_mask, image_features, 
                mlm_labels=None, itm_labels=None):
        # MLM预测
        mlm_logits = self.visual_bert(
            text_ids, text_mask, image_features, task='mlm'
        )
        
        # ITM预测
        itm_logits = self.visual_bert(
            text_ids, text_mask, image_features, task='itm'
        )
        
        losses = {}
        if mlm_labels is not None:
            losses['mlm_loss'] = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.shape[-1]),
                mlm_labels.view(-1),
                ignore_index=-100
            )
        
        if itm_labels is not None:
            losses['itm_loss'] = F.cross_entropy(itm_logits, itm_labels)
        
        if losses:
            losses['total_loss'] = sum(losses.values())
        
        return mlm_logits, itm_logits, losses

# 使用示例
if __name__ == "__main__":
    model = VisualBERT()
    
    # 模拟输入
    B, N_t, N_v = 2, 20, 36
    text_ids = torch.randint(0, 30522, (B, N_t))
    text_mask = torch.ones(B, N_t)
    image_features = torch.randn(B, N_v, 2048)
    
    # 测试不同任务
    vqa_out = model(text_ids, text_mask, image_features, task='vqa')
    itm_out = model(text_ids, text_mask, image_features, task='itm')
    outputs = model(text_ids, text_mask, image_features, task='all')
    
    print(f"VQA输出形状: {vqa_out.shape}")        # (2, 3129)
    print(f"ITM输出形状: {itm_out.shape}")        # (2, 2)
    print(f"序列输出形状: {outputs['sequence_output'].shape}")  # (2, 56, 768)
    print(f"文本部分形状: {outputs['text_output'].shape}")      # (2, 20, 768)
    print(f"图像部分形状: {outputs['image_output'].shape}")     # (2, 36, 768)
    print("VisualBERT前向传播成功!")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandcraftVisualBERT(nn.Module):
    """
    手工实现的简化VisualBERT
    核心：图文拼接 + 自注意力跨模态交互
    """
    def __init__(self, vocab_size=30522, d_model=768, n_heads=12, 
                 n_layers=12, d_ff=3072, max_seq_len=512, max_img=50):
        super().__init__()
        
        # 文本嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.text_pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # 图像嵌入
        self.img_projection = nn.Linear(2048, d_model)
        self.img_pos_encoding = nn.Parameter(torch.zeros(1, max_img, d_model))
        
        # 片段嵌入
        self.segment_embedding = nn.Embedding(3, d_model)
        
        # Transformer编码器
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # 输出头
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.itm_head = nn.Linear(d_model, 2)
        
    def forward(self, text_ids, image_features, task='all'):
        """
        手工VisualBERT前向传播
        """
        B = text_ids.shape[0]
        N_t = text_ids.shape[1]
        N_v = image_features.shape[1]
        
        # 文本嵌入
        t_emb = self.token_embedding(text_ids)
        t_emb = t_emb + self.text_pos_encoding[:, :N_t, :]
        t_emb = t_emb + self.segment_embedding(
            torch.zeros(B, N_t, dtype=torch.long)
        )
        
        # 图像嵌入
        v_emb = self.img_projection(image_features)
        v_emb = v_emb + self.img_pos_encoding[:, :N_v, :]
        v_emb = v_emb + self.segment_embedding(
            torch.ones(B, N_v, dtype=torch.long)
        )
        
        # 拼接
        x = torch.cat([t_emb, v_emb], dim=1)
        
        # 通过Transformer
        for layer in self.layers:
            x = layer(x)
        
        # 序列分离
        t_out = x[:, :N_t]
        v_out = x[:, N_t:]
        cls_out = x[:, 0]  # [CLS] token
        
        results = {}
        if task in ('all', 'mlm'):
            results['mlm_logits'] = self.mlm_head(t_out)
        if task in ('all', 'itm'):
            results['itm_logits'] = self.itm_head(cls_out)
        
        results['text_output'] = t_out
        results['image_output'] = v_out
        results['cls_output'] = cls_out
        
        return results

# 测试手工实现
if __name__ == "__main__":
    model = HandcraftVisualBERT()
    
    text_ids = torch.randint(0, 30522, (2, 20))
    image_features = torch.randn(2, 36, 2048)
    
    outputs = model(text_ids, image_features)
    
    print(f"MLM logits形状: {outputs['mlm_logits'].shape}")  # (2, 20, 30522)
    print(f"ITM logits形状: {outputs['itm_logits'].shape}")  # (2, 2)
    print(f"文本输出形状: {outputs['text_output'].shape}")    # (2, 20, 768)
    print(f"图像输出形状: {outputs['image_output'].shape}")   # (2, 36, 768)
    print("手工VisualBERT测试通过!")
```

---

## 9. 可视化与结果理解

### 9.1 注意力可视化

VisualBERT的跨模态注意力可以可视化：
- 文本中的"cat"与图像中猫区域的注意力权重很高
- 文本中的"on"与图像中空间关系的注意力
- [CLS] token融合了所有图文信息

### 9.2 序列编码可视化

拼接序列的编码结果：
- 文本部分：每个token都是上下文感知的文本表示
- 图像部分：每个区域都是"图文上下文感知"的视觉表示
- 序列边界处（文本和图像交界）附近呈现过渡特征

### 9.3 单塔 vs 双塔特征分布

- 单塔（VisualBERT）：文本和图像特征在同一语义空间中，分布混合
- 双塔（ViLBERT）：文本和图像特征在各自空间，通过交叉注意力连接

---

## 10. 模型评估

### 10.1 评估指标

| 任务 | 评估指标 | VisualBERT结果 |
|------|---------|---------------|
| VQA 2.0 | 准确率 | 70.8% |
| NLVR2 | 准确率 | 71.3% |
| Flickr30K IR | Recall@1 | 55.2% |
| RefCOCO+ | 定位准确率 | 71.6% |

### 10.2 消融实验

- 移除图像片段嵌入 → 性能下降2.1%
- 移除图像位置编码 → 性能下降3.5%
- 使用随机图像特征 → 性能大幅下降（说明图像特征是关键）

---

## 11. 常见问题与易错点

### Q1: VisualBERT和BERT的区别？

VisualBERT就是BERT加上了图像区域作为额外的输入token。核心架构与BERT完全相同，只是输入多了一种模态。

### Q2: 单塔架构中图文交互是如何发生的？

通过自注意力机制。文本token的query可以关注图像token的key/value，图像token的query也可以关注文本token的key/value。这种交互在每一层Transformer中持续发生。

### Q3: 为什么VisualBERT比双塔架构更简单？

双塔架构需要设计交叉注意力层来让两个塔交互。单塔架构只需要标准的自注意力，因为所有token都在同一个序列中。

### Q4: 序列长度增加带来的问题？

图文拼接后序列长度增加（比如20+36=56），自注意力的计算复杂度是 O(L²)，所以序列越长计算量越大。这是单塔架构的主要缺点。

### Q5: VisualBERT如何区分文本和图像token？

通过片段嵌入（segment embedding）。文本token的segment_id=0，图像token的segment_id=1，这样模型可以区分两种模态。

---

## 12. 学习总结

### 核心知识点

1. **VisualBERT = BERT + 图像区域作为额外token**
2. **单塔架构**：图文拼接后统一编码
3. **自注意力实现跨模态交互**：不需要专门的交叉注意力
4. **两大预训练目标**：MLM + ITM

### 架构速记

VisualBERT = BERT输入中增加图像区域token + 片段嵌入

### 关键洞见

最简单的架构往往是最优雅的。VisualBERT证明了只要输入表示设计得当，标准的Transformer自注意力可以完美处理多模态交互。

---

## 13. 练习题与思考题（含答案）

### 习题1：序列长度

**问题**：如果文本20个token，图像36个区域，VisualBERT的自注意力复杂度是多少？

**答案**：$(20+36)^2 = 56^2 = 3136$。复杂度是序列长度的平方。

### 习题2：模态区分

**问题**：如果VisualBERT不使用片段嵌入，模型还能区分文本和图像token吗？

**答案**：可以部分区分。图像特征的分布和文本特征的分布不同（图像来自线性投影，文本来自embedding），模型可以从数值分布上区分。但片段嵌入提供了明确的模态指示信号。

### 习题3：单塔 vs 双塔

**问题**：单塔架构（VisualBERT）相比双塔架构（ViLBERT）的优势是什么？

**答案**：单塔更简单，不需要交叉注意力，参数共享，容易扩展到更多模态。双塔可以独立使用各领域的预训练模型（如BERT、ResNet），各模态有更独立的编码空间。

### 习题4：图像顺序

**问题**：图像区域的顺序对VisualBERT重要吗？

**答案**：重要。图像区域的位置编码告诉模型每个区域的空间位置。如果区域顺序打乱但没有位置编码，模型无法知道哪个区域在图像的哪个位置。

### 习题5：思考题

**问题**：如果向VisualBERT的输入中加入音频特征作为第三种模态，应该怎么做？

**答案**：在输入中增加音频特征，加上对应的片段嵌入（segment_id=2），然后拼接在序列末尾。由于自注意力机制，新模态会自动与其他模态交互。这就是单塔架构的扩展优势。

---

## 14. 学习路径建议

### 前置知识
- BERT / Transformer
- 自注意力机制
- Faster R-CNN（目标检测）
- 预训练-微调范式

### 平行模型
- **ViLBERT**：双塔+协同注意力（更复杂但更灵活）
- **UNITER**：改进的单塔架构（统一Transformer）
- **VL-BERT**：类似VisualBERT的单塔模型

### 进阶方向
- **OSCAR**：使用检测标签作为锚点改进单塔
- **ViLT**：去掉CNN和RPN的极简单塔
- **ALBEF**：对比学习+单塔融合的混合架构

### 学习顺序建议

```
① BERT → ② 目标检测 → ③ VisualBERT（单塔入门） → ④ ViLBERT（双塔对比） → ⑤ 进阶VL模型
```
