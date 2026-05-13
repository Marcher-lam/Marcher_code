# SCAN 学习文档

> 基于交叉注意力的图像-文本相似度模型（Stacked Cross Attention），通过细粒度区域-词汇对齐实现精确的图文匹配。

## 1. 算法基础认知

### 一句话定义

SCAN（Stacked Cross Attention Network）是用于图像-文本相似度计算的模型，通过堆叠的交叉注意力机制实现图像区域与文本词汇之间的细粒度对齐。

### 直觉类比

SCAN的工作方式类似于拼图游戏：
- 把图像拆成若干区域（如猫头、猫身、沙发）
- 把文本拆成若干词汇（"猫"、"沙发"、"在...上"）
- 然后找出哪些词汇描述哪些区域（"猫"→猫区域，"沙发"→沙发区域）
- 最后计算整体的匹配程度

### 历史背景

- **2018年**：SCAN由Kuang-Huei Lee等人在CVPR 2018发表
- **核心创新**：堆叠交叉注意力实现细粒度区域-词汇对齐
- **与AttnGAN的关系**：SCAN的注意力机制后来被AttnGAN的DAMSM模块借鉴

### 算法定位

SCAN是**图文匹配模型**，属于跨模态检索和理解任务。核心是计算图像和文本之间的细粒度相似度。

---

## 2. 核心原理

### 双流架构

SCAN采用双流架构分别提取图像和文本特征：

```
图像 → Faster R-CNN → 区域特征 (v1, v2, ..., vk)
文本 → Bi-LSTM → 词特征 (w1, w2, ..., wn)
                                      ↓
                             堆叠交叉注意力
                                      ↓
                             细粒度相似度得分
```

### TI-SCA 和 IT-SCA

SCAN的核心是两种交叉注意力：

1. **TI-SCA（Text-to-Image Stacked Cross Attention）**：文本到图像的注意力
   - 每个词关注图像中的所有区域
   - 计算每个词与加权图像上下文的相似度

2. **IT-SCA（Image-to-Text Stacked Cross Attention）**：图像到文本的注意力
   - 每个区域关注文本中的所有词
   - 计算每个区域与加权文本上下文的相似度

### 注意力计算

对于文本中的第 $i$ 个词 $w_i$，关注所有图像区域 $V = \{v_1, ..., v_k\}$：

$$\alpha_{ij} = \text{Softmax}(w_i^T v_j)$$

加权图像上下文：

$$c_i = \sum_{j=1}^{k} \alpha_{ij} v_j$$

词 $i$ 与图像的整体相似度：

$$s_i = \cos(w_i, c_i) = \frac{w_i^T c_i}{\|w_i\|\|c_i\|}$$

---

## 3. 数学公式与推导

### 3.1 图像-文本相似度

图像-文本对的相似度是两个方向相似度的平均：

$$S(I, T) = \frac{1}{2} (S_{t2i}(I, T) + S_{i2t}(I, T))$$

### 3.2 文本到图像相似度（TI-SCA）

$$S_{t2i}(I, T) = \frac{1}{n} \sum_{i=1}^{n} \max_{j} (w_i^T v_j)$$

或者使用注意力加权版本：

$$S_{t2i}(I, T) = \frac{1}{n} \sum_{i=1}^{n} \frac{w_i^T c_i}{\|w_i\|\|c_i\|}$$

其中 $c_i = \sum_{j} \alpha_{ij} v_j$ 是注意力加权的图像上下文。

### 3.3 图像到文本相似度（IT-SCA）

$$S_{i2t}(I, T) = \frac{1}{k} \sum_{j=1}^{k} \max_{i} (v_j^T w_i)$$

注意力加权版本：

$$S_{i2t}(I, T) = \frac{1}{k} \sum_{j=1}^{k} \frac{v_j^T c_j'}{\|v_j\|\|c_j'\|}$$

其中 $c_j' = \sum_{i} \beta_{ji} w_i$ 是注意力加权的文本上下文。

### 3.4 排名损失（Triplet Loss）

SCAN使用triplet loss进行训练：

$$\mathcal{L} = \sum_{(I,T)} [\max(0, \alpha - S(I,T) + S(I,T')) + \max(0, \alpha - S(I,T) + S(I', T))]$$

其中 $T'$ 是 $I$ 的不匹配文本（负样本），$I'$ 是 $T$ 的不匹配图像，$\alpha$ 是margin。

### 3.5 难负样本挖掘

SCAN在一个batch内挖掘难负样本（hard negatives）：

在batch中，对于每个正样本对 $(I_i, T_i)$：
- 图像到文本的难负样本：$T_{hard} = \arg\max_{j \neq i} S(I_i, T_j)$
- 文本到图像的难负样本：$I_{hard} = \arg\max_{j \neq i} S(I_j, T_i)$

---

## 4. 训练过程讲解

### 阶段一：特征提取

1. **图像**：Faster R-CNN提取36个区域特征，每个2048维
2. **文本**：Bi-LSTM编码每个单词，得到词级别的特征序列

### 阶段二：特征投影

将图像和文本特征投影到共同的语义空间（通常512维）。

### 阶段三：交叉注意力匹配

1. 计算图像区域和文本词汇的相似度矩阵
2. 通过注意力机制得到加权的上下文向量
3. 计算双向的相似度得分

### 阶段四：损失计算

- 使用triplet loss + 难负样本挖掘
- 拉近匹配图文对的距离
- 推远不匹配图文对的距离

### 训练技巧

- **预训练Faster R-CNN**：在Visual Genome上预训练
- **正则化**：特征L2归一化后计算相似度
- **梯度裁剪**：防止梯度爆炸

---

## 5. 应用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| 图文检索 | 用文字搜索图像 / 用图像搜索文字 | 搜索"白猫"找到相关图片 |
| 图像描述评估 | 计算生成描述与真实描述的匹配度 | 评估image captioning模型 |
| 跨模态验证 | 验证图文是否匹配 | 检测图文不一致 |
| 零样本分类 | 通过文本描述进行分类 | 用类别描述作query |
| 多模态搜索 | 跨模态搜索引擎 | 用草图搜索真实图像 |

---

## 6. 优缺点分析

### 优点

1. **细粒度对齐**：区域-词汇级别的匹配，比全局匹配更精确
2. **双向注意力**：TI-SCA和IT-SCA互为补充
3. **难负样本挖掘**：训练更有效
4. **可解释性**：注意力权重可视化显示图文对应关系

### 缺点

1. **计算复杂度高**：需要计算所有区域-词汇对的相似度
2. **依赖检测质量**：Faster R-CNN的区域特征质量影响大
3. **无跨模态融合**：只是计算相似度，不做深度融合
4. **无预训练**：SCAN是从头训练，没有利用大规模预训练

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """图像编码器: 使用预训练Faster R-CNN（模拟）"""
    def __init__(self, input_dim=2048, hidden_dim=512):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        # x: (B, N_regions, input_dim)
        x = self.fc(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        return F.normalize(x, dim=2)

class TextEncoder(nn.Module):
    """文本编码器: Bi-LSTM编码词特征"""
    def __init__(self, vocab_size=10000, embed_dim=300, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, 2, 
                           batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        # x: (B, seq_len)
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)  # (B, seq_len, hidden_dim)
        # outputs是每个位置的词特征
        outputs = self.bn(outputs.transpose(1, 2)).transpose(1, 2)
        return F.normalize(outputs, dim=2)

class CrossAttentionScorer(nn.Module):
    """
    交叉注意力相似度计算
    包含TI-SCA和IT-SCA
    """
    def __init__(self, lambda_=0.5):
        super().__init__()
        self.lambda_ = lambda_  # 平衡两个方向的权重
        
    def forward(self, img_feat, txt_feat):
        """
        Args:
            img_feat: (B, N_v, D) 归一化的图像区域特征
            txt_feat: (B, N_t, D) 归一化的文本词特征
        Returns:
            similarity: (B,) 批次中每个样本的相似度
        """
        # 相似度矩阵 (B, N_t, N_v)
        sim_matrix = torch.bmm(txt_feat, img_feat.transpose(1, 2))
        
        # TI-SCA: 文本到图像
        # 对每个词，找出最相关的图像区域
        t2i_sim, _ = sim_matrix.max(dim=2)  # (B, N_t)
        # 或者使用注意力加权版本
        t2i_attn = F.softmax(sim_matrix, dim=2)  # (B, N_t, N_v)
        t2i_context = torch.bmm(t2i_attn, img_feat)  # (B, N_t, D)
        t2i_sim_attn = (txt_feat * t2i_context).sum(dim=2)  # (B, N_t)
        
        # 汇总所有词
        t2i_score = t2i_sim_attn.mean(dim=1)  # (B,)
        
        # IT-SCA: 图像到文本
        i2t_sim, _ = sim_matrix.max(dim=1)  # (B, N_v)
        i2t_attn = F.softmax(sim_matrix.transpose(1, 2), dim=2)  # (B, N_v, N_t)
        i2t_context = torch.bmm(i2t_attn, txt_feat)  # (B, N_v, D)
        i2t_sim_attn = (img_feat * i2t_context).sum(dim=2)  # (B, N_v)
        
        # 汇总所有区域
        i2t_score = i2t_sim_attn.mean(dim=1)  # (B,)
        
        # 综合得分
        score = self.lambda_ * t2i_score + (1 - self.lambda_) * i2t_score
        
        return score, t2i_score, i2t_score

class SCAN(nn.Module):
    """SCAN模型"""
    def __init__(self, vocab_size=10000, img_dim=2048, 
                 embed_dim=300, hidden_dim=512):
        super().__init__()
        self.img_encoder = ImageEncoder(img_dim, hidden_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, hidden_dim)
        self.scorer = CrossAttentionScorer()
        
    def forward(self, img_feat, text_ids):
        """
        Args:
            img_feat: (B, N_v, 2048) 图像区域特征
            text_ids: (B, N_t) 文本token
        Returns:
            score: (B,) 相似度得分
            t2i_score: (B,) 文本到图像得分
            i2t_score: (B,) 图像到文本得分
        """
        img_emb = self.img_encoder(img_feat)
        txt_emb = self.text_encoder(text_ids)
        return self.scorer(img_emb, txt_emb)

class SCANWithLoss(nn.Module):
    """SCAN + 排名损失"""
    def __init__(self, vocab_size=10000, img_dim=2048, 
                 embed_dim=300, hidden_dim=512, margin=0.2):
        super().__init__()
        self.scan = SCAN(vocab_size, img_dim, embed_dim, hidden_dim)
        self.margin = margin
        
    def triplet_loss(self, scores, labels):
        """
        排名损失 + 难负样本挖掘
        scores: (B, B) 相似度矩阵
        labels: (B,) 正样本索引（对角线）
        """
        B = scores.shape[0]
        
        # 正样本得分
        pos_scores = scores.diag()  # (B,)
        
        # 难负样本挖掘
        # 对每个图像，找出得分最高的不匹配文本
        mask = torch.eye(B, dtype=torch.bool, device=scores.device)
        neg_scores_i2t = scores.masked_fill(mask, float('-inf')).max(dim=1)[0]
        # 对每个文本，找出得分最高的不匹配图像
        neg_scores_t2i = scores.masked_fill(mask, float('-inf')).max(dim=0)[0]
        
        # Triplet loss
        loss_i2t = F.relu(self.margin + neg_scores_i2t - pos_scores).mean()
        loss_t2i = F.relu(self.margin + neg_scores_t2i - pos_scores).mean()
        
        return (loss_i2t + loss_t2i) / 2
    
    def forward(self, img_feat, text_ids, return_scores=False):
        B = img_feat.shape[0]
        
        # 编码所有图像和文本
        # (B, N_v, D) and (B, N_t, D)
        img_emb = self.scan.img_encoder(img_feat)
        txt_emb = self.scan.text_encoder(text_ids)
        
        # 计算所有对之间的相似度
        scores = torch.zeros(B, B, device=img_feat.device)
        t2i_scores = torch.zeros(B, B, device=img_feat.device)
        i2t_scores = torch.zeros(B, B, device=img_feat.device)
        
        for i in range(B):
            for j in range(B):
                score, t2i, i2t = self.scan.scorer(
                    img_emb[i:i+1], txt_emb[j:j+1]
                )
                scores[i, j] = score
                t2i_scores[i, j] = t2i
                i2t_scores[i, j] = i2t
        
        # 计算损失
        loss = self.triplet_loss(scores, torch.arange(B))
        
        if return_scores:
            return loss, scores
        return loss

# 使用示例
if __name__ == "__main__":
    model = SCAN()
    
    # 模拟输入
    B, N_v, N_t = 4, 36, 20
    img_feat = torch.randn(B, N_v, 2048)
    text_ids = torch.randint(0, 10000, (B, N_t))
    
    score, t2i, i2t = model(img_feat, text_ids)
    
    print(f"相似度得分: {score}")
    print(f"文本→图像得分: {t2i}")
    print(f"图像→文本得分: {i2t}")
    print("SCAN前向传播成功!")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HandcraftCrossAttention(nn.Module):
    """
    手工交叉注意力相似度计算
    实现TI-SCA和IT-SCA的核心逻辑
    """
    def __init__(self, use_max=False):
        super().__init__()
        self.use_max = use_max
        
    def compute_similarity(self, feat_a, feat_b):
        """
        计算两个特征序列之间的细粒度相似度
        feat_a: (B, N_a, D)
        feat_b: (B, N_b, D)
        """
        # 相似度矩阵 (B, N_a, N_b)
        sim = torch.bmm(feat_a, feat_b.transpose(1, 2))
        
        if self.use_max:
            # 使用max聚合：每个a元素找最佳匹配的b
            sim_ab = sim.max(dim=2)[0]  # (B, N_a)
        else:
            # 使用注意力加权聚合
            attn = F.softmax(sim, dim=2)  # (B, N_a, N_b)
            context = torch.bmm(attn, feat_b)  # (B, N_a, D)
            sim_ab = (feat_a * context).sum(dim=2)  # (B, N_a)
        
        return sim_ab.mean(dim=1)  # (B,)

class HandcraftSCAN(nn.Module):
    """
    手工实现的SCAN核心
    无依赖的纯PyTorch实现
    """
    def __init__(self, img_dim=2048, text_dim=300, hidden_dim=512, vocab_size=10000):
        super().__init__()
        
        # 图像投影
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 文本编码
        self.word_embedding = nn.Embedding(vocab_size, text_dim)
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 交叉注意力
        self.t2i_attn = HandcraftCrossAttention()
        self.i2t_attn = HandcraftCrossAttention()
        
    def forward(self, img_regions, text_ids):
        """
        手工SCAN前向传播
        Args:
            img_regions: (B, N_v, 2048) 图像区域
            text_ids: (B, N_t) 文本token IDs
        Returns:
            t2i_score: 文本到图像相似度
            i2t_score: 图像到文本相似度
            avg_score: 平均相似度
        """
        # 图像编码 + L2归一化
        img_feat = self.img_proj(img_regions)
        img_feat = F.normalize(img_feat, dim=2)
        
        # 文本编码 + L2归一化
        word_emb = self.word_embedding(text_ids)
        txt_feat = self.text_fc(word_emb)
        txt_feat = F.normalize(txt_feat, dim=2)
        
        # 双向交叉注意力
        t2i_score = self.t2i_attn.compute_similarity(txt_feat, img_feat)
        i2t_score = self.i2t_attn.compute_similarity(img_feat, txt_feat)
        
        avg_score = (t2i_score + i2t_score) / 2
        
        return t2i_score, i2t_score, avg_score

# 测试手工实现
if __name__ == "__main__":
    model = HandcraftSCAN()
    
    img_regions = torch.randn(4, 36, 2048)
    text_ids = torch.randint(0, 10000, (4, 20))
    
    t2i, i2t, avg = model(img_regions, text_ids)
    
    print(f"文本→图像: {t2i}")
    print(f"图像→文本: {i2t}")
    print(f"平均相似度: {avg}")
    print("手工SCAN测试通过!")
```

---

## 9. 可视化与结果理解

### 9.1 注意力可视化

SCAN的交叉注意力权重可以被可视化，展示图文对应：
- 文本"dog"的区域和图像中狗的区域有高注意力
- 文本"grass"的区域和图像中草的区域有高注意力
- 这种对应关系证明了SCAN学到了细粒度的语义对齐

### 9.2 相似度矩阵

相似度矩阵是对角线主导的，表明匹配的图文对得分高，不匹配的得分低。

### 9.3 检索结果示例

文本query "a white cat on a red couch" 的检索结果：
1. [图] 白猫在红色沙发上 (得分: 0.87) ← 正确
2. [图] 白猫在地上 (得分: 0.65)
3. [图] 橘猫在沙发上 (得分: 0.58)

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| R@1 (Recall@1) | 检索结果中排名第一的准确率 |
| R@5 (Recall@5) | 前5个结果中正确的比例 |
| R@10 (Recall@10) | 前10个结果中正确的比例 |
| Med r (Median Rank) | 正确结果的中位数排名 |

### 10.2 COCO数据集上的表现

| 方向 | R@1 | R@5 | R@10 | Med r |
|------|-----|-----|------|-------|
| 图像→文本 | 56.4% | 85.3% | 93.2% | 1 |
| 文本→图像 | 45.5% | 78.6% | 88.5% | 2 |

### 10.3 评估代码

```python
def evaluate_retrieval(model, dataloader, device='cuda'):
    """评估图文检索性能"""
    model.eval()
    all_img_feats = []
    all_txt_feats = []
    
    with torch.no_grad():
        for img_feat, text_ids in dataloader:
            img_feat = img_feat.to(device)
            text_ids = text_ids.to(device)
            
            img_emb = model.scan.img_encoder(img_feat)
            txt_emb = model.scan.text_encoder(text_ids)
            
            all_img_feats.append(img_emb.mean(dim=1))  # 全局特征
            all_txt_feats.append(txt_emb.mean(dim=1))
    
    img_feats = torch.cat(all_img_feats)
    txt_feats = torch.cat(all_txt_feats)
    img_feats = F.normalize(img_feats, dim=1)
    txt_feats = F.normalize(txt_feats, dim=1)
    
    # 文本检索图像
    sim_matrix = txt_feats @ img_feats.t()
    ranks = torch.argsort(sim_matrix, descending=True)
    
    N = sim_matrix.shape[0]
    r1 = (ranks == torch.arange(N).unsqueeze(1)).any(dim=1).float().mean()
    
    return {'R@1': r1.item()}
```

---

## 11. 常见问题与易错点

### Q1: SCAN和常规图文匹配的区别？

常规图文匹配使用全局特征（整个图像 vs 整个句子）计算相似度。SCAN使用细粒度的区域-词汇级别匹配，通过注意力机制找到最相关的局部对应。

### Q2: 为什么需要TI-SCA和IT-SCA两个方向？

TI-SCA回答"文本中的每个词在图像中是否有对应"；IT-SCA回答"图像中的每个区域在文本中是否有对应"。两者互补，覆盖了双向的细粒度匹配。

### Q3: max pooling和attention pooling的选择？

max pooling关注"最匹配"的那个区域/词汇，attention pooling关注"整体加权"的上下文。SCAN论文中两者效果相近，但attention pooling梯度更平滑。

### Q4: Triplet Loss的margin如何选择？

margin控制正负样本之间的距离。margin太大导致训练困难（损失一直很大），太小导致区分度不够。通常选0.1-0.3之间。

### Q5: SCAN和CLIP的区别？

SCAN是细粒度的区域-词汇匹配，CLIP是全局的图文对比学习。SCAN适合需要精确对应关系的任务（如指代表达），CLIP适合快速检索和零样本分类。

---

## 12. 学习总结

### 核心知识点

1. **SCAN = 图像编码器 + 文本编码器 + 堆叠交叉注意力**
2. **TI-SCA**：文本到图像的注意力，每个词关注图像区域
3. **IT-SCA**：图像到文本的注意力，每个区域关注文本词汇
4. **Triplet Loss**：拉近匹配对，推远不匹配对

### 关键洞见

SCAN证明了"细粒度匹配优于全局匹配"——在图文检索任务中，局部对应关系比全局相似度更可靠。

---

## 13. 练习题与思考题（含答案）

### 习题1：注意力计算

**问题**：假设图像有36个区域，文本有20个词，计算TI-SCA的相似度矩阵形状？

**答案**：相似度矩阵形状是(B, 20, 36)，其中B是batch size。

### 习题2：Triplet Loss

**问题**：Triplet Loss的公式 $\max(0, \alpha - S(I,T) + S(I,T'))$ 中，$\alpha$ 和 $S$ 分别代表什么？

**答案**：$\alpha$ 是margin（间隔），$S$ 是相似度得分。Loss希望正样本得分比负样本至少高 $\alpha$。

### 习题3：方向选择

**问题**：如果只使用TI-SCA（文本到图像），会有什么问题？

**答案**：只使用TI-SCA，模型只会优化"文本能匹配到图像"的方向，但不会优化"图像能匹配到文本"的方向。可能出现所有图像都匹配到同一个文本的情况。

### 习题4：特征归一化

**问题**：SCAN为什么需要对特征进行L2归一化？

**答案**：L2归一化将特征映射到单位超球面上，余弦相似度等价于内积。这让不同特征的尺度统一，训练更稳定。

### 习题5：思考题

**问题**：SCAN的交叉注意力与Transformer的交叉注意力有什么区别？

**答案**：SCAN的交叉注意力更简单：计算相似度→Softmax→加权求和，没有残差连接、LayerNorm等组件。Transfomer的交叉注意力是完整的attention layer，包含残差连接和FFN。

---

## 14. 学习路径建议

### 前置知识
- 注意力机制基础
- RNN / LSTM
- 图像特征提取
- Triplet Loss

### 平行模型
- **AttnGAN**：使用类似注意力的文本到图像生成
- **VSE++**：全局视觉语义嵌入
- **VSRN**：视觉语义推理网络

### 进阶方向
- **CLIP**：大规模对比学习的全局检索
- **ALIGN**：Google的大规模图文对比学习
- **BLIP**：检索+生成的统一模型

### 学习顺序建议

```
① 注意力机制 → ② 图文检索基础 → ③ SCAN（细粒度匹配） → ④ CLIP（大规模检索）
```
