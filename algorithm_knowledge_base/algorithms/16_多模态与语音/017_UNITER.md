# UNITER 学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
UNITER（UNiversal Image-TExt Representation）是微软于2019年提出的通用视觉语言预训练模型，通过四种预训练任务联合学习跨模态表示，成为"图文版BERT"的代表作之一。

### 1.2 直觉类比
UNITER就像一个"全能实习生"——它同时做四项训练：1）完形填空（预测被遮住的词）；2）拼图还原（预测被遮住的图像区域）；3）配对检查（图文是否匹配）；4）翻译对照（确认词汇对应哪个图像区域）。四项技能互相促进，让它的图文理解能力全面提升。

### 1.3 历史背景
- **2019年9月**：微软提出UNITER
- **2020年**：发布Base和Large两个版本
- **2021年**：成为VLP任务的重要基线模型

### 1.4 算法定位
UNITER属于**视觉语言预训练模型**，使用单流Transformer架构和多任务学习，覆盖VQA、图像描述、图文检索等多种任务。

---

## 2. 核心原理

### 2.1 单流Transformer架构
UNITER使用单流编码器同时处理文本和图像：
- 文本：BERT WordPiece token
- 图像：Faster R-CNN区域特征 + 位置编码
- 输入：[CLS] + 文本 + [SEP] + 图像区域

### 2.2 四种预训练任务

**1) 掩码语言建模（MLM）**
- 随机mask 15%的文本token
- 利用图像区域和未mask的文本预测被mask的词

**2) 掩码图像建模（MIM）**
三种具体策略：
- **MRFR**（Masked Region Feature Regression）：回归被mask区域的特征
- **MRC**（Masked Region Classification）：分类被mask区域的标签（object class）
- **MRC-kl**：使用KL散度匹配检测器的软标签分布

**3) 图文匹配（ITM）**
- 二分类任务，判断图文是否匹配
- 使用[CLS] token的表示分类

**4) 词汇-区域对齐（WRA）**
- 基于最优运输（Optimal Transport）的细粒度对齐
- 计算文本词和图像区域之间的对齐成本

### 2.3 条件掩码策略
UNITER采用条件掩码：mask文本时保留全部图像，mask图像时保留全部文本。这迫使模型在跨模态上下文中进行预测。

---

## 3. 数学公式与推导

### 3.1 MLM损失
$$L_{MLM} = -\mathbb{E}_{(w,v)\sim D} \sum_{i \in \mathcal{M}_w} \log P(w_i | w_{\backslash \mathcal{M}_w}, v)$$

其中 $\mathcal{M}_w$ 是mask的文本位置，$v$ 是图像区域特征。

### 3.2 MIM损失（MRFR）
$$L_{MRFR} = \mathbb{E}_{(w,v)\sim D} \sum_{j \in \mathcal{M}_v} \text{MSE}(r_j, \hat{r}_j)$$

其中 $r_j$ 是原始区域特征，$\hat{r}_j$ 是预测的特征，$\mathcal{M}_v$ 是mask的图像区域。

### 3.3 MIM损失（MRC-kl）
$$L_{MRC-kl} = \mathbb{E}_{(w,v)\sim D} \sum_{j \in \mathcal{M}_v} D_{KL}(c_j \parallel \hat{c}_j)$$

其中 $c_j$ 是检测器输出的类别分布，$\hat{c}_j$ 是模型预测的类别分布。

### 3.4 ITM损失
$$L_{ITM} = -\mathbb{E}_{(w,v)\sim D} \left[ y \log \hat{y} + (1-y) \log(1-\hat{y}) \right]$$

其中 $y \in \{0,1\}$ 是图文是否匹配的标签。

### 3.5 最优运输对齐（WRA）
给定文本词特征 $T = \{t_1, ..., t_n\}$ 和图像区域特征 $V = \{v_1, ..., v_m\}$，WRA计算运输矩阵 $P \in \mathbb{R}^{n \times m}$：

$$\min_{P} \langle P, C \rangle + \frac{1}{\lambda} \sum_{i,j} P_{ij} \log P_{ij}$$

s.t. $P\mathbf{1}_m = \mu, P^T\mathbf{1}_n = \nu$

其中 $C_{ij} = 1 - \cos(t_i, v_j)$ 是对齐成本，$\mu, \nu$ 是边际分布。

### 3.6 总损失
$$L_{UNITER} = L_{MLM} + \alpha L_{MIM} + \beta L_{ITM} + \gamma L_{WRA}$$

---

## 4. 训练过程讲解

### 4.1 数据预处理
1. 使用Faster R-CNN从图像中提取36个区域特征和类别分布
2. 使用BERT tokenizer对文本进行分词
3. 生成负样本（图文不匹配的对）
4. 随机mask 15%的文本token和15%的图像区域

### 4.2 预训练步骤
1. 从数据集中采样一个batch的图文对（含负样本）
2. 前向传播计算四种损失
3. 反向传播更新UNITER参数
4. 每隔一定步数评估验证集

### 4.3 两阶段训练
- **第一阶段**：仅在MS-COCO上训练（120K图片，每图5描述）
- **第二阶段**：加入Visual Genome、Conceptual Captions等更大数据

---

## 5. 应用场景

| 场景 | 任务 | 输入 | 输出 |
|------|------|------|------|
| 视觉问答 | VQA | 图像+问题 | 答案 |
| 图像描述 | Captioning | 图像 | 文本描述 |
| 图文检索 | Retrieval | 文本/图像 | 匹配的图像/文本 |
| 指代理解 | Referring Expression | 图像+文本 | 目标区域 |

---

## 6. 优缺点分析

### 优点
1. **多任务学习**：四种任务互补，学习到丰富的表示
2. **细粒度对齐**：最优运输提供精确的词汇-区域对应
3. **通用性强**：一个模型适配多种下游任务

### 缺点
1. **训练复杂度高**：四种任务联合训练计算量大
2. **依赖检测器**：图像特征质量影响大
3. **最优运输计算慢**：Sinkhorn迭代增加训练时间

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class UNITERModel(nn.Module):
    """UNITER视觉语言预训练模型"""
    def __init__(self, hidden_dim=768, num_labels=30522, num_obj_classes=1600):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 基于BERT的文本编码器
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # 图像特征投影
        self.img_fc = nn.Linear(2048, hidden_dim)
        self.img_ln = nn.LayerNorm(hidden_dim)
        
        # 图像位置编码
        self.loc_fc = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # MLM预测头
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_labels)
        )
        
        # MIM预测头（MRFR）
        self.mim_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2048)  # 回归原始特征
        )
        
        # MIM分类头（MRC）
        self.mrc_head = nn.Linear(hidden_dim, num_obj_classes)
        
        # ITM分类头
        self.itm_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, input_ids, attention_mask, img_features, img_locs, 
                img_labels=None, task='mlm'):
        """
        前向传播
        Args:
            input_ids: 文本token [B, L]
            attention_mask: 文本mask [B, L]
            img_features: 图像区域特征 [B, N, 2048]
            img_locs: 位置 [B, N, 7]
            img_labels: 区域标签 [B, N] (用于MRC)
            task: 任务类型
        """
        B, N = img_features.shape[:2]
        
        # 文本编码
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_emb = text_out.last_hidden_state
        
        # 图像编码
        img_feat = self.img_ln(self.img_fc(img_features))
        img_loc = self.loc_fc(img_locs)
        img_emb = img_feat + img_loc
        
        # 拼接文本和图像
        combined = self._merge_sequence(text_emb, attention_mask, img_emb)
        
        # 根据任务输出
        text_len = input_ids.shape[1]
        
        if task == 'mlm':
            return self.mlm_head(combined[:, 1:text_len-1, :])
        elif task == 'mim':
            img_start = text_len + 1  # + [SEP]
            img_out = combined[:, img_start+1:img_start+1+N, :]
            pred_feats = self.mim_head(img_out)
            if img_labels is not None:
                pred_cls = self.mrc_head(img_out)
                return pred_feats, pred_cls
            return pred_feats
        elif task == 'itm':
            cls_repr = combined[:, 0, :]
            return self.itm_head(cls_repr)
            
    def _merge_sequence(self, text_emb, attention_mask, img_emb):
        """拼接文本和图像序列"""
        B = text_emb.shape[0]
        sep = text_emb[:, -1:, :]
        # [CLS] + text + [SEP] + image + [SEP]
        combined = torch.cat([
            text_emb[:, :1],   # [CLS]
            text_emb[:, 1:-1], # text tokens (excluding [CLS] and [SEP])
            sep,               # [SEP]
            img_emb,           # image regions
            sep,               # [SEP]
        ], dim=1)
        return combined
    
    def compute_ot_loss(self, text_feats, img_feats, reg_lambda=0.05, num_iters=10):
        """
        计算最优运输对齐损失（WRA）
        Args:
            text_feats: [B, text_len, D]
            img_feats: [B, num_regions, D]
        """
        B, T, D = text_feats.shape
        N = img_feats.shape[1]
        
        # 标准化
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        img_feats = F.normalize(img_feats, p=2, dim=-1)
        
        # 成本矩阵: 1 - cosine similarity
        cost = 1.0 - torch.bmm(text_feats, img_feats.transpose(1, 2))
        
        # Sinkhorn算法计算最优运输
        mu = torch.ones(B, T, device=text_feats.device) / T
        nu = torch.ones(B, N, device=text_feats.device) / N
        
        # 初始化
        K = torch.exp(-cost / reg_lambda)
        a = torch.ones(B, N, device=text_feats.device)
        
        for _ in range(num_iters):
            b = nu / (a.unsqueeze(1) * K + 1e-8).transpose(1, 2).sum(dim=-1)
            a = mu / (b.unsqueeze(1) * K + 1e-8).sum(dim=-1)
        
        T_matrix = a.unsqueeze(2) * K * b.unsqueeze(1)
        
        # 运输损失 = <T, C>
        ot_loss = (T_matrix * cost).sum(dim=(1, 2)).mean()
        return ot_loss


def test_uniter():
    """测试UNITER模型"""
    model = UNITERModel()
    
    B, L, N = 2, 20, 36
    input_ids = torch.randint(0, 100, (B, L))
    attention_mask = torch.ones(B, L)
    img_features = torch.randn(B, N, 2048)
    img_locs = torch.randn(B, N, 7)
    
    # MLM
    mlm_out = model(input_ids, attention_mask, img_features, img_locs, task='mlm')
    print(f"MLM输出: {mlm_out.shape}")
    
    # ITM
    itm_out = model(input_ids, attention_mask, img_features, img_locs, task='itm')
    print(f"ITM输出: {itm_out.shape}")
    
    # MIM
    mim_out = model(input_ids, attention_mask, img_features, img_locs, task='mim')
    print(f"MIM输出: {mim_out.shape}")
    
    print("UNITER测试通过！")

if __name__ == "__main__":
    test_uniter()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandwrittenUNITER(nn.Module):
    """UNITER核心逻辑手工实现"""
    def __init__(self, vocab_size=30000, d_model=768, nhead=12, num_layers=12):
        super().__init__()
        self.d_model = d_model
        
        # 文本嵌入
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        
        # 图像嵌入
        self.img_fc = nn.Linear(2048, d_model)
        self.loc_fc = nn.Linear(7, d_model)
        
        # Transformer
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # 任务头
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.mim_head = nn.Linear(d_model, 2048)
        self.mrc_head = nn.Linear(d_model, 1600)
        self.itm_head = nn.Linear(d_model, 2)
        
    def forward(self, text_ids, img_feats, img_locs, task='mlm'):
        B, T = text_ids.shape
        N = img_feats.shape[1]
        
        # 文本嵌入
        text_emb = self.word_emb(text_ids) + self.pos_emb(
            torch.arange(T, device=text_ids.device).unsqueeze(0)
        )
        
        # 图像嵌入
        img_emb = self.img_fc(img_feats) + self.loc_fc(img_locs)
        
        # 拼接
        seq = torch.cat([text_emb, img_emb], dim=1)
        
        # 编码
        for layer in self.encoder_layers:
            seq = layer(seq)
        seq = self.norm(seq)
        
        # 任务输出
        if task == 'mlm':
            return self.mlm_head(seq[:, :T, :])
        elif task == 'mim':
            img_out = seq[:, T:, :]
            return self.mim_head(img_out), self.mrc_head(img_out)
        elif task == 'itm':
            return self.itm_head(seq[:, 0, :])


def test_handwritten():
    model = HandwrittenUNITER()
    B, T, N = 2, 16, 10
    text_ids = torch.randint(0, 3000, (B, T))
    img_feats = torch.randn(B, N, 2048)
    img_locs = torch.randn(B, N, 7)
    
    out = model(text_ids, img_feats, img_locs, 'mlm')
    print(f"手工UNITER MLM输出: {out.shape}")

if __name__ == "__main__":
    test_handwritten()
```

---

## 9. 可视化与结果理解

### 9.1 注意力可视化
UNITER的跨模态注意力显示：文本词对其语义对应的图像区域有更高的注意力权重。例如"dog"对包含狗的图像区域。

### 9.2 最优运输对齐可视化
WRA学习到的运输矩阵显示文本词和图像区域的对应关系：
- 每一行是一个文本词，每一列是一个图像区域
- 高权重表示该词与该区域对齐

---

## 10. 模型评估

| 任务 | 指标 | UNITER-Base | UNITER-Large |
|------|------|-------------|--------------|
| VQA 2.0 | test-dev acc | 72.70 | 73.82 |
| NLVR² | test acc | 77.18 | 79.12 |
| IR (COCO) | R@1 | 51.85 | 54.70 |
| TR (COCO) | R@1 | 64.00 | 66.10 |

---

## 11. 常见问题与易错点

### Q1: UNITER的四种预训练任务是否都需要？
A: 实验表明MLM和ITM贡献最大，MIM和WRA提供额外提升。但在计算资源受限时，可以去掉WRA（计算最慢）。

### Q2: UNITER和OSCAR有何区别？
A: OSCAR引入对象标签作为显式锚点，UNITER通过WRA隐式学习对齐。UNITER的预训练任务更多，但OSCAR对齐更直接。

### Q3: 最优运输计算为什么慢？
A: Sinkhorn迭代需要矩阵指数运算，每次迭代 $O(N^2)$，且需要多次迭代（通常10-20次）才能收敛。

---

## 12. 学习总结

UNITER通过**四种预训练任务**（MLM + MIM + ITM + WRA）学习通用视觉语言表示，证明了多任务学习在VLP中的有效性。其**最优运输对齐**方法为细粒度跨模态理解提供了新思路。

---

## 13. 练习题与思考题（含答案）

### 习题1：UNITER为什么使用条件掩码策略？
**答案**：mask文本时保留图像、mask图像时保留文本，迫使模型利用跨模态信息进行预测，学习到真正的图文联合表示。

### 习题2：Sinkhorn算法的迭代次数对WRA有什么影响？
**答案**：迭代太少不收敛，运输矩阵不准确；迭代太多过拟合噪声分布。通常10-20次效果较好。

### 习题3：编程实现MIM的MRFR损失。
**答案**：
```python
def mrfr_loss(pred_features, target_features, masked_positions):
    loss = F.mse_loss(pred_features[masked_positions], target_features[masked_positions])
    return loss
```

### 习题4：思考题：如果去掉图像特征，UNITER退化成什么模型？
**答案**：去掉图像特征后只剩MLM，退化为BERT模型。

---

## 14. 学习路径建议

### 前置
- BERT、Transformer
- Faster R-CNN
- 最优运输理论

### 进阶
- Oscar、VinVL、ALBEF
- VisualBERT、LXMERT
