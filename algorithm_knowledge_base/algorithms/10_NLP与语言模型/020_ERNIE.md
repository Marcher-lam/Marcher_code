# ERNIE（Enhanced Representation through Knowledge Integration）学习文档

> 百度提出的知识增强预训练模型，通过整合实体和短语级别的知识来学习更好的语言表示，在中文NLP任务上表现卓越。

## 1. 算法基础认知

### 一句话定义

ERNIE（Enhanced Representation through Knowledge Integration）是百度提出的知识增强预训练模型，通过在mask策略中融入实体和短语级别的先验知识，学习到更丰富的语义表示。

### 直觉类比

ERNIE就像一个在学习中文时不仅背单词，还理解"词语背后的知识"的学生。当BERT在学"北京是中国的首都"时，它只是随机mask单字如"北"让模型预测。而ERNIE会mask整个"北京"作为整体，让模型知道这是一个实体。它还会mask"首都"这样的短语，让模型理解概念关系。

### 历史背景

- **2019年3月**：百度发布ERNIE 1.0（NAACL 2019）
- **2019年7月**：ERNIE 2.0发布（AAAI 2020）
- **2020年**：ERNIE 3.0发布（参数量最大的中文预训练模型之一）
- **核心创新**：从"字级mask"升级为"知识级mask"

### 算法定位

ERNIE是**知识增强语言模型**，属于自编码预训练模型（类似BERT），但通过知识注入的方式改进了预训练策略。

---

## 2. 核心原理

### 知识增强的Mask策略

ERNIE最核心的创新是**知识级Mask**，分为三个层次：

```
BERT: 我 [MASK] 北 [MASK] 是 中 [MASK] 的 首都
ERNIE 1.0: 我 [MASK][MASK][MASK] 是 [MASK][MASK][MASK] 的 首都
                    ↑ 实体级mask           ↑ 短语级mask
```

#### 1. 基础级Mask（Basic-level Masking）
- 随机mask单个字/词（同BERT）

#### 2. 实体级Mask（Entity-level Masking）
- 识别命名实体（人名、地名、组织名等）
- mask整个实体中的所有字
- 迫使模型学习实体级别的语义

#### 3. 短语级Mask（Phrase-level Masking）
- 识别短语（名词短语、动词短语等）
- mask整个短语
- 迫使模型学习短语级别的语义

### 多任务学习

ERNIE 2.0引入持续多任务学习：

1. **词级别任务**：MLM（Masked Language Modeling）
2. **结构级别任务**：
   - 句子排序（判断句子顺序）
   - 句子距离（判断两个句子的距离）
3. **语义级别任务**：
   - NSP（Next Sentence Prediction）
   - 关系分类（实体之间的关系）
   - 指代消解（代词指代的对象）

### 持续预训练机制

ERNIE 2.0使用**持续预训练**（Continual Pretraining）：
- 多任务不是同时训练，而是持续累加
- 先训练基础任务
- 逐步加入新任务
- 旧任务不遗忘（通过多任务损失联合训练）

---

## 3. 数学公式与推导

### 3.1 实体级Mask

对于实体 $E = \{w_i, w_{i+1}, ..., w_j\}$，将其所有token替换为[MASK]：

$$P(E | context) = \prod_{k=i}^{j} P(w_k | w_{\backslash E}, context)$$

相比BERT的独立Mask，ERNIE的实体级Mask需要同时预测所有被Mask的token，迫使模型学习实体级别的语义。

### 3.2 多任务联合损失

$$\mathcal{L} = \sum_{t=1}^{T} w_t \cdot \mathcal{L}_t$$

其中 $T$ 是任务数量，$w_t$ 是第 $t$ 个任务的权重，$\mathcal{L}_t$ 是第 $t$ 个任务的损失。

### 3.3 ERNIE 2.0的持续预训练

持续学习策略：在训练的第 $k$ 阶段，使用前 $k$ 个任务的联合损失：

$$\mathcal{L}^{(k)} = \sum_{t=1}^{k} \lambda_t^{(k)} \mathcal{L}_t$$

其中 $\lambda_t^{(k)}$ 是第 $k$ 阶段中第 $t$ 个任务的权重。

### 3.4 句子排序任务

给定两个句子 $S_1$ 和 $S_2$，判断它们在原文中的顺序：

$$P(\text{before} | S_1, S_2) = \text{Sigmoid}(f([CLS; S_1; SEP; S_2]))$$

### 3.5 句子距离任务

给定 $S_1$ 和 $S_2$，判断它们在原文中的距离：
- 0：同一文档中的相邻句子
- 1：同一文档中的非相邻句子
- 2：不同文档中的句子

这是一个三分类任务：

$$P(d | S_1, S_2) = \text{Softmax}(f([CLS; S_1; SEP; S_2]))$$

---

## 4. 训练过程讲解

### 阶段一：知识标注

1. 使用命名实体识别（NER）工具标注文本中的实体
2. 使用分词和短语标注工具标注短语
3. 构建"知识词典"

### 阶段二：多粒度Mask

训练时，按照一定比例选择不同级别的Mask：
- 20% 基础级Mask（单字）
- 40% 实体级Mask（整个实体）
- 40% 短语级Mask（整个短语）

### 阶段三：多任务预训练

ERNIE 2.0采用持续预训练策略：
1. 阶段1：MLM + NSP
2. 阶段2：加入句子排序
3. 阶段3：加入句子距离
4. 阶段4：加入关系分类
5. 每个阶段在新任务上学习，同时保持旧任务能力

### 训练细节

- 优化器：Adam，学习率1e-4
- 训练数据：百度百科+新闻+对话
- 总参数量：110M（Base版）
- 训练设备：数十块GPU

---

## 5. 应用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| 中文文本分类 | 情感分析、新闻分类 | 判断评论情感 |
| 中文命名实体识别 | 识别人名、地名等 | "张三去了北京"→实体识别 |
| 关系抽取 | 实体关系判断 | "北京是中国的首都"→"首都"关系 |
| 阅读理解 | 中文机器阅读 | 根据文章回答问题 |
| 对话系统 | 中文对话理解 | 意图识别、槽位填充 |
| 搜索引擎 | 语义匹配 | Query-文档匹配 |

---

## 6. 优缺点分析

### 优点

1. **知识增强**：融入实体和短语知识，语义理解更深
2. **中文优势**：在中文NLP任务上显著优于BERT
3. **持续学习**：支持多任务增量训练，可不断扩展
4. **多任务丰富**：多个预训练目标提供丰富的学习信号
5. **百度生态**：与百度产品线深度集成

### 缺点

1. **知识依赖**：需要NER等工具标注知识，质量影响大
2. **训练复杂**：多任务持续训练流程复杂
3. **语言限制**：英文任务提升有限（英文的短语和实体级mask收益不如中文）
4. **开源版本**：早期开源版本有限（后续逐渐完善）
5. **领域偏移**：知识库来自百科领域，领域迁移可能有偏差

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class ERNIEForClassification(nn.Module):
    """
    ERNIE文本分类器
    使用HuggingFace的ERNIE模型
    """
    def __init__(self, model_name="nghuyong/ernie-1.0-base-zh", num_classes=2):
        super().__init__()
        # ERNIE的HuggingFace实现基于BERT架构
        # 使用nghuyong的ERNIE中文预训练权重
        self.ernie = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.ernie.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.ernie(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

class ERNIEForNER(nn.Module):
    """ERNIE命名实体识别"""
    def __init__(self, model_name="nghuyong/ernie-1.0-base-zh", num_labels=7):
        super().__init__()
        self.ernie = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.ernie.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.ernie(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        seq_output = outputs.last_hidden_state
        seq_output = self.dropout(seq_output)
        logits = self.classifier(seq_output)
        return logits

class ERNIEWithKnowledgeMask(nn.Module):
    """
    知识增强的ERNIE预训练
    模拟实体级Mask和短语级Mask
    """
    def __init__(self, vocab_size=21128, hidden_dim=768, num_layers=12, 
                 num_heads=12, max_len=512):
        super().__init__()
        
        self.config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'max_len': max_len
        }
        
        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.token_type_embedding = nn.Embedding(2, hidden_dim)
        
        # Transformer编码器
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, num_heads, hidden_dim*4, batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.ln = nn.LayerNorm(hidden_dim)
        
        # MLM预测头
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # 句子排序任务头
        self.sentence_order_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                mlm_labels=None):
        B, L = input_ids.shape
        
        # 嵌入
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(B, L, dtype=torch.long, 
                                        device=input_ids.device)
        type_emb = self.token_type_embedding(token_type_ids)
        
        x = token_emb + pos_emb + type_emb
        
        # Transformer编码
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln(x)
        
        results = {}
        
        # MLM预测
        if mlm_labels is not None:
            mlm_logits = self.mlm_head(x)
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config['vocab_size']),
                mlm_labels.view(-1),
                ignore_index=-100
            )
            results['mlm_logits'] = mlm_logits
            results['mlm_loss'] = mlm_loss
        
        # 句子排序（使用[CLS]特征）
        cls_feat = x[:, 0]
        
        return results

# 使用示例
class ERNIEKnowledgeMaskTrainer:
    """
    ERNIE知识Mask训练器示例
    演示实体级和短语级Mask策略
    """
    @staticmethod
    def apply_entity_mask(input_ids, entity_spans, mask_token_id, mask_prob=0.15):
        """
        应用实体级Mask
        entity_spans: [(start, end), ...] 实体位置
        """
        masked_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        
        for span_start, span_end in entity_spans:
            if torch.rand(1) < mask_prob:
                # 80% mask整个实体
                if torch.rand(1) < 0.8:
                    masked_ids[:, span_start:span_end] = mask_token_id
                # 10% 随机替换
                elif torch.rand(1) < 0.5:
                    random_ids = torch.randint(0, 21128, (1, span_end - span_start))
                    masked_ids[:, span_start:span_end] = random_ids
                # 10% 保持不变
            
                labels[:, span_start:span_end] = input_ids[:, span_start:span_end]
        
        return masked_ids, labels
    
    @staticmethod
    def apply_phrase_mask(input_ids, phrase_spans, mask_token_id, mask_prob=0.15):
        """应用短语级Mask"""
        return ERNIEKnowledgeMaskTrainer.apply_entity_mask(
            input_ids, phrase_spans, mask_token_id, mask_prob
        )

if __name__ == "__main__":
    # 初始化分类器
    classifier = ERNIEForClassification(num_classes=2)
    
    # 模拟输入
    input_ids = torch.randint(0, 21128, (2, 128))
    attention_mask = torch.ones(2, 128)
    
    logits = classifier(input_ids, attention_mask)
    print(f"分类输出形状: {logits.shape}")  # (2, 2)
    
    # 测试知识Mask
    trainer = ERNIEKnowledgeMaskTrainer()
    input_ids = torch.randint(0, 21128, (1, 20))
    entity_spans = [(2, 4), (7, 10)]  # 两个实体的位置
    mask_token_id = 103  # [MASK] ID
    
    masked_ids, labels = trainer.apply_entity_mask(input_ids, entity_spans, mask_token_id)
    
    print(f"原始IDs: {input_ids[0][:12]}")
    print(f"Mask后IDs: {masked_ids[0][:12]}")
    print(f"标签: {labels[0][:12]}")
    print("ERNIE模型测试成功!")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandcraftERNIE(nn.Module):
    """
    手工实现ERNIE核心功能
    知识增强的Transformer编码器 + 多粒度Mask预测
    """
    def __init__(self, vocab_size=21128, d_model=768, n_heads=12, 
                 n_layers=12, max_len=512):
        super().__init__()
        
        self.d_model = d_model
        
        # 嵌入层
        self.word_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.ln = nn.LayerNorm(d_model)
        
        # 多头预测头
        # ERNIE需要同时预测单个token、实体和短语
        self.predict_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, token_type_ids=None):
        B, L = input_ids.shape
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(B, L, dtype=torch.long)
        
        # 嵌入
        x = self.word_embed(input_ids)
        x = x + self.pos_embed(torch.arange(L).unsqueeze(0))
        x = x + self.type_embed(token_type_ids)
        
        # Transformer
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln(x)
        
        # 预测
        logits = self.predict_head(x)  # (B, L, V)
        
        return logits

class KnowledgeMasking:
    """
    ERNIE知识Mask策略
    重点：实体级Mask和短语级Mask
    """
    def __init__(self, vocab_size=21128, mask_token_id=103):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        
    def mask_tokens(self, input_ids, entity_spans, phrase_spans):
        """
        多粒度Mask
        Args:
            input_ids: (B, L)
            entity_spans: [[(s1,e1), ...], ...] 每个样本的实体位置
            phrase_spans: [[(s1,e1), ...], ...] 每个样本的短语位置
        """
        B, L = input_ids.shape
        masked = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        
        for b in range(B):
            # 1. 实体级Mask (40%概率)
            for s, e in entity_spans[b]:
                if torch.rand(1) < 0.4:
                    # 80% -> [MASK]
                    if torch.rand(1) < 0.8:
                        masked[b, s:e] = self.mask_token_id
                    # 10% -> 随机词
                    elif torch.rand(1) < 0.5:
                        masked[b, s:e] = torch.randint(0, self.vocab_size, (e-s,))
                    # 10% -> 不变
                    labels[b, s:e] = input_ids[b, s:e]
            
            # 2. 短语级Mask (40%概率)
            for s, e in phrase_spans[b]:
                if torch.rand(1) < 0.4 and (labels[b, s:e] == -100).all():
                    if torch.rand(1) < 0.8:
                        masked[b, s:e] = self.mask_token_id
                    elif torch.rand(1) < 0.5:
                        masked[b, s:e] = torch.randint(0, self.vocab_size, (e-s,))
                    labels[b, s:e] = input_ids[b, s:e]
            
            # 3. 基础级Mask (20%概率)
            remaining = (labels[b] == -100).nonzero().squeeze()
            n_mask = max(1, int(len(remaining) * 0.15))
            mask_pos = remaining[torch.randperm(len(remaining))[:n_mask]]
            
            for pos in mask_pos:
                if torch.rand(1) < 0.8:
                    masked[b, pos] = self.mask_token_id
                elif torch.rand(1) < 0.5:
                    masked[b, pos] = torch.randint(0, self.vocab_size, (1,))
                labels[b, pos] = input_ids[b, pos]
        
        return masked, labels

# 测试
if __name__ == "__main__":
    model = HandcraftERNIE()
    masking = KnowledgeMasking()
    
    input_ids = torch.randint(0, 21128, (2, 20))
    entity_spans = [[(2, 4)], [(5, 7)]]
    phrase_spans = [[(8, 11)], [(12, 15)]]
    
    masked, labels = masking.mask_tokens(input_ids, entity_spans, phrase_spans)
    logits = model(masked)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"Mask后形状: {masked.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"输出logits形状: {logits.shape}")  # (2, 20, 21128)
    print(f"被Mask的token数: {(labels != -100).sum().item()}")
    print("手工ERNIE测试通过!")
```

---

## 9. 可视化与结果理解

### 9.1 Mask策略对比

BERT的Mask vs ERNIE的Mask：
```
原文: "北京是中国的首都"
BERT Mask: "北[MASK]是中[MASK]的首[MASK]" 
          → 模型预测"京"、"国"、"都"
ERNIE Mask: "[MASK][MASK]是中国的首都"
           → 模型预测"北京"（整个实体）
```

### 9.2 知识增强的效果

在中文NLP任务上，ERNIE比BERT的提升：
- 命名实体识别：+3-5%
- 关系抽取：+4-6%  
- 阅读理解：+2-3%

### 9.3 注意力可视化

ERNIE的注意力更关注"知识单元"：
- BERT的注意力：分散在单字之间
- ERNIE的注意力：集中在实体内部和实体之间

---

## 10. 模型评估

### 10.1 评估指标

| 任务 | 指标 | BERT-Base | ERNIE 1.0 | ERNIE 2.0 |
|------|------|-----------|-----------|-----------|
| GLUE | 平均分 | 80.5 | 81.8 | 83.6 |
| 中文NER | F1 | 92.8 | 93.5 | 94.2 |
| 中文情感分析 | Acc | 90.1 | 91.8 | 92.5 |
| 中文阅读理解 | F1/EM | 84.5/69.1 | 86.3/72.1 | 88.1/75.3 |

### 10.2 消融实验

- 移除实体级Mask → NER下降2.1%
- 移除短语级Mask → 关系抽取下降3.5%
- 使用单任务 vs 多任务 → 多任务提升2.8%

---

## 11. 常见问题与易错点

### Q1: ERNIE和BERT的根本区别？

ERNIE改进了BERT的Mask策略——从"字级Mask"升级为"知识级Mask"。其他架构和训练方式与BERT基本一致。

### Q2: 英文任务上ERNIE为什么不强？

英文的实体和短语边界更明确（空格分隔），BERT的WordPiece分词已经在一定程度上包含子词信息。中文的语义单位（词/词组）边界模糊，因此知识级Mask对中文的收益更大。

### Q3: 实体级Mask需要额外的标注数据吗？

是的。ERNIE在预训练时需要先使用NER工具标注实体位置，这增加了预训练的复杂度。不过这些标注可以在预处理阶段一次性完成。

### Q4: ERNIE 2.0的持续预训练和普通多任务学习有何区别？

普通多任务学习：所有任务同时训练。持续预训练：分阶段加入新任务，每个新阶段保持旧任务能力。持续预训练避免了"灾难性遗忘"。

### Q5: ERNIE如何在推理时利用"知识"？

ERNIE不是在推理时查询外部知识库，而是通过预训练阶段的"知识级Mask"将知识编码到了模型参数中。这是一种"隐式知识融合"。

---

## 12. 学习总结

### 核心知识点

1. **ERNIE = BERT + 知识增强的Mask策略**
2. **三级Mask**：基础级（字）、实体级（实体）、短语级（短语）
3. **持续预训练**：多任务分阶段累加训练
4. **中文强项**：在中文NLP任务上显著优于BERT

### 架构速记

ERNIE = 标准BERT架构 + 知识Mask策略 + 持续多任务预训练

### 关键洞见

好的预训练策略比好的模型架构更重要——ERNIE没有改变BERT的架构，仅通过改进Mask策略就取得了显著提升。

---

## 13. 练习题与思考题（含答案）

### 习题1：Mask策略

**问题**：如果文本中有实体"中华人民共和国"，BERT和ERNIE会如何Mask它？

**答案**：BERT会随机Mask其中的一个字（如"民"），而ERNIE会将"中华人民共和国"整体Mask，迫使模型预测这7个字。

### 习题2：持续预训练

**问题**：ERNIE 2.0为什么使用持续预训练而不是一次性训练所有任务？

**答案**：一次性训练所有任务可能导致任务冲突或梯度干扰。分阶段训练让模型先掌握基础能力，在稳定基础上逐步学习新能力。

### 习题3：中文vs英文

**问题**：为什么ERNIE在中文任务上的提升比英文任务更显著？

**答案**：中文没有天然的分词边界，BERT的"字级Mask"无法学到词/短语级别的语义。ERNIE的知识Mask填补了这一空白。英文的WordPiece分词已经包含了子词级别信息。

### 习题4：实体识别

**问题**：ERNIE的预训练中依赖NER工具标注实体，如果NER工具有错误怎么办？

**答案**：这是ERNIE的一个局限。但实验表明，即使使用有噪声的NER标注，知识Mask仍然带来显著提升（噪声的标注>无标注）。

### 习题5：思考题

**问题**：除了实体和短语，你还能想到哪些级别的"知识Mask"？

**答案**：可能的扩展包括：句子级Mask（如同时Mask一个句子的多个相关词）、关系级Mask（如同时Mask"北京"和"中国"让模型预测它们的关系）、属性级Mask（如Mask"北京"让模型预测"首都"）。

---

## 14. 学习路径建议

### 前置知识
- BERT / Transformer
- 命名实体识别（NER）
- 预训练-微调范式
- 中文NLP基础

### 平行模型
- **BERT**：ERNIE的基础
- **RoBERTa**：BERT的优化版（更多数据、动态Mask）
- **ALBERT**：更轻量的BERT

### 进阶方向
- **ERNIE 3.0**：百亿参数级知识增强模型
- **ERNIE-Doctor**：融合医学知识的ERNIE
- **ERNIE-Sage**：知识图谱增强的ERNIE
- **ENRIE-Twin**：双塔匹配的ERNIE

### 学习顺序建议

```
① BERT基础 → ② 中文NLP特点 → ③ ERNIE 1.0（知识Mask） → ④ ERNIE 2.0（持续预训练） → ⑤ ERNIE 3.0
```
