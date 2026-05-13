# XLM（Cross-lingual Language Model）学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
XLM（Cross-lingual Language Model）是Facebook AI于2019年提出的跨语言预训练模型，通过共享词汇表和跨语言训练目标，让单一模型同时理解多种语言并在不同语言之间迁移知识。

### 1.2 直觉类比
XLM就像一个精通多国语言的学生——他发现"猫"（中文）、"cat"（英文）、"chat"（法文）、"Katze"（德文）都指向同一概念。这种跨语言的共享理解让他可以用英语学到的知识帮助理解法语。

### 1.3 历史背景
- **2019年1月**：Facebook AI提出XLM
- **2019年8月**：发布XLM-R（多语言RoBERTa）
- **2020年**：XLM-R成为多语言NLP任务的标准基线

### 1.4 算法定位
XLM属于**跨语言预训练模型**，使用自监督学习从多语言文本中学习语言无关的语义表示。

---

## 2. 核心原理

### 2.1 共享BPE词汇表
XLM使用语言无关的BPE（Byte-Pair Encoding）分词：
- 将不同语言的文本统一到同一个词汇表中
- 频繁出现的子词（如"tion"）在不同语言中共享token ID
- 词汇表大小通常为200K（远大于单语言的30K）

### 2.2 跨语言掩码语言建模（MLM）
类似BERT的MLM，但在多语言混合语料上训练：
- 输入：来自任意语言的单句
- 任务：预测被mask的token
- 通过共享嵌入层，模型学会将不同语言的"猫"映射到相近的向量空间

### 2.3 翻译语言建模（TLM）
XLM的核心创新——利用平行语料进行跨语言对齐：

**输入**：一对翻译句（如"猫坐在垫子上 || The cat sits on the mat"）
**mask策略**：随机mask源语言和目标语言的token
**挑战**：预测目标语言中被mask的token可能需要参考源语言的对应部分
**效果**：强制模型学习跨语言的对齐关系

---

## 3. 数学公式与推导

### 3.1 跨语言MLM损失
给定单语言文本 $x$ 和 mask 版本 $\hat{x}$：

$$L_{MLM} = -\mathbb{E}_{x\sim D} \sum_{i \in \mathcal{M}} \log P(x_i | \hat{x})$$

其中 $\mathcal{M}$ 是被mask的位置集合。

### 3.2 TLM损失
给定平行句对 $(x, y)$（$x$ 是源语言，$y$ 是目标语言）：

$$L_{TLM} = -\mathbb{E}_{(x,y)\sim D_{parallel}} \sum_{i \in \mathcal{M}} \log P(x_i | \hat{x}, \hat{y}) - \mathbb{E}_{(x,y)\sim D_{parallel}} \sum_{j \in \mathcal{N}} \log P(y_j | \hat{x}, \hat{y})$$

其中 $\mathcal{M}$ 是 $x$ 中的mask位置，$\mathcal{N}$ 是 $y$ 中的mask位置。

### 3.3 联合训练目标
XLM联合使用单语和多语数据：

$$L_{XLM} = L_{MLM} + \lambda L_{TLM}$$

其中 $\lambda$ 控制TLM的权重，通常设为1.0。

### 3.4 跨语言对齐的理论分析
令 $e_{lang1}(c)$ 和 $e_{lang2}(c)$ 分别表示概念 $c$ 在语言1和语言2中的词嵌入。TLM迫使：

$$e_{lang1}(c) \approx e_{lang2}(c)$$

因为如果一个词在源语言中未被mask，而其对应的翻译在目标语言中被mask，模型需要通过源语言词嵌入来预测目标语言的mask位置。这迫使两种语言的嵌入空间对齐。

---

## 4. 训练过程讲解

### 4.1 训练数据
- 单语数据：Wikipedia各语言版本（100种语言）
- 平行数据：从MultiUN、Europarl等语料库收集（约2亿句对）
- 采样策略：按语言大小进行指数平滑采样

### 4.2 训练步骤
1. **数据预处理**：对所有语言文本进行BPE分词
2. **构建batch**：每个batch包含混合语言的数据
3. **MLM step**：从单语数据中采样，随机mask 15% token
4. **TLM step**：从平行数据中采样，随机mask两种语言的token
5. **模型更新**：计算总损失并反向传播

### 4.3 语言采样策略
为防止小语种被大语种淹没，使用多项式平滑采样：

$$P(lang) \propto p_{lang}^{\alpha}$$

其中 $p_{lang}$ 是语言占比，$\alpha=0.7$ 控制平滑程度。

---

## 5. 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| 跨语言分类 | 用英语数据训练，预测其他语言 | XNLI: 英语训练→中文预测 |
| 跨语言检索 | 不同语言间的相似度计算 | 中文查询→英文文档 |
| 机器翻译 | 微调后用于翻译任务 | 英语→德语 |
| 零样本跨语言 | 目标语言无训练数据 | 法语情感分析（用英语模型） |

---

## 6. 优缺点分析

### 优点
1. **跨语言迁移**：单语言标注数据可用于多语言任务
2. **共享表示**：不同语言在同一语义空间
3. **支持多语言**：单模型支持100+语言
4. **数据高效**：利用平行语料对齐语言

### 缺点
1. **词汇表大**：200K词汇表导致嵌入层巨大
2. **TLM依赖平行语料**：低资源语言缺乏平行数据
3. **大语言不均衡**：英语性能远优于小语种

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMModel, XLMTokenizer, XLMConfig

class XLMMultiLingualClassifier(nn.Module):
    """XLM跨语言文本分类器"""
    def __init__(self, model_name="xlm-mlm-100-1280", num_classes=15):
        super().__init__()
        # 加载预训练XLM模型
        self.xlm = XLMModel.from_pretrained(model_name)
        self.config = XLMConfig.from_pretrained(model_name)
        hidden_dim = self.config.emb_dim  # 1280
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, langs):
        """
        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] mask
            langs: [B, L] 语言id（XLM需要）
        """
        outputs = self.xlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs
        )
        # 使用[CLS] token的表示
        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_rep)
        return logits


class XLMCausalGeneration(nn.Module):
    """XLM跨语言文本生成"""
    def __init__(self):
        super().__init__()
        # 使用XLM的LM head版本做生成
        from transformers import XLMWithLMHeadModel
        self.model = XLMWithLMHeadModel.from_pretrained("xlm-mlm-enfr-1024")
        self.tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-enfr-1024")
        
    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # 提供语言id（1=英语, 2=法语等）
        langs = torch.full_like(inputs.input_ids, 1)
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            langs=langs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_xlm():
    """测试XLM跨语言分类"""
    tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-100-1280")
    model = XLMMultiLingualClassifier()
    
    texts = ["Hello world", "Bonjour le monde", "你好世界"]
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        langs = torch.zeros_like(inputs.input_ids)  # 默认语言id
        logits = model(inputs.input_ids, inputs.attention_mask, langs)
        print(f"{text}: 输出形状 {logits.shape}")
    
    print("XLM测试通过！")

if __name__ == "__main__":
    test_xlm()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandwrittenXLM(nn.Module):
    """XLM核心逻辑手工实现（跨语言MLM + TLM）"""
    def __init__(self, vocab_size=200000, d_model=1024, nhead=16, num_layers=12):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入（共享所有语言）
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 语言嵌入（区分不同语言）
        self.lang_embedding = nn.Embedding(100, d_model)  # 支持100种语言
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=512)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # 预测头（共享所有语言）
        self.predict_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, langs, attention_mask=None):
        """
        Args:
            input_ids: [B, L]
            langs: [B, L] 每个token的语言id
        """
        # 词嵌入 + 语言嵌入 + 位置编码
        word_emb = self.embedding(input_ids)
        lang_emb = self.lang_embedding(langs)
        pos_enc = self.pos_encoding(input_ids.size(1))
        
        x = word_emb + lang_emb + pos_enc.unsqueeze(0)
        
        # 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        logits = self.predict_head(x)
        return logits
    
    def compute_mlm_loss(self, input_ids, langs, masked_positions, labels):
        """计算MLM损失"""
        logits = self.forward(input_ids, langs)
        # 只计算mask位置的损失
        masked_logits = logits[masked_positions]
        loss = F.cross_entropy(masked_logits, labels)
        return loss
    
    def compute_tlm_loss(self, src_ids, tgt_ids, src_langs, tgt_langs, 
                          masked_src_positions, masked_tgt_positions,
                          src_labels, tgt_labels):
        """计算TLM损失（双语平行句）"""
        # 拼接源语言和目标语言
        B = src_ids.size(0)
        combined_ids = torch.cat([src_ids, tgt_ids], dim=1)
        combined_langs = torch.cat([src_langs, tgt_langs], dim=1)
        
        logits = self.forward(combined_ids, combined_langs)
        
        src_len = src_ids.size(1)
        src_logits = logits[:, :src_len, :]
        tgt_logits = logits[:, src_len:, :]
        
        # 分别计算损失
        loss_src = F.cross_entropy(src_logits[masked_src_positions], src_labels)
        loss_tgt = F.cross_entropy(tgt_logits[masked_tgt_positions], tgt_labels)
        
        return loss_src + loss_tgt


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x_len):
        return self.pe[:x_len, :]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


def test_xlm_handwritten():
    model = HandwrittenXLM(vocab_size=50000)
    B, L = 2, 20
    ids = torch.randint(0, 50000, (B, L))
    langs = torch.randint(0, 10, (B, L))
    
    logits = model(ids, langs)
    print(f"手工XLM输出: {logits.shape}")

if __name__ == "__main__":
    test_xlm_handwritten()
```

---

## 9. 可视化与结果理解

### 9.1 跨语言嵌入空间可视化
使用t-SNE可视化XLM学到的多语言词嵌入：
- 同一概念在不同语言中的嵌入在空间中聚在一起
- "dog"（英）、"chien"（法）、"Hund"（德）在嵌入空间中接近
- 这验证了XLM实现了跨语言对齐

### 9.2 跨语言迁移效果
对比有/无TLM训练的效果：
- 仅有MLM：不同语言的嵌入空间是分离的（各有各的区域）
- 加入TLM：嵌入空间对齐，跨语言迁移效果大幅提升

---

## 10. 模型评估

### 10.1 跨语言分类评估（XNLI基准）
```python
def evaluate_xnli(model, dataloader, languages=['en', 'fr', 'zh']):
    model.eval()
    results = {}
    for lang in languages:
        correct, total = 0, 0
        for batch in dataloader[lang]:
            input_ids, attention_mask, langs, labels = batch
            logits = model(input_ids, attention_mask, langs)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        results[lang] = correct / total
    return results
```

### 10.2 跨语言检索评估
| 指标 | 同语言检索 | 跨语言检索 |
|------|-----------|-----------|
| R@1 (英语→英语) | 85.2% | - |
| R@1 (英语→法语) | - | 82.1% |
| R@1 (英语→德语) | - | 79.8% |

---

## 11. 常见问题与易错点

### Q1: XLM和mBERT有什么区别？
A: mBERT对每种语言各自用BERT训练，语言间共享知识有限。XLM通过TLM显式对齐语言嵌入空间，跨语言迁移能力更强。

### Q2: TLM需要大量平行语料吗？
A: TLM确实需要平行语料，但XLM对平行语料的需求量相对较小——大部分学习来自单语MLM，TLM只起对齐作用。

### Q3: 如何处理多语言输入中不同语言的token？
A: XLM为每个token分配一个语言id，通过语言嵌入层将语言信息注入模型。

### Q4: XLM-R和XLM的区别是什么？
A: XLM-R用RoBERTa的训练方式替代XLM的BERT方式，去掉了NSP任务，使用更多数据，在大语种上表现更好。

---

## 12. 学习总结

### 核心贡献
1. **TLM（翻译语言建模）**：用平行语料对齐多语言嵌入空间
2. **共享BPE词汇表**：不同语言共享子词单元
3. **跨语言迁移学习框架**：为多语言NLP建立标准范式

### 技术要点
- 语言嵌入 + 词嵌入的融合
- MLM + TLM的联合训练
- 指数平滑语言采样

---

## 13. 练习题与思考题（含答案）

### 习题1：理解题
XLM如何实现跨语言语义对齐？TLM在其中起什么作用？

**答案**：XLM通过共享词汇表和TLM目标实现跨语言对齐。TLM在平行句对两侧同时mask token，模型为了预测目标语言中被mask的token，需要参考源语言的上下文。这迫使模型将两种语言的语义表示对齐到同一空间。

### 习题2：推导题
假设TLM中输入句对为（英语: "The cat sits"；法语: "Le chat s'assoit"），英语侧的"cat"被mask。模型需要利用哪些信息来预测"cat"？

**答案**：模型可以利用：1）英语侧上下文："The [MASK] sits"中的"The"和"sits"；2）法语侧信息："Le chat s'assoit"中的"chat"（法语"猫"）可以作为强线索。这就是TLM的核心对齐机制。

### 习题3：编程题
实现TLM的损失计算函数。

**答案**：
```python
def tlm_loss(model, src_ids, src_langs, tgt_ids, tgt_langs, src_mask, tgt_mask):
    combined_ids = torch.cat([src_ids, tgt_ids], dim=1)
    combined_langs = torch.cat([src_langs, tgt_langs], dim=1)
    logits = model(combined_ids, combined_langs)
    src_len = src_ids.size(1)
    src_logits = logits[:, :src_len, :]
    tgt_logits = logits[:, src_len:, :]
    loss_src = F.cross_entropy(src_logits[src_mask], src_ids[src_mask])
    loss_tgt = F.cross_entropy(tgt_logits[tgt_mask], tgt_ids[tgt_mask])
    return loss_src + loss_tgt
```

### 习题4：思考题
如果没有平行语料，还能做跨语言预训练吗？有什么替代方案？

**答案**：可以。替代方案包括：1）跨语言对比学习（如InfoXLM），利用对齐的正负样本对做对比损失；2）跨语言翻译后MLM，先用翻译模型将一种语言翻译为另一种，在翻译数据上做MLM；3）语言对抗训练，通过判别器使模型无法区分语言，迫使语言无关的表示。

---

## 14. 学习路径建议

### 前置知识
- **BERT**：MLM预训练基础
- **Transformer**：自注意力机制
- **BPE分词**：子词分词原理
- **机器翻译基础**：平行语料的概念

### 进阶方向
1. **XLM-R**：扩展XLM到RoBERTa训练方式
2. **InfoXLM**：引入对比学习改进对齐
3. **mT5**：多语言T5（Text-to-Text框架）
4. **mBART**：多语言BART（Seq2Seq去噪）
5. **M2M-100**：多语言机器翻译模型

### 学习路线
```
BERT → XLM → XLM-R → mT5/mBART → M2M-100
```
