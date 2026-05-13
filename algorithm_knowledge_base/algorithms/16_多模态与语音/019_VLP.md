# VLP 学习文档
> 来源线索：本节内容根据原书相关章节整理。

## 1. 算法基础认知

### 1.1 一句话定义
VLP（Vision-Language Pre-training）是2019年提出的统一视觉语言预训练模型，首次在同一个框架中支持**理解和生成**两类任务，开创了多任务VLP的先河。

### 1.2 直觉类比
VLP就像一个"双模式"翻译器——模式一是"听写模式"（理解）：听懂你说什么然后回答；模式二是"朗读模式"（生成）：看着图片描述出来。同一个大脑，两种工作方式，由不同的注意力掩码控制切换。

### 1.3 历史背景
- **2019年9月**：微软提出VLP
- **意义**：最早支持理解+生成统一的VLP模型之一
- **技术路线**：基于UniLM的跨模态扩展

### 1.4 算法定位
VLP属于**统一视觉语言预训练模型**，支持视觉推理（VQA）和视觉生成（Captioning）的双重目标。

---

## 2. 核心原理

### 2.1 统一架构设计
VLP基于Transformer的单流架构，通过不同的注意力掩码模式切换任务：

**理解模式（Comprehension Mode）**：
- 使用双向注意力（类似BERT）
- 图像特征和文本特征全连接
- 适用于VQA、图文匹配等理解任务

**生成模式（Generation Mode）**：
- 使用单向注意力（类似GPT）
- 文本生成时只能看到左侧的文本和全部图像
- 适用于图像描述等生成任务

### 2.2 双模式注意力掩码
VLP的核心在于三种注意力掩码：

1. **双向掩码（Bidirectional）**：文本和图像区域之间全连接
2. **序列到序列掩码（Seq2Seq）**：编码器（图像）→解码器（文本）单向流
3. **语言模型掩码（LM）**：文本内部从左到右

### 2.3 输入表示
- **图像**：Faster R-CNN提取34个区域特征 + 5维位置编码
- **文本**：BERT WordPiece + 位置编码 + 段编码
- 输入序列：[CLS] + 文本 + [SEP] + 图像区域

---

## 3. 数学公式与推导

### 3.1 理解模式的目标
在理解模式下，VLP使用掩码语言建模（MLM）目标：

$$L_{mlm} = -\mathbb{E}_{(v,w)\sim D} \log P(w_m | w_{\backslash m}, v)$$

其中 $w_m$ 是被mask的文本词，$v$ 是图像区域特征。

### 3.2 生成模式的目标
在生成模式下，VLP使用自回归语言建模目标：

$$L_{gen} = -\mathbb{E}_{(v,w)\sim D} \sum_{t=1}^{T} \log P(w_t | w_{<t}, v)$$

其中 $w_t$ 是第 $t$ 个token，$w_{<t}$ 是之前生成的token。

### 3.3 联合训练目标
VLP联合优化两个目标：

$$L_{VLP} = L_{mlm} + \lambda L_{gen}$$

其中 $\lambda$ 平衡理解和生成两个目标。

### 3.4 注意力机制的数学形式
理解模式的注意力计算：

$$\text{Attn}_{bidir}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

所有位置互相可见。生成模式的注意力：

$$\text{Attn}_{unidir}(Q_t, K_{<t}, V_{<t}) = \text{softmax}\left(\frac{Q_t K_{<t}^T}{\sqrt{d_k}}\right)V_{<t}$$

位置 $t$ 只能看到 $\leq t$ 的位置。

---

## 4. 训练过程讲解

### 4.1 预训练阶段
1. **数据准备**：使用MS-COCO图像描述数据集
2. **图像处理**：Faster R-CNN提取34个区域特征
3. **文本处理**：随机mask 15%的文本token
4. **前向传播**：分别在理解模式和生成模式下计算损失
5. **反向传播**：联合优化两个目标

### 4.2 模式切换机制
- 每个训练step以50%概率选择理解模式或生成模式
- 两种模式共享除注意力掩码外的所有参数
- 通过attention mask矩阵控制信息流

### 4.3 微调过程
- VQA微调：使用理解模式，分类所有候选答案
- 图像描述微调：使用生成模式，自回归生成文本
- 图文检索微调：计算图文相似度排序

---

## 5. 应用场景

| 场景 | 模式 | 输入 | 输出 |
|------|------|------|------|
| 视觉问答 | 理解 | 图像 + 问题 | 答案 |
| 图像描述 | 生成 | 图像 | 文本描述 |
| 图文匹配 | 理解 | 图像 + 文本 | 匹配/不匹配 |
| 指代理解 | 理解 | 图像 + 指代短语 | 目标区域 |

---

## 6. 优缺点分析

### 优点
1. **统一架构**：一个模型支持理解和生成
2. **参数共享**：两种模式共享参数，训练高效
3. **灵活性**：可动态切换任务模式

### 缺点
1. **模式干扰**：理解模式和生成模式可能互相干扰
2. **数据需求**：需要大量图文对数据
3. **掩码设计复杂**：需要精心设计三种注意力掩码

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig
import math

class VLPModel(nn.Module):
    """VLP统一视觉语言预训练模型"""
    def __init__(self, vocab_size=30522, hidden_dim=768, num_heads=12, 
                 num_layers=6, max_seq_len=50, num_regions=34):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # 文本嵌入层
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len + num_regions + 2, hidden_dim)
        self.segment_embedding = nn.Embedding(3, hidden_dim)  # 文本/图像/特殊
        
        # 图像区域投影
        self.img_projection = nn.Linear(2048, hidden_dim)
        self.loc_layer = nn.Sequential(
            nn.Linear(5, hidden_dim),  # [x1,y1,x2,y2,w]
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # MLM预测头
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # 输出预测头（生成模式）
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
    def create_attention_mask(self, text_len, num_regions, mode='bidirectional'):
        """
        创建注意力掩码
        Args:
            text_len: 文本长度（含[CLS]和[SEP]）
            num_regions: 图像区域数
            mode: 'bidirectional' / 'seq2seq' / 'lm'
        Returns:
            mask: [text_len+num_regions, text_len+num_regions]
        """
        total_len = text_len + num_regions
        mask = torch.zeros(total_len, total_len)
        
        if mode == 'bidirectional':
            # 所有位置互相可见
            mask[:, :] = 1
        elif mode == 'seq2seq':
            # 编码器部分（图像）可见全部，解码器部分（文本）单向
            # 图像区域: 最后num_regions个位置
            mask[text_len:, :] = 1  # 图像看全部
            # 文本: 单向
            for i in range(text_len):
                mask[i, :i+1] = 1
                mask[i, text_len:] = 1  # 文本看图像
        elif mode == 'lm':
            # 纯单向语言模型
            for i in range(total_len):
                mask[i, :i+1] = 1
                
        return mask
    
    def forward(self, input_ids, img_features, img_locs, mode='bidirectional', 
                attention_mask=None, masked_positions=None):
        """
        Args:
            input_ids: 文本token [B, text_len]
            img_features: 图像区域特征 [B, num_regions, feat_dim]
            img_locs: 图像位置 [B, num_regions, 5]
            mode: 'bidirectional' / 'seq2seq' / 'lm'
        """
        B, T = input_ids.shape
        N = img_features.shape[1]
        
        # 1. 文本嵌入
        text_emb = self.word_embedding(input_ids)
        text_pos = self.pos_embedding(torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1))
        text_seg = self.segment_embedding(torch.zeros(B, T, dtype=torch.long, device=input_ids.device))
        text_emb = text_emb + text_pos + text_seg
        
        # 2. 图像嵌入
        img_feat = self.img_projection(img_features)
        img_loc = self.loc_layer(img_locs)
        img_emb = img_feat + img_loc
        img_pos = self.pos_embedding(
            torch.arange(T, T + N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        )
        img_seg = self.segment_embedding(torch.ones(B, N, dtype=torch.long, device=input_ids.device))
        img_emb = img_emb + img_pos + img_seg
        
        # 3. 拼接
        combined = torch.cat([text_emb, img_emb], dim=1)
        total_len = T + N
        
        # 4. 创建注意力掩码
        attn_mask = self.create_attention_mask(T, N, mode).to(input_ids.device)
        
        # 5. 通过编码器层
        for layer in self.encoder_layers:
            combined = layer(combined, attn_mask)
        combined = self.final_norm(combined)
        
        # 6. 输出
        text_output = combined[:, :T, :]
        
        if mode == 'bidirectional':
            # 理解模式：MLM预测
            if masked_positions is not None:
                # 只预测被mask的位置
                masked_output = text_output.gather(
                    1, masked_positions.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
                )
                logits = self.mlm_head(masked_output)
            else:
                logits = self.mlm_head(text_output)
        else:
            # 生成模式：语言模型预测
            logits = self.lm_head(text_output)
            
        return logits


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


def test_vlp():
    """测试VLP模型"""
    model = VLPModel()
    B, T, N = 2, 20, 34
    
    input_ids = torch.randint(0, 1000, (B, T))
    img_features = torch.randn(B, N, 2048)
    img_locs = torch.randn(B, N, 5)
    
    # 理解模式
    logits = model(input_ids, img_features, img_locs, mode='bidirectional')
    print(f"理解模式输出: {logits.shape}")
    
    # 生成模式
    logits = model(input_ids, img_features, img_locs, mode='seq2seq')
    print(f"生成模式输出: {logits.shape}")
    
    print("VLP模型测试通过！")

if __name__ == "__main__":
    test_vlp()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HandwrittenVLP(nn.Module):
    """VLP核心逻辑手工实现"""
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(500, d_model)
        
        # 图像到文本投影
        self.img_project = nn.Linear(2048, d_model)
        
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        
        # 输出层
        self.output = nn.Linear(d_model, vocab_size)
        
    def generate_attention_mask(self, text_len, total_len, mode='bidirectional'):
        """生成不同模式的注意力掩码"""
        mask = torch.full((total_len, total_len), float('-inf'))
        
        if mode == 'bidirectional':
            # 全双向（理解模式）
            mask = torch.zeros(total_len, total_len)
        elif mode == 'seq2seq':
            # 编码器-解码器模式
            img_start = text_len
            # 文本可以看到文本左侧 + 全部图像
            for i in range(text_len):
                mask[i, :i+1] = 0  # 看左侧文本
                mask[i, img_start:] = 0  # 看全部图像
            # 图像可以互相看
            for i in range(img_start, total_len):
                mask[i, :] = 0
        else:  # lm
            # 纯自回归
            for i in range(total_len):
                mask[i, :i+1] = 0
                
        return mask
    
    def forward(self, text_ids, img_feats, mode='bidirectional'):
        B, T = text_ids.shape
        N = img_feats.shape[1]
        total_len = T + N
        
        # 文本嵌入
        text_emb = self.token_emb(text_ids) + self.pos_emb(torch.arange(T).unsqueeze(0))
        
        # 图像嵌入
        img_emb = self.img_project(img_feats)
        
        # 拼接
        combined = torch.cat([text_emb, img_emb], dim=1)
        
        # 创建掩码
        mask = self.generate_attention_mask(T, total_len, mode).to(text_ids.device)
        
        # 自注意力
        attn_out, _ = self.self_attn(combined, combined, combined, attn_mask=mask)
        combined = self.norm(combined + attn_out)
        
        # FFN
        ffn_out = self.ffn(combined)
        combined = self.norm(combined + ffn_out)
        
        # 输出（只对文本部分做预测）
        logits = self.output(combined[:, :T, :])
        return logits


def test_handwritten():
    model = HandwrittenVLP()
    B, T, N = 2, 10, 4
    text_ids = torch.randint(0, 1000, (B, T))
    img_feats = torch.randn(B, N, 2048)
    
    out_bidir = model(text_ids, img_feats, 'bidirectional')
    out_gen = model(text_ids, img_feats, 'seq2seq')
    print(f"手工VLP - 理解输出: {out_bidir.shape}, 生成输出: {out_gen.shape}")

if __name__ == "__main__":
    test_handwritten()
```

---

## 9. 可视化与结果理解

### 9.1 模式切换的可视化
通过注意力热力图可以清晰看到两种模式的差异：
- **理解模式**：文本位置i可以看到所有位置，热力图呈现均匀分布
- **生成模式**：文本位置i只能看到左侧和图像，热力图呈现三角形+右侧均匀

### 9.2 图像-文本对齐效果
VLP在理解模式下，文本词的注意力会集中到相关的图像区域。例如：
- 文本"dog" → 图像区域中检测到狗的区域（高注意力）
- 文本"running" → 动物的腿部区域（高注意力）

---

## 10. 模型评估

### 10.1 VQA评估
在VQA v2.0数据集上评估答案准确率：
```python
def evaluate_vqa(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, questions, answers in dataloader:
            # 理解模式预测
            logits = model(questions, images, mode='bidirectional')
            preds = logits.argmax(dim=-1)
            correct += (preds == answers).sum().item()
            total += answers.size(0)
    return correct / total
```

### 10.2 图像描述评估（使用CIDEr指标）
```python
def evaluate_captioning(model, dataloader):
    """评估图像描述生成质量"""
    # 使用生成模式生成描述文本
    # 计算CIDEr、BLEU等指标
    pass
```

---

## 11. 常见问题与易错点

### Q1: 理解模式和生成模式共享参数会带来什么问题？
A: 共享参数可能导致"模式冲突"——同一个参数在理解模式下做双向建模，在生成模式下做单向建模。VLP通过层归一化和残差连接缓解了这个问题，但理论上解耦参数可能更好（如T5的做法）。

### Q2: 为什么VLP采用单流而非双流？
A: 单流架构更简单，参数更少，且通过注意力掩码即可实现模式切换。但单流的缺点是图文特征交互不够深入。

### Q3: VLP的输入序列长度限制是多少？
A: 取决于Transformer的最大位置编码。VLP通常支持文本长度+图像区域数不超过512。

### Q4: 训练时如何平衡理解和生成两个目标？
A: 通过超参数 $\lambda$ 控制，通常设置为1.0。实践中可以动态调整，例如在训练初期侧重理解，后期侧重生成。

---

## 12. 学习总结

### 核心贡献
1. **首个统一理解和生成的VLP模型**：验证了单流+注意力掩码方案的有效性
2. **双模式预训练**：通过任务切换实现多任务学习
3. **理论基础**：为后续UniLM风格的跨模态模型奠定了基础

### 关键设计
- 三种注意力掩码（双向/Seq2Seq/LM）
- 共享Transformer参数
- 联合优化MLM和生成目标

---

## 13. 练习题与思考题（含答案）

### 习题1：理解题
VLP是如何在不改变模型架构的情况下切换理解和生成模式的？

**答案**：通过改变注意力掩码矩阵。理解模式使用双向注意力（全连接），生成模式使用单向注意力（从左到右+看图像）。模型架构完全相同，只有掩码不同。

### 习题2：推导题
假设文本长度为T，图像区域数为N，请推导理解模式和生成模式下注意力计算的时间复杂度。

**答案**：
- 理解模式：$O((T+N)^2 \cdot d)$，所有位置互相注意
- 生成模式（Seq2Seq）：$O(T(T+N) \cdot d)$，文本看左侧+全部图像，图像看全部
- 生成模式下，生成第t个token需要计算 $O((t+N) \cdot d)$ 次注意力

### 习题3：编程题
实现VLP的三种注意力掩码生成函数。

**答案**：
```python
def create_vlp_mask(T, N, mode):
    total = T + N
    mask = torch.zeros(total, total)
    if mode == 'bidirectional':
        mask[:, :] = 1
    elif mode == 'seq2seq':
        for i in range(T):
            mask[i, :i+1] = 1
            mask[i, T:] = 1
        mask[T:, :] = 1
    else:  # lm
        for i in range(total):
            mask[i, :i+1] = 1
    return mask
```

### 习题4：思考题
VLP的设计对后续的BLIP和Unified-IO等模型有什么启发？

**答案**：VLP验证了"统一架构+任务条件化"的可行性。后续模型在此基础上扩展：1）BLIP使用了类似的模式切换但增加了带噪数据过滤；2）Unified-IO扩展到更多任务类型（检测、分割等）；3）任务提示（prompt）取代了显式的模式切换，更加灵活。

---

## 14. 学习路径建议

### 前置知识
- **Transformer**：理解自注意力和掩码机制
- **BERT**：双向语言模型预训练
- **UniLM**：统一语言模型的前身
- **Faster R-CNN**：图像区域特征提取

### 进阶方向
1. **UniLM v2**：VLP的文本版本，更复杂的掩码策略
2. **BLIP**：引入CapFilt机制改进VLP的数据质量
3. **OFA**：统一所有任务的序列到序列框架
4. **Unified-IO**：扩展到更多模态和任务

### 学习路线
```
Transformer → BERT → UniLM → VLP → BLIP → OFA → Unified-IO
```
