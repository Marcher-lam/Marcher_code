# 层级注意力网络HAN 学习文档

> 文档级文本分类中的分层注意力——从词级到句子级，让模型关注最重要的词和句子。

> 来源线索：本节内容根据原书第3章关于"层级注意力机制"的相关章节整理。

## 1. 算法基础认知

**一句话定义：** 层级注意力网络（Hierarchical Attention Network, HAN）由Yang等人于2016年提出，采用两层注意力结构（词级+句子级），在文档分类任务中让模型学会关注重要的词和重要的句子。

**直觉类比：** 阅读一篇论文时，你会先快速浏览段落，找出关键段落（句子级注意），然后在关键段落中细读关键词句（词级注意）。HAN模拟了这个"从整体到局部"的注意力机制。

**历史背景：** 2016年在NAACL上发表，提出了一种利用文档层次结构（词→句子→文档）的注意力模型。

---

## 2. 核心原理

### 2.1 模型架构

```
文档 → [句子1, 句子2, ..., 句子N]
  ↓ 词编码器
  ↓ 词级注意力
  ↓ 句子向量
  ↓ 句子编码器(双向GRU)
  ↓ 句子级注意力
  ↓ 文档向量 → 分类
```

### 2.2 词级注意力

对每个句子中的词计算注意力权重：

$$u_{it} = \tanh(W_w h_{it} + b_w)$$
$$\alpha_{it} = \frac{\exp(u_{it}^\top u_w)}{\sum_t \exp(u_{it}^\top u_w)}$$
$$s_i = \sum_t \alpha_{it} h_{it}$$

### 2.3 句子级注意力

对每个句子计算注意力权重：

$$u_i = \tanh(W_s s_i + b_s)$$
$$\alpha_i = \frac{\exp(u_i^\top u_s)}{\sum_i \exp(u_i^\top u_s)}$$
$$v = \sum_i \alpha_i s_i$$

---

## 3. 调库实现

```python
"""
层级注意力网络HAN的PyTorch实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordLevelAttention(nn.Module):
    """词级注意力"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_fc = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.context_vector = nn.Parameter(torch.randn(2 * hidden_size))
    
    def forward(self, word_outputs):
        """word_outputs: (batch, seq_len, 2*hidden)"""
        u = torch.tanh(self.attention_fc(word_outputs))
        scores = torch.matmul(u, self.context_vector)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), word_outputs).squeeze(1)
        return context, weights


class SentenceLevelAttention(nn.Module):
    """句子级注意力"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_fc = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.context_vector = nn.Parameter(torch.randn(2 * hidden_size))
    
    def forward(self, sentence_outputs):
        """sentence_outputs: (batch, n_sentences, 2*hidden)"""
        u = torch.tanh(self.attention_fc(sentence_outputs))
        scores = torch.matmul(u, self.context_vector)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), sentence_outputs).squeeze(1)
        return context, weights


class HAN(nn.Module):
    """层级注意力网络"""
    
    def __init__(self, vocab_size, embed_size=200, hidden_size=50, n_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 词级层
        self.word_gru = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.word_attention = WordLevelAttention(hidden_size)
        self.word_dropout = nn.Dropout(0.5)
        
        # 句子级层
        self.sentence_gru = nn.GRU(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.sentence_attention = SentenceLevelAttention(hidden_size)
        self.sentence_dropout = nn.Dropout(0.5)
        
        # 分类器
        self.classifier = nn.Linear(2 * hidden_size, n_classes)
    
    def forward(self, x):
        """
        x: (batch, n_sentences, seq_len)
        """
        batch, n_sentences, seq_len = x.size()
        x = x.view(-1, seq_len)  # (batch*n_sentences, seq_len)
        
        # 词级
        embed = self.embedding(x)  # (batch*n_sentences, seq_len, embed)
        word_output, _ = self.word_gru(embed)
        word_context, word_weights = self.word_attention(word_output)
        word_context = self.word_dropout(word_context)
        
        # 句子级
        sentence_input = word_context.view(batch, n_sentences, -1)
        sentence_output, _ = self.sentence_gru(sentence_input)
        doc_context, sentence_weights = self.sentence_attention(sentence_output)
        doc_context = self.sentence_dropout(doc_context)
        
        # 分类
        logits = self.classifier(doc_context)
        return logits, word_weights, sentence_weights


def demo():
    batch, n_sents, seq_len, vocab_size = 4, 5, 20, 5000
    model = HAN(vocab_size, embed_size=100, hidden_size=32, n_classes=5)
    x = torch.randint(0, vocab_size, (batch, n_sents, seq_len))
    logits, word_attn, sent_attn = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {logits.shape}")
    print(f"词注意力: {word_attn.shape}")
    print(f"句子注意力: {sent_attn.shape}")


if __name__ == "__main__":
    demo()
```

---

## 4. 应用场景

1. **情感分析：** 识别文档中情感倾向最强的句子和关键词
2. **新闻分类：** 关注新闻中的关键段落
3. **文章摘要：** 提取最重要的句子生成摘要

---

## 5. 学习路径

**前置：** Seq2Seq、注意力机制、双向GRU/LSTM
**平行：** Transformer
**进阶：** BERT中的自注意力、Longformer等长文档模型