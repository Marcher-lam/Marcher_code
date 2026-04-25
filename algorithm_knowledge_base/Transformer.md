# Transformer 学习文档

## 1. 算法基础认知
基于注意力机制的序列处理模型，包含编码器和解码器。

## 2. 核心原理
自注意力机制允许模型关注输入序列的相关部分。

## 3. 数学公式
注意力：$Attention(Q,K,V) = 	ext{softmax}(rac{QK^T}{\sqrt{d_k}})V$

## 4. 训练过程
编码器-解码器结构，多头注意力机制。

## 5. 应用场景
- 机器翻译
- 文本摘要
- 问答系统

## 6. 优缺点
优点：并行计算、全局依赖；缺点：计算资源需求大

## 7. 代码实现
```python
from transformers import BertModel
model = BertModel.from_pretrained('bert-base-uncased')
```

## 8. 手工实现
```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
    
    def forward(self, src):
        return self.transformer_encoder(src)
```
