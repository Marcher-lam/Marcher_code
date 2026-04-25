# DRNN 学习文档

> DRNN (Deep RNN) 深度循环神经网络是一类能够处理序列数据的深度学习架构,通过堆叠多层RNN/LSTM/GRU单元来增强模型的表达能力。

---

## 1. 算法基础认知

### 一句话定义
DRNN 通过堆叠多个循环层,增强网络对长期依赖关系的建模能力,是处理序列数据的强大工具。

### 直觉类比
想象阅读一本小说:
- **浅层RNN**:只能记住上一段的内容
- **深层RNN**:能够记住前几章甚至全书的情节
- 堆叠层数越多,"记忆"越深入持久

### 历史背景
- 1986年,Elman提出Jordan/Elman网络
- 1997年,Hochreiter和Schmidhuber提出LSTM
- 2014年,GRU简化LSTM
- 深度化:堆叠多层LSTM/GRU

### 算法定位
- **类型**:序列模型/深度学习
- **输出**:序列标注/分类/生成
- **模型类型**:多层LSTM/GRU

### 前置知识
- 基础RNN/LSTM
- 反向传播 Through Time (BPTT)
- 深度学习优化

---

## 2. 核心原理

### 2.1 核心思想
DRNN核心是**多层堆叠**:

1. **第一层**:处理原始输入序列
2. **后续层**:处理前一层的输出
3. **信息逐层抽象**:底层捕获细粒度特征,顶层捕获宏观模式

### 2.2 工作流程
```
x_t → LSTM层1 → h_t^1 → LSTM层2 → h_t^2 → ... → 输出
```

### 2.3 关键概念
- **时间步展开**:BPTT沿时间展开
- **隐藏状态传递**:每层 $h_t^l$ 传递信息
- **梯度流**:多层可能面临梯度消失/爆炸

### 2.4 架构图
```
┌─────────────────────────────────────────────┐
│          深度RNN架构                        │
│                                             │
│ 输入 x_t    h_t^(l-1)    输出 y_t           │
│   ↓          ↓             ↑                │
│ ┌─────┐    ┌─────┐        ┌─────┐           │
│ │LSTM│──→ │LSTM│───→ ...→│Dense│           │
│ │层1 │    │层2 │        │     │            │
│ └─────┘    └─────┘        └─────┘           │
│   ↑          ↑                              │
│ h_t^1      h_t^2                           │
│                                             │
│ 多层信息逐层抽象                            │
└─────────────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $x_t$ | t时刻输入 |
| $h_t^l$ | 第l层t时刻隐藏状态 |
| $c_t^l$ | 第l层t时刻记忆单元 |
| $W, U, b$ | 权重和偏置 |

### 3.2 多层LSTM

**第一层**:
$$i_t^1 = \sigma(W^i x_t + U^i h_{t-1}^1 + b^i)$$
$$f_t^1 = \sigma(W^f x_t + U^f h_{t-1}^1 + b^f)$$
$$o_t^1 = \sigma(W^o x_t + U^o h_{t-1}^1 + b^o)$$
$$\tilde{c}_t^1 = \tanh(W^c x_t + U^c h_{t-1}^1 + b^c)$$
$$c_t^1 = f_t^1 \odot c_{t-1}^1 + i_t^1 \odot \tilde{c}_t^1$$
$$h_t^1 = o_t^1 \odot \tanh(c_t^1)$$

**第l层 (l>1)**:
用 $h_t^{l-1}$ 作为输入。

### 3.3 多层GRU

$$z_t^l = \sigma(W^z h_t^{l-1} + U^z h_{t-1}^l)$$
$$r_t^l = \sigma(W^r h_t^{l-1} + U^r h_{t-1}^l)$$
$$\tilde{h}_t^l = \tanh(W^h h_t^{l-1} + U^r (r_t^l \odot h_{t-1}^l))$$
$$h_t^l = (1-z_t^l) \odot h_{t-1}^l + z_t^l \odot \tilde{h}_t^l$$

### 3.4 训练目标

$$\mathcal{L} = -\sum_t \log P(y_t|x_{1:T})$$

使用BPTT反向传播。

---

## 4. 训练过程

### 4.1 实现代码

```python
"""
DRNN 完整实现 (PyTorch)
"""

import torch
import torch.nn as nn
import numpy as np

class DeepLSTM(nn.Module):
    """深度LSTM"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.0):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 多层LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后时刻的输出
        out = self.fc(lstm_out[:, -1, :])
        return out


class DeepGRU(nn.Module):
    """深度GRU"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.0):
        super().__init__()
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out


class Seq2SeqEncoder(nn.Module):
    """序列到序列编码器"""
    
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        # 合并双向最后隐藏状态
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return outputs, hidden


class BidirectionalDRNN(nn.Module):
    """双向深度RNN"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 双向: hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 取最后时刻的bidirectional输出
        out = self.fc(lstm_out[:, -1, :])
        return out


class AttentionDRNN(nn.Module):
    """带注意力机制的深度RNN"""
    
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embed_dim)
        
        # 编码器
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            embed_dim + hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        
        # 注意力
        self.attention = nn.Linear(hidden_dim, 1)
        
        # 输出
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src, tgt):
        # 编码
        src_embed = self.embedding(src)
        encoder_outputs, _ = self.encoder(src_embed)
        
        # 解码
        tgt_embed = self.embedding(tgt)
        decoder_outputs = []
        
        for t in range(tgt_embed.size(1)):
            # 简单注意力
            attn_scores = self.attention(encoder_outputs).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            
            context = (attn_weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)
            
            decoder_input = torch.cat([tgt_embed[:, t], context], dim=1).unsqueeze(1)
            decoder_out, _ = self.decoder(decoder_input)
            decoder_outputs.append(decoder_out)
        
        outputs = torch.cat(decoder_outputs, dim=1)
        return self.fc(outputs)


def train_drnn(model, dataloader, epochs=10, lr=1e-3, device='cuda'):
    """训练"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model
```

### 4.2 训练要点
- 梯度裁剪防止爆炸
- 双向通常效果更好
- 层数选择(2-4层常见)

### 4.3 超参数

| 参数 | 推荐范围 |
|------|----------|
| hidden_dim | 128-512 |
| num_layers | 2-4 |
| dropout | 0.1-0.3 |
| lr | 1e-4-1e-3 |

---

## 5. 应用场景

### 5.1 典型应用
- **语言模型**:单词预测
- **机器翻译**: Seq2Seq
- **语音识别**:时序建模
- **时间序列预测**

### 5.2 适用数据
- 序列数据
- 长期依赖
- 变长输入输出

---

## 6. 优缺点

### 6.1 优点
| 优点 | 说明 |
|------|------|
| 长期依赖 | LSTM/GRU门控 |
| 表达力强 | 多层抽象 |
| 灵活架构 | Seq2Seq等 |

### 6.2 缺点
| 缺点 | 缓解 |
|------|------|
| 慢(序列长度) | Attention |
| 梯度问题 | 梯度裁剪,层归一化 |
| 并行化差 | Truncated BPTT |

---

## 7. 调库实现

```python
"""
Hugging Face Transformers
"""

from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 使用
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

---

## 8. 手工实现

```python
"""
DRNN 核心简化版
"""

import torch
import torch.nn as nn

class SimpleDRNN(nn.Module):
    """简化深度RNN"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        
        # 多层RNN
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.1 * (num_layers > 1)
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (batch, seq, input)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class SimpleLSTMClassifier(nn.Module):
    """简单序列分类器"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        # 取最后一层最后时刻
        return self.fc(h_n[-1])
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_attention(attention_weights, words, save_path='attn.png'):
    """可视化注意力"""
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='Blues')
    plt.xticks(range(len(words)), words, rotation=45)
    plt.yticks(range(len(words)), words)
    plt.colorbar()
    plt.savefig(save_path)
    plt.show()


def plot_loss_curve(losses, save_path='loss.png'):
    """训练曲线"""
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.show()
```

---

## 10. 评估

```python
from sklearn.metrics import accuracy_score, f1_score

def evaluate_classifier(model, dataloader):
    """评估"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro')
    }
```

---

## 11. 常见问题

### 11.1 梯度爆炸
- 使用梯度裁剪
- 降低学习率

### 11.2 过拟合
- 添加dropout
- 减少层数

---

## 12. 总结

### 核心要点
1. **多层堆叠**:2-4层
2. **LSTM/GRU**:门控机制
3. **BPTT**:时间展开
4. **双向**:增强上下文

### 算法链
```
DRNN → BiLSTM → Attention → Transformer(自注意力)
    → Encoder-Decoder
```

---

## 13. 练习题

**习题1**: 多层LSTM计算

<details>
<summary>答案</summary>

第l层用第l-1层的输出作为输入,遵循标准LSTM公式。

</details>

**习题2**: 梯度裁剪原因

<details>
<summary>答案</summary>

防止BPTT时梯度指数级放大导致数值溢出。

</details>

---

## 14. 学习路径

- **初级**: 理解LSTM/GRU,序列分类
- **中级**: Seq2Seq,机器翻译
- **高级**: Transformer,预训练语言模型

### 推荐资源
- **论文**: Hochreiter & Schmidhuber "Long Short-Term Memory" (1997)
- **课程**: CS224N (Stanford)