# 编码器-解码器 (Encoder-Decoder) 学习文档

> 用一句话说明这个算法的核心价值，不超过30字。

Encoder-Decoder是序列到序列学习的基础架构，通过编码器理解输入序列，通过解码器生成输出，广泛应用于机器翻译、对话生成等任务。

---

## 1. 算法基础认知

### 1.1 什么是Encoder-Decoder

Encoder-Decoder（编码器-解码器）是一种深度学习架构，用于处理序列到序列（Sequence-to-Sequnece, Seq2Seq）的转换任务。核心思想是：
- **编码器**（Encoder）：将输入序列编码为稠密的向量表示（上下文向量）
- **解码器**（Decoder）：基于编码器输出的上下文向量，逐步生成输出序列

这种架构的革命性意义在于：它可以处理输入输出长度不同的序列任务，这是传统神经网络无法做到的。

### 1.2 直觉类比

把Encoder-Decoder想象成一个翻译员：
1. **编码器**（ Encoder）：当你用中文讲述一个故事时，翻译员认真聆听并理解整个故事的内容，把信息存储在脑海中
2. **上下文向量**：这就像翻译员对故事的理解总结
3. **解码器**（Decoder）：翻译员根据脑中存储的理解，用英文重新组织并表达出来

在这个过程中，翻译员必须完整听完整个故事（输入），才能开始翻译（输出），并且需要记住整个故事的内容（上下文向量）。

### 1.3 历史背景

Encoder-Decoder架构由Cho等人在2014年提出论文《Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation》。同一年，Google发布了基于此架构的神经机器翻译系统，效果远超传统的统计机器翻译。

2015年，Sutskever等人提出将Encoder-Decoder与Attention机制结合，进一步提升了长序列翻译的效果，奠定了现代神经机器翻译的基础。

### 1.4 算法定位

Encoder-Decoder是**序列到序列学习**的**基础架构**，属于监督学习，广泛应用于：
- 机器翻译
- 文本摘要
- 对话系统
- 代码生成
- 语音识别

### 1.5 前置知识

- 循环神经网络（RNN）/LSTM/GRU
- 词嵌入（Word Embedding）
- 注意力机制（Attention）
- 深度学习基础

---

## 2. 核心原理

### 2.1 核心思想

Encoder-Decoder的核心是两个神经网络：
- **编码器**：$h_t = f(x_t, h_{t-1})$，将输入序列编码为隐藏状态
- **解码器**：$y_t = g(y_{t-1}, c)$，基于上下文向量生成输出

最终输出：$y = (y_1, y_2, ..., y_T)$

### 2.2 工作流程

1. **编码阶段**：
   - 输入序列：$x = (x_1, x_2, ..., x_n)$
   - 逐步更新隐藏状态：$h_t = f(x_t, h_{t-1})$
   - 最终隐藏状态 $h_n$ 作为上下文向量 $c$

2. **解码阶段**：
   - 初始化：$s_0 = h_n$
   - 逐步生成：$y_t = g(y_{t-1}, s_{t-1}, c)$
   - 输出序列：$y = (y_1, y_2, ..., y_m)$

### 2.3 关键概念

**上下文向量（Context Vector）**：
- 编码器最后一个隐藏状态
- 包含了整个输入序列的信息
- 是解码器的唯一输入

**教师强制（Teacher Forcing）**：
- 训练时使用真实标签作为输入
- 加速训练但可能导致暴露偏差

**暴露偏差（Exposure Bias）**：
- 训练时用真实标签，测试时用预测标签
- 训练和测试分布不一致
- 缓解方法：Scheduled Sampling

### 2.4 几何解释

```
输入序列: [x1, x2, x3, x4, x5]
           ↓  ↓  ↓  ↓  ↓
编码器:   h1→h2→h3→h4→h5
           ↓
上下文c:  (h5)
           ↓
解码器:   s0→s1→s2→s3→s4
           ↓  ↓  ↓  ↓  ↓
输出序列: [y1, y2, y3, y4, y5]
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $x$ | 输入序列 $(x_1, x_2, ..., x_n)$ |
| $y$ | 输出序列 $(y_1, y_2, ..., y_m)$ |
| $h$ | 编码器隐藏状态 |
| $s$ | 解码器隐藏状态 |
| $c$ | 上下文向量 |
| $f$ | 编码器RNN |
| $g$ | 解码器RNN |
| $E$ | 词嵌入矩阵 |

### 3.2 编码器数学表示

对于输入序列 $x = (x_1, x_2, ..., x_n)$：

**词嵌入**：
$$e_t = E[x_t]$$

**RNN更新**：
$$h_t = f(e_t, h_{t-1})$$

如果使用LSTM：
$$h_t, c_t = LSTM(e_t, h_{t-1}, c_{t-1})$$

最终上下文向量：
$$c = h_n$$

### 3.3 解码器数学表示

基于上下文 $c$ 和之前生成的词 $(y_1, ..., y_{t-1})$：

**词嵌入**：
$$e_{t-1} = E[y_{t-1}]$$

**解码器RNN**：
$$s_t = g(e_{t-1}, s_{t-1}, c)$$

**输出分布**：
$$P(y_t|y_{<t}, x) = Softmax(W \cdot s_t)$$

**预测**：
$$\hat{y}_t = \arg\max P(y_t|y_{<t}, x)$$

### 3.4 损失函数

Seq2Seq使用交叉熵损失：

$$L = -\sum_{t=1}^{m} \log P(y_t|y_{<t}, x)$$

展开：
$$L = -\sum_{t=1}^{m} \log \frac{\exp(s_t^T W_{y_t})}{\sum_j \exp(s_t^T W_j)}$$

### 3.5 束搜索（Beam Search）

解码时使用束搜索获得更好的输出：

```python
def beam_searchdecoder(c, beam_size=3, max_len=20):
    """束搜索解码"""
    beams = [(0, [START_TOKEN])]  # (score, sequence)
    
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == END_TOKEN:
                candidates.append((score, seq))
                continue
            
            s = get_decoder_state(seq, c)
            probs = softmax(s)
            top_k = np.argsort(probs)[-beam_size:]
            
            for idx in top_k:
                new_score = score + np.log(probs[idx])
                new_seq = seq + [idx]
                candidates.append((new_score, new_seq))
        
        beams = sorted(candidates, key=lambda x: x[0])[:beam_size]
    
    return beams[0][1]
```

---

## 4. 训练过程讲解

### 4.1 数据准备

```python
import torch
from torch.utils.data import Dataset, DataLoader

class Seq2SeqDataset(Dataset):
    """Seq2Seq数据集"""
    
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src = self.src_tokenizer.encode(self.src_texts[idx])
        tgt = self.tgt_tokenizer.encode(self.tgt_texts[idx])
        return {
            'src': torch.tensor(src),
            'tgt': torch.tensor(tgt)
        }

def collate_fn(batch):
    """处理变长序列"""
    srcs = [item['src'] for item in batch]
    tgts = [item['tgt'] for item in batch]
    
    src_lens = torch.tensor([len(s) for s in srcs])
    tgt_lens = torch.tensor([len(t) for t in tgts])
    
    srcs = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True)
    tgts = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True)
    
    return srcs, tgts, src_lens, tgt_lens
```

### 4.2 模型定义

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """编码器"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, 
                        batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        outputs, (h, c) = self.rnn(embedded)
        # outputs: (batch, seq_len, hidden_dim)
        # h: (n_layers, batch, hidden_dim)
        return outputs, (h, c)

class Decoder(nn.Module):
    """解码器"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                        batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        # x: (batch, 1) - 上一时刻的输出
        # hidden: (h, c)
        embedded = self.embedding(x)  # (batch, 1, embed_dim)
        output, hidden = self.rnn(embedded, hidden)
        # output: (batch, 1, hidden_dim)
        logits = self.fc(output.squeeze(1))  # (batch, vocab_size)
        return logits, hidden

class Seq2SeqModel(nn.Module):
    """完整的Seq2Seq模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 embed_dim=256, hidden_dim=512, n_layers=1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, n_layers)
        
        # 初始化上下文
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
    def forward(self, src, tgt):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)
        
        # 编码
        _, (h, c) = self.encoder(src)
        
        # 解码（Teacher Forcing）
        outputs = []
        for t in range(tgt.size(1)):
            input_t = tgt[:, t:t+1]  # (batch, 1)
            logits, (h, c) = self.decoder(input_t, (h, c))
            outputs.append(logits)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, tgt_len, vocab_size)
        return outputs
    
    def encode(self, src):
        return self.encoder(src)
    
    def decode(self, x, hidden, context):
        return self.decoder(x, hidden)
```

### 4.3 训练循环

```python
import torch.optim as optim

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        src, tgt, _, _ = batch
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(src, tgt[:, :-1])  # 输入不包括最后一个token
        
        # 计算损失
        # outputs: (batch, tgt_len-1, vocab_size)
        # tgt: (batch, tgt_len)
        loss = criterion(outputs.view(-1, outputs.size(-1)), 
                       tgt[:, 1:].view(-1))
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """评估"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, _, _ = batch
            src = src.to(device)
            tgt = tgt.to(device)
            
            outputs = model(src, tgt[:, :-1])
            loss = criterion(outputs.view(-1, outputs.size(-1)), 
                          tgt[:, 1:].view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

---

## 5. 应用场景

### 5.1 机器翻译

```python
class TranslationDemo:
    """机器翻译示例"""
    
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.src_tokenizer = GPT2Tokenizer()
        self.tgt_tokenizer = GPT2Tokenizer()
    
    @torch.no_grad()
    def translate(self, text, max_length=100):
        # 编码输入
        src = self.src_tokenizer.encode(text).unsqueeze(0)
        
        # 编码
        _, (h, c) = self.model.encode(src)
        
        # 解码
        result = [self.tgt_tokenizer.bos_id]
        for _ in range(max_length):
            input_t = torch.tensor([[result[-1]]])
            logits, (h, c) = self.model.decode(input_t, (h, c))
            next_token = logits.argmax(-1).item()
            result.append(next_token)
            if next_token == self.tgt_tokenizer.eos_id:
                break
        
        return self.tgt_tokenizer.decode(result)
```

### 5.2 文本摘要

```python
class SummarizationModel(nn.Module):
    """文本摘要模型"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size, 256, 512, n_layers=2)
        self.decoder = Decoder(vocab_size, 256, 512, n_layers=2)
        self.fc = nn.Linear(512, vocab_size)
    
    def forward(self, src, tgt):
        _, (h, c) = self.encoder(src)
        
        outputs = []
        for t in range(tgt.size(1)):
            input_t = tgt[:, t:t+1]
            out, (h, c) = self.decoder(input_t, (h, c))
            out = self.fc(out)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)
```

### 5.3 对话系统

```python
class ChatbotModel(nn.Module):
    """对话系统"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size, 512, 1024, n_layers=2)
        self.decoder = Decoder(vocab_size, 512, 1024, n_layers=2)
    
    def forward(self, src, tgt):
        _, encoder_hidden = self.encoder(src)
        outputs = []
        
        # 添加上下文向量
        decoder_hidden = encoder_hidden
        
        for t in range(tgt.size(1)):
            input_t = tgt[:, t:t+1]
            logits, decoder_hidden = self.decoder(input_t, decoder_hidden)
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)
```

### 5.4 代码生成

```python
class CodeGenerationModel(nn.Module):
    """代码生成（Python）"""
    
    def __init__(self, vocab_size):
        super().__init__()
        # 特殊：使用双向上下文
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        self.vocab_size = vocab_size
    
    def forward(self, src, tgt):
        # 编码
        memory = self.encoder(src)
        
        # 解码
        output = self.decoder(tgt, memory)
        return self.fc(output)
```

### 5.5 语音识别

```python
class SpeechRecognitionModel(nn.Module):
    """语音识别（ASR）"""
    
    def __init__(self, n_mels, vocab_size):
        super().__init__()
        # 声学模型： CNN + Transformer
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        
        # Encoder处理声学特征
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        
        # Decoder输出文本
        self.decoder = Decoder(vocab_size, 256, 512)
    
    def forward(self, x, text):
        # x: (batch, n_mels, time)
        x = x.unsqueeze(1)  # 添加通道维
        x = self.conv(x)   # 声学特征提取
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        
        # Encoder
        encoder_output = self.encoder(x)
        
        # Seq2Seq Decode
        outputs = []
        h, c = None, None
        
        for t in range(text.size(1)):
            input_t = text[:, t:t+1]
            if h is None:
                h = encoder_output[:, -1:]
            logits, (h, c) = self.decoder(input_t, (h, c))
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 通用性 | 处理任意序列到序列任务 |
| 端到端 | 无需特征工程 |
| 可扩展 | 可以与各种技术结合 |
| 效果好 | 超越传统方法 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 暴露偏差 | 训练测试不一致 | Scheduled Sampling |
| 梯度消失 | 长序列问题 | Attention、Transformer |
| 信息瓶颈 | 所有信息压缩到一个向量 | 多层编码、Attention |
| 计算量大 | 并行化困难 | Transformer |

### 6.3 架构对比

| 特性 | RNN Encoder-Decoder | Transformer | CNN Seq2Seq |
|------|-------------------|-------------|------------|
| 并行化 | 差 | 好 | 好 |
| 长序列 | 一般 | 好 | 好 |
| 参数量 | 中 | 大 | 中 |
| 计算量 | 中 | 大 | 中 |

---

## 7. 调库实现

### 7.1 PyTorch官方实现

```python
import torch
import torch.nn as nn

# 使用PyTorch内置的Seq2Seq模型
class Seq2SeqTransformer(nn.Module):
    """Transformer Seq2Seq模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    
    def forward(self, src, tgt):
        src = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model))
        tgt = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model))
        
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
        
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.fc(output)

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 训练代码
def train_transformer():
    """训练Transformer"""
    from torch.utils.data import DataLoader
    
    # 数据
    from datasets import load_dataset
    dataset = load_dataset('wmt14', 'fr-en', split='train[:1%]')
    
    # Tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 模型
    model = Seq2SeqTransformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 训练
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in DataLoader(dataset, batch_size=32):
            src = tokenizer(batch['en'], return_tensors='pt')['input_ids']
            tgt = tokenizer(batch['fr'], return_tensors='pt')['input_ids']
            
            optimizer.zero_grad()
            outputs = model(src, tgt[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss/len(dataset):.4f}")

train_transformer()
```

### 7.2 Hugging Face Transformers实现

```python
from transformers import EncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 使用预训练模型
model = EncoderDecoderModel.from_pretrained("bert-base-uncased", "bert-base-uncased")

# 设置tokenizer
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model.config.pad_token_id = tokenizer.pad_token_id

# 配置
model.config.encoder.hidden_size = 768
model.config.decoder.hidden_size = 768
model.config.decoder.n_features = model.config.encoder.hidden_size

# 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    fp16=True,
    save_steps=1000,
    save_total_limit=2,
)

# 训练器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 训练
trainer.train()

# 使用
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    generated_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

### 7.3 小数据训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    """简单的Seq2Seq数据集"""
    
    def __init__(self):
        self.data = [
            ("hello", "你好"),
            ("how are you", "我很好"),
            ("thank you", "谢谢"),
            ("goodbye", "再见"),
            ("what is your name", "我叫机器人"),
        ]
        
        # 构建词表
        self.src_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.tgt_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        
        for src, tgt in self.data:
            for word in src.split():
                if word not in self.src_vocab:
                    self.src_vocab[word] = len(self.src_vocab)
            for word in tgt.split():
                if word not in self.tgt_vocab:
                    self.tgt_vocab[word] = len(self.tgt_vocab)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = [self.src_vocab[w] for w in src.split()]
        tgt_ids = [self.tgt_vocab[w] for w in tgt.split()] + [self.tgt_vocab['<EOS>']]
        return {
            'src': torch.tensor(src_ids),
            'tgt': torch.tensor([self.tgt_vocab['<SOS>']] + tgt_ids)
        }

# 数据
dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 模型
class SimpleSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)
    
    def forward(self, src, tgt):
        # 编码
        src_emb = self.src_embed(src)
        _, (h, c) = self.encoder(src_emb)
        
        # 解码
        tgt_emb = self.tgt_embed(tgt)
        out, _ = self.decoder(tgt_emb, (h, c))
        
        return self.fc(out)

# 训练
model = SimpleSeq2Seq(
    src_vocab_size=len(dataset.src_vocab),
    tgt_vocab_size=len(dataset.tgt_vocab)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0)

model.train()
for epoch in range(200):
    total_loss = 0
    for batch in dataloader:
        src, tgt = batch['src'], batch['tgt']
        
        optimizer.zero_grad()
        out = model(src, tgt[:, :-1])
        loss = criterion(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataset):.4f}")

# 测试
def translate(word):
    model.eval()
    src_ids = torch.tensor([dataset.src_vocab.get(word, 0)])
    src_emb = model.src_embed(src_ids.unsqueeze(0))
    _, (h, c) = model.encoder(src_emb)
    
    result = [dataset.tgt_vocab['<SOS>']]
    for _ in range(10):
        tgt_tensor = torch.tensor([[result[-1]]])
        tgt_emb = model.tgt_embed(tgt_tensor)
        out, (h, c) = model.decoder(tgt_emb, (h, c))
        next_word = out.argmax(-1).item()
        result.append(next_word)
        if next_word == dataset.tgt_vocab['<EOS>']:
            break
    
    for word, idx in dataset.tgt_vocab.items():
        if idx in result:
            print(word, end=' ')
    print()

translate("hello")
translate("thank you")
```

---

## 8. 手工代码实现

### 8.1 基础RNN Seq2Seq

```python
import numpy as np
import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    """RNN编码器实现"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, hidden = self.rnn(embedded)
        # output: (batch, seq_len, hidden_dim)
        # hidden: (1, batch, hidden_dim)
        return output, hidden

class RNNDecoder(nn.Module):
    """RNN解码器实现"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        # x: (batch, 1)
        embedded = self.embedding(x)  # (batch, 1, embed_dim)
        output, hidden = self.rnn(embedded, hidden)
        # output: (batch, 1, hidden_dim)
        logits = self.fc(output.squeeze(1))  # (batch, vocab_size)
        return logits, hidden

class SimpleSeq2Seq(nn.Module):
    """简单的Seq2Seq模型"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = RNNEncoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = RNNDecoder(vocab_size, embed_dim, hidden_dim)
    
    def forward(self, src, tgt):
        # 编码
        _, hidden = self.encoder(src)
        
        # 解码（Teacher Forcing）
        outputs = []
        for t in range(tgt.size(1)):
            logits, hidden = self.decoder(tgt[:, t:t+1], hidden)
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)
```

### 8.2 完整的LSTM Seq2Seq

```python
import torch
import torch.nn as nn

class LSTMSeq2Seq(nn.Module):
    """LSTM Seq2Seq模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 embed_dim=256, hidden_dim=512, n_layers=2, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 编码器
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # 投影层（双向->单向）
        self.encoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 解码器
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.decoder = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout
        )
        
        # 输出层
        self.output_projection = nn.Linear(hidden_dim, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        
        # 编码
        src_embedded = self.dropout(self.src_embedding(src))
        encoder_outputs, (h, c) = self.encoder(src_embedded)
        
        # 处理双向隐藏状态
        h = self.encoder_projection(torch.cat([h[-2], h[-1]], dim=-1))
        c = self.encoder_projection(torch.cat([c[-2], c[-1]], dim=-1))
        h = h.unsqueeze(0).repeat(self.n_layers, 1, 1)
        c = c.unsqueeze(0).repeat(self.n_layers, 1, 1)
        
        # 解码
        decoder_hidden = (h, c)
        outputs = []
        
        input_t = tgt[:, 0]  # Start token
        for t in range(max_len):
            embedded = self.dropout(self.tgt_embedding(input_t))
            decoder_output, decoder_hidden = self.decoder(
                embedded.unsqueeze(1), decoder_hidden
            )
            prediction = self.output_projection(decoder_output.squeeze(1))
            outputs.append(prediction)
            
            # Teacher forcing
            if np.random.random() < teacher_forcing_ratio:
                input_t = tgt[:, t]
            else:
                input_t = prediction.argmax(-1)
        
        return torch.stack(outputs, dim=1)
```

### 8.3 Greedy解码实现

```python
@torch.no_grad()
def greedy_decode(model, src, max_len=50, start_token=1, end_token=2):
    """贪婪解码"""
    model.eval()
    
    # 编码
    src_embedded = model.src_embedding(src)
    _, (h, c) = model.encoder(src_embedded)
    
    # 初始化
    result = [start_token]
    decoder_hidden = (h, c)
    
    for _ in range(max_len):
        input_t = torch.tensor([result[-1]]).long()
        embedded = model.tgt_embedding(input_t)
        
        decoder_output, decoder_hidden = model.decoder(
            embedded.unsqueeze(0), decoder_hidden
        )
        prediction = model.output_projection(decoder_output.squeeze(0))
        
        next_token = prediction.argmax(-1).item()
        result.append(next_token)
        
        if next_token == end_token:
            break
    
    return result
```

### 8.4 Beam Search解码实现

```python
@torch.no_grad()
def beam_search_decode(model, src, max_len=50, beam_size=5, 
                       start_token=1, end_token=2):
    """束搜索解码"""
    model.eval()
    
    # 编码
    src_embedded = model.src_embedding(src)
    encoder_outputs, (h, c) = model.encoder(src_embedded)
    decoder_hidden = (h, c)
    
    # 初始化beam
    beams = [(0.0, [start_token], decoder_hidden)]
    completed = []
    
    for _ in range(max_len):
        candidates = []
        
        for score, seq, hidden in beams:
            if seq[-1] == end_token:
                completed.append((score, seq))
                continue
            
            input_t = torch.tensor([seq[-1]]).long()
            embedded = model.tgt_embedding(input_t)
            
            decoder_output, new_hidden = model.decoder(
                embedded.unsqueeze(0), hidden
            )
            prediction = model.output_projection(decoder_output.squeeze(0))
            
            # Top-k
            probs = torch.softmax(prediction, dim=-1)
            topk_probs, topk_ids = probs.topk(beam_size)
            
            for prob, token_id in zip(topk_probs, topk_ids):
                new_score = score + torch.log(prob + 1e-10).item()
                new_seq = seq + [token_id.item()]
                candidates.append((new_score, new_seq, new_hidden))
        
        # 选择top-beam_size
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        beams = candidates
    
    # 返回最佳结果
    all_results = beams + completed
    best = sorted(all_results, key=lambda x: x[0], reverse=True)[0]
    return best[1]
```

### 8.5 带Attentino的Seq2Seq

```python
import torch
import torch.nn as nn

class AttentionSeq2Seq(nn.Module):
    """带Attention的Seq2Seq"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 embed_dim=256, hidden_dim=512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 编码器
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Attention
        self.attention = Attention(hidden_dim)
        
        # 解码器
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.decoder = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        
        # 输出层
        self.output = nn.Linear(hidden_dim, tgt_vocab_size)
    
    def forward(self, src, tgt):
        # 编码
        src_embedded = self.src_embedding(src)
        encoder_outputs, (h, c) = self.encoder(src_embedded)
        
        # 解码
        decoder_outputs = []
        for t in range(tgt.size(1)):
            # 计算attention
            context = self.attention(h, encoder_outputs)
            
            # 解码器输入
            tgt_embedded = self.tgt_embedding(tgt[:, t])
            decoder_input = torch.cat([tgt_embedded, context.squeeze(0)], dim=-1).unsqueeze(1)
            
            # 解码
            decoder_output, (h, c) = self.decoder(decoder_input, (h, c))
            decoder_outputs.append(decoder_output)
        
        output = torch.cat(decoder_outputs, dim=1)
        return self.output(output)

class Attention(nn.Module):
    """Bahdanau Attention"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (1, batch, hidden_dim)
        # encoder_outputs: (batch, src_len, hidden_dim)
        
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # 扩展decoder_hidden
        decoder_hidden = decoder_hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        
        # 计算attention score
        score = self.V(torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs)))
        attention_weights = torch.softmax(score, dim=1)
        
        # context vector
        context = torch.sum(attention_weights * encoder_outputs, dim=1)
        
        return context.unsqueeze(1)
```

---

## 9. 可视化与结果理解

### 9.1 训练可视化

```python
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses):
    """绘制训练曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### 9.2 注意力可视化

```python
import seaborn as sns

def visualize_attention(attention_weights, src_tokens, tgt_tokens):
    """可视化注意力权重"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # attention_weights: (tgt_len, src_len)
    sns.heatmap(attention_weights, 
                xticklabels=src_tokens, 
                yticklabels=tgt_tokens,
                cmap='Blues',
                ax=ax)
    
    ax.set_xlabel('Source Tokens')
    ax.set_ylabel('Target Tokens')
    ax.set_title('Attention Weights')
    
    plt.tight_layout()
    plt.show()
```

### 9.3 架构可视化

```python
def visualize_architecture():
    """可视化Encoder-Decoder架构"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # 编码器
    encoder_box = dict(boxstyle='round,pad=0.5', fc='lightblue', ec='blue')
    ax.text(0.15, 0.5, 'Encoder\n(RNN/LSTM/BERT)', fontsize=14,
            ha='center', va='center', bbox=encoder_box)
    
    # 上下文向量
    ax.text(0.4, 0.5, '→', fontsize=20, ha='center', va='center')
    ax.text(0.4, 0.35, 'Context\nVector', fontsize=10,
           ha='center', va='center')
    
    # 解码器
    ax.text(0.65, 0.5, 'Decoder\n(RNN/LSTM/GPT)', fontsize=14,
            ha='center', va='center', bbox=encoder_box)
    
    # 输出
    ax.text(0.85, 0.5, '→', fontsize=20, ha='center', va='center')
    ax.text(0.92, 0.5, 'Output', fontsize=12, ha='center', va='center')
    
    # 流程线
    arrows = dict(arrowstyle='->', color='gray', lw=2)
    ax.annotate('', xy=(0.25, 0.5), xytext=(0.05, 0.5), arrowprops=arrows)
    ax.annotate('', xy=(0.55, 0.5), xytext=(0.45, 0.5), arrowprops=arrows)
    ax.annotate('', xy=(0.75, 0.5), xytext=(0.65, 0.5), arrowprops=arrows)
    ax.annotate('', xy=(0.88, 0.5), xytext=(0.82, 0.5), arrowprops=arrows)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Encoder-Decoder Architecture', fontsize=16)
    
    plt.tight_layout()
    plt.show()
```

### 9.4 生成质量评估

```python
import matplotlib.pyplot as plt

def plot_generation_quality():
    """比较不同解码策略的生成质量"""
    strategies = ['Greedy', 'Beam Search k=3', 'Beam Search k=5', 'Top-k', 'Top-p']
    scores = [0.65, 0.78, 0.82, 0.75, 0.80]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(strategies, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom')
    
    ax.set_ylabel('Quality Score')
    ax.set_title('Decoding Strategy Comparison')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
```

### 9.5 Seq2Seq应用场景图

```python
def visualize_applications():
    """可视化应用场景"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    applications = [
        ('Machine\nTranslation', 0.15, 0.85),
        ('Text\nSummarization', 0.45, 0.85),
        ('Dialogue\nSystem', 0.75, 0.85),
        ('Code\nGeneration', 0.15, 0.5),
        ('Speech\nRecognition', 0.45, 0.5),
        ('Image\nCaption', 0.75, 0.5),
    ]
    
    box = dict(boxstyle='round,pad=0.5', fc='lightgreen', ec='green')
    
    for app, x, y in applications:
        ax.text(x, y, app, fontsize=12, ha='center', va='center', bbox=box)
    
    # 中心
    ax.text(0.5, 0.15, 'Seq2Seq\nLearning', fontsize=14, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', ec='blue'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Seq2Seq Applications', fontsize=16)
    
    plt.tight_layout()
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 备注 |
|------|------|------|
| BLEU | N-gram重叠度 | 机器翻译常用 |
| ROUGE | N-gram召回率 | 摘要常用 |
| Perplexity | 困惑度 | 语言模型 |
| Meteor | 同义词��配 | 语义评估 |
| CIDEr | 图像描述 | 图像字幕 |

### 10.2 BLEU计算

```python
from collections import Counter
import numpy as np

def calculate_bleu(prediction, reference, n_grams=4):
    """计算BLEU分数"""
    
    # 分词
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    
    # 截取长度
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    
    # Brevity penalty
    if pred_len > ref_len:
        bp = 1
    else:
        bp = np.exp(1 - ref_len / max(pred_len, 1))
    
    # N-gram precision
    precisions = []
    for n in range(1, n_grams + 1):
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
        
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        # 匹配
        matches = sum((pred_counter & ref_counter).values())
        
        if len(pred_ngrams) > 0:
            precision = matches / len(pred_ngrams)
        else:
            precision = 0
        
        precisions.append(precision)
    
    # 几何平均
    if all(precisions):
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geo_mean = 0
    
    bleu = bp * geo_mean
    return bleu

# 测试
pred = "the cat sat on the mat"
ref = "the cat is on the mat"
print(f"BLEU: {calculate_bleu(pred, ref):.4f}")
```

### 10.3 Perplexity计算

```python
import torch
import torch.nn.functional as F

def calculate_perplexity(loss):
    """计算困惑度"""
    return torch.exp(torch.tensor(loss)).item()

def evaluate_model(model, dataloader):
    """完整评估"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch['src'], batch['tgt']
            outputs = model(src, tgt[:, :-1])
            
            loss = F.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)),
                tgt[:, 1:].reshape(-1),
                ignore_index=0
            )
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = calculate_perplexity(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }
```

### 10.4 评估脚本

```python
class Seq2SeqEvaluator:
    """Seq2Seq模型评估器"""
    
    def __init__(self, model, tokenizer, metrics=['bleu', 'perplexity']):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = metrics
    
    def translate(self, text):
        """翻译句子"""
        ids = self.tokenizer.encode(text)
        output = greedy_decode(self.model, torch.tensor([ids]))
        return self.tokenizer.decode(output)
    
    def evaluate_batch(self, pairs):
        """评估一批句子"""
        results = {
            'bleu': [],
            'perplexity': []
        }
        
        for src, tgt in pairs:
            pred = self.translate(src)
            
            if 'bleu' in self.metrics:
                bleu = calculate_bleu(pred, tgt)
                results['bleu'].append(bleu)
            
            if 'perplexity' in self.metrics:
                # 计算perplexity
                pass
        
        return results
```

---

## 11. 常见问题与易错点

### 11.1 问题诊断表

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 损失不下降 | 学习率问题 | 调整学习率 |
| 生成重复 | 解码策略 | 使用Beam Search |
| 梯度消失 | 序列太长 | 使用LSTM/Attention |
| OOV问��� | ���表太小 | 扩大词表或使用BPE |
| PAD问题 | 忽略padding | 使用mask |

### 11.2 Teacher Forcing问题

```python
# 问题：暴露偏差
# 解决方法：Scheduled Sampling
teacher_forcing_ratio = max(0.5 - epoch * 0.01, 0)
```

### 11.3 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 11.4 序列长度问题

```python
# 问题：长序列梯度消失
# 解决方法1：使用注意力机制
# 解决方法2：使用Transformer
# 解决方法3：序列截断
max_len = 512
```

---

## 12. 学习总结

### 核心思想

Encoder-Decoder通过编码器将输入序列编码为上下文向量，解码器基于上下文向量生成输出序列，实现任意序列到序列的转换。

### 关键公式

编码器：$h_t = f(x_t, h_{t-1})$

解码器：$y_t = g(y_{t-1}, s_{t-1}, c)$

### 后续学习

1. **Attention**：增强长序列处理能力
2. **Transformer**：完全基于Attention的架构
3. **Pre-training**：BERT + GPT预训练

---

## 13. 练习题与思考题

### 基础题

**题目1**：为什么需要Encoder-Decoder架构？

**答案**：因为输入和输出序列长度可能不同，传统的神经网络无法处理这种情况。

**题目2**：Teacher Forcing的作用是什么？

**答案**：训练时使用真实标签作为解码器输入，可以加速训练但可能导致暴露偏差。

### 进阶题

**题目3**：实现一个Scheduled Sampling。

```python
def scheduled_sampling(epoch, total_epochs):
    """逐渐减少teacher forcing"""
    return max(0, 1 - epoch / total_epochs)
```

### 思考题

**题目4**：Transformer相比RNN Seq2Seq的优势？

**答案**：Transformer可以并行计算，有更强的长距离建模能力。

---

## 14. 学习路径建议

### 前置知识

1. **RNN/LSTM**：理解序列建模
2. **词嵌入**：理解文本表示
3. **注意力机制**：理解信息聚合

### 推荐学习路线

1. **入门**（1-2周）：
   - 理解Seq2Seq架构
   - 实现基础模型

2. **进阶**（2-3周）：
   - Attention机制
   - Beam Search

3. **实践**（持续）：
   - 机器翻译项目
   - 对话系统

### 推荐资源

1. **论文**：
   - Sequence to Sequence Learning
   - Neural Machine Translation

2. **课程**：
   - Stanford CS224N
   - Hugging Face Course

3. **工具**：
   - Hugging Face Transformers
   - OpenNMT

---

