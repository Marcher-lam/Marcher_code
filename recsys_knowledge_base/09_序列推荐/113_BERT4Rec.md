# BERT4Rec 学习文档

## 1. 算法基础认知

### 1.1 什么是 BERT4Rec？

BERT4Rec 是阿里巴巴在 2019 年提出的序列推荐模型，将 BERT（Bidirectional Encoder Representations from Transformers）的思想应用于推荐系统。它是 SASRec 的改进版本。

### 1.2 核心思想

- **双向编码**：不同于 SASRec 的单向注意力，BERT4Rec 使用双向上下文
- **Cloze 任务**：使用 Masked Item Prediction 进行训练
- **Transformer**：使用 Transformer 编码器建模用户序列

### 1.3 与 SASRec 的区别

| 维度 | SASRec | BERT4Rec |
|------|--------|----------|
| 注意力方向 | 单向（因果） | 双向 |
| 训练任务 | Next Item Prediction | Masked Item Prediction |
| 预测方式 | 使用最后一个位置 | 使用最后位置 + MASK |
| 上下文 | 只看历史 | 看两边（训练时） |

## 2. 模型架构

### 2.1 整体架构

```
输入序列: [item_1, item_2, ..., item_n]
           ↓
     Item Embeddings + Positional Encodings
           ↓
     ┌─────────────────────────┐
     │   Transformer Encoder   │ ← L 层
     │   - Multi-Head Attention│
     │   - Feed Forward        │
     │   - Layer Norm          │
     └─────────────────────────┘
           ↓
     Hidden States: [h_1, h_2, ..., h_n]
           ↓
     Output Layer (预测 masked 位置)
           ↓
     预测物品概率分布
```

### 2.2 训练 vs 推理

**训练阶段：**
```
输入: [item_1, [MASK], item_3, ..., item_n]
目标: 预测 [MASK] 位置的物品
```

**推理阶段：**
```
输入: [item_1, item_2, ..., item_n, [MASK]]
目标: 预测 [MASK] 位置的物品（即下一个推荐）
```

## 3. PyTorch 完整实现

### 3.1 BERT4Rec 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    位置编码（正弦/余弦）
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        参数:
            query, key, value: (batch, seq_len, d_model)
            mask: (batch, seq_len) 或 (batch, 1, seq_len)

        返回:
            output: (batch, seq_len, d_model)
            attention: (batch, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # 线性变换
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用 mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 加权求和
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        output = self.W_o(output)

        return output, attention


class TransformerBlock(nn.Module):
    """
    Transformer 编码器块
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差 + LayerNorm
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class BERT4Rec(nn.Module):
    """
    BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations

    论文: BERT4Rec: Sequential Recommendation with Bidirectional Encoder
          Representations from Transformer (CIKM 2019)
    """

    def __init__(self, n_items, d_model=64, n_heads=2, n_layers=2,
                 d_ff=256, max_seq_len=100, dropout=0.2,
                 mask_prob=0.15):
        """
        参数:
            n_items: 物品数量
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: Transformer 层数
            d_ff: 前馈网络维度
            max_seq_len: 最大序列长度
            dropout: Dropout 比例
            mask_prob: 训练时的 mask 概率
        """
        super().__init__()

        self.n_items = n_items
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob

        # 物品嵌入
        self.item_embedding = nn.Embedding(n_items + 2, d_model, padding_idx=0)
        # +2 是因为: 0=padding, n_items+1=mask token

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer 编码器
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, n_items + 1)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化
        self._init_weights()

        # 特殊 token
        self.mask_token = n_items + 1

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_masked_input(self, item_seq, mask_prob=0.15):
        """
        创建 masked 输入

        参数:
            item_seq: (batch, seq_len) 原始序列
            mask_prob: mask 概率

        返回:
            masked_seq: 被掩盖的序列
            mask_labels: 被 mask 的位置标签（-100 表示未被 mask）
        """
        batch_size, seq_len = item_seq.shape

        # 创建 mask（排除 padding）
        mask = (item_seq != 0).float()

        # 随机选择要 mask 的位置
        prob_matrix = torch.rand(batch_size, seq_len, device=item_seq.device)
        prob_matrix = prob_matrix * mask  # 排除 padding

        # 选择 mask 位置
        masked_indices = prob_matrix < mask_prob

        # 创建标签（只有被 mask 的位置有标签）
        mask_labels = item_seq.clone()
        mask_labels[~masked_indices] = -100  # CrossEntropy 忽略 -100

        # 创建 masked 序列
        masked_seq = item_seq.clone()
        masked_seq[masked_indices] = self.mask_token

        return masked_seq, mask_labels

    def forward(self, item_seq, attention_mask=None):
        """
        前向传播

        参数:
            item_seq: (batch, seq_len) 物品序列
            attention_mask: (batch, seq_len) 注意力掩码

        返回:
            logits: (batch, seq_len, n_items) 每个位置的物品预测
        """
        # 嵌入
        x = self.item_embedding(item_seq)  # (batch, seq_len, d_model)

        # 位置编码
        x = self.positional_encoding(x)

        # Transformer 编码器
        for block in self.transformer_blocks:
            x = block(x, attention_mask)

        # 输出层
        x = self.layer_norm(x)
        logits = self.output_layer(x)  # (batch, seq_len, n_items+1)

        return logits

    def compute_loss(self, logits, labels):
        """
        计算损失（只计算被 mask 的位置）

        参数:
            logits: (batch, seq_len, n_items+1)
            labels: (batch, seq_len) -100 表示忽略

        返回:
            loss: 标量
        """
        # 展平
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        # CrossEntropy（自动忽略 -100）
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

        return loss

    def predict(self, item_seq, top_k=10):
        """
        预测下一个物品

        参数:
            item_seq: (batch, seq_len) 或 (seq_len,) 历史序列
            top_k: 返回数量

        返回:
            top_items: (batch, top_k) 或 (top_k,) 推荐物品
            top_scores: (batch, top_k) 或 (top_k,) 分数
        """
        squeeze_output = False
        if item_seq.dim() == 1:
            item_seq = item_seq.unsqueeze(0)
            squeeze_output = True

        with torch.no_grad():
            # 在序列末尾添加 mask token
            batch_size, seq_len = item_seq.shape

            if seq_len >= self.max_seq_len:
                # 截断
                item_seq = item_seq[:, -self.max_seq_len + 1:]

            # 添加 mask token
            mask_tokens = torch.full(
                (batch_size, 1), self.mask_token,
                dtype=torch.long, device=item_seq.device
            )
            input_seq = torch.cat([item_seq, mask_tokens], dim=1)

            # 前向传播
            logits = self.forward(input_seq)  # (batch, seq_len+1, n_items+1)

            # 取最后一个位置的预测
            last_logits = logits[:, -1, :self.n_items]  # 排除 mask token

            # Top-K
            top_scores, top_items = torch.topk(last_logits, top_k, dim=-1)

            if squeeze_output:
                return top_items.squeeze(0), top_scores.squeeze(0)

        return top_items, top_scores


class BERT4RecTrainer:
    """
    BERT4Rec 训练器
    """

    def __init__(self, model, learning_rate=1e-4, weight_decay=0.01,
                 warmup_steps=100, total_steps=10000):
        """
        参数:
            model: BERT4Rec 模型
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            total_steps: 总训练步数
        """
        self.model = model
        self.learning_rate = learning_rate

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = self._create_scheduler(
            self.optimizer, warmup_steps, total_steps
        )

    def _create_scheduler(self, optimizer, warmup_steps, total_steps):
        """创建学习率调度器"""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_step(self, item_seq):
        """
        训练一步

        参数:
            item_seq: (batch, seq_len) 物品序列

        返回:
            loss: 损失值
        """
        self.model.train()

        # 创建 masked 输入
        masked_seq, mask_labels = self.model.create_masked_input(
            item_seq, self.model.mask_prob
        )

        # 创建注意力掩码
        attention_mask = (masked_seq != 0).float()

        # 前向传播
        logits = self.model(masked_seq, attention_mask)

        # 计算损失
        loss = self.model.compute_loss(logits, mask_labels)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()


# 使用示例
if __name__ == "__main__":
    # 配置
    config = {
        'n_items': 1000,
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'max_seq_len': 100,
        'dropout': 0.2,
        'mask_prob': 0.15,
        'learning_rate': 1e-4,
        'batch_size': 64,
        'epochs': 10
    }

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERT4Rec(
        n_items=config['n_items'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        mask_prob=config['mask_prob']
    ).to(device)

    # 模拟数据
    batch_size = 32
    seq_len = 20
    item_seq = torch.randint(1, config['n_items'], (batch_size, seq_len)).to(device)

    # 测试前向传播
    logits = model(item_seq)
    print(f"输入形状: {item_seq.shape}")
    print(f"输出形状: {logits.shape}")

    # 测试预测
    test_seq = torch.randint(1, config['n_items'], (1, 10)).to(device)
    top_items, top_scores = model.predict(test_seq, top_k=5)
    print(f"Top-5 推荐: {top_items}")
    print(f"分数: {top_scores}")
```

### 3.2 数据处理

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np


class BERT4RecDataset(Dataset):
    """
    BERT4Rec 数据集
    """

    def __init__(self, sequences, n_items, max_seq_len=100):
        """
        参数:
            sequences: list of item sequences
            n_items: 物品数量
            max_seq_len: 最大序列长度
        """
        self.sequences = sequences
        self.n_items = n_items
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # 截断
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]

        # 填充
        seq_len = len(seq)
        padded_seq = [0] * (self.max_seq_len - seq_len) + seq

        return {
            'item_seq': torch.LongTensor(padded_seq),
            'seq_len': seq_len
        }


def train_bert4rec():
    """完整训练示例"""
    # 配置
    config = {
        'n_items': 1000,
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'max_seq_len': 100,
        'dropout': 0.2,
        'mask_prob': 0.15,
        'learning_rate': 1e-4,
        'batch_size': 64,
        'epochs': 10
    }

    # 生成模拟数据
    n_sequences = 10000
    sequences = []
    for _ in range(n_sequences):
        seq_len = np.random.randint(5, 50)
        seq = np.random.randint(1, config['n_items'], seq_len).tolist()
        sequences.append(seq)

    # 创建数据集
    dataset = BERT4RecDataset(sequences, config['n_items'], config['max_seq_len'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERT4Rec(
        n_items=config['n_items'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        mask_prob=config['mask_prob']
    ).to(device)

    # 训练器
    total_steps = len(dataloader) * config['epochs']
    trainer = BERT4RecTrainer(
        model,
        learning_rate=config['learning_rate'],
        total_steps=total_steps
    )

    # 训练
    for epoch in range(config['epochs']):
        total_loss = 0
        n_batches = 0

        for batch in dataloader:
            item_seq = batch['item_seq'].to(device)
            loss = trainer.train_step(item_seq)

            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    model = train_bert4rec()
    print("BERT4Rec 训练完成！")
```

## 4. BERT4Rec vs SASRec

### 4.1 架构对比

```python
# SASRec: 因果注意力
# 位置 t 只能看到位置 1 到 t

# BERT4Rec: 双向注意力
# 位置 t 可以看到所有位置（训练时）
# 推理时使用 mask token 预测

class SASRec(nn.Module):
    """
    SASRec 简化版（用于对比）
    """
    def __init__(self, n_items, d_model=64, n_heads=2, n_layers=2):
        super().__init__()

        self.item_embedding = nn.Embedding(n_items, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.output_layer = nn.Linear(d_model, n_items)

    def forward(self, item_seq):
        # 嵌入
        x = self.item_embedding(item_seq)
        x = self.positional_encoding(x)

        # 因果掩码（上三角为 True）
        seq_len = item_seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_seq.device) * float('-inf'),
            diagonal=1
        )

        # Transformer（带因果掩码）
        x = self.transformer(x, mask=causal_mask)

        # 取最后一个位置
        last_hidden = x[:, -1, :]

        # 输出
        logits = self.output_layer(last_hidden)

        return logits
```

### 4.2 效果对比

| 数据集 | SASRec | BERT4Rec | 提升 |
|--------|--------|----------|------|
| Beauty | 0.065 | 0.073 | +12.3% |
| Steam | 0.056 | 0.062 | +10.7% |
| ML-1M | 0.058 | 0.067 | +15.5% |

## 5. 优缺点分析

### 5.1 优点

1. **双向上下文**：能同时利用历史和未来信息（训练时）
2. **并行计算**：Transformer 可并行化
3. **长程依赖**：比 RNN 更好地捕获长距离依赖
4. **预训练友好**：可以预训练后微调

### 5.2 缺点

1. **训练推理不一致**：训练用双向，推理用单向
2. **计算开销大**：比 SASRec 计算量更大
3. **短序列效果有限**：短序列时双向优势不明显

## 6. 调参建议

### 6.1 模型参数

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| d_model | 64-256 | 嵌入维度 |
| n_heads | 2-8 | 注意力头数 |
| n_layers | 2-4 | 层数不宜过多 |
| mask_prob | 0.1-0.3 | mask 比例 |
| dropout | 0.1-0.3 | 防止过拟合 |

### 6.2 训练参数

| 参数 | 推荐值 |
|------|--------|
| learning_rate | 1e-4 |
| weight_decay | 0.01 |
| warmup_steps | 总步数的 10% |
| batch_size | 64-256 |

## 7. 学习总结

### 7.1 核心要点

1. **双向注意力是关键**：训练时利用双向上下文
2. **Cloze 任务**：随机 mask 预测，类似 BERT
3. **预训练思想**：可以借鉴 NLP 的预训练方法

### 7.2 应用建议

- **序列较长**：BERT4Rec 优势明显
- **序列较短**：考虑 SASRec
- **需要实时性**：SASRec 更快

## 8. 练习题

1. 实现 BERT4Rec 的增量更新策略。

2. 比较不同 mask 比例对效果的影响。

3. 设计一个结合内容特征的 BERT4Rec 变体。
