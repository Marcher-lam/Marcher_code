# SASRec（自注意力序列推荐） 学习文档

## 1. 算法基础认知

### 1.1 什么是 SASRec？

SASRec（Self-Attentive Sequential Recommendation）是 2018 年提出的基于自注意力机制的序列推荐模型，将 Transformer 的自注意力机制应用到推荐系统。

### 1.2 序列推荐问题

**问题定义：**
- 输入：用户的历史行为序列 $(s_1, s_2, ..., s_n)$
- 输出：预测用户下一个可能交互的物品

**与 CTR 预估的区别：**
- CTR 预估：给定用户-物品对，预测点击概率
- 序列推荐：基于序列预测下一个物品

### 1.3 为什么用自注意力？

**传统方法的问题：**

| 方法 | 问题 |
|------|------|
| MC（马尔可夫链） | 只考虑最后一个物品 |
| FPMC | 只考虑一阶转移 |
| GRU4Rec | 序列建模，但难以捕捉长距离依赖 |

**自注意力的优势：**
1. 并行计算效率高
2. 能捕捉长距离依赖
3. 自适应关注相关历史

## 2. 模型架构

### 2.1 整体架构

```
输入序列: [item_1, item_2, ..., item_n]
         ↓
Embedding 层 (Item Embedding + Position Embedding)
         ↓
┌────────────────────────────────────────┐
│         Self-Attention Block (×L)      │
│  ┌────────────────────────────────┐    │
│  │     Multi-Head Self-Attention  │    │
│  └────────────────────────────────┘    │
│              ↓                          │
│  ┌────────────────────────────────┐    │
│  │     Feed-Forward Network       │    │
│  └────────────────────────────────┘    │
└────────────────────────────────────────┘
         ↓
预测层 (预测下一个物品)
```

### 2.2 Embedding 层

$$E = I + P$$

其中：
- $I \in \mathbb{R}^{n \times d}$：物品 Embedding
- $P \in \mathbb{R}^{n \times d}$：位置 Embedding
- $E$：最终输入 Embedding

### 2.3 自注意力机制

**Scaled Dot-Product Attention：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head Attention：**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 2.4 因果掩码

SASRec 使用因果掩码，确保位置 $t$ 只能看到位置 $1$ 到 $t-1$ 的信息：

```
掩码矩阵（1 表示可见，0 表示掩码）：
位置  1  2  3  4
1  [  1, 0, 0, 0 ]
2  [  1, 1, 0, 0 ]
3  [  1, 1, 1, 0 ]
4  [  1, 1, 1, 1 ]
```

## 3. 代码实现

### 3.1 完整 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SASRec(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation
    """

    def __init__(self, num_items, embed_dim=64, max_seq_len=50,
                 num_heads=2, num_blocks=2, dropout=0.2):
        """
        参数:
            num_items: 物品数量
            embed_dim: Embedding 维度
            max_seq_len: 最大序列长度
            num_heads: 注意力头数
            num_blocks: Transformer 块数
            dropout: Dropout 比例
        """
        super().__init__()

        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # ========== Embedding 层 ==========
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim)  # +1 for padding
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # ========== Transformer 块 ==========
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        # ========== 预测层 ==========
        self.output_layer = nn.Linear(embed_dim, num_items + 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, item_seq, mask=None):
        """
        前向传播

        参数:
            item_seq: (batch, seq_len) 物品序列
            mask: (batch, seq_len) 有效位置 mask

        返回:
            logits: (batch, seq_len, num_items) 预测 logits
        """
        batch_size, seq_len = item_seq.shape

        # 物品 Embedding
        item_emb = self.item_embedding(item_seq)  # (batch, seq_len, embed_dim)

        # 位置 Embedding
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, embed_dim)

        # 组合
        x = item_emb + pos_emb
        x = self.dropout(x)

        # 因果掩码
        causal_mask = self._get_causal_mask(seq_len, item_seq.device)

        # Transformer 块
        for block in self.transformer_blocks:
            x = block(x, causal_mask)

        # 预测
        logits = self.output_layer(x)  # (batch, seq_len, num_items)

        return logits

    def _get_causal_mask(self, seq_len, device):
        """生成因果掩码"""
        # 下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0)  # (1, seq_len, seq_len)
        return mask

    def predict_next(self, item_seq, mask=None):
        """
        预测下一个物品

        返回最后一个位置的预测
        """
        logits = self.forward(item_seq, mask)  # (batch, seq_len, num_items)

        # 取最后一个位置
        last_logits = logits[:, -1, :]  # (batch, num_items)

        return last_logits

    def get_item_embedding(self):
        """获取物品 Embedding"""
        return self.item_embedding.weight[1:]  # 排除 padding


class TransformerBlock(nn.Module):
    """Transformer 块"""

    def __init__(self, embed_dim, num_heads, dropout=0.2, ff_dim=None):
        super().__init__()

        if ff_dim is None:
            ff_dim = embed_dim * 4

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # Layer Norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        参数:
            x: (batch, seq_len, embed_dim)
            attn_mask: (1, seq_len, seq_len) 注意力掩码
        """
        # 自注意力
        # 注意：nn.MultiheadAttention 的 mask 是加到注意力分数上的
        # 0 表示不掩码，负无穷表示掩码
        if attn_mask is not None:
            # 转换为加法掩码
            attn_mask = (1 - attn_mask) * (-1e9)

        residual = x
        x = self.norm1(x)

        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout1(attn_output)

        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout2(x)

        return x


class SASRecWithSampling(nn.Module):
    """
    带负采样的 SASRec（用于训练）
    """

    def __init__(self, num_items, embed_dim=64, max_seq_len=50,
                 num_heads=2, num_blocks=2, dropout=0.2):
        super().__init__()

        self.sasrec = SASRec(
            num_items, embed_dim, max_seq_len,
            num_heads, num_blocks, dropout
        )

    def forward(self, item_seq, pos_items, neg_items, mask=None):
        """
        训练前向传播

        参数:
            item_seq: (batch, seq_len) 输入序列
            pos_items: (batch, seq_len) 正样本（下一个物品）
            neg_items: (batch, seq_len, num_neg) 负样本
            mask: (batch, seq_len) 有效位置 mask

        返回:
            loss: 损失值
        """
        # 获取序列表示
        seq_output = self._get_sequence_output(item_seq, mask)
        # (batch, seq_len, embed_dim)

        # 正样本 Embedding
        pos_emb = self.sasrec.item_embedding(pos_items)  # (batch, seq_len, embed_dim)

        # 负样本 Embedding
        batch_size, seq_len, num_neg = neg_items.shape
        neg_emb = self.sasrec.item_embedding(neg_items.view(-1))
        neg_emb = neg_emb.view(batch_size, seq_len, num_neg, -1)

        # 计算分数
        pos_scores = torch.sum(seq_output * pos_emb, dim=-1)  # (batch, seq_len)

        neg_scores = torch.einsum('bsf,bsnf->bsn', seq_output, neg_emb)
        # (batch, seq_len, num_neg)

        # BPR 损失
        pos_scores = pos_scores.unsqueeze(-1)  # (batch, seq_len, 1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        # 应用 mask
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    def _get_sequence_output(self, item_seq, mask=None):
        """获取序列输出"""
        batch_size, seq_len = item_seq.shape

        item_emb = self.sasrec.item_embedding(item_seq)

        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        pos_emb = self.sasrec.position_embedding(positions)

        x = item_emb + pos_emb
        x = self.sasrec.dropout(x)

        causal_mask = self.sasrec._get_causal_mask(seq_len, item_seq.device)

        for block in self.sasrec.transformer_blocks:
            x = block(x, causal_mask)

        return x
```

### 3.2 训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SequenceDataset(Dataset):
    """序列推荐数据集"""

    def __init__(self, sequences, max_seq_len=50, num_neg=4, num_items=10000):
        """
        参数:
            sequences: list of list, 用户行为序列
            max_seq_len: 最大序列长度
            num_neg: 负样本数量
            num_items: 物品总数
        """
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.num_neg = num_neg
        self.num_items = num_items

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # 构建训练样本
        # 输入: [item_1, ..., item_{t-1}]
        # 目标: item_t

        samples = []

        for t in range(1, len(seq)):
            # 输入序列
            input_seq = seq[:t]
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]

            # 填充
            padding_len = self.max_seq_len - len(input_seq)
            input_seq = [0] * padding_len + input_seq

            # 正样本
            pos_item = seq[t]

            # 负样本
            neg_items = []
            while len(neg_items) < self.num_neg:
                neg = np.random.randint(1, self.num_items)
                if neg != pos_item and neg not in seq:
                    neg_items.append(neg)

            # 创建 mask
            mask = [0] * padding_len + [1] * (self.max_seq_len - padding_len)

            samples.append({
                'input_seq': torch.LongTensor(input_seq),
                'pos_item': torch.LongTensor([pos_item]),
                'neg_items': torch.LongTensor(neg_items),
                'mask': torch.FloatTensor(mask)
            })

        # 返回最后一个样本（简化）
        return samples[-1] if samples else {
            'input_seq': torch.LongTensor([0] * self.max_seq_len),
            'pos_item': torch.LongTensor([1]),
            'neg_items': torch.LongTensor([2, 3, 4, 5]),
            'mask': torch.FloatTensor([0] * (self.max_seq_len - 1) + [1])
        }


def train_sasrec():
    """训练 SASRec"""
    config = {
        'num_items': 10000,
        'embed_dim': 64,
        'max_seq_len': 50,
        'num_heads': 2,
        'num_blocks': 2,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 0.001,
        'epochs': 10,
        'num_neg': 4
    }

    # 创建模型
    model = SASRecWithSampling(
        num_items=config['num_items'],
        embed_dim=config['embed_dim'],
        max_seq_len=config['max_seq_len'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout=config['dropout']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 模拟数据
    sequences = [
        [np.random.randint(1, config['num_items']) for _ in range(np.random.randint(5, 20))]
        for _ in range(10000)
    ]

    dataset = SequenceDataset(
        sequences,
        max_seq_len=config['max_seq_len'],
        num_neg=config['num_neg'],
        num_items=config['num_items']
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # 训练
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_seq = batch['input_seq'].to(device)
            pos_item = batch['pos_item'].squeeze(-1).to(device)
            neg_items = batch['neg_items'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            loss = model(input_seq, pos_item, neg_items, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    model = train_sasrec()
    print("SASRec 训练完成！")
```

## 4. 推理与评估

### 4.1 推理

```python
def recommend(model, user_history, top_k=10):
    """
    为用户推荐

    参数:
        model: 训练好的 SASRec 模型
        user_history: list, 用户历史物品
        top_k: 推荐数量
    """
    model.eval()

    with torch.no_grad():
        # 准备输入
        seq = user_history[-model.max_seq_len:]
        padding_len = model.max_seq_len - len(seq)
        seq = [0] * padding_len + seq

        item_seq = torch.LongTensor([seq]).to(next(model.parameters()).device)

        # 预测
        logits = model.predict_next(item_seq)  # (1, num_items)

        # 排除已交互物品
        for item in user_history:
            logits[0, item] = -float('inf')

        # 获取 top-k
        top_items = torch.argsort(logits, descending=True)[:,:top_k]

    return top_items[0].cpu().numpy()
```

### 4.2 评估指标

```python
def evaluate_sequence_model(model, test_data, k_list=[1, 5, 10, 20]):
    """
    评估序列推荐模型

    参数:
        model: 模型
        test_data: [(user_history, next_item), ...]
        k_list: 评估的 K 值列表
    """
    model.eval()
    metrics = {f'Hit@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})

    with torch.no_grad():
        for user_history, next_item in test_data:
            # 获取推荐
            seq = user_history[-model.max_seq_len:]
            padding_len = model.max_seq_len - len(seq)
            seq = [0] * padding_len + seq

            item_seq = torch.LongTensor([seq]).to(next(model.parameters()).device)
            logits = model.predict_next(item_seq)

            # 排除历史物品
            for item in user_history:
                logits[0, item] = -float('inf')

            # 排序
            sorted_items = torch.argsort(logits, descending=True)[0].cpu().numpy()

            for k in k_list:
                top_k = sorted_items[:k]

                # Hit@K
                if next_item in top_k:
                    metrics[f'Hit@{k}'].append(1)
                    # NDCG@K
                    rank = np.where(top_k == next_item)[0][0]
                    ndcg = 1 / np.log2(rank + 2)
                    metrics[f'NDCG@{k}'].append(ndcg)
                else:
                    metrics[f'Hit@{k}'].append(0)
                    metrics[f'NDCG@{k}'].append(0)

    # 平均
    return {k: np.mean(v) for k, v in metrics.items()}
```

## 5. 与其他模型对比

| 模型 | 序列建模 | 注意力 | 并行性 | 长序列 |
|------|----------|--------|--------|--------|
| GRU4Rec | GRU | 无 | 低 | 中等 |
| SASRec | Transformer | 自注意力 | 高 | 好 |
| BERT4Rec | Transformer | 双向注意力 | 高 | 好 |

## 6. 学习总结

### 6.1 核心要点

1. **自注意力**：捕捉序列中任意位置的关系
2. **因果掩码**：保证只看历史，不看未来
3. **位置编码**：保留序列顺序信息
4. **并行计算**：比 RNN 更高效

### 6.2 适用场景

- 序列推荐
- 会话推荐
- 用户行为建模

## 7. 练习题

1. 实现 SASRec 的多头注意力机制。

2. 比较不同 num_blocks 对模型效果的影响。

3. 实现 BERT4Rec（双向自注意力）并与 SASRec 对比。
