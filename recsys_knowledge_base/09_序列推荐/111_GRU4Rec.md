# GRU4Rec 学习文档

## 1. 算法基础认知

### 1.1 什么是 GRU4Rec？

GRU4Rec 是第一个将 RNN（循环神经网络）应用于会话推荐的模型，由 Balázs Hidasi 等人在 2015 年的 ICLR 上发表。它使用 GRU（Gated Recurrent Unit）来建模用户在会话中的行为序列。

### 1.2 核心思想

- **会话建模**：将用户一次连续的交互作为一个会话（session）
- **序列依赖**：使用 GRU 捕获物品之间的顺序关系
- **实时推荐**：根据当前会话中的行为实时更新推荐

### 1.3 应用场景

- **电商网站**：用户一次浏览 session 中的商品推荐
- **视频平台**：基于当前观看 session 推荐下一个视频
- **音乐平台**：基于当前收听 session 推荐下一首歌
- **新闻应用**：基于当前阅读 session 推荐下一篇新闻

## 2. GRU 原理回顾

### 2.1 GRU 结构

GRU 是 LSTM 的简化版本，只有两个门：

$$
\begin{aligned}
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) & \text{(更新门)} \\
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) & \text{(重置门)} \\
\tilde{h}_t &= \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) & \text{(候选隐状态)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t & \text{(隐状态)}
\end{aligned}
$$

### 2.2 GRU 在推荐中的作用

- **更新门**：控制保留多少历史信息
- **重置门**：控制忽略多少历史信息
- **隐状态**：编码用户当前兴趣状态

## 3. GRU4Rec 模型架构

### 3.1 整体架构

```
输入序列: [item_1, item_2, ..., item_t]
           ↓       ↓            ↓
        Embed   Embed        Embed
           ↓       ↓            ↓
         GRU_1   GRU_2  ...  GRU_t
           ↓       ↓            ↓
         h_1     h_2    ...   h_t
                              ↓
                          输出层
                              ↓
                         预测下一个物品
```

### 3.2 损失函数

GRU4Rec 支持多种损失函数：

1. **TOP1 损失**（论文提出）
$$L_{TOP1} = \frac{1}{N_s} \sum_{j=1}^{N_s} \sigma(r_{j} - r_i) + \sigma(r_j^2)$$

2. **BPR 损失**
$$L_{BPR} = -\frac{1}{N_s} \sum_{j=1}^{N_s} \ln(\sigma(r_i - r_j))$$

3. **Cross-Entropy 损失**
$$L_{CE} = -\log(\text{softmax}(r_i))$$

## 4. PyTorch 实现

### 4.1 基础实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRU4Rec(nn.Module):
    """
    GRU4Rec: Session-based Recommendations with RNNs

    论文: Session-based Recommendations with Recurrent Neural Networks (ICLR 2016)
    """

    def __init__(self, n_items, embed_dim=128, hidden_dim=128,
                 n_layers=1, dropout=0.2, loss_type='bpr'):
        """
        参数:
            n_items: 物品数量
            embed_dim: 嵌入维度
            hidden_dim: GRU 隐藏层维度
            n_layers: GRU 层数
            dropout: Dropout 比例
            loss_type: 损失函数类型 ('bpr', 'top1', 'ce')
        """
        super().__init__()

        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.loss_type = loss_type

        # 物品嵌入层
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # GRU 层
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, n_items)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, item_seq, seq_lengths=None):
        """
        前向传播

        参数:
            item_seq: (batch, seq_len) 物品序列
            seq_lengths: 序列实际长度（可选）

        返回:
            logits: (batch, n_items) 每个物品的分数
        """
        # 嵌入
        x = self.item_embedding(item_seq)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)

        # GRU
        if seq_lengths is not None:
            # 打包序列
            packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, hidden = self.gru(packed)
            # 解包
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, hidden = self.gru(x)  # output: (batch, seq_len, hidden_dim)

        # 使用最后一个时间步的输出
        if seq_lengths is not None:
            # 获取每个序列的最后一个有效输出
            batch_size = item_seq.size(0)
            last_output = output[torch.arange(batch_size), seq_lengths - 1]
        else:
            last_output = output[:, -1, :]  # (batch, hidden_dim)

        last_output = self.dropout(last_output)

        # 输出层
        logits = self.output_layer(last_output)  # (batch, n_items)

        return logits

    def compute_loss(self, logits, pos_items, neg_items=None):
        """
        计算损失

        参数:
            logits: (batch, n_items) 所有物品分数
            pos_items: (batch,) 正样本物品
            neg_items: (batch, n_neg) 负样本物品（可选）

        返回:
            loss: 标量损失
        """
        if self.loss_type == 'ce':
            # Cross-Entropy 损失
            loss = F.cross_entropy(logits, pos_items)

        elif self.loss_type == 'bpr':
            # BPR 损失
            pos_scores = logits[torch.arange(len(pos_items)), pos_items]

            if neg_items is None:
                # 随机采样负样本
                neg_items = torch.randint(0, self.n_items, (len(pos_items), 10), device=logits.device)

            neg_scores = logits[torch.arange(len(pos_items)).unsqueeze(1), neg_items]

            # BPR: -log(sigmoid(pos - neg))
            bpr_loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()
            loss = bpr_loss

        elif self.loss_type == 'top1':
            # TOP1 损失
            pos_scores = logits[torch.arange(len(pos_items)), pos_items]

            if neg_items is None:
                neg_items = torch.randint(0, self.n_items, (len(pos_items), 10), device=logits.device)

            neg_scores = logits[torch.arange(len(pos_items)).unsqueeze(1), neg_items]

            # TOP1: sigmoid(neg - pos) + sigmoid(neg^2)
            top1_loss = torch.sigmoid(neg_scores - pos_scores.unsqueeze(1)).mean()
            top1_loss += torch.sigmoid(neg_scores ** 2).mean()
            loss = top1_loss

        return loss

    def predict(self, item_seq, top_k=10):
        """
        预测

        参数:
            item_seq: (batch, seq_len) 或 (seq_len,) 物品序列
            top_k: 返回数量

        返回:
            top_indices: (batch, top_k) 或 (top_k,) 推荐物品索引
            top_scores: (batch, top_k) 或 (top_k,) 推荐分数
        """
        # 确保是 2D
        if item_seq.dim() == 1:
            item_seq = item_seq.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        with torch.no_grad():
            logits = self.forward(item_seq)  # (batch, n_items)

            # 排序
            scores, indices = torch.topk(logits, top_k, dim=-1)

            if squeeze_output:
                return indices.squeeze(0), scores.squeeze(0)

        return indices, scores


class GRU4RecWithEmbedding(nn.Module):
    """
    带物品嵌入共享的 GRU4Rec
    使用嵌入矩阵作为输出层权重（类似 Word2Vec）
    """

    def __init__(self, n_items, embed_dim=128, hidden_dim=128,
                 n_layers=1, dropout=0.2):
        super().__init__()

        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 物品嵌入
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # 投影层：将 hidden_dim 映射到 embed_dim
        self.projection = nn.Linear(hidden_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, item_seq):
        """
        前向传播

        参数:
            item_seq: (batch, seq_len)

        返回:
            scores: (batch, n_items) 使用嵌入矩阵计算的内积分数
        """
        # 嵌入
        x = self.item_embedding(item_seq)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)

        # GRU
        output, hidden = self.gru(x)  # output: (batch, seq_len, hidden_dim)

        # 最后一个时间步
        last_hidden = output[:, -1, :]  # (batch, hidden_dim)

        # 投影到嵌入空间
        query = self.projection(last_hidden)  # (batch, embed_dim)

        # 计算与所有物品的内积
        all_embeddings = self.item_embedding.weight  # (n_items, embed_dim)
        scores = torch.matmul(query, all_embeddings.t())  # (batch, n_items)

        return scores
```

### 4.2 训练示例

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SessionDataset(Dataset):
    """
    会话数据集
    """

    def __init__(self, sessions, n_items, max_len=50):
        """
        参数:
            sessions: list of item sequences
            n_items: 物品数量
            max_len: 最大序列长度
        """
        self.sessions = sessions
        self.n_items = n_items
        self.max_len = max_len

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]

        # 截断到最大长度
        if len(session) > self.max_len:
            session = session[-self.max_len:]

        # 输入：前 n-1 个物品
        # 目标：后 n-1 个物品（预测下一个）
        input_seq = session[:-1]
        target_seq = session[1:]

        # 填充
        seq_len = len(input_seq)
        input_padded = [0] * (self.max_len - seq_len) + input_seq
        target_padded = [0] * (self.max_len - seq_len) + target_seq

        return {
            'input_seq': torch.LongTensor(input_padded),
            'target_seq': torch.LongTensor(target_padded),
            'seq_len': seq_len
        }


def collate_fn(batch):
    """自定义批处理函数"""
    input_seqs = torch.stack([item['input_seq'] for item in batch])
    target_seqs = torch.stack([item['target_seq'] for item in batch])
    seq_lens = torch.LongTensor([item['seq_len'] for item in batch])

    return {
        'input_seq': input_seqs,
        'target_seq': target_seqs,
        'seq_len': seq_lens
    }


def train_gru4rec():
    """训练 GRU4Rec"""
    # 配置
    config = {
        'n_items': 1000,
        'embed_dim': 64,
        'hidden_dim': 128,
        'n_layers': 1,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 10,
        'max_len': 50
    }

    # 生成模拟数据
    n_sessions = 10000
    sessions = []
    for _ in range(n_sessions):
        session_len = np.random.randint(3, 20)
        session = np.random.randint(1, config['n_items'], session_len).tolist()
        sessions.append(session)

    # 创建数据集
    dataset = SessionDataset(sessions, config['n_items'], config['max_len'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRU4Rec(
        n_items=config['n_items'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        loss_type='bpr'
    ).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 训练
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in dataloader:
            input_seq = batch['input_seq'].to(device)
            target_seq = batch['target_seq'].to(device)
            seq_len = batch['seq_len']

            optimizer.zero_grad()

            # 前向传播
            logits = model(input_seq, seq_len)  # (batch, n_items)

            # 取每个序列最后一个位置的目标
            # 注意：target_seq 中的位置对应 input_seq 的下一个
            batch_indices = torch.arange(len(seq_len), device=device)
            last_pos = (seq_len - 1).to(device)
            target_items = target_seq[batch_indices, last_pos]

            # 采样负样本
            neg_items = torch.randint(
                1, config['n_items'],
                (len(seq_len), 10),
                device=device
            )

            # 计算损失
            loss = model.compute_loss(logits, target_items, neg_items)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    model = train_gru4rec()
    print("GRU4Rec 训练完成！")

    # 测试预测
    test_seq = torch.randint(1, 1000, (1, 10))
    top_indices, top_scores = model.predict(test_seq, top_k=5)
    print(f"Top-5 推荐: {top_indices}")
    print(f"分数: {top_scores}")
```

## 5. 数据处理与负采样

### 5.1 负采样策略

```python
import numpy as np
from collections import Counter


class NegativeSampler:
    """
    负采样器
    """

    def __init__(self, item_freq, n_items, sample_method='uniform'):
        """
        参数:
            item_freq: 物品频率字典
            n_items: 物品数量
            sample_method: 采样方法 ('uniform', 'pop', 'unpop')
        """
        self.n_items = n_items
        self.sample_method = sample_method

        if sample_method == 'pop':
            # 按流行度采样
            items = list(item_freq.keys())
            probs = np.array([item_freq[i] for i in items], dtype=np.float64)
            probs = probs / probs.sum()
            self.pop_items = items
            self.pop_probs = probs
        elif sample_method == 'unpop':
            # 按不流行度采样（更倾向于冷门物品）
            items = list(item_freq.keys())
            freqs = np.array([item_freq[i] for i in items], dtype=np.float64)
            probs = 1.0 / (freqs + 1)
            probs = probs / probs.sum()
            self.pop_items = items
            self.pop_probs = probs

    def sample(self, n_samples, exclude=None):
        """
        采样负样本

        参数:
            n_samples: 采样数量
            exclude: 要排除的物品集合

        返回:
            负样本列表
        """
        exclude = exclude or set()

        if self.sample_method == 'uniform':
            samples = []
            while len(samples) < n_samples:
                item = np.random.randint(0, self.n_items)
                if item not in exclude:
                    samples.append(item)
            return samples

        else:
            samples = []
            while len(samples) < n_samples:
                idx = np.random.choice(len(self.pop_items), p=self.pop_probs)
                item = self.pop_items[idx]
                if item not in exclude:
                    samples.append(item)
            return samples


class SessionDataProcessor:
    """
    会话数据处理器
    """

    def __init__(self, min_session_len=3, max_session_len=50,
                 session_gap=30 * 60):  # 30 分钟
        """
        参数:
            min_session_len: 最小会话长度
            max_session_len: 最大会话长度
            session_gap: 会话间隔（秒）
        """
        self.min_session_len = min_session_len
        self.max_session_len = max_session_len
        self.session_gap = session_gap

    def split_sessions(self, interactions):
        """
        将交互序列分割成会话

        参数:
            interactions: [(user_id, item_id, timestamp), ...]

        返回:
            sessions: [[item_id, ...], ...]
        """
        # 按时间排序
        interactions = sorted(interactions, key=lambda x: x[2])

        sessions = []
        current_session = []
        last_time = None

        for user_id, item_id, timestamp in interactions:
            if last_time is not None and (timestamp - last_time) > self.session_gap:
                # 开始新会话
                if len(current_session) >= self.min_session_len:
                    sessions.append(current_session)
                current_session = []

            current_session.append(item_id)
            last_time = timestamp

        # 最后一个会话
        if len(current_session) >= self.min_session_len:
            sessions.append(current_session)

        return sessions

    def build_item_vocab(self, sessions):
        """构建物品词表"""
        all_items = set()
        for session in sessions:
            all_items.update(session)

        item_to_idx = {item: idx + 1 for idx, item in enumerate(all_items)}
        item_to_idx[0] = 0  # padding
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}

        return item_to_idx, idx_to_item

    def encode_sessions(self, sessions, item_to_idx):
        """编码会话"""
        encoded = []
        for session in sessions:
            encoded.append([item_to_idx[item] for item in session])
        return encoded
```

## 6. GRU4Rec 改进版本

### 6.1 GRU4Rec+ (2016)

```python
class GRU4RecPlus(nn.Module):
    """
    GRU4Rec+ 改进版本

    改进点:
    1. 使用 embedding 共享
    2. 添加偏置项
    3. 使用 mini-batch 训练优化
    """

    def __init__(self, n_items, embed_dim=128, hidden_dim=128,
                 n_layers=1, dropout=0.2):
        super().__init__()

        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 嵌入层
        self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)

        # GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # 输出层（使用嵌入矩阵）
        self.W = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(n_items))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, item_seq, seq_lengths=None):
        """前向传播"""
        # 嵌入
        x = self.item_embedding(item_seq)
        x = self.dropout(x)

        # GRU
        output, hidden = self.gru(x)

        # 取最后一个有效输出
        if seq_lengths is not None:
            batch_size = item_seq.size(0)
            last_output = output[torch.arange(batch_size), seq_lengths - 1]
        else:
            last_output = output[:, -1, :]

        # Layer Norm
        last_output = self.layer_norm(last_output)

        # 投影
        query = self.W(last_output)  # (batch, embed_dim)

        # 与所有物品嵌入计算内积
        all_embeddings = self.item_embedding.weight  # (n_items, embed_dim)
        scores = torch.matmul(query, all_embeddings.t()) + self.b  # (batch, n_items)

        return scores


class GRU4RecWithAttention(nn.Module):
    """
    带注意力机制的 GRU4Rec
    """

    def __init__(self, n_items, embed_dim=128, hidden_dim=128,
                 n_layers=1, dropout=0.2, n_heads=4):
        super().__init__()

        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 嵌入层
        self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)

        # GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, n_items)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, item_seq, attention_mask=None):
        """前向传播"""
        # 嵌入
        x = self.item_embedding(item_seq)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)

        # GRU
        gru_output, _ = self.gru(x)  # (batch, seq_len, hidden_dim)

        # 自注意力
        attn_output, _ = self.attention(
            gru_output, gru_output, gru_output,
            key_padding_mask=attention_mask
        )

        # 残差连接 + LayerNorm
        output = self.layer_norm(gru_output + attn_output)

        # 取最后一个位置
        last_output = output[:, -1, :]

        # 输出
        logits = self.output_layer(last_output)

        return logits
```

## 7. 评估与对比

### 7.1 与其他方法对比

| 方法 | 类型 | 优点 | 缺点 |
|------|------|------|------|
| Item-KNN | 基于相似度 | 简单、可解释 | 忽略序列 |
| Markov Chain | 概率模型 | 建模转移 | 只考虑短程 |
| GRU4Rec | RNN | 长程依赖、实时 | 并行化困难 |
| SASRec | Transformer | 并行化、长程 | 计算量大 |

### 7.2 适用场景

**适合 GRU4Rec：**
- 会话较短（< 50 个物品）
- 需要实时更新
- 对延迟敏感

**考虑替代方案：**
- 序列很长（考虑 Transformer）
- 需要全局信息（考虑图模型）
- 冷启动严重（考虑内容特征）

## 8. 学习总结

### 8.1 核心要点

1. **RNN 天然适合序列**：隐状态传递建模时序依赖
2. **GRU 比 LSTM 简洁**：两个门控，参数更少
3. **负采样很关键**：影响训练效率和效果

### 8.2 关键改进点

- 使用 **embedding 共享** 减少参数
- 使用 **batch 训练** 提高效率
- 使用 **TOP1/BPR 损失** 优化排序

## 9. 练习题

1. 实现 GRU4Rec 的 mini-batch 负采样策略。

2. 比较不同损失函数（BPR vs TOP1 vs CE）的效果。

3. 在真实数据集（如 MovieLens）上训练并评估 GRU4Rec。
