# SASRec 详解 学习文档

## 1. SASRec 概述

### 1.1 什么是 SASRec？

```
SASRec (Self-Attentive Sequential Recommendation)

- 2018年 ICLR 提出
- 首次将 Transformer 引入序列推荐
- 使用自注意力机制建模用户行为序列

核心思想:
- 用户历史行为序列 → Transformer 编码 → 预测下一个物品
- 自注意力捕捉序列中的长距离依赖
- 位置编码保留序列顺序信息
```

### 1.2 与其他序列模型对比

```
模型           架构            优点                缺点
───────────────────────────────────────────────────────
GRU4Rec       RNN            捕捉序列依赖        长序列梯度消失
Caser         CNN            并行计算            难捕捉长距离依赖
SASRec        Transformer    长距离依赖、并行    计算复杂度 O(n²)
BERT4Rec      BERT           双向建模            训练成本高
```

## 2. 模型架构

### 2.1 整体架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class SASRec(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation

    架构:
    1. Item Embedding Layer
    2. Position Embedding Layer
    3. Multi-Head Self-Attention (× N layers)
    4. Feed-Forward Network (× N layers)
    5. Prediction Layer
    """

    def __init__(self,
                 n_items: int,
                 embed_dim: int = 64,
                 n_heads: int = 2,
                 n_layers: int = 2,
                 max_seq_len: int = 50,
                 dropout: float = 0.2,
                 item_padding_idx: int = 0):
        super().__init__()

        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 1. Item Embedding
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items + 1,  # +1 for padding
            embedding_dim=embed_dim,
            padding_idx=item_padding_idx
        )

        # 2. Position Embedding
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embed_dim
        )

        # 3. Self-Attention Blocks
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 4. Feed-Forward Networks
        self.ffn_layers = nn.ModuleList([
            FeedForwardNetwork(embed_dim, dropout)
            for _ in range(n_layers)
        ])

        # 5. Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(n_layers * 2)
        ])

        self.dropout = nn.Dropout(dropout)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, item_seq: torch.Tensor,
                seq_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        前向传播

        参数:
            item_seq: (batch, seq_len) 物品序列
            seq_mask: (batch, seq_len) 序列mask (1 for valid, 0 for padding)

        返回:
            seq_output: (batch, seq_len, embed_dim) 序列表示
        """
        batch_size, seq_len = item_seq.shape

        # 1. Item Embedding
        x = self.item_embedding(item_seq)  # (batch, seq_len, embed_dim)

        # 2. Position Embedding
        positions = torch.arange(seq_len, device=item_seq.device)
        x = x + self.position_embedding(positions)

        # 3. Dropout
        x = self.dropout(x)

        # 4. Self-Attention Layers
        # 创建因果mask (下三角矩阵)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_seq.device),
            diagonal=1
        ).bool()

        # 合并padding mask
        if seq_mask is not None:
            # (batch, 1, seq_len) & (1, seq_len, seq_len)
            padding_mask = (seq_mask == 0).unsqueeze(1)
            attention_mask = causal_mask | padding_mask
        else:
            attention_mask = causal_mask

        for i in range(len(self.attention_layers)):
            # Self-Attention
            residual = x
            x = self.layer_norms[i * 2](x)
            x = self.attention_layers[i](
                x, x, x,
                attn_mask=attention_mask
            )
            x = self.dropout(x)
            x = residual + x

            # Feed-Forward
            residual = x
            x = self.layer_norms[i * 2 + 1](x)
            x = self.ffn_layers[i](x)
            x = self.dropout(x)
            x = residual + x

        # 最后的 LayerNorm
        x = self.layer_norms[-1](x)

        return x

    def predict(self, seq_output: torch.Tensor,
                target_items: torch.Tensor = None
                ) -> torch.Tensor:
        """
        预测

        返回:
            如果 target_items 给定: (batch,) 每个样本的得分
            否则: (batch, n_items) 所有物品的得分
        """
        # 取最后一个位置的输出
        last_output = seq_output[:, -1, :]  # (batch, embed_dim)

        if target_items is not None:
            # 计算目标物品得分
            target_embed = self.item_embedding(target_items)  # (batch, embed_dim)
            scores = (last_output * target_embed).sum(dim=-1)  # (batch,)
            return scores
        else:
            # 计算所有物品得分
            all_item_embed = self.item_embedding.weight  # (n_items, embed_dim)
            scores = torch.matmul(last_output, all_item_embed.T)  # (batch, n_items)
            return scores
```

### 2.2 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """
    多头自注意力
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Q, K, V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        前向传播

        参数:
            query, key, value: (batch, seq_len, embed_dim)
            attn_mask: (seq_len, seq_len) or (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape

        # 线性投影
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 分割多头
        # (batch, seq_len, embed_dim) -> (batch, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        # (batch, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用mask
        if attn_mask is not None:
            # attn_mask: (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        # (batch, n_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        # (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        # 输出投影
        output = self.out_proj(attn_output)

        return output
```

### 2.3 Feed-Forward Network

```python
class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, hidden_ratio: int = 4):
        super().__init__()

        hidden_dim = embed_dim * hidden_ratio

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),  # 或 nn.ReLU()
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
```

## 3. 训练

### 3.1 数据处理

```python
class SASRecDataset:
    """
    SASRec 数据集
    """

    def __init__(self,
                 interactions: List[Tuple[int, List[int]]],
                 max_seq_len: int = 50,
                 n_items: int = None):
        """
        参数:
            interactions: [(user_id, item_sequence), ...]
            max_seq_len: 最大序列长度
            n_items: 物品总数
        """
        self.interactions = interactions
        self.max_seq_len = max_seq_len
        self.n_items = n_items

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id, item_seq = self.interactions[idx]

        # 截断或填充
        if len(item_seq) > self.max_seq_len:
            item_seq = item_seq[-self.max_seq_len:]

        # 输入序列 (去掉最后一个)
        input_seq = item_seq[:-1]
        # 目标 (最后一个)
        target = item_seq[-1]

        # 填充
        seq_len = len(input_seq)
        padding_len = self.max_seq_len - seq_len

        if padding_len > 0:
            input_seq = [0] * padding_len + input_seq

        # 创建mask
        mask = [0] * padding_len + [1] * seq_len

        return {
            'user_id': user_id,
            'item_seq': torch.tensor(input_seq, dtype=torch.long),
            'seq_mask': torch.tensor(mask, dtype=torch.float),
            'target': torch.tensor(target, dtype=torch.long),
            'seq_len': seq_len
        }


def collate_fn(batch):
    """批处理函数"""
    return {
        'user_id': torch.tensor([item['user_id'] for item in batch]),
        'item_seq': torch.stack([item['item_seq'] for item in batch]),
        'seq_mask': torch.stack([item['seq_mask'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
        'seq_len': torch.tensor([item['seq_len'] for item in batch])
    }
```

### 3.2 训练器

```python
class SASRecTrainer:
    """
    SASRec 训练器
    """

    def __init__(self,
                 model: SASRec,
                 n_items: int,
                 lr: float = 0.001,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 100,
                 max_steps: int = 10000):
        self.model = model
        self.n_items = n_items

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器 (带预热)
        self.scheduler = WarmupLinearScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps
        )

        # 负采样
        self.neg_sample_size = 100

    def train_epoch(self, dataloader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # 移动到设备
            item_seq = batch['item_seq']
            seq_mask = batch['seq_mask']
            target = batch['target']

            # 前向传播
            seq_output = self.model(item_seq, seq_mask)

            # 负采样
            neg_items = self._sample_negatives(target)

            # 正样本得分
            pos_scores = self.model.predict(seq_output, target)

            # 负样本得分
            batch_size = target.shape[0]
            seq_output_expand = seq_output[:, -1, :].unsqueeze(1).expand(
                -1, self.neg_sample_size, -1
            )
            neg_embed = self.model.item_embedding(neg_items)
            neg_scores = (seq_output_expand * neg_embed).sum(dim=-1)

            # BPR Loss
            loss = self._bpr_loss(pos_scores, neg_scores)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _sample_negatives(self, targets: torch.Tensor) -> torch.Tensor:
        """负采样"""
        batch_size = targets.shape[0]

        neg_items = torch.randint(
            1, self.n_items + 1,
            (batch_size, self.neg_sample_size)
        )

        return neg_items

    def _bpr_loss(self, pos_scores: torch.Tensor,
                  neg_scores: torch.Tensor) -> torch.Tensor:
        """
        BPR Loss

        -log(sigmoid(pos_score - neg_score))
        """
        pos_scores = pos_scores.unsqueeze(1)  # (batch, 1)
        diff = pos_scores - neg_scores  # (batch, neg_samples)

        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

        return loss

    def evaluate(self, dataloader, k_list: List[int] = [10, 20, 50]):
        """评估"""
        self.model.eval()

        metrics = {f'Hit@{k}': [] for k in k_list}
        metrics.update({f'NDCG@{k}': [] for k in k_list})

        with torch.no_grad():
            for batch in dataloader:
                item_seq = batch['item_seq']
                seq_mask = batch['seq_mask']
                target = batch['target']

                # 预测
                seq_output = self.model(item_seq, seq_mask)
                scores = self.model.predict(seq_output)  # (batch, n_items)

                # 排除序列中的物品
                for i in range(len(item_seq)):
                    for item in item_seq[i]:
                        if item > 0:
                            scores[i, item] = float('-inf')

                # 计算指标
                for k in k_list:
                    _, topk_indices = torch.topk(scores, k, dim=-1)

                    for i, target_item in enumerate(target):
                        hit = (target_item in topk_indices[i])

                        metrics[f'Hit@{k}'].append(float(hit))

                        if hit:
                            rank = (topk_indices[i] == target_item).nonzero().item()
                            ndcg = 1.0 / np.log2(rank + 2)
                        else:
                            ndcg = 0.0
                        metrics[f'NDCG@{k}'].append(ndcg)

        # 平均
        for key in metrics:
            metrics[key] = np.mean(metrics[key])

        return metrics


class WarmupLinearScheduler:
    """
    带预热的线性衰减学习率调度器
    """

    def __init__(self, optimizer, warmup_steps: int, max_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            lr_scale = self.current_step / self.warmup_steps
        else:
            lr_scale = max(0.0,
                          (self.max_steps - self.current_step) /
                          (self.max_steps - self.warmup_steps))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_scale
```

## 4. 推理与部署

### 4.1 在线推理

```python
class SASRecInference:
    """
    SASRec 在线推理
    """

    def __init__(self, model: SASRec, item_embeddings: np.ndarray):
        self.model = model
        self.model.eval()

        # 预计算物品嵌入
        self.item_embeddings = item_embeddings  # (n_items, embed_dim)

    def encode_sequence(self, item_seq: List[int], max_len: int = 50
                        ) -> np.ndarray:
        """
        编码序列，返回用户表示向量
        """
        # 预处理
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]

        seq_len = len(item_seq)
        padding_len = max_len - seq_len

        if padding_len > 0:
            item_seq = [0] * padding_len + item_seq

        # 创建 tensor
        item_tensor = torch.tensor([item_seq], dtype=torch.long)
        mask = torch.tensor([[0] * padding_len + [1] * seq_len], dtype=torch.float)

        # 前向传播
        with torch.no_grad():
            seq_output = self.model(item_tensor, mask)
            user_repr = seq_output[0, -1, :].numpy()  # (embed_dim,)

        return user_repr

    def get_recommendations(self,
                            item_seq: List[int],
                            exclude_items: set = None,
                            top_k: int = 100) -> List[Tuple[int, float]]:
        """
        获取推荐结果
        """
        # 编码序列
        user_repr = self.encode_sequence(item_seq)

        # 计算相似度
        scores = np.dot(self.item_embeddings, user_repr)

        # 排除已交互物品
        if exclude_items:
            for item in exclude_items:
                if 0 <= item < len(scores):
                    scores[item] = float('-inf')

        # Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        recommendations = [(int(idx), float(scores[idx])) for idx in top_indices]

        return recommendations
```

## 5. 变体与改进

### 5.1 SASRec + 位置编码改进

```python
class LearnablePositionSASRec(SASRec):
    """
    可学习位置编码的 SASRec
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 替换为可学习的位置编码
        del self.position_embedding

        # 使用相对位置编码
        self.relative_position_embedding = nn.Embedding(
            2 * kwargs['max_seq_len'] - 1,
            kwargs['n_heads']
        )


class RecencyBiasSASRec(SASRec):
    """
    带近因偏差的 SASRec

    给近期行为更高的权重
    """

    def __init__(self, *args, recency_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.recency_weight = recency_weight

    def forward(self, item_seq, seq_mask=None):
        seq_output = super().forward(item_seq, seq_mask)

        # 添加近因权重
        batch_size, seq_len, _ = seq_output.shape

        # 线性衰减权重
        weights = torch.linspace(
            1 - self.recency_weight,
            1 + self.recency_weight,
            seq_len,
            device=seq_output.device
        )

        seq_output = seq_output * weights.view(1, -1, 1)

        return seq_output
```

### 5.2 SASRec + 对比学习

```python
class ContrastiveSASRec(SASRec):
    """
    对比学习增强的 SASRec
    """

    def __init__(self, *args, temperature: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def contrastive_loss(self, seq_output: torch.Tensor,
                         aug_output: torch.Tensor) -> torch.Tensor:
        """
        对比损失

        seq_output: 原始序列输出
        aug_output: 增强后的序列输出
        """
        # 取最后位置
        z1 = F.normalize(seq_output[:, -1, :], dim=-1)
        z2 = F.normalize(aug_output[:, -1, :], dim=-1)

        batch_size = z1.shape[0]

        # 相似度矩阵
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # 正样本在对角线
        labels = torch.arange(batch_size, device=z1.device)

        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def augment_sequence(self, item_seq: torch.Tensor,
                         mask_ratio: float = 0.1
                         ) -> torch.Tensor:
        """
        序列增强: 随机mask
        """
        aug_seq = item_seq.clone()

        mask = torch.rand_like(aug_seq.float()) < mask_ratio
        aug_seq[mask] = 0  # mask to padding

        return aug_seq
```

## 6. 学习总结

### 6.1 核心要点

```
1. 自注意力: 捕捉长距离依赖
2. 因果mask: 只看历史，不看未来
3. 位置编码: 保留序列顺序
4. 并行计算: 比 RNN 更高效
```

### 6.2 调参建议

```
参数            推荐值           说明
──────────────────────────────────────
embed_dim       64-256          嵌入维度
n_heads         2-4             注意力头数
n_layers        2-4             层数
max_seq_len     50-200          最大序列长度
dropout         0.2-0.5         防止过拟合
learning_rate   1e-4 - 1e-3     学习率
```

### 6.3 与 BERT4Rec 对比

```
                SASRec          BERT4Rec
────────────────────────────────────────────
方向            单向            双向
预训练目标      因果LM          Masked LM
推理效率        高              需要mask
训练效率        高              更高 (并行)
效果            一般略低        一般略高
```
