# BST (Behavior Sequence Transformer) 学习文档

## 1. 算法基础认知

### 1.1 什么是 BST？

BST (Behavior Sequence Transformer) 是阿里巴巴提出的模型，将 Transformer 应用于用户行为序列建模，用于 CTR 预估。

### 1.2 核心创新

```
传统方法: 用户历史行为平均池化
BST:     使用 Transformer 捕捉序列中的时序关系和兴趣变化
```

### 1.3 模型架构

```
用户行为序列 → Transformer → 用户兴趣表示
                               ↓
其他特征 ──────────────────→ MLP → CTR 预测
```

## 2. 核心原理

### 2.1 序列嵌入

对于用户行为序列 $S = [i_1, i_2, ..., i_n]$:

1. **物品嵌入**: 每个物品 id 映射为向量
2. **位置嵌入**: 添加位置信息
3. **组合嵌入**: $E_i = e_{item_i} + e_{pos_i}$

### 2.2 Transformer Encoder

```
Multi-Head Self-Attention:
  Attention(Q, K, V) = softmax(QK^T / √d_k) V

Feed Forward Network:
  FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

Layer Normalization + Residual Connection
```

### 2.3 最终预测

$$\hat{y} = \sigma(W \cdot [\text{concat}(E_{target}, E_{other}, T_{output})] + b)$$

## 3. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    位置编码
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch_size, seq_len, d_model)

        返回:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder 层
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        # Multi-Head Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            src: (seq_len, batch, d_model)
            src_mask: (seq_len, seq_len)
            src_key_padding_mask: (batch, seq_len)

        返回:
            (seq_len, batch, d_model)
        """
        # Self-Attention with Residual
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout(src2))

        # Feed Forward with Residual
        src2 = self.feed_forward(src)
        src = self.norm2(src + src2)

        return src


class BST(nn.Module):
    """
    Behavior Sequence Transformer for CTR Prediction
    """

    def __init__(self,
                 # 特征配置
                 user_num: int,
                 item_num: int,
                 category_num: int,
                 other_feature_dims: List[int],
                 # 序列配置
                 seq_len: int = 50,
                 embed_dim: int = 64,
                 # Transformer 配置
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.2):
        """
        参数:
            user_num: 用户数量
            item_num: 物品数量
            category_num: 类别数量
            other_feature_dims: 其他特征的维度列表
            seq_len: 行为序列长度
            embed_dim: 嵌入维度
            nhead: 注意力头数
            num_encoder_layers: Encoder 层数
            dim_feedforward: FFN 隐藏层维度
            dropout: Dropout 比率
        """
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # ===== 用户特征嵌入 =====
        self.user_embedding = nn.Embedding(user_num, embed_dim)

        # ===== 物品特征嵌入（用于目标物品和行为序列） =====
        self.item_embedding = nn.Embedding(item_num, embed_dim)
        self.category_embedding = nn.Embedding(category_num, embed_dim // 4)

        # ===== 位置编码 =====
        self.pos_encoding = PositionalEncoding(embed_dim, seq_len, dropout)

        # ===== Transformer Encoder =====
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.ModuleList([
            encoder_layer for _ in range(num_encoder_layers)
        ])

        # ===== 其他特征嵌入 =====
        self.other_embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim // 4)
            for dim in other_feature_dims
        ])

        # ===== MLP 输出层 =====
        # 输入维度:
        # - 用户嵌入: embed_dim
        # - 目标物品嵌入: embed_dim + embed_dim//4 (item + category)
        # - 序列输出: embed_dim
        # - 其他特征: len(other_feature_dims) * embed_dim//4
        other_dim = len(other_feature_dims) * (embed_dim // 4)
        input_dim = embed_dim + embed_dim + embed_dim // 4 + embed_dim + other_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,
                user_id: torch.Tensor,
                target_item: torch.Tensor,
                target_category: torch.Tensor,
                behavior_seq: torch.Tensor,
                behavior_category_seq: torch.Tensor,
                behavior_mask: torch.Tensor,
                other_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            user_id: (batch,) 用户ID
            target_item: (batch,) 目标物品ID
            target_category: (batch,) 目标物品类别
            behavior_seq: (batch, seq_len) 行为序列物品ID
            behavior_category_seq: (batch, seq_len) 行为序列类别
            behavior_mask: (batch, seq_len) 行为序列掩码（1=有效，0=填充）
            other_features: (batch, num_other) 其他特征

        返回:
            (batch, 1) 预测点击概率
        """
        batch_size = user_id.size(0)

        # ===== 用户嵌入 =====
        user_embed = self.user_embedding(user_id)  # (batch, embed_dim)

        # ===== 目标物品嵌入 =====
        target_item_embed = self.item_embedding(target_item)  # (batch, embed_dim)
        target_cat_embed = self.category_embedding(target_category)  # (batch, embed_dim//4)
        target_embed = torch.cat([target_item_embed, target_cat_embed], dim=-1)

        # ===== 行为序列嵌入 =====
        behavior_item_embed = self.item_embedding(behavior_seq)  # (batch, seq_len, embed_dim)
        behavior_cat_embed = self.category_embedding(behavior_category_seq)  # (batch, seq_len, embed_dim//4)

        # 只用物品嵌入作为序列表示
        seq_embed = behavior_item_embed

        # 添加位置编码
        seq_embed = self.pos_encoding(seq_embed)  # (batch, seq_len, embed_dim)

        # Transformer 编码
        # 转换为 (seq_len, batch, embed_dim) 格式
        seq_embed = seq_embed.transpose(0, 1)

        # 创建 key padding mask (True 表示要 mask)
        key_padding_mask = (behavior_mask == 0)  # (batch, seq_len)

        for encoder_layer in self.transformer_encoder:
            seq_embed = encoder_layer(seq_embed, src_key_padding_mask=key_padding_mask)

        # 取最后一个位置的输出作为序列表示
        seq_output = seq_embed[-1]  # (batch, embed_dim)

        # 或者取平均
        # seq_output = seq_embed.mean(dim=0)

        # ===== 其他特征嵌入 =====
        other_embeds = []
        for i, embedding_layer in enumerate(self.other_embeddings):
            embed = embedding_layer(other_features[:, i])  # (batch, embed_dim//4)
            other_embeds.append(embed)
        other_embeds = torch.cat(other_embeds, dim=-1) if other_embeds else torch.zeros(batch_size, 0, device=user_id.device)

        # ===== 拼接所有特征 =====
        combined = torch.cat([
            user_embed,
            target_embed,
            seq_output,
            other_embeds
        ], dim=-1)

        # ===== MLP 预测 =====
        output = self.mlp(combined)
        output = self.sigmoid(output)

        return output


class BSTTrainer:
    """
    BST 训练器
    """

    def __init__(self, model: BST, learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()

    def train_step(self, batch: Dict) -> float:
        """训练一步"""
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(
            user_id=batch['user_id'],
            target_item=batch['target_item'],
            target_category=batch['target_category'],
            behavior_seq=batch['behavior_seq'],
            behavior_category_seq=batch['behavior_category_seq'],
            behavior_mask=batch['behavior_mask'],
            other_features=batch['other_features']
        )

        loss = self.criterion(pred.squeeze(), batch['label'].float())
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, batch: Dict) -> np.ndarray:
        """预测"""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(
                user_id=batch['user_id'],
                target_item=batch['target_item'],
                target_category=batch['target_category'],
                behavior_seq=batch['behavior_seq'],
                behavior_category_seq=batch['behavior_category_seq'],
                behavior_mask=batch['behavior_mask'],
                other_features=batch['other_features']
            )
        return pred.squeeze().cpu().numpy()


def create_mock_batch(batch_size: int = 32, seq_len: int = 20,
                     user_num: int = 10000, item_num: int = 50000,
                     category_num: int = 100, num_other_features: int = 5):
    """创建模拟数据"""
    return {
        'user_id': torch.randint(0, user_num, (batch_size,)),
        'target_item': torch.randint(0, item_num, (batch_size,)),
        'target_category': torch.randint(0, category_num, (batch_size,)),
        'behavior_seq': torch.randint(0, item_num, (batch_size, seq_len)),
        'behavior_category_seq': torch.randint(0, category_num, (batch_size, seq_len)),
        'behavior_mask': torch.ones(batch_size, seq_len),
        'other_features': torch.randint(0, 100, (batch_size, num_other_features)),
        'label': torch.randint(0, 2, (batch_size,)).float()
    }


def demo_bst():
    """BST 示例"""
    # 配置
    config = {
        'user_num': 10000,
        'item_num': 50000,
        'category_num': 100,
        'other_feature_dims': [50, 30, 20, 10, 5],
        'seq_len': 20,
        'embed_dim': 64,
        'nhead': 4,
        'num_encoder_layers': 2
    }

    # 创建模型
    model = BST(**config)

    # 创建模拟数据
    batch = create_mock_batch(
        batch_size=32,
        seq_len=20,
        user_num=config['user_num'],
        item_num=config['item_num'],
        category_num=config['category_num'],
        num_other_features=len(config['other_feature_dims'])
    )

    # 前向传播
    output = model(
        user_id=batch['user_id'],
        target_item=batch['target_item'],
        target_category=batch['target_category'],
        behavior_seq=batch['behavior_seq'],
        behavior_category_seq=batch['behavior_category_seq'],
        behavior_mask=batch['behavior_mask'],
        other_features=batch['other_features']
    )

    print(f"输出形状: {output.shape}")
    print(f"预测范围: [{output.min():.4f}, {output.max():.4f}]")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 训练一步
    trainer = BSTTrainer(model)
    loss = trainer.train_step(batch)
    print(f"训练损失: {loss:.4f}")


if __name__ == "__main__":
    demo_bst()
```

## 4. 序列处理技巧

### 4.1 序列截断与填充

```python
class SequenceProcessor:
    """
    序列预处理
    """

    def __init__(self, max_len: int = 50, padding_value: int = 0):
        self.max_len = max_len
        self.padding_value = padding_value

    def process(self, sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理变长序列

        返回:
            sequences: (batch, max_len)
            masks: (batch, max_len)
        """
        batch_size = len(sequences)
        result = torch.full((batch_size, self.max_len), self.padding_value, dtype=torch.long)
        masks = torch.zeros(batch_size, self.max_len)

        for i, seq in enumerate(sequences):
            # 截断或填充
            if len(seq) >= self.max_len:
                result[i] = torch.tensor(seq[-self.max_len:])
                masks[i] = 1
            else:
                result[i, -len(seq):] = torch.tensor(seq)
                masks[i, -len(seq):] = 1

        return result, masks

    def process_with_time_gap(self, sequences: List[List[int]],
                              timestamps: List[List[float]]) -> torch.Tensor:
        """
        处理带时间间隔的序列

        添加时间间隔嵌入
        """
        batch_size = len(sequences)
        time_gaps = torch.zeros(batch_size, self.max_len)

        for i, (seq, ts) in enumerate(zip(sequences, timestamps)):
            if len(seq) >= self.max_len:
                seq_ts = ts[-self.max_len:]
            else:
                seq_ts = [0] * (self.max_len - len(seq)) + ts

            # 计算与下一个行为的时间间隔
            gaps = [0]  # 第一个位置没有前一个
            for j in range(1, len(seq_ts)):
                if seq_ts[j-1] == 0:
                    gaps.append(0)
                else:
                    gaps.append(seq_ts[j] - seq_ts[j-1])

            # 归一化
            max_gap = max(gaps) if max(gaps) > 0 else 1
            time_gaps[i] = torch.tensor([g / max_gap for g in gaps])

        return time_gaps
```

### 4.2 Target Item 与序列交互

```python
class TargetAwareBST(nn.Module):
    """
    目标感知的 BST

    将目标物品与行为序列一起做 Attention
    """

    def __init__(self, item_num: int, embed_dim: int = 64,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__()

        self.item_embedding = nn.Embedding(item_num, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, behavior_seq: torch.Tensor,
                target_item: torch.Tensor,
                behavior_mask: torch.Tensor) -> torch.Tensor:
        """
        参数:
            behavior_seq: (batch, seq_len)
            target_item: (batch,)
            behavior_mask: (batch, seq_len)
        """
        batch_size = behavior_seq.size(0)
        seq_len = behavior_seq.size(1)

        # 嵌入
        behavior_embed = self.item_embedding(behavior_seq)  # (batch, seq_len, embed_dim)
        target_embed = self.item_embedding(target_item).unsqueeze(1)  # (batch, 1, embed_dim)

        # 拼接: [target, behaviors]
        combined = torch.cat([target_embed, behavior_embed], dim=1)  # (batch, seq_len+1, embed_dim)

        # 位置编码
        combined = self.pos_encoding(combined)

        # 构建掩码
        target_mask = torch.ones(batch_size, 1, device=behavior_seq.device)
        full_mask = torch.cat([target_mask, behavior_mask], dim=1)

        # Transformer
        key_padding_mask = (full_mask == 0)
        output = self.transformer(combined, src_key_padding_mask=key_padding_mask)

        # 取 target 位置的输出
        target_output = output[:, 0, :]

        # 预测
        pred = self.sigmoid(self.output(target_output))
        return pred
```

## 5. 与其他序列模型对比

| 模型 | 序列建模方式 | 长序列处理 | 训练复杂度 |
|------|--------------|------------|------------|
| DIN | Attention | 较差 | 低 |
| DIEN | GRU | 中等 | 中 |
| BST | Transformer | 好 | 高 |
| SASRec | Transformer | 好 | 中 |

## 6. 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| seq_len | 20-50 | 行为序列长度 |
| embed_dim | 64-128 | 嵌入维度 |
| nhead | 4-8 | 注意力头数 |
| num_layers | 2-4 | Transformer 层数 |
| dropout | 0.2-0.3 | 防止过拟合 |

## 7. 学习总结

### 7.1 核心要点

1. **序列建模**: 使用 Transformer 捕捉用户行为序列
2. **位置编码**: 保留时序信息
3. **目标物品**: 与序列一起建模

### 7.2 适用场景

- 用户行为序列较长
- 需要捕捉兴趣变化
- 序列中存在长期依赖

## 8. 练习题

1. 实现 BST 的多头注意力可视化。

2. 比较不同序列长度对 BST 性能的影响。

3. 实现 BST 的在线推理优化。
