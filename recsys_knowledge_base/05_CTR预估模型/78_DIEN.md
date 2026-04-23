# DIEN 学习文档

## 1. 算法基础认知

### 1.1 什么是 DIEN？

DIEN（Deep Interest Evolution Network）是阿里巴巴在 2019 年提出的 CTR 预估模型。它是 DIN 的改进版本，专注于建模用户兴趣的**演化过程**。

### 1.2 动机

**DIN 的局限：**
- 只关注当前兴趣，忽略兴趣变化
- 行为序列是扁平的，没有时序建模
- 无法捕获兴趣的演化趋势

**DIEN 的改进：**
- 提取用户兴趣序列
- 建模兴趣演化过程
- 使用 GRU + 注意力机制

### 1.3 应用场景

- 电商推荐：用户兴趣随时间演化
- 视频推荐：观看偏好的变化
- 广告投放：用户行为序列建模

## 2. 模型架构

### 2.1 整体结构

```
用户行为序列 [b1, b2, ..., bT]
           ↓
    Behavior Embedding Layer
           ↓
    Interest Extractor Layer (GRU)
           ↓
    Interest Evolving Layer (AUGRU)
           ↓
    与候选广告 Embedding 注意力
           ↓
    MLP + 输出
```

### 2.2 关键组件

1. **Interest Extractor Layer**
   - 使用 GRU 提取兴趣序列
   - 辅助损失函数帮助学习

2. **Interest Evolving Layer**
   - AUGRU（GRU with Attentional Update Gate）
   - 建模兴趣演化

3. **辅助损失**
   - 监督兴趣提取质量
   - 使用下一行为预测

## 3. PyTorch 完整实现

### 3.1 DIEN 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InterestExtractor(nn.Module):
    """
    兴趣提取层

    使用 GRU 从行为序列中提取兴趣序列
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.auxiliary_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, behavior_emb):
        """
        前向传播

        参数:
            behavior_emb: (batch, seq_len, input_dim)

        返回:
            interest_sequence: (batch, seq_len, hidden_dim)
            auxiliary_output: (batch, seq_len, input_dim) 辅助任务输出
        """
        # GRU 提取兴趣序列
        interest_seq, _ = self.gru(behavior_emb)  # (batch, seq_len, hidden_dim)

        # 辅助任务：预测下一行为
        auxiliary_output = self.auxiliary_fc(interest_seq)  # (batch, seq_len, input_dim)

        return interest_seq, auxiliary_output


class AUGRUCell(nn.Module):
    """
    AUGRU Cell: Attentional Update Gate GRU

    将注意力权重融入 GRU 的更新门
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        # GRU 参数
        self.W_ir = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(hidden_dim))

        self.W_iz = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias_z = nn.Parameter(torch.zeros(hidden_dim))

        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias_n = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h, attention):
        """
        参数:
            x: (batch, input_dim)
            h: (batch, hidden_dim)
            attention: (batch, 1) 注意力权重

        返回:
            h_new: (batch, hidden_dim)
        """
        # 重置门
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h) + self.bias_r)

        # 更新门
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h) + self.bias_z)

        # 注意力调整的更新门
        # z' = attention * z
        z_tilde = attention * z

        # 候选隐状态
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h) + self.bias_n)

        # 新隐状态
        h_new = (1 - z_tilde) * n + z_tilde * h

        return h_new


class InterestEvolving(nn.Module):
    """
    兴趣演化层

    使用 AUGRU 建模兴趣演化过程
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.augru_cell = AUGRUCell(input_dim, hidden_dim)

    def forward(self, interest_seq, candidate_emb, attention_mask=None):
        """
        前向传播

        参数:
            interest_seq: (batch, seq_len, hidden_dim)
            candidate_emb: (batch, embed_dim) 候选物品嵌入
            attention_mask: (batch, seq_len) 可选

        返回:
            final_interest: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
        """
        batch_size, seq_len, _ = interest_seq.shape

        # 计算注意力权重（兴趣序列与候选物品的相关性）
        # interest_seq: (batch, seq_len, hidden_dim)
        # candidate_emb: (batch, embed_dim)
        attention_scores = torch.bmm(
            interest_seq,
            candidate_emb.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq_len)

        # 应用 mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, seq_len)

        # 保存注意力权重
        self.attention_weights = attention_weights

        # AUGRU 演化
        h = torch.zeros(batch_size, self.hidden_dim, device=interest_seq.device)

        for t in range(seq_len):
            x = interest_seq[:, t, :]
            attn = attention_weights[:, t:t+1]  # (batch, 1)
            h = self.augru_cell(x, h, attn)

        return h, attention_weights


class DIEN(nn.Module):
    """
    DIEN: Deep Interest Evolution Network

    论文: Deep Interest Evolution Network for Click-Through Rate Prediction
          (AAAI 2019)

    组成部分:
    1. Embedding Layer
    2. Interest Extractor Layer
    3. Interest Evolving Layer
    4. MLP + 输出
    """

    def __init__(self, feature_configs, embed_dim=16,
                 interest_hidden_dim=64, mlp_hidden_units=[256, 128],
                 dropout=0.2, aux_loss_weight=0.1):
        """
        参数:
            feature_configs: dict, 特征配置
            embed_dim: 嵌入维度
            interest_hidden_dim: 兴趣隐状态维度
            mlp_hidden_units: MLP 隐藏层
            dropout: Dropout
            aux_loss_weight: 辅助损失权重
        """
        super().__init__()

        self.feature_configs = feature_configs
        self.embed_dim = embed_dim
        self.aux_loss_weight = aux_loss_weight

        # ========== Embedding 层 ==========
        self.embeddings = nn.ModuleDict()

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(
                    config['vocab_size'],
                    config.get('embed_dim', embed_dim)
                )
            elif config['type'] == 'sequence':
                self.embeddings[name] = nn.Embedding(
                    config['vocab_size'],
                    config.get('embed_dim', embed_dim)
                )

        # ========== Interest Extractor ==========
        # 计算行为嵌入维度
        behavior_dim = self._get_behavior_dim()

        self.interest_extractor = InterestExtractor(
            input_dim=behavior_dim,
            hidden_dim=interest_hidden_dim
        )

        # ========== Interest Evolving ==========
        self.interest_evolving = InterestEvolving(
            input_dim=interest_hidden_dim,
            hidden_dim=interest_hidden_dim
        )

        # ========== MLP ==========
        # 输入维度：兴趣表示 + 其他特征
        other_feature_dim = self._get_other_feature_dim()
        mlp_input_dim = interest_hidden_dim + other_feature_dim

        mlp_layers = []
        for hidden in mlp_hidden_units:
            mlp_layers.append(nn.Linear(mlp_input_dim, hidden))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            mlp_input_dim = hidden

        self.mlp = nn.Sequential(*mlp_layers)

        # 输出层
        self.output_layer = nn.Linear(mlp_hidden_units[-1], 1)

    def _get_behavior_dim(self):
        """获取行为序列特征维度"""
        dim = 0
        for name, config in self.feature_configs.items():
            if config['type'] == 'sequence':
                dim += config.get('embed_dim', self.embed_dim)
        return dim

    def _get_other_feature_dim(self):
        """获取其他特征维度"""
        dim = 0
        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical' and not config.get('is_target', False):
                dim += config.get('embed_dim', self.embed_dim)
            elif config['type'] == 'numerical':
                dim += 1
        return dim

    def forward(self, features):
        """
        前向传播

        参数:
            features: dict, 包含:
                - behavior_seq: (batch, seq_len) 行为序列
                - target_item: (batch,) 目标物品
                - 其他特征...

        返回:
            output: (batch, 1) CTR 预测
            aux_loss: 辅助损失
        """
        # ========== 提取行为嵌入 ==========
        behavior_emb_list = []
        behavior_seq = None
        behavior_mask = None

        for name, config in self.feature_configs.items():
            if config['type'] == 'sequence':
                emb = self.embeddings[name](features[name])
                behavior_emb_list.append(emb)
                behavior_seq = features[name]
                if f'{name}_mask' in features:
                    behavior_mask = features[f'{name}_mask']

        if behavior_emb_list:
            behavior_emb = torch.cat(behavior_emb_list, dim=-1)
        else:
            raise ValueError("No behavior sequence found")

        # ========== Interest Extractor ==========
        interest_seq, aux_output = self.interest_extractor(behavior_emb)

        # ========== 目标物品嵌入 ==========
        target_emb_list = []
        for name, config in self.feature_configs.items():
            if config.get('is_target', False):
                emb = self.embeddings[name](features[name])
                if len(emb.shape) == 3:
                    emb = emb.squeeze(1)
                target_emb_list.append(emb)

        if target_emb_list:
            target_emb = torch.cat(target_emb_list, dim=-1)
        else:
            # 使用行为维度的平均
            target_emb = behavior_emb.mean(dim=1)

        # ========== Interest Evolving ==========
        final_interest, attention_weights = self.interest_evolving(
            interest_seq, target_emb, behavior_mask
        )

        # ========== 其他特征 ==========
        other_features = []
        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical' and not config.get('is_behavior', False):
                emb = self.embeddings[name](features[name])
                if len(emb.shape) == 3:
                    emb = emb.squeeze(1)
                other_features.append(emb)
            elif config['type'] == 'numerical':
                val = features[name]
                if len(val.shape) == 1:
                    val = val.unsqueeze(-1)
                other_features.append(val)

        # ========== 拼接 ==========
        if other_features:
            other_emb = torch.cat(other_features, dim=-1)
            mlp_input = torch.cat([final_interest, other_emb], dim=-1)
        else:
            mlp_input = final_interest

        # ========== MLP ==========
        mlp_output = self.mlp(mlp_input)

        # ========== 输出 ==========
        logit = self.output_layer(mlp_output)
        output = torch.sigmoid(logit)

        # ========== 辅助损失 ==========
        # 预测下一行为
        # 这里简化处理，实际应该用真实的下一行为作为标签
        aux_loss = self._compute_aux_loss(aux_output, behavior_emb)

        return output, aux_loss

    def _compute_aux_loss(self, aux_output, behavior_emb):
        """计算辅助损失"""
        # 预测 t 时刻的行为，使用 t+1 时刻的真实行为
        # aux_output: (batch, seq_len, dim)
        # behavior_emb: (batch, seq_len, dim)

        # 简化：使用负采样损失
        batch_size, seq_len, dim = aux_output.shape

        # 正样本：shift 一位
        pos_emb = behavior_emb[:, 1:, :]  # (batch, seq_len-1, dim)
        pred_emb = aux_output[:, :-1, :]  # (batch, seq_len-1, dim)

        # 负样本：随机采样
        neg_idx = torch.randint(0, batch_size, (batch_size, seq_len - 1))
        neg_emb = behavior_emb[neg_idx, torch.arange(seq_len - 1)]  # 简化

        # BPR 损失
        pos_score = (pred_emb * pos_emb).sum(dim=-1)
        neg_score = (pred_emb * neg_emb).sum(dim=-1)

        aux_loss = -F.logsigmoid(pos_score - neg_score).mean()

        return aux_loss

    def get_attention_weights(self):
        """获取注意力权重（用于分析）"""
        return self.attention_weights


class DIENSimplified(nn.Module):
    """
    简化版 DIEN

    用于快速实验和理解
    """

    def __init__(self, n_items, embed_dim=64, hidden_dim=64,
                 mlp_units=[128, 64], max_seq_len=50):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 物品嵌入
        self.item_embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)

        # Interest Extractor (GRU)
        self.gru_extractor = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # Interest Evolving (AUGRU)
        self.gru_evolving = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 注意力层
        self.attention = nn.Linear(hidden_dim, 1)

        # MLP
        mlp_input = hidden_dim + embed_dim  # 兴趣 + 目标物品
        mlp_layers = []
        for unit in mlp_units:
            mlp_layers.append(nn.Linear(mlp_input, unit))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            mlp_input = unit
        self.mlp = nn.Sequential(*mlp_layers)

        # 输出
        self.output = nn.Linear(mlp_units[-1], 1)

    def forward(self, behavior_seq, target_item, seq_mask=None):
        """
        参数:
            behavior_seq: (batch, seq_len) 行为序列
            target_item: (batch,) 目标物品
            seq_mask: (batch, seq_len) 序列 mask

        返回:
            output: (batch, 1) 预测
        """
        batch_size = behavior_seq.size(0)

        # 嵌入
        behavior_emb = self.item_embedding(behavior_seq)  # (batch, seq_len, embed_dim)
        target_emb = self.item_embedding(target_item)      # (batch, embed_dim)

        # Interest Extractor
        interest_seq, _ = self.gru_extractor(behavior_emb)  # (batch, seq_len, hidden_dim)

        # 计算注意力
        attn_scores = torch.bmm(
            interest_seq,
            target_emb.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq_len)

        if seq_mask is not None:
            attn_scores = attn_scores.masked_fill(seq_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)

        # Interest Evolving (使用注意力加权的输入)
        weighted_interest = interest_seq * attn_weights.unsqueeze(-1)
        evolving_output, _ = self.gru_evolving(weighted_interest)
        final_interest = evolving_output[:, -1, :]  # (batch, hidden_dim)

        # MLP
        mlp_input = torch.cat([final_interest, target_emb], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # 输出
        logit = self.output(mlp_output)
        output = torch.sigmoid(logit)

        return output


# 使用示例
if __name__ == "__main__":
    # 配置
    n_items = 10000
    batch_size = 32
    seq_len = 20

    # 创建简化版模型
    model = DIENSimplified(
        n_items=n_items,
        embed_dim=64,
        hidden_dim=64,
        mlp_units=[128, 64]
    )

    # 模拟输入
    behavior_seq = torch.randint(0, n_items, (batch_size, seq_len))
    target_item = torch.randint(0, n_items, (batch_size,))
    seq_mask = torch.ones(batch_size, seq_len)

    # 前向传播
    output = model(behavior_seq, target_item, seq_mask)
    print(f"输出形状: {output.shape}")
    print(f"预测值范围: [{output.min():.4f}, {output.max():.4f}]")
```

### 3.2 训练示例

```python
from torch.utils.data import Dataset, DataLoader


class DIENDataset(Dataset):
    """DIEN 数据集"""

    def __init__(self, data, n_items, max_seq_len=50):
        """
        参数:
            data: [(user_id, [behavior_seq], target_item, label), ...]
            n_items: 物品数量
            max_seq_len: 最大序列长度
        """
        self.data = data
        self.n_items = n_items
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, behavior_seq, target_item, label = self.data[idx]

        # 截断/填充序列
        if len(behavior_seq) > self.max_seq_len:
            behavior_seq = behavior_seq[-self.max_seq_len:]

        seq_len = len(behavior_seq)
        behavior_seq_padded = [0] * (self.max_seq_len - seq_len) + behavior_seq
        seq_mask = [0] * (self.max_seq_len - seq_len) + [1] * seq_len

        return {
            'behavior_seq': torch.LongTensor(behavior_seq_padded),
            'seq_mask': torch.FloatTensor(seq_mask),
            'target_item': torch.LongTensor([target_item]),
            'label': torch.FloatTensor([label])
        }


def train_dien():
    """训练 DIEN"""
    # 配置
    config = {
        'n_items': 10000,
        'embed_dim': 64,
        'hidden_dim': 64,
        'mlp_units': [128, 64],
        'max_seq_len': 50,
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 10,
        'n_samples': 10000
    }

    # 生成模拟数据
    data = []
    for _ in range(config['n_samples']):
        seq_len = np.random.randint(5, 30)
        behavior_seq = np.random.randint(1, config['n_items'], seq_len).tolist()
        target_item = np.random.randint(1, config['n_items'])
        label = np.random.randint(0, 2)
        user_id = np.random.randint(0, 1000)
        data.append((user_id, behavior_seq, target_item, label))

    # 数据集
    dataset = DIENDataset(data, config['n_items'], config['max_seq_len'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DIENSimplified(
        n_items=config['n_items'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        mlp_units=config['mlp_units']
    ).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss()

    # 训练
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for batch in dataloader:
            behavior_seq = batch['behavior_seq'].to(device)
            seq_mask = batch['seq_mask'].to(device)
            target_item = batch['target_item'].squeeze().to(device)
            label = batch['label'].squeeze().to(device)

            optimizer.zero_grad()

            output = model(behavior_seq, target_item, seq_mask).squeeze()
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {total_loss / len(dataloader):.4f}")

    return model


if __name__ == "__main__":
    model = train_dien()
    print("DIEN 训练完成！")
```

## 4. DIEN vs DIN vs BST

### 4.1 模型对比

| 模型 | 序列建模 | 兴趣演化 | 时序信息 |
|------|----------|----------|----------|
| DIN | 注意力 | 无 | 无 |
| DIEN | GRU | AUGRU | 有 |
| BST | Transformer | 自注意力 | 有 |

### 4.2 适用场景

**DIN：**
- 行为序列较短
- 只关注当前兴趣

**DIEN：**
- 需要建模兴趣变化
- 行为有序列性

**BST：**
- 需要长程依赖
- 计算资源充足

## 5. 调参建议

### 5.1 模型参数

| 参数 | 推荐值 |
|------|--------|
| embed_dim | 64-128 |
| hidden_dim | 64-128 |
| mlp_units | [256, 128] |
| aux_loss_weight | 0.1 |

### 5.2 训练参数

| 参数 | 推荐值 |
|------|--------|
| learning_rate | 0.001 |
| batch_size | 256-1024 |
| max_seq_len | 50 |

## 6. 学习总结

### 6.1 核心要点

1. **兴趣演化**：用户兴趣是动态变化的
2. **GRU 建模**：捕获序列依赖
3. **注意力机制**：聚焦相关行为

### 6.2 关键创新

- **AUGRU**：注意力融入更新门
- **辅助损失**：帮助兴趣提取

## 7. 练习题

1. 实现 AUGRU 的并行版本。

2. 比较不同序列长度对 DIEN 效果的影响。

3. 设计一个结合 DIEN 和 Transformer 的模型。
