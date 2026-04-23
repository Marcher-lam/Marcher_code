# DIN（深度兴趣网络） 学习文档

## 1. 算法基础认知

### 1.1 什么是 DIN？

DIN（Deep Interest Network）是阿里巴巴在 2018 年提出的 CTR 预估模型，核心创新是**使用注意力机制对用户历史行为序列进行加权聚合**。

### 1.2 动机

**传统 DNN 的问题：**

```python
# 传统 DNN 对用户历史行为的处理
user_history = [item1, item2, item3, ..., item50]  # 用户历史点击的 50 个物品

# 传统方法：简单池化
user_embedding = mean([emb(item) for item in user_history])

# 问题：
# 1. 所有历史物品同等重要
# 2. 忽略了用户兴趣的多样性
# 3. 没有考虑当前候选物品的相关性
```

**DIN 的洞察：**

用户兴趣是多样化的，与当前候选物品相关的历史行为更重要。

```
例如：
- 用户历史：点击了手机、衣服、电脑、鞋子
- 候选物品：笔记本电脑
- 相关历史：手机、电脑（权重高）
- 不相关历史：衣服、鞋子（权重低）
```

### 1.3 核心思想

$$V_u = f(V) = \sum_{i=1}^{K} a(V_i, V_a) \cdot V_i = \sum_{i=1}^{K} w_i \cdot V_i$$

其中：
- $V_u$：用户兴趣表示
- $V_i$：第 i 个历史物品的 Embedding
- $V_a$：候选物品的 Embedding
- $a(V_i, V_a)$：注意力分数，衡量历史物品与候选物品的相关性

## 2. 模型架构

### 2.1 整体架构

```
输入层
├── 用户特征
│   ├── 用户 ID
│   ├── 用户画像特征
│   └── 历史行为序列（关键！）
├── 物品特征
│   ├── 候选物品 ID
│   └── 物品属性
└── 上下文特征

        ↓

Embedding 层
    所有类别特征 → Embedding 向量

        ↓

兴趣提取层（DIN 核心）
    历史序列 × 候选物品 → 注意力加权聚合

        ↓

MLP 层
    拼接所有特征 → 多层感知机

        ↓

输出层
    Sigmoid → CTR 预测
```

### 2.2 注意力机制

**DIN 的注意力计算：**

$$a(V_i, V_a) = \frac{\exp(w \cdot \text{concat}(V_i, V_a, V_i - V_a, V_i \odot V_a))}{\sum_j \exp(w \cdot \text{concat}(V_j, V_a, V_j - V_a, V_j \odot V_a))}$$

**特点：**
1. 输入：历史物品 Embedding、候选物品 Embedding
2. 交互方式：拼接、差值、点积
3. 输出：归一化的注意力权重

## 3. 代码实现

### 3.1 完整 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionLayer(nn.Module):
    """
    DIN 的注意力层

    计算历史物品与候选物品的相关性
    """

    def __init__(self, embed_dim, hidden_units=[64], dropout=0.0):
        super().__init__()

        # 注意力网络
        layers = []
        input_dim = 4 * embed_dim  # concat: item + candidate + diff + prod

        for hidden in hidden_units:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden

        layers.append(nn.Linear(input_dim, 1))  # 输出注意力分数
        self.attention_mlp = nn.Sequential(*layers)

    def forward(self, history_emb, candidate_emb, mask=None):
        """
        参数:
            history_emb: (batch, seq_len, embed_dim) 历史物品 Embedding
            candidate_emb: (batch, embed_dim) 候选物品 Embedding
            mask: (batch, seq_len) 有效位置 mask（1=有效，0=填充）

        返回:
            weighted_interest: (batch, embed_dim) 加权聚合的用户兴趣
            attention_weights: (batch, seq_len) 注意力权重
        """
        batch_size, seq_len, embed_dim = history_emb.shape

        # 扩展候选物品维度
        candidate_emb = candidate_emb.unsqueeze(1)  # (batch, 1, embed_dim)
        candidate_emb = candidate_emb.expand(-1, seq_len, -1)  # (batch, seq_len, embed_dim)

        # 计算交互特征
        diff = history_emb - candidate_emb  # 差值
        prod = history_emb * candidate_emb  # 点积

        # 拼接
        concat = torch.cat([history_emb, candidate_emb, diff, prod], dim=-1)
        # (batch, seq_len, 4 * embed_dim)

        # 注意力分数
        attention_scores = self.attention_mlp(concat).squeeze(-1)  # (batch, seq_len)

        # 应用 mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Softmax 归一化
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, seq_len)

        # 加权聚合
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, seq_len, 1)
        weighted_interest = torch.sum(attention_weights * history_emb, dim=1)
        # (batch, embed_dim)

        return weighted_interest, attention_weights.squeeze(-1)


class DIN(nn.Module):
    """
    DIN (Deep Interest Network) 完整实现
    """

    def __init__(self, feature_configs, embed_dim=16,
                 mlp_hidden_units=[256, 128, 64], attention_hidden=[64],
                 dropout=0.2, use_dice=True):
        """
        参数:
            feature_configs: dict, 特征配置
                {
                    'user_id': {'type': 'categorical', 'vocab_size': 1000},
                    'item_id': {'type': 'categorical', 'vocab_size': 10000},
                    'history_item_ids': {'type': 'sequence', 'vocab_size': 10000, 'max_len': 50},
                    ...
                }
            embed_dim: int, Embedding 维度
            mlp_hidden_units: list, MLP 隐藏层维度
            attention_hidden: list, 注意力网络隐藏层
            dropout: float, Dropout 比例
            use_dice: bool, 是否使用 Dice 激活
        """
        super().__init__()

        self.feature_configs = feature_configs
        self.embed_dim = embed_dim

        # ========== Embedding 层 ==========
        self.embeddings = nn.ModuleDict()

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(config['vocab_size'], embed_dim)
            elif config['type'] == 'sequence':
                self.embeddings[name] = nn.Embedding(config['vocab_size'], embed_dim)

        # 初始化
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

        # ========== 注意力层 ==========
        self.attention_layer = AttentionLayer(
            embed_dim=embed_dim,
            hidden_units=attention_hidden,
            dropout=dropout
        )

        # ========== MLP 层 ==========
        # 计算输入维度
        self._compute_mlp_input_dim(feature_configs, embed_dim)

        mlp_layers = []
        input_dim = self.mlp_input_dim

        for hidden in mlp_hidden_units:
            mlp_layers.append(nn.Linear(input_dim, hidden))
            if use_dice:
                mlp_layers.append(Dice(hidden))
            else:
                mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden

        mlp_layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*mlp_layers)

    def _compute_mlp_input_dim(self, feature_configs, embed_dim):
        """计算 MLP 输入维度"""
        dim = 0

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                dim += embed_dim
            elif config['type'] == 'sequence':
                # 序列特征会被注意力聚合为一个向量
                dim += embed_dim
            elif config['type'] == 'numerical':
                dim += 1

        # 历史兴趣向量（注意力输出）
        if 'history_item_ids' in feature_configs:
            dim += embed_dim

        self.mlp_input_dim = dim

    def forward(self, features):
        """
        前向传播

        参数:
            features: dict, 特征字典
                {
                    'user_id': (batch,),
                    'item_id': (batch,),
                    'history_item_ids': (batch, max_len),
                    'history_mask': (batch, max_len),  # 可选
                    ...
                }

        返回:
            output: (batch, 1) CTR 预测
        """
        # 收集 Embedding
        embeddings = []

        # 处理普通特征
        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical':
                emb = self.embeddings[name](features[name])
                embeddings.append(emb)
            elif config['type'] == 'numerical':
                embeddings.append(features[name].unsqueeze(-1))

        # 处理历史序列（DIN 核心）
        if 'history_item_ids' in features:
            history_emb = self.embeddings['history_item_ids'](features['history_item_ids'])
            # (batch, seq_len, embed_dim)

            # 候选物品 Embedding
            candidate_emb = self.embeddings['item_id'](features['item_id'])
            # (batch, embed_dim)

            # 注意力加权
            mask = features.get('history_mask')
            weighted_interest, _ = self.attention_layer(history_emb, candidate_emb, mask)

            embeddings.append(weighted_interest)

        # 拼接所有特征
        concat = torch.cat(embeddings, dim=-1)

        # MLP 预测
        output = self.mlp(concat)
        output = torch.sigmoid(output)

        return output


class Dice(nn.Module):
    """
    Dice 激活函数（DIN 论文提出）

    改进版的 PReLU，使用概率调整
    """

    def __init__(self, num_features, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # 计算 sigmoid 的概率
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # 标准化
        x_norm = (x - mean) / (std + self.eps)

        # 概率
        p = torch.sigmoid(x_norm)

        # Dice 输出
        return p * x + (1 - p) * self.alpha * x
```

### 3.2 训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DINDataset(Dataset):
    """DIN 训练数据集"""

    def __init__(self, data, max_history_len=50):
        """
        参数:
            data: list of dict
                {
                    'user_id': int,
                    'item_id': int,
                    'history_item_ids': list,
                    'click': int,  # 0 or 1
                    ...
                }
        """
        self.data = data
        self.max_history_len = max_history_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 处理历史序列
        history = sample['history_item_ids'][-self.max_history_len:]
        history_len = len(history)

        # 填充
        if history_len < self.max_history_len:
            history = history + [0] * (self.max_history_len - history_len)

        # 创建 mask
        mask = [1] * history_len + [0] * (self.max_history_len - history_len)

        return {
            'user_id': torch.LongTensor([sample['user_id']]),
            'item_id': torch.LongTensor([sample['item_id']]),
            'history_item_ids': torch.LongTensor(history),
            'history_mask': torch.FloatTensor(mask),
            'click': torch.FloatTensor([sample['click']])
        }


def train_din():
    """训练 DIN 模型"""
    # 配置
    config = {
        'num_users': 10000,
        'num_items': 100000,
        'embed_dim': 64,
        'max_history_len': 50,
        'batch_size': 256,
        'learning_rate': 0.001,
        'epochs': 10,
    }

    # 特征配置
    feature_configs = {
        'user_id': {'type': 'categorical', 'vocab_size': config['num_users']},
        'item_id': {'type': 'categorical', 'vocab_size': config['num_items']},
        'history_item_ids': {
            'type': 'sequence',
            'vocab_size': config['num_items'],
            'max_len': config['max_history_len']
        }
    }

    # 创建模型
    model = DIN(
        feature_configs=feature_configs,
        embed_dim=config['embed_dim'],
        mlp_hidden_units=[256, 128, 64]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss()

    # 模拟数据
    train_data = [
        {
            'user_id': np.random.randint(0, config['num_users']),
            'item_id': np.random.randint(0, config['num_items']),
            'history_item_ids': [np.random.randint(0, config['num_items'])
                                for _ in range(np.random.randint(1, 30))],
            'click': np.random.randint(0, 2)
        }
        for _ in range(10000)
    ]

    train_dataset = DINDataset(train_data, config['max_history_len'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # 训练
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for batch in train_loader:
            # 移动到设备
            features = {
                'user_id': batch['user_id'].squeeze(-1).to(device),
                'item_id': batch['item_id'].squeeze(-1).to(device),
                'history_item_ids': batch['history_item_ids'].to(device),
                'history_mask': batch['history_mask'].to(device)
            }
            labels = batch['click'].to(device)

            # 前向传播
            optimizer.zero_grad()
            predictions = model(features)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    model = train_din()
    print("DIN 训练完成！")
```

## 4. 关键技术点

### 4.1 Dice 激活函数

DIN 论文提出的新激活函数，解决 PReLU 的局限：

```python
# PReLU
# output = max(0, x) + alpha * min(0, x)

# Dice
# output = p * x + (1 - p) * alpha * x
# 其中 p = sigmoid((x - mean) / std)
```

**优势：**
- 自适应调整激活强度
- 对不同分布的数据更鲁棒

### 4.2 小批量感知正则化

```python
class MiniBatchAwareRegularization(nn.Module):
    """
    小批量感知正则化（MBA）

    只对当前 batch 中出现的特征进行正则化
    """

    def __init__(self, embedding_layers, lambda_reg=0.001):
        super().__init__()
        self.embedding_layers = embedding_layers
        self.lambda_reg = lambda_reg

    def forward(self, features):
        """计算正则化损失"""
        reg_loss = 0

        for name, emb_layer in self.embedding_layers.items():
            if name in features:
                # 获取当前 batch 使用的特征
                indices = features[name].unique()
                weights = emb_layer.weight[indices]

                # L2 正则
                reg_loss += torch.sum(weights ** 2)

        return self.lambda_reg * reg_loss
```

## 5. 与其他模型对比

| 模型 | 历史行为处理 | 注意力机制 | 适用场景 |
|------|--------------|------------|----------|
| Wide&Deep | 平均池化 | 无 | 通用 |
| DeepFM | 平均池化 | 无 | 通用 |
| DIN | 加权聚合 | 有 | 丰富历史行为 |
| DIEN | 兴趣演化 | 有 | 序列行为 |

## 6. 学习总结

### 6.1 核心贡献

1. **注意力机制**：历史行为根据候选物品加权
2. **Dice 激活**：改进的激活函数
3. **MBA 正则**：高效的稀疏特征正则化

### 6.2 适用场景

- 用户历史行为丰富
- 需要个性化推荐
- 电商、内容推荐

## 7. 练习题

1. 实现 DIN 的注意力机制，并可视化注意力权重。

2. 比较 DIN 与传统 DNN（简单平均池化）的效果差异。

3. 分析 Dice 激活函数相比 ReLU/PReLU 的优势。
