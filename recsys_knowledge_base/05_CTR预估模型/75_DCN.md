# DCN（深度交叉网络） 学习文档

## 1. 算法基础认知

### 1.1 什么是 DCN？

DCN（Deep & Cross Network）是 Google 在 2017 年提出的 CTR 预估模型。它的核心创新是设计了**交叉网络（Cross Network）**来显式地学习有界阶的特征交叉。

### 1.2 动机

**DNN 学习特征交叉的问题：**

1. **隐式学习**：DNN 通过 MLP 隐式学习特征交叉
2. **低效**：需要大量参数和深层次网络
3. **不可控**：难以控制交叉的阶数

**DCN 的解决方案：**

- **交叉网络**：显式、高效地学习有界阶特征交叉
- **深度网络**：学习高阶隐式特征交叉
- **联合训练**：结合两者的优势

### 1.3 模型架构

```
输入（Embedding + Dense Features）
            ↓
    ┌───────┴───────┐
    ↓               ↓
交叉网络         深度网络
(Cross Net)     (Deep Net)
    ↓               ↓
    └───────┬───────┘
            ↓
        拼接层
            ↓
        输出层
```

## 2. 交叉网络原理

### 2.1 交叉层

交叉网络的核心是交叉层（Cross Layer）：

$$x_{l+1} = x_0 \odot (W_l \cdot x_l + b_l) + x_l$$

其中：
- $x_0$：原始输入（第 0 层）
- $x_l$：第 l 层的输出
- $W_l, b_l$：第 l 层的权重和偏置
- $\odot$：逐元素乘法

### 2.2 交叉网络的特性

**1. 有界阶数：**
- L 层交叉网络学习的是 L+1 阶特征交叉
- 阶数受网络层数控制

**2. 残差连接：**
- 每层都有残差连接，避免梯度消失

**3. 参数高效：**
- 每层只有 d 个参数（d 是输入维度）

### 2.3 交叉的直观理解

```python
# 第 1 层交叉
x_1 = x_0 * (W_0 @ x_0 + b_0) + x_0
# x_0 * (W_0 @ x_0) 产生 2 阶交叉

# 第 2 层交叉
x_2 = x_0 * (W_1 @ x_1 + b_1) + x_1
# x_0 * (W_1 @ x_1) 产生 3 阶交叉

# 以此类推...
```

## 3. DCN 完整实现

### 3.1 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayer(nn.Module):
    """
    DCN 的交叉层
    """

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0, x):
        """
        参数:
            x0: 原始输入 (batch, input_dim)
            x: 当前层输入 (batch, input_dim)

        返回:
            output: 下一层输入 (batch, input_dim)
        """
        # x_0 ⊙ (W @ x + b) + x
        output = x0 * self.weight(x) + x
        return output


class CrossNetwork(nn.Module):
    """
    交叉网络
    """

    def __init__(self, input_dim, num_layers=6):
        """
        参数:
            input_dim: 输入维度
            num_layers: 交叉层数量
        """
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入 (batch, input_dim)

        返回:
            output: 交叉网络输出 (batch, input_dim)
        """
        x0 = x
        for layer in self.cross_layers:
            x = layer(x0, x)
        return x


class DeepNetwork(nn.Module):
    """
    深度网络（普通 MLP）
    """

    def __init__(self, input_dim, hidden_units=[256, 128, 64], dropout=0.2):
        super().__init__()

        layers = []
        for hidden in hidden_units:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DCN(nn.Module):
    """
    DCN: Deep & Cross Network

    论文: Deep & Cross Network for Ad Click Predictions (ADKDD'17)
    """

    def __init__(self, feature_configs, embed_dim=16,
                 cross_num_layers=6, deep_hidden_units=[256, 128, 64],
                 dropout=0.2):
        """
        参数:
            feature_configs: dict, 特征配置
            embed_dim: Embedding 维度
            cross_num_layers: 交叉网络层数
            deep_hidden_units: 深度网络隐藏层
            dropout: Dropout 比例
        """
        super().__init__()

        # ========== Embedding 层 ==========
        self.embeddings = nn.ModuleDict()
        self.feature_configs = feature_configs

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(
                    config['vocab_size'],
                    config.get('embed_dim', embed_dim)
                )

        # 计算输入维度
        self._compute_input_dim(feature_configs, embed_dim)

        # ========== 交叉网络 ==========
        self.cross_net = CrossNetwork(
            input_dim=self.input_dim,
            num_layers=cross_num_layers
        )

        # ========== 深度网络 ==========
        self.deep_net = DeepNetwork(
            input_dim=self.input_dim,
            hidden_units=deep_hidden_units,
            dropout=dropout
        )

        # ========== 输出层 ==========
        # 交叉网络输出 + 深度网络输出
        cross_output_dim = self.input_dim
        deep_output_dim = deep_hidden_units[-1]
        total_dim = cross_output_dim + deep_output_dim

        self.output_layer = nn.Sequential(
            nn.Linear(total_dim, 1),
            nn.Sigmoid()
        )

    def _compute_input_dim(self, feature_configs, embed_dim):
        """计算输入维度"""
        dim = 0
        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                dim += config.get('embed_dim', embed_dim)
            elif config['type'] == 'numerical':
                dim += 1
        self.input_dim = dim

    def forward(self, features):
        """
        前向传播

        参数:
            features: dict, 特征字典

        返回:
            output: (batch, 1) CTR 预测
        """
        # Embedding
        embeddings = []
        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical':
                emb = self.embeddings[name](features[name])
                if len(emb.shape) == 3:
                    emb = emb.squeeze(1)
                embeddings.append(emb)
            elif config['type'] == 'numerical':
                val = features[name]
                if len(val.shape) == 1:
                    val = val.unsqueeze(-1)
                embeddings.append(val)

        # 拼接
        x = torch.cat(embeddings, dim=-1)  # (batch, input_dim)

        # 交叉网络
        cross_output = self.cross_net(x)  # (batch, input_dim)

        # 深度网络
        deep_output = self.deep_net(x)  # (batch, deep_hidden_units[-1])

        # 拼接
        combined = torch.cat([cross_output, deep_output], dim=-1)

        # 输出
        output = self.output_layer(combined)

        return output


class DCNV2(nn.Module):
    """
    DCN-V2: Improved Deep & Cross Network

    论文: DCN V2: Improved Deep & Cross Network for Web-Scale Learning
    to Rank Systems (WWW'21)

    改进:
    1. 使用矩阵替代向量作为交叉层权重
    2. 提出 MoE 版本减少参数
    """

    def __init__(self, feature_configs, embed_dim=16,
                 cross_num_layers=6, deep_hidden_units=[256, 128, 64],
                 dropout=0.2, use_moe=False, num_experts=4):
        super().__init__()

        self.use_moe = use_moe

        # Embedding 层
        self.embeddings = nn.ModuleDict()
        self.feature_configs = feature_configs

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(
                    config['vocab_size'],
                    config.get('embed_dim', embed_dim)
                )

        self._compute_input_dim(feature_configs, embed_dim)

        # 交叉网络 V2
        if use_moe:
            self.cross_net = CrossNetworkMoE(
                input_dim=self.input_dim,
                num_layers=cross_num_layers,
                num_experts=num_experts
            )
        else:
            self.cross_net = CrossNetworkV2(
                input_dim=self.input_dim,
                num_layers=cross_num_layers
            )

        # 深度网络
        self.deep_net = DeepNetwork(
            input_dim=self.input_dim,
            hidden_units=deep_hidden_units,
            dropout=dropout
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.input_dim + deep_hidden_units[-1], 1),
            nn.Sigmoid()
        )

    def _compute_input_dim(self, feature_configs, embed_dim):
        dim = 0
        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                dim += config.get('embed_dim', embed_dim)
            elif config['type'] == 'numerical':
                dim += 1
        self.input_dim = dim

    def forward(self, features):
        # Embedding
        embeddings = []
        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical':
                emb = self.embeddings[name](features[name])
                if len(emb.shape) == 3:
                    emb = emb.squeeze(1)
                embeddings.append(emb)
            elif config['type'] == 'numerical':
                val = features[name]
                if len(val.shape) == 1:
                    val = val.unsqueeze(-1)
                embeddings.append(val)

        x = torch.cat(embeddings, dim=-1)

        # 交叉网络
        cross_output = self.cross_net(x)

        # 深度网络
        deep_output = self.deep_net(x)

        # 拼接输出
        combined = torch.cat([cross_output, deep_output], dim=-1)
        output = self.output_layer(combined)

        return output


class CrossLayerV2(nn.Module):
    """
    DCN-V2 交叉层
    使用矩阵权重替代向量权重
    """

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x):
        return x * self.weight(x) + x


class CrossNetworkV2(nn.Module):
    """V2 交叉网络"""

    def __init__(self, input_dim, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossLayerV2(input_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CrossNetworkMoE(nn.Module):
    """
    MoE 版本的交叉网络
    减少参数量
    """

    def __init__(self, input_dim, num_layers=6, num_experts=4, low_rank=32):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        # 每层有多个专家
        self.experts = nn.ModuleList()
        self.gates = nn.ModuleList()

        for _ in range(num_layers):
            # 专家（低秩分解）
            experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, low_rank, bias=False),
                    nn.Linear(low_rank, input_dim, bias=True)
                ) for _ in range(num_experts)
            ])
            self.experts.append(experts)

            # 门控
            self.gates.append(nn.Linear(input_dim, num_experts))

    def forward(self, x):
        for layer_idx in range(self.num_layers):
            # 计算门控权重
            gate_weights = F.softmax(self.gates[layer_idx](x), dim=-1)

            # 计算专家输出
            expert_outputs = torch.stack(
                [expert(x) for expert in self.experts[layer_idx]],
                dim=-1
            )  # (batch, input_dim, num_experts)

            # 加权组合
            combined = torch.einsum('bde,be->bd', expert_outputs, gate_weights)

            # 残差连接
            x = x * combined + x

        return x
```

### 3.2 训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CTRDataset(Dataset):
    """CTR 数据集"""

    def __init__(self, X_cat, X_num, y):
        self.X_cat = X_cat  # 类别特征 (n_samples, n_cat_features)
        self.X_num = X_num  # 数值特征 (n_samples, n_num_features)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = {}

        # 类别特征
        for i in range(self.X_cat.shape[1]):
            features[f'cat_{i}'] = torch.tensor(self.X_cat[idx, i])

        # 数值特征
        for i in range(self.X_num.shape[1]):
            features[f'num_{i}'] = torch.tensor(self.X_num[idx, i], dtype=torch.float)

        return features, torch.tensor(self.y[idx], dtype=torch.float)


def train_dcn():
    """训练 DCN"""
    # 配置
    config = {
        'n_cat_features': 10,
        'n_num_features': 5,
        'vocab_sizes': [100] * 10,
        'embed_dim': 16,
        'n_samples': 10000,
        'batch_size': 256,
        'epochs': 10,
        'learning_rate': 0.001,
    }

    # 生成数据
    X_cat = np.random.randint(0, 100, (config['n_samples'], config['n_cat_features']))
    X_num = np.random.randn(config['n_samples'], config['n_num_features']).astype(np.float32)
    y = np.random.randint(0, 2, config['n_samples'])

    # 构建特征配置
    feature_configs = {}
    for i in range(config['n_cat_features']):
        feature_configs[f'cat_{i}'] = {
            'type': 'categorical',
            'vocab_size': config['vocab_sizes'][i],
            'embed_dim': config['embed_dim']
        }
    for i in range(config['n_num_features']):
        feature_configs[f'num_{i}'] = {'type': 'numerical'}

    # 创建模型
    model = DCN(
        feature_configs=feature_configs,
        embed_dim=config['embed_dim'],
        cross_num_layers=4,
        deep_hidden_units=[128, 64]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 数据加载器
    dataset = CTRDataset(X_cat, X_num, y)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss()

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for features, labels in dataloader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(features).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    model = train_dcn()
    print("DCN 训练完成！")
```

## 4. DCN vs 其他模型

| 模型 | 特征交叉 | 交叉阶数 | 参数效率 |
|------|----------|----------|----------|
| Wide&Deep | 手工 | 无限制 | 低 |
| DeepFM | FM（二阶） | 2 | 中 |
| DCN | 显式 | 有界（L+1） | 高 |
| DCN-V2 | 显式 | 有界 | 高 |

## 5. 调参建议

### 5.1 交叉网络

- **层数**：通常 4-8 层
- **输入维度**：不宜过大，可以先用 MLP 降维

### 5.2 深度网络

- **隐藏层**：[256, 128, 64] 或 [512, 256, 128]
- **Dropout**：0.1-0.3

### 5.3 DCN-V2 特有

- **低秩**：32-128
- **专家数**：4-8

## 6. 学习总结

### 6.1 核心创新

1. **显式特征交叉**：交叉网络显式建模
2. **有界阶数**：通过层数控制交叉阶数
3. **参数高效**：每层只有 d 个参数

### 6.2 适用场景

- 需要显式建模特征交叉
- 特征维度不是特别大
- 对可解释性有要求

## 7. 练习题

1. 比较不同交叉层数对模型效果的影响。

2. 实现 DCN-V2 的低秩 MoE 版本。

3. 比较交叉网络和 FM 的特征交叉学习方式。
