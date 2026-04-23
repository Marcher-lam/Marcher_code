# xDeepFM 学习文档

## 1. 算法基础认知

### 1.1 什么是 xDeepFM？

xDeepFM（eXtreme Deep Factorization Machine）是微软在 2018 年提出的 CTR 预估模型。它设计了 **CIN（Compressed Interaction Network）** 来显式学习高阶特征交叉，同时结合了显式交叉和隐式交叉。

### 1.2 动机

**DCN 的问题：**
- 交叉网络学习的是 bit-wise 交叉（特征内部维度交叉）
- 交叉阶数有界，但不够灵活

**xDeepFM 的改进：**
- CIN 学习 **vector-wise** 交叉（特征向量级别交叉）
- 可以学习任意阶数的显式交叉
- 结合 DNN 学习隐式交叉

### 1.3 Bit-wise vs Vector-wise

- **Bit-wise**：在 embedding 维度内部交叉（DCN）
- **Vector-wise**：特征向量之间的交叉（xDeepFM）

```
Bit-wise:  [e1, e2, e3] ⊙ [e1', e2', e3']
Vector-wise: [v1] × [v2] → 交叉向量
```

## 2. CIN 网络原理

### 2.1 CIN 结构

CIN（Compressed Interaction Network）的核心思想是：
1. 每一层产生新的特征映射
2. 使用外积计算特征交叉
3. 使用 1D 卷积压缩交叉结果

### 2.2 数学公式

第 k 层的输出 $X^k$：

$$X^k_{h,*} = \sum_{i=1}^{H_{k-1}} \sum_{j=1}^{m} W_{ij}^k (X_{i,*}^{k-1} \circ X_{j,*}^0)$$

其中：
- $X^0$：原始嵌入矩阵
- $H_{k-1}$：第 k-1 层的特征映射数
- $m$：特征域数量
- $\circ$：Hadamard 积（逐元素乘）
- $W^k$：第 k 层的权重矩阵

### 2.3 CIN 可视化

```
原始输入 X^0: (batch, m, D)
              ↓
         ┌────────┐
         │  外积   │  X^0 与 X^{k-1} 的所有组合
         └────────┘
              ↓
         (batch, m×H_{k-1}, D)
              ↓
         ┌────────┐
         │ 1D Conv│  压缩到 H_k 个特征映射
         └────────┘
              ↓
         X^k: (batch, H_k, D)
              ↓
         ┌────────┐
         │ Sum Pooling │  每层产生输出
         └────────┘
              ↓
         拼接所有层的输出
```

## 3. PyTorch 完整实现

### 3.1 xDeepFM 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CIN(nn.Module):
    """
    Compressed Interaction Network (CIN)

    xDeepFM 的核心组件，用于显式学习高阶特征交叉
    """

    def __init__(self, field_num, embed_dim, cin_layer_sizes=[128, 128]):
        """
        参数:
            field_num: 特征域数量
            embed_dim: 嵌入维度
            cin_layer_sizes: 每层 CIN 的特征映射数量列表
        """
        super().__init__()

        self.field_num = field_num
        self.embed_dim = embed_dim
        self.cin_layer_sizes = cin_layer_sizes

        # 每层的卷积层
        self.conv_layers = nn.ModuleList()
        prev_layer_size = field_num

        for i, layer_size in enumerate(cin_layer_sizes):
            # 1D 卷积：输入通道 = field_num * prev_layer_size
            #         输出通道 = layer_size
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=field_num * prev_layer_size,
                    out_channels=layer_size,
                    kernel_size=1  # 1x1 卷积
                )
            )
            prev_layer_size = layer_size

        # 输出维度：每层产生 embed_dim 个特征
        self.output_dim = sum(cin_layer_sizes)

    def forward(self, x):
        """
        前向传播

        参数:
            x: (batch, field_num, embed_dim) 嵌入矩阵

        返回:
            output: (batch, sum(cin_layer_sizes)) CIN 输出
        """
        batch_size = x.size(0)

        # 保存原始输入
        x0 = x  # (batch, field_num, embed_dim)

        # 保存每层的池化输出
        pooling_outputs = []

        # 当前层输入
        xl = x0

        for i, conv in enumerate(self.conv_layers):
            # 计算 xl 与 x0 的外积
            # xl: (batch, H_{l-1}, D)
            # x0: (batch, m, D)

            # 外积：expand 和 multiply
            # (batch, H_{l-1}, 1, D) * (batch, 1, m, D) = (batch, H_{l-1}, m, D)
            xl_expand = xl.unsqueeze(2)  # (batch, H_{l-1}, 1, D)
            x0_expand = x0.unsqueeze(1)  # (batch, 1, m, D)

            # 逐元素乘法（Hadamard 积）
            cross = xl_expand * x0_expand  # (batch, H_{l-1}, m, D)

            # 重塑为卷积输入格式
            # (batch, H_{l-1} * m, D)
            cross = cross.view(batch_size, -1, self.embed_dim)

            # 1D 卷积（在 embed_dim 维度上，kernel_size=1）
            # (batch, H_{l-1} * m, D) -> (batch, H_l, D)
            xl = conv(cross)  # (batch, layer_size, D)

            # 激活
            xl = F.relu(xl)

            # Sum pooling（在 embed_dim 维度上求和）
            # (batch, layer_size)
            pooling_output = xl.sum(dim=-1)
            pooling_outputs.append(pooling_output)

        # 拼接所有层的输出
        output = torch.cat(pooling_outputs, dim=-1)  # (batch, sum(cin_layer_sizes))

        return output


class xDeepFM(nn.Module):
    """
    xDeepFM: eXtreme Deep Factorization Machine

    论文: xDeepFM: Combining Explicit and Implicit Feature Interactions
          for Recommender Systems (KDD 2018)

    组成部分:
    1. Embedding Layer: 类别特征嵌入
    2. CIN: 显式高阶特征交叉
    3. DNN: 隐式高阶特征交叉
    4. Linear: 一阶特征
    """

    def __init__(self, feature_configs, embed_dim=16,
                 cin_layer_sizes=[128, 128],
                 dnn_hidden_units=[256, 128],
                 dnn_dropout=0.2,
                 use_cin=True, use_dnn=True, use_linear=True):
        """
        参数:
            feature_configs: dict, 特征配置
            embed_dim: 嵌入维度
            cin_layer_sizes: CIN 每层大小
            dnn_hidden_units: DNN 隐藏层
            dnn_dropout: DNN dropout
            use_cin: 是否使用 CIN
            use_dnn: 是否使用 DNN
            use_linear: 是否使用线性部分
        """
        super().__init__()

        self.feature_configs = feature_configs
        self.embed_dim = embed_dim
        self.use_cin = use_cin
        self.use_dnn = use_dnn
        self.use_linear = use_linear

        # ========== Embedding 层 ==========
        self.embeddings = nn.ModuleDict()
        self.field_num = 0

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(
                    config['vocab_size'],
                    config.get('embed_dim', embed_dim)
                )
                self.field_num += 1

        # 计算输入维度
        self._compute_input_dim(feature_configs, embed_dim)

        # ========== Linear 部分 ==========
        if use_linear:
            self.linear = nn.Linear(self.input_dim, 1)

        # ========== CIN 部分 ==========
        if use_cin:
            self.cin = CIN(
                field_num=self.field_num,
                embed_dim=embed_dim,
                cin_layer_sizes=cin_layer_sizes
            )
            self.cin_output = nn.Linear(self.cin.output_dim, 1)

        # ========== DNN 部分 ==========
        if use_dnn:
            dnn_layers = []
            input_dim = self.input_dim

            for hidden in dnn_hidden_units:
                dnn_layers.append(nn.Linear(input_dim, hidden))
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Dropout(dnn_dropout))
                input_dim = hidden

            self.dnn = nn.Sequential(*dnn_layers)
            self.dnn_output = nn.Linear(dnn_hidden_units[-1], 1)

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
        # ========== Embedding ==========
        embeddings = []
        embedding_matrix = []  # 用于 CIN

        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical':
                emb = self.embeddings[name](features[name])
                if len(emb.shape) == 3:
                    emb = emb.squeeze(1)
                embeddings.append(emb)
                embedding_matrix.append(emb)

        # 数值特征
        numerical_features = []
        for name, config in self.feature_configs.items():
            if config['type'] == 'numerical':
                val = features[name]
                if len(val.shape) == 1:
                    val = val.unsqueeze(-1)
                numerical_features.append(val)

        # 拼接所有特征（用于 linear 和 DNN）
        if numerical_features:
            all_features = torch.cat(embeddings + numerical_features, dim=-1)
        else:
            all_features = torch.cat(embeddings, dim=-1)

        # CIN 输入：嵌入矩阵
        embedding_matrix = torch.stack(embedding_matrix, dim=1)  # (batch, field_num, embed_dim)

        # ========== 计算各部分输出 ==========
        logits = []

        # Linear 部分
        if self.use_linear:
            linear_logit = self.linear(all_features)
            logits.append(linear_logit)

        # CIN 部分
        if self.use_cin:
            cin_output = self.cin(embedding_matrix)
            cin_logit = self.cin_output(cin_output)
            logits.append(cin_logit)

        # DNN 部分
        if self.use_dnn:
            dnn_output = self.dnn(all_features)
            dnn_logit = self.dnn_output(dnn_output)
            logits.append(dnn_logit)

        # 求和
        logit = sum(logits)

        # Sigmoid
        output = torch.sigmoid(logit)

        return output


class xDeepFMVariant(nn.Module):
    """
    xDeepFM 变体：结合 FM 的二阶交叉
    """

    def __init__(self, feature_configs, embed_dim=16,
                 cin_layer_sizes=[128, 128],
                 dnn_hidden_units=[256, 128],
                 use_fm=True, use_cin=True, use_dnn=True):
        super().__init__()

        self.feature_configs = feature_configs
        self.embed_dim = embed_dim
        self.use_fm = use_fm
        self.use_cin = use_cin
        self.use_dnn = use_dnn

        # Embedding 层
        self.embeddings = nn.ModuleDict()
        self.field_num = 0

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(
                    config['vocab_size'],
                    embed_dim
                )
                self.field_num += 1

        # Linear 部分
        self._compute_input_dim(feature_configs, embed_dim)
        self.linear = nn.Linear(self.input_dim, 1)

        # FM 部分
        if use_fm:
            self.fm = FMModule()

        # CIN 部分
        if use_cin:
            self.cin = CIN(
                field_num=self.field_num,
                embed_dim=embed_dim,
                cin_layer_sizes=cin_layer_sizes
            )
            self.cin_output = nn.Linear(self.cin.output_dim, 1)

        # DNN 部分
        if use_dnn:
            dnn_layers = []
            input_dim = self.input_dim
            for hidden in dnn_hidden_units:
                dnn_layers.append(nn.Linear(input_dim, hidden))
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Dropout(0.2))
                input_dim = hidden
            self.dnn = nn.Sequential(*dnn_layers)
            self.dnn_output = nn.Linear(dnn_hidden_units[-1], 1)

    def _compute_input_dim(self, feature_configs, embed_dim):
        dim = 0
        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                dim += embed_dim
            elif config['type'] == 'numerical':
                dim += 1
        self.input_dim = dim

    def forward(self, features):
        # Embedding
        embeddings = []
        embedding_matrix = []

        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical':
                emb = self.embeddings[name](features[name])
                if len(emb.shape) == 3:
                    emb = emb.squeeze(1)
                embeddings.append(emb)
                embedding_matrix.append(emb)

        numerical_features = []
        for name, config in self.feature_configs.items():
            if config['type'] == 'numerical':
                val = features[name]
                if len(val.shape) == 1:
                    val = val.unsqueeze(-1)
                numerical_features.append(val)

        if numerical_features:
            all_features = torch.cat(embeddings + numerical_features, dim=-1)
        else:
            all_features = torch.cat(embeddings, dim=-1)

        embedding_matrix = torch.stack(embedding_matrix, dim=1)

        # 各部分输出
        logits = []

        # Linear
        logits.append(self.linear(all_features))

        # FM
        if self.use_fm:
            fm_output = self.fm(embedding_matrix)
            logits.append(fm_output)

        # CIN
        if self.use_cin:
            cin_output = self.cin(embedding_matrix)
            logits.append(self.cin_output(cin_output))

        # DNN
        if self.use_dnn:
            dnn_output = self.dnn(all_features)
            logits.append(self.dnn_output(dnn_output))

        logit = sum(logits)
        output = torch.sigmoid(logit)

        return output


class FMModule(nn.Module):
    """FM 二阶交叉模块"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        参数:
            x: (batch, field_num, embed_dim)

        返回:
            output: (batch, 1)
        """
        # FM 公式: 0.5 * (sum(x)^2 - sum(x^2))
        square_of_sum = torch.pow(x.sum(dim=1), 2)  # (batch, embed_dim)
        sum_of_square = torch.pow(x, 2).sum(dim=1)   # (batch, embed_dim)

        fm_output = 0.5 * (square_of_sum - sum_of_square)  # (batch, embed_dim)
        fm_output = fm_output.sum(dim=-1, keepdim=True)    # (batch, 1)

        return fm_output


# 使用示例
if __name__ == "__main__":
    # 配置
    feature_configs = {
        'cat1': {'type': 'categorical', 'vocab_size': 100},
        'cat2': {'type': 'categorical', 'vocab_size': 200},
        'cat3': {'type': 'categorical', 'vocab_size': 150},
        'num1': {'type': 'numerical'},
        'num2': {'type': 'numerical'},
    }

    # 创建模型
    model = xDeepFM(
        feature_configs=feature_configs,
        embed_dim=16,
        cin_layer_sizes=[64, 64],
        dnn_hidden_units=[128, 64],
        use_cin=True,
        use_dnn=True,
        use_linear=True
    )

    # 模拟输入
    batch_size = 32
    features = {
        'cat1': torch.randint(0, 100, (batch_size,)),
        'cat2': torch.randint(0, 200, (batch_size,)),
        'cat3': torch.randint(0, 150, (batch_size,)),
        'num1': torch.randn(batch_size),
        'num2': torch.randn(batch_size),
    }

    # 前向传播
    output = model(features)
    print(f"输出形状: {output.shape}")
    print(f"预测值范围: [{output.min():.4f}, {output.max():.4f}]")
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
        self.X_cat = X_cat
        self.X_num = X_num
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = {}
        for i in range(self.X_cat.shape[1]):
            features[f'cat_{i}'] = torch.tensor(self.X_cat[idx, i])
        for i in range(self.X_num.shape[1]):
            features[f'num_{i}'] = torch.tensor(self.X_num[idx, i], dtype=torch.float)
        return features, torch.tensor(self.y[idx], dtype=torch.float)


def train_xdeepfm():
    """训练 xDeepFM"""
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
        'cin_layer_sizes': [64, 64],
        'dnn_hidden_units': [128, 64]
    }

    # 生成数据
    X_cat = np.random.randint(0, 100, (config['n_samples'], config['n_cat_features']))
    X_num = np.random.randn(config['n_samples'], config['n_num_features']).astype(np.float32)
    y = np.random.randint(0, 2, config['n_samples'])

    # 特征配置
    feature_configs = {}
    for i in range(config['n_cat_features']):
        feature_configs[f'cat_{i}'] = {
            'type': 'categorical',
            'vocab_size': config['vocab_sizes'][i]
        }
    for i in range(config['n_num_features']):
        feature_configs[f'num_{i}'] = {'type': 'numerical'}

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xDeepFM(
        feature_configs=feature_configs,
        embed_dim=config['embed_dim'],
        cin_layer_sizes=config['cin_layer_sizes'],
        dnn_hidden_units=config['dnn_hidden_units']
    ).to(device)

    # 数据
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

        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(dataloader):.4f}")

    return model


if __name__ == "__main__":
    model = train_xdeepfm()
    print("xDeepFM 训练完成！")
```

## 4. 模型对比

### 4.1 与其他模型对比

| 模型 | 显式交叉 | 交叉类型 | 交叉阶数 |
|------|----------|----------|----------|
| FM | ✓ | Vector-wise | 2 |
| DeepFM | ✓ | Vector-wise | 2 |
| DCN | ✓ | Bit-wise | 有界 |
| xDeepFM | ✓ | Vector-wise | 可控 |

### 4.2 特点

**xDeepFM 的优势：**
1. **Vector-wise 交叉**：比 bit-wise 更有语义
2. **可控阶数**：通过层数控制
3. **结合隐式交叉**：DNN 学习隐式高阶

**xDeepFM 的劣势：**
1. **计算开销**：外积计算量大
2. **参数量**：比 DCN 大
3. **内存占用**：中间结果需要存储

## 5. 调参建议

### 5.1 CIN 参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| cin_layer_sizes | [100, 100] | 不宜太大 |
| 层数 | 2-3 | 太深收益递减 |

### 5.2 DNN 参数

| 参数 | 推荐值 |
|------|--------|
| hidden_units | [256, 128] |
| dropout | 0.2 |

## 6. 学习总结

### 6.1 核心要点

1. **CIN 是核心**：显式学习 vector-wise 交叉
2. **外积 + 卷积**：高效计算特征交叉
3. **结合 DNN**：同时学习隐式交叉

### 6.2 适用场景

- 特征交叉丰富
- 对高阶交叉有需求
- 计算资源充足

## 7. 练习题

1. 实现 CIN 的并行优化版本。

2. 比较 xDeepFM 和 DCN 在相同数据上的效果。

3. 分析 CIN 的计算复杂度。
