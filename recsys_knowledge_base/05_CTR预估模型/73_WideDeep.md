# Wide & Deep 学习文档

## 1. 算法基础认知

### 1.1 什么是 Wide & Deep？

Wide & Deep Learning 是 Google 在 2016 年提出的推荐模型架构，它结合了**线性模型（Wide）的记忆能力**和**深度神经网络（Deep）的泛化能力**。

### 1.2 核心思想

**记忆（Memorization）：**
- 学习历史数据中的频繁模式
- 通过特征交叉实现
- 类似查表，对已知模式效果好

**泛化（Generalization）：**
- 学习特征之间的隐含关系
- 通过 Embedding 和 MLP 实现
- 对未见过的特征组合有预测能力

**Wide & Deep = 记忆 + 泛化**

```
        Wide 部分（记忆）           Deep 部分（泛化）
              ↓                         ↓
        手工特征交叉              Embedding + MLP
              ↓                         ↓
              └───────────┬─────────────┘
                          ↓
                    联合训练
                          ↓
                       输出
```

### 1.3 为什么需要 Wide & Deep？

**Wide（线性模型）的问题：**
- 需要人工特征工程
- 无法学习未见过的特征组合
- 泛化能力弱

**Deep（深度模型）的问题：**
- 可能过度泛化
- 对低频特征组合效果差
- 需要大量数据

**Wide & Deep 的优势：**
- 结合两者优点
- 既能记住历史模式，又能泛化到新组合
- 互补增强

## 2. 模型架构

### 2.1 Wide 部分

Wide 部分是一个广义线性模型：

$$y_{wide} = w^T x + b$$

其中：
- $x$ 是输入特征，包括原始特征和交叉特征
- $w$ 是权重向量
- $b$ 是偏置

**特征交叉（Cross-product transformation）：**

$$\phi_k(x) = \prod_{i=1}^{d} x_i^{c_{ki}}$$

最常用的是 AND 交叉：

$$\phi(x_i, x_j) = x_i \cdot x_j$$

```python
def cross_features(feature_a, feature_b):
    """
    特征交叉

    例如：user_gender=F AND item_category=beauty
    """
    return f"{feature_a} AND {feature_b}"
```

### 2.2 Deep 部分

Deep 部分是一个前馈神经网络：

$$a^{(l+1)} = f(W^{(l)} a^{(l)} + b^{(l)})$$

流程：
1. 类别特征通过 Embedding 层转换为稠密向量
2. 所有特征拼接成输入向量
3. 通过多层 MLP
4. 输出高阶特征表示

### 2.3 联合训练

最终预测：

$$P(Y=1|x) = \sigma(y_{wide} + y_{deep})$$

$$= \sigma(w_{wide}^T x + b_{wide} + w_{deep}^T a^{(L)} + b_{deep})$$

联合训练 vs 集成学习：
- **联合训练**：Wide 和 Deep 同时优化，共享梯度
- **集成学习**：独立训练，然后组合预测

## 3. 数学公式与推导

### 3.1 损失函数

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

其中：
$$\hat{y}_i = \sigma(y_{wide,i} + y_{deep,i})$$

### 3.2 反向传播

联合训练时，梯度同时传给 Wide 和 Deep 部分：

$$\frac{\partial L}{\partial w_{wide}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial y_{wide}} \cdot \frac{\partial y_{wide}}{\partial w_{wide}}$$

$$\frac{\partial L}{\partial w_{deep}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial y_{deep}} \cdot \frac{\partial y_{deep}}{\partial w_{deep}}$$

## 4. 模型实现

### 4.1 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WideAndDeep(nn.Module):
    """
    Wide & Deep 模型实现
    """

    def __init__(self, wide_feature_dims, deep_feature_dims,
                 embed_dim=16, hidden_units=[256, 128, 64], dropout=0.2):
        """
        参数:
            wide_feature_dims: dict, Wide 部分特征维度
            deep_feature_dims: dict, Deep 部分特征维度
            embed_dim: int, Embedding 维度
            hidden_units: list, MLP 隐藏层维度
            dropout: float, Dropout 比例
        """
        super().__init__()

        # ========== Wide 部分 ==========
        # 线性层处理交叉特征
        self.wide_feature_dims = wide_feature_dims
        wide_input_dim = sum(wide_feature_dims.values())
        self.wide_linear = nn.Linear(wide_input_dim, 1)

        # ========== Deep 部分 ==========
        # Embedding 层
        self.embeddings = nn.ModuleDict()
        for name, dim in deep_feature_dims.items():
            self.embeddings[name] = nn.Embedding(dim, embed_dim)

        # MLP
        deep_input_dim = len(deep_feature_dims) * embed_dim
        mlp_layers = []
        input_dim = deep_input_dim

        for hidden_dim in hidden_units:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

        # Deep 输出层
        self.deep_output = nn.Linear(hidden_units[-1], 1)

    def forward(self, wide_features, deep_features):
        """
        参数:
            wide_features: dict, Wide 部分特征 {name: tensor(batch,)}
            deep_features: dict, Deep 部分特征 {name: tensor(batch,)}

        返回:
            predictions: tensor(batch, 1)
        """
        # ========== Wide 部分 ==========
        wide_inputs = []
        for name, values in wide_features.items():
            # One-hot 编码
            one_hot = F.one_hot(values, self.wide_feature_dims[name]).float()
            wide_inputs.append(one_hot)

        wide_concat = torch.cat(wide_inputs, dim=1)
        wide_output = self.wide_linear(wide_concat)

        # ========== Deep 部分 ==========
        deep_embeddings = []
        for name, values in deep_features.items():
            emb = self.embeddings[name](values)  # (batch, embed_dim)
            deep_embeddings.append(emb)

        deep_concat = torch.cat(deep_embeddings, dim=1)  # (batch, num_features * embed_dim)
        deep_hidden = self.mlp(deep_concat)
        deep_output = self.deep_output(deep_hidden)

        # ========== 合并输出 ==========
        output = wide_output + deep_output
        output = torch.sigmoid(output)

        return output


class WideAndDeepSimple(nn.Module):
    """
    简化版 Wide & Deep
    """

    def __init__(self, num_features_wide, num_features_deep,
                 embed_dim=16, hidden_units=[128, 64]):
        super().__init__()

        # Wide: 线性层
        self.wide = nn.Linear(num_features_wide, 1)

        # Deep: Embedding + MLP
        self.embedding = nn.Embedding(num_features_deep, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], 1)
        )

    def forward(self, wide_x, deep_x):
        wide_out = self.wide(wide_x)
        deep_emb = self.embedding(deep_x).mean(dim=1)
        deep_out = self.mlp(deep_emb)
        return torch.sigmoid(wide_out + deep_out)
```

### 4.2 TensorFlow/Keras 实现

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_wide_and_deep(wide_vocab_sizes, deep_vocab_sizes,
                        embed_dim=16, hidden_units=[256, 128, 64]):
    """
    构建 Wide & Deep 模型（Keras 版本）
    """
    # ========== Wide 输入 ==========
    wide_inputs = []
    wide_outputs = []

    for name, vocab_size in wide_vocab_sizes.items():
        inp = layers.Input(shape=(1,), name=f'wide_{name}')
        wide_inputs.append(inp)

        # One-hot
        one_hot = layers.Embedding(vocab_size, vocab_size, embeddings_initializer='identity',
                                   trainable=False)(inp)
        one_hot = layers.Flatten()(one_hot)
        wide_outputs.append(one_hot)

    wide_concat = layers.Concatenate()(wide_outputs)
    wide_output = layers.Dense(1, use_bias=True)(wide_concat)

    # ========== Deep 输入 ==========
    deep_inputs = []
    deep_embeddings = []

    for name, vocab_size in deep_vocab_sizes.items():
        inp = layers.Input(shape=(1,), name=f'deep_{name}')
        deep_inputs.append(inp)

        emb = layers.Embedding(vocab_size, embed_dim)(inp)
        emb = layers.Flatten()(emb)
        deep_embeddings.append(emb)

    deep_concat = layers.Concatenate()(deep_embeddings)

    # MLP
    x = deep_concat
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

    deep_output = layers.Dense(1)(x)

    # ========== 合并 ==========
    output = layers.Add()([wide_output, deep_output])
    output = layers.Activation('sigmoid')(output)

    model = Model(
        inputs=wide_inputs + deep_inputs,
        outputs=output
    )

    return model


# 使用示例
if __name__ == "__main__":
    # 配置
    wide_vocab_sizes = {
        'user_item_cross': 10000,  # 用户-物品交叉特征
        'user_category_cross': 1000
    }

    deep_vocab_sizes = {
        'user_id': 1000,
        'item_id': 500,
        'category': 50
    }

    # 构建模型
    model = build_wide_and_deep(wide_vocab_sizes, deep_vocab_sizes)
    model.summary()

    # 编译
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC']
    )
```

## 5. 特征工程

### 5.1 Wide 部分特征

Wide 部分需要手工设计交叉特征：

```python
class WideFeatureEngineer:
    """Wide 部分特征工程"""

    def __init__(self):
        self.cross_features = []

    def add_cross_feature(self, feature_a, feature_b):
        """添加交叉特征"""
        self.cross_features.append((feature_a, feature_b))

    def transform(self, data):
        """转换数据"""
        features = {}

        # 原始特征
        for col in data.columns:
            features[col] = data[col]

        # 交叉特征
        for fa, fb in self.cross_features:
            cross_name = f"{fa}_{fb}"
            features[cross_name] = data[fa].astype(str) + "_" + data[fb].astype(str)

        return features

    def get_recommendation_crosses(self):
        """推荐的交叉特征组合"""
        return [
            ('user_id', 'item_id'),          # 用户-物品
            ('user_id', 'item_category'),    # 用户-类目
            ('user_gender', 'item_category'), # 性别-类目
            ('user_age_bucket', 'item_price_bucket'), # 年龄-价格
            ('time_period', 'item_category'), # 时间-类目
        ]
```

### 5.2 Deep 部分特征

Deep 部分可以直接使用原始特征：

```python
class DeepFeatureEngineer:
    """Deep 部分特征工程"""

    def __init__(self):
        self.categorical_features = []
        self.numerical_features = []

    def add_categorical(self, name):
        self.categorical_features.append(name)

    def add_numerical(self, name):
        self.numerical_features.append(name)

    def transform(self, data):
        """转换数据"""
        features = {}

        for col in self.categorical_features:
            features[col] = data[col]

        for col in self.numerical_features:
            # 归一化
            features[col] = (data[col] - data[col].mean()) / data[col].std()

        return features
```

## 6. 训练与调优

### 6.1 训练脚本

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np

def train_wide_deep(model, train_loader, val_loader, config):
    """
    训练 Wide & Deep 模型
    """
    device = config.get('device', 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCELoss()

    best_auc = 0
    best_state = None

    for epoch in range(config['epochs']):
        # 训练
        model.train()
        train_loss = 0

        for batch in train_loader:
            wide_x, deep_x, y = batch
            wide_x = {k: v.to(device) for k, v in wide_x.items()}
            deep_x = {k: v.to(device) for k, v in deep_x.items()}
            y = y.float().to(device)

            optimizer.zero_grad()
            pred = model(wide_x, deep_x)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                wide_x, deep_x, y = batch
                wide_x = {k: v.to(device) for k, v in wide_x.items()}
                deep_x = {k: v.to(device) for k, v in deep_x.items()}

                pred = model(wide_x, deep_x)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y.numpy())

        val_auc = roc_auc_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}, "
              f"Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    return model, best_auc
```

### 6.2 超参数调优

```python
def hyperparameter_search(train_data, val_data, param_grid):
    """
    超参数搜索
    """
    best_score = 0
    best_params = None

    for embed_dim in param_grid['embed_dim']:
        for hidden_units in param_grid['hidden_units']:
            for lr in param_grid['lr']:
                # 构建模型
                model = WideAndDeep(
                    wide_feature_dims=...,
                    deep_feature_dims=...,
                    embed_dim=embed_dim,
                    hidden_units=hidden_units
                )

                # 训练
                config = {'lr': lr, 'epochs': 10}
                _, score = train_wide_deep(model, train_data, val_data, config)

                if score > best_score:
                    best_score = score
                    best_params = {
                        'embed_dim': embed_dim,
                        'hidden_units': hidden_units,
                        'lr': lr
                    }

    return best_params, best_score
```

## 7. 应用场景

### 7.1 Google Play 应用推荐

原始论文的应用场景：
- **目标**：预测用户是否会安装应用
- **Wide 特征**：用户-应用交叉、用户-类目交叉
- **Deep 特征**：用户 ID、应用 ID、类目、语言等

### 7.2 电商推荐

```python
# 电商推荐的特征设计
wide_features = {
    'user_item_cross': '用户ID_物品ID',
    'user_brand_cross': '用户ID_品牌',
    'user_cat_cross': '用户ID_类目',
}

deep_features = {
    'user_id': '用户ID',
    'item_id': '物品ID',
    'category': '类目',
    'brand': '品牌',
    'price_bucket': '价格区间',
    'user_age': '用户年龄',
    'user_history_avg_price': '用户历史平均消费',
}
```

## 8. 与 DeepFM 对比

| 方面 | Wide & Deep | DeepFM |
|------|-------------|--------|
| Wide 部分 | 手工特征交叉 | FM（自动交叉） |
| 特征工程 | 需要 | 不需要 |
| Embedding | 独立 | 共享 |
| 复杂度 | 较低 | 较高 |
| 灵活性 | 高（可定制交叉） | 中等 |

## 9. 常见问题与易错点

### 9.1 常见问题

**Q1：Wide 部分应该设计哪些交叉特征？**

A：
- 业务知识指导
- 分析历史数据中的强关联
- 常见：用户-物品、用户-类目、时间-类目

**Q2：Wide 和 Deep 的特征可以相同吗？**

A：可以。相同特征可以同时出现在两部分，互补学习。

**Q3：如何平衡 Wide 和 Deep 的权重？**

A：联合训练自动学习权重。如果需要调整，可以加权重系数。

### 9.2 易错点

1. **Wide 部分特征过多**：导致稀疏和过拟合
2. **交叉特征设计不当**：低频交叉噪声大
3. **两部分未联合训练**：独立训练效果差
4. **Embedding 维度不当**：太大过拟合，太小欠拟合

## 10. 学习总结

### 10.1 核心要点

1. **Wide = 记忆**：线性模型 + 手工交叉
2. **Deep = 泛化**：Embedding + MLP
3. **联合训练**：端到端优化
4. **互补增强**：结合两者优点

### 10.2 知识图谱

```
Wide & Deep
├── Wide 部分
│   ├── 线性模型
│   ├── 特征交叉
│   └── 记忆能力
├── Deep 部分
│   ├── Embedding
│   ├── MLP
│   └── 泛化能力
└── 联合训练
    ├── 损失函数
    ├── 反向传播
    └── 互补增强
```

## 11. 练习题

### 11.1 基础题

1. Wide 部分和 Deep 部分分别负责什么？

2. 为什么需要联合训练？

3. Wide & Deep 相比纯深度模型有什么优势？

### 11.2 进阶题

4. 实现一个完整的 Wide & Deep 训练流程。

5. 设计一组适合电商场景的 Wide 交叉特征。

### 11.3 思考题

6. 如何自动发现有效的交叉特征？

7. Wide & Deep 在什么情况下效果提升不明显？

## 12. 学习路径建议

### 12.1 学习顺序

1. 理解线性模型 → Wide 部分
2. 理解深度模型 → Deep 部分
3. 学习联合训练 → 组合
4. 特征工程实践 → 设计交叉特征
5. 调参与优化 → 提升效果

### 12.2 推荐资源

- **论文**：Wide & Deep Learning for Recommender Systems (Google, 2016)
- **代码**：TensorFlow 官方实现
- **实践**：Kaggle CTR 竞赛
