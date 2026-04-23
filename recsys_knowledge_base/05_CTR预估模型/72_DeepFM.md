# DeepFM 学习文档

## 1. 算法基础认知

### 1.1 什么是 DeepFM？

DeepFM（Deep Factorization Machine）是华为在 2017 年提出的 CTR 预估模型。它将 FM 和深度学习结合，**同时学习低阶特征交叉和高阶特征交叉**，且不需要人工特征工程。

### 1.2 为什么需要 DeepFM？

**FM 的局限：**
- 只能学习二阶特征交叉
- 高阶交叉需要堆叠多层 FM，效果有限

**深度模型的局限：**
- 纯 DNN 学习的特征交叉是隐式的、bit-wise 的
- 对低阶交叉的学习不如 FM 直接

**Wide&Deep 的局限：**
- Wide 部分需要人工特征工程
- 特征交叉需要手动设计

**DeepFM 的解决方案：**
- FM 部分：学习低阶（一阶、二阶）特征交叉
- Deep 部分：学习高阶特征交叉
- 两者共享输入 Embedding，无需人工特征工程

### 1.3 模型架构

```
                    输出层 (Sigmoid)
                         ↑
         ┌───────────────┴───────────────┐
         │                               │
    FM 部分                           Deep 部分
    (低阶交叉)                       (高阶交叉)
         │                               │
    ┌────┴────┐                    ┌─────┴─────┐
    │         │                    │           │
 一阶项    二阶交叉             Embedding    MLP
    │         │                    │           │
    └────┬────┘                    └─────┬─────┘
         │                               │
         └───────────────┬───────────────┘
                         ↑
                   Sparse Features
                   (Embedding Layer)
```

## 2. 核心原理

### 2.1 模型结构

DeepFM 的预测输出：

$$\hat{y} = \sigma(y_{FM} + y_{Deep})$$

其中：
- $y_{FM}$：FM 部分的输出（一阶 + 二阶交叉）
- $y_{Deep}$：深度部分的输出（高阶交叉）

### 2.2 FM 部分

FM 部分与标准 FM 相同：

$$y_{FM} = \langle w, x \rangle + \sum_{i=1}^{d} \sum_{j=i+1}^{d} \langle V_i, V_j \rangle x_i \cdot x_j$$

- 一阶项：$\langle w, x \rangle$
- 二阶交叉：$\sum_{i<j} \langle V_i, V_j \rangle x_i x_j$

### 2.3 Deep 部分

Deep 部分是一个前馈神经网络：

$$a^{(0)} = [e_1, e_2, ..., e_m]$$

$$a^{(l+1)} = \sigma(W^{(l)} a^{(l)} + b^{(l)})$$

$$y_{Deep} = W^{(H)} a^{(H)} + b^{(H)}$$

其中：
- $e_i$：第 i 个字段的 Embedding 向量
- $a^{(l)}$：第 l 层的激活值
- $H$：隐藏层数量

### 2.4 共享 Embedding

**关键设计**：FM 和 Deep 共享相同的 Embedding 层。

```
输入特征 ──→ Embedding 层 ──┬──→ FM 部分
                            │
                            └──→ Deep 部分
```

**好处：**
1. 减少参数量
2. FM 和 Deep 互相增强
3. 无需人工特征工程

## 3. 数学公式与推导

### 3.1 完整预测公式

$$\hat{y} = \sigma\left( \underbrace{w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j}_{FM部分} + \underbrace{f_{DNN}(E)}_{Deep部分} \right)$$

其中 $E = [e_1, e_2, ..., e_m]$ 是所有字段的 Embedding 拼接。

### 3.2 损失函数

二分类交叉熵损失：

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

### 3.3 参数量分析

假设：
- n 个特征，m 个字段
- k 维 Embedding
- d 维隐层，L 层

**FM 部分参数：**
- 一阶权重：O(n)
- 二阶隐向量：O(n × k)

**Deep 部分参数：**
- Embedding：O(n × k)（与 FM 共享）
- MLP：O(m × k × d + d² × L)

**总计**：O(n × k + m × k × d + d² × L)

## 4. 训练过程讲解

### 4.1 数据准备

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CTRDataset(Dataset):
    """CTR 数据集"""

    def __init__(self, data, label):
        """
        参数:
            data: dict, {field_name: feature_ids}
            label: array, 点击标签
        """
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        features = {field: torch.LongTensor([values[idx]])
                   for field, values in self.data.items()}
        label = torch.FloatTensor([self.label[idx]])
        return features, label
```

### 4.2 训练循环

```python
def train_deepfm(model, train_loader, val_loader, epochs, lr, device):
    """训练 DeepFM"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_auc = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for features, labels in train_loader:
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证
        model.eval()
        val_auc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, "
              f"Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict().copy()

    return best_model
```

## 5. 应用场景

### 5.1 典型应用

| 场景 | 特点 |
|------|------|
| 广告 CTR | 大量稀疏特征，需要特征交叉 |
| 推荐排序 | 用户-物品-上下文多维度 |
| 商品推荐 | 类别、品牌、价格等特征 |
| 内容推荐 | 用户兴趣与内容特征匹配 |

### 5.2 特征设计

```python
# 典型的 CTR 特征设计
features = {
    # 用户特征
    'user_id': '12345',
    'user_age': '25-30',
    'user_gender': 'M',
    'user_city': 'Beijing',
    'user_history_cats': ['tech', 'sports'],

    # 物品特征
    'item_id': '67890',
    'item_category': 'electronics',
    'item_brand': 'Apple',
    'item_price': 'high',

    # 上下文特征
    'time': 'evening',
    'device': 'mobile',
    'position': 1
}
```

## 6. 优缺点分析

### 6.1 优点

1. **自动特征交叉**：无需人工特征工程
2. **低阶+高阶**：同时学习一阶、二阶、高阶交叉
3. **端到端训练**：FM 和 Deep 联合优化
4. **共享 Embedding**：参数效率高
5. **效果优秀**：在多个数据集上取得 SOTA

### 6.2 缺点

1. **计算量大**：比纯 FM 或 LR 慢
2. **超参数多**：Embedding 维度、隐层大小、层数等
3. **高阶交叉隐式**：Deep 部分的高阶交叉不够显式

### 6.3 模型对比

| 模型 | 低阶交叉 | 高阶交叉 | 人工特征 | 效果 |
|------|----------|----------|----------|------|
| LR | ✓ | ✗ | 需要 | 一般 |
| FM | ✓(二阶) | ✗ | 不需要 | 较好 |
| Wide&Deep | ✓ | ✓ | 需要 | 好 |
| DeepFM | ✓ | ✓ | 不需要 | 好 |
| DCN | ✓ | ✓(显式) | 不需要 | 好 |

## 7. PyTorch 实现

### 7.1 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    """
    DeepFM 实现
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout=0.2):
        """
        参数:
            field_dims: list, 每个字段的特征数量
            embed_dim: int, Embedding 维度
            mlp_dims: list, MLP 各层维度
            dropout: float, Dropout 比例
        """
        super().__init__()

        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # ============ Embedding 层 ============
        # 共享 Embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])

        # 初始化
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

        # ============ FM 部分 ============
        # 一阶特征权重
        self.linear = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in field_dims
        ])

        # ============ Deep 部分 ============
        mlp_input_dim = self.num_fields * embed_dim
        mlp_layers = []

        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(mlp_input_dim, dim))
            mlp_layers.append(nn.BatchNorm1d(dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            mlp_input_dim = dim

        self.mlp = nn.Sequential(*mlp_layers)

        # 最终输出层
        self.output_layer = nn.Linear(mlp_dims[-1] + 1, 1)  # +1 for FM 2nd order

    def forward(self, x):
        """
        参数:
            x: dict, {field_name: tensor(batch, 1)}

        返回:
            predictions: (batch, 1)
        """
        batch_size = next(iter(x.values())).size(0)
        device = next(iter(x.values())).device

        # 收集每个字段的索引
        x_indices = torch.cat([x[f] for f in sorted(x.keys())], dim=1)  # (batch, num_fields)

        # ============ Embedding ============
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            emb = emb_layer(x_indices[:, i])  # (batch, embed_dim)
            embeddings.append(emb)
        embeddings = torch.stack(embeddings, dim=1)  # (batch, num_fields, embed_dim)

        # ============ FM 部分 ============
        # 一阶特征
        linear_out = torch.zeros(batch_size, 1, device=device)
        for i, linear_layer in enumerate(self.linear):
            linear_out += linear_layer(x_indices[:, i])  # (batch, 1)

        # 二阶交叉（优化计算）
        # sum(embed)^2 - sum(embed^2)
        sum_emb = torch.sum(embeddings, dim=1)  # (batch, embed_dim)
        sum_emb_sq = sum_emb ** 2

        emb_sq = embeddings ** 2
        sum_emb_sq_ = torch.sum(emb_sq, dim=1)

        fm_cross = 0.5 * torch.sum(sum_emb_sq - sum_emb_sq_, dim=1, keepdim=True)  # (batch, 1)

        # ============ Deep 部分 ============
        # 拼接所有 Embedding
        deep_input = embeddings.view(batch_size, -1)  # (batch, num_fields * embed_dim)

        # MLP
        deep_out = self.mlp(deep_input)  # (batch, mlp_dims[-1])

        # ============ 输出 ============
        # 合并 FM 和 Deep 的输出
        combined = torch.cat([fm_cross, deep_out], dim=1)  # (batch, 1 + mlp_dims[-1])

        output = self.output_layer(combined) + linear_out
        output = torch.sigmoid(output)

        return output

    def get_embeddings(self, x):
        """获取特征 Embedding"""
        x_indices = torch.cat([x[f] for f in sorted(x.keys())], dim=1)

        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            emb = emb_layer(x_indices[:, i])
            embeddings.append(emb)

        return torch.stack(embeddings, dim=1)


class DeepFMLite(nn.Module):
    """
    简化版 DeepFM，用于快速实验
    """

    def __init__(self, field_dims, embed_dim=16, mlp_dims=[128, 64], dropout=0.2):
        super().__init__()

        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # Embedding
        self.embedding = nn.ModuleDict({
            str(i): nn.Embedding(dim, embed_dim)
            for i, dim in enumerate(field_dims)
        })

        # FM Linear
        self.linear = nn.ModuleDict({
            str(i): nn.Embedding(dim, 1)
            for i, dim in enumerate(field_dims)
        })

        # MLP
        input_dim = len(field_dims) * embed_dim
        layers = []
        for hidden_dim in mlp_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # 输出
        self.fm_output = nn.Linear(1, 1)
        self.deep_output = nn.Linear(mlp_dims[-1], 1)

    def forward(self, x):
        batch_size = x.size(0)

        # Embedding
        emb_list = []
        linear_sum = torch.zeros(batch_size, 1, device=x.device)

        for i in range(self.num_fields):
            field_x = x[:, i]
            emb_list.append(self.embedding[str(i)](field_x))
            linear_sum += self.linear[str(i)](field_x)

        embeddings = torch.stack(emb_list, dim=1)  # (B, F, K)

        # FM 二阶
        sum_emb = embeddings.sum(dim=1)
        sum_emb_sq = sum_emb ** 2
        emb_sq = embeddings ** 2
        sum_emb_sq_ = emb_sq.sum(dim=1)
        fm_cross = 0.5 * (sum_emb_sq - sum_emb_sq_).sum(dim=1, keepdim=True)

        # Deep
        deep_input = embeddings.view(batch_size, -1)
        deep_out = self.mlp(deep_input)

        # 合并
        output = linear_sum + fm_cross + self.deep_output(deep_out)
        return torch.sigmoid(output)


# ============ 使用示例 ============
if __name__ == "__main__":
    # 配置
    field_dims = [1000, 500, 100, 50, 20]  # 5 个字段，每个字段的特征数量
    embed_dim = 16
    mlp_dims = [128, 64]
    batch_size = 32

    # 创建模型
    model = DeepFM(field_dims, embed_dim, mlp_dims)

    # 模拟输入
    x = {
        'field_0': torch.randint(0, 1000, (batch_size, 1)),
        'field_1': torch.randint(0, 500, (batch_size, 1)),
        'field_2': torch.randint(0, 100, (batch_size, 1)),
        'field_3': torch.randint(0, 50, (batch_size, 1)),
        'field_4': torch.randint(0, 20, (batch_size, 1)),
    }

    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")
    print(f"预测值范围: [{output.min():.4f}, {output.max():.4f}]")

    # 损失计算
    labels = torch.randint(0, 2, (batch_size, 1)).float()
    criterion = nn.BCELoss()
    loss = criterion(output, labels)
    print(f"Loss: {loss.item():.4f}")
```

### 7.2 训练脚本

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np

def train_deepfm_model(model, train_data, val_data, config):
    """
    训练 DeepFM 模型

    参数:
        model: DeepFM 模型
        train_data: 训练数据
        val_data: 验证数据
        config: 训练配置
    """
    device = config.get('device', 'cpu')
    model = model.to(device)

    # 数据加载器
    train_loader = DataLoader(
        train_data, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=config['batch_size'], shuffle=False
    )

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0)
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # 损失函数
    criterion = nn.BCELoss()

    best_auc = 0
    best_state = None

    for epoch in range(config['epochs']):
        # 训练
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            pred = model({'field_' + str(i): batch_x[:, i].unsqueeze(1)
                         for i in range(batch_x.size(1))})
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                pred = model({'field_' + str(i): batch_x[:, i].unsqueeze(1)
                             for i in range(batch_x.size(1))})
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(batch_y.numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{config['epochs']}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()

    # 恢复最佳模型
    model.load_state_dict(best_state)
    return model, best_auc


# 运行示例
if __name__ == "__main__":
    # 模拟数据
    n_samples = 10000
    n_fields = 5
    field_dims = [1000, 500, 100, 50, 20]

    # 生成数据
    X = np.column_stack([
        np.random.randint(0, dim, n_samples) for dim in field_dims
    ])
    y = np.random.randint(0, 2, n_samples)

    # 划分
    split = int(0.8 * n_samples)
    train_X, val_X = X[:split], X[split:]
    train_y, val_y = y[:split], y[split:]

    train_data = TensorDataset(torch.LongTensor(train_X), torch.LongTensor(train_y))
    val_data = TensorDataset(torch.LongTensor(val_X), torch.LongTensor(val_y))

    # 配置
    config = {
        'batch_size': 256,
        'learning_rate': 0.001,
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 创建模型
    model = DeepFM(field_dims, embed_dim=16, mlp_dims=[128, 64])

    # 训练
    model, best_auc = train_deepfm_model(model, train_data, val_data, config)
    print(f"\n最佳验证 AUC: {best_auc:.4f}")
```

## 8. 可视化与结果理解

### 8.1 特征重要性分析

```python
def analyze_feature_importance(model, field_names):
    """分析特征重要性"""
    importances = {}

    # 一阶特征重要性
    for i, (name, linear) in enumerate(zip(field_names, model.linear)):
        weight = linear.weight.abs().mean().item()
        importances[f'{name}_linear'] = weight

    # Embedding 重要性（通过方差）
    for i, (name, emb) in enumerate(zip(field_names, model.embeddings)):
        variance = emb.weight.var().item()
        importances[f'{name}_embedding'] = variance

    return importances
```

### 8.2 Embedding 可视化

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(model, field_idx, field_name, n_samples=100):
    """可视化某个字段的 Embedding"""
    emb = model.embeddings[field_idx].weight.detach().numpy()

    if emb.shape[0] > n_samples:
        indices = np.random.choice(emb.shape[0], n_samples, replace=False)
        emb = emb[indices]

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'{field_name} Embedding Visualization')
    plt.show()
```

## 9. 模型评估

```python
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
import matplotlib.pyplot as plt

def evaluate_deepfm(model, test_loader, device):
    """全面评估 DeepFM 模型"""
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model({'field_' + str(i): batch_x[:, i].unsqueeze(1)
                         for i in range(batch_x.size(1))})
            predictions.extend(pred.cpu().numpy())
            labels.extend(batch_y.numpy())

    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()

    metrics = {
        'AUC': roc_auc_score(labels, predictions),
        'LogLoss': log_loss(labels, predictions),
        'Calibration': labels.mean() / predictions.mean()
    }

    # ROC 曲线
    fpr, tpr, _ = roc_curve(labels, predictions)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {metrics["AUC"]:.4f})')

    # 预测分布
    plt.subplot(1, 2, 2)
    plt.hist(predictions[labels == 0], bins=50, alpha=0.5, label='Negative')
    plt.hist(predictions[labels == 1], bins=50, alpha=0.5, label='Positive')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Prediction Distribution')

    plt.tight_layout()
    plt.show()

    return metrics
```

## 10. 常见问题与易错点

### 10.1 常见问题

**Q1：DeepFM 和 Wide&Deep 有什么区别？**

A：
- Wide&Deep 的 Wide 部分需要人工特征工程
- DeepFM 的 FM 部分自动学习特征交叉
- DeepFM 共享 Embedding，参数更少

**Q2：Embedding 维度如何选择？**

A：通常在 8-64 之间。维度越大，表达能力越强，但参数也越多。

**Q3：如何处理新特征？**

A：需要重新训练模型，或者在特征表预留位置。

### 10.2 易错点

1. **忘记 Sigmoid**：输出层需要 Sigmoid 激活
2. **共享 Embedding 未实现**：FM 和 Deep 应该共享 Embedding
3. **批次归一化位置**：应该在 ReLU 之前
4. **学习率太大**：深度模型对学习率敏感

## 11. 学习总结

### 11.1 核心要点

1. **DeepFM = FM + Deep**：同时学习低阶和高阶特征交叉
2. **共享 Embedding**：FM 和 Deep 共享输入表示
3. **端到端训练**：无需人工特征工程
4. **效果优秀**：在多个场景取得好效果

### 11.2 模型演进

```
LR → FM → FFM → DeepFM → ...
                    ↓
              Wide&Deep
                    ↓
                 DCN
                    ↓
                 DIN/DIEN
```

## 12. 练习题

### 12.1 基础题

1. DeepFM 由哪两部分组成？各负责什么？

2. 为什么 FM 和 Deep 要共享 Embedding？

3. DeepFM 相比 Wide&Deep 有什么优势？

### 12.2 进阶题

4. 实现一个支持连续特征的 DeepFM。

5. 比较 DeepFM 和 FM 在相同数据上的效果。

### 12.3 思考题

6. DeepFM 的高阶交叉是隐式的，如何改进为显式交叉？

7. 如何将 DeepFM 扩展到多任务学习？

## 13. 学习路径建议

### 13.1 前置知识

- [ ] FM 原理
- [ ] 深度学习基础
- [ ] PyTorch/TensorFlow

### 13.2 学习顺序

1. 理解 FM → 低阶交叉
2. 理解 DNN → 高阶交叉
3. 学习 DeepFM → 组合两者
4. 调参实践 → Embedding 维度、MLP 结构
5. 学习扩展 → DCN、DIN 等

### 13.3 下一步学习

- **Wide&Deep**：Google 的经典架构
- **DCN**：显式高阶交叉
- **DIN**：注意力机制
- **AutoInt**：自动特征交互
