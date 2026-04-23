# NFM (Neural Factorization Machines) 学习文档

## 1. 算法基础认知

### 1.1 什么是 NFM？

NFM 将 FM 与神经网络结合，用神经网络来学习高阶特征交互，同时保留 FM 的二阶交互能力。

### 1.2 核心创新

```
传统FM: y = w·x + Σ⟨v_i, v_j⟩x_i·x_j
NFM:   y = w·x + f(V·x)

其中 f 是一个神经网络，输入是二阶交互池化后的向量
```

### 1.3 模型架构

```
Input → Embedding → Bi-Interaction Pooling → Hidden Layers → Output
```

## 2. 核心原理

### 2.1 Bi-Interaction Pooling

这是 NFM 的核心创新：

$$f_{BI}(V, x) = \sum_{i=1}^{n} \sum_{j=i+1}^{n} x_i x_j (v_i \odot v_j)$$

其中 $\odot$ 是逐元素乘积。

性质：
- 输出是一个 k 维向量（而非标量）
- 可高效计算：$\frac{1}{2}[(\sum_i x_i v_i)^2 - \sum_i (x_i v_i)^2]$

### 2.2 完整公式

$$\hat{y}_{NFM}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + f_{BI}(V, x)$$

其中:
$$f_{BI}(V, x) = \text{MLP}(\sum_{i=1}^{n} \sum_{j=i+1}^{n} x_i x_j (v_i \odot v_j))$$

## 3. 完整实现

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional


class BiInteractionPooling(nn.Module):
    """
    Bi-Interaction Pooling 层
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            embeddings: (batch_size, num_fields, embed_dim)
            mask: (batch_size, num_fields) 可选的掩码

        返回:
            (batch_size, embed_dim)
        """
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1)

        # 方法1: 直接计算 (O(n^2))
        # sum_of_square = (embeddings.sum(dim=1)) ** 2
        # square_of_sum = (embeddings ** 2).sum(dim=1)
        # return 0.5 * (sum_of_square - square_of_sum)

        # 方法2: 高效计算 O(n)
        sum_embed = embeddings.sum(dim=1)  # (batch, embed_dim)
        square_embed = (embeddings ** 2).sum(dim=1)  # (batch, embed_dim)

        bi_pooling = 0.5 * (sum_embed ** 2 - square_embed)

        return bi_pooling


class NFM(nn.Module):
    """
    Neural Factorization Machine
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.2,
                 use_batch_norm: bool = True):
        """
        参数:
            field_dims: 各域的特征数量
            embed_dim: 嵌入维度
            hidden_dims: 隐藏层维度列表
            dropout: Dropout 比率
            use_batch_norm: 是否使用 BatchNorm
        """
        super().__init__()

        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # 线性部分
        self.linear = nn.Embedding(sum(field_dims), 1)
        self.field_offsets = torch.tensor([0] + np.cumsum(field_dims)[:-1].tolist())
        self.bias = nn.Parameter(torch.zeros(1))

        # 嵌入层
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)

        # Bi-Interaction Pooling
        self.bi_pooling = BiInteractionPooling()

        # 神经网络部分
        layers = []
        input_dim = embed_dim

        for hidden_dim in hidden_dims:
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # 输出层
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            X: (batch_size, num_fields) 特征索引

        返回:
            (batch_size, 1) 预测值
        """
        batch_size = X.size(0)
        device = X.device

        # 加上域偏移
        X_offset = X + self.field_offsets.to(device)

        # 线性部分
        linear_part = self.bias + self.linear(X_offset).sum(dim=1)

        # 嵌入
        embeddings = self.embedding(X_offset)  # (batch, num_fields, embed_dim)

        # Bi-Interaction Pooling
        bi_vector = self.bi_pooling(embeddings)  # (batch, embed_dim)

        # MLP
        mlp_output = self.mlp(bi_vector)  # (batch, hidden_dim[-1])

        # 输出
        dnn_part = self.output(mlp_output).squeeze(-1)

        # 组合
        output = linear_part.squeeze(-1) + dnn_part

        return output.unsqueeze(-1)


class NFMClassifier(nn.Module):
    """
    NFM 分类器
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        self.nfm = NFM(field_dims, embed_dim, hidden_dims, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.nfm(X))


class NFMTrainer:
    """
    NFM 训练器
    """

    def __init__(self, model: NFMClassifier,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(X)
        loss = self.criterion(pred.squeeze(), y.float())

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """预测"""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
        return pred.squeeze().cpu().numpy()


def demo_nfm():
    """NFM 示例"""
    # 模拟数据配置
    field_dims = [1000, 500, 100, 50]  # 4个域
    batch_size = 64
    num_samples = 1000

    # 创建模型
    model = NFMClassifier(
        field_dims=field_dims,
        embed_dim=16,
        hidden_dims=[128, 64],
        dropout=0.2
    )

    # 模拟数据
    X = torch.zeros(num_samples, len(field_dims), dtype=torch.long)
    for i, dim in enumerate(field_dims):
        X[:, i] = torch.randint(0, dim, (num_samples,))

    y = torch.randint(0, 2, (num_samples,)).float()

    # 创建数据集
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练
    trainer = NFMTrainer(model, learning_rate=0.001)

    for epoch in range(5):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            loss = trainer.train_step(batch_X, batch_y)
            total_loss += loss

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    # 预测
    test_X = torch.randint(0, 100, (10, len(field_dims)))
    for i, dim in enumerate(field_dims):
        test_X[:, i] = torch.randint(0, dim, (10,))

    predictions = trainer.predict(test_X)
    print(f"\n预测值: {predictions}")


if __name__ == "__main__":
    demo_nfm()
```

## 4. NFM 变体

### 4.1 带注意力的 NFM

```python
class AttentionNFM(nn.Module):
    """
    带注意力的 NFM
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 hidden_dims: List[int] = [128, 64],
                 num_attention_heads: int = 4):
        super().__init__()

        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # 嵌入层
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.field_offsets = torch.tensor([0] + np.cumsum(field_dims)[:-1].tolist())

        # 自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )

        # Bi-Interaction
        self.bi_pooling = BiInteractionPooling()

        # MLP
        layers = []
        input_dim = embed_dim * 2  # 注意力输出 + Bi-Interaction

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], 1)

        # 线性部分
        self.linear = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size = X.size(0)
        device = X.device

        X_offset = X + self.field_offsets.to(device)

        # 线性
        linear_part = self.bias + self.linear(X_offset).sum(dim=1)

        # 嵌入
        embeddings = self.embedding(X_offset)  # (batch, fields, embed_dim)

        # 自注意力
        attn_output, _ = self.attention(embeddings, embeddings, embeddings)
        attn_pooled = attn_output.mean(dim=1)  # (batch, embed_dim)

        # Bi-Interaction
        bi_vector = self.bi_pooling(embeddings)  # (batch, embed_dim)

        # 拼接
        combined = torch.cat([attn_pooled, bi_vector], dim=1)

        # MLP
        mlp_output = self.mlp(combined)
        dnn_part = self.output(mlp_output).squeeze(-1)

        return linear_part.squeeze(-1) + dnn_part
```

## 5. 与 FM/DeepFM 对比

### 5.1 结构对比

```
FM:    Linear + 2nd-order Interaction
NFM:   Linear + MLP(2nd-order Interaction)
DeepFM: Linear + FM + DNN (parallel)
```

### 5.2 特点对比

| 模型 | 低阶交互 | 高阶交互 | 交互方式 |
|------|----------|----------|----------|
| FM | ✓ | ✗ | 内积 |
| NFM | ✓ | ✓ | MLP学习 |
| DeepFM | ✓ | ✓ | 并行结构 |
| cross layers | ✓ | ✓ | 串行结构 |

## 6. 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| embed_dim | 10-20 | 嵌入维度 |
| hidden_dims | [128, 64] | 隐藏层配置 |
| dropout | 0.1-0.3 | 防止过拟合 |
| learning_rate | 0.001 | Adam默认 |

## 7. 学习总结

### 7.1 核心要点

1. **Bi-Interaction Pooling**: 将二阶交互压缩为向量
2. **MLP**: 学习高阶特征交互
3. **串联结构**: 先池化再MLP

### 7.2 适用场景

- 需要高阶特征交互
- 特征数量适中
- 对模型可解释性有要求

## 8. 练习题

1. 推导 Bi-Interaction Pooling 的高效计算公式。

2. 比较 NFM 和 DeepFM 的性能差异。

3. 实现 NFM 的特征重要性分析。
