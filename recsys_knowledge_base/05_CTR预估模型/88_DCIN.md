# DCN (Deep & Cross Network) 学习文档

## 1. 算法基础认知

### 1.1 什么是 DCN？

DCN 是 Google 提出的 CTR 预估模型，通过 Cross Network 显式建模有界阶特征交叉，与 Deep Network 并行学习高阶隐式交叉。

### 1.2 核心创新

```
Cross Network: 显式、有界阶特征交叉
Deep Network:  隐式、高阶特征交叉
并行结构:      两者结合
```

### 1.3 模型架构

```
       ┌─────────────┐
       │  Embedding  │
       └──────┬──────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌─────────┐      ┌─────────┐
│  Cross  │      │  Deep   │
│ Network │      │ Network │
└────┬────┘      └────┬────┘
     │                │
     └────────┬───────┘
              ▼
       ┌─────────────┐
       │ Combination │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │   Output    │
       └─────────────┘
```

## 2. 核心原理

### 2.1 Cross Network

Cross Network 的核心是 Cross Layer:

$$x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$$

其中:
- $x_0$: 原始输入（常数）
- $x_l$: 第 l 层输出
- $W_l, b_l$: 可学习参数
- $\odot$: 逐元素乘积

### 2.2 特征交叉的阶数

```
Layer 0: x_0 (1阶)
Layer 1: x_0 ⊙ (W_1 x_0) (2阶交叉)
Layer 2: x_0 ⊙ (W_2 (x_0 ⊙ (W_1 x_0))) (3阶交叉)
...
Layer L: L+1 阶交叉
```

### 2.3 完整实现

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class CrossLayer(nn.Module):
    """
    Cross Network 单层
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x0: 原始输入 (batch, input_dim)
            x: 当前层输入 (batch, input_dim)

        返回:
            下一层输出 (batch, input_dim)
        """
        # x_{l+1} = x_0 * (W * x_l + b) + x_l
        linear = torch.matmul(x, self.weight.unsqueeze(-1)).squeeze(-1) + self.bias
        cross = x0 * linear.unsqueeze(-1)
        return cross + x


class CrossNetwork(nn.Module):
    """
    Cross Network
    """

    def __init__(self, input_dim: int, num_layers: int = 6):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: (batch, input_dim)

        返回:
            (batch, input_dim)
        """
        x0 = x
        for layer in self.layers:
            x = layer(x0, x)
        return x


class DeepNetwork(nn.Module):
    """
    Deep Network (标准 MLP)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DCN(nn.Module):
    """
    Deep & Cross Network
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 num_cross_layers: int = 6,
                 deep_hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.2):
        """
        参数:
            field_dims: 各域特征数量
            embed_dim: 嵌入维度
            num_cross_layers: Cross 层数
            deep_hidden_dims: Deep 网络隐藏层维度
            dropout: Dropout 比率
        """
        super().__init__()

        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # 嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])

        # 线性部分
        self.linear = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in field_dims
        ])
        self.bias = nn.Parameter(torch.zeros(1))

        # 输入维度
        input_dim = len(field_dims) * embed_dim

        # Cross Network
        self.cross_net = CrossNetwork(input_dim, num_cross_layers)

        # Deep Network
        self.deep_net = DeepNetwork(input_dim, deep_hidden_dims, dropout)

        # 输出层
        cross_output_dim = input_dim
        deep_output_dim = deep_hidden_dims[-1]
        self.output = nn.Linear(cross_output_dim + deep_output_dim, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            X: (batch, num_fields) 特征索引

        返回:
            (batch, 1) 预测值
        """
        batch_size = X.size(0)

        # 线性部分
        linear_part = self.bias
        for i, embedding in enumerate(self.linear):
            linear_part = linear_part + embedding(X[:, i])

        # 嵌入
        embeds = []
        for i, embedding in enumerate(self.embeddings):
            embeds.append(embedding(X[:, i]))

        # 拼接嵌入
        concat_embed = torch.cat(embeds, dim=-1)  # (batch, num_fields * embed_dim)

        # Cross Network
        cross_output = self.cross_net(concat_embed)  # (batch, input_dim)

        # Deep Network
        deep_output = self.deep_net(concat_embed)  # (batch, hidden_dims[-1])

        # 组合
        combined = torch.cat([cross_output, deep_output], dim=-1)

        # 输出
        output = self.output(combined)

        return output + linear_part


class DCNClassifier(nn.Module):
    """
    DCN 分类器
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 num_cross_layers: int = 6,
                 deep_hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.2):
        super().__init__()
        self.dcn = DCN(field_dims, embed_dim, num_cross_layers,
                      deep_hidden_dims, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.dcn(X))


class DCNTrainer:
    """
    DCN 训练器
    """

    def __init__(self, model: DCNClassifier,
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
        """训练一步"""
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(X)
        loss = self.criterion(pred.squeeze(), y.float())

        loss.backward()
        self.optimizer.step()

        return loss.item()


def demo_dcn():
    """DCN 示例"""
    # 配置
    field_dims = [1000, 500, 100, 50]

    # 创建模型
    model = DCNClassifier(
        field_dims=field_dims,
        embed_dim=10,
        num_cross_layers=4,
        deep_hidden_dims=[128, 64],
        dropout=0.2
    )

    # 模拟数据
    batch_size = 32
    X = torch.zeros(batch_size, len(field_dims), dtype=torch.long)
    for i, dim in enumerate(field_dims):
        X[:, i] = torch.randint(0, dim, (batch_size,))

    # 前向传播
    output = model(X)

    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"预测范围: [{output.min():.4f}, {output.max():.4f}]")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")


if __name__ == "__main__":
    demo_dcn()
```

## 3. DCN v2 (DCN-V2)

### 3.1 改进点

DCN-V2 使用矩阵代替向量作为 Cross Layer 的权重:

$$x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$$

其中 $W_l$ 是矩阵而非向量。

### 3.2 实现

```python
class CrossLayerV2(nn.Module):
    """
    Cross Network V2 单层（矩阵版本）
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x0: 原始输入 (batch, input_dim)
            x: 当前层输入 (batch, input_dim)

        返回:
            下一层输出 (batch, input_dim)
        """
        # x_{l+1} = x_0 * (W * x_l + b) + x_l
        linear = torch.matmul(x, self.weight) + self.bias
        cross = x0 * linear
        return cross + x


class LowRankCrossLayer(nn.Module):
    """
    低秩 Cross Layer（参数更少）
    """

    def __init__(self, input_dim: int, low_rank: int = 64):
        super().__init__()
        self.U = nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(low_rank, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # W ≈ U @ V, 低秩分解
        # linear = x @ U @ V + b
        linear = torch.matmul(torch.matmul(x, self.U), self.V) + self.bias
        cross = x0 * linear
        return cross + x
```

## 4. Cross Network 分析

### 4.1 特征交叉可视化

```python
def visualize_cross_interactions(cross_weights: List[torch.Tensor],
                                feature_names: List[str]):
    """
    可视化 Cross Network 的特征交互
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_layers = len(cross_weights)
    n_features = len(feature_names)

    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))

    for i, (weight, ax) in enumerate(zip(cross_weights, axes)):
        # 计算特征重要性
        importance = weight.abs().mean(dim=-1).cpu().numpy()

        sns.barplot(x=feature_names, y=importance, ax=ax)
        ax.set_title(f'Layer {i+1}')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')

    plt.tight_layout()
    plt.show()
```

## 5. 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| num_cross_layers | 4-6 | Cross 层数 |
| embed_dim | 10-20 | 嵌入维度 |
| deep_hidden_dims | [256, 128] | Deep 网络结构 |
| dropout | 0.1-0.3 | 防止过拟合 |

## 6. 与其他模型对比

| 模型 | 显式交叉 | 隐式交叉 | 交叉阶数 |
|------|----------|----------|----------|
| FM | ✓ | ✗ | 2 |
| DeepFM | ✓ | ✓ | 2 + 高阶 |
| DCN | ✓ | ✓ | 有界 + 高阶 |
| xDeepFM | ✓ | ✓ | 向量级 |

## 7. 学习总结

### 7.1 核心要点

1. **Cross Layer**: 显式、有界阶特征交叉
2. **残差连接**: 保持信息流动
3. **并行结构**: 结合显式和隐式交叉

### 7.2 优缺点

**优点:**
- 显式建模特征交叉
- 参数效率高（DCN v1）
- 可解释性强

**缺点:**
- 交叉阶数有限
- 可能不如纯深度模型灵活

## 8. 练习题

1. 推导 Cross Layer 的特征交叉阶数。

2. 比较 DCN v1 和 DCN v2 的效果差异。

3. 实现混合专家版本的 Cross Network。