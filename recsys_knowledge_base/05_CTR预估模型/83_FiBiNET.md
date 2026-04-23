# FiBiNET (Feature Importance & Bilinear Interaction) 学习文档

## 1. 算法基础认知

### 1.1 什么是 FiBiNET？

FiBiNET 是新浪提出的 CTR 预估模型，通过 **SENET** 学习特征重要性，并使用**双线性交互**代替传统的内积来捕捉特征交互。

### 1.2 核心创新

```
1. SENET: 动态学习特征重要性权重
2. 双线性交互: 比 FM 内积更强的交互能力
3. 结合池化: 提取更丰富的交互信息
```

### 1.3 与 FM 的区别

| 方法 | 特征交互 | 表达能力 |
|------|----------|----------|
| FM | 内积 | 线性 |
| FiBiNET | 双线性 | 非线性 |

## 2. 核心原理

### 2.1 SENET (Squeeze-and-Excitation)

SENET 动态学习每个特征域的重要性：

```
1. Squeeze: 对每个域的嵌入求均值
   z_i = mean(e_i)

2. Excitation: 两层 MLP 学习权重
   a = σ(W_2 · ReLU(W_1 · z))

3. Reweight: 用权重缩放嵌入
   e'_i = a_i · e_i
```

### 2.2 双线性交互

传统 FM 交互:
$$\text{FM}(i, j) = \langle e_i, e_j \rangle$$

FiBiNET 双线性交互:
$$\text{Bilinear}(i, j) = e_i^T W e_j$$

其中 $W$ 是可学习的交互矩阵。

### 2.3 池化策略

```python
# 组合池化
output = concat([
    combination_pool(bilinear_interactions),  # 组合池化
    attention_pool(bilinear_interactions)     # 注意力池化
])
```

## 3. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class SENETLayer(nn.Module):
    """
    Squeeze-and-Excitation Network for Feature Importance
    """

    def __init__(self, num_fields: int, reduction_ratio: float = 3):
        """
        参数:
            num_fields: 特征域数量
            reduction_ratio: 压缩比率
        """
        super().__init__()

        self.num_fields = num_fields

        # 中间维度
        reduced_size = max(1, num_fields // reduction_ratio)

        # Excitation 网络
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields),
            nn.ReLU()
        )

    def forward(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        参数:
            embeddings: 各域的嵌入列表，每个形状 (batch, embed_dim)

        返回:
            加权后的嵌入列表
        """
        # Squeeze: 对每个域的嵌入求均值
        squeeze = torch.stack(
            [emb.mean(dim=-1) for emb in embeddings],
            dim=-1
        )  # (batch, num_fields) - 这里应该是直接 mean

        # 修正：对每个嵌入整体求均值
        squeeze = torch.stack(
            [emb.mean(dim=1) for emb in embeddings],
            dim=1
        )  # (batch, num_fields, embed_dim) -> mean -> (batch, num_fields)

        squeeze = torch.stack(
            [emb.mean(dim=1) for emb in embeddings],
            dim=1
        ).mean(dim=-1)  # (batch, num_fields)

        # Excitation: 学习权重
        excitation = self.excitation(squeeze)  # (batch, num_fields)
        weights = torch.sigmoid(excitation)

        # Reweight: 加权
        reweighted = []
        for i, emb in enumerate(embeddings):
            reweighted.append(emb * weights[:, i:i+1])

        return reweighted


class BilinearInteraction(nn.Module):
    """
    双线性特征交互
    """

    def __init__(self, embed_dim: int, num_fields: int,
                 bilinear_type: str = 'all'):
        """
        参数:
            embed_dim: 嵌入维度
            num_fields: 特征域数量
            bilinear_type: 双线性类型
                - 'all': 所有交互共享一个 W
                - 'each': 每个域一个 W
                - 'interaction': 每对交互一个 W
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_fields = num_fields
        self.bilinear_type = bilinear_type

        num_interactions = num_fields * (num_fields - 1) // 2

        if bilinear_type == 'all':
            # 所有交互共享一个矩阵
            self.W = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        elif bilinear_type == 'each':
            # 每个域一个矩阵
            self.W_list = nn.ParameterList([
                nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
                for _ in range(num_fields)
            ])
        elif bilinear_type == 'interaction':
            # 每对交互一个矩阵
            self.W_list = nn.ParameterList([
                nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
                for _ in range(num_interactions)
            ])

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        计算双线性交互

        参数:
            embeddings: 嵌入列表

        返回:
            (batch, num_interactions, embed_dim)
        """
        interactions = []

        if self.bilinear_type == 'all':
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # e_i^T W e_j
                    interaction = torch.matmul(
                        embeddings[i],
                        self.W
                    ) * embeddings[j]
                    interactions.append(interaction)

        elif self.bilinear_type == 'each':
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    interaction = torch.matmul(
                        embeddings[i],
                        self.W_list[i]
                    ) * embeddings[j]
                    interactions.append(interaction)

        elif self.bilinear_type == 'interaction':
            idx = 0
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    interaction = torch.matmul(
                        embeddings[i],
                        self.W_list[idx]
                    ) * embeddings[j]
                    interactions.append(interaction)
                    idx += 1

        return torch.stack(interactions, dim=1)


class CombinationLayer(nn.Module):
    """
    组合池化层
    """

    def __init__(self):
        super().__init__()

    def forward(self, interactions: torch.Tensor) -> torch.Tensor:
        """
        参数:
            interactions: (batch, num_interactions, embed_dim)

        返回:
            (batch, embed_dim)
        """
        # 对所有交互求和
        return interactions.sum(dim=1)


class AttentionLayer(nn.Module):
    """
    注意力池化层
    """

    def __init__(self, embed_dim: int, attention_dim: int = 16):
        super().__init__()

        self.attention_mlp = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, interactions: torch.Tensor) -> torch.Tensor:
        """
        参数:
            interactions: (batch, num_interactions, embed_dim)

        返回:
            (batch, embed_dim)
        """
        # 计算注意力分数
        scores = self.attention_mlp(interactions)  # (batch, num_interactions, 1)
        weights = F.softmax(scores, dim=1)

        # 加权求和
        output = torch.sum(weights * interactions, dim=1)

        return output


class FiBiNET(nn.Module):
    """
    Feature Importance & Bilinear Interaction Network
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 reduction_ratio: int = 3,
                 bilinear_type: str = 'all',
                 mlp_dims: List[int] = [128, 64],
                 dropout: float = 0.2):
        """
        参数:
            field_dims: 各域特征数量
            embed_dim: 嵌入维度
            reduction_ratio: SENET 压缩比率
            bilinear_type: 双线性类型
            mlp_dims: MLP 维度
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

        # SENET
        self.senet = SENETLayer(self.num_fields, reduction_ratio)

        # 双线性交互（原始和 SENET 加权后各一个）
        self.bilinear_original = BilinearInteraction(
            embed_dim, self.num_fields, bilinear_type
        )
        self.bilinear_senet = BilinearInteraction(
            embed_dim, self.num_fields, bilinear_type
        )

        # 池化层
        self.combination = CombinationLayer()
        self.attention = AttentionLayer(embed_dim)

        # MLP 输入维度
        num_interactions = self.num_fields * (self.num_fields - 1) // 2
        mlp_input_dim = embed_dim * 4  # comb_orig + attn_orig + comb_senet + attn_senet

        self.mlp = self._build_mlp(mlp_input_dim, mlp_dims, dropout)

    def _build_mlp(self, input_dim: int, mlp_dims: List[int],
                   dropout: float) -> nn.Sequential:
        """构建 MLP"""
        layers = []
        prev_dim = input_dim

        for dim in mlp_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            X: (batch, num_fields) 特征索引

        返回:
            (batch, 1) logits
        """
        # 线性部分
        linear_part = self.bias
        for i, embedding in enumerate(self.linear):
            linear_part = linear_part + embedding(X[:, i])

        # 嵌入
        embeddings = [
            self.embeddings[i](X[:, i])
            for i in range(self.num_fields)
        ]

        # SENET 加权
        senet_embeddings = self.senet(embeddings)

        # 原始嵌入的双线性交互
        interactions_orig = self.bilinear_original(embeddings)
        comb_orig = self.combination(interactions_orig)
        attn_orig = self.attention(interactions_orig)

        # SENET 嵌入的双线性交互
        interactions_senet = self.bilinear_senet(senet_embeddings)
        comb_senet = self.combination(interactions_senet)
        attn_senet = self.attention(interactions_senet)

        # 拼接
        combined = torch.cat([
            comb_orig, attn_orig, comb_senet, attn_senet
        ], dim=-1)

        # MLP
        mlp_output = self.mlp(combined)

        return linear_part + mlp_output


class FiBiNETClassifier(nn.Module):
    """
    FiBiNET 分类器
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 reduction_ratio: int = 3, bilinear_type: str = 'all',
                 mlp_dims: List[int] = [128, 64]):
        super().__init__()
        self.fibinet = FiBiNET(
            field_dims, embed_dim, reduction_ratio,
            bilinear_type, mlp_dims
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fibinet(X))


def demo_fibinet():
    """FiBiNET 示例"""
    # 配置
    field_dims = [100, 50, 20, 10]
    batch_size = 32
    n_samples = 1000

    # 创建模型
    model = FiBiNETClassifier(
        field_dims=field_dims,
        embed_dim=10,
        reduction_ratio=2,
        bilinear_type='all',
        mlp_dims=[64, 32]
    )

    # 模拟数据
    X = torch.zeros(n_samples, len(field_dims), dtype=torch.long)
    for i, dim in enumerate(field_dims):
        X[:, i] = torch.randint(0, dim, (n_samples,))
    y = (torch.rand(n_samples) > 0.8).long()

    # 前向传播
    output = model(X[:batch_size])
    print(f"输入形状: {X[:batch_size].shape}")
    print(f"输出形状: {output.shape}")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")


if __name__ == "__main__":
    demo_fibinet()
```

## 4. 三种双线性交互对比

| 类型 | 参数量 | 表达能力 | 适用场景 |
|------|--------|----------|----------|
| all | O(d²) | 低 | 特征域少 |
| each | O(nd²) | 中 | 特征域适中 |
| interaction | O(n²d²) | 高 | 特征域少、精度要求高 |

## 5. 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| embed_dim | 10-20 | 嵌入维度 |
| reduction_ratio | 3 | SENET 压缩比率 |
| bilinear_type | 'each' | 平衡精度和参数量 |
| mlp_dims | [128, 64] | MLP 结构 |

## 6. 学习总结

### 6.1 核心要点

1. **SENET**: 动态学习特征重要性
2. **双线性交互**: 比 FM 更强的交互能力
3. **双路径**: 原始和 SENET 加权特征分别交互

### 6.2 适用场景

- 特征重要性差异大
- 需要更强的特征交互
- 中等规模数据

## 7. 练习题

1. 比较三种双线性交互的效果和效率。

2. 将 FiBiNET 与其他 CTR 模型（如 DeepFM）对比。

3. 实现 SENET 的其他变体（如使用 max pooling）。
