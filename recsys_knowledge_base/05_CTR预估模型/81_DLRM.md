# DLRM (Deep Learning Recommendation Model) 学习文档

## 1. 算法基础认知

### 1.1 什么是 DLRM？

DLRM 是 Facebook（Meta）提出的深度学习推荐模型，是工业级 CTR 预估的代表性架构，被广泛应用于大规模推荐系统。

### 1.2 核心特点

```
1. 底层特征交互：点积计算特征对之间的交互
2. 顶层 MLP：与稠密特征拼接后预测
3. 高效实现：针对推理优化
```

### 1.3 架构概览

```
稀疏特征 → 嵌入表 → 特征交互（点积）
                          ↓
稠密特征 → Bottom MLP ────→ 拼接 → Top MLP → 输出
```

## 2. 核心原理

### 2.1 特征交互

DLRM 的核心是用点积计算特征嵌入之间的交互：

$$\text{interaction}_{i,j} = \langle e_i, e_j \rangle$$

其中 $e_i, e_j$ 是第 i 和 j 个特征的嵌入向量。

### 2.2 模型结构

```
1. 稀疏特征处理:
   - Embedding Bag: sum/mean pooling

2. 稠密特征处理:
   - Bottom MLP: 提取高阶特征

3. 特征交互:
   - 所有序数特征嵌入两两点积

4. 预测:
   - Top MLP: 处理交互结果 + 稠密特征
```

## 3. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class EmbeddingBagLayer(nn.Module):
    """
    Embedding Bag 层

    对同类特征进行池化
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 mode: str = 'sum'):
        """
        参数:
            num_embeddings: 词表大小
            embedding_dim: 嵌入维度
            mode: 池化方式 ('sum', 'mean', 'max')
        """
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            mode=mode,
            include_last_offset=True
        )

    def forward(self, input: torch.Tensor,
                offsets: torch.Tensor = None) -> torch.Tensor:
        return self.embedding(input, offsets)


class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model
    """

    def __init__(self,
                 sparse_feature_sizes: List[int],
                 dense_feature_dim: int,
                 embedding_dim: int = 64,
                 bottom_mlp_dims: List[int] = [512, 256],
                 top_mlp_dims: List[int] = [1024, 512, 256, 1]):
        """
        参数:
            sparse_feature_sizes: 各稀疏特征的词表大小
            dense_feature_dim: 稠密特征维度
            embedding_dim: 嵌入维度
            bottom_mlp_dims: 底层 MLP 维度
            top_mlp_dims: 顶层 MLP 维度
        """
        super().__init__()

        self.num_sparse = len(sparse_feature_sizes)
        self.dense_feature_dim = dense_feature_dim
        self.embedding_dim = embedding_dim

        # 稀疏特征嵌入
        self.sparse_embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in sparse_feature_sizes
        ])

        # Bottom MLP (处理稠密特征)
        self.bottom_mlp = self._build_mlp(
            dense_feature_dim,
            bottom_mlp_dims,
            embedding_dim  # 输出与嵌入维度相同
        )

        # 计算 Top MLP 输入维度
        # 交互数量: C(n, 2) = n*(n-1)/2
        num_interactions = self.num_sparse * (self.num_sparse + 1) // 2
        top_input_dim = num_interactions + bottom_mlp_dims[-1] if bottom_mlp_dims else embedding_dim

        # Top MLP
        self.top_mlp = self._build_mlp(top_input_dim, top_mlp_dims[:-1], top_mlp_dims[-1])

    def _build_mlp(self, input_dim: int, hidden_dims: List[int],
                   output_dim: int = None) -> nn.Sequential:
        """构建 MLP"""
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim

        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _interact_features(self, sparse_embeds: List[torch.Tensor],
                          dense_embed: torch.Tensor) -> torch.Tensor:
        """
        计算特征交互

        参数:
            sparse_embeds: 各稀疏特征的嵌入
            dense_embed: 稠密特征经过 Bottom MLP 的输出

        返回:
            交互向量
        """
        batch_size = sparse_embeds[0].size(0)

        # 拼接所有嵌入
        all_embeds = torch.stack(sparse_embeds + [dense_embed], dim=1)
        # (batch, num_features, embedding_dim)

        # 计算两两内积
        interactions = torch.bmm(all_embeds, all_embeds.transpose(1, 2))
        # (batch, num_features, num_features)

        # 提取上三角（含对角线）并展平
        num_features = all_embeds.size(1)

        interaction_list = []
        for i in range(num_features):
            for j in range(i, num_features):
                interaction_list.append(interactions[:, i, j])

        interactions_flat = torch.stack(interaction_list, dim=1)
        # (batch, num_interactions)

        return interactions_flat

    def forward(self, sparse_features: torch.Tensor,
               dense_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            sparse_features: (batch, num_sparse) 稀疏特征索引
            dense_features: (batch, dense_dim) 稠密特征

        返回:
            (batch, 1) logits
        """
        # 稀疏特征嵌入
        sparse_embeds = [
            self.sparse_embeddings[i](sparse_features[:, i])
            for i in range(self.num_sparse)
        ]

        # Bottom MLP
        dense_embed = self.bottom_mlp(dense_features)

        # 特征交互
        interactions = self._interact_features(sparse_embeds, dense_embed)

        # Top MLP
        logits = self.top_mlp(interactions)

        return logits

    def predict(self, sparse_features: torch.Tensor,
               dense_features: torch.Tensor) -> torch.Tensor:
        """预测概率"""
        logits = self.forward(sparse_features, dense_features)
        return torch.sigmoid(logits)


class DLRMTrainer:
    """
    DLRM 训练器
    """

    def __init__(self, model: DLRM, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, sparse_features: torch.Tensor,
                  dense_features: torch.Tensor,
                  labels: torch.Tensor) -> float:
        """训练一步"""
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(sparse_features, dense_features)
        loss = self.criterion(logits.squeeze(-1), labels.float())

        loss.backward()
        self.optimizer.step()

        return loss.item()


def demo_dlrm():
    """DLRM 示例"""
    # 配置
    sparse_feature_sizes = [1000, 500, 200, 100]  # 4个稀疏特征
    dense_feature_dim = 50
    batch_size = 32
    n_samples = 1000

    # 创建模型
    model = DLRM(
        sparse_feature_sizes=sparse_feature_sizes,
        dense_feature_dim=dense_feature_dim,
        embedding_dim=32,
        bottom_mlp_dims=[128, 32],
        top_mlp_dims=[256, 128, 64, 1]
    )

    # 模拟数据
    sparse_features = torch.zeros(n_samples, len(sparse_feature_sizes), dtype=torch.long)
    for i, size in enumerate(sparse_feature_sizes):
        sparse_features[:, i] = torch.randint(0, size, (n_samples,))

    dense_features = torch.randn(n_samples, dense_feature_dim)
    labels = (torch.rand(n_samples) > 0.8).long()

    # 训练
    trainer = DLRMTrainer(model)

    for epoch in range(5):
        indices = torch.randperm(n_samples)[:batch_size]
        loss = trainer.train_step(
            sparse_features[indices],
            dense_features[indices],
            labels[indices]
        )
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # 预测
    model.eval()
    with torch.no_grad():
        probs = model.predict(sparse_features[:5], dense_features[:5])
        print(f"\n预测概率: {probs.squeeze().numpy()}")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")


if __name__ == "__main__":
    demo_dlrm()
```

## 4. 特征交互优化

### 4.1 高效交互计算

```python
class OptimizedInteraction(nn.Module):
    """
    优化的特征交互计算
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        高效计算所有特征对的内积

        参数:
            embeds: (batch, num_features, embedding_dim)

        返回:
            (batch, num_interactions)
        """
        batch_size, num_features, embedding_dim = embeds.shape

        # 方式1: 矩阵乘法
        interactions = torch.bmm(embeds, embeds.transpose(1, 2))

        # 方式2: 直接计算（更高效）
        # 使用 torch.einsum
        # interactions = torch.einsum('bif,bjf->bij', embeds, embeds)

        # 提取上三角
        mask = torch.triu(torch.ones(num_features, num_features, device=embeds.device))
        interactions = interactions[:, mask.bool()]

        return interactions
```

## 5. 分布式训练

### 5.1 数据并行

```python
class DLRMDistributed:
    """
    DLRM 分布式训练
    """

    @staticmethod
    def setup_model_parallel(sparse_feature_sizes: List[int],
                            world_size: int):
        """
        设置模型并行

        嵌入表分布在不同 GPU 上
        """
        import torch.distributed as dist

        # 按大小排序，均匀分配
        sorted_sizes = sorted(enumerate(sparse_feature_sizes),
                             key=lambda x: x[1], reverse=True)

        assignments = [[] for _ in range(world_size)]
        loads = [0] * world_size

        for idx, size in sorted_sizes:
            # 分配到负载最小的 GPU
            min_gpu = loads.index(min(loads))
            assignments[min_gpu].append(idx)
            loads[min_gpu] += size

        return assignments
```

## 6. 推理优化

### 6.1 量化

```python
def quantize_model(model: nn.Module) -> nn.Module:
    """
    模型量化

    动态量化 MLP，嵌入表保持 FP32
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # 只量化线性层
        dtype=torch.qint8
    )
    return quantized_model
```

### 6.2 TorchScript 导出

```python
def export_torchscript(model: DLRM,
                       sparse_example: torch.Tensor,
                       dense_example: torch.Tensor,
                       save_path: str):
    """
    导出为 TorchScript
    """
    model.eval()

    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (sparse_example, dense_example)
        )

    traced_model.save(save_path)
    print(f"模型已导出到: {save_path}")
```

## 7. 与其他模型对比

| 模型 | 特征交互方式 | 适用场景 |
|------|-------------|----------|
| DLRM | 点积 | 大规模工业场景 |
| DeepFM | FM内积 | 中小规模 |
| DCN | Cross Layer | 需要高阶交叉 |

## 8. 学习总结

### 8.1 核心要点

1. **特征交互**: 稀疏特征嵌入间的点积
2. **双路结构**: Bottom MLP + Top MLP
3. **工业优化**: 针对大规模推理设计

### 8.2 适用场景

- 大规模 CTR 预估
- 实时推荐系统
- 需要高效推理的场景

## 9. 练习题

1. 比较 DLRM 和 DeepFM 的性能差异。

2. 实现 DLRM 的模型并行版本。

3. 优化 DLRM 的特征交互计算效率。
