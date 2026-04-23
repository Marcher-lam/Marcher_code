# AFM (Attentional Factorization Machines) 学习文档

## 1. 算法基础认知

### 1.1 什么是 AFM？

AFM 在 FM 的基础上引入 **注意力机制**，让模型能够学习不同特征交互的重要性权重。

### 1.2 核心思想

```
FM:  所有特征交互同等重要
AFM: 不同特征交互有不同的权重（通过注意力学习）
```

### 1.3 为什么需要 AFM？

- 不是所有特征交互都有用
- 无关特征的交互可能引入噪声
- 注意力机制可以让模型关注有意义的交互

## 2. 核心原理

### 2.1 FM 回顾

$$\hat{y}_{FM} = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j$$

### 2.2 AFM 公式

$$\hat{y}_{AFM} = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \alpha_{ij} \langle v_i, v_j \rangle x_i x_j$$

注意力权重:
$$\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{(i,j)} \exp(s_{ij})}$$

注意力得分:
$$s_{ij} = h^T \text{ReLU}(W (v_i \odot v_j) \odot (x_i x_j) + b)$$

## 3. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class AttentionLayer(nn.Module):
    """
    AFM 的注意力层
    """

    def __init__(self, embed_dim: int, attention_dim: int = 16):
        """
        参数:
            embed_dim: 嵌入维度
            attention_dim: 注意力隐藏层维度
        """
        super().__init__()

        self.attention_dim = attention_dim

        # 注意力网络
        self.attention_mlp = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, interactions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            interactions: (batch_size, num_interactions, embed_dim)
                         特征对的逐元素积

        返回:
            weighted_sum: (batch_size, embed_dim) 加权求和
            attention_weights: (batch_size, num_interactions) 注意力权重
        """
        # 计算注意力得分
        attention_scores = self.attention_mlp(interactions)  # (batch, num_inter, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, num_inter)

        # Softmax 归一化
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, num_inter)

        # 加权求和
        weighted_sum = torch.einsum('bi,bid->bd', attention_weights, interactions)

        return weighted_sum, attention_weights


class AFM(nn.Module):
    """
    Attentional Factorization Machine
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 attention_dim: int = 16, dropout: float = 0.2):
        """
        参数:
            field_dims: 各域特征数量
            embed_dim: 嵌入维度
            attention_dim: 注意力隐藏维度
            dropout: Dropout比率
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

        # 注意力层
        self.attention = AttentionLayer(embed_dim, attention_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 输出投影
        self.projection = nn.Linear(embed_dim, 1)

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

        # 构建特征对的逐元素积
        interactions = self._build_pairwise_interactions(embeddings)

        # 注意力加权
        weighted_interactions, attention_weights = self.attention(interactions)

        # Dropout
        weighted_interactions = self.dropout(weighted_interactions)

        # 投影到标量
        afm_part = self.projection(weighted_interactions).squeeze(-1)

        # 组合
        output = linear_part.squeeze(-1) + afm_part

        return output.unsqueeze(-1)

    def _build_pairwise_interactions(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        构建特征对的逐元素积

        参数:
            embeddings: (batch, num_fields, embed_dim)

        返回:
            (batch, num_pairs, embed_dim)
        """
        batch_size, num_fields, embed_dim = embeddings.shape

        # 获取所有特征对
        interactions = []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                # 逐元素积
                interaction = embeddings[:, i, :] * embeddings[:, j, :]  # (batch, embed_dim)
                interactions.append(interaction)

        # 堆叠
        interactions = torch.stack(interactions, dim=1)  # (batch, num_pairs, embed_dim)

        return interactions

    def get_attention_weights(self, X: torch.Tensor) -> torch.Tensor:
        """获取注意力权重（用于可解释性）"""
        device = X.device
        X_offset = X + self.field_offsets.to(device)
        embeddings = self.embedding(X_offset)
        interactions = self._build_pairwise_interactions(embeddings)
        _, weights = self.attention(interactions)
        return weights


class AFMClassifier(nn.Module):
    """
    AFM 分类器
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 attention_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.afm = AFM(field_dims, embed_dim, attention_dim, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.afm(X))


class AFMTrainer:
    """
    AFM 训练器
    """

    def __init__(self, model: AFMClassifier,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 l2_attention: float = 0.0):
        """
        参数:
            l2_attention: 注意力权重的L2正则化系数
        """
        self.model = model
        self.l2_attention = l2_attention

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
        bce_loss = self.criterion(pred.squeeze(), y.float())

        # 注意力正则化（可选）
        if self.l2_attention > 0:
            attention_weights = self.model.afm.get_attention_weights(X)
            attention_reg = torch.mean(attention_weights ** 2)
            loss = bce_loss + self.l2_attention * attention_reg
        else:
            loss = bce_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_feature_importance(self, X: torch.Tensor,
                               feature_names: List[str]) -> Dict:
        """
        获取特征交互重要性

        参数:
            X: 单个样本 (1, num_fields)
            feature_names: 特征名称列表

        返回:
            {特征对: 注意力权重}
        """
        self.model.eval()
        with torch.no_grad():
            weights = self.model.afm.get_attention_weights(X)

        weights = weights.squeeze().cpu().numpy()

        # 构建特征对
        pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                pairs.append(f"{feature_names[i]} x {feature_names[j]}")

        importance = {pair: float(weight) for pair, weight in zip(pairs, weights)}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def demo_afm():
    """AFM 示例"""
    # 配置
    field_dims = [100, 50, 20, 10]
    feature_names = ['user', 'item', 'category', 'time']

    # 创建模型
    model = AFMClassifier(
        field_dims=field_dims,
        embed_dim=10,
        attention_dim=16,
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

    # 获取注意力权重
    trainer = AFMTrainer(model)
    single_X = X[:1]
    importance = trainer.get_feature_importance(single_X, feature_names)

    print("\n特征交互重要性:")
    for pair, weight in importance.items():
        print(f"  {pair}: {weight:.4f}")


if __name__ == "__main__":
    demo_afm()
```

## 4. 可视化注意力权重

```python
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_attention(attention_weights: np.ndarray,
                       feature_names: List[str],
                       title: str = "Feature Interaction Attention"):
    """
    可视化注意力权重热力图

    参数:
        attention_weights: (num_pairs,) 注意力权重
        feature_names: 特征名称
    """
    n = len(feature_names)
    matrix = np.zeros((n, n))

    # 填充矩阵
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = attention_weights[idx]
            matrix[j, i] = attention_weights[idx]
            idx += 1

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd'
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_top_interactions(importance: Dict, top_k: int = 10):
    """可视化Top-K特征交互"""
    items = list(importance.items())[:top_k]
    pairs = [item[0] for item in items]
    weights = [item[1] for item in items]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(pairs)), weights)
    plt.yticks(range(len(pairs)), pairs)
    plt.xlabel('Attention Weight')
    plt.title(f'Top {top_k} Feature Interactions')
    plt.gca().invert_yaxis()

    # 添加数值标签
    for bar, weight in zip(bars, weights):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{weight:.4f}', va='center')

    plt.tight_layout()
    plt.show()
```

## 5. AFM vs FM 对比

### 5.1 模型复杂度

```
FM 参数: n × k (嵌入)
AFM 参数: n × k + k × a + a (注意力网络)

其中:
- n: 特征数
- k: 嵌入维度
- a: 注意力隐藏维度
```

### 5.2 性能对比

| 指标 | FM | AFM |
|------|-----|------|
| 参数量 | 少 | 中 |
| 训练速度 | 快 | 中 |
| 预测速度 | 快 | 中 |
| 可解释性 | 低 | 高 |
| 精度 | 中 | 高 |

## 6. 应用场景

### 6.1 适合 AFM 的场景

```
1. 特征交互差异大的场景
   - 用户 x 商品类别 重要
   - 用户 x 时间段 不重要

2. 需要可解释性
   - 分析哪些特征组合最重要

3. 稀疏特征
   - 自动学习有意义的交互
```

### 6.2 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| embed_dim | 10-20 | 嵌入维度 |
| attention_dim | 16-64 | 注意力隐藏层 |
| dropout | 0.1-0.3 | 防止过拟合 |
| l2_attention | 0.001 | 注意力正则化 |

## 7. 学习总结

### 7.1 核心要点

1. **注意力机制**: 学习交互权重
2. **加权求和**: 替代简单求和
3. **可解释性**: 可分析重要交互

### 7.2 优缺点

**优点:**
- 自动学习交互重要性
- 提供可解释性
- 精度通常优于 FM

**缺点:**
- 参数量增加
- 计算复杂度增加
- 需要更多训练数据

## 8. 练习题

1. 实现多头注意力版本的 AFM。

2. 比较 AFM 和 FM 在不同数据集上的表现。

3. 设计实验验证 AFM 的可解释性。
