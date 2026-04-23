# FFM (Field-aware Factorization Machines) 学习文档

## 1. 算法基础认知

### 1.1 什么是 FFM？

FFM 是 FM 的改进版本，引入了 **Field（域）** 的概念，不同域的特征在交互时使用不同的隐向量。

### 1.2 FFM vs FM

| 模型 | 隐向量数量 | 交互方式 |
|------|-----------|----------|
| FM | 每个特征1个 | 所有交互共享 |
| FFM | 每个特征f个 | 按域区分交互 |

### 1.3 为什么需要 FFM？

- FM 假设所有特征交互使用相同隐向量
- 实际中，特征属于不同域（如用户域、物品域、上下文域）
- 不同域之间的交互应该有不同的表示

## 2. 核心原理

### 2.1 Field 概念

```
特征分组为不同的域(Field):
- 用户域: user_id, age, gender
- 物品域: item_id, category, price
- 上下文域: time, location, device

每个特征对不同域有独立的隐向量:
- user_id 有: 对物品域的隐向量、对上下文域的隐向量
```

### 2.2 数学公式

**FM 的交互项:**
$$\hat{y}_{FM} = \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j$$

**FFM 的交互项:**
$$\hat{y}_{FFM} = \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_{i,f_j}, v_{j,f_i} \rangle x_i x_j$$

其中:
- $f_j$ 是特征 j 所属的域
- $v_{i,f_j}$ 是特征 i 对域 $f_j$ 的隐向量

### 2.3 参数量对比

```
FM 参数量: n × k
FFM 参数量: n × f × k

其中:
- n: 特征数量
- k: 隐向量维度
- f: 域的数量
```

## 3. 完整实现

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional


class FieldAwareFactorizationMachine(nn.Module):
    """
    FFM 模型
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10):
        """
        参数:
            field_dims: 各域的特征数量 [field1_dim, field2_dim, ...]
            embed_dim: 隐向量维度
        """
        super().__init__()

        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # 线性部分
        self.linear = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in field_dims
        ])

        # FFM 嵌入: 每个域的特征对其他所有域有独立的嵌入
        # embeddings[i][j]: 域i的特征对域j的嵌入
        self.ffm_embeddings = nn.ModuleList([
            nn.ModuleList([
                nn.Embedding(field_dims[i], embed_dim)
                for j in range(self.num_fields)
            ])
            for i in range(self.num_fields)
        ])

        # 偏置
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            X: (batch_size, num_fields) 每个域的特征索引

        返回:
            (batch_size, 1) 预测值
        """
        batch_size = X.size(0)

        # 线性部分
        linear_part = self.bias
        for i in range(self.num_fields):
            linear_part = linear_part + self.linear[i](X[:, i])

        # FFM 交互部分
        ffm_part = torch.zeros(batch_size, 1, device=X.device)

        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                # 特征i对域j的嵌入
                v_i = self.ffm_embeddings[i][j](X[:, i])  # (batch, embed_dim)
                # 特征j对域i的嵌入
                v_j = self.ffm_embeddings[j][i](X[:, j])  # (batch, embed_dim)

                # 内积
                interaction = torch.sum(v_i * v_j, dim=1, keepdim=True)
                ffm_part = ffm_part + interaction

        # 组合
        output = linear_part + ffm_part

        return output


class FFMClassifier(nn.Module):
    """
    FFM 分类器
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10):
        super().__init__()
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.ffm(X))


class FFMTrainer:
    """
    FFM 训练器
    """

    def __init__(self, model: FFMClassifier,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()

    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for X, y in dataloader:
            self.optimizer.zero_grad()

            pred = self.model(X)
            loss = self.criterion(pred.squeeze(), y.float())

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X)
                loss = self.criterion(pred.squeeze(), y.float())
                total_loss += loss.item()

                all_preds.extend(pred.squeeze().cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算 AUC
        from sklearn.metrics import roc_auc_score, log_loss
        auc = roc_auc_score(all_labels, all_preds)
        logloss = log_loss(all_labels, all_preds)

        return {
            'loss': total_loss / len(dataloader),
            'auc': auc,
            'logloss': logloss
        }


def demo_ffm():
    """FFM 示例"""
    # 模拟数据
    # 假设有3个域: 用户(1000), 物品(500), 上下文(100)
    field_dims = [1000, 500, 100]

    # 创建模型
    model = FFMClassifier(field_dims, embed_dim=10)

    # 模拟输入
    batch_size = 32
    X = torch.randint(0, 1000, (batch_size, 3))
    X[:, 1] = torch.randint(0, 500, (batch_size,))
    X[:, 2] = torch.randint(0, 100, (batch_size,))

    # 前向传播
    output = model(X)
    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"预测值范围: [{output.min():.4f}, {output.max():.4f}]")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 对比 FM 参数量
    n = sum(field_dims)
    k = 10
    fm_params = n * k + n + 1
    ffm_params = n * len(field_dims) * k + n + 1
    print(f"FM 参数量: {fm_params:,}")
    print(f"FFM 参数量: {ffm_params:,}")


if __name__ == "__main__":
    demo_ffm()
```

## 4. Field 划分策略

### 4.1 按特征类型划分

```python
class FieldMapper:
    """
    Field 映射器
    """

    def __init__(self):
        self.field_info = {}
        self.field_dims = []

    def add_field(self, field_name: str, feature_names: List[str],
                  vocab_sizes: Dict[str, int]):
        """
        添加一个域

        参数:
            field_name: 域名称
            feature_names: 该域包含的特征
            vocab_sizes: 各特征的词表大小
        """
        field_idx = len(self.field_dims)

        # 计算该域的总维度
        field_dim = sum(vocab_sizes.get(f, 1) for f in feature_names)

        self.field_dims.append(field_dim)
        self.field_info[field_name] = {
            'idx': field_idx,
            'features': feature_names,
            'vocab_sizes': vocab_sizes
        }

    def get_field_dims(self) -> List[int]:
        """获取各域维度"""
        return self.field_dims


def create_field_mapper() -> FieldMapper:
    """创建推荐系统的 Field 映射"""
    mapper = FieldMapper()

    # 用户域
    mapper.add_field(
        'user',
        ['user_id', 'age', 'gender'],
        {'user_id': 10000, 'age': 10, 'gender': 3}
    )

    # 物品域
    mapper.add_field(
        'item',
        ['item_id', 'category', 'brand'],
        {'item_id': 50000, 'category': 100, 'brand': 1000}
    )

    # 上下文域
    mapper.add_field(
        'context',
        ['hour', 'day_of_week', 'device'],
        {'hour': 24, 'day_of_week': 7, 'device': 5}
    )

    return mapper
```

## 5. 优化技巧

### 5.1 内存优化

```python
class MemoryEfficientFFM(nn.Module):
    """
    内存优化的 FFM

    使用共享嵌入减少参数量
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 use_shared_embedding: bool = True):
        super().__init__()

        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.use_shared = use_shared_embedding

        # 线性部分
        self.linear = nn.Embedding(sum(field_dims), 1)
        self.field_offsets = torch.tensor([0] + np.cumsum(field_dims)[:-1].tolist())

        if use_shared_embedding:
            # 共享基础嵌入
            self.base_embedding = nn.Embedding(sum(field_dims), embed_dim)
            # 每个域的变换矩阵
            self.field_transforms = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim, bias=False)
                for _ in range(self.num_fields)
            ])
        else:
            # 标准 FFM 嵌入
            self.ffm_embeddings = nn.ModuleList([
                nn.ModuleList([
                    nn.Embedding(field_dims[i], embed_dim)
                    for j in range(self.num_fields)
                ])
                for i in range(self.num_fields)
            ])

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size = X.size(0)

        # 加上偏移
        X_offset = X + self.field_offsets.to(X.device)

        # 线性部分
        linear_part = self.bias + self.linear(X_offset).sum(dim=1)

        # FFM 交互部分
        ffm_part = torch.zeros(batch_size, 1, device=X.device)

        if self.use_shared:
            # 获取基础嵌入
            base_embeds = self.base_embedding(X_offset)  # (batch, num_fields, embed_dim)

            for i in range(self.num_fields):
                for j in range(i + 1, self.num_fields):
                    # 应用域变换
                    v_i = self.field_transforms[j](base_embeds[:, i, :])
                    v_j = self.field_transforms[i](base_embeds[:, j, :])

                    interaction = torch.sum(v_i * v_j, dim=1, keepdim=True)
                    ffm_part = ffm_part + interaction
        else:
            for i in range(self.num_fields):
                for j in range(i + 1, self.num_fields):
                    v_i = self.ffm_embeddings[i][j](X[:, i])
                    v_j = self.ffm_embeddings[j][i](X[:, j])

                    interaction = torch.sum(v_i * v_j, dim=1, keepdim=True)
                    ffm_part = ffm_part + interaction

        return linear_part.unsqueeze(-1) + ffm_part
```

## 6. 与其他模型对比

### 6.1 实验对比

| 模型 | 参数量 | Criteo AUC | 训练时间 |
|------|--------|------------|----------|
| LR | 小 | 0.780 | 快 |
| FM | 中 | 0.785 | 中 |
| FFM | 大 | 0.788 | 慢 |
| DeepFM | 大 | 0.790 | 慢 |

### 6.2 适用场景

```
FFM 适合:
1. 特征有明确域划分的场景
2. 域数量适中的情况 (f < 20)
3. 内存资源充足

不适合:
1. 域数量过多 (参数爆炸)
2. 稀疏特征很多
3. 需要快速迭代的场景
```

## 7. 学习总结

### 7.1 核心要点

1. **Field-aware**: 不同域使用不同隐向量
2. **参数量**: 比 FM 大 f 倍
3. **适用性**: 特征域划分明确的场景

### 7.2 与 FM 的关系

- FFM 是 FM 的泛化
- 当 f=1 时，FFM 退化为 FM
- FFM 能学习更细粒度的特征交互

## 8. 练习题

1. 实现 FFM 的梯度计算。

2. 比较 FFM 和 FM 在稀疏数据上的表现。

3. 设计一种减少 FFM 参数量的方法。
