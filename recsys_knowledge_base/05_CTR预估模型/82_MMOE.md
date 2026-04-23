# MMOE 学习文档

## 1. 算法基础认知

### 1.1 什么是 MMOE？

MMOE（Multi-gate Mixture-of-Experts）是 Google 在 2018 年提出的**多任务学习**模型。它通过多个专家网络和门控机制来解决多任务之间的冲突问题。

### 1.2 动机

**多任务学习的挑战：**
- 任务之间可能存在冲突
- 简单的参数共享效果不好
- 需要平衡不同任务的学习

**MMOE 的解决方案：**
- 多个专家网络学习不同的表示
- 门控机制为每个任务选择专家
- 灵活地组合专家输出

### 1.3 应用场景

- 推荐系统：同时优化 CTR 和 CVR
- 广告系统：点击率和转化率
- 视频推荐：点击率、观看时长、点赞率

## 2. 模型架构

### 2.1 整体结构

```
                输入特征
                    ↓
        ┌──────────────────────┐
        │   Shared Bottom      │ (可选)
        └──────────────────────┘
                    ↓
    ┌───────┬───────┬───────┬───────┐
    │Expert1│Expert2│Expert3│Expert4│  ← 专家网络
    └───────┴───────┴───────┴───────┘
         ↓       ↓       ↓       ↓
    ┌─────────────────────────────┐
    │      Gate (Task A)          │  ← 门控网络 A
    └─────────────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │   Tower (Task A)            │  ← 任务塔 A
    └─────────────────────────────┘
              ↓
          Output A

    (类似地，Gate B + Tower B -> Output B)
```

### 2.2 数学表示

专家输出：$E_i(x) = f_i(x), i = 1, ..., n$

门控输出：$g^k(x) = \text{softmax}(W_g^k x)$

任务输入：$h^k(x) = \sum_{i=1}^{n} g_i^k(x) E_i(x)$

任务输出：$y^k = \text{tower}^k(h^k(x))$

## 3. PyTorch 完整实现

### 3.1 MMOE 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExpertNetwork(nn.Module):
    """
    专家网络

    一个简单的 MLP
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


class GateNetwork(nn.Module):
    """
    门控网络

    输出专家的权重
    """

    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)


class TaskTower(nn.Module):
    """
    任务塔

    专门处理某个任务的 MLP
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class MMOE(nn.Module):
    """
    MMOE: Multi-gate Mixture-of-Experts

    论文: Modeling Task Relationships in Multi-task Learning with
          Multi-gate Mixture-of-Experts (KDD 2018)

    组成部分:
    1. Expert Networks: 多个专家网络
    2. Gate Networks: 每个任务一个门控网络
    3. Task Towers: 每个任务一个塔网络
    """

    def __init__(self, feature_configs, embed_dim=16,
                 num_experts=8, expert_hidden_dim=128, expert_output_dim=64,
                 tower_hidden_dim=64, num_tasks=2, task_names=None,
                 dropout=0.1):
        """
        参数:
            feature_configs: dict, 特征配置
            embed_dim: 嵌入维度
            num_experts: 专家数量
            expert_hidden_dim: 专家隐藏层维度
            expert_output_dim: 专家输出维度
            tower_hidden_dim: 塔隐藏层维度
            num_tasks: 任务数量
            task_names: 任务名称列表
            dropout: Dropout 比例
        """
        super().__init__()

        self.feature_configs = feature_configs
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]

        # ========== Embedding 层 ==========
        self.embeddings = nn.ModuleDict()

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(
                    config['vocab_size'],
                    config.get('embed_dim', embed_dim)
                )

        # 计算输入维度
        self._compute_input_dim(feature_configs, embed_dim)

        # ========== 专家网络 ==========
        self.experts = nn.ModuleList([
            ExpertNetwork(self.input_dim, expert_hidden_dim, expert_output_dim, dropout)
            for _ in range(num_experts)
        ])

        # ========== 门控网络（每个任务一个）==========
        self.gates = nn.ModuleList([
            GateNetwork(self.input_dim, num_experts)
            for _ in range(num_tasks)
        ])

        # ========== 任务塔（每个任务一个）==========
        self.towers = nn.ModuleList([
            TaskTower(expert_output_dim, tower_hidden_dim, 1, dropout)
            for _ in range(num_tasks)
        ])

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
            outputs: dict, 每个任务的输出
        """
        # ========== Embedding ==========
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

        x = torch.cat(embeddings, dim=-1)  # (batch, input_dim)

        # ========== 专家输出 ==========
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1
        )  # (batch, num_experts, expert_output_dim)

        # ========== 每个任务的处理 ==========
        outputs = {}

        for task_idx, task_name in enumerate(self.task_names):
            # 门控权重
            gate_weights = self.gates[task_idx](x)  # (batch, num_experts)

            # 加权组合专家输出
            # (batch, num_experts) @ (batch, num_experts, output_dim) -> (batch, output_dim)
            weighted_expert = torch.einsum('bn,bnd->bd', gate_weights, expert_outputs)

            # 任务塔
            tower_output = self.towers[task_idx](weighted_expert)  # (batch, 1)

            # Sigmoid
            outputs[task_name] = torch.sigmoid(tower_output)

        return outputs

    def compute_loss(self, outputs, labels, task_weights=None):
        """
        计算多任务损失

        参数:
            outputs: dict, 模型输出
            labels: dict, 标签
            task_weights: dict, 任务权重

        返回:
            total_loss: 总损失
            task_losses: dict, 每个任务的损失
        """
        if task_weights is None:
            task_weights = {name: 1.0 for name in self.task_names}

        task_losses = {}
        total_loss = 0

        for task_name in self.task_names:
            if task_name in labels:
                pred = outputs[task_name].squeeze()
                target = labels[task_name]

                # BCE 损失
                loss = F.binary_cross_entropy(pred, target.float())
                task_losses[task_name] = loss

                total_loss += task_weights[task_name] * loss

        return total_loss, task_losses


class PLE(nn.Module):
    """
    PLE: Progressive Layered Extraction

    论文: Progressive Layered Extraction (PLE): A Novel Multi-Task Learning
          (TML) Model for Personalized Recommendations (RecSys 2020)

    MMOE 的改进版本，增加了任务专属专家
    """

    def __init__(self, feature_configs, embed_dim=16,
                 num_shared_experts=4, num_task_experts=2,
                 expert_hidden_dim=128, expert_output_dim=64,
                 tower_hidden_dim=64, num_tasks=2, task_names=None,
                 dropout=0.1):
        super().__init__()

        self.feature_configs = feature_configs
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]

        # Embedding 层
        self.embeddings = nn.ModuleDict()
        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(
                    config['vocab_size'],
                    config.get('embed_dim', embed_dim)
                )

        self._compute_input_dim(feature_configs, embed_dim)

        # 共享专家
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(self.input_dim, expert_hidden_dim, expert_output_dim, dropout)
            for _ in range(num_shared_experts)
        ])

        # 任务专属专家
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertNetwork(self.input_dim, expert_hidden_dim, expert_output_dim, dropout)
                for _ in range(num_task_experts)
            ])
            for _ in range(num_tasks)
        ])

        # 门控网络（每个任务）
        # 输入：共享专家 + 任务专属专家
        total_experts = num_shared_experts + num_task_experts
        self.gates = nn.ModuleList([
            GateNetwork(self.input_dim, total_experts)
            for _ in range(num_tasks)
        ])

        # 任务塔
        self.towers = nn.ModuleList([
            TaskTower(expert_output_dim, tower_hidden_dim, 1, dropout)
            for _ in range(num_tasks)
        ])

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

        # 共享专家输出
        shared_outputs = torch.stack(
            [expert(x) for expert in self.shared_experts],
            dim=1
        )  # (batch, num_shared, dim)

        # 每个任务
        outputs = {}

        for task_idx, task_name in enumerate(self.task_names):
            # 任务专属专家输出
            task_outputs = torch.stack(
                [expert(x) for expert in self.task_experts[task_idx]],
                dim=1
            )  # (batch, num_task, dim)

            # 拼接共享和任务专属
            all_expert_outputs = torch.cat([shared_outputs, task_outputs], dim=1)

            # 门控
            gate_weights = self.gates[task_idx](x)  # (batch, total_experts)

            # 加权组合
            weighted_expert = torch.einsum('bn,bnd->bd', gate_weights, all_expert_outputs)

            # 塔
            tower_output = self.towers[task_idx](weighted_expert)

            outputs[task_name] = torch.sigmoid(tower_output)

        return outputs

    def compute_loss(self, outputs, labels, task_weights=None):
        if task_weights is None:
            task_weights = {name: 1.0 for name in self.task_names}

        task_losses = {}
        total_loss = 0

        for task_name in self.task_names:
            if task_name in labels:
                pred = outputs[task_name].squeeze()
                target = labels[task_name]
                loss = F.binary_cross_entropy(pred, target.float())
                task_losses[task_name] = loss
                total_loss += task_weights[task_name] * loss

        return total_loss, task_losses


# 使用示例
if __name__ == "__main__":
    # 特征配置
    feature_configs = {
        'cat1': {'type': 'categorical', 'vocab_size': 100},
        'cat2': {'type': 'categorical', 'vocab_size': 200},
        'num1': {'type': 'numerical'},
    }

    # 创建 MMOE 模型
    model = MMOE(
        feature_configs=feature_configs,
        embed_dim=16,
        num_experts=8,
        expert_hidden_dim=64,
        expert_output_dim=32,
        tower_hidden_dim=32,
        num_tasks=2,
        task_names=['ctr', 'cvr']
    )

    # 模拟输入
    batch_size = 32
    features = {
        'cat1': torch.randint(0, 100, (batch_size,)),
        'cat2': torch.randint(0, 200, (batch_size,)),
        'num1': torch.randn(batch_size),
    }

    labels = {
        'ctr': torch.randint(0, 2, (batch_size,)).float(),
        'cvr': torch.randint(0, 2, (batch_size,)).float(),
    }

    # 前向传播
    outputs = model(features)
    print("MMOE 输出:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")

    # 计算损失
    total_loss, task_losses = model.compute_loss(outputs, labels)
    print(f"\n总损失: {total_loss:.4f}")
    for task, loss in task_losses.items():
        print(f"  {task} 损失: {loss:.4f}")

    # PLE 模型
    print("\n" + "="*50)
    print("PLE 模型")

    ple_model = PLE(
        feature_configs=feature_configs,
        embed_dim=16,
        num_shared_experts=4,
        num_task_experts=2,
        expert_hidden_dim=64,
        expert_output_dim=32,
        tower_hidden_dim=32,
        num_tasks=2,
        task_names=['ctr', 'cvr']
    )

    ple_outputs = ple_model(features)
    print("PLE 输出:")
    for task, output in ple_outputs.items():
        print(f"  {task}: {output.shape}")
```

### 3.2 训练示例

```python
from torch.utils.data import Dataset, DataLoader


class MultiTaskDataset(Dataset):
    """多任务数据集"""

    def __init__(self, data, feature_configs):
        self.data = data
        self.feature_configs = feature_configs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        features = {}
        labels = {}

        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical':
                features[name] = torch.tensor(row[name])
            elif config['type'] == 'numerical':
                features[name] = torch.tensor(row[name], dtype=torch.float)

        # 多任务标签
        labels['ctr'] = torch.tensor(row['click'])
        labels['cvr'] = torch.tensor(row['convert'])

        return features, labels


def train_mmoe():
    """训练 MMOE"""
    # 配置
    config = {
        'n_samples': 10000,
        'batch_size': 256,
        'epochs': 10,
        'learning_rate': 0.001,
    }

    # 生成模拟数据
    data = []
    for i in range(config['n_samples']):
        row = {
            'cat1': np.random.randint(0, 100),
            'cat2': np.random.randint(0, 200),
            'num1': np.random.randn(),
            'click': np.random.randint(0, 2),
            'convert': np.random.randint(0, 2),
        }
        data.append(row)

    feature_configs = {
        'cat1': {'type': 'categorical', 'vocab_size': 100},
        'cat2': {'type': 'categorical', 'vocab_size': 200},
        'num1': {'type': 'numerical'},
    }

    # 数据集
    dataset = MultiTaskDataset(data, feature_configs)

    def collate_fn(batch):
        features_list, labels_list = zip(*batch)
        features = {}
        for key in features_list[0]:
            features[key] = torch.stack([f[key] for f in features_list])

        labels = {}
        for key in labels_list[0]:
            labels[key] = torch.stack([l[key] for l in labels_list])

        return features, labels

    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                           shuffle=True, collate_fn=collate_fn)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MMOE(
        feature_configs=feature_configs,
        embed_dim=16,
        num_experts=8,
        expert_hidden_dim=64,
        expert_output_dim=32,
        tower_hidden_dim=32,
        num_tasks=2,
        task_names=['ctr', 'cvr']
    ).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 训练
    for epoch in range(config['epochs']):
        model.train()
        total_loss_epoch = 0
        task_loss_epoch = {'ctr': 0, 'cvr': 0}

        for features, labels in dataloader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()

            outputs = model(features)
            total_loss, task_losses = model.compute_loss(outputs, labels)

            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()
            for task, loss in task_losses.items():
                task_loss_epoch[task] += loss.item()

        n_batches = len(dataloader)
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"  Total Loss: {total_loss_epoch / n_batches:.4f}")
        print(f"  CTR Loss: {task_loss_epoch['ctr'] / n_batches:.4f}")
        print(f"  CVR Loss: {task_loss_epoch['cvr'] / n_batches:.4f}")

    return model


if __name__ == "__main__":
    model = train_mmoe()
    print("MMOE 训练完成！")
```

## 4. MMOE vs PLE vs Shared Bottom

### 4.1 架构对比

| 模型 | 共享方式 | 灵活性 | 参数量 |
|------|----------|--------|--------|
| Shared Bottom | 全共享 | 低 | 最小 |
| MMOE | 专家共享 | 中 | 中等 |
| PLE | 部分+专属 | 高 | 较大 |

### 4.2 适用场景

**Shared Bottom：**
- 任务高度相关
- 数据量少

**MMOE：**
- 任务有一定相关性
- 需要一定灵活性

**PLE：**
- 任务相关性较弱
- 需要高灵活性

## 5. 调参建议

### 5.1 模型参数

| 参数 | 推荐值 |
|------|--------|
| num_experts | 4-16 |
| expert_hidden_dim | 128-256 |
| tower_hidden_dim | 64-128 |

### 5.2 训练参数

| 参数 | 推荐值 |
|------|--------|
| learning_rate | 0.001 |
| batch_size | 256-1024 |
| task_weights | 根据任务重要性调整 |

## 6. 学习总结

### 6.1 核心要点

1. **多任务学习**：同时优化多个目标
2. **专家混合**：灵活组合表示
3. **门控机制**：任务特定的专家选择

### 6.2 关键设计

- 多个专家网络学习不同表示
- 门控网络为每个任务选择专家
- 任务塔处理特定任务

## 7. 练习题

1. 实现一个三任务的 MMOE 模型（CTR、CVR、留存）。

2. 比较不同专家数量对模型效果的影响。

3. 实现 MMOE 的梯度分析，观察任务间的干扰。
