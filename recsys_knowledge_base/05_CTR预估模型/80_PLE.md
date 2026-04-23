# PLE (Progressive Layered Extraction) 学习文档

## 1. 算法基础认知

### 1.1 什么是 PLE？

PLE（Progressive Layered Extraction）是腾讯提出的改进版 MMOE，通过显式分离任务共享和任务特定组件来解决多任务学习中的**负迁移**问题。

### 1.2 与 MMOE 的区别

| 特性 | MMOE | PLE |
|------|------|-----|
| Expert 类型 | 全部共享 | 共享 + 任务特定 |
| Expert 交互 | 无显式分离 | 分层渐进提取 |
| 负迁移 | 可能存在 | 显式减少 |

### 1.3 核心思想

```
MMOE: 所有 Expert 对所有任务可见
PLE:  将 Expert 分为共享和任务特定两类

任务 A → Expert A (专用) + Expert Shared (共享)
任务 B → Expert B (专用) + Expert Shared (共享)
```

## 2. 核心原理

### 2.1 架构设计

```
PLE 结构:
┌─────────────────────────────────────────┐
│              Gate A     Gate B          │
│                 ↓         ↓              │
│  ┌─────────┬─────────┬─────────┐       │
│  │Expert A │Expert B │Expert S │       │
│  │ (任务A) │ (任务B) │ (共享)  │       │
│  └─────────┴─────────┴─────────┘       │
│              CGC (Customized Gate)      │
└─────────────────────────────────────────┘

多层 PLE = 多个 CGC 层堆叠
```

### 2.2 CGC (Customized Gate Control)

对于任务 k:

$$g^k(x) = \sum_{i=1}^{M_k} w_k^i(x) \cdot E_k^i(x) + \sum_{j=1}^{M_s} w_s^j(x) \cdot S^j(x)$$

其中:
- $E_k^i$: 任务 k 的第 i 个专用 Expert
- $S^j$: 第 j 个共享 Expert
- $w_k^i, w_s^j$: Gate 输出的权重

## 3. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class ExpertLayer(nn.Module):
    """
    Expert 层
    """

    def __init__(self, input_dim: int, expert_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, expert_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CGCLayer(nn.Module):
    """
    Customized Gate Control Layer
    """

    def __init__(self, input_dim: int, expert_dim: int,
                 n_shared_experts: int = 2,
                 n_task_experts: Dict[str, int] = None):
        """
        参数:
            input_dim: 输入维度
            expert_dim: Expert 输出维度
            n_shared_experts: 共享 Expert 数量
            n_task_experts: {task_name: num_experts}
        """
        super().__init__()

        self.expert_dim = expert_dim
        self.n_shared_experts = n_shared_experts
        self.n_task_experts = n_task_experts or {}
        self.task_names = list(self.n_task_experts.keys())

        # 共享 Experts
        self.shared_experts = nn.ModuleList([
            ExpertLayer(input_dim, expert_dim)
            for _ in range(n_shared_experts)
        ])

        # 任务特定 Experts
        self.task_experts = nn.ModuleDict({
            task: nn.ModuleList([
                ExpertLayer(input_dim, expert_dim)
                for _ in range(n)
            ])
            for task, n in self.n_task_experts.items()
        })

        # 任务 Gates
        self.gates = nn.ModuleDict({
            task: nn.Linear(input_dim, n_shared_experts + self.n_task_experts[task])
            for task in self.task_names
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        返回:
            {task_name: expert_output}
        """
        # 计算所有共享 Expert 输出
        shared_outputs = torch.stack(
            [expert(x) for expert in self.shared_experts],
            dim=1
        )  # (batch, n_shared, expert_dim)

        # 对每个任务计算输出
        task_outputs = {}

        for task in self.task_names:
            # 计算该任务的专用 Expert 输出
            task_expert_outputs = torch.stack(
                [expert(x) for expert in self.task_experts[task]],
                dim=1
            )  # (batch, n_task_experts, expert_dim)

            # 拼接共享和专用 Expert
            all_expert_outputs = torch.cat(
                [shared_outputs, task_expert_outputs],
                dim=1
            )  # (batch, n_shared + n_task_experts, expert_dim)

            # Gate 权重
            gate_weights = F.softmax(
                self.gates[task](x),
                dim=-1
            )  # (batch, n_shared + n_task_experts)

            # 加权组合
            task_output = torch.einsum(
                'be,bef->bf',
                gate_weights,
                all_expert_outputs
            )  # (batch, expert_dim)

            task_outputs[task] = task_output

        return task_outputs


class PLE(nn.Module):
    """
    Progressive Layered Extraction
    """

    def __init__(self, field_dims: List[int],
                 embed_dim: int = 10,
                 expert_dim: int = 64,
                 n_shared_experts: int = 2,
                 n_task_experts: Dict[str, int] = None,
                 n_cgc_layers: int = 2,
                 tower_dims: List[int] = [64, 32]):
        """
        参数:
            field_dims: 各域特征数量
            embed_dim: 嵌入维度
            expert_dim: Expert 维度
            n_shared_experts: 共享 Expert 数量
            n_task_experts: {task_name: num_experts}
            n_cgc_layers: CGC 层数
            tower_dims: Tower 网络维度
        """
        super().__init__()

        self.num_fields = len(field_dims)
        self.task_names = list(n_task_experts.keys()) if n_task_experts else ['task_a', 'task_b']

        # 默认每个任务 2 个专用 Expert
        if n_task_experts is None:
            n_task_experts = {task: 2 for task in self.task_names}

        # 嵌入层
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.field_offsets = torch.tensor([0] + np.cumsum(field_dims)[:-1].tolist())

        # CGC 层
        input_dim = embed_dim * len(field_dims)

        self.cgc_layers = nn.ModuleList()
        for i in range(n_cgc_layers):
            # 第一层输入是嵌入，后续层输入是 expert_dim
            layer_input_dim = input_dim if i == 0 else expert_dim
            self.cgc_layers.append(
                CGCLayer(layer_input_dim, expert_dim, n_shared_experts, n_task_experts)
            )

        # Tower 层
        self.towers = nn.ModuleDict({
            task: self._build_tower(expert_dim, tower_dims)
            for task in self.task_names
        })

    def _build_tower(self, input_dim: int, tower_dims: List[int]) -> nn.Sequential:
        """构建 Tower"""
        layers = []
        prev_dim = input_dim

        for dim in tower_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            X: (batch, num_fields) 特征索引

        返回:
            {task_name: logits}
        """
        # 嵌入
        X_offset = X + self.field_offsets.to(X.device)
        embeds = self.embedding(X_offset).view(X.size(0), -1)

        # CGC 层
        x = embeds
        for cgc_layer in self.cgc_layers:
            task_outputs = cgc_layer(x)
            # 取最后一个 CGC 层的输出
            x = task_outputs  # 保存所有任务输出

        # Tower 预测
        logits = {}
        for task in self.task_names:
            logits[task] = self.towers[task](task_outputs[task]).squeeze(-1)

        return logits

    def predict(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测概率"""
        logits = self.forward(X)
        return {task: torch.sigmoid(logit) for task, logit in logits.items()}


class PLETrainer:
    """
    PLE 训练器
    """

    def __init__(self, model: PLE, task_weights: Dict[str, float] = None,
                 learning_rate: float = 0.001):
        self.model = model
        self.task_weights = task_weights or {task: 1.0 for task in model.task_names}

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, X: torch.Tensor,
                   labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练一步

        参数:
            X: 特征
            labels: {task_name: label}
        """
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(X)

        # 计算各任务损失
        losses = {}
        total_loss = 0

        for task in self.model.task_names:
            if task in labels:
                loss = self.bce_loss(logits[task], labels[task].float())
                losses[f'{task}_loss'] = loss.item()
                total_loss += self.task_weights.get(task, 1.0) * loss

        losses['total_loss'] = total_loss.item()

        total_loss.backward()
        self.optimizer.step()

        return losses


def demo_ple():
    """PLE 示例"""
    # 配置
    field_dims = [100, 50, 20, 10]
    n_samples = 1000

    # 创建模型
    model = PLE(
        field_dims=field_dims,
        embed_dim=10,
        expert_dim=32,
        n_shared_experts=2,
        n_task_experts={'click': 2, 'conversion': 2},
        n_cgc_layers=2,
        tower_dims=[32, 16]
    )

    # 模拟数据
    X = torch.zeros(n_samples, len(field_dims), dtype=torch.long)
    for i, dim in enumerate(field_dims):
        X[:, i] = torch.randint(0, dim, (n_samples,))

    labels = {
        'click': (torch.rand(n_samples) > 0.7).float(),
        'conversion': (torch.rand(n_samples) > 0.9).float()
    }

    # 训练
    trainer = PLETrainer(model, task_weights={'click': 1.0, 'conversion': 2.0})

    for epoch in range(5):
        losses = trainer.train_step(X, labels)
        print(f"Epoch {epoch+1}: {losses}")

    # 预测
    preds = model.predict(X[:5])
    print("\n预测示例:")
    for task, probs in preds.items():
        print(f"{task}: {probs.detach().numpy()}")


if __name__ == "__main__":
    demo_ple()
```

## 4. PLE vs MMOE 对比

### 4.1 结构对比

```
MMOE:
┌─────────────────────────────┐
│ Expert1  Expert2  Expert3   │ ← 所有 Expert 共享
│    ↓        ↓        ↓      │
│  GateA   GateA   GateA      │
│  GateB   GateB   GateB      │
└─────────────────────────────┘

PLE:
┌─────────────────────────────┐
│ Expert_A1  Expert_B1  Expert_S1  │ ← 分离
│    ↓          ↓          ↓      │
│   GateA     GateB      (无)     │
└─────────────────────────────┘
```

### 4.2 性能对比

| 模型 | 参数量 | 负迁移 | 性能 |
|------|--------|--------|------|
| MMOE | 中 | 可能 | 基线 |
| PLE | 高 | 少 | 更好 |

## 5. 多层 PLE

### 5.1 渐进式提取

```python
class MultiLayerPLE(nn.Module):
    """
    多层 PLE

    每层的共享和任务特定特征渐进提取
    """

    def __init__(self, field_dims: List[int],
                 embed_dim: int = 10,
                 expert_dims: List[int] = [64, 64],
                 n_shared_experts: int = 2,
                 n_task_experts: Dict[str, int] = None):
        super().__init__()

        # ... 嵌入层

        # 多层 CGC
        self.cgc_layers = nn.ModuleList()

        for i, expert_dim in enumerate(expert_dims):
            input_dim = embed_dim * len(field_dims) if i == 0 else expert_dims[i-1]

            self.cgc_layers.append(
                CGCLayer(input_dim, expert_dim, n_shared_experts, n_task_experts)
            )

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 逐层提取
        x = self._get_embeddings(X)

        for cgc in self.cgc_layers:
            x = cgc(x)

        # Tower 预测
        # ...
```

## 6. 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| n_shared_experts | 2-4 | 共享 Expert 数量 |
| n_task_experts | 2-3 | 每任务 Expert 数量 |
| expert_dim | 64-128 | Expert 隐藏维度 |
| n_cgc_layers | 1-2 | CGC 层数 |
| task_weights | 按业务重要性 | 任务损失权重 |

## 7. 学习总结

### 7.1 核心要点

1. **Expert 分离**: 共享 Expert + 任务特定 Expert
2. **渐进提取**: 多层 CGC 逐步精炼特征
3. **减少负迁移**: 显式隔离任务特定组件

### 7.2 适用场景

- 多目标推荐（点击+转化+收藏）
- 相关性较弱的多任务
- 需要精细控制共享程度的场景

## 8. 练习题

1. 比较 PLE 和 MMOE 在不同任务相关性下的表现。

2. 实现一个 3 层 PLE 模型。

3. 分析各 Expert 的激活模式。
