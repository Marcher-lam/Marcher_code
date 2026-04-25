# PLE (Progressive Layered Extraction) 学习文档

## 1. 算法基础认知

PLE 是腾讯在 2020 年提出的递进式多层多任务学习模型，发表于 RecSys 2020 并获 Best Paper。它针对 MMoE 的两个问题进行了改进：（1）MMoE 所有专家对所有任务开放，无法区分共享和独占知识；（2）单层路由表达能力有限。

PLE 的核心思想是**显式分离共享专家和任务专属专家**，并通过**多层递进路由**（Progressive Routing）逐层提炼任务相关知识。CGC（Customized Gate Control）模块是基础单元，多个 CGC 层堆叠形成完整的 PLE 架构。

## 2. 核心原理

PLE 的架构包含两个层次：

**CGC（Customized Gate Control）模块**：
- **共享专家** $E_s$：所有任务共同使用，捕捉跨任务共享知识
- **任务专属专家** $E_i$：仅对应任务使用，捕捉任务独有知识
- **门控**：每个任务的 Gate 从其专属专家 + 共享专家中选择

**多层递进**：
- 多个 CGC 模块堆叠，上一层的输出作为下一层的 Expert 输入
- 信息逐层提炼：浅层捕捉基础模式，深层捕捉高层抽象
- 每一层都重新进行门控路由，实现渐进式知识分离

## 3. 数学公式与推导

**CGC 模块（单层）**：

共享专家输出：$\{e_{s,1}(x), ..., e_{s,M}(x)\}$

任务 $i$ 专属专家输出：$\{e_{i,1}(x), ..., e_{i,N}(x)\}$

任务 $i$ 的门控选择范围：

$$S_i = \{E_s\} \cup \{E_i\}$$

门控权重：

$$g_i(x) = \text{softmax}(W_{g,i} \cdot x), \quad g_i \in \mathbb{R}^{M+N}$$

门控加权输出：

$$c_i(x) = \sum_{k \in S_i} g_i^k(x) \cdot e_k(x)$$

**多层递进公式**：

$$c_i^{(l)} = \text{CGC}_i^{(l)}(c_1^{(l-1)}, ..., c_T^{(l-1)}, e_{s,1}^{(l-1)}, ..., e_{s,M}^{(l-1)})$$

**任务预测**：

$$\hat{y}_i = \text{Tower}_i(c_i^{(L)})$$

其中 $L$ 是 CGC 层数。

## 4. 训练过程讲解

1. **输入 Embedding**：稀疏特征映射为稠密向量
2. **第一层 CGC**：共享专家和任务专属专家分别处理输入，各任务 Gate 选择加权
3. **多层递进**：上一层的加权输出和共享专家输出作为下一层的输入
4. **Tower 预测**：最后一层 CGC 输出送入各任务 Tower 得到预测
5. **联合优化**：各任务损失加权求和，反向传播更新全部参数

## 5. 应用场景

- **视频推荐多目标排序**：同时预估点击、点赞、收藏、转发、评论等多个目标
- **广告多目标优化**：CTR + CVR + 用户停留时长联合预估
- **内容分发**：阅读率 + 完读率 + 互动率多任务学习
- **腾讯视频推荐**：PLE 的原始应用场景，线上效果显著

## 6. 优缺点分析

**优点**：
- 显式分离共享/独占知识，比 MMoE 更好地避免负迁移
- 多层递进路由提升表达能力
- 任务专属专家保证每个任务有独立参数空间
- 在腾讯线上 A/B 测试中取得显著提升

**缺点**：
- 参数量随任务数和层数快速增长
- 超参数多（专家数、层数、每个任务的专属专家数）
- 任务数很多时，专属专家总数大，训练成本高
- 多层堆叠可能导致梯度传播困难

## 7. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class CGC(nn.Module):
    def __init__(self, input_dim, num_tasks, num_shared_experts, num_task_experts, expert_dim):
        super().__init__()
        self.shared_experts = nn.ModuleList(
            [Expert(input_dim, expert_dim) for _ in range(num_shared_experts)]
        )
        self.task_experts = nn.ModuleList([
            nn.ModuleList([Expert(input_dim, expert_dim) for _ in range(num_task_experts)])
            for _ in range(num_tasks)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_shared_experts + num_task_experts)
            for _ in range(num_tasks)
        ])
        self.num_tasks = num_tasks

    def forward(self, x):
        shared_outs = torch.stack([e(x) for e in self.shared_experts], dim=1)
        task_outputs = []
        for i in range(self.num_tasks):
            task_exp_outs = torch.stack([e(x) for e in self.task_experts[i]], dim=1)
            all_experts = torch.cat([shared_outs, task_exp_outs], dim=1)
            gate_weights = torch.softmax(self.gates[i](x), dim=-1).unsqueeze(-1)
            weighted = torch.sum(gate_weights * all_experts, dim=1)
            task_outputs.append(weighted)
        return task_outputs


class PLE(nn.Module):
    def __init__(self, input_dim, num_tasks=2, num_shared_experts=2,
                 num_task_experts=1, expert_dim=32, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.cgc_layers = nn.ModuleList()
        for l in range(num_layers):
            dim = input_dim if l == 0 else expert_dim
            self.cgc_layers.append(
                CGC(dim, num_tasks, num_shared_experts, num_task_experts, expert_dim)
            )
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 16), nn.ReLU(), nn.Linear(16, 1))
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        layer_outputs = self.cgc_layers[0](x)
        for l in range(1, self.num_layers):
            stacked = torch.stack(layer_outputs, dim=1)
            combined = stacked.view(stacked.size(0), -1)
            layer_outputs = self.cgc_layers[l](combined)
        return [tower(out) for tower, out in zip(self.towers, layer_outputs)]


input_dim = 64
model = PLE(input_dim, num_tasks=2, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    x = torch.randn(128, input_dim)
    y1 = torch.randint(0, 2, (128, 1)).float()
    y2 = torch.randint(0, 2, (128, 1)).float()
    preds = model(x)
    loss1 = nn.functional.binary_cross_entropy(torch.sigmoid(preds[0]), y1)
    loss2 = nn.functional.binary_cross_entropy(torch.sigmoid(preds[1]), y2)
    loss = loss1 + loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
```

## 8. 手工代码实现

```python
import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def expert_forward(x, W1, b1, W2, b2):
    h = relu(W1 @ x + b1)
    return relu(W2 @ h + b2)


def cgc_forward(x, shared_params, task_expert_params, gate_params):
    num_tasks = len(task_expert_params)
    shared_outs = []
    for W1, b1, W2, b2 in shared_params:
        shared_outs.append(expert_forward(x, W1, b1, W2, b2))

    task_outputs = []
    for i in range(num_tasks):
        task_outs = []
        for W1, b1, W2, b2 in task_expert_params[i]:
            task_outs.append(expert_forward(x, W1, b1, W2, b2))
        all_outs = np.array(shared_outs + task_outs)
        gW, gb = gate_params[i]
        gate_w = softmax(gW @ x + gb)
        weighted = np.sum(gate_w[:, None] * all_outs, axis=0)
        task_outputs.append(weighted)
    return task_outputs


np.random.seed(42)
input_dim = 16
expert_dim = 8
num_shared = 2
num_task_exp = 1
num_tasks = 2

shared_params = [
    (np.random.randn(expert_dim, input_dim) * 0.1, np.zeros(expert_dim),
     np.random.randn(expert_dim, expert_dim) * 0.1, np.zeros(expert_dim))
    for _ in range(num_shared)
]

task_expert_params = [
    [(np.random.randn(expert_dim, input_dim) * 0.1, np.zeros(expert_dim),
      np.random.randn(expert_dim, expert_dim) * 0.1, np.zeros(expert_dim))]
    for _ in range(num_tasks)
]

total_experts = num_shared + num_task_exp
gate_params = [
    (np.random.randn(total_experts, input_dim) * 0.1, np.zeros(total_experts))
    for _ in range(num_tasks)
]

x = np.random.randn(input_dim)
outputs = cgc_forward(x, shared_params, task_expert_params, gate_params)
for i, out in enumerate(outputs):
    print(f"任务{i + 1} CGC输出: {np.round(out, 3)}")
```

## 9. 可视化与结果理解

- **门控权重对比**：PLE 的任务专属专家只被对应任务使用，共享专家被多个任务共同使用，可视化这种分工模式
- **多层信息流**：展示 CGC 各层输出如何逐层提炼任务特征
- **与 MMoE 的门控分布对比**：MMoE 门控分布可能混乱，PLE 门控更清晰地分为共享/专属

效果对比（腾讯视频推荐数据集）：

| 模型 | 任务A AUC | 任务B AUC | 任务C AUC |
|------|-----------|-----------|-----------|
| Shared-Bottom | 0.7512 | 0.7823 | 0.7034 |
| MMoE | 0.7598 | 0.7891 | 0.7142 |
| PLE | **0.7656** | **0.7948** | **0.7231** |

## 10. 模型评估

- **各任务独立 AUC**：每个任务单独评估排序能力
- **帕累托分析**：对比基线模型，PLE 应在各任务上均不下降
- **线上 A/B 测试**：关注各目标指标的同步提升
- **专家利用率分析**：统计各专家的平均激活权重，确保无冗余

## 11. 常见问题与易错点

- **专属专家数量**：每个任务通常 1-2 个专属专家，太多会增加参数冗余
- **CGC 层数**：通常 2-3 层，过深收益递减且训练困难
- **门控输入**：门控可以用原始输入或上一层输出，不同选择影响路由质量
- **任务间权重平衡**：不同任务的损失量级差异大时需要仔细调节权重

## 12. 学习总结

PLE 的核心贡献是显式区分共享专家和任务专属专家，配合多层递进路由，实现了更精细的多任务参数共享控制。相比 MMoE 的全共享专家，PLE 让每个任务保有自己的"私有空间"，从根本上缓解了负迁移问题。其递进式多层结构也提供了更强的表达能力。

## 13. 练习题与思考题（含答案）

**Q1：PLE 相比 MMoE 的核心改进是什么？**

A1：MMoE 的所有专家对所有任务开放，无法区分哪些知识应该共享、哪些应该独占。PLE 将专家分为共享专家和任务专属专家，任务只能通过 Gate 选择共享专家和自己的专属专家，不能访问其他任务的专属专家，从架构层面保证了任务独占知识的隔离。

**Q2：为什么任务专属专家有助于缓解负迁移？**

A2：当多个任务目标冲突时（如点击率 vs 停留时长，一个鼓励标题党，一个鼓励深度内容），共享参数会被拉向不同方向。任务专属专家为每个任务提供独立的参数空间，可以学习任务独有的模式，不受其他任务干扰，从根本上减少了任务间的参数冲突。

**Q3：多层 CGC 堆叠的作用是什么？**

A3：单层 CGC 的路由是一次性的，表达能力有限。多层 CGC 逐层提炼：第一层捕捉基础特征模式，后续层在前一层输出的更高层表示上再次路由，逐步分离出更精纯的任务特征。类似于深度网络逐层提取抽象特征的思想。

## 14. 学习路径建议

```
多任务学习基础（Shared-Bottom）
        ↓
MMoE（多门控混合专家）
        ↓
PLE（递进式分层提取）  ← 你在这里
        ↓
ESMM / ESM2（CVR 预估专用多任务）
        ↓
PEP / PLE 变体（更多路由策略）
        ↓
多任务学习在推荐系统中的工业实践
```
