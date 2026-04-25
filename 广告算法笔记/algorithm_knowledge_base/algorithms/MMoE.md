# MMoE (Multi-gate Mixture-of-Experts) 学习文档

## 1. 算法基础认知

MMoE 是 Google 在 2018 年提出的多人多任务学习模型，发表于 KDD 2018。它解决了多任务学习中**负迁移**的问题——当多个任务相关性弱时，共享参数反而会互相干扰，导致所有任务效果下降。

传统 Shared-Bottom 架构将所有任务共享底层网络，任务差异只能通过各自的 Tower 层区分。MMoE 引入多专家网络（Experts）和任务特定的门控（Gates），每个任务通过门控软性选择不同专家的组合，实现参数共享与任务差异化的平衡。

## 2. 核心原理

MMoE 的架构由三个部分组成：

- **Expert 网络**：$K$ 个并行的专家网络 $e_1(x), e_2(x), ..., e_K(x)$，每个是一个全连接网络
- **Gate 网络**：每个任务有一个专属门控 $g_i(x) = \text{softmax}(W_g \cdot x)$，输出对 $K$ 个专家的权重
- **Task Tower**：每个任务将门控加权后的专家输出送入各自的 Tower 网络得到最终预测

关键思想：不同任务通过门控关注不同专家的输出，任务相关时门控权重相似（接近共享），任务无关时门控权重差异大（接近独立），自适应调节共享程度。

## 3. 数学公式与推导

**Expert 输出**：

$$e_k(x) = h_k(x), \quad k = 1, 2, ..., K$$

**Gate（任务 $i$）**：

$$g_i(x) = \text{softmax}(W_{g,i} \cdot x), \quad g_i \in \mathbb{R}^K$$

**门控加权输出（任务 $i$）**：

$$f_i(x) = \sum_{k=1}^{K} g_i^k(x) \cdot e_k(x)$$

其中 $g_i^k(x)$ 是门控输出的第 $k$ 个权重。

**任务预测（任务 $i$）**：

$$\hat{y}_i = \text{Tower}_i(f_i(x))$$

**联合损失**：

$$\mathcal{L} = \sum_{i=1}^{n} w_i \cdot \mathcal{L}_i$$

其中 $w_i$ 是任务权重超参数，$\mathcal{L}_i$ 是任务 $i$ 的损失函数。

## 4. 训练过程讲解

1. **输入处理**：共享特征经过 Embedding 层得到稠密表示
2. **Expert 前向**：$K$ 个专家网络并行处理输入，各自输出中间表示
3. **Gate 前向**：每个任务的 Gate 网络接收输入，输出 softmax 权重
4. **加权融合**：每个任务用各自的 Gate 权重对 Expert 输出做加权求和
5. **Tower 预测**：加权结果送入各任务的 Tower 网络输出预测
6. **联合优化**：各任务损失加权求和，反向传播更新所有参数

## 5. 应用场景

- **广告多目标排序**：同时预估 CTR、CVR、用户停留时长、互动率等
- **推荐系统多任务**：点击 + 点赞 + 收藏 + 分享等多目标优化
- **信息流排序**：点击率 + 阅读完成率 + 评论率联合优化
- **多领域学习**：不同业务场景共享特征但目标不同

## 6. 优缺点分析

**优点**：
- 自适应共享：任务相关时自动共享参数，无关时自动隔离
- 门控机制可解释：可视化 Gate 权重可分析任务间关系
- 相比 Shared-Bottom，任务差异大时效果提升明显
- 工程实现简洁，易于扩展到更多任务

**缺点**：
- 专家数量和结构需要调参
- 门控网络参数量随任务数线性增长
- 当任务数量很多时，Gate 权重可能过于稀疏，部分专家未被充分利用
- 没有显式区分任务共享和任务独占的专家

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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Gate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class MMoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, num_tasks=2, expert_dim=32):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, expert_dim) for _ in range(num_experts)])
        self.gates = nn.ModuleList([Gate(input_dim, num_experts) for _ in range(num_tasks)])
        self.towers = nn.ModuleList([Tower(expert_dim) for _ in range(num_tasks)])

    def forward(self, x):
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        task_outputs = []
        for gate, tower in zip(self.gates, self.towers):
            gate_weights = gate(x).unsqueeze(-1)
            weighted = torch.sum(gate_weights * expert_outputs, dim=1)
            task_outputs.append(tower(weighted))
        return task_outputs


input_dim = 64
model = MMoE(input_dim, num_experts=4, num_tasks=2)
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def mmoe_forward(x, expert_weights, gate_weights, tower_weights):
    num_experts = len(expert_weights)
    expert_outputs = []
    for W1, b1, W2, b2 in expert_weights:
        h = relu(W1 @ x + b1)
        e = relu(W2 @ h + b2)
        expert_outputs.append(e)
    expert_stack = np.stack(expert_outputs, axis=0)

    task_preds = []
    for gw_W, gw_b, tw_W, tw_b in gate_weights:
        gate_logits = gw_W @ x + gw_b
        gate_probs = softmax(gate_logits)
        weighted = np.sum(gate_probs[:, None] * expert_stack, axis=0)
        tower_W1, tower_b1, tower_W2, tower_b2 = tower_weights
        h = relu(tower_W1 @ weighted + tower_b1)
        pred = sigmoid((tower_W2 @ h + tower_b2).item())
        task_preds.append(pred)
    return task_preds, gate_probs


np.random.seed(42)
input_dim = 16
expert_dim = 8
K = 3
num_tasks = 2

expert_weights = []
for _ in range(K):
    W1 = np.random.randn(expert_dim, input_dim) * 0.1
    b1 = np.zeros(expert_dim)
    W2 = np.random.randn(expert_dim, expert_dim) * 0.1
    b2 = np.zeros(expert_dim)
    expert_weights.append((W1, b1, W2, b2))

gate_weights = []
tower_weights_list = []
for _ in range(num_tasks):
    gW = np.random.randn(K, input_dim) * 0.1
    gb = np.zeros(K)
    gate_weights.append((gW, gb))
    tW1 = np.random.randn(16, expert_dim) * 0.1
    tb1 = np.zeros(16)
    tW2 = np.random.randn(1, 16) * 0.1
    tb2 = np.zeros(1)
    tower_weights_list.append((tW1, tb1, tW2, tb2))

x = np.random.randn(input_dim)
preds, gate_probs = mmoe_forward(x, expert_weights, gate_weights, tower_weights_list)
print(f"任务1预测: {preds[0]:.4f}, 任务2预测: {preds[1]:.4f}")
print(f"任务1 Gate权重: {np.round(gate_probs, 3)}")
```

## 9. 可视化与结果理解

- **Gate 权重热力图**：可视化不同任务对各专家的权重分布，任务相关时权重相似，无关时差异大
- **专家利用率**：统计每个专家被各任务的平均使用权重，识别"闲置"专家
- **任务相关性 vs 性能提升**：在人工构造的不同任务相关性数据集上，对比 MMoE 与 Shared-Bottom

效果对比（Google 内部数据集）：

| 模型 | 任务A AUC | 任务B AUC |
|------|-----------|-----------|
| Shared-Bottom | 0.7521 | 0.7834 |
| One-gate MoE | 0.7598 | 0.7871 |
| MMoE | **0.7634** | **0.7912** |

## 10. 模型评估

- **各任务独立指标**：CTR 用 AUC，CVR 用 AUC，时长用 MAE 等
- **联合指标**：多任务加权指标，权重根据业务优先级设定
- **帕累托最优分析**：对比 Shared-Bottom，MMoE 在各任务上应同时不下降

## 11. 常见问题与易错点

- **专家数量选择**：通常 4-8 个，太少无法表达任务差异，太多参数冗余
- **任务损失权重**：不同任务损失量级可能差很大，需要调节 $w_i$ 平衡训练
- **门控退化**：训练后期 Gate 权重可能趋近 one-hot，退化为路由选择而非软加权
- **负迁移检测**：如果某个任务效果下降，说明共享专家存在冲突，考虑 PLE 的任务专属专家

## 12. 学习总结

MMoE 的核心贡献是提出了多门控混合专家机制，让多任务学习能够自适应调节参数共享程度。门控权重随输入动态变化，不同样本、不同任务关注不同专家的输出，实现了灵活的共享-隔离平衡。它是多任务学习领域的里程碑工作。

## 13. 练习题与思考题（含答案）

**Q1：MMoE 相比 Shared-Bottom 的核心优势是什么？**

A1：Shared-Bottom 强制所有任务共享底层网络，任务差异大时产生负迁移。MMoE 通过门控机制让每个任务软性选择专家组合，任务相关时门控权重趋同（等效共享），任务无关时门控权重分化（等效独立），自适应调节共享程度。

**Q2：为什么门控机制能缓解负迁移？**

A2：门控为每个任务学习独立的专家权重分配。当两个任务冲突时，Gate 会给它们分配不同的专家组合，减少参数共享带来的干扰。这种软路由机制比 Shared-Bottom 的硬共享更灵活。

**Q3：MMoE 与 PLE 的区别是什么？**

A3：MMoE 的所有专家对所有任务共享，没有区分。PLE 将专家分为任务专属专家和共享专家，任务只能通过 Gate 选择自己的专属专家和共享专家，不能访问其他任务的专属专家，进一步隔离任务冲突。

## 14. 学习路径建议

```
多任务学习基础（Shared-Bottom 架构）
        ↓
MMoE（多门控混合专家）  ← 你在这里
        ↓
PLE（Progressive Layered Extraction）
        ↓
ESMM / ESM2（CVR 预估专用多任务）
        ↓
SNR（Sub-Network Routing）
```
