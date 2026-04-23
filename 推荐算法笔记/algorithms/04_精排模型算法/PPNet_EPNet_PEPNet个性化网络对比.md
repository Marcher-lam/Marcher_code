# PPNet、EPNet、PEPNet 个性化网络对比

## 1. 概述

PPNet、EPNet、PEPNet 是快手提出的系列工作，通过**门控机制**实现模型参数的个性化/场景化调整，解决推荐系统中的多任务跷跷板和多场景跷跷板问题。

| 模型 | 解决问题 | 核心方法 | 发表 |
|------|---------|---------|------|
| PPNet | 多任务跷跷板 | Gate 生成动态权重作用于 DNN 层 | RecSys 2021 |
| EPNet | 多场景特征对齐 | Gate 调整 Embedding 表征 | RecSys 2022 |
| PEPNet | 多任务 + 多场景 | EPNet + PPNet 级联 | KDD 2023 |

## 2. 问题背景

### 2.1 多任务跷跷板效应

同时优化多个任务（如点击率、转化率、收藏率）时，一个任务指标提升往往导致另一个任务指标下降。原因是不同任务需要不同的模型参数，共享参数会产生冲突。

### 2.2 多场景跷跷板效应

在多个场景（如首页推荐、搜索推荐、购物车推荐）共享模型时，场景间数据量和分布差异导致大场景主导优化方向，小场景性能下降。

### 2.3 核心思路

不是为每个任务/场景训练独立模型（成本太高），而是**在共享模型基础上，用门控机制生成个性化的参数调制**，实现"一个模型，千面千参"。

## 3. PPNet（Parameter Personalized Network）

### 3.1 核心思想

PPNet 通过 Gate 网络根据用户/物品 ID 特征生成动态权重，调制 DNN 每一层的参数，使不同任务获得个性化的网络表征。

### 3.2 架构详解

```
         Task-specific Features (ID embedding)
                   |
              [Gate NU]
              /    |    \
           Gate_1 Gate_2 Gate_3
             |      |      |
  DNN Layer1 →×   →×    →×     (×: 逐元素乘法)
             |      |      |
  DNN Layer2 →×   →×    →×
             |      |      |
           Output_1 Output_2 Output_3
           (点击)   (转化)   (收藏)
```

### 3.3 数学公式

Gate 网络的输出：

$$g_{task} = \gamma \cdot \text{Sigmoid}(\text{ReLU}(xW_1 + b_1)W_2 + b_2)$$

其中：
- $x$ 是任务特定的 ID 特征（用户 ID + 物品 ID 的 Embedding 拼接）
- $W_1 \in \mathbb{R}^{d \times d'}$，$W_2 \in \mathbb{R}^{d' \times d}$ 是两层 MLP 的权重
- $\gamma = 2$ 是缩放因子，初始化时 Gate 输出 ≈ 1（因为 Sigmoid(0) = 0.5，× 2 = 1）

Gate 对 DNN 第 $l$ 层的调制：

$$h_l' = h_l \odot g_{task}^{(l)}$$

其中 $\odot$ 是逐元素乘法，$h_l$ 是第 $l$ 层的隐藏表示。

### 3.4 为什么初始化为 1

$\gamma = 2$ 且 Sigmoid 输出初始化在 0.5 附近，因此 Gate 输出 ≈ 1。这意味着训练开始时 PPNet 退化为普通 DNN，随着训练逐步分化出任务个性化参数，保证训练稳定性。

### 3.5 PyTorch 实现

```python
import torch
import torch.nn as nn


class GateNU(nn.Module):
    def __init__(self, input_dim, output_dim, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.w1 = nn.Linear(input_dim, output_dim)
        self.w2 = nn.Linear(output_dim, output_dim)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, gate_input):
        h = torch.relu(self.w1(gate_input))
        gate = torch.sigmoid(self.w2(h))
        return self.gamma * gate


class PPNetLayer(nn.Module):
    def __init__(self, input_dim, output_dim, gate_input_dim, gamma=2.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = GateNU(gate_input_dim, output_dim, gamma)

    def forward(self, x, gate_input):
        h = self.fc(x)
        g = self.gate(gate_input)
        return h * g


class PPNet(nn.Module):
    def __init__(self, num_features, embedding_dim=16,
                 hidden_dims=[128, 64], num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in num_features
        ])

        total_dim = len(num_features) * embedding_dim
        user_item_dim = 2 * embedding_dim

        self.task_towers = nn.ModuleList()
        self.task_gates = nn.ModuleList()

        for t in range(num_tasks):
            layers = nn.ModuleList()
            prev_dim = total_dim
            for h_dim in hidden_dims:
                layers.append(PPNetLayer(prev_dim, h_dim, user_item_dim))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, 1))

            self.task_towers.append(layers)
            self.task_gates.append(nn.Embedding(num_features[0], embedding_dim))

        self.user_emb = self.embeddings[0]
        self.item_emb = self.embeddings[1]

    def forward(self, x):
        emb_list = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        concat_emb = torch.cat(emb_list, dim=-1)

        gate_input = torch.cat([self.user_emb(x[:, 0]),
                                 self.item_emb(x[:, 1])], dim=-1)

        outputs = []
        for t in range(self.num_tasks):
            h = concat_emb
            for i, layer in enumerate(self.task_towers[t][:-1]):
                h = layer(h, gate_input)
                h = torch.relu(h)
            out = torch.sigmoid(self.task_towers[t][-1](h))
            outputs.append(out.squeeze(-1))

        return outputs
```

## 4. EPNet（Embedding Personalized Network）

### 4.1 核心思想

EPNet 通过场景特征生成门控权重，调整共享 Embedding 的表征，使不同场景获得适合的输入特征表示。

### 4.2 架构详解

```
    Scene Features (场景ID + 统计特征)
              |
         [Scene Encoder]
              |
         [Gate Network]
              |
     Scene-specific Gate Weights
              |
    Shared Embedding → × Gate → Personalized Embedding
              |
         [DNN Backbone]
              |
         [Task Outputs]
```

### 4.3 数学公式

场景编码：

$$z_s = \text{Encoder}(f_{scene})$$

门控权重：

$$g_s = \gamma \cdot \text{Sigmoid}(W_g z_s + b_g)$$

Embedding 调制：

$$e_{personalized} = e_{shared} \odot g_s$$

其中 $e_{shared}$ 是共享 Embedding 层的输出，$g_s$ 是场景生成的门控权重。

### 4.4 为什么调制 Embedding 而非 DNN

- Embedding 是特征入口，调制 Embedding 等效于为每个场景选择不同的特征子空间
- 不同场景对特征的偏好确实不同（首页推荐重兴趣，搜索推荐重相关性）
- 调制发生在输入层，影响更全局

### 4.5 PyTorch 实现

```python
class EPNet(nn.Module):
    def __init__(self, num_features, embedding_dim=16,
                 hidden_dims=[128, 64],
                 scene_feature_dim=8):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in num_features
        ])

        total_dim = len(num_features) * embedding_dim

        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, total_dim)
        )

        nn.init.zeros_(self.scene_encoder[-1].weight)
        nn.init.zeros_(self.scene_encoder[-1].bias)
        self.gamma = 2.0

        self.dnn = nn.Sequential(
            nn.Linear(total_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x, scene_features):
        emb_list = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        concat_emb = torch.cat(emb_list, dim=-1)

        scene_gate = self.gamma * torch.sigmoid(
            self.scene_encoder(scene_features)
        )

        personalized_emb = concat_emb * scene_gate

        output = torch.sigmoid(self.dnn(personalized_emb).squeeze(-1))
        return output
```

## 5. PEPNet（Parameter & Embedding Personalized Network）

### 5.1 核心思想

PEPNet 是 PPNet 和 EPNet 的级联，同时解决多任务跷跷板和多场景跷跷板：

- **第一层（EPNet）**：场景特征调制 Embedding → 解决多场景问题
- **第二层（PPNet）**：任务特征调制 DNN 参数 → 解决多任务问题

### 5.2 架构详解

```
Input Features
      |
[Shared Embedding]
      |
EPNet: × Scene Gate  ← Scene Features
      |
[Personalized Embedding]
      |
      /        \
  [Tower A]  [Tower B]
      |          |
PPNet: × Task Gate_A  × Task Gate_B  ← User/Item ID Features
      |          |
  Output A    Output B
```

### 5.3 数学公式

EPNet 阶段：

$$e' = e \odot g_{scene}, \quad g_{scene} = \gamma \cdot \sigma(\text{Enc}(f_{scene}))$$

PPNet 阶段：

$$h_l' = h_l \odot g_{task}^{(l)}, \quad g_{task}^{(l)} = \gamma \cdot \sigma(\text{ReLU}(f_{id}W_1)W_2)$$

总输出：

$$\hat{y}_{task,scene} = \text{DNN}_{task}(e' \odot g_{scene}) \odot g_{task}$$

### 5.4 PyTorch 实现

```python
class PEPNet(nn.Module):
    def __init__(self, num_features, embedding_dim=16,
                 hidden_dims=[128, 64], num_tasks=2,
                 scene_feature_dim=8):
        super().__init__()
        self.num_tasks = num_tasks
        self.gamma = 2.0

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in num_features
        ])
        total_dim = len(num_features) * embedding_dim

        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, total_dim)
        )
        nn.init.zeros_(self.scene_encoder[-1].weight)
        nn.init.zeros_(self.scene_encoder[-1].bias)

        self.user_emb = self.embeddings[0]
        self.item_emb = self.embeddings[1]
        gate_input_dim = 2 * embedding_dim

        self.task_towers = nn.ModuleList()
        for t in range(num_tasks):
            tower = nn.ModuleList()
            prev_dim = total_dim
            for h_dim in hidden_dims:
                tower.append(PPNetLayer(prev_dim, h_dim, gate_input_dim))
                prev_dim = h_dim
            tower.append(nn.Linear(prev_dim, 1))
            self.task_towers.append(tower)

    def forward(self, x, scene_features):
        emb_list = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        concat_emb = torch.cat(emb_list, dim=-1)

        scene_gate = self.gamma * torch.sigmoid(
            self.scene_encoder(scene_features)
        )
        personalized_emb = concat_emb * scene_gate

        gate_input = torch.cat([self.user_emb(x[:, 0]),
                                 self.item_emb(x[:, 1])], dim=-1)

        outputs = []
        for t in range(self.num_tasks):
            h = personalized_emb
            for layer in self.task_towers[t][:-1]:
                h = layer(h, gate_input)
                h = torch.relu(h)
            out = torch.sigmoid(self.task_towers[t][-1](h))
            outputs.append(out.squeeze(-1))

        return outputs
```

## 6. 三模型对比总结

| 维度 | PPNet | EPNet | PEPNet |
|------|-------|-------|--------|
| **目标** | 多任务个性化 | 多场景个性化 | 多任务 + 多场景 |
| **调制对象** | DNN 层参数 | Embedding 表征 | Embedding + DNN |
| **Gate 输入** | 用户/物品 ID | 场景特征 | 场景特征 + 用户/物品 ID |
| **Gate 输出** | 每层权重 | Embedding 权重 | 两级权重 |
| **初始化** | γ=2, Sigmoid → ≈1 | γ=2, Sigmoid → ≈1 | 两级均为 ≈1 |
| **复杂度** | 中 | 中 | 高 |
| **参数增量** | O(d×L×T) | O(d) | O(d×(1+L×T)) |
| **适用场景** | 多任务推荐 | 多场景推荐 | 多场景多任务推荐 |

其中 $d$ 是隐藏层维度，$L$ 是 DNN 层数，$T$ 是任务数。

## 7. 训练技巧与生产经验

### 7.1 Gate 初始化

所有 Gate 的最后一层权重和偏置初始化为 0，配合 $\gamma=2$，使得 Gate 初始输出 ≈ 1。这保证训练初期不破坏预训练 Backbone 的性能。

### 7.2 渐进式训练

```
阶段1：冻结 Gate，只训练 Backbone（warm-up）
阶段2：解冻 Gate，联合训练
阶段3：调小学习率，精调
```

### 7.3 Gate 特征选择

- **PPNet**：使用用户 ID + 物品 ID，不使用上下文特征（避免噪声）
- **EPNet**：使用场景 ID + 场景统计特征（曝光量、CTR 等）
- **PEPNet**：两个 Gate 使用不同的特征源

### 7.4 调参建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| $\gamma$ | 2.0 | 缩放因子 |
| Gate 隐藏层 | 与调制维度相同 | 简单有效 |
| Gate 学习率 | Backbone 的 10 倍 | 加速 Gate 学习 |
| DNN 层数 | 2-3 层 | 太深收益递减 |

## 8. 完整训练示例

```python
def train_pepnet():
    num_features = [1000, 5000, 200, 50]
    num_tasks = 2
    batch_size = 256
    scene_feature_dim = 8
    num_epochs = 20

    model = PEPNet(
        num_features=num_features,
        embedding_dim=16,
        hidden_dims=[128, 64],
        num_tasks=num_tasks,
        scene_feature_dim=scene_feature_dim
    )

    optimizer = torch.optim.Adam([
        {'params': model.embeddings.parameters(), 'lr': 1e-4},
        {'params': model.scene_encoder.parameters(), 'lr': 1e-3},
        {'params': model.task_towers.parameters(), 'lr': 1e-3},
    ])

    for epoch in range(num_epochs):
        x = torch.zeros(batch_size, len(num_features), dtype=torch.long)
        for i, size in enumerate(num_features):
            x[:, i] = torch.randint(0, size, (batch_size,))

        scene_features = torch.rand(batch_size, scene_feature_dim)
        labels = [
            (torch.rand(batch_size) > 0.7).float(),
            (torch.rand(batch_size) > 0.9).float(),
        ]

        outputs = model(x, scene_features)

        loss = 0
        for t in range(num_tasks):
            loss += nn.functional.binary_cross_entropy(
                outputs[t].clamp(1e-7, 1-1e-7), labels[t]
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                  f"Task0_mean={outputs[0].mean().item():.4f}, "
                  f"Task1_mean={outputs[1].mean().item():.4f}")


if __name__ == "__main__":
    train_pepnet()
```

## 9. 应用场景

| 场景 | 适用模型 | 说明 |
|------|---------|------|
| 电商多目标推荐 | PPNet | 同时预估点击、加购、购买 |
| 多场景推荐 | EPNet | 首页/搜索/购物车多场景 |
| 复杂推荐系统 | PEPNet | 多场景 + 多目标联合 |
| 广告多任务 | PPNet | 点击率 + 转化率 |
| 内容多场景 | EPNet | 信息流/短视频/直播 |

## 10. 优缺点分析

### PPNet

- **优点**：简单有效，即插即用，每个任务获得个性化参数
- **缺点**：只解决多任务，未考虑场景差异

### EPNet

- **优点**：解决场景间特征对齐，Embedding 级调制影响全局
- **缺点**：只解决多场景，未考虑任务差异

### PEPNet

- **优点**：同时解决双重跷跷板，效果最优
- **缺点**：架构复杂，超参数多，训练和推理开销增加

## 11. 常见问题与易错点

### Q1：Gate 用什么激活函数？

Sigmoid，不用 Softmax。每个维度独立调制（不是互斥选择），Sigmoid 输出 $[0, \gamma]$ 适合逐元素缩放。

### Q2：为什么 Gate 输入只用 ID 特征？

上下文特征（时间、位置等）信息量大但噪声也大，ID 特征更稳定、更具个性化语义。Gate 需要稳定的信号来生成调制权重。

### Q3：Gate 和 Attention 的区别？

- **Attention**：不同位置的加权聚合（横向调制）
- **Gate**：同一位置的特征缩放（纵向调制）
- 二者正交互补，可以组合使用

### Q4：推理时 Gate 有额外开销吗？

有但很小。Gate 只是两层小 MLP + 逐元素乘法，相比 Backbone 开销可忽略。

### Q5：PPNet 能和 MMoE 结合吗？

可以。PPNet 的 Gate 作用于 DNN 层参数，MMoE 的 Gate 作用于 Expert 输出。两者在不同层面调制，可以叠加使用。

## 12. 演进路线

```
PPNet (2021)          EPNet (2022)           PEPNet (2023)
  多任务个性化          多场景个性化           多任务 + 多场景
      ↓                     ↓                     ↓
 Gate 调制 DNN        Gate 调制 Embedding    两级 Gate 级联
      ↓                     ↓                     ↓
 解决任务跷跷板        解决场景跷跷板         解决双重跷跷板
```

## 13. 学习总结

| 要点 | 内容 |
|------|------|
| 核心机制 | Gate 网络生成动态权重，调制模型参数 |
| 初始化技巧 | γ=2 + 零初始化，保证训练稳定性 |
| PPNet | 任务 ID → Gate → 调制 DNN 层 |
| EPNet | 场景特征 → Gate → 调制 Embedding |
| PEPNet | EPNet + PPNet 级联 |

## 14. 练习题与思考题

1. **推导题**：推导 Gate 输出关于 Gate 输入的梯度，说明为什么零初始化是安全的。
2. **思考题**：如果场景数量很多（如上百个），EPNet 是否还能有效？如何改进？
3. **实现题**：将 PPNet 与 MMoE 结合，实现一个多任务模型。
4. **分析题**：为什么 EPNet 调制 Embedding 比 DNN 层更适合多场景问题？

## 15. 学习路径建议

1. **前置知识**：多任务学习、门控机制、MoE/MMoE
2. **论文**：PPNet (RecSys 2021), EPNet (RecSys 2022), PEPNet (KDD 2023)
3. **进阶**：STAR、SAR-Net、PLE 等多任务/多场景模型
4. **延伸**：动态网络、条件计算、元学习在推荐中的应用
