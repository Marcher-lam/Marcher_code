# DeepFM 学习文档

## 1. 算法基础认知

DeepFM 是华为在 2017 年提出的 CTR 预估模型，发表于 IJCAI 2017。它将 FM（Factorization Machines）和 DNN 融合为一个端到端模型，无需人工特征工程即可同时捕捉低阶和高阶特征交互。

核心动机：Wide & Deep 的 Wide 部分仍需人工设计交叉特征，工程成本高。DeepFM 用 FM 替代线性 Wide 部分，自动学习二阶交叉，同时 DNN 学习高阶交叉，两者共享 Embedding 实现端到端联合训练。

## 2. 核心原理

DeepFM 由两个组件组成，共享同一套 Embedding：

- **FM 组件**：对 Embedding 向量做两两内积，捕捉二阶特征交互
- **DNN 组件**：将 Embedding 拼接后送入多层全连接网络，隐式学习高阶交叉

关键设计：FM 和 DNN 共享底层 Embedding，避免单独训练导致的特征表示不一致，同时减少参数量。

预测公式：

$$\hat{y} = \sigma(y_{FM} + y_{DNN})$$

## 3. 数学公式与推导

**FM 组件**：

$$y_{FM} = \langle w, x \rangle + \sum_{i=1}^{d}\sum_{j=i+1}^{d} \langle V_i, V_j \rangle x_i x_j$$

其中 $V_i \in \mathbb{R}^k$ 是第 $i$ 个特征的隐向量。交叉项可用以下等价公式高效计算：

$$\sum_{i=1}^{d}\sum_{j=i+1}^{d} \langle V_i, V_j \rangle x_i x_j = \frac{1}{2}\left(\left\|\sum_{i=1}^{d} V_i x_i\right\|^2 - \sum_{i=1}^{d}\|V_i x_i\|^2\right)$$

**DNN 组件**：

$$y_{DNN} = W_{H+1} \cdot \sigma(\cdots \sigma(W_1 \cdot [e_1 \oplus e_2 \oplus \cdots \oplus e_n] + b_1) \cdots) + b_{H+1}$$

其中 $e_i$ 是第 $i$ 个字段的 Embedding 向量，$\oplus$ 表示拼接。

**联合损失**：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

## 4. 训练过程讲解

1. **Embedding 查表**：将每个稀疏特征字段映射为稠密向量，FM 和 DNN 共享这些向量
2. **FM 前向**：一阶项直接对原始特征加权求和，二阶项用简化公式计算两两交互
3. **DNN 前向**：将所有 Embedding 拼接后通过多层 ReLU 全连接网络
4. **预测融合**：FM 输出与 DNN 输出相加后过 Sigmoid
5. **反向传播**：统一用交叉熵损失，梯度同时回传到 FM 和 DNN，更新共享 Embedding

## 5. 应用场景

- **广告 CTR 预估**：华为应用市场 App 推荐的原始应用场景
- **推荐系统排序**：商品推荐、内容推荐的精排阶段
- **金融风控**：用户-交易特征的交叉建模
- **竞价广告**：搜索广告和信息流广告的点击率预测
- 适用于稀疏特征场景下需要自动特征交叉的任务

## 6. 优缺点分析

**优点**：
- 无需人工特征工程，FM 自动学习二阶交叉
- 共享 Embedding 使 FM 和 DNN 互相增强
- 端到端训练，工程实现简洁
- 同时捕捉低阶和高阶特征交互

**缺点**：
- FM 部分仅捕捉二阶交叉，未建模更高阶的显式交互
- DNN 部分的高阶交互是隐式的，可解释性较差
- 对稠密特征为主的场景优势不明显

## 7. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class FMComponent(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(100, embedding_dim) for _ in range(num_fields)
        ])
        self.linear = nn.Linear(num_fields, 1)

    def forward(self, x):
        emb_list = [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))]
        emb_stack = torch.stack(emb_list, dim=1)
        square_of_sum = torch.sum(emb_stack, dim=1) ** 2
        sum_of_square = torch.sum(emb_stack ** 2, dim=1)
        fm_out = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        linear_out = self.linear(x.float())
        return fm_out + linear_out


class DeepFM(nn.Module):
    def __init__(self, num_fields, embedding_dim=8, hidden_dims=[64, 32]):
        super().__init__()
        self.fm = FMComponent(num_fields, embedding_dim)
        layers = []
        dim = num_fields * embedding_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.1)])
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.dnn = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields

    def forward(self, x):
        fm_out = self.fm(x)
        emb_list = [self.fm.embeddings[i](x[:, i]) for i in range(self.num_fields)]
        concat = torch.cat(emb_list, dim=-1)
        dnn_out = self.dnn(concat)
        return torch.sigmoid(fm_out + dnn_out)


num_fields = 10
batch_size = 256
model = DeepFM(num_fields)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(10):
    x = torch.randint(0, 100, (batch_size, num_fields))
    y = torch.randint(0, 2, (batch_size, 1)).float()
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
```

## 8. 手工代码实现

```python
import numpy as np


def fm_pairwise_interaction(embeddings):
    sum_vec = np.sum(embeddings, axis=0)
    square_of_sum = np.dot(sum_vec, sum_vec)
    sum_of_square = np.sum(np.sum(embeddings ** 2, axis=0))
    return 0.5 * (square_of_sum - sum_of_square)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0, x)


def deepfm_forward(x, V, w_linear, W1, b1, W2, b2):
    embs = V[x]
    fm_score = fm_pairwise_interaction(embs)
    linear_score = np.sum(x.astype(float))
    concat = embs.flatten()
    h = relu(W1 @ concat + b1)
    dnn_score = (W2 @ h + b2).item()
    logit = fm_score + linear_score + dnn_score
    return sigmoid(logit)


np.random.seed(42)
num_fields = 5
k = 4
V = np.random.randn(100, k) * 0.1
W1 = np.random.randn(32, num_fields * k) * 0.1
b1 = np.zeros(32)
W2 = np.random.randn(1, 32) * 0.1
b2 = np.zeros(1)

x = np.random.randint(0, 100, num_fields)
pred = deepfm_forward(x, V, None, W1, b1, W2, b2)
print(f"预测概率: {pred:.4f}")
```

## 9. 可视化与结果理解

- **FM 二阶交叉热力图**：可视化 $\langle V_i, V_j \rangle$ 矩阵，观察哪些特征对交叉贡献最大
- **Embedding t-SNE**：将学到的特征 Embedding 降维可视化，同类别特征应聚簇
- **FM vs DNN 贡献**：分别统计两个组件对最终 logit 的贡献比例，理解各自的作用

模型效果对比（Criteo 数据集）：

| 模型 | AUC | LogLoss |
|------|-----|---------|
| LR | 0.7821 | 0.4670 |
| FM | 0.7903 | 0.4582 |
| Wide & Deep | 0.7938 | 0.4547 |
| DeepFM | **0.8012** | **0.4481** |

## 10. 模型评估

- **AUC**：衡量排序能力的核心指标，DeepFM 在公开数据集上通常比 FM 高 1-2 个百分点
- **LogLoss**：交叉熵损失，反映概率校准质量
- **GAUC**：按用户分组 AUC 加权均值，推荐场景更贴合线上效果
- **线上指标**：CTR 和 RPM 提升，DeepFM 相比 LR 通常提升 5-10%

## 11. 常见问题与易错点

- **共享 Embedding 的必要性**：如果不共享，FM 和 DNN 学到的特征表示不一致，联合训练的意义减弱
- **Embedding 维度选择**：通常 8-64，太低表达不足，太高容易过拟合
- **FM 的简化公式实现**：必须用 $\|\sum V_i x_i\|^2 - \sum\|V_i x_i\|^2$ 而非两两循环，否则复杂度为 $O(n^2)$
- **DNN 深度选择**：通常 2-3 层足够，过深在稀疏 CTR 场景收益有限

## 12. 学习总结

DeepFM 的核心贡献是：用 FM 自动替代 Wide & Deep 中 Wide 部分的人工特征交叉，同时通过共享 Embedding 实现端到端训练。它是"记忆+泛化"范式的进一步自动化，成为工业界 CTR 预估的标准基线之一。

## 13. 练习题与思考题（含答案）

**Q1：为什么 FM 和 DNN 要共享 Embedding？**

A1：共享 Embedding 使两个组件对同一特征有统一的语义表示，避免信息孤立。训练时 FM 的梯度可以指导 Embedding 学习有意义的交叉方向，DNN 的梯度帮助 Embedding 捕捉高阶语义，两者互补增强。

**Q2：DeepFM 相比 Wide & Deep 的核心改进是什么？**

A2：Wide & Deep 的 Wide 部分需要人工设计交叉特征（如 AND 特征），工程成本高且依赖领域知识。DeepFM 用 FM 组件自动学习所有二阶交叉，无需人工干预，真正实现了端到端的特征交互学习。

**Q3：FM 的二阶交叉为什么可以用简化公式？**

A3：因为 $\sum_{i<j}\langle V_i, V_j\rangle x_i x_j = \frac{1}{2}(\|\sum_i V_i x_i\|^2 - \sum_i \|V_i x_i\|^2)$，将 $O(n^2)$ 的两两计算降为 $O(nk)$ 的求和运算，其中 $k$ 是隐向量维度。

## 14. 学习路径建议

```
LR / FM（理解线性模型和特征交叉）
        ↓
Wide & Deep（记忆+泛化混合架构）
        ↓
DeepFM（自动特征交叉）  ← 你在这里
        ↓
DCN / xDeepFM（显式高阶交叉网络）
        ↓
DIN / DIEN（行为序列建模）
        ↓
多任务学习（ESMM / MMoE / PLE）
```
