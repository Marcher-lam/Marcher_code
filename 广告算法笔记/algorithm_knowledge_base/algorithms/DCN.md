# DCN（Deep & Cross Network）学习文档

## 1. 算法基础认知

DCN（Deep & Cross Network）由 Google 于 2017 年提出，通过 Cross Network 自动学习有界阶的显式特征交叉，与 Deep Network 并行，实现自动化的高阶特征交互。DCN v2 进一步引入低秩矩阵提升表达能力，是广告 CTR 预估中广泛使用的模型。

## 2. 核心原理

### Cross Network

第 $l$ 层交叉操作：

$$
x_{l+1} = x_0 \odot (w_l^T x_l + b_l) + x_l = f(x_l, w_l, b_l) + x_l
$$

每一层将当前状态 $x_l$ 与原始输入 $x_0$ 做 Hadamard 逐元素乘积，实现以 $x_0$ 为基础的逐阶交叉，且具有残差连接。

### DCN v2

DCN v2 使用低秩矩阵替代向量权重，大幅提升表达能力：

$$
x_{l+1} = x_0 \odot (V_l U_l^T x_l + b_l) + x_l
$$

## 3. 数学公式与推导

**交叉层的阶数推导**：第 $l$ 层输出包含从 1 阶到 $l+1$ 阶的所有多项式交叉项。设 $x_0 \in \mathbb{R}^d$，则：

$$
x_l = \sum_{i=0}^{l} \alpha_i (w^T x_0)^i \cdot x_0
$$

Cross Network 的参数量仅为 $O(d \times L)$（$L$ 为交叉层数），远低于全连接的 $O(d^2)$。

**最终预测**：

$$
\hat{y} = \sigma(W_{out}^T [x_{cross}^{(L)}, x_{deep}^{(L)}] + b)
$$

## 4. 训练过程讲解

1. 输入稀疏特征通过 Embedding 层转为稠密向量，拼接得到 $x_0$
2. Cross 分支：$x_0$ 经过多层交叉网络，每一层与 $x_0$ 做逐元素乘积
3. Deep 分支：$x_0$ 通过多层全连接 ReLU 网络
4. 两个分支输出拼接后过线性层 + Sigmoid 得到预测概率
5. 交叉熵损失反向传播，同时更新 Cross 和 Deep 的参数

## 5. 应用场景

- 广告 CTR 预估中的自动化特征交叉
- 推荐系统排序模型
- DCNv2 + DIN 组合是工业界常用的排序模型架构
- 适合高维稀疏特征场景（广告、推荐）

## 6. 优缺点分析

**优点**：
- 自动学习显式特征交叉，无需人工设计
- Cross Network 参数量极小，计算高效
- DCN v2 低秩矩阵进一步提升表达能力

**缺点**：
- 原始 DCN 的交叉层受限于向量权重，表达力有限
- Cross 网络每层交互都是 bit-wise 的，不如 FM 的 vector-wise 交互灵活
- 需要调节交叉层数这个关键超参数

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])

    def forward(self, x0):
        xl = x0
        for i in range(len(self.w)):
            xl = x0 * (torch.matmul(xl, self.w[i]) + self.b[i]) + xl
        return xl

class DCN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], num_cross_layers=3):
        super().__init__()
        self.cross = CrossNetwork(input_dim, num_cross_layers)
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(dim, h), nn.ReLU()])
            dim = h
        self.deep = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim + hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cross_out = self.cross(x)
        deep_out = self.deep(x)
        combined = torch.cat([cross_out, deep_out], dim=-1)
        return self.sigmoid(self.output(combined))

model = DCN(input_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(32, 64)
y = torch.randint(0, 2, (32, 1)).float()
for epoch in range(10):
    pred = model(x)
    loss = nn.BCELoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class CrossNetworkNumpy:
    def __init__(self, dim, num_layers=3):
        self.num_layers = num_layers
        self.w = [np.random.randn(dim) * 0.01 for _ in range(num_layers)]
        self.b = [np.zeros(dim) for _ in range(num_layers)]
        self.lr = 0.01

    def forward(self, x0):
        self.x_layers = [x0.copy()]
        xl = x0.copy()
        for i in range(self.num_layers):
            xl = x0 * (xl @ self.w[i] + self.b[i]) + xl
            self.x_layers.append(xl.copy())
        return xl

    def train_step(self, x0, grad_out):
        xl = self.x_layers[-1]
        for i in reversed(range(self.num_layers)):
            xl_prev = self.x_layers[i]
            self.w[i] -= self.lr * (x0 * xl_prev).T @ grad_out / len(x0)
            self.b[i] -= self.lr * np.mean(grad_out, axis=0)
```

## 9. 可视化与结果理解

- 绘制不同交叉层数（1~6层）对 AUC 的影响曲线
- 可视化 Cross 网络每一层输出的特征分布变化
- 对比 DCN 与纯 Deep 模型的特征交叉热力图

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

input_dim = 8
x0 = np.random.randn(input_dim) * 0.5 + 0.5

w1 = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.6, -0.1])
w2 = np.array([0.2, 0.4, -0.3, 0.5, 0.1, -0.2, 0.3, 0.4])
w3 = np.array([-0.1, 0.3, 0.2, -0.4, 0.5, 0.1, -0.3, 0.2])

xl_prev = x0.copy()
layers_output = [x0.copy()]
for w in [w1, w2, w3]:
    xl = x0 * (np.dot(xl_prev, w)) + xl_prev
    layers_output.append(xl.copy())
    xl_prev = xl.copy()

feature_labels = [f'f{i + 1}' for i in range(input_dim)]
layer_names = ['Input $x_0$', 'Cross $x_1$', 'Cross $x_2$', 'Cross $x_3$']
colors = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350']

data_matrix = np.array(layers_output)
im = axes[0].imshow(data_matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
axes[0].set_xticks(range(input_dim))
axes[0].set_xticklabels(feature_labels, fontsize=10)
axes[0].set_yticks(range(len(layer_names)))
axes[0].set_yticklabels(layer_names, fontsize=11)
axes[0].set_title('Cross Network Layer Outputs (Feature Heatmap)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=axes[0], shrink=0.8, label='Feature Value')

epochs = np.arange(1, 51)
deep_only_loss = 0.72 * np.exp(-0.04 * epochs) + 0.35 + np.random.normal(0, 0.008, len(epochs))
dcn_loss = 0.68 * np.exp(-0.06 * epochs) + 0.28 + np.random.normal(0, 0.006, len(epochs))
cross_only_loss = 0.75 * np.exp(-0.03 * epochs) + 0.42 + np.random.normal(0, 0.01, len(epochs))

axes[1].plot(epochs, cross_only_loss, 'b--', linewidth=2, label='Cross Only', alpha=0.8)
axes[1].plot(epochs, deep_only_loss, 'g-.', linewidth=2, label='Deep Only', alpha=0.8)
axes[1].plot(epochs, dcn_loss, 'r-', linewidth=2.5, label='DCN (Cross + Deep)', alpha=0.9)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('LogLoss', fontsize=12)
axes[1].set_title('Training Loss: Cross vs Deep vs DCN', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11, loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.suptitle('DCN — Cross Network Structure & Training Visualization', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('dcn_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **离线指标**：AUC、LogLoss、GAUC（分组AUC）
- **特征交叉效果**：对比有/无 Cross 网络的 AUC 差异
- **效率指标**：Cross 网络参数量仅为同等 DNN 的 1/10

## 11. 常见问题与易错点

- 交叉层数设置过高可能导致过拟合，通常 3~6 层足够
- 注意 Cross 网络中 $x_0$ 是固定的原始输入，不是上一层输出
- DCN v2 的低秩分解需要选择合适的中间维度 $r$
- 交叉层是 bit-wise 交互，对于需要 vector-wise 交互的场景可考虑 xDeepFM

## 12. 学习总结

DCN 是广告 CTR 预估中广泛使用的特征交叉模型。Cross Network 以极少的参数实现了显式高阶特征交叉，DCN v2 的低秩矩阵进一步提升了表达力。在工业界，DCNv2 + DIN 组合是常用的排序模型架构。

## 13. 练习题与思考题（含答案）

**Q1**: Cross Network 的参数量与层数 $L$ 和维度 $d$ 的关系是什么？
> A1: 参数量为 $O(d \times L)$，每层只有一个 $d$ 维权重向量和一个 $d$ 维偏置。

**Q2**: 为什么 Cross 网络每层都要与 $x_0$ 做乘积而不是与前一层 $x_l$？
> A2: 与 $x_0$ 相乘确保每一层都在原始特征基础上生成新的交叉项，保证交叉的多样性和有界性。

**Q3**: DCN v2 为什么用低秩矩阵 $VU^T$ 替代向量 $w$？
> A3: 向量权重表达力有限，低秩矩阵增加参数量可控地提升模型容量。

## 14. 学习路径建议

1. 先学习 FM 理解特征交叉概念
2. 学习 Wide & Deep 理解混合架构
3. 阅读 DCN 论文（Google 2017）和 DCN v2 论文
4. 进阶：学习 xDeepFM（CIN，vector-wise 交叉）
