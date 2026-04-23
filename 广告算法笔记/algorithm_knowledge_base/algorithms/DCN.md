# DCN（Deep & Cross Network）学习文档

## 1. 算法基础认知

DCN 通过 Cross Network 自动学习有界阶的特征交叉，与 Deep Network 并行，实现自动化的高阶特征交互。

在 full.md 中，DCN 被列为 2.0 阶段的代表性模型之一："2.0 阶段聚焦于交叉特征挖掘，FM、DCN 等模型实现了自动化的特征交互"。

## 2. 核心原理

### Cross Network

第 l 层交叉：

$$
x_{l+1} = x_0 \odot (w_l^T x_l + b_l) + x_l = f(x_l, w_l, b_l) + x_l
$$

每一层都有效地增加了特征交叉的阶数，且具有残差连接。

### DCN v2

DCN v2 使用低秩矩阵替代向量权重，提升表达能力：

$$
x_{l+1} = x_0 \odot (V_l U_l^T x_l + b_l) + x_l
$$

## 3. 在广告中的应用

- CTR 预估中的特征交叉
- 模型内隐式交叉：DCN/CIN（xDeepFM）自动学习高阶交叉
- 与 DIN 等序列模型组合使用（如 DCNv2 + DIN）

## 4. 代码实现

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
```

## 5. 学习总结

DCN 是广告 CTR 预估中广泛使用的特征交叉模型。DCN v2 与 DIN 组合是工业界常用的排序模型架构。
