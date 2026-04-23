# Wide & Deep 学习文档

## 1. 算法基础认知

Wide & Deep（2016, Google）结合了线性模型的记忆能力和深度网络的泛化能力。Wide 部分通过显式特征交叉实现记忆，Deep 部分通过隐式特征交互实现泛化。

## 2. 核心原理

### 模型结构

```
Input → ┌─ Wide Part (交叉特征) ─┐
        └─ Deep Part (DNN) ──────┴→ Concat → Output
```

- **Wide 部分**：线性模型 y = w^T x + b，需要手动设计交叉特征
- **Deep 部分**：Embedding → 多层全连接 → 隐式特征交互
- **联合训练**：Wide 和 Deep 同时训练，共享梯度

### 预测公式

$$
P(Y=1|x) = \sigma(w_{wide}^T [x, \phi(x)] + w_{deep}^T a^{(l_f)} + b)
$$

## 3. 应用场景

- 通用推荐、广告排序
- Google Play App 推荐的首个工业应用
- 适合需要同时利用记忆（历史规律）和泛化（新组合）的场景

## 4. 代码实现

```python
import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, deep_dim, hidden_dims=[128, 64], embed_dim=8):
        super().__init__()
        self.wide = nn.Linear(wide_dim, 1)
        layers = []
        input_dim = deep_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(0.2)])
            input_dim = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wide_input, deep_input):
        wide_logit = self.wide(wide_input)
        deep_logit = self.deep_out(self.deep(deep_input))
        return self.sigmoid(wide_logit + deep_logit)
```

## 5. 学习总结

Wide & Deep 是深度学习推荐模型的里程碑，开创了"记忆+泛化"的混合架构范式。后续的 DeepFM、DCN 等模型都在此基础上改进了特征交叉方式。
