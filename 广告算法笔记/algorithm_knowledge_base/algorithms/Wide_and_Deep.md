# Wide & Deep 学习文档

## 1. 算法基础认知

Wide & Deep（2016, Google）结合了线性模型的**记忆能力**和深度网络的**泛化能力**。Wide 部分通过显式特征交叉实现记忆，Deep 部分通过隐式特征交互实现泛化。它是深度学习推荐模型的里程碑，开创了"记忆+泛化"的混合架构范式。

核心思想：线性模型擅长记忆已知的规则模式（如"已购牛奶→购酸奶"），DNN 擅长发现新的特征组合，两者联合训练取长补短。

## 2. 核心原理

### 模型结构

```
Input → ┌─ Wide Part (交叉特征) ─┐
        └─ Deep Part (DNN) ──────┴→ Concat → Output
```

- **Wide 部分**：线性模型 $y = w^T x + b$，需要手动设计交叉特征 $\phi(x)$（如特征笛卡尔积）
- **Deep 部分**：Embedding → 多层全连接 → 隐式特征交互，自动学习非线性组合
- **联合训练**：Wide 和 Deep 同时训练，共享梯度，端到端优化

### 预测公式

$$
P(Y=1|x) = \sigma(w_{wide}^T [x, \phi(x)] + w_{deep}^T a^{(l_f)} + b)
$$

其中 $a^{(l_f)}$ 是 Deep 部分最后一层隐藏层的输出，$\phi(x)$ 是人工设计的交叉特征变换。

## 3. 数学公式与推导

**Wide 部分线性输出**：

$$
y_{wide} = w_{wide}^T [x, \phi(x)]
$$

**Deep 部分多层前馈**：

$$
a^{(l+1)} = f(W^{(l)} a^{(l)} + b^{(l)})
$$

**联合损失**（二分类交叉熵）：

$$
L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]
$$

梯度同时回传到 Wide 和 Deep 两个分支，通过反向传播联合更新所有参数。

## 4. 训练过程讲解

1. **输入处理**：稀疏特征通过 Embedding 层转为稠密向量，稠密特征直接输入
2. **Wide 分支**：对原始特征做交叉变换（如 AND(user_installed_app=netflix, impression_app=pandora)），送入线性层
3. **Deep 分支**：拼接所有 Embedding 和稠密特征，通过多层 ReLU 全连接层
4. **融合输出**：两个分支的 logit 相加后过 Sigmoid 得到预测概率
5. **反向传播**：统一用交叉熵损失，Wide 和 Deep 共享梯度更新

## 5. 应用场景

- Google Play App 推荐（首个工业应用）
- 通用推荐系统排序阶段
- 广告 CTR 预估与排序
- 适合需要同时利用历史记忆规律和新组合泛化的场景

## 6. 优缺点分析

**优点**：
- 同时具备记忆与泛化能力，互补增强
- Wide 部分可注入业务先验知识
- 联合训练避免了两阶段pipeline的误差累积

**缺点**：
- Wide 部分仍需人工设计交叉特征，工程成本高
- Deep 部分的隐式交叉可解释性差
- 后续 DeepFM、DCN 等模型已自动化了特征交叉

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn
import numpy as np

class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, deep_dim, hidden_dims=[128, 64]):
        super().__init__()
        self.wide = nn.Linear(wide_dim, 1)
        layers = []
        dim = deep_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.2)])
            dim = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wide_input, deep_input):
        wide_logit = self.wide(wide_input)
        deep_logit = self.deep_out(self.deep(deep_input))
        return self.sigmoid(wide_logit + deep_logit)

model = WideAndDeep(wide_dim=100, deep_dim=50)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

wide_x = torch.randn(32, 100)
deep_x = torch.randn(32, 50)
y = torch.randint(0, 2, (32, 1)).float()

for epoch in range(10):
    pred = model(wide_x, deep_x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class WideAndDeepNumpy:
    def __init__(self, wide_dim, deep_dim, hidden=32):
        self.w_wide = np.random.randn(wide_dim, 1) * 0.01
        self.w1 = np.random.randn(deep_dim, hidden) * 0.01
        self.w2 = np.random.randn(hidden, 1) * 0.01
        self.lr = 0.01

    def predict(self, wide_x, deep_x):
        self.z1 = deep_x @ self.w1
        self.a1 = np.maximum(0, self.z1)
        deep_logit = self.a1 @ self.w2
        wide_logit = wide_x @ self.w_wide
        return sigmoid(wide_logit + deep_logit)

    def train_step(self, wide_x, deep_x, y):
        pred = self.predict(wide_x, deep_x)
        d = pred - y
        self.w_wide -= self.lr * (wide_x.T @ d) / len(y)
        d2 = d @ self.w2.T * (self.z1 > 0)
        self.w2 -= self.lr * (self.a1.T @ d) / len(y)
        self.w1 -= self.lr * (deep_x.T @ d2) / len(y)
        return np.mean(-y * np.log(pred + 1e-8) - (1 - y) * np.log(1 - pred + 1e-8))
```

## 9. 可视化与结果理解

- 绘制 Wide 和 Deep 分支 logit 的分布，观察两者贡献比例
- 对比纯 Wide、纯 Deep、Wide & Deep 三者的 AUC 曲线
- 观察 Deep 部分 Embedding 的 t-SNE 可视化，检查语义聚类是否合理

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

epochs = np.arange(1, 61)

wide_loss = 0.65 * np.exp(-0.03 * epochs) + 0.45 + np.random.normal(0, 0.01, len(epochs))
deep_loss = 0.70 * np.exp(-0.05 * epochs) + 0.30 + np.random.normal(0, 0.008, len(epochs))
wide_deep_loss = 0.72 * np.exp(-0.06 * epochs) + 0.22 + np.random.normal(0, 0.006, len(epochs))

wide_auc = 0.70 + 0.08 * (1 - np.exp(-0.03 * epochs)) + np.random.normal(0, 0.005, len(epochs))
deep_auc = 0.72 + 0.10 * (1 - np.exp(-0.04 * epochs)) + np.random.normal(0, 0.004, len(epochs))
wide_deep_auc = 0.73 + 0.13 * (1 - np.exp(-0.05 * epochs)) + np.random.normal(0, 0.003, len(epochs))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(epochs, wide_loss, 'b--', linewidth=2, label='Wide Only', alpha=0.8)
axes[0].plot(epochs, deep_loss, 'g-.', linewidth=2, label='Deep Only', alpha=0.8)
axes[0].plot(epochs, wide_deep_loss, 'r-', linewidth=2.5, label='Wide & Deep', alpha=0.9)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('LogLoss', fontsize=12)
axes[0].set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, wide_auc, 'b--', linewidth=2, label=f'Wide Only (final={wide_auc[-1]:.3f})', alpha=0.8)
axes[1].plot(epochs, deep_auc, 'g-.', linewidth=2, label=f'Deep Only (final={deep_auc[-1]:.3f})', alpha=0.8)
axes[1].plot(epochs, wide_deep_auc, 'r-', linewidth=2.5, label=f'Wide & Deep (final={wide_deep_auc[-1]:.3f})', alpha=0.9)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('AUC', fontsize=12)
axes[1].set_title('AUC Comparison', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10, loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Wide & Deep — Training Loss & AUC Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('wide_deep_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **离线指标**：AUC、LogLoss、NDCG@K（排序场景）
- **在线指标**：CTR、CVR、eCPM 提升
- Google 实验表明 Wide & Deep 相比纯 Wide 提升 App 获得率 3.9%

## 11. 常见问题与易错点

- Wide 部分的交叉特征需要根据业务经验设计，自动化程度低
- 两个分支的输入维度和特征需要对齐，否则融合时维度不匹配
- Deep 部分过深可能导致过拟合，需要配合 Dropout 和正则化
- 联合训练时 Wide 分支的稀疏梯度与 Deep 分支的稠密梯度更新频率可能不一致

## 12. 学习总结

Wide & Deep 是深度学习推荐模型的里程碑，开创了"记忆+泛化"的混合架构范式。后续的 DeepFM、DCN 等模型都在此基础上改进了特征交叉方式，实现了自动化的特征交互。

## 13. 练习题与思考题（含答案）

**Q1**: Wide 部分和 Deep 部分分别擅长什么？
> A1: Wide 擅长记忆已知的规则模式（显式交叉），Deep 擅长泛化发现新的特征组合（隐式交互）。

**Q2**: 为什么使用联合训练而非独立训练后再融合？
> A2: 联合训练让两个分支共享梯度信息，可以端到端优化，避免两阶段pipeline的误差累积。

**Q3**: 预测公式中 $a^{(l_f)}$ 代表什么？
> A3: Deep 部分最后一层隐藏层的激活输出，作为 Deep 分支对最终预测的贡献。

## 14. 学习路径建议

1. 先掌握线性回归和逻辑回归（理解 Wide 部分）
2. 学习前馈神经网络和 Embedding（理解 Deep 部分）
3. 学习 Wide & Deep 论文原文（Google 2016）
4. 进阶：学习 DeepFM（用 FM 替代 Wide）、DCN（自动交叉网络）
