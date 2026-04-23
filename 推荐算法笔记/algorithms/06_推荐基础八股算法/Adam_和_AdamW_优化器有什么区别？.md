# 面试题：Adam 和 AdamW 优化器有什么区别？

面试题：Adam 和 AdamW 优化器有什么区别？

Adam 与 AdamW 优化器的核心区别体现在权重衰减的实现机制上，这种差异影响了梯度计算、参数更新规则以及模型的泛化能力。

# 一、权重衰减的数学形式差异

# 1. Adam 的 L2 正则化耦合机制

在 Adam 中，权重衰减通过梯度叠加 L2 正则项实现：$g_t = \nabla f(\theta_t) + \lambda \theta_t$

此时权重衰减被嵌入梯度计算，导致后续的动量计算（$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$）和二阶矩估计（$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$）均包含了正则化项。

这会导致自适应学习率（如 $\sqrt{v_t} + \epsilon$）对权重衰减产生干扰。

**干扰机制的数学分析：** 考虑 Adam 中参数更新的有效步长：

$$
\text{有效步长} = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

由于 $g_t$ 包含了 L2 正则项 $\lambda\theta_t$，$v_t$ 中也包含了 $(\lambda\theta_t)^2$ 的累积。当参数 $\theta$ 较大时，$v_t$ 会被正则项放大，导致有效步长减小——即自适应学习率"惩罚"了权重衰减项，使得正则化效果弱于预期。

# 2. AdamW 的解耦更新规则

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta\lambda\theta_t
$$

AdamW 将权重衰减从梯度计算中剥离，独立施加到参数更新步骤：

其中梯度 $g_t$ 仅包含原始损失函数的梯度，权重衰减项 $\lambda\theta_t$ 独立作用于参数。这使得动量与二阶矩估计仅反映原始梯度信息，不受正则化干扰。

**解耦的理论意义：** Loshchilov 和 Hutter 在 2019 年的论文《Decoupled Weight Decay Regularization》中证明，解耦权重衰减与 SGD + Momentum 的正则化行为等价，而 Adam 中的 L2 正则化则会因自适应学习率的干扰而产生不可预测的正则化效果。

# 二、参数更新过程的数学推导对比

| 步骤 | Adam | AdamW |
|------|------|-------|
| 梯度计算 | $g_t = \nabla f + \lambda\theta$ | $g_t = \nabla f$ (仅原始梯度) |
| 动量计算 | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ | 同左(但 $g_t$ 不含 L2 项) |
| 二阶矩估计 | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ | 同左(但 $g_t$ 不含 L2 项) |
| 参数更新 | $\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ | $\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\theta_t$ |

关键差异：Adam的正则化项被动量机制放大/缩小，而AdamW 的衰减项直接线性作用于参数，独立于自适应学习率。

# 三、理论影响与实验分析

1. 自适应学习率的干扰问题：Adam 中 L2 项会被 $v_t$ 缩放，导致实际衰减强度与理论值 $\lambda$ 产生偏差。例如当梯度较小时，$v_t$ 的缩小效应会放大衰减项，造成参数过度收缩。

2. 泛化性能的理论保障：AdamW 符合解耦权重衰减理论（Decoupled Weight Decay），其行为更接近 SGD with Momentum 的正则化效果。

3. 收敛稳定性分析：AdamW 的独立衰减项使参数更新方向更稳定。以 LLaMA-2 7B 训练为例，AdamW 的损失曲线震荡幅度比 Adam 减少 $30\%$，且达到相同精度所需的训练步数更少。

**Adam 中 L2 正则化失效的极端案例：** 假设某参数的梯度长期为零（如不活跃的 Embedding），Adam 的二阶矩 $v_t$ 会趋近于零，导致有效步长 $\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ 非常大。此时 L2 正则项会被放大，造成参数剧烈收缩甚至变为零。而在 AdamW 中，权重衰减独立施加，不受 $v_t$ 影响，行为更可预测。

# 四、Adam 优化器完整公式

# 1. 一阶矩估计（动量项）

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

超参数：$\beta_1$ 通常设为 0.9，控制历史梯度与当前梯度的权重分配。

含义：通过指数移动平均（EMA）计算当前梯度 $g_t$ 的历史加权平均，类似于动量（Momentum）机制，用于平滑梯度方向。

# 2. 二阶矩估计（自适应学习率项）

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

含义：通过梯度平方的指数移动平均，估计梯度的方差，用于自适应调整每个参数的学习率。

超参数：$\beta_2$ 设为 0.999，反映历史梯度平方的影响。

# 3. 偏差校正

由于初始时刻 $m_0$ 和 $v_0$ 初始化为 0，会导致早期估计偏向零，因此需进行修正：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

作用：修正初期估计的偏差，使动量与方差估计更准确。例如在初始 $t=1$ 时，$m_t = (1-\beta_1)g_t$，偏差校正后 $\hat{m}_t = g_t$。当 $t$ 变大时，$1 - \beta_1^t \approx 1$，校正效果消失。

# 五、核心作用

**一阶矩估计的作用：**
- 加速收敛：通过动量机制保留历史梯度方向，减少震荡，使参数更新更稳定
- 捕捉梯度趋势：在非凸优化问题中，帮助模型避开局部极小值，向全局最优方向移动

**二阶矩估计的作用：**
- 自适应学习率：根据梯度方差调整步长。梯度变化大时，学习率自动减小（因 $v_t$ 较大），防止震荡；梯度变化小时，学习率增大，加快收敛
- 处理稀疏梯度：对稀疏数据（如自然语言处理任务）中的低频参数分配更大更新步长，提升训练效率

# 六、模型参数更新公式

最终参数更新公式结合一阶矩和二阶矩的修正估计：

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

- $\alpha$：基础学习率，控制整体步长
- $\epsilon$：极小常数（如 $10^{-8}$），防止分母为零

# 七、与其他优化器的对比

| 特性 | Adam | AdamW | SGD/Momentum | RMSprop |
|------|------|-------|-------------|---------|
| 动量机制 | 一阶矩估计平滑梯度方向 | 同 Adam | 仅保留动量项 | 仅依赖梯度平方平均 |
| 自适应学习率 | 二阶矩估计动态调整步长 | 同 Adam | 固定学习率 | 类似二阶矩但无偏差校正 |
| 权重衰减 | L2正则（耦合） | 解耦权重衰减 | 需手动实现 | L2正则（耦合） |
| 计算复杂度 | 中等 | 中等 | 低 | 中等 |
| 泛化性能 | 较差 | 好 | 好 | 中等 |
| 适用场景 | 非凸优化、稀疏数据 | 大模型训练 | 小规模凸优化 | 非平稳目标函数 |

# 八、Python 代码对比

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np

X, y = make_classification(n_samples=2000, n_features=50, random_state=42)
X_train, X_test = torch.FloatTensor(X[:1600]), torch.FloatTensor(X[1600:])
y_train, y_test = torch.LongTensor(y[:1600]), torch.LongTensor(y[1600:])

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=64, shuffle=True
)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_and_evaluate(optimizer_class, optimizer_kwargs, name):
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    for epoch in range(20):
        model.train()
        for batch_x, batch_y in train_loader:
            loss = criterion(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).numpy()
    acc = accuracy_score(y_test.numpy(), preds)
    print(f"{name}: Test Accuracy = {acc:.4f}")

train_and_evaluate(
    torch.optim.Adam,
    {"lr": 0.001, "weight_decay": 0.01},
    "Adam (L2正则)"
)
train_and_evaluate(
    torch.optim.AdamW,
    {"lr": 0.001, "weight_decay": 0.01},
    "AdamW (解耦权重衰减)"
)
train_and_evaluate(
    torch.optim.SGD,
    {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.01},
    "SGD + Momentum"
)
```

# 九、常见问题与面试追问

1. **为什么大模型训练都使用 AdamW 而不是 Adam？** 大模型（如 GPT、LLaMA）训练需要稳定的权重衰减来实现正则化。Adam 中 L2 正则化的效果因自适应学习率的干扰而不稳定，AdamW 的解耦权重衰减提供了更可预测和有效的正则化。

2. **Adam 的学习率如何设置？** 常用初始值为 $3 \times 10^{-4}$（所谓"Adam 默认学习率"）。大模型预训练通常使用学习率预热（Warmup）+余弦衰减策略。预热阶段将学习率从 0 线性增加到峰值，避免训练初期的不稳定。

3. **AdamW 的 weight_decay 如何选择？** 常见范围为 $0.01 \sim 0.1$。Transformer 模型中通常对所有参数施加统一的 weight_decay，但对偏置项和 LayerNorm 参数通常不施加（设为 0）。

4. **Adam 和 SGD 各自的优缺点？** Adam 收敛快、对超参数不敏感、适合稀疏数据，但泛化性能可能不如 SGD。SGD 泛化性能好但收敛慢、需要仔细调参。实践中，预训练阶段常用 AdamW（快速收敛），微调阶段可尝试 SGD（更好的泛化）。
