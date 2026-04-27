# Sigmoid 激活函数学习文档

## 1. 算法基础认知

### 1.1 定义

Sigmoid（又称 Logistic 函数）是深度学习中最早使用的激活函数之一，其数学表达式为：

$$
\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}
$$

### 1.2 直观类比

将 Sigmoid 想象为一个**概率转换器**：它将任意实数映射到 $(0, 1)$ 区间，就像将温度转换为"炎热概率"。当输入为 0 时，输出恰好为 0.5（中性概率）。

### 1.3 历史背景

- 1990年代至 2010年代初，Sigmoid 是神经网络的主导激活函数
- 常用于二分类任务的输出层（输出解释为概率）
- 在早期的 RNN 和早期深度网络中广泛使用

---

## 2. 核心原理

### 2.1 数学性质

**值域与定义域：**
- 定义域：$\mathbb{R}$（所有实数）
- 值域：$(0, 1)$（严格单调递增）

**核心特性：**

| 性质 | 表达式 | 说明 |
|------|--------|------|
| 单调性 | $\sigma'(x) > 0$ | 处处可导，单调递增 |
| 对称性 | $\sigma(-x) = 1 - \sigma(x)$ | 关于点 $(0, 0.5)$ 中心对称 |
| 不变性 | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ | 导数可以用自身表示 |
| 输出中心 | 均值约为 0.5（未数据中心化） | 可能导致梯度更新方向一致 |

### 2.2 导数公式推导

已知：
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

对 $x$ 求导：
$$
\sigma'(x) = \frac{d}{dx}\left(1 + e^{-x}\right)^{-1}
$$

使用链式法则：
$$
\sigma'(x) = -1 \cdot (1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}
$$

进一步化简：
$$
\sigma'(x) = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x)(1 - \sigma(x))
$$

**重要性质**：Sigmoid 的导数可以用函数自身表示，这在计算上非常高效。

### 2.3 工作流程

```
输入 x ∈ (-∞, +∞)
    ↓
计算 σ(x) = 1 / (1 + e^(-x))
    ↓
输出 y ∈ (0, 1)
```

---

## 3. PyTorch 实现

### 3.1 PyTorch 内置实现

```python
import torch
import torch.nn as nn

# 方法1：直接使用 torch.sigmoid
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = torch.sigmoid(x)
print(f"Input: {x}")
print(f"Sigmoid: {y}")
# Output: [0.1192, 0.2689, 0.5000, 0.7311, 0.8808]

# 方法2：使用 nn.Sigmoid 模块
sigmoid_layer = nn.Sigmoid()
y_module = sigmoid_layer(x)

# 方法3：在 Sequential 中使用
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.Sigmoid(),  # 激活函数层
    nn.Linear(5, 1),
    nn.Sigmoid()   # 输出层，输出作为概率
)
```

### 3.2 手写实现

```python
import torch

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """手动实现 Sigmoid 函数"""
    return 1.0 / (1.0 + torch.exp(-x))

def sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
    """手动实现 Sigmoid 导数（用于反向传播）"""
    s = sigmoid(x)
    return s * (1 - s)

# 验证实现正确性
x = torch.tensor([0.5], requires_grad=True)
y = sigmoid(x)
y.backward()

print(f"Forward: {y.item():.6f}")
print(f"Gradient: {x.grad.item():.6f}")

# 与 PyTorch 内置比较
x_test = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
assert torch.allclose(sigmoid(x_test), torch.sigmoid(x_test))
print("手写实现验证通过！")
```

### 3.3 二分类场景的完整实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SigmoidBinaryClassifier(nn.Module):
    """使用 Sigmoid 的二分类器"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # 线性变换
        linear_out = self.linear(x)
        # Sigmoid 将输出映射到 (0, 1) 区间
        prob = torch.sigmoid(linear_out)
        return prob

def train_binary_classifier():
    # 生成模拟数据（二分类）
    torch.manual_seed(42)
    n_samples = 200
    
    # 类别0的样本
    X0 = torch.randn(n_samples, 2) - torch.tensor([2.0, 2.0])
    y0 = torch.zeros(n_samples, 1)
    
    # 类别1的样本
    X1 = torch.randn(n_samples, 2) + torch.tensor([2.0, 2.0])
    y1 = torch.ones(n_samples, 1)
    
    # 合并数据
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([y0, y1], dim=0)
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 模型、优化器、损失函数
    model = SigmoidBinaryClassifier(input_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    
    # 训练
    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            prob = model(batch_X)
            loss = criterion(prob, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # 预测
    model.eval()
    with torch.no_grad():
        test_point = torch.tensor([[0.0, 0.0]])
        prob = model(test_point)
        print(f"Test point [0, 0]: probability = {prob.item():.4f}")
        print(f"Predicted class: {1 if prob > 0.5 else 0}")

train_binary_classifier()
```

---

## 4. 代码示例

### 4.1 Sigmoid 在神经网络中的作用

```python
import torch
import torch.nn as nn

# Sigmoid 作为门控机制
class SigmoidGate(nn.Module):
    """模拟 LSTM 中的门控机制"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden):
        """
        x: 输入特征
        hidden: 隐藏状态
        """
        # 拼接输入和隐藏状态
        combined = torch.cat([x, hidden], dim=-1)
        
        # 计算遗忘门（决定保留多少旧信息）
        # 使用 Sigmoid 确保输出在 (0, 1) 区间
        # 输出接近 1: 保留旧信息；输出接近 0: 遗忘旧信息
        forget_gate = self.sigmoid(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)(combined)
        )
        
        return forget_gate

# 验证门控机制
gate = SigmoidGate(hidden_dim=4)
x = torch.randn(2, 4)  # batch_size=2, input_dim=4
hidden = torch.randn(2, 4)  # batch_size=2, hidden_dim=4

forget = gate(x, hidden)
print(f"Forget gate output shape: {forget.shape}")
print(f"Forget gate values:\n{forget}")
print(f"Values range: [{forget.min():.4f}, {forget.max():.4f}]")
```

### 4.2 梯度可视化

```python
import torch
import matplotlib.pyplot as plt

# 可视化 Sigmoid 和其导数
x = torch.linspace(-10, 10, 1000)
sigmoid_values = torch.sigmoid(x)
sigmoid_grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), sigmoid_values.numpy(), 'b-', label='Sigmoid(x)', linewidth=2)
plt.plot(x.numpy(), sigmoid_grad.numpy(), 'r-', label="Sigmoid'(x)", linewidth=2)
plt.axhline(y=0.25, color='g', linestyle='--', alpha=0.5, label='Max gradient = 0.25')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Sigmoid Function and Its Derivative')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.savefig('sigmoid_visualization.png', dpi=150)
plt.show()
```

---

## 5. 应用场景

### 5.1 二分类输出层

在神经网络中，Sigmoid 常用于二分类问题的输出层：

$$
\hat{y} = \sigma(w^T x + b) = P(y=1|x)
$$

决策边界为 $\hat{y} > 0.5$。

### 5.2 门控机制

LSTM 和 GRU 中的门控单元使用 Sigmoid：

- **遗忘门** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- **输入门** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- **输出门** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

### 5.3 概率建模

变分自编码器（VAE）中，隐变量先验通常假设为高斯分布，但可以使用 Sigmoid 二元分布。

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 输出有界 | 始终在 $(0, 1)$ 区间，便于解释为概率 |
| 可导性 | 处处可导，梯度计算稳定 |
| 形式简单 | 数学表达式简洁，易于实现 |
| 概率解释 | 自然对应bernoulli分布的对数几率 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 梯度消失 | $\|x\| > 4$ 时，导数接近 0 | 使用 ReLU/Leaky ReLU |
| 非零中心 | 输出均值 0.5，梯度总是正 | 批归一化 |
| 计算开销 | 指数运算较慢 | 使用查表近似 |
| 梯度饱和 | 两侧梯度接近 0 | 组合激活函数 |

### 6.3 梯度消失问题分析

对于深度网络，当 $x$ 很大或很小时：

$$
\sigma'(x) = \sigma(x)(1-\sigma(x)) \leq 0.25
$$

连乘多个小于 0.25 的数会导致梯度指数级衰减：

$$
\frac{\partial L}{\partial w_1} = \prod_{i=1}^{n} \sigma'(x_i) \cdot \frac{\partial L}{\partial x_{n+1}}
$$

---

## 7. 调库实现

```python
# scikit-learn 实现（逻辑回归内置 Sigmoid）
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 生成二分类数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测概率
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
```

---

## 8. 手工代码实现

```python
import numpy as np

class ManualSigmoid:
    """手动实现 Sigmoid 及其导数"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid 函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid 导数"""
        s = ManualSigmoid.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
        """二元交叉熵损失"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def gradient(y_true, y_pred, X):
        """计算梯度"""
        n = len(y_true)
        error = y_pred - y_true
        return (1/n) * (X.T @ error)

# 验证实现
model = ManualSigmoid()

x = np.array([-2, -1, 0, 1, 2])
y = model.sigmoid(x)
y_deriv = model.sigmoid_derivative(x)

print("Input:", x)
print("Sigmoid:", y)
print("Derivative:", y_deriv)

# 验证导数关系
assert np.allclose(y_deriv, y * (1 - y))
print("导数关系验证通过！")
```

---

## 9. 可视化与结果理解

### 9.1 Sigmoid 曲线特征

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 图1: Sigmoid 函数
x = np.linspace(-10, 10, 1000)
y = 1 / (1 + np.exp(-x))
axes[0].plot(x, y, 'b-', linewidth=2)
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[0].set_title(r'Sigmoid: $\sigma(x) = \frac{1}{1+e^{-x}}$')
axes[0].set_xlabel('x')
axes[0].set_ylabel(r'$\sigma(x)$')
axes[0].grid(True, alpha=0.3)

# 图2: Sigmoid 导数
y_deriv = y * (1 - y)
axes[1].plot(x, y_deriv, 'r-', linewidth=2)
axes[1].axhline(y=0.25, color='g', linestyle='--', alpha=0.5)
axes[1].set_title(r"Sigmoid Derivative: $\sigma'(x) = \sigma(x)(1-\sigma(x))$")
axes[1].set_xlabel('x')
axes[1].set_ylabel(r"$\sigma'(x)$")
axes[1].grid(True, alpha=0.3)

# 图3: 梯度流
axes[2].bar(['$\\sigma\'(0)=0.25$', '$\\sigma\'(2)≈0.1$', '$\\sigma\'(4)≈0.02$'], 
           [0.25, 0.1, 0.02], color=['green', 'orange', 'red'])
axes[2].set_title('Gradient Vanishing')
axes[2].set_ylabel('Gradient Value')

plt.tight_layout()
plt.savefig('sigmoid_analysis.png', dpi=150)
plt.show()
```

---

## 10. 模型评估

### 10.1 二分类评估指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import numpy as np

def evaluate_binary_classifier(y_true, y_pred, y_prob):
    """评估二分类器"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_prob),
    }
    
    print("=== Binary Classifier Evaluation ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    return metrics

# 示例
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
y_prob = np.array([0.2, 0.3, 0.8, 0.4, 0.1, 0.9, 0.6, 0.7])

evaluate_binary_classifier(y_true, y_pred, y_prob)
```

---

## 11. 常见问题与易错点

### 11.1 梯度消失

**问题**：当 $|x|$ 较大时，Sigmoid 导数接近 0，导致深层网络无法训练。

**解决方案**：
- 使用 ReLU 替代 Sigmoid
- 使用残差连接
- 使用批归一化

### 11.2 概率输出理解

**问题**：Sigmoid 输出不是严格的概率分布（不满足互斥事件）。

**解释**：
- $P(y=1|x) = \sigma(x)$
- $P(y=0|x) = 1 - \sigma(x) = \sigma(-x)$
- 两者之和为 1，符合 Bernoulli 分布

### 11.3 数值稳定性

**问题**：当 $x$ 很大时，$e^{-x}$ 可能下溢。

**解决方案**：
```python
def stable_sigmoid(x):
    """数值稳定的 Sigmoid"""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )
```

---

## 12. 学习总结

### 12.1 核心要点

1. **函数定义**：$\sigma(x) = \frac{1}{1+e^{-x}}$
2. **导数形式**：$\sigma'(x) = \sigma(x)(1-\sigma(x))$
3. **值域范围**：$(0, 1)$，可解释为概率
4. **梯度特性**：最大值为 0.25，容易梯度消失

### 12.2 与其他激活函数对比

| 激活函数 | 值域 | 导数最大值 | 适用场景 |
|----------|------|------------|----------|
| Sigmoid | $(0, 1)$ | 0.25 | 二分类输出 |
| Tanh | $(-1, 1)$ | 1.0 | 隐藏层 |
| ReLU | $[0, +\infty)$ | 1.0 | 深度网络 |
| Leaky ReLU | $(-\infty, +\infty)$ | 1.0 | 缓解死亡神经元 |

### 12.3 学��路��

1. 理解 Sigmoid 数学定义和性质
2. 掌握 PyTorch 实现
3. 了解梯度消失问题
4. 对比其他激活函数

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：计算 $\sigma(0)$ 和 $\sigma'(0)$。

**答案**：
- $\sigma(0) = \frac{1}{1+e^0} = \frac{1}{2} = 0.5$
- $\sigma'(0) = \sigma(0)(1-\sigma(0)) = 0.5 \times 0.5 = 0.25$

**练习2**：证明 $\sigma(-x) = 1 - \sigma(x)$。

**答案**：
$$
\sigma(-x) = \frac{1}{1+e^x} = \frac{e^{-x}}{e^{-x} + 1} = 1 - \frac{1}{1+e^{-x}} = 1 - \sigma(x)
$$

### 13.2 编程实践

**练习3**：实现一个多层感知机，使用 Sigmoid 作为激活函数，在 mnist 数据集上训练。

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义网络
class MLPSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 10),
            nn.Sigmoid()  # 注意：通常最后一层不用激活
        )
    
    def forward(self, x):
        return self.net(x)

# 训练代码（省略数据加载）
# model = MLPSigmoid()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()
# 训练循环...
```

### 13.3 思考题

**思考题**：为什么 Sigmoid 不适合作为深度网络的隐层激活函数？

**答案**：
1. 梯度消失：由于导数最大值为 0.25，深层网络反向传播时梯度指数级衰减
2. 计算开销：涉及指数运算，计算较慢
3. 非零中心：输出均值 0.5，导致梯度总是同方向更新

---

## 14. 学习路径建议

### 14.1 第一阶段：基础概念（1-2天）

1. 理解 Sigmoid 数学定义
2. 掌握导数推导
3. 了解基本性质

### 14.2 第二阶段：实现与实践（2-3天）

1. PyTorch 实现
2. 手写实现
3. 训练简单模型

### 14.3 第三阶段：深入理解（3-5天）

1. 梯度消失问题分析
2. 与其他激活函数对比
3. 应用场景拓展

### 14.4 推荐资源

- **书籍**：《深度学习》花书第6章
- **论文**：《Learning Representations by Back-propagating Errors》
- **课程**：CS231n 神经网络基础

---

*Sigmoid 激活函数是深度学习历史上的重要里程碑，虽然现在更多用于输出层，但其数学美感至今仍值得我们学习。*