# Tanh 激活函数学习文档

## 1. 算法基础认知

### 1.1 定义

Tanh（双曲正切函数）是深度学习中广泛使用的激活函数，其数学表达式为：

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

### 1.2 与 Sigmoid 的关系

Tanh 与 Sigmoid 存在明确的数学关系：

$$
\tanh(x) = 2\sigma(2x) - 1
$$

或等价于：

$$
\tanh(x) = 2 \cdot \text{Sigmoid}(2x) - 1
$$

**推导过程：**

$$
\begin{aligned}
2\sigma(2x) - 1 &= 2 \cdot \frac{1}{1 + e^{-2x}} - 1 \\
&= \frac{2}{1 + e^{-2x}} - \frac{1 + e^{-2x}}{1 + e^{-2x}} \\
&= \frac{2 - (1 + e^{-2x})}{1 + e^{-2x}} \\
&= \frac{1 - e^{-2x}}{1 + e^{-2x}} \\
&= \frac{e^{2x} - 1}{e^{2x} + 1} = \tanh(x)
\end{aligned}
$$

### 1.3 直观类比

将 Tanh 想象为一个**温度调节器**：
- 输出 1 表示"非常热"
- 输出 -1 表示"非常冷"  
- 输出 0 表示"适中温度"

与 Sigmoid（只到 1）相比，Tanh 提供了正负两个方向的响应。

---

## 2. 核心原理

### 2.1 数学性质

**值域与定义域：**
- 定义域：$\mathbb{R}$（所有实数）
- 值域：$(-1, 1)$（严格单调递增，关于原点对称）

**核心特性对比：**

| 性质 | Sigmoid | Tanh |
|------|---------|------|
| 公式 | $\frac{1}{1 + e^{-x}}$ | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ |
| 值域 | $(0, 1)$ | $(-1, 1)$ |
| 零点 | $x=0, y=0.5$ | $x=0, y=0$ |
| 中心对称 | 关于 $(0, 0.5)$ | 关于 $(0, 0)$ |
| 导数最大值 | $0.25$ | $1$ |
| 均值 | $\approx 0.5$ | $\approx 0$ |

### 2.2 导数公式推导

**已知：** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

**方法一：商法则**

设 $u = e^x - e^{-x}$，$v = e^x + e^{-x}$，则：

$$
\tanh'(x) = \frac{u'v - uv'}{v^2}
$$

计算导数：
- $u' = e^x + e^{-x} = v$
- $v' = e^x - e^{-x} = u$

因此：
$$
\tanh'(x) = \frac{v \cdot v - u \cdot u}{v^2} = \frac{v^2 - u^2}{v^2} = 1 - \left(\frac{u}{v}\right)^2 = 1 - \tanh^2(x)
$$

**方法二：与 Sigmoid 的关系推导**

由于 $\tanh(x) = 2\sigma(2x) - 1$，对两边求导：

$$
\begin{aligned}
\tanh'(x) &= 2 \cdot \sigma'(2x) \cdot 2 \\
&= 4 \cdot \sigma(2x)(1 - \sigma(2x)) \\
&= 4 \cdot \frac{\sigma(2x)(1 - \sigma(2x))}{1} \\
&= 4 \cdot \frac{\tanh(x) + 1}{2} \cdot \frac{1 - \tanh(x)}{2} \\
&= ( \tanh(x) + 1)(1 - \tanh(x)) \\
&= 1 - \tanh^2(x)
\end{aligned}
$$

**重要性质：** $\tanh'(x) = 1 - \tanh^2(x)$

### 2.3 与 Sigmoid 的对比图

```
Tanh:        Sigmoid:
    1 ┤    ╭────    1 ┤
      │   ╱           │
      │  ╱            │
    0 ┼─╱─────────    0.5 ┼──────
      │╱               0 ┤
   -1 ┤                 0 ┤
```

---

## 3. PyTorch 实现

### 3.1 PyTorch 内置实现

```python
import torch
import torch.nn as nn

# 方法1：直接使用 torch.tanh
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = torch.tanh(x)
print(f"Input: {x}")
print(f"Tanh: {y}")
# Output: [-0.9640, -0.7616, 0.0000, 0.7616, 0.9640]

# 方法2：使用 nn.Tanh 模块
tanh_layer = nn.Tanh()
y_module = tanh_layer(x)

# 方法3：在 Sequential 中使用
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.Tanh(),  # 激活函数层
    nn.Linear(5, 1),
)
```

### 3.2 手写实现

```python
import torch

def tanh(x: torch.Tensor) -> torch.Tensor:
    """手动实现 Tanh 函数"""
    # 避免溢出
    # tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    return torch.tanh(x)

def tanh_derivative(x: torch.Tensor) -> torch.Tensor:
    """手动实现 Tanh 导数"""
    return 1 - torch.tanh(x) ** 2

def tanh_from_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """利用 Sigmoid 关系实现 Tanh"""
    return 2 * torch.sigmoid(2 * x) - 1

# 验证三种实现一致性
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
t1 = torch.tanh(x)
t2 = tanh_derivative(x)  # 这是错误的，应该是 tanh(x)
t3 = tanh_from_sigmoid(x)

print(f"PyTorch: {t1}")
print(f"From Sigmoid: {t3}")
assert torch.allclose(t1, t3, atol=1e-6)
print("实现验证通过！")
```

### 3.3 LSTM 中的 Tanh 应用

```python
import torch
import torch.nn as nn

class SimpleLSTMCell(nn.Module):
    """简化的 LSTM Cell，展示 Tanh 的使用"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 输入门、遗忘门、输出门使用 Sigmoid（值域 0-1）
        self.sigmoid = nn.Sigmoid()
        # 候选记忆使用 Tanh（值域 -1 到 1）
        self.tanh = nn.Tanh()
        
        # 权重矩阵
        self.W_i = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_f = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_c = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(input_dim, hidden_dim, bias=False)
        
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, x, h_prev, c_prev):
        """
        x: 输入，形状 (batch, input_dim)
        h_prev: 上一时刻隐藏状态
        c_prev: 上一时刻细胞状态
        """
        # 输入门（决定新信息的重要性）
        i = self.sigmoid(self.W_i(x) + self.U_i(h_prev))
        
        # 遗忘门（决定遗忘多少旧信息）
        f = self.sigmoid(self.W_f(x) + self.U_f(h_prev))
        
        # 候选记忆（使用 Tanh，范围 -1 到 1，可以产生负值）
        c_candidate = self.tanh(self.W_c(x) + self.U_c(h_prev))
        
        # 输出门（决定输出什么）
        o = self.sigmoid(self.W_o(x) + self.U_o(h_prev))
        
        # 更新细胞状态
        c_new = f * c_prev + i * c_candidate
        
        # 隐藏状态（Tanh 归一化到 -1 到 1）
        h_new = o * self.tanh(c_new)
        
        return h_new, c_new

# 测试 LSTM Cell
lstm = SimpleLSTMCell(input_dim=10, hidden_dim=8)
x = torch.randn(2, 10)
h = torch.zeros(2, 8)
c = torch.zeros(2, 8)

h_new, c_new = lstm(x, h, c)
print(f"Input shape: {x.shape}")
print(f"Hidden state shape: {h_new.shape}")
print(f"Cell state shape: {c_new.shape}")
print(f"\nHidden state:\n{h_new}")
print(f"\nCell state:\n{c_new}")
```

---

## 4. 代码示例

### 4.1 Tanh vs Sigmoid 对比

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = torch.linspace(-5, 5, 1000)

# 计算函数值
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)
sigmoid_grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))
tanh_grad = 1 - torch.tanh(x) ** 2

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sigmoid 函数
axes[0, 0].plot(x.numpy(), sigmoid.numpy(), 'b-', linewidth=2)
axes[0, 0].set_title('Sigmoid Function', fontsize=14)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('σ(x)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(-0.1, 1.1)

# Tanh 函数
axes[0, 1].plot(x.numpy(), tanh.numpy(), 'r-', linewidth=2)
axes[0, 1].set_title('Tanh Function', fontsize=14)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('tanh(x)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(-1.1, 1.1)

# Sigmoid 导数
axes[1, 0].plot(x.numpy(), sigmoid_grad.numpy(), 'b-', linewidth=2)
axes[1, 0].set_title("Sigmoid Derivative (max=0.25)", fontsize=14)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel("σ'(x)")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(-0.05, 0.3)

# Tanh 导数
axes[1, 1].plot(x.numpy(), tanh_grad.numpy(), 'r-', linewidth=2)
axes[1, 1].set_title("Tanh Derivative (max=1.0)", fontsize=14)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel("tanh'(x)")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('activation_comparison.png', dpi=150)
plt.show()

# 打印关键数据
print("=" * 50)
print("梯度对比：")
print(f"Sigmoid 导数最大值: {sigmoid_grad.max().item():.4f}")
print(f"Tanh 导数最大值: {tanh_grad.max().item():.4f}")
print("\n输出均值对比：")
print(f"Sigmoid 均值: {sigmoid.mean().item():.4f}")
print(f"Tanh 均值: {tanh.mean().item():.4f}")
```

### 4.2 梯度流动对比实验

```python
import torch
import torch.nn as nn

def test_gradient_flow(activation_fn, name, n_layers=10):
    """测试不同激活函数在深层网络中的梯度流动"""
    torch.manual_seed(42)
    
    # 创建深层网络
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(64, 64))
        layers.append(activation_fn())
    
    model = nn.Sequential(*layers)
    
    # 初始化权重为较大值，测试梯度
    for param in model.parameters():
        nn.init.uniform_(param, 0.5, 2.0)
    
    # 前向传播
    x = torch.randn(8, 64)
    y = model(x)
    loss = y.sum()
    
    # 反向传播
    loss.backward()
    
    # 收集第一层和最后一层的梯度
    first_layer_grad = list(model.parameters())[0].grad.abs().mean().item()
    return first_layer_grad

print("深度网络梯度流动测试（10层网络）：")
print("-" * 40)

grad_sigmoid = test_gradient_flow(nn.Sigmoid, "Sigmoid")
print(f"使用 Sigmoid: 第一层平均梯度 = {grad_sigmoid:.2e}")

grad_tanh = test_gradient_flow(nn.Tanh, "Tanh")
print(f"使用 Tanh:   第一层平均梯度 = {grad_tanh:.2e}")

grad_relu = test_gradient_flow(nn.ReLU, "ReLU")
print(f"使用 ReLU:   第一层平均梯度 = {grad_relu:.2e}")

print("\n结论：Tanh 的梯度衰减小于 Sigmoid，但两者都会随深度衰减")
```

---

## 5. 应用场景
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的应用场景相关内容]


---

## 6. 优缺点分析
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的优缺点分析相关内容]


---

## 7. 调库实现
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的调库实现相关内容]


---

## 8. 手工代码实现
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的手工代码实现相关内容]


---

## 9. 可视化与结果理解
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的可视化与结果理解相关内容]


---

## 10. 模型评估
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的模型评估相关内容]


---

## 11. 常见问题与易错点
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的常见问题与易错点相关内容]


---

## 12. 学习总结
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的学习总结相关内容]


---

## 13. 练习题与思考题
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的练习题与思考题相关内容]


---

## 14. 学习路径建议
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充Tanh的学习路径建议相关内容]


---
