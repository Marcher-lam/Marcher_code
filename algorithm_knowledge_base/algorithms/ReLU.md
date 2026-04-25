# ReLU 激活函数学习文档

> 深度学习中最广泛使用的激活函数，简单高效的非线性变换

---

## 1. 算法基础认知

### 1.1 一句话定义

ReLU(Rectified Linear Unit)是分段线性函数，将负数置为0，正数保持不变，是深度学习CNN的标配激活函数。

### 1.2 直觉类比

ReLU就像"红绿灯"——红灯停（输出0），绿灯行（保持输入）。简单高效，让神经网络"活过来"！

想象：
- 输入 -2 → 输出 0（"负数不要"）
- 输入 3 → 输出 3（"正数保留"）

### 1.3 发展背景

- 2012年：Hinton在论文中推广ReLU Restricted Boltzmann Machines
- 2014年：Srivastava提出Dropout
- 2015年后：成为CNN标配
- 2017年后：Transformer也用ReLU/GELU

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 激活函数 |
| 公式 | $f(x) = max(0, x)$ |
| 非线性 | 分段线性 |
| 计算复杂度 | O(1) |

---

## 2. 核心原理

### 2.1 公式

$$ReLU(x) = max(0, x) = \begin{cases} 0 & x < 0 \\ x & x \ge 0 \end{cases}$$

### 2.2 导数（梯度）

$$ReLU'(x) = \begin{cases} 0 & x < 0 \\ 1 & x > 0 \end{cases}$$

注意：$x=0$处不可导，通常设为0或1。PyTorch默认设为0（亚梯度0）。

### 2.3 梯度消失对比

为什么ReLU没有梯度消失？

| 函数 | 导数范围 | 深度问题 |
|------|----------|----------|
| Sigmoid | (0, 0.25] | 严重消失 |
| Tanh | (0, 1] | 中等消失 |
| **ReLU** | {0, 1} | **无** |

---

## 3. 数学推导

### 3.1 梯度流

正向传播：$y = ReLU(x)$

反向传播：$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot ReLU'(x)$

对于正数区域，梯度=1，直接传递！深层网络信号可以无衰减传播。

### 3.2 期望稀疏度

对于随机输入（均值为0的高斯分布），约50%的值会是负数：

$$\mathbb{P}(x < 0) \approx 0.5$$

这产生了天然的稀疏表示。

### 3.3 期望稀疏度验证

```python
import torch

# 统计稀疏性
x = torch.randn(1000, 512)
y = torch.relu(x)
sparsity = (y == 0).float().mean()
print(f"稀疏度: {sparsity:.2%}")  # 约50%
```

### 3.4 梯度对比可视化

```python
import numpy as np

x = np.linspace(-5, 5, 100)

# Sigmoid梯度
sigmoid_grad = 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))

# ReLU梯度
relu_grad = (x > 0).astype(float)

print(f"Sigmoid梯度: max={sigmoid_grad.max():.3f}, min={sigmoid_grad.min():.3f}")
print(f"ReLU梯度: {relu_grad.max()}, {relu_grad.min()}")
```

---

## 4. PyTorch实现

### 4.1 函数形式

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 函数调用
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = F.relu(x)
print(output)  # tensor([0., 0., 0., 1., 2.])

# inplace 版本（节省内存）
output_inplace = F.relu_(x.clone())
print(output_inplace)  # tensor([0., 0., 0., 1., 2.])
```

### 4.2 Module形式

```python
# nn.ReLU
relu = nn.ReLU()
output = relu(x)
print(output)  # tensor([0., 0., 0., 1., 2.])

# 参数查看
print(relu.inplace)  # False (默认)

# inplace 版本
relu_inplace = nn.ReLU(inplace=True)
relu_inplace(x)
```

### 4.3 梯度查看

```python
x = torch.tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)
y = F.relu(x)
loss = y.sum()
loss.backward()

print(x.grad)  # tensor([0., 0., 1., 1.])
# 负数区域的梯度为0，正数区域的梯度为1
```

---

## 5. 代码示例

### 5.1 MLP中使用

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)
        return x

# 使用示例
model = MLP(784, 256, 10)
x = torch.randn(32, 784)
output = model(x)
print(f"输出形状: {output.shape}")  # [32, 10]
```

### 5.2 CNN中使用

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64*8*8, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)  # 激活函数
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 使用示例
model = CNN()
x = torch.randn(4, 3, 32, 32)
output = model(x)
print(f"输出形状: {output.shape}")  # [4, 10]
```

### 5.3 完整训练

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 数据
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 模型
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 6. 变体

### 6.1 LeakyReLU

解决"神经元死亡"问题：

```python
# Leaky ReLU: 负数给一个小斜率
output = F.leaky_relu(x, negative_slope=0.01)
# x = -2 → output = -0.02

leaky_relu = nn.LeakyReLU(negative_slope=0.01)
output = leaky_relu(x)
```

公式：
$$LeakyReLU(x) = \begin{cases} x & x > 0 \\ 0.01x & x \le 0 \end{cases}$$

### 6.2 PReLU

可学习的斜率：

```python
prelu = nn.PReLU(num_parameters=1)
print(prelu.weight)  # 可学习参数 tensor([0.25])

output = prelu(x)
# 训练中斜率会自动更新
```

### 6.3 ELU

指数线性单位：

```python
output = F.elu(x)
# 对于负数：e^x - 1
# 特点：输出均值接近0

elu = nn.ELU()
output = elu(x)
```

公式：
$$ELU(x) = \begin{cases} x & x > 0 \\ e^x - 1 & x \le 0 \end{cases}$$

### 6.4 GELU（Transformer标配）

```python
output = F.gelu(x)
# 高斯误差线性单元

gelu = nn.GELU()
output = gelu(x)
```

近似公式：
$$\text{GELU}(x) = x \cdot \Phi(x)$$

---

## 7. 常见问题

### Q1: ReLU的"神经元死亡"是什么？

- 现象：部分神经元输出恒为0
- 原因：负数梯度=0，参数不再更新
- 解决：使用LeakyReLU/ELU/GELU

### Q2: 为什么CNN用ReLU？

- 梯度保持好，不消失
- 计算极快（比较操作）
- 产生稀疏表示

### Q3: ReLU可以用于输出层吗？

- 不适合！输出可能为0
- 输出层：
  - 多分类 → Softmax
  - 二分类 → Sigmoid

### Q4: 为什么ResNet用ReLU？

- 与残差连接配合好
- 正区域梯度恒为1，信号无衰减

---

## 8. 学习路径

### 8.1 激活函数选择

```
二分类 → Sigmoid
多分类 → Softmax
CNN → ReLU/GELU
Transformer → GELU
需要平滑 → ELU
```

### 8.2 变体选择

| 场景 | 推荐 |
|------|------|
| 标准CNN | ReLU |
| 防止死亡 | LeakyReLU |
| Transformer | GELU |
| 快速部署 | ReLU |

---

## 9. 练习题

### 选择题

1. ReLU(-5)输出？
   - A) -5   B) 0   C) 1
   - **答案：B（0）**

2. ReLU在x=0处导数？
   - A) 0   B) 1   C) 不存在/任意
   - **答案：C（通常设为0）**

3. ReLU和Sigmoid的主要区别？
   - A) ReLU是线性的
   - B) ReLU不会梯度消失
   - C) ReLU计算快
   - **答案：B+C**

### 简答题

1. 解释ReLU的稀疏性优势？

   **答案**：负数置0产生稀疏表示，减少过拟合，提高计算效率。神经网络可以自动学习有用的特征，忽略噪声。

2. 为什么ResNet用ReLU不用Sigmoid？

   **答案**：ResNet需要梯度畅通传递，Sigmoid梯度在深层会消失，ReLU梯度恒为1，信号无衰减。

### 编程题

实现ReLU并计算稀疏度：

```python
class ReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0)

# 统计稀疏度
x = torch.randn(10000)
relu = ReLU()
y = relu(x)
sparsity = (y == 0).float().mean()
print(f"稀疏度: {sparsity:.2%}")
```

---

## 10. 附录

### A. PyTorch版本演进

| 版本 | ReLU |
|------|-----|
| torch ≤ 0.1 | F.relu |
| modern | nn.ReLU |

### B. 参考

- Hinton et al. (2012). "Rectified Linear Units Improve Restricted Boltzmann Machines"
- Nair & Hinton (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines"

---

## 11. Softmax函数

### 11.1 Softmax公式

$$softmax(x)_i = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}$$

### 11.2 维度处理

```python
# 一维向量
x = torch.tensor([3.0, 1.0, -1.0])
output = F.softmax(x, dim=0)
print(output)  # tensor([0.6703, 0.2433, 0.0864])

# 二维批次 (dim=1)
x_batch = torch.tensor([[3.0, 1.0, -1.0], [2.0, 0.0, 1.0]])
output_batch = F.softmax(x_batch, dim=1)
```

### 11.3 数值稳定性

原始问题：$e^{100}$会溢出

解决：减去最大值

$$softmax(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, m = \max(x)$$

### 11.4 Log-Softmax

```python
log_probs = F.log_softmax(x, dim=0)
# 数值更稳定
```

---

## 12. GELU详解

### 12.1 公式

$$GELU(x) = x \cdot P(X \le x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的CDF。

### 12.2 近似

$$GELU(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$

### 12.3 实现

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

---

**文档结束**