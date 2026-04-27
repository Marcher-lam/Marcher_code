# 多层感知机 (MLP) 学习文档

> 用一句话说明这个算法的核心价值，不超过30字。

多层感知机是一种前馈神经网络，通过多层非线性变换学习复杂模式。

## 1. 算法基础认知

### 1.1 什么是MLP

多层感知机（Multi-Layer Perceptron, MLP）是最基础的前馈神经网络，由输入层、一个或多个隐藏层和输出层组成。每层由多个神经元组成，层与层之间全连接。

### 1.2 直觉类比

想象一个工厂的流水线：原材料（输入）进入，经过多个车间的处理（隐藏层），最终成为产品（输出）。每个车间都有自己的加工规则（激活函数），并且会把信息传递给下一个车间。

### 1.3 历史背景

MLP的概念可以追溯到1958年Frank Rosenblatt提出的感知机。但由于单层感知机无法解决异或问题，直到1986年Rumelhart等人提出反向传播算法，MLP才得以广泛应用。

### 1.4 算法定位

MLP是**监督学习**的**神经网络**，是深度学习的基础模块。

### 1.5 前置知识

- 线性代数
- 基础微积分（梯度）
- Python / NumPy

## 2. 核心原理

### 2.1 核心思想

MLP通过多层非线性变换学习复杂函数：
- 每层：线性变换 + 非线性激活
- 堆叠多层：学习更复杂的表示

### 2.2 工作流程

1. 前向传播：输入→隐藏层→输出
2. 计算损失
3. 反向传播：更新参数

### 2.3 关键概念

- 全连接层：每个神经元连接上一层的所有神经元
- 激活函数：ReLU, Sigmoid, Tanh
- Softmax：多分类输出

## 3. 数学公式

### 3.1 前向传播

对于一层：
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = f(z^{(l)})$$

### 3.2 损失函数

多分类交叉熵：
$$L = -sum_k y_k log(\hat{y}_k)$$

### 3.3 反向传播

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

## 4. 训练过程

### 4.1 超参数

| 超参数 | 作用 | 推荐范围 |
|--------|------|---------|
| hidden_size | 隐藏层维度 | 64-512 |
| num_layers | 隐藏层数量 | 1-5 |
| learning_rate | 学习率 | 0.001-0.1 |
| batch_size | 批次大小 | 32-256 |

## 5. 应用场景

1. 分类任务（MNIST图像分类）
2. 回归任务
3. 作为其他网络的组件

## 6. 优缺点

### 6.1 优点

- 可拟合任意函数
- 理论基础成熟

### 6.2 缺点

- 参数量大
- 难以捕捉序列/空间结构

## 7. 调库实现

使用PyTorch实现MLP：

```python
"""
多层感知机(MLP) - 使用PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("多层感知机(MLP)示例")
print("=" * 60)

# 1. 准备数据 - 模拟分类问题
print("\n准备训练数据...")

torch.manual_seed(42)
n_samples = 1000
n_features = 20
n_classes = 3

# 生成模拟数据
X = torch.randn(n_samples, n_features)
y = torch.randint(0, n_classes, (n_samples,))

# 分割数据集
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"训练集: {X_train.shape[0]}样本")
print(f"测试集: {X_test.shape[0]}样本")

# 2. 定义MLP模型
print("\n" + "-" * 40)
print("定义MLP模型")
print("-" * 40)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        
        layers = []
        for i in num_layers range(1):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())  # 激活函数
            layers.append(nn.Dropout(0.2))  # Dropout
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 创建模型
model = MLPClassifier(
    input_dim=n_features,
    hidden_dim=64,
    output_dim=n_classes,
    num_layers=2
)

print(f"模型结构:\n{model}")
print(f"\n模型参数: {sum(p.numel() for p in model.parameters())}")

# 3. 训练模型
print("\n" + "-" * 40)
print("训练模型")
print("-" * 40)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 20

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 验证
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

# 4. 评估
print("\n" + "-" * 40)
print("模型评估")
print("-" * 40)

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f"测试集准确率: {accuracy:.4f}")

print("\n" + "=" * 60)
print("示例完成")
print("=" * 60)
## 8. 手工代码实现（核心算法纯代码实现）

以下是MLP的纯手写实现，包含前向传播、全连接层、激活函数和反向传播：

```python
"""
多层感知机(MLP) - 纯手写实现
核心：全连接层、激活函数、反向传播
"""

import numpy as np

class Layer:
    """全连接层"""
    
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        参数：
        - input_dim: 输入维度
        - output_dim: 输出维度
        - activation: 激活函数类型
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Xavier初始化
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = np.random.randn(input_dim, output_dim) * scale
        self.b = np.zeros((1, output_dim))
        
        # 用于反向传播
        self.input_data = None
        self.output_data = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        """前向传播"""
        self.input_data = x
        # 线性变换
        z = np.matmul(x, self.W) + self.b
        # 激活函数
        a = self._activate(z)
        self.output_data = a
        return a
    
    def backward(self, da, learning_rate=0.01):
        """
        反向传播
        
        da: 上游传来的梯度
        """
        # 获取激活函数导数
        dz = da * self._activate_derivative(self.output_data)
        
        # 梯度
        self.dW = np.matmul(self.input_data.T, dz) / dz.shape[0]
        self.db = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]
        
        # 传回上游的梯度
        dx = np.matmul(dz, self.W.T)
        
        # 梯度裁剪
        clip = 1.0
        self.dW = np.clip(self.dW, -clip, clip)
        
        # 更新参数
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        
        return dx
    
    def _activate(self, x):
        """激活函数"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        else:
            return x
    
    def _activate_derivative(self, a):
        """激活函数导数"""
        if self.activation == 'relu':
            return (a > 0).astype(float)
        elif self.activation == 'sigmoid':
            return a * (1 - a)
        elif self.activation == 'tanh':
            return 1 - a ** 2
        else:
            return np.ones_like(a)

class MLP:
    """多层感知机"""
    
    def __init__(self, layer_dims, activations=None):
        """
        参数：
        - layer_dims: 每层维度列表，如[20, 64, 32, 3]
        - activations: 每层激活函数
        """
        self.layers = []
        self.layer_dims = layer_dims
        
        if activations is None:
            activations = ['relu'] * (len(layer_dims) - 2) + ['softmax']
        
        # 构建各层
        for i in range(len(layer_dims) - 1):
            layer = Layer(layer_dims[i], layer_dims[i+1], activations[i])
            self.layers.append(layer)
    
    def forward(self, x):
        """前向传播"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_true, learning_rate=0.01):
        """反向传播（从最后一层开始）"""
        # 计算softmax交叉熵的梯度
        y_pred = self.layers[-1].output_data
        
        # 交叉熵损失对softmax输入的梯度
        # 当使用softmax+crossentropy时：dL/dz = y_pred - y_true
        da = y_pred - y_true
        
        # 反向传播
        for layer in reversed(self.layers):
            da = layer.backward(da, learning_rate)
    
    def train_step(self, X, y_true, learning_rate=0.01):
        """一步训练"""
        # 前向传播
        y_pred = self.forward(X)
        
        # 计算损失（交叉熵）
        eps = 1e-10
        loss = -np.mean(y_true * np.log(y_pred + eps))
        
        # 反向传播
        self.backward(y_true, learning_rate)
        
        return loss
    
    def predict(self, X):
        """预测"""
        probs = self.forward(X)
        return np.argmax(probs, axis=-1)

def main():
    print("=" * 60)
    print("MLP - 纯手写实现")
    print("=" * 60)
    
    # 参数
    input_dim = 20
    hidden_dims = [64, 32]
    output_dim = 3
    
    # 创建模型
    layer_dims = [input_dim] + hidden_dims + [output_dim]
    mlp = MLP(layer_dims)
    
    print(f"\n模型结构:")
    for i, layer in enumerate(mlp.layers):
        print(f"  Layer {i+1}: {layer.input_dim} -> {layer.output_dim}, {layer.activation}")
    
    # 生成训练数据
    np.random.seed(0)
    n_samples = 200
    X_train = np.random.randn(n_samples, input_dim)
    y_train_onehot = np.zeros((n_samples, output_dim))
    y_labels = np.random.randint(0, output_dim, n_samples)
    for i, label in enumerate(y_labels):
        y_train_onehot[i, label] = 1
    
    print(f"\n训练数据:")
    print(f"  X: {X_train.shape}")
    print(f"  y: {y_train_onehot.shape}")
    
    # 训练
    print("\n训练过程:")
    for epoch in range(30):
        loss = mlp.train_step(X_train, y_train_onehot, learning_rate=0.1)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}")
    
    # 评估
    print("\n评估:")
    y_pred_labels = mlp.predict(X_train)
    accuracy = np.mean(y_pred_labels == y_labels)
    print(f"  训练准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main()
```

**代码核心要点**：

1. **全连接层**：矩阵乘法 + 偏置
2. **激活函数**：ReLU、Sigmoid、Tanh、Softmax
3. **反向传播**：链式法则 + 梯度计算
4. **参数更新**：梯度下降

---

## 9. 可视化与结果理解

### 9.1 决策边界可视化

```python
"""
MLP决策边界可视化
"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_decision_boundary(model, X, y, save_path='mlp_boundary.png'):
    """
    可视化MLP在2D数据上的决策边界
    
    仅适用于2维输入的分类问题
    """
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_probs = model.forward(grid_points)
    grid_pred = np.argmax(grid_probs, axis=1).reshape(xx.shape)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, grid_pred, alpha=0.3, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('MLP Decision Boundary')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"决策边界已保存为 {save_path}")
    plt.close()

# 测试
print("\n" + "-" * 40)
print("MLP决策边界可视化")
print("-" * 40)

# 2维数据
np.random.seed(0)
n_samples = 100
X_2d = np.random.randn(n_samples, 2)
y_2d = (X_2d[:, 0] * X_2d[:, 1] > 0).astype(int)

# 创建2维MLP
mlp_2d = MLP([2, 16, 8, 2])

# 训练
for epoch in range(100):
    X_input = np.eye(2)[(X_2d * 10).astype(int).clip(0, 1).sum(axis=1).astype(int)] if False else X_2d
    y_onehot = np.zeros((n_samples, 2))
    y_onehot[np.arange(n_samples), y_2d] = 1
    loss = mlp_2d.train_step(X_2d, y_onehot, learning_rate=0.1)

try:
    visualize_decision_boundary(mlp_2d, X_2d, y_2d)
except Exception as e:
    print(f"可视化失败: {e}")
```

### 9.2 结果解读

**决策边界解读**：

1. **非线性**：MLP可以学习非线性边界
2. **复杂边界**：多层网络能学习复杂模式
3. **过拟合**：如果边界太复杂可能过拟合

---

## 10. 模型评估

### 10.1 评估指标

```python
"""
MLP模型评估
"""

def evaluate_mlp(model, X_test, y_test):
    """评估MLP模型"""
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.forward(X_test)
    
    # 准确率
    accuracy = np.mean(y_pred == y_test)
    
    # 计算各类指标（二分类/多分类）
    n_classes = y_pred_proba.shape[1]
    
    # Precision, Recall, F1
    precision = {}
    recall = {}
    f1 = {}
    
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_test == c))
        fp = np.sum((y_pred == c) & (y_test != c))
        fn = np.sum((y_pred != c) & (y_test == c))
        
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 评估示例
print("\n" + "-" * 40)
print("MLP模型评估")
print("-" * 40)

try:
    metrics = evaluate_mlp(mlp, X_val, y_val)
    print(f"准确率: {metrics['accuracy']:.4f}")
except Exception as e:
    print(f"评估失败: {e}")
```

### 10.2 评估指标说明

| 指标 | 说明 |
|------|------|
| Accuracy | 正确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1 | 精确率和召回率的调和平均 |

---

## 11. 常见问题与易错点

### 11.1 训练问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 难以收敛 | 学习率太大/小 | 调整学习率 |
| 过拟合 | 模型太复杂 | Dropout/正则化 |
| 梯度消失 | 激活函数饱和 | 使用ReLU |

### 11.2 使用问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 维度不匹配 | 输入/输出维度错误 | 检查Layer定义 |
| 内存溢出 | batch太大 | 减小batch |

### 11.3 典型易错点

1. **激活函数选择**：最后一层用softmax/无激活
2. **学习率设置**：0.001-0.1范围调整
3. **梯度裁剪**：防止梯度爆炸
4. **BatchNorm**：可加速收敛

---

## 12. 学习总结

### 12.1 核心思想

MLP的核心是**多层非线性变换**：通过堆叠全连接层+激活函数学习复杂函数。

### 12.2 关键公式

**前向传播**：
$$a^{(l)} = f(W^{(l)} a^{(l-1)} + b^{(l)})$$

**反向传播**：
$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} * f'(z^{(l)})$$

### 12.3 后续学习

1. **Dropout**：防止过拟合
2. **BatchNorm**：加速训练
3. **残差网络**：解决深层网络梯度问题

---

## 13. 练习题与思考题

### 13.1 基础题

**问题**：MLP为什么需要多层？一层不行吗？

**答案**：单层感知机只能学习线性分割，无法解决XOR问题。多层可以通过非线性激活学习任意函数。

### 13.2 进阶题

**问题**：ReLU激活函数有什么优缺点？

**答案**：
- 优点：计算快、缓解梯度消失
- 缺点：可能"dying ReLU"（负值区域永远不激活）

### 13.3 开放题

**问题**：如何选择MLP的层数和每层维度？

**答案可包含**：
1. 数据复杂度：复杂数据需要更深网络
2. 验证集调参：尝试不同配置
3. 参数量：与数据量匹配

---

## 14. 学习路径建议

### 14.1 前置算法

1. **感知机**：单层神经网络
2. **梯度下降**：优化方法
3. **损失函数**：MSE/交叉熵

### 14.2 平行算法

1. **卷积神经网络**：处理图像
2. **循环神经网络**：处理序列

### 14.3 进阶算法

1. **ResNet**：残差网络
2. **Transformer**：注意力网络
3. **GAN**：生成网络

### 14.4 推荐资源

| 资源 | 类型 |
|------|------|
| 感知机原始论文 | Rosenblatt, 1958 |
| 反向传播论文 | Rumelhart et al., 1986 |
| CS231N课程 | CNN视觉标准 |

---

*第8-14章内容添加完成*
