# Softmax 学习文档

> 将任意实数向量转化为概率分布的归一化指数函数。深度学习中无处不在的核心操作。

---

## 1. 算法基础认知

**Softmax** 是深度学习和机器学习中最重要的激活函数之一，它将一个实数向量（logits）转换为概率分布，使得所有输出值都在(0,1)区间内且和为1。这使得Softmax特别适合多分类问题的输出层。

### 1.1 为什么需要Softmax？

在多分类问题中，模型输出需要满足概率的两条基本性质：
1. **非负性**：$P(y_i) \geq 0$
2. **归一性**：$\sum_i P(y_i) = 1$

普通线性输出的值可能为负、可能任意大，不满足概率的性质。Softmax通过指数化和归一化恰好满足这两条性质，使得输出可以解释为各类别的概率。

### 1.2 Softmax vs Sigmoid

| 特性 | Softmax | Sigmoid |
|------|--------|--------|
| 输出值域 | 所有元素和为1 | 各自独立在(0,1) |
| 相互关系 | 互斥（竞争关系） | 独立（可同时激活） |
| 应用场景 | 多分类 | 二分类/多标签 |
| 数学形式 | $\frac{e^{x_i}}{\sum e^{x_j}}$ | $\frac{1}{1+e^{-x}}$ |

---

## 2. 核心原理

### 2.1 定义

对于输入向量$\mathbf{x} = (x_1, x_2, ..., x_n)$，Softmax定义为：

$$\text{Softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

其中：
- $x_i$：第i个类别的logit（未归一化的分数）
- 分母：所有类别指数的和（归一化常数）

### 2.2 本质理解

**为什么使用指数函数？**

1. **放大差异**：大的logit会获得更大的概率
2. **保证非负**：$e^x > 0$ always
3. **数学性质好**：导数形式优美

**为什么不是其他函数？**

- 归一化需要累加和，Sigmoid只能归一化单个值
- softplus、relu等不满足概率分布的要求

### 2.3 温度参数

带温度参数的Softmax：

$$\text{Softmax}(x_i, T) = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}$$

- **T > 1**：更平滑（近似均匀分布）
- **T < 1**：更尖锐（接近argmax）
- **T → 0**：趋近于one-hot分布

---

## 3. 数学公式与推导

### 3.1 导数推导

设$y_i = \frac{e^{x_i}}{Z}$，其中$Z = \sum_j e^{x_j}$

**情况1：i=j（自Derivative）**

$$\frac{\partial y_i}{\partial x_i} = \frac{e^{x_i} \cdot Z - e^{x_i} \cdot e^{x_i}}{Z^2} = y_i(1-y_i)$$

**情况2：i≠j（交叉导数）**

$$\frac{\partial y_i}{\partial x_j} = \frac{0 \cdot Z - e^{x_i} \cdot e^{x_j}}{Z^2} = -y_i y_j$$

写成矩阵形式：

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \text{diag}(\mathbf{y}) - \mathbf{y}\mathbf{y}^T$$

### 3.2 数值稳定性技巧

直接计算$e^{x_i}$可能导致数值溢出（当$x_i$很大时）。

**Log-Softmax**：在softmax内部计算log，避免溢出

$$\log \text{Softmax}(x_i) = x_i - \log \sum_j e^{x_j}$$

使用Log-Sum-Exp技巧：

$$L = \max_j x_j$$
$$\log \sum_j e^{x_j} = L + \log \sum_j e^{x_j - L}$$

### 3.3 梯度流动

$$\frac{\partial \mathcal{L}}{\partial x_i} = y_i \cdot \left(\frac{\partial \mathcal{L}}{\partial y_i} - \sum_k y_k \frac{\partial \mathcal{L}}{\partial y_k}\right)$$

这个形式说明：梯度流回时，会减去加权平均，实现归一化效果。

---

## 4. 训练过程讲解

### 4.1 Softmax在网络中的位置

```
输入 → 特征提取 → 全连接层 → Logits → Softmax → Probabilities → Loss
```

### 4.2 配合Cross Entropy使用

Softmax通常与Cross Entropy Loss一起使用，实现分类任务的标准训练：

```python
# 标准组合
logits = model(input)
probs = softmax(logits)
loss = cross_entropy_loss(logits, labels)
# 或直接使用
loss = F.cross_entropy(logits, labels)  # 内部组合了
```

### 4.3 训练注意事项

1. **数值稳定性**：使用log_softmax提高数值稳定性
2. **温度调参**：可学习的温度参数
3. **Label Smoothing**：配合标签平滑减少过拟合

---

## 5. 应用场景

### 5.1 多分类

最经典的应用：ImageNet 1000类分类、文本分类等。

### 5.2 注意力机制

Transformer中的注意力权重计算：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 5.3 语言模型

语言模型输出下一个词的概率分布。

### 5.4 强化学习

策略网络的输出（动作概率分布）。

### 5.5 生成模型

GAN生成器的输出分布控制。

---

## 6. 优缺点分析

### 6.1 优点

1. **数学优美**：满足概率分布的两条性质
2. **可导**：梯度形式良好
3. **相互竞争**：自然实现竞争机制
4. **与交叉熵配合**：梯度形式简化

### 6.2 缺点

1. **排他性**：不适合多标签分类
2. **数值问题**：指数可能溢出
3. **计算成本**：O(n)对于大词汇表

### 6.3 改进

1. **Sparse Softmax**：减少计算
2. **Adaptive Softmax**：大词汇表优化
3. **Temperature Softmax**：可调平滑度

---

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoftmaxClassifier(nn.Module):
    """带Softmax的分类器"""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        return probs


class TemperatureSoftmax(nn.Module):
    """带温度参数的Softmax"""
    
    def __init__(self, input_dim, num_classes, initial_temp=1.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.temp = nn.Parameter(torch.tensor(initial_temp))
    
    def forward(self, x):
        logits = self.fc(x)
        # 温度缩放
        scaled_logits = logits / self.temp
        probs = F.softmax(scaled_logits, dim=-1)
        return probs


def softmax_demo():
    """Softmax演示"""
    print("=== Softmax 演示 ===\n")
    
    # 模拟数据
    torch.manual_seed(42)
    batch_size = 4
    num_classes = 5
    
    # 随机logits
    logits = torch.randn(batch_size, num_classes)
    print(f"原始Logits:\n{logits}\n")
    
    # PyTorch Softmax
    probs = F.softmax(logits, dim=-1)
    print(f"Softmax概率:\n{probs}\n")
    print(f"每行总和: {probs.sum(dim=-1)}")
    
    # 温度效果
    print("\n=== 温度参数效果 ===")
    print(f"T=0.5 (尖锐): {F.softmax(logits/0.5, dim=-1)[0]}")
    print(f"T=1.0 (标准): {F.softmax(logits/1.0, dim=-1)[0]}")
    print(f"T=2.0 (平滑): {F.softmax(logits/2.0, dim=-1)[0]}")


def stable_softmax():
    """数值稳定的Softmax"""
    
    def stable_softmax(x):
        """手动实现数值稳定的softmax"""
        # 减去最大值避免溢出
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def log_softmax(x):
        """log_softmax"""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = x - x_max
        return exp_x - np.log(np.sum(np.exp(exp_x), axis=-1, keepdims=True))
    
    print("\n=== 数值稳定性 ===")
    # 大值测试
    x = np.array([1000, 1001, 1002])
    
    try:
        result = np.exp(x) / np.sum(np.exp(x))
    except:
        print("标准计算溢出")
    
    result = stable_softmax(x)
    print(f"稳定Softmax: {result}")


if __name__ == "__main__":
    softmax_demo()
    stable_softmax()
```

---

## 8. 手工代码实现

```python
import numpy as np
import math

class SimpleSoftmax:
    """纯Python实现Softmax"""
    
    @staticmethod
    def softmax(x):
        """计算softmax
        
        Args:
            x: numpy array of logits
            
        Returns:
            softmax probabilities
        """
        # 数值稳定版本
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def log_softmax(x):
        """计算log_softmax
        
        Args:
            x: numpy array
            
        Returns:
            log_softmax values
        """
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = x - x_max
        return exp_x - np.log(np.sum(np.exp(exp_x), axis=-1, keepdims=True))
    
    @staticmethod
    def softmax_with_temp(x, temperature=1.0):
        """带温度的softmax
        
        Args:
            x: input logits
            temperature: temperature parameter
            
        Returns:
            temperature-scaled softmax
        """
        return SimpleSoftmax.softmax(x / temperature)


def manual_gradient_check():
    """梯度检查"""
    print("\n=== Softmax梯度检查 ===")
    
    x = np.array([2.0, 1.0, 0.1], require_grad=True)
    
    # PyTorch计算
    import torch
    x_torch = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
    y = torch.sum(torch.softmax(x_torch, dim=0))
    y.backward()
    
    print(f"PyTorch梯度: {x_torch.grad.numpy()}")
    
    # 解析导数: dy_i/dx_j = y_i * (δ_ij - y_j)
    y_np = SimpleSoftmax.softmax(x)
    jacobian = np.diag(y_np) - np.outer(y_np, y_np)
    analytic_grad = jacobian.sum(axis=1)
    
    print(f"解析梯度: {analytic_grad}")
    print(f"误差: {np.abs(analytic_grad - x_torch.grad.numpy()).max()}")


def demo():
    """演示"""
    print("=== Softmax手工实现 ===\n")
    
    # 测试数据
    logits = np.array([2.0, 1.0, 0.1])
    
    # 计算
    probs = SimpleSoftmax.softmax(logits)
    log_probs = SimpleSoftmax.log_softmax(logits)
    
    print(f"输入: {logits}")
    print(f"Softmax: {probs}")
    print(f"Log-Softmax: {log_probs}")
    print(f"概率和: {probs.sum()}")


if __name__ == "__main__":
    demo()
    manual_gradient_check()
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_softmax():
    """可视化Softmax"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Softmax Analysis', fontsize=14, fontweight='bold')
    
    # 1. Softmax变换效果
    ax1 = axes[0, 0]
    x = np.linspace(-5, 5, 100)
    y = np.exp(x) / np.sum(np.exp(x))
    
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Input Value', fontsize=10)
    ax1.set_ylabel('Probability', fontsize=10)
    ax1.set_title('Single Value Softmax', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 多类分布
    ax2 = axes[0, 1]
    logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5])
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 5))
    ax2.bar(range(5), probs, color=colors)
    ax2.set_xlabel('Class', fontsize=10)
    ax2.set_ylabel('Probability', fontsize=10)
    ax2.set_title('Multi-class Distribution', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 温度效果
    ax3 = axes[1, 0]
    temps = [0.1, 0.5, 1.0, 2.0, 5.0]
    x_base = np.array([3.0, 2.0, 1.0])
    
    max_probs = []
    for temp in temps:
        probs = np.exp(x_base/temp) / np.sum(np.exp(x_base/temp))
        max_probs.append(probs[0])
    
    ax3.plot(temps, max_probs, 'b-o', linewidth=2, markersize=8)
    ax3.axhline(y=0.33, color='r', linestyle='--', label='Uniform')
    ax3.set_xlabel('Temperature', fontsize=10)
    ax3.set_ylabel('Max Probability', fontsize=10)
    ax3.set_title('Temperature Effect', fontsize=11)
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 梯度曲面
    ax4 = axes[1, 1]
    
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(X) / (np.exp(X) + np.exp(Y))
    
    im = ax4.contourf(X, Y, Z, levels=20, cmap='RdBu_r')
    ax4.set_xlabel('x₁', fontsize=10)
    ax4.set_ylabel('x₂', fontsize=10)
    ax4.set_title('Softmax Surface (x₁, x₂)', fontsize=11)
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('softmax_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved to softmax_analysis.png")


def visualize_attention_softmax():
    """可视化注意力Softmax"""
    
    print("\n=== Attention Softmax ===\n")
    
    # 模拟注意力分数
    QK = np.random.randn(5, 8)
    
    # Softmax
    attn = np.exp(QK) / np.exp(QK).sum(axis=-1, keepdims=True)
    
    print("Attention权重 (第一行):")
    print(attn[0].round(3))
    print(f"\n最大值位置: {np.argmax(attn[0])}")
    print(f"最大值: {attn[0].max():.3f}")


if __name__ == "__main__":
    visualize_softmax()
    visualize_attention_softmax()
```

---

## 10. 模型评估

```python
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

def evaluate_softmax_predictions(probs, labels):
    """评估Softmax输出"""
    
    results = {}
    
    # 预测类别
    preds = np.argmax(probs, axis=1)
    
    # 准确率
    results['accuracy'] = accuracy_score(labels, preds)
    
    # 交叉熵
    results['log_loss'] = log_loss(labels, probs)
    
    # 置信度
    results['avg_confidence'] = np.mean(np.max(probs, axis=1))
    results['avg_entropy'] = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    return results


def calibration_analysis():
    """校准分析"""
    print("\n=== 校准分析 ===\n")
    
    # 完美校准 vs 过度自信
    confidences = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    accuracies = [0.3, 0.5, 0.7, 0.85, 0.90, 0.92]
    expected_accuracy = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    
    print(f"{'置信度':<12} {'实际准确率':<12} {'期望准确率':<12}")
    print("-" * 40)
    for c, a, e in zip(confidences, accuracies, expected_accuracy):
        print(f"{c:<12.2f} {a:<12.2f} {e:<12.2f}")
    
    print("\n结论：使用Temperature可调整校准")


if __name__ == "__main__":
    calibration_analysis()
```

---

## 11. 常见问题与易错点

### 11.1 数值溢出

**问题**：大logit导致e^x溢出

**解决**：使用log_softmax技巧

```python
# 错误
probs = np.exp(logits) / sum(np.exp(logits))

# 正确
logits_max = np.max(logits)
probs = np.exp(logits - logits_max) / sum(np.exp(logits - logits_max))
```

### 11.2 dim参数遗漏

**问题**：维度错误

**解决**：指定正确的dim

```python
# 多分类
probs = F.softmax(logits, dim=-1)  # 最后一个维度

# 注意力
attn = F.softmax(scores, dim=-1)  # 行维度
```

### 11.3 与Cross Entropy配合

**问题**：数值不稳定

**解决**：使用官方的cross_entropy，内部优化

```python
# 错误
loss = -sum(y * log(softmax(x))

# 正确
loss = F.cross_entropy(logits, labels)  # 内部log_softmax
```

### 11.4 大词汇表

**问题**：词汇表太大计算慢

**解决**：使用Hierarchical Softmax或采样

### 11.5 梯度消失

**问题**：one-hot标签导致梯度消失

**解决**：使用label smoothing

---

## 12. 学习总结

**Softmax核心要点**：

1. **概率分布**：输出归一化为概率
2. **指数函数**：放大差异，保证非负
3. **温度控制**：调节平滑度
4. **数值稳定**：使用log-softmax
5. **配合CE Loss**：标准分类训练

**为什么Softmax有效**：
- 数学性质好，可导且梯度形式优美
- 满足概率分布的定义
- 与交叉熵配合实现标准分类训练

---

## 13. 练习题与思考题与思考题

### 13.1 选择题

1. Softmax(0, 0, 0) = ?
   - A) (0, 0, 0)
   - B) (1, 1, 1)
   - C) (1/3, 1/3, 1/3)
   - D) (0.5, 0.5, 0)
   
   **答案：C**

2. Softmax的输出满足什么性质？
   - A) 总和为0
   - B) 总和为1
   - C) 总和为n
   - D) 无限制
   
   **答案：B**

3. 温度T>1时，Softmax会变怎样？
   - A) 更尖锐
   - B) 更平滑
   - C) 不变
   - D) 发散
   
   **答案：B**

### 13.2 简答题

1. Softmax和Sigmoid的区别是什么？
   
   **答案**：Softmax是互斥的，总和为1；Sigmoid独立，各自在(0,1)。

2. 为什么Softmax用于多分类而不用于二分类？
   
   **答案**：多分类需要互斥，二分类用Sigmoid更简单。

### 13.3 编程题

实现带温度的Softmax分类器：

```python
class TempSoftmax:
    def __init__(self, temp=1.0):
        self.temp = temp
    
    def __call__(self, x):
        x = np.array(x)
        return np.exp(x/self.temp) / np.exp(x/self.temp).sum()
```

---

## 14. ��习路径建议

### 14.1 入门

1. 理解概率分布
2. 掌握Softmax公式
3. 学会使用PyTorch API

### 14.2 进阶

1. 理解梯度推导
2. 学习温度调参
3. 数值稳定性技巧

### 14.3 应用

1. 多分类任务
2. 注意力机制
3. 损失函数

*Softmax是深度学习的基础模块，掌握它对理解整个深度学习系统至关重要。*