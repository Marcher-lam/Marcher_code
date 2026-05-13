# Cross Entropy Loss 学习文档

> 分类问题的标准损失函数，深度学习分类模型的基石。

---

## 1. 算法基础认知

**Cross Entropy Loss（交叉熵损失）** 是分类问题中最重要的损失函数之一，它衡量两个概率分布之间的差异。在深度学习中，我们用交叉熵来衡量模型预测的概率分布与真实标签（通常表示为one-hot分布）之间的距离。

### 1.1 信息论基础

交叉熵的概念来自信息论。对于一个概率分布$p$，用$q$来近似它，交叉熵定义为：

$$H(p, q) = -\sum_i p_i \log q_i$$

这本质上是用$q$编码来自分布$p$的事件所需的平均比特数。

### 1.2 为什么是分类标准损失？

| 特性 | MSE | Cross Entropy |
|------|-----|--------------|
| 输出分布 | 回归 | 分类(概率) |
| 梯度 | 衰减(梯度小) | 平滑(梯度大) |
| 最优解 | 均值 | 概率匹配 |
| 对错惩罚 | 温和 | 强烈 |

交叉熵对错误预测的惩罚更大，训练时收敛更快。

---

## 2. 核心原理

### 2.1 定义

二分类交叉熵：

$$L = -\left[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\right]$$

多分类交叉熵：

$$L = -\sum_{c=1}^{K} y_c \log(\hat{y}_c)$$

其中：
- $K$：类别数
- $y_c$：真实标签（one-hot）
- $\hat{y}_c$：预测概率（softmax输出）

### 2.2 与负对数似然的关系

交叉熵等价于负对数似然：

$$\text{NLL} = -\log P(y|x; \theta) = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$$

这是因为给定输入$x$和参数$\theta$，预测$\hat{y}$就是条件概率$P(y|x; \theta)$。

### 2.3 与KL散度的关系

$$D_{KL}(p||q) = \sum_i p_i \log\frac{p_i}{q_i} = H(p,q) - H(p)$$

当$p$是one-hot时（真实标签），$H(p)=0$，所以：

$$D_{KL}(p||q) = H(p,q)$$

即交叉熵等价于KL散度。

---

## 3. 数学公式与推导

### 3.1 二分类推导

设$\hat{y} = \sigma(z)$，其中$\sigma$是sigmoid函数。

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$

计算各项：
$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

$$\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$$

组合得：
$$\frac{\partial L}{\partial z} = \hat{y} - y$$

这表明梯度只是"预测减真实"，非常简洁！

### 3.2 多分类推导

设$\hat{y} = \text{softmax}(z)$。

使用之前推导的softmax导数：
$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

无论多少类，梯度形式都是"预测减真实"！

### 3.3 组合梯度

这解释了为什么Cross Entropy + Softmax是黄金组合：

- Softmax导数：$-y_j \hat{y}_i$（当$i \neq j$）
- Cross Entropy导数：$-\frac{y_i}{\hat{y}_i}$
- 组合：$\hat{y}_i - y_i$

---

## 4. 训练过程讲解

### 4.1 标准训练流程

```python
# 标准训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        x, y = batch
        
        # 前向
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4.2 标签形式

```python
# 整数标签（用于交叉熵）
y_int = 2  # 第二类

# One-hot编码
y_onehot = [0, 0, 1, 0, 0]

# PyTorch内部会转换
logits = model(x)
loss = F.cross_entropy(logits, y_int)  # 自动one-hot
```

### 4.3 权重平衡

```python
# 不平衡类别
class_weights = torch.tensor([1.0, 2.0, 0.5])
loss = F.cross_entropy(logits, y, weight=class_weights)
```

---

## 5. 应用场景

### 5.1 图像分类

ImageNet 1000类分类，使用Cross Entropy训练CNN。

### 5.2 文本分类

情感分析、主题分类等NLP任务。

### 5.3 目标检测

Faster R-CNN等检测器的分类损失。

### 5.4 语言模型

下一词预测的标准损失。

### 5.5 分割任务

语义分割的像素级分类。

---

## 6. 优缺点分析

### 6.1 优点

1. **梯度平滑**：不会出现梯度消失
2. **单峰优化**：只有一个全局最优
3. **效率高**：计算简洁
4. **兼容性**：配合多输出结构

### 6.2 缺点

1. **One-hot假设**：不适合多标签
2. **对噪声敏感**：交叉熵对错误惩罚大

### 6.3 改进

1. **Label Smoothing**：缓解过拟合
2. **Focal Loss**：处理类别不平衡
3. **Class Weight**：类别权重

---

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropyTrainer:
    """Cross Entropy训练器"""
    
    def __init__(self, num_classes, learning_rate=0.001):
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
    
    def train_step(self, x, y):
        """单步训练"""
        self.model.train()
        
        # 前向
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        
        # 反向
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, x, y):
        """评估"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()
        
        return loss.item(), acc.item()


class LabelSmoothingLoss(nn.Module):
    """标签平滑交叉熵"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, K] logit
            target: [B] integer
        """
        # 創建平滑标签
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_label = one_hot * self.confidence + self.smoothing / self.num_classes
        
        # Softmax + 交叉熵
        log_prob = F.log_softmax(pred, dim=1)
        loss = -torch.sum(smooth_label * log_prob, dim=1).mean()
        
        return loss


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def demo():
    """演示"""
    print("=== Cross Entropy Loss 演示 ===\n")
    
    # 模拟数据
    torch.manual_seed(42)
    batch_size = 4
    num_classes = 10
    
    # 随机batch
    x = torch.randn(batch_size, 784)
    y = torch.randint(0, num_classes, (batch_size,))
    
    print(f"批次大小: {batch_size}")
    print(f"类别数: {num_classes}")
    print(f"标签: {y.tolist()}")
    
    # 测试标准损失
    model = nn.Linear(784, num_classes)
    logits = model(x)
    
    loss = F.cross_entropy(logits, y)
    print(f"\nCross Entropy Loss: {loss.item():.4f}")
    
    # 标签平滑
    smooth_loss = LabelSmoothingLoss(num_classes, smoothing=0.1)
    loss_smooth = smooth_loss(logits, y)
    print(f"Label Smoothing Loss: {loss_smooth.item():.4f}")
    
    # Focal Loss
    focal = FocalLoss(gamma=2.0)
    loss_focal = focal(logits, y)
    print(f"Focal Loss: {loss_focal.item():.4f}")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
import numpy as np

def softmax(x):
    """Softmax函数"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_manual(pred, target):
    """
    手动实现交叉熵
    
    Args:
        pred: [B, K] logit
        target: [B] integer
    """
    # Softmax
    probs = softmax(pred)
    
    # One-hot
    B, K = pred.shape
    target_onehot = np.zeros((B, K))
    target_onehot[np.arange(B), target] = 1
    
    # 交叉熵
    loss = -np.sum(target_onehot * np.log(probs + 1e-10), axis=1)
    
    return np.mean(loss)


def cross_entropy_gradient(pred, target):
    """计算交叉熵的梯度"""
    probs = softmax(pred)
    
    B, K = pred.shape
    target_onehot = np.zeros((B, K))
    target_onehot[np.arange(B), target] = 1
    
    # 梯度: softmax - onehot
    grad = probs - target_onehot
    
    return grad


def demo_manual():
    """演示手动实现"""
    print("=== 手动实现Cross Entropy ===\n")
    
    np.random.seed(42)
    
    # 数据
    pred = np.random.randn(3, 5)
    target = np.array([0, 2, 4])
    
    print(f"Logits:\n{pred}")
    print(f"标签: {target}")
    
    # Softmax输出
    probs = softmax(pred)
    print(f"\nSoftmax概率:\n{probs}")
    
    # 损失
    loss = cross_entropy_manual(pred, target)
    print(f"\n交叉熵损失: {loss:.4f}")
    
    # 梯度
    grad = cross_entropy_gradient(pred, target)
    print(f"\n梯度:\n{grad}")
    print(f"梯度性质: pred - onehot = {grad[0].round(3)}")


if __name__ == "__main__":
    demo_manual()
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_ce():
    """可视化Cross Entropy"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cross Entropy Loss Analysis', fontsize=14)
    
    # 1. 预测vs损失
    ax1 = axes[0, 0]
    y_true = 1  # 真实类别
    y_pred = np.linspace(0.01, 0.99, 100)
    
    loss = -y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)
    ax1.plot(y_pred, loss, 'b-', linewidth=2)
    ax1.set_xlabel('Predicted Probability', fontsize=10)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=10)
    ax1.set_title('Binary Cross Entropy', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='y=0.5')
    
    # 2. 交叉熵 vs 预测错误
    ax2 = axes[0, 1]
    correct_prob = np.linspace(0.5, 1.0, 50)
    wrong_prob = np.linspace(0, 0.5, 50)[::-1]
    
    ce_correct = -np.log(correct_prob)
    ce_wrong = -np.log(wrong_prob)
    
    ax2.plot(correct_prob, ce_correct, 'g-', label='Correct', linewidth=2)
    ax2.plot(correct_prob, np.flip(ce_wrong), 'r-', label='Wrong', linewidth=2)
    ax2.set_xlabel('Prediction Confidence', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.set_title('Loss vs Prediction', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 与其他损失对比
    ax3 = axes[1, 0]
    x = np.linspace(0.01, 0.99, 100)
    y = np.array([1])
    
    ce = -np.log(x)
    mse = (x - 1)**2
    hinge = np.maximum(0, 1 - x)
    
    ax3.plot(x, ce, 'b-', label='Cross Entropy', linewidth=2)
    ax3.plot(x, mse, 'r-', label='MSE', linewidth=2)
    ax3.plot(x, hinge, 'g-', label='Hinge', linewidth=2)
    ax3.set_xlabel('Prediction', fontsize=10)
    ax3.set_ylabel('Loss', fontsize=10)
    ax3.set_title('Loss Functions Comparison', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Label Smoothing效果
    ax4 = axes[1, 1]
    probs = np.linspace(0.01, 0.99, 100)
    
    ce_hard = -np.log(probs)
    smooth_0_1 = -(0.9*np.log(probs) + 0.1*np.log(1-probs)/9)
    smooth_0_2 = -(0.8*np.log(probs) + 0.2*np.log(1-probs)/9)
    
    ax4.plot(probs, ce_hard, 'b-', label='Hard', linewidth=2)
    ax4.plot(probs, smooth_0_1, 'r-', label='Smoothing 0.1', linewidth=2)
    ax4.plot(probs, smooth_0_2, 'g-', label='Smoothing 0.2', linewidth=2)
    ax4.set_xlabel('Prediction', fontsize=10)
    ax4.set_ylabel('Loss', fontsize=10)
    ax4.set_title('Label Smoothing Effect', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_entropy_analysis.png', dpi=150)
    plt.show()
    print("\nSaved to cross_entropy_analysis.png")


if __name__ == "__main__":
    visualize_ce()
```

---

## 10. 模型评估

```python
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_classification(preds, labels, num_classes):
    """评估分类模型"""
    
    results = {}
    
    # 准确率
    results['accuracy'] = accuracy_score(labels, preds)
    
    # 混淆矩阵
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    results['confusion_matrix'] = cm
    
    # 各类准确率
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    results['per_class_accuracy'] = per_class_acc
    
    return results


def log_loss_evaluate(pred_probs, labels):
    """交叉熵损失评估"""
    
    # 转换为对数概率
    log_probs = np.log(pred_probs + 1e-10)
    
    # One-hot标签
    n = len(labels)
    one_hot = np.zeros_like(pred_probs)
    one_hot[np.arange(n), labels] = 1
    
    # 交叉熵
    ce = -np.sum(one_hot * log_probs, axis=1)
    
    return np.mean(ce)


if __name__ == "__main__":
    print("=== Cross Entropy 评估示例 ===\n")
    
    # 模拟预测
    np.random.seed(42)
    n = 100
    num_classes = 5
    
    labels = np.random.randint(0, num_classes, n)
    probs = np.random.rand(n, num_classes)
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    
    results = evaluate_classification(preds, labels, num_classes)
    print(f"准确率: {results['accuracy']:.2%}")
    print(f"各类准确率: {results['per_class_accuracy'].round(2)}")
```

---

## 11. 常见问题与易错点

### 11.1 数值不稳定

```python
# 错误
loss = -np.sum(y * np.log(probs))

# 正确
loss = F.cross_entropy(logits, y)  # 内部log_softmax
```

### 11.2 维度不匹配

```python
# 确保logit shape正确
# [batch_size, num_classes] vs [batch_size]
```

### 11.3 稀疏标签

```python
# 直接用整数标签，不要one-hot
loss = F.cross_entropy(logits, y_int)  # 正确
loss = F.cross_entropy(logits, y_onehot)  # 可能出错
```

### 11.4 忽略索引

```python
# 检查类别数是否匹配
# num_classes必须>=最大标签+1
```

---

## 12. 学习总结

**Cross Entropy核心要点**：

1. **定义**：$H(p,q) = -\sum p \log q$
2. **梯度**：$\hat{y} - y$（与Softmax组合）
3. **标签**：整数或one-hot
4. **最优化**：唯一全局最优
5. **改进**：Label Smoothing、Focal Loss

---

## 13. 练习题与思考题与思考题

### 13.1 选择题

1. 二分类中，y=1, $\hat{y}$=0.1，损失是多少？
   - A) 0.1
   - B) 1.0
   - C) 2.3
   - D) -log(0.1)
   
   **答案：D**

2. CE + Softmax的梯度是？
   - A) $\hat{y} \cdot y$
   - B) $\hat{y} - y$
   - C) $y - \hat{y}$
   - D) $\hat{y} / y$
   
   **答案：B**

### 13.2 编程题

实现带权重的Cross Entropy：

```python
def weighted_cross_entropy(pred, target, weights):
    probs = softmax(pred)
    # ...实现权重
```

---

## 14. 学习路径建议建议

1. **基础**：理解信息论中的熵
2. **实现**：PyTorch API
3. **改进**：Label Smoothing等变体

*Cross Entropy是分类任务的基石，熟练掌握它是深度学习的第一步。*

## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Cross_Entropy_Loss的核心思想及适用场景。
<details><summary>参考答案</summary>
Cross_Entropy_Loss通过数据驱动学习输入到输出的映射，适用于人工智能中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Cross_Entropy_Loss的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Cross_Entropy_Loss核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Cross_Entropy_Loss在什么情况下会失效？
2. 训练数据很少时，Cross_Entropy_Loss还能有效工作吗？
3. 如何将Cross_Entropy_Loss与其他方法结合？

