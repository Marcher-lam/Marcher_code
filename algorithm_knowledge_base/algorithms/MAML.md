# MAML 学习文档

> Model-Agnostic Meta-Learning，模型无关元学习方法。

---

## 1. 算法基础认知

**MAML** 是2017年提出的元学习方法，核心思想是学习一个好的**初始化参数**，使得模型能够快速适应新任务。

### 1.1 核心思想

训练一个初始化参数$\theta$，对来自任务分布$\mathcal{T}$的任何新任务，只需几步梯度下降就能快速适应。

### 1.2 与传统学习的区别

传统学习：
```
参数 ← 梯度下降 ← 任务数据
```

MAML：
```
参数* ← 梯度下降 ← 元梯度 ← 任务最优
```

### 1.3 特点

- task-agnostic：适用于任何可微学习器
- few-shot：只需要几个样本
- quick adaptation：少量梯度步

---

## 2. 核心原理

### 2.1 目标

找到参数$\theta$最小化任务损失：

$$\min_\theta \mathcal{L}_{\mathcal{T}}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta))$$

### 2.2 两层优化

内层：任务适应
$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$

外层：元更新
$$\theta = \theta - \beta \nabla_{\theta'} \mathcal{L'}(\theta')$$

### 2.3 算法

```python
# MAML算法
for episode in range(meta_steps):
    # 采样任务
    tasks = sample_tasks()
    
    for task in tasks:
        # 内层：任务自适应
        theta_prime = theta - alpha * grad(task)
    
    # 外层：元更新
    loss = sum(task_loss(theta_prime) for task in tasks)
    grad = grad(loss, theta)
    theta = theta - beta * grad
```

---

## 3. 数学公式

### 3.1 损失函数

任务$\mathcal{T}_i$的损失：$\mathcal{L}_{\mathcal{T}_i}$

### 3.2 更新公式

$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$$

$$\theta = \theta - \beta \nabla_{\theta'} (\mathcal{L}_{\mathcal{T}_i}(\theta'))$$

### 3.3 一阶近似

FOMAML：忽略二阶项
$$\theta = \theta - \beta \nabla_{\theta'} \mathcal{L}(\theta)$$

---

## 5. 应用场景

### 4.1 Image Classification

Few-shot图像分类

### 4.2 Reinforcement Learning

快速策略适应

### 4.3 Regression

函数回归

---

## 5. 调库实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class MAML:
    """MAML元学习器"""
    
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
    
    def inner_update(self, support_x, support_y, task):
        """内层：任务适应"""
        # 快速复制模型
        model_copy = deepcopy(self.model)
        model_copy.load_state_dict(self.model.state_dict())
        
        # 任务梯度下降
        logits = model_copy(support_x)
        loss = F.cross_entropy(logits, support_y)
        
        # 一步梯度
        grad = torch.autograd.grad(loss, model_copy.parameters(), 
                               allow_unused=True)
        
        # 更新参数
        with torch.no_grad():
            for param, g in zip(model_copy.parameters(), grad):
                if param is not None and g is not None:
                    param -= self.inner_lr * g
        
        return model_copy
    
    def forward(self, query_x, query_y, support_x, support_y):
        """前向"""
        # 内层
        adapted = self.inner_update(support_x, support_y)
        
        # 在查询集上评估
        logits = adapted(query_x)
        loss = F.cross_entropy(logits, query_y)
        
        return loss
    
    def meta_train(self, tasks):
        """元训练"""
        total_loss = 0
        
        for task in tasks:
            support_x, support_y, query_x, query_y = task
            loss = self.forward(query_x, query_y, support_x, support_y)
            total_loss += loss
        
        # 外层更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


def demo_maml():
    print("=== MAML 演示 ===\n")
    print("内层LR:", 0.01)
    print("外层LR:", 0.001)
    print("内层更新: 1-5步梯度")
    print("外层更新: 任务后")


if __name__ == "__main__":
    demo_maml()
```

---

## 6. 手工代码实现

```python
import numpy as np

class SimpleMAML:
    """简化版MAML"""
    
    def __init__(self, param_dim, inner_lr=0.01):
        self.param = np.random.randn(param_dim)
        self.inner_lr = inner_lr
    
    def inner_update(self, x, y):
        """快速适应"""
        pred = x @ self.param
        loss = np.mean((pred - y) ** 2)
        
        # 解析梯度
        grad = 2 * x.T @ (pred - y) / len(y)
        
        # 梯度下降
        self.param -= self.inner_lr * grad
        
        return self.param
    
    def meta_update(self, tasks):
        """元更新"""
        grads = []
        
        for task in tasks:
            x, y = task
            param = self.inner_update(x, y)
            
            # 计算查询损失梯度
            pred = x @ param
            grad = 2 * x.T @ (pred - y) / len(y)
            grads.append(grad)
        
        # 平均更新
        avg_grad = np.mean(grads, axis=0)
        
        return avg_grad


if __name__ == "__main__":
    print("=== MAML实现 ===\n")
    print("1. 内层: 任务自适应")
    print("2. 外层: 元更新")
```

---

## 7. 可视化

```python
def visualize():
    print("\n=== MAML流程 ===\n")
    print("""
任务1, 任务2, ..., 任务N
    ↓
各自 内层更新 (1-5步)
    ↓
聚合梯度 → 元更新
    ↓
θ* (更优初始化)
    """)


if __name__ == "__main__":
    visualize()
```

---

## 10. 模型评估

### 8.1 Few-shot准确率

- 5-way 1-shot
- 5-way 5-shot

### 8.2 基线对比

- 直接fine-tuning
- MAML
- Reptile

---

## 9. 常见问题

### 9.1 计算成本

- 两层梯度计算
- 解决：FOMAML, 迭代次数

### 9.2 任务采样

- 需要大量不同任务

### 9.3 泛化差距

- 训练任务→测试任务

---

## 10. 学习总结

**MAML核心要点**：

1. **双重梯度**：内层+外层
2. **快速适应**：几步梯度
3. **初始化**：好的起点
4. **任务分布**：从分布学习

---

## 11. 练习题

1. MAML和普通fine-tuning的区别？
2. 为什么内层只用几步？

答案：
1. MAML学习初始化+快速适应，普通fine-tuning直接优化
2. 避免过拟合到任务，学习元知识

---

## 12. 学习路径

1. 理解元学习概念
2. 学习MAML推导
3. 实践few-shot分类

---

*MAML是元学习的里程碑，让模型学会学习。*
```
## 13. 练习题与思考题与思考题
### 13.1 基础练习题
**练习1**：本算法的核心机制是什么？请简述其工作原理。
**答案**：本算法的核心是[机制]，通过[步骤]实现[目标]。

**练习2**：给定以下数据，手动计算第一次参数更新。
**答案**：根据[公式]计算，第一次迭代参数更新为[结果]。

### 13.2 进阶思考题
**思考题**：本算法存在哪些局限性？请提出至少2种改进方案。
**答案**：1. [局限性1]→[改进方案1]；2. [局限性2]→[改进方案2]。

## 14. 学习路径建议建议
### 14.1 前置知识
- 线性代数基础（矩阵运算、向量空间）
- 微积分基础（偏导数、梯度）
- Python编程基础（NumPy/PyTorch）
- 机器学习基本概念

### 14.2 平行算法
- [算法A]：[简要对比]
- [算法B]：[简要对比]

### 14.3 进阶算法
- [进阶算法1]：进一步发展方向
- [进阶算法2]：改进方向

### 14.4 推荐资源
**书籍**：《机器学习》周志华，《深度学习》花书
**论文**：[算法名]原论文
**课程**：Andrew Ng机器学习课程
