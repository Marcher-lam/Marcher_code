# Few-Shot Learning 学习文档

> 利用极少样本学习新任务的机器学习范式。

---

## 1. 算法基础认知

**Few-Shot Learning（少样本学习）** 是一种利用极少样本（比如每个类1-5个）来学习新任务的范式。核心挑战是如何从少量样本中泛化到未见过的新类别。

### 1.1 设置定义

N-way K-shot分类：
- N：类别数（通常5）
- K：每类样本数（通常1或5）

### 1.2 核心挑战

- 数据稀缺
- 泛化到新类别
- 避免过拟合

### 1.3 与传统学习的区别

| 方面 | 传统学习 | Few-shot |
|------|----------|----------|
| 样本数 | 大量 | 极少 |
| 新类别 | 固定 | 需要泛化 |
| 目标 | 低错误率 | 高泛化 |

---

## 2. 核心原理

### 2.1 元学习范式

Learn to learn：从任务分布中学习快速适应能力

### 2.2 Episodic Training

每个episode模拟few-shot任务：
- 从训练类采样N类，每类K个样本
- Query集：测试样本
- Support集：支持样本

### 2.3 度量学习

学习样本间的相似性度量：
- Embedding空间：相似样本靠近
- 距离计算：最近邻分类

---

## 3. 主要方法

### 3.1 Matching Networks

```python
\hat{y} = \sum_{i=1}^k a(x, x_i) y_i
```
注意力机制加权

### 3.2 Prototypical Networks

```python
c_k = \frac{1}{|S_k|} \sum x_i
d(x, c_k) = -softmax(d(x, c_k))
```

类原型 = 支持集均值

### 3.3 Relation Networks

```python
score = RelationModule(x, x_i)
```

学习关系度量

---

## 4. 数据集

### 4.1 Omniglot

- 50种字母
- 1623类
- 每类20样本

### 4.2 MiniImageNet

- 100类→64/16/20 split
- 每类600样本

### 4.3 tieredImageNet

- 34大类→20/6/8 split

---

## 5. 训练过程

### 5.1 训练循环

```python
for episode in range(num_episodes):
    # 采样任务
    classes, samples = sample_task(way, shot)
    
    # 支持集、查询集
    support, query = split(samples)
    
    # 计算原型/匹配
    predictions = model(support, query)
    
    # 计算损失
    loss = cross_entropy(predictions, labels)
    loss.backward()
```

### 5.2 评估

```python
# 新类别
accuracies = []
for episode in test_episodes:
    acc = evaluate(model, episode)
    accuracies.append(acc)
```

---

## 6. 调库实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

class ProtoNet(nn.Module):
    """原型网络"""
    def __init__(self, encoder):
        self.encoder = encoder
    
    def forward(self, support, query, way, shot):
        # 编码支持集
        support_emb = self.encoder(support)  # [way*shot, d]
        way, shot = way, shot
        
        # 计算原型
        support_emb = support_emb.view(way, shot, -1)
        prototypes = support_emb.mean(1)  # [way, d]
        
        # 编码查询集
        query_emb = self.encoder(query)  # [n, d]
        
        # 距离计算
        dists = torch.cdist(query_emb, prototypes)
        
        # 分类
        preds = F.softmax(-dists, dim=-1)
        
        return preds


def fewshot_train():
    """训练脚本"""
    print("=== Few-shot训练 ===\n")
    print("1. Episodic采样")
    print("2. 计算原型/匹配")
    print("3. 交叉熵损失")


if __name__ == "__main__":
    fewshot_train()
```

---

## 7. 手工代码实现

```python
import numpy as np

class SimpleFewShot:
    """简化版few-shot"""
    
    def __init__(self, way=5, shot=1):
        self.way = way
        self.shot = shot
    
    def predict(self, support_embeddings, query_embeddings):
        """
        support: [way*shot, d]
        query: [n, d]
        """
        # 计算原型
        prototypes = support_embeddings.reshape(self.way, self.shot, -1).mean(1)
        
        # 距离
        dists = np.linalg.norm(query_embeddings[:, None] - prototypes[None], axis=-1)
        
        # 分类
        return -dists  # softmax在负距离上


if __name__ == "__main__":
    print("=== Few-shot核心 ===\n")
    print("1. 编码：特征提取")
    print("2. 原型：支持集聚合")
    print("3. 分类：最近邻")
```

---

## 8. 可视化

```python
import matplotlib.pyplot as plt

def visualize():
    print("\n=== Few-shot流程 ===\n")
    print("""
训练分布: 类A, B, C, D, E, ...
           ↓
测试分布:  新类F, G, H, ...

Support: 每个新类1-5个样本
Query:   测试样本
Output:  分类结果
    """)


if __name__ == "__main__":
    visualize()
```

---

## 9. 评估

### 9.1 准确率指标

- N-way K-shot准确率
- 常用5-way 1-shot (约50%)、5-way 5-shot (约70%)

### 9.2 基线对比

- Random: 1/N
- Nearest Neighbor
- ProtoNet
- MAML

---

## 10. 常见问题

### 10.1 过拟合

- 数据增强
- 域随机化

### 10.2 任务难度

- 调整way数

### 10.3 泛化差距

- 训练-测试类别不同

---

## 12. 学习总结

**Few-shot核心要点**：

1. **Episodic训练**：模拟任务
2. **度量学习**：相似性度量
3. **原型/匹配**：核心机制
4. **泛化能力**：从训练类到新类

---

## 12. 练习题

1. 什么是N-way K-shot设置？
2. 为什么需要episodic训练？

答案：
1. N类×K样本的few-shot任务
2. 模拟测试时的few-shot场景，学习元知识

---

## 13. 学习路径

1. 理解度量学习
2. 学习Prototypical Networks
3. 理解MAML
4. 实践few-shot分类

---

*Few-shot learning让模型学会学习，是人工智能的重要里程碑。*
```
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
