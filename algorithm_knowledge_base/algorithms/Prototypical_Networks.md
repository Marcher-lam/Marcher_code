# Prototypical Networks 学习文档

> 基于原型向量的少样本度量学习方法。

---

## 1. 算法基础认知

**Prototypical Networks（原型网络）** 是2017年提出的few-shot学习方法，核心思想是为每个类别计算一个原型向量，查询样本通过与原型距离进行分类。

### 1.1 核心思想

- 类原型：支持集样本的嵌入均值
- 分类：最近邻距离

### 1.2 与Matching Networks对比

| 方面 | Matching Networks | Prototypical Networks |
|------|----------------|---------------------|
| 表示 | 全部支持集 | 原型向量 |
| 计算 | Attention加权 | 均值 |
| 复杂度 | O(N) | O(1) |

### 1.3 优点

- 简单高效
- 内存需求低
- 泛化性好

---

## 2. 核心原理

### 2.1 原型计算

对于类$c$，原型为：
$$c_c = \frac{1}{|S_c|} \sum_{(x_i,y_i) \in S_c} f_\phi(x_i)$$

其中：
- $S_c$：类$c$的支持集
- $f_\phi$：嵌入函数

### 2.2 分类

查询样本$x$的预测：
$$P(y=c|x) = \text{softmax}(-d(f_\phi(x), c_c))$$

距离使用欧氏距离：$d(\cdot) = ||\cdot||_2$

### 2.3 损失

使用交叉熵：
$$\mathcal{L} = -\log P(y_c|x)$$

---

## 4. 训练过程讲解

### 3.1 Episodic Training

每个episode：
1. 采样N类，每类K个样本
2. 前K个作为支持集
3. 后Q个作为查询集
4. 计算原型，分类查询

### 3.2 评估

```python
# 编码支持集
support_emb = encoder(support)

# 计算原型
prototypes = support_emb.view(way, shot, -1).mean(1)

# 编码查询集
query_emb = encoder(query)

# 距离 + softmax
dists = torch.cdist(query_emb, prototypes)
preds = F.softmin(dists, dim=-1)
```

---

## 4. 调库实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    """原型网络"""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, support_x, support_y, query_x, way, shot):
        """
        support_x: [way*shot, ...]
        query_x: [query_batch, ...]
        """
        # 编码
        support_emb = self.encoder(support_x)
        query_emb = self.encoder(query_x)
        
        # 计算原型
        way, shot = way, shot
        prototypes = support_emb.view(way, shot, -1).mean(1)
        
        # 距离
        dists = torch.cdist(query_emb, prototypes)
        
        # 分类
        preds = F.softmax(-dists, dim=-1)
        
        return preds


def proto_train():
    """训练"""
    print("=== 原型网络训练 ===\n")
    print("1. 嵌入支持集")
    print("2. 均值计算原型")
    print("3. 距离+softmax")


if __name__ == "__main__":
    proto_train()
```

---

## 5. 手工代码实现

```python
import numpy as np

class SimplePrototypical:
    """简化版原型网络"""
    
    def __init__(self, encoder):
        self.encoder = encoder
    
    def compute_prototype(self, support_embeddings):
        """计算原型"""
        return support_embeddings.mean(0)
    
    def predict(self, query_embedding, prototypes):
        """预测"""
        dists = np.linalg.norm(query_embedding[None] - prototypes, axis=-1)
        return -dists


if __name__ == "__main__":
    print("=== 原型网络 ===\n")
    print("1. 编码支持集")
    print("2. 计算原型")
    print("3. 距离分类")
```

---

## 6. 应用场景

### 6.1 Few-shot分类

图像/文本分类

### 6.2 域适应

### 6.3 增量学习

---

## 7. 评估

### 7.1 Omniglot

- 5-way 1-shot: ~99%
- 5-way 5-shot: ~99.9%

### 7.2 MiniImageNet

- 5-way 1-shot: ~49%
- 5-way 5-shot: ~69%

---

## 8. 常见问题

### 8.1 类别不平衡

### 8.2 度量选择

欧氏vs余弦

---

## 9. 学习总结

**Prototypical核心**：

1. **原型**：类均值
2. **距离**：最近邻分类
3. **简单**：无需复杂计算
4. **有效**：few-shot SOTA

---

## 10. 练习题

1. 原型为什么用均值？
2. 为什么用欧氏距离？

---

## 11. 学习路径

1. 理解度量学习
2. 实现原型网络
3. 对比其他方法

---

*Prototypical Networks是简单有效的few-shot方法。*
```
## 12. 学习总结
### 12.1 核心要点回顾
1. **算法核心**：通过[核心机制]解决[具体问题]
2. **数学本质**：[目标函数]的[优化方法]
3. **关键创新**：相比前代算法引入了[改进]
4. **适用场景**：在[数据类型/任务]下表现优异
5. **局限性**：对[数据特征]有较高要求

### 12.2 关键公式汇总
**预测公式**：$$\hat{y} = f(x; \theta)$$
**损失函数**：$$L(\theta) = \frac{1}{n} \sum \ell(y_i, \hat{y}_i)$$
**参数更新**：$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

### 12.3 与前序/后续算法联系
- **前序算法**：[前置算法]，本算法在其基础上[改进]
- **后续发展**：[后续算法]，进一步[发展方向]
- **相关算法**：[同类算法]采用[不同策略]

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
