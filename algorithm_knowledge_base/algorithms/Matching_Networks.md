# Matching Networks 学习文档

> 基于注意力机制的少样本匹配学习方法。

---

## 1. 算法基础认知

**Matching Networks（匹配网络）** 是2016年提出的few-shot学习方法，核心思想是使用注意力机制将查询样本与支持集样本进行匹配。

### 1.1 核心思想

- 编码支持集：处理每个支持集样本
- 注意力：计算查询与支持集相似度
- 预测：加权求和

### 1.2 与ProtoNet对比

| 方面 | Matching Networks | Prototypical Networks |
|------|---------------|-------------------|
| 表示 | 全支持集 | 原型均值 |
| 注意力 | 完整 | 简化 |
| 计算复杂度 | O(N) | O(1) |

### 1.3 变体

1. Full Contextual Embedding (FCE)
2. Bidirectional

---

## 2. 核心原理

### 2.1 注意力核

基于余弦相似度：

$$a(x, x_i) = \frac{\exp(d(f(x), g(x_i)))}{\sum_j \exp(d(f(x), g(x_j))}$$

其中：
- $f$：查询编码器
- $g$：支持集编码器

### 2.2 预测

$$\hat{y} = \sum_{i} a(x, x_i) y_i$$

### 2.3 双向匹配

两个编码器互相编码：
$$\hat{y} = \sum_i a(x, x_i) y_i + \sum_i a(x_i, x) y_i$$

---

## 4. 训练过程讲解

### 3.1 Episodic Training

每个episode：
1. 采样N类，K个样本/类
2. 支持集：前K个，查询集：后K个
3. 编码+注意力+预测

### 3.2 算法

```python
# 编码支持集
g_support = g(support_set)

# 编码查询
f_query = f(query)

# 注意力
attn = softmax(cosine(f_query, g_support))

# 预测
pred = sum(attn * support_labels)
```

---

## 4. 调库实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingNetwork(nn.Module):
    """匹配网络"""
    
    def __init__(self, encoder):
        super().__init__()
        self.f = encoder  # query编码
        self.g = encoder  # support编码
    
    def forward(self, support_x, support_y, query_x, way, shot):
        """
        support_x: [way*shot, ...]
        support_y: [way*shot]
        query_x: [query_n, ...]
        """
        # 编码
        support_emb = self.g(support_x)
        query_emb = self.f(query_x)
        
        # 注意力
        attn = F.cosine_similarity(query_emb[:, None], support_emb[None], dim=-1)
        attn = F.softmax(attn, dim=-1)
        
        # 预测
        preds = attn @ F.one_hot(support_y, way).float()
        
        return preds


def matching_train():
    """训练"""
    print("=== 匹配网络训练 ===\n")
    print("1. f编码查询g编码支持集")
    print("2. 注意力计算")
    print("3. 加权预测")


if __name__ == "__main__":
    matching_train()
```

---

## 5. 手工代码实现

```python
import numpy as np

class SimpleMatching:
    """简化版匹配网络"""
    
    def __init__(self, encoder):
        self.encoder = encoder
    
    def predict(self, query_x, support_x, support_y):
        """预测"""
        # 编码
        query_emb = self.encoder(query_x)
        support_emb = self.encoder(support_x)
        
        # 注意力（余弦相似度）
        q = query_emb / (np.linalg.norm(query_emb, axis=-1, keepdims=True) + 1e-8)
        s = support_emb / (np.linalg.norm(support_emb, axis=-1, keepdims=True) + 1e-8)
        
        attn = np.exp(q @ s.T)
        attn = attn / (attn.sum(-1, keepdims=True))
        
        # 加权预测
        one_hot = np.zeros((support_y.max()+1, support_y.shape[0]))
        one_hot[support_y, np.arange(len(support_y))] = 1
        
        return attn @ one_hot.T


if __name__ == "__main__":
    print("=== 匹配网络 ===\n")
    print("1. 双编码器")
    print("2. 注意力核")
    print("3. 加权预测")
```

---

## 6. 应用场景

### 6.1 Few-shot分类

### 6.2 文本匹配

### 6.3 关系抽取

---

## 7. 评估

### 7.1 Om niglot

- 5-way 1-shot: ~93%
- 5-way 5-shot: ~97%

### 7.2 MiniImageNet

- 5-way 1-shot: ~44%

---

## 8. 常见问题

### 8.1 复杂度

- 解决：ProtoNet简化

### 8.2 支持集过拟合

---

## 9. 学习总结

**Matching Networks要点**：

1. **双编码器**：f/g独立
2. **注意力核**：相似度加权
3. **非参数**：无需权重层
4. **灵活**：可建模复杂关系

---

## 10. 练习题

1. f和g编码器可以共享吗？
2. 为什么用余弦相似度？

---

## 11. 学习路径

1. 理解注意力机制
2. 实现Matching Networks
3. 对比ProtoNet

---

*Matching Networks开创了few-shot学习的注意力范式。*
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
