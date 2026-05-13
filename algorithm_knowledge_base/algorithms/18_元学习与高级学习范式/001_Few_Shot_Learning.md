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

---

## 补充材料：Few-Shot Learning变体与扩展

### A1. 元学习优化器的深入理解

MAML（Model-Agnostic Meta-Learning）的核心优化过程：

梯度更新公式：
$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_\mathcal{T}(f_\theta)$$

元梯度：
$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T} \sim p(\mathcal{T})} \mathcal{L}_\mathcal{T}(f_{\theta'})$$

其中α是内部更新学习率，β是元更新学习率。

**代码实现**：
```python
class MAML(nn.Module):
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
    
    def inner_update(self, support, way, shot):
        """支持集的内部更新"""
        # 计算原型
        embeddings = self.model(support)
        embeddings = embeddings.view(way, shot, -1)
        prototypes = embeddings.mean(dim=1)
        
        # 查询集分类
        query = self.model(support)
        dists = torch.cdist(query, prototypes)
        probs = F.softmax(-dists, dim=-1)
        
        return prototypes
    
    def forward(self, task):
        """任务级前向传播"""
        support, query, y_support, y_query = task
        
        # 复制原始参数
        original_params = {k: v.clone() for k, v in self.model.named_parameters()}
        
        # 内部更新（k步）
        for _ in range(5):
            prototypes = self.inner_update(support, way=5, shot=1)
        
        # 查询集评估
        query_emb = self.model(query)
        dists = torch.cdist(query_emb, prototypes)
        logits = F.softmax(-dists, dim=-1)
        
        # 恢复原始参数（不破坏模型）
        for k, v in original_params.items():
            getattr(self.model, k).data = v
        
        return logits
```

### A2. 关系网络（Relation Network）的改进

```python
class RelationNetwork(nn.Module):
    """关系网络实现"""
    def __init__(self, encoder_dim=64):
        super().__init__()
        
        # 关系模块
        self.relation = nn.Sequential(
            nn.Linear(encoder_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_emb, prototypes):
        """计算查询样本与每个原型的关系分数"""
        n_way, n_shot = prototypes.shape[:2]
        n_query = query_emb.shape[0]
        
        # 扩展维度以便计算关系
        query_expanded = query_emb.unsqueeze(1).expand(n_query, n_way, -1)
        proto_expanded = prototypes.unsqueeze(0).expand(n_query, n_way, -1)
        
        # 连接查询和原型
        combined = torch.cat([query_expanded, proto_expanded], dim=-1)
        
        # 关系分数
        relations = self.relation(combined.view(-1, encoder_dim * 2))
        relations = relations.view(n_query, n_way)
        
        return relations
```

### A3. 跨域Few-Shot学习

解决训练域与测试域分布不同的问题：

```python
class CrossDomainFewShot:
    """跨域Few-Shot学习"""
    
    def __init__(self, domain_discriminator):
        self.domain_discriminator = domain_discriminator
    
    def adversarial_training(self, encoder, source_loader, target_loader):
        """对抗训练"""
        source_domain_loss = 0
        target_domain_loss = 0
        
        for (source_x, _), (target_x, _) in zip(source_loader, target_loader):
            source_emb = encoder(source_x)
            target_emb = encoder(target_x)
            
            # 域分类器
            source_domain = self.domain_discriminator(source_emb)
            target_domain = self.domain_discriminator(target_emb)
            
            source_domain_loss += F.cross_entropy(source_domain, torch.zeros(len(source_x)))
            target_domain_loss += F.cross_entropy(target_domain, torch.ones(len(target_x)))
        
        return source_domain_loss + target_domain_loss
```

### A4. Few-Shot学习的评估协议

```python
def evaluate_fewshot(model, test_tasks, way=5, shot=1, num_episodes=1000):
    """标准的Few-Shot评估"""
    accuracies = []
    
    for _ in range(num_episodes):
        # 采样测试任务
        task = sample_task(way, shot, num_query=15)
        
        support, query, y_support, y_query = task
        
        # 不更新参数（仅前向）
        with torch.no_grad():
            logits = model(support, query)
        
        predictions = logits.argmax(dim=-1)
        correct = (predictions == y_query).sum().item()
        accuracy = correct / len(y_query)
        accuracies.append(accuracy)
    
    # 报告统计
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    ci_95 = 1.96 * std_acc / np.sqrt(len(accuracies))
    
    return {
        'accuracy': mean_acc,
        'std': std_acc,
        'ci_95': ci_95
    }


def bootstrap_confidence_interval(model, test_tasks, way=5, shot=1, n_bootstrap=10000):
    """Bootstrap置信区间"""
    accuracies = []
    
    for _ in range(n_bootstrap):
        # 随机采样
        task = sample_task(way, shot)
        accuracy = compute_accuracy(model, task)
        accuracies.append(accuracy)
    
    # 计算百分位置信区间
    lower = np.percentile(accuracies, 2.5)
    upper = np.percentile(accuracies, 97.5)
    
    return lower, upper
```

### A5. Few-Shot学习可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_episodic_learning():
    """可视化 episodic 训练过程"""
    np.random.seed(42)
    
    # 模拟类别在特征空间中的分布
    n_classes = 20
    n_samples = 100
    
    class_centers = {i: np.random.randn(2) * 3 for i in range(n_classes)}
    
    # 训练类 vs 测试类
    train_classes = list(range(15))
    test_classes = list(range(15, 20))
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 训练集特征
    ax = axes[0]
    for i in train_classes:
        samples = class_centers[i] + np.random.randn(n_samples, 2) * 0.5
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
    ax.set_title('Training Classes')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    # 2. 测试集（Few-Shot场景）
    ax = axes[1]
    for i in test_classes:
        samples = class_centers[i] + np.random.randn(5, 2) * 0.5  # 少量样本
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.8, s=30, 
                  label=f'Novel {i}', edgecolors='black')
    ax.set_title('Few-Shot Test (5 samples/class)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend()
    
    # 3. 原型空间
    ax = axes[2]
    prototype_prototype = np.array([class_centers[i] for i in test_classes])
    query_prototype = prototype_prototype + np.random.randn(10, 2)
    
    ax.scatter(prototype_prototype[:, 0], prototype_prototype[:, 1], 
              c='red', s=100, marker='*', label='Prototypes')
    ax.scatter(query_prototype[:, 0], query_prototype[:, 1], 
              c='blue', alpha=0.5, s=30, label='Queries')
    
    ax.set_title('Prototype Matching')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('fewshot_episodic.png', dpi=150)
    plt.show()


def plot_way_shot_tradeoff():
    """N-way K-shot权衡"""
    ways = [3, 5, 10, 20]
    shots = [1, 5, 10]
    
    accuracies = {
        (5, 1): 65.2,
        (5, 5): 78.5,
        (5, 10): 84.2,
        (3, 1): 72.1,
        (3, 5): 82.3,
        (10, 1): 48.5,
        (10, 5): 62.1,
        (20, 1): 38.2
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(ways))
    width = 0.25
    
    for i, shot in enumerate(shots):
        accs = [accuracies.get((way, shot), 0) for way in ways]
        ax.bar(x + i*width, accs, width, label=f'{shot}-shot')
    
    ax.set_xlabel('N-way')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('N-way K-shot Accuracy Tradeoff')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{w}-way' for w in ways])
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('fewshot_way_shot.png', dpi=150)
    plt.show()


def plot_meta_learning_curve():
    """元学习收敛曲线"""
    np.random.seed(42)
    
    epochs = np.arange(0, 100, 5)
    
    # 不同方法
    losses = {
        'MAML': 2.0 * np.exp(-epochs / 30) + 0.2,
        'Prototypical': 1.8 * np.exp(-epochs / 25) + 0.15,
        'RelationNet': 1.5 * np.exp(-epochs / 20) + 0.1,
        'Baseline': 2.2 * np.exp(-epochs / 40) + 0.5
    }
    
    plt.figure(figsize=(10, 6))
    
    for method, loss in losses.items():
        plt.plot(epochs, loss, 'o-', linewidth=2, label=method)
    
    plt.xlabel('Meta-Training Epoch')
    plt.ylabel('Meta-Loss')
    plt.title('Meta-Learning Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fewshot_convergence.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_episodic_learning()
    plot_way_shot_tradeoff()
    plot_meta_learning_curve()
```


## 3. 数学公式与推导

Few_Shot_Learning的数学基础：

### 前向传播
$$h = \sigma(W_1 x + b_1), \quad \hat{y} = W_2 h + b_2$$

### 损失函数（交叉熵）
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

### 反向传播（链式法则）
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W}$$


## 5. 应用场景

Few_Shot_Learning在以下领域有广泛应用：

- 计算机视觉（分类、检测、分割）
- 自然语言处理（生成、翻译、摘要）
- 语音识别与合成
- 跨模态理解与生成
- 科学计算与仿真

在工业实践中，Few_Shot_Learning通常与完整的数据管道配合使用。选择Few_Shot_Learning时需要根据数据特点、性能要求和计算资源综合考量。

## 6. 优缺点分析

### 优点
1. **理论成熟**：有着坚实的理论基础和大量研究支撑
2. **效果可靠**：在适当场景下能取得稳定优秀的性能
3. **社区支持**：完善的开源实现和活跃社区生态
4. **可解释性**：决策过程在一定程度上可理解和解释
5. **易于使用**：主流框架提供简洁API

### 缺点
1. **数据依赖**：性能高度依赖训练数据质量和数量
2. **超参敏感**：某些超参数对结果影响较大
3. **计算开销**：大规模数据下需要较多计算资源
4. **泛化限制**：分布外数据上表现可能下降
5. **假设约束**：理论假设在实际数据中可能不成立

