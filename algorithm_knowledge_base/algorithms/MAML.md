# MAML 学习文档

> Model-Agnostic Meta-Learning，模型无关元学习方法，让模型学会学习。

---

## 1. 算法基础认知

### 1.1 一句话定义

MAML（Model-Agnostic Meta-Learning）是2017年Chelsea Finn提出的元学习方法，核心思想是学习一个好的**初始化参数**，使得模型能够快速适应新任务，只需少量梯度步骤即可完成适应。

### 1.2 直觉类比

将MAML想象为**学习学习方法**：就像人类在学习骑自行车时，不是记住如何骑每一辆特定的自行车，而是学习"骑车的平衡感"，这种通用能力可以快速迁移到任何自行车。MAML正是让神经网络学习这种"元知识"。

### 1.3 历史背景

- **1990s**：Learning to Learn研究
- **2017年**：MAML在ICSS论文中提出
- **2018年**：Reptile简化版
- **2019年**：Meta-SGD、自适应MAML
- **2020s**：元学习在少样本学习中广泛使用

### 1.4 算法定位

- **类型**：元学习 -> 少样本学习
- **输出**：快速适应新任务的初始化参数
- **模型类型**：元模型/快速学习器
- **核心创新**：双层优化

### 1.5 前置知识

- 深度学习基础：神经网络、梯度下降
- 优化基础：梯度计算、反向传播
- PyTorch基础：张量操作
- 少样本学习概念：N-way K-shot

---

## 2. 核心原理

### 2.1 核心思想

MAML的核心是在**任务分布**上学习一个初始化参数 $\theta$，使得对于来自该分布的任意新任务，只需K步梯度下降就能达到好的性能。

**关键洞察**：不是直接学习针对特定任务的参数，而是学习如何快速学习。

### 2.2 双层优化

```
外层循环（元学习）：
  从任务分布采样任务
  ↓
  对每个任务执行内层梯度更新
  ↓
  计算内层更新后的损失
  ↓
  更新初始化参数θ

内层循环（任务适应）：
  支持集计算梯度
  执行几步梯度下降
  在查询集评估
```

### 2.3 算法流程

```python
# MAML伪代码
for episode in range(meta_iterations):
    # 采样任务batch
    tasks = sample_tasks(task_distribution)
    
    for task in tasks:
        # 支持集（用于内层更新）
        support_x, support_y = task.support_set
        
        # 查询集（用于外层评估）
        query_x, query_y = task.query_set
        
        # 任务适应（内层）
        theta_prime = theta - alpha * gradient(support_x, support_y, theta)
        
        # 元更新（外层）
        loss_on_query = compute_loss(query_x, query_y, theta_prime)
        gradient_from_task = gradient(loss_on_query, theta)
    
    # 聚合所有任务的梯度
    meta_gradient = average(gradients_from_all_tasks)
    
    # 更新初始化参数
    theta = theta - beta * meta_gradient
```

### 2.4 关键概念

| 概念 | 说明 |
|------|------|
| 任务分布 | p(T) 所有可能任务 |
| 支持集 | 用于适应任务的少量样本 |
| 查询集 | 用于评估的样本 |
| 快速适应 | 仅需几步梯度下降 |
| 元知识 | 初始化参数中的知识 |

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $\theta$ | 初始参数 |
| $\theta'$ | 任务适应后的参数 |
| $\alpha$ | 内层学习率 |
| $\beta$ | 外层学习率 |
| $\mathcal{L}_T$ | 任务T的损失 |
| $\mathcal{T}$ | 任务分布 |

### 3.2 目标函数

MAML的目标是找到最优初始化参数：
$$
\min_\theta \mathbb{E}_{T \sim p(\mathcal{T})}[\mathcal{L}_T(f_{\theta'})]
$$

其中 $\theta'$ 是对任务T进行K步梯度后的参数：
$$
\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_T(f_\theta)
$$

### 3.3 梯度推导

**内层梯度**（任务适应）：
$$
g_{inner} = \nabla_\theta \mathcal{L}_T(f_\theta)
$$

**外层梯度**（元更新）：
$$
g_{outer} = \nabla_\theta \mathcal{L}_T(f_{\theta'})
$$

使用链式法则：
$$
g_{outer} = \frac{\partial \mathcal{L}_T}{\partial \theta'} \cdot \frac{\partial \theta'}{\partial \theta}
$$

由于 $\theta' = \theta - \alpha g_{inner}$，得：
$$
\frac{\partial \theta'}{\partial \theta} = I - \alpha \nabla_\theta^2 \mathcal{L}_T(f_\theta)
$$

### 3.4 FOMAML简化

一阶MAML（FOMAML）忽略二阶项：
$$
g_{outer} \approx \mathbb{E}[\nabla_{\theta'} \mathcal{L}_T(f_{\theta'})]
$$

这样：
1. 计算更高效
2. 内存占用更小
3. 在实践中效果相近

### 3.5 损失函数选择

| 任务类型 | 损失函数 |
|---------|----------|
| 分类 | 交叉熵 |
| 回归 | MSE |
| 检测 | Focal Loss |
| 强化学习 | 策略梯度 |

---

## 4. 训练过程讲解

### 4.1 任务采样

```python
import numpy as np

class TaskDistribution:
    """任务分布采样器"""
    
    def __init__(self, num_classes=5, num_support=5, num_query=15):
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
    
    def sample_task(self):
        """采样一个任务"""
        
        # 随机选择类别
        classes = np.random.choice(
            range(self.num_classes * 2),
            size=self.num_classes,
            replace=False
        )
        
        # 采样支持集和查询集
        support_idx = np.random.choice(
            classes, size=(self.num_classes, self.num_support)
        )
        query_idx = np.random.choice(
            classes, size=(self.num_classes, self.num_query)
        )
        
        return {
            'classes': classes,
            'support_x': support_idx[:, :self.num_support],
            'support_y': support_idx[:, self.num_support],
            'query_x': query_idx[:, :self.num_query],
            'query_y': query_idx[:, self.num_query]
        }
    
    def sample_batch(self, batch_size):
        """采样批量任务"""
        return [self.sample_task() for _ in range(batch_size)]
```

### 4.2 MAML模型实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class MAML(nn.Module):
    """MAML元学习器"""
    
    def __init__(self, network, inner_lr=0.01, num_inner_steps=5):
        super().__init__()
        self.network = network
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
    
    def forward(self, x, params=None):
        """前向传播"""
        if params is None:
            params = dict(self.network.named_parameters())
        return self.network(x, params=params)
    
    def inner_update(self, support_x, support_y):
        """内层更新：任务适应"""
        
        # 复制一份参数
        params = {k: v.clone() for k, v in self.network.named_parameters()}
        
        # 几步梯度下降
        for _ in range(self.num_inner_steps):
            output = self.forward(support_x, params=params)
            loss = F.cross_entropy(output, support_y)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss, params.values(),
                create_graph=True,
                allow_unused=True
            )
            
            # ��度下降更新
            for (k, v), g in zip(params.items(), grads):
                if g is not None:
                    params[k] = params[k] - self.inner_lr * g
        
        return params
    
    def meta_loss(self, task):
        """计算任务损失"""
        
        support_x, support_y = task['support_x'], task['support_y']
        query_x, query_y = task['query_x'], task['query_y']
        
        # 内层更新
        adapted_params = self.inner_update(support_x, support_y)
        
        # 在查询集上计算损失
        query_output = self.forward(query_x, params=adapted_params)
        loss = F.cross_entropy(query_output, query_y)
        
        return loss
    
    def meta_update(self, tasks):
        """元更新"""
        
        total_loss = 0
        
        for task in tasks:
            loss = self.meta_loss(task)
            total_loss += loss
        
        # 反向传播
        self.network.zero_grad()
        total_loss.backward()
        
        # 返回优化器步
        return total_loss / len(tasks)
```

### 4.3 完整训练循环

```python
def train_maml(model, task_dist, meta_lr=0.001, num_iterations=10000):
    """MAML训练"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    
    for iteration in range(num_iterations):
        # 采样任务batch
        tasks = task_dist.sample_batch(batch_size=4)
        
        # 元更新
        loss = model.meta_update(tasks)
        
        optimizer.step()
        
        if iteration % 100 == 0:
            # 在测试任务上评估
            test_task = task_dist.sample_task()
            test_loss = model.meta_loss(test_task)
            print(f"Iter {iteration}, Loss: {loss:.4f}, Test: {test_loss:.4f}")
    
    return model
```

### 4.4 少样本评估

```python
@torch.no_grad()
def evaluate_few_shot(model, task_dist, num_episodes=100):
    """少样本分类评估"""
    
    accuracies = []
    
    for _ in range(num_episodes):
        task = task_dist.sample_task()
        
        # 内层更新
        adapted_params = model.inner_update(
            task['support_x'],
            task['support_y']
        )
        
        # 在查询集上评估
        query_output = model.forward(task['query_x'], params=adapted_params)
        preds = query_output.argmax(dim=1)
        
        accuracy = (preds == task['query_y']).float().mean()
        accuracies.append(accuracy.item())
    
    return {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies)
    }
```

### 4.5 超参数推荐

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| inner_lr | 任务学习率 | 0.01-0.1 |
| num_inner_steps | 内层步数 | 1-10 |
| meta_lr | 元学习率 | 0.001-0.01 |
| batch_size | 任务batch | 4-16 |
| num_iterations | 迭代次数 | 10000+ |

---

## 5. 应用场景

### 5.1 典型应用

- **少样本图像分类**：5-way 1-shot/5-shot
- **快速策略适应**：机器人新任务快速学习
- **药物发现**：新分子属性预测
- **个性化推荐**：用户冷启动

### 5.2 适用问题

- 多任务学习
- 任务分布明确
- 少量样本可用
- 需要快速适应

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 快速适应 | 几步梯度即可 |
| 通用性 | 适用于任何可微模型 |
| 理论保证 | 有收敛性证明 |
| 少样本 | 仅需少量样本 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 计算 | 两层梯度 | FOMAML |
| 内存 | 图构建 | 只保存一阶 |
| 任务 | 需要任务分布 | 预定义分布 |
| 泛化 | 分布外任务差 | 域随机化 |

---

## 7. 调库实现

### 7.1 Higher库

```python
# pip install higher
import higher

def use_higher_maml():
    """使用higher库的MAML"""
    
    import torch.nn as nn
    
    net = nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    # 使用higher进行内层更新
    with higher.inner_loop_ctx(net, torch.optim.Adam(net.parameters(), lr=0.01)) as fast_net:
        # 快速梯度更新
        output = fast_net(x)
        loss = F.mse_loss(output, y)
        fast_net.step(loss)
    
    return net
```

### 7.2 PyTorch Geometric

```python
def use_pyg_maml():
    """使用PyG的少样本学习"""
    
    from torch_geometric.nn import MetaLayer
    
    layer = MetaLayer(edge_model=None, node_model=None, global_model=None)
    
    return layer
```

---

## 8. 手工代码实现

### 8.1 简化NumPy实现

```python
import numpy as np

class SimpleMAML:
    """简化MAML NumPy实现"""
    
    def __init__(self, param_dim, inner_lr=0.01):
        self.param = np.random.randn(param_dim) * 0.1
        self.inner_lr = inner_lr
    
    def predict(self, X, params=None):
        """预测"""
        if params is None:
            params = self.param
        return X @ params
    
    def inner_update(self, X, y):
        """内层更新"""
        
        pred = self.predict(X)
        loss = np.mean((pred - y) ** 2)
        
        # 梯度
        grad = 2 * X.T @ (pred - y) / len(y)
        
        # 更新
        self.param = self.param - self.inner_lr * grad
        
        return self.param
    
    def meta_loss(self, X_sup, y_sup, X_qry, y_qry):
        """元损失"""
        
        params = self.inner_update(X_sup, y_sup)
        
        pred = self.predict(X_qry, params)
        loss = np.mean((pred - y_qry) ** 2)
        
        return loss
```

---

## 9. 可视化与结果理解

### 9.1 训练曲线

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(train_losses, test_losses):
    """训练曲线"""
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('MAML Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('training.png', dpi=150)
    plt.show()
```

### 9.2 任务适应曲线

```python
def plot_adaptation_curve(model, task):
    """任务适应曲线"""
    
    losses = []
    params = {k: v.clone() for k, v in model.named_parameters()}
    
    for step in range(10):
        output = model(task['support_x'], params=params)
        loss = F.cross_entropy(output, task['support_y'])
        
        grads = torch.autograd.grad loss, params.values())
        
        for k, v in params.items():
            params[k] = params[k] - model.inner_lr * grads[k]
        
        losses.append(loss.item())
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o')
    plt.xlabel('Inner Step')
    plt.ylabel('Loss')
    plt.title('Task Adaptation')
    plt.savefig('adaptation.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 少样本分类准确率

```python
def evaluate_way_shot(model, task_dist, ways, shots):
    """N-way K-shot评估"""
    
    results = {}
    
    for way in ways:
        for shot in shots:
            task_dist.num_classes = way
            task_dist.num_support = shot
            
            acc = evaluate_few_shot(model, task_dist)
            results[f'{way}-way-{shot}-shot'] = acc
    
    return results
```

### 10.2 基线对比

- **MAML**
- **Reptile**
- **预训练 + Fine-tune**
- **从头训练**

---

## 11. 常见问题与易错点

### 11.1 内层步数选择

**问题**：步数太多？

**解决**：通常1-5步即可

### 11.2 外层梯度二阶项

**解决**：使用FOMAML忽略二阶项

---

## 12. 学习总结

### 12.1 核心要点

1. **双层优化**：内层+外层
2. **快速适应**：几步梯度
3. **初始化**：好的起点
4. **任务分布**：关键

### 12.2 进阶方向

- **Meta-SGD**：学习内层学习率
- **Meta-LSTM**：学习优化器

---

## 13. 练习题与思考题

### 练习题

**练习1**：MAML vs 预训练+Fine-tune

<details>
<summary>答案</summary>

MAML学习如何学习，预训练学习任务。

</details>

### 思考题

**思考题1**：内层步数影响？

<details>
<summary>答案</summary>

步数多，泛化更好但计算更大。

</details>

---

## 14. 学习路径建议

### 第一阶段（1-2天）

1. 元学习概念
2. MAML原理
3. 代码理解

### 第二阶段（2-3天）

1. 实现MAML
2. 少样本实验
3. 对比baseline

### 第三阶段（3-5天）

1. 改进变体
2. 实际应用
3. 项目实战

### 推荐资源

- **论文**：《Model-Agnostic Meta-Learning》
- **代码**：higher库
- **项目**：少样本学习

---

*MAML是元学习的里程碑，让模型学会学习。*

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
