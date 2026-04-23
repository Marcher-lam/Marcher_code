# 机器学习优化方法详解

> 梯度下降、学习率、正则化、过拟合与欠拟合完整指南

---

## 目录

1. [梯度下降基础](#1-梯度下降基础)
2. [学习率调优](#2-学习率调优)
3. [正则化方法](#3-正则化方法)
4. [过拟合与欠拟合](#4-过拟合与欠拟合)
5. [优化算法进阶](#5-优化算法进阶)
6. [实战代码示例](#6-实战代码示例)
7. [常见问题与调优技巧](#7-常见问题与调优技巧)

---

## 1. 梯度下降基础

### 1.1 什么是梯度下降

**核心思想**: 沿着损失函数的负梯度方向更新参数，逐步到达最优解

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta)$$

**其中**:
- θ: 模型参数
- η (eta): 学习率（步长）
- ∇θJ(θ): 损失函数对θ的梯度
- t: 迭代次数

**直观理解**:
```
                    ↑ 梯度方向
                    │
                    │
                    ▼
              ───●───→  逐步下山
            损失函数曲面
```

### 1.2 批量梯度下降（Batch GD）

**定义**: 每次迭代使用全部训练样本计算梯度

**优点**:
- 收敛稳定
- 每步方向准确

**缺点**:
- 计算量大（需遍历所有样本）
- 不适合大规模数据集

**适用场景**:
- 小数据集（<10K样本）
- 内存充足
- 需要稳定收敛

### 1.3 随机梯度下降（SGD）

**定义**: 每次迭代随机选一个样本计算梯度

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta; x^{(i)}, y^{(i)})$$

**优点**:
- 计算快（每步只算一个样本）
- 能跳出局部最优
- 适合在线学习

**缺点**:
- 收敛不稳定（震荡）
- 需要调学习率

**适用场景**:
- 大规模数据集
- 在线更新需求
- 深度学习首选

### 1.4 Mini-batch GD

**定义**: 每次迭代使用一小批样本计算梯度

**Batch Size选择**:
- 16, 32, 64, 128, 256, 512

**优点**:
- 平衡计算效率和收敛稳定性
- 可并行化（GPU友好）
- 现代标准做法

**在推荐中**:
- CTR预估常用batch size: 1024-4096
- 序列模型常用: 256-512
- 图模型常用: 256-1024

---

## 2. 学习率调优

### 2.1 学习率过大的问题

**表现**:
- 损失函数震荡
- 无法收敛
- 可能发散

```
大学习率
Loss ↗  ↘  ↗  ↘  ↗  （震荡不收敛）
     └────────────────────────→ 迭代次数
```

**症状**:
- Loss忽高忽低
- 梯度爆炸（NaN）
- 模型性能下降

### 2.2 学习率过小的问题

**表现**:
- 收敛极其缓慢
- 训练时间长
- 容易陷入局部最优

```
小学习率
Loss ↘  ↘  ↘  （收敛太慢）
     └────────────────────────→ 迭代次数
```

**症状**:
- 需要很多轮次才能收敛
- 损失下降曲线平缓
- 训练时间过长

### 2.3 学习率衰减策略

**阶梯式衰减（Step Decay）**:
$$\eta_t = \eta_0 \times \gamma^{\lfloor t / k \rfloor}$$

其中：
- η₀: 初始学习率
- γ: 衰减因子（0.9-0.99）
- k: 衰减周期（epoch数）

**指数衰减（Exponential Decay）**:
$$\eta_t = \eta_0 \times \gamma^t$$

**余弦退火（Cosine Annealing）**:
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})[1 + \cos(\frac{\pi t}{T})]$$

### 2.4 学习率搜索策略

**Warmup**: 先小学习率逐步增大
- 前5%步数：η从0到η_target线性增长
- 防止初始化不稳定

**网格搜索（Grid Search）**:
- 预设学习率列表: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
- 每个学习率训练少量轮次
- 选择验证集最好的

**推荐学习率表**:
| 场景 | 初始学习率 | 衰减策略 |
|-------|-----------|----------|
| 深度学习（DNN） | 1e-3 | Cosine + Warmup |
| GBDT/XGBoost | 0.05-0.1 | 每轮降低 |
| 逻辑回归 | 0.1-1.0 | 固定或衰减 |
| 梯度提升树 | 0.05 | Step decay |
| 强化学习 | 1e-4 | Adam调度器 |

---

## 3. 正则化方法

### 3.1 为什么需要正则化

**问题**: 模型复杂度过高

**表现**:
- 训练集表现好，测试集差
- 过度拟合训练数据
- 泛化能力差

**原因**:
- 参数过多，模型太复杂
- 数据噪声被拟合为规律
- 特征与样本比例失衡

### 3.2 L2正则化（Ridge）

**定义**: 损失函数加参数平方和

$$J(\theta) = \frac{1}{n}\sum_{i=1}^{n}L(y_i, \hat{y}_i) + \lambda \sum_{j=1}^{m}\theta_j^2$$

**特点**:
- 参数趋向于小但不为零
- 所有参数均匀收缩
- 保留所有特征

**适用场景**:
- 特征间有相关性
- 不希望特征选择
- 大部分场景通用

**在推荐中**:
- 逻辑回归CTR预估（常用）
- 矩阵分解（BiasSVD）
- 线性模型正则化

### 3.3 L1正则化（LASSO）

**定义**: 损失函数加参数绝对值和

$$J(\theta) = \frac{1}{n}\sum_{i=1}^{n}L(y_i, \hat{y}_i) + \lambda \sum_{j=1}^{m}|\theta_j|$$

**特点**:
- 产生稀疏解（很多参数为0）
- 自动特征选择
- 计算困难（不可导）

**适用场景**:
- 高维稀疏特征
- 需要特征筛选
- 希望简化模型

**在推荐中**:
- 特征工程筛选
- 用户/物品ID类特征
- 词袋模型文本特征

### 3.4 Elastic Net

**定义**: L1和L2正则化的加权组合

$$J(\theta) = \frac{1}{n}\sum_{i=1}^{n}L(y_i, \hat{y}_i) + \lambda_1 \sum_{j=1}^{m}|\theta_j| + \lambda_2 \sum_{j=1}^{m}\theta_j^2$$

**特点**:
- 结合L1和L2优点
- 兼顾稀疏性和稳定性
- 通过α参数调节L1/L2权重

$$\lambda_1 = \alpha \lambda, \lambda_2 = \frac{1-\alpha}{2}\lambda$$

**在推荐中**:
- 高维特征建模
- 需要部分特征选择
- 大规模特征工程

### 3.5 Dropout（神经网络）

**定义**: 训练时随机"丢弃"神经元

**实现**:
```python
class DropoutLayer(nn.Module):
    def __init__(self, p=0.5):
        self.p = p  # 丢弃概率
        
    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.shape) > self.p).float()
            return x * mask / (1 - self.p)
        return x
```

**特点**:
- 训练时：随机丢弃部分节点
- 推理时：使用全部节点（不dropout）
- 类似模型集成

**在推荐中**:
- 深度排序模型（DeepFM, xDeepFM）
- 用户行为序列模型（DIEN, BST）
- Transformer模型

---

## 4. 过拟合与欠拟合

### 4.1 过拟合（Overfitting）

**定义**: 训练集表现很好，测试/验证集表现差

**症状**:
- ✅ 训练集Loss: 0.01
- ❌ 验证集Loss: 0.50（差距巨大）
- ❌ 在线指标不提升

**可视化**:
```
准确率
100%│╭──── 训练集 ───╮
    │     高性能     │
 80%│╰────────────────╯
    │
    │  ╭──── 验证集 ───╮
    │  性能低        │
 60%│  ╰────────────────╯
    │
    └──────────────────→ 模型复杂度增加
```

**原因**:
1. **数据太少** → 模型记住了噪声
2. **特征太多** → 模型过于复杂
3. **训练太长** → 过拟合训练数据

**在推荐中的例子**:
- 用户行为数据稀疏，模型过拟合少数活跃用户
- 物品特征维度高，模型记忆了所有物品
- CTR模型对少数热门物品过拟合

### 4.2 欠拟合（Underfitting）

**定义**: 训练集、验证集、测试集表现都不好

**症状**:
- ✅ 训练集Loss: 1.20（高）
- ✅ 验证集Loss: 1.25（也高）
- ❌ 在线指标远低于基线

**可视化**:
```
准确率
100%│
    │
 80%│    ╭──── 两者都低 ───╮
    │  性能不足          │
 60%│    ╰────────────────╯
    │
    └──────────────────→ 模型复杂度低
```

**原因**:
1. **模型太简单** → 无法捕捉复杂模式
2. **特征太少** → 信息不足
3. **训练不足** → 没学到规律

**在推荐中的例子**:
- 线性模型无法捕捉用户非线性兴趣
- 特征工程不足，模型决策边界简单
- 树模型深度太浅

### 4.3 Bias-Variance权衡

**定义**: 模型误差可分解为Bias² + Variance + Noise

$$E[(y - \hat{y})^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

**Bias**:
- 模型假设过于简化
- 表现：欠拟合
- 解决：增加模型复杂度

**Variance**:
- 模型对训练数据微小变化敏感
- 表现：过拟合
- 解决：减少复杂度，增加数据

**权衡曲线**:
```
测试误差
   │           ╱╲ 最优
   │         ╱  ╲
   │       ╱      ╲  Bias²+Variance最小
   │     ╱        ╲
   │   ╱  欠拟合   ╲ 过拟合
   │  ╱            ╲
   └──╱────────────────╲→ 模型复杂度
    低 ──────── 高
```

**在推荐中的应用**:
- 树模型深度选择：浅树欠拟合，深树过拟合
- 神经网络层数：少层欠拟合，多层过拟合
- 正则化强度：λ太小欠拟合，λ太大过拟合

---

## 5. 优化算法进阶

### 5.1 动量法（Momentum）

**思想**: 累积历史梯度方向，加速收敛

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta_t)$$

$$\theta_{t+1} = \theta_t - v_t$$

**其中**:
- γ (gamma): 动量系数（0.9-0.99）
- v: 速度（累加的梯度）

**优点**:
- 减少震荡
- 加速收敛
- 跳出局部最优

**在推荐中**:
- 深度学习训练标配
- 推荐库默认优化器
- 大规模模型训练稳定器

### 5.2 Adam优化器

**思想**: 结合动量和自适应学习率

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**其中**:
- β₁ (beta1): 一阶矩衰减（0.9）
- β₂ (beta2): 二阶矩衰减（0.999）
- g: 梯度
- ε (epsilon): 防止除零（1e-8）

**优点**:
- 自适应学习率（不同参数不同学习率）
- 结合动量加速
- 对超参数不敏感

**在推荐中**:
- ⭐ **最常用优化器**
- PyTorch默认推荐
- XGBoost默认优化策略

### 5.3 学习率调度器

**ReduceLROnPlateau**: 验证指标不下降时降低学习率

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',           # 监控指标变小
    factor=0.1,           # 降低到10%
    patience=5,            # 5轮不降才降低
    min_lr=1e-7
)
```

**CosineAnnealing**: 余弦退火到最小学习率

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,           # 总轮次
    eta_min=1e-6          # 最小学习率
)
```

**OneCycleLR**: 一个周期内先增后减

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,         # 最大学习率
    total_steps=10000,     # 总步数
    pct_start=0.3,       # 前30%warmup
)
```

### 5.4 早停法（Early Stopping）

**定义**: 验证集指标不再提升时停止训练

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience      # 允许不提升的轮次
        self.min_delta = min_delta   # 最小提升阈值
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
        else:
            self.best_score = val_score
            self.counter = 0
            
        return self.counter >= self.patience
```

**在推荐中**:
- 防止CTR模型过拟合
- 节省训练资源
- 标准实践配置：patience=5-10

---

## 6. 实战代码示例

### 6.1 梯度下降实现

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    批量梯度下降实现
    :param X: 特征矩阵 (n_samples, n_features)
    :param y: 标签 (n_samples,)
    :param learning_rate: 学习率
    :param epochs: 迭代轮次
    :return: 权重w, 偏置b
    """
    n_samples, n_features = X.shape
    
    # 初始化参数
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(epochs):
        # 预测
        y_pred = np.dot(X, w) + b
        
        # 计算梯度
        error = y_pred - y
        dw = (2 / n_samples) * np.dot(X.T, error)
        db = (2 / n_samples) * np.sum(error)
        
        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # 每100轮打印loss
        if epoch % 100 == 0:
            loss = np.mean(error ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w, b

# 使用示例
X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 5]])
y = np.array([3, 5, 7, 6, 8])

w, b = gradient_descent(X, y, learning_rate=0.01, epochs=1000)
print(f"权重: {w}, 偏置: {b}")
```

### 6.2 正则化对比

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 20)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.5

# 比较不同正则化
models = {
    '无正则化': LinearRegression(),
    'L2 (Ridge)': Ridge(alpha=1.0),
    'L1 (LASSO)': Lasso(alpha=1.0),
    'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(X, y)
    # 计算非零权重（L1会产生稀疏解）
    if hasattr(model, 'coef_'):
        nonzero = np.sum(np.abs(model.coef_) > 1e-6)
        print(f"{name}: 非零特征数 = {nonzero}/20")
```

### 6.3 学习率调度器

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 简单模型
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ReduceLROnPlateau: 验证不降时降低学习率
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3
)

# Cosine退火
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=100, 
    eta_min=1e-6
)

# 训练循环
losses = []
for epoch in range(100):
    # 模拟训练
    loss = torch.randn(1)  # 实际这里应该计算真实loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # 更新学习率
    scheduler1.step(loss)
    scheduler2.step()
    
    if epoch % 20 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, LR: {current_lr:.6f}, Loss: {loss:.4f}")
```

### 6.4 早停法实现

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        if self.counter >= self.patience:
            self.early_stop = True
            print(f"Early stopping at loss {val_loss:.4f}")
            
        return self.early_stop

# 使用示例
early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

for epoch in range(1000):
    # 模拟训练和验证
    train_loss = np.random.rand() * 0.5 + 0.1
    val_loss = np.random.rand() * 0.3 + 0.1
    
    if early_stopping(val_loss):
        print(f"Stopped at epoch {epoch}")
        break
```

---

## 7. 常见问题与调优技巧

### 7.1 梯度消失/爆炸

**梯度消失（Vanishing Gradient）**:
- 问题：深层网络梯度逐层衰减到0
- 症状：浅层参数几乎不更新
- 解决：BatchNorm, ResNet, 残差连接

**梯度爆炸（Exploding Gradient）**:
- 问题：梯度值过大导致数值不稳定
- 症状：梯度变成NaN/Inf
- 解决：梯度裁剪（Gradient Clipping）

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 7.2 损失函数选择

| 任务 | 推荐损失函数 |
|------|-------------|
| 回归 | MSE, MAE, Huber Loss |
| 二分类（CTR） | Binary Cross-Entropy, LogLoss |
| 多分类 | Cross-Entropy, Focal Loss |
| 排序（BPR） | BPR Loss, LambdaRank |
| 序列推荐 | Cross-Entropy + Attention Mask |

**Focal Loss**（类别不平衡）:
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- γ (gamma): 聚焦困难样本（2-3）
- α (alpha): 类别权重（平衡正负样本）

### 7.3 批次大小选择

**原则**:
- GPU内存限制：最大batch size受显存限制
- 收敛稳定性：batch太小不稳定
- 训练速度：batch太慢，太小浪费计算

**经验规则**:
```python
# 根据GPU显存调整batch size
import torch

# 假设显存8GB
GPU_MEMORY = 8  # GB

if GPU_MEMORY >= 16:
    BATCH_SIZE = 4096
elif GPU_MEMORY >= 8:
    BATCH_SIZE = 2048
elif GPU_MEMORY >= 4:
    BATCH_SIZE = 1024
else:
    BATCH_SIZE = 256
```

### 7.4 优化检查清单

**训练前**:
- [ ] 损失函数选择正确吗？
- [ ] 初始化策略合理吗？
- [ ] 学习率初始化了吗？

**训练中**:
- [ ] Loss下降曲线是否平滑？
- [ ] 有异常值（NaN/Inf）？
- [ ] 验证集表现如何？

**调试技巧**:
- [ ] 先在小数据集上验证代码
- [ ] 梯度检查（手动计算验证）
- [ ] 可视化Loss曲线
- [ ] 打印中间结果

### 7.5 推荐系统特定优化

**特征归一化**:
```python
# 数值特征标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

**负采样策略**:
- **重要性采样**: 对困难样本多采样
- **均匀采样**: 保证长尾物品有曝光
- **流行度纠偏**: 降低热门物品权重

**批次内平衡**:
```python
# 每个batch内保证正负样本平衡
def balanced_batch_generator(X, y, batch_size=1024):
    # 采样正负样本
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    # 按比例采样
    pos_size = batch_size // 4  # 25%正样本
    neg_size = batch_size - pos_size
    
    pos_batch = np.random.choice(pos_idx, pos_size, replace=True)
    neg_batch = np.random.choice(neg_idx, neg_size, replace=True)
    
    # 拼接batch
    batch_idx = np.concatenate([pos_batch, neg_batch])
    return X[batch_idx], y[batch_idx]
```

---

## 8. 学习总结

### 8.1 核心要点

1. **梯度下降**: 沿负梯度方向迭代更新参数
2. **学习率**: 控制步长，需要调优和衰减策略
3. **正则化**: L1（稀疏）vs L2（稳定）vs Elastic Net（平衡）
4. **过拟合**: 训练好测试差，增加数据/正则化/减少复杂度
5. **欠拟合**: 都不好，增加复杂度/增加特征/增加训练
6. **Bias-Variance**: 找到复杂度的平衡点

### 8.2 优化算法选择

| 场景 | 推荐算法 |
|-------|----------|
| 通用深度学习 | Adam + ReduceLROnPlateau + EarlyStopping |
| 传统机器学习 | SGD + Momentum + Learning Rate Decay |
| 强化学习 | Adam + 学习率调度 |
| 大规模推荐 | AdamW + BatchNorm + Warmup |
| 低延迟要求 | 小Batch + Aggressive Early Stopping |

### 8.3 超参数优先级

**高优先级**（影响最大）:
1. Learning Rate (学习率)
2. Batch Size (批次大小)
3. Model Capacity (模型容量）

**中优先级**（次重要）:
4. Regularization Strength (正则化强度)
5. Optimizer Choice (优化器选择）
6. Loss Function (损失函数）

**低优先级**（微调）:
7. Dropout Rate (丢弃率）
8. Weight Decay (权重衰减）
9. Gradient Clipping (梯度裁剪）

### 8.4 面试必答

**Q1: 学习率过大/过小的表现？**
A: 过大→Loss震荡不收敛；过小→收敛太慢

**Q2: L1和L2正则化的区别？**
A: L1产生稀疏解（特征选择），L2让所有参数收缩但不为零。

**Q3: 如何判断过拟合？**
A: 训练集远好于验证集，验证Loss不再下降甚至上升。

**Q4: 早停法的patience参数怎么设？**
A: 根据任务复杂度：简单任务3-5轮，复杂任务10-20轮。

---

## 9. 学习路径建议

```
梯度下降基础 → 学习率调优 → 正则化方法 → 过/欠拟合理解 → 优化器进阶（Adam/Momentum） → 学习率调度器 → 早停法 → 推荐系统优化实战
```

**推荐实践顺序**:
1. 先理解梯度下降基本原理
2. 手动实现简单优化器（SGD）
3. 学习Adam优化器及其变体（AdamW, Adamax）
4. 实践学习率调度（ReduceLROnPlateau, Cosine）
5. 掌握正则化（L1/L2/Dropout）
6. 理解Bias-Variance权衡
7. 在推荐系统中应用优化
