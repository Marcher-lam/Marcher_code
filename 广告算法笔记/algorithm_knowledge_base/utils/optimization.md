# 优化基础

## 一、梯度下降

梯度下降是参数优化的核心方法。核心思想：沿损失函数梯度反方向更新参数，逐步逼近极小值。

**为什么沿梯度反方向？** 梯度是函数增长最快的方向，反方向则是下降最快的方向。

### 三种实现方式

| 方式 | 每步样本数 | 优点 | 缺点 |
|------|-----------|------|------|
| Batch GD | 全部 N | 收敛稳定 | 速度慢，无法在线更新 |
| Mini-Batch GD | B (32~512) | 兼顾速度与稳定 | 需调 batch_size |
| SGD | 1 | 速度快，可在线学习 | 震荡剧烈 |

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
loss_fn = nn.MSELoss()

optimizers = {
    "Batch GD": torch.optim.SGD(model.parameters(), lr=0.01),       # 全量数据
    "Mini-Batch": torch.optim.SGD(model.parameters(), lr=0.01),     # batch=64
    "SGD": torch.optim.SGD(model.parameters(), lr=0.01),            # batch=1
}

# Mini-Batch 训练示例
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
batch_size = 64

for epoch in range(10):
    indices = torch.randperm(len(X))
    for i in range(0, len(X), batch_size):
        batch_X = X[indices[i:i + batch_size]]
        batch_y = y[indices[i:i + batch_size]]
        loss = loss_fn(model(batch_X), batch_y)
        optimizers["Mini-Batch"].zero_grad()
        loss.backward()
        optimizers["Mini-Batch"].step()
```

### 常用优化器

#### SGD + Momentum

**为什么需要 Momentum？** 标准 SGD 在"峡谷"地形中反复震荡，Momentum 引入"惯性"，沿历史方向平滑前进，加速收敛。

$$v_t = \beta v_{t-1} + \nabla L(w_t)$$
$$w_{t+1} = w_t - \eta \cdot v_t$$

- $\beta$ 通常取 0.9，控制历史梯度保留比例
- 物理意义：类似球滚下山坡，积累动量

```python
optimizer_momentum = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### Adam

**为什么 Adam 是工业界标配？** 它结合了 Momentum（一阶矩）和 RMSProp（二阶矩），自动调整每个参数的学习率，对超参数不敏感。

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(w_t) \quad \text{(一阶矩，梯度的指数移动平均)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(w_t))^2 \quad \text{(二阶矩，梯度平方的指数移动平均)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(偏差校正)}$$
$$w_{t+1} = w_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

默认 $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$。

```python
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

#### FTRL

**为什么广告系统偏爱 FTRL？** FTRL（Follow-The-Regularized-Leader）天然产生稀疏解，适合超大规模稀疏特征（如用户ID、广告ID），节省线上内存和计算。

$$w_{t+1} = \arg\min_w \left( \sum_{i=1}^{t} g_i \cdot w + \frac{1}{2} \sum_{i=1}^{t} \sigma_i \|w\|^2 + \lambda_1 \|w\|_1 \right)$$

- $g_i$：第 $i$ 步梯度
- $\lambda_1$：L1 正则系数，**控制稀疏性**，越大越稀疏
- 逐坐标闭式解：$w_j = \begin{cases} 0 & |z_j| \le \lambda_1 \\ -\frac{z_j - \text{sign}(z_j)\lambda_1}{\eta_j} & \text{otherwise} \end{cases}$

```python
# FTRL 在 PyTorch 中无内置实现，业界通常自行实现
import numpy as np

class FTRL:
    def __init__(self, dim, alpha=0.05, beta=1.0, l1=1.0, l2=1.0):
        self.dim = dim
        self.z = np.zeros(dim)
        self.n = np.zeros(dim)
        self.alpha, self.beta, self.l1, self.l2 = alpha, beta, l1, l2
        self.w = np.zeros(dim)

    def update(self, grad):
        for j in range(self.dim):
            sigma = (np.sqrt(self.n[j] + grad[j] ** 2) - np.sqrt(self.n[j])) / self.alpha
            self.z[j] += grad[j] - sigma * self.w[j]
            self.n[j] += grad[j] ** 2
            if abs(self.z[j]) <= self.l1:
                self.w[j] = 0
            else:
                self.w[j] = -(self.z[j] - np.sign(self.z[j]) * self.l1) / \
                             (self.beta + (self.alpha + np.sqrt(self.n[j])) / self.alpha + self.l2)
```

### 学习率策略

**为什么需要学习率调度？** 固定学习率要么前期收敛慢（太小），要么后期震荡（太大）。调度策略让学习率"先大后小"。

**Warmup**：训练初期梯度不稳定，先用小学习率"热身"，避免初期梯度破坏预训练权重。

$$\eta_t = \eta_{base} \cdot \min\left(1, \frac{t}{T_{warmup}}\right)$$

**余弦退火（Cosine Decay）**：平滑衰减到接近零。

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

```python
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math

# Linear Warmup
def warmup_fn(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

scheduler_warmup = LambdaLR(optimizer_adam, lr_lambda=warmup_fn)

# Cosine Decay
scheduler_cosine = CosineAnnealingLR(optimizer_adam, T_max=100, eta_min=1e-6)
```

## 二、正则化

**为什么需要正则化？** 模型在训练集上学到"噪声"即为过拟合，正则化通过约束模型复杂度提升泛化能力。

### L1 正则化 (Lasso)

$$L = L_{loss} + \lambda \sum_{i} |w_i|$$

**为什么 L1 产生稀疏解？** L1 的等高线是菱形，与损失函数等高线更容易在坐标轴上相切（即某个 $w_i=0$），从而将不重要的特征权重直接压为 0。

```python
# PyTorch 中手动添加 L1 正则
def l1_penalty(model, lambda_l1=1e-4):
    return lambda_l1 * sum(p.abs().sum() for p in model.parameters())

loss = loss_fn(model(X), y) + l1_penalty(model, lambda_l1=0.001)
loss.backward()
```

### L2 正则化 (Weight Decay)

$$L = L_{loss} + \lambda \sum_{i} w_i^2$$

**为什么 L2 防止过拟合？** L2 惩罚大权重，迫使模型使用所有特征但每个贡献较小，避免依赖单个特征。从贝叶斯角度看，L2 等价于对权重施加高斯先验。

```python
# 方式一：optimizer 的 weight_decay 参数（等价于 L2）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 方式二：手动添加
def l2_penalty(model, lambda_l2=1e-4):
    return lambda_l2 * sum((p ** 2).sum() for p in model.parameters())
```

### Dropout

**为什么 Dropout 有效？** 训练时随机丢弃神经元，迫使网络不依赖任何单一神经元，相当于训练了多个子网络的集成。

```python
drop = nn.Dropout(p=0.5)
x = torch.randn(4, 10)
out_train = drop(x)       # 训练：约 50% 置零，其余值 ×2（inverted dropout）
drop.eval()
out_test = drop(x)         # 推理：不丢弃
```

### Label Smoothing

**为什么防止过度自信？** 硬标签 `[0, 0, 1]` 驱使模型输出极端概率，Label Smoothing 将目标软化，提升泛化。

$$y_i^{smooth} = (1 - \epsilon) \cdot y_i + \frac{\epsilon}{K}$$

```python
import torch.nn.functional as F

def label_smoothing_loss(pred, target, num_classes, smoothing=0.1):
    log_probs = F.log_softmax(pred, dim=-1)
    smooth_target = torch.zeros_like(log_probs).scatter(
        1, target.unsqueeze(1), 1
    )
    smooth_target = smooth_target * (1 - smoothing) + smoothing / num_classes
    return (-smooth_target * log_probs).sum(dim=-1).mean()
```

### Early Stopping

**原理**：监控验证集损失，连续若干轮不下降则停止训练，防止过拟合。

```python
best_val_loss = float("inf")
patience, counter = 5, 0

for epoch in range(100):
    train_loss = train_one_epoch()
    val_loss = evaluate()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best.pt")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stop at epoch {epoch}")
            break
```

## 三、过拟合 vs 欠拟合

### 判断方法

通过观察 **训练/验证损失曲线** 判断：

| 状态 | Train Loss | Val Loss | 曲线特征 |
|------|-----------|----------|---------|
| 欠拟合 | 高 | 高 | 两条曲线都很高且接近 |
| 良好 | 低 | 低 | 两条曲线都低且接近 |
| 过拟合 | 低 | 高 | Train 持续下降，Val 开始上升 |

### 解决方案

**过拟合**：增加数据 / 数据增强 / L1&L2 正则化 / Dropout / Early Stopping / 降低模型复杂度 / Label Smoothing

**欠拟合**：增加模型层数或维度 / 减小正则化强度 / 增加特征 / 训练更多轮次 / 使用更好的优化器

## 四、广告系统优化技巧

### 在线学习 (FTRL Streaming)

**为什么需要在线学习？** 广告系统中用户兴趣和竞价环境实时变化，离线模型会"过时"。FTRL 支持逐条样本更新参数，实现分钟级模型刷新。

```
# 伪代码：FTRL Streaming 更新
for sample in stream:                      # 逐条接收实时日志
    features = extract_features(sample)     # 特征提取
    pred = predict(features, w)             # 预测
    grad = compute_gradient(pred, label)    # 计算梯度
    ftrl.update(grad)                       # FTRL 更新参数
    if batch_count % 10000 == 0:
        save_checkpoint(w)                  # 定期保存
```

### 负采样策略

广告场景中点击率通常 < 1%，正负样本极度不平衡。

- **随机负采样**：按固定比例随机抽取负样本（如 1:10），简单有效
- **Hard Negative Mining**：挑选模型容易预测错的样本作为负样本，提升判别能力

```python
import random

def random_neg_sample(pos_items, all_items, neg_ratio=5):
    neg_items = random.sample(list(set(all_items) - set(pos_items)), neg_ratio * len(pos_items))
    return neg_items

def hard_neg_mining(model, query, all_items, top_k=10):
    scores = model.predict(query, all_items)
    hard_negs = sorted(zip(all_items, scores), key=lambda x: -x[1])[:top_k]
    return [item for item, _ in hard_negs if item not in query.positives]
```

### 梯度裁剪

**为什么需要梯度裁剪？** RNN/Transformer 中容易出现梯度爆炸，裁剪梯度范数到阈值内，保证训练稳定。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 在训练循环中使用
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```
