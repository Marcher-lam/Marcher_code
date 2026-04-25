# 优化算法详解

## 1. 梯度下降法 (Gradient Descent)

### 1.1 基本原理
通过迭代沿着目标函数梯度的反方向更新参数，以寻找函数的最小值。

### 1.2 数学公式
$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$$

其中：
- $\theta$：参数向量
- $\alpha$：学习率
- $\nabla J(\theta)$：目标函数关于参数的梯度

### 1.3 变体
- **批量梯度下降 (BGD)**：使用全部训练数据计算梯度
- **随机梯度下降 (SGD)**：每次使用一个样本更新参数
- **小批量梯度下降 (MBGD)**：使用小批量数据计算梯度（最常用）

### 1.4 代码实现
```python
import numpy as np

def gradient_descent(X, y, lr=0.01, n_iters=1000):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    
    for _ in range(n_iters):
        gradients = (2/n_samples) * X.T @ (X @ theta - y)
        theta -= lr * gradients
    
    return theta
```

## 2. 学习率调度 (Learning Rate Scheduling)

### 2.1 固定学习率
最简单的策略，但可能收敛慢或不稳定。

### 2.2 时间衰减
$$\alpha_t = \frac{\alpha_0}{1 + \text{decay} \cdot t}$$

### 2.3 1/t衰减
$$\alpha_t = \frac{\alpha_0}{t}$$

### 2.4 自适应方法
- **Adagrad**：为每个参数分配不同的学习率
- **RMSprop**：解决Adagrad学习率过早衰减问题
- **Adam**：结合动量和RMSprop的优点

## 3. 正则化方法 (Regularization)

### 3.1 L1正则化 (Lasso)
在损失函数中添加参数绝对值之和：
$$J_{\text{Lasso}}(\theta) = J(\theta) + \lambda \sum_{i=1}^n |\theta_i|$$

特点：产生稀疏解，特征选择效果好。

### 3.2 L2正则化 (Ridge)
在损失函数中添加参数平方和：
$$J_{\text{Ridge}}(\theta) = J(\theta) + \lambda \sum_{i=1}^n \theta_i^2$$

特点：参数趋于较小值，防止过拟合。

### 3.3 Elastic Net
结合L1和L2正则化：
$$J(\theta) = J(\theta) + \lambda_1 \sum |\theta_i| + \lambda_2 \sum \theta_i^2$$

## 4. 数值优化方法

### 4.1 牛顿法 (Newton's Method)
使用二阶导数信息，收敛速度快。

迭代公式：
$$\theta_{t+1} = \theta_t - H^{-1} \nabla J(\theta_t)$$

其中$H$是Hessian矩阵。

### 4.2 拟牛顿法 (Quasi-Newton Methods)
- **BFGS**：使用正定矩阵近似Hessian的逆
- **L-BFGS**：BFGS的低内存版本，适合大规模问题

### 4.3 共轭梯度法 (Conjugate Gradient)
结合梯度下降和牛顿法的优点，不需要计算Hessian矩阵。

## 5. 深度学习优化技巧

### 5.1 批归一化 (Batch Normalization)
对每一层的输入进行归一化，加速训练：
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

### 5.2 动量法 (Momentum)
引入动量项加速收敛：
$$v_{t+1} = \beta v_t + \alpha \nabla J(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

### 5.3 Nesterov动量
先进行一步预测，再计算梯度：
$$v_{t+1} = \beta v_t + \alpha \nabla J(\theta_t - \beta v_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

## 6. 收敛性分析

### 6.1 收敛条件
- 目标函数是凸的
- 学习率足够小但不过小
- 梯度有界

### 6.2 收敛速度
- 批量梯度下降：线性收敛
- 随机梯度下降：次线性收敛，O(1/√T)
- 牛顿法：二次收敛（接近最优时）

## 7. 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 损失不下降 | 学习率太大 | 减小学习率 |
| 收敛太慢 | 学习率太小 | 增大学习率或使用自适应方法 |
| 振荡 | 学习率不合适 | 使用动量或减小学习率 |
| 过拟合 | 模型复杂度过高 | 添加正则化、早停 |
| 梯度消失 | 深层网络 | 使用ReLU、批归一化、残差连接 |
| 梯度爆炸 | 梯度累积过大 | 梯度裁剪、学习率衰减 |

## 8. 实践建议

1. **学习率选择**：
   - 从较小值开始（如1e-3）
   - 使用学习率衰减策略
   - 考虑使用自适应优化器（Adam）

2. **正则化应用**：
   - L2正则化几乎总是有益的
   - Dropout用于防止过拟合
   - 数据增强也是一种正则化

3. **优化技巧**：
   - 批量大小影响噪声和内存使用
   - 梯度裁剪防止爆炸
   - Warm-up策略初期使用小学习率

4. **监控指标**：
   - 训练和验证损失
   - 参数范数
   - 梯度范数
   - 学习率

## 9. 代码模板

```python
import torch
import torch.optim as optim

# 常见优化器对比
model = MyModel()

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (自适应，通常效果较好)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW (Adam的改进版，权重衰减更合理)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 学习率调度
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)

# 训练循环
for epoch in range(n_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    scheduler.step(loss)
```

## 10. 高级优化技术

### 10.1 二阶优化
- 使用Hessian矩阵或其近似
- 计算代价高但收敛快
- 适用于小到中型问题

### 10.2 分层学习率
- 不同层使用不同的学习率
- 底层使用较小学习率
- 顶层使用较大学习率

### 10.3 课程学习 (Curriculum Learning)
- 从简单样本开始训练
- 逐渐增加难度
- 模仿人类学习过程

### 10.4 对抗训练
- 加入对抗样本
- 提高模型鲁棒性
- 防止对抗攻击

## 参考文献
1. Bottou, L. (2010). Large-Scale Machine Learning with Stochastic Gradient Descent
2. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
