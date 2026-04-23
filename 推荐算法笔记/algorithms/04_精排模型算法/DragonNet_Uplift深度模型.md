# DragonNet Uplift 深度模型

## 1. 算法基础认知

Uplift Modeling（增量建模）旨在估计**因果效应**——即某个干预（Treatment）对个体产生的增量效果。与传统预测模型不同，Uplift 关注的是 $P(Y|do(T=1), X) - P(Y|do(T=0), X)$。DragonNet 是一种基于深度神经网络的 Uplift 建模方法，通过端到端学习同时估计倾向得分和条件结果，实现双重稳健的因果效应估计。

## 2. Uplift 建模背景

### 2.1 核心问题

在营销、医疗、推荐等场景中，我们需要回答：

> "给这个用户发优惠券，比不发能多带来多少转化概率？"

这等价于估计**个体因果效应（ITE/CHTE）**：

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$$

### 2.2 基础方法对比

| 方法 | 策略 | 优点 | 缺点 |
|------|------|------|------|
| S-learner | 单模型，T 作为特征 | 简单 | T 信号被淹没 |
| T-learner | 两个独立模型 | 直观 | 两个模型误差叠加 |
| X-learner | 两阶段+倾向加权 | 减少偏差 | 复杂度高 |
| DragonNet | 端到端三头网络 | 双重稳健 | 需要正则化 |

## 3. DragonNet 详细架构

### 3.1 三头网络结构

DragonNet 由三部分组成：

1. **共享表示层** $Z(X)$：将输入特征映射为共享隐藏表示
2. **倾向得分头** $\hat{g}(x)$：预测 $P(T=1|X)$
3. **条件结果头** $\hat{Q}(t, x)$：分别预测 $t=0$ 和 $t=1$ 时的结果

```
输入 X
  │
共享表示层 Z(X)（多层 MLP）
  ├──────────┬──────────┐
  │          │          │
倾向得分头  结果头 t=0  结果头 t=1
  g(x)      Q(0,x)    Q(1,x)
```

### 3.2 关键设计思想

**为什么需要倾向得分头？**

根据因果推断理论，估计 ATE 只需要条件结果 $\hat{Q}(t,x)$，但加入倾向得分头 $\hat{g}(x)$ 有两个好处：
1. 正则化作用：迫使表示层学到与处理分配相关的特征
2. 实现双重稳健估计

**目标正则化（Targeted Regularization）**

引入扰动参数 $\epsilon$，对 $\hat{Q}$ 进行一阶修正：

$$\tilde{Q}(t, x) = \hat{Q}(t, x) + \epsilon \cdot \left(\frac{t}{\hat{g}(x)} - \frac{1-t}{1-\hat{g}(x)}\right)$$

## 4. 数学推导

### 4.1 损失函数

$$\hat{R}(\theta) = \frac{1}{n}\sum_{i=1}^{n} \left[ (Q^{nn}(t_i, x_i; \theta) - y_i)^2 + \alpha \cdot CE(g^{nn}(x_i; \theta), t_i) \right]$$

其中：
- 第一项是条件结果的 MSE 损失
- 第二项是倾向得分的交叉熵损失
- $\alpha$ 控制正则化强度

### 4.2 目标正则化损失

$$\hat{R}^{tar}(\theta, \epsilon) = \frac{1}{n}\sum_{i=1}^{n} \left[ (\tilde{Q}^{nn}(t_i, x_i; \theta, \epsilon) - y_i)^2 \right]$$

### 4.3 ATE 估计

训练完成后，ATE 通过以下公式估计：

$$\hat{\psi} = \frac{1}{n}\sum_{i=1}^{n} [\hat{Q}(1, x_i) - \hat{Q}(0, x_i)]$$

### 4.4 双重稳健性证明

若 $\hat{Q}$ 正确，ATE 直接无偏；若 $\hat{Q}$ 有偏但 $\hat{g}$ 正确，则目标正则化的 $\epsilon$ 修正项补偿偏差。因此只要两者之一正确，ATE 估计无偏。

## 5. 训练过程

1. 将 $(X, T, Y)$ 输入共享表示层，得到隐藏表示 $Z$
2. 倾向得分头从 $Z$ 预测 $\hat{g}(x)$
3. 两个结果头分别预测 $\hat{Q}(0, x)$ 和 $\hat{Q}(1, x)$
4. 根据 $t_i$ 选择对应结果头输出，计算 MSE + CE 联合损失
5. 第二阶段：固定其他参数，优化 $\epsilon$ 进行目标正则化
6. 用训练好的模型预测 $\hat{Q}(1, x) - \hat{Q}(0, x)$ 作为因果效应估计

## 6. 应用场景

| 场景 | Treatment | Outcome | 目标 |
|------|-----------|---------|------|
| 营销补贴 | 发/不发优惠券 | 是否下单 | 找到对补贴敏感的用户 |
| 广告投放 | 曝光/不曝光 | 转化率 | 评估广告真实增量价值 |
| 药物治疗 | 用药/不用药 | 治愈率 | 识别受益人群 |
| 推荐策略 | 推/不推某类内容 | 点击率 | 评估推荐干预效果 |

## 7. 优缺点分析

**优点**：
- 端到端训练，表示学习针对因果效应优化
- 双重稳健，对模型误设鲁棒
- 目标正则化提供渐近无偏保证
- 灵活的神经网络架构

**缺点**：
- 深度模型需要较多数据
- 超参数敏感（$\alpha$、目标正则化步数）
- 倾向得分接近 0 或 1 时估计不稳定
- 不适合小样本场景

## 8. PyTorch 代码实现

```python
import torch
import torch.nn as nn

class DragonNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[200, 100], alpha=0.01):
        super().__init__()
        self.alpha = alpha
        
        self.representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
        )
        
        self.propensity_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.outcome_t0 = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.outcome_t1 = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.epsilon = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        z = self.representation(x)
        g = self.propensity_head(z)
        q0 = self.outcome_t0(z)
        q1 = self.outcome_t1(z)
        return g, q0, q1
    
    def targeted_regularization(self, x, t, y, g, q0, q1):
        t_float = t.float().unsqueeze(-1)
        q_t = torch.where(t_float == 1, q1, q0)
        g_clamped = torch.clamp(g, min=1e-6, max=1 - 1e-6)
        h = t_float / g_clamped - (1 - t_float) / (1 - g_clamped)
        q_corrected = q_t + self.epsilon * h.detach()
        return nn.functional.mse_loss(q_corrected.squeeze(), y)
    
    def compute_loss(self, x, t, y):
        g, q0, q1 = self.forward(x)
        t_float = t.float().unsqueeze(-1)
        q_t = torch.where(t_float == 1, q1, q0)
        outcome_loss = nn.functional.mse_loss(q_t.squeeze(), y)
        propensity_loss = nn.functional.binary_cross_entropy(g.squeeze(), t.float())
        target_reg = self.targeted_regularization(x, t, y, g, q0, q1)
        return outcome_loss + self.alpha * propensity_loss + target_reg
    
    def predict_ite(self, x):
        g, q0, q1 = self.forward(x)
        return (q1 - q0).squeeze()


def train_dragonnet(model, X, T, Y, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    T_tensor = torch.tensor(T, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.compute_loss(X_tensor, T_tensor, Y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model
```

## 9. 与相关方法对比

| 方法 | 模型数量 | 偏差控制 | 数据需求 | 稳健性 |
|------|---------|---------|---------|--------|
| S-learner | 1 | 弱 | 低 | 低 |
| T-learner | 2 | 中 | 中 | 中 |
| X-learner | 3 | 强 | 中 | 中 |
| DragonNet | 1（多头部） | 强（DR） | 高 | 高 |

## 10. 常见问题与易错点

1. **混淆协变量与碰撞变量**：输入 X 应只包含预处理协变量，不能包含受 T 影响的变量
2. **倾向得分未 clamp**：$\hat{g}(x)$ 过小导致 IPW 权重爆炸，必须 clamp 到 $[\epsilon, 1-\epsilon]$
3. **$\alpha$ 设置过大**：倾向损失过强会使表示层过度关注处理分配而非结果预测
4. **忽略目标正则化**：不加目标正则化的 DragonNet 退化为普通 T-learner
5. **评估指标选择**：不能用 AUC 评估 Uplift 模型，应使用 AUUC（Area Under Uplift Curve）或 Qini Curve

## 11. 学习总结

DragonNet 的核心洞察是：将倾向得分建模与条件结果建模统一到一个端到端网络中，通过共享表示层使因果效应估计更加高效。目标正则化进一步提供了双重稳健的理论保证。适合中大规模数据集的 Uplift 建模场景。

## 12. 学习路径建议

- **前置知识**：因果推断基础、反事实框架、倾向得分
- **进阶方向**：CEVAE、TARNet、Orthogonal Forest、双重机器学习
- **推荐论文**：DragonNet (NeurIPS 2019)、TARNet (ICML 2017)
