# DESCN（Deep Entire Space Cross Networks）全空间交叉网络 Uplift 模型

## 1. 论文信息

- **论文**：*DESCN: Deep Entire Space Cross Networks for Individual Uplift Estimation*
- **链接**：https://arxiv.org/pdf/2207.09920
- **单位**：阿里巴巴

## 2. 算法基础认知

DESCN 是阿里巴巴提出的 Uplift 建模框架，用于预估**个体处理效应（ITE, Individual Treatment Effect）**，即对用户施加某种处理（如发券）相对于不处理（不发券）带来的增量效果。

### 核心概念

- **Uplift**：处理组与对照组之间的因果效应差异
- **倾向评分（Propensity Score）**：$\pi(X) = p(T=1|X)$，用户被分配到处理组的概率
- **ITE**：$\tau(X) = \mu_1(X) - \mu_0(X)$，个体层面的处理效应

## 3. 问题背景：为什么 Uplift 建模需要特殊处理

### 3.1 传统响应模型的局限

传统模型预测 $p(Y=1|X)$，即用户购买的概率。但这混淆了两种情况：
- **自然转化**：用户本来就打算购买（即使不发券也会买）
- **增量转化**：因为发了券才购买

传统模型无法区分这两种情况，导致营销资源浪费在"本就会购买"的用户上。

### 3.2 样本偏差问题

Uplift 建模中：
- 我们能观察到处理组响应 $\mu_1(X)$，但无法观察到其对照响应 $\mu_0(X)$
- 我们能观察到对照组响应 $\mu_0(X)$，但无法观察到其处理响应 $\mu_1(X)$
- 即存在**反事实（Counterfactual）**问题

### 3.3 全空间训练需求

在随机化实验中，处理组和对照组的样本比例通常不均衡，且特征分布可能有差异。全空间建模确保模型在两种条件下都能准确预估。

## 4. 核心架构

### 4.1 整体架构

```
               Input Features X
                     |
              [Embedding Layer]
               /      |      \
         [Tower T]  [Tower μ₁]  [Tower μ₀]
              |          |           |
           π(X)      μ₁(X)      μ₀(X)
              |          |           |
         ESTR=μ₁·π  X-Network   ESCR=μ₀·(1-π)
              |      /      \        |
              |   μ₁'(cross) μ₀'(cross)
              |          \   /
              +--- 合并损失 ---+
```

### 4.2 ESN（全空间网络）

ESN（Entire Space Network）联合建模三个量：
- 倾向评分：$\pi(X) = p(T=1|X)$
- 处理组响应：$\mu_1(X) = p(Y=1|T=1, X)$
- 对照组响应：$\mu_0(X) = p(Y=1|T=0, X)$

为了实现全空间训练，引入以下变换：

$$ESTR = \mu_1(X) \cdot \pi(X) = p(Y=1, T=1|X)$$

$$ESCR = \mu_0(X) \cdot (1-\pi(X)) = p(Y=1, T=0|X)$$

**关键洞察**：ESTR 和 ESCR 的标签在全部样本上都有定义——每个样本要么是处理组（$T=1$），要么是对照组（$T=0$）。这消除了反事实样本缺失的问题。

### 4.3 X-Network（交叉网络）

X-Network 引入伪处理效应 $\tau'(X)$，通过交叉计算反事实结果：

$$\mu_1'(X) = \sigma\left(\sigma^{-1}(\mu_0(X)) + \sigma^{-1}(\tau'(X))\right)$$

$$\mu_0'(X) = \sigma\left(\sigma^{-1}(\mu_1(X)) - \sigma^{-1}(\tau'(X))\right)$$

**为什么用 logit 空间（$\sigma^{-1}$）而非概率空间做加减？**

在概率空间直接加减可能导致值超出 $[0, 1]$。在 logit 空间做加减等价于概率空间做乘除，保证结果仍在有效范围内。

### 4.4 X-Network 的信息流动

```
处理组样本 (T=1, Y=y):
  μ₁ 有真实标签 → 学习 μ₁
  通过 τ' 交叉计算 μ₀' → 提供对照信息的梯度信号

对照组样本 (T=0, Y=y):
  μ₀ 有真实标签 → 学习 μ₀
  通过 τ' 交叉计算 μ₁' → 提供处理信息的梯度信号
```

这种交叉机制使得处理组和对照组的信息能够互相补充。

## 5. 损失函数

$$L_{DESCN} = \alpha L_\pi + \beta_1 L_{ESTR} + \beta_0 L_{ESCR} + \gamma_1 L_{CrossTR} + \gamma_0 L_{CrossCR}$$

各项含义：

| 损失项 | 标签 | 定义 |
|--------|------|------|
| $L_\pi$ | $T$ | 倾向评分的交叉熵损失 |
| $L_{ESTR}$ | $\mathbb{1}[T=1] \cdot Y$ | 处理组全空间响应损失 |
| $L_{ESCR}$ | $\mathbb{1}[T=0] \cdot Y$ | 对照组全空间响应损失 |
| $L_{CrossTR}$ | 处理组真实响应 | 交叉网络对处理组的预估损失 |
| $L_{CrossCR}$ | 对照组真实响应 | 交叉网络对对照组的预估损失 |

其中 ESTR 和 ESCR 的具体形式：

$$L_{ESTR} = -\frac{1}{N}\sum_{i:T_i=1}\left[y_i \log(\mu_1(x_i) \cdot \pi(x_i)) + (1-y_i)\log(1 - \mu_1(x_i) \cdot \pi(x_i))\right]$$

$$L_{ESCR} = -\frac{1}{N}\sum_{i:T_i=0}\left[y_i \log(\mu_0(x_i) \cdot (1-\pi(x_i))) + (1-y_i)\log(1 - \mu_0(x_i) \cdot (1-\pi(x_i)))\right]$$

## 6. 训练过程讲解

1. **数据准备**：每个样本包含特征 $X$、处理标签 $T$（0/1）、响应标签 $Y$（0/1）
2. **前向传播**：
   - 特征经 Embedding 层得到稠密表示
   - 三个塔分别输出 $\pi(X)$、$\mu_1(X)$、$\mu_0(X)$
   - 计算 ESTR、ESCR
   - X-Network 计算 $\mu_1'$、$\mu_0'$ 和 $\tau'$
3. **损失计算**：五项损失的加权和
4. **反向传播**：联合更新所有参数
5. **推理**：Uplift = $\mu_1(X) - \mu_0(X)$

### 推理时的 Uplift 计算

$$\hat{\tau}(X) = \mu_1(X) - \mu_0(X)$$

选择 $\hat{\tau}(X) > 0$ 且值最大的用户进行营销投放。

## 7. PyTorch 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationLayer(nn.Module):
    def __init__(self, num_features, embedding_dim=16, hidden_dim=64):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in num_features
        ])
        total_dim = len(num_features) * embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        emb_list = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        concat = torch.cat(emb_list, dim=-1)
        return self.fc(concat)


class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))


class PseudoTreatmentEffect(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))


class DESCN(nn.Module):
    def __init__(self, num_features, embedding_dim=16, hidden_dim=64,
                 tower_dim=32, alpha=1.0, beta1=1.0, beta0=1.0,
                 gamma1=0.5, gamma0=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta0 = beta0
        self.gamma1 = gamma1
        self.gamma0 = gamma0

        self.backbone = RepresentationLayer(num_features, embedding_dim, hidden_dim)

        self.propensity_tower = Tower(hidden_dim, tower_dim)
        self.treatment_tower = Tower(hidden_dim, tower_dim)
        self.control_tower = Tower(hidden_dim, tower_dim)

        self.pseudo_treatment = PseudoTreatmentEffect(hidden_dim, tower_dim)

    def forward(self, x):
        rep = self.backbone(x)

        pi = self.propensity_tower(rep)
        mu_1 = self.treatment_tower(rep)
        mu_0 = self.control_tower(rep)
        tau_prime = self.pseudo_treatment(rep)

        estr = mu_1 * pi
        escr = mu_0 * (1 - pi)

        eps = 1e-7
        mu_1_cross = torch.sigmoid(
            torch.log(mu_0 + eps) + torch.log(tau_prime + eps)
            - torch.log(1 - tau_prime + eps)
        )
        mu_0_cross = torch.sigmoid(
            torch.log(mu_1 + eps) - torch.log(tau_prime + eps)
            + torch.log(1 - tau_prime + eps)
        )

        return {
            'pi': pi, 'mu_1': mu_1, 'mu_0': mu_0,
            'estr': estr, 'escr': escr,
            'tau_prime': tau_prime,
            'mu_1_cross': mu_1_cross, 'mu_0_cross': mu_0_cross
        }

    def compute_loss(self, outputs, treatment, label):
        pi = outputs['pi']
        estr = outputs['estr']
        escr = outputs['escr']
        mu_1_cross = outputs['mu_1_cross']
        mu_0_cross = outputs['mu_0_cross']

        t = treatment.float()
        y = label.float()

        l_pi = F.binary_cross_entropy(pi.clamp(1e-7, 1-1e-7), t)

        t_mask = t
        c_mask = 1 - t

        l_estr = F.binary_cross_entropy(
            estr.clamp(1e-7, 1-1e-7), y * t_mask,
            reduction='none'
        ).mean()

        l_escr = F.binary_cross_entropy(
            escr.clamp(1e-7, 1-1e-7), y * c_mask,
            reduction='none'
        ).mean()

        l_cross_tr = (t_mask * F.binary_cross_entropy(
            mu_1_cross.clamp(1e-7, 1-1e-7), y, reduction='none'
        )).sum() / (t_mask.sum() + 1e-8)

        l_cross_cr = (c_mask * F.binary_cross_entropy(
            mu_0_cross.clamp(1e-7, 1-1e-7), y, reduction='none'
        )).sum() / (c_mask.sum() + 1e-8)

        total_loss = (self.alpha * l_pi
                      + self.beta1 * l_estr
                      + self.beta0 * l_escr
                      + self.gamma1 * l_cross_tr
                      + self.gamma0 * l_cross_cr)

        return total_loss, {
            'l_pi': l_pi.item(), 'l_estr': l_estr.item(),
            'l_escr': l_escr.item(),
            'l_cross_tr': l_cross_tr.item(),
            'l_cross_cr': l_cross_cr.item()
        }

    def predict_uplift(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            uplift = outputs['mu_1'] - outputs['mu_0']
        return uplift


def train_descn():
    num_features = [100, 200, 50, 30]
    batch_size = 256

    model = DESCN(num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        x = torch.zeros(batch_size, len(num_features), dtype=torch.long)
        for i, size in enumerate(num_features):
            x[:, i] = torch.randint(0, size, (batch_size,))

        treatment = (torch.rand(batch_size) > 0.5).long()
        label = (torch.rand(batch_size) > 0.8).long()

        outputs = model(x)
        loss, loss_dict = model.compute_loss(outputs, treatment, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            uplift = model.predict_uplift(x)
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                  f"Uplift_mean={uplift.mean().item():.4f}, "
                  f"Uplift_std={uplift.std().item():.4f}")


if __name__ == "__main__":
    train_descn()
```

## 8. 应用场景

| 场景 | 处理变量 | 响应变量 | Uplift 含义 |
|------|---------|---------|------------|
| 优惠券发放 | 是否发券 | 是否购买 | 发券带来的增量购买率 |
| 广告投放 | 是否展示广告 | 是否转化 | 广告带来的增量转化 |
| 推送通知 | 是否推送 | 是否活跃 | 推送带来的增量活跃 |
| 价格优惠 | 是否降价 | 是否下单 | 降价带来的增量下单率 |
| 营销活动 | 是否参与活动 | 是否留存 | 活动带来的增量留存 |

## 9. 优缺点分析

### 优点

- **全空间训练**：ESN 模块确保处理组和对照组都在全样本空间训练
- **交叉学习**：X-Network 让两组信息互相补充，缓解反事实问题
- **端到端训练**：所有模块联合优化，无需多阶段训练
- **灵活的损失权重**：$\alpha, \beta, \gamma$ 可根据业务调整

### 局限

- 超参数较多（五项损失权重），调参成本高
- 需要随机化实验数据（RCT），观测数据可能存在混淆偏差
- X-Network 的伪处理效应 $\tau'$ 的收敛需要足够数据
- 架构较复杂，线上推理延迟增加

## 10. 与相关方法对比

| 模型 | 方法 | SSB | 反事实 | 复杂度 |
|------|------|-----|--------|--------|
| Two-Model | 分别训练处理/对照模型 | ✗ | 部分 | 低 |
| S-Learner | 单模型 + 处理特征 | ✗ | 部分 | 低 |
| T-Learner | 双塔 + 处理特征 | ✗ | 部分 | 中 |
| **DESCN** | ESN + X-Network | ✓ | ✓ | 高 |
| ESMM | CTR + CTCVR | ✓ | 部分 | 中 |
| EUEN | 全空间 Uplift | ✓ | 部分 | 中 |
| DragonNet | 倾向评分 + 结果建模 | 部分 | ✓ | 高 |

### DESCN vs ESMM

ESMM 解决的是 CVR 预估中的 SSB 问题，DESCN 在此基础上进一步解决 Uplift 建模中的反事实问题。DESCN 的 ESN 模块借鉴了 ESMM 的全空间思想。

### DESCN vs DragonNet

DragonNet 用倾向评分正则化来消除混淆偏差，主要面向观测数据。DESCN 面向随机化实验数据，用交叉网络增强反事实预测。

## 11. 常见问题与易错点

### Q1：Uplift 值能为负吗？

可以。负值表示处理对该用户有负面影响（如频繁推送反而降低活跃度），这些用户不应被处理。

### Q2：ESN 中 ESTR 和 ESCR 的标签怎么构造？

- ESTR 标签 = $Y \cdot \mathbb{1}[T=1]$（处理组中转化的样本标签为 1，其余为 0）
- ESCR 标签 = $Y \cdot \mathbb{1}[T=0]$（对照组中转化的样本标签为 1，其余为 0）

### Q3：损失权重怎么调？

建议从 $\alpha=1, \beta_1=\beta_0=1, \gamma_1=\gamma_0=0.5$ 开始，根据验证集 AUUC 调整。倾向评分的准确度对整体效果影响大，$\alpha$ 通常不低于 1。

### Q4：非随机化数据能用 DESCN 吗？

可以但效果下降。非随机化数据存在混淆偏差，倾向评分 $\pi(X)$ 的估计可能不准确。建议结合 IPW（逆倾向加权）使用。

## 12. 可视化说明

```
Uplift 分布示例：

    频率
     |
     |    处理组响应 μ₁
     |   /\
     |  /  \        对照组响应 μ₀
     | /    \      /\
     |/      \    /  \
     +--------+--+----+---→ Uplift
    负效应  中立  正效应

营销策略：
- Uplift > 阈值：投放（说服型用户）
- Uplift ≈ 0：不投放（无论是否处理都不在意）
- Uplift < 0：不投放（不要打扰型用户）
```

## 13. 学习总结

| 要点 | 内容 |
|------|------|
| 核心目标 | 预估个体处理效应（Uplift） |
| ESN | 全空间建模，消除 SSB |
| X-Network | 交叉学习，缓解反事实问题 |
| 输出 | Uplift = μ₁(X) - μ₀(X) |
| 关键创新 | logit 空间的交叉计算 |

## 14. 练习题与思考题

1. **推导题**：推导 X-Network 中 $\mu_1'$ 的梯度，说明梯度如何从处理组传递到对照组。
2. **思考题**：如果没有 X-Network，只用 ESN，模型会有什么问题？
3. **实现题**：在代码中加入 AUUC（Area Under Uplift Curve）评估指标。
4. **分析题**：为什么在 logit 空间做加减比在概率空间做加减更合理？

## 15. 学习路径建议

1. **前置知识**：因果推断基础、倾向评分、Uplift 建模概念
2. **原始论文**：*DESCN* (KDD 2022)
3. **进阶阅读**：DragonNet、EUEN、CES
4. **延伸主题**：多处理 Uplift、连续处理效应、观测数据因果推断
