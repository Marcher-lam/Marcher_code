# DFM（Delayed Feedback Model）延迟反馈模型

## 1. 算法基础认知

DFM（Delayed Feedback Model）是解决推荐/广告系统中**延迟反馈**问题的经典方法。在真实业务场景中，用户的转化行为（购买、注册等）可能在点击后数小时甚至数天才发生，但模型训练时只能观察到截至当前时刻的反馈数据，导致大量**假负样本**问题。

论文：*Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR prediction*

## 2. 问题背景

### 2.1 延迟反馈问题

假设用户在 $t_0$ 时刻看到广告并点击：
- 即时转化：几分钟后购买 → 标签正确
- 延迟转化：2 天后购买 → 训练时标签为"未转化"（假负样本）

如果直接用当前标签训练 CVR 模型，模型会将大量真实转化样本当作负样本学习，**系统性低估真实转化率**。

### 2.2 假负样本的影响

- 模型偏向预测低转化率
- 高延迟转化品类（如大宗商品）被低估
- 模型对转化时间的分布敏感

### 2.3 为什么不能等足够久再训练

- 数据分布随时间变化（概念漂移）
- 新广告/商品需要快速获得预估
- 实时性要求高的场景不能等待

## 3. 核心思想

DFM 将转化建模为两个随机过程的联合：
1. **是否转化**：$p(C=1|X)$ — 用户最终是否会转化
2. **延迟时间**：$p(D|X, C=1)$ — 转化发生的时间

通过引入延迟时间建模，区分"真的不转化"和"还没转化但将来会转化"。

## 4. 概率模型

### 4.1 转化概率模型

$$p(C=1|X) = \sigma(w_c^T X)$$

其中 $\sigma$ 是 sigmoid 函数。

### 4.2 延迟时间模型

假设延迟时间 $D$ 服从指数分布：

$$p(D=d|X, C=1) = \lambda(X) e^{-\lambda(X) d}$$

其中 $\lambda(X) = \exp(w_d^T X) > 0$ 是与特征相关的延迟速率参数。

指数分布的期望延迟为 $E[D] = 1/\lambda(X)$。

### 4.3 观测模型

设观察窗口为 $E$（从点击到当前的时间），定义观测标签 $Y$：

- $Y=1$：在窗口 $E$ 内已观察到转化，转化延迟为 $D$
- $Y=0$：在窗口 $E$ 内未观察到转化

$Y=0$ 有两种可能：
1. 用户永远不会转化（$C=0$）
2. 用户会转化，但延迟 $D > E$（将来才会转化）

## 5. 数学推导详解

### 5.1 似然函数构建

对于已转化样本（$Y=1$）：

$$p(Y=1, D=d|X) = p(C=1|X) \cdot p(D=d|X, C=1) = \sigma(w_c^T X) \cdot \lambda(X) e^{-\lambda(X) d}$$

对于未转化样本（$Y=0$）：

$$p(Y=0|X) = p(C=0|X) + p(C=1|X) \cdot p(D > E|X, C=1)$$

其中 $p(D > E|X, C=1) = e^{-\lambda(X) E}$，所以：

$$p(Y=0|X) = 1 - \sigma(w_c^T X) + \sigma(w_c^T X) \cdot e^{-\lambda(X) E}$$

### 5.2 对数似然

$$\log L = \sum_{Y_i=1}\left[\log\sigma(w_c^T X_i) + \log\lambda(X_i) - \lambda(X_i) D_i\right]$$
$$+ \sum_{Y_i=0}\log\left[1 - \sigma(w_c^T X_i) + \sigma(w_c^T X_i) e^{-\lambda(X_i) E_i}\right]$$

### 5.3 负对数似然损失

$$L = -\sum_{Y_i=1}\left[\log\sigma(w_c^T X_i) + \log\lambda(X_i) - \lambda(X_i) D_i\right]$$
$$- \sum_{Y_i=0}\log\left[1 - \sigma(w_c^T X_i) + \sigma(w_c^T X_i) e^{-\lambda(X_i) E_i}\right]$$

### 5.4 梯度分析

对 $w_c$（转化参数）的梯度（$Y=0$ 部分）：

$$\frac{\partial L}{\partial w_c}\bigg|_{Y=0} = -\frac{\sigma(w_c^T X)(e^{-\lambda E} - 1)}{1 - \sigma(w_c^T X) + \sigma(w_c^T X) e^{-\lambda E}} \cdot X$$

这个梯度方向使得：观察时间 $E$ 越短，模型越倾向于认为未转化样本可能只是延迟了，而非真的不转化。

## 6. EM 算法详解

由于 $Y=0$ 样本的真实转化状态 $C$ 是隐变量，DFM 使用 EM 算法迭代求解。

### 6.1 E 步：计算隐变量后验

对于 $Y=0$ 的样本，计算其"会转化但延迟超过观察窗口"的后验概率：

$$p(C=1|Y=0, X) = \frac{p(C=1|X) \cdot p(D > E|X, C=1)}{p(Y=0|X)}$$

$$= \frac{\sigma(w_c^T X) \cdot e^{-\lambda(X) E}}{1 - \sigma(w_c^T X) + \sigma(w_c^T X) \cdot e^{-\lambda(X) E}}$$

直觉：如果延迟速率 $\lambda$ 小（延迟长），或观察窗口 $E$ 短，则后验概率大，说明更可能是假负样本。

### 6.2 M 步：最大化期望完整对数似然

引入后验概率 $q_i = p(C_i=1|Y_i=0, X_i)$，定义期望完整对数似然：

$$Q = \sum_{Y_i=1}\left[\log\sigma(w_c^T X_i) + \log\lambda(X_i) - \lambda(X_i) D_i\right]$$
$$+ \sum_{Y_i=0}\left[q_i(\log\sigma(w_c^T X_i) + \log\lambda(X_i) - \lambda(X_i) E_i) + (1-q_i)\log(1-\sigma(w_c^T X_i))\right]$$

M 步分别优化：
- **$w_c$（转化参数）**：加权交叉熵损失，$q_i$ 作为软标签
- **$w_d$（延迟参数）**：只对转化样本和假负样本拟合延迟分布

### 6.3 EM 迭代过程

1. 初始化 $w_c, w_d$
2. E 步：用当前参数计算所有 $Y=0$ 样本的 $q_i$
3. M 步：用 $q_i$ 作为软标签优化 $w_c$ 和 $w_d$
4. 重复 2-3 直到收敛

## 7. PyTorch 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DFM(nn.Module):
    def __init__(self, num_features, embedding_dim=16, hidden_dim=64):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in num_features
        ])

        total_dim = len(num_features) * embedding_dim

        self.cvr_net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.delay_net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        emb_list = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        concat_emb = torch.cat(emb_list, dim=-1)

        cvr_logit = self.cvr_net(concat_emb).squeeze(-1)
        p_cvr = torch.sigmoid(cvr_logit)

        delay_logit = self.delay_net(concat_emb).squeeze(-1)
        lam = torch.exp(delay_logit) + 1e-8

        return p_cvr, lam

    def compute_loss(self, p_cvr, lam, label, delay, elapsed):
        pos_mask = (label == 1).float()
        neg_mask = (label == 0).float()

        pos_loss = pos_mask * (
            torch.log(p_cvr + 1e-8)
            + torch.log(lam + 1e-8)
            - lam * delay
        )

        no_click_prob = 1.0 - p_cvr
        delayed_click_prob = p_cvr * torch.exp(-lam * elapsed)
        neg_prob = no_click_prob + delayed_click_prob

        neg_loss = neg_mask * torch.log(neg_prob + 1e-8)

        loss = -(pos_loss.sum() + neg_loss.sum()) / label.size(0)
        return loss

    def em_e_step(self, p_cvr, lam, elapsed):
        with torch.no_grad():
            delayed_prob = p_cvr * torch.exp(-lam * elapsed)
            total_prob = (1.0 - p_cvr) + delayed_prob
            q = delayed_prob / (total_prob + 1e-8)
        return q


class DFMTrainer:
    def __init__(self, model, lr=1e-3, em_steps=3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.em_steps = em_steps

    def train_step(self, x, label, delay, elapsed):
        p_cvr, lam = self.model(x)

        loss = self.model.compute_loss(p_cvr, lam, label, delay, elapsed)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), p_cvr.detach(), lam.detach()

    def em_train_step(self, x, label, delay, elapsed):
        for em_iter in range(self.em_steps):
            p_cvr, lam = self.model(x)

            q = self.model.em_e_step(p_cvr, lam, elapsed)

            pos_mask = (label == 1).float()
            neg_mask = (label == 0).float()

            pos_loss = pos_mask * (
                torch.log(p_cvr + 1e-8)
                + torch.log(lam + 1e-8)
                - lam * delay
            )

            neg_loss_cvr = neg_mask * (
                q * torch.log(p_cvr + 1e-8)
                + (1 - q) * torch.log(1 - p_cvr + 1e-8)
            )

            neg_loss_delay = neg_mask * q * (
                torch.log(lam + 1e-8) - lam * elapsed
            )

            loss = -(pos_loss.sum() + neg_loss_cvr.sum() + neg_loss_delay.sum())
            loss = loss / x.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()


def demo():
    num_features = [100, 200, 50]
    batch_size = 128

    model = DFM(num_features)
    trainer = DFMTrainer(model)

    for epoch in range(20):
        x = torch.zeros(batch_size, len(num_features), dtype=torch.long)
        for i, size in enumerate(num_features):
            x[:, i] = torch.randint(0, size, (batch_size,))

        label = (torch.rand(batch_size) > 0.9).long()
        delay = torch.rand(batch_size) * 10
        elapsed = torch.rand(batch_size) * 5 + 0.1

        loss = trainer.em_train_step(x, label, delay, elapsed)

        if epoch % 5 == 0:
            p_cvr, lam = model(x)
            print(f"Epoch {epoch}: Loss={loss:.4f}, "
                  f"CVR_mean={p_cvr.mean().item():.4f}, "
                  f"Delay_mean={lam.mean().item():.4f}")


if __name__ == "__main__":
    demo()
```

## 8. 应用场景

| 场景 | 延迟特点 | 适用性 |
|------|---------|--------|
| 广告转化追踪 | 点击到购买可能延迟数天 | 高 |
| APP 安装归因 | 安装可能在点击后数小时 | 高 |
| 电商复购预测 | 复购周期可能数周 | 中 |
 | 信贷审批 | 违约可能在数月后 | 中 |
| 内容消费 | 阅读完成通常即时 | 低 |

## 9. 优缺点分析

### 优点

- 显式建模延迟时间，区分"不转化"和"未转化"
- 利用观察时间信息，减少假负样本的影响
- 理论框架清晰，概率模型可解释

### 局限

- 指数分布假设可能不符合实际延迟分布（实际常为长尾分布）
- EM 算法需要多次迭代，训练开销较大
- 需要记录每个样本的观察窗口 $E$，数据存储成本增加
- 未利用实时特征更新的能力

## 10. 与相关方法对比

| 模型 | 延迟分布 | 训练方式 | 特点 |
|------|---------|---------|------|
| **DFM** | 指数分布 | EM 算法 | 经典方法，理论清晰 |
| FSIVR | 混合分布 | 重要性采样 | 更灵活的分布假设 |
| DFN | - | 假负样本校正 | 工程实现简单 |
| DEFER | - | 延迟正样本重赋权 | 在线学习友好 |
| NoDeF | 非参数 | 神经网络 | 不假设分布形式 |

### DFM vs DEFER

DEFER（Delayed Feedback with Regularization）不建模延迟分布，而是通过重要性加权校正假负样本，实现更简单但理论保证弱于 DFM。

### DFM vs NoDeF

NoDeF 用神经网络直接学习延迟分布，不假设指数分布，更灵活但需要更多数据。

## 11. 常见问题与易错点

### Q1：观察窗口 $E$ 怎么确定？

$E$ 是每个样本从曝光/点击到当前训练时刻的时间差，不是固定值。每个样本的 $E$ 不同。

### Q2：指数分布假设合理吗？

实际广告转化延迟通常是长尾分布（对数正态或 Gamma 更合适），但指数分布计算简便且在实践中效果可接受。后续工作（如 FSIVR）使用了更灵活的分布。

### Q3：EM 算法会陷入局部最优吗？

会。初始化对结果影响较大。实践中可用直接优化的方式（不用 EM）作为替代。

### Q4：延迟信息和转化信息用同一个网络吗？

不建议。DFM 中 $w_c$ 和 $w_d$ 是两组独立参数，但可以共享底层 Embedding。延迟速率和转化概率是不同语义的信息，分开建模更合理。

## 12. 可视化说明

延迟反馈的典型数据分布：

```
时间轴：
|--E--|
曝光→点击→........→转化
      ↑               ↑
      当前训练时刻    真实转化时刻

Y=0样本的真实构成：
├── 真不转化 (C=0): ████████████  ~90%
└── 延迟转化 (C=1, D>E): ██  ~10%
```

随着观察窗口 $E$ 增大，假负样本比例降低，但数据时效性也降低。

## 13. 学习总结

| 要点 | 内容 |
|------|------|
| 核心问题 | 转化延迟导致假负样本 |
| 建模方法 | 转化概率 + 延迟时间联合建模 |
| 分布假设 | 指数分布 |
| 求解方法 | EM 算法或直接梯度优化 |
| 关键创新 | 区分"不转化"和"还没转化" |

## 14. 练习题与思考题

1. **推导题**：将指数分布替换为 Gamma 分布，重新推导损失函数。
2. **思考题**：如果在在线学习场景中使用 DFM，EM 算法如何改造？
3. **实现题**：在上述代码中加入共享 Embedding，对比共享前后 CVR 预估效果。
4. **分析题**：为什么延迟速率 $\lambda$ 用 $\exp(\cdot)$ 而不是直接用线性输出？

## 15. 学习路径建议

1. **前置知识**：EM 算法、指数分布、CVR 预估
2. **原始论文**：*Observation-Dependent Delayed Feedback Models* (DFM 原论文)
3. **进阶阅读**：FSIVR、DEFER、NoDeF、JADF（联合延迟反馈模型）
4. **延伸主题**：ESMM（全空间建模）、在线学习与概念漂移
