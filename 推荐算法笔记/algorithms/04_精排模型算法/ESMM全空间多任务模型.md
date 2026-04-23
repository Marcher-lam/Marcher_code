# ESMM（Entire Space Multi-Task Model）全空间多任务模型

## 1. 算法基础认知

ESMM（Entire Space Multi-Task Model）是阿里巴巴在 2018 年提出的转化率（CVR）预估模型，旨在解决 CVR 建模中的两大核心难题：**样本选择偏差（SSB）** 和 **数据稀疏（DS）**。

论文：*Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate*

### 核心概念

- **CTR（Click-Through Rate）**：点击率，即 $p(\text{click}=1 | \text{impression})$
- **CVR（Conversion Rate）**：转化率，即 $p(\text{conversion}=1 | \text{click})$
- **CTCVR（Click-Through Conversion Rate）**：点击转化率，即 $p(\text{conversion}=1 | \text{impression})$

三者之间的关系：

$$pCTCVR = pCTR \times pCVR$$

## 2. 问题背景

### 2.1 样本选择偏差（SSB）

传统 CVR 模型只在**已点击**的样本上训练，但推理时需要在**全部曝光**样本上预测。训练和推理的数据分布不一致：

- 训练集：$\{x | \text{click}=1\}$（仅点击样本）
- 推理集：$\{x | \text{任意曝光}\}$（全部曝光样本）

这导致模型在未点击样本上的预测不可靠。

### 2.2 数据稀疏（DS）

转化是稀有事件。以电商为例，曝光→点击率约 5%，点击→转化率约 2%，因此曝光→转化率仅约 0.1%。CVR 训练样本远少于 CTR 训练样本，模型容易过拟合。

## 3. 核心思想

ESMM 的核心洞察：**不直接建模 CVR，而是同时建模 CTR 和 CTCVR，通过二者之比间接得到 CVR。**

$$\hat{pCVR} = \frac{\hat{pCTCVR}}{\hat{pCTR}} = \frac{\hat{pCTR} \times \hat{pCVR}}{\hat{pCTR}}$$

### 关键设计

1. **全空间训练**：CTCVR 和 CTR 都在全部曝光样本上训练，消除了 SSB
2. **共享 Embedding**：CTR 和 CVR 两个塔共享底层 Embedding 参数，CVR 塔借助 CTR 塔的大量数据缓解 DS
3. **乘法结构**：利用概率公理 $p(A \cap B) = p(A) \cdot p(B|A)$ 建模

## 4. 数学公式与推导

### 4.1 概率分解

对于一次曝光 $x$，用户行为链路为：曝光→点击→转化。定义：

$$p(y=1, z=1 | x) = p(y=1 | x) \cdot p(z=1 | y=1, x)$$

其中：
- $y=1$ 表示点击
- $z=1$ 表示转化
- $p(y=1|x) = pCTR$
- $p(z=1|y=1,x) = pCVR$
- $p(y=1,z=1|x) = pCTCVR$

### 4.2 损失函数

$$L = L_{CTR} + L_{CTCVR}$$

展开为：

$$L = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log pCTR(x_i) + (1-y_i)\log(1-pCTR(x_i))\right]$$
$$- \frac{1}{N}\sum_{i=1}^{N}\left[c_i \log(pCTR(x_i) \cdot pCVR(x_i)) + (1-c_i)\log(1 - pCTR(x_i) \cdot pCVR(x_i))\right]$$

其中 $y_i$ 是点击标签，$c_i$ 是转化标签。注意 $c_i=1$ 必然有 $y_i=1$。

### 4.3 为什么不用 CVR 的直接损失

CVR 标签只在点击样本上有意义。如果在全空间直接加 CVR 损失，未点击样本的 CVR 标签是 0（但真实标签未知），会引入噪声。ESMM 通过 CTCVR 的间接建模巧妙绕过了这个问题。

## 5. 架构详解

```
            Input Features
                 |
           [Embedding Layer]  ← 共享参数
              /        \
         [CTR Tower]  [CVR Tower]
              |            |
          pCTR(x)      pCVR(x)
              \            /
            [Element-wise Multiply]
                   |
               pCTCVR(x)
```

### 具体流程

1. 输入特征经过共享 Embedding 层
2. CTR 塔输出 $pCTR = \sigma(f_{CTR}(x))$
3. CVR 塔输出 $pCVR = \sigma(f_{CVR}(x))$
4. CTCVR = pCTR × pCVR（逐元素相乘）
5. 分别计算 CTR 损失和 CTCVR 损失，求和作为总损失

## 6. 训练过程讲解

1. **数据准备**：每个样本包含特征 $x$、点击标签 $y$、转化标签 $c$
2. **前向传播**：
   - 特征经 Embedding 层得到稠密表示
   - CTR 塔输出点击概率
   - CVR 塔输出转化概率
   - 计算 CTCVR = CTR × CVR
3. **损失计算**：
   - CTR 损失：在全部曝光样本上计算的交叉熵
   - CTCVR 损失：在全部曝光样本上计算的交叉熵
4. **反向传播**：总损失 = L_CTR + L_CTCVR，梯度同时更新共享 Embedding 和两个塔的参数
5. **推理**：pCVR = pCTCVR / pCTR

## 7. PyTorch 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMM(nn.Module):
    def __init__(self, num_features, embedding_dim=16, hidden_dims=[128, 64]):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_features[i], embedding_dim)
            for i in range(len(num_features))
        ])

        total_emb_dim = len(num_features) * embedding_dim

        self.ctr_tower = self._build_tower(total_emb_dim, hidden_dims)
        self.cvr_tower = self._build_tower(total_emb_dim, hidden_dims)

        self.ctr_output = nn.Linear(hidden_dims[-1], 1)
        self.cvr_output = nn.Linear(hidden_dims[-1], 1)

    def _build_tower(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        emb_list = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        concat_emb = torch.cat(emb_list, dim=-1)

        ctr_hidden = self.ctr_tower(concat_emb)
        cvr_hidden = self.cvr_tower(concat_emb)

        p_ctr = torch.sigmoid(self.ctr_output(ctr_hidden)).squeeze(-1)
        p_cvr = torch.sigmoid(self.cvr_output(cvr_hidden)).squeeze(-1)

        p_ctcvr = p_ctr * p_cvr

        return p_ctr, p_cvr, p_ctcvr

    def compute_loss(self, p_ctr, p_ctcvr, click_label, conversion_label):
        ctr_loss = F.binary_cross_entropy(p_ctr, click_label.float())
        ctcvr_loss = F.binary_cross_entropy(
            p_ctcvr.clamp(1e-8, 1 - 1e-8),
            conversion_label.float()
        )
        return ctr_loss + ctcvr_loss


def train_esmm():
    num_features = [100, 200, 50, 30]
    batch_size = 256
    num_epochs = 10

    model = ESMM(num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        x = torch.randint(0, 10, (batch_size, len(num_features)))
        for i in range(len(num_features)):
            x[:, i] = torch.randint(0, num_features[i], (batch_size,))

        click_label = (torch.rand(batch_size) > 0.8).long()
        conversion_label = (click_label == 1) & (torch.rand(batch_size) > 0.9)
        conversion_label = conversion_label.long()

        p_ctr, p_cvr, p_ctcvr = model(x)
        loss = model.compute_loss(p_ctr, p_ctcvr, click_label, conversion_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                  f"CTR_mean={p_ctr.mean().item():.4f}, "
                  f"CVR_mean={p_cvr.mean().item():.4f}")


if __name__ == "__main__":
    train_esmm()
```

## 8. 应用场景

| 场景 | 说明 |
|------|------|
| 电商推荐 | 预估商品点击率和下单转化率，优化 GMV |
| 广告投放 | 同时优化点击率和转化率，按 oCPM 出价 |
| 内容推荐 | 预估文章点击率和阅读完成率 |
| APP 推广 | 预估广告点击率和 APP 安装率 |

## 9. 优缺点分析

### 优点

- **消除 SSB**：全空间训练，训练和推理数据分布一致
- **缓解 DS**：共享 Embedding 使 CVR 塔获得更多梯度信号
- **端到端训练**：无需多阶段训练，联合优化
- **部署简单**：推理时只需一次前向传播

### 局限

- **乘法假设**：$pCTCVR = pCTR \times pCVR$ 假设点击和转化条件独立，但实际中转化依赖于点击
- **潜在独立性优先（PIP）**：当 $pCTR \to 0$ 时 $pCVR$ 可能趋于无穷大以补偿，导致数值不稳定
- **无法建模序列行为**：不考虑点击到转化之间的时间延迟
- **CVR 塔梯度信号弱**：CVR 梯度需要经过 CTCVR = CTR × CVR 的乘法路径反传，信号被 CTR 缩放

## 10. 与相关方法对比

| 模型 | SSB | DS | 核心思路 | 局限 |
|------|-----|-----|---------|------|
| 传统 CVR | ✗ | ✗ | 直接在点击样本上训练 | 训练推理分布不一致 |
| **ESMM** | ✓ | ✓ | CTR + CTCVR 联合建模 | 乘法假设，PIP 问题 |
| ESMM2 | ✓ | ✓ | 引入 CTCVR 和 CTR 的中间任务 | 架构更复杂 |
| ESCM² | ✓ | ✓ | 用因果推断消除 PIP | 实现复杂 |
| DBMTL | 部分 | 部分 | 多任务多目标贝叶斯 | 未完全解决 SSB |

### ESMM vs ESMM2

ESMM2 在 ESMM 基础上引入更多辅助任务（如加入收藏、加购等行为），构建更丰富的行为链路。

### ESMM vs ESCM²

ESCM²（Entire Space Counterfactual Multi-Task Model）用反事实因果推断方法，显式消除 PIP 问题，理论上更严谨。

## 11. 常见问题与易错点

### Q1：推理时如何获取 CVR？

$$pCVR = \frac{pCTCVR}{pCTR}$$

注意分母 clip 到 $[eps, 1]$ 避免除零。

### Q2：为什么 CTCVR 损失用全部样本？

CTCVR 的标签（是否转化）在全部曝光样本上都有定义：转化了就是 1，没转化就是 0。不存在标签缺失问题。

### Q3：CVR 塔输出的值域需要限制吗？

需要。pCVR 经过 sigmoid 映射到 $(0, 1)$，但理论上 $pCVR = pCTCVR / pCTR$ 可能超过 1。实践中通常 clip 到 $[0, 1]$。

### Q4：CTR 和 CVR 塔是否需要相同结构？

不需要。可以根据数据量差异调整塔的复杂度，CVR 塔通常可以更轻量。

## 12. 学习总结

| 要点 | 内容 |
|------|------|
| 核心创新 | 全空间多任务建模，间接学习 CVR |
| 解决问题 | SSB（样本选择偏差）和 DS（数据稀疏） |
| 关键公式 | pCTCVR = pCTR × pCVR |
| 训练目标 | L_CTR + L_CTCVR |
| 主要局限 | 乘法独立性假设、PIP 问题 |

## 13. 练习题与思考题

1. **推导题**：证明 ESMM 的 CTCVR 损失等价于在全曝光空间上对 CVR 进行隐式建模。
2. **思考题**：如果转化不依赖点击（如直接购买），ESMM 还适用吗？
3. **实现题**：在上述代码基础上添加第三个塔预估"收藏率"，构建 CTR→收藏→转化的行为链路。
4. **分析题**：为什么共享 Embedding 能缓解数据稀疏？梯度流向是怎样的？

## 14. 学习路径建议

1. **前置知识**：多任务学习、CTR 预估、Embedding + MLP 架构
2. **原始论文**：*Entire Space Multi-Task Model* (SIGIR 2018)
3. **进阶阅读**：ESMM2、ESCM²、DBMTL
4. **延伸主题**：延迟反馈建模（DFM）、Uplift 建模（DESCN）
