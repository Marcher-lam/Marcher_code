# ESMM (Entire Space Multi-Task Model) 学习文档

## 1. 算法基础认知

ESMM 是阿里巴巴在 2018 年提出的 CVR 预估模型，发表于 SIGIR 2018。它解决了转化率（CVR）预估中的两大难题：**样本选择偏差**和**数据稀疏**。

在广告系统中，用户行为路径为：曝光 → 点击 → 转化。传统 CVR 模型仅在点击样本上训练，但线上预测时面对的是全部曝光样本，训练和预测的数据分布不一致，这就是样本选择偏差（SSB）。同时，转化行为极其稀疏（点击到转化率通常 <1%），导致模型训练困难。

ESMM 的核心思路：不直接预估 CVR，而是建模 CTR 和 CTCVR 两个任务，通过 $pCTCVR = pCTR \times pCVR$ 的关系推导出 CVR，在全部曝光样本上训练。

## 2. 核心原理

ESMM 的架构包含两个共享 Embedding 的塔：

- **CTR 塔**：预估点击率 $P(click=1|impression)$，在全部曝光样本上有标签
- **CVR 塔**：预估转化率 $P(conversion=1|click, impression)$，通过 CTCVR 间接训练

关键公式：

$$pCTCVR = pCTR \times pCVR$$

其中 $pCTCVR = P(click=1, conversion=1|impression)$，即曝光后既点击又转化的概率。

训练时使用**全部曝光样本**：点击样本的 click 标签为 1，未点击样本的 click 标签为 0；只有点击且转化的样本 CTCVR 标签为 1。这样消除了样本选择偏差。

## 3. 数学公式与推导

**CTR 塔输出**：

$$pCTR = \sigma(f_{CTR}(x))$$

**CVR 塔输出**：

$$pCVR = \sigma(f_{CVR}(x))$$

**CTCVR 推导**：

$$pCTCVR = P(click=1, conversion=1|x) = P(click=1|x) \cdot P(conversion=1|click=1, x) = pCTR \times pCVR$$

**联合损失函数**：

$$\mathcal{L} = \mathcal{L}_{CTR} + \lambda \cdot \mathcal{L}_{CTCVR}$$

$$\mathcal{L}_{CTR} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_{click}^{(i)} \log pCTR^{(i)} + (1-y_{click}^{(i)})\log(1-pCTR^{(i)})\right]$$

$$\mathcal{L}_{CTCVR} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_{ctcvr}^{(i)} \log pCTCVR^{(i)} + (1-y_{ctcvr}^{(i)})\log(1-pCTCVR^{(i)})\right]$$

注意：CVR 塔没有独立的损失函数，它通过 $pCTCVR = pCTR \times pCVR$ 间接获得梯度。

## 4. 训练过程讲解

1. **数据准备**：使用全部曝光数据，每条样本包含特征 $x$、click 标签、conversion 标签
2. **前向传播**：CTR 塔和 CVR 塔分别输出 $pCTR$ 和 $pCVR$，相乘得到 $pCTCVR$
3. **损失计算**：$\mathcal{L}_{CTR}$ 用 click 标签监督，$\mathcal{L}_{CTCVR}$ 用 $click \times conversion$ 标签监督
4. **反向传播**：梯度通过 $pCTCVR = pCTR \times pCVR$ 分别回传到两个塔和共享 Embedding
5. **推理**：直接取 CVR 塔输出作为转化率预估

## 5. 应用场景

- **广告 CVR 预估**：电商广告的购买转化率预测
- **oCPX 竞价**：按转化出价，需要准确的 CVR 预估
- **后链路优化**：点击后的深度转化（加购、收藏、购买）预估
- **内容推荐深度优化**：预测用户阅读完成率、互动率等

## 6. 优缺点分析

**优点**：
- 解决了 CVR 预估的样本选择偏差问题
- 全空间训练，充分利用未点击样本的信息
- 共享 Embedding 缓解了转化数据的稀疏性
- 联合训练 CTR 和 CVR 互相增强

**缺点**：
- CVR 塔通过间接梯度训练，信号较弱
- 乘积关系假设 CTR 和 CVR 条件独立，实际中可能不成立
- 没有对 CVR 输出做范围约束，理论上可能出现 $pCVR > pCTR$ 的不合理情况

## 7. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ESMM(nn.Module):
    def __init__(self, num_features, embedding_dim=16, hidden_dims=[64, 32]):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_dim)
        tower_input = embedding_dim * 10

        def build_tower():
            layers = []
            dim = tower_input
            for h in hidden_dims:
                layers.extend([nn.Linear(dim, h), nn.ReLU()])
                dim = h
            layers.append(nn.Linear(dim, 1))
            return nn.Sequential(*layers)

        self.ctr_tower = build_tower()
        self.cvr_tower = build_tower()

    def forward(self, x):
        emb = self.embedding(x)
        emb_flat = emb.view(emb.size(0), -1)
        p_ctr = torch.sigmoid(self.ctr_tower(emb_flat))
        p_cvr = torch.sigmoid(self.cvr_tower(emb_flat))
        p_ctcvr = p_ctr * p_cvr
        return p_ctr, p_cvr, p_ctcvr


num_features = 1000
num_fields = 10
model = ESMM(num_features)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    x = torch.randint(0, num_features, (256, num_fields))
    click_label = torch.randint(0, 2, (256, 1)).float()
    conversion_label = torch.randint(0, 2, (256, 1)).float()
    ctcvr_label = click_label * conversion_label

    p_ctr, p_cvr, p_ctcvr = model(x)
    loss_ctr = nn.functional.binary_cross_entropy(p_ctr, click_label)
    loss_ctcvr = nn.functional.binary_cross_entropy(p_ctcvr, ctcvr_label)
    loss = loss_ctr + loss_ctcvr
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, CTR Loss: {loss_ctr.item():.4f}")
```

## 8. 手工代码实现

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0, x)


def esmm_forward(x, E, W_ctr, b_ctr, W_cvr, b_cvr):
    embs = E[x]
    flat = embs.flatten()
    ctr_logit = relu(W_ctr[0] @ flat + b_ctr[0])
    ctr_logit = W_ctr[1] @ ctr_logit + b_ctr[1]
    p_ctr = sigmoid(ctr_logit.item())

    cvr_logit = relu(W_cvr[0] @ flat + b_cvr[0])
    cvr_logit = W_cvr[1] @ cvr_logit + b_cvr[1]
    p_cvr = sigmoid(cvr_logit.item())

    p_ctcvr = p_ctr * p_cvr
    return p_ctr, p_cvr, p_ctcvr


def compute_loss(p_ctr, p_cvr, click, conversion):
    ctcvr = click * conversion
    eps = 1e-7
    loss_ctr = -(click * np.log(p_ctr + eps) + (1 - click) * np.log(1 - p_ctr + eps))
    p_ctcvr = p_ctr * p_cvr
    loss_ctcvr = -(ctcvr * np.log(p_ctcvr + eps) + (1 - ctcvr) * np.log(1 - p_ctcvr + eps))
    return loss_ctr + loss_ctcvr


np.random.seed(42)
d = 4
E = np.random.randn(100, d) * 0.1
W_ctr = [np.random.randn(16, d * 5) * 0.1, np.random.randn(1, 16) * 0.1]
b_ctr = [np.zeros(16), np.zeros(1)]
W_cvr = [np.random.randn(16, d * 5) * 0.1, np.random.randn(1, 16) * 0.1]
b_cvr = [np.zeros(16), np.zeros(1)]

x = np.random.randint(0, 100, 5)
p_ctr, p_cvr, p_ctcvr = esmm_forward(x, E, W_ctr, b_ctr, W_cvr, b_cvr)
loss = compute_loss(p_ctr, p_cvr, click=1, conversion=1)
print(f"pCTR: {p_ctr:.4f}, pCVR: {p_cvr:.4f}, pCTCVR: {p_ctcvr:.4f}, Loss: {loss:.4f}")
```

## 9. 可视化与结果理解

- **训练数据分布**：展示曝光-点击-转化的漏斗结构，转化率远低于点击率
- **CVR 间接梯度流**：可视化 $pCTCVR = pCTR \times pCVR$ 的梯度回传路径
- **CTR 与 CVR 预测分布**：对比两个塔的输出分布，CVR 应远低于 CTR

效果对比（淘宝公开数据集）：

| 模型 | CVR AUC | CTCVR AUC |
|------|---------|-----------|
| 仅 CVR 塔（点击样本训练） | 0.6821 | 0.6534 |
| AMAN（过采样） | 0.6912 | 0.6621 |
| ESMM | **0.7034** | **0.6798** |

## 10. 模型评估

- **CTCVR AUC**：衡量点击+转化联合预估的排序能力
- **CVR AUC**：仅在点击样本上评估 CVR 预估质量
- **CTCVR 校准**：预测的 CTCVR 值与实际转化率的比值应接近 1
- **线上指标**：CPA（单次转化成本）、GMV（交易额）提升

## 11. 常见问题与易错点

- **CVR 梯度消失**：当 $pCTR$ 很小时，$pCTCVR = pCTR \times pCVR$ 的梯度被严重压缩，导致 CVR 塔训练不充分
- **延迟反馈问题**：用户可能在点击后很久才转化，训练时标签可能不完整（后续 ESM2 引入订单行为建模）
- **不能单独使用 CVR 损失**：CVR 塔没有独立损失，必须通过 CTCVR 间接训练
- **$pCVR > pCTR$ 的问题**：理论上 $pCTCVR \leq pCTR$，但 ESMM 不保证 $pCVR \leq 1$ 且 $pCVR \times pCTR \leq pCTR$

## 12. 学习总结

ESMM 是 CVR 预估领域的里程碑工作。它通过 $pCTCVR = pCTR \times pCVR$ 的概率分解，巧妙地将 CVR 预估转化为全空间训练的多任务学习问题，同时解决了样本选择偏差和数据稀疏两大难题。后续的 ESM2、ESCM² 等模型在此基础上进一步改进。

## 13. 练习题与思考题（含答案）

**Q1：什么是 CVR 预估中的样本选择偏差？**

A1：传统 CVR 模型仅在点击样本上训练（因为只有点击后才可能转化），但线上预测时面对的是全部曝光样本。点击样本是曝光样本的有偏子集（点击用户本身倾向性更强），导致训练分布与推理分布不一致，模型预估偏高。

**Q2：为什么 ESMM 在全空间训练能缓解数据稀疏？**

A2：ESMM 的 Embedding 层由 CTR 和 CVR 两个塔共享。CTR 任务有大量点击/未点击样本提供丰富的梯度信号，这些信号通过共享 Embedding 传递给 CVR 塔，相当于用 CTR 的丰富数据辅助了 CVR 的稀疏训练。

**Q3：ESMM 的损失函数为什么没有 CVR 的独立损失？**

A3：因为 CVR 无法直接在曝光样本上获得标签——未点击的样本无法判断是否会转化。CVR 只能通过 $pCTCVR = pCTR \times pCVR$ 的关系间接获得梯度，这是 ESMM 巧妙设计的核心。

## 14. 学习路径建议

```
多任务学习基础（共享 Embedding + 多塔）
        ↓
ESMM（CVR 全空间训练）  ← 你在这里
        ↓
ESM2（引入更多中间行为：加购/收藏）
        ↓
ESCM²（解决 ESMM 的概率偏移问题）
        ↓
MMoE / PLE（通用多任务学习架构）
```
