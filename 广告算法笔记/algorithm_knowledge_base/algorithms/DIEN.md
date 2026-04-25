# DIEN (Deep Interest Evolution Network) 学习文档

## 1. 算法基础认知

DIEN 是阿里巴巴在 2019 年提出的 CTR 预估模型，发表于 AAAI 2019。它是 DIN 的后续工作，核心改进是从"兴趣激活"升级为"兴趣演化"建模。

DIN 只关注"哪些历史行为与当前候选相关"，忽略了行为之间的时序关系和兴趣变迁。DIEN 引入两层 GRU 结构：兴趣提取层（Interest Extraction Layer）从行为序列中提取兴趣状态，兴趣演化层（Interest Evolution Layer）建模兴趣随时间的演化过程。

## 2. 核心原理

DIEN 的整体架构包含三个核心模块：

- **Behavior Layer**：将用户行为序列映射为 Embedding 序列
- **Interest Extractor Layer**：用 GRU 从行为 Embedding 中提取兴趣隐状态，配合辅助损失（Auxiliary Loss）增强兴趣表示质量
- **Interest Evolving Layer**：用 AUGRU（Attentional Update Gate GRU）建模兴趣的演化过程，注意力权重由目标广告与各时刻兴趣状态的相关性决定

关键创新：AUGRU 将注意力权重融入 GRU 的更新门，使兴趣演化过程能聚焦于与目标相关的兴趣变迁。

## 3. 数学公式与推导

**兴趣提取层（GRU）**：

$$h_t = \text{GRU}(e_t, h_{t-1})$$

**辅助损失**：用下一时刻的行为作为正样本预测兴趣是否匹配：

$$\mathcal{L}_{aux} = -\sum_{t} \left[\log\sigma(h_t \cdot e_{t+1}) + \log(1 - \sigma(h_t \cdot e_j))\right]$$

其中 $e_j$ 是随机采样的负样本行为 Embedding，$\sigma$ 是 Sigmoid 函数。

**注意力权重**：

$$\alpha_t = \frac{\exp(g(h_t, e_a))}{\sum_{j}\exp(g(h_j, e_a))}$$

**AUGRU 更新**：

$$\tilde{u}_t = \alpha_t \odot u_t$$

$$h'_t = (1 - \tilde{u}_t) \odot h'_{t-1} + \tilde{u}_t \odot \tilde{h}_t$$

其中 $u_t$ 是 GRU 的更新门，$\tilde{h}_t$ 是候选隐状态，$\alpha_t$ 作为注意力权重对更新门进行软调制。

**总损失**：

$$\mathcal{L} = \mathcal{L}_{target} + \beta \cdot \mathcal{L}_{aux}$$

## 4. 训练过程讲解

1. **行为 Embedding**：将用户历史行为序列映射为 Embedding 序列 $\{e_1, e_2, ..., e_T\}$
2. **兴趣提取**：GRU 逐步处理行为序列，输出兴趣隐状态 $\{h_1, h_2, ..., h_T\}$
3. **辅助损失计算**：用每个 $h_t$ 预测 $e_{t+1}$ 是否为真实下一步行为，增强表示学习
4. **注意力计算**：以候选广告为 Query，计算与各时刻 $h_t$ 的相关性
5. **AUGRU 演化**：注意力加权 GRU 逐步演化兴趣，输出最终兴趣表示 $h'_T$
6. **预测与反向传播**：拼接兴趣表示与其他特征，通过 MLP 输出 CTR 预测

## 5. 应用场景

- **电商 CTR 预估**：淘宝展示广告，用户行为序列长且兴趣随时间迁移
- **内容推荐**：新闻、短视频等信息流场景，用户兴趣存在明显的演化模式
- **搜索广告**：结合搜索 query 和用户行为历史进行精准预估
- **需要长序列行为建模的场景**：行为序列长度 >50 时 DIEN 优势更明显

## 6. 优缺点分析

**优点**：
- 建模兴趣演化过程，比 DIN 的静态兴趣激活更贴近真实用户行为
- 辅助损失有效增强兴趣表示质量，缓解 GRU 长序列信息遗忘
- AUGRU 将注意力融入门控机制，实现软性兴趣聚焦

**缺点**：
- 双层 GRU 导致推理延迟高于 DIN，线上部署需要优化
- 辅助损失引入额外计算开销，需要调节超参 $\beta$
- 对短行为序列用户（如新用户）效果有限
- 训练复杂度较高，收敛速度慢于 DIN

## 7. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class InterestExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, behavior_embs, hidden=None):
        outputs, hidden = self.gru(behavior_embs, hidden)
        return outputs, hidden


class AUGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, interests, attn_weights):
        B = interests.size(0)
        h = torch.zeros(B, self.hidden_dim, device=interests.device)
        for t in range(interests.size(1)):
            z = torch.sigmoid(nn.Linear(interests_dim + self.hidden_dim, self.hidden_dim)(
                torch.cat([interests[:, t], h], dim=-1)))
            r = torch.sigmoid(nn.Linear(interests_dim + self.hidden_dim, self.hidden_dim)(
                torch.cat([interests[:, t], h], dim=-1)))
            h_tilde = torch.tanh(nn.Linear(interests_dim + self.hidden_dim, self.hidden_dim)(
                torch.cat([interests[:, t], r * h], dim=-1)))
            u_t = z
            a_t = attn_weights[:, t].unsqueeze(-1)
            u_tilde = a_t * u_t
            h = (1 - u_tilde) * h + u_tilde * h_tilde
        return h


class DIEN(nn.Module):
    def __init__(self, num_items, embedding_dim=32, hidden_dim=32):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.interest_extractor = InterestExtractor(embedding_dim, hidden_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute_attention(self, interests, target_emb):
        target_expanded = target_emb.unsqueeze(1).expand_as(interests)
        concat = torch.cat([interests, target_expanded,
                            interests - target_expanded,
                            interests * target_expanded], dim=-1)
        scores = self.attn_mlp(concat).squeeze(-1)
        return torch.softmax(scores, dim=-1)

    def forward(self, behaviors, target):
        behavior_embs = self.item_emb(behaviors)
        target_emb = self.item_emb(target)
        interests, _ = self.interest_extractor(behavior_embs)
        attn = self.compute_attention(interests, target_emb)
        weighted = torch.bmm(attn.unsqueeze(1), interests).squeeze(1)
        concat = torch.cat([weighted, target_emb], dim=-1)
        return torch.sigmoid(self.mlp(concat))


num_items = 5000
seq_len = 30
model = DIEN(num_items)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(10):
    behaviors = torch.randint(0, num_items, (64, seq_len))
    targets = torch.randint(0, num_items, (64,))
    labels = torch.randint(0, 2, (64, 1)).float()
    preds = model(behaviors, targets)
    loss = criterion(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
```

## 8. 手工代码实现

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def gru_step(x_t, h_prev, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh):
    z = sigmoid(Wz @ x_t + Uz @ h_prev + bz)
    r = sigmoid(Wr @ x_t + Ur @ h_prev + br)
    h_tilde = np.tanh(Wh @ x_t + Uh @ (r * h_prev) + bh)
    h = (1 - z) * h_prev + z * h_tilde
    return h


def augru_step(x_t, h_prev, attn, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh):
    z = sigmoid(Wz @ x_t + Uz @ h_prev + bz)
    r = sigmoid(Wr @ x_t + Ur @ h_prev + br)
    h_tilde = np.tanh(Wh @ x_t + Uh @ (r * h_prev) + bh)
    u_tilde = attn * z
    h = (1 - u_tilde) * h_prev + u_tilde * h_tilde
    return h


np.random.seed(42)
d = 8
h_dim = 8
T = 5

Wz = np.random.randn(h_dim, d) * 0.1
Uz = np.random.randn(h_dim, h_dim) * 0.1
bz = np.zeros(h_dim)
Wr, Ur, br = [np.random.randn(h_dim, d) * 0.1, np.random.randn(h_dim, h_dim) * 0.1, np.zeros(h_dim)]
Wh, Uh, bh = [np.random.randn(h_dim, d) * 0.1, np.random.randn(h_dim, h_dim) * 0.1, np.zeros(h_dim)]

embs = np.random.randn(T, d)
attn_weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

h_gru = np.zeros(h_dim)
for t in range(T):
    h_gru = gru_step(embs[t], h_gru, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh)
print("标准 GRU 最终隐状态:", np.round(h_gru, 3))

h_augru = np.zeros(h_dim)
for t in range(T):
    h_augru = augru_step(embs[t], h_augru, attn_weights[t], Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh)
print("AUGRU 最终隐状态:", np.round(h_augru, 3))
```

## 9. 可视化与结果理解

- **兴趣演化轨迹**：将 AUGRU 各时刻的隐状态降维可视化，观察兴趣随时间的迁移路径
- **注意力分布**：不同候选广告下注意力分布不同，高权重集中在相关兴趣时刻
- **辅助损失收敛曲线**：观察辅助损失与主损失的同步下降情况

效果对比（阿里公开数据集）：

| 模型 | AUC | 相对提升 |
|------|-----|---------|
| Embedding+MLP | 0.6798 | baseline |
| DIN | 0.6957 | +2.3% |
| DIEN | **0.7021** | **+3.3%** |

## 10. 模型评估

- **AUC / GAUC**：衡量排序能力的核心指标
- **线上 CTR / RPM**：最终业务收益衡量
- **辅助损失监控**：辅助损失的下降反映兴趣表示质量的提升
- **推理延迟**：双层 GRU 的推理时间需满足线上 RT 要求（通常 <50ms）

## 11. 常见问题与易错点

- **辅助损失的负采样数量**：通常采样 1-4 个负样本，太多会增加计算开销
- **GRU 序列长度**：过长导致训练慢且梯度不稳定，建议截断到 50-100
- **AUGRU 与 AGRU 的区别**：AUGRU 将注意力乘在更新门 $u_t$ 上（软调制），AGRU 直接用注意力替代更新门（硬替代），AUGRU 效果更好
- **兴趣提取层与演化层的维度**：两层可以不同维度，但通常保持一致以简化实现

## 12. 学习总结

DIEN 的核心贡献是引入"兴趣演化"的概念：用户的兴趣不是静态的，而是随时间不断变迁的。通过 Interest Extractor + Interest Evolving 两层结构，DIEN 能捕捉兴趣的动态变化趋势。AUGRU 将注意力机制与 GRU 门控巧妙融合，是序列建模与目标注意力结合的典范。

## 13. 练习题与思考题（含答案）

**Q1：DIEN 相比 DIN 的核心改进是什么？**

A1：DIN 只做了兴趣激活——找出与当前候选相关的历史行为。DIEN 在此基础上建模兴趣的时序演化，通过 GRU 捕捉兴趣变迁趋势，能更好地理解用户从"浏览"到"比较"到"决策"的完整路径。

**Q2：为什么辅助损失能帮助训练？**

A2：单独的 GRU 只依赖最终的 CTR 损失反向传播，长序列场景下梯度难以有效传递到早期时刻。辅助损失在每个时刻都提供监督信号（预测下一个行为），使 GRU 的每个隐状态都包含有意义的兴趣信息，缓解长程梯度消失。

**Q3：AUGRU 与标准 GRU 的区别是什么？**

A3：标准 GRU 的更新门 $u_t$ 由输入和上一隐状态决定，对所有时刻同等对待。AUGRU 将注意力权重 $\alpha_t$ 乘到更新门上得到 $\tilde{u}_t = \alpha_t \cdot u_t$，使与目标更相关的时刻获得更大的更新幅度，实现目标驱动的兴趣演化。

## 14. 学习路径建议

```
DIN（目标注意力 + 行为序列建模）
        ↓
DIEN（兴趣演化 + GRU 时序建模）  ← 你在这里
        ↓
BST（Transformer 建模行为序列）
        ↓
SASRec（自注意力序列推荐）
        ↓
SIM / ETA（超长序列行为建模）
```
