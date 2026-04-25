# DIN (Deep Interest Network) 学习文档

## 1. 算法基础认知

DIN 是阿里巴巴在 2018 年提出的 CTR 预估模型，发表于 KDD 2018。

在电商广告场景中，用户的历史行为（点击、收藏、购买等）蕴含丰富的兴趣信息。传统方法将所有行为序列压缩为固定长度的向量，丢失了兴趣的多样性。DIN 的核心洞察是：

- 用户兴趣是**多样化**的，且往往**不互斥**
- 面对不同的候选广告，应该**自适应地激活**与之相关的历史行为
- 例如：用户点击过衣服和手机，当候选广告是球衣时，应该更关注其服装相关的历史行为

DIN 通过**目标注意力机制（Target Attention）**实现了这一思想，成为行为序列建模的里程碑工作。

## 2. 核心原理

DIN 的整体架构基于 Embedding & MLP 范式，核心创新在于**注意力激活单元（Activation Unit）**。

**传统做法：** 将用户所有行为 embedding 做 Sum / Mean Pooling → 得到固定长度用户表示。

**DIN 做法：** 对每个候选广告，动态计算它与历史行为的注意力权重，做加权求和。

关键模块：

- **Embedding 层：** 将稀疏特征（商品 ID、类目、用户画像等）映射为稠密向量
- **Activation Unit：** 计算行为与候选广告之间的相关性分数
- **加权池化：** 根据注意力权重对行为序列加权求和
- **MLP 网络：** 拼接所有特征后通过全连接层输出 CTR 预测

Activation Unit 的输入不仅仅是行为 embedding 和广告 embedding 的点积，而是将二者的差值、外积等交互信息拼接后通过 MLP 学习，表达能力更强。

## 3. 数学公式与推导

**注意力权重计算：**

$$\alpha_i = \frac{\exp(f(\mathbf{v}_i, \mathbf{v}_a))}{\sum_{j=1}^{T} \exp(f(\mathbf{v}_j, \mathbf{v}_a))}$$

其中 $\mathbf{v}_i$ 是第 $i$ 个历史行为的 embedding，$\mathbf{v}_a$ 是候选广告的 embedding。

**Activation Unit：**

$$f(\mathbf{v}_i, \mathbf{v}_a) = \text{ReLU}\left(\mathbf{W} \cdot [\mathbf{v}_i, \mathbf{v}_a, \mathbf{v}_i - \mathbf{v}_a, \mathbf{v}_i \odot \mathbf{v}_a] + \mathbf{b}\right)$$

这里 $[\cdot]$ 表示向量拼接，$\odot$ 表示逐元素乘积。差值和外积提供了显式的特征交互信息。

**用户兴趣表示：**

$$\mathbf{v}_U = \sum_{i=1}^{T} \alpha_i \cdot \mathbf{v}_i$$

**CTR 预测输出：**

$$\hat{y} = \sigma\left(\mathbf{W}_{\text{out}} \cdot [\mathbf{v}_U, \mathbf{v}_a, \text{other\_features}] + \mathbf{b}_{\text{out}}\right)$$

**损失函数（负对数似然）：**

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

## 4. 训练过程讲解

1. **数据准备：** 从用户行为日志中提取行为序列，每个样本包含用户历史行为列表 + 候选广告 + 标签（是否点击）
2. **Embedding 查表：** 将行为中的商品 ID、类目 ID 等映射为稠密向量
3. **注意力计算：** 对每个候选广告，遍历行为序列计算 Activation Unit 输出的注意力分数
4. **加权池化：** 对行为 embedding 做注意力加权求和，得到用户兴趣表示
5. **特征拼接与预测：** 将用户表示、广告 embedding、其他特征拼接，送入 MLP
6. **反向传播：** 使用 Adam 优化器，配合 Dice 激活函数进行 mini-batch 训练
7. **正则化：** 论文提出 Mini-batch Aware 正则化，只对每个 batch 中出现过的特征 embedding 做 L2 正则

## 5. 应用场景

- **淘宝展示广告 CTR 预估：** DIN 的原始应用场景，工业级验证
- **电商推荐系统：** 商品推荐、店铺推荐等有丰富行为历史的场景
- **内容信息流推荐：** 新闻、短视频等场景的用户兴趣建模
- **搜索广告：** 结合搜索 query 与用户行为进行精准预估
- **后续模型基线：** DIN 是 DIEN、BST、ETA 等序列模型的标准基线

## 6. 优缺点分析

**优点：**
- 自适应兴趣建模，不同广告激活不同历史行为，表达力强
- 注意力权重具有可解释性，可以分析用户关注点
- 用户表示不再固定，解决了 Sum Pooling 的信息压缩问题
- 工程落地性强，在阿里线上取得了显著的 CTR 提升

**缺点：**
- 每个候选广告都需要独立计算注意力，候选集大时推理开销高
- 仅建模兴趣激活，未建模兴趣的时序演化
- Dice 激活函数增加了训练复杂度
- 对行为序列长度敏感：太短信息不足，太长延迟增大

## 7. 调库实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActivationUnit(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.PReLU(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, behavior_emb, target_emb):
        diff = behavior_emb - target_emb
        prod = behavior_emb * target_emb
        concat = torch.cat([behavior_emb, target_emb, diff, prod], dim=-1)
        return self.mlp(concat).squeeze(-1)


class DIN(nn.Module):
    def __init__(self, num_items, embedding_dim=64, hidden_dims=[256, 128, 64]):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.activation_unit = ActivationUnit(embedding_dim)
        layers = []
        input_dim = embedding_dim * 3
        for h in hidden_dims:
            layers.extend([nn.Linear(input_dim, h), nn.PReLU(h)])
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, candidate_items, behavior_seqs, behavior_masks):
        candidate_emb = self.item_embedding(candidate_items)
        behavior_emb = self.item_embedding(behavior_seqs)
        attn_scores = self.activation_unit(behavior_emb, candidate_emb.unsqueeze(1))
        attn_scores = attn_scores.masked_fill(~behavior_masks, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~behavior_masks, 0.0)
        user_repr = torch.bmm(attn_weights.unsqueeze(1), behavior_emb).squeeze(1)
        concat_features = torch.cat([user_repr, candidate_emb], dim=-1)
        logits = self.mlp(concat_features)
        return torch.sigmoid(logits)


num_items = 10000
seq_len = 50
batch_size = 256

model = DIN(num_items)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(10):
    candidates = torch.randint(0, num_items, (batch_size,))
    behaviors = torch.randint(0, num_items, (batch_size, seq_len))
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    labels = torch.randint(0, 2, (batch_size,)).float()
    preds = model(candidates, behaviors, masks)
    loss = criterion(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
```

## 8. 手工代码实现

```python
import numpy as np


def activation_unit_forward(behavior_embs, target_emb, W1, b1, W2, b2):
    T = behavior_embs.shape[0]
    d = target_emb.shape[0]
    scores = np.zeros(T)
    for i in range(T):
        diff = behavior_embs[i] - target_emb
        prod = behavior_embs[i] * target_emb
        concat = np.concatenate([behavior_embs[i], target_emb, diff, prod])
        h = np.maximum(0, W1 @ concat + b1)
        scores[i] = (W2 @ h + b2).item()
    return scores


def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def attention_pooling(behavior_embs, target_emb, W1, b1, W2, b2):
    scores = activation_unit_forward(behavior_embs, target_emb, W1, b1, W2, b2)
    weights = softmax(scores)
    user_repr = np.sum(weights[:, None] * behavior_embs, axis=0)
    return user_repr, weights


np.random.seed(42)
d = 8
T = 10
behavior_embs = np.random.randn(T, d)
target_emb = np.random.randn(d)

hidden_dim = 16
W1 = np.random.randn(hidden_dim, d * 4) * 0.1
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(1, hidden_dim) * 0.1
b2 = np.zeros(1)

user_repr, weights = attention_pooling(behavior_embs, target_emb, W1, b1, W2, b2)

print("注意力权重:", np.round(weights, 3))
print("用户表示向量:", np.round(user_repr, 3))
print("最相关行为索引:", np.argmax(weights))
```

## 9. 可视化与结果理解

**注意力热力图分析：**

对于同一个用户，面对不同候选广告时，注意力分布应该呈现不同模式：
- 候选广告为"运动鞋"时 → 用户点击过的运动类商品获得更高权重
- 候选广告为"手机壳"时 → 用户点击过的数码配件类行为权重上升

**模型效果对比（淘宝公开数据集）：**

| 模型 | AUC | 相对提升 |
|------|-----|---------|
| LR | 0.6723 | baseline |
| FM | 0.6831 | +1.6% |
| Wide & Deep | 0.6862 | +2.1% |
| DIN | **0.6957** | **+3.5%** |

线上 A/B 测试中，DIN 相比基线模型 CTR 提升约 10%，RPM（千次展示收入）提升约 4%。

## 10. 模型评估

**离线指标：**
- **AUC（Area Under ROC Curve）：** 衡量模型排序能力的核心指标，DIN 在阿里数据集上 AUC 提升约 1-2 个百分点
- **LogLoss：** 交叉熵损失，反映预测概率的准确度
- **GAUC（Group AUC）：** 按用户分组计算 AUC 后加权平均，消除用户间点击率差异的影响，更贴合线上效果

**线上指标：**
- **CTR：** 点击率提升是最终衡量标准
- **CVR / RPM：** 转化率和收入指标

**评估建议：** 优先关注 GAUC 而非全局 AUC，因为广告系统中不同用户的点击行为差异很大。

## 11. 常见问题与易错点

**行为序列长度选择：** 太短（<10）会丢失长尾兴趣信息，太长（>200）会增加推理延迟。阿里论文中使用长度约 50。

**注意力坍缩问题：** 训练中可能出现注意力权重趋于均匀分布的现象。论文提出辅助损失（Auxiliary Loss），用行为序列中下一个行为作为正样本预测是否点击，帮助学习更丰富的 embedding。

**Dice 激活函数：** DIN 提出用 Dice 替代 PReLU：

$$\text{Dice}(x) = x \cdot \sigma\left(p \cdot \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}\right)$$

其中 $p$ 是可学习参数，$\sigma$ 是 sigmoid 函数。Dice 根据输入分布自适应调整激活，优于固定负斜率的 PReLU。

**负采样策略：** 正负样本比例通常严重不平衡（约 1:100），需要合理的负采样和采样权重调整。

**Mini-batch Aware 正则化：** 只对当前 batch 中实际出现的特征 ID 做 L2 正则，避免全量 embedding 正则化带来的计算开销。

## 12. 学习总结

DIN 的核心贡献是**目标注意力（Target Attention）**范式的提出：

- 将用户行为序列建模引入 CTR 预估领域
- 证明了"根据候选目标动态激活兴趣"比"固定压缩所有行为"更有效
- Activation Unit 的设计（拼接差值和外积）提供了丰富的交互信息
- 工程实践上提出了 Dice 激活和 MBA 正则化

这一范式深刻影响了后续工作：DIEN（兴趣演化）、BST（Transformer 建模行为）、ETA（端到端注意力）等，是理解现代推荐系统中用户建模的必经之路。

## 13. 练习题与思考题（含答案）

**Q1：DIN 为什么使用目标注意力而不是自注意力？**

A1：自注意力建模的是行为序列内部元素之间的关系，用于理解序列本身的结构。但在 CTR 预估中，核心问题是"这个用户对这个候选广告是否感兴趣"，因此需要以候选广告为 Query，去检索历史行为中与之相关的部分。目标注意力直接服务于预估目标。

**Q2：Dice 激活函数相比 PReLU 有什么优势？**

A2：PReLU 的负半轴斜率是固定的可学习标量 $f(x) = \max(0, x) + \alpha \cdot \min(0, x)$。Dice 根据当前 mini-batch 的均值和方差自适应调整激活的转折点，使得不同层的激活函数能够根据数据分布灵活调整，训练更稳定。

**Q3：DIN 与 DIEN 的区别是什么？**

A3：DIN 只建模了兴趣激活（Interest Activation）——哪些历史行为与当前广告相关。DIEN 在此基础上增加了兴趣演化（Interest Evolution）建模，使用 GRU + 注意力来捕捉兴趣随时间的变化趋势，能更好地理解用户的兴趣迁移过程。

**Q4：为什么 Activation Unit 要拼接差值和外积，而不只用原始 embedding？**

A4：原始 embedding 的拼接只能让 MLP 隐式学习交互关系。显式提供差值（$\mathbf{v}_i - \mathbf{v}_a$）和逐元素积（$\mathbf{v}_i \odot \mathbf{v}_a$）相当于给了网络先验知识，让模型更容易捕获行为与广告之间的相似性和差异性，加速收敛并提升效果。

## 14. 学习路径建议

```
Embedding + MLP 基线
        ↓
FM / DeepFM（特征交叉）
        ↓
DIN（目标注意力 + 行为序列建模）  ← 你在这里
        ↓
DIEN（兴趣演化 + GRU 时序建模）
        ↓
BST（Transformer 建模行为序列）
        ↓
SASRec（自注意力序列推荐）
        ↓
ETA / SIM（长序列行为建模）
```

推荐阅读顺序：
1. 先理解 DeepFM 等基础 CTR 模型
2. 精读 DIN 原论文，重点理解 Activation Unit 的设计动机
3. 动手实现 DIN 并在 MovieLens 或淘宝数据集上实验
4. 对比阅读 DIEN 论文，理解"兴趣演化"的必要性
5. 进阶学习 BST、SASRec，了解 Transformer 在行为建模中的应用
