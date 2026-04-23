# SENet（Squeeze-and-Excitation Network）推荐系统特征域注意力

## 算法定位

SENet 最初在计算机视觉领域提出，用于通道注意力。后被引入推荐系统（如 FiBinet），用于对特征域（Field）级别的重要性进行动态加权。

## 核心思想

SENet 的核心是一个三步操作：**Squeeze → Excitation → Re-Weight**，实现对每个特征域 embedding 的自适应重要性缩放。

## 数学公式

**输入**：Embedding 矩阵 $\mathbf{E} = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_f] \in \mathbb{R}^{f \times k}$

**Step 1: Squeeze（压缩）** — 均值池化

$$z_i = \frac{1}{k} \sum_{j=1}^{k} e_{i,j}, \quad \mathbf{z} = [z_1, z_2, \dots, z_f] \in \mathbb{R}^f$$

**Step 2: Excitation（激励）** — 两层 FC 瓶颈网络

$$\mathbf{A} = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z}))$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{f/r \times f}$（降维），$\mathbf{W}_2 \in \mathbb{R}^{f \times f/r}$（升维），$r$ 为压缩比。

**Step 3: Re-Weight（重加权）**

$$\mathbf{v}_i = a_i \cdot \mathbf{e}_i, \quad \mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_f]$$

## 业务创新点

- **动态性**：不同样本获得不同的特征域权重，而非静态的特征选择
- **轻量级**：仅引入两个小型 FC 层，参数量极小 $O(f^2/r)$
- **即插即用**：可嵌入任何基于 Embedding 拼接的 CTR 模型中
- **可解释性**：输出的注意力权重 $a_i$ 可直接反映各域重要性

## 详细原理推导

### 为什么需要特征域级注意力？

在推荐系统中，一个样本通常包含数十个特征域（如用户年龄、性别、物品类别、价格等）。并非所有特征域对每个样本都同等重要：

- 对于一个"电子产品"的推荐场景，"价格"特征域可能比"颜色"更重要
- 对于一个"时尚服饰"的推荐场景，"品牌"特征域可能比"重量"更重要

静态的特征选择方法（如基于互信息的特征筛选）无法捕捉这种**样本级别**的重要性差异。SENet 通过轻量级的门控机制，让模型自动学习每个样本应该关注哪些特征域。

### Squeeze 操作的数学本质

Squeeze 本质上是一个聚合操作，将每个特征域的 $k$ 维 embedding 压缩为一个标量。除了均值池化，也可以使用最大池化：

$$z_i^{\text{avg}} = \frac{1}{k} \sum_{j=1}^{k} e_{i,j}, \quad z_i^{\text{max}} = \max_{j} e_{i,j}$$

实践中均值池化更稳定，最大池化对异常值更敏感但在某些场景下能捕捉更显著的信号。

### Excitation 瓶颈结构的设计动机

瓶颈结构（先降维再升维）的设计有两个目的：

1. **参数效率**：直接使用 $\mathbb{R}^{f \times f}$ 的全连接层参数量为 $f^2$，而瓶颈结构参数量为 $2f \cdot f/r$，当 $r=2$ 时参数量减少约一半
2. **信息瓶颈效应**：降维迫使网络学习特征域之间的相关性模式，而非简单记忆

压缩比 $r$ 是一个关键超参数，通常取 2~3。过大的 $r$ 会丢失信息，过小的 $r$ 则无法有效压缩。

## SENet 在 FiBiNet 中的应用

FiBiNet（Feature Importance and Bilinear feature Interaction NETwork）是将 SENet 与双线性特征交叉结合的经典模型。其架构为：

1. **SENet 层**：对原始 embedding 施加 Squeeze-Excitation，生成重要性加权的 embedding
2. **双线性交叉层**：对原始 embedding 和 SENet 加权后的 embedding 分别做双线性交互
3. **DNN 层**：将交叉特征拼接后送入深度网络

FiBiNet 的关键创新在于：不仅用 SENet 做了特征域筛选，还用双线性交互（而非简单的内积或哈达玛积）来捕捉更丰富的特征交叉关系。双线性交叉的公式为：

$$\mathbf{c}_{ij} = \mathbf{e}_i^T \mathbf{W}_{ij} \mathbf{e}_j$$

其中 $\mathbf{W}_{ij}$ 是特征域 $i$ 和 $j$ 之间的交互矩阵，可以是共享的、域独立的或参数分解的。

## 与其他注意力机制对比

| 对比维度 | SENet（特征域注意力） | Self-Attention（自注意力） | FiBiNet 的注意力 |
|---------|---------------------|--------------------------|-----------------|
| 注意力粒度 | 特征域级别 | Token/位置级别 | 特征域级别 |
| 计算方式 | 全局池化 + FC | QK^T 点积 | 全局池化 + FC |
| 复杂度 | $O(f^2/r)$ | $O(n^2 d)$ | $O(f^2/r + f^2 k^2)$ |
| 适用场景 | CTR 预估中特征域筛选 | 序列建模、NLP | 特征交叉 + 特征筛选 |
| 是否保留原始信息 | 缩放（乘以权重） | 重新加权组合 | 分别用于双线性交叉 |
| 可解释性 | 强（权重直接对应特征域） | 弱（多头注意力难以解释） | 强 |

**SENet vs 逐特征注意力**：有些工作尝试对每个 embedding 维度施加独立注意力，但这会引入过多参数且容易过拟合。SENet 的域级别注意力是一个更好的归纳偏置。

**SENet vs FM 中的特征重要性**：FM 通过学习每个特征的 embedding 隐式表达特征重要性，但这是全局静态的。SENet 是样本级别动态的，两者互补。

## 优缺点深度分析

### 优点

1. **极低的计算开销**：仅增加两个小型全连接层，FLOPs 增加可忽略不计
2. **即插即用**：不需要修改原有模型结构，只需在 embedding 层后插入
3. **通用性强**：可嵌入 DCN、DeepFM、xDeepFM 等任何 CTR 模型
4. **可解释性**：输出的注意力权重可用于分析哪些特征域对当前预测贡献最大
5. **缓解噪声特征**：对不相关特征域自动降权，减少噪声干扰

### 缺点

1. **信息压缩损失**：Squeeze 步骤将整个 embedding 域压缩为一个标量，可能丢失域内的细粒度信息
2. **注意力粒度受限**：只能区分不同特征域的重要性，无法区分同一域内不同维度的重要性
3. **瓶颈结构限制**：降维-升维的结构可能无法捕捉复杂的特征域交互模式
4. **对冷启动特征不友好**：冷启动特征的 embedding 通常接近零向量，Squeeze 后更难获得有意义的注意力权重

## 超参数调优建议

| 超参数 | 推荐范围 | 说明 |
|--------|---------|------|
| reduction_ratio $r$ | 2~3 | 越小表达能力越强但参数越多 |
| 瓶颈层激活函数 | ReLU / SiLU | SiLU 在某些实验中略优 |
| 输出激活函数 | Sigmoid | 保证注意力权重在 [0, 1] |
| 是否共享 SENet | 不共享 | 对原始 embedding 和交叉特征分别使用 SENet 效果更好 |

## PyTorch 实现

### 基础 SENet 实现

```python
import torch
import torch.nn as nn

class SENet(nn.Module):
    def __init__(self, field_num, reduction_ratio=2):
        super().__init__()
        reduced = max(1, field_num // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(field_num, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, field_num, bias=False),
            nn.Sigmoid()
        )

    def forward(self, embeddings):
        z = embeddings.mean(dim=-1)
        a = self.excitation(z)
        return embeddings * a.unsqueeze(-1)
```

### FiBiNet 中的完整 SENet + 双线性交叉实现

```python
class BilinearInteraction(nn.Module):
    def __init__(self, field_num, emb_dim, bilinear_type="interaction"):
        super().__init__()
        self.bilinear_type = bilinear_type
        if bilinear_type == "all":
            self.bilinear = nn.Linear(emb_dim, emb_dim, bias=False)
        elif bilinear_type == "each":
            self.bilinear = nn.ModuleList([
                nn.Linear(emb_dim, emb_dim, bias=False)
                for _ in range(field_num)
            ])
        elif bilinear_type == "interaction":
            self.bilinear = nn.ModuleList([
                nn.Linear(emb_dim, emb_dim, bias=False)
                for _ in range(field_num * (field_num - 1) // 2)
            ])

    def forward(self, features):
        features = list(torch.unbind(features, dim=1))
        interactions = []
        idx = 0
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if self.bilinear_type == "all":
                    val = self.bilinear(features[i]) * features[j]
                elif self.bilinear_type == "each":
                    val = self.bilinear[i](features[i]) * features[j]
                elif self.bilinear_type == "interaction":
                    val = self.bilinear[idx](features[i]) * features[j]
                interactions.append(val)
                idx += 1
        return torch.stack(interactions, dim=1)


class FiBiNet(nn.Module):
    def __init__(self, field_num, emb_dim, hidden_dims, reduction_ratio=2, bilinear_type="interaction"):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(100, emb_dim) for _ in range(field_num)
        ])
        self.senet = SENet(field_num, reduction_ratio)
        self.bilinear = BilinearInteraction(field_num, emb_dim, bilinear_type)
        self.senet_bilinear = BilinearInteraction(field_num, emb_dim, bilinear_type)
        cross_dim = field_num * (field_num - 1)
        dnn_input_dim = cross_dim * emb_dim
        layers = []
        dims = [dnn_input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        emb = torch.stack([self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))], dim=1)
        senet_emb = self.senet(emb)
        cross = self.bilinear(emb)
        senet_cross = self.senet_bilinear(senet_emb)
        combined = torch.cat([cross, senet_cross], dim=1)
        combined = combined.flatten(start_dim=1)
        return torch.sigmoid(self.dnn(combined))


model = FiBiNet(field_num=10, emb_dim=8, hidden_dims=[128, 64])
x = torch.randint(0, 100, (32, 10))
print(model(x).shape)
```

## 常见问题与易错点

1. **注意 SENet 的输入形状**：输入应为 `(batch, field_num, emb_dim)`，Squeeze 在最后一个维度上做均值
2. **reduction_ratio 不宜过大**：当 field_num 较小（如 < 10）时，过大的 reduction_ratio 会导致瓶颈层维度为 1，严重限制表达能力
3. **不要对 one-hot 特征直接使用**：SENet 前需要先经过 embedding 层将稀疏特征转为稠密表示
4. **注意力权重监控**：训练过程中应监控 SENet 输出的注意力权重分布，如果所有权重趋近相同值，说明 SENet 未学到有效信息

## 学习总结

SENet 是推荐系统中特征域注意力机制的代表作，其核心价值在于用极低的成本为模型引入了样本级别的特征域重要性感知能力。在实际应用中，SENet 最常与 FiBiNet 结合使用，通过双线性交叉进一步增强特征交互能力。理解 SENet 的关键是把握"特征域级别注意力"这一粒度选择——它既避免了过细的逐维度注意力带来的过拟合风险，又比全局静态特征选择更具灵活性。

## 练习题

1. 如果将 SENet 的 Squeeze 操作从均值池化改为最大池化，在什么场景下可能更优？
2. 为什么 SENet 使用 Sigmoid 而非 Softmax 作为 Excitation 层的激活函数？
3. 设计一个实验来验证 SENet 中压缩比 $r$ 对模型性能的影响。

### 参考答案

1. 当特征域 embedding 中存在少量显著维度时（如某些域的 embedding 值分布极度偏斜），最大池化能更好地捕捉这些显著信号。但当 embedding 分布较均匀时，均值池化更稳定。
2. Sigmoid 保证每个域的注意力权重独立，允许模型同时"关注"多个特征域。Softmax 会强制各域权重之和为 1，形成竞争关系，可能导致某些重要域被不恰当地抑制。
3. 固定其他超参数，分别测试 $r \in \{1, 2, 3, 4, 8\}$，在验证集上记录 AUC 和 LogLoss。预期结果：$r$ 过小会导致参数冗余和过拟合，$r$ 过大会导致表达能力不足，通常 $r=2$ 或 $r=3$ 是最优的。
