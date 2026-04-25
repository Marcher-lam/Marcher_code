# COLD（Computing power cost-aware Online Lightweight Deep pre-ranking）学习文档

## 1. 算法基础认知

COLD 是阿里巴巴（2020）提出的计算资源感知粗排系统。它在双塔模型基础上引入 SE（Squeeze-and-Excitation）模块进行轻量级特征交叉，同时根据可用算力动态调整模型复杂度，是粗排模型从简单双塔向轻量交叉演进的重要里程碑。

## 2. 核心原理

### 模型公式

$$
\hat{y} = \sigma(\mathbf{u}^T \mathbf{v} + \text{SE}(\mathbf{x}))
$$

其中 $\mathbf{u}$ 为用户 Embedding，$\mathbf{v}$ 为广告 Embedding，SE 为 Squeeze-and-Excitation 注意力模块。

### SE 模块

$$
\text{SE}(\mathbf{x}) = \mathbf{s}^T \mathbf{x}, \quad \mathbf{s} = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{Pool}(\mathbf{x})))
$$

SE 模块学习特征维度的重要性权重，实现轻量级的特征选择和交互。

### 核心特点

- **计算资源感知**：根据可用算力动态调整模型复杂度
- **在线和轻量级**：适合粗排的高吞吐场景
- **SE 注意力**：在双塔基础上增加特征重要性建模

## 3. 数学公式与推导

**粗排打分**：

$$
\text{score} = \sigma(f_{dual\_tower}(u, v) + f_{SE}(x))
$$

**双塔内积**：$f_{dual\_tower}(u, v) = u^T v$

**SE 权重计算**：

$$
s_i = \sigma(W_2 \delta(W_1 z_i)), \quad z_i = \text{GAP}(x_i)
$$

**资源感知训练**：通过知识蒸馏将精排模型的知识迁移到粗排模型：

$$
L = \alpha L_{CE}(y, \hat{y}) + (1-\alpha) L_{KL}(p_{teacher} \| p_{student})
$$

## 4. 训练过程讲解

1. 用户特征和广告特征分别过 Embedding 层得到 $\mathbf{u}$ 和 $\mathbf{v}$
2. 计算双塔内积 $u^T v$ 作为基础打分
3. 拼接特征向量通过 SE 模块计算特征重要性加权分数
4. 两个分数相加后过 Sigmoid 得到预测概率
5. 使用蒸馏损失 + 交叉熵损失联合训练
6. 部署时根据算力预算决定 SE 模块的复杂度

## 5. 应用场景

- 广告粗排阶段（阿里妈妈广告系统）
- 需要平衡精度和延迟的高吞吐场景
- 多级排序架构中的前置筛选环节
- 适合千级→百级的候选集筛选

### 粗排模型对比

| 模型 | 核心公式 | 复杂度 | 特点 |
|------|---------|--------|------|
| 双塔 | $\hat{y} = \sigma(u^T v)$ | $O(d)$ | 最快，无交叉 |
| 交叉双塔 | $\hat{y} = \sigma(u^T v + u^T W v)$ | $O(d^2)$ | 轻量交叉 |
| COLD | $\hat{y} = \sigma(u^T v + SE(x))$ | $O(d+k)$ | SE 注意力 |
| 蒸馏 | $L = \alpha L_{CE} + (1-\alpha)L_{KL}$ | - | 继承精排知识 |

## 6. 优缺点分析

**优点**：
- 在双塔基础上增加轻量交叉，精度提升但延迟增加可控
- SE 模块提供特征可解释性（特征重要性权重）
- 资源感知设计可适配不同算力环境

**缺点**：
- SE 模块引入额外计算，在高 QPS 场景仍有延迟压力
- 交叉能力有限，不如精排模型（DIN/DCN）精细
- 需要精心调节蒸馏损失权重 $\alpha$

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import torch
import torch.nn as nn

class SEModule(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.relu(self.fc1(x))
        s = self.sigmoid(self.fc2(s))
        return (x * s).sum(dim=-1, keepdim=True)

class COLD(nn.Module):
    def __init__(self, user_dim, ad_dim, feat_dim, hidden=64, embed=32):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed)
        )
        self.ad_tower = nn.Sequential(
            nn.Linear(ad_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed)
        )
        self.se = SEModule(feat_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_feat, ad_feat, cross_feat):
        u = self.user_tower(user_feat)
        v = self.ad_tower(ad_feat)
        dual_score = torch.sum(u * v, dim=-1, keepdim=True)
        se_score = self.se(cross_feat)
        return self.sigmoid(dual_score + se_score)

model = COLD(user_dim=50, ad_dim=50, feat_dim=100)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
u_feat = torch.randn(32, 50)
a_feat = torch.randn(32, 50)
c_feat = torch.randn(32, 100)
y = torch.randint(0, 2, (32, 1)).float()
for epoch in range(10):
    pred = model(u_feat, a_feat, c_feat)
    loss = nn.BCELoss()(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

class COLDNumpy:
    def __init__(self, user_dim, ad_dim, feat_dim, embed=16, reduction=4):
        scale = 0.01
        self.Wu1 = np.random.randn(user_dim, embed) * scale
        self.Wu2 = np.random.randn(ad_dim, embed) * scale
        self.W_se1 = np.random.randn(feat_dim, feat_dim // reduction) * scale
        self.W_se2 = np.random.randn(feat_dim // reduction, feat_dim) * scale

    def se_module(self, x):
        z = relu(x @ self.W_se1)
        s = sigmoid(z @ self.W_se2)
        return (x * s).sum(axis=-1, keepdims=True)

    def predict(self, user_feat, ad_feat, cross_feat):
        u = user_feat @ self.Wu1
        v = ad_feat @ self.Wu2
        dual = np.sum(u * v, axis=-1, keepdims=True)
        se = self.se_module(cross_feat)
        return sigmoid(dual + se)
```

## 9. 可视化与结果理解

- 绘制 SE 模块输出的特征重要性权重分布，识别关键特征
- 对比纯双塔 vs COLD 的 Recall@K 曲线
- 展示不同算力预算下模型精度的 trade-off 曲线

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_channels = 20
channel_names = [f'Ch{i + 1}' for i in range(n_channels)]

se_weights = np.random.uniform(0.1, 0.9, n_channels)
se_weights[[2, 5, 8, 14, 17]] = np.random.uniform(0.85, 0.99, 5)
se_weights[[1, 7, 11]] = np.random.uniform(0.02, 0.1, 3)

sort_idx = np.argsort(se_weights)[::-1]
sorted_weights = se_weights[sort_idx]
sorted_names = [channel_names[i] for i in sort_idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = ['#F44336' if w > 0.8 else '#FFEB3B' if w > 0.4 else '#9E9E9E' for w in sorted_weights]
bars = axes[0].barh(range(n_channels), sorted_weights, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_yticks(range(n_channels))
axes[0].set_yticklabels(sorted_names, fontsize=10)
axes[0].set_xlabel('SE Channel Weight', fontsize=12)
axes[0].set_title('SE Attention Module — Channel Importance', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold=0.5')
for i, w in enumerate(sorted_weights):
    axes[0].text(w + 0.01, i, f'{w:.2f}', va='center', fontsize=9)
axes[0].legend(fontsize=9)
axes[0].grid(axis='x', alpha=0.3)

highlight = ['Ch3', 'Ch6', 'Ch9', 'Ch15', 'Ch18']
highlight_colors = ['#E53935' if sorted_names[i].replace('Ch', '') in ['3', '6', '9', '15', '18'] else '#BDBDBD' for i in range(n_channels)]

axes[1].hist(se_weights, bins=15, color='#42A5F5', edgecolor='black', alpha=0.8)
axes[1].set_xlabel('SE Weight Value', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('SE Weight Distribution', fontsize=13, fontweight='bold')
axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
axes[1].axvline(x=np.mean(se_weights), color='green', linestyle='--', linewidth=2, label=f'Mean={np.mean(se_weights):.2f}')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('COLD — SE Module Channel Attention Weights', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cold_se_weights_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **粗排核心指标**：Recall@K（召回率）、Hit Rate
- **效率指标**：QPS（每秒查询数）、P99 延迟
- **对比基线**：与纯双塔模型对比精度和延迟

## 11. 常见问题与易错点

- SE 模块的 reduction ratio 需要根据特征维度调节，过大会丢失信息
- 粗排模型的蒸馏损失权重 $\alpha$ 需要在线上 A/B 测试调优
- 双塔 Embedding 维度过高会增加延迟，需在精度和速度间权衡
- 粗排目标应与精排目标对齐，否则会导致"通道不匹配"

## 12. 学习总结

COLD 的核心贡献在于首次将"计算资源感知"引入粗排模型设计：不是追求模型精度的最大化，而是在给定延迟预算（QPS 和 P99 延迟约束）下寻找精度最优的模型配置。SE（Squeeze-and-Excitation）模块以极小的额外计算代价实现了特征重要性的自适应建模，使粗排模型突破了纯双塔无法特征交叉的表达力瓶颈。

COLD 的关键优势是在精度和延迟之间取得了精巧的平衡，SE 模块提供特征重要性权重具有可解释性，且可根据算力动态调整模型复杂度。它最适合广告/推荐系统中的粗排阶段，需要在毫秒级延迟内对数千候选进行打分筛选。但 SE 的交叉能力仍有限，无法替代精排模型（如 DIN、DCN）的深度交叉。

在知识体系中，COLD 是本库中双塔模型向精排模型过渡的桥梁——它在双塔架构上引入轻量级注意力（SE），与 Attention 机制的思想一脉相承，同时通过知识蒸馏从精排模型（如本库中的 DIN）迁移知识。它代表了"计算感知建模"这一工业实战中的重要理念。

工业实践中，COLD 的 SE 模块 reduction ratio 和蒸馏损失权重 $\alpha$ 需要在 A/B 测试中精细调优。粗排目标必须与精排目标对齐（如都用 CTR+CVR 联合优化），否则粗排选出的候选在精排阶段会被大量拒绝，造成"通道不匹配"问题。

## 13. 练习题与思考题（含答案）

**Q1**: COLD 相比纯双塔模型的核心改进是什么？
> A1: 引入 SE 模块实现轻量级特征交叉，增强模型表达能力的同时保持低延迟。

**Q2**: 为什么粗排模型需要"资源感知"？
> A2: 粗排需要处理大量候选（数千个），延迟预算极紧（通常 <10ms），必须根据算力动态调整复杂度。

**Q3**: SE 模块中 reduction ratio 的作用是什么？
> A3: 控制 bottleneck 层的维度，实现特征压缩后再扩展，学习通道间的重要性权重。

## 14. 学习路径建议

1. 先学习 DSSM（双塔模型）理解基础架构
2. 学习 SENet（Squeeze-and-Excitation）理解注意力机制
3. 学习知识蒸馏（Knowledge Distillation）理解模型压缩
4. 进阶：学习 FSCD、COPP 等更新的粗排模型
