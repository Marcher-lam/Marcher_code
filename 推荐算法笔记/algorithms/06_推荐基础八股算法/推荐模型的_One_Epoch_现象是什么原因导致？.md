# 面试题：推荐模型的 One Epoch 现象是什么原因导致？

面试题：推荐模型的 One Epoch 现象是什么原因导致？

相关论文：Towards Understanding the Overfitting Phenomenon of Deep CTR Prediction

# 一、One Epoch 现象的定义

One Epoch 现象是指在深度点击率（CTR）预估模型的训练过程中，测试集 AUC（模型效果指标）在第一个 epoch 内逐步提升，但从第二个 epoch 开始突然剧烈下降的现象。这种现象在工业界（如阿里、快手等）的推荐系统中普遍存在，其核心特点是：

- 时间点明确：恰好出现在第二个 epoch 开始时
- 突发性：效果下降剧烈且迅速，而非缓慢过拟合

![](images/affd99f79e5ddc15e9384413846ba26f8bc38880eae3eb59fff59c224fa9bd19.jpg)

**One Epoch 与传统过拟合的区别：** 传统过拟合表现为验证集指标随训练轮次缓慢下降（渐进式），而 One Epoch 现象是断崖式下降，在第二个 epoch 开始的瞬间就急剧恶化。这种特殊性表明其成因不同于一般的过拟合机制。

# 二、原理与机制

# 1. Embedding 与 MLP 层的联合分布适配

深度 CTR 模型通常采用Embedding+MLP 结构：Embedding 层将高维稀疏特征（如用户 ID、商品 ID）映射为低维向量，MLP层基于这些向量进行预测。

- 在第一个 epoch 中，Embedding 层和 MLP 层共同学习训练数据的联合分布，模型逐渐收敛至较优状态
- 进入第二个 epoch 时，MLP 层会快速适配已训练过的 Embedding 分布，导致对训练数据的过度拟合。此时，Embedding层参数相对稳定，但 MLP 层参数剧烈调整，使得模型无法泛化到未见过的测试数据

**数学解释：** 设 Embedding 层参数为 $\Theta_E$，MLP 层参数为 $\Theta_M$。在第一个 epoch 中，联合优化目标为：

$$
\min_{\Theta_E, \Theta_M} \mathcal{L}(f_{\Theta_E, \Theta_M}(X), Y)
$$

第一个 epoch 结束后，$\Theta_E$ 已经对训练集中的 ID 特征形成了特定的 Embedding 映射。第二个 epoch 中，MLP 层快速适配这些已固定的 Embedding 模式，记忆了 $\Theta_E \to Y$ 的映射关系，而非学习 $X \to Y$ 的泛化模式。

# 2. 训练数据与非训练数据的分布差异

- 推荐系统的特征具有高维稀疏性（如长尾 ID 特征），导致训练数据与非训练数据（如测试集或线上新数据）的 Embedding分布差异显著
- 在第二个 epoch 中，模型重新接触训练数据时，MLP 层会优先适应已见过的 Embedding 分布，而非学习更泛化的模式，从而加剧过拟合

# 三、核心原因分析

根据阿里团队的研究，One Epoch 现象主要由以下三方面因素共同作用引起：

# 1. 模型结构特性（Embedding+MLP）

a. Embedding 层的敏感性：稀疏 ID 特征的高维性导致 Embedding 层容易过拟合，尤其当特征出现频率低时（长尾 ID），Embedding 向量难以充分学习泛化表示

b. MLP 层的快速适应：MLP 层在第 2 个 epoch 迅速调整权重，优先拟合训练数据 Embedding 分布，而非学习真实特征关系

**实验证据：** 阿里团队对比了不同模型结构，发现：
- 纯 LR 模型（无 Embedding 层）不存在 One Epoch 现象
- 纯 MLP 模型（使用稠密特征，无 Embedding 层）也不存在此现象
- 只有 Embedding+MLP 结构才会出现 One Epoch 现象

这证明了 Embedding 与 MLP 的交互是 One Epoch 现象的必要条件。

# 2. 优化器的快速收敛特性

a. 使用 Adam、RMSprop 等强优化器或大学习率时，模型在第 1 个 epoch 内快速收敛至局部最优
b. 这种快速收敛导致模型在第二个 epoch 中缺乏继续探索能力，过度拟合训练数据

**优化器对比实验：** 研究表明，使用 SGD 优化器时 One Epoch 现象有所缓解（因为 SGD 收敛较慢，第二个 epoch 的更新幅度较小），但训练效率大幅降低。使用 Adam 时，第二个 epoch 的参数更新方向主要是记忆训练数据，导致快速过拟合。

# 3. 特征稀疏性与数据分布特性

a. 高维稀疏特征（如用户 ID、商品 ID）是推荐系统的核心特征，但这些特征的稀疏性（尤其是长尾 ID）导致模型在第二个 epoch 中难以泛化

b. 实验表明，通过减少稀疏性（如过滤低频 ID、哈希压缩）可显著缓解 One Epoch 现象，但会牺牲模型精度

**稀疏性定量分析：** 研究者通过控制实验发现，当 ID 特征的最低出现频次阈值从 1 增加到 10 时，One Epoch 现象显著减轻，但模型精度下降约 2-3%。这说明稀疏性是 One Epoch 现象的重要诱因。

# 四、实验验证与结论

# 1. 关键实验发现

- 模型结构对比：LR 模型无此现象，而 Embedding+MLP 结构的深度模型普遍存在 One Epoch 现象
- 参数无关性：模型参数量、激活函数、Batch Size、正则化（如 Weight Decay、Dropout）等与现象无关
- 稀疏性影响：减少特征稀疏性（如压缩ID空间）可缓解现象，但牺牲模型效果

**参数无关性的重要性：** 这一发现排除了模型容量过大导致过拟合的假设。无论模型参数量大小、是否使用 Dropout 或 Weight Decay，One Epoch 现象依然存在。这进一步证明问题的根源在于 Embedding+MLP 的交互机制，而非模型的正则化不足。

# 2. 工业实践启示

- 阿里、快手等公司主流方案是仅训练一个 epoch，或采用流式训练（数据仅使用一次），以避免效果下降
- 快手提出的 MEDA 方法（每个 epoch 重新初始化 Embedding 层）通过数据增强缓解过拟合，但需权衡计算成本

# 五、缓解 One Epoch 现象的策略

| 策略 | 原理 | 效果 | 代价 |
|------|------|------|------|
| 单 Epoch 训练 | 只使用数据一次，避免重复暴露 | 工业界主流方案 | 数据利用率低 |
| 流式训练 | 数据按时间流式输入，不重复 | 线上实时性好 | 需要持续数据源 |
| MEDA | 每 epoch 重初始化 Embedding | 显著缓解 | 计算成本高 |
| 知识蒸馏 | 用第一 epoch 模型指导后续训练 | 缓解过拟合 | 需额外蒸馏框架 |
| 数据增强 | 对 Embedding 加入噪声/扰动 | 缓解分布偏移 | 需调参 |

# 六、Python 复现实验

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

class CTRModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=16, hidden_dims=[64, 32]):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        layers = []
        input_dim = embed_dim * 2
        for h in hidden_dims:
            layers.extend([nn.Linear(input_dim, h), nn.ReLU()])
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        x = torch.cat([u, i], dim=-1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)

np.random.seed(42)
num_users, num_items = 1000, 5000
n_train, n_test = 50000, 10000

train_users = np.random.randint(0, num_users, n_train)
train_items = np.random.randint(0, num_items, n_train)
train_labels = (np.random.random(n_train) > 0.7).astype(float)

test_users = np.random.randint(0, num_users, n_test)
test_items = np.random.randint(0, num_items, n_test)
test_labels = (np.random.random(n_test) > 0.7).astype(float)

train_ds = TensorDataset(
    torch.LongTensor(train_users), torch.LongTensor(train_items), torch.FloatTensor(train_labels)
)
test_ds = TensorDataset(
    torch.LongTensor(test_users), torch.LongTensor(test_items), torch.FloatTensor(test_labels)
)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

model = CTRModel(num_users, num_items)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(5):
    model.train()
    for users, items, labels in train_loader:
        preds = model(users, items)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for users, items, labels in test_loader:
            preds = model(users, items)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}: Test AUC = {auc:.4f}")
```

总结：One Epoch 现象的本质是深度 CTR 模型在高维稀疏特征下，因 Embedding 与 MLP 层的联合分布适配失衡导致的突发性过拟合。其解决需在模型结构、优化策略与特征工程间权衡，而工业界更倾向于通过单 epoch 训练或动态更新机制平衡效果与效率。

在 Self-Attention 的计算公式中，除以 $\sqrt{d_k}$ 的核心目的是控制点积的数值范围，避免梯度消失并稳定训练过程。

# 1. 防止 Softmax输入过大导致梯度消失

- 问题背景：当 $Q$ 和 $K$ 的点积值过大时，Softmax 函数会进入"饱和区"（即输入值过大时，输出的概率分布接近极端值0或1），此时Softmax的梯度趋近于 0，导致反向传播时参数更新困难。
- 数学推导：假设 Q 和 $K$ 的维度为 $d_k$，若每个元素的方差为 1，则点积 $QK^T$ 的方差为 $d_k$，标准差为 $\sqrt{d_k}$。除以 $\sqrt{d_k}$ 后，点积的方差被缩放为 1，数值范围更稳定，避免 Softmax 梯度消失。

# 2. 保持注意力分数的方差稳定

- 统计假设：假设 $Q$ 和 $K$ 的元素是独立同分布的随机变量，均值为 0，方差为 1。QK 点积再除以 $\sqrt{d_k}$ 的方差为：

$$
\text{Var}\left(\frac{Q \cdot K}{\sqrt{d_k}}\right) = \frac{1}{d_k} \sum_{i=1}^{d_k} \text{Var}(Q_i K_i) = \frac{1}{d_k}[d_k \cdot \text{Var}(Q_i) \text{Var}(K_i)] = 1
$$

因此，除以 $\sqrt{d_k}$ 后点积结果的方差为 1，使注意力分数的分布更符合 Softmax 的输入要求。

# 3. 适应不同维度的嵌入空间

- 维度影响：当嵌入维度 $d_k$ 较高时（如 Transformer 中常见的 512 或 1024 维），点积的绝对值会随维度增加而显著增大。例如，在低维空间中点积可能为个位数，而在高维空间中可能达到数百甚至上千。缩放操作能统一不同维度的数值范围，确保模型在不同层和不同配置下的行为一致。
- 实验验证：通过对比不同维度下的点积方差（如 3 维和 512 维），可观察到高维点积的方差远大于低维，验证了缩放的必要性。

总结：除以 $\sqrt{d_k}$ 的实质是一种数值稳定性设计，包含以下作用：

- 避免 Softmax 梯度消失：控制输入范围，防止训练停滞
- 统计归一化：使注意力分数的分布稳定（均值为 0，方差为 1）
- 统一多维度场景：消除嵌入维度对数值范围的干扰
