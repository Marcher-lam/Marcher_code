# 面试题：CVR 样本稀疏问题如何解决？

面试题：CVR 样本稀疏问题如何解决？

在广告转化率（CVR）预估中，针对付费、下单等非常稀疏的转化样本问题，可通过多任务学习、对比学习、辅助建模等方法解决。

# 一、多任务学习与辅助建模

# 1. ESMM 全空间建模

核心思想：通过 CTR（点击率）和 CTCVR（点击后转化率）两个辅助任务联合建模 CVR，利用全量曝光样本而非仅点击样本，解决样本选择偏差和数据稀疏性。

# 实现方式：

 数学公式： $. p C T C V R = p C T R \times p C V R$ ，模型通过共享 Embedding 层从 CTR 任务中迁移特征表达，缓解 CVR任务样本稀疏的问题。  
 损失函数设计：仅优化 CTR 和 CTCVR 任务，避免直接处理 CVR 的稀疏 Label。

其他相关改进模型：Multi-IPW/DR、DCMT、ESCM²等。

![](images/8fc1bea79f0a4ef91f6c76d32595b15e6fc409c08fcceda93b6d20f6bfe8c8d6.jpg)

# 二、对比学习与特征增强

# 1. CL4CVR 论文框架

# 技术原理：

通过对比学习（如 Embedding Mask）生成增广样本，增强稀疏数据的特征表达。

给定给定锚点样本，可以将另一个增广样本作为正样本，而其他样本的增广样本作为负样本，在一个 batch 内，假设 batch_size $\mathrel { \mathop = } \mathsf { N }$ ，可以得到 2N个增广后的样本。可以构建经典的对比学习损失函数（NCE Loss）为下式，其中s(e_i, e_j)为余弦相似度， 是温度系数。

$$
L _ {0} = - \frac {1}{2 N} \sum_ {u = 1} ^ {2 N} \log \frac {\exp \left(s \left(e _ {i} , e _ {j}\right) / \tau\right)}{\sum_ {k \neq i} \exp \left(s \left(e _ {i} , e _ {k}\right)\right) / \tau}
$$

![](images/57354e8e93a1da1e77ca145d641a2adaa8c736e88d9ba2fbbe51f3b735406e29.jpg)

# CL4CVR 论文主要有以下 3 个组件：

# （1）Embedding Mask（EM）

方法：在特征嵌入(Embedding)维度随机 Mask 部分元素（非传统特征级 Mask），保留更多语义信息。EM对每个特征的嵌入随机遮蔽部分元素。可增强特征细粒度表达，避免破坏特征整体语义。

# （2）False Negative Elimination（FNE）

动机：用户行为存在不确定性（如多次点击同一商品但仅部分转化），相同特征可能对应不同 Label。

方法：在对比学习中排除与锚点样本特征相同但标签不同的样本。通过重复性指标判断特征是否相同，构建负样本集合时过滤特征冲突样本。

# （3）Supervised Positive Inclusion（SPI）

方法：转化标签稀疏但价值高，需充分利用。若锚点样本标签为转化（ $z = 1$ ），将同一批次内其他转化样本加入正样本集合，增强监督信号。

# 三、泛 Label 辅助建模优化

# 1. 泛化 Label 与辅助任务

例如，在飞猪高客单场景中，通过引入"用户在同类目商品购买"和"用户在同目的地商品购买"等泛化标签作为辅助任务，利用更丰富的辅助样本增强主任务学习。

通过共享 Embedding 层参数后，主任务 CVR 的稀疏性得到缓解，模型泛化能力提升。

# 稀疏场景下CVR模型优化-泛label建模

![](images/5807e4c2c8247cdce6d990243825fabfec3776b2183f152aec950fe03828dc4c.jpg)

![](images/ec71419e0a400f73398a0ec4c809480b423f65f8f830797c2ffad35b2c0cec62.jpg)

# 2. 层次化多任务建模（如 AutoHERI）

任务分解：将用户行为漏斗分解为"曝光 点击 商品详情页浏览 加购 $\longrightarrow$ 转化"多级任务。

层次聚合：通过多任务学习框架，自动学习前级任务（如 CTR、加购率）到后级任务（CVR）的特征聚合路径。利用前链路事件的任务（如 CTR、加购率）增强 CVR 建模。

---

# 四、数学推导补充

## 1. ESMM 的概率分解推导

在电商场景中，用户行为链路为：曝光 → 点击 → 转化。根据概率乘法规则：

$$
p(\text{点击且转化} | \text{曝光}) = p(\text{点击} | \text{曝光}) \times p(\text{转化} | \text{点击})
$$

即：

$$
pCTCVR = pCTR \times pCVR
$$

关键推导：由于 $pCTCVR$ 和 $pCTR$ 都可以在全量曝光样本上计算，因此 $pCVR = \frac{pCTCVR}{pCTR}$ 可以被间接推导出来，无需仅使用点击样本训练。

## 2. 对比学习损失的梯度分析

对于 InfoNCE 损失中锚点嵌入 $\mathbf{e}_i$ 的梯度：

$$
\frac{\partial L}{\partial \mathbf{e}_i} = \frac{1}{\tau}\left(\sum_{k \neq i} w_{ik} (\mathbf{e}_i - \mathbf{e}_k) - w_{ij} (\mathbf{e}_j - \mathbf{e}_i)\right)
$$

其中 $w_{ik} = \frac{\exp(s(\mathbf{e}_i, \mathbf{e}_k)/\tau)}{\sum_{k \neq i}\exp(s(\mathbf{e}_i, \mathbf{e}_k)/\tau)}$，梯度方向同时推开负样本、拉近正样本。

## 3. 样本选择偏差的形式化

传统 CVR 模型的训练数据分布为 $P(x | \text{clicked}=1)$，但线上推理时的数据分布为 $P(x | \text{exposed}=1)$。当两者不一致时：

$$
E_{x \sim P_{\text{train}}}[\text{loss}(x)] \neq E_{x \sim P_{\text{serve}}}[\text{loss}(x)]
$$

ESMM 通过在全量曝光样本上训练，使训练分布与线上分布一致，从根本上消除偏差。

# 五、各方法对比总结

| 方法 | 核心思路 | 稀疏缓解能力 | 实现复杂度 | 延迟影响 | 适用场景 |
|------|---------|-------------|-----------|---------|---------|
| ESMM | 全空间建模+多任务 | 强 | 低 | 无 | 电商广告CVR |
| CL4CVR | 对比学习+数据增强 | 中强 | 中 | 无 | 稀疏特征场景 |
| 泛Label建模 | 辅助任务+共享嵌入 | 中 | 低 | 无 | 多行为场景 |
| AutoHERI | 层次化多任务 | 强 | 高 | 低 | 行为漏斗清晰 |
| Multi-IPW | 逆倾向加权 | 中 | 中 | 无 | 偏差校正 |

# 六、应用场景

**电商广告**：用户点击后购买、加购等行为的转化率预估，转化样本通常不到点击样本的5%。

**游戏广告**：用户点击广告后下载、注册、付费的转化率预估，付费样本极度稀疏。

**金融风控**：用户授信后实际借款的转化预估，正样本比例极低。

**O2O 平台**：用户搜索后到店消费的转化预估，涉及线上到线下的跨域转化。

# 七、优缺点分析

## 优点

- 多任务学习充分利用了点击等密集信号，显著缓解数据稀疏
- 对比学习无需额外标注数据，通过数据增强自动生成训练信号
- 泛Label建模将稀疏的转化标签扩展到更丰富的行为信号
- 层次化建模符合用户行为的真实漏斗结构

## 缺点

- 多任务学习可能出现负迁移（辅助任务干扰主任务）
- 对比学习的负样本质量难以保证，可能引入假负样本
- 泛Label的选择需要领域知识，泛化不当可能引入噪声
- 方法组合时训练调参难度增大

# 八、Python 代码实现（ESMM 简化版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


class ESMM(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=16, hidden_dims=[64, 32]):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        input_dim = embed_dim * 2

        layers_ctr = []
        layers_cvr = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers_ctr.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            layers_cvr.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        layers_ctr.append(nn.Linear(prev_dim, 1))
        layers_cvr.append(nn.Linear(prev_dim, 1))

        self ctr_tower = nn.Sequential(*layers_ctr)
        self.cvr_tower = nn.Sequential(*layers_cvr)

    def forward(self, user_ids, item_ids):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        x = torch.cat([u, i], dim=-1)

        pctr = torch.sigmoid(self.ctr_tower(x))
        pcvr = torch.sigmoid(self.cvr_tower(x))
        pctcvr = pctr * pcvr

        return pctr, pcvr, pctcvr


def train_esmm():
    np.random.seed(42)
    torch.manual_seed(42)

    num_samples = 10000
    num_users = 1000
    num_items = 500

    user_ids = torch.randint(0, num_users, (num_samples,))
    item_ids = torch.randint(0, num_items, (num_samples,))

    clicked = (np.random.random(num_samples) < 0.1).astype(float)
    converted = clicked * (np.random.random(num_samples) < 0.05).astype(float)

    clicked_t = torch.FloatTensor(clicked).unsqueeze(-1)
    converted_t = torch.FloatTensor(converted).unsqueeze(-1)
    ctcvr_t = converted_t

    model = ESMM(num_users, num_items, embed_dim=16, hidden_dims=[64, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        pctr, pcvr, pctcvr = model(user_ids, item_ids)

        loss_ctr = F.binary_cross_entropy(pctr, clicked_t)
        loss_ctcvr = F.binary_cross_entropy(pctcvr.clamp(1e-8, 1 - 1e-8), ctcvr_t)
        loss = loss_ctr + loss_ctcvr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                  f"CTR Loss: {loss_ctr.item():.4f}, CTCVR Loss: {loss_ctcvr.item():.4f}")

    model.eval()
    with torch.no_grad():
        pctr, pcvr, pctcvr = model(user_ids, item_ids)
        ctr_auc = roc_auc_score(clicked, pctr.numpy())
        print(f"\nCTR AUC: {ctr_auc:.4f}")
        print(f"平均预测CVR: {pcvr.mean().item():.4f}")
        print(f"实际转化率: {converted.mean():.4f}")


train_esmm()
```

# 九、常见问题与易错点

## 1. pCTCVR 数值下溢

$pCTCVR = pCTR \times pCVR$，两个小数相乘可能接近0，导致梯度消失。实践中需要使用 `clamp` 操作确保数值稳定。

## 2. 点击样本的遗漏处理

ESMM 只在全量曝光样本上训练，但 CVR 信号仅来自点击样本。如果点击率本身很低（如1%），则有效的 CVR 信号更加稀疏。

## 3. 对比学习中的假负样本

用户可能对同一商品感兴趣但未转化，将其作为负样本会引入噪声。CL4CVR 的 FNE 模块通过特征一致性检查来缓解。

## 4. 多任务负迁移

当辅助任务与主任务的相关性较弱时，共享参数可能引入噪声。建议监控各任务独立指标，必要时使用 Task-Specific Layer。

# 十、学习路径建议

1. **基础**：理解 CVR 预估的任务定义和样本稀疏性成因
2. **核心**：深入学习 ESMM 系列论文（ESMM → ESM² → ESCM²）
3. **进阶**：掌握对比学习在推荐系统中的应用（CL4CVR、CoSeRec）
4. **拓展**：研究因果推断在推荐中的应用（IPW、DRL），从因果角度理解偏差问题
