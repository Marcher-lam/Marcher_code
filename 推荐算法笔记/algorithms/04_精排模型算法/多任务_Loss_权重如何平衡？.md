# 面试题：多任务 Loss 权重如何平衡？

面试题：多任务 Loss 权重如何平衡？

在推荐系统多任务学习中，不同任务的损失函数（Loss）权重分配直接影响模型的优化方向和最终性能。

几种主流的 loss 权衡方法如下：

# 1. 固定权重加权平均

$$
L _ {t o t a l} = \sum_ {i} ^ {N} w _ {i} \cdot L _ {i}
$$

 原理：为每个任务的损失分配固定权重，通过人工经验或网格搜索确定权重组合。  
 公式： ，其中 $w _ { i }$ 为固定权重，需手动调整； $L _ { i }$ 为第 i 个任务的损失。  
 适用场景：任务间相关性高或先验知识明确的情况，依据人工经验拍定权重。  
 缺点：难以动态适应训练不同阶段的权重需求，易受任务量级差异影响。

# 2. 基于不确定性的权重调整（Uncertainty Weighting）

 原理：通过建模任务的不确定性动态调整权重。不确定性越大（任务噪声多或难度高），权重越小。

$$
L _ {t o t a l} = \sum_ {i = 1} ^ {N} \left(\frac {1}{2 \sigma_ {i} ^ {2}} L _ {i} + \log \sigma_ {i}\right)
$$

公式：

$\sigma _ { i }$ 为可学习参数，表示任务 i 的不确定性。

推导：假设每个任务的损失服从高斯分布，通过最大化似然估计推导出上述公式。在反向传播中， $\sigma _ { i }$ 会自适应调整。  
优势：无需人工干预，自动平衡分类与回归任务的权重。

# 3. 动态加权平均（DWA，Dynamic Weight Averaging）

 原理：根据任务的学习速度动态调整权重。Loss下降快的任务权重降低，反之权重增加。

$$
w _ {i} (t) = \frac {N \cdot e ^ {r _ {i} (t - 1) / T}}{\sum_ {k = 1} ^ {N} e ^ {r _ {k} (t - 1) / T}}, \quad r _ {i} (t - 1) = \frac {L _ {i} (t - 1)}{L _ {i} (t - 2)}
$$

公式：

$r _ { i }$ 表示任务 i 的损失下降速率；T 为温度参数，控制权重分布平滑度。

应用：适用于任务学习速度差异显著的场景（如点击率 CTR 与转化率 CVR 预测）。

# 4. 梯度标准化（GradNorm）

 原理：在原来的总损失以外，额外引入梯度标准化 Loss，通过平衡各任务梯度的 L2 范数，使所有任务以相近速度学习。

 步骤：

 计算梯度范数：对共享层参数 W，计算各任务梯度的 L2 范数 $G _ { i } ( t )$ 。  
定义目标梯度范数：

$$
\tilde {G} _ {i} (t) = \bar {G} (t) \cdot [ r _ {i} (t) ] ^ {\alpha}
$$

$$
\tilde {L} _ {i} (t) = \frac {L _ {i} (t)}{L _ {i} (0)}, \quad r _ {i} (t) = \frac {\tilde {L} _ {i} (t)}{E _ {\text {t a s k}} [ \tilde {L} _ {i} (t) ]}
$$

为平均梯度范数； 控制任务学习速度平衡强度。 $\bar { G } _ { i } ( t )$ $_ \alpha$

 优化梯度范数Loss：最小化实际梯度范数与目标梯度范数的差异：

$$
L _ {g r a d} = \sum_ {i} ^ {N} | G _ {i} (t) - \tilde {G} _ {i} (t) |
$$

优势：有效解决梯度冲突，尤其适用于任务复杂度差异大的场景。

# 5. 动态任务优先级（DTP，Dynamic Task Prioritization）

原理：根据任务的关键指标（KPI）动态调整权重，KPI 高的任务权重降低。

$$
w _ {i} (t) = \frac {\left(1 - k _ {i} (t) ^ {\gamma_ {i}}\right)}{\sum_ {j = 1} ^ {N} \left(1 - k _ {j} (t) ^ {\gamma_ {j}}\right)}
$$

为任务 i 在时间步 t 的 KPI（如准确率、AUC 等）； 为人工调节参数。 $\gamma _ { i }$

应用：推荐系统中任务重要性评估指标明确的情况。

不同方法对比：  

<table><tr><td>方法</td><td>优点</td><td>缺点</td><td>适用场景</td></tr><tr><td>固定权重加权平均</td><td>简单易实现</td><td>依赖人工调参，灵活性差</td><td>任务量级相近且相关性高</td></tr><tr><td>不确定性加权</td><td>自适应平衡分类/回归任务</td><td>需引入额外参数，可能训练不稳定</td><td>多任务类型混合（如CTR+CVR）</td></tr><tr><td>GradNorm</td><td>解决梯度冲突，平衡学习速度</td><td>计算复杂度高，需调参α</td><td>任务复杂度差异大</td></tr><tr><td>DWA</td><td>无需梯度计算，实现简单</td><td>对温度参数T敏感</td><td>任务学习速度差异显著</td></tr></table>

MMoE（Multi-gate Mixture-of-Experts）和 PLE（Progressive Layered Extraction）是多任务学习（MTL，Multi-Task Learning）中的两种代表性模型，它们的核心区别如下：

# 1. 核心结构设计对比

MMoE：通过动态调整共享专家权重实现多任务学习。所有任务共享同一组专家网络（Experts），每个任务通过独立的门控网络（Gate）计算专家权重，组合不同专家的输出作为任务输入。

对于任务 k，其输出 $_ { y _ { k } }$ 为：

$$
y _ {k} = h ^ {k} \left(f ^ {k} (x)\right), \quad f ^ {k} (x) = \sum_ {i = 1} ^ {n} g ^ {k} (x) _ {i} \cdot f _ {i} (x), \quad g ^ {k} (x) = \operatorname {s o f t m a x} \left(W _ {g k} x\right)
$$

 $f _ { i } ( x )$ ：第 i 个共享专家网络的输出。  
：任务 k 的门控网络，其输出是一个概率分布，表示该任务对每个共享专家的权重。门控网络的输 $g ^ { k } ( x )$ 入是原始特征 ${ \sf X } _ { \sf \circ }$ 。  
：任务 k 专用的 Tower 网络。 $h ^ { k } ( \cdot )$

PLE：采用分层提取机制，显式区分共享专家和任务专属专家。通过渐进式分层结构（多级 CGC网络），逐层分离共享特征与任务特定特征，减少任务间的参数干扰。

PLE 的结构更复杂，这里以单层 CGC（PLE 的基础模块）中任务 k 的融合过程为例：

$$
y _ {k} = h ^ {k} \left(f ^ {k} (x)\right)
$$

$$
f ^ {k} (x) = \sum_ {i = 1} ^ {m _ {k}} g ^ {k} (x) _ {i} \cdot E _ {(k, i)} (x) + \sum_ {j = 1} ^ {m _ {s}} g ^ {k} (x) _ {m _ {k} + j} \cdot E _ {(s, j)} (x)
$$

$$
g ^ {k} (x) = \operatorname {s o f t m a x} \left(W _ {g k} \cdot [ x; S ^ {k} (x) ]\right)
$$

：任务 k 的第 i 个任务专属专家的输出。 $E _ { \left( k , i \right) } ( x )$ $\mathsf { k }$   
， ：第 j 个共享专家的输出。 $E _ { ( s , j ) } ( x )$   
 ：任务 k 的门控网络的输入，它由所有任务专属专家和共享专家的输出拼接而成； $S ^ { k } ( x )$ $\mathsf { k }$   
 ：任务 k 的门控网络，其权重基于更丰富的输入（融合了原始特征和专家输出）计算得出，从而能 $g ^ { k } ( x )$ 更精准地分配权重。

# 2. 专家网络的配置

 MMoE：仅包含共享专家，所有任务共用同一组专家，缺乏任务专属参数空间。这可能导致任务冲突时专家被不同任

务“撕扯”，影响效果。

 PLE：引入共享专家+任务专属专家的双轨结构。例如，任务 A 的输入由其专属专家和共享专家共同组合而成，其他任务的专属专家不参与该任务计算，从而减少噪声。

# 3. 任务冲突处理机制

 MMoE：依赖门控网络的动态权重分配，但共享专家可能被多个任务争夺，导致跷跷板现象 （一个任务效果提升伴随另一任务下降）。  
 PLE：通过分层分离和参数隔离缓解冲突。底层允许共享与任务专属专家交互，高层逐步细化任务特定特征，实现更鲁棒的参数共享。

# 4. 门控网络的设计

MMoE：每个任务的门控网络仅基于原始输入特征计算权重，未考虑分层特征抽象。  
 PLE：门控网络在多层提取结构中工作，每一层的输入是前一层输出的抽象特征，从而学习更高级别的语义组合关系。

# 5. 适用场景与效果

 MMoE：适合任务相关性较弱的场景（如点击率与互动率预测），通过动态权重适配不同任务需求。  
 PLE：在任务相关性较强或冲突明显的场景（如电商中点击率与购买率）表现更优。实验表明，PLE 相比 MMoE 可显著提升多任务 AUC（例如腾讯实验中 PLE 对 3 个任务的 AUC 提升均超过 MMoE）。

总结对比表：  

<table><tr><td>特性</td><td>MMoE</td><td>PLE</td></tr><tr><td>专家类型</td><td>共享专家</td><td>共享专家 + 任务专属专家</td></tr><tr><td>门控机制</td><td>单层门控，基于原始输入特征</td><td>多层门控，基于分层抽象特征</td></tr><tr><td>任务冲突处理</td><td>动态调整权重，可能引发跷跷板</td><td>分层隔离参数，减少干扰</td></tr><tr><td>结构复杂度</td><td>单层专家组合</td><td>多层渐进式提取（多级CGC）</td></tr><tr><td>适用场景</td><td>任务相关性弱（如点击/互动）</td><td>任务相关性强或冲突明显</td></tr></table>

通过以上对比可以看出，PLE 通过更精细的专家分工和分层结构，在多任务复杂场景下实现了更强的鲁棒性，而 MMoE 更适合轻量级的多任务需求。实际应用中需根据任务相关性选择模型架构。

# 一、MMOE极化现象的原理

![](images/622c271d263ae3e1bb9f3b3017d164fc683c95434c52d4ba2db6167fec4d3989.jpg)

MMOE（Multi-gate Mixture-of-Experts）模型中的极化现象指在训练过程中，某些任务的门控网络（Gate）对专家网络（Expert）的权重分配出现极端分布，即某个专家权重接近 1，而其他专家权重接近 0。这种现象导致任务仅依赖单一专家网络，无法充分利用多专家模型的优势。具体原因如下：

# 1. 任务特异性与专家冗余

不同任务对底层特征的需求存在差异，若某些专家网络的特征表达能力显著优于其他专家，门控网络会通过梯度下降自动强化对优势专家的依赖，形成“赢者通吃”的局面。

# 2. 参数初始化与优化偏差

门控网络的权重初始化若存在偏差，叠加任务间的梯度冲突，会导致参数更新过程中某些专家权重被过度放大。例如，专家网络的初始权重差异可能通过 Softmax函数的指数放大效应加剧极化。

# 3. 模型容量与任务冲突

当专家数量过多或任务间差异较大时，模型可能因容量不足无法有效学习多专家协同机制，转而退化为单一专家模式以降低优化难度。

影响：极化现象会削弱 MMOE 的多任务协同能力，导致任务间干扰（负迁移）、泛化性能下降，且专家网络利用率低（部分专家未被激活）。

# 二、解决极化现象的方法

针对极化现象，可从模型设计、训练策略和后处理三方面进行优化：

# 1. 模型结构优化

#  门控网络复杂性增强

增加门控网络的层数或引入非线性激活函数（如 ReLU），提升其对任务差异性的建模能力。例如，将单层线性投影的门控网络改为两层 MLP，以捕捉更复杂的专家组合模式。

#  专家数量动态调整

根据任务相关性调整专家数量：对高冲突任务减少专家数量（如从 8个减至 4个），降低冗余；对低冲突任务增加专家数量以提升表达能力。

# 2. 训练策略改进

#  Dropout 正则化

在门控网络的 Softmax 输出前引入随机丢弃（如 $10 \%$ 概率 Mask 部分权重），强制模型分散对特定专家的依赖。Youtube实践表明，该方法可使专家利用率提升 $30 \%$ 。

#  权重约束与归一化

 L1/L2 正则化：对门控网络参数施加正则化惩罚，限制权重极端值。  
 Logit 缩放：对 Softmax 输入（Logit）进行归一化，例如将 Logit 除以最大值的平方根，缓解指数函数的放大效应。

# 3. 后处理与评估

#  专家贡献度监控

训练过程中统计各专家被门控网络选中的频率，若某专家长期未被激活（如频率 $\text{‰}$ ），可移除或重置其参数。

#  自适应权重融合

在推理阶段，对门控权重施加温度系数（Temperature Scaling），通过调整温度参数 τ控制权重分布的平滑度：

$$
w _ {i} = \frac {e ^ {z _ {i} / \tau}}{\sum_ {j = 1} ^ {N} e ^ {z _ {j} / \tau}}, \text {当} \tau > 1 \text {时 ， 权 重 分 布 更 均 匀 ；} \tau <   1 \text {时 更 尖 锐 。}
$$

# 三、实践建议

 任务相关性分析先行：使用任务间梯度相似性（如 GradNorm）评估任务冲突程度，高冲突任务组合需谨慎设计专家数量。  
 极化现象的双面性：若任务高度独立且存在显著优势专家，适度极化可能是合理选择，此时可减少专家数量以简化模型。  
 MMOE 极化现象的本质是任务需求与专家能力不匹配导致的模型退化。通过增强门控网络复杂性、引入随机丢弃和权重约束，可有效缓解极化问题。实际应用中需结合任务特性动态调整策略，平衡模型性能与计算效率。

# 一、模型解析与对比

# 1. PPNet（Parameter Personalized Network）

PPNet 主要针对多任务学习中的跷跷板效应 （不同任务目标相互冲突导致模型性能不平衡）。它通过动态调整 DNN 网络参数，实现用户粒度的任务个性化，缓解多任务稀疏性和依赖性问题。

创新点：

 参数级个性化：将用户 ID、物品 ID 等特征输入门控网络（Gate NU），生成动态权重作用于 DNN 每一层，实现参数动态选择。  
 梯度隔离：在训练时，Gate 网络对嵌入层（Embedding）的梯度进行隔离，避免干扰底层特征学习。

原理公式：

# 1.Gate NU 门控单元：

$$
g _ {t a s k} = \gamma \cdot \text {S i g m o i d} (R e L U (x W _ {1} + b _ {1}) W _ {2} + b _ {2})
$$

其中， 为输入特征（用户 ID/物品 ID 特征）， 为缩放因子，一般取 $\gamma = 2$ ，则门控单元的输出范围[0,2]。

# 2.DNN 参数调整：

$$
H ^ {(l + 1)} = f \left(\left(g _ {t a s k} ^ {(l)} \otimes H ^ {(l)}\right) W ^ {(l)} + b ^ {(l)}\right)
$$

$H ^ { ( l ) }$ 为第 层隐藏层输出， $\otimes$ 为逐元素乘法（哈达玛积）。

![](images/8b35769369cf4f221c713971e427e42b8ea826a36b800f49ffc0a2bc79241682.jpg)  
PPNet结构图

# 2. EPNet（Embedding Personalized Network）

EPNet 针对多场景学习中的场景跷跷板效应 （不同场景特征分布差异导致的模型偏差）。它通过场景特征动态调整嵌入层，实现跨场景特征对齐。

创新点：

Embedding 级个性化：以场景 ID、场景统计特征为输入，生成场景门控权重，筛选重要特征嵌入。  
特征增强机制：通过缩放因子 $\gamma = 2$ 增强场景信号，强化与当前场景相关的特征。

# 原理公式：

# 1. 场景门控生成：

$$
g _ {t a s k} = \gamma \cdot S i g m o i d (R e L U ([ E (F _ {s c e n e}) ] [ E (F _ {s c e n e - s t a t}) ] W _ {1} + b _ {1}) W _ {2} + b _ {2})
$$

$F _ { s c e n e }$ 为场景 ID， $F _ { s c e n e \_ s t a t }$ 为场景统计特征。

# 2. 嵌入调整：

$$
O _ {e p} = g _ {\text {d o m a i n}} \otimes E (F _ {\text {g e n e r a l}})
$$

为通用特征 Embedding，通过门控网络调整 embeding 后用于后续网络。

# 3. PEPNet（Parameter and Embedding Personalized Network）

PEPNet 同时解决多任务与多场景的双重跷跷板问题 （即任务冲突与场景分布差异），实现全局个性化建模。

原理：整体结构为 EPNet 与 PPNet 的级联，具体结构如下图。最终任务塔输出为多场景多任务的联合预测。

# 创新点：

 分层个性化：EPNet 处理场景特征对齐，PPNet 处理任务参数调整，形成端到端联合优化。  
 工程优化策略：包括特征淘汰机制、嵌入与 DNN 分层优化（AdaGrad vs Adam）、在线学习同步策略。

![](images/52d0f9a7de4bed2367a756fcedf9113f0776f669d484c7c90ff753e7f568e67d.jpg)

# 二、综合对比

<table><tr><td>维度</td><td>PPNet</td><td>EPNet</td><td>PEPNet</td></tr><tr><td>核心目标</td><td>多任务个性化参数调整</td><td>多场景个性化嵌入调整</td><td>多场景+多任务联合优化</td></tr><tr><td>输入特征</td><td>用户/物品ID、行为特征</td><td>场景ID、场景统计特征</td><td>用户、物品、场景特征</td></tr><tr><td>作用层级</td><td>DNN隐藏层参数</td><td>嵌入层(Embedding)</td><td>嵌入层+DNN参数</td></tr><tr><td>门控机制</td><td>用户粒度的任务门控</td><td>场景粒度的嵌入门控</td><td>场景+任务双层门控</td></tr><tr><td>主要创新</td><td>动态参数选择，梯度隔离</td><td>场景特征对齐，信号增强</td><td>分层个性化与工程优化</td></tr><tr><td>适用场景</td><td>多任务推荐（如点击/转化）</td><td>多场景推荐（如首页/搜索）</td><td>复杂多场景多任务联合建模</td></tr></table>

# 三、关键差异与选择建议

# 1. PPNet vs EPNet：

 PPNet 侧重任务粒度的参数动态化，适合目标稀疏但用户行为差异大的场景（如电商点击/加购/购买）。  
.  EPNet 侧重场景粒度的特征对齐，适合页面布局或用户意图差异大的场景（如短视频的推荐/朋友页）。

# 2. PEPNet 的优势：

 通过分层门控机制，同时捕捉场景共性与任务依赖性，例如在快手短视频推荐中，EPNet 解决“推荐页”与“朋友页”的特征分布差异，PPNet 解决“点赞”与“关注”的任务冲突。  
三者关系可概括为： EPNet 和 PPNet 是PEPNet 的核心组件，分别从 Embedding 层和参数层注入个性化先验，而 PEPNet 通过联合优化实现多场景多任务的全局最优。实际应用中，若仅需解决单一问题（如仅多任务或多场景），可独立使用 PPNet 或 EPNet；若需综合优化，PEPNet 是更优选择。

# Loss 权重平衡代码实现

```python
import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """基于不确定性的自适应权重"""
    def __init__(self, n_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total


class DynamicWeightAveraging:
    """动态加权平均 DWA"""
    def __init__(self, n_tasks, temperature=2.0):
        self.n_tasks = n_tasks
        self.temperature = temperature
        self.prev_losses = None

    def get_weights(self, current_losses):
        if self.prev_losses is None:
            self.prev_losses = current_losses
            return [1.0 / self.n_tasks] * self.n_tasks
        ratios = [c / (p + 1e-8) for c, p in zip(current_losses, self.prev_losses)]
        exp_ratios = [torch.exp(r / self.temperature) for r in ratios]
        total = sum(exp_ratios)
        weights = [e / total for e in exp_ratios]
        self.prev_losses = current_losses
        return [w.item() for w in weights]


class GradNormController:
    """梯度标准化 GradNorm 权重控制器"""
    def __init__(self, n_tasks, alpha=1.5):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.initial_losses = None

    def compute_grad_loss(self, shared_params, losses, task_grads):
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in losses]
        loss_ratios = [l.item() / (il + 1e-8) for l, il in zip(losses, self.initial_losses)]
        avg_ratio = sum(loss_ratios) / len(loss_ratios)
        inv_rates = [r / (avg_ratio + 1e-8) for r in loss_ratios]
        grad_norms = [sum(g.norm() ** 2 for g in tg) ** 0.5 for tg in task_grads]
        avg_gn = sum(grad_norms) / len(grad_norms)
        targets = [avg_gn * (ir ** self.alpha) for ir in inv_rates]
        grad_loss = sum(abs(gn - tgt) for gn, tgt in zip(grad_norms, targets))
        return grad_loss


def demo_weight_methods():
    """演示不同权重策略的效果"""
    n_tasks = 2
    losses = [torch.tensor(0.5, requires_grad=True), torch.tensor(2.0, requires_grad=True)]

    uw = UncertaintyWeighting(n_tasks)
    print(f"Uncertainty Loss: {uw(losses).item():.4f}")

    dwa = DynamicWeightAveraging(n_tasks, temperature=2.0)
    w = dwa.get_weights(losses)
    print(f"DWA 权重: {[round(x, 3) for x in w]}")

    total_dwa = sum(wi * li.item() for wi, li in zip(w, losses))
    print(f"DWA 加权 Loss: {total_dwa:.4f}")


if __name__ == "__main__":
    demo_weight_methods()
```

## 常见问题与易错点

1. **不确定性加权的 log_var 初始化**：初始化为 0 意味着初始权重相等，若任务量级差异大可手动设置初始值
2. **DWA 的温度参数**：T 过大则权重趋于均匀（失去调节效果），T 过小则权重过于尖锐，通常取 1.0-2.0
3. **GradNorm 的 α 参数**：α=0 表示不调整（所有任务目标梯度相同），α 越大越强调逆学习速率的任务
4. **固定权重的陷阱**：CTR 和 CVR 的 Loss 量级可能差 10 倍，固定权重必须归一化后再设置

## 学习总结

多任务 Loss 权重平衡的核心矛盾是"任务竞争"：不同任务对共享参数的梯度方向和量级不一致。固定权重最简单但不灵活；Uncertainty Weighting 通过可学习参数自动平衡量级；DWA 根据学习速度动态调节；GradNorm 直接在梯度层面平衡。实践中推荐从 Uncertainty Weighting 起步（实现简单、效果好），任务冲突严重时升级到 GradNorm。

# 4.5 因果推断与 Uplift

