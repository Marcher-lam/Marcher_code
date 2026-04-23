# 面试题：DCN 和 DCN-v2 的原理与区别

# 面试题：DCN 和 DCN-v2 的原理与区别

以下是 DCN（Deep & Cross Network）与 DCN-v2 模型的原理详解及对比分析：

# 一、 DCN 模型原理

论文地址：Deep & Cross Network for Ad Click Predictions

核心思想：DCN 模型通过显式交叉网络（Cross Network） 与深度网络（Deep Network） 结合，实现特征的高阶交叉和非线性学习，主要用于推荐系统的点击率预测（CTR）任务。

![](images/e69333a3c19143fb21d82c64b755d961ca23ad63baea552999c1e1a0dfa97843.jpg)  
Figure 1: The Deep & Cross Network

# 1. 交叉网络（Cross Network）

 数学公式：第 $l + 1$ 层的交叉计算为： $x _ { l + 1 } = x _ { 0 } \odot \left( w _ { l } \cdot x _ { l } \right) + x _ { l } + b _ { l }$

其中 $x _ { 0 }$ 是初始输入特征向量， $w _ { l }$ 是权重向量，⋅ 表示逐元素相乘。通过逐层叠加，交叉网络可显式构造最高 $l + 1$ 阶的特征交叉。

 特点：

 参数高效：每层仅增加 $d \times 2$ 参数（ $^ d$ 为特征维度）。  
显式特征交互：通过外积实现特征交叉，避免人工特征工程。

# 2. 深度网络（Deep Network）

由多层全连接层（MLP）构成，学习非线性特征组合，与交叉网络并行或串行输出结果。

# 3. 优势

 结合显式高阶交叉与隐式深度学习，适用于稀疏特征场景。  
 相比传统 Wide&Deep 模型，交叉网络更高效地捕捉特征交互。

# 二、DCN-v2 模型原理

核心改进：DCN-v2 在 DCN 基础上通过矩阵化交叉权重、低秩分解和 MoE（混合专家）结构提升表达能力与效率。

![](images/ce61d0bf6ffbf414a4447f31f554475723e490013ccc04babbf040eca3289da6.jpg)  
(a) Stacked

![](images/46fe9df535c85ea90daf0d2759dde48cfea5f72fd1d7d4c193488bfc18aaa16c.jpg)  
(b) Parallel

# 1. 交叉网络改进

 矩阵化权重：将权重向量 扩展为矩阵 $W _ { l } \in \mathbb { R } ^ { d \times d }$ ，增强特征交叉的灵活性和表达能力。公式更新为：

$$
x _ {l + 1} = x _ {0} \odot \left(W _ {l} \cdot x _ {l}\right) + x _ {l} + b _ {l}
$$

 低秩分解：对矩阵 $W _ { l }$ 进行低秩分解（如 $W _ { l } = U _ { l } V _ { l } ^ { T }$ ，其中 $U _ { l } , V _ { l } \in \mathbb { R } ^ { d \times r }$ ），减少参数量同时保持性能。

# 2. 引入 MoE 结构

使用多个专家（Experts）学习不同子空间的特征交叉，公式为： $\boldsymbol { x } _ { l + 1 } = \sum _ { i = 1 } ^ { K } G _ { i } ( \boldsymbol { x } _ { l } ) \times U _ { l , i } ( V _ { l , i } ^ { T } \boldsymbol { x } _ { l } ) \odot \boldsymbol { x } _ { 0 }$

其中 $G _ { i } ( x _ { l } )$ 为门控函数，动态分配不同专家的权重，提升模型对不同交叉模式的适应性。

# 3. 模型组合方式

 并行结构：交叉网络与深度网络并行输出（类似 DCN-v1）。  
 堆叠结构 （Stacking）：交叉网络输出作为深度网络的输入，实现更深的特征融合。

# 4. 优势

 参数效率提升 $30 \%$ 以上，且效果优于 DCN。  
 在 Criteo 数据集上，AUC 提升 $0 . 5 \% { - } 1 \%$ 。

# 三、DCN 与 DCN-v2 的区别对比

<table><tr><td>维度</td><td>DCN</td><td>DCN-v2</td></tr><tr><td>交叉网络参数</td><td>权重为向量 (wl ∈ Rd)</td><td>权重为矩阵 (Wl ∈ Rd×d)</td></tr><tr><td>计算复杂度</td><td>低（每层O(d)）</td><td>高（矩阵运算O(d²)），但可通过低秩分解优化</td></tr><tr><td>特征交叉能力</td><td>显式但表达能力有限</td><td>支持子空间交叉与非线性变换，表达能力更强</td></tr><tr><td>模型结构</td><td>仅支持并行结构</td><td>新增堆叠结构（交叉网络→深度网络）</td></tr><tr><td>工业落地</td><td>适合中等规模数据</td><td>通过低秩和MoE支持超大规模数据（如十亿级样本）</td></tr></table>

# 四、Python 代码实现

```python
import torch
import torch.nn as nn

class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        return x0 * (xl @ self.w.unsqueeze(1)).squeeze(1) + xl + self.b

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl

class CrossLayerV2(nn.Module):
    def __init__(self, input_dim, low_rank=None):
        super().__init__()
        self.low_rank = low_rank
        if low_rank is not None:
            self.U = nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
            self.V = nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
        else:
            self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        if self.low_rank is not None:
            weight = self.U @ self.V.T
        else:
            weight = self.W
        return x0 * (xl @ weight) + xl + self.b

class DCN(nn.Module):
    def __init__(self, num_features, embedding_dims, hidden_dims=[256, 128], num_cross_layers=3):
        super().__init__()
        total_dim = num_features * embedding_dims
        self.cross_net = CrossNetwork(total_dim, num_cross_layers)
        deep_layers = []
        prev_dim = total_dim
        for h in hidden_dims:
            deep_layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        self.deep_net = nn.Sequential(*deep_layers)
        self.output = nn.Linear(total_dim + hidden_dims[-1], 1)

    def forward(self, x):
        cross_out = self.cross_net(x)
        deep_out = self.deep_net(x)
        combined = torch.cat([cross_out, deep_out], dim=-1)
        return torch.sigmoid(self.output(combined))

class DCNv2(nn.Module):
    def __init__(self, num_features, embedding_dims, hidden_dims=[256, 128], num_cross_layers=3, low_rank=64):
        super().__init__()
        total_dim = num_features * embedding_dims
        self.cross_layers = nn.ModuleList([
            CrossLayerV2(total_dim, low_rank=low_rank) for _ in range(num_cross_layers)
        ])
        deep_layers = []
        prev_dim = total_dim
        for h in hidden_dims:
            deep_layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        self.deep_net = nn.Sequential(*deep_layers)
        self.output = nn.Linear(total_dim + hidden_dims[-1], 1)

    def forward(self, x):
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        cross_out = xl
        deep_out = self.deep_net(x)
        combined = torch.cat([cross_out, deep_out], dim=-1)
        return torch.sigmoid(self.output(combined))

batch_size, num_fields, emb_dim = 32, 10, 8
x = torch.randn(batch_size, num_fields * emb_dim)

dcn = DCN(num_fields, emb_dim)
dcnv2 = DCNv2(num_fields, emb_dim)

out_dcn = dcn(x)
out_dcnv2 = dcnv2(x)

dcn_params = sum(p.numel() for p in dcn.parameters())
dcnv2_params = sum(p.numel() for p in dcnv2.parameters())

print(f"DCN参数量: {dcn_params:,}")
print(f"DCN-v2参数量: {dcnv2_params:,}")
print(f"DCN输出: {out_dcn[:4].detach().numpy().flatten().round(4)}")
print(f"DCN-v2输出: {out_dcnv2[:4].detach().numpy().flatten().round(4)}")
```

# 五、与其他CTR模型对比

<table><tr><td>模型</td><td>特征交叉方式</td><td>交叉阶数</td><td>参数效率</td><td>适用场景</td></tr><tr><td>FM</td><td>隐向量内积</td><td>二阶</td><td>高</td><td>中小规模稀疏数据</td></tr><tr><td>DeepFM</td><td>FM+DNN并行</td><td>二阶+高阶隐式</td><td>中</td><td>通用CTR预估</td></tr><tr><td>DCN</td><td>向量级显式交叉</td><td>有界高阶</td><td>高</td><td>中等规模特征交叉</td></tr><tr><td>DCN-v2</td><td>矩阵级显式交叉+MoE</td><td>任意高阶</td><td>中（低秩可优化）</td><td>大规模工业推荐</td></tr><tr><td>xDeepFM</td><td>CIN压缩交互网络</td><td>向量级高阶</td><td>中低</td><td>精细特征交互</td></tr></table>

标题：Wukong: Towards a Scaling Law for Large-Scale Recommendation

链接：https://arxiv.org/pdf/2403.02545.pdf

单位：Meta 公司

会议：ICML2024

Meta 公司的 Wukong 模型是一种针对大规模推荐系统设计的深度学习架构，旨在解决传统推荐模型缺乏缩放定律（ScalingLaw）的问题。

# 一、核心原理

Wukong 通过 Dense 扩展（Dense Scaling）而非传统推荐模型的稀疏扩展（如扩大嵌入表），结合高阶特征交叉和结构化堆叠，首次在推荐领域实现了模型效果与复杂度的正相关缩放规律。

# 1 特征交互机制

 因子分解机块（FMB）：堆叠多层 FM 模块，显式捕获特征间二阶交互，并通过 MLP 转换为高阶交叉（如三阶、四阶）。  
 线性压缩块（LCB）：线性重组输入特征，保留当前阶数交叉信息，避免信息丢失。  
 残差连接与层归一化：稳定训练过程，缓解梯度消失问题。

# 2 缩放定律设计

 分层扩展策略：优先增加交互堆叠层数（捕获更高阶交叉），再扩展嵌入数量、MLP宽度等参数，确保模型容量与效果同步提升。  
 低秩分解优化：通过矩阵降维（如将 FM 的 $n { \times } n$ 输出压缩为 $n { \times } k$ ，k⋅n）降低计算复杂度。

# 二、实现方法

# 1 模型结构

Wukong 由三部分组成：

 嵌入层（Embedding Layer）：根据特征重要性分配动态维度（如重要特征分配更多维度），通过池化聚合。  
 交互堆叠（Interaction Stack）：多层"Wukong Layer"串联，每层包含并行的 FMB 和 LCB 模块，输出拼接后经残差连接传递至下一层。  
 MLP 预测层：将交互结果映射为最终预测值（如点击率）。

![](images/a45cfb674ea4abe4271b2ee6d68f773e854e15475bbd334b167652ebd9a37fab.jpg)

# 2. 关键技术细节

 因子分解机模块 FMB计算流程：

$$
F M B (X) = M L P \left(L a y e r N o r m \left(F l a t t e n (F M (X))\right)\right)
$$

其中 FM 模块实现特征间两两交叉，MLP 提升非线性表达能力。

 线性压缩模块 LCB作用：通过权重矩阵W 压缩特征维度（如X⋅W），保留当前阶数信息。  
 自适应训练：嵌入层使用 Rowwise Adagrad 优化器，Dense 层使用 Adam，支持千亿级参数训练。

# 三、解决的问题

# 1 传统推荐模型缺乏 Scaling Raw

此前推荐模型（如 DLRM、DCNv2）仅通过扩大嵌入表参数（稀疏扩展）提升效果，但参数增长与效果提升不成正比。Wukong 通过密集扩展实现两个数量级的缩放定律（计算量每翻两番，效果提升 $0 . 1 \%$ ）。

# 2 高阶特征交互不足

传统模型依赖 MLP 隐式学习交叉特征，而 Wukong 通过显式堆叠 FM 模块捕获任意阶交互（实验显示高阶交叉对复杂任务至关重要）。

# 3 计算效率与硬件适配

 低秩分解技术将 FM 复杂度从 O(n2)降至 $O ( n k )$ ，适配 GPU 并行计算。  
 残差结构减少训练波动，支持千卡级分布式训练（如使用 128-256 块 H100 GPU）。

# 四、实际效果

 公开数据集：在 Frappe、MovieLens 等 6 个数据集上，AUC 提升 $0 . 5 \% { - 2 . 3 \% }$ ，显著优于 $\mathsf { A F N + }$ 、xDeepFM 等基线模型。  
 Meta 内部场景：在 1460 亿条目的广告推荐任务中，训练计算量从 1 GFLOP/example 扩展至 100 GFLOP/example（相当于 GPT-3规模），效果持续提升且未饱和。

# 六、DCN系列模型选型建议

1. 小规模场景（特征维度<100，样本<千万）：优先DCN-v1，参数少、训练快
2. 中大规模场景（特征维度100-1000，样本亿级）：DCN-v2并行结构+低秩分解
3. 超大规模场景（千亿样本）：DCN-v2堆叠结构+MoE，参考Wukong的缩放策略
4. 工程实践Tips：低秩分解rank通常取特征维度的1/4到1/8；MoE专家数取4-8个效果最佳

# 七、常见问题与易错点

1. 误区："交叉层数越多越好"
   - 实际：DCN中交叉层存在退化问题（高阶层趋近线性），3-6层即可。DCN-v2通过矩阵权重缓解了此问题。

2. 误区："DCN-v2一定比DCN好"
   - 实际：在小数据集上，DCN-v2的矩阵参数容易过拟合，此时DCN的向量参数反而更稳定。需要根据数据规模选择。

3. 误区："低秩分解会严重损失模型效果"
   - 实际：实验表明rank=64时，效果与全矩阵几乎无差异，但参数量减少90%以上。关键在于交叉矩阵本身的低秩特性。
