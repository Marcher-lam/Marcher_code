# 面试题：多模态 Embedding 语义 ID 编码方法介绍

# 面试题：多模态 Embedding 语义 ID 编码方法介绍

多模态 Embedding 语义 ID 编码的业界主流方法介绍如下：

# 一、 残差量化变分自编码器（RQ-VAE）

link：Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID

# 1. 核心原理

![](images/240d2e806bfc57a3f6f45feb129156fb2e5f1c16c16276b01d06dfe80712f1d7.jpg)  
Figure1 The RQVAE model with $L = 3$

通过分层向量量化将连续 Embedding 映射为离散语义ID序列，解决高基数 ID的嵌入不稳定问题

 输入：广告多模态 Embedding $\boldsymbol { x } \in \mathbb { R } ^ { d }$ （由文本/视觉模型生成）  
 分层量化：

$$
c _ {1} = \arg \min  _ {k \in [ 1, K ]} \| x - e _ {k} ^ {(1)} \|
$$

 第 1 层：

$$
r _ {l} = r _ {l - 1} - e _ {c _ {l - 1}} ^ {(l - 1)}, c _ {l} = \arg \min  _ {k} \| r _ {l} - e _ {k} ^ {(l)} \|
$$

 第 层残差：

 输出语义 ID： $S = ( c _ { 1 } , c _ { 2 } , \ldots , c _ { L } )$ ，其中 $L$ 为量化层数（比如 $L { = } 6$ ， $\scriptstyle K = 2 0 4 8$

# 2. 训练目标

$$
\mathcal {L} = \underbrace {\| x - \operatorname {D e c o d e r} (S) \| ^ {2}} _ {\text {重 建 损 失}} + \lambda \underbrace {\sum_ {l = 1} ^ {L} \| \mathrm {s g} [ r _ {l} ] - e _ {c _ {l}} ^ {(l)} \| ^ {2}} _ {\text {承 诺 损 失}} + \gamma \underbrace {\sum_ {l = 1} ^ {L} \| r _ {l} - \mathrm {s g} [ e _ {c _ {l}} ^ {(l)} ] \| ^ {2}} _ {\text {码 本 c o d e b o o k 损 失}}
$$

其中 sg[⋅] 为梯度截断操作， 为超参数。

# 3. 工业应用

Meta 广告系统：将广告文本 $^ { + }$ 视觉 Embedding 输入 RQ-VAE，生成 6 层语义 ID，在线服务时通过前缀组合映射到嵌入表，新广告 NDCG@100 提升 $0 . 3 3 \%$ ，长尾广告点击率方差降低 $4 3 \%$

# 二、 SentencePiece 动态子词编码（SPM-based）

Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations

# 1. 核心原理

将语义 ID 序列视为特殊语料，利用 BPE（Byte Pair Encoding）算法动态构建子词词表：

 输入：RQ-VAE 生成的语义 ID 序列 $S = ( c _ { 1 } , c _ { 2 } , \ldots , c _ { L } )$   
 合并策略：迭代合并最高频共现的 ID 对，直至词表大小 $V$ 达到预设值  
 输出：子词 ID 序列 $T = ( t _ { 1 } , t _ { 2 } , \dots , t _ { M } )$ ，其中 $M \ll L$

# 2. 数学表达

$\operatorname* { m a x } _ { V } \sum _ { ( t _ { i } , t _ { j } ) \in V } \mathrm { f r e q } ( t _ { i } , t _ { j } ) \cdot \mathbb { I } _ { ( t _ { i } , t _ { j } ) \in \mathrm { m e r g e } }$ freq(ti,tj)·I(t,t)merge词表构建目标函数： ，其中 为共现频率， 为指示函数。

# 3. 优势

 动态长度适配：高频语义 ID 组合被压缩为单一子词（如"手机-游戏"→单一 token）

# 三、快手 RQ-Kmeans

RQ-Kmeans 是快手 OneRec 针对海量物品高维多模态 embedding 设计的分层残差量化聚类方法，核心通过多层残差迭代量化 $^ +$ 平衡K-means将高维向量转化为分层离散语义ID，实现粗 细的语义空间建模。

# 核心算法流程

#  Step1 训练阶段（构建分层码本）

初始化 embedding 为初始残差，逐层对残差执行平衡 K-means（保证各簇样本量均衡，避免码本浪费），得到每层聚类中心（码本）；用当前层码本量化残差，计算新残差传递至下一层，直至完成所有层数训练，输出分层码本。

#  Step2 编码阶段（生成语义 ID）

以物品 embedding 为初始残差，逐层匹配对应层码本的最近聚类中心，记录中心索引作为语义 ID 片段；更新残差后进入下一层，最终拼接各层索引得到分层语义 ID 序列（前缀为粗语义，后缀为细语义）。

#  Step3 解码阶段（重建 embedding）

根据语义 ID 序列，从各层码本中提取对应聚类中心，求和得到量化后的重建 embedding，用于相似度计算或检索。

# 关键优化亮点

 平衡 K-means：解决普通 K-means 簇分布不均问题，提升码本利用率和检索效率；  
 分层残差：逐层拟合上一层量化残差，降低整体量化误差，同时保留粗 细语义结构，相似物品共享 ID 前缀；  
 轻量高效：无复杂模型训练，仅通过聚类 $^ +$ 残差迭代实现，适配十亿级物品的大规模工程化落地。

# 四、双通道级联表示（COBRA 框架）

# 1. 核心架构

百度提出融合语义 ID 与原始 Embedding 的级联表示

 输入：广告多模态 Embedding $_ x$   
 语义 ID 分支： $S = \mathrm { { R Q - V A E } } ( x ) _ {  }$ 嵌入向量 $e _ { s }$

 稠密向量分支：可训练编码器 $e _ { d } = \operatorname { E n c o d e r } ( x )$   
 级联输出： $e = [ e _ { s } ; e _ { d } ] \in \mathbb { R } ^ { d _ { s } + d _ { d } }$

# 2. 训练目标

$$
\mathcal {L} = \alpha \underbrace {\operatorname {C r o s s E n t r o p y} (\operatorname {D e c o d e r} (e) , S)} + \beta \quad \underbrace {\| x - e _ {d} \| ^ {2}}
$$

双任务联合优化：

ID重建损失

Embedding对齐损失

# 3. 工业效果

百度信息流广告：CVR 提升 $3 . 6 \%$ ，嵌入空间聚类紧密度提升 $4 1 \%$ 。推理速度比纯稠密模型快 3.2 倍（因语义 ID 提供先验筛选）。

# 五、四种方法综合对比

<table><tr><td>对比维度</td><td>RQ-VAE</td><td>SPM-based</td><td>RQ-Kmeans</td><td>COBRA</td></tr><tr><td>核心思路</td><td>分层残差向量量化</td><td>BPE子词压缩</td><td>平衡K-means聚类</td><td>语义ID+稠密向量双通道</td></tr><tr><td>训练复杂度</td><td>高（需训练VAE）</td><td>低（统计方法）</td><td>中（仅聚类）</td><td>高（双分支训练）</td></tr><tr><td>语义保留</td><td>强（分层量化）</td><td>中（频率驱动）</td><td>强（聚类保持相似性）</td><td>最强（双通道互补）</td></tr><tr><td>推理速度</td><td>快（查表即可）</td><td>快（更短序列）</td><td>快（聚类查表）</td><td>中（双分支计算）</td></tr><tr><td>新物品适应</td><td>需增量训练</td><td>需更新词表</td><td>可实时编码</td><td>可实时编码</td></tr><tr><td>代表公司</td><td>Meta</td><td>Google</td><td>快手</td><td>百度</td></tr><tr><td>适用场景</td><td>大规模广告推荐</td><td>语义ID序列压缩</td><td>十亿级物品语义ID</td><td>需要高精度的场景</td></tr></table>

# 六、Python 代码实现：RQ-VAE 简化版

```python
import torch
import torch.nn as nn
import numpy as np

class Codebook(nn.Module):
    def __init__(self, num_codes=256, code_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z):
        dist = torch.cdist(z.unsqueeze(1), self.embedding.weight.unsqueeze(0))
        indices = dist.argmin(dim=-1)
        z_q = self.embedding(indices).squeeze(1)
        return z_q, indices

class SimpleRQVAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_codes=256, num_layers=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.codebooks = nn.ModuleList([
            Codebook(num_codes, hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.num_layers = num_layers

    def forward(self, x):
        z = self.encoder(x)
        residual = z
        z_q_total = torch.zeros_like(z)
        indices_list = []
        for codebook in self.codebooks:
            z_q, idx = codebook(residual)
            z_q_total = z_q_total + z_q
            indices_list.append(idx)
            residual = residual - z_q.detach()
        x_recon = self.decoder(z_q_total)
        return x_recon, z_q_total, z, indices_list

    def encode(self, x):
        z = self.encoder(x)
        residual = z
        indices_list = []
        for codebook in self.codebooks:
            z_q, idx = codebook(residual)
            indices_list.append(idx)
            residual = residual - z_q.detach()
        return torch.stack(indices_list, dim=-1)

model = SimpleRQVAE(input_dim=32, hidden_dim=16, num_codes=64, num_layers=3)
x = torch.randn(8, 32)
x_recon, z_q, z, indices = model(x)
semantic_ids = model.encode(x)
print(f"输入形状: {x.shape}")
print(f"重建形状: {x_recon.shape}")
print(f"语义ID形状: {semantic_ids.shape}")
print(f"语义ID示例（第1个样本）: {semantic_ids[0].tolist()}")
```

# 一些 Trick：

1、跨模态对齐增强：在RQ-VAE输入前加入 CLIP式对比损失：

$$
\mathcal {L} _ {\text {a l i g n}} = - \log \frac {\exp (\sin (x _ {\text {t e x t}} , x _ {\text {i m a g e}}) / \tau)}{\sum_ {j} \exp (\sin (x _ {\text {t e x t}} , x _ {j}) / \tau)}
$$

2、动态码本更新：每 24 小时用新广告 Embedding 增量训练 RQ-VAE，解决广告内容频繁修改问题；  
3、图结构编码：将用户-广告交互建模为异构图，语义 ID作为节点属性注入 GNN。

# 七、常见问题与面试追问

1. Q: 语义 ID 编码相比传统 ID Embedding 的核心优势是什么？
A: 传统ID Embedding对新物品冷启动困难，且Embedding空间随物品数量线性增长。语义ID基于内容生成，新物品可直接编码；且通过分层码本实现O(1)查找，存储效率高数个量级。

2. Q: RQ-VAE 的量化误差如何控制？
A: 三层机制：(1) 残差迭代——每层拟合上一层的量化残差；(2) 码本学习——通过commitment loss和codebook loss联合优化码本向量；(3) 增加量化层数L——层数越多，重建精度越高，但推理开销也增大。

3. Q: 为什么快手选择RQ-Kmeans而不是RQ-VAE？
A: RQ-Kmeans无需训练VAE解码器，仅通过聚类即可构建码本，工程实现更简单，适合十亿级物品的快速迭代。代价是重建精度略低于RQ-VAE，但在大规模场景下性价比更优。
