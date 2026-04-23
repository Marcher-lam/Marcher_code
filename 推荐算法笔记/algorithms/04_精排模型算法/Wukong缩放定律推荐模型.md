# Meta Wukong：面向大规模推荐的缩放定律模型

## 1. 算法基础认知

Wukong（悟空）是 Meta 在 ICML 2024 发表的推荐系统基础模型，**首次在推荐领域验证了缩放定律（Scaling Law）**：即模型效果随计算量增加而持续提升，且未见饱和。Wukong 通过 Dense 扩展策略（非传统稀疏扩展如扩大 Embedding 表），结合高阶特征交叉和结构化堆叠，在 Meta 内部 1460 亿条目的广告推荐系统中取得显著效果。

## 2. 详细原理

### 2.1 推荐系统中的缩放定律

NLP 领域已验证模型规模与效果的幂律关系：

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha}$$

但推荐系统具有特殊性：
- **特征稀疏性**：大量类别特征，Embedding 表占据大部分参数
- **特征交互**：推荐依赖高阶特征交叉，非单纯的序列建模
- **延迟约束**：在线推理延迟严格受限（<100ms）

Wukong 证明：通过合理的架构设计，推荐模型同样遵循缩放定律，且**Dense 层的扩展比 Embedding 表的扩展更高效**。

### 2.2 核心架构

Wukong 由三种基本模块组成：

**1. 因子分解机块（FMB）**

堆叠多层 FM 模块，显式捕获二阶交互，通过 MLP 提升高阶交叉：

$$FMB(X) = MLP(LayerNorm(Flatten(FM(X))))$$

FM 部分计算二阶交叉：

$$FM(X) = \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j \rangle \cdot x_i \cdot x_j$$

**2. 线性压缩块（LCB）**

线性重组输入特征，保留当前阶数信息并传递到下一层：

$$LCB(X) = X \cdot W$$

**3. 交互块（Interaction Block）**

将 FMB 和 LCB 组合：

$$IB(X) = Concat(FMB(X), LCB(X))$$

### 2.3 缩放策略

Wukong 的缩放遵循明确优先级：

1. **优先增加交互堆叠层数**（深度扩展）
2. **其次扩展 Embedding 维度**（宽度扩展）
3. **最后增加 MLP 隐藏层宽度**

这种优先级来自实验观察：深度扩展的收益最大且持续时间最长。

### 2.4 低秩分解

FM 的原始输出维度为 $O(n^2)$（n 为特征数），Wukong 通过低秩分解压缩为 $O(nk)$：

$$\langle v_i, v_j \rangle \approx u_i^T V_j$$

其中 $u_i \in \mathbb{R}^k$, $k \ll n$。

## 3. 数学推导

### 3.1 缩放定律验证

Wukong 实验验证了推荐模型的计算量-效果关系：

$$AUC(C) = AUC_{\infty} - \alpha \cdot C^{-\beta}$$

其中 $C$ 为 FLOPs，$AUC_{\infty}$ 为理论极限。

在 Meta 内部数据集上，从 1 GFLOP 扩展到 100 GFLOP（接近 GPT-3 规模），AUC 持续提升，未观察到饱和。

### 3.2 FMB 的计算复杂度

设 n 个特征，每个维度 d，低秩维度 k：

$$O(FMB) = O(n \cdot k \cdot d + n \cdot d_{mlp})$$

相比 DCN-v2 的 $O(n^2 \cdot d)$，大幅降低。

### 3.3 梯度流分析

LCB 提供类似 ResNet 的恒等映射路径：

$$X_{l+1} = IB(X_l) = Concat(FMB(X_l), LCB(X_l))$$

LCB 的线性连接确保梯度可以无衰减地流向浅层，缓解深层网络的梯度消失。

## 4. 训练过程

1. 将稀疏特征通过 Embedding 表映射为稠密向量
2. 拼接所有稠密特征形成输入 X
3. 依次通过 L 个交互块（IB），每层捕获更高阶的交互
4. 最终输出经过 MLP 映射为预测分数
5. 使用 Binary Cross-Entropy 损失训练

**训练优化**：
- 大规模分布式训练（数据并行 + 模型并行）
- 混合精度训练（FP16 + FP32）
- 学习率 Warmup + Cosine Decay

## 5. 应用场景

| 场景 | 规模 | 效果 |
|------|------|------|
| Meta 广告推荐 | 1460 亿条目 | AUC 持续提升 |
| 公开数据集（Frappe, ML等） | 中等 | AUC +0.5%~2.3% |
| 电商推荐 | 大规模 | 适合工业部署 |
| 内容推荐 | 大规模 | 通用架构 |

## 6. 优缺点分析

**优点**：
- 首次验证推荐领域的缩放定律
- Dense 扩展比稀疏扩展更高效
- 结构化堆叠，计算可控
- 低秩分解降低计算复杂度
- LCB 残差连接保证深层网络可训练

**缺点**：
- 需要大规模计算资源（100 GFLOP 级别）
- 缩放策略的优先级可能因数据集而异
- 对中小规模数据集可能过拟合
- 工程实现复杂度高

## 7. 与相关方法对比

| 方法 | 缩放方式 | 交互阶数 | 计算效率 | 缩放定律 |
|------|---------|---------|---------|---------|
| DCN-v2 | 增加交叉层 | 显式有界 | 中 | 未验证 |
| DeepFM | 增加MLP | FM+隐式 | 高 | 未验证 |
| DLRM | 增加Embedding | 底层交互 | 高 | 未验证 |
| Wukong | Dense全栈扩展 | 显式高阶 | 中 | 已验证 |

## 8. PyTorch 代码实现

```python
import torch
import torch.nn as nn

class FMBlock(nn.Module):
    def __init__(self, num_fields, embed_dim, latent_dim=16):
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.W = nn.Linear(embed_dim, latent_dim, bias=False)
    
    def forward(self, x):
        B = x.size(0)
        x_reshape = x.view(B, self.num_fields, self.embed_dim)
        x_proj = self.W(x_reshape)
        
        sum_sq = torch.sum(x_proj, dim=1) ** 2
        sq_sum = torch.sum(x_proj ** 2, dim=1)
        fm_out = 0.5 * (sum_sq - sq_sum)
        return fm_out

class FactorizationMachineBlock(nn.Module):
    def __init__(self, num_fields, embed_dim, latent_dim=16, mlp_dims=[128, 64]):
        super().__init__()
        self.fm = FMBlock(num_fields, embed_dim, latent_dim)
        fm_output_dim = latent_dim
        layers = []
        input_dim = fm_output_dim
        for hidden_dim in mlp_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(mlp_dims[-1])
    
    def forward(self, x):
        fm_out = self.fm(x)
        mlp_out = self.mlp(fm_out)
        return self.layer_norm(mlp_out)

class LinearCompressionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        return self.layer_norm(self.linear(x))

class InteractionBlock(nn.Module):
    def __init__(self, num_fields, embed_dim, latent_dim=16,
                 mlp_dims=[128, 64], output_dim=None):
        super().__init__()
        self.fmb = FactorizationMachineBlock(num_fields, embed_dim, latent_dim, mlp_dims)
        fmb_out_dim = mlp_dims[-1]
        output_dim = output_dim or embed_dim * num_fields
        self.lcb = LinearCompressionBlock(embed_dim * num_fields, output_dim)
        self.fmb_proj = nn.Linear(fmb_out_dim, output_dim)
    
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        fmb_out = self.fmb_proj(self.fmb(x))
        lcb_out = self.lcb(x_flat)
        return torch.cat([fmb_out, lcb_out], dim=-1)

class Wukong(nn.Module):
    def __init__(self, num_fields, num_embeddings, embed_dim=16,
                 num_blocks=4, latent_dim=16, mlp_dims=[128, 64]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embed_dim) 
            for _ in range(num_fields)
        ])
        
        ib_output_dim = embed_dim
        block_output = embed_dim * num_fields + ib_output_dim * num_fields
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(InteractionBlock(
                num_fields, embed_dim, latent_dim,
                mlp_dims, ib_output_dim * num_fields
            ))
        
        final_dim = ib_output_dim * num_fields * 2 * num_blocks
        self.head = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        emb_list = [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))]
        x_emb = torch.cat(emb_list, dim=-1)
        
        block_outputs = []
        current = x_emb
        for block in self.blocks:
            current = block(current)
            block_outputs.append(current)
        
        all_outputs = torch.cat(block_outputs, dim=-1)
        return self.head(all_outputs).squeeze(-1)
```

## 9. 可视化与结果理解

- **缩放曲线**：绘制 FLOPs vs AUC，验证幂律关系
- **各层特征交互热力图**：可视化不同层的特征交互强度
- **消融实验**：逐层移除交互块，观察 AUC 下降幅度

## 10. 常见问题与易错点

1. **FMB 的 FM 部分容易数值溢出**：需要 LayerNorm 或梯度裁剪
2. **堆叠层数选择**：不是越多越好，中小数据集 4-6 层为宜
3. **低秩维度 $k$ 的选择**：$k$ 过小丢失交互信息，建议 $k=16\sim64$
4. **分布式训练策略**：Embedding 表需要分片，FMB 的 MLP 需要跨卡同步
5. **推理延迟**：深层堆叠增加推理延迟，需配合模型蒸馏或早退机制

## 11. 学习总结

Wukong 的核心贡献在于验证了推荐模型的缩放定律，并给出了 Dense 扩展优于稀疏扩展的实证。其结构化设计（FMB+LCB 交互块）兼具理论优雅性和工程实用性。对推荐系统从业者的启示是：与其无脑扩大 Embedding 表，不如投入更多计算到高阶特征交叉层。

## 12. 学习路径建议

- **前置知识**：FM、DeepFM、DCN-v2、Scaling Law 基础
- **进阶方向**：DCN-v3、FinalMLP、推荐系统基础模型
- **推荐论文**：Wukong (ICML 2024)、DCN-v2 (WWW 2021)、Scaling Laws for Neural Language Models (2020)
