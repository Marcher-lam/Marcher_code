# 面试题：FFM 模型原理介绍

面试题：FFM 模型原理介绍

链接: https://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf

# 一、原理与核心思想

FFM（Field-aware Factorization Machine）是 FM（Factorization Machine）的改进版， 核心创新在于引入"Field（域）"概念，将特征按业务逻辑分组，使每个特征在与不同域的特征交互时使用不同的隐向量，从而提升特征交叉的精细度。

 FM 的局限性：FM 中每个特征仅有一个隐向量，无法区分不同域特征交互的差异（如"用户年龄"与"电影类型"的交互和"用户年龄"与"价格"的交互使用相同隐向量）。  
 稀疏数据下的特征交互：在推荐系统中，特征高度稀疏（如用户行为、商品类别），FFM 通过域感知隐向量，更精准地捕捉跨域特征组合（如用户性别与电影类型的交互）。

# 1. Field 定义

 同一类特征（如用户域特征、电影域特征等）归为一个 Field，例如：

Field 示例：

User（用户域）: 年龄、性别、职业

Movie（电影域）: 类型、导演、主演

# 2. 模型公式

 FFM 的预测公式为：

$$
\hat {y} (x) = w _ {0} + \sum_ {i = 1} ^ {n} w _ {i} x _ {i} + \sum_ {i = 1} ^ {n} \sum_ {j = i + 1} ^ {n} \left\langle \mathbf {v} _ {i, f _ {j}}, \mathbf {v} _ {j, f _ {i}} \right\rangle x _ {i} x _ {j}
$$

 $w _ { 0 }$ ：全局偏置项， $w _ { i }$ ：一阶特征权重  
$\mathbf { v } _ { i , f _ { j } }$ ：特征 i 针对特征 j 所属域 的隐向量  
 ${ \bf v } _ { j , f _ { i } }$ ：特征 j 针对特征 i 所属域 $f _ { i }$ 的隐向量

# 3. 参数规模

 隐向量维度为 $k$ ，域数量为 F 时，FFM 参数总量为 $\scriptstyle { n \times F \times k }$ ，远高于 FM 的 $\scriptstyle n \times k _ { \circ }$

# 二、优缺点

# 1. 优点

 精细特征交互：通过filed 域感知隐向量，区分不同场景下的特征组合，使某一特征与不同特征做交互是，可发挥不同的重要性，提升模型表达能力；  
可解释性：可解释性强，可提供某些特征组合的重要性。

# 2. 缺点

 复杂度高：时间复杂度为 $O ( k n ^ { 2 } )$ （FM 为 $O ( k n )$ ），特征数 $n$ 较大时训练耗时。模型参数量为 $\scriptstyle { n \times F \times k }$ ，存储和计算资源消耗大，易过拟合（需强正则化）。  
域划分依赖：域划分不合理会导致性能下降，需结合业务经验调整。

# 三、与 FM 的对比

<table><tr><td>维度</td><td>FM</td><td>FFM</td></tr><tr><td>隐向量</td><td>每个特征1个隐向量</td><td>每个特征针对不同域有多个隐向量</td></tr><tr><td>时间复杂度</td><td>O(kn)</td><td>O(kn2)</td></tr><tr><td>参数数量</td><td>n×k</td><td>n×F×k</td></tr><tr><td>适用场景</td><td>中小规模特征</td><td>高维稀疏特征（需强正则化）</td></tr></table>

# 四、FFM 与其他CTR模型对比

<table><tr><td>模型</td><td>特征交互方式</td><td>参数量</td><td>优势</td><td>劣势</td></tr><tr><td>LR</td><td>无（需人工组合）</td><td>n</td><td>简单高效</td><td>无自动交叉</td></tr><tr><td>FM</td><td>隐向量内积（统一）</td><td>n×k</td><td>自动二阶交叉</td><td>交互粒度粗</td></tr><tr><td>FFM</td><td>域感知隐向量</td><td>n×F×k</td><td>精细域间交互</td><td>参数多，易过拟合</td></tr><tr><td>DeepFM</td><td>FM + DNN</td><td>n×k + DNN</td><td>二阶+高阶交叉</td><td>训练复杂</td></tr><tr><td>DCN-v2</td><td>显式交叉网络</td><td>较大</td><td>任意高阶交叉</td><td>计算开销大</td></tr></table>

# 五、Python 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FM Layer(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_dim)
        self.linear = nn.Embedding(num_features, 1)

    def forward(self, x):
        linear_part = self.linear(x).sum(dim=1)
        emb = self.embedding(x)
        square_of_sum = emb.sum(dim=1).pow(2)
        sum_of_square = emb.pow(2).sum(dim=1)
        fm_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=-1, keepdim=True)
        return linear_part.squeeze(-1) + fm_part.squeeze(-1)

class FFMLayer(nn.Module):
    def __init__(self, num_features, num_fields, embedding_dim):
        super().__init__()
        self.num_fields = num_fields
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_features, embedding_dim) for _ in range(num_fields)
        ])
        self.linear = nn.Embedding(num_features, 1)

    def forward(self, x, field_ids):
        linear_part = self.linear(x).sum(dim=1).squeeze(-1)
        batch_size, num_feats = x.shape
        emb_list = []
        for i in range(num_feats):
            field_i = field_ids[i]
            emb_list.append(self.embeddings[field_i](x[:, i]))
        embs = torch.stack(emb_list, dim=1)
        ffm_part = torch.tensor(0.0, device=x.device)
        for i in range(num_feats):
            for j in range(i + 1, num_feats):
                v_i = self.embeddings[field_ids[j]](x[:, i])
                v_j = self.embeddings[field_ids[i]](x[:, j])
                ffm_part = ffm_part + (v_i * v_j).sum(dim=-1)
        return linear_part + ffm_part

class FFMModel(nn.Module):
    def __init__(self, num_features, num_fields, embedding_dim=8):
        super().__init__()
        self.ffm = FFMLayer(num_features, num_fields, embedding_dim)

    def forward(self, x, field_ids):
        logits = self.ffm(x, field_ids)
        return torch.sigmoid(logits)

num_features = 100
num_fields = 5
num_feats_per_sample = 10

model = FFMModel(num_features, num_fields, embedding_dim=8)
x = torch.randint(0, num_features, (32, num_feats_per_sample))
field_ids = torch.tensor([i % num_fields for i in range(num_feats_per_sample)])
output = model(x, field_ids)
print(f"FFM输出形状: {output.shape}")
print(f"FFM输出示例: {output[:4].detach().numpy().round(4)}")
print(f"FFM参数量: {sum(p.numel() for p in model.parameters()):,}")
```

# 一、背景与动机

 在推荐系统中，特征通常由大量的稀疏特征（如用户 ID、物品 ID、类目等）经过 Embedding 后拼接而成。然而，不同特征域（field）对最终预测的重要性是不同的，而且这种重要性会随着不同的输入样本动态变化。例如，对于某个用户，"年龄"特征可能比"城市"更重要；而对另一个用户则可能相反。  
 传统的 CTR 模型（如 DeepFM、DCN）通常对所有特征域一视同仁地拼接后送入 DNN，缺乏对特征域重要性的动态建模能力。  
 SENet（Squeeze-and-Excitation Network）最早在计算机视觉领域提出（用于通道注意力），后被引入推荐系统，用于对特征域级别的重要性进行动态加权。

# 二、核心思想

SENet 的核心是一个三步操作：Squeeze Excitation Re-Weight，实现对每个特征域 embedding 的自适应重要性缩放。

输入EMBEDDING矩阵

$$
\mathbf {E} = \left[ \mathbf {e} _ {1}, \mathbf {e} _ {2}, \dots , \mathbf {e} _ {f} \right] \in \mathbb {R} ^ {f \times k}
$$

STEP1:SQUEEZE-均值池化

$$
z _ {i} = \frac {1}{k} \sum_ {j = 1} ^ {k} e _ {i, j}, \quad \mathbf {z} = [ z _ {1}, z _ {2}, \dots , z _ {f} ] \in \mathbb {R} ^ {f}
$$

STEP2:EXCITATION-两层FC瓶颈网络

$$
\mathbf {A} = \sigma \left(\mathbf {W} _ {2} \cdot \operatorname {R e L U} \left(\mathbf {W} _ {1} \cdot \mathbf {z}\right)\right), \quad \mathbf {W} _ {1} \in \mathbb {R} ^ { [ f / r ] \times f}, \quad \mathbf {W} _ {2} \in \mathbb {R} ^ {f \times [ f / r ]}
$$

$$
\mathbf {v} _ {i} = a _ {i} \cdot \mathbf {e} _ {i}, \quad \mathbf {V} = [ \mathbf {v} _ {1}, \mathbf {v} _ {2}, \dots , \mathbf {v} _ {f} ] \in \mathbb {R} ^ {f \times k}
$$

# 2.1 输入表示

假设模型有 $f$ 个特征域（fields），每个特征域经过 Embedding 层后得到一个 $k$ 维向量：

$$
\mathbf {E} = \left[ \mathbf {e} _ {1}, \mathbf {e} _ {2}, \dots , \mathbf {e} _ {f} \right], \quad \mathbf {e} _ {i} \in \mathbb {R} ^ {k}
$$

# 2.2 Squeeze（压缩）

对每个特征域的 embedding 向量进行统计量提取，将其压缩为一个标量，形成一个 $f$ 维的全局描述向量。最常用的方式是均值池化（Mean Pooling）：

$$
z _ {i} = \frac {1}{k} \sum_ {j = 1} ^ {k} e _ {i, j}, \quad \mathbf {z} = [ z _ {1}, z _ {2}, \dots , z _ {f} ] \in \mathbb {R} ^ {f}
$$

这一步的目的是将每个域的 embedding 浓缩为一个全局统计量，为后续的注意力计算提供输入。

# 2.3 Excitation（激励）

用一个两层全连接网络（bottleneck 结构）来学习特征域之间的非线性依赖关系，输出每个域的重要性权重：

$$
\mathbf {A} = \sigma \left(\mathbf {W} _ {2} \cdot \operatorname {R e L U} \left(\mathbf {W} _ {1} \cdot \mathbf {z}\right)\right)
$$

其中：

 $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { f / r \times f }$ 是降维矩阵， 是压缩比（reduction ratio），用于控制瓶颈层的维度  
$\mathbf { W } _ { 2 } \in \mathbb { R } ^ { f \times f / r }$ 是升维矩阵，恢复到原始域数  
 $\sigma$ 是 Sigmoid 激活函数，将权重限制在 (0, 1)  
 $\mathbf { A } = [ a _ { 1 } , a _ { 2 } , \dotsc , a _ { f } ] \in \mathbb { R } ^ { f }$ 是学到的各域注意力权重

# 2.4 Re-Weight（重加权）

将学到的注意力权重作用回原始 embedding，对每个特征域的 embedding 进行逐元素缩放：

$$
\mathbf {v} _ {i} = a _ {i} \cdot \mathbf {e} _ {i}, \quad \mathbf {V} = [ \mathbf {v} _ {1}, \mathbf {v} _ {2}, \dots , \mathbf {v} _ {f} ]
$$

这样，重要的特征域会被放大，不重要的特征域会被抑制。

# 三、优势与总结

<table><tr><td>优势</td><td>说明</td></tr><tr><td>动态性</td><td>不同样本获得不同的特征域权重，而非静态的特征选择</td></tr><tr><td>轻量级</td><td>仅引入两个小型全连接层，参数量极小 O(f^2/r)</td></tr><tr><td>即插即用</td><td>可嵌入任何基于 Embedding 拼接的 CTR 模型中</td></tr><tr><td>可解释性</td><td>输出的注意力权重 ai 可直接反映各域重要性</td></tr></table>

# 四、代码示例（PyTorch）

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

if __name__ == "__main__":
    batch, fields, dim = 4, 8, 16
    x = torch.randn(batch, fields, dim)
    senet = SENet(field_num=fields, reduction_ratio=2)
    out = senet(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"各域权重示例: {senet.excitation(x.mean(dim=-1))[0].detach().numpy().round(3)}")
```

# 六、FFM + SENet 组合应用

在工业实践中，FFM可以与SENet组合使用：先用SENet对各特征域的Embedding进行动态加权，再输入FFM进行域感知交叉。这种组合能同时获得动态特征选择和精细特征交互的能力。

组合流程：原始稀疏特征 → Embedding层 → SENet动态加权 → FFM域感知交叉 → 输出层

# 七、常见面试追问

1. Q: FFM的域划分有哪些常见策略？
A: (1) 按特征类型：用户域、物品域、上下文域；(2) 按特征来源：画像特征、行为特征、环境特征；(3) 按特征粒度：ID类、类别类、数值类。划分原则是同一域内特征语义相近。

2. Q: FFM在工业上为什么不常用？
A: 参数量是FM的F倍，在大规模稀疏场景（如千万级用户ID）下，参数爆炸严重。此外，域划分需要业务经验，自动化程度低。现代工业实践中更倾向于DeepFM、DCN-v2等模型。

3. Q: SENet中reduction_ratio如何选择？
A: 通常取2-4。比值太小（如1）则失去瓶颈结构的正则化效果；比值太大（如16）则压缩过度，信息损失严重。实践中r=2是推荐系统中常用的设置。
