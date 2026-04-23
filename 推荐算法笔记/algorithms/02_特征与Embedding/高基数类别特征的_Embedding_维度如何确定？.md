# 面试题：高基数类别特征的 Embedding 维度如何确定？

# 面试题：高基数类别特征的 Embedding 维度如何确定？

高基数类别型特征（如用户 ID、商品ID等）的 Embedding维度确定需综合考虑特征基数、模型复杂度和任务需求，以下是embedding维度确定的主要方法：

# 一、基于特征基数对数关系

- 理论上，Embedding 维度应与特征基数（即唯一值数量 vocab_size）的对数成正比。例如，若特征基数为 100 万（如内容 ID），可选用约 20 维；而基数较小的特征（如性别、地域）则适合更低维度（如 4-8 维）。
- 公式参考：dim≈log(vocab_size)

其中 vocab_size 为特征唯一值数量。例如，若 vocab_size=1e6，则 $\mathsf { l o g } 2 ( 1 \mathsf { e } 6 ) { \approx } 2 0 _ { \ast }$

## 1.1 维度计算的代码实现

```python
import math
import numpy as np

def calc_embedding_dim(vocab_size, method="log2"):
    if method == "log2":
        return max(4, int(math.log2(vocab_size)))
    elif method == "sqrt":
        return max(4, int(math.sqrt(vocab_size)))
    elif method == "log10_4x":
        return max(4, int(4 * math.log10(vocab_size)))
    elif method == "google_rule":
        return min(600, max(4, int(round(vocab_size ** 0.25))))

feature_cardinalities = {
    "gender": 3,
    "city": 500,
    "user_id": 10000000,
    "item_id": 50000000,
    "category": 200,
    "tag": 10000,
}

print(f"{'特征':<12} {'基数':<15} {'log2':<8} {'sqrt':<8} {'log10_4x':<10} {'google':<8}")
for feat, card in feature_cardinalities.items():
    dims = {
        "log2": calc_embedding_dim(card, "log2"),
        "sqrt": calc_embedding_dim(card, "sqrt"),
        "log10_4x": calc_embedding_dim(card, "log10_4x"),
        "google": calc_embedding_dim(card, "google_rule"),
    }
    print(f"{feat:<12} {card:<15} {dims['log2']:<8} {dims['sqrt']:<8} {dims['log10_4x']:<10} {dims['google']:<8}")
```

## 1.2 经验公式对比

| 经验法则 | 公式 | 100万基数结果 | 适用场景 |
|---------|------|-------------|---------|
| 对数法则 | $\log_2(N)$ | 20 | 通用推荐 |
| 平方根法则 | $\sqrt{N}$ | 1000 | 需要更高表达力 |
| Google 规则 | $N^{0.25}$ | 32 | Word2Vec 等场景 |
| 4倍对数法则 | $4 \times \log_{10}(N)$ | 24 | 推荐系统常用 |

# 二、平衡模型性能与计算资源

- 高基数特征：若直接采用固定维度（如所有特征统一为 16维），可能导致大基数特征欠拟合（信息压缩不足）或小基数特征过拟合（冗余参数）。推荐采用动态维度分配，例如：

  - 对百万级特征使用 20-64 维；
  - 对千级特征使用 8-16 维；
  - 对百级以下特征使用 4-8 维。

- 资源限制：在内存或计算资源受限时，可通过哈希分桶（如将百万级特征映射到 1 万桶）降低实际基数，再设置合理维度。

## 2.1 哈希分桶（Hashing Trick）代码实现

```python
import torch
import torch.nn as nn

class HashingEmbedding(nn.Module):
    def __init__(self, vocab_size, num_buckets=10000, embedding_dim=16):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embedding_dim)
        self.vocab_size = vocab_size

    def _hash(self, ids):
        return ids % self.num_buckets

    def forward(self, ids):
        bucket_ids = self._hash(ids)
        return self.embedding(bucket_ids)

class FeatureAwareHashEmbedding(nn.Module):
    def __init__(self, vocab_sizes, num_buckets_per_feature, embedding_dims):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        for feat_name, vocab_size in vocab_sizes.items():
            n_buckets = min(vocab_size, num_buckets_per_feature.get(feat_name, 10000))
            dim = embedding_dims.get(feat_name, 16)
            self.embeddings[feat_name] = HashingEmbedding(vocab_size, n_buckets, dim)

    def forward(self, feature_dict):
        return {name: self.embeddings[name](ids) for name, ids in feature_dict.items()}

vocab_sizes = {"user_id": 10000000, "item_id": 50000000, "city": 500}
num_buckets = {"user_id": 50000, "item_id": 100000, "city": 500}
emb_dims = {"user_id": 32, "item_id": 64, "city": 8}
model = FeatureAwareHashEmbedding(vocab_sizes, num_buckets, emb_dims)
sample = {
    "user_id": torch.tensor([1234567, 7654321]),
    "item_id": torch.tensor([100, 200]),
    "city": torch.tensor([10, 20]),
}
outputs = model(sample)
for name, emb in outputs.items():
    print(f"{name}: shape={emb.shape}")
```

# 三、高级方法参考

- 变长 Embedding：使用矩阵变换（如线性投影）将不同维度统一到固定长度，或通过分块拼接/截断处理，兼顾灵活性与计算效率。
- 自动化方法：如谷歌的 DHE（Deep Hash Embeddings）[详见 https://arxiv.org/pdf/2010.10784v2]通过多层神经网络动态生成 Embedding，避免预设维度，适用于超大规模特征。

DHE 将嵌入生成分为编码阶段 （Encoding）和解码阶段 （Decoding）：

编码阶段：多哈希函数映射，使用多个（如 1024 个）哈希函数将特征值（如 ID）映射为一个高维稠密向量。每个哈希函数生成一个整数，并通过归一化转化为均匀分布或高斯分布的实数向量。
解码阶段：深度神经网络（DNN）转换，将编码后的向量输入多层神经网络（如 MLP），通过非线性激活函数（如 Mish）生成最终嵌入向量。DNN 的参数规模与特征词表无关，显著降低内存消耗。

![](images/5ba865e9768fdc19f4d685258e288bde47cf2428dddd92293c3270169db7ffd6.jpg)
(a) One-hot
(b)Deep Hash Embedding (ours)

## 3.1 DHE（Deep Hash Embedding）代码实现

```python
import torch
import torch.nn as nn

class DHEncoder(nn.Module):
    def __init__(self, num_hashes=1024, hash_range=1000000):
        super().__init__()
        self.num_hashes = num_hashes
        self.hash_range = hash_range
        self.register_buffer("seeds", torch.randint(0, hash_range, (num_hashes,)))

    def _multi_hash(self, ids):
        batch_size = ids.shape[0]
        encoded = torch.zeros(batch_size, self.num_hashes, device=ids.device)
        for i in range(self.num_hashes):
            encoded[:, i] = ((ids + self.seeds[i]) % self.hash_range).float() / self.hash_range
            encoded[:, i] = (encoded[:, i] - 0.5) * 2.0
        return encoded

    def forward(self, ids):
        return self._multi_hash(ids)

class DHEDecoder(nn.Module):
    def __init__(self, num_hashes=1024, embedding_dim=32, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        layers = []
        input_dim = num_hashes
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, h_dim), nn.Mish()])
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, encoded):
        return self.mlp(encoded)

class DeepHashEmbedding(nn.Module):
    def __init__(self, num_hashes=1024, embedding_dim=32, hidden_dims=None):
        super().__init__()
        self.encoder = DHEncoder(num_hashes)
        self.decoder = DHEDecoder(num_hashes, embedding_dim, hidden_dims)

    def forward(self, ids):
        encoded = self.encoder(ids)
        return self.decoder(encoded)

dhe_model = DeepHashEmbedding(num_hashes=1024, embedding_dim=32)
test_ids = torch.tensor([12345, 67890, 11111])
embeddings = dhe_model(test_ids)
print(f"DHE embedding shape: {embeddings.shape}")
print(f"参数量: {sum(p.numel() for p in dhe_model.parameters())}")
```

## 3.2 DHE vs 传统 Embedding 对比

| 维度 | 传统 Embedding | DHE |
|------|---------------|-----|
| 内存占用 | $O(V \times d)$，与词表成正比 | $O(1)$，与词表大小无关 |
| 查询速度 | O(1) 直接查表 | 需要 MLP 前向计算 |
| 冷启动 | 新 ID 无表示 | 任意 ID 都有表示 |
| 参数量 | 随词表线性增长 | 固定，仅 MLP 参数 |

# 四、实践建议

- 初始设定：按 dim=log2(voc_size)作为基准，例如百万级特征设为 20 维，万级特征设为 16 维。然后，通过交叉验证测试不同维度的模型效果。例如，从较低维度（如 16 维）开始逐步增加，观察验证集性能变化，选择边际收益显著下降前的临界点。
- 动态调整：根据任务类型（分类/回归）和模型结构（如是否结合注意力机制）灵活调整，复杂交互任务需更高维度。

## 4.1 消融实验代码

```python
import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import numpy as np

class SimpleRecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.emb(x).mean(dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

def ablation_embedding_dim(vocab_size, dims_to_test, train_loader, val_loader, epochs=10):
    results = {}
    for dim in dims_to_test:
        model = SimpleRecModel(vocab_size, dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y.float())
                loss.backward()
                optimizer.step()
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    pred = model(batch_x)
                    val_losses.append(criterion(pred, batch_y.float()).item())
            avg_val_loss = np.mean(val_losses)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        results[dim] = best_val_loss
        print(f"dim={dim:3d}, best_val_loss={best_val_loss:.4f}, params={sum(p.numel() for p in model.parameters())}")
    return results

vocab_size = 100000
dims_to_test = [4, 8, 16, 32, 64, 128]
print("Embedding 维度消融实验:")
print("-" * 50)
for d in dims_to_test:
    param_count = vocab_size * d + d * 64 + 64 + 64 * 1 + 1
    print(f"dim={d:3d}: 预估参数量={param_count:,}")
```

## 4.2 自适应维度分配策略

```python
def auto_assign_dims(vocab_sizes, total_budget=500000, min_dim=4, max_dim=128):
    log_cardinalities = {f: math.log2(max(v, 2)) for f, v in vocab_sizes.items()}
    total_log = sum(log_cardinalities.values())
    raw_dims = {}
    for feat, log_c in log_cardinalities.items():
        proportion = log_c / total_log
        budget_share = total_budget * proportion
        dim = max(min_dim, min(max_dim, int(budget_share / vocab_sizes[feat])))
        raw_dims[feat] = dim
    actual_total = sum(vocab_sizes[f] * d for f, d in raw_dims.items())
    if actual_total > total_budget:
        scale = total_budget / actual_total
        raw_dims = {f: max(min_dim, int(d * scale)) for f, d in raw_dims.items()}
    return raw_dims

vocab = {"user_id": 10000000, "item_id": 50000000, "city": 500, "category": 200, "tag": 10000}
dims = auto_assign_dims(vocab, total_budget=10_000_000)
print("自适应维度分配结果:")
for feat, dim in dims.items():
    print(f"  {feat}: vocab={vocab[feat]:,}, dim={dim}, params={vocab[feat]*dim:,}")
total_params = sum(vocab[f] * d for f, d in dims.items())
print(f"  总参数量: {total_params:,}")
```

# 五、常见问题与易错点

1. **维度过高导致过拟合**：高基数特征使用过大维度（如 user_id 用 256 维），在数据稀疏时严重过拟合。低频 ID 的 Embedding 几乎没被训练过。
2. **维度过低导致欠拟合**：将所有特征统一为 4 维，大基数特征无法充分表达信息。
3. **忽视频率信息**：高频 ID 可以用较高维度，低频 ID 应限制维度或使用哈希分桶。
4. **哈希冲突未评估**：使用哈希分桶时，桶数过少会导致大量冲突，信息严重丢失。建议桶数 >= 基数的 1/10。

# 六、学习路径建议

1. 理解 Embedding 的本质：将离散 ID 映射到连续向量空间
2. 掌握维度计算的经验公式并实践
3. 学习哈希分桶等大规模特征处理技巧
4. 阅读论文：DHE（Google 2021）、DCN V2 中的特征交互
5. 实践：在真实推荐数据集上进行消融实验，找到最优维度配置
