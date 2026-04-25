# FFM（Field-aware Factorization Machine）学习文档

## 1. 算法基础认知

FFM（Field-aware Factorization Machine）是 FM 的扩展，引入了"域"（Field）的概念。不同域的特征交互使用不同的隐向量，比 FM 更精细地建模特征交叉。在 CTR 预估竞赛中表现优异，是稀疏特征交叉的经典方法。

## 2. 核心原理

### FM 公式回顾

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

FM 中每个特征只有一个隐向量 $\mathbf{v}_i$，无论与哪个域的特征交互都使用同一个。

### FFM 公式

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_{i,f_j}, \mathbf{v}_{j,f_i}\rangle x_i x_j
$$

其中 $f_j$ 是特征 $j$ 所属的域（Field）。每个特征对每个域维护一个独立的隐向量，即特征 $i$ 与不同域的特征交互时使用不同的嵌入。

## 3. 数学公式与推导

**参数量分析**：
- FM：每个特征 1 个 $k$ 维隐向量，总参数 $O(nk)$
- FFM：每个特征 $f$ 个 $k$ 维隐向量（$f$ 为域数量），总参数 $O(nfk)$

**梯度推导**：对于特征对 $(i, j)$，FFM 的交叉项梯度为：

$$
\frac{\partial \hat{y}}{\partial \mathbf{v}_{i,f_j}} = x_i x_j \mathbf{v}_{j,f_i}
$$

FFM 使用 AdaGrad 优化器，每个参数维护独立的学习率，适应稀疏数据的梯度分布不均匀问题。

**为什么 FFM 比 FM 强**：当特征来自不同语义域时（如用户性别 × 广告类目），FFM 可以为每种跨域交互学习专门的表示，更加精准。

## 4. 训练过程讲解

1. 特征编码：将原始特征按域（Field）组织，如用户特征域、广告特征域、上下文特征域
2. 初始化：为每个特征在每个域初始化一个隐向量
3. 前向计算：遍历所有非零特征对，根据对方所属域选择对应隐向量做内积
4. 梯度更新：使用 AdaGrad 对每个参数独立更新学习率
5. 正则化：L2 正则 + 早停防止过拟合

## 5. 应用场景

- 广告 CTR 预估（竞赛常用 baseline）
- 推荐系统排序阶段
- 稀疏特征交叉场景（用户属性 × 物品属性）
- 与 Wide & Deep 结合使用

## 6. 优缺点分析

| 特性 | FM | FFM |
|------|----|----|
| 隐向量数量 | 每特征 1 个 | 每特征×每域 1 个 |
| 参数量 | $O(nk)$ | $O(nfk)$ |
| 表达能力 | 中等 | 较强 |
| 训练速度 | 快 | 较慢 |

**优点**：跨域交互建模更精细，在 CTR 预估竞赛中效果通常优于 FM。

**缺点**：参数量随域数线性增长，训练和推理速度较 FM 慢，内存占用更大。

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

class FieldAwareFactorizationMachine:
    def __init__(self, n_features, n_fields, k=8, lr=0.1, lambda_reg=0.001):
        self.n_features = n_features
        self.n_fields = n_fields
        self.k = k
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.randn(n_features, n_fields, k) * 0.01
        self.G_w0 = 1.0
        self.G_w = np.ones(n_features)
        self.G_V = np.ones((n_features, n_fields, k))

    def predict(self, X, fields):
        pred = self.w0 + X @ self.w
        nonzero = np.nonzero(X)[0]
        for idx_i in range(len(nonzero)):
            for idx_j in range(idx_i + 1, len(nonzero)):
                i, j = nonzero[idx_i], nonzero[idx_j]
                vi = self.V[i, fields[j]]
                vj = self.V[j, fields[i]]
                pred += X[i] * X[j] * np.dot(vi, vj)
        return 1.0 / (1.0 + np.exp(-pred))

    def fit_one(self, X, fields, y):
        pred = self.predict(X, fields)
        grad = pred - y
        self._adagrad_update(grad, X, fields)
        return -y * np.log(pred + 1e-8) - (1 - y) * np.log(1 - pred + 1e-8)

    def _adagrad_update(self, grad, X, fields):
        self.G_w0 += grad ** 2
        self.w0 -= self.lr * grad / np.sqrt(self.G_w0)
        nonzero = np.nonzero(X)[0]
        for i in nonzero:
            g = grad * X[i]
            self.G_w[i] += g ** 2
            self.w[i] -= self.lr * g / np.sqrt(self.G_w[i])
            for j in nonzero:
                if i == j:
                    continue
                g_v = grad * X[i] * X[j] * self.V[j, fields[i]]
                self.G_V[i, fields[j]] += g_v ** 2
                self.V[i, fields[j]] -= self.lr * (g_v + self.lambda_reg * self.V[i, fields[j]]) / np.sqrt(self.G_V[i, fields[j]])
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class FFMSimple:
    def __init__(self, n_feat, n_field, k=4):
        self.V = np.random.randn(n_feat, n_field, k) * 0.01
        self.w = np.zeros(n_feat)
        self.b = 0.0

    def predict(self, X, fields):
        logit = self.b + np.dot(X, self.w)
        nz = np.where(X != 0)[0]
        for ii in range(len(nz)):
            for jj in range(ii + 1, len(nz)):
                i, j = nz[ii], nz[jj]
                logit += X[i] * X[j] * np.dot(self.V[i, fields[j]], self.V[j, fields[i]])
        return sigmoid(logit)
```

## 9. 可视化与结果理解

- 对比 FM 和 FFM 的 AUC 随 epoch 变化曲线
- 可视化同一特征在不同域下的隐向量差异（PCA 降维展示）
- 绘制不同隐向量维度 $k$ 对模型效果的影响

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_features = 6
n_fields = 4
k = 4
feature_names = ['用户年龄', '用户性别', '广告类目', '广告价格', '时间时段', '设备类型']
field_names = ['用户域', '广告域', '上下文域', '历史域']

V = np.random.randn(n_features, n_fields, k) * 0.5

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

heatmaps = []
for f_idx in range(n_features):
    row = []
    for field_j in range(n_fields):
        for field_i in range(n_fields):
            if field_i >= field_j:
                continue
            sim = np.dot(V[f_idx, field_i], V[f_idx, field_j])
            row.append(sim)
    heatmaps.append(row)

n_pairs = n_fields * (n_fields - 1) // 2
pair_labels = []
for fi in range(n_fields):
    for fj in range(fi + 1, n_fields):
        pair_labels.append(f'{field_names[fi][:2]}×{field_names[fj][:2]}')

heatmap_data = np.zeros((n_features, n_pairs))
for f_idx in range(n_features):
    pair_idx = 0
    for fi in range(n_fields):
        for fj in range(fi + 1, n_fields):
            heatmap_data[f_idx, pair_idx] = np.dot(V[f_idx, fi], V[f_idx, fj])
            pair_idx += 1

im = axes[0].imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-1.5, vmax=1.5)
axes[0].set_xticks(range(n_pairs))
pair_labels_actual = []
idx = 0
for fi in range(n_fields):
    for fj in range(fi + 1, n_fields):
        pair_labels_actual.append(f'Field{fi}×Field{fj}')
        idx += 1
axes[0].set_xticklabels(pair_labels_actual, fontsize=8, rotation=45, ha='right')
axes[0].set_yticks(range(n_features))
axes[0].set_yticklabels(feature_names, fontsize=10)
axes[0].set_title('FFM: Cross-Field Embedding Similarity', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=axes[0], shrink=0.8, label='Dot Product')

fm_sim = np.random.randn(n_features, k) * 0.5
fm_heatmap = fm_sim @ fm_sim.T
im2 = axes[1].imshow(fm_heatmap[:n_features, :n_features], cmap='RdBu_r', aspect='auto', vmin=-1.5, vmax=1.5)
axes[1].set_xticks(range(n_features))
axes[1].set_xticklabels(feature_names, fontsize=9, rotation=45, ha='right')
axes[1].set_yticks(range(n_features))
axes[1].set_yticklabels(feature_names, fontsize=10)
axes[1].set_title('FM: Single Embedding Similarity', fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Dot Product')

plt.suptitle('FFM vs FM — Field-Aware Embedding 可视化对比', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ffm_embedding_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- **离线指标**：LogLoss、AUC
- **对比基线**：与 LR、FM 对比，通常 FFM > FM > LR
- **效率指标**：训练时间、预测延迟（FFM 通常比 FM 慢 2~5 倍）

## 11. 常见问题与易错点

- FFM 的特征必须按域组织，域的划分直接影响效果
- 参数量 $O(nfk)$，域数过多会导致参数爆炸和过拟合
- 必须使用 AdaGrad 而非普通 SGD，否则稀疏特征训练不稳定
- FFM 不适合特征稠密的场景，此时 FM 已足够

## 12. 学习总结

FFM 是 FM/FFM 系列中处理稀疏特征交叉的经典方法，在 CTR 预估竞赛中表现优异。核心改进是为每个特征在每个域维护独立隐向量，实现更精细的跨域交互建模。

## 13. 练习题与思考题（含答案）

**Q1**: FFM 相比 FM 的核心改进是什么？
> A1: 引入域（Field）概念，每个特征对不同域的交互使用不同隐向量，而非共享一个。

**Q2**: 为什么 FFM 使用 AdaGrad 而非 SGD？
> A2: 稀疏数据中不同特征的梯度频率差异大，AdaGrad 为每个参数自适应学习率，更适合。

**Q3**: 若有 100 个特征、5 个域、隐向量维度 8，FFM 的参数量是多少？
> A3: $100 \times 5 \times 8 = 4000$ 个隐向量参数，加上 100 个 $w_i$ 和 1 个 $w_0$。

## 14. 学习路径建议

1. 先学习线性回归和逻辑回归
2. 学习 FM（Factorization Machine）理解隐向量交叉
3. 学习 FFM 理解域感知交互
4. 进阶：学习 DeepFM（FM + DNN）、xDeepFM（CIN 交叉网络）
