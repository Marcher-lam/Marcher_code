# NMF 学习文档

## 1. 算法基础认知

非负矩阵分解（Non-negative Matrix Factorization, NMF）是一种在非负约束下将矩阵分解为两个低维非负矩阵乘积的降维方法。由 Lee 和 Seung 于 1999 年在 Nature 上提出。NMF 的核心优势在于分解结果的非负性使其具有"部分组合"的直观解释，特别适合文本主题模型、图像分解等场景。

## 2. 核心原理

给定非负矩阵 $V \in \mathbb{R}_{+}^{m \times n}$，NMF 寻找两个非负矩阵 $W \in \mathbb{R}_{+}^{m \times k}$ 和 $H \in \mathbb{R}_{+}^{k \times n}$，使得：

$$V \approx WH$$

- $W$：基矩阵（basis），每列是一个"部件"或"主题"
- $H$：系数矩阵（encoding），每列是对应样本在基上的非负组合系数

非负约束保证了分解结果是一种"加性组合"，而非 SVD 中的正负抵消，因此更符合许多物理现实（如图像像素值、词频计数等均非负）。

## 3. 数学公式与推导

### 目标函数

常用 Frobenius 范数或 KL 散度：

$$\min_{W,H \geq 0} \|V - WH\|_F^2 = \sum_{i,j}(V_{ij} - (WH)_{ij})^2$$

或：

$$\min_{W,H \geq 0} D_{KL}(V \| WH) = \sum_{i,j}\left(V_{ij}\log\frac{V_{ij}}{(WH)_{ij}} - V_{ij} + (WH)_{ij}\right)$$

### 乘法更新规则（Frobenius 范数）

Lee 和 Seung 证明了如下更新规则单调递减目标函数：

$$H_{aj} \leftarrow H_{aj} \cdot \frac{(W^T V)_{aj}}{(W^T W H)_{aj} + \epsilon}$$

$$W_{ia} \leftarrow W_{ia} \cdot \frac{(V H^T)_{ia}}{(W H H^T)_{ia} + \epsilon}$$

其中 $\epsilon$ 为防止除零的小常数。更新后自然保持非负性。

## 4. 训练过程讲解

1. **初始化**：随机初始化 $W$ 和 $H$（或使用 NNDSVD 初始化以加速收敛）
2. **交替更新**：
   - 固定 $W$，按乘法规则更新 $H$
   - 固定 $H$，按乘法规则更新 $W$
3. **收敛判断**：当目标函数变化小于阈值或达到最大迭代次数时停止
4. **归一化**：通常对 $W$ 按列归一化，对应调整 $H$

该算法不保证找到全局最优解（目标函数非凸），但乘法更新保证目标函数单调下降。

## 5. 应用场景

- 文本主题提取（词-文档矩阵分解）
- 图像部件分解（人脸特征提取）
- 推荐系统（用户-物品矩阵补全）
- 音频信号源分离
- 广告点击率预测中的特征分解

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 非负约束使结果可解释 | 解不唯一（$W,H$ 可同时缩放） |
| 稀疏性好，适合高维数据 | 对初始化敏感 |
| 实现简单（乘法更新） | 不保证全局最优 |
| 存储量低（低秩近似） | 主题数 $k$ 需手动设定 |

## 7. 调库实现（Python + 完整代码 + 注释）

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

docs = [
    "深度学习 神经网络 模型训练",
    "广告推荐 用户画像 点击率预估",
    "自然语言处理 文本分类 情感分析",
    "卷积神经网络 图像识别 目标检测",
    "推荐系统 协同过滤 深度学习",
    "广告投放 竞价策略 实时竞价",
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

nmf = NMF(n_components=3, random_state=42, max_iter=500)
W = nmf.fit_transform(X)
H = nmf.components_

feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(H):
    top_words = [feature_names[i] for i in topic.argsort()[-4:][::-1]]
    print(f"主题 {topic_idx}: {top_words}")

print(f"\n重构误差: {nmf.reconstruction_err_:.4f}")
print(f"W shape: {W.shape}, H shape: {H.shape}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def nmf(V, k, max_iter=200, tol=1e-4):
    m, n = V.shape
    W = np.random.rand(m, k) + 1e-10
    H = np.random.rand(k, n) + 1e-10
    eps = 1e-10

    for iteration in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ H @ H.T + eps)

        if iteration % 10 == 0:
            loss = np.linalg.norm(V - W @ H, 'fro')
            if iteration > 0 and abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

    return W, H, loss

np.random.seed(42)
V = np.random.rand(20, 10)
W, H, final_loss = nmf(V, k=3)
print(f"重构误差: {final_loss:.6f}")
print(f"W 非负: {(W >= 0).all()}, H 非负: {(H >= 0).all()}")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(W[:20, :], aspect='auto', cmap='Reds')
axes[0].set_title("W (词项-主题矩阵)")
axes[0].set_xlabel("主题")
axes[0].set_ylabel("词项")

axes[1].imshow(H, aspect='auto', cmap='Reds')
axes[1].set_title("H (主题-文档矩阵)")
axes[1].set_xlabel("文档")
axes[1].set_ylabel("主题")

plt.tight_layout()
plt.savefig("nmf_visual.png", dpi=150)
plt.close()
```

热力图中颜色深浅表示词项/文档与各主题的关联强度，红色越深关联越强。

## 10. 模型评估

- **重构误差**：$\|V - WH\|_F$，越小越好
- **稀疏度**：$\text{sparsity}(H) = \frac{\sqrt{n} - \|h\|_1/\|h\|_2}{\sqrt{n}-1}$，衡量分解的稀疏性
- **主题一致性（Topic Coherence）**：衡量提取主题中词的语义相关性
- **下游分类/检索任务的准确率**

## 11. 常见问题与易错点

- **初始化敏感**：不同初始化可能收敛到不同解，建议使用 NNDSVD 初始化或多次运行取最优
- **未加小常数 $\epsilon$**：乘法更新中分母可能为零导致 NaN
- **$k$ 值选择不当**：$k$ 太大会出现过拟合，太小则主题粒度过粗
- **数据未保证非负**：输入矩阵必须非负，TF-IDF 值天然满足此约束
- **与 SVD 混淆**：NMF 的非负约束使解不具有 SVD 的正交性，但换来了可解释性

## 12. 学习总结

NMF 是一种简洁而实用的矩阵分解方法，非负约束使分解结果具有直观的"部分组合"语义。乘法更新规则简单高效，保证目标函数单调下降。NMF 在主题模型中是 LSA 的有力替代，与 PLSA/LDA 形成互补。

## 13. 练习题与思考题（含答案）

**Q1**：为什么 NMF 的乘法更新规则能保持非负性？

> A1：更新规则的形式是 $H_{aj} \leftarrow H_{aj} \times \frac{\text{非负}}{\text{非负}}$，即当前值乘以一个非负比例因子，因此只要初始化非负，所有后续值都保持非负。

**Q2**：NMF 解的唯一性为什么不能保证？

> A2：对任意可逆对角阵 $D$，有 $WH = (WD)(D^{-1}H)$，且 $WD$ 和 $D^{-1}H$ 仍非负。因此存在等价解的缩放族，通常通过归一化 $W$ 列来缓解。

**Q3**：NMF 与 LSA 在主题模型中的本质区别是什么？

> A3：LSA 使用 SVD，分解结果有正有负，主题缺乏直观语义；NMF 的非负约束使得每个文档只能由主题的"正向组合"表示，更符合"文档由若干主题混合而成"的直觉。

## 14. 学习路径建议

- **前置知识**：线性代数、矩阵分解、优化基础
- **进阶方向**：PLSA → LDA → 深度主题模型（Neural Topic Model）
- **推荐实践**：在 20Newsgroups 数据集上对比 NMF 和 LDA 的主题质量
