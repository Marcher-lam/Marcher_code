# PLSA 学习文档

## 1. 算法基础认知

概率潜在语义分析（Probabilistic Latent Semantic Analysis, PLSA）由 Hofmann 于 1999 年提出，是对 LSA 的概率化扩展。PLSA 引入隐变量（主题），将文档-词共现关系建模为文档生成主题、主题生成词的概率过程，并使用 EM 算法进行参数估计。

与 LSA 相比，PLSA 给出了清晰的概率解释：每个文档以一定概率分布选择主题，每个主题以一定概率分布生成词。

## 2. 核心原理

PLSA 的生成过程为：

1. 对每篇文档 $d$，选择一个隐主题 $z \sim P(z|d)$
2. 给定主题 $z$，生成一个词 $w \sim P(w|z)$

因此文档-词的联合概率为：

$$P(d, w) = P(d) \sum_{z} P(z|d) P(w|z)$$

模型参数为：
- $P(z|d)$：文档-主题分布（每篇文档的主题概率）
- $P(w|z)$：主题-词分布（每个主题的词概率）

## 3. 数学公式与推导

### 对数似然

给定观测语料 $\{(d_i, w_j)\}$，完整数据对数似然：

$$\mathcal{L} = \sum_{d}\sum_{w} n(d, w) \log P(d, w) = \sum_{d}\sum_{w} n(d, w) \log\left[P(d)\sum_{z} P(z|d)P(w|z)\right]$$

由于隐变量 $z$ 的存在，直接优化困难，采用 EM 算法。

### E 步

计算隐变量的后验分布：

$$P(z|d, w) = \frac{P(z|d)P(w|z)}{\sum_{z'} P(z'|d)P(w|z')}$$

### M 步

最大化期望完整数据对数似然，更新参数：

$$P(w|z) = \frac{\sum_{d} n(d, w) P(z|d, w)}{\sum_{d}\sum_{w'} n(d, w') P(z|d, w')}$$

$$P(z|d) = \frac{\sum_{w} n(d, w) P(z|d, w)}{\sum_{z'}\sum_{w} n(d, w) P(z'|d, w)}$$

## 4. 训练过程讲解

1. **初始化**：随机初始化 $P(z|d)$ 和 $P(w|z)$（需归一化）
2. **E 步**：对每个 $(d, w)$ 对，计算 $P(z|d, w)$
3. **M 步**：利用 E 步结果重新估计 $P(z|d)$ 和 $P(w|z)$
4. **收敛判断**：计算对数似然，当变化量小于阈值时停止
5. **主题解读**：对每个主题 $z$，取 $P(w|z)$ 最高的词作为主题关键词

## 5. 应用场景

- 文本主题挖掘与聚类
- 文档分类的特征提取
- 信息检索中的语义匹配
- 广告关键词与用户兴趣的主题匹配
- 推荐系统中的用户兴趣建模

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 概率框架，结果可解释 | 参数随文档数线性增长 |
| 能处理一词多义 | 容易过拟合（文档数多时参数多） |
| EM 算法保证收敛 | 没有对文档主题分布的先验 |
| 优于 LSA 的语义捕获 | 不属于生成式模型（$P(d)$ 未知） |

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = [
    "深度学习 神经网络 模型 训练 反向传播",
    "广告 推荐 点击率 预估 用户画像",
    "自然语言 处理 文本 分类 情感分析",
    "卷积 神经网络 图像 识别 目标检测",
    "推荐 系统 协同过滤 深度学习 用户",
    "广告 竞价 策略 实时 竞价",
    "文本 挖掘 主题 模型 聚类",
    "图像 生成 对抗 网络 深度学习",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

lda = LatentDirichletAllocation(
    n_components=3, max_iter=50,
    learning_method='em', random_state=42
)
lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
    print(f"主题 {topic_idx}: {top_words}")

doc_topic = lda.transform(X)
print(f"\n文档-主题分布 shape: {doc_topic.shape}")
print(f"perplexity: {lda.perplexity(X):.2f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def plsa_em(n_dw, n_topics, max_iter=100, tol=1e-6):
    n_docs, n_words = n_dw.shape
    P_z_d = np.random.rand(n_docs, n_topics)
    P_z_d /= P_z_d.sum(axis=1, keepdims=True)

    P_w_z = np.random.rand(n_topics, n_words)
    P_w_z /= P_w_z.sum(axis=1, keepdims=True)

    for iteration in range(max_iter):
        P_z_dw = np.zeros((n_docs, n_words, n_topics))
        for z in range(n_topics):
            P_z_dw[:, :, z] = P_z_d[:, z:z+1] * P_w_z[z:z+1, :]
        P_z_dw /= (P_z_dw.sum(axis=2, keepdims=True) + 1e-15)

        P_w_z_new = np.zeros_like(P_w_z)
        P_z_d_new = np.zeros_like(P_z_d)
        for z in range(n_topics):
            P_w_z_new[z] = (n_dw * P_z_dw[:, :, z]).sum(axis=0)
            P_z_d_new[:, z] = (n_dw * P_z_dw[:, :, z]).sum(axis=1)
        P_w_z = P_w_z_new / (P_w_z_new.sum(axis=1, keepdims=True) + 1e-15)
        P_z_d = P_z_d_new / (P_z_d_new.sum(axis=1, keepdims=True) + 1e-15)

        log_likelihood = 0.0
        for d in range(n_docs):
            for w in range(n_words):
                if n_dw[d, w] > 0:
                    p = sum(P_z_d[d, z] * P_w_z[z, w] for z in range(n_topics))
                    log_likelihood += n_dw[d, w] * np.log(p + 1e-15)

        if iteration > 0 and abs(prev_ll - log_likelihood) < tol:
            break
        prev_ll = log_likelihood

    return P_z_d, P_w_z, log_likelihood

np.random.seed(42)
n_dw = np.random.randint(0, 5, (6, 8)).astype(float)
P_z_d, P_w_z, ll = plsa_em(n_dw, n_topics=3)
print(f"文档-主题分布:\n{P_z_d.round(3)}")
print(f"最终对数似然: {ll:.4f}")
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(P_z_d, cmap='Blues', aspect='auto')
axes[0].set_title("P(z|d) 文档-主题分布")
axes[0].set_xlabel("主题")
axes[0].set_ylabel("文档")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(P_w_z, cmap='Reds', aspect='auto')
axes[1].set_title("P(w|z) 主题-词分布")
axes[1].set_xlabel("词")
axes[1].set_ylabel("主题")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig("plsa_visual.png", dpi=150)
plt.close()
```

## 10. 模型评估

- **对数似然**：越高越好，直接衡量模型对数据的拟合程度
- **Perplexity**：$\text{Perplexity} = \exp(-\mathcal{L}/N)$，越低越好
- **主题一致性**：UMass 或 UCI 指标衡量主题词的语义关联度
- **下游任务**：用主题分布作为特征，评估分类/检索性能

## 11. 常见问题与易错点

- **EM 收敛到局部最优**：多次随机初始化取最优对数似然的解
- **主题数选择**：过少则主题粗粒度，过多则过拟合且语义分散
- **未做平滑处理**：概率为 0 时 log 会出问题，需加小常数
- **与 LDA 混淆**：PLSA 没有对 $P(z|d)$ 加先验，LDA 引入 Dirichlet 先验解决了过拟合问题
- **参数规模问题**：PLSA 中 $P(z|d)$ 的参数量与文档数成正比，对新文档无法直接推断

## 12. 学习总结

PLSA 是 LSA 到概率模型的关键桥梁。它引入隐主题变量，通过 EM 算法交替估计文档-主题和主题-词分布。虽然存在参数过多和过拟合的问题（后续被 LDA 的贝叶斯框架解决），但 PLSA 的建模思想深刻影响了概率主题模型的发展。

## 13. 练习题与思考题（含答案）

**Q1**：PLSA 与 LSA 的本质区别是什么？

> A1：LSA 基于线性代数（SVD 降维），无概率解释；PLSA 引入概率模型和隐变量，假设文档通过主题生成词，参数通过最大似然估计。

**Q2**：为什么 PLSA 容易过拟合？LDA 如何解决？

> A2：PLSA 对每篇文档都有一组独立的 $P(z|d)$ 参数，参数量随文档数线性增长。LDA 对 $P(z|d)$ 引入 Dirichlet 先验，将其视为从先验中采样的随机变量，从而实现参数共享和正则化。

**Q3**：PLSA 的 EM 算法中，E 步和 M 步分别计算什么？

> A3：E 步计算隐变量的后验 $P(z|d,w)$；M 步在固定后验的条件下，最大化期望完整数据对数似然，更新 $P(w|z)$ 和 $P(z|d)$。

## 14. 学习路径建议

- **前置知识**：概率论、EM 算法、LSA
- **进阶方向**：LDA（Dirichlet 先验）→ HDP（非参数主题模型）→ 神经主题模型
- **推荐实践**：在新闻数据集上实现 PLSA 并与 LDA 对比主题质量
