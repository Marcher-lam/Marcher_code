# Word2Vec 学习文档

> 从稀疏到稠密——词向量化的革命性方法

---

## 1. 算法基础认知

### 1.1 什么是Word2Vec

**Word2Vec** 是Google在2013年提出的词嵌入方法，将词映射为低维稠密向量，使得语义相近的词在向量空间中距离相近。

```
One-Hot: "国王" → [1,0,0,...,0]  (10000维，稀疏)
Word2Vec: "国王" → [0.21, -0.15, 0.83, ...]  (300维，稠密)

最神奇的性质:
  vec("国王") - vec("男") + vec("女") ≈ vec("女王")
```

### 1.2 两种架构

| 架构 | 输入 | 输出 | 特点 |
|------|------|------|------|
| **CBOW** | 上下文词 | 中心词 | 根据上下文预测中心词 |
| **Skip-Gram** | 中心词 | 上下文词 | 根据中心词预测上下文 |

### 1.3 在推荐系统中的应用

- **Item2Vec**：把用户点击序列当作"句子"，物品当作"词"
- **Embedding层**：推荐系统中用户/物品的向量化表示
- **特征表示**：文本特征的稠密表示

---

## 2. 核心原理

### 2.1 CBOW

```
上下文: "今天 天气 很好 出去"
                ↓
        Embedding + 求平均
                ↓
            隐藏层 (向量)
                ↓
           Softmax 预测
                ↓
             "散步"（中心词）
```

### 2.2 Skip-Gram

```
中心词: "天气"
    ↓
  Embedding
    ↓
  隐藏层
    ↓
  Softmax → 预测上下文: "今天", "很好", "散步"
```

---

## 3. 数学公式与推导

### 3.1 CBOW目标函数

给定上下文词 $w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}$，预测中心词 $w_t$：

$$\max \sum_{t=1}^{T} \log P(w_t | w_{t-m:t+m, \neq t})$$

$$P(w_t | \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^T \cdot \bar{\mathbf{v}})}{\sum_{w=1}^{V}\exp(\mathbf{u}_w^T \cdot \bar{\mathbf{v}})}$$

其中 $\bar{\mathbf{v}} = \frac{1}{2m}\sum_{j \neq t}\mathbf{v}_{w_j}$ 是上下文词向量的均值。

### 3.2 Skip-Gram目标函数

$$\max \sum_{t=1}^{T}\sum_{-m \leq j \leq m, j \neq 0}\log P(w_{t+j}|w_t)$$

$$P(w_O|w_I) = \frac{\exp(\mathbf{u}_{w_O}^T \cdot \mathbf{v}_{w_I})}{\sum_{w=1}^{V}\exp(\mathbf{u}_w^T \cdot \mathbf{v}_{w_I})}$$

### 3.3 负采样（Negative Sampling）

Softmax分母计算量太大（遍历所有词），用负采样近似：

$$\log \sigma(\mathbf{u}_{w_O}^T \mathbf{v}_{w_I}) + \sum_{k=1}^{K}\mathbb{E}_{w_k \sim P(w)}[\log \sigma(-\mathbf{u}_{w_k}^T \mathbf{v}_{w_I})]$$

- 第一项：正样本（真实上下文词）
- 第二项：K个负样本（噪声词）
- $\sigma$ 是sigmoid函数

---

## 7. 调库实现

```python
"""
Word2Vec 完整实现 + Item2Vec推荐应用
"""
from gensim.models import Word2Vec
import numpy as np

# ============================================================
# 1. 基本Word2Vec
# ============================================================
# 模拟文本数据
sentences = [
    ["推荐", "系统", "算法", "工程师", "需要", "机器", "学习"],
    ["深度", "学习", "在", "推荐", "系统", "中", "广泛", "应用"],
    ["协同", "过滤", "是", "推荐", "系统", "的", "经典", "算法"],
    ["深度", "学习", "神经网络", "用于", "图像", "识别"],
    ["机器", "学习", "算法", "包括", "监督", "学习", "和", "无监督", "学习"],
]

model = Word2Vec(
    sentences=sentences,
    vector_size=64,     # 向量维度
    window=3,           # 上下文窗口大小
    min_count=1,        # 最小词频
    sg=1,               # 1=Skip-Gram, 0=CBOW
    negative=5,         # 负采样数
    epochs=100
)

# 查看词向量
print("词向量维度:", model.wv["推荐"].shape)
print("与'推荐'最相似的词:")
for word, sim in model.wv.most_similar("推荐", topn=5):
    print(f"  {word}: {sim:.4f}")

# ============================================================
# 2. Item2Vec: 把Word2Vec用于推荐
# ============================================================
# 模拟用户点击序列（每个数字是物品ID）
user_sequences = [
    ["item_1", "item_5", "item_3", "item_8", "item_2"],
    ["item_2", "item_5", "item_9", "item_1", "item_7"],
    ["item_3", "item_8", "item_1", "item_5", "item_2"],
    ["item_4", "item_6", "item_7", "item_9", "item_3"],
    ["item_1", "item_2", "item_5", "item_3", "item_8"],
]

item2vec = Word2Vec(
    sentences=user_sequences,
    vector_size=32,
    window=3,
    min_count=1,
    sg=1,
    epochs=100
)

# 用Item2Vec做推荐
print("\n=== Item2Vec 推荐 ===")
target_item = "item_5"
print(f"与 {target_item} 最相似的物品:")
for item, sim in item2vec.wv.most_similar(target_item, topn=3):
    print(f"  {item}: {sim:.4f}")

# 新用户浏览了 item_1, item_3，推荐相似物品
user_history = ["item_1", "item_3"]
user_vec = np.mean([item2vec.wv[i] for i in user_history], axis=0)

all_items = [f"item_{i}" for i in range(1, 10)]
recommendations = []
for item in all_items:
    if item not in user_history:
        sim = np.dot(user_vec, item2vec.wv[item]) / (
            np.linalg.norm(user_vec) * np.linalg.norm(item2vec.wv[item])
        )
        recommendations.append((item, sim))

recommendations.sort(key=lambda x: -x[1])
print(f"\n用户历史: {user_history}")
print("推荐结果:")
for item, sim in recommendations[:3]:
    print(f"  {item}: {sim:.4f}")
```

---

## 12. 学习总结

1. **Word2Vec核心**：通过上下文预测（或被预测）学习词向量
2. **CBOW vs Skip-Gram**：CBOW快但精度稍低，Skip-Gram慢但精度高
3. **负采样**：解决Softmax计算瓶颈
4. **Item2Vec**：将Word2Vec思想迁移到推荐系统，把物品当作"词"
5. **向量性质**：词向量能捕捉语义关系（国王-男+女≈女王）

---

## 14. 学习路径

```
TF-IDF → [当前: Word2Vec] → Item2Vec(推荐) → GloVe → BERT
```
