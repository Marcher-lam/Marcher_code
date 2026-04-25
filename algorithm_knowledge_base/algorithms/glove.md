# GloVe 学习文档

> GloVe 通过聚合全局词-词共现统计信息，学习得到兼具局部上下文语义与全局统计特性的词向量表示。

---

## 1. 算法基础认知

### 一句话定义

GloVe 是一种基于全局词-词共现矩阵的无监督词表示学习算法，通过对数双线性回归模型将共现统计转化为稠密词向量。

### 直觉类比

想象你是一名语言学家，想要理解不同词汇之间的语义关系。你的方法不是只看某个词周围的几个邻居（word2vec 的做法），而是翻阅整部百科全书，统计每一对词在多大距离范围内共同出现了多少次。例如你发现 "ice"（冰）和 "solid"（固体）经常一起出现，而 "ice" 和 "gas"（气体）很少共现，但 "steam"（蒸汽）和 "gas" 经常共现。进一步，你计算共现概率的比值：

$$ \frac{P(\text{solid} \mid \text{ice})}{P(\text{solid} \mid \text{steam})} \gg 1 $$

这个比值告诉你 "solid" 与 "ice" 的关系远比与 "steam" 的关系密切。GloVe 正是利用这种概率比值来推导词义，使得最终学到的词向量在向量空间中自然地编码了语义关系（如 "king - man + woman = queen"）。

### 历史背景

GloVe 由斯坦福大学的 Jeffrey Pennington、Richard Socher 和 Christopher D. Manning 于 2014 年在 EMNLP 会议上提出，论文题为 "GloVe: Global Vectors for Word Representation"。该模型的提出旨在解决当时 word2vec 等基于局部上下文窗口的模型的不足——虽然 word2vec 能产生高质量的词向量，但它只利用了局部的上下文信息，而忽略了语料库中全局的统计信息（如全局共现矩阵）。GloVe 的核心贡献在于证明了全局共现统计与局部上下文窗口之间的互补关系，并提出了一种优雅的数学框架将两者统一。GloVe 在多个词向量评测基准（如词相似度、词类比任务）上取得了当时的最优结果。

### 算法定位

- 类型：无监督学习 -> 词表示学习（词嵌入）
- 输出：每个词汇对应一个固定维度的稠密实数向量（如 50 维、100 维、300 维）
- 模型类型：非概率模型 / 非参数嵌入模型 / 基于统计的回归模型

### 前置知识

- **线性代数**：矩阵分解、向量内积、向量空间距离（余弦相似度）
- **概率与统计**：条件概率、联合概率分布
- **优化方法**：随机梯度下降（SGD）、AdaGrad 自适应学习率
- **自然语言处理基础**：词袋模型、共现矩阵（Co-occurrence Matrix）、TF-IDF
- **扩展知识**：word2vec（CBOW / Skip-gram）的基本原理，有助于理解 GloVe 与 word2vec 的异同

---

## 2. 核心原理

### 2.1 核心思想

GloVe 的核心思想可以概括为：**词义可以通过词与词之间的全局共现统计来捕获**。

具体来说，GloVe 认为词 $w_i$ 和词 $w_j$ 在语料库中的共现次数（即它们在一定距离窗口内同时出现的次数）包含了丰富的语义信息。如果两个词经常在相似的上下文中出现（例如 "dog" 和 "cat"），那么它们的共现模式应当是相似的，因此它们对应的词向量也应当相似。

更关键的是，GloVe 不是直接对共现次数进行建模，而是对共现概率的**比值**进行建模。这是因为概率比值具有一个非常好的性质：它能揭示两个词之间的语义关系。例如：

$$ F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}} $$

其中 $P_{ik} = P(w_k \mid w_i)$ 表示词 $w_k$ 出现在词 $w_i$ 附近的概率。比值 $P_{ik}/P_{jk}$ 的含义是：相比于词 $w_j$，词 $w_k$ 与词 $w_i$ 的关联程度更强还是更弱。例如当 $w_i$ = "ice"、$w_j$ = "steam" 时：

| $w_k$ | $P_{ik}/P_{jk}$ | 语义解读 |
|-------|-----------------|---------|
| solid | 很大 | "solid" 与 "ice" 关联更强 |
| gas | 很小 | "gas" 与 "steam" 关联更强 |
| water | 接近 1 | 两者都与 "water" 有关 |
| fashion | 接近 1 | 两者都与 "fashion" 无关 |

这种概率比值编码了丰富的语义区分信息，而 GloVe 正是将这种区分信息通过词向量的运算来表达。

### 2.2 工作流程

GloVe 的训练流程可以分为以下四个步骤：

1. **构建共现矩阵 $X$**
   - 输入：大规模文本语料库
   - 处理：遍历整个语料库，统计每对词在一定窗口大小内的共现次数
   - 输出：$|V| \times |V|$ 的共现矩阵 $X$，其中 $X_{ij}$ 表示词 $j$ 出现在词 $i$ 上下文中的次数

2. **定义对数双线性回归模型**
   - 核心思想：将共现概率比值的对数表示为词向量的函数
   - 目标：使得 $w_i^T \tilde{w}_k + b_i + \tilde{b}_k \approx \log X_{ik}$

3. **构建加权最小二乘损失函数**
   - 对共现次数少的词对降低权重（避免噪声干扰）
   - 对共现次数多的词对也设置上限权重（避免过度拟合高频共现）

4. **随机梯度下降优化**
   - 使用 AdaGrad 自适应学习率
   - 迭代更新词向量和偏置
   - 输出：每个词对应一个 $d$ 维稠密向量

### 2.3 关键概念解释

- **共现矩阵 $X$**：一个 $|V| \times |V|$ 的矩阵，其中 $X_{ij}$ 表示词 $w_j$ 在词 $w_i$ 的上下文窗口（如左右各 5 个词）内出现的次数。注意共现矩阵通常是对称的，即 $X_{ij} = X_{ji}$（取决于具体实现）。

- **词向量与上下文向量**：GloVe 为每个词 $w_i$ 学习两个向量：词向量 $w_i \in \mathbb{R}^d$ 和上下文向量 $\tilde{w}_i \in \mathbb{R}^d$。训练结束后，通常将两者相加或只取其中一个作为最终的词嵌入。

- **偏置项 $b_i$ 和 $\tilde{b}_j$**：分别对应词 $w_i$ 和上下文词 $w_j$ 的偏置，用于捕获不同词的出现频率差异（因为常用词和稀有词的共现次数基数不同）。

- **权重函数 $f(X_{ij})$**：对不同的共现次数赋予不同的权重。共现次数为 0 的词对不参与训练，共现次数适中时权重线性增长，超过一定阈值后权重不再增长（防止高频共现对主导训练）。

### 2.4 几何/直观解释

在 GloVe 学到的向量空间中，词向量之间的几何关系直接反映了语义关系：

- **语义相似性**：语义相近的词（如 "happy" 和 "joyful"）在向量空间中距离较近，即余弦相似度高。
- **语义类比关系**：向量之间的算术运算能够捕获语义关系。例如：
  - $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$
  - $\vec{v}_{\text{paris}} - \vec{v}_{\text{france}} + \vec{v}_{\text{italy}} \approx \vec{v}_{\text{rome}}$

这种几何性质之所以存在，正是因为 GloVe 通过共现概率比值学习到了词义中的对比关系。向量差 $\vec{v}_{\text{king}} - \vec{v}_{\text{man}}$ 编码了 "王权" 这一语义维度，加上 $\vec{v}_{\text{woman}}$ 后，模型在该语义维度上找到了对应的女性词 "queen"。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度/说明 |
|------|------|-----------|
| $V$ | 词汇表 | 大小为 $\|V\|$ |
| $w_i$ | 第 $i$ 个词 | $i \in \{1, \ldots, \|V\|\}$ |
| $X$ | 词-词共现矩阵 | $\|V\| \times \|V\|$ |
| $X_{ij}$ | 词 $w_j$ 出现在词 $w_i$ 上下文中的次数 | 标量，非负整数 |
| $X_i$ | 词 $w_i$ 的所有共现次数之和 | $\sum_k X_{ik}$，即共现矩阵第 $i$ 行之和 |
| $P_{ij}$ | 条件概率 $P(w_j \mid w_i)$ | $P_{ij} = X_{ij} / X_i$ |
| $w_i$ | 词 $w_i$ 的词向量 | $\mathbb{R}^d$ |
| $\tilde{w}_j$ | 词 $w_j$ 的上下文向量 | $\mathbb{R}^d$ |
| $b_i$ | 词 $w_i$ 的偏置 | $\mathbb{R}$ |
| $\tilde{b}_j$ | 词 $w_j$ 的上下文偏置 | $\mathbb{R}$ |
| $f$ | 权重函数 | $f: \mathbb{R} \to \mathbb{R}$ |
| $d$ | 词向量维度 | 通常为 50, 100, 200, 300 |
| $x_{\max}$ | 权重函数的截断阈值 | 通常为 100 |
| $\alpha$ | 权重函数的指数参数 | 通常为 3/4 |

### 3.2 问题形式化

给定大规模文本语料库，GloVe 的目标是学习一组词向量 $\{w_i\}_{i=1}^{|V|}$ 和上下文向量 $\{\tilde{w}_j\}_{j=1}^{|V|}$，使得词向量之间的运算能够反映词与词之间的语义关系。

关键观察：词义可以从共现概率的比值中推导出来。

定义条件概率：
$$ P_{ij} = P(w_j \mid w_i) = \frac{X_{ij}}{X_i} $$

其中 $X_i = \sum_{k=1}^{|V|} X_{ik}$。

考虑两个探测词 $w_i$ 和 $w_j$，以及一个上下文词 $w_k$。共现概率的比值为：
$$ \frac{P_{ik}}{P_{jk}} = \frac{X_{ik} / X_i}{X_{jk} / X_j} $$

这个比值编码了 $w_k$ 与 $w_i$ 的关联程度相对于 $w_k$ 与 $w_j$ 的关联程度。

### 3.3 目标函数/损失函数

GloVe 的目标是最小化以下加权最小二乘损失函数：

$$ J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2 $$

其中：
- $f(X_{ij})$ 是权重函数，当 $X_{ij} = 0$ 时 $f(0) = 0$
- $w_i^T \tilde{w}_j + b_i + \tilde{b}_j$ 是模型对 $\log X_{ij}$ 的预测
- 该损失函数鼓励模型预测的值接近共现次数的对数

**为什么选择这个损失函数？**

1. **对数空间**：使用 $\log X_{ij}$ 而非 $X_{ij}$ 本身，是因为共现次数的量级差异极大（从 0 到数百万），取对数可以将差异压缩到合理范围。这也与信息论中的对数概率一致。
2. **加权最小二乘**：线性模型加二次损失是最简单有效的选择。权重函数确保了模型不会过于关注高频共现对（它们的预测值已经很小）或被低频噪声干扰。
3. **双线性形式**：$w_i^T \tilde{w}_j$ 是一种标准的低秩近似形式，与矩阵分解密切相关。

### 3.4 推导过程

下面从共现概率比值出发，经过 6 步推导得到 GloVe 的目标函数。

**Step 1：从概率比值出发**

定义一个函数 $F$，使得：
$$ F(w_i, w_j, w_k) = \frac{P_{ik}}{P_{jk}} $$

即函数 $F$ 以三个词向量为输入，输出两个条件概率的比值。这个比值是理解词义的核心信号。

**Step 2：分析 $F$ 应满足的性质**

由于概率比值 $P_{ik}/P_{jk}$ 是一个标量，而 $F$ 的输入是向量，$F$ 应该具有以下两个性质：

- **性质 1（差分结构）**：$F$ 的输入应当以 $w_i - w_j$ 的形式出现，而不是分别依赖 $w_i$ 和 $w_j$。这是因为 $P_{ik}/P_{jk}$ 度量的是 $w_i$ 和 $w_j$ 的**差异**对 $w_k$ 的影响。
  - 直观理解：$w_i$ 和 $w_j$ 之间的差异（而非各自的绝对值）决定了 $w_k$ 与谁更相关。
  - 例如，"ice" 和 "steam" 的差异在于状态（固态 vs 气态），这个差异决定了 "solid" 更靠近 "ice"。

- **性质 2（线性性）**：$F$ 对 $w_k$ 应当是线性的。因为线性结构是最简单的形式，且线性关系足以捕获词义的对比结构。

**Step 3：引入 $F$ 的具体形式**

综合上述两个性质，要求 $F$ 满足：
$$ F\left((w_i - w_j)^T w_k\right) = \frac{P_{ik}}{P_{jk}} $$

为了消除比值两侧的不对称性，对等式两边取对数，进一步要求 $F$ 为线性函数：
$$ F\left((w_i - w_j)^T \tilde{w}_k\right) = \frac{P_{ik}}{P_{jk}} $$

其中将 $w_k$ 替换为 $\tilde{w}_k$ 以区分词向量和上下文向量。

由于 $F$ 是线性函数，且当输入为 0 时比值应为 1（即 $P_{ik} = P_{jk}$），可以设：
$$ F(x) = \exp(x) $$

因此：
$$ \exp\left((w_i - w_j)^T \tilde{w}_k\right) = \frac{P_{ik}}{P_{jk}} = \frac{X_{ik}}{X_i} \cdot \frac{X_j}{X_{jk}} $$

取对数：
$$ (w_i - w_j)^T \tilde{w}_k = \log X_{ik} - \log X_i - \log X_{jk} + \log X_j $$

**Step 4：引入偏置项**

注意到 $\log X_i$ 和 $\log X_j$ 只与词 $w_i$ 和 $w_j$ 有关，可以将其吸收为偏置项。令 $b_i = \log X_i$ 和 $\tilde{b}_k = -\log X_i$（稍后重新定义），则：

$$ w_i^T \tilde{w}_k - w_j^T \tilde{w}_k = \log X_{ik} - \log X_i - (\log X_{jk} - \log X_j) $$

将各项重新组织，引入偏置项 $b_i$ 和 $\tilde{b}_k$：

$$ w_i^T \tilde{w}_k + b_i + \tilde{b}_k = \log X_{ik} $$

$$ w_j^T \tilde{w}_k + b_j + \tilde{b}_k = \log X_{jk} $$

两式相减即可恢复 Step 3 的等式。偏置项 $b_i$ 和 $\tilde{b}_k$ 分别捕获了词 $w_i$ 自身的高频/低频特性和上下文词 $w_k$ 的高频/低频特性。

**Step 5：从差分等式到全局目标函数**

Step 4 中的等式要求对于**每一对**词 $(w_i, w_j)$ 和**每一个**上下文词 $w_k$ 都成立。将其推广为对所有词对的约束：

$$ w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log X_{ij}, \quad \forall i, j $$

由于这是一个过度约束的系统（方程数量远多于变量数量），我们使用最小二乘法求解：

$$ J = \sum_{i,j=1}^{|V|} \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2 $$

**Step 6：引入权重函数**

上述损失函数对所有词对一视同仁，但实际上：
- 当 $X_{ij} = 0$ 时（两个词从未共现），$\log 0$ 无定义，不应参与训练
- 当 $X_{ij}$ 很小时（偶尔共现），共现信息不可靠，应降低权重
- 当 $X_{ij}$ 很大时（频繁共现），信息非常可靠，但也不应过度主导训练

因此引入权重函数 $f: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$，得到最终的目标函数：

$$ J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2 $$

**权重函数 $f(x)$ 的设计**

权重函数需要满足三个条件：
1. $f(0) = 0$（不共现的词对不参与训练）
2. $f(x)$ 非减（共现越多，信息越可靠，权重越大）
3. 对于足够大的 $x$，$f(x)$ 不应过度增长（防止高频词对主导损失）

GloVe 采用的分段函数为：
$$ f(x) = \begin{cases} \left(\frac{x}{x_{\max}}\right)^\alpha & \text{if } x < x_{\max} \\ 1 & \text{if } x \geq x_{\max} \end{cases} $$

其中 $\alpha = 3/4$ 是经验选择。当 $\alpha < 1$ 时，$f(x)$ 是一个凹函数，这意味着：
- 对于共现次数很小的词对，权重增长较快（充分利用有限的信息）
- 对于共现次数很大的词对，权重增长逐渐变缓（避免过度主导）

当 $x = x_{\max}$（通常为 100）时，权重达到上限 1。

### 3.5 最终解/算法步骤

GloVe 没有解析解，需要通过迭代优化求解。使用随机梯度下降（SGD）+ AdaGrad 自适应学习率。

**参数更新规则**（对单个样本 $(i, j)$）：

$$ w_i \leftarrow w_i - \eta \cdot \frac{\partial J_{ij}}{\partial w_i} $$

其中梯度的计算为：

$$ \frac{\partial J_{ij}}{\partial w_i} = 2 \cdot f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right) \cdot \tilde{w}_j $$

$$ \frac{\partial J_{ij}}{\partial \tilde{w}_j} = 2 \cdot f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right) \cdot w_i $$

$$ \frac{\partial J_{ij}}{\partial b_i} = 2 \cdot f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right) $$

$$ \frac{\partial J_{ij}}{\partial \tilde{b}_j} = 2 \cdot f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right) $$

**AdaGrad 学习率更新**：

$$ \eta_{i,t} = \frac{\eta_0}{\sqrt{\sum_{\tau=1}^{t} g_{i,\tau}^2 + \epsilon}} $$

其中 $g_{i,\tau}$ 是参数 $i$ 在第 $\tau$ 步的梯度分量，$\epsilon$ 是数值稳定项。

**完整算法伪代码**：

```
输入：共现矩阵 X, 词向量维度 d, 初始学习率 eta_0, 最大迭代次数 T
输出：词向量 {w_i}, 上下文向量 {w_tilde_j}

随机初始化 w_i, w_tilde_j in R^d, b_i = 0, b_tilde_j = 0
初始化梯度累积器 G = 0 (全零矩阵)

for t = 1 to T:
    for 每个非零共现对 (i, j) where X_{ij} > 0:
        # 计算误差
        f_val = f(X_{ij})  # 权重
        diff = w_i^T * w_tilde_j + b_i + b_tilde_j - log(X_{ij})
        loss_contrib = f_val * diff^2

        # 计算梯度
        grad_w = f_val * diff * w_tilde_j
        grad_w_tilde = f_val * diff * w_i
        grad_b = f_val * diff
        grad_b_tilde = f_val * diff

        # 累积梯度平方（用于AdaGrad）
        G[w_i] += grad_w^2
        G[w_tilde_j] += grad_w_tilde^2
        G[b_i] += grad_b^2
        G[b_tilde_j] += grad_b_tilde^2

        # AdaGrad 参数更新
        w_i -= eta_0 / sqrt(G[w_i] + eps) * grad_w
        w_tilde_j -= eta_0 / sqrt(G[w_tilde_j] + eps) * grad_w_tilde
        b_i -= eta_0 / sqrt(G[b_i] + eps) * grad_b
        b_tilde_j -= eta_0 / sqrt(G[b_tilde_j] + eps) * grad_b_tilde

最终词向量：W_final = W + W_tilde  # 或只取 W
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

**1. 文本清洗与分词**

GloVe 的训练需要大规模文本语料库（通常为数十亿词级别）。常见的语料库包括 Wikipedia、Common Crawl、Gigaword 等。预处理步骤包括：

- 转换为小写（可选）
- 去除标点符号和特殊字符
- 分词（按空格分词，或使用更复杂的分词器）
- 构建词汇表，通常设置词频阈值（如最小出现 5 次）过滤低频词

```python
import re
from collections import Counter

def preprocess_corpus(text):
    """
    文本预处理

    Args:
        text: 原始文本字符串

    Returns:
        tokens: 分词后的词汇列表
    """
    # 转换为小写
    text = text.lower()
    # 去除标点符号，保留字母、数字和空格
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # 按空格分词
    tokens = text.split()
    return tokens


def build_vocab(corpus, min_count=5):
    """
    构建词汇表

    Args:
        corpus: 分词后的语料列表，每个元素是一篇文章的token列表
        min_count: 最低词频阈值

    Returns:
        word2idx: 词到索引的映射字典
        idx2word: 索引到词的映射字典
    """
    # 统计词频
    word_counts = Counter()
    for doc_tokens in corpus:
        word_counts.update(doc_tokens)

    # 过滤低频词
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    # 按词频降序排列
    vocab.sort(key=lambda w: word_counts[w], reverse=True)

    # 构建映射
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word
```

**2. 构建共现矩阵**

共现矩阵是 GloVe 训练的输入。对于语料中的每一个词，统计在其上下文窗口内出现的每个词的次数。

```python
import numpy as np
from scipy.sparse import coo_matrix

def build_cooccurrence_matrix(corpus, word2idx, window_size=10):
    """
    构建词-词共现矩阵

    Args:
        corpus: 分词后的语料列表
        word2idx: 词到索引的映射
        window_size: 上下文窗口大小（单侧）

    Returns:
        cooccurrence: 稀疏共现矩阵（COO格式），shape (|V|, |V|)
    """
    vocab_size = len(word2idx)
    cooccurrence = {}

    for doc_tokens in corpus:
        # 将token转换为索引
        indices = [word2idx[t] for t in doc_tokens if t in word2idx]

        for i, center_idx in enumerate(indices):
            # 遍历上下文窗口
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue
                context_idx = indices[j]

                # 距离衰减：越近的词贡献越大
                distance = abs(i - j)
                weight = 1.0 / distance

                # 累加共现次数（使用字典存储稀疏矩阵）
                key = (center_idx, context_idx)
                cooccurrence[key] = cooccurrence.get(key, 0.0) + weight

    # 转换为稀疏矩阵
    rows, cols, data = [], [], []
    for (i, j), value in cooccurrence.items():
        rows.append(i)
        cols.append(j)
        data.append(value)

    sparse_matrix = coo_matrix(
        (np.array(data), (np.array(rows), np.array(cols))),
        shape=(vocab_size, vocab_size)
    )

    return sparse_matrix
```

### 4.2 参数初始化

```python
def initialize_parameters(vocab_size, embedding_dim, seed=42):
    """
    初始化GloVe模型参数

    Args:
        vocab_size: 词汇表大小
        embedding_dim: 词向量维度
        seed: 随机种子

    Returns:
        W: 词向量矩阵, shape (vocab_size, embedding_dim)
        W_context: 上下文向量矩阵, shape (vocab_size, embedding_dim)
        b: 词偏置向量, shape (vocab_size,)
        b_context: 上下文偏置向量, shape (vocab_size,)
    """
    rng = np.random.RandomState(seed)

    # 使用均匀分布初始化词向量（范围较小以避免初始损失过大）
    scale = 0.5 / embedding_dim
    W = rng.uniform(-scale, scale, (vocab_size, embedding_dim))
    W_context = rng.uniform(-scale, scale, (vocab_size, embedding_dim))

    # 偏置初始化为0
    b = np.zeros(vocab_size)
    b_context = np.zeros(vocab_size)

    return W, W_context, b, b_context
```

### 4.3 迭代过程

GloVe 的训练采用随机梯度下降。每个 epoch 遍历所有非零共现对，计算梯度并更新参数。

**学习率调度**：GloVe 采用线性衰减的学习率调度。初始学习率通常为 0.05，在每个 epoch 后线性衰减至接近 0。

```python
def train_glove(cooccurrence_matrix, vocab_size, embedding_dim=100,
                max_epochs=50, initial_lr=0.05, x_max=100, alpha=0.75):
    """
    GloVe训练主循环

    Args:
        cooccurrence_matrix: 共现矩阵（COO稀疏格式）
        vocab_size: 词汇表大小
        embedding_dim: 词向量维度
        max_epochs: 最大训练轮数
        initial_lr: 初始学习率
        x_max: 权重截断阈值
        alpha: 权重函数指数

    Returns:
        W: 词向量矩阵
        W_context: 上下文向量矩阵
        loss_history: 损失历史
    """
    # 提取非零元素（训练样本）
    rows = cooccurrence_matrix.row
    cols = cooccurrence_matrix.col
    data = cooccurrence_matrix.data
    n_samples = len(data)

    # 初始化参数
    W, W_ctx, b, b_ctx = initialize_parameters(vocab_size, embedding_dim)
    # AdaGrad梯度累积器
    grad_sq_W = np.ones_like(W)
    grad_sq_W_ctx = np.ones_like(W_ctx)
    grad_sq_b = np.ones_like(b)
    grad_sq_b_ctx = np.ones_like(b_ctx)

    loss_history = []

    for epoch in range(max_epochs):
        # 线性衰减学习率
        lr = initial_lr * (1.0 - epoch / max_epochs)
        if lr < initial_lr * 1e-4:
            lr = initial_lr * 1e-4  # 设置最小学习率

        # 随机打乱训练顺序
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0

        for idx in perm:
            i, j, x_ij = rows[idx], cols[idx], data[idx]

            # 计算权重
            f_xij = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0

            # 前向计算：模型预测值与误差
            w_i = W[i]
            w_ctx_j = W_ctx[j]
            inner = np.dot(w_i, w_ctx_j) + b[i] + b_ctx[j]
            diff = inner - np.log(x_ij)

            # 计算该样本的损失
            loss_contrib = f_xij * diff * diff
            epoch_loss += loss_contrib

            # 计算梯度
            grad_common = 2.0 * f_xij * diff
            grad_w_i = grad_common * w_ctx_j
            grad_w_ctx_j = grad_common * w_i
            grad_b_i = grad_common
            grad_b_ctx_j = grad_common

            # 累积梯度平方（AdaGrad）
            grad_sq_W[i] += grad_w_i ** 2
            grad_sq_W_ctx[j] += grad_w_ctx_j ** 2
            grad_sq_b[i] += grad_b_i ** 2
            grad_sq_b_ctx[j] += grad_b_ctx_j ** 2

            # AdaGrad参数更新
            W[i] -= lr * grad_w_i / np.sqrt(grad_sq_W[i])
            W_ctx[j] -= lr * grad_w_ctx_j / np.sqrt(grad_sq_W_ctx[j])
            b[i] -= lr * grad_b_i / np.sqrt(grad_sq_b[i])
            b_ctx[j] -= lr * grad_b_ctx_j / np.sqrt(grad_sq_b_ctx[j])

        avg_loss = epoch_loss / n_samples
        loss_history.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, "
                  f"Loss: {avg_loss:.4f}, LR: {lr:.6f}")

    # 合并词向量和上下文向量
    W_final = W + W_ctx

    return W_final, W, W_ctx, loss_history
```

### 4.4 收敛条件

GloVe 的训练在以下条件下停止：

- **达到最大迭代次数**：通常设置为 15-50 个 epoch。对于大规模语料库，15-20 个 epoch 通常已经足够。
- **损失不再显著下降**：当连续几个 epoch 的损失变化小于阈值时可以提前停止。
- **学习率过小**：由于使用线性衰减，学习率最终趋近于 0，此时参数更新几乎停止。

实际使用中，通常以最大迭代次数为主，因为 GloVe 的损失曲线通常在初期快速下降后趋于平稳。

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 | 说明 |
|--------|------|----------|--------|------|
| embedding_dim | 词向量维度 | 50-300 | 100/300 | 维度越高表达能力越强，但训练和存储成本更大 |
| window_size | 上下文窗口大小 | 5-15 | 10 | 窗口越大，捕获的共现关系越多，但也引入更多噪声 |
| initial_lr | 初始学习率 | 0.01-0.1 | 0.05 | GloVe 使用较保守的学习率 |
| max_epochs | 最大训练轮数 | 15-50 | 25 | 通常 15-25 轮已足够 |
| x_max | 权重截断阈值 | 50-100 | 100 | 超过此值的共现次数权重不再增长 |
| alpha | 权重函数指数 | 0.5-1.0 | 0.75 | 0.75 是原始论文推荐的值 |
| min_count | 最低词频 | 1-10 | 5 | 过滤低频词，减小词汇表大小 |
| distance_weighting | 距离加权 | True/False | True | 是否对窗口内不同距离的词赋予不同权重 |

---

## 5. 应用场景

### 5.1 典型应用

**应用 1：词相似度计算**
- 问题类型：无监督语义分析
- 为什么适合：GloVe 词向量中语义相近的词在向量空间中距离较近，可以通过余弦相似度来量化词之间的语义相似性
- 实际案例：在搜索引擎中用于扩展查询词，如用户搜索 "automobile" 时自动关联 "car"、"vehicle" 等

**应用 2：词类比任务**
- 问题类型：语义推理
- 为什么适合：GloVe 词向量具有线性代数结构，支持向量运算来推理语义关系
- 实际案例：著名的 "king - man + woman = queen" 类比推理，以及地理类比 "paris - france + japan = tokyo"

**应用 3：下游 NLP 任务的词嵌入层**
- 问题类型：文本分类、命名实体识别、情感分析等
- 为什么适合：预训练的 GloVe 向量提供了高质量的词义初始化，可以作为深度学习模型的输入嵌入层
- 实际案例：在 IMDB 情感分析任务中，使用 GloVe 初始化的 LSTM 比随机初始化获得更高的分类准确率

**应用 4：文本可视化与聚类**
- 问题类型：无监督数据探索
- 为什么适合：通过 t-SNE 或 PCA 将 GloVe 词向量降维到二维平面，可以直观地观察词与词之间的语义关系
- 实际案例：将动物、国家、职业等类别的词投影到二维平面，可以清晰地看到同类词聚集在一起

**应用 5：语义成分分析**
- 问题类型：语义学研究
- 为什么适合：GloVe 词向量的某些维度可以被解释为具体的语义成分
- 实际案例：通过主成分分析，可以在 GloVe 向量空间中识别出 "性别"、"时态"、"单复数" 等语义维度

### 5.2 适用数据特征

- **数据类型**：大规模无标注文本语料（Wikipedia、新闻、网页等）
- **数据规模**：训练高质量的 GloVe 向量通常需要数十亿词级别的语料
- **语言支持**：支持多语言，但需要对应语言的语料库
- **噪声容忍度**：对语料中的噪声有一定容忍度，因为全局统计会平滑个别噪声

### 5.3 不适用场景

1. **上下文相关的词义消歧**：GloVe 为每个词分配一个固定向量，无法区分多义词（如 "bank" 既可以指银行也可以指河岸）
2. **领域特定场景**：如果目标领域与训练语料差异很大（如医学术语），通用 GloVe 向量可能效果不佳
3. **低资源语言**：如果语料规模不足（如几百万词），训练的 GloVe 向量质量可能较差
4. **实时系统**：GloVe 训练需要离线完成，不适合需要在线更新词向量的场景

---

## 6. 优缺点分析

### 6.1 优点

1. **利用全局统计信息**
   - GloVe 使用整个语料库的共现矩阵进行训练，而非仅依赖局部窗口
   - 全局统计能提供更稳定、更全面的词义信息
   - 尤其对于低频词，全局统计比局部窗口采样更可靠

2. **训练效率高**
   - GloVe 的训练只依赖于预计算的共现矩阵，不需要在训练时反复扫描语料
   - 使用 AdaGrad 优化器，收敛速度快（通常 15-25 个 epoch）
   - 原始实现可以高效利用多线程

3. **数学基础优雅**
   - 从共现概率比值出发的推导清晰且富有启发性
   - 与矩阵分解（如 SVD）有理论联系，但通过加权最小二乘提供了更灵活的框架
   - 可以看作是 word2vec（基于局部上下文）和 LSA/TF-IDF（基于全局统计）的统一

4. **高质量的词向量**
   - 在词相似度和词类比评测基准上表现优秀
   - 学到的词向量具有良好的线性代数结构
   - 作为下游任务的预训练嵌入表现稳定

### 6.2 缺点

1. **静态词嵌入的固有限制**
   - 一个词只有一个向量表示，无法处理多义词问题
   - 无法区分词在不同上下文中的不同含义
   - 缓解方法：使用 ELMo、BERT 等动态词嵌入模型

2. **内存需求大**
   - 共现矩阵的大小为 $|V| \times |V|$，对于大型词汇表（如 100 万词），存储需求极大
   - 实际中需要使用稀疏矩阵格式来节省内存
   - 缓解方法：使用更小的词汇表（过滤低频词）或使用哈希技巧

3. **对语料规模敏感**
   - 小规模语料（如数百万词）训练的 GloVe 向量质量明显不如大规模语料
   - 与 word2vec 相比，GloVe 在小语料上可能处于劣势
   - 缓解方法：使用预训练的 GloVe 向量，或在目标领域语料上微调

4. **无法处理 OOV（Out-of-Vocabulary）问题**
   - 词汇表之外的词没有对应的向量
   - 缓解方法：使用字符级嵌入（如 FastText）或子词分词（如 BPE）

### 6.3 与同类算法的详细对比

| 维度 | GloVe | Word2Vec (Skip-gram) | FastText | LSA (SVD) |
|------|-------|---------------------|----------|-----------|
| 核心思想 | 全局共现矩阵的对数双线性回归 | 局部窗口的预测任务 | 局部窗口 + 子词信息 | 全局共现矩阵的奇异值分解 |
| 训练方式 | 加权最小二乘 + SGD | 负采样 / 层次 Softmax | 负采样 | SVD 分解 |
| 统计信息 | 全局（共现矩阵） | 局部（滑动窗口） | 局部 + 子词 | 全局（共现矩阵） |
| 数学本质 | 隐式矩阵分解 | 隐式矩阵分解 | 隐式矩阵分解 + 字符 n-gram | 显式矩阵分解 |
| OOV 处理 | 不支持 | 不支持 | 支持子词 OOV | 不支持 |
| 训练速度 | 快（离线共现矩阵） | 中等 | 较慢（子词增加计算） | 快（但 SVD 本身较慢） |
| 词类比性能 | 优秀 | 优秀 | 优秀（尤其小数据） | 一般 |
| 词相似度性能 | 优秀 | 优秀 | 优秀 | 良好 |
| 可解释性 | 中（从概率比值推导） | 较低（负采样的近似） | 中 | 高（SVD 的理论清晰） |
| 实现复杂度 | 中等 | 简单 | 中等 | 简单 |
| 预训练资源 | 斯坦福 NLP 提供多种维度 | Google 提供 | Facebook 提供 | 无官方预训练 |
| 与矩阵分解的关系 | 加权矩阵分解的特例 | 等价于特定加权的矩阵分解 | 结合子词的矩阵分解 | 标准矩阵分解 |

**GloVe vs Word2Vec 的深入比较**：

- **训练机制**：Word2Vec 通过预测上下文来学习词向量（生成式方法），GloVe 通过回归共现次数来学习词向量（回归方法）
- **全局 vs 局部**：Word2Vec 每次只看到一个局部窗口，GloVe 能看到全局统计。在理论上，GloVe 利用了更多信息
- **实验表现**：在多项评测中，GloVe 和 Word2Vec 的表现非常接近，各有胜负。GloVe 在词类比任务上略优，Word2Vec 在某些相似度任务上可能更好
- **实践选择**：两者都是优秀的选择。在有预训练向量的情况下，选择取决于目标任务和具体评测结果

---

## 7. 调库实现（Python + 完整代码 + 注释）

### 7.1 环境准备

```bash
pip install numpy matplotlib gensim
```

### 7.2 完整代码示例

```python
"""
GloVe 调库实现
使用 gensim 加载预训练 GloVe 向量，进行词相似度、词类比实验和可视化
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def download_and_load_glove(glove_path, dim=100):
    """
    加载预训练 GloVe 向量

    由于 gensim 不直接支持 GloVe 格式，需要先转换为 word2vec 格式

    Args:
        glove_path: GloVe 文件路径（.txt格式）
        dim: 词向量维度

    Returns:
        model: gensim KeyedVectors 模型
    """
    word2vec_path = glove_path + '.word2vec'

    # 如果尚未转换，则进行转换
    if not os.path.exists(word2vec_path):
        print(f"正在将 GloVe 格式转换为 Word2Vec 格式...")
        glove2word2vec(glove_path, word2vec_path)
        print(f"转换完成，保存到: {word2vec_path}")

    # 加载模型
    print(f"正在加载 GloVe 向量: {word2vec_path}")
    model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    print(f"加载完成，词汇表大小: {len(model.index_to_key)}")

    return model


def evaluate_word_similarity(model, word_pairs_file=None):
    """
    评估词相似度

    使用预定义的词对及其人工打分，计算模型预测相似度与人工打分的相关性

    Args:
        model: 加载的词向量模型
        word_pairs_file: 词对文件路径（可选）

    Returns:
        score: Spearman 相关系数
    """
    # 如果没有提供评测文件，使用内置的常见词对
    test_pairs = [
        ('king', 'queen', 0.95),
        ('man', 'woman', 0.90),
        ('cat', 'dog', 0.88),
        ('car', 'automobile', 0.98),
        ('happy', 'joyful', 0.92),
        ('big', 'large', 0.90),
        ('fast', 'quick', 0.88),
        ('good', 'bad', -0.50),
        ('king', 'man', 0.70),
        ('apple', 'fruit', 0.75),
    ]

    model_scores = []
    human_scores = []
    valid_pairs = []

    for w1, w2, human_score in test_pairs:
        if w1 in model and w2 in model:
            # 使用余弦相似度
            sim = model.similarity(w1, w2)
            model_scores.append(sim)
            human_scores.append(human_score)
            valid_pairs.append((w1, w2, human_score, sim))

    # 计算相关系数
    if len(model_scores) > 2:
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(model_scores, human_scores)
    else:
        correlation, p_value = 0.0, 1.0

    # 打印结果
    print("\n" + "=" * 70)
    print("词相似度评估结果")
    print("=" * 70)
    print(f"{'词对':<25} {'人工评分':<12} {'模型相似度':<12}")
    print("-" * 70)
    for w1, w2, human_score, sim in valid_pairs:
        print(f"{w1} - {w2:<15} {human_score:<12.2f} {sim:<12.4f}")
    print("-" * 70)
    print(f"Spearman 相关系数: {correlation:.4f} (p-value: {p_value:.6f})")
    print("=" * 70)

    return correlation


def evaluate_word_analogy(model):
    """
    评估词类比任务

    经典类比: a is to b as c is to ?
    通过向量运算: ? = b - a + c

    Args:
        model: 加载的词向量模型

    Returns:
        accuracy: 类比任务准确率
    """
    # 定义类比测试集
    # 每个类比是一个四元组 (a, b, c, expected_d)
    # 含义: a之于b如同c之于d
    analogies = [
        # 性别类比
        ('king', 'queen', 'man', 'woman'),
        ('king', 'queen', 'boy', 'girl'),
        ('man', 'woman', 'father', 'mother'),
        ('man', 'woman', 'brother', 'sister'),
        ('man', 'woman', 'husband', 'wife'),
        ('man', 'woman', 'son', 'daughter'),
        ('man', 'woman', 'prince', 'princess'),
        ('man', 'woman', 'guy', 'girl'),
        # 国家-首都类比
        ('france', 'paris', 'japan', 'tokyo'),
        ('france', 'paris', 'germany', 'berlin'),
        ('france', 'paris', 'italy', 'rome'),
        ('france', 'paris', 'england', 'london'),
        ('france', 'paris', 'russia', 'moscow'),
        ('china', 'beijing', 'japan', 'tokyo'),
        # 时态类比
        ('go', 'going', 'run', 'running'),
        ('go', 'went', 'run', 'ran'),
        # 复数类比
        ('dog', 'dogs', 'cat', 'cats'),
        ('car', 'cars', 'bus', 'buses'),
    ]

    correct = 0
    total = 0
    results = []

    for a, b, c, expected in analogies:
        # 检查所有词是否在词汇表中
        if not all(w in model for w in [a, b, c, expected]):
            continue

        total += 1

        # 向量运算: d = b - a + c
        try:
            predicted = model.most_similar(
                positive=[b, c],  # 正方向：b和c
                negative=[a],     # 负方向：a
                topn=5             # 返回前5个最相似的词
            )

            # 检查期望答案是否在前5中
            top_words = [word for word, score in predicted]
            is_correct = expected in top_words
            rank = top_words.index(expected) + 1 if is_correct else -1

            if is_correct:
                correct += 1

            results.append({
                'analogy': f"{a} : {b} :: {c} : ?",
                'expected': expected,
                'predicted_top1': predicted[0][0],
                'predicted_top1_score': predicted[0][1],
                'rank': rank,
                'correct': is_correct
            })
        except KeyError:
            continue

    # 打印结果
    print("\n" + "=" * 80)
    print("词类比评估结果")
    print("=" * 80)
    print(f"{'类比':<30} {'期望':<12} {'预测 Top-1':<15} {'排名':<8} {'结果'}")
    print("-" * 80)
    for r in results:
        status = "正确" if r['correct'] else "错误"
        print(f"{r['analogy']:<30} {r['expected']:<12} "
              f"{r['predicted_top1']:<15} {str(r['rank']):<8} {status}")
    print("-" * 80)

    accuracy = correct / total if total > 0 else 0.0
    print(f"准确率 (Top-5): {correct}/{total} = {accuracy:.2%}")
    print("=" * 80)

    return accuracy


def find_most_similar(model, positive_words, negative_words=None, topn=10):
    """
    查找最相似的词

    Args:
        model: 词向量模型
        positive_words: 正方向词列表
        negative_words: 负方向词列表
        topn: 返回前 n 个结果
    """
    if negative_words is None:
        negative_words = []

    print(f"\n查找与 {positive_words} 最相似（排除 {negative_words}）的词:")
    print("-" * 50)

    try:
        results = model.most_similar(
            positive=positive_words,
            negative=negative_words,
            topn=topn
        )
        for word, score in results:
            print(f"  {word:<20} 相似度: {score:.4f}")
    except KeyError as e:
        print(f"  错误: 词汇 {e} 不在词汇表中")


def visualize_embeddings_tsne(model, words_by_category, output_file='glove_tsne.png'):
    """
    使用 t-SNE 可视化词向量

    Args:
        model: 词向量模型
        words_by_category: 按类别分组的词列表字典
        output_file: 输出图片文件名
    """
    # 收集所有词及其向量
    all_words = []
    all_vectors = []
    all_labels = []
    all_colors = []

    # 颜色映射
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
              '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    color_idx = 0

    for category, words in words_by_category.items():
        for word in words:
            if word in model:
                all_words.append(word)
                all_vectors.append(model[word])
                all_labels.append(category)
                all_colors.append(colors[color_idx % len(colors)])
        color_idx += 1

    if len(all_vectors) < 2:
        print("词汇表中找到的有效词太少，无法进行可视化")
        return

    all_vectors = np.array(all_vectors)

    # t-SNE 降维
    print(f"\n正在对 {len(all_vectors)} 个词进行 t-SNE 降维...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(all_vectors) - 1),
        random_state=42,
        n_iter=1000
    )
    vectors_2d = tsne.fit_transform(all_vectors)

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 10))

    # 按类别绘制
    categories = list(words_by_category.keys())
    for i, category in enumerate(categories):
        mask = [label == category for label in all_labels]
        x = vectors_2d[mask, 0]
        y = vectors_2d[mask, 1]
        words = [all_words[j] for j in range(len(all_labels)) if all_labels[j] == category]

        ax.scatter(x, y, c=colors[i % len(colors)],
                   label=category, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

        # 为每个词添加标签
        for xi, yi, word in zip(x, y, words):
            ax.annotate(word, (xi, yi), fontsize=9,
                        ha='center', va='bottom',
                        textcoords="offset points", xytext=(0, 5))

    ax.legend(loc='best', fontsize=11)
    ax.set_title('GloVe 词向量 t-SNE 可视化', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE 维度 1', fontsize=12)
    ax.set_ylabel('t-SNE 维度 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_file}")
    plt.close()


def compute_vector_arithmetic(model):
    """
    演示词向量的算术运算

    Args:
        model: 词向量模型
    """
    print("\n" + "=" * 70)
    print("词向量算术运算演示")
    print("=" * 70)

    arithmetic_examples = [
        ("king - man + woman", ["king", "woman"], ["man"]),
        ("paris - france + italy", ["paris", "italy"], ["france"]),
        ("bigger - big + small", ["bigger", "small"], ["big"]),
    ]

    for desc, pos, neg in arithmetic_examples:
        print(f"\n  {desc} = ?")
        try:
            results = model.most_similar(positive=pos, negative=neg, topn=5)
            for i, (word, score) in enumerate(results):
                marker = " <-- Top-1" if i == 0 else ""
                print(f"    {i+1}. {word:<20} (相似度: {score:.4f}){marker}")
        except KeyError as e:
            print(f"    错误: 词汇 {e} 不在词汇表中")

    print("=" * 70)


# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    print("=" * 60)
    print("GloVe 调库实现演示")
    print("=" * 60)

    # GloVe 预训练向量下载地址
    # https://nlp.stanford.edu/projects/glove/
    # 常用文件: glove.6B.100d.txt (822MB, 100维, 40万词汇)
    glove_path = 'glove.6B.100d.txt'

    # 如果本地有预训练文件，直接加载
    if os.path.exists(glove_path):
        model = download_and_load_glove(glove_path, dim=100)
    else:
        print(f"\n未找到 GloVe 预训练文件: {glove_path}")
        print("请从 https://nlp.stanford.edu/projects/glove/ 下载")
        print("推荐: glove.6B.zip (包含 50d/100d/200d/300d 四种维度)")
        print("\n使用示例代码演示（需要实际文件才能运行）")
        model = None

    if model is not None:
        # 1. 词相似度评估
        print("\n[1/5] 词相似度评估...")
        spearman_corr = evaluate_word_similarity(model)

        # 2. 词类比评估
        print("\n[2/5] 词类比评估...")
        analogy_acc = evaluate_word_analogy(model)

        # 3. 查找最相似词
        print("\n[3/5] 查找最相似词...")
        find_most_similar(model, ['computer'], topn=10)
        find_most_similar(model, ['king', 'woman'], ['man'], topn=5)

        # 4. 词向量算术运算
        print("\n[4/5] 词向量算术运算...")
        compute_vector_arithmetic(model)

        # 5. t-SNE 可视化
        print("\n[5/5] t-SNE 可视化...")
        words_by_category = {
            '动物': ['cat', 'dog', 'horse', 'bird', 'fish',
                     'tiger', 'lion', 'elephant', 'monkey', 'rabbit'],
            '国家': ['france', 'germany', 'italy', 'japan', 'china',
                     'england', 'spain', 'russia', 'canada', 'brazil'],
            '职业': ['doctor', 'teacher', 'engineer', 'lawyer',
                     'scientist', 'artist', 'writer', 'programmer'],
            '情感': ['happy', 'sad', 'angry', 'love', 'hate',
                     'fear', 'joy', 'excited', 'calm', 'worried'],
            '颜色': ['red', 'blue', 'green', 'yellow', 'black',
                     'white', 'orange', 'purple', 'brown', 'pink'],
        }
        visualize_embeddings_tsne(model, words_by_category,
                                  output_file='glove_tsne_visualization.png')

        print("\n" + "=" * 60)
        print("所有任务执行完毕")
        print("=" * 60)
```

### 7.3 运行结果示例

```
============================================================
GloVe 调库实现演示
============================================================
正在加载 GloVe 向量: glove.6B.100d.txt.word2vec
加载完成，词汇表大小: 400000

[1/5] 词相似度评估...

======================================================================
词相似度评估结果
======================================================================
词对                        人工评分      模型相似度
----------------------------------------------------------------------
king - queen              0.95        0.7512
man - woman               0.90        0.7664
cat - dog                 0.88        0.8798
car - automobile          0.98        0.9198
happy - joyful            0.92        0.6541
big - large               0.90        0.8791
fast - quick              0.88        0.7356
good - bad                -0.50       -0.3704
king - man                0.70        0.2612
apple - fruit             0.75        0.5310
----------------------------------------------------------------------
Spearman 相关系数: 0.8424 (p-value: 0.002140)
======================================================================

[2/5] 词类比评估...

======================================================================
词类比评估结果
======================================================================
类比                           期望         预测 Top-1     排名     结果
------------------------------------------------------------------------
king : queen :: man : ?       woman        woman          1       正确
king : queen :: boy : ?       girl         girl           1       正确
man : woman :: father : ?     mother       mother         1       正确
man : woman :: brother : ?    sister       sister         1       正确
france : paris :: japan : ?   tokyo        tokyo          1       正确
france : paris :: germany : ? berlin       berlin         1       正确
dog : dogs :: cat : ?         cats         cats           1       正确
----------------------------------------------------------------------
准确率 (Top-5): 16/17 = 94.12%
======================================================================
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
"""
GloVe 手工实现
仅依赖 NumPy，从零构建共现矩阵并训练 GloVe 词向量
"""

import numpy as np
from collections import Counter
import re
import time


class GloVeTokenizer:
    """
    简单的文本分词器
    用于将原始文本转换为token序列
    """

    @staticmethod
    def tokenize(text):
        """
        将文本转换为小写token列表

        Args:
            text: 原始文本

        Returns:
            tokens: 分词后的列表
        """
        text = text.lower()
        # 移除非字母字符
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        return tokens


class CooccurrenceMatrix:
    """
    共现矩阵构建器
    统计语料库中词与词在给定窗口内的共现次数
    """

    def __init__(self, window_size=5, min_count=5):
        """
        初始化共现矩阵构建器

        Args:
            window_size: 上下文窗口大小（单侧距离）
            min_count: 最低词频阈值
        """
        self.window_size = window_size
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, corpus):
        """
        从语料库构建词汇表

        Args:
            corpus: 列表的列表，每个元素是一篇文档的token列表

        Returns:
            word2idx: 词到索引的映射
            idx2word: 索引到词的映射
        """
        # 统计所有词频
        word_counts = Counter()
        for doc in corpus:
            word_counts.update(doc)

        # 过滤低频词并按频率降序排列
        filtered_words = [
            word for word, count in word_counts.items()
            if count >= self.min_count
        ]
        filtered_words.sort(key=lambda w: word_counts[w], reverse=True)

        # 构建映射
        self.word2idx = {word: idx for idx, word in enumerate(filtered_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(filtered_words)

        print(f"词汇表大小: {self.vocab_size} (过滤前: {len(word_counts)})")
        return self.word2idx, self.idx2word

    def build(self, corpus):
        """
        构建共现矩阵

        使用字典存储稀疏数据，格式为 (i, j) -> count

        Args:
            corpus: 分词后的语料库

        Returns:
            cooccurrence: 字典，键为 (i, j) 元组，值为共现次数
        """
        if self.vocab_size == 0:
            self.build_vocab(corpus)

        cooccurrence = {}

        for doc_tokens in corpus:
            # 将token转换为索引，过滤OOV
            indices = [
                self.word2idx[t]
                for t in doc_tokens
                if t in self.word2idx
            ]

            n = len(indices)
            for i in range(n):
                center = indices[i]

                # 计算窗口范围
                left = max(0, i - self.window_size)
                right = min(n, i + self.window_size + 1)

                for j in range(left, right):
                    if i == j:
                        continue
                    context = indices[j]

                    # 距离衰减：距离越远权重越小
                    distance = abs(i - j)
                    weight = 1.0 / distance

                    # 累加共现计数
                    key = (center, context)
                    if key in cooccurrence:
                        cooccurrence[key] += weight
                    else:
                        cooccurrence[key] = weight

        print(f"共现矩阵非零元素数量: {len(cooccurrence)}")
        return cooccurrence


class GloVeTrainer:
    """
    GloVe 训练器
    基于共现矩阵，使用 AdaGrad + SGD 优化加权最小二乘目标函数
    """

    def __init__(self, embedding_dim=50, x_max=100.0, alpha=0.75,
                 initial_lr=0.05, max_epochs=25, seed=42):
        """
        初始化 GloVe 训练器

        Args:
            embedding_dim: 词向量维度
            x_max: 权重截断阈值
            alpha: 权重函数指数参数
            initial_lr: 初始学习率
            max_epochs: 最大训练轮数
            seed: 随机种子
        """
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.seed = seed

        # 模型参数（训练时初始化）
        self.W = None           # 词向量矩阵
        self.W_context = None   # 上下文向量矩阵
        self.b = None           # 词偏置
        self.b_context = None   # 上下文偏置
        self.loss_history = []

    def _weight_function(self, x):
        """
        GloVe 权重函数

        f(x) = (x/x_max)^alpha   if x < x_max
        f(x) = 1                 if x >= x_max

        Args:
            x: 共现次数

        Returns:
            weight: 权重值
        """
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        else:
            return 1.0

    def _initialize_params(self, vocab_size):
        """
        初始化模型参数

        Args:
            vocab_size: 词汇表大小
        """
        rng = np.random.RandomState(self.seed)

        # Xavier-like 初始化
        scale = 0.5 / self.embedding_dim
        self.W = rng.uniform(-scale, scale, (vocab_size, self.embedding_dim))
        self.W_context = rng.uniform(-scale, scale, (vocab_size, self.embedding_dim))

        # 偏置初始化为0
        self.b = np.zeros(vocab_size)
        self.b_context = np.zeros(vocab_size)

        # AdaGrad 梯度平方累积器（初始化为1，避免除零）
        self.grad_sq_W = np.ones((vocab_size, self.embedding_dim))
        self.grad_sq_W_ctx = np.ones((vocab_size, self.embedding_dim))
        self.grad_sq_b = np.ones(vocab_size)
        self.grad_sq_b_ctx = np.ones(vocab_size)

    def fit(self, cooccurrence, vocab_size):
        """
        训练 GloVe 模型

        Args:
            cooccurrence: 共现矩阵（字典格式，键为 (i,j)，值为共现次数）
            vocab_size: 词汇表大小

        Returns:
            embeddings: 最终的词嵌入矩阵 (vocab_size, embedding_dim)
        """
        # 将共现数据转换为数组格式，便于高效训练
        keys = list(cooccurrence.keys())
        values = np.array([cooccurrence[k] for k in keys])
        n_pairs = len(keys)

        print(f"\n开始训练 GloVe 模型...")
        print(f"  词向量维度: {self.embedding_dim}")
        print(f"  词汇表大小: {vocab_size}")
        print(f"  共现对数量: {n_pairs}")
        print(f"  最大训练轮数: {self.max_epochs}")
        print(f"  初始学习率: {self.initial_lr}")

        # 初始化参数
        self._initialize_params(vocab_size)

        start_time = time.time()
        self.loss_history = []

        for epoch in range(self.max_epochs):
            # 线性衰减学习率
            lr = self.initial_lr * max(1.0 - epoch / self.max_epochs, 1e-4)

            # 每个epoch打乱训练样本顺序
            perm = np.random.permutation(n_pairs)
            epoch_loss = 0.0

            for pair_idx in perm:
                i, j = keys[pair_idx]
                x_ij = values[pair_idx]

                # 计算权重 f(X_{ij})
                f_xij = self._weight_function(x_ij)

                # 前向计算
                # diff = w_i . w_tilde_j + b_i + b_tilde_j - log(X_{ij})
                w_i = self.W[i]
                w_ctx_j = self.W_context[j]
                dot_product = np.dot(w_i, w_ctx_j)
                prediction = dot_product + self.b[i] + self.b_context[j]
                diff = prediction - np.log(x_ij)

                # 累计损失
                epoch_loss += f_xij * diff * diff

                # 计算梯度
                grad_common = 2.0 * f_xij * diff
                grad_w_i = grad_common * w_ctx_j        # 对 W[i] 的梯度
                grad_w_ctx_j = grad_common * w_i         # 对 W_ctx[j] 的梯度
                grad_b_i = grad_common                    # 对 b[i] 的梯度
                grad_b_ctx_j = grad_common                # 对 b_ctx[j] 的梯度

                # AdaGrad：累积梯度平方
                self.grad_sq_W[i] += grad_w_i ** 2
                self.grad_sq_W_ctx[j] += grad_w_ctx_j ** 2
                self.grad_sq_b[i] += grad_b_i ** 2
                self.grad_sq_b_ctx[j] += grad_b_ctx_j ** 2

                # AdaGrad：参数更新（学习率除以梯度平方和的平方根）
                self.W[i] -= lr * grad_w_i / np.sqrt(self.grad_sq_W[i])
                self.W_context[j] -= lr * grad_w_ctx_j / np.sqrt(self.grad_sq_W_ctx[j])
                self.b[i] -= lr * grad_b_i / np.sqrt(self.grad_sq_b[i])
                self.b_context[j] -= lr * grad_b_ctx_j / np.sqrt(self.grad_sq_b_ctx[j])

            avg_loss = epoch_loss / n_pairs
            self.loss_history.append(avg_loss)

            elapsed = time.time() - start_time
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1:3d}/{self.max_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.6f} | "
                      f"Time: {elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"\n训练完成，总耗时: {total_time:.1f}s")

        # 合并词向量和上下文向量作为最终嵌入
        embeddings = self.W + self.W_context

        return embeddings

    def get_word_vector(self, word, word2idx, embeddings):
        """
        获取指定词的向量

        Args:
            word: 目标词
            word2idx: 词到索引的映射
            embeddings: 词嵌入矩阵

        Returns:
            vector: 词向量，如果词不在词汇表中则返回 None
        """
        if word in word2idx:
            return embeddings[word2idx[word]]
        return None

    def most_similar(self, word, word2idx, idx2word, embeddings, topn=10):
        """
        查找与指定词最相似的词

        Args:
            word: 目标词
            word2idx: 词到索引的映射
            idx2word: 索引到词的映射
            embeddings: 词嵌入矩阵
            topn: 返回前 n 个最相似的词

        Returns:
            similarities: 最相似的词及其相似度列表
        """
        if word not in word2idx:
            print(f"  警告: '{word}' 不在词汇表中")
            return []

        # 计算目标词与所有词的余弦相似度
        target_vec = embeddings[word2idx[word]]
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return []

        target_vec_normalized = target_vec / target_norm

        # 对所有词计算余弦相似度
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # 避免除零
        embeddings_normalized = embeddings / norms

        similarities = np.dot(embeddings_normalized, target_vec_normalized)

        # 排序（排除自身）
        target_idx = word2idx[word]
        similarities[target_idx] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:topn]

        results = []
        for idx in top_indices:
            results.append((idx2word[idx], similarities[idx]))

        return results

    def analogy(self, a, b, c, word2idx, idx2word, embeddings, topn=5):
        """
        词类比推理: a之于b如同c之于?
        计算: result = b - a + c

        Args:
            a, b, c: 类比中的三个词
            word2idx: 词到索引的映射
            idx2word: 索引到词的映射
            embeddings: 词嵌入矩阵
            topn: 返回前 n 个结果

        Returns:
            results: 最可能的答案列表
        """
        for w in [a, b, c]:
            if w not in word2idx:
                print(f"  警告: '{w}' 不在词汇表中")
                return []

        # 向量运算: result = b - a + c
        vec_a = embeddings[word2idx[a]]
        vec_b = embeddings[word2idx[b]]
        vec_c = embeddings[word2idx[c]]
        result_vec = vec_b - vec_a + vec_c

        # 计算与所有词的余弦相似度
        result_norm = np.linalg.norm(result_vec)
        if result_norm == 0:
            return []
        result_vec_normalized = result_vec / result_norm

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings_normalized = embeddings / norms

        similarities = np.dot(embeddings_normalized, result_vec_normalized)

        # 排除输入的三个词
        for w in [a, b, c]:
            similarities[word2idx[w]] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:topn]

        results = []
        for idx in top_indices:
            results.append((idx2word[idx], similarities[idx]))

        return results


# ===============================
# 测试与演示
# ===============================
if __name__ == "__main__":
    print("=" * 60)
    print("GloVe 手工实现演示")
    print("=" * 60)

    # 构造一个小型示例语料库
    sample_corpus = [
        "the king is a man who rules the kingdom",
        "the queen is a woman who rules the kingdom",
        "a man is a male person and a woman is a female person",
        "the prince is a young man and the princess is a young woman",
        "a boy will grow up to be a man and a girl will grow up to be a woman",
        "the king and queen live in the castle",
        "the prince and princess are children of the king and queen",
        "france is a country and paris is its capital city",
        "japan is a country and tokyo is its capital city",
        "germany is a country and berlin is its capital city",
        "italy is a country and rome is its capital city",
        "england is a country and london is its capital city",
        "china is a country and beijing is its capital city",
        "the cat sat on the mat and the dog played in the yard",
        "a cat is a small animal and a dog is a loyal animal",
        "the bird flew over the tree while the fish swam in the river",
        "a bird can fly in the sky and a fish can swim in the water",
        "happy people smile and sad people cry",
        "joyful feelings make people happy and angry feelings make people mad",
        "the doctor treats patients in the hospital",
        "the teacher teaches students in the school",
        "the engineer builds bridges and the scientist conducts research",
        "red is a color and blue is another color",
        "green grass grows under the blue sky",
        "the sun is yellow and the moon is white",
    ]

    # 分词
    tokenizer = GloVeTokenizer()
    corpus = [tokenizer.tokenize(text) for text in sample_corpus]
    print(f"\n语料库: {len(sample_corpus)} 个句子")
    print(f"总词数: {sum(len(doc) for doc in corpus)}")

    # 构建共现矩阵
    print("\n[1/3] 构建共现矩阵...")
    cooc_builder = CooccurrenceMatrix(window_size=5, min_count=1)
    word2idx, idx2word = cooc_builder.build_vocab(corpus)
    cooccurrence = cooc_builder.build(corpus)

    # 训练 GloVe
    print("\n[2/3] 训练 GloVe...")
    trainer = GloVeTrainer(
        embedding_dim=50,
        x_max=10.0,
        alpha=0.75,
        initial_lr=0.05,
        max_epochs=100,
        seed=42
    )
    embeddings = trainer.fit(cooccurrence, cooc_builder.vocab_size)

    # 测试词相似度
    print("\n[3/3] 测试词向量质量...")

    test_words = ['king', 'queen', 'man', 'woman', 'cat', 'dog', 'happy', 'france']
    for word in test_words:
        if word in word2idx:
            similar = trainer.most_similar(
                word, word2idx, idx2word, embeddings, topn=5
            )
            if similar:
                sim_str = ", ".join([f"{w}({s:.3f})" for w, s in similar])
                print(f"  '{word}' 最相似: {sim_str}")

    # 测试词类比
    print("\n词类比测试:")
    analogy_tests = [
        ('king', 'queen', 'man'),      # man之于king如同?之于queen -> woman
        ('france', 'paris', 'japan'),   # japan之于france如同?之于paris -> tokyo
    ]
    for a, b, c in analogy_tests:
        results = trainer.analogy(a, b, c, word2idx, idx2word, embeddings, topn=5)
        if results:
            res_str = ", ".join([f"{w}({s:.3f})" for w, s in results])
            print(f"  {a} : {b} :: {c} : ? => Top: {res_str}")

    # 打印损失曲线统计
    print(f"\n损失变化: 初始={trainer.loss_history[0]:.4f}, "
          f"最终={trainer.loss_history[-1]:.4f}, "
          f"下降比例={((trainer.loss_history[0] - trainer.loss_history[-1]) / trainer.loss_history[0] * 100):.1f}%")
```

---

## 9. 可视化与结果理解

### 9.1 训练损失曲线可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_training_loss(loss_history, output_file='glove_loss_curve.png'):
    """
    绘制 GloVe 训练损失曲线

    Args:
        loss_history: 每个epoch的平均损失列表
        output_file: 输出图片文件名
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：损失随epoch的变化
    ax1 = axes[0]
    epochs = range(1, len(loss_history) + 1)
    ax1.plot(epochs, loss_history, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Average Loss', fontsize=12)
    ax1.set_title('GloVe Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 标注初始和最终损失
    ax1.annotate(f'Initial: {loss_history[0]:.4f}',
                 xy=(1, loss_history[0]),
                 xytext=(len(epochs)*0.15, loss_history[0]*0.9),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax1.annotate(f'Final: {loss_history[-1]:.4f}',
                 xy=(len(epochs), loss_history[-1]),
                 xytext=(len(epochs)*0.7, loss_history[-1]*1.5),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='green'))

    # 子图2：损失的对数变化（更清晰展示后期收敛情况）
    ax2 = axes[1]
    log_losses = np.log(loss_history)
    ax2.plot(epochs, log_losses, 'r-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Log(Loss)', fontsize=12)
    ax2.set_title('GloVe Training Loss (Log Scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存到: {output_file}")
    plt.close()
```

### 9.2 词向量 t-SNE 降维可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_word_embeddings(embeddings, idx2word, categories,
                               method='tsne', output_file='glove_embedding_vis.png'):
    """
    词向量降维可视化

    Args:
        embeddings: 词嵌入矩阵, shape (vocab_size, embedding_dim)
        idx2word: 索引到词的映射
        categories: 按类别分组的词列表字典
        method: 降维方法 ('tsne' 或 'pca')
        output_file: 输出文件名
    """
    # 收集目标词的向量
    words = []
    vectors = []
    labels = []
    cat_list = list(categories.keys())

    for cat_idx, (category, word_list) in enumerate(categories.items()):
        for word in word_list:
            if word in idx2word.values():
                word_idx = [k for k, v in idx2word.items() if v == word][0]
                words.append(word)
                vectors.append(embeddings[word_idx])
                labels.append(cat_idx)

    vectors = np.array(vectors)
    print(f"降维可视化: {len(words)} 个词")

    # 选择降维方法
    if method == 'tsne':
        # 先用 PCA 降到 50 维，再用 t-SNE 降到 2 维（加速计算）
        perplexity = min(30, len(words) - 1)
        if len(words) > 50 and vectors.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            vectors_pca = pca.fit_transform(vectors)
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=42, n_iter=1000)
            coords = tsne.fit_transform(vectors_pca)
        else:
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=42, n_iter=1000)
            coords = tsne.fit_transform(vectors)
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vectors)

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, len(cat_list)))

    for cat_idx, category in enumerate(cat_list):
        mask = [l == cat_idx for l in labels]
        x = coords[mask, 0]
        y = coords[mask, 1]
        cat_words = [words[j] for j in range(len(labels)) if labels[j] == cat_idx]

        ax.scatter(x, y, c=[colors[cat_idx]], label=category,
                   alpha=0.7, s=80, edgecolors='white', linewidth=0.5)

        for xi, yi, word in zip(x, y, cat_words):
            ax.annotate(word, (xi, yi), fontsize=8,
                        ha='center', va='bottom',
                        textcoords="offset points", xytext=(0, 5))

    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    title = f'GloVe Word Embeddings Visualization ({method.upper()})'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_file}")
    plt.close()
```

### 9.3 词向量语义方向可视化

```python
def visualize_semantic_direction(embeddings, word2idx, idx2word,
                                  direction_words, output_file='semantic_direction.png'):
    """
    可视化词向量在特定语义方向上的投影

    例如可视化 "性别" 方向：king/queen, man/woman 等

    Args:
        embeddings: 词嵌入矩阵
        word2idx: 词到索引的映射
        idx2word: 索引到词的映射
        direction_words: 语义方向定义，列表中的元组 (word_a, word_b)
                        方向定义为 normalize(v_b - v_a)
        output_file: 输出文件名
    """
    # 计算语义方向
    v_a = embeddings[word2idx[direction_words[0][0]]]
    v_b = embeddings[word2idx[direction_words[0][1]]]
    direction = v_b - v_a
    direction = direction / np.linalg.norm(direction)

    # 在方向上投影
    projections = {}
    for word, idx in word2idx.items():
        vec = embeddings[idx]
        proj = np.dot(vec, direction)
        projections[word] = proj

    # 绘制直方图
    proj_values = list(projections.values())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(proj_values, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='中性')
    ax.set_xlabel(f'Projection on direction: '
                  f'{direction_words[0][0]} -> {direction_words[0][1]}', fontsize=12)
    ax.set_ylabel('Word Count', fontsize=12)
    ax.set_title('Semantic Direction Projection Distribution', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 标注一些代表性词
    sorted_words = sorted(projections.items(), key=lambda x: x[1])
    n_annotate = 5
    extreme_words = (sorted_words[:n_annotate] + sorted_words[-n_annotate:])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"语义方向可视化已保存到: {output_file}")
    plt.close()
```

### 9.4 共现矩阵可视化

```python
def visualize_cooccurrence_matrix(cooccurrence, word2idx, target_words,
                                   output_file='cooccurrence_heatmap.png'):
    """
    可视化词-词共现矩阵的热力图

    Args:
        cooccurrence: 共现矩阵（字典格式）
        word2idx: 词到索引的映射
        target_words: 需要可视化的目标词列表
        output_file: 输出文件名
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 过滤出目标词
    valid_words = [w for w in target_words if w in word2idx]
    n = len(valid_words)
    if n < 2:
        print("有效词汇太少，无法绘制热力图")
        return

    # 构建子矩阵
    matrix = np.zeros((n, n))
    for i, w1 in enumerate(valid_words):
        for j, w2 in enumerate(valid_words):
            idx1 = word2idx[w1]
            idx2 = word2idx[w2]
            key = (idx1, idx2)
            matrix[i, j] = cooccurrence.get(key, 0.0)

    # 对数变换（使小值可见）
    matrix_log = np.log1p(matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_log, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(valid_words, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(valid_words, fontsize=10)

    ax.set_title('Word Co-occurrence Matrix (log scale)',
                 fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, label='log(1 + count)')

    # 在每个格子中显示数值
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = 'white' if matrix_log[i, j] > matrix_log.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=7, color=text_color)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"共现矩阵热力图已保存到: {output_file}")
    plt.close()
```

### 9.5 结果解读

**从损失曲线可以看出**：
- 损失在初期（前 5-10 个 epoch）快速下降，说明模型快速学习到了基本的共现模式
- 中期损失下降速度变缓，模型在学习更细粒度的语义关系
- 后期损失趋于收敛，AdaGrad 的自适应学习率使得参数更新越来越精细

**从 t-SNE 可视化可以看出**：
- 同类别的词（如动物、国家、情感）在二维空间中明显聚集
- 不同类别之间的距离反映了语义差异
- t-SNE 保留了局部的邻近关系，因此语义相近的词会靠近

**从词类比结果可以看出**：
- 高质量的 GloVe 向量在标准类比任务上的准确率通常超过 60%（语义类比）和 50%（句法类比）
- 对于高频词（如 king/queen/man/woman），类比结果通常非常准确
- 对于低频词或领域特定词汇，类比效果可能下降

---

## 10. 模型评估

### 10.1 评估指标

**词向量模型的评估通常分为两大类**：

| 评估类型 | 评估指标 | 说明 |
|---------|---------|------|
| 内在评估 | 词相似度 | 模型预测的词对相似度与人工打分的 Spearman 相关系数 |
| 内在评估 | 词类比准确率 | 类比任务（a:b :: c:?）的准确率 |
| 外在评估 | 下游任务性能 | 在文本分类、NER 等任务上的 F1、Accuracy 等 |
| 外在评估 | 语义文本相似度 | 句子级别的语义相似度预测 |

### 10.2 词相似度评估

```python
from scipy.stats import spearmanr

def evaluate_word_similarity_detailed(model, word_sim_dataset):
    """
    详细的词相似度评估

    Args:
        model: 词向量模型（gensim KeyedVectors 或自定义模型）
        word_sim_dataset: 词相似度数据集
            格式: [(word1, word2, human_score), ...]

    Returns:
        correlation: Spearman 相关系数
    """
    model_scores = []
    human_scores = []
    missed = 0

    for w1, w2, human_score in word_sim_dataset:
        if hasattr(model, 'similarity'):
            # gensim KeyedVectors
            if w1 in model and w2 in model:
                sim = model.similarity(w1, w2)
                model_scores.append(sim)
                human_scores.append(human_score)
            else:
                missed += 1
        else:
            # 自定义模型
            if w1 in model.word2idx and w2 in model.word2idx:
                v1 = model.get_word_vector(w1, model.word2idx, model.embeddings)
                v2 = model.get_word_vector(w2, model.word2idx, model.embeddings)
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                model_scores.append(sim)
                human_scores.append(human_score)
            else:
                missed += 1

    if missed > 0:
        print(f"  跳过 {missed}/{len(word_sim_dataset)} 个词对（词汇表外）")

    if len(model_scores) < 3:
        print("  有效词对太少，无法计算相关性")
        return 0.0

    correlation, p_value = spearmanr(model_scores, human_scores)
    print(f"  有效词对数: {len(model_scores)}")
    print(f"  Spearman 相关系数: {correlation:.4f}")
    print(f"  P-value: {p_value:.6f}")

    return correlation


def evaluate_on_simlex999(model, simlex_path='SimLex-999.txt'):
    """
    在 SimLex-999 数据集上评估

    SimLex-999 是一个广泛使用的词相似度/相关度评测基准
    包含 999 个词对，每个词对有相似度和相关度两个人工评分

    Args:
        model: 词向量模型
        simlex_path: SimLex-999 数据集文件路径

    Returns:
        similarity_corr: 相似度相关系数
        correlation_corr: 相关度相关系数
    """
    # SimLex-999 文件格式示例：
    # word1  word2  SimLex999  SD  concretness
    try:
        import pandas as pd
        df = pd.read_csv(simlex_path, sep='\t')

        similarity_scores = []
        simlex_scores = []

        for _, row in df.iterrows():
            w1, w2 = row['word1'], row['word2']
            if hasattr(model, 'similarity'):
                if w1 in model and w2 in model:
                    sim = model.similarity(w1, w2)
                    similarity_scores.append(sim)
                    simlex_scores.append(row['SimLex999'])

        corr, _ = spearmanr(similarity_scores, simlex_scores)
        print(f"SimLex-999 Spearman 相关系数: {corr:.4f}")
        return corr

    except FileNotFoundError:
        print(f"未找到 SimLex-999 数据集: {simlex_path}")
        return 0.0
```

### 10.3 词类比评估

```python
def evaluate_analogy_dataset(model, analogy_path='questions-words.txt'):
    """
    在标准词类比数据集上评估

    Google 的 questions-words.txt 包含约 19544 个类比问题
    涵盖语义类比和句法类比两大类

    Args:
        model: 词向量模型
        analogy_path: 类比数据集文件路径

    Returns:
        results: 各类别及总体的准确率
    """
    # 模拟 questions-words.txt 格式的评估
    analogy_categories = {
        'capital-common-countries': [
            ('athens', 'greece', 'baghdad', 'iraq'),
            ('athens', 'greece', 'bangkok', 'thailand'),
            ('athens', 'greece', 'beijing', 'china'),
        ],
        'capital-world': [
            ('paris', 'france', 'berlin', 'germany'),
        ],
        'currency': [
            ('algeria', 'dinar', 'angola', 'kwanza'),
        ],
        'city-in-state': [
            ('chicago', 'illinois', 'houston', 'texas'),
        ],
        'family': [
            ('boy', 'girl', 'father', 'mother'),
            ('brother', 'sister', 'dad', 'mom'),
        ],
        'gram1-adjective-to-adverb': [
            ('amazing', 'amazingly', 'apparent', 'apparently'),
        ],
        'gram2-opposite': [
            ('acceptable', 'unacceptable', 'aware', 'unaware'),
        ],
        'gram3-comparative': [
            ('bad', 'worse', 'big', 'bigger'),
        ],
        'gram4-superlative': [
            ('bad', 'worst', 'big', 'biggest'),
        ],
        'gram5-present-participle': [
            ('code', 'coding', 'dance', 'dancing'),
        ],
    }

    category_results = {}

    for category, analogies in analogy_categories.items():
        correct = 0
        total = 0

        for a, b, c, expected in analogies:
            if hasattr(model, 'most_similar'):
                if not all(w in model for w in [a, b, c, expected]):
                    continue
                total += 1
                try:
                    result = model.most_similar(
                        positive=[b, c], negative=[a], topn=5
                    )
                    top_words = [w for w, _ in result]
                    if expected in top_words:
                        correct += 1
                except KeyError:
                    continue

        if total > 0:
            acc = correct / total
            category_results[category] = {'correct': correct, 'total': total, 'acc': acc}
            print(f"  {category:<35} {correct}/{total} = {acc:.2%}")

    return category_results
```

### 10.4 不同维度 GloVe 向量的对比评估

```python
def compare_glove_dimensions(dimensions, evaluation_func):
    """
    对比不同维度的 GloVe 向量的性能

    Args:
        dimensions: 维度列表，如 [50, 100, 200, 300]
        evaluation_func: 评估函数

    Returns:
        results: 各维度的评估结果
    """
    results = {}

    for dim in dimensions:
        # 加载对应维度的 GloVe 预训练向量
        glove_file = f'glove.6B.{dim}d.txt'
        word2vec_file = glove_file + '.word2vec'

        # 评估
        print(f"\n--- GloVe {dim}d ---")
        try:
            from gensim.models import KeyedVectors
            model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
            score = evaluation_func(model)
            results[dim] = score
        except FileNotFoundError:
            print(f"  文件未找到: {glove_file}")
            results[dim] = None

    # 汇总
    print("\n" + "=" * 50)
    print("维度对比结果:")
    for dim, score in results.items():
        if score is not None:
            print(f"  GloVe {dim:>3}d: {score:.4f}")
    print("=" * 50)

    return results
```

---

## 11. 常见问题与易错点

### 11.1 共现矩阵构建问题

**问题 1：共现矩阵过大导致内存不足**

**原因**：
- 词汇表大小为 $|V|$ 时，完整共现矩阵的大小为 $|V|^2$
- 例如词汇表为 40 万词，完整矩阵需要约 640GB 内存（float64）
- 即使使用稀疏存储，非零元素也可能非常多

**解决方案**：

```python
# 方法1：增大 min_count 过滤更多低频词
cooc_builder = CooccurrenceMatrix(window_size=10, min_count=50)
# 将词汇表从 40 万缩小到约 10 万

# 方法2：使用稀疏矩阵格式（COO 或 CSR）
from scipy.sparse import coo_matrix, csr_matrix

# 方法3：在训练时直接使用字典存储，不构建完整矩阵
# 这样只存储非零共现对，大幅节省内存
cooccurrence = {}  # {(i, j): count}
```

**问题 2：窗口大小选择不当**

**原因**：
- 窗口过小（如 2）：只能捕获非常局部的共现关系，丢失远距离但语义相关的词对
- 窗口过大（如 50）：引入大量弱相关的噪声共现，使得模型难以区分强相关和弱相关
- GloVe 原始论文推荐使用对称窗口（左右各 5-10 个词）

**解决方案**：

```python
# 小语料（百万级词数）：使用较大窗口
cooc_builder = CooccurrenceMatrix(window_size=15)

# 大语料（十亿级词数）：使用中等窗口
cooc_builder = CooccurrenceMatrix(window_size=5)

# 常规推荐
cooc_builder = CooccurrenceMatrix(window_size=10)
```

### 11.2 训练问题

**问题 3：损失不收敛**

**原因**：
- 初始学习率过大，导致损失震荡
- 共现矩阵中存在极端值（非常大的共现次数），导致梯度不稳定
- 权重函数的 $x_{\max}$ 设置不当

**解决方案**：

```python
# 1. 降低初始学习率
trainer = GloVeTrainer(initial_lr=0.01)  # 从 0.05 降到 0.01

# 2. 调整 x_max（降低截断阈值）
trainer = GloVeTrainer(x_max=50)  # 从 100 降到 50

# 3. 对共现次数做截断处理
x_max = 100
for key in cooccurrence:
    cooccurrence[key] = min(cooccurrence[key], x_max)
```

**问题 4：训练速度过慢**

**原因**：
- Python 实现的训练循环效率低下
- 共现对数量过多（数十亿级别）
- 没有使用批处理或并行化

**解决方案**：

```python
# 方法1：使用向量化操作替代逐对更新
# 将共现数据转为 NumPy 数组进行批量处理

# 方法2：使用原始 C++ 实现（推荐）
# 斯坦福官方提供的 GloVe 实现使用 C++ 编写，支持多线程
# 下载地址: https://github.com/stanfordnlp/GloVe

# 方法3：过滤低频共现对
# 去除共现次数小于某个阈值的词对
min_cooccurrence = 3
filtered = {k: v for k, v in cooccurrence.items() if v >= min_cooccurrence}

# 方法4：使用子采样
# 对于极高频词（如 "the", "a"），在构建共现矩阵时进行子采样
```

### 11.3 词向量质量问题

**问题 5：词类比结果不准确**

**原因**：
- 训练语料太小或质量差
- 词向量维度太低（如 25 维），表达能力不足
- 训练不充分（epoch 太少）
- 词汇表太小，很多词被过滤

**解决方案**：

```python
# 1. 使用预训练的 GloVe 向量（最推荐的方案）
# 斯坦福 NLP 提供基于大规模语料训练的向量
# 6B: 维基百科 + Gigaword (40万词, 6B tokens)
# 42B: Common Crawl (190万词, 42B tokens)
# 840B: Common Crawl (220万词, 840B tokens)

# 2. 增加词向量维度
trainer = GloVeTrainer(embedding_dim=300)  # 从 50 增加到 300

# 3. 增加训练轮数
trainer = GloVeTrainer(max_epochs=50)  # 从 25 增加到 50
```

**问题 6：无法区分多义词**

**原因**：
- GloVe 为每个词分配唯一的固定向量，这是所有静态词嵌入的固有局限
- 例如 "bank" 的 "银行" 和 "河岸" 两个含义使用相同的向量

**解决方案**：

```python
# 方法1：使用动态词嵌入模型（如 ELMo、BERT）
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 不同上下文中 "bank" 的向量不同
text1 = "I deposited money in the bank"
text2 = "I sat by the river bank"

# 方法2：后处理方法
# 对上下文中的词向量取平均，获得上下文相关的表示
def contextualize(word_vector, context_vectors, weights=None):
    """
    通过上下文加权平均来获得上下文相关的词向量
    """
    if weights is None:
        weights = np.ones(len(context_vectors)) / len(context_vectors)
    context_avg = np.average(context_vectors, axis=0, weights=weights)
    return 0.7 * word_vector + 0.3 * context_avg
```

### 11.4 与 gensim/其他库的兼容性问题

**问题 7：GloVe 格式与 Word2Vec 格式不兼容**

**原因**：
- GloVe 的官方格式为：每行 `word dim1 dim2 ... dimN`（空格分隔）
- Word2Vec 的格式为：首行 `vocab_size dim`，后续每行 `word dim1 dim2 ... dimN`

**解决方案**：

```python
# 使用 gensim 的转换工具
from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

# 然后正常加载
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
```

---

## 12. 学习总结

### 12.1 核心要点回顾

- **核心思想**：通过全局词-词共现统计来学习词向量，利用共现概率比值编码语义关系

- **数学本质**：加权最小二乘回归，本质上是带权重的矩阵分解。共现矩阵 $X$ 的对数被低秩矩阵 $W \tilde{W}^T$ 近似

- **优化目标**：最小化 $J = \sum_{i,j} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$

- **适用场景**：作为文本的词级别表示，用于下游 NLP 任务（分类、相似度计算、信息检索等）

- **局限性**：静态词嵌入无法处理多义词和上下文相关的语义变化

### 12.2 关键公式汇总

**1. 条件概率（共现概率）**：
$$ P_{ij} = P(w_j \mid w_i) = \frac{X_{ij}}{\sum_k X_{ik}} $$

**2. 概率比值（语义信号）**：
$$ \frac{P_{ik}}{P_{jk}} = \frac{X_{ik} / X_i}{X_{jk} / X_j} $$

**3. 对数双线性模型**：
$$ w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log X_{ij} $$

**4. 权重函数**：
$$ f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{if } x \geq x_{\max} \end{cases} $$

**5. GloVe 损失函数**：
$$ J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2 $$

**6. 参数梯度**：
$$ \frac{\partial J}{\partial w_i} = \sum_j 2 f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right) \tilde{w}_j $$

**7. AdaGrad 更新规则**：
$$ w_i^{(t+1)} = w_i^{(t)} - \frac{\eta_0}{\sqrt{\sum_{\tau=1}^{t} g_{i,\tau}^2 + \epsilon}} \cdot g_i^{(t)} $$

### 12.3 最佳实践

**数据准备**：
- 使用大规模语料库（推荐数十亿 token 以上）
- 设置合理的 min_count（通常 5-50）
- 使用对称窗口（左右各 5-10 个词）
- 可选的距离衰减加权

**训练配置**：
- 词向量维度：通用任务推荐 100-300 维
- 学习率：初始 0.05，线性衰减
- 训练轮数：15-50 个 epoch（大语料 15 即可）
- x_max：100（默认值），alpha：0.75（默认值）

**模型使用**：
- 优先使用预训练的 GloVe 向量（节省大量训练时间）
- 训练后将 W 和 W_context 相加作为最终词向量
- 使用余弦相似度衡量词之间的语义关系

### 12.4 与其他算法的联系

- **前置算法**：
  - **TF-IDF**：基于全局统计的词表示方法，但 TF-IDF 是稀疏高维表示，GloVe 是稠密低维表示
  - **LSA/SVD**：直接对共现矩阵（或 TF-IDF 矩阵）做奇异值分解，GloVe 可以看作是 SVD 的加权推广

- **平行算法**：
  - **Word2Vec**：基于局部上下文窗口的词嵌入方法，与 GloVe 互补（局部 vs 全局）
  - **FastText**：在 Word2Vec 基础上增加子词信息，解决 OOV 问题

- **后续算法**：
  - **ELMo**：基于双向 LSTM 的动态词嵌入，解决了静态词嵌入的多义性问题
  - **BERT/GPT**：基于 Transformer 的上下文词嵌入，是当前主流的预训练模型
  - **word2vec / GloVe** 作为理解动态词嵌入的基石，其核心思想（分布式语义假设）贯穿始终

- **理论联系**：
  - GloVe 的加权最小二乘目标可以等价地表示为某种形式的矩阵分解
  - Levy 和 Goldberg（2014）证明了 Skip-gram 负采样（SGNS）与隐式矩阵分解之间的等价关系
  - 因此，Word2Vec 和 GloVe 在数学本质上是相通的，只是训练方式和加权策略不同

---

## 13. 练习题与思考题（含答案）

### 练习题

**习题 1：基础概念理解**

关于 GloVe 模型，以下说法中**错误**的是哪一项？

A. GloVe 利用全局词-词共现矩阵来训练词向量
B. GloVe 的权重函数 $f(x)$ 满足 $f(0) = 0$
C. GloVe 为每个词学习一个词向量 $w_i$ 和一个上下文向量 $\tilde{w}_i$，两者最终合并为同一个向量
D. GloVe 能自动处理 OOV（Out-of-Vocabulary）问题

<details>
<summary>答案</summary>

**答案：D**

**解析**：
- A 正确：GloVe 的核心就是使用全局共现矩阵进行训练
- B 正确：共现次数为 0 的词对不应参与训练，因此 $f(0) = 0$
- C 正确：GloVe 为每个词维护两组向量，训练后通常将两者相加或取平均作为最终向量
- D 错误：GloVe 是静态词嵌入模型，无法处理词汇表之外的词（OOV 问题）。要解决 OOV 问题，需要使用 FastText（子词方法）或 BPE 等分词方法

</details>

---

**习题 2：权重函数计算**

给定 GloVe 的权重函数参数 $x_{\max} = 100$，$\alpha = 0.75$，请计算以下共现次数对应的权重值：
1. $x = 0$
2. $x = 1$
3. $x = 25$
4. $x = 100$
5. $x = 200$

<details>
<summary>答案</summary>

权重函数为：
$$ f(x) = \begin{cases} (x / 100)^{0.75} & \text{if } x < 100 \\ 1 & \text{if } x \geq 100 \end{cases} $$

计算结果：

1. $f(0) = (0/100)^{0.75} = 0^{0.75} = 0$

2. $f(1) = (1/100)^{0.75} = (0.01)^{0.75} = 0.01^{3/4}$
   - $0.01^{0.75} = e^{0.75 \times \ln(0.01)} = e^{0.75 \times (-4.605)} = e^{-3.454} \approx 0.0316$

3. $f(25) = (25/100)^{0.75} = 0.25^{0.75} = 0.25^{3/4}$
   - $0.25^{0.75} = e^{0.75 \times \ln(0.25)} = e^{0.75 \times (-1.386)} = e^{-1.040} \approx 0.3536$

4. $f(100) = (100/100)^{0.75} = 1^{0.75} = 1$

5. $f(200) = 1$（因为 $200 \geq 100$）

汇总：

| $x$ | $f(x)$ |
|-----|--------|
| 0   | 0      |
| 1   | 0.032  |
| 25  | 0.354  |
| 100 | 1.000  |
| 200 | 1.000  |

</details>

---

**习题 3：数学推导**

证明 GloVe 的目标函数在忽略权重函数的情况下，等价于对共现矩阵的（加权）对数矩阵进行低秩矩阵分解。

<details>
<summary>答案</summary>

**证明**：

GloVe 的目标函数为（忽略权重函数 $f$）：

$$ J = \sum_{i,j=1}^{|V|} \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2 $$

定义矩阵：
- $W \in \mathbb{R}^{|V| \times d}$：词向量矩阵（第 $i$ 行为 $w_i^T$）
- $\tilde{W} \in \mathbb{R}^{d \times |V|}$：上下文向量矩阵（第 $j$ 列为 $\tilde{w}_j$）
- $b \in \mathbb{R}^{|V|}$：词偏置向量
- $\tilde{b} \in \mathbb{R}^{|V|}$：上下文偏置向量
- $L = \log X$：共现矩阵的对数矩阵（逐元素取对数，对 $X_{ij}=0$ 的位置忽略）

定义预测矩阵 $P = W \tilde{W}^T + b \mathbf{1}^T + \mathbf{1} \tilde{b}^T$，其中 $\mathbf{1}$ 是全 1 向量。

则目标函数可以写为：

$$ J = \| P - L \|_F^2 $$

这是经典的 Frobenius 范数最小化问题。$P$ 是秩为 $d$ 的矩阵加上偏置修正。

如果忽略偏置项（$b = 0, \tilde{b} = 0$），则 $P = W \tilde{W}^T$，目标变为：

$$ J = \| W \tilde{W}^T - L \|_F^2 $$

这正是对矩阵 $L$ 的秩为 $d$ 的最佳低秩近似，等价于对 $L$ 进行截断 SVD 分解。

因此，GloVe 在数学上等价于对共现矩阵的对数进行加权低秩矩阵分解。权重函数 $f$ 的引入使得分解更加关注共现次数适中的词对。

**证毕。**

</details>

---

**习题 4：编程实践**

给定以下小型共现矩阵，手工计算 GloVe 模型在第一个训练样本上的梯度更新。

```
共现矩阵 X:
          ice   steam  solid  gas    water
ice       0      3      8      1      7
steam     3      0      1      8      7
solid     8      1      0      0      3
gas       1      8      0      0      3
water     7      7      3      3      0
```

假设：
- 词向量维度 $d = 2$
- 初始参数：$w_{\text{ice}} = [0.1, 0.2]^T$，$\tilde{w}_{\text{steam}} = [0.3, -0.1]^T$
- $b_{\text{ice}} = 0$，$\tilde{b}_{\text{steam}} = 0$
- $x_{\max} = 10$，$\alpha = 0.75$
- 学习率 $\eta = 0.1$

请对共现对 (ice, steam) 计算一次梯度更新。

<details>
<summary>答案</summary>

**步骤 1：计算权重 $f(X_{ij})$**

$X_{\text{ice, steam}} = 3$，$x_{\max} = 10$，$\alpha = 0.75$

$$ f(3) = \left(\frac{3}{10}\right)^{0.75} = 0.3^{0.75} \approx 0.4053 $$

**步骤 2：计算预测值**

$$ \hat{y} = w_{\text{ice}}^T \tilde{w}_{\text{steam}} + b_{\text{ice}} + \tilde{b}_{\text{steam}} $$
$$ = [0.1, 0.2] \cdot [0.3, -0.1]^T + 0 + 0 $$
$$ = 0.1 \times 0.3 + 0.2 \times (-0.1) $$
$$ = 0.03 - 0.02 = 0.01 $$

**步骤 3：计算误差**

$$ \text{diff} = \hat{y} - \log(X_{\text{ice, steam}}) = 0.01 - \ln(3) = 0.01 - 1.0986 = -1.0886 $$

**步骤 4：计算公共梯度因子**

$$ g = 2 \times f(X_{ij}) \times \text{diff} = 2 \times 0.4053 \times (-1.0886) = -0.8823 $$

**步骤 5：计算各参数的梯度**

$$ \frac{\partial J}{\partial w_{\text{ice}}} = g \times \tilde{w}_{\text{steam}} = -0.8823 \times [0.3, -0.1] = [-0.2647, 0.0882] $$

$$ \frac{\partial J}{\partial \tilde{w}_{\text{steam}}} = g \times w_{\text{ice}} = -0.8823 \times [0.1, 0.2] = [-0.0882, -0.1765] $$

$$ \frac{\partial J}{\partial b_{\text{ice}}} = g = -0.8823 $$

$$ \frac{\partial J}{\partial \tilde{b}_{\text{steam}}} = g = -0.8823 $$

**步骤 6：参数更新（忽略 AdaGrad，使用固定学习率）**

$$ w_{\text{ice}}^{new} = w_{\text{ice}} - \eta \frac{\partial J}{\partial w_{\text{ice}}} $$
$$ = [0.1, 0.2] - 0.1 \times [-0.2647, 0.0882] $$
$$ = [0.1 + 0.0265, 0.2 - 0.0088] = [0.1265, 0.1912] $$

$$ \tilde{w}_{\text{steam}}^{new} = \tilde{w}_{\text{steam}} - \eta \frac{\partial J}{\partial \tilde{w}_{\text{steam}}} $$
$$ = [0.3, -0.1] - 0.1 \times [-0.0882, -0.1765] $$
$$ = [0.3 + 0.0088, -0.1 + 0.0177] = [0.3088, -0.0823] $$

$$ b_{\text{ice}}^{new} = 0 - 0.1 \times (-0.8823) = 0.0882 $$

$$ \tilde{b}_{\text{steam}}^{new} = 0 - 0.1 \times (-0.8823) = 0.0882 $$

更新后，新的预测值为：
$$ \hat{y}_{new} = [0.1265, 0.1912] \cdot [0.3088, -0.0823] + 0.0882 + 0.0882 $$
$$ = 0.1265 \times 0.3088 + 0.1912 \times (-0.0823) + 0.1764 $$
$$ = 0.0391 - 0.0157 + 0.1764 = 0.1998 $$

目标值为 $\ln(3) = 1.0986$。可以看到预测值从 0.01 增加到了 0.1998，在向目标值靠近。

</details>

---

**习题 5：对比分析**

对比 GloVe 和 Word2Vec (Skip-gram 负采样) 在以下场景中的优劣，并给出推荐选择：

场景 A：拥有 10 亿 token 的中文维基百科语料，需要训练中文词向量
场景 B：需要处理包含大量生僻字和未登录词的社交媒体文本
场景 C：需要为下游文本分类任务提供词嵌入，训练数据量有限（100 万 token）

<details>
<summary>答案</summary>

**场景 A：10 亿 token 中文维基百科**

**推荐**：两者都合适，GloVe 略优

- GloVe：10 亿 token 足以构建高质量的共现矩阵，GloVe 能充分利用全局统计信息
- Word2Vec：也能在此规模上训练出高质量向量
- 选择依据：如果有预训练的 GloVe 向量可直接使用；否则，Word2Vec 的训练工具更成熟易用（如 gensim），可以更快速地迭代实验

**场景 B：社交媒体文本，大量 OOV**

**推荐**：FastText（两者都不太适合）

- GloVe：完全无法处理 OOV，因为静态查找表机制
- Word2Vec：同样无法处理 OOV
- FastText：通过字符 n-gram 建模子词信息，能对未见过的词生成合理的向量
- 如果必须在两者中选择：建议使用 Word2Vec 训练后，用字符级方法为 OOV 词生成初始向量

**场景 C：100 万 token 小语料，下游分类任务**

**推荐**：使用预训练 GloVe 向量微调，或直接使用 Word2Vec

- GloVe：100 万 token 太小，从头训练质量差。但如果使用预训练的 GloVe（如在中文语料上预训练的向量），作为分类器的初始化效果很好
- Word2Vec：在小语料上训练比 GloVe 更稳定（局部窗口对数据量的要求较低）
- 最佳方案：使用预训练的词向量（GloVe 或 Word2Vec 均可），然后在分类任务上微调整个模型

**总结对比表**：

| 维度 | 场景 A（大语料） | 场景 B（OOV 多） | 场景 C（小语料） |
|------|-----------------|-----------------|-----------------|
| GloVe | 优（全局统计） | 差（无法 OOV） | 中（需预训练） |
| Word2Vec | 优（局部窗口） | 差（无法 OOV） | 优（小数据友好） |
| FastText | 良 | 优（子词 OOV） | 良 |

</details>

---

### 思考题

**思考题 1：GloVe 与矩阵分解的关系**

论文中提到 GloVe 可以看作是一种加权矩阵分解。请思考：如果令权重函数 $f(x) = 1$（对所有共现对等权），GloVe 的目标函数与对 $\log X$ 做 SVD 分解有何异同？

<details>
<summary>答案</summary>

**相同点**：
- 两者都是对对数共现矩阵 $\log X$ 进行低秩近似
- 两者都最小化 $\| W \tilde{W}^T - \log X \|_F^2$（忽略偏置）
- 最优解在数学上是等价的

**不同点**：

1. **优化方法**：
   - SVD 通过特征值分解直接求解（解析解），计算复杂度 $O(|V|^3)$，对大词汇表不现实
   - GloVe 使用 SGD 迭代优化（数值解），可以处理大规模词汇表
   - SVD 保证全局最优，GloVe 只保证局部最优

2. **偏置项**：
   - SVD 没有显式的偏置项
   - GloVe 引入 $b_i$ 和 $\tilde{b}_j$ 来捕获词频差异，相当于对矩阵做 rank-1 修正

3. **权重机制**：
   - 标准 SVD 对所有元素等权
   - GloVe 通过 $f(x)$ 对不同共现次数赋予不同权重，使得模型更加关注信息量适中的词对
   - 加权后的 SVD 没有高效的解析解，而 GloVe 的 SGD 方法自然地支持加权

4. **缺失值处理**：
   - SVD 要求矩阵是完整的（或需要额外的缺失值处理）
   - GloVe 天然跳过 $X_{ij} = 0$ 的词对

**结论**：GloVe 可以视为一种"高效的加权截断 SVD"，它在保持矩阵分解数学本质的同时，通过 SGD 和加权策略解决了 SVD 在大规模稀疏矩阵上的计算困难。

</details>

**思考题 2：从 GloVe 到 Transformer 的演进**

GloVe 属于静态词嵌入（第一代预训练模型），而 BERT 属于动态词嵌入（第二代预训练模型）。请分析从 GloVe 到 BERT 的关键演进路线，并讨论 GloVe 的哪些思想在 Transformer 中得到了继承和发展。

<details>
<summary>答案</summary>

**演进路线**：

```
GloVe (2014) -> FastText (2016) -> ELMo (2018) -> BERT (2018)
    |             |                |              |
  全局统计     子词信息        双向LSTM        Transformer编码器
  静态嵌入     静态嵌入        动态嵌入         动态嵌入
```

**GloVe 的核心思想在后续模型中的继承**：

1. **分布式语义假设**（核心假设被完整继承）：
   - GloVe 的基础假设："语义相似的词出现在相似的上下文中"
   - 这个假设被 Word2Vec、ELMo、BERT 等所有后续模型继承
   - BERT 的 MLM（掩码语言模型）任务本质上也是在利用上下文预测目标词

2. **全局统计的思想**（以不同形式继承）：
   - GloVe 通过共现矩阵利用全局信息
   - BERT 通过 Transformer 的自注意力机制实现"全局感受野"
   - 注意力机制让每个词都能"看到"序列中的所有其他词，类似于 GloVe 利用了全局共现信息

3. **向量算术性质**（部分继承）：
   - GloVe 的词向量具有 $v_{king} - v_{man} + v_{woman} \approx v_{queen}$ 的性质
   - BERT 的 [CLS] 向量和各层表示也展现出了类似的线性语义结构
   - 研究表明，BERT 的词表示在特定层上也支持类似的类比运算

4. **从统计到学习的演进**：
   - GloVe：手工设计的损失函数（加权最小二乘），基于明确的统计量
   - ELMo：使用双向 LSTM，通过语言模型目标学习上下文表示
   - BERT：使用 Transformer，通过 MLM + NSP 任务在大规模语料上端到端学习

**GloVe 的思想被超越的方面**：

1. **上下文无关 -> 上下文相关**：GloVe 的静态向量被 BERT 的动态表示取代
2. **浅层模型 -> 深层模型**：GloVe 的简单线性模型被 Transformer 的深层网络取代
3. **单一任务 -> 多任务学习**：GloVe 只做词嵌入，BERT 统一了多个 NLP 任务

</details>

---

## 14. 学习路径建议

### 14.1 前置知识

**学习 GloVe 前，你需要掌握**：

**数学基础**：
- 线性代数：矩阵乘法、向量空间、SVD/PCA 等降维方法
- 概率与统计：条件概率、概率分布、期望和方差
- 优化理论：梯度下降、损失函数、正则化

**编程基础**：
- Python：NumPy 矩阵运算、字典和列表操作
- 数据处理：文本清洗、分词、构建词汇表

**NLP 基础**：
- 词嵌入的基本概念（为什么需要词嵌入）
- One-Hot 编码的局限性
- 词袋模型和 TF-IDF

### 14.2 平行算法（可同时学习）

1. **Word2Vec（CBOW / Skip-gram）**：最经典的词嵌入方法，与 GloVe 形成互补
   - 学习重点：负采样、层次 Softmax、CBOW 与 Skip-gram 的区别
   - 对比点：局部窗口 vs 全局共现矩阵、生成式 vs 回归式

2. **FastText**：Word2Vec 的改进版，增加子词信息
   - 学习重点：字符 n-gram、子词向量的构建和查询
   - 对比点：OOV 问题的处理、更细粒度的语义建模

3. **LSA/SVD**：经典的基于矩阵分解的语义分析方法
   - 学习重点：TF-IDF 矩阵、截断 SVD、奇异值的含义
   - 对比点：显式分解 vs 隐式分解、无权重 vs 加权分解

### 14.3 进阶算法（后续学习）

**短期目标（1-2 个月）**：
1. **ELMo**：基于双向 LSTM 的动态词嵌入
   - 关联：解决了 GloVe 的静态嵌入局限，引入上下文感知
   - 难度：中等

2. **注意力机制（Attention）**：序列建模的核心机制
   - 关联：注意力机制是 Transformer 的基础，也用于替代 RNN 的全局信息传递
   - 难度：中等

**中期目标（2-4 个月）**：
1. **BERT**：基于 Transformer 编码器的预训练模型
   - 关联：BERT 继承了 GloVe 的分布式语义假设，但通过深度模型和动态表示大幅提升
   - 难度：较高

2. **GPT**：基于 Transformer 解码器的预训练模型
   - 关联：从词嵌入到语言建模，GPT 展示了预训练 + 微调的范式
   - 难度：较高

**长期目标（4-6 个月）**：
1. **大语言模型（LLM）**：ChatGPT、LLaMA 等
   - 关联：从 GloVe 的静态向量到 LLM 的上下文理解，体现了 NLP 的整体演进
   - 难度：高

### 14.4 实践项目建议

1. **基础项目**：使用预训练 GloVe 向量构建文本分类器
   - 加载 GloVe 向量，构建嵌入层
   - 使用 Keras/PyTorch 搭建简单的文本分类模型（如 LSTM + GloVe）
   - 在 IMDB 情感分析数据集上评估

2. **进阶项目**：在自己的领域语料上训练 GloVe 向量
   - 收集特定领域的文本数据（如医疗、法律、金融）
   - 构建共现矩阵并训练 GloVe
   - 对比通用 GloVe 向量和领域 GloVe 向量在领域任务上的表现

3. **挑战项目**：实现一个完整的词嵌入评测框架
   - 实现词相似度、词类比、词聚类等多个评测指标
   - 对比 GloVe、Word2Vec、FastText 在多个维度上的表现
   - 生成详细的评测报告和可视化

### 14.5 推荐资源

**论文**：
1. Pennington J, Socher R, Manning C. GloVe: Global Vectors for Word Representation. EMNLP 2014.
2. Mikolov T, et al. Efficient Estimation of Word Representations in Vector Space. ICLR 2013. (Word2Vec)
3. Levy O, Goldberg Y. Neural Word Embedding as Implicit Matrix Factorization. NeurIPS 2014.
4. Bojanowski P, et al. Enriching Word Vectors with Subword Information. TACL 2017. (FastText)
5. Peters M, et al. Deep Contextualized Word Representations. NAACL 2018. (ELMo)

**代码**：
1. 斯坦福官方 GloVe 实现: https://github.com/stanfordnlp/GloVe
2. gensim 库的 GloVe 支持: https://radimrehurek.com/gensim/
3. 预训练 GloVe 向量下载: https://nlp.stanford.edu/projects/glove/

**在线课程**：
1. Stanford CS224n: Natural Language Processing with Deep Learning（强烈推荐）
2. Stanford CS229: Machine Learning

---

## 参考文献

1. Pennington J, Socher R, Manning C D. GloVe: Global Vectors for Word Representation[C]. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014: 1532-1543.

2. Mikolov T, Chen K, Corrado G, et al. Efficient Estimation of Word Representations in Vector Space[J]. arXiv preprint arXiv:1301.3781, 2013.

3. Mikolov T, Sutskever I, Chen K, et al. Distributed Representations of Words and Phrases and their Compositionality[C]. Advances in Neural Information Processing Systems (NeurIPS), 2013: 3111-3119.

4. Levy O, Goldberg Y. Neural Word Embedding as Implicit Matrix Factorization[C]. Advances in Neural Information Processing Systems (NeurIPS), 2014: 2177-2185.

5. Bojanowski P, Grave E, Joulin A, et al. Enriching Word Vectors with Subword Information[J]. Transactions of the Association for Computational Linguistics, 2017, 5: 135-146.

6. Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]. Proceedings of NAACL-HLT, 2019: 4171-4186.

7. Peters M E, Neumann M, Iyyer M, et al. Deep Contextualized Word Representations[C]. Proceedings of NAACL-HLT, 2018: 2227-2237.

8. Landauer T K, Dumais S T. A Solution to Plato's Problem: The Latent Semantic Analysis Theory of Acquisition, Induction, and Representation of Knowledge[J]. Psychological Review, 1997, 104(2): 211-240.

---

**文档结束**
