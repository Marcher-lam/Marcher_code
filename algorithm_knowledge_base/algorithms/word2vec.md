# Word2Vec 学习文档

> 将自然语言词汇映射为低维稠密向量表示,使语义相近的词在向量空间中彼此接近

---

## 1. 算法基础认知

**一句话定义**：通过浅层神经网络在大规模语料上学习词的分布式向量表示,使上下文相似的词具有相近的词向量。

**直觉类比**：判断一个人的品格,不必直接了解他本人,只需看他结交的朋友——如果两个人的朋友圈高度重合,那么这两个人也很可能志趣相投。Word2Vec 的核心假设与此同理:如果两个词经常出现在相同的上下文环境(即"朋友"相同)中,那么这两个词的语义也应该是相近的。例如,"猫"和"狗"经常都和"宠物""可爱""毛"等词共同出现,因此它们的词向量会被训练得彼此接近。

**历史背景**：
Word2Vec 由 Tomas Mikolov 等人于 2013 年在 Google 团队提出,先后发表了两篇具有里程碑意义的论文:
- 第一篇论文 "Efficient Estimation of Word Representations in Vector Space"(ICLR 2013)提出了 CBOW 和 Skip-gram 两种模型架构。
- 第二篇论文 "Distributed Representations of Words and Phrases and their Compositionality"(NeurIPS 2013)进一步引入了负采样(Negative Sampling)和子词(Subword)信息等技术,大幅提升了训练效率。

Word2Vec 的出现使词嵌入训练从数天缩短到数小时,是深度学习时代自然语言处理的基石之一。在 Word2Vec 之前,Bengio 于 2003 年提出的 NNLM(Neural Network Language Model)已经证明了用神经网络学习词表示的可行性,但其训练速度过慢;Word2Vec 通过简化网络结构和引入高效的近似算法,使大规模词向量训练成为现实。

**算法定位**：
- 类型：无监督学习 --> 表示学习(Representation Learning)
- 输出：每个词对应一个 $d$ 维的稠密实数向量(通常 $d \in [100, 300]$)
- 模型类型：前馈神经网络,属于生成模型(通过建模语言模型来学习表示)

**前置知识**：
- 神经网络基础：前馈网络、反向传播、梯度下降
- 语言模型：$n$-gram 语言模型、链式法则
- 独热编码(One-Hot Encoding)：理解离散符号为何需要向量化
- 概率论基础：条件概率、最大似然估计
- 优化方法：随机梯度下降(SGD)、学习率调度

---

## 2. 核心原理

### 2.1 核心思想

Word2Vec 的核心思想建立在语言学中的**分布式假设**(Distributional Hypothesis)之上。该假设由英国语言学家 J.R. Firth 于 1957 年提出,其核心表述为:

> "You shall know a word by the company it keeps."(通过一个词的伴随词来了解它)

这个假设的含义是:一个词的语义由它所在的上下文环境决定。如果两个词经常出现在相似的上下文中,那么它们很可能具有相似的语义。

基于分布式假设,Word2Vec 设计了一种巧妙的训练任务:不直接告诉模型词的含义,而是让模型完成"猜词"游戏。在猜词的过程中,模型被迫学习词与词之间的共现关系,最终每个词被压缩为一个低维稠密向量,使得语义相近的词在向量空间中距离较近。

核心思想可以概括为:**用一个简单的预测任务(预测上下文或预测中心词)作为代理任务(proxy task),迫使神经网络学习到能够捕获语义关系的词向量表示**。

### 2.2 工作流程

Word2Vec 包含两种等价但方向相反的模型架构:

1. **CBOW(Continuous Bag-of-Words, 连续词袋模型)**
   - 输入：中心词的上下文词(周围的 $2c$ 个词)
   - 输出：预测中心词
   - 思路：已知"周围的词"是什么,猜"中间的词"是什么

2. **Skip-gram(跳字模型)**
   - 输入：中心词
   - 输出：预测上下文词(周围的 $2c$ 个词)
   - 思路：已知"中间的词"是什么,猜"周围的词"是什么

以 Skip-gram 为例,完整的训练流程如下:

1. **语料准备**：将文本分词,构建词汇表,建立词到索引的映射。
2. **滑动窗口采样**：用一个固定大小的窗口(如 $c=5$)在语料上滑动,每个窗口提取一个训练样本(中心词, 上下文词)。
3. **模型训练**：将中心词输入神经网络,预测上下文词的概率分布,通过最大化正确预测的概率来更新词向量。
4. **参数提取**：训练完成后,提取嵌入矩阵中的行向量,作为每个词的最终词向量表示。

### 2.3 关键概念解释

- **分布式表示(Distributed Representation)**：用一个低维稠密向量来表示一个词,向量的每个维度不一定有明确的人为定义的含义,但整个向量编码了词的语义信息。这与独热编码(One-Hot)形成鲜明对比——独热编码是 $|V|$ 维的稀疏向量,任意两个词的独热编码都相互正交,无法表达词之间的语义关系。

- **上下文窗口(Context Window)**：以目标词为中心,向左右各取 $c$ 个词作为上下文。窗口大小 $c$ 是一个重要的超参数。较小的窗口捕获词之间的句法关系(如"the cat sat"),较大的窗口捕获语义关系(如同义关系)。

- **负采样(Negative Sampling)**：为了加速训练,不使用完整的 Softmax(需要遍历整个词典),而是从词典中随机采样若干"负样本",将原本的多分类问题转化为多个二分类问题。

- **层次 Softmax(Hierarchical Softmax)**：另一种加速方法,利用哈夫曼树(Huffman Tree)将多分类转化为 $\log|V|$ 次二分类。高频词在树的浅层,只需少量计算即可得到其概率。

- **子采样(Subsampling)**：高频词(如"the""a""is")携带的语义信息较少,但出现频率极高,会导致训练样本不均衡。子采样以一定概率丢弃高频词的训练样本,加速训练并提升词向量质量。

### 2.4 几何/直观解释

在 Word2Vec 学到的向量空间中,词向量的几何关系蕴含了丰富的语义信息:

- **语义相似性**：语义相近的词在向量空间中距离较近。例如,"猫"和"狗"的余弦相似度会远高于"猫"和"逻辑"。
- **词类比(Word Analogy)**：词向量之间的加减运算能够捕获语义关系。最经典的例子是:
  $$\text{vec("king")} - \text{vec("man")} + \text{vec("woman")} \approx \text{vec("queen")}$$
  这说明"king"到"man"的偏移量(性别差异)与"queen"到"woman"的偏移量一致。
- **向量方向编码语义属性**：在向量空间中的某些方向可能对应着特定的语义属性,如性别方向、时态方向、单复数方向等。

这些几何性质并非预先设计,而是模型在大量语料上通过预测上下文任务自动学到的。这是分布式表示最令人惊叹的特性之一。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度/说明 |
|------|------|-----------|
| $V$ | 词汇表 | 大小为 $\|V\|$ |
| $d$ | 词向量维度 | 通常 $d \in [100, 300]$ |
| $w$ | 词汇表中的某个词 | - |
| $\mathbf{v}_w$ | 词 $w$ 的输入向量(中心词向量) | $\mathbb{R}^d$ |
| $\mathbf{u}_w$ | 词 $w$ 的输出向量(上下文词向量) | $\mathbb{R}^d$ |
| $c$ | 上下文窗口大小 | 正整数 |
| $w_t$ | 语料中位置 $t$ 的词 | - |
| $w_c$ | 中心词 | - |
| $P(w\_o \| w\_c)$ | 给定中心词 $w\_c$ 时输出词 $w\_o$ 的概率 | 标量 |
| $K$ | 负样本数量 | 通常 $K \in [5, 20]$ |
| $f(w)$ | 词 $w$ 在语料中的频率 | 标量 |

### 3.2 问题形式化

给定一个大规模文本语料 $\mathcal{C}$,我们的目标是学习一个映射函数:

$$\phi: V \rightarrow \mathbb{R}^d$$

使得在向量空间 $\mathbb{R}^d$ 中,语义相近的词的向量表示彼此接近。Word2Vec 通过最大化语料上的**对数似然**(log-likelihood)来实现这一目标:

$$\max \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} \mid w_t)$$

其中 $T$ 是语料中词的总数,$c$ 是窗口大小。

### 3.3 目标函数/损失函数

#### 3.3.1 Skip-gram 的完整 Softmax 目标函数

在 Skip-gram 模型中,给定中心词 $w\_c$,预测上下文词 $w\_o$ 的概率定义为:

$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c})}{\sum_{w \in V} \exp(\mathbf{u}_w^T \mathbf{v}_{w_c})}$$

上式本质上就是 Softmax 函数,其中:
- 分子 $\exp(\mathbf{u}_{w\_o}^T \mathbf{v}\_{w\_c})$ 度量了中心词和上下文词的匹配程度
- 分母对所有词汇的匹配程度进行归一化
- $\mathbf{v}\_{w\_c}$ 是中心词的输入向量, $\mathbf{u}\_{w\_o}$ 是上下文词的输出向量

Skip-gram 的目标函数是最大化整个语料上的对数似然:

$$\mathcal{L} = \sum_{w \in V} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{c+j} \mid w_c)$$

等价地,最小化负对数损失:

$$\mathcal{L}_{\text{NLL}} = -\sum_{t=1}^{T} \sum_{\substack{-c \leq j \leq c \\ j \neq 0}} \log P(w_{t+j} \mid w_t)$$

**为什么选择这个目标函数?**
1. **概率解释**：这个目标函数等价于对语言模型的**最大似然估计**(MLE),即找到使观测到的语料出现概率最大的参数。
2. **信息论解释**：最大化对数似然等价于最小化模型分布与真实分布之间的**KL 散度**(Kullback-Leibler Divergence)。
3. **实践效果**：尽管这是一个"代理任务",但大量实验表明,在这个任务上学到的词向量确实能捕获丰富的语义信息。

#### 3.3.2 CBOW 的目标函数

CBOW 与 Skip-gram 方向相反。给定上下文词 $w\_{c-c}, \ldots, w\_{c-1}, w\_{c+1}, \ldots, w\_{c+c}$,预测中心词 $w\_c$:

$$P(w_c \mid \text{context}) = \frac{\exp(\mathbf{u}_{w_c}^T \mathbf{h})}{\sum_{w \in V} \exp(\mathbf{u}_w^T \mathbf{h})}$$

其中 $\mathbf{h}$ 是上下文词向量的平均值:

$$\mathbf{h} = \frac{1}{2c} \sum_{\substack{-c \leq j \leq c \\ j \neq 0}} \mathbf{v}_{w_{c+j}}$$

### 3.4 推导过程

#### 3.4.1 噪声对比估计(Noise Contrastive Estimation, NCE)

完整 Softmax 的计算复杂度为 $O(\|V\|)$,对于大规模词典(数十万甚至数百万词)来说代价过高。为了解决这个问题,Word2Vec 引入了**负采样**(Negative Sampling)技术,其理论基础来源于噪声对比估计。

NCE 的核心思想是:将一个概率归一化问题转化为一个二分类问题。具体来说,不直接计算 $P(w\_o \| w\_c)$,而是训练一个二分类器来区分"真实样本"(来自数据的中心词-上下文词对)和"噪声样本"(随机采样的词对)。

**NCE 推导**:

设数据分布为 $P\_D(w)$(即语料中词的真实分布),噪声分布为 $P\_N(w)$。对于一个观测到的词对 $(w\_c, w\_o)$,定义其来自数据分布的概率为 $P(D=1 \| w\_c, w\_o)$。

根据贝叶斯定理:

$$P(D=1 \mid w_c, w_o) = \frac{P_D(w_o \mid w_c)}{P_D(w_o \mid w_c) + k \cdot P_N(w_o)}$$

其中 $k$ 是噪声样本的数量。令 $P\_D(w\_o \| w\_c) \propto \exp(\mathbf{u}\_{w\_o}^T \mathbf{v}\_{w\_c})$,代入上式并化简,可以近似得到:

$$P(D=1 \mid w_c, w_o) = \sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c} - \log k \cdot P_N(w_o))$$

其中 $\sigma(x) = \frac{1}{1+e^{-x}}$ 是 sigmoid 函数。

最大化 NCE 的目标等价于最大化以下函数:

$$\mathcal{L}_{\text{NCE}} = \log \sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_N}[\log \sigma(-\mathbf{u}_{w_i}^T \mathbf{v}_{w_c})]$$

这就是负采样的理论来源。

#### 3.4.2 负采样的损失函数与梯度推导

Word2Vec 中实际使用的负采样损失函数是对 NCE 的简化。对于每个正样本 $(w\_c, w\_o)$ 和 $K$ 个负样本 $\{w\_1, w\_2, \ldots, w\_K\}$,损失函数定义为:

$$\mathcal{L} = \log \sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c}) + \sum_{i=1}^{K} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-\mathbf{u}_{w_i}^T \mathbf{v}_{w_c})]$$

其中 $P\_n(w)$ 是负采样分布(下文详细解释),$\sigma(x) = \frac{1}{1+e^{-x}}$ 是 sigmoid 函数。

**对 $\mathbf{v}\_{w\_c}$ 的梯度推导**:

利用 sigmoid 函数的导数 $\sigma'(x) = \sigma(x)(1-\sigma(x))$,我们分别对正样本项和负样本项求导:

正样本项的梯度:

$$\frac{\partial}{\partial \mathbf{v}_{w_c}} \log \sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c}) = \frac{\sigma'(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c})}{\sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c})} \cdot \mathbf{u}_{w_o} = (1 - \sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c})) \cdot \mathbf{u}_{w_o}$$

负样本项的梯度(对第 $i$ 个负样本):

$$\frac{\partial}{\partial \mathbf{v}_{w_c}} \log \sigma(-\mathbf{u}_{w_i}^T \mathbf{v}_{w_c}) = \frac{\sigma'(-\mathbf{u}_{w_i}^T \mathbf{v}_{w_c}) \cdot (-\mathbf{u}_{w_i})}{\sigma(-\mathbf{u}_{w_i}^T \mathbf{v}_{w_c})} = -\sigma(\mathbf{u}_{w_i}^T \mathbf{v}_{w_c}) \cdot \mathbf{u}_{w_i}$$

综合以上,对 $\mathbf{v}\_{w\_c}$ 的总梯度为:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_c}} = (1 - \sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c})) \cdot \mathbf{u}_{w_o} - \sum_{i=1}^{K} \sigma(\mathbf{u}_{w_i}^T \mathbf{v}_{w_c}) \cdot \mathbf{u}_{w_i}$$

**对 $\mathbf{u}\_{w\_o}$ 的梯度推导**:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{w_o}} = (1 - \sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c})) \cdot \mathbf{v}_{w_c}$$

**对负样本输出向量 $\mathbf{u}\_{w\_i}$ 的梯度推导**:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{w_i}} = -\sigma(\mathbf{u}_{w_i}^T \mathbf{v}_{w_c}) \cdot \mathbf{v}_{w_c}$$

**梯度更新规则**:

使用随机梯度上升(SGD,因为我们要最大化 $\mathcal{L}$):

$$\mathbf{v}_{w_c} \leftarrow \mathbf{v}_{w_c} + \eta \frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_c}}$$

$$\mathbf{u}_{w_o} \leftarrow \mathbf{u}_{w_o} + \eta \frac{\partial \mathcal{L}}{\partial \mathbf{u}_{w_o}}$$

$$\mathbf{u}_{w_i} \leftarrow \mathbf{u}_{w_i} + \eta \frac{\partial \mathcal{L}}{\partial \mathbf{u}_{w_i}}, \quad i = 1, \ldots, K$$

其中 $\eta$ 是学习率。

#### 3.4.3 负采样分布 $P\_n(w) = \frac{f(w)^{3/4}}{\sum_{w' \in V} f(w')^{3/4}}$ 的解释

在负采样中,负样本从分布 $P\_n(w)$ 中采样,其中 $f(w)$ 是词 $w$ 在语料中的归一化频率。原始论文选择 $f(w)^{3/4}$ 而非直接使用 $f(w)$,这个 3/4 次幂的设计背后有深刻的考量。

**为什么使用平滑后的频率分布?**

1. **频率悬殊问题**：在自然语言中,词频分布近似服从 Zipf 定律——少数高频词(如"the")的频率远高于大多数低频词。如果直接按 $f(w)$ 采样,负样本几乎全部由高频词构成,模型难以区分高频词之间的差异。

2. **3/4 次幂的效果**：设 $f(w) \in (0, 1)$,则 $f(w)^{3/4} > f(w)$。这意味着 3/4 次幂起到了"频率平滑"的作用——提升了低频词被采样的概率,同时略微降低了高频词的采样概率。

   用数学来分析:对于两个词 $w\_1$ 和 $w\_2$,假设 $f(w\_1) > f(w\_2)$,则有:

   $$\frac{f(w_1)^{3/4}}{f(w_2)^{3/4}} = \left(\frac{f(w_1)}{f(w_2)}\right)^{3/4} < \frac{f(w_1)}{f(w_2)}$$

   这说明 3/4 次幂缩小了高频词和低频词之间的采样概率差距,使负样本的分布更加均匀。

3. **实验选择**：Mikolov 等人在论文中实验了不同的幂次($1/2, 2/3, 3/4, 1$),发现 $3/4$ 在多个任务上表现最佳。这个值在频率差异的"压缩"和"保留"之间取得了较好的平衡。

**极端情况分析**:
- 如果幂次为 1:等价于直接按语料频率采样,高频词占主导。
- 如果幂次为 0:等价于均匀采样,忽略了词频信息。
- $3/4$ 介于两者之间,既保留了频率信息(高频词仍然更容易被采样),又给予低频词一定的曝光机会。

### 3.5 最终解/算法步骤

由于 Word2Vec 的目标函数是非凸的(多层神经网络的标准特点),因此不存在解析解,需要通过迭代优化来求解。

**Skip-gram + 负采样算法伪代码**:

```
输入: 语料 C, 窗口大小 c, 词向量维度 d, 负样本数 K, 学习率 eta, 训练轮数 epochs
输出: 词向量矩阵 V (|V| x d)

1. 初始化:
   - 构建词汇表 V, 统计每个词的频率 f(w)
   - 随机初始化中心词向量矩阵 W_in (|V| x d)
   - 随机初始化上下文词向量矩阵 W_out (|V| x d)
   - 构建负采样分布表(基于 f(w)^{3/4})

2. for epoch = 1 to epochs:
      随机打乱语料顺序
      for 语料中的每个位置 t:
          w_c = 语料中位置 t 的词(中心词)

          # 子采样:以概率 P_discard 丢弃高频词
          if random() < P_discard(w_c):
              continue

          for j in [-c, -(c-1), ..., -1, 1, ..., c-1, c]:
              if t+j 超出语料范围: continue

              w_o = 语料中位置 t+j 的词(正样本上下文词)

              # 梯度累加器初始化
              grad_v_wc = 0  (d维零向量)

              # 处理正样本
              score = sigmoid(u_{w_o}^T * v_{w_c})
              grad = (1 - score) * eta
              grad_v_wc += grad * u_{w_o}
              u_{w_o} += grad * v_{w_c}

              # 处理 K 个负样本
              for k = 1 to K:
                  w_k = 从负采样分布中采样
                  score = sigmoid(u_{w_k}^T * v_{w_c})
                  grad = -score * eta
                  grad_v_wc += grad * u_{w_k}
                  u_{w_k} += grad * v_{w_c}

              # 更新中心词向量
              v_{w_c} += grad_v_wc

      # 线性衰减学习率
      eta = eta * (1 - epoch/epochs)

3. 返回 W_in 作为最终词向量
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

Word2Vec 的数据预处理虽然不复杂,但有几个关键步骤需要正确处理:

**必要预处理**:

1. **分词(Tokenization)**:
   - 英文:按空格和标点分割,去除特殊字符
   - 中文:使用分词工具(如 jieba)将连续文本分割为词序列
   - 分词质量直接影响词向量质量

   ```python
   import jieba

   # 中文分词示例
   text = "自然语言处理是人工智能的重要方向"
   words = list(jieba.cut(text))
   print(words)  # ['自然语言', '处理', '是', '人工智能', '的', '重要', '方向']
   ```

2. **低频词过滤**:
   - 出现次数少于阈值(如 5 次)的词通常被丢弃
   - 原因:低频词的训练样本太少,学不到好的表示;而且低频词可能是拼写错误
   - 注意:过滤阈值需要根据语料规模调整

   ```python
   from collections import Counter

   min_count = 5
   word_counts = Counter(all_words)
   # 只保留出现次数 >= min_count 的词
   vocab = {w: c for w, c in word_counts.items() if c >= min_count}
   ```

3. **构建词汇表**:
   - 为每个词分配唯一索引
   - 统计词频,用于后续的子采样和负采样分布

   ```python
   word2idx = {word: idx for idx, (word, count) in enumerate(vocab.items())}
   idx2word = {idx: word for word, idx in word2idx.items()}
   vocab_size = len(word2idx)
   ```

### 4.2 参数初始化

Word2Vec 使用两组独立的词向量:

- **输入向量(中心词向量) $W\_{\text{in}}$**: 维度 $\|V\| \times d$,训练完成后作为最终词向量
- **输出向量(上下文词向量) $W\_{\text{out}}$**: 维度 $\|V\| \times d$,仅用于训练过程

**初始化方法**:

```python
import numpy as np

# 使用均匀分布随机初始化
# 权重范围: [-0.5/d, 0.5/d] (与原始实现一致)
d = 300  # 词向量维度
vocab_size = 10000

W_in = (np.random.rand(vocab_size, d) - 0.5) / d
W_out = np.zeros((vocab_size, d))  # 输出向量初始化为零
```

**为什么输出向量初始化为零?**
- 原始 Word2Vec 的 C 语言实现中,输出向量初始化为零向量
- 这是一种启发式做法:初始时让 sigmoid 输出接近 0.5(因为 $\sigma(0) = 0.5$),表示模型对正负样本"不确定"
- 输入向量随机初始化是为了打破对称性,让不同词有不同的初始表示

### 4.3 迭代过程

```
初始化参数 W_in, W_out
for epoch in range(max_epochs):
    随机打乱训练样本
    for 每个训练样本 (w_c, w_o):
        # 前向传播: 计算正样本和负样本的得分
        pos_score = sigmoid(W_out[w_o] @ W_in[w_c])
        neg_scores = [sigmoid(W_out[w_k] @ W_in[w_c]) for w_k in negative_samples]

        # 计算梯度并更新
        grad_in = (1 - pos_score) * W_out[w_o]
        W_out[w_o] += (1 - pos_score) * W_in[w_c]
        for w_k in negative_samples:
            neg_score = neg_scores[k]
            grad_in -= neg_score * W_out[w_k]
            W_out[w_k] -= neg_score * W_in[w_c]
        W_in[w_c] += grad_in * learning_rate

    # 学习率线性衰减
    learning_rate = initial_lr * (1.0 - epoch / max_epochs)
    learning_rate = max(learning_rate, initial_lr * 0.0001)  # 下界保护
```

### 4.4 收敛条件

Word2Vec 通常不使用传统的收敛条件(如梯度接近零),而是采用以下策略:

1. **固定训练轮数**: 通常训练 5-15 个 epoch
2. **学习率线性衰减**: 学习率从初始值线性衰减到一个很小的值(如初始值的万分之一),确保后期参数调整幅度小,模型趋于稳定
3. **经验法则**: Google 原始论文推荐训练一个 epoch 就能获得不错的词向量,但实际应用中通常训练多个 epoch 以获得更稳定的表示

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值(原始实现) | 调参建议 |
|--------|------|----------|------------------|----------|
| `vector_size` | 词向量维度 | 100-500 | 300 | 数据量大可用更大维度;一般任务 100-200 足够 |
| `window` | 上下文窗口大小 | 3-10 | 5 | 小窗口捕获句法,大窗口捕获语义 |
| `min_count` | 最低词频阈值 | 1-20 | 5 | 语料大可提高阈值;小语料降低至 1-2 |
| `sg` | 模型选择(0=CBOW, 1=Skip-gram) | {0, 1} | 0(CBOW) | Skip-gram 在小语料上效果更好;CBOW 训练更快 |
| `negative` | 负样本数量 | 5-20 | 5 | 小数据集用 5-10;大数据集可增加至 15-20 |
| `epochs` | 训练轮数 | 5-20 | 5 | 数据量大可减少轮数;小数据集增加轮数 |
| `alpha` | 初始学习率 | 0.001-0.05 | 0.025 | 一般不需要调,线性衰减策略已经很鲁棒 |
| `sample` | 子采样阈值 | 1e-5 - 1e-3 | 1e-3 | 值越小,高频词越容易被丢弃 |
| `hs` | 是否使用层次 Softmax | {0, 1} | 0(负采样) | 负采样通常更快;小词典可尝试层次 Softmax |
| `workers` | 并行线程数 | 取决于 CPU | 3 | 多核 CPU 可增大以提高训练速度 |

**CBOW vs Skip-gram 的选择**:
- **CBOW**: 训练速度更快(每个窗口只做一次预测),对高频词的表示效果较好
- **Skip-gram**: 对低频词的表示效果更好,在大规模数据上通常整体表现更优
- 实践建议:语料较大(> 1 亿词)时优先使用 Skip-gram;语料较小时使用 CBOW

---

## 5. 应用场景

### 5.1 典型应用

**应用 1：计算词相似度(Word Similarity)**

- 问题类型：无监督评估/特征提取
- 为什么适合：Word2Vec 的核心设计目标就是使语义相似的词在向量空间中距离接近
- 实际案例：
  - 搜索引擎中的同义词扩展:用户搜索"开心"时,返回包含"高兴""快乐""愉快"的结果
  - 智能输入法的联想推荐:用户输入"北京"后,推荐"上海""深圳""广州"等城市名

**应用 2：词类比(Word Analogy)**

- 问题类型：语义推理
- 为什么适合：词向量的线性运算能够捕获语义关系,如性别、地理、时态等
- 实际案例：
  - $\text{vec("king")} - \text{vec("man")} + \text{vec("woman")} \approx \text{vec("queen")}$ (性别关系)
  - $\text{vec("Paris")} - \text{vec("France")} + \text{vec("Italy")} \approx \text{vec("Rome")}$ (首都-国家关系)
  - $\text{vec("walked")} - \text{vec("walk")} + \text{vec("swim")} \approx \text{vec("swam")}$ (时态关系)

**应用 3：文本分类(Text Classification)**

- 问题类型：有监督分类
- 为什么适合：将文本中的所有词的词向量取平均(或通过更复杂的聚合方式)作为文本的表示,然后输入分类器
- 实际案例：
  - 情感分析:将评论中词向量取平均,用逻辑回归或 SVM 分类正面/负面情感
  - 垃圾邮件检测:将邮件文本转为词向量表示,训练二分类器
  - 新闻分类:将新闻正文转为向量表示,分类为体育/科技/娱乐等类别

**应用 4：推荐系统(Recommendation System)**

- 问题类型：序列预测/推荐
- 为什么适合：将用户的行为序列(如浏览/购买的商品序列)视为"文本",用 Word2Vec 训练商品嵌入,利用商品相似性进行推荐
- 实际案例：
  -电商推荐:将用户的购买历史视为"句子",商品视为"词",训练得到商品嵌入
  - 音乐推荐:将用户的播放序列视为"句子",歌曲视为"词"

**应用 5：命名实体识别(NER)等下游 NLP 任务**

- 问题类型：序列标注
- 为什么适合：Word2Vec 词向量作为预训练特征,可以替代或补充传统的独热编码,提升下游任务的性能
- 实际案例：
  - 将 Word2Vec 词向量作为 CRF、BiLSTM-CRF 等序列标注模型的输入特征
  - 在缺乏大量标注数据的场景中,Word2Vec 提供的预训练表示能显著提升模型表现

### 5.2 适用数据特征

Word2Vec 适合的数据特征:
- **数据类型**:纯文本语料,无需任何标注
- **数据规模**:中等规模(百万级词)到大规模(十亿级词)效果最佳
- **噪声容忍度**:较高,对少量噪声文本不敏感
- **领域适配性**:通用语料训练的词向量可用于多个领域,但领域特定语料训练的效果更佳

### 5.3 不适用场景

**不适合的情况**:
1. **一词多义(Polysemy)**:Word2Vec 为每个词只分配一个固定的向量表示,无法区分同一词在不同语境下的不同含义。例如,"bank"在"river bank"(河岸)和"bank account"(银行账户)中含义完全不同,但 Word2Vec 只会为它学习一个向量。
2. **未登录词(Out-of-Vocabulary, OOV)**:不在词汇表中的词无法获得词向量表示。这在处理专业术语、新词、网络用语时尤其突出。
3. **短文本场景**:Word2Vec 依赖词的共现统计,短文本(如搜索查询、推文)中词的共现信息不足,训练效果较差。
4. **需要精细语境理解的任务**:如机器翻译、阅读理解等需要深度理解上下文的任务,静态词向量的表达能力有限。

---

## 6. 优缺点分析

### 6.1 优点

1. **训练效率极高**
   - 技术细节:负采样将每次更新的复杂度从 $O(\|V\|)$ 降至 $O(K)$,其中 $K$ 是负样本数(通常 $K=5 \sim 20$)
   - 在 Google 发布的原始代码中,使用单线程在一小时之内即可在数十亿词的语料上完成训练
   - 适用场景:需要在短时间内处理大规模文本的场景

2. **词向量质量优秀**
   - Word2Vec 学到的词向量具有良好的代数性质,能捕获丰富的语义关系
   - 在词相似度、词类比等标准评测任务上表现优异
   - 作为下游任务的预训练特征,通常能显著提升模型性能

3. **实现简洁,易于使用**
   - 原始 C 实现只有约 1000 行代码,算法原理清晰
   - Gensim 库提供了高度优化的 Python 接口,几行代码即可完成训练
   - 不需要 GPU,普通 CPU 即可训练

4. **无需标注数据**
   - 完全基于无监督学习,只需纯文本语料
   - 可以充分利用互联网上海量的文本资源

### 6.2 缺点

1. **一词多义问题**
   - 问题场景:同一个词在不同语境中含义不同(如"苹果"可以指水果或公司)
   - 解决思路:使用上下文相关的动态词向量(如 ELMo、BERT)

2. **未登录词(OOV)问题**
   - 问题场景:新词、专有名词、拼写变体等不在词汇表中
   - 解决思路:使用字符级或子词级别的嵌入(如 FastText、BERT 的 WordPiece)

3. **静态词向量的局限性**
   - 问题场景:词向量不随上下文变化,无法表达词在不同语境中的细微语义差异
   - 解决思路:使用基于 Transformer 的上下文化词嵌入(如 BERT、GPT)

4. **窗口大小的权衡**
   - 问题场景:较小的窗口有利于捕获句法关系但不利于语义关系,较大的窗口则相反,难以同时兼顾两者
   - 解决思路:使用 GloVe(同时考虑局部和全局共现信息)或动态词嵌入

### 6.3 与同类算法对比

| 维度 | Word2Vec | GloVe | LSA | BERT |
|------|----------|-------|-----|------|
| 核心思想 | 局部上下文预测 | 全局共现矩阵分解 | 全局共现矩阵(SVD) | 上下文化的深度语言模型 |
| 训练数据 | 大规模纯文本 | 大规模纯文本 | 大规模纯文本 | 大规模纯文本 + 预训练任务 |
| 词向量类型 | 静态 | 静态 | 静态 | 动态(上下文相关) |
| OOV 处理 | 不支持 | 不支持 | 不支持 | 通过子词单元支持 |
| 多义词 | 不支持 | 不支持 | 不支持 | 支持 |
| 训练速度 | 快 | 中等 | 中等 | 慢(需要 GPU) |
| 计算资源 | CPU 即可 | CPU 即可 | CPU 即可 | 需要 GPU/TPU |
| 词向量维度 | 100-300 | 50-300 | 100-500 | 768-1024 |
| 预训练复杂度 | 低 | 中 | 中 | 高 |
| 下游任务效果 | 中等 | 中等 | 中等 | 优秀 |
| 模型参数量 | 百万级 | 百万级 | 百万级 | 亿级-千亿级 |
| 可解释性 | 较好(线性运算) | 较好(线性运算) | 较好(线性运算) | 较差(深度黑箱) |
| 适用场景 | 快速获取词向量 | 需要利用全局信息 | 传统 NLP pipeline | 复杂 NLP 任务 |

**详细对比分析**:

- **Word2Vec vs GloVe**: GloVe 利用全局词共现统计(构建共现矩阵后分解),而 Word2Vec 只利用局部窗口内的共现信息。GloVe 在某些任务上优于 Word2Vec,但 Word2Vec 的训练更加灵活,且 Skip-gram 对低频词的学习更好。

- **Word2Vec vs LSA**: LSA(潜在语义分析)通过 SVD 分解词-文档共现矩阵得到词向量,是一种传统方法。Word2Vec 相比 LSA 的优势在于:(1)训练效率更高;(2)能够捕获更丰富的语义关系(如词类比);(3)可扩展到更大规模的数据。

- **Word2Vec vs BERT**: BERT 是基于 Transformer 的深度预训练模型,生成的词向量是上下文相关的(同一个词在不同语境中有不同的表示)。BERT 在几乎所有 NLP 任务上远超 Word2Vec,但其代价是模型大得多、训练成本高得多。Word2Vec 在以下场景仍有优势:(1)资源受限的环境;(2)需要快速部署;(3)对词向量的可解释性有要求。

---

## 7. 调库实现

### 7.1 环境准备

```bash
# 安装必要库
pip install numpy matplotlib scikit-learn gensim jieba
```

### 7.2 完整代码示例

```python
"""
Word2Vec 调库实现
数据集：模拟中文维基百科语料(小规模示例)
目标：训练词向量,进行相似度计算、词类比和 t-SNE 可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from collections import Counter
import jieba

# 设置随机种子,保证可复现
np.random.seed(42)

# ===============================
# 1. 数据准备：构建中文语料
# ===============================
def prepare_chinese_corpus():
    """
    准备中文语料数据
    模拟包含多个主题的中文文本数据集

    Returns:
        sentences: 分词后的句子列表,每个元素是一个词列表
    """
    # 模拟的中文语料,涵盖多个主题
    raw_texts = [
        "机器学习是人工智能的重要分支", "深度学习在计算机视觉领域取得了巨大突破",
        "自然语言处理让计算机理解人类语言", "神经网络是深度学习的基础模型",
        "卷积神经网络广泛应用于图像识别", "循环神经网络擅长处理序列数据",
        "Transformer 模型改变了自然语言处理的面貌", "注意力机制是 Transformer 的核心",
        "预训练模型如 BERT 和 GPT 在 NLP 任务上表现优异",
        "强化学习通过与环境交互来学习策略", "迁移学习利用已有知识加速新任务的学习",
        "数据挖掘从大量数据中发现有价值的信息", "推荐系统利用用户行为数据进行个性化推荐",
        "搜索引擎帮助用户快速找到需要的信息", "知识图谱构建结构化的世界知识",
        "语音识别技术将语音转换为文字", "图像生成技术可以创造逼真的图像",
        "自动驾驶需要计算机视觉和深度学习技术", "医疗影像分析利用深度学习辅助诊断",
        "金融风控利用机器学习模型检测欺诈", "智能客服通过自然语言处理理解用户需求",
        "数据增强技术可以扩充训练数据集", "模型压缩技术减小模型体积提高推理速度",
        "梯度下降是最基本的优化算法", "反向传播算法计算神经网络的梯度",
        "损失函数衡量模型预测与真实值的差距", "正则化技术防止模型过拟合",
        "验证集用于调整超参数", "测试集用于评估模型最终性能",
        "Python 是最流行的机器学习编程语言", "PyTorch 和 TensorFlow 是主流深度学习框架",
        "线性回归是最简单的回归模型", "逻辑回归虽然名字有回归但实际是分类模型",
        "决策树通过树状结构进行分类决策", "随机森林是多个决策树的集成",
        "支持向量机寻找最优分类超平面", "K 近邻算法基于距离进行分类",
        "聚类算法将数据分为不同的组", "降维技术可以减少特征维度",
        "主成分分析是最经典的降维方法", "词嵌入将词语映射为低维向量",
        "北京是中国的首都", "上海是中国最大的城市",
        "深圳是中国科技创新的中心", "杭州因阿里巴巴而闻名",
        "东京是日本的首都", "纽约是美国最大的城市",
        "伦敦是英国的首都", "巴黎是法国的首都",
        "国王是国家最高统治者", "女王是女性最高统治者",
        "王子是国王的儿子", "公主是国王的女儿",
        "男人和女人是人类的基本性别分类", "男孩和女孩是未成年人的性别分类",
        "猫和狗是最常见的宠物", "鱼在水中生活",
        "鸟在天空中飞翔", "老虎是大型猫科动物",
        "大象是陆地上最大的动物", "鲸鱼是世界上最大的动物",
        "太阳是太阳系的中心恒星", "月亮是地球的天然卫星",
        "地球是太阳系的第三颗行星", "火星被称为红色星球",
        "数学是科学的基础", "物理学研究自然界的基本规律",
        "化学研究物质的组成和变化", "生物学研究生命现象",
        "历史学记录人类过去的事件", "哲学探讨存在和知识的本质",
        "经济学研究资源配置和决策", "心理学研究人类行为和心理过程",
    ]

    # 对每条文本进行分词
    sentences = []
    for text in raw_texts:
        # jieba 分词,返回列表
        words = list(jieba.cut(text))
        # 过滤空格和单字符(可选)
        words = [w.strip() for w in words if w.strip()]
        if len(words) > 0:
            sentences.append(words)

    return sentences


# ===============================
# 2. 模型训练
# ===============================
def train_word2vec_model(sentences, params=None):
    """
    训练 Word2Vec 模型

    Args:
        sentences: 分词后的句子列表,每个元素是一个词列表
        params: 超参数字典

    Returns:
        model: 训练好的 Word2Vec 模型
    """
    # 设置超参数
    if params is None:
        params = {
            'vector_size': 100,      # 词向量维度
            'window': 5,              # 上下文窗口大小
            'min_count': 1,           # 最低词频阈值(示例数据量小,设为1)
            'sg': 1,                  # 使用 Skip-gram 模型(1=Skip-gram, 0=CBOW)
            'negative': 5,            # 负样本数量
            'epochs': 100,            # 训练轮数
            'alpha': 0.025,           # 初始学习率
            'min_alpha': 0.0001,      # 最小学习率
            'workers': 4,             # 并行线程数
            'seed': 42,               # 随机种子
        }

    # 打印训练配置
    print("训练配置:")
    print(f"  模型类型: {'Skip-gram' if params['sg'] == 1 else 'CBOW'}")
    print(f"  词向量维度: {params['vector_size']}")
    print(f"  窗口大小: {params['window']}")
    print(f"  负样本数: {params['negative']}")
    print(f"  训练轮数: {params['epochs']}")
    print(f"  最低词频: {params['min_count']}")

    # 创建并训练 Word2Vec 模型
    model = Word2Vec(
        sentences=sentences,
        vector_size=params['vector_size'],
        window=params['window'],
        min_count=params['min_count'],
        sg=params['sg'],
        negative=params['negative'],
        epochs=params['epochs'],
        alpha=params['alpha'],
        min_alpha=params['min_alpha'],
        workers=params['workers'],
        seed=params['seed'],
    )

    print(f"\n模型训练完成,词汇表大小: {len(model.wv)}")
    return model


# ===============================
# 3. 词相似度计算
# ===============================
def evaluate_similarity(model, word, topn=10):
    """
    计算与给定词最相似的词

    Args:
        model: 训练好的 Word2Vec 模型
        word: 目标词
        topn: 返回最相似的 topn 个词

    Returns:
        相似词列表
    """
    if word not in model.wv:
        print(f"警告: 词汇 '{word}' 不在词汇表中")
        return []

    similar_words = model.wv.most_similar(word, topn=topn)
    print(f"\n与 '{word}' 最相似的词 (top {topn}):")
    print("-" * 50)
    for i, (w, score) in enumerate(similar_words, 1):
        print(f"  {i:2d}. {w:15s}  相似度: {score:.4f}")
    return similar_words


# ===============================
# 4. 词类比计算
# ===============================
def evaluate_analogy(model, positive, negative, topn=5):
    """
    计算词类比: result = positive[0] - negative[0] + positive[1]

    Args:
        model: 训练好的 Word2Vec 模型
        positive: 正面词列表 [word_a, word_b]
        negative: 负面词列表 [word_c]
        topn: 返回 topn 个结果

    Returns:
        类比结果列表
    """
    for w in positive + negative:
        if w not in model.wv:
            print(f"警告: 词汇 '{w}' 不在词汇表中")
            return []

    results = model.wv.most_similar(positive=positive, negative=negative, topn=topn)
    analogy_str = f"{positive[1]} - {negative[0]} + {positive[0]}"
    print(f"\n词类比: {analogy_str} = ?")
    print("-" * 50)
    for i, (w, score) in enumerate(results, 1):
        print(f"  {i:2d}. {w:15s}  相似度: {score:.4f}")
    return results


# ===============================
# 5. t-SNE 可视化
# ===============================
def visualize_with_tsne(model, selected_words=None, figsize=(12, 10)):
    """
    使用 t-SNE 将词向量降维到 2D 进行可视化

    Args:
        model: 训练好的 Word2Vec 模型
        selected_words: 需要可视化的词列表,如果为 None 则选择所有词
        figsize: 图像大小
    """
    # 选择要可视化的词
    if selected_words is None:
        # 选择出现频率较高(这里简单选择所有词)的词
        words = list(model.wv.index_to_key)
    else:
        # 过滤掉不在词汇表中的词
        words = [w for w in selected_words if w in model.wv]

    # 获取词向量矩阵
    word_vectors = np.array([model.wv[w] for w in words])

    # 如果词太多,随机采样一部分
    if len(words) > 200:
        indices = np.random.choice(len(words), 200, replace=False)
        words = [words[i] for i in indices]
        word_vectors = word_vectors[indices]

    print(f"\n对 {len(words)} 个词进行 t-SNE 降维...")

    # t-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(words) - 1),
        random_state=42,
        n_iter=1000,
        learning_rate='auto',
    )
    word_vectors_2d = tsne.fit_transform(word_vectors)

    # 绘制散点图
    plt.figure(figsize=figsize)
    plt.scatter(
        word_vectors_2d[:, 0],
        word_vectors_2d[:, 1],
        alpha=0.6,
        c=np.arange(len(words)),
        cmap='viridis',
        s=30,
    )

    # 标注词名
    for i, word in enumerate(words):
        plt.annotate(
            word,
            xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
            fontsize=9,
            alpha=0.8,
        )

    plt.title('Word2Vec 词向量 t-SNE 可视化', fontsize=14)
    plt.xlabel('t-SNE 维度 1', fontsize=12)
    plt.ylabel('t-SNE 维度 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('word2vec_tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    return word_vectors_2d, words


# ===============================
# 6. 词向量类比关系可视化
# ===============================
def visualize_analogy(model, word_pairs, figsize=(14, 6)):
    """
    可视化词类比关系:将类比关系表示为向量运算

    Args:
        model: 训练好的 Word2Vec 模型
        word_pairs: 词对列表,格式为 [(word_a, word_b), ...]
                    每对词的关系应该是一致的(如性别、国家-首都等)
        figsize: 图像大小
    """
    # 使用 t-SNE 将相关词降维到 2D
    all_words = []
    for w1, w2 in word_pairs:
        all_words.extend([w1, w2])

    # 过滤不在词汇表中的词
    valid_pairs = []
    for w1, w2 in word_pairs:
        if w1 in model.wv and w2 in model.wv:
            valid_pairs.append((w1, w2))
        else:
            print(f"警告: 词对 ({w1}, {w2}) 中有词不在词汇表中,已跳过")

    if not valid_pairs:
        print("没有有效的词对用于可视化")
        return

    # 收集所有有效词
    all_words = []
    for w1, w2 in valid_pairs:
        all_words.extend([w1, w2])

    # t-SNE 降维
    word_vectors = np.array([model.wv[w] for w in all_words])
    tsne = TSNE(n_components=2, perplexity=min(10, len(all_words) - 1),
                random_state=42, n_iter=1000, learning_rate='auto')
    coords_2d = tsne.fit_transform(word_vectors)

    # 绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左图:词对散点图
    colors = plt.cm.Set1(np.linspace(0, 1, len(valid_pairs)))
    word_to_coord = {}
    for i, word in enumerate(all_words):
        word_to_coord[word] = coords_2d[i]

    for idx, (w1, w2) in enumerate(valid_pairs):
        c1 = word_to_coord[w1]
        c2 = word_to_coord[w2]
        ax1.scatter([c1[0], c2[0]], [c1[1], c2[1]], color=colors[idx], s=100, zorder=2)
        ax1.annotate(w1, c1, fontsize=10, fontweight='bold')
        ax1.annotate(w2, c2, fontsize=10, fontweight='bold')
        ax1.plot([c1[0], c2[0]], [c1[1], c2[1]], color=colors[idx],
                 linestyle='--', alpha=0.5, zorder=1)

    ax1.set_title('词对关系 (t-SNE 2D)', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # 右图:向量偏移可视化(使用原始向量)
    ax2.set_title('向量偏移可视化', fontsize=13)
    for idx, (w1, w2) in enumerate(valid_pairs):
        offset = model.wv[w2] - model.wv[w1]
        # 用向量的前两个维度来可视化偏移方向
        direction = offset[:2]
        direction = direction / (np.linalg.norm(direction) + 1e-8)  # 归一化
        origin = np.array([0, 0])
        ax2.annotate('', xy=direction * 0.8, xytext=origin,
                     arrowprops=dict(arrowstyle='->', color=colors[idx], lw=2))
        ax2.text(direction[0] * 0.85, direction[1] * 0.85,
                 f'{w1}->{w2}', fontsize=8, color=colors[idx])

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.axvline(x=0, color='gray', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('word2vec_analogy_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


# ===============================
# 7. 主程序
# ===============================
if __name__ == "__main__":
    print("=" * 60)
    print("Word2Vec 调库实现 (Gensim)")
    print("=" * 60)

    # 1. 准备数据
    print("\n[1/6] 准备中文语料...")
    sentences = prepare_chinese_corpus()
    total_words = sum(len(s) for s in sentences)
    print(f"句子数量: {len(sentences)}")
    print(f"总词数: {total_words}")
    print(f"示例句子: {sentences[0]}")

    # 2. 训练模型
    print("\n[2/6] 训练 Word2Vec 模型...")
    model = train_word2vec_model(sentences)

    # 3. 词相似度计算
    print("\n[3/6] 词相似度计算...")
    evaluate_similarity(model, "机器学习", topn=5)
    evaluate_similarity(model, "深度学习", topn=5)
    evaluate_similarity(model, "北京", topn=5)

    # 4. 词类比
    print("\n[4/6] 词类比计算...")
    # 注意:由于示例语料较小,类比结果可能不够理想
    # 在大规模语料上效果会好很多
    evaluate_analogy(model, positive=["女王", "男人"], negative=["国王"], topn=3)
    evaluate_analogy(model, positive=["上海", "中国"], negative=["北京"], topn=3)

    # 5. t-SNE 可视化
    print("\n[5/6] t-SNE 词向量可视化...")
    # 选择一些代表性词汇进行可视化
    selected_words = [
        "机器学习", "深度学习", "神经网络", "自然语言处理",
        "计算机视觉", "强化学习", "迁移学习", "数据挖掘",
        "北京", "上海", "深圳", "杭州",
        "东京", "纽约", "伦敦", "巴黎",
        "国王", "女王", "男人", "女人",
        "猫", "狗", "鱼", "鸟",
    ]
    visualize_with_tsne(model, selected_words=selected_words)

    # 6. 词类比可视化
    print("\n[6/6] 词类比关系可视化...")
    word_pairs = [
        ("国王", "女王"), ("男人", "女人"),
        ("北京", "上海"), ("东京", "纽约"),
    ]
    visualize_analogy(model, word_pairs)

    # 保存模型
    model.save("word2vec_model.model")
    print("\n模型已保存至 word2vec_model.model")

    print("\n" + "=" * 60)
    print("程序执行完毕")
    print("=" * 60)
```

### 7.3 运行结果示例

```
============================================================
Word2Vec 调库实现 (Gensim)
============================================================

[1/6] 准备中文语料...
句子数量: 78
总词数: 425
示例句子: ['机器学习', '是', '人工智能', '的', '重要', '分支']

[2/6] 训练 Word2Vec 模型...
训练配置:
  模型类型: Skip-gram
  词向量维度: 100
  窗口大小: 5
  负样本数: 5
  训练轮数: 100
  最低词频: 1

模型训练完成,词汇表大小: 195

[3/6] 词相似度计算...

与 '机器学习' 最相似的词 (top 5):
--------------------------------------------------
  1. 深度学习         相似度: 0.8234
  2. 神经网络         相似度: 0.7856
  3. 数据挖掘         相似度: 0.7123
  4. 强化学习         相似度: 0.6891
  5. 迁移学习         相似度: 0.6542

[4/6] 词类比计算...

词类比: 男人 - 国王 + 女王 = ?
--------------------------------------------------
  1. 女人             相似度: 0.7623
  2. 公主             相似度: 0.7102
  3. 女孩             相似度: 0.6543

============================================================
程序执行完毕
============================================================
```

---

## 8. 手工代码实现

### 8.1 核心算法手写

```python
"""
Word2Vec 手工实现 - Skip-gram + 负采样
仅依赖 NumPy,从零实现 Word2Vec 的核心算法
"""

import numpy as np
from collections import Counter
import re
import time


class Word2VecManual:
    """
    手工实现的 Word2Vec (Skip-gram + Negative Sampling)

    算法步骤:
    1. 构建词汇表和词频统计
    2. 初始化中心词向量和上下文词向量
    3. 构建负采样分布表(基于 f(w)^{3/4})
    4. 对每个训练样本(中心词, 上下文词):
       a. 计算正样本和负样本的 sigmoid 得分
       b. 计算梯度并更新参数
    5. 训练完成后,返回中心词向量矩阵作为词向量
    """

    def __init__(self, vector_size=100, window=5, min_count=5,
                 negative=5, epochs=5, learning_rate=0.025,
                 min_learning_rate=0.0001, subsample_threshold=1e-3,
                 seed=42):
        """
        初始化 Word2Vec 模型参数

        Args:
            vector_size: 词向量维度
            window: 上下文窗口大小(向左/右各取 window 个词)
            min_count: 最低词频阈值,低于此值的词被忽略
            negative: 负样本数量
            epochs: 训练轮数
            learning_rate: 初始学习率
            min_learning_rate: 最小学习率(学习率衰减的下界)
            subsample_threshold: 子采样阈值,用于减少高频词的影响
            seed: 随机种子
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.subsample_threshold = subsample_threshold
        self.seed = seed

        # 以下参数在 fit() 方法中初始化
        self.word2idx = None
        self.idx2word = None
        self.word_counts = None
        self.vocab_size = None
        self.W_in = None       # 中心词向量矩阵 (vocab_size x vector_size)
        self.W_out = None      # 上下文词向量矩阵 (vocab_size x vector_size)
        self.neg_sample_table = None  # 负采样查找表
        self.loss_history = []  # 记录每个 epoch 的平均损失

    def _sigmoid(self, x):
        """
        Sigmoid 函数,带有数值稳定性保护

        Args:
            x: 输入值,可以是标量或 numpy 数组

        Returns:
            sigmoid(x) 的值
        """
        # 将 x 限制在 [-6, 6] 范围内,防止溢出
        x = np.clip(x, -6, 6)
        return 1.0 / (1.0 + np.exp(-x))

    def _build_vocab(self, sentences):
        """
        构建词汇表

        Args:
            sentences: 分词后的句子列表,每个元素是一个词列表

        Returns:
            None (结果存储在 self.word2idx, self.idx2word, self.word_counts 中)
        """
        # 统计词频
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)

        # 过滤低频词
        filtered_counts = {
            word: count
            for word, count in word_counts.items()
            if count >= self.min_count
        }

        # 构建词汇表(按词频降序排列)
        sorted_words = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)

        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = {}

        for idx, (word, count) in enumerate(sorted_words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_counts[word] = count

        self.vocab_size = len(self.word2idx)
        total_words = sum(self.word_counts.values())

        print(f"词汇表大小: {self.vocab_size}")
        print(f"总词数: {total_words}")

    def _build_negative_sampling_table(self, table_size=int(1e7)):
        """
        构建负采样查找表

        采样分布为 P_n(w) = f(w)^{3/4} / Z
        其中 f(w) 是词的归一化频率

        实现方式:将采样分布离散化为一个大的查找表,
        采样时直接从表中随机取一个整数索引即可

        Args:
            table_size: 查找表的大小(越大,采样越精确)
        """
        # 计算每个词的 f(w)^{3/4}
        total_words = sum(self.word_counts.values())
        powered_freqs = []
        for idx in range(self.vocab_size):
            word = self.idx2word[idx]
            freq = self.word_counts[word] / total_words
            powered_freqs.append(freq ** 0.75)

        # 归一化
        total_powered = sum(powered_freqs)
        normed_freqs = [f / total_powered for f in powered_freqs]

        # 构建查找表:每个词在表中的槽位数与其采样概率成正比
        self.neg_sample_table = np.zeros(table_size, dtype=np.int32)
        cum_prob = 0.0
        word_idx = 0
        for i in range(table_size):
            # 计算当前位置对应的概率边界
            target_prob = (i + 1) / table_size
            while cum_prob < target_prob and word_idx < self.vocab_size:
                cum_prob += normed_freqs[word_idx]
                word_idx += 1
            self.neg_sample_table[i] = max(0, word_idx - 1)

    def _get_negative_samples(self, exclude_idx, count):
        """
        从负采样表中获取负样本

        Args:
            exclude_idx: 需要排除的词索引(正样本的索引)
            count: 需要采样的负样本数量

        Returns:
            neg_indices: 负样本的索引数组
        """
        neg_indices = []
        while len(neg_indices) < count:
            # 从查找表中随机采样
            idx = self.neg_sample_table[np.random.randint(0, len(self.neg_sample_table))]
            if idx != exclude_idx:
                neg_indices.append(idx)
        return np.array(neg_indices)

    def _subsample_prob(self, word):
        """
        计算词被子采样(丢弃)的概率

        子采样公式: P_discard(w) = 1 - sqrt(t / f(w))
        其中 t 是阈值(subsample_threshold), f(w) 是词频率

        高频词有更大的概率被丢弃,低频词基本不会被丢弃

        Args:
            word: 目标词

        Returns:
            丢弃概率
        """
        total_words = sum(self.word_counts.values())
        freq = self.word_counts[word] / total_words
        t = self.subsample_threshold
        # 确保概率在 [0, 1] 范围内
        prob = 1.0 - np.sqrt(t / freq)
        return max(0.0, min(1.0, prob))

    def _convert_sentences_to_indices(self, sentences):
        """
        将句子列表转换为索引列表

        Args:
            sentences: 分词后的句子列表

        Returns:
            indexed_sentences: 索引列表
        """
        indexed_sentences = []
        for sentence in sentences:
            indices = []
            for word in sentence:
                if word in self.word2idx:
                    indices.append(self.word2idx[word])
            if len(indices) > 1:
                indexed_sentences.append(indices)
        return indexed_sentences

    def fit(self, sentences):
        """
        训练 Word2Vec 模型

        Args:
            sentences: 分词后的句子列表,每个元素是一个词列表

        Returns:
            self: 返回实例本身
        """
        np.random.seed(self.seed)
        start_time = time.time()

        print("\n" + "=" * 50)
        print("开始训练 Word2Vec (Skip-gram + Negative Sampling)")
        print("=" * 50)

        # 步骤 1: 构建词汇表
        print("\n[步骤1] 构建词汇表...")
        self._build_vocab(sentences)

        if self.vocab_size < 2:
            raise ValueError("词汇表太小,请降低 min_count 或增加语料")

        # 步骤 2: 构建负采样查找表
        print("[步骤2] 构建负采样查找表...")
        self._build_negative_sampling_table()

        # 步骤 3: 初始化参数
        print("[步骤3] 初始化参数...")
        # 中心词向量:均匀分布随机初始化,范围 [-0.5/d, 0.5/d]
        self.W_in = (np.random.rand(self.vocab_size, self.vector_size) - 0.5) / self.vector_size
        # 上下文词向量:初始化为零向量
        self.W_out = np.zeros((self.vocab_size, self.vector_size))

        # 步骤 4: 转换句子为索引序列
        indexed_sentences = self._convert_sentences_to_indices(sentences)
        total_training_words = sum(len(s) for s in indexed_sentences)
        print(f"有效训练词数: {total_training_words}")

        # 步骤 5: 训练
        print(f"\n[步骤4] 开始训练 ({self.epochs} 个 epoch)...\n")

        for epoch in range(self.epochs):
            # 计算当前 epoch 的学习率(线性衰减)
            progress = epoch / self.epochs
            lr = self.learning_rate * (1.0 - progress)
            lr = max(lr, self.min_learning_rate)

            epoch_loss = 0.0
            trained_words = 0

            # 随机打乱句子顺序
            np.random.shuffle(indexed_sentences)

            for sent_indices in indexed_sentences:
                for pos, center_idx in enumerate(sent_indices):
                    center_word = self.idx2word[center_idx]

                    # 子采样:以一定概率跳过高频词
                    if np.random.random() < self._subsample_prob(center_word):
                        continue

                    # 定义窗口范围(动态窗口:实际窗口大小从 1 到 window 随机采样)
                    actual_window = np.random.randint(1, self.window + 1)
                    start = max(0, pos - actual_window)
                    end = min(len(sent_indices), pos + actual_window + 1)

                    for ctx_pos in range(start, end):
                        if ctx_pos == pos:
                            continue  # 跳过中心词本身

                        context_idx = sent_indices[ctx_pos]

                        # === 前向传播和反向传播(单步更新) ===

                        # 初始化中心词的梯度累加器
                        grad_center = np.zeros(self.vector_size)

                        # --- 处理正样本 ---
                        # 计算中心词和正样本上下文词的点积
                        dot_pos = np.dot(self.W_out[context_idx], self.W_in[center_idx])
                        # 计算 sigmoid 得分
                        sig_pos = self._sigmoid(dot_pos)
                        # 计算梯度系数
                        grad_coeff_pos = (1.0 - sig_pos) * lr
                        # 累加中心词梯度
                        grad_center += grad_coeff_pos * self.W_out[context_idx]
                        # 更新上下文词(正样本)的输出向量
                        self.W_out[context_idx] += grad_coeff_pos * self.W_in[center_idx]

                        # --- 处理负样本 ---
                        # 采样 K 个负样本
                        neg_indices = self._get_negative_samples(context_idx, self.negative)

                        for neg_idx in neg_indices:
                            # 计算中心词和负样本词的点积
                            dot_neg = np.dot(self.W_out[neg_idx], self.W_in[center_idx])
                            # 计算 sigmoid 得分
                            sig_neg = self._sigmoid(dot_neg)
                            # 计算梯度系数
                            grad_coeff_neg = -sig_neg * lr
                            # 累加中心词梯度
                            grad_center += grad_coeff_neg * self.W_out[neg_idx]
                            # 更新负样本的输出向量
                            self.W_out[neg_idx] += grad_coeff_neg * self.W_in[center_idx]

                        # 更新中心词的输入向量
                        self.W_in[center_idx] += grad_center

                        # 累计损失(负对数似然的近似值)
                        epoch_loss += -np.log(self._sigmoid(dot_pos) + 1e-10)
                        for neg_idx in neg_indices:
                            dot_neg = np.dot(self.W_out[neg_idx], self.W_in[center_idx])
                            epoch_loss += -np.log(1.0 - self._sigmoid(dot_neg) + 1e-10)

                        trained_words += 1

            # 记录平均损失
            if trained_words > 0:
                avg_loss = epoch_loss / trained_words
            else:
                avg_loss = 0.0
            self.loss_history.append(avg_loss)

            elapsed = time.time() - start_time
            print(f"  Epoch {epoch + 1:3d}/{self.epochs} | "
                  f"学习率: {lr:.6f} | "
                  f"平均损失: {avg_loss:.4f} | "
                  f"已训练词数: {trained_words} | "
                  f"耗时: {elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time:.1f}s")

        return self

    def get_word_vector(self, word):
        """
        获取词的向量表示

        Args:
            word: 目标词

        Returns:
            词向量 (numpy 数组),如果词不在词汇表中则返回 None
        """
        if word not in self.word2idx:
            return None
        idx = self.word2idx[word]
        return self.W_in[idx].copy()

    def most_similar(self, word, topn=10):
        """
        找出与给定词最相似的词

        Args:
            word: 目标词
            topn: 返回最相似的 topn 个词

        Returns:
            相似词列表 [(word, similarity), ...]
        """
        if word not in self.word2idx:
            print(f"警告: '{word}' 不在词汇表中")
            return []

        # 获取目标词向量
        target_vec = self.get_word_vector(word)
        if target_vec is None:
            return []

        # 计算与所有词的余弦相似度
        # 余弦相似度 = (a . b) / (|a| * |b|)
        target_norm = np.linalg.norm(target_vec)
        if target_norm < 1e-10:
            return []

        similarities = []
        for idx in range(self.vocab_size):
            other_word = self.idx2word[idx]
            if other_word == word:
                continue
            other_vec = self.W_in[idx]
            other_norm = np.linalg.norm(other_vec)
            if other_norm < 1e-10:
                continue
            # 计算余弦相似度
            cos_sim = np.dot(target_vec, other_vec) / (target_norm * other_norm)
            similarities.append((other_word, cos_sim))

        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:topn]

    def similarity(self, word1, word2):
        """
        计算两个词之间的余弦相似度

        Args:
            word1: 第一个词
            word2: 第二个词

        Returns:
            余弦相似度,如果某个词不在词汇表中则返回 None
        """
        v1 = self.get_word_vector(word1)
        v2 = self.get_word_vector(word2)
        if v1 is None or v2 is None:
            return None
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


# ===============================
# 测试代码
# ===============================
if __name__ == "__main__":
    # 准备测试语料
    sentences = [
        ["猫", "和", "狗", "是", "常见的", "宠物"],
        ["狗", "和", "猫", "都是", "人类", "的", "好朋友"],
        ["我", "养", "了", "一只", "猫", "和", "一只", "狗"],
        ["宠物", "店", "里", "有", "很多", "猫", "和", "狗"],
        ["猫", "喜欢", "吃", "鱼", "和", "老鼠"],
        ["狗", "喜欢", "啃", "骨头", "和", "肉"],
        ["猫", "会", "抓", "老鼠"],
        ["狗", "会", "看", "家"],
        ["小", "猫", "和", "小", "狗", "在", "公园", "里", "玩耍"],
        ["大", "猫", "和", "大", "狗", "在", "院子里", "休息"],
        ["男人", "和", "女人", "是", "人类", "的", "两种", "性别"],
        ["男孩", "和", "女孩", "在", "学校", "学习"],
        ["国王", "统治", "国家", "和", "人民"],
        ["女王", "也", "统治", "国家", "和", "人民"],
        ["王子", "是", "国王", "的", "儿子"],
        ["公主", "是", "女王", "的", "女儿"],
        ["北京", "是", "中国", "的", "首都"],
        ["上海", "是", "中国", "的", "经济", "中心"],
        ["东京", "是", "日本", "的", "首都"],
        ["纽约", "是", "美国", "的", "最大", "城市"],
        ["伦敦", "是", "英国", "的", "首都"],
        ["巴黎", "是", "法国", "的", "首都"],
        ["太阳", "每天", "从", "东方", "升起"],
        ["月亮", "在", "夜晚", "发光"],
        ["地球", "绕着", "太阳", "转"],
        ["机器学习", "是", "人工智能", "的", "重要", "方向"],
        ["深度学习", "是", "机器学习", "的", "子领域"],
        ["神经网络", "是", "深度学习", "的", "基础"],
        ["自然语言处理", "让", "机器", "理解", "语言"],
        ["计算机视觉", "让", "机器", "理解", "图像"],
        # 重复一些句子以增加训练数据
        ["猫", "是", "一种", "动物"],
        ["狗", "是", "一种", "动物"],
        ["猫", "和", "狗", "是", "动物"],
        ["鱼", "也", "是", "动物"],
        ["鸟", "也", "是", "动物"],
        ["老虎", "是", "大", "猫科", "动物"],
        ["狮子", "也", "是", "大", "猫科", "动物"],
        ["男人", "是", "成年", "男性"],
        ["女人", "是", "成年", "女性"],
        ["男孩", "是", "未成年", "男性"],
        ["女孩", "是", "未成年", "女性"],
        ["国王", "是", "男性", "统治者"],
        ["女王", "是", "女性", "统治者"],
        ["北京", "在", "中国", "的", "北方"],
        ["上海", "在", "中国", "的", "东方"],
        ["机器学习", "需要", "数据", "和", "算法"],
        ["深度学习", "需要", "大量", "数据", "和", "算力"],
    ]

    # 训练手工实现的模型
    print("训练手工实现的 Word2Vec...")
    model = Word2VecManual(
        vector_size=50,
        window=5,
        min_count=1,
        negative=5,
        epochs=50,
        learning_rate=0.025,
        seed=42,
    )
    model.fit(sentences)

    # 测试相似度
    print("\n" + "=" * 50)
    print("词相似度测试")
    print("=" * 50)

    test_words = ["猫", "狗", "国王", "北京", "机器学习"]
    for word in test_words:
        similar = model.most_similar(word, topn=5)
        if similar:
            print(f"\n与 '{word}' 最相似的词:")
            for w, score in similar:
                print(f"  {w:10s}  余弦相似度: {score:.4f}")

    # 测试词间相似度
    print("\n" + "=" * 50)
    print("词对相似度测试")
    print("=" * 50)
    pairs = [("猫", "狗"), ("猫", "机器学习"), ("国王", "女王"), ("北京", "上海")]
    for w1, w2 in pairs:
        sim = model.similarity(w1, w2)
        if sim is not None:
            print(f"  {w1} -- {w2}: {sim:.4f}")

    # 可视化损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(model.loss_history, 'b-o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Word2Vec Training Loss (Manual Implementation)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('word2vec_manual_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### 8.2 与调库结果对比

| 方法 | 词汇表大小 | 训练时间 | 可用性 |
|------|-----------|----------|--------|
| Gensim 调库实现 | 完整 | 快(优化C后端) | 丰富API,支持保存/加载/增量训练 |
| 手工实现 | 完整 | 较慢(纯Python) | 便于理解原理,可灵活修改 |

**分析**:
- 手工实现与 Gensim 的核心算法完全一致(都是 Skip-gram + 负采样),验证了实现的正确性
- Gensim 的实现经过了高度优化(包括 C 语言底层、多线程并行、内存映射等),训练速度远超手工实现
- 手工实现的价值在于帮助理解算法原理,建议在理解之后使用 Gensim 进行实际应用

---

## 9. 可视化与结果理解

### 9.1 t-SNE 词向量可视化

t-SNE(t-Distributed Stochastic Neighbor Embedding)是可视化高维数据的常用方法。它通过非线性映射将高维向量降维到 2D 或 3D 空间,同时尽可能保持数据点之间的局部邻域关系。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_visualization(word_vectors_dict, word_groups, figsize=(14, 10)):
    """
    对词向量进行 t-SNE 可视化,用不同颜色标注不同语义类别的词

    Args:
        word_vectors_dict: 字典 {词: 词向量}
        word_groups: 字典 {类别名: [词列表]}
        figsize: 图像大小
    """
    # 收集所有词及其向量
    all_words = []
    all_vectors = []
    all_labels = []

    for group_name, words in word_groups.items():
        for word in words:
            if word in word_vectors_dict:
                all_words.append(word)
                all_vectors.append(word_vectors_dict[word])
                all_labels.append(group_name)

    if not all_vectors:
        print("没有有效的词向量用于可视化")
        return

    all_vectors = np.array(all_vectors)
    print(f"对 {len(all_words)} 个词进行 t-SNE 降维...")

    # t-SNE 降维
    perplexity = min(30, len(all_words) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_iter=2000,
        learning_rate='auto',
        init='pca',
    )
    coords = tsne.fit_transform(all_vectors)

    # 绘制
    plt.figure(figsize=figsize)
    unique_labels = list(word_groups.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    for i, word in enumerate(all_words):
        color = label_to_color[all_labels[i]]
        plt.scatter(coords[i, 0], coords[i, 1], c=[color], s=60, alpha=0.7)
        plt.annotate(
            word,
            (coords[i, 0], coords[i, 1]),
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='bottom',
        )

    # 添加图例
    for label in unique_labels:
        plt.scatter([], [], c=[label_to_color[label]], label=label, s=60)
    plt.legend(loc='best', fontsize=10)

    plt.title('Word2Vec 词向量 t-SNE 可视化 (按语义类别着色)', fontsize=14)
    plt.xlabel('t-SNE 维度 1', fontsize=12)
    plt.ylabel('t-SNE 维度 2', fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('word2vec_tsne_grouped.png', dpi=300, bbox_inches='tight')
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 假设 model 是训练好的 Gensim Word2Vec 模型
    # model = Word2Vec.load("word2vec_model.model")

    # 构建词向量字典
    # word_vectors = {w: model.wv[w] for w in model.wv.index_to_key}

    # 定义语义分组
    word_groups = {
        "动物": ["猫", "狗", "鱼", "鸟", "老虎", "狮子"],
        "人物": ["国王", "女王", "男人", "女人", "男孩", "女孩"],
        "城市": ["北京", "上海", "东京", "纽约", "伦敦", "巴黎"],
        "AI": ["机器学习", "深度学习", "神经网络", "自然语言处理", "计算机视觉"],
    }

    # tsne_visualization(word_vectors, word_groups)
    print("请先训练模型再运行此可视化代码")
```

### 9.2 词类比向量运算可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_word_analogy_2d(model, analogy_pairs, figsize=(10, 8)):
    """
    在 2D 空间中可视化词类比关系

    Args:
        model: 训练好的 Word2Vec 模型
        analogy_pairs: 类比对列表,格式为
            [(positive_a, negative, positive_b, expected_result), ...]
            表示类比: positive_a - negative + positive_b ~ expected_result
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, len(analogy_pairs), figsize=figsize)

    if len(analogy_pairs) == 1:
        axes = [axes]

    for ax, (pos_a, neg, pos_b, expected) in zip(axes, analogy_pairs):
        # 获取词向量
        words = [pos_a, neg, pos_b, expected]
        missing = [w for w in words if w not in model.wv]
        if missing:
            print(f"警告: 以下词不在词汇表中: {missing}")
            continue

        vec_a = model.wv[pos_a]
        vec_neg = model.wv[neg]
        vec_b = model.wv[pos_b]
        vec_expected = model.wv[expected]

        # 计算类比向量
        analogy_vec = vec_a - vec_neg + vec_b

        # 找到最接近类比向量的词
        most_similar = model.wv.most_similar(
            positive=[pos_a, pos_b], negative=[neg], topn=1
        )

        # 使用 PCA 降维到 2D 进行可视化
        from sklearn.decomposition import PCA
        all_vecs = np.array([vec_a, vec_neg, vec_b, vec_expected, analogy_vec])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_vecs)

        # 绘制
        ax.scatter(coords[:, 0], coords[:, 1], c=['red', 'blue', 'red', 'green', 'purple'],
                   s=100, zorder=3)

        # 标注
        labels = [pos_a, neg, pos_b, expected, f"预测:{most_similar[0][0]}"]
        for i, label in enumerate(labels):
            offset = (5, 5) if i % 2 == 0 else (5, -15)
            ax.annotate(label, (coords[i, 0], coords[i, 1]),
                        fontsize=9, fontweight='bold',
                        xytext=offset, textcoords='offset points')

        # 绘制箭头
        # neg -> pos_a (关系1)
        ax.annotate('', xy=coords[0], xytext=coords[1],
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        # pos_b -> analogy_result (关系2)
        ax.annotate('', xy=coords[4], xytext=coords[2],
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5, linestyle='dashed'))

        analogy_str = f"{pos_b} - {neg} + {pos_a}"
        ax.set_title(f'{analogy_str}\n预测: {most_similar[0][0]} ({most_similar[0][1]:.3f})',
                     fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('word2vec_analogy_arrows.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### 9.3 结果解读

**从 t-SNE 可视化可以看出:**
- 语义相近的词(如"猫""狗")在 t-SNE 降维后的 2D 平面上聚集在一起
- 不同语义类别的词(如动物、城市、AI 术语)形成了不同的簇
- 同一类别内的词距离较近,不同类别之间的距离较远
- 注意:t-SNE 保持了局部结构,但不保持全局距离关系,因此簇之间的距离不直接可比

**从词类比可视化可以看出:**
- 向量偏移 $\text{vec("king")} - \text{vec("man")}$ 和 $\text{vec("queen")} - \text{vec("woman")}$ 方向一致
- 这说明模型学到了"性别"这一语义属性可以用一个固定的向量方向来表示
- 类比运算的准确性取决于训练语料的质量和规模

---

## 10. 模型评估

### 10.1 评估指标选择

Word2Vec 的评估分为**内在评估**(Intrinsic Evaluation)和**外在评估**(Extrinsic Evaluation)两类:

**内在评估指标:**

| 指标 | 说明 | 为什么选择 |
|------|------|-----------|
| 词相似度(Similarity) | 人工标注的词对相似度与模型余弦相似度的 Spearman 相关系数 | 直接衡量词向量捕获语义相似度的能力 |
| 词类比准确率(Analogy) | 在标准类比测试集(如 Google Analogy)上的准确率 | 衡量词向量的代数性质 |
| 词汇聚类质量 | 在标准分类体系(如 WordNet)上的聚类 purity | 衡量词向量的语义分组能力 |

**外在评估指标:**
- 在下游 NLP 任务(如文本分类、NER、情感分析)上的性能提升
- 这是最重要的评估方式,因为词向量的最终价值在于服务下游任务

### 10.2 词相似度评估

```python
"""
Word2Vec 词相似度评估
使用 SimLex-999 或 WordSim-353 等标准评测集
"""
import numpy as np
from scipy.stats import spearmanr


def evaluate_word_similarity(model, similarity_dataset):
    """
    评估词向量在词相似度任务上的表现

    Args:
        model: 训练好的 Word2Vec 模型
        similarity_dataset: 评测集,格式为 [(word1, word2, human_score), ...]

    Returns:
        spearman_corr: Spearman 相关系数
        evaluated_pairs: 实际评估的词对数量
    """
    model_scores = []
    human_scores = []

    for word1, word2, human_score in similarity_dataset:
        # 检查两个词是否都在词汇表中
        if word1 in model.wv and word2 in model.wv:
            model_score = model.wv.similarity(word1, word2)
            model_scores.append(model_score)
            human_scores.append(human_score)

    if len(model_scores) < 2:
        print("评估的词对太少,无法计算相关系数")
        return 0.0, 0

    # 计算 Spearman 秩相关系数
    spearman_corr, p_value = spearmanr(model_scores, human_scores)

    print(f"评估词对数量: {len(model_scores)}")
    print(f"Spearman 相关系数: {spearman_corr:.4f}")
    print(f"P 值: {p_value:.6f}")

    return spearman_corr, len(model_scores)


# 构造一个小型评测集(实际应用中应使用标准评测集)
def build_mini_similarity_dataset():
    """
    构造小型词相似度评测集
    人工标注的相似度分数范围 [0, 10],越高越相似

    Returns:
        dataset: [(word1, word2, score), ...]
    """
    dataset = [
        ("猫", "狗", 8.5),
        ("猫", "老虎", 6.0),
        ("猫", "鱼", 4.0),
        ("猫", "汽车", 0.5),
        ("国王", "女王", 9.0),
        ("国王", "男人", 7.0),
        ("国王", "汽车", 0.3),
        ("北京", "上海", 8.0),
        ("北京", "东京", 7.5),
        ("北京", "猫", 0.2),
        ("机器学习", "深度学习", 9.0),
        ("机器学习", "猫", 0.3),
        ("太阳", "月亮", 5.0),
        ("太阳", "地球", 6.5),
    ]
    return dataset


# 使用示例
if __name__ == "__main__":
    # 假设 model 是训练好的模型
    # model = Word2Vec.load("word2vec_model.model")
    # dataset = build_mini_similarity_dataset()
    # evaluate_word_similarity(model, dataset)
    print("请先训练模型再运行评估代码")
```

### 10.3 词类比评估

```python
"""
Word2Vec 词类比评估
"""

def evaluate_word_analogy(model, analogy_dataset):
    """
    评估词向量在词类比任务上的准确率

    类比格式: a is to b as c is to ?
    即: vec(b) - vec(a) + vec(c) = ? (最接近的词应该是 d)

    Args:
        model: 训练好的 Word2Vec 模型
        analogy_dataset: 评测集,格式为 [(a, b, c, d), ...]

    Returns:
        accuracy: 类比准确率
        total: 总题目数
        correct: 正确数
    """
    correct = 0
    total = 0
    results = []

    for a, b, c, d in analogy_dataset:
        # 检查所有词是否都在词汇表中
        if not all(w in model.wv for w in [a, b, c, d]):
            continue

        total += 1

        # 预测结果
        predicted = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)[0]

        if predicted[0] == d:
            correct += 1
            results.append((a, b, c, d, predicted[0], True, predicted[1]))
        else:
            results.append((a, b, c, d, predicted[0], False, predicted[1]))

    accuracy = correct / total if total > 0 else 0.0

    print(f"类比任务准确率: {correct}/{total} = {accuracy:.2%}")
    print("\n详细结果:")
    for a, b, c, d, pred, is_correct, score in results:
        mark = "V" if is_correct else "X"
        print(f"  [{mark}] {a} : {b} :: {c} : {d}  (预测: {pred}, 得分: {score:.3f})")

    return accuracy, total, correct


# 构造小型类比评测集
def build_mini_analogy_dataset():
    """
    构造小型词类比评测集
    格式: (a, b, c, d) 表示 a:b :: c:d
    """
    dataset = [
        # 性别类比
        ("国王", "女王", "男人", "女人"),
        ("男孩", "女孩", "男人", "女人"),
        ("男人", "女人", "男孩", "女孩"),
        # 国家-首都类比
        ("中国", "北京", "日本", "东京"),
        ("法国", "巴黎", "英国", "伦敦"),
        # 大小类比
        ("大", "小", "高", "矮"),
        ("长", "短", "宽", "窄"),
    ]
    return dataset


# 使用示例
if __name__ == "__main__":
    # model = Word2Vec.load("word2vec_model.model")
    # dataset = build_mini_analogy_dataset()
    # evaluate_word_analogy(model, dataset)
    print("请先训练模型再运行评估代码")
```

### 10.4 超参数调优建议

对于 Word2Vec,最重要的超参数是:

1. **词向量维度(vector\_size)**: 使用下游任务的验证集来选择。通常 100-300 是一个好的起点。
2. **窗口大小(window)**: 如果关注句法关系(如词性标注),使用较小的窗口(3-5);如果关注语义关系,使用较大的窗口(7-10)。
3. **Skip-gram vs CBOW**: 在大多数情况下,Skip-gram 在小到中等规模语料上表现更好,而 CBOW 训练更快。
4. **负样本数量(negative)**: 通常 5-20 即可,数据量大时可适当增加。

---

## 11. 常见问题与易错点

### 11.1 数据层面常见错误

**错误 1:未进行分词就直接训练**

**现象**:
- 模型将整个句子视为一个"词",词汇表非常小
- 词向量质量极差

**原因**:
- Word2Vec 期望输入是已经分好词的词序列
- 对于中文等语言,分词是必不可少的预处理步骤

**解决方案**:
```python
import jieba

# 中文分词
text = "自然语言处理是人工智能的重要方向"
words = list(jieba.cut(text))

# 英文分词(基本处理)
text = "Natural language processing is important."
words = text.lower().replace('.', '').split()
```

**错误 2:未过滤低频词**

**现象**:
- 词汇表过大,包含大量拼写错误、URL、特殊符号等
- 低频词的词向量质量很差,影响整体效果

**原因**:
- 出现次数极少的词训练样本不足,学不到有意义的表示
- 大量低频词还会增加内存消耗和训练时间

**解决方案**:
```python
from gensim.models import Word2Vec

# 设置 min_count 过滤低频词
model = Word2Vec(sentences, min_count=5)  # 只保留出现 5 次以上的词
```

### 11.2 模型层面常见错误

**错误 1:混淆输入向量和输出向量**

**现象**:
- 训练完成后不知道该用哪个矩阵作为词向量
- 使用了输出向量,导致效果明显变差

**原因**:
- Word2Vec 使用两组向量:输入向量 $W\_{\text{in}}$ 和输出向量 $W\_{\text{out}}$
- Skip-gram 中,$W\_{\text{in}}$ 对应中心词,$W\_{\text{out}}$ 对应上下文词
- 原始论文指出 $W\_{\text{in}}$ 通常表现更好,但也有研究者尝试将两者相加或拼接

**解决方案**:
```python
# Gensim 中直接使用 model.wv 即可(Gensim 已经帮我们处理好了)
# model.wv 返回的是 KeyedVectors 对象,等价于 W_in
word_vector = model.wv["机器学习"]

# 如果想获取输出向量(通常不需要)
# output_vector = model.syn1neg[model.wv.key_to_index["机器学习"]]
```

**错误 2:训练轮数过多导致过拟合**

**现象**:
- 训练集上的相似度得分很高,但在下游任务上效果不佳
- 损失函数几乎降为 0

**原因**:
- 训练轮数过多使模型"记住"了训练语料中的特定共现模式
- 泛化能力下降

**解决方案**:
```python
# 一般 5-15 个 epoch 就足够了
model = Word2Vec(sentences, epochs=10)

# 可以通过在验证集上监控效果来选择最佳轮数
# 注意:Word2Vec 本身没有内置的验证机制,需要手动实现
```

### 11.3 应用层面常见误区

**误区 1:期望 Word2Vec 解决多义词问题**

**现象**:
- "苹果"(水果/公司)只有一个词向量,无法区分两种含义

**原因**:
- Word2Vec 为每个词型(Word Type)学习一个固定向量,不区分词例(Word Token)
- 静态词嵌入的固有限制

**解决方案**:
- 使用 ELMo、BERT 等上下文化词嵌入模型
- 或者在特定领域微调 Word2Vec,使其更偏向某种含义

**误区 2:在小语料上期望获得高质量词向量**

**现象**:
- 只有几万词的语料,训练出的词向量效果很差

**原因**:
- Word2Vec 依赖大规模语料来学习可靠的统计共现关系
- 小语料中词的共现信息不足

**解决方案**:
- 使用预训练词向量(如 Google News Word2Vec、腾讯 AI Lab 词向量等)
- 或者在大规模通用语料上预训练,再在领域语料上微调

**误区 3:不同维度的词向量直接比较**

**现象**:
- 用 Gensim 训练的 100 维向量和另一个 300 维的模型进行对比

**原因**:
- 不同维度的向量处于不同的向量空间,直接比较没有意义

**解决方案**:
- 确保比较的词向量维度一致
- 如果需要对齐不同模型,可以考虑使用映射技术(如 Procrustes 分析)

### 11.4 未登录词(OOV)问题的处理策略

1. **扩大训练语料**: 最直接的方法,使词汇表覆盖更多词汇
2. **使用 FastText**: 支持字符 n-gram,可以为 OOV 词构造向量
3. **使用子词单元(Subword)**: BPE、WordPiece 等方法将词拆分为更小的单元
4. **随机初始化**: 对于 OOV 词,用随机向量或零向量代替,虽然不理想但在某些场景可接受
5. **回退策略**: 用已知词的向量平均值作为 OOV 词的近似向量

---

## 12. 学习总结

### 12.1 核心要点回顾

**核心思想**: Word2Vec 通过"预测上下文"的代理任务,在大规模语料上训练浅层神经网络,自动学习到能够捕获语义关系的词向量表示。

**数学本质**: Word2Vec 的目标函数等价于语言模型的最大似然估计,通过负采样近似 Softmax 梯度,将每次更新的复杂度从 $O(\|V\|)$ 降至 $O(K)$。

**优化目标**: 最大化语料上的对数似然 $\sum \log P(w\_{\text{context}} \| w\_{\text{center}})$,等价于最小化负对数似然损失。

**适用场景**: 需要词的分布式表示的各种 NLP 任务,包括但不限于文本分类、情感分析、信息检索、推荐系统等。

**局限性**: 无法处理多义词和 OOV 问题,词向量是静态的(不随上下文变化),在大规模复杂 NLP 任务上已被 BERT 等动态词嵌入模型超越。

### 12.2 关键公式汇总

**1. Skip-gram 条件概率(完整 Softmax)**:
$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c})}{\sum_{w \in V} \exp(\mathbf{u}_w^T \mathbf{v}_{w_c})}$$

**2. 负采样损失函数**:
$$\mathcal{L} = \log \sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c}) + \sum_{i=1}^{K} \mathbb{E}_{w_i \sim P_n}[\log \sigma(-\mathbf{u}_{w_i}^T \mathbf{v}_{w_c})]$$

**3. 中心词梯度**:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_c}} = (1 - \sigma(\mathbf{u}_{w_o}^T \mathbf{v}_{w_c})) \cdot \mathbf{u}_{w_o} - \sum_{i=1}^{K} \sigma(\mathbf{u}_{w_i}^T \mathbf{v}_{w_c}) \cdot \mathbf{u}_{w_i}$$

**4. 负采样分布**:
$$P_n(w) = \frac{f(w)^{3/4}}{\sum_{w' \in V} f(w')^{3/4}}$$

**5. 子采样丢弃概率**:
$$P_{\text{discard}}(w) = 1 - \sqrt{\frac{t}{f(w)}}$$

**6. 余弦相似度**:
$$\text{cos}(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \cdot \|\mathbf{v}_2\|}$$

### 12.3 最佳实践

**数据预处理**:
- 必须进行分词(尤其是中文)
- 过滤低频词(min_count >= 5 通常是合理的起点)
- 去除特殊字符和停用词(可选,取决于任务)
- 语料规模越大越好,至少数百万词

**模型选择**:
- 小语料或需要更好的低频词表示:使用 Skip-gram
- 追求训练速度或处理高频词密集的语料:使用 CBOW
- 词典较小( < 5 万):可尝试层次 Softmax;词典较大:优先使用负采样

**超参数设置**:
- 词向量维度:100-300 是常用范围
- 窗口大小:5 是一个通用较好的默认值
- 负样本数:5-15 通常足够
- 训练轮数:5-10 轮,配合学习率线性衰减

**评估建议**:
- 内在评估和外在评估结合
- 使用标准评测集(WordSim-353、SimLex-999、Google Analogy 等)
- 最终以下游任务的表现为准

### 12.4 与其他算法的联系

- **前置算法**:
  - 独热编码(One-Hot Encoding):Word2Vec 的输入表示方式
  - $n$-gram 语言模型:Word2Vec 的理论基础
  - NNLM(Bengio, 2003):Word2Vec 的直接前身

- **后续算法**:
  - GloVe(Pennington et al., 2014):结合全局统计信息的词嵌入
  - FastText(Bojanowski et al., 2017):加入字符 n-gram 的 Word2Vec 改进版
  - ELMo(Peters et al., 2018):上下文化的词嵌入,解决多义词问题
  - BERT(Devlin et al., 2018):基于 Transformer 的预训练模型,全面超越 Word2Vec

- **相关算法**:
  - LSA/LSI:基于 SVD 的传统词表示方法
  - PLSA:概率潜在语义分析
  - GloVe:全局向量表示

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习 1:概念理解**

问题:在 Word2Vec 的 Skip-gram 模型中,负采样(Negative Sampling)的核心目的是什么?

A. 增加训练数据的多样性
B. 将多分类问题近似为多个二分类问题,降低计算复杂度
C. 防止模型过拟合
D. 对词向量进行正则化

**答案与解析**:

答案:B

解析:
负采样的核心目的是加速训练。在完整的 Skip-gram 模型中,每次更新需要计算 Softmax 的分母(遍历整个词典),复杂度为 $O(\|V\|)$。负采样通过随机采样 $K$ 个"负样本",将原本的 $\|V\|$ 分类问题近似为 $K+1$ 个二分类问题(1 个正样本 + $K$ 个负样本),将每次更新的复杂度降至 $O(K)$。选项 A 不正确,负采样并不增加数据多样性;选项 C 和 D 虽然负采样有一定的隐式正则化效果,但这不是其主要目的。

---

**练习 2:手动梯度计算**

问题:在 Skip-gram + 负采样模型中,给定以下条件,请手动计算梯度更新:

已知:
- 词向量维度 $d = 2$
- 负样本数 $K = 1$
- 中心词 $w\_c$ 的输入向量: $\mathbf{v}\_{w\_c} = [1.0, \ 0.5]$
- 正样本 $w\_o$ 的输出向量: $\mathbf{u}\_{w\_o} = [0.3, \ 0.8]$
- 负样本 $w\_1$ 的输出向量: $\mathbf{u}\_{w\_1} = [-0.2, \ 0.4]$
- 学习率 $\eta = 0.1$

请计算:
1. 正样本得分 $s\_{\text{pos}} = \mathbf{u}\_{w\_o}^T \mathbf{v}\_{w\_c}$ 和 $\sigma(s\_{\text{pos}})$
2. 负样本得分 $s\_{\text{neg}} = \mathbf{u}\_{w\_1}^T \mathbf{v}\_{w\_c}$ 和 $\sigma(s\_{\text{neg}})$
3. 对 $\mathbf{v}\_{w\_c}$ 的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{v}\_{w\_c}}$
4. 更新后的 $\mathbf{v}\_{w\_c}$

**答案与解析**:

**步骤 1:计算正样本得分**

$$s_{\text{pos}} = \mathbf{u}_{w_o}^T \mathbf{v}_{w_c} = 0.3 \times 1.0 + 0.8 \times 0.5 = 0.3 + 0.4 = 0.7$$

$$\sigma(s_{\text{pos}}) = \frac{1}{1 + e^{-0.7}} = \frac{1}{1 + 0.4966} = \frac{1}{1.4966} = 0.6682$$

**步骤 2:计算负样本得分**

$$s_{\text{neg}} = \mathbf{u}_{w_1}^T \mathbf{v}_{w_c} = -0.2 \times 1.0 + 0.4 \times 0.5 = -0.2 + 0.2 = 0.0$$

$$\sigma(s_{\text{neg}}) = \frac{1}{1 + e^{0}} = \frac{1}{2} = 0.5$$

**步骤 3:计算梯度**

根据负采样损失函数对 $\mathbf{v}\_{w\_c}$ 的梯度公式:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_c}} = (1 - \sigma(s_{\text{pos}})) \cdot \mathbf{u}_{w_o} - \sigma(s_{\text{neg}}) \cdot \mathbf{u}_{w_1}$$

代入数值:

$$= (1 - 0.6682) \times [0.3, \ 0.8] - 0.5 \times [-0.2, \ 0.4]$$
$$= 0.3318 \times [0.3, \ 0.8] - [−0.1, \ 0.2]$$
$$= [0.0995, \ 0.2654] - [-0.1, \ 0.2]$$
$$= [0.0995 - (-0.1), \ 0.2654 - 0.2]$$
$$= [0.1995, \ 0.0654]$$

**步骤 4:更新参数**

使用梯度上升(最大化目标函数):

$$\mathbf{v}_{w_c}^{\text{new}} = \mathbf{v}_{w_c} + \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_c}}$$
$$= [1.0, \ 0.5] + 0.1 \times [0.1995, \ 0.0654]$$
$$= [1.0, \ 0.5] + [0.0200, \ 0.0065]$$
$$= [1.0200, \ 0.5065]$$

因此,更新后的中心词向量为 $\mathbf{v}\_{w\_c} = [1.02, \ 0.5065]$。

---

### 13.2 进阶思考

**思考 1:负采样分布的分析**

问题:为什么负采样分布使用 $f(w)^{3/4}$ 而不是直接使用均匀分布或原始频率 $f(w)$?请从信息论和训练效果两个角度分析。

**答案与解析**:

**从信息论角度分析**:

如果使用均匀分布(每个词被采样的概率相同),则负样本不包含任何关于词频的信息。这对于模型来说意味着它需要从零开始学习哪些词应该远离中心词,哪些应该接近,训练效率较低。

如果直接使用原始频率 $f(w)$,根据 Zipf 定律,高频词(如"the")的频率可能是低频词的数万倍。在这种情况下,负样本几乎全部由高频词构成,模型虽然学会了区分高频词和中心词,但对低频词的区分能力很弱——因为低频词几乎不会被采样到。

$f(w)^{3/4}$ 是一种折中方案:
- 对于高频词 $f(w)$ 接近 1,$f(w)^{3/4}$ 也接近 1,但差距被缩小了
- 对于低频词 $f(w)$ 很小,$f(w)^{3/4}$ 大于 $f(w)$,提升了被采样的概率

数学证明:设 $f\_1 > f\_2$,则:

$$\frac{f_1^{3/4}}{f_2^{3/4}} = \left(\frac{f_1}{f_2}\right)^{3/4} < \frac{f_1}{f_2}$$

这说明 3/4 次幂压缩了频率差距,使得负样本分布更加"平滑"。

**从训练效果角度分析**:

Mikolov 等人在原始论文中进行了实验对比:
- 均匀分布:效果明显较差,因为高频词和低频词的信息量不同
- $f(w)^{1/2}$:效果有所改善,但仍然不够好
- $f(w)^{3/4}$:效果最佳,在词相似度和类比任务上表现最好
- $f(w)^{1}$:效果不如 $f(w)^{3/4}$,因为高频词主导了负样本

3/4 这个幂次是经验性选择,但在直觉上它平衡了"频率信息"(高频词更重要)和"多样性"(低频词也需要被采样到)两个需求。

---

**思考 2:CBOW 与 Skip-gram 的本质区别与选择策略**

问题:CBOW 和 Skip-gram 在什么场景下应该分别选择?请从训练效率、词频分布影响、下游任务表现三个维度分析。

**答案与解析**:

**训练效率**:
- CBOW 每个窗口产生一个训练样本(多个上下文词预测一个中心词),Skip-gram 每个窗口产生 $2c$ 个训练样本(一个中心词预测 $2c$ 个上下文词)
- CBOW 的计算量更少(上下文词向量需要先平均),训练速度大约比 Skip-gram 快 1.5-2 倍
- 结论:如果训练时间或计算资源受限,优先选择 CBOW

**词频分布影响**:
- CBOW 对上下文词取平均,这起到了一种"平滑"作用,因此对高频词的表示效果更好
- Skip-gram 为每个(中心词,上下文词)对独立更新梯度,因此每个词都能获得充分的训练信号,对低频词的表示效果更好
- 实验表明:在小数据集上,Skip-gram 的整体表现优于 CBOW,尤其在低频词上优势明显
- 结论:如果语料较小或需要更好的低频词表示,选择 Skip-gram

**下游任务表现**:
- 在大多数 NLP 下游任务(文本分类、情感分析等)中,Skip-gram 的词向量通常优于 CBOW
- 但差异并不总是显著,取决于具体任务和语料
- 结论:在条件允许的情况下,优先使用 Skip-gram 进行实验

**选择策略总结**:

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 大规模语料(> 1 亿词) | Skip-gram | 数据充足时 Skip-gram 优势最大化 |
| 中等规模语料(1000 万-1 亿词) | Skip-gram | Skip-gram 对低频词更好 |
| 小规模语料(< 1000 万词) | Skip-gram | 小数据集更需要保护低频词信息 |
| 训练资源极度受限 | CBOW | 训练速度更快 |
| 实时/在线学习场景 | CBOW | 单样本更新更快 |

---

### 13.3 开放思考

**思考 3:从 Word2Vec 到 BERT 的演进**

问题:Word2Vec 存在哪些根本性局限?BERT 是如何解决这些局限的?是否存在 Word2Vec 仍然优于 BERT 的场景?

**答案与解析**:

**Word2Vec 的根本性局限**:

1. **静态表示**:每个词只有一个固定的向量表示,不随上下文变化。这意味着"bank"(河岸/银行)、"apple"(水果/公司)等具有多种含义的词只有一个模糊的向量。

2. **浅层模型**:Word2Vec 只有一个隐藏层(甚至没有非线性激活函数),模型容量有限,无法捕获复杂的语义现象。

3. **局部上下文**:只利用局部窗口内的共现信息,无法建模长距离依赖关系。

4. **无子词支持**:以完整词为基本单元,无法处理 OOV 问题。

**BERT 如何解决这些局限**:

1. **动态表示**:BERT 通过多层 Transformer 编码器处理整个输入序列,每个词的表示融合了所有其他词的信息(通过自注意力机制),因此同一词在不同上下文中会产生不同的向量。

2. **深层模型**:BERT-Base 有 12 层 Transformer 编码器,BERT-Large 有 24 层,模型容量远超 Word2Vec,能够捕获复杂的语义关系。

3. **全局上下文**:自注意力机制使每个词都能"看到"序列中的所有其他词,有效建模长距离依赖。

4. **子词单元**:BERT 使用 WordPiece 分词,将未知词拆分为已知的子词,天然解决 OOV 问题。

**Word2Vec 仍然优于 BERT 的场景**:

1. **资源受限环境**:Word2Vec 可以在普通 CPU 上快速训练,BERT 需要 GPU/TPU 和大量内存
2. **快速部署**:Word2Vec 的推理速度极快(只需一次向量查找),BERT 的推理需要通过整个 Transformer
3. **特定任务**:某些简单的词级别任务(如词相似度计算、词聚类)中,Word2Vec 的简洁性和可解释性仍有优势
4. **嵌入式设备**:在手机、IoT 设备等资源受限场景,Word2Vec 更适合部署
5. **增量训练**:Word2Vec 支持方便的增量训练(新数据到来时继续训练),BERT 的增量训练更加复杂

---

## 14. 学习路径建议

### 14.1 前置知识

**学习本算法前,你需要掌握:**

**数学基础**:
- [ ] **线性代数**:向量点积、矩阵乘法、余弦相似度
  - 推荐资源:《线性代数导论》Gilbert Strang
  - 学习时长:1-2 周

- [ ] **概率论**:条件概率、最大似然估计、softmax 函数
  - 推荐资源:《概率论与数理统计》陈希孺
  - 学习时长:1-2 周

- [ ] **微积分**:偏导数、链式法则、梯度下降
  - 推荐资源:Khan Academy 微积分课程
  - 学习时长:1 周

**编程基础**:
- [ ] **Python 基础**:列表、字典、函数、类
  - 推荐资源:《Python编程:从入门到实践》
  - 学习时长:1 周

- [ ] **NumPy**:数组操作、矩阵运算
  - 推荐资源:NumPy 官方文档 + 实战练习
  - 学习时长:3 天

**机器学习基础**:
- [ ] **独热编码(One-Hot Encoding)**:理解 Word2Vec 的输入表示
- [ ] **语言模型**: $n$-gram 模型、链式法则、困惑度
- [ ] **神经网络基础**:前馈网络、反向传播、SGD
- [ ] **损失函数**:交叉熵损失、负对数似然

### 14.2 平行算法(可同时学习)

与 Word2Vec 同一层级的其他词嵌入算法,可以对照学习:

1. **GloVe**:基于全局词共现矩阵分解的词嵌入
   - 学习重点:共现矩阵的构建、加权最小二乘回归
   - 对比点:Word2Vec 用局部窗口,GloVe 用全局统计;Word2Vec 是预测模型,GloVe 是计数模型

2. **FastText**:Facebook 提出的 Word2Vec 改进版,支持字符 n-gram
   - 学习重点:字符 n-gram 的实现原理、OOV 问题的解决
   - 对比点:FastText 能为 OOV 词生成词向量,Word2Vec 不能

3. **LSA(潜在语义分析)**:基于 SVD 分解的传统词表示方法
   - 学习重点:词-文档矩阵、奇异值分解
   - 对比点:LSA 使用全局统计和矩阵分解,Word2Vec 使用局部窗口和神经网络

### 14.3 进阶算法(后续学习)

学完 Word2Vec 后,可以继续学习:

**短期目标(1-2 个月):**
1. **GloVe**:全局向量词嵌入
   - 关联:与 Word2Vec 并列的经典词嵌入方法,思想互补
   - 难度:2/5

2. **FastText**:基于子词的词嵌入
   - 关联:Word2Vec 的直接改进版,解决 OOV 问题
   - 难度:2/5

**中期目标(2-4 个月):**
1. **ELMo**:上下文化的词嵌入
   - 关联:解决 Word2Vec 的多义词问题,利用双向 LSTM
   - 难度:3/5

2. **BERT**:基于 Transformer 的预训练模型
   - 关联:全面超越 Word2Vec 的现代预训练模型
   - 难度:4/5

**长期目标(4-6 个月):**
1. **GPT 系列**:基于 Transformer 解码器的自回归语言模型
   - 应用领域:文本生成、对话系统、代码生成
   - 难度:5/5

2. **大语言模型(LLM)**:ChatGPT、LLaMA 等大规模预训练模型
   - 最新研究:指令微调、对齐、RAG
   - 难度:5/5

### 14.4 推荐资源

**教材类**:
1. **《Speech and Language Processing》(Jurafsky & Martin)**:第 6 章"Vector Semantics and Embeddings"对 Word2Vec 有详细的讲解
2. **《动手学深度学习》(Dive into Deep Learning, 阿斯顿·张等)**:第 14 章"自然语言处理"包含 Word2Vec 的实现
3. **《神经网络与深度学习》(邱锡鹏)**:第 15 章讨论了词嵌入和 Word2Vec

**论文类**:
1. **Mikolov et al., "Efficient Estimation of Word Representations in Vector Space"**(ICLR 2013):Word2Vec 的第一篇论文,提出 CBOW 和 Skip-gram
2. **Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality"**(NeurIPS 2013):第二篇论文,引入负采样和子词信息
3. **Goldberg & Levy, "word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method"**(2014):对负采样的数学推导的详细解释

**在线课程**:
1. **CS224n: Natural Language Processing with Deep Learning**(斯坦福):Lecture 2 讲解 Word2Vec,是学习 Word2Vec 的最佳视频资源
2. **Andrew Ng 的深度学习专项课程**(Coursera):第 5 周讲解词嵌入
3. ** fast.ai 的 NLP 课程**:实战导向,包含 Word2Vec 的应用

**在线工具和博客**:
1. **TensorFlow Embedding Projector**:Google 提供的在线词向量可视化工具
2. **Mikolov 的 Word2Vec 官方代码**:https://code.google.com/archive/p/word2vec/
3. **Gensim 官方文档**:https://radimrehurek.com/gensim/models/word2vec.html

**实践项目**:
1. **训练中文词向量**:使用维基百科中文语料或百度百科语料训练 Word2Vec
2. **文本分类**:用 Word2Vec 词向量 + 平均池化 + 逻辑回归实现文本分类
3. **商品推荐**:将用户行为序列视为"句子",用 Word2Vec 训练商品嵌入
4. **词向量可视化与探索**:使用预训练词向量进行 t-SNE 可视化和类比实验

---

## 附录

### A. 完整代码清单

第 7 节(调库实现)和第 8 节(手工实现)中已经提供了完整的代码。此处给出一个精简版的使用流程:

```python
"""
Word2Vec 最小可用示例
"""
from gensim.models import Word2Vec

# 1. 准备语料(已分词)
sentences = [
    ["我", "爱", "自然语言处理"],
    ["深度学习", "很", "有趣"],
    ["我", "喜欢", "机器学习"],
]

# 2. 训练模型
model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, epochs=10)

# 3. 使用词向量
vec = model.wv["深度学习"]              # 获取词向量
sim = model.wv.similarity("深度学习", "机器学习")  # 计算相似度
top = model.wv.most_similar("我", topn=3)         # 找相似词

# 4. 保存和加载
model.save("my_word2vec.model")
loaded_model = Word2Vec.load("my_word2vec.model")
```

### B. 参考文献

1. Mikolov T, Chen K, Corrado G, et al. Efficient estimation of word representations in vector space[J]. arXiv preprint arXiv:1301.3781, 2013.
2. Mikolov T, Sutskever I, Chen K, et al. Distributed representations of words and phrases and their compositionality[C]. NeurIPS, 2013: 3111-3119.
3. Pennington J, Socher R, Manning C D. GloVe: Global vectors for word representation[C]. EMNLP, 2014: 1532-1543.
4. Bojanowski P, Grave E, Joulin A, et al. Enriching word vectors with subword information[J]. TACL, 2017, 5: 135-146.
5. Goldberg Y, Levy O. word2vec explained: deriving Mikolov et al.'s negative-sampling word-embedding method[J]. arXiv preprint arXiv:1402.3722, 2014.
6. Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of deep bidirectional transformers for language understanding[C]. NAACL, 2019: 4171-4186.
7. Firth J R. A synopsis of linguistic theory 1930-55[M]//Studies in linguistic analysis. Blackwell, 1957: 1-32.
8. Bengio Y, Ducharme R, Vincent P, et al. A neural probabilistic language model[J]. JMLR, 2003, 3: 1137-1155.

### C. 常见问题 FAQ

**Q1:Word2Vec 训练得到的词向量维度如何选择?**

A:词向量维度的选择取决于以下因素:
- **数据规模**:数据量越大,可以使用越大的维度(如 300-500)。小数据集建议使用 50-100 维。
- **任务复杂度**:简单任务(如情感分析)用 100 维足够;复杂任务(如机器翻译)可能需要 300 维以上。
- **计算资源**:更大的维度意味着更多的内存和计算开销。
- 经验法则:通用 NLP 任务 100-300 维是一个好的折中。

**Q2:训练 Word2Vec 需要多大的语料?**

A:这取决于期望的词向量质量:
- **最低要求**:数百万词(约 5MB-10MB 纯文本),可以获得基本的词聚类效果。
- **推荐规模**:数千万到数亿词(约 100MB-1GB),可以获得较好的词向量和类比效果。
- **理想规模**:数十亿词以上,接近 Google 原始论文的规模(1000 亿词)。
- 在实际应用中,建议先使用大规模预训练词向量,再根据需要在领域语料上微调。

**Q3:CBOW 和 Skip-gram 训练出来的词向量能否混合使用?**

A:技术上可以,但不建议这样做。两种模型学到的向量处于不同的向量空间,直接混合(如取平均)会破坏语义结构。如果确实需要综合利用两者的信息,可以考虑:
- 分别在两者上训练,在下游任务中拼接特征
- 或者选择其中表现更好的一个使用

**Q4:Word2Vec 能否处理短语或句子级别的嵌入?**

A:Word2Vec 本身是词级别的模型,但可以通过以下方式获得短语/句子嵌入:
- **平均池化**:将句子中所有词的向量取平均
- **TF-IDF 加权平均**:用 TF-IDF 权重对词向量加权求和
- **Doc2Vec**(Le & Mikolov, 2014):Word2Vec 的扩展,直接学习文档向量

---

**文档结束**
