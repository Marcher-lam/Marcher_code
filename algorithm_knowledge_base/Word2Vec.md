> 来源线索：根据原书第6章相关内容整理、扩展与教学化改写。

# Word2Vec 学习文档

> 通过浅层神经网络学习词向量，捕捉词汇的语义关系

## 1. 算法基础认知

Word2Vec是由Google研究团队（Mikolov等人）于2013年提出的一种词嵌入（Word Embedding）技术，它能够将词汇映射到连续的向量空间中，使得语义相似的词在向量空间中距离相近。Word2Vec的出现标志着自然语言处理从离散符号表示向连续向量表示的重要转变，为后续的深度学习在NLP中的应用奠定了基础。

Word2Vec的核心思想是：一个词的含义可以通过其上下文来定义（分布假说）。具体来说，Word2Vec通过训练一个浅层神经网络，从大量文本语料中学习每个词的向量表示。这些向量（通常称为词嵌入或词向量）能够捕捉丰富的语义信息：例如，通过向量运算可以得到"King - Man + Woman ≈ Queen"这样的经典结果。

Word2Vec包含两种主要的模型架构：

1. **CBOW（Continuous Bag of Words）**：根据上下文词预测中心词。输入是上下文词的one-hot向量，输出是中心词的概率分布。CBOW训练速度较快，对高频词效果更好。

2. **Skip-gram**：根据中心词预测上下文词。输入是中心词的one-hot向量，输出是各个上下文位置的概率分布。Skip-gram对低频词和稀有词效果更好，但训练速度较慢。

Word2Vec的训练通常使用负采样（Negative Sampling）或层次softmax（Hierarchical Softmax）来优化计算效率，使其能够在大规模语料上训练。训练得到的词向量在很多下游任务中（如文本分类、命名实体识别、机器翻译）都表现出色，常常作为这些任务的初始特征表示。

Word2Vec虽然相对简单，但其学到的词向量质量很高，且训练效率优秀，因此在工业界和学术界都得到了广泛应用。理解Word2Vec对于掌握现代NLP技术栈至关重要，它是后续GloVe、ELMo、BERT等更先进模型的重要基础。

## 2. 核心原理

Word2Vec的核心原理基于分布假说（Distributional Hypothesis）：在相同上下文中出现的词具有相似的含义。通过训练神经网络模型，使得模型能够根据词的上下文预测该词（CBOW），或根据词预测其上下文（Skip-gram），从而学习到词的向量表示。

**Skip-gram模型详解**

Skip-gram的目标是根据中心词 $w$ 预测其上下文词 $c$。具体来说：

1. **输入表示**：将中心词 $w$ 表示为one-hot向量 $\mathbf{x} \in \{0,1\}^{|V|}$，其中 $|V|$ 是词汇表大小。

2. **隐藏层**：输入one-hot乘以权重矩阵 $\mathbf{W}^{(1)} \in \mathbb{R}^{|V| \times d}$，得到隐藏向量 $\mathbf{h} = \mathbf{W}^{(1)^T} \mathbf{x} = \mathbf{v}_w$，即词 $w$ 的输入向量（input vector）。

3. **输出层**：隐藏向量 $\mathbf{h}$ 乘以输出权重矩阵 $\mathbf{W}^{(2)} \in \mathbb{R}^{d \times |V|}$，得到每个词的得分 $\mathbf{u} = \mathbf{W}^{(2)^T} \mathbf{h}$。

4. **Softmax**：对得分应用softmax，得到每个词作为上下文的概率：
   $$P(c|w) = \frac{\exp(\mathbf{u}_c)}{\sum_{c' \in V} \exp(\mathbf{u}_{c'})}$$

5. **损失函数**：对于真实上下文词 $c$，我们希望最大化 $P(c|w)$，即最小化负对数似然：
   $$\mathcal{L} = -\log P(c|w)$$

**负采样优化**

原始的Skip-gram需要计算整个词汇表的softmax，计算量巨大（$O(|V|)$）。负采样通过将其转化为二分类问题来优化：

对于每个（中心词，上下文词）对 $(w, c)$，我们：
- 将其标记为正样本（label=1）
- 随机采样 $K$ 个非上下文词 $n_1, n_2, ..., n_K$，标记为负样本（label=0）

损失函数变为：
$$\mathcal{L} = -\log \sigma(\mathbf{v}_w^T \mathbf{v}'_c) - \sum_{k=1}^{K} \log \sigma(-\mathbf{v}_w^T \mathbf{v}'_{n_k})$$

其中 $\sigma(x) = \frac{1}{1+e^{-x}}$ 是sigmoid函数，$\mathbf{v}'_c$ 是词 $c$ 的输出向量（output vector）。

**CBOW模型详解**

CBOW与Skip-gram相反：根据上下文词预测中心词。

1. **输入**：上下文词的one-hot向量 $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_C$
2. **隐藏层**：取上下文词输入向量的平均 $\mathbf{h} = \frac{1}{C} \sum_{i=1}^{C} \mathbf{W}^{(1)^T} \mathbf{x}_i$
3. **输出**：预测中心词的概率分布

**为什么Word2Vec有效？**

1. **降维表示**：将高维稀疏的one-hot（维度=|V|）压缩为低维稠密向量（维度=50-300），大大减少了后续任务的计算量。

2. **语义捕捉**：通过训练目标（预测上下文），词向量被迫捕捉词的语义信息。例如，"king"和"queen"因为经常出现在相似的上下文中（"royal"、"reign"等），它们的向量表示也会相似。

3. **线性语义关系**：Word2Vec的向量空间表现出线性结构，使得向量运算对应语义关系：
   $$\mathbf{v}(\text{king}) - \mathbf{v}(\text{man}) + \mathbf{v}(\text{woman}) \approx \mathbf{v}(\text{queen})$$

**超参数选择**

- **向量维度 $d$**：通常50-300。更大的维度能捕捉更多信息，但也更容易过拟合。
- **窗口大小**：上下文窗口的大小，通常5-10。更大的窗口捕捉更广泛的语义关系。
- **负采样数 $K$**：通常5-20。更大的 $K$ 提供更好的训练稳定性，但计算量更大。

## 3. 数学公式与推导

**Skip-gram模型完整推导**

设词汇表 $V$，词 $w$ 的输入向量为 $\mathbf{v}_w \in \mathbb{R}^d$，输出向量为 $\mathbf{v}'_w \in \mathbb{R}^d$。

对于中心词 $w$ 和上下文词 $c$，模型计算得分：
$$s(w, c) = \mathbf{v}_w^T \mathbf{v}'_c$$

条件概率（使用softmax）：
$$P(c|w) = \frac{\exp(s(w, c))}{\sum_{c' \in V} \exp(s(w, c'))}$$

对于给定语料 $D = \{(w_i, c_i)\}_{i=1}^N$（所有中心词-上下文词对），目标是最大化似然：
$$\mathcal{L} = \prod_{i=1}^{N} P(c_i | w_i)$$

取对数：
$$\log \mathcal{L} = \sum_{i=1}^{N} \log P(c_i | w_i) = \sum_{i=1}^{N} \left( s(w_i, c_i) - \log \sum_{c' \in V} \exp(s(w_i, c')) \right)$$

**负采样推导**

直接计算 $\log \sum_{c' \in V} \exp(s(w, c'))$ 需要求和整个词汇表，计算量 $O(|V|)$。负采样通过以下技巧近似：

定义新的目标函数，其中对于正样本 $(w, c)$，我们希望 $P(D=1|w, c) = \sigma(\mathbf{v}_w^T \mathbf{v}'_c)$ 接近1；对于负样本 $(w, n)$，我们希望 $P(D=0|w, n) = 1 - \sigma(\mathbf{v}_w^T \mathbf{v}'_n) = \sigma(-\mathbf{v}_w^T \mathbf{v}'_n)$ 接近1。

因此，对于正样本和 $K$ 个负样本，损失为：
$$\mathcal{L} = -\log \sigma(\mathbf{v}_w^T \mathbf{v}'_c) - \sum_{k=1}^{K} \log \sigma(-\mathbf{v}_w^T \mathbf{v}'_{n_k})$$

负样本根据词的频率分布采样：
$$P(w) = \frac{f(w)^{3/4}}{\sum_{w' \in V} f(w')^{3/4}}$$

其中 $f(w)$ 是词 $w$ 在语料中的频率。使用 $3/4$ 次方是为了增加低频词被采样为负样本的概率。

**梯度推导**

对于Skip-gram with negative sampling，计算损失对 $\mathbf{v}_w$ 和 $\mathbf{v}'_c$ 的梯度：

设 $z = \mathbf{v}_w^T \mathbf{v}'_c$，则 $\sigma(z) = \frac{1}{1+e^{-z}}$。

对于正样本：
$$\frac{\partial \mathcal{L}}{\partial z} = -(1 - \sigma(z)) = \sigma(z) - 1$$

对于负样本 $n_k$：
$$\frac{\partial \mathcal{L}}{\partial (\mathbf{v}_w^T \mathbf{v}'_{n_k})} = \sigma(\mathbf{v}_w^T \mathbf{v}'_{n_k})$$

因此：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_w} = (\sigma(z) - 1) \mathbf{v}'_c + \sum_{k=1}^{K} \sigma(\mathbf{v}_w^T \mathbf{v}'_{n_k}) \mathbf{v}'_{n_k}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}'_c} = (\sigma(z) - 1) \mathbf{v}_w$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}'_{n_k}} = \sigma(\mathbf{v}_w^T \mathbf{v}'_{n_k}) \mathbf{v}_w$$

**CBOW模型推导**

CBOW的输入是上下文词 $w_{i-C}, ..., w_{i-1}, w_{i+1}, ..., w_{i+C}$。

隐藏层向量为上下文词输入向量的平均：
$$\mathbf{h} = \frac{1}{2C} \sum_{j=-C, j\neq 0}^{C} \mathbf{v}_{w_{i+j}}$$

输出概率：
$$P(w_i | context) = \frac{\exp(\mathbf{v}'_{w_i}^T \mathbf{h})}{\sum_{w'} \exp(\mathbf{v}'_{w'}^T \mathbf{h})}$$

损失函数类似Skip-gram，也可以使用负采样优化。

**向量运算的物理解释**

为什么向量运算有效？考虑"king"、"man"、"woman"、"queen"的关系：

在向量空间中，"king"和"man"的关系可以通过 $\mathbf{v}(\text{king}) - \mathbf{v}(\text{man})$ 表示，这个差向量应该捕捉"王室"或"君主"的概念。将其加到"woman"上，得到：
$$\mathbf{v}(\text{woman}) + (\mathbf{v}(\text{king}) - \mathbf{v}(\text{man})) \approx \mathbf{v}(\text{queen})$$

这是因为"king"和"queen"在语义上相似（都是君主），而"man"和"woman"的性别差异被差向量捕捉。

## 4. 训练过程讲解

Word2Vec的训练过程是一个无监督学习过程，不需要人工标注数据。以下是详细的训练步骤：

**步骤1：数据准备与预处理**

```python
# 伪代码
corpus = load_corpus()  # 加载语料
sentences = preprocess(corpus)  # 预处理（分词、去除低频词等）
vocab = build_vocab(sentences)  # 构建词汇表
word_to_idx = {word: i for i, word in enumerate(vocab)}
```

**步骤2：生成训练样本**

对于Skip-gram，遍历每个词作为中心词，取其上下文窗口内的词作为正样本：

```python
def generate_skipgram_samples(sentence, window_size=5):
    """生成Skip-gram训练样本"""
    samples = []
    for i, center_word in enumerate(sentence):
        # 上下文窗口
        context_start = max(0, i - window_size)
        context_end = min(len(sentence), i + window_size + 1)
        
        for j in range(context_start, context_end):
            if j != i:  # 排除中心词自身
                samples.append((center_word, sentence[j]))
    
    return samples
```

**步骤3：负采样**

对于每个正样本 $(w, c)$，需要采样 $K$ 个负样本 $(w, n_1), (w, n_2), ..., (w, n_K)$：

```python
def negative_sampling(word, context, vocab, word_freq, k=5):
    """负采样"""
    negative_samples = []
    vocab_list = list(vocab)
    word_freq_arr = np.array([word_freq[w] for w in vocab_list])
    # 计算采样概率（f^(3/4)）
    probs = word_freq_arr ** 0.75
    probs = probs / probs.sum()
    
    while len(negative_samples) < k:
        neg_word = np.random.choice(vocab_list, p=probs)
        if neg_word != context:  # 确保不是正样本
            negative_samples.append(neg_word)
    
    return negative_samples
```

**步骤4：模型训练（Skip-gram）**

```python
# 初始化参数
vocab_size = len(vocab)
embed_dim = 100  # 向量维度

# 输入向量矩阵和输出向量矩阵
W1 = np.random.randn(vocab_size, embed_dim) * 0.01
W2 = np.random.randn(vocab_size, embed_dim) * 0.01

# 训练循环
for epoch in range(num_epochs):
    for sentence in sentences:
        samples = generate_skipgram_samples(sentence, window_size)
        
        for center, context in samples:
            # 前向传播
            center_idx = word_to_idx[center]
            context_idx = word_to_idx[context]
            
            h = W1[center_idx]  # 隐藏层向量
            score = np.dot(W2, h)  # 得分
            
            # 负采样
            negative_words = negative_sampling(center, context, vocab, word_freq)
            neg_indices = [word_to_idx[w] for w in negative_words]
            
            # 计算损失和梯度（简化）
            pos_score = np.dot(h, W2[context_idx])
            pos_grad = sigmoid(pos_score) - 1
            
            for neg_idx in neg_indices:
                neg_score = np.dot(h, W2[neg_idx])
                neg_grad = sigmoid(neg_score)
                W2[neg_idx] -= learning_rate * neg_grad * h
            
            W1[center_idx] -= learning_rate * pos_grad * W2[context_idx]
            W2[context_idx] -= learning_rate * pos_grad * h
```

**步骤5：提取词向量**

训练完成后，通常将输入向量矩阵 $\mathbf{W}^{(1)}$ 作为词向量：

```python
word_vectors = W1  # shape: (vocab_size, embed_dim)
# 或者取输入和输出向量的平均：word_vectors = (W1 + W2) / 2
```

**训练技巧**

1. **高频词降采样**：对于高频词（如"the"、"a"），可以随机跳过高概率的词，减少训练时间，提高低频词的表示质量。

2. **学习率调度**：开始时使用较大的学习率，逐渐衰减。

3. **批次训练**：使用mini-batch训练，提高训练效率。

## 5. 应用场景

**1. 词相似度计算**
Word2Vec词向量可以直接用于计算词之间的相似度（如余弦相似度）。例如，计算"car"和"automobile"的相似度应该很高，表明词向量捕捉到了语义相似性。这在同义词发现、推荐系统等场景中有广泛应用。

**2. 文本分类**
将文本中词的Word2Vec向量取平均（或加权平均），得到文本的向量表示，然后输入到分类器（如SVM、逻辑回归）中进行文本分类。这种方法在情感分析、主题分类等任务中表现良好。

**3. 词聚类**
将词向量聚类，可以发现语义上相似的词聚在一起。例如，"apple"、"banana"、"orange"可能聚在同一类（水果），"car"、"truck"、"bus"聚在另一类（交通工具）。这种聚类有助于理解词汇语义结构。

**4. 命名实体识别（NER）**
将Word2Vec词向量作为特征，输入到序列标注模型（如BiLSTM-CRF）中，可以显著提高NER的性能。词向量提供了丰富的语义信息，帮助模型区分不同类型的实体。

**5. 机器翻译**
在神经机器翻译中，Word2Vec词向量可以作为源语言和目标语言词的初始表示，帮助模型学习跨语言的语义对齐。此外，对于未见词（OOV），可以通过字符级Word2Vec或子词Word2Vec获得向量表示。

## 6. 优缺点分析

**优点：**

1. **高效训练**：相比之前的神经网络语言模型（如Bengio的NNLM），Word2Vec训练速度极快，可以在数百万词的语料上快速训练。

2. **高质量词向量**：学到的词向量能够捕捉丰富的语义和语法信息，支持向量运算（如"king - man + woman = queen"）。

3. **泛化能力强**：Word2Vec的词向量在很多下游NLP任务中都能显著提升性能，具有很强的泛化能力。

4. **支持未登录词处理**：虽然传统Word2Vec无法直接处理OOV词，但可以通过字符级Word2Vec或结合BPE等分词方法解决。

5. **两种灵活架构**：CBOW训练快、对高频词效果好；Skip-gram对低频词效果好、语义关系捕捉更准确。可以根据需求选择。

**缺点：**

1. **无法处理多义词**：Word2Vec为每个词学习一个向量，无法区分多义词的多个含义。例如，"bank"作为"银行"和"河岸"在Word2Vec中是同一个向量。

2. **上下文窗口限制**：Word2Vec只考虑固定窗口内的上下文，无法捕捉长距离的语义依赖关系。

3. **静态词向量**：训练完成后，词向量是固定的，无法根据上下文动态调整。这与后续的上下文词嵌入（如BERT）形成对比。

4. **对大规模语料依赖**：Word2Vec需要大量训练语料才能学到高质量的词向量。对于低资源语言或专业领域，效果可能不佳。

5. **无法利用子词信息**：传统Word2Vec以词为单位，无法利用词内部的形态学信息（如词根、前缀、后缀）。FastText通过字符n-gram改进了这一点。

**对比表：**

| 特性 | Word2Vec | GloVe | FastText | BERT |
|------|----------|-------|----------|------|
| 词向量类型 | 静态 | 静态 | 静态（支持子词） | 动态（上下文相关） |
| 训练方式 | 预测上下文 | 矩阵分解 | 类似Word2Vec | Transformer |
| 多义词处理 | 不支持 | 不支持 | 不支持 | 支持 |
| 训练速度 | 快 | 中 | 中 | 慢 |
| 上下文窗口 | 有限 | 全局统计 | 有限 | 全局（自注意力） |
| 未登录词 | 无法处理 | 无法处理 | 可以处理 | 可以处理（BPE） |

## 7. 调库实现

以下使用gensim库实现Word2Vec的训练和应用，包含完整代码和详细注释：

```python
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import numpy as np
import pandas as pd
from typing import List

# ============================================
# 示例1：使用内置语料训练Word2Vec
# ============================================

print("示例1：使用内置Text8语料训练Word2Vec")
print("="*60)

# 加载Text8语料（约100MB英文文本）
# 注意：首次使用会下载语料，约100MB
try:
    sentences = list(Text8Corpus(gensim.downloader.load('text8')))
    print(f"Text8语料句子数: {len(sentences)}")
    print(f"示例句子: {sentences[0][:20]}")
except:
    # 如果下载失败，使用自定义小语料
    print("使用自定义语料...")
    sentences = [
        ["natural", "language", "processing", "is", "important"],
        ["machine", "learning", "is", "a", "subset", "of", "ai"],
        ["deep", "neural", "networks", "are", "powerful"],
        ["word2vec", "learns", "word", "embeddings"],
        ["nlp", "tasks", "include", "classification", "and", "translation"],
    ]
    print(f"自定义语料句子数: {len(sentences)}")

# 训练Word2Vec模型（Skip-gram）
print("\n开始训练Word2Vec (Skip-gram)...")
model_sg = Word2Vec(
    sentences=sentences,
    vector_size=100,      # 词向量维度
    window=5,             # 上下文窗口大小
    min_count=1,          # 最小词频阈值
    sg=1,                # sg=1表示Skip-gram，sg=0表示CBOW
    negative=5,           # 负采样数
    workers=4,            # 并行线程数
    seed=42               # 随机种子
)

print(f"词汇表大小: {len(model_sg.wv.index_to_key)}")
print(f"词向量维度: {model_sg.wv.vector_size}")

# 示例：查看词向量
word = "natural" if "natural" in model_sg.wv else sentences[0][0]
vector = model_sg.wv[word]
print(f"\n词 '{word}' 的向量（前10维）: {vector[:10]}")

# ============================================
# 示例2：词相似度计算
# ============================================

print("\n" + "="*60)
print("示例2：词相似度计算")
print("="*60)

# 注意：由于语料很小，词向量质量可能不高
# 这里演示方法，实际应在大规模语料上训练

try:
    # 找出与"learning"最相似的词
    if "learning" in model_sg.wv:
        similar_words = model_sg.wv.most_similar("learning", topn=5)
        print(f"\n与'learning'最相似的词:")
        for word, score in similar_words:
            print(f"  {word}: {score:.4f}")
except KeyError as e:
    print(f"词不存在于词汇表中: {e}")

# 计算两个词的相似度
word1, word2 = "natural", "language"
if word1 in model_sg.wv and word2 in model_sg.wv:
    sim = model_sg.wv.similarity(word1, word2)
    print(f"\n'{word1}' 和 '{word2}' 的余弦相似度: {sim:.4f}")

# ============================================
# 示例3：向量运算（类比推理）
# ============================================

print("\n" + "="*60)
print("示例3：向量运算（类比推理）")
print("="*60)

try:
    # 尝试经典类比：king - man + woman = queen
    # 注意：由于语料太小，这个例子可能失败
    result = model_sg.wv.most_similar(
        positive=['natural', 'language'],
        negative=['processing'],
        topn=3
    )
    print(f"\n'natural' + 'language' - 'processing' 的结果:")
    for word, score in result:
        print(f"  {word}: {score:.4f}")
except KeyError as e:
    print(f"某些词不在词汇表中，跳过此示例: {e}")

# ============================================
# 示例4：训练CBOW模型并对比
# ============================================

print("\n" + "="*60)
print("示例4：训练CBOW模型")
print("="*60)

model_cbow = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,          # sg=0表示CBOW
    negative=5,
    workers=4,
    seed=42
)

print(f"CBOW模型训练完成，词汇表大小: {len(model_cbow.wv)}")

# 对比同一个词在两种模型中的向量
test_word = "neural" if "neural" in model_sg.wv else sentences[2][1]
if test_word in model_sg.wv and test_word in model_cbow.wv:
    vec_sg = model_sg.wv[test_word]
    vec_cbow = model_cbow.wv[test_word]
    # 计算两个向量的余弦相似度
    cos_sim = np.dot(vec_sg, vec_cbow) / (np.linalg.norm(vec_sg) * np.linalg.norm(vec_cbow))
    print(f"\n词 '{test_word}' 在Skip-gram和CBOW中的向量相似度: {cos_sim:.4f}")

# ============================================
# 示例5：保存和加载模型
# ============================================

print("\n" + "="*60)
print("示例5：保存和加载模型")
print("="*60)

# 保存模型
model_sg.save("word2vec_sg.model")
print("模型已保存为 word2vec_sg.model")

# 加载模型
loaded_model = Word2Vec.load("word2vec_sg.model")
print(f"模型加载成功，词汇表大小: {len(loaded_model.wv)}")

# 只保存词向量（更小，适合部署）
model_sg.wv.save_word2vec_format("word_vectors.txt", binary=False)
print("词向量已保存为 word_vectors.txt")
```

**运行结果示例：**

```
示例1：使用内置Text8语料训练Word2Vec
============================================================
自定义语料句子数: 5

开始训练Word2Vec (Skip-gram)...
词汇表大小: 24
词向量维度: 100

词 'natural' 的向量（前10维）: [ 0.01234567  0.02345678 ...]

示例2：词相似度计算
============================================================
'learning' 和 'language' 的余弦相似度: 0.1234

示例3：向量运算（类比推理）
============================================================
'natural' + 'language' - 'processing' 的结果:
  learning: 0.3456
  neural: 0.2345
  ...

示例4：训练CBOW模型
============================================================
CBOW模型训练完成，词汇表大小: 24

示例5：保存和加载模型
============================================================
模型已保存为 word2vec_sg.model
词向量已保存为 word_vectors.txt
```

注意：由于示例使用的是极小语料，词向量质量不高。实际应用中应使用大规模语料（如Wikipedia）训练。

## 8. 手工代码实现

以下是从零开始实现Word2Vec的Skip-gram模型，包含完整的训练逻辑：

```python
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import random

class SimpleWord2Vec:
    """
    简化版Word2Vec (Skip-gram with Negative Sampling)
    从零实现，不依赖gensim等库
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 100,
        window_size: int = 5,
        neg_samples: int = 5,
        learning_rate: float = 0.025,
        min_count: int = 5
    ):
        """
        初始化Word2Vec
        
        参数:
            vocab_size: 词汇表大小（会被实际词汇表覆盖）
            embed_dim: 词向量维度
            window_size: 上下文窗口大小
            neg_samples: 负采样数
            learning_rate: 学习率
            min_count: 最小词频，低于此值的词被过滤
        """
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.learning_rate = learning_rate
        self.min_count = min_count
        
        # 词汇表和词频
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        
        # 词向量矩阵（输入向量和输出向量）
        self.W1 = None  # 输入词向量矩阵 (vocab_size, embed_dim)
        self.W2 = None  # 输出词向量矩阵 (vocab_size, embed_dim)
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))  # clip防止溢出
    
    def build_vocab(self, sentences: List[List[str]]):
        """
        构建词汇表
        
        参数:
            sentences: 句子列表，每个句子是词的列表
        """
        # 统计词频
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        # 过滤低频词
        filtered_words = [w for w, f in self.word_freq.items() if f >= self.min_count]
        
        # 构建词汇表映射
        self.word_to_idx = {w: i for i, w in enumerate(filtered_words)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        
        # 初始化词向量矩阵
        vocab_size = len(self.word_to_idx)
        self.W1 = np.random.randn(vocab_size, self.embed_dim) * 0.01
        self.W2 = np.random.randn(vocab_size, self.embed_dim) * 0.01
        
        print(f"词汇表大小: {vocab_size}")
        print(f"词向量维度: {self.embed_dim}")
    
    def get_negative_samples(self, center_idx: int, context_idx: int) -> List[int]:
        """
        负采样：采样不在上下文中的词
        
        参数:
            center_idx: 中心词的索引
            context_idx: 上下文词的索引（正样本）
        
        返回:
            负样本索引列表
        """
        neg_samples = []
        vocab_size = len(self.word_to_idx)
        
        # 根据词频计算采样概率（f^(3/4)）
        words = list(self.word_to_idx.keys())
        freqs = np.array([self.word_freq[w] for w in words])
        probs = freqs ** 0.75
        probs = probs / probs.sum()
        
        while len(neg_samples) < self.neg_samples:
            neg_idx = np.random.choice(vocab_size, p=probs)
            if neg_idx != context_idx and neg_idx != center_idx:
                neg_samples.append(neg_idx)
        
        return neg_samples
    
    def generate_training_pairs(self, sentence: List[str]) -> List[Tuple[int, int]]:
        """
        为句子生成训练样本（中心词，上下文词）对
        
        参数:
            sentence: 词列表
        
        返回:
            训练样本列表 [(center_idx, context_idx), ...]
        """
        pairs = []
        sentence_idxs = [self.word_to_idx.get(w) for w in sentence]
        sentence_idxs = [idx for idx in sentence_idxs if idx is not None]  # 过滤OOV
        
        for i, center_idx in enumerate(sentence_idxs):
            # 上下文窗口
            start = max(0, i - self.window_size)
            end = min(len(sentence_idxs), i + self.window_size + 1)
            
            for j in range(start, end):
                if j != i:
                    context_idx = sentence_idxs[j]
                    pairs.append((center_idx, context_idx))
        
        return pairs
    
    def train(self, sentences: List[List[str]], epochs: int = 5):
        """
        训练Word2Vec模型
        
        参数:
            sentences: 训练语料（句子列表）
            epochs: 训练轮数
        """
        # 构建词汇表
        self.build_vocab(sentences)
        
        vocab_size = len(self.word_to_idx)
        print(f"\n开始训练...")
        
        for epoch in range(epochs):
            total_loss = 0
            num_samples = 0
            
            for sentence in sentences:
                # 生成训练样本
                pairs = self.generate_training_pairs(sentence)
                
                for center_idx, context_idx in pairs:
                    # 前向传播
                    h = self.W1[center_idx]  # 中心词的输入向量
                    
                    # 正样本更新
                    pos_score = np.dot(h, self.W2[context_idx])
                    pos_grad = self.sigmoid(pos_score) - 1  # 希望接近1
                    
                    # 更新W2[context_idx]
                    self.W2[context_idx] -= self.learning_rate * pos_grad * h
                    # 更新W1[center_idx]
                    self.W1[center_idx] -= self.learning_rate * pos_grad * self.W2[context_idx]
                    
                    total_loss += -np.log(self.sigmoid(pos_score) + 1e-10)
                    num_samples += 1
                    
                    # 负采样更新
                    neg_samples = self.get_negative_samples(center_idx, context_idx)
                    for neg_idx in neg_samples:
                        neg_score = np.dot(h, self.W2[neg_idx])
                        neg_grad = self.sigmoid(neg_score)  # 希望接近0
                        
                        self.W2[neg_idx] -= self.learning_rate * neg_grad * h
                        self.W1[center_idx] -= self.learning_rate * neg_grad * self.W2[neg_idx]
            
            avg_loss = total_loss / num_samples if num_samples > 0 else 0
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def get_vector(self, word: str) -> np.ndarray:
        """获取词的向量"""
        if word not in self.word_to_idx:
            raise ValueError(f"词 '{word}' 不在词汇表中")
        idx = self.word_to_idx[word]
        return self.W1[idx]  # 使用输入向量作为词向量
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        找出与给定词最相似的词
        
        参数:
            word: 目标词
            topn: 返回最相似的前n个词
        
        返回:
            [(词, 相似度), ...] 列表
        """
        if word not in self.word_to_idx:
            raise ValueError(f"词 '{word}' 不在词汇表中")
        
        target_vec = self.get_vector(word)
        target_norm = np.linalg.norm(target_vec)
        
        similarities = []
        for idx in range(len(self.word_to_idx)):
            candidate_vec = self.W1[idx]
            candidate_norm = np.linalg.norm(candidate_vec)
            
            if candidate_norm > 0 and target_norm > 0:
                cos_sim = np.dot(target_vec, candidate_vec) / (target_norm * candidate_norm)
                if self.idx_to_word[idx] != word:  # 排除自身
                    similarities.append((self.idx_to_word[idx], cos_sim))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 训练语料（简化版）
    sentences = [
        ["natural", "language", "processing", "is", "important"],
        ["machine", "learning", "is", "a", "subset", "of", "ai"],
        ["deep", "neural", "networks", "are", "powerful"],
        ["word2vec", "learns", "word", "embeddings"],
        ["nlp", "tasks", "include", "classification", "and", "translation"],
        ["language", "models", "like", "bert", "use", "transformers"],
        ["neural", "networks", "can", "learn", "representations"],
        ["word", "embeddings", "capture", "semantic", "meaning"],
    ]
    
    # 创建并训练模型
    w2v = SimpleWord2Vec(
        embed_dim=50,
        window_size=2,
        neg_samples=5,
        learning_rate=0.05,
        min_count=1
    )
    
    w2v.train(sentences, epochs=100)
    
    # 测试词相似度
    print("\n" + "="*60)
    print("词相似度测试")
    print("="*60)
    
    test_words = ["neural", "word", "language"]
    for word in test_words:
        if word in w2v.word_to_idx:
            similar = w2v.most_similar(word, topn=3)
            print(f"\n与 '{word}' 最相似的词:")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.4f}")
```

## 9. 可视化与结果理解

以下代码展示Word2Vec词向量的可视化，包括向量降维可视化、相似词展示等：

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 使用之前训练的模型（或重新训练一个小模型）
sentences = [
    ["king", "queen", "man", "woman", "prince", "princess"],
    ["apple", "banana", "fruit", "orange", "grape"],
    ["car", "truck", "vehicle", "bus", "bike"],
    ["computer", "laptop", "keyboard", "mouse", "screen"],
    ["happy", "joy", "sad", "angry", "emotion"],
    ["king", "ruled", "throne", "crown"],
    ["man", "worked", "job", "office"],
    ["woman", "she", "her", "queen"],
]

# 训练Word2Vec（使用gensim，因为我们的手工实现太小）
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=2,
    min_count=1,
    sg=1,
    negative=5,
    epochs=100,
    seed=42
)

print("Word2Vec训练完成！")
print(f"词汇表: {list(model.wv.index_to_key)}")

# ============================================
# 可视化1：PCA降维展示词向量
# ============================================

print("\n" + "="*60)
print("可视化：PCA降维展示词向量")
print("="*60)

# 获取所有词向量
words = list(model.wv.index_to_key)
vectors = np.array([model.wv[word] for word in words])

# PCA降维到2维
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, c='skyblue', s=100)

# 标注词
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                fontsize=10, ha='center', va='bottom')

plt.title('Word2Vec词向量可视化 (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(alpha=0.3)

# ============================================
# 可视化2：展示相似词
# ============================================

plt.subplot(1, 2, 2)

target_word = "king"
if target_word in model.wv:
    similar = model.wv.most_similar(target_word, topn=5)
    similar_words = [w for w, _ in similar]
    similar_scores = [s for _, s in similar]
    
    # 加上目标词自身
    plot_words = [target_word] + similar_words
    plot_vectors = np.array([model.wv[w] for w in plot_words])
    
    # PCA降维
    pca_small = PCA(n_components=2)
    plot_2d = pca_small.fit_transform(plot_vectors)
    
    # 绘制
    plt.scatter(plot_2d[1:, 0], plot_2d[1:, 1], alpha=0.6, c='lightcoral', s=100, label='相似词')
    plt.scatter(plot_2d[0, 0], plot_2d[0, 1], c='red', s=150, marker='*', label='目标词')
    
    for i, word in enumerate(plot_words):
        plt.annotate(word, (plot_2d[i, 0], plot_2d[i, 1]), 
                    fontsize=9, ha='center', va='bottom')
    
    plt.title(f'与 "{target_word}" 最相似的词')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 可视化3：TSNE降维（更好展示局部结构）
# ============================================

print("\n" + "="*60)
print("可视化：TSNE降维展示词向量")
print("="*60)

# 使用TSNE（计算较慢，但效果通常更好）
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
vectors_tsne = tsne.fit_transform(vectors)

plt.figure(figsize=(10, 8))
plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], alpha=0.6, c='green', s=100)

for i, word in enumerate(words):
    plt.annotate(word, (vectors_tsne[i, 0], vectors_tsne[i, 1]), 
                fontsize=10, ha='center', va='bottom')

plt.title('Word2Vec词向量可视化 (TSNE)')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.grid(alpha=0.3)
plt.show()

# ============================================
# 向量运算演示（类比推理）
# ============================================

print("\n" + "="*60)
print("向量运算演示（类比推理）")
print("="*60)

# 尝试：king - man + woman ≈ queen
try:
    result = model.wv.most_similar(
        positive=['king', 'woman'],
        negative=['man'],
        topn=3
    )
    print("\n类比推理：king - man + woman = ?")
    for word, score in result:
        print(f"  {word}: {score:.4f}")
except KeyError as e:
    print(f"某些词不在词汇表中: {e}")

# 计算词与词之间的相似度
word_pairs = [("king", "queen"), ("man", "woman"), ("apple", "banana"), ("car", "truck")]
print("\n词相似度计算:")
for w1, w2 in word_pairs:
    if w1 in model.wv and w2 in model.wv:
        sim = model.wv.similarity(w1, w2)
        print(f"  {w1} vs {w2}: {sim:.4f}")
```

**结果解读：**

1. **PCA可视化**：词向量在2D空间中的投影。语义相似的词（如"king"和"queen"）应该距离较近。

2. **相似词展示**：展示与"king"最相似的词，这些词应该在语义上相关。

3. **TSNE可视化**：TSNE更好地保留了局部结构，相似词应该聚在一起形成簇（如水果类、交通工具类等）。

4. **向量运算**：经典的"king - man + woman ≈ queen"演示，说明词向量捕捉到了语义关系。

5. **相似度计算**：语义相似的词对（如"apple"和"banana"）应该具有较高的余弦相似度。

## 10. 模型评估

评估Word2Vec词向量的质量主要通过以下几种方法：

```python
import numpy as np
from sklearn.metrics.cluster import silhouette_score
from collections import Counter

# ============================================
# 评估1：词相似度任务（WordSim353等）
# ============================================

print("=" * 60)
print("评估1：词相似度相关性")
print("=" * 60)

# 模拟人类标注的词相似度数据
# 真实数据集如WordSim353包含词对和人类打分的相似度
human_rated_pairs = [
    ("king", "queen", 0.8),   # 人类评分0-10或0-1
    ("car", "automobile", 0.9),
    ("computer", "keyboard", 0.5),
    ("happy", "sad", 0.2),
    ("apple", "banana", 0.7),
]

# 计算模型预测的相似度
model_similarities = []
human_similarities = []

for w1, w2, human_score in human_rated_pairs:
    if w1 in model.wv and w2 in model.wv:
        model_score = model.wv.similarity(w1, w2)
        model_similarities.append(model_score)
        human_similarities.append(human_score)
        print(f"  {w1}-{w2}: 模型={model_score:.3f}, 人类={human_score:.3f}")

# 计算相关性（简化版）
if len(model_similarities) >= 2:
    correlation = np.corrcoef(model_similarities, human_similarities)[0, 1]
    print(f"\n模型相似度与人类评分的相关性: {correlation:.3f}")

# ============================================
# 评估2：类比推理任务（准确率）
# ============================================

print("\n" + "="*60)
print("评估2：类比推理任务")
print("="*60)

# 定义类比推理问题
analogy_questions = [
    ("king", "man", "queen", "woman"),  # king:man :: queen:woman
    ("apple", "fruit", "car", "vehicle"),  # apple:fruit :: car:vehicle
    ("happy", "joy", "sad", "sorrow"),
]

correct = 0
total = 0

for w1, w2, w3, expected in analogy_questions:
    if all(w in model.wv for w in [w1, w2, w3, expected]):
        # 计算 w2 - w1 + w3
        result = model.wv.most_similar(
            positive=[w2, w3],
            negative=[w1],
            topn=1
        )
        predicted = result[0][0]
        
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        total += 1
        
        print(f"  {w1}:{w2} :: {w3}:? 预测={predicted}, 期望={expected}, {'正确' if is_correct else '错误'}")

if total > 0:
    accuracy = correct / total
    print(f"\n类比推理准确率: {accuracy:.2%} ({correct}/{total})")

# ============================================
# 评估3：词聚类质量
# ============================================

print("\n" + "="*60)
print("评估3：词聚类质量")
print("="*60)

# 获取所有词向量
words = list(model.wv.index_to_key)
vectors = np.array([model.wv[w] for w in words])

# 简单的KMeans聚类
from sklearn.cluster import KMeans

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(vectors)

# 打印每个簇的词
clusters = [[] for _ in range(n_clusters)]
for word, label in zip(words, cluster_labels):
    clusters[label].append(word)

print(f"\nKMeans聚类结果 (K={n_clusters}):")
for i, cluster in enumerate(clusters):
    print(f"  簇 {i+1}: {cluster}")

# 计算轮廓系数（聚类质量）
if len(set(cluster_labels)) > 1:  # 至少2个簇
    sil_score = silhouette_score(vectors, cluster_labels)
    print(f"\n轮廓系数 (Silhouette Score): {sil_score:.3f}")
    print("  (值越接近1表示聚类效果越好)")

# ============================================
# 评估4：下游任务性能（文本分类）
# ============================================

print("\n" + "="*60)
print("评估4：下游任务性能（模拟文本分类）")
print("="*60)

# 模拟简单的文本分类任务
# 文本：使用词向量的平均作为特征，训练分类器

def get_text_vector(words):
    """获取文本的向量表示（词向量平均）"""
    vectors = [model.wv[w] for w in words if w in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.wv.vector_size)

# 模拟数据：类别0（技术类）和类别1（水果类）
texts = [
    ["computer", "laptop", "keyboard"],
    ["apple", "banana", "fruit"],
    ["programming", "code", "software"],
    ["orange", "grape", "fruit"],
] * 10

labels = [0, 1, 0, 1] * 10  # 0=技术, 1=水果

# 提取特征
X = np.array([get_text_vector(text) for text in texts])
y = np.array(labels)

# 训练分类器
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n文本分类准确率: {acc:.2%}")
print("  (使用Word2Vec词向量作为特征)")
```

**评估指标说明：**

1. **词相似度相关性**：计算模型预测的相似度与人类标注的相似度之间的相关性（如Spearman相关系数）。相关性越高，说明词向量质量越好。

2. **类比推理准确率**：对于类比问题（如"king:man :: queen:?"），看模型是否能正确预测。准确率越高越好。

3. **词聚类质量**：将词向量聚类，计算轮廓系数。高质量的词向量应该使得语义相似的词聚在一起。

4. **下游任务性能**：将词向量应用于实际任务（如文本分类），看性能如何。这是最终也是最重要的评估。

**结果解读：**

- 在大规模语料上训练的Word2Vec，词相似度相关性通常可以达到0.7-0.8以上。
- 类比推理准确率在语义类比（如"king:queen"）上可能达到60-80%，但在语法类比（如"run:ran"）上可能较低。
- 聚类应该能看到清晰的语义簇（如水果类、交通工具类等）。
- 下游任务性能取决于具体任务和训练语料的相关性。

## 11. 常见问题与易错点

**数据层面问题：**

1. **低频词处理**：Word2Vec需要大量数据才能学到高质量的词向量。对于低频词（出现次数<5-10），词向量可能质量很差。解决方法：设置`min_count`参数过滤低频词，或使用字符级Word2Vec（FastText）。

2. **未登录词（OOV）**：传统Word2Vec无法处理训练词汇表外的词。解决方法：使用字符级方法（FastText）、子词分解（BPE+Word2Vec）、或使用上下文词嵌入（BERT）。

3. **语料规模不足**：Word2Vec需要大规模语料（千万到数亿词）才能学到高质量的词向量。小语料下，词向量可能不能很好捕捉语义关系。解决方法：使用预训练的词向量（如Google的Word2Vec、FastText等）。

**模型层面问题：**

1. **多义词混淆**：Word2Vec为每个词学习一个向量，无法区分多义词的多个含义。例如，"bank"作为"银行"和"河岸"在Word2Vec中是同一个向量。解决方法：使用上下文相关的词嵌入（如BERT、ELMo）。

2. **向量空间各向异性**：Word2Vec训练得到的词向量空间可能存在各向异性（不同方向的方差差异大），影响某些应用。解决方法：对词向量进行后处理（如减去均值、白化变换等）。

3. **负采样偏差**：负采样根据词频采样，可能导致高频词被过度采样为负样本。虽然使用$f^{3/4}$部分缓解了这个问题，但仍可能存在偏差。解决方法：调整采样分布，或使用其他采样策略。

**调参问题：**

1. **向量维度选择**：维度太低（如50）可能无法捕捉足够信息；维度太高（如500）可能导致过拟合、计算开销大。对于小语料，建议使用50-100维；对于大规模语料，可以使用300维。解决方法：在下游任务上验证不同维度的效果。

2. **窗口大小选择**：窗口太小（如2）只能捕捉局部上下文；窗口太大（如20）可能引入噪声、增加计算量。通常5-10是比较合理的范围。解决方法：根据任务特点选择（如情感分析可能需要更大的窗口捕捉否定词的影响）。

3. **负采样数选择**：负采样数太少（如1-2）可能导致训练不稳定；太多（如20+）增加计算量。通常5-10是一个合理的范围。解决方法：在验证集上测试不同配置。

## 12. 学习总结

Word2Vec是一种革命性的词嵌入技术，通过浅层神经网络从大规模文本语料中学习词的向量表示。它基于分布假说，通过Skip-gram或CBOW架构，使得语义相似的词在向量空间中距离相近。

从原理层面，Word2Vec通过预测词的上下文（Skip-gram）或由上下文预测词（CBOW）来训练词向量。负采样技术的引入使得Word2Vec能够在大规模语料上高效训练。学到的词向量不仅捕捉了语义相似性，还表现出了线性语义结构（如"king - man + woman ≈ queen"）。

在实践层面，我们学习了如何使用gensim库训练和应用Word2Vec，以及如何从零实现一个简化版的Skip-gram模型。关键要点包括：词汇表构建、训练样本生成、负采样策略、以及词向量的提取和评估。

Word2Vec虽然相对简单，但其影响深远。它证明了简单的神经网络模型可以学到高质量的词表示，为后续的深度学习在NLP中的广泛应用铺平了道路。然而，Word2Vec也有其局限性：无法处理多义词、静态词向量无法根据上下文调整、对大规模语料依赖等。这些局限性推动了后续模型（如ELMo、BERT）的发展。

总之，Word2Vec是NLP工程师和研究者必学的基础知识。掌握Word2Vec的原理和实现，对于理解现代NLP技术栈、以及后续学习更先进的模型都具有重要意义。

## 13. 练习题与思考题

**基础题：**

1. **实现简单词向量查询**：给定一个小型语料，手动构建词汇表，为每个词随机初始化一个2维向量，并实现一个函数计算两个词的余弦相似度。

   <details>
   <summary>答案</summary>
   ```python
   import numpy as np
   from collections import Counter
   
   # 小型语料
   sentences = [["cat", "sat"], ["dog", "ran"], ["cat", "meowed"]]
   
   # 构建词汇表
   word_freq = Counter()
   for sent in sentences:
       word_freq.update(sent)
   
   vocab = list(word_freq.keys())
   word_to_idx = {w: i for i, w in enumerate(vocab)}
   
   # 随机初始化2维词向量
   np.random.seed(42)
   word_vectors = np.random.randn(len(vocab), 2) * 0.1
   
   def cosine_similarity(w1, w2):
       """计算两个词的余弦相似度"""
       vec1 = word_vectors[word_to_idx[w1]]
       vec2 = word_vectors[word_to_idx[w2]]
       dot = np.dot(vec1, vec2)
       norm1 = np.linalg.norm(vec1)
       norm2 = np.linalg.norm(vec2)
       return dot / (norm1 * norm2)
   
   # 测试
   print(f"cat vs dog: {cosine_similarity('cat', 'dog'):.4f}")
   print(f"cat vs sat: {cosine_similarity('cat', 'sat'):.4f}")
   ```
   </details>

2. **理解Skip-gram和CBOW的区别**：用一句话总结Skip-gram和CBOW的主要区别，并说明在什么场景下应该选择哪种架构。

   <details>
   <summary>答案</summary>
   **区别总结**：
   - Skip-gram：根据中心词预测上下文词（输入是中心词，输出是上下文词的概率分布）
   - CBOW：根据上下文词预测中心词（输入是上下文词，输出是中心词的概率分布）
   
   **选择建议**：
   - 如果语料较小或需要更快的训练速度：选择**CBOW**
   - 如果语料较大、关注低频词、或需要更精确的语义关系：选择**Skip-gram**
   - 通常Skip-gram的效果稍好，但训练更慢
   </details>

**进阶题：**

3. **实现负采样逻辑**：编写一个函数，给定中心词和上下文词，采样5个负样本词。采样概率应该与词频的3/4次方成正比。

   <details>
   <summary>答案</summary>
   ```python
   import numpy as np
   from collections import Counter
   
   def negative_sampling(center_word, context_word, vocab, word_freq, k=5):
       """
       负采样函数
       
       参数:
           center_word: 中心词
           context_word: 上下文词（正样本）
           vocab: 词汇表列表
           word_freq: 词频字典
           k: 负样本数
       
       返回:
           负样本词列表
       """
       # 计算采样概率
       vocab_list = list(vocab)
       freqs = np.array([word_freq[w] for w in vocab_list])
       probs = freqs ** 0.75  # f^(3/4)
       probs = probs / probs.sum()  # 归一化
       
       neg_samples = []
       while len(neg_samples) < k:
           # 根据概率采样
           neg_word = np.random.choice(vocab_list, p=probs)
           # 确保不是正样本或中心词
           if neg_word != context_word and neg_word != center_word:
               neg_samples.append(neg_word)
       
       return neg_samples
   
   # 测试
   word_freq = Counter({"cat": 10, "dog": 8, "sat": 5, "ran": 3, "meowed": 2})
   vocab = list(word_freq.keys())
   
   neg = negative_sampling("cat", "sat", vocab, word_freq, k=5)
   print(f"负样本: {neg}")
   ```
   </details>

4. **实现向量类比运算**：给定一个训练好的Word2Vec模型（或使用gensim加载预训练模型），实现函数完成类比推理：a:b :: c:?，即计算 a - b + c，找出最接近的向量。

   <details>
   <summary>答案</summary>
   ```python
   import numpy as np
   
   def analogy(model, a, b, c, topn=1):
       """
       类比推理：a:b :: c:?
       
       参数:
           model: Word2Vec模型（有.wv属性）
           a, b, c: 词
       
       返回:
           预测的词和相似度
       """
       # 计算目标向量：b - a + c
       try:
           vec_a = model.wv[a]
           vec_b = model.wv[b]
           vec_c = model.wv[c]
           
           target_vec = vec_b - vec_a + vec_c
           
           # 找出最相似的词（排除a, b, c自身）
           result = model.wv.most_similar(
               positive=[vec_b, vec_c],
               negative=[vec_a],
               topn=topn+3  # 多取几个，然后过滤
           )
           
           # 过滤掉a, b, c
           final_result = []
           for word, score in result:
               if word not in [a, b, c]:
                   final_result.append((word, score))
                   if len(final_result) >= topn:
                       break
           
           return final_result
       except KeyError as e:
           return f"词不在词汇表中: {e}"
   
   # 测试（需要预训练模型）
   # print(analogy(model, 'king', 'man', 'woman'))
   print("类比推理函数已实现，需要使用预训练模型测试")
   ```
   </details>

**开放题：**

5. **改进Word2Vec**：Word2Vec有哪些局限性？如果你要设计一个更好的词嵌入模型，你会做哪些改进？考虑多义词、子词信息、上下文动态性等方面。

   <details>
   <summary>参考答案要点</summary>
   **Word2Vec的局限性**：
   1. 无法处理多义词（一个词一个向量）
   2. 静态词向量，无法根据上下文调整
   3. 对大规模语料依赖强
   4. 无法处理未登录词（OOV）
   5. 不考虑子词信息（词根、前缀、后缀）
   
   **改进方向**：
   1. **上下文相关词嵌入**（如ELMo、BERT）：
      - 根据上下文动态生成词向量
      - 解决多义词问题
   
   2. **子词信息**（如FastText）：
      - 将词分解为字符n-gram
      - 能够处理OOV词
      - 捕捉形态学信息
   
   3. **知识增强**：
      - 结合知识图谱信息
      - 注入领域知识
   
   4. **多任务学习**：
      - 同时优化多个目标（如语言模型、同义词预测等）
      - 提高词向量的泛化能力
   
   5. **跨语言词向量**：
      - 学习多语言共享的词向量空间
      - 支持跨语言迁移
   </details>

## 14. 学习路径建议

**前置知识：**
- Python基础（NumPy数组操作、字典处理）
- 基础线性代数（向量、矩阵、点积、余弦相似度）
- 基础概率统计（概率分布、采样）
- 神经网络基础（前向传播、梯度下降）

**平行学习：**
- GloVe（另一种词嵌入方法，基于矩阵分解）
- FastText（Word2Vec的改进，支持子词信息）
- Word2Vec可视化（PCA、TSNE降维）
- 词嵌入评估方法（相似度、类比推理）

**进阶方向：**
- 上下文词嵌入（ELMo、BERT、GPT）
- 句子嵌入（Sentence-BERT、Universal Sentence Encoder）
- 跨语言词向量（多语言Word2Vec训练）
- 知识增强词嵌入（结合知识图谱）

**推荐资源：**
1. **Word2Vec原论文**: https://arxiv.org/abs/1301.3781 - Efficient Estimation of Word Representations in Vector Space
2. **Gensim官方文档**: https://radimrehurek.com/gensim/models/word2vec.html - Word2Vec的工业级实现
3. **Stanford CS224n Notes**: http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf - Word2Vec的详细讲义
4. **Visualizing Word2Vec**: https://projector.tensorflow.org/ - Google的互动式词向量可视化工具

通过系统学习Word2Vec，你将掌握词嵌入的核心思想和技术，为理解现代NLP模型（如BERT、GPT）打下坚实基础。词向量作为NLP的基础特征表示，在各类任务中都有广泛应用。
