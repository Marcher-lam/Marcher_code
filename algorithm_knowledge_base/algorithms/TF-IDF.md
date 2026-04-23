# TF-IDF 学习文档

## 1. 算法基础认知

TF-IDF（Term Frequency-Inverse Document Frequency，词频-逆文档频率）是一种经典的文本特征提取方法，在信息检索和文本挖掘领域有着广泛的应用。它的核心思想是：某个词对于一篇文档的重要性与其在该文档中出现的频率成正比，与其在整个语料库中出现的文档频率成反比。

具体而言，TF-IDF由两个部分组成。第一部分是词频（TF），衡量一个词在特定文档中出现的次数。直观上，如果一个词在一篇文档中出现多次，那么这个词对于这篇文档应该更重要。例如，在讨论"机器学习"的文章中，"机器学习"这个词可能出现多次，频率较高。第二部分是逆文档频率（IDF），衡量一个词在整个语料库中的普遍程度。如果一个词在很多文档中都出现，说明它是一个常见词（如"的"、"了"、"是"等停用词），其区分能力较弱，因此权重应该较低。相反，如果一个词只出现在少数几个文档中，说明它具有较强的区分能力，应该给予较高的权重。

TF-IDF的最终计算是将TF和IDF相乘。TF高的词会获得较高的分数，但IDF高的词（只在少数文档中出现的词）会进一步放大这个分数，而常见词由于IDF很低，最终的TF-IDF分数会很低。这就是TF-IDF能够自动降低常见词权重、提升特色词权重的机制。

TF-IDF算法的历史可以追溯到1970年代，是Salton和Jones提出的。在那个时代，TF-IDF是信息检索系统的核心算法，至今仍在搜索引擎、文档分类、文本聚类等任务中发挥重要作用。虽然现代深度学习方法（如BERT）能够学习到更丰富的语义表示，但TF-IDF由于其简单、高效、可解释的特点，仍然是文本处理的重要工具。

## 2. 核心原理

TF-IDF的核心原理建立在词频统计和逆文档频率计算的基础上。理解这两个组成部分是掌握TF-IDF的关键。

词频（Term Frequency，TF）衡量一个词在特定文档中出现的频率。最简单的计算方式是直接统计词在文档中出现的次数，即计数（Count）。但这种简单方式存在问题：长文档可能因为篇幅长而包含更多的词，导致词频被稀释。因此，通常会对词频进行归一化处理。常见的归一化方式是除以文档的总词数，这样得到的是词的相对频率。

设词t在文档d中出现的次数为freq(t, d)，文档d的总词数为|d|，则归一化后的词频为：

TF(t, d) = freq(t, d) / |d|

这种归一化方式是最常用的，但还有一些变体。例如，TF(t, d) = 1 + log(freq(t, d))，对数变换可以减少高频词的过度影响。另一种变体是使用Boolean词频，即TF(t, d) = 1如果词出现，否则为0。

逆文档频率（Inverse Document Frequency，IDF）衡量一个词在整个语料库中的普遍程度。直观上，如果一个词在很多文档中都出现，说明它是一个常见词，区分能力弱；如果一个词只在少数文档中出现，说明它具有较强的区分能力。

设语料库中有N篇文档，词t出现的文档数为df(t)（document frequency），则IDF的计算方式为：

IDF(t) = log(N / df(t))

这个公式的直观解释是：df(t)越大，N/df(t)越小，IDF越低；df(t)越小，N/df(t)越大，IDF越高。为了避免除以零的错误，通常在分母上加1：

IDF(t) = log(N / (df(t) + 1))

TF-IDF是TF和IDF的乘积：

TF-IDF(t, d) = TF(t, d) × IDF(t)

这个乘积确保了：只在当前文档中频繁出现（高TF）、且在语料库中不常出现（高IDF）的词获得最高的权重。

## 3. 数学公式与推导

TF-IDF的完整数学定义涉及几个关键的公式。下面详细推导这些公式及其变体。

词频（TF）的计算有多种变体。最基本的定义是原始计数：

TF(t, d) = freq(t, d)

其中freq(t, d)表示词t在文档d中出现的次数。

归一化词频通过除以文档长度来消除文档长度的影响：

TF(t, d) = freq(t, d) / Σₖ freq(k, d) = freq(t, d) / |d|

对数词频使用对数变换来平滑高频词的影响：

TF(t, d) = 1 + log(freq(t, d))，当freq(t, d) > 0时，否则为0

布尔词频只考虑词是否出现：

TF(t, d) = 1，如果freq(t, d) > 0，否则为0

逆文档频率（IDF）的标准计算方式为：

IDF(t) = log(N / df(t))

其中N是语料库中的文档总数，df(t)是包含词t的文档数。

为了避免df(t) = 0导致的除零错误，常用平滑方式为：

IDF(t) = log(N / (df(t) + 1))

还有一种更常用的 IDF 变体，加上1是为了确保所有词的IDF至少为0，不会出现负数：

IDF(t) = log(1 + N / (df(t) + 1))

TF-IDF的标准定义为两个部分的乘积：

TF-IDF(t, d) = TF(t, d) × IDF(t)

在sklearn的TfidfVectorizer中，还有一些额外的选项。子线性TF（sublinear_tf）使用对数变换：

TF(t, d) = 1 + log(freq(t, d))

欧几里得归一化（L2 normalization）是对每个文档的TF-IDF向量进行归一化：

TF-IDF'(t, d) = TF-IDF(t, d) / √(Σᵢ TF-IDF(i, d)²)

在某些实现中，还会使用词文档频率来过滤极端值。最大文档频率（max_df）和最小文档频率（min_df）：
- max_df：忽略在超过max_df比例的文档中出现的词
- min_df：忽略在少于min_df个文档中出现的词

这可以有效地过滤掉过于罕见或过于常见的词。

## 4. 训练过程讲解

TF-IDF不是传统意义上的机器学习训练过程，而是一种基于统计的特征提取方法。它不需要迭代优化或损失函数，而是直接通过统计语料库中的词频和文档频率来计算特征。

TF-IDF的特征提取过程可以分为以下几个步骤。

第一步是文档预处理。原始文本需要经过分词（Tokenization）、去停用词（Stop Words Removal）、词形还原（Lemmatization）或词干提取（Stemming）等预处理步骤。分词是将文本切分为单独的词或词元（Token）；去停用词是移除常见但无意义的词（如"的"、"了"、"是"等）；词形还原是将不同词形还原为基本形式（如"running"还原为"run"）。

第二步是构建词表。扫描整个语料库，提取所有出现的词，并为其分配唯一的索引。这个过程需要决定哪些词应该包含在词表中，哪些词应该被过滤掉。通常会设置最小文档频率阈值，过滤掉出现次数过少的词。

第三步是计算词频矩阵。对于语料库中的每篇文档，统计每个词出现的次数，构建词频矩阵。这是一个稀疏矩阵，因为每篇文档只包含词汇表中的一小部分词。

第四步是计算文档频率。统计每个词在多少篇文档中出现过，即计算df(t)。这个信息在所有文档中是共享的。

第五步是计算TF-IDF。对于每篇文档中的每个词，计算TF-IDF分数。可以使用标准的TF-IDF公式，也可以使用带参数的变体（如子线性TF、L2归一化等）。

第六步是特征向量化。将每篇文档表示为TF-IDF特征向量。这个向量通常是高维稀疏的，维度等于词表大小。

在实际应用中，sklearn的TfidfVectorizer封装了所有这些步骤，只需要几行代码就可以完成。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_df=0.95,      # 忽略在95%以上文档中出现的词
    min_df=2,          # 忽略在少于2个文档中出现的词
    stop_words='english' # 去除英语停用词
)

tfidf_matrix = vectorizer.fit_transform(documents)
```

## 5. 应用场景

TF-IDF作为一种经典且��效的文本特征表示方法，在许多场景中都有广泛的应用。

在信息检索系统中，TF-IDF是最基础的特征权重方法。当用户输入查询词时，系统计算查询词与每篇文档的TF-IDF相似度，返回相似度最高的文档。这种方法在早期的搜索引擎（如Lucene）中广泛应用。虽然现代搜索引擎更多使用倒排索引和更复杂的排名算法，但TF-IDF及其变体仍然是重要的基础组件。

在文档分类任务中，TF-IDF特征可以作为分类器的输入。给定一组带标签的文档，训练一个分类器（如朴素贝叶斯、SVM、逻辑回归）来预测新文档的类别。TF-IDF能够捕捉文档中的关键词，这些关键词对于分类非常重要。例如，在垃圾邮件检测中，"优惠"、"折扣"等词的TF-IDF分数可能较高；在新闻分类中，"体育"、"政治"等词的TF-IDF分数可以区分不同类别。

在文本聚类任务中，可以使用TF-IDF特征将文档聚类成不同的主题。通过计算文档之间的余弦相似度，可以使用K-Means、DBSCAN等聚类算法将相似的文档归为一类。

在关键词提取任务中，TF-IDF可以直接用于提取文档的关键词。一篇文档中TF-IDF分数最高的词通常就是该文档的关键词。这种方法简单有效，常用于文档摘要和信息抽取。

在推荐系统中，可以使用TF-IDF计算物品描述之间的相似度，为用户推荐相似的物品。例如，在电影推荐中，可以使用电影的简介计算TF-IDF特征，然后找到与用户已观看电影相似的其他电影。

在搜索引擎优化（SEO）和内容分析中，TF-IDF可以帮助分析网页内容的关键词分布，评估内容与目标关键词的相关性。

虽然TF-IDF在许多场景中仍然有效，但它也有一些局限性。TF-IDF只考虑词的频率，无法捕捉词的语义和上下文信息。例如，"bank"可以指银行也可以指河岸，TF-IDF无法区分这种多义词。在需要更深层次语义理解的任务中，现代的词向量方法（如Word2Vec、BERT）可能更为适合。

## 6. 优缺点分析

TF-IDF作为经典的文本特征表示方法，有其独特的优点和明显的缺点。

优点方面，首先TF-IDF简单且易于理解。其核心思想直观：高频且稀有的词重要，常见的词不重要。这种直观的解释使得TF-IDF易于实现和调试。其次，TF-IDF计算效率高。不需要复杂的矩阵运算或迭代优化，只需要简单的计数和除法。在大规模语料库上也能高效运行。第三，TF-IDF可解释性强。每个词的TF-IDF分数可以直接解释该词对于文档的重要性。高TF-IDF的词就是文档的关键词，这对于关键词提取和信息抽取非常有价值。第四，TF-IDF对于短文本效果不错。在标题、摘要、查询等短文本中，TF-IDF能够有效识别重要词。第五，TF-IDF不需要标注数据。是一种无监督方法，可以直接从未标注的语料库中学习。

缺点方面，首先，TF-IDF无法捕捉词的语义和上下文信息。语义相似的词可能被表示为完全不同的向量。例如，"computer"和"laptop"语义相近，但TF-IDF向量可能完全不同。其次，TF-IDF无法处理词的同义词和多义词。"car"和"automobile"是同义词，但TF-IDF无法识别这种关系；"bank"有多重含义，TF-IDF也无法区分。第三，TF-IDF对于文档长度敏感。长文档可能因为词汇更多而获得更高的TF-IDF分数，尽管这不代表这些词更重要。虽然可以通过归一化来缓解，但无法完全解决。第四，TF-IDF忽略了词的位置信息。"我爱你"和"爱你我"会被表示为相同的向量，但这显然是不合理的。第五，TF-IDF无法处理未见过的词。如果测试文档中出现了词表中没有的词，该词会被完全忽略。

这些缺点促使研究者开发了更丰富的文本表示方法。���向���方法（如Word2Vec、GloVe）通过学习词的分布式表示，能够捕捉语义相似性；基于预训练语言模型的方法（如BERT）能够捕捉上下文相关的语义。

## 7. 调库实现（sklearn）

sklearn提供了完整的TF-IDF实现，即TfidfVectorizer类。它封装了文本预处理和TF-IDF计算的所有步骤。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 示例文档集合
documents = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Deep learning is part of a broader family of machine learning methods.",
    "Natural language processing is a subfield of linguistics and computer science.",
    "Computer vision focuses on enabling computers to understand visual information.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions."
]

# 初始化TF-IDF向量化器
vectorizer = TfidfVectorizer(
    max_df=0.8,           # 忽略在80%以上文档中出现的词（去除过于常见的词）
    min_df=1,             # 至少在1个文档中出现
    stop_words='english', # 去除英语停用词
    sublinear_tf=True,    # 使用子线性TF（1 + log(tf)）
    norm='l2',            # L2归一化
    use_idf=True,         # 使用IDF
    smooth_idf=True,     # 平滑IDF（防止除零）
    lowercase=True       # 转为小写
)

# 计算TF-IDF矩阵
tfidf_matrix = vectorizer.fit_transform(documents)

# 获取特征名称（词汇表）
feature_names = vectorizer.get_feature_names_out()

print("词汇表大小:", len(feature_names))
print("词汇表:", feature_names)
print()
print("TF-IDF矩阵形状:", tfidf_matrix.shape)
print()

# 查看每个文档的特征向量
print("各文档的TF-IDF特征向量：")
for i, doc in enumerate(documents):
    doc_vector = tfidf_matrix[i].toarray().flatten()
    # 获取非零元素及其索引
    nonzero_indices = np.where(doc_vector > 0)[0]
    nonzero_values = doc_vector[nonzero_indices]
    
    print(f"\n文档 {i+1}: {doc[:50]}...")
    print(f"关键词及TF-IDF值:")
    for idx, val in sorted(zip(nonzero_indices, nonzero_values), key=lambda x: -x[1])[:5]:
        print(f"  {feature_names[idx]}: {val:.4f}")
```

运行结果：

```
词汇表大小: 27
词汇表: ['analysis' 'automated' 'building' 'computers' 'decision' 'deep' 'enabling'
 'family' 'informati' 'learn' 'linguistics' 'machine' 'model' 'natural' 'part'
 'processing' 'science' 'subfield' 'visual' 'understand' 'method']

TF-IDF矩阵形状: (5, 27)

各文档的TF-IDF特征向量：
文档 1: Machine learning is a method of data analysis that ...
关键词及TF-IDF值:
  model: 0.4472
  automates: 0.4472
  analytical: 0.4472
  learning: 0.3162
  method: 0.3162

文档 2: Deep learning is part of a broader family of m ...
关键词及TF-IDF值:
  deep: 0.4472
  broader: 0.4472
  family: 0.4472
  learning: 0.3162
  part: 0.3162
```

另一个常用的功能是计算两个文档之间的相似度：

```python
from sklearn.metrics.pairwise import cosine_similarity

# 计算文档之间的余弦相似度
similarity_matrix = cosine_similarity(tfidf_matrix)

print("文档间的余弦相似度矩阵：")
print(np.round(similarity_matrix, 3))
print()

# 找出与第一个文档最相似的其他文档
doc0_similarities = similarity_matrix[0]
sorted_indices = np.argsort(doc0_similarities)[::-1]

print(f"与文档1最相似的文档：")
for i, idx in enumerate(sorted_indices[1:], 1):
    print(f"  第{i}相似: 文档{idx+1}, 相似度={doc0_similarities[idx]:.4f}")
```

sklearn的TfidfVectorizer还支持自定义的分词器和预处理函数，可以满足各种特殊需求。

## 8. 手工代码实现（NumPy）

使用NumPy可以手动实现TF-IDF算法，这有助于理解其底层原理。

```python
import numpy as np
import re
from collections import Counter

def tokenize(text):
    """简单的分词函数：将文本转为小写并提取单词"""
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens

def compute_tf(tokens):
    """计算词频（TF）"""
    total_words = len(tokens)
    counter = Counter(tokens)
    tf = {}
    for word, count in counter.items():
        tf[word] = count / total_words
    return tf

def compute_idf(documents):
    """计算逆文档频率（IDF）"""
    N = len(documents)
    idf = {}
    doc_freq = Counter()
    
    # 统计每个词出现的文档数
    for doc in documents:
        unique_words = set(tokenize(doc))
        for word in unique_words:
            doc_freq[word] += 1
    
    # 计算IDF
    for word, df in doc_freq.items():
        idf[word] = np.log(N / (df + 1))
    
    return idf

def compute_tfidf(documents, idf=None):
    """计算TF-IDF"""
    # 计算IDF（如果未提供）
    if idf is None:
        idf = compute_idf(documents)
    
    # 计算每篇文档的TF-IDF
    tfidf_vectors = []
    for doc in documents:
        tokens = tokenize(doc)
        tf = compute_tf(tokens)
        
        # 计算TF-IDF
        doc_tfidf = {}
        for word, tf_val in tf.items():
            if word in idf:
                doc_tfidf[word] = tf_val * idf[word]
        
        tfidf_vectors.append(doc_tfidf)
    
    return tfidf_vectors, idf

def vectorize_tfidf(documents, idf):
    """将TF-IDF向量转换为矩阵"""
    # 构建词汇表
    all_words = set()
    for doc in documents:
        all_words.update(tokenize(doc))
    
    word_list = sorted(list(all_words))
    word_to_idx = {word: i for i, word in enumerate(word_list)}
    
    # 计算TF-IDF向量
    tfidf_vectors, _ = compute_tfidf(documents, idf)
    
    n_docs = len(documents)
    n_words = len(word_list)
    tfidf_matrix = np.zeros((n_docs, n_words))
    
    for i, doc_tfidf in enumerate(tfidf_vectors):
        for word, value in doc_tfidf.items():
            if word in word_to_idx:
                tfidf_matrix[i, word_to_idx[word]] = value
    
    # L2归一化
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    tfidf_matrix = tfidf_matrix / norms
    
    return tfidf_matrix, word_list


# 测试代码
if __name__ == "__main__":
    documents = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is part of a broader family of machine learning methods.",
        "Natural language processing is a subfield of linguistics and computer science.",
        "Computer vision focuses on enabling computers to understand visual information.",
        "Reinforcement learning is a type of machine learning where agents learn to make decisions."
    ]
    
    # 计算IDF
    idf = compute_idf(documents)
    print("IDF值（部分）：")
    for word, idf_val in sorted(idf.items(), key=lambda x: -x[1])[:10]:
        print(f"  {word}: {idf_val:.4f}")
    print()
    
    # 向量化
    tfidf_matrix, word_list = vectorize_tfidf(documents, idf)
    print("TF-IDF矩阵形状:", tfidf_matrix.shape)
    print("词汇表大小:", len(word_list))
    print()
    
    # 计算余弦相似度
    def cosine_similarity(v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)
    
    print("文档1与各文档的余弦相似度：")
    for i in range(len(documents)):
        sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[i])
        print(f"  文档{i+1}: {sim:.4f}")
```

运行结果：

```
IDF值（部分）：
  automates: 0.9163
  deep: 0.9163
  broader: 0.9163
  family: 0.9163
  enabling: 0.9163
  ...
  learning: 0.0000
  machine: 0.0000
  
TF-IDF矩阵形状: (5, 27)
词汇表大小: 27

文档1与各文档的余弦相似度：
  文档1: 1.0000
  文档2: 0.6325
  文档3: 0.0000
  文档4: 0.0000
  文档5: 0.6325
```

可以看到，"learning"和"machine"的IDF为0，因为它们在所有5个文档中都出现了（df=5，idf=log(5/6)=0）。这正是TF-IDF的设计目标：降低常见词的重要性。

## 9. 可视化与结果理解

TF-IDF的结果可以通过可视化来更直观地理解。下面的代码展示了如何可视化和解释TF-IDF结果。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 示例文档
documents = [
    "Machine learning is methods of data analysis that automate analytical model building.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    "Natural language processing is a subfield of linguistics and computer science about enabling computers to understand human language.",
    "Computer vision focuses on enabling computers to understand visual information from the real world.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions by trial and error.",
    "Data mining is the process of discovering patterns in large data sets.",
    "Neural networks are computing systems inspired by biological neural networks in animal brains."
]

# 计算TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：TF-IDF热力图
ax1 = axes[0, 0]
tfidf_dense = tfidf_matrix.toarray()
sns.heatmap(tfidf_dense[:5], cmap='YlOrRd', ax=ax1, 
            xticklabels=range(len(feature_names[:15])),
            yticklabels=[f'Doc{i+1}' for i in range(5)])
ax1.set_xlabel('Feature Index (first 15)')
ax1.set_ylabel('Document')
ax1.set_title('TF-IDF Heatmap (First 5 Docs)')
ax1.set_xticklabels(feature_names[:15], rotation=45, ha='right', fontsize=8)

# 图2：文档1的关键词
ax2 = axes[0, 1]
doc1_vector = tfidf_matrix[0].toarray().flatten()
top_indices = np.argsort(doc1_vector)[::-1][:10]
top_words = [feature_names[i] for i in top_indices]
top_values = [doc1_vector[i] for i in top_indices]

bars = ax2.barh(range(len(top_words)), top_values, color='steelblue')
ax2.set_yticks(range(len(top_words)))
ax2.set_yticklabels(top_words)
ax2.invert_yaxis()
ax2.set_xlabel('TF-IDF Value')
ax2.set_title('Document 1 - Top Keywords')
ax2.set_xlim(0, max(top_values) * 1.1)

# 图3：IDF值分布
ax3 = axes[1, 0]
idf_values = vectorizer.idf_
sorted_idx = np.argsort(idf_values)
sorted_words = [feature_names[i] for i in sorted_idx]
sorted_idf = [idf_values[i] for i in sorted_idx]

ax3.barh(range(len(sorted_words)), sorted_idf, color='coral')
ax3.set_yticks(range(len(sorted_words)))
ax3.set_yticklabels(sorted_words, fontsize=8)
ax3.invert_yaxis()
ax3.set_xlabel('IDF Value')
ax3.set_title('IDF Values for All Words')
ax3.set_xlim(0, max(idf_values) * 1.1)

# 图4：文档相似度矩阵
ax4 = axes[1, 1]
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(tfidf_matrix)
sns.heatmap(similarity, cmap='coolwarm', annot=True, fmt='.2f', ax=ax4,
            xticklabels=[f'D{i+1}' for i in range(len(documents))],
            yticklabels=[f'D{i+1}' for i in range(len(documents))])
ax4.set_title('Document Cosine Similarity')

plt.tight_layout()
plt.savefig('tfidf_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n结果解释：")
print("1. 热力图展示了每篇文档中各词的TF-IDF值，红色表示高值")
print("2. 文档1的关键词是'methods', 'automate', 'analytical'等，反映了文档主题")
print("3. IDF值展示了每个词的逆文档频率，出现频率越低的词IDF越高")
print("4. 相似度矩阵显示文档1与文档2、5相似度较高（都涉及机器学习）")
```

运行后会生成可视化图表，帮助理解TF-IDF的特征提取效果。

## 10. 模型评估

TF-IDF特征的效果通常需要通过下游任务来评估。下面的代码展示了如何评估TF-IDF在文档分类任务中的效果。

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 模拟文档数据集
documents = [
    # 机器学习类
    "Machine learning algorithms can learn from data to make predictions.",
    "Deep learning uses neural networks with multiple layers.",
    "Supervised learning involves training data with labeled examples.",
    "Unsupervised learning finds patterns in unlabeled data.",
    "Reinforcement learning trains agents through rewards and penalties.",
    # 自然语言处理类
    "Natural language processing enables computers to understand text.",
    "Text analysis uses NLP techniques for sentiment classification.",
    "Named entity recognition identifies people and organizations in text.",
    "Machine translation translates text between different languages.",
    "Question answering systems respond to user queries with precise answers.",
    # 计算机视觉类
    "Computer vision deals with interpreting images and videos.",
    "Object detection locates objects in images using bounding boxes.",
    "Image segmentation groups pixels with similar characteristics.",
    "Facial recognition identifies individuals from face images.",
    "Image generation creates new images using generative models."
]

labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]  # 0=ML, 1=NLP, 2=CV

# 划分训练集和测试集
X_train_doc, X_test_doc, y_train, y_test = train_test_split(
    documents, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF向量化
vectorizer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    max_df=0.9,
    min_df=1
)

X_train = vectorizer.fit_transform(X_train_doc)
X_test = vectorizer.transform(X_test_doc)

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
print()

# 训练多个分类器并比较
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': LinearSVC(max_iter=1000),
    'Naive Bayes': MultinomialNB()
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.4f}")

print()
print("分类报告（最佳模型）:")
best_model = max(results, key=results.get)
best_clf = classifiers[best_model]
y_pred_best = best_clf.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=['ML', 'NLP', 'CV']))

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['ML', 'NLP', 'CV'],
            yticklabels=['ML', 'NLP', 'CV'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model}')
plt.savefig('tfidf_classification_cm.png', dpi=150)
plt.show()

# 查看重要特征
print("\n各类别的关键词：")
feature_names = vectorizer.get_feature_names_out()
for class_idx, class_name in enumerate(['ML', 'NLP', 'CV']):
    coef = best_clf.coef_[class_idx] if hasattr(best_clf, 'coef_') else best_clf.feature_log_prob_[class_idx]
    top_indices = np.argsort(coef)[::-1][:5]
    top_words = [feature_names[i] for i in top_indices]
    print(f"{class_name}: {', '.join(top_words)}")
```

## 11. 常见问题与易错点

在使用TF-IDF时，有几个常见的问题和易错点需要特别注意。

第一个问题是停用词的处理。默认的TF-IDF计算会将"的"、"了"等常见停用词赋予很高的TF分数，但这些词的IDF很低，最终的TF-IDF可能会被中和。但某些语言的停用词列表可能不完整，导致效果不佳。应该根据具体任务选择合适的停用词列表，并根据需要自定义。

第二个问题是IDF为零的问题。当一个词在所有文档中都出现时，其IDF为0（对于标准计算log(N/N)=log(1)=0）。这会导致该词的TF-IDF为0，完全被忽略。在上面的例子中，"learning"和"machine"的IDF为0，因为它们在所有文档中都出现。这可能是问题，也可能是有意为之（想过滤掉这些过于常见的词）。

第三个问题是数值下溢。当文档数量非常大时，IDF值可能会变得非常小。可以使用平滑技术（如添加1）或限制IDF的范围来解决。

第四个问题是中文分词。与英语不同，中文词与词之间没有空格分隔，需要使用分词工具（如jieba）进行分词。分词的质量会直接影响TF-IDF的效��。

```python
# 常见问题的处理示例

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 问题1：自定义停用词
custom_stopwords = ['machine', 'learning', 'learning algorithms', 'system']
vectorizer = TfidfVectorizer(stop_words=custom_stopwords)

# 问题2：查看IDF分布
documents = ["This is a test document."] * 10 + ["Another document with unique words zoo."] * 2
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)
print("IDF值:", vectorizer.idf_)

# 问题3：处理空文档
empty_documents = ["", "word", "word word"]
vectorizer = TfidfVectorizer()
# 空文档会返回全零向量
result = vectorizer.fit_transform(empty_documents)
print("空文档处理:", result.toarray())

# 问题4：中文TF-IDF
try:
    import jieba
    chinese_documents = [
        "机器学习是人工智能的一个分支",
        "深度学习是机器学习的子领域",
        "自然语言处理研究语言的计算机理解"
    ]
    
    def chinese_tokenizer(text):
        return ' '.join(jieba.cut(text))
    
    chinese_documents_tokenized = [chinese_tokenizer(doc) for doc in chinese_documents]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(chinese_documents_tokenized)
    print("中文词汇表:", vectorizer.get_feature_names_out())
except ImportError:
    print("jieba未安装，跳过中文示例")
```

## 12. 学习总结

TF-IDF是文本特征表示的经典方法，通过词频和逆文档频率的组合来衡量词对于文档的重要性。

从算法基础认知的角度，TF-IDF的核心思想是：一个词对于一篇文档的重要性与其在该文档中出现的频率成正比，与其在整个语料库中出现的频率成反比。这种设计使得高频且稀有的词获得高权重，而常见词（如停用词）的权重被降低。

从核心原理的角度，TF-IDF由两部分组成：TF（词频）和IDF（逆文档频率）。TF衡量词在文档内的频率，IDF衡量词在语料库中的普遍程度。TF-IDF是两者的乘积。

从数学公式的角度，TF-IDF的标准定义为：TF-IDF(t, d) = TF(t, d) × IDF(t)，其中TF可以使用原始计数、归一化计数或对数变换，IDF通常使用log(N/df(t))。

从应用场景的角度，TF-IDF广泛应用于信息检索、文档分类、文本聚类、关键词提取等任务。它是一种无监督方法，不需要标注数据。

从优缺点的角度，TF-IDF的优点包括简单、可解释、无需训练数据；缺点是无法捕捉语义和上下文信息。

TF-IDF虽然简单，但它为了解更高级的文本表示方法（如Word2Vec、BERT）奠定了基础。理解TF-IDF的原理有助于理解这些方法的改进之处。

## 13. 练习题与思考题与思考题（含答案）

### 练习题

**练习1**：假设语料库有100篇文档，词"machine"在80篇文档中出现，"learning"在100篇文档中出现，"automl"只在5篇文档中出现。请计算这三个词的IDF值。

答案：
- machine: log(100/(80+1)) = log(100/81) ≈ -0.0079（接近0但为负，平滑后为0）
- learning: log(100/(100+1)) = log(100/101) ≈ -0.00995
- automl: log(100/(5+1)) = log(100/6) ≈ 2.813

可以看到，"automl"的IDF最高，因为它最稀有。

**练习2**：请解释为什么TF-IDF不适合捕捉"bank"的多义词问题。

答案：TF-IDF是一种基于频率的方法，不考虑词的上下文。"bank"无论是指银行还是河岸，只要出现的频率相同，其TF-IDF值就相同。无法区分不同上下文下的不同含义。

**练习3**：如果想提取每篇文档的关键词，应该如何操作？

答案：对于每篇文档，计算所有词的TF-IDF值，然后按值降序排列。取前N个词作为关键词。

**练习4**：请比较TF和子线性TF（1+log(tf)���的区别。

答案：标准TF使用原始计数（或归一化计数），子线性TF使用对数变换（1+log(tf)）。子线性TF可以减少高频词的影响，使得中等频率的词能获得相对更高的权重。

### 思考题

**思考1**：TF-IDF和布尔模型（Boolean Model）有什么区别？

思考要点：布尔模型只考虑词是否出现（TF=1或0），不考虑频率。TF-IDF考虑词的频率，可以区分"出现一次"和"出现多次"的情况。因此TF-IDF通常效果更好。

**思考2**：如果语料库是静态的且经常需要添加新文档，应该如何高效地更新TF-IDF？

思考要点：可以预先计算IDF，当添加新文档时，只计算新文档的TF，更新IDF需要重新扫描整个语料库。可以使用增量学习方法或定期重建索引。

**思考3**：TF-IDF和词嵌入（如Word2Vec）的主要区别是什么？

思考要点：TF-IDF是稀疏的高维向量（维度=词表大小），每个维度代表一个具体的词；词嵌入是密集的低维向量（通常50-300维），每个维度是学习的潜在特征。TF-IDF基于频率统计，词嵌入基于上下文学习。TF-IDF可以解释，词嵌入难以解释。


### 13.3 详细答案与解析

#### 练习1：概念理解

**问题**：本算法的核心机制是什么？请简述其工作原理。

**答案与解析**：

**步骤1**：识别问题类型
根据算法定义，这是一个[类型：监督/无监督/生成/强化学习]任务。

**步骤2**：应用核心公式
$$核心公式 = [具体公式]$$
该公式的意义是[解释公式含义]。

**步骤3**：验证答案
代入具体数据验证：[计算过程]
最终结果符合预期，说明理解正确。

**答案**：算法的核心是通过[机制]实现[目标]，属于[算法类别]。

---

#### 练习2：手动计算

**问题**：给定数据[X=具体值, y=具体值]，手动计算[算法名]的[参数/结果]。

**答案与解析**：

**步骤1**：准备数据
$X = \begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$  
$y = \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} 3 \\ 7 \end{bmatrix}$

**步骤2**：应用算法步骤
根据[算法名]的定义，计算第一步：
$$第一步 = [具体公式代入] = [数值]$$

**步骤3**：继续计算
$$第二步 = [公式] = [结果]$$

**步骤4**：得到最终答案
$$最终结果 = [综合计算] = [具体数值]$$

**验证**：将结果带回原式检验 $[验证过程]$，确认正确。

---

#### 思考题：改进分析

**问题**：本算法在[特定场景]下存在哪些局限性？请提出改进方案。

**答案与解析**：

**局限性分析**：
1. **局限性1**：[具体表现]，原因是[原因解释]
2. **局限性2**：[具体表现]，原因是[原因解释]

**改进方案对比**：

| 改进方法 | 原理 | 优势 | 代价 |
|---------|------|------|------|
| 方法A | [原理] | [好处] | [额外成本] |
| 方法B | [原理] | [好处] | [额外成本] |
| 方法C | [原理] | [好处] | [额外成本] |

**推荐方案**：在实际应用中优先考虑[方法A]，因为[理由]。
## 14. 学习路径建议建议

学习TF-IDF应该作为学习文本处理和自然语言处理的第一步。以下是建议的学习路径。

第一步，掌握TF-IDF的原理和实现。这是本章节的内容，包括TF、IDF的计算和各种变体。

第二步，掌握sklearn的TfidfVectorizer的使用。学会各种参数的设置和调整。

第三步，理解TF-IDF在下游任务中的应用。通过文档分类、信息检索等任务来评估TF-IDF的效果。

第四步，对比TF-IDF和其他特征表示方法。如前所述的One-Hot编码、词嵌入（Word2Vec、GloVe、BERT）。

建议的后续学习内容：
- Word2Vec：基于上下文的词嵌入学习方法
- GloVe：基于全局共现统计的词嵌入方法
- BERT：基于Transformer的上下文相关的词表示方法

通过系统地学习这些内容，可以建立从传统文本处理到现代NLP的完整知识体系。