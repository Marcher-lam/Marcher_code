# TF-IDF 学习文档

> 词频-逆文档频率，信息检索与文本挖掘中的经典算法

---

## 1. 算法基础认知

### 1.1 一句话定义

TF-IDF（Term Frequency - Inverse Document Frequency，词频-逆文档频率）是一种用于评估词语在文档集中重要性的统计方法，核心思想是：一个词在某文档中出现次数多（TF高），但在整个文档集中出现次数少（IDF高），则该词对该文档越重要。

### 1.2 直觉类比

想象你在图书馆找书时，如果某本书的某个词只在这本书中出现很少几次，但在几乎所有书中都出现，那这个词不能帮你区分这本书。反之，如果一个词只在这本书中出现很多次，却在其他书中很少出现，那这个词能帮你精准定位这本书。TF-IDF就是这个"帮你定位重要词汇"的量化指标。搜索引擎用它来排序结果，垃圾邮件检测用它来识别特征词，学术论文分析用它来提取关键词。

### 1.3 历史背景

TF-IDF由美国康奈尔大学的Karen Spärck Jones于1972年首次提出，当时称为"特异性"（specificity）概念。1983年，Gerard Salton和McGill在现代信息检索教材中正式命名为TF-IDF并推广使用。此后30多年，TF-IDF成为信息检索领域的基石算法，几乎所有搜索引擎和文本处理系统都以其为基础。现代变体包括Okapi BM25、TF-IWF等。2006年Google发表PageRank算法后，常与TF-IDF结合使用。

### 1.4 算法定位

| 特性 | 说明 |
|------|------|
| 类型 | 有监督特征加权 / 无监督关键词提取 |
| 输出 | 词语权重向量 / 排序列表 |
| 模型类型 | 词袋模型（Bag of Words） |
| 时间复杂度 | O(N×M)，N为文档数，M为词汇数 |

### 1.5 前置知识

- [必备]：概率基础（对数、乘法原理）
- [必备]：文本预处理（分词、去停用词）
- [扩展]：向量空间模型（VSM）
- [扩展]：BM25算法

---

## 2. 核心原理

### 2.1 核心思想

TF-IDF的核心思想是**区分性加权**：高频词未必重要，常见词必定不重要。只有同时满足"在本文档中频繁出现"和"在总体文档中稀有"两个条件，才是重要词汇。数学上体现为TF和IDF的乘积关系。

### 2.2 工作流程

```
原始文档 → 分词 → 去停用词 → 统计词频 → 计算TF → 计算IDF → 计算TF-IDF → 排序输出
```

### 2.3 关键概念解释

- **Term Frequency (TF)**：词频，指词语在当前文档中出现的次数。原始定义为该词在文档中出现次数，但实际使用时常用对数平滑：$TF(t,d) = 1 + \log(freq(t,d))$。

- **Document Frequency (DF)**：文档频率，指包含该词的文档数量。DF越高说明该词越常见，区分度越低。

- **Inverse Document Frequency (IDF)**：逆文档频率，$IDF(t) = \log(N / DF(t))$。IDF越高说明该词越稀有，区分度越高。

- **TF-IDF Weight**：综合权重，$TF-IDF(t,d) = TF(t,d) \times IDF(t)$。

### 2.4 几何/直观解释

将每个文档表示为高维空间中的一个向量，维度数等于词汇量。TF-IDF就是每个维度上的坐标值。相似文档在这个空间中距离较近。可以理解为：为每个词分配一个"区分度坐标"，文档由其词汇的区分度坐标构成。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $t$ | 词语/词项 | 标量 |
| $d$ | 单个文档 | 标量 |
| $D$ | 文档集合 | N×1 |
| $V$ | 词汇表 | M×1 |
| $N$ | 文档总数 | 标量 |
| $M$ | 词汇表大小 | 标量 |
| $f_{t,d}$ | 词t在文档d中出现次数 | 标量 |
| $DF_t$ | 词语t的文档频率 | 标量 |
| $TF(t,d)$ | 词语t在文档d中的词频 | 标量 |
| $IDF(t)$ | 词语t的逆文档频率 | 标量 |
| $w_{t,d}$ | 词语t在文档d中的TF-IDF权重 | 标量 |

### 3.2 问题形式化

给定文档集合D和词汇表V，为每个词项t计算权重：

$$w_{t,d} = TF(t,d) \times IDF(t)$$

其中：
- 词频：$TF(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$
- 逆文档频率：$IDF(t) = \log\frac{N}{DF_t + 1} + 1$（加1防零）

### 3.3 目标函数/损失函数

TF-IDF本身不是机器学习模型，无目标函数。其实质是特征权重分配。但可从信息论角度理解：目标是最大化区分信息。

从贝叶斯角度，假设词语独立，文档概率：
$$P(d) = \prod_{t \in d} P(t)^{f_{t,d}}$$

取对数后：
$$\log P(d) = \sum_{t \in d} f_{t,d} \log P(t)$$

TF-IDF可理解为：$\log P(t)$的负值（加上归一化常数）。

### 3.4 推导过程

**步骤1：词频的定义**

词频最直观定义是原始计数：
$$TF_{raw}(t,d) = f_{t,d}$$

但原始计数对不同长度文档不公平，于是做归一化：
$$TF(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

实践中还常用对数平滑：
$$TF(t,d) = 1 + \log(f_{t,d})$$

**步骤2：逆文档频率的定义**

文档频率越高，区分度越低。定义：
$$DF(t) = |\{d \in D : t \in d\}$$

逆文档频率应为DF的单调递减函数。常用对数：
$$IDF(t) = \log\frac{N}{DF(t)}$$

为避免除零，加平滑项：
$$IDF(t) = \log\left(\frac{N}{DF(t) + 1}\right) + 1$$

**步骤3：TF-IDF的组合**

组合原则：TF衡量本地重要性，IDF衡量全局稀有性。直接相乘：
$$TF-IDF(t,d) = TF(t,d) \times IDF(t)$$

展开：
$$= \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}} \times \log\left(\frac{N}{DF(t) + 1}\right)$$

### 3.5 最终解/算法步骤

**TF-IDF计算算法**：

```
输入：文档集合 D = {d1, d2, ..., dN}
输出：每个文档的TF-IDF向量

1. 建立词汇表 V
   for each d in D:
       分词得到词列表
       去停用词
       加入V（去重）

2. 计算DF
   for each t in V:
       DF[t] = 统计包含t的文档数

3. 计算IDF
   for each t in V:
       IDF[t] = log(N / (DF[t] + 1))

4. 计算TF-IDF
   for each d in D:
       for each t in d:
           TF[t,d] = f[t,d] / sum(f)
           w[t,d] = TF[t,d] × IDF[t]
```

---

## 4. 训练过程讲解

TF-IDF不是机器学习模型，无"训练"过程。仅有"构建"过程。但从工程角度，可类比：

### 4.1 数据预处理

- **分词**：中文用jieba等工具，英文用NLTK的word_tokenize
- **小写化**：统一转为小写
- **去停用词**：如the, a, 的, 了等常见词
- **词形还原**：如running→run, dogs→dog
- **n-gram**：可加入bigram, trigram增加上下文

```python
# 预处理示例
import re
from collections import Counter

def preprocess(text):
    # 小写化
    text = text.lower()
    # 去除标点
    text = re.sub(r'[^\w\s]', '', text)
    # 分词（英文简单实现）
    words = text.split()
    # 去除停用词
    stopwords = {'the', 'a', 'an', 'is', 'are', 'of', 'in', 'to', 'and'}
    words = [w for w in words if w not in stopwords]
    return words
```

### 4.2 向量构建

每文档构建M维向量（词汇表大小）。常用稀疏矩阵存储，节省内存。

### 4.3 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| max_features | 词汇表大小 | 5000-50000 | 10000 |
| min_df | 最小文档频率 | 2-5 | 2 |
| max_df | 最大文档频率 | 0.7-0.95 | 0.95 |
| sublinear_tf | 对数TF | True/False | True |
| norm | 向量归一化 | 'l1'/'l2'/None | 'l2' |

---

## 5. 应用场景

### 5.1 典型应用

**搜索引擎结���排序**：Google早期使用TF-IDF对查询和文档匹配度打分。

**文本分类特征提取**：作为朴素贝叶斯、SVM等分类器的输入特征。

**关键词自动提取**：取TF-IDF最高的k个词作为文档关键词。

**相似文档检索**：用余弦相似度计算TF-IDF向量距离。

**垃圾邮件检测**：TF-IDF高的词往往是邮件特征词。

### 5.2 适用数据特征

- **大规模语料**：TF-IDF对语料规模敏感，越多越准确
- **同领域文档**：领域相关词汇区分度更高
- **短文本**：如标题、摘要的关键词提取
- **实时性要求高**：计算简单，快速响应

### 5.3 不适用场景

- **语义理解**：不考虑词序和上下文
- **新词语**：需要更新词汇表
- **多语言**：需针对每种语言预处理
- **极短文档**：如单句，无区分度

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 计算简单 | 复杂度低，易实现 | 文档数量不太大 |
| 可解释性强 | 权重直观，符合直觉 | 无需复杂数学 |
|效果好 | 在基础任务上表现优秀 | 检索/分类 |
| 无需标注 | 无监督，适用于所有文档 | 语料足够 |
| 速度快 | 实时计算，内存占用小 | 稀疏表示 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 忽略词序 | "不好+好"与"好+不好"相同 | 加入n-gram |
| 忽略语义 | 同义词无法识别 | WordNet扩展 |
| 长文档偏差 |  长文档TF高但可能无关 | 归一化 |
| 停用词依赖 | 停用词表影响大 | 动态停用词 |

### 6.3 与同类算法对比

| 算法 | 复杂度 | 效果 | 特点 |
|------|--------|------|------|
| TF-IDF | O(N×M) | baseline | 简单高效 |
| BM25 | O(N×M) | +10-20% | 文档长度归一化 |
| LSA | O(N×M²) | +15-30% | 语义降维 |
| Word2Vec | O(N) | +20-40% | 语义表示 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

```python
"""
TF-IDF 调库实现 - 使用scikit-learn
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

class TFIDFExtractor:
    """
    TF-IDF特征提取器
    
    使用scikit-learn的TfidfVectorizer，支持中文分词和多种配置
    """
    
    def __init__(self, max_features=10000, min_df=2, max_df=0.95, 
                 sublinear_tf=True, norm='l2'):
        """
        初始化
        
        参数:
            max_features: 最大特征数
            min_df: 最小文档频率
            max_df: 最大文档频率 
            sublinear_tf: 使用对数TF
            norm: 归一化方式
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm=norm,
            use_idf=True,
            smooth_idf=True
        )
        
    def fit_transform(self, documents):
        """
        拟合并转换
        
        参数:
            documents: 文档列表（字符串列表）
            
        返回:
            TF-IDF矩阵: scipy sparse matrix (N, M)
        """
        return self.vectorizer.fit_transform(documents)
    
    def transform(self, documents):
        """
        仅转换（用于新文档）
        
        参数:
            documents: 文档列表
            
        返回:
            TF-IDF矩阵
        """
        return self.vectorizer.transform(documents)
    
    def get_feature_names(self):
        """获取特征名称（词汇表）"""
        return self.vectorizer.get_feature_names_out()
    
    def get_top_keywords(self, doc_idx, top_k=10):
        """
        获取某文档的top关键词
        
        参数:
            doc_idx: 文档索引
            top_k: 返回前k个
            
        返回:
            [(词, 权重), ...]
        """
        # 获取该文档的向量
        doc_vector = self.fit_transform([])[doc_idx]
        
        # 获取非零索引和值
        indices = doc_vector.nonzero()[1]
        values = doc_vector.toarray()[0, indices]
        
        # 排序
        top_indices = indices[np.argsort(values)[-top_k:]]
        top_values = values[top_indices]
        
        feature_names = self.get_feature_names()
        return [(feature_names[i], v) for i, v in zip(top_indices, top_values)]


class ChineseTFIDFExtractor(TFIDFExtractor):
    """
    中文TF-IDF提取器
    使用jieba分词
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 替换Tokenizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._chinese_tokenizer,
            **kwargs
        )
    
    def _chinese_tokenizer(self, text):
        """中文分词器"""
        return list(jieba.cut(text))


def demo_tfidf():
    """演示TF-IDF"""
    
    # 示例文档
    documents = [
        "机器学习是人工智能的一个分支",
        "深度学习是机器学习的一个分支",
        "自然语言处理是人工智能的应用领域",
        "计算机视觉是人工智能的应用领域",
        "机器学习用于数据分析"
    ]
    
    print("=" * 50)
    print("TF-IDF 示例演示")
    print("=" * 50)
    
    # 创建提取器
    extractor = ChineseTFIDFExtractor(
        max_features=100,
        min_df=1
    )
    
    # 拟合并转换
    tfidf_matrix = extractor.fit_transform(documents)
    
    print(f"\n文档数: {tfidf_matrix.shape[0]}")
    print(f"特征数: {tfidf_matrix.shape[1]}")
    print(f"词汇表: {extractor.get_feature_names()}")
    
    # 输出每个文档的TF-IDF
    print("\n各文档TF-IDF向量:")
    for i, doc in enumerate(documents):
        print(f"\n文档{i}: {doc}")
        top_kw = extractor.get_top_keywords(i, top_k=3)
        for kw, w in top_kw:
            print(f"  {kw}: {w:.4f}")
    
    # 相似度计算
    print("\n文档相似度矩阵:")
    sim_matrix = cosine_similarity(tfidf_matrix)
    print(np.round(sim_matrix, 3))


if __name__ == "__main__":
    demo_tfidf()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

```python
"""
TF-IDF 手工实现 - 核心算法
"""

import math
import re
from collections import Counter

class ManualTFIDF:
    """
    手工实现的TF-IDF算法
    
    只依赖标准库，不使用sklearn
    """
    
    def __init__(self):
        self.vocabulary = []
        self.idf = {}
        self.num_docs = 0
        
    def _tokenize(self, text):
        """
        分词 - 简单英文实现
        
        参数:
            text: 输入文本
            
        返回:
            词列表
        """
        # 转小写
        text = text.lower()
        # 去除标点
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分词
        words = text.split()
        # 去除空词
        words = [w for w in words if w]
        return words
    
    def _compute_tf(self, words):
        """
        计算词频
        
        参数:
            words: 词列表
            
        返回:
            词频字典
        """
        counter = Counter(words)
        total = len(words)
        # 归一化TF
        tf = {word: count / total for word, count in counter.items()}
        return tf
    
    def _compute_df(self, docs_tokens):
        """
        计算文档频率
        
        参数:
            docs_tokens: 所有文档的分词列表
            
        返回:
            词DF字典
        """
        df = Counter()
        for tokens in docs_tokens:
            # 每个词只计一次
            for word in set(tokens):
                df[word] += 1
        return df
    
    def _compute_idf(self, df):
        """
        计算IDF
        
        参数:
            df: 文档频率
            
        返回:
            IDF字典
        """
        idf = {}
        N = self.num_docs
        for word, freq in df.items():
            # 加1平滑
            idf[word] = math.log((N + 1) / (freq + 1)) + 1
        return idf
    
    def fit(self, documents):
        """
        拟合文档集合
        
        参数:
            documents: 文档列表
        """
        self.num_docs = len(documents)
        
        # 分词
        docs_tokens = [self._tokenize(doc) for doc in documents]
        
        # 构建词汇表
        self.vocabulary = list(set(word for tokens in docs_tokens for word in tokens))
        
        # 计算DF
        df = self._compute_df(docs_tokens)
        
        # 计算IDF
        self.idf = self._compute_idf(df)
        
    def transform(self, document):
        """
        转换单个文档
        
        参数:
            document: 文档字符串
            
        返回:
            TF-IDF向量字典 {(词, 权重), ...}
        """
        # 分词
        tokens = self._tokenize(document)
        
        # 计算TF
        tf = self._compute_tf(tokens)
        
        # 计算TF-IDF
        tfidf = {}
        for word, tf_val in tf.items():
            if word in self.idf:
                tfidf[word] = tf_val * self.idf[word]
        
        return tfidf
    
    def fit_transform(self, documents):
        """
        拟合并转换
        
        参数:
            documents: 文档列表
            
        返回:
            TF-IDF向量列表
        """
        self.fit(documents)
        return [self.transform(doc) for doc in documents]
    
    def get_top_keywords(self, document, top_k=5):
        """
        获取top关键词
        
        参数:
            document: 文档
            top_k: 返回数量
            
        返回:
            [(词, 权重), ...]
        """
        tfidf = self.transform(document)
        sorted_items = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]


def demo_manual():
    """手工实现演示"""
    
    documents = [
        "machine learning is a subfield of artificial intelligence",
        "deep learning is a subfield of machine learning",
        "natural language processing is an application of AI",
        "computer vision is an application of AI",
        "machine learning is used for data analysis"
    ]
    
    print("=" * 50)
    print("TF-IDF 手工实现演示")
    print("=" * 50)
    
    tfidf = ManualTFIDF()
    results = tfidf.fit_transform(documents)
    
    print(f"\n文档数: {tfidf.num_docs}")
    print(f"词汇表大小: {len(tfidf.vocabulary)}")
    
    # 输出IDF值
    print("\n各词IDF值:")
    for word, idf_val in sorted(tfidf.idf.items(), key=lambda x: x[1], reverse=True):
        print(f"  {word}: {idf_val:.4f}")
    
    # 输出每个文档的top词
    print("\n各���档top关键词:")
    for i, doc in enumerate(documents):
        top_kw = tfidf.get_top_keywords(doc, top_k=3)
        print(f"\n文档{i}: {doc}")
        for kw, w in top_kw:
            print(f"  {kw}: {w:.4f}")


if __name__ == "__main__":
    demo_manual()
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def visualize_tfidf():
    """TF-IDF可视化"""
    
    # 示例数据
    documents = [
        "machine learning is great",
        "deep learning is better",
        "natural language processing",
        "computer vision",
        "artificial intelligence"
    ]
    
    # 计算TF-IDF
    extractor = ManualTFIDF()
    results = extractor.fit_transform(documents)
    
    # 提取权重
    words = list(extractor.idf.keys())
    idf_values = [extidf.idf[w] for w in words]
    
    # 可视化IDF分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：IDF柱状图
    axes[0].barh(words, idf_values, color='steelblue')
    axes[0].set_xlabel('IDF Value')
    axes[0].set_title('Word IDF Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # 右图：模拟文档向量
    # 简化：只用前几个词
    sample_words = words[:5]
    sample_vectors = []
    for doc in documents:
        tfidf = extractor.transform(doc)
        vec = [tfidf.get(w, 0) for w in sample_words]
        sample_vectors.append(vec)
    
    sample_vectors = np.array(sample_vectors)
    axes[1].imshow(sample_vectors, cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(len(sample_words)))
    axes[1].set_xticklabels(sample_words, rotation=45)
    axes[1].set_yticks(range(len(documents)))
    axes[1].set_yticklabels([f'doc{i}' for i in range(len(documents))])
    axes[1].set_title('TF-IDF Matrix')
    
    plt.tight_layout()
    plt.savefig('tfidf_visualization.png', dpi=150)
    plt.show()


def plot_word_cloud():
    """词云可视化（模拟）"""
    import matplotlib.pyplot as plt
    
    # IDF值（模拟）
    words = ['learning', 'machine', 'deep', 'neural', 'network', 
            'vision', 'language', 'natural', 'computer', 'artificial']
    idf_vals = [3.2, 2.8, 2.5, 2.2, 1.9, 
                1.8, 1.6, 1.5, 1.2, 1.0]
    
    # 按大小排序
    sorted_idx = np.argsort(idf_vals)[::-1]
    sorted_words = [words[i] for i in sorted_idx]
    sorted_vals = [idf_vals[i] for i in sorted_idx]
    
    # 柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_words)), sorted_vals, color='coral')
    plt.xticks(range(len(sorted_words)), sorted_words, rotation=45)
    plt.ylabel('IDF Value')
    plt.title('Word Importance (IDF)')
    plt.tight_layout()
    plt.savefig('word_importance.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    visualize_tfidf()
    plot_word_cloud()
```

**结果解读**：

1. IDF分布分析：越高越稀有，越有区分度。in/the等词IDF接近0。
2. TF-IDF矩阵：稀疏矩阵，大部分为0，少量高权重值。
3. 文档相似度：对角线为1，其他值体现相似程度。

---

## 10. 模型评估

### 10.1 评估指标

TF-IDF本身不做预测，无需评估。但作为特征：

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| 召回率 | 相关文档检出率 | 搜索 |
| 精确率 | 检出相关率 | 搜索 |
| F1 | 综合评估 | 搜索 |
| 分类准确率 | 分类正确率 | 分类 |
| 余弦相似度 | 向量距离 | 聚类/检索 |

### 10.2 代码示例

```python
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例文档和标签
documents = ["machine learning", "deep learning", "natural language"]
labels = [0, 1, 1]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print(f"TF-IDF矩阵形状: {X.shape}")
print(f"特征数: {len(vectorizer.get_feature_names_out())}")
```

---

## 11. 常见问题与易错点

### 11.1 问题1：长文档偏差

**原因**：长文档词汇多，TF计算时每个词权重被稀释。

**解决方案**：使用BM25或对数TF归一化。

```python
# sklearn sublinear_tf
vectorizer = TfidfVectorizer(sublinear_tf=True)
# 效果: 1 + log(tf) 而非直接tf
```

### 11.2 问题2：新文档包含未登录词

**原因**：词汇表只包含训练文档的词。

**解决方案**：设置max_df过滤常见词，或增加n-gram。

```python
# 增大max_features
vectorizer = TfidfVectorizer(max_features=50000)
```

### 11.3 问题3：中文分词效果差

**原因**：英文按空格分词，中文需专门分词器。

**解决方案**：使用jieba分词。

```python
import jieba

def chinese_tokenizer(text):
    return list(jieba.cut(text))

vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
```

---

## 12. 学习总结

### 核心要点回顾：

1. **TF = 本地频率**：词在当前文档中出现次数，越高说明在本文档中越重要
2. **IDF = 全局稀有度**：词在所有文档中出现文档数越少，IDF越高，区分度越高
3. **TF-IDF = 乘积**：综合本地重要性和全局区分度，两个条件都满足才重要
4. **无语义理解**：只考虑词频，不理解上下文和同义词

### 从TF-IDF到其他算法：

- TF-IDF → BM25（加入文档长度归一化）
- TF-IDF → LSA（加入语义降维）
- TF-IDF → word2vec（加入语义表示）
- TF-IDF → BERT（深度语义表示）

### 实践建议：

1. 默认用sklearn的TfidfVectorizer，默认参数够用
2. 记得设置sublinear_tf=True，对数TF效果更好
3. 配合去停用词、去标点等预处理
4. 中文记得用jieba分词
5. 极少见词可设min_df过滤

---

## 13. 练习题与思考题（含答案）

### 练习题

**习题1：基础计算**

问题：给定两个文档D1="cat cat dog", D2="cat bird"，计算"cat"的IDF值。

<details>
<summary>答案</summary>

文档数N=2，包含"cat"的文档数DF=2。

IDF = log((2+1)/(2+1)) + 1 = log(1) + 1 = 0 + 1 = 1

所以cat的IDF=1。

</details>

**习题2：编程实践**

问题：用Python手动实现TF-IDF计算。

<details>
<summary>答案</summary>

```python
import math
from collections import Counter

def compute_tfidf(documents):
    # 1. 分词（简化）
    doc_tokens = [doc.split() for doc in documents]
    
    # 2. 计算DF
    df = Counter()
    for tokens in doc_tokens:
        for word in set(tokens):
            df[word] += 1
    
    # 3. 计算IDF
    N = len(documents)
    idf = {word: math.log((N+1)/(df[word]+1)) + 1 for word in df}
    
    # 4. 计算TF-IDF
    results = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        tfidf = {word: (1 + math.log(cnt)) * idf[word] for word, cnt in tf.items()}
        results.append(tfidf)
    
    return results

# 测试
docs = ["machine learning", "deep learning"]
print(compute_tfidf(docs))
```

</details>

**习题3：理论推导**

问题：推导IDF取对数的原因。

<details>
<summary>答案</summary>

1. 直观原因：词频差异大。如DF从1到10000，用对数后差异从0到9.2，缓和。
2. 信息论：IDF = log(N/DF)。N/DF是逆文档概率，取对数后是惊奇度/信息量。
3. 乘法变加法：对数把乘除变加减，便于计算。
4. 马太效应：常用词IDF不会变成0或负数。

</details>

### 思考题

**思考题1**：TF-IDF有哪些改进方向？

<details>
<summary>答案</summary>

1. BM25：加入文档长度归一化
2. TF-IWF：使用逆词频权重
3. 语义扩展：结合WordNet处理同义词
4. 位置加权：标题、开头词权重更高
5. 词性加权：名词权重高于动词
6. 深度学习：用BERT等替代词袋模型

</details>

**思考题2**：TF-IDF在搜索引擎中如何工作？

<details>
<summary>答案</summary>

1. 索引建立：为每个文档计算TF-IDF向量，存入倒排索引
2. 查询处理：将查询也转为TF-IDF向量
3. 相似度计算：计算查询与候选文档的余弦相似度
4. 排序输出：按相似度降序排列

实际系统中还会结合PageRank、点击率等信号。

</details>

---

## 14. 学习路径建议

### 初级阶段（掌握TF-IDF基础）

1. 理解TF和IDF的概念
2. 掌握核心公式
3. 手动计算简单例子
4. 使用sklearn实现

**学习时间**：1-2天

### 中级阶段（理解原理和扩展）

1. 理解对数平滑的原因
2. 学习BM25算法
3. 理解TF-IDF的局限性
4. 实践文本分类应用

**学习时间**：1周

### 高级阶段（扩展到其他算法）

1. 学习LSA/LDA降维
2. 学习Word2Vec语义表示
3. 学习BERT等深度学习方法
4. 研究信息检索最新进展

**学习时间**：2-3周

### 实践项目建议

1. **基础项目**：实现简易搜索引擎
2. **进阶项目**：新闻分类系统
3. **挑战项目**：对话系统关键词提取

### 推荐资源

- **书籍**：《信息检索导论》- Manning
- **课程**：Stanford CS276 信息检索
- **论文**：TF-IDF原始论文（Spärck Jones 1972）
- **代码**：sklearn TfidfVectorizer文档

---

**文档结束**