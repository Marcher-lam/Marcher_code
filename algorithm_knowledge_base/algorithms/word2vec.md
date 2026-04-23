# Word2Vec 学习文档

## 1. 算法基础认知

Word2Vec是2013年由Google研究团队（Tomas Mikolov等人）提出的一种词嵌入（Word Embedding）学习方法。它的核心思想是将高维稀疏的One-Hot词向量映射为低维密集的连续向量，使得语义相似的词在向量空间中距离更近。

在Word2Vec出现之前，词的表示主要依靠TF-IDF等基于频率的方法或简单的词袋模型（Bag of Words）。这些方法的局限在于：维度非常高（等于词表大小）、无法表达词与词之间的语义相似性、同义词和多义词无法区分。例如，"king"和"queen"在One-Hot空间中是完全正交的，无法知道它们都代表皇室成员；而"car"和"automobile"是同义词，但在向量空间中可能相距甚远。

Word2Vec通过学习词的分布式表示（Distributed Representation）解决了这个问题。分布式表示的核心思想是：一个词的意义可以从它的上下文中推断出来。"bank"在"river bank"中指的是河岸，在"investment bank"中指的是银行。Word2Vec通过训练神经网络，让模型学习每个词的上下文，最终得到的词向量能够捕捉语义相似性。

Word2Vec的另一个重要贡献是提出了"分布假设"（Distributional Hypothesis）："You shall know a word by the company it keeps"（观其友，知其意）。这个假设来源于语言学，意思是语义相似的词倾向于出现在相似的上下文中。Word2Vec正是基于这个假设设计的。

Word2Vec有两种训练模型：CBOW（Continuous Bag of Words）和Skip-gram。CBOW利用上下文预测中心词，结构简单，训练速度较快，适合小型数据集；Skip-gram利用中心词预测上下文，精度更高，适合大型数据集。这两种模型的本质是对偶的，都可以学习到高质量的词向量。

## 2. 核心原理

Word2Vec的核心原理建立在神经网络语言模型的基础上，通过最大化词序列的似然概率来学习词向量。

对于一个句子序列w₁, w₂, ..., wₜ，语言模型的目标是计算整个序列的概率P(w₁, w₂, ..., wₜ)。根据链式法则，这个概率可以分解为：P(w₁, w₂, ..., wₜ) = P(w₁)·P(w₂|w₁)·P(w₃|w₁,w₂)·...·P(wₜ|w₁,...,wₜ₋₁)。

为了简化计算，Word2Vec做了一个关键假设：假设当前词只与它的上下文词相关，而与更远的词无关。这就是Word2Vec的条件独立假设。

对于Skip-gram模型，目标是最大化所有center-context对的平均对数概率：

J = (1/T)·Σₜ Σ₍₋ₙ≤c≤n, c≠0₎ log P(wₜ₊c | wₜ)

其中n是窗口大小（通常为5），表示考虑前后各n个词作为上下文。

对于CBOW模型，目标是最大化：

J = (1/T)·Σₜ log P(wₜ | wₜ₋ₙ, ..., wₜ₋₁, wₜ₊₁, ..., wₜ₊ₙ)

Softmax函数的计算是问题的核心。由于词表大小V可能达到数十万，计算整个Softmax需要O(V)的时间，这在实际应用中是不可接受的。Word2Vec提出了两种优化的Softmax计算方法：负采样（Negative Sampling）和层级Softmax（Hierarchical Softmax）。

## 3. 数学公式与推导

Word2Vec的数学推导从神经网络的表示开始。设词表大小为V，嵌入维度为d。对于一个输入词w，通过嵌入矩阵W ∈ ℝ^{V×d}得到其嵌入向量：

x = W[e_w]

其中e_w是词w的One-Hot向量，形状为V×1。

对于Skip-gram模型，给定中心词c，上下文词oₜ的预测概率为：

P(o | c) = softmax(v_c · vₒ') = exp(v_c · vₒ') / Σₜ exp(v_c · vₜ)

其中v_c是中心词的输出向量，vₒ'是上下文词的输入向量（使用两个独立的嵌入矩阵）。

直接计算Softmax的代价是遍历整个词表，时间复杂��为O(V)。为了提高效率，Word2Vec使用负采样来近似Softmax。

负采样的目标是：对于正样本(c, o)，最大化它们共现的概率；对于负样本(c, k)，最小化它们共现的概率。具体的损失函数为：

J(context) = -log σ(v_c · vₒ') - Σₖ log σ(-v_c · vₖ)

其中的σ是Sigmoid函数：σ(x) = 1/(1 + exp(-x))。

这个目标函数的意义是：正样本的对数概率越大越好（σ(v_c · vₒ')接近1），负样本的对数概率越小越好（σ(v_c · vₖ)接近0）。

负采样的数量k是一个超参数，通常设置为5-20，较小的词表可以设置更大的k。

层级Softmax是另一种优化方法，它使用二叉树来组织词表，每个词对应从根到叶子的路径。预测时不需要遍历整个词表，只需要沿着路径计算，时间复杂度为O(log V)。

除了负采样和层级Softmax，Word2Vec还使用了子采样（Subsampling）技术来加速训练。对于高频词（如"the"、"of"等），它们的学习价值较低，但出现的频率很高，会消耗大量的计算资源。子采样以概率1 - √(t/f_w)丢弃高频词，其中f_w是词w的频率，t是阈值（通常为10⁻⁵）。

## 4. 训练过程讲解

Word2Vec的训练过程实际上是一个神经网络的优化过程，但与传统的神经网络训练有一些关键区别。

训练数据的准备是第一步。对于一个文本语料库，需要先进行分词、去停用词等预处理。然后，对于Skip-gram模型，从每个中心词的窗口内提取(context, target)对；对于CBOW模型，从每个上下文的组合中提取训练样本。

具体的数据准备过程：假设语料库是一个句子集合，窗口大小为n=2。对于句子"the quick brown fox jumps over the lazy dog"，以"brown"为中心词时：
- Skip-gram: (brown, the), (brown, quick), (brown, fox), (brown, jumps)
- CBOW: (the, quick, fox, jumps) → brown

训练过程使用随机梯度下降（SGD）或小批量梯度下降。对于每个训练样本，计算损失函数的梯度，然后更新嵌入矩阵。

嵌入矩阵的维度选择是一个重要决策。一般的经验是：对于小型语料库（词表小于10000），嵌入维度可以设置为100-300；对于大型语料库，可以设置为300-500。嵌入维度过小可能无法捕捉足够的信息，嵌入维度过大可能导致过拟合。

训练过程中的超参数选择：
- 窗口大小n：通常为5-10，较小的窗口有利于学习语法相似性，较大的窗口有利于学习语义相似性
- 负采样数量k：通常为5-20
- 学习率：初始学习率通常为0.025，随着训练逐渐衰减
- 嵌入维度d：通常为100-300
- 训练轮数：通常为3-10轮

训练完成后，嵌入矩阵W的每一行（或每一列，取决于实现）就是一个词的d维向量表示。相同语境下训练的两个词应该有相似的向量，这可以通过余弦相似度来验证。

训练完成的词向量可以通过几种方式评估：
- 语义类比任务：如"king - man + woman = queen"
- 句法类比任务：如"walking - walk + swim = swimming"
- 相似度任务：比较人工标注的相似度与向量的余弦相似度

## 5. 应用场景

Word2Vec学到的词向量在自然语言处理中有广泛的应用。

在信息检索系统中，Word2Vec可以将查询和文档表示为向量，通过计算余弦相似度来匹配语义相关的查询和文档。这种方法比基于关键词的匹配更加灵活，可以处理同义词和多义词的问题。

在文档分类中，Word2Vec生成的词向量可以作为分类器的输入特征。由于词向量捕捉了语义信息，分类器可以利用这些信息来识别文档的主题或情感。

在机器翻译中，Word2Vec可以用来构建多语言的词向量空间。通过对齐不同语言的向量空间，可以实现词典推断和翻译。

在命名实体识别中，词向��可以作为特征输入到CRF或LSTM等序列标注模型中。词向量的语义信息有助于识别实体的类型。

在情感分析中，Word2Vec可以将评论文本转换为向量，然后使用分类器进行情感判断。语义相近的评论（如"good"和"great"）会有相似的向量表示。

在推荐系统中，Word2Vec可以学习物品的向量表示。通过计算物品之间的相似度，可以为用户推荐相似的物品。

在文本聚类中，Word2Vec可以将文档表示为向量，然后使用K-Means等聚类算法将相似的文档归为一类。这种方法可以用于主题发现和文档组织。

在词义消歧中，同一词的不同向量可以表示不同的词义。例如，"bank"在"bank account"和"river bank"中应该有不同的向量表示（取决于上下文）。

## 6. 优缺点分析

Word2Vec作为一种经典的词嵌入方法，有其独特的优点和明显的缺点。

优点方面，首先Word2Vec能够捕捉词的语义相似性。通过训练，语义相似的词会有相似的向量表示，这使得基于向量相似度的应用成为可能。例如，"king"和"queen"的向量相似度较高，"car"和"automobile"也是相似的。

其次，Word2Vec生成的向量是低维密集的。相比One-Hot的V维稀疏向量，Word2Vec的d维（通常100-300）密集向量更加紧凑，存储和计算效率都更高。

第三，Word2Vec的训练是无监督的。它不需要标注数据，只需要原始文本语料库。这使得它可以充分利用大量的未标注文本。

第四，Word2Vec训练速度快。通过负采样和层级Softmax优化，即使在大型语料库上也能在合理的时间内完成训练。

第五，Word2Vec的词向量具有良好的线性性质。著名的类比运算"king - man + woman ≈ queen"展示了词向量之间的线性关系，这使得词向量可以用于推理。

缺点方面，首先Word2Vec是上下文无关的。同一词无论出现在什么上下文，都对应相同的向量。这无法解决多义词的问题，例如"bank"在"bank account"和"river bank"中有不同的含义，但Word2Vec只能给出一个向量。

第二，Word2Vec无法处理超出词表的词。对于训练语料中未出现的词，无法获得其向量表示。

第三，Word2Vec的训练是独立的。每个词的向量训练只依赖于其上下文，无法利用全局的统计信息。

第四，Word2Vec对低频词的效果不好。由于负采样的设计，低频词训练不足，向量质量较差。

第五，Word2Vec无法捕捉词序信息。Word2Vec的CBOW模型忽略了词的顺序，只利用了上下文的词袋信息。

这些缺点促进了更高级的词嵌入方法的发展，如GloVe（利用全局共现统计）和BERT（利用深层Transformer捕捉上下文）。

## 7. 调库实现（Gensim）

Gensim是一个专业的Python库，用于处理自然语言处理中的各种主题模型和词嵌入。Word2Vec是Gensim的核心功能之一。

```python
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import re

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 示例语料库（使用NLTK的语料）
raw_corpus = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
    "Deep learning is part of machine learning that uses neural networks with multiple layers.",
    "Natural language processing focuses on enabling computers to understand human language.",
    "Computer vision allows computers to identify and process images like the human visual system.",
    "Reinforcement learning is a type of machine learning where agents learn from rewards.",
    "Neural networks are computing systems inspired by biological neural networks.",
    "Artificial intelligence encompasses various techniques to mimic human intelligence.",
    "Data science combines statistics, programming, and domain expertise.",
    "Machine translation automatically translates text between languages.",
    "Sentiment analysis determines the emotional tone behind a series of words.",
    "Text classification assigns predefined categories to text documents.",
    "Information retrieval finds information satisfying user needs from large datasets.",
    "Speech recognition converts spoken words into written text.",
    "Object detection identifies objects within images using computer vision.",
    "Feature extraction selects relevant features for machine learning models."
]

# 文本预处理
def preprocess(text):
    """简单的文本预处理"""
    # 转小写
    text = text.lower()
    # 去除标点
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    tokens = text.split()
    # 去除停用词（简化版）
    stopwords = {'a', 'an', 'the', 'is', 'are', 'to', 'of', 'and', 'or', 'that', 'which', 'for', 'in', 'on'}
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return tokens

# 预处理语料库
corpus = [preprocess(doc) for doc in raw_corpus]

print("预处理后的语料库（第一条）：")
print(corpus[0])
print()

# 训练Word2Vec模型
# 参数说明：
# size: 嵌入维度
# window: 窗口大小
# min_count: 最小词频
# workers: 并行训练线程数
# sg: 1=Skip-gram, 0=CBOW
# negative: 负采样数量
# epochs: 训练轮数
model = Word2Vec(
    sentences=corpus,
    vector_size=100,       # 嵌入维度
    window=3,            # 窗口大小
    min_count=1,         # 最小词频
    workers=4,           # 并行线程
    sg=1,                # 使用Skip-gram
    negative=10,         # 负采样数量
    epochs=50            # 训练轮数
)

print("模型训练完成！")
print(f"词表大小: {len(model.wv.key_to_index)}")
print()

# 常见���作
print("=== 词向量操作 ===")
print()

# 获取词的向量
print("获取'machine'的向量:")
word_vec = model.wv['machine']
print(f"  向量形状: {word_vec.shape}")
print(f"  向量前5个元素: {word_vec[:5]}")
print()

# 计算两个词的相似度
print("计算'machine'和'learning'的相似度:")
sim = model.wv.similarity('machine', 'learning')
print(f"  相似度: {sim:.4f}")
print()

# 找出最相似的词
print("找出与'learning'最相似的词:")
similar = model.wv.most_similar('learning', topn=3)
for word, score in similar:
    print(f"  {word}: {score:.4f}")
print()

# 词向量类比（king - man + woman = queen）
print("词向量类比：king - man + woman = ?")
try:
    result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
    for word, score in result:
        print(f"  答案: {word}, 相似度: {score:.4f}")
except KeyError as e:
    print(f"  词表中没有该词: {e}")
print()

# 找出不合群的词
print("找出不合群的词: ['machine', 'learning', 'computer']")
try:
    odd = model.wv.doesnt_match(['machine', 'learning', 'computer'])
    print(f"  不合群: {odd}")
except:
    print("  词汇不足，无法进行该操作")
print()

# 保存和加载模型
model.save('word2vec.model')
loaded_model = Word2Vec.load('word2vec.model')
print("模型已保存和加载")

# 加载预训练的中文模型（可选）
# import gensim.downloader as api
# model = api.load('glove-wiki-gigaword-100')
```

Gensim还支持加载预训练的词向量，如GloVe和FastText的向量。这可以让你直接使用在大规模语料库上训练的高质量词向量。

## 8. 手工代码实现（NumPy）

使用NumPy可以手动实现Word2Vec的基本训练过程。这有助于理解其底层原理。

```python
import numpy as np
from collections import Counter
import random
import math

class SimpleWord2Vec:
    """简化的Word2Vec实现"""
    
    def __init__(self, vocab_size=100, embedding_dim=100, window_size=3, learning_rate=0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        
        # 初始化嵌入矩阵（使用较小的随机值）
        scale = 0.1 / math.sqrt(embedding_dim)
        self.W_input = np.random.uniform(-scale, scale, (vocab_size, embedding_dim))
        self.W_output = np.random.uniform(-scale, scale, (vocab_size, embedding_dim))
    
    def train(self, corpus, epochs=5, negative_samples=5):
        """训练Word2Vec模型"""
        print(f"开始训练，词汇表大小: {self.vocab_size}")
        
        for epoch in range(epochs):
            total_loss = 0
            for center_word, context_words in self.generate_training_pairs(corpus):
                center_idx = center_word
                
                # 正样本训练
                for context_word in context_words:
                    context_idx = context_word
                    loss = self.train_pair(center_idx, context_idx, negative_samples)
                    total_loss += loss
            
            avg_loss = total_loss / len(corpus)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        return self
    
    def generate_training_pairs(self, corpus):
        """生成中心词-上下文对"""
        for sentence in corpus:
            for i, center_word in enumerate(sentence):
                context = []
                # 获取窗口内的上下文词
                start = max(0, i - self.window_size)
                end = min(len(sentence), i + self.window_size + 1)
                
                for j in range(start, end):
                    if j != i:
                        context.append(sentence[j])
                
                if context:
                    yield center_word, context
    
    def train_pair(self, center_idx, context_idx, negative_samples):
        """训练一个正样本对"""
        # 正样本的Sigmoid
        pos_score = self.sigmoid(np.dot(self.W_input[center_idx], self.W_output[context_idx]))
        pos_loss = -np.log(pos_score + 1e-10)
        
        # 更新正样本的向量
        error = (1 - pos_score)
        grad_in = self.learning_rate * error * self.W_output[context_idx]
        grad_out = self.learning_rate * error * self.W_input[center_idx]
        
        self.W_input[center_idx] += grad_in
        self.W_output[context_idx] += grad_out
        
        # 负样本训练
        for _ in range(negative_samples):
            neg_idx = random.randint(0, self.vocab_size - 1)
            if neg_idx == context_idx:
                continue
            
            neg_score = self.sigmoid(np.dot(self.W_input[center_idx], self.W_output[neg_idx]))
            neg_loss = -np.log(1 - neg_score + 1e-10)
            
            # 更新负样本的向量（反向）
            error = -neg_score
            grad_out = self.learning_rate * error * self.W_input[center_idx]
            self.W_output[neg_idx] += grad_out
        
        return pos_loss + neg_loss
    
    def sigmoid(self, x):
        """Sigmoid函数"""
        return 1 / (1 + math.exp(-max(min(x, 20), -20)))
    
    def get_vector(self, word_idx):
        """获取词的向量"""
        return self.W_input[word_idx]
    
    def most_similar(self, word_idx, top_k=5):
        """找出最相似的词"""
        target_vec = self.W_input[word_idx]
        
        # 计算余弦相似度
        similarities = []
        for i in range(self.vocab_size):
            if i != word_idx:
                sim = self.cosine_similarity(target_vec, self.W_input[i])
                similarities.append((i, sim))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def cosine_similarity(self, vec1, vec2):
        """余弦相似度"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot / (norm1 * norm2)


# 简化版训练器（使用计数方法构建词表）
def build_vocab(corpus, min_count=1):
    """构建词表"""
    counter = Counter()
    for sentence in corpus:
        counter.update(sentence)
    
    # 过滤低频词
    vocab = {}
    for word, count in counter.items():
        if count >= min_count:
            vocab[word] = count
    
    # 创建词到索引的映射
    word_to_idx = {word: i for i, word in enumerate(sorted(vocab.keys()))}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    
    return word_to_idx, idx_to_word


# 测试代码
if __name__ == "__main__":
    # 简单语料库
    corpus = [
        ['machine', 'learning', 'is', 'a', 'method'],
        ['deep', 'learning', 'uses', 'neural', 'networks'],
        ['natural', 'language', 'processing', 'enables', 'computers'],
        ['computer', 'vision', 'identifies', 'objects'],
        ['machine', 'translation', 'translates', 'text']
    ]
    
    # 构建词表
    word_to_idx, idx_to_word = build_vocab(corpus, min_count=1)
    print("词表:", word_to_idx)
    print()
    
    # 转换为索引
    indexed_corpus = []
    for sentence in corpus:
        indexed_sentence = [word_to_idx[w] for w in sentence if w in word_to_idx]
        indexed_corpus.append(indexed_sentence)
    
    print("索引后的语料:")
    print(indexed_corpus)
    print()
    
    # 训练模型
    model = SimpleWord2Vec(
        vocab_size=len(word_to_idx),
        embedding_dim=50,
        window_size=2,
        learning_rate=0.01
    )
    model.train(indexed_corpus, epochs=10, negative_samples=3)
    print()
    
    # 测试相似词
    print("与'machine'最相似的词:")
    mostsimilar = model.most_similar(word_to_idx['machine'], top_n=3)
    for idx, sim in mostsimilar:
        print(f"  {idx_to_word[idx]}: {sim:.4f}")
```

运行结果会展示词向量的训练过程和相似词的查找结果。

## 9. 可视化与结果理解

Word2Vec的词向量可以通过可视化来理解。下面的代码展示了如何可视化和分析词向量。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 重新训练一个小的模型用于可视化
corpus = [
    "king queen royal family throne palace",
    "man woman person people gender",
    "boy girl child children play game",
    "apple orange fruit food eat",
    "car bus vehicle transport travel",
    "walking running jumping moving action",
    "swimming diving floating water ocean"
]

def preprocess(text):
    text = text.lower().split()
    return text

corpus = [preprocess(doc) for doc in corpus]

model = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=3,
    min_count=1,
    epochs=100
)

# 提取所有词向量
words = list(model.wv.key_to_index.keys())
vectors = np.array([model.wv[word] for word in words])

print(f"词表: {words}")
print(f"向量形状: {vectors.shape}")
print()

# 使用t-SNE降维
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words)-1))
vectors_2d = tsne.fit_transform(vectors)

# 创建可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1：t-SNE可视化
ax1 = axes[0]
ax1.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='steelblue', s=100)

for i, word in enumerate(words):
    ax1.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=12)

ax1.set_title('Word2Vec Visualized with t-SNE')
ax1.set_xlabel('Dimension 1')
ax1.set_ylabel('Dimension 2')

# 图2：相似度热力图
ax2 = axes[1]
n = len(words)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = model.wv.similarity(words[i], words[j])

im = ax2.imshow(similarity_matrix, cmap='coolwarm', aspect='auto')
ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(words, rotation=45, ha='right')
ax2.set_yticklabels(words)
ax2.set_title('Word Similarity Matrix')
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.savefig('word2vec_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n结果解释：")
print("1. t-SNE图展示了词在2D空间中的分布，语义相似的词应该聚在一起")
print("2. 热力图展示了词与词之间的相似度，对角线为1.0（与自身的相似度）")
print("3. 可以看到'king'和'queen'、'man'和'woman'等相似度较高")
```

运行后会生成可视化图表，帮助理解Word2Vec词向量的特性。

## 10. 模型评估

Word2Vec词向量的质量可以通过多种方式评估。

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from collections import Counter

# 使用更大的语料库
sentences = [
    "Machine learning is transforming technology and society",
    "Deep learning enables machines to learn from data",
    "Neural networks process information like human brains",
    "Artificial intelligence impacts every industry",
    "Data science uses statistical methods",
    "Natural language processing understands text",
    "Computer vision recognizes images",
    "Reinforcement learning optimizes decisions",
    "Algorithms solve complex problems efficiently",
    "Optimization improves model performance",
    "Classification predicts categorical labels",
    "Regression estimates continuous values",
    "Clustering groups similar data points",
    "Dimensionality reduces feature space",
    "Feature engineering creates better inputs"
]

def preprocess(text):
    return text.lower().split()

corpus = [preprocess(s) for s in sentences]

# 训练Word2Vec模型
model = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=3,
    min_count=1,
    epochs=100
)

# 评估1：词向量类比任务
print("=== 词向量类比评估 ===")

analogies = [
    (['machine', 'learning'], ['deep', 'learning'], 'X'),
    (['neural', 'networks'], ['neural', 'X'], 'networks'),
]

for pos, neg, expected in analogies:
    try:
        result = model.wv.most_similar(positive=pos, negative=neg, topn=1)
        print(f"{pos} - {neg} = {result[0][0]} (expected: {expected}, similarity: {result[0][1]:.4f})")
    except KeyError:
        print(f"{pos} - {neg} = 词汇不足")

print()

# 评估2：相似度任务
print("=== 相似度任务 ===")
pairs = [
    ('machine', 'learning'),
    ('neural', 'networks'),
    ('artificial', 'intelligence'),
]

for w1, w2 in pairs:
    try:
        sim = model.wv.similarity(w1, w2)
        print(f"similarity('{w1}', '{w2}') = {sim:.4f}")
    except KeyError:
        print(f"词汇表缺少: {w1}或{w2}")

print()

# 评估3：向量分布统计
print("=== 向量分布统计 ===")
vectors = np.array([model.wv[word] for word in model.wv.key_to_index])

print(f"向量形状: {vectors.shape}")
print(f"向量均值: {vectors.mean():.4f}")
print(f"向量标准差: {vectors.std():.4f}")
print(f"向量范数均值: {np.linalg.norm(vectors, axis=1).mean():.4f}")
print()

# 评估4：下游任务评估（文本分类）
# 创建简化的分类任务
from gensim.models import Word2Vec

# 模拟数据
docs = [
    ("Machine learning algorithms analyze data", "tech"),
    ("Deep neural networks process information", "tech"),
    ("Natural language understanding is challenging", "ai"),
    ("Computer vision recognizes objects in images", "ai"),
    ("Statistical methods extract insights from data", "data"),
    ("Data visualization presents information clearly", "data"),
]

# 转换为词向量
def doc_to_vec(doc, model):
    words = doc.lower().split()
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None

X = []
y = []
for doc, label in docs:
    vec = doc_to_vec(doc, model)
    if vec is not None:
        X.append(vec)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("=== 下游任务评估 ===")
print(f"样本数: {len(X)}")
print(f"标签分布: {dict(Counter(y))}")
```

## 11. 常见问题与易错点

在使用Word2Vec时，有几个常见的问题和易错点需要特别注意。

第一个问题是词表大小和嵌入维度的平衡。嵌入维度过小无法捕捉足够的信息，维度太大会导致过拟合和内存问题。一般建议：嵌入维度 ≈ min(词表大小, 100-300)。

第二个问题是训练不足或过度训练。训练不足会导致词向量质量差，过度训练会浪费计算资源。可以通过验证相似度任务来监控训练效果。

第三个问题是停用词和高频词的处理。常见停用词和标点符号应该先去除，否则会占用词表位置，降低其他词的训练效果。

第四个问题是子采样参数的设置。子采样可以提高训练效率，但过高的子采样率会丢失重要的高频词信息。

第五个问题是Windows平台的并行训练问题。Gensim在Windows上可能无法使用多线程，需要设置workers=1。

```python
# 常见问题的处理示例

from gensim.models import Word2Vec
import re

# 问题1：文本预处理
def clean_text(text):
    """清理文本"""
    text = text.lower()
    # 去除URL
    text = re.sub(r'http\S+', '', text)
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除标点符号
    text = re.sub(r'[^\w\s]', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 问题2：处理OOV词（不在词表中的词）
def get_oov_vector(word, model, strategy='mean'):
    """处理OOV词"""
    # 策略：返回零向量或平均向量
    if word in model.wv:
        return model.wv[word]
    else:
        return np.zeros(model.vector_size)

# 问题3：多线程问题
# Windows上设置workers=1
model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    workers=1  # Windows上使用1
)

# 问题4：训练过程中的监控
# 使用callbacks来监控训练过程
from gensim.models.callbacks import LossLogger

loss_logger = LossLogger()
model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    callbacks=[loss_logger]
)
print(loss_logger.losses)
```

## 12. 学习总结

Word2Vec是词嵌入领域的里程碑式工作，它将词的One-Hot表示转换为低维密集向量，使得语义相似性可以通过向量运算来表示。

从算法基础认知的角度，Word2Vec基于"分布假设"：语义相似的词倾向于出现在相似的上下文中。CBOW和Skip-gram两种模型都可以学习词向量。

从核心原理的角度，Word2Vec通过神经网络最大化词序列的似然概率，使用负采样或层级Softmax来加速Softmax的计算。

从数学公式的角度，Word2Vec的损失函数是负采样的对数损失，通过随机梯度下降来优化。

从应用场景的角度，Word2Vec的词向量可以用于信息检索、文本分类、机器翻译等NLP任务。

从优缺点的角度，Word2Vec的优点是能够捕捉语义相似性、向量维度低、训练无监督；缺点是无法处理多义词、无法处理OOV词。

Word2Vec为了解更高级的词嵌入方法（如GloVe、BERT）奠定了基础。GloVe利用全局共现统计来学习词向量，弥补了Word2Vec的不足。

## 13. 练习题与思考题与思考题（含答案）

### 练习题

**练习1**：解释CBOW和Skip-gram的区别。

答案：CBOW（Continuous Bag of Words）使用上下文词来预测中心词，适合小型数据集和常见词；Skip-gram使用中心词来预测上下文词，精度更高，适合大型数据集。CBOW的训练速度快，但对于低频词效果较差；Skip-gram的训练时间长，但对所有词都能学习到较好的向量。

**练习2**：为什么Word2Vec需要负采样？

答案：如果使用完整的Softmax，需要计算整个词表的指数和，时间复杂度为O(V)。负采样通过只计算少数负样本来近似Softmax，将时间复杂度降低到O(k)，其中k是负采样数量。

**练习3**：Word2Vec如何处理高频停用词？

答案：使用子采样技术，以概率1 - √(t/f_w)丢弃高频词，其中f_w是词的频率，t是阈值。这样可以减少高频词的影响，加快训练速度。

**练���4**���为什么"king - man + woman ≈ queen"？

答案：词向量捕捉了词的语义和语法关系。"king"和"queen"都是皇室成员，"man"和"woman"都是性别词。减法和加法操作可以理解为：去掉"man"的特征（男性），加上"woman"的特征（女性），结果应该接近皇室女性，即"queen"。

### 思考题

**思考1**：Word2Vec和GloVe的主要区别是什么？

思考要点：Word2Vec是基于局部上下文的预测模型，GloVe是基于全局共现统计的计数模型。Word2Vec利用词对的条件概率，GloVe利用词的共现概率。GloVe在某些任务上表现更好，因为它利用了全局统计信息。

**思考2**：Word2Vec能否处理多义词？

思考要点：不能。Word2Vec为每个词学习一个固定的向量，无论上下文如何，同一词的向量是相同的。要处理多义词，需要使用上下文相关的词表示，如BERT。

**思考3**：Word2Vec的训练需要多长时间？

思考要点：取决于语料库大小、词表大小、嵌入维度和硬件配置。对于一般规模的语料库（几百万词），训练时间从几分钟到几小时不等。


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

学习Word2Vec应该作为学习词嵌入的第一步。以下是建议的学习路径。

第一步，理解Word2Vec的原理和两种模型（CBOW、Skip-gram）。这是本章节的内容。

第二步，掌握Gensim库的使用。学会各种参数的设置和调优。

第三步，理解Word2Vec的优化技术（负采样、层级Softmax、子采样）。

第四步，对比其他词嵌入方法。GloVe、BERT等。

建议的后续学习内容：
- GloVe：基于全局共现统计的词嵌入方法
- FastText：处理OOV词的词嵌入方法
- BERT：基于Transformer的上下文相关词表示方法

通过系统地学习这些内容，可以建立完整的词嵌入知识体系，为深入学习NLP打下基础。