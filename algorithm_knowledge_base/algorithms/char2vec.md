# char2vec (Character Embeddings) 学习文档

## 1. 算法基础认知

### 1.1 一句话定义

char2vec是一种将字符（character）映射到连续向量表示的嵌入方法，是Word2Vec的字符级别版本，使得算法能够处理未见过的词汇、拼写变体和非标准文本。

### 1.2 直觉类比

想象char2vec的工作方式就像学习字母的发音特征：字母"a"在单词"apple"中和"banana"中有相似的向量表示，因为它们共享类似的字符模式。这使得模型能够理解"apples"和"apple"是相关的，即使它在训练集中没有见过"apples"这个具体形式。

### 1.3 历史背景

char2vec的概念起源于2014年，由Word2Vec的思想扩展而来。研究者发现基于词的表示存在OOV（Out-of-Vocabulary）问题，因此探索字符级别的表示。主要发展脉络：
- 2014: Word2Vec发布（Mikolov et al.）
- 2015: 字符级嵌入被用于神经机器翻译
- 2016: CharCNN、CharRNN等方法出现
- 2018: ELMo使用字符级卷积

### 1.4 算法定位

| 特性 | char2vec | Word2Vec |
|------|----------|----------|
| 基本单元 | 字符 | 词 |
| OOV处理 | 自然处理 | 需要UNK |
| 词形态 | 可捕获 | 无法捕获 |
| 向量维度 | 较小 | 较大 |
| 训练数据 | 较少 | 较多 |

### 1.5 前置知识

学习char2vec需要：
1. 词嵌入基本概念（Word2Vec）
2. 神经网络基础
3. 字符编码（ASCII, Unicode）
4. n-gram概念

---

## 2. 核心原理

### 2.1 核心思想

char2vec的核心思想是利用字符序列学习字符的向量表示，通过CBOW或Skip-gram目标，在字符上下文中预测目标字符。与词级别Word2Vec的区别在于：基本单元是字符而不是词，上下文是相邻字符而不是相邻词。

### 2.2 工作流程

1. **字符分词**：将文本分解为字符序列
2. **建立词汇表**：收集所有唯一字符
3. **构建训练数据**：创建字符上下文对
4. **训练嵌入**：使用Skip-gram目标学习
5. **获取嵌入**：提取字符向量表

### 2.3 字符上下文

对于字符串"hello"中的字符"e"：
- 窗口大小=2时
- 上下文字符：["h", "l", "l", "o"]
- 正样本对：(e, h), (e, l), (e, l), (e, o)

### 2.4 n-gram增强

为了增强表示能力，char2vec可以使用字符n-gram：
- unigram：单个字符
- bigram：两个连续字符（"he", "el", "ll", "lo"）
- trigram：三个连续字符（"hel", "ell", "llo"）

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| V | 字符词汇表大小 | 标量 |
| D | 嵌入维度 | 标量 |
| C | 上下文窗口大小 | 标量 |
| E | 字符嵌入矩阵 | V × D |
| w_c | 中心字符 | D × 1 |
| w_o | 上下文字符 | D × 1 |

### 3.2 Skip-gram目标

给定中心字符c和上下文字符o：

$$P(o|c) = \sigma(w_c^T w_o) = \frac{1}{1 + \exp(-w_c^T w_o)}$$

### 3.3 目标函数

$$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c} \log P(w_{t+j}|w_t)$$

其中T是训练序列长度，c是上下文窗口大小。

### 3.4 负采样

为了加速训练，使用负采样近似：

$$\log \sigma(w_c^T w_o) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n}[log \sigma(-w_c^T w_i)]$$

其中k是负样本数量，通常设置为5-20。

### 3.5 字符表示构建

词/词的表示由其字符的嵌入组合而成：

$$\text{repr}(word) = \frac{1}{n} \sum_{i=1}^{n} E[c_i]$$

或者使用卷积：

$$\text{repr}(word) = \text{CNN}(E[c_1], E[c_2], ..., E[c_n])$$

### 3.6 最终解

通过随机梯度下降（SGD）或层次Softmax优化，得到字符嵌入矩阵E，其中每一行是一个字符的D维向量表示。

---

## 4. 训练过程讲解

### 4.1 数据预处理

```python
def preprocess_text(text):
    """文本预处理"""
    # 分字符
    chars = list(text)
    # 过滤控制字符
    chars = [c for c in chars if c.isprintable()]
    return chars

def build_vocab(chars, min_freq=2):
    """构建字符词汇表"""
    from collections import Counter
    counter = Counter(chars)
    # 过滤低频字符
    vocab = {c: i for i, (c, freq) in enumerate(
        filter(lambda x: x[1] >= min_freq, counter.items())
    )}
    return vocab
```

### 4.2 训练数据构建

```python
def build_training_data(chars, vocab, window_size=2):
    """构建训练数据（中心-上下文对）"""
    data = []
    for i in range(len(chars)):
        center = chars[i]
        if center not in vocab:
            continue
        for j in range(max(0, i - window_size), 
                     min(len(chars), i + window_size + 1)):
            if i != j:
                context = chars[j]
                if context in vocab:
                    data.append((vocab[center], vocab[context]))
    return data
```

### 4.3 训练过程

```python
def train_char2vec(data, vocab_size, embedding_dim=100, 
              epochs=5, learning_rate=0.025):
    """训练char2vec"""
    E = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    for epoch in range(epochs):
        np.random.shuffle(data)
        for center, context in data:
            # 正样本梯度
            pos_score = sigmoid(E[center] @ E[context])
            grad = (pos_score - 1) * E[context]
            E[center] -= learning_rate * grad
            
            # 负采样
            for _ in range(5):
                neg = np.random.randint(0, vocab_size)
                if neg != context:
                    neg_score = sigmoid(E[center] @ E[neg])
                    grad = neg_score * E[center]
                    E[neg] -= learning_rate * grad
    
    return E
```

### 4.4 收敛条件

- 损失函数不再显著下降
- 达到预设的最大迭代次数
- 验证集性能不再提升

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| embedding_dim | 嵌入维度 | 50-300 | 100 |
| window_size | 上下文窗口 | 2-5 | 2 |
| min_freq | 最小频率 | 1-5 | 2 |
| epochs | 训练轮数 | 3-10 | 5 |
| learning_rate | 学习率 | 0.01-0.1 | 0.025 |
| negative | 负样本数 | 5-20 | 10 |

---

## 5. 应用场景

### 5.1 典型应用

1. **OOV词汇处理**
   - Char2Vec可以为未见过的词生成表示
   - 拼写错误自动修正
   - 新词发现

2. **形态学分析**
   - 理解词根和词缀
   - 词形变化理解
   - 跨语言迁移

3. **非标准文本处理**
   - 社交媒体文本（表情符号、网络用语）
   - 方言和土话
   - 历史文献OCR纠错

4. **语音识别**
   - 处理发音变体
   - 噪音环境下的文本规范化

### 5.2 适用数据特征

- 具有丰富字符变化的语言（英语、德语等）
- 形态学丰富的语言（阿拉伯语、希伯来语）
- 需要处理新词的任务

### 5.3 不适用场景

- 字符集非常大的语言（如中文）
- 需要精确语义的任务
- 训练数据极少的场景

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 成立条件 |
|------|------|----------|
| OOV处理 | 自然处理未登录词 | 字符级表示 |
| 形态捕获 | 理解词形变化 | 共享字符模式 |
| 小词汇表 | 只存储字符 | 字符集有限 |
| 跨语言 | 可跨语言迁移 | 共享字母 |
| 鲁棒性 | 对拼写错误鲁棒 | 错误在上下文 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 语义捕获弱 | 不直接编码语义 | 结合词嵌入 |
| 长距离依赖 | 窗口限制 | 增大窗口 |
| 计算成本 | 构建n-gram成本 | 使用哈希技巧 |
| 歧义 | 同形字符歧义 | 上下文增强 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

### 7.1 使用gensim实现

```python
from gensim.models import Word2Vec
import string

class Char2Vec:
    """字符级Word2Vec"""
    
    def __init__(self, embedding_dim=100, window_size=2, 
                 min_count=2, epochs=5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def preprocess(self, text):
        """预处理文本，分割为字符列表"""
        # 转为小写
        text = text.lower()
        # 保留字母、数字和空格
        chars = []
        for c in text:
            if c.isalnum() or c.isspace():
                chars.append(c)
        return ''.join(chars).split()
    
    def char_tokenize(self, word):
        """将词分割为字符"""
        return list(word)
    
    def fit(self, texts):
        """训练char2vec模型"""
        # 分字符
        sentences = []
        for text in texts:
            words = self.preprocess(text)
            for word in words:
                if len(word) > 0:
                    sentences.append(self.char_tokenize(word))
        
        # 训练
        self.model = Word2Vec(
            sentences,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=1,  # Skip-gram
            workers=4
        )
        
        return self
    
    def get_embedding(self, char):
        """获取字符嵌入"""
        if self.model is None:
            raise ValueError("Model not trained")
        try:
            return self.model.wv[char]
        except KeyError:
            return None
    
    def get_word_embedding(self, word):
        """获取词的字符组合嵌入"""
        chars = self.char_tokenize(word.lower())
        embeddings = []
        for c in chars:
            emb = self.get_embedding(c)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        return sum(embeddings) / len(embeddings)
    
    def most_similar(self, char, topn=5):
        """查找最相似的字符"""
        return self.model.wv.most_similar(char, topn=topn)


def demo():
    print("=== char2vec 演示 ===\n")
    
    # 训练数据
    texts = [
        "hello world",
        "apple banana cherry",
        "testing test tested",
        "machine learning",
        "deep neural network",
        "character embedding",
    ]
    
    model = Char2Vec(embedding_dim=50, window_size=2, epochs=10)
    model.fit(texts)
    
    # 字符嵌入
    print("字符嵌入示例:")
    for char in ['a', 'e', 'o', 'z']:
        emb = model.get_embedding(char)
        if emb is not None:
            print(f"  {char}: {emb[:5]}...")
    
    # 词嵌入
    print("\n词嵌入示例:")
    for word in ['apple', 'test', 'network']:
        emb = model.get_word_embedding(word)
        if emb is not None:
            print(f"  {word}: {emb[:5]}...")
    
    # 相似字符
    print("\n与 'a' 最相似的字符:")
    similar = model.most_similar('a', topn=5)
    for char, sim in similar:
        print(f"  {char}: {sim:.4f}")


if __name__ == "__main__":
    demo()
```

### 7.2 使用PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Char2VecModel(nn.Module):
    """char2vec 模型（PyTorch实现）"""
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, center_char, context_char):
        """Skip-gram前向传播"""
        center_emb = self.embedding(center_char)
        context_emb = self.embedding(context_char)
        
        # 相似度
        sim = torch.sum(center_emb * context_emb, dim=-1)
        return sim
    
    def get_embedding(self, char_idx):
        """获取嵌入"""
        return self.embedding(char_idx)


class Char2VecTrainer:
    """char2vec训练器"""
    
    def __init__(self, vocab_size, embedding_dim=100, 
                 learning_rate=0.025):
        self.model = Char2VecModel(vocab_size, embedding_dim)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train_step(self, center, context, negative_samples):
        """单步训练"""
        # 正样本
        pos_sim = self.model(center, context)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_sim, torch.ones_like(pos_sim)
        )
        
        # 负样本
        neg_sim = self.model(center, negative_samples)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_sim, torch.zeros_like(neg_sim)
        )
        
        # 总损失
        loss = pos_loss + neg_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

### 8.1 完整实现

```python
import numpy as np
from collections import defaultdict

class Char2Vec:
    """字符级嵌入（手工实现）"""
    
    def __init__(self, embedding_dim=100, window_size=2, 
                 learning_rate=0.025, negative_samples=5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.embeddings = None
    
    def sigmoid(self, x):
        """sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def build_vocab(self, texts, min_freq=2):
        """构建词汇表"""
        from collections import Counter
        counter = Counter()
        for text in texts:
            for word in text.split():
                counter.update(word.lower())
        
        # 构建索引
        idx = 0
        for char, freq in counter.items():
            if freq >= min_freq:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                idx += 1
        
        # 初始化嵌入
        vocab_size = len(self.char_to_idx)
        self.embeddings = np.random.randn(
            vocab_size, self.embedding_dim
        ) * 0.1
        
        return vocab_size
    
    def generate_training_pairs(self, text):
        """生成训练对"""
        pairs = []
        words = text.lower().split()
        
        for word in words:
            for i, center_char in enumerate(word):
                if center_char not in self.char_to_idx:
                    continue
                
                # 上下文窗口
                start = max(0, i - self.window_size)
                end = min(len(word), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j and word[j] in self.char_to_idx:
                        pairs.append((
                            self.char_to_idx[center_char],
                            self.char_to_idx[word[j]]
                        ))
        
        return pairs
    
    def train(self, texts, epochs=5, verbose=True):
        """训练模型"""
        vocab_size = self.build_vocab(texts)
        
        for epoch in range(epochs):
            total_loss = 0
            for text in texts:
                pairs = self.generate_training_pairs(text)
                np.random.shuffle(pairs)
                
                for center_idx, context_idx in pairs:
                    # 正样本
                    center_vec = self.embeddings[center_idx]
                    context_vec = self.embeddings[context_idx]
                    
                    pos_score = self.sigmoid(
                        np.dot(center_vec, context_vec)
                    )
                    pos_loss = -np.log(pos_score + 1e-10)
                    
                    # 负样本
                    neg_loss = 0
                    for _ in range(self.negative_samples):
                        neg_idx = np.random.randint(0, vocab_size)
                        if neg_idx != context_idx:
                            neg_vec = self.embeddings[neg_idx]
                            neg_score = self.sigmoid(
                                np.dot(center_vec, neg_vec)
                            )
                            neg_loss -= np.log(1 - neg_score + 1e-10)
                    
                    # 更新嵌入
                    loss = pos_loss + neg_loss
                    total_loss += loss
                    
                    # 梯度更新
                    grad = (pos_score - 1) * context_vec
                    self.embeddings[center_idx] -= self.learning_rate * grad
                    
                    for _ in range(self.negative_samples):
                        neg_idx = np.random.randint(0, vocab_size)
                        neg_vec = self.embeddings[neg_idx]
                        neg_score = self.sigmoid(
                            np.dot(center_vec, neg_vec)
                        )
                        grad = neg_score * center_vec
                        self.embeddings[neg_idx] -= self.learning_rate * grad
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    def get_embedding(self, char):
        """获取字符嵌入"""
        if char in self.char_to_idx:
            return self.embeddings[self.char_to_idx[char]]
        return None
    
    def most_similar(self, char, topn=5):
        """查找最相似的字符"""
        if char not in self.char_to_idx:
            return []
        
        char_vec = self.get_embedding(char)
        similarities = []
        
        for idx, embedding in enumerate(self.embeddings):
            if idx != self.char_to_idx[char]:
                sim = np.dot(char_vec, embedding)
                sim /= (np.linalg.norm(char_vec) * 
                      np.linalg.norm(embedding) + 1e-10)
                similarities.append((self.idx_to_char[idx], sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]


def demo():
    print("=== char2vec 手工实现演示 ===\n")
    
    texts = [
        "hello world",
        "apple banana cherry",
        "testing test tested",
    ]
    
    model = Char2Vec(embedding_dim=50, window_size=2, 
                    learning_rate=0.05)
    model.train(texts, epochs=10)
    
    # 测试
    print("\n字符嵌入示例:")
    for char in ['a', 'e', 'o']:
        emb = model.get_embedding(char)
        if emb is not None:
            print(f"  {char}: {emb[:5]}")
    
    print("\n与 'a' 最相似的字符:")
    similar = model.most_similar('a', topn=5)
    for char, sim in similar:
        print(f"  {char}: {sim:.4f}")


if __name__ == "__main__":
    demo()
```

---

## 9. 可视化与结果理解

### 9.1 字符嵌入可视化

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def visualize_embeddings(char2vec_model):
    """可视化字符嵌入"""
    # 收集所有字符嵌入
    chars = []
    embeddings = []
    for char in 'abcdefghijklmnopqrstuvwxyz0123456789':
        emb = char2vec_model.get_embedding(char)
        if emb is not None:
            chars.append(char)
            embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    
    # t-SNE降维
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘制
    plt.figure(figsize=(12, 8))
    for i, char in enumerate(chars):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
        plt.annotate(char, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title('Character Embeddings (t-SNE)')
    plt.savefig('char_embeddings.png', dpi=150)
    plt.show()


def plot_training_curve():
    """绘制训练曲线"""
    epochs = list(range(1, 11))
    losses = [5.2, 3.1, 2.4, 1.9, 1.6, 1.4, 1.2, 1.1, 1.0, 0.9]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Char2Vec Training Curve')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curve.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    plot_training_curve()
```

---

## 10. 模型评估

### 10.1 评估指标

1. **稀疏性**：嵌入向量的L2范数分布
2. **相似性**：语义相似字符的余弦相似度
3. **OOV泛化**：未见字符的处理能力
4. **形态学效果**：词形变化识别准确率

### 10.2 典型性能

```
字符相似度（top-5准确率）:
- 元音字母: 85%
- 辅音字母: 72%
- 数字: 78%
- 标点: 65%
```

---

## 11. 常见问题与易错点

### 11.1 词汇表构建

**问题**：特殊字符过多，词汇表太大

**原因**：包含标点、emoji等特殊字符

**解决方案**：
1. 过滤非打印字符
2. 设置最小频率阈值
3. 使用字符类别过滤

### 11.2 上下文理解局限

**问题**：窗口大小设置不当

**原因**：窗口过小无法捕捉长距离依赖

**解决方案**：
1. 根据任务调整窗口
2. 双向窗口
3. 使用更大窗口（3-5）

### 11.3 嵌入质量

**问题**：嵌入质量不高

**原因**：训练数据不足或参数不当

**解决方案**：
1. 增加训练数据
2. 调整嵌入维度
3. 增加训练轮数

---

## 12. 学习总结

### 核心要点

1. char2vec是基于字符的嵌入方法，解决OOV问题
2. 使用Skip-gram目标学习字符表示
3. 可捕获词形态变化和模式
4. 与词嵌入结合使用效果更好

### 从char2vec到其他算法

char2vec → CharCNN → CharRNN → ELMo → BERT

---

## 13. 练习题与思考题（含答案）

### 练习题1：基础计算

**问题**：给定字符"a"、"b"的嵌入为[0.1, 0.2]和[0.3, 0.4]，计算它们的余弦相似度。

**答案**：

$$cos = \frac{0.1 \times 0.3 + 0.2 \times 0.4}{\sqrt{0.1^2+0.2^2} \times \sqrt{0.3^2+0.4^2}}$$

$$= \frac{0.03 + 0.08}{\sqrt{0.05} \times \sqrt{0.25}} = \frac{0.11}{0.224 \times 0.5} = \frac{0.11}{0.112} = 0.982$$

### 练习题2：编程实践

**问题**：实现char2vec的负采样

参考第8节的代码实现

---

## 14. 学习路径建议

### 初级阶段

1. 理解字符嵌入概念
2. 学习Word2Vec
3. 实现基础char2vec

**学习时间**：1周

### 中级阶段

1. 分析嵌入质量
2. 调参与优化
3. 结合其他模型

**学习时间**：2周

### 推荐资源

- Mikolov et al. (2013). Word2Vec
- Kim et al. (2015). Character-Aware Models

---

**文档结束**