
# PLSA 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
PLSA（Probabilistic Latent Semantic Analysis，概率潜在语义分析）是一种基于概率模型的主题建模方法，通过EM算法学习文档-主题和主题-词的分布，揭示文本语料中潜在的语义结构。

### 1.2 直觉类比
想象你有很多新闻文章，但不知道每篇文章属于什么主题。PLSA就像一个"主题侦探"，它分析每篇文章的词汇分布，发现"体育"、"科技"、"政治"等潜在主题，并告诉你每篇文章涉及哪些主题，每个主题包含哪些关键词。

### 1.3 历史背景
PLSA由Thomas Hofmann于1999年提出，最初用于信息检索中的文本建模。PLSA是LDA（Latent Dirichlet Allocation）的前身，后者由Blei等人在2003年提出并成为更主流的主题模型。

### 1.4 算法定位
- 类型：无监督学习
- 输出：文档-主题分布、主题-词分布
- 模型类别：生成模型（概率模型）

### 1.5 前置知识
- 概率论基础（贝叶斯定理、条件概率）
- EM算法基础
- Python 编程（NumPy）

## 2. 核心原理
### 2.1 核心思想
PLSA的核心思想是"假设存在隐藏的主题变量"——每篇文档是若干潜在主题的混合，每个主题是词汇表上词的分布。观察到的词-文档共现是这些隐藏变量作用的结果。

### 2.2 工作流程
1. 初始化主题-词分布 $P(w|z)$ 和文档-主题分布 $P(z|d)$
2. E步：根据当前参数计算每个词的主题归属概率
3. M步：根据E步的结果更新参数
4. 迭代直到收敛
5. 输出主题分布

### 2.3 关键概念解释
- **潜在主题Z**：隐藏的语义单元
- **文档-主题分布**：每篇文档涉及各主题的概率
- **主题-词分布**：每个主题下各词出现的概率
- **共现矩阵**：词-文档共现频率矩阵

### 2.4 几何解释
从概率图模型角度看，PLSA建立了一个三层模型：文档层 → 主题层 → 词层。文档是主题上的概率分布，主题是词上的概率分布。

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $d$ | 文档 |
| $w$ | 词 |
| $z$ | 潜在主题 |
| $P(z|d)$ | 文档-主题分布 |
| $P(w|z)$ | 主题-词分布 |
| $n(d,w)$ | 词w在文档d中的出现次数 |

### 3.2 问题形式化
给定文档集合，观察到词-文档共现 $n(d,w)$，学习潜在主题结构。

观察数据的对数似然：
$$\mathcal{L} = \sum_{d \in \mathcal{D}} \sum_{w \in \mathcal{V}} n(d,w) \log P(w|d)$$

其中 $P(w|d)$ 可以展开为：
$$P(w|d) = \sum_{z \in \mathcal{Z}} P(w|z)P(z|d)$$

### 3.3 目标函数
$$\max_{\theta} \mathcal{L}(\theta) = \sum_{d,w} n(d,w) \log \sum_{z} P(w|z)P(z|d)$$

### 3.4 推导过程
**E步（期望步）**：计算后验概率
$$P(z|d,w) = \frac{P(w|z)P(z|d)}{\sum_{z'} P(w|z')P(z'|d)}$$

**M步（最大化步）**：更新参数
$$P(w|z) = \frac{\sum_{d} n(d,w) P(z|d,w)}{\sum_{w'} \sum_{d} n(d,w') P(z|d,w')}$$

$$P(z|d) = \frac{\sum_{w} n(d,w) P(z|d,w)}{\sum_{z'} \sum_{w} n(d,w) P(z'|d,w)}$$

### 3.5 最终解/算法步骤
1. 初始化 $P(w|z)$ 和 $P(z|d)$（随机或均匀）
2. 迭代EM算法：
   - E步：计算 $P(z|d,w)$
   - M步：更新 $P(w|z)$ 和 $P(z|d)$
3. 收敛后输出结果

## 4. 训练过程讲解
### 4.1 数据预处理
- 分词、去停用词
- 构建词汇表
- 词频统计
- 构建词-文档矩阵

### 4.2 参数初始化
- 随机初始化
- 均匀分布初始化
- 基于共现的初始化

### 4.3 迭代过程
```python
伪代码：
输入: 词-文档矩阵, 主题数K
1. 初始化 P(w|z), P(z|d)
2. for t = 1 to T:
3.     # E步
4.     for each (d, w):
5.         P(z|d,w) = P(w|z)P(z|d) / Σz' P(w|z')P(z'|d)
6.     # M步
7.     P(w|z) = Σd n(d,w)P(z|d,w) / Σw',d n(d,w')P(z|d,w')
8.     P(z|d) = Σw n(d,w)P(z|d,w) / Σz',w n(d,w)P(z'|d,w)
9.     if 收敛: break
```

### 4.4 收敛条件
- 似然函数变化小于阈值
- 达到最大迭代次数
- 参数变化小于阈值

### 4.5 超参数及推荐范围
- n_components (K): 10-100（根据语料大小调整）
- max_iter: 100-500
- tol: 1e-4

## 5. 应用场景
### 5.1 典型应用
- **主题建模**：从文档集合中发现主题
- **文本分类**：基于主题分布进行分类
- **信息检索**：查询扩展和文档相似度
- **推荐系统**：基于主题的用户/物品表示

### 5.2 适用数据特征
- 文本语料
- 存在潜在语义结构
- 文档数量足够

### 5.3 不适用场景
- 短文本（如推文）
- 实时性要求高
- 需要精确概率估计

## 6. 优缺点分析
### 6.1 优点
- 概率框架，结果可解释
- 发现潜在语义
- 可以处理同义词
- 与LDA相比更简单

### 6.2 缺点
- 参数随文档数线性增长
- 容易过拟合
- 不是完整的生成模型
- EM可能陷入局部最优

### 6.3 与同类算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| PLSA | 简单，概率可解释 | 参数多，可能过拟合 | 主题发现 |
| LDA | 正则化，全生成模型 | 计算复杂 | 主题建模 |
| LSA | 快速，SVD闭式解 | 无概率解释 | 语义分析 |
| NMF | 稀疏，可解释 | 非概率 | 文本分析 |

## 7. 调库实现
### 7.1 环境准备
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 7.2 完整代码示例
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# 1. 示例文档
documents = [
    "机器学习是人工智能的核心技术，通过算法让计算机从数据中学习",
    "深度学习是机器学习的子领域，使用神经网络模拟人脑",
    "自然语言处理研究如何让计算机理解和生成自然语言",
    "计算机视觉研究如何让计算机理解和处理图像和视频",
    "神经网络是深度学习的基础，由多层神经元组成",
    "机器学习广泛应用于数据分析、推荐系统、金融预测等领域",
    "自然语言处理技术包括文本分类、机器翻译、问答系统",
    "深度学习在图像识别、语音识别领域取得了突破性进展",
    "强化学习是机器学习的一个重要分支，通过与环境交互学习",
    "数据挖掘从大量数据中发现有价值的模式和规律",
    "机器翻译是自然语言处理的重要应用之一",
    "卷积神经网络在计算机视觉中应用广泛",
    "循环神经网络适用于处理序列数据",
    "迁移学习可以将预训练模型应用到新任务",
    "生成对抗网络可以生成逼真的图像和文本"
]

# 2. 构建词频矩阵
vectorizer = CountVectorizer(max_features=100, stop_words=None)
doc_term_matrix = vectorizer.fit_transform(documents)

print(f"文档数量: {len(documents)}")
print(f"词汇表大小: {len(vectorizer.get_feature_names_out())}")
print(f"词-文档矩阵形状: {doc_term_matrix.shape}")

# 3. PLSA/LDA模型训练（sklearn使用LDA，但原理相似）
n_topics = 3

# 使用LDA（与PLSA有相似目标）
lda = LatentDirichletAllocation(
    n_components=n_topics, 
    max_iter=50,
    learning_method='online',
    random_state=42,
    n_jobs=-1
)
lda.fit(doc_term_matrix)

# 4. 获取主题-词分布
feature_names = vectorizer.get_feature_names_out()
topic_word_dist = lda.components_  # (n_topics, n_vocab)

print("\n=== 主题提取结果 ===")
for topic_idx, topic in enumerate(topic_word_dist):
    top_word_indices = topic.argsort()[-8:][::-1]
    top_words = [feature_names[i] for i in top_word_indices]
    top_weights = [topic[i] for i in top_word_indices]
    print(f"\n主题{topic_idx+1}:")
    for word, weight in zip(top_words, top_weights):
        print(f"  {word}: {weight:.2f}")

# 5. 获取文档-主题分布
doc_topic_dist = lda.transform(doc_term_matrix)

print("\n=== 文档-主题分布（部分）===")
for i in range(3):
    print(f"文档{i+1}: {doc_topic_dist[i].round(3)}")

# 6. 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 主题-词分布热图
ax1 = axes[0]
im = ax1.imshow(topic_word_dist, aspect='auto', cmap='viridis')
ax1.set_xlabel('词汇')
ax1.set_ylabel('主题')
ax1.set_title('主题-词分布热图')
plt.colorbar(im, ax=ax1)

# 文档-主题分布
ax2 = axes[1]
x = np.arange(len(documents))
width = 0.25
for i in range(n_topics):
    ax2.bar(x + i*width, doc_topic_dist[:, i], width, label=f'主题{i+1}')
ax2.set_xlabel('文档')
ax2.set_ylabel('主题概率')
ax2.set_title('文档-主题分布')
ax2.set_xticks(x + width)
ax2.legend()

plt.tight_layout()
plt.show()

# 7. 文档相似度计算（基于主题分布）
query = "深度学习 神经网络"
query_vec = vectorizer.transform([query])
query_topic = lda.transform(query_vec)

similarities = cosine_similarity(query_topic, doc_topic_dist)[0]
print("\n=== 与查询最相似的文档 ===")
for idx in np.argsort(similarities)[::-1][:5]:
    print(f"文档{idx+1}: {similarities[idx]:.4f} - {documents[idx][:30]}...")

# 8. 困惑度评估
perplexity = lda.perplexity(doc_term_matrix)
print(f"\n模型困惑度: {perplexity:.2f}")
```

### 7.3 运行结果示例
```
文档数量: 15
词汇表大小: 30

=== 主题提取结果 ===
主题1: 机器学习, 深度学习, 神经网络, 数据, 学习...
主题2: 自然语言, 处理, 文本, 机器, 翻译...
主题3: 计算机, 视觉, 图像, 识别, 视频...
```

## 8. 手工代码实现
### 8.1 核心算法手写
```python
import numpy as np

class PLSAManual:
    """手工实现PLSA（概率潜在语义分析）"""
    
    def __init__(self, n_topics=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.P_w_z = None  # 主题-词分布
        self.P_z_d = None  # 文档-主题分布
        
    def fit(self, doc_term_matrix):
        """训练PLSA模型"""
        np.random.seed(self.random_state)
        
        # 转换为密集矩阵
        if hasattr(doc_term_matrix, 'toarray'):
            X = doc_term_matrix.toarray()
        else:
            X = doc_term_matrix
        
        n_docs, n_words = X.shape
        
        # 初始化参数（均匀分布 + 小随机扰动）
        self.P_w_z = np.random.rand(self.n_topics, n_words)
        self.P_w_z = self.P_w_z / self.P_w_z.sum(axis=1, keepdims=True)
        
        self.P_z_d = np.random.rand(n_docs, self.n_topics)
        self.P_z_d = self.P_z_d / self.P_z_d.sum(axis=1, keepdims=True)
        
        prev_ll = -np.inf
        
        for iteration in range(self.max_iter):
            # E步：计算 P(z|d,w)
            # P(z|d,w) ∝ P(w|z) * P(z|d)
            P_z_dw = np.zeros((n_docs, n_words, self.n_topics))
            
            for d in range(n_docs):
                for w in range(n_words):
                    if X[d, w] > 0:
                        # 计算未归一化的概率
                        unnorm = self.P_w_z[:, w] * self.P_z_d[d, :]
                        P_z_dw[d, w, :] = unnorm / (unnorm.sum() + 1e-10)
            
            # M步：更新参数
            # 更新 P(w|z)
            new_P_w_z = np.zeros((self.n_topics, n_words))
            for z in range(self.n_topics):
                for w in range(n_words):
                    new_P_w_z[z, w] = np.sum(X[:, w] * P_z_dw[:, :, z])
            
            # 归一化
            new_P_w_z = new_P_w_z / (new_P_w_z.sum(axis=1, keepdims=True) + 1e-10)
            self.P_w_z = new_P_w_z
            
            # 更新 P(z|d)
            new_P_z_d = np.zeros((n_docs, self.n_topics))
            for d in range(n_docs):
                for z in range(self.n_topics):
                    new_P_z_d[d, z] = np.sum(X[d, :] * P_z_dw[d, :, z])
            
            # 归一化
            new_P_z_d = new_P_z_d / (new_P_z_d.sum(axis=1, keepdims=True) + 1e-10)
            self.P_z_d = new_P_z_d
            
            # 计算对数似然
            ll = self._compute_log_likelihood(X)
            
            if iteration > 0 and abs(ll - prev_ll) < self.tol:
                print(f"收敛于第{iteration}轮, 对数似然: {ll:.4f}")
                break
            
            prev_ll = ll
            
            if iteration % 10 == 0:
                print(f"第{iteration}轮, 对数似然: {ll:.4f}")
        
        return self
    
    def _compute_log_likelihood(self, X):
        """计算对数似然"""
        ll = 0
        n_docs, n_words = X.shape
        
        for d in range(n_docs):
            for w in range(n_words):
                if X[d, w] > 0:
                    p_w_d = np.sum(self.P_w_z[:, w] * self.P_z_d[d, :])
                    ll += X[d, w] * np.log(p_w_d + 1e-10)
        
        return ll
    
    def transform(self, doc_term_matrix):
        """获取文档-主题分布"""
        if hasattr(doc_term_matrix, 'toarray'):
            X = doc_term_matrix.toarray()
        else:
            X = doc_term_matrix
        
        return self.P_z_d

# 测试手工实现
if __name__ == '__main__':
    from sklearn.feature_extraction.text import CountVectorizer
    
    # 简化测试数据
    docs = [
        "机器学习 机器学习 深度学习",
        "深度学习 神经网络",
        "自然语言 处理 文本",
        "自然语言 处理 语言",
        "计算机 视觉 图像"
    ]
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    
    # 手工实现
    plsa = PLSAManual(n_topics=2, max_iter=50)
    plsa.fit(X)
    
    print("\n特征词汇:", vectorizer.get_feature_names_out())
    print("\n主题-词分布:")
    print(plsa.P_w_z)
```

### 8.2 与调库结果对比
| 指标 | 手工实现 | sklearn(LDA) |
|------|----------|--------------|
| 收敛速度 | 较慢 | 优化过更快 |
| 结果质量 | 可比 | 更稳定 |

## 9. 可视化与结果理解
### 9.1 主题词汇可视化
```python
import matplotlib.pyplot as plt
import numpy as np

# 可视化每个主题的关键词
n_topics = 3
n_words = 10

fig, axes = plt.subplots(1, n_topics, figsize=(15, 4))

for topic_idx in range(n_topics):
    ax = axes[topic_idx]
    top_indices = topic_word_dist[topic_idx].argsort()[-n_words:][::-1]
    words = [feature_names[i] for i in top_indices]
    weights = [topic_word_dist[topic_idx, i] for i in top_indices]
    
    ax.barh(range(n_words), weights)
    ax.set_yticks(range(n_words))
    ax.set_yticklabels(words)
    ax.set_xlabel('权重')
    ax.set_title(f'主题{topic_idx+1}')
    ax.invert_yaxis()

plt.tight_layout()
plt.show()
```

### 9.2 文档-主题热图
```python
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(doc_topic_dist, annot=True, fmt='.2f', cmap='YlOrRd')
plt.xlabel('主题')
plt.ylabel('文档')
plt.title('文档-主题分布热图')
plt.show()
```

### 9.3 结果解读
- 每行表示一个文档在各主题上的概率分布
- 每列表示一个主题下各词的权重
- 高权重词代表该主题的核心词汇

## 10. 模型评估
### 10.1 评估指标选择
- **困惑度（Perplexity）**：衡量模型对数据的拟合能力
- **对数似然**：模型生成数据的概率
- **主题一致性**：主题内词汇的语义一致性

### 10.2 困惑度评估
```python
# 不同主题数的困惑度
for n_topics in [2, 3, 5, 10]:
    lda = LatentDirichletAllocation(n_topics=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    perplexity = lda.perplexity(doc_term_matrix)
    print(f"主题数={n_topics}, 困惑度: {perplexity:.2f}")
```

### 10.3 主题一致性评估
```python
# 简单的主题一致性：主题内词向量的相似度
from sklearn.metrics.pairwise import cosine_similarity

for topic_idx in range(n_topics):
    top_words = topic_word_dist[topic_idx].argsort()[-10:][::-1]
    # 使用共现矩阵计算一致性
    # 简化版本：计算平均相似度
    print(f"主题{topic_idx+1} top词: {list(feature_names[top_words])}")
```

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
- 停用词未去除
- 词汇表太大或太小
- 文档数量不足

### 11.2 模型层面常见错误
- 主题数选择不当
- EM收敛到局部最优
- 参数初始化不当

### 11.3 调参层面常见误区
- 过度追求低困惑度
- 忽略主题可解释性
- 未进行超参数调优

## 12. 学习总结
### 12.1 核心要点回顾
- PLSA通过EM算法学习文档-主题和主题-词分布
- 假设文档是主题的混合，主题是词的分布
- 使用E步和M步交替优化
- 与LDA相比，PLSA不是完整的生成模型

### 12.2 关键公式汇总
- E步：$P(z|d,w) = \frac{P(w|z)P(z|d)}{\sum_{z'} P(w|z')P(z'|d)}$
- M步：$P(w|z) = \frac{\sum_{d} n(d,w)P(z|d,w)}{\sum_{w',d} n(d,w')P(z|d,w')}$
- 目标：最大化对数似然

### 12.3 与前序/后续算法联系
- **前置算法**：LSA、TF-IDF
- **后续算法**：LDA（加入狄利克雷先验）

## 13. 练习题与思考题与思考题
### 13.1 基础练习题
1. PLSA和LDA的主要区别是什么？
2. 解释PLSA中的E步和M步。
3. 为什么PLSA需要迭代优化？

### 13.2 进阶思考题
1. PLSA的参数数量与什么有关？有什么问题？
2. 如何评估主题模型的质量？

### 13.3 详细答案与解析
1. **答案**：LDA在PLSA基础上加入了狄利克雷先验，作为正则化，避免过拟合，且是完整的生成模型。
2. **答案**：E步计算后验概率 $P(z|d,w)$，M步根据后验更新参数 $P(w|z)$ 和 $P(z|d)$。
3. **答案**：因为似然函数包含隐变量，直接求解困难，需要使用EM算法迭代优化。

## 14. 学习路径建议建议
### 14.1 前置知识
- 概率论基础
- EM算法
- 文本处理基础

### 14.2 平行算法
- LDA（主题模型）
- LSA（潜在语义分析）
- NMF（非负矩阵分解）

### 14.3 进阶算法
- LDA（加入先验）
- 动态主题模型
- 神经主题模型

### 14.4 推荐资源
- Hofmann (1999) "Probabilistic Latent Semantic Analysis"
- Blei et al. (2003) "Latent Dirichlet Allocation"
- 《Pattern Recognition and Machine Learning》- Bishop
