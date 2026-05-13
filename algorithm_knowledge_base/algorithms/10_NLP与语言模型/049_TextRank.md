# TextRank 文本排名算法

> TextRank是一种基于图排序的文本 summarization算法，通过构建词/句图计算重要性，是抽取式摘要的经典baseline。

## 1. 算法基础认知

### 1.1 什么是TextRank

TextRank是一种从文本中自动提取重要句子或关键词的算法。它将文本表示为图，使用类似Google PageRank的方法计算每个节点的重要性。

### 1.2 直觉类比

想象你在一个派对上，想找出最"有声望"的人。你可以记录谁提到了谁，被重要的人提到的人更重要。TextRank用类似思路评估句子或词的重要性。

### 1.3 历史背景

- **2004年**：Mihalcea和Tarau在EMNLP上提出
- **灵感来源**：Google的PageRank算法

### 1.4 算法定位

- **任务类型**：抽取式文本摘要、关键词提取
- **所属类别**：无监督学习/图方法

## 2. 核心原理

### 2.1 核心思想

将文本中的句子（或词）作为图中的节点，句子之间的相似度作为边的权重，然后迭代计算每个节点的PageRank得分。

### 2.2 工作流程

1. **分句**：将文本分成句子
2. **建图**：节点=句子，边=句子相似度
3. **迭代**：迭代传播分数
4. **排序**：选择高分句子作为摘要

### 2.3 参数说明

- **damping factor** ($\delta$)：通常设为0.85
- **迭代次数**：默认100
- **收敛阈值**：默认$10^{-6}$

## 3. 数学公式与推导

### 3.1 核心公式

$$WS(V_i) = (1-\delta)/N + \delta \cdot \sum_{j \in In(V_i)} \frac{WS(V_j) \cdot w_{ji}}{\sum_{k \in Out(V_j)} w_{jk}}$$

其中：
- $WS(V_i)$：节点$V_i$的权重分数
- $w_{ji}$：从$V_j$到$V_i$的边权重
- $N$：节点总数
- $\delta$：阻尼因子（通常0.85）

### 3.2 相似度计算

句子$s_i$和$s_j$的相似度：
$$\text{sim}(s_i, s_j) = \frac{|\{w_k | w_k \in s_i \cap s_j\}|}{log(|s_i|) + log(|s_j|)}$$

使用共享词的数量计算。

## 4. 应用场景

### 4.1 典型应用

1. **抽取式摘要**：选择重要句子组合成摘要
2. **关键词提取**：选择重要的词
3. **句子相似度**：可用于聚类

### 4.2 适用数据特征

- 新闻文章（信息密集）
- 可以提取式摘要的文档
- 需要无监督方法的场景

## 5. 优缺点分析

### 5.1 优点

| 优点 | 说明 |
|------|------|
| 无监督 | 不需要标注数据 |
| 简单高效 | 实现和计算都简单 |
| 可解释 | 基于图的结构清晰 |

### 5.2 缺点

| 缺点 | 说明 |
|------|------|
| 抽取式 | 不能生成新内容 |
| 倾向首句 | 容易选择开头句子 |
| 无语义理解 | 只是表层匹配 |

## 6. 调库实现

```python
from summa.summarizer import Summarizer
from summa.keywords import KeywordsExtractor

# 使用summa库进行摘要
text = """
Machine learning is a subfield of artificial intelligence.
Deep learning is a technique used in machine learning.
Supervised learning is a popular machine learning method.
Unsupervised learning is another machine learning technique.
"""

# 生成摘要（默认取前几句）
summary = Summarizer.summarize(text, words=50)
print(f"摘要: {summary}")

# 提取关键词
keywords_extractor = KeywordsExtractor()
keywords = keywords_extractor.extract(text, words=5)
print(f"关键词: {keywords}")
```

## 7. 手工代码实现

```python
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer


class TextRank:
    """TextRank算法实现"""
    
    def __init__(self, damping=0.85, max_iter=100, tol=1e-6):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
    
    def summarize(self, text, top_k=3):
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return sentences
        
        similarity_matrix = self._build_similarity_matrix(sentences)
        scores = self._pagerank(similarity_matrix)
        
        # 按分数排序，取top k
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked[:top_k])
        
        return ' '.join([sentences[i] for i in sorted(top_indices)])
    
    def _split_sentences(self, text):
        """分句"""
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _build_similarity_matrix(self, sentences):
        """构建相似度矩阵"""
        n = len(sentences)
        
        # 使用TF-IDF向量化
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(sentences)
        
        # 计算余弦相似度
        similarity = (tfidf * tfidf.T).toarray()
        
        return similarity
    
    def _pagerank(self, similarity):
        """PageRank算法"""
        n = similarity.shape[0]
        
        # 初始化
        scores = np.ones(n) / n
        
        # 归一化（按行）
        out_degrees = similarity.sum(axis=1, keepdims=True)
        out_degrees[out_degrees == 0] = 1  # 避免除零
        transition = similarity / out_degrees
        
        # 迭代
        for _ in range(self.max_iter):
            new_scores = (1 - self.damping) / n + self.damping * transition.T @ scores
            
            if np.linalg.norm(new_scores - scores) < self.tol:
                break
            
            scores = new_scores
        
        return scores


# 测试
text = """
Machine learning is a subfield of artificial intelligence.
Deep learning is a technique used in machine learning.
 supervised learning is a popular machine learning method.
Unsupervised learning is another machine learning technique.
"""

textrank = TextRank()
summary = textrank.summarize(text, top_k=2)
print(f"摘要: {summary}")
```

## 8. 评估指标

TextRank常用评估指标：

| 指标 | 说明 |
|------|------|
| ROUGE-1 | unigram召回 |
| ROUGE-2 | bigram召回 |
| ROUGE-L | 最长公共子串 |

典型Baseline分数（CNN/DailyMail）：
```
ROUGE-1: 0.38-0.44
ROUGE-2: 0.15-0.22
```

## 9. 学习总结

TextRank是抽取式摘要的经典Baseline，基于PageRank思想实现。虽然简单，但在没有Transformer的情况下是一个有用的 baseline。

## 10. 练习题

**基础题**：TextRank vs Transformer摘要的区别？

**答案**：
- TextRank是抽取式：从原文选择完整句子
- Transformer：可以生成新的表述（抽象式）

## 11. 学习路径

- **前置**：PageRank算法
- **进阶**：Transformer摘要（BART、T5）
- **资源**：TextRank原始论文

## 4. 训练过程讲解
### 训练步骤
1. **数据加载**：Dataset + DataLoader 批处理
2. **前向传播**：数据通过网络计算输出
3. **损失计算**：对比预测与标签
4. **反向传播**：计算梯度
5. **参数更新**：优化器更新权重
6. **循环迭代**：重复直至收敛

## 9. 可视化与结果理解

### 推荐可视化
1. **训练曲线**：损失随训练轮次变化，观察收敛趋势
2. **性能对比**：TextRank与基准方法对比
3. **特征重要性**（如适用）：各特征贡献度

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.title('TextRank Training Loss')
plt.show()
```

### 结果解读
- 训练损失持续下降表明模型在学习
- 验证损失上升可能过拟合
- 差距过大需考虑正则化


## 10. 模型评估

### 评估指标
- **准确率(Accuracy)**：正确预测比例
- **精确率/召回率/F1**：综合评估分类质量
- **AUC-ROC**：分类器整体性能
- **损失值**：训练收敛关键指标

### 评估方法
1. **K折交叉验证**：稳健的性能估计
2. **留出法**：独立训练/验证/测试集
3. **时间序列验证**：滚动窗口（金融场景）

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```


## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估

