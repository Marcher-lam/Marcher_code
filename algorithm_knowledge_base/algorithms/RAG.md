# RAG（检索增强生成）学习文档

> 将外部知识库检索与大语言模型生成相结合的技术，让AI拥有实时、可扩展的知识

---

## 1. 算法基础认知

**一句话定义**：RAG（Retrieval-Augmented Generation，检索增强生成）是一种将外部知识库检索与大语言模型生成相结合的技术，通过先从知识库中检索相关信息，再让模型基于检索结果生成回答，从而解决模型知识过时和幻觉问题。

**直觉类比**：RAG就像一个学生在考试时翻书查资料——不用把所有知识都死记硬背（像纯参数模型），而是学会在需要时快速查找正确答案。模型就像有了"哆啦A梦的口袋"，需要什么知识就掏出来用。

**历史背景**：2020年，Facebook AI研究院（Meta）的Lewis等人提出RAG论文"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"，开创了将预训练检索与生成结合的范式。此后RAG成为大模型应用的主流架构。

**算法定位**：
- 类型：NLP → 检索+生成
- 输出：基于检索结果的生成文本
- 模型类型：检索器+生成器

**前置知识**：
- [必备]：Transformer、注意力机制
- [必备]：向量检索、Embedding
- [扩展]：LangChain、提示工程

---

## 2. 核心原理

### 2.1 核心思想

RAG的核心创新是**将参数化知识和非参数化知识分离**：
1. **参数化知识**：存储在模型参数中的通用知识
2. **非参数化知识**：存储在外部知识库中的实时、特定领域知识

核心思想可以概括为：**"检索+生成"两阶段架构，检索提供上下文，生成基于上下文回答**。

### 2.2 工作流程

```
用户问题 → 编码 → 向量数据库检索 → Top-K相关文档 → 构造成上下文提示 → LLM生成 → 最终回答
```

### 2.3 关键概念

- **Retriever（检索器）**：负责从知识库中找出相关文档，常用BM25或Dense Retriever
- **Generator（生成器）：负责基于检索结果生成回答，通常是预训练语言模型
- **Index（索引）**：将文档向量化后存储的结构，用于快速检索
- **Chunk（文本块）**：将长文档切分成的较小单元
- **Embedding（嵌入）**：将文本映射到向量空间的技术

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $q$ | 用户查询 |
| $D$ | 知识文档集合 |
| $d$ | 单个文档 |
| $z$ | 检索到的上下文 |
| $y$ | 生成的回答 |
| $\theta$ | 生成器参数 |

### 3.2 问题形式化

给定查询 $q$ 和文档集 $D$，RAG的目标是生成回答 $y$：

$$P(y|q) = \sum_{d \in D} P_{\phi}(d|q) P_{\theta}(y|d, q)$$

其中 $P_{\phi}$ 是检索器，$P_{\theta}$ 是生成器。

### 3.3 目标函数

**Seq2Seq Loss**：
$$L = -\sum_{(q, d, y)} \log P_{\theta}(y|d, q)$$

### 3.4 检索增强方式

**DPR (Dense Passage Retrieval)**：
$$sim(q, d) = E_q(q)^T E_d(d)$$

使用双编码器计算相似度。

---

## 4. 训练过程

### 4.1 数据准备

```python
# 文档切分
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_text)
```

### 4.2 构建索引

```python
# 向量化并存储
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)
```

### 4.3 检索生成

```python
# RAG链
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
```

---

## 5. 应用场景

### 5.1 典型应用

**企业知识库问答**：让AI回答公司内部文档问题

**医疗咨询**：基于医学文献提供建议

**法律助手**：检索法律条文生成分析

**客服系统**：结合产品手册回答用户问题

### 5.2 适用场景

- 需要最新/特定领域知识
- 减少模型幻觉
- 需要可解释性
- 大规模文档问答

---

## 6. 优缺点

### 6.1 优点

1. **知识可更新**：无需重新训练模型
2. **减少幻觉**：基于真实文档
3. **可解释性**：可追溯来源
4. **成本低**：比微调便宜

### 6.2 缺点

1. **依赖检索质量**
2. **延迟较高**
3. **上下文长度限制**

---

## 7. 调库实现

### 7.1 完整代码

```python
"""
RAG 检索增强生成实现
"""

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. 加载文档
loader = TextLoader("knowledge.txt")
documents = loader.load()

# 2. 文档切分
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 3. 构建向量索引
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# 4. 创建RAG链
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 5. 问答
question = "什么是RAG？"
answer = qa.run(question)
print(answer)
```

---

## 8. 手工实现

```python
"""
简化RAG实现
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    def __init__(self, documents, embedder):
        self.documents = documents
        self.embedder = embedder
        self.doc_embeddings = embedder.embed_documents(documents)
    
    def retrieve(self, query, k=3):
        query_embedding = self.embedder.embed_query(query)
        similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        top_k = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k]
    
    def generate(self, query, context):
        # 简化：直接拼接上下文
        prompt = f"基于以下上下文回答问题：\n\n上下文：{context}\n\n问题：{query}\n回答："
        return prompt

# 测试
docs = ["RAG是一种检索增强生成技术", "它可以结合外部知识库"]
rag = SimpleRAG(docs, embedding_model)
context = rag.retrieve("什么是RAG？")
answer = rag.generate("什么是RAG？", context)
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt

def visualize_rag():
    # 检索得分可视化
    scores = [0.9, 0.7, 0.5, 0.3, 0.1]
    labels = [f"Doc {i+1}" for i in range(5)]
    
    plt.figure(figsize=(10, 5))
    plt.barh(labels, scores, color='steelblue')
    plt.xlabel('Retrieval Score')
    plt.title('RAG Retrieval Results')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig('rag_retrieval.png')
    plt.show()
```

---

## 10. 评估

### 10.1 指标

| 指标 | 含义 |
|------|------|
| Recall@K | 检索覆盖率 |
| 生成质量 | 答案相关性 |
| 来源引用 | 是否准确引用 |

### 10.2 代码

```python
def evaluate_rag(qa, questions, ground_truth):
    results = []
    for q in questions:
        answer = qa.run(q)
        # 计算相关指标
        results.append({"question": q, "answer": answer})
    return results
```

---

## 11. 常见问题

### 11.1 问题

**检索不到相关内容**：检查文档质量和切分方式

**生成质量差**：调整检索数量或提示词

### 11.2 解决方案

```python
# 增加检索数量
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 使用更好的检索器
from langchain.retrievers import ContextualCompressionRetriever
```

---

## 12. 学习总结

### 12.1 核心

✓ 检索+生成架构
✓ 外部知识库
✓ 减少幻觉

### 12.2 算法联系

- 前置：Transformer、Embedding
- 相关：LangChain、LlamaIndex
- 进阶：Agent、Tool Learning

---

## 13. 练习题

**问题**：RAG和微调的区别？

答案：RAG可更新知识库，微调更新模型参数。

---

## 14. 学习路径

### 14.1 前置

- [ ] Transformer基础
- [ ] 向量检索

### 14.2 进阶

- [ ] LangChain
- [ ] Agent

### 14.3 资源

1. 论文：Lewis et al., "RAG", 2020
2. LangChain文档

---

## 附录

### A. 代码

见第7节。

### B. 参考文献

1. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020

---

**文档结束**