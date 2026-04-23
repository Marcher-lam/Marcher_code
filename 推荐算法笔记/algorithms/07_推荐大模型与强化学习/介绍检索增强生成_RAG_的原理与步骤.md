# 面试题：介绍检索增强生成 RAG 的原理与步骤

# 面试题：介绍检索增强生成 RAG 的原理与步骤

检索增强生成（RAG）是一种将信息检索与大型语言模型的文本生成能力相结合的技术架构。

其核心思想是：让大模型在回答问题时，能够先从一个外部知识库中查找相关信息，然后基于这些信息生成答案，从而提升回答的准确性、相关性和时效性。下表展示了 RAG 的核心工作流程与关键要点。

<table><tr><td>阶段</td><td>核心任务</td><td>关键方法与技术</td><td>主要目标</td></tr><tr><td>1.索引(Indexing)</td><td>知识库准备与向量化</td><td>数据清洗、文档分块、向量嵌入、存入向量数据库</td><td>构建一个可供高效检索的外部知识源</td></tr><tr><td>2.检索(Retrieval)</td><td>查找相关信息</td><td>将用户查询向量化，在向量数据库中进行相似性搜索（如余弦相似度）</td><td>从知识库中找出与用户问题最相关的文档片段</td></tr><tr><td>3.增强(Augmentation)</td><td>构建提示词</td><td>将检索到的文档片段和用户原始查询一起填入预设的提示词模板中</td><td>为语言模型提供生成答案所需的全部上下文信息</td></tr><tr><td>4.生成(Generation)</td><td>产生最终答案</td><td>大型语言模型（LLM）读取增强后的提示词，并生成自然、流畅且基于上下文的答案</td><td>输出准确、可靠且可追溯的响应</td></tr></table>

# RAG检索增强生成工作流程

![](images/b6ddd0b5a243f0f692adec0324e03743ea3ea33c3a2b17ae52c27af0263a190a.jpg)  
建立索引

![](images/2ba11ae74196cacb63b6be24ec5613ad6f41210cbb2e4ad98ae50816e4e93f38.jpg)  
检索生成

表：RAG算法关键步骤与核心技术选型概览  

<table><tr><td>关键步骤</td><td>核心任务</td><td>常见技术选型</td></tr><tr><td>数据预处理</td><td>文本分块、清洗</td><td>LangChain TextSplitter, 正则表达式</td></tr><tr><td>向量化</td><td>生成文本嵌入</td><td>Sentence-BERT, OpenAI text-embedding-ada-002</td></tr><tr><td>向量索引</td><td>存储与索引向量</td><td>FAISS, Chroma, Pinecone, Weaviate</td></tr><tr><td>检索器</td><td>相似度搜索</td><td>语义检索（FAISS），混合检索（BM25+向量）</td></tr><tr><td>重排序</td><td>优化检索结果</td><td>Cross-Encoder, Cohere rerank API</td></tr><tr><td>生成模型</td><td>生成最终答案</td><td>GPT系列, LLaMA 2, Claude</td></tr></table>

# $\sqsubset$ RAG 的架构演进

为了更好地应对复杂场景，RAG 架构也在不断演进，从最初的基础范式发展出更强大的形态：

 基础 RAG (Naive RAG)：即上述最基本的"检索-增强-生成"三步流程。其简单性也带来了检索质量不高、生成内容可能不准确等挑战。  
 高级 RAG (Advanced RAG)：在基础流程上增加了"检索前"和"检索后"的优化步骤。例如，在检索前对用户查询进行重写或扩展，或在检索后对结果进行重排序和过滤，以提升输入 LLM的信息质量。  
 模块化 RAG (Modular RAG)：将 RAG 系统拆分为像乐高积木一样可自由组合的功能模块（如查询理解、检索器、记忆模块等），提供了极大的灵活性，可以针对特定需求构建复杂的流水线，例如支持多轮对话或复杂推理。

# 向量检索的核心原理

向量检索是 RAG 系统的基石。其核心思想是将文本转化为高维向量空间中的点，通过计算向量间的距离来衡量语义相似度。

## 常用相似度计算方法

**余弦相似度**：衡量两个向量方向的相似性，取值范围 $[-1, 1]$：

$$\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| \cdot |\mathbf{b}|}$$

**欧氏距离**：衡量向量间的绝对距离：

$$d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

**内积（点积）**：对已归一化的向量，内积等价于余弦相似度。

## 向量索引加速方法

精确搜索在海量数据上代价过高，常用近似最近邻（ANN）算法加速：

- **IVF (Inverted File Index)**：将向量空间聚类为多个 Voronoi 单元，搜索时只检查最近邻单元
- **HNSW (Hierarchical Navigable Small World)**：构建多层图结构，通过贪心搜索快速定位近邻
- **PQ (Product Quantization)**：将高维向量分解为多个子空间并量化，大幅降低内存占用

## Embedding 模型对比

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|---------|
| text-embedding-ada-002 | 1536 | OpenAI 商用模型，质量高 | 通用 RAG |
| bge-large-zh | 1024 | 中文效果优秀 | 中文 RAG |
| m3e-base | 768 | 中文开源轻量 | 轻量级部署 |
| Sentence-BERT | 768 | 经典方案 | 英文场景 |
| BGE-M3 | 1024 | 多语言多粒度 | 多语言混合 |

# 文本分块策略详解

分块策略直接影响检索质量，常见策略如下：

**固定长度分块**：按字符或 token 数量切分，设置重叠窗口避免语义断裂。简单但可能切断完整语义。

**语义分块**：利用 embedding 相似度检测语义转折点，在语义变化处切分。保留语义完整性，但计算成本较高。

**递归字符分块**：按段落 → 句子 → 字符的优先级递归切分，LangChain 的 RecursiveCharacterTextSplitter 默认策略。工程实践中最常用。

**文档结构分块**：利用 Markdown 标题、HTML 标签等文档结构信息进行分块。适合结构化文档。

经验建议：chunk_size 通常设为 256-512 token，overlap 设为 chunk_size 的 10%-20%。

# RAG 评估指标

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| 上下文精确率 (Context Precision) | 检索到的相关文档占所有检索文档的比例 | 相关文档数 / 检索文档总数 |
| 上下文召回率 (Context Recall) | 被检索到的相关文档占所有相关文档的比例 | 被检索的相关文档数 / 总相关文档数 |
| 答案相关性 (Answer Relevance) | 生成答案与用户问题的相关程度 | LLM 评判或嵌入相似度 |
| 忠实度 (Faithfulness) | 生成答案与检索上下文的一致性 | 可溯源声明数 / 总声明数 |
| EM (Exact Match) | 答案与参考答案完全匹配的比例 | 精确匹配数 / 总问题数 |
| F1 | 答案 token 与参考答案 token 的 F1 | 2PR / (P+R) |

# 简易 RAG 流水线代码实现

以下代码实现了一个完整的简易 RAG 流水线，包含文档分块、向量化、检索和生成：

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleEmbedding:
    def __init__(self):
        self.word_vectors = {}
        self.dim = 128

    def fit(self, documents):
        word_set = set()
        for doc in documents:
            for word in doc.split():
                word_set.add(word.lower())
        rng = np.random.RandomState(42)
        for word in word_set:
            self.word_vectors[word] = rng.randn(self.dim)
        self.idf = {}
        n_docs = len(documents)
        word_doc_count = {}
        for word in word_set:
            count = sum(1 for doc in documents if word.lower() in doc.lower())
            word_doc_count[word] = count
        for word in word_set:
            self.idf[word] = np.log(n_docs / (1 + word_doc_count.get(word, 0)))

    def embed(self, text):
        words = [w.lower() for w in text.split() if w.lower() in self.word_vectors]
        if not words:
            return np.zeros(self.dim)
        weights = [self.idf.get(w, 1.0) for w in words]
        total = sum(weights)
        weights = [w / total for w in weights]
        vec = np.zeros(self.dim)
        for w, wt in zip(words, weights):
            vec += wt * self.word_vectors[w]
        return vec / (np.linalg.norm(vec) + 1e-8)


class SimpleChunker:
    def __init__(self, chunk_size=100, overlap=20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks


class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, texts, embeddings):
        for text, emb in zip(texts, embeddings):
            self.texts.append(text)
            self.vectors.append(emb)

    def search(self, query_embedding, top_k=3):
        if not self.vectors:
            return []
        matrix = np.array(self.vectors)
        sims = cosine_similarity([query_embedding], matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "text": self.texts[idx],
                "score": float(sims[idx]),
                "index": int(idx)
            })
        return results


class SimpleRAG:
    def __init__(self, chunk_size=80, overlap=15, top_k=3):
        self.chunker = SimpleChunker(chunk_size, overlap)
        self.embedder = SimpleEmbedding()
        self.vector_store = SimpleVectorStore()
        self.top_k = top_k
        self.knowledge_base = []

    def build_index(self, documents):
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.split(doc)
            all_chunks.extend(chunks)
        self.knowledge_base = all_chunks
        self.embedder.fit(all_chunks)
        embeddings = [self.embedder.embed(chunk) for chunk in all_chunks]
        self.vector_store.add(all_chunks, embeddings)
        print(f"索引构建完成: {len(documents)} 篇文档, {len(all_chunks)} 个分块")

    def retrieve(self, query):
        query_emb = self.embedder.embed(query)
        results = self.vector_store.search(query_emb, self.top_k)
        return results

    def augment(self, query, retrieved_docs):
        context = "\n\n".join([f"[文档{i+1}] {doc['text']}" for i, doc in enumerate(retrieved_docs)])
        prompt = (
            f"请根据以下参考资料回答用户问题。如果资料中没有相关信息，请说明。\n\n"
            f"参考资料：\n{context}\n\n"
            f"用户问题：{query}\n\n"
            f"回答："
        )
        return prompt

    def query(self, query):
        retrieved = self.retrieve(query)
        prompt = self.augment(query, retrieved)
        return {
            "query": query,
            "retrieved": retrieved,
            "prompt": prompt,
            "context_count": len(retrieved)
        }


def demo_rag():
    documents = [
        "推荐系统是信息过滤系统的一个子类，用于预测用户对物品的评分或偏好。"
        "常见的推荐算法包括协同过滤、基于内容的推荐和混合推荐方法。"
        "协同过滤是最经典的推荐算法，分为基于用户的协同过滤和基于物品的协同过滤。",

        "深度学习在推荐系统中的应用越来越广泛。"
        "Wide & Deep 模型结合了记忆能力和泛化能力。"
        "DeepFM 模型通过因子分解机自动学习特征交叉。"
        "DIN（Deep Interest Network）通过注意力机制捕捉用户兴趣的动态变化。",

        "向量检索是现代推荐系统的核心技术之一。"
        "FAISS 是 Meta 开源的高效向量相似性搜索库。"
        "ANN（近似最近邻）算法通过牺牲少量精度换取数量级的速度提升。"
        "常用的 ANN 算法包括 IVF、HNSW 和 PQ 等方法。",

        "Transformer 架构由多头自注意力机制和前馈神经网络组成。"
        "BERT 是基于 Transformer 编码器的预训练语言模型。"
        "GPT 是基于 Transformer 解码器的生成式预训练模型。"
        "Transformer 在推荐系统中也有广泛应用，如 SASRec 和 BST 等模型。"
    ]

    rag = SimpleRAG(chunk_size=60, overlap=10, top_k=2)
    rag.build_index(documents)

    queries = [
        "推荐系统有哪些常见算法？",
        "深度学习如何应用于推荐？",
        "什么是向量检索？",
    ]

    for q in queries:
        result = rag.query(q)
        print(f"\n{'='*60}")
        print(f"问题: {result['query']}")
        print(f"检索到 {result['context_count']} 个相关文档:")
        for r in result['retrieved']:
            print(f"  [相似度: {r['score']:.4f}] {r['text'][:80]}...")


if __name__ == "__main__":
    demo_rag()
```

# RAG 的优化技巧

## 检索前优化

- **查询改写**：使用 LLM 将模糊查询改写为更精确的检索查询
- **HyDE (Hypothetical Document Embedding)**：让 LLM 先生成一个假设性答案，用该答案的 embedding 去检索
- **查询扩展**：用同义词或 LLM 扩展原始查询，增加召回率

## 检索后优化

- **重排序 (Re-ranking)**：使用 Cross-Encoder 对初筛结果做精排
- **上下文压缩**：对检索到的长文档进行摘要，减少噪声
- **混合检索**：结合 BM25 关键词检索和向量语义检索，取长补短

## 生成优化

- **引用溯源**：要求模型在回答中标注信息来源
- **自我反思 (Self-RAG)**：让模型评估自己回答的可靠性，不确定时重新检索
- **多轮迭代检索 (Iterative Retrieval)**：根据初次生成结果决定是否需要补充检索

# 常见问题与陷阱

- **分块过大**：检索精度下降，噪声信息过多。建议控制在 256-512 token
- **分块过小**：上下文不完整，语义碎片化。需设置合理的 overlap
- **忽略关键词检索**：纯向量检索在精确匹配场景（如产品编号）表现不佳，应混合 BM25
- **Embedding 模型不匹配**：中英文场景需选用对应的 embedding 模型
- **知识库未更新**：RAG 的时效性依赖知识库的更新频率，需建立定期更新机制
- **过度依赖检索**：当检索结果质量差时，不如让模型直接回答并标注不确定性

# 总结

RAG 通过"检索-增强-生成"三阶段流水线，有效解决了大模型的幻觉、时效性和领域知识不足问题。关键技术选型包括：分块策略（递归字符分块最常用）、向量检索（HNSW 效果与速度的平衡）、Embedding 模型（中文推荐 bge 系列）和重排序（Cross-Encoder）。从基础 RAG 到高级 RAG 再到模块化 RAG，架构在不断演进以应对更复杂的应用场景。
