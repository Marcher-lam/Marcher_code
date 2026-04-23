# Item2Vec 学习文档

## 1. 算法基础认知

### 1.1 什么是 Item2Vec？

Item2Vec 是将 **Word2Vec** 的思想应用于**物品嵌入学习**的方法。它将用户的购买/点击序列类比为文本中的句子，将物品类比为单词，通过 Skip-gram 模型学习物品的向量表示。

### 1.2 核心思想

```
Word2Vec:
文本 = [word1, word2, word3, ...]
目标：学习单词的语义向量

Item2Vec:
用户行为序列 = [item1, item2, item3, ...]
目标：学习物品的语义向量
```

### 1.3 与传统方法对比

| 方法 | 物品关系 | 可扩展性 | 冷启动 |
|------|----------|----------|--------|
| ItemCF | 显式相似度 | 差 | 差 |
| MF | 隐式分解 | 中 | 差 |
| Item2Vec | 语义相似 | 好 | 中 |

## 2. 核心原理

### 2.1 Skip-gram 模型

给定物品序列，预测上下文物品：

$$\max \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \ne 0} \log p(i_{t+j} | i_t)$$

其中：
- $T$：序列长度
- $c$：上下文窗口大小
- $p(i_j | i_t)$：Softmax 概率

### 2.2 负采样

为避免 Softmax 计算量过大，使用负采样：

$$\log \sigma(v_{i_O}'^T v_{i_I}) + \sum_{k=1}^{K} E_{i_k \sim P_n(i)} [\log \sigma(-v_{i_k}'^T v_{i_I})]$$

### 2.3 Item2Vec 的特点

1. **序列无关性**：物品顺序不重要，只要是同一用户的行为
2. **窗口自适应**：窗口大小可以覆盖整个序列
3. **联合训练**：所有用户序列一起训练

## 3. 完整实现

### 3.1 Item2Vec 模型

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Dict, Tuple
import random


class Item2VecDataset(Dataset):
    """
    Item2Vec 数据集

    处理用户行为序列，生成 Skip-gram 训练数据
    """

    def __init__(self, sequences: List[List[int]], window_size: int = 5,
                 n_negatives: int = 5, item_counts: Dict = None):
        """
        参数:
            sequences: 用户行为序列列表
            window_size: 上下文窗口大小
            n_negatives: 负采样数量
            item_counts: 物品计数（用于负采样分布）
        """
        self.window_size = window_size
        self.n_negatives = n_negatives

        # 构建训练数据
        self.pairs = []

        for seq in sequences:
            if len(seq) < 2:
                continue

            # 为每个物品，选择其他物品作为上下文
            for i, target in enumerate(seq):
                # Item2Vec 特点：可以与序列中所有其他物品配对
                context_items = seq[:i] + seq[i+1:]

                for context in context_items:
                    self.pairs.append((target, context))

        # 负采样分布
        if item_counts:
            items = list(item_counts.keys())
            counts = np.array([item_counts[i] for i in items], dtype=np.float64)
            # 3/4 次方平滑
            counts = np.power(counts, 0.75)
            self.neg_probs = counts / counts.sum()
            self.neg_items = items
        else:
            self.neg_probs = None
            self.neg_items = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, context = self.pairs[idx]

        # 负采样
        if self.neg_probs is not None:
            neg_indices = np.random.choice(
                len(self.neg_items),
                size=self.n_negatives,
                p=self.neg_probs
            )
            negatives = [self.neg_items[i] for i in neg_indices]
        else:
            negatives = [random.randint(0, len(self.neg_items) - 1)
                        for _ in range(self.n_negatives)]

        return {
            'target': torch.LongTensor([target]),
            'context': torch.LongTensor([context]),
            'negatives': torch.LongTensor(negatives)
        }


class Item2Vec(nn.Module):
    """
    Item2Vec: Neural Item Embedding for Collaborative Filtering

    基于 Word2Vec 的物品嵌入学习方法
    """

    def __init__(self, n_items: int, embed_dim: int = 64):
        """
        参数:
            n_items: 物品数量
            embed_dim: 嵌入维度
        """
        super().__init__()

        self.n_items = n_items
        self.embed_dim = embed_dim

        # 中心物品嵌入
        self.target_embeddings = nn.Embedding(n_items, embed_dim)
        # 上下文物品嵌入
        self.context_embeddings = nn.Embedding(n_items, embed_dim)

        # 初始化
        self._init_embeddings()

    def _init_embeddings(self):
        """初始化嵌入"""
        init_range = 0.5 / self.embed_dim
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, target, context, negatives):
        """
        前向传播

        参数:
            target: (batch, 1) 目标物品
            context: (batch, 1) 正样本上下文
            negatives: (batch, n_neg) 负样本

        返回:
            loss: 损失
        """
        # 获取嵌入
        target_emb = self.target_embeddings(target).squeeze(1)  # (batch, dim)
        context_emb = self.context_embeddings(context).squeeze(1)  # (batch, dim)
        neg_emb = self.context_embeddings(negatives)  # (batch, n_neg, dim)

        # 正样本分数
        pos_score = torch.sum(target_emb * context_emb, dim=1)  # (batch,)
        pos_loss = F.logsigmoid(pos_score)

        # 负样本分数
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze(2)  # (batch, n_neg)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # (batch,)

        # 总损失
        loss = -(pos_loss + neg_loss).mean()

        return loss

    def get_item_embedding(self, item_id: int) -> torch.Tensor:
        """获取物品嵌入"""
        return self.target_embeddings.weight[item_id]

    def get_all_embeddings(self) -> torch.Tensor:
        """获取所有物品嵌入"""
        return self.target_embeddings.weight.data

    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple]:
        """
        获取相似物品

        参数:
            item_id: 物品 ID
            top_k: 返回数量

        返回:
            [(item_id, similarity), ...]
        """
        with torch.no_grad():
            all_emb = self.get_all_embeddings()
            target_emb = self.get_item_embedding(item_id)

            # 计算余弦相似度
            similarities = F.cosine_similarity(
                target_emb.unsqueeze(0),
                all_emb,
                dim=1
            )

            # 排序
            top_values, top_indices = torch.topk(similarities, top_k + 1)

            results = []
            for idx, sim in zip(top_indices[1:], top_values[1:]):  # 排除自己
                results.append((int(idx), float(sim)))

            return results


class Item2VecTrainer:
    """
    Item2Vec 训练器
    """

    def __init__(self, n_items: int, embed_dim: int = 64,
                 learning_rate: float = 0.01):
        self.model = Item2Vec(n_items, embed_dim)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate
        )

    def train(self, train_loader, epochs: int = 10):
        """训练"""
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0

            for batch in train_loader:
                target = batch['target']
                context = batch['context']
                negatives = batch['negatives']

                self.optimizer.zero_grad()

                loss = self.model(target, context, negatives)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return self.model


class Item2VecRecommender:
    """
    基于 Item2Vec 的推荐器
    """

    def __init__(self, model: Item2Vec, item_to_idx: Dict, idx_to_item: Dict):
        self.model = model
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item

    def recommend(self, user_history: List, top_k: int = 10,
                  exclude_items: List = None) -> List[Tuple]:
        """
        为用户推荐

        参数:
            user_history: 用户历史物品列表
            top_k: 推荐 K 个
            exclude_items: 排除的物品

        返回:
            [(item_id, score), ...]
        """
        exclude = set(exclude_items or [])

        # 获取用户历史物品的嵌入
        with torch.no_grad():
            user_emb = torch.zeros(self.model.embed_dim)
            count = 0

            for item in user_history:
                if item in self.item_to_idx:
                    idx = self.item_to_idx[item]
                    user_emb += self.model.get_item_embedding(idx)
                    count += 1

            if count == 0:
                return []

            user_emb = user_emb / count

            # 计算与所有物品的相似度
            all_emb = self.model.get_all_embeddings()
            similarities = F.cosine_similarity(
                user_emb.unsqueeze(0),
                all_emb,
                dim=1
            )

            # 排序
            top_values, top_indices = torch.topk(similarities, top_k + len(user_history))

            results = []
            for idx, sim in zip(top_indices, top_values):
                item_id = self.idx_to_item[int(idx)]
                if item_id not in exclude and item_id not in user_history:
                    results.append((item_id, float(sim)))
                    if len(results) >= top_k:
                        break

            return results

    def get_similar_items(self, item_id, top_k: int = 10) -> List[Tuple]:
        """获取相似物品"""
        if item_id not in self.item_to_idx:
            return []

        idx = self.item_to_idx[item_id]
        similar = self.model.get_similar_items(idx, top_k)

        return [(self.idx_to_item[i], s) for i, s in similar]


def prepare_data(interactions: List[Tuple]) -> Tuple[List, Dict, Dict]:
    """
    准备数据

    参数:
        interactions: [(user_id, item_id), ...]

    返回:
        sequences: 用户行为序列
        item_to_idx: 物品到索引映射
        idx_to_item: 索引到物品映射
    """
    from collections import defaultdict

    # 按用户分组
    user_items = defaultdict(list)
    all_items = set()

    for user_id, item_id in interactions:
        user_items[user_id].append(item_id)
        all_items.add(item_id)

    # 构建索引
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    # 转换序列
    sequences = []
    for user_id, items in user_items.items():
        seq = [item_to_idx[item] for item in items]
        sequences.append(seq)

    # 计算物品计数
    item_counts = Counter()
    for seq in sequences:
        item_counts.update(seq)

    return sequences, item_to_idx, idx_to_item, item_counts


# 使用示例
def demo_item2vec():
    """Item2Vec 示例"""
    # 模拟数据
    interactions = []
    np.random.seed(42)

    # 生成模拟交互
    n_users = 100
    n_items = 50

    for user in range(n_users):
        # 每个用户有 5-15 个交互
        n_interactions = np.random.randint(5, 15)
        items = np.random.choice(n_items, n_interactions, replace=False)
        for item in items:
            interactions.append((f'u{user}', f'i{item}'))

    # 准备数据
    sequences, item_to_idx, idx_to_item, item_counts = prepare_data(interactions)

    print(f"用户数: {len(sequences)}")
    print(f"物品数: {len(item_to_idx)}")

    # 创建数据集
    dataset = Item2VecDataset(
        sequences,
        window_size=5,
        n_negatives=5,
        item_counts=item_counts
    )

    train_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True
    )

    # 训练
    trainer = Item2VecTrainer(
        n_items=len(item_to_idx),
        embed_dim=32,
        learning_rate=0.01
    )

    model = trainer.train(train_loader, epochs=10)

    # 创建推荐器
    recommender = Item2VecRecommender(model, item_to_idx, idx_to_item)

    # 测试推荐
    user_history = [f'i{np.random.randint(0, n_items)}' for _ in range(5)]
    recs = recommender.recommend(user_history, top_k=5)

    print("\n为用户推荐:")
    for item, score in recs:
        print(f"  {item}: {score:.4f}")

    # 测试相似物品
    item_id = f'i{np.random.randint(0, n_items)}'
    similar = recommender.get_similar_items(item_id, top_k=5)

    print(f"\n与 {item_id} 相似的物品:")
    for item, score in similar:
        print(f"  {item}: {score:.4f}")


if __name__ == "__main__":
    demo_item2vec()
```

### 3.2 Gensim 实现

```python
from gensim.models import Word2Vec
import numpy as np


class Item2VecGensim:
    """
    使用 Gensim 实现的 Item2Vec

    更简洁，适合快速实验
    """

    def __init__(self, embed_dim=64, window=5, min_count=1,
                 workers=4, sg=1, negative=5):
        """
        参数:
            embed_dim: 嵌入维度
            window: 窗口大小
            min_count: 最小出现次数
            workers: 并行数
            sg: 1=Skip-gram, 0=CBOW
            negative: 负采样数
        """
        self.embed_dim = embed_dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.negative = negative

        self.model = None
        self.item_to_idx = {}
        self.idx_to_item = {}

    def fit(self, sequences: List[List]):
        """
        训练模型

        参数:
            sequences: 物品序列列表
        """
        # 构建索引
        all_items = set()
        for seq in sequences:
            all_items.update(seq)

        self.item_to_idx = {item: str(idx) for idx, item in enumerate(all_items)}
        self.idx_to_item = {str(idx): item for item, idx in self.item_to_idx.items()}

        # 转换为字符串
        str_sequences = [
            [self.item_to_idx[item] for item in seq]
            for seq in sequences
        ]

        # 训练 Word2Vec
        self.model = Word2Vec(
            sentences=str_sequences,
            vector_size=self.embed_dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            negative=self.negative
        )

        return self

    def get_embedding(self, item) -> np.ndarray:
        """获取物品嵌入"""
        if item not in self.item_to_idx:
            return None

        idx = self.item_to_idx[item]
        try:
            return self.model.wv[idx]
        except KeyError:
            return None

    def get_similar_items(self, item, top_k=10) -> List[Tuple]:
        """获取相似物品"""
        if item not in self.item_to_idx:
            return []

        idx = self.item_to_idx[item]
        try:
            similar = self.model.wv.most_similar(idx, topn=top_k)
            return [(self.idx_to_item[i], s) for i, s in similar]
        except KeyError:
            return []
```

## 4. Item2Vec 应用

### 4.1 推荐流程

```python
class Item2VecRecPipeline:
    """
    Item2Vec 推荐流水线
    """

    def __init__(self, embed_dim=64):
        self.embed_dim = embed_dim
        self.model = None
        self.item_embeddings = None

    def train(self, interactions):
        """训练"""
        sequences, item_to_idx, idx_to_item, item_counts = prepare_data(interactions)

        dataset = Item2VecDataset(sequences, item_counts=item_counts)
        train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        trainer = Item2VecTrainer(len(item_to_idx), self.embed_dim)
        self.model = trainer.train(train_loader, epochs=10)

        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.item_embeddings = self.model.get_all_embeddings()

    def recommend(self, user_history, top_k=10):
        """推荐"""
        # 获取用户历史嵌入
        user_emb = torch.zeros(self.embed_dim)
        count = 0

        for item in user_history:
            if item in self.item_to_idx:
                idx = self.item_to_idx[item]
                user_emb += self.model.get_item_embedding(idx)
                count += 1

        if count == 0:
            return []

        user_emb = user_emb / count

        # 计算相似度
        similarities = torch.matmul(self.item_embeddings, user_emb)

        # 排除历史物品
        for item in user_history:
            if item in self.item_to_idx:
                similarities[self.item_to_idx[item]] = -float('inf')

        # Top-K
        top_values, top_indices = torch.topk(similarities, top_k)

        return [(self.idx_to_item[int(idx)], float(score))
               for idx, score in zip(top_indices, top_values)]
```

## 5. 学习总结

### 5.1 核心要点

1. **Word2Vec 思想迁移**：将物品序列视为句子
2. **Skip-gram + 负采样**：高效训练
3. **语义相似性**：学习的嵌入捕获物品关系

### 5.2 优势与局限

**优势：**
- 实现简单
- 可扩展性好
- 嵌入质量高

**局限：**
- 忽略顺序信息
- 冷启动问题
- 需要足够数据

## 6. 练习题

1. 比较 Item2Vec 和 ItemCF 的推荐效果。

2. 实现带时间衰减的 Item2Vec。

3. 将 Item2Vec 与其他特征结合用于排序模型。
