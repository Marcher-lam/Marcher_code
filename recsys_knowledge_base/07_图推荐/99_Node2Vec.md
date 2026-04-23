# Node2Vec 学习文档

## 1. 算法基础认知

### 1.1 什么是 Node2Vec？

Node2Vec 是一种**图嵌入算法**，通过**带偏置的随机游走**来学习图节点的向量表示。它是 DeepWalk 的改进版本，由 Aditya Grover 和 Jure Leskovec 在 2016 年的 KDD 上发表。

### 1.2 核心创新

**DeepWalk 的问题：**
- 随机游走完全随机，无法控制探索策略
- 无法同时捕获局部和全局结构

**Node2Vec 的改进：**
- 引入两个参数 p 和 q 控制游走策略
- 可以在 BFS（广度优先）和 DFS（深度优先）之间平衡
- 更灵活地捕获不同的网络结构

### 1.3 应用场景

- 社交网络分析
- 推荐系统（用户-物品二部图）
- 知识图谱
- 生物信息学

## 2. 核心原理

### 2.1 游走策略

Node2Vec 定义了两种游走策略：

1. **BFS（广度优先）**：倾向于探索**邻近节点**
   - 捕获局部结构（如社区）

2. **DFS（深度优先）**：倾向于探索**远处节点**
   - 捕获全局结构（如节点角色）

### 2.2 参数控制

- **p（返回参数 Return Parameter）**：控制回到上一个节点的概率
  - p < 1：更可能返回
  - p > 1：更可能向前

- **q（出入参数 In-out Parameter）**：控制探索远近节点
  - q < 1：偏向 BFS
  - q > 1：偏向 DFS

### 2.3 转移概率

从节点 t 游走到 v 后，下一步的转移概率：

$$
\alpha_{pq}(t, x) = \begin{cases}
\frac{1}{p} & \text{if } d_{tx} = 0 \text{ (回到 t)} \\
1 & \text{if } d_{tx} = 1 \text{ (距离1的邻居)} \\
\frac{1}{q} & \text{if } d_{tx} = 2 \text{ (距离2的节点)}
\end{cases}
$$

其中 $d_{tx}$ 是节点 t 和 x 之间的最短距离。

## 3. 完整实现

### 3.1 Node2Vec 模型

```python
import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import networkx as nx
from gensim.models import Word2Vec


class Node2Vec:
    """
    Node2Vec: Scalable Feature Learning for Networks

    论文: node2vec: Scalable Feature Learning for Networks (KDD 2016)
    """

    def __init__(self, graph, p=1.0, q=1.0, walk_length=80,
                 num_walks=10, window_size=5, embed_dim=128,
                 min_count=1, workers=4):
        """
        参数:
            graph: NetworkX 图
            p: 返回参数
            q: 出入参数
            walk_length: 游走长度
            num_walks: 每个节点的游走次数
            window_size: Skip-gram 窗口大小
            embed_dim: 嵌入维度
            min_count: 最小出现次数
            workers: 并行数
        """
        self.graph = graph
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.min_count = min_count
        self.workers = workers

        self.model = None
        self._precompute_transition_probs()

    def _precompute_transition_probs(self):
        """
        预计算转移概率

        为每条边计算转移概率，使用 Alias 采样优化
        """
        self.alias_nodes = {}
        self.alias_edges = {}

        # 计算节点的归一化转移概率
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                # 均匀分布
                unnormalized_probs = [1.0] * len(neighbors)
                norm_const = sum(unnormalized_probs)
                normalized_probs = [p / norm_const for p in unnormalized_probs]

                self.alias_nodes[node] = self._create_alias_table(
                    normalized_probs, neighbors
                )

        # 计算边的转移概率
        for edge in self.graph.edges():
            self._precompute_edge_transition_probs(edge[0], edge[1])
            if not self.graph.is_directed():
                self._precompute_edge_transition_probs(edge[1], edge[0])

    def _precompute_edge_transition_probs(self, src, dst):
        """预计算边的转移概率"""
        neighbors = list(self.graph.neighbors(dst))
        if not neighbors:
            return

        src_neighbors = set(self.graph.neighbors(src))

        unnormalized_probs = []
        for neighbor in neighbors:
            weight = self.graph[dst][neighbor].get('weight', 1.0)

            if neighbor == src:
                # 回到上一个节点
                unnormalized_probs.append(weight / self.p)
            elif neighbor in src_neighbors:
                # 距离为1（共同邻居）
                unnormalized_probs.append(weight)
            else:
                # 距离为2
                unnormalized_probs.append(weight / self.q)

        norm_const = sum(unnormalized_probs)
        normalized_probs = [p / norm_const for p in unnormalized_probs]

        self.alias_edges[(src, dst)] = self._create_alias_table(
            normalized_probs, neighbors
        )

    def _create_alias_table(self, probs: List[float],
                           items: List) -> Dict:
        """
        创建 Alias 采样表

        Alias 方法可以在 O(1) 时间内进行加权采样

        参数:
            probs: 概率列表
            items: 对应的物品列表

        返回:
            alias_table: {prob, alias, items}
        """
        n = len(probs)
        prob = np.array(probs) * n

        smaller = []
        larger = []

        for i, p in enumerate(prob):
            if p < 1.0:
                smaller.append(i)
            else:
                larger.append(i)

        alias = [0] * n

        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()

            prob[small] = 1.0
            alias[small] = large

            prob[large] = prob[large] + prob[small] - 1.0

            if prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return {
            'prob': prob,
            'alias': alias,
            'items': items
        }

    def _alias_sample(self, alias_table: Dict) -> any:
        """
        Alias 采样

        O(1) 时间的加权采样
        """
        n = len(alias_table['items'])
        i = random.randint(0, n - 1)

        if random.random() < alias_table['prob'][i]:
            return alias_table['items'][i]
        else:
            return alias_table['items'][alias_table['alias'][i]]

    def _node2vec_walk(self, start_node: any) -> List:
        """
        执行一次 Node2Vec 游走

        参数:
            start_node: 起始节点

        返回:
            walk: 游走序列
        """
        walk = [start_node]

        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = list(self.graph.neighbors(current))

            if not neighbors:
                break

            if len(walk) == 1:
                # 第一步：均匀采样
                next_node = self._alias_sample(self.alias_nodes[current])
            else:
                # 后续步骤：使用 Node2Vec 采样
                prev = walk[-2]
                edge_key = (prev, current)

                if edge_key in self.alias_edges:
                    next_node = self._alias_sample(self.alias_edges[edge_key])
                else:
                    # 如果没有预计算，使用节点采样
                    next_node = self._alias_sample(self.alias_nodes[current])

            walk.append(next_node)

        return walk

    def generate_walks(self) -> List[List]:
        """
        生成所有游走序列

        返回:
            walks: 游走序列列表
        """
        print("生成游走序列...")
        walks = []
        nodes = list(self.graph.nodes())

        for walk_iter in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._node2vec_walk(node)
                walks.append(walk)

        print(f"生成 {len(walks)} 条游走序列")
        return walks

    def fit(self):
        """
        训练 Node2Vec 模型

        返回:
            self
        """
        # 生成游走序列
        walks = self.generate_walks()

        # 转换为字符串（gensim 要求）
        walks_str = [[str(node) for node in walk] for walk in walks]

        # 使用 Word2Vec 学习嵌入
        print("训练 Word2Vec...")
        self.model = Word2Vec(
            walks_str,
            vector_size=self.embed_dim,
            window=self.window_size,
            min_count=self.min_count,
            sg=1,  # Skip-gram
            workers=self.workers
        )

        return self

    def get_embedding(self, node) -> np.ndarray:
        """
        获取节点嵌入

        参数:
            node: 节点

        返回:
            嵌入向量
        """
        try:
            return self.model.wv[str(node)]
        except KeyError:
            return None

    def get_all_embeddings(self) -> Dict:
        """获取所有节点嵌入"""
        embeddings = {}
        for node in self.graph.nodes():
            emb = self.get_embedding(node)
            if emb is not None:
                embeddings[node] = emb
        return embeddings

    def get_similar_nodes(self, node, top_k=10) -> List[Tuple]:
        """
        获取相似节点

        参数:
            node: 节点
            top_k: 返回数量

        返回:
            [(node, similarity), ...]
        """
        try:
            similar = self.model.wv.most_similar(str(node), topn=top_k)
            return [(n, s) for n, s in similar]
        except KeyError:
            return []


class Node2VecForRecommendation(Node2Vec):
    """
    用于推荐的 Node2Vec

    专门针对用户-物品二部图优化
    """

    def __init__(self, graph, p=1.0, q=1.0, **kwargs):
        super().__init__(graph, p, q, **kwargs)

        # 识别用户和物品节点
        self.user_nodes = set()
        self.item_nodes = set()

        for node in graph.nodes():
            if isinstance(node, str):
                if node.startswith('u_'):
                    self.user_nodes.add(node)
                elif node.startswith('i_'):
                    self.item_nodes.add(node)

    def recommend(self, user_id: str, top_k: int = 10,
                  exclude_items: set = None) -> List[Tuple]:
        """
        为用户推荐物品

        参数:
            user_id: 用户 ID
            top_k: 推荐 K 个
            exclude_items: 排除的物品

        返回:
            [(item_id, score), ...]
        """
        user_node = f'u_{user_id}' if not user_id.startswith('u_') else user_id
        user_emb = self.get_embedding(user_node)

        if user_emb is None:
            return []

        exclude = exclude_items or set()

        # 计算与所有物品的相似度
        scores = []
        for item_node in self.item_nodes:
            if item_node in exclude:
                continue

            item_emb = self.get_embedding(item_node)
            if item_emb is not None:
                similarity = np.dot(user_emb, item_emb) / (
                    np.linalg.norm(user_emb) * np.linalg.norm(item_emb)
                )
                # 去掉前缀
                item_id = item_node[2:] if item_node.startswith('i_') else item_node
                scores.append((item_id, similarity))

        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


def create_user_item_graph(interactions: List[Tuple],
                          user_prefix: str = 'u_',
                          item_prefix: str = 'i_') -> nx.Graph:
    """
    创建用户-物品二部图

    参数:
        interactions: [(user_id, item_id, weight), ...]
        user_prefix: 用户节点前缀
        item_prefix: 物品节点前缀

    返回:
        NetworkX Graph
    """
    graph = nx.Graph()

    for user_id, item_id, weight in interactions:
        user_node = f'{user_prefix}{user_id}'
        item_node = f'{item_prefix}{item_id}'

        graph.add_node(user_node, type='user')
        graph.add_node(item_node, type='item')
        graph.add_edge(user_node, item_node, weight=weight)

    return graph


# 使用示例
def demo_node2vec():
    """Node2Vec 示例"""
    # 创建图
    interactions = [
        ('u1', 'i1', 1), ('u1', 'i2', 1), ('u1', 'i3', 1),
        ('u2', 'i1', 1), ('u2', 'i4', 1),
        ('u3', 'i2', 1), ('u3', 'i3', 1), ('u3', 'i5', 1),
        ('u4', 'i4', 1), ('u4', 'i5', 1), ('u4', 'i6', 1),
        ('u5', 'i1', 1), ('u5', 'i6', 1),
    ]

    graph = create_user_item_graph(interactions)

    print(f"节点数: {graph.number_of_nodes()}")
    print(f"边数: {graph.number_of_edges()}")

    # 训练 Node2Vec
    # BFS 模式 (p=1, q=0.5)
    print("\n=== BFS 模式 ===")
    model_bfs = Node2VecForRecommendation(graph, p=1, q=0.5,
                                          walk_length=20, num_walks=10,
                                          embed_dim=32)
    model_bfs.fit()

    # DFS 模式 (p=1, q=2)
    print("\n=== DFS 模式 ===")
    model_dfs = Node2VecForRecommendation(graph, p=1, q=2,
                                          walk_length=20, num_walks=10,
                                          embed_dim=32)
    model_dfs.fit()

    # 推荐
    print("\n=== BFS 模式推荐 ===")
    recs_bfs = model_bfs.recommend('u4', top_k=3)
    for item, score in recs_bfs:
        print(f"  {item}: {score:.4f}")

    print("\n=== DFS 模式推荐 ===")
    recs_dfs = model_dfs.recommend('u4', top_k=3)
    for item, score in recs_dfs:
        print(f"  {item}: {score:.4f}")


if __name__ == "__main__":
    demo_node2vec()
```

### 3.2 参数调优

```python
class Node2VecParamTuner:
    """
    Node2Vec 参数调优器
    """

    def __init__(self, graph):
        self.graph = graph

    def grid_search(self, p_values: List, q_values: List,
                   eval_func=None) -> Dict:
        """
        网格搜索

        参数:
            p_values: p 值列表
            q_values: q 值列表
            eval_func: 评估函数

        返回:
            最佳参数和结果
        """
        results = {}
        best_score = -np.inf
        best_params = None

        for p in p_values:
            for q in q_values:
                print(f"\n测试 p={p}, q={q}")

                model = Node2Vec(self.graph, p=p, q=q,
                               walk_length=30, num_walks=5,
                               embed_dim=64)
                model.fit()

                if eval_func:
                    score = eval_func(model)
                else:
                    # 使用默认评估
                    score = self._default_eval(model)

                results[(p, q)] = score

                if score > best_score:
                    best_score = score
                    best_params = (p, q)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

    def _default_eval(self, model) -> float:
        """默认评估：链路预测"""
        from sklearn.metrics import roc_auc_score

        # 简化：计算已知边的平均相似度
        scores = []
        for edge in list(self.graph.edges())[:100]:
            emb1 = model.get_embedding(edge[0])
            emb2 = model.get_embedding(edge[1])
            if emb1 is not None and emb2 is not None:
                sim = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                scores.append(sim)

        return np.mean(scores) if scores else 0


# 参数选择建议
PARAMETER_GUIDE = {
    'BFS_like': {'p': 1, 'q': 0.5, 'effect': '捕获局部结构'},
    'DFS_like': {'p': 1, 'q': 2, 'effect': '捕获全局结构'},
    'balanced': {'p': 1, 'q': 1, 'effect': '平衡局部和全局'},
    'return_biased': {'p': 0.5, 'q': 1, 'effect': '倾向于返回'},
    'explore_biased': {'p': 2, 'q': 1, 'effect': '倾向于探索'},
}
```

## 4. Node2Vec vs DeepWalk

### 4.1 对比

| 维度 | DeepWalk | Node2Vec |
|------|----------|----------|
| 游走策略 | 均匀随机 | 带偏置 |
| 参数 | 无 | p, q |
| 灵活性 | 低 | 高 |
| 结构捕获 | 中等 | 可控 |

### 4.2 适用场景

**Node2Vec 适合：**
- 需要控制探索策略
- 图结构复杂
- 对不同结构有不同需求

**DeepWalk 适合：**
- 快速基线
- 不需要调参
- 结构简单的图

## 5. 调参建议

### 5.1 参数选择

| 目标 | p | q | 说明 |
|------|---|---|------|
| 社区发现 | 1 | 0.5 | BFS 模式 |
| 角色发现 | 1 | 2 | DFS 模式 |
| 平衡 | 1 | 1 | 等同于 DeepWalk |

### 5.2 其他参数

| 参数 | 推荐值 |
|------|--------|
| walk_length | 40-80 |
| num_walks | 5-10 |
| embed_dim | 64-128 |

## 6. 学习总结

### 6.1 核心要点

1. **p 和 q 控制游走策略**：BFS vs DFS
2. **Alias 采样优化**：O(1) 采样
3. **结合 Skip-gram**：学习节点嵌入

### 6.2 关键创新

- 带偏置的随机游走
- 灵活的参数控制
- 高效的实现

## 7. 练习题

1. 比较 Node2Vec 在不同 p、q 参数下的嵌入质量。

2. 实现 Node2Vec 的并行化版本。

3. 将 Node2Vec 应用于推荐系统并评估效果。
