# Neural CF（神经协同过滤）学习文档

> 用深度学习替代矩阵分解的推荐算法

---

## 1. 算法基础认知

### 1.1 一句话定义

Neural CF（Neural Collaborative Filtering，神经协同过滤）是由He等人在2017年提出的推荐模型，用神经网络替代传统的矩阵分解（MF），同时学习用户和项目的非线性特征交互。

### 1.2 直觉类比

Neural CF就像把"协同过滤"放进神经网络。传统的矩阵分解MF像是用简单的"两个向量相乘"来预测；Neural CF则用神经网络来自动学习更复杂的交互模式——可能不只是"点积"，而是更复杂的函数关系！

想象你在交友网站做匹配推荐：
- 传统MF：用户A和用户B的兴趣向量点积 > 阈值 → 匹配
- Neural CF：把用户A和用户B的所有信息（年龄、兴趣、地区等）一起送入神经网络，让网络自动学习复杂的"匹配规则"——比如"北方人+90后+喜欢电影"的组合和"南方人+80后+喜欢音乐"的组合有特殊关系！

### 1.3 发展背景

- 2017年，He等人在WWW会议发表"Neural Collaborative Filtering"
- 作为NCF框架的核心论文，被引用5000+
- 后续引出GMF、NeuMF等重要变体

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 推荐系统 → 协同过滤 |
| 输出 | 预测评分/点击率 |
| 模型类型 | 神经网络 |
| 特点 | 非线性交互学习 |

---

## 2. 核心原理

### 2.1 为什么需要Neural CF？

**传统矩阵分解MF的局限**：
- 只能学习线性的用户-物品交互
- 无法捕捉复杂的非线性模式
- 对稀疏数据效果下降

**Neural CF的优势**：
- 可以学习任意复杂的交互函数
- 对稀疏数据更鲁棒
- 端到端训练

### 2.2 vs 传统CF对比

| 方法 | 交互函数 | 表达能力 | 复杂度 |
|------|-----------|----------|--------|
| 矩阵分解MF | $r_{ui} = p_u \cdot q_i$ | 线性 | 低 |
| SVD++ | 加入隐式反馈 | 线性+隐式 | 中 |
| **Neural CF** | **神经网络** | **非线性** | **高** |

### 2.3 架构流程

```
用户ID → User Embedding ──┐
                       │  拼接
项目ID → Item Embedding ┘
              │
              ▼
        ┌──────────────────┐
        │  Multi-Layer      │
        │   Perceptron     │
        │ (MLP隐藏层)      │
        └──────┬───────────┘
               │
               ▼
          Sigmoid
               │
              输出
        评分概率 r̂_ui
```

### 2.4 核心思想

Neural CF使用多层感知机（MLP）来学习用户和物品之间的非线性交互，而不是简单的点积。

---

## 3. 数学公式与推导

### 3.1 嵌入层

用户和物品分别嵌入到低维空间：

$$P_u = E_u \cdot user_u$$
$$Q_i = E_i \cdot item_i$$

其中 $E_u \in \mathbb{R}^{d \times |U|}$，$E_i \in \mathbb{R}^{d \times |I|}$

### 3.2 特征拼接

将用户和物品嵌入拼接：

$$x = [P_u; Q_i] \in \mathbb{R}^{2d}$$

### 3.3 MLP前向传播

$$
h^{(1)} = \sigma(W^{(1)} x + b^{(1)})\\
h^{(2)} = \sigma(W^{(2)} h^{(1)} + b^{(2)})\\
\cdots\\
h^{(L)} = \sigma(W^{(L)} h^{(L-1)} + b^{(L)})
$$

其中激活函数通常为ReLU。

### 3.4 输出层

预测评分概率：

$$\hat{r}_{ui} = \sigma(w^T h^{(L)} + b)$$

或者二分类输出表示点击概率。

### 3.5 损失函数

使用Binary Cross-Entropy（二分类）或MSE（回归）：

$$L = -\sum_{(u,i) \in \mathcal{D}^+} \log \hat{r}_{ui} - \sum_{(u,i) \in \mathcal{D}^-} \log (1 - \hat{r}_{ui})$$

---

## 4. 训练过程讲解

### 4.1 数据准备

```python
# 交互数据格式
# user_id, item_id, rating/interaction

# 正样本：用户交互过的物品
positive_samples = [(u1, i1), (u2, i2), ...]

# 负样本：用户未交互的物品
negative_samples = [(u1, i_neg), ...]

# 训练集
train_data = positive_samples + negative_samples
train_labels = [1]*len(positive_samples) + [0]*len(negative_samples)
```

### 4.2 负采样策略

由于未交互的物品数量巨大，需要负采样：

```python
def neg_sampling(positive_samples, all_items, negative_ratio=4):
    """负采样：每个正样本采样negative_ratio个负样本"""
    negatives = []
    
    for user, item in positive_samples:
        # 随机采样用户未交互的物品
        user_neg_items = random.sample(
            [i for i in all_items if i not in user_interactions[user]],
            negative_ratio
        )
        for neg_item in user_neg_items:
            negatives.append((user, neg_item))
    
    return negatives
```

### 4.3 训练配置

```python
# 训练参数
config = {
    'embed_dim': 32,           # 嵌入维度
    'mlp_dims': [64, 32, 16], # MLP各层维度
    'lr': 0.001,             # 学习率
    'batch_size': 256,
    'epochs': 20,
    'dropout': 0.2,
    'num_neg_samples': 4      # 每个正样本负采样数
}
```

---

## 5. 应用场景

### 5.1 用户推荐

```python
# 给用户推荐物品
def recommend(model, user_id, items, top_k=10):
    model.eval()
    
    scores = []
    for item_id in items:
        score = model.predict(user_id, item_id)
        scores.append((item_id, score))
    
    # 排序
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
```

### 5.2 评分预测

```python
# 预测用户对物品的评分
predicted_rating = model.predict(user_id, item_id)
print(f"预测评分: {predicted_rating:.2f}")
```

### 5.3 排序学习

```python
# 预测并排序候选物品
candidates = [...]

predictions = model.predict_batch([user_id]*len(candidates), candidates)
ranked = sorted(zip(candidates, predictions), key=lambda x: x[1], reverse=True)
```

### 5.4 对比传统方法

| 方法 | MovieLens@HR@10 | 数据集 |
|------|----------------|--------|
| MF | 0.65 | MovieLens |
| SVD++ | 0.68 | MovieLens |
| **Neural CF** | **0.72** | MovieLens |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 非线性建模 | 能学习复杂交互 |
| 端到端 | 无需特征工程 |
| 灵活性 | 可加更多特征 |
| 泛化能力 | 对稀疏数据鲁棒 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 计算复杂 | 比MF慢 |
| 超参数多 | 需要调参 |
| 可解释性弱 | 神经网络黑箱 |
| 显存需求 | 嵌入+MLP |

### 6.3 注意事项

- embed_dim不宜太大，32-64足够
- MLP深度2-3层即可
- 负采样比例一般为4:1

---

## 7. 调库实现（Python）

### 7.1 PyTorch完整实现

```python
import torch
import torch.nn as nn

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, mlp_dims=[64, 32, 16], dropout=0.2):
        super().__init__()
        
        # 嵌入层
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        
        # MLP层
        mlp_layers = []
        input_dim = embed_dim * 2
        
        for dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 输出层
        self.output = nn.Linear(mlp_dims[-1], 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, user, item):
        # 嵌入
        u_emb = self.user_embed(user)
        i_emb = self.item_embed(item)
        
        # 拼接
        x = torch.cat([u_emb, i_emb], dim=-1)
        
        # MLP
        x = self.mlp(x)
        
        # 输出
        output = torch.sigmoid(self.output(x))
        
        return output.squeeze(-1)
    
    def predict(self, user, item):
        """预测"""
        with torch.no_grad():
            return self.forward(user, item)


# 训练函数
def train_neural_cf(model, train_loader, val_loader, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_val_auc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for user, item, label in train_loader:
            optimizer.zero_grad()
            
            output = model(user, item)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for user, item, label in val_loader:
                pred = model(user, item)
                val_preds.extend(pred.numpy())
                val_labels.extend(label.numpy())
        
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_preds)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}")
    
    return best_val_auc
```

### 7.2 变体：GMF实现

```python
class GMF(nn.Module):
    """Generalized Matrix Factorization - 简化版Neural CF"""
    
    def __init__(self, num_users, num_items, embed_dim=32):
        super().__init__()
        
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.output = nn.Linear(embed_dim, 1)
    
    def forward(self, user, item):
        u_emb = self.user_embed(user)
        i_emb = self.item_embed(item)
        
        # 元素级乘法
        interaction = u_emb * i_emb
        
        output = torch.sigmoid(self.output(interaction))
        return output.squeeze(-1)
```

### 7.3 变体：NeuMF实现

```python
class NeuMF(nn.Module):
    """NeuMF: 结合GMF和MLP"""
    
    def __init__(self, num_users, num_items, embed_dim=32, mlp_dims=[64, 32]):
        super().__init__()
        
        # GMF部分
        self.gmf_user = nn.Embedding(num_users, embed_dim)
        self.gmf_item = nn.Embedding(num_items, embed_dim)
        
        # MLP部分
        self.mlp_user = nn.Embedding(num_users, embed_dim)
        self.mlp_item = nn.Embedding(num_items, embed_dim)
        
        mlp = []
        input_dim = embed_dim * 2
        for dim in mlp_dims:
            mlp.extend([nn.Linear(input_dim, dim), nn.ReLU()])
            input_dim = dim
        self.mlp = nn.Sequential(*mlp)
        
        # 输出融合
        self.output = nn.Linear(embed_dim + mlp_dims[-1], 1)
    
    def forward(self, user, item):
        # GMF路径
        gmf_out = self.gmf_user(user) * self.gmf_item(item)
        
        # MLP路径
        mlp_in = torch.cat([self.mlp_user(user), self.mlp_item(item)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        
        # 融合
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        output = torch.sigmoid(self.output(combined))
        
        return output.squeeze(-1)
```

### 7.4 训练示例

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 准备数据（示例）
num_users, num_items = 1000, 500
interactions = np.random.rand(5000, 3)  # user, item, rating

train_data, test_data = train_test_split(interactions, test_size=0.2)

# 创建模型
model = NeuralCF(num_users, num_items)

# 训练
train_auc = train_neural_cf(model, train_data, test_data)
print(f"Train AUC: {train_auc:.4f}")
```

---

## 8. 手工代码实现（理解原理）

```python
import numpy as np

class NeuralCFManual:
    """简化版Neural CF - 理解原理"""
    
    def __init__(self, num_users, num_items, embed_dim=8, hidden_dim=16, lr=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.lr = lr
        
        # 嵌入
        self.P = np.random.randn(num_users, embed_dim) * 0.01
        self.Q = np.random.randn(num_items, embed_dim) * 0.01
        
        # MLP权重
        self.W1 = np.random.randn(embed_dim*2, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros(1)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, user, item):
        # 嵌入
        u_emb = self.P[user]
        i_emb = self.Q[item]
        
        # 拼接
        x = np.concatenate([u_emb, i_emb])
        
        # MLP
        h = self.relu(x @ self.W1 + self.b1)
        
        # 输出
        score = self.sigmoid(h @ self.W2 + self.b2)
        
        return score.flatten()
    
    def predict(self, user, item):
        return self.forward(user, item)
    
    def train_step(self, user, item, label):
        """单步训练"""
        # 前向
        score = self.forward(user, item)
        
        # 损失（简化版）
        error = label - score
        
        # 梯度（简化）
        # 这里简化处理，实际应更复杂
        
        return score


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成数据
    num_users, num_items = 100, 50
    num_samples = 1000
    
    users = np.random.randint(0, num_users, num_samples)
    items = np.random.randint(0, num_items, num_samples)
    labels = np.random.randint(0, 2, num_samples)
    
    # 训练
    model = NeuralCFManual(num_users, num_items)
    
    for epoch in range(10):
        total_loss = 0
        for i in range(num_samples):
            pred = model.train_step(users[i], items[i], labels[i])
            total_loss += abs(labels[i] - pred)
        
        print(f"Epoch {epoch}: Loss={total_loss/num_samples:.4f}")
    
    # 预测
    test_user, test_item = 0, 0
    pred = model.predict(test_user, test_item)
    print(f"预测: {pred:.3f}")
```

---

## 9. 评估与可视化

### 9.1 评估指标

| 指标 | 说明 | 计算 |
|------|------|------|
| HR@K | Top-K命中率 | 命中的测试样本/K |
| NDCG | 归一化折扣累积增益 | 2^rel-1/log2(pos+2) |
| AUC | 排序质量 | (M-1/12)/M |
| Recall@K | K召回率 | TP/(TP+FN) |

### 9.2 评估代码

```python
from sklearn.metrics import roc_auc_score, precision_score, recall_score

def evaluate(model, test_data):
    # 预测
    predictions = []
    actuals = []
    
    for user, item, label in test_data:
        pred = model.predict(user, item)
        predictions.append(pred)
        actuals.append(label)
    
    # 计算指标
    auc = roc_auc_score(actuals, predictions)
    
    # Top-K推荐
    hits = 0
    for user, item, label in test_data:
        top_k = recommend(model, user, all_items, k=10)
        if item in top_k:
            hits += 1
    
    hr10 = hits / len(test_data)
    
    return {'AUC': auc, 'HR@10': hr10}


# 计算Hit Rate
def compute_hr(model, test_data, all_items, k=10):
    hits = 0
    
    for user, item, label in test_data:
        if label == 1:
            recommendations = recommend(model, user, all_items, k=k)
            if item in recommendations:
                hits += 1
    
    return hits / sum(1 for _, _, l in test_data if l == 1)
```

### 9.3 可视化嵌入

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_embeddings(model):
    """可视化用户和物品嵌入"""
    
    # 降维
    pca = PCA(n_components=2)
    
    # 用户嵌入
    user_2d = pca.fit_transform(model.P)
    
    # 物品嵌入
    item_2d = pca.transform(model.Q)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    plt.scatter(user_2d[:, 0], user_2d[:, 1], c='blue', alpha=0.5, label='用户')
    plt.scatter(item_2d[:, 0], item_2d[:, 1], c='red', alpha=0.5, label='物品')
    plt.legend()
    plt.title('Neural CF嵌入可视化')
    plt.savefig('neural_cf_embeddings.png', dpi=100)
    plt.show()
```

---

## 10. 常见问题与易错点

### Q1: 如何选择嵌入维度？

**答案**：32-64维足够。过大会过拟合。

### Q2: 需要多少训练数据？

**答案**：建议正样本至少1000+。

### Q3: 负采样比例多少合适？

**答案**：通常4:1到10:1。

### Q4: GMF和MLP哪个更好？

**答案**：GMF快速但表达能力有限，MLP更灵活。NeuMF结合两者。

### Q5: 为什么训练不稳定？

**答��**：学习率太高或嵌入未初始化好。尝试更小的lr或预训练初始化。

---

## 11. 学习总结

### 11.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心 | 神经网络学习交互 |
| 输入 | 用户+物品嵌入 |
| 隐藏 | MLP层 |
| 输出 | 预测概率 |

### 11.2 公式汇总

嵌入：
$$P_u = Embed(user), Q_i = Embed(item)$$

拼接：
$$x = [P_u; Q_i]$$

MLP：
$$h = \sigma(Wx + b)$$

输出：
$$\hat{r} = \sigma(w^T h + b)$$

---

## 12. 练习题

### 12.1 选择题

1. Neural CF相比MF的优势是：
   - A) 计算更快
   - B) 能学习非线性
   - C) 更少参数

2. Neural CF的输出通常是：
   - A) 回归值
   - B) 分类概率
   - C) 排序

### 12.2 简答题

1. 解释为什么需要负采样？
2. 比较GMF和MLP的区别。

### 12.3 编程题

1. 实现NeuMF并对比效果。
2. 在MovieLens数据集上测试。

---

## 13. 学习路径建议

### 13.1 进阶路径

```
协同过滤基础
    ↓
矩阵分解MF
    ↓
Neural CF
    ↓
NeuMF/DeepFM
    ↓
图神经网络推荐
```

### 13.2 相关算法

| 算法 | 关系 |
|------|------|
| GMF | Neural CF简化版 |
| NeuMF | 结合GMF和MLP |
| DeepFM | 加入特征交互 |
| NGCF | 图神经网络版 |

### 13.3 扩展阅读

- He et al. (2017). Neural Collaborative Filtering. WWW.

---

## 附录

### 参考

1. He et al. (2017). Neural Collaborative Filtering. WWW.
2. NCF GitHub:He XiangN/Neural-CF

---

**文档结束**

## 9. 可视化与结果理解

### 推荐可视化
1. **训练曲线**：损失随训练轮次变化，观察收敛趋势
2. **性能对比**：Neural_CF与基准方法对比
3. **特征重要性**（如适用）：各特征贡献度

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.title('Neural_CF Training Loss')
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

