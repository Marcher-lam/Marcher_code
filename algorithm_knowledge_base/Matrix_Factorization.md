# Matrix Factorization 学习文档

## 1. 算法基础认知

### 1.1 定义

Matrix Factorization（矩阵分解）是一种经典的协同过滤算法，其核心思想是将**用户-物品交互矩阵**分解为两个低秩矩阵的乘积：

$$
\mathbf{R} \approx \mathbf{U} \cdot \mathbf{V}^T
$$

其中：
- $\mathbf{R} \in \mathbb{R}^{M \times N}$：用户-物品评分矩阵（$M$ 用户，$N$ 物品）
- $\mathbf{U} \in \mathbb{R}^{M \times K}$：用户隐因子矩阵
- $\mathbf{V} \in \mathbb{R}^{N \times K}$：物品隐因子矩阵
- $K$：隐因子维度（通常 $K \ll \min(M, N)$）

### 1.2 直观类比

将 Matrix Factorization 想象为**发现潜在兴趣**：比如一个用户喜欢"科幻"和"动作"电影，我们不知道这个属性，但算法可以通过评分数据自动学习到这些隐因子。

### 1.3 历史背景

- **2006**：Netflix Prize 比赛，SVD++ 一举夺冠
- **2007**：随后的比赛中引入偏置、时序等
- 现在：推荐系统中仍然广泛使用

---

## 2. 核心原理

### 2.1 基本模型

预测评分：

$$
\hat{r}_{ui} = \mathbf{u}_i^T \mathbf{v}_j = \sum_{k=1}^{K} u_{ik} \cdot v_{jk}
$$

### 2.2 带偏置的版本

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_i^T \mathbf{v}_j
$$

其中：
- $\mu$：全局平均评分
- $b_u$：用户偏置
- $b_i$：物品偏置

### 2.3 SVD++（考虑隐式反馈）

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{q}_i^T \left(\mathbf{p}_u + \frac{1}{\sqrt{|N(u)|}} \sum_{j \in N(u)} \mathbf{y}_j\right)
$$

其中 $N(u)$ 是用户 $u$ 有过交互的物品集合。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $\mathbf{R}$ | 评分矩阵 | $(M, N)$ |
| $\mathbf{U}$ | 用户矩阵 | $(M, K)$ |
| $\mathbf{V}$ | 物品矩阵 | $(N, K)$ |
| $r_{ui}$ | 评分 | 标量 |
| $\mathcal{K}$ | 已知评分集合 | - |

### 3.2 目标函数（带正则化）

$$
\min_{\mathbf{U}, \mathbf{V}} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mathbf{u}_i^T \mathbf{v}_j)^2 + \lambda \left(\lVert \mathbf{U} \Vert_F^2 + \lVert \mathbf{V} \Vert_F^2\right)
$$

### 3.3 交替最小二乘法（ALS）

```python
# 更新 U：固定 V，求 min
# (V^T V + lambda*I) U = V^T R

# 更新 V：固定 U，求 min
# (U^T U + lambda*I) V = U^T R^T
```

---

## 4. 训练过程讲解

### 4.1 PyTorch 实现

```python
import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    """矩阵分解"""
    
    def __init__(self, num_users, num_items, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 偏置
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        """
        user_ids: [B]
        item_ids: [B]
        """
        # 嵌入
        u = self.user_embedding(user_ids)  # [B, K]
        v = self.item_embedding(item_ids)  # [B, K]
        
        # 偏置
        b_u = self.user_bias(user_ids).squeeze(-1)  # [B]
        b_i = self.item_bias(item_ids).squeeze(-1)  # [B]
        
        # 预测
        dot = (u * v).sum(dim=-1)  # [B]
        pred = self.global_bias + b_u + b_i + dot
        
        return pred
    
    def get_user_embedding(self, user_id):
        return self.user_embedding(user_id)
    
    def get_item_embedding(self, item_id):
        return self.item_embedding(item_id)
```

### 4.2 训练循环

```python
import torch.optim as optim

def train_mf():
    """训练矩阵分解模型"""
    
    # 创建模型
    model = MatrixFactorization(num_users=1000, num_items=500, embedding_dim=50)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 损失
    criterion = nn.MSELoss()
    
    # 训练
    for epoch in range(10):
        total_loss = 0
        
        for batch in dataloader:
            user_ids, item_ids, ratings = batch
            
            optimizer.zero_grad()
            
            # 前向
            preds = model(user_ids, item_ids)
            loss = criterion(preds, ratings)
            
            # 反向
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model

train_mf()
```

### 4.3 推荐生成

```python
def generate_recommendations(model, user_id, top_k=10):
    """生成推荐"""
    
    model.eval()
    
    with torch.no_grad():
        # 获取所有物品的评分预测
        all_items = torch.arange(num_items)
        user_ids = torch.full((num_items,), user_id)
        
        scores = model(user_ids, all_items)
        
        # 排序
        _, indices = torch.topk(scores, top_k)
        
        return indices.tolist()

generate_recommendations(model, user_id=0, top_k=10)
```

---

## 5. 应用场景

### 5.1 推荐系统

矩阵分解的主要应用：
- 电影推荐（MovieLens）
- 商品推荐（电商）
- 音乐推荐（Spotify）

### 5.2 评分预测

预测用户对物品的评分：
- 个性化服务
- A/B 测试

### 5.3 缺失值填补

填补缺失数据：
- 医疗数据
- 问卷数据

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 可扩展 | 处理大规模数据 |
| 泛化 | 隐因子可解释 |
| 稀疏友好 | 只需存储非零元素 |
| 效果好 | 推荐系统常用 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 冷启动 | 新用户/物品难处理 |
| 隐式反馈 | 需处理缺失值 |
| 超参数 | K 需要调参 |

---

## 7. 调库实现

### 7.1 Surprise 库

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# 加载数据
data = Dataset.load_from_file('ratings.csv')
trainset = data.build_full_trainset()

# 训练
algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005)
algo.fit(trainset)

# 预测
pred = algo.predict('user_id', 'item_id')
print(f"预测评分: {pred.est}")
```

### 7.2 Implicit 库

```python
import implicit

# 创建模型
model = implicit.als.AlternatingLeastSquares(
    factors=50,
    regularization=0.01,
    iterations=15
)

# 训练（稀疏矩阵）
model.fit(user_item_matrix)

# 推荐
item_ids, scores = model.recommend(user_id, user_item_matrix[user_id])
```

### 7.3 完全训练流程

```python
def full_training_pipeline():
    """完整训练流程"""
    
    import numpy as np
    from scipy.sparse import csr_matrix
    from surprise import Dataset, Reader, SVD
    
    # 1. 加载数据
    # data = load_ratings()
    
    # 2. 构建数据集
    # reader = Reader(rating_scale=(1, 5))
    # dataset = Dataset.load_from_arrays(ratings, reader)
    
    # 3. ���练
    # algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    # trainset = dataset.build_full_trainset()
    
    # 4. 预测
    # predictions = algo.test(testset)
    
    # 5. 评估
    # accuracy.rmse(predictions)
    
    print("训练流程就绪")
    return True

full_training_pipeline()
```

---

## 8. 手工代码实现

### 8.1 基础 ALS 实现

```python
import numpy as np

class ManualALS:
    """手动实现交替最小二乘法"""
    
    def __init__(self, num_users, num_items, n_factors=50, reg=0.1, n_iters=10):
        self.n_factors = n_factors
        self.reg = reg
        self.n_iters = n_iters
        
        # 初始化
        self.U = np.random.randn(num_users, n_factors) * 0.1
        self.V = np.random.randn(num_items, n_factors) * 0.1
        
        # 偏置
        self.b_u = np.zeros(num_users)
        self.b_i = np.zeros(num_items)
        self.global_mean = 0
    
    def fit(self, R):
        """训练
        
        R: 稀疏矩阵或评分列表 [(user, item, rating)]
        """
        # 收集评分
        ratings = []
        for u, i, r in R:
            ratings.append((u, i, r))
        
        # 迭代
        for it in range(self.n_iters):
            print(f"Iter {it+1}/{self.n_iters}")
            
            # 更新 U
            self._update_U(ratings)
            
            # 更新 V
            self._update_V(ratings)
    
    def _update_U(self, ratings):
        """更新用户矩阵"""
        for u in range(len(self.U)):
            items = [(i, r) for u_, i, r in ratings if u_ == u]
            if not items:
                continue
            
            # 构建 V_i
            V_i = self.V[[i for i, r in items]]
            r_i = np.array([r for i, r in items])
            r_pred = self.b_u[u] + self.b_i[[i for i, r in items]] + (V_i @ self.U[u])
            
            # 求解
            A = V_i.T @ V_i + self.reg * np.eye(self.n_factors)
            b = V_i.T @ (r_i - self.b_u[u] - self.b_i[[i for i, r in items]])
            
            self.U[u] = np.linalg.solve(A, b)
    
    def _update_V(self, ratings):
        """更新物品矩阵"""
        for i in range(len(self.V)):
            users = [(u, r) for u_, i_, r in ratings if i_ == i]
            if not users:
                continue
            
            U_i = self.U[[u for u, r in users]]
            r_i = np.array([r for u, r in users])
            
            A = U_i.T @ U_i + self.reg * np.eye(self.n_factors)
            b = U_i.T @ (r_i - self.global_mean - self.b_u[[u for u, r in users]])
            
            self.V[i] = np.linalg.solve(A, b)
    
    def predict(self, u, i):
        """预测"""
        return self.global_mean + self.b_u[u] + self.b_i[i] + self.U[u] @ self.V[i]

# 测试
# als = ManualALS(100, 50)
# als.fit(ratings)
```

### 8.2 带偏置的版本

```python
class BiasedMF:
    """带偏置的矩阵分解"""
    
    def __init__(self, n_users, n_items, k=50, lr=0.005, reg=0.02, n_epochs=20):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        
        # 初始化
        self.U = np.random.randn(n_users, k) * 0.1
        self.V = np.random.randn(n_items, k) * 0.1
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.mu = 0
    
    def fit(self, R, n_items):
        """训练"""
        ratings = [(u, i, r) for u, i, r in R]
        self.mu = np.mean([r for u, i, r in ratings])
        
        for epoch in range(self.n_epochs):
            for u, i, r in ratings:
                # 预测
                pred = self.predict_one(u, i)
                error = r - pred
                
                # 更新
                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])
                
                u_f = self.U[u].copy()
                v_f = self.V[i].copy()
                
                self.U[u] += self.lr * (error * v_f - self.reg * u_f)
                self.V[i] += self.lr * (error * u_f - self.reg * v_f)
    
    def predict_one(self, u, i):
        return self.mu + self.b_u[u] + self.b_i[i] + self.U[u] @ self.V[i]
    
    def predict(self, user_items):
        return [self.predict_one(u, i) for u, i in user_items]

# 验证
# mf = BiasedMF(100, 50, k=20)
# mf.fit(ratings, 50)
```

---

## 9. 可视化与结果理解

### 9.1 隐因子可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_embeddings():
    """可视化隐因子"""
    
    # 假设已有用户和物品嵌入
    np.random.seed(42)
    user_emb = np.random.randn(100, 2)
    item_emb = np.random.randn(50, 2)
    
    plt.figure(figsize=(10, 5))
    
    # 绘制
    plt.scatter(user_emb[:, 0], user_emb[:, 1], c='blue', alpha=0.5, label='Users')
    plt.scatter(item_emb[:, 0], item_emb[:, 1], c='red', alpha=0.5, label='Items')
    
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.legend()
    plt.title('Matrix Factorization Embeddings')
    plt.savefig('mf_embeddings.png', dpi=150)
    plt.show()

visualize_embeddings()
```

### 9.2 评分分布

```python
def plot_rating_distribution():
    """绘制评分分布"""
    
    ratings = np.random.randint(1, 6, 1000)
    
    plt.figure(figsize=(8, 4))
    plt.hist(ratings, bins=5, edgecolor='black')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig('rating_dist.png', dpi=150)
    plt.show()

plot_rating_distribution()
```

---

## 10. 模型评估

### 10.1 常用指标

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_mf():
    """评估矩阵分解模型"""
    
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)
    
    # 计算
    y_true = np.array([3, 4, 5, 4])
    y_pred = np.array([3.2, 3.8, 4.7, 4.1])
    
    print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"MAE: {mae(y_true, y_pred):.4f}")
    
    return rmse(y_true, y_pred)

evaluate_mf()
```

---

## 11. 常见问题与易错点

### 11.1 冷启动问题

**问题**：新用户/物品没有评分？

**解答**：
- 使用内容特征
- 混合推荐

### 11.2 过拟合

**问题**：训练集表现好但测试集差？

**解答**：
- 增加正则化
- 交叉验证

### 11.3 稀疏性

**问题**：数据稀疏？

**解答**：使用 SGD、ALS 只处理非零元素。

---

## 12. 学习总结

### 12.1 核心要点

1. **矩阵分解**：$\mathbf{R} \approx \mathbf{U} \mathbf{V}^T$
2. **隐因子**：学习用户/物品的低维表示
3. **优化**：SGD、ALS
4. **偏置**：可加用户/物品偏置

### 12.2 扩展

- **SVD++**：考虑隐式反馈
- **ALS**：交替最小二乘
- **BPR**：贝叶斯个性化排序

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：实现带偏置的矩阵分解。

### 13.2 思考题

**思考题**：隐因子 K 如何选择？

---

## 14. 学习路径建议

### 14.1 第一阶段

1. 理解协同过滤
2. 理解矩阵分解

### 14.2 第二阶段

1. 实现 SGD/ALS
2. 添加偏置

### 14.3 第三��段

1. 实际应用
2. 对比方法

### 14.4 推荐资源

- **论文**: Netflix Prize
- **书籍**: 《推荐系统实践》

---

*Matrix Factorization 是推荐系统中里程碑式的算法，它的低维嵌入思想深刻影响了后续的深度学习推荐系统。*