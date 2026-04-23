# FunkSVD 学习文档

## 1. 算法基础认知

### 1.1 什么是 FunkSVD？

FunkSVD 是由 Simon Funk 在 2006 年 Netflix Prize 比赛中提出的一种矩阵分解方法。它的核心思想是**只对已知评分进行建模**，而不是像经典 SVD 那样要求完整的矩阵。

FunkSVD 也被称为 **SVD with Implicit Feedback** 或 **Regularized SVD**，是现代推荐系统中最基础的矩阵分解方法之一。

### 1.2 历史背景

2006 年，Netflix 举办了著名的 Netflix Prize 比赛，奖金 100 万美元。Simon Funk 在博客中公开了他的矩阵分解方法，这种方法：

1. 不需要完整的评分矩阵
2. 使用随机梯度下降优化
3. 添加正则化防止过拟合

这个简单而有效的方法震惊了整个推荐系统社区，奠定了现代矩阵分解方法的基础。

### 1.3 与经典 SVD 的区别

| 特性 | 经典 SVD | FunkSVD |
|------|----------|---------|
| 矩阵要求 | 完整矩阵 | 稀疏矩阵（只要求已知评分） |
| 分解形式 | R = UΣV^T | R ≈ PQ^T |
| 正交性 | U、V 正交 | P、Q 不一定正交 |
| 奇异值 | 显式计算 | 隐式包含在 P、Q 中 |
| 优化方法 | 特征值分解 | 梯度下降 |
| 适用场景 | 密集矩阵、降维 | 推荐系统、稀疏矩阵 |

### 1.4 直观理解

想象用户-物品评分矩阵：

```
         物品1  物品2  物品3  物品4  物品5
用户1     5      ?      3      ?      1
用户2     ?      4      ?      2      ?
用户3     3      ?      ?      5      ?
用户4     ?      2      4      ?      ?
```

FunkSVD 的目标：
1. 学习每个用户的隐因子向量（比如对动作、爱情、科幻等的偏好程度）
2. 学习每个物品的隐因子向量（比如物品的动作程度、爱情程度等）
3. 用户偏好 × 物品特征 = 预测评分

## 2. 核心原理

### 2.1 模型定义

FunkSVD 将预测评分建模为：

$$\hat{r}_{ui} = p_u \cdot q_i^T = \sum_{k=1}^{K} p_{uk} q_{ik}$$

其中：
- $p_u$：用户 u 的 K 维隐因子向量
- $q_i$：物品 i 的 K 维隐因子向量
- K：隐因子数量（超参数）

### 2.2 损失函数

FunkSVD 的损失函数（只对已知评分计算）：

$$J = \frac{1}{2} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - p_u \cdot q_i^T)^2 + \frac{\lambda}{2} (||p_u||^2 + ||q_i||^2)$$

其中：
- $\mathcal{K}$：已知评分的集合
- $r_{ui}$：用户 u 对物品 i 的真实评分
- $\lambda$：正则化系数

**关键区别**：只对已知评分 $(u,i) \in \mathcal{K}$ 计算损失，而不是对整个矩阵。

### 2.3 优化：随机梯度下降

对于每个已知评分 $(u, i)$：

**计算误差：**
$$e_{ui} = r_{ui} - p_u \cdot q_i^T$$

**计算梯度：**
$$\frac{\partial J}{\partial p_{uk}} = -e_{ui} \cdot q_{ik} + \lambda p_{uk}$$

$$\frac{\partial J}{\partial q_{ik}} = -e_{ui} \cdot p_{uk} + \lambda q_{ik}$$

**更新参数：**
$$p_{uk} \leftarrow p_{uk} + \alpha (e_{ui} \cdot q_{ik} - \lambda p_{uk})$$

$$q_{ik} \leftarrow q_{ik} + \alpha (e_{ui} \cdot p_{uk} - \lambda q_{ik})$$

其中 $\alpha$ 是学习率。

### 2.4 算法流程

```
输入：
  - 评分数据 {(u, i, r_ui)}
  - 隐因子数 K
  - 学习率 α
  - 正则化参数 λ
  - 迭代次数 T

输出：
  - 用户矩阵 P
  - 物品矩阵 Q

算法：
1. 随机初始化 P (|U| × K) 和 Q (|I| × K)
2. for t = 1 to T:
3.     打乱训练数据
4.     for each (u, i, r_ui) in 训练数据:
5.         e = r_ui - P[u] · Q[i]^T
6.         P[u] = P[u] + α(e · Q[i] - λ · P[u])
7.         Q[i] = Q[i] + α(e · P[u] - λ · Q[i])
8. 返回 P, Q
```

## 3. 数学公式与推导

### 3.1 损失函数详解

完整的损失函数：

$$J(P, Q) = \frac{1}{2} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \sum_{k=1}^{K} p_{uk} q_{ik})^2 + \frac{\lambda}{2} \sum_{u} \sum_{k} p_{uk}^2 + \frac{\lambda}{2} \sum_{i} \sum_{k} q_{ik}^2$$

三个部分：
1. **预测误差项**：使预测评分接近真实评分
2. **用户正则化项**：防止用户因子过大
3. **物品正则化项**：防止物品因子过大

### 3.2 梯度推导

令 $\hat{r}_{ui} = \sum_{k=1}^{K} p_{uk} q_{ik}$，$e_{ui} = r_{ui} - \hat{r}_{ui}$

对 $p_{uk}$ 求偏导：

$$\frac{\partial J}{\partial p_{uk}} = \sum_{(u,i) \in \mathcal{K}} \frac{\partial}{\partial p_{uk}} \left[ \frac{1}{2}(r_{ui} - \hat{r}_{ui})^2 \right] + \lambda p_{uk}$$

$$= \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui}) \cdot (-q_{ik}) + \lambda p_{uk}$$

$$= -\sum_{i: (u,i) \in \mathcal{K}} e_{ui} \cdot q_{ik} + \lambda p_{uk}$$

在 SGD 中，每次只考虑一个样本：

$$\frac{\partial J}{\partial p_{uk}} = -e_{ui} \cdot q_{ik} + \lambda p_{uk}$$

同理：

$$\frac{\partial J}{\partial q_{ik}} = -e_{ui} \cdot p_{uk} + \lambda q_{ik}$$

### 3.3 预测公式

训练完成后，预测用户 u 对物品 i 的评分：

$$\hat{r}_{ui} = \sum_{k=1}^{K} p_{uk} q_{ik} = p_u \cdot q_i^T$$

## 4. 训练过程讲解

### 4.1 数据准备

```python
import numpy as np
from collections import defaultdict

# 评分数据格式：[(user_id, item_id, rating), ...]
ratings = [
    (0, 0, 5), (0, 1, 3), (0, 3, 1),
    (1, 0, 4), (1, 3, 1),
    (2, 0, 1), (2, 1, 1), (2, 3, 5),
    (3, 2, 5), (3, 3, 4),
    (4, 1, 1), (4, 2, 5), (4, 3, 4),
]

# 转换为用户-物品-评分的字典结构（便于快速访问）
user_items = defaultdict(dict)
for u, i, r in ratings:
    user_items[u][i] = r

# 统计用户数和物品数
n_users = max(u for u, _, _ in ratings) + 1
n_items = max(i for _, i, _ in ratings) + 1
```

### 4.2 初始化

```python
def initialize_factors(n_users, n_items, n_factors, mean=0, std=0.1):
    """
    初始化用户和物品因子矩阵

    参数:
        n_users: 用户数量
        n_items: 物品数量
        n_factors: 隐因子数量
        mean: 初始化均值
        std: 初始化标准差

    返回:
        P, Q
    """
    # 使用较小的随机值初始化
    P = np.random.normal(mean, std, (n_users, n_factors))
    Q = np.random.normal(mean, std, (n_items, n_factors))

    return P, Q
```

### 4.3 训练循环

```python
def train_funksvd(ratings, n_factors=50, learning_rate=0.005,
                  reg_param=0.02, n_epochs=100, verbose=True):
    """
    训练 FunkSVD 模型

    参数:
        ratings: 评分列表 [(user_id, item_id, rating), ...]
        n_factors: 隐因子数量
        learning_rate: 学习率
        reg_param: 正则化参数
        n_epochs: 迭代次数
        verbose: 是否打印训练信息

    返回:
        P, Q, losses
    """
    # 统计维度
    n_users = max(u for u, _, _ in ratings) + 1
    n_items = max(i for _, i, _ in ratings) + 1

    # 初始化
    P, Q = initialize_factors(n_users, n_items, n_factors)

    # 转换为数组便于打乱
    ratings_array = np.array(ratings)
    losses = []

    for epoch in range(n_epochs):
        # 打乱数据
        np.random.shuffle(ratings_array)

        total_loss = 0

        for u, i, r in ratings_array:
            u, i = int(u), int(i)

            # 预测
            pred = np.dot(P[u], Q[i])

            # 误差
            error = r - pred
            total_loss += error ** 2

            # 更新（注意使用旧的 P 值更新 Q）
            P_old = P[u].copy()
            P[u] += learning_rate * (error * Q[i] - reg_param * P[u])
            Q[i] += learning_rate * (error * P_old - reg_param * Q[i])

        # 计算正则化损失
        reg_loss = reg_param * (np.sum(P ** 2) + np.sum(Q ** 2))
        total_loss += reg_loss

        # 计算训练集 RMSE
        rmse = np.sqrt(total_loss / len(ratings))
        losses.append(rmse)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, RMSE: {rmse:.4f}")

    return P, Q, losses
```

### 4.4 训练技巧

#### 学习率调度

```python
def get_learning_rate(initial_lr, epoch, decay_rate=0.95, decay_steps=10):
    """指数衰减学习率"""
    return initial_lr * (decay_rate ** (epoch // decay_steps))

def get_learning_rate_step(initial_lr, epoch, milestones=[50, 80], decay=0.1):
    """分段衰减学习率"""
    lr = initial_lr
    for m in milestones:
        if epoch >= m:
            lr *= decay
    return lr
```

#### 早停

```python
def train_with_early_stopping(train_ratings, val_ratings, patience=5,
                               n_factors=50, learning_rate=0.005,
                               reg_param=0.02, max_epochs=200):
    """带早停的训练"""
    # ... 初始化 ...

    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        # 训练一个 epoch
        # ...

        # 验证集评估
        val_rmse = evaluate(val_ratings, P, Q)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            # 保存最佳模型
            best_P, best_Q = P.copy(), Q.copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 epoch {epoch+1}")
                break

    return best_P, best_Q
```

## 5. 应用场景

### 5.1 适用场景

| 场景 | 适用性 | 说明 |
|------|--------|------|
| 电影推荐 | ★★★★★ | 经典应用，评分数据 |
| 音乐推荐 | ★★★★☆ | 可用于评分或播放量 |
| 电商推荐 | ★★★★☆ | 可用于评分或购买行为 |
| 新闻推荐 | ★★★☆☆ | 点击行为（需转换为隐式反馈） |

### 5.2 显式反馈 vs 隐式反馈

**显式反馈**（FunkSVD 直接适用）：
- 用户评分（1-5 星）
- 点赞/踩
- 满意度调查

**隐式反馈**（需要修改）：
- 点击、浏览
- 购买记录
- 停留时间

对于隐式反馈，通常：
1. 将行为转换为二值（0/1）或强度值
2. 使用置信度加权
3. 对负样本进行采样

## 6. 优缺点分析

### 6.1 优点

1. **简单有效**：算法直观，易于实现
2. **处理稀疏性**：只对已知评分建模
3. **可扩展**：容易添加额外特征
4. **预测快速**：训练后预测只需向量点积
5. **理论基础**：有良好的数学支撑

### 6.2 缺点

1. **冷启动**：无法处理新用户/新物品
2. **可解释性差**：隐因子难以解释
3. **超参数敏感**：需要调优 K、α、λ
4. **全局偏置**：没有考虑用户/物品的整体评分倾向
5. **单点估计**：不提供不确定性

### 6.3 与其他方法对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| FunkSVD | 简单、有效 | 无偏置项 |
| BiasSVD | 考虑偏置 | 略复杂 |
| SVD++ | 考虑隐式反馈 | 更复杂 |
| UserCF | 可解释 | 稀疏性差 |
| ItemCF | 可解释 | 冷启动差 |

## 7. 调库实现

### 7.1 使用 Surprise 库

```python
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV

# 加载数据
data = Dataset.load_builtin('ml-100k')

# 创建 FunkSVD 模型（Surprise 中的 SVD 就是 FunkSVD）
model = SVD(
    n_factors=100,      # 隐因子数量
    n_epochs=20,        # 迭代次数
    biased=False,       # 不使用偏置项（纯 FunkSVD）
    lr_all=0.005,       # 学习率
    reg_all=0.02        # 正则化参数
)

# 交叉验证
print("5折交叉验证:")
results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# 训练和预测
trainset, testset = train_test_split(data, test_size=0.25)
model.fit(trainset)
predictions = model.test(testset)

# 评估
print(f"\n测试集 RMSE: {accuracy.rmse(predictions):.4f}")
print(f"测试集 MAE: {accuracy.mae(predictions):.4f}")

# 单个预测
uid = str(196)  # 用户 ID
iid = str(302)  # 物品 ID
pred = model.predict(uid, iid)
print(f"\n用户 {uid} 对物品 {iid} 的预测评分: {pred.est:.2f}")
```

### 7.2 超参数调优

```python
from surprise.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_factors': [50, 100, 150],
    'n_epochs': [20, 30, 50],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}

# 网格搜索
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# 最佳参数
print("最佳 RMSE 参数:", gs.best_params['rmse'])
print("最佳 RMSE 分数:", gs.best_score['rmse'])

# 使用最佳模型
best_model = gs.best_estimator['rmse']
```

### 7.3 获取用户/物品因子

```python
def get_factors_from_surprise(model, trainset):
    """
    从 Surprise 模型提取因子矩阵

    返回:
        P: 用户因子矩阵（numpy 数组）
        Q: 物品因子矩阵（numpy 数组）
        user_id_map: 用户 ID 映射
        item_id_map: 物品 ID 映射
    """
    # 用户因子
    P = model.pu  # 内部用户因子矩阵

    # 物品因子
    Q = model.qi  # 内部物品因子矩阵

    # ID 映射
    user_id_map = {trainset.to_raw_uid(i): i for i in range(trainset.n_users)}
    item_id_map = {trainset.to_raw_iid(i): i for i in range(trainset.n_items)}

    return P, Q, user_id_map, item_id_map
```

## 8. 手工代码实现

### 8.1 完整的 FunkSVD 实现

```python
import numpy as np
from collections import defaultdict
import time

class FunkSVD:
    """
    FunkSVD 矩阵分解实现
    """

    def __init__(self, n_factors=100, learning_rate=0.005,
                 reg_param=0.02, n_epochs=100, random_state=42,
                 verbose=True):
        """
        参数:
            n_factors: 隐因子数量
            learning_rate: 学习率
            reg_param: 正则化参数
            n_epochs: 迭代次数
            random_state: 随机种子
            verbose: 是否打印训练信息
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

        # 模型参数
        self.P = None  # 用户因子矩阵
        self.Q = None  # 物品因子矩阵

        # ID 映射
        self.user_id_map = {}
        self.item_id_map = {}
        self.n_users = 0
        self.n_items = 0

        # 训练历史
        self.train_loss_history = []
        self.val_loss_history = []

    def _build_id_maps(self, user_ids, item_ids):
        """构建 ID 到索引的映射"""
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))

        self.user_id_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_id_map = {iid: i for i, iid in enumerate(unique_items)}
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

    def fit(self, user_ids, item_ids, ratings,
            val_user_ids=None, val_item_ids=None, val_ratings=None):
        """
        训练模型

        参数:
            user_ids: 用户 ID 列表
            item_ids: 物品 ID 列表
            ratings: 评分列表
            val_user_ids: 验证集用户 ID（可选）
            val_item_ids: 验证集物品 ID（可选）
            val_ratings: 验证集评分（可选）
        """
        np.random.seed(self.random_state)
        start_time = time.time()

        # 构建 ID 映射
        self._build_id_maps(user_ids, item_ids)

        # 初始化因子矩阵
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))

        # 准备训练数据
        train_data = []
        for u, i, r in zip(user_ids, item_ids, ratings):
            if u in self.user_id_map and i in self.item_id_map:
                train_data.append((self.user_id_map[u],
                                   self.item_id_map[i], r))
        train_data = np.array(train_data)

        # 准备验证数据
        val_data = None
        if val_user_ids is not None:
            val_data = []
            for u, i, r in zip(val_user_ids, val_item_ids, val_ratings):
                if u in self.user_id_map and i in self.item_id_map:
                    val_data.append((self.user_id_map[u],
                                     self.item_id_map[i], r))
            val_data = np.array(val_data) if val_data else None

        # 训练循环
        for epoch in range(self.n_epochs):
            # 打乱数据
            np.random.shuffle(train_data)

            total_loss = 0

            # SGD 更新
            for u_idx, i_idx, r in train_data:
                u_idx, i_idx = int(u_idx), int(i_idx)

                # 预测
                pred = np.dot(self.P[u_idx], self.Q[i_idx])

                # 误差
                error = r - pred
                total_loss += error ** 2

                # 更新（使用旧的 P 值）
                P_old = self.P[u_idx].copy()
                self.P[u_idx] += self.learning_rate * (
                    error * self.Q[i_idx] - self.reg_param * self.P[u_idx]
                )
                self.Q[i_idx] += self.learning_rate * (
                    error * P_old - self.reg_param * self.Q[i_idx]
                )

            # 正则化损失
            reg_loss = self.reg_param * (np.sum(self.P ** 2) + np.sum(self.Q ** 2))
            train_rmse = np.sqrt((total_loss + reg_loss) / len(train_data))
            self.train_loss_history.append(train_rmse)

            # 验证集评估
            if val_data is not None:
                val_rmse = self._compute_rmse(val_data)
                self.val_loss_history.append(val_rmse)

                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, "
                          f"Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, "
                          f"Train RMSE: {train_rmse:.4f}")

        train_time = time.time() - start_time
        if self.verbose:
            print(f"\n训练完成，耗时 {train_time:.2f} 秒")

        return self

    def _compute_rmse(self, data):
        """计算 RMSE"""
        errors = []
        for u_idx, i_idx, r in data:
            u_idx, i_idx = int(u_idx), int(i_idx)
            pred = np.dot(self.P[u_idx], self.Q[i_idx])
            errors.append((r - pred) ** 2)
        return np.sqrt(np.mean(errors))

    def predict(self, user_id, item_id):
        """
        预测用户对物品的评分
        """
        if user_id not in self.user_id_map:
            # 新用户，返回默认值
            return 0
        if item_id not in self.item_id_map:
            # 新物品，返回默认值
            return 0

        u_idx = self.user_id_map[user_id]
        i_idx = self.item_id_map[item_id]

        return np.dot(self.P[u_idx], self.Q[i_idx])

    def recommend(self, user_id, n_items=10, exclude_items=None):
        """
        为用户推荐物品

        参数:
            user_id: 用户 ID
            n_items: 推荐物品数量
            exclude_items: 要排除的物品列表

        返回:
            [(item_id, score), ...] 推荐列表
        """
        if user_id not in self.user_id_map:
            return []

        u_idx = self.user_id_map[user_id]
        user_factors = self.P[u_idx]

        # 计算所有物品的预测评分
        scores = np.dot(self.Q, user_factors)

        # 排除已知物品
        if exclude_items:
            for item_id in exclude_items:
                if item_id in self.item_id_map:
                    scores[self.item_id_map[item_id]] = -np.inf

        # 获取 top-n
        top_indices = np.argsort(scores)[::-1][:n_items]

        # 转换回原始 ID
        idx_to_item = {v: k for k, v in self.item_id_map.items()}
        recommendations = [
            (idx_to_item[idx], scores[idx]) for idx in top_indices
        ]

        return recommendations

    def get_similar_items(self, item_id, n_items=10):
        """
        获取相似物品（基于余弦相似度）
        """
        if item_id not in self.item_id_map:
            return []

        i_idx = self.item_id_map[item_id]
        item_vec = self.Q[i_idx]

        # 计算余弦相似度
        norms = np.linalg.norm(self.Q, axis=1)
        item_norm = norms[i_idx]

        if item_norm < 1e-10:
            return []

        similarities = np.dot(self.Q, item_vec) / (norms * item_norm + 1e-10)
        similarities[i_idx] = -np.inf  # 排除自身

        # 获取 top-n
        top_indices = np.argsort(similarities)[::-1][:n_items]

        idx_to_item = {v: k for k, v in self.item_id_map.items()}
        similar_items = [
            (idx_to_item[idx], similarities[idx]) for idx in top_indices
        ]

        return similar_items

    def get_user_embedding(self, user_id):
        """获取用户嵌入向量"""
        if user_id not in self.user_id_map:
            return None
        return self.P[self.user_id_map[user_id]]

    def get_item_embedding(self, item_id):
        """获取物品嵌入向量"""
        if item_id not in self.item_id_map:
            return None
        return self.Q[self.item_id_map[item_id]]


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)

    # 创建评分数据
    user_ids = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    item_ids = [0, 1, 2, 3, 0, 2, 3, 1, 2, 3, 0, 1, 3, 1, 2, 3]
    ratings = [5, 3, 4, 1, 4, 5, 2, 3, 4, 5, 2, 4, 3, 1, 5, 4]

    # 划分训练集和验证集
    indices = np.random.permutation(len(user_ids))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_users = [user_ids[i] for i in train_idx]
    train_items = [item_ids[i] for i in train_idx]
    train_ratings = [ratings[i] for i in train_idx]

    val_users = [user_ids[i] for i in val_idx]
    val_items = [item_ids[i] for i in val_idx]
    val_ratings = [ratings[i] for i in val_idx]

    # 训练模型
    print("=" * 50)
    print("FunkSVD 训练")
    print("=" * 50)

    model = FunkSVD(
        n_factors=10,
        learning_rate=0.01,
        reg_param=0.1,
        n_epochs=100,
        verbose=True
    )

    model.fit(
        train_users, train_items, train_ratings,
        val_users, val_items, val_ratings
    )

    # 预测
    print("\n" + "=" * 50)
    print("预测示例")
    print("=" * 50)
    print(f"用户 0 对物品 3 的预测评分: {model.predict(0, 3):.2f}")

    # 推荐
    print("\n" + "=" * 50)
    print("推荐示例")
    print("=" * 50)
    recs = model.recommend(0, n_items=5, exclude_items=[0, 1, 2])
    print("为用户 0 推荐:")
    for item_id, score in recs:
        print(f"  物品 {item_id}: 预测评分 {score:.2f}")

    # 相似物品
    print("\n" + "=" * 50)
    print("相似物品")
    print("=" * 50)
    similar = model.get_similar_items(0, n_items=3)
    print("与物品 0 相似的物品:")
    for item_id, sim in similar:
        print(f"  物品 {item_id}: 相似度 {sim:.4f}")
```

## 9. 可视化与结果理解

### 9.1 训练过程可视化

```python
import matplotlib.pyplot as plt

def plot_training_history(model):
    """绘制训练历史"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model.train_loss_history, 'b-', label='Train RMSE')
    if model.val_loss_history:
        plt.plot(model.val_loss_history, 'r-', label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # 学习曲线（训练误差 vs 模型复杂度）
    factors_range = [10, 20, 50, 100, 200]
    train_errors = []
    val_errors = []

    # ... 实验不同 n_factors 的效果 ...

    plt.tight_layout()
    plt.show()
```

### 9.2 隐因子可视化

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_latent_space(model, method='pca'):
    """可视化隐因子空间"""
    if method == 'pca' and model.n_factors > 2:
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = None

    plt.figure(figsize=(14, 6))

    # 用户空间
    plt.subplot(1, 2, 1)
    if reducer:
        user_2d = reducer.fit_transform(model.P)
    else:
        user_2d = model.P[:, :2]

    plt.scatter(user_2d[:, 0], user_2d[:, 1], c='blue', alpha=0.6)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('User Latent Space')
    plt.grid(True, alpha=0.3)

    # 物品空间
    plt.subplot(1, 2, 2)
    if reducer:
        item_2d = reducer.fit_transform(model.Q)
    else:
        item_2d = model.Q[:, :2]

    plt.scatter(item_2d[:, 0], item_2d[:, 1], c='red', alpha=0.6)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Item Latent Space')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

### 9.3 预测分布可视化

```python
def visualize_predictions(model, test_data):
    """可视化预测结果"""
    actual = []
    predicted = []

    for u, i, r in test_data:
        actual.append(r)
        predicted.append(model.predict(u, i))

    actual = np.array(actual)
    predicted = np.array(predicted)

    plt.figure(figsize=(12, 5))

    # 分布对比
    plt.subplot(1, 2, 1)
    plt.hist(actual, bins=20, alpha=0.7, label='Actual', color='blue')
    plt.hist(predicted, bins=20, alpha=0.7, label='Predicted', color='red')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution')
    plt.legend()

    # 预测 vs 实际
    plt.subplot(1, 2, 2)
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Actual vs Predicted')
    plt.legend()

    plt.tight_layout()
    plt.show()
```

## 10. 模型评估

### 10.1 评估函数

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_data):
    """
    全面的模型评估

    参数:
        model: 训练好的模型
        test_data: 测试数据 [(user_id, item_id, rating), ...]

    返回:
        评估指标字典
    """
    actual = []
    predicted = []

    for user_id, item_id, rating in test_data:
        pred = model.predict(user_id, item_id)
        actual.append(rating)
        predicted.append(pred)

    actual = np.array(actual)
    predicted = np.array(predicted)

    # 回归指标
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    # 覆盖率（预测了多少用户-物品对）
    valid_predictions = sum(1 for p in predicted if not np.isnan(p))

    # R² 分数
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Valid_Predictions': valid_predictions,
        'Total_Predictions': len(test_data)
    }


def evaluate_ranking(model, test_data, k=10):
    """
    排序质量评估

    参数:
        model: 训练好的模型
        test_data: 字典格式 {user_id: [item_id1, item_id2, ...]}
        k: 推荐列表长度

    返回:
        排序指标
    """
    precisions = []
    recalls = []
    ndcgs = []

    for user_id, relevant_items in test_data.items():
        if not relevant_items:
            continue

        # 获取推荐
        recs = model.recommend(user_id, n_items=k)
        recommended = [item_id for item_id, _ in recs]

        # Precision@K
        n_relevant = len(set(recommended) & set(relevant_items))
        precision = n_relevant / k if k > 0 else 0
        precisions.append(precision)

        # Recall@K
        recall = n_relevant / len(relevant_items) if relevant_items else 0
        recalls.append(recall)

        # NDCG@K
        dcg = 0
        for i, item_id in enumerate(recommended):
            if item_id in relevant_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    return {
        f'Precision@{k}': np.mean(precisions),
        f'Recall@{k}': np.mean(recalls),
        f'NDCG@{k}': np.mean(ndcgs)
    }
```

## 11. 常见问题与易错点

### 11.1 常见问题

**Q1：隐因子数量 K 如何选择？**

A：通过交叉验证选择：
- 太小：欠拟合，无法捕捉复杂模式
- 太大：过拟合，计算量增加
- 通常在 20-200 之间

**Q2：学习率太大或太小会怎样？**

A：
- 太大：训练不稳定，损失震荡甚至发散
- 太小：收敛太慢，可能陷入局部最优

**Q3：如何处理新用户/新物品？**

A：
1. 基于内容/人口统计学特征初始化
2. 使用混合推荐策略
3. 增量更新模型

### 11.2 易错点

1. **更新顺序错误**：更新 Q 时必须使用旧的 P 值
2. **忘记正则化**：导致过拟合
3. **学习率太大**：训练不稳定
4. **初始化太大**：初始误差过大
5. **未验证冷启动**：对未知用户/物品返回异常值

## 12. 学习总结

### 12.1 核心要点

1. **FunkSVD = SVD + 稀疏性处理**：只对已知评分建模
2. **优化方法**：随机梯度下降（SGD）
3. **正则化**：L2 正则化防止过拟合
4. **预测**：用户向量 × 物品向量

### 12.2 知识图谱

```
FunkSVD
├── 核心概念
│   ├── 隐因子
│   ├── 矩阵分解
│   └── 只对已知评分建模
├── 优化
│   ├── SGD
│   ├── 正则化
│   └── 学习率调度
├── 扩展
│   ├── BiasSVD
│   ├── SVD++
│   └── TimeSVD
└── 应用
    ├── 评分预测
    ├── Top-N 推荐
    └── 相似物品
```

## 13. 练习题与思考题

### 13.1 基础题

1. **（填空）** FunkSVD 只对 ______ 进行建模，而不像经典 SVD 那样要求完整矩阵。

2. **（判断）** FunkSVD 中的用户因子矩阵 P 和物品因子矩阵 Q 是正交矩阵。（ ）

3. **（简答）** 解释 FunkSVD 中正则化项的作用。

### 13.2 进阶题

4. **（推导）** 推导 FunkSVD 中用户因子 $p_{uk}$ 的更新公式。

5. **（编程）** 实现一个带学习率衰减的 FunkSVD。

6. **（分析）** 比较不同隐因子数量 K 对模型性能的影响。

### 13.3 思考题

7. FunkSVD 如何与现代深度学习模型（如 Neural CF）结合？

8. 如何将 FunkSVD 扩展到隐式反馈场景？

9. 在实际工业应用中，FunkSVD 面临哪些挑战？

### 参考答案

1. 已知评分

2. 错误。FunkSVD 的 P 和 Q 不是正交矩阵，这与经典 SVD 不同。

3. 正则化项通过惩罚大的因子值来防止过拟合，提高模型在新数据上的泛化能力。

4. 见第 3 节数学公式推导。

5-6. 提示：参考本文档的代码实现。

## 14. 学习路径建议

### 14.1 前置知识

- [ ] 矩阵分解基础
- [ ] 梯度下降优化
- [ ] Python/NumPy 编程

### 14.2 学习顺序

1. **理解原理** → 阅读 Simon Funk 的原始博客
2. **动手实现** → 手写 FunkSVD
3. **调库实践** → 使用 Surprise 库
4. **参数调优** → 实验不同参数组合
5. **学习扩展** → 学习 BiasSVD、SVD++

### 14.3 推荐资源

- **博客**：Simon Funk 的 Netflix Prize 博客（2006）
- **论文**：Matrix Factorization Techniques for Recommender Systems (Koren et al., 2009)
- **书籍**：《推荐系统实践》- 项亮
- **课程**：Coursera - Recommender Systems

### 14.4 下一步学习

- **BiasSVD**：添加用户/物品偏置
- **SVD++**：考虑隐式反馈
- **深度学习**：Neural CF、DeepFM 等
