# BiasSVD 学习文档

## 1. 算法基础认知

### 1.1 什么是 BiasSVD？

BiasSVD（也称为 SVD with Bias）是在 FunkSVD 基础上加入偏置项（Bias）的矩阵分解方法。它考虑了用户和物品的整体评分倾向，而不仅仅是用户对特定物品的个性化偏好。

### 1.2 为什么需要偏置项？

**问题场景**：

| 用户 | 电影A | 电影B | 电影C |
|------|-------|-------|-------|
| 张三 | 5 | 4 | 5 |
| 李四 | 3 | 2 | 3 |
| 王五 | 4 | 3 | ? |

观察：
- **张三**：给分普遍较高，是"宽容评分者"
- **李四**：给分普遍较低，是"严格评分者"
- **电影A**：整体评分较高，是"热门好片"
- **电影B**：整体评分较低，是"普通影片"

**FunkSVD 的问题**：
- 只建模用户-物品交互 $\hat{r}_{ui} = p_u \cdot q_i^T$
- 无法区分"严格用户给高分"和"宽容用户给低分"的含义
- 将整体偏好和个性化偏好混淆

**BiasSVD 的解决方案**：
$$\hat{r}_{ui} = \mu + b_u + b_i + p_u \cdot q_i^T$$

- $\mu$：全局平均评分
- $b_u$：用户偏置（用户的整体评分倾向）
- $b_i$：物品偏置（物品的整体受欢迎程度）
- $p_u \cdot q_i^T$：用户-物品的个性化交互

### 1.3 直观理解

```
预测评分 = 全局均值 + 用户偏置 + 物品偏置 + 个性化交互

例如：预测张三对电影A的评分
= 3.5 (全局均值)
+ 1.0 (张三偏置：比平均水平高1分)
+ 0.5 (电影A偏置：比平均电影高0.5分)
+ 0.3 (个性化偏好：张三特别喜欢这类电影)
= 5.3
```

### 1.4 与 FunkSVD 的区别

| 特性 | FunkSVD | BiasSVD |
|------|---------|---------|
| 预测公式 | $p_u \cdot q_i^T$ | $\mu + b_u + b_i + p_u \cdot q_i^T$ |
| 参数量 | P, Q | P, Q, $\mu$, $b_u$, $b_i$ |
| 偏置建模 | 无 | 有 |
| 准确性 | 较低 | 较高 |
| 可解释性 | 较低 | 较高 |

## 2. 核心原理

### 2.1 模型定义

BiasSVD 的预测公式：

$$\hat{r}_{ui} = \mu + b_u + b_i + p_u \cdot q_i^T$$

其中：
- $\mu$：全局平均评分（所有已知评分的均值）
- $b_u$：用户 u 的偏置（可正可负）
- $b_i$：物品 i 的偏置（可正可负）
- $p_u$：用户 u 的 K 维隐因子向量
- $q_i$：物品 i 的 K 维隐因子向量

### 2.2 各项含义

| 项 | 含义 | 示例 |
|----|------|------|
| $\mu$ | 整体评分水平 | 全局平均 3.5 分 |
| $b_u$ | 用户的评分习惯 | 严格用户 $b_u = -0.5$，宽容用户 $b_u = +1.0$ |
| $b_i$ | 物品的受欢迎程度 | 热门电影 $b_i = +0.8$，冷门电影 $b_i = -0.3$ |
| $p_u \cdot q_i^T$ | 个性化偏好 | 用户对这类物品的特殊喜好 |

### 2.3 损失函数

$$J = \frac{1}{2} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui})^2 + \frac{\lambda}{2} (||P||^2 + ||Q||^2 + \sum_u b_u^2 + \sum_i b_i^2)$$

正则化项包括：
- 用户因子矩阵 P
- 物品因子矩阵 Q
- 用户偏置 $b_u$
- 物品偏置 $b_i$

**注意**：全局均值 $\mu$ 通常不加正则化（它是数据的统计量）。

### 2.4 梯度与更新规则

**误差计算：**
$$e_{ui} = r_{ui} - \hat{r}_{ui} = r_{ui} - (\mu + b_u + b_i + p_u \cdot q_i^T)$$

**梯度：**
$$\frac{\partial J}{\partial b_u} = -e_{ui} + \lambda b_u$$

$$\frac{\partial J}{\partial b_i} = -e_{ui} + \lambda b_i$$

$$\frac{\partial J}{\partial p_{uk}} = -e_{ui} \cdot q_{ik} + \lambda p_{uk}$$

$$\frac{\partial J}{\partial q_{ik}} = -e_{ui} \cdot p_{uk} + \lambda q_{ik}$$

**更新规则：**
$$b_u \leftarrow b_u + \alpha (e_{ui} - \lambda b_u)$$

$$b_i \leftarrow b_i + \alpha (e_{ui} - \lambda b_i)$$

$$p_{uk} \leftarrow p_{uk} + \alpha (e_{ui} \cdot q_{ik} - \lambda p_{uk})$$

$$q_{ik} \leftarrow q_{ik} + \alpha (e_{ui} \cdot p_{uk} - \lambda q_{ik})$$

### 2.5 算法流程

```
输入：
  - 评分数据 {(u, i, r_ui)}
  - 隐因子数 K
  - 学习率 α
  - 正则化参数 λ
  - 迭代次数 T

输出：
  - 全局均值 μ
  - 用户偏置 b_u
  - 物品偏置 b_i
  - 用户矩阵 P
  - 物品矩阵 Q

算法：
1. 计算全局均值 μ = mean(所有已知评分)
2. 初始化 b_u = 0, b_i = 0
3. 随机初始化 P 和 Q
4. for t = 1 to T:
5.     打乱训练数据
6.     for each (u, i, r_ui):
7.         e = r_ui - (μ + b_u + b_i + P[u]·Q[i]^T)
8.         b_u = b_u + α(e - λ·b_u)
9.         b_i = b_i + α(e - λ·b_i)
10.        P[u] = P[u] + α(e·Q[i] - λ·P[u])
11.        Q[i] = Q[i] + α(e·P_old[u] - λ·Q[i])
12. 返回 μ, b, P, Q
```

## 3. 数学公式与推导

### 3.1 损失函数展开

$$J = \frac{1}{2} \sum_{(u,i) \in \mathcal{K}} \left( r_{ui} - \mu - b_u - b_i - \sum_{k=1}^{K} p_{uk} q_{ik} \right)^2 + \frac{\lambda}{2} \left( \sum_{u,k} p_{uk}^2 + \sum_{i,k} q_{ik}^2 + \sum_u b_u^2 + \sum_i b_i^2 \right)$$

### 3.2 梯度推导详解

**对 $b_u$ 的偏导：**

$$\frac{\partial J}{\partial b_u} = \sum_{(u,i) \in \mathcal{K}_u} \frac{\partial}{\partial b_u} \left[ \frac{1}{2}(r_{ui} - \hat{r}_{ui})^2 \right] + \lambda b_u$$

其中 $\mathcal{K}_u$ 是用户 u 的所有已知评分。

对于单个样本 $(u, i)$：

$$\frac{\partial}{\partial b_u} \left[ \frac{1}{2}(r_{ui} - \hat{r}_{ui})^2 \right] = (r_{ui} - \hat{r}_{ui}) \cdot \frac{\partial}{\partial b_u}[-\hat{r}_{ui}]$$

$$= (r_{ui} - \hat{r}_{ui}) \cdot (-1) = -e_{ui}$$

因此：

$$\frac{\partial J}{\partial b_u} = -e_{ui} + \lambda b_u$$

### 3.3 为什么 Bias 有效？

**数学解释：**

假设评分的真实生成过程包含偏置：
$$r_{ui} = \mu + b_u + b_i + \epsilon_{ui}$$

其中 $\epsilon_{ui}$ 是个性化的残差。

如果模型只使用 $p_u \cdot q_i^T$ 来拟合，它需要同时学习：
1. 全局均值 $\mu$
2. 用户偏置 $b_u$
3. 物品偏置 $b_i$
4. 个性化交互 $\epsilon_{ui}$

这增加了学习难度。显式建模偏置后，隐因子只需学习真正的个性化交互。

**实验验证：**

在 Netflix Prize 中，添加偏置项通常能降低 RMSE 约 0.01-0.02，这在比赛中是显著的提升。

## 4. 训练过程讲解

### 4.1 初始化

```python
import numpy as np

def initialize_bias_svd(n_users, n_items, n_factors, ratings):
    """
    初始化 BiasSVD 模型参数
    """
    # 全局均值
    mu = np.mean(ratings)

    # 偏置初始化为 0
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)

    # 隐因子矩阵小随机初始化
    P = np.random.normal(0, 0.1, (n_users, n_factors))
    Q = np.random.normal(0, 0.1, (n_items, n_factors))

    return mu, b_u, b_i, P, Q
```

### 4.2 训练循环

```python
def train_bias_svd(ratings_data, n_factors=50, learning_rate=0.005,
                   reg_param=0.02, n_epochs=100, verbose=True):
    """
    训练 BiasSVD 模型

    参数:
        ratings_data: [(user_id, item_id, rating), ...]
        n_factors: 隐因子数量
        learning_rate: 学习率
        reg_param: 正则化参数
        n_epochs: 迭代次数
        verbose: 是否打印信息

    返回:
        mu, b_u, b_i, P, Q
    """
    # 统计
    user_ids = [r[0] for r in ratings_data]
    item_ids = [r[1] for r in ratings_data]
    ratings = [r[2] for r in ratings_data]

    n_users = max(user_ids) + 1
    n_items = max(item_ids) + 1

    # 初始化
    mu, b_u, b_i, P, Q = initialize_bias_svd(
        n_users, n_items, n_factors, ratings
    )

    # 转换为数组
    data = np.array(ratings_data)
    n_samples = len(data)

    for epoch in range(n_epochs):
        # 打乱数据
        np.random.shuffle(data)

        total_loss = 0

        for u, i, r in data:
            u, i = int(u), int(i)

            # 预测
            pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])

            # 误差
            error = r - pred
            total_loss += error ** 2

            # 更新偏置
            b_u[u] += learning_rate * (error - reg_param * b_u[u])
            b_i[i] += learning_rate * (error - reg_param * b_i[i])

            # 更新隐因子
            P_old = P[u].copy()
            P[u] += learning_rate * (error * Q[i] - reg_param * P[u])
            Q[i] += learning_rate * (error * P_old - reg_param * Q[i])

        # 计算正则化损失
        reg_loss = reg_param * (
            np.sum(P ** 2) + np.sum(Q ** 2) +
            np.sum(b_u ** 2) + np.sum(b_i ** 2)
        )
        rmse = np.sqrt((total_loss + reg_loss) / n_samples)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, RMSE: {rmse:.4f}")

    return mu, b_u, b_i, P, Q
```

### 4.3 偏置项的初始化策略

除了初始化为 0，还可以使用更合理的初始值：

```python
def smart_bias_initialization(ratings_data, n_users, n_items):
    """
    使用统计量初始化偏置
    """
    # 构建评分矩阵
    from collections import defaultdict
    user_ratings = defaultdict(list)
    item_ratings = defaultdict(list)
    all_ratings = []

    for u, i, r in ratings_data:
        user_ratings[u].append(r)
        item_ratings[i].append(r)
        all_ratings.append(r)

    # 全局均值
    mu = np.mean(all_ratings)

    # 用户偏置 = 用户均值 - 全局均值
    b_u = np.zeros(n_users)
    for u, ratings in user_ratings.items():
        b_u[u] = np.mean(ratings) - mu

    # 物品偏置 = 物品均值 - 全局均值
    b_i = np.zeros(n_items)
    for i, ratings in item_ratings.items():
        b_i[i] = np.mean(ratings) - mu

    return mu, b_u, b_i
```

## 5. 应用场景

### 5.1 适用场景

BiasSVD 特别适合以下场景：

| 场景 | 说明 |
|------|------|
| 评分差异大 | 不同用户评分标准差异大 |
| 物品质量差异大 | 不同物品的受欢迎程度差异大 |
| 稀疏评分 | 每个用户/物品评分较少时，偏置提供先验 |

### 5.2 偏置分析

```python
def analyze_biases(b_u, b_i, user_ids=None, item_ids=None, top_k=10):
    """
    分析用户和物品偏置
    """
    print("=" * 50)
    print("用户偏置分析")
    print("=" * 50)

    # 最宽容的用户
    top_users = np.argsort(b_u)[::-1][:top_k]
    print(f"\n最宽容的 {top_k} 个用户:")
    for idx in top_users:
        print(f"  用户 {idx}: 偏置 {b_u[idx]:.3f}")

    # 最严格的用户
    bottom_users = np.argsort(b_u)[:top_k]
    print(f"\n最严格的 {top_k} 个用户:")
    for idx in bottom_users:
        print(f"  用户 {idx}: 偏置 {b_u[idx]:.3f}")

    print("\n" + "=" * 50)
    print("物品偏置分析")
    print("=" * 50)

    # 最受欢迎的物品
    top_items = np.argsort(b_i)[::-1][:top_k]
    print(f"\n最受欢迎的 {top_k} 个物品:")
    for idx in top_items:
        print(f"  物品 {idx}: 偏置 {b_i[idx]:.3f}")

    # 最不受欢迎的物品
    bottom_items = np.argsort(b_i)[:top_k]
    print(f"\n最不受欢迎的 {top_k} 个物品:")
    for idx in bottom_items:
        print(f"  物品 {idx}: 偏置 {b_i[idx]:.3f}")
```

## 6. 优缺点分析

### 6.1 优点

1. **准确性提升**：相比 FunkSVD，RMSE 通常降低 0.01-0.02
2. **可解释性**：偏置项有明确的含义
3. **简单有效**：只增加了少量参数，但效果显著
4. **处理稀疏性**：偏置提供先验信息
5. **冷启动友好**：新用户/物品可以用偏置均值预测

### 6.2 缺点

1. **参数增加**：多了 |U| + |I| 个偏置参数
2. **可能过拟合**：偏置项也需要正则化
3. **静态偏置**：不考虑时间因素

### 6.3 与其他方法对比

| 方法 | RMSE 提升 | 参数增加 | 复杂度 |
|------|-----------|----------|--------|
| FunkSVD | 基准 | - | 低 |
| BiasSVD | +0.01~0.02 | \|U\|+\|I\| | 低 |
| SVD++ | +0.02~0.03 | \|I\|×K | 中 |
| TimeSVD | +0.01~0.02 | 时间相关参数 | 高 |

## 7. 调库实现

### 7.1 使用 Surprise 库

```python
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate

# 加载数据
data = Dataset.load_builtin('ml-100k')

# BiasSVD（Surprise 的 SVD 默认使用偏置）
model = SVD(
    n_factors=100,      # 隐因子数量
    n_epochs=20,        # 迭代次数
    biased=True,        # 使用偏置（默认为 True）
    lr_all=0.005,       # 学习率
    reg_all=0.02        # 正则化参数
)

# 交叉验证
results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# 训练
trainset, testset = train_test_split(data, test_size=0.25)
model.fit(trainset)

# 获取偏置和因子
print(f"全局均值: {trainset.global_mean:.4f}")
print(f"用户偏置形状: {model.bu.shape}")
print(f"物品偏置形状: {model.bi.shape}")
print(f"用户因子形状: {model.pu.shape}")
print(f"物品因子形状: {model.qi.shape}")
```

### 7.2 预测分解

```python
def explain_prediction(model, trainset, user_id, item_id):
    """
    解释 BiasSVD 的预测结果
    """
    # 获取内部索引
    try:
        u_inner = trainset.to_inner_uid(user_id)
        i_inner = trainset.to_inner_iid(item_id)
    except:
        return None

    # 各项分解
    mu = trainset.global_mean
    b_u = model.bu[u_inner]
    b_i = model.bi[i_inner]
    interaction = np.dot(model.pu[u_inner], model.qi[i_inner])

    total = mu + b_u + b_i + interaction

    print(f"预测用户 {user_id} 对物品 {item_id} 的评分:")
    print(f"  全局均值 μ:    {mu:.4f}")
    print(f"  用户偏置 b_u:  {b_u:.4f}")
    print(f"  物品偏置 b_i:  {b_i:.4f}")
    print(f"  个性化交互:    {interaction:.4f}")
    print(f"  -------------------------")
    print(f"  预测总分:      {total:.4f}")

    return total
```

## 8. 手工代码实现

### 8.1 完整的 BiasSVD 实现

```python
import numpy as np
from collections import defaultdict

class BiasSVD:
    """
    BiasSVD 矩阵分解实现
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
        self.mu = 0           # 全局均值
        self.b_u = None       # 用户偏置
        self.b_i = None       # 物品偏置
        self.P = None         # 用户因子矩阵
        self.Q = None         # 物品因子矩阵

        # ID 映射
        self.user_id_map = {}
        self.item_id_map = {}

        # 训练历史
        self.loss_history = []

    def _build_id_maps(self, user_ids, item_ids):
        """构建 ID 映射"""
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))

        self.user_id_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_id_map = {iid: i for i, iid in enumerate(unique_items)}

        return len(unique_users), len(unique_items)

    def fit(self, user_ids, item_ids, ratings,
            val_user_ids=None, val_item_ids=None, val_ratings=None):
        """
        训练模型
        """
        np.random.seed(self.random_state)

        # 构建 ID 映射
        n_users, n_items = self._build_id_maps(user_ids, item_ids)

        # 初始化
        self.mu = np.mean(ratings)
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # 准备训练数据
        train_data = []
        for u, i, r in zip(user_ids, item_ids, ratings):
            if u in self.user_id_map and i in self.item_id_map:
                train_data.append((
                    self.user_id_map[u],
                    self.item_id_map[i], r
                ))
        train_data = np.array(train_data)

        # 准备验证数据
        val_data = None
        if val_user_ids is not None:
            val_data = []
            for u, i, r in zip(val_user_ids, val_item_ids, val_ratings):
                if u in self.user_id_map and i in self.item_id_map:
                    val_data.append((
                        self.user_id_map[u],
                        self.item_id_map[i], r
                    ))
            val_data = np.array(val_data) if val_data else None

        # 训练循环
        for epoch in range(self.n_epochs):
            np.random.shuffle(train_data)

            total_loss = 0

            for u_idx, i_idx, r in train_data:
                u_idx, i_idx = int(u_idx), int(i_idx)

                # 预测
                pred = self.mu + self.b_u[u_idx] + self.b_i[i_idx] + np.dot(self.P[u_idx], self.Q[i_idx])

                # 误差
                error = r - pred
                total_loss += error ** 2

                # 更新偏置
                self.b_u[u_idx] += self.learning_rate * (error - self.reg_param * self.b_u[u_idx])
                self.b_i[i_idx] += self.learning_rate * (error - self.reg_param * self.b_i[i_idx])

                # 更新因子
                P_old = self.P[u_idx].copy()
                self.P[u_idx] += self.learning_rate * (error * self.Q[i_idx] - self.reg_param * self.P[u_idx])
                self.Q[i_idx] += self.learning_rate * (error * P_old - self.reg_param * self.Q[i_idx])

            # 计算损失
            reg_loss = self.reg_param * (
                np.sum(self.P ** 2) + np.sum(self.Q ** 2) +
                np.sum(self.b_u ** 2) + np.sum(self.b_i ** 2)
            )
            train_rmse = np.sqrt((total_loss + reg_loss) / len(train_data))
            self.loss_history.append(train_rmse)

            if self.verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{self.n_epochs}, Train RMSE: {train_rmse:.4f}"
                if val_data is not None:
                    val_rmse = self._compute_rmse(val_data)
                    msg += f", Val RMSE: {val_rmse:.4f}"
                print(msg)

        return self

    def _compute_rmse(self, data):
        """计算 RMSE"""
        errors = []
        for u_idx, i_idx, r in data:
            u_idx, i_idx = int(u_idx), int(i_idx)
            pred = self.mu + self.b_u[u_idx] + self.b_i[i_idx] + np.dot(self.P[u_idx], self.Q[i_idx])
            errors.append((r - pred) ** 2)
        return np.sqrt(np.mean(errors))

    def predict(self, user_id, item_id):
        """
        预测评分
        """
        # 冷启动处理
        if user_id not in self.user_id_map:
            if item_id not in self.item_id_map:
                return self.mu  # 完全未知
            # 新用户，返回物品均值
            i_idx = self.item_id_map[item_id]
            return self.mu + self.b_i[i_idx]

        if item_id not in self.item_id_map:
            # 新物品，返回用户均值
            u_idx = self.user_id_map[user_id]
            return self.mu + self.b_u[u_idx]

        u_idx = self.user_id_map[user_id]
        i_idx = self.item_id_map[item_id]

        return self.mu + self.b_u[u_idx] + self.b_i[i_idx] + np.dot(self.P[u_idx], self.Q[i_idx])

    def explain_prediction(self, user_id, item_id):
        """
        解释预测结果
        """
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return None

        u_idx = self.user_id_map[user_id]
        i_idx = self.item_id_map[item_id]

        interaction = np.dot(self.P[u_idx], self.Q[i_idx])
        total = self.mu + self.b_u[u_idx] + self.b_i[i_idx] + interaction

        return {
            'global_mean': self.mu,
            'user_bias': self.b_u[u_idx],
            'item_bias': self.b_i[i_idx],
            'interaction': interaction,
            'prediction': total
        }

    def recommend(self, user_id, n_items=10, exclude_items=None):
        """
        为用户推荐
        """
        if user_id not in self.user_id_map:
            # 新用户，推荐热门物品
            return self._recommend_popular(n_items, exclude_items)

        u_idx = self.user_id_map[user_id]

        # 计算所有物品的预测评分
        scores = self.mu + self.b_u[u_idx] + self.b_i + np.dot(self.Q, self.P[u_idx])

        # 排除已知物品
        if exclude_items:
            for item_id in exclude_items:
                if item_id in self.item_id_map:
                    scores[self.item_id_map[item_id]] = -np.inf

        # 获取 top-n
        top_indices = np.argsort(scores)[::-1][:n_items]

        idx_to_item = {v: k for k, v in self.item_id_map.items()}
        recommendations = [(idx_to_item[idx], scores[idx]) for idx in top_indices]

        return recommendations

    def _recommend_popular(self, n_items, exclude_items):
        """推荐热门物品"""
        scores = self.mu + self.b_i

        if exclude_items:
            for item_id in exclude_items:
                if item_id in self.item_id_map:
                    scores[self.item_id_map[item_id]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n_items]

        idx_to_item = {v: k for k, v in self.item_id_map.items()}
        return [(idx_to_item[idx], scores[idx]) for idx in top_indices]

    def get_user_bias(self, user_id):
        """获取用户偏置"""
        if user_id not in self.user_id_map:
            return 0
        return self.b_u[self.user_id_map[user_id]]

    def get_item_bias(self, item_id):
        """获取物品偏置"""
        if item_id not in self.item_id_map:
            return 0
        return self.b_i[self.item_id_map[item_id]]


# ==================== 使用示例 ====================
if __name__ == "__main__":
    np.random.seed(42)

    # 模拟数据
    user_ids = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    item_ids = [0, 1, 2, 3, 0, 2, 3, 1, 2, 3, 0, 1, 3, 1, 2, 3]
    ratings = [5, 3, 4, 1, 4, 5, 2, 3, 4, 5, 2, 4, 3, 1, 5, 4]

    # 划分数据
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
    print("BiasSVD 训练")
    print("=" * 50)

    model = BiasSVD(
        n_factors=10,
        learning_rate=0.01,
        reg_param=0.1,
        n_epochs=100,
        verbose=True
    )

    model.fit(train_users, train_items, train_ratings,
              val_users, val_items, val_ratings)

    # 预测解释
    print("\n" + "=" * 50)
    print("预测解释")
    print("=" * 50)

    explanation = model.explain_prediction(0, 3)
    if explanation:
        print(f"用户 0 对物品 3 的预测分解:")
        print(f"  全局均值: {explanation['global_mean']:.4f}")
        print(f"  用户偏置: {explanation['user_bias']:.4f}")
        print(f"  物品偏置: {explanation['item_bias']:.4f}")
        print(f"  个性化交互: {explanation['interaction']:.4f}")
        print(f"  总预测: {explanation['prediction']:.4f}")

    # 推荐示例
    print("\n" + "=" * 50)
    print("推荐示例")
    print("=" * 50)

    recs = model.recommend(0, n_items=3, exclude_items=[0, 1, 2])
    print("为用户 0 推荐:")
    for item_id, score in recs:
        print(f"  物品 {item_id}: 预测评分 {score:.2f}")
```

## 9. 可视化与结果理解

### 9.1 偏置分布可视化

```python
import matplotlib.pyplot as plt

def visualize_bias_distribution(b_u, b_i):
    """可视化偏置分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 用户偏置分布
    axes[0].hist(b_u, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', label='Zero')
    axes[0].axvline(x=np.mean(b_u), color='g', linestyle='-', label=f'Mean: {np.mean(b_u):.3f}')
    axes[0].set_xlabel('User Bias')
    axes[0].set_ylabel('Count')
    axes[0].set_title('User Bias Distribution')
    axes[0].legend()

    # 物品偏置分布
    axes[1].hist(b_i, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(x=0, color='r', linestyle='--', label='Zero')
    axes[1].axvline(x=np.mean(b_i), color='g', linestyle='-', label=f'Mean: {np.mean(b_i):.3f}')
    axes[1].set_xlabel('Item Bias')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Item Bias Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
```

### 9.2 预测分解可视化

```python
def visualize_prediction_decomposition(model, user_id, item_ids):
    """可视化多个物品的预测分解"""
    explanations = []
    for item_id in item_ids:
        exp = model.explain_prediction(user_id, item_id)
        if exp:
            explanations.append((item_id, exp))

    if not explanations:
        return

    # 堆叠柱状图
    fig, ax = plt.subplots(figsize=(14, 6))

    items = [str(e[0]) for e in explanations]
    global_means = [model.mu] * len(explanations)
    user_biases = [e[1]['user_bias'] for e in explanations]
    item_biases = [e[1]['item_bias'] for e in explanations]
    interactions = [e[1]['interaction'] for e in explanations]

    x = np.arange(len(items))
    width = 0.5

    # 堆叠
    ax.bar(x, global_means, width, label='Global Mean', color='gray')
    ax.bar(x, user_biases, width, bottom=global_means, label='User Bias', color='blue')
    ax.bar(x, item_biases, width, bottom=np.array(global_means)+np.array(user_biases),
           label='Item Bias', color='green')
    ax.bar(x, interactions, width,
           bottom=np.array(global_means)+np.array(user_biases)+np.array(item_biases),
           label='Interaction', color='red')

    ax.set_xlabel('Item ID')
    ax.set_ylabel('Rating')
    ax.set_title(f'Prediction Decomposition for User {user_id}')
    ax.set_xticks(x)
    ax.set_xticklabels(items)
    ax.legend()
    ax.axhline(y=model.mu, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
```

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_bias_svd(model, test_data):
    """评估 BiasSVD 模型"""
    actual = []
    predicted = []

    for user_id, item_id, rating in test_data:
        pred = model.predict(user_id, item_id)
        actual.append(rating)
        predicted.append(pred)

    actual = np.array(actual)
    predicted = np.array(predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    # 分解误差
    explained_by_bias = 0
    explained_by_interaction = 0

    return {
        'RMSE': rmse,
        'MAE': mae
    }
```

## 11. 常见问题与易错点

### 11.1 常见问题

**Q1：偏置项需要正则化吗？**

A：需要。偏置项过大会导致过拟合。通常偏置的正则化系数可以和隐因子相同或略小。

**Q2：偏置项初始化为 0 还是使用统计量？**

A：都可以。使用统计量初始化可以加速收敛，但最终效果差别不大。

**Q3：如何处理新用户/新物品？**

A：BiasSVD 有更好的冷启动处理：
- 新用户：使用 μ + b_i（物品偏置）
- 新物品：使用 μ + b_u（用户偏置）
- 都新：使用 μ（全局均值）

### 11.2 易错点

1. **忘记更新偏置**：只更新了 P 和 Q
2. **偏置正则化太强**：导致偏置趋近于 0
3. **冷启动处理不当**：对新用户/物品返回 NaN

## 12. 学习总结

### 12.1 核心要点

1. **偏置项作用**：建模用户和物品的整体评分倾向
2. **预测公式**：$\hat{r}_{ui} = \mu + b_u + b_i + p_u \cdot q_i^T$
3. **优势**：准确性提升、可解释性增强、冷启动友好
4. **成本**：增加 |U| + |I| 个参数

### 12.2 演进路径

```
FunkSVD (p·q)
    ↓ 添加偏置
BiasSVD (μ + b_u + b_i + p·q)
    ↓ 添加隐式反馈
SVD++ (μ + b_u + b_i + p·(q + y))
    ↓ 添加时间因素
TimeSVD (时变参数)
```

## 13. 练习题与思考题

### 13.1 基础题

1. **（填空）** BiasSVD 的预测公式为：$\hat{r}_{ui} = $ ______。

2. **（简答）** 解释用户偏置 $b_u$ 和物品偏置 $b_i$ 的含义。

### 13.2 进阶题

3. **（编程）** 实现一个带有单独偏置正则化参数的 BiasSVD。

4. **（分析）** 比较 BiasSVD 和 FunkSVD 在相同数据上的表现差异。

### 13.3 思考题

5. 偏置项在什么情况下最有用？什么时候效果不明显？

6. 如何将 BiasSVD 扩展到隐式反馈场景？

### 参考答案

1. $\mu + b_u + b_i + p_u \cdot q_i^T$

2. 用户偏置 $b_u$ 表示用户相对于全局平均水平的评分倾向（正=宽容，负=严格）；物品偏置 $b_i$ 表示物品相对于全局平均水平的受欢迎程度（正=受欢迎，负=不受欢迎）。

## 14. 学习路径建议

### 14.1 前置知识

- [ ] FunkSVD 基础
- [ ] 梯度下降优化
- [ ] 正则化概念

### 14.2 学习顺序

1. 理解偏置项的作用 → 阅读本文档
2. 动手实现 → 手写 BiasSVD
3. 实验对比 → 比较 FunkSVD 和 BiasSVD
4. 分析偏置 → 理解用户/物品的评分特征

### 14.3 下一步学习

- **SVD++**：考虑隐式反馈
- **TimeSVD**：添加时间因素
- **深度学习模型**：Neural CF、DeepFM
