# XGBoost 学习文档

## 1. 算法基础认知

### 1.1 什么是 XGBoost？

XGBoost（eXtreme Gradient Boosting）是 GBDT 的优化版本，由陈天奇在 2014 年提出。它在 Kaggle 竞赛中取得了巨大成功，成为数据科学竞赛的"神器"。

### 1.2 XGBoost vs GBDT

| 特性 | GBDT | XGBoost |
|------|------|---------|
| 目标函数 | 损失函数 | 损失函数 + 正则化项 |
| 优化方法 | 一阶梯度 | 一阶 + 二阶梯度 |
| 正则化 | 较弱 | L1 + L2 正则化 |
| 缺失值处理 | 需预处理 | 自动处理 |
| 并行化 | 无 | 特征粒度并行 |
| 计算速度 | 较慢 | 快 |

### 1.3 XGBoost 的核心创新

1. **正则化目标函数**：添加 L1/L2 正则化
2. **二阶泰勒展开**：使用一阶和二阶梯度
3. **特征预排序**：加速分裂点搜索
4. **稀疏感知**：自动处理缺失值
5. **并行计算**：特征级别的并行

## 2. 数学原理

### 2.1 正则化目标函数

XGBoost 的目标函数：

$$\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

其中正则化项：

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $T$：叶节点数量
- $w_j$：叶节点权重
- $\gamma$：叶节点数惩罚
- $\lambda$：L2 正则化系数

### 2.2 二阶泰勒展开

使用泰勒展开近似损失函数：

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} [l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$

其中：
- $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$（一阶导数）
- $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$（二阶导数）

### 2.3 最优叶节点权重

对于叶节点 $j$，最优权重：

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

其中：
- $G_j = \sum_{i \in I_j} g_i$（一阶导数和）
- $H_j = \sum_{i \in I_j} h_i$（二阶导数和）

### 2.4 最优目标函数值

$$\mathcal{L}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

### 2.5 分裂准则

寻找最优分裂点，最大化增益：

$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

## 3. 代码实现

### 3.1 简化版 XGBoost

```python
import numpy as np
from collections import defaultdict

class SimpleXGBoostTreeNode:
    """XGBoost 树节点"""
    def __init__(self):
        self.is_leaf = False
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.weight = None  # 叶节点权重


class SimpleXGBoostTree:
    """
    简化版 XGBoost 回归树
    """

    def __init__(self, max_depth=3, min_child_weight=1, reg_lambda=1.0, gamma=0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None

    def fit(self, X, g, h):
        """
        训练一棵树

        参数:
            X: 特征矩阵
            g: 一阶梯度
            h: 二阶梯度
        """
        self.root = self._build_tree(X, g, h, depth=0)
        return self

    def _build_tree(self, X, g, h, depth):
        """递归构建树"""
        node = SimpleXGBoostTreeNode()

        # 计算当前节点的权重
        G = np.sum(g)
        H = np.sum(h)
        node.weight = -G / (H + self.reg_lambda)

        # 终止条件
        if depth >= self.max_depth or len(X) < 2:
            node.is_leaf = True
            return node

        # 寻找最优分裂
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # 按特征值排序
            sorted_idx = np.argsort(X[:, feature_idx])
            X_sorted = X[sorted_idx]
            g_sorted = g[sorted_idx]
            h_sorted = h[sorted_idx]

            # 枚举分裂点
            G_L, H_L = 0, 0
            G_R, H_R = G, H

            for i in range(len(X) - 1):
                G_L += g_sorted[i]
                H_L += h_sorted[i]
                G_R -= g_sorted[i]
                H_R -= h_sorted[i]

                # 检查 min_child_weight
                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue

                # 检查阈值是否相同
                if X_sorted[i, feature_idx] == X_sorted[i+1, feature_idx]:
                    continue

                # 计算增益
                gain = (G_L**2 / (H_L + self.reg_lambda) +
                       G_R**2 / (H_R + self.reg_lambda) -
                       G**2 / (H + self.reg_lambda)) / 2 - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = (X_sorted[i, feature_idx] + X_sorted[i+1, feature_idx]) / 2
                    best_left_idx = sorted_idx[:i+1]
                    best_right_idx = sorted_idx[i+1:]

        # 如果没有找到有效分裂
        if best_gain <= 0:
            node.is_leaf = True
            return node

        # 分裂
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(X[best_left_idx], g[best_left_idx], h[best_left_idx], depth + 1)
        node.right = self._build_tree(X[best_right_idx], g[best_right_idx], h[best_right_idx], depth + 1)

        return node

    def predict(self, X):
        """预测"""
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        """预测单个样本"""
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.weight


class SimpleXGBoost:
    """
    简化版 XGBoost 实现
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_child_weight=1, reg_lambda=1.0, gamma=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma

        self.trees = []
        self.base_score = None

    def fit(self, X, y):
        """
        训练 XGBoost

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值 (n_samples,)
        """
        # 初始化预测值
        self.base_score = np.mean(y)
        y_pred = np.full(len(y), self.base_score)

        for m in range(self.n_estimators):
            # 计算梯度和 hessian（MSE 损失）
            # L = 0.5 * (y - y_pred)^2
            # g = dL/dy_pred = y_pred - y
            # h = d^2L/dy_pred^2 = 1
            g = y_pred - y
            h = np.ones_like(y)

            # 训练树
            tree = SimpleXGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma
            )
            tree.fit(X, g, h)

            # 更新预测
            y_pred += self.learning_rate * tree.predict(X)

            self.trees.append(tree)

            if (m + 1) % 20 == 0:
                mse = np.mean((y - y_pred) ** 2)
                print(f"迭代 {m+1}/{self.n_estimators}, MSE: {mse:.4f}")

        return self

    def predict(self, X):
        """预测"""
        y_pred = np.full(X.shape[0], self.base_score)

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred


# 使用示例
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # 生成数据
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练
    xgb = SimpleXGBoost(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=1.0
    )
    xgb.fit(X_train, y_train)

    # 预测
    y_pred = xgb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n测试 MSE: {mse:.4f}")
```

## 4. 使用官方 XGBoost 库

### 4.1 基本使用

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# 准备数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 DMatrix（XGBoost 专用数据结构）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 参数
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'eval_metric': 'rmse',
    'max_depth': 6,
    'eta': 0.1,  # 学习率
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,  # L2 正则化
    'alpha': 0.0,    # L1 正则化
    'seed': 42
}

# 训练
num_round = 100
watchlist = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=10)

# 预测
y_pred = model.predict(dtest)
mse = mean_squared_error(y_test, y_pred)
print(f"\n测试 MSE: {mse:.4f}")
```

### 4.2 分类任务

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

# 生成分类数据
X, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 XGBClassifier（sklearn API）
clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

# 训练（带早停）
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=10
)

# 预测
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n测试准确率: {accuracy:.4f}")
print(f"测试 AUC: {auc:.4f}")

# 特征重要性
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
xgb.plot_importance(clf, max_num_features=20)
plt.show()
```

### 4.3 超参数调优

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [0.1, 1.0, 10.0]
}

# 随机搜索
random_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, n_jobs=-1),
    param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"最佳参数: {random_search.best_params_}")
print(f"最佳 AUC: {random_search.best_score_:.4f}")
```

## 5. XGBoost 在推荐系统中的应用

### 5.1 CTR 预估

```python
class XGBoostCTRModel:
    """XGBoost CTR 预估模型"""

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            min_child_weight=50,  # 防止过拟合
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X, y, X_val=None, y_val=None):
        """训练"""
        eval_set = [(X, y)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=30,
            verbose=10
        )
        return self

    def predict_proba(self, X):
        """预测点击概率"""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """预测点击"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names, importance_type='gain'):
        """获取特征重要性"""
        importance = self.model.get_booster().get_score(
            importance_type=importance_type
        )

        # 构建完整的重要性字典
        full_importance = {}
        for i, name in enumerate(feature_names):
            key = f'f{i}'
            full_importance[name] = importance.get(key, 0)

        # 排序
        sorted_importance = sorted(
            full_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_importance


# 使用示例
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # 模拟 CTR 数据
    X, y = make_classification(
        n_samples=100000,
        n_features=50,
        n_informative=20,
        weights=[0.9, 0.1],  # 模拟 CTR 数据不平衡
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )

    # 训练
    model = XGBoostCTRModel()
    model.fit(X_train, y_train, X_val, y_val)

    # 评估
    from sklearn.metrics import roc_auc_score, log_loss

    y_pred_proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"\n测试 AUC: {auc:.4f}")
    print(f"测试 LogLoss: {logloss:.4f}")

    # 特征重要性
    importance = model.get_feature_importance(feature_names)
    print("\nTop 10 重要特征:")
    for name, score in importance[:10]:
        print(f"  {name}: {score:.2f}")
```

## 6. 重要参数说明

### 6.1 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| booster | gbtree | 基学习器类型 |
| n_jobs | 1 | 并行线程数 |
| random_state | 0 | 随机种子 |

### 6.2 Tree Booster 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| eta (learning_rate) | 0.3 | 学习率 |
| max_depth | 6 | 树的最大深度 |
| min_child_weight | 1 | 最小叶子权重和 |
| subsample | 1 | 样本采样比例 |
| colsample_bytree | 1 | 特征采样比例 |
| lambda (reg_lambda) | 1 | L2 正则化 |
| alpha (reg_alpha) | 0 | L1 正则化 |
| gamma (min_split_loss) | 0 | 最小分裂增益 |

### 6.3 学习任务参数

| 参数 | 说明 |
|------|------|
| objective | 学习目标 |
| eval_metric | 评估指标 |
| seed | 随机种子 |

## 7. 调参技巧

### 7.1 调参顺序

1. **固定 learning_rate=0.1，调 n_estimators**
2. **调 max_depth 和 min_child_weight**
3. **调 gamma、subsample、colsample_bytree**
4. **调正则化参数 reg_lambda、reg_alpha**
5. **减小 learning_rate，增加 n_estimators**

### 7.2 防止过拟合

- 减小 max_depth
- 增大 min_child_weight
- 增大 gamma
- 减小 subsample 和 colsample_bytree
- 增大正则化参数
- 使用早停

## 8. 学习总结

### 8.1 XGBoost 优势

1. **正则化**：内置 L1/L2 正则化
2. **二阶优化**：收敛更快
3. **并行化**：训练速度快
4. **缺失值处理**：自动学习
5. **灵活性**：支持自定义目标函数

### 8.2 与其他模型对比

| 模型 | 速度 | 精度 | 可扩展性 |
|------|------|------|----------|
| GBDT | 慢 | 高 | 中 |
| XGBoost | 快 | 高 | 高 |
| LightGBM | 更快 | 高 | 高 |
| CatBoost | 中 | 高 | 高 |

## 9. 练习题

1. 比较 XGBoost 不同 max_depth 对模型效果的影响。

2. 实现 XGBoost 的交叉验证和早停。

3. 使用 XGBoost 进行特征选择。
