# GBDT（梯度提升决策树） 学习文档

## 1. 算法基础认知

### 1.1 什么是 GBDT？

GBDT（Gradient Boosting Decision Tree，梯度提升决策树）是一种集成学习算法，通过迭代地训练决策树来纠正前一轮的残差。

### 1.2 提升法（Boosting）思想

```
初始预测 F₀(x)
    ↓
训练第1棵树 h₁(x) 预测残差
    ↓
F₁(x) = F₀(x) + h₁(x)
    ↓
训练第2棵树 h₂(x) 预测新残差
    ↓
F₂(x) = F₁(x) + h₂(x)
    ↓
...重复 M 轮...
    ↓
最终预测 F_M(x) = F₀(x) + Σ hₘ(x)
```

### 1.3 GBDT vs Random Forest

| 特性 | GBDT | Random Forest |
|------|------|---------------|
| 组合方式 | 串行（依赖前一轮） | 并行（独立） |
| 基学习器 | 通常 CART 回归树 | CART 决策树 |
| 拟合目标 | 残差（负梯度） | 减少方差 |
| 过拟合 | 容易，需要正则化 | 不容易 |
| 计算速度 | 较慢 | 较快 |

## 2. 数学原理

### 2.1 加法模型

GBDT 的最终模型是加法模型：

$$F_M(x) = \sum_{m=1}^{M} \gamma_m h_m(x)$$

其中：
- $h_m(x)$：第 m 棵决策树
- $\gamma_m$：第 m 棵树的学习率/权重

### 2.2 梯度提升

对于损失函数 $L(y, F(x))$，每一轮拟合的是损失函数的**负梯度**：

$$r_{mi} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}}$$

### 2.3 常见损失函数

**回归：MSE**
$$L(y, F) = \frac{1}{2}(y - F)^2$$

负梯度 = 残差：
$$r = y - F$$

**分类：Log Loss**
$$L(y, F) = -y \log(\sigma(F)) - (1-y)\log(1-\sigma(F))$$

负梯度：
$$r = y - \sigma(F)$$

## 3. GBDT 算法流程

```
输入：训练数据 {(xᵢ, yᵢ)}，迭代次数 M，学习率 η

1. 初始化 F₀(x) = argmin_c Σ L(yᵢ, c)

2. for m = 1 to M:
   a) 计算负梯度（伪残差）：
      rᵢ = -∂L(yᵢ, F)/∂F |_{F=F_{m-1}(xᵢ)}

   b) 用 {rᵢ} 训练一棵回归树 hₘ(x)

   c) 线性搜索确定最佳权重：
      γₘ = argmin_γ Σ L(yᵢ, F_{m-1}(xᵢ) + γ·hₘ(xᵢ))

   d) 更新模型：
      Fₘ(x) = F_{m-1}(x) + η·γₘ·hₘ(x)

输出：F_M(x)
```

## 4. 代码实现

### 4.1 简单 GBDT 回归实现

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SimpleGBDTRegressor:
    """
    简单的 GBDT 回归实现
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        参数:
            n_estimators: 树的数量
            learning_rate: 学习率
            max_depth: 每棵树的最大深度
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        """
        训练 GBDT

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值 (n_samples,)
        """
        # 初始化：使用均值作为初始预测
        self.initial_prediction = np.mean(y)
        F = np.full(len(y), self.initial_prediction)

        for m in range(self.n_estimators):
            # 计算残差（MSE 损失的负梯度）
            residuals = y - F

            # 训练回归树拟合残差
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=5
            )
            tree.fit(X, residuals)

            # 预测残差
            pred_residuals = tree.predict(X)

            # 更新预测
            F += self.learning_rate * pred_residuals

            # 保存树
            self.trees.append(tree)

            # 打印进度
            if (m + 1) % 20 == 0:
                mse = np.mean((y - F) ** 2)
                print(f"树 {m+1}/{self.n_estimators}, MSE: {mse:.4f}")

        return self

    def predict(self, X):
        """
        预测

        参数:
            X: 特征矩阵 (n_samples, n_features)

        返回:
            预测值 (n_samples,)
        """
        F = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        return F


# 使用示例
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # 生成数据
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练 GBDT
    gbdt = SimpleGBDTRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4
    )
    gbdt.fit(X_train, y_train)

    # 预测
    y_pred = gbdt.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n测试集 MSE: {mse:.4f}")
    print(f"测试集 R²: {r2:.4f}")
```

### 4.2 GBDT 分类实现

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SimpleGBDTClassifier:
    """
    GBDT 二分类实现
    使用 Log Loss
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_log_odds = None

    def _sigmoid(self, x):
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        """训练"""
        # 初始化：使用对数几率
        p = np.mean(y)
        self.initial_log_odds = np.log(p / (1 - p))
        F = np.full(len(y), self.initial_log_odds)

        for m in range(self.n_estimators):
            # 计算概率
            prob = self._sigmoid(F)

            # 计算负梯度（y - p）
            residuals = y - prob

            # 训练树
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=5
            )
            tree.fit(X, residuals)

            # 预测
            pred = tree.predict(X)

            # 更新
            F += self.learning_rate * pred

            self.trees.append(tree)

            if (m + 1) % 20 == 0:
                log_loss = -np.mean(
                    y * np.log(prob + 1e-10) + (1 - y) * np.log(1 - prob + 1e-10)
                )
                print(f"树 {m+1}/{self.n_estimators}, Log Loss: {log_loss:.4f}")

        return self

    def predict_proba(self, X):
        """预测概率"""
        F = np.full(X.shape[0], self.initial_log_odds)

        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        prob = self._sigmoid(F)
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        """预测类别"""
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)
```

## 5. 正则化技术

### 5.1 学习率（Shrinkage）

```python
# 学习率越小，需要更多的树
learning_rate = 0.01  # 小学习率
n_estimators = 1000   # 需要更多树
```

### 5.2 子采样（Subsampling）

每棵树使用部分样本：

```python
class GBRTWithSubsampling:
    """带子采样的 GBDT"""

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, subsample=0.8):
        self.subsample = subsample
        # ... 其他参数

    def fit(self, X, y):
        n_samples = len(y)

        for m in range(self.n_estimators):
            # 子采样
            sample_idx = np.random.choice(
                n_samples,
                int(n_samples * self.subsample),
                replace=False
            )
            X_sample, residuals_sample = X[sample_idx], residuals[sample_idx]

            # 训练树
            tree.fit(X_sample, residuals_sample)
            # ...
```

### 5.3 特征采样

每棵树使用部分特征：

```python
max_features = 'sqrt'  # 使用 sqrt(n_features) 个特征
```

### 5.4 早停（Early Stopping）

```python
class GBDTWithEarlyStopping:
    """带早停的 GBDT"""

    def fit(self, X, y, X_val=None, y_val=None, patience=10):
        best_score = float('inf')
        no_improve = 0

        for m in range(self.n_estimators):
            # 训练一棵树
            # ...

            # 验证
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_score = mean_squared_error(y_val, val_pred)

                if val_score < best_score:
                    best_score = val_score
                    no_improve = 0
                    self.best_trees = self.trees.copy()
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"早停于第 {m+1} 棵树")
                    self.trees = self.best_trees
                    break
```

## 6. 使用 scikit-learn

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# 回归
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None,
    random_state=42
)
gbr.fit(X_train, y_train)

print(f"训练 R²: {gbr.score(X_train, y_train):.4f}")
print(f"测试 R²: {gbr.score(X_test, y_test):.4f}")

# 特征重要性
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(range(len(gbr.feature_importances_)), gbr.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

# 训练过程
plt.figure(figsize=(10, 6))
plt.plot(gbr.train_score_)
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.title('Training Process')
plt.show()

# 超参数调优
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
```

## 7. GBDT 在推荐系统中的应用

### 7.1 CTR 预估

```python
class GBDTCTRModel:
    """
    使用 GBDT 进行 CTR 预估
    """

    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=5):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            min_samples_leaf=50,  # 防止过拟合
            random_state=42
        )

    def fit(self, X, y):
        """训练"""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """预测点击概率"""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """预测点击"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names):
        """获取特征重要性"""
        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        print("特征重要性排名:")
        for i in sorted_idx[:20]:
            print(f"  {feature_names[i]}: {importance[i]:.4f}")

        return importance
```

### 7.2 GBDT + LR（特征转换）

Facebook 提出的 GBDT+LR 方法：

```python
from sklearn.preprocessing import OneHotEncoder

class GBDTLRModel:
    """
    GBDT + LR 模型
    GBDT 用于特征转换，LR 用于最终预测
    """

    def __init__(self, n_estimators=50, max_depth=4):
        self.gbdt = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.encoder = OneHotEncoder()
        self.lr = None

    def fit(self, X, y):
        """训练"""
        # 1. 训练 GBDT
        self.gbdt.fit(X, y)

        # 2. 获取叶节点索引作为新特征
        leaf_ids = self.gbdt.apply(X)[:, :, 0]  # (n_samples, n_estimators)

        # 3. One-Hot 编码
        self.encoder.fit(leaf_ids)
        X_transformed = self.encoder.transform(leaf_ids)

        # 4. 训练 LR
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(X_transformed, y)

        return self

    def predict_proba(self, X):
        """预测"""
        leaf_ids = self.gbdt.apply(X)[:, :, 0]
        X_transformed = self.encoder.transform(leaf_ids)
        return self.lr.predict_proba(X_transformed)[:, 1]
```

## 8. 模型对比

| 模型 | 优点 | 缺点 |
|------|------|------|
| GBDT | 精度高、处理各种数据 | 容易过拟合、串行训练 |
| XGBoost | 正则化、并行化 | 参数多 |
| LightGBM | 速度快、内存少 | 可能欠拟合 |
| CatBoost | 处理类别特征 | 训练慢 |

## 9. 学习总结

### 9.1 核心要点

1. **串行提升**：每棵树拟合残差
2. **梯度优化**：拟合损失函数的负梯度
3. **正则化**：学习率、子采样、早停
4. **应用广泛**：排序、分类、回归

### 9.2 调参经验

1. **先定学习率**：0.1 较好
2. **调 n_estimators**：配合早停
3. **调 max_depth**：5-8 较好
4. **加子采样**：0.8-0.9

## 10. 练习题

1. 从零实现一个带早停的 GBDT 回归器。

2. 比较不同学习率和树数量的组合效果。

3. 实现 GBDT+LR 模型并测试效果。
