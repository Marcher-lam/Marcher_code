# LightGBM 学习文档

> 微软开发的高效梯度提升框架，使用基于直方图的决策树算法实现超高速训练

---

## 1. 算法基础认知

**一句话定义**：LightGBM是一种基于梯度提升的机器学习框架，通过创新的直方图算法和叶-wise生长策略实现比传统GBDT快10倍以上的训练速度，同时保持几乎相同的预测精度。

**直觉类比**：想象你在整理书架——传统GBDT是一本一本按顺序找位置放，而LightGBM先把书按大小分成几堆（直方图），再在同一堆里找具体位置，这样就快很多！

**历史背景**：2017年，微软亚洲研究院的Ke等人在论文"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"中提出。LightGBM在Kaggle等竞赛中表现优异，已成为最流行的GBDT实现之一。

**算法定位**：
- 类型：监督学习 → 梯度提升树
- 输出：回归/分类预测
- 模型类型：决策树集成

**前置知识**：
- [必备]：决策树基础、GBDT概念
- [必备]：梯度下降
- [扩展]：XGBoost

---

## 2. 核心原理

### 2.1 核心思想

LightGBM的核心创新是**直方图算法 + 叶-wise生长**：
1. **直方图算法**：将连续特征离散化为k个箱子，减少分裂候选
2. **叶-wise生长**：按增益最大叶子分裂，而非深度优先

核心思想可以概括为：**用空间换时间，通过离散化和最优叶子分裂实现高效训练**。

### 2.2 工作流程

1. **特征离散化**：将连续值划分为直方图bin
2. **寻找最优分裂**：在直方图上找最佳分裂点
3. **叶子生长**：按叶而非按层分裂
4. **梯度计算**：使用直方图差分加速

### 2.3 关键概念解释

- **GOSS (Gradient-based One-Side Sampling)**：保留大梯度样本，随机采样小梯度样本
- **EFB (Exclusive Feature Bundling)**：将互斥特征打包减少特征数量
- **直方图**：连续的离散化bins
- **Leaf-wise**：每次分裂收益最大的叶子

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $L$ | 损失函数 |
| $g_i$ | 一阶梯度 $∂L/∂\hat{y}_i$ |
| $h_i$ | 二阶梯度 $∂²L/∂\hat{y}_i²$ |
| $\lambda$ | 正则化参数 |
| $Gain$ | 分裂增益 |

### 3.2 Gradient统计

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} - \frac{\sum_{i \in I_g} g_i}{\sum_{i \in I_g} h_i + \lambda}$$

增益计算：
$$Gain = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}$$

### 3.3 直方图分裂

将特征值映射到bin：
$$bin_j = \lfloor \frac{x - x_{min}}{x_{max} - x_{min}} \rfloor$$

### 3.4 GOSS采样

保留阈值$\alpha$的大梯度样本，随机采样$\beta$比例的小梯度样本。

---

## 4. 训练过程讲解

### 4.1 数据预处理

```python
import lightgbm as lgb

# 创建数据集
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

# 参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
}
```

### 4.2 迭代过程

```python
# 训练
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

### 4.3 超参数

| 参数 | 作用 | 推荐值 |
|------|------|----------|
| num_leaves | 叶子数 | 31-127 |
| learning_rate | 学习率 | 0.05-0.1 |
| max_depth | 最大深度 | -1(无限制) |
| min_data_in_leaf | 最小叶子数据 | 20-100 |
| feature_fraction | 特征采样 | 0.6-0.9 |
| bagging_fraction | 数据采样 | 0.6-0.9 |

---

## 5. 应用场景

### 5.1 典型应用

- **表格数据分析**：结构化数据分类/回归
- **Kaggle竞赛**：结构化数据比赛常胜
- **推荐系统**：特征工程后的排序
- **时序预测**：时间序列特征

### 5.2 适用数据特征

- 大规模数据（100万+样本）
- 高维特征（100+特征）
- 需要快速训练

---

## 6. 优缺点分析

### 6.1 优点

1. **速度快**：比XGBoost快10倍
2. **内存省**：直方图压缩
3. **准确率**：叶子生长更优
4. **易用**：API简洁

### 6.2 缺点

1. **对小数据**：可能过拟合
2. **调参**：叶子数敏感
3. **类别特征**：需转换

### 6.3 对比

| 维度 | LightGBM | XGBoost |
|------|---------|----------|
| 速度 | 快 | 中 |
| 内存 | 省 | 中 |
| 准确率 | 相近 | 相近 |
| API | 简洁 | 丰富 |

---

## 7. 调库实现

### 7.1 环境准备

```bash
pip install lightgbm scikit-learn
```

### 7.2 完整代码示例

```python
"""
LightGBM 完整实现 - 分类/回归
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error


# ===============================
# 1. 数据准备
# ===============================
def load_data(task='classification'):
    """加载示例数据"""
    if task == 'classification':
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=10000, n_features=20,
            n_informative=15, n_redundant=5,
            random_state=42
        )
    else:
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=10000, n_features=20,
            noise=0.1, random_state=42
        )
    
    return X, y


# ===============================
# 2. 分类任务
# ===============================
def lightgbm_classification():
    """LightGBM分类"""
    
    # 加载数据
    X, y = load_data('classification')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # 参数
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }
    
    # 训练
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
    )
    
    # 预测
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # 评估
    acc = accuracy_score(y_test, y_pred)
    print(f"准确率: {acc:.4f}")
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': [f'f{i}' for i in range(20)],
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    print(f"\n特征重要性:\n{importance.head(10)}")
    
    return model


# ===============================
# 3. 回归任务
# ===============================
def lightgbm_regression():
    """LightGBM回归"""
    
    X, y = load_data('regression')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
    )
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}")
    
    return model


# ===============================
# 4. 调参优化
# ===============================
def tune_lightgbm():
    """交叉验证调参"""
    from sklearn.model_selection import GridSearchCV
    
    X, y = load_data('classification')
    
    param_grid = {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
    }
    
    model = lgb.LGBMClassifier()
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid.fit(X, y)
    
    print(f"最佳参数: {grid.best_params_}")
    print(f"最佳分数: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


# ===============================
# 5. 主程序
# ===============================
if __name__ == "__main__":
    print("=" * 50)
    print("LightGBM 示例")
    print("=" * 50)
    
    print("\n[1/3] 分类任务...")
    model_clf = lightgbm_classification()
    
    print("\n[2/3] 回归任务...")
    model_reg = lightgbm_regression()
    
    print("\n[3/3] 调参优化...")
    best_model = tune_lightgbm()
    
    print("\n✓ 程序执行完毕")
```

---

## 8. 手工代码实现

### 8.1 核心模块

```python
"""
LightGBM 手工实现（简化版）
"""

import numpy as np


class HistGradientBoosting:
    """基于直方图的梯度提升"""
    
    def __init__(self, n_estimators=100, num_leaves=31, lr=0.1):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.lr = lr
        self.trees = []
        
    def _compute_hist(self, X, y, bin_edges):
        """计算直方图"""
        n_bins = len(bin_edges) - 1
        hist = np.zeros((n_bins, 2))
        
        for i in range(len(X)):
            idx = np.searchsorted(bin_edges, X[i]) - 1
            idx = max(0, min(idx, n_bins - 1))
            hist[idx, 0] += y[i]  # sum of gradients
            hist[idx, 1] += 1   # count
        
        return hist
    
    def fit(self, X, y):
        n = len(X)
        
        # 初始化
        self.base_score = np.mean(y)
        y_pred = np.full(n, self.base_score)
        
        for t in range(self.n_estimators):
            # 计算梯度
            residuals = y - y_pred
            
            # 简单决策树（简化版）
            tree = self._build_tree(X, residuals)
            self.trees.append(tree)
            
            # 更新预测
            y_pred += self.lr * self._predict_tree(X, tree)
        
        return self
    
    def _build_tree(self, X, y):
        """构建简单决策树"""
        # 选择最佳分裂点（简化）
        best_gain = 0
        best_split = 0
        
        for j in range(X.shape[1]):
            for split in np.percentile(X[:, j], [25, 50, 75]):
                left = X[:, j] <= split
                right = ~left
                
                if left.sum() == 0 or right.sum() == 0:
                    continue
                
                gain = self._compute_gain(y, y, left, right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (j, split)
        
        return best_split
    
    def _compute_gain(self, y, y_pred, left_mask, right_mask):
        """计算分裂增益"""
        def variance(y):
            return np.var(y) if len(y) > 1 else 0
        
        y_left, y_right = y[left_mask], y[right_mask]
        
        parent = variance(y)
        child = (len(y_left) * variance(y_left) + len(y_right) * variance(y_right)) / len(y)
        
        return parent - child
    
    def _predict_tree(self, X, tree):
        """决策树预测"""
        if tree is None:
            return 0
        
        feature, split = tree
        return np.where(X[:, feature] <= split, 1, -1) * self.lr
    
    def predict(self, X):
        pred = np.full(len(X), self.base_score)
        for tree in self.trees:
            pred += self.lr * self._predict_tree(X, tree)
        return pred


# 测试
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    
    model = HistGradientBoosting(n_estimators=10)
    model.fit(X, y)
    
    print(f"训练完成, 树数量: {len(model.trees)}")
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import lightgbm as lgb


def plot_importance(model, feature_names=None):
    """特征重要性可视化"""
    importance = model.feature_importance()
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance)
    plt.yticks(range(len(importance)), feature_names)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('importance.png')
    plt.show()


def plot_learning_curve(train_results):
    """学习曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_results['valid_logloss'])
    plt.plot(train_results['training_logloss'])
    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.title('Learning Curve')
    plt.legend(['Valid', 'Train'])
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 分类 | 回归 |
|------|------|------|
| Accuracy | ✓ | |
| AUC | ✓ | |
| F1 | ✓ | |
| RMSE | | ✓ |
| MAE | | ✓ |
| R² | | ✓ |

### 10.2 代码

```python
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

def evaluate(model, X_test, y_test, task='classification'):
    if task == 'classification':
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
    else:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE: {rmse:.4f}")
```

---

## 11. 常见问题

### 11.1 数据问题

- **过拟合**：调小num_leaves，增加min_data
- **欠拟合**：增加n_estimators，调大学习率

### 11.2 类别特征

```python
# LightGBM直接处理类别
categorical_features = ['city', 'category']
train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_features)
```

---

## 12. 学习总结

### 12.1 核心

✓ 直方图加速 + 叶-wise分裂 + GOSS采样

### 12.2 算法联系

- 前置：GBDT、XGBoost
- 同类：CatBoost
- 进阶：GBDT+Transformer

---

## 13. 练习题

**问题**：LightGBM为什么比XGBoost快？

答案：直方图离散化减少候选分裂点，叶-wise减少分裂次数。

---

## 14. 学习路径

### 14.1 前置

- [ ] GBDT基础
- [ ] 决策树

### 14.2 进阶

- [ ] XGBoost对比
- [ ] CatBoost

### 14.3 资���

1. 论文：LightGBM原始论文
2. 文档：LightGBM官方文档

---

## 附录

### A. 代码

见第7节。

### B. 参考文献

1. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", 2017

---

**文档结束**