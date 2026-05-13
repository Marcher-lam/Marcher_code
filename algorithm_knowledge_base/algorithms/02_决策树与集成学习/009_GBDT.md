# GBDT 算法学习文档

## 1. 算法基础认知

GBDT（Gradient Boosting Decision Tree，梯度提升决策树）是最成功的机器学习算法之一，由加州大学伯克利分校的Jerome Friedman于1999年提出。GBDT是Boosting思想与决策树算法的深度结合，通过迭代训练决策树来逐步降低损失函数，在分类和回归任务上都表现优异。

GBDT的核心思想可以类比为"持续改进"：每一轮训练一棵决策树，让这棵树去学习前面所有树的预测与真实标签之间的"残差"（或者说负梯度）。通过不断添加新的决策树来逐步拟合残差，GBDT能够达到很高的预测准确性。这个过程就像一个团队不断完善解决方案，每个成员都专注于解决前面成员未能解决的问题。

GBDT在机器学习领域具有里程碑式的地位。它是许多著名算法的基础，包括XGBoost、LightGBM、CatBoost等。GBDT及其变体在Kaggle等数据科学竞赛中表现出色，是处理表格数据的最强算法之一。与深度学习相比，GBDT在结构化数据上通常表现更好，且具有更好的可解释性。

GBDT的训练过程可以理解为梯度下降在函数空间中的推广。在传统的梯度下降中，我们通过更新参数来降低损失；在GBDT中，我们通过添加新的决策树来降低损失。每棵新树都向着损失函数梯度的方向添加，这使得GBDT具有理论上收敛的保证。

## 2. 核心原理

GBDT的核心原理建立在梯度提升（Gradient Boosting）和决策树的基础上。

### 2.1 梯度提升框架

GBDT是一个加法模型，最终的预测是所有决策树的累加：

$$F(x) = F_0(x) + \sum_{m=1}^{M} \gamma_m h_m(x)$$

其中 $F_0(x)$ 是初始预测（通常是均值或先验概率），$h_m(x)$ 是第m棵决策树，$\gamma_m$ 是对应的学习率。

GBDT通过梯度下降来优化这个加法模型。设损失函数为 $L(y, F(x))$，在第m步，我们需要找到一棵决策树 $h_m$ 来拟合损失函数的负梯度：

$$r_{im} = - \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$$

$r_{im}$ 称为响应（response）或伪残差（pseudo-residual），它表示当前预测值需要改进的方向和程度。

### 2.2 损失函数

GBDT支持多种损失函数，根据任务类型选择：

对于回归任务，常用的损失函数是MSE（均方误差）：

$$L(y, F) = \frac{1}{2} (y - F)^2$$

负梯度为：

$$r = y - F$$

对于二分类任务，常用的损失函数是对数损失：

$$L(y, F) = -[y \log \sigma(F) + (1-y) \log(1-\sigma(F))]$$

其中 $\sigma(F) = 1/(1+exp(-F))$ 是sigmoid函数。负梯度为：

$$r = y - \sigma(F)$$

### 2.3 决策树拟合

在GBDT中，每棵新决策树都拟合当前预测的负梯度。设当前模型为 $F_{m-1}$，计算负梯度：

$$r_i = - \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$$

然后用决策树来拟合 $\{(x_i, r_i)\}$。

决策树的每个叶子节点 $j$ 对应一个叶节点值 $\gamma_{mj}$。对于回归树，叶节点值是使损失函数最小的常数；对于分类树，需要转换为叶节点权重。

### 2.4 叶节点值计算

决策树拟合后，需要计算每个叶子节点的值。

对于回归任务，使用使MSE最小的值：

$$\gamma_{mj} = \arg\min_\gamma \sum_{x_i \in R_{mj}} L(y_i, F_{m-1}(x_i) + \gamma)$$

对于MSE，这简化为：

$$\gamma_{mj} = \text{mean}(y_i - F_{m-1}(x_i)) \quad \text{对于} x_i \in R_{mj}$$

为了防止过拟合，可以在叶节点值上应用学习率 $\nu$ 进行收缩：

$$F_m(x) = F_{m-1}(x) + \nu \gamma_{mj}$$

其中 $\nu$ 是学习率（通常0.01-0.1），也称为收缩因子（shrinkage）。

## 3. 数学公式与推导

### 3.1 梯度提升算法

设训练数据为 $\{(x_i, y_i)\}_{i=1}^{N}$，损失函数为 $L(y, F(x))$，GBDT的算法流程如下：

初始化：

$$F_0(x) = \arg\min_\gamma \sum_{i=1}^{N} L(y_i, \gamma)$$

对于 $m = 1$ 到 $M$：

1. 计算伪残差：

$$r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}}$$

2. 用决策树拟合 $\{(x_i, r_{im})\}$，得到 $h_m(x)$

3. 对每个叶子节点 $j$，计算叶节点值：

$$\gamma_{mj} = \arg\min_\gamma \sum_{x_i \in R_{mj}} L(y_i, F_{m-1}(x_i) + \gamma)$$

4. 更新模型：

$$F_m(x) = F_{m-1}(x) + \nu \sum_{j} \gamma_{mj} I(x \in R_{mj})$$

最终模型为：

$$F_M(x) = F_0(x) + \nu \sum_{m=1}^{M} \sum_{j} \gamma_{mj} I(x \in R_{mj})$$

### 3.2 MSE损失函数的推导

设损失函数为 $L(y, F) = \frac{1}{2}(y - F)^2$。

负梯度为：

$$r = - \frac{\partial L}{\partial F} = -(-(y - F)) = y - F$$

这正好是真实的残差！所以GBDT在MSE损失下是在拟合真实残差。

叶节点值计算简化：

$$\gamma_{mj} = \text{mean}(y_i - F_{m-1}(x_i))$$

因为 $\frac{\partial}{\partial \gamma} \sum (y - F_{m-1} - \gamma)^2 = 0$ 给出 $\gamma = \text{mean}(y - F_{m-1})$。

### 3.3 对数损失函数的推导

设损失函数为 $L(y, F) = -[y \log \sigma(F) + (1-y) \log(1-\sigma(F))]$，其中 $\sigma(F) = 1/(1+e^{-F})$。

负梯度为：

$$r = y - \sigma(F)$$

这也是真实标签与预测概率的残差。

对于二分类，最终预测为：

$$P(y=1|x) = \sigma(F_M(x))$$

### 3.4 Huber损失函数的推导

Huber损失结合了MSE和MAE的优点，对异常值更鲁棒：

$$L(y, F) = \begin{cases} \frac{1}{2}(y-F)^2 & \text{if } |y-F| \leq \delta \\ \delta(|y-F| - \frac{\delta}{2}) & \text{if } |y-F| > \delta \end{cases}$$

负梯度和叶节点值的计算需要分情况讨论。

## 4. 训练过程讲解

### 4.1 GBDT的训练

GBDT的训练过程可以分为以下步骤：

第一步，初始化模型。通常选择使损失函数最小的常数作为初始预测。

对于回归，使用均值：$F_0(x) = \bar{y}$

对于二分类，使用对数几率：$F_0(x) = \log(\frac{p}{1-p})$，其中 $p$ 是正类比例

第二步，迭代训练决策树：

对于每轮迭代 m = 1, 2, ..., M：

1. 计算伪残差 $r_i$（负梯度）

2. 用决策树拟合伪残差，得到叶子节点

3. 计算每个叶子节点的最优值 $\gamma_j$

4. 更新模型：$F_m = F_{m-1} + \nu \gamma_j$

第三步，得到最终模型。

### 4.2 决策树的构建

在GBDT中，决策树通常使用CART算法构建。决策树的构建包括：

分裂特征选择：使用MSE减少量或基尼指数选择最优特征

分裂点选择：对每个特征，选择使目标值方差最小的分裂点

树的生长：递归构建子树，通常限制最大深度或最小样本数

叶节点值计算：使叶节点内损失函数最小

### 4.3 正则化

GBDT通过以下方式防止过拟合：

学习率收缩（Shrinkage）：每棵树的贡献乘以学习率 $\nu$（通常0.01-0.1）

限制树的复杂度：限制最大深度、最小样本数等

限制树的数量：使用验证集选择最优树数量

### 4.4 预测过程

对于新样本，预测过程如下：

将样本输入每棵决策树，获取叶节点值

将所有叶节点值乘以学习率并累加

对于分类问题，通过sigmoid函数转换为概率

## 5. 应用场景

GBDT在实际应用中广泛的场景，主要用于以下领域：

在金融风控领域，GBDT可以用于信用评分和风险评估。GBDT能够自动处理非线性关系和高维特征，是信用评分卡模型的主流算法。

在搜索排序领域，GBDT是学习排序（Learning to Rank）的主流算法。著名的LambdaMART就是基于GBDT的排序算法。

在点击率预估领域，GBDT可以用于在线广告的点击率预估。GBDT加上特征工程可以达到很好的效果。

在异常检测领域，GBDT可以用于识别异常交易、欺诈检测等。GBDT的残差分析方法可以识别异常模式。

在推荐系统领域，GBDT可以用于排序推荐。通过学习用户对物品的偏好，GBDT可以生成个性化的推荐列表。

在医疗健康领域，GBDT可以用于疾病预测、医疗费用预测等。其可解释性有助于医生理解决策依据。

## 6. 优缺点分析

### 6.1 优点

GBDT算法具有以下显著优点：

首先，GBDT具有很高的准确性。GBDT通过迭代拟合残差，能够捕捉复杂的非线性关系，在很多数据集上表现优异。

其次，GBDT可以处理各种类型的数据。包括连续特征、离散特征、缺失值等。

第三，GBDT具有很好的可解释性。可以查看特征重要性，理解模型的决策逻辑。

第四，GBDT可以处理高维稀疏数据。与正则化结合，GBDT可以进行特征选择。

第五，GBDT可以自然处理缺失值。不需要额外的数据填充。

第六，GBDT的泛化能力很强。通过学习率收缩和树复杂度控制，GBDT通常不容易过拟合。

### 6.2 缺点

GBDT算法也存在一些缺点：

首先，GBDT的训练时间较长。由于需要序列化训练多棵决策树，训练时间比随机森林长。

其次，GBDT的预测时间较长。需要遍历所有决策树来获取预测。

第三，GBDT对超参数敏感。包括树的数量、学习率、树的深度等，需要仔细调优。

第四，GBDT不适合高维稀疏数据。在文本分类等高维稀疏数据上可能不如线性模型。

第五，GBDT不能并行训练。虽然单棵树内部可以并行，但迭代过程不能并行。

## 7. 调库实现

### 7.1 分类问题实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.datasets import load_iris, make_classification


def gbdt_classification():
    """GBDT分类器 - sklearn实现"""
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    gb_clf = GradientBoostingClassifier(
        n_estimators=100,          # 树的数量
        learning_rate=0.1,         # 学习率
        max_depth=3,              # 最大深度
        min_samples_split=2,      # 最小分裂样本数
        min_samples_leaf=1,        # 叶节点最小样本数
        subsample=1.0,            # 子采样比例
        random_state=42
    )
    
    gb_clf.fit(X_train, y_train)
    y_pred = gb_clf.predict(X_test)
    
    print("=" * 60)
    print("GBDT分类器 - sklearn实现")
    print("=" * 60)
    print(f"\n准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred)}")
    
    cv_scores = cross_val_score(gb_clf, X, y, cv=5)
    print(f"\n5折交叉验证: {cv_scores}")
    print(f"平均准确率: {cv_scores.mean():.4f}")
    
    print(f"\n特征重要性:")
    for name, imp in zip(iris.feature_names, gb_clf.feature_importances_):
        print(f"  {name}: {imp:.4f}")
    
    return gb_clf


def gbdt_binary_classification():
    """GBDT二分类示例"""
    
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    gb_clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    gb_clf.fit(X_train, y_train)
    y_pred = gb_clf.predict(X_test)
    y_prob = gb_clf.predict_proba(X_test)[:, 1]
    
    print(f"\n二分类准确率: {accuracy_score(y_test, y_pred):.4f}")
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    
    cv_scores = cross_val_score(gb_clf, X, y, cv=5)
    print(f"5折交叉验证: {cv_scores.mean():.4f}")
    
    return gb_clf


def visualize_learning_curve():
    """可视化学习曲线"""
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    n_estimators = [10, 25, 50, 100, 200]
    train_scores = []
    test_scores = []
    
    for n in n_estimators:
        gb = GradientBoostingClassifier(n_estimators=n, random_state=42)
        gb.fit(X_train, y_train)
        train_scores.append(gb.score(X_train, y_train))
        test_scores.append(gb.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators, train_scores, 'b-o', label='训练准确率')
    plt.plot(n_estimators, test_scores, 'r-o', label='测试准确率')
    plt.xlabel('树的数量')
    plt.ylabel('准确率')
    plt.title('GBDT学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gbdt_learning_curve.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    gb_clf = gbdt_classification()
    gb_clf = gbdt_binary_classification()
    visualize_learning_curve()
```

### 7.2 回归问题实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def gbdt_regression():
    """GBDT回归器 - sklearn实现"""
    
    np.random.seed(42)
    
    n_samples = 200
    X = np.sort(np.random.rand(n_samples) * 10, axis=0).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(n_samples) * 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    gb_reg = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42
    )
    
    gb_reg.fit(X_train, y_train)
    y_pred = gb_reg.predict(X_test)
    
    print("=" * 60)
    print("GBDT回归器 - sklearn实现")
    print("=" * 60)
    print(f"\nMSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    return gb_reg


def visualize_regression():
    """可视化GBDT回归结果"""
    
    np.random.seed(42)
    
    X = np.sort(np.random.rand(100) * 10).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1
    
    gb_reg = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, 
        max_depth=3, random_state=42
    )
    gb_reg.fit(X, y)
    
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    y_pred = gb_reg.predict(X_plot)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='真实值')
    plt.plot(X_plot, y_pred, 'r-', linewidth=2, label='预测值')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('GBDT回归')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gbdt_regression.png', dpi=150)
    plt.show()


def visualize_feature_importance():
    """可视化特征重要性"""
    
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_reg.fit(X, y)
    
    importance = gb_reg.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance[indices], color='steelblue')
    plt.yticks(range(len(importance)), np.array(housing.feature_names)[indices])
    plt.xlabel('重要性')
    plt.title('GBDT - 特征重要性')
    plt.tight_layout()
    plt.savefig('gbdt_feature_importance.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    gbdt_regression()
    visualize_regression()
    visualize_feature_importance()
```

### 7.3 代码解释

上述代码展示了sklearn中GBDT的实现方式。关键参数说明：

- `n_estimators`：树的数量，通常100-200
- `learning_rate`：学习率（收缩因子），0.01-0.1
- `max_depth`：树的最大深度，通常3-5
- `min_samples_split`：分裂所需的最小样本数
- `min_samples_leaf`：叶节点最小样本数
- `subsample`：子采样比例，用于随机梯度提升
- `random_state`：随机种子

## 8. 手工代码实现

### 8.1 完整NumPy实现

```python
import numpy as np
from collections import Counter
import math


class GBDTClassifier:
    """
    GBDT分类器的纯NumPy实现
    
    使用对数损失函数，即二分类的Logistic回归
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.F = None
    
    def _sigmoid(self, x):
        """Sigmoid函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _gini(self, y):
        """计算基尼指数"""
        if len(y) == 0:
            return 0.0
        
        counter = Counter(y)
        n = len(y)
        gini = 1.0
        
        for count in counter.values():
            p = count / n
            gini -= p * p
        
        return gini
    
    def _find_best_split(self, X, y, feature_indices):
        """找到最佳分裂"""
        n_samples = len(y)
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for idx in feature_indices:
            unique_values = np.unique(X[:, idx])
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_mask = X[:, idx] <= threshold
                right_mask = X[:, idx] > threshold
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                y_left, y_right = y[left_mask], y[right_mask]
                
                n_left = len(y_left)
                n_right = len(y_right)
                
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                
                gain = (self._gini(y) - 
                       (n_left / n_samples * gini_left + 
                        n_right / n_samples * gini_right))
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, feature_indices, depth=0):
        """构建决策树"""
        n_samples = len(y)
        
        if len(set(y)) == 1:
            return {'value': y[0], 'children': {}}
        
        if len(feature_indices) == 0:
            return {'value': Counter(y).most_common(1)[0][0], 'children': {}}
        
        if n_samples < self.min_samples_split:
            return {'value': Counter(y).most_common(1)[0][0], 'children': {}}
        
        if self.max_depth is not None and depth >= self.max_depth:
            return {'value': Counter(y).most_common(1)[0][0], 'children': {}}
        
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, y, feature_indices
        )
        
        if best_feature is None or best_gain <= 0:
            return {'value': Counter(y).most_common(1)[0][0], 'children': {}}
        
        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'children': {}
        }
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        remaining_features = [f for f in feature_indices if f != best_feature]
        
        if np.sum(left_mask) >= self.min_samples_leaf:
            tree['children']['<='] = self._build_tree(
                X[left_mask], y[left_mask], remaining_features, depth + 1
            )
        else:
            tree['children']['<='] = {
                'value': Counter(y).most_common(1)[0][0], 
                'children': {}
            }
        
        if np.sum(right_mask) >= self.min_samples_leaf:
            tree['children']['>'] = self._build_tree(
                X[right_mask], y[right_mask], remaining_features, depth + 1
            )
        else:
            tree['children']['>'] = {
                'value': Counter(y).most_common(1)[0][0],
                'children': {}
            }
        
        return tree
    
    def _get_leaf_value(self, tree, x):
        """获取叶节点值"""
        if 'value' in tree:
            return tree['value']
        
        feature_idx = tree['feature']
        threshold = tree['threshold']
        
        if x[feature_idx] <= threshold:
            return self._get_leaf_value(tree['children']['<='], x)
        else:
            return self._get_leaf_value(tree['children']['>'], x)
    
    def _compute_leaf_value(self, tree, X, F_prev):
        """计算叶节点的最优值"""
        if 'value' in tree:
            return tree['value']
        
        for direction, child in tree['children'].items():
            if direction == '<=':
                mask = X[:, tree['feature']] <= tree['threshold']
            else:
                mask = X[:, tree['feature']] > tree['threshold']
            
            if np.sum(mask) > 0:
                leaf_value = self._compute_leaf_value(child, X[mask], F_prev[mask])
                tree['children'][direction]['value'] = leaf_value
        
        return None
    
    def fit(self, X, y):
        """训练GBDT分类器"""
        X = np.array(X)
        y = np.array(y)
        
        n_samples = len(y)
        n_features = X.shape[1]
        
        pos_rate = np.mean(y)
        self.F = np.log(pos_rate / (1 - pos_rate)) * np.ones(n_samples)
        
        for m in range(self.n_estimators):
            r = y - self._sigmoid(self.F)
            
            tree = self._build_tree(X, r, list(range(n_features)))
            
            feature_indices = []
            def collect_indices(t, indices):
                if 'feature' in t:
                    indices.append(t['feature'])
                for child in t.get('children', {}).values():
                    if isinstance(child, dict) and 'feature' in child:
                        collect_indices(child, indices)
            collect_indices(tree, feature_indices)
            
            if feature_indices:
                unique_features = sorted(set(feature_indices))
                for i in range(n_features):
                    if i not in unique_features:
                        tree_i = self._build_tree(X, r, [i])
                        tree = tree_i
            
            self.trees.append(tree)
            
            for tree in [tree]:
                leaf_predictions = {}
                for i in range(n_samples):
                    leaf_val = self._get_leaf_value(tree, X[i])
                    if leaf_val is not None:
                        leaf_predictions[leaf_val] = r[i]
                
                for direction, child in tree.get('children', {}).items():
                    if 'value' in child:
                        continue
                    for dir2, leaf in child.get('children', {}).items():
                        if 'value' in leaf:
                            continue
            
            leaf_corrections = []
            for i in range(n_samples):
                leaf_corrections.append(self._get_leaf_value(tree, X[i]))
            
            predictions = []
            for leaf_val in leaf_corrections:
                if leaf_val is not None:
                    predictions.append(leaf_val)
                else:
                    predictions.append(0)
            
            predictions = np.array(predictions)
            mask = ~np.isnan(predictions)
            
            if np.sum(mask) > 0:
                gradient = r[mask]
                correction = np.mean(gradient[mask])
                
                self.F[mask] += self.learning_rate * gradient[mask]
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        X = np.array(X)
        n_samples = len(X)
        
        F = np.zeros(n_samples)
        
        pos_rate = 0.5
        F += np.log(pos_rate / (1 - pos_rate))
        
        for tree in self.trees:
            for i in range(n_samples):
                leaf_val = self._get_leaf_value(tree, X[i])
                if leaf_val is not None:
                    F[i] += self.learning_rate * leaf_val
        
        proba = self._sigmoid(F)
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class GBDTRegressor:
    """
    GBDT回归器的纯NumPy实现
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = None
    
    def _mse(self, y):
        """计算MSE"""
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _find_best_split(self, X, y, feature_indices):
        """找到最佳分裂"""
        n_samples = len(y)
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for idx in feature_indices:
            unique_values = np.unique(X[:, idx])
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_mask = X[:, idx] <= threshold
                right_mask = X[:, idx] > threshold
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                y_left, y_right = y[left_mask], y[right_mask]
                
                n_left = len(y_left)
                n_right = len(y_right)
                
                mse_left = self._mse(y_left)
                mse_right = self._mse(y_right)
                
                gain = (self._mse(y) - 
                       (n_left / n_samples * mse_left + 
                        n_right / n_samples * mse_right))
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, residual, feature_indices, depth=0):
        """构建决策树"""
        n_samples = len(y)
        
        if n_samples < self.min_samples_split:
            return {'value': np.mean(residual), 'children': {}}
        
        if self.max_depth is not None and depth >= self.max_depth:
            return {'value': np.mean(residual), 'children': {}}
        
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, y, feature_indices
        )
        
        if best_feature is None or best_gain <= 0:
            return {'value': np.mean(residual), 'children': {}}
        
        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'children': {}
        }
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        remaining_features = [f for f in feature_indices if f != best_feature]
        
        if np.sum(left_mask) >= self.min_samples_leaf:
            tree['children']['<='] = self._build_tree(
                X[left_mask], y[left_mask], 
                residual[left_mask], remaining_features, depth + 1
            )
        else:
            tree['children']['<='] = {'value': np.mean(residual), 'children': {}}
        
        if np.sum(right_mask) >= self.min_samples_leaf:
            tree['children']['>'] = self._build_tree(
                X[right_mask], y[right_mask],
                residual[right_mask], remaining_features, depth + 1
            )
        else:
            tree['children']['>'] = {'value': np.mean(residual), 'children': {}}
        
        return tree
    
    def _get_leaf_value(self, tree, x):
        """获取叶节点值"""
        if 'value' in tree:
            return tree['value']
        
        feature_idx = tree['feature']
        threshold = tree['threshold']
        
        if x[feature_idx] <= threshold:
            return self._get_leaf_value(tree['children']['<='], x)
        else:
            return self._get_leaf_value(tree['children']['>'], x)
    
    def fit(self, X, y):
        """训练GBDT回归器"""
        X = np.array(X)
        y = np.array(y)
        
        n_samples = len(y)
        n_features = X.shape[1]
        
        self.initial_prediction = np.mean(y)
        F = np.full(n_samples, self.initial_prediction)
        
        for m in range(self.n_estimators):
            residual = y - F
            
            tree = self._build_tree(
                X, y, residual, list(range(n_features))
            )
            
            leaf_values = {}
            for i in range(n_samples):
                leaf_val = self._get_leaf_value(tree, X[i])
                leaf_values[leaf_val] = residual[i]
            
            for direction, child in tree.get('children', {}).items():
                values = []
                for leaf in child.get('children', {}).values():
                    if 'value' in leaf:
                        values.append(leaf['value'])
                
                if values:
                    mean_val = np.mean(values)
                    if 'value' in child:
                        child['value'] = mean_val
                    else:
                        for leaf in child.get('children', {}).values():
                            if 'value' not in leaf:
                                leaf['value'] = mean_val
            
            new_predictions = np.zeros(n_samples)
            for i in range(n_samples):
                leaf_val = self._get_leaf_value(tree, X[i])
                if leaf_val is not None:
                    new_predictions[i] = self.learning_rate * leaf_val
            
            F = F + new_predictions
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """预测新样本"""
        X = np.array(X)
        n_samples = len(X)
        
        predictions = np.full(n_samples, self.initial_prediction)
        
        for tree in self.trees:
            for i in range(n_samples):
                leaf_val = self._get_leaf_value(tree, X[i])
                if leaf_val is not None:
                    predictions[i] += self.learning_rate * leaf_val
        
        return predictions


def demo():
    """GBDT演示"""
    
    print("=" * 60)
    print("GBDT分类器 - 手工实现")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    
    gb_clf = GBDTClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb_clf.fit(X, y)
    
    predictions = gb_clf.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\n训练准确率: {accuracy:.4f}")
    
    print(f"\n树的数量: {len(gb_clf.trees)}")


if __name__ == '__main__':
    demo()
```

### 8.2 代码关键点解释

上述代码是GBDT的完整NumPy实现，主要包含两个类：

GBDTClassifier：使用对数损失函数的分类器，使用sigmoid函数转换预测值为概率。在每轮迭代中：计算负梯度（伪残差）；用决策树拟合伪残差；更新累计预测。

GBDTRegressor：使用MSE损失函数的回归器。在每轮迭代中：计算真实残差；用决策树拟合残差；更新累计预测。

两个实现的核心都是迭代添加决策树，每棵树都试图拟合前面所有树的预测与真实值之间的残差。

## 9. 可视化与结果理解

### 9.1 决策边界可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier


def visualize_decision_boundary():
    """可视化GBDT决策边界"""
    
    np.random.seed(42)
    
    X1 = np.random.normal(-2, 1, 100)
    X2 = np.random.normal(2, 1, 100)
    X = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
    y = np.array([0] * 100 + [1] * 100)
    
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    gb.fit(X, y)
    
    xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    Z = gb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', alpha=0.5)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('GBDT决策边界')
    plt.tight_layout()
    plt.savefig('gbdt_boundary.png', dpi=150)
    plt.show()


def compare_estimators():
    """比较不同树数量的效果"""
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    n_estimators = [10, 25, 50, 100, 200]
    scores = []
    
    for n in n_estimators:
        gb = GradientBoostingClassifier(n_estimators=n, max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        scores.append(gb.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators, scores, 'bo-')
    plt.xlabel('树的数量')
    plt.ylabel('准确率')
    plt.title('GBDT: 树数量与准确率的关系')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gbdt_estimators.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_decision_boundary()
    compare_estimators()
```

### 9.2 结果理解

通过可视化可以看到，GBDT的决策边界随着迭代逐步优化。更多的树通常会提高准确性，但过多的树可能导致过拟合。

## 10. 模型评估

### 10.1 评估指标

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)


def evaluate_gbdt():
    """评估GBDT分类器"""
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    gb.fit(X, y)
    y_pred = gb.predict(X)
    
    print("=" * 60)
    print("GBDT分类器评估")
    print("=" * 60)
    
    print(f"\n准确率: {accuracy_score(y, y_pred):.4f}")
    
    cv_scores = cross_val_score(gb, X, y, cv=5)
    print(f"\n5折交叉验证: {cv_scores}")
    print(f"平均准确率: {cv_scores.mean():.4f}")
    
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y, y_pred))
    
    print(f"\n分类报告:")
    print(classification_report(y, y_pred))
    
    print(f"\n特征重要性:")
    for name, imp in zip(iris.feature_names, gb.feature_importances_):
        print(f"  {name}: {imp:.4f}")


def evaluate_gbdt_regressor():
    """评估GBDT回归器"""
    
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    np.random.seed(42)
    X = np.sort(np.random.rand(200) * 10).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(200) * 0.1
    
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb.fit(X, y)
    y_pred = gb.predict(X)
    
    print("=" * 60)
    print("GBDT回归器评估")
    print("=" * 60)
    
    print(f"\nMSE: {mean_squared_error(y, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    print(f"R²: {r2_score(y, y_pred):.4f}")


if __name__ == '__main__':
    evaluate_gbdt()
    evaluate_gbdt_regressor()
```

### 10.2 评估指标解释

GBDT的评估指标与普通分类器和回归器相同。对于分类问题，关键是准确率、交叉验证分数、混淆矩阵等。对于回归问题，关键是MSE、RMSE、R²。

## 11. 常见问题与易错点

### 11.1 树数量选择

树数量是GBDT最重要的超参数之一。过多的树可能导致过拟合，过少的树可能导致欠拟合。

通常通过交叉验证选择最优数量，或使用早停策略。

### 11.2 学习率选择

学习率与树数量通常需要权衡：较小的学习率需要更多的树，但泛化能力更好。

常见做法：从较小的学习率（如0.1）开始，使用验证集选择最优树数量。

### 11.3 树深度选择

树深度影响模型的复杂度。较浅的树方差小但可能欠拟合，较深的树复杂但可能过拟合。

常见深度为3-5。

### 11.4 子采样

子采样（Subsample）是在每轮迭代中随机选择部分样本来训练决策树。这可以增加模型的多样性，提高泛化能力。

常用的子采样比例为0.8。

### 11.5 特征重要性

GBDT的特征重要性基于各特征对损失的贡献。但在存在高度相关特征时，重要性可能被分散。

## 12. 学习总结

GBDT是梯度提升框架下的决策树算法，通过迭代拟合残差来逐步提升模型性能。GBDT的核心贡献包括：

引入梯度提升框架。GBDT将梯度下降推广到函数空间，通过添加新的决策树来降低损失。

支持多种损失函数。GBDT可以用于回归、二分类、多分类等不同任务。

使用学习率收缩。通过收缩因子来防止过拟合，提高泛化能力。

支持特征重要性。GBDT可以识别重要的特征，这在特征选择中很有价值。

GBDT是许多高级算法的基础，包括XGBoost、LightGBM等。这些算法在GBDT的基础上进行了工程优化，提高了训练效率。

## 13. 练习题与思考题与思考题

### 13.1 选择题

1. GBDT使用什么方法来逐步提升模型性能？
   A. Bootstrap采样
   B. 梯度提升
   C. 随机特征选择
   D. 样本权重调整
   答案：B

2. GBDT的每棵新树拟合的是什么？
   A. 真实标签
   B. 伪残差（负梯度）
   C. 上一棵树的预测
   D. 随机噪声
   答案：B

3. GBDT的学习率（收缩因子）的作用是什么？
   A. 加快训练
   B. 防止过拟合
   C. 增加特征重要性
   D. 减少树的数量
   答案：B

### 13.2 计算题

假设初始预测为 $F_0 = 0.5$，第一轮迭代计算得到伪残差为 $r = [0.3, 0.2, -0.1, 0.1]$，决策树叶节点值为 $gamma = [0.15, -0.05]$，学习率为 $nu = 0.1$，计算更新后的预测。

解：$F_1 = F_0 + nu * gamma = 0.5 + 0.1 * [0.15, -0.05, 0.15, -0.05] = [0.515, 0.495, 0.485, 0.495]$

### 13.3 思考题

1. GBDT与随机森林有什么区别和联系？
   
   答案：区别：训练方式，GBDT是序列化（每棵树依赖前面的树），随机森林是并行（各树独立）；集成方式，GBDT是加法模型，随机森林是投票/平均；过拟合控制，GBDT使用学习率收缩，随机森林使用特征随机选择。联系：都使用决策树作为基学习器；都可以用于分类和回归。

2. 为什么GBDT不容易过拟合？
   
   答案：GBDT使用学习率收缩，每棵树的贡献被缩小；GBDT使用树的深度限制，控制单棵树的复杂度；GBDT的树数量虽然多，但由于每棵树只拟合残差，整体容量有限。

3. GBDT如何处理高维稀疏数据？
   
   答案：GBDT不太适合高维稀疏数据（如文本的one-hot编码）。原因：树模型对稀疏特征的分裂效率低；容易过拟合。解决方法：使用特征哈希、使用线性模型作为基学习器（如GBLinear）。

### 13.4 编程题

使用sklearn实现GBDT，调整不同参数（树数量、学习率、最大深度），观察对模型性能的影响。

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

params = [
    {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8},
]

for p in params:
    gb = GradientBoostingClassifier(**p, random_state=42)
    scores = cross_val_score(gb, X, y, cv=5)
    print(f"{p}: {scores.mean():.4f}")
```

## 14. 学习路径建议建议

学习GBDT算法应该按照以下路径进行：

首先，理解决策树的基础知识。如果不熟悉决策树，需要先学习CART算法。

然后，理解梯度下降的原理。梯度下降是优化的基础方法，需要深入理解。

第三，理解梯度提升的框架。梯度提升将梯度下降推广到函数空间，是GBDT的理论基础。

第四，理解GBDT的训练过程。伪残差计算、决策树拟合、模型更新是核心步骤。

第五，学习如何使用sklearn实现GBDT。sklearn的GradientBoostingClassifier和GradientBoostingRegressor是标准实现。

第六，理解GBDT的正则化方法。学习率收缩、树深度限制、子采样等。

最后，可以进一步学习XGBoost、LightGBM等算法。这些算法在GBDT的基础上进行了优化，是竞赛和工业的主流算法。

## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述GBDT的核心思想及适用场景。
<details><summary>参考答案</summary>
GBDT通过数据驱动学习输入到输出的映射，适用于传统机器学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出GBDT的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现GBDT核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. GBDT在什么情况下会失效？
2. 训练数据很少时，GBDT还能有效工作吗？
3. 如何将GBDT与其他方法结合？

