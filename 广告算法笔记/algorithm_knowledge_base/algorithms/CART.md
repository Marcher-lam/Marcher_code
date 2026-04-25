# CART 学习文档

## 1. 算法基础认知

CART（Classification and Regression Tree）由 Breiman 等人于 1984 年提出，是当前最广泛使用的决策树算法。与 ID3 和 C4.5 不同，CART 构建的是**二叉树**——每个非叶节点恰好有两个子节点。CART 既可用于分类（使用基尼指数），也可用于回归（使用平方误差），是 sklearn 中 `DecisionTreeClassifier` 和 `DecisionTreeRegressor` 的底层算法。

## 2. 核心原理

CART 的核心思想：选择一个特征和一个切分点，将当前数据集分为两个子集，使得划分后的不纯度最小化。

**分类树**：使用基尼指数衡量不纯度，选择使加权基尼指数最小的特征-切分点组合。

**回归树**：使用平方误差衡量不纯度，选择使加权平方误差最小的特征-切分点组合。叶节点输出为子集样本的均值。

CART 是一棵严格的二叉树，无论特征是离散还是连续，每次只做二分划分。

## 3. 数学公式与推导

**基尼指数**（分类树）：

$$\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2 = \sum_{k \neq l} p_k p_l$$

基尼指数表示从数据集中随机抽取两个样本，其类别不一致的概率。值越小，纯度越高。

**特征 $A$ 的切分点 $t$ 的基尼指数**：

$$\text{Gini}(D, A, t) = \frac{|D_1|}{|D|} \text{Gini}(D_1) + \frac{|D_2|}{|D|} \text{Gini}(D_2)$$

其中 $D_1 = \{(x,y) \in D \mid A(x) \leq t\}$，$D_2 = D \setminus D_1$。

**平方误差**（回归树）：

$$\text{MSE}(D) = \frac{1}{|D|} \sum_{(x_i, y_i) \in D} (y_i - \bar{y}_D)^2$$

$$\bar{y}_D = \frac{1}{|D|} \sum_{(x_i, y_i) \in D} y_i$$

**代价复杂度剪枝**（Cost-Complexity Pruning）：

定义子树 $T$ 的代价复杂度函数：

$$R_\alpha(T) = R(T) + \alpha |T|$$

其中 $R(T)$ 是训练误差，$|T|$ 是叶节点数，$\alpha \geq 0$ 是复杂度参数。$\alpha$ 越大，树越简单。

对每个内部节点 $t$，计算剪枝阈值：

$$\alpha_t = \frac{R(t) - R(T_t)}{|T_t| - 1}$$

其中 $T_t$ 是以 $t$ 为根的子树。选择最小的 $\alpha_t$ 进行剪枝，逐步生成一系列子树 $T_0 \supset T_1 \supset \dots \supset T_k$，最后通过交叉验证选择最优 $\alpha$。

## 4. 训练过程讲解

**CART 分类树构建**：

1. 对每个特征的每个候选切分点，计算划分后的基尼指数
2. 选择基尼指数最小的（特征，切分点）组合
3. 按此切分点将数据分为左右两个子集
4. 对子集递归执行1-3
5. 停止条件：节点样本数小于阈值、基尼指数为0、达到最大深度

**CART 回归树构建**：

与分类树类似，只是将基尼指数替换为平方误差，叶节点输出为样本均值。

**剪枝流程**：

1. 生成最大树 $T_0$
2. 从 $T_0$ 开始，反复选择使 $R_\alpha$ 最小的节点进行剪枝
3. 得到一系列候选子树
4. 用交叉验证选择最优 $\alpha$ 和对应子树

## 5. 应用场景

- **分类任务**：客户分类、疾病诊断
- **回归任务**：房价预测、销量预测
- **特征工程**：通过树的结构理解特征重要性
- **广告系统**：用户点击率预估（作为基线模型）
- **集成学习基础**：随机森林、GBDT 的基学习器

## 6. 优缺点分析

**优点**：
- 统一的分类和回归框架
- 二叉树结构，计算高效
- 能同时处理连续和离散特征
- 对异常值较鲁棒（回归树）
- sklearn 默认实现，生态完善

**缺点**：
- 贪心算法，局部最优
- 单棵树容易过拟合
- 对数据扰动敏感（不稳定）
- 只能做轴平行划分

## 7. 调库实现（Python + 完整代码 + 注释）

```python
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# 分类
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, ccp_alpha=0.01, random_state=42)
clf.fit(X_train, y_train)
print(f"分类准确率: {accuracy_score(y_test, clf.predict(X_test)):.4f}")

# 回归
X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
reg = DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=42)
reg.fit(X_train_r, y_train_r)
print(f"回归MSE: {mean_squared_error(y_test_r, reg.predict(X_test_r)):.4f}")
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from collections import Counter

class CARTClassifier:
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def _gini(self, y):
        probs = np.bincount(y) / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain, best_feat, best_t = -1, None, None
        n = len(y)
        for f in range(X.shape[1]):
            vals = np.unique(X[:, f])
            for v in vals:
                left = X[:, f] <= v
                if left.sum() == 0 or left.sum() == n:
                    continue
                right = ~left
                g = self._gini(y) - (left.sum() / n) * self._gini(y[left]) - (right.sum() / n) * self._gini(y[right])
                if g > best_gain:
                    best_gain, best_feat, best_t = g, f, v
        return best_feat, best_t, best_gain

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples or len(set(y)) == 1:
            return {'leaf': True, 'value': Counter(y).most_common(1)[0][0]}
        f, t, g = self._best_split(X, y)
        if f is None:
            return {'leaf': True, 'value': Counter(y).most_common(1)[0][0]}
        left = X[:, f] <= t
        return {'feature': f, 'threshold': t,
                'left': self._build(X[left], y[left], depth + 1),
                'right': self._build(X[~left], y[~left], depth + 1)}

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def _pred(self, x, node):
        if 'leaf' in node and node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._pred(x, node['left'])
        return self._pred(x, node['right'])

    def predict(self, X):
        return np.array([self._pred(x, self.tree) for x in X])

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
cart = CARTClassifier(max_depth=5)
cart.fit(X_train, y_train)
print(f"手工CART准确率: {accuracy_score(y_test, cart.predict(X_test)):.4f}")
```

## 9. 可视化与结果理解

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=load_iris().feature_names, class_names=load_iris().target_names, filled=True)
plt.title("CART Decision Tree")
plt.savefig('cart_tree.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print(f"分类5折CV: {scores.mean():.4f} ± {scores.std():.4f}")

importances = clf.fit(X, y).feature_importances_
for name, imp in zip(load_iris().feature_names, importances):
    print(f"  {name}: {imp:.4f}")
```

## 11. 常见问题与易错点

1. **ccp_alpha 参数**：sklearn 的代价复杂度剪枝参数，$\alpha=0$ 时不剪枝，值越大树越简单。需通过交叉验证选择。
2. **基尼指数 vs 熵**：两者效果相似，基尼指数计算更快（无对数运算）。sklearn 默认用基尼。
3. **回归树外推能力差**：回归树只能预测训练数据目标值范围内的值，无法外推。
4. **特征重要性**：sklearn 的 `feature_importances_` 基于特征带来的不纯度减少量，可能偏向高基数特征。

## 12. 学习总结

CART 的核心贡献在于提出了统一的二叉树框架：无论特征是离散还是连续，每次只做最优二分划分，用基尼指数（分类）或平方误差（回归）作为不纯度度量。二叉树结构不仅计算高效（搜索空间小），还允许同一特征在路径上多次使用，表达力强于多叉树。其代价复杂度剪枝（$R_\alpha(T) = R(T) + \alpha|T|$）提供了理论完善的模型选择机制。

CART 的关键优势是同时支持分类和回归、sklearn 默认实现即基于 CART、二叉树结构天然适合集成方法。它最适合作为基学习器参与集成（如随机森林、GBDT），也适合需要快速可解释模型的场景。但单棵 CART 对数据扰动敏感、容易过拟合，工业上很少单独使用。

在知识体系中，CART 是本库中 ID3 和 C4.5 的升级替代，同时也是随机森林（Bagging+CART）、AdaBoost、GBDT、XGBoost、LightGBM 等所有主流集成方法的底层基学习器。可以说掌握了 CART 就掌握了通往现代树模型的大门。

工业实践中，CART 的 `ccp_alpha` 参数（代价复杂度剪枝）需通过交叉验证选择，而 `max_depth` 和 `min_samples_leaf` 是控制过拟合最常用的超参数。特征重要性（`feature_importances_`）可直接用于特征筛选，但需注意它可能偏向高基数特征。

## 13. 练习题与思考题（含答案）

**题目1**：数据集 $D=\{(x,y)\}$ 共6个样本，3个正例3个反例。按特征 $A \leq 2.5$ 划分后，左子集4个样本（3正1反），右子集2个样本（0正2反）。求该划分的基尼指数。

**解答**：

$\text{Gini}(D_1) = 1 - (3/4)^2 - (1/4)^2 = 1 - 9/16 - 1/16 = 6/16 = 0.375$

$\text{Gini}(D_2) = 1 - 0^2 - 1^2 = 0$

$\text{Gini}(D, A, 2.5) = \frac{4}{6} \times 0.375 + \frac{2}{6} \times 0 = 0.25$

**题目2**：CART 为什么只构建二叉树而非多叉树？

**解答**：二叉树结构使得每次划分只寻找一个切分点，搜索空间更小，计算效率更高。多叉树可以将一个多值特征一次用完，但二叉树允许同一特征在路径上多次使用，表达力更强。此外，二叉树的剪枝和理论分析更简单。

## 14. 学习路径建议

1. 理解 CART 的二叉树结构与 ID3/C4.5 多叉树的区别
2. 掌握基尼指数（分类）和平方误差（回归）两种不纯度度量
3. 学习代价复杂度剪枝原理
4. 进阶：学习随机森林（Bagging + CART）
5. 高级：学习 GBDT、XGBoost、LightGBM（Boosting + CART）
