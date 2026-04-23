# CART 算法学习文档

## 1. 算法基础认知

CART（Classification and Regression Tree，分类与回归树）是由美国统计学家Leo Breiman等人于1984年提出的决策树学习算法。与ID3和C4.5不同，CART是一种通用的决策树框架，可以同时处理分类问题和回归问题，这使其成为机器学习领域中最广泛使用的决策树算法之一。

CART算法的核心特点包括三个方面。首先，CART生成的是二叉树（Binary Tree），即每个内部节点最多有两个子节点，这与ID3和C4.5的多叉树结构不同。二叉树结构使得树的深度可能更大，但每个分裂更加纯粹。

其次，CART使用基尼指数（Gini Index）作为特征选择准则。基尼指数度量的是数据集的纯度，与信息熵类似但计算更简单。基尼指数越小，数据集越纯净。

第三，CART具有完整的剪枝策略。CART使用代价复杂度剪枝（Cost-Complexity Pruning，CCP）来简化决策树，这是一种后剪枝技术，通过在验证集上评估子树的代价来选择最优的树大小。

CART算法在商业和工业应用中非常成功。sklearn中的DecisionTreeClassifier默认使用CART算法（通过设置criterion='gini'），许多流行的集成算法（如随机森林、GBDT）也使用CART作为基学习器。

## 2. 核心原理

CART算法的核心原理建立在基尼指数的基础上，同时也引入了代价复杂度剪枝来处理过拟合问题。

对于分类问题，CART使用基尼指数（Gini Index）作为分裂准则。设数据集D包含K个类别，每个类别的比例为 $p_k$，则基尼指数定义为：

$$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$$

基尼指数度量的是从数据集中随机抽取两个样本，它们属于不同类别的概率。当数据集完全纯净时（只有一个类别），基尼指数为0；当数据均匀分布在各类别时，基尼指数最大。

对于特征A的分裂，假设A有m个不同取值，可以将数据划分为m个子集 $D_1, D_2, ..., D_m$，分裂后的加权基尼指数为：

$$Gini_A(D) = \sum_{i=1}^{m} \frac{|D_i|}{|D|} Gini(D_i)$$

特征A的信息增益（这里称为基尼增益）为：

$$GiniGain(A) = Gini(D) - Gini_A(D)$$

CART选择基尼增益最大的特征进行分裂。

对于回归问题，CART使用均方误差（Mean Squared Error，MSE）作为分裂准则。设数据集D的目标值为 $y_1, y_2, ..., y_n$，均值为 $\bar{y}$，则MSE定义为：

$$MSE(D) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$$

特征的分裂增益为分裂前后MSE的减少量。

## 3. 数学公式与推导

### 3.1 基尼指数的数学推导

基尼指数源于经济学中的洛伦兹曲线度量不平等的程度。在机器学习中，基尼指数被用来度量数据集的纯度。

设数据集D包含N个样本，每个样本属于K个类别中的一个。设类别k的样本数量为 $n_k$，则类别k的比例为 $p_k = n_k / N$。基尼指数定义为：

$$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$$

这个公式的直观理解是：如果从数据集中随机抽取两个样本，它们属于相同类别的概率是 $\sum_k p_k^2$，因此属于不同类别的概率是 $1 - \sum_k p_k^2$。

对于二分类问题，设正类比例为p，负类比例为1-p，则：

$$Gini(D) = 1 - p^2 - (1-p)^2 = 2p(1-p)$$

这个函数在 $p=0.5$ 时取得最大值0.5，在 $p=0$ 或 $p=1$ 时取得最小值0。

### 3.2 基尼指数与信息熵的关系

基尼指数和信息熵都度量数据集的纯度，它们之间存在近似关系。

根据泰勒展开，当p接近0.5时：

$$H(p) = -p \log_2 p - (1-p) \log_2(1-p) \approx 1 - (p - 0.5)^2$$

而：

$$Gini(p) = 2p(1-p) = 0.5 - 2(p - 0.5)^2$$

因此，$Gini(p) approx 0.5 \times \log_2(e) \times H(p)$，二者趋势相同但度量范围不同。基尼指数的计算更简单，不需要对数运算，因此在实际应用中更高效。

### 3.3 二叉分裂的数学推导

CART的二叉分裂将数据划分为两个子集：$D_L$（左子集）和 $D_R$（右子集）。对于离散特征，CART仍然创建二叉树：如果特征有m个取值，需要考虑 $2^m - 2$ 种可能的二分方式，选择基尼增益最大的方式。

对于连续特征，二分更为简单：找到一个阈值T，将数据划分为小于等于T和大于T的两个子集。

设左子集的样本比例为 $\alpha = |D_L|/|D|$，右子集的比例为 $1-\alpha$，则加权基尼指数为：

$$Gini_{split}(D) = \alpha Gini(D_L) + (1-\alpha) Gini(D_R)$$

基尼增益为：

$$GiniGain = Gini(D) - Gini_{split}(D)$$

CART选择使基尼增益最大的阈值T。

### 3.4 代价复杂度剪枝

代价复杂度剪枝（CCP）是CART的后剪枝策略，核心思想是在树的复杂度和预测误差之间寻求平衡。

设原始决策树为 $T_0$，依次剪掉一个节点得到 $T_1, T_2, ..., T_n$（只有根节点的树）。定义树的复杂度为叶节点数量 $|T|$。

定义树的整体代价为：

$$R_{\alpha}(T) = R(T) + \alpha |T|$$

其中 $R(T)$ 是树T在验证集上的误分类率或MSE，$\alpha$ 是复杂度参数。

CCP找到使验证集代价最小的树序列。对于每个 $\alpha$，存在唯一的最优树 $T(\alpha)$。

剪枝过程中，定义非叶子节点的"有效叶子数"来衡量剪枝的收益。对于非叶子节点t，定义：

$$\alpha_t = \frac{R(t) - R(T_t)}{|T_t| - 1}$$

其中 $R(t)$ 是节点t的代价，$R(T_t)$ 是以t为根的子树的代价，$|T_t|$ 是子树的叶子数。

依次剪掉 $\alpha$ 最小的节点，得到嵌套的树序列。

## 4. 训练过程讲解

### 4.1 分类树的训练

CART分类树的训练过程如下：

第一步，准备训练数据。数据可以是离散特征或连续特征，CART可以自动处理这两种类型的特征。

第二步，对每个候选特征计算基尼指数和基尼增益。对于离散特征，计算所有可能的二分方式的基尼增益；对于连续特征，找到最优二分阈值并计算基尼增益。

第三步，选择基尼增益最大的特征和分裂方式创建当前节点。

第四步，递归对左右子集重复第二步和第三步，直到满足停止条件。停止条件包括：节点纯度（所有样本属于同一类）、节点样本数少于阈值、没有可用特征等。

### 4.2 回归树的训练

CART回归树的训练过程与分类树类似，但使用MSE作为分裂准则：

第一步，对每个候选特征找到最优二分阈值，计算分裂后的MSE减少量。

第二步，选择使MSE减少最多的特征和阈值进行分裂。

第三步，递归构建子树。

第四步，回归树的叶节点是一个常数，通常是该叶子节点所有样本的目标均值。

### 4.3 剪枝过程

CART使用CCP进行后剪枝：

第一步，在训练集上构建、完全生长的决策树。

第二步，在验证集上评估每个候选树的代价。

第三步，依次剪掉有效 $\alpha$ 最小的节点，得到嵌套的树序列。

第四步，使用交叉验证或验证集选择最优的树大小。

## 5. 应用场景

CART算法在实际应用中广泛的场景，特别是在以下领域：

在医疗诊断领域，CART可以用于辅助诊断。由于其可解释性强，医生可以直接理解决策树的分类逻辑。例如，根据患者的各项检查指标，CART可以预测是否需要进行进一步检查。

在金融风控领域，CART可以用于信用评分。CART生成的决策规则简单明了，可以直接用于贷款审批。例如，收入、负债比率等特征的阈值可以帮助快速评估���用��险。

在客户关系管理领域，CART可以用于客户细分。通过分析客户的特征，CART可以识别出不同类型的客户，帮助企业制定针对性的营销策略。

在工业质量控制领域，CART可以用于产品缺陷检测。通过分析生产过程中的各项参数，CART可以识别出可能导致缺陷的关键因素。

在营销分析领域，CART可以用于购买预测。通过分析用户的行为特征，CART可以预测用户是否会购买某产品，帮助确定目标客户群体。

## 6. 优缺点分析

### 6.1 优点

CART算法具有以下显著优点：

首先，CART可以同时处理分类和回归问题。这种通用性使得CART成为一个强大的工具，可以在不同的场景中重复使用。

其次，CART生成二叉树，结构更简洁。每个分裂只有两个分支，这使得树的深度可能更大，但每个决策更清晰。

第三，CART的CCP剪枝策略可以有效地控制过拟合。通过在验证集上评估代价，CART可以找到最优的树大小。

第四，CART使用基尼指数，计算更高效。不需要像信息熵那样计算对数，在大规模数据集上训练速度更快。

第五，CART具有良好的可解释性。生成的决策树可以清晰地展示分类规则，便于业务理解。

### 6.2 缺点

CART算法也存在一些缺点：

首先，CART只能处理二叉分裂。对于取值较多的离散特征，二叉树可能需要多层级分裂才能完全区分这些取值。

其次，CART的剪枝策略是后剪枝，需要额外的验证集。这增加了数据需求和计算开销。

第三，CART对噪声敏感。由于树会完全拟合训练数据，噪声可能导致不必要的分裂，降低泛化能力。

第四，CART不稳定。数据的微小变化可能导致完全不同的树结构。这是决策树的普遍问题。

第五，CART偏向于选择取值较多的特征。与ID3类似，需要通过剪枝来控制这个问题。

## 7. 调库实现

### 7.1 使用sklearn实现CART分类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.datasets import load_iris


def cart_classification():
    """CART分类树 - sklearn实现"""
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    clf = DecisionTreeClassifier(
        criterion='gini',           # CART使用基尼指数
        max_depth=5,               # 限制树的深度
        min_samples_split=5,       # 最小分裂样本数
        min_samples_leaf=2,       # 叶节点最小样本数
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("=" * 60)
    print("CART分类树 - sklearn实现")
    print("=" * 60)
    print(f"\n准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred)}")
    
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\n5折交叉验证: {cv_scores}")
    print(f"平均准确率: {cv_scores.mean():.4f}")
    
    print(f"\n特征重要性:")
    for name, imp in zip(iris.feature_names, clf.feature_importances_):
        print(f"  {name}: {imp:.4f}")
    
    return clf


def cart_binary_classification():
    """CART二分类示例"""
    
    np.random.seed(42)
    
    n_pos = 100
    n_neg = 100
    
    X_pos = np.random.normal(2, 1, (n_pos, 2))
    X_neg = np.random.normal(-2, 1, (n_neg, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"\n二分类准确率: {accuracy_score(y_test, y_pred):.4f}")
    
    return clf


if __name__ == '__main__':
    clf = cart_classification()
    clf = cart_binary_classification()
```

### 7.2 使用sklearn实现CART回归

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def cart_regression():
    """CART回归树 - sklearn实现"""
    
    np.random.seed(42)
    
    n_samples = 200
    X = np.sort(np.random.rand(n_samples) * 10, axis=0).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(n_samples) * 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    reg = DecisionTreeRegressor(
        criterion='squared_error',  # 使用MSE作为准则
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    print("=" * 60)
    print("CART回归树 - sklearn实现")
    print("=" * 60)
    print(f"\nMSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    return reg


def visualize_regression():
    """可视化CART回归树"""
    
    np.random.seed(42)
    
    X = np.sort(np.random.rand(100) * 10).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1
    
    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(X, y)
    
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    y_pred = reg.predict(X_plot)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='真实值')
    plt.plot(X_plot, y_pred, 'r-', linewidth=2, label='预测值')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('CART回归树')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cart_regression.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    cart_regression()
    visualize_regression()
```

### 7.3 代码解释

上述代码展示了sklearn中CART算法的实现方式。关键点包括：

- `criterion='gini'`：使用基尼指数作为分裂准则，对应CART算法
- `criterion='squared_error'`：对于回归树，使用MSE作为分裂准则
- 其他参数如`max_depth`、`min_samples_split`、`min_samples_leaf`用于控制树的复杂度，避免过拟合

sklearn的DecisionTreeRegressor默认生成二叉回归树，每个叶节点是叶子节点样本的目标均值。

## 8. 手工代码实现

### 8.1 完整NumPy实现 - 分类树

```python
import numpy as np
from collections import Counter


class CARTClassifier:
    """
    CART分类树的纯NumPy实现
    
    使用基尼指数作为分裂准则，生成二叉树
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_names = None
    
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
    
    def _gini_gain(self, X, y, feature_idx, threshold=None):
        """计算基尼增益"""
        gini_parent = self._gini(y)
        n = len(y)
        
        if threshold is not None:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = X[:, feature_idx] > threshold
            
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            
            if n_left == 0 or n_right == 0:
                return 0.0
            
            gini_left = self._gini(y[left_mask])
            gini_right = self._gini(y[right_mask])
            
            gini_split = (n_left / n) * gini_left + (n_right / n) * gini_right
        else:
            unique_values = np.unique(X[:, feature_idx])
            
            if len(unique_values) == 0:
                return 0.0
            
            gini_split = 0.0
            n_total = n
            
            for value in unique_values:
                mask = X[:, feature_idx] == value
                n_v = np.sum(mask)
                
                if n_v == 0:
                    continue
                
                gini_v = self._gini(y[mask])
                gini_split += (n_v / n_total) * gini_v
        
        return gini_parent - gini_split
    
    def _find_best_split(self, X, y, feature_indices):
        """找到最佳分裂特征和阈值"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for idx in feature_indices:
            unique_values = np.unique(X[:, idx])
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                gain = self._gini_gain(X, y, idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, feature_indices, depth=0):
        """递归构建决策树"""
        n_samples = len(y)
        n_classes = len(set(y))
        
        if n_classes == 1:
            return {'class': y[0]}
        
        if len(feature_indices) == 0:
            return {'class': Counter(y).most_common(1)[0][0]}
        
        if n_samples < self.min_samples_split:
            return {'class': Counter(y).most_common(1)[0][0]}
        
        if self.max_depth is not None and depth >= self.max_depth:
            return {'class': Counter(y).most_common(1)[0][0]}
        
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, y, feature_indices
        )
        
        if best_feature is None or best_gain <= 0:
            return {'class': Counter(y).most_common(1)[0][0]}
        
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
                'class': Counter(y).most_common(1)[0][0]
            }
        
        if np.sum(right_mask) >= self.min_samples_leaf:
            tree['children']['>'] = self._build_tree(
                X[right_mask], y[right_mask], remaining_features, depth + 1
            )
        else:
            tree['children']['>'] = {
                'class': Counter(y).most_common(1)[0][0]
            }
        
        return tree
    
    def fit(self, X, y, feature_names=None):
        """训练CART决策树"""
        X = np.array(X)
        y = np.array(y)
        
        n_features = X.shape[1]
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        self.feature_names = feature_names
        
        feature_indices = list(range(n_features))
        
        self.tree = self._build_tree(X, y, feature_indices)
        
        return self
    
    def _predict_single(self, x, tree):
        """预测单个样本"""
        if 'class' in tree:
            return tree['class']
        
        feature_idx = tree['feature']
        threshold = tree['threshold']
        
        if x[feature_idx] <= threshold:
            return self._predict_single(x, tree['children']['<='])
        else:
            return self._predict_single(x, tree['children']['>'])
    
    def predict(self, X):
        """预测新样本"""
        X = np.array(X)
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def print_tree(self, tree=None, indent=""):
        """打印决策树"""
        if tree is None:
            tree = self.tree
        
        if 'class' in tree:
            print(f"{indent}叶节点: 类别={tree['class']}")
            return
        
        feature_idx = tree['feature']
        feature_name = self.feature_names[feature_idx]
        threshold = tree['threshold']
        
        print(f"{indent}[分裂: {feature_name} <= {threshold:.4f}]")
        
        print(f"{indent}  <=:")
        self.print_tree(tree['children']['<='], indent + "    ")
        
        print(f"{indent}  >:")
        self.print_tree(tree['children']['>'], indent + "    ")


class CARTRegressor:
    """
    CART回归树的纯NumPy实现
    
    使用MSE作为分裂准则
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def _mse(self, y):
        """计算均方误差"""
        if len(y) == 0:
            return 0.0
        
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
    
    def _mse_gain(self, X, y, feature_idx, threshold):
        """计算MSE减少量"""
        mse_parent = self._mse(y)
        n = len(y)
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        mse_left = self._mse(y[left_mask])
        mse_right = self._mse(y[right_mask])
        
        mse_split = (n_left / n) * mse_left + (n_right / n) * mse_right
        
        return mse_parent - mse_split
    
    def _find_best_split(self, X, y, feature_indices):
        """找到最佳分裂"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for idx in feature_indices:
            unique_values = np.unique(X[:, idx])
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                gain = self._mse_gain(X, y, idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, feature_indices, depth=0):
        """构建回归树"""
        n_samples = len(y)
        
        if n_samples < self.min_samples_split:
            return {'value': np.mean(y)}
        
        if self.max_depth is not None and depth >= self.max_depth:
            return {'value': np.mean(y)}
        
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, y, feature_indices
        )
        
        if best_feature is None or best_gain <= 0:
            return {'value': np.mean(y)}
        
        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'children': {}
        }
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        remaining_features = [f for f in feature_indices if f != best_feature]
        
        if np.sum(left_mask) < self.min_samples_leaf:
            tree['children']['<='] = {'value': np.mean(y)}
        else:
            tree['children']['<='] = self._build_tree(
                X[left_mask], y[left_mask], remaining_features, depth + 1
            )
        
        if np.sum(right_mask) < self.min_samples_leaf:
            tree['children']['>'] = {'value': np.mean(y)}
        else:
            tree['children']['>'] = self._build_tree(
                X[right_mask], y[right_mask], remaining_features, depth + 1
            )
        
        return tree
    
    def fit(self, X, y):
        """训练回归树"""
        X = np.array(X)
        y = np.array(y)
        
        n_features = X.shape[1]
        feature_indices = list(range(n_features))
        
        self.tree = self._build_tree(X, y, feature_indices)
        
        return self
    
    def _predict_single(self, x, tree):
        """预测单个样本"""
        if 'value' in tree:
            return tree['value']
        
        feature_idx = tree['feature']
        threshold = tree['threshold']
        
        if x[feature_idx] <= threshold:
            return self._predict_single(x, tree['children']['<='])
        else:
            return self._predict_single(x, tree['children']['>'])
    
    def predict(self, X):
        """预测新样本"""
        X = np.array(X)
        return np.array([self._predict_single(x, self.tree) for x in X])


def demo():
    """CART算法演示"""
    
    print("=" * 60)
    print("CART分类树 - 手工实现")
    print("=" * 60)
    
    X = np.array([
        [0, 30, 85], [0, 28, 90], [1, 26, 95], [2, 20, 80],
        [2, 18, 70], [2, 15, 65], [1, 16, 60], [0, 25, 75],
        [0, 22, 80], [1, 24, 55], [2, 22, 50], [0, 30, 50],
        [1, 28, 70], [2, 25, 60],
    ])
    
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    
    clf = CARTClassifier(max_depth=5)
    clf.fit(X, y, feature_names=['天气', '温度', '湿度'])
    
    print("\n决策树结构:")
    clf.print_tree()
    
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\n训练准确率: {accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("CART回归树 - 手工实现")
    print("=" * 60)
    
    np.random.seed(42)
    X = np.sort(np.random.rand(100) * 10).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1
    
    reg = CARTRegressor(max_depth=5)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    print(f"\n训练MSE: {mse:.4f}")


if __name__ == '__main__':
    demo()
```

### 8.2 代码关键点解释

上述代码是CART算法的完整NumPy实现，包括分类树和回归树：

分类树的`_gini`方法计算基尼指数，公式为 $Gini(D) = 1 - \sum_k p_k^2$。`_gini_gain`方法计算基尼增益，用于选择最佳分裂特征。

回归树的`_mse`方法计算均方误差，公式为 $MSE(D) = \frac{1}{n} \sum_i (y_i - \bar{y})^2$。`_mse_gain`方法计算MSE减少量。

两个树都使用二叉分裂，通过`_find_best_split`方法找到使增益最大的特征和阈值，然后递归构建子树。

## 9. 可视化与结果理解

### 9.1 决策边界可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def visualize_decision_boundary():
    """可视化CART决策边界"""
    
    np.random.seed(42)
    
    X1 = np.random.normal(-1, 1, 100)
    X2 = np.random.normal(1, 1, 100)
    X = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
    y = np.array([0] * 100 + [1] * 100)
    
    clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    clf.fit(X, y)
    
    xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1 = axes[0]
    ax1.contourf(xx, yy, Z, alpha=0.3)
    ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', alpha=0.5)
    ax1.set_xlabel('特征1')
    ax1.set_ylabel('特征2')
    ax1.set_title('CART决策边界')
    
    ax2 = axes[1]
    plot_tree(clf, feature_names=['特征1', '特征2'], 
             class_names=['类0', '类1'], filled=True, ax=ax2)
    ax2.set_title('CART决策树')
    
    plt.tight_layout()
    plt.savefig('cart_visualization.png', dpi=150)
    plt.show()


def compare_gini_entropy():
    """比较基尼指数和信息熵"""
    
    p = np.linspace(0.001, 0.999, 100)
    gini = 2 * p * (1 - p)
    entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, gini, 'b-', label='基尼指数', linewidth=2)
    plt.plot(p, entropy, 'r-', label='信息熵', linewidth=2)
    plt.xlabel('正类比例 p')
    plt.ylabel('纯度度量')
    plt.title('基尼指数 vs 信息熵')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cart_compare.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    visualize_decision_boundary()
    compare_gini_entropy()
```

### 9.2 结果理解

通过可视化可以看到，CART的决策边界是轴平行的直线（或者多段直线），这是决策树的特点。决策树通过一系列的"特征 <= 阈值"判断来划分特征空间。

基尼指数和信息熵的曲线形状相似，都是在p=0.5时达到最大，在p=0或1时达到最小。二者的趋势一致，但度量范围不同。

## 10. 模型评估

### 10.1 分类树评估

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


def evaluate_cart_classifier():
    """评估CART分类树"""
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    clf.fit(X, y)
    y_pred = clf.predict(X)
    
    print("=" * 60)
    print("CART分类树评估")
    print("=" * 60)
    
    print(f"\n准确率: {accuracy_score(y, y_pred):.4f}")
    print(f"精确率: {precision_score(y, y_pred, average='weighted'):.4f}")
    print(f"召回率: {recall_score(y, y_pred, average='weighted'):.4f}")
    print(f"F1分数: {f1_score(y, y_pred, average='weighted'):.4f}")
    
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y, y_pred))
    
    print(f"\n分类报告:")
    print(classification_report(y, y_pred))
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv)
    print(f"\n5折交叉验证: {cv_scores}")
    print(f"平均准确率: {cv_scores.mean():.4f}")


if __name__ == '__main__':
    evaluate_cart_classifier()
```

### 10.2 回归树评估

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_cart_regressor():
    """评估CART回归树"""
    
    np.random.seed(42)
    X = np.sort(np.random.rand(200) * 10).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(200) * 0.1
    
    reg = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    reg.fit(X, y)
    y_pred = reg.predict(X)
    
    print("=" * 60)
    print("CART回归树评估")
    print("=" * 60)
    
    print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    print(f"R²: {r2_score(y, y_pred):.4f}")
    
    cv_scores = cross_val_score(reg, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"\n5折交叉验证MSE: {-cv_scores}")
    print(f"平均RMSE: {np.sqrt(-cv_scores.mean()):.4f}")


if __name__ == '__main__':
    evaluate_cart_regressor()
```

### 10.3 评估指标解释

对于分类树，准确率、精确率、召回率、F1分数是常用的评估指标。交叉验证可以评估模型的稳定性和泛化能力。

对于回归树，MSE、RMSE、R²是常用的评估指标。MSE是误差的平方，对异常值更敏感；R²表示模型解释的方差比例。

## 11. 常见问题与易错点

### 11.1 过拟合问题

CART完全生长时可能会过拟合训练数据。解决方法是使用预剪枝（max_depth、min_samples_split等参数）或后剪枝（CCP）。

### 11.2 特征选择偏向

CART的基尼指数也倾向于选择取值较多的特征。通过设置适当的预剪枝参数可以缓解这个问题。

### 11.3 数据不平衡

当类别分布不均匀时，决策树可能偏向多数类。可以使用class_weight参数来处理类别不平衡。

### 11.4 连续特征阈值

CART对连续特征使用二分阈值，这可能不是最优的分裂方式。但二叉分裂使得树的深度可能更大，每个决策更清晰。

### 11.5 树的不稳定性

决策树对训练数据的微小变化可能很敏感。这是决策树的普遍问题，可以通过集成方法来缓解（如随机森林）。

## 12. 学习总结

CART算法是经典的决策树学习算法，可以同时处理分类和回归问题。CART的核心贡献包括：

使用基尼指数作为分裂准则。基尼指数与信息熵类似但计算更高效，这使得CART在大规模数据集上训练速度更快。

生成二叉树结构。CART的每个节点最多有两个子节点，这使得树的决策更清晰，便于理解。

使用CCP后剪枝策略。CART通过代价复杂度剪枝来控制过拟合���在���的复杂度和预测误差之间寻求平衡。

CART是许多高级算法的基础，包括随机森林、GBDT等。这些算法使用CART作为基学习器，结合集成学习的思想来提高预测性能和稳定性。

学习CART对于理解决策树算法的原理和实现非常重要，也为学习更高级的集成算法打下基础。

## 13. 练习题与思考题与思考题

### 13.1 选择题

1. CART算法使用什么准则选择分裂特征？
   A. 信息增益
   B. 信息增益率
   C. 基尼指数
   D. MSE
   答案：C

2. CART生成什么类型的决策树？
   A. 多叉树
   B. 二叉树
   C. 平衡树
   D. AVL树
   答案：B

3. 基尼指数 $Gini(p)$ 的最大值是多少（二分类问题）？
   A. 0.5
   B. 1.0
   C. 0.25
   D. 1.0
   答案：A

### 13.2 计算题

假设数据集有10个样本，正类6个，负类4个。计算基尼指数。

解：正类比例 $p = 6/10 = 0.6$，负类比例 $= 0.4$

$$Gini(D) = 1 - (0.6)^2 - (0.4)^2 = 1 - 0.36 - 0.16 = 0.48$$

### 13.3 思考题

1. 基尼指数与信息熵有什么相同点和不同点？
   
   答案：相同点：都度量数据集的纯度，都是在类别均匀分布时最大，在类别单一时报小说，都用于特征选择。不同点：计算公式不同，基尼指数不需要对数运算，计算更高效；取值范围不同，基尼指数的最大值是0.5（二分类），而信息熵的最大值是1（二分类）。

2. 为什么CART生成二叉树而不是多叉树？
   
   答案：二叉树更简单，每个节点的决策只有两个分支，便于理解；二叉树使得每个分裂更纯粹，可以更精细地划分数据；二叉树可以通过多层级分裂来覆盖多叉树的能力。

3. CCP剪枝的原理是什么？
   
   答案：代价复杂度剪枝（CCP）的核心是在树的复杂度和预测误差之间寻求平衡。通过定义整体代价 $R_\alpha(T) = R(T) + \alpha |T|$，其中 $R(T)$ 是验证集上的误差，$|T|$ 是叶节点数量，$\alpha$ 是复杂度参数。依次剪掉使代价增加最少的节点，得到嵌套的树序列。使用交叉验证选择最优的树。

### 13.4 编程题

使用sklearn实现CART分类树和回归树，并比较不同参数对模型性能的影响。

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
reg = DecisionTreeRegressor(criterion='squared_error', max_depth=5)

print(f"分类树准确率: {cross_val_score(clf, X, y, cv=5).mean():.4f}")
print(f"回归树R²: {cross_val_score(reg, X, y, cv=5).mean():.4f}")
```

## 14. 学习路径建议建议

学习CART算法应该按照以下路径进行：

首先，理解决策树的基础概念，包括节点、分支、叶子等。这是理解ID3、C4.5、CART的基础。

然后，理解基尼指数的数学原理和计算方法。基尼指数是CART的核心准则，需要深入理解其含义和与信息熵的关系。

第三，理解CART的二叉分裂策略。CART生成二叉树，每个节点最多有两个子节点，这与ID3和C4.5的多叉树不同。

第四，理解CCP后剪枝策略。剪枝是控制过拟合的重要手段，需要理解其原理和实现方式。

第五，学习如何使用sklearn实现CART算法。sklearn的DecisionTreeClassifier和DecisionTreeRegressor是CART的标准实现。

第六，学习CART回归树的实现。回归树使用MSE作为分裂准则，叶节点是目标均值。

最后，可以进一步学习基于CART的集成算法，如随机森林和GBDT。这些算法使用CART作为基学习器，在CART的基础上通过集成学习来提高性能。