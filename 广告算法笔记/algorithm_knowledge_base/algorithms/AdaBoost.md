# AdaBoost 学习文档

## 1. 算法基础认知

AdaBoost（Adaptive Boosting，自适应提升）由 Freund 和 Schapire 于 1997 年提出，是 Boosting 算法家族的代表作。其核心思想是：**串行训练多个弱分类器，每个新分类器重点关注前一轮被错误分类的样本，最终将所有弱分类器加权组合**。

"弱分类器"指仅比随机猜测略好的分类器（如决策树桩）。AdaBoost 能够将弱分类器提升为强分类器，这一性质有严格的理论保证。

## 2. 核心原理

AdaBoost 的工作流程：

1. 初始化所有样本的权重为 $1/N$
2. 训练第 $t$ 个弱分类器 $h_t(x)$，使用当前样本权重
3. 计算该分类器的加权错误率 $\epsilon_t$
4. 计算该分类器的权重 $\alpha_t$（错误率越低，权重越大）
5. 更新样本权重：正确分类的样本权重降低，错误分类的样本权重升高
6. 重复 2-5 共 $T$ 轮
7. 最终分类器为所有弱分类器的加权投票

**关键洞察**：每一轮都在"修复"上一轮的错误，这是一种加法模型的前向分步优化。

## 3. 数学公式与推导

**分类器权重**：

$$\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$$

**样本权重更新**：

$$D_{t+1}(i) = \frac{D_t(i)}{Z_t} \cdot \exp(-\alpha_t y_i h_t(x_i))$$

其中 $Z_t$ 是归一化因子，确保 $\sum_i D_{t+1}(i) = 1$。

**最终分类器**：

$$H(x) = \text{sign}\left(\sum_{t=1}^{T}\alpha_t h_t(x)\right)$$

**推导——为什么是指数损失**：AdaBoost 等价于在加法模型上最小化指数损失函数 $L(y, f(x)) = \exp(-y f(x))$。前向分步算法每步固定已有模型，只优化当前基分类器及其系数。

**训练误差上界**：

$$\text{training error} \leq \prod_{t=1}^{T} Z_t = \prod_{t=1}^{T} 2\sqrt{\epsilon_t(1-\epsilon_t)}$$

只要每个弱分类器好于随机（$\epsilon_t < 0.5$），训练误差就以指数速率下降。

## 4. 训练过程讲解

```
输入：训练集 {(x_i, y_i)}, y_i ∈ {-1, +1}, 弱分类器算法, 轮数 T
1. 初始化权重 D_1(i) = 1/N
2. for t = 1 to T:
   a. 用分布 D_t 训练弱分类器 h_t
   b. 计算错误率 ε_t = Σ D_t(i) · I(h_t(x_i) ≠ y_i)
   c. 计算 α_t = 0.5 · ln((1-ε_t)/ε_t)
   d. 更新权重：D_{t+1}(i) ∝ D_t(i) · exp(-α_t · y_i · h_t(x_i))
   e. 归一化 D_{t+1}
3. 输出 H(x) = sign(Σ α_t · h_t(x))
```

## 5. 应用场景

- 二分类任务（人脸检测中经典的 Viola-Jones 框架）
- 广告系统中的 CTR 预估特征选择
- 客户流失预测
- 信用评分
- 任何需要"弱学习器组合"的场景

## 6. 优缺点分析

**优点**：
- 不需要先验知识关于弱分类器的性能下界
- 自动调整样本权重，聚焦困难样本
- 不容易过拟合（尽管没有显式正则化）
- 可以使用任意弱分类器作为基学习器

**缺点**：
- 对噪声和异常值敏感（异常值会被反复赋予高权重）
- 串行训练，无法并行化
- 弱分类器太弱时效果有限

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_estimator = DecisionTreeClassifier(max_depth=1)
clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=1.0, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Estimator errors:", clf.estimator_errors_[:5])
print("Estimator weights:", clf.estimator_weights_[:5])
print(classification_report(y_test, y_pred))
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        n_samples = X.shape[0]
        y_ = np.where(y == 0, -1, 1)
        self.weights = np.ones(n_samples) / n_samples
        self.alphas = []
        self.classifiers = []

        for t in range(self.n_estimators):
            threshold = np.median(X[:, 0])
            feature_idx = 0
            predictions = np.where(X[:, feature_idx] <= threshold, -1, 1)

            incorrect = (predictions != y_)
            error = np.dot(self.weights, incorrect)

            if error >= 0.5:
                predictions = -predictions
                error = 1 - error

            if error < 1e-10:
                self.alphas.append(1.0)
                self.classifiers.append((feature_idx, threshold))
                break

            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            self.classifiers.append((feature_idx, threshold))

            self.weights *= np.exp(-alpha * y_ * predictions)
            self.weights /= np.sum(self.weights)

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for alpha, (feat_idx, threshold) in zip(self.alphas, self.classifiers):
            result += alpha * np.where(X[:, feat_idx] <= threshold, -1, 1)
        return np.where(result >= 0, 1, 0)

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=5, n_informative=2, n_redundant=0, random_state=42)
model = AdaBoostScratch(n_estimators=50)
model.fit(X, y)
print("Accuracy:", np.mean(model.predict(X) == y))
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=30, random_state=42)
clf.fit(X, y)

xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("AdaBoost 决策边界")
plt.savefig("adaboost_boundary.png", dpi=100)
plt.close()
```

随着弱分类器数量增加，决策边界逐渐变得复杂，能够拟合非线性分布。

## 10. 模型评估

- **分类指标**：Accuracy、F1-Score、AUC-ROC
- **误差分析**：查看 `estimator_errors_` 观察每轮错误率变化
- **权重分析**：查看 `estimator_weights_` 了解各弱分类器的贡献
- **调参重点**：`n_estimators`（轮数）、`learning_rate`（学习率）、基学习器复杂度

## 11. 常见问题与易错点

1. **n_estimators 过大**：虽然 AdaBoost 不容易过拟合，但过多轮数仍可能导致过拟合
2. **异常值**：噪声样本会被反复放大权重，严重时可考虑用 Gentle Boost 或 RUSBoost
3. **learning_rate 与 n_estimators 的配合**：通常减小 learning_rate 的同时增加 n_estimators
4. **基学习器选择**：默认 DecisionTreeClassifier(max_depth=1) 即决策树桩，实践中 max_depth=2~3 效果更好

## 12. 学习总结

AdaBoost 是 Boosting 方法的基石，其核心贡献是：自适应地调整样本权重，将弱学习器组合为强学习器。理解 AdaBoost 有助于掌握后续的 GBDT、XGBoost、LightGBM 等更强大的提升方法。从理论角度看，AdaBoost 与指数损失函数的前向分步优化等价。

## 13. 练习题与思考题（含答案）

**Q1**：为什么 AdaBoost 不容易过拟合？

**A1**：理论研究表明，AdaBoost 的泛化误差上界中有一个 margin 项。随着训练轮数增加，虽然模型复杂度增加，但样本到决策边界的 margin 也在增大，从而提供了额外的泛化保证。不过这并不意味着 AdaBoost 完全不会过拟合。

**Q2**：如果某个弱分类器的错误率 $\epsilon_t = 0.5$，会发生什么？

**A2**：此时 $\alpha_t = 0$，该分类器权重为零，相当于被忽略。如果 $\epsilon_t > 0.5$，AdaBoost 会翻转该分类器的预测（等价于用 $\epsilon_t' = 1-\epsilon_t$），使其变为有用的。

**Q3**：AdaBoost 中样本权重的物理意义是什么？

**A3**：权重越大，表示该样本在之前被错误分类的次数越多，下一轮的分类器会更加关注它。这是一种"查漏补缺"的策略。

## 14. 学习路径建议

1. **前置知识**：决策树、指数损失函数、集成学习基础
2. **本算法重点**：权重更新公式的推导、指数损失的等价性
3. **进阶方向**：GBDT → XGBoost → LightGBM → CatBoost
4. **推荐资源**：李航《统计学习方法》第八章、周志华《机器学习》第八章、The Original AdaBoost Paper (Freund & Schapire, 1997)
