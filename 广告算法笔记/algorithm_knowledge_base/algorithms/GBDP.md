# GBDP 学习文档

## 1. 算法基础认知

GBDP（Gradient Boosted Decision Trees，梯度提升决策树，也常写作 GBDT）由 Friedman 于 1999 年提出，是 Boosting 方法的另一重要分支。与 AdaBoost 通过调整样本权重不同，GBDT **通过拟合前一轮模型的残差（负梯度）来构建新的决策树**。

GBDT 是广告算法和推荐系统中最重要的模型之一，XGBoost、LightGBM、CatBoost 都是在 GBDT 基础上的工程优化。

## 2. 核心原理

GBDT 采用**加法模型**和**前向分步算法**：

1. 初始化模型为一个常数值（如均值）
2. 每一轮训练一棵新的决策树，这棵树去拟合当前模型的**负梯度方向**
3. 将新树以一定学习率加到当前模型上
4. 重复直到达到预设轮数

**为什么叫"梯度"提升**？因为在平方损失下，残差 $y - F(x)$ 正好等于负梯度；推广到一般损失函数时，负梯度就是"伪残差"（pseudo-residual），每棵树拟合的是负梯度方向。

## 3. 数学公式与推导

**加法模型**：

$$F_T(x) = \sum_{t=1}^{T} \eta \cdot h_t(x) + F_0(x)$$

其中 $\eta$ 为学习率（shrinkage），$h_t(x)$ 为第 $t$ 棵决策树。

**初始化**：

$$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma)$$

对于平方损失，$F_0 = \bar{y}$。

**伪残差（负梯度）**：

$$r_{ti} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{t-1}}$$

**拟合决策树**：用 $\{(x_i, r_{ti})\}$ 训练一棵回归树，得到叶子节点区域 $R_{tj}$。

**叶子节点最优值**：

$$\gamma_{tj} = \arg\min_{\gamma} \sum_{x_i \in R_{tj}} L(y_i, F_{t-1}(x_i) + \gamma)$$

**模型更新**：

$$F_t(x) = F_{t-1}(x) + \eta \sum_{j} \gamma_{tj} \mathbb{I}(x \in R_{tj})$$

## 4. 训练过程讲解

```
输入：训练集 {(x_i, y_i)}, 损失函数 L, 轮数 T, 学习率 η
1. 初始化 F_0(x) = argmin_γ Σ L(y_i, γ)
2. for t = 1 to T:
   a. 计算伪残差 r_ti = -∂L/∂F(x_i)|_{F=F_{t-1}}
   b. 用 {(x_i, r_ti)} 拟合一棵回归树，得到叶子区域 R_tj
   c. 对每个叶子计算最优值 γ_tj
   d. 更新模型 F_t = F_{t-1} + η · Σ γ_tj · I(x ∈ R_tj)
3. 输出 F_T(x)
```

**不同损失函数的伪残差**：
- 平方损失：$r_{ti} = y_i - F_{t-1}(x_i)$
- 绝对损失：$r_{ti} = \text{sign}(y_i - F_{t-1}(x_i))$
- Log损失（分类）：$r_{ti} = y_i - \frac{1}{1+e^{-F(x_i)}}$

## 5. 应用场景

- 广告 CTR/CVR 预估（工业界最常用的模型之一）
- 搜索排序（LambdaMART）
- 推荐系统中的特征交叉
- 金融风控、保险定价
- Kaggle 竞赛中最常见的基线模型

## 6. 优缺点分析

**优点**：
- 能处理各种类型的特征（连续、离散）
- 对异常值有一定鲁棒性（通过 Huber 损失）
- 天然处理特征交互
- 预测速度快
- 可解释性较好（特征重要性）

**缺点**：
- 串行训练，无法像随机森林那样并行
- 在高维稀疏特征上不如线性模型
- 需要调参（轮数、学习率、树深度等）
- 容易过拟合（需要 early stopping）

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    subsample=0.8,
    random_state=42
)
clf.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
print("Log Loss:", log_loss(y_test, clf.predict_proba(X_test)))
print("特征重要性 Top 5:", np.argsort(clf.feature_importances_)[-5:])

X_r, y_r = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
reg.fit(Xr_train, yr_train)
print("MSE:", mean_squared_error(yr_test, reg.predict(Xr_test)))
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GBDTScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        self.init_pred = np.mean(y)
        F = np.full(len(y), self.init_pred)
        self.trees = []

        for t in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        F = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
model = GBDTScratch(n_estimators=50, learning_rate=0.1, max_depth=3)
model.fit(X, y)
pred = model.predict(X)
print("R2:", 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2))
```

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
clf.fit(X, y)

xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, levels=20)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.colorbar(label='P(class=1)')
plt.title("GBDT 概率预测")
plt.savefig("gbdt_boundary.png", dpi=100)
plt.close()

train_scores = []
for i, y_pred in enumerate(clf.staged_predict(X)):
    train_scores.append(accuracy_score(y, y_pred))
plt.plot(range(1, len(train_scores)+1), train_scores)
plt.xlabel("迭代轮数")
plt.ylabel("训练集 Accuracy")
plt.title("GBDT 训练曲线")
plt.savefig("gbdt_curve.png", dpi=100)
plt.close()
```

## 10. 模型评估

- **回归**：MSE、RMSE、MAE、$R^2$
- **分类**：Accuracy、AUC、Log Loss
- **特征重要性**：`feature_importances_`（基于分裂增益累加）
- **Early Stopping**：使用验证集监控，当验证误差连续多轮不再下降时停止训练

## 11. 常见问题与易错点

1. **n_estimators 与 learning_rate 的权衡**：较小的学习率需要更多轮数，但通常效果更好
2. **不做 early stopping**：容易过拟合，务必使用验证集监控
3. **max_depth 过大**：GBDT 中每棵树应该是"弱学习器"，max_depth 通常设为 3~8
4. **忘记处理类别特征**：sklearn 的 GBDT 不直接支持类别特征，需先编码
5. **混淆 GBDT 与随机森林**：RF 是 Bagging（并行、降方差），GBDT 是 Boosting（串行、降偏差）

## 12. 学习总结

GBDT 是工业界最成功的机器学习模型之一。其核心思想——用决策树拟合负梯度方向——简洁而强大。从 GBDT 到 XGBoost（二阶泰勒展开 + 正则化）再到 LightGBM（直方图加速 + 叶子优先生长），每一代都在工程效率上做了巨大提升，但核心思想不变。

## 13. 练习题与思考题（含答案）

**Q1**：GBDT 为什么使用回归树而不是分类树？

**A1**：因为 GBDT 每轮拟合的是连续值的负梯度（伪残差），即使最终任务是分类，中间步骤也是回归问题。分类通过损失函数的转换（如 log loss → sigmoid）实现。

**Q2**：shrinkage（学习率）的作用是什么？

**A2**：shrinkage 缩小每棵树的贡献，使模型更新更保守。这相当于给优化过程增加正则化，需要更多轮数来拟合数据，但通常能获得更好的泛化性能。类比梯度下降中的学习率。

**Q3**：GBDT 如何处理多分类问题？

**A3**：采用 One-vs-Rest 策略，为每个类别训练一个 GBDT 模型，最后通过 softmax 归一化得到概率分布。sklearn 的 `GradientBoostingClassifier` 内部就是这样实现的。

## 14. 学习路径建议

1. **前置知识**：决策树、梯度下降、AdaBoost
2. **本算法重点**：伪残差的概念、前向分步算法、shrinkage 与 early stopping
3. **进阶方向**：XGBoost（二阶优化）→ LightGBM（直方图加速）→ CatBoost（类别特征处理）
4. **推荐资源**：Friedman 原始论文 "Greedy Function Approximation: A Gradient Boosting Machine"、sklearn 文档
