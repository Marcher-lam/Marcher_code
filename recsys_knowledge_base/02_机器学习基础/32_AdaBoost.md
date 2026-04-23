# AdaBoost 学习文档

> 集成学习的经典——弱分类器组合成强分类器

---

## 1. 算法基础认知

**AdaBoost（Adaptive Boosting）** 是一种集成学习方法，通过组合多个弱分类器（如决策树桩），每个新分类器重点关注前一轮分类错误的样本。

---

## 3. 数学公式

### 3.1 算法流程

初始化样本权重 $w_i = 1/N$

对 $t = 1, 2, ..., T$：

1. 用权重 $w$ 训练弱分类器 $h_t$
2. 计算加权错误率：$\epsilon_t = \sum_{i}w_i\mathbb{1}[h_t(x_i) \neq y_i]$
3. 计算分类器权重：$\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$
4. 更新样本权重：$w_i \leftarrow w_i \cdot \exp(-\alpha_t y_i h_t(x_i))$

### 3.2 最终分类器

$$H(x) = \text{sign}\left(\sum_{t=1}^{T}\alpha_t h_t(x)\right)$$

---

## 7. 调库实现

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0
)
ada.fit(X_train, y_train)
print(f"AdaBoost准确率: {ada.score(X_test, y_test):.4f}")
```

---

## 12. 学习总结

1. AdaBoost = 组合弱分类器，关注难分样本
2. 每轮增加错分样本权重，减少正确样本权重
3. 是GBDT/XGBoost的思想前身
4. 在推荐中：排序模型的基学习器
