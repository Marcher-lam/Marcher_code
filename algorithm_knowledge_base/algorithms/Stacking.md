# Stacking（堆叠泛化）学习文档

> 集成学习核心技术，通过多层堆叠实现模型融合与性能提升

---

## 1. 算法基础认知

**一句话定义**：Stacking（Stacked Generalization，堆叠泛化）是由Wolpert于1992年提出的集成学习方法，通过组合多个不同模型的预测结果作为新特征，在上层训练一个元分类器来综合各个模型的输出，实现"集思广益"的效果。

**直觉类比**：Stacking就像一个"专家会诊"过程。想象你有一个疑难杂症（预测问题），单找一个医生（单个模型）可能诊断不准确，你请来了不同专科的医生：内科医生、外科医生、影像科医生（基学习器），每个医生给出自己的诊断意见（预测概率）。然后你再把这些意见汇总，交给一个经验丰富的主任医师（元学习器）来做最终诊断。关键在于：主任医师不是简单投票，而是学习每个医生的意见如何组合最有效。这就是Stacking的核心思想——让元学习器去学习最佳的模型组合方式。

**历史背景**：
- 1992年，David Wolpert在论文"Stacked Generalization"中首次提出Stacking
- 后续被Breiman推广应用于回归问题
- 现在是Kaggle等竞赛中的常胜技术

**算法定位**：
- 类型：集成学习 → 模型融合
- 输出：分类/回归预测
- 模型类型：二级集成

**前置知识**：
- [必备]：机器学习基础（分类器、回归器）
- [必备]：交叉验证
- [推荐]：Bagging、Boosting

---

## 2. 核心原理

### 2.1 集成学习的三个层次

```
                        第一层：基学习器
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │  模型1(x) → 预测1  ──┐                              │
    │  模型2(x) → 预测2  ──┼──→ [预测1, 预测2, ..., 预测n]  │
    │  ...               ──┤              ↓              │
    │  模型n(x) → 预测n  ──┘       新特征向量            │
    │                                       │           │
    └───────────────────────────────────────┼───────────┘
                                        ▼
                        第二层：元学习器
    ┌───────────────────────────────────────────────┐
    │         最终预测 = 元学习器(新特征)            │
    └───────────────────────────────────────────┘
```

### 2.2 Stacking vs 其他集成

| 方法 | 核心思想 | 组合方式 | 学习方式 |
|------|----------|----------|----------|
| Bagging | 并行集成 | 投票平均 | 无学习 |
| Boosting | 串行集成 | 加权投票 | 依次学习 |
| **Stacking** | **堆叠集成** | **元特征** | **有学习** |

### 2.3 为什么Stacking有效？

**核心洞察**：不同模型errors不同，元学习器可以学习何时相信哪个模型！

```
模型A对样本1-100预测好
模型B对样本101-200预测好
模型C对样本201-300预测好
元学习器→学会在不同区域使用不同模型的预测
```

这就是"差异化优势"——每个模型擅长的区域不同，元学习器自动选择最优组合。

---

## 3. 数学公式与推导

### 3.1 两层Stacking

**第一层（基学习器）**：

$$Z = \{z_1, z_2, ..., z_m\}$$ 表示m个基学习器的输出

对于分类：
$$z_j = P(y|x, h_j)$$ 表示第j个基学习器的类别概率向量

对于回归：
$$z_j = h_j(x)$$ 表示第j个基学习器的预测值

**第二层（元学习器）**：

$$y_{final} = g(Z)$$

其中g是元学习器的预测函数。

### 3.2 训练流程

**算法1：Stacking训练算法**

```
输入：训练集D，验证集V
      基学习器集合H = {h1, ..., hm}
      元学习器h0

输出：最终模型

1. for each 基学习器 hj in H:
2.     在D上训练hj（使用交叉验证）
3.     对D中的每个样本，使用hj的OOF预测
4.     形成新特征矩阵：Z = [h1(x), ..., hm(x)]

5. 在新特征Z上训练元学习器h0

6. 最终模型：h(x) = h0([h1(x), ..., hm(x)])
```

### 3.3 交叉验证生成OOF

**为什么需要OOF？**

如果在同一份数据上训练基学习器和元学习器，会导致过拟合！

**解决方案**：使用Out-of-Fold预测

```python
# 伪代码
def get_oof_predictions(X, y, base_models):
    n_samples = X.shape[0]
    n_models = len(base_models)
    oof = np.zeros((n_samples, n_models))
    
    kf = KFold(n_splits=5)
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        
        for i, model in enumerate(base_models):
            model.fit(X_train, y_train)
            oof[val_idx, i] = model.predict(X_val)
    
    return oof
```

### 3.4 元特征构建

**分类任务**：
- 类别概率向量（每个类别的概率）
- 预测argmax（硬标签）
- 类别概率的熵（不确定性）

**回归任务**：
- 预测值
- 预测值方差（如果用多折）
- 残差

---

## 4. 训练过程讲解

### 4.1 完整训练流程

```
       原始数据
           │
           ▼ 划分训练/测试
    ┌───────────────┐
    │ 训练/测试划分│
    └───────┬───────┘
           │
           ▼ 多折交叉验证
    ┌───────────────┐
    │ 训练基学习器 │ ← 每个基学习器用K-1折训练
    │ 生成OOF预测  │ ← 用验证折生成OOF特征
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 构建新特征   │ ← 原始特征 + OOF预测
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 训练元学习器│ ← 在新特征上训练
    └───────┬───────┘
           ▼
    ┌───────────────┐
    │ 集成预测   │
    └───────────────┘
```

### 4.2 基学习器的选择原则

| 原则 | 说明 | 示例 |
|------|------|------|
| 多样性 | 不同类型的模型 | 树模型 + 线性模型 |
| 差异化 | 不同决策边界 | KNN + SVM |
| 独立性 | 不共享数据 | 不同特征子集 |

### 4.3 元学习器选择

| 类型 | 元学习器 | 适用场景 |
|------|----------|----------|
| 线性 | Logistic Regression | 基学习器少，简单 |
| 非线性 | Random Forest | 基学习器多 |
| 强学习器 | XGBoost | 需要强拟合能力 |

### 4.4 训练技巧

| 技巧 | 说明 |
|------|------|
| K折 | 通常5折 |
| 元特征 + 原始特征 | 可加入原始特征 |
| 概率作为特征 | 包含更多信息 |

---

## 5. 应用场景

### 5.1 竞赛常胜技术

Stacking是Kaggle等竞赛的常用技术：

```python
# 典型竞赛pipeline
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier()),
    ('svm', SVC(probability=True)),
    ('knn', KNeighborsClassifier())
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
```

### 5.2 回归问题

```python
from sklearn.ensemble import StackingRegressor

estimators = [
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('rf', RandomForestRegressor())
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor()
)
```

### 5.3 多输出问题

Stacking可以扩展到多输出回归/分类。

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **性能提升** | 通常能获得最优结果 |
| **灵活性强** | 可组合任意模型 |
| **可解释** | 知道每个模型贡献 |
| **防止过拟合** | 使用OOF避免泄露 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **计算复杂** | 需要训练多个模型 |
| **实现繁琐** | 需要写多折训练代码 |
| **时间较长** | 训练时间成倍增加 |
| **调参难** | 需要同时调多个模型 |

### 6.3 改进方向

| 改进 | 方法 |
|------|------|
| 多层Stacking | 增加更多层 |
| 特征选择 | 只选重要的OOF |
| 混合其他集成 | Blending |

---

## 7. 调库实现

### 7.1 scikit-learn实现（推荐）

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                        n_redundant=5, random_state=42)

# 定义基学习器
base_estimators = [
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(probability=True)),  # 需要probability=True
    ('lr', LogisticRegression(max_iter=1000))
]

# 定义元学习器
final_estimator = LogisticRegression()

# Stacking分类器
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_estimator,
    cv=5,  # 5折交叉验证生成OOF
    passthrough=False  # 是否包含原始特征
)

# 评估
scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
print(f"Stacking准确率: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 7.2 手动实现Stacking

```python
import numpy as np
from sklearn.model_selection import KFold


class ManualStackingClassifier:
    """手动实现Stacking"""
    
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        
    def fit(self, X, y):
        self.meta_model.fit(X, y)  # 保留接口
        
    def fit_oof(self, X, y):
        """训练OOF特征"""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # OOF predictions
        oof_preds = np.zeros((n_samples, n_models))
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for model_idx, model in enumerate(self.base_models):
                model.fit(X_train, y_train)
                oof_preds[val_idx, model_idx] = model.predict_proba(X_val)[:, 1]
        
        # 在全部数据上训练基学习器
        for model in self.base_models:
            model.fit(X, y)
            
        # ���练���学习器
        self.meta_model.fit(oof_preds, y)
        
        self.oof_preds = oof_preds
        return self
        
    def predict_proba(self, X):
        """预测概率"""
        # 第一层预测
        first_layer = np.column_stack([
            model.predict_proba(X)[:, 1] 
            for model in self.base_models
        ])
        
        # 元学习器预测
        return self.meta_model.predict_proba(first_layer)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def demo():
    """演示Stacking"""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # 数据
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    # 基学习器
    base_models = [
        DecisionTreeClassifier(max_depth=5),
        KNeighborsClassifier(n_neighbors=5),
        RandomForestClassifier(n_estimators=50)
    ]
    
    # 元学习器
    meta_model = LogisticRegression()
    
    # 训练
    stacking = ManualStackingClassifier(base_models, meta_model)
    stacking.fit_oof(X, y)
    
    # 预测
    pred = stacking.predict(X)
    acc = accuracy_score(y, pred)
    print(f"训练集准确率: {acc:.4f}")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

### 8.1 完整Stacking实现

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class StackingEnsemble:
    """完整Stacking集成"""
    
    def __init__(self, base_models, meta_model, n_folds=5, use_original_features=False):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_original_features = use_original_features
        self.fitted_base_models = []
        
    def _get_oof_predictions(self, X, y):
        """生成OOF预测"""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # 模型数量对应的类别数
        first_model = self.base_models[0]
        if hasattr(first_model, 'predict_proba'):
            n_classes = len(first_model.classes_)
        else:
            n_classes = 1
            
        oof_preds = np.zeros((n_samples, n_models * n_classes))
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for model_idx, model in enumerate(self.base_models):
                # 克隆模型
                model_copy = self._clone_model(model)
                model_copy.fit(X_train, y_train)
                
                # 预测
                if hasattr(model_copy, 'predict_proba'):
                    proba = model_copy.predict_proba(X_val)
                    if proba.ndim == 1:
                        proba = proba.reshape(-1, 1)
                else:
                    proba = model_copy.predict(X_val).reshape(-1, 1)
                
                # 填充
                feat_start = model_idx * n_classes
                oof_preds[val_idx, feat_start:feat_start+n_classes] = proba
        
        return oof_preds
    
    def _clone_model(self, model):
        """简单模型克隆"""
        import pickle
        return pickle.loads(pickle.dumps(model))
    
    def fit(self, X, y):
        """训练Stacking模型"""
        
        # 1. 生成OOF预测
        print("生成OOF预测...")
        oof_preds = self._get_oof_predictions(X, y)
        
        # 2. 构建新特征
        if self.use_original_features:
            X_meta = np.hstack([X, oof_preds])
        else:
            X_meta = oof_preds
            
        # 3. 在全部数据上训练基学习器
        print("训练基学习器...")
        self.fitted_base_models = []
        for model in self.base_models:
            model_copy = self._clone_model(model)
            model_copy.fit(X, y)
            self.fitted_base_models.append(model_copy)
            
        # 4. 训练元学习器
        print("训练元学习器...")
        self.meta_model.fit(X_meta, y)
        
        print("训练完成!")
        return self
    
    def predict(self, X):
        """预测"""
        
        # 1. 基学习器预测
        first_layer_preds = []
        for model in self.fitted_base_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                if proba.ndim == 1:
                    proba = proba.reshape(-1, 1)
            else:
                proba = model.predict(X).reshape(-1, 1)
            first_layer_preds.append(proba)
            
        # 2. 合并预测
        X_meta = np.hstack(first_layer_preds)
        
        if self.use_original_features:
            X_meta = np.hstack([X, X_meta])
            
        # 3. 元学习器预测
        return self.meta_model.predict(X_meta)
    
    def predict_proba(self, X):
        """预测概率"""
        first_layer_preds = []
        for model in self.fitted_base_models:
            proba = model.predict_proba(X)
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)
            first_layer_preds.append(proba)
            
        X_meta = np.hstack(first_layer_preds)
        
        if self.use_original_features:
            X_meta = np.hstack([X, X_meta])
            
        return self.meta_model.predict_proba(X_meta)


def stacking_comparison():
    """Stacking对比实验"""
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    # 生成数据
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 定义基学习器
    base_models = [
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(n_estimators=50),
        KNeighborsClassifier(n_neighbors=5),
        GradientBoostingClassifier(n_estimators=50)
    ]
    
    # 元学习器
    meta_model = LogisticRegression()
    
    # 单独模型性能
    print("单独模型性能:")
    for name, model in base_models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"  {name}: {acc:.4f}")
    
    # Stacking性能
    print("\nStacking性能:")
    stacking = StackingEnsemble(base_models, meta_model, n_folds=5)
    stacking.fit(X_train, y_train)
    pred = stacking.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"  Stacking: {acc:.4f}")


if __name__ == "__main__":
    stacking_comparison()
```

---

## 9. ���视化与结果理解

### 9.1 特征重要性

```python
import matplotlib.pyplot as plt

def plot_feature_importance(stacking, feature_names):
    """绘制元特征重要性"""
    
    # 如果元学习器是线性模型
    if hasattr(stacking.meta_model, 'coef_'):
        coefs = stacking.meta_model.coef_[0]
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, coefs)
        plt.xlabel('系数')
        plt.title('元学习器特征系数')
        plt.show()
```

### 9.2 模型权重可视化

```python
def plot_model_weights(base_models, meta_model):
    """可视化各模型的权重"""
    
    if hasattr(meta_model, 'coef_'):
        weights = meta_model.coef_[0]
        
        names = [type(m).__name__ for m in base_models]
        
        plt.figure(figsize=(8, 4))
        plt.bar(names, weights)
        plt.title('基学习器权重')
        plt.ylabel('权重值')
        plt.show()
```

---

## 10. 模型评估

### 10.1 性能对比

| 单独模型 | 准确率 | Stacking | 准确率提升 |
|----------|--------|----------|-----------|
| DecisionTree | 0.85 | Stacking | +5% |
| RandomForest | 0.88 | Stacking | +2% |
| GradientBoosting | 0.87 | Stacking | +3% |

### 10.2 评估代码

```python
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def evaluate_stacking(stacking, X_test, y_test):
    """评估Stacking模型"""
    
    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
```

---

## 11. 常见问题与易错点

### 11.1 数据泄露

**问题**：OOF生成时信息泄露

**解决**：严格使用交叉验证！

```python
# 错误！会导致泄露
for model in models:
    model.fit(X_train, y_train)
    oof = model.predict(X_test)  # ❌
```

### 11.2 元学习器过拟合

**问题**：元学习器在OOF上过拟合

**解决**：
- 使用简单的元学习器
- 增加交叉验证折数
- 使用原始特征

### 11.3 基学习器选择

**问题**：基学习器太相似

**解决**：选择多样化的模型！
- 树模型 + 线性模型
- 不同核的SVM
- 不同参数的KNN

### 11.4 推理时间

**问题**：推理时很慢

**解决**：
- 减少基学习器数量
- 使用更快的模型

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | 两层堆叠：基学习器 + 元学习器 |
| 核心 | OOF预测避免数据泄露 |
| 优势 | 自动学习最佳模型组合 |
| 应用 | 竞赛常用技术 |

### 12.2 公式记忆

**OOF生成**：
$$z_{ij} = h_j(x_i; D_{-i})$$

**元学习**：
$$\hat{y} = g([z_{i1}, z_{i2}, ..., z_{im}])$$

### 12.3 扩展方法

| 方法 | 说明 |
|------|------|
| Blending | 简单 hold-out 版本的Stacking |
| 多层Stacking | 更多层的堆叠 |
| 深度Stacking | 类似神经网络的多层结构 |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：为什么需要OOF预测而不是直接用验证集？

**答案**：为了避免数据泄露。如果在同一份数据上训练基学习器和元学习器，元学习器会"记住"基学习器的错误，导致过拟合。OOF通过交叉验证确保训练基学习器时"看不到"验证样本的标签。

**练习2**：Stacking和Bagging的区别？

**答案**：Bagging是并行训练多个同类模型，通过投票/平均组合；Stacking是训练多个不同类型的模型，用另一个模型（称为元学习器）来学习如何组合它��的��测。

**练习3**：如何选择基学习器？

**答案**：选择多样化的模型！最好是不同类型的模型（树、线性、核方法等），这样它们犯的错误不同，元学习器才能学到如何组合。

### 13.2 进阶思考

**思考1**：元学习器是否需要很复杂？

**答案**：通常不需要。简单的元学习器（如Logistic Regression）效果往往更好，因为可以防止过拟合。

**思考2**：可以加入原始特征吗？

**答案**：可以（passthrough=True）。但这会增加复杂度，需要更多数据来防止过拟合。

---

## 14. 学习路径建议

### 14.1 入门（1周）

| 天 | 内容 | 目标 |
|----|------|------|
| 1-2 | 集成学习基础 | 理解Bagging/Boosting |
| 3-4 | 交叉验证 | 理解OOF |
| 5-6 | Stacking原理 | 理解两层架构 |
| 7 | 代码 | 使用sklearn |

### 14.2 进阶（2周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | 完整实现 | 手动实现 |
| 2 | 调参优化 | 基学习器选择 |

### 14.3 实战（3周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | 竞赛项目 | Kaggle竞赛 |
| 2 | 多层Stacking | 增加更多层 |
| 3 | 部署 | 工程化 |

---

## 附录

### A. 重要参考

| 参考 | 链接 |
|------|------|
| Wolpert原始论文 | - |
| sklearn文档 | https://scikit-learn.org/stable/modules/ensemble.html#stacking |

### B. 代码资源

```python
# 推荐实现
# 1. sklearn.ensemble.StackingClassifier
# 2. mlxtend.regressor
# 3. 自己实现（更灵活）
```

---

**文档结束**