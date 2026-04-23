# Bagging 学习文档

> Bootstrap Aggregating，通过自助采样和集成学习降低方差、提升泛化能力。

---

## 1. 算法基础认知

### 1.1 发展背景

Bagging（Bootstrap Aggregating，自助聚合）由 Leo Breiman 于 1996 年在论文《Bagging Predictors》中提出，是集成学习方法的开创性工作。其核心思想是通过**自助采样（Bootstrap Sampling）**生成多个弱学习器，然后通过投票或平均的方式聚合预测结果，显著降低模型的方差。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 并行集成学习方法 |
| 核心 | 自助采样 + 多模型聚合 |
| 目标 | 降低方差、提升泛化能力 |
| 代表算法 | Random Forest |

### 1.3 与 Boosting 对比

| 特性 | Bagging | Boosting |
|------|--------|---------|
| 采样 | 有放回并行采样 | 序列化重采样 |
| 基学习器 | 独立训练 | 串行依赖训练 |
| 目标 | 降低方差 | 降低偏差 |
| 并行性 | 高 | 低 |
| 典型算法 | Random Forest | AdaBoost, GBDT |

---

## 2. 核心原理

### 2.1 自助采样

给定原始数据集 $D = \{(x_1, y_1), ..., (x_N, y_N)\}$，自助采样过程：

```python
for t in range(T):  # T个基学习器
    D_t = []  # 第t个采样 datasets
    for i in range(N):
        x = random.choice(D)  # 有放回采样
        D_t.append(x)
```

### 2.2 并行训练

每个基学习器 $h_t$ 在对应的自助采样数据集 $D_t$ 上独立训练：

$$h_t = \text{Learn}(D_t)$$

### 2.3 预测聚合

**分类任务（投票）**：
$$\hat{y} = \text{mode}\{h_1(x), h_2(x), ..., h_T(x)\}$$

或概率平均：
$$P(y|x) = \frac{1}{T}\sum_t P(y|x, h_t)$$

**回归任务（平均）**：
$$\hat{y} = \frac{1}{T}\sum_t h_t(x)$$

---

## 3. 数学公式与推导

### 3.1 自助采样概率

给定 N 个样本，每个样本被选中的概率：

$$P(x_i \text{ 在 } D_t \text{ 中}) = 1 - (1 - \frac{1}{N})^N$$

当 $N \to \infty$：
$$P \to 1 - e^{-1} \approx 0.632$$

即每个自助采样数据集包含约 63.2% 的原始样本。

### 3.2 方差降低

假设基学习器的预测为 $Y$，真实值为 $\mu$，方差为：

$$\text{Var}(\bar{Y}) = \frac{\sigma^2}{T} + \frac{T-1}{T}\rho\sigma^2$$

其中 $\rho$ 是基学习器之间的相关系数。

当 $\rho < 1$ 时，集成方差不增：

$$\text{Var}(\bar{Y}) < \sigma^2$$

### 3.3 泛化误差

Breiman 证明 Bagging 的泛化误差为：

$$G_{bag} \leq \frac{\bar{\rho}}{1 - \bar{\rho}^2} \cdot \bar{\Omega}$$

其中 $\bar{\Omega}$ 是平均过拟合，$\bar{\rho}$ 是平均相关系数。

---

## 4. 训练过程讲解

### 4.1 算法流程

```
Input: 原始数据集 D, 基学习器算法 Learn, T (基学习器数量)
Output: 集成模型

1. For t = 1 to T:
2.     D_t = Bootstrap(D)  # 自助采样
3.     h_t = Learn(D_t)    # 训练基学习器
4. Return 集成预测: 聚合 h_1, ..., h_T
```

### 4.2 超参数

| 参数 | 说明 | 常用值 |
|------|------|--------|
| n_estimators | 基学习器数量 | 10-500 |
| max_samples | 采样比例 | 0.5-1.0 |
| bootstrap | 是否放回采样 | True |
| max_features | 特征采样比例 | 1.0 |

### 4.3 Out-of-Bag 评估

未被采样的样本（OOB）可用于验证：

```python
# OOB 预测
h_t 对 D\D_t 的预测 → 投票 → OOB 误差
```

---

## 5. 应用场景

### 5.1 典型应用

- **回归预测**：房价、销量预测
- **分类任务**：图像分类、文本分类
- **异常检测**：多模型不一致检测
- **特征选择**：基于重要性的选择

### 5.2 sklearn 使用

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier

# Bagging 分类
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True
)
bagging.fit(X, y)
y_pred = bagging.predict(X_test)
```

---

## 6. 优缺点分析

### 6.1 优点

1. **降低方差**：多模型聚合减少预测波动
2. **并行训练**：计算效率高
3. **无需调参**：基学习器类型可选
4. **OOB 评估**：无需单独验证集

### 6.2 缺点

1. **计算成本**：T 倍基学习器开销
2. **内存需求**：存储多个模型
3. **提升有限**：不如 Boosting 显著

### 6.3 改进方向

- **Random Forest**：引入特征采样
- **Extra Trees**：极端随机树
- **Pasting**：无放回采样

---

## 7. 调库实现

### 7.1 sklearn 实现

```python
import numpy as np
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
import matplotlib.pyplot as plt

class BAGGING:
    """Bagging 集成学习
    
    参数:
        n_estimators: 基学习器数量
        max_samples: 采样比例
        bootstrap: 是否放回采样
    """
    
    def __init__(self, n_estimators=10, max_samples=1.0, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.estimators_ = []
        
    def fit(self, X, y):
        """训练 Bagging 模型
        
        参数:
            X: 数据 (n_samples, n_features)
            y: 标签 (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        n = len(X)
        
        # 训练集大小
        n_samples = int(n * self.max_samples)
        
        for t in range(self.n_estimators):
            # 自助采样
            indices = np.random.choice(n, n_samples, replace=self.bootstrap)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # 训练基学习器
            estimator = DecisionTreeClassifier()
            estimator.fit(X_boot, y_boot)
            self.estimators_.append(estimator)
        
        return self
    
    def predict(self, X):
        """预测（投票）"""
        X = np.array(X)
        
        # 获取所有基学习器的预测
        predictions = np.array([est.predict(X) for est in self.estimators_])
        
        # 投票
        result = []
        for i in range(len(X)):
            votes = predictions[:, i]
            result.append(np.bincount(votes.astype(int)).argmax())
        
        return np.array(result)
    
    def predict_proba(self, X):
        """概率预测"""
        X = np.array(X)
        
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        return np.mean(probas, axis=0)
    
    def score(self, X, y):
        """准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def demo():
    """Bagging 演示"""
    print("=== Bagging 集成学习演示 ===\n")
    
    # 生成分类数据
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                          n_redundant=2, random_state=42)
    
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y.astype(int))}")
    
    # Bagging 训练
    bagging = BAGGING(n_estimators=50, max_samples=0.8)
    bagging.fit(X, y)
    
    # 评���
    accuracy = bagging.score(X, y)
    print(f"训练准确率: {accuracy:.4f}")
    
    # 对比单模型
    single = DecisionTreeClassifier()
    single.fit(X, y)
    single_acc = np.mean(single.predict(X) == y)
    print(f"单模型准确率: {single_acc:.4f}")
    print(f"Bagging 提升: {accuracy - single_acc:.4f}")
    
    return bagging


if __name__ == "__main__":
    demo()
```

### 7.2 RandomForest

```python
# Random Forest（特征采样版）
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # 特征采样
    bootstrap=True,
    oob_score=True
)
rf.fit(X, y)
print(f"OOB 准确率: {rf.oob_score_:.4f}")
```

---

## 8. 手工代码实现

### 8.1 完整 Bagging 实现

```python
import numpy as np
from collections import Counter

class BaggingManual:
    """Bagging 手动实现
    
    参数:
        base_estimator: 基学习器类
        n_estimators: 基学习器数量
        max_samples: 采样比例
        bootstrap: 是否放回采样
    """
    
    def __init__(self, base_estimator=None, n_estimators=10, 
                 max_samples=1.0, bootstrap=True):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.estimators_ = []
        
    def fit(self, X, y):
        """训练模型
        
        参数:
            X: (n_samples, n_features)
            y: (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        n = len(X)
        
        for t in range(self.n_estimators):
            # 采样
            if self.bootstrap:
                indices = np.random.choice(n, int(n * self.max_samples), 
                                  replace=True)
            else:
                indices = np.random.choice(n, int(n * self.max_samples),
                                  replace=False)
            
            X_boot = X[indices]
            y_boot = y[indices]
            
            # 训练基学习器
            if self.base_estimator is None:
                # 使用决策树桩
                estimator = DecisionStump()
            else:
                estimator = self.base_estimator()
                
            estimator.fit(X_boot, y_boot)
            self.estimators_.append(estimator)
        
        return self
    
    def predict(self, X):
        """预测（投票）"""
        X = np.array(X)
        
        # 所有基学习器预测
        all_preds = np.array([est.predict(X) for est in self.estimators_])
        
        # 投票
        final_pred = []
        for i in range(len(X)):
            votes = all_preds[:, i]
            counter = Counter(votes)
            final_pred.append(counter.most_common(1)[0][0])
        
        return np.array(final_pred)
    
    def predict_proba(self, X):
        """概率预测"""
        X = np.array(X)
        
        probas = []
        for est in self.estimators_:
            if hasattr(est, 'predict_proba'):
                proba = est.predict_proba(X)
            else:
                # 简化
                pred = est.predict(X)
                proba = np.zeros((len(X), 2))
                proba[np.arange(len(X)), pred.astype(int)] = 1
            probas.append(proba)
        
        return np.mean(probas, axis=0)


class DecisionStump:
    """决策树桩（单层决策树）"""
    
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.predictions = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        best_loss = float('inf')
        
        # 遍历所有特征
        for f in range(n_features):
            values = X[:, f]
            thresholds = np.percentile(values, [25, 50, 75])
            
            for th in thresholds:
                # 左右划分
                left = y[values <= th]
                right = y[values > th]
                
                if len(left) == 0 or len(right) == 0:
                    continue
                
                # 多数类
                pred_left = Counter(left).most_common(1)[0][0]
                pred_right = Counter(right).most_common(1)[0][0]
                
                # 计算误差
                pred = np.full(len(y), pred_left)
                pred[values > th] = pred_right
                loss = np.mean(pred != y)
                
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = f
                    self.threshold = th
                    self.predictions = (pred_left, pred_right)
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        pred = np.full(len(X), self.predictions[0])
        pred[X[:, self.feature_idx] > self.threshold] = self.predictions[1]
        return pred


def demo_manual():
    """手动实现演示"""
    print("=== Bagging 手动实现演示 ===\n")
    
    from sklearn.datasets import make_classification
    
    # 数据
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    
    # Bagging
    bagging = BaggingManual(n_estimators=20, max_samples=0.8)
    bagging.fit(X, y)
    
    y_pred = bagging.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"样本数: {X.shape[0]}")
    print(f"预测准确率: {accuracy:.4f}")
    print(f"基学习器数: {len(bagging.estimators_)}")


if __name__ == "__main__":
    demo_manual()
```

---

## 9. 可视化与结果理解

### 9.1 误差降低可视化

```python
def plot_error_reduction():
    """可视化误差随基学习器数量变化"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_estimators = range(1, 101)
    
    # 模拟误差曲线
    single_error = 0.25
    bagging_error = single_error * np.exp(-0.05 * np.array(n_estimators))
    
    plt.figure(figsize=(10, 6))
    plt.axhline(y=single_error, color='r', linestyle='--', label='单模型')
    plt.plot(n_estimators, bagging_error, 'b-', label='Bagging')
    plt.xlabel('基学习器数量')
    plt.ylabel('预测误差')
    plt.title('Bagging 误差降低')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bagging_error.png', dpi=150)
    plt.show()
```

### 9.2 决策边界

```python
def plot_decision_boundary():
    """可视化决策边界"""
    print("""
    决策边界对比:
    
    单模型决策边界        Bagging 决策边界
    
    ● ● ● ○ ○        ● ● ● ● ○
    ● ● ● ○ ○        ● ● ● ● ○
    ────────         ● ● ● ● ●
    ○ ○ ○ ● ●        ○ ○ ○ ● ●
    ○ ○ ○ ● ●        ○ ○ ○ ● ●
    
    折线更平滑        决策更稳定
    """)
```

---

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate_bagging(model, X, y, metric='accuracy'):
    """评估 Bagging 模型"""
    y_pred = model.predict(X)
    
    if metric == 'accuracy':
        return accuracy_score(y, y_pred)
    elif metric == 'mse':
        return mean_squared_error(y, y_pred)
```

### 10.2 收敛分析

| 基学习器数量 | 单模型 | Bagging | 提升 |
|-------------|--------|--------|------|
| 1 | 0.85 | 0.85 | 0% |
| 10 | 0.85 | 0.88 | 3% |
| 50 | 0.85 | 0.91 | 6% |
| 100 | 0.85 | 0.92 | 7% |

---

## 11. 常见问题与易错点

### 11.1 基学习器数量

**问题**：选择多少基学习器合适？

**解答**：
- 太少：集成效果不明显
- 太多：计算成本高
- 经验：50-200，常用 100

### 11.2 采样比例

**问题**：max_samples 如何选择？

**解答**：
- 1.0：使用全部样本
- 0.8：常用值，增加多样性
- 过小：欠拟合

### 11.3 基学习器类型

**问题**：选择什么基学习器？

**解答**：
- 决策树：最常用
- 神经网络：计算量大
- 其他：任何稳定模型

---

## 12. 学习总结

**核心要点**：

1. **自助采样**：有放回采样生成不同数据集
2. **并行训练**：各基学习器独立训练
3. **投票聚合**：多数投票或平均
4. **方差降低**：多模型降低预测波动

**Bagging 核心优势**：
- 降低方差，提升泛化能力
- 并行训练，计算效率高
- 实现简单，效果显著

**学习建议**：

1. 理解自助采样机制
2. 掌握投票聚合原理
3. 对比单模型与集成效果

---

## 13. 练习题与思考题

### 13.1 基础练习

1. 自助采样的概率推导
2. 投票聚合的数学原理
3. 方差降低理论证明

### 13.2 进阶练习

1. 实现完整 Bagging
2. OOB 评估实现
3. 特征采样版（Random Forest）

### 13.3 思考题

1. Bagging vs Boosting 的选择
2. 基学习器多样性的作用

---

### 13.4 详细答案与解析

#### 练习1：自助采样概率

**问题**：推导样本被选中概率

**解答**：

N 个样本，一个样本在一次采样中未被选中的概率：
$$P(\text{未选中}) = (1 - \frac{1}{N})^N \approx e^{-1}$$

被选中的概率：
$$P(\text{选中}) = 1 - e^{-1} \approx 0.632$$

#### 练习2：方差降低

**问题**：为什么 Bagging 能降低方差？

**解答**：

设基学习器预测 $X_i$ 的方差为 $\sigma^2$，相关系数为 $\rho$：

$$\text{Var}(\bar{X}) = \frac{\sigma^2}{T} + \frac{T-1}{T}\rho\sigma^2$$

当 $\rho < 1$ 时：
$$\text{Var}(\bar{X}) < \sigma^2$$

---

## 14. 学习路径建议

### 入门阶段

1. 掌握机器学习基础
2. 理解集成学习思想
3. 学习 Bagging 原理

### 进阶阶段

1. 实现 Bagging
2. 掌握 Random Forest
3. 对比不同集成方法

### 高级阶段

1. 改进基学习器
2. 特征重要性分析
3. 深度集成学习

**推荐路线**：

```
决策树 → Bagging → Random Forest → 
Boosting → GBDT → XGBoost → Stacking
```

**Bagging 是集成学习的基石，熟练掌握它是进入集成学习领域的第一步。**