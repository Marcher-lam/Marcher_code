# CatBoost 学习文档

> 处理类别特征能力最强的梯度提升框架，Yandex 开源的高性能 GBDT 实现。

---

## 1. 算法基础认知

### 1.1 发展背景

CatBoost（Categorical Boosting）由 Yandex 于 2017 年发布，是专门针对类别特征优化的梯度提升框架。与 XGBoost、LightGBM 不同，CatBoost 内置了强大的类别特征处理能力，无需手动编码即可自动处理类别特征。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 梯度提升框架 |
| 核心创新 | 目标编码 + 有序提升 |
| 类别特征 | 原生支持 |
| 计算效率 | 高（对称树） |

### 1.3 模型系列

| 模型 | 参数 | 树深度 | 特点 |
|------|------|--------|------|
| CatBoost-Small | 256 | 6 | 轻量快速 |
| CatBoost-Medium | 512 | 8 | 平衡 |
| CatBoost-Large | 1024 | 10 | 高精度 |

---

## 2. 核心原理

### 2.1 目标编码（Target Encoding）

将类别特征转换为数值，同时避免过拟合：

$$\text{encoded}(x) = \frac{\sum_{i: x_i=x} y_i + \lambda \cdot \mu}{\text{count}(x) + \lambda}$$

其中 $\mu$ 是全局均值，$\lambda$ 是平滑系数。

### 2.2 有序提升（Ordered Boosting）

解决梯度提升中的预测偏移问题：

1. 打乱训练样本顺序
2. 使用前 m-1 个样本训练第 m 个
3. 计算梯度时使用历史信息

### 2.3 对称树（Symmetric Tree）

CatBoost 使用对称树结构：

- 所有叶子节点深度相同
- 所有内部节点使用相同特征和阈值
- 预测速度快

### 2.4 与 XGBoost 对比

| 特性 | CatBoost | XGBoost |
|------|---------|---------|
| 类别特征 | 原生支持 | 需编码 |
| 过拟合防护 | 有序提升 | L1/L2 |
| 树结构 | 对称 | 不对称 |
| GPU 加速 | 原生支持 | CUDA |

---

## 3. 数学公式与推导

### 3.1 梯度计算

给定损失函数 $L(y, \hat{y})$，第 m 步的伪残差：

$$g_{mi} = \frac{\partial L(y_i, \hat{y}_{m-1,i})}{\partial \hat{y}_{m-1,i}}$$

### 3.2 叶子节点值

对于回归任务：
$$w_{mj} = -\frac{\sum_{i \in R_{mj}} g_{mi}}{\sum_{i \in R_{mj}} |g_{mi}|}$$

### 3.3 类别特征编码

CatBoost 使用多种编码：

1. **标签编码**：数值化类别
2. **目标编码**：使用标签均值
3. **Ordered 编码**：防止过拟合

$$code(x_k) = \frac{1}{k} \sum_{i=1}^k y_{(i)}$$

### 3.4 对称树分裂

分裂准则使用特征重要性：
$$\text{gain} = \frac{\text{Var}(parent) - \sum \text{Var}(child)}{\# splits}$$

---

## 4. 训练过程讲解

### 4.1 算法流程

```
Input: 数据 D, 迭代次数 M
Output: 模型

1. 初始化: F_0(x) = argmin_y sum L(y_i, y)
2. For m = 1 to M:
3.     计算伪残差: g_i = -[dL/dF]_{F=F_{m-1}}
4.     拟合决策树: h_m(x) -> g_i
5.     更新: F_m = F_{m-1} + eta * h_m
6.     更新类别特征编码
7. Return F_M
```

### 4.2 核心参数

| 参数 | 说明 | 常用值 |
|------|------|--------|
| iterations | 迭代次数 | 1000 |
| depth | 树深度 | 6-10 |
| learning_rate | 学习率 | 0.01-0.1 |
| l2_leaf_reg | L2 正则化 | 3-10 |
| random_strength | 随机化 | 1-10 |

### 4.3 GPU 训练

```python
# GPU 加速
model = CatBoostClassifier(
    tasks_type='GPU',
    devices='0',
    iterations=1000
)
```

---

## 5. 应用场景

### 5.1 典型应用

- **金融风控**：信用评分、欺诈检测
- **推荐系统**：点击率预测
- **医疗诊断**：疾病预测
- **用户分类**：流失预测

### 5.2 代码示例

```python
from catboost import CatBoostClassifier, Pool

# 数据（���含类别特征）
train_data = Pool(data=X, label=y, cat_features=[0, 2, 5])

# 模型
model = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    loss_function='Logloss'
)

# 训练
model.fit(train_data)

# 预测
predictions = model.predict(X_test)
```

---

## 6. 优缺点分析

### 6.1 优点

1. **类别特征原生支持**：无需预处理
2. **防止过拟合**：Ordered Boosting
3. **GPU 加速**：原生支持
4. **鲁棒性强**：对小数据友好

### 6.2 缺点

1. **训练速度**：对称树可能慢
2. **内存**：树结构存储
3. **不如 XGBoost 灵活**

### 6.3 改进方向

- **改进 Ordered 编码**
- **GPU 优化**
- **特征组合**

---

## 7. 调库实现

### 7.1 CatBoost 实现

```python
import numpy as np
try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost 未安装，请 pip install catboost")

class CATBOOST:
    """CatBoost 梯度提升
    
    参数:
        iterations: 迭代次数
        depth: 树深度
        learning_rate: 学习率
    """
    
    def __init__(self, iterations=100, depth=6, learning_rate=0.1,
                 loss_function='Logloss'):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.model = None
        self.cat_features = None
        
    def fit(self, X, y, cat_features=None):
        """训练 CatBoost
        
        参数:
            X: 数据 (n_samples, n_features)
            y: 标签 (n_samples,)
            cat_features: 类别特征索引列表
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("请安装 catboost: pip install catboost")
        
        X = np.array(X)
        y = np.array(y)
        
        self.cat_features = cat_features
        
        # 创建 Pool
        if cat_features:
            train_pool = Pool(X, y, cat_features=cat_features)
        else:
            train_pool = Pool(X, y)
        
        # 模型
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            verbose=False
        )
        
        # 训练
        self.model.fit(train_pool)
        
        return self
    
    def predict(self, X):
        """预测类别"""
        if not CATBOOST_AVAILABLE:
            raise ImportError("请安装 catboost")
        
        X = np.array(X)
        
        if self.cat_features:
            test_pool = Pool(X, cat_features=self.cat_features)
        else:
            test_pool = Pool(X)
        
        return self.model.predict(test_pool)
    
    def predict_proba(self, X):
        """预测概率"""
        if not CATBOOST_AVAILABLE:
            raise ImportError("请安装 catboost")
        
        X = np.array(X)
        
        if self.cat_features:
            test_pool = Pool(X, cat_features=self.cat_features)
        else:
            test_pool = Pool(X)
        
        return self.model.predict_proba(test_pool)
    
    def score(self, X, y):
        """准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def demo():
    """CatBoost 演示"""
    print("=== CatBoost 演示 ===\n")
    
    if not CATBOOST_AVAILABLE:
        print("CatBoost 未安装，跳过演示")
        print("安装: pip install catboost")
        return None
    
    # 生成分类��据
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=10,
                          n_informative=5, n_redundant=2,
                          random_state=42)
    
    # 添加类别特征
    X[:, 0] = np.random.randint(0, 5, len(X))  # 第0列是类别
    X[:, 2] = np.random.randint(0, 3, len(X))  # 第2列是类别
    
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"类别特征: 0, 2")
    
    # 训练
    model = CATBOOST(iterations=100, depth=6)
    model.fit(X, y, cat_features=[0, 2])
    
    # 评估
    accuracy = model.score(X, y)
    print(f"训练准确率: {accuracy:.4f}")
    
    return model


if __name__ == "__main__":
    demo()
```

### 7.2 回归任务

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=500,
    depth=8,
    loss_function='RMSE'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 8. 手工代码实现

### 8.1 简化 CatBoost 核心

```python
import numpy as np

class SimulatedCatBoost:
    """简化版 CatBoost（核心思想）
    
    参数:
        iterations: 迭代次数
        depth: 树深度
    """
    
    def __init__(self, iterations=10, depth=3):
        self.iterations = iterations
        self.depth = depth
        self.trees = []
        
    def fit(self, X, y):
        """训练"""
        X = np.array(X)
        y = np.array(y)
        n, m = X.shape
        
        # 初始化
        self.base_pred = np.mean(y)
        residuals = y - self.base_pred
        
        # 迭代
        for it in range(self.iterations):
            # 简单决策树
            tree = SimpleDecisionTree(max_depth=self.depth)
            tree.fit(X, residuals)
            
            # 预测和更新
            pred = tree.predict(X)
            residuals -= self.learning_rate * pred
            
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """预测"""
        X = np.array(X)
        
        # 所有树的预测
        pred = np.full(len(X), self.base_pred)
        
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        
        return pred


class SimpleDecisionTree:
    """简单决策树"""
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
        
    def fit(self, X, y):
        """训练"""
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        """构建树"""
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)
        
        # 选择最佳分裂
        best_gain = -1
        best_split = None
        
        n, m = X.shape
        for feature in range(m):
            thresholds = np.percentile(X[:, feature], [25, 50, 75])
            
            for th in thresholds:
                left = y[X[:, feature] <= th]
                right = y[X[:, feature] > th]
                
                if len(left) == 0 or len(right) == 0:
                    continue
                
                gain = np.var(y) - (len(left) * np.var(left) + len(right) * np.var(right)) / n
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, th, np.mean(left), np.mean(right))
        
        if best_split is None:
            return np.mean(y)
        
        feature, th, left_val, right_val = best_split
        return {
            'feature': feature,
            'threshold': th,
            'left': self._build_tree(X[X[:, feature] <= th], 
                                    y[X[:, feature] <= th], depth+1),
            'right': self._build_tree(X[X[:, feature] > th],
                                   y[X[:, feature] > th], depth+1)
        }
    
    def predict(self, X):
        """预测"""
        pred = np.zeros(len(X))
        
        for i, x in enumerate(X):
            node = self.tree
            while isinstance(node, dict):
                if x[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            pred[i] = node
        
        return pred


def demo_manual():
    """手工实现演示"""
    print("=== CatBoost 手工实现演示 ===\n")
    
    from sklearn.datasets import make_regression
    
    # 数据
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1,
                         random_state=42)
    
    # 训练
    model = SimulatedCatBoost(iterations=10, depth=3)
    model.fit(X, y)
    
    # 预测
    pred = model.predict(X)
    
    # 误差
    mse = np.mean((y - pred) ** 2)
    print(f"训练 MSE: {mse:.4f}")


if __name__ == "__main__":
    demo_manual()
```

---

## 9. 可视化与结果理解

### 9.1 特征重要性

```python
def plot_feature_importance():
    """特征重要性可视化"""
    import matplotlib.pyplot as plt
    
    features = [f'Feature {i}' for i in range(10)]
    importance = np.random.rand(10)
    importance = importance / importance.sum()
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance)
    plt.xlabel('重要性')
    plt.title('CatBoost 特征重要性')
    plt.tight_layout()
    plt.savefig('catboost_importance.png', dpi=150)
    plt.show()
```

### 9.2 学习曲线

```python
def plot_learning_curve():
    """学习曲线"""
    import matplotlib.pyplot as plt
    
    iterations = range(1, 501)
    train_score = 1 - 0.3 * np.exp(-0.01 * np.array(iterations))
    val_score = 1 - 0.35 * np.exp(-0.01 * np.array(iterations))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_score, label='训练')
    plt.plot(iterations, val_score, label='验证')
    plt.xlabel('迭代')
    plt.ylabel('准确率')
    plt.title('CatBoost 学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('catboost_learning.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_catboost(model, X, y):
    """评估 CatBoost"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    
    return {'accuracy': accuracy, 'auc': auc}
```

### 10.2 对比实验

| 数据集 | CatBoost | XGBoost | LightGBM |
|--------|---------|---------|---------|
| 类别特征多 | **最优** | 需编码 | 需编码 |
| 数据量小 | **最优** | 一般 | 一般 |
| GPU 加速 | 原生 | CUDA | CUDA |

---

## 11. 常见问题与易错点

### 11.1 类别特征识别

**问题**：CatBoost 无法识别类别特征

**解决**：
- 指定 cat_features 参数
- 使用 Pool 对象

```python
pool = Pool(X, y, cat_features=[0, 2])
model.fit(pool)
```

### 11.2 过拟合

**问题**：训练过拟合

**解决**：
- early_stoking_rounds
- 增加正则化
- 减少深度

### 11.3 GPU 内存

**问题**：GPU 内存不足

**解决**：
- 减少 batch_size
- 使用 CPU

---

## 12. 学习总结

**核心要点**：

1. **目标编码**：自动处理类别特征
2. **Ordered Boosting**：防止过拟合
3. **对称树**：快速预测
4. **GPU 加速**：原生支持

**CatBoost 核心优势**：
- 类别特征处理最强
- 小数据表现好
- 鲁棒性强

**学习建议**：

1. 理解目标编码原理
2. 掌握 Ordered Boosting
3. 对比 XGBoost、LightGBM

---

## 13. 练习题与思考题

### 13.1 基础练习

1. 目标编码公式推导
2. Ordered Boosting 原理
3. 对称树结构

### 13.2 进阶练习

1. 实现目标编码
2. 对比三种 GBDT

### 13.3 思考题

1. CatBoost vs XGBoost 选择
2. 何时使用 CatBoost

---

### 13.4 详细答案与解析

#### 练习1：目标编码

**问题**：推导目标编码公式

**解答**：

$$encoded(x) = \frac{\sum_{x_i=x} y_i + \lambda \mu}{\#(x_i=x) + \lambda}$$

- 分子：类别样本标签和 + 全局均值 × 平滑
- 分母：样本数 + 平滑系数

防止：当某类别只有一个样本时，避免过拟合。

#### 练习2：Ordered Boosting

**问题**：为什么需要 Ordered

**解答**：

- 传统 GB DT 使用全量数据计算梯度，导致预测偏移
- Ordered Boosting 将数据分区，防止泄露

---

## 14. 学习路径建议

### 入门阶段

1. 学习决策树
2. 掌握梯度提升
3. 理解 CatBoost 原理

### 进阶阶段

1. 实现目标编码
2. 参数调优
3. GPU 训练

### 高级阶段

1. 特征工程
2. 模型融合
3. 分布式训练

**推荐路线**：

```
决策树 → GBDT → XGBoost → LightGBM → CatBoost → 集成学习
```

**CatBoost 是处理类别特征的首选框架，熟练掌握它对处理实际业务数据很重要。**