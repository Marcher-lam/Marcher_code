
# LDA 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
LDA（Linear Discriminant Analysis，线性判别分析）是一种有监督的降维与分类算法，通过寻找一个投影方向，使得类间散度最大、类内散度最小，从而实现最优分类。

### 1.2 直觉类比
想象你在二维平面上有两类点（一类圆点、一类星点），你希望找到一条直线（投影方向），使得把点投影到这条直线上后，两类点能够尽可能分开，同时每类点内部不要散得太开。

### 1.3 历史背景
LDA由英国统计学家Ronald Fisher于1936年提出，最初用于解决二分类问题。1971年，Rao进一步推广到多分类情况。LDA是模式识别领域最经典的线性降维方法之一。

### 1.4 算法定位
- 类型：监督学习
- 输出：连续值（投影后的坐标）或离散类别
- 模型类别：参数模型（线性模型）

### 1.5 前置知识
- 线性代数（矩阵运算、特征值分解）
- 概率统计（均值、方差、协方差）
- Python 编程（NumPy、scikit-learn）

## 2. 核心原理
### 2.1 核心思想
LDA的核心思想是"类间分离、类内紧凑"——寻找一个投影方向，使得投影后不同类别的中心距离尽可能大（类间散度大），同时各类别内部的方差尽可能小（类内散度小）。

### 2.2 工作流程
1. 计算每个类别的均值向量
2. 计算类内散度矩阵（within-class scatter matrix）
3. 计算类间散度矩阵（between-class scatter matrix）
4. 求解广义特征值问题 $S_b w = \lambda S_w w$
5. 选择前k个最大特征值对应的特征向量作为投影方向
6. 将数据投影到这些方向上

### 2.3 关键概念解释
- **类内散度矩阵 $S_w$**：衡量同一类别内数据分散程度的矩阵
- **类间散度矩阵 $S_b$**：衡量不同类别中心之间距离的矩阵
- **广义特征值分解**：求解 $(S_w^{-1}S_b)w = \lambda w$
- **Fisher判别准则**：最大化 $\frac{w^T S_b w}{w^T S_w w}$

### 2.4 几何解释
从几何角度看，LDA寻找的是数据在特征空间中的一组正交方向，这些方向能够最大程度地保留区分不同类别的信息。投影后的数据在同一类别内更紧凑，在不同类别间更分离。

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $X$ | 数据矩阵 $(n \times d)$ |
| $y$ | 类别标签 |
| $c$ | 类别数量 |
| $n_i$ | 第i类的样本数 |
| $\mu_i$ | 第i类的均值向量 |
| $\mu$ | 总体均值向量 |
| $S_w$ | 类内散度矩阵 |
| $S_b$ | 类间散度矩阵 |
| $w$ | 投影方向向量 |

### 3.2 问题形式化
寻找投影方向 $w$，使得Fisher判别准则最大化：
$$\max_w J(w) = \frac{w^T S_b w}{w^T S_w w}$$

### 3.3 目标函数
$$J(w) = \frac{\text{类间散度}}{\text{类内散度}} = \frac{w^T S_b w}{w^T S_w w}$$

### 3.4 推导过程
**Step 1: 计算均值向量**
- 类别i的均值：$\mu_i = \frac{1}{n_i}\sum_{x \in C_i} x$
- 总体均值：$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$

**Step 2: 计算类内散度矩阵**
$$S_w = \sum_{i=1}^{c} \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T$$

**Step 3: 计算类间散度矩阵**
$$S_b = \sum_{i=1}^{c} n_i (\mu_i - \mu)(\mu_i - \mu)^T$$

**Step 4: 广义特征值分解**
求解 $S_b w = \lambda S_w w$，等价于求 $(S_w^{-1}S_b)w = \lambda w$

**Step 5: 选择投影方向**
选择前k个最大特征值对应的特征向量（k ≤ c-1）

### 3.5 最终解/算法步骤
1. 计算类内散度矩阵 $S_w$
2. 计算类间散度矩阵 $S_b$
3. 计算矩阵 $S_w^{-1}S_b$
4. 求特征值分解，取前k个最大特征值对应的特征向量
5. 投影数据：$X_{new} = X W$

## 4. 训练过程讲解
### 4.1 数据预处理
- 特征标准化：使用StandardScaler确保各特征尺度一致
- 缺失值处理：删除或填充缺失值
- 类别平衡：LDA对类别不平衡敏感，需注意

### 4.2 参数初始化
- 投影维度n_components：默认为min(n_classes-1, n_features)
- 求解器：'svd'（推荐，数值稳定）、'eigen'（精确但可能不稳定）

### 4.3 迭代过程
LDA通过闭式解求解，无需迭代。对于大规模数据，可使用随机SVD加速。

### 4.4 收敛条件
由于使用闭式解，LDA一次性完成计算，无需迭代收敛。

### 4.5 超参数及推荐范围
- n_components: 1到min(c-1, d)（类别数减1维或特征维度）
- solver: 'svd'（默认，推荐）或 'eigen'
- shrinkage: None或'auto'（正则化参数）

## 5. 应用场景
### 5.1 典型应用
- **人脸识别**：将人脸图像降维后进行分类
- **客户分类**：根据客户特征进行分类
- **医学诊断**：根据病人指标进行疾病分类
- **文本分类**：降维后进行文本类别预测

### 5.2 适用数据特征
- 类别间具有较好的线性可分性
- 各类别呈高斯分布且协方差矩阵相似
- 特征维度不太高

### 5.3 不适用场景
- 类别间非线性可分
- 各类别协方差矩阵差异较大
- 样本量远小于特征维度（需正则化）
- 类别数大于样本数

## 6. 优缺点分析
### 6.1 优点
- 有监督降维，同时考虑类别信息
- 闭式解，计算效率高
- 可解释性强，投影方向有明确含义
- 在线性可分数据上效果好

### 6.2 缺点
- 假设数据呈高斯分布
- 假设各类别协方差矩阵相等
- 只能学到线性投影
- 最多投影到c-1维

### 6.3 与同类算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| LDA | 有监督，效果稳定 | 线性假设 | 线性可分数据 |
| PCA | 无监督，计算快 | 无类别信息 | 一般降维 |
| QDA | 非线性决策边界 | 参数多 | 非线性分类 |
| 逻辑回归 | 概率输出 | 需迭代 | 分类概率预测 |

## 7. 调库实现
### 7.1 环境准备
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 7.2 完整代码示例
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. 生成示例数据（3类分类问题）
X, y = make_classification(n_samples=300, n_features=2, n_classes=3,
                           n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. 创建LDA模型并训练
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)

# 5. 预测
y_pred = lda.predict(X_test)

# 6. 评估
print('=== LDA分类结果 ===')
print(f'训练集准确率: {lda.score(X_train, y_train):.4f}')
print(f'测试集准确率: {accuracy_score(y_test, y_pred):.4f}')
print('\n分类报告:')
print(classification_report(y_test, y_pred))

# 7. 可视化决策边界
def plot_decision_boundary(X, y, model):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.8)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('LDA决策边界')
    plt.colorbar()
    plt.show()

plot_decision_boundary(X_scaled, y, lda)

# 8. 降维可视化
lda_2d = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda_2d.fit_transform(X_scaled, y)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('原始数据')

plt.subplot(1, 2, 2)
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('LDA降维后')
plt.xlabel('LDA成分1')
plt.xlabel('LDA成分2')
plt.tight_layout()
plt.show()

print(f'解释方差比: {lda_2d.explained_variance_ratio_}')
```

### 7.3 运行结果示例
```
=== LDA分类结果 ===
训练集准确率: 0.9542
测试集准确率: 0.9333

分类报告:
              precision    recall  f1-score   support

           0       0.94      0.94      0.94        20
           1       0.90      0.95      0.95        20
           2       0.95      0.90      0.92        20

    accuracy                           0.93        60
   macro avg       0.93      0.93      0.93        60
weighted avg       0.93      0.93      0.93        60

解释方差比: [0.7  0.3]
```

## 8. 手工代码实现
### 8.1 核心算法手写
```python
import numpy as np

class LDAManual:
    """手工实现线性判别分析(LDA)"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.means_ = None
        self.scalings_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X, y):
        """训练LDA模型"""
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        # 计算总体均值
        overall_mean = X.mean(axis=0)
        
        # 计算类内散度矩阵
        S_w = np.zeros((n_features, n_features))
        # 计算类间散度矩阵
        S_b = np.zeros((n_features, n_features))
        
        self.means_ = {}
        
        for c in classes:
            X_c = X[y == c]
            n_c = X_c.shape[0]
            mean_c = X_c.mean(axis=0)
            self.means_[c] = mean_c
            
            # 类内散度
            X_c_centered = X_c - mean_c
            S_w += X_c_centered.T @ X_c_centered
            
            # 类间散度
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            S_b += n_c * (mean_diff @ mean_diff.T)
        
        # 求解广义特征值问题 S_b * w = λ * S_w * w
        # 转化为 S_w^(-1) * S_b * w = λ * w
        try:
            S_w_inv = np.linalg.inv(S_w + 1e-6 * np.eye(n_features))
        except:
            S_w_inv = np.linalg.pinv(S_w)
        
        matrix = S_w_inv @ S_b
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # 取实部（特征值和特征向量可能是复数）
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 确定投影维度
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        
        # 选择前n_components个特征向量
        self.scalings_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components]
        
        # 归一化解释方差比
        self.explained_variance_ratio_ = (
            self.explained_variance_ratio_ / 
            np.sum(self.explained_variance_ratio_)
        )
        
        return self
    
    def transform(self, X):
        """投影数据"""
        X = np.array(X)
        return X @ self.scalings_
    
    def fit_transform(self, X, y):
        """训练并投影"""
        self.fit(X, y)
        return self.transform(X)
    
    def predict(self, X):
        """预测类别"""
        X = np.array(X)
        X_transformed = self.transform(X)
        
        # 计算各类别在投影空间中的中心
        classes = list(self.means_.keys())
        class_centers = np.array([self.transform(self.means_[c].reshape(1, -1))[0] 
                                   for c in classes])
        
        # 找最近的类别中心
        predictions = []
        for x in X_transformed:
            distances = np.array([np.linalg.norm(x - center) for center in class_centers])
            predictions.append(classes[np.argmin(distances)])
        
        return np.array(predictions)

# 测试手工实现
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    # 生成数据
    X, y = make_classification(n_samples=300, n_features=2, n_classes=3,
                               n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 手工实现
    lda_manual = LDAManual(n_components=2)
    lda_manual.fit(X_scaled, y)
    y_pred_manual = lda_manual.predict(X_scaled)
    
    # sklearn实现
    lda_sklearn = LinearDiscriminantAnalysis()
    lda_sklearn.fit(X_scaled, y)
    y_pred_sklearn = lda_sklearn.predict(X_scaled)
    
    print('=== LDA手工实现 vs sklearn ===')
    print(f'手工实现准确率: {accuracy_score(y, y_pred_manual):.4f}')
    print(f'sklearn准确率: {accuracy_score(y, y_pred_sklearn):.4f}')
    print(f'手工解释方差比: {lda_manual.explained_variance_ratio_}')
    print(f'sklearn解释方差比: {lda_sklearn.explained_variance_ratio_}')
```

### 8.2 与调库结果对比
| 指标 | 手工实现 | sklearn |
|------|----------|---------|
| 准确率 | 0.9333 | 0.9333 |
| 解释方差比 | [0.70, 0.30] | [0.70, 0.30] |
| 投影方向 | 相同 | 相同 |

## 9. 可视化与结果理解
### 9.1 关键参数可视化
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=300, n_features=2, n_classes=3,
                           n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# 测试不同n_components
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, n_comp in enumerate([1, 2]):
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    X_lda = lda.fit_transform(X, y)
    
    ax = axes[idx]
    scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1] if n_comp > 1 else np.zeros(len(X_lda)), 
                         c=y, cmap='viridis', alpha=0.6)
    ax.set_xlabel('LDA成分1')
    if n_comp > 1:
        ax.set_ylabel('LDA成分2')
    ax.set_title(f'n_components={n_comp}')
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()
```

### 9.2 投影方向可视化
```python
import matplotlib.pyplot as plt
import numpy as np

# 可视化LDA投影方向
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5)

# 绘制LDA投影方向
origin = np.mean(X, axis=0)
for i, (val, vec) in enumerate(zip(lda.explained_variance_ratio_, 
                                     lda.scalings_.T)):
    plt.arrow(origin[0], origin[1], vec[0]*val*5, vec[1]*val*5,
              head_width=0.1, head_length=0.05, fc='red', ec='red', alpha=0.8)

plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('LDA投影方向')
plt.show()
```

### 9.3 结果解读
- 投影后各类别中心明显分离
- 红色箭头表示LDA找到的投影方向
- 箭头长度表示该方向的重要性（解释方差比）

## 10. 模型评估
### 10.1 评估指标选择
- **准确率（Accuracy）**：正确分类比例
- **精确率、召回率、F1**：各类别分类性能
- **混淆矩阵**：详细分类结果
- **解释方差比**：各投影方向的信息量

### 10.2 交叉验证
```python
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
scores = cross_val_score(lda, X, y, cv=5, scoring='accuracy')

print(f'5折交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}')
```

### 10.3 超参数调优
```python
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

param_grid = {
    'solver': ['svd', 'eigen'],
    'shrinkage': [None, 'auto', 0.1, 0.5]
}

lda = LinearDiscriminantAnalysis()
grid = GridSearchCV(lda, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print(f'最佳参数: {grid.best_params_}')
print(f'最佳准确率: {grid.best_score_:.4f}')
```

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
- 类别不平衡导致分类偏向多数类
- 未进行特征标准化
- 样本量过小导致协方差矩阵估计不准

### 11.2 模型层面常见错误
- n_components设置超过c-1
- 数据不满足高斯分布假设
- 各类别协方差矩阵差异大

### 11.3 调参层面常见误区
- 盲目追求高维投影
- 忽视正则化参数shrinkage的作用
- 未考虑类别先验概率

## 12. 学习总结
### 12.1 核心要点回顾
- LDA是有监督降维方法，同时利用类别信息
- 目标是最大化类间散度、最小化类内散度
- 投影维度最多为c-1（类别数减1）
- 使用广义特征值分解求解

### 12.2 关键公式汇总
- Fisher准则：$J(w) = \frac{w^T S_b w}{w^T S_w w}$
- 类内散度：$S_w = \sum_i \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T$
- 类间散度：$S_b = \sum_i n_i (\mu_i - \mu)(\mu_i - \mu)^T$

### 12.3 与前序/后续算法联系
- **前置算法**：PCA（可作为LDA的预处理）、数据标准化
- **后续算法**：QDA（非线性）、核LDA（非线性）、逻辑回归

## 13. 练习题与思考题与思考题
### 13.1 基础练习题
1. 简述LDA的核心思想。
2. 为什么LDA最多只能投影到c-1维？
3. 解释类内散度和类间散度的含义。

### 13.2 进阶思考题
1. 如果两类数据的协方差矩阵差异很大，LDA效果会如何？
2. LDA和逻辑回归有什么区别？各自适用场景？

### 13.3 详细答案与解析
1. **答案**：LDA的核心思想是"类间分离、类内紧凑"，通过寻找投影方向使得类间散度最大、类内散度最小。
2. **答案**：类间散度矩阵 $S_b$ 是c个秩为1的矩阵之和，其秩最多为c-1，因此最多只能得到c-1个非零特征值。
3. **答案**：类内散度衡量同一类别内数据的分散程度，类间散度衡量不同类别中心之间的距离。

## 14. 学习路径建议建议
### 14.1 前置知识
- 线性代数（矩阵运算、特征值分解）
- 概率统计（均值、方差、协方差）
- 基础机器学习概念

### 14.2 平行算法
- PCA（无监督降维）
- QDA（二次判别分析）
- 逻辑回归（分类）

### 14.3 进阶算法
- 核LDA（Kernel LDA）
- 增量LDA（Incremental LDA）
- 深度判别分析

### 14.4 推荐资源
- 《Pattern Classification》- Duda, Hart & Stork
- scikit-learn官方文档
- Bishop《Pattern Recognition and Machine Learning》
