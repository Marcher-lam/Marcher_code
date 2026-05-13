# 支持向量机（Support Vector Machine, SVM）学习文档

> 用一句话说明这个算法的核心价值，不超过30字。

支持向量机是一种基于间隔最大化的分类算法，通过寻找最优超平面将不同类别的数据分开，在小样本和medium高维数据上表现优异，是经典的机器学习算法之一。

---

## 1. 算法基础认知

### 1.1 什么是支持向量机

支持向量机（Support Vector Machine，简称SVM）是一种有监督学习的分类算法，其核心思想是找到一个能够最大化两类样本之间间隔的超平面，将不同类别的数据分开。SVM不仅能处理线性可分的分类问题，还能通过核函数处理非线性可分的问题。

SVM的名字来自于"支持向量"（Support Vectors）的概念，这些支持向量是指位于决策边界附近的样本点，它们决定了超平面的位置。SVM的神奇之处在于：即使数据维度很高，只需要少数支持向量就能进行分类决策。

### 1.2 直觉类比

将SVM想象成一个二维平面上的分类问题：我们需要用一条直线将两类点分开。如果有很多条直线都能分开这两类点，SVM会选择"最中间"的那条直线——即让两类点到直线的距离都尽可能大的那条。

想象你正在摆放一个书架，你需要把书分成两类（左边放技术书，右边放文学书）。如果你只是随意摆放，可能书的边界会靠近中间，容易倒。但如果你把两类书分别往两边放得尽可能远，中间留出足够的空间，那么整个排列就更稳定——这就是SVM的间隔最大化思想。

### 1.3 历史背景

SVM算法由Vapnik等人在1992年提出，其理论基础来自于统计学习理论中的VC维和结构风险最小化原理。1995年，Vapnik发表了关于软间隔SVM的经典论文，使得SVM能够处理线性不可分的问题。随后，1998年提出的核技巧（Kernel Trick）使得SVM能够处理非线性问题，成为当时最流行的分类算法之一。

SVM在20世纪90年代末到21世纪初一直是分类任务的首选算法，尤其在手写数字识别、文本分类等任务上取得了突破性的成果。虽然近年来被深度学习超越，但在小样本数据和结构化数据的分类任务上，SVM仍然是首选。

### 1.4 算法定位

SVM是**监督学习**的**判别式**分类算法，属于**参数化模型**，广泛应用于：
- 文本分类（垃圾邮件检测、情感分析）
- 图像分类（手写数字识别、人脸检测）
- 生物信息学（基因分类、蛋白质结构预测）
- 网络安全（入侵检测）

### 1.5 前置知识

- 线性代数（超平面概念、拉格朗日乘数法）
- 基础微积分（梯度、导数）
- 概率论基础（分布函数）
- 凸优化基础

---

## 2. 核心原理

### 2.1 核心思想

SVM的核心目标是找到一个能够最大化两类样本之间间隔的超平面。对于线性可分的情况，优化目标是：

$$\min_{w,b} \frac{1}{2}||w||^2$$

约束条件为：

$$y_i(w^Tx_i+b) \geq 1, \forall i$$

其中：$w$是超平面的法向量，$b$是偏置，$y_i$是类别标签（+1或-1）。

### 2.2 工作流程

1. **选择超平面**：找到一个能够正确分类所有样本的超平面
2. **最大化间隔**：确保超平面到两类最近样本的距离最大化
3. **支持向量识别**：识别出位于决策边界附近的样本
4. **分类决策**：对于新样本，计算其到超平面的距离，决定类别

### 2.3 关键概念

**硬间隔（Hard Margin）**：要求所有样本都被正确分类且满足间隔要求，适用于线性可分的数据。

**软间隔（Soft Margin）**：允许一些样本违反间隔约束，通过引入松弛变量$\xi_i$来平衡分类错误和间隔最大化，适用于线性不可分的数据。

**支持向量（Support Vectors）**：距离超平面最近的样本点，它们决定了超平面的位置。

**核函数（Kernel Function）**：将数据映射到高维空间，使其线性可分。常用核函数包括：线性核、多项式核、高斯核（RBF）、sigmoid核。

### 2.4 几何解释

```
           ●  ●
          ●   ●  ●
         +-------------
          ●   ●  ●
           ●  ●

   +--------------------+  <- 正类支持向量
   |                    |
   |      超平面        |  <- 决策边界 w·x + b = 0
   |                    |
   +--------------------+  <- 负类支持向量
```

在二维空间中，SVM找到的是一条直线（超平面）；在三维空间中，SVM找到的是一个平面；在更高维空间中，SVM找到的是一个超平面。$w$是超平面的法向量，$b$决定了超平面的位置。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $x_i$ | 第$i$个样本的特征向量 |
| $y_i$ | 第$i$个样本的标签（+1或-1） |
| $w$ | 超平面的法向量 |
| $b$ | 超平面的偏置 |
| $\xi_i$ | 第$i$个样本的松弛变量 |
| $C$ | 正则化参数（惩罚系数） |
| $\alpha_i$ | 拉格朗日乘子 |
| $K(x_i,x_j)$ | 核函数 |

### 3.2 线性SVM的优化目标

对于线性可分的SVM，优化问题是：

$$\min_{w,b} \frac{1}{2}||w||^2$$

$$\text{s.t.} \quad y_i(w^Tx_i+b) \geq 1, \quad \forall i$$

这个优化问题的目标是最大化几何间隔$\frac{2}{||w||}$，等价于最小化$||w||^2$。

### 3.3 软间隔SVM

对于线性不可分的情况，引入松弛变量$\xi_i$：

$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

$$\text{s.t.} \quad y_i(w^Tx_i+b) \geq 1-\xi_i, \quad \xi_i \geq 0, \quad \forall i$$

其中$C$是正则化参数，控制对误分类样本的惩罚程度。$C$越大，对误分类的惩罚越重，模型越"硬"；$C$越小，允许更多的误分类，模型越"软"。

### 3.4 对偶问题

使用拉格朗日乘数法，将原问题转换为对偶问题：

$$\max_{\alpha} \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^Tx_j$$

$$\text{s.t.} \quad \sum_{i=1}^{n}\alpha_iy_i=0, \quad 0 \leq \alpha_i \leq C, \quad \forall i$$

这个对偶问题可以通过SMO（Sequential Minimal Optimization）等算法高效求解。

### 3.5 核函数

核函数是SVM处理非线性问题的关键。核函数$K(x_i,x_j)$满足Mercer定理，表示将数据映射到高维空间后的内积：

- **线性核**：$K(x_i,x_j)=x_i^Tx_j$
- **多项式核**：$K(x_i,x_j)=(x_i^Tx_j+1)^d$
- **高斯核（RBF）**：$K(x_i,x_j)=\exp(-\gamma||x_i-x_j||^2)$
- **sigmoid核**：$K(x_i,x_j)=\tanh(\beta x_i^Tx_j+\theta)$

### 3.6 决策函数

最终的分类决策函数为：

$$f(x) = \text{sign}\left(\sum_{i=1}^{n}\alpha_i y_i K(x_i,x) + b\right)$$

由于$\alpha_i$只有在支持向量上才非零，因此决策时只需要计算支持向量与输入样本的核函数。

---

## 4. 训练过程讲解

### 4.1 数据预处理

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train, X_test):
    """
    SVM的数据预处理
    
    SVM对特征尺度敏感，需要进行标准化
    """
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
```

**为什么需要标准化？**

SVM使用核函数计算样本之间的相似度，如果不同特征的尺度差异很大，会导致尺度大的特征主导核函数的计算，从而忽略尺度小的特征。标准化可以确保所有特征对模型有同等的贡献。

### 4.2 超参数选择

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(X_train, y_train):
    """
    使用网格搜索选择最优超参数
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_, grid_search.best_score_
```

### 4.3 参数解释

| 超参数 | 作用 | 推荐范围 | 说明 |
|--------|------|----------|------|
| C | 正则化强度 | 0.01-1000 | 控制对误分类的惩罚 |
| kernel | 核函数类型 | linear/rbf/poly | 决定数据映射方式 |
| gamma | RBF核参数 | scale/auto/数值 | 控制高斯核的宽度 |
| degree | 多项式核阶数 | 1-5 | 多项式核的阶数 |

### 4.4 训练流程

```python
def train_svm(X_train, y_train, C=1.0, kernel='rbf', gamma='scale'):
    """
    训练SVM模型
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        C: 正则化参数
        kernel: 核函数类型
        gamma: 核函数参数
    
    返回:
        训练好的模型
    """
    model = SVC(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    
    return model
```

---

## 5. 应用场景

### 5.1 文本分类

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def text_classification_demo():
    """
    文本分类示例：垃圾邮件检测
    """
    # 文本数据
    emails = [
        "恭喜您获得100万大奖，点击链接领取",
        "您的快递已送达，请查收",
        "限时优惠，买一送一",
        "会议定在下午3点",
        "您好，我想咨询产品信息"
    ]
    labels = [1, 0, 1, 0, 0]  # 1: 垃圾邮件， 0: 正常邮件
    
    # 文本特征提取
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(emails)
    
    # 训练SVM
    model = LinearSVC(C=1.0)
    model.fit(X, labels)
    
    # 预测
    new_email = "免费送iphone了，快来抢"
    new_features = vectorizer.transform([new_email])
    prediction = model.predict(new_features)
    
    print(f"邮件 '{new_email}' 分类结果: {'垃圾邮件' if prediction[0] == 1 else '正常邮件'}")
```

### 5.2 图像分类

```python
from sklearn.datasets import load_digits
from sklearn.svm import SVC

def image_classification_demo():
    """
    手写数字识别示例
    """
    # 加载手写数字数据集
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练SVM
    model = SVC(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    # 评估
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"手写数字识别准确率: {accuracy:.4f}")
```

### 5.3 异常检测

```python
from sklearn.svm import OneClassSVM

def anomaly_detection_demo():
    """
    异常检测示例：网络入侵检测
    """
    # 正常网络流量数据（模拟）
    np.random.seed(42)
    normal_traffic = np.random.randn(1000, 5) * 0.5 + 0.5
    
    # 训练One-Class SVM
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    model.fit(normal_traffic)
    
    # 测试数据
    test_traffic = np.random.randn(10, 5) * 0.5 + 0.5
    test_traffic[0] = [5, 5, 5, 5, 5]  # 异常流量
    
    # 预测
    predictions = model.predict(test_traffic)
    
    for i, pred in enumerate(predictions):
        if pred == 1:
            print(f"样本 {i}: 正常")
        else:
            print(f"样本 {i}: 异常")
```

### 5.4 生物信息学

```python
def bioinformatics_demo():
    """
    基因分类示例
    """
    # 模拟基因表达数据
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    
    # 正类（疾病相关基因）
    X_positive = np.random.randn(n_samples // 2, n_features) + 2
    y_positive = np.ones(n_samples // 2)
    
    # 负类（正常基因）
    X_negative = np.random.randn(n_samples // 2, n_features)
    y_negative = -np.ones(n_samples // 2)
    
    # 合并数据
    X = np.vstack([X_positive, X_negative])
    y = np.hstack([y_positive, y_negative])
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练SVM
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_scaled, y)
    
    print(f"疾病相关基因识别模型训练完成")
    print(f"支持向量数量: {sum(model.n_support_)}")
```

### 5.5 推荐系统

```python
def recommendation_demo():
    """
    电影推荐示例：基于用户喜好的分类
    """
    # 模拟用户评分数据
    # 特征: [���作片占比, 爱情片占比, 喜剧片占比, 科幻片占比, 纪录片占比]
    np.random.seed(42)
    
    # 喜欢动作片的用户
    X_action = np.random.randn(50, 5)
    X_action[:, 0] += 2
    y_action = np.ones(50)  # 喜欢
    
    # 不喜欢动作片的用户
    X_no_action = np.random.randn(50, 5)
    X_no_action[:, 0] -= 2
    y_no_action = -np.ones(50)  # 不喜欢
    
    # 合并
    X = np.vstack([X_action, X_no_action])
    y = np.hstack([y_action, y_no_action])
    
    # 训练SVM
    model = SVC(kernel='rbf', C=1.0)
    model.fit(X, y)
    
    # 预测新用户
    new_user = np.array([[3, 1, 1, 2, 0]])  # 喜欢动作和科幻
    prediction = model.predict(new_user)
    
    print(f"新用户分类: {'推荐动作片' if prediction[0] == 1 else '不推荐动作片'}")
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 效果好 | 在小样本上表现优异 |
| 泛化能力强 | 最大化间隔减少过拟合 |
| 内存效率高 | 只存储支持向量 |
| 可解释性强 | 可以分析支持向量 |
| 适用于高维数据 | 通过核函数处理高维数据 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 对大规模数据效率低 | $O(n^2)$或$O(n^3)$ | 使用线性SVM或采样 |
| 对参数敏感 | C和gamma需要调参 | 网格搜索 |
| 不适用于多分类 | 需要改造 | One-vs-One或One-vs-Rest |
| 核函数选择困难 | 需要尝试 | 交叉验证 |

### 6.3 与其他算法对比

| 特性 | SVM | 逻辑回归 | 决策树 |
|------|-----|---------|--------|
| 适用场景 | 中小规模数据 | 大规模数据 | 中等规模 |
| 训练复杂度 | $O(n^2)$ | $O(n)$ | $O(n\log n)$ |
| 可解释性 | 中 | 高 | 高 |
| 对参数敏感度 | 高 | 中 | 低 |
| 核函数 | 支持 | 不支持 | 不适用 |

---

## 7. 调库实现

### 7.1 sklearn实现

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# 生成分类数据
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练SVM
model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# 评估
print(f"SVM准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 打印支持向量数量
print(f"支持向量数量: {sum(model.n_support_)}")
```

### 7.2 线性SVM实现

```python
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs

# 生成聚类数据
X, y = make_blobs(n_samples=500, centers=2, random_state=42)

# 训练线性SVM（更快）
model = LinearSVC(C=1.0, max_iter=10000)
model.fit(X, y)

# 绘制决策边界
def plot_decision_boundary(X, y, model):
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.3)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(X, y, model)
```

### 7.3 多分类SVM

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# 加载iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 训练多分类SVM（OvR策略）
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X, y)

# 预测
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"多分类SVM准确率: {accuracy:.4f}")
print(f"类别: {iris.target_names}")
```

### 7.4 带概率输出的SVM

```python
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=500, random_state=42)

# 使用CalibratedClassifierCV获取概率
base_model = SVC(kernel='rbf', C=1.0)
model = CalibratedClassifierCV(base_model, cv=5)
model.fit(X, y)

# 预测概率
probs = model.predict_proba(X)
print(f"预测概率示例: {probs[0]}")
```

---

## 8. 手工代码实现

### 8.1 简化版SMO算法实现

```python
import numpy as np

class SimpleSVM:
    """
    使用简化版SMO算法实现的SVM
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma=1.0, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.X_train = None
        self.y_train = None
    
    def _kernel(self, x1, x2):
        """核函数"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
        else:
            return np.dot(x1, x2)
    
    def _select_j(self, i, Ei):
        """选择第二个优化变量"""
        # 简化的选择策略：随机选择
        j = np.random.randint(len(self.X_train))
        while j == i:
            j = np.random.randint(len(self.X_train))
        return j
    
    def fit(self, X, y):
        """训练SVM"""
        self.X_train = X
        self.y_train = y
        n_samples, n_features = X.shape
        
        # 初始化
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # 简化版SMO
        for epoch in range(self.max_iter):
            num_changed = 0
            for i in range(n_samples):
                # 计算Ei
                Ei = self._compute_E(i)
                
                # 检查KKT条件
                if (self.y_train[i] * Ei < -0.001 and self.alpha[i] < self.C) or \
                   (self.y_train[i] * Ei > 0.001 and self.alpha[i] > 0):
                    
                    # 选择j并计算Ej
                    j = self._select_j(i, Ei)
                    Ej = self._compute_E(j)
                    
                    # 保存旧的alpha
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # 计算边界
                    if self.y_train[i] != self.y_train[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # 计算eta
                    eta = 2 * self._kernel(X[i], X[j]) - \
                          self._kernel(X[i], X[i]) - \
                          self._kernel(X[j], X[j])
                    
                    if eta >= 0:
                        continue
                    
                    # 更新alpha_j
                    self.alpha[j] -= (self.y_train[j] * (Ei - Ej)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    # 检查变化
                    if abs(self.alpha[j] - alpha_j_old) < 0.00001:
                        continue
                    
                    # 更新alpha_i
                    self.alpha[i] += self.y_train[i] * self.y_train[j] * \
                                   (alpha_j_old - self.alpha[j])
                    
                    # 更新b
                    b1 = self.b - Ei - self.y_train[i] * (self.alpha[i] - alpha_i_old) * \
                              self._kernel(X[i], X[i]) - \
                         self.y_train[j] * (self.alpha[j] - alpha_j_old) * \
                              self._kernel(X[i], X[j])
                    
                    b2 = self.b - Ej - self.y_train[i] * (self.alpha[i] - alpha_i_old) * \
                              self._kernel(X[i], X[j]) - \
                         self.y_train[j] * (self.alpha[j] - alpha_j_old) * \
                              self._kernel(X[j], X[j])
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed += 1
            
            if num_changed == 0:
                print(f"在第{epoch}次迭代收敛")
                break
        
        return self
    
    def _compute_E(self, i):
        """计算Ei"""
        fx = 0
        for j in range(len(self.X_train)):
            fx += self.alpha[j] * self.y_train[j] * \
                  self._kernel(self.X_train[j], self.X_train[i])
        fx += self.b
        return fx - self.y_train[i]
    
    def predict(self, X):
        """预测"""
        result = []
        for x in X:
            fx = 0
            for i in range(len(self.X_train)):
                if self.alpha[i] > 0:
                    fx += self.alpha[i] * self.y_train[i] * \
                          self._kernel(self.X_train[i], x)
            fx += self.b
            result.append(np.sign(fx) if fx != 0 else 1)
        
        return np.array(result)

# 测试
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.array([1 if x[0] + x[1] > 0 else -1 for x in X])

svm = SimpleSVM(C=1.0, kernel='linear', max_iter=100)
svm.fit(X, y)

y_pred = svm.predict(X)
accuracy = np.mean(y_pred == y)
print(f"准确率: {accuracy:.4f}")
```

### 8.2 核函数实现

```python
import numpy as np

class KernelSVM:
    """
    带核函数的SVM实现
    """
    
    def __init__(self, C=1.0, gamma=1.0, max_iter=1000, tol=1e-3):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        self.support_vectors = None
        self.support_labels = None
        self.support_alphas = None
    
    def linear_kernel(self, x1, x2):
        """线性核"""
        return np.dot(x1, x2)
    
    def rbf_kernel(self, x1, x2):
        """RBF核"""
        return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
    
    def polynomial_kernel(self, x1, x2, d=3):
        """多项式核"""
        return (np.dot(x1, x2) + 1) ** d
    
    def sigmoid_kernel(self, x1, x2, beta=0.1, theta=0):
        """Sigmoid核"""
        return np.tanh(beta * np.dot(x1, x2) + theta)
    
    def kernel(self, x1, x2):
        """选择核函数"""
        return self.rbf_kernel(x1, x2)
    
    def fit(self, X, y):
        """训练（简化版）"""
        self.X_train = X
        self.y_train = y
        n = len(X)
        
        # 初始化
        self.alpha = np.zeros(n)
        self.b = 0
        
        # 预计算核矩阵
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])
        
        # SMO算法
        for epoch in range(self.max_iter):
            alpha_changed = 0
            for i in range(n):
                # 计算Ei
                Ei = np.sum(self.alpha * y * K[:, i]) + self.b - y[i]
                
                # 检查是否违反KKT条件
                if (y[i] * Ei < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * Ei > self.tol and self.alpha[i] > 0):
                    
                    # 随机选择j
                    j = np.random.randint(n)
                    while j == i:
                        j = np.random.randint(n)
                    
                    Ej = np.sum(self.alpha * y * K[:, j]) + self.b - y[j]
                    
                    # 保存旧值
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # 计算边界
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # 计算eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    
                    if eta >= 0:
                        continue
                    
                    # 更新alpha_j
                    self.alpha[j] = alpha_j_old - y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # 更新alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # 更新b
                    self.b = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                               y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                
                alpha_changed += 1
        
        # 提取支持向量
        sv_idx = self.alpha > 1e-5
        self.support_vectors = X[sv_idx]
        self.support_labels = y[sv_idx]
        self.support_alphas = self.alpha[sv_idx]
        
        return self
    
    def predict(self, X):
        """预测"""
        result = []
        for x in X:
            fx = 0
            for sv_x, sv_y, sv_alpha in zip(
                self.support_vectors, self.support_labels, self.support_alphas
            ):
                fx += sv_alpha * sv_y * self.kernel(sv_x, x)
            fx += self.b
            result.append(np.sign(fx) if fx != 0 else 1)
        
        return np.array(result)

# 测试
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.array([1 if x[0]**2 + x[1]**2 > 1 else -1 for x in X])

svm = KernelSVM(C=1.0, gamma=0.5)
svm.fit(X, y)

y_pred = svm.predict(X)
accuracy = np.mean(y_pred == y)
print(f"RBF核SVM准确率: {accuracy:.4f}")
print(f"支持向量数量: {len(svm.support_vectors)}")
```

### 8.3 完整SMO实现

```python
import numpy as np

class SVMWithSMO:
    """
    完整的SMO算法实现的SVM
    """
    
    def __init__(self, C=1.0, tol=0.001, max_iter=1000):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.b = 0
        self.alpha = None
        self.X = None
        self.y = None
    
    def compute_E(self, i):
        """计算E_i"""
        return np.sum(self.alpha * self.y * 
                  np.dot(self.X, self.X[i])) + self.b - self.y[i]
    
    def takeStep(self, i1, i2, E1, E2):
        """优化一对alpha"""
        if i1 == i2:
            return 0
        
        alpha1_old = self.alpha[i1]
        alpha2_old = self.alpha[i2]
        y1, y2 = self.y[i1], self.y[i2]
        
        # 计算L和H
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if L == H:
            return 0
        
        # 计算核
        k11 = np.dot(self.X[i1], self.X[i1])
        k22 = np.dot(self.X[i2], self.X[i2])
        k12 = np.dot(self.X[i1], self.X[i2])
        eta = 2 * k12 - k11 - k22
        
        if eta < 0:
            alpha2_new = alpha2_old - y2 * (E1 - E2) / eta
            alpha2_new = max(L, min(H, alpha2_new))
        else:
            # 计算边界的f1和f2
            f1 = y1 * E1 + alpha1_old * y1 * k11 + y1 * y2 * alpha2_old * k12 + self.b
            f2 = y2 * E2 + y1 * alpha1_old * y1 * k12 + alpha2_old * y2 * k22 + self.b
            L1 = alpha1_old + y1 * y2 * (alpha2_old - L)
            H1 = alpha1_old + y1 * y2 * (alpha2_old - H)
            
            if L1 < f1 - self.tol:
                alpha2_new = L
            elif L1 > f1 + self.tol:
                alpha2_new = H
            elif H1 < f2 + self.tol:
                alpha2_new = H
            elif H1 > f2 - self.tol:
                alpha2_new = L
            else:
                alpha2_new = alpha2_old
        
        if abs(alpha2_new - alpha2_old) < 1e-5:
            return 0
        
        # 更新alpha
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        
        # 更新b
        b1 = E1 + y1 * (alpha1_new - alpha1_old) * k11 + \
             y2 * (alpha2_new - alpha2_old) * k12 + self.b
        b2 = E2 + y1 * (alpha1_new - alpha1_old) * k12 + \
             y2 * (alpha2_new - alpha2_old) * k22 + self.b
        
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        self.alpha[i1] = alpha1_new
        self.alpha[i2] = alpha2_new
        
        return 1
    
    def examineExample(self, i2):
        """检查和优化一个例子"""
        y2 = self.y[i2]
        alpha2 = self.alpha[i2]
        E2 = self.compute_E_E(i2)
        r2 = E2 * y2
        
        if (r2 < -self.tol and alpha2 < self.C) or \
           (r2 > self.tol and alpha2 > 0):
            
            # 获取所有非边界和边界的alpha
            non_bound = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
            
            if len(non_bound) > 1:
                # 选择最大化步长的j
                max_step = 0
                j = 0
                for i in non_bound:
                    E1 = self.compute_E_E(i)
                    step = abs(E2 - E1)
                    if step > max_step:
                        max_step = step
                        j = i
                
                if self.takeStep(j, i2, max_step, E2):
                    return 1
            
            # 随机选择
            idx = np.random.permutation(len(self.X))
            for j in idx:
                if self.takeStep(j, i2, 0, E2):
                    return 1
            
            # 检查所有
            for j in range(len(self.X)):
                if self.takeStep(j, i2, 0, E2):
                    return 1
        
        return 0
    
    def compute_E_E(self, i):
        """计算误差"""
        return np.sum(self.alpha * self.y * 
                  np.dot(self.X, self.X[i])) + self.b - self.y[i]
    
    def fit(self, X, y):
        """训练"""
        self.X = X
        self.y = y
        n = len(X)
        
        self.alpha = np.zeros(n)
        self.b = 0
        
        num_changed = 0
        examine_all = True
        
        for _ in range(self.max_iter):
            if examine_all:
                for i in range(n):
                    num_changed += self.examineExample(i)
            else:
                non_bound = np.where(self.alpha != 0)[0]
                for i in non_bound:
                    num_changed += self.examineExample(i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                break
            
            if examine_all == False and num_changed == 0:
                examine_all = True
        
        # 存储支持向量
        sv_idx = self.alpha > 1e-5
        self.X = self.X[sv_idx]
        self.y = self.y[sv_idx]
        self.alpha = self.alpha[sv_idx]
        
        return self
    
    def predict(self, X):
        """预测"""
        return np.sign(np.sum(self.alpha * self.y * 
                          np.dot(X, self.X.T), axis=1) + self.b)
```

---

## 9. 可视化与结果理解

### 9.1 决策边界可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

def plot_svm_decision_boundary():
    """绘制SVM决策边界"""
    # 生成数据
    X, y = make_blobs(n_samples=200, centers=2, random_state=42)
    
    # 训练SVM
    model = SVC(kernel='linear', C=1.0)
    model.fit(X, y)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
    
    # 预测网格
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)
    
    # 标记支持向量
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
              s=200, facecolors='none', edgecolors='k', linewidths=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_svm_decision_boundary()
```

### 9.2 核函数效果对比

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

def compare_kernels():
    """对比不同核函数的效果"""
    # 生成环形数据
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
    
    kernels = ['linear', 'rbf', 'poly']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, kernel in zip(axes, kernels):
        # 训练SVM
        model = SVC(kernel=kernel, C=1.0)
        model.fit(X, y)
        
        # 创建网格
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
        
        # 预测网格
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)
        ax.set_title(f'{kernel} kernel')
    
    plt.tight_layout()
    plt.show()

compare_kernels()
```

### 9.3 C参数影响可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons

def visualize_C_effect():
    """可视化C参数的影响"""
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    
    C_values = [0.01, 1, 100]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, C in zip(axes, C_values):
        model = SVC(kernel='rbf', C=C)
        model.fit(X, y)
        
        # 创建网格
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
        
        # 预测网格
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50,
                 edgecolors='k', linewidths=0.5)
        ax.set_title(f'C = {C}')
    
    plt.tight_layout()
    plt.show()

visualize_C_effect()
```

### 9.4 间隔可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

def visualize_margin():
    """可视化SVM的间隔"""
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    
    model = SVC(kernel='linear', C=1.0)
    model.fit(X, y)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)
    
    # 获取支持向量
    sv = model.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
               edgecolors='k', linewidths=2, label='Support Vectors')
    
    # 绘制决策边界和间隔
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 100)
    
    # 决策边界
    w = model.coef_[0]
    b = model.intercept_[0]
    y_decision = -w[0] / w[1] * xx - b / w[1]
    
    # 间隔边界
    margin = 1 / np.linalg.norm(w)
    y_pos = y_decision + margin
    y_neg = y_decision - margin
    
    plt.plot(xx, y_decision, 'k-', label='Decision Boundary')
    plt.plot(xx, y_pos, 'k--', label='Margin')
    plt.plot(xx, y_neg, 'k--')
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.title('SVM Margin')
    plt.show()

visualize_margin()
```

### 9.5 学习曲线

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_classification

def plot_learning_curve():
    """绘制学习曲线"""
    X, y = make_classification(n_samples=1000, random_state=42)
    
    train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        SVC(kernel='rbf'), X, y, 
        train_sizes=train_sizes, cv=5
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes_abs, train_mean - train_std,
                  train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes_abs, test_mean - test_std,
                  test_mean + test_std, alpha=0.1)
    plt.plot(train_sizes_abs, train_mean, 'o-', label='Training')
    plt.plot(train_sizes_abs, test_mean, 'o-', label='Validation')
    
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('SVM Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_learning_curve()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| Accuracy | 准确率 | 正确分类/总样本 |
| Precision | 精确率 | TP/(TP+FP) |
| Recall | 召回率 | TP/(TP+FN) |
| F1 | F1分数 | 2*Precision*Recall/(Precision+Recall) |
| AUC | ROC下面积 | 积分 |

### 10.2 完整评估方法

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=500, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 评估
print("分类报告:")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAUC: {roc_auc_score(y_test, y_prob):.4f}")
```

### 10.3 参数敏感性分析

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

def parameter_sensitivity_analysis():
    """参数敏感性分析"""
    X, y = make_classification(n_samples=500, random_state=42)
    
    # C的影响
    C_values = [0.01, 0.1, 1, 10, 100]
    scores = []
    
    for C in C_values:
        model = SVC(kernel='rbf', C=C)
        cv_scores = cross_val_score(model, X, y, cv=5)
        scores.append(np.mean(cv_scores))
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, scores, 'o-')
    plt.xlabel('C')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('C Parameter Sensitivity')
    plt.grid(True)
    plt.show()

parameter_sensitivity_analysis()
```

---

## 11. 常见问题与易错点

### 11.1 问题诊断表

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 准确率低 | C太小或太大 | 调整C或gamma |
| 过拟合 | C太大 | 减小C |
| 欠拟合 | C太小 | 增大C |
| 运行太慢 | 数据太大 | 使用线性核或采样 |
| 对参数敏感 | 核函数选择不当 | 尝试不同核函数 |

### 11.2 常见错误

```python
# 错误1: 没有标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 错误2: C参数选择不当
# C太大可能导致过拟合
# C太小可能导致欠拟合

# 错误3: 核函数选择不当
# 线性数据用RBF核
# 非线性数据用线性核
```

### 11.3 选择建议

```python
# 选择建议：
# 1. 先尝试线性核
# 2. 如果效果不好，尝试RBF核
# 3. 多分类用OvR策略
# 4. 大数据用LinearSVC
```

---

## 12. 学习总结

### 核心思想

SVM通过最大化两类样本之间的间隔来找到最优超平面，将不同类别的数据分开。支持向量是决定超平面位置的关键样本。

### 关键公式

优化目标：$\min_{w,b} \frac{1}{2}||w||^2$

决策函数：$f(x) = \text{sign}(w^Tx + b)$

### 后续学习

1. **核方法**：深入理解核技巧
2. **SMO算法**：高效求解
3. **多分类SVM**：One-vs-Rest、One-vs-One

---

## 13. 练习题与思考题

### 基础题

**题目1**：为什么SVM需要标准化？

**答案**：SVM使用核函数计算样本相似度，如果特征尺度不同，会导致尺度大的特征主导计算，忽略尺度小的特征。

**题目2**：支持向量是什么？

**答案**：距离决策边界最近的样本点，它们决定了超平面的位置。

### 进阶题

**题目3**：如何选择核函数？

**答案**：先尝试线性核，如果效果不好再尝试其他核函数。可以使用交叉验证来选择最优核函数。

### 思考题

**题目4**：SVM和神经网络相比有什么优缺点？

**答案**：SVM在小样本上效果好、可解释性强、理论基础扎实，但在大规模数据上效率较低，处理复杂模式的能力不如神经网络。

---

## 14. 学习路径建议

### 前置知识

1. **线性代数**：理解超平面
2. **优化理论**：理解拉格朗日乘数法
3. **概率统计**：理解分类指标

### 推荐学习路线

1. **入门**（1-2周）：
   - 理解SVM原理
   - 实现线性SVM

2. **进阶**（2-3周）：
   - 核函数
   - SMO算法

3. **实践**（持续）：
   - 实际项目应用
   - 超参数调优

### 推荐资源

1. **论文**：
   - Support-Vector Networks
   - LIBLINEAR

2. **书籍**：
   - 《统计学习方法》李航
   - 《机器学习》周志华

3. **工具**：
   - scikit-learn
   - LIBSVM

---

