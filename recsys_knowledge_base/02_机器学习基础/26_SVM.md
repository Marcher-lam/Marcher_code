# SVM 支持向量机 学习文档

## 1. 算法基础认知

### 1.1 什么是 SVM？

```
支持向量机 (Support Vector Machine):

- 寻找最优分类超平面
- 最大化分类间隔 (Margin)
- 核技巧处理非线性问题

在推荐系统中的应用:
- 文本分类 (情感分析、意图识别)
- 用户画像分类
- 异常检测
- 早期 CTR 预估

核心思想:
- 找到一个超平面，使两类样本距离最大化
- 距离超平面最近的样本点叫"支持向量"
```

### 1.2 几何理解

```python
"""
超平面方程: w·x + b = 0

点到超平面的距离:
d = |w·x + b| / ||w||

间隔 (Margin):
Margin = 2 / ||w||

目标: 最大化间隔 = 最小化 ||w||
"""

import numpy as np
from typing import List, Tuple, Optional
from cvxopt import matrix, solvers


class SVMBasic:
    """
    SVM 基础实现
    """

    def __init__(self, C: float = 1.0, kernel: str = 'linear'):
        """
        参数:
            C: 软间隔惩罚参数
            kernel: 核函数类型
        """
        self.C = C
        self.kernel = kernel
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_labels = None
        self.alphas = None

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """核函数"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            gamma = 0.1
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'poly':
            degree = 3
            return (np.dot(x1, x2) + 1) ** degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练 SVM

        使用二次规划求解
        """
        n_samples, n_features = X.shape

        # 构建二次规划问题
        # min 1/2 * α.T * Q * α - e.T * α
        # s.t. y.T * α = 0, 0 <= α <= C

        # 计算核矩阵 Q
        Q = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                Q[i, j] = y[i] * y[j] * self._kernel(X[i], X[j])

        # 转为 cvxopt 格式
        P = matrix(Q)
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack([
            -np.eye(n_samples),
            np.eye(n_samples)
        ]))
        h = matrix(np.hstack([
            np.zeros(n_samples),
            np.ones(n_samples) * self.C
        ]))
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix(0.0)

        # 求解
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x']).flatten()

        # 找到支持向量
        sv_threshold = 1e-5
        sv_indices = np.where(alphas > sv_threshold)[0]

        self.support_vectors = X[sv_indices]
        self.support_labels = y[sv_indices]
        self.alphas = alphas[sv_indices]

        # 计算权重 (线性核)
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for i, idx in enumerate(sv_indices):
                self.w += self.alphas[i] * self.support_labels[i] * self.support_vectors[i]

        # 计算 b
        b_values = []
        for i, idx in enumerate(sv_indices):
            if self.alphas[i] < self.C - sv_threshold:  # 自由支持向量
                b = self.support_labels[i]
                for j in range(len(self.support_vectors)):
                    b -= self.alphas[j] * self.support_labels[j] * self._kernel(
                        self.support_vectors[j], self.support_vectors[i]
                    )
                b_values.append(b)

        self.b = np.mean(b_values) if b_values else 0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        decision = self.decision_function(X)
        return np.sign(decision)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """决策函数值"""
        if self.kernel == 'linear':
            return np.dot(X, self.w) + self.b
        else:
            # 核方法
            decision = np.zeros(len(X))
            for i, x in enumerate(X):
                for j, sv in enumerate(self.support_vectors):
                    decision[i] += self.alphas[j] * self.support_labels[j] * self._kernel(sv, x)
            return decision + self.b


class SoftMarginSVM:
    """
    软间隔 SVM

    允许一些样本被错误分类
    """

    def __init__(self, C: float = 1.0, learning_rate: float = 0.01,
                 n_epochs: int = 1000):
        self.C = C
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        使用梯度下降训练软间隔 SVM

        损失函数: Hinge Loss
        L = 1/2 ||w||² + C * Σ max(0, 1 - y_i(w·x_i + b))
        """
        n_samples, n_features = X.shape

        # 初始化
        self.w = np.zeros(n_features)
        self.b = 0

        # 梯度下降
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                # 计算边界
                margin = y[i] * (np.dot(X[i], self.w) + self.b)

                if margin < 1:
                    # 错误分类或边界内
                    self.w += self.lr * (self.w - self.C * y[i] * X[i])
                    self.b += self.lr * (-self.C * y[i])
                else:
                    # 正确分类
                    self.w += self.lr * self.w

            # 学习率衰减
            self.lr *= 0.99

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return np.sign(np.dot(X, self.w) + self.b)
```

## 2. 核方法

### 2.1 核函数

```python
class KernelSVM:
    """
    核 SVM

    使用核技巧处理非线性分类
    """

    def __init__(self, C: float = 1.0, kernel: str = 'rbf',
                 gamma: float = 0.1, degree: int = 3):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree

        self.support_vectors = None
        self.support_labels = None
        self.alphas = None
        self.b = None

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """核函数"""
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)

        elif self.kernel_type == 'rbf':
            # K(x, y) = exp(-γ ||x - y||²)
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

        elif self.kernel_type == 'poly':
            # K(x, y) = (x · y + 1)^d
            return (np.dot(x1, x2) + 1) ** self.degree

        elif self.kernel_type == 'sigmoid':
            # K(x, y) = tanh(γ x · y + r)
            return np.tanh(self.gamma * np.dot(x1, x2) + 0)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(X[i], X[j])
        return K


class RBFKernelSVM:
    """
    RBF 核 SVM 详解

    RBF (径向基函数) 核也叫高斯核
    """

    def __init__(self, C: float = 1.0, gamma: float = 'scale'):
        """
        参数:
            C: 正则化参数
            gamma: 核参数
                - 'scale': 1 / (n_features * X.var())
                - 'auto': 1 / n_features
                - float: 指定值
        """
        self.C = C
        self.gamma = gamma
        self._gamma_value = None

    def _set_gamma(self, X: np.ndarray):
        """设置 gamma 值"""
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                self._gamma_value = 1 / (X.shape[1] * X.var())
            elif self.gamma == 'auto':
                self._gamma_value = 1 / X.shape[1]
        else:
            self._gamma_value = self.gamma

    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF 核函数

        K(x, y) = exp(-γ ||x - y||²)

        参数 γ 控制单个样本的影响范围:
        - γ 大: 每个样本影响范围小，容易过拟合
        - γ 小: 每个样本影响范围大，容易欠拟合
        """
        # 计算平方欧氏距离
        # ||x - y||² = ||x||² + ||y||² - 2x·y
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)

        distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)

        return np.exp(-self._gamma_value * distances)
```

## 3. 推荐系统应用

### 3.1 文本分类

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


class SVMTextClassifier:
    """
    SVM 文本分类器
    """

    def __init__(self, C: float = 1.0, kernel: str = 'linear'):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('svm', SVC(C=C, kernel=kernel, probability=True))
        ])

    def fit(self, texts: List[str], labels: List[int]):
        """训练"""
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """预测"""
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """预测概率"""
        return self.pipeline.predict_proba(texts)


class SentimentSVM:
    """
    基于 SVM 的情感分析
    """

    def __init__(self):
        self.classifier = SVMTextClassifier(C=1.0, kernel='linear')

    def train(self, texts: List[str], sentiments: List[str]):
        """训练"""
        # 转换标签
        label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        labels = [label_map[s] for s in sentiments]

        self.classifier.fit(texts, labels)

    def analyze(self, text: str) -> Dict[str, float]:
        """分析情感"""
        proba = self.classifier.predict_proba([text])[0]

        return {
            'negative': float(proba[0]),
            'neutral': float(proba[1]),
            'positive': float(proba[2])
        }
```

### 3.2 异常检测

```python
class OneClassSVM:
    """
    单类 SVM

    用于异常检测
    """

    def __init__(self, nu: float = 0.1, kernel: str = 'rbf', gamma: float = 'scale'):
        """
        参数:
            nu: 异常比例上限 (0, 1]
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

        self.support_vectors = None
        self.rho = None  # 偏置

    def fit(self, X: np.ndarray):
        """
        训练

        只使用正常样本
        """
        from sklearn.svm import OneClassSVM as SklearnOCSVM

        self.model = SklearnOCSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model.fit(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        1: 正常
        -1: 异常
        """
        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """决策函数"""
        return self.model.decision_function(X)


class UserBehaviorAnomalyDetector:
    """
    用户行为异常检测

    使用 One-Class SVM
    """

    def __init__(self, nu: float = 0.05):
        self.svm = OneClassSVM(nu=nu, kernel='rbf')
        self.fitted = False

    def fit(self, normal_behaviors: np.ndarray):
        """
        训练

        normal_behaviors: 正常用户的行为特征
        """
        self.svm.fit(normal_behaviors)
        self.fitted = True

    def detect(self, behavior: np.ndarray) -> bool:
        """
        检测是否异常

        返回 True 表示异常
        """
        if not self.fitted:
            return False

        prediction = self.svm.predict(behavior.reshape(1, -1))
        return prediction[0] == -1

    def get_anomaly_score(self, behavior: np.ndarray) -> float:
        """
        获取异常分数

        负值越小越异常
        """
        if not self.fitted:
            return 0.0

        return float(self.svm.decision_function(behavior.reshape(1, -1))[0])
```

## 4. 使用 scikit-learn

### 4.1 快速实现

```python
from sklearn.svm import SVC, SVR, LinearSVC, OneClassSVM
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


def train_svm_classifier(X, y, kernel='rbf', C=1.0, gamma='scale'):
    """
    训练 SVM 分类器
    """
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练
    clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    clf.fit(X_scaled, y)

    return clf, scaler


def svm_grid_search(X, y):
    """
    SVM 超参数网格搜索
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }

    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train_scaled, y_train)

    # 预测
    y_pred = clf.predict(X_test_scaled)

    # 评估
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 支持向量数量
    print(f"\nNumber of support vectors: {len(clf.support_vectors_)}")
```

## 5. 学习总结

### 5.1 核心要点

```
1. 硬间隔: 严格可分，无误差
2. 软间隔: 允许误差，使用 C 参数控制
3. 核技巧: 映射到高维空间处理非线性
4. 支持向量: 决定分类边界的样本点
```

### 5.2 核函数选择

```
核函数        适用场景                特点
─────────────────────────────────────────────
线性核        高维稀疏数据、文本      速度快，可解释
RBF核         一般场景                灵活，需要调参
多项式核      特征间有非线性关系      需要选择阶数
Sigmoid核     类似神经网络            较少使用
```

### 5.3 参数调优

```
参数        影响                    调优建议
───────────────────────────────────────────────
C           控制软间隔              交叉验证选择
gamma       RBF核的影响范围         过大过拟合，过小欠拟合
kernel      核函数类型              根据数据特点选择
```

### 5.4 优缺点

```
优点:
- 在高维空间表现好
- 核技巧处理非线性
- 泛化能力强
- 只依赖支持向量

缺点:
- 大数据集训练慢
- 需要调参
- 对噪声敏感
- 不直接输出概率
```
