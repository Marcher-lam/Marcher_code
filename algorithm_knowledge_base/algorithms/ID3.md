# ID3 学习文档

> ID3 (Iterative Dichotomiser 3) 是一种经典的决策树学习算法,使用信息增益作为分裂准则递归地构建决策树。

---

## 1. 算法基础认知

### 一句话定义
ID3 通过计算各特征的信息增益,选择信息增益最大的特征作为当前节点的分裂属性,递归构建决策树。

### 直觉类比
想象医生诊断疾病:
- "患者有发烧吗?" → 是 → "咳嗽吗?" → ...
- 每次问一个最能区分病情的问题
- ID3就是自动找出"最好问题"的算法

### 历史背景
- 1986年,Quinlan在Machine Learning提出
- 是决策树算法的开山之作
- 后续C4.5、CART的基石

### 算法定位
- **类型**:有监督学习/分类
- **输出**:决策树规则
- **模型类型**:树模型

### 前置知识
- 信息熵
- 信息增益
- 递归算法

---

## 2. 核心原理

### 2.1 核心思想
ID3核心是**选择最优分裂特征**:

1. **计算熵**: 衡量数据的纯度
2. **计算信息增益**: 分裂前后的熵减
3. **选择最大增益特征**: 作为分裂属性
4. **递归**: 对每个分支重复

### 2.2 工作流程
```
数据D, 特征集A
    ↓
计算D的熵 H(D)
    ↓
对每个特征a∈A:
    计算条件熵 H(D|a)
    计算增益 Gain(a) = H(D) - H(D|a)
    ↓
选择最大增益的特征作为根节点
    ↓
对每个分支递归构建子树
```

### 2.3 关键概念
- **信息熵**: $H(S) = -\sum_{i} p_i \log_2 p_i$
- **条件熵**: $H(S|A) = \sum_{v} \frac{|S_v|}{|S|} H(S_v)$
- **信息增益**: $IG(A) = H(S) - H(S|A)$

### 2.4 架构图
```
┌─────────────────────────────────────────────┐
│              ID3决策树                      │
│                                             │
│           [根节点: 天气?]                    │
│          /      |        \                  │
│      晴天      阴天      下雨               │
│        ↓        ↓         ↓               │
│    [去?]     [去?]   [不去?]               │
│    /   \     /   \                           │
│  去  不去   去  不去                          │
└─────────────────────────────────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $S$ | 训练样本集 |
| $C$ | 类别集合 |
| $A$ | 属性集合 |
| $H(S)$ | 熵 |
| $IG(A)$ | 属性A的信息增益 |

### 3.2 信息熵
$$H(S) = -\sum_{c \in C} p_c \log_2 p_c$$

其中 $p_c = \frac{|S_c|}{|S|}$, $S_c$ 是类别c的样本数。

### 3.3 条件熵
$$H(S|A) = \sum_{v \in V_a} \frac{|S_v|}{|S|} H(S_v)$$

其中 $V_a$ 是属性a的可能值, $S_v$ 是在属性a上取v的样本。

### 3.4 信息增益
$$IG(A) = H(S) - H(S|A)$$

选择增益最大的属性作为分裂属性。

### 3.5 算法步骤

```python
# ID3 伪代码

def ID3(S, A, C):
    # 创建根节点
    if 所有样本同类:
        return 单节点树, 类别C
    
    if A为空:
        return 单节点树, 最多数类别
    
    # 选择最优属性
    A* = argmax_a Gain(S, a)
    
    # 创建决策节点
    对 A* 的每个值 v:
        S_v = {s ∈ S | s[A*] = v}
        
        如果 S_v 为空:
            添加叶子节点, 类别=最多数类别
        否则:
            添加子树 = ID3(S_v, A-{A*}, C)
    
    return 决策节点
```

---

## 4. 训练过程

### 4.1 实现代码

```python
"""
ID3 决策树完整实现
"""

import numpy as np
from collections import Counter
import math

class ID3DecisionTree:
    """ID3决策树"""
    
    def __init__(self):
        self.tree_ = None
        self.feature_names_ = None
    
    def _entropy(self, labels):
        """计算熵"""
        counter = Counter(labels)
        total = len(labels)
        
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _information_gain(self, X, y, feature_idx):
        """计算信息增益"""
        # 原始熵
        H_S = self._entropy(y)
        
        # 按特征分组
        groups = {}
        for i, x in enumerate(X[:, feature_idx]):
            if x not in groups:
                groups[x] = []
            groups[x].append(y[i])
        
        # 加权条件熵
        H_S_A = 0
        total = len(y)
        
        for group_labels in groups.values():
            weight = len(group_labels) / total
            H_S_A += weight * self._entropy(group_labels)
        
        return H_S - H_S_A
    
    def _best_feature(self, X, y, feature_indices):
        """选择最优特征"""
        best_gain = -1
        best_feature = None
        
        for idx in feature_indices:
            gain = self._information_gain(X, y, idx)
            if gain > best_gain:
                best_gain = gain
                best_feature = idx
        
        return best_feature, best_gain
    
    def _build_tree(self, X, y, feature_indices, feature_names=None):
        """递归构建树"""
        # 情况1: 所有样本同类
        if len(set(y)) == 1:
            return {'leaf': y[0]}
        
        # 情况2: 无特征可用或样本为空
        if len(feature_indices) == 0 or len(y) == 0:
            return {'leaf': Counter(y).most_common(1)[0][0]}
        
        # 选择最优特征
        best_feature, best_gain = self._best_feature(X, y, feature_indices)
        
        if best_gain <= 0:
            return {'leaf': Counter(y).most_common(1)[0][0]}
        
        # 创建节点
        node = {'feature': best_feature}
        if feature_names is not None:
            node['feature_name'] = feature_names[best_feature]
        
        # 获取特征值
        feature_values = set(X[:, best_feature])
        remaining_features = [f for f in feature_indices if f != best_feature]
        
        # 递归构建子树
        branches = {}
        for value in feature_values:
            mask = X[:, best_feature] == value
            X_subset = X[mask]
            y_subset = y[mask]
            
            if len(y_subset) == 0:
                branches[value] = {'leaf': Counter(y).most_common(1)[0][0]}
            else:
                branches[value] = self._build_tree(
                    X_subset, y_subset, 
                    remaining_features, 
                    feature_names
                )
        
        node['branches'] = branches
        
        return node
    
    def fit(self, X, y, feature_names=None):
        """训练"""
        self.feature_names_ = feature_names
        feature_indices = list(range(X.shape[1]))
        self.tree_ = self._build_tree(X, y, feature_indices, feature_names)
        return self
    
    def _predict_single(self, x, node):
        """预测单个样本"""
        if 'leaf' in node:
            return node['leaf']
        
        feature_idx = node['feature']
        feature_value = x[feature_idx]
        
        if feature_value in node['branches']:
            return self._predict_single(x, node['branches'][feature_value])
        else:
            # 如果遇到未见的特征值,返回最常见类别
            return None
    
    def predict(self, X):
        """预测"""
        return np.array([self._predict_single(x, self.tree_) for x in X])
    
    def print_tree(self, node=None, depth=0):
        """打印树"""
        if node is None:
            node = self.tree_
        
        indent = "  " * depth
        
        if 'leaf' in node:
            print(f"{indent}-> {node['leaf']}")
        else:
            fname = node.get('feature_name', f"Feature {node['feature']}")
            print(f"{indent}[{fname}]?")
            
            for value, child in node['branches'].items():
                print(f"{indent}  {value}:")
                self.print_tree(child, depth + 1)


def train_id3():
    """训练示例"""
    # 天气数据
    X = np.array([
        ['Sunny', 'Hot', 'High', 'Weak'],
        ['Sunny', 'Hot', 'High', 'Strong'],
        ['Overcast', 'Hot', 'High', 'Weak'],
        ['Rainy', 'Mild', 'High', 'Weak'],
        ['Rainy', 'Cool', 'Normal', 'Weak'],
        ['Rainy', 'Cool', 'Normal', 'Strong'],
        ['Overcast', 'Cool', 'Normal', 'Strong'],
        ['Sunny', 'Mild', 'High', 'Weak'],
        ['Sunny', 'Cool', 'Normal', 'Weak'],
        ['Rainy', 'Mild', 'Normal', 'Weak'],
        ['Sunny', 'Mild', 'Normal', 'Strong'],
        ['Overcast', 'Mild', 'High', 'Strong'],
        ['Overcast', 'Hot', 'Normal', 'Weak'],
        ['Rainy', 'Mild', 'High', 'Strong'],
    ])
    
    y = np.array([
        'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
        'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'
    ])
    
    feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    
    tree = ID3DecisionTree()
    tree.fit(X, y, feature_names)
    
    return tree
```

### 4.2 超参数

| 参数 | 说明 |
|------|------|
| max_depth | 最大树深度 |
| min_samples_split | 分裂最小样本数 |
| criterion | 分裂准则(仅ID3:仅信息增益) |

---

## 5. 应用场景

### 5.1 典型应用
- **分类**:医疗诊断、风险评估
- **规则提取**:可解释性强
- **特征选择**:基于树的特征重要性

### 5.2 适用数据
- 离散/类别特征
- 小规模数据
- 需要可解释性

---

## 6. 优缺点

### 6.1 优点
| 优点 | 说明 |
|------|------|
| 简单 | 实现容易 |
| 可解释 | 规则直观 |
| 高效 | O(n * d * log n) |

### 6.2 缺点
| 缺点 | 缓解 |
|------|------|
| 过拟合 | 剪枝(后剪枝) |
| 偏向多值 | 使用增益率(C4.5) |
| 连续特征 | 离散化 |

---

## 7. 调库实现

```python
"""
sklearn实现
"""
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy', max_depth=10)
clf.fit(X, y)

# 预测
predictions = clf.predict(X_test)
```

---

## 8. 手工实现

```python
"""
ID3 核心简化版
"""

import numpy as np
from collections import Counter
import math

class SimpleID3:
    """简化ID3"""
    
    def fit(self, X, y):
        self.tree_ = self._build(X, y, list(range(X.shape[1])))
        return self
    
    def _entropy(self, labels):
        counter = Counter(labels)
        return -sum(
            c/len(labels) * math.log2(c/len(labels)) 
            for c in counter.values()
        )
    
    def _gain(self, X, y, idx):
        H = self._entropy(y)
        
        groups = {}
        for i, v in enumerate(X[:, idx]):
            groups.setdefault(v, []).append(y[i])
        
        weighted_H = sum(
            len(g) * self._entropy(g) / len(y) 
            for g in groups.values()
        )
        
        return H - weighted_H
    
    def _build(self, X, y, indices):
        if len(set(y)) == 1:
            return y[0]
        
        if len(indices) == 0:
            return Counter(y).most_common(1)[0][0]
        
        best = max(indices, key=lambda i: self._gain(X, y, i))
        node = {best: {}}
        
        remaining = [i for i in indices if i != best]
        groups = {}
        
        for i, v in enumerate(X[:, best]):
            groups.setdefault(v, ([], [])).append((X[i], y[i]))
        
        for v, (X_sub, y_sub) in groups.items():
            node[best][v] = self._build(
                np.array(X_sub), np.array(y_sub), remaining
            ) if len(y_sub) else Counter(y).most_common(1)[0][0]
        
        return node
    
    def predict(self, X):
        return np.array([self._predict(x, self.tree_) for x in X])
    
    def _predict(self, x, node):
        if not isinstance(node, dict):
            return node
        feature = list(node)[0]
        v = x[feature]
        return self._predict(x, node[feature].get(v, list(node.values())[0]))
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt

def plot_tree_importance():
    """特征重要性"""
    plt.figure(figsize=(8, 4))
    # 略
    plt.show()
```

---

## 10. 总结

### 核心要点
1. **信息增益**:选择最优分裂
2. **熵度量**:衡量纯度
3. **递归构建**:自顶向下
4. **可解释**:规则树

### 算法链
```
ID3 → C4.5 (增益率) → CART (基尼)
```

---

## 11. 练习题

**习题1**: 信息增益计算

<details>
<summary>答案</summary>

$IG(A) = H(S) - H(S|A)$

</details>

**习题2**: 为什么不能处理连续特征?

<details>
<summary>答案</summary>

需要离散化处理连续特征。

</details>

---

## 12. 学习路径

- **初级**: 信息熵、增益计算
- **中级**: 实现完整树、剪枝
- **高级**: C4.5、CART对比