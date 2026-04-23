# 模型评估指标详解

> 如何衡量模型的好坏？这是机器学习最核心的问题之一。

---

## 1. 回归评估指标

回归问题预测连续值，主要评估预测值与真实值的差异。

### 1.1 MSE（均方误差）

**公式**：
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**理解**：
- 计算预测值与真实值差的平方的平均值
- 对大误差更敏感（因为平方）
- 单位是原始单位的平方

**Python实现**：
```python
import numpy as np
from sklearn.metrics import mean_squared_error

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

# 手写实现
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# sklearn实现
mse_value = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse_value:.4f}")
```

### 1.2 RMSE（均方根误差）

**公式**：
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**理解**：
- MSE的平方根
- 与原始数据单位相同，更直观
- 最常用的回归指标

**Python实现**：
```python
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.4f}")
```

### 1.3 MAE（平均绝对误差）

**公式**：
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**理解**：
- 预测值与真实值差的绝对值的平均
- 对异常值不如MSE敏感
- 单位与原始数据相同

**Python实现**：
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")
```

**MSE vs MAE 对比**：

| 特性 | MSE | MAE |
|------|-----|-----|
| 对异常值敏感度 | 高（平方放大） | 低 |
| 可导性 | 处处可导 | 在0点不可导 |
| 优化方法 | 方便求导优化 | 需要线性规划 |
| 适用场景 | 数据干净、无异常值 | 数据有噪声/异常值 |

### 1.4 R²（决定系数）

**公式**：
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

**理解**：
- 衡量模型解释数据变异的比例
- 范围：通常在0-1之间
  - R² = 1：完美预测
  - R² = 0：和用均值预测一样
  - R² < 0：比用均值预测还差
- 无单位，便于比较不同数据集

**Python实现**：
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
```

**重要说明**：
```python
# R²的局限性：随着特征增加，R²总是增加
# 使用调整R²（Adjusted R²）更准确
n = len(y_true)  # 样本数
p = 3  # 特征数
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Adjusted R²: {adjusted_r2:.4f}")
```

---

## 2. 分类评估指标

分类问题预测离散标签，评估更为复杂。

### 2.1 混淆矩阵

**二分类混淆矩阵**：

```
                 预测为正    预测为负
实际为正    TP (真正例)  FN (假负例)
实际为负    FP (假正例)  TN (真负例)
```

**Python实现**：
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print("混淆矩阵:")
print(cm)

# 可视化
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### 2.2 准确率（Accuracy）

**公式**：
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**理解**：
- 预测正确的比例
- 适合类别平衡的数据
- 类别不平衡时会产生误导

**Python实现**：
```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")
```

### 2.3 精确率（Precision）

**公式**：
$$Precision = \frac{TP}{TP + FP}$$

**理解**：
- 预测为正的样本中，实际为正的比例
- 关注：预测为正的可靠性
- 也叫：查准率

**应用场景**：
- 垃圾邮件分类：不想把正常邮件误判为垃圾邮件
- 推荐系统：推荐的内容要准确

**Python实现**：
```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")
```

### 2.4 召回率（Recall）

**公式**：
$$Recall = \frac{TP}{TP + FN}$$

**理解**：
- 实际为正的样本中，预测为正的比例
- 关注：正样本被找出来的能力
- 也叫：查全率、敏感度（Sensitivity）

**应用场景**：
- 癌症诊断：不能漏掉任何患者
- 欺诈检测：宁可误判也不能漏掉欺诈

**Python实现**：
```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")
```

### 2.5 F1分数

**公式**：
$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

**理解**：
- 精确率和召回率的调和平均
- 平衡精确率和召回率
- 适合类别不平衡的场景

**Python实现**：
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")
```

**为什么要用调和平均？**
```python
# 算术平均 vs 调和平均
precision, recall = 0.9, 0.1

arithmetic_mean = (precision + recall) / 2  # 0.5
harmonic_mean = 2 * precision * recall / (precision + recall)  # 0.18

# 调和平均对极端值更敏感，更能反映短板
```

### 2.6 特异度（Specificity）

**公式**：
$$Specificity = \frac{TN}{TN + FP}$$

**理解**：
- 实际为负的样本中，预测为负的比例
- 与召回率对应

---

## 3. ROC与AUC

### 3.1 ROC曲线

**理解**：
- 横轴：FPR（假正率）= FP / (FP + TN)
- 纵轴：TPR（真正率）= TP / (TP + FN) = Recall
- 通过改变分类阈值绘制

**Python实现**：
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 需要概率输出，不是类别标签
y_scores = [0.9, 0.1, 0.8, 0.3, 0.2, 0.85, 0.7, 0.1, 0.95, 0.05]

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

### 3.2 AUC（ROC曲线下面积）

**理解**：
- AUC = 1.0：完美分类器
- AUC = 0.5：随机猜测
- AUC < 0.5：比随机还差（反转预测即可）

**AUC的物理意义**：
```
AUC = P(正样本得分 > 负样本得分)
```

**Python实现**：
```python
from sklearn.metrics import roc_auc_score

auc_score = roc_auc_score(y_true, y_scores)
print(f"AUC: {auc_score:.4f}")
```

### 3.3 ROC曲线解读

```python
# 不同模型的ROC曲线对比
plt.figure(figsize=(10, 8))

# 好的模型：AUC接近1
fpr_good, tpr_good, _ = roc_curve(y_true, y_scores)
plt.plot(fpr_good, tpr_good, label=f'Good Model (AUC = {auc(fpr_good, tpr_good):.2f})')

# 随机模型：AUC接近0.5
random_scores = np.random.random(len(y_true))
fpr_random, tpr_random, _ = roc_curve(y_true, random_scores)
plt.plot(fpr_random, tpr_random, label=f'Random (AUC = {auc(fpr_random, tpr_random):.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
plt.legend()
plt.show()
```

---

## 4. 多分类指标

### 4.1 多分类混淆矩阵

```python
from sklearn.metrics import confusion_matrix

y_true_multi = [0, 1, 2, 0, 1, 2]
y_pred_multi = [0, 2, 2, 0, 1, 1]

cm = confusion_matrix(y_true_multi, y_pred_multi)
print("多分类混淆矩阵:")
print(cm)
```

### 4.2 多分类评估

```python
from sklearn.metrics import classification_report

print(classification_report(y_true_multi, y_pred_multi))

# 输出示例:
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00         2
#            1       1.00      0.50      0.67         2
#            2       0.50      0.50      0.50         2
#
#     accuracy                           0.67         6
#    macro avg       0.83      0.67      0.72         6
# weighted avg       0.83      0.67      0.72         6
```

**宏平均 vs 微平均**：
- **Macro-average**：各类别指标取平均（每个类别权重相同）
- **Micro-average**：所有样本一起计算（大类别权重更大）
- **Weighted-average**：按类别样本数加权平均

---

## 5. 指标选择指南

### 5.1 回归问题

| 场景 | 推荐指标 |
|------|----------|
| 一般回归 | RMSE + R² |
| 有异常值 | MAE |
| 需要解释性 | R² |
| 比较不同模型 | R² 或 RMSE |

### 5.2 分类问题

| 场景 | 推荐指标 |
|------|----------|
| 类别平衡 | Accuracy |
| 类别不平衡 | F1-score, AUC |
| 关注假正例 | Precision |
| 关注假负例 | Recall |
| 医疗诊断 | Recall（不能漏） |
| 垃圾邮件 | Precision（不能误判） |

### 5.3 决策流程

```
开始
  │
  ▼
类别是否平衡？
  │
  ├─ 是 ──> 使用Accuracy
  │
  └─ 否 ──> 假正例和假负例哪个更严重？
              │
              ├─ 假正例严重 ──> 优化Precision
              │
              ├─ 假负例严重 ──> 优化Recall
              │
              └─ 都严重 ──> 优化F1或AUC
```

---

## 6. 实战示例

### 6.1 完整评估流程

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# 生成数据
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 评估
print("=" * 50)
print("模型评估报告")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_test, y_prob):.4f}")
print("=" * 50)
print("\n详细报告:")
print(classification_report(y_test, y_pred))
```

---

## 7. 常见误区

### 7.1 过度依赖准确率

```python
# 类别不平衡的例子
y_true = [0] * 990 + [1] * 10  # 99%是负类
y_pred = [0] * 1000  # 全部预测为负类

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")  # 0.99!
print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")  # 0.0
```

### 7.2 阈值选择

```python
# 默认阈值是0.5，但不一定最优
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# 找到F1最大的阈值
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"最优阈值: {best_threshold:.4f}")
```

---

## 8. 总结

| 指标 | 公式 | 适用场景 |
|------|------|----------|
| MSE | 均方误差 | 回归，对异常值敏感 |
| RMSE | √MSE | 回归，最常用 |
| MAE | 平均绝对误差 | 回归，有异常值 |
| R² | 决定系数 | 回归，解释性 |
| Accuracy | 准确率 | 平衡分类 |
| Precision | TP/(TP+FP) | 关注误报 |
| Recall | TP/(TP+FN) | 关注漏报 |
| F1 | PR调和平均 | 不平衡分类 |
| AUC | ROC曲线面积 | 概率排序能力 |

记住：**没有最好的指标，只有最适合的指标！**
