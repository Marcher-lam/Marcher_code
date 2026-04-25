# 评估指标详解

## 回归任务指标

### 1. 均方误差 (MSE)
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
- 特点：惩罚大误差，对异常值敏感
- 适用：连续值预测

### 2. 均方根误差 (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}}$$
- 特点：与目标变量同量纲，更易解释
- 适用：需要具体误差尺度的场景

### 3. 平均绝对误差 (MAE)
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
- 特点：鲁棒性好，对异常值不敏感
- 适用：异常值较多的数据

### 4. R²决定系数
$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
- 范围：(-∞, 1]，值越大越好
- 解释：模型解释的方差比例

## 分类任务指标

### 1. 准确率 (Accuracy)
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
- 适用：类别均衡的数据集
- 缺点：类别不平衡时失效

### 2. 精确率 (Precision)
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- 适用：关注假阳性代价高的场景（如垃圾邮件检测）

### 3. 召回率 (Recall)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- 适用：关注假阴性代价高的场景（如疾病诊断）

### 4. F1分数
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- 适用：精确率和召回率需要平衡的场景

### 5. ROC曲线与AUC
- ROC：以假阳性率为横轴，真阳性率为纵轴
- AUC：ROC曲线下的面积，范围[0,1]
- 适用：二分类模型比较

## 排序任务指标

### 1. NDCG (Normalized Discounted Cumulative Gain)
- 考虑排序位置的增益度量
- 范围[0,1]，值越大越好

### 2. MAP (Mean Average Precision)
- 多查询场景下的平均准确率
- 适用于信息检索

## 回归任务代码示例
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_true = np.array([3, -0, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

## 分类任务代码示例
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])
y_prob = np.array([0.1, 0.9, 0.3, 0.2, 0.8])  # 预测为正类的概率

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print(f"AUC: {auc:.4f}")
```

## 不同场景的指标选择

| 场景 | 推荐指标 | 说明 |
|------|---------|------|
| 房价预测 | RMSE、R² | 误差有明确物理意义 |
| 文本分类 | F1、Precision | 关注类别不平衡 |
| 医疗诊断 | Recall、AUC | 避免漏诊更重要 |
| 推荐系统 | NDCG、MAP | 排序质量更重要 |
| 欺诈检测 | Precision、F1 | 假阳性代价高 |

## 过拟合/欠拟合诊断
- **训练误差大，验证误差大**：欠拟合 → 增加模型复杂度
- **训练误差小，验证误差大**：过拟合 → 增加正则化
- **训练误差≈验证误差**：良好拟合

## 多任务学习指标
- 加权平均：根据任务重要性加权
- 不平衡任务：使用宏平均（macro）而非微平均（micro）
