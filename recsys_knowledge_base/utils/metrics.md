# 推荐系统评估指标详解

> 从基础回归指标到高级排序指标的完整指南

---

## 目录

1. [回归评估指标](#1-回归评估指标)
2. [分类评估指标](#2-分类评估指标)
3. [排序评估指标](#3-排序评估指标)
4. [指标选择指南](#4-指标选择指南)
5. [实战代码示例](#5-实战代码示例)

---

## 1. 回归评估指标

### 1.1 均方误差（MSE）

**定义**: 预测值与真实值差值平方的平均

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**特点**:
- 对异常值敏感（平方放大误差）
- 单位：原始值的平方
- 可微分，适合优化

**在推荐中**:
- 评分预测（MovieLens评分预测）
- 时长预测（预测用户观看时长）
- 价格预测

### 1.2 均方根误差（RMSE）

**定义**: MSE的平方根

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**与MSE关系**: RMSE = √MSE

**特点**:
- 单位与原始值一致
- 更直观解释
- 对异常值敏感

**在推荐中**:
- 推荐系统评估中最常用
- Netflix Prize官方评估指标
- 评分任务标准指标

### 1.3 平均绝对误差（MAE）

**定义**: 预测值与真实值差值绝对值的平均

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**特点**:
- 对异常值不敏感（绝对值）
- 单位与原始值一致
- 线性误差

**在推荐中**:
- 预测评分的鲁棒指标
- 时长预测（不受极端值影响）

### 1.4 决定系数（R²）

**定义**: 模型解释的方差比例

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**含义**:
- R² = 1: 完美拟合
- R² = 0: 等同于预测均值
- R² < 0: 比预测均值还差

**在推荐中**:
- 评估回归模型的拟合质量
- 判断模型是否有效（R² > 0）

---

## 2. 分类评估指标

### 2.1 准确率（Accuracy）

**定义**: 预测正确的样本占比

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**适用场景**:
- 类别平衡（正负样本接近）
- 每个类别同等重要

**在推荐中**:
- ⚠️ **不常用** - 推荐通常样本不平衡
- 少量推荐物品可能被误分类

### 2.2 精确率（Precision）

**定义**: 预测为正样本中，真正为正样本的比例

$$Precision = \frac{TP}{TP + FP}$$

**含义**:
- Precision高 → 推荐准确，少推不相关的
- 推荐系统非常关注Precision
- 用户体验：少误推

**在推荐中**:
- CTR预估的关键指标
- Top-K推荐的精确度
- 用户满意度直接相关

### 2.3 召回率（Recall）

**定义**: 所有真正样本中，被正确预测为正样本的比例

$$Recall = \frac{TP}{TP + FN}$$

**含义**:
- Recall高 → 覆盖面广，少漏推
- 推荐系统关注Recall
- 内容曝光广度

**在推荐中**:
- 召回层核心指标
- 用户兴趣覆盖度
- 长尾物品推荐

### 2.4 F1-Score

**定义**: Precision和Recall的调和平均

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**特点**:
- 平衡精确率和召回率
- 当两者都重要时使用
- 取值范围[0, 1]

**在推荐中**:
- 排序模型综合评估
- 需要平衡准确和覆盖
- 常用于召回+排序联合优化

### 2.5 准确率-召回率曲线（PR Curve）

**定义**: 横轴为Recall，纵轴为Precision的曲线

**特点**:
- 类别不平衡时比ROC更有信息
- 关注正样本的表现
- 曲线下方面积AUPRC

**在推荐中**:
- 样本不平衡时的评估
- CTR预估的评估可视化

---

## 3. 排序评估指标

### 3.1 Precision@K

**定义**: 前K个推荐中，准确推荐的比例

$$Precision@K = \frac{\text{Top-K中的命中数}}{K}$$

**特点**:
- 关注推荐列表的前K个
- 用户通常只看前几个
- K值选择：5, 10, 20

**在推荐中**:
- ⭐ **最常用指标之一**
- 直接反映用户体验
- 业务K值：首页5个，列表10个

### 3.2 Recall@K

**定义**: 前K个推荐中，命中的相关物品占用户所有相关物品的比例

$$Recall@K = \frac{\text{Top-K中的命中数}}{\text{用户所有相关物品数}}$$

**特点**:
- 评估推荐覆盖度
- 需要知道用户所有相关物品
- 常用K值：10, 20, 50

**在推荐中**:
- 召回层关键指标
- 用户兴趣覆盖评估
- 长尾推荐效果

### 3.3 平均精度均值（MAP, Mean Average Precision）

**定义**: 对每个用户计算AP，然后求平均

$$AP = \frac{1}{|R|}\sum_{k=1}^{|R|} Precision@k \times rel(k)$$

$$MAP = \frac{1}{N}\sum_{u=1}^{N} AP_u$$

**其中**:
- |R|: 用户相关物品数
- rel(k): 第k个位置是否相关
- N: 用户数

**特点**:
- 考虑排序质量
- 排在前面的命中权重更大
- 排序模型核心指标

**在推荐中**:
- ⭐ **排序评估黄金标准**
- Netflix Prize评估指标
- 全局排序质量评估

### 3.4 归一化折损累计增益（NDCG, Normalized DCG）

**定义**: 考虑排序位置权重的折损增益

$$DCG@K = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

**其中**:
- rel_i: 第i个位置的相关性（0/1或分值）
- IDCG: 理想排序的DCG
- log: 位置衰减函数

**特点**:
- 考虑位置相关性（排在前更重要）
- 支持多级相关性评分
- 归一化到[0, 1]

**在推荐中**:
- ⭐ **最常用排序指标**
- 搜索推荐系统标准
- 评估排序位置敏感度

### 3.5 平均倒数排名（MRR, Mean Reciprocal Rank）

**定义**: 第一个相关物品的排名倒数的均值

$$MRR = \frac{1}{N}\sum_{u=1}^{N} \frac{1}{rank_u}$$

**其中**:
- rank_u: 用户u第一个相关物品的位置
- 如果没有相关物品，rank_u = ∞，该项为0

**特点**:
- 关注第一个命中位置
- 用户通常只看前几个
- 对排序敏感

**在推荐中**:
- 用户首屏效果评估
- 用户满意度快速反馈
- 短视频/短视频推荐常用

### 3.6 AUC（Area Under ROC Curve）

**定义**: ROC曲线下方面积

**ROC曲线**: TPR vs FPR
- TPR (True Positive Rate) = Recall
- FPR (False Positive Rate) = FP / (FP + TN)

**含义**:
- AUC = 0.5: 随机猜测
- AUC = 1.0: 完美分类
- AUC范围: [0.5, 1.0]

**特点**:
- 不受阈值影响
- 类别平衡时有效
- 衡量排序质量

**在推荐中**:
- CTR预估最常用指标
- 不需要设定阈值
- 离线评估标准

---

## 4. 指标选择指南

### 4.1 推荐各阶段指标选择

| 阶段 | 主要指标 | 次要指标 | 关注点 |
|-------|---------|---------|--------|
| **召回层** | Recall@K | Precision@K, 覆盖率 | 用户兴趣覆盖 |
| **粗排层** | Precision@K | Recall@K, 延迟 | 候选集质量 |
| **精排层** | NDCG@K, MAP | AUC, Precision@K | 排序质量 |
| **重排层** | 多样性, 惊喜度 | NDCG, 时长 | 用户体验优化 |
| **在线实验** | CTR, CVR, GMV | 时长, 人均VV | 业务指标 |

### 4.2 回归 vs 分类 vs 排序指标

| 任务类型 | 适用指标 |
|---------|---------|
| **评分预测** | RMSE, MAE, R² |
| **点击预估（CTR）** | AUC, LogLoss, Precision |
| **转化预估（CVR）** | AUC, Precision, Recall |
| **时长预估** | RMSE, MAE, 相对误差 |
| **排序推荐** | NDCG@K, MAP, MRR, Recall@K |
| **召回** | Recall@K, 覆盖率, 多样性 |

### 4.3 指标冲突与权衡

**Precision vs Recall**:
- 提高Precision → 减少误推，可能降低覆盖
- 提高Recall → 扩大覆盖，可能降低精确
- **权衡**: F1-Score, Precision@K/Recall@K双指标

**准确率 vs 多样性**:
- 纯准确率推荐可能热门商品堆砌
- 多样性提升长尾曝光但可能降低点击
- **权衡**: MMR, 多样性加权

**AUC vs LogLoss**:
- AUC反映排序质量
- LogLoss反映预测置信度
- **权衡**: 都关注，但不同视角

### 4.4 离线指标 vs 在线指标

**离线指标**:
- AUC, NDCG, MAP, Recall@K
- 快速评估，A/B测试前验证
- 未必对应在线效果

**在线指标**:
- CTR, CVR, GMV, 时长
- 真实业务效果
- 最终衡量标准

**相关性分析**:
- AUC↑ → CTR通常↑
- NDCG↑ → 时长通常↑
- 但需A/B测试最终验证

---

## 5. 实战代码示例

### 5.1 回归指标示例

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 真实值和预测值
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

# 计算各指标
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

### 5.2 分类指标示例

```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

# 真实标签和预测概率
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
y_proba = [0.9, 0.1, 0.8, 0.3, 0.2, 0.7, 0.6, 0.1, 0.2, 0.8]

# 计算各指标
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
```

### 5.3 排序指标示例

```python
import numpy as np

def compute_ndcg(relevance_list, k=10):
    """
    计算NDCG@K
    :param relevance_list: 相关性列表，如[1, 0, 1, 1, 0]
    :param k: Top-K
    :return: NDCG@K
    """
    relevance = np.array(relevance_list[:k])
    
    # DCG
    dcg = np.sum((2**relevance - 1) / np.log2(np.arange(2, len(relevance) + 2)))
    
    # IDCG (理想排序：按相关性降序)
    ideal_relevance = np.sort(relevance)[::-1][:k]
    idcg = np.sum((2**ideal_relevance - 1) / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def compute_map(user_relevances):
    """
    计算MAP（多用户平均）
    :param user_relevances: 每个用户的相关性列表
    :return: MAP
    """
    ap_list = []
    
    for rel_list in user_relevances:
        if sum(rel_list) == 0:
            continue  # 无相关物品
        
        precision_sum = 0.0
        num_hits = 0
        for i, rel in enumerate(rel_list):
            if rel > 0:
                num_hits += 1
                precision_sum += num_hits / (i + 1)
        
        if num_hits > 0:
            ap_list.append(precision_sum / num_hits)
    
    return np.mean(ap_list) if len(ap_list) > 0 else 0.0

# 示例
user_relevances = [
    [1, 0, 1, 0, 0],      # 用户1
    [0, 0, 0, 1, 1],        # 用户2
    [1, 1, 1, 0, 0],        # 用户3
]

print(f"NDCG@10 for user1: {compute_ndcg(user_relevances[0], k=10):.4f}")
print(f"MAP across all users: {compute_map(user_relevances):.4f}")
```

### 5.4 综合评估报告

```python
from sklearn.metrics import classification_report
import pandas as pd

# 生成评估报告
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]

report = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(report).T

print("=== 分类评估报告 ===")
print(df[['precision', 'recall', 'f1-score', 'support']])
```

---

## 6. 常见问题与易错点

### 6.1 指标误用

**❌ CTR任务用Accuracy**:
- 问题: 样本不平衡（点击率<5%）
- 正确: 用AUC, Precision, Recall

**❌ 评分预测用NDCG**:
- 问题: NDCG用于排序，评分是回归
- 正确: 用RMSE, MAE, R²

**❌ 只看离线指标，忽略在线**:
- 问题: 离线≠在线，需A/B验证
- 正确: 离线筛选 + A/B测试

### 6.2 K值选择不当

**问题**: 不同业务场景K值不同

| 场景 | 推荐K值 | 原因 |
|-------|-----------|--------|
| 首页推荐 | 5-10 | 屏幕空间有限 |
| 详情页推荐 | 10-20 | 用户滚动浏览 |
| 搜索推荐 | 20-50 | 用户有明确意图 |

### 6.3 归一化问题

**问题**: 不同模型/时间AUC不可直接比较

**解决方案**:
- 使用相对提升（+1%, +5%）
- 设定对照组基线
- 关注A/B测试增量

### 6.4 样本偏差

**问题**: 训练集分布与线上不一致

**影响**:
- 离线AUC高，线上CTR低
- 排序好但用户不感兴趣

**解决方案**:
- 离线评估采样真实用户行为
- 新模型上线做流量渐进放量
- 关注实时指标反馈

---

## 7. 学习总结

### 7.1 核心要点

1. **指标分类**: 回归（RMSE/MAE/R²）、分类（Precision/Recall/F1/AUC）、排序（NDCG/MAP/Recall@K）
2. **推荐系统重点**: NDCG@K、MAP、AUC、Precision@K、Recall@K
3. **离线+在线**: 离线筛选模型，A/B验证业务效果
4. **权衡理解**: Precision vs Recall，准确率 vs 多样性，不同阶段不同优先级

### 7.2 指标速查表

| 任务 | 首选指标 | 次选指标 |
|-----|-----------|----------|
| 评分预测 | RMSE, MAE | R² |
| 点击预估 | AUC, LogLoss | Precision |
| 排序推荐 | NDCG@K, MAP | Recall@K, MRR |
| 召回 | Recall@K | 覆盖率, 多样性 |
| 在线实验 | CTR, CVR, GMV | 时长, 人均VV |

### 7.3 面试必答

**Q1: 为什么推荐系统不用Accuracy？**
A: 样本高度不平衡，点击率通常<5%，Accuracy无意义。

**Q2: NDCG和MAP的区别？**
A: NDCG考虑位置权重和相关性等级，MAP是平均精度，都用于排序但角度不同。

**Q3: 离线AUC和线上CTR正相关吗？**
A: 通常正相关，但不是绝对，需要A/B测试最终验证业务指标。

---

## 8. 学习路径建议

```
基础指标（MSE/MAE/R²） → 分类指标（Precision/Recall/F1/AUC） → 排序指标（NDCG/MAP/MRR/Recall@K） → 业务指标（CTR/CVR/GMV）
```

**推荐学习顺序**:
1. 先学回归指标（简单直观）
2. 再学分类指标（CTR预估基础）
3. 重点学排序指标（推荐系统核心）
4. 理解离线+在线关系（实战关键）
5. 掌握多指标权衡（系统设计必备）
