# 机器学习基础

本模块涵盖机器学习的核心概念、算法和实践。

## 📚 目录结构

```
002-machine_learning/
├── 01-introduction.ipynb         # 机器学习概论
├── 02-linear-models.ipynb        # 线性模型
├── 03-tree-models.ipynb          # 决策树与集成学习
├── 04-svm.ipynb                  # 支持向量机
├── 05-clustering.ipynb           # 聚类算法
├── 06-dimensionality-reduction.ipynb  # 降维算法
├── 07-model-evaluation.ipynb     # 模型评估
└── README.md
```

## 🎯 学习目标

### 1. 机器学习基础
- 监督学习、无监督学习、强化学习
- 过拟合与欠拟合
- 偏差-方差权衡
- 交叉验证

### 2. 经典算法
- 线性回归、逻辑回归
- 决策树、随机森林、GBDT
- 支持向量机（SVM）
- K-means、DBSCAN

### 3. 模型评估
- 准确率、精确率、召回率、F1分数
- ROC曲线和AUC
- 混淆矩阵
- 交叉验证

## 📖 核心概念

### 监督学习
```
输入特征 → [模型] → 预测输出
   ↓             ↓          ↓
  X          f(X;θ)        ŷ
                 ↓
              损失函数 L(y, ŷ)
                 ↓
              优化更新 θ
```

### 学习范式
| 类型 | 输入数据 | 输出 | 应用 |
|------|---------|------|------|
| 监督学习 | 有标签数据 | 预测标签 | 分类、回归 |
| 无监督学习 | 无标签数据 | 数据结构 | 聚类、降维 |
| 强化学习 | 环境+奖励 | 动作序列 | 游戏、机器人 |

## 🛠️ 技术栈

```python
# 经典机器学习库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
```

## 📝 学习路径

```
1. 概念理解
   ↓
2. 算法实现
   ↓
3. 案例实践
   ↓
4. 参数调优
   ↓
5. 模型部署
```

## 💡 实践项目

### 初级
- [ ] 鸢尾花分类（经典入门）
- [ ] 房价预测（波士顿数据集）
- [ ] 手写数字识别（MNIST）

### 中级
- [ ] 泰坦尼克号生存预测
- [ ] 信用风险评估
- [ ] 客户细分聚类

### 高级
- [ ] 时间序列预测
- [ ] 推荐系统
- [ ] 异常检测

## 📚 推荐资源

### 书籍
- 《统计学习方法》- 李航
- 《机器学习》- 周志华（西瓜书）
- 《Python机器学习》- Sebastian Raschka

### 课程
- Andrew Ng机器学习课程
- 林轩田《机器学习基石》

### 数据集
- UCI Machine Learning Repository
- Kaggle Datasets
- scikit-learn datasets

## 🔗 与深度学习的关系

```
机器学习                     深度学习
    ↓                           ↓
特征工程 + 简单模型    →    自动特征提取 + 深层网络
    ↓                           ↓
适合结构化数据            适合非结构化数据
```

## 💻 编码实践

### 标准流程
1. 数据加载和探索
2. 数据预处理
3. 特征工程
4. 模型选择和训练
5. 模型评估
6. 参数调优
7. 模型保存和部署

### 最佳实践
- 总是划分训练集和测试集
- 使用交叉验证评估模型
- 特征标准化很重要
- 先用简单模型建立baseline
- 记录实验结果
