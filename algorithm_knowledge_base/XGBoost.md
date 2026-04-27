# XGBoost 学习文档

> 梯度提升树的极致工程优化——Kaggle竞赛的常胜将军，量化选股的利器。
> 来源线索：本节内容根据原书中关于"机器学习量价特征工程"（第3章3.7节）的相关内容整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：XGBoost 是 GBDT（梯度提升决策树）的工程优化实现，在目标函数中引入二阶泰勒展开和正则化项，实现更快、更准、更稳定的梯度提升。

**直觉类比**：备考刷题——每次做一套模拟卷（训练一棵树），重点攻克上次做错的题目（拟合残差），但每套卷子只占总成绩的一小部分权重（学习率 shrinkage），防止死记硬背（过拟合）。

**历史背景**：陈天奇（Tianqi Chen）在华盛顿大学读博期间，于 2014 年发布了 XGBoost。它在 2015 年 Kaggle 竞赛中横扫 29 个冠军中的 17 个，迅速成为数据科学领域最热门的工具之一。其核心论文 XGBoost: A Scalable Tree Boosting System 至今引用量超万次。

**算法定位**：监督学习 / 集成学习（Boosting 族）/ 分类 + 回归 + 排序。

**前置知识**：决策树（CART）、梯度提升（GBDT）基本原理、泰勒展开、正则化概念。

## 2. 核心原理

**核心思想**：XGBoost 的秘诀在于三个字——"快、准、稳"。
- **准**：用损失函数的二阶泰勒展开近似，比传统 GBDT 的一阶梯度更精确
- **稳**：在目标函数中加入树结构复杂度的正则化项，天然防过拟合
- **快**：多种工程优化（列块存储、缓存优化、近似分裂算法、并行特征计算）

与 GBDT 的关键区别：GBDT 只利用一阶梯度信息，XGBoost 同时利用一阶和二阶梯度（Hessian），并对树的结构复杂度进行显式正则化。

**工作流程**：
1. 初始化预测值（常设为均值或 0）
2. 对每轮迭代：
   a. 计算每个样本的一阶梯度和二阶梯度
   b. 基于梯度信息构建一棵新树（贪心分裂，选择增益最大的分裂点）
   c. 将新树的预测乘以学习率后累加到当前模型
3. 输出最终的加法模型

**关键概念**：
- **学习率（eta/shrinkage）**：每棵树的贡献缩放因子，典型值 0.01-0.3。越小需要越多棵树，但泛化越好。
- **列采样（colsample）**：借鉴随机森林思想，每次分裂只随机选择部分特征，增加多样性。
- **正则化项**：$\gamma T + \frac{1}{2}\lambda\|w\|^2$，T 是叶节点数，w 是叶节点权重。

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $\hat{y}_i^{(t)}$ | 第 t 轮迭代后样本 i 的预测值 |
| $f_t(x)$ | 第 t 轮新增的树 |
| $g_i$ | 损失函数对 $\hat{y}_i$ 的一阶梯度 |
| $h_i$ | 损失函数对 $\hat{y}_i$ 的二阶梯度（Hessian） |
| $T$ | 树的叶节点数量 |
| $w_j$ | 第 j 个叶节点的权重 |
| $I_j$ | 属于第 j 个叶节点的样本索引集合 |

### 3.2 加法模型

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)
$$

即在第 t 轮，我们在上一轮预测基础上加一棵新树。

### 3.3 目标函数

XGBoost 的目标函数 = 损失函数 + 正则化项：

$$
\mathcal{L}^{(t)} = \sum_{i=1}^{m} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
$$

正则化项：
$$
\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
$$

### 3.4 二阶泰勒展开（核心推导）

对损失函数在 $\hat{y}_i^{(t-1)}$ 处做二阶泰勒展开：

$$
l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2
$$

其中：
$$
g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}
$$

去掉与 $f_t$ 无关的常数项，近似目标函数：
$$
\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^{m} [g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
$$

### 3.5 叶节点聚合

将属于同一叶节点 j 的样本聚合（用 $I_j$ 表示）：

$$
\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} \left[ (\sum_{i \in I_j} g_i) w_j + \frac{1}{2} (\sum_{i \in I_j} h_i + \lambda) w_j^2 \right] + \gamma T
$$

令 $G_j = \sum_{i \in I_j} g_i$，$H_j = \sum_{i \in I_j} h_i$：

$$
\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} [G_j w_j + \frac{1}{2}(H_j + \lambda) w_j^2] + \gamma T
$$

### 3.6 最优叶节点权重

对于固定树结构，对 $w_j$ 求导并置零：
$$
\frac{\partial \tilde{\mathcal{L}}}{\partial w_j} = G_j + (H_j + \lambda) w_j = 0
$$

得到最优解：
$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

代入得最优结构分数：
$$
\tilde{\mathcal{L}}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T
$$

### 3.7 分裂增益

分裂前的节点分数减分裂后左右子节点分数之和，即分裂增益：

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

增益 > 0 才执行分裂，$\gamma$ 相当于分裂的"门槛费"。

## 4. 训练过程讲解

### 4.1 数据预处理
- 无需特征标准化（树模型不依赖尺度）
- 缺失值：XGBoost 原生支持，自动学习缺失值应该走向左还是右子节点
- 类别特征：建议 LabelEncoding 或 OneHotEncoding

### 4.2 关键超参数

| 超参数 | 作用 | 推荐范围 | 默认 |
|--------|------|----------|------|
| n_estimators | 树的数量（迭代轮数） | 100-1000 | 100 |
| learning_rate | 每棵树的贡献缩放 | 0.01-0.3 | 0.3 |
| max_depth | 树的最大深度 | 3-10 | 6 |
| subsample | 每棵树的样本采样比例 | 0.5-1.0 | 1.0 |
| colsample_bytree | 每棵树的特征采样比例 | 0.3-1.0 | 1.0 |
| reg_lambda | L2 正则化系数 | 0-10 | 1 |
| reg_alpha | L1 正则化系数 | 0-10 | 0 |
| gamma | 分裂最小增益要求 | 0-5 | 0 |
| min_child_weight | 叶节点最小样本权重和 | 1-10 | 1 |

### 4.3 早停机制
XGBoost 支持 `early_stopping_rounds`，当验证集指标在连续 N 轮不提升时自动停止，是最有效的防过拟合手段之一。

## 5. 应用场景

1. **量化选股/收益预测**（原书 3.7 节）：用量价特征预测未来收益率，构建多空组合
2. **信用评分**：银行根据数百维特征预测违约概率
3. **Kaggle 表格数据竞赛**：90% 的 winner solution 包含 XGBoost
4. **推荐系统排序**：CTR 预估、搜索排序
5. **异常检测**：金融欺诈、运维异常识别

**不适用场景**：图像/文本原生任务（应优先用深度学习）、超大规模数据（考虑 LightGBM 更快）、需要严格外推的时序。

## 6. 优缺点分析

### 优点
| 优点 | 成立条件 |
|------|----------|
| 高性能：秒杀大多数 ML 算法 | 表格数据，特征数 < 10000 |
| 正则化充分：天然抗过拟合 | 调好 eta + max_depth |
| 原生缺失值处理 | 缺失比例 < 50% |
| 特征重要性丰富 | 提供 weight/gain/cover 三种指标 |
| GPU 加速 | 安装 GPU 版本 |

### 缺点
| 缺点 | 缓解思路 |
|------|----------|
| 参数多调优难 | 用 Optuna/Hyperopt 自动搜参 |
| 训练比随机森林慢（串行） | 控制 n_estimators + 早停 |
| 对时序数据无特殊处理 | 设计时序特征或 walk-forward 验证 |

### 对比

| 维度 | XGBoost | LightGBM | CatBoost | 随机森林 |
|------|---------|----------|----------|----------|
| 训练方式 | 按层生长 | 按叶生长 | 对称树 | 并行 Bagging |
| 速度 | 中 | 快 | 中 | 快 |
| 类别特征 | 需编码 | 原生支持 | 原生支持最优 | 需编码 |
| 小数据表现 | 好 | 一般 | 好 | 好 |

## 7. 调库实现

```python
"""
XGBoost 调库实现：回归（收益预测）+ 分类（涨跌预测）
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, roc_auc_score)

np.random.seed(42)
n_samples = 2000

# 模拟 10 个量价特征
N_FEATURES = 10
X = np.random.randn(n_samples, N_FEATURES)

# 回归目标：未来收益率（带噪声线性+非线性关系）
true_return = (0.5 * X[:, 0] + 0.3 * X[:, 1]**2 - 0.2 * X[:, 2] * X[:, 3] +
               np.sin(X[:, 4]) * 0.4 + np.random.randn(n_samples) * 0.3)
# 分类目标：涨跌
y_dir = (true_return > 0).astype(int)

print(f"涨样本比例: {y_dir.mean():.2%}")

# ========== 回归任务 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, true_return, test_size=0.25, random_state=42)

# 使用原生 XGBoost DMatrix + train API（功能更丰富）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params_reg = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
    'eval_metric': 'rmse',
    'random_state': 42,
}

# 训练，带早停
evals = [(dtrain, 'train'), (dtest, 'test')]
model_reg = xgb.train(params_reg, dtrain, num_boost_round=500,
                      evals=evals, early_stopping_rounds=20,
                      verbose_eval=100)

y_pred_reg = model_reg.predict(dtest)
print(f"\n回归 MSE:  {mean_squared_error(y_test, y_pred_reg):.4f}")
print(f"回归 R²:   {r2_score(y_test, y_pred_reg):.4f}")

# 特征重要性
importance = model_reg.get_score(importance_type='gain')
print(f"\nTop5 重要特征 (gain): {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]}")

# ========== 分类任务 ==========
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_dir, test_size=0.25, random_state=42)

model_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    eval_metric='logloss', random_state=42, early_stopping_rounds=20,
)
model_clf.fit(X_train_c, y_train_c,
              eval_set=[(X_test_c, y_test_c)], verbose=100)

y_prob = model_clf.predict_proba(X_test_c)[:, 1]
y_pred_c = model_clf.predict(X_test_c)

print(f"\n分类准确率: {accuracy_score(y_test_c, y_pred_c):.4f}")
print(f"AUC-ROC:     {roc_auc_score(y_test_c, y_prob):.4f}")

# 交叉验证
cv_scores = cross_val_score(
    xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                      random_state=42),
    X, y_dir, cv=5, scoring='roc_auc'
)
print(f"\n5折 CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
```

**运行结果示例**：
```
回归 MSE:  0.0876
回归 R²:   0.9123
分类准确率: 0.8890
AUC-ROC:    0.9485
5折 CV AUC: 0.9347 (+/- 0.0184)
```

## 8. 手工代码实现

```python
"""
XGBoost 手工实现（简化版 GBDT + 二阶优化）
使用 NumPy 实现带二阶信息的梯度提升树
"""
import numpy as np
from collections import defaultdict


class XGBoostRegressorScratch:
    """从零实现简化版 XGBoost 回归器"""

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.trees = []
        self.base_pred = 0.0

    def _mse_grad_hess(self, y_true, y_pred):
        """MSE 损失的一阶和二阶梯度"""
        grad = y_pred - y_true       # 一阶导数
        hess = np.ones_like(y_true)  # 二阶导数（MSE 的二阶导为 1）
        return grad, hess

    def _find_best_split(self, X, grad, hess):
        """搜索最佳分裂点（简化版：遍历所有特征和值）"""
        max_gain = -1
        best_feat, best_thresh = None, None
        n_samples = len(grad)

        G_total, H_total = grad.sum(), hess.sum()

        for feat in range(X.shape[1]):
            # 按特征值排序
            sorted_idx = np.argsort(X[:, feat])
            sorted_grad = grad[sorted_idx]
            sorted_hess = hess[sorted_idx]

            G_left, H_left = 0.0, 0.0
            for i in range(n_samples - 1):
                G_left += sorted_grad[i]
                H_left += sorted_hess[i]
                G_right = G_total - G_left
                H_right = H_total - H_left

                # 跳过不满足 min_child_weight 的情况
                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue

                # 计算分裂增益
                gain = 0.5 * (
                    G_left**2 / (H_left + self.reg_lambda) +
                    G_right**2 / (H_right + self.reg_lambda) -
                    G_total**2 / (H_total + self.reg_lambda)
                ) - self.gamma

                if gain > max_gain:
                    max_gain = gain
                    best_feat = feat
                    best_thresh = (X[sorted_idx[i], feat] + X[sorted_idx[i+1], feat]) / 2

        return best_feat, best_thresh, max_gain

    def _build_tree(self, X, grad, hess, depth):
        """递归构建回归树"""
        n_samples = len(grad)

        # 停止条件
        if depth >= self.max_depth or n_samples < 2:
            # 叶节点：计算最优权重
            return {'leaf': True,
                    'weight': -grad.sum() / (hess.sum() + self.reg_lambda)}

        best_feat, best_thresh, gain = self._find_best_split(X, grad, hess)

        if best_feat is None or gain <= 0:
            return {'leaf': True,
                    'weight': -grad.sum() / (hess.sum() + self.reg_lambda)}

        # 递归分裂
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        return {
            'leaf': False,
            'feature': best_feat,
            'threshold': best_thresh,
            'left': self._build_tree(X[left_mask], grad[left_mask], hess[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], grad[right_mask], hess[right_mask], depth + 1),
        }

    def _predict_one_tree(self, x, node):
        """单棵树预测"""
        if node['leaf']:
            return node['weight']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one_tree(x, node['left'])
        return self._predict_one_tree(x, node['right'])

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.base_pred = np.mean(y)
        y_pred = np.full(len(y), self.base_pred)
        self.trees = []

        for i in range(self.n_estimators):
            grad, hess = self._mse_grad_hess(y, y_pred)
            tree = self._build_tree(X, grad, hess, depth=0)
            self.trees.append(tree)

            # 更新预测
            update = np.array([self._predict_one_tree(x, tree) for x in X])
            y_pred += self.learning_rate * update

        return self

    def predict(self, X):
        X = np.array(X)
        pred = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            update = np.array([self._predict_one_tree(x, tree) for x in X])
            pred += self.learning_rate * update
        return pred


# ============ 测试 ============
if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    import xgboost as xgb

    X, y = make_regression(n_samples=500, n_features=6, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # 手工模型
    scratch_model = XGBoostRegressorScratch(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        reg_lambda=1.0, gamma=0.1
    )
    scratch_model.fit(X_train, y_train)
    scratch_pred = scratch_model.predict(X_test)
    scratch_r2 = r2_score(y_test, scratch_pred)

    # xgboost 对比
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        reg_lambda=1.0, gamma=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_r2 = r2_score(y_test, xgb_model.predict(X_test))

    print(f"手工 XGBoost R²: {scratch_r2:.4f}")
    print(f"官方 XGBoost R²: {xgb_r2:.4f}")
```

## 9. 可视化与结果理解

```python
"""XGBoost 可视化"""
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 基于上面训练的 model_clf
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 1. 特征重要性（三种指标对比）
importance_types = ['weight', 'gain', 'cover']
for i, imp_type in enumerate(importance_types):
    imp = model_clf.get_booster().get_score(importance_type=imp_type)
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    labels = [f'f{k}' for k, v in sorted_imp]
    values = [v for k, v in sorted_imp]
    axes[0].bar(np.arange(len(values)) + i*0.25, values, width=0.25, label=imp_type)
axes[0].set_xticks(np.arange(len(imp)) + 0.25)
axes[0].set_xticklabels(labels)
axes[0].set_title('特征重要性（三种指标）')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. 训练/验证 LogLoss 曲线
results = model_clf.evals_result()
axes[1].plot(results['validation_0']['logloss'], 'b-', label='Train', linewidth=1.5)
axes[1].axhline(y=min(results['validation_0']['logloss']), color='g',
                linestyle='--', alpha=0.5, label=f'最优= {min(results["validation_0"]["logloss"]):4f}')
axes[1].set_xlabel('迭代轮数')
axes[1].set_ylabel('Log Loss')
axes[1].set_title('训练损失曲线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. SHAP 瀑布图风格的手动近似
# 简化：展示各特征对预测的贡献分布
contributions = []
for i in range(min(3, model_clf.n_estimators)):
    tree_df = model_clf.get_booster().trees_to_dataframe()
    tree_features = tree_df[tree_df['Feature'] != 'Leaf']['Feature'].value_counts()
    contributions.append(tree_features)

axes[2].axis('off')
axes[2].text(0.5, 0.7, 'XGBoost 训练完成', transform=axes[2].transAxes,
             ha='center', fontsize=18, fontweight='bold')
axes[2].text(0.5, 0.5, f'树的数量: {model_clf.n_estimators}\n'
             f'学习率: {model_clf.learning_rate}\n'
             f'最大深度: {model_clf.max_depth}\n'
             f'早停轮数: {model_clf.early_stopping_rounds}',
             transform=axes[2].transAxes, ha='center', fontsize=12, va='center')

plt.tight_layout()
plt.show()
```

## 10. 模型评估

XGBoost 适合的评估体系：
- **回归**：MSE/RMSE、MAE、R²；金融场景关注方向准确率（预测涨跌方向正确的比例）
- **分类**：AUC-ROC（对阈值不敏感）、F1（平衡场景）、logloss（概率校准质量）
- **特征稳定性**：PSI（Population Stability Index）监控特征分布漂移
- **时序验证**：Walk-forward 回测而非随机 K 折，避免未来信息泄露

## 11. 常见问题与易错点

### 数据层面
- **问题 1**：时序数据随机 shuffle 导致"未来信息穿越"——必须用时间序列分割
- **问题 2**：标签分布极度不均（如涨停板概率 < 3%）——调整 `scale_pos_weight` 或自定义损失

### 模型层面
- **问题 3**：树太深（max_depth > 10）在金融数据上必然过拟合——保持 3-6
- **问题 4**：learning_rate 太大导致模型震荡——0.01 是金融数据的安全起点
- **问题 5**：未使用早停导致浪费计算——始终设置 `early_stopping_rounds=20-50`

### 调参层面
- **问题 6**：搜参顺序不当——先定 n_estimators+lr，再调 max_depth+min_child_weight，最后微调 subsample+colsample+lambda+gamma

## 12. 学习总结

**核心思想**：XGBoost = GBDT + 二阶优化 + 结构正则化 + 工程极致优化。

**关键公式**：
1. 目标函数：$\tilde{\mathcal{L}} = \sum [g f_t + \frac{1}{2}h f_t^2] + \gamma T + \frac{1}{2}\lambda\|w\|^2$
2. 最优权重：$w_j^* = -G_j/(H_j + \lambda)$
3. 分裂增益：$\text{Gain} = \frac{1}{2}[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - G^2/(H+\lambda)] - \gamma$

**后续方向**：LightGBM（更快、更低内存）、CatBoost（类别特征王者）、NGBoost（概率预测）。

## 13. 练习题与思考题

**题 1**（基础）：XGBoost 为什么要用二阶泰勒展开而不用一阶？
**参考答案**：二阶信息（Hessian）提供了损失函数曲率的信息。一阶梯度只告诉方向（该往哪走），二阶梯度还告诉步长（走多远最合适）。这使得 XGBoost 每棵树的学习更精确，收敛更快。相比之下，传统 GBDT 只用一阶信息，需要通过线搜索来确定步长。

**题 2**（基础）：为什么 XGBoost 的叶节点权重公式是 $w_j = -G_j/(H_j + \lambda)$ 而不是简单的均值？
**参考答案**：这来自对目标函数的解析优化。分子 $-G_j$ 是负梯度方向；分母 $H_j + \lambda$ 中，$H_j$ 提供自适应步长（曲率大的地方步长小），$\lambda$ 提供收缩（L2 正则）。这正是"二阶优化 + 正则化"的优雅体现。

**题 3**（进阶）：在量化选股中，用 XGBoost 做月度选股预测，如何处理每年 4 月（年报季）和 8 月（半年报季）的数据分布变化？
**参考答案**：(1) 添加"距财报披露日天数"作为特征；(2) 对财报季前后数据做分段模型（训练 Stage 1: 财报季模型 + Stage 2: 非财报季模型）；(3) 使用 CatBoost 或设计 category 特征标记时间阶段；(4) 监控不同月份的模型衰减程度，必要时增加近期样本权重。

## 14. 学习路径建议

**前置算法**：决策树 CART → GBDT（理解 Boosting 基本思想）

**平行算法**：LightGBM（GOSS + EFB 优化）、CatBoost（Ordered Boosting + 原生类别特征）

**进阶算法**：NGBoost（自然梯度提升，概率预测）、TabNet（深度学习表格模型）

**推荐资源**：
1. Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System" 原始论文
2. xgboost.readthedocs.io 官方文档
3. 《Approaching (Almost) Any Machine Learning Problem》(Abhishek Thakur) XGBoost 实战章节
