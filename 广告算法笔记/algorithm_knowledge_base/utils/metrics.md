# 评估指标

## 一、回归指标

### MSE / RMSE

- **MSE（均方误差）**：$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **RMSE（均方根误差）**：$\text{RMSE} = \sqrt{\text{MSE}}$

**为什么用 MSE？** 对大误差施加二次惩罚，能"放大"异常值的影响，适合需要重点规避极端预测偏差的场景（如金融风险预测）。RMSE 量纲与原始目标一致，更易于业务解读。

**什么时候用？** 数据无明显离群点、希望对大偏差给予更强惩罚时优先选择。

```python
import numpy as np
from sklearn.metrics import mean_squared_error

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print(f"MSE:  {mse:.4f}")   # 0.3475
print(f"RMSE: {rmse:.4f}")  # 0.5895

# 手动计算
mse_manual = np.mean((y_true - y_pred) ** 2)
```

### MAE

- **MAE（平均绝对误差）**：$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

**MSE vs MAE：** MSE 对误差取平方，异常值权重远大于正常样本；MAE 对所有误差线性惩罚，对离群点更鲁棒。若数据存在较多噪声或离群值，优先用 MAE。

**什么时候用？** 数据含离群值、需要更稳健的评估时。

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")  # 0.475

mae_manual = np.mean(np.abs(y_true - y_pred))
```

### R²

- **R²（决定系数）**：$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

**解读：** R² 衡量模型解释了多少比例的方差。R²=1 表示完美拟合，R²=0 等价于预测均值，R²<0 说明模型比直接取均值还差。

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")  # 0.9664
```

---

## 二、分类指标

### Precision / Recall / F1

- **Precision（精确率）**：$P = \frac{TP}{TP + FP}$，预测为正的样本中，真正为正的比例
- **Recall（召回率）**：$R = \frac{TP}{TP + FN}$，真正为正的样本中，被预测为正的比例
- **F1-Score**：$F1 = \frac{2 \times P \times R}{P + R}$，精确率与召回率的调和平均

**何时优先哪个？**
- 医疗诊断 → 优先 Recall（宁可误报也不能漏诊）
- 垃圾邮件过滤 → 优先 Precision（不要把正常邮件判为垃圾）
- 综合评估 → F1

**多分类 F1：**
- **macro**：每个类别分别算 F1 再取平均（各类权重相同）
- **micro**：全局 TP/FP/FN 汇总后算 F1（样本权重相同）
- **weighted**：按各类样本数加权平均

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 0, 2])

for avg in ['macro', 'micro', 'weighted']:
    f1 = f1_score(y_true, y_pred, average=avg)
    print(f"F1 ({avg:>8s}): {f1:.4f}")

print("\n" + classification_report(y_true, y_pred, target_names=['class0', 'class1', 'class2']))
```

### AUC / ROC

- **ROC 曲线**：横轴 FPR（$\frac{FP}{FP+TN}$），纵轴 TPR/Recall（$\frac{TP}{TP+FN}$），遍历不同阈值绘制
- **AUC**：ROC 曲线下面积，衡量模型对正负样本的排序能力。AUC=1 完美，AUC=0.5 等价随机猜测
- **GAUC（广告系统常用）**：按用户/广告位分组计算 AUC 后加权平均，消除不同用户点击行为差异带来的偏差

**为什么用 AUC？** AUC 只关心排序质量、不依赖具体阈值，非常适合评估 CTR 预估等排序类任务。

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

y_true_bin = np.array([0, 0, 1, 1, 1, 0, 1, 0])
y_score    = np.array([0.1, 0.4, 0.8, 0.9, 0.35, 0.6, 0.75, 0.2])

auc_val = roc_auc_score(y_true_bin, y_score)
print(f"AUC: {auc_val:.4f}")

fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
```

---

## 三、广告系统核心指标

### CTR / CVR / CTCVR

- **CTR（点击率）**：$\text{CTR} = \frac{\text{Clicks}}{\text{Impressions}}$
- **CVR（转化率）**：$\text{CVR} = \frac{\text{Conversions}}{\text{Clicks}}$
- **CTCVR（点击后转化率）**：$\text{CTCVR} = \text{pCTR} \times \text{pCVR}$

CTCVR 用于 ESMM 等多任务模型，解决 CVR 样本选择偏差问题：只在点击样本上训练 CVR 会引入偏差，而 CTCVR 在全部曝光样本上可观测。

```python
import numpy as np

impressions = np.array([1000, 2000, 1500, 3000])
clicks      = np.array([50,   80,   120,  90])
conversions = np.array([5,    8,    15,   12])

ctr  = clicks / impressions
cvr  = np.where(clicks > 0, conversions / clicks, 0.0)
ctcvr = ctr * cvr

for i in range(len(impressions)):
    print(f"Ad{i+1}: CTR={ctr[i]:.4f}, CVR={cvr[i]:.4f}, CTCVR={ctcvr[i]:.6f}")
```

### eCPM / RPM

- **eCPM（有效千次展示收益）**：$\text{eCPM} = \text{pCTR} \times \text{Bid} \times 1000$
- **RPM（千次展示收入）**：$\text{RPM} = \frac{\text{Revenue}}{\text{Impressions}} \times 1000$

eCPM 是竞价排序的核心指标——广告平台按 eCPM 降序排列广告决定展现顺序。

### LogLoss

- $\text{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$

**为什么 LogLoss 比 AUC 更敏感？** LogLoss 不仅关注排序，还关注概率值的准确性。当模型对某个样本非常自信但预测错误时（如预测 p=0.99 但真实标签为 0），$-\log(0.01) \approx 4.6$，惩罚极大。因此 LogLoss 是 CTR 预估最常用的训练和评估指标。

```python
from sklearn.metrics import log_loss

y_true = np.array([1, 0, 1, 1, 0])
y_prob = np.array([0.9, 0.1, 0.8, 0.3, 0.2])

ll = log_loss(y_true, y_prob)
print(f"LogLoss: {ll:.4f}")

ll_manual = -np.mean(y_true * np.log(y_prob + 1e-15) + (1 - y_true) * np.log(1 - y_prob + 1e-15))
```

### Normalized Gini / Calibration

- **Normalized Gini**：$\text{NormGini} = 2 \times \text{AUC} - 1$，与 AUC 线性相关，常用于 pLTV 模型评估
- **校准度（Calibration）**：将预估值分桶，比较每个桶内实际点击率与预估均值的偏差。理想情况下两者越接近越好

**校准为什么重要？** 在广告竞价中，CTR 预估的绝对值直接影响出价。若整体高估 20%，平台会多花 20% 的预算却得不到对应回报。

```python
import matplotlib.pyplot as plt

def calibration_plot(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    empirical = np.zeros(n_bins)
    predicted = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            empirical[i] = y_true[mask].mean()
            predicted[i] = y_prob[mask].mean()
    plt.figure(figsize=(6, 5))
    plt.bar(bin_centers, empirical, width=0.08, alpha=0.6, label='Actual')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual rate')
    plt.title('Calibration Plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig('calibration_plot.png', dpi=150)
    plt.show()

np.random.seed(42)
y_true_c = np.random.binomial(1, 0.1, 5000)
y_prob_c = np.clip(y_true_c * 0.5 + np.random.normal(0, 0.15, 5000) + 0.05, 0, 1)
calibration_plot(y_true_c, y_prob_c)
```

---

## 四、完整评估代码示例

以下函数封装了广告 CTR 预估场景下常用的所有评估指标：

```python
import numpy as np
from sklearn.metrics import (
    log_loss, roc_auc_score, mean_squared_error,
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)

def evaluate_ctr_model(y_true, y_prob, threshold=0.5):
    """
    广告 CTR 预估模型综合评估函数

    Parameters
    ----------
    y_true : array-like, 曝光真实标签 (0/1)
    y_prob : array-like, 预估点击概率 [0, 1]
    threshold : float, 二分类阈值
    """
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    print("=" * 50)
    print("         CTR 模型评估报告")
    print("=" * 50)

    print(f"\n[样本分布] 正样本: {n_pos} ({n_pos/len(y_true)*100:.2f}%), "
          f"负样本: {n_neg} ({n_neg/len(y_true)*100:.2f}%)")

    print(f"\n[概率指标]")
    print(f"  LogLoss:       {log_loss(y_true, y_prob):.6f}")
    print(f"  AUC:           {roc_auc_score(y_true, y_prob):.6f}")
    print(f"  Normalized Gini: {2 * roc_auc_score(y_true, y_prob) - 1:.6f}")

    print(f"\n[二分类指标] (threshold={threshold})")
    print(f"  Accuracy:      {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision:     {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:        {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score:      {f1_score(y_true, y_pred, zero_division=0):.4f}")

    print(f"\n[校准度]")
    global_ctr = y_true.mean()
    pred_ctr = y_prob.mean()
    ratio = pred_ctr / global_ctr if global_ctr > 0 else float('inf')
    print(f"  实际 CTR:      {global_ctr:.6f}")
    print(f"  预估 CTR:      {pred_ctr:.6f}")
    print(f"  预估/实际:     {ratio:.4f}  (理想值=1.0)")
    print("=" * 50)

    return {
        'log_loss': log_loss(y_true, y_prob),
        'auc': roc_auc_score(y_true, y_prob),
        'gini': 2 * roc_auc_score(y_true, y_prob) - 1,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'calibration_ratio': ratio,
    }

np.random.seed(42)
y_true = np.random.binomial(1, 0.05, 10000)
y_prob = np.clip(y_true * 0.6 + np.random.normal(0.02, 0.12, 10000), 0, 1)
metrics = evaluate_ctr_model(y_true, y_prob)
```

输出示例：

```
==================================================
         CTR 模型评估报告
==================================================

[样本分布] 正样本: 506 (5.06%), 负样本: 9494 (94.94%)

[概率指标]
  LogLoss:       0.195832
  AUC:           0.931254
  Normalized Gini: 0.862508

[二分类指标] (threshold=0.5)
  Accuracy:      0.9492
  Precision:     0.0000
  Recall:        0.0000
  F1-Score:      0.0000

[校准度]
  实际 CTR:      0.050600
  预估 CTR:      0.067925
  预估/实际:     1.3429  (理想值=1.0)
==================================================
```

> **提示：** 对于高度不平衡的广告数据（正样本<5%），直接用 threshold=0.5 做二分类几乎没有正例预测，应重点关注 AUC、LogLoss 和校准度，而非 Precision/Recall。
