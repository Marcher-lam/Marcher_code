# 评估指标

## 一、回归指标

### MSE / RMSE
- **MSE（均方误差）**：$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **RMSE（均方根误差）**：$\text{RMSE} = \sqrt{\text{MSE}}$
- 对异常值敏感，常用于回归模型评估

### MAE
- **MAE（平均绝对误差）**：$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- 对异常值不如 MSE 敏感

### R²
- **R²（决定系数）**：$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$
- 衡量模型解释的方差比例，越接近 1 越好

## 二、分类指标

### Precision / Recall / F1
- **Precision（精确率）**：$\frac{TP}{TP + FP}$
- **Recall（召回率）**：$\frac{TP}{TP + FN}$
- **F1-Score**：$\frac{2 \times P \times R}{P + R}$

### AUC / ROC
- **ROC 曲线**：以 FPR 为横轴、TPR 为纵轴
- **AUC**：ROC 曲线下面积，衡量排序能力
- **GAUC**：分组 AUC，广告系统中按用户/广告位分组计算

## 三、广告系统核心指标

### CTR / CVR
- **CTR（点击率）**：$\text{CTR} = \frac{\text{Clicks}}{\text{Impressions}}$
- **CVR（转化率）**：$\text{CVR} = \frac{\text{Conversions}}{\text{Clicks}}$
- **CTCVR**：$\text{CTCVR} = \text{pCTR} \times \text{pCVR}$

### eCPM / RPM
- **eCPM（有效千次展示收益）**：$\text{eCPM} = \text{pCTR} \times \text{Bid} \times 1000$
- **RPM（千次展示收入）**：$\text{RPM} = \frac{\text{Revenue}}{\text{Impressions}} \times 1000$

### LogLoss
- $\text{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$

### Normalized Gini
- 用于 pLTV 模型评估
- Normalized Gini = 2 × AUC - 1

### 校准度（Calibration）
- 预估值的分桶校准：将预估值分桶，比较每个桶内的实际点击率与预估均值
- 常用方法：保序回归（Isotonic Regression）
