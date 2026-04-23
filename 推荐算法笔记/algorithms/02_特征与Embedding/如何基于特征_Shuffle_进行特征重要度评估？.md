# 面试题：如何基于特征 Shuffle 进行特征重要度评估？

# 面试题：如何基于特征 Shuffle 进行特征重要度评估？

基于特征 Shuffle 的特征重要度评估算法（Permutation Importance）是一种模型无关的方法（该方法目前在大厂里用的比较多），通过破坏特征与目标变量的关联性来评估其对模型预测的影响。

# 一、算法步骤

$\textcircled{1}$ 训练基准模型

使用原始数据集（包含特征 $X _ { 1 } , X _ { 2 } , \ldots , X _ { n }$ 和目标变量 ）训练模型，并在测试集上评估基线性能指标（如 AUC，记为 AUC_origin）。

$\textcircled{2}$ 特征逐个打乱

特征扰动：对每个特征 $X _ { i }$ 独立进行随机打乱（Shuffle），破坏其与 的关联性，其他特征不变。

$\textcircled{3}$ 计算性能指标变化

Shuffle 特征后，评估模型在测试集上的 AUC（记为 AUC_shuffle）；

重要性得分【AUC_lift $=$ AUC_shuffle - AUC_origin】，一般来说，如果特征越重要，AUC_lift 负的更多。

$\textcircled{4}$ 排序与筛选

按重要性得分对特征排序，筛选出对模型影响显著的特征。

# 二、优点

- 模型无关性：适用于任何模型（如神经网络、树模型等），无需依赖模型内部机制（如梯度或分裂次数）。
- 直观可解释：通过性能变化的绝对值量化重要性，结果易于理解（如"打乱年龄特征后 AUC 下降 $10\%$"）。
- 捕捉非线性：即使特征与目标 呈非线性或存在交互作用，也能通过 AUC 变化间接反映其重要度。

# 三、缺点

- 计算成本高：需对每个特征多次打乱并重新评估模型，高维数据场景下效率低。
- 高估冗余特征：若多个特征高度相关，单独打乱某一特征时，其他相关特征可能补偿其作用，导致特征重要性被低估。
- 破坏数据分布：打乱特征可能生成非现实数据（如将年龄替换为不合理的极端值），影响模型评估的可靠性。

# 四、完整 Python 代码实现

## 4.1 基础版本

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

def permutation_importance_custom(model, X_test, y_test, metric_func, n_repeats=10, n_jobs=-1, random_state=42):
    rng = np.random.RandomState(random_state)
    baseline_score = metric_func(y_test, model.predict_proba(X_test)[:, 1])
    feature_names = X_test.columns if isinstance(X_test, pd.DataFrame) else [f"feat_{i}" for i in range(X_test.shape[1])]
    X_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test.copy()

    def _compute_single_feature(col_idx):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X_arr.copy()
            X_permuted[:, col_idx] = rng.permutation(X_permuted[:, col_idx])
            permuted_score = metric_func(y_test, model.predict_proba(X_permuted)[:, 1])
            scores.append(permuted_score)
        scores = np.array(scores)
        importance = baseline_score - scores.mean()
        std = scores.std()
        return importance, std

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_feature)(i) for i in range(X_arr.shape[1])
    )
    importances = np.array([r[0] for r in results])
    stds = np.array([r[1] for r in results])
    result_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "std": stds,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return result_df, baseline_score

X, y = make_classification(n_samples=5000, n_features=20, n_informative=8,
                           n_redundant=5, random_state=42)
feature_names = [f"feature_{i}" for i in range(20)]
X = pd.DataFrame(X, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
result_df, baseline = permutation_importance_custom(
    model, X_test, y_test, roc_auc_score, n_repeats=10, n_jobs=-1
)
print(f"基线 AUC: {baseline:.4f}")
print(f"\n特征重要度排序（Top 10）:")
print(result_df.head(10).to_string(index=False))
```

## 4.2 分组 Shuffle 版本（处理高度相关特征）

```python
def grouped_permutation_importance(model, X_test, y_test, metric_func, feature_groups, n_repeats=10):
    baseline_score = metric_func(y_test, model.predict_proba(X_test)[:, 1])
    rng = np.random.RandomState(42)
    X_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test.copy()
    col_map = {name: i for i, name in enumerate(
        X_test.columns if isinstance(X_test, pd.DataFrame) else [f"feat_{i}" for i in range(X_arr.shape[1])]
    )}
    results = []
    for group_name, features in feature_groups.items():
        col_indices = [col_map[f] for f in features]
        scores = []
        for _ in range(n_repeats):
            X_permuted = X_arr.copy()
            perm = rng.permutation(X_arr.shape[0])
            for idx in col_indices:
                X_permuted[:, idx] = X_arr[perm, idx]
            score = metric_func(y_test, model.predict_proba(X_permuted)[:, 1])
            scores.append(score)
        importance = baseline_score - np.mean(scores)
        results.append({"group": group_name, "features": features,
                        "importance": importance, "std": np.std(scores)})
    return pd.DataFrame(results).sort_values("importance", ascending=False)
```

## 4.3 可视化

```python
def plot_permutation_importance(result_df, top_n=15):
    plot_df = result_df.head(top_n).copy()
    print("\n特征重要度可视化（文本版）:")
    max_imp = plot_df["importance"].max()
    for _, row in plot_df.iterrows():
        bar_len = int(row["importance"] / max_imp * 40) if max_imp > 0 else 0
        bar = "█" * bar_len
        print(f"  {row['feature']:<20s} | {bar} {row['importance']:.4f} ± {row['std']:.4f}")

plot_permutation_importance(result_df)
```

# 五、与其他特征重要度方法对比

## 5.1 与 SHAP 对比

| 维度 | Permutation Importance | SHAP |
|------|----------------------|------|
| 计算方式 | 打乱特征看性能变化 | 基于 Shapley 值分解 |
| 计算速度 | 较快 | 较慢（尤其是 Kernel SHAP） |
| 可解释粒度 | 全局特征级别 | 全局 + 局部样本级别 |
| 交互效应 | 无法单独量化 | 可通过 SHAP 交互值量化 |
| 理论保证 | 经验性方法 | 满足一致性等公理 |

```python
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
    print("SHAP 未安装，跳过 SHAP 对比")

if shap_available:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])
    if isinstance(shap_values, list):
        shap_importance = np.abs(shap_values[1]).mean(axis=0)
    else:
        shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "shap_importance": shap_importance})
    shap_df = shap_df.sort_values("shap_importance", ascending=False).reset_index(drop=True)
    print("\nSHAP 特征重要度（Top 10）:")
    print(shap_df.head(10).to_string(index=False))
```

## 5.2 与树模型内置重要度对比

```python
tree_importance = pd.DataFrame({
    "feature": feature_names,
    "tree_importance": model.feature_importances_,
}).sort_values("tree_importance", ascending=False)

comparison = result_df[["feature", "importance"]].merge(
    tree_importance[["feature", "tree_importance"]], on="feature"
)
comparison["rank_permutation"] = comparison["importance"].rank(ascending=False).astype(int)
comparison["rank_tree"] = comparison["tree_importance"].rank(ascending=False).astype(int)
comparison["rank_diff"] = abs(comparison["rank_permutation"] - comparison["rank_tree"])
print("\nPermutation vs 树模型内置重要度对比:")
print(comparison.sort_values("importance", ascending=False).head(10).to_string(index=False))
```

# 六、稳定性分析

特征重要度的稳定性直接影响结论的可靠性。以下代码评估重要度排名的稳定性：

```python
from sklearn.model_selection import KFold

def stability_analysis(X, y, n_splits=5, n_repeats=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_ranks = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        fold_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        fold_model.fit(X_tr, y_tr)
        result, _ = permutation_importance_custom(
            fold_model, X_te, y_te, roc_auc_score, n_repeats=n_repeats, n_jobs=-1
        )
        rank_dict = {row["feature"]: rank for rank, row in result.iterrows()}
        all_ranks.append(rank_dict)
    rank_df = pd.DataFrame(all_ranks).T
    rank_df["mean_rank"] = rank_df.mean(axis=1)
    rank_df["std_rank"] = rank_df.std(axis=1)
    rank_df = rank_df.sort_values("mean_rank")
    print("特征重要度排名稳定性分析:")
    print(rank_df[["mean_rank", "std_rank"]].head(10).to_string())
    return rank_df

rank_df = stability_analysis(X, y)
```

# 七、并行计算优化

## 7.1 并行策略

```python
from joblib import Parallel, delayed
import time

def benchmark_parallel(model, X_test, y_test, n_repeats=30):
    start = time.time()
    result_seq, _ = permutation_importance_custom(
        model, X_test, y_test, roc_auc_score, n_repeats=n_repeats, n_jobs=1
    )
    time_seq = time.time() - start

    start = time.time()
    result_par, _ = permutation_importance_custom(
        model, X_test, y_test, roc_auc_score, n_repeats=n_repeats, n_jobs=-1
    )
    time_par = time.time() - start

    print(f"串行耗时: {time_seq:.2f}s")
    print(f"并行耗时: {time_par:.2f}s")
    print(f"加速比: {time_seq / time_par:.2f}x")

benchmark_parallel(model, X_test, y_test)
```

# 八、改进方法

- 并行化计算：利用多线程或分布式计算加速特征打乱过程。
- 特征分组打乱：对高度相关的特征组进行联合打乱，避免低估重要性。
- 稳定性验证：通过交叉验证多次运行算法，取重要性得分的均值以降低随机性影响。

# 九、常见问题与易错点

1. **在训练集上计算重要度**：会导致过于乐观的估计。务必在独立的测试集或验证集上计算。
2. **只 shuffle 一次**：单次 shuffle 随机性太大，建议至少重复 10-30 次。
3. **忽略特征相关性**：高度相关特征的单独 shuffle 会互相补偿，导致两者重要性都被低估。应使用分组 shuffle。
4. **用 accuracy 做指标**：对于不平衡数据集，accuracy 不敏感，建议用 AUC 或 log loss。
5. **shuffle 破坏了行内关系**：对于有时序关系或行内依赖的数据，shuffle 可能生成不合理样本。

# 十、学习路径建议

1. 理解 Permutation Importance 的核心思想：打乱特征 → 性能下降 → 衡量重要性
2. 动手实现基础版本并理解每个参数的影响
3. 学习 SHAP 值理论，理解更精细的特征归因方法
4. 在实际项目中对比多种特征重要度方法
5. 阅读论文：Permutation Importance（Breiman 2001）、SHAP（Lundberg 2017）
