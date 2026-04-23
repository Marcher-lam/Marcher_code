# 推荐系统 AutoML 学习文档

## 1. AutoML 概述

### 1.1 什么是 AutoML？

```
AutoML (Automated Machine Learning):

- 自动化机器学习流程
- 减少人工干预
- 提高效率和质量

推荐系统 AutoML 应用:
1. 自动特征工程
2. 超参数优化 (HPO)
3. 神经架构搜索 (NAS)
4. 模型选择
5. AutoML 端到端
```

### 1.2 推荐系统 AutoML 流程

```python
"""
推荐系统 AutoML 流程:

数据 → 自动特征工程 → 模型搜索 → 超参数优化 → 模型评估 → 部署
         ↓                ↓              ↓
    特征生成/选择    架构搜索       贝叶斯优化
    特征交叉        模型选择       Hyperband
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
import itertools
```

## 2. 自动特征工程

### 2.1 自动特征生成

```python
class AutoFeatureGenerator:
    """
    自动特征生成器
    """

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.generated_features = []

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        自动生成特征
        """
        result = df.copy()

        # 1. 数值特征变换
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            # 对数变换
            result[f'{col}_log'] = np.log1p(df[col])

            # 平方根变换
            result[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))

            # 二值化
            if df[col].nunique() > 2:
                median = df[col].median()
                result[f'{col}_bin'] = (df[col] > median).astype(int)

        # 2. 分类特征编码
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in cat_cols:
            # 频率编码
            freq = df[col].value_counts() / len(df)
            result[f'{col}_freq'] = df[col].map(freq)

            # 目标编码 (需要目标变量)
            # 在 fit_transform 中处理

        # 3. 时间特征
        time_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        for col in time_cols:
            result[f'{col}_hour'] = df[col].dt.hour
            result[f'{col}_dayofweek'] = df[col].dt.dayofweek
            result[f'{col}_month'] = df[col].dt.month

        # 4. 交叉特征 (选择性生成)
        cross_pairs = list(itertools.combinations(cat_cols[:5], 2))[:20]

        for col1, col2 in cross_pairs:
            result[f'{col1}_{col2}'] = df[col1].astype(str) + '_' + df[col2].astype(str)

        # 记录生成的特征
        self.generated_features = [c for c in result.columns if c not in df.columns]

        print(f"生成了 {len(self.generated_features)} 个新特征")

        return result


class AutoFeatureSelector:
    """
    自动特征选择
    """

    def __init__(self, n_features: int = 100, method: str = 'auto'):
        self.n_features = n_features
        self.method = method
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        自动选择特征
        """
        if self.method == 'auto':
            # 根据数据量自动选择方法
            if X.shape[1] < 100:
                method = 'tree'
            elif X.shape[1] < 500:
                method = 'mutual_info'
            else:
                method = 'variance'
        else:
            method = self.method

        if method == 'tree':
            self._tree_based_selection(X, y)
        elif method == 'mutual_info':
            self._mutual_info_selection(X, y)
        elif method == 'variance':
            self._variance_selection(X, y)

        return self

    def _tree_based_selection(self, X: pd.DataFrame, y: pd.Series):
        """基于树模型的特征选择"""
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=100,
            verbose=-1
        )
        model.fit(X, y)

        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        self.selected_features = importance.head(self.n_features).index.tolist()

    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series):
        """基于互信息的特征选择"""
        from sklearn.feature_selection import mutual_info_classif

        mi_scores = mutual_info_classif(X, y)
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        self.selected_features = mi_series.head(self.n_features).index.tolist()

    def _variance_selection(self, X: pd.DataFrame, y: pd.Series):
        """基于方差的特征选择"""
        variances = X.var().sort_values(ascending=False)
        self.selected_features = variances.head(self.n_features).index.tolist()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换"""
        return X[self.selected_features]
```

## 3. 超参数优化 (HPO)

### 3.1 贝叶斯优化

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


class BayesianHPO:
    """
    贝叶斯超参数优化
    """

    def __init__(self, model_class, param_space: Dict, n_trials: int = 50):
        self.model_class = model_class
        self.param_space = param_space
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None

    def optimize(self, X_train, y_train, X_val, y_val,
                 metric: str = 'auc') -> Dict:
        """
        执行优化
        """
        # 定义搜索空间
        dimensions = []
        param_names = []

        for name, space in self.param_space.items():
            param_names.append(name)

            if isinstance(space, list):
                dimensions.append(Categorical(space, name=name))
            elif isinstance(space, tuple) and len(space) == 2:
                if isinstance(space[0], int):
                    dimensions.append(Integer(space[0], space[1], name=name))
                else:
                    dimensions.append(Real(space[0], space[1], name=name))

        # 目标函数
        @use_named_args(dimensions=dimensions)
        def objective(**params):
            # 训练模型
            model = self.model_class(**params)
            model.fit(X_train, y_train)

            # 预测
            pred = model.predict_proba(X_val)[:, 1]

            # 计算指标
            from sklearn.metrics import roc_auc_score, log_loss

            if metric == 'auc':
                score = roc_auc_score(y_val, pred)
            else:
                score = -log_loss(y_val, pred)  # 负号因为要最大化

            return -score  # 最小化负分数

        # 运行优化
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=self.n_trials,
            random_state=42
        )

        # 提取最佳参数
        self.best_params = dict(zip(param_names, result.x))
        self.best_score = -result.fun

        print(f"最佳分数: {self.best_score:.4f}")
        print(f"最佳参数: {self.best_params}")

        return self.best_params


class RecSysHPO:
    """
    推荐系统超参数优化
    """

    # 推荐系统常用超参数空间
    LIGHTGBM_SPACE = {
        'n_estimators': (100, 1000),
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'max_depth': (3, 12),
        'num_leaves': (20, 200),
        'min_child_samples': (5, 100),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.0, 1.0),
    }

    DEEPFM_SPACE = {
        'embed_dim': [16, 32, 64, 128],
        'hidden_dims': [[256, 128], [512, 256], [256, 128, 64]],
        'dropout': (0.1, 0.5),
        'learning_rate': (0.0001, 0.01, 'log-uniform'),
        'batch_size': [256, 512, 1024, 2048],
    }

    DSSM_SPACE = {
        'embed_dim': [64, 128, 256],
        'hidden_dims': [[256, 128], [512, 256]],
        'dropout': (0.1, 0.5),
        'temperature': (0.05, 1.0),
    }
```

### 3.2 Hyperband

```python
class HyperbandOptimizer:
    """
    Hyperband 优化器

    通过早停策略加速超参数搜索
    """

    def __init__(self,
                 model_class,
                 param_space: Dict,
                 max_iter: int = 100,
                 eta: int = 3):
        self.model_class = model_class
        self.param_space = param_space
        self.max_iter = max_iter
        self.eta = eta

    def search(self, X_train, y_train, X_val, y_val,
               n_trials: int = 100) -> Dict:
        """
        执行 Hyperband 搜索
        """
        # 计算 bracket 配置
        logeta = lambda x: np.log(x) / np.log(self.eta)
        s_max = int(logeta(self.max_iter))
        B = (s_max + 1) * self.max_iter

        results = []

        for s in reversed(range(s_max + 1)):
            n = int(np.ceil(B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)

            # 采样配置
            configs = [self._sample_config() for _ in range(n)]

            # Successive Halving
            for i in range(s + 1):
                n_i = n * self.eta ** (-i)
                r_i = r * self.eta ** i

                # 训练和评估
                for config in configs[:int(n_i)]:
                    score = self._train_and_eval(
                        config, int(r_i),
                        X_train, y_train, X_val, y_val
                    )
                    results.append((config, score))

                # 保留最好的
                configs = sorted(configs, key=lambda c: results[-1][1], reverse=True)
                configs = configs[:int(n_i / self.eta)]

        # 返回最佳配置
        best = max(results, key=lambda x: x[1])
        return best[0]

    def _sample_config(self) -> Dict:
        """随机采样配置"""
        config = {}
        for name, space in self.param_space.items():
            if isinstance(space, list):
                config[name] = np.random.choice(space)
            elif isinstance(space, tuple):
                config[name] = np.random.uniform(space[0], space[1])
        return config

    def _train_and_eval(self, config, n_estimators, X_train, y_train, X_val, y_val):
        """训练并评估"""
        config['n_estimators'] = n_estimators

        model = self.model_class(**config)
        model.fit(X_train, y_train)

        pred = model.predict_proba(X_val)[:, 1]

        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_val, pred)
```

## 4. 神经架构搜索 (NAS)

### 4.1 搜索空间定义

```python
@dataclass
class RecSysArchitecture:
    """
    推荐系统架构定义
    """
    # Embedding
    use_shared_embedding: bool = True
    embed_dim: int = 64

    # 特征交叉
    cross_type: str = 'fm'  # fm, cross, attention, none
    cross_layers: int = 2

    # Deep 部分
    deep_layers: List[int] = None
    deep_activation: str = 'relu'
    deep_dropout: float = 0.2

    # 多任务
    n_tasks: int = 1
    task_type: str = 'mmoe'  # mmoe, ple, esmm


class NASSearchSpace:
    """
    NAS 搜索空间
    """

    # 搜索空间定义
    EMBED_DIMS = [16, 32, 64, 128, 256]
    CROSS_TYPES = ['fm', 'cross', 'attention', 'none']
    CROSS_LAYERS = [0, 1, 2, 3, 4]
    DEEP_LAYER_OPTIONS = [
        [128],
        [256, 128],
        [512, 256, 128],
        [1024, 512, 256],
        [256, 128, 64]
    ]
    ACTIVATIONS = ['relu', 'gelu', 'swish', 'tanh']
    DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]

    @staticmethod
    def sample_architecture() -> RecSysArchitecture:
        """随机采样架构"""
        return RecSysArchitecture(
            use_shared_embedding=np.random.choice([True, False]),
            embed_dim=np.random.choice(NASSearchSpace.EMBED_DIMS),
            cross_type=np.random.choice(NASSearchSpace.CROSS_TYPES),
            cross_layers=np.random.choice(NASSearchSpace.CROSS_LAYERS),
            deep_layers=np.random.choice(NASSearchSpace.DEEP_LAYER_OPTIONS),
            deep_activation=np.random.choice(NASSearchSpace.ACTIVATIONS),
            deep_dropout=np.random.choice(NASSearchSpace.DROPOUT_RATES),
            n_tasks=1,
            task_type='single'
        )
```

### 4.2 架构搜索

```python
class NASSearcher:
    """
    神经架构搜索
    """

    def __init__(self,
                 search_space: NASSearchSpace,
                 n_trials: int = 50,
                 epochs_per_trial: int = 5):
        self.search_space = search_space
        self.n_trials = n_trials
        self.epochs_per_trial = epochs_per_trial

        self.trial_history = []

    def search(self, train_loader, val_loader) -> RecSysArchitecture:
        """
        执行架构搜索
        """
        for trial in range(self.n_trials):
            # 采样架构
            arch = self.search_space.sample_architecture()

            # 构建模型
            model = self._build_model(arch)

            # 训练评估
            score = self._train_and_eval(model, train_loader, val_loader)

            self.trial_history.append({
                'architecture': arch,
                'score': score,
                'trial': trial
            })

            print(f"Trial {trial}: score={score:.4f}, arch={arch}")

        # 返回最佳架构
        best_trial = max(self.trial_history, key=lambda x: x['score'])
        return best_trial['architecture']

    def _build_model(self, arch: RecSysArchitecture) -> nn.Module:
        """根据架构构建模型"""
        # 这里简化，实际需要根据架构参数动态构建
        class DynamicRecModel(nn.Module):
            def __init__(self, architecture):
                super().__init__()
                self.arch = architecture

                # Embedding
                self.embedding = nn.Embedding(10000, architecture.embed_dim)

                # Cross 部分
                if architecture.cross_type == 'fm':
                    self.cross = FMModule(architecture.embed_dim)
                elif architecture.cross_type == 'cross':
                    self.cross = CrossNetwork(architecture.embed_dim, architecture.cross_layers)
                else:
                    self.cross = None

                # Deep 部分
                layers = []
                input_dim = architecture.embed_dim
                for hidden_dim in (architecture.deep_layers or [256, 128]):
                    layers.extend([
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU() if architecture.deep_activation == 'relu' else nn.GELU(),
                        nn.Dropout(architecture.deep_dropout)
                    ])
                    input_dim = hidden_dim
                layers.append(nn.Linear(input_dim, 1))

                self.deep = nn.Sequential(*layers)

            def forward(self, x):
                embed = self.embedding(x)

                if self.cross:
                    cross_out = self.cross(embed)
                else:
                    cross_out = 0

                deep_out = self.deep(embed.flatten(1))

                return cross_out + deep_out

        return DynamicRecModel(arch)

    def _train_and_eval(self, model, train_loader, val_loader) -> float:
        """训练并评估"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        # 训练
        model.train()
        for epoch in range(self.epochs_per_trial):
            for batch in train_loader:
                x, y = batch['features'], batch['label']
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred.squeeze(), y.float())
                loss.backward()
                optimizer.step()

        # 评估
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch['features'], batch['label']
                pred = torch.sigmoid(model(x).squeeze())
                all_preds.extend(pred.numpy())
                all_labels.extend(y.numpy())

        from sklearn.metrics import roc_auc_score
        return roc_auc_score(all_labels, all_preds)


class FMModule(nn.Module):
    """FM 模块"""
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        return self.linear(x.mean(dim=1))


class CrossNetwork(nn.Module):
    """Cross Network"""
    def __init__(self, embed_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])

    def forward(self, x):
        x0 = x.flatten(1)
        xl = x0
        for layer in self.layers:
            xl = x0 * layer(xl) + xl
        return xl.mean(dim=1, keepdim=True)
```

## 5. AutoML 端到端系统

### 5.1 完整 AutoML 流水线

```python
class RecSysAutoML:
    """
    推荐系统 AutoML 端到端系统
    """

    def __init__(self,
                 target_metric: str = 'auc',
                 max_time_minutes: int = 60,
                 n_trials: int = 50):
        self.target_metric = target_metric
        self.max_time_minutes = max_time_minutes
        self.n_trials = n_trials

        # 组件
        self.feature_generator = AutoFeatureGenerator()
        self.feature_selector = AutoFeatureSelector()
        self.hpo = None
        self.nas = None

        # 结果
        self.best_model = None
        self.best_config = None
        self.best_score = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        自动训练
        """
        print("=" * 50)
        print("开始 AutoML 训练")
        print("=" * 50)

        # 1. 自动特征工程
        print("\n[Step 1] 自动特征工程...")
        X_enhanced = self.feature_generator.generate_features(X)

        if X_val is not None:
            X_val_enhanced = self.feature_generator.generate_features(X_val)
        else:
            X_val_enhanced = None

        # 2. 自动特征选择
        print("\n[Step 2] 自动特征选择...")
        self.feature_selector.fit(X_enhanced, y)
        X_selected = self.feature_selector.transform(X_enhanced)

        if X_val_enhanced is not None:
            X_val_selected = self.feature_selector.transform(X_val_enhanced)
        else:
            # 自动划分验证集
            from sklearn.model_selection import train_test_split
            X_selected, X_val_selected, y, y_val = train_test_split(
                X_selected, y, test_size=0.2
            )

        # 3. 模型选择与超参数优化
        print("\n[Step 3] 模型选择与超参数优化...")

        # 尝试多个模型
        model_results = []

        # LightGBM
        lgb_result = self._try_lightgbm(X_selected, y, X_val_selected, y_val)
        model_results.append(('LightGBM', lgb_result))

        # 深度模型 (如果数据量足够)
        if len(X_selected) > 10000:
            deep_result = self._try_deepfm(X_selected, y, X_val_selected, y_val)
            model_results.append(('DeepFM', deep_result))

        # 选择最佳模型
        best_model_name, best_result = max(model_results, key=lambda x: x[1]['score'])
        self.best_model = best_result['model']
        self.best_config = best_result['config']
        self.best_score = best_result['score']

        print(f"\n最佳模型: {best_model_name}")
        print(f"最佳分数: {self.best_score:.4f}")
        print(f"最佳配置: {self.best_config}")

        return self

    def _try_lightgbm(self, X_train, y_train, X_val, y_val) -> Dict:
        """尝试 LightGBM"""
        import lightgbm as lgb

        # 简化的超参数搜索
        configs = [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8},
            {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 10},
        ]

        best_score = 0
        best_model = None
        best_config = None

        for config in configs:
            model = lgb.LGBMClassifier(**config, verbose=-1)
            model.fit(X_train, y_train)

            pred = model.predict_proba(X_val)[:, 1]

            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_val, pred)

            if score > best_score:
                best_score = score
                best_model = model
                best_config = config

        return {'model': best_model, 'config': best_config, 'score': best_score}

    def _try_deepfm(self, X_train, y_train, X_val, y_val) -> Dict:
        """尝试 DeepFM"""
        # 简化版本
        config = {
            'embed_dim': 32,
            'hidden_dims': [256, 128],
            'dropout': 0.2,
            'epochs': 10
        }

        # 训练 DeepFM (简化)
        score = 0.7  # 示例分数

        return {'model': None, 'config': config, 'score': score}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        # 特征工程
        X_enhanced = self.feature_generator.generate_features(X)

        # 特征选择
        X_selected = self.feature_selector.transform(X_enhanced)

        # 预测
        return self.best_model.predict_proba(X_selected)[:, 1]
```

## 6. 学习总结

### 6.1 AutoML 组件

```
组件            方法                    工具
────────────────────────────────────────────────
特征工程        自动生成、选择          Featuretools
超参数优化      贝叶斯、Hyperband       Optuna, Hyperopt
架构搜索        搜索空间、评估          AutoKeras
模型选择        多模型对比              TPOT, Auto-sklearn
```

### 6.2 推荐系统 AutoML 最佳实践

```
1. 特征工程优先: 好的特征比复杂模型更重要
2. 增量优化: 先优化主要超参数
3. 早停策略: 快速淘汰差配置
4. 并行搜索: 多进程加速
5. 经验利用: 使用已知好的配置作为起点
```
