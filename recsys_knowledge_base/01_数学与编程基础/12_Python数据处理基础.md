# Python数据处理基础 学习文档

## 1. 数据处理概述

### 1.1 为什么推荐系统需要数据处理？

```
推荐系统数据处理需求:
1. 数据清洗: 处理缺失值、异常值
2. 数据转换: 特征编码、归一化
3. 数据聚合: 用户行为统计、物品统计
4. 数据采样: 正负样本采样、下采样

典型数据流:
原始日志 → 数据清洗 → 特征工程 → 样本构造 → 模型训练
```

### 1.2 核心库介绍

```python
import numpy as np          # 数值计算
import pandas as pd         # 数据处理
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import json
from datetime import datetime, timedelta
```

## 2. NumPy 数值计算

### 2.1 数组基础

```python
class NumpyBasics:
    """
    NumPy 基础操作
    """

    @staticmethod
    def array_creation():
        """数组创建"""
        # 从列表创建
        arr1 = np.array([1, 2, 3, 4, 5])

        # 创建特殊数组
        zeros = np.zeros((3, 4))       # 3x4 零矩阵
        ones = np.ones((2, 3))         # 2x3 一矩阵
        eye = np.eye(3)                # 3x3 单位矩阵
        random = np.random.randn(3, 3) # 标准正态分布

        # 序列数组
        range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
        linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

        return arr1, zeros, ones, eye

    @staticmethod
    def array_operations():
        """数组运算"""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        # 基本运算（逐元素）
        add = a + b          # 加法
        sub = a - b          # 减法
        mul = a * b          # 逐元素乘法
        div = a / b          # 逐元素除法

        # 矩阵运算
        dot = np.dot(a, b)   # 矩阵乘法
        matmul = a @ b       # 矩阵乘法（Python 3.5+）

        # 聚合运算
        sum_all = a.sum()           # 所有元素求和
        sum_axis0 = a.sum(axis=0)   # 按列求和
        sum_axis1 = a.sum(axis=1)   # 按行求和
        mean = a.mean()             # 均值
        std = a.std()               # 标准差

        return dot, sum_all, mean
```

### 2.2 推荐系统常用操作

```python
class RecSysNumpy:
    """
    推荐系统中的 NumPy 操作
    """

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        计算余弦相似度

        常用于:
        - 用户相似度
        - 物品相似度
        - 向量召回
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b + 1e-10)

    @staticmethod
    def batch_cosine_similarity(embeddings: np.ndarray,
                                 queries: np.ndarray) -> np.ndarray:
        """
        批量计算余弦相似度

        embeddings: (n_items, dim) 物品嵌入
        queries: (n_queries, dim) 查询嵌入
        返回: (n_queries, n_items) 相似度矩阵
        """
        # 归一化
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)

        # 批量点积
        return queries_norm @ embeddings_norm.T

    @staticmethod
    def top_k_indices(scores: np.ndarray, k: int = 10) -> np.ndarray:
        """
        获取 Top-K 索引

        用于召回排序
        """
        # 方法1: argpartition + sort（更高效）
        if k >= len(scores):
            return np.argsort(scores)[::-1]

        partitioned = np.argpartition(scores, -k)[-k:]
        return partitioned[np.argsort(scores[partitioned])[::-1]]

    @staticmethod
    def sparse_to_dense(sparse_dict: Dict[int, float],
                        dim: int) -> np.ndarray:
        """
        稀疏向量转稠密向量

        常用于用户行为向量
        """
        dense = np.zeros(dim)
        for idx, val in sparse_dict.items():
            if 0 <= idx < dim:
                dense[idx] = val
        return dense
```

## 3. Pandas 数据处理

### 3.1 DataFrame 基础

```python
class PandasBasics:
    """
    Pandas 基础操作
    """

    @staticmethod
    def create_dataframe():
        """创建 DataFrame"""
        # 从字典创建
        data = {
            'user_id': [1, 2, 3, 4, 5],
            'item_id': [101, 102, 103, 104, 105],
            'click': [1, 0, 1, 1, 0],
            'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02',
                                          '2024-01-03', '2024-01-04',
                                          '2024-01-05'])
        }
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def basic_operations(df: pd.DataFrame):
        """基本操作"""
        # 查看数据
        print(df.head())          # 前5行
        print(df.info())          # 数据类型
        print(df.describe())      # 统计描述

        # 选择列
        user_ids = df['user_id']  # 单列
        subset = df[['user_id', 'item_id']]  # 多列

        # 筛选行
        clicked = df[df['click'] == 1]
        filtered = df[(df['click'] == 1) & (df['user_id'] > 2)]

        return clicked
```

### 3.2 推荐系统数据处理

```python
class RecSysPandas:
    """
    推荐系统中的 Pandas 操作
    """

    @staticmethod
    def load_interaction_data(filepath: str) -> pd.DataFrame:
        """
        加载交互数据
        """
        df = pd.read_csv(filepath)

        # 确保数据类型正确
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    @staticmethod
    def compute_user_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算用户统计特征

        常用特征:
        - 用户点击次数
        - 用户活跃天数
        - 用户偏好类目
        """
        user_stats = df.groupby('user_id').agg({
            'item_id': 'count',           # 交互次数
            'click': 'sum',               # 点击次数
            'timestamp': ['min', 'max']   # 活跃时间范围
        }).reset_index()

        user_stats.columns = ['user_id', 'interaction_count',
                              'click_count', 'first_active', 'last_active']

        # 计算活跃天数
        user_stats['active_days'] = (
            user_stats['last_active'] - user_stats['first_active']
        ).dt.days + 1

        # 点击率
        user_stats['click_rate'] = (
            user_stats['click_count'] / user_stats['interaction_count']
        )

        return user_stats

    @staticmethod
    def compute_item_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算物品统计特征

        常用特征:
        - 物品曝光次数
        - 物品点击次数
        - 物品 CTR
        """
        item_stats = df.groupby('item_id').agg({
            'user_id': 'count',  # 曝光次数
            'click': 'sum'       # 点击次数
        }).reset_index()

        item_stats.columns = ['item_id', 'exposure_count', 'click_count']

        # 计算 CTR
        item_stats['ctr'] = item_stats['click_count'] / item_stats['exposure_count']

        # 平滑 CTR（贝叶斯平滑）
        alpha, beta = 1, 100  # 先验参数
        item_stats['smooth_ctr'] = (
            (item_stats['click_count'] + alpha) /
            (item_stats['exposure_count'] + alpha + beta)
        )

        return item_stats

    @staticmethod
    def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        创建用户-物品交互矩阵

        用于协同过滤
        """
        # 方式1: pivot
        matrix = df.pivot_table(
            index='user_id',
            columns='item_id',
            values='click',
            fill_value=0
        )

        return matrix

    @staticmethod
    def negative_sampling(df: pd.DataFrame,
                          item_pool: List[int],
                          neg_ratio: int = 4) -> pd.DataFrame:
        """
        负采样

        将正样本和负样本组合成训练数据
        """
        # 获取正样本
        positive = df[df['click'] == 1][['user_id', 'item_id']].copy()
        positive['label'] = 1

        # 获取用户已交互物品
        user_items = df.groupby('user_id')['item_id'].apply(set).to_dict()

        # 为每个用户采样负样本
        negative_samples = []
        for user_id, interacted_items in user_items.items():
            # 候选负样本（未交互物品）
            candidates = [i for i in item_pool if i not in interacted_items]

            # 随机采样
            n_neg = min(len(interacted_items) * neg_ratio, len(candidates))
            if n_neg > 0:
                neg_items = np.random.choice(candidates, n_neg, replace=False)
                for item_id in neg_items:
                    negative_samples.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'label': 0
                    })

        negative = pd.DataFrame(negative_samples)

        # 合并
        return pd.concat([positive, negative], ignore_index=True)
```

### 3.3 时间序列处理

```python
class TimeSeriesProcessing:
    """
    时间序列数据处理
    """

    @staticmethod
    def create_time_features(df: pd.DataFrame,
                             time_col: str = 'timestamp') -> pd.DataFrame:
        """
        创建时间特征

        常用特征:
        - 小时、星期几、月份
        - 是否周末
        - 时间段（早中晚）
        """
        df = df.copy()

        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['day_of_month'] = df[time_col].dt.day
        df['month'] = df[time_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # 时间段
        def get_time_period(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 22:
                return 'evening'
            else:
                return 'night'

        df['time_period'] = df['hour'].apply(get_time_period)

        return df

    @staticmethod
    def create_sequence_features(df: pd.DataFrame,
                                  user_col: str = 'user_id',
                                  item_col: str = 'item_id',
                                  time_col: str = 'timestamp',
                                  max_seq_len: int = 50) -> pd.DataFrame:
        """
        创建序列特征

        用于序列推荐
        """
        df = df.sort_values([user_col, time_col])

        sequences = []

        for user_id, group in df.groupby(user_col):
            items = group[item_col].tolist()
            times = group[time_col].tolist()

            for i in range(1, len(items)):
                # 历史序列
                history = items[max(0, i-max_seq_len):i]

                sequences.append({
                    user_col: user_id,
                    'target_item': items[i],
                    'history_items': history,
                    'history_length': len(history),
                    'time_gap': (times[i] - times[i-1]).total_seconds() / 3600  # 小时
                })

        return pd.DataFrame(sequences)
```

## 4. 数据读取与存储

### 4.1 多格式读写

```python
class DataIO:
    """
    数据读写
    """

    @staticmethod
    def read_large_csv(filepath: str,
                        chunksize: int = 100000) -> pd.DataFrame:
        """
        分块读取大文件
        """
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            # 处理每个块
            chunks.append(chunk)

        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    def save_to_parquet(df: pd.DataFrame, filepath: str):
        """
        保存为 Parquet 格式

        优势:
        - 列式存储，读取快
        - 压缩率高
        - 保留数据类型
        """
        df.to_parquet(filepath, index=False)

    @staticmethod
    def load_from_parquet(filepath: str,
                           columns: List[str] = None) -> pd.DataFrame:
        """读取 Parquet 文件"""
        return pd.read_parquet(filepath, columns=columns)

    @staticmethod
    def save_embeddings(embeddings: np.ndarray,
                        item_ids: List[int],
                        filepath: str):
        """
        保存嵌入向量
        """
        np.savez_compressed(
            filepath,
            embeddings=embeddings,
            item_ids=np.array(item_ids)
        )

    @staticmethod
    def load_embeddings(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载嵌入向量"""
        data = np.load(filepath)
        return data['embeddings'], data['item_ids']
```

## 5. 数据预处理流水线

### 5.1 完整流水线

```python
class DataPipeline:
    """
    数据预处理流水线
    """

    def __init__(self):
        self.user_stats = None
        self.item_stats = None
        self.label_encoders = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换"""
        # 1. 数据清洗
        df = self.clean_data(df)

        # 2. 特征工程
        df = self.engineer_features(df)

        # 3. 编码
        df = self.encode_features(df)

        # 4. 标签构造
        df = self.create_labels(df)

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        df = df.copy()

        # 删除重复
        df = df.drop_duplicates()

        # 处理缺失值
        df['click'] = df['click'].fillna(0)

        # 删除异常值（如超过100岁的用户）
        if 'age' in df.columns:
            df = df[(df['age'] > 0) & (df['age'] < 100)]

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        df = df.copy()

        # 用户统计特征
        self.user_stats = RecSysPandas.compute_user_stats(df)
        df = df.merge(self.user_stats, on='user_id', how='left')

        # 物品统计特征
        self.item_stats = RecSysPandas.compute_item_stats(df)
        df = df.merge(self.item_stats, on='item_id', how='left')

        # 时间特征
        if 'timestamp' in df.columns:
            df = TimeSeriesProcessing.create_time_features(df)

        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征编码"""
        df = df.copy()

        # Label Encoding
        from sklearn.preprocessing import LabelEncoder

        cat_cols = ['user_id', 'item_id']
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """构造标签"""
        df = df.copy()

        # 点击标签
        df['label_click'] = (df['click'] > 0).astype(int)

        # 转化标签（如果有）
        if 'conversion' in df.columns:
            df['label_conversion'] = (df['conversion'] > 0).astype(int)

        return df
```

## 6. 学习总结

### 6.1 核心要点

```
1. NumPy: 高效数值计算，向量化操作
2. Pandas: 数据处理，聚合统计
3. 向量化: 避免循环，使用矩阵运算
4. 内存管理: 分块处理大数据
```

### 6.2 最佳实践

```python
# 1. 向量化操作（快）
# 好
result = df['col1'] * df['col2']

# 慢
result = [row['col1'] * row['col2'] for _, row in df.iterrows()]

# 2. 避免 copy
# 好
df['new_col'] = values

# 慢
df = df.assign(new_col=values)

# 3. 使用分类类型
df['category'] = df['category'].astype('category')

# 4. 分块处理大文件
for chunk in pd.read_csv('large.csv', chunksize=100000):
    process(chunk)
```

### 6.3 常用操作速查

```
操作                    代码
────────────────────────────────────────
读取CSV                 pd.read_csv()
保存                    df.to_csv()
筛选                    df[df['col'] > 0]
分组统计                df.groupby('col').agg()
合并                    df1.merge(df2, on='key')
去重                    df.drop_duplicates()
排序                    df.sort_values('col')
填充缺失                df.fillna(0)
类型转换                df['col'].astype(int)
```
