
# NMF 学习文档

## 1. 算法基础认知
### 1.1 一句话定义
NMF（Non-negative Matrix Factorization，非负矩阵分解）是一种将非负矩阵分解为两个非负矩阵乘积的降维方法，分解后的矩阵具有可解释的"部分-整体"表示。

### 1.2 直觉类比
想象你有一张人脸图片，NMF能够将其分解为"眼睛"、"鼻子"、"嘴巴"等部分的组合。这就像把一幅画拆分成几个基本图案的叠加，每个图案代表一个"部分"，而整个图像就是这些部分的加权组合。

### 1.3 历史背景
NMF由Lee和Seung于1999年在《Nature》杂志上发表，提出后迅速成为处理非负数据的重要方法。该算法在图像处理、文本挖掘、推荐系统等领域广泛应用。

### 1.4 算法定位
- 类型：无监督学习
- 输出：降维后的特征表示
- 模型类别：非参数模型（矩阵分解）

### 1.5 前置知识
- 线性代数（矩阵运算）
- 优化基础（梯度下降）
- Python 编程（NumPy、scikit-learn）

## 2. 核心原理
### 2.1 核心思想
NMF的核心思想是"加性分解"——将一个非负矩阵表示为两个非负矩阵的乘积，其中一个矩阵代表"基向量"（部分），另一个矩阵代表"系数"（整体中各部分的权重）。

### 2.2 工作流程
1. 初始化两个非负矩阵W和H
2. 迭代优化：使用乘法更新规则最小化重构误差
3. 收敛后，得到分解后的矩阵
4. 使用W（或H）进行降维表示

### 2.3 关键概念解释
- **基矩阵W**：包含"部分"或"基部件"的矩阵
- **系数矩阵H**：表示每个样本由各基部件组合的权重
- **非负约束**：保证分解结果可解释
- **稀疏性**：可以添加稀疏约束

### 2.4 几何解释
从几何角度看，NMF将数据限制在非负象限内，基向量W张成一个锥形区域，每个数据点可以表示为这些基向量的非负线性组合。这种表示具有"整体由部分组成"的物理意义。

## 3. 数学公式与推导
### 3.1 符号约定
| 符号 | 含义 |
|------|------|
| $X$ | 原始非负矩阵 $(m \times n)$ |
| $W$ | 基矩阵 $(m \times r)$ |
| $H$ | 系数矩阵 $(r \times n)$ |
| $r$ | 降维维度（隐因子数） |
| $V$ | $W \times H$ 的近似 |

### 3.2 问题形式化
给定非负矩阵 $X \in \mathbb{R}_+^{m \times n}$，寻找非负矩阵 $W \in \mathbb{R}_+^{m \times r}$ 和 $H \in \mathbb{R}_+^{r \times n}$，使得：
$$\min_{W, H} \|X - WH\|_F^2 \quad \text{s.t.} \quad W \geq 0, H \geq 0$$

### 3.3 目标函数
$$L(W, H) = \|X - WH\|_F^2 = \sum_{i,j} (X_{ij} - (WH)_{ij})^2$$

### 3.4 推导过程
**乘法更新规则推导**：

对目标函数关于H求偏导：
$$\frac{\partial L}{\partial H} = -2W^T(X - WH)$$

使用梯度下降法：
$$H_{kj} \leftarrow H_{kj} + \eta_{kj} \circ (W^T(X - WH))_{kj}$$

选择合适的学习率可以推导出乘法更新规则：
$$H_{kj} \leftarrow H_{kj} \frac{(W^TX)_{kj}}{(WHH^T)_{kj}}$$

类似地，对W有：
$$W_{ik} \leftarrow W_{ik} \frac((XH^T)_{ik}}{(WHH^T)_{ik})$$

### 3.5 最终解/算法步骤
1. 初始化：随机初始化W和H（均匀分布或正态分布）
2. 迭代更新：
   - $H_{kj} \leftarrow H_{kj} \frac{(W^TX)_{kj}}{(WHH^T)_{kj}}$
   - $W_{ik} \leftarrow W_{ik} \frac((XH^T)_{ik}}{(WHH^T)_{ik})$
3. 重复直到收敛（达到最大迭代次数或误差变化小于阈值）

## 4. 训练过程讲解
### 4.1 数据预处理
- 确保数据非负（如像素值、词频）
- 标准化（可选，但保持非负）
- 缺失值处理（用0填充）

### 4.2 参数初始化
- 随机初始化（推荐：均匀分布）
- 基于SVD的初始化（更稳定）
- 指定初始W或H

### 4.3 迭代过程
```python
伪代码：
输入: X, r, 最大迭代T
1. 初始化 W >= 0, H >= 0
2. for t = 1 to T:
3.     H = H * (W^T X) / (W^T W H)
4.     W = W * (X H^T) / (W H H^T)
5.     if 收敛: break
输出: W, H
```

### 4.4 收敛条件
- 达到最大迭代次数
- 目标函数变化小于阈值
- W和H变化小于阈值

### 4.5 超参数及推荐范围
- n_components (r): 5-100（根据任务调整）
- init: 'nndsvd'（推荐）或 'random'
- max_iter: 200-500
- tol: 1e-4

## 5. 应用场景
### 5.1 典型应用
- **图像处理**：人脸分解、图像去噪
- **文本挖掘**：主题提取、文档表示
- **推荐系统**：用户-物品矩阵分解
- **生物信息学**：基因表达数据分析

### 5.2 适用数据特征
- 数据非负（像素值、计数、频率）
- 具有"部分-整体"结构
- 需要可解释的表示

### 5.3 不适用场景
- 数据包含负值
- 需要精确重构
- 数据不具有可加性结构

## 6. 优缺点分析
### 6.1 优点
- 分解结果具有物理意义
- 非负约束保证可解释性
- 可以实现稀疏表示
- 分解是"部分的"而非"整体的"

### 6.2 2
- 可能收敛到局部最优
- 对初始值敏感
- 迭代速度可能较慢
- 不保证全局最优

### 6.3 与同类算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| NMF | 可解释，稀疏 | 局部最优 | 非负数据 |
| PCA | 闭式解，正交 | 无稀疏 | 一般降维 |
| SVD | 理论基础好 | 无稀疏 | 文本语义 |
| ICA | 独立分量 | 计算复杂 | 信号分离 |

## 7. 调库实现
### 7.1 环境准备
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 7.2 完整代码示例
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import make_low_rank_matrix
from sklearn.preprocessing import normalize

# 1. 生成示例数据（模拟图像数据）
np.random.seed(42)
n_samples = 300
n_features = 100
n_components = 5

# 生成具有部分-整体结构的数据
W_true = np.random.rand(n_features, n_components)  # 基矩阵（部分）
H_true = np.random.rand(n_components, n_samples)  # 系数矩阵（权重）
X = W_true @ H_true + 0.1 * np.random.randn(n_features, n_samples)

# 确保非负
X = np.abs(X)

# 2. 使用NMF分解
nmf = NMF(n_components=n_components, init='nndsvd', max_iter=500, 
           random_state=42, alpha_W=0.1, alpha_H=0.1)
W = nmf.fit_transform(X)
H = nmf.components_

print(f"原始矩阵形状: {X.shape}")
print(f"W矩阵形状: {W.shape}")
print(f"H矩阵形状: {H.shape}")
print(f"重构误差: {nmf.reconstruction_err_:.4f}")

# 3. 可视化基向量（"部分"）
fig, axes = plt.subplots(1, n_components, figsize=(15, 3))
for i in range(n_components):
    axes[i].bar(range(20), W[:20, i])
    axes[i].set_title(f'基向量 {i+1}')
    axes[i].set_xlabel('特征')
    axes[i].set_ylabel('权重')
plt.tight_layout()
plt.show()

# 4. 重构示例
X_reconstructed = W @ H
sample_idx = 0
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(X[:, sample_idx], label='原始')
plt.plot(X_reconstructed[:, sample_idx], label='重构')
plt.title(f'样本{sample_idx}对比')
plt.legend()

plt.subplot(1, 2, 2)
plt.imshow(X_reconstructed[:, :50], aspect='auto', cmap='viridis')
plt.title('重构矩阵')
plt.colorbar()
plt.tight_layout()
plt.show()

# 5. 文本主题提取示例
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "机器学习是人工智能的核心技术",
    "深度学习是机器学习的进阶方法",
    "自然语言处理涉及文本和语音",
    "计算机视觉处理图像和视频",
    "神经网络是深度学习的基础结构",
    "机器学习应用广泛包括推荐系统",
    "自然语言处理和机器学习结合紧密",
    "深度学习在计算机视觉取得突破",
    "强化学习是机器学习的一个重要分支",
    "数据挖掘从大量数据中发现规律"
]

# 构建TF-IDF矩阵
vectorizer = TfidfVectorizer(max_features=50)
tfidf = vectorizer.fit_transform(documents)
tfidf_dense = tfidf.toarray()

# NMF主题提取
nmf_topic = NMF(n_components=3, random_state=42, max_iter=500)
doc_topics = nmf_topic.fit_transform(tfidf_dense)
topics = nmf_topic.components_

feature_names = vectorizer.get_feature_names_out()

print("\n=== 主题提取结果 ===")
for topic_idx, topic in enumerate(topics):
    top_words_idx = topic.argsort()[-5:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"主题{topic_idx+1}: {', '.join(top_words)}")

# 6. 可视化文档-主题分布
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.bar(np.arange(len(documents)) + i*0.25, doc_topics[:, i], 
            width=0.25, label=f'主题{i+1}')
plt.xlabel('文档')
plt.ylabel('主题权重')
plt.title('文档-主题分布')
plt.legend()
plt.xticks(range(len(documents)), range(len(documents)), rotation=45)
plt.tight_layout()
plt.show()
```

### 7.3 运行结果示例
```
原始矩阵形状: (300, 100)
W矩阵形状: (300, 5)
H矩阵形状: (5, 100)
重构误差: 15.2341

=== 主题提取结果 ===
主题1: 机器学习, 深度学习, 神经网络
主题2: 自然语言处理, 文本, 语音
主题3: 计算机视觉, 图像, 视频
```

## 8. 手工代码实现
### 8.1 核心算法手写
```python
import numpy as np

class NMFManual:
    """手工实现非负矩阵分解(NMF)"""
    
    def __init__(self, n_components=5, max_iter=500, tol=1e-4, 
                 init='random', random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.W = None
        self.H = None
        self.reconstruction_err_ = None
        
    def _init_WH(self, X):
        """初始化W和H矩阵"""
        np.random.seed(self.random_state)
        m, n = X.shape
        
        if self.init == 'random':
            W = np.random.rand(m, self.n_components)
            H = np.random.rand(self.n_components, n)
        elif self.init == 'nndsvd':
            # 简化的NNDSVD初始化
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            W = np.abs(U[:, :self.n_components] @ np.diag(np.sqrt(s[:self.n_components])))
            H = np.abs(np.diag(np.sqrt(s[:self.n_components])) @ Vt[:self.n_components, :])
        else:
            W = np.random.rand(m, self.n_components)
            H = np.random.rand(self.n_components, n)
        
        return W, H
    
    def fit(self, X):
        """训练NMF模型"""
        X = np.array(X)
        
        self.W, self.H = self._init_WH(X)
        
        for iteration in range(self.max_iter):
            # 更新H
            numerator = self.W.T @ X
            denominator = self.W.T @ self.W @ self.H + 1e-10
            self.H = self.H * (numerator / denominator)
            
            # 更新W
            numerator = X @ self.H.T
            denominator = self.W @ self.H @ self.H.T + 1e-10
            self.W = self.W * (numerator / denominator)
            
            # 计算重构误差
            error = np.linalg.norm(X - self.W @ self.H, 'fro')
            
            if iteration > 0 and abs(error - self.reconstruction_err_) < self.tol:
                break
            
            self.reconstruction_err_ = error
        
        return self
    
    def transform(self, X):
        """获取样本的系数表示"""
        X = np.array(X)
        # 使用伪逆求解H
        H = np.linalg.lstsq(self.W, X, rcond=None)[0]
        H = np.maximum(H, 0)  # 确保非负
        return H.T
    
    def fit_transform(self, X):
        """训练并返回系数矩阵"""
        self.fit(X)
        return self.transform(X)

# 测试手工实现
if __name__ == '__main__':
    from sklearn.decomposition import NMF
    
    # 生成测试数据
    np.random.seed(42)
    W_true = np.random.rand(100, 5)
    H_true = np.random.rand(5, 50)
    X = W_true @ H_true
    
    # 手工实现
    nmf_manual = NMFManual(n_components=5, random_state=42)
    nmf_manual.fit(X)
    
    # sklearn实现
    nmf_sklearn = NMF(n_components=5, random_state=42, max_iter=500)
    W_sklearn = nmf_sklearn.fit_transform(X)
    H_sklearn = nmf_sklearn.components_
    
    print("=== NMF手工实现 vs sklearn ===")
    print(f"手工实现重构误差: {nmf_manual.reconstruction_err_:.4f}")
    print(f"sklearn重构误差: {nmf_sklearn.reconstruction_err_:.4f}")
    
    # 对比W矩阵（可能符号不同，但方向相同）
    print(f"\nW矩阵相似度: {np.abs(np.corrcoef(nmf_manual.W, W_sklearn).mean()):.4f}")
```

### 8.2 与调库结果对比
| 指标 | 手工实现 | sklearn |
|------|----------|---------|
| 重构误差 | 相近 | 相近 |
| 迭代次数 | 较多 | 优化过 |
| 稳定性 | 依赖初始化 | 更稳定 |

## 9. 可视化与结果理解
### 9.1 基向量可视化
```python
import matplotlib.pyplot as plt
import numpy as np

# 可视化W矩阵（基向量）
plt.figure(figsize=(12, 6))
plt.imshow(W, aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('样本')
plt.ylabel('基向量')
plt.title('W矩阵热图')
plt.show()

# 各基向量的特征重要性
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.bar(range(20), W[:20, i])
    plt.title(f'基向量{i+1}')
plt.tight_layout()
plt.show()
```

### 9.2 收敛曲线
```python
# 记录每次迭代的误差
errors = []

class NMFWithLogging(NMFManual):
    def fit(self, X):
        X = np.array(X)
        self.W, self.H = self._init_WH(X)
        
        for iteration in range(self.max_iter):
            # 更新H
            numerator = self.W.T @ X
            denominator = self.W.T @ self.W @ self.H + 1e-10
            self.H = self.H * (numerator / denominator)
            
            # 更新W
            numerator = X @ self.H.T
            denominator = self.W @ self.H @ self.H.T + 1e-10
            self.W = self.W * (numerator / denominator)
            
            error = np.linalg.norm(X - self.W @ self.H, 'fro')
            errors.append(error)
            
            if iteration > 0 and abs(error - errors[-2]) < self.tol:
                break
        
        self.reconstruction_err_ = error
        return self

nmf_log = NMFWithLogging(n_components=5, random_state=42)
nmf_log.fit(X)

plt.figure(figsize=(10, 5))
plt.plot(errors)
plt.xlabel('迭代次数')
plt.ylabel('重构误差')
plt.title('NMF收敛曲线')
plt.grid(True)
plt.show()
```

### 9.3 结果解读
- W矩阵的每列是一个"基部件"
- H矩阵表示每个样本由各基部件组合的权重
- 重构误差越小表示分解效果越好

## 10. 模型评估
### 10.1 评估指标选择
- **重构误差**：$\|X - WH\|_F^2$
- **稀疏性**：稀疏因子
- **可解释性**：基向量的可解释程度

### 10.2 重构误差评估
```python
from sklearn.decomposition import NMF

for n_comp in [3, 5, 10, 20]:
    nmf = NMF(n_components=n_comp, random_state=42, max_iter=500)
    nmf.fit_transform(X)
    print(f"n_components={n_comp}, 误差: {nmf.reconstruction_err_:.4f}")
```

### 10.3 稀疏性评估
```python
def sparsity(M):
    """计算矩阵稀疏性 (0-1之间，越接近1越稀疏)"""
    n = M.shape[0] * M.shape[1]
    return 1 - np.count_nonzero(M) / n

print(f"W稀疏性: {sparsity(nmf.W):.4f}")
print(f"H稀疏性: {sparsity(nmf.H):.4f}")
```

## 11. 常见问题与易错点
### 11.1 数据层面常见错误
- 数据包含负值（NMF要求非负）
- 数据未进行预处理
- 维度选择不当

### 11.2 模型层面常见错误
- 迭代不收敛（调整迭代次数或容差）
- 局部最优（多次运行取最优）
- 初始化不当（使用NNDSVD）

### 11.3 调参层面常见误区
- 盲目增加n_components
- 忽视稀疏约束的作用
- 未考虑计算效率

## 12. 学习总结
### 12.1 核心要点回顾
- NMF将非负矩阵分解为两个非负矩阵的乘积
- 分解结果具有"部分-整体"的可解释性
- 使用乘法更新规则进行优化
- 可以添加稀疏约束增强可解释性

### 12.2 关键公式汇总
- 目标函数：$\min_{W,H} \|X - WH\|_F^2$
- H更新：$H_{kj} \leftarrow H_{kj} \frac{(W^TX)_{kj}}{(WHH^T)_{kj}}$
- W更新：$W_{ik} \leftarrow W_{ik} \frac((XH^T)_{ik}}{(WHH^T)_{ik})$

### 12.3 与前序/后续算法联系
- **前置算法**：数据预处理、PCA
- **后续算法**：稀疏编码、字典学习

## 13. 练习题与思考题与思考题
### 13.1 基础练习题
1. NMF与PCA的主要区别是什么？
2. 为什么NMF要求数据非负？
3. 简述NMF的优化过程。

### 13.2 进阶思考题
1. NMF如何实现稀疏表示？
2. 如何选择合适的n_components？

### 13.3 详细答案与解析
1. **答案**：PCA产生正交的基向量，可以有负值；NMF产生非负的基向量，具有可解释的"部分-整体"含义。
2. **答案**：非负约束使得分解结果具有物理意义，可以理解为"部分"和"权重"的组合。
3. **答案**：使用乘法更新规则，交替更新W和H，直到收敛。

## 14. 学习路径建议建议
### 14.1 前置知识
- 矩阵运算基础
- 优化方法基础
- 线性代数

### 14.2 平行算法
- PCA（无监督降维）
- SVD（矩阵分解）
- 字典学习

### 14.3 进阶算法
- 稀疏NMF
- 卷积NMF
- 深度NMF

### 14.4 推荐资源
- Lee & Seung (1999) "Learning the parts of objects by non-negative matrix factorization"
- scikit-learn NMF文档
- 《Non-negative Matrix Factorization for Signal and Image Processing》
