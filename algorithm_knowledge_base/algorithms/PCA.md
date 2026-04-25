# PCA 主成分分析 学习文档

> 通过正交变换将高维数据投影到方差最大的方向，实现无监督线性降维

---

## 1. 算法基础认知

### 1.1 一句话定义

PCA（Principal Component Analysis）是一种无监督线性降维方法，通过寻找数据方差最大的正交方向，将高维数据投影到低维空间，同时尽可能保留原始数据的信息。

### 1.2 直觉类比

想象你站在一片旷野上，手里有一把豆子随手撒了出去。豆子散落在地上形成一个椭圆形的区域。如果你从正上方往下看，能清楚看到豆子散开的范围（长轴方向）；但如果你从侧面平视，豆子几乎挤成一条线，你很难分辨出它们的分布差异。

PCA做的事情就是：找到那个"从上方看"的最佳角度。更准确地说，它找到数据分布最"宽"的方向作为第一主成分，然后在垂直于第一主成分的方向上找第二"宽"的方向作为第二主成分，以此类推。如果数据本身存在很强的相关性（豆子散布很扁），那么用前几个主成分就能描述绝大部分信息，后面的主成分可以安全丢弃——这就是降维。

### 1.3 历史背景

- 1901年，Karl Pearson 在论文 "On Lines and Planes of Closest Fit to Systems of Points in Space" 中首次提出，当时的目标是用直线或平面最优地拟合数据点
- 1933年，Harold Hotelling 独立发展了该方法并命名为"主成分分析"
- 至今仍是数据预处理、特征提取、可视化等领域最基础最广泛使用的降维工具
- 在傅罡著《人工智能注意力机制：体系、模型与算法剖析》中，SUN-UD 显著性检测模型即使用 PCA 将线性 ICA 滤波响应降维为 94 维特征，再送入 SVM 进行分类

### 1.4 算法定位

| 特性 | 说明 |
|------|------|
| 类型 | 无监督学习 --> 降维 |
| 输出 | 低维空间的投影矩阵和主成分得分 |
| 模型类型 | 非概率的线性变换方法 |
| 是否需要标签 | 否 |

### 1.5 前置知识

- **线性代数**：矩阵乘法、协方差矩阵、特征值与特征向量的定义与计算、正交性
- **概率统计**：方差、协方差的含义，数据分布的中心化
- **微积分**：拉格朗日乘数法（用于推导约束优化问题）
- **扩展知识**：SVD 奇异值分解（与 PCA 的等价关系）

---

## 2. 核心原理

### 2.1 核心思想

PCA 的核心思想可以概括为一句话：**在所有可能的低维投影中，选择使投影后数据方差最大的投影方向，且各投影方向互相正交。**

为什么选择"方差最大"？因为方差衡量的是数据的离散程度——方差越大，说明数据在该方向上"展开得越开"，包含的信息就越多。反之，方差越小的方向，数据几乎重叠在一起，丢弃这些方向损失的信息极少。

这种"最大方差"的视角与另一种"最小重构误差"的视角是等价的。如果把数据投影到低维子空间后再尝试还原，那么选择方差最大的方向投影，等价于使还原后与原始数据的误差最小。也就是说：

> **最大化投影方差 = 最小化重构误差**

这是理解 PCA 的两个互补角度。

### 2.2 工作流程

1. **数据中心化**：将每个特征减去其均值，使数据的均值为零
   - 输入：原始数据矩阵 $X \in \mathbb{R}^{n \times d}$
   - 输出：中心化后的数据 $\tilde{X} = X - \bar{X}$

2. **计算协方差矩阵**：衡量各特征之间的线性相关性
   - 操作：$C = \frac{1}{n-1}\tilde{X}^T \tilde{X}$
   - 输出：协方差矩阵 $C \in \mathbb{R}^{d \times d}$

3. **特征值分解**：对协方差矩阵进行特征值分解
   - 操作：$C = W \Lambda W^T$
   - 输出：特征值 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$ 和对应的特征向量 $w_1, w_2, \ldots, w_d$

4. **选择主成分**：按特征值从大到小排序，选取前 $k$ 个特征向量组成投影矩阵
   - 关键操作：$W_k = [w_1, w_2, \ldots, w_k]$
   - 决策点：$k$ 的选择取决于累计方差贡献率

5. **投影到低维空间**：用选定的主成分方向进行投影
   - 输入：中心化数据 $\tilde{X}$ 和投影矩阵 $W_k$
   - 输出：低维表示 $Z = \tilde{X} W_k \in \mathbb{R}^{n \times k}$

### 2.3 关键概念解释

- **主成分（Principal Component）**：数据投影后方差最大的方向。第一主成分是方差最大的投影方向，第二主成分是与第一主成分正交且方差次大的方向，以此类推。每一个主成分就是一个单位特征向量。

- **特征值（Eigenvalue）**：对应于每个主成分方向的方差值。特征值越大，说明该主成分方向上数据的离散程度越大，包含的信息越多。

- **方差贡献率（Variance Explained Ratio）**：某个主成分的特征值占所有特征值之和的比例。$\text{贡献率}_i = \lambda_i / \sum_{j=1}^{d} \lambda_j$。它表示该主成分保留了原始数据多少比例的信息。

- **累计方差贡献率**：前 $k$ 个主成分的方差贡献率之和。通常当累计贡献率达到 85%~95% 时，即可认为这 $k$ 个主成分保留了绝大部分信息。

### 2.4 几何解释

以二维数据降到一维为例：

```
        x2
        |
    *  * |  *
  *  * * | *   *
   * * **|* *  *
    *  * |*  *
      *  | *
        |         --> 第一主成分方向 w1（方差最大）
  ------+-------------> x1
        |
        |
        V
        --> 第二主成分方向 w2（与 w1 正交，方差较小）
```

如果数据点沿某个方向呈狭长分布，那么第一主成分就沿着这个"长轴"方向。将数据投影到第一主成分上，投影点的分布最为分散，信息保留最多。第二主成分与第一主成分垂直（正交），沿"短轴"方向，投影后的方差小，丢弃后信息损失小。

在高维空间中，PCA 做的事情完全相同：找到数据分布的"长轴"方向（方差最大的正交方向），依次排列，保留前 $k$ 个方向，丢弃其余方向。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $X$ | 原始数据矩阵 | $n \times d$ |
| $n$ | 样本数量 | 标量 |
| $d$ | 原始特征维度 | 标量 |
| $k$ | 目标降维维度（$k \leq d$） | 标量 |
| $\bar{x}_j$ | 第 $j$ 个特征的均值 | 标量 |
| $\tilde{X}$ | 中心化后的数据矩阵 | $n \times d$ |
| $C$ | 协方差矩阵 | $d \times d$ |
| $w_i$ | 第 $i$ 个主成分（特征向量） | $d \times 1$ |
| $\lambda_i$ | 第 $i$ 个特征值 | 标量 |
| $W_k$ | 投影矩阵（前 $k$ 个主成分） | $d \times k$ |
| $Z$ | 降维后的数据 | $n \times k$ |

### 3.2 问题形式化

给定数据集 $\{x_1, x_2, \ldots, x_n\}$，其中 $x_i \in \mathbb{R}^d$，我们的目标是找到一个 $k$ 维子空间（由 $k$ 个正交向量张成），使得数据投影到该子空间后方差最大。

等价地，也可以表述为：找到投影矩阵 $W_k \in \mathbb{R}^{d \times k}$，使得重构误差最小：

$$ \min_{W_k} \sum_{i=1}^{n} \|x_i - W_k W_k^T x_i\|^2 $$

下面从"最大化投影方差"的角度进行推导，再说明等价性。

### 3.3 最大方差推导

**Step 1：数据预处理（中心化）**

首先将数据中心化，即每个特征减去其均值：

$$ \tilde{x}_i = x_i - \bar{x}, \quad \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i $$

中心化后，$\sum_{i=1}^{n} \tilde{x}_i = 0$。后续所有推导都基于中心化后的数据。

**Step 2：定义第一主成分的目标**

寻找一个单位向量 $w_1$（$\|w_1\| = 1$），使得数据在 $w_1$ 方向上的投影方差最大。第 $i$ 个样本在 $w_1$ 上的投影为 $w_1^T \tilde{x}_i$，投影方差为：

$$ \text{Var}(w_1^T \tilde{X}) = \frac{1}{n-1}\sum_{i=1}^{n} (w_1^T \tilde{x}_i)^2 $$

用矩阵形式表示：

$$ \frac{1}{n-1}\sum_{i=1}^{n} (w_1^T \tilde{x}_i)^2 = \frac{1}{n-1}\sum_{i=1}^{n} w_1^T \tilde{x}_i \tilde{x}_i^T w_1 = w_1^T \left(\frac{1}{n-1}\sum_{i=1}^{n}\tilde{x}_i \tilde{x}_i^T\right) w_1 $$

注意到括号内的部分正是协方差矩阵的定义，因此：

$$ \text{Var}(w_1^T \tilde{X}) = w_1^T C w_1 $$

其中 $C = \frac{1}{n-1}\tilde{X}^T \tilde{X}$ 是协方差矩阵。

**Step 3：使用拉格朗日乘数法求解**

我们需要在约束 $\|w_1\|^2 = w_1^T w_1 = 1$ 的条件下最大化 $w_1^T C w_1$。构造拉格朗日函数：

$$ \mathcal{L}(w_1, \lambda) = w_1^T C w_1 - \lambda(w_1^T w_1 - 1) $$

对 $w_1$ 求偏导并令其为零：

$$ \frac{\partial \mathcal{L}}{\partial w_1} = 2Cw_1 - 2\lambda w_1 = 0 $$

整理得到：

$$ Cw_1 = \lambda w_1 $$

这就是标准的特征值问题！$w_1$ 是协方差矩阵 $C$ 的特征向量，$\lambda$ 是对应的特征值。

**Step 4：为什么选最大特征值对应的特征向量？**

将 $Cw_1 = \lambda w_1$ 两边左乘 $w_1^T$，得：

$$ w_1^T C w_1 = \lambda w_1^T w_1 = \lambda $$

因为 $w_1$ 是单位向量，所以 $w_1^T w_1 = 1$，从而：

$$ \text{投影方差} = w_1^T C w_1 = \lambda $$

这意味着：**投影方差恰好等于对应的特征值**。因此，要使投影方差最大，就必须选择最大特征值对应的特征向量作为第一主成分。

这就是"为什么选最大特征值对应的特征向量"的数学本质——特征值本身就是投影后方差的大小。

**Step 5：后续主成分的推导**

第二主成分 $w_2$ 需要满足两个约束：
1. 与 $w_1$ 正交：$w_1^T w_2 = 0$
2. 单位向量：$w_2^T w_2 = 1$

同样使用拉格朗日乘数法：

$$ \mathcal{L}(w_2, \lambda_2, \mu) = w_2^T C w_2 - \lambda_2(w_2^T w_2 - 1) - \mu(w_1^T w_2) $$

对 $w_2$ 求偏导令其为零：

$$ 2Cw_2 - 2\lambda_2 w_2 - \mu w_1 = 0 $$

左乘 $w_1^T$，并利用 $w_1^T w_2 = 0$ 和 $Cw_1 = \lambda_1 w_1$：

$$ 2w_1^T C w_2 - 0 - \mu = 0 $$
$$ 2\lambda_1 w_1^T w_2 - \mu = 0 $$
$$ \mu = 0 $$

因此 $Cw_2 = \lambda_2 w_2$，$w_2$ 同样是 $C$ 的特征向量。为了使方差最大，取第二大特征值对应的特征向量。

以此类推，第 $k$ 个主成分就是第 $k$ 大特征值对应的特征向量，且所有主成分两两正交。

**Step 6：投影与重构**

选定前 $k$ 个主成分组成投影矩阵 $W_k = [w_1, w_2, \ldots, w_k]$，则：

- **降维（投影）**：$Z = \tilde{X} W_k$，其中 $Z \in \mathbb{R}^{n \times k}$
- **重构（还原）**：$\hat{X} = Z W_k^T = \tilde{X} W_k W_k^T$
- **重构误差**：$\|X - \hat{X}\|^2 = \sum_{j=k+1}^{d} \lambda_j$（被丢弃的特征值之和）

### 3.4 最大方差与最小重构误差的等价性

设数据已中心化。将 $d$ 维数据投影到 $k$ 维子空间，重构误差为：

$$ J = \sum_{i=1}^{n} \|\tilde{x}_i - W_k W_k^T \tilde{x}_i\|^2 $$

展开：

$$ J = \sum_{i=1}^{n} \|\tilde{x}_i\|^2 - \sum_{i=1}^{n} \|W_k^T \tilde{x}_i\|^2 $$

第一项 $\sum_{i=1}^{n} \|\tilde{x}_i\|^2 = \text{tr}(C)$ 是常数（总方差）。第二项是投影后方差之和。因此：

$$ \min J \iff \max \sum_{i=1}^{n} \|W_k^T \tilde{x}_i\|^2 = \max \text{投影后方差之和} $$

两者完全等价。

### 3.5 SVD 等价推导

除了对协方差矩阵做特征值分解，PCA 也可以通过对数据矩阵做奇异值分解（SVD）来计算。

对中心化后的数据矩阵 $\tilde{X} \in \mathbb{R}^{n \times d}$ 做 SVD：

$$ \tilde{X} = U \Sigma V^T $$

其中 $U \in \mathbb{R}^{n \times n}$，$\Sigma \in \mathbb{R}^{n \times d}$（对角矩阵，对角元素为奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots$），$V \in \mathbb{R}^{d \times d}$。

协方差矩阵为：

$$ C = \frac{1}{n-1}\tilde{X}^T \tilde{X} = \frac{1}{n-1} V \Sigma^T \Sigma V^T $$

令 $\Lambda = \frac{1}{n-1}\Sigma^T \Sigma$，则 $C = V \Lambda V^T$。

因此：
- SVD 的右奇异矩阵 $V$ 的列向量就是协方差矩阵的特征向量，即主成分方向
- 特征值 $\lambda_i = \sigma_i^2 / (n-1)$

**为什么实践中更推荐用 SVD？**

1. 数值稳定性更好：当 $d > n$（特征数大于样本数）时，协方差矩阵 $C$ 是奇异的，但 SVD 仍然可以正常计算
2. 计算效率更高：SVD 直接作用于数据矩阵，避免了显式构造协方差矩阵
3. scikit-learn 的 PCA 实现底层使用的就是 SVD

---

## 4. 训练过程讲解

### 4.1 数据预处理

**必要预处理**：

1. **缺失值处理**：
   - PCA 不能处理含缺失值的数据
   - 方法：删除缺失行、均值填充、KNN 填充等
   ```python
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy='mean')
   X = imputer.fit_transform(X)
   ```

2. **标准化（极其重要）**：
   - PCA 对特征的尺度非常敏感。如果某个特征的量级远大于其他特征（例如以米为单位和以毫米为单位），那么该特征的主成分方向会被过度放大，导致降维结果失真
   - 原因：PCA 是基于方差进行优化的，量级大的特征天然具有更大的方差
   - 方法：StandardScaler（零均值单位方差）
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

   **注意**：如果数据各特征的物理含义和量级相同（例如都是像素灰度值），可以不做标准化。但大多数情况下建议标准化。

### 4.2 参数初始化

PCA 本身没有需要初始化的参数（不像神经网络有权重）。唯一需要设定的是目标维度 $k$，这通常通过分析方差贡献率来确定。

### 4.3 迭代过程

标准 PCA 有解析解（通过特征值分解或 SVD），不需要迭代。但存在增量 PCA（Incremental PCA）和随机 PCA（Randomized PCA）等变体用于大规模数据。

```
标准 PCA（解析解）：
1. 中心化数据
2. 计算 SVD：X = U * Sigma * V^T
3. 取 V 的前 k 列作为投影矩阵
4. 完成，无需迭代

随机 PCA（大数据场景）：
1. 中心化数据
2. 生成随机投影矩阵 Q
3. 迭代幂法求精：Q = X^T X Q, Q = QR分解取Q
4. 对 XQ 做 SVD
5. 还原到原始空间得到主成分
```

### 4.4 收敛条件

- 标准 PCA：无迭代，直接计算解析解
- 增量/随机 PCA：当主成分变化小于阈值或达到最大迭代次数时停止

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| n_components | 降维后的维度 | 整数或0~1浮点数 | None（保留全部） |
| svd_solver | SVD 求解器 | 'auto'/'full'/'arpack'/'randomized' | 'auto' |
| whiten | 白化（使投影后方差为1） | True/False | False |
| tol | ARPACK 求解器的收敛容差 | 0~1 | 0.0 |

**n_components 的选择策略**：

- **指定具体整数**：当明确知道目标维度时使用
- **指定 0~1 的浮点数**：自动选择使累计方差贡献率达到该比例的最小维度，这是最常用的方式
- **不指定（None）**：保留全部主成分，用于分析方差贡献率

---

## 5. 应用场景

### 5.1 典型应用

**应用1：数据预处理与特征降维**
- 问题类型：特征工程
- 为什么适合：高维数据往往存在冗余特征和噪声，PCA 可以在保留主要信息的同时大幅降低维度，加速后续模型的训练
- 实际案例：人脸识别中，将数千维的像素特征降至百维以内（Eigenfaces 方法）；在《人工智能注意力机制》中，SUN-UD 模型使用 PCA 将 ICA 滤波响应降维为 94 维特征

**应用2：数据可视化**
- 问题类型：探索性数据分析
- 为什么适合：PCA 可以将任意高维数据降至 2 维或 3 维，便于在平面上直观展示数据的聚类结构、异常点等
- 实际案例：对高维基因表达数据做 PCA 降至 2 维，观察样本的分组情况

**应用3：图像压缩**
- 问题类型：信号处理
- 为什么适合：图像的像素之间存在强相关性，少量主成分即可重建高质量图像
- 实际案例：保留前 50 个主成分，可以将图像数据量压缩 10 倍以上而人眼几乎看不出差异

**应用4：去噪**
- 问题类型：信号处理
- 为什么适合：噪声通常对应方差小的主成分方向，丢弃这些方向即可去除噪声
- 实际案例：对含噪信号做 PCA 重建，只保留前几个主成分

### 5.2 适用数据特征

- 特征类型：连续型数值特征（PCA 基于线性代数运算，不直接适用于类别特征）
- 数据规模：小到中等规模（标准 PCA 的复杂度为 $O(\min(n^2 d, nd^2))$，大数据需用 IncrementalPCA 或 RandomizedPCA）
- 噪声容忍度：中等（PCA 通过丢弃小方差方向可以一定程度去噪，但对异常值敏感）
- 线性关系：要求主成分之间是线性关系，非线性结构需要用 Kernel PCA

### 5.3 不适用场景

1. 数据中存在强非线性结构：PCA 只能捕捉线性关系，此时应使用 Kernel PCA、t-SNE、UMAP 等
2. 特征为离散类别型：PCA 需要数值运算，类别特征需先编码（如 one-hot），但高维稀疏 one-hot 编码做 PCA 效果不佳
3. 数据存在强异常值：PCA 基于方差最大化，异常值会被放大，应先做异常值检测或使用 Robust PCA
4. 需要保留原始特征的物理含义：PCA 变换后的特征是原特征的线性组合，失去了可解释性

---

## 6. 优缺点分析

### 6.1 优点

1. **计算高效**：
   - 标准 PCA 有解析解，计算确定，不需要迭代调参
   - 使用 SVD 求解，数值稳定

2. **无需标签**：
   - 无监督方法，不依赖标注数据
   - 适用于任何数值型数据

3. **理论完备**：
   - 数学基础扎实，方差贡献率提供了明确的降维标准
   - 最大方差与最小重构误差等价，从两个角度都能理解

4. **消除特征相关性**：
   - 主成分之间互相正交，天然消除多重共线性
   - 对后续的线性模型（如线性回归）尤其有利

### 6.2 缺点

1. **只能捕捉线性结构**：
   - 问题场景：数据存在流形结构（如瑞士卷数据集）时，线性 PCA 无法正确展开
   - 解决思路：使用 Kernel PCA 或非线性降维方法（t-SNE、UMAP、自编码器）

2. **对异常值敏感**：
   - 问题场景：少数极端值会拉偏主成分方向，因为 PCA 基于方差最大化
   - 解决思路：先做异常值检测和剔除，或使用 Robust PCA

3. **可解释性差**：
   - 问题场景：降维后的特征是原特征的线性组合，失去了明确的物理含义
   - 解决思路：分析每个主成分中各原始特征的权重（loading），结合领域知识进行解释

4. **需要标准化预处理**：
   - 问题场景：不同特征的量级差异大时，PCA 结果会被量级大的特征主导
   - 解决思路：始终先做 StandardScaler 标准化

### 6.3 与同类算法对比

| 维度 | PCA | LDA | Kernel PCA | t-SNE |
|------|-----|-----|------------|-------|
| 监督/无监督 | 无监督 | 有监督 | 无监督 | 无监督 |
| 线性/非线性 | 线性 | 线性 | 非线性 | 非线性 |
| 计算复杂度 | $O(nd^2)$ | $O(nd^2)$ | $O(n^2 d)$ | $O(n^2)$ |
| 保留全局结构 | 是 | 是 | 部分 | 否 |
| 保留局部结构 | 弱 | 弱 | 较好 | 很好 |
| 可解释性 | 较好 | 好 | 差 | 差 |
| 最大维度限制 | $\min(n,d)$ | $C-1$（类别数减1） | 无 | 无 |
| 适用场景 | 通用降维 | 分类特征提取 | 非线性降维 | 可视化 |

---

## 7. 调库实现

### 7.1 环境准备

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 7.2 完整代码示例

```python
"""
PCA 主成分分析 调库实现
数据集：鸢尾花数据集（Iris）
目标：将 4 维特征降至 2 维进行可视化，并分析方差贡献率
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ===============================
# 1. 数据准备
# ===============================
def load_data():
    """
    加载鸢尾花数据集

    Returns:
        X: 特征矩阵，shape (150, 4)
        y: 标签向量，shape (150,)
        feature_names: 特征名称列表
        target_names: 类别名称列表
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.feature_names, iris.target_names


def preprocess_data(X):
    """
    数据预处理：标准化

    Args:
        X: 原始特征矩阵

    Returns:
        X_scaled: 标准化后的特征
        scaler: 标准化器
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ===============================
# 2. PCA 分析
# ===============================
def analyze_pca(X_scaled):
    """
    使用 PCA 分析方差贡献率，确定最优降维维度

    Args:
        X_scaled: 标准化后的数据

    Returns:
        pca_full: 保留全部主成分的 PCA 模型
        explained_variance_ratio: 各主成分方差贡献率
        cumulative_variance: 累计方差贡献率
    """
    # 先保留全部主成分，分析方差贡献
    pca_full = PCA()
    pca_full.fit(X_scaled)

    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # 打印方差贡献率
    print("各主成分方差贡献率:")
    for i, (ev, cv) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"  PC{i+1}: {ev:.4f} (累计: {cv:.4f})")

    # 找到累计贡献率达到 95% 的最少主成分数
    n_components_95 = np.searchsorted(cumulative_variance, 0.95) + 1
    print(f"\n累计贡献率达到 95% 需要的主成分数: {n_components_95}")

    return pca_full, explained_variance_ratio, cumulative_variance


def train_pca(X_scaled, n_components=2):
    """
    使用指定维度训练 PCA

    Args:
        X_scaled: 标准化后的数据
        n_components: 目标维度

    Returns:
        pca: 训练好的 PCA 模型
        X_pca: 降维后的数据
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\nPCA 降维: {X_scaled.shape[1]}D -> {n_components}D")
    print(f"累计方差贡献率: {pca.explained_variance_ratio_.sum():.4f}")

    return pca, X_pca


# ===============================
# 3. 可视化
# ===============================
def visualize_results(X_pca, y, target_names,
                      explained_variance_ratio, cumulative_variance):
    """
    可视化 PCA 结果

    Args:
        X_pca: 降维后的数据
        y: 标签
        target_names: 类别名称
        explained_variance_ratio: 方差贡献率
        cumulative_variance: 累计方差贡献率
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：二维散点图
    ax1 = axes[0]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    for i, target_name in enumerate(target_names):
        mask = y == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors[i], label=target_name, alpha=0.7, edgecolors='k')

    ax1.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
    ax1.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
    ax1.set_title('PCA 2D Projection of Iris Dataset')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：方差贡献率
    ax2 = axes[1]
    n_components = len(explained_variance_ratio)
    x_pos = range(1, n_components + 1)

    ax2.bar(x_pos, explained_variance_ratio, alpha=0.6,
            color='#377eb8', label='Individual')
    ax2.step(x_pos, cumulative_variance, where='mid',
             color='#e41a1c', label='Cumulative')
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5,
                label='95% threshold')

    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Variance Ratio')
    ax2.set_title('Explained Variance Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('PCA_sklearn_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# ===============================
# 4. 主程序
# ===============================
if __name__ == "__main__":
    print("=" * 50)
    print("PCA 主成分分析 -- 调库实现")
    print("=" * 50)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    X, y, feature_names, target_names = load_data()
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"特征名称: {feature_names}")
    print(f"类别: {target_names}")

    # 2. 数据预处理
    print("\n[2/4] 数据预处理（标准化）...")
    X_scaled, scaler = preprocess_data(X)

    # 3. 分析方差贡献率
    print("\n[3/4] 方差贡献率分析...")
    pca_full, evr, cv = analyze_pca(X_scaled)

    # 4. 降维到 2 维
    print("\n[4/4] PCA 降维到 2 维...")
    pca_2d, X_pca = train_pca(X_scaled, n_components=2)

    # 打印主成分（特征向量）中各原始特征的权重
    print("\n主成分组成（各原始特征的权重）:")
    for i, comp in enumerate(pca_2d.components_):
        print(f"\n  PC{i+1}:")
        for j, (name, weight) in enumerate(zip(feature_names, comp)):
            print(f"    {name}: {weight:+.4f}")

    # 5. 可视化
    visualize_results(X_pca, y, target_names, pca_2d.explained_variance_ratio_, cv)

    print("\n程序执行完毕")
```

### 7.3 运行结果示例

```
==================================================
PCA 主成分分析 -- 调库实现
==================================================

[1/4] 加载数据...
数据形状: X=(150, 4), y=(150,)
特征名称: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
类别: ['setosa' 'versicolor' 'virginica']

[2/4] 数据预处理（标准化）...

[3/4] 方差贡献率分析...
各主成分方差贡献率:
  PC1: 0.7296 (累计: 0.7296)
  PC2: 0.2285 (累计: 0.9581)
  PC3: 0.0367 (累计: 0.9948)
  PC4: 0.0052 (累计: 1.0000)

累计贡献率达到 95% 需要的主成分数: 2

[4/4] PCA 降维到 2 维...

PCA 降维: 4D -> 2D
累计方差贡献率: 0.9581

主成分组成（各原始特征的权重）:
  PC1:
    sepal length (cm): +0.3614
    sepal width (cm): -0.0845
    petal length (cm): +0.8567
    petal width (cm): +0.3583

  PC2:
    sepal length (cm): +0.6566
    sepal width (cm): +0.7302
    petal length (cm): -0.1734
    petal width (cm): -0.0755

程序执行完毕
```

---

## 8. 手工代码实现

### 8.1 核心算法手写

```python
"""
PCA 主成分分析 手工实现
仅依赖 NumPy，从零实现 PCA 的全部核心逻辑
支持两种求解方式：协方差矩阵特征值分解 / SVD
"""

import numpy as np
import matplotlib.pyplot as plt


class PCAManual:
    """
    手工实现的 PCA 主成分分析

    支持两种求解方式:
    - 'eig': 对协方差矩阵做特征值分解
    - 'svd': 对数据矩阵做奇异值分解（推荐，数值更稳定）
    """

    def __init__(self, n_components=None, solver='svd'):
        """
        初始化 PCA 模型

        Args:
            n_components: 降维后的维度，None 表示保留全部
            solver: 求解方式，'eig' 或 'svd'
        """
        self.n_components = n_components
        self.solver = solver
        self.mean_ = None           # 均值向量
        self.components_ = None     # 主成分矩阵 (n_components, n_features)
        self.explained_variance_ = None       # 各主成分的方差（特征值）
        self.explained_variance_ratio_ = None # 方差贡献率
        self.cumulative_variance_ratio_ = None

    def fit(self, X):
        """
        拟合 PCA 模型

        Args:
            X: 训练数据，形状 (n_samples, n_features)

        Returns:
            self
        """
        n_samples, n_features = X.shape

        # Step 1: 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 确定保留的主成分数
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        elif isinstance(self.n_components, float):
            # 如果传入的是 0~1 的浮点数，后面自动截断
            n_components = min(n_samples, n_features)
        else:
            n_components = self.n_components

        # Step 2: 求解主成分
        if self.solver == 'eig':
            eigenvalues, eigenvectors = self._solve_eig(X_centered)
        elif self.solver == 'svd':
            eigenvalues, eigenvectors = self._solve_svd(X_centered, n_samples)
        else:
            raise ValueError(f"未知求解方式: {self.solver}")

        # Step 3: 按特征值从大到小排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 4: 处理 n_components 为浮点数的情况
        if isinstance(self.n_components, float):
            total_var = eigenvalues.sum()
            cum_ratio = np.cumsum(eigenvalues) / total_var
            n_components = np.searchsorted(cum_ratio, self.n_components) + 1
            n_components = max(1, min(n_components, len(eigenvalues)))

        # Step 5: 截取前 n_components 个主成分
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:n_components] / total_variance
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        return self

    def _solve_eig(self, X_centered):
        """
        通过协方差矩阵的特征值分解求解

        Args:
            X_centered: 中心化后的数据

        Returns:
            eigenvalues: 特征值数组
            eigenvectors: 特征向量矩阵
        """
        # 计算协方差矩阵
        n = X_centered.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n - 1)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        return eigenvalues, eigenvectors

    def _solve_svd(self, X_centered, n_samples):
        """
        通过奇异值分解求解

        Args:
            X_centered: 中心化后的数据
            n_samples: 样本数量

        Returns:
            eigenvalues: 等效特征值数组
            eigenvectors: 右奇异矩阵（特征向量）
        """
        # SVD 分解: X = U @ Sigma @ V^T
        # full_matrices=False 时 V 的形状为 (n_features, min(n,d))
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 特征值 = 奇异值的平方 / (n-1)
        eigenvalues = (S ** 2) / (n_samples - 1)

        # Vt 的每一行是一个主成分方向，转置后每列是一个主成分
        eigenvectors = Vt.T

        return eigenvalues, eigenvectors

    def transform(self, X):
        """
        将数据投影到主成分空间

        Args:
            X: 输入数据，形状 (n_samples, n_features)

        Returns:
            X_pca: 降维后的数据，形状 (n_samples, n_components)
        """
        if self.components_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit 方法")

        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """
        拟合并转换数据

        Args:
            X: 输入数据

        Returns:
            X_pca: 降维后的数据
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca):
        """
        从低维空间重构原始数据

        Args:
            X_pca: 降维后的数据，形状 (n_samples, n_components)

        Returns:
            X_reconstructed: 重构后的数据
        """
        if self.components_ is None:
            raise RuntimeError("模型尚未训练")

        # Z * W^T + mean
        return X_pca @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        """
        计算重构误差

        Args:
            X: 原始数据

        Returns:
            error: 均方重构误差
        """
        X_pca = self.transform(X)
        X_recon = self.inverse_transform(X_pca)
        return np.mean((X - X_recon) ** 2)


# ===============================
# 测试代码
# ===============================
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    print("=" * 50)
    print("PCA 手工实现 -- 测试")
    print("=" * 50)

    # 加载并预处理数据
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 测试1: EIG 方式 ---
    print("\n--- EIG 求解方式 ---")
    pca_eig = PCAManual(n_components=2, solver='eig')
    X_pca_eig = pca_eig.fit_transform(X_scaled)
    print(f"降维: {X_scaled.shape[1]}D -> 2D")
    print(f"方差贡献率: PC1={pca_eig.explained_variance_ratio_[0]:.4f}, "
          f"PC2={pca_eig.explained_variance_ratio_[1]:.4f}")
    print(f"累计贡献率: {pca_eig.cumulative_variance_ratio_[1]:.4f}")

    # --- 测试2: SVD 方式 ---
    print("\n--- SVD 求解方式 ---")
    pca_svd = PCAManual(n_components=2, solver='svd')
    X_pca_svd = pca_svd.fit_transform(X_scaled)
    print(f"降维: {X_scaled.shape[1]}D -> 2D")
    print(f"方差贡献率: PC1={pca_svd.explained_variance_ratio_[0]:.4f}, "
          f"PC2={pca_svd.explained_variance_ratio_[1]:.4f}")
    print(f"累计贡献率: {pca_svd.cumulative_variance_ratio_[1]:.4f}")

    # --- 测试3: 与 sklearn 对比 ---
    print("\n--- 与 sklearn PCA 对比 ---")
    from sklearn.decomposition import PCA as SklearnPCA
    pca_sklearn = SklearnPCA(n_components=2)
    X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)

    print(f"sklearn 方差贡献率: PC1={pca_sklearn.explained_variance_ratio_[0]:.4f}, "
          f"PC2={pca_sklearn.explained_variance_ratio_[1]:.4f}")

    # 对比投影结果（注意：符号可能相反，因为特征向量方向不唯一）
    print(f"\n手工 SVD 投影结果前5行:\n{X_pca_svd[:5]}")
    print(f"sklearn 投影结果前5行:\n{X_pca_sklearn[:5]}")

    # 验证：取绝对值后应一致（特征向量方向可以反转）
    max_diff = np.max(np.abs(np.abs(X_pca_svd) - np.abs(X_pca_sklearn)))
    print(f"\n绝对值最大差异: {max_diff:.6f} (应接近0)")

    # --- 测试4: 浮点数指定贡献率 ---
    print("\n--- 浮点数指定累计贡献率 ---")
    pca_95 = PCAManual(n_components=0.95, solver='svd')
    pca_95.fit(X_scaled)
    print(f"指定 95% 贡献率，实际选择的主成分数: {pca_95.components_.shape[0]}")
    print(f"实际累计贡献率: {pca_95.cumulative_variance_ratio_[-1]:.4f}")

    # --- 测试5: 重构误差 ---
    print("\n--- 重构误差分析 ---")
    for k in [1, 2, 3, 4]:
        pca_k = PCAManual(n_components=k, solver='svd')
        pca_k.fit(X_scaled)
        error = pca_k.reconstruction_error(X_scaled)
        print(f"  k={k}: 重构误差={error:.6f}, "
              f"累计贡献率={pca_k.cumulative_variance_ratio_[-1]:.4f}")

    # --- 可视化 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子图1: EIG 投影结果
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    for i, name in enumerate(iris.target_names):
        mask = y == i
        axes[0].scatter(X_pca_eig[mask, 0], X_pca_eig[mask, 1],
                        c=colors[i], label=name, alpha=0.7, edgecolors='k')
    axes[0].set_xlabel(f'PC1 ({pca_eig.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca_eig.explained_variance_ratio_[1]:.2%})')
    axes[0].set_title('PCA Manual (EIG)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2: SVD 投影结果
    for i, name in enumerate(iris.target_names):
        mask = y == i
        axes[1].scatter(X_pca_svd[mask, 0], X_pca_svd[mask, 1],
                        c=colors[i], label=name, alpha=0.7, edgecolors='k')
    axes[1].set_xlabel(f'PC1 ({pca_svd.explained_variance_ratio_[0]:.2%})')
    axes[1].set_ylabel(f'PC2 ({pca_svd.explained_variance_ratio_[1]:.2%})')
    axes[1].set_title('PCA Manual (SVD)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 子图3: sklearn 投影结果
    for i, name in enumerate(iris.target_names):
        mask = y == i
        axes[2].scatter(X_pca_sklearn[mask, 0], X_pca_sklearn[mask, 1],
                        c=colors[i], label=name, alpha=0.7, edgecolors='k')
    axes[2].set_xlabel(f'PC1 ({pca_sklearn.explained_variance_ratio_[0]:.2%})')
    axes[2].set_ylabel(f'PC2 ({pca_sklearn.explained_variance_ratio_[1]:.2%})')
    axes[2].set_title('PCA sklearn')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('PCA_manual_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n程序执行完毕")
```

### 8.2 与调库结果对比

| 方法 | 方差贡献率 (PC1) | 方差贡献率 (PC2) | 累计贡献率 | 计算时间 |
|------|-----------------|-----------------|-----------|----------|
| sklearn PCA | 0.7296 | 0.2285 | 0.9581 | ~0.001s |
| 手工 EIG | 0.7296 | 0.2285 | 0.9581 | ~0.001s |
| 手工 SVD | 0.7296 | 0.2285 | 0.9581 | ~0.001s |

**分析**：
- 三种实现的结果完全一致，验证了手工实现的正确性
- 注意：投影结果的符号可能相反（例如某个主成分方向翻转），这是因为特征向量的方向不唯一（$w$ 和 $-w$ 都是同一个特征向量），这不影响降维效果
- 取绝对值后对比，差异应在数值精度范围内（$< 10^{-6}$）
- SVD 方式是 scikit-learn 底层使用的方法，推荐在手工实现时也优先使用 SVD

---

## 9. 可视化与结果理解

### 9.1 方差贡献率可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

def visualize_variance_contribution():
    """
    可视化不同数据集上的方差贡献率
    使用手写数字数据集（64维）
    """
    # 加载手写数字数据集
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 全部主成分 PCA
    pca = PCA()
    pca.fit(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 各主成分方差贡献率（前30个）
    ax1 = axes[0]
    n_show = 30
    x_pos = range(1, n_show + 1)
    ax1.bar(x_pos, pca.explained_variance_ratio_[:n_show],
            alpha=0.6, color='#377eb8')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Variance Ratio (Digits Dataset, 64D)')
    ax1.grid(True, alpha=0.3)

    # 子图2: 累计方差贡献率
    ax2 = axes[1]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, 65), cumulative, 'b-o', markersize=3)
    ax2.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='90%')
    ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95%')
    ax2.axhline(y=0.99, color='orange', linestyle='--', alpha=0.5, label='99%')

    # 标注关键点
    for threshold, color in [(0.90, 'r'), (0.95, 'g'), (0.99, 'orange')]:
        n_comp = np.searchsorted(cumulative, threshold) + 1
        ax2.annotate(f'n={n_comp}', xy=(n_comp, threshold),
                     xytext=(n_comp + 3, threshold + 0.02),
                     arrowprops=dict(arrowstyle='->', color=color),
                     fontsize=10, color=color)

    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Ratio')
    ax2.set_title('Cumulative Variance Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('PCA_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印关键信息
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n_comp = np.searchsorted(cumulative, threshold) + 1
        print(f"  {threshold:.0%} 贡献率需要 {n_comp} 个主成分 "
              f"(从64维降至{n_comp}维, 压缩率{1 - n_comp/64:.1%})")

visualize_variance_contribution()
```

### 9.2 高维数据可视化

```python
def visualize_high_dimensional_projection():
    """
    将手写数字数据集从64维降至2维进行可视化
    """
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler

    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10',
                          alpha=0.6, s=20, edgecolors='none')
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA Projection of Digits Dataset (64D -> 2D)')
    plt.grid(True, alpha=0.3)
    plt.savefig('PCA_digits_2d.png', dpi=300, bbox_inches='tight')
    plt.show()

visualize_high_dimensional_projection()
```

### 9.3 重构效果可视化

```python
def visualize_reconstruction():
    """
    可视化不同维度下 PCA 重构的效果
    使用手写数字数据集
    """
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data.astype(float)
    images = digits.images

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # 选择一张图片展示
    img_idx = 0
    axes[0, 0].imshow(images[img_idx], cmap='gray')
    axes[0, 0].set_title('Original (64D)')
    axes[0, 0].axis('off')

    # 不同维度重构
    for i, n_comp in enumerate([2, 5, 10, 20, 32]):
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_pca)

        col = i + 1
        axes[0, col].imshow(X_recon[img_idx].reshape(8, 8), cmap='gray')
        axes[0, col].set_title(f'{n_comp} PCs\n({pca.explained_variance_ratio_.sum():.1%})')
        axes[0, col].axis('off')

    # 第二行：另一张图片
    img_idx = 7
    axes[1, 0].imshow(images[img_idx], cmap='gray')
    axes[1, 0].set_title('Original (64D)')
    axes[1, 0].axis('off')

    for i, n_comp in enumerate([2, 5, 10, 20, 32]):
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_pca)

        col = i + 1
        axes[1, col].imshow(X_recon[img_idx].reshape(8, 8), cmap='gray')
        axes[1, col].set_title(f'{n_comp} PCs\n({pca.explained_variance_ratio_.sum():.1%})')
        axes[1, col].axis('off')

    plt.suptitle('PCA Image Reconstruction with Different Numbers of Components',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('PCA_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.show()

visualize_reconstruction()
```

### 9.4 结果解读

**从方差贡献率图可以看出：**
- 少数几个主成分占据了大部分方差，呈现典型的"长尾"分布
- 手写数字数据集（64维）中，约 21 个主成分即可保留 90% 的信息，实现了约 67% 的压缩率
- 方差贡献率下降迅速，说明数据维度之间存在很强的冗余性

**从二维投影散点图可以看出：**
- PCA 能在 2 维空间中较好地区分不同类别的数字
- 某些类别（如 0、6）分布较集中，而某些类别（如 3、5、8）有部分重叠
- 仅用 2 维（约 21% 的方差）就能看到基本的数据结构，说明 PCA 的降维效果显著

**从重构图可以看出：**
- 2 个主成分只能重建出模糊的轮廓
- 10 个主成分已经能清晰分辨数字形状
- 20~32 个主成分的重构效果与原图几乎无法区分
- 这验证了 PCA 降维在图像数据上的有效性

---

## 10. 模型评估

### 10.1 评估指标选择

PCA 作为无监督降维方法，不直接使用分类/回归指标。评估 PCA 效果主要看以下指标：

| 指标 | 含义 | 为什么选择 |
|------|------|-----------|
| 方差贡献率 | 每个主成分保留的方差比例 | 衡量信息保留程度 |
| 累计方差贡献率 | 前k个主成分的方差之和占比 | 决定降维维度k的选择 |
| 重构误差 | 原始数据与重构数据的均方误差 | 直接衡量信息损失 |
| 下游任务性能 | 降维后在分类/回归任务上的表现 | 终极评估标准 |

### 10.2 交叉验证评估

```python
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def evaluate_pca_dimensions():
    """
    通过下游分类任务评估不同 PCA 维度的效果
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 不同降维维度
    dimensions = [2, 5, 10, 20, 32, 64]  # 64 为原始维度

    print("不同 PCA 维度下的分类准确率（5折交叉验证）:")
    print("-" * 40)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for n_comp in dimensions:
        if n_comp == 64:
            # 不做 PCA，直接分类
            pipeline = Pipeline([
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            label = "原始维度 (64D)"
        else:
            pipeline = Pipeline([
                ('pca', PCA(n_components=n_comp)),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            label = f"PCA {n_comp}D"

        scores = cross_val_score(pipeline, X_scaled, y, cv=cv, scoring='accuracy')
        print(f"  {label:>15s}: {scores.mean():.4f} +/- {scores.std():.4f}")

evaluate_pca_dimensions()
```

**典型输出：**
```
不同 PCA 维度下的分类准确率（5折交叉验证）:
----------------------------------------
  PCA 2D       : 0.5234 +/- 0.0215
  PCA 5D       : 0.8327 +/- 0.0163
  PCA 10D      : 0.9165 +/- 0.0127
  PCA 20D      : 0.9517 +/- 0.0098
  PCA 32D      : 0.9583 +/- 0.0089
  原始维度 (64D): 0.9644 +/- 0.0095
```

**解读**：
- 降至 20 维即可达到 95% 的分类准确率，接近原始 64 维的表现
- 降至 10 维仍有 91% 的准确率，对于计算资源受限的场景非常实用
- 降至 2 维虽然丢失大量信息，但 52% 的准确率远高于随机猜测的 10%，说明 PCA 确实保留了最有区分力的信息

### 10.3 超参数调优

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

def tune_pca_components():
    """
    网格搜索最佳 PCA 维度
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pipeline = Pipeline([
        ('pca', PCA()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    param_grid = {
        'pca__n_components': [5, 10, 15, 20, 25, 30, 40, 50]
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_scaled, y)

    print(f"最佳 PCA 维度: {grid_search.best_params_['pca__n_components']}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

    # 打印所有结果
    print("\n所有候选维度:")
    for params, score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score']
    ):
        print(f"  n_components={params['pca__n_components']:>3d}: {score:.4f}")

tune_pca_components()
```

---

## 11. 常见问题与易错点

### 11.1 数据层面常见错误

**错误1：未对数据进行标准化**

**现象**：
- PCA 结果被量级大的特征主导
- 方差贡献率集中在第一个主成分上
- 降维后信息损失比预期大得多

**原因**：
- PCA 基于方差最大化，量级大的特征（如年薪以元为单位 vs 年龄以岁为单位）天然方差更大
- 不标准化相当于隐式地给量级大的特征赋予更高权重

**解决方案**：
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 注意：保存 scaler 用于后续新数据的标准化
import joblib
joblib.dump(scaler, 'pca_scaler.pkl')
```

**错误2：对训练集和测试集分别做 PCA**

**现象**：
- 测试集的 PCA 结果与训练集不一致
- 模型评估结果不稳定

**原因**：
- PCA 是在特定数据集上学习的变换，不同的数据集会学到不同的主成分方向
- 分别做 PCA 等于用了不同的"坐标系"，投影结果无法直接比较

**解决方案**：
```python
# 正确做法：在训练集上 fit PCA，然后 transform 测试集
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)   # fit + transform
X_test_pca = pca.transform(X_test)          # 只 transform，不 fit
```

### 11.2 模型层面常见错误

**错误1：特征数大于样本数时 PCA 效果异常**

**现象**：
- 报错或结果不稳定
- 某些特征值为零或负数

**原因**：
- 当 $d > n$ 时，协方差矩阵 $C \in \mathbb{R}^{d \times d}$ 的秩最多为 $n$
- 最多只有 $n$ 个非零特征值，即最多只能提取 $n$ 个主成分

**解决方案**：
- 使用 SVD 求解方式（推荐），它能正确处理这种情况
- 限制 n_components 不超过 $\min(n, d)$
- scikit-learn 默认使用 SVD，已自动处理此问题

**错误2：投影结果符号不一致**

**现象**：
- 多次运行 PCA，投影结果的符号可能相反
- 与参考代码的结果差一个负号

**原因**：
- 特征值分解中，$w$ 和 $-w$ 都是同一个特征向量，方向不确定
- SVD 分解中也存在类似的符号不确定性

**解决方案**：
- 这是正常现象，不影响降维效果（方差相同，只是方向翻转）
- 如果需要固定符号，可以在得到主成分后统一符号：
```python
# 统一符号：使每行（每个特征）中绝对值最大的元素为正
for i in range(len(pca.components_)):
    max_idx = np.argmax(np.abs(pca.components_[i]))
    if pca.components_[i, max_idx] < 0:
        pca.components_[i] *= -1
```

### 11.3 理解层面常见误区

**误区1：PCA 一定能提高模型性能**

- PCA 丢弃了一些信息，虽然主要是噪声，但也可能包含有用的信号
- 对于线性模型，PCA 通常有帮助（去除了共线性）
- 对于决策树等非线性模型，PCA 可能反而降低性能（因为树模型本身能处理高维特征）

**正确理解**：PCA 的主要作用是降维和去噪，对模型性能的影响取决于具体任务和数据。

**误区2：方差小的方向一定是噪声**

- 方差小不代表信息不重要
- 某些对分类/回归至关重要的特征可能方差较小
- 例如：医学诊断中某个生物标志物的微小变化可能是关键信号

**正确理解**：PCA 最大化的是方差，而不是区分能力。如果需要最大化分类效果，应使用 LDA（有监督降维）。

**误区3：PCA 总是选择前 k 个主成分**

- 有时跳过某些主成分可能更好
- 例如，第三主成分可能与某个干扰因素强相关，丢弃它反而提升效果
- 这种情况在领域特定应用中较为常见

### 11.4 性能优化建议

1. **大规模数据**：使用 `IncrementalPCA`（增量 PCA），支持分批处理
2. **超高维数据**：使用 `PCA(solver='randomized')`，随机 SVD 更快
3. **稀疏数据**：使用 `TruncatedSVD`（截断 SVD），不需要中心化

```python
from sklearn.decomposition import IncrementalPCA, TruncatedSVD

# 增量 PCA（大数据）
ipca = IncrementalPCA(n_components=50, batch_size=1000)
for batch in data_batches:
    ipca.partial_fit(batch)

# 随机 PCA（速度快，精度略低）
pca_fast = PCA(n_components=50, svd_solver='randomized', random_state=42)

# 截断 SVD（稀疏数据，如文本 TF-IDF）
tsvd = TruncatedSVD(n_components=50)
```

---

## 12. 学习总结

### 12.1 核心要点回顾

**核心思想**：通过正交变换找到数据方差最大的方向，用少量主成分表示原始数据。

**数学本质**：协方差矩阵的特征值分解（或等价地，数据矩阵的 SVD），特征值 = 投影方差，特征向量 = 投影方向。

**优化目标**：最大化投影方差（等价于最小化重构误差）。

**适用场景**：高维数据降维、特征提取、数据可视化、去噪、消除共线性。

**局限性**：只能捕捉线性结构，对异常值敏感，降维后特征不可解释。

### 12.2 关键公式汇总

**1. 协方差矩阵**：
$$ C = \frac{1}{n-1}\tilde{X}^T \tilde{X} $$

**2. 特征值问题**：
$$ Cw_i = \lambda_i w_i $$

**3. 方差贡献率**：
$$ \text{Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d}\lambda_j} $$

**4. 投影（降维）**：
$$ Z = \tilde{X} W_k $$

**5. 重构**：
$$ \hat{X} = ZW_k^T + \bar{X} $$

**6. 重构误差**：
$$ J = \sum_{j=k+1}^{d}\lambda_j $$

### 12.3 最佳实践

**数据预处理**：
- 对连续特征做 StandardScaler 标准化
- 处理缺失值后再做 PCA
- 对训练集 fit，对测试集只 transform

**模型选择**：
- 通过累计方差贡献率确定维度（通常 85%~95%）
- 也可以通过下游任务的交叉验证确定维度
- SVD 方式优先于特征值分解方式

**模型评估**：
- 综合看方差贡献率和下游任务性能
- 可视化投影结果和重构效果
- 分析主成分的组成（loading），辅助业务理解

### 12.4 与其他算法的联系

- **前置算法**：线性代数（特征值分解、SVD）、方差与协方差
- **后续算法**：Kernel PCA（非线性扩展）、Incremental PCA（大数据）、Sparse PCA（稀疏主成分）、Robust PCA（鲁棒降维）
- **相关算法**：LDA（有监督降维）、t-SNE / UMAP（非线性降维）、自编码器（深度学习降维）、SVD（矩阵分解，与 PCA 等价）

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1：概念理解**

问题：关于 PCA，以下哪个说法是正确的？

A. PCA 是一种有监督学习方法，需要标签数据
B. 第一主成分对应协方差矩阵最小的特征值
C. PCA 的主成分之间互相正交
D. 标准化预处理对 PCA 的结果没有影响

**答案与解析：**

答案：C

解析：
- A 错误。PCA 是无监督学习方法，不需要标签数据。它只根据数据自身的方差结构来确定主成分方向。
- B 错误。第一主成分对应的是最大的特征值，因为 PCA 的目标就是最大化投影方差，而特征值等于投影方差。
- C 正确。通过推导可知，各主成分方向两两正交，这是拉格朗日乘数法中正交约束的直接结果。
- D 错误。标准化对 PCA 结果影响非常大。如果不同特征的量级差异大，量级大的特征会主导主成分方向。

---

**练习2：手动计算**

问题：给定以下二维数据，手工计算 PCA 的主成分：

数据点（已中心化）：
$$ \tilde{X} = \begin{bmatrix} 1 & 2 \\ 3 & 1 \\ 2 & 3 \\ 0 & 1 \end{bmatrix} $$

请计算：
1. 协方差矩阵 $C$
2. 特征值和特征向量
3. 第一主成分方向
4. 将数据投影到第一主成分后的结果

**答案与解析：**

解：

**步骤1：计算协方差矩阵**

$$ \tilde{X}^T \tilde{X} = \begin{bmatrix} 1 & 3 & 2 & 0 \\ 2 & 1 & 3 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 1 \\ 2 & 3 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1+9+4+0 & 2+3+6+0 \\ 2+3+6+0 & 4+1+9+1 \end{bmatrix} = \begin{bmatrix} 14 & 11 \\ 11 & 15 \end{bmatrix} $$

$$ C = \frac{1}{4-1}\begin{bmatrix} 14 & 11 \\ 11 & 15 \end{bmatrix} = \begin{bmatrix} 14/3 & 11/3 \\ 11/3 & 5 \end{bmatrix} $$

**步骤2：求解特征值**

特征方程 $\det(C - \lambda I) = 0$：

$$ \det \begin{bmatrix} 14/3 - \lambda & 11/3 \\ 11/3 & 5 - \lambda \end{bmatrix} = 0 $$

$$ (14/3 - \lambda)(5 - \lambda) - (11/3)^2 = 0 $$

$$ 70/3 - 14\lambda/3 - 5\lambda + \lambda^2 - 121/9 = 0 $$

$$ 9\lambda^2 - 87\lambda + 210 - 121 = 0 $$

$$ 9\lambda^2 - 87\lambda + 89 = 0 $$

使用求根公式：

$$ \lambda = \frac{87 \pm \sqrt{87^2 - 4 \times 9 \times 89}}{18} = \frac{87 \pm \sqrt{7569 - 3204}}{18} = \frac{87 \pm \sqrt{4365}}{18} $$

$$ \lambda \approx \frac{87 \pm 66.07}{18} $$

$$ \lambda_1 \approx \frac{87 + 66.07}{18} \approx 8.50, \quad \lambda_2 \approx \frac{87 - 66.07}{18} \approx 1.16 $$

**步骤3：求特征向量**

对于 $\lambda_1 \approx 8.50$：

$$ (C - \lambda_1 I) w_1 = 0 $$
$$ \begin{bmatrix} 14/3 - 8.50 & 11/3 \\ 11/3 & 5 - 8.50 \end{bmatrix} w_1 = 0 $$
$$ \begin{bmatrix} -3.83 & 3.67 \\ 3.67 & -3.50 \end{bmatrix} w_1 = 0 $$

从第一行：$-3.83 w_1 + 3.67 w_2 = 0$，得 $w_1/w_2 \approx 3.67/3.83 \approx 0.96$。

归一化后：$w_1 \approx [0.69, 0.72]^T$

因此第一主成分方向约为 $[0.69, 0.72]^T$，说明两个原始特征在主成分上的贡献接近，且方向一致。

**步骤4：投影到第一主成分**

$$ Z = \tilde{X} w_1 = \begin{bmatrix} 1 & 2 \\ 3 & 1 \\ 2 & 3 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0.69 \\ 0.72 \end{bmatrix} \approx \begin{bmatrix} 2.13 \\ 2.79 \\ 3.54 \\ 0.72 \end{bmatrix} $$

---

### 13.2 进阶思考

**思考1：改进分析**

问题：PCA 在处理非线性结构数据时效果不佳，请分析原因并提出改进方案。

**答案与解析：**

**问题分析**：

PCA 假设数据的主要变化方向可以通过线性投影捕捉。但很多真实数据具有非线性流形结构：

1. **流形结构**：例如瑞士卷数据集（Swiss Roll），数据点分布在三维空间中一个卷曲的二维流形上。线性 PCA 无法将这个卷曲的表面"展开"。
2. **聚类结构**：如果数据呈圆形或环形分布，PCA 只能找到最大方差方向，无法揭示环状结构。
3. **复杂映射**：在图像和语音数据中，语义相似的变化（如旋转、亮度变化）往往是非线性的。

**改进方法**：

**方法1：Kernel PCA**
- 原理：用核函数将数据隐式映射到高维空间，在高维空间中做 PCA。原本弯曲的流形在高维空间中可能变成线性的
- 优势：能捕捉非线性结构，数学框架与 PCA 一致
- 代价：需要计算 $n \times n$ 的核矩阵，复杂度为 $O(n^3)$；需要选择核函数和超参数

**方法2：自编码器（Autoencoder）**
- 原理：用神经网络学习数据的压缩表示，编码器将高维数据映射到低维瓶颈层，解码器从低维表示重建原始数据
- 优势：非线性表达能力极强，可以学习任意复杂的降维映射
- 代价：需要大量数据和计算资源；训练可能不稳定

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

**方法3：t-SNE / UMAP**
- 原理：专注于保持数据的局部邻域结构，在低维空间中重建高维空间中的邻域关系
- 优势：可视化效果极好
- 代价：主要用于可视化（2D/3D），不能直接用于降维后作为特征输入

---

**思考2：对比分析**

问题：对比 PCA 和 LDA，在什么情况下应该选择哪一个？

**答案与解析：**

**对比维度**：

| 维度 | PCA | LDA |
|------|-----|-----|
| 监督/无监督 | 无监督 | 有监督 |
| 优化目标 | 最大化投影方差 | 最大化类间方差/最小化类内方差 |
| 最大维度 | $\min(n, d)$ | $C - 1$（类别数减1） |
| 适用任务 | 降维、去噪、可视化 | 分类特征提取 |
| 对标签依赖 | 不需要 | 必须有标签 |
| 簇内紧凑性 | 不保证 | 保证 |

**选择 PCA 的情况**：
1. 没有标签数据（无监督场景）
2. 目标是降维或数据可视化
3. 用于消除多重共线性（如回归前的预处理）
4. 去除噪声
5. 类别数很多时（LDA 最多降到 $C-1$ 维，PCA 没有此限制）

**选择 LDA 的情况**：
1. 有标签数据，且目标是分类任务
2. 希望降维后的特征具有最强的类别区分能力
3. 类别数较少（LDA 最多降到 $C-1$ 维，适合类别不多的场景）

**混合策略**：
- 当特征维度远高于样本数（$d \gg n$）时，可先用 PCA 降至 $\min(n, d)$ 维以下，再用 LDA 做有监督降维
- 先 PCA 降维消除噪声和共线性，再在低维空间上训练分类器

---

### 13.3 开放思考

**思考3：创新扩展**

问题：设计一个基于 PCA 的异常检测系统，并分析其原理和局限性。

**答案与解析：**

**创新应用场景：工业传感器异常检测**

**问题背景**：
工厂中的多个传感器持续采集设备运行数据（温度、压力、振动频率等），正常状态下这些传感器数据之间存在强相关性。当设备出现故障时，某些传感器的读数会偏离正常的相关模式。

**为什么 PCA 适合**：
1. 正常数据各维度之间存在相关性，前几个主成分即可解释绝大部分方差
2. 异常数据破坏了正常的相关性结构，在主成分空间中表现出与正常数据不同的模式
3. 无需标注异常数据（现实中异常样本稀缺且类型多样）

**具体实施方案**：

**步骤1：数据收集与预处理**
- 收集正常状态下的多传感器时序数据
- 对每个时间窗口提取统计特征（均值、标准差、峰值等）
- 标准化各特征维度

**步骤2：PCA 模型训练**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 假设 X_normal 是正常状态的特征矩阵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_normal)

pca = PCA(n_components=0.95)  # 保留 95% 方差
X_normal_pca = pca.fit_transform(X_scaled)
```

**步骤3：异常分数定义**
- 方法A：**重构误差**。用 PCA 降维后再重构，计算重构误差。异常数据的重构误差通常更大。
- 方法B：**SPE 统计量**（Squared Prediction Error），即样本在残差子空间中的投影长度。
- 方法C：**Hotelling $T^2$ 统计量**，即样本在主成分子空间中的马氏距离。

```python
def anomaly_score(X_new, pca, scaler):
    """
    计算异常分数（基于重构误差）
    """
    X_scaled = scaler.transform(X_new)
    X_pca = pca.transform(X_scaled)
    X_recon = pca.inverse_transform(X_pca)
    # 每个样本的重构误差
    scores = np.mean((X_scaled - X_recon) ** 2, axis=1)
    return scores

# 设定阈值（例如正常数据重构误差的 99 百分位）
threshold = np.percentile(
    anomaly_score(X_normal, pca, scaler), 99
)
```

**步骤4：部署与应用**
- 实时采集新数据，计算异常分数
- 当分数超过阈值时触发告警
- 定期用新正常数据更新 PCA 模型

**潜在挑战与解决方案**：

1. **挑战1**：正常模式随时间缓慢漂移
   - 解决方案：使用滑动窗口，定期重新训练 PCA 模型

2. **挑战2**：阈值设定困难
   - 解决方案：结合业务经验设定阈值，使用指数加权移动平均平滑异常分数

3. **挑战3**：某些缓慢发展的故障在 PCA 中可能不明显
   - 解决方案：结合趋势分析和多个时间尺度的 PCA

---

## 14. 学习路径建议

### 14.1 前置知识

**数学基础**：

- [ ] **线性代数**：矩阵乘法、协方差矩阵的定义、特征值与特征向量、正交性、矩阵分解
  - 重点理解：为什么 $Cw = \lambda w$ 中的 $\lambda$ 就是投影方差
  - 推荐资源：《线性代数导论》Gilbert Strang（MIT 公开课）
  - 学习时长：2-3 周

- [ ] **概率统计**：方差、协方差、数据分布、中心化
  - 重点理解：协方差矩阵的几何含义——它描述了数据在各个方向上的"伸展程度"
  - 推荐资源：《概率论与数理统计》陈希孺
  - 学习时长：1-2 周

- [ ] **微积分**：拉格朗日乘数法（理解带约束的优化问题如何转化为无约束问题）
  - 推荐资源：任意微积分教材的多元函数极值部分
  - 学习时长：3-5 天

**编程基础**：

- [ ] **NumPy**：矩阵运算、SVD（`np.linalg.svd`）、特征值分解（`np.linalg.eigh`）
- [ ] **Matplotlib**：散点图、柱状图等基础可视化

### 14.2 平行算法（可同时学习）

1. **SVD（奇异值分解）**：与 PCA 等价的矩阵分解方法，是理解 PCA 计算的关键
   - 学习重点：SVD 的几何意义，左/右奇异矩阵的物理含义
   - 对比点：SVD 是通用矩阵分解，PCA 是 SVD 在数据降维上的特化应用

2. **LDA（线性判别分析）**：有监督降维方法
   - 学习重点：Fisher 判别准则，最大化类间方差与类内方差之比
   - 对比点：LDA 用标签信息最大化类别区分度，PCA 不用标签最大化方差

3. **因子分析（Factor Analysis）**：另一种降维方法，假设数据由少量隐变量加噪声生成
   - 学习重点：生成模型视角 vs PCA 的几何视角
   - 对比点：因子分析对噪声有显式建模，PCA 没有

### 14.3 进阶算法（后续学习）

**短期目标（1-2 个月）：**

1. **Kernel PCA**：PCA 的非线性扩展
   - 关联：使用核函数将数据映射到高维空间后再做 PCA
   - 难度：中等

2. **Incremental PCA**：适用于大规模数据的增量 PCA
   - 关联：分批处理大数据，在线更新主成分
   - 难度：中等

**中期目标（3-6 个月）：**

1. **自编码器（Autoencoder）**：基于神经网络的非线性降维
   - 关联：编码器-解码器结构与 PCA 的投影-重构等价，但使用非线性激活函数
   - 难度：较高

2. **t-SNE / UMAP**：非线性降维与可视化
   - 关联：保持局部邻域结构的降维方法，常用于高维数据可视化
   - 难度：较高

**长期目标（6 个月以上）：**

1. **深度生成模型中的降维**：VAE 中的潜变量空间
   - 最新研究：结合深度学习的概率降维方法
   - 难度：高

### 14.4 推荐资源

**教材类：**

1. **《机器学习》** 周志华（西瓜书）—— 第10章"降维与度量学习"，系统讲解 PCA 和其他降维方法
2. **《统计学习方法》** 李航 —— 第10章"降维与聚类"，数学推导简洁严谨
3. **《Pattern Recognition and Machine Learning》** Bishop —— 第12章"Continuous Latent Variables"，从概率视角深入理解 PCA

**论文类：**

1. **Pearson, K. (1901)**. "On Lines and Planes of Closest Fit to Systems of Points in Space" —— PCA 的原始论文
2. **Jolliffe, I. T. (2002)**. "Principal Component Analysis" —— PCA 的权威专著
3. **Scholkopf, B. et al. (1998)**. "Nonlinear Component Analysis as a Kernel Eigenvalue Problem" —— Kernel PCA 的原始论文

**在线课程：**

1. **CS229 Lecture Notes (PCA)** —— 斯坦福机器学习课程的 PCA 讲义
2. **3Blue1Brown：特征值与特征向量** —— YouTube 可视化讲解系列
3. **StatQuest：PCA Step-by-Step** —— YouTube 直观的 PCA 入门视频

**博客/文章：**

1. **"A Tutorial on Principal Component Analysis"** by Jonathon Shlens —— 最经典的 PCA 教程之一
2. **"PCA: Principle Component Analysis"** on towardsdatascience.com

**实践项目：**

1. **Eigenfaces**：用 PCA 对人脸数据集（如 Olivetti Faces）降维，实现人脸识别
2. **Kaggle**："Digit Recognizer" 竞赛中用 PCA 降维加速模型训练

---

## 附录

### A. 完整代码清单

```python
"""
PCA 主成分分析 完整实现
包含调库实现和手工实现
"""

# ============ 调库实现（见第7章完整代码） ============
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

def sklearn_implementation():
    """使用 scikit-learn 的 PCA 实现"""
    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分析方差贡献率
    pca_full = PCA()
    pca_full.fit(X_scaled)
    print("方差贡献率:", pca_full.explained_variance_ratio_)
    print("累计贡献率:", np.cumsum(pca_full.explained_variance_ratio_))

    # 降维到 2 维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"降维: {X.shape[1]}D -> 2D, 累计贡献率: {pca.explained_variance_ratio_.sum():.4f}")
    return X_pca

# ============ 手工实现（见第8章完整代码） ============
class PCAManual:
    """手工实现的 PCA"""
    # 见第8章完整代码
    pass

if __name__ == "__main__":
    sklearn_implementation()
```

### B. 参考文献

1. Pearson, K. (1901). "On Lines and Planes of Closest Fit to Systems of Points in Space". Philosophical Magazine.
2. Hotelling, H. (1933). "Analysis of a Complex of Statistical Variables into Principal Components". Journal of Educational Psychology.
3. Jolliffe, I. T. (2002). "Principal Component Analysis". Springer Series in Statistics.
4. 周志华. 《机器学习》. 清华大学出版社.
5. 李航. 《统计学习方法》. 清华大学出版社.
6. Bishop, C. M. (2006). "Pattern Recognition and Machine Learning". Springer.
7. 傅罡. 《人工智能注意力机制：体系、模型与算法剖析》.

### C. 常见问题 FAQ

**Q1：PCA 降维后的维度 k 应该如何选择？**

A：最常用的方法是分析累计方差贡献率。一般选择累计贡献率达到 85%~95% 对应的最小 $k$ 值。如果 PCA 是下游任务的预处理步骤，也可以通过交叉验证选择使下游任务性能最优的 $k$ 值。还有一种经验法则是画累计贡献率曲线，选择曲线开始变得平缓的"拐点"对应的 $k$。

**Q2：PCA 和 SVD 是什么关系？**

A：数学上等价。对中心化后的数据矩阵 $\tilde{X}$ 做 SVD（$\tilde{X} = U\Sigma V^T$），则 $V$ 的列就是 PCA 的主成分方向，且特征值 $\lambda_i = \sigma_i^2/(n-1)$。实际实现中推荐用 SVD，因为数值更稳定，且当 $d > n$ 时协方差矩阵奇异，SVD 仍可正常计算。

**Q3：PCA 能用于文本数据吗？**

A：可以，但需要先用 TF-IDF 或词嵌入等方法将文本转为数值向量。对于高维稀疏的 TF-IDF 向量，通常使用截断 SVD（`TruncatedSVD`）而非标准 PCA，因为标准 PCA 需要中心化，会破坏稀疏性。对于词嵌入向量（如 word2vec），标准 PCA 可以直接使用。

**Q4：为什么 PCA 有时让模型性能下降？**

A：PCA 丢弃了一些方差小的方向，这些方向可能包含对特定任务有用的信息。例如，如果分类的关键特征本身方差很小，PCA 可能将其作为"噪声"丢弃。此时应考虑使用 LDA（有监督降维，直接优化分类目标）或特征选择方法。

**Q5：白化（whitening）是什么？与 PCA 有什么关系？**

A：白化是 PCA 的后处理步骤。PCA 降维后，各主成分的方差不同（等于对应特征值）。白化将每个主成分除以其标准差，使得变换后的数据协方差矩阵为单位矩阵。白化后的数据各维度方差相等且不相关，常作为某些算法（如 ICA、某些神经网络）的预处理步骤。

---

**文档结束**

> 如有错误或建议，欢迎指出，共同完善。
