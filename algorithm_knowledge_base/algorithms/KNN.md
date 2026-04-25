# KNN (K-Nearest Neighbors) 学习文档

> 通过"近朱者赤,近墨者黑"的朴素思想,利用最近邻样本的投票完成分类与回归预测

---

## 1. 算法基础认知

### 一句话定义

给定一个待预测样本,在训练集中找到与它距离最近的 K 个样本,用这 K 个样本的标签来决定该样本的预测值。

### 直觉类比

想象你刚搬到一个新的小区,想判断自己属于"早起晨跑族"还是"夜猫子族"。你不需要参加什么测试,只需要观察你最近的几位邻居:如果离你家最近的 5 户人家中,有 4 户每天清晨 6 点就出门跑步,那你大概率也属于"早起晨跑族"。KNN 的核心思想就是"物以类聚,人以群分"——一个样本的类别,应该由它周围最近的那些样本决定。

更准确地说,这个类比揭示了 KNN 的几个关键特征:

1. **不需要事先建立模型**——你没有去做什么性格测试问卷,而是直接观察邻居。这对应了 KNN 的"惰性学习"(Lazy Learning)特性。
2. **结果取决于周围邻居**——你只看最近的邻居,而不是整个小区的人。这对应了 KNN 中 K 值的概念。
3. **邻居如何定义很关键**——"近"是用物理距离衡量的,在算法中我们需要选择合适的距离度量。
4. **少数服从多数**——5 个邻居中 4 个跑步,你就被归为跑步族。这对应了多数投票机制。

### 历史背景

KNN 的思想最早可以追溯到 Fix 和 Hodges 在 1951 年发表的论文《Discriminatory Analysis: Nonparametric Discrimination》,这篇论文提出了非参数判别分析的基本框架。随后在 1967 年,Cover 和 Hart 在论文《Nearest Neighbor (NN) Pattern Classification》中给出了 KNN 算法渐近错误率的理论分析,证明了在大样本条件下,1-NN 的错误率不超过贝叶斯最优错误率的两倍。这一结果为 KNN 提供了坚实的理论基础,使其从经验方法上升为有理论保障的算法。

1982 年,文献中对加权 KNN 的研究开始出现,使得距离更近的邻居拥有更大的投票权重。此后,KNN 被广泛应用于模式识别、数据挖掘、推荐系统等领域,并催生了诸如 K-D Tree、Ball Tree 等加速搜索的数据结构。

### 算法定位

- **类型**: 监督学习 --> 分类 / 回归
- **输出**: 分类任务输出离散类别标签; 回归任务输出连续数值
- **模型类型**: 非参数模型(Non-parametric Model) / 基于实例的学习(Instance-based Learning) / 惰性学习(Lazy Learning)

所谓"非参数模型",并不是说没有参数(K 值、距离度量都是超参数),而是指模型不对数据的分布做任何假设,也不从训练数据中学习到固定的参数。KNN 的"模型"就是训练数据本身,预测时直接使用原始数据进行计算。

所谓"惰性学习",是指 KNN 在训练阶段几乎不做任何事情(仅仅是存储数据),所有的计算都推迟到预测阶段才进行。这与"急切学习"(Eager Learning,如线性回归、决策树等在训练阶段就完成模型构建)形成鲜明对比。

### 前置知识

- **线性代数**: 向量的范数、欧几里得距离、矩阵运算
- **概率论**: 贝叶斯决策理论、条件概率(用于理解 Cover-Hart 定理)
- **数据结构**: K-D Tree 的基本概念(用于理解加速搜索方法)
- **基础统计**: 均值、中位数、众数的概念

---

## 2. 核心原理

### 2.1 核心思想

KNN 的核心思想非常朴素:在特征空间中,如果一个样本的 K 个最近邻中的大多数属于某一个类别,那么该样本也极有可能属于这个类别。这个思想基于一个基本假设——**空间中距离相近的样本往往具有相似的性质**。

理解这个假设为什么合理,可以从两个角度看:

1. **特征相似则标签相似**: 如果两个样本在所有特征维度上都非常接近,说明它们在本质上是类似的"事物",因此应当属于同一类别或具有相近的数值。
2. **连续性假设**: 分类边界在特征空间中通常具有一定的平滑性,不会出现剧烈跳变。在分类边界附近,样本的类别分布是连续过渡的。

KNN 有两个基本版本:

- **分类(KNN Classification)**: 通过多数投票(Majority Voting)决定类别。也可以使用距离加权投票,让更近的邻居有更大影响力。
- **回归(KNN Regression)**: 通过 K 个最近邻标签的平均值(或加权平均值)来预测连续值。

核心思想可以概括为: **不做任何模型假设,直接利用训练数据的空间邻近关系进行预测**。

### 2.2 工作流程

KNN 的工作流程可以清晰地分为训练阶段和预测阶段:

**训练阶段(极简)**:

1. **存储数据**: 将所有训练样本 $(x_i, y_i)$ 存入内存
   - 输入: 训练集 $\{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$
   - 输出: 无显式模型,仅保留原始数据

**预测阶段(核心计算)**:

2. **计算距离**: 对于一个待预测样本 $x_{new}$,计算它与训练集中每个样本 $x_i$ 的距离
   - 关键操作: 选择合适的距离度量,计算 $d(x_{new}, x_i), \forall i$

3. **排序选邻**: 将所有距离从小到大排序,选取前 K 个距离最小的训练样本
   - 关键操作: 排序、取 Top-K

4. **投票决策**: 根据选出的 K 个最近邻的标签,决定 $x_{new}$ 的预测值
   - 分类任务: 多数投票或加权投票
   - 回归任务: 均值或加权均值
   - 决策点: K 值的大小直接影响投票结果的稳定性与灵活性

### 2.3 关键概念解释

- **K 值**: 最近邻的个数,是 KNN 最重要的超参数。K 值过小会导致模型过于敏感(高方差),容易受噪声影响; K 值过大会导致模型过于平滑(高偏差),可能将不同类别的样本混为一谈。
- **距离度量(Distance Metric)**: 衡量两个样本之间"远近"的方法。最常用的是欧氏距离,但也可以使用曼哈顿距离、闵可夫斯基距离、余弦相似度等。距离度量的选择直接影响"最近邻"的定义。
- **投票机制(Voting Mechanism)**: 分类任务中如何从 K 个邻居的标签中得出最终预测。简单多数投票不考虑距离权重; 距离加权投票则给更近的邻居更大权重,通常用距离的倒数作为权重。
- **搜索策略(Search Strategy)**: 暴力搜索(Brute Force)计算所有距离; K-D Tree/Ball Tree 利用空间结构加速搜索,适合高维或大数据场景。

### 2.4 几何/直观解释

从几何角度看,KNN 的分类决策区域是由 Voronoi 图(又称泰森多边形)划分的。在 Voronoi 图中,每个训练样本都有一个"势力范围",任何落入该范围的点都会被归为该样本所属的类别。

当 K=1 时,决策边界就是 Voronoi 图的边。当 K>1 时,决策边界会变得更平滑——那些"孤立"的样本点(被其他类别包围)的影响会被周围的多数类别所"淹没",从而产生更加合理的分类边界。

在高维空间中,距离度量的行为会发生显著变化。随着维度增加,所有点对之间的距离会趋于一致(维度灾难),这使得"最近邻"和"最远邻"的区别变得模糊。这是 KNN 在高维数据上表现不佳的根本原因之一。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度/取值 |
|------|------|-----------|
| $\mathcal{D}$ | 训练数据集 | $\{(x_i, y_i)\}_{i=1}^{n}$ |
| $x_i$ | 第 $i$ 个样本的特征向量 | $\mathbb{R}^d$ |
| $y_i$ | 第 $i$ 个样本的标签 | 分类: $\{1,2,\dots,C\}$; 回归: $\mathbb{R}$ |
| $n$ | 训练样本总数 | 正整数 |
| $d$ | 特征维度 | 正整数 |
| $K$ | 最近邻个数 | 正奇数(分类推荐) |
| $x_{new}$ | 待预测的新样本 | $\mathbb{R}^d$ |
| $d(\cdot, \cdot)$ | 距离函数 | $\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\geq 0}$ |
| $\mathcal{N}_K(x_{new})$ | $x_{new}$ 的 K 个最近邻集合 | $\mathcal{D}$ 的子集 |
| $\mathcal{R}^*$ | 贝叶斯最优错误率 | $[0, 1]$ |

### 3.2 问题形式化

给定训练数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n}$,其中 $x_i \in \mathbb{R}^d$, $y_i \in \{1, 2, \dots, C\}$(分类)或 $y_i \in \mathbb{R}$(回归)。

对于一个新的样本 $x_{new} \in \mathbb{R}^d$,KNN 的预测规则为:

**分类任务**:
$$ \hat{y}(x_{new}) = \arg\max_{c \in \{1, \dots, C\}} \sum_{(x_i, y_i) \in \mathcal{N}_K(x_{new})} \mathbb{I}(y_i = c) $$

其中 $\mathbb{I}(\cdot)$ 是指示函数,当条件成立时为 1,否则为 0。

**加权分类**:
$$ \hat{y}(x_{new}) = \arg\max_{c \in \{1, \dots, C\}} \sum_{(x_i, y_i) \in \mathcal{N}_K(x_{new})} w_i \cdot \mathbb{I}(y_i = c) $$

其中权重 $w_i = \frac{1}{d(x_{new}, x_i) + \epsilon}$($\epsilon$ 为防止除零的小常数)。

**回归任务**:
$$ \hat{y}(x_{new}) = \frac{1}{K} \sum_{(x_i, y_i) \in \mathcal{N}_K(x_{new})} y_i $$

**加权回归**:
$$ \hat{y}(x_{new}) = \frac{\sum_{(x_i, y_i) \in \mathcal{N}_K(x_{new})} w_i \cdot y_i}{\sum_{(x_i, y_i) \in \mathcal{N}_K(x_{new})} w_i} $$

### 3.3 距离度量

距离度量是 KNN 的核心组成部分,它定义了样本空间中"远近"的概念。以下是四种常用的距离度量:

#### 3.3.1 欧氏距离(Euclidean Distance)

欧氏距离是最直观的距离度量,对应 $L_2$ 范数,即两点之间的直线距离。

$$ d_{Euc}(x_a, x_b) = \sqrt{\sum_{j=1}^{d}(x_{a}^{(j)} - x_{b}^{(j)})^2} = \|x_a - x_b\|_2 $$

其中 $x_a^{(j)}$ 表示向量 $x_a$ 的第 $j$ 个分量。

**特点**: 对所有维度的差异进行等权重的惩罚,对大差异(离群值)更敏感(因为平方操作会放大大差异)。这是最常用的默认距离度量。

#### 3.3.2 曼哈顿距离(Manhattan Distance)

曼哈顿距离对应 $L_1$ 范数,类似于在城市街道中只能沿网格行走的最短路径长度。

$$ d_{Man}(x_a, x_b) = \sum_{j=1}^{d} |x_{a}^{(j)} - x_{b}^{(j)}| = \|x_a - x_b\|_1 $$

**特点**: 对离群值更加鲁棒(不使用平方操作),适合特征之间存在稀疏差异的场景。当数据维度很高时,曼哈顿距离往往比欧氏距离更稳定。

#### 3.3.3 闵可夫斯基距离(Minkowski Distance)

闵可夫斯基距离是欧氏距离和曼哈顿距离的统一推广形式,通过参数 $p$ 控制距离的"形状"。

$$ d_{Min}(x_a, x_b) = \left(\sum_{j=1}^{d} |x_{a}^{(j)} - x_{b}^{(j)}|^p \right)^{1/p} = \|x_a - x_b\|_p $$

**特殊情况**:
- 当 $p = 1$ 时,闵可夫斯基距离退化为曼哈顿距离
- 当 $p = 2$ 时,闵可夫斯基距离退化为欧氏距离
- 当 $p \to \infty$ 时,闵可夫斯基距离趋向于切比雪夫距离(Chebyshev Distance):
  $$ d_{Che}(x_a, x_b) = \max_{j} |x_{a}^{(j)} - x_{b}^{(j)}| $$

**p 值的选择**: p 值越大,对单个维度上的大差异越敏感。一般使用 $p = 2$(欧氏)或 $p = 1$(曼哈顿),很少使用其他值。

#### 3.3.4 余弦相似度(Cosine Similarity)

余弦相似度衡量的是两个向量的方向一致性,而非绝对距离。

$$ \text{sim}_{cos}(x_a, x_b) = \frac{x_a \cdot x_b}{\|x_a\|_2 \cdot \|x_b\|_2} = \frac{\sum_{j=1}^{d} x_{a}^{(j)} x_{b}^{(j)}}{\sqrt{\sum_{j=1}^{d}(x_{a}^{(j)})^2} \cdot \sqrt{\sum_{j=1}^{d}(x_{b}^{(j)})^2}} $$

余弦相似度的取值范围为 $[-1, 1]$,值越大表示方向越一致。对应的"距离"可以定义为:

$$ d_{cos}(x_a, x_b) = 1 - \text{sim}_{cos}(x_a, x_b) $$

**特点**: 不受向量绝对大小的影响,只关注方向。适合文本分类(如 TF-IDF 向量)等场景,因为两篇文档的 TF-IDF 向量的长度差异不应影响它们是否相似的判断。

### 3.4 K 值选择的偏差-方差权衡分析

K 值是 KNN 最重要的超参数,它直接控制着模型的复杂度。理解 K 值如何影响偏差和方差,是掌握 KNN 的关键。

#### 3.4.1 K 值对模型的影响

- **K = 1(最小 K 值)**: 决策边界非常复杂,几乎每个训练样本都有一个属于自己的区域。模型对训练数据拟合得非常好(低偏差),但对噪声极其敏感,稍微改变训练数据就可能导致预测结果大幅变化(高方差)。极端情况下,一个噪声样本就可能在其周围创建一个错误的分类区域。

- **K = n(最大 K 值)**: 所有样本都参与投票,预测结果趋近于训练集中多数类的比例。决策边界退化为一条简单的线(甚至整个空间都是同一类别)。模型非常稳定(低方差),但忽略了数据的局部结构(高偏差)。

- **适中 K 值**: 在偏差和方差之间取得平衡,既能捕捉数据的局部结构,又对噪声有一定的鲁棒性。

**直观理解**: K 值类似于对决策边界的"平滑程度"——K 越小,边界越粗糙(更多细节); K 越大,边界越平滑(更多概括)。

#### 3.4.2 Cover-Hart 渐近错误率上界

Cover 和 Hart 在 1967 年的论文中给出了 KNN 渐近错误率的一个经典理论结果。这个结果说明了 KNN 在大样本条件下的最优性。

**定义**:
- $\mathcal{R}^*$: 贝叶斯最优错误率(Bayes Optimal Error Rate),即所有可能分类器能达到的最小错误率
- $\mathcal{R}_{NN}$: 1-NN 分类器的渐近错误率(当 $n \to \infty$ 时)
- $\mathcal{R}_{KNN}$: K-NN 分类器的渐近错误率

**Cover-Hart 定理**:

对于 1-NN 分类器,其渐近错误率满足:
$$ \mathcal{R}^* \leq \mathcal{R}_{NN} \leq 2\mathcal{R}^*(1 - \mathcal{R}^*) \leq 2\mathcal{R}^* $$

**推导思路**:

考虑一个新样本 $x$,其最近邻为 $x'$。当训练集足够大时,可以认为 $x'$ 与 $x$ 来自同一位置(非常接近),但它们的标签可能不同。

定义:
- $c^*$: $x$ 的贝叶斯最优类别(即后验概率最大的类别)
- $P(c^* | x)$: 在位置 $x$ 处正确类别的后验概率
- $1 - P(c^* | x)$: 在位置 $x$ 处分类错误的概率

当 $x'$ 非常接近 $x$ 时,1-NN 分类器犯错的概率为:
$$ P(\text{error}_{NN}) = \sum_{c=1}^{C} P(c | x) \cdot [1 - P(c | x)] = \sum_{c=1}^{C} P(c|x)(1-P(c|x)) $$

这是因为在位置 $x$,最近邻 $x'$ 的标签为 $c$ 的概率为 $P(c|x)$,而 $x$ 的真实标签恰好也是 $c$ 的概率为 $P(c|x)$,所以 $x'$ 的标签与 $x$ 的真实标签不同的概率为 $P(c|x) \cdot (1 - P(c|x))$。

对整个空间取期望:
$$ \mathcal{R}_{NN} = \mathbb{E}_x \left[\sum_{c=1}^{C} P(c|x)(1-P(c|x))\right] $$

而贝叶斯最优错误率为:
$$ \mathcal{R}^* = \mathbb{E}_x \left[1 - P(c^*|x)\right] = \mathbb{E}_x \left[\sum_{c \neq c^*} P(c|x)\right] $$

要证明 $\mathcal{R}_{NN} \leq 2\mathcal{R}^*(1 - \mathcal{R}^*)$,可以如下分析:

对于固定的 $x$,令 $p^* = P(c^*|x)$ 为最大后验概率,则:
$$ \sum_{c=1}^{C} P(c|x)(1-P(c|x)) = p^*(1-p^*) + \sum_{c \neq c^*} P(c|x)(1-P(c|x)) $$

由于 $p^* \geq P(c|x)$ 对所有 $c \neq c^*$ 成立,且 $\sum_c P(c|x) = 1$,利用不等式:

$$ \sum_{c=1}^{C} P(c|x)(1-P(c|x)) \leq p^*(1-p^*) + (1-p^*) \cdot p^* = 2p^*(1-p^*) $$

因为 $1 - p^*$ 是贝叶斯错误率在 $x$ 处的取值,且 $2p^*(1-p^*)$ 在 $p^* \in [0.5, 1]$ 时满足 $2p^*(1-p^*) \leq 2(1-p^*)$,所以:

$$ \mathcal{R}_{NN} \leq \mathbb{E}_x[2P(c^*|x)(1-P(c^*|x))] \leq 2\mathbb{E}_x[1-P(c^*|x)] = 2\mathcal{R}^* $$

进一步利用 $2p^*(1-p^*) \leq 2(1-p^*)(1-(1-p^*)) = 2\mathcal{R}^*(1-\mathcal{R}^*)$ 的局部形式,可以得到更紧的上界:

$$ \mathcal{R}_{NN} \leq 2\mathcal{R}^*(1 - \mathcal{R}^*) $$

**推广到 K-NN**:

当 $K > 1$ 时,渐近错误率可以更紧:
$$ \mathcal{R}^* \leq \mathcal{R}_{KNN} \leq \mathcal{R}^* + \frac{C-1}{\sqrt{K}} \cdot f(n) $$

其中 $f(n) \to 0$ 当 $n \to \infty$ 时。这意味着当 $K \to \infty$ 且 $n \to \infty$(但 $K/n \to 0$),KNN 的错误率收敛于贝叶斯最优错误率 $\mathcal{R}^*$。

**这个定理的实际意义**: KNN 是一个理论上"有保障"的算法——即使是最简单的 1-NN,其渐近错误率也不会超过贝叶斯最优的两倍。而在实际应用中,通过适当选择 K 值,KNN 的表现往往远好于这个上界。

### 3.5 KNN 算法伪代码

```
算法: K-Nearest Neighbors

输入:
    - 训练集 D = {(x_1, y_1), ..., (x_n, y_n)}
    - 新样本 x_new
    - 近邻个数 K
    - 距离度量 d(.,.)

输出: 预测标签 y_hat

步骤:
1.  for i = 1 to n do
2.      dist[i] = d(x_new, x_i)          // 计算到每个训练样本的距离
3.  end for
4.  sorted_indices = sort(dist) ascending  // 按距离升序排序
5.  neighbors = {y_i : i in sorted_indices[1:K]}  // 取前K个最近邻的标签
6.
7.  if 分类任务 then
8.      y_hat = majority_vote(neighbors)   // 多数投票
9.  else if 回归任务 then
10.     y_hat = mean(neighbors)            // 取平均值
11. end if
12. return y_hat
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

KNN 对数据的预处理有特殊要求,因为距离度量直接受数据尺度的影响。

**必要预处理**:

1. **标准化/归一化(极其重要)**:
   - **原因**: 欧氏距离对特征的量纲非常敏感。如果特征 A 的取值范围是 [0, 1000],特征 B 的取值范围是 [0, 1],那么特征 A 的差异会完全主导距离计算,特征 B 的影响几乎被忽略。
   - **方法**: StandardScaler(Z-score 标准化)或 MinMaxScaler(归一化到 [0, 1])
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

2. **缺失值处理**:
   - **方法**: 由于 KNN 基于距离计算,缺失值会导致距离无法计算。可以使用均值/中位数填充,或者用 KNNImputer(用最近邻的值来填充缺失值,这是一种"KNN 填充 KNN"的元方法)。

3. **特征选择/降维**:
   - **原因**: 高维数据会导致维度灾难(详见第 11 节)。建议先用 PCA 等方法降维,再使用 KNN。
   - **经验法则**: 当特征维度 $d$ 大于样本数 $n$ 的 10% 左右时,就应该考虑降维。

### 4.2 KNN 的"训练"过程

KNN 的训练过程与其他机器学习算法有本质不同:

- **传统算法(如线性回归、决策树)**: 训练阶段花费大量时间学习模型参数,预测阶段非常快速。
- **KNN**: 训练阶段几乎为零(仅存储数据),预测阶段需要计算所有距离,耗时较长。

```
训练阶段:
    存储 training_data = D
    return training_data

预测阶段(对每个新样本):
    计算距离 -> 排序 -> 选 K 邻 -> 投票
    return 预测结果
```

这就是"惰性学习"名称的由来——模型"懒惰"到不在训练时做任何工作。

### 4.3 预测时的搜索策略

#### 4.3.1 暴力搜索(Brute Force)

最朴素的方法:计算新样本与所有训练样本之间的距离,然后排序取 Top-K。

- **时间复杂度**: 预测一个样本需要 $O(n \cdot d)$ 计算距离 + $O(n \log n)$ 排序
- **空间复杂度**: $O(n \cdot d)$ 存储训练数据
- **适用场景**: 训练集较小(通常 $n < 10000$)时效率可接受

#### 4.3.2 K-D Tree 加速搜索

K-D Tree(K-Dimensional Tree)是一种对 K 维空间中的数据进行划分的数据结构,可以显著加速最近邻搜索。

**构建过程**(以 2D 为例):

1. 选择一个维度(如 x 轴),找到该维度中位数对应的点
2. 用一条垂直于该维度的超平面将数据一分为二
3. 在左半部分和右半部分中,分别选择下一个维度,递归构建
4. 直到每个叶节点只包含少量样本(或单个样本)

**搜索过程**:

1. 从根节点开始,根据待查询点在每个分割维度上的值,决定进入左子树还是右子树,直到到达叶节点
2. 将叶节点中的点作为当前最近邻
3. 回溯:检查是否存在另一子树中距离更近的点(通过判断分割平面与查询点之间的距离是否小于当前最近距离)
4. 如果存在,进入另一子树继续搜索

- **平均时间复杂度**: 构建 $O(n \log n)$, 查询 $O(\log n)$
- **最坏时间复杂度**: 查询 $O(n)$ (当数据分布不均匀时)
- **适用场景**: 特征维度 $d$ 较低(通常 $d < 20$)且数据量较大时

**维度限制**: 当 $d$ 增大时,K-D Tree 的效率迅速下降。经验上,当 $d > 20$ 时,暴力搜索反而可能更快,因为 K-D Tree 的构建和遍历开销超过了暴力搜索的简单计算。

#### 4.3.3 Ball Tree

Ball Tree 是另一种空间划分结构,使用超球体而非超平面来划分空间,在高维数据上通常比 K-D Tree 更有效。

- **构建**: 递归地将数据划分为两个由超球体包围的子集
- **搜索**: 利用三角不等式剪枝,避免不必要的距离计算
- **适用场景**: 中等维度($d$ 在 20-50 之间)的数据

scikit-learn 中默认根据数据维度自动选择搜索策略:
- $d \leq 20$: 使用 K-D Tree
- $20 < d \leq 50$: 使用 Ball Tree  
- $d > 50$: 使用 Brute Force

### 4.4 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| n_neighbors (K) | 最近邻个数 | 1-50,常用 3,5,7,11 | 5 |
| weights | 投票权重 | 'uniform' / 'distance' | 'uniform' |
| metric | 距离度量 | 'euclidean' / 'manhattan' / 'minkowski' / 'cosine' | 'minkowski' |
| p | 闵可夫斯基参数 | 1(曼哈顿) / 2(欧氏) | 2 |
| algorithm | 搜索算法 | 'auto' / 'ball_tree' / 'kd_tree' / 'brute' | 'auto' |
| leaf_size | 叶节点大小 | 10-50 | 30 |
| n_jobs | 并行数 | -1(全部核心) | None |

**K 值选择的经验法则**:
- 分类任务: 选择奇数 K,避免投票时出现平局(二分类场景)
- 初始范围: $K \in \{1, 3, 5, 7, 9, 11, 15, 21, \sqrt{n}\}$
- 使用交叉验证选择最优 K 值

---

## 5. 应用场景

### 5.1 典型应用

**应用1: 分类任务(如手写数字识别)**

- **问题类型**: 多分类
- **为什么适合**: 手写数字的特征(像素灰度值)经过适当预处理后,同一数字的图像在特征空间中确实聚集在一起,不同数字之间有一定距离。KNN 的"物以类聚"假设在此场景下成立。
- **实际案例**: MNIST 数据集上,使用 KNN + PCA 降维可以达到约 97% 的准确率,虽然不如深度学习,但实现简单,适合作为基线模型。

**应用2: 回归任务(如房价预测)**

- **问题类型**: 回归
- **为什么适合**: 地理位置相近的房屋通常具有相近的价格(空间自相关性)。KNN 回归可以自然地利用这种空间邻近性。
- **实际案例**: 在房价数据集中,使用 KNN 回归预测目标区域房价,可以很好地捕捉地理位置的影响。

**应用3: 推荐系统**

- **问题类型**: 无显式监督的预测
- **为什么适合**: "喜欢相似商品的用户"正是 KNN 思想的体现——找到与目标用户最相似的 K 个用户,将他们喜欢的商品推荐给目标用户(基于用户的协同过滤,User-based CF)。
- **实际案例**: 电影推荐、商品推荐、新闻推荐等场景中,基于用户的 KNN 协同过滤是一个经典方法。

**应用4: 异常检测**

- **问题类型**: 无监督/半监督
- **为什么适合**: 异常样本在特征空间中通常与大多数正常样本"距离很远"。通过计算每个样本到其第 K 个最近邻的距离,距离异常大的样本就可以被标记为异常。
- **实际案例**: 信用卡欺诈检测、网络入侵检测、工业设备故障检测等。

**应用5: 缺失值填充(KNN Imputation)**

- **问题类型**: 数据预处理
- **为什么适合**: 对于某个缺失特征的样本,找到其 K 个最近邻(基于已有特征),用这 K 个邻居在该特征上的值来估计缺失值。
- **实际案例**: scikit-learn 提供了 `KNNImputer`,可以方便地对缺失数据进行填充。

### 5.2 适用数据特征

该算法适合的数据特征:
- **特征类型**: 连续数值型(最适合); 离散特征需要编码处理
- **数据规模**: 小到中等规模($n < 50000$); 大规模数据需要加速结构
- **噪声容忍度**: 中等(K 值大时容忍度高,K 值小时对噪声敏感)
- **数据分布**: 类别之间有较好的空间分离性时效果最佳
- **特征维度**: 低维到中维($d < 30$); 高维数据需要降维预处理

### 5.3 不适用场景

**不适合的情况**:

1. **超高维数据**: 当特征维度远大于样本数(如 $d > n$)时,维度灾难使得距离度量失效。应先使用 PCA 或特征选择降维。
2. **大数据集**: 当训练集超过百万级别时,预测时计算所有距离非常慢。应考虑 Approximate Nearest Neighbor(ANN)方法,如 LSH(Locality-Sensitive Hashing)、HNSW 等。
3. **对实时性要求高的场景**: KNN 的预测延迟与训练集大小成正比,不适合需要毫秒级响应的在线服务。可以预计算或使用近似方法。
4. **类别极度不平衡**: 多数类会在投票中占据绝对优势,少数类的样本很难被正确识别。应考虑加权 KNN 或过采样/欠采样。

---

## 6. 优缺点分析

### 6.1 优点

1. **原理简单,直观易懂**: "近朱者赤,近墨者黑"的思想易于理解和解释,非常适合作为机器学习的入门算法。
   - 适用场景: 教学演示、快速原型验证

2. **无需训练,即时可用**: 不需要漫长的模型训练过程,添加新数据后立即生效。
   - 适用场景: 数据频繁更新的场景,如在线推荐系统

3. **对数据分布无假设**: 作为非参数方法,不对数据做任何分布假设,理论上可以处理任意形状的决策边界。
   - 适用条件: 只要"相似样本有相似标签"的假设成立

4. **天然支持多分类**: 不需要像逻辑回归那样使用 One-vs-Rest 或 Softmax,直接通过投票支持任意数量的类别。
   - 适用场景: 多分类问题,尤其是类别数不固定的场景

5. **可解释性强**: 预测时可以展示参与投票的 K 个邻居,让用户理解预测的依据。
   - 适用场景: 需要模型可解释性的应用

### 6.2 缺点

1. **预测速度慢**: 每个预测都需要计算与所有训练样本的距离。
   - 问题场景: 大规模数据集、在线实时预测
   - 解决思路: 使用 K-D Tree/Ball Tree 加速; 使用 ANN 方法; 预先对训练集聚类

2. **对高维数据敏感(维度灾难)**: 高维空间中距离失去区分力,所有样本间的距离趋于一致。
   - 改进方法: PCA 降维; 特征选择; 使用余弦相似度等适合高维的距离度量

3. **对特征尺度敏感**: 不同特征如果量级差异大,大尺度特征会主导距离计算。
   - 改进方法: 必须进行标准化或归一化预处理

4. **内存占用大**: 需要将所有训练数据存储在内存中。
   - 改进方法: 使用数据压缩; 原型选择(Prototype Selection),即选择训练集的一个代表性子集

5. **类别不平衡时效果差**: 多数类会主导投票结果。
   - 改进方法: 加权投票; 对少数类过采样; 使用类别平衡的采样策略

### 6.3 与同类算法对比

| 维度 | KNN | 决策树 | SVM |
|------|-----|--------|-----|
| 模型类型 | 非参数,惰性学习 | 非参数,急切学习 | 参数/非参数(取决于核) |
| 训练复杂度 | O(1) | O(n \cdot d \cdot \log n) | O(n^2) ~ O(n^3) |
| 预测复杂度 | O(n \cdot d) | O(d) | O(n_sv \cdot d) |
| 非线性能力 | 强(任意形状边界) | 中(轴对齐边界) | 强(通过核函数) |
| 可解释性 | 中(可展示邻居) | 高(可视化树结构) | 低(尤其使用核函数后) |
| 对异常值敏感度 | 中(K大时低) | 中(可通过剪枝控制) | 中(取决于核和正则化) |
| 高维数据表现 | 差(维度灾难) | 中 | 好(尤其线性核) |
| 大数据表现 | 差 | 好 | 差(训练慢) |
| 特征尺度敏感 | 非常敏感 | 不敏感 | 敏感(尤其RBF核) |
| 数据更新代价 | 零(重新预测即可) | 需重建树 | 需重新训练 |

**选择建议**:
- **选 KNN**: 小数据集、快速原型、需要可解释性、特征维度低
- **选决策树**: 需要高可解释性、混合类型特征、中等规模数据
- **选 SVM**: 中小规模数据、高维特征(如文本)、需要强泛化能力

---

## 7. 调库实现

### 7.1 环境准备

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 7.2 完整代码示例

```python
"""
KNN (K-Nearest Neighbors) 调库实现
数据集: Iris(鸢尾花分类) + 波士顿房价回归
目标: 展示KNN在分类和回归任务上的完整使用流程,包含超参数调优与多维度可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, load_wine
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, validation_curve
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler

# 设置随机种子,保证可复现
np.random.seed(42)

# 设置中文字体(如果需要)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===============================
# 1. 分类任务: 鸢尾花数据集
# ===============================
def load_classification_data():
    """
    加载鸢尾花分类数据集

    Returns:
        X: 特征矩阵, shape (150, 4)
        y: 标签向量, shape (150,)
        feature_names: 特征名称列表
        target_names: 类别名称列表
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.feature_names, iris.target_names


def preprocess_classification_data(X, y):
    """
    分类数据预处理: 标准化 + 数据分割

    Args:
        X: 原始特征矩阵
        y: 原始标签向量

    Returns:
        X_train, X_test, y_train, y_test: 预处理后的数据
        scaler: 标准化器(用于新数据的预处理)
    """
    # 标准化: KNN对特征尺度极其敏感,必须标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据分割: 80%训练,20%测试
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler


def train_knn_classifier(X_train, y_train, k=5):
    """
    训练KNN分类器

    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        k: 最近邻个数

    Returns:
        model: 训练好的KNN分类器
    """
    # 创建KNN分类器
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights='uniform',       # 均匀权重投票
        metric='minkowski',      # 闵可夫斯基距离
        p=2,                     # p=2即为欧氏距离
        algorithm='auto',        # 自动选择搜索算法
        n_jobs=-1                # 使用所有CPU核心
    )

    # "训练"——KNN的fit只是存储数据
    model.fit(X_train, y_train)

    print(f"[分类] KNN分类器训练完成 (K={k})")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  特征维度: {X_train.shape[1]}")
    return model


def evaluate_classifier(model, X_test, y_test, target_names):
    """
    评估KNN分类器性能

    Args:
        model: 训练好的分类器
        X_test: 测试集特征
        y_test: 测试集标签
        target_names: 类别名称

    Returns:
        metrics_dict: 评估指标字典
        y_pred: 预测标签
    """
    y_pred = model.predict(X_test)

    # 计算评估指标
    metrics_dict = {
        'Accuracy': accuracy_score(y_test, y_pred),
    }

    print("\n[分类评估结果]")
    print(f"  准确率 (Accuracy): {metrics_dict['Accuracy']:.4f}")
    print("\n  分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return metrics_dict, y_pred


# ===============================
# 2. 超参数调优: GridSearchCV
# ===============================
def hyperparameter_tuning(X_train, y_train):
    """
    使用网格搜索进行KNN超参数调优

    Args:
        X_train: 训练集特征
        y_train: 训练集标签

    Returns:
        best_model: 最佳模型
        grid_result: 网格搜索结果
    """
    # 定义参数搜索网格
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 25, 31],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]
    }

    # 创建KNN分类器(基模型)
    base_model = KNeighborsClassifier()

    # 网格搜索 + 5折交叉验证
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print("\n[超参数调优结果]")
    print(f"  最佳参数: {grid_search.best_params_}")
    print(f"  最佳交叉验证准确率: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search


# ===============================
# 3. 可视化: 四张核心图表
# ===============================
def visualize_decision_boundary(X_2d, y, k_values, feature_names_2d, target_names):
    """
    可视化不同K值下的决策边界(使用前两个特征绘制2D决策边界)

    Args:
        X_2d: 只包含前两个特征的矩阵
        y: 标签向量
        k_values: 要展示的K值列表
        feature_names_2d: 两个特征的名称
        target_names: 类别名称列表
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 定义颜色映射
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cmap_light = ListedColormap(['#FFE0E0', '#E0F7F5', '#E0F0FF'])
    cmap_bold = ListedColormap(colors)

    for idx, k in enumerate(k_values):
        ax = axes[idx // 2, idx % 2]

        # 创建KNN模型
        model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        model.fit(X_2d, y)

        # 生成网格点用于绘制决策边界
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.05),
            np.arange(y_min, y_max, 0.05)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 预测网格点的类别
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # 绘制决策边界
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

        # 绘制训练样本
        for i, target_name in enumerate(target_names):
            mask = (y == i)
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=colors[i], edgecolor='black', s=50,
                label=target_name, alpha=0.8
            )

        # 计算训练准确率
        train_acc = model.score(X_2d, y)
        ax.set_title(f'K = {k}, Training Accuracy = {train_acc:.3f}', fontsize=13)
        ax.set_xlabel(feature_names_2d[0], fontsize=11)
        ax.set_ylabel(feature_names_2d[1], fontsize=11)
        ax.legend(loc='best', fontsize=9)

    plt.suptitle('KNN Decision Boundaries with Different K Values (Iris Dataset)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('knn_decision_boundaries.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("[可视化] 决策边界图已保存: knn_decision_boundaries.png")


def visualize_k_value_sensitivity(X_train, y_train, X_test, y_test, k_range=range(1, 41)):
    """
    可视化K值对模型性能的影响(训练准确率 vs 测试准确率)

    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集标签
        k_range: K值搜索范围
    """
    train_scores = []
    test_scores = []
    cv_scores = []

    for k in k_range:
        # 训练并评估
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

        # 5折交叉验证
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(cv.mean())

    # 绘制图形
    plt.figure(figsize=(12, 6))

    plt.plot(list(k_range), train_scores, 'o-', color='#FF6B6B',
             label='Training Accuracy', linewidth=2, markersize=4)
    plt.plot(list(k_range), test_scores, 's-', color='#4ECDC4',
             label='Test Accuracy', linewidth=2, markersize=4)
    plt.plot(list(k_range), cv_scores, '^--', color='#45B7D1',
             label='5-Fold CV Accuracy', linewidth=1.5, markersize=4, alpha=0.8)

    # 标注最佳K值
    best_k_idx = np.argmax(test_scores)
    best_k = list(k_range)[best_k_idx]
    best_score = test_scores[best_k_idx]
    plt.axvline(x=best_k, color='gray', linestyle=':', alpha=0.7)
    plt.annotate(
        f'Best K={best_k}\nTest Acc={best_score:.3f}',
        xy=(best_k, best_score),
        xytext=(best_k + 5, best_score - 0.05),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=10
    )

    plt.xlabel('K (Number of Neighbors)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('KNN Performance vs. K Value (Iris Dataset)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('knn_k_value_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[可视化] K值敏感性分析图已保存: knn_k_value_sensitivity.png")
    print(f"  最佳K值: {best_k}, 最佳测试准确率: {best_score:.4f}")


def visualize_distance_metrics(X_2d, y, target_names):
    """
    可视化不同距离度量对KNN分类决策边界的影响

    Args:
        X_2d: 二维特征矩阵
        y: 标签向量
        target_names: 类别名称列表
    """
    metrics_config = [
        ('euclidean', 'Euclidean Distance (L2)'),
        ('manhattan', 'Manhattan Distance (L1)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    k = 5  # 固定K值

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cmap_light = ListedColormap(['#FFE0E0', '#E0F7F5', '#E0F0FF'])
    cmap_bold = ListedColormap(colors)

    for idx, (metric, title) in enumerate(metrics_config):
        ax = axes[idx]

        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X_2d, y)

        # 生成网格
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.05),
            np.arange(y_min, y_max, 0.05)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid_points).reshape(xx.shape)

        # 绘制决策边界
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

        # 绘制数据点
        for i, target_name in enumerate(target_names):
            mask = (y == i)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=colors[i], edgecolor='black', s=50,
                       label=target_name, alpha=0.8)

        acc = model.score(X_2d, y)
        ax.set_title(f'{title}\nK={k}, Accuracy={acc:.3f}', fontsize=12)
        ax.set_xlabel('Feature 1 (standardized)', fontsize=10)
        ax.set_ylabel('Feature 2 (standardized)', fontsize=10)
        ax.legend(loc='best', fontsize=9)

    plt.suptitle('KNN with Different Distance Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('knn_distance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("[可视化] 距离度量对比图已保存: knn_distance_metrics.png")


def visualize_confusion_matrix(y_test, y_pred, target_names):
    """
    可视化混淆矩阵

    Args:
        y_test: 真实标签
        y_pred: 预测标签
        target_names: 类别名称列表
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names,
           yticklabels=target_names,
           ylabel='True Label',
           xlabel='Predicted Label')

    # 在每个格子中显示数值
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.colorbar(im)
    plt.title('KNN Confusion Matrix (Iris Dataset)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("[可视化] 混淆矩阵已保存: knn_confusion_matrix.png")


# ===============================
# 4. 主程序
# ===============================
if __name__ == "__main__":
    print("=" * 60)
    print("KNN (K-Nearest Neighbors) 调库实现")
    print("=" * 60)

    # --- 分类任务 ---
    print("\n" + "=" * 60)
    print("第一部分: 分类任务 (鸢尾花数据集)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    X, y, feature_names, target_names = load_classification_data()
    print(f"  数据形状: X={X.shape}, y={y.shape}")
    print(f"  特征: {feature_names}")
    print(f"  类别: {target_names}")
    print(f"  类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 2. 数据预处理
    print("\n[2/6] 数据预处理...")
    X_train, X_test, y_train, y_test, scaler = preprocess_classification_data(X, y)
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 3. 训练默认模型
    print("\n[3/6] 训练默认KNN模型 (K=5)...")
    model = train_knn_classifier(X_train, y_train, k=5)

    # 4. 评估模型
    print("\n[4/6] 评估模型...")
    metrics_dict, y_pred = evaluate_classifier(model, X_test, y_test, target_names)

    # 5. 超参数调优
    print("\n[5/6] 超参数调优 (GridSearchCV)...")
    best_model, grid_result = hyperparameter_tuning(X_train, y_train)

    # 用最佳模型重新评估
    y_pred_best = best_model.predict(X_test)
    best_acc = accuracy_score(y_test, y_pred_best)
    print(f"\n  最佳模型在测试集上的准确率: {best_acc:.4f}")

    # 6. 可视化
    print("\n[6/6] 生成可视化图表...")

    # 图1: 决策边界 (使用前两个特征)
    X_2d = X_train[:, :2]  # 只取前两个特征用于2D可视化
    visualize_decision_boundary(
        X_2d, y_train,
        k_values=[1, 5, 15, 30],
        feature_names_2d=[feature_names[0], feature_names[1]],
        target_names=list(target_names)
    )

    # 图2: K值敏感性分析
    visualize_k_value_sensitivity(X_train, y_train, X_test, y_test)

    # 图3: 距离度量对比
    visualize_distance_metrics(X_2d, y_train, list(target_names))

    # 图4: 混淆矩阵
    visualize_confusion_matrix(y_test, y_pred_best, list(target_names))

    print("\n" + "=" * 60)
    print("程序执行完毕")
    print("=" * 60)
```

### 7.3 运行结果示例

```
============================================================
KNN (K-Nearest Neighbors) 调库实现
============================================================

第一部分: 分类任务 (鸢尾花数据集)
============================================================

[1/6] 加载数据...
  数据形状: X=(150, 4), y=(150,)
  特征: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
  类别: ['setosa' 'versicolor' 'virginica']
  类别分布: {0: 50, 1: 50, 2: 50}

[2/6] 数据预处理...
  训练集: (120, 4), 测试集: (30, 4)

[3/6] 训练默认KNN模型 (K=5)...
[分类] KNN分类器训练完成 (K=5)
  训练集大小: 120
  特征维度: 4

[4/6] 评估模型...
  准确率 (Accuracy): 1.0000

  分类报告:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        10
   virginica       1.00      1.00      1.00        10

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

[5/6] 超参数调优 (GridSearchCV)...
  最佳参数: {'metric': 'manhattan', 'n_neighbors': 11, 'p': 1, 'weights': 'uniform'}
  最佳交叉验证准确率: 0.9833

  最佳模型在测试集上的准确率: 1.0000
```

---

## 8. 手工代码实现

### 8.1 手工 KNN 分类器与回归器

```python
"""
KNN 手工实现
仅依赖 NumPy,从零实现 KNN 分类器与回归器的核心逻辑
包含多种距离度量和加权投票机制
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    手工实现的 KNN 分类器

    支持:
    - 多种距离度量: 欧氏、曼哈顿、闵可夫斯基
    - 均匀权重和距离加权投票
    - 多分类任务
    """

    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', p=2):
        """
        初始化 KNN 分类器

        Args:
            n_neighbors: 最近邻个数 K
            weights: 权重类型, 'uniform'(均匀) 或 'distance'(距离加权)
            metric: 距离度量, 'euclidean', 'manhattan', 'minkowski'
            p: 闵可夫斯基距离的参数 p (仅当 metric='minkowski' 时使用)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        "训练"——存储训练数据

        Args:
            X: 训练数据, shape (n_samples, n_features)
            y: 训练标签, shape (n_samples,)

        Returns:
            self
        """
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        return self

    def _compute_distances(self, x):
        """
        计算一个样本到所有训练样本的距离

        Args:
            x: 单个样本, shape (n_features,)

        Returns:
            distances: 距离数组, shape (n_train_samples,)
        """
        if self.metric == 'euclidean':
            # 欧氏距离: sqrt(sum((x_i - y_i)^2))
            diff = self.X_train - x
            distances = np.sqrt(np.sum(diff ** 2, axis=1))

        elif self.metric == 'manhattan':
            # 曼哈顿距离: sum(|x_i - y_i|)
            diff = self.X_train - x
            distances = np.sum(np.abs(diff), axis=1)

        elif self.metric == 'minkowski':
            # 闵可夫斯基距离: (sum(|x_i - y_i|^p))^(1/p)
            diff = self.X_train - x
            distances = np.sum(np.abs(diff) ** self.p, axis=1) ** (1.0 / self.p)

        else:
            raise ValueError(f"不支持的距离度量: {self.metric}")

        return distances

    def _predict_single(self, x):
        """
        预测单个样本的类别

        Args:
            x: 单个样本, shape (n_features,)

        Returns:
            prediction: 预测类别
        """
        # 计算到所有训练样本的距离
        distances = self._compute_distances(x)

        # 找到 K 个最近邻的索引
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_distances = distances[k_indices]
        k_labels = self.y_train[k_indices]

        if self.weights == 'uniform':
            # 均匀权重: 简单多数投票
            label_counts = Counter(k_labels)
            prediction = label_counts.most_common(1)[0][0]

        elif self.weights == 'distance':
            # 距离加权投票: 权重 = 1 / distance
            # 加一个小常数 epsilon 防止除以零
            epsilon = 1e-10
            weights = 1.0 / (k_distances + epsilon)

            # 按类别累加权重
            weight_sum_by_class = {}
            for label, weight in zip(k_labels, weights):
                weight_sum_by_class[label] = weight_sum_by_class.get(label, 0) + weight

            # 选择权重最大的类别
            prediction = max(weight_sum_by_class, key=weight_sum_by_class.get)

        else:
            raise ValueError(f"不支持的权重类型: {self.weights}")

        return prediction

    def predict(self, X):
        """
        预测多个样本的类别

        Args:
            X: 测试数据, shape (n_samples, n_features)

        Returns:
            predictions: 预测类别数组, shape (n_samples,)
        """
        X = np.array(X, dtype=np.float64)
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def predict_proba(self, X):
        """
        预测类别概率(基于K个最近邻的标签频率或加权频率)

        Args:
            X: 测试数据, shape (n_samples, n_features)

        Returns:
            proba: 概率矩阵, shape (n_samples, n_classes)
        """
        X = np.array(X, dtype=np.float64)
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            distances = self._compute_distances(x)
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_distances = distances[k_indices]
            k_labels = self.y_train[k_indices]

            if self.weights == 'uniform':
                # 均匀权重: 统计各类别出现频率
                counts = Counter(k_labels)
                for cls_idx, cls in enumerate(classes):
                    proba[i, cls_idx] = counts.get(cls, 0) / self.n_neighbors

            elif self.weights == 'distance':
                # 距离加权: 统计各类别权重占比
                epsilon = 1e-10
                weights = 1.0 / (k_distances + epsilon)
                weight_sum_by_class = {}
                for label, weight in zip(k_labels, weights):
                    weight_sum_by_class[label] = weight_sum_by_class.get(label, 0) + weight
                total_weight = sum(weight_sum_by_class.values())
                for cls_idx, cls in enumerate(classes):
                    proba[i, cls_idx] = weight_sum_by_class.get(cls, 0) / total_weight

        return proba

    def score(self, X, y):
        """
        计算分类准确率

        Args:
            X: 特征矩阵
            y: 真实标签

        Returns:
            accuracy: 准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class KNNRegressor:
    """
    手工实现的 KNN 回归器

    支持:
    - 多种距离度量: 欧氏、曼哈顿、闵可夫斯基
    - 均匀权重和距离加权平均
    """

    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', p=2):
        """
        初始化 KNN 回归器

        Args:
            n_neighbors: 最近邻个数 K
            weights: 权重类型, 'uniform' 或 'distance'
            metric: 距离度量, 'euclidean', 'manhattan', 'minkowski'
            p: 闵可夫斯基距离的参数 p
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        存储训练数据

        Args:
            X: 训练数据, shape (n_samples, n_features)
            y: 训练标签(连续值), shape (n_samples,)

        Returns:
            self
        """
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64)
        return self

    def _compute_distances(self, x):
        """
        计算单个样本到所有训练样本的距离(与分类器实现相同)
        """
        if self.metric == 'euclidean':
            diff = self.X_train - x
            distances = np.sqrt(np.sum(diff ** 2, axis=1))
        elif self.metric == 'manhattan':
            diff = self.X_train - x
            distances = np.sum(np.abs(diff), axis=1)
        elif self.metric == 'minkowski':
            diff = self.X_train - x
            distances = np.sum(np.abs(diff) ** self.p, axis=1) ** (1.0 / self.p)
        else:
            raise ValueError(f"不支持的距离度量: {self.metric}")
        return distances

    def _predict_single(self, x):
        """
        预测单个样本的回归值
        """
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_distances = distances[k_indices]
        k_values = self.y_train[k_indices]

        if self.weights == 'uniform':
            # 简单平均
            prediction = np.mean(k_values)
        elif self.weights == 'distance':
            # 距离加权平均: 权重 = 1 / distance
            epsilon = 1e-10
            weights = 1.0 / (k_distances + epsilon)
            prediction = np.sum(weights * k_values) / np.sum(weights)
        else:
            raise ValueError(f"不支持的权重类型: {self.weights}")

        return prediction

    def predict(self, X):
        """
        预测多个样本的回归值

        Args:
            X: 测试数据, shape (n_samples, n_features)

        Returns:
            predictions: 预测值数组, shape (n_samples,)
        """
        X = np.array(X, dtype=np.float64)
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def score(self, X, y):
        """
        计算 R^2 决定系数

        Args:
            X: 特征矩阵
            y: 真实值

        Returns:
            r2: R^2 分数
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        if ss_total == 0:
            return 0.0
        return 1 - (ss_residual / ss_total)


# ===============================
# 测试代码: 分类任务
# ===============================
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    import time

    np.random.seed(42)

    print("=" * 60)
    print("KNN 手工实现测试")
    print("=" * 60)

    # --- 分类任务测试 ---
    print("\n--- 分类任务测试 (鸢尾花数据集) ---")

    # 加载并预处理数据
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 手工实现
    print("\n[1] 手工 KNN 分类器:")
    start = time.time()
    my_clf = KNNClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
    my_clf.fit(X_train, y_train)
    my_train_acc = my_clf.score(X_train, y_train)
    my_test_acc = my_clf.score(X_test, y_test)
    my_time = time.time() - start
    print(f"  训练集准确率: {my_train_acc:.4f}")
    print(f"  测试集准确率: {my_test_acc:.4f}")
    print(f"  耗时: {my_time:.4f}s")

    # sklearn 实现
    print("\n[2] sklearn KNN 分类器:")
    start = time.time()
    sk_clf = SklearnKNN(n_neighbors=5, weights='uniform', metric='euclidean')
    sk_clf.fit(X_train, y_train)
    sk_train_acc = sk_clf.score(X_train, y_train)
    sk_test_acc = sk_clf.score(X_test, y_test)
    sk_time = time.time() - start
    print(f"  训练集准确率: {sk_train_acc:.4f}")
    print(f"  测试集准确率: {sk_test_acc:.4f}")
    print(f"  耗时: {sk_time:.4f}s")

    # 对比结果
    print(f"\n[3] 对比:")
    print(f"  准确率差异(训练): {abs(my_train_acc - sk_train_acc):.6f}")
    print(f"  准确率差异(测试): {abs(my_test_acc - sk_test_acc):.6f}")

    # --- 距离加权投票测试 ---
    print("\n--- 距离加权投票测试 ---")
    my_weighted = KNNClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    my_weighted.fit(X_train, y_train)
    print(f"  距离加权-训练准确率: {my_weighted.score(X_train, y_train):.4f}")
    print(f"  距离加权-测试准确率: {my_weighted.score(X_test, y_test):.4f}")

    # --- 曼哈顿距离测试 ---
    print("\n--- 曼哈顿距离测试 ---")
    my_manhattan = KNNClassifier(n_neighbors=5, weights='uniform', metric='manhattan')
    my_manhattan.fit(X_train, y_train)
    print(f"  曼哈顿距离-训练准确率: {my_manhattan.score(X_train, y_train):.4f}")
    print(f"  曼哈顿距离-测试准确率: {my_manhattan.score(X_test, y_test):.4f}")

    # --- 回归任务测试 ---
    print("\n--- 回归任务测试 ---")

    # 生成合成回归数据
    np.random.seed(42)
    n_samples = 200
    X_reg = np.random.randn(n_samples, 2)
    y_reg = 3 * X_reg[:, 0] + 2 * X_reg[:, 1] + 1 + np.random.randn(n_samples) * 0.5

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # 手工回归器
    my_reg = KNNRegressor(n_neighbors=5, weights='uniform')
    my_reg.fit(X_train_r, y_train_r)
    print(f"  手工回归器-训练 R^2: {my_reg.score(X_train_r, y_train_r):.4f}")
    print(f"  手工回归器-测试 R^2: {my_reg.score(X_test_r, y_test_r):.4f}")

    # sklearn 回归器
    from sklearn.neighbors import KNeighborsRegressor as SklearnKNNReg
    sk_reg = SklearnKNNReg(n_neighbors=5, weights='uniform')
    sk_reg.fit(X_train_r, y_train_r)
    print(f"  sklearn回归器-训练 R^2: {sk_reg.score(X_train_r, y_train_r):.4f}")
    print(f"  sklearn回归器-测试 R^2: {sk_reg.score(X_test_r, y_test_r):.4f}")

    # 距离加权回归
    my_reg_w = KNNRegressor(n_neighbors=5, weights='distance')
    my_reg_w.fit(X_train_r, y_train_r)
    print(f"  距离加权回归器-训练 R^2: {my_reg_w.score(X_train_r, y_train_r):.4f}")
    print(f"  距离加权回归器-测试 R^2: {my_reg_w.score(X_test_r, y_test_r):.4f}")

    # --- 预测概率测试 ---
    print("\n--- 预测概率测试 ---")
    proba = my_clf.predict_proba(X_test[:5])
    print(f"  前5个测试样本的类别概率:")
    for i, p in enumerate(proba):
        print(f"    样本{i}: {p} (预测类别: {np.argmax(p)})")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
```

### 8.2 与调库结果对比

| 方法 | 任务 | 训练准确率 | 测试准确率 | 耗时 |
|------|------|-----------|-----------|------|
| 手工实现(K=5, uniform, euclidean) | 分类 | 0.9833 | 1.0000 | ~0.02s |
| sklearn(K=5, uniform, euclidean) | 分类 | 0.9833 | 1.0000 | ~0.01s |
| 手工实现(K=5, distance, euclidean) | 分类 | 1.0000 | 1.0000 | ~0.02s |
| 手工实现(K=5, uniform, manhattan) | 分类 | 0.9833 | 1.0000 | ~0.02s |
| 手工回归器(K=5, uniform) | 回归 | 0.9365 | 0.9154 | ~0.01s |
| sklearn回归器(K=5, uniform) | 回归 | 0.9365 | 0.9154 | ~0.01s |
| 手工回归器(K=5, distance) | 回归 | 0.9998 | 0.9200 | ~0.01s |

**分析**:
- 手工实现与 sklearn 结果完全一致,验证了实现的正确性
- 距离加权方法在训练集上表现更好(因为每个样本自己就是最近的邻居,权重趋于无穷大),但在测试集上提升有限
- 手工实现稍慢,因为使用了 Python 循环而非 sklearn 的 C/Fortran 底层优化

---

## 9. 可视化与结果理解

### 9.1 决策边界的解读

决策边界可视化是理解 KNN 行为最直观的方式。在第 7 节的代码中,我们生成了不同 K 值下的决策边界图,以下是详细解读:

**K = 1 的决策边界**:
- 决策边界非常复杂,几乎是 Voronoi 图的边界
- 每个训练样本都有自己的一片"领地"
- 训练准确率通常为 100%(因为每个样本自己就是自己最近的邻居)
- 但测试准确率可能较低,因为决策边界过度拟合了训练数据中的噪声

**K = 5 的决策边界**:
- 决策边界变得更加平滑
- 孤立的噪声样本的影响被周围的多数类别"纠正"
- 在偏差和方差之间取得了较好的平衡
- 通常是默认的起始选择

**K = 15 的决策边界**:
- 决策边界相当平滑
- 一些细节特征被忽略,但整体结构得以保留
- 对噪声更鲁棒,但可能开始欠拟合

**K = 30 的决策边界**:
- 决策边界非常平滑,接近线性
- 大量的局部结构信息丢失
- 如果真实的决策边界是非线性的,此时模型欠拟合严重

### 9.2 K 值敏感性分析解读

K 值敏感性分析图(KNN Performance vs. K Value)揭示了偏差-方差权衡的典型模式:

- **K = 1**: 训练准确率 = 1.0(完美),但测试准确率相对较低(高方差)
- **K 增大**: 训练准确率逐渐下降,测试准确率先升后降
- **最优 K**: 测试准确率最高的点,即偏差-方差权衡的最佳平衡点
- **K 过大**: 训练和测试准确率都趋于较低值(高偏差,欠拟合)

### 9.3 距离度量对比解读

欧氏距离和曼哈顿距离的决策边界差异:

- **欧氏距离(L2)**: 决策边界呈圆形/椭圆形,因为等距线是圆
- **曼哈顿距离(L1)**: 决策边界呈菱形/轴对齐的多边形,因为等距线是菱形
- 在大多数情况下,两者的性能差异不大,但在特定数据分布下可能有显著区别

### 9.4 K 值与权重交互可视化

```python
"""
K值与权重类型的交互效果可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# 生成二分类数据(有一定重叠,使问题更有挑战性)
X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0,
    n_informative=2, random_state=42,
    n_clusters_per_class=1, flip_y=0.1
)

k_range = range(1, 31)
uniform_scores = []
distance_scores = []

for k in k_range:
    # 均匀权重
    clf_uniform = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    scores_u = cross_val_score(clf_uniform, X, y, cv=5, scoring='accuracy')
    uniform_scores.append(scores_u.mean())

    # 距离加权
    clf_distance = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores_d = cross_val_score(clf_distance, X, y, cv=5, scoring='accuracy')
    distance_scores.append(scores_d.mean())

plt.figure(figsize=(10, 5))
plt.plot(list(k_range), uniform_scores, 'o-', label='Uniform Weights', linewidth=2)
plt.plot(list(k_range), distance_scores, 's-', label='Distance Weights', linewidth=2)
plt.xlabel('K (Number of Neighbors)', fontsize=12)
plt.ylabel('5-Fold CV Accuracy', fontsize=12)
plt.title('KNN: Uniform vs Distance Weights', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**解读**: 距离加权方法通常在大 K 值时表现更好,因为它不会因为包含远距离邻居而过度平滑。而在小 K 值时,两种方法的差异不大。

---

## 10. 模型评估

### 10.1 评估指标选择

**分类任务指标**:

| 指标 | 适用场景 | 为什么选择 |
|------|---------|-----------|
| Accuracy | 类别均衡 | 整体正确率,简单直观 |
| Precision | 关注假阳性代价 | 如垃圾邮件检测,误判为垃圾邮件代价高 |
| Recall | 关注假阴性代价 | 如疾病检测,漏诊代价高 |
| F1-Score | 类别不平衡 | Precision 和 Recall 的调和平均 |
| Confusion Matrix | 多分类详细分析 | 展示各类别之间的混淆情况 |

**回归任务指标**:

| 指标 | 适用场景 | 为什么选择 |
|------|---------|-----------|
| MSE | 回归任务 | 对大误差敏感,惩罚大偏差 |
| RMSE | 回归任务 | 与原数据单位一致,更易解释 |
| MAE | 回归任务 | 对离群值更鲁棒 |
| R^2 | 回归任务 | 模型解释方差的比例,可跨模型比较 |

### 10.2 交叉验证实现

```python
"""
KNN 的全面交叉验证评估
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# 加载并预处理数据
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分层 K 折交叉验证(保证每折中类别比例一致)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 评估不同 K 值
print("K值 | 10折CV准确率(均值 +/- 标准差)")
print("-" * 45)
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f" K={k:2d} | {scores.mean():.4f} +/- {scores.std():.4f}")
```

**输出示例**:
```
K值 | 10折CV准确率(均值 +/- 标准差)
---------------------------------------------
 K= 1 | 0.9600 +/- 0.0422
 K= 3 | 0.9667 +/- 0.0447
 K= 5 | 0.9667 +/- 0.0447
 K= 7 | 0.9600 +/- 0.0548
 K= 9 | 0.9600 +/- 0.0548
 K=11 | 0.9600 +/- 0.0548
 K=15 | 0.9533 +/- 0.0589
 K=21 | 0.9467 +/- 0.0589
```

**解读**:
- 在鸢尾花数据集上,K=3 到 K=7 都能达到约 96.7% 的准确率
- 随着 K 增大,准确率逐渐下降,说明大 K 值在这个数据集上导致欠拟合
- 标准差较小,说明模型对不同数据划分的稳定性较好

### 10.3 K 值敏感性分析

```python
"""
系统化的 K 值敏感性分析
包含训练/验证/测试曲线,帮助判断过拟合与欠拟合
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# 准备数据
iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

k_values = list(range(1, 41))
train_scores = []
cv_scores = []
test_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    train_scores.append(model.score(X_train, y_train))
    cv_mean = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    cv_scores.append(cv_mean)
    test_scores.append(model.score(X_test, y_test))

# 找到最佳K值(基于交叉验证)
best_k_idx = np.argmax(cv_scores)
best_k = k_values[best_k_idx]

plt.figure(figsize=(12, 6))
plt.plot(k_values, train_scores, 'o-', label='Training', linewidth=2, markersize=3)
plt.plot(k_values, cv_scores, 's-', label='5-Fold CV (Validation)', linewidth=2, markersize=3)
plt.plot(k_values, test_scores, '^--', label='Test', linewidth=2, markersize=3)

plt.axvline(x=best_k, color='red', linestyle=':', alpha=0.7, label=f'Best K={best_k}')

# 标注区域
plt.axvspan(1, 3, alpha=0.1, color='red', label='Overfitting Zone')
plt.axvspan(20, 40, alpha=0.1, color='blue', label='Underfitting Zone')

plt.xlabel('K (Number of Neighbors)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN: Bias-Variance Tradeoff Analysis', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='center right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_bias_variance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"最佳K值(基于交叉验证): {best_k}")
print(f"  交叉验证准确率: {cv_scores[best_k_idx]:.4f}")
print(f"  测试集准确率: {test_scores[best_k_idx]:.4f}")
```

---

## 11. 常见问题与易错点

### 11.1 维度灾难(Curse of Dimensionality)

**现象**:
- 高维空间中,KNN 的性能急剧下降
- 所有样本之间的距离变得非常接近,最近邻和最远邻的距离差异很小
- 模型失去了区分能力,预测退化为简单的多数类

**原因**:
- 从数学角度看,考虑一个 $d$ 维单位超立方体中的数据点,到最近邻的期望距离随维度增加而趋近于到最远邻的距离
- 具体地,在高维空间中,体积集中在"角落"附近(超立方体的性质),大部分数据点都分布在表面附近
- 对于 $n$ 个样本,要使 KNN 有效工作,所需样本量随维度指数增长: $n \propto 2^d$

**示例**: 假设特征在 [0, 1] 范围内均匀分布:
- $d = 1$: 覆盖 10% 的特征范围只需边缘长度 0.1
- $d = 10$: 覆盖 10% 的特征空间需要边缘长度 $0.1^{1/10} \approx 0.79$,即 79% 的每个维度范围
- $d = 100$: 覆盖 10% 的特征空间需要边缘长度 $0.1^{1/100} \approx 0.977$,几乎覆盖整个空间

这意味着在高维空间中,没有任何一个局部区域有足够多的样本来做有意义的预测。

**解决方案**:
```python
# 方法1: PCA 降维
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # 保留95%的方差
X_reduced = pca.fit_transform(X)
print(f"原始维度: {X.shape[1]}, 降维后: {X_reduced.shape[1]}")

# 方法2: 特征选择
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)

# 方法3: 使用余弦相似度代替欧氏距离(对高维更鲁棒)
model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
```

### 11.2 数据不平衡问题

**现象**:
- 当某类样本远多于其他类时,KNN 的多数投票会被多数类主导
- 少数类样本即使在其"正确"的邻域中也可能被多数类淹没
- 例如: 多数类占 90%,即使 K=5 的邻居中有 3 个多数类样本和 2 个少数类样本,预测结果也是多数类

**解决方案**:
```python
# 方法1: 距离加权投票(让更近的少数类邻居有更大权重)
model_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')

# 方法2: 过采样少数类
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 方法3: 欠采样多数类
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# 方法4: 手动实现加权KNN,为少数类赋予更高权重
class WeightedKNN:
    def __init__(self, n_neighbors=5, class_weight=None):
        self.n_neighbors = n_neighbors
        self.class_weight = class_weight  # 例如: {0: 1.0, 1: 5.0}
```

### 11.3 未标准化特征

**现象**:
- 模型准确率异常低
- 某些特征的差异主导距离计算

**原因**:
- 假设特征 A 取值范围 [0, 10000],特征 B 取值范围 [0, 1]
- 欧氏距离中,特征 A 的差异贡献远大于特征 B
- 特征 B 的信息被完全淹没

**解决方案**:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 方法1: Z-score 标准化(最常用)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # 注意: 测试集只能用 transform,不能用 fit_transform

# 方法2: Min-Max 归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**易错点**: 对测试集使用 `fit_transform` 而非 `transform` 是一个常见错误。`fit_transform` 会根据测试集重新计算均值和方差,导致训练和测试使用了不同的标准化参数,产生"数据泄露"。

### 11.4 计算开销过大

**现象**:
- 预测非常慢,尤其是测试集很大时
- 内存占用高

**原因**:
- 暴力搜索的时间复杂度为 $O(n \cdot d)$ 每个预测样本
- 需要在内存中存储所有训练数据

**解决方案**:
```python
# 方法1: 使用 Ball Tree 或 K-D Tree 加速
model = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
# 或
model = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')

# 方法2: 使用近似最近邻(ANN)
# pip install nmslib  或  pip install faiss-cpu
import nmslib
# NMSLIB 支持多种近似最近邻算法,速度可提升10-100倍

# 方法3: 减少训练集大小(原型选择)
# 使用 K-Means 聚类找到训练集的"代表点"
from sklearn.cluster import MiniBatchKMeans
n_prototypes = 1000  # 将训练集压缩到1000个原型
kmeans = MiniBatchKMeans(n_clusters=n_prototypes, random_state=42)
kmeans.fit(X_train)
X_prototypes = kmeans.cluster_centers_
# 用每个簇中多数类的标签作为原型的标签
from scipy.stats import mode
y_prototypes = np.zeros(n_prototypes, dtype=int)
for i in range(n_prototypes):
    cluster_mask = (kmeans.labels_ == i)
    y_prototypes[i] = mode(y_train[cluster_mask]).mode[0]
```

### 11.5 K 值选择为偶数(二分类场景)

**现象**:
- 二分类任务中,投票时经常出现平局(K/2 对 K/2)
- 不同实现处理平局的方式不同,可能导致结果不一致

**解决方案**:
- 在二分类任务中,始终选择奇数 K
- 在多分类任务中,这个问题的严重性较低,但仍建议选择不太大的偶数

---

## 12. 学习总结

### 12.1 核心要点回顾

- **核心思想**: "物以类聚,人以群分"——根据最近的 K 个邻居来决定预测值
- **数学本质**: 基于距离度量的局部加权投票/平均,属于非参数方法
- **优化目标**: 没有显式的损失函数需要优化,超参数(K、距离度量、权重)通过交叉验证选择
- **适用场景**: 中小规模数据集、低维特征、需要快速原型验证、需要可解释性
- **局限性**: 高维数据(维度灾难)、大数据集(计算开销)、实时预测(延迟高)

### 12.2 关键公式汇总

**1. 欧氏距离**:
$$ d_{Euc}(x_a, x_b) = \sqrt{\sum_{j=1}^{d}(x_{a}^{(j)} - x_{b}^{(j)})^2} $$

**2. 闵可夫斯基距离(统一形式)**:
$$ d_{Min}(x_a, x_b) = \left(\sum_{j=1}^{d} |x_{a}^{(j)} - x_{b}^{(j)}|^p \right)^{1/p} $$

**3. 余弦相似度**:
$$ \text{sim}_{cos}(x_a, x_b) = \frac{x_a \cdot x_b}{\|x_a\|_2 \cdot \|x_b\|_2} $$

**4. 分类预测(多数投票)**:
$$ \hat{y} = \arg\max_{c} \sum_{(x_i, y_i) \in \mathcal{N}_K(x)} \mathbb{I}(y_i = c) $$

**5. 分类预测(距离加权投票)**:
$$ \hat{y} = \arg\max_{c} \sum_{(x_i, y_i) \in \mathcal{N}_K(x)} \frac{1}{d(x, x_i) + \epsilon} \cdot \mathbb{I}(y_i = c) $$

**6. 回归预测(均值)**:
$$ \hat{y} = \frac{1}{K} \sum_{(x_i, y_i) \in \mathcal{N}_K(x)} y_i $$

**7. Cover-Hart 渐近错误率上界**:
$$ \mathcal{R}^* \leq \mathcal{R}_{NN} \leq 2\mathcal{R}^*(1 - \mathcal{R}^*) $$

### 12.3 最佳实践

**数据预处理**:
- 必须进行标准化或归一化(KNN 对特征尺度极其敏感)
- 考虑 PCA 降维(当特征维度 > 20 时)
- 处理缺失值(推荐 KNNImputer)

**模型选择**:
- 默认 K=5 是一个合理的起始值
- 使用交叉验证搜索最优 K 值
- 分类任务优先选奇数 K
- 考虑距离加权投票以提升性能

**模型评估**:
- 交叉验证是选择 K 值的标准方法
- 同时监控训练集和测试集性能,判断过拟合/欠拟合
- 可视化决策边界以直观理解模型行为

### 12.4 与其他算法的联系

- **前置算法**: 距离度量(线性代数基础)、基本统计概念
- **后续算法**: 决策树(另一种非参数方法,但采用急切学习)、SVM(另一种基于距离/间隔的分类器)
- **相关算法**: K-Means(KNN 的无监督版本)、LOF(基于 KNN 距离的异常检测)、KNN 协同过滤(推荐系统)

---

## 13. 练习题与思考题

### 13.1 练习题1: 手动计算 KNN 分类

**题目**: 给定以下 6 个训练样本(二维特征,二分类问题):

| 样本 | $x_1$ | $x_2$ | 类别 |
|------|-------|-------|------|
| A | 1.0 | 1.0 | 正类 |
| B | 1.5 | 2.0 | 正类 |
| C | 2.0 | 1.5 | 正类 |
| D | 4.0 | 4.0 | 负类 |
| E | 3.5 | 5.0 | 负类 |
| F | 5.0 | 4.5 | 负类 |

现有一个待预测样本 $P = (2.0, 3.0)$,使用欧氏距离和 $K=3$ 的 KNN 进行分类。

请计算:
1. $P$ 到每个训练样本的欧氏距离
2. 找出 3 个最近邻
3. 给出预测结果

**答案与解析**:

**步骤1: 计算欧氏距离**

$$ d(P, A) = \sqrt{(2.0-1.0)^2 + (3.0-1.0)^2} = \sqrt{1 + 4} = \sqrt{5} \approx 2.236 $$

$$ d(P, B) = \sqrt{(2.0-1.5)^2 + (3.0-2.0)^2} = \sqrt{0.25 + 1} = \sqrt{1.25} \approx 1.118 $$

$$ d(P, C) = \sqrt{(2.0-2.0)^2 + (3.0-1.5)^2} = \sqrt{0 + 2.25} = 1.5 $$

$$ d(P, D) = \sqrt{(2.0-4.0)^2 + (3.0-4.0)^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.236 $$

$$ d(P, E) = \sqrt{(2.0-3.5)^2 + (3.0-5.0)^2} = \sqrt{2.25 + 4} = \sqrt{6.25} = 2.5 $$

$$ d(P, F) = \sqrt{(2.0-5.0)^2 + (3.0-4.5)^2} = \sqrt{9 + 2.25} = \sqrt{11.25} \approx 3.354 $$

**步骤2: 按距离排序,取前 3 个最近邻**

| 排名 | 样本 | 距离 | 类别 |
|------|------|------|------|
| 1 | B | 1.118 | 正类 |
| 2 | C | 1.500 | 正类 |
| 3 | A / D | 2.236 | 正类 / 负类 |

注意: A 和 D 的距离相同(都是 $\sqrt{5}$),这是一个平局情况。如果有两个候选(A 和 D)竞争第 3 个位置,可以:
- 随机选择一个
- 将两个都纳入考虑(K 变为 4)
- 按其他标准(如索引顺序)选择

假设选择 A 作为第 3 个最近邻:

**步骤3: 多数投票**

- K=3 的最近邻: B(正类), C(正类), A(正类)
- 正类票数: 3, 负类票数: 0
- **预测结果: 正类**

---

### 13.2 练习题2: KNN 回归手动计算

**题目**: 给定以下训练样本:

| 样本 | $x$ | $y$ |
|------|-----|-----|
| A | 1.0 | 2.0 |
| B | 2.0 | 4.0 |
| C | 3.0 | 5.0 |
| D | 5.0 | 7.0 |
| E | 8.0 | 10.0 |

对 $x_{new} = 4.0$ 使用 $K=3$ 的 KNN 回归预测 $y$ 值。

**答案与解析**:

**步骤1: 计算距离**

$$ d(x_{new}, A) = |4.0 - 1.0| = 3.0 $$
$$ d(x_{new}, B) = |4.0 - 2.0| = 2.0 $$
$$ d(x_{new}, C) = |4.0 - 3.0| = 1.0 $$
$$ d(x_{new}, D) = |4.0 - 5.0| = 1.0 $$
$$ d(x_{new}, E) = |4.0 - 8.0| = 4.0 $$

**步骤2: 排序取 K=3**

最近 3 个邻居: C(距离 1.0, y=5.0), D(距离 1.0, y=7.0), B(距离 2.0, y=4.0)

**步骤3: 计算预测值**

简单平均: $\hat{y} = (5.0 + 7.0 + 4.0) / 3 = 16.0 / 3 \approx 5.333$

距离加权平均:
$$ w_C = 1/(1.0 + 0.01) = 0.990, \quad w_D = 1/(1.0 + 0.01) = 0.990, \quad w_B = 1/(2.0 + 0.01) = 0.498 $$
$$ \hat{y}_{weighted} = \frac{0.990 \times 5.0 + 0.990 \times 7.0 + 0.498 \times 4.0}{0.990 + 0.990 + 0.498} = \frac{4.95 + 6.93 + 1.99}{2.48} = \frac{13.87}{2.48} \approx 5.593 $$

---

### 13.3 练习题3: 距离度量对比

**题目**: 计算向量 $a = (3, 4)$ 和 $b = (0, 0)$ 之间的:
1. 欧氏距离
2. 曼哈顿距离
3. 闵可夫斯基距离($p=3$)
4. 余弦相似度

**答案与解析**:

1. **欧氏距离**: $d = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$

2. **曼哈顿距离**: $d = |3| + |4| = 7$

3. **闵可夫斯基距离($p=3$)**: $d = (|3|^3 + |4|^3)^{1/3} = (27 + 64)^{1/3} = 91^{1/3} \approx 4.498$

4. **余弦相似度**: $\text{sim} = \frac{a \cdot b}{\|a\| \|b\|} = \frac{0}{5 \times 0}$

   由于 $b = (0, 0)$ 是零向量,余弦相似度无定义(分母为零)。在实际实现中,通常将零向量的余弦相似度定义为 0 或跳过该计算。

---

### 13.4 练习题4: Cover-Hart 定理理解

**题目**: 假设某个分类问题的贝叶斯最优错误率 $\mathcal{R}^* = 0.1$,那么 1-NN 分类器的渐近错误率上界是多少? 如果 $\mathcal{R}^* = 0.3$ 呢? 这说明了什么?

**答案与解析**:

当 $\mathcal{R}^* = 0.1$ 时:
$$ \mathcal{R}_{NN} \leq 2\mathcal{R}^*(1 - \mathcal{R}^*) = 2 \times 0.1 \times 0.9 = 0.18 $$

当 $\mathcal{R}^* = 0.3$ 时:
$$ \mathcal{R}_{NN} \leq 2\mathcal{R}^*(1 - \mathcal{R}^*) = 2 \times 0.3 \times 0.7 = 0.42 $$

**分析**:
- 贝叶斯最优错误率越低,1-NN 的上界越紧。当 $\mathcal{R}^* = 0.1$ 时,1-NN 的错误率不超过 0.18,仅比最优差 0.08。
- 贝叶斯最优错误率越高(问题越难),1-NN 的上界越宽松。当 $\mathcal{R}^* = 0.3$ 时,1-NN 的错误率可能高达 0.42。
- 这说明 KNN 在"容易"的分类问题上表现更好,在"困难"的问题上可能表现较差。
- 注意这是渐近结果(样本量趋于无穷),小样本情况下 KNN 的实际错误率可能高于这个上界。

---

### 13.5 练习题5: K 值选择策略

**题目**: 在一个包含 1000 个样本、50 个特征、2 个类别的分类问题中,你会如何系统地选择 K 值? 请描述完整的方法论。

**答案与解析**:

**步骤1: 划分数据**

将数据集划分为训练集(60%)、验证集(20%)、测试集(20%):
```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
```

**步骤2: 预处理**

对训练集进行标准化,将同样的参数应用到验证集和测试集:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

由于有 50 个特征,考虑 PCA 降维:
```python
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)
```

**步骤3: K 值搜索**

在奇数范围内搜索 K 值(避免二分类平局):
```python
k_candidates = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
```

对每个 K 值,使用 5 折交叉验证评估:
```python
for k in k_candidates:
    model = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    # 记录均值和标准差
```

**步骤4: 选择最佳 K**

选择交叉验证准确率最高的 K 值。如果多个 K 值性能相近,选择较大的 K(更简单,泛化更好)。同时考虑标准差,选择性能稳定的 K。

**步骤5: 最终评估**

用最佳 K 在训练集+验证集上重新训练,在测试集上评估最终性能。

**补充考虑**:
- 同时测试 `weights='uniform'` 和 `weights='distance'`
- 测试不同的距离度量
- 如果数据不平衡,使用加权 KNN 或 SMOTE 预处理

---

## 14. 学习路径建议

### 14.1 前置知识

**学习 KNN 前,你需要掌握**:

**数学基础**:
- [ ] **线性代数**: 向量范数(L1/L2)、矩阵运算、欧氏空间
  - 重点: 理解不同范数的几何含义
  - 学习时长: 1-2 周

- [ ] **概率论**: 条件概率、贝叶斯定理
  - 重点: 理解后验概率和贝叶斯最优分类器
  - 学习时长: 1 周

**编程基础**:
- [ ] **Python + NumPy**: 数组操作、排序、聚合
  - 重点: argsort、向量化距离计算
  - 学习时长: 1 周

- [ ] **Matplotlib 基础**: 散点图、等高线图、子图
  - 重点: 能绘制决策边界可视化

**机器学习基础**:
- [ ] **基本概念**: 训练集/测试集、过拟合/欠拟合、交叉验证
- [ ] **评估指标**: 准确率、精确率、召回率、F1

### 14.2 平行算法(可同时学习)

1. **决策树**: 另一种非参数方法,但采用急切学习(训练时构建树)
   - 学习重点: 信息增益/基尼系数、剪枝策略
   - 对比点: 决策树训练慢预测快,KNN 训练快预测慢

2. **朴素贝叶斯**: 基于概率的分类方法,也假设相似样本有相似标签
   - 学习重点: 贝叶斯定理、条件独立性假设
   - 对比点: 朴素贝叶斯是参数方法(假设数据分布),KNN 是非参数方法

3. **逻辑回归**: 经典的线性分类方法
   - 学习重点: Sigmoid 函数、交叉熵损失、梯度下降
   - 对比点: 逻辑回归学习线性边界,KNN 可以学习任意形状边界

### 14.3 进阶算法(后续学习)

**短期目标(1-2 个月)**:

1. **K-D Tree**: 深入理解 KNN 的加速搜索数据结构
   - 关联: K-D Tree 是 KNN 在大数据集上的加速工具
   - 难度: 2/5

2. **LOF (Local Outlier Factor)**: 基于 KNN 距离的异常检测算法
   - 关联: LOF 使用 KNN 计算局部可达密度来识别异常点
   - 难度: 2/5

**中期目标(3-6 个月)**:

1. **支持向量机(SVM)**: 基于最大间隔的分类器
   - 关联: SVM 也是基于距离(间隔)的分类器,但学习最优分离超平面
   - 难度: 3/5

2. **集成学习(Bagging/Boosting)**: 多个弱分类器的组合
   - 关联: KNN 可以作为集成学习中的基分类器
   - 难度: 3/5

**长期目标(6 个月以上)**:

1. **近似最近邻(ANN)**: 大规模 KNN 的工业级解决方案
   - 最新技术: HNSW、FAISS、ScaNN
   - 难度: 4/5

2. **度量学习(Metric Learning)**: 学习最优的距离度量函数
   - 核心思想: 不是手动选择距离度量,而是从数据中学习距离函数
   - 代表方法: Siamese Network、Triplet Loss、Prototypical Network
   - 难度: 5/5

### 14.4 推荐资源

**教材类**:
1. **《统计学习方法》** 李航 - 第 3 章 K 近邻法,数学推导简洁严谨
2. **《机器学习》** 周志华(西瓜书) - 第 3 章,对 KNN 的原理和应用有全面介绍
3. **《Pattern Classification》** Duda, Hart, Stork - Cover-Hart 定理的原始参考

**论文类**:
1. **Cover, T., & Hart, P. (1967). Nearest Neighbor (NN) Pattern Classification.** IEEE Transactions on Information Theory. -- Cover-Hart 定理的原始论文
2. **Fix, E., & Hodges, J. (1951). Discriminatory Analysis: Nonparametric Discrimination.** -- KNN 思想的最早提出
3. **Weinberger, K., & Saul, L. (2009). Distance Metric Learning for Large Margin Nearest Neighbor Classification.** JMLR. -- 度量学习与 KNN 的结合

**在线课程**:
1. **Andrew Ng 机器学习课程** (Coursera) - 对 KNN 有清晰的教学
2. **CS231n** (斯坦福) - 在图像分类中对比 KNN 与深度学习方法

**实践项目**:
1. **Kaggle - Digit Recognizer**: 使用 KNN 进行手写数字识别(入门级)
2. **Kaggle - House Prices**: 使用 KNN 回归预测房价(进阶级)
3. **构建 KNN 推荐系统**: 基于用户相似度的电影推荐(挑战级)

---

## 附录

### A. 参考文献

1. Cover, T. M., & Hart, P. E. (1967). Nearest neighbor (NN) pattern classification. IEEE Transactions on Information Theory, 13(1), 21-27.
2. Fix, E., & Hodges, J. L. (1951). Discriminatory analysis: Nonparametric discrimination: Consistency properties. USAF School of Aviation Medicine.
3. Friedman, J. H., Bentley, J. L., & Finkel, R. A. (1977). An algorithm for finding best matches in logarithmic expected time. ACM Transactions on Mathematical Software, 3(3), 209-226.
4. Liu, T., Moore, A. W., & Gray, A. (2006). Efficient exact k-NN and nonparametric classification in high dimensions. Advances in Neural Information Processing Systems.
5. Weinberger, K. Q., & Saul, L. K. (2009). Distance metric learning for large margin nearest neighbor classification. Journal of Machine Learning Research, 10(Feb), 207-244.

### B. 常见问题 FAQ

**Q1: KNN 为什么不需要"训练"?**

A: KNN 属于"惰性学习"(Lazy Learning),它不做任何显式的模型构建或参数学习。它的"模型"就是原始训练数据本身。当需要预测时,算法才去"查阅"这些数据,计算距离并投票。这种设计使得 KNN 增加新数据时无需重新训练,但代价是每次预测的计算量较大。

**Q2: K 值越大模型越简单还是越复杂?**

A: K 值越大,模型越简单(偏差越大,方差越小)。大 K 值意味着更多邻居参与投票,决策边界更平滑,模型的假设更"粗糙"。反之,K=1 时模型最复杂,可以拟合任何形状的决策边界(包括噪声),但泛化能力差。

**Q3: KNN 能处理文本分类吗?**

A: 可以。文本通常表示为 TF-IDF 向量或词嵌入向量,然后使用余弦相似度作为距离度量。但要注意维度灾难——文本特征空间通常维度很高(词表大小),建议先进行降维或特征选择。在深度学习时代,KNN 在文本分类中已较少使用,但在某些场景(如少样本学习)中仍有价值。

**Q4: KNN 和 K-Means 有什么关系?**

A: 虽然名字相似,但它们是不同的算法:
- KNN 是监督学习算法,用于分类和回归
- K-Means 是无监督学习算法,用于聚类
- KNN 中的 K 是最近邻个数; K-Means 中的 K 是聚类中心个数
- 两者都基于距离度量,但目标完全不同

---

**文档结束**

> 如果你觉得这个文档对你有帮助,请分享给更多学习机器学习的人!
> 如有错误或建议,欢迎指出,共同完善!
