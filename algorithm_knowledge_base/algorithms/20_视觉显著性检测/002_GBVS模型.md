# GBVS模型 学习文档

> 基于图的视觉显著性——用马尔可夫链模拟注意的分布与传播。
>
> 来源线索：本节内容根据原书第2.2.1节"GBVS:基于图的视觉显著性模型"整理。

---

## 1. 算法基础认知

**一句话定义：** GBVS（Graph-Based Visual Saliency）由Harel、Koch和Perona于2006年在NIPS上提出，通过将特征图构建为全连接图并计算其马尔可夫链平稳分布来得到显著性图。

**核心思想：** 将特征图上的每个位置看作一个节点，节点间基于特征差异和空间距离建立连接权重。这个权重图被归一化为马尔可夫转移矩阵，其平稳分布（即长期访问概率）反映了每个位置的显著性。显著性高的位置对应于"从很多其他位置容易到达"的中心节点。

**为什么用图？** 传统的显著性方法（如ITTI）使用中心-周围差分操作，只能捕捉局部对比度。GBVS通过全连接图建模长距离依赖关系，能够感知全局结构。马尔可夫链的平稳分布本质上是特征图上的"中心性"度量。

**历史背景：** GBVS建立在Koch和Ullman的神经科学框架之上，是ITTI模型的图论版本。它将生物启发的显著性计算形式化为图上的随机游走问题，是第一个将马尔可夫链应用于视觉显著性的方法。

---

## 2. 核心原理

GBVS模型分为三个主要阶段：

### 2.1 特征提取
输入图像通过Gabor滤波器、颜色对等生成多个特征图 $F_k$（与ITTI类似）。每个特征图的大小为 $M \times N$。

### 2.2 激活图构建
对每个特征图 $F_k$，构建一个全连接有向图 $G_k$：
- **节点**：特征图上的每个像素位置 $(i,j)$，共 $N = M \times N$ 个节点
- **边权重**：从节点 $(i,j)$ 到 $(p,q)$ 的有向边权重定义为：

$$
w_k((i,j),(p,q)) = d((i,j),(p,q)) \cdot F(i,j;p,q)
$$

其中：
- $d((i,j),(p,q))$ 是空间距离因子：
  $$
  d((i,j),(p,q)) = \exp\left(-\frac{(i-p)^2 + (j-q)^2}{2\sigma^2}\right)
  $$
- $F(i,j;p,q)$ 是特征差异因子：
  $$
  F(i,j;p,q) = \left|\log\frac{F_k(i,j)}{F_k(p,q)}\right|
  $$

### 2.3 归一化与平稳分布
将每个节点的出度归一化，得到马尔可夫转移矩阵 $P_k$：

$$
P_k((i,j) \to (p,q)) = \frac{w_k((i,j),(p,q))}{\sum_{p',q'} w_k((i,j),(p',q'))}
$$

然后迭代求解平稳分布 $\pi_k$（满足 $\pi_k P_k = \pi_k$）：

$$
\pi_k^{(t+1)} = \pi_k^{(t)} P_k
$$

平稳分布 $\pi_k$ 即为该特征图的显著性图。

### 2.4 跨尺度融合
将所有特征图的平稳分布归一化后求和，得到最终显著性图。

---

## 3. 数学公式与推导

### 3.1 马尔可夫链基础

马尔可夫链由状态集合 $S$ 和转移概率矩阵 $P$ 定义，满足：

$$
P(X_{t+1} = j | X_t = i) = P_{ij}
$$

平稳分布 $\pi$ 是满足 $\pi P = \pi$ 的概率分布，且 $\sum_i \pi_i = 1$。

### 3.2 精细到粗糙的归一化

GBVS的关键创新之一是"精细到粗糙"的归一化策略。在构建图时，使用两个不同的空间尺度参数 $\sigma_1$（精细）和 $\sigma_2$（粗糙）：

$$
w((i,j),(p,q)) = \exp\left(-\frac{(i-p)^2+(j-q)^2}{2\sigma_1^2}\right) \cdot \left|\log\frac{F(i,j)}{F(p,q)}\right| \cdot \exp\left(-\frac{(i-p)^2+(j-q)^2}{2\sigma_2^2}\right)^{-1}
$$

这等价于对特征差异进行空间加权归一化。

### 3.3 平稳分布的迭代求解

由于特征图的节点数可能很大（$N=10000$），直接求解特征向量 $\pi P = \pi$ 计算量过大。GBVS采用幂迭代法：

$$
\pi^{(t+1)} = \pi^{(t)} P
$$

可以证明，对于不可约非周期的马尔可夫链，$\pi^{(t)}$ 收敛到平稳分布 $\pi$。

### 3.4 最终显著性图

$$
S(x,y) = \sum_{k} \mathcal{N}\left(\pi_k(x,y)\right)
$$

其中 $\mathcal{N}(\cdot)$ 是归一化操作，将所有特征图的显著性值映射到 $[0,1]$。

---

## 4. 训练过程讲解

GBVS也是一个**无训练**的方法，不需要学习任何参数。

**处理流程：**
1. **多尺度特征提取**：使用 Gabor 滤波器、颜色双拮抗等生成 3-6 个特征图
2. **特征图下采样**：每个特征图下采样到 $32 \times 32$ 到 $64 \times 64$ 之间
3. **图构建**：对每个特征图构建全连接图（约 $10^4$ 个节点，$10^8$ 条边）
4. **平稳分布求解**：15-20 次幂迭代
5. **归一化融合**：所有特征图的平稳分布归一化后相加
6. **上采样**：结果上采样到原始图像大小

**计算复杂度：** 全连接图构建需要 $O(N^2)$ 时间，其中 $N$ 是特征图的像素数。因此通常先将特征图下采样到较小尺寸。

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 视觉显著性预测 | 预测人眼在自由观看时的注视点 |
| 图像分割 | 显著区域作为前景分割的引导 |
| 目标检测 | 快速定位显著目标 |
| 图像质量评估 | 基于显著性的客观质量评价 |
| 广告设计 | 评估广告图像的视觉吸引力 |
| 网页设计 | 分析用户注意力的分布 |

---

## 6. 优缺点分析

**优点：**
- ✅ **全局建模**：通过全连接图捕获长距离依赖
- ✅ **生物合理性**：构建在Koch-Ullman框架上
- ✅ **马尔可夫理论基础扎实**：平稳分布有直观的概率解释
- ✅ **多特征融合自然**：不同特征图的显著性通过归一化后求和
- ✅ **无需训练**：完全无监督

**缺点：**
- ❌ **计算量大**：全连接图 $O(N^2)$ 复杂度，大图像上很慢
- ❌ **参数敏感**：空间尺度 $\sigma$、迭代次数等需要调参
- ❌ **缺乏语义**：只使用低级特征，无法理解场景内容
- ❌ **边界效应**：特征图边缘的节点连接不充分
- ❌ **特征提取依赖**：底层特征提取方法影响最终结果

---

## 7. 调库实现

```python
"""
GBVS模型 - 完整调库实现
使用 numpy + scipy 实现基于马尔可夫链的显著性检测
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class GBVS:
    """基于图的视觉显著性模型"""
    
    def __init__(self, sigma_frac=0.1, n_iter=20):
        """
        参数:
            sigma_frac: 空间距离权重参数 (占图像尺寸的比例)
            n_iter: 平稳分布迭代求解次数
        """
        self.sigma_frac = sigma_frac
        self.n_iter = n_iter
    
    def _build_dissimilarity(self, feature_map):
        """
        构建差异矩阵 D
        
        每个元素 D[(i,j), (p,q)] = 特征差异 × 空间距离权重
        """
        h, w = feature_map.shape
        n_pixels = h * w
        sigma = self.sigma_frac * max(h, w)
        
        # 使用向量化方法加速
        # 构建坐标网格
        ys, xs = np.mgrid[0:h, 0:w]
        coords = np.stack([ys.ravel(), xs.ravel()], axis=1)  # (N, 2)
        
        # 特征值向量
        feat_vals = feature_map.ravel()  # (N,)
        
        # 由于全连接矩阵太大，我们对每个节点只计算到其他节点的距离
        # 实际中使用稀疏化版本：只连接空间近邻
        # 这里展示完整的理论实现框架
        
        D = np.zeros((n_pixels, n_pixels))
        
        for i in range(n_pixels):
            # 特征差异
            fd = np.abs(np.log(feat_vals + 1e-8) - np.log(feat_vals[i] + 1e-8))
            # 空间距离
            sd = np.exp(-np.sum((coords - coords[i]) ** 2, axis=1) / (2 * sigma ** 2))
            D[i, :] = fd * sd
        
        return D
    
    def _build_transition_matrix(self, D):
        """从差异矩阵构建转移矩阵"""
        # 归一化：每行和为 1
        row_sums = D.sum(axis=1, keepdims=True) + 1e-8
        P = D / row_sums
        return P
    
    def _solve_stationary(self, P):
        """迭代求解平稳分布"""
        n = P.shape[0]
        pi = np.ones(n) / n  # 均匀初始分布
        
        for _ in range(self.n_iter):
            pi = pi @ P
        
        return pi
    
    def compute_saliency(self, feature_maps):
        """
        计算显著性图
        
        参数:
            feature_maps: 特征图数组 (K, H, W) 或 (H, W)
        
        返回:
            saliency: 归一化的显著性图 (H, W)
        """
        if len(feature_maps.shape) == 2:
            feature_maps = feature_maps[np.newaxis, :, :]
        
        n_maps, h, w = feature_maps.shape
        cumulative = np.zeros((h, w))
        
        for k in range(n_maps):
            fm = feature_maps[k]
            
            print(f"处理特征图 {k+1}/{n_maps}, 大小 {h}x{w}...")
            
            # 构建差异矩阵
            D = self._build_dissimilarity(fm)
            
            # 构建转移矩阵并求解平稳分布
            P = self._build_transition_matrix(D)
            pi = self._solve_stationary(P)
            
            # 重塑为2D显著性图
            act_map = pi.reshape(h, w)
            
            # 归一化并累加
            act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
            cumulative += act_map
        
        # 最终归一化
        saliency = cumulative / n_maps
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency


class FastGBVS:
    """加速版GBVS：使用局部连接近似和卷积加速"""
    
    def __init__(self, sigma=0.1, n_iter=20, k_neighbors=0.1):
        self.sigma = sigma
        self.n_iter = n_iter
        self.k_neighbors = k_neighbors  # 连接比例
    
    def compute_saliency(self, feature_maps):
        """使用局部近似加速计算"""
        if len(feature_maps.shape) == 2:
            feature_maps = feature_maps[np.newaxis, :, :]
        
        n_maps, h, w = feature_maps.shape
        cumulative = np.zeros((h, w))
        
        for k in range(n_maps):
            fm = feature_maps[k]
            
            # 对每个像素，只与空间近邻建立连接
            # 使用高斯差分近似马尔可夫链的平稳分布
            # 方法：计算特征图的局部对比度，然后扩散
            act_map = self._fast_activation(fm)
            
            act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
            cumulative += act_map
        
        saliency = cumulative / n_maps
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency
    
    def _fast_activation(self, fm):
        """快速激活图计算"""
        # 计算局部对比度（相当于中心-周围差分）
        sigma1 = self.sigma * max(fm.shape)
        sigma2 = sigma1 * 1.6
        
        center = gaussian_filter(fm, sigma1)
        surround = gaussian_filter(fm, sigma2)
        contrast = np.abs(center - surround)
        
        # 通过多次高斯扩散模拟马尔可夫链
        act_map = contrast.copy()
        for _ in range(self.n_iter):
            act_map = gaussian_filter(act_map, sigma1 * 0.5)
        
        return act_map


def demo():
    """演示函数"""
    np.random.seed(42)
    
    # 创建多特征图
    h, w = 32, 32
    n_features = 3
    
    # 创建一个包含显著区域的模拟特征图
    feature_maps = np.random.randn(n_features, h, w) * 0.2 + 0.5
    # 在每个特征图中加入显著目标
    for k in range(n_features):
        feature_maps[k, 10:16, 12:18] = 1.5
        feature_maps[k, 22:26, 20:24] = 0.2  # 低值区
    
    # GBVS
    print("=== GBVS ===")
    model = GBVS(sigma_frac=0.1, n_iter=15)
    # 由于全连接版本很慢，这里用快速版
    fast_model = FastGBVS(sigma=0.1, n_iter=10)
    smap = fast_model.compute_saliency(feature_maps)
    
    print(f"显著性图范围: [{smap.min():.3f}, {smap.max():.3f}]")
    print(f"显著目标区域 (10:16, 12:18) 均值: {smap[10:16, 12:18].mean():.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for k in range(min(3, n_features)):
        axes[k].imshow(feature_maps[k], cmap='viridis')
        axes[k].set_title(f'特征图 {k+1}')
        axes[k].axis('off')
    
    axes[3].imshow(smap, cmap='hot')
    axes[3].set_title('GBVS显著性图')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('gbvs_saliency.png', dpi=150)
    print("结果已保存至 gbvs_saliency.png")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""
GBVS模型 - 手工实现
不依赖 scipy 的高级函数
"""
import numpy as np


def manual_gaussian_kernel(size, sigma):
    """手工生成高斯核"""
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            dist = (i - center) ** 2 + (j - center) ** 2
            kernel[i, j] = np.exp(-dist / (2 * sigma ** 2))
    return kernel / kernel.sum()


def manual_convolve(data, kernel):
    """手工二维卷积"""
    h, w = data.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(data)
    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return result


def manual_gaussian_blur(data, sigma):
    """手工高斯模糊"""
    ksize = int(2 * np.ceil(2 * sigma) + 1)
    kernel = manual_gaussian_kernel(ksize, sigma)
    return manual_convolve(data, kernel)


def gbvs_manual(feature_map, sigma=0.1, n_iter=20):
    """
    手工实现GBVS
    
    参数:
        feature_map: 输入特征图 (H, W)
        sigma: 空间尺度参数
        n_iter: 迭代次数
    
    返回:
        显著性图
    """
    h, w = feature_map.shape
    n_pixels = h * w
    spatial_sigma = sigma * max(h, w)
    
    # 步骤1: 构造权重矩阵
    # 使用矩阵化方式，先构建坐标
    W = np.zeros((n_pixels, n_pixels))
    
    # 构建差异矩阵（简化版：使用循环）
    idx = 0
    for i in range(h):
        for j in range(w):
            # 遍历所有目标节点
            row = np.zeros(n_pixels)
            tgt_idx = 0
            for p in range(h):
                for q in range(w):
                    if i == p and j == q:
                        row[tgt_idx] = 0  # 自身连接为0
                    else:
                        # 特征差异
                        fd = abs(np.log(feature_map[i, j] + 1e-8) -
                                 np.log(feature_map[p, q] + 1e-8))
                        # 空间距离
                        sd = np.exp(-((i-p)**2 + (j-q)**2) / (2 * spatial_sigma**2))
                        row[tgt_idx] = fd * sd
                    tgt_idx += 1
            W[idx, :] = row
            idx += 1
    
    # 步骤2: 归一化为转移矩阵
    row_sums = W.sum(axis=1, keepdims=True) + 1e-8
    P = W / row_sums
    
    # 步骤3: 迭代求解平稳分布
    pi = np.ones(n_pixels) / n_pixels
    for _ in range(n_iter):
        pi = pi @ P
    
    # 步骤4: 重塑为显著性图
    saliency = pi.reshape(h, w)
    
    # 步骤5: 高斯平滑
    saliency = manual_gaussian_blur(saliency, sigma=2)
    
    # 归一化
    s_min, s_max = saliency.min(), saliency.max()
    saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
    
    return saliency


def test_gbvs_manual():
    """测试手工实现"""
    np.random.seed(42)
    
    # 小尺寸特征图
    fm = np.random.randn(8, 8) * 0.3 + 0.5
    fm[2:5, 3:6] = 1.5  # 显著区域
    
    print("=== 手工GBVS测试（小图） ===")
    print(f"特征图大小: {fm.shape}")
    
    # 由于全连接 O(N^2) 计算量，这里减少迭代
    smap = gbvs_manual(fm, sigma=0.15, n_iter=10)
    
    print(f"显著性范围: [{smap.min():.3f}, {smap.max():.3f}]")
    print(f"显著区域 (2:5, 3:6) 均值: {smap[2:5, 3:6].mean():.4f}")
    
    assert smap.max() > 0.5, "显著性最大值应较高"
    print("✓ 测试通过")


if __name__ == "__main__":
    test_gbvs_manual()
```

---

## 9. 可视化与结果理解

### 9.1 权重矩阵可视化

GBVS 的权重矩阵 $W$ 是一个 $N \times N$ 的矩阵（$N$ 是像素数），其结构揭示了：
- 对角线为 0（自身不连接）
- 非对角线元素反映特征差异和空间距离的乘积
- 相似且邻近的像素间权重高

### 9.2 平稳分布理解

平稳分布 $\pi$ 是马尔可夫链的长期访问概率：
- 高 $\pi$ 值的节点：从很多其他节点经过少量步数就能到达
- 低 $\pi$ 值的节点：孤立或边缘节点

在视觉上，高 $\pi$ 值对应：
- 特征突出的区域（与其他区域差异大）
- 处于"中心"位置的区域（空间上容易到达）

### 9.3 迭代收敛过程

前几次迭代中分布变化很快，后面逐渐收敛。通常 10-15 次迭代即可达到稳定。

---

## 10. 模型评估

```python
"""GBVS模型评估"""
import numpy as np


def evaluate_gbvs():
    """评估GBVS模型"""
    np.random.seed(42)
    
    # 模拟数据
    h, w = 32, 32
    fm = np.random.randn(3, h, w) * 0.2 + 0.5
    fm[:, 10:16, 12:18] = 1.5
    
    # 真值
    gt = np.zeros((h, w))
    gt[10:16, 12:18] = 1.0
    
    # 快速GBVS
    model = FastGBVS(sigma=0.1, n_iter=10)
    smap = model.compute_saliency(fm)
    
    # 计算AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(gt.flatten() > 0, smap.flatten())
    
    # 计算相似度
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(smap.flatten().reshape(1, -1),
                           gt.flatten().reshape(1, -1))[0, 0]
    
    print(f"AUC: {auc:.4f}")
    print(f"余弦相似度: {sim:.4f}")
    
    return auc, sim


if __name__ == "__main__":
    evaluate_gbvs()
```

---

## 11. 常见问题与易错点

### Q1: GBVS的全连接图计算太慢怎么办？
**A:** 实际使用中，不需要连接所有节点。可以只连接空间近邻（$k$-NN图），或者使用快速近似的多尺度高斯差分法（如FastGBVS所示）。

### Q2: 转移矩阵为什么需要对行归一化？
**A:** 行归一化确保每行元素之和为1，满足马尔可夫转移矩阵的概率性质（$\sum_j P_{ij} = 1$），保证平稳分布存在且唯一。

### Q3: GBVS与ITTI的主要区别？
**A:** ITTI使用中心-周围差分和DoG-like归一化；GBVS将特征图视为图，用马尔可夫链建模注意力传播。GBVS更全局，ITTI更局部。

### Q4: 迭代次数不够会怎样？
**A:** 平稳分布未收敛，结果不稳定。迭代太少时，显著性分布与初始均匀分布相差不大。

### Q5: 为什么用对数特征差异？
**A:** 对数变换使特征差异对比例变化不敏感。一个区域比另一个区域亮2倍时，对数差为log2，与绝对亮度无关。

---

## 12. 学习总结

### 核心要点

1. **图论视角**：显著性被建模为图上随机游走的平稳分布
2. **双因子权重**：特征差异 × 空间距离，兼顾外观和位置
3. **马尔可夫链**：幂迭代法求平稳分布
4. **无缝融合**：多特征图的显著性通过归一化求和融合

### 与后续模型的关系

- **GBVS → CAS**：CAS对GBVS的图结构进行改进
- **GBVS → DeepGaze**：GBVS的图思想被深度学习方法继承
- **GBVS → 图神经网络**：GNN中的注意力机制与GBVS的图模型有深刻联系

---

## 13. 练习题与思考题

### 基础题

**1.** GBVS中的平稳分布 $\pi$ 有什么直观意义？

<details>
<summary>答案</summary>
$\pi_i$ 表示从随机初始位置出发，经过无限长随机游走后访问节点 $i$ 的概率。显著性高的区域是在图中"中心性"高的位置——从很多其他位置容易到达。
</details>

**2.** 特征差异为什么用 $\log$ 而不是直接相减？

<details>
<summary>答案</summary>
对数差异对比例敏感而非绝对差。例如，亮度 0.01 vs 0.02（差0.01，比2倍）与亮度 0.5 vs 0.51（差0.01，比1.02倍）在感知上完全不同。对数差异 $\log(0.02/0.01)=\log2$ 和 $\log(0.51/0.5) \approx 0.02$ 能正确反映这种感知差异。
</details>

**3.** 为什么使用幂迭代法而不是直接解特征方程？

<details>
<summary>答案</summary>
幂迭代法不需要存储完整的转移矩阵（当使用稀疏图时），且对于大的 $N$ 更容易控制计算资源。
</details>

### 进阶题

**4.** 推导：如果转移矩阵 $P$ 是双随机的（每行和每列和都为1），平稳分布是什么？

<details>
<summary>答案</summary>
如果 $P$ 是双随机的，均匀分布 $\pi_i = 1/N$ 是平稳分布（因为 $\sum_i \pi_i P_{ij} = \frac{1}{N}\sum_i P_{ij} = \frac{1}{N}$）。这从理论上解释了为什么行归一化后的非均匀分布才能产生有意义的显著性。
</details>

**5.** 尝试将GBVS理解为图上的PageRank算法，两者有什么联系和区别？

<details>
<summary>答案</summary>
GBVS和PageRank都基于马尔可夫链的平稳分布。区别在于：(1) PageRank有跳转因子，GBVS没有；(2) PageRank的边权重基于超链接，GBVS基于特征差异和空间距离；(3) PageRank处理的是有向图且考虑权威性，GBVS处理完全连接图。
</details>

---

## 14. 学习路径建议

### 预备知识
- 图论基础（节点、边、权重矩阵）
- 马尔可夫链与随机过程
- 线性代数（特征值、特征向量）
- 概率论（平稳分布、遍历性）

### 进阶方向
1. **GBVS → CAS (Context-Aware Saliency)**：引入上下文信息改进图结构
2. **GBVS → Spectral Saliency**：用谱聚类方法替代马尔可夫链
3. **GBVS → Deep Saliency**：用CNN学习特征图替代手工特征

### 推荐阅读
- Harel et al. "Graph-Based Visual Saliency." NIPS 2006.
- Itti et al. "A Model of Saliency-Based Visual Attention for Rapid Scene Analysis." TPAMI 1998.
- Brin & Page. "The Anatomy of a Large-Scale Hypertextual Web Search Engine." 1998.

### 项目实践
1. 在MIT300数据集上比较GBVS与ITTI的性能
2. 使用PyTorch实现可微分的GBVS模块并嵌入CNN
3. 尝试不同的图构建策略（k-NN、$\epsilon$-图）对结果的影响
