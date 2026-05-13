# HC与RC显著性模型 学习文档

> 基于全局对比度的显著性检测——像素级（HC）与区域级（RC）。
>
> 来源线索：本节内容根据原书第2.2.2节"HC与RC：基于全局对比度的显著性检测"整理。

---

## 1. 算法基础认知

**一句话定义：** HC（Histogram Contrast）和RC（Region Contrast）由程明明（Ming-Ming Cheng）等人于2011年在CVPR上提出，HC以像素颜色在全局图像中的对比度计算显著性，RC以分割区域间的颜色对比度（加权空间距离）计算显著性。

**核心思想：** 显著性 = 颜色与所有其他颜色的对比度的频率加权和。公式化为：

$$
S(I_k) = \sum_{\forall I_j \in I} D(I_k, I_j) \cdot w(I_j)
$$

其中 $D(I_k, I_j)$ 是像素/区域 $k$ 和 $j$ 之间的颜色距离，$w(I_j)$ 是权重（频率或区域大小）。

**HC vs RC：**

| 方法 | 基本单元 | 空间信息 | 计算效率 | 效果 |
|------|---------|---------|---------|------|
| HC | 像素 | 无 | 高 | 速度最快 |
| RC | 区域（超像素） | 有（空间距离加权） | 中 | 效果最好 |

**历史背景：** 程明明等人提出HC和RC时，显著性检测主要依赖局部对比度和频域方法。HC和RC首次证明全局对比度比局部对比度更有效。RC与GrabCut结合形成SaliencyCut，实现了显著性驱动的图像分割。

---

## 2. 核心原理

### 2.1 HC：基于直方图对比度的显著性

**全分辨率计算：** 对每个像素 $I_k$，计算其与图像中所有其他像素的颜色距离之和：

$$
S(I_k) = \sum_{j=1}^N D(I_k, I_j)
$$

其中 $N$ 是图像像素总数，$D(\cdot)$ 是LAB空间的欧氏距离。

**直方图加速：** 直接计算 $O(N^2)$ 太慢。将颜色量化到 $n$ 个颜色（通常 $n=85$），然后：

$$
S(c_k) = \sum_{j=1}^n f_j \cdot D(c_k, c_j)
$$

其中 $c_k$ 是颜色 $k$ 的值，$f_j$ 是颜色 $j$ 在图像中出现的频率。

**颜色空间平滑：** 由于量化可能带来不连续，用颜色相似性进行平滑：

$$
S'(c) = \frac{1}{(m-1)T} \sum_{i=1}^m (T - D(c, c_i)) \cdot S(c_i)
$$

其中 $m$ 是最近邻数，$T = \sum_{i=1}^m D(c, c_i)$。

### 2.2 RC：基于区域对比度的显著性

**超像素分割：** 使用基于图的图像分割将图像分割为 $K$ 个区域 $\{R_k\}$。

**区域颜色表示：** 每个区域 $R_k$ 用颜色直方图 $H_k$ 表示（通常取每个颜色中心值）。

**区域显著性：**

$$
S(R_k) = \sum_{R_k \neq R_i} w(R_i) \cdot D_s(R_k, R_i) \cdot D_c(R_k, R_i)
$$

其中：
- $w(R_i)$ 是区域 $R_i$ 的像素数权重
- $D_c(R_k, R_i)$ 是区域间的颜色距离
- $D_s(R_k, R_i)$ 是空间距离权重：$\exp(-d_{spatial}^2 / \sigma^2)$

---

## 3. 数学公式与推导

### 3.1 HC的数学形式

全分辨率HC的等价形式：

$$
S(I_k) = \sum_{j=1}^N \|I_k - I_j\|_2
$$

展开为：

$$
\begin{aligned}
S(I_k) &= N \cdot \|I_k\|_2 + \sum_j \|I_j\|_2 - 2 I_k \cdot \sum_j I_j \\
&= N \cdot \|I_k\|_2 + C - 2 I_k \cdot (N \cdot \bar{I})
\end{aligned}
$$

其中 $\bar{I}$ 是均值颜色，$C$ 是常数。但注意范数不是线性的，实际不能这样分解。需要通过颜色量化加速。

### 3.2 颜色量化

原始图像可能有16M+种颜色，通过K-means量化为 $n$ 种：

$$
\min \sum_{i=1}^N \min_{j=1}^n \|I_i - \mu_j\|_2^2
$$

量化后每个颜色 $c_j$ 有频率 $f_j = |\{i: \text{label}(i)=j\}| / N$。

### 3.3 RC的空间加权

RC引入空间距离权重使显著性具有局部性：

$$
S(R_k) = \sum_{i \neq k} \frac{|R_i|}{\sqrt{|R_k|}} \cdot \exp\left(-\frac{d_{spatial}^2(R_k, R_i)}{\sigma^2}\right) \cdot D_c(R_k, R_i)
$$

$\sigma$ 控制空间影响范围，$\sigma$ 小则只有邻近区域影响显著性。

### 3.4 SaliencyCut

RC的显著性与GrabCut结合：

1. 计算RC显著性图
2. 阈值化得到前/背景种子
3. GrabCut迭代分割
4. 输出前景掩膜

---

## 4. 训练过程讲解

HC和RC都是**无训练**的方法。

**HC流程：**
1. 将图像从RGB转换到LAB颜色空间
2. 用K-means将颜色量化为85种
3. 计算每种颜色的频率
4. 计算每种颜色的显著性（与其他颜色的距离 × 频率）
5. 颜色空间平滑
6. 每个像素赋予其颜色对应的显著性值
7. 归一化到 [0, 1]

**RC流程：**
1. 将图像从RGB转换到LAB
2. 基于图的图像分割得到超像素区域
3. 对每个区域计算颜色直方图
4. 计算区域间颜色距离和空间距离
5. 计算每个区域的显著性（加权和）
6. 区域内所有像素赋相同显著性值
7. 归一化

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 显著性检测 | HC（快速）/ RC（精确）两种选择 |
| 图像分割 | RC + GrabCut = SaliencyCut |
| 目标检测 | 显著性作为候选区域生成 |
| 图像裁剪 | 保留最显著的区域 |
| 图像缩略图 | 显著性保持的图像缩放 |
| 内容感知编辑 | 显著性引导的图像编辑 |

---

## 6. 优缺点分析

**优点（HC）：**
- ✅ **快速**：颜色量化后 $O(n^2)$，$n=85$
- ✅ **全局对比度**：比局部对比度更准确
- ✅ **实现简单**

**缺点（HC）：**
- ❌ **无空间信息**：忽略位置关系
- ❌ **量化误差**：K-means可能丢失细节
- ❌ **颜色不连续**：量化边界处显著性跳跃

**优点（RC）：**
- ✅ **含空间信息**：空间加权更合理
- ✅ **区域级描述**：抗噪性强
- ✅ **效果好**：在多个数据集上SOTA

**缺点（RC）：**
- ❌ **依赖分割质量**：过分割或欠分割影响结果
- ❌ **计算较慢**：需要分割和区域间计算
- ❌ **参数多**：分割参数、$\sigma$ 等需要调节

---

## 7. 调库实现

```python
"""HC和RC显著性模型 - 完整调库实现"""
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class HC:
    """基于直方图对比度的显著性 (Histogram Contrast)"""
    
    def __init__(self, n_colors=85, smooth_neighbors=5):
        self.n_colors = n_colors
        self.smooth_neighbors = smooth_neighbors
    
    def compute_saliency(self, image):
        """
        计算HC显著性
        
        参数:
            image: 输入图像 (H, W, 3) RGB
        
        返回:
            saliency: 归一化显著性图
        """
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # 颜色量化 (K-means)
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=5)
        labels = kmeans.fit_predict(pixels)
        colors = kmeans.cluster_centers_  # (n_colors, 3)
        
        # 颜色频率
        n_pixels = len(pixels)
        freqs = np.array([(labels == c).sum() / n_pixels 
                          for c in range(self.n_colors)])
        
        # 计算颜色显著性
        color_saliency = np.zeros(self.n_colors)
        for i in range(self.n_colors):
            dists = np.sqrt(np.sum((colors[i] - colors) ** 2, axis=1))
            color_saliency[i] = np.sum(freqs * dists)
        
        # 颜色空间平滑
        color_saliency = self._smooth_color_saliency(colors, color_saliency)
        
        # 归一化
        color_saliency = (color_saliency - color_saliency.min()) / \
                         (color_saliency.max() - color_saliency.min() + 1e-8)
        
        # 映射回像素
        saliency = color_saliency[labels].reshape(h, w)
        
        # 高斯平滑
        saliency = gaussian_filter(saliency, sigma=3)
        
        return saliency
    
    def _smooth_color_saliency(self, colors, saliency):
        """颜色空间平滑：相似颜色应有相似显著性"""
        n = len(colors)
        smoothed = saliency.copy()
        
        for i in range(n):
            dists = np.sqrt(np.sum((colors[i] - colors) ** 2, axis=1))
            nearest = np.argsort(dists)[1:self.smooth_neighbors+1]
            
            T = np.sum(dists[nearest])
            if T > 0:
                weights = (T - dists[nearest]) / ((self.smooth_neighbors - 1) * T)
                smoothed[i] = np.sum(weights * saliency[nearest])
        
        return smoothed


class RC:
    """基于区域对比度的显著性 (Region Contrast)"""
    
    def __init__(self, n_regions=200, sigma_spatial=0.4):
        self.n_regions = n_regions
        self.sigma_spatial = sigma_spatial
    
    def compute_saliency(self, image):
        """
        计算RC显著性
        
        参数:
            image: 输入图像 (H, W, 3) RGB
        
        返回:
            saliency: 归一化显著性图
        """
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # 使用K-means模拟区域分割
        kmeans = KMeans(n_clusters=self.n_regions, random_state=42, n_init=3)
        region_labels = kmeans.fit_predict(pixels).reshape(h, w)
        region_colors = kmeans.cluster_centers_  # (n_regions, 3)
        
        # 计算区域中心（用于空间距离）
        region_centers = {}
        for r in range(self.n_regions):
            ys, xs = np.where(region_labels == r)
            if len(ys) > 0:
                region_centers[r] = (ys.mean() / h, xs.mean() / w)
            else:
                region_centers[r] = (0.5, 0.5)
        
        # 区域大小
        region_sizes = np.array([(region_labels == r).sum() 
                                 for r in range(self.n_regions)])
        total_pixels = h * w
        
        # 计算区域显著性
        saliency_map = np.zeros((h, w))
        
        for r in range(self.n_regions):
            if region_sizes[r] == 0:
                continue
            
            sal = 0.0
            for s in range(self.n_regions):
                if r == s or region_sizes[s] == 0:
                    continue
                
                # 颜色距离
                dc = np.sqrt(np.sum((region_colors[r] - region_colors[s]) ** 2))
                
                # 空间距离
                cy_r, cx_r = region_centers[r]
                cy_s, cx_s = region_centers[s]
                ds = np.sqrt((cy_r - cy_s) ** 2 + (cx_r - cx_s) ** 2)
                spatial_weight = np.exp(-ds / self.sigma_spatial)
                
                # 区域大小权重
                size_weight = region_sizes[s] / total_pixels
                
                sal += size_weight * spatial_weight * dc
            
            # 标记该区域
            mask = (region_labels == r)
            saliency_map[mask] = sal
        
        # 归一化
        s_min, s_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - s_min) / (s_max - s_min + 1e-8)
        saliency_map = gaussian_filter(saliency_map, sigma=3)
        
        return saliency_map


def demo():
    np.random.seed(42)
    h, w = 64, 64
    img = np.random.rand(h, w, 3) * 0.3 + 0.35
    img[20:32, 24:38, :] = 0.9
    img[45:52, 10:18, :] = [0.9, 0.1, 0.1]
    
    # HC
    print("=== HC ===")
    hc = HC(n_colors=64, smooth_neighbors=5)
    smap_hc = hc.compute_saliency(img)
    print(f"HC: [{smap_hc.min():.3f}, {smap_hc.max():.3f}]")
    print(f"  矩形区域: {smap_hc[20:32, 24:38].mean():.4f}")
    
    # RC
    print("\n=== RC ===")
    rc = RC(n_regions=50, sigma_spatial=0.4)
    smap_rc = rc.compute_saliency(img)
    print(f"RC: [{smap_rc.min():.3f}, {smap_rc.max():.3f}]")
    print(f"  矩形区域: {smap_rc[20:32, 24:38].mean():.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img); axes[0].set_title('原始图像'); axes[0].axis('off')
    axes[1].imshow(smap_hc, cmap='hot'); axes[1].set_title('HC显著性'); axes[1].axis('off')
    axes[2].imshow(smap_rc, cmap='hot'); axes[2].set_title('RC显著性'); axes[2].axis('off')
    plt.tight_layout(); plt.savefig('hc_rc_saliency.png', dpi=150)


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""HC和RC - 手工K-means实现"""
import numpy as np


def kmeans_manual(data, k, max_iter=20):
    """手工K-means聚类"""
    n, d = data.shape
    # 随机初始化中心
    indices = np.random.choice(n, k, replace=False)
    centers = data[indices].copy()
    
    for _ in range(max_iter):
        # 分配标签
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.sqrt(np.sum((data[i] - centers) ** 2, axis=1))
            labels[i] = np.argmin(dists)
        
        # 更新中心
        new_centers = np.zeros((k, d))
        for j in range(k):
            mask = (labels == j)
            if mask.sum() > 0:
                new_centers[j] = data[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]
        
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return labels, centers


def hc_manual(image, n_colors=32):
    """手工HC实现"""
    h, w, c = image.shape
    pixels = image.reshape(-1, c).astype(float)
    
    labels, colors = kmeans_manual(pixels, n_colors)
    n_pixels = len(pixels)
    
    freqs = np.array([(labels == j).sum() / n_pixels for j in range(n_colors)])
    
    # 颜色显著性
    color_sal = np.zeros(n_colors)
    for i in range(n_colors):
        dists = np.sqrt(np.sum((colors[i] - colors) ** 2, axis=1))
        color_sal[i] = np.sum(freqs * dists)
    
    c_min, c_max = color_sal.min(), color_sal.max()
    color_sal = (color_sal - c_min) / (c_max - c_min + 1e-8)
    
    saliency = color_sal[labels].reshape(h, w)
    return saliency


def test_hc_manual():
    np.random.seed(42)
    img = np.random.rand(16, 16, 3) * 0.3 + 0.35
    img[4:10, 5:12, :] = 0.9
    smap = hc_manual(img, n_colors=16)
    print(f"HC手工: [{smap.min():.3f}, {smap.max():.3f}]")
    assert smap[4:10, 5:12].mean() > smap.mean(), "显著区域应高于均值"


if __name__ == "__main__":
    test_hc_manual()
```

---

## 9. 可视化与结果理解

### 9.1 HC显著性图

- 均匀背景的颜色频率高，显著性低
- 罕见颜色显著性高
- 颜色空间平滑消除量化伪影

### 9.2 RC显著性图

- 加入空间距离后，远离当前区域的区域贡献减小
- 与大区域颜色差异大的小区域显著性高
- 比HC更精确，边界更清晰

### 9.3 HC vs RC 对比

HC速度快但忽略了空间结构；RC效果更好但计算量大。RC通常比HC在标准数据集上高出5-10%的AUC。

---

## 10. 模型评估

```python
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_hc_rc():
    h, w = 48, 48
    img = np.random.rand(h, w, 3) * 0.3 + 0.35
    img[15:28, 18:32, :] = 0.9
    gt = np.zeros((h, w))
    gt[15:28, 18:32] = 1.0
    
    hc = HC(n_colors=32)
    smap_hc = hc.compute_saliency(img)
    auc_hc = roc_auc_score(gt.flatten() > 0, smap_hc.flatten())
    
    rc = RC(n_regions=30)
    smap_rc = rc.compute_saliency(img)
    auc_rc = roc_auc_score(gt.flatten() > 0, smap_rc.flatten())
    
    print(f"HC AUC: {auc_hc:.4f}")
    print(f"RC AUC: {auc_rc:.4f}")


if __name__ == "__main__":
    evaluate_hc_rc()
```

---

## 11. 常见问题与易错点

### Q1: HC为什么要做颜色量化？
**A:** 不量化时，每个像素需遍历所有其他像素（$O(N^2)$）。量化为85种颜色后，只需 $O(85^2)$，加速数千倍。

### Q2: RC中空间距离为什么用指数衰减？
**A:** 指数衰减 $\exp(-d/\sigma)$ 使邻近区域的影响远大于远处区域，符合"局部对比度更重要"的直觉。

### Q3: HC和RC为什么不使用频域方法？
**A:** 全局对比度在空域计算比频域更直观，且在自然图像上效果优于当时的频域方法。

### Q4: 颜色空间平滑的作用？
**A:** 量化导致相似颜色被分到不同簇，显著性可能突变。平滑使相似颜色的显著性趋于一致。

### Q5: HC和RC在实时场景中能用吗？
**A:** HC可以（约10fps在CPU上，640x480图像），RC较慢（约1fps）。使用SLIC超像素分割可加速RC。

---

## 12. 学习总结

**核心要点：**
1. 显著性 = 全局颜色对比度的频率加权和
2. HC：像素级，无空间信息，速度快
3. RC：区域级，含空间信息，效果好
4. 颜色量化 + K-means 加速计算
5. SaliencyCut = RC + GrabCut

**公式总结：**
- HC: $S(c_k) = \sum_j f_j \cdot \|c_k - c_j\|$
- RC: $S(R_k) = \sum_{i \neq k} w_i \cdot e^{-d_{spatial}/\sigma} \cdot \|c_k - c_i\|$

---

## 13. 练习题与思考题

### 基础题

**1.** 为什么HC中使用LAB颜色空间而不是RGB？

<details>
<summary>答案</summary>
LAB的欧氏距离更接近人眼的感知差异。RGB空间中相同的欧氏距离可能对应不同的感知差异。LAB中 $L$ 通道的差异和 $ab$ 通道的差异在感知上意义明确。
</details>

**2.** K-means的颜色数 $n$ 越大越好吗？

<details>
<summary>答案</summary>
不是。$n$ 太小会丢失颜色细节，显著区域不准确；$n$ 太大会接近原始颜色数，失去加速优势且可能过拟合。$n=85$ 是论文推荐的平衡值。
</details>

**3.** RC中 $\sigma$ 的作用是什么？

<details>
<summary>答案</summary>
$\sigma$ 控制空间距离的影响范围。$\sigma$ 小则只有邻近区域影响显著性（局部对比度），$\sigma$ 大则全局所有区域都影响。典型值 0.2-0.4（归一化坐标）。
</details>

### 进阶题

**4.** 推导HC的 $O(N^2)$ 到 $O(n^2)$ 的加速过程。

<details>
<summary>答案</summary>
原始: $S(p_k) = \sum_{j=1}^N \|p_k - p_j\|$，$O(N^2)$。颜色量化后，将像素映射为 $n$ 种颜色: $S(c_k) = \sum_{j=1}^n f_j \cdot \|c_k - c_j\|$，$O(n^2)$。其中 $f_j$ 是颜色频率，$n \ll N$。
</details>

**5.** 设计实验比较HC、RC与SR、IS等频域方法的性能。

<details>
<summary>答案</summary>
(1) 在MIT1003、ECSSD等标准数据集上计算AUC、NSS等指标；(2) HC、RC在均匀背景上优于频域方法；(3) 频域方法在纹理背景上可能更鲁棒；(4) HC在速度上最优。
</details>

---

## 14. 学习路径建议

### 预备知识
- 颜色空间（RGB/LAB）
- 聚类算法（K-means）
- 图像分割基础

### 进阶方向
1. **HC/RC -> SaliencyCut**：显著性驱动的图像分割
2. **HC/RC -> Deep Saliency**：深度显著性与HC/RC的融合
3. **HC/RC -> BMS**：基于Boolean Map的显著性检测

### 推荐阅读
- Cheng et al. "Global Contrast based Salient Region Detection." CVPR 2011.
- Cheng et al. "Salient Object Detection and Segmentation." TPAMI 2015.
- Achanta et al. "Frequency-tuned Salient Region Detection." CVPR 2009.

### 项目实践
1. 在ECSSD数据集上复现HC和RC的性能
2. 使用SLIC超像素替代K-means改进RC
3. 实现SaliencyCut（RC + GrabCut）
