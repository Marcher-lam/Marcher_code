# SF显著性滤波模型 学习文档

> 用高斯滤波加速的全局对比度显著性检测。
> 来源线索：原书第2.2.2节"SF：基于超像素分割的显著性滤波"。

---

## 1. 算法基础认知

**一句话定义：** SF（Saliency Filters）由Perazzi等人于2012年提出，基于SLIC超像素分割和对比度传播滤波实现显著性检测，巧妙地将对比度计算转化为高斯滤波操作，将算法复杂度从 O(N^2) 降至 O(N)。

**核心思想：** 显著性可以分解为两个因素：
1. **唯一性（Uniqueness）**：区域在颜色上与其他区域的差异程度
2. **分布（Distribution）**：区域颜色的空间分散程度

这两个因素都可以通过高斯滤波在超像素图上高效计算，避免了逐像素对对比度的暴力计算。

**关键创新：** 对比度计算通常需要计算每个区域与所有其他区域的加权颜色差异（O(N^2)）。SF发现，在适当假设下，这个运算等价于对颜色特征进行高斯滤波（O(N)）。

---

## 2. 核心原理

### 2.1 整体框架

输入图像 -> SLIC超像素分割 -> 提取超像素特征 -> 计算唯一性（颜色高斯滤波）-> 计算分布（空间方差）-> 融合 -> 像素级上采样 -> 输出显著图

### 2.2 SLIC超像素分割

SLIC（Simple Linear Iterative Clustering）将图像分割为 K 个紧凑的超像素。每个超像素 i 具有：
- 平均颜色 c_i（LAB空间）
- 中心位置 p_i（归一化坐标）

### 2.3 唯一性（Uniqueness）

超像素 i 的唯一性定义为它与所有其他超像素的加权颜色差异：

U_i = sum_j ||c_i - c_j||^2 * w_ij^(p)

其中 w_ij^(p) = (1/Z_i) * exp(-||p_i - p_j||^2 / (2*sigma_p^2)) 是空间距离权重。

### 2.4 分布（Distribution）

超像素 i 的分布衡量其颜色在图像中的空间分散程度：

D_i = sum_j ||p_j - mu_i||^2 * w_ij^(c)

其中 w_ij^(c) = (1/Z_i) * exp(-||c_i - c_j||^2 / (2*sigma_c^2)) 是颜色相似度权重。

### 2.5 显著性融合

S_i = U_i * exp(-k * D_i)

其中 k 控制分布项的惩罚强度。

---

## 3. 数学公式与推导

### 3.1 唯一性的滤波形式

将唯一性公式展开：

U_i = sum_j ||c_i - c_j||^2 * w_ij^(p) = ||c_i||^2 - 2*c_i * G(c_i) + G(||c||^2)_i

其中 G* 表示以 sigma_p 为核宽的高斯滤波。因此唯一性可以分解为三个高斯滤波操作。

### 3.2 分布的滤波形式

D_i = G(||p||^2)_i - ||G(p)_i||^2

即分布等于颜色加权位置方差，同样可以通过两个高斯滤波计算。

验证：D_i = E_w[||p||^2] - ||E_w[p]||^2 = Var_w(p)

### 3.3 最终显著性公式

S_i = [||c||^2 - 2c*G(c) + G(||c||^2)] * exp(-k * [G(||p||^2) - ||G(p)||^2])

所有涉及 N^2 运算的对比度计算都被转化为 O(N) 的高斯滤波操作。

---

## 4. 训练过程讲解

SF是无监督方法，不需要训练。

### 4.1 处理流程

输入：RGB图像 I (H x W x 3)
参数：超像素数 K=250, sigma_p=0.25, sigma_c=20, k=6

1. SLIC超像素分割 -> K个超像素
2. 提取特征：颜色均值 c_i (LAB)，位置中心 p_i (归一化)
3. 计算唯一性 U_i：||c_i||^2 - 2*c_i*G(c_i) + G(||c_i||^2)
4. 计算分布 D_i：G(||p||^2)_i - ||G(p)_i||^2
5. 融合：S_i = U_i * exp(-k * D_i)
6. 归一化后映射回像素，高斯平滑

### 4.2 参数调优建议

| 参数 | 作用 | 推荐值 | 调高效果 |
|------|------|--------|---------|
| sigma_p | 空间影响范围 | 0.25 | 更多全局对比度 |
| sigma_c | 颜色相似度阈值 | 20 | 颜色分群更粗糙 |
| k | 分布惩罚强度 | 6 | 背景分布广的区域被更多抑制 |

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 图像分割 | 显著性图作为前景检测的先验 |
| 内容感知编辑 | 保护显著区域，修改背景 |
| 图像检索 | 基于显著区域的特征提取 |
| 物体检测 | 快速定位潜在物体区域 |
| 图像压缩 | 显著区域高质量编码 |

---

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 计算高效，O(N)复杂度 | 依赖于SLIC超像素质量 |
| 唯一性+分布双线索互补 | 参数较多需手动调节 |
| 超像素级别运算，抗噪性强 | 无法利用高层语义信息 |
| 数学推导严密，滤波加速巧妙 | 小物体可能被超像素合并丢失 |
| 全分辨率输出，边缘保持较好 | 对颜色相似的前景和背景区分困难 |

---

## 7. 调库实现（scikit-image）

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, segmentation
from scipy.ndimage import gaussian_filter


class SaliencyFilters:
    def __init__(self, n_superpixels=250, sigma_p=0.25, sigma_c=20.0, k=6.0):
        self.n_superpixels = n_superpixels
        self.sigma_p = sigma_p
        self.sigma_c = sigma_c
        self.k = k

    def _segment_superpixels(self, image):
        if image.max() > 1.0:
            image = image / 255.0
        lab = color.rgb2lab(image.astype(np.float64))
        segments = segmentation.slic(image, n_segments=self.n_superpixels,
                                     compactness=20, sigma=1, start_label=0)
        K = segments.max() + 1
        colors = np.zeros((K, 3))
        positions = np.zeros((K, 2))
        counts = np.zeros(K)
        h, w = lab.shape[:2]
        for i in range(h):
            for j in range(w):
                sid = segments[i, j]
                colors[sid] += lab[i, j]
                positions[sid] += [i / h, j / w]
                counts[sid] += 1
        colors /= counts[:, None]
        positions /= counts[:, None]
        return segments, colors, positions, K

    def _compute_uniqueness(self, colors):
        color_norm = np.sum(colors ** 2, axis=1)
        colors_filt = np.zeros_like(colors)
        for c in range(3):
            colors_filt[:, c] = gaussian_filter(colors[:, c], sigma=self.sigma_p, mode='wrap')
        norm_filt = gaussian_filter(color_norm, sigma=self.sigma_p, mode='wrap')
        dot = np.sum(colors * colors_filt, axis=1)
        return np.maximum(color_norm - 2 * dot + norm_filt, 0)

    def _compute_distribution(self, positions):
        pos_filt = np.zeros_like(positions)
        for d in range(2):
            pos_filt[:, d] = gaussian_filter(positions[:, d], sigma=self.sigma_c, mode='wrap')
        pos_norm = np.sum(positions ** 2, axis=1)
        pos_norm_filt = gaussian_filter(pos_norm, sigma=self.sigma_c, mode='wrap')
        mu_norm = np.sum(pos_filt ** 2, axis=1)
        return np.maximum(pos_norm_filt - mu_norm, 0)

    def compute_saliency(self, image):
        segments, colors, positions, K = self._segment_superpixels(image)
        U = self._compute_uniqueness(colors)
        D = self._compute_distribution(positions)
        S = U * np.exp(-self.k * D)
        S = (S - S.min()) / (S.max() - S.min() + 1e-8)
        saliency = S[segments]
        saliency = gaussian_filter(saliency, sigma=2)
        return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)


def demo_sf():
    np.random.seed(42)
    img = np.ones((100, 100, 3), dtype=np.float64) * 0.2
    img[25:75, 25:75] = [0.7, 0.3, 0.3]
    model = SaliencyFilters(n_superpixels=50, sigma_p=0.25, sigma_c=20, k=6)
    saliency = model.compute_saliency(img)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title('Input'); axes[0].axis('off')
    im1 = axes[1].imshow(saliency, cmap='jet')
    axes[1].set_title('SF Saliency'); axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    segs = segmentation.slic(img, n_segments=50, compactness=20, sigma=1, start_label=0)
    axes[2].imshow(segmentation.mark_boundaries(img, segs))
    axes[2].set_title('Superpixels'); axes[2].axis('off')
    plt.tight_layout(); plt.savefig('sf_demo.png', dpi=150); plt.show()
    print(f"SF: [{saliency.min():.3f}, {saliency.max():.3f}]")

if __name__ == '__main__':
    demo_sf()
```

---

## 8. 手工代码实现（NumPy）

```python
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class SFNumpy:
    def __init__(self, n_superpixels=100, sigma_p=0.25, sigma_c=20.0, k=6.0):
        self.n_superpixels = n_superpixels
        self.sigma_p = sigma_p
        self.sigma_c = sigma_c
        self.k = k

    def _simple_superpixel(self, image):
        h, w = image.shape[:2]
        ratio = np.sqrt(self.n_superpixels * h / w)
        grid_h = max(1, int(ratio))
        grid_w = max(1, self.n_superpixels // grid_h)
        features, positions = [], []
        step_h = max(1, h // grid_h)
        step_w = max(1, w // grid_w)
        for i in range(0, h, step_h):
            for j in range(0, w, step_w):
                patch = image[max(0,i-2):min(h,i+3), max(0,j-2):min(w,j+3)]
                if patch.size > 0:
                    features.append(patch.mean(axis=(0,1)))
                    positions.append([i/h, j/w])
        if len(features) < 10:
            return None, None, None
        features_arr = np.array(features)
        positions_arr = np.array(positions)
        kmeans = KMeans(n_clusters=min(self.n_superpixels, len(features_arr)),
                        random_state=42, n_init=3, max_iter=10)
        kmeans.fit(np.concatenate([features_arr, positions_arr * 0.3], axis=1))
        centers = kmeans.cluster_centers_
        colors = centers[:, :3]
        sp_pos = centers[:, 3:]
        seg_map = np.zeros((h, w), dtype=np.int32)
        for i in range(h):
            for j in range(w):
                feat = np.concatenate([image[i,j], np.array([i/h, j/w]) * 0.3])
                seg_map[i,j] = cdist(feat[None,:], centers)[0].argmin()
        return seg_map, colors, sp_pos

    def compute_saliency(self, image):
        if image.max() > 1.0:
            image = image / 255.0
        seg_map, colors, positions = self._simple_superpixel(image)
        if seg_map is None:
            return np.zeros(image.shape[:2])
        cn = np.sum(colors**2, 1)
        cf = np.zeros_like(colors)
        for c in range(3):
            cf[:,c] = gaussian_filter(colors[:,c], self.sigma_p, mode='wrap')
        nf = gaussian_filter(cn, self.sigma_p, mode='wrap')
        U = np.maximum(cn - 2*np.sum(colors*cf,1) + nf, 0)
        pf = np.zeros_like(positions)
        for d in range(2):
            pf[:,d] = gaussian_filter(positions[:,d], self.sigma_c, mode='wrap')
        pn = np.sum(positions**2, 1)
        D = np.maximum(gaussian_filter(pn, self.sigma_c, mode='wrap') - np.sum(pf**2, 1), 0)
        S = U * np.exp(-self.k * D)
        S = (S - S.min()) / (S.max() - S.min() + 1e-8)
        sal = S[seg_map]
        sal = gaussian_filter(sal, 1)
        return (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)


def demo_numpy():
    np.random.seed(42)
    img = np.random.rand(48, 48, 3).astype(np.float64)
    img[15:33, 15:33] = [0.8, 0.2, 0.2]
    m = SFNumpy(n_superpixels=50)
    s = m.compute_saliency(img)
    print(f"SF手工: [{s.min():.3f}, {s.max():.3f}]")

if __name__ == '__main__':
    demo_numpy()
```

---

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, segmentation
from scipy.ndimage import gaussian_filter


def visualize_sf_process():
    np.random.seed(42)
    img = np.ones((100,100,3), dtype=np.float64) * 0.15
    img[20:80,20:80] = [0.65,0.25,0.25]
    img += np.random.randn(*img.shape) * 0.03
    img = np.clip(img, 0, 1)
    lab = color.rgb2lab(img)
    segs = segmentation.slic(img, 60, compactness=20, sigma=1, start_label=0)
    K = segs.max() + 1
    cols = np.zeros((K,3)); poss = np.zeros((K,2)); cnts = np.zeros(K)
    h,w = lab.shape[:2]
    for i in range(h):
        for j in range(w):
            s = segs[i,j]; cols[s] += lab[i,j]; poss[s] += [i/h, j/w]; cnts[s] += 1
    cols /= cnts[:,None]; poss /= cnts[:,None]
    cn = np.sum(cols**2,1)
    U = np.maximum(cn - 2*np.sum(cols*gaussian_filter(cols,0.25,mode='wrap'),1)
                   + gaussian_filter(cn,0.25,mode='wrap'), 0)
    pf = gaussian_filter(poss, 20, mode='wrap')
    D = np.maximum(gaussian_filter(np.sum(poss**2,1),20,mode='wrap') - np.sum(pf**2,1), 0)
    fig, axes = plt.subplots(2,3,figsize=(15,10))
    axes[0,0].imshow(img); axes[0,0].set_title('(a) Input'); axes[0,0].axis('off')
    axes[0,1].imshow(segmentation.mark_boundaries(img, segs))
    axes[0,1].set_title(f'(b) SLIC K={K}'); axes[0,1].axis('off')
    um = np.zeros((h,w))
    for i in range(K): um[segs==i] = U[i]
    im = axes[0,2].imshow(um, cmap='hot'); axes[0,2].set_title('(c) Uniqueness'); axes[0,2].axis('off')
    plt.colorbar(im, ax=axes[0,2], fraction=0.046)
    dm = np.zeros((h,w))
    for i in range(K): dm[segs==i] = D[i]
    im = axes[1,0].imshow(dm, cmap='Blues'); axes[1,0].set_title('(d) Distribution'); axes[1,0].axis('off')
    plt.colorbar(im, ax=axes[1,0], fraction=0.046)
    wm = np.exp(-6*dm)
    im = axes[1,1].imshow(wm, cmap='Greens'); axes[1,1].set_title('(e) Weight'); axes[1,1].axis('off')
    plt.colorbar(im, ax=axes[1,1], fraction=0.046)
    sm = um * wm; sm = gaussian_filter(sm,1)
    sm = (sm-sm.min())/(sm.max()-sm.min()+1e-8)
    im = axes[1,2].imshow(sm, cmap='jet'); axes[1,2].set_title('(f) Final'); axes[1,2].axis('off')
    plt.colorbar(im, ax=axes[1,2], fraction=0.046)
    plt.suptitle('SF流程', fontsize=14); plt.tight_layout()
    plt.savefig('sf_process.png',dpi=150); plt.show()
    print("SF可视化已保存")

if __name__ == '__main__':
    visualize_sf_process()
```

---

## 10. 模型评估

### 10.1 评估方法
SF使用标准SOD评估协议：PR曲线、F-measure (beta^2=0.3)、MAE。

### 10.2 在公开数据集上的性能
| 方法 | F-measure(ASD) | MAE(ASD) |
|------|----------------|----------|
| FT | 0.624 | 0.178 |
| AC | 0.554 | 0.201 |
| RC | 0.714 | 0.145 |
| SF | 0.736 | 0.131 |

### 10.3 评估实现
```python
def evaluate(saliency, gt_mask, thresholds=np.linspace(0,1,256)):
    precisions, recalls = [], []
    for t in thresholds:
        bin_ = (saliency > t).astype(np.int32)
        tp = np.sum((bin_ == 1) & (gt_mask > 0.5))
        fp = np.sum((bin_ == 1) & (gt_mask <= 0.5))
        fn = np.sum((bin_ == 0) & (gt_mask > 0.5))
        precisions.append(tp/(tp+fp+1e-8))
        recalls.append(tp/(tp+fn+1e-8))
    mae = np.mean(np.abs(saliency - gt_mask))
    return np.array(precisions), np.array(recalls), mae
```

---

## 11. 常见问题与易错点

### Q1: 为什么唯一性可以用高斯滤波加速？
A: 展开唯一性公式后，求和化为高斯滤波，将O(N^2)降为O(N)。

### Q2: SF中的"分布"衡量什么？
A: 颜色的空间分散程度。天空的蓝色分布广则D高被抑制，红色花朵集中则D低。

### Q3: 超像素数量如何选择？
A: 太少则边界不准，太多则失去区域抽象优势。典型值200-400。

### Q4: SF与RC区别？
A: RC基于直方图对比度；SF基于超像素+高斯滤波加速，引入"分布"概念。

### Q5: 参数sigma_p越大越好？
A: 否。sigma_p控制空间影响范围，过大导致全局对比度模糊，过小则只有局部对比度。

---

## 12. 学习总结

### 12.1 核心要点
- 超像素处理：SLIC将图像从像素级抽象为区域级
- 唯一性：颜色差异加权和，通过高斯滤波加速
- 分布：颜色空间分散程度，通过高斯滤波加速
- 融合：唯一性 x 空间紧凑性

### 12.2 算法复杂度魅力
SF核心贡献是将O(N^2)对比度计算转化为O(N)高斯滤波。

### 12.3 局限性
超像素质量决定上限、缺乏语义信息、参数需手动调整。

---

## 13. 练习题与思考题（含答案）

### 练习1：验证滤波加速
题目：证明 sum_j ||c_i-c_j||^2 w_ij = ||c_i||^2 - 2c_i*(G*c)_i + (G*||c||^2)_i。

答案：展开平方项，利用 sum w_ij = 1，得到三项：||c_i||^2、-2c_i*sum w_ij c_j、sum w_ij ||c_j||^2。后两项即高斯滤波。

### 练习2：分布的理解
题目：为什么分布可以抑制背景？

答案：背景颜色分布广，位置方差D大，exp(-kD)小，显著性被抑制。前景颜色集中，D小，惩罚小。

### 练习3：参数选择
题目：sigma_c过大或过小的影响？

答案：过大则所有颜色相似，D趋近全局方差，失去区分度。过小则仅相同颜色被视为相似，D趋近0，分布项失效。

### 练习4：思考题
题目：如何改进SF以利用语义信息？

答案：1. 在超像素特征中加入CNN深度特征；2. 用数据集训练超像素级分类器；3. 结合语义分割结果赋予类别先验。

---

## 14. 学习路径建议

### 前置知识
1. 超像素分割：SLIC算法
2. 图像滤波：高斯滤波、卷积
3. LAB色彩空间

### 后续学习
1. MR（流形排序）显著性检测
2. DS（密集显著性）
3. 深度显著物体检测：MDF, U-Net, BASNet
4. 视频显著性检测

### 推荐文献
1. Perazzi F, et al. "Saliency filters." CVPR 2012.
2. Achanta R, et al. "SLIC superpixels." TPAMI 2012.
3. Cheng M-M, et al. "Global contrast based salient region detection." CVPR 2011.
