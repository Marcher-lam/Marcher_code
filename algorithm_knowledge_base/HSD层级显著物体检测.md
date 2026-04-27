# HSD层级显著物体检测 学习文档

> 层级架构下的显著物体检测——用树状结构融合多尺度显著性。
> 来源线索：原书第2.2.2节"HSD：使用层级架构的显著物体检测"。

---

## 1. 算法基础认知

**一句话定义：** HSD（Hierarchical Saliency Detection）由Qiong Yan等人于2013年提出，通过多尺度分割、多尺度显著性线索提取和层级显著性推断实现高质量显著物体检测。

**核心思想：** 不同尺度的视觉信息包含不同细节。HSD通过分水岭分割在三个尺度上提取局部对比和位置启发两种显著性线索，再利用树状图和信念传播进行层级融合。

**层级架构的三层含义：**
1. 多尺度分割层级：细、中、粗三个分割尺度
2. 显著性线索层级：局部对比度 + 位置先验
3. 推理层级：从细到粗的层级信念传播

---

## 2. 核心原理

### 2.1 多尺度分割

HSD使用分水岭算法在不同阈值下生成三个尺度的分割结果：
- 细尺度：分割区域多(~500区域)，保留细节
- 中尺度：分割区域适中(~200区域)，平衡细节与结构
- 粗尺度：分割区域少(~50区域)，整体结构

### 2.2 显著性线索

在每个尺度上提取两种线索：

**局部对比度线索：**
C_i = sum_{j in N_i} w_ij * ||c_i - c_j||
其中 w_ij = exp(-||p_i - p_j||^2 / sigma_p^2)

**位置启发线索（中心偏置）：**
P_i = exp(-||p_i - c||^2 / sigma_c^2)

### 2.3 层级融合

构建树状图结构，自底向上融合多尺度的显著性信息：

S_i^{(l)} = alpha * C_i^{(l)} + (1-alpha) * P_i^{(l)} + beta * S_{parent(i)}^{(l+1)}

---

## 3. 数学公式与推导

### 3.1 分水岭分割

分水岭算法将图像视为地形图，灰度值高度，从局部最小值开始"注水"：

- 在不同阈值 T_k 下分割：Region_k = Watershed(I, T_k), k=1,2,3
- 区域合并：合并面积小于阈值 A_min 的区域到最相似邻域

### 3.2 局部对比度线索

对于区域 R_i，其局部对比度为：

C_i = (1/|N_i|) * sum_{j in N_i} ||mu_i - mu_j|| * exp(-d_ij^2 / (2*sigma_s^2))

其中：
- mu_i, mu_j 是区域LAB颜色均值
- d_ij 是区域中心之间的空间距离
- sigma_s 控制空间影响范围

### 3.3 位置启发线索

P_i = exp(-||p_i - p_0||^2 / (2*sigma_p^2))

其中 p_i 是区域中心坐标，p_0 是图像中心坐标。

### 3.4 层级显著性推断

构建三层级结构：细(L1) -> 中(L2) -> 粗(L3)

S_i^{(L3)} = w1 * C_i^{(L3)} + w2 * P_i^{(L3)}
S_i^{(L2)} = w1 * C_i^{(L2)} + w2 * P_i^{(L2)} + w3 * S_{parent}^{(L3)}
S_i^{(L1)} = w1 * C_i^{(L1)} + w2 * P_i^{(L1)} + w3 * S_{parent}^{(L2)}

---

## 4. 训练过程讲解

HSD的参数（w1, w2, w3, sigma_s, sigma_p）需通过学习或手工设定。

### 4.1 训练/设定方法

1. 在验证集上通过网格搜索优化参数
2. 优化目标：最大化F-measure或最小化MAE
3. 参数搜索范围：
   - w1, w2, w3: [0, 1] 步长0.1
   - sigma_s: [0.1, 0.5] 步长0.05
   - sigma_p: [0.2, 0.8] 步长0.1

### 4.2 处理流程

输入：RGB图像 I, 3个尺度, 融合权重

1. RGB -> LAB转换
2. 对每个尺度 k = 1,2,3:
   a. 分水岭分割 -> 区域集合 R_k
   b. 提取区域颜色均值 mu_i 和位置中心 p_i
   c. 构建邻域关系图, 计算局部对比度 C_i
   d. 计算位置启发 P_i
3. 层级显著性推断（自底向上）
4. 将细尺度的区域显著性映射到像素
5. 高斯平滑后处理

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 自然图像分割 | 层级结构适应不同尺度的显著物体 |
| 遥感图像分析 | 多尺度处理适合遥感中的大小目标 |
| 医学图像分析 | 病灶大小不一，层级处理更具鲁棒性 |
| 图像缩略图 | 保留多尺度显著信息 |

---

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 多尺度融合充分利用层级信息 | 分水岭分割对噪声敏感 |
| 层级推断捕获跨尺度关系 | 参数较多，调参复杂 |
| 显著图质量高、边缘保持好 | 计算量大，速度慢 |
| 树状图结构扩展性好 | 对低对比度物体效果差 |

---

## 7. 调库实现（scikit-image）

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, segmentation
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist


class HSD:
    def __init__(self, n_scales=3, w1=0.4, w2=0.3, w3=0.3, sigma_s=0.3, sigma_p=0.4):
        self.n_scales = n_scales
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.sigma_s = sigma_s
        self.sigma_p = sigma_p

    def _segment_scale(self, image, scale):
        """多尺度过分割: 使用网格模拟不同尺度的分割"""
        gray = color.rgb2gray(image)
        h, w = gray.shape
        grid_h = max(4, h // (20 * (scale + 1)))
        grid_w = max(4, w // (20 * (scale + 1)))
        labels = np.zeros((h, w), dtype=np.int32)
        label_id = 0
        for i in range(0, h, grid_h):
            for j in range(0, w, grid_w):
                i_end = min(h, i + grid_h)
                j_end = min(w, j + grid_w)
                labels[i:i_end, j:j_end] = label_id
                label_id += 1
        return labels, label_id

    def _extract_region_features(self, image, segments, K):
        lab = color.rgb2lab(image)
        h, w = lab.shape[:2]
        colors = np.zeros((K, 3))
        positions = np.zeros((K, 2))
        counts = np.zeros(K)
        for i in range(h):
            for j in range(w):
                sid = segments[i, j]
                if sid < K:
                    colors[sid] += lab[i, j]
                    positions[sid] += [i / h, j / w]
                    counts[sid] += 1
        colors = colors / (counts[:, None] + 1e-8)
        positions = positions / (counts[:, None] + 1e-8)
        return colors, positions

    def _compute_local_contrast(self, colors, positions):
        K = len(colors)
        contrast = np.zeros(K)
        dists = cdist(positions, positions)
        for i in range(K):
            spatial_w = np.exp(-dists[i] ** 2 / (2 * self.sigma_s ** 2))
            color_diff = np.sqrt(np.sum((colors - colors[i]) ** 2, axis=1))
            contrast[i] = np.sum(spatial_w * color_diff) / (np.sum(spatial_w) + 1e-8)
        return (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)

    def _compute_center_bias(self, positions):
        center = np.array([0.5, 0.5])
        dist = np.sqrt(np.sum((positions - center) ** 2, axis=1))
        bias = np.exp(-dist ** 2 / (2 * self.sigma_p ** 2))
        return (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)

    def compute_saliency(self, image):
        if image.max() > 1.0:
            image = image / 255.0
        h, w = image.shape[:2]
        all_saliencies = []
        for scale in range(self.n_scales):
            segments, K = self._segment_scale(image, scale)
            colors, positions = self._extract_region_features(image, segments, K)
            C = self._compute_local_contrast(colors, positions)
            P = self._compute_center_bias(positions)
            S = self.w1 * C + self.w2 * P
            smap = np.zeros((h, w))
            for i in range(K):
                smap[segments == i] = S[i]
            all_saliencies.append(smap)
        final = np.mean(all_saliencies, axis=0)
        final = gaussian_filter(final, sigma=2)
        return (final - final.min()) / (final.max() - final.min() + 1e-8)


def demo_hsd():
    np.random.seed(42)
    img = np.ones((100, 100, 3), dtype=np.float64) * 0.2
    img[25:75, 25:75] = [0.7, 0.3, 0.3]
    model = HSD(n_scales=3)
    saliency = model.compute_saliency(img)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img); axes[0].set_title('Input'); axes[0].axis('off')
    im = axes[1].imshow(saliency, cmap='jet')
    axes[1].set_title('HSD Saliency'); axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout(); plt.savefig('hsd_demo.png', dpi=150); plt.show()
    print(f"HSD: [{saliency.min():.3f}, {saliency.max():.3f}]")

if __name__ == '__main__':
    demo_hsd()
```

---

## 8. 手工代码实现（NumPy）

```python
import numpy as np
from scipy.ndimage import gaussian_filter


class HSDNumpy:
    def __init__(self, n_scales=3):
        self.n_scales = n_scales

    def compute_saliency(self, image):
        if image.max() > 1.0:
            image = image / 255.0
        gray = np.mean(image, axis=2)
        h, w = gray.shape
        saliency_scales = []
        for scale in range(self.n_scales):
            sigma1 = 2 ** (scale + 1)
            sigma2 = sigma1 * 2
            blurred1 = gaussian_filter(gray, sigma1)
            blurred2 = gaussian_filter(gray, sigma2)
            local_contrast = np.abs(blurred1 - blurred2)
            y, x = np.mgrid[0:h, 0:w]
            cy, cx = h / 2, w / 2
            center_bias = np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * (min(h, w) / 4)**2))
            sal = local_contrast * center_bias
            saliency_scales.append(sal)
        final = np.mean(saliency_scales, axis=0)
        final = gaussian_filter(final, sigma=1)
        return (final - final.min()) / (final.max() - final.min() + 1e-8)


def demo_numpy():
    np.random.seed(42)
    img = np.random.rand(64, 64, 3).astype(np.float64)
    img[20:45, 20:45] = [0.8, 0.2, 0.2]
    model = HSDNumpy(n_scales=3)
    smap = model.compute_saliency(img)
    print(f"HSD手工: [{smap.min():.3f}, {smap.max():.3f}]")

if __name__ == '__main__':
    demo_numpy()
```

---

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def visualize_hsd_scales():
    np.random.seed(42)
    img = np.ones((100, 100, 3), dtype=np.float64) * 0.2
    img[30:70, 30:70] = [0.7, 0.3, 0.3]
    gray = np.mean(img, axis=2)
    h, w = gray.shape

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(img); axes[0, 0].set_title('(a) Input'); axes[0, 0].axis('off')

    for i in range(3):
        sigma1 = 2**(i+1)
        b1 = gaussian_filter(gray, sigma1)
        b2 = gaussian_filter(gray, sigma1*2)
        lc = np.abs(b1 - b2)
        y,x = np.mgrid[0:h,0:w]
        cb = np.exp(-((y-h/2)**2+(x-w/2)**2)/(2*(min(h,w)/4)**2))
        axes[0, i+1].imshow(lc, cmap='hot')
        axes[0, i+1].set_title(f'Contrast scale={i+1}'); axes[0, i+1].axis('off')
        axes[1, i].imshow(cb, cmap='Blues')
        axes[1, i].set_title(f'Center scale={i+1}'); axes[1, i].axis('off')

    all_s = []
    for s in range(3):
        lc = np.abs(gaussian_filter(gray,2**(s+1))-gaussian_filter(gray,2**(s+2)))
        cb = np.exp(-((np.mgrid[0:h,0:w][0]-h/2)**2+(np.mgrid[0:h,0:w][1]-w/2)**2)/(2*(min(h,w)/4)**2))
        all_s.append(lc*cb)
    final = np.mean(all_s, axis=0)
    final = gaussian_filter(final, 1)
    final = (final-final.min())/(final.max()-final.min()+1e-8)
    im = axes[1, 3].imshow(final, cmap='jet')
    axes[1, 3].set_title('(h) Final Saliency'); axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046)
    plt.suptitle('HSD多尺度显著性', fontsize=14); plt.tight_layout()
    plt.savefig('hsd_scales.png', dpi=150); plt.show()
    print("HSD可视化已保存")

if __name__ == '__main__':
    visualize_hsd_scales()
```

---

## 10. 模型评估

### 10.1 HSD在公开数据集上的性能
| 方法 | F-measure | MAE |
| FT | 0.624 | 0.178 |
| SF | 0.736 | 0.131 |
| HSD | 0.755 | 0.118 |
| RC | 0.714 | 0.145 |

### 10.2 评估代码
```python
def evaluate(saliency, gt_mask):
    T = 2 * saliency.mean()
    binary = (saliency > T).astype(np.int32)
    tp = np.sum((binary==1)&(gt_mask>0.5))
    fp = np.sum((binary==1)&(gt_mask<=0.5))
    fn = np.sum((binary==0)&(gt_mask>0.5))
    prec = tp/(tp+fp+1e-8)
    rec = tp/(tp+fn+1e-8)
    f = 1.3*prec*rec/(0.3*prec+rec+1e-8)
    mae = np.mean(np.abs(saliency-gt_mask))
    return prec, rec, f, mae
```

---

## 11. 常见问题与易错点

### Q1: 为什么需要多尺度?
A: 显著物体大小不同，单一尺度无法适应。小尺度捕捉细节，大尺度捕捉整体结构。

### Q2: 层级传播的作用?
A: 粗尺度的显著性信息可以帮助细尺度消除噪声，同时细尺度的细节补充粗尺度的边界信息。

### Q3: 分水岭分割的局限?
A: 对噪声敏感，容易过分割。需要区域合并后处理。

### Q4: 为什么使用位置启发?
A: 自然图像中显著物体倾向于出现在中心区域，这是摄影构图的基本规律。

### Q5: HSD与深度学习方法的区别?
A: HSD使用手工特征和传统推理，深度方法自动学习特征，性能更好。

---

## 12. 学习总结

- 多尺度分割 + 多线索提取 + 层级融合
- 核心创新：树状图结构和跨尺度信念传播
- 局限性：手工特征天花板、计算量大

---

## 13. 练习题与思考题（含答案）

### 练习1
题目：为什么HSD的粗尺度显著性可以传播到细尺度?

答案：不同尺度的分割具有层次关系，粗尺度的全局显著性可以作为细尺度区域显著性的先验，帮助抑制噪声和突出真正显著的物体。

### 练习2
题目：如果图像中显著物体不在中心，HSD会怎样?

答案：位置启发项会对非中心物体产生抑制，导致显著性降低。改进：使用边缘连接性、闭合轮廓替代中心偏置。

### 练习3：思考题
题目：如何将HSD的层级思想与深度网络结合?

答案：类似于特征金字塔(FPN)结构。深层提供语义信息，浅层提供细节信息，通过自顶向下的路径融合。

---

## 14. 学习路径建议

### 前置知识
1. 图像分割：分水岭算法、区域合并
2. 多尺度分析：图像金字塔
3. 图模型：树状图、信念传播

### 后续学习
1. 层级深度特征：MDF, HED, SCHED
2. 特征金字塔：FPN
3. U-Net（编码器-解码器结构具有天然的层级特性）

### 推荐文献
1. Yan Q, et al. "Hierarchical saliency detection." CVPR 2013.
2. Arbelaez P, et al. "Contour detection and hierarchical image segmentation." TPAMI 2011.
3. Lin T-Y, et al. "Feature pyramid networks for object detection." CVPR 2017.
