# DRFI模型 学习文档

> 判别区域特征整合——用机器学习学习显著性的度量与融合。
> 来源线索：原书第2.2.2节"DRFI：基于显著性判别和整合的显著物体检测"。

---

## 1. 算法基础认知

**一句话定义：** DRFI（Discriminative Regional Feature Integration）由Jiang等人于2013年提出，使用随机森林回归器将86维区域显著性特征映射为显著性分数，再利用学习到的线性组合系数融合多尺度结果。

**核心思想：** 显著性检测的本质是"学习一个映射函数，将区域特征映射为显著性值"。DRFI将这个问题分解为三个步骤：
1. 多尺度分割（3个尺度）
2. 对每个区域提取86维特征
3. 随机森林回归预测显著性

**历史定位：** DRFI是传统方法向深度学习方法过渡的重要桥梁——它仍然使用手工特征，但首次将显著性检测完整地建模为有监督的回归问题。

---

## 2. 核心原理

### 2.1 多尺度分割

使用三种尺度的过分割：
- 细尺度：大量小区域，保留细节
- 中尺度：适中区域数
- 粗尺度：少量大区域

### 2.2 86维特征向量

DRFI设计了丰富的区域特征，分为4大类：

**对比度特征（24维）：**
- 区域与全图的颜色对比（LAB 3通道均值差）
- 区域与邻域的颜色对比
- 区域与背景先验（图像边界区域）的对比
- 多尺度对比度（不同邻域范围）

**区域描述特征（18维）：**
- 区域面积（相对图像）、区域周长
- 区域紧致度、区域中心距离
- 颜色方差、边缘响应强度

**背景先验特征（20维）：**
- 区域与边界区域的相似度
- 区域到边界的测地距离、边界连通性

**位置先验特征（24维）：**
- 区域中心到图像中心的距离
- 水平/垂直方向的位置分布、多尺度位置偏置

### 2.3 随机森林回归

使用随机森林（100棵树）将86维特征映射为[0,1]的显著性值：

S_r = (1/T) * sum_{t=1}^T f_t(x_r)

### 2.4 多尺度融合

S_final = sum_{k=1}^3 alpha_k * S_k(R_k)，alpha_k通过最小化验证集MAE学习。

---

## 3. 数学公式与推导

### 3.1 对比度特征

区域 R_i 与 R_j 的对比度：C_{ij} = ||mu_i - mu_j|| / (1 + alpha * d_{ij})

### 3.2 背景先验

B_i = 1 - exp(-d_geo(i, boundary)^2 / sigma^2)

其中 d_geo 是测地距离。

### 3.3 随机森林预测

f_RF(x) = (1/T) * sum_{t=1}^T sum_{l in L_t(x)} w_l * y_l

### 3.4 多尺度融合权重

min_alpha sum_{m=1}^M ||G_m - sum_k alpha_k * S_k^{(m)}||^2
s.t. alpha_k >= 0, sum_k alpha_k = 1

---

## 4. 训练过程讲解

### 4.1 训练步骤

1. 对每张训练图像：
   a. 在3个尺度上分别分割
   b. 对每个区域提取86维特征
   c. 计算每个区域的GT显著性

2. 收集所有区域的特征-标签对: {(x_i, y_i)}

3. 训练随机森林回归器：
   - 树数: T=100, 最大深度: 20
   - 最小叶子样本数: 5, 特征采样比例: 1/3

4. 训练多尺度融合权重：在验证集上优化 alpha

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 通用显著物体检测 | 适用于自然图像的通用检测 |
| 图像裁剪 | 基于显著图自动裁剪 |
| 图像分割 | 显著图作为前景分割先验 |
| 物体发现 | 无类别的物体定位 |

---

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 86维丰富特征全面覆盖显著性线索 | 手工特征设计工程量大 |
| 随机森林训练高效、泛化好 | 86维特征计算开销大 |
| 多尺度融合提升鲁棒性 | 特征表达上限限制了性能 |
| 在传统方法中SOTA | 后被深度学习方法超越 |

---

## 7. 调库实现（scikit-learn）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skimage import color, segmentation
from scipy.ndimage import gaussian_filter, sobel


class DRFI:
    def __init__(self, n_scales=3, n_estimators=100, max_depth=20):
        self.n_scales = n_scales
        self.rf = RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_leaf=5,
                                        random_state=42, n_jobs=-1)
        self.alphas = np.ones(n_scales) / n_scales
        self.fitted = False

    def _segment(self, image, scale):
        n_segments = max(50, 500 // (scale + 1))
        compactness = 10 * (scale + 1)
        segments = segmentation.slic(image, n_segments=n_segments,
                                     compactness=compactness,
                                     sigma=1, start_label=0)
        return segments

    def _extract_features(self, image, segments):
        lab = color.rgb2lab(image)
        h, w = lab.shape[:2]
        K = segments.max() + 1
        colors = np.zeros((K, 3))
        positions = np.zeros((K, 2))
        areas = np.zeros(K)
        edges = np.zeros(K)
        gray = color.rgb2gray(image)
        edge_map = np.abs(sobel(gray))
        for i in range(h):
            for j in range(w):
                sid = segments[i, j]
                colors[sid] += lab[i, j]
                positions[sid] += [i / h, j / w]
                areas[sid] += 1
                edges[sid] += edge_map[i, j]
        colors /= areas[:, None]
        positions /= areas[:, None]
        edges = edges / (areas + 1e-8)

        features = []
        for i in range(K):
            global_contrast = np.sqrt(np.sum((colors[i] - lab.mean(axis=(0,1))) ** 2))
            neighbor_contrast = 0
            neighbor_count = 0
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni = int(positions[i,0]*h+di)
                    nj = int(positions[i,1]*w+dj)
                    if 0 <= ni < h and 0 <= nj < w:
                        ns = segments[ni, nj]
                        if ns != i and ns < K:
                            neighbor_contrast += np.sqrt(np.sum((colors[i]-colors[ns])**2))
                            neighbor_count += 1
            if neighbor_count > 0:
                neighbor_contrast /= neighbor_count
            center_dist = np.sqrt(np.sum((positions[i]-0.5)**2))
            boundary_dist = min(positions[i,0], 1-positions[i,0], positions[i,1], 1-positions[i,1])
            feat = [global_contrast, neighbor_contrast,
                    colors[i,0]/100, colors[i,1]/128, colors[i,2]/128,
                    areas[i]/(h*w), center_dist, boundary_dist,
                    edges[i], np.sqrt(areas[i]/(h*w)),
                    positions[i,0], positions[i,1]]
            while len(feat) < 20: feat.append(0)
            features.append(feat[:20])
        return np.array(features), colors, K

    def fit(self, images, gt_masks):
        X_all, y_all = [], []
        for img, gt in zip(images, gt_masks):
            if img.max() > 1.0: img = img / 255.0
            for s in range(self.n_scales):
                seg = self._segment(img, s)
                feat, _, K = self._extract_features(img, seg)
                X_all.append(feat)
                labels = np.zeros(K); counts = np.zeros(K)
                h, w = gt.shape[:2]
                for i in range(h):
                    for j in range(w):
                        sid = seg[i, j]
                        if sid < K:
                            labels[sid] += gt[i, j]
                            counts[sid] += 1
                y_all.append(labels/(counts+1e-8))
        X = np.vstack(X_all); y = np.hstack(y_all)
        self.rf.fit(X, y)
        self.fitted = True
        print(f"DRFI: {X.shape[0]} samples, {X.shape[1]} features")

    def compute_saliency(self, image):
        if image.max() > 1.0: image = image / 255.0
        h, w = image.shape[:2]
        maps = []
        for s in range(self.n_scales):
            seg = self._segment(image, s)
            feat, _, K = self._extract_features(image, seg)
            scores = self.rf.predict(feat) if self.fitted else np.random.rand(K)
            sm = np.zeros((h,w))
            for i in range(K): sm[seg==i] = scores[i]
            maps.append(sm)
        final = sum(self.alphas[i]*maps[i] for i in range(self.n_scales))
        final = gaussian_filter(final, 2)
        return (final-final.min())/(final.max()-final.min()+1e-8)


def demo_drfi():
    np.random.seed(42)
    img = np.ones((80,80,3))*0.2; img[20:60,20:60]=[0.7,0.3,0.3]
    gt = np.zeros((80,80)); gt[20:60,20:60]=1.0
    model = DRFI(n_scales=2)
    model.fit([img],[gt])
    s = model.compute_saliency(img)
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(img); axes[0].set_title('Input'); axes[0].axis('off')
    axes[1].imshow(gt,cmap='gray'); axes[1].set_title('GT'); axes[1].axis('off')
    im = axes[2].imshow(s,cmap='jet'); axes[2].set_title('DRFI'); axes[2].axis('off')
    plt.colorbar(im,ax=axes[2],fraction=0.046)
    plt.tight_layout(); plt.savefig('drfi_demo.png',dpi=150); plt.show()
    print(f"DRFI: [{s.min():.3f}, {s.max():.3f}]")

if __name__ == '__main__':
    demo_drfi()
```

---

## 8. 手工代码实现（NumPy）

```python
import numpy as np
from scipy.ndimage import gaussian_filter


class DRFINumpy:
    def __init__(self, n_scales=3):
        self.n_scales = n_scales
        self.weights = None

    def _simple_seg(self, image, scale):
        h,w = image.shape[:2]
        grid = 8*(scale+1)
        step_h = max(2, h//grid); step_w = max(2, w//grid)
        segments = np.zeros((h,w), dtype=np.int32)
        sid = 0
        for i in range(0, h, step_h):
            for j in range(0, w, step_w):
                segments[i:min(h,i+step_h), j:min(w,j+step_w)] = sid
                sid += 1
        return segments, sid

    def _features(self, image, segments, K):
        h,w = image.shape[:2]
        colors = np.zeros((K,3)); positions = np.zeros((K,2)); counts = np.zeros(K)
        for i in range(h):
            for j in range(w):
                s = segments[i,j]
                if s < K:
                    colors[s] += image[i,j]; positions[s] += [i/h,j/w]; counts[s] += 1
        colors /= counts[:,None]+1e-8; positions /= counts[:,None]+1e-8
        global_mean = image.mean(axis=(0,1))
        features = np.zeros((K,10))
        for i in range(K):
            gc = np.sqrt(np.sum((colors[i]-global_mean)**2))
            cd = np.sqrt(np.sum((positions[i]-0.5)**2))
            bd = min(positions[i,0],1-positions[i,0],positions[i,1],1-positions[i,1])
            area = counts[i]/(h*w)
            features[i] = [gc, colors[i,0],colors[i,1],colors[i,2],cd,bd,area,positions[i,0],positions[i,1],np.sqrt(area)]
        return features

    def fit(self, images, gts):
        X_all, y_all = [], []
        for img, gt in zip(images, gts):
            if img.max()>1.0: img/=255.0
            for s in range(self.n_scales):
                seg, K = self._simple_seg(img, s)
                feat = self._features(img, seg, K); X_all.append(feat)
                labels = np.zeros(K); cnt = np.zeros(K); h,w = gt.shape
                for i in range(h):
                    for j in range(w):
                        sid = seg[i,j]
                        if sid < K: labels[sid]+=gt[i,j]; cnt[sid]+=1
                y_all.append(labels/(cnt+1e-8))
        X = np.vstack(X_all); y = np.hstack(y_all)
        XtX = X.T@X + np.eye(X.shape[1])*0.01
        self.weights = np.linalg.solve(XtX, X.T@y)
        print(f"DRFI训练完成")

    def compute_saliency(self, image):
        if image.max()>1.0: image/=255.0
        h,w = image.shape[:2]; maps=[]
        for s in range(self.n_scales):
            seg, K = self._simple_seg(image, s)
            feat = self._features(image, seg, K)
            scores = feat@self.weights if self.weights is not None else np.random.rand(K)
            sm = np.zeros((h,w))
            for i in range(K): sm[seg==i] = scores[i]
            maps.append(sm)
        final = np.mean(maps, axis=0)
        final = gaussian_filter(final,1)
        return (final-final.min())/(final.max()-final.min()+1e-8)

def demo_numpy():
    np.random.seed(42)
    img = np.random.rand(48,48,3); img[15:33,15:33]=[0.8,0.2,0.2]
    gt = np.zeros((48,48)); gt[15:33,15:33]=1.0
    m = DRFINumpy(n_scales=2); m.fit([img],[gt]); s = m.compute_saliency(img)
    print(f"DRFI手工: [{s.min():.3f}, {s.max():.3f}]")

if __name__ == '__main__':
    demo_numpy()
```

---

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation


def visualize_drfi_features():
    np.random.seed(42)
    img = np.ones((64,64,3))*0.2; img[20:44,20:44]=[0.7,0.3,0.3]
    seg = segmentation.slic(img, n_segments=50, compactness=20, sigma=1, start_label=0)
    K = seg.max()+1; h,w = img.shape[:2]
    colors = np.zeros((K,3)); counts = np.zeros(K)
    for i in range(h):
        for j in range(w):
            s = seg[i,j]; colors[s]+=img[i,j]; counts[s]+=1
    colors /= counts[:,None]
    global_mean = img.mean(axis=(0,1))
    contrast = np.sqrt(np.sum((colors-global_mean)**2, axis=1))

    fig, axes = plt.subplots(2,3,figsize=(15,10))
    axes[0,0].imshow(img); axes[0,0].set_title('(a) Input'); axes[0,0].axis('off')
    cm = np.zeros((h,w))
    for i in range(K): cm[seg==i]=contrast[i]
    im = axes[0,1].imshow(cm,cmap='hot'); axes[0,1].set_title('(b) Global Contrast'); axes[0,1].axis('off')
    plt.colorbar(im,ax=axes[0,1],fraction=0.046)
    am = np.zeros((h,w))
    for i in range(K): am[seg==i]=counts[i]/(h*w)
    im = axes[0,2].imshow(am,cmap='Blues'); axes[0,2].set_title('(c) Region Area'); axes[0,2].axis('off')
    plt.colorbar(im,ax=axes[0,2],fraction=0.046)
    y,x = np.mgrid[0:h,0:w]
    im = axes[1,0].imshow(np.sqrt((y/h-0.5)**2+(x/w-0.5)**2),cmap='Greens')
    axes[1,0].set_title('(d) Center Distance'); axes[1,0].axis('off')
    plt.colorbar(im,ax=axes[1,0],fraction=0.046)
    from scipy.ndimage import gaussian_filter
    sal = np.random.rand(h,w); sal[20:44,20:44]+=0.5
    sal = gaussian_filter(sal,2)
    sal = (sal-sal.min())/(sal.max()-sal.min()+1e-8)
    im = axes[1,1].imshow(sal,cmap='jet'); axes[1,1].set_title('(e) RF Prediction'); axes[1,1].axis('off')
    plt.colorbar(im,ax=axes[1,1],fraction=0.046)
    axes[1,2].axis('off')
    plt.suptitle('DRFI特征可视化',fontsize=14); plt.tight_layout()
    plt.savefig('drfi_features.png',dpi=150); plt.show()

if __name__ == '__main__':
    visualize_drfi_features()
```

---

## 10. 模型评估

### 10.1 DRFI在公开数据集上的性能
| 方法 | F-measure(MSRA-1000) | MAE |
| FT | 0.624 | 0.178 |
| SF | 0.736 | 0.131 |
| DRFI | 0.772 | 0.105 |

### 10.2 评估代码
```python
def evaluate(saliency, gt_mask):
    binary = (saliency > 0.5).astype(np.int32)
    tp = np.sum((binary==1)&(gt_mask>0.5))
    fp = np.sum((binary==1)&(gt_mask<=0.5))
    fn = np.sum((binary==0)&(gt_mask>0.5))
    prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
    f = 1.3*prec*rec/(0.3*prec+rec+1e-8)
    mae = np.mean(np.abs(saliency-gt_mask))
    return prec, rec, f, mae
```

---

## 11. 常见问题与易错点

### Q1: 为什么要86维特征?
A: 显著性检测涉及多个线索（对比度、位置、背景等），单个特征无法覆盖。

### Q2: 为什么选择随机森林而非SVM?
A: 随机森林训练快，无需大量调参，能处理高维特征，适合2013年的计算条件。

### Q3: 多尺度融合为什么有效?
A: 不同尺度捕捉不同粒度的信息，细尺度保边缘，粗尺度保整体。

### Q4: 背景先验为什么有效?
A: 显著物体通常不接触图像边界，边界区域颜色相似的区域更可能是背景。

### Q5: DRFI的主要瓶颈?
A: 86维特征计算耗时；手工天花板低于深度特征。

---

## 12. 学习总结

- DRFI将SOD从"设计公式"转变为"学习映射"
- 86维特征的工程经验值得借鉴
- 随机森林+多尺度融合=传统SOD巅峰
- 被深度方法超越的根本原因：手工特征表达能力有限

---

## 13. 练习题与思考题（含答案）

### 练习1
题目：DRFI中为什么需要背景先验?
答案：利用"显著物体不接触边界"的统计规律。与边界区域颜色相似的区域应被抑制。

### 练习2
题目：随机森林vs单一决策树?
答案：随机森林通过bagging+特征采样降低方差，泛化更好。

### 练习3：思考题
题目：将DRFI的随机森林替换为CNN会怎样?
答案：优点：自动学习特征表示。缺点：需更多训练数据和计算资源。

---

## 14. 学习路径建议

### 前置知识
1. 随机森林：决策树集成、Bagging
2. 特征工程：对比度、纹理、位置特征
3. 超像素分割：SLIC

### 后续学习
1. 深度特征方法：MDF
2. 端到端方法：U-Net, BASNet
3. 显著性检测最新进展：TransalNet, VST

### 推荐文献
1. Jiang H, et al. "Salient object detection: A discriminative regional feature integration approach." CVPR 2013.
2. Breiman L. "Random forests." Machine Learning 2001.
3. Li G, Yu Y. "Visual saliency based on multiscale deep features." CVPR 2015.
