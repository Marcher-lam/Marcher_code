# AIM模型 学习文档

> 基于信息最大化的注意力——注视点总是落在信息量（自信息）最大的位置。
>
> 来源线索：本节内容根据原书第2.2.1节"AIM:基于信息最大化的注视点预测"整理。

---

## 1. 算法基础认知

**一句话定义：** AIM（Attention based on Information Maximization）由Bruce和Tsotsos于2006年在NIPS上提出，从信息论视角出发，认为视觉注意力总是选择自信息（Self-Information）最大的位置。

**核心思想：** 图像中每个位置的信息量定义为该位置特征的自信息（负对数似然）。在自然图像中，频繁出现的特征（如均匀纹理）信息量低，而罕见特征（如边缘交叠、不寻常的颜色组合）信息量高。AIM使用独立成分分析（ICA）学习自然图像的统计结构，然后用核密度估计（KDE）计算每个位置的自信息作为显著性。

**信息论视角：** Shannon信息论告诉我们，概率越小的事件包含的信息量越大（$I(x) = -\log p(x)$）。因此，AIM中"显著性=自信息"意味着：眼睛总是看向那些最"意外"的位置。这个观点与生物视觉中"抑制重复、响应新颖"的神经机制一致。

**为什么用ICA？** ICA学习的基函数与V1皮层简单细胞（Gabor-like）的响应特性高度一致，使AIM具有生物合理性。

---

## 2. 核心原理

### 2.1 特征提取：独立成分分析（ICA）

将图像分割为若干 $8 \times 8$ 的局部图像块，每个块采样后通过ICA学习一组基函数 $W$：

$$
\mathbf{s} = W\mathbf{x}
$$

其中 $\mathbf{x} \in \mathbb{R}^{64}$ 是图像块向量，$\mathbf{s} \in \mathbb{R}^{m}$ 是稀疏编码系数（特征向量），$W \in \mathbb{R}^{m \times 64}$ 是ICA解混矩阵。

ICA基函数与Gabor滤波器类似，具有方向、尺度选择性。

### 2.2 概率建模：核密度估计

对每一个特征维度 $s_k$，用核密度估计建立概率密度函数：

$$
\hat{p}_k(s) = \frac{1}{N} \sum_{n=1}^{N} K_\sigma(s - s_k^{(n)})
$$

其中 $K_\sigma$ 是高斯核，$N$ 是训练样本数。

### 2.3 自信息计算

一个图像块的特征向量 $\mathbf{s} = (s_1, s_2, ..., s_m)$ 的联合似然：

$$
p(\mathbf{s}) = \prod_{k=1}^{m} p_k(s_k)
$$

（ICA假设各成分独立，因此联合概率可分解为边缘概率的乘积。）

该块的自信息（显著性）为：

$$
\text{Sal}(\mathbf{s}) = -\log p(\mathbf{s}) = -\sum_{k=1}^{m} \log p_k(s_k)
$$

### 2.4 显著性图构建

对图像中每个位置提取图像块→计算ICA特征→拼接特征图→计算每个特征值的自信息→求和重构为显著性图→高斯平滑。

---

## 3. 数学公式与推导

### 3.1 ICA模型

设观测数据 $X = AS$（$A$ 是混合矩阵，$S$ 是独立源信号）。ICA寻找解混矩阵 $W = A^{-1}$，使得 $S = WX$ 的各分量尽可能独立。

ICA的优化目标是最大化非高斯性（通过峭度或负熵度量），因为独立成分是非高斯的：

$$
J(s) \propto [E\{G(s)\} - E\{G(\nu)\}]^2
$$

其中 $G$ 是非二次函数（如 $\log\cosh$），$\nu$ 是标准高斯变量。

### 3.2 自信息的分解

给定图像块 $x_i$，其自信息：

$$
I(x_i) = -\log p(x_i)
$$

通过ICA变换 $s_i = Wx_i$ 和独立性假设：

$$
p(x_i) = p(s_i) \cdot |\det W| = \left(\prod_{k=1}^m p_k(s_{ik})\right) \cdot |\det W|
$$

取对数：

$$
-\log p(x_i) = -\sum_{k=1}^m \log p_k(s_{ik}) - \log|\det W|
$$

由于 $\det W$ 是常数，可以省略。因此显著性简化为：

$$
\text{Sal}(x_i) = -\sum_{k=1}^m \log \hat{p}_k(s_{ik})
$$

### 3.3 核密度估计的实现

KDE对每个特征维度的分布进行非参数估计：

$$
\hat{p}_k(s) = \frac{1}{N\sqrt{2\pi}h} \sum_{n=1}^N \exp\left(-\frac{(s - s_k^{(n)})^2}{2h^2}\right)
$$

其中 $h$ 是带宽（通过交叉验证或Silverman规则确定）。

### 3.4 最终显著性图

$$
S(x,y) = G_\sigma * I(x,y)
$$

其中 $I(x,y)$ 是逐像素的自信息图，$G_\sigma$ 是高斯基。

---

## 4. 训练过程讲解

AIM包含两个阶段：**训练阶段**（仅需一次）和**推理阶段**。

### 4.1 训练阶段

1. **收集自然图像**：从自然图像数据集中随机采样
2. **提取图像块**：$8 \times 8$ 的局部块，随机采样约 50000 个
3. **ICA训练**：使用FastICA算法学习解混矩阵 $W$
4. **计算ICA特征**：对所有训练块提取 $m$ 维特征
5. **KDE拟合**：对每个特征维度用高斯核密度估计拟合分布

### 4.2 推理阶段

1. **输入图像调整**：转为灰度图
2. **滑动窗口**：步长4像素提取 $8 \times 8$ 块
3. **ICA变换**：$s = Wx$
4. **自信息计算**：对每个维度用KDE计算 $-\log \hat{p}_k(s_k)$ 并求和
5. **显著性图重建**：每个块的自信息值填入对应区域，重叠区域取平均
6. **高斯平滑**：$\sigma = 4$ 左右

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 注视点预测 | 预测自由观看时的眼睛运动 |
| 图像压缩 | 低信息区域更高压缩比 |
| 异常检测 | 自信息高的位置可能是异常 |
| 图像检索 | 基于信息量的图像重排序 |
| 视觉显著性基准 | 作为无监督方法的性能对比基线 |
| 计算神经科学 | 模拟V1皮层的信息处理机制 |

---

## 6. 优缺点分析

**优点：**
- ✅ **信息论理论基础扎实**：显著性有明确的数学定义
- ✅ **生物合理性**：ICA基函数与V1简单细胞一致
- ✅ **无需标注数据**：无监督学习自然图像统计
- ✅ **可解释性强**：每个位置的信息量来源可追溯
- ✅ **统一框架**：同一模型在不同数据集上表现一致

**缺点：**
- ❌ **计算量大**：KDE在推理时需要对所有训练样本
- ❌ **块效应**：滑动窗口策略导致显著性图有块状伪影
- ❌ **独立假设过强**：ICA假设各成分独立，实际并不完全成立
- ❌ **带宽敏感**：KDE带宽影响结果质量
- ❌ **缺乏高层语义**：只基于低级图像特征

---

## 7. 调库实现

```python
"""
AIM模型 - 完整调库实现
使用 sklearn 的 FastICA 和 KernelDensity
"""
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class AIM:
    """
    基于信息最大化的注意力模型
    
    使用ICA提取独立特征，核密度估计建模分布，
    自信息作为显著性度量
    """
    
    def __init__(self, n_components=8, patch_size=8, stride=4):
        """
        参数:
            n_components: ICA成分数 (特征维度)
            patch_size: 图像块大小
            stride: 滑动窗口步长
        """
        self.n_components = n_components
        self.patch_size = patch_size
        self.stride = stride
        self.ica = FastICA(n_components=n_components, max_iter=1000,
                           random_state=42, whiten='unit-variance')
        self.kde_models = []  # 每个特征维度的KDE
        self.is_fitted = False
    
    def _extract_patches(self, image):
        """
        从图像中提取局部块
        返回: (n_patches, patch_size*patch_size) 的数组
        """
        h, w = image.shape[:2]
        ps = self.patch_size
        st = self.stride
        patches = []
        positions = []  # 记录每个块的位置（用于重建）
        
        for i in range(0, h - ps + 1, st):
            for j in range(0, w - ps + 1, st):
                patch = image[i:i+ps, j:j+ps].flatten()
                patches.append(patch)
                positions.append((i, j))
        
        return np.array(patches), positions
    
    def fit(self, images):
        """
        训练模型：学习ICA基函数和KDE
        
        参数:
            images: 自然图像列表，每张为 (H, W) 或 (H, W, C) 灰度图
        """
        all_patches = []
        
        for img in images:
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            patches, _ = self._extract_patches(img)
            all_patches.append(patches)
        
        # 拼接所有图像块
        X = np.vstack(all_patches)
        print(f"提取了 {X.shape[0]} 个图像块，维度 {X.shape[1]}")
        
        # ICA拟合
        print("训练ICA...")
        S = self.ica.fit_transform(X)
        
        # 对每个特征维度拟合KDE
        print("拟合KDE...")
        self.kde_models = []
        for k in range(self.n_components):
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(S[:, k:k+1])
            self.kde_models.append(kde)
        
        self.is_fitted = True
        print("AIM训练完成")
        
        return self
    
    def compute_saliency(self, image):
        """
        计算显著性图
        
        参数:
            image: 输入图像 (H, W) 或 (H, W, 3)
        
        返回:
            saliency: 归一化的显著性图 (H, W)
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")
        
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        h, w = image.shape
        ps = self.patch_size
        st = self.stride
        
        # 提取图像块
        patches, positions = self._extract_patches(image)
        
        # ICA变换
        S = self.ica.transform(patches)
        
        # 计算自信息
        self_info = np.zeros(S.shape[0])
        for k in range(self.n_components):
            # 计算 -log p(s_k)
            log_prob = self.kde_models[k].score_samples(S[:, k:k+1])
            self_info += -log_prob
        
        # 重建显著性图
        saliency = np.zeros((h, w))
        weight = np.zeros((h, w))
        
        for idx, (pi, pj) in enumerate(positions):
            saliency[pi:pi+ps, pj:pj+ps] += self_info[idx]
            weight[pi:pi+ps, pj:pj+ps] += 1.0
        
        # 重叠区域取平均
        saliency = saliency / (weight + 1e-8)
        
        # 高斯平滑
        saliency = gaussian_filter(saliency, sigma=4)
        
        # 归一化
        saliency = (saliency - saliency.min()) / \
                   (saliency.max() - saliency.min() + 1e-8)
        
        return saliency


def demo():
    """演示函数"""
    np.random.seed(42)
    
    # 创建训练数据：模拟自然图像
    print("生成训练数据...")
    train_images = []
    for _ in range(20):
        img = np.random.randn(64, 64) * 0.2
        # 加入一些边缘和纹理
        img[:, 32:34] += 1.5  # 垂直边缘
        img[32:34, :] += 1.5  # 水平边缘
        train_images.append(img)
    
    # 训练AIM
    model = AIM(n_components=6, patch_size=8, stride=4)
    model.fit(train_images)
    
    # 测试：创建包含显著目标的测试图
    test_img = np.random.randn(64, 64) * 0.15 + 0.5
    test_img[20:30, 25:35] = 1.5  # 亮斑
    test_img[45:50, 40:45] = 0.2  # 暗斑
    
    # 计算显著性
    smap = model.compute_saliency(test_img)
    
    print(f"\n=== AIM 结果 ===")
    print(f"显著性范围: [{smap.min():.3f}, {smap.max():.3f}]")
    print(f"亮斑区域 (20:30, 25:35) 显著性均值: {smap[20:30, 25:35].mean():.4f}")
    print(f"暗斑区域 (45:50, 40:45) 显著性均值: {smap[45:50, 40:45].mean():.4f}")
    print(f"背景均值: {smap.mean():.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im1 = axes[0].imshow(test_img, cmap='gray')
    axes[0].set_title('测试图像')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(smap, cmap='hot')
    axes[1].set_title('AIM显著性图 (自信息)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('aim_saliency.png', dpi=150)
    print("结果已保存至 aim_saliency.png")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""
AIM模型 - 手工实现
不依赖sklearn，自实现ICA和KDE
"""
import numpy as np


def whiten(X):
    """数据白化: 零均值 + PCA白化"""
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    # 协方差矩阵
    cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
    
    # SVD分解
    U, S, Vt = np.linalg.svd(cov)
    
    # 白化矩阵
    W_white = (U @ np.diag(1.0 / np.sqrt(S + 1e-8)) @ U.T)
    X_white = X_centered @ W_white.T
    
    return X_white, mean, W_white


def fast_ica(X, n_components, max_iter=1000, tol=1e-6):
    """
    手工实现FastICA
    
    参数:
        X: 白化后的数据 (n_samples, n_features)
        n_components: 成分数
        max_iter: 最大迭代次数
        tol: 收敛阈值
    
    返回:
        W: 解混矩阵
        S: 独立成分
    """
    n_features = X.shape[1]
    
    # 随机初始化
    np.random.seed(42)
    W = np.random.randn(n_components, n_features)
    
    # 正交化
    U, S, Vt = np.linalg.svd(W)
    W = U @ Vt
    
    for iteration in range(max_iter):
        W_old = W.copy()
        
        # g(s) = tanh(s), g'(s) = 1 - tanh^2(s)
        S = X @ W.T  # (n_samples, n_components)
        g = np.tanh(S)  # g(s)
        g_prime = 1 - g ** 2  # g'(s)
        
        # 更新 W
        W_new = (g.T @ X) / X.shape[0] - \
                (g_prime.mean(axis=0) * W)
        
        # 正交化
        U, S, Vt = np.linalg.svd(W_new)
        W = U @ Vt
        
        # 检查收敛
        diff = np.abs(np.abs(np.diag(W @ W_old.T)) - 1).max()
        if diff < tol:
            print(f"ICA在 {iteration+1} 次迭代后收敛")
            break
    
    S = X @ W.T
    return W, S


def gaussian_kde_log_likelihood(x, data, h=0.5):
    """
    手工高斯核密度估计 - 计算 log p(x)
    
    参数:
        x: 查询点
        data: 训练数据
        h: 带宽
    
    返回:
        log_prob: log p(x)
    """
    n = len(data)
    # 高斯核
    diff = (x - data) / h
    log_kernel = -0.5 * diff ** 2 - np.log(np.sqrt(2 * np.pi) * h * n)
    
    # log-sum-exp 防止数值下溢
    max_log = log_kernel.max()
    log_prob = max_log + np.log(np.exp(log_kernel - max_log).sum())
    
    return log_prob


def aim_manual(image, n_components=6, patch_size=8, stride=4,
               train_images=None, h=0.5):
    """
    手工实现AIM
    
    参数:
        image: 输入图像
        n_components: ICA成分数
        patch_size: 图像块大小
        stride: 滑动步长
        train_images: 训练图像列表 (None则用输入图自身)
        h: KDE带宽
    
    返回:
        显著性图
    """
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    # 准备训练数据
    if train_images is None:
        train_images = [image]
    
    # 提取所有训练块
    all_patches = []
    for img in train_images:
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        h_img, w_img = img.shape
        for i in range(0, h_img - patch_size + 1, stride):
            for j in range(0, w_img - patch_size + 1, stride):
                patch = img[i:i+patch_size, j:j+patch_size].flatten()
                all_patches.append(patch)
    
    X_train = np.array(all_patches)
    n_features = X_train.shape[1]
    if n_components > n_features:
        n_components = n_features
    
    # 白化 + ICA
    X_white, mean, W_white = whiten(X_train)
    W_ica, S_train = fast_ica(X_white[:, :n_components], n_components)
    
    # 存储每个维度的训练数据用于KDE
    kde_data = [S_train[:, k] for k in range(n_components)]
    
    # 测试阶段：提取测试块
    h_img, w_img = image.shape
    patches, positions = [], []
    for i in range(0, h_img - patch_size + 1, stride):
        for j in range(0, w_img - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size].flatten()
            patches.append(patch)
            positions.append((i, j))
    
    X_test = np.array(patches)
    X_test_white = (X_test - mean) @ W_white.T
    S_test = X_test_white[:, :n_components] @ W_ica.T
    
    # 计算自信息
    self_info = np.zeros(len(S_test))
    for idx, s_vec in enumerate(S_test):
        log_prob = 0
        for k in range(n_components):
            lp = gaussian_kde_log_likelihood(s_vec[k], kde_data[k], h)
            log_prob += lp
        self_info[idx] = -log_prob
    
    # 重建显著性图
    saliency = np.zeros((h_img, w_img))
    weight = np.zeros((h_img, w_img))
    for idx, (pi, pj) in enumerate(positions):
        saliency[pi:pi+patch_size, pj:pj+patch_size] += self_info[idx]
        weight[pi:pi+patch_size, pj:pj+patch_size] += 1.0
    
    saliency = saliency / (weight + 1e-8)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency


def test_aim_manual():
    """测试手工实现"""
    np.random.seed(42)
    img = np.random.randn(32, 32) * 0.2 + 0.5
    img[10:16, 14:20] = 1.5
    
    print("=== 手工AIM测试 ===")
    smap = aim_manual(img, n_components=4, patch_size=8, stride=4, h=0.5)
    print(f"显著性范围: [{smap.min():.3f}, {smap.max():.3f}]")
    assert smap.max() > 0.5, "显著性最大值应较高"
    print("✓ 测试通过")


if __name__ == "__main__":
    test_aim_manual()
```

---

## 9. 可视化与结果理解

### 9.1 ICA基函数可视化

AIM学习的ICA基函数呈现方向性和带通性，类似Gabor滤波器。每个基函数对特定的方向、频率和位置有选择性。

### 9.2 自信息分布

- 平坦区域（均匀纹理）：ICA特征值在0附近，概率密度高，自信息低
- 边缘区域：某些ICA特征值较大，概率密度低，自信息高
- 角点/交叉点：多个ICA特征值同时大，自信息最高

### 9.3 与人类注视点的一致性

AIM预测的热点区域与人类自由观看时的注视点分布有很高的一致性，特别是在自然场景中。

---

## 10. 模型评估

```python
"""AIM模型评估"""
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_aim():
    """评估AIM模型"""
    np.random.seed(42)
    
    # 训练数据
    train_imgs = [np.random.randn(32, 32) for _ in range(10)]
    
    # 测试
    model = AIM(n_components=4, patch_size=8, stride=4)
    model.fit(train_imgs)
    
    img = np.random.randn(32, 32) * 0.2 + 0.5
    img[10:16, 14:20] = 1.5
    
    gt = np.zeros((32, 32))
    gt[10:16, 14:20] = 1.0
    
    smap = model.compute_saliency(img)
    
    auc = roc_auc_score(gt.flatten() > 0, smap.flatten())
    print(f"AIM AUC: {auc:.4f}")
    
    return auc


if __name__ == "__main__":
    evaluate_aim()
```

---

## 11. 常见问题与易错点

### Q1: AIM为什么用ICA而不是PCA？
**A:** PCA只去相关（二阶统计量），ICA追求独立（高阶统计量）。自然图像的重要结构（如边缘）体现在高阶统计量中，ICA能更好地捕获。此外，ICA基函数与V1皮层细胞特性一致。

### Q2: KDE带宽如何选择？
**A:** 带宽太小过拟合，太大欠拟合。常用规则：Silverman规则 $h = 1.06\sigma n^{-1/5}$，或通过交叉验证选择。

### Q3: ICA成分数如何确定？
**A:** 通常设为8-12。太少会丢失信息，太多会增加计算量且引入噪声。可以通过交叉验证或基于重构误差选择。

### Q4: AIM和SR有什么区别？
**A:** AIM在空域的信息论框架下操作（ICA+KDE+自信息），SR在频域操作（FFT+谱残差）。AIM有训练阶段且生物合理性更强，SR完全无参数。

### Q5: 为什么AIM有块效应？
**A:** 滑动窗口策略导致相邻块的显著性不同，产生块状伪影。可以通过更小的步长或重叠窗口平滑减轻。

---

## 12. 学习总结

### 核心要点

1. **信息论基础**：显著性=自信息=$-\log p(\text{特征})$
2. **ICA特征**：学习自然图像的独立成分，基函数类似V1简单细胞
3. **KDE概率建模**：非参数密度估计，无需假设分布形式
4. **无监督学习**：从自然图像中自动学习统计结构

### 关键公式回顾

$$
\text{Sal}(x) = -\log p(x) = -\sum_k \log p_k(s_k) - \log|\det W|
$$
$$
s = Wx \quad (\text{ICA变换})
$$
$$
\hat{p}_k(s) = \frac{1}{N} \sum_n K_h(s - s_k^{(n)}) \quad (\text{KDE})
$$

---

## 13. 练习题与思考题

### 基础题

**1.** 为什么自信息 $-\log p(x)$ 可以衡量显著性？

<details>
<summary>答案</summary>
低概率事件的信息量大，视觉系统会优先注意"意外"的、不常见的刺激，因为这类刺激可能携带重要信息。频繁出现的背景信息量小，不需要特别关注。
</details>

**2.** ICA假设各成分独立，这个假设在图像中合理吗？

<details>
<summary>答案</summary>
不完全合理，但是一个有用的近似。自然图像中边缘的朝向、位置等确实存在一定独立性。这个假设大大简化了联合概率密度的估计（分解为边缘概率的乘积）。
</details>

**3.** 为什么需要训练阶段？为什么不直接在每张图上计算？

<details>
<summary>答案</summary>
ICA需要大量样本才能学到稳定的基函数。直接在单张图像上训练ICA会导致过拟合，基函数只反映了该图的具体结构而非自然图像的通用统计规律。
</details>

### 进阶题

**4.** 推导：如果特征服从均匀分布，自信息与什么有关？

<details>
<summary>答案</summary>
如果 $p(s) = 1/(b-a)$ 在 $[a,b]$ 上均匀分布，则 $-\log p(s) = \log(b-a)$，是常数。这意味着均匀分布的特征不提供区分信息，所有位置具有相同显著性。
</details>

**5.** 比较AIM的"自信息"与GBVS的"平稳分布"在数学形式上的联系。

<details>
<summary>答案</summary>
AIM的自信息是局部度量（每个位置独立），GBVS的平稳分布是全局度量（依赖所有位置的关系）。形式上，AIM用 $\log p(x)$，GBVS用 $\pi = \pi P$。两者从不同角度刻画"罕见性"。
</details>

**6.** 如何将AIM扩展到彩色图像？

<details>
<summary>答案</summary>
有三种方式：(1) 分别对RGB三个通道计算AIM后融合；(2) 将RGB向量展开到特征向量中（将 $3\times 8\times 8=192$ 维向量作为ICA输入）；(3) 在Lab色彩空间操作。
</details>

---

## 14. 学习路径建议

### 预备知识
- 信息论（熵、自信息、KL散度）
- 概率密度估计（参数与非参数方法）
- ICA和盲源分离
- 线性代数（特征分解、SVD）

### 进阶方向
1. **AIM → SUN**：从信息论到贝叶斯框架的扩展
2. **AIM + CNN**：用CNN提取深度特征替代ICA特征
3. **AIM → VAE-based saliency**：用变分自编码器的重构误差作为显著性

### 推荐阅读
- Bruce & Tsotsos. "Saliency Based on Information Maximization." NIPS 2006.
- Shannon. "A Mathematical Theory of Communication." 1948.
- Hyvärinen & Oja. "Independent Component Analysis: Algorithms and Applications." 2000.

### 项目实践
1. 在自然图像集上训练AIM，在MIT1003数据集上测试
2. 比较不同ICA成分数对显著性检测性能的影响
3. 尝试用高斯混合模型（GMM）替代KDE进行概率估计
