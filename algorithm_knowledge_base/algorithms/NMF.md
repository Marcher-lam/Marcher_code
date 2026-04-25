# NMF 学习文档

## 1. 算法基础认知

### 1.1 定义

NMF（Non-negative Matrix Factorization，非负矩阵分解）是 Lee 和 Seung 于 1999 年提出的一种矩阵分解方法。其核心约束是：**所有分解后的矩阵元素必须非负**，这使得分解结果具有明确的物理意义（可解释为部分的总和）。

给定非负矩阵 $V \in \mathbb{R}^{m \times n}_+$，NMF 将其分解为：

$$
V \approx W H
$$

其中：
- $W \in \mathbb{R}^{m \times r}_+$：基矩阵（basis matrix）
- $H \in \mathbb{R}^{r \times n}_+$：系数矩阵（coefficient matrix）
- $r$：隐因子数量，通常 $r < \min(m, n)$

### 1.2 直观类比

将 NMF 想象为**拼图游戏**：每个部分（列）只能贡献正面的颜色，整体是各部分的累加。这与加法乘法对应，不可分割或消除。

### 1.3 历史背景

- **1999**：Lee 和 Seung 在 Nature 发表 NMF
- **2000s**：扩展到稀疏约束、体积约束
- 现在：广泛用于语音识别、图像分解、推荐系统

---

## 2. 核心原理

### 2.1 非负约束

NMF 的核心约束：

$$
w_{ij} \geq 0, \quad h_{jk} \geq 0
$$

这确保了：
- 部分表示（parts-based representation）
- 可加性而非减法
- 更稀疏的表示

### 2.2 与标准 MF 对比

| 方面 | SVD | NMF |
|------|-----|-----|
| 符号 | 可正可负 | 仅非负 |
| 表示 | 基/系数相减 | 部分累加 |
| 稀疏 | 否 | 是 |
| 可解释 | 困难 | 容易 |

### 2.3 优化目标

$$
\min_{W, H \geq 0} D(V || WH) = \sum_{i,j} d(v_{ij} || (WH)_{ij})
$$

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $V$ | 输入矩阵 | $(m, n)$ |
| $W$ | 基矩阵 | $(m, r)$ |
| $H$ | 系数矩阵 | $(r, n)$ |
| $r$ | 隐因子数 | 标量 |

### 3.2 损失函数

**欧几里得距离**：
$$
L_{Euc}(V, WH) = ||V - WH||_F^2 = \sum_{i,j} (v_{ij} - \sum_k w_{ik}h_{kj})^2
$$

**KL 散度**：
$$
L_{KL}(V, WH) = \sum_{i,j} \left( v_{ij} \log\frac{v_{ij}}{(WH)_{ij}} - v_{ij} + (WH)_{ij} \right)
$$

### 3.3 乘法更新规则

对于欧几里得距离：
$$
h_{kj} \leftarrow h_{kj} \frac{(W^T V)_{kj}}{(W^T W H)_{kj}}
$$
$$
w_{ik} \leftarrow w_{ik} \frac{(V H^T)_{ik}}{(W H H^T)_{ik}}
$$

---

## 4. 训练过程讲解

### 4.1 PyTorch 实现

```python
import torch
import torch.nn as nn

class NMF(nn.Module):
    """非负矩阵分解"""
    
    def __init__(self, m, n, r):
        super().__init__()
        self.m = m
        self.n = n
        self.r = r
        
        # 初始化为正数
        self.W = nn.Parameter(torch.rand(m, r))
        self.H = nn.Parameter(torch.rand(r, n))
    
    def forward(self):
        return torch.mm(self.W, self.H)
    
    def reconstruct(self):
        return self.forward()
```

### 4.2 优化器实现

```python
import torch.optim as optim

def train_nmf():
    """训练 NMF"""
    
    # 数据
    V = torch.rand(100, 50)  # 100 x 50 矩阵
    r = 10  # 隐因子数
    
    # 模型
    model = NMF(100, 50, r)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 损失
    criterion = nn.MSELoss()
    
    # 训练
    for epoch in range(100):
        optimizer.zero_grad()
        
        V_pred = model()
        
        # 掩码非负（强制非负）
        with torch.no_grad():
            model.W.clamp_(min=0)
            model.H.clamp_(min=0)
        
        loss = criterion(V_pred, V)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model

train_nmf()
```

### 4.3 完整 NMF 算法

```python
def complete_nmf(V, r, max_iters=100, tol=1e-4):
    """完整 NMF 算法（乘法更新）"""
    
    m, n = V.shape
    
    # 初始化
    W = torch.rand(m, r)
    H = torch.rand(r, n)
    
    # 更新
    for it in range(max_iters):
        # 更新 H
        WTV = torch.mm(W.t(), V)
        WWH = torch.mm(torch.mm(W.t(), W), H)
        H = H * (WTV / (WWH + 1e-8))
        H = torch.clamp(H, min=0)
        
        # 更新 W
        VHT = torch.mm(V, H.t())
        WHHT = torch.mm(W, torch.mm(H, H.t()))
        W = W * (VHT / (WHHT + 1e-8))
        W = torch.clamp(W, min=0)
        
        # 检查收敛
        if it % 10 == 0:
            loss = torch.norm(V - torch.mm(W, H)) / torch.norm(V)
            print(f"Iter {it}, Rel Error: {loss:.4f}")
    
    return W, H
```

---

## 5. ��用场景

### 5.1 图像分解

NMF 的经典应用：
- 人脸分解（Lee 论文）
- 文档-词矩阵分解
- 音乐分析

### 5.2 推荐系统

- 用户-物品矩阵分解
- 隐因子可解释

### 5.3 语音处理

- 频谱分解
- 声音分离

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 非负约束 | 部分表示 |
| 可解释 | 隐因子有意义 |
| 稀疏 | 表示简洁 |
| 可加性 | 符合直觉 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 局部最优 | 非凸优化 |
| 初始化敏感 | 不同初始不同结果 |
| 计算慢 | 迭代更新 |

---

## 7. 调库实现

### 7.1 sklearn 实现

```python
from sklearn.decomposition import NMF
import numpy as np

def use_sklearn_nmf():
    """sklearn NMF 使用"""
    
    # 数据
    np.random.seed(42)
    V = np.abs(np.random.randn(100, 50))
    
    # NMF
    model = NMF(n_components=10, init='random', random_state=42, max_iter=200)
    W = model.fit_transform(V)
    H = model.components_
    
    # 重构
    V_pred = W @ H
    
    print(f"原始形状: {V.shape}")
    print(f"W 形状: {W.shape}")
    print(f"H 形状: {H.shape}")
    print(f"重构误差: {np.linalg.norm(V - V_pred):.4f}")
    
    return W, H, model

use_sklearn_nmf()
```

### 7.2 带约束的 NMF

```python
def constrained_nmf():
    """稀疏 NMF"""
    
    from sklearn.decomposition import NMF
    
    V = np.abs(np.random.randn(100, 50))
    
    # L1 正则化（稀疏）
    model = NMF(
        n_components=10,
        alpha_W=0.1,
        alpha_H=0.1,
        l1_ratio=0.5
    )
    
    W = model.fit_transform(V)
    H = model.components_
    
    print(f"W 稀疏度: {(W < 0.1).sum() / W.size:.2f}")
    
    return W, H

constrained_nmf()
```

---

## 8. 手工代码实现

### 8.1 完整 NMF 实现

```python
import numpy as np

class ManualNMF:
    """手动实现 NMF"""
    
    def __init__(self, r, max_iters=200, tol=1e-4):
        self.r = r
        self.max_iters = max_iters
        self.tol = tol
        self.W = None
        self.H = None
    
    def fit(self, V):
        """拟合 NMF
        
        V: [m, n]
        """
        m, n = V.shape
        
        # 初始化（基于 V 的列归一化）
        self.W = np.random.rand(m, self.r)
        self.H = np.random.rand(self.r, n)
        
        # 迭代更新
        for it in range(self.max_iters):
            # 更新 H（乘法规则）
            WH = self.W @ self.H
            WH[WH == 0] = 1e-8
            
            h_new = self.H * (self.W.T @ V) / (self.W.T @ WH)
            self.H = np.clip(h_new, 0, None)
            
            # 更新 W
            WH = self.W @ self.H
            WH[WH == 0] = 1e-8
            
            w_new = self.W * (V @ self.H.T) / (WH @ self.H.T)
            self.W = np.clip(w_new, 0, None)
            
            # 检查收敛
            if it % 20 == 0:
                err = np.linalg.norm(V - self.W @ self.H) / np.linalg.norm(V)
                print(f"Iter {it}: error = {err:.4f}")
                
                if err < self.tol:
                    break
        
        return self
    
    def transform(self):
        """返回系数"""
        return self.H
    
    def components_(self):
        """返回基"""
        return self.W
    
    def reconstruct(self):
        """重构"""
        return self.W @ self.H

# 测试
V = np.abs(np.random.randn(100, 50))
nmf = ManualNMF(r=10)
nmf.fit(V)
print(f"W 形状: {nmf.W.shape}, H 形状: {nmf.H.shape}")
```

### 8.2 验证对比

```python
def verify_against_sklearn():
    """对比 sklearn"""
    
    from sklearn.decomposition import NMF
    
    np.random.seed(42)
    V = np.abs(np.random.randn(50, 30))
    
    # 手写
    manual = ManualNMF(r=5)
    manual.fit(V)
    V_manual = manual.reconstruct()
    
    # sklearn
    sklearn_model = NMF(n_components=5, init='random', random_state=42)
    W_sk = sklearn_model.fit_transform(V)
    H_sk = sklearn_model.components_
    V_sk = W_sk @ H_sk
    
    diff_manual = np.linalg.norm(V - V_manual)
    diff_sklearn = np.linalg.norm(V - V_sk)
    
    print(f"手写 NMF 误差: {diff_manual:.4f}")
    print(f"sklearn NMF 误差: {diff_sklearn:.4f}")
    
    return diff_manual, diff_sklearn

verify_against_sklearn()
```

---

## 9. 可视化与结果理解

### 9.1 基矩阵可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_components():
    """可视化基矩阵"""
    
    W = np.random.rand(100, 10)
    
    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(W[:, i].reshape(10, 10), cmap='gray')
        plt.axis('off')
        plt.title(f'Component {i+1}')
    
    plt.tight_layout()
    plt.savefig('nmf_components.png', dpi=150)
    plt.show()

visualize_components()
```

### 9.2 重构质量

```python
def plot_reconstruction():
    """绘制重构质量"""
    
    import matplotlib.pyplot as plt
    
    # 原始 vs 重构
    V = np.random.rand(100, 50)
    V_pred = V * 0.9 + 0.05
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(V.flatten(), bins=30, alpha=0.7)
    plt.xlabel('Value')
    plt.title('Original')
    
    plt.subplot(1, 2, 2)
    plt.hist(V_pred.flatten(), bins=30, alpha=0.7)
    plt.xlabel('Value')
    plt.title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig('nmf_reconstruction.png', dpi=150)
    plt.show()

plot_reconstruction()
```

---

## 10. 模型评估

### 10.1 质量指标

```python
import numpy as np

def evaluate_nmf():
    """评估 NMF"""
    
    # 重构误差
    V = np.abs(np.random.randn(100, 50))
    W = np.abs(np.random.randn(100, 10))
    H = np.abs(np.random.randn(10, 50))
    
    V_pred = W @ H
    
    # 指标
    mse = np.mean((V - V_pred) ** 2)
    rmse = np.sqrt(mse)
    sparsity = (W < 0.1).sum() / W.size
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'Sparsity': sparsity,
    }
    
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics

evaluate_nmf()
```

---

## 11. 常见问题与易错点

### 11.1 零值处理

**问题**：出现零导致除零错误？

**解答**：添加 epsilon 防零。

### 11.2 局部最优

**问题**：结果依赖初始化？

**解答**：多次运行取最优。

---

## 12. 学习总结

### 12.1 核心要点

1. **非负约束**：$W, H \geq 0$
2. **乘法更新**：简单高效
3. **部分表示**：可解释性强
4. **广泛用途**：图像、推荐等

### 12.2 变体

| 名称 | 特点 |
|------|------|
| 稀疏 NMF | L1 正则 |
| 卷积 NMF | 时序 |
| 半监督 NMF | 约束 |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：实现 NMF 的乘法更新规则。

### 13.2 思考题

**思考题**：NMF 和 SVD 的本质区别？

---

## 14. 学习路径建议

### 14.1 第一阶段

1. 理解非负约束
2. 理解优化目标

### 14.2 第二阶段

1. 实现乘法更新
2. 实现完整算法

### 14.3 第三阶段

1. 实际应用
2. 对比其他方法

### 14.4 推荐资源

- **论文**：《Learning the Parts of Objects by Non-negative Matrix Factorization》
- **代码**：sklearn

---

*NMF 是一种具有物理意义的矩阵分解方法，它的非负约束使得分解结果具有天然的可解释性。*