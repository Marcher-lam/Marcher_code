# Swin Transformer 学习文档

> 层级式视觉Transformer，引入移动窗口注意力。

---

## 1. 算法基础认知

### 1.1 发展背景

Swin Transformer 由 Microsoft Research Asia 于 2021 年在论文《Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows》中提出，通过移动窗口注意力机制解决了 ViT 的二次复杂度问题，在目标检测和分割任务上取得了 SOTA。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 层级式 Vision Transformer |
| 复杂度 | O(N) 线性 |
| 窗口大小 | 7×7 |
| 移动窗口 | 逐步移动 |

### 1.3 模型系列

| 模型 | 参数量 | GFLOPs |
|------|--------|--------|
| Swin-T | 28M | 4.5 |
| Swin-S | 50M | 8.7 |
| Swin-B | 88M | 15.4 |
| Swin-L | 197M | 37.0 |

---

## 2. 核心原理

### 2.1 移动窗口注意力

```
标准注意力: 每个 token 关注所有其他 token (O(N²))
窗口注意力: 每个 token 只关注同窗口内 token (O(N×M))

Swin: 窗口内注意力 + 移动窗口
  - 第1层: 固定窗口
  - 第2层: 移动半个窗口
```

### 2.2 层级结构

```
输入 → Patch Embedding → Stage1 → Stage2 → Stage3 → Stage4
              ↓              ↓        ↓        ↓        ↓
           4×4           8×8    16×16   32×32   64×64
```

---

## 3. 数学公式与推导

### 3.1 窗口注意力

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V$$

在窗口内计算：
$$Attention_W(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V$$

### 3.2 移动窗口

```
W-MSA:  窗口多头自注意力
SW-MSA: 移动窗口MSA
```

### 3.3 相对位置编码

$$B_{ij} = \text{RelativeBias}(i, j)$$

---

## 4. 训练过程讲解

### 4.1 预训练配置

| 参数 | 值 |
|------|-----|
| ImageNet | 1M images |
| Batch | 4096 |
| Epochs | 300 |
| LR | 5e-4 |

### 4.2 下游任务

- ImageNet 分类
- COCO 检测
- ADE20K 分割

---

## 5. 应用场景

### 5.1 典型应用

- **目标检测**：COCO
- **语义分割**：ADE20K
- **图像分类**：ImageNet

### 5.2 代码示例

```python
import timm

# 加载 Swin
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

# 推理
output = model(x)
```

---

## 6. 调库实现

### 6.1 timm 实现

```python
import torch
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

class SwinTransformer:
    """Swin Transformer 层级视觉模型"""
    
    def __init__(self, model_name='swin_base_patch4_window7_224'):
        self.model_name = model_name
        
        if TIMM_AVAILABLE:
            self.model = timm.create_model(model_name, pretrained=True)
            
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        return self.model.forward_features(x)


def demo():
    print("=== Swin Transformer 演示 ===\n")
    
    if TIMM_AVAILABLE:
        swin = SwinTransformer('swin_base_patch4_window7_224')
        params = sum(p.numel() for p in swin.model.parameters())
        print(f"模型: {swin.model_name}")
        print(f"参数量: {params:,}")
    else:
        print("timm 未安装，安装: pip install timm")


if __name__ == "__main__":
    demo()
```

---

## 7. 手工代码实现

### 7.1 核心组件

```python
import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    """窗口注意力"""
    
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## 8. 优缺点分析

### 8.1 优点

1. **线性复杂度**：O(N) 而非 O(N²)
2. **层级结构**：多尺度特征
3. **移动窗口**：全局建模

### 8.2 缺点

1. **实现复杂**：窗口计算
2. **调参**：敏感

---

## 9. 可视化与结果理解

### 9.1 移动窗口示意

```python
def visualize():
    print("""
    Swin 移动窗口:
    
    层1:      层2:
    ┌─┬┐     ┌┬─┐
    │█│█│     │█│█│
    ├─┼┤     ├┼─┤
    │█│█│     │█│█│
    └─┴┘     └┴─┘
    (固定)    (移动)
    
    通过移动实现全局建模
    """)
```

---

## 10. 模型评估

### 10.1 ImageNet 分类

| 模型 | Top-1 |
|------|-------|
| Swin-T | 81.3% |
| Swin-S | 83.0% |
| Swin-B | 83.5% |
| Swin-L | 84.7% |

---

## 11. 学习总结

**核心要点**：

1. **移动窗口**：降低复杂度
2. **层级结构**：多尺度
3. **线性 O(N)**：高效

**Swin 核心优势**：
- 性能 SOTA
- 效率高

---

## 12. 学习路径建议

### 入门阶段

1. ViT 基础
2. 注意力机制

### 进阶阶段

1. Swin 实现
2. 下游任务

### 高级阶段

1. 改进 Swin
2. 变体研究

**推荐路线**：

```
ViT → Swin → SwinV2 → CSwin
```

**Swin Transformer 是视觉 Transformer 的里程碑，熟练掌握它对学习视觉模型很重要。**