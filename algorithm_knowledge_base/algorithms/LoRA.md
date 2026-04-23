# LoRA (Low-Rank Adaptation) 学习文档

> 低秩适配，高效微调大模型。

---

## 1. 算法基础认知

### 1.1 发展背景

LoRA 由微软亚研院于 2021 年在论文《LoRA: Low-Rank Adaptation of Large Language Models》中提出，是一种参数高效微调方法，通过向预训练模型添加低秩矩阵实现微调，无需训练全部参数。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 参数高效微调 |
| 核心 | 低秩矩阵适配 |
| 参数 | 仅训练 0.1-5% |
| 应用 | 大模型微调 |

---

## 2. 核心原理

### 2.1 预训练权重冻结

预训练权重 $W_0$ 冻结，添加可训练的低秩矩阵：

$$W' = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$

### 2.2 秩分解

$$W_{d \times k} \approx A_{d \times r} \cdot B_{r \times k}$$

### 2.3 前向传播

$$h = W_0 x + BAx$$

---

## 3. 数学公式与推导

### 3.1 损失函数

$$\mathcal{L} = \mathcal{L}_{task}(W_0 + BA)$$

### 3.2 梯度

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial (W' )} \cdot A^T$$
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial (W')^T} \cdot B^T$$

### 3.3 秩的选择

| r 值 | 参数占比 |
|------|----------|
| 4 | 0.5% |
| 8 | 1% |
| 16 | 2% |
| 32 | 5% |

---

## 4. 训练过程讲解

### 4.1 应用位置

| 注意力层 | 应用位置 |
|----------|----------|
| Query | W_Q |
| Key | W_K |
| Value | W_V |
| Output | W_O |

### 4.2 配置

```python
# PEFT 配置
lora_config = {
    r: 8,
    lora_alpha: 16,
    lora_dropout: 0.1,
    target_modules: ["q_proj", "v_proj"]
}
```

---

## 5. 应用场景

### 5.1 典型应用

- **领域微调**：医学、法律
- **指令微调**：InstructGPT
- **多语言**：翻译

### 5.2 代码示例

```python
from peft import LoraModel, LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)
model = LoraModel(base_model, config)
```

---

## 6. 调库实现

### 6.1 transformers 实现

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# 加载模型
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 配置 LoRA
config = LoraConfig(r=8, target_modules=["c_attn"])
model = get_peft_model(model, config)

# 训练
model.train()
```

### 6.2 手工实现

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """LoRA 层"""
    
    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        
        # 冻结原始权重
        self.weight = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.weight.requires_grad = False
        
        # 可训练低秩矩阵
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, out_dim))
        
    def forward(self, x):
        # Wx + BAx
        return F.linear(x, self.weight) + (x @ self.A) @ self.B


class LoraModel(nn.Module):
    """简化 LoRA 模型"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        
        # 应用 LoRA 到注意力层
        for layer in self.base.transformer.h:
            layer.attn.q_proj = LoRALayer(768, 768, rank=8)
            layer.attn.v_proj = LoRALayer(768, 768, rank=8)


def demo():
    print("=== LoRA 演示 ===\n")
    print(f"参数效率: ~1-5%")
    print(f"应用: 大模型微调")


if __name__ == "__main__":
    demo()
```

---

## 7. 手工代码实现

### 7.1 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """LoRA 线性层"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=1):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(
            torch.zeros(in_features, out_features),
            requires_grad=False
        )
        
        # A 和 B（可训练）
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank) * 0.01
        )
        
    def forward(self, x):
        # 原始输出
        base_output = F.linear(x, self.weight)
        
        # LoRA 输出
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.alpha
        
        return base_output + lora_output


def demo():
    print("=== LoRA 手工实现演示 ===\n")
    
    layer = LoRALinear(768, 768, rank=8)
    x = torch.randn(1, 768)
    
    output = layer(x)
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in layer.parameters()):,}")


if __name__ == "__main__":
    demo()
```

---

## 8. 可视化与结果理解

### 8.1 参数对比

```python
def plot_params():
    import matplotlib.pyplot as plt
    
    methods = ['Full FT', 'Adapter', 'LoRA', 'Prefix']
    params = [100, 3, 1, 0.5]
    
    plt.figure(figsize=(8, 5))
    plt.bar(methods, params, color='steelblue')
    plt.ylabel('参数量 (%)')
    plt.title('微调方法参数对比')
    plt.tight_layout()
    plt.savefig('lora_params.png')
```

---

## 9. 模型评估

### 9.1 微调效果

| 方法 | 参数 | GPT-3 性能 |
|------|------|-----------|
| Full | 100% | 基准 |
| Adapter | 3% | 95% |
| LoRA | 1% | 97% |
| Prefix | 0.5% | 93% |

---

## 10. 常见问题与易错点

### 10.1 秩选择

**问题**：r 太小效果差

**解决**：根据任务调整 r=8-32

---

## 11. 学习总结

**核心要点**：

1. **低秩适配**：冻结原权重
2. **参数效率**：仅 1-5%
3. **可切换**：卸载重装
4. **效果保持**：接近全参数微调

**LoRA 核心优势**：
- 参数量极少
- 效果接近全参数
- 可随时切换

**学习建议**：

1. 理解秩分解
2. 掌握 PEFT
3. 实践微调

---

## 12. 练习题与思考题

### 12.1 基础练习

1. LoRA vs 全参数微调
2. 秩选择原则

### 12.2 思考题

1. 为什么有效
2. 适用场景

---

### 12.3 详细答案

**问题**：为什么有效

**解答**：

预训练模型权重变化具有低秩结构，LoRA 可以捕获这种低秩变化。

---

## 14. 学习路径建议

### 入门阶段

1. 大模型基础
2. 微调概念

### 进阶阶段

1. LoRA 原理
2. PEFT 库

### 高级阶段

1. 实战微调
2. 多任务学习

**推荐路线**：

```
Fine-tuning → Adapter → LoRA → IA3 → Prompt Tuning
```

**LoRA 是大模型微调的行业标准，熟练掌握它对应用大模型很重要。**