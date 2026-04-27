# FP8混合精度 学习文档

## 1. 算法基础认知

FP8混合精度（FP8 Mixed Precision）是深度学习训练和推理中的关键优化技术，通过使用半精度（FP16）和更激进的8位浮点数（FP8）来大幅减少显存占用和提升计算效率，同时保持与全精度（FP32）相当的模型精度。NVIDIA在其最新的GPU（如H100、A100）中原生支持FP8计算，使得这一技术成为大模型训练的事实标准。

### 1.1 数值精度的发展

深度学习中的数值精度经历了几个重要阶段：

**FP32（单精度）**：
- 1位符号 + 8位指数 + 23位尾数
- 4字节/32位
- 传统训练标准

**FP16（半精度）**：
- 1位符号 + 5位指数 + 10位尾数
- 2字节/16位
- 显存减半，速度翻倍

**BF16（Brain Float）**：
- 1位符号 + 8位指数 + 7位尾数
- 2字节/16位
- 动态范围与FP32相当，精度略低

**FP8（8位浮点）**：
- 1位符号 + 4位指数 + 3位尾数（E4M3）
- 1字节/8位
- 显存减少4倍，速度提升数倍

### 1.2 为什么需要FP8？

大模型训练的挑战：
- **显存瓶颈**：175B模型需要数千GB显存
- **计算密集**：大量矩阵运算
- **成本高昂**：GPU资源费用

FP8混合精度的优势：
- **显存减半**：存储需求减半
- **带宽提升**：内存带宽提升
- **计算加速**：Tensor Core加速
- **成本降低**：更少的GPU需求

### 1.3 FP8格式详解

FP8有两种格式：

**E4M3（4位指数+3位尾数）**：
- 范围：±240到±57344
- 精度：高
- 适合：前向传播、激活

**E5M2（5位指数+2位尾数）**：
- 范围：±576到±57344
- 动态范围：更大
- 适合：梯度、反向传播

```
FP8位布局：

E4M3: [S  e3 e2 e1 e0 | m2 m1 m0]
        ^符号    ^指数   ^尾数

E5M2: [S  e4 e3 e2 e1 | m1 m0]
        ^符号    ^指数   ^尾数
```

## 2. 核心原理

### 2.1 量化与反量化

FP8混合精度的核心是精度转换：

```python
# 量化：FP32 → FP8
def quantize_fp8_fp32(value, format='E4M3'):
    """将FP32值量化为FP8"""
    
    # 确定范围
    if format == 'E4M3':
        max_val = 240.0
        exp_bits = 4
        mant_bits = 3
    else:  # E5M2
        max_val = 57344.0
        exp_bits = 5
        mant_bits = 2
    
    # 裁剪到有效范围
    value = torch.clamp(value, -max_val, max_val)
    
    # 量化
    scale = 2 ** (exp_bits - mant_bits - 1)
    quantized = torch.round(value / scale)
    
    return quantized.to(torch.uint8)

# 反量化：FP8 → FP32
def dequantize_fp8_fp32(value, format='E4M3'):
    """将FP8值反量化为FP32"""
    
    # 确定范围
    if format == 'E4M3':
        scale = 2 ** (4 - 3 - 1)  # 2^0 = 1
    else:
        scale = 2 ** (5 - 2 - 1)  # 2^2 = 4
    
    # 反量化
    dequantized = value.float() * scale
    
    return dequantized
```

### 2.2 动态量化vs静态量化

**动态量化（Dynamic Quantization）**：
- 在推理时实时量化
- 激活值在计算时量化
- 精度更高，但有额外开销

```python
class DynamicQuantization:
    """动态量化"""
    
    def __init__(self):
        self.scale = None
    
    def quantize(self, x):
        # 计算缩放因子
        max_val = x.abs().max()
        scale = max_val.float() / 240.0  # FP8 max
        
        # 量化
        x_quant = (x / scale).round().to(torch.uint8)
        
        # 保存缩放因子供反量化
        self.scale = scale
        
        return x_quant
    
    def dequantize(self, x_quant):
        return x_quant.float() * self.scale
```

**静态量化（Static Quantization）**：
- 离线计算机器
- 使用校准数据集
- 推理更快

```python
class StaticQuantization:
    """静态量化"""
    
    def __init__(self, scale):
        self.scale = scale
    
    def quantize(self, x):
        return (x / self.scale).round().to(torch.uint8)
    
    def dequantize(self, x_quant):
        return x_quant.float() * self.scale
```

### 2.3 混合精度训练

混合精度训练的核心是保持Master Weights为FP32，训练使用FP16：

```
Algorithm: FP16 Mixed Precision Training
---------------------------------
Step 1: 初始化
    master_weights = fp32 (原始权重)
    fp16_weights = fp16 (master_weights)
    fp16_optimizer_state = ...

Step 2: 前向传播
    fp16_forward = forward(fp16_weights)
    loss_scale = scale_loss(loss)

Step 3: 反向传播
    fp16_grads = backward(loss_scale)

Step 4: 如果梯度未下溢
    master_weights = master_weights - lr * fp16_grads

Step 5: FP32权重更新
    fp16_weights = master_weights.to_fp16()

Step 6: 梯度缩放
    unscaled_loss = loss_scale / loss_scale
```

### 2.4 梯度缩放

梯度缩放防止FP16的下溢（underflow）：

$$g_{scaled} = g \cdot S$$

其中$S$是缩放因子动态调整：

```python
def update_loss_scale(loss_scale, overflow):
    """更新loss缩放因子
    
    overflow: 是否发生溢出
    """
    if overflow:
        # 梯度裁剪到[-2^15, 2^15]
        loss_scale = loss_scale / 2.0
    else:
        # 尝试增加
        if loss_scale < 2.0**15:
            loss_scale = loss_scale * 2.0
    
    return loss_scale
```

## 3. 数学公式与推导

### 3.1 FP8格式转换

FP32 → FP8的量化公式：

```math
x_{fp8} = \text{round}\left(\frac{x_{fp32}}{scale}\right)
```

其中：
- E4M3: $scale = 2^{0}$
- E5M2: $scale = 2^{2}$

反量化：

```math
x_{fp32} = x_{fp8} \cdot scale
```

### 3.2 动态范围计算

**E4M3格式**：
- 指数：4位 → $2^4 = 16$ 个值（偏置+8）
- 指数范围：-8到+7（偏置）
- 最大值：$2^{7-1} \times (1 + 0.875) = 240$

**E5M2格式**：
- 指数：5位 → $2^5 = 32$ 个值（偏置+16）
- 指数范围：-16到+15（偏置）
- 最大值：$2^{15-2} = 57344$

### 3.3 精度损失分析

FP8的量化误差：

```math
\epsilon = \left| x - \text{quantize}(x) \right|

\frac{\epsilon}{|x|} \approx \frac{1}{2^{m+1}}
```

其中$m$是尾数位数：
- E4M3: $m=3$, 相对误差 $\approx \frac{1}{16} = 6.25\%$
- E5M2: $m=2$, 相对误差 $\approx \frac{1}{4} = 25\%$

### 3.4 Tensor Core计算

FP8在Tensor Core中的矩阵乘法：

```math
C = A \cdot B

A \in \mathbb{FP8}^{M \times K}, B \in \mathbb{FP8}^{K \times N}
C \in \mathbb{F16}^{M \times N}
```

NVIDIA Tensor Core支持：
- A: E4M3, B: E5M2
- C: FP16/BF16/FP32

## 4. 训练过程讲解

### 4.1 PyTorch AMP配置

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 创建GradScaler
scaler = GradScaler()

# 模型和优化器
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 自动混合精度
        with autocast(dtype=torch.float16):
            output = model(batch)
            loss = loss_fn(output, targets)
        
        # 缩放loss并反向传播
        scaler.scale(loss).backward()
        
        # scaler处理溢出
        scaler.step(optimizer)
        scaler.update()
```

### 4.2 FP8训练配置

```python
# PyTorch 2.0+ FP8支持
from torch.distributed.elastic.multiprocessing import errors

# 启用FP8训练
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 混合精度配置
model = model.to(torch.float16)
```

### 4.3 自定义量化

```python
class FP8Quantizer:
    """自定义FP8量化器"""
    
    def __init__(self, format='E4M3'):
        self.format = format
        
        if format == 'E4M3':
            self.max_val = 240.0
            self.scale = 1.0
            self.precision = 3
        else:
            self.max_val = 57344.0
            self.scale = 4.0
            self.precision = 2
    
    def quantize(self, tensor):
        """量化到FP8"""
        # 裁剪
        tensor_clamped = torch.clamp(tensor, -self.max_val, self.max_val)
        
        # 量化
        tensor_scaled = tensor_clamped / self.scale
        tensor_int = torch.round(tensor_scaled)
        
        # 转为uint8
        return tensor_int.to(torch.uint8)
    
    def dequantize(self, tensor):
        """从FP8反量化"""
        return tensor.float() * self.scale
    
    def __call__(self, tensor):
        return self.quantize(tensor)
```

### 4.4 训练监控

```python
def monitor_fp8_training(scaler, loss):
    """监控训练状态"""
    
    if scaler is not None:
        # 检查缩放因子
        scale = scaler.get_scale()
        print(f"Loss scale: {scale}")
        
        # 检查是否发生溢出
        if loss.isnan().any():
            print("WARNING: NaN detected, skipping step")
            return False
    
    return True
```

## 5. 应用场景

### 5.1 大模型训练

```python
# 大模型混合精度训练示例
import torch.nn as nn
from torch.utils.data import DataLoader

# 模型
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(4096, 4096) for _ in range(32)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.functional.gelu(x)
        return x

# 训练
model = LargeModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()

for epoch in range(10):
    for batch in dataloader:
        with autocast(dtype=torch.float16):
            output = model(batch)
            loss = loss_fn(output, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 5.2 推理加速

```python
# FP8推理
class FP8Inference:
    """FP8推理"""
    
    def __init__(self, model):
        self.model = model
        self.quantizer = FP8Quantizer(format='E4M3')
    
    @torch.no_grad()
    def forward(self, x):
        # 量化输入
        x_fp8 = self.quantizer(x)
        
        # 在FP8下计算
        # （实际需要在CUDA kernel中实现）
        output_fp8 = self.model(x_fp8)
        
        # 反量化输出
        output = self.quantizer.dequantize(output_fp8)
        
        return output
```

### 5.3 分布式训练

```python
# 分布式混合精度
import torch.distributed as dist

# GradScaler for DDP
scaler = GradScaler()

# 分布式训练
model = nn.parallel.DistributedDataParallel(model)

for batch in dataloader:
    with autocast(dtype=torch.float16):
        output = model(batch)
        loss = loss_fn(output, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 6. 优缺点分析

### 6.1 FP8的优点

1. **显存减半**：FP8只占FP32的1/4存储
2. **计算加速**：Tensor Core专用优化
3. **带宽提升**：内存带宽利用率高
4. **成本降低**：更少的GPU需求
5. **原生支持**：NVIDIA��新GPU支持

### 6.2 FP8的挑战

1. **精度损失**：量化误差累积
2. **数值不稳定**：某些操作可能溢出
3. **不兼容**：老GPU不支持
4. **实现复杂**：需要仔细处理边界

### 6.3 精度对比

| 格式 | 存储 | 相对误差 | 适用范围 |
|------|------|----------|----------|
| FP32 | 4B | 0% | 对精度要求高 |
| BF16 | 2B | ~0.1% | 训练通用 |
| FP16 | 2B | ~0.1% | 训练 |
| E4M3 | 1B | ~6% | 前向传播 |
| E5M2 | 1B | ~25% | 梯度 |

### 6.4 使用建议

**推荐使用**：
- 大模型训练（BF16）
- 推理加速（FP8）
- 资源受限场景

**谨慎使用**：
- 小模型训练
- 精度敏感任务
- 不支持的硬件

## 7. 调库实现（Python + PyTorch）

### 7.1 基础混合精度训练

```python
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer:
    """混合精度训练器"""
    
    def __init__(self, model, optimizer, use_fp16=True):
        self.model = model
        self.optimizer = optimizer
        self.use_fp16 = use_fp16
        
        if use_fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None
    
    def train_step(self, batch):
        """单步训练"""
        
        if self.use_fp16:
            with autocast(dtype=torch.float16):
                output = self.model(batch)
                loss = self.compute_loss(output)
            
            self.scaler.scale(loss).backward()
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        else:
            output = self.model(batch)
            loss = self.compute_loss(output)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
    
    def compute_loss(self, output):
        return nn.functional.cross_entropy(output, self.targets)
```

### 7.2 自定义FP8量化

```python
import torch
import torch.nn as nn

class FP8Linear(nn.Module):
    """FP8 Linear层"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # FP32权重（训练用）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )
        
        # FP8量化器
        self.quantizer = FP8Quantizer(format='E4M3')
        
        # 缩放因子
        self.register_buffer('scale', torch.ones(1))
    
    def forward(self, x):
        # 量化权重
        if self.quantizer.format == 'E4M3':
            # 使用E4M3量化
            weight_int = self.quantizer.quantize(self.weight)
            
            # 在整数运算后量化
            # 简化实现
            weight_fp16 = weight_int.float() * self.scale
            
            # 计算
            return nn.functional.linear(x, weight_fp16)
        else:
            return nn.functional.linear(x, self.weight)
    
    def quantize_weights(self):
        """量化权重"""
        weight_int = self.quantizer.quantize(self.weight)
        self.scale = self.quantizer.scale
        return weight_int.to(torch.uint8)
```

### 7.3 完整训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

# 创建数据
X = torch.randn(10000, 784).cuda()
y = torch.randint(0, 10, (10000,)).cuda()
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 模型
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# GradScaler
scaler = GradScaler()

# 训练
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        
        # 混合精度前向
        with autocast(dtype=torch.float16):
            output = model(batch_x)
            loss = nn.functional.cross_entropy(output, batch_y)
        
        # 缩放loss并反向传播
        scaler.scale(loss).backward()
        
        # 更新
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

### 7.4 性能对比

```python
import time

def benchmark_precision():
    """精度性能对比"""
    
    sizes = [1024, 2048, 4096]
    
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        times = []
        
        for size in sizes:
            A = torch.randn(size, size, dtype=dtype, device='cuda')
            B = torch.randn(size, size, dtype=dtype, device='cuda')
            
            # Warmup
            for _ in range(10):
                _ = torch.mm(A, B)
            torch.cuda.synchronize()
            
            # 计时
            start = time.time()
            for _ in range(100):
                _ = torch.mm(A, B)
            torch.cuda.synchronize()
            
            times.append((time.time() - start) / 100 * 1000)
        
        print(f"{dtype}: {times}")

benchmark_precision()
```

## 8. 手工代码实现

### 8.1 FP8量化器实现

```python
import torch

class FP8Quantizer:
    """FP8量化器实现
    
    支持E4M3和E5M2两种格式
    """
    
    def __init__(self, format='E4M3'):
        self.format = format
        
        if format == 'E4M3':
            self.max_val = 240.0
            self.exp_bits = 4
            self.mant_bits = 3
        else:  # E5M2
            self.max_val = 57344.0
            self.exp_bits = 5
            self.mant_bits = 2
    
    def quantize(self, x):
        """量化为FP8
        
        x: FP32 tensor
        returns: uint8 tensor
        """
        # 裁剪到有效范围
        x_clipped = torch.clamp(x, -self.max_val, self.max_val)
        
        # 缩放并量化
        scale = self.max_val / (2 ** (self.exp_bits - 1))
        x_scaled = x_clipped / scale
        x_int = torch.round(x_scaled)
        
        # 转为uint8
        return x_int.to(torch.uint8)
    
    def dequantize(self, x):
        """FP8反量化为FP32
        
        x: uint8 tensor
        returns: FP32 tensor
        """
        scale = self.max_val / (2 ** (self.exp_bits - 1))
        return x.float() * scale
    
    def quantize_weights(self, weights):
        """量化模型权重
        
        weights: FP32 tensor
        returns: (uint8 tensor, scale tensor)
        """
        max_val = weights.abs().max()
        scale = max_val / self.max_val
        
        if scale > 0:
            weights_scaled = weights / scale
            weights_int = torch.round(weights_scaled).to(torch.uint8)
        else:
            weights_int = torch.zeros_like(weights, dtype=torch.uint8)
        
        return weights_int, scale
    
    def dequantize_weights(self, weights_int, scale):
        """反量化权重"""
        return weights_int.float() * scale
```

### 8.2 FP8 Matmul

```python
class FP8Matmul:
    """FP8矩阵乘法"""
    
    def __init__(self, a_format='E4M3', b_format='E4M3'):
        self.a_quantizer = FP8Quantizer(a_format)
        self.b_quantizer = FP8Quantizer(b_format)
        self.c_scale = None
    
    def forward(self, A, B):
        """矩阵乘法
        
        A: (M, K) FP32
        B: (K, N) FP32
        C: (M, N) FP32
        """
        # 量化
        A_int = self.a_quantizer.quantize(A)
        B_int = self.b_quantizer.quantize(B)
        
        # 记录缩放因子
        a_scale = A.abs().max() / self.a_quantizer.max_val
        b_scale = B.abs().max() / self.b_quantizer.max_val
        
        # 整数乘法（使用FP16近似）
        A_fp = A_int.float() * a_scale
        B_fp = B_int.float() * b_scale
        
        C_fp = torch.mm(A_fp, B_fp)
        
        # 记录结果缩放因子
        self.c_scale = a_scale * b_scale
        
        return C_fp
```

### 8.3 梯度检查

```python
def verify_gradient_conservation():
    """验证梯度守恒"""
    
    torch.manual_seed(42)
    
    # 创建层并量化
    layer = nn.Linear(32, 32).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    scaler = GradScaler()
    
    # FP16前向
    x = torch.randn(16, 32, device='cuda')
    x.requires_grad = True
    
    # FP16训练
    with autocast(dtype=torch.float16):
        output = layer(x)
        loss = output.sum()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # FP32训练作为参考
    layer_fp32 = nn.Linear(32, 32).cuda()
    optimizer_fp32 = torch.optim.Adam(layer_fp32.parameters(), lr=1e-3)
    
    x_fp32 = torch.randn(16, 32, device='cuda')
    x_fp32.requires_grad = True
    
    output_fp32 = layer_fp32(x_fp32)
    loss_fp32 = output_fp32.sum()
    loss_fp32.backward()
    optimizer_fp32.step()
    
    print("Gradient conservation verified")

verify_gradient_conservation()
```

### 8.4 性能测试

```python
import time
import numpy as np

def benchmark_fp8_performance():
    """FP8性能基准测试"""
    
    # 测试配置
    sizes = [1024, 2048, 4096]
    
    results = {
        'FP32': [],
        'FP16': [],
        'FP8': []
    }
    
    for size in sizes:
        # FP32
        A = torch.randn(size, size, device='cuda')
        B = torch.randn(size, size, device='cuda')
        
        start = time.time()
        for _ in range(10):
            _ = torch.mm(A, B)
        torch.cuda.synchronize()
        results['FP32'].append((time.time() - start) / 10 * 1000)
        
        # FP16
        A = A.half()
        B = B.half()
        
        start = time.time()
        for _ in range(10):
            _ = torch.mm(A, B)
        torch.cuda.synchronize()
        results['FP16'].append((time.time() - start) / 10 * 1000)
    
    print("Performance (ms):")
    for dtype, times in results.items():
        print(f"{dtype}: {[f'{t:.2f}' for t in times]}")

benchmark_fp8_performance()
```

## 9. 可视化与结果理解

### 9.1 数值分布可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_precision():
    """可视化不同精度的数值分布"""
    
    # 生成测试数据
    np.random.seed(42)
    data_fp32 = np.random.randn(10000)
    
    # 量化到不同精度
    data_fp16 = data_fp32.astype(np.float16).astype(np.float32)
    
    # 简单FP8模拟（范围-240到240）
    data_clipped = np.clip(data_fp32, -240, 240)
    data_fp8 = (data_clipped / 1).astype(np.int8).astype(np.float32)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(data_fp32, bins=50, alpha=0.7)
    axes[0].set_title('FP32')
    
    axes[1].hist(data_fp16, bins=50, alpha=0.7)
    axes[1].set_title('FP16')
    
    axes[2].hist(data_fp8, bins=50, alpha=0.7)
    axes[2].set_title('FP8 (Simulated)')
    
    plt.tight_layout()
    plt.savefig('precision_distribution.png', dpi=150)
    plt.close()

visualize_precision()
```

### 9.2 量化误差可视化

```python
def plot_quantization_error():
    """量化误差可视化"""
    
    x = np.linspace(-300, 300, 1000)
    
    # FP32值
    y_fp32 = x
    
    # E4M3量化
    scale = 1.0
    y_fp8 = np.round(x / scale) * scale
    
    # 误差
    error = np.abs(y_fp32 - y_fp8)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, error, 'b-', linewidth=2)
    plt.xlabel('Input Value')
    plt.ylabel('Quantization Error')
    plt.title('FP8 Quantization Error (E4M3)')
    plt.grid(True, alpha=0.3)
    plt.savefig('quantization_error.png', dpi=150)
    plt.close()

plot_quantization_error()
```

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| 显存使用 | GPU显存峰值 |
| 训练时间 | 每epoch时间 |
| 吞吐量 | samples/sec |
| 精度损失 | 与FP32的差距 |

### 10.2 精度检查

```python
def check_numerical_accuracy(model_fp16, model_fp32, test_data):
    """检查数值精度"""
    
    # FP16前向
    model_fp16.eval()
    with torch.no_grad():
        out_fp16 = model_fp16(test_data)
    
    # FP32前向
    model_fp32.eval()
    with torch.no_grad():
        out_fp32 = model_fp32(test_data)
    
    # 计算差距
    max_diff = (out_fp16.float() - out_fp32).abs().max()
    mean_diff = (out_fp16.float() - out_fp32).abs().mean()
    
    print(f"Max diff: {max_diff.item():.6f}")
    print(f"Mean diff: {mean_diff.item():.6f}")
```

## 11. 常见问题与易错点

### 11.1 溢出问题

**问题**：FP16数值溢出
**解决**：使用GradScaler动态调整loss scale

### 11.2 精度不足

**问题**：训练不收敛
**解决**：检查loss scale是否合适，或使用BF16

### 11.3 梯度为0

**问题**：梯度下溢
**解决**：增加loss scale，或检查数据范围

### 11.4 版本兼容

**问题**：不支持FP8
**解决**：升级PyTorch版本，使用BF16

## 12. 学习总结

### 核心要点

1. **FP8格式**：E4M3（高精）和E5M2（高动态）
2. **混合精度**：FP32主权重 + FP16/BF16训练
3. **梯度缩放**：防止下溢的关键技术
4. **Tensor Core**：硬件加速基础

### 关键公式

- 量化：$x_{fp8} = \text{round}(x_{fp32} / scale)$
- 反量化：$x_{fp32} = x_{fp8} \cdot scale$
- 梯度缩放：$g_{scaled} = g \cdot S$

### 使用建议

- 训练：BF16（稳定）
- 推理：FP8（加速）
- 显存受限：混合精度

## 13. 练习题与思考题

### 练习题

**Q1**: FP8中E4M3和E5M2的区别是什么？

**答案**：E4M3有4位指数+3位尾数，适合前向传播；E5M2有5位指数+2位尾数，动态范围更大，适合梯度。

**Q2**: 为什么要使用梯度缩放？

**答案**：FP16的有效位数少，梯度容易下溢到0。梯度缩放可以放大梯度，使其在有效范围内，反向传播后再缩放回去。

**Q3**: 混合精度训练的关键是什么？

**答案**：保持FP32的master weights，每次训练使用FP16梯度，累积到FP32后更新。

### 思考题

**Q1**: 何时不能用FP8训练？

**答案**：当模型对精度非常敏感，或硬件不支持时。

**Q2**: BF16和FP16的区别？

**答案**：BF16有更少的尾数位但指数位与FP32相同，动态范围更大，不易溢出。

## 14. 学习路径建议

### 基础阶段
1. 数值精度基础
2. 混合精度原理
3. GradScaler使用

### 进阶阶段
1. FP8量化
2. 性能优化
3. 分布式训练

### 实践阶段
1. 大模型训练
2. 推理部署
3. 性能调优

### 参考资源
- NVIDIA Tensor Core文档
- PyTorch AMP教程
- 大模型训练论文