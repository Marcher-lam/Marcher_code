# CUDA 学习文档

## 1. 算法基础认知

CUDA（Compute Unified Device Architecture）是NVIDIA提供的通用并行计算平台和编程模型，使得开发者能够使用NVIDIA GPU进行高性能计算。CUDA的引入彻底改变了深度学习训练的速度，使得大规模神经网络训练成为可能。

### 1.1 为什么需要GPU计算？

GPU与CPU的计算架构存在本质差异：
- **CPU**：优化序列计算，少量强大核心（通常4-16核）
- **GPU**：优化并行计算，大量小型核心（数百到数千核）

深度学习的主要计算操作（如矩阵乘法、卷积）本质上是高度并行的，天然适合GPU加速。以矩阵乘法为例：将两个大矩阵相乘，可以分解为成千上万独立的点积运算，这些运算可以同时在GPU的不同核心上执行。

### 1.2 CUDA的核心概念

理解CUDA需要掌握几个关键概念：
- **Kernel**：在GPU上执行的函数，启动后会并行运行成千上万个线程
- **Thread**：GPU调度的最小单位，一个kernel包含多个thread
- **Block**：一组thread，最多512/1023个thread，可以协同访问共享内存
- **Grid**：一组block，覆盖整个kernel的调用范围
- **Warp**：32个thread的集合，GPU调度的基本单位

### 1.3 CUDA编程模型

CUDA程序通常遵循以下模式：
1. 将数据从CPU内存（host）拷贝到GPU内存（device）
2. 配置kernel启动参数（grid、block维度）
3. 调用kernel在GPU上执行计算
4. 将结果从GPU拷贝回CPU

## 2. 核心原理

### 2.1 GPU架构基础

理解CUDA需要理解GPU的硬件架构：

**Streaming Multiprocessor (SM)**：GPU的基本执行单元，包含：
- 多个CUDA核心（ALU）
- 共享内存（shared memory）
- 寄存器文件
- 调度单元

**Memory Hierarchy**：
- Global Memory：最大但访问最慢（~500GB/s）
- Shared Memory：小但快速（~10TB/s）
- Registers：最快但数量有限
- Constant/Texture Memory：只读缓存

### 2.2 CUDA编程模型详解

```cuda
// CUDA kernel定义
__global__ void matrixAdd(float *A, float *B, float *C, int N) {
    // 计算当前thread的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 主机代码
int main() {
    // 分配GPU内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    matrixAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    
    // 拷贝结果回CPU
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

### 2.3 Memory Coalescing（内存合并）

GPU内存访问效率的关键是让相邻线程访问相邻内存：

**合并访问**：Warp内所有thread访问连续的内存地址
```cuda
// 效率高的代码（合并访问）
__global__ void copy(float *src, float *dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[idx];  // 合并访问
    }
}

// 效率低的代码（非合并访问）
__global__ void copy_bad(float *src, float *dst, int N) {
    int idx = threadIdx.x * blockDim.x + blockIdx.x;  // 错误的索引顺序
    if (idx < N) {
        dst[idx] = src[idx];  // 非合并访问
    }
}
```

### 2.4 Shared Memory��Bank Conflicts

Shared Memory访问优化避免bank conflicts：

```cuda
__global__ void sharedAccess(float *data, int N) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    shared[tid] = data[blockIdx.x * blockDim.x + tid];
    __syncthreads();
    
    // 避免bank conflict的访问模式
    int offset = tid / 4;  // 相邻thread访问不同bank
    int idx = offset * 256 + (tid % 4);
    data[blockIdx.x * blockDim.x + tid] = shared[idx];
}
```

## 3. 数学公式与推导

### 3.1 GPU并行计算复杂度分析

对于矩阵运算，GPU相比CPU的加速比可以用Amdahl定律分析：

$$S_{GPU}(n) = \frac{1}{(1-P) + \frac{P}{n \cdot k}}$$

其中：
- $P$: 可并行化比例（通常>0.95）
- $n$: GPU核心数
- $k$: 每核心相对于CPU的加速比

### 3.2 Memory Bandwidth计算

GPU内存带宽是性能瓶颈：
$$BW = \text{memory\_clock} \times \text{memory\_bus\_width} \times \text{efficiency}$$

例如：GDDR6X 192-bit @ 19.5 GHz，有效带宽约1.5TB/s。

Compute Capability对应表：
| Capability | GPU型号 | 计算能力 |
|------------|---------|----------|
| 8.0/8.6 | A100 | ~15.8 TFLOPS |
| 8.9 | H100 | ~30 TFLOPS |
| 9.0 | H200/B100 | ~40+ TFLOPS |

### 3.3 Kernel Launch参数选择

Grid和Block维度选择原则：
- **Block大小**：通常是32的倍数（warp size），常见值128/256/512
- **Grid大小**：覆盖所有需要处理的数据元素

```python
def calculate_kernel_params(N, block_size=256):
    """计算kernel launch参数"""
    grid_size = (N + block_size - 1) // block_size
    return grid_size, block_size
```

## 4. 训练过程讲解

### 4.1 PyTorch中的CUDA基础

PyTorch提供了简洁的CUDA接口：

```python
import torch

# 检查CUDA可用性
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_count())

# 创建CUDA张量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# 在GPU上进行计算
z = torch.mm(x, y)  # 自动在GPU上执行
```

### 4.2 Device Placement（设备放置）

将模型和数据正确放置到GPU上：

```python
# 方法1：创建时指定设备
model = MyModel().to('cuda')
x = torch.randn(batch_size, input_dim, device='cuda')
y = model(x)

# 方法2：使用.to(device)
model = MyModel().to(device)
x = data.to(device)

# 方法3：使用.cuda()
model = MyModel().cuda()
x = x.cuda()
```

### 4.3 CUDA Streams（流）

使用多个流并行执行独立操作：

```python
import torch.cuda

# 创建多个流
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# 使用stream1加载数据
with torch.cuda.stream(stream1):
    data1 = preprocessor1(input1)
    output1 = model1(data1)

# 使用stream2加载另一个数据
with torch.cuda.stream(stream2):
    data2 = preprocessor2(input2)
    output2 = model2(data2)

# 等待所有流完成
torch.cuda.synchronize()
```

### 4.4 异步执行与同步

```python
# 异步提交kernel，无需等待完成
model(x)  # 立即返回，继续执行CPU代码

# 显式同步
torch.cuda.synchronize()  # 等待所有GPU操作完成

# 或者使用事件同步
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
model(x)
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
```

### 4.5 内存管理

```python
# 查看GPU内存使用
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())
print(torch.cuda.max_memory_allocated())

# 清理内存
torch.cuda.empty_cache()  # 释放未使用的缓存

# 删除张量释放内存
del x
torch.cuda.empty_cache()
```

## 5. 应用场景

### 5.1 深度学习训练

GPU最广泛的应用场景：

```python
import torch.nn as nn
import torch.optim as optim

# 完整的训练循环
model = nn.Linear(784, 10).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for batch_x, batch_y in train_loader:
        # 数据移到GPU
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        
        optimizer.zero_grad()
        
        # 前向传播（GPU自动执行）
        output = model(batch_x)
        loss = criterion(output, batch_y)
        
        # 反向传播（GPU自动执行）
        loss.backward()
        optimizer.step()
```

### 5.2 混合精度训练

利用Tensor Core加速：

```python
# PyTorch自动混合精度
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model = MyModel().cuda()

for batch_x, batch_y in train_loader:
    batch_x = batch_x.cuda()
    batch_y = batch_y.cuda()
    
    optimizer.zero_grad()
    
    # 自动使用FP16计算
    with autocast():
        output = model(batch_x)
        loss = criterion(output, batch_y)
    
    # 缩放loss避免下溢
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 5.3 CUDA扩展

使用CUDA C++扩展PyTorch：

```cpp
// my_extension.cu
#include <torch/torch.h>

__global__ void my_kernel(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

torch::Tensor my_function(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    int N = input.numel();
    int blocks = (N + 255) / 256;
    
    my_kernel<<<blocks, 256>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    
    return output;
}
```

## 6. 优缺点分析

### 6.1 优点

1. **极高并行度**：数千核心同时计算
2. **高内存带宽**：快速数据传输
3. **Tensor Core加速**：专用矩阵计算单元
4. **成熟的软件栈**：完善的工具和库
5. **深度学习优化**：cuDNN、cuBLAS高度优化

### 6.2 缺点

1. **NVIDIA独家**：需要NVIDIA GPU
2. **CUDA学习曲线**：比PyTorch更复杂
3. **调试困难**：GPU调试工具有限
4. **内存限制**：VRAM有限，大模型需技巧
5. **功耗较高**：数据中心应用需考虑

### 6.3 使用场景选择

**使用CUDA**：
- 大规模深度学习训练
- 需要充分利用GPU性能
- 自定义GPU kernel

**使用PyTorch自动**：
- 快速原型开发
- 标准模型训练
- 生产环境部署

## 7. 调库实现（Python + PyTorch）

### 7.1 完整训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 检查并配置CUDA
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# 准备数据
X = torch.randn(10000, 784)
y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 初始化模型和优化器
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 训练循环
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        # 数据移到GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        output = model(batch_x)
        loss = criterion(output, batch_y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 推理
model.eval()
with torch.no_grad():
    test_x = torch.randn(100, 784).to(device)
    output = model(test_x)
    pred = output.argmax(dim=1)
    print(f"Predictions shape: {pred.shape}")
```

### 7.2 多GPU训练

```python
import torch.nn as nn
from torch.nn.parallel import DataParallel

# 多GPU并行
model = MyModel().cuda()
model = nn.DataParallel(model)

# 或者使用DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

### 7.3 性能分析工具

```python
# CUDA事件计时
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# 执行计算
result = model(data)
end.record()

torch.cuda.synchronize()
print(f"Time: {start.elapsed_time(end)} ms")

# 使用PyTorch Profiler
from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    with record_function("inference"):
        model(data)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 7.4 Memory Profiling

```python
# 内存分析
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# 重置峰值统计
torch.cuda.reset_peak_memory_stats()
```

## 8. 手工代码实现（CUDA C++扩展）

### 8.1 简单的CUDA Kernel

```cpp
// add.cu
#include <stdio.h>

__global__ void add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    // Host内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Device内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 拷贝到GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 启动Kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    // 拷贝结果回CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 验证
    printf("Result[0]: %f\n", h_c[0]);
    printf("Result[%d]: %f\n", n-1, h_c[n-1]);
    
    // 释放
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

### 8.2 使用共享内存

```cuda
// matrix_transpose.cu
__global__ void transpose(float *input, float *output, int rows, int cols) {
    __shared__ float tile[256][256];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 加载到共享内存
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // 转置写入
    int tx = blockIdx.y * blockDim.y + threadIdx.x;
    int ty = blockIdx.x * blockDim.x + threadIdx.y;
    
    if (tx < rows && ty < cols) {
        output[tx * rows + ty] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 8.3 使用Warp Shuffle

```cuda
// warp_shuffle.cu
__global__ void warp_shuffle_reduce(float *input, float *output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = input[tid];
    
    // Warp级别并行归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset);
    }
    
    // 只第一个thread写入结果
    if (threadIdx.x % warpSize == 0) {
        output[blockIdx.x] = value;
    }
}
```

## 9. 可视化与结果理解

### 9.1 GPU利用率监控

```python
import subprocess
import time

def get_gpu_utilization():
    """获取GPU利用率"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv, ',noheader,nounits'],
        capture_output=True,
        text=True
    )
    return float(result.stdout.strip().split('\n')[0])

def monitor_gpu():
    """监控GPU使用情况"""
    for _ in range(10):
        util = get_gpu_utilization()
        mem_used = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv, ',noheader,nounits'],
            capture_output=True,
            text=True
        ).stdout.strip()
        print(f"GPU Utilization: {util}%, Memory: {mem_used} MiB")
        time.sleep(1)
```

### 9.2 性能对比可视化

```python
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

def benchmark_cpu_vs_gpu():
    """对比CPU和GPU性能"""
    sizes = [100, 500, 1000, 2000, 5000]
    cpu_times = []
    gpu_times = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for N in sizes:
        A = torch.randn(N, N)
        B = torch.randn(N, N)
        
        # CPU
        start = time.time()
        for _ in range(3):
            C = torch.mm(A, B)
        cpu_times.append((time.time() - start) / 3)
        
        # GPU
        if torch.cuda.is_available():
            A = A.to(device)
            B = B.to(device)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(3):
                C = torch.mm(A, B)
            torch.cuda.synchronize()
            gpu_times.append((time.time() - start) / 3)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, 'b-o', label='CPU')
    if gpu_times:
        plt.plot(sizes, gpu_times, 'r-o', label='GPU')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (s)')
    plt.title('CPU vs GPU Matrix Multiplication')
    plt.legend()
    plt.grid(True)
    plt.savefig('gpu_benchmark.png', dpi=150)
```

## 10. 模型评估

### 10.1 GPU性能指标

关键性能指标：
- **TFLOPS**：理论浮点性能
- **Memory Bandwidth**：内存带宽
- **SM Utilization**：Streaming Multiprocessor利用率
- **Occupancy**：并行度

### 10.2 性能瓶颈诊断

```python
# 检查GPU瓶颈
torch.autograd.set_detect_anomaly(True)

# 使用NVTX标记
import torch.cuda.nvtx as nvtx

nvtx.range_push("forward")
output = model(input)
nvtx.range_pop()

nvtx.range_push("backward")
loss.backward()
nvtx.range_pop()
```

## 11. 常见问题与易错点

### 11.1 CUDA Out of Memory

**错误**：GPU内存不足
**解决**：
```python
# 减少batch size
# 使用gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential(layers, input)

# 清理缓存
torch.cuda.empty_cache()
```

### 11.2 设备不一致

**错误**：模型在GPU，数据在CPU
**解决**：
```python
# 确保在同一设备
x = x.to(model.device)
# 或
model = model.cuda()
x = x.cuda()
```

### 11.3 未同步

**错误**：CPU代码在GPU操作完成前执行
**解决**：
```python
torch.cuda.synchronize()
```

### 11.4 CUDA版本不匹配

**错误**：CUDA版本与PyTorch不兼容
**解决**：安装匹配版本的PyTorch

## 12. 学习总结

### 核心要点

1. **GPU并行计算**：充分利用大规模并行
2. **CUDA编程模型**：理解grid/block/thread层次
3. **内存访问优化**：合并访问，避免bank conflicts
4. **PyTorch集成**：使用.cuda()和.to(device)
5. **性能调优**：使用profiler分析瓶颈

### 关键概念

- **Kernel**：GPU上执行的并行函数
- **Stream**：异步执行队列
- **Memory**：global/shared/registers
- **Synchronization**：同步控制

## 13. 练习题与思考题

### 练习题

**Q1**: 解释CUDA中grid、block、thread的关系

**答案**：Grid包含多个block，每个block包含多个thread。block内的thread可以协作访问共享内存，通过__syncthreads()同步。

**Q2**: 为什么GPU适合深度学习

**答案**：深度学习的主要操作（矩阵乘法、卷积）是高度并行的，可以分解为大量独立的计算单元。

**Q3**: 什么是memory coalescing

**答案**：让相邻thread访问相邻内存地址的优化技术，可以提高内存带宽利用率。

### 思考题

**Q1**: CUDA和OpenCL的区别

**答案**：CUDA是NVIDIA专有，OpenCL是跨平台的。CUDA工具链更成熟。

**Q2**: 如何优化CUDA性能

**答案**：1) 合并内存访问 2) 使用共享内存 3) 避免bank conflicts 4) 调整block大小 5) 使用warp shuffle

## 14. 学习路径建议

### 基础阶段
1. 学习PyTorch CUDA API
2. 理解GPU计算模型
3. ��践设备迁移

### 进阶阶段
1. 学习CUDA C++编程
2. 内存优化技术
3. 使用profiler分析

### 实践阶段
1. 性能优化项目
2. 自定义CUDA kernel
3. 分布式训练