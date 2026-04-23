# 面试题：深度模型训练出现 NaN 是什么原因？

面试题：深度模型训练出现 NaN 是什么原因？

深度模型训练中出现 NaN（Not a Number）通常由数值不稳定或计算错误导致，以下是常见原因分析：

# 一、数据问题

# 1. 输入数据含异常值

 原因：数据中存在 NaN、Inf 或极端值（如全零、极大/极小值），导致前向传播计算溢出。  
 解决：

 使用 numpy.isnan() 或 torch.isnan() 检查输入和标签数据。  
 确保数据预处理正确（如归一化、标准化），避免未处理的离群值。

# 2. 数据预处理缺陷

 原因：未归一化的数据（如图像未除以 255）或缺失值处理不当，引发激活值过大。  
 解决：

 对输入数据执行归一化（如缩放到 [0,1] 或 [-1,1]）。  
 对缺失值填充合理数值（如均值）或剔除异常样本。

# 二、模型问题

# 1. 梯度爆炸（Gradient Explosion）

 原因：反向传播时梯度指数级增长，导致权重更新后输出溢出。表现为 Loss 骤增后突变为 NaN，梯度值远超正常范围（如 >1e5）。  
 解决：

 梯度裁剪：限制梯度范数（如 PyTorch 的 clip_grad_norm_(max_norm=1.0)）。  
 降低学习率：初始学习率设为较小值（如 1e-4），或使用自适应优化器（Adam）。

# 2. 权重初始化不当

 原因：初始权重过大（如方差过大）或过小，引发激活值指数级变化。  
 解决：

 使用 Xavier （Tanh/Sigmoid）或 He 初始化 （ReLU）。  
 避免全零初始化导致对称性破坏。

# 三、训练策略问题

# 1. 混合精度训练问题

 原因：FP16 精度下数值范围小，易出现上/下溢出。  
 方案：启用梯度缩放（GradScaler in PyTorch），关键计算（如 Softmax）转为 FP32。

# 2. 学习率过高

 原因：过大学习率使权重更新剧烈，输出超出浮点范围。  
 调整：使用学习率调度器（如余弦退火、Warmup 等学习率调整策略）。

# 四、NaN 调试工具代码

```python
import torch
import torch.nn as nn
import numpy as np

class NaNDetector:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.nan_found = False

    def _check_tensor(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"[NaN检测] {name} 包含NaN! 形状: {tensor.shape}")
            self.nan_found = True
        if torch.isinf(tensor).any():
            print(f"[Inf检测] {name} 包含Inf! 形状: {tensor.shape}")
            self.nan_found = True

    def register_hooks(self):
        for name, param in self.model.named_parameters():
            def make_hook(n):
                def hook(grad):
                    self._check_tensor(grad, f"梯度-{n}")
                    if grad.abs().max() > 1e5:
                        print(f"[梯度爆炸警告] {n} 最大梯度: {grad.abs().max():.2e}")
                return hook
            param.register_hook(make_hook(name))

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                def make_fwd_hook(n):
                    def hook(module, input, output):
                        for i, inp in enumerate(input):
                            if isinstance(inp, torch.Tensor):
                                self._check_tensor(inp, f"前向输入-{n}-{i}")
                        if isinstance(output, torch.Tensor):
                            self._check_tensor(output, f"前向输出-{n}")
                    return hook
                self.hooks.append(module.register_forward_hook(make_fwd_hook(name)))

    def check_data(self, dataloader, max_batches=5):
        for i, (batch_x, batch_y) in enumerate(dataloader):
            if i >= max_batches:
                break
            self._check_tensor(batch_x, f"输入数据-batch{i}")
            self._check_tensor(batch_y, f"标签数据-batch{i}")

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()

def diagnose_nan(model, dataloader, criterion, optimizer, max_steps=100):
    detector = NaNDetector(model)
    detector.register_hooks()
    detector.check_data(dataloader)

    model.train()
    for step, (batch_x, batch_y) in enumerate(dataloader):
        if step >= max_steps:
            break
        optimizer.zero_grad()
        output = model(batch_x)
        if torch.isnan(output).any():
            print(f"Step {step}: 输出包含NaN!")
            break
        loss = criterion(output, batch_y)
        if torch.isnan(loss):
            print(f"Step {step}: Loss为NaN!")
            break
        loss.backward()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > 100:
            print(f"Step {step}: 梯度范数过大 {total_norm:.2f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    detector.cleanup()
    return not detector.nan_found
```

# 五、混合精度训练安全代码

```python
from torch.cuda.amp import autocast, GradScaler

class SafeMixedPrecisionTrainer:
    def __init__(self, model, optimizer, criterion, grad_clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.grad_clip = grad_clip
        self.scaler = GradScaler(init_scale=2.**14, growth_interval=1000)
        self.nan_count = 0
        self.max_nan_count = 10

    def train_step(self, batch_x, batch_y):
        self.optimizer.zero_grad()

        with autocast():
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)

        if torch.isnan(loss) or torch.isinf(loss):
            print("检测到NaN/Inf Loss，跳过此步")
            self.nan_count += 1
            if self.nan_count >= self.max_nan_count:
                raise RuntimeError(f"连续{self.max_nan_count}次NaN，训练终止")
            return None

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.nan_count = 0

        return loss.item()
```

# 六、梯度监控工具

```python
class GradientMonitor:
    def __init__(self, model, log_interval=100):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        self.grad_stats = {}

    def log_gradients(self):
        self.step_count += 1
        if self.step_count % self.log_interval != 0:
            return

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                stats = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "max": grad.max().item(),
                    "min": grad.min().item(),
                    "norm": grad.norm().item(),
                    "has_nan": torch.isnan(grad).any().item(),
                }
                self.grad_stats[name] = stats

        total_norm = sum(s["norm"] ** 2 for s in self.grad_stats.values()) ** 0.5
        nan_layers = [n for n, s in self.grad_stats.items() if s["has_nan"]]

        print(f"\n=== 梯度监控 Step {self.step_count} ===")
        print(f"总梯度范数: {total_norm:.4f}")
        if nan_layers:
            print(f"包含NaN的层: {nan_layers}")
        for name, stats in list(self.grad_stats.items())[:5]:
            print(f"  {name}: norm={stats['norm']:.4f}, "
                  f"mean={stats['mean']:.6f}, max={stats['max']:.4f}")
```

# 七、实际调试案例

**案例1：Embedding 查找导致 NaN**

```python
embedding = nn.Embedding(1000, 64)
ids = torch.tensor([500, 1001, 300])
try:
    output = embedding(ids)
except IndexError:
    print("Embedding索引越界，将越界ID裁剪到有效范围")
    ids = ids.clamp(0, embedding.num_embeddings - 1)
    output = embedding(ids)
```

**案例2：Log 运算导致 NaN**

```python
x = torch.tensor([0.0, 1e-10, 1.0])
unsafe_log = torch.log(x)
safe_log = torch.log(x + 1e-8)
print(f"不安全log: {unsafe_log}")
print(f"安全log: {safe_log}")
```

**案例3：除零导致 NaN**

```python
a = torch.tensor([1.0, 2.0, 0.0])
b = torch.tensor([1.0, 0.0, 3.0])
unsafe_div = a / b
safe_div = a / (b + 1e-8)
print(f"不安全除法: {unsafe_div}")
print(f"安全除法: {safe_div}")
```

# 八、NaN 排查清单

| 排查步骤 | 检查项 | 工具/方法 |
|---------|-------|----------|
| 1 | 输入数据是否含 NaN/Inf | `torch.isnan()`, `np.isfinite()` |
| 2 | 标签数据是否合法 | 检查标签范围、是否有负值 |
| 3 | 权重是否含 NaN | `model.parameters()` 遍历检查 |
| 4 | 梯度是否爆炸 | 梯度监控工具，检查范数 |
| 5 | 学习率是否过大 | 从 1e-5 开始逐步增大 |
| 6 | Loss 函数是否稳定 | 添加 epsilon 防止除零 |
| 7 | 混合精度是否溢出 | GradScaler, 关键层用 FP32 |
| 8 | Embedding 索引越界 | `clamp()` 限制索引范围 |
| 9 | 激活函数溢出 | Softmax 用 log_softmax |
| 10 | 正则化是否过强 | 降低 weight_decay |

# 九、学习路径建议

1. 理解浮点数精度限制（FP16/FP32/FP64）
2. 掌握梯度裁剪和权重初始化方法
3. 学习混合精度训练的原理与安全实践
4. 建立系统化的 NaN 调试流程
5. 研究数值稳定性的理论分析（Lipschitz 连续性等）
