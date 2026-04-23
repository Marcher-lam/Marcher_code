# RMSprop 学习文档

## 1. 算法基础认知

RMSprop（Root Mean Square Propagation）是由Geoff Hinton在Coursera课程中提出的自适应学习率优化算法。其核心思想是通过对梯度的平方进行指数加权移动平均来自动调整每个参数的学习率，从而解决不同参数需要不同学习率的问题。

在训练深度神经网络时，不同层的参数面临的优化难度完全不同：有的参数梯度变化剧烈，需要较小的学习率；有的参数梯度变化平缓，需要较大的学习率。传统的SGD使用固定学习率，难以同时满足所有参数的需求。RMSprop通过自适应调整学习率来解决这个问题。

RMSprop的工作机制可以理解为：对于梯度变化剧烈的参数，其平方梯度的累积值会较大，因此有效学习率会自动减小；对于梯度变化平缓的参数，其平方梯度的累积值较小，因此有效学习率会自动增大。这种自适应的机制使得RMSprop能够在复杂的多维优化问题中表现出色。

Geoff Hinton提出RMSprop的灵感来自于一个简单但深刻的观察：如果一个参数在过去的迭代中梯度一直很大，我们应该在未来的迭代中使用更小的学习率来避免震荡；反之，如果梯度一直很小，可以使用更大的学习率来加速收敛。这种思想简单而有效，使RMSprop成为深度学习中不可或缺的优化器。

## 2. 核心原理

### 2.1 二阶矩估计

RMSprop维护一个对梯度平方的指数加权移动平均：

```
v_t = β * v_{t-1} + (1 - β) * g_t²
```

其中v_t是二阶矩估计，β是衰减系数（通常0.9），g_t是当前梯度。这种设计使得v_t反映了近期梯度平方的平均大小。

### 2.2 自适应学习率

基于二阶矩估计，RMSprop计算每个参数的自适应学习率：

```
Δθ_t = -α * g_t / (√v_t + ε)
```

其中ε是一个小常数（通常10⁻⁸），防止除零。直观上，当梯度平方较大时（即参数变化剧烈），分母较大，学习率自动变小；当梯度平方较小时，学习率自动变大。

### 2.3 与AdaGrad的关系

RMSprop可以看作是AdaGrad的改进。AdaGrad累积所有历史梯度平方，存在问题是：随着训练进行，累积和学习率会趋近于零，导致训练提前停止。RMSprop通过使用指数加权移动平均替代简单累积来解决这个问题。

AdaGrad更新：
```
v_t = v_{t-1} + g_t²
```

RMSprop更新：
```
v_t = β * v_{t-1} + (1 - β) * g_t²
```

### 2.4 与动量结合

RMSprop可以与动量结合，形成更强大的优化器：

```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
Δθ_t = -α * m_t / (√v_t + ε)
```

这种组合（实际上是Adam的简化版本）在实践中表现出色。

## 3. 数学公式与推导

### 3.1 完整算法

RMSprop算法可以形式化如下：

Initialize: v_0 = 0

For iteration t = 1, 2, ...:

1. Compute gradient g_t = ∇θ L(θ)

2. Update second raw moment estimate:
   v_t = β * v_{t-1} + (1 - β) * g_t²

3. Compute update:
   Δθ = -α * g_t / (√v_t + ε)

4. Update parameters:
   θ_t = θ_{t-1} + Δθ

### 3.2 参数更新的直观理解

从另一个角度看RMSprop的参数更新。设每个参数维度独立考虑：

对于第i个参数：
```
θ_i ← θ_i - α * g_i / √(E[g_i²])
```

这里√v是梯度平方的均方根（RMS），因此方法命名为Root Mean Square Propagation。

### 3.3 收敛性分析

对于凸损失函数，RMSprop可以证明有O(1/√t)的收敛速度。这是因为二阶矩估计提供了对曲率的近似信息，帮助算法更有效地收敛。

### 3.4 与AdaGrad的对比

| 特性 | AdaGrad | RMSprop |
|------|--------|--------|
| 梯度累积 | 简单累积 | 指数移动平均 |
| 学习率趋势 | 单调递减 | 保持稳定 |
| 适用场景 | 凸优化 | 非凸深度学习 |

## 4. 训练过程讲解

### 4.1 训练流程

1. **初始化**：设置学习率α、二阶矩衰减β、数值稳定常数ε

2. **迭代更新**：
   - 计算当前batch的梯度
   - 更新二阶矩估计v
   - 计算自适应学习率
   - 更新参数

3. **重复**：重复直到收敛

### 4.2 超参数设置

- **学习率α**：0.001-0.01常用
- **衰减β**：0.9常用（对应4个epoch的记忆）
- **ε**：10⁻⁸防止除零

### 4.3 学习率调度

虽然RMSprop自动调整学习率，但仍可与调度结合：

- **学习率衰减**：每个epoch或一定步骤后降低
- **warmup**：前几个epoch使用较小的学习率
- **阶梯衰减**：固定epoch数后减半

### 4.4 实际应用

1. **循环神经网络**：RMSprop是RNN的标准优化器
2. **自动语音识别**：处理变长音频序列
3. **图像生成**：GAN训练中常用

## 5. 应用场景

### 5.1 循环神经网络

RMSprop特别适合训练RNN/LSTM，因为：

- 自适应学习率处理不同时间步的梯度变化
- 累积的平方梯度帮助稳定训练
- 避免RNN常见的梯度消失和爆炸

### 5.2 变长序列处理

在处理变长数据时，每个batch的统计量不同：

- 音频帧数不同
- 文本长度不同
- 时间序列长度不同

RMSprop的自适应特性处理这些变化。

### 5.3 生成模型

在GAN等生成模型中，RMSprop帮助：

- 处理判别器和生成器的不同学习动态
- 在非凸损失函数中稳定收敛
- 平衡不同参数的学习率

### 5.4 其他应用

1. **推荐系统**：大规模嵌入训练
2. **强化学习**：policy gradient方法
3. **多任务学习**：不同任务有不同学习动态

## 6. 优缺点分析

### 6.1 优点

1. **自适应学习率**：每个参数自动调整
2. **解决学习率敏感问题**：减少手动调参
3. **处理稀疏特征**：对稀疏特征友好
4. **内存效率高**：只存储v一个状态
5. **适用于RNN**：是RNN训练的标准选择

### 6.2 缺点

1. **二阶矩衰减敏感**：β选择影响性能
2. **可能不收敛**：某些情况下
3. **理论基础较弱**：收敛性证明不完善
4. **缺少动量**：收敛可能慢于带动量的方法

### 6.3 注意事项

1. **与动量结合**：可加入动量加速
2. **初始化**：v初始化为0
3. **数值稳定性**：需要ε防止除零

## 7. 调库实现（PyTorch完整代码）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_data(n_samples=1000, input_dim=20, noise=0.1):
    X = np.random.randn(n_samples, input_dim)
    true_weights = np.random.randn(input_dim)
    y = X @ true_weights + noise * np.random.randn(n_samples)
    n_train = int(0.7 * n_samples)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

def train_with_optimizer(optimizer_name, X_train, y_train, X_val, y_val, 
                       epochs=100, lr=0.001, batch_size=64):
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    model = SimpleNet(input_dim, 64, 1)
    
    if optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop_momentum':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t)
            val_pred = model(X_val_t)
            train_loss = criterion(train_pred, y_train_t).item()
            val_loss = criterion(val_pred, y_val_t).item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    return train_losses, val_losses

def compare_rmsprop():
    X_train, y_train, X_val, y_val = generate_data()
    
    print("=" * 50)
    print("Comparing RMSprop with Other Optimizers")
    print("=" * 50)
    
    optimizers = ['RMSprop', 'RMSprop_momentum', 'SGD', 'Adam']
    colors = {'RMSprop': 'blue', 'RMSprop_momentum': 'green', 'SGD': 'red', 'Adam': 'purple'}
    
    plt.figure(figsize=(10, 6))
    
    for opt in optimizers:
        train_losses, val_losses = train_with_optimizer(opt, X_train, y_train, X_val, y_val)
        plt.plot(val_losses, label=opt, color=colors[opt], linewidth=2)
        print(f"{opt}: Final Val Loss={val_losses[-1]:.4f}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('RMSprop Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('rmsprop_comparison.png', dpi=150)
    plt.show()

def visualize_learning_rate_effect():
    torch.manual_seed(42)
    
    def train_with_lr(lr):
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        X = torch.randn(500, 20)
        y = X @ torch.randn(20, 1) + torch.randn(500, 1) * 0.1
        
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
        criterion = nn.MSELoss()
        
        losses = []
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        return losses[-1]
    
    lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    final_losses = [train_with_lr(lr) for lr in lrs]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, final_losses, 'o-', linewidth=2)
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Loss')
    plt.title('RMSprop Learning Rate Effect')
    plt.grid(True, alpha=0.3)
    plt.savefig('rmsprop_lr.png', dpi=150)
    plt.show()
    
    print(f"Best LR: {lrs[np.argmin(final_losses)]}")
    print(f"Best Final Loss: {min(final_losses):.4f}")

def demonstrate_rmsprop_parameters():
    torch.manual_seed(42)
    
    model = SimpleNet(20, 64, 1)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    print("\n--- RMSprop Optimizer State ---")
    print(f"Number of parameter groups: {len(optimizer.param_groups)}")
    
    for i, group in enumerate(optimizer.param_groups):
        print(f"\nParameter Group {i}:")
        print(f"  Learning rate (alpha): {group['lr']}")
        print(f"  Alpha (decay): {group['alpha']}")
        print(f"  Epsilon: {group['eps']}")
        print(f"  Momentum: {group['momentum']}")
    
    sample_param = list(model.parameters())[0]
    if sample_param in optimizer.state:
        state = optimizer.state[sample_param]
        print(f"\nFirst param state keys: {state.keys()}")
        if 'square_avg' in state:
            print(f"  square_avg shape: {state['square_avg'].shape}")

def rmsprops_default_parameters():
    print("\n--- RMSprop Default Parameters ---")
    print("lr (learning rate): 0.01")
    print("alpha (decay): 0.99")
    print("eps: 1e-08")
    print("momentum: 0")

if __name__ == "__main__":
    compare_rmsprop()
    visualize_learning_rate_effect()
    demonstrate_rmsprop_parameters()
    rmsprops_default_parameters()
```

## 8. 手工代码实现（NumPy/PyTorch）

### 8.1 NumPy实现

```python
import numpy as np

class RMSpropOptimizer:
    def __init__(self, parameters, lr=0.001, alpha=0.99, eps=1e-8, momentum=0.0):
        self.parameters = parameters
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        
        self.square_avg = [np.zeros_like(p) for p in parameters]
        
        if momentum > 0:
            self.velocity = [np.zeros_like(p) for p in parameters]
    
    def step(self, gradients):
        for i, (p, g) in enumerate(zip(self.parameters, gradients)):
            self.square_avg[i] = self.alpha * self.square_avg[i] + (1 - self.alpha) * g ** 2
            
            avg = np.sqrt(self.square_avg[i] + self.eps)
            
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] + self.lr * g / avg
                p -= self.velocity[i]
            else:
                p -= self.lr * g / avg
        
        return self.parameters

class FullyConnected:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)
        self.input_cache = None
    
    def forward(self, x):
        self.input_cache = x
        return x @ self.weights + self.bias
    
    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        grad_weights = self.input_cache.T @ grad_output / batch_size
        grad_bias = np.sum(grad_output, axis=0) / batch_size
        grad_input = grad_output @ self.weights.T
        return grad_weights, grad_bias, grad_input

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class ManualRMSpropNet:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001):
        self.fc1 = FullyConnected(input_dim, hidden_dim)
        self.fc2 = FullyConnected(hidden_dim, hidden_dim)
        self.fc3 = FullyConnected(hidden_dim, output_dim)
        
        self.parameters = [
            self.fc1.weights, self.fc1.bias,
            self.fc2.weights, self.fc2.bias,
            self.fc3.weights, self.fc3.bias
        ]
        
        self.optimizer = RMSpropOptimizer(self.parameters, lr=lr)
        self.cache = {}
    
    def forward(self, x):
        out = relu(self.fc1.forward(x))
        self.cache['fc1_out'] = out
        out = relu(self.fc2.forward(out))
        self.cache['fc2_out'] = out
        out = self.fc3.forward(out)
        return out
    
    def train_step(self, x, y):
        output = self.forward(x)
        loss = np.mean((output - y) ** 2)
        
        grad = 2 * (output - y) / x.shape[0]
        
        gw3, gb3, grad = self.fc3.backward(grad)
        
        out = self.cache['fc2_out']
        grad = grad * relu_derivative(out)
        gw2, gb2, grad = self.fc2.backward(grad)
        
        out = self.cache['fc1_out']
        grad = grad * relu_derivative(out)
        gw1, gb1, _ = self.fc1.backward(grad)
        
        gradients = [gw1, gb1, gw2, gb2, gw3, gb3]
        self.parameters = self.optimizer.step(gradients)
        
        return loss

def train_manual_rmsprop():
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = X @ np.random.randn(20) + 0.1 * np.random.randn(1000)
    
    train_size = 700
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    
    model = ManualRMSpropNet(20, 64, 1, lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(100):
        indices = np.random.permutation(len(X_train))
        train_loss = 0.0
        for i in range(0, len(X_train), 32):
            batch_indices = indices[i:i+32]
            loss = model.train_step(X_train[batch_indices], y_train[batch_indices])
            train_loss += loss
        train_loss /= (len(X_train) // 32)
        
        val_pred = model.forward(X_val)
        val_loss = np.mean((val_pred - y_val) ** 2)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    return train_losses, val_losses

if __name__ == "__main__":
    print("Training with manual RMSprop implementation:")
    train_losses, val_losses = train_manual_rmsprop()
```

### 8.2 PyTorch手动实现

```python
import torch
import torch.nn as nn

class ManualRMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, momentum=0.0):
        defaults = {
            'lr': lr,
            'alpha': alpha,
            'eps': eps,
            'momentum': momentum
        }
        super(ManualRMSprop, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                
                square_avg = state['square_avg']
                alpha = group['alpha']
                
                state['step'] += 1
                
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                
                avg = torch.sqrt(square_avg + group['eps'])
                
                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad / avg)
                    p.data.add_(buf, value=-group['lr'])
                else:
                    p.data.addcdiv_(grad, avg, value=-group['lr'])
        
        return loss

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def compare_rmsprop_vs_pytorch():
    torch.manual_seed(42)
    
    model_manual = SimpleNet()
    model_pytorch = SimpleNet()
    
    for p_manual, p_pytorch in zip(model_manual.parameters(), model_pytorch.parameters()):
        p_manual.data.copy_(p_pytorch.data)
    
    X = torch.randn(32, 20)
    y = torch.randn(32, 1)
    
    optimizer_manual = ManualRMSprop(model_manual.parameters(), lr=0.01)
    optimizer_pytorch = torch.optim.RMSprop(model_pytorch.parameters(), lr=0.01)
    
    criterion = nn.MSELoss()
    
    for _ in range(10):
        optimizer_manual.zero_grad()
        loss_manual = criterion(model_manual(X), y)
        loss_manual.backward()
        optimizer_manual.step()
        
        optimizer_pytorch.zero_grad()
        loss_pytorch = criterion(model_pytorch(X), y)
        loss_pytorch.backward()
        optimizer_pytorch.step()
    
    print("Manual RMSprop params:", list(model_manual.parameters())[0][:3])
    print("PyTorch RMSprop params:", list(model_pytorch.parameters())[0][:3])
    print("Close:", torch.allclose(
        list(model_manual.parameters())[0],
        list(model_pytorch.parameters())[0],
        atol=1e-5
    ))

def visualize_square_avg():
    torch.manual_seed(42)
    
    model = SimpleNet()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    
    X = torch.randn(100, 20)
    y = torch.randn(100, 1)
    
    square_avg_norms = []
    criterion = nn.MSELoss()
    
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        
        norms = []
        for p in model.parameters():
            if p in optimizer.state and 'square_avg' in optimizer.state[p]:
                norms.append(optimizer.state[p]['square_avg'].norm().item())
        
        optimizer.step()
        
        if norms:
            square_avg_norms.append(np.mean(norms))
    
    print(f"\nSquare Average Norm Range: {min(square_avg_norms):.4f} - {max(square_avg_norms):.4f}")

if __name__ == "__main__":
    compare_rmsprop_vs_pytorch()
    visualize_square_avg()
```

## 9. 可视化与结果理解

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def visualize_adaptive_lr():
    torch.manual_seed(42)
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(2, 4)
            self.fc2 = nn.Linear(4, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleNet()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    
    X = torch.randn(100, 2)
    y = X[:, 0:1] + X[:, 1:2]
    
    parameter_grads = []
    effective_lrs = []
    criterion = nn.MSELoss()
    
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        
        for name, p in model.named_parameters():
            if p.grad is not None and p in optimizer.state:
                sq_avg = optimizer.state[p]['square_avg']
                grad = p.grad
                eff_lr = (0.01 / torch.sqrt(sq_avg + 1e-8)).mean().item()
                effective_lrs.append(eff_lr)
        
        optimizer.step()
    
    print(f"Effective LR Range: {min(effective_lrs):.6f} - {max(effective_lrs):.6f}")
    print("RMSprop adapts learning rate based on gradient history!")

def visualize_convergence():
    np.random.seed(42)
    torch.manual_seed(42)
    
    def train_and_record(optimizer_name, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        X = torch.randn(500, 20)
        y = X @ torch.randn(20, 1)
        
        if optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        elif optimizer_name == 'RMSprop_momentum':
            optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        criterion = nn.MSELoss()
        losses = []
        
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        return losses
    
    plt.figure(figsize=(12, 6))
    
    for name, color in [('RMSprop', 'blue'), ('RMSprop_momentum', 'green'), ('Adam', 'red')]:
        losses = train_and_record(name, 42)
        plt.plot(losses, label=name, color=color, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RMSprop Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('rmsprop_convergence.png', dpi=150)
    plt.show()

def visualize_square_avg_evolution():
    torch.manual_seed(42)
    
    model = nn.Sequential(
        nn.Linear(20, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    
    X = torch.randn(100, 20)
    y = X @ torch.randn(20, 1)
    
    square_avg_evolution = []
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        
        epoch_sq_avg = []
        for p in model.parameters():
            if p in optimizer.state and 'square_avg' in optimizer.state[p]:
                epoch_sq_avg.append(optimizer.state[p]['square_avg'].mean().item())
        
        if epoch_sq_avg:
            square_avg_evolution.append(np.mean(epoch_sq_avg))
        
        optimizer.step()
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(square_avg_evolution, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Square Average (log scale)')
    plt.title('RMSprop Square Average Evolution')
    plt.grid(True, alpha=0.3)
    plt.savefig('rmsprop_square_avg.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    visualize_adaptive_lr()
    visualize_convergence()
    visualize_square_avg_evolution()
```

## 10. 模型评估

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def comprehensive_evaluation():
    torch.manual_seed(42)
    np.random.seed(42)
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(20, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    X = np.random.randn(500, 20)
    y = X @ np.random.randn(20) + 0.1 * np.random.randn(500)
    
    n_train = 350
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1)
    
    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t).numpy()
            test_pred = model(X_test_t).numpy()
            
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
        
        train_losses.append({'mse': train_mse, 'r2': train_r2})
        test_losses.append({'mse': test_mse, 'r2': test_r2})
    
    model.eval()
    with torch.no_grad():
        final_pred = model(X_test_t).numpy()
        final_mse = mean_squared_error(y_test, final_pred)
        final_r2 = r2_score(y_test, final_pred)
    
    print("=" * 50)
    print("RMSprop Model Evaluation")
    print("=" * 50)
    print(f"Final Test MSE: {final_mse:.4f}")
    print(f"Final Test R²: {final_r2:.4f}")
    print(f"Final Train MSE: {train_losses[-1]['mse']:.4f}")
    print(f"Final Train R²: {train_losses[-1]['r2']:.4f}")
    
    print(f"\n--- RMSprop State ---")
    for name, param in model.named_parameters():
        optimizer_state = optimizer.state[param]
        if len(optimizer_state) > 0:
            if 'square_avg' in optimizer_state:
                print(f"{name}: square_avg norm = {optimizer_state['square_avg'].norm().item():.4f}")
    
    return train_losses, test_losses

if __name__ == "__main__":
    comprehensive_evaluation()
```

## 11. 常见问题与易错点

### 11.1 学习率选择

**问题**：学习率太大导致发散，太小收敛慢。

**解决方案**：默认0.001-0.01，取决于任务。

### 11.2 Alpha选择

**alpha**参数控制二阶矩的衰减速度。alpha太接近1会导致学习率趋近于零。

### 11.3 与动量结合

**问题**：何时使用动量？

**解决方案**：动量加速收敛，特别在RNN中常用。

### 11.4 数值不稳定性

**问题**：除零错误。

**解决方案**：设置eps=1e-8。

### 11.5 初始值为0

**问题**：square_avg初始化为0，早期估计有偏。

**解决方案**：这是RMSprop的设计特点，通常不影响收敛。

## 12. 学习总结

RMSprop是深度学习中重要的自适应学习率优化器，通过对梯度平方的指数加权移动平均实现了学习率的自适应调整。

**关键要点**：
1. 二阶矩估计：对梯度平方的指数移动平均
2. 自适应学习率：梯度大时学习率小，梯度小时学习率大
3. 解决AdaGrad问题：通过衰减累积替代简单累积
4. 适用于RNN：是RNN训练的标准选择

**实现要点**：
1. alpha=0.99是常用默认值
2. 可以添加动量加速收敛
3. 需要eps防止除零
4. 与Adam对比，缺少一阶矩

**最佳实践**：
1. 从默认参数开始
2. 在RNN中优先使用
3. 可与动量结合
4. 复杂任务考虑Adam

## 13. 练习题与思考题与思考题（含答案）

### 练习题

1. **简答题**：解释RMSprop如何自适应调整学习率。

2. **计算题**：如果alpha=0.9，计算当grad²=1.0时的有效学习率（lr=0.01, eps=1e-8）。

3. **代码题**：实现带动量的RMSprop。

4. **思考题**：为什么RMSprop适合训练RNN？

5. **分析题**：比较RMSprop与Adam的异同。

### 答案

1. **答案**：通过二阶矩v = alpha * v + (1-alpha) * g²，有效学习率为lr / √(v + eps)。梯度大时v增加，学习率减小。

2. **答案**：v=1.0，有效学习率=lr/√(v+eps)=0.01/√1=0.01

3. **答案**：见第8节代码实现。

4. **答案**：RNN中不同时间步梯度变化差异大，RMSprop的自适应学习率能处理这种变化。

5. **答案**：RMSprop只用二阶矩（梯度平方），Adam同时使用一阶矩（动量）和二阶矩。Adam更复杂但通常效果更好。

## 14. 学习路径建议建议

### 入门阶段
1. 理解SGD和梯度下降
2. 学习AdaGrad的问题
3. 实践RMSprop

### 进阶阶段
1. 与动量结合
2. 超参数调优
3. RNN训练应用

### 高级阶段
1. 收敛性理论
2. Adam对比
3. 大规模训练

### 推荐资源
- Hinton课程：Neural Networks for Machine Learning
- PyTorch文档：torch.optim.RMSprop
- 优化器对比实验论文