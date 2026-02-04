# 模型训练与推理

本模块涵盖AI模型的训练优化、部署推理和生产实践。

## 🎯 学习目标

### 1. 训练优化
- **数据并行**：多GPU训练
- **分布式训练**：多机多卡
- **混合精度**：FP16、BF16、FP8
- **梯度累积**：大batch训练
- **梯度检查点**：节省显存

### 2. 推理优化
- **模型量化**：INT8、INT4量化
- **模型蒸馏**：知识蒸馏
- **模型剪枝**：移除冗余参数
- **模型融合**：算子融合
- **批处理优化**

### 3. 部署方案
- **本地部署**：CPU/GPU推理
- **云服务**：AWS、Azure、阿里云
- **边缘设备**：移动端、嵌入式
- **API服务**：REST、gRPC
- **实时推理**：低延迟优化

## 📚 训练技术

### 数据并行
```python
# PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# 模型包装
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 训练
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # 自动混合精度
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 梯度检查点
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        # 使用检查点节省显存
        return checkpoint(self._forward, x)

    def _forward(self, x):
        # 实际前向传播
        return self.layers(x)
```

### DeepSpeed / ZeRO
```python
import deepspeed

# 配置
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 2,  # ZeRO-2优化
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    },
    "fp16": {
        "enabled": True
    }
}

# 初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)
```

## 📚 推理优化

### 量化
```python
# 动态量化
import torch.quantization

model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
)

# 静态量化
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)
# ... 校准数据 ...
model_quantized = torch.quantization.convert(model_prepared)
```

### 模型蒸馏
```python
class DistillationLoss(nn.Module):
    def __init__(self, teacher, temperature=4.0, alpha=0.7):
        super().__init__()
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_output, labels, student_input):
        # 教师模型预测（soft targets）
        with torch.no_grad():
            teacher_output = self.teacher(student_input)

        # 蒸馏损失
        distill_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # 学生损失
        student_loss = F.cross_entropy(student_output, labels)

        # 组合
        return self.alpha * distill_loss + (1 - self.alpha) * student_loss
```

### TensorRT优化
```python
import tensorrt as trt

# 创建builder
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# 解析ONNX模型
parser.parse_from_file(model_onnx_path)

# 构建引擎
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
engine = builder.build_serialized_network(network)
```

### ONNX导出
```python
# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## 📚 部署方案

### FastAPI部署
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image

app = FastAPI()
model = torch.load('model.pth')
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # 预处理
    input_tensor = preprocess(image)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    return JSONResponse({"prediction": output.tolist()})

# 运行：uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker部署
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Triton Inference Server
```python
# Triton配置
# config.pbtxt
name: "my_model"
platform: "pytorch_libtorch"
max_batch_size: 16
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 224, 224, 3 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
```

### 移动端部署（CoreML）
```python
import coremltools

# 转换为CoreML
mlmodel = coremltools.convert(
    model,
    inputs=[coremltools.TensorType(shape=(1, 3, 224, 224))]
)
mlmodel.save('MyModel.mlmodel')
```

## 📚 监控和调试

### TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    # 训练...
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)
    writer.add_scalar('Learning_Rate', lr, epoch)

writer.close()
```

### Weights & Biases
```python
import wandb

wandb.init(project="my-project")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 32
}

for epoch in range(100):
    loss = train_epoch()
    wandb.log({"loss": loss, "epoch": epoch})

wandb.finish()
```

## 📚 推理框架对比

| 框架 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| ONNX Runtime | 跨平台、高性能 | 不支持所有模型 | 通用推理 |
| TensorRT | 极致性能 | 仅NVIDIA GPU | 生产环境 |
| OpenVINO | Intel优化 | 仅Intel硬件 | 边缘设备 |
| TorchScript | PyTorch原生 | 功能有限 | 快速部署 |
| TFLite | 移动端 | 功能受限 | 移动/嵌入式 |

## 📖 学习资源

### 文档
- PyTorch Distributed
- NVIDIA TensorRT
- ONNX Runtime
- FastAPI Documentation

### 课程
- Stanford CS231N（部署部分）
- Deep Learning Engineering（AI工程）

### 工具
- **调试**：GDB、CUDA-GDB
- **性能分析**：Nsight Systems、nvprof
- **监控**：Prometheus、Grafana

## 💡 最佳实践

### 训练优化
1. **使用混合精度**：加速训练，节省显存
2. **梯度累积**：模拟大batch
3. **分布式训练**：多卡并行
4. **数据预处理**：提前预加载
5. **混合CPU-GPU**：部分操作在CPU

### 推理优化
1. **模型量化**：减少模型大小和计算
2. **批处理**：充分利用GPU
3. **异步推理**：提高吞吐量
4. **缓存**：缓存常见结果
5. **负载均衡**：多实例部署

### 生产部署
1. **容器化**：Docker部署
2. **自动伸缩**：根据负载调整
3. **版本管理**：模型版本控制
4. **A/B测试**：对比模型效果
5. **监控告警**：性能和错误监控

## 🔧 性能指标

### 训练性能
- **Throughput**：samples/second
- **GPU利用率**：≥80%
- **显存占用**：≤90%

### 推理性能
- **延迟**：P50、P95、P99
- **吞吐量**：requests/second
- **资源利用率**：GPU/CPU使用率

### 业务指标
- **可用性**：SLA（如99.9%）
- **成本**：每1000次推理成本
- **用户体验**：响应时间<200ms

## 📝 学习路径

```
1. 基础训练优化（混合精度、梯度累积）
   ↓
2. 分布式训练（DDP、DeepSpeed）
   ↓
3. 推理优化（量化、ONNX）
   ↓
4. 服务部署（FastAPI、Docker）
   ↓
5. 生产实践（监控、日志、自动伸缩）
   ↓
6. 持续优化和迭代
```

## 💻 实践项目

### 初级
- [ ] 实现混合精度训练
- [ ] 简单模型量化
- [ ] FastAPI部署

### 中级
- [ ] 多GPU训练
- [ ] 模型蒸馏
- [ ] Docker容器化

### 高级
- [ ] DeepSpeed大模型训练
- [ ] TensorRT推理优化
- [ ] Kubernetes部署
- [ ] 自动化CI/CD
