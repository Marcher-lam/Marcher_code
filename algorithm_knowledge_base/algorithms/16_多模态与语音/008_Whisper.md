# Whisper 学习文档

> OpenAI开源的通用语音识别模型，zero-shot能力强

---

## 1. 算法基础认知

### 1.1 一句话定义

Whisper是由OpenAI于2022年开源的语音识别（ASR）模型，基于Transformer编码器-解码器架构，在68万小时标注音频上训练，具有强大的zero-shot能力，无需微调即可识别多种语言和任务。

### 1.2 直觉类比

Whisper就像一个"精通多语言的翻译员"。它学会了68种语言的各种说话方式——有的人口音重，有的人说话快，有的人在嘈杂环境。重要的是，这个"翻译员"不需要专门学习某个人的声音，只需要听到声音就能准确转写成文字。这就是预训练模型的强大：见过足够多的例子后，就能在新场景下泛化！

想象你有一个同声传译员：
- 他听了68万小时的语音（包括各种语言、各种口音、各种场景）
- 他不仅能识别说的什么，还能识别语言、翻译、标注时间戳
- 更神奇的是：他不需要专门学习某个人的声音就能工作！

### 1.3 发展背景

- 2022年9月，OpenAI开源Whisper
- 训练数据：68万小时语音数据（涵盖96种语言）
- 后续发布Whisper.cpp（高效推理，CPU也能运行）
- Model Scope和HuggingFace集成

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 语音识别 → 端到端ASR |
| 输出 | 文本转录、翻译、时间戳 |
| 模型 | Transformer Encoder-Decoder |
| 特点 | Zero-shot、多语言、多任务 |

---

## 2. 核心原理

### 2.1 架构设计

```
音频输入
    │
    ▼
Mel频谱 → 编码器（Transformer Encoder）
    │
    ▼
跨模态注意力 → 解码器（Transformer Decoder）
    │
    ▼
文本输出 + 语言识别 + 翻译 + 时间戳
```

### 2.2 vs 其他ASR模型

| 模型 | 训练方式 | Zero-shot | 特点 |
|------|----------|----------|------|
| Wav2Vec2 | 判别预训练 | 需要微调 | 纯Encoder |
| WHisper | Seq2Seq | **直接可用** | Encoder-Decoder |
| Coqui | CTC | 需要微调 | 实时 |

### 2.3 核心创新

1. **大规模预训练**：68万小时，覆盖96种语言
2. **多任务学习**：识别+翻译+语言检测+时间戳
3. **Encoder-Decoder**：跨模态注意力机制
4. **强泛化能力**：无需微调即可工作

### 2.4 训练目标

```python
# 多任务训练
loss = loss_识别 + loss_翻译 + loss_语言检测 + loss_时间戳
```

---

## 3. 数学公式与推导

### 3.1 音频预处理：Mel频谱

**参数**：
- 采样率：16kHz
- 帧长：25ms（400点）
- 帧移：10ms（160点）
- Mel维度：80

```python
# Mel频谱提取
def extract_mel(audio):
    # STFT
    stft = librosa.stft(audio, n_fft=400, hop_length=160)
    
    # 功率谱
    power = np.abs(stft)**2
    
    # Mel滤波
    mel_basis = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
    mel = mel_basis @ power
    
    # 对数
    log_mel = np.log(mel + 1e-8)
    
    return log_mel  # [80, T]
```

### 3.2 编码器

$$H_{enc} = \text{Encoder}(Mel_{spectrogram})$$

Transformer Encoder层：
$$H_{enc} = \text{MHA}(H_{enc}) + H_{enc}$$

### 3.3 解码器（自回归）

$$P(y_t|y_{<t}, H_{enc}) = \text{Decoder}(y_{<t}, H_{enc})$$

```python
# 伪代码
def decode(encoder_outputs, max_length=100):
    # 开始token
    ys = [tokenizer.bos_token_id]
    
    for _ in range(max_length):
        # 预测下一个token
        logits = decoder(ys, encoder_outputs)
        next_token = logits.topk(1)[1]
        
        if next_token == tokenizer.eos_token_id:
            break
        
        ys.append(next_token)
    
    return tokenizer.decode(ys)
```

### 3.4 多任务输出

| 任务 | Token | 说明 |
|------|-------|------|
| 转录 | `<|transcribe|>` | 语音转文字 |
| 翻译 | `<|translate|>` | 英译其他 |
| 语言检测 | `<|zh|>` 等 | 语言识别 |
| 时间戳 | `<|0.00|>` | 时间戳 |

---

## 4. 训练过程讲解

### 4.1 训练数据

- **68万小时**标注语音数据
- 96种语言（包括英语、中文、日语、西班牙语等）
- 多种场景（访谈、辩论、电话、会议）

### 4.2 模型规模

| 模型 | 参数 | FLOPs | 速度 |
|------|------|-------|------|
| tiny | 39M | 1x | 32x |
| base | 74M | 1x | 16x |
| small | 244M | 6x | 6x |
| medium | 769M | 18x | 2x |
| large | 1550M | 36x | 1x |

### 4.3 训练配置

```python
# 训练参数
config = {
    'batch_size': 16,
    'lr': 1e-5,
    'epochs': 10,
    'warmup_steps': 500,
}
```

---

## 5. 应用场景

### 5.1 语音识别

```python
import whisper

# 加载模型
model = whisper.load_model("base")

# 识别
result = model.transcribe("audio.wav")
print(result["text"])
```

### 5.2 多语言识别

```python
# 自动检测语言
result = model.transcribe("chinese_audio.wav")
print(result["text"])
print(f"Language: {result['language']}")
```

### 5.3 语音翻译

```python
# 英译中
result = model.transcribe("english.wav", task="translate", language="zh")
print(result["text"])
```

### 5.4 时间戳

```python
# 获取时间戳
result = model.transcribe("audio.wav", return_timestamps=True)
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}")
```

### 5.5 Whisper.cpp（高效推理）

```bash
# 安装
pip install whisper-cpp

# 转录
from whisper_cpp import Whisper
model = Whisper("ggml-base.bin")
result = model.transcribe("audio.wav")
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **Zero-shot强** | 无需微调直接可用 |
| 多语言 | 支持96种语言 |
| 开源 | 可商用（MIT协议） |
| 多任务 | 识别+翻译+时间戳 |
| 规模全 | tiny到large |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 大模型需GPU | large需24GB显存 |
| 不是实时 | 逐token生成 |
| 单语言翻译 | 仅英译其他 |
| 延迟 | 不是为实时设计 |

### 6.3 注意事项

- 音频质量影响大（降噪预处理）
- 长音频需要分段
- 建议从small开始测试

---

## 7. 调库实现（Python）

### 7.1 基本用法

```python
import whisper
import torch

# 检查GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型（可选tiny/base/small/medium/large）
model = whisper.load_model("base", device=device)

# 加载音频
audio = whisper.load_audio("speech.wav")

# 预处理为Mel频谱
mel = whisper.log_mel_spectrogram(audio).to(device)

# 识别
result = model.recognize(mel)
print(result[0].text)
```

### 7.2 完整参数

```python
result = model.transcribe(
    "audio.wav",
    language="Chinese",      # 指定语言（默认自动检测）
    task="transcribe",     # transcribe/translate
    beam_size=5,         # beam search大小
    best_of=5,           # 采样数
    temperature=0.0,    # 温度（0=确定性）
    compression_ratio_threshold=2.0,  # 压缩比阈值
    logprob_threshold=-1.0,       # log概率阈值
    condition_on_prev_text=True,      # 条件
)
```

### 7.3 Whisper.cpp使用

```bash
# 1. 编译
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make

# 2. 下载模型
./main -m models/ggml-base.bin -f audio.wav

# 3. Python绑定
pip install whisper-cpp-python

# 4. Python使用
from whisper_cpp import Whisper
model = Whisper("ggml-base.bin")
text = model.transcribe("audio.wav")
```

---

## 8. 手工代码实现（理解原理）

### 8.1 简化版Whisper

```python
import torch
import torch.nn as nn

class SimpleWhisper(nn.Module):
    """简化版Whisper - 理解原理"""
    def __init__(self, mel_dim=80, vocab_size=5000, encoder_dim=512):
        super().__init__()
        
        # 编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=8),
            num_layers=6
        )
        
        # 解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=encoder_dim, nhead=8),
            num_layers=6
        )
        
        # 嵌入
        self.mel_proj = nn.Linear(mel_dim, encoder_dim)
        self.token_embed = nn.Embedding(vocab_size, encoder_dim)
        
        # 输出
        self.output = nn.Linear(encoder_dim, vocab_size)
    
    def forward(self, mel_spectrogram, token_ids):
        # 编码器
        x = self.mel_proj(mel_spectrogram)  # [B, T, D]
        encoder_output = self.encoder(x)
        
        # 解码器
        tokens = self.token_embed(token_ids)  # [B, L, D]
        decoder_output = self.decoder(tokens, encoder_output)
        
        # 输出
        logits = self.output(decoder_output)
        
        return logits
    
    def transcribe(self, mel_spectrogram, max_len=100):
        """简化版转录"""
        device = mel_spectrogram.device
        ys = torch.tensor([[0]], device=device)  # start token
        
        for _ in range(max_len):
            logits = self.forward(mel_spectrogram, ys)
            next_token = logits[:, -1].argmax(-1)
            
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            
            if next_token == 0:  # end token
                break
        
        return ys[:, 1:]  # 去掉start token
```

### 8.2 测试

```python
if __name__ == "__main__":
    model = SimpleWhisper()
    mel = torch.randn(1, 100, 80)  # [B, T, Mel]
    tokens = torch.tensor([[1, 2, 3]])  # [B, L]
    
    output = model(mel, tokens)
    print(f"输出形状: {output.shape}")  # [B, L, vocab]
```

---

## 9. 可视化与结果理解

### 9.1 频谱可视化

```python
import matplotlib.pyplot as plt
import librosa

# 加载音频并提取Mel
audio, sr = librosa.load("speech.wav", sr=16000)
mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=80)

# 对数
log_mel = librosa.power_to_db(mel)

# 可视化
plt.figure(figsize=(12, 6))
plt.imshow(log_mel, aspect='auto', origin='lower')
plt.title("MelSpectrogram")
plt.xlabel("Time")
plt.ylabel("Mel")
plt.colorbar()
plt.savefig('whisper_mel.png', dpi=100)
plt.show()
```

### 9.2 注意力可视化

```python
# 可视化Cross Attention
def visualize_attention(encoder_output, decoder_output):
    """可视化跨模态注意力"""
    attn = torch.matmul(decoder_output, encoder_output.transpose(-2, -1))
    
    plt.figure(figsize=(12, 8))
    plt.imshow(attn[0].detach().numpy(), aspect='auto')
    plt.title("Cross-modal Attention")
    plt.xlabel("Encoder positions")
    plt.ylabel("Decoder positions")
    plt.colorbar()
    plt.savefig('whisper_attention.png', dpi=100)
    plt.show()
```

---

## 10. 模型评估

### 10.1 WERS（词错误率）

在Common Voice和LibriSpeech等数据集上的表现：

| 模型 | Whisper tiny | Whisper base | Whisper large |
|------|--------------|--------------|---------------|
| WER | 8.6% | 6.4% | 3.0% |

### 10.2 Zero-shot评估

无需微调，直接评估：

| 任务 | Zero-shot | Fine-tuned |
|------|------------|-------------|
| English ASR | 优秀 | 更优秀 |
| 多语言ASR | 良好 | 优秀 |

### 10.3 评估代码

```python
import whisper

# 加载模型
model = whisper.load_model("base")

# 评估WER
def compute_wer(predicted, reference):
    import editdistance
    return editdistance.distance(predicted, reference) / len(reference)

# 测试
result = model.transcribe("test.wav")
wer = compute_wer(result["text"], reference_text)
print(f"WER: {wer:.2%}")
```

---

## 11. 常见问题与易错点

### Q1: 为什么需要GPU？

**答案**：large模型1550M参数，需要24GB显存。tiny可在CPU运行。

### Q2: 长音频如何处理？

**答案**：自动分段处理，whisper会自动合并。

### Q3: 多语言如何选择？

**答案**：不指定则自动检测，建议指定language提高精度。

### Q4: 翻译支持哪些语言？

**答案**：只能翻译为英语，其他语言需其他模型。

### Q5: 如何提升准确性？

**答案**：短语音可在安静环境、降低temperature。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 架构 | Transformer Encoder-Decoder |
| 训练数据 | 68万小时语音 |
| 核心优势 | Zero-shot泛化 |
| 输出 | 文本+语言+翻译+时间戳 |

### 12.2 公式汇总

Mel频谱：
$$Mel = \text{MelFilterBank}(|STFT(audio)|^2)$$

编码器输出：
$$H_{enc} = \text{Encoder}(Mel)$$

解码器：
$$P(y_t) = \text{Decoder}(y_{<t}, H_{enc})$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. Whisper的核心优势是：
   - A) 计算快
   - B) Zero-shot
   - C) 小模型

2. Whisper不支持：
   - A) 语音识别
   - B) 实时转录
   - C) 语音翻译

### 13.2 简答题

1. 解释Whisper的Zero-shot能力来源。
2. 比较Whisper和Wav2Vec2的区别。

### 13.3 编程题

1. 实现基于Whisper的语音转文字。
2. 比较不同规模Whisper的精度和速度。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
语音识别基础
    ↓
Transformer
    ↓
Whisper原理
    ↓
实战部署
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| Wav2Vec2 | 类似但需要微调 |
| Whisper.cpp | 高效推理版 |
| Coqui | 实时ASR |

### 14.3 扩展阅读

- Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. OpenAI

---

## 附录

### 参考

1. Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. OpenAI
2. https://github.com/openai/whisper
3. https://github.com/ggerganov/whisper.cpp

---

**文档结束**