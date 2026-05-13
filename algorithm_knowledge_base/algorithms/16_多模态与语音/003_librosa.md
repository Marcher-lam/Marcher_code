# librosa 音频处理学习文档

## 1. 算法基础认知

librosa是一个功能强大的Python音频信号处理库，广泛用于音乐信息检索（MIR）、语音识别、声音合成等音频相关任务。librosa提供了丰富的音频分析和特征提取功能，包括梅尔频谱图、短时傅里叶变换、梅尔频率倒谱系数等核心算法。

librosa的核心设计理念是提供简单直观的API，同时保持底层算法的专业性和准确性。它建立在FFT、离散傅里叶变换等数学基础之上，针对音频信号处理的特殊需求进行了优化。

librosa的常用功能包括：
1. 加载和保存音频文件
2. 计算各种频谱表示
3. 提取音频特征
4. 音频预处理和增强
5. 可视化音频数据

## 2. 核心原理

### 2.1 短时傅里叶变换（STFT）

短时傅里叶变换是librosa大多数频谱分析的基础。其基本原理是将音频信号分帧，对每一帧进行傅里叶变换。

STFT的数学定义为：

$$X(n, \omega) = \sum_{m=-\infty}^{\infty} x[m] \cdot w[n-m] \cdot e^{-j\omega m}$$

其中：
- $x[m]$ 是输入信号
- $w[n-m]$ 是窗函数（通常是汉宁窗或汉明窗）
- $\omega$ 是角频率

实现代码：

```python
import numpy as np
import librosa

def manual_stft(y, n_fft=2048, hop_length=512, window='hann'):
    """
    手动实现STFT
    
    参数:
        y: 时域信号 [N]
        n_fft: FFT点数
        hop_length: 帧移
        window: 窗函数类型
        
    返回:
        X: 频谱 [1 + n_fft//2, num_frames]
    """
    # 创建窗函数
    if window == 'hann':
        win = np.hanning(n_fft)
    else:
        win = np.hanning(n_fft)
    
    # 计算帧数
    num_frames = 1 + (len(y) - n_fft) // hop_length
    
    # 分帧
    frames = []
    for i in range(num_frames):
        start = i * hop_length
        frame = y[start:start + n_fft]
        
        # 补零如果不够
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        
        # 乘窗函数
        frame = frame * win
        frames.append(frame)
    
    # 对每帧进行FFT
    X = np.fft.rfft(frames, axis=1)
    
    return X


def librosa_stft_example():
    """librosa STFT示例"""
    # 加载音频
    y, sr = librosa.load('audio.wav', sr=None)
    
    # 使用librosa计算STFT
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    
    # D的形状: [1 + n_fft//2, num_frames]
    print(f"STFT shape: {D.shape}")
    
    return D, sr
```

### 2.2 梅尔频谱图（Mel Spectrogram）

梅尔频谱图是将频率转换为梅尔刻度的频谱表示，更符合人耳对频率的感知。梅尔频率与Hz频率的关系：

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

$$f = 700 \cdot (10^{m/2595} - 1)$$

实现代码：

```python
def hz_to_mel(frequencies, htk=False):
    """
    将Hz频率转换为梅尔频率
    
    参数:
        frequencies: Hz频率
        htk: 是否使用HTK公式
        
    返回:
        mel: 梅尔频率
    """
    if htk:
        return 2595 * np.log10(1 + frequencies / 700)
    else:
        return 2595 * (1 + frequencies / 700).apply(lambda x: np.log(x) if x > 0 else 0)


def mel_to_hz(mel, htk=False):
    """
    将梅尔频率转换为Hz频率
    
    参数:
        mel: 梅尔频率
        htk: 是否使用HTK公式
        
    返回:
        frequencies: Hz频率
    """
    if htk:
        return 700 * (10 ** (mel / 2595) - 1)
    else:
        return (10 ** (mel / 2595) - 1) * 700


def create_mel_filterbank(
    n_freqs: int,
    n_mels: int,
    fmin: float = 0,
    fmax: float = None,
    htk: bool = False
):
    """
    创建梅尔滤波器组
    
    参数:
        n_freqs: 频率点数
        n_mels: 梅尔带数
        fmin: 最小频率
        fmax: 最大频率
        htk: 是否使用HTK公式
        
    返回:
        mel_basis: 梅尔滤波器组 [n_mels, n_freqs]
    """
    # 计算频率轴
    freqs = np.linspace(fmin, fmax, n_freqs)
    
    # 转换为梅尔频率
    mel = hz_to_mel(freqs, htk=htk)
    mel_min = hz_to_mel(fmin, htk=htk)
    mel_max = hz_to_mel(fmax, htk=htk)
    
    # 创建梅尔点
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points, htk=htk)
    
    # 创建滤波器
    fdiff = np.diff(hz_points)
    ramps = hz_points.reshape(-1, 1) - freqs.reshape(1, -1)
    
    # 下三角形
    lower = -ramps[:-1] / fdiff[:-1].reshape(-1, 1)
    upper = ramps[1:] / fdiff[1:].reshape(-1, 1)
    
    mel_basis = np.maximum(0, np.minimum(lower, upper))
    
    # 归一化
    mel_basis = mel_basis / mel_basis.sum(axis=1, keepdims=True)
    
    return mel_basis
```

### 2.3 梅尔频率倒谱系数（MFCC）

MFCC是语音识别中最常用的特征，它模拟了人耳的听觉感知特性。MFCC的计算流程：

```
原始信号 → 预加重 → 分帧 → 加窗 → FFT → 梅尔滤波器组 → 对数 → DCT → MFCC
```

```python
import scipy.fftpack

def compute_mfcc(
    y: np.ndarray,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    n_mfcc: int = 40,
    dct_type: int = 2
) -> np.ndarray:
    """
    计算MFCC特征
    
    参数:
        y: 时域信号
        sr: 采样率
        n_fft: FFT点数
        hop_length: 帧移
        n_mels: 梅尔带数
        n_mfcc: MFCC维度
        dct_type: DCT类型
        
    返回:
        mfcc: MFCC特征 [n_mfcc, num_frames]
    """
    # 预加重
    y_filtered = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # STFT
    D = manual_stft(y_filtered, n_fft, hop_length)
    S = np.abs(D) ** 2
    
    # 梅尔滤波器组
    mel_basis = create_mel_filterbank(1 + n_fft // 2, n_mels, fmin=0, fmax=sr/2)
    mel_S = mel_basis @ S
    
    # 对数能量
    log_mel_S = np.log(mel_S + 1e-10)
    
    # DCT
    mfcc = scipy.fftpack.dct(log_mel_S, type=dct_type, axis=0)[:n_mfcc]
    
    return mfcc


def librosa_mfcc_example():
    """librosa MFCC示例"""
    # 加载音频
    y, sr = librosa.load('audio.wav', sr=22050)
    
    # 计算MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # mfcc: [40, num_frames]
    
    # 计算MFCC差分
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta)
    
    print(f"MFCC shape: {mfcc.shape}")
    print(f"MFCC delta shape: {mfcc_delta.shape}")
    
    return mfcc, mfcc_delta, mfcc_delta2
```

## 3. 数学公式与推导

### 3.1 频谱分析基础

#### 离散傅里叶变换（DFT）

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j 2\pi kn/N}$$

逆变换：

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{j 2\pi kn/N}$$

#### 功率谱

$$P = \frac{1}{N} |X[k]|^2$$

### 3.2 频谱表示

#### 振幅谱

$$A = |X[k]|$$

#### 相位谱

$$\phi = \arg(X[k])$$

复数表示：

$$X[k] = A \cdot e^{j\phi}$$

```python
def spectral_transforms_example():
    """频谱变换示例"""
    # 加载音频
    y, sr = librosa.load('audio.wav', sr=22050)
    
    # STFT: [freq, time]
    D = librosa.stft(y, n_fft=2048)
    
    # 振幅谱
    amplitude = np.abs(D)
    
    # 相位谱
    phase = np.angle(D)
    
    # 功率谱
    power = np.abs(D) ** 2
    
    # 对数功率谱
    power_db = librosa.amplitude_to_db(power, ref=np.max)
    
    return {
        'amplitude': amplitude,
        'phase': phase,
        'power': power,
        'power_db': power_db
    }
```

## 4. librosa 核心功能

### 4.1 音频加载和保存

```python
import librosa
import soundfile as sf

def audio_io_examples():
    """音频输入输出示例"""
    
    # 加载音频（默认采样率22050）
    y, sr = librosa.load('audio.wav', sr=22050)
    
    # 加载原始采样率
    y_native, sr_native = librosa.load('audio.wav', sr=None)
    
    # 加载单声道
    y_mono = librosa.load('audio.wav', mono=True)
    
    # 加载指定时间段
    y_segment, sr = librosa.load(
        'audio.wav',
        sr=22050,
        offset=2.0,  # 从2秒开始
        duration=5.0  # 持续5秒
    )
    
    # 保存音频
    sf.write('output.wav', y, sr)
    
    # 使用librosa保存
    librosa.output.write_wav('output.wav', y, sr)
    
    return y, sr


def audio_information(y, sr):
    """获取音频信息"""
    duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"采样率: {sr} Hz")
    print(f"时长: {duration:.2f} 秒")
    print(f"样本数: {len(y)}")
    print(f"通道数: {1 if len(y.shape) == 1 else y.shape[1]}")
```

### 4.2 特征提取

```python
def feature_extraction_examples():
    """特征提取示例"""
    
    y, sr = librosa.load('audio.wav', sr=22050)
    
    # ==================== 频域特征 ====================
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    print(f"MFCC shape: {mfcc.shape}")
    
    # 色度特征
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    print(f"Chroma shape: {chroma.shape}")
    
    # 频谱对比度
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    print(f"Spectral contrast shape: {spectral_contrast.shape}")
    
    # 频谱熵
    spectral_entropy = librosa.feature.spectral_flatness(y=y)
    print(f"Spectral flatness shape: {spectral_entropy.shape}")
    
    # ==================== 时域特征 ====================
    
    # 过零率
    zcr = librosa.feature.zero_crossing_rate(y)
    print(f"ZCR shape: {zcr.shape}")
    
    # 均方根能量
    rms = librosa.feature.rms(y=y)
    print(f"RMS shape: {rms.shape}")
    
    # ==================== 节奏特征 ====================
    
    # 节拍估计
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Tempo: {tempo}, Beats: {len(beats)}")
    
    # 开始时间
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"Onsets: {len(onset_times)}")
    
    return {
        'mfcc': mfcc,
        'chroma': chroma,
        'spectral_contrast': spectral_contrast,
        'zcr': zcr,
        'rms': rms,
        'tempo': tempo
    }
```

### 4.3 频谱图可视化

```python
import matplotlib.pyplot as plt
import librosa.display

def visualize_spectrograms():
    """可视化频谱图"""
    
    y, sr = librosa.load('audio.wav', sr=22050)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ==================== 短时傅里叶变换 ====================
    D = librosa.stft(y)
    ax = axes[0, 0]
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(D), ref=np.max),
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='hz',
        ax=ax
    )
    ax.set_title('STFT Magnitude')
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    
    # ==================== 梅尔频谱图 ====================
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    ax = axes[0, 1]
    librosa.display.specshow(
        librosa.power_to_db(mel_spec, ref=np.max),
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        ax=ax
    )
    ax.set_title('Mel Spectrogram')
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    
    # ==================== MFCC ====================
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    ax = axes[1, 0]
    librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='linear',
        ax=ax
    )
    ax.set_title('MFCC')
    plt.colorbar(ax.collections[0], ax=ax)
    
    # ==================== 色度图 ====================
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    ax = axes[1, 1]
    librosa.display.specshow(
        chroma,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='chroma',
        ax=ax
    )
    ax.set_title('Chroma')
    plt.colorbar(ax.collections[0], ax=ax)
    
    plt.tight_layout()
    plt.show()
```

## 5. 应用场景
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充librosa的应用场景相关内容]


---

## 6. 优缺点分析
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充librosa的优缺点分析相关内容]


---

## 7. 调库实现
2 =$ 歌曲）。 ● 情绪（ $0 1 =$ 中性， $0 2 =$ 平静， $0 3 =$ 快乐， $ 0 4 =$ 悲伤， $0 5 =$ 愤怒， $0 6 =$ 恐惧， $0 7 =$ 厌恶， $0 8 =$ 惊讶）。 ● 情绪强度（ $0 1 =$ 正常， $0 2 =$ 强烈）。注意：“中性”情绪没有强烈的强度。 ● 内容（ $0 1 =$ Kids are talking by the door， $0 2 =$ Dogs are sitting by the door）。 ● 重复（ $0 1 =$ 第一次重复， $0 2 =$ 第二次重复）。 ● 演员（01~24。奇数为男性，偶数为女性）。 通过对比，03-01-02-01-01-01-01.wav这个文件对应的信息如下： 纯音频（03）。 语音（01）。 平静（02）。 正常强度（01）。 语调“正常”(01)。 ● 第一次重复（01）。 ● 第一号男演员（01）。 另外，需要注意的是，在这个数据集中，音频的采样率为22050，这一点可以设定或采用在第5章介绍的librosa进行读取。 # 6.2.2 情绪数据集的读取 下面对情绪数据集进行读取。在读取数据之前需要注意，数据集中每个文件都存放在不同的文件夹中，而每个文件夹都包含若干不同的情绪文件。因此，在


---

## 8. 手工代码实现
2 =$ 歌曲）。 ● 情绪（ $0 1 =$ 中性， $0 2 =$ 平静， $0 3 =$ 快乐， $ 0 4 =$ 悲伤， $0 5 =$ 愤怒， $0 6 =$ 恐惧， $0 7 =$ 厌恶， $0 8 =$ 惊讶）。 ● 情绪强度（ $0 1 =$ 正常， $0 2 =$ 强烈）。注意：“中性”情绪没有强烈的强度。 ● 内容（ $0 1 =$ Kids are talking by the door， $0 2 =$ Dogs are sitting by the door）。 ● 重复（ $0 1 =$ 第一次重复， $0 2 =$ 第二次重复）。 ● 演员（01~24。奇数为男性，偶数为女性）。 通过对比，03-01-02-01-01-01-01.wav这个文件对应的信息如下： 纯音频（03）。 语音（01）。 平静（02）。 正常强度（01）。 语调“正常”(01)。 ● 第一次重复（01）。 ● 第一号男演员（01）。 另外，需要注意的是，在这个数据集中，音频的采样率为22050，这一点可以设定或采用在第5章介绍的librosa进行读取。 # 6.2.2 情绪数据集的读取 下面对情绪数据集进行读取。在读取数据之前需要注意，数据集中每个文件都存放在不同的文件夹中，而每个文件夹都包含若干不同的情绪文件。因此，在


---

## 9. 可视化与结果理解
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充librosa的可视化与结果理解相关内容]


---

## 10. 模型评估
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充librosa的模型评估相关内容]


---

## 11. 常见问题与易错点
2 =$ 歌曲）。 ● 情绪（ $0 1 =$ 中性， $0 2 =$ 平静， $0 3 =$ 快乐， $ 0 4 =$ 悲伤， $0 5 =$ 愤怒， $0 6 =$ 恐惧， $0 7 =$ 厌恶， $0 8 =$ 惊讶）。 ● 情绪强度（ $0 1 =$ 正常， $0 2 =$ 强烈）。注意：“中性”情绪没有强烈的强度。 ● 内容（ $0 1 =$ Kids are talking by the door， $0 2 =$ Dogs are sitting by the door）。 ● 重复（ $0 1 =$ 第一次重复， $0 2 =$ 第二次重复）。 ● 演员（01~24。奇数为男性，偶数为女性）。 通过对比，03-01-02-01-01-01-01.wav这个文件对应的信息如下： 纯音频（03）。 语音（01）。 平静（02）。 正常强度（01）。 语调“正常”(01)。 ● 第一次重复（01）。 ● 第一号男演员（01）。 另外，需要注意的是，在这个数据集中，音频的采样率为22050，这一点可以设定或采用在第5章介绍的librosa进行读取。 # 6.2.2 情绪数据集的读取 下面对情绪数据集进行读取。在读取数据之前需要注意，数据集中每个文件都存放在不同的文件夹中，而每个文件夹都包含若干不同的情绪文件。因此，在


---

## 12. 学习总结
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充librosa的学习总结相关内容]


---

## 13. 练习题与思考题
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充librosa的练习题与思考题相关内容]


---

## 14. 学习路径建议
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充librosa的学习路径建议相关内容]


---


## 4. 训练过程讲解
### 训练步骤
1. **数据加载**：Dataset + DataLoader 批处理
2. **前向传播**：数据通过网络计算输出
3. **损失计算**：对比预测与标签
4. **反向传播**：计算梯度
5. **参数更新**：优化器更新权重
6. **循环迭代**：重复直至收敛
