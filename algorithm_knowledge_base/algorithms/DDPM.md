# DDPM (去噪扩散概率模型) 学习文档

> 通过逐步添加噪声再逐步去噪的方式,学习数据分布并生成高质量样本的深度生成模型

---

## 1. 算法基础认知

### 一句话定义

DDPM（Denoising Diffusion Probabilistic Models）是一种基于马尔可夫链的去噪生成模型,通过先对数据逐步添加高斯噪声将其破坏为纯噪声,再训练一个神经网络逐步去除噪声来恢复原始数据,从而学习数据的生成分布。

### 直觉类比

DDPM 的核心思想可以用"往墨水里滴墨,再一步步把墨洗干净"来形象理解:

- **前向扩散（加噪）**:想象你有一张清晰的照片,现在往上面撒一层薄薄的灰尘,然后再撒一层,再撒一层......经过很多层灰尘之后,照片已经完全被灰尘覆盖,什么都看不清了。这就是前向扩散过程 -- 逐步添加噪声,最终将原始信息完全淹没。
- **反向去噪**:现在反过来,如果你学会了"除尘术" -- 能够准确地判断并去除每一层灰尘,那么你就能从满是灰尘的照片中一步步恢复出原始的清晰照片。这就是反向去噪过程 -- 训练一个神经网络学会逐步去除噪声。
- **生成新照片**:如果你想生成一张全新的照片,只需要先随机生成一张满是灰尘的纸（纯噪声）,然后用你的"除尘术"一步步去除灰尘,最终得到的就是一张全新的、看起来像真实照片的图像。

这个类比的精妙之处在于:除尘术并不需要知道原始照片长什么样,它只需要学会"如何从当前状态的灰尘照片中去除一层灰尘",最终自然就能得到一张合理的图像。这正是 DDPM 的工作原理 -- 神经网络不需要直接生成图像,只需要学会"去噪"。

### 历史背景

DDPM 由 Jonathan Ho、Ajay Jain 和 Pieter Abbeel 于 2020 年在其论文《Denoising Diffusion Probabilistic Models》中正式提出。这篇论文将扩散概率模型（Diffusion Probabilistic Models, 最早由 Jascha Sohl-Dickstein 等人于 2015 年提出）成功应用于图像生成领域,可以算作扩散模型应用于图像生成方面的开山之作。

DDPM 的发展脉络如下:

1. **2015 年**:Sohl-Dickstein 等人提出扩散概率模型（DPM）的理论框架,但受限于当时的计算资源和网络架构,生成质量并不理想。
2. **2020 年**:Ho 等人提出 DDPM,通过引入噪声预测（noise prediction）的简化训练目标和 U-Net 架构,大幅提升了生成质量,达到了接近当时最优 GAN 的水平。
3. **2021 年**:Dhariwal 和 Nichol 提出 ADM（Ablated Diffusion Model）,通过改进 U-Net 架构（多头注意力、多分辨率注意力）和引入分类器引导（classifier guidance）,扩散模型首次在图像合成指标上击败 GAN。
4. **2021 年**:Ho 和 Salimans 提出无分类器引导（classifier-free guidance）,进一步提升了条件生成的质量。
5. **2021 年**:Song 等人提出 DDIM（Denoising Diffusion Implicit Models）,将采样步数从 1000 步大幅减少到 50 步甚至更少。
6. **2022 年**:Rombach 等人提出潜扩散模型（Latent Diffusion Model, LDM）,Stable Diffusion 正是基于此工作,将扩散操作从像素空间搬到低维潜空间,大幅降低了计算成本。

### 算法定位

- 类型:无监督学习 --> 生成模型 --> 基于分数的生成模型（Score-based Generative Models）
- 输出:与真实数据同分布的生成样本（如图像、音频等）
- 模型类型:隐式生成模型（通过马尔可夫链定义数据分布）
- 与 VAE 的关系:DDPM 可以看作是对分层 VAE（Hierarchical VAE）的极限推广,当分层数 T 趋近无穷大且每步的编码器只是添加固定高斯噪声时,分层 VAE 就演变为扩散模型。

### 前置知识

- 神经网络基础:前馈神经网络、反向传播、卷积神经网络
- 概率论:高斯分布（正态分布）、贝叶斯定理、KL 散度、ELBO（证据下界）
- VAE 基础:变分推断、重参数化技巧、ELBO 推导（DDPM 的训练目标由 VAE 的 ELBO 演化而来）
- U-Net 架构:编码器-解码器结构、跳跃连接（DDPM 使用 U-Net 作为噪声预测网络）
- 优化方法:梯度下降、Adam 优化器

---

## 2. 核心原理

### 2.1 核心思想

DDPM 的核心思想是将数据的生成过程分解为两个相反的马尔可夫链:

1. **前向过程（Forward Process, 也叫扩散过程）**:定义一个固定的马尔可夫链 $q(x_{1:T}|x_0)$,逐步向数据 $x_0$ 添加高斯噪声,经过 $T$ 步后将其变为纯高斯噪声 $x_T$。这个过程不需要学习任何参数,是完全确定的。
2. **反向过程（Reverse Process, 也叫去噪过程）**:定义一个可学习的马尔可夫链 $p_\theta(x_{0:T})$,从纯高斯噪声 $x_T$ 出发,通过神经网络逐步去除噪声,最终恢复出原始数据 $x_0$。这个过程需要训练一个参数化的噪声预测模型。

核心思想可以概括为:与其直接让神经网络生成数据（这很难）,不如让它学会"如何去除噪声"（这相对容易）,然后从纯噪声出发,反复去噪,就能生成出逼真的数据。

### 2.2 工作流程

DDPM 的完整工作流程分为训练和采样（生成）两个阶段:

**训练阶段:**

1. **获取干净数据**:从训练数据集中随机采样一张干净的图像 $x_0$
2. **随机选择时间步**:从 $\{1, 2, \ldots, T\}$ 中均匀随机选择一个时间步 $t$
3. **采样噪声**:从标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 中采样一个噪声 $\boldsymbol{\epsilon}$
4. **一步加噪**:利用"一步到位"公式,直接从 $x_0$ 计算出时间步 $t$ 的噪声图像 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$
5. **噪声预测**:将 $x_t$ 和 $t$ 输入噪声预测网络 $\boldsymbol{\epsilon}_\theta(x_t, t)$（通常是一个 U-Net）,得到预测的噪声
6. **计算损失**:计算预测噪声与真实噪声之间的均方误差（MSE）$\mathcal{L} = \|\boldsymbol{\epsilon}_\theta(x_t, t) - \boldsymbol{\epsilon}\|^2$
7. **更新参数**:通过反向传播更新噪声预测网络的参数 $\theta$

**采样（生成）阶段:**

1. **初始化噪声**:从标准正态分布中采样一个纯噪声图像 $x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. **迭代去噪**:从 $t = T$ 到 $t = 1$,每一步:
   - 将当前 $x_t$ 和 $t$ 输入噪声预测网络,得到预测噪声 $\boldsymbol{\epsilon}_\theta(x_t, t)$
   - 利用去噪公式计算 $x_{t-1}$:
     $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(x_t, t) \right) + \sigma_t z$$
     其中 $z \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（当 $t = 1$ 时 $z = 0$）
3. **输出结果**:经过 $T$ 步去噪后,得到的 $x_0$ 即为生成的图像

### 2.3 关键概念解释

- **噪声调度（Noise Schedule / Beta Schedule）**:一组预定义的超参数 $\{\beta_1, \beta_2, \ldots, \beta_T\}$,控制每一步添加的噪声大小。$\beta_t$ 越大,该步添加的噪声越多。DDPM 原论文使用线性调度,从 $\beta_1 = 10^{-4}$ 线性增加到 $\beta_T = 0.02$。
- **$\alpha_t$ 和 $\bar{\alpha}_t$**:定义 $\alpha_t = 1 - \beta_t$（保留原始信息的比例）,$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$（从 $x_0$ 到 $x_t$ 累积保留信息的比例）。当 $t$ 足够大时,$\bar{\alpha}_t \approx 0$,意味着 $x_T$ 中几乎不包含 $x_0$ 的信息。
- **重参数化技巧（Reparameterization Trick）**:为了在反向传播中处理随机采样操作,将随机噪声作为输入的一部分。例如,$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$ 中,$\boldsymbol{\epsilon}$ 是一个从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 采样的随机变量,但它不依赖于 $\theta$,因此可以正常反向传播。
- **噪声预测（Noise Prediction）**:DDPM 的一个关键设计选择 -- 让神经网络预测添加到 $x_t$ 中的噪声 $\boldsymbol{\epsilon}$,而不是直接预测 $x_0$ 或 $x_{t-1}$。实践证明,预测噪声的效果最好,这可能是因为噪声比原始图像更容易建模。
- **U-Net 噪声预测网络**:DDPM 使用 U-Net 作为噪声预测网络 $\boldsymbol{\epsilon}_\theta(x_t, t)$。U-Net 的编码器-解码器结构和跳跃连接使其能够同时捕捉图像的全局结构和局部细节,非常适合图像重建任务。时间步 $t$ 通过正弦位置编码（Sinusoidal Positional Encoding）嵌入到网络中。

### 2.4 几何/直观解释

从几何角度来看,DDPM 的前向过程可以理解为:原始数据 $x_0$ 位于高维空间中的某个低维流形（manifold）上。每一步添加高斯噪声,就像是在流形上添加一个随机偏移,使数据点逐渐偏离流形。经过 $T$ 步后,数据点已经完全远离流形,变成了空间中一个各向同性的高斯分布。

反向过程则是在做相反的事情:从空间中随机选取一个点（高斯噪声 $x_T$）,然后通过神经网络学习到的"去噪方向",逐步将这个点"拉回"到数据流形上。这个"去噪方向"实际上就是数据分布的分数函数（score function）$\nabla_{x_t} \log p(x_t)$ 的近似。

### 2.5 与其他生成模型的对比直觉

- **GAN**:直接让生成器"从零开始画一幅画",生成器需要同时学会全局构图和局部细节,训练不稳定。DDPM 则是"从噪声中逐步还原画面",每一步只需要学习一小步修正,训练更稳定。
- **VAE**:通过编码器将数据压缩到低维潜空间,再从潜空间解码。如果潜空间维度太低,会丢失信息导致图像模糊;如果维度太高,又难以训练。DDPM 不需要压缩到低维空间,直接在原始像素空间操作,理论上能保留所有信息。
- **Flow-based Models（如 Glow）**:通过可逆变换精确计算似然,但网络架构受到严格的可逆性约束。DDPM 不需要可逆变换,网络架构更灵活。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 维度/说明 |
|------|------|----------|
| $x_0$ | 原始（干净）数据 | 与数据维度相同,如 $(C, H, W)$ |
| $x_t$ | 时间步 $t$ 的噪声数据 | 与 $x_0$ 同维度 |
| $T$ | 总时间步数 | 标量,DDPM 原论文中 $T = 1000$ |
| $\beta_t$ | 时间步 $t$ 的噪声方差 | 标量,控制该步添加噪声的强度 |
| $\alpha_t$ | 信息保留系数 | $\alpha_t = 1 - \beta_t$ |
| $\bar{\alpha}_t$ | 累积信息保留系数 | $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ |
| $\boldsymbol{\epsilon}$ | 标准高斯噪声 | $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| $\boldsymbol{\epsilon}_\theta$ | 噪声预测网络 | 参数为 $\theta$ 的神经网络 |
| $\mu_\theta$ | 均值预测网络 | 参数为 $\theta$ 的神经网络 |
| $\sigma_t$ | 时间步 $t$ 的噪声标准差 | 标量,预定义或可学习 |
| $\theta$ | 神经网络参数 | 噪声预测网络的全部参数 |
| $\mathcal{L}$ | 损失函数 | 标量 |

### 3.2 前向扩散过程

#### 3.2.1 单步扩散公式

前向扩散过程的每一步定义为:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t) \mathbf{I})$$

其中 $\alpha_t = 1 - \beta_t$。使用重参数化技巧,单步采样可以表示为:

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}, \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**为什么需要乘以 $\sqrt{\alpha_t}$ 而不是直接加噪声?** 如果不加缩小因子,即 $x_t = x_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}$,那么每步都会增加方差,经过 $T$ 步后方差会非常大。乘以 $\sqrt{\alpha_t}$ 确保每步的数据保持在合理的方差范围内。

#### 3.2.2 一步到位公式（重要性质）

这是 DDPM 中最重要的数学性质之一。利用高斯噪声的可加性（两个独立高斯随机变量之和仍服从高斯分布,且均值相加、方差相加）,可以推导出从 $x_0$ 直接到 $x_t$ 的闭式表达。

**推导过程:**

从单步公式出发:

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$$

将 $x_{t-1}$ 进一步展开:

$$x_{t-1} = \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \boldsymbol{\epsilon}_{t-2}$$

代入得:

$$x_t = \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \boldsymbol{\epsilon}_{t-2} \right) + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$$

$$= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$$

由于 $\boldsymbol{\epsilon}_{t-1}$ 和 $\boldsymbol{\epsilon}_{t-2}$ 都是从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 独立采样的,利用高斯噪声可加性:

$$\sqrt{\alpha_t (1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, (\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t))\mathbf{I})$$

$$= \mathcal{N}(\mathbf{0}, (1 - \alpha_t \alpha_{t-1})\mathbf{I})$$

因此:

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \boldsymbol{\epsilon}$$

其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

重复此过程 $t$ 次,最终得到:

$$\boxed{x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}$$

用概率分布表示:

$$\boxed{q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})}$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$。

**这个公式的意义非常重大**:它使得我们可以从 $x_0$ 一步到位地计算出任意时间步 $t$ 的噪声图像 $x_t$,而不需要逐步计算 $x_1, x_2, \ldots, x_{t-1}$。这大大加速了训练过程,因为我们只需要一次前向传播就能得到训练所需的 $x_t$。

**为什么最终 $x_T$ 服从标准高斯分布?** 当 $T$ 足够大且 $\{\beta_1, \ldots, \beta_T\}$ 合理设置（如线性从 $10^{-4}$ 到 $0.02$）时,$\bar{\alpha}_T \approx 0$,因此:

$$q(x_T | x_0) = \mathcal{N}(x_T; \sqrt{\bar{\alpha}_T} x_0, (1 - \bar{\alpha}_T) \mathbf{I}) \approx \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$$

这意味着 $x_T$ 已经不包含任何关于 $x_0$ 的信息,变成了纯高斯噪声。

### 3.3 逆高斯分布真实后验

在反向过程中,我们需要估计 $p_\theta(x_{t-1} | x_t)$。一个自然的问题是:如果我们已经知道了 $x_0$ 和 $x_t$,能否精确计算出 $x_{t-1}$ 的分布?答案是肯定的,这就是逆高斯分布真实后验 $q(x_{t-1} | x_t, x_0)$。

#### 3.3.1 贝叶斯展开

根据贝叶斯定理:

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

由马尔可夫性,$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$。代入前向扩散过程的三个高斯分布:

$$q(x_{t-1} | x_t, x_0) = \frac{\mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)\mathbf{I}) \cdot \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1})\mathbf{I})}{\mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I})}$$

#### 3.3.2 重点分析指数部分

将三个高斯分布写成指数形式,重点分析关于 $x_{t-1}$ 的二次项。高斯分布 $\mathcal{N}(x; \mu, \sigma^2 \mathbf{I})$ 的概率密度正比于:

$$\propto \exp\left(-\frac{1}{2\sigma^2} \|x - \mu\|^2\right)$$

因此,分子中第一个高斯分布（关于 $x_t$,展开 $\|x_t - \sqrt{\alpha_t} x_{t-1}\|^2$）关于 $x_{t-1}$ 的指数部分为:

$$\propto \exp\left(-\frac{\alpha_t \|x_{t-1}\|^2 - 2\sqrt{\alpha_t} x_t \cdot x_{t-1}}{2(1 - \alpha_t)}\right)$$

分子中第二个高斯分布（关于 $x_{t-1}$）的指数部分为:

$$\propto \exp\left(-\frac{\|x_{t-1}\|^2 - 2\sqrt{\bar{\alpha}_{t-1}} x_0 \cdot x_{t-1}}{2(1 - \bar{\alpha}_{t-1})}\right)$$

分母的指数部分不依赖于 $x_{t-1}$,记为常数 $C(x_t, x_0)$。

合并分子中关于 $x_{t-1}$ 的二次项和一次项:

$$\propto \exp\left(-\frac{1}{2}\left[\left(\frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}\right) \|x_{t-1}\|^2 - 2\left(\frac{\sqrt{\alpha_t}}{1 - \alpha_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0\right) \cdot x_{t-1} + C(x_t, x_0)\right]\right)$$

配成完全平方形式 $\exp\left(-\frac{1}{2\sigma_q^2(t)} \|x_{t-1} - \mu_q(x_t, x_0)\|^2\right)$,比较系数可得:

#### 3.3.3 方差 $\sigma_q^2(t)$

$$\frac{1}{\sigma_q^2(t)} = \frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} = \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t)}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} = \frac{1 - \alpha_t \bar{\alpha}_{t-1}}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}$$

由于 $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$,所以 $1 - \alpha_t \bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$,因此:

$$\boxed{\sigma_q^2(t) = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}}$$

注意:这个方差只依赖于超参数 $\{\alpha_t\}$,不需要学习,是一个预定义的常数。

#### 3.3.4 均值 $\mu_q(x_t, x_0)$

$$\mu_q(x_t, x_0) = \frac{\sigma_q^2(t)}{1} \left(\frac{\sqrt{\alpha_t}}{1 - \alpha_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0\right)$$

$$= \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \cdot \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t) x_0}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}$$

$$\boxed{\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t) x_0}{1 - \bar{\alpha}_t}}$$

**直觉理解**:均值 $\mu_q(x_t, x_0)$ 是 $x_t$ 和 $x_0$ 的加权平均（"内分点"）。权重由 $\alpha_t$ 和 $\bar{\alpha}_t$ 决定:当噪声水平较高（$t$ 较大,$\bar{\alpha}_t$ 较小）时,$x_t$ 的权重更大;$x_0$ 的权重较小;当噪声水平较低时,$x_0$ 的权重更大。这符合直觉 -- 当噪声很多时,我们更依赖当前的噪声图像 $x_t$;当噪声较少时,我们更依赖原始的干净图像 $x_0$。

#### 3.3.5 完整的后验分布

$$\boxed{q(x_{t-1} | x_t, x_0) = \mathcal{N}\left(x_{t-1}; \mu_q(x_t, x_0), \sigma_q^2(t) \mathbf{I}\right)}$$

### 3.4 ELBO 目标函数推导

DDPM 的训练目标推导与 VAE 类似,基于最大化数据的对数似然 $\log p_\theta(x_0)$ 的下界（ELBO）。

#### 3.4.1 对数似然的 ELBO

根据概率的边际化:

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) dx_{1:T}$$

引入前向分布 $q(x_{1:T} | x_0)$（不依赖于 $\theta$）,乘以 $\frac{q}{q} = 1$:

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) \frac{q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)} dx_{1:T}$$

$$= \log \mathbb{E}_{q(x_{1:T}|x_0)}\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

利用 Jensen 不等式（$\log$ 是凹函数,$\mathbb{E}[\log(\cdot)] \leq \log \mathbb{E}[\cdot]$）:

$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = \text{ELBO}(x_0; \theta)$$

#### 3.4.2 ELBO 的展开

利用马尔可夫性展开分子和分母:

**分子**（联合分布 $p_\theta(x_{0:T})$）:

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)$$

其中 $p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$ 是标准高斯分布。

**分母**（前向分布 $q(x_{1:T}|x_0)$）:

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})$$

**ELBO 展开**:

$$\text{ELBO}(x_0; \theta) = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T} q(x_t|x_{t-1})}\right]$$

$$= \mathbb{E}_{q}\left[\log p(x_T) + \sum_{t=1}^{T} \log p_\theta(x_{t-1}|x_t) - \sum_{t=1}^{T} \log q(x_t|x_{t-1})\right]$$

将不依赖于 $\theta$ 的项移到一起:

$$= \mathbb{E}_{q}\left[\sum_{t=1}^{T} \log p_\theta(x_{t-1}|x_t)\right] + \mathbb{E}_{q}\left[\log p(x_T) - \sum_{t=1}^{T} \log q(x_t|x_{t-1})\right]$$

由于我们的目标是最大化关于 $\theta$ 的 ELBO,因此可以忽略不包含 $\theta$ 的第二项。目标函数为:

$$J(\theta) = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\sum_{t=1}^{T} \log p_\theta(x_{t-1}|x_t)\right]$$

#### 3.4.3 进一步简化:从 T 步到 1 步

利用期望的线性性和条件边缘化:

$$J(\theta) = \sum_{t=1}^{T} \mathbb{E}_{q(x_{t-1}, x_t|x_0)}\left[\log p_\theta(x_{t-1}|x_t)\right]$$

$$= T \cdot \mathbb{E}_{t \sim U\{1,T\}}\left[\mathbb{E}_{q(x_{t-1}, x_t|x_0)}\left[\log p_\theta(x_{t-1}|x_t)\right]\right]$$

这里将 $T$ 个时间步的求和转化为均匀分布 $U\{1, T\}$ 下的期望。

引入真实后验 $q(x_{t-1}|x_t, x_0)$（不依赖于 $\theta$）,减去它（减去常数不影响优化方向）:

$$\arg\max_\theta J(\theta) = \arg\max_\theta \mathbb{E}_{q(x_t|x_0)}\left[D_{\text{KL}}\left(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)\right)\right]$$

也就是最小化 KL 散度:

$$\min_\theta \mathbb{E}_{q(x_t|x_0)}\left[D_{\text{KL}}\left(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)\right)\right]$$

#### 3.4.4 KL 散度的计算

两个同方差高斯分布的 KL 散度有闭式解。设:

$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \mu_q(x_t,x_0), \sigma_q^2(t)\mathbf{I})$$

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \sigma_q^2(t)\mathbf{I})$$

注意:DDPM 选择让 $p_\theta$ 的方差与真实后验 $q$ 的方差相同,即 $\sigma_q^2(t)\mathbf{I}$（不学习方差）。

同方差高斯分布的 KL 散度为:

$$D_{\text{KL}}(q \| p_\theta) = \frac{1}{2\sigma_q^2(t)} \|\mu_\theta(x_t,t) - \mu_q(x_t,x_0)\|^2$$

因此,完整的损失函数为:

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \boldsymbol{\epsilon}}\left[\frac{1}{2\sigma_q^2(t)} \|\mu_\theta(x_t,t) - \mu_q(x_t,x_0)\|^2\right]$$

### 3.5 噪声预测参数化简化

这是 DDPM 论文中一个关键的简化步骤。上一节的损失函数中,网络需要预测均值 $\mu_\theta(x_t,t)$ 来逼近 $\mu_q(x_t, x_0)$。我们可以通过变量替换,将预测目标从"均值"变为"噪声",从而简化计算。

#### 3.5.1 将 $\mu_q$ 用噪声 $\boldsymbol{\epsilon}$ 表示

从一步到位公式 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$,可以解出:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

将此式代入 $\mu_q(x_t, x_0)$:

$$\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t) \cdot \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}}{1 - \bar{\alpha}_t}$$

合并 $x_t$ 项:

$$= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) + \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{\sqrt{\bar{\alpha}_t}}}{1 - \bar{\alpha}_t} x_t - \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t) \sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_t)} \boldsymbol{\epsilon}$$

经过化简（利用 $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$,即 $\sqrt{\bar{\alpha}_{t-1}} = \sqrt{\bar{\alpha}_t / \alpha_t}$）:

$$\boxed{\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)}$$

#### 3.5.2 参数化 $\mu_\theta$

用同样的形式参数化 $\mu_\theta$,但将 $\boldsymbol{\epsilon}$ 替换为神经网络的输出 $\boldsymbol{\epsilon}_\theta(x_t, t)$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(x_t, t) \right)$$

#### 3.5.3 损失函数简化

将 $\mu_\theta$ 和 $\mu_q$ 代入 KL 散度:

$$\|\mu_\theta(x_t, t) - \mu_q(x_t, x_0)\|^2 = \left\| \frac{1 - \alpha_t}{\sqrt{\alpha_t} \sqrt{1 - \bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(x_t, t) - \boldsymbol{\epsilon}) \right\|^2$$

$$= \frac{(1 - \alpha_t)^2}{\alpha_t (1 - \bar{\alpha}_t)} \|\boldsymbol{\epsilon}_\theta(x_t, t) - \boldsymbol{\epsilon}\|^2$$

DDPM 论文进一步简化了损失函数。原论文中的最终损失为:

$$\boxed{\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t \sim U\{1,T\}, x_0 \sim q(x_0), \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})}\left[\|\boldsymbol{\epsilon}_\theta(x_t, t) - \boldsymbol{\epsilon}\|^2\right]}$$

其中 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$。

**为什么可以省略前面的系数?** 因为前面的系数 $\frac{(1-\alpha_t)^2}{\alpha_t(1-\bar{\alpha}_t)}$ 不依赖于 $\theta$,相当于对不同的时间步赋予了不同的权重。DDPM 论文通过实验发现,使用简化版（等权重）的 MSE 损失效果反而更好,可能是因为等权重使得每个时间步对训练的贡献更加均衡。

#### 3.5.4 采样（生成）公式

训练完成后,采样过程使用噪声预测网络 $\boldsymbol{\epsilon}_\theta$ 进行反向去噪:

$$\boxed{x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(x_t, t) \right) + \sigma_t z}$$

其中:
- 当 $t > 1$ 时,$z \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（添加随机性以保证生成多样性）
- 当 $t = 1$ 时,$z = 0$（不添加噪声,直接输出最终结果）
- $\sigma_t = \sqrt{\sigma_q^2(t)} = \sqrt{\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}}$（DDPM 使用逆过程方差）

### 3.6 方差调度（Beta Schedule）

方差调度 $\{\beta_1, \beta_2, \ldots, \beta_T\}$ 对模型的性能至关重要。

#### 3.6.1 线性调度（DDPM 原论文）

$$\beta_t = \beta_{\min} + \frac{t - 1}{T - 1} (\beta_{\max} - \beta_{\min})$$

DDPM 原论文使用 $\beta_{\min} = 10^{-4}$, $\beta_{\max} = 0.02$, $T = 1000$。

#### 3.6.2 余弦调度（Improved DDPM）

Nichol 和 Dhariwal（2021）提出了余弦调度,在训练初期和末期变化更平缓:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s$ 是一个小的偏移量（原文取 $s = 0.008$）,用于防止 $\beta_t$ 在训练初期过大。余弦调度的优点是能够避免线性调度在前端变化过快、后端变化过慢的问题。

#### 3.6.3 方差对 $\bar{\alpha}_T$ 的影响

| 调度方式 | $\beta_{\min}$ | $\beta_{\max}$ | $\bar{\alpha}_{1000}$ | 效果 |
|----------|---------------|---------------|---------------------|------|
| 线性 | 0.0001 | 0.02 | ~0.0001 | 基准 |
| 余弦 | N/A | N/A | ~0.00001 | 更好 |

$\bar{\alpha}_T$ 越小,意味着 $x_T$ 越接近标准高斯分布,前向过程的信息损失越彻底,反向过程的重建潜力越大。

---

## 4. 训练过程讲解

### 4.1 数据预处理

DDPM 的数据预处理较为简单:

1. **图像归一化**:将像素值从 $[0, 255]$ 归一化到 $[-1, 1]$,这是因为:
   - 前向过程假设数据服从零均值高斯分布附近
   - 添加的噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 均值为零
   - 如果数据范围是 $[0, 1]$,加噪后数据会有正偏移

2. **数据增强**:可以使用常见的图像增强方法,如随机裁剪、水平翻转等。DDPM 原论文使用了随机水平翻转。

3. **数据集**:DDPM 原论文在 CIFAR-10、LSUN、CelebA-HQ 等数据集上进行了验证。

### 4.2 参数初始化

噪声预测网络（U-Net）的初始化遵循标准深度学习实践:

- **卷积层**:使用 Kaiming 初始化（He 初始化）
- **BatchNorm / GroupNorm**:权重初始化为 1,偏置初始化为 0
- **线性层**:使用 Xavier 初始化
- **时间嵌入**:正弦位置编码不需要学习参数,后续的线性映射层使用标准初始化

### 4.3 迭代过程（伪代码）

DDPM 的训练算法（噪声预测版本）如下:

```
算法: DDPM 训练
重复执行:
    1. 从训练数据中随机采样 x_0
    2. 从 {1, 2, ..., T} 中均匀随机采样时间步 t
    3. 从标准正态分布采样噪声 epsilon ~ N(0, I)
    4. 计算噪声图像: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    5. 前向传播: 将 (x_t, t) 输入噪声预测网络,得到 epsilon_theta(x_t, t)
    6. 计算损失: L = ||epsilon_theta(x_t, t) - epsilon||^2
    7. 反向传播并更新参数 theta
```

DDPM 的采样算法如下:

```
算法: DDPM 采样（生成）
1. 从标准正态分布采样纯噪声: x_T ~ N(0, I)
2. for t = T, T-1, ..., 1:
    3. 采样噪声 z ~ N(0, I)
    4. if t == 1: z = 0
    5. 计算标准差: sigma_t = sqrt((1 - alpha_t)(1 - alpha_bar_{t-1}) / (1 - alpha_bar_t))
    6. 去噪: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - ((1-alpha_t)/sqrt(1-alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z
7. 返回 x_0
```

### 4.4 收敛条件

DDPM 的收敛判断不像 GAN 那样困难（GAN 中判别器和生成器的平衡很难判断）。DDPM 的收敛标志:

1. **训练损失持续下降**:MSE 损失应该随着训练进行持续下降并趋于稳定
2. **生成样本质量提升**:定期从纯噪声采样,观察生成图像的质量是否逐渐提高
3. **FID 分数下降**:使用 Fréchet Inception Distance 定量评估生成质量
4. **训练损失的绝对值**:一般来说,MSE 损失降到 $0.01 \sim 0.1$ 之间（取决于数据集）

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | DDPM 原论文默认值 |
|--------|------|----------|-----------------|
| $T$ (总时间步) | 扩散步数,越大生成质量越好但采样越慢 | 500 ~ 2000 | 1000 |
| $\beta_{\min}$ | 最小噪声方差 | $10^{-5}$ ~ $10^{-3}$ | $10^{-4}$ |
| $\beta_{\max}$ | 最大噪声方差 | 0.01 ~ 0.05 | 0.02 |
| 学习率 | Adam 优化器的学习率 | $10^{-5}$ ~ $2 \times 10^{-4}$ | $2 \times 10^{-4}$ |
| Batch Size | 每次训练的样本数 | 64 ~ 256 | 128 |
| EMA 衰减率 | 指数移动平均的衰减率 | 0.999 ~ 0.9999 | 0.9999 |
| 训练轮数 | 完整遍历数据集的次数 | 100 ~ 2000 (取决于数据集) | 800K iterations (CIFAR-10) |

---

## 5. 应用场景

### 5.1 图像生成

DDPM 最主要的应用是图像生成。通过从纯高斯噪声出发,经过 $T$ 步反向去噪,可以生成高质量的图像。

- **人脸生成**:在 CelebA-HQ、FFHQ 等人脸数据集上,DDPM 能够生成高保真度的人脸图像,细节丰富且多样性好。
- **场景生成**:在 LSUN 卧室、教堂等场景数据集上,DDPM 生成的图像在结构合理性和细节丰富度上表现出色。
- **无条件生成**:标准 DDPM 不需要任何条件信息,可以直接生成与训练数据分布一致的随机图像。

### 5.2 图像编辑与修复

基于扩散模型的图像编辑是近年来非常热门的研究方向:

- **图像修复（Inpainting）**:将图像的某个区域遮盖（替换为噪声）,然后用扩散模型重新生成该区域的内容。例如,可以去除照片中的多余物体或修复破损的照片。
- **图像补全（Outpainting）**:在图像的边界外扩展生成新的内容,可以用于扩展画布、创建全景图等。
- **图像到图像翻译**:在扩散过程中保留部分信息,修改其他部分,可以实现风格迁移、语义编辑等功能。

### 5.3 文本到图像生成

通过引入条件机制（如分类器引导或无分类器引导）,扩散模型可以实现文本到图像的生成:

- **DALL-E 2**:OpenAI 基于 CLIP 和扩散模型构建的文生图模型,将扩散模型从无条件生成升级为条件生成。
- **Stable Diffusion**:基于潜扩散模型（LDM）的文生图系统,是目前最流行的开源文生图工具之一。它在低维潜空间中进行扩散,大幅降低了计算成本。
- **Imagen**:Google 提出的文生图模型,使用级联扩散模型实现高分辨率图像生成。

### 5.4 音频和视频生成

扩散模型不限于图像领域,还可以应用于:

- **音频生成**:AudioLDM、AudioDiffusion 等模型将扩散过程应用于音频频谱图,实现语音合成、音乐生成等功能。
- **视频生成**:通过在时间维度上扩展扩散过程,可以实现视频的生成和预测。例如,Video Diffusion Models 通过在 3D（空间 + 时间）上添加噪声来实现视频生成。

### 5.5 科学计算

- **蛋白质结构生成**:DiffusionFold 等工作将扩散模型应用于蛋白质 3D 结构的生成和预测。
- **分子生成**:在药物发现中,扩散模型可以用于生成具有特定性质的分子结构。

---

## 6. 优缺点分析

### 6.1 优点

1. **生成质量高**:在多个基准数据集上,DDPM 及其改进版本已经超越 GAN,成为图像生成质量的新标准。
2. **训练稳定**:不像 GAN 那样需要精心平衡判别器和生成器的训练,DDPM 的训练目标简单明确（MSE 损失）,几乎不会出现模式崩溃。
3. **理论基础扎实**:DDPM 有严格的数学推导（ELBO、变分推断、贝叶斯定理）,训练目标有明确的概率解释。
4. **生成多样性好**:由于采样过程中每步都添加随机噪声,DDPM 生成的样本多样性优于 GAN（GAN 容易陷入模式崩溃,只生成少数几种样本）。
5. **架构灵活**:可以使用各种神经网络架构（U-Net、Transformer 等）作为噪声预测网络。

### 6.2 缺点

1. **采样速度慢**:标准 DDPM 需要 $T = 1000$ 步才能生成一张图像,而 GAN 只需一步前向传播。即使使用 DDIM 等加速方法,也需要几十到上百步。
2. **计算成本高**:训练和推理的计算量都很大。每张图像的生成需要 $T$ 次神经网络前向传播。
3. **图像分辨率受限**:在像素空间直接操作时,高分辨率图像的计算开销呈平方级增长（$H \times W$）。这是 LDM/Stable Diffusion 将操作搬到潜空间的原因。
4. **需要大量训练数据**:与 GAN 类似,在数据量较少时效果可能不理想。

### 6.3 与其他生成模型的对比

| 维度 | DDPM | GAN | VAE | Flow (如 Glow) |
|------|------|-----|-----|----------------|
| 生成质量 | 高（SOTA） | 高 | 中等 | 中等 |
| 训练稳定性 | 高（MSE损失） | 低（对抗训练） | 高 | 高 |
| 采样速度 | 慢（1000步） | 快（1步） | 快（1步） | 快（1步） |
| 生成多样性 | 高 | 低（易模式崩溃） | 中等 | 高 |
| 精确似然 | 否 | 否 | 下界（ELBO） | 是 |
| 可控生成 | 需要额外引导 | 可通过条件GAN | 通过潜空间操作 | 通过潜空间操作 |
| 数学基础 | 变分推断 + ELBO | 博弈论 | 变分推断 | 可逆变换理论 |
| 代表应用 | Stable Diffusion, DALL-E 2 | StyleGAN | 图像隐空间操作 | 图像压缩 |

---

## 7. 调库实现（PyTorch 简化版 DDPM）

下面是一个使用 PyTorch 实现的简化版 DDPM,在 MNIST 数据集上进行训练和生成。该实现包含完整的噪声调度、U-Net 噪声预测网络、训练循环和采样过程。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# ========================================
# 1. 噪声调度（Beta Schedule）模块
# ========================================

class GaussianDiffusionSchedule:
    """
    高斯扩散过程的噪声调度管理器
    负责计算和管理 alpha_t, alpha_bar_t 等关键参数
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        """
        参数:
            num_timesteps: 扩散总步数 T
            beta_start: 噪声方差的最小值
            beta_end: 噪声方差的最大值
            schedule_type: 调度类型, 'linear' 或 'cosine'
        """
        self.num_timesteps = num_timesteps

        if schedule_type == 'linear':
            # 线性调度: beta 从 beta_start 线性增长到 beta_end
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            # 余弦调度: 在训练初期和末期变化更平缓
            steps = num_timesteps + 1
            s = 0.008  # 偏移量,防止 beta 在初期过大
            t = torch.linspace(0, num_timesteps, steps) / num_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"不支持的调度类型: {schedule_type}")

        # 计算 alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas
        # 计算 alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t (累积乘积)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 计算 alpha_bar_{t-1},用于反向过程
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # 计算反向过程的标准差
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # 计算 sqrt(alpha_bar_t) 和 sqrt(1 - alpha_bar_t)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        """
        从参数张量 a 中提取时间步 t 对应的值,并调整为广播形状
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0, t, noise=None):
        """
        前向扩散: 根据一步到位公式从 x_0 直接计算 x_t
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        参数:
            x_0: 原始数据, shape (B, C, H, W)
            t: 时间步, shape (B,)
            noise: 可选,如果不提供则随机采样
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        计算真实后验 q(x_{t-1} | x_t, x_0) 的均值和方差

        均值: mu = (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) * x_t
                     + sqrt(alpha_bar_{t-1}) * (1 - alpha_t) * x_0) / (1 - alpha_bar_t)
        方差: var = (1 - alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        """
        # 提取当前步和上一步的 alpha_bar
        alpha_bar_t = self.extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_t_prev = self.extract(self.alphas_cumprod_prev, t, x_t.shape)
        alpha_t = self.extract(self.alphas, t, x_t.shape)

        # 计算后验均值
        posterior_mean = (
            torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) * x_t
            + torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) * x_0
        ) / (1 - alpha_bar_t)

        # 计算后验方差
        posterior_variance = (1 - alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

        return posterior_mean, posterior_variance


# ========================================
# 2. 正弦位置编码（Sinusoidal Embedding）
# ========================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    将时间步 t（整数）编码为向量
    使用与 Transformer 中相同的正弦位置编码方案
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        参数:
            time: 时间步张量, shape (B,)
        返回:
            嵌入向量, shape (B, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        # 频率指数
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 将时间步乘以频率
        embeddings = time[:, None].float() * embeddings[None, :]
        # 拼接 sin 和 cos
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


# ========================================
# 3. 简化的残差块
# ========================================

class Block(nn.Module):
    """
    包含 GroupNorm + Conv + SiLU 激活 + 时间嵌入的残差块
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        # 时间嵌入的线性投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        # 如果输入输出通道数不同,使用 1x1 卷积调整
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        """
        参数:
            x: 特征图, shape (B, C, H, W)
            t_emb: 时间嵌入向量, shape (B, time_emb_dim)
        """
        h = self.conv1(x)
        # 将时间嵌入加到特征图上
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# ========================================
# 4. 简化的 U-Net 噪声预测网络
# ========================================

class SimpleUNet(nn.Module):
    """
    简化版 U-Net,用于预测噪声 epsilon_theta(x_t, t)
    架构: 编码器(下采样) -> 瓶颈层 -> 解码器(上采样) + 跳跃连接
    """
    def __init__(self, in_channels=1, model_channels=64, out_channels=1, time_emb_dim=256):
        super().__init__()

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # 编码器: 逐层下采样
        self.encoder1 = Block(model_channels, model_channels, time_emb_dim)
        self.encoder2 = Block(model_channels, model_channels * 2, time_emb_dim)
        self.encoder3 = Block(model_channels * 2, model_channels * 2, time_emb_dim)

        # 下采样层
        self.downsample1 = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)

        # 瓶颈层
        self.bottleneck = Block(model_channels * 2, model_channels * 2, time_emb_dim)

        # 上采样层
        self.upsample1 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(model_channels * 2, model_channels, 3, stride=2, padding=1, output_padding=1)

        # 解码器: 逐层上采样
        self.decoder1 = Block(model_channels * 4, model_channels * 2, time_emb_dim)
        self.decoder2 = Block(model_channels * 4, model_channels, time_emb_dim)

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        """
        参数:
            x: 噪声图像 x_t, shape (B, C, H, W)
            t: 时间步, shape (B,)
        返回:
            预测的噪声, shape 与 x 相同
        """
        # 时间嵌入
        t_emb = self.time_embed(t)

        # 编码路径
        x1 = self.init_conv(x)          # (B, 64, 28, 28)
        x2 = self.encoder1(x1, t_emb)   # (B, 64, 28, 28)
        x3 = self.downsample1(x2)       # (B, 64, 14, 14)
        x4 = self.encoder2(x3, t_emb)   # (B, 128, 14, 14)
        x5 = self.downsample2(x4)       # (B, 128, 7, 7)
        x6 = self.encoder3(x5, t_emb)   # (B, 128, 7, 7)

        # 瓶颈层
        x7 = self.bottleneck(x6, t_emb) # (B, 128, 7, 7)

        # 解码路径（带跳跃连接）
        x8 = self.upsample1(x7)         # (B, 128, 14, 14)
        x9 = self.decoder1(torch.cat([x8, x4], dim=1), t_emb)  # 拼接跳跃连接
        x10 = self.upsample2(x9)        # (B, 64, 28, 28)
        x11 = self.decoder2(torch.cat([x10, x2], dim=1), t_emb)

        # 输出预测噪声
        return self.final_conv(x11)


# ========================================
# 5. DDPM 完整模型
# ========================================

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model 的完整实现
    包含前向扩散、反向去噪、训练和采样功能
    """
    def __init__(self, model, schedule):
        """
        参数:
            model: 噪声预测网络（如 U-Net）
            schedule: GaussianDiffusionSchedule 实例
        """
        super().__init__()
        self.model = model
        self.schedule = schedule

    def p_losses(self, x_0, t):
        """
        计算训练损失: MSE(epsilon_theta(x_t, t), epsilon)

        参数:
            x_0: 原始图像, shape (B, C, H, W)
            t: 时间步, shape (B,)
        返回:
            损失标量
        """
        # 从标准正态分布采样噪声
        noise = torch.randn_like(x_0)

        # 一步到位计算 x_t
        x_t = self.schedule.q_sample(x_0, t, noise=noise)

        # 预测噪声
        noise_pred = self.model(x_t, t)

        # 计算 MSE 损失
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        单步反向去噪: 从 x_t 采样 x_{t-1}

        参数:
            x_t: 当前噪声图像, shape (B, C, H, W)
            t: 当前时间步, shape (B,) (标量 t, 不是张量)
        """
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)

        # 预测噪声
        noise_pred = self.model(x_t, t_batch)

        # 提取 alpha 相关参数
        alpha_t = self.schedule.extract(self.schedule.alphas, t_batch, x_t.shape)
        alpha_bar_t = self.schedule.extract(self.schedule.alphas_cumprod, t_batch, x_t.shape)
        alpha_bar_t_prev = self.schedule.extract(self.schedule.alphas_cumprod_prev, t_batch, x_t.shape)

        # 计算均值: 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon_theta)
        mean = (
            (1 / torch.sqrt(alpha_t)) *
            (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred)
        )

        # 计算方差
        variance = (1 - alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

        # 添加随机噪声（当 t > 1 时）
        if t > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
        else:
            # 最后一步不添加噪声
            return mean

    @torch.no_grad()
    def sample(self, image_shape, device='cpu'):
        """
        完整的采样过程: 从纯噪声出发,逐步去噪生成图像

        参数:
            image_shape: 生成图像的形状 (B, C, H, W)
            device: 计算设备
        返回:
            生成的图像, shape (B, C, H, W)
        """
        # 从标准正态分布采样初始噪声
        img = torch.randn(image_shape, device=device)

        # 从 T 到 1 逐步去噪
        for t in reversed(range(self.schedule.num_timesteps)):
            img = self.p_sample(img, t)
            # 每 100 步打印一次进度
            if (t + 1) % 200 == 0 or t == 0:
                print(f"  采样进度: {self.schedule.num_timesteps - t}/{self.schedule.num_timesteps}")

        return img


# ========================================
# 6. 训练主程序
# ========================================

def train_ddpm():
    """
    DDPM 在 MNIST 数据集上的完整训练流程
    """
    # 超参数
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 2e-4
    NUM_TIMESTEPS = 1000
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"使用设备: {DEVICE}")

    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.ToTensor(),       # 转为 [0, 1] 的张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 初始化噪声调度
    schedule = GaussianDiffusionSchedule(num_timesteps=NUM_TIMESTEPS)

    # 初始化噪声预测网络
    model = SimpleUNet(in_channels=1, model_channels=64, out_channels=1, time_emb_dim=256).to(DEVICE)

    # 初始化 DDPM
    ddpm = DDPM(model, schedule).to(DEVICE)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    print("=" * 50)
    print("开始训练 DDPM...")
    print("=" * 50)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)
            batch_size = images.shape[0]

            # 随机采样时间步
            t = torch.randint(0, NUM_TIMESTEPS, (batch_size,), device=DEVICE).long()

            # 计算损失
            loss = ddpm.p_losses(images, t)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{EPOCHS}], 平均损失: {avg_loss:.6f}")

        # 每 5 个 epoch 生成并保存样本
        if (epoch + 1) % 5 == 0:
            model.eval()
            print(f"  生成样本中...")
            samples = ddpm.sample(image_shape=(16, 1, 28, 28), device=DEVICE)
            samples = samples.clamp(-1, 1)  # 裁剪到 [-1, 1]

            # 可视化
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                ax.axis('off')
            plt.suptitle(f'Epoch {epoch+1} - 生成样本')
            plt.tight_layout()
            plt.savefig(f'ddpm_samples_epoch_{epoch+1}.png', dpi=100)
            plt.close()
            print(f"  样本已保存: ddpm_samples_epoch_{epoch+1}.png")

    print("训练完成!")
    return ddpm


# ========================================
# 7. 可视化前向扩散过程
# ========================================

def visualize_forward_process():
    """
    可视化前向扩散过程: 展示图像如何被逐步加噪
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 取一张 MNIST 图像
    x_0, _ = dataset[0]
    x_0 = x_0.unsqueeze(0)  # (1, 1, 28, 28)

    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    num_steps_to_show = 8

    # 选择要展示的时间步
    steps = [0, 100, 200, 400, 600, 800, 900, 999]

    fig, axes = plt.subplots(1, num_steps_to_show, figsize=(16, 2))
    for i, t in enumerate(steps):
        if t == 0:
            img = x_0
        else:
            t_tensor = torch.tensor([t])
            noise = torch.randn_like(x_0)
            img = schedule.q_sample(x_0, t_tensor, noise=noise)

        axes[i].imshow(img[0, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[i].set_title(f't={t}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle('前向扩散过程: 图像被逐步加噪', fontsize=14)
    plt.tight_layout()
    plt.savefig('forward_process.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    # 可视化前向扩散过程
    visualize_forward_process()
    # 训练 DDPM
    ddpm_model = train_ddpm()
```

---

## 8. 手工代码实现（NumPy 核心数学）

下面使用纯 NumPy 实现 DDPM 的核心数学运算,包括前向扩散、反向去噪和重参数化技巧。这有助于深入理解算法的数学本质。

```python
import numpy as np
import matplotlib.pyplot as plt

# ========================================
# 1. 噪声调度参数计算
# ========================================

def compute_beta_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """
    计算线性噪声调度的所有参数

    参数:
        num_timesteps: 扩散总步数 T
        beta_start: 噪声方差最小值
        beta_end: 噪声方差最大值

    返回:
        包含所有调度参数的字典
    """
    # 线性调度: beta_t 从 beta_start 到 beta_end 线性增长
    betas = np.linspace(beta_start, beta_end, num_timesteps)

    # alpha_t = 1 - beta_t (信息保留比例)
    alphas = 1.0 - betas

    # alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t (累积信息保留比例)
    alphas_cumprod = np.cumprod(alphas)

    # alpha_bar_{t-1},用于反向过程
    alphas_cumprod_prev = np.concatenate(([1.0], alphas_cumprod[:-1]))

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': np.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': np.sqrt(1.0 - alphas_cumprod),
    }


# ========================================
# 2. 前向扩散（一步到位公式）
# ========================================

def forward_diffusion(x_0, t, schedule, noise=None):
    """
    前向扩散: 使用重参数化技巧从 x_0 直接计算 x_t
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    这就是 DDPM 中最重要的"一步到位"公式,
    无需逐步计算 x_1, x_2, ..., x_{t-1}

    参数:
        x_0: 原始图像, shape (H, W) 或 (C, H, W)
        t: 时间步 (标量, 0 <= t < T)
        schedule: 噪声调度参数字典
        noise: 可选,预设的噪声.如果为 None 则随机生成

    返回:
        x_t: 时间步 t 的噪声图像
        epsilon: 使用的噪声（用于后续计算损失）
    """
    if noise is None:
        noise = np.random.randn(*x_0.shape)

    # 提取时间步 t 对应的参数
    sqrt_alpha_bar_t = schedule['sqrt_alphas_cumprod'][t]
    sqrt_one_minus_alpha_bar_t = schedule['sqrt_one_minus_alphas_cumprod'][t]

    # 一步到位公式
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

    return x_t, noise


# ========================================
# 3. 前向扩散（逐步过程,用于可视化）
# ========================================

def forward_diffusion_step_by_step(x_0, schedule, save_every=100):
    """
    逐步执行前向扩散,保存中间结果用于可视化

    参数:
        x_0: 原始图像
        schedule: 噪声调度参数
        save_every: 每隔多少步保存一次

    返回:
        results: 字典,{时间步: 噪声图像}
    """
    T = len(schedule['betas'])
    x_t = x_0.copy()
    results = {0: x_t.copy()}

    for t in range(1, T):
        # 单步扩散: x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1-alpha_t) * noise
        noise = np.random.randn(*x_t.shape)
        alpha_t = schedule['alphas'][t]
        x_t = np.sqrt(alpha_t) * x_t + np.sqrt(1 - alpha_t) * noise

        if (t + 1) % save_every == 0 or t == T - 1:
            results[t] = x_t.copy()

    return results


# ========================================
# 4. 真实后验均值和方差
# ========================================

def compute_posterior_params(x_t, x_0, t, schedule):
    """
    计算真实后验 q(x_{t-1} | x_t, x_0) 的均值和方差

    均值: mu_q = (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) * x_t
                   + sqrt(alpha_bar_{t-1}) * (1 - alpha_t) * x_0) / (1 - alpha_bar_t)
    方差: sigma^2 = (1 - alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)

    参数:
        x_t: 当前噪声图像
        x_0: 原始图像（已知时可用于计算真实后验）
        t: 时间步
        schedule: 噪声调度参数

    返回:
        mean: 后验均值
        variance: 后验方差
    """
    alpha_t = schedule['alphas'][t]
    alpha_bar_t = schedule['alphas_cumprod'][t]
    alpha_bar_t_prev = schedule['alphas_cumprod_prev'][t]

    # 计算后验均值
    numerator = (
        np.sqrt(alpha_t) * (1 - alpha_bar_t_prev) * x_t
        + np.sqrt(alpha_bar_t_prev) * (1 - alpha_t) * x_0
    )
    denominator = 1 - alpha_bar_t
    mean = numerator / denominator

    # 计算后验方差
    variance = (1 - alpha_t) * (1 - alpha_bar_t_prev) / denominator

    return mean, variance


# ========================================
# 5. 从噪声预测结果反推 x_0
# ========================================

def predict_x0_from_noise(x_t, noise_pred, t, schedule):
    """
    根据预测的噪声反推原始图像 x_0
    由 x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
    可得 x_0 = (x_t - sqrt(1-alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t)

    参数:
        x_t: 当前噪声图像
        noise_pred: 预测的噪声 epsilon_theta(x_t, t)
        t: 时间步
        schedule: 噪声调度参数

    返回:
        x_0_pred: 预测的原始图像
    """
    alpha_bar_t = schedule['alphas_cumprod'][t]
    x_0_pred = (x_t - np.sqrt(1 - alpha_bar_t) * noise_pred) / np.sqrt(alpha_bar_t)
    return x_0_pred


# ========================================
# 6. 反向去噪单步（给定噪声预测）
# ========================================

def reverse_diffusion_step(x_t, noise_pred, t, schedule):
    """
    单步反向去噪: 从 x_t 和预测噪声计算 x_{t-1}

    x_{t-1} = (1/sqrt(alpha_t)) * (x_t - ((1-alpha_t)/sqrt(1-alpha_bar_t)) * epsilon_theta)
              + sigma_t * z

    参数:
        x_t: 当前噪声图像
        noise_pred: 神经网络预测的噪声
        t: 时间步
        schedule: 噪声调度参数

    返回:
        x_{t_minus_1}: 去噪后的图像
    """
    alpha_t = schedule['alphas'][t]
    alpha_bar_t = schedule['alphas_cumprod'][t]
    alpha_bar_t_prev = schedule['alphas_cumprod_prev'][t]

    # 计算去噪均值
    mean = (1.0 / np.sqrt(alpha_t)) * (
        x_t - ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * noise_pred
    )

    # 计算噪声标准差
    variance = (1 - alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
    sigma_t = np.sqrt(variance)

    # 添加随机噪声（当 t > 1 时）
    if t > 1:
        z = np.random.randn(*x_t.shape)
        x_t_minus_1 = mean + sigma_t * z
    else:
        # 最后一步不添加噪声
        x_t_minus_1 = mean

    return x_t_minus_1


# ========================================
# 7. 模拟训练过程（简化版）
# ========================================

def simulate_training(x_0, schedule, noise_predictor_fn, num_steps=1000):
    """
    模拟 DDPM 的训练过程

    参数:
        x_0: 原始图像
        schedule: 噪声调度参数
        noise_predictor_fn: 模拟的噪声预测函数 f(x_t, t) -> predicted_noise
        num_steps: 训练步数
    """
    losses = []

    for step in range(num_steps):
        # 随机采样时间步
        T = len(schedule['betas'])
        t = np.random.randint(0, T)

        # 采样噪声
        noise = np.random.randn(*x_0.shape)

        # 一步到位计算 x_t
        x_t, _ = forward_diffusion(x_0, t, schedule, noise=noise)

        # 预测噪声
        noise_pred = noise_predictor_fn(x_t, t)

        # 计算 MSE 损失
        loss = np.mean((noise_pred - noise) ** 2)
        losses.append(loss)

    return losses


# ========================================
# 8. 完整示例: 可视化前向和反向过程
# ========================================

def demo_diffusion_process():
    """
    完整演示: 前向扩散（加噪）+ 反向去噪（恢复）
    使用模拟的噪声预测器
    """
    # 生成一个简单的测试图像（渐变图案）
    np.random.seed(42)
    img_size = 64
    x_0 = np.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            x_0[i, j] = np.sin(2 * np.pi * i / img_size) * np.cos(2 * np.pi * j / img_size)
    x_0 = (x_0 - x_0.min()) / (x_0.max() - x_0.min()) * 2 - 1  # 归一化到 [-1, 1]

    # 计算噪声调度
    schedule = compute_beta_schedule(num_timesteps=1000)

    # 可视化前向扩散过程
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    steps_to_show = [0, 100, 300, 600, 999]

    # 固定噪声以确保可重复性
    noise = np.random.randn(*x_0.shape)

    print("前向扩散过程:")
    for idx, t in enumerate(steps_to_show):
        if t == 0:
            x_t = x_0.copy()
        else:
            x_t, _ = forward_diffusion(x_0, t, schedule, noise=noise)

        axes[0, idx].imshow(x_t, cmap='gray', vmin=-1, vmax=1)
        axes[0, idx].set_title(f'前向 t={t}')
        axes[0, idx].axis('off')
        snr = np.var(x_0) / np.var(x_t - x_0) if t > 0 else float('inf')
        print(f"  t={t:4d}, alpha_bar={schedule['alphas_cumprod'][t]:.6f}, SNR={snr:.4f}")

    # 模拟反向去噪过程（使用真实噪声作为"完美预测器"）
    print("\n反向去噪过程（使用完美噪声预测器）:")
    x_t, used_noise = forward_diffusion(x_0, 999, schedule)

    reverse_steps = [999, 800, 600, 300, 100, 0]
    x_current = x_t.copy()

    for idx, t in enumerate(reverse_steps):
        if t < 999:
            # 从上一步的 x_{t+1} 继续去噪到 x_t
            # 注意:这里我们使用真实的 used_noise 作为"完美预测"
            # 实际中应该用 noise_predictor_fn(x_current, t)
            x_current = reverse_diffusion_step(x_current, used_noise, t, schedule)

        axes[1, min(idx, 4)].imshow(x_current, cmap='gray', vmin=-1, vmax=1)
        axes[1, min(idx, 4)].set_title(f'反向 t={t}')
        axes[1, min(idx, 4)].axis('off')
        print(f"  t={t:4d}, 与原图MSE={np.mean((x_current - x_0)**2):.6f}")

    plt.suptitle('DDPM 前向扩散与反向去噪过程演示', fontsize=14)
    plt.tight_layout()
    plt.savefig('diffusion_process_demo.png', dpi=150)
    plt.show()

    # 打印 alpha_bar 随时间的变化
    print("\nalpha_bar_t 的变化:")
    for t in [0, 100, 200, 500, 800, 999]:
        print(f"  t={t:4d}: alpha_bar_t = {schedule['alphas_cumprod'][t]:.8f}")


# ========================================
# 9. 重参数化技巧演示
# ========================================

def demo_reparameterization():
    """
    演示重参数化技巧在扩散模型中的应用

    重参数化技巧的核心思想:
    不直接从参数化分布中采样,而是将随机性分离出来:
    如果 z ~ N(mu, sigma^2),则 z = mu + sigma * epsilon, epsilon ~ N(0,1)

    这样梯度可以通过 mu 和 sigma 流向参数
    """
    np.random.seed(42)

    # 原始数据
    x_0 = np.array([1.0, 0.5, -0.5, -1.0])

    # 时间步
    t = 500
    schedule = compute_beta_schedule(num_timesteps=1000)

    # 方法1: 逐步计算（效率低）
    print("方法1: 逐步计算 x_500")
    x = x_0.copy()
    for step in range(1, t + 1):
        noise = np.random.randn(*x.shape)
        alpha = schedule['alphas'][step]
        x = np.sqrt(alpha) * x + np.sqrt(1 - alpha) * noise
    print(f"  x_{t} = {x}")

    # 方法2: 一步到位（重参数化技巧,效率高）
    print("\n方法2: 一步到位计算 x_500（重参数化技巧）")
    np.random.seed(42)
    epsilon = np.random.randn(*x_0.shape)
    x_direct = np.sqrt(schedule['alphas_cumprod'][t]) * x_0 + np.sqrt(1 - schedule['alphas_cumprod'][t]) * epsilon
    print(f"  x_{t} = {x_direct}")
    print(f"  注意: 两种方法的分布相同,但方法2只需一步计算")


if __name__ == '__main__':
    demo_diffusion_process()
    demo_reparameterization()
```

---

## 9. 可视化与结果理解

### 9.1 前向扩散过程可视化

前向扩散过程展示了图像如何被逐步破坏。在可视化中可以观察到:

- **t = 0**: 原始图像,完全清晰
- **t = 100**: 轻微噪声,图像仍可辨认,但细节开始模糊
- **t = 300**: 中等噪声,图像轮廓尚存,但大部分细节已消失
- **t = 600**: 大量噪声,只能隐约看到物体的基本形状
- **t = 1000**: 完全噪声,无法辨认任何信息

关键观察:$\bar{\alpha}_t$ 随 $t$ 的增加呈指数级衰减,这意味着信息损失在初期较慢、后期加速。

### 9.2 反向去噪过程可视化

反向去噪过程是前向扩散的逆过程:

- **t = 1000**: 纯高斯噪声
- **t = 800**: 噪声中开始出现模糊的形状
- **t = 600**: 物体的轮廓逐渐清晰
- **t = 300**: 细节逐步恢复
- **t = 0**: 最终生成的清晰图像

### 9.3 去噪过程中噪声预测的可视化

可视化神经网络在每个时间步预测的噪声 $\boldsymbol{\epsilon}_\theta(x_t, t)$ 可以帮助理解:

- 在早期（大 $t$）,网络预测的是"大致的噪声方向"
- 在后期（小 $t$）,网络预测的是"精细的噪声纹理"
- 网络学会了根据不同的噪声水平采取不同的去噪策略

### 9.4 $\bar{\alpha}_t$ 的可视化

```python
import numpy as np
import matplotlib.pyplot as plt

# 可视化 alpha_bar_t 随时间步的变化
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(alphas_cumprod)
plt.xlabel('Time step t')
plt.ylabel(r'$\bar{\alpha}_t$')
plt.title('Cumulative product of alphas')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(np.sqrt(alphas_cumprod), label=r'$\sqrt{\bar{\alpha}_t}$ (signal)')
plt.plot(np.sqrt(1 - alphas_cumprod), label=r'$\sqrt{1 - \bar{\alpha}_t}$ (noise)')
plt.xlabel('Time step t')
plt.ylabel('Coefficient')
plt.title('Signal and noise coefficients')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('alpha_bar_visualization.png', dpi=150)
plt.show()
```

### 9.5 生成样本质量随训练的演变

在训练过程中,定期生成样本并观察:

- **初期（几百步迭代）**:生成的图像是模糊的、无结构的
- **中期（几万步迭代）**:开始出现物体的大致形状,但细节不正确
- **后期（几十万步迭代）**:生成质量显著提高,细节丰富,多样性好
- **收敛后**:损失趋于稳定,生成质量不再显著提升

### 9.6 生成多样性评估

生成多个样本并观察:

```python
# 生成多个样本并展示多样性
samples = ddpm.sample(image_shape=(64, 1, 28, 28))  # 生成 64 个 MNIST 图像
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
    ax.axis('off')
plt.suptitle('DDPM 生成的 MNIST 样本（展示多样性）')
plt.tight_layout()
plt.savefig('ddpm_diversity.png', dpi=150)
plt.show()
```

---

## 10. 模型评估

### 10.1 Fréchet Inception Distance (FID)

FID 是评估生成图像质量的最常用指标。它衡量生成图像分布和真实图像分布在 Inception 网络特征空间中的距离。

**计算方法:**

1. 使用预训练的 Inception-V3 网络,分别提取真实图像和生成图像的特征向量
2. 计算两组特征向量的均值 $\mu_r, \mu_g$ 和协方差矩阵 $\Sigma_r, \Sigma_g$
3. 计算 FID 距离:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

**解读:**
- FID 越低,表示生成图像质量越好、与真实图像的分布越接近
- FID < 10: 生成质量优秀
- FID 在 10 ~ 50: 生成质量良好
- FID > 100: 生成质量较差

**PyTorch 实现:**

```python
from scipy import linalg

def calculate_fid(real_features, generated_features):
    """
    计算 FID 分数

    参数:
        real_features: 真实图像的特征, shape (N_real, D)
        generated_features: 生成图像的特征, shape (N_gen, D)
    返回:
        FID 分数
    """
    # 计算均值和协方差
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    # 计算均值差的平方
    diff = mu_real - mu_gen

    # 计算矩阵平方根
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

    # 处理数值不稳定
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
    return float(fid)
```

### 10.2 Inception Score (IS)

IS 评估生成图像的质量和多样性。

**计算方法:**

1. 使用预训练的 Inception 网络对生成图像进行分类
2. 对于每个生成图像,得到条件分布 $p(y|x)$
3. 计算边缘分布 $p(y) = \mathbb{E}_x[p(y|x)]$
4. 计算 KL 散度并取指数:

$$\text{IS} = \exp\left(\mathbb{E}_x\left[D_{\text{KL}}(p(y|x) \| p(y))\right]\right)$$

**解读:**
- IS 越高越好（理论上无上界）
- IS 的高分意味着:每个生成图像都"像"某个类别（质量高）,且生成图像覆盖了多个类别（多样性好）

### 10.3 其他评估指标

- **NLL（Negative Log-Likelihood）**: 负对数似然,衡量模型对真实数据的拟合程度。扩散模型通过 ELBO 提供对数似然的下界。
- **Precision 和 Recall**:分别衡量生成图像的质量（Precision,生成图像中有多少是"真实"的）和覆盖率（Recall,真实图像中有多少被生成覆盖）。
- **LPIPS（Learned Perceptual Image Patch Similarity）**:感知相似度,衡量生成图像与参考图像在感知上的差异。

### 10.4 DDPM 在标准数据集上的评估结果

DDPM 原论文在 CIFAR-10 数据集上的 FID 结果:

| 模型 | FID (CIFAR-10) | 采样步数 |
|------|---------------|---------|
| DDPM（原论文） | 3.17 | 1000 |
| DDIM | 4.47 | 100 |
| 改进 DDPM | 2.92 | 1000 |
| ADM (Dhariwal et al.) | 2.97 | 250 |
| StyleGAN2 | 2.84 | 1 |

---

## 11. 常见问题与易错点

### 11.1 采样速度慢

**问题**:标准 DDPM 需要 1000 步采样,生成一张图像可能需要数十秒到数分钟。

**解决方案**:

1. **DDIM（Denoising Diffusion Implicit Models）**:将采样步数从 1000 减少到 50 甚至 20,同时保持生成质量。DDIM 通过构建非马尔可夫的前向过程来实现确定性采样。
2. **DPM-Solver**:使用高阶 ODE 求解器来加速扩散模型的采样,可以将步数减少到 10-20 步。
3. **一致性模型（Consistency Models）**:训练一个单步生成模型,实现与多步 DDPM 相当的生成质量。
4. **潜扩散模型（LDM / Stable Diffusion）**:将扩散过程从像素空间搬到低维潜空间（如 $64 \times 64 \times 4$ 而非 $512 \times 512 \times 3$）,大幅减少计算量。

### 11.2 生成质量不够好

**问题**:生成的图像模糊或缺乏细节。

**解决方案**:

1. **增加训练时间**:DDPM 需要较长的训练时间才能收敛。
2. **改进噪声调度**:尝试余弦调度,通常比线性调度效果更好。
3. **增大模型容量**:增加 U-Net 的通道数、层数或注意力模块。
4. **使用 EMA**:使用指数移动平均（EMA）的参数进行采样,通常比直接使用训练参数效果更好。EMA 衰减率通常取 $0.999$ 或 $0.9999$。
5. **分类器引导 / 无分类器引导**:在条件生成任务中,使用引导可以显著提升生成质量。

### 11.3 训练不稳定

**问题**:训练损失不下降或出现波动。

**常见原因及解决方案**:

1. **学习率过大**:尝试降低学习率,从 $10^{-4}$ 开始调整。
2. **数据归一化不正确**:确保图像像素值归一化到 $[-1, 1]$ 而非 $[0, 1]$。
3. **梯度爆炸**:检查是否使用了梯度裁剪（gradient clipping）。
4. **U-Net 架构问题**:确保跳跃连接正确连接,特征图的尺寸匹配。

### 11.4 模式覆盖不足

**问题**:模型只生成了少数几种样本,没有覆盖数据分布的全部模式。

**分析**:相比 GAN 的模式崩溃（mode collapse）,DDPM 的模式覆盖通常更好。但如果训练不充分或模型容量不足,也可能出现类似问题。增加训练时间和模型容量通常可以解决。

### 11.5 一步到位公式的正确理解

**常见误区**:认为前向扩散必须逐步计算 $x_1 \to x_2 \to \cdots \to x_t$。

**纠正**:利用高斯噪声的可加性,DDPM 的前向过程可以一步到位地从 $x_0$ 计算 $x_t$,这大大加速了训练。只有反向过程（采样）才需要逐步计算。

### 11.6 $\bar{\alpha}_t$ 的计算顺序

**注意**:$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ 是**从 $\alpha_1$ 乘到 $\alpha_t$**,而不是反过来。也就是说,$\bar{\alpha}_1 = \alpha_1$,$\bar{\alpha}_2 = \alpha_1 \alpha_2$,以此类推。

### 11.7 采样时最后一步不加噪声

**原因**:当 $t = 1$ 时,设 $z = 0$（不添加随机噪声）。这是因为最后一步应该输出最终的确定结果,而不是一个还带有噪声的版本。如果不这样做,最后一步仍然会引入方差,导致输出质量下降。

---

## 12. 学习总结

DDPM（去噪扩散概率模型）是深度生成模型领域的一个里程碑式工作。它的核心思想简洁而优雅:先逐步加噪将数据破坏为纯噪声,再训练一个神经网络学会逐步去噪来恢复数据。

**核心要点回顾:**

1. **数学基础**:DDPM 基于 VAE 的变分推断框架,通过最大化 ELBO（证据下界）来训练模型。ELBO 的推导利用了贝叶斯定理、马尔可夫性和 Jensen 不等式。

2. **一步到位公式**:$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$,这个公式利用了高斯噪声的可加性,使得训练时可以一步从 $x_0$ 得到任意 $t$ 的 $x_t$。

3. **噪声预测**:DDPM 的关键设计选择是让神经网络预测添加的噪声 $\boldsymbol{\epsilon}$,而不是直接预测 $x_0$ 或 $x_{t-1}$。这使得训练损失简化为简单的 MSE: $\mathcal{L} = \|\boldsymbol{\epsilon}_\theta(x_t, t) - \boldsymbol{\epsilon}\|^2$。

4. **真实后验**:给定 $x_t$ 和 $x_0$,$x_{t-1}$ 的条件分布 $q(x_{t-1}|x_t, x_0)$ 是一个解析可求的高斯分布。DDPM 的反向过程试图用神经网络逼近这个后验分布。

5. **采样过程**:从纯高斯噪声 $x_T$ 出发,利用噪声预测网络逐步去噪,经过 $T$ 步后得到生成的图像 $x_0$。

**DDPM 的历史地位:**

DDPM 开启了扩散模型在图像生成领域的黄金时代。从 DDPM 到 DDIM（加速采样）,到 ADM（击败 GAN）,再到 LDM/Stable Diffusion（效率革命）,扩散模型在短短几年内从一个"有趣的理论模型"发展为最强大的图像生成范式之一。理解 DDPM 的原理,是掌握整个扩散模型家族的基础。

---

## 13. 练习题与思考题（含答案）

### 练习题 1: 前向扩散公式的推导

**题目**:给定单步扩散公式 $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$,其中 $\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$,推导两步公式 $x_2 = \sqrt{\alpha_1 \alpha_2} x_0 + \sqrt{1 - \alpha_1 \alpha_2} \boldsymbol{\epsilon}$,其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

**答案**:

从 $t = 1$ 开始:

$$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \boldsymbol{\epsilon}_0, \quad \boldsymbol{\epsilon}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

$$x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1, \quad \boldsymbol{\epsilon}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

将 $x_1$ 代入 $x_2$:

$$x_2 = \sqrt{\alpha_2} \left(\sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \boldsymbol{\epsilon}_0\right) + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1$$

$$= \sqrt{\alpha_1 \alpha_2} x_0 + \sqrt{\alpha_2(1 - \alpha_1)} \boldsymbol{\epsilon}_0 + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1$$

由于 $\boldsymbol{\epsilon}_0$ 和 $\boldsymbol{\epsilon}_1$ 独立同分布于 $\mathcal{N}(\mathbf{0}, \mathbf{I})$,利用高斯噪声可加性:

$$\sqrt{\alpha_2(1 - \alpha_1)} \boldsymbol{\epsilon}_0 + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1 \sim \mathcal{N}\left(\mathbf{0}, \left[\alpha_2(1 - \alpha_1) + (1 - \alpha_2)\right]\mathbf{I}\right)$$

$$= \mathcal{N}\left(\mathbf{0}, (1 - \alpha_1 \alpha_2)\mathbf{I}\right)$$

因此:

$$x_2 = \sqrt{\alpha_1 \alpha_2} x_0 + \sqrt{1 - \alpha_1 \alpha_2} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### 练习题 2: 噪声预测与 $x_0$ 预测的关系

**题目**:已知一步到位公式 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$,证明:如果噪声预测网络 $\boldsymbol{\epsilon}_\theta(x_t, t)$ 能完美预测噪声（即 $\boldsymbol{\epsilon}_\theta(x_t, t) = \boldsymbol{\epsilon}$）,则:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

**答案**:

由 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$,将 $\boldsymbol{\epsilon} = \boldsymbol{\epsilon}_\theta(x_t, t)$ 代入:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(x_t, t)$$

移项:

$$\sqrt{\bar{\alpha}_t} x_0 = x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(x_t, t)$$

两边除以 $\sqrt{\bar{\alpha}_t}$:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

### 练习题 3: 反向去噪均值的推导

**题目**:DDPM 的反向去噪均值公式为 $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(x_t, t) \right)$。请解释为什么 $x_t$ 的系数是 $\frac{1}{\sqrt{\alpha_t}}$,以及 $\boldsymbol{\epsilon}_\theta$ 的系数是 $-\frac{1 - \alpha_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}}$。

**答案**:

这个公式的推导基于将真实后验均值用噪声 $\boldsymbol{\epsilon}$ 重新参数化。已知:

$$\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t) x_0}{1 - \bar{\alpha}_t}$$

将 $x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$ 代入（利用 $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$,即 $\sqrt{\bar{\alpha}_{t-1}} = \sqrt{\bar{\alpha}_t/\alpha_t}$）:

$$\mu_q = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{\alpha_t}}(1 - \alpha_t) \cdot \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}}{1 - \bar{\alpha}_t}$$

$$= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}} x_t - \frac{(1 - \alpha_t)\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\alpha_t}} \boldsymbol{\epsilon}}{1 - \bar{\alpha}_t}$$

合并 $x_t$ 项:

$$= \frac{\left[\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) + \frac{1-\alpha_t}{\sqrt{\alpha_t}}\right] x_t - \frac{(1 - \alpha_t)\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\alpha_t}} \boldsymbol{\epsilon}}{1 - \bar{\alpha}_t}$$

方括号中的系数:

$$\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) + \frac{1 - \alpha_t}{\sqrt{\alpha_t}} = \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + 1 - \alpha_t}{\sqrt{\alpha_t}} = \frac{1 - \alpha_t \bar{\alpha}_{t-1}}{\sqrt{\alpha_t}} = \frac{1 - \bar{\alpha}_t}{\sqrt{\alpha_t}}$$

代入回原式:

$$\mu_q = \frac{\frac{1 - \bar{\alpha}_t}{\sqrt{\alpha_t}} x_t - \frac{(1 - \alpha_t)\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\alpha_t}} \boldsymbol{\epsilon}}{1 - \bar{\alpha}_t} = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t} \sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}$$

$$= \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$

将 $\boldsymbol{\epsilon}$ 替换为 $\boldsymbol{\epsilon}_\theta(x_t, t)$ 即得到 $\mu_\theta$ 的参数化形式。

### 练习题 4: 方差调度的设计

**题目**:假设 $\beta_t = 0.01$（常数,不随 $t$ 变化）,$T = 1000$,计算 $\bar{\alpha}_{1000}$。这个值是否足够接近 0? 如果不够,应该如何调整?

**答案**:

$\alpha_t = 1 - \beta_t = 0.99$

$\bar{\alpha}_{1000} = 0.99^{1000}$

$\log_{10}(0.99^{1000}) = 1000 \times \log_{10}(0.99) \approx 1000 \times (-0.00436) = -4.36$

$\bar{\alpha}_{1000} = 10^{-4.36} \approx 4.3 \times 10^{-5}$

这个值已经非常接近 0,说明 $x_T$ 几乎不包含 $x_0$ 的信息,满足要求。但如果 $\beta_t$ 更大（如 $\beta_t = 0.1$）,则 $\bar{\alpha}_{1000} = 0.9^{1000} \approx 0$,满足得更快。反之,如果 $\beta_t = 0.001$,则 $\bar{\alpha}_{1000} = 0.999^{1000} \approx 0.368$,信息保留较多,需要增加 $T$ 或增大 $\beta_t$。

### 练习题 5: DDPM 与 GAN 的对比分析

**题目**:请从以下维度对比 DDPM 和 GAN:(1) 训练目标;(2) 训练稳定性;(3) 采样速度;(4) 生成多样性;(5) 似然计算。

**答案**:

| 维度 | DDPM | GAN |
|------|------|-----|
| (1) 训练目标 | 最小化预测噪声与真实噪声的 MSE | 生成器和判别器的极小极大博弈（min-max game） |
| (2) 训练稳定性 | 高,MSE 损失是凸优化,不涉及对抗训练 | 低,需要平衡两个网络的训练,容易发生模式崩溃或梯度消失 |
| (3) 采样速度 | 慢,需要 1000 次神经网络前向传播 | 快,只需 1 次生成器前向传播 |
| (4) 生成多样性 | 高,每步采样都添加随机噪声,覆盖数据分布的模式 | 低,容易发生模式崩溃（mode collapse）,只生成少数几种样本 |
| (5) 似然计算 | 可以通过 ELBO 计算对数似然的下界 | 无法直接计算似然,属于隐式生成模型 |

---

## 14. 学习路径建议

### 14.1 前置知识学习路径

在学习 DDPM 之前,建议按以下顺序掌握前置知识:

1. **概率论基础**:高斯分布的性质（均值、方差、可加性）,贝叶斯定理,KL 散度的定义和性质
2. **VAE**:变分自编码器是理解 DDPM 最重要的前置知识。DDPM 的 ELBO 推导与 VAE 完全类似,建议先彻底理解 VAE 的 ELBO 推导过程
3. **U-Net 架构**:DDPM 使用 U-Net 作为噪声预测网络,需要理解其编码器-解码器结构和跳跃连接的作用
4. **重参数化技巧**:在 VAE 中已经学过,DDPM 中也大量使用

### 14.2 DDPM 核心知识学习路径

1. **前向扩散过程**:理解单步公式 $q(x_t|x_{t-1})$,掌握一步到位公式 $q(x_t|x_0)$ 的推导
2. **反向去噪过程**:理解 $p_\theta(x_{t-1}|x_t)$ 的参数化方式
3. **ELBO 推导**:从对数似然出发,推导 ELBO,理解为什么 DDPM 的训练目标可以简化为 MSE
4. **噪声预测参数化**:理解为什么预测噪声比预测均值或 $x_0$ 效果更好
5. **训练和采样算法**:掌握训练伪代码和采样伪代码

### 14.3 进阶学习路径

掌握 DDPM 之后,建议按以下顺序学习后续模型:

1. **DDPM（当前）**:理解和实现基础的去噪扩散概率模型
2. **DDIM（Denoising Diffusion Implicit Models）**:学习如何加速 DDPM 的采样过程（从 1000 步减少到 50 步）,理解非马尔可夫前向过程和确定性采样
3. **Improved DDPM**:学习余弦调度、学习反向方差等改进技术
4. **分类器引导和无分类器引导**:学习如何在扩散模型中引入条件信息（如类别标签、文本描述）
5. **Stable Diffusion / 潜扩散模型（LDM）**:学习如何将扩散过程从像素空间搬到低维潜空间,理解预训练的自编码器的作用
6. **ControlNet**:学习如何在 Stable Diffusion 中引入精确的空间控制（如边缘图、深度图）
7. **一致性模型（Consistency Models）**:学习如何实现单步或少量步数的高质量生成
8. **扩散模型与 LLM 的结合**:了解扩散模型在多模态大语言模型中的应用

### 14.4 推荐阅读材料

1. **论文**:
   - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020 -- DDPM 原论文,必读
   - Song et al., "Denoising Diffusion Implicit Models", ICLR 2021 -- DDIM,加速采样
   - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021 -- 改进 DDPM
   - Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021 -- ADM,首次击败 GAN
   - Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS Workshop 2021 -- 无分类器引导
   - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022 -- LDM / Stable Diffusion

2. **教程和博客**:
   - Lilian Weng 的博客 "What are Diffusion Models?" -- 优秀的综述性博客
   - Yang Song 的博客 "Generative Modeling by Estimating Gradients of the Data Distribution" -- 从分数匹配角度理解扩散模型
   - "鱼书"系列《深度学习入门5:生成模型》-- 从 VAE 到 DDPM 的循序渐进推导

3. **代码参考**:
   - Hugging Face Diffusers 库 -- 工业级扩散模型实现
   - lucidrains 的 DDPM PyTorch 实现 -- 简洁清晰的代码参考

---

## 参考文献

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.
2. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. ICML 2015.
3. Song, Y., Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS 2019.
4. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. ICLR 2021.
5. Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. ICML 2021.
6. Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. NeurIPS 2021.
7. Ho, J., & Salimans, T. (2021). Classifier-Free Diffusion Guidance. NeurIPS Workshop 2021.
8. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
9. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

---
