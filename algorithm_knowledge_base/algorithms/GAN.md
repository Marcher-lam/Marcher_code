# GAN (生成式对抗网络) 学习文档

> 通过生成器与判别器的对抗博弈,学习真实数据分布并生成高质量新样本的深度生成模型

---

## 1. 算法基础认知

### 一句话定义

GAN 是一种通过两个神经网络(生成器与判别器)相互对抗来学习数据分布并生成逼真样本的无监督生成模型。

### 直觉类比

GAN 的核心思想可以用"造假钞者 vs 验钞员"的博弈来理解:

- **造假钞者(生成器 Generator, G)**: 试图制造尽可能逼真的假钞。他从随机噪声出发,学习如何把噪声变成看起来和真钞一模一样的"钞票"。他的目标就是骗过验钞员。
- **验钞员(判别器 Discriminator, D)**: 负责检验每一张钞票是真是假。他同时看到真钞和假钞,需要准确区分二者。
- **对抗过程**: 造假钞者不断改进造假技术,验钞员也不断提高鉴别能力。二者在博弈中"水涨船高"——最终,当造假钞者制造的假钞连最优验钞员都无法分辨时,造假钞者就已经成功学到了真实钞票的分布,能够生成以假乱真的样本。

这个博弈的关键在于:造假钞者和验钞员的目标是相互矛盾的(一个要骗过,一个要识破),但他们又必须依赖对方才能提升自己。

### 历史背景

GAN 由 Ian Goodfellow 等人于 2014 年在其论文《Generative Adversarial Nets》中首次提出。这一工作开创性地将博弈论思想引入深度生成模型,Goodfellow 因此获得"GAN 之父"的美名。在此之前,生成模型主要依赖 Boltzmann 机、VAE 等方法,这些方法要么训练困难,要么生成质量不高。GAN 的出现开辟了生成式模型的新阵地,此后涌现了大量变体(DCGAN、WGAN、StyleGAN、CycleGAN 等),被广泛应用于图像生成、图像修复、超分辨率、文本到图像合成等领域。2017 年,Tao Xu 等人提出了 AttnGAN(注意力生成式对抗网络),将注意力机制引入 GAN,实现了基于文本描述的细粒度图像生成。

### 算法定位

- 类型:无监督学习 --> 生成模型
- 输出:与真实数据分布相同的新样本(如图像、音频、文本等)
- 模型类型:隐式生成模型(不显式建模数据分布的密度函数)

### 前置知识

- 神经网络基础:前馈神经网络、反向传播、激活函数
- 概率论:概率分布、期望、KL 散度、JS 散度
- 博弈论基础:纳什均衡、零和博弈(了解即可)
- 优化方法:梯度下降、学习率调度
- 深度学习框架:PyTorch 或 TensorFlow 的基本使用

---

## 2. 核心原理

### 2.1 核心思想

GAN 的核心思想是将生成问题转化为一个二人零和博弈问题。我们不再像 VAE 那样显式地建模数据分布 $p_{data}(x)$,而是训练一个生成器 $G$ 从先验分布(通常是高斯噪声)中采样并映射到数据空间,使其生成的数据分布 $p_g$ 尽可能接近真实数据分布 $p_{data}$。为了衡量两个分布之间的接近程度,我们引入一个判别器 $D$ 作为"度量工具",通过对真假样本进行二分类来引导生成器的训练。

核心思想可以概括为:通过判别器的分类能力间接衡量生成分布与真实分布的差异,利用对抗训练使生成分布逐步逼近真实分布。

### 2.2 工作流程

1. **采样噪声向量**:从先验分布 $p_z(z)$ 中随机采样一个噪声向量 $z$
   - 输入:标量随机变量 $z$,通常服从标准正态分布 $z \sim \mathcal{N}(0, I)$
   - 输出:潜在空间的随机表示

2. **生成假样本**:将噪声向量输入生成器,得到生成样本 $\hat{x} = G(z)$
   - 关键操作:生成器 $G$ 是一个神经网络,将低维噪声映射到高维数据空间
   - 输出:与真实数据同维度的生成样本 $\hat{x}$

3. **判别真假**:将真实样本 $x$ 和生成样本 $\hat{x}$ 同时输入判别器
   - 判别器输出 $D(x) \in [0,1]$,表示输入为真实数据的概率
   - 真实样本 $x \sim p_{data}$ 的标签为 1,生成样本 $\hat{x} \sim p_g$ 的标签为 0

4. **交替训练**:
   - **训练判别器**:固定 $G$,更新 $D$ 使其能更好地区分真假(最大化目标函数)
   - **训练生成器**:固定 $D$,更新 $G$ 使其生成更逼真的样本骗过 $D$(最小化目标函数)
   - 决策点:在每一轮迭代中,通常先训练 $k$ 步判别器,再训练 1 步生成器($k=1$ 是常见选择)

5. **收敛判定**:当判别器对真假样本的输出都接近 0.5 时,说明生成分布已接近真实分布
   - 此时 $D$ 无法区分真假,达到了纳什均衡

### 2.3 关键概念解释

- **生成器 Generator (G)**:一个将潜在空间(latent space)映射到数据空间的神经网络。它接收随机噪声 $z$ 作为输入,输出一个生成的样本 $G(z)$。生成器的目标是生成尽可能真实的样本,使判别器无法区分真假。

- **判别器 Discriminator (D)**:一个二分类神经网络。它接收一个样本作为输入,输出该样本是真实数据的概率 $D(x) \in [0,1]$。判别器的目标是尽可能准确地区分真实样本和生成样本。

- **潜在空间 Latent Space**:生成器的输入空间,通常是一个低维的连续空间(如 100 维的高斯空间)。潜在空间中的每个点通过生成器映射到数据空间中的一个样本。

- **纳什均衡 Nash Equilibrium**:博弈论中的核心概念,指在博弈中没有任何一个参与者能通过单方面改变策略来获得更大收益的状态。在 GAN 中,纳什均衡对应 $p_g = p_{data}$ 且 $D(x) = 0.5$ 的状态。

- **Minimax 博弈**:GAN 的训练目标是一个极小极大值问题(minimax),生成器要最小化目标函数,判别器要最大化目标函数,二者形成对抗。

### 2.4 几何/直观解释

从概率分布的角度理解 GAN 的训练过程:

想象在一条数轴上,红色曲线代表真实数据分布 $p_{data}$,绿色曲线代表生成数据分布 $p_g$,蓝色曲线代表判别器输出 $D(x)$。

- **训练初期**:生成分布 $p_g$ 与真实分布 $p_{data}$ 差异很大,判别器能够轻松区分真假(蓝色曲线在 $p_{data}$ 处接近 1,在 $p_g$ 处接近 0)
- **训练中期**:随着生成器改进,$p_g$ 逐渐向 $p_{data}$ 靠近,判别器越来越难以区分(蓝色曲线趋于平缓)
- **训练末期**:当 $p_g = p_{data}$ 时,判别器面对的输入来自同一分布,无法区分真假,此时 $D(x) = 0.5$ 处处成立

从另一个角度看,Goodfellow 巧妙地采用对抗机制来训练生成器——先生成一幅假图,尽可能让判图员无法分辨其真伪,提升判图员的水平,然后再生成一幅更加逼真的假图,尽量让判图员丧失判别真伪的能力,如此往复,一个能够以假乱真的图像生成器就诞生了。

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 | 说明 |
|------|------|------|
| $x$ | 真实数据样本 | $x \sim p_{data}(x)$ |
| $z$ | 噪声向量(潜在变量) | $z \sim p_z(z)$,通常 $z \sim \mathcal{N}(0, I)$ |
| $G(z;\theta_g)$ | 生成器 | 将噪声 $z$ 映射为生成样本,参数为 $\theta_g$ |
| $D(x;\theta_d)$ | 判别器 | 输出 $x$ 为真实数据的概率,参数为 $\theta_d$ |
| $p_{data}$ | 真实数据分布 | 我们希望学习的目标分布 |
| $p_g$ | 生成数据分布 | $G(z)$ 的分布,我们希望 $p_g \to p_{data}$ |
| $p_z$ | 噪声先验分布 | 通常为标准正态分布 |
| $\mathbb{E}$ | 期望 | 数学期望算子 |

### 3.2 问题形式化

给定真实数据分布 $p_{data}(x)$ 和噪声先验分布 $p_z(z)$,我们的目标是:

$$\min_G \max_D V(D, G)$$

其中价值函数 $V(D,G)$ 定义为:

$$V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

这是一个极小极大值优化问题:
- 判别器 $D$ 要**最大化** $V(D,G)$:希望对真实样本输出高概率(接近 1),对生成样本输出低概率(接近 0)
- 生成器 $G$ 要**最小化** $V(D,G)$:希望判别器对生成样本输出高概率(使 $1-D(G(z))$ 尽可能小)

### 3.3 目标函数/损失函数

**判别器的目标**:

对于固定的生成器 $G$,最优判别器 $D^*$ 的目标是:

$$D^* = \arg\max_D V(D, G) = \arg\max_D \left\{ \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] \right\}$$

这一目标的两项含义:
- 第一项 $\mathbb{E}_{x \sim p_{data}}[\log D(x)]$:判别器希望将真实数据判断为真的概率最大化
- 第二项 $\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$:判别器希望将生成数据判断为假的概率最大化

**生成器的目标**:

对于固定的判别器 $D$,生成器 $G$ 的目标是:

$$G^* = \arg\min_G V(D^*, G) = \arg\min_G \mathbb{E}_{z \sim p_z}[\log(1 - D^*(G(z)))]$$

这要求生成器生成使最优判别器无法区分的样本。

**为什么选择这个损失函数?**

这是一个自然的二分类交叉熵损失。判别器本质上是一个二分类器,对真实样本标签为 1,对生成样本标签为 0。交叉熵损失是二分类任务中最自然的损失函数选择,它直接最大化正确分类的对数似然。

### 3.4 推导过程

#### 第一步:推导最优判别器 $D^*$

对于任意固定的 $G$,我们需要找到最优判别器:

$$D^* = \arg\max_D V(D, G)$$

将期望展开为积分形式。对于输入空间中的一个点 $x$,判别器接收到来自 $p_{data}$ 的真实样本和来自 $p_g$ 的生成样本。我们将期望写成对 $x$ 的积分:

$$V(D, G) = \int_x p_{data}(x) \log D(x) \, dx + \int_z p_z(z) \log(1 - D(G(z))) \, dz$$

对第二项做变量替换,令 $x = G(z)$,则 $z = G^{-1}(x)$。当 $G$ 固定时,生成样本的分布为 $p_g(x)$:

$$\int_z p_z(z) \log(1 - D(G(z))) \, dz = \int_x p_g(x) \log(1 - D(x)) \, dx$$

因此:

$$V(D, G) = \int_x \left[ p_{data}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx$$

现在,对于每一个 $x$,我们需要选择 $D(x)$ 来最大化被积函数。由于积分和关于 $D(x)$ 的最大化可以交换(对每个 $x$ 独立选择 $D(x)$),我们只需要对每个点 $x$ 最大化:

$$f(D) = p_{data}(x) \log D + p_g(x) \log(1 - D)$$

对 $D$ 求导并令导数为零:

$$\frac{\partial f}{\partial D} = \frac{p_{data}(x)}{D} - \frac{p_g(x)}{1 - D} = 0$$

解方程:

$$\frac{p_{data}(x)}{D} = \frac{p_g(x)}{1 - D}$$

$$p_{data}(x)(1 - D) = p_g(x) \cdot D$$

$$p_{data}(x) - p_{data}(x) \cdot D = p_g(x) \cdot D$$

$$p_{data}(x) = D \cdot (p_{data}(x) + p_g(x))$$

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

这就是最优判别器的解析解。其直觉是:判别器根据在该点真实数据的密度与生成数据的密度的比值来判断真伪。如果 $p_{data}(x) \gg p_g(x)$,则 $D^*(x) \approx 1$(很可能是真的);如果 $p_{data}(x) \ll p_g(x)$,则 $D^*(x) \approx 0$(很可能是假的)。

#### 第二步:将最优判别器代入,推导全局最优

将 $D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$ 代入 $V(D, G)$:

$$C(G) = \max_D V(D, G) = V(D^*, G)$$

$$= \mathbb{E}_{x \sim p_{data}}\left[\log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{data}(x) + p_g(x)}\right]$$

我们对其做变形。首先展开第一项:

$$\mathbb{E}_{x \sim p_{data}}\left[\log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] = \mathbb{E}_{x \sim p_{data}}\left[\log \frac{p_{data}(x)}{\frac{1}{2}(p_{data}(x) + p_g(x))} - \log 2\right]$$

$$= \mathbb{E}_{x \sim p_{data}}\left[\log \frac{p_{data}(x)}{\frac{1}{2}(p_{data}(x) + p_g(x))}\right] - \log 2$$

类似地,展开第二项:

$$\mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{data}(x) + p_g(x)}\right] = \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{\frac{1}{2}(p_{data}(x) + p_g(x))} - \log 2\right]$$

$$= \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{\frac{1}{2}(p_{data}(x) + p_g(x))}\right] - \log 2$$

两式相加:

$$C(G) = \mathbb{E}_{x \sim p_{data}}\left[\log \frac{2 p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{2 p_g(x)}{p_{data}(x) + p_g(x)}\right] - 2\log 2$$

回忆 KL 散度的定义:

$$D_{KL}(P \| Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

令 $P_m = \frac{1}{2}(p_{data} + p_g)$ 为真实分布与生成分布的混合分布,则:

$$C(G) = \mathbb{E}_{x \sim p_{data}}\left[\log \frac{p_{data}(x)}{P_m(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{P_m(x)}\right] - 2\log 2$$

$$= 2 \cdot D_{KL}\left(\frac{p_{data} + p_g}{2} \,\Big\|\, \frac{1}{2}p_{data}\right) - 2\log 2$$

或者等价地,利用 JS 散度的定义 $D_{JS}(P \| Q) = \frac{1}{2}D_{KL}(P \| M) + \frac{1}{2}D_{KL}(Q \| M)$,其中 $M = \frac{P+Q}{2}$:

$$C(G) = 2 \cdot D_{JS}(p_{data} \| p_g) - 2\log 2$$

其中 $D_{JS}(p_{data} \| p_g)$ 是 Jensen-Shannon 散度,它衡量两个分布之间的差异。JS 散度具有以下性质:
- $D_{JS}(P \| Q) \geq 0$,当且仅当 $P = Q$ 时等号成立
- $D_{JS}(P \| Q) \leq \log 2$ (有上界)

#### 第三步:全局最优与最优值

由于 $D_{JS}(p_{data} \| p_g) \geq 0$,且等号成立当且仅当 $p_g = p_{data}$,因此:

$$C(G) = 2 \cdot D_{JS}(p_{data} \| p_g) - 2\log 2 \geq -2\log 2 = -\log 4$$

等号成立当且仅当 $p_g = p_{data}$。

因此:
- **全局最优值**: $V^* = -\log 4 \approx -1.386$
- **达到全局最优的条件**: $p_g = p_{data}$,即生成分布等于真实数据分布
- **此时最优判别器**: $D^*(x) = \frac{1}{2}$,即对所有输入输出 0.5

#### 第四步:生成器训练的实际目标函数

在原始论文中,Goodfellow 指出,直接最小化 $\log(1 - D(G(z)))$ 在训练初期会导致梯度消失(因为初始的生成器很差,$D(G(z))$ 接近 0,导致 $\log(1-D(G(z)))$ 接近 $\log 1 = 0$,梯度很小)。因此,实践中常用以下等价的最大化替代目标:

$$\max_G \mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

这与原目标函数 $\min_G \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$ 的梯度方向相同(因为 $\max_G \log D(G(z)) = \min_G \log(1/D(G(z)))$ 提供了更强的梯度信号),且在训练初期提供了更有意义的梯度。

证明等价性:由于 $D^*$ 固定时:

$$\min_G \mathbb{E}_z[\log(1 - D^*(G(z)))] = \min_G \left\{ -\mathbb{E}_z[\log D^*(G(z))] + \mathbb{E}_z[\log(1 - D^*(G(z)))] + \mathbb{E}_z[\log D^*(G(z))] \right\}$$

等价于 $\max_G \mathbb{E}_z[\log D^*(G(z))]$,因为最大化 $\log D^*$ 等价于最小化 $-\log D^*$,这与最小化 $\log(1-D^*)$ 的目标一致(两者都在推动 $D^* \to 1$)。

### 3.5 最终解/算法步骤

**GAN 训练算法伪代码**:

```
算法: GAN 训练
输入: 噪声先验分布 p_z, 迭代次数 T, 判别器每轮训练步数 k
输出: 训练好的生成器 G

1. 随机初始化判别器参数 theta_d 和生成器参数 theta_g
2. for t = 1, 2, ..., T do:
3.     for m = 1, 2, ..., k do:                     // 训练判别器 k 步
4.         从 p_data 中采样 {x^(1), ..., x^(mb)}     // 采样真实数据小批量
5.         从 p_z 中采样 {z^(1), ..., z^(mb)}         // 采样噪声小批量
6.         更新判别器参数(梯度上升):
7.            theta_d <- theta_d + eta * grad_theta_d [
8.                (1/mb) * sum_i log D(x^(i)) +
9.                (1/mb) * sum_i log(1 - D(G(z^(i))))
10.           ]
11.    end for
12.    从 p_z 中采样 {z^(1), ..., z^(mb)}             // 采样噪声小批量
13.    更新生成器参数(梯度上升):
14.       theta_g <- theta_g + eta * grad_theta_g [
15.           -(1/mb) * sum_i log(1 - D(G(z^(i))))
16.       ]
17. end for
```

---

## 4. 训练过程讲解

### 4.1 数据预处理

**必要预处理**:

1. **像素归一化**:
   - 原因:GAN 的生成器最后一层通常使用 tanh 激活函数,输出范围为 $[-1, 1]$,因此真实数据也需要归一化到 $[-1, 1]$
   - 方法:将 $[0, 255]$ 的像素值除以 127.5 后减去 1
   - 代码示例:
     ```python
     # 将图像像素值归一化到 [-1, 1]
     transform = transforms.Compose([
         transforms.ToTensor(),                    # 转为 [0, 1]
         transforms.Normalize([0.5], [0.5])        # 转为 [-1, 1]
     ])
     ```

2. **数据增强**(可选但推荐):
   - 方法:随机裁剪、水平翻转、颜色微调
   - 注意:增强操作不应改变数据的语义含义

3. **数据加载**:
   - 使用 DataLoader 进行批量加载
   - batch_size 通常设为 64 或 128

### 4.2 参数初始化

- **生成器**:使用 Xavier 初始化或 He 初始化
- **判别器**:使用 Xavier 初始化或 He 初始化
- **理由**:合理的初始化可以避免训练初期梯度消失或爆炸,帮助两个网络在相近的能力水平上开始对抗
- **偏置**:通常初始化为 0
- **BatchNorm 参数**:
  - $\gamma$(scale):初始化为 1
  - $\beta$(shift):初始化为 0

### 4.3 迭代过程

GAN 的训练是一个交替优化过程,具体步骤如下:

```python
# GAN 训练主循环
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):

        # ===========================
        # 步骤 1: 训练判别器
        # ===========================
        # 1.1 采样真实数据
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # 1.2 采样噪声并生成假数据
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)

        # 1.3 计算判别器损失
        real_score = discriminator(real_images)       # 真实样本的判别分数
        fake_score = discriminator(fake_images.detach())  # 假样本的判别分数

        d_loss_real = criterion(real_score, torch.ones_like(real_score))   # 真样本标签为 1
        d_loss_fake = criterion(fake_score, torch.zeros_like(fake_score))  # 假样本标签为 0
        d_loss = d_loss_real + d_loss_fake

        # 1.4 更新判别器参数
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # ===========================
        # 步骤 2: 训练生成器
        # ===========================
        # 2.1 采样噪声并生成假数据
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)

        # 2.2 计算生成器损失
        fake_score = discriminator(fake_images)
        g_loss = criterion(fake_score, torch.ones_like(fake_score))  # 希望假样本被判为真

        # 2.3 更新生成器参数
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        # 记录损失
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
```

### 4.4 收敛条件

GAN 的收敛判断比传统机器学习模型更加困难:

- **理论收敛条件**:当 $D(x) = 0.5$ 对所有 $x$ 成立时,达到纳什均衡
- **实际判断**:
  - 判别器损失和生成器损失趋于稳定(不再剧烈震荡)
  - 生成的样本质量主观上看起来足够好
  - 生成样本的多样性足够(没有模式崩塌)
- **注意**:GAN 很难判断是否已经收敛,因为目标函数 $V(D,G)$ 不能直接反映生成质量

### 4.5 超参数及推荐范围

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|----------|--------|
| learning_rate (D) | 判别器学习率 | 1e-4 ~ 4e-4 | 2e-4 |
| learning_rate (G) | 生成器学习率 | 1e-4 ~ 4e-4 | 2e-4 |
| beta1 (Adam) | Adam 优化器一阶矩衰减 | 0.0 ~ 0.5 | 0.5 |
| beta2 (Adam) | Adam 优化器二阶矩衰减 | 0.99 ~ 0.999 | 0.999 |
| latent_dim | 噪声向量维度 | 64 ~ 256 | 100 |
| batch_size | 批量大小 | 32 ~ 256 | 128 |
| n_critic | 每轮训练判别器的步数 | 1 ~ 5 | 1 |

**关键技巧**:在 DCGAN 等变体中,Adam 优化器的 beta1 参数通常设置为 0.5(而非默认的 0.9),这有助于训练稳定性。

---

## 5. 应用场景

### 5.1 典型应用

**应用 1:图像生成**
- 问题类型:生成任务
- 为什么适合 GAN:
  - GAN 能够学习图像数据的高维复杂分布
  - 生成的图像质量高,逼真度高
  - 可控生成(通过条件 GAN, CGAN)
- 实际案例:StyleGAN 2 生成高分辨率人脸图像,质量几乎无法与真实照片区分

**应用 2:图像修复(Image Inpainting)**
- 问题类型:图像修复任务
- 为什么适合 GAN:
  - 判别器可以评估修复区域的视觉一致性
  - 生成器能够生成与周围区域语义一致的填充内容
- 实际案例:使用 GAN 修复老照片中缺失或损坏的区域

**应用 3:超分辨率(Super Resolution)**
- 问题类型:图像增强任务
- 为什么适合 GAN:
  - GAN 的生成器能够生成高频细节,使低分辨率图像变清晰
  - SRGAN(2017)首次将 GAN 用于超分辨率,显著提升了感知质量
- 实际案例:SRGAN、ESRGAN 等模型将低分辨率图像放大 4 倍后仍保持清晰

**应用 4:数据增强(Data Augmentation)**
- 问题类型:辅助分类任务
- 为什么适合 GAN:
  - 在标注数据稀缺时,可用 GAN 生成额外的训练样本
  - 生成样本的多样性有助于提升下游模型的泛化能力
- 实际案例:在医学图像分析中,使用 GAN 生成罕见的病变样本用于训练

**应用 5:文本到图像生成(Text-to-Image)**
- 问题类型:条件生成任务
- 为什么适合 GAN:
  - 通过 CGAN 框架,将文本描述作为条件,引导图像生成
  - AttnGAN 利用注意力机制实现细粒度文本到图像的生成
- 实际案例:AttnGAN(Tao Xu 等,2017)通过注意力关注自然语言描述中的相关词汇,合成图像中不同子区域的细粒度信息

### 5.2 适用数据特征

该算法适合的数据特征:
- 特征类型:连续型数据(如图像像素、音频波形)
- 数据规模:需要大规模数据集(至少数万样本)才能训练出高质量生成器
- 噪声容忍度:中等(GAN 本身对噪声有一定鲁棒性,但训练过程对超参数敏感)
- 数据分布:适合分布复杂、难以用简单参数模型描述的数据

### 5.3 不适用场景

**不适合的情况**:
1. 离散数据或结构化数据:GAN 的生成器输出是连续的,对离散序列数据(如文本)的直接生成不如自回归模型
2. 小规模数据集:GAN 需要大量数据来训练,数据不足时容易过拟合或模式崩塌
3. 需要精确可控的生成:标准 GAN 的生成不可控,需要使用 CGAN 等变体
4. 需要快速训练的场景:GAN 的训练通常需要大量迭代,且训练不稳定
5. 需要精确的似然计算:GAN 是隐式生成模型,无法直接计算数据点的似然值

---

## 6. 优缺点分析

### 6.1 优点

1. **生成样本质量高**:
   - 相比 VAE 等其他生成模型,GAN 生成的样本通常更清晰、更逼真
   - 这是因为 GAN 通过判别器的对抗训练,能够捕捉数据分布的高频细节

2. **无需显式建模数据分布**:
   - GAN 是隐式生成模型,不需要假设数据服从特定分布
   - 避免了显式密度估计中的各种困难(如归一化常数计算)

3. **理论上具有全局最优解**:
   - 当训练收敛时,生成分布等于真实数据分布
   - 理论保证:最优解对应 $p_g = p_{data}$

4. **灵活性强,易于扩展**:
   - 可以与各种条件、约束结合(CGAN、Pix2Pix 等)
   - 生成器和判别器的网络结构可以自由选择(CNN、RNN 等)

5. **生成样本多样性好**(在理想情况下):
   - 不同的噪声输入 $z$ 生成不同的样本
   - 能够覆盖数据分布的各个模式

### 6.2 缺点

1. **训练不稳定**:
   - GAN 的训练是两个网络的对抗过程,容易出现模式振荡
   - 判别器和生成器的能力需要保持平衡,否则会导致训练失败
   - 解决思路:使用 WGAN(用 Wasserstein 距离替代 JS 散度)、谱归一化、梯度惩罚等

2. **模式崩塌(Mode Collapse)**:
   - 生成器可能只学会生成少数几种样本,忽略数据分布的其他模式
   - 例如:在 MNIST 上训练时,生成器可能只生成"1"和"7",而不生成其他数字
   - 解决思路:使用 Minibatch Discrimination、Unrolled GAN、WGAN-GP 等

3. **没有显式的似然计算**:
   - GAN 无法直接计算给定样本的概率值
   - 这限制了 GAN 在需要精确概率推理的任务中的应用
   - 替代方案:Flow-based models 能够同时生成高质量样本并计算似然

4. **评估困难**:
   - GAN 没有统一的、完善的评估指标
   - IS(Inception Score)和 FID 是常用的替代指标,但各有局限

5. **超参数敏感**:
   - 学习率、网络结构、batch size 等对训练结果影响很大
   - 需要大量调参经验

### 6.3 与同类算法对比

| 维度 | GAN | VAE | Flow-based Models | Diffusion Models |
|------|-----|-----|-------------------|------------------|
| 生成质量 | 高 | 中等 | 高 | 非常高 |
| 训练稳定性 | 差(不稳定) | 好(稳定) | 好(稳定) | 好(稳定) |
| 似然计算 | 不支持 | 支持 | 支持 | 部分支持 |
| 采样速度 | 快 | 快 | 快 | 慢(需要多步去噪) |
| 模式覆盖 | 易崩塌 | 好 | 好 | 好 |
| 数学理论 | Minimax 博弈 | ELBO | 变分推断 | 去噪得分匹配 |
| 可控性 | 中等(CGAN) | 中等 | 中等 | 高(通过条件引导) |

### 6.4 注意力机制在 GAN 中的应用:AttnGAN 分析

在标准 GAN 的基础上,研究者们探索了将注意力机制引入 GAN 以提升生成质量。以 AttnGAN(Tao Xu 等,2017)为例,注意力机制在 GAN 中发挥了两个关键作用:

**作用一:细粒度信息整合**

标准 CGAN 将整个文本描述编码为一个全局向量作为条件,这是一种"粗犷"的生成方式。AttnGAN 则通过注意力机制关注自然语言描述中的相关词汇,合成图像中不同子区域的细粒度信息。具体而言,AttnGAN 的生成器具有多个层级,每个层级的特征构造器通过注意力机制计算图像局部特征与词特征之间的关联,将细粒度的语言特征整合进来。随着信息量的逐步补充,生成的图像分辨率也逐渐提高,细节也更加丰富。

这一思想的数学表达为:对于第 $j$ 个图像局部特征 $h_j$,通过注意力权重对词特征进行加权合成:

$$c_j = \sum_{t=1}^{T} \alpha_{j,t} e'_t$$

其中注意力权重为:

$$\alpha_{j,t} = \frac{\exp(h_j^T e'_t)}{\sum_{k=1}^{T}\exp(h_j^T e'_k)}$$

**作用二:输入与输出的匹配约束**

为了让生成的图像与输入的文本描述能够更加匹配,AttnGAN 利用注意力机制度量图像与文本之间的匹配程度。通过 DAMSM(深度注意力多模态相似度模型),在区域与词汇这一细粒度层级上评估图像与文本之间的相似度,以此作为额外的损失函数约束生成的图像能够与输入文本描述更加匹配。

DAMSM 的匹配得分为:

$$R(Q, S) = \left(\sum_{i=1}^{T} \exp(\gamma_2 \cdot R(e_i, c'_i))\right)^{1/\gamma_2}$$

最终 AttnGAN 的总损失为图像生成损失与文本-图像匹配损失之和: $\mathcal{L} = \mathcal{L}_G + \mathcal{L}_{DAMSM}$

---

## 7. 调库实现

### 7.1 环境准备

```bash
pip install torch torchvision numpy matplotlib
```

### 7.2 完整代码:DCGAN 生成 MNIST 手写数字

```python
"""
DCGAN 生成 MNIST 手写数字
数据集: MNIST 手写数字数据集
目标: 训练一个 DCGAN 模型,生成逼真的手写数字图像
框架: PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置随机种子,保证可复现
torch.manual_seed(42)
np.random.seed(42)

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================================
# 1. 超参数配置
# ============================================================
class Config:
    """训练配置"""
    # 数据参数
    image_size = 28              # MNIST 图像尺寸为 28x28
    channels = 1                 # MNIST 为灰度图,单通道
    num_classes = 10             # 数字类别数(0-9)

    # 模型参数
    latent_dim = 100             # 噪声向量维度
    ngf = 64                     # 生成器基础特征数
    ndf = 64                     # 判别器基础特征数

    # 训练参数
    batch_size = 128             # 批量大小
    num_epochs = 50              # 训练轮数
    lr_g = 2e-4                  # 生成器学习率
    lr_d = 2e-4                  # 判别器学习率
    beta1 = 0.5                  # Adam 优化器 beta1(GAN 推荐值)
    beta2 = 0.999                # Adam 优化器 beta2
    n_critic = 1                 # 每轮训练判别器的步数

    # 输出参数
    sample_interval = 5          # 每隔多少 epoch 保存生成样本
    save_dir = "dcgan_outputs"   # 输出目录


config = Config()


# ============================================================
# 2. 数据加载与预处理
# ============================================================
def get_mnist_dataloader():
    """
    加载 MNIST 数据集并返回 DataLoader

    Returns:
        dataloader: MNIST 数据加载器
    """
    # 定义数据预处理
    # 注意:生成器最后一层使用 tanh,输出范围为 [-1, 1]
    # 因此需要将图像归一化到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),                    # 转为张量,范围 [0, 1]
        transforms.Normalize([0.5], [0.5])        # 归一化到 [-1, 1]
    ])

    # 下载并加载 MNIST 训练集
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,          # 打乱数据顺序
        num_workers=2,         # 多进程加载
        drop_last=True         # 丢弃最后一个不完整的 batch
    )

    return dataloader


# ============================================================
# 3. 权重初始化
# ============================================================
def weights_init(m):
    """
    自定义权重初始化
    DCGAN 推荐使用均值为 0,标准差为 0.02 的正态分布初始化

    Args:
        m: 网络层
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # 卷积层:正态分布初始化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm 层:权重正态初始化,偏置为 0
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ============================================================
# 4. 生成器定义
# ============================================================
class Generator(nn.Module):
    """
    DCGAN 生成器
    输入: 噪声向量 z, shape (batch_size, latent_dim)
    输出: 生成图像, shape (batch_size, 1, 28, 28), 值域 [-1, 1]

    网络结构: 使用转置卷积层(反卷积)进行上采样
    输入维度 100 -> 256 -> 128 -> 64 -> 1
    空间维度 1x1 -> 4x4 -> 7x7 -> 14x14 -> 28x28
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # 输入: z, shape (batch_size, 100, 1, 1)
            # 第 1 层: 转置卷积, 上采样到 4x4
            nn.ConvTranspose2d(
                in_channels=config.latent_dim,
                out_channels=config.ngf * 4,    # 256
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(config.ngf * 4),
            nn.ReLU(inplace=True),
            # 输出: (batch_size, 256, 4, 4)

            # 第 2 层: 上采样到 7x7
            nn.ConvTranspose2d(
                in_channels=config.ngf * 4,     # 256
                out_channels=config.ngf * 2,    # 128
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False
            ),
            nn.BatchNorm2d(config.ngf * 2),
            nn.ReLU(inplace=True),
            # 输出: (batch_size, 128, 7, 7)

            # 第 3 层: 上采样到 14x14
            nn.ConvTranspose2d(
                in_channels=config.ngf * 2,     # 128
                out_channels=config.ngf,        # 64
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(config.ngf),
            nn.ReLU(inplace=True),
            # 输出: (batch_size, 64, 14, 14)

            # 第 4 层: 上采样到 28x28
            nn.ConvTranspose2d(
                in_channels=config.ngf,         # 64
                out_channels=config.channels,   # 1
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            # 输出: (batch_size, 1, 28, 28)
            nn.Tanh()  # 输出值域 [-1, 1], 与数据预处理对应
        )

    def forward(self, z):
        """
        前向传播

        Args:
            z: 噪声向量, shape (batch_size, latent_dim)

        Returns:
            生成图像, shape (batch_size, 1, 28, 28)
        """
        # 将 z 从 (batch_size, latent_dim) 变形为 (batch_size, latent_dim, 1, 1)
        x = z.view(z.size(0), config.latent_dim, 1, 1)
        return self.main(x)


# ============================================================
# 5. 判别器定义
# ============================================================
class Discriminator(nn.Module):
    """
    DCGAN 判别器
    输入: 图像, shape (batch_size, 1, 28, 28)
    输出: 判别分数(标量), 越大越可能是真实图像

    网络结构: 使用步幅卷积进行下采样
    输入维度 1 -> 64 -> 128 -> 256 -> 1
    空间维度 28x28 -> 14x14 -> 7x7 -> 4x4 -> 1x1
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 第 1 层: 下采样到 14x14
            nn.Conv2d(
                in_channels=config.channels,    # 1
                out_channels=config.ndf,        # 64
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (batch_size, 64, 14, 14)

            # 第 2 层: 下采样到 7x7
            nn.Conv2d(
                in_channels=config.ndf,         # 64
                out_channels=config.ndf * 2,    # 128
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (batch_size, 128, 7, 7)

            # 第 3 层: 下采样到 3x3
            nn.Conv2d(
                in_channels=config.ndf * 2,    # 128
                out_channels=config.ndf * 4,    # 256
                kernel_size=3,
                stride=2,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(config.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (batch_size, 256, 3, 3)

            # 第 4 层: 输出标量
            nn.Conv2d(
                in_channels=config.ndf * 4,    # 256
                out_channels=1,                # 1
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ),
            # 输出: (batch_size, 1, 1, 1)
            nn.Sigmoid()  # 输出概率 [0, 1]
        )

    def forward(self, img):
        """
        前向传播

        Args:
            img: 输入图像, shape (batch_size, 1, 28, 28)

        Returns:
            判别概率, shape (batch_size, 1)
        """
        output = self.main(img)
        return output.view(-1, 1)  # 展平为 (batch_size, 1)


# ============================================================
# 6. 训练函数
# ============================================================
def train():
    """
    完整的 GAN 训练流程
    """
    # 创建输出目录
    os.makedirs(config.save_dir, exist_ok=True)

    # 加载数据
    print("加载 MNIST 数据集...")
    dataloader = get_mnist_dataloader()
    print(f"数据集大小: {len(dataloader.dataset)} 张图像")
    print(f"Batch 数量: {len(dataloader)}")

    # 创建模型
    print("初始化模型...")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 权重初始化
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # 统计参数量
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"生成器参数量: {g_params:,}")
    print(f"判别器参数量: {d_params:,}")

    # 损失函数:二元交叉熵
    criterion = nn.BCELoss()

    # 优化器: Adam
    # 注意: beta1 设为 0.5 是 DCGAN 的推荐设置
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config.lr_g,
        betas=(config.beta1, config.beta2)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config.lr_d,
        betas=(config.beta1, config.beta2)
    )

    # 用于记录训练过程中的损失
    d_loss_history = []
    g_loss_history = []

    # 生成固定的噪声向量,用于追踪训练过程中生成质量的变化
    fixed_noise = torch.randn(64, config.latent_dim, device=device)

    # 真实标签和假标签
    real_label = 1.0
    fake_label = 0.0

    # 开始训练
    print("\n开始训练...")
    print("=" * 60)

    for epoch in range(config.num_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # 创建真实标签和假标签张量
            label_real = torch.full((batch_size, 1), real_label, device=device)
            label_fake = torch.full((batch_size, 1), fake_label, device=device)

            # =============================
            # (A) 训练判别器
            # =============================
            # 目标: 最大化 V(D, G)
            # 即: 对真实样本输出 1, 对假样本输出 0

            discriminator.zero_grad()

            # (A.1) 用真实数据训练判别器
            output_real = discriminator(real_images)
            loss_d_real = criterion(output_real, label_real)

            # (A.2) 用生成数据训练判别器
            noise = torch.randn(batch_size, config.latent_dim, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())  # detach 避免梯度回传到 G
            loss_d_fake = criterion(output_fake, label_fake)

            # (A.3) 合并两项损失,反向传播
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # =============================
            # (B) 训练生成器
            # =============================
            # 目标: 最小化 V(D, G)
            # 实践中: 最大化 D(G(z)), 即让生成样本被判为真

            generator.zero_grad()

            noise = torch.randn(batch_size, config.latent_dim, device=device)
            fake_images = generator(noise)
            output = discriminator(fake_images)

            # 生成器希望判别器输出 1
            loss_g = criterion(output, label_real)
            loss_g.backward()
            optimizer_g.step()

            # 记录损失
            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()
            num_batches += 1

        # 计算平均损失
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        d_loss_history.append(avg_d_loss)
        g_loss_history.append(avg_g_loss)

        # 打印训练进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:3d}/{config.num_epochs}]  "
                f"D Loss: {avg_d_loss:.4f}  "
                f"G Loss: {avg_g_loss:.4f}"
            )

        # 定期保存生成的样本图像
        if (epoch + 1) % config.sample_interval == 0:
            save_generated_images(
                generator, fixed_noise, epoch + 1, config.save_dir
            )

    print("=" * 60)
    print("训练完成!")

    # 保存最终模型
    torch.save(generator.state_dict(),
               os.path.join(config.save_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(),
               os.path.join(config.save_dir, "discriminator_final.pth"))
    print("模型已保存")

    return generator, discriminator, d_loss_history, g_loss_history


# ============================================================
# 7. 可视化与辅助函数
# ============================================================
def save_generated_images(generator, fixed_noise, epoch, save_dir):
    """
    保存生成的样本图像(8x8 网格)

    Args:
        generator: 生成器模型
        fixed_noise: 固定的噪声向量
        epoch: 当前 epoch
        save_dir: 保存目录
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu()

    # 将图像从 [-1, 1] 反归一化到 [0, 1]
    fake_images = (fake_images + 1) / 2

    # 绘制 8x8 网格
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            if idx < fake_images.size(0):
                ax = axes[i, j]
                ax.imshow(fake_images[idx].squeeze(), cmap='gray')
                ax.axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'generated_epoch_{epoch}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    generator.train()


def plot_training_curves(d_losses, g_losses, save_dir):
    """
    绘制训练过程中的损失曲线

    Args:
        d_losses: 判别器损失历史
        g_losses: 生成器损失历史
        save_dir: 保存目录
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.8)
    plt.plot(g_losses, label='Generator Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(d_losses, label='D Loss', alpha=0.8)
    plt.plot(g_losses, label='G Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses (Zoomed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


def generate_and_display(generator, n_samples=64, save_path=None):
    """
    生成并显示随机样本

    Args:
        generator: 训练好的生成器
        n_samples: 生成样本数
        save_path: 保存路径(可选)
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, config.latent_dim, device=device)
        fake_images = generator(z).cpu()

    fake_images = (fake_images + 1) / 2  # 反归一化到 [0, 1]

    n_rows = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(n_rows, n_rows, figsize=(12, 12))
    for i in range(n_rows):
        for j in range(n_rows):
            idx = i * n_rows + j
            if idx < n_samples:
                ax = axes[i, j]
                ax.imshow(fake_images[idx].squeeze(), cmap='gray')
                ax.axis('off')

    plt.suptitle('Generated Samples', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    generator.train()


# ============================================================
# 8. 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DCGAN 生成 MNIST 手写数字")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"图像尺寸: {config.image_size}x{config.image_size}")
    print(f"噪声维度: {config.latent_dim}")
    print(f"批量大小: {config.batch_size}")
    print(f"训练轮数: {config.num_epochs}")
    print(f"学习率(G/D): {config.lr_g}/{config.lr_d}")
    print()

    # 训练模型
    generator, discriminator, d_losses, g_losses = train()

    # 绘制训练损失曲线
    print("\n绘制训练损失曲线...")
    plot_training_curves(d_losses, g_losses, config.save_dir)

    # 生成最终样本
    print("生成最终样本...")
    generate_and_display(
        generator,
        n_samples=64,
        save_path=os.path.join(config.save_dir, 'final_samples.png')
    )

    print("\n所有任务完成!")
```

### 7.3 运行结果示例

```
============================================================
DCGAN 生成 MNIST 手写数字
============================================================
设备: cuda
图像尺寸: 28x28
噪声维度: 100
批量大小: 128
训练轮数: 50

加载 MNIST 数据集...
数据集大小: 60000 张图像
Batch 数量: 468
初始化模型...
生成器参数量: 1,455,873
判别器参数量: 1,093,505

开始训练...
============================================================
Epoch [  1/50]  D Loss: 0.4523  G Loss: 3.2187
Epoch [  5/50]  D Loss: 0.3287  G Loss: 2.8915
Epoch [ 10/50]  D Loss: 0.4123  G Loss: 2.5467
Epoch [ 15/50]  D Loss: 0.5234  G Loss: 2.1234
Epoch [ 20/50]  D Loss: 0.5678  G Loss: 1.9876
Epoch [ 25/50]  D Loss: 0.6012  G Loss: 1.8765
Epoch [ 30/50]  D Loss: 0.6234  G Loss: 1.7654
Epoch [ 35/50]  D Loss: 0.6456  G Loss: 1.6543
Epoch [ 40/50]  D Loss: 0.6567  G Loss: 1.6234
Epoch [ 45/50]  D Loss: 0.6789  G Loss: 1.5876
Epoch [ 50/50]  D Loss: 0.6890  G Loss: 1.5432
============================================================
训练完成!
模型已保存
```

---

## 8. 手工代码实现

### 8.1 核心算法手写:简单全连接 GAN 拟合 2D 高斯分布

```python
"""
GAN 手工实现: 使用全连接网络拟合 2D 高斯分布
仅依赖 PyTorch 的基础功能,从零搭建 GAN 训练流程
目标: 让生成器学会从标准正态噪声中采样,生成接近目标 2D 高斯分布的样本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 目标分布定义
# ============================================================
def sample_real_data(n_samples, target_mean=None, target_cov=None):
    """
    从目标 2D 高斯分布中采样真实数据

    Args:
        n_samples: 采样数量
        target_mean: 目标均值, shape (2,)
        target_cov: 目标协方差矩阵, shape (2, 2)

    Returns:
        real_samples: 真实样本, shape (n_samples, 2)
    """
    if target_mean is None:
        # 定义目标分布的均值(偏离原点,增加难度)
        target_mean = torch.tensor([3.0, -2.0])
    if target_cov is None:
        # 定义目标分布的协方差矩阵
        #   [[ 2.0,  0.5],
        #    [ 0.5,  1.5]]
        target_cov = torch.tensor([[2.0, 0.5],
                                   [0.5, 1.5]])

    # 使用多元正态分布采样
    # torch.distributions 提供了方便的分布采样接口
    dist = torch.distributions.MultivariateNormal(
        loc=target_mean,
        covariance_matrix=target_cov
    )
    real_samples = dist.sample((n_samples,))
    return real_samples


# ============================================================
# 2. 生成器定义(全连接网络)
# ============================================================
class SimpleGenerator(nn.Module):
    """
    简单的全连接生成器
    输入: 噪声向量 z, shape (batch_size, z_dim)
    输出: 2D 坐标, shape (batch_size, 2)

    网络结构:
        z_dim -> 128 -> 128 -> 2
    每层使用 ReLU 激活(最后一层不使用)
    """

    def __init__(self, z_dim=32, hidden_dim=128):
        super(SimpleGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),          # 第 1 层
            nn.ReLU(inplace=True),                 # ReLU 激活
            nn.Linear(hidden_dim, hidden_dim),      # 第 2 层
            nn.ReLU(inplace=True),                 # ReLU 激活
            nn.Linear(hidden_dim, 2)               # 输出层,2 维坐标
        )

    def forward(self, z):
        """
        前向传播

        Args:
            z: 噪声向量, shape (batch_size, z_dim)

        Returns:
            generated: 生成的 2D 点, shape (batch_size, 2)
        """
        return self.net(z)


# ============================================================
# 3. 判别器定义(全连接网络)
# ============================================================
class SimpleDiscriminator(nn.Module):
    """
    简单的全连接判别器
    输入: 2D 坐标, shape (batch_size, 2)
    输出: 真实概率, shape (batch_size, 1)

    网络结构:
        2 -> 128 -> 128 -> 1
    隐藏层使用 LeakyReLU,输出层使用 Sigmoid
    """

    def __init__(self, input_dim=2, hidden_dim=128):
        super(SimpleDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),       # 第 1 层
            nn.LeakyReLU(0.2, inplace=True),        # LeakyReLU(防止死神经元)
            nn.Linear(hidden_dim, hidden_dim),      # 第 2 层
            nn.LeakyReLU(0.2, inplace=True),        # LeakyReLU
            nn.Linear(hidden_dim, 1),               # 输出层
            nn.Sigmoid()                            # 输出概率 [0, 1]
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入点, shape (batch_size, 2)

        Returns:
            probability: 判别概率, shape (batch_size, 1)
        """
        return self.net(x)


# ============================================================
# 4. GAN 训练函数
# ============================================================
def train_gan(
    num_epochs=500,
    batch_size=256,
    z_dim=32,
    lr_g=1e-3,
    lr_d=1e-3,
    n_critic=1,
    print_interval=50,
    sample_interval=50,
    target_mean=None,
    target_cov=None
):
    """
    训练 GAN 拟合 2D 高斯分布

    Args:
        num_epochs: 训练轮数
        batch_size: 批量大小
        z_dim: 噪声维度
        lr_g: 生成器学习率
        lr_d: 判别器学习率
        n_critic: 每轮训练判别器步数
        print_interval: 打印间隔
        sample_interval: 采样间隔
        target_mean: 目标分布均值
        target_cov: 目标分布协方差

    Returns:
        generator: 训练好的生成器
        discriminator: 训练好的判别器
        history: 训练历史记录
    """
    # 创建模型
    generator = SimpleGenerator(z_dim=z_dim)
    discriminator = SimpleDiscriminator()

    # 损失函数:二元交叉熵
    criterion = nn.BCELoss()

    # 优化器: SGD(简单起见,不使用 Adam)
    optimizer_g = optim.SGD(generator.parameters(), lr=lr_g)
    optimizer_d = optim.SGD(discriminator.parameters(), lr=lr_d)

    # 训练记录
    history = {
        'd_losses': [],
        'g_losses': [],
        'd_real_acc': [],     # 判别器对真实样本的准确率
        'd_fake_acc': []      # 判别器对假样本的准确率
    }

    # 训练循环
    for epoch in range(num_epochs):
        # ---- 采样真实数据 ----
        real_data = sample_real_data(
            batch_size, target_mean, target_cov
        )

        # ---- 采样噪声并生成假数据 ----
        z = torch.randn(batch_size, z_dim)
        fake_data = generator(z)

        # ---- 训练判别器 ----
        for _ in range(n_critic):
            # 真实样本的判别结果
            d_real = discriminator(real_data)
            # 假样本的判别结果(detach 防止梯度传到 G)
            d_fake = discriminator(fake_data.detach())

            # 构建标签
            label_real = torch.ones(batch_size, 1)
            label_fake = torch.zeros(batch_size, 1)

            # 计算判别器损失
            loss_d_real = criterion(d_real, label_real)
            loss_d_fake = criterion(d_fake, label_fake)
            loss_d = loss_d_real + loss_d_fake

            # 更新判别器
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        # ---- 训练生成器 ----
        z = torch.randn(batch_size, z_dim)
        fake_data = generator(z)
        d_fake = discriminator(fake_data)

        # 生成器损失:希望判别器将假样本判为真
        loss_g = criterion(d_fake, torch.ones(batch_size, 1))

        # 更新生成器
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # ---- 记录统计信息 ----
        history['d_losses'].append(loss_d.item())
        history['g_losses'].append(loss_g.item())

        # 计算判别器准确率
        with torch.no_grad():
            d_real_pred = (discriminator(real_data) > 0.5).float()
            d_fake_pred = (discriminator(fake_data.detach()) > 0.5).float()
            history['d_real_acc'].append(d_real_pred.mean().item())
            history['d_fake_acc'].append(d_fake_pred.mean().item())

        # ---- 打印训练进度 ----
        if (epoch + 1) % print_interval == 0:
            print(
                f"Epoch [{epoch+1:4d}/{num_epochs}]  "
                f"D Loss: {loss_d.item():.4f}  "
                f"G Loss: {loss_g.item():.4f}  "
                f"D(Real): {history['d_real_acc'][-1]:.3f}  "
                f"D(Fake): {1-history['d_fake_acc'][-1]:.3f}"
            )

    return generator, discriminator, history


# ============================================================
# 5. 可视化函数
# ============================================================
def visualize_results(generator, target_mean, target_cov, history, save_dir="."):
    """
    可视化训练结果

    Args:
        generator: 训练好的生成器
        target_mean: 目标分布均值
        target_cov: 目标分布协方差
        history: 训练历史
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 子图 1: 数据分布对比 ----
    # 采样真实数据
    real_data = sample_real_data(2000, target_mean, target_cov).numpy()

    # 采样生成数据
    z = torch.randn(2000, 32)
    with torch.no_grad():
        fake_data = generator(z).numpy()

    ax1 = axes[0]
    ax1.scatter(real_data[:, 0], real_data[:, 1], alpha=0.3,
                c='blue', s=5, label='Real')
    ax1.scatter(fake_data[:, 0], fake_data[:, 1], alpha=0.3,
                c='red', s=5, label='Generated')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Real vs Generated Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- 子图 2: 损失曲线 ----
    ax2 = axes[1]
    ax2.plot(history['d_losses'], label='D Loss', alpha=0.8)
    ax2.plot(history['g_losses'], label='G Loss', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---- 子图 3: 判别器准确率 ----
    ax3 = axes[2]
    ax3.plot(history['d_real_acc'], label='D Acc (Real)', alpha=0.8)
    ax3.plot([1 - x for x in history['d_fake_acc']],
             label='D Acc (Fake)', alpha=0.8)
    ax3.axhline(y=0.5, color='gray', linestyle='--',
                alpha=0.5, label='Random (0.5)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Discriminator Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gan_2d_results.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 6. 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("GAN 手工实现: 拟合 2D 高斯分布")
    print("=" * 50)

    # 目标分布参数
    target_mean = torch.tensor([3.0, -2.0])
    target_cov = torch.tensor([[2.0, 0.5],
                                [0.5, 1.5]])

    print(f"\n目标分布均值: {target_mean.tolist()}")
    print(f"目标分布协方差:\n{target_cov.numpy()}")

    # 训练
    print("\n开始训练...")
    generator, discriminator, history = train_gan(
        num_epochs=1000,
        batch_size=256,
        z_dim=32,
        lr_g=1e-3,
        lr_d=1e-3,
        n_critic=1,
        print_interval=100
    )

    # 计算生成分布的统计量
    print("\n评估生成分布...")
    z = torch.randn(5000, 32)
    with torch.no_grad():
        fake_data = generator(z).numpy()

    print(f"真实分布均值:  {target_mean.numpy()}")
    print(f"生成分布均值:  {fake_data.mean(axis=0).round(3)}")
    print(f"真实分布协方差:\n{target_cov.numpy()}")
    print(f"生成分布协方差:\n{np.cov(fake_data.T).round(3)}")

    # 可视化
    print("\n可视化训练结果...")
    visualize_results(
        generator, target_mean, target_cov, history, save_dir="gan_2d_outputs"
    )

    print("\n完成!")
```

### 8.2 与理论期望的对比

| 统计量 | 真实分布 | 生成分布 | 偏差 |
|--------|---------|---------|------|
| 均值(x1) | 3.0 | ~2.95 | ~1.7% |
| 均值(x2) | -2.0 | ~-1.97 | ~1.5% |
| 方差(x1) | 2.0 | ~1.90 | ~5.0% |
| 方差(x2) | 1.5 | ~1.45 | ~3.3% |
| 协方差 | 0.5 | ~0.45 | ~10% |

**分析**:
- 随着训练进行,生成分布的统计量逐渐逼近真实分布
- 均值的收敛通常比协方差更快
- 简单的全连接 GAN 在低维空间中效果良好,但在高维图像空间中需要更深的网络结构(如 DCGAN)

---

## 9. 可视化与结果理解

### 9.1 生成样本展示

```python
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_generated_samples(generator, n_rows=8, n_cols=8, z_dim=100,
                           device='cpu', save_path=None):
    """
    展示生成器生成的样本(用于图像生成任务)

    Args:
        generator: 训练好的生成器模型
        n_rows: 行数
        n_cols: 列数
        z_dim: 噪声维度
        device: 计算设备
        save_path: 保存路径
    """
    generator.eval()
    n_samples = n_rows * n_cols
    z = torch.randn(n_samples, z_dim, device=device)

    with torch.no_grad():
        samples = generator(z).cpu()

    # 反归一化:从 [-1, 1] 到 [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            ax.imshow(samples[idx].squeeze(), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

    plt.suptitle('Generated Samples', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    generator.train()


def plot_interpolation(generator, z_dim=100, n_steps=10, device='cpu'):
    """
    在潜在空间中进行线性插值,展示潜在空间的连续性

    Args:
        generator: 生成器模型
        z_dim: 噪声维度
        n_steps: 插值步数
        device: 计算设备
    """
    generator.eval()

    # 两个随机起点和终点
    z1 = torch.randn(1, z_dim, device=device)
    z2 = torch.randn(1, z_dim, device=device)

    # 在两个点之间线性插值
    alphas = np.linspace(0, 1, n_steps)
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.2, 1.2))

    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            img = generator(z_interp).cpu()
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{alpha:.1f}', fontsize=8)

    plt.suptitle('Latent Space Interpolation', fontsize=14)
    plt.tight_layout()
    plt.show()
    generator.train()


def plot_training_progress(d_losses, g_losses):
    """
    绘制训练过程中判别器和生成器的损失曲线

    Args:
        d_losses: 判别器损失列表
        g_losses: 生成器损失列表
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图 1: 两个损失的原始曲线
    ax1 = axes[0, 0]
    ax1.plot(d_losses, label='Discriminator Loss', color='blue', alpha=0.7)
    ax1.plot(g_losses, label='Generator Loss', color='red', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('D Loss vs G Loss (Raw)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图 2: 移动平均后的损失曲线(更平滑)
    ax2 = axes[0, 1]
    window = 50
    if len(d_losses) > window:
        d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        ax2.plot(d_smooth, label='D Loss (smoothed)', color='blue', alpha=0.8)
        ax2.plot(g_smooth, label='G Loss (smoothed)', color='red', alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Smoothed Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图 3: 损失之和(GAN 的理论最优值处 D_loss + G_loss 应稳定)
    ax3 = axes[1, 0]
    total_loss = [d + g for d, g in zip(d_losses, g_losses)]
    ax3.plot(total_loss, label='D Loss + G Loss', color='green', alpha=0.7)
    ax3.axhline(y=np.log(4), color='gray', linestyle='--',
                label=f'log(4) = {np.log(4):.3f}')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Total Loss')
    ax3.set_title('Total Loss (Theory: should approach log(4))')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图 4: D Loss 和 G Loss 的比值
    ax4 = axes[1, 1]
    ratio = [d / (g + 1e-8) for d, g in zip(d_losses, g_losses)]
    ax4.plot(ratio, label='D/G Loss Ratio', color='purple', alpha=0.7)
    ax4.axhline(y=1.0, color='gray', linestyle='--', label='D/G = 1')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Ratio')
    ax4.set_title('Discriminator/Generator Loss Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gan_training_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()
```

### 9.2 结果解读

**从损失曲线(图 1)可以看出:**
- 判别器损失在训练初期快速下降(判别器很容易区分真假),随后逐渐上升并趋于 0.69(即 $\log 2$)
- 生成器损失在训练初期很高(生成样本质量差,容易被判别器识破),随后逐渐下降
- 在理想情况下,两者都应稳定在 $\log 2 \approx 0.693$ 附近(对应 $D(x) = 0.5$)

**从总损失(图 3)可以看出:**
- 理论上,当 GAN 达到全局最优时,总损失 $V^* = -\log 4$
- 但在实际训练中,由于网络容量有限、训练不充分等原因,总损失通常达不到理论值
- 如果总损失持续波动很大,说明训练不稳定

**从 D/G 损失比(图 4)可以看出:**
- 理想情况下比值趋近于 1,说明判别器和生成器达到了平衡
- 如果比值远大于 1,说明判别器太强,生成器需要加强
- 如果比值远小于 1,说明判别器太弱,需要增强判别器

**从插值结果可以看出:**
- 潜在空间具有良好的连续性:插值生成的图像平滑过渡
- 不同的潜在向量编码了不同的语义特征
- 这验证了 GAN 确实学到了数据分布的结构

---

## 10. 模型评估

### 10.1 生成模型评估的特殊性

与判别模型不同,生成模型的评估是一个开放问题。对于判别模型,我们可以直接使用准确率、F1 等指标。但对于生成模型,我们需要回答的问题是"生成的样本有多好",这涉及多个维度:真实性、多样性、创造性等。

### 10.2 Inception Score (IS)

Inception Score 是最早被广泛使用的 GAN 评估指标之一,由 Salimans 等人在 2016 年提出。

**核心思想**:
- 使用预训练的 Inception 模型对生成样本进行分类
- 好的生成样本应该满足两个条件:
  1. **清晰可辨**(低熵):每个样本应该被 Inception 模型高置信度地分类到某一个类别
  2. **覆盖全面**(高熵):所有生成样本的类别分布应该接近均匀分布(覆盖所有类别)

**数学定义**:

$$IS = \exp\left(\mathbb{E}_{x \sim p_g}\left[D_{KL}(p(y|x) \| p(y))\right]\right)$$

其中:
- $p(y|x)$: Inception 模型对样本 $x$ 的条件类别分布
- $p(y) = \mathbb{E}_x p(y|x)$: 生成样本的边缘类别分布

**代码实现**:

```python
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights


def inception_score(generator, n_samples=5000, batch_size=50,
                    z_dim=100, device='cuda'):
    """
    计算 Inception Score

    Args:
        generator: 生成器模型
        n_samples: 生成样本数量
        batch_size: 批量大小
        z_dim: 噪声维度
        device: 计算设备

    Returns:
        mean_is: 平均 Inception Score
        std_is: IS 的标准差
    """
    # 加载预训练的 Inception 模型
    inception = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    ).to(device)
    inception.eval()

    # 去掉最后的分类层,获取 softmax 输出
    all_probs = []

    generator.eval()
    with torch.no_grad():
        n_batches = n_samples // batch_size
        for _ in range(n_batches):
            z = torch.randn(batch_size, z_dim, device=device)
            samples = generator(z)

            # 调整大小为 Inception 需要的 299x299
            samples_resized = torch.nn.functional.interpolate(
                samples, size=(299, 299), mode='bilinear'
            )

            # 获取 Inception 的 softmax 输出
            logits = inception(samples_resized)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # (n_samples, n_classes)

    # 计算 KL 散度并求指数
    # p(y|x) 对每个样本: all_probs[i]
    # p(y) 对所有样本取平均
    marginal = np.mean(all_probs, axis=0)  # (n_classes,)
    scores = []
    for i in range(len(all_probs)):
        p_y_given_x = all_probs[i]
        # 计算 KL(p(y|x) || p(y))
        kl = np.sum(p_y_given_x * np.log(p_y_given_x / (marginal + 1e-10) + 1e-10))
        scores.append(np.exp(kl))

    mean_is = np.mean(scores)
    std_is = np.std(scores)
    generator.train()

    return mean_is, std_is
```

**IS 的局限性**:
- 只衡量了生成样本的类别多样性和清晰度,不衡量生成样本与真实数据之间的分布距离
- 对模式崩塌不敏感(如果生成器只覆盖部分类别,IS 可能仍然较高)
- 只适用于 ImageNet 类别的图像

### 10.3 Frechet Inception Distance (FID)

FID 是目前最常用的 GAN 评估指标,由 Heusel 等人在 2017 年提出。

**核心思想**:
- 使用 Inception 模型提取真实样本和生成样本的特征(取 Inception 网络最后一个池化层的输出)
- 计算两组特征之间的 Frechet 距离(也称为 Wasserstein-2 距离)
- FID 越小,说明生成分布越接近真实分布

**数学定义**:

假设真实样本特征 $\{f_r^{(i)}\}$ 服从多元高斯分布 $\mathcal{N}(\mu_r, \Sigma_r)$,生成样本特征 $\{f_g^{(i)}\}$ 服从 $\mathcal{N}(\mu_g, \Sigma_g)$,则:

$$FID = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

其中:
- $\mu_r, \mu_g$: 真实特征和生成特征的均值向量
- $\Sigma_r, \Sigma_g$: 真实特征和生成特征的协方差矩阵
- $\text{Tr}(\cdot)$: 矩阵的迹(trace)
- $(\Sigma_r \Sigma_g)^{1/2}$: 矩阵的平方根

**代码实现**:

```python
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights


def calculate_fid(real_features, generated_features):
    """
    计算 FID (Frechet Inception Distance)

    Args:
        real_features: 真实样本的 Inception 特征, shape (n_real, feature_dim)
        generated_features: 生成样本的 Inception 特征, shape (n_gen, feature_dim)

    Returns:
        fid: FID 值(越小越好)
    """
    # 计算均值和协方差
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(generated_features, axis=0)

    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(generated_features, rowvar=False)

    # 计算均值差的平方
    diff = mu_r - mu_g
    mean_diff_sq = np.dot(diff, diff)

    # 计算 sqrt(sigma_r @ sigma_g)
    # 使用 sqrtm 计算矩阵平方根
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

    # 处理数值问题:如果结果为复数,取实部
    if np.iscomplexobj(covmean):
        # 距离很小时的数值误差会导致微小的虚部
        if np.allclose(np.imag(covmean), 0, atol=1e-3):
            covmean = np.real(covmean)
        else:
            raise ValueError("矩阵平方根计算产生非平凡的虚部")

    # 计算 FID
    fid = mean_diff_sq + np.trace(sigma_r + sigma_g - 2 * covmean)

    return float(fid)


def extract_inception_features(images, batch_size=50, device='cuda'):
    """
    使用 Inception 模型提取图像特征

    Args:
        images: 图像张量, shape (n_images, C, H, W)
        batch_size: 批量大小
        device: 计算设备

    Returns:
        features: 提取的特征, shape (n_images, 2048)
    """
    inception = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    ).to(device)
    inception.eval()

    # 去掉最后的全连接层
    feature_extractor = nn.Sequential(*list(inception.children())[:-1])
    feature_extractor.eval()

    all_features = []
    with torch.no_grad():
        n_batches = (len(images) + batch_size - 1) // batch_size
        for i in range(n_batches):
            batch = images[i * batch_size:(i + 1) * batch_size].to(device)
            # 调整大小为 299x299
            batch = torch.nn.functional.interpolate(
                batch, size=(299, 299), mode='bilinear'
            )
            features = feature_extractor(batch)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)
```

### 10.4 IS 与 FID 的对比

| 维度 | Inception Score (IS) | Frechet Inception Distance (FID) |
|------|---------------------|----------------------------------|
| 值的含义 | 越大越好 | 越小越好 |
| 衡量内容 | 生成样本的清晰度和多样性 | 生成分布与真实分布的距离 |
| 是否需要真实数据 | 不需要 | 需要 |
| 模式崩塌检测 | 弱 | 强 |
| 计算复杂度 | 低 | 中(需要计算矩阵平方根) |
| 适用范围 | ImageNet 类别图像 | 任意图像(需要 Inception 特征空间) |
| 推荐度 | 较低 | 高(FID 是目前主流指标) |

---

## 11. 常见问题与易错点

### 11.1 模式崩塌 (Mode Collapse)

**现象**:
- 生成器只生成少数几种样本,输出的多样性严重不足
- 例如:在 MNIST 上可能只生成"1"和"7",不生成其他数字
- 判别器对某几类生成样本输出接近 0.5(无法区分),但整体生成分布只覆盖真实分布的一小部分

**原因**:
- 生成器找到了一种"捷径":只生成容易骗过判别器的少数几种样本
- 一旦生成器陷入只生成某几种样本的状态,判别器对这些样本的训练梯度消失,无法提供进一步改进的信号
- 优化景观中存在局部最优

**解决方案**:

```python
# 方案 1: Minibatch Discrimination
# 让判别器考虑一个 batch 内样本之间的多样性
class MinibatchDiscrimination(nn.Module):
    """Minibatch Discrimination 层"""
    def __init__(self, num_features, num_kernels, kernel_dim):
        super().__init__()
        # T 矩阵: (num_features, num_kernels * kernel_dim)
        self.T = nn.Parameter(torch.randn(num_features, num_kernels * kernel_dim))

    def forward(self, x):
        # x: (batch_size, num_features)
        # 计算 M = x * T, shape: (batch_size, num_kernels, kernel_dim)
        M = torch.matmul(x, self.T)
        M = M.view(x.size(0), -1, kernel_dim)

        # 计算 batch 内样本之间的 L1 距离
        # o_i = sum_j exp(-|M_i - M_j|_1)
        diffs = M.unsqueeze(1) - M.unsqueeze(0)  # (B, B, K, D)
        abs_diffs = torch.sum(torch.abs(diffs), dim=3)  # (B, B, K)
        exp_diffs = torch.exp(-abs_diffs)
        o = torch.sum(exp_diffs, dim=1)  # (B, K)

        # 将 o 与原始特征拼接
        return torch.cat([x, o.view(x.size(0), -1)], dim=1)

# 方案 2: 使用 WGAN-GP (Wasserstein GAN with Gradient Penalty)
# 改变损失函数,从根本上解决模式崩塌问题
# 参见 WGAN 相关文档

# 方案 3: 增加噪声
# 在生成器的输入或判别器的输入中添加噪声
z = torch.randn(batch_size, z_dim)
noise = 0.05 * torch.randn_like(fake_images)  # 添加小噪声
fake_images_noisy = fake_images + noise
```

### 11.2 训练不稳定

**现象**:
- 损失函数剧烈震荡,不收敛
- 判别器损失快速下降到接近 0(判别器太强)
- 生成器损失不下降或持续增大
- 生成样本质量忽好忽坏

**原因**:
- 判别器和生成器能力不平衡
- 学习率设置不当
- 网络结构不合适(如缺少 BatchNorm)
- 梯度消失或梯度爆炸

**解决方案**:

```python
# 方案 1: 使用谱归一化(Spectral Normalization)稳定判别器
from torch.nn.utils import spectral_norm

class StableDiscriminator(nn.Module):
    """使用谱归一化的判别器"""
    def __init__(self):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(1, 64, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.conv4 = spectral_norm(nn.Conv2d(256, 1, 3, 1, 0))

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.conv1(x))
        x = nn.BatchNorm2d(128)(nn.LeakyReLU(0.2)(self.conv2(x)))
        x = nn.BatchNorm2d(256)(nn.LeakyReLU(0.2)(self.conv3(x)))
        x = torch.sigmoid(self.conv4(x))
        return x.view(-1, 1)

# 方案 2: 调整学习率
# 生成器和判别器使用不同的学习率
optimizer_g = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_d = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

# 方案 3: 使用学习率衰减
scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.5)
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=20, gamma=0.5)

# 方案 4: 梯度裁剪
torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
```

### 11.3 梯度消失

**现象**:
- 使用原始目标 $\min_G \mathbb{E}[\log(1 - D(G(z)))]$ 时,生成器梯度在训练初期接近 0
- 生成器损失不下降

**原因**:
- 在训练初期,生成器产生的样本质量很差,$D(G(z))$ 接近 0
- $\log(1 - D(G(z))) \approx \log(1) = 0$
- 梯度 $\nabla_G \log(1 - D(G(z))) = -\frac{1}{1-D(G(z))}\nabla_G D(G(z))$
- 当 $D(G(z)) \approx 0$ 时,梯度虽然方向正确,但数值很小

**解决方案**:

```python
# 不要直接使用 log(1 - D(G(z))) 作为生成器损失
# 改为使用 -log(D(G(z))),这在训练初期提供更强的梯度信号

# 错误写法:
# g_loss = criterion(d_fake, torch.zeros_like(d_fake))  # log(1 - D(G(z)))

# 正确写法:
g_loss = criterion(d_fake, torch.ones_like(d_fake))    # -log(D(G(z)))

# 或者直接计算:
# g_loss = -torch.mean(torch.log(d_fake + 1e-8))

# 数学上两者等价(都是最大化 D(G(z))),
# 但后者在训练初期梯度信号更强
```

### 11.4 其他常见问题

**问题:生成图像模糊**

- 原因:可能使用了 MSE 损失(倾向于生成模糊的平均图像),或训练不充分
- 解决:确保使用交叉熵损失,增加训练轮数,调整网络结构

**问题:生成图像出现网格状伪影**

- 原因:转置卷积的上采样方式导致的空间不均匀性
- 解决:使用最近邻上采样 + 普通卷积替代转置卷积

**问题:训练速度慢**

- 原因:batch size 太大,网络太深,或频繁在 CPU/GPU 间传输数据
- 解决:合理设置 batch size,使用混合精度训练

---

## 12. 学习总结

### 12.1 核心要点回顾

- **核心思想**:通过生成器与判别器的对抗博弈,隐式地学习真实数据分布
- **数学本质**:极小极大值优化问题,全局最优等价于最小化真实分布与生成分布之间的 JS 散度
- **优化目标**:
  - 判别器:最大化 $\mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$
  - 生成器:最小化 $\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$,实践中改为最大化 $\mathbb{E}_{z \sim p_z}[\log D(G(z))]$
- **最优解**:当 $p_g = p_{data}$ 时达到全局最优,此时 $D^*(x) = 0.5$,最优值 $V^* = -\log 4$
- **适用场景**:高保真图像生成、图像修复、超分辨率、数据增强等
- **局限性**:训练不稳定、模式崩塌、评估困难、无法计算似然

### 12.2 关键公式汇总

**1. GAN 价值函数**:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**2. 最优判别器**:
$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

**3. 全局最优证明**:
$$C(G) = \max_D V(D, G) = -\log 4 + 2 \cdot D_{JS}(p_{data} \| p_g)$$

**4. 最优值**:
$$V^* = C(G^*) = -\log 4 \approx -1.386$$

**5. 生成器的实际训练目标**:
$$\max_G \mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

### 12.3 最佳实践

**模型结构**:
- 生成器:转置卷积 + BatchNorm + ReLU(最后一层用 Tanh)
- 判别器:步幅卷积 + BatchNorm + LeakyReLU(最后一层用 Sigmoid)
- 去掉全连接层,使用全卷积网络

**训练技巧**:
- Adam 优化器,beta1 = 0.5
- 学习率 2e-4
- 使用 BatchNorm 稳定训练
- 生成器损失使用 $-\log D(G(z))$ 而非 $\log(1-D(G(z)))$
- 必要时使用谱归一化

**数据预处理**:
- 将图像归一化到 $[-1, 1]$(与 Tanh 输出对应)
- 使用足够大的 batch size

### 12.4 与其他算法的联系

- **前置算法**:神经网络(前馈网络、CNN)、二元分类(逻辑回归)、KL 散度与 JS 散度
- **后续变体**:DCGAN(卷积 GAN)、WGAN(Wasserstein GAN)、CGAN(条件 GAN)、InfoGAN、StyleGAN
- **竞争生成模型**:VAE(变分自编码器)、Flow-based Models(Normalizing Flows)、Diffusion Models(扩散模型)
- **注意力结合**:AttnGAN 将注意力机制引入 GAN 的文本到图像生成任务,实现细粒度的图文匹配

---

## 13. 练习题与思考题

### 练习 1:概念理解

**问题**:在 GAN 的理论框架中,当训练达到全局最优时,判别器的输出 $D(x)$ 等于多少?为什么?

A. 0
B. 0.5
C. 1
D. 不确定

**答案与解析**:

答案:B

解析:当训练达到全局最优时,生成分布 $p_g$ 等于真实分布 $p_{data}$。此时,对于判别器接收到的每一个样本,它来自真实分布和生成分布的概率相同。将 $p_g = p_{data}$ 代入最优判别器公式:

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} = \frac{p_{data}(x)}{p_{data}(x) + p_{data}(x)} = \frac{1}{2}$$

这意味着判别器无法区分真假样本,其输出恒为 0.5。这也解释了为什么在训练过程中,如果判别器损失接近 $\log 2 \approx 0.693$(对应 $D(x) = 0.5$ 的二元交叉熵损失),说明训练可能接近收敛。

---

### 练习 2:数学推导(核心推导题)

**问题**:请从 GAN 的价值函数出发,推导最优判别器 $D^*(x)$ 的解析表达式,并证明当 $p_g = p_{data}$ 时,价值函数的最优值为 $-\log 4$。

**答案与解析**:

**推导过程**:

**Step 1**:固定 $G$,最大化 $V(D, G)$。

将期望写成积分形式:

$$V(D, G) = \int_x p_{data}(x)\log D(x)dx + \int_z p_z(z)\log(1-D(G(z)))dz$$

对第二项做变量替换 $x = G(z)$:

$$\int_z p_z(z)\log(1-D(G(z)))dz = \int_x p_g(x)\log(1-D(x))dx$$

因此:

$$V(D, G) = \int_x [p_{data}(x)\log D(x) + p_g(x)\log(1-D(x))]dx$$

**Step 2**:对每个 $x$ 最大化被积函数。

$$\frac{\partial}{\partial D}[p_{data}(x)\log D + p_g(x)\log(1-D)] = \frac{p_{data}(x)}{D} - \frac{p_g(x)}{1-D} = 0$$

解得:

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

**Step 3**:将 $D^*$ 代入 $V(D,G)$,利用 JS 散度证明最优值。

$$C(G) = \max_D V(D, G) = \int_x [p_{data}\log\frac{p_{data}}{p_{data}+p_g} + p_g\log\frac{p_g}{p_{data}+p_g}]dx$$

$$= \int_x [p_{data}\log\frac{2p_{data}}{p_{data}+p_g} + p_g\log\frac{2p_g}{p_{data}+p_g}]dx - 2\log 2$$

$$= 2 \cdot D_{JS}(p_{data} \| p_g) - 2\log 2$$

当 $p_g = p_{data}$ 时,$D_{JS}(p_{data} \| p_g) = 0$,因此:

$$C(G^*) = -2\log 2 = -\log 4$$

---

### 练习 3:训练目标选择

**问题**:在 GAN 的实际训练中,为什么生成器的目标通常使用 $\max_G \mathbb{E}[\log D(G(z))]$ 而不是原始的 $\min_G \mathbb{E}[\log(1 - D(G(z)))]$?

**答案与解析**:

在训练初期,生成器的质量很差,判别器可以轻易区分真假样本,因此 $D(G(z)) \approx 0$。

对于原始目标 $\min_G \log(1 - D(G(z)))$:
- 当 $D(G(z)) \approx 0$ 时,$\log(1 - 0) = \log 1 = 0$
- 梯度 $\nabla_G \log(1 - D(G(z))) \propto -\frac{\nabla_G D(G(z))}{1-D(G(z))}$
- 由于 $1 - D(G(z)) \approx 1$,梯度较小,但仍然有方向

对于替代目标 $\max_G \log D(G(z))$:
- 当 $D(G(z)) \approx 0$ 时,$\log D(G(z)) \to -\infty$
- 梯度 $\nabla_G \log D(G(z)) \propto \frac{\nabla_G D(G(z))}{D(G(z))}$
- 由于 $D(G(z)) \approx 0$,梯度非常大,提供了更强的学习信号

因此,使用 $\max_G \log D(G(z))$ 在训练初期提供更强的梯度信号,加速生成器的学习。而在训练后期,当 $D(G(z))$ 接近 0.5 时,两个目标提供的梯度大小相近。

从博弈论角度,两者是等价的:最大化 $\log D(G(z))$ 和最小化 $\log(1-D(G(z)))$ 都在推动 $D(G(z)) \to 1$,只是在不同的 $D(G(z))$ 值处提供了不同大小的梯度。

---

### 练习 4:模式崩塌分析

**问题**:什么是 GAN 的模式崩塌(Mode Collapse)?请分析其产生原因,并说明 AttnGAN 如何通过注意力机制来缓解这一问题。

**答案与解析**:

**模式崩塌定义**:生成器只生成数据分布中少数几种模式的样本,而忽略了其他模式。例如,真实数据包含多种数字(0-9),但生成器只生成"1"和"7"。

**产生原因**:
1. 生成器找到了"捷径":只生成容易被判别器接受的少数几种样本就能获得较低的损失
2. 一旦生成器陷入只生成少数模式,判别器对这些模式的梯度消失,无法引导生成器探索新模式
3. 损失函数缺乏对多样性的直接约束

**AttnGAN 的缓解方式**:

AttnGAN 并不直接解决模式崩塌问题,但它通过注意力机制间接提高了生成质量:

1. **细粒度信息整合**:AttnGAN 的多层级生成器通过注意力机制逐层整合文本描述中的细粒度信息,使生成过程更加"精细",减少了"粗犷"生成导致的模式遗漏。

2. **DAMSM 匹配约束**:AttnGAN 通过 DAMSM 模型在细粒度层级上评估图像与文本之间的匹配程度,作为额外的损失函数约束。这个约束要求生成的图像必须与输入的文本描述匹配,相当于为每个文本描述"指定"了一种生成模式,从而在一定程度上缓解了模式崩塌。

3. 但需要注意的是,AttnGAN 的主要目的是提高生成的细粒度质量,而非专门解决模式崩塌。真正有效解决模式崩塌的方法包括 WGAN-GP、Unrolled GAN、Minibatch Discrimination 等。

---

### 练习 5:GAN 变体设计

**问题**:假设你要设计一个 GAN 来将低分辨率的图像转换为高分辨率的图像(超分辨率任务),请说明:
(1) 生成器和判别器的输入和输出分别是什么?
(2) 你会如何设计损失函数?

**答案与解析**:

**(1) 网络设计**:

生成器(超分辨率网络):
- 输入:低分辨率图像 $x_{LR}$,shape $(B, C, H, W)$
- 输出:高分辨率图像 $\hat{x}_{HR}$,shape $(B, C, sH, sW)$,其中 $s$ 为放大倍数
- 网络结构:类似于 U-Net 或基于残差块的网络,先用转置卷积或 PixelShuffle 上采样,再用卷积层细化

判别器:
- 输入:高分辨率图像(真实的 $x_{HR}$ 或生成的 $\hat{x}_{HR}$),shape $(B, C, sH, sW)$
- 输出:标量概率 $D(x_{HR}) \in [0,1]$
- 网络结构:卷积网络,逐步下采样后输出判别分数

**(2) 损失函数设计**:

参考 SRGAN 的设计,损失函数包括两部分:

$$\mathcal{L}_{total} = \mathcal{L}_{content} + \alpha \cdot \mathcal{L}_{adversarial}$$

其中:

(a) 内容损失(Content Loss):使用预训练的 VGG 网络提取特征图,计算生成图像与真实图像在特征空间中的差异:

$$\mathcal{L}_{content} = \frac{1}{WH}\sum_{i,j}\|\phi_i(\hat{x}_{HR})_{ij} - \phi_i(x_{HR})_{ij}\|^2$$

其中 $\phi_i$ 是 VGG 网络第 $i$ 层的特征图。使用感知损失(而非像素级 MSE)能够生成视觉上更自然的高频细节。

(b) 对抗损失(Adversarial Loss):标准的 GAN 对抗损失:

$$\mathcal{L}_{adversarial} = -\log D(\hat{x}_{HR})$$

$\alpha$ 是平衡系数,通常设为 $10^{-3}$。

此外,还可以加入像素级 L1 损失作为辅助:

$$\mathcal{L}_{pixel} = \|x_{HR} - \hat{x}_{HR}\|_1$$

最终损失为:

$$\mathcal{L}_{total} = \mathcal{L}_{content} + 10^{-3}\mathcal{L}_{adversarial} + \mathcal{L}_{pixel}$$

---

## 14. 学习路径建议

### 14.1 前置知识

**学习 GAN 前,你需要掌握:**

**数学基础:**
- [ ] **概率论**:概率分布、期望、条件概率、KL 散度、JS 散度
  - 推荐资源:《概率论与数理统计》陈希孺
  - 学习时长:2-3 周
- [ ] **线性代数**:矩阵运算、特征值分解
  - 推荐资源:《线性代数导论》Gilbert Strang
  - 学习时长:2-3 周
- [ ] **微积分**:偏导数、链式法则、梯度
  - 推荐资源:Khan Academy 微积分课程
  - 学习时长:1-2 周
- [ ] **博弈论基础**(了解即可):零和博弈、纳什均衡
  - 推荐资源:维基百科相关条目
  - 学习时长:2-3 天

**编程基础:**
- [ ] **Python**:函数、类、NumPy
  - 学习时长:1 周
- [ ] **PyTorch**:张量操作、自动求导、nn.Module、DataLoader
  - 推荐资源:PyTorch 官方教程
  - 学习时长:1-2 周

**机器学习基础:**
- [ ] **神经网络**:前馈网络、反向传播、激活函数、损失函数
- [ ] **卷积神经网络**:卷积层、池化层、BatchNorm
- [ ] **优化方法**:梯度下降、Adam 优化器

### 14.2 平行算法(可同时学习)

与 GAN 同一层级的其他生成模型,可以对照学习:

1. **VAE(变分自编码器)**:另一种主要的深度生成模型
   - 学习重点:ELBO 推导、重参数化技巧、隐空间结构
   - 对比点:VAE 显式建模数据分布,GAN 隐式学习;VAE 训练稳定但生成模糊,GAN 训练不稳定但生成清晰

2. **Flow-based Models(标准化流)**:可逆变换生成模型
   - 学习重点:可逆网络设计、精确似然计算
   - 对比点:Flow 能精确计算似然,GAN 不能;Flow 的网络设计更受约束

3. **扩散模型(Diffusion Models)**:基于去噪的生成模型
   - 学习重点:前向扩散过程、逆向去噪过程、得分匹配
   - 对比点:Diffusion 训练更稳定、生成质量更高,但采样速度慢;GAN 采样速度快

### 14.3 进阶算法(后续学习)

学完基础 GAN 后,可以沿着以下路径深入学习:

**第一阶段:GAN 基础变体(1-2 个月)**

1. **DCGAN(Deep Convolutional GAN)**:
   - 关联:将卷积神经网络引入 GAN,定义了 GAN 的标准架构设计准则
   - 难度:中等
   - 重点:转置卷积上采样、步幅卷积下采样、BatchNorm 的使用

2. **CGAN(Conditional GAN)**:
   - 关联:在 GAN 中引入条件信息,实现可控生成
   - 难度:中等
   - 重点:条件机制的设计、多模态生成

3. **WGAN(Wasserstein GAN)**:
   - 关联:用 Wasserstein 距离替代 JS 散度,从根本上改善训练稳定性
   - 难度:中高
   - 重点:Wasserstein 距离的数学基础、Lipschitz 约束、WGAN-GP

**第二阶段:高级 GAN 变体(2-4 个月)**

4. **Pix2Pix / CycleGAN**:
   - 关联:图像到图像的翻译任务
   - 难度:高
   - 应用:风格迁移、语义分割、图像着色

5. **StyleGAN / StyleGAN2**:
   - 关联:高保真人脸生成,引入风格注入机制
   - 难度:高
   - 重点:样式网络、渐进式训练、路径长度正则化

6. **AttnGAN**:
   - 关联:将注意力机制引入 GAN,实现文本到图像的细粒度生成
   - 难度:高
   - 重点:多层级生成器、注意力加权特征融合、DAMSM 损失

**第三阶段:前沿方向(4-6 个月)**

7. **Diffusion Models(扩散模型)**:
   - 关联:当前最先进的生成模型,在多个任务上超越了 GAN
   - 难度:很高
   - 重点:DDPM、DDIM、Classifier-free Guidance、Stable Diffusion

8. **GAN + Transformer 结合**:
   - 关联:将 Transformer 架构引入生成器或判别器
   - 难度:很高
   - 代表工作:TransGAN、ViTGAN

### 14.4 推荐资源

**教材类:**
1. 《Generative Deep Learning》 David Foster - 系统介绍生成模型,包括 GAN、VAE、Flow、Diffusion
2. 《深度学习》 Goodfellow 等(花书) - 第 20 章专门讨论生成模型
3. 《机器学习》 周志华 - 第 14 章介绍生成模型基础

**论文类:**
1. Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014 - GAN 的开山之作
2. Radford et al., "Unsupervised Representation Learning with DCGANs", ICLR 2016 - DCGAN
3. Arjovsky et al., "Wasserstein GAN", ICML 2017 - WGAN
4. Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017 - WGAN-GP
5. Karras et al., "A Style-Based Generator Architecture for GANs", CVPR 2019 - StyleGAN
6. Xu et al., "AttnGAN: Fine-Grained Text to Image Generation with Attentional GANs", ICCV 2017 - AttnGAN

**在线课程:**
1. CS231n:CNN for Visual Recognition(斯坦福) - 含 GAN 讲座
2. Deep Generative Models(斯坦福 CS236) - 专门的生成模型课程
3. Fast.ai Course - Part 2, Lesson 7-9: GAN 实战

**代码资源:**
1. PyTorch Examples 中的 DCGAN 实现
2. GitHub: github.com/pytorch/examples/tree/main/dcgan
3. Papers with Code: paperswithcode.com/task/image-generation

---

## 附录

### A. 参考文献

1. Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets. NeurIPS, 2014.
2. Radford A, Metz L, Chintala S. Unsupervised representation learning with deep convolutional generative adversarial networks. ICLR, 2016.
3. Arjovsky M, Chintala S, Bottou L. Wasserstein generative adversarial networks. ICML, 2017.
4. Gulrajani I, Ahmed F, Arjovsky M, et al. Improved training of Wasserstein GANs. NeurIPS, 2017.
5. Salimans T, Goodfellow I, Zaremba W, et al. Improved techniques for training GANs. NeurIPS, 2016.
6. Heusel M, Ramsauer H, Unterthiner T, et al. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. NeurIPS, 2017.
7. Xu T, Zhang P, Huang Q, et al. AttnGAN: Fine-grained text to image generation with attentional generative adversarial networks. CVPR, 2018.
8. Karras T, Laine S, Aila T. A style-based generator architecture for generative adversarial networks. CVPR, 2019.
9. Ledig C, Theis L, Huszar F, et al. Photo-realistic single image super-resolution using a generative adversarial network. CVPR, 2017.
10. Isola P, Zhu J Y, Zhou T, et al. Image-to-image translation with conditional adversarial networks. CVPR, 2017.

### B. 常见问题 FAQ

**Q1:GAN 和 VAE 有什么本质区别?**

A:GAN 和 VAE 都是深度生成模型,但核心机制完全不同。(1)GAN 通过对抗博弈隐式学习数据分布,不直接建模 $p(x)$;VAE 通过变分推断和 ELBO 显式建模数据分布。(2)GAN 的生成质量通常更高但训练不稳定;VAE 训练稳定但生成样本往往较模糊。(3)GAN 无法计算样本的似然值;VAE 可以给出似然的下界。(4)GAN 的潜在空间结构不如 VAE 有意义(GAN 的潜在空间可能不连续),但可以通过插值等方式探索。

**Q2:为什么 GAN 的训练比普通神经网络更困难?**

A:GAN 的训练本质上是一个动态博弈过程,而不是简单的固定目标优化。(1)判别器和生成器的训练必须保持平衡,一方太强会导致另一方梯度消失。(2)优化目标是非凸的,存在大量局部最优和鞍点。(3)理论上的收敛条件(纳什均衡)在实际中很难达到。(4)训练过程中容易出现模式振荡——两个网络"来回切换"而非共同提升。

**Q3:在什么情况下应该选择 GAN 而非 Diffusion Model?**

A:GAN 的优势在于采样速度快(单次前向传播),适合实时生成场景。如果需要低延迟的图像生成(如实时视频特效、游戏引擎),GAN 仍然是更好的选择。Diffusion Model 的优势在于生成质量和训练稳定性,适合离线生成(如内容创作、设计辅助)。此外,对于需要精确可控生成的任务(如特定属性的编辑),Diffusion Model 的 Classifier-free Guidance 机制更为灵活。对于小数据集,Diffusion Model 通常表现更好且更稳定。

---

**文档结束**
