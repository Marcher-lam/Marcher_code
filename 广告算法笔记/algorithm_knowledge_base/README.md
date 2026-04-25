# 算法知识库

> 一套系统化的机器学习与广告算法学习文档库

## 项目介绍

本知识库涵盖 88 个算法，从经典机器学习到前沿深度学习，从通用算法到广告系统专有技术。每个算法都有独立的 Markdown 文档，遵循统一的 14 节结构。

## 文档结构

- `algorithms/` — 88 个算法的独立学习文档
- `utils/metrics.md` — 评估指标详解（回归/分类/广告系统）
- `utils/optimization.md` — 优化方法详解（梯度下降/正则化/学习率策略）
- `roadmap.md` — 学习路径指南

## 使用方式

1. 参考 `roadmap.md` 选择学习阶段
2. 在 `algorithms/` 目录下找到对应算法的 `.md` 文件
3. 按以下顺序阅读每个文档：
   - 算法基础认知 → 核心原理 → 数学公式 → 训练过程
   - 调库实现 → 手工代码 → 可视化 → 模型评估
   - 常见问题 → 学习总结 → 练习题

## 算法分类

### 传统机器学习（17个）
线性回归、岭回归、LASSO回归、多项式线性回归、感知机、多层感知机、KNN、k-D tree、朴素贝叶斯、决策树、ID3、C4.5、CART、支持向量机、二项逻辑回归、多项式逻辑回归、最大熵模型

### 集成学习（2个）
AdaBoost、GBDT

### 概率图模型（2个）
隐马尔可夫、条件随机场

### 无监督学习（11个）
K-Means、奇异值分解、PCA、LDA、EM、变分EM、高斯混合EM、马尔可夫链蒙特卡洛、LSA、NMF、PLSA

### 深度学习基础（10个）
前馈神经网络、反向传播算法、卷积神经网络、残差神经网络、RNN、LSTM、GRU、DRNN、RNN-Search、Unet

### 注意力与Transformer（4个）
Attention机制、Encoder-Decoder、MHA、Transformer

### NLP与表示学习（7个）
one hot、TF-IDF、word2vec、char2vec、glove、GPT、Bert

### 生成模型（8个）
AE、VAE、DAE、GAN、DCGAN、DDPM、DM、SMLD

### 强化学习（14个）
MDP、multi-armed bandits、蒙特卡洛预测、TD、SARSA、Q-learning、DQN、DDPG、REINFORCE、PPO、A2C、ACER、SAC、TD3

### 广告系统专项（13个）
PID、MPC、GSP、VCG、Bandit、UCB、Thompson Sampling、FFM、Wide & Deep、DCN、DIN、DSSM、COLD

## 前置知识

- 线性代数（矩阵运算、特征值分解）
- 概率论与统计（贝叶斯、分布、期望）
- 微积分（偏导数、链式法则）
- Python 编程基础

## 文档统计

- 算法文档：88 个
- 工具文档：2 个
- 每个文档统一 14 节结构
