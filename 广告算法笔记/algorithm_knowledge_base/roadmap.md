# 学习路线图

## 初级（回归、分类、基础模型）

### 统计学习基础
- 线性回归、岭回归、LASSO回归、多项式线性回归
- 逻辑回归（二项/多项式）
- 朴素贝叶斯、KNN、k-D tree

### 决策树系列
- 决策树基础 → ID3 → C4.5 → CART

### 感知机与神经网络基础
- 感知机、多层感知机（MLP）
- 前馈神经网络、反向传播算法

## 中级（特征交互、序列建模、多任务学习）

### 特征交叉模型
- FM → FFM → Wide & Deep → DeepFM → DCN

### 序列建模
- RNN → LSTM → GRU → DRNN
- Attention 机制 → Transformer

### 用户兴趣建模
- DIN（Target Attention）→ DIEN（GRU + Attention）
- BST（Behavior Sequence Transformer）

### 多任务学习
- Shared-Bottom → MMoE → PLE → ESMM

### 召回与粗排
- DSSM（双塔模型）→ COLD

## 高级（强化学习、生成式模型、前沿技术）

### 强化学习基础
- MDP、蒙特卡洛预测、TD、SARSA、Q-learning

### 深度强化学习
- DQN → DDPG → TD3
- REINFORCE → PPO → A2C → SAC → ACER

### 广告出价算法
- PID（第一代）→ MPC（第二代）→ RL 出价（第三代）→ 生成式 RL（第四代）

### Bandit 与探索
- Multi-armed Bandits → UCB → Thompson Sampling

### 生成式模型
- AE → VAE → DAE
- GAN → DCGAN
- DDPM → DM → SMLD
- Unet

### NLP 与表示学习
- One Hot → TF-IDF → word2vec → char2vec → glove
- GPT → BERT
- Encoder-Decoder → MHA → Transformer

### 降维与主题模型
- PCA → LDA → SVD
- LSA → NMF → PLSA
- EM → 变分EM → 高斯混合EM
- 马尔可夫链蒙特卡洛（MCMC）

### 概率图模型
- 隐马尔可夫（HMM）
- 条件随机场（CRF）
- 最大熵模型

### 集成学习
- AdaBoost → GBDP

### 深度学习进阶
- 卷积神经网络（CNN）
- 残差神经网络（ResNet）
- RNN-Search

## 广告系统专项

### 定价机制
- GSP（广义第二价格）→ VCG

### 广告系统核心
- CTR 预估、CVR 预估
- 预算控制、流量调控
- 冷启动策略
