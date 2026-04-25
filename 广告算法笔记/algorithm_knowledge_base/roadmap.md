# 学习路线图

> 按「初级 → 中级 → 高级」的层级逐步学习，箭头表示推荐的学习先后顺序。

---

## 初级（回归、分类、基础模型）

### 统计学习基础
- 线性回归 → 岭回归 → LASSO回归 → 多项式线性回归
- 朴素贝叶斯 → KNN → k-D tree
- 支持向量机（SVM）

### 分类模型
- 感知机 → 多层感知机（MLP）
- 二项逻辑回归 → 多项式逻辑回归
- 最大熵模型

### 决策树系列
- 决策树基础 → ID3 → C4.5 → CART

### NLP 基础
- one hot → TF-IDF → word2vec → char2vec → glove

### 神经网络基础
- 前馈神经网络 → 反向传播算法

---

## 中级（集成学习、序列建模、特征交互、无监督学习）

### 集成学习
- AdaBoost → GBDT（梯度提升决策树）

### 深度学习基础
- 卷积神经网络（CNN）→ 残差神经网络（ResNet）
- RNN → LSTM → GRU → DRNN
- RNN-Search

### 注意力与序列建模
- Attention 机制 → Encoder-Decoder → MHA → Transformer

### 无监督学习
- K-Means（聚类）
- PCA → LDA → 奇异值分解（SVD）（降维）
- LSA → NMF → PLSA（主题模型）
- EM → 变分EM → 高斯混合EM
- 马尔可夫链蒙特卡洛（MCMC）

### 概率图模型
- 隐马尔可夫（HMM）
- 条件随机场（CRF）

### NLP 进阶
- GPT → Bert

### 广告系统 — 特征交叉模型
- FFM → Wide & Deep → DeepFM → DCN

### 广告系统 — 用户兴趣与召回
- DIN（Target Attention）→ DSSM（双塔模型）→ COLD

---

## 高级（强化学习、生成式模型、广告系统进阶）

### 强化学习基础
- MDP → multi-armed bandits
- 蒙特卡洛预测 → TD → SARSA → Q-learning

### 深度强化学习
- DQN → DDPG → TD3
- REINFORCE → PPO → A2C → ACER → SAC

### Bandit 与探索
- Bandit → UCB → Thompson Sampling

### 生成式模型
- AE → VAE → DAE
- GAN → DCGAN
- DDPM → DM → SMLD
- Unet

### 广告系统 — 定价机制
- GSP（广义第二价格）→ VCG

### 广告系统 — 出价算法
- PID（第一代）→ MPC（第二代）→ RL 出价（第三代）
