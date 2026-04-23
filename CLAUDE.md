你是一位顶级算法专家 + 工程架构师 + 技术文档专家。

现在请你执行一个“算法知识库自动生成任务”。

---

# 🎯 任务目标
为多个算法自动生成系统化学习文档，并构建一个完整的算法知识库（Markdown 格式）。

---

# 📦 输入
算法列表 = [线性回归，岭回归，LASSO回归，多项式线性回归，感知机，多层感知机，KNN，k-D tree，朴素贝叶斯，决策树，ID3，C4.5，CART，逻辑回归，二项逻辑回归，多项式逻辑回归，最大熵模型，支持向量机，AdaBoost，GBDP，隐马尔可夫，条件随机场，K-Means，奇异值分解，PCA，LDA，EM，变分EM，高斯混合EM，马尔可夫链蒙特卡洛，LSA，NMF，PLSA，前馈神经网络，反向传播算法，卷积神经网络，残差神经网络，RNN，LSTM，GRU，DRNN，RNN-Search，Attention机制，Encoder-Decoder，MHA，Transformer，one hot，TF-IDF，word2vec，char2vec，glove，GPT，Bert，AE，VAE，DAE，GAN，DCGAN，DDPM，DM，SMLD，Unet，MDP，multi-armed bandits，UCB，Thompson Sampling，蒙特卡洛预测，TD，SARSA，Q-learing，DQN，REINFORCE，PPO，A2C，DDPG，ACER，SAC，TD3]

---

# 📁 输出要求（非常重要）

请你模拟“工程项目结构”，输出如下内容：

algorithm_knowledge_base/
├── README.md
├── roadmap.md
├── algorithms/
│   ├── 线性回归.md
│   ├── 逻辑回归.md
│   ├── ...
├── utils/
│   ├── metrics.md
│   ├── optimization.md

---

# 📄 各文件生成规则

## 1️⃣ README.md
包含：
- 项目介绍（这是一个算法学习知识库）
- 使用方式
- 学习路径说明
- 如何阅读这些文档

---

## 2️⃣ roadmap.md
给出算法学习路径，例如：
- 初级（回归、分类）
- 中级（树模型、聚类、降维）
- 高级（深度学习、Transformer）

---

## 3️⃣ 每个算法.md（核心）

对于【算法列表】中的每一个算法，分别生成一个 Markdown 文档，必须包含以下结构：

# <算法名称> 学习文档

## 1. 算法基础认知
## 2. 核心原理
## 3. 数学公式与推导
## 4. 训练过程讲解
## 5. 应用场景
## 6. 优缺点分析
## 7. 调库实现（Python + 完整代码 + 注释）
## 8. 手工代码实现（核心算法手写 + 注释）
## 9. 可视化与结果理解
## 10. 模型评估
## 11. 常见问题与易错点
## 12. 学习总结
## 13. 练习题与思考题（含答案）
## 14. 学习路径建议

---

# ⚙️ 统一规范（非常关键）

所有算法文档必须满足：

1. 面向初学者（但不降低深度）
2. 必须解释“为什么”，不能只写结论
3. 数学推导要清晰，不跳步
4. 代码必须：
    - 可运行
    - 带详细注释
5. 优先使用：
    - scikit-learn（传统算法）
    - PyTorch（深度学习）
6. 风格统一（像一本书，而不是拼凑内容）

---

# 🧩 utils 文件

## metrics.md
讲解：
- MSE / RMSE / MAE / R²
- Precision / Recall / F1
- AUC / ROC

## optimization.md
讲解：
- 梯度下降
- 学习率
- 正则化（L1 / L2）
- 过拟合 vs 欠拟合

---

# 🚀 执行方式

请你：

1️⃣ 遍历算法列表  
2️⃣ 为每个算法生成完整 Markdown 内容  
3️⃣ 按“项目结构”输出  
4️⃣ 所有内容一次性生成

---

# ❗ 输出格式要求

请严格按照如下格式输出：

【文件路径】
<文件内容>

例如：

【algorithm_knowledge_base/README.md】
内容...

【algorithm_knowledge_base/algorithms/线性回归.md】
内容...

---

# 🔥 目标

最终输出应当是：
👉 一套可以直接复制到本地使用的完整算法知识库  
👉 每个算法都是高质量教学文档  
👉 整体结构清晰、统一、专业

开始执行。