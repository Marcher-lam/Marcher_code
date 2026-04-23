import os, re, textwrap

# Keyword categories for algorithm detection
REGRESSION_KW = ['回归', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'PolynomialRegression']
CLASSIFICATION_KW = ['分类', 'Logistic', 'SVM', 'DecisionTree', 'RandomForest', 'AdaBoost', 'GBDT', 'XGBoost', 'LightGBM', 'CatBoost', 'KNN', '朴素贝叶斯', 'NaiveBayes']
CLUSTERING_KW = ['K-Means', 'DBSCAN', '层次聚类', 'MeanShift', 'Spectral', 'OPTICS']
DIMRED_KW = ['PCA', 'LDA', 'SVD', 'NMF', 'LSA', 't-SNE', 'UMAP']
DEEP_KW = ['神经网络', 'CNN', 'RNN', 'LSTM', 'GRU', 'Transformer', 'GAN', 'VAE', 'AE', 'BERT', 'GPT', 'T5', 'BART', 'DistilBERT', 'ALBERT', 'ViT', 'UNet', 'U-Net', 'ResNet', 'ResNeXt', 'EfficientNet']
RL_KW = ['DQN', 'DDPG', 'PPO', 'A2C', 'SAC', 'TD3', 'REINFORCE', 'SARSA', 'Q-learning', 'Actor', 'Critic']

def detect_category(name):
    lower = name.lower()
    for kw in REGRESSION_KW:
        if kw.lower() in lower:
            return 'regression'
    for kw in CLASSIFICATION_KW:
        if kw.lower() in lower:
            return 'classification'
    for kw in CLUSTERING_KW:
        if kw.lower() in lower:
            return 'clustering'
    for kw in DIMRED_KW:
        if kw.lower() in lower:
            return 'dimred'
    for kw in DEEP_KW:
        if kw.lower() in lower:
            return 'deep'
    for kw in RL_KW:
        if kw.lower() in lower:
            return 'rl'
    return 'generic'

def lib_impl(name, cat):
    if cat == 'regression':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 回归实现示例（LinearRegression）
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        print('R^2:', model.score(X_test, y_test))
        ```
        """)
    if cat == 'classification':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 分类实现示例（LogisticRegression）
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        print('Accuracy:', model.score(X_test, y_test))
        ```
        """)
    if cat == 'clustering':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 聚类实现示例（KMeans）
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        print('Cluster centers:', model.cluster_centers_)
        ```
        """)
    if cat == 'dimred':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 降维实现示例（PCA）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        print('Explained variance:', pca.explained_variance_ratio_)
        ```
        """)
    if cat == 'deep':
        return textwrap.dedent(f"""
        ```python
        # PyTorch 简单模型示例（全连接网络）
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, output_dim)
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))
        model = Net(input_dim=100, hidden_dim=50, output_dim=10)
        print(model)
        ```
        """)
    if cat == 'rl':
        return textwrap.dedent(f"""
        ```python
        # OpenAI Gym 环境示例（CartPole）\nimport gym\nenv = gym.make('CartPole-v1')\nobs = env.reset()\nfor _ in range(200):\n    action = env.action_space.sample()\n    obs, reward, done, info = env.step(action)\n    env.render()\n    if done:\n        break\nenv.close()\n        ```
        """)
    return "\n```python\n# TODO: 根据实际算法补充库实现代码\n```"

def hand_impl(name, cat):
    class_name = re.sub(r'[^A-Za-z0-9]', '', name)
    if cat in ('regression', 'classification'):
        return textwrap.dedent(f"""
        ```python
        # 手工实现示例（结构模板）\nimport numpy as np\n\nclass {class_name}:\n    def __init__(self, *args, **kwargs):\n        pass\n    def fit(self, X, y):\n        # TODO: 实现训练过程\n        pass\n    def predict(self, X):\n        # TODO: 实现预测过程\n        return np.zeros(len(X))\n        ```
        """)
    if cat == 'deep':
        return textwrap.dedent(f"""
        ```python
        # 手工实现深度模型简化示例（使用 numpy）\nimport numpy as np\n\nclass SimpleNN:\n    def __init__(self, input_dim, hidden_dim, output_dim):\n        self.W1 = np.random.randn(input_dim, hidden_dim)\n        self.b1 = np.zeros(hidden_dim)\n        self.W2 = np.random.randn(hidden_dim, output_dim)\n        self.b2 = np.zeros(output_dim)\n    def forward(self, x):\n        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU\n        return h @ self.W2 + self.b2\n        ```
        """)
    return "\n```python\n# TODO: 根据实际算法补充手工实现代码\n```"

def visualization(name, cat):
    if cat in ('regression', 'classification'):
        return textwrap.dedent(f"""
        ```python
        # 可视化示例（散点图或决策边界）\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nX = np.random.randn(200, 2)\ny = np.random.randint(0, 2, 200)\nplt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')\nplt.title('{name} 可视化示例')\nplt.show()\n        ```
        """)
    if cat == 'dimred':
        return textwrap.dedent(f"""
        ```python
        # 降维后可视化（2D）\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nX_reduced = np.random.randn(200, 2)\nplt.scatter(X_reduced[:,0], X_reduced[:,1], cmap='plasma')\nplt.title('{name} 降维可视化')\nplt.show()\n        ```
        """)
    if cat == 'deep':
        return textwrap.dedent(f"""
        ```python
        # 特征图可视化示例\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nfeature = np.random.rand(8, 8)\nplt.imshow(feature, cmap='gray')\nplt.title('{name} 特征图')\nplt.show()\n        ```
        """)
    return "\n- TODO: 添加可视化示例\n"

def evaluation(name, cat):
    if cat == 'regression':
        metric = 'mean_squared_error'
    elif cat == 'classification':
        metric = 'accuracy_score'
    elif cat == 'clustering':
        metric = 'silhouette_score'
    elif cat == 'dimred':
        metric = 'explained_variance_score'
    elif cat == 'rl':
        metric = 'episode_reward'
    else:
        metric = 'accuracy_score'
    return textwrap.dedent(f"""
        ```python
        # 评估示例\nfrom sklearn.metrics import {metric}\n# y_true, y_pred 需自行准备\n# print('{metric}:', {metric}(y_true, y_pred))\n        ```
        """)

def common_issues():
    return textwrap.dedent("""
    - 未对特征进行标准化或归一化导致模型不收敛。\n- 超参数（学习率、正则化、层数）需要调参。\n- 过拟合：模型在训练集表现好但在测试集表现差。\n- 计算资源：深度模型常需 GPU 加速。\n""")

def learning_summary(name):
    return f"**学习要点**：{name} 的核心思想是 …（请根据实际算法补充）。掌握其数学推导、实现细节以及适用场景是后续深入学习的基础。"

def exercises(name):
    return textwrap.dedent(f"""
    1. 手动实现 {name} 的核心迭代步骤，并在合成数据上验证。\n2. 使用不同库（如 scikit‑learn 与 PyTorch）实现，并比较训练时间与精度。\n3. 设计可视化函数，展示 {name} 在不同超参数下的表现。\n""")

def learning_path():
    return textwrap.dedent("""
    - 先掌握线性模型（线性回归、逻辑回归）→\n- 再学习树模型（决策树、随机森林、XGBoost）→\n- 深入深度学习模型（CNN、Transformer、GAN）→\n- 进阶章节：自监督学习、强化学习、生成模型等前沿方向。\n""")

# Mapping from section number to generator function
SECTION_FUNCS = {
    7: lib_impl,
    8: hand_impl,
    9: visualization,
    10: evaluation,
    11: lambda n, c: common_issues(),
    12: lambda n, c: learning_summary(n),
    13: lambda n, c: exercises(n),
    14: lambda n, c: learning_path(),
}

def replace_section(content, heading_num, name, cat):
    # pattern matches heading line and everything until next heading (## <digit>. ) or EOF
    pattern = rf'(##\s+{heading_num}\.\s+[^\n]*\n)(.*?)(?=\n##\s+\d+\.|\Z)'
    repl_text = SECTION_FUNCS[heading_num](name, cat)
    # Ensure a blank line after heading then the content
    replacement = rf"\1\n{repl_text}\n"
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    return new_content, count

def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    name = os.path.splitext(os.path.basename(path))[0]
    cat = detect_category(name)
    changed = False
    for num in range(7, 15):
        content, cnt = replace_section(content, num, name, cat)
        if cnt:
            changed = True
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    alg_dir = 'algorithm_knowledge_base/algorithms'
    for fname in os.listdir(alg_dir):
        if fname.endswith('.md'):
            process_file(os.path.join(alg_dir, fname))

if __name__ == '__main__':
    main()
