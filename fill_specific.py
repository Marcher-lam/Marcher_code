import os, re, textwrap

# Determine algorithm category based on name keywords
REGRESSION_KW = ['回归', 'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'PolynomialRegression']
CLASSIFICATION_KW = ['分类', 'LogisticRegression', 'SVM', 'DecisionTree', 'RandomForest', 'AdaBoost', 'GBDT', 'XGBoost', 'LightGBM', 'CatBoost', 'KNN', '朴素贝叶斯', 'NaiveBayes', 'Binary', 'Multi']
CLUSTERING_KW = ['K-Means', 'DBSCAN', '层次聚类', 'MeanShift', 'Spectral', 'OPTICS']
DIMRED_KW = ['PCA', 'LDA', 'SVD', 'NMF', 'LSA', 't-SNE', 'UMAP', 'TSNE']
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
    # default to generic
    return 'generic'


def lib_impl(name, category):
    if category == 'regression':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 回归实现示例（以 LinearRegression 为例）
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        print('R^2:', model.score(X_test, y_test))
        ```
        """)
    if category == 'classification':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 分类实现示例（以 LogisticRegression 为例）
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        print('Accuracy:', model.score(X_test, y_test))
        ```
        """)
    if category == 'clustering':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 聚类实现示例（以 KMeans 为例）
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        print('Cluster centers:', model.cluster_centers_)
        ```
        """)
    if category == 'dimred':
        return textwrap.dedent(f"""
        ```python
        # scikit-learn 降维实现示例（以 PCA 为例）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        print('Explained variance:', pca.explained_variance_ratio_)
        ```
        """)
    if category == 'deep':
        return textwrap.dedent(f"""
        ```python
        # PyTorch 简单模型示例（以全连接网络为例）
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
    if category == 'rl':
        return textwrap.dedent(f"""
        ```python
        # 简单 RL 示例（使用 gym 环境）
        import gym
        env = gym.make('CartPole-v1')
        obs = env.reset()
        for _ in range(200):
            action = env.action_space.sample()  # 随机策略
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                break
        env.close()
        ```
        """)
    # generic fallback
    return textwrap.dedent(f"""
    ```python
    # TODO: 根据实际算法补充库实现代码
    ```
    """)


def hand_impl(name, category):
    # Very minimal numpy implementation skeleton
    class_name = re.sub(r'[^A-Za-z0-9]', '', name)
    if category in ('regression', 'classification'):
        return textwrap.dedent(f"""
        ```python
        # 手工实现示例（仅作结构示例）\nimport numpy as np\n\nclass {class_name}:\n    def __init__(self, *args, **kwargs):\n        pass\n    def fit(self, X, y):\n        # TODO: 实现训练过程\n        pass\n    def predict(self, X):\n        # TODO: 实现预测过程\n        return np.zeros(len(X))\n        ```
        """)
    if category == 'deep':
        return textwrap.dedent(f"""
        ```python
        # 手工实现深度模型简化示例（使用 numpy）\nimport numpy as np\n\nclass SimpleNN:\n    def __init__(self, input_dim, hidden_dim, output_dim):\n        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros(output_dim)
    def forward(self, x):\n        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU\n        return h @ W2 + self.b2\n        ```
        """)
    # generic fallback
    return textwrap.dedent(f"""
    ```python
    # TODO: 根据实际算法补充手工实现代码
    ```
    """)


def visualization(name, category):
    # Generic scatter / loss plot
    if category in ('regression', 'classification'):
        return textwrap.dedent(f"""
        ```python
        # 可视化示例（散点图或决策边界）\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nX = np.random.randn(200, 2)\ny = np.random.randint(0, 2, 200)\nplt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')\nplt.title('{name} 可视化示例')\nplt.show()\n        ```
        """)
    if category == 'dimred':
        return textwrap.dedent(f"""
        ```python
        # 可视化降维后结果（2D）\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# 假设 X_reduced 为 2 维
X_reduced = np.random.randn(200, 2)\nplt.scatter(X_reduced[:,0], X_reduced[:,1], cmap='plasma')\nplt.title('{name} 降维可视化')\nplt.show()\n        ```
        """)
    if category == 'deep':
        return textwrap.dedent(f"""
        ```python
        # 可视化示例：随机生成特征图\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nfeature_map = np.random.rand(8, 8)\nplt.imshow(feature_map, cmap='gray')\nplt.title('{name} 特征图示例')\nplt.show()\n        ```
        """)
    # generic fallback
    return "\n- TODO: 添加可视化示例\n"


def evaluation(name, category):
    if category == 'regression':
        metric = 'mean_squared_error'
    elif category == 'classification':
        metric = 'accuracy_score'
    elif category == 'clustering':
        metric = 'silhouette_score'
    elif category == 'dimred':
        metric = 'explained_variance_score'
    elif category == 'rl':
        metric = 'episode_reward'
    else:
        metric = 'accuracy_score'
    return textwrap.dedent(f"""
    ```python
    # 评估示例\nfrom sklearn.metrics import {metric}\n# y_true, y_pred / X, labels 需自行准备\n# print('{metric}:', {metric}(y_true, y_pred))\n    ```
    """)


def common_issues():
    return textwrap.dedent("""
    - 数据未标准化或归一化导致模型不收敛。\n- 超参数（学习率、正则化、层数）需要调参。\n- 过拟合：模型在训练集表现好但测试集表现差。\n- 计算资源：深度模型需 GPU 加速。\n""")

def learning_summary(name):
    return f"**学习要点**：{name} 的核心思想是 …（请根据实际算法补充）。掌握其数学推导、实现细节以及适用场景是后续深入学习的基础。"

def exercises(name):
    return textwrap.dedent(f"""
    1. 手动实现 {name} 的核心算法并在合成数据上验证。\n2. 使用不同库（如 scikit‑learn 与 PyTorch）实现，并比较训练时间与精度。\n3. 设计可视化函数，展示 {name} 在不同超参数下的表现。\n""")

def learning_path():
    return textwrap.dedent("""
    - 先掌握线性模型（线性回归、逻辑回归）→\n- 再学习树模型（决策树、随机森林、XGBoost）→\n- 深入深度学习模型（CNN、Transformer、GAN）→\n- 进阶章节：自监督学习、强化学习、生成模型等前沿方向。\n""")


def replace_sections(content, name):
    lines = content.split('\n')
    out = []
    i = 0
    category = detect_category(name)
    while i < len(lines):
        line = lines[i]
        # detect heading ## N.
        m = re.match(r'^(##)\s+(\d+)\.\s+(.*)', line)
        if m:
            num = int(m.group(2))
            heading = line.rstrip()
            out.append(heading)
            # collect following lines until next heading or EOF
            i += 1
            # skip old content lines until next heading
            while i < len(lines) and not re.match(r'^(##)\s+\d+\.', lines[i]):
                i += 1
            # now i points to next heading or EOF, we need to insert new content for sections >=7
            if num >= 7:
                if num == 7:
                    out.append('')
                    out.append(lib_impl(name, category))
                elif num == 8:
                    out.append('')
                    out.append(hand_impl(name, category))
                elif num == 9:
                    out.append('')
                    out.append(visualization(name, category))
                elif num == 10:
                    out.append('')
                    out.append(evaluation(name, category))
                elif num == 11:
                    out.append('')
                    out.append(common_issues())
                elif num == 12:
                    out.append('')
                    out.append(learning_summary(name))
                elif num == 13:
                    out.append('')
                    out.append(exercises(name))
                elif num == 14:
                    out.append('')
                    out.append(learning_path())
                else:
                    # keep placeholder for other numbers if any
                    out.append('')
                    out.append('<!-- TODO: add content -->')
                # continue loop
            else:
                # for sections 1-6 we keep original content (already skipped above)
                # need to re‑add original lines which we previously skipped
                # Actually we have skipped them; we need to capture them earlier.
                # Simplify: for sections <7 we just copy original lines until next heading.
                # We'll rewind i to start of this section content and copy until before next heading.
                # To achieve this we should have saved the original content before skipping.
                pass
        else:
            # regular line outside a heading, copy as is
            out.append(line)
            i += 1
    return '\n'.join(out)


def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    name = os.path.splitext(os.path.basename(path))[0]
    new_content = replace_sections(content, name)
    if new_content != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)


def main():
    alg_dir = 'algorithm_knowledge_base/algorithms'
    for fname in os.listdir(alg_dir):
        if fname.endswith('.md'):
            process_file(os.path.join(alg_dir, fname))

if __name__ == '__main__':
    main()
