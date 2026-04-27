# Passive Aggressive Learning Document

> Passive Aggressive (PA) Algorithm is an online learning algorithm for binary/multi-class classification. Introduced by Crammer et al. in 2006, it combines the simplicity of perceptron with margin-based learning, achieving theoretical guarantees similar to SVM while being more efficient.

---

## 1. Algorithm Basic Understanding

### One-Sentence Definition

PA is an online learning algorithm that updates the model only when a mistake occurs, using a "passive" (no update on correct predictions) and "aggressive" (large update on mistakes) update strategy.

### Intuition Analogy

Think of PA like learning from mistakes:
- **Passive**: When you predict correctly with high confidence, you stay calm (no update)
- **Aggressive**: When you make a mistake or are uncertain, you react strongly (large update)

This is similar to how humans learn - we don't change our beliefs for every small thing, but we significantly adjust when we're wrong.

### Historical Background

- **2006**: Crammer et al. propose PA algorithm at ICML
- **PA-I**: Passive Aggressive I (simple margin-based)
- **PA-II**: Passive Aggressive II (slack variable formulation)
- **2010s**: Extended to multi-class, regression, deep learning

PA bridges perceptron (online, simple) and SVM (margin-based, theoretical guarantees).

### Algorithm Positioning

- **Type**: Online learning / Large-scale classification
- **Output**: Linear classifier weights
- **Model Type**: Discriminative linear model
- **Category**: Classification → Online Learning

### Prerequisites

- Linear classifier basics: perceptron, logistic regression
- Margin and hinge loss concept
- Optimization basics: gradient descent

---

## 2. Core Principles

### 2.1 Core Idea

PA updates the weight vector only when a mistake occurs. The key innovation is the **aggression parameter C** that controls the update magnitude:

- **Passive step**: If $y_t \cdot (w_t \cdot x_t) \geq 1$ (correct and confident), stay passive: $w_{t+1} = w_t$
- **Aggressive step**: If $y_t \cdot (w_t \cdot x_t) < 1$ (mistake or low margin), update aggressively

### 2.2 Algorithm Flow

1. Initialize weight vector $w_1 = 0$
2. For each training example $(x_t, y_t)$:
   - Predict: $\hat{y}_t = \text{sign}(w_t \cdot x_t)$
   - If mistake: update using PA rule
3. Return final weight vector

### 2.3 Key Concepts

| Concept | Symbol | Meaning |
|---------|--------|---------|
| Weight vector | $w$ | Classifier parameters |
| Margin | $y_t(w_t \cdot x_t)$ | Confidence of prediction |
| Aggression | $C$ | Maximum update size |
| Loss | $\ell_t = \max(0, 1 - y_t(w_t \cdot x_t))$ | Hinge loss |

### 2.4 Mathematical Formulation

The PA update can be derived from optimization:
$$\min_{w} \frac{1}{2} ||w - w_t||^2 + C \cdot \ell_t$$

Where $\ell_t$ is the hinge loss:
$$\ell_t = \max(0, 1 - y_t(w_t \cdot x_t))$$

Closed-form solution:
$$w_{t+1} = w_t + \tau_t \cdot y_t x_t$$

Where:
$$\tau_t = \frac{\ell_t}{||x_t||^2}$$

This is PA-I (Passive Aggressive I).

For PA-II, we add regularization:
$$\tau_t = \frac{\ell_t}{||x_t||^2 + \frac{1}{2C}}$$

---

## 3. Mathematical Formulas and Derivation

### 3.1 Notation

| Symbol | Meaning | Dimensions |
|--------|---------|------------|
| $x_t$ | Feature vector | $d$ |
| $y_t$ | Label (±1) | scalar |
| $w_t$ | Weight vector | $d$ |
| $C$ | Aggression parameter | scalar |
| $\tau_t$ | Update step size | scalar |

### 3.2 Problem Formulation

**Online learning protocol**:
1. Receive feature vector $x_t$
2. Predict $\hat{y}_t = \text{sign}(w_t \cdot x_t)$
3. Receive true label $y_t$
4. Update if $y_t \neq \hat{y}_t$

**Objective**: Minimize cumulative mistake rate.

### 3.3 Loss Function

Hinge loss in batch form:
$$\mathcal{L}_{hinge}(w) = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w \cdot x_i))$$

PA uses online version, updating only when $\ell_t > 0$.

### 3.4 Derivation

**Step 1**: Deriving PA-I update

Given mistake at $(x_t, y_t)$:
$$\min_w \frac{1}{2} ||w - w_t||^2 + C \cdot \ell_t$$

Where:
$$\ell_t = \max(0, 1 - y_t(w \cdot x_t))$$

**Step 2**: Lagrangian

Since $\ell_t > 0$ (mistake), we have constraint:
$$y_t(w \cdot x_t) \leq 1 - \ell_t$$

In standard form:
$$\min_w \frac{1}{2} ||w - w_t||^2 + C \cdot \ell_t$$
$$\text{s.t. } y_t(w \cdot x_t) \leq 1 - \ell_t, \ell_t \geq 0$$

**Step 3**: Solving

Using Lagrangian, we get:
$$w = w_t + \tau \cdot y_t x_t$$

Plug back:
$$\tau = \frac{\ell_t}{||x_t||^2}$$

**Step 4**: PA-II variant

Add regularization to $\ell$:
$$\ell_t = \max(0, 1 - y_t(w_t \cdot x_t))$$

Then:
$$\tau = \frac{\ell_t}{||x_t||^2 + \frac{1}{2C}}$$

### 3.5 Final Algorithm

```
Input: C, number of iterations T
Output: Weight vector w

1. Initialize: w_1 = 0
2. For t = 1 to T:
   a. Get example (x_t, y_t)
   b. Compute margin: m_t = y_t(w_t · x_t)
   c. If m_t < 1:  # mistake
      τ_t = (1 - m_t) / ||x_t||^2
      w_{t+1} = w_t + τ_t · y_t x_t
   d. Else:
      w_{t+1} = w_t  # passive

3. Return w
```

---

## 4. Training Process

### 4.1 Data Preprocessing

- Feature normalization: [0,1] or standardization
- Handle categorical: one-hot encoding
- Feature hashing for large-scale

### 4.2 Parameter Initialization

- $w_1 = 0$ (common)
- $w_1$ = random small values
- C selection: 0.1 to 1.0

### 4.3 Training Code (Python)

```python
import numpy as np

class PassiveAggressiveClassifier:
    """
    Passive Aggressive Classifier (PA-I and PA-II)
    
    Online learning algorithm for large-scale classification.
    """
    
    def __init__(self, C=1.0, mode='PA-I', n_iter=10):
        """
        Initialize PA classifier
        
        Args:
            C: Aggression parameter (regularization)
            mode: 'PA-I' or 'PA-II'
            n_iter: Number of passes over training data
        """
        self.C = C
        self.mode = mode
        self.n_iter = n_iter
        self.w = None
        self.mistakes = 0
        
    def fit(self, X, y):
        """
        Train PA classifier (offline mode for convenience)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (±1)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.mistakes = 0
        
        # Multiple passes (epoch)
        for epoch in range(self.n_iter):
            for t in range(n_samples):
                # Get example
                x_t = X[t]
                y_t = y[t]
                
                # Compute margin
                margin = y_t * np.dot(self.w, x_t)
                
                # Check for update (mistake or low margin)
                if margin < 1.0:
                    self.mistakes += 1
                    
                    # Compute step size
                    x_norm_sq = np.dot(x_t, x_t)
                    
                    if self.mode == 'PA-I':
                        tau = (1 - margin) / x_norm_sq
                    else:  # PA-II
                        tau = (1 - margin) / (x_norm_sq + 0.5 / self.C)
                    
                    # Update weight
                    self.w = self.w + tau * y_t * x_t
        
        return self
    
    def predict(self, X):
        """
        Predict labels
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels (±1)
        """
        scores = np.dot(X, self.w)
        return np.sign(scores)
    
    def score(self, X, y):
        """
        Compute accuracy
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demo_passive_aggressive():
    """Demo PA classifier"""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 500
    n_features = 20
    
    # Create linearly separable data
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features)
    y = np.sign(np.dot(X, true_w))
    
    # Add noise to some labels
    noise_idx = np.random.choice(n_samples, 50, replace=False)
    y[noise_idx] *= -1
    
    # Train-test split
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    
    print("=" * 60)
    print("Passive Aggressive Classifier Demo")
    print("=" * 60)
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {n_features}")
    
    # Train PA-I
    pa1 = PassiveAggressiveClassifier(C=1.0, mode='PA-I', n_iter=5)
    pa1.fit(X_train, y_train)
    acc1 = pa1.score(X_test, y_test)
    print(f"\nPA-I Accuracy: {acc1:.4f}")
    print(f"PA-I Mistakes: {pa1.mistakes}")
    
    # Train PA-II
    pa2 = PassiveAggressiveClassifier(C=1.0, mode='PA-II', n_iter=5)
    pa2.fit(X_train, y_train)
    acc2 = pa2.score(X_test, y_test)
    print(f"\nPA-II Accuracy: {acc2:.4f}")
    print(f"PA-II Mistakes: {pa2.mistakes}")


if __name__ == "__main__":
    demo_passive_aggressive()
```

### 4.4 Convergence Criteria

- Maximum epochs reached
- Mistake rate stabilizes
- Validation accuracy stops improving

### 4.5 Hyperparameters

| Hyperparameter | Role | Range | Default |
|-----------|------|-------|---------|
| C | Aggression | 0.01-10 | 1.0 |
| mode | PA variant | PA-I/PA-II | PA-I |
| n_iter | Epochs | 1-20 | 10 |

---

## 5. Application Scenarios

### 5.1 Typical Applications

1. **Large-scale text classification**:
   - News categorization
   - Spam detection
   - Sentiment analysis

2. **Online learning**:
   - Streaming data
   - Real-time updates

3. **Efficient baselines**:
   - Quick baseline for new tasks
   - Resource-constrained settings

### 5.2 Suitable Data

- Linearly separable or near-separable
- High-dimensional (sparse features)
- Large-scale (streaming)

### 5.3 Not Suitable

- Highly non-linear patterns
- Complex feature interactions
- Small datasets (overfits easily)

---

## 6. Advantages and Disadvantages

### 6.1 Advantages

| Advantage | Explanation | Condition |
|-----------|------------|-----------|
| Simple | No complex optimization | - |
| Efficient | Single pass per example | Online setting |
| Theoretical | Mistake bound guarantees | Linearly separable |
| Memory | No need to store all data | Online learning |

### 6.2 Disadvantages

| Disadvantage | Explanation | Mitigation |
|-----------|------------|------------|
| Linear | Cannot handle non-linear | Kernel trick |
| Sensitive | Sensitive to noise | Increase C |
| No probabilities | Hard to get confidence | Calibration |
| Last-update | Depends on last examples | Multiple passes |

---

## 7. Library Implementation (scikit-learn)

```python
import numpy as np
from sklearn.linear_model import SGDClassifier

class PassiveAggressiveWrapper:
    """
    Wrapper for PA using SGDClassifier with hinge loss
    
    The PA algorithm can be implemented using SGD with hinge loss
    and appropriate learning rate schedule.
    """
    
    def __init__(self, C=1.0, max_iter=100, random_state=42):
        """
        Initialize wrapper
        
        Args:
            C: Regularization (inverse of PA aggression)
            max_iter: Maximum iterations
            random_state: Random seed
        """
        # SGDClassifier with hinge loss approximates PA
        self.model = SGDClassifier(
            loss='hinge',
            alpha=1.0/C,  # Regularization
            max_iter=max_iter,
            random_state=random_state,
            learning_rate='constant',
            eta0=1.0,  # Constant learning rate
            penalty=None  # No L2 penalty (unlike SVM)
        )
        
    def fit(self, X, y):
        """Train"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict"""
        return self.model.predict(X)
    
    def score(self, X, y):
        """Accuracy"""
        return self.model.score(X, y)


def demo_with_sklearn():
    """Demo using sklearn"""
    np.random.seed(42)
    
    # Generate data
    X = np.random.randn(300, 20)
    w_true = np.random.randn(20)
    y = np.sign(np.dot(X, w_true))
    
    # Split
    X_train, X_test = X[:250], X[250:]
    y_train, y_test = y[:250], y[250:]
    
    print("=" * 60)
    print("PA with scikit-learn Demo")
    print("=" * 60)
    
    # Train
    model = PassiveAggressiveWrapper(C=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    demo_with_sklearn()
```

---

## 8. Manual Implementation (Core Algorithm)

```python
import numpy as np

def passive_aggressive_online(X, y, C=1.0, mode='PA-I', verbose=True):
    """
    Online Passive Aggressive learning
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (±1)
        C: Aggression parameter
        mode: 'PA-I' or 'PA-II'
        
    Returns:
        w: Learned weight vector
        mistakes: Number of updates
    """
    n_samples, n_features = X.shape
    
    # Initialize
    w = np.zeros(n_features)
    mistakes = 0
    
    # Online learning
    for t in range(n_samples):
        x_t = X[t]
        y_t = y[t]
        
        # Compute margin
        margin = y_t * np.dot(w, x_t)
        
        # Passive or Aggressive
        if margin < 1.0:
            mistakes += 1
            
            # Compute step size
            x_norm_sq = np.dot(x_t, x_t) + 1e-10
            
            if mode == 'PA-I':
                tau = (1 - margin) / x_norm_sq
            else:
                tau = (1 - margin) / (x_norm_sq + 0.5 / C)
            
            # Update
            w = w + tau * y_t * x_t
        
        if verbose and (t + 1) % 100 == 0:
            print(f"Processed {t+1} examples, {mistakes} mistakes")
    
    return w, mistakes


# Batch version (multiple passes)
def passive_aggressive_batch(X, y, C=1.0, mode='PA-II', n_epochs=10, verbose=True):
    """
    Batch PA learning (multiple epochs)
    """
    n_samples, n_features = X.shape
    
    w = np.zeros(n_features)
    total_mistakes = 0
    
    for epoch in range(n_epochs):
        mistakes = 0
        
        for t in range(n_samples):
            x_t = X[t]
            y_t = y[t]
            
            margin = y_t * np.dot(w, x_t)
            
            if margin < 1.0:
                mistakes += 1
                x_norm_sq = np.dot(x_t, x_t) + 1e-10
                
                if mode == 'PA-I':
                    tau = (1 - margin) / x_norm_sq
                else:
                    tau = (1 - margin) / (x_norm_sq + 0.5 / C)
                
                w = w + tau * y_t * x_t
        
        total_mistakes += mistakes
        
        if verbose:
            print(f"Epoch {epoch+1}: {mistakes} mistakes")
    
    return w, total_mistakes


# Demo
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate data
    n = 500
    d = 30
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = np.sign(np.dot(X, w_true))
    
    # Flip some labels
    flip_idx = np.random.choice(n, 30, replace=False)
    y[flip_idx] *= -1
    
    print("=" * 60)
    print("Manual PA Implementation")
    print("=" * 60)
    
    w, mistakes = passive_aggressive_batch(X, y, C=1.0, mode='PA-II', n_epochs=5)
    
    # Test
    preds = np.sign(np.dot(X, w))
    acc = np.mean(preds == y)
    print(f"\nFinal accuracy: {acc:.4f}")
    print(f"Total mistakes: {mistakes}")
```

---

## 9. Visualization and Understanding

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_pa_evolution():
    """Visualize PA learning process"""
    
    np.random.seed(42)
    
    # Simple 1D example
    X = np.random.randn(100, 1)
    y = np.sign(X.flatten() + 0.3 * np.random.randn(100))
    
    # Online PA
    w_history = [0]
    mistakes_history = []
    
    w = 0
    mistakes = 0
    
    for t in range(len(X)):
        x_t = X[t]
        y_t = y[t]
        
        margin = y_t * w * x_t
        
        if margin < 1.0:
            tau = (1 - margin) / (x_t**2 + 1e-10)
            w = w + tau * y_t * x_t
            mistakes += 1
        
        w_history.append(w)
        mistakes_history.append(mistakes)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Weight evolution
    ax1 = axes[0, 0]
    ax1.plot(w_history)
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Target')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Weight')
    ax1.set_title('Weight Evolution')
    ax1.legend()
    
    # 2. Mistake accumulation
    ax2 = axes[0, 1]
    ax2.plot(mistakes_history)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cumulative Mistakes')
    ax2.set_title('Mistake Accumulation')
    
    # 3. Data and decision boundary
    ax3 = axes[1, 0]
    pos_idx = y == 1
    neg_idx = y == -1
    ax3.scatter(X[pos_idx], np.zeros_like(X)[pos_idx], c='blue', label='+1')
    ax3.scatter(X[neg_idx], np.zeros_like(X)[neg_idx], c='red', label='-1')
    ax3.axvline(x=-1/w, color='green', linestyle='--', label=f'Decision w={w:.2f}')
    ax3.set_xlabel('Feature')
    ax3.set_title('Data and Decision Boundary')
    ax3.legend()
    
    # 4. Margin distribution
    ax4 = axes[1, 1]
    margins = y * (w * X.flatten())
    ax4.hist(margins, bins=20)
    ax4.axvline(x=1.0, color='r', linestyle='--', label='Margin=1')
    ax4.set_xlabel('Margin')
    ax4.set_ylabel('Count')
    ax4.set_title('Margin Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('pa_visualization.png', dpi=150)
    plt.show()
    print("Visualization saved to pa_visualization.png")


if __name__ == "__main__":
    visualize_pa_evolution()
```

---

## 10. Model Evaluation

### 10.1 Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_pa(y_true, y_pred):
    """Evaluate PA classifier"""
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Precision
    prec = precision_score(y_true, y_pred)
    print(f"Precision: {prec:.4f}")
    
    # Recall
    rec = recall_score(y_true, y_pred)
    print(f"Recall: {rec:.4f}")
    
    # F1
    f1 = f1_score(y_true, y_pred)
    print(f"F1: {f1:.4f}")
```

### 10.2 Theoretical Bounds

PA has theoretical mistake bounds:
$$\text{Mistakes} \leq \frac{R^2}{2\gamma^2}$$

Where:
- $R$: Maximum feature norm
- $\gamma$: Margin

---

## 11. Common Problems

### 11.1 Problem 1: Poor Performance on Noisy Data
**Reason**: Too aggressive updates on noise

**Solution**: Increase C (more regularization)
```python
# Before: C=1.0
# After: C=0.1
pa = PassiveAggressiveClassifier(C=0.1)
```

### 11.2 Problem 2: Slow Convergence
**Reason**: Learning rate too small

**Solution**: Adjust mode
```python
# PA-II is more stable than PA-I
pa = PassiveAggressiveClassifier(mode='PA-II')
```

### 11.3 Problem 3: Doesn't Converge
**Reason**: Data not linearly separable

**Solution**: Add kernel or use other models
```python
# Use kernel trick
# Or use SVM with RBF kernel
from sklearn.svm import SVC
svm = SVC(kernel='rbf')
```

---

## 12. Learning Summary

### Core Points:
1. **Online**: Updates only on mistakes
2. **Passive-Aggressive**: No update on correct, large update on mistakes
3. **Margin**: Uses hinge loss
4. **Efficient**: O(d) per example

### From PA to Other Algorithms:
- PA → SVM: Add kernel, regularization
- PA → Perceptron: No margin (C → ∞)
- PA → Logistic Regression: Log loss

### Practice:
1. Use for quick baselines
2. Try PA-II first
3. Tune C appropriately

---

## 13. Exercises

### Exercise 1: Calculation
Q: With data x=2, y=+1, current w=0.5, compute update for PA-I.

<details>
<summary>Answer</summary>

Margin: m = 1 * 0.5 * 2 = 1.0

Since m = 1.0 ≥ 1.0, prediction is correct with margin=1, NO UPDATE (passive).

Ans: No update (w stays at 0.5)

</details>

### Exercise 2: Implementation  
Q: Implement PA for multi-class classification.

<details>
<summary>Answer</summary>

```python
class PAMultiClass:
    def __init__(self, n_classes, C=1.0):
        self.n_classes = n_classes
        self.C = C
        self.W = np.zeros((n_classes, n_features))
    
    def fit(self, X, y):
        for x_t, y_t in zip(X, y):
            scores = np.dot(self.W, x_t)
            pred = np.argmax(scores)
            
            if pred != y_t:
                # Update all vs one
                self.W[y_t] += self.C * x_t / (np.dot(x_t, x_t) + 1e-10)
                self.W[pred] -= self.C * x_t / (np.dot(x_t, x_t) + 1e-10)
```

</details>

### Exercise 3: Theory
Q: Derive the update formula.

<details>
<summary>Answer</summary>

Solve:
$$\min_w \frac{1}{2} ||w - w_t||^2 + C \cdot \ell$$

$$\ell = \max(0, 1 - y_t(w \cdot x_t))$$

Using lagrangian and KKT conditions:
$$w = w_t + \tau \cdot y_t x_t$$

Substituting:
$$\tau = \frac{1 - y_t(w_t \cdot x_t)}{||x_t||^2}$$

For PA-II:
$$\tau = \frac{1 - y_t(w_t \cdot x_t)}{||x_t||^2 + \frac{1}{2C}}$$

Derivation complete.

</details>

### Thinking 1: Why is PA different from Perceptron?
- Perceptron: Update on any mistake
- PA: Update only when margin < 1

### Thinking 2: How to extend to regression?
- Use squared loss instead of hinge
- Similar update rule

---

## 14. Learning Path

### Beginner:
1. Understand perceptron
2. Learn hinge loss
3. Implement PA-I
4. Try on simple data

### Intermediate:
1. Study theoretical bounds
2. Compare PA-I vs PA-II
3. Extend to multi-class
4. Tune hyperparameters

### Advanced:
1. Kernel PA
2. Online learning theory
3. Deep learning extensions

### Projects:
1. Spam classifier
2. News categorization
3. Real-time text classification

### Resources:
- **Paper**: Crammer et al. 2003 PA paper
- **Book**: "Online Learning in Online Advertising"
- **Code**: Vowpal Wabbit library