# Platt Scaling Learning Document

> Platt Scaling (Platt Scaling) is a post-processing calibration method for converting classifier scores into probability estimates. Proposed by John Platt in 1999, it uses logistic regression on SVM outputs to produce well-calibrated probabilities, and is widely used in classification systems requiring probability estimates.

---

## 1. Algorithm Basic Understanding

### One-Sentence Definition

Platt Scaling fits a sigmoid-based logistic regression model on the raw outputs (decision values) of a classifier to convert them into calibrated probability estimates, ensuring that predicted probabilities match the true frequency of correct predictions.

### Intuition Analogy

Imagine a teacher (SVM) gives you scores (0-10) for your homework, but these scores don't directly represent your actual probability of passing. Platt Scaling is like having the teacher also record whether each score actually led to passing, then learning a mapping from scores to passing probability. Now when you get a new score, you can accurately estimate your passing probability.

### Historical Background

- **1999**: John Platt proposes Platt Scaling for SVM probability estimation
- **2000s**: Widely adopted in SVM tools (LIBSVM)
- **2010s**: Extended to neural networks, ensemble methods
- **Modern**: Often compared with isotonic regression

### Algorithm Positioning

- **Type**: Probability calibration
- **Output**: Calibrated probabilities P(y=1|x)
- **Model Type**: Post-hoc calibration (not a standalone classifier)
- **Category**: Calibration methods

### Prerequisites

- Logistic regression basics
- SVM decision function
- Probability calibration concepts
- Maximum likelihood estimation

---

## 2. Core Principles

### 2.1 Core Idea

Platt Scaling assumes the relationship between classifier scores and true probabilities follows a sigmoid function:
$$P(y=+1|f) = \frac{1}{1 + \exp(A \cdot f + B)}$$

where $f$ is the SVM decision value (decision function), A and B are parameters learned via maximum likelihood on training data.

### 2.2 Working Process

1. Train base classifier (e.g., SVM)
2. Collect decision values on validation set with labels
3. Fit logistic regression on (f, y)
4. Apply sigmoid transformation to new predictions

### 2.3 Key Concepts

| Concept | Symbol | Meaning |
|---------|--------|---------|
| Decision value | $f(x)$ | Raw SVM output |
| Calibration | - | Mapping scores to probabilities |
| Sigmoid | $\sigma(x)$ | $1/(1+e^{-x})$ |
| Calibrated probability | $P(y=+1\|f)$ | Final probability |
| A, B | - | Sigmoid parameters |

### 2.4 Geometric Interpretation

The calibration curve is a sigmoid (S-curve) mapping linear decision values to [0,1]:
- Very negative f → probability → 0
- f = 0 → probability = 0.5
- Very positive f → probability → 1

---

## 3. Mathematical Formulas and Derivation

### 3.1 Notation

| Symbol | Meaning | Dimensions |
|--------|--------|----------|
| $f_i$ | Decision value for sample i | n |
| $y_i$ | Label (±1) | n |
| $A$ | Slope parameter | scalar |
| $B$ | Bias parameter | scalar |
| $P_i$ | Calibrated probability | n |

### 3.2 Problem Formulation

**Goal**: Find parameters A, B to minimize negative log-likelihood:
$$\mathcal{L}(A, B) = -\sum_{i=1}^{n} \log P(y_i|f_i)$$

where:
$$P(y_i|f_i) = \frac{1}{1 + \exp(-y_i(A \cdot f_i + B))}$$

### 3.3 Objective Function

Using labels y ∈ {+1, -1}:
$$P(y=+1|f) = \sigma(A \cdot f + B) = \frac{1}{1 + e^{-(Af + B)}}$$
$$P(y=-1|f) = 1 - P(y=+1|f) = \frac{1}{1 + e^{Af + B}}$$

### 3.4 Derivation

**Step 1**: Negative log-likelihood

$$\mathcal{L} = -\sum_i \left[ \mathbb{1}(y_i=+1) \log \sigma(Af_i + B) + \mathbb{1}(y_i=-1) \log (1-\sigma(Af_i + B)) \right]$$

**Step 2**: Gradient

Set y' = (y_i + 1)/2 ∈ {0, 1}:
$$\mathcal{L} = -\sum_i [y' \log \sigma(z) + (1-y') \log (1-\sigma(z))]$$

where z = Af_i + B

**Step 3**: Derivative with respect to A, B

$$\frac{\partial \mathcal{L}}{\partial A} = \sum_i (P_i - y') f_i$$
$$\frac{\partial \mathcal{L}}{\partial B} = \sum_i (P_i - y')$$

where P_i = σ(z)

**Step 4**: Update via gradient descent

Using standard logistic regression optimization (Newton or L-BFGS).

### 3.5 Final Algorithm

```
Input: decision values f_i, labels y_i (±1)
Output: parameters A, B

1. Convert y_i ∈ {±1} to y_i' ∈ {0,1}
2. Initialize A=0, B=0 (or from prior)
3. Optimize using L-BFGS:
   while not converged:
       z_i = A * f_i + B
       p_i = sigmoid(z_i)
       grad_A = sum((p_i - y_i') * f_i)
       grad_B = sum(p_i - y_i')
       update A, B using gradients
4. Return A, B
```

---

## 4. Training Process

### 4.1 Data Preprocessing

- Use held-out validation set (not training)
- Decision values from base classifier
- Labels already converted to ±1

### 4.2 Parameter Initialization

- A = 1.0, B = 0 (common start)
- Use核 initial values from prior

### 4.3 Training Code (Python)

```python
import numpy as np
from scipy.optimize import minimize

class PlattScaling:
    """
    Platt Scaling for probability calibration
    
    Fits logistic regression on decision values to produce
    calibrated probabilities.
    """
    
    def __init__(self):
        """Initialize"""
        self.A = None
        self.B = None
        self.fitted = False
        
    def fit(self, decision_values, labels):
        """
        Fit Platt Scaling
        
        Args:
            decision_values: Raw classifier outputs (f)
            labels: True labels (y in {0,1} or {±1})
        """
        # Convert labels to {0, 1}
        if labels.max() == 1 and labels.min() == -1:
            labels = (labels + 1) / 2
        
        # Store for reference
        self.decision_values = decision_values
        self.labels = labels
        
        # Initial parameters
        x0 = [1.0, 0.0]
        
        # Optimize negative log-likelihood
        result = minimize(
            self._negative_log_likelihood,
            x0,
            args=(decision_values, labels),
            method='L-BFGS-B'
        )
        
        self.A, self.B = result.x
        self.fitted = True
        
        return self
    
    def _negative_log_likelihood(self, params, f, y):
        """Compute negative log-likelihood"""
        A, B = params
        z = A * f + B
        
        # Clip for numerical stability
        z = np.clip(z, -500, 500)
        
        # Logistic sigmoid
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        
        # Negative log-likelihood
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        return nll
    
    def predict_proba(self, decision_values):
        """
        Predict calibrated probabilities
        
        Args:
            decision_values: Raw classifier outputs
            
        Returns:
            Probabilities P(y=1|x)
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        z = self.A * decision_values + self.B
        z = np.clip(z, -500, 500)
        
        proba = 1 / (1 + np.exp(-z))
        
        return proba
    
    def predict(self, decision_values):
        """
        Predict class labels
        
        Args:
            decision_values: Raw classifier outputs
            
        Returns:
            Predicted labels (0 or 1)
        """
        proba = self.predict_proba(decision_values)
        return (proba >= 0.5).astype(int)


def demo_platt_scaling():
    """Demo Platt Scaling"""
    np.random.seed(42)
    
    # Generate synthetic decision values
    # Positive class: higher decision values
    pos_f = np.random.normal(2.0, 1.0, 200)
    neg_f = np.random.normal(-2.0, 1.0, 200)
    
    f = np.concatenate([pos_f, neg_f])
    y = np.concatenate([np.ones(200), np.zeros(200)])
    
    # Shuffle
    idx = np.random.permutation(len(f))
    f = f[idx]
    y = y[idx]
    
    print("=" * 60)
    print("Platt Scaling Demo")
    print("=" * 60)
    print(f"Positive samples: {sum(y)}, Negative samples: {len(y)-sum(y)}")
    print(f"Mean f (positive): {pos_f.mean():.3f}")
    print(f"Mean f (negative): {neg_f.mean():.3f}")
    
    # Fit Platt Scaling
    platt = PlattScaling()
    platt.fit(f, y)
    
    print(f"\nFitted parameters:")
    print(f"A = {platt.A:.4f}")
    print(f"B = {platt.B:.4f}")
    
    # Test calibration
    test_f = np.array([-5, -2, 0, 2, 5])
    test_proba = platt.predict_proba(test_f)
    
    print(f"\nCalibration test:")
    print(f"f = {test_f}")
    print(f"P(y=1|f) = {test_proba.round(3)}")


if __name__ == "__main__":
    demo_platt_scaling()
```

### 4.4 Convergence Criteria

- L-BFGS convergence
- Maximum iterations
- Parameter change threshold

### 4.5 Hyperparameters

| Hyperparameter | Role | Range | Default |
|-----------|------|-------|---------|
| Solver | Optimization | L-BFGS, Newton | L-BFGS |
| Max iterations | Stop condition | 100-1000 | 100 |
| Tolerance | Convergence | 1e-4 ~ 1e-8 | 1e-5 |

---

## 5. Application Scenarios

### 5.1 Typical Applications

1. **SVM probability outputs**:
   - Convert SVM scores to probabilities
   - Used in LIBSVM, sklearn.svm

2. **Classification with probabilities**:
   - Risk assessment
   - Medical diagnosis
   - Credit scoring

3. **Ensemble methods**:
   - Probability combination
   - Weighted voting

### 5.2 Suitable Data

- Sufficient calibration samples (≥100)
- Balanced class distribution
- No extreme class imbalance

### 5.3 Limitations

- Binary classification (main)
- Homoscedastic assumption
- Requires held-out data

---

## 6. Advantages and Disadvantages

### 6.1 Advantages

| Advantage | Explanation | Condition |
|-----------|------------|-----------|
| Well-calibrated | Matches true probability | Enough samples |
| Simple | Sigmoid transformation | - |
| Fast inference | Single sigmoid | - |
| Theoretical | Maximum likelihood | - |

### 6.2 Disadvantages

| Disadvantage | Explanation | Mitigation |
|-----------|------------|------------|
| Binary only | Not for multi-class | One-vs-rest |
| Requires data | Needs held-out set | Cross-validation |
| Homoscedastic | Assumes sigmoid variance | Isotonic regression |

---

## 7. Library Implementation (scikit-learn)

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

class PlattScalingSklearn:
    """
    Platt Scaling wrapper using sklearn
    
    Uses CalibratedClassifierCV with sigmoid method.
    """
    
    def __init__(self, base_estimator=None, cv=5):
        """
        Initialize
        
        Args:
            base_estimator: Base classifier
            cv: Cross-validation folds for calibration
        """
        self.base_estimator = base_estimator
        self.cv = cv
        self.model = None
        
    def fit(self, X, y):
        """
        Train calibrated classifier
        
        Args:
            X: Feature matrix
            y: Labels (0/1 or ±1)
        """
        if self.base_estimator is None:
            self.base_estimator = SVC(kernel='linear')
        
        # CalibratedClassifierCV with sigmoid = Platt Scaling
        self.model = CalibratedClassifierCV(
            self.base_estimator,
            method='sigmoid',
            cv=self.cv
        )
        
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """Predict labels"""
        return self.model.predict(X)


def demo_sklearn_platt():
    """Demo with sklearn"""
    np.random.seed(42)
    
    # Generate data
    X = np.random.randn(500, 20)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Split
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    
    print("=" * 60)
    print("Platt Scaling with sklearn")
    print("=" * 60)
    
    # Train
    model = PlattScalingSklearn(cv=3)
    model.fit(X_train, y_train)
    
    # Evaluate
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    acc = np.mean(preds == y_test)
    print(f"Accuracy: {acc:.4f}")
    print(f"Probability range: [{proba.min():.3f}, {proba.max():.3f}]")


if __name__ == "__main__":
    demo_sklearn_platt()
```

---

## 8. Manual Implementation (Core Algorithm)

```python
import numpy as np

def platt_scaling(f, y):
    """
    Manual Platt Scaling implementation
    
    Args:
        f: Decision values (array of shape (n,))
        y: Labels (array of {0,1} or {±1})
        
    Returns:
        A, B: Sigmoid parameters
    """
    # Convert labels to {0, 1}
    if y.min() < 0:
        y = (y + 1) / 2
    
    # Initial parameters
    A = 1.0
    B = 0.0
    
    # Gradient descent
    lr = 0.01
    n_iter = 1000
    
    for _ in range(n_iter):
        z = A * f + B
        z = np.clip(z, -500, 500)
        
        p = 1 / (1 + np.exp(-z))
        
        # Gradients
        grad_A = np.sum((p - y) * f)
        grad_B = np.sum(p - y)
        
        # Update
        A -= lr * grad_A
        B -= lr * grad_B
    
    return A, B


def platt_predict_proba(f, A, B):
    """Predict probabilities using fitted parameters"""
    z = A * f + B
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# Demo
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate data
    pos_f = np.random.normal(1.5, 1.0, 300)
    neg_f = np.random.normal(-1.5, 1.0, 300)
    
    f = np.concatenate([pos_f, neg_f])
    y = np.concatenate([np.ones(300), np.zeros(300)])
    
    # Fit
    A, B = platt_scaling(f, y)
    
    print("=" * 60)
    print("Manual Platt Scaling")
    print("=" * 60)
    print(f"A = {A:.4f}")
    print(f"B = {B:.4f}")
    
    # Test
    test_f = np.array([-3, -1, 0, 1, 3])
    test_proba = platt_predict_proba(test_f, A, B)
    
    print(f"\nTest f: {test_f}")
    print(f"Proba: {test_proba.round(3)}")
```

---

## 9. Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_platt():
    """Visualize Platt Scaling"""
    np.random.seed(42)
    
    # Generate training data
    pos_f = np.random.normal(2.0, 1.0, 200)
    neg_f = np.random.normal(-2.0, 1.0, 200)
    
    f = np.concatenate([pos_f, neg_f])
    y = np.concatenate([np.ones(200), np.zeros(200)])
    
    # Fit
    from scipy.optimize import minimize
    platt = PlattScaling()
    platt.fit(f, y)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Decision value distribution
    ax1 = axes[0, 0]
    ax1.hist(pos_f, bins=30, alpha=0.5, label='Positive')
    ax1.hist(neg_f, bins=30, alpha=0.5, label='Negative')
    ax1.set_xlabel('Decision Value')
    ax1.set_ylabel('Count')
    ax1.set_title('Decision Value Distribution')
    ax1.legend()
    
    # 2. Calibration curve
    ax2 = axes[0, 1]
    f_range = np.linspace(-6, 6, 100)
    proba = platt.predict_proba(f_range)
    ax2.plot(f_range, proba, 'b-', label='Platt Scaling')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Decision Value')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'Sigmoid: P(y=1|f) = 1/(1+exp({platt.A:.2f}*f + {platt.B:.2f}))')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reliability diagram (binned)
    ax3 = axes[1, 0]
    proba = platt.predict_proba(f)
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    true_freq = []
    
    for i in range(len(bins)-1):
        mask = (proba >= bins[i]) & (proba < bins[i+1])
        if mask.sum() > 0:
            true_freq.append(y[mask].mean())
        else:
            true_freq.append(0)
    
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax3.plot(bin_centers, true_freq, 'bo', label='Actual')
    ax3.set_xlabel('Mean Predicted Probability')
    ax3.set_ylabel('Fraction of Positives')
    ax3.set_title('Reliability Diagram')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Predicted vs actual
    ax4 = axes[1, 1]
    ax4.scatter(proba, y, alpha=0.3, s=10)
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Actual Label')
    ax4.set_title('Predictions vs Labels')
    
    plt.tight_layout()
    plt.savefig('platt_scaling_visualization.png', dpi=150)
    plt.show()
    print("Visualization saved to platt_scaling_visualization.png")


if __name__ == "__main__":
    visualize_platt()
```

---

## 10. Model Evaluation

### 10.1 Metrics

```python
from sklearn.metrics import brier_score_loss, log_loss

def evaluate_calibration(y_true, y_proba):
    """
    Evaluate probability calibration
    
    Args:
        y_true: True labels (0/1)
        y_proba: Predicted probabilities
    """
    print("=" * 60)
    print("Calibration Evaluation")
    print("=" * 60)
    
    # Brier Score (lower is better)
    brier = brier_score_loss(y_true, y_proba)
    print(f"Brier Score: {brier:.4f}")
    
    # Log Loss (lower is better)
    logloss = log_loss(y_true, y_proba)
    print(f"Log Loss: {logloss:.4f}")
    
    # Expected Calibration Error (ECE)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_proba[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    
    ece /= len(y_true)
    print(f"ECE: {ece:.4f}")
```

---

## 11. Common Problems

### 11.1 Problem 1: Poor Calibration
**Cause**: Too few calibration samples

**Solution**: Increase calibration set or use CV
```python
# Use cross-validation for small datasets
model = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
```

### 11.2 Problem 2: Extreme Probabilities
**Cause**: Large decision values

**Solution**: Clip or use temperature scaling
```python
# Clip decision values
z = np.clip(z, -500, 500)
```

### 11.3 Problem 3: Class Imbalance
**Cause**: Unbalanced training data

**Solution**: Use sample weights or adjust
```python
# Sample weights inversely proportional to frequency
```

---

## 12. Learning Summary

### Core Points:
1. **Sigmoid mapping**: f → P(y=1) via sigmoid
2. **Maximum likelihood**: Fits A, B
3. **Post-hoc**: Works with any base classifier
4. **Well-calibrated**: Probabilities match true frequency

### From Platt to Other Methods:
- Platt → Isotonic: Non-parametric calibration
- Platt → Temperature: Scaling for neural networks
- Platt → Beta calibration: Flexible distribution

### Practice:
1. Use SVM with Platt Scaling
2. Compare with isotonic regression
3. Evaluate with ECE

---

## 13. Exercises

### Exercise 1: Calculation
Q: If A=1, B=0, compute P(y=1|f=0).

<details>
<summary>Answer</summary>

P = 1/(1 + exp(-(1*0 + 0))) = 1/(1 + 1) = 0.5

Answer: 0.5

</details>

### Exercise 2: Implementation
Q: Extend Platt to multi-class using one-vs-all.

<details>
<summary>Answer</summary>

```python
class PlattMultiClass:
    def __init__(self, n_classes):
        self.n_classes = n_classess
        self.calibrators = [None] * n_classes
    
    def fit(self, decision_values, labels):
        for c in range(self.n_classes):
            y_binary = (labels == c).astype(int)
            self.calibrators[c] = PlattScaling()
            self.calibrators[c].fit(decision_values[:, c], y_binary)
    
    def predict_proba(self, decision_values):
        proba = np.zeros_like(decision_values)
        for c in range(self.n_classes):
            proba[:, c] = self.calibrators[c].predict_proba(decision_values[:, c])
        return proba / proba.sum(axis=1, keepdims=True)
```

</details>

### Exercise 3: Theory
Q: Why is Platt Scaling better than naive division?

<details>
<summary>Answer</summary>

A: Naive method (score/total) assumes linear relationship, but SVM decision values are not linearly related to probability. Platt uses maximum likelihood on sigmoid, which is theoretically grounded and produces well-calibrated probabilities empirically.

</details>

### Thinking 1: When to use isotonic?
- When data is sufficient (≥300 per class)
- When calibration curve may be non-monotonic

### Thinking 2: Extensions
- Temperature scaling: P ∝ exp(f/T)
- Beta calibration: Flexible distribution

---

## 14. Learning Path

### Beginner:
1. Understand logistic regression
2. Learn SVM basics
3. Implement Platt Scaling
4. Apply to real data

### Intermediate:
1. Compare calibration methods
2. Study ECE metrics
3. Handle class imbalance
4. Multi-class extension

### Advanced:
1. Isotonic regression
2. Temperature scaling
3. Deep learning calibration

### Projects:
1. Calibrate SVM for medical diagnosis
2. Compare with isotonic
3. Build calibrated ensemble

### Resources:
- **Paper**: Platt 1999 "Probabilistic Outputs for SVMs"
- **Book**: "Pattern Recognition" by Bishop
- **Code**: LIBSVM, sklearn