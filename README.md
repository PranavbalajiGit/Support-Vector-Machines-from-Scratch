# Support Vector Machine (SVM) Classifier

## Overview
This is a custom implementation of a Support Vector Machine (SVM) classifier built from scratch using NumPy. The implementation uses gradient descent optimization to find the optimal hyperplane that separates binary classes.

## Algorithm Description

### What is SVM?
Support Vector Machine is a supervised learning algorithm used for binary classification. It works by finding the optimal hyperplane that maximally separates two classes in the feature space. The decision boundary is defined by:

**f(x) = w · x - b**

where:
- `w` is the weight vector (normal to the hyperplane)
- `b` is the bias term
- `x` is the input feature vector

### Objective Function
The SVM optimization problem minimizes:

**L = λ||w||² + (1/m) Σ max(0, 1 - yᵢ(w·xᵢ - b))**

This combines:
1. **Regularization term**: `λ||w||²` - Controls model complexity
2. **Hinge loss**: Penalizes misclassified points and points within the margin

## Code Structure

### Class: `SVM_classifier`

#### Initialization Parameters
```python
SVM_classifier(no_of_iterations, learning_rate, lambda_parameter)
```
- **no_of_iterations**: Number of training iterations (epochs)
- **learning_rate**: Step size for gradient descent updates
- **lambda_parameter**: Regularization strength (controls margin width)

#### Methods

**1. `fit(X, Y)`**
- Trains the SVM model on training data
- **Parameters**:
  - `X`: Feature matrix of shape (m, n) where m = samples, n = features
  - `Y`: Label vector of shape (m,) with binary labels (0 or 1)
- **Process**:
  - Initializes weights `w` to zeros (n-dimensional)
  - Initializes bias `b` to 0
  - Iteratively updates weights using gradient descent

**2. `update_weights()`**
- Performs one iteration of weight updates for all training samples
- **Label Encoding**: Converts labels from {0, 1} to {-1, 1} format required by SVM
- **Gradient Calculation**:
  - If sample is correctly classified with margin ≥ 1:
    - `dw = 2λw` (only regularization)
    - `db = 0`
  - If sample violates margin or is misclassified:
    - `dw = 2λw - yᵢxᵢ` (regularization + hinge loss gradient)
    - `db = yᵢ`
- **Parameter Update**:
  - `w = w - learning_rate × dw`
  - `b = b - learning_rate × db`

**3. `predict(X)`**
- Makes predictions on new data
- **Parameters**:
  - `X`: Feature matrix for prediction
- **Process**:
  - Computes decision function: `output = w · X - b`
  - Applies sign function: positive → class 1, negative → class -1
  - Converts back to {0, 1} labels
- **Returns**: Predicted labels

## Usage Example

```python
import numpy as np

# Initialize the classifier
model = SVM_classifier(
    no_of_iterations=1000,
    learning_rate=0.001,
    lambda_parameter=0.01
)

# Train the model
X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
y_train = np.array([0, 0, 1, 1])
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([[2, 2.5], [4, 5]])
predictions = model.predict(X_test)
print(predictions)  # Output: [0 1]
```

## Hyperparameter Tuning

- **learning_rate**: Typically 0.0001 to 0.01
  - Too high: May overshoot optimal solution
  - Too low: Slow convergence
  
- **lambda_parameter**: Typically 0.001 to 0.1
  - Higher values: Wider margin, simpler model
  - Lower values: Narrower margin, more complex model
  
- **no_of_iterations**: Typically 1000 to 10000
  - Depends on dataset size and convergence

## Dependencies
- NumPy: For numerical computations and array operations

## Mathematical Intuition

The SVM finds a hyperplane that:
1. Correctly classifies all training points (if possible)
2. Maximizes the margin (distance from hyperplane to nearest points of each class)
3. The "support vectors" are the points closest to the decision boundary that define the margin

The regularization parameter λ balances:
- **Large λ**: Emphasizes regularization → wider margin, more errors allowed
- **Small λ**: Emphasizes classification accuracy → narrower margin, fewer errors

## License
Custom implementation for educational purposes.