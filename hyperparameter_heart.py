import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load and preprocess the heart disease dataset
data = pd.read_csv("heart.csv")
X = data.drop("output", axis=1)
y = data["output"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
mlp = MLPClassifier(random_state=42)

# Generate validation curves for hidden_layer_sizes
param_range = [(10,), (50,), (100,), (50, 50), (100, 50)]
train_scores, test_scores = validation_curve(
    mlp,
    X_train,
    y_train,
    param_name="hidden_layer_sizes",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

# Convert scores to error rates
train_errors = 1 - train_scores
test_errors = 1 - test_scores

# Calculate mean and standard deviation of errors
train_errors_mean = np.mean(train_errors, axis=1)
train_errors_std = np.std(train_errors, axis=1)
test_errors_mean = np.mean(test_errors, axis=1)
test_errors_std = np.std(test_errors, axis=1)

# Plot the validation curve
plt.figure()
plt.title("Model Complexity Graph (Neural Network) - Heart Disease")
plt.xlabel("Hidden Layer Sizes")
plt.ylabel("Error Rate")
plt.grid()
plt.fill_between(
    [str(x) for x in param_range],
    train_errors_mean - train_errors_std,
    train_errors_mean + train_errors_std,
    alpha=0.1,
    color="r",
)
plt.fill_between(
    [str(x) for x in param_range],
    test_errors_mean - test_errors_std,
    test_errors_mean + test_errors_std,
    alpha=0.1,
    color="g",
)
plt.plot(
    [str(x) for x in param_range],
    train_errors_mean,
    "o-",
    color="r",
    label="Training error",
)
plt.plot(
    [str(x) for x in param_range],
    test_errors_mean,
    "o-",
    color="g",
    label="Cross-validation error",
)
plt.legend(loc="best")
plt.show()
