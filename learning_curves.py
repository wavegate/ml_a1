import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("heart.csv")

# Separate features and target variable
X = data.drop("output", axis=1)
y = data["output"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Print the sizes of the training and test sets
print(f"Total samples: {len(data)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def plot_learning_curve(
    estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)
):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Error Rate")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring="accuracy",
    )

    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    train_errors_mean = np.mean(train_errors, axis=1)
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_mean = np.mean(test_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)

    plt.grid()

    plt.fill_between(
        train_sizes,
        train_errors_mean - train_errors_std,
        train_errors_mean + train_errors_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_errors_mean - test_errors_std,
        test_errors_mean + test_errors_std,
        alpha=0.1,
        color="g",
    )

    plt.plot(train_sizes, train_errors_mean, "o-", color="r", label="Training error")
    plt.plot(
        train_sizes, test_errors_mean, "o-", color="g", label="Cross-validation error"
    )

    plt.legend(loc="best")
    return plt


mlp = MLPClassifier(random_state=42)
title = "Learning Curves (Neural Network) - Heart Disease"
plot_learning_curve(mlp, title, X_train, y_train, cv=5, n_jobs=4)
plt.show()

svm = SVC(random_state=42)
title = "Learning Curves (SVM) - Heart Disease"
plot_learning_curve(svm, title, X_train, y_train, cv=5, n_jobs=4)
plt.show()

knn = KNeighborsClassifier()
title = "Learning Curves (KNN) - Heart Disease"
plot_learning_curve(knn, title, X_train, y_train, cv=5, n_jobs=4)
plt.show()

import time

# Track training time for Neural Network
start_time = time.time()
mlp.fit(X_train, y_train)
mlp_training_time = time.time() - start_time
print(f"Neural Network Training Time: {mlp_training_time} seconds")
print(f"Neural Network Iterations: {mlp.n_iter_}")

# Track training time for SVM
start_time = time.time()
svm.fit(X_train, y_train)
svm_training_time = time.time() - start_time
print(f"SVM Training Time: {svm_training_time} seconds")
print(
    f"SVM Iterations: {svm.n_iter_ if hasattr(svm, 'n_iter_') else 'Not available for this solver'}"
)


def plot_learning_curve_iterations(
    estimator, title, X_train, y_train, X_test, y_test, iteration_range
):
    train_errors = []
    test_errors = []

    for n_iter in iteration_range:
        # Set the maximum number of iterations for the model
        if isinstance(estimator, MLPClassifier):
            estimator.set_params(max_iter=n_iter)
        elif isinstance(estimator, SVC):
            estimator.set_params(max_iter=n_iter)

        # Train the model
        estimator.fit(X_train, y_train)

        # Calculate training error
        train_score = estimator.score(X_train, y_train)
        train_errors.append(1 - train_score)

        # Calculate cross-validation error
        test_score = estimator.score(X_test, y_test)
        test_errors.append(1 - test_score)

    # Plot learning curves
    plt.figure()
    plt.title(title)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Error Rate")
    plt.plot(iteration_range, train_errors, "o-", color="r", label="Training error")
    plt.plot(
        iteration_range, test_errors, "o-", color="g", label="Cross-validation error"
    )
    plt.legend(loc="best")
    plt.grid()
    plt.show()


# Define the range of iterations to test
iteration_range = range(1, 700, 50)
iteration_range2 = range(1, 200, 10)

# Plot learning curve for MLPClassifier
mlp = MLPClassifier(random_state=42)
plot_learning_curve_iterations(
    mlp,
    "Learning Curves (Neural Network) - Heart Disease - Iterations",
    X_train,
    y_train,
    X_test,
    y_test,
    iteration_range,
)

# Plot learning curve for SVC
svm = SVC(random_state=42)
plot_learning_curve_iterations(
    svm,
    "Learning Curves (SVM) - Heart Disease - Iterations",
    X_train,
    y_train,
    X_test,
    y_test,
    iteration_range2,
)
