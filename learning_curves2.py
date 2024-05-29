import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv("stroke.csv")

data = data.drop(columns=["id"])

data = data.dropna(subset=["bmi"])

data["bmi"].fillna(data["bmi"].mean(), inplace=True)

# Separate features and target variable
X = data.drop("stroke", axis=1)
y = data["stroke"]

print(X)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

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


mlp = MLPClassifier(max_iter=1000, random_state=42)
title = "Learning Curves (Neural Network) - Stroke"
plot_learning_curve(mlp, title, X_train, y_train, cv=5, n_jobs=4)
plt.show()

svm = SVC(random_state=42)
title = "Learning Curves (SVM) - Stroke"
plot_learning_curve(svm, title, X_train, y_train, cv=5, n_jobs=4)
plt.show()

knn = KNeighborsClassifier()
title = "Learning Curves (KNN) - Stroke"
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
iteration_range = range(1, 1001, 100)
iteration_range2 = range(1, 200, 10)

# Plot learning curve for MLPClassifier
mlp = MLPClassifier(random_state=42)
plot_learning_curve_iterations(
    mlp,
    "Learning Curves (Neural Network) - Stroke - Iterations",
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
    "Learning Curves (SVM) - Stroke - Iterations",
    X_train,
    y_train,
    X_test,
    y_test,
    iteration_range2,
)
