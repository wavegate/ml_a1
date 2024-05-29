import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and preprocess the stroke dataset
data = pd.read_csv("stroke.csv")
data = data.drop(columns=["id"])
data = data.dropna(subset=["bmi"])

data["bmi"].fillna(data["bmi"].mean(), inplace=True)
# Separate features and target variable
X = data.drop("stroke", axis=1)
y = data["stroke"]

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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Define the model
svc = SVC(random_state=42)

# Generate validation curves for C parameter
param_range = np.logspace(-3, 2, 6)
train_scores, test_scores = validation_curve(
    svc,
    X_train,
    y_train,
    param_name="C",
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
plt.title("Model Complexity Graph (SVM) - Stroke")
plt.xlabel("C")
plt.ylabel("Error Rate")
plt.xscale("log")
plt.grid()
plt.fill_between(
    param_range,
    train_errors_mean - train_errors_std,
    train_errors_mean + train_errors_std,
    alpha=0.1,
    color="r",
)
plt.fill_between(
    param_range,
    test_errors_mean - test_errors_std,
    test_errors_mean + test_errors_std,
    alpha=0.1,
    color="g",
)
plt.plot(param_range, train_errors_mean, "o-", color="r", label="Training error")
plt.plot(param_range, test_errors_mean, "o-", color="g", label="Cross-validation error")
plt.legend(loc="best")
plt.show()
