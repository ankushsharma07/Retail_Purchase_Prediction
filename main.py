import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data_cleaning import load_data, clean_data
from src.preprocessing import split_data, encode_data, split_train_test
from src.train_model import train_logistic_regression, train_random_forest


# Load dataset
file_path = "data/PS02_Retail_Customer_Purchase_data.csv"

print("Loading dataset...")

df = load_data(file_path)

print("Dataset shape:", df.shape)


# Clean the data
print("\nCleaning data...")

df = clean_data(df)

print("Missing values:", df.isnull().sum().sum())


# Separate features and target
X, y, groups = split_data(df)


# Convert categorical data into numbers
X = encode_data(X)

print("\nFeatures after encoding:", X.shape[1])


# Split data into training and testing
X_train, X_test, y_train, y_test = split_train_test(
    X, y, groups
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Train Logistic Regression
print("\nTraining Logistic Regression...")

logistic_model = train_logistic_regression(
    X_train,
    y_train
)

logistic_prediction = logistic_model.predict(X_test)


# Logistic Regression evaluation
logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

logistic_precision = precision_score(
    y_test,
    logistic_prediction
)

logistic_recall = recall_score(
    y_test,
    logistic_prediction
)

logistic_f1 = f1_score(
    y_test,
    logistic_prediction
)


# Train Random Forest
print("\nTraining Random Forest...")

random_forest_model = train_random_forest(
    X_train,
    y_train
)

random_forest_prediction = random_forest_model.predict(
    X_test
)


# Random Forest evaluation
random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_prediction
)

random_forest_precision = precision_score(
    y_test,
    random_forest_prediction
)

random_forest_recall = recall_score(
    y_test,
    random_forest_prediction
)

random_forest_f1 = f1_score(
    y_test,
    random_forest_prediction
)


# Show results
print("\nModel Results")

print("\nLogistic Regression")

print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1 Score :", round(logistic_f1, 4))


print("\nRandom Forest")

print("Accuracy :", round(random_forest_accuracy, 4))
print("Precision:", round(random_forest_precision, 4))
print("Recall   :", round(random_forest_recall, 4))
print("F1 Score :", round(random_forest_f1, 4))


# Confusion matrix
print("\nRandom Forest Confusion Matrix:")

matrix = confusion_matrix(
    y_test,
    random_forest_prediction
)

print(matrix)


# Save trained models
os.makedirs("models", exist_ok=True)

joblib.dump(
    logistic_model,
    "models/logistic_model.pkl"
)

joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)

joblib.dump(
    list(X.columns),
    "models/feature_columns.pkl"
)

print("\nModels saved successfully.")