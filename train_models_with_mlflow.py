import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost

# Load datasets
train_path = r"C:\Users\yazan\OneDrive\Desktop\MLOps\pythonProject\datasets\customer_churn_dataset-training-master.csv"
test_path = r"C:\Users\yazan\OneDrive\Desktop\MLOps\pythonProject\datasets\customer_churn_dataset-testing-master.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_train = df_train.dropna(subset=["Churn"])
df_test = df_test.dropna(subset=["Churn"])

# Define features and target
features = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction"
]
target = "Churn"

if target not in df_train.columns or target not in df_test.columns:
    raise ValueError(f"Target column '{target}' not found in dataset.")

X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

# Impute missing values in features
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Evaluation metrics
def log_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

# Model training and logging
def train_and_log_model(model, model_name, model_type="sklearn"):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = log_metrics(y_test, preds)

        mlflow.log_params(model.get_params())

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, "model", registered_model_name=model_name)
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, "model", registered_model_name=model_name)

        print(f"{model_name} metrics:", metrics)

# Set experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Churn_Prediction_7"

try:
    mlflow.create_experiment(EXPERIMENT_NAME)
except mlflow.exceptions.MlflowException:
    print(f"Experiment '{EXPERIMENT_NAME}' already exists.")

mlflow.set_experiment(EXPERIMENT_NAME)

# Train selected models
print("X_train columns:", X_train.columns.tolist())

train_and_log_model(RandomForestClassifier(n_estimators=100, random_state=42), "Random_Forest")
train_and_log_model(LGBMClassifier(random_state=42), "LightGBM", model_type="lightgbm")
train_and_log_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "XGBoost", model_type="xgboost")
