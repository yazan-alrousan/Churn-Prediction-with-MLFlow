import os
import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.lightgbm
import pickle

# Paths to your dataset files
train_path = r"C:\Users\yazan\OneDrive\Desktop\MLOps\pythonProject\datasets\customer_churn_dataset-training-master.csv"
test_path = r"C:\Users\yazan\OneDrive\Desktop\MLOps\pythonProject\datasets\customer_churn_dataset-testing-master.csv"

# Load and concatenate the dataset
df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)
df.dropna(subset=["Churn"], inplace=True)

# Basic preprocessing
df.drop(columns="CustomerID", errors="ignore", inplace=True)
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# Convert discrete columns to int
discrete_cols = ["age", "tenure", "usage_frequency", "support_calls", "payment_delay", "last_interaction", "churn"]
for col in discrete_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Define features and target
features = [
    "age", "tenure", "usage_frequency", "support_calls", "payment_delay",
    "total_spend", "last_interaction", "gender", "subscription_type", "contract_length"
]
target = "churn"

X = df[features]
y = df[target]

# One-hot encode categorical columns
cat_cols = ["gender", "subscription_type", "contract_length"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(X[cat_cols])

encoded_cols = encoder.get_feature_names_out(cat_cols)
X_encoded = encoder.transform(X[cat_cols])
X_cat = pd.DataFrame(X_encoded, columns=encoded_cols)

X_num = X.drop(columns=cat_cols).copy()
X_num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_num), columns=X_num.columns)

X_preprocessed = pd.concat([X_num, X_cat], axis=1)

# Save encoder if needed later
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

# MLflow tracking setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Optuna_LightGBM_Churn"
mlflow.set_experiment(EXPERIMENT_NAME)

def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    recall = recall_score(y_val, y_pred)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("recall", recall)
        mlflow.lightgbm.log_model(model, "model")

    return recall

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="lgbm_churn_recall")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
