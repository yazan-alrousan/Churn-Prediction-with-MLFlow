import os
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
import mlflow
import mlflow.sklearn
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

# Save encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Train-test split (with stratification)
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

# Scaling and PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Load best PCA component count
with open("best_pca_components.pkl", "rb") as f:
    best_n_components = pickle.load(f)

# Apply PCA
pca = PCA(n_components=best_n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)


# MLflow tracking setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Optuna_RF_Churn2"
mlflow.set_experiment(EXPERIMENT_NAME)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_val_pca)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

    return recall

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="rf_churn_recall")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
