import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost

# --- MLflow Config ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Churn_Classification_MLflow"
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Data Loading ---
df = pd.concat([
    pd.read_csv("C:/Users/yazan/OneDrive/Desktop/MLOps/pythonProject/datasets/customer_churn_dataset-training-master.csv"),
    pd.read_csv("C:/Users/yazan/OneDrive/Desktop/MLOps/pythonProject/datasets/customer_churn_dataset-testing-master.csv")
], axis=0)

df.drop(columns='CustomerID', inplace=True)
df.columns = [col.lower().replace(" ", "_") for col in df.columns]
df.dropna(inplace=True)

# Convert discrete features to int
discrete_cols = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'last_interaction', 'churn']
for col in discrete_cols:
    df[col] = df[col].astype(int)

# Train-test split
X = df.drop(columns='churn')
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Reset index
for df_ in [X_train, X_test, y_train, y_test]:
    df_.reset_index(drop=True, inplace=True)

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[['gender', 'subscription_type', 'contract_length']])
feature_names = encoder.get_feature_names_out(['gender', 'subscription_type', 'contract_length'])

train_ohe = pd.DataFrame(encoder.transform(X_train[['gender', 'subscription_type', 'contract_length']]), columns=feature_names)
test_ohe = pd.DataFrame(encoder.transform(X_test[['gender', 'subscription_type', 'contract_length']]), columns=feature_names)

X_train.drop(columns=['gender', 'subscription_type', 'contract_length'], inplace=True)
X_test.drop(columns=['gender', 'subscription_type', 'contract_length'], inplace=True)

X_train = pd.concat([X_train, train_ohe], axis=1)
X_test = pd.concat([X_test, test_ohe], axis=1)

# Save encoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# --- Utility Functions ---
def log_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": 2 * (precision_score(y_true, y_pred, zero_division=0) * recall_score(y_true, y_pred, zero_division=0)) /
                    (precision_score(y_true, y_pred, zero_division=0) + recall_score(y_true, y_pred, zero_division=0) + 1e-9)
    }

def train_and_log_model(model, model_name, mlflow_module):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = log_metrics(y_test, preds)
        mlflow.log_params(model.get_params())
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow_module.log_model(model, "model", registered_model_name=model_name)
        print(f"\n{model_name} Metrics:")
        for metric, val in metrics.items():
            print(f"{metric}: {val:.4f}")

# --- Train and Track Models ---
train_and_log_model(RandomForestClassifier(n_estimators=100, random_state=42), "RandomForest", mlflow.sklearn)
train_and_log_model(LGBMClassifier(random_state=42), "LightGBM", mlflow.lightgbm)
train_and_log_model(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "XGBoost", mlflow.xgboost)
