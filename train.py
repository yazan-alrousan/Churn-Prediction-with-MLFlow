import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
from sklearn.decomposition import PCA

# --- MLflow Config ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Churn_Classification_FeatReduc3"
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Data Loading ---
df = pd.concat([
    pd.read_csv("C:/Users/yazan/OneDrive/Desktop/MLOps/pythonProject/datasets/customer_churn_dataset-training-master.csv"),
    pd.read_csv("C:/Users/yazan/OneDrive/Desktop/MLOps/pythonProject/datasets/customer_churn_dataset-testing-master.csv")
], axis=0)

df.drop(columns='CustomerID', inplace=True)
df.columns = [col.lower().replace(" ", "_") for col in df.columns]
df.dropna(inplace=True)

# ✅ Limit to the 10 specified features + target
selected_features = [
    'age', 'gender', 'tenure', 'usage_frequency', 'support_calls',
    'payment_delay', 'subscription_type', 'contract_length',
    'total_spend', 'last_interaction', 'churn'
]
df = df[selected_features]

print("✅ Features used for training:")
print(df.drop(columns='churn').columns.tolist())

# Convert discrete features to int
discrete_cols = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'last_interaction', 'churn']
for col in discrete_cols:
    df[col] = df[col].astype(int)

# --- Stratified train/val/test split ---
X = df.drop(columns='churn')
y = df['churn']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)

for df_ in [X_train, X_val, X_test, y_train, y_val, y_test]:
    df_.reset_index(drop=True, inplace=True)

print("\n✅ Class distribution:")
print(f"Train churn ratio: {y_train.mean():.4f}")
print(f"Val churn ratio:   {y_val.mean():.4f}")
print(f"Test churn ratio:  {y_test.mean():.4f}")

# --- One-Hot Encoding ---
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[['gender', 'subscription_type', 'contract_length']])
feature_names = encoder.get_feature_names_out(['gender', 'subscription_type', 'contract_length'])

train_ohe = pd.DataFrame(encoder.transform(X_train[['gender', 'subscription_type', 'contract_length']]), columns=feature_names)
X_train.drop(columns=['gender', 'subscription_type', 'contract_length'], inplace=True)
X_train = pd.concat([X_train, train_ohe], axis=1)

val_ohe = pd.DataFrame(encoder.transform(X_val[['gender', 'subscription_type', 'contract_length']]), columns=feature_names)
X_val.drop(columns=['gender', 'subscription_type', 'contract_length'], inplace=True)
X_val = pd.concat([X_val, val_ohe], axis=1)

test_ohe = pd.DataFrame(encoder.transform(X_test[['gender', 'subscription_type', 'contract_length']]), columns=feature_names)
X_test.drop(columns=['gender', 'subscription_type', 'contract_length'], inplace=True)
X_test = pd.concat([X_test, test_ohe], axis=1)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# --- Scaling + PCA ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

with open('best_pca_components.pkl', 'rb') as f:
    n_components = pickle.load(f)

pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\n✅ PCA applied:")
print(f"Explained variance ratio (sum): {sum(pca.explained_variance_ratio_):.4f}")
print(f"Original feature count: {X_train.shape[1]} → After PCA: {X_train_pca.shape[1]}")

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
        model.fit(X_train_pca, y_train)
        preds = model.predict(X_test_pca)

        metrics = log_metrics(y_test, preds)
        mlflow.log_params(model.get_params())
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow_module.log_model(model, "model", registered_model_name=model_name)
        print(f"\n{model_name} Metrics:")
        for metric, val in metrics.items():
            print(f"{metric}: {val:.4f}")

# --- Train and Track Models ---
train_and_log_model(RandomForestClassifier(n_estimators=100, random_state=42), "RandomForest_PCA", mlflow.sklearn)
train_and_log_model(LGBMClassifier(random_state=42), "LightGBM_PCA", mlflow.lightgbm)
train_and_log_model(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "XGBoost_PCA", mlflow.xgboost)
