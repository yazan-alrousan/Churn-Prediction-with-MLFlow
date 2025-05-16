import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
import pickle

# Load and preprocess data
df = pd.concat([
    pd.read_csv(
        "C:/Users/yazan/OneDrive/Desktop/MLOps/pythonProject/datasets/customer_churn_dataset-training-master.csv"),
    pd.read_csv(
        "C:/Users/yazan/OneDrive/Desktop/MLOps/pythonProject/datasets/customer_churn_dataset-testing-master.csv")
], axis=0)

df.drop(columns='CustomerID', inplace=True)
df.columns = [col.lower().replace(" ", "_") for col in df.columns]
df.dropna(inplace=True)

selected_features = [
    'age', 'gender', 'tenure', 'usage_frequency', 'support_calls',
    'payment_delay', 'subscription_type', 'contract_length',
    'total_spend', 'last_interaction', 'churn'
]
df = df[selected_features]

# ✅ Subsample for faster PCA tuning
df = df.sample(n=30000, random_state=42)


# Encode and cast
discrete_cols = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'last_interaction', 'churn']
for col in discrete_cols:
    df[col] = df[col].astype(int)

X = df.drop(columns='churn')
y = df['churn']

# Preprocessing pipeline
categorical_features = ['gender', 'subscription_type', 'contract_length']
numeric_features = X.drop(columns=categorical_features).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# Range of components to test
scores = []
component_range = range(2, 16)

for n_components in component_range:
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('pca', PCA(n_components=n_components)),
        ('clf', LGBMClassifier(random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(pipeline, X, y, cv=cv, scoring='f1').mean()
    scores.append((n_components, cv_score))
    print(f"PCA components: {n_components}, Mean F1: {cv_score:.4f}")

# Find best number of components
best_n, best_score = max(scores, key=lambda x: x[1])
print(f"\n✅ Best number of PCA components: {best_n} (F1: {best_score:.4f})")

# Save the best value
with open('best_pca_components.pkl', 'wb') as f:
    pickle.dump(best_n, f)
