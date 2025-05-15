import pandas as pd

# Paths to datasets
train_path = "datasets/customer_churn_dataset-training-master.csv"
test_path = "datasets/customer_churn_dataset-testing-master.csv"

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Encoding maps
gender_map = {"Male": 0, "Female": 1}
subscription_map = {"Standard": 0, "Basic": 1, "Premium": 2}
contract_map = {"Monthly": 0, "Annual": 1, "Quarterly": 2}

# Apply encoding to training set
train_df["Gender"] = train_df["Gender"].map(gender_map)
train_df["Subscription Type"] = train_df["Subscription Type"].map(subscription_map)
train_df["Contract Length"] = train_df["Contract Length"].map(contract_map)

# Apply encoding to testing set
test_df["Gender"] = test_df["Gender"].map(gender_map)
test_df["Subscription Type"] = test_df["Subscription Type"].map(subscription_map)
test_df["Contract Length"] = test_df["Contract Length"].map(contract_map)

print("Encoded Training DataFrame:")
print(train_df.head())
print("\nEncoded Testing DataFrame:")
print(test_df.head())