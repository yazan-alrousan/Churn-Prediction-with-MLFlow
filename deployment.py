import numpy as np
import pickle
import mlflow.lightgbm

class CustomerChurnLGBMPredictor:
    def __init__(self, model_uri: str, encoder_path: str):
        # Load LightGBM model from MLflow
        self.model = mlflow.lightgbm.load_model(model_uri)

        # Load encoder
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)

    def predict(
        self,
        age: int,
        tenure: int,
        usage_frequency: int,
        support_calls: int,
        payment_delay: int,
        total_spend: float,
        last_interaction: int,
        gender: str,
        subscription_type: str,
        contract_length: str
    ) -> str:
        # Validate input types
        expected_types = [int, int, int, int, int, float, int, str, str, str]
        inputs = [age, tenure, usage_frequency, support_calls, payment_delay,
                  total_spend, last_interaction, gender, subscription_type, contract_length]
        field_names = ["age", "tenure", "usage_frequency", "support_calls", "payment_delay",
                       "total_spend", "last_interaction", "gender", "subscription_type", "contract_length"]

        for val, exp_type, name in zip(inputs, expected_types, field_names):
            if not isinstance(val, exp_type):
                raise TypeError(f"Expected '{name}' to be {exp_type.__name__}, got {type(val).__name__}.")

        # Validate categorical values
        valid_genders = ['Female', 'Male']
        valid_subscriptions = ['Standard', 'Basic', 'Premium']
        valid_contracts = ['Annual', 'Monthly', 'Quarterly']

        if gender not in valid_genders:
            raise ValueError(f"Invalid gender: {gender}. Must be one of {valid_genders}.")
        if subscription_type not in valid_subscriptions:
            raise ValueError(f"Invalid subscription_type: {subscription_type}. Must be one of {valid_subscriptions}.")
        if contract_length not in valid_contracts:
            raise ValueError(f"Invalid contract_length: {contract_length}. Must be one of {valid_contracts}.")

        # One-Hot Encode
        ohe_vector = self.encoder.transform([[gender, subscription_type, contract_length]])
        ohe_vector = ohe_vector[0].tolist()

        # Combine with numeric features
        numeric_features = [age, tenure, usage_frequency, support_calls,
                            payment_delay, total_spend, last_interaction]
        final_input = np.array(numeric_features + ohe_vector).reshape(1, -1)

        # Predict
        proba = self.model.predict(final_input)[0]

        # Use 0.5 threshold for classification; you can tune this based on recall
        return "Will Churn" if proba >= 0.5 else "Won't Churn"
