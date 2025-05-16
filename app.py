import gradio as gr
import pandas as pd
import mlflow
import pickle

# Load preprocessing objects
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Load model from MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_NAME = "CustomerChurnClassifier_RFC"
MODEL_STAGE = "Production"
model = mlflow.sklearn.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Expected categorical columns
cat_cols = ["gender", "subscription_type", "contract_length"]


# Prepare input data for prediction
def prepare_input(age, tenure, usage_frequency, support_calls, payment_delay,
                  total_spend, last_interaction, gender, subscription_type, contract_length):
    # Input validation
    if age < 18:
        raise ValueError("Age must be 18 or older.")
    if tenure < 1:
        raise ValueError("Tenure must be at least 1 month.")
    if usage_frequency < 1:
        raise ValueError("Usage Frequency must be at least 1.")
    if support_calls < 0:
        raise ValueError("Support Calls cannot be negative.")
    if not (0 <= payment_delay <= 30):
        raise ValueError("Payment Delay must be between 0 and 30.")
    if total_spend < 100:
        raise ValueError("Total Spend must be at least 100.")
    if last_interaction < 0:
        raise ValueError("Last Interaction cannot be negative.")

    num_input = pd.DataFrame([{
        "age": age,
        "tenure": tenure,
        "usage_frequency": usage_frequency,
        "support_calls": support_calls,
        "payment_delay": payment_delay,
        "total_spend": total_spend,
        "last_interaction": last_interaction
    }])

    cat_input = pd.DataFrame([[gender, subscription_type, contract_length]], columns=cat_cols)
    cat_encoded = encoder.transform(cat_input)
    cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

    combined = pd.concat([num_input, cat_df], axis=1)
    scaled = scaler.transform(combined)
    pca_input = pca.transform(scaled)
    return pca_input


# Prediction function
def predict_churn(age, tenure, usage_frequency, support_calls, payment_delay,
                  total_spend, last_interaction, gender, subscription_type, contract_length):
    try:
        input_vector = prepare_input(age, tenure, usage_frequency, support_calls, payment_delay,
                                     total_spend, last_interaction, gender, subscription_type, contract_length)
        proba = model.predict_proba(input_vector)[0][1]
        label = "Will Churn" if proba >= 0.5 else "Won't Churn"
        confidence = f"{proba * 100:.2f}% confidence"
        return f"{label} ({confidence})"
    except Exception as e:
        return f"Error: {str(e)}"


# Gradio Interface
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Age", info="The age of the customer", minimum=18),
        gr.Number(label="Tenure (months)", info="Months using the service", minimum=1),
        gr.Number(label="Usage Frequency", info="Times used service last month", minimum=1),
        gr.Number(label="Support Calls", info="Customer support calls", minimum=0),
        gr.Number(label="Payment Delay (days)", info="Payment delay in days (0-30)", minimum=0, maximum=30),
        gr.Number(label="Total Spend", info="Total amount spent", minimum=100),
        gr.Number(label="Days Since Last Interaction", info="Days since last interaction", minimum=0),
        gr.Radio(["Male", "Female"], label="Gender", info="Gender of the customer"),
        gr.Radio(["Basic", "Standard", "Premium"], label="Subscription Type", info="Type of subscription"),
        gr.Radio(["Monthly", "Quarterly", "Annual"], label="Contract Length", info="Length of the contract")
    ],
    outputs="text",
    title="Customer Churn Prediction",
    description="Enter customer details to predict churn. Hover over inputs for more info."
)

if __name__ == "__main__":
    demo.launch()
