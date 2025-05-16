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
MODEL_NAME = "CustomerChurnClassifier_RFC"  # Replace with your final model name if needed
MODEL_STAGE = "Production"
model = mlflow.sklearn.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Expected categorical columns
cat_cols = ["gender", "subscription_type", "contract_length"]

# Prepare input data for prediction
def prepare_input(age, tenure, usage_frequency, support_calls, payment_delay,
                  total_spend, last_interaction, gender, subscription_type, contract_length):
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
    pca_input = pca.transform(scaled)  # Only transform, do NOT fit again
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
        gr.Number(label="Age", info="The age of the customer"),
        gr.Number(label="Tenure (months)", info="Duration in months for which a customer has been using the company's products or services"),
        gr.Number(label="Usage Frequency", info="Number of times the customer used the company’s services in the last month"),
        gr.Number(label="Support Calls", info="Number of customer support calls in the last month"),
        gr.Number(label="Payment Delay (days)", info="Days delayed in payment for the last month"),
        gr.Number(label="Total Spend", info="Total money the customer has spent on company services"),
        gr.Number(label="Days Since Last Interaction", info="Number of days since last customer interaction"),
        gr.Radio(["Male", "Female"], label="Gender", info="Gender of the customer"),
        gr.Radio(["Basic", "Standard", "Premium"], label="Subscription Type", info="Customer’s selected subscription plan"),
        gr.Radio(["Monthly", "Quarterly", "Annual"], label="Contract Length", info="The duration of the signed customer contract")
    ],
    outputs="text",
    title="Customer Churn Prediction",
    description="Enter customer details to predict whether they will churn. Hover over each input for more information."
)

if __name__ == "__main__":
    demo.launch()
