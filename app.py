import gradio as gr
import pandas as pd
import mlflow

# Load model
mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_NAME = "CustomerChurnClassifier_LGBM"
MODEL_STAGE = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Expected one-hot encoded features in correct order
FEATURE_ORDER = [
    'age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay',
    'total_spend', 'last_interaction',
    'gender_Female', 'gender_Male',
    'subscription_type_Basic', 'subscription_type_Premium', 'subscription_type_Standard',
    'contract_length_Annual', 'contract_length_Monthly', 'contract_length_Quarterly'
]

# Input processing function
def prepare_input(age, tenure, usage_frequency, support_calls, payment_delay,
                  total_spend, last_interaction, gender, subscription_type, contract_length):
    # Base features
    input_dict = {
        "age": age,
        "tenure": tenure,
        "usage_frequency": usage_frequency,
        "support_calls": support_calls,
        "payment_delay": payment_delay,
        "total_spend": total_spend,
        "last_interaction": last_interaction,
        f"gender_{gender}": 1,
        f"subscription_type_{subscription_type}": 1,
        f"contract_length_{contract_length}": 1
    }

    # Fill missing one-hot encodings with 0
    full_input = {feature: input_dict.get(feature, 0) for feature in FEATURE_ORDER}

    return pd.DataFrame([full_input])

# Gradio prediction function
def predict_churn(age, tenure, usage_frequency, support_calls, payment_delay,
                  total_spend, last_interaction, gender, subscription_type, contract_length):
    try:
        input_df = prepare_input(age, tenure, usage_frequency, support_calls, payment_delay,
                                 total_spend, last_interaction, gender, subscription_type, contract_length)
        prediction = model.predict(input_df)[0]
        return "Will Churn" if prediction > 0.5 else "Won't Churn"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Usage Frequency (last month)"),
        gr.Number(label="Support Calls (last month)"),
        gr.Number(label="Payment Delay (days)"),
        gr.Number(label="Total Spend"),
        gr.Number(label="Days Since Last Interaction"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["Basic", "Standard", "Premium"], label="Subscription Type"),
        gr.Radio(["Monthly", "Quarterly", "Annual"], label="Contract Length")
    ],
    outputs="text",
    title="Customer Churn Prediction",
    description="Fill in the customer details to predict whether they will churn."
)

if __name__ == "__main__":
    demo.launch()
