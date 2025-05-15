import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # Change this if your MLflow server uses a different host/port
EXPERIMENT_NAME = "CustomerChurn_Model_Registry"
RUN_ID = "c7d55974debc48a9bacb0b37862440ec"
MODEL_NAME = "CustomerChurnClassifier_LGBM"
ARTIFACT_PATH = "model"  # Path within the run where the model was logged

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Optional: create experiment if not exists
if not client.get_experiment_by_name(EXPERIMENT_NAME):
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# Register the model
model_uri = f"runs:/{RUN_ID}/{ARTIFACT_PATH}"

try:
    print(f"Registering model from URI: {model_uri}")
    registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Model registered successfully under name: {MODEL_NAME}")
    print(f"Version: {registered_model.version}")

    # Optional: Transition the registered model to "Production"
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered_model.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model transitioned to stage: Production")

except Exception as e:
    print("Error during model registration:", str(e))
