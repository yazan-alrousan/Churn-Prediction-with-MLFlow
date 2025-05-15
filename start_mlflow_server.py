import subprocess
import os
import platform

# --- Configuration ---
MLFLOW_DATA_PARENT_DIR = "mlflow_server_data"  # Directory to store all MLflow server related data
BACKEND_STORE_FILENAME = "mlflow.db"
ARTIFACTS_DIR_NAME = "artifacts"

# Construct full paths
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
MLFLOW_DATA_DIR = os.path.join(WORKSPACE_ROOT, MLFLOW_DATA_PARENT_DIR)

# SQLite backend store URI
backend_store_uri = f"sqlite:///{os.path.join(MLFLOW_DATA_DIR, BACKEND_STORE_FILENAME)}"

# Artifact root path with proper file URI format (important for MLflow)
raw_artifact_path = os.path.abspath(os.path.join(MLFLOW_DATA_DIR, ARTIFACTS_DIR_NAME))
default_artifact_root_path = f"file:///{raw_artifact_path.replace(os.sep, '/')}"

# Server configuration
HOST = "127.0.0.1"
PORT = "5000"

def prepare_directories():
    """Create necessary directories for MLflow server."""
    if not os.path.exists(MLFLOW_DATA_DIR):
        os.makedirs(MLFLOW_DATA_DIR)
        print(f"Created MLflow base data directory: {MLFLOW_DATA_DIR}")

    db_dir = os.path.dirname(os.path.join(MLFLOW_DATA_DIR, BACKEND_STORE_FILENAME))
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # Strip file:// prefix for actual directory creation
    local_artifact_path = raw_artifact_path
    if not os.path.exists(local_artifact_path):
        os.makedirs(local_artifact_path)
        print(f"Created MLflow artifact directory: {local_artifact_path}")

def start_mlflow_server():
    """Constructs and runs the MLflow server command."""
    prepare_directories()

    command = [
        "mlflow", "server",
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", default_artifact_root_path,
        "--host", HOST,
        "--port", PORT
    ]

    print("\nStarting MLflow Tracking Server...")
    print(f"  Data directory: {MLFLOW_DATA_DIR}")
    print(f"  Backend Store URI (SQLite): {backend_store_uri}")
    print(f"  Default Artifact Root: {default_artifact_root_path}")
    print(f"  UI will be available at: http://{HOST}:{PORT}")
    print("\nCommand to be executed:")
    print(f"  {' '.join(command)}")
    print("\nServer will run in the foreground. Press Ctrl+C in this terminal to stop.")
    print("Ensure you have 'mlflow' installed and in your PATH (e.g., via 'pip install mlflow').")

    try:
        is_windows = platform.system() == "Windows"
        subprocess.run(command, check=True, shell=is_windows)
    except subprocess.CalledProcessError as e:
        print(f"Error starting MLflow server: {e}")
        print("This might be due to the port already being in use or other configuration issues.")
    except FileNotFoundError:
        print("Error: 'mlflow' command not found.")
        print("Please ensure MLflow is installed and the 'mlflow' command is in your system's PATH.")
        print("You can typically install it using: pip install mlflow")
    except KeyboardInterrupt:
        print("\nMLflow server stopped by user (Ctrl+C).")

if __name__ == "__main__":
    start_mlflow_server()


