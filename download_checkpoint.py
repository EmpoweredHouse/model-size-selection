import os
import mlflow
from dotenv import load_dotenv, find_dotenv

# Load environment variables
try:
    load_dotenv(find_dotenv())
except Exception:
    pass

# Set MLflow tracking URI for Databricks
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"

# Your artifact URI here - replace with actual URI
artifact_uri = "dbfs:/databricks/mlflow-tracking/<rest of the uri>"  # Update this

# Create checkpoints directory
checkpoint_dir = "/workspace/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Download artifacts
try:
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=checkpoint_dir)
    print(f"Downloaded checkpoint to: {local_path}")
except Exception as e:
    print(f"Failed to download artifacts from URI: {artifact_uri}. Error: {e}")
