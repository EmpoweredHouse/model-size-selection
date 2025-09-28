import os
import argparse
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

# Optional CLI override for artifact_uri
parser = argparse.ArgumentParser(description="Download MLflow artifacts to a local directory")
parser.add_argument(
    "--artifact-uri",
    dest="cli_artifact_uri",
    default=None,
    help="Override the default artifact URI defined in the script",
)
args = parser.parse_args()
if args.cli_artifact_uri:
    artifact_uri = args.cli_artifact_uri

# Create checkpoints directory
checkpoint_dir = "/workspace/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Download artifacts
try:
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=checkpoint_dir)
    print(f"Downloaded checkpoint to: {local_path}")
except Exception as e:
    print(f"Failed to download artifacts from URI: {artifact_uri}. Error: {e}")
