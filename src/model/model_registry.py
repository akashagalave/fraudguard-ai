import json
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

dagshub.init(
    repo_owner="akashagalaveaaa1",
    repo_name="fraudguard-ai",
    mlflow=True
)

MODEL_NAME = "fraudguard_lightgbm"

def main():
    client = MlflowClient()

    with open("run_information.json") as f:
        run_info = json.load(f)

    model_version = mlflow.register_model(
        model_uri=run_info["model_uri"],
        name=MODEL_NAME
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=True
    )

    print(f"Model registered as {MODEL_NAME} v{model_version.version}")

if __name__ == "__main__":
    main()
