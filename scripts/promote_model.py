import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import logging


dagshub.init(
    repo_owner="akashagalaveaaa1",
    repo_name="fraudguard-ai",
    mlflow=True
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promote_model")

MODEL_NAME = "fraudguard_lightgbm"
SOURCE_STAGE = "Staging"
TARGET_STAGE = "Production"


def main():
    client = MlflowClient()

    logger.info(
        f"Fetching latest model version from stage: {SOURCE_STAGE}"
    )

    versions = client.get_latest_versions(
        name=MODEL_NAME,
        stages=[SOURCE_STAGE]
    )

    if not versions:
        raise RuntimeError(
            f"No model found in stage '{SOURCE_STAGE}' for '{MODEL_NAME}'"
        )

    model_version = versions[0].version

    logger.info(
        f"Promoting model '{MODEL_NAME}' version {model_version} "
        f"from {SOURCE_STAGE} → {TARGET_STAGE}"
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage=TARGET_STAGE,
        archive_existing_versions=True
    )

    logger.info(
        f" Model '{MODEL_NAME}' v{model_version} is now in {TARGET_STAGE}"
    )


if __name__ == "__main__":
    main()
