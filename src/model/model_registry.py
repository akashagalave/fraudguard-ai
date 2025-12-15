import mlflow
import logging
import dagshub
from mlflow.tracking import MlflowClient

dagshub.init(
    repo_owner="akashagalaveaaa1",
    repo_name="fraudguard-ai",
    mlflow=True
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_registry")

MODEL_NAME = "fraudguard_lightgbm"
EVAL_EXPERIMENT = "fraudguard_lightgbm_evaluation"
TRAIN_EXPERIMENT = "fraudguard_lightgbm_training"
METRIC = "pr_auc"

def main():
    client = MlflowClient()

    logger.info("Fetching evaluation experiment")
    eval_exp = client.get_experiment_by_name(EVAL_EXPERIMENT)
    if eval_exp is None:
        raise RuntimeError("Evaluation experiment not found")

    eval_runs = client.search_runs(
        experiment_ids=[eval_exp.experiment_id],
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1
    )

    if not eval_runs:
        raise RuntimeError("No evaluation runs found")

    best_eval_run = eval_runs[0]
    best_score = best_eval_run.data.metrics[METRIC]

    logger.info(f"Best PR-AUC from evaluation: {best_score}")

    logger.info("Fetching latest training run")
    train_exp = client.get_experiment_by_name(TRAIN_EXPERIMENT)
    if train_exp is None:
        raise RuntimeError("Training experiment not found")

    train_runs = client.search_runs(
        experiment_ids=[train_exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not train_runs:
        raise RuntimeError("No training runs found")

    train_run = train_runs[0]
    train_run_id = train_run.info.run_id

    logger.info(f"Registering model from training run: {train_run_id}")

    model_uri = f"runs:/{train_run_id}/model"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=True
    )

    logger.info(
        f"Model '{MODEL_NAME}' version {model_version.version} promoted to STAGING"
    )

if __name__ == "__main__":
    main()
