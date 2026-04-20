from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import mlflow
import structlog

logger = structlog.get_logger()


def _setup_dagshub() -> str:
    """Configure MLflow tracking via DagsHub if DAGSHUB_TOKEN is set."""
    import dagshub
    token = os.environ["DAGSHUB_TOKEN"]
    dagshub.auth.add_app_token(token=token)
    dagshub.init(repo_owner="tdutra-dev", repo_name="visual-finetune-lab", mlflow=True)
    tracking_uri = mlflow.get_tracking_uri()
    logger.info("dagshub_configured", tracking_uri=tracking_uri)
    return tracking_uri


class ExperimentTracker:
    """
    Thin wrapper around MLflow to track fine-tuning experiments.

    Supports three backends (set MLFLOW_TRACKING_URI):
      - "dagshub"         → DagsHub remote tracking (requires DAGSHUB_TOKEN)
      - "sqlite:///..."   → local SQLite file
      - "http://..."      → self-hosted MLflow server

    Example:
        tracker = ExperimentTracker("visual-finetune-lab")
        with tracker.run("lora-r16-invoices") as run:
            tracker.log_params({"lora_rank": 16, "epochs": 3})
            tracker.log_metrics({"train_loss": 0.45, "eval_loss": 0.51})
            tracker.log_artifact(Path("checkpoints/best"))
    """

    def __init__(self, experiment_name: str = "visual-finetune-lab"):
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        if tracking_uri == "dagshub":
            tracking_uri = _setup_dagshub()
        else:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info("mlflow_configured", tracking_uri=tracking_uri, experiment=experiment_name)

    @contextmanager
    def run(self, run_name: str):
        with mlflow.start_run(run_name=run_name) as active_run:
            logger.info("mlflow_run_started", run_id=active_run.info.run_id)
            yield active_run
            logger.info("mlflow_run_ended", run_id=active_run.info.run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def log_eval_results(self, results: list) -> None:
        if not results:
            return
        n = len(results)
        mlflow.log_metrics({
            "eval/bleu_avg": sum(r.bleu for r in results) / n,
            "eval/rouge_l_avg": sum(r.rouge_l for r in results) / n,
            "eval/llm_judge_avg": sum(r.llm_judge_score for r in results) / n,
        })
        judges = [r.llm_judge_score for r in results]
        mlflow.log_metrics({
            "eval/llm_judge_ge4_pct": sum(1 for s in judges if s >= 4) / n,
        })
