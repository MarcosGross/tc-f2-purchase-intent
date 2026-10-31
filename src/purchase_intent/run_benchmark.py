"""Executa o benchmark comparativo (3 algoritmos x Com/Sem balanceamento).

Roda os 6 experimentos nos mesmos conjuntos de treino e teste, registra todas
as execuções no MLflow, promove o melhor modelo para o alias configurado no
Model Registry e exibe uma tabela Markdown pronta para o relatório da equipe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from purchase_intent.config import (
    configure_logging,
    load_data_parameters,
    load_params,
    load_settings,
    set_global_seed,
)
from purchase_intent.data_schema import read_dataset, write_json
from purchase_intent.data_split import prepare_dataset, split_features_target
from purchase_intent.evaluation import evaluate_pipeline
from purchase_intent.preprocessing import build_preprocessor, infer_feature_types
from purchase_intent.registry import build_model_uri, promote_model, register_model
from purchase_intent.train import (
    MODEL_ARTIFACT_PATH,
    build_estimator,
    build_pipeline,
    resolve_model_params,
    train_pipeline,
)

LOGGER = logging.getLogger(__name__)

# Grade dos 6 experimentos: 3 famílias de algoritmos x (Sem e Com balanceamento)
BENCHMARK_GRID: list[dict[str, Any]] = [
    {
        "label": "Logistic Regression (Sem Balanceamento)",
        "name": "logistic_regression",
        "balanced": False,
        "params": {"C": 1.0, "max_iter": 1000},
    },
    {
        "label": "Logistic Regression (Com Balanceamento)",
        "name": "logistic_regression",
        "balanced": True,
        "params": {"C": 1.0, "max_iter": 1000},
    },
    {
        "label": "Random Forest (Sem Balanceamento)",
        "name": "random_forest",
        "balanced": False,
        "params": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5},
    },
    {
        "label": "Random Forest (Com Balanceamento)",
        "name": "random_forest",
        "balanced": True,
        "params": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5},
    },
    {
        "label": "HistGradientBoosting (Sem Balanceamento)",
        "name": "hist_gradient_boosting",
        "balanced": False,
        "params": {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        },
    },
    {
        "label": "HistGradientBoosting (Com Balanceamento)",
        "name": "hist_gradient_boosting",
        "balanced": True,
        "params": {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        },
    },
]


def _ensure_splits_exist(
    raw_path: Path,
    processed_dir: Path,
    params_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Garante que train.csv e test.csv existem e os carrega."""
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        LOGGER.info("Arquivos processados ausentes; executando preparação dos dados...")
        data_params = load_data_parameters(params_path)
        prepare_dataset(raw_path, processed_dir, data_params)
    return read_dataset(train_path), read_dataset(test_path)


def _format_markdown_table(results: list[dict[str, Any]], primary_metric: str) -> str:
    """Formata os resultados dos 6 experimentos em uma tabela Markdown."""
    headers = (
        "| Experimento | Algoritmo | Balanceado | ROC-AUC | PR-AUC | "
        "Accuracy | Precision | Recall | F1-Score |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    rows: list[str] = [headers]
    best_value = max(item["metrics"][primary_metric] for item in results)

    for item in results:
        m = item["metrics"]
        is_best = m[primary_metric] == best_value
        prefix = "**" if is_best else ""
        suffix = "** 🏆" if is_best else ""
        balanced_str = "Sim (`balanced`)" if item["balanced"] else "Não"
        row = (
            f"| {prefix}{item['label']}{suffix} "
            f"| `{item['model_name']}` "
            f"| {balanced_str} "
            f"| {m['roc_auc']:.4f} "
            f"| {m['average_precision']:.4f} "
            f"| {m['accuracy']:.4f} "
            f"| {m['precision']:.4f} "
            f"| {m['recall']:.4f} "
            f"| **{m['f1']:.4f}** |"
        )
        rows.append(row)
    return "\n".join(rows)


def run_benchmark() -> None:
    """Executa a bateria de experimentos, registra no MLflow e salva relatório."""
    settings = load_settings()
    configure_logging(settings.log_level)
    params = load_params(settings.params_path)

    seed = int(params.get("seed", settings.random_seed))
    target_column = params["dataset"]["target_column"]
    threshold = float(params.get("evaluation", {}).get("decision_threshold", 0.5))
    primary_metric = str(params.get("evaluation", {}).get("primary_metric", "f1"))
    alias = str(params.get("registry", {}).get("alias", "champion"))

    train_df, test_df = _ensure_splits_exist(
        settings.raw_data_path,
        settings.processed_data_dir,
        settings.params_path,
    )
    x_train, y_train = split_features_target(train_df, target_column)
    x_test, y_test = split_features_target(test_df, target_column)
    numeric, categorical = infer_feature_types(x_train)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    results: list[dict[str, Any]] = []
    example = x_train.head(5)

    for scenario in BENCHMARK_GRID:
        set_global_seed(seed)
        model_name = scenario["name"]
        balanced = bool(scenario["balanced"])
        final_params = resolve_model_params(model_name, scenario["params"], balanced=balanced)

        preprocessor = build_preprocessor(numeric, categorical)
        estimator = build_estimator(model_name, final_params, seed=seed, balanced=balanced)
        pipeline = train_pipeline(build_pipeline(preprocessor, estimator), x_train, y_train)
        metrics = evaluate_pipeline(pipeline, x_test, y_test, threshold=threshold)

        balance_tag = "balanced" if balanced else "unbalanced"
        run_name = f"{model_name}_{balance_tag}"
        signature = infer_signature(example, pipeline.predict(example))

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "experiment.model_family": model_name,
                    "experiment.balancing": balance_tag,
                    "experiment.variant": run_name,
                    "evaluation.decision_threshold": str(threshold),
                }
            )
            mlflow.log_params(
                {
                    "model.name": model_name,
                    "model.balanced": str(balanced),
                    "seed": str(seed),
                    **{f"model.{k}": str(v) for k, v in final_params.items()},
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=MODEL_ARTIFACT_PATH,
                signature=signature,
                input_example=example,
            )
            run_id = run.info.run_id

        LOGGER.info(
            "[%s] F1=%.4f | Recall=%.4f | Precision=%.4f | PR-AUC=%.4f | ROC-AUC=%.4f",
            scenario["label"],
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["average_precision"],
            metrics["roc_auc"],
        )
        results.append(
            {
                "label": scenario["label"],
                "model_name": model_name,
                "balanced": balanced,
                "params": final_params,
                "metrics": metrics,
                "run_id": run_id,
            }
        )

    # Identifica o vencedor pela métrica primária e promove no Model Registry
    best_entry = max(results, key=lambda item: item["metrics"][primary_metric])
    best_version = register_model(
        best_entry["run_id"],
        MODEL_ARTIFACT_PATH,
        settings.mlflow_registered_model_name,
    )
    promote_model(settings.mlflow_registered_model_name, best_version, alias)
    champion_uri = build_model_uri(settings.mlflow_registered_model_name, alias)

    summary_payload = {
        "primary_metric": primary_metric,
        "decision_threshold": threshold,
        "champion": {
            "label": best_entry["label"],
            "model_name": best_entry["model_name"],
            "balanced": best_entry["balanced"],
            "version": best_version,
            "alias": alias,
            "model_uri": champion_uri,
            "metrics": best_entry["metrics"],
            "run_id": best_entry["run_id"],
        },
        "experiments": results,
    }
    report_path = settings.reports_dir / "benchmark_comparison.json"
    write_json(report_path, summary_payload)

    markdown_table = _format_markdown_table(results, primary_metric)
    print("\n" + "=" * 90)
    print("RESULTADO DO BENCHMARK COMPARATIVO (COPIE E COLE NO RELATÓRIO / README)")
    print("=" * 90 + "\n")
    print(markdown_table)
    print(
        f"\nModelo promovido para '{champion_uri}' (versão {best_version}): "
        f"{best_entry['label']} ({primary_metric}={best_entry['metrics'][primary_metric]:.4f})\n"
    )


if __name__ == "__main__":
    run_benchmark()
