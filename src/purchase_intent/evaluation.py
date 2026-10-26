"""Avaliação do modelo treinado e registro das métricas no MLflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from purchase_intent.config import configure_logging, load_params, load_settings
from purchase_intent.data_schema import read_dataset, write_json
from purchase_intent.data_split import split_features_target

LOGGER = logging.getLogger(__name__)


def compute_metrics(
    target: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas de ranking e de decisão para classificação binária.

    Args:
        target: Classes verdadeiras do conjunto de teste.
        predictions: Classes previstas após aplicar o limiar.
        probabilities: Probabilidades previstas para a classe positiva.

    Returns:
        Métricas com valores Python float, prontas para JSON e MLflow.
    """
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def _positive_probabilities(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Retorna a coluna de probabilidade correspondente ao rótulo booleano True.

    Args:
        pipeline: Pipeline Scikit-Learn treinado.
        features: Features sem a coluna alvo.

    Returns:
        Probabilidade da classe positiva para cada linha.
    """
    classes = list(pipeline.classes_)
    if True not in classes:
        raise ValueError("O classificador treinado não possui a classe positiva True.")
    return pipeline.predict_proba(features)[:, classes.index(True)]


def evaluate_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Calcula métricas do pipeline em dados que não foram usados no treino.

    Args:
        pipeline: Pipeline Scikit-Learn treinado.
        features: Features de teste.
        target: Alvo de teste.
        threshold: Limiar de decisão para classificar uma compra.

    Returns:
        Métricas de ranking e classificação.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("O limiar de decisão deve estar entre 0 e 1.")
    probabilities = _positive_probabilities(pipeline, features)
    predictions = probabilities >= threshold
    return compute_metrics(target, predictions, probabilities)


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Grava métricas em JSON determinístico para o DVC.

    Args:
        metrics: Métricas numéricas da avaliação.
        output_path: Caminho do relatório JSON.
    """
    write_json(output_path, metrics)


def _parse_arguments() -> argparse.Namespace:
    """Lê caminhos declarados pelo stage evaluate do DVC.

    Returns:
        Namespace com os caminhos de teste, modelo, run, métricas e parâmetros.
    """
    parser = argparse.ArgumentParser(description="Avalia o classificador treinado.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-id", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    return parser.parse_args()


def _log_evaluation(
    run_id: str,
    metrics: dict[str, float],
    threshold: float,
) -> None:
    """Anexa limiar e métricas de teste à run de treinamento no MLflow."""
    settings = load_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    client.set_tag(run_id, "evaluation.decision_threshold", str(threshold))
    for name, value in metrics.items():
        client.log_metric(run_id, name, value)


def main() -> None:
    """Avalia o artefato treinado, salva métricas e atualiza a run do MLflow."""
    args = _parse_arguments()
    settings = load_settings()
    configure_logging(settings.log_level)
    params = load_params(args.params)
    test_data = read_dataset(args.test)
    features, target = split_features_target(test_data, params["dataset"]["target_column"])
    pipeline: Pipeline = mlflow.sklearn.load_model(str(args.model))
    metrics = evaluate_pipeline(
        pipeline,
        features,
        target,
        float(params["evaluation"]["decision_threshold"]),
    )
    run_id = args.run_id.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"Identificador da run MLflow vazio: {args.run_id}")
    save_metrics(metrics, args.metrics)
    _log_evaluation(run_id, metrics, float(params["evaluation"]["decision_threshold"]))
    LOGGER.info("Métricas de teste registradas: %s", metrics)


if __name__ == "__main__":
    main()
