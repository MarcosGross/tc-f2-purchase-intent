"""Registro e promoção do modelo no MLflow Model Registry."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from purchase_intent.config import configure_logging, load_params, load_settings
from purchase_intent.data_schema import read_json, write_json
from purchase_intent.train import MODEL_ARTIFACT_PATH

LOGGER = logging.getLogger(__name__)


def register_model(run_id: str, artifact_path: str, model_name: str) -> str:
    """Registra o modelo da run e evita duplicar a mesma run no Registry.

    Args:
        run_id: Identificador da run de treinamento no MLflow.
        artifact_path: Caminho do modelo dentro dos artefatos da run.
        model_name: Nome do modelo registrado.

    Returns:
        Número da versão criada ou já existente.
    """
    client = MlflowClient()
    for version in client.search_model_versions(f"name='{model_name}'"):
        if version.run_id == run_id:
            return str(version.version)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    return str(registered.version)


def promote_model(model_name: str, version: str, alias: str) -> None:
    """Aponta um alias do Registry para uma versão do modelo.

    Args:
        model_name: Nome do modelo registrado.
        version: Versão a ser promovida.
        alias: Alias de destino, como `champion`.
    """
    MlflowClient().set_registered_model_alias(model_name, alias, version)


def build_model_uri(model_name: str, alias: str) -> str:
    """Monta a URI de uma versão promovida para uso na inferência.

    Args:
        model_name: Nome do modelo registrado.
        alias: Alias que identifica a versão promovida.

    Returns:
        URI do MLflow no formato `models:/nome@alias`.
    """
    return f"models:/{model_name}@{alias}"


def _candidate_is_better(
    client: MlflowClient,
    model_name: str,
    alias: str,
    metric_name: str,
    candidate_value: float,
) -> bool:
    """Compara a métrica candidata com o valor da versão champion atual.

    Args:
        client: Cliente conectado ao Model Registry.
        model_name: Nome do modelo registrado.
        alias: Alias da versão atual, se houver.
        metric_name: Nome da métrica primária.
        candidate_value: Valor obtido pelo modelo candidato.

    Returns:
        `True` quando a versão candidata deve receber o alias.
    """
    try:
        current_version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return True
    current_run = client.get_run(current_version.run_id)
    current_value = current_run.data.metrics.get(metric_name)
    return current_value is None or candidate_value >= current_value


def _parse_arguments() -> argparse.Namespace:
    """Lê os arquivos declarados no stage register do DVC.

    Returns:
        Namespace com caminhos para run, métricas, parâmetros e relatório.
    """
    parser = argparse.ArgumentParser(description="Registra e promove o melhor modelo.")
    parser.add_argument("--run-id", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Registra a versão avaliada e atualiza o alias se a métrica melhorar."""
    args = _parse_arguments()
    settings = load_settings()
    configure_logging(settings.log_level)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    params: dict[str, Any] = load_params(args.params)
    metrics = read_json(args.metrics)
    metric_name = params["evaluation"]["primary_metric"]
    if metric_name not in metrics:
        raise ValueError(f"Métrica principal ausente: {metric_name}")
    run_id = args.run_id.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"Identificador da run MLflow vazio: {args.run_id}")

    model_name = settings.mlflow_registered_model_name
    alias = params["registry"]["alias"]
    client = MlflowClient()
    version = register_model(run_id, MODEL_ARTIFACT_PATH, model_name)
    metric_value = float(metrics[metric_name])
    promoted = _candidate_is_better(client, model_name, alias, metric_name, metric_value)
    if promoted:
        promote_model(model_name, version, alias)
    result = {
        "model_name": model_name,
        "version": version,
        "alias": alias if promoted else None,
        "promoted": promoted,
        "primary_metric": metric_name,
        "metric_value": metric_value,
        "model_uri": build_model_uri(model_name, alias) if promoted else None,
    }
    write_json(args.report, result)
    LOGGER.info("Registro do modelo concluído: %s", result)


if __name__ == "__main__":
    main()
