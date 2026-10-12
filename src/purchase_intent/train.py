"""Treinamento do pipeline e rastreamento do experimento no MLflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from purchase_intent.config import configure_logging, load_params, load_settings
from purchase_intent.data_schema import read_dataset
from purchase_intent.data_split import split_features_target
from purchase_intent.preprocessing import build_preprocessor, infer_feature_types

LOGGER = logging.getLogger(__name__)
MODEL_ARTIFACT_PATH = "model"


def build_estimator(
    model_name: str,
    model_params: dict[str, Any],
    seed: int,
) -> ClassifierMixin:
    """Instancia o classificador solicitado nos parâmetros.

    Args:
        model_name: Identificador do modelo configurado.
        model_params: Hiperparâmetros do estimador.
        seed: Semente para aleatoriedade do modelo.

    Returns:
        Classificador Scikit-Learn ainda não treinado.
    """
    if model_name != "random_forest":
        raise ValueError(f"Modelo não suportado: {model_name}")
    return RandomForestClassifier(random_state=seed, **model_params)


def build_pipeline(preprocessor: ColumnTransformer, estimator: ClassifierMixin) -> Pipeline:
    """Une pré-processamento e estimador em um pipeline serializável.

    Args:
        preprocessor: Transformador Scikit-Learn para as features.
        estimator: Classificador Scikit-Learn.

    Returns:
        Pipeline ainda não treinado.
    """
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def train_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> Pipeline:
    """Ajusta pré-processamento e classificador somente nos dados de treino.

    Args:
        pipeline: Pipeline Scikit-Learn não treinado.
        features: Features de treino.
        target: Alvo de treino.

    Returns:
        Pipeline treinado.
    """
    return pipeline.fit(features, target)


def _parse_arguments() -> argparse.Namespace:
    """Lê os caminhos explícitos que o DVC declara para o stage train."""
    parser = argparse.ArgumentParser(description="Treina o classificador de compra.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--run-id-output", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    return parser.parse_args()


def _flatten_model_params(params: dict[str, Any]) -> dict[str, str]:
    """Converte hiperparâmetros heterogêneos em strings aceitas pelo MLflow.

    Args:
        params: Parâmetros do classificador, que podem conter valores nulos.

    Returns:
        Mapa achatado com chaves prefixadas por `model.`.
    """
    return {f"model.{key}": str(value) for key, value in params.items()}


def _log_model_run(
    pipeline: Pipeline,
    features: pd.DataFrame,
    model_name: str,
    model_params: dict[str, Any],
    seed: int,
    model_path: Path,
    run_id_path: Path,
) -> str:
    """Registra parâmetros e modelo, e grava o run ID como artefato do DVC.

    Args:
        pipeline: Pipeline já treinado.
        features: Features de treino usadas como exemplo de assinatura.
        model_name: Nome do estimador configurado.
        model_params: Hiperparâmetros efetivamente usados.
        seed: Semente aplicada ao estimador.
        model_path: Diretório versionado pelo DVC para o modelo.
        run_id_path: Arquivo DVC que receberá o ID da run MLflow.

    Returns:
        Identificador da run criada.
    """
    settings = load_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    run_id_path.parent.mkdir(parents=True, exist_ok=True)
    example = features.head(5)
    signature = infer_signature(example, pipeline.predict(example))
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model.name": model_name,
                "seed": seed,
                **_flatten_model_params(model_params),
            }
        )
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=MODEL_ARTIFACT_PATH,
            signature=signature,
            input_example=example,
        )
        mlflow.sklearn.save_model(
            sk_model=pipeline,
            path=str(model_path),
            signature=signature,
            input_example=example,
        )
        run_id_path.write_text(run.info.run_id + "\n", encoding="utf-8")
    return run.info.run_id


def main() -> None:
    """Treina o modelo e salva pipeline e identificador da run."""
    args = _parse_arguments()
    configure_logging(load_settings().log_level)
    params = load_params(args.params)
    model = params["model"]
    seed = int(params["seed"])
    dataframe = read_dataset(args.train)
    features, target = split_features_target(dataframe, params["dataset"]["target_column"])
    numeric, categorical = infer_feature_types(features)
    preprocessor = build_preprocessor(numeric, categorical)
    pipeline = train_pipeline(
        build_pipeline(preprocessor, build_estimator(model["name"], model["params"], seed)),
        features,
        target,
    )
    run_id = _log_model_run(
        pipeline,
        features,
        model["name"],
        model["params"],
        seed,
        args.model_output,
        args.run_id_output,
    )
    LOGGER.info("Modelo treinado e registrado no MLflow; run_id=%s", run_id)


if __name__ == "__main__":
    main()
