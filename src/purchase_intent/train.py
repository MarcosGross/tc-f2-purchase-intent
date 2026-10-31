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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from purchase_intent.config import configure_logging, load_params, load_settings
from purchase_intent.data_schema import read_dataset
from purchase_intent.data_split import split_features_target
from purchase_intent.preprocessing import build_preprocessor, infer_feature_types

LOGGER = logging.getLogger(__name__)
MODEL_ARTIFACT_PATH = "model"

SUPPORTED_MODELS: dict[str, type[ClassifierMixin]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def resolve_model_params(
    model_name: str,
    model_params: dict[str, Any] | None,
    balanced: bool = False,
) -> dict[str, Any]:
    """Combina hiperparâmetros do modelo com a estratégia de balanceamento.

    Args:
        model_name: Nome do algoritmo configurado.
        model_params: Hiperparâmetros declarados no `params.yaml`.
        balanced: Quando verdadeiro, aplica `class_weight='balanced'` caso
            `class_weight` não tenha sido definido explicitamente.

    Returns:
        Dicionário final de hiperparâmetros pronto para instanciar o estimador.
    """
    resolved = dict(model_params or {})
    if model_name == "logistic_regression" and "max_iter" not in resolved:
        resolved["max_iter"] = 1000
    if "class_weight" not in resolved:
        resolved["class_weight"] = "balanced" if balanced else None
    return resolved


def build_estimator(
    model_name: str,
    model_params: dict[str, Any] | None,
    seed: int,
    *,
    balanced: bool = False,
) -> ClassifierMixin:
    """Instancia o classificador solicitado nos parâmetros.

    Args:
        model_name: Identificador do modelo (`logistic_regression`,
            `random_forest` ou `hist_gradient_boosting`).
        model_params: Hiperparâmetros do estimador.
        seed: Semente para aleatoriedade do modelo.
        balanced: Ativa ponderação de classes (`class_weight='balanced'`).

    Returns:
        Classificador Scikit-Learn ainda não treinado.
    """
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Modelo não suportado: {model_name}. Opções válidas: {supported}")
    final_params = resolve_model_params(model_name, model_params, balanced=balanced)
    estimator_cls = SUPPORTED_MODELS[model_name]
    return estimator_cls(random_state=seed, **final_params)


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
    balanced: bool,
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
        balanced: Indica se o balanceamento de classes foi ativado.
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
    balance_label = "balanced" if balanced else "unbalanced"
    run_name = f"{model_name}_{balance_label}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "experiment.model_family": model_name,
                "experiment.balancing": balance_label,
                "experiment.variant": run_name,
            }
        )
        mlflow.log_params(
            {
                "model.name": model_name,
                "model.balanced": str(balanced),
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
    model_cfg = params["model"]
    model_name = model_cfg["name"]
    raw_model_params = model_cfg.get("params") or {}
    balanced = bool(
        model_cfg.get("balanced", False) or raw_model_params.get("class_weight") is not None
    )
    final_model_params = resolve_model_params(model_name, raw_model_params, balanced=balanced)
    seed = int(params["seed"])

    dataframe = read_dataset(args.train)
    features, target = split_features_target(dataframe, params["dataset"]["target_column"])
    numeric, categorical = infer_feature_types(features)
    preprocessor = build_preprocessor(numeric, categorical)
    estimator = build_estimator(model_name, final_model_params, seed, balanced=balanced)
    pipeline = train_pipeline(
        build_pipeline(preprocessor, estimator),
        features,
        target,
    )
    run_id = _log_model_run(
        pipeline,
        features,
        model_name,
        final_model_params,
        balanced,
        seed,
        args.model_output,
        args.run_id_output,
    )
    LOGGER.info(
        "Modelo treinado (%s | balanced=%s) e registrado no MLflow; run_id=%s",
        model_name,
        balanced,
        run_id,
    )


if __name__ == "__main__":
    main()
