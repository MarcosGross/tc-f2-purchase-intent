"""Treino do modelo e rastreamento no MLflow — stage `train` do pipeline.

Responsabilidade única: montar o `Pipeline` (pré-processamento + estimador),
ajustá-lo e registrar parâmetros, métricas e o artefato do modelo em uma run
do MLflow. A avaliação detalhada fica em `evaluation`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline


def build_estimator(model_name: str, model_params: dict[str, Any], seed: int) -> ClassifierMixin:
    """Instancia o estimador do Scikit-Learn indicado nos parâmetros.

    Args:
        model_name: Identificador do modelo (ex.: "random_forest").
        model_params: Hiperparâmetros vindos de `configs/params.yaml`.
        seed: Semente aplicada ao estimador.

    Returns:
        Estimador ainda não treinado.
    """
    raise NotImplementedError("Implementar na Etapa 5.")


def build_pipeline(preprocessor: Any, estimator: ClassifierMixin) -> Pipeline:
    """Une pré-processamento e estimador em um único `Pipeline`.

    Manter os dois juntos garante que a inferência aplique as mesmas
    transformações do treino.

    Args:
        preprocessor: `ColumnTransformer` de `preprocessing.build_preprocessor`.
        estimator: Classificador do Scikit-Learn.

    Returns:
        `Pipeline` pronto para ser ajustado.
    """
    raise NotImplementedError("Implementar na Etapa 5.")


def train_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> Pipeline:
    """Ajusta o pipeline ao conjunto de treino.

    Args:
        pipeline: Pipeline não treinado.
        features: Features de treino.
        target: Alvo de treino.

    Returns:
        Pipeline treinado.
    """
    raise NotImplementedError("Implementar na Etapa 5.")


def main() -> None:
    """Ponto de entrada do stage `train` (executado via `python -m`)."""
    raise NotImplementedError("Implementar na Etapa 5.")


if __name__ == "__main__":
    main()
