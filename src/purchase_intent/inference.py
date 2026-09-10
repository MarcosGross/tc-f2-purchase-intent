"""Predição de propensão de compra a partir do modelo promovido.

Responsabilidade única: carregar o `Pipeline` do Model Registry e aplicá-lo a
novos dados. Como o pré-processamento viaja dentro do pipeline serializado,
este módulo não replica nenhuma transformação.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(model_uri: str) -> Pipeline:
    """Carrega do MLflow o pipeline treinado.

    Args:
        model_uri: URI do modelo (ver `registry.build_model_uri`).

    Returns:
        Pipeline pronto para inferência.
    """
    raise NotImplementedError("Implementar na Etapa 6.")


def predict_proba(model: Pipeline, sessions: pd.DataFrame) -> np.ndarray:
    """Calcula a probabilidade de compra de cada sessão de navegação.

    Args:
        model: Pipeline treinado.
        sessions: Features das sessões, no mesmo schema do treino.

    Returns:
        Vetor de probabilidades da classe positiva.
    """
    raise NotImplementedError("Implementar na Etapa 6.")


def predict(model: Pipeline, sessions: pd.DataFrame, threshold: float) -> np.ndarray:
    """Converte as probabilidades em decisão binária de propensão.

    Args:
        model: Pipeline treinado.
        sessions: Features das sessões.
        threshold: Limiar de decisão.

    Returns:
        Vetor de classes previstas.
    """
    raise NotImplementedError("Implementar na Etapa 6.")
