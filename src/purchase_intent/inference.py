"""Predição de propensão de compra a partir do modelo promovido."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from purchase_intent.config import load_settings


def load_model(model_uri: str) -> Pipeline:
    """Carrega do Model Registry o pipeline associado à URI recebida.

    Args:
        model_uri: URI do modelo, como `models:/purchase-intent-classifier@champion`.

    Returns:
        Pipeline treinado pronto para inferência.
    """
    mlflow.set_tracking_uri(load_settings().mlflow_tracking_uri)
    return mlflow.sklearn.load_model(model_uri)


def predict_proba(model: Pipeline, sessions: pd.DataFrame) -> np.ndarray:
    """Calcula a probabilidade de compra de cada sessão.

    Args:
        model: Pipeline treinado.
        sessions: Features no mesmo schema do treinamento, sem `Revenue`.

    Returns:
        Vetor unidimensional com a probabilidade da classe `True`.
    """
    classes = list(model.classes_)
    if True not in classes:
        raise ValueError("O classificador não possui a classe positiva True.")
    return model.predict_proba(sessions)[:, classes.index(True)]


def predict(model: Pipeline, sessions: pd.DataFrame, threshold: float) -> np.ndarray:
    """Converte probabilidades em decisões binárias de propensão.

    Args:
        model: Pipeline treinado.
        sessions: Features das sessões.
        threshold: Limiar de probabilidade entre zero e um.

    Returns:
        Vetor booleano com a classe prevista para cada sessão.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("O limiar de decisão deve estar entre 0 e 1.")
    return predict_proba(model, sessions) >= threshold
