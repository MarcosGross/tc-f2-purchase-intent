"""Avaliação do modelo treinado — stage `evaluate` do pipeline.

Responsabilidade única: calcular métricas sobre o conjunto de teste e gravar os
resultados em `reports/`, onde o DVC os lê como `metrics` e o MLflow os registra
como métricas da run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def compute_metrics(
    target: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calcula as métricas de classificação binária do experimento.

    Args:
        target: Valores reais do conjunto de teste.
        predictions: Classes previstas.
        probabilities: Probabilidades da classe positiva.

    Returns:
        Dicionário métrica -> valor (ex.: roc_auc, f1, precision, recall).
    """
    raise NotImplementedError("Implementar na Etapa 5.")


def evaluate_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Executa a inferência no conjunto de teste e devolve as métricas.

    Args:
        pipeline: Pipeline treinado.
        features: Features de teste.
        target: Alvo de teste.
        threshold: Limiar de decisão aplicado à probabilidade.

    Returns:
        Dicionário com as métricas calculadas.
    """
    raise NotImplementedError("Implementar na Etapa 5.")


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Grava as métricas em JSON para consumo do DVC.

    Args:
        metrics: Dicionário de métricas.
        output_path: Caminho do arquivo de saída (ex.: `reports/metrics.json`).
    """
    raise NotImplementedError("Implementar na Etapa 5.")


def main() -> None:
    """Ponto de entrada do stage `evaluate` (executado via `python -m`)."""
    raise NotImplementedError("Implementar na Etapa 5.")


if __name__ == "__main__":
    main()
