"""Separação entre treino e teste — stage `prepare` do pipeline.

Responsabilidade única: dividir o dataset bruto de forma estratificada e
reprodutível, gravando os conjuntos em `data/processed/`. Nenhum encoding é
aplicado aqui — isso pertence ao `Pipeline` do modelo (ver `preprocessing`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataSplits:
    """Conjuntos de treino e teste produzidos pela divisão estratificada.

    Existe para que os quatro conjuntos circulem com nome próprio. Como tupla
    posicional, seria possível trocar treino por teste sem que nenhuma
    ferramenta percebesse — o modelo treinaria no conjunto errado e as métricas
    pareceriam válidas.
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features do vetor alvo.

    Args:
        dataframe: DataFrame completo, com a coluna alvo.
        target_column: Nome da coluna alvo.

    Returns:
        Tupla `(features, alvo)`.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    seed: int,
) -> DataSplits:
    """Divide os dados preservando a proporção de classes do alvo.

    Args:
        features: Matriz de features.
        target: Vetor alvo.
        test_size: Fração destinada ao conjunto de teste.
        seed: Semente que torna a divisão reprodutível.

    Returns:
        Conjuntos de treino e teste nomeados.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def save_splits(splits: DataSplits, output_dir: Path) -> None:
    """Grava os conjuntos de treino e teste em `data/processed/`.

    Args:
        splits: Conjuntos produzidos por `stratified_split`.
        output_dir: Diretório de destino.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def main() -> None:
    """Ponto de entrada do stage `prepare` (executado via `python -m`)."""
    raise NotImplementedError("Implementar na Etapa 2.")


if __name__ == "__main__":
    main()
