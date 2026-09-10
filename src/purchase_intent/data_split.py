"""Separação entre treino e teste — stage `prepare` do pipeline.

Responsabilidade única: dividir o dataset bruto de forma estratificada e
reprodutível, gravando os conjuntos em `data/processed/`. Nenhum encoding é
aplicado aqui — isso pertence ao `Pipeline` do modelo (ver `preprocessing`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features do vetor alvo.

    Args:
        dataframe: DataFrame completo, com a coluna alvo.
        target_column: Nome da coluna alvo.

    Returns:
        Tupla `(X, y)`.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide os dados preservando a proporção de classes do alvo.

    Args:
        features: Matriz de features.
        target: Vetor alvo.
        test_size: Fração destinada ao conjunto de teste.
        seed: Semente que torna a divisão reprodutível.

    Returns:
        Tupla `(X_train, X_test, y_train, y_test)`.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def save_splits(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    output_dir: Path,
) -> None:
    """Grava os conjuntos de treino e teste em `data/processed/`.

    Args:
        splits: Tupla `(X_train, X_test, y_train, y_test)`.
        output_dir: Diretório de destino.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def main() -> None:
    """Ponto de entrada do stage `prepare` (executado via `python -m`)."""
    raise NotImplementedError("Implementar na Etapa 2.")


if __name__ == "__main__":
    main()
