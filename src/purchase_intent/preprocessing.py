"""Pré-processamento compartilhado entre treino e inferência.

Este módulo NÃO é um stage do pipeline: ele é importado por `train` e por
`inference`. O `ColumnTransformer` construído aqui é serializado dentro do
`Pipeline` do modelo, de forma que treino e produção apliquem exatamente as
mesmas transformações — o aprendizado central da Fase 1.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer

# Coluna alvo do dataset (booleana na origem: houve ou não compra na sessão).
TARGET_COLUMN = "Revenue"

# As listas de features serão preenchidas na Etapa 2, após inspecionar o CSV
# baixado. Nada é assumido sobre o schema antes da verificação.
NUMERIC_FEATURES: list[str] = []
CATEGORICAL_FEATURES: list[str] = []


def infer_feature_types(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa as colunas em numéricas e categóricas a partir dos dtypes.

    Serve para validar as listas fixas acima contra o schema real do arquivo.

    Args:
        dataframe: DataFrame de features (sem a coluna alvo).

    Returns:
        Tupla `(numericas, categoricas)` com os nomes das colunas.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Monta o `ColumnTransformer` de escalonamento e encoding.

    Args:
        numeric_features: Colunas numéricas a escalonar.
        categorical_features: Colunas categóricas a codificar.

    Returns:
        `ColumnTransformer` ainda não ajustado, pronto para entrar no `Pipeline`.
    """
    raise NotImplementedError("Implementar na Etapa 2.")
