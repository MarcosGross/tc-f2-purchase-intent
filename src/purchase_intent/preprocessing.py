"""Pré-processamento compartilhado entre treino e inferência.

Este módulo é importado pelo treinamento e pela inferência. O
`ColumnTransformer` viaja serializado dentro do `Pipeline`, garantindo que as
mesmas transformações sejam usadas nos dois momentos.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Códigos inteiros representam categorias nominais, não quantidades ordenadas.
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def infer_feature_types(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Confere o schema e devolve os grupos semânticos de features.

    Args:
        dataframe: DataFrame de features, sem a coluna alvo.

    Returns:
        Nomes das features numéricas e categóricas, nessa ordem.
    """
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    actual = set(dataframe.columns)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"Schema de features inválido; ausentes={sorted(missing)}, "
            f"inesperadas={sorted(unexpected)}"
        )
    return NUMERIC_FEATURES.copy(), CATEGORICAL_FEATURES.copy()


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Preserva números e codifica categorias para o estimador.

    Args:
        numeric_features: Colunas numéricas que passam sem escala.
        categorical_features: Colunas nominais que recebem one-hot encoding.

    Returns:
        `ColumnTransformer` ainda não ajustado, pronto para entrar no `Pipeline`.
    """
    if set(numeric_features) & set(categorical_features):
        raise ValueError("Uma feature não pode ser numérica e categórica ao mesmo tempo.")
    if not numeric_features or not categorical_features:
        raise ValueError("Os grupos numérico e categórico devem conter features.")
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
