"""Schema e identidade dos dados; nenhuma transformação de modelagem."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from purchase_intent.schema import TARGET_COLUMN

EXPECTED_COLUMNS = (
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
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    TARGET_COLUMN,
)
CATEGORICAL_COLUMNS = ("Month", "VisitorType")
BOOLEAN_COLUMNS = ("Weekend", TARGET_COLUMN)
NUMERIC_COLUMNS = tuple(
    name for name in EXPECTED_COLUMNS if name not in (*CATEGORICAL_COLUMNS, *BOOLEAN_COLUMNS)
)


def file_sha256(path: Path) -> str:
    """Calcula identidade dos bytes sem carregar o arquivo inteiro em memória."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Grava JSON estável em UTF-8/LF, sem metadados dependentes da máquina."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(content + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> dict[str, Any]:
    """Lê um documento JSON que deve conter um mapa no nível superior."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON deve conter um mapa: {path}")
    return payload


def read_dataset(path: Path) -> pd.DataFrame:
    """Lê e valida o CSV, conservando precisão de números no round-trip."""
    dataframe = pd.read_csv(path, float_precision="round_trip")
    validate_dataframe(dataframe)
    return dataframe


def validate_dataframe(dataframe: pd.DataFrame) -> None:
    """Rejeita dados vazios, schema inesperado, nulos e tipos inválidos."""
    if dataframe.empty:
        raise ValueError("Dataset vazio.")
    if tuple(dataframe.columns) != EXPECTED_COLUMNS:
        raise ValueError("Schema ou ordem de colunas diferente do dataset UCI esperado.")
    if dataframe.isna().any().any():
        raise ValueError("Dataset contém valores ausentes.")
    _validate_column_types(dataframe)
    if set(dataframe[TARGET_COLUMN].unique()) != {False, True}:
        raise ValueError("Revenue deve conter as duas classes booleanas.")


def _validate_column_types(dataframe: pd.DataFrame) -> None:
    """Confere categorias, booleanos e números finitos sem converter valores."""
    for name in BOOLEAN_COLUMNS:
        if not pd.api.types.is_bool_dtype(dataframe[name]):
            raise ValueError(f"{name} deve ser booleano.")
    for name in CATEGORICAL_COLUMNS:
        if not dataframe[name].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"{name} deve conter categorias textuais não vazias.")
    for name in NUMERIC_COLUMNS:
        _validate_numeric_column(dataframe[name])


def _validate_numeric_column(column: pd.Series) -> None:
    """Rejeita categoria textual, booleano ou infinito em coluna numérica."""
    if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
        raise ValueError(f"{column.name} deve ser numérico.")
    if not np.isfinite(column.to_numpy(dtype=float)).all():
        raise ValueError(f"{column.name} contém valores infinitos.")


def class_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    """Resume as classes de dados, sem calcular desempenho de modelo."""
    counts = dataframe[TARGET_COLUMN].value_counts()
    return {str(label): int(counts.get(label, 0)) for label in (False, True)}
