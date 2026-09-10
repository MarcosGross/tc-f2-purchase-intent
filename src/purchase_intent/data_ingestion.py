"""Obtenção do dataset bruto — primeiro stage do pipeline.

Responsabilidade única: trazer o CSV do UCI para `data/raw/` e lê-lo como
DataFrame. Nenhuma transformação acontece aqui; limpeza e encoding pertencem a
`preprocessing`.

Fonte: Online Shoppers Purchasing Intention Dataset (UCI, id 468).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def download_dataset(url: str, destination: Path) -> Path:
    """Baixa o dataset da origem e grava o CSV em disco.

    Args:
        url: URL do arquivo (ZIP) publicado pelo UCI.
        destination: Caminho do CSV de destino em `data/raw/`.

    Returns:
        Caminho do arquivo CSV gravado.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Lê o CSV bruto sem aplicar qualquer transformação.

    Args:
        path: Caminho do CSV em `data/raw/`.

    Returns:
        DataFrame com os dados exatamente como publicados pela fonte.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def describe_dataset(dataframe: pd.DataFrame) -> dict[str, object]:
    """Resume o dataset (dimensões, tipos e balanceamento do alvo).

    Usado para registrar no MLflow e para documentar o schema real, que ainda
    será verificado após o primeiro download.

    Args:
        dataframe: DataFrame bruto.

    Returns:
        Dicionário com as estatísticas descritivas do dataset.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def main() -> None:
    """Ponto de entrada do stage `ingest` (executado via `python -m`)."""
    raise NotImplementedError("Implementar na Etapa 2.")


if __name__ == "__main__":
    main()
