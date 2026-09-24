"""Obtenção do dataset bruto — primeiro stage do pipeline.

Responsabilidade única: trazer o CSV do UCI para `data/raw/` e lê-lo como
DataFrame. Nenhuma transformação acontece aqui; limpeza e encoding pertencem a
`preprocessing`.

Fonte: Online Shoppers Purchasing Intention Dataset (UCI, id 468).
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

from purchase_intent.config import configure_logging, load_settings

LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "Revenue"
CSV_FILENAME = "online_shoppers_intention.csv"


def download_dataset(url: str, destination: Path) -> Path:
    """Baixa o dataset da origem e grava o CSV em disco.

    Args:
        url: URL do arquivo (ZIP) publicado pelo UCI.
        destination: Caminho do CSV de destino em `data/raw/`.

    Returns:
        Caminho do arquivo CSV gravado.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "purchase-intent/0.1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
    except OSError as error:
        raise RuntimeError(f"Falha ao baixar o dataset de {url}: {error}") from error

    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            csv_files = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if not csv_files:
                raise ValueError("O arquivo ZIP baixado não contém nenhum CSV.")
            preferred = next(
                (name for name in csv_files if Path(name).name.lower() == CSV_FILENAME),
                csv_files[0],
            )
            destination.write_bytes(archive.read(preferred))
    except BadZipFile as error:
        raise ValueError(f"A resposta de {url} não é um arquivo ZIP válido.") from error

    LOGGER.info("Dataset salvo em %s", destination)
    return destination


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Lê o CSV bruto sem aplicar qualquer transformação.

    Args:
        path: Caminho do CSV em `data/raw/`.

    Returns:
        DataFrame com os dados exatamente como publicados pela fonte.
    """
    if not path.is_file():
        raise FileNotFoundError(f"CSV bruto não encontrado: {path}")
    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"O CSV não contém registros: {path}")
    return dataframe


def describe_dataset(dataframe: pd.DataFrame) -> dict[str, object]:
    """Resume o dataset (dimensões, tipos e balanceamento do alvo).

    Usado para registrar no MLflow e para documentar o schema real, que ainda
    será verificado após o primeiro download.

    Args:
        dataframe: DataFrame bruto.

    Returns:
        Dicionário com as estatísticas descritivas do dataset.
    """
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Coluna alvo obrigatória ausente: {TARGET_COLUMN}")
    target_counts = dataframe[TARGET_COLUMN].value_counts(dropna=False)
    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_types": {name: str(dtype) for name, dtype in dataframe.dtypes.items()},
        "target_distribution": {
            str(label): int(count) for label, count in target_counts.items()
        },
    }


def main() -> None:
    """Ponto de entrada do stage `ingest` (executado via `python -m`)."""
    settings = load_settings()
    configure_logging(settings.log_level)
    path = download_dataset(settings.dataset_url, settings.raw_data_path)
    dataframe = load_raw_dataset(path)
    summary = describe_dataset(dataframe)
    LOGGER.info("Dataset carregado: %s", summary)


if __name__ == "__main__":
    main()
