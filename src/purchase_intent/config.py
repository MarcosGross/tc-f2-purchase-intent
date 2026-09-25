"""Configuração centralizada do projeto.

Este é o ÚNICO módulo que lê variáveis de ambiente e o arquivo de parâmetros.
Todos os demais recebem a configuração já pronta, o que mantém os módulos de
pipeline puros e testáveis (sem dependência de ambiente).
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Configuração de infraestrutura, carregada a partir do `.env`."""

    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_registered_model_name: str
    dataset_url: str
    raw_data_path: Path
    processed_data_dir: Path
    models_dir: Path
    reports_dir: Path
    params_path: Path
    random_seed: int
    log_level: str


def load_settings(env_file: Path | None = None) -> Settings:
    """Lê o `.env` (e as variáveis de ambiente) e devolve as configurações.

    Args:
        env_file: Caminho do arquivo `.env`. Quando `None`, usa a raiz do projeto.

    Returns:
        Instância imutável de `Settings`.
    """
    load_dotenv(dotenv_path=env_file or PROJECT_ROOT / ".env", override=False)

    def configured_path(name: str, default: str) -> Path:
        path = Path(os.getenv(name, default)).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    return Settings(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "purchase-intent"),
        mlflow_registered_model_name=os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME", "purchase-intent-classifier"
        ),
        dataset_url=os.getenv(
            "DATASET_URL",
            "https://archive.ics.uci.edu/static/public/468/"
            "online+shoppers+purchasing+intention+dataset.zip",
        ),
        raw_data_path=configured_path("RAW_DATA_PATH", "data/raw/online_shoppers_intention.csv"),
        processed_data_dir=configured_path("PROCESSED_DATA_DIR", "data/processed"),
        models_dir=configured_path("MODELS_DIR", "models"),
        reports_dir=configured_path("REPORTS_DIR", "reports"),
        params_path=configured_path("PARAMS_PATH", "configs/params.yaml"),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def load_params(params_path: Path) -> dict[str, Any]:
    """Carrega os parâmetros de modelagem do `configs/params.yaml`.

    Args:
        params_path: Caminho do arquivo YAML de parâmetros.

    Returns:
        Dicionário com os parâmetros do experimento.
    """
    with params_path.open("r", encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    if not isinstance(params, dict):
        raise ValueError(f"O arquivo de parâmetros deve conter um mapa YAML: {params_path}")
    return params


def set_global_seed(seed: int) -> None:
    """Fixa a semente global (`random` e `numpy`) para garantir reprodutibilidade.

    Deve ser chamada no início de cada stage, antes de qualquer operação aleatória.

    Args:
        seed: Semente a ser aplicada.
    """
    random.seed(seed)
    np.random.seed(seed)


def configure_logging(level: str) -> None:
    """Configura o logging padrão da aplicação.

    Args:
        level: Nível de log (ex.: "INFO", "DEBUG").
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Nível de log inválido: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
