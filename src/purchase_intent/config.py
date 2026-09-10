"""Configuração centralizada do projeto.

Este é o ÚNICO módulo que lê variáveis de ambiente e o arquivo de parâmetros.
Todos os demais recebem a configuração já pronta, o que mantém os módulos de
pipeline puros e testáveis (sem dependência de ambiente).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    raise NotImplementedError("Implementar na Etapa 2.")


def load_params(params_path: Path) -> dict[str, Any]:
    """Carrega os parâmetros de modelagem do `configs/params.yaml`.

    Args:
        params_path: Caminho do arquivo YAML de parâmetros.

    Returns:
        Dicionário com os parâmetros do experimento.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def set_global_seed(seed: int) -> None:
    """Fixa a semente global (`random` e `numpy`) para garantir reprodutibilidade.

    Deve ser chamada no início de cada stage, antes de qualquer operação aleatória.

    Args:
        seed: Semente a ser aplicada.
    """
    raise NotImplementedError("Implementar na Etapa 2.")


def configure_logging(level: str) -> None:
    """Configura o logging padrão da aplicação.

    Args:
        level: Nível de log (ex.: "INFO", "DEBUG").
    """
    raise NotImplementedError("Implementar na Etapa 2.")
