"""Integração com o MLflow Model Registry — stage `register` do pipeline.

Responsabilidade única: promover o modelo de uma run para o Model Registry e
gerenciar seus estágios, isolando o restante do código da API do MLflow.
"""

from __future__ import annotations


def register_model(run_id: str, artifact_path: str, model_name: str) -> str:
    """Registra o modelo de uma run no Model Registry.

    Args:
        run_id: Identificador da run do MLflow que produziu o modelo.
        artifact_path: Caminho do artefato do modelo dentro da run.
        model_name: Nome do modelo registrado.

    Returns:
        Versão criada no Model Registry.
    """
    raise NotImplementedError("Implementar na Etapa 6.")


def promote_model(model_name: str, version: str, alias: str) -> None:
    """Aponta um alias (ex.: "champion") para uma versão do modelo.

    Args:
        model_name: Nome do modelo registrado.
        version: Versão a ser promovida.
        alias: Alias de destino.
    """
    raise NotImplementedError("Implementar na Etapa 6.")


def build_model_uri(model_name: str, alias: str) -> str:
    """Monta a URI usada para carregar o modelo promovido.

    Args:
        model_name: Nome do modelo registrado.
        alias: Alias desejado (ex.: "champion").

    Returns:
        URI no formato aceito pelo MLflow.
    """
    raise NotImplementedError("Implementar na Etapa 6.")


def main() -> None:
    """Ponto de entrada do stage `register` (executado via `python -m`)."""
    raise NotImplementedError("Implementar na Etapa 6.")


if __name__ == "__main__":
    main()
