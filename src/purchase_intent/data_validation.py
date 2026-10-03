"""Validação do contrato de dados preparados, sem avaliação de modelos."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from purchase_intent.config import DataParameters, configure_logging, load_data_parameters
from purchase_intent.data_schema import (
    class_counts,
    file_sha256,
    read_dataset,
    read_json,
    write_json,
)
from purchase_intent.data_split import split_features_target, stratified_split

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationPaths:
    """Arquivos consumidos e produzido pelo estágio de integridade."""

    raw: Path
    train: Path
    test: Path
    manifest: Path
    report: Path


def _require(condition: bool, message: str) -> None:
    """Converte uma violação de contrato em falha explícita da aplicação."""
    if not condition:
        raise ValueError(message)


def _positions(metadata: dict[str, Any], rows: int) -> list[int]:
    """Confere posições inteiras, únicas e válidas, preservando sua ordem."""
    _require(isinstance(metadata, dict), "Manifesto: metadados devem ser um mapa.")
    positions = metadata.get("positions")
    _require(isinstance(positions, list), "Manifesto: positions deve ser uma lista.")
    _require(
        all(type(value) is int and 0 <= value < rows for value in positions),
        "Manifesto: posição inválida.",
    )
    _require(len(set(positions)) == len(positions), "Manifesto: posição repetida.")
    return positions


def _validate_manifest(
    paths: ValidationPaths,
    raw: pd.DataFrame,
    manifest: dict[str, Any],
    params: DataParameters,
) -> tuple[list[int], list[int]]:
    """Verifica identidade do bruto, configuração e cobertura das posições."""
    _require(manifest.get("version") == 1, "Versão de manifesto não suportada.")
    expected_source = {"sha256": file_sha256(paths.raw), "rows": len(raw), "columns": list(raw)}
    _require(manifest["source"] == expected_source, "Identidade ou schema do bruto divergente.")
    _require(manifest["parameters"] == asdict(params), "Parâmetros do manifesto divergentes.")
    train = _positions(manifest["train"], len(raw))
    test = _positions(manifest["test"], len(raw))
    _require(set(train).isdisjoint(test), "Posição presente em treino e teste.")
    _require(set(train) | set(test) == set(range(len(raw))), "Cobertura incompleta do bruto.")
    _validate_recipe(raw, params, train, test)
    return train, test


def _validate_recipe(
    raw: pd.DataFrame,
    params: DataParameters,
    train: list[int],
    test: list[int],
) -> None:
    """Recalcula a divisão para comprovar seed, proporção e estratificação reais."""
    features, target = split_features_target(raw, params.target_column)
    expected = stratified_split(features, target, params.test_size, params.seed, params.stratify)
    _require(train == expected.x_train.index.tolist(), "Posições de treino não seguem a receita.")
    _require(test == expected.x_test.index.tolist(), "Posições de teste não seguem a receita.")


def _validate_partition(
    path: Path,
    metadata: dict[str, Any],
    raw: pd.DataFrame,
    positions: list[int],
) -> pd.DataFrame:
    """Confere hash, contagem, registros e alvo alinhado do CSV preparado."""
    _require(metadata["sha256"] == file_sha256(path), f"Hash divergente: {path.name}")
    actual = read_dataset(path)
    _require(metadata["rows"] == len(actual) == len(positions), "Contagem divergente.")
    expected = raw.iloc[positions].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as error:
        raise ValueError(f"Registros ou alvo divergentes: {path.name}") from error
    return actual


def validate_prepared_data(paths: ValidationPaths, params: DataParameters) -> None:
    """Valida todos os arquivos antes de publicar um relatório de sucesso."""
    inputs = (paths.raw, paths.train, paths.test, paths.manifest)
    _require(
        paths.report.resolve() not in {path.resolve() for path in inputs},
        "O relatório não pode sobrescrever uma entrada.",
    )
    try:
        _validate_and_report(paths, params)
    except (ValueError, KeyError, TypeError, OSError) as error:
        paths.report.unlink(missing_ok=True)
        raise ValueError(f"Falha na validação dos dados: {error}") from error


def _validate_and_report(paths: ValidationPaths, params: DataParameters) -> None:
    """Monta evidências determinísticas somente após verificar o contrato completo."""
    raw = read_dataset(paths.raw)
    manifest = read_json(paths.manifest)
    train_positions, test_positions = _validate_manifest(paths, raw, manifest, params)
    train = _validate_partition(paths.train, manifest["train"], raw, train_positions)
    test = _validate_partition(paths.test, manifest["test"], raw, test_positions)
    report = {
        "status": "ok",
        "parameters": asdict(params),
        "columns": list(raw),
        "raw": {"rows": len(raw), "sha256": file_sha256(paths.raw), "classes": class_counts(raw)},
        "train": {
            "rows": len(train),
            "sha256": file_sha256(paths.train),
            "classes": class_counts(train),
        },
        "test": {
            "rows": len(test),
            "sha256": file_sha256(paths.test),
            "classes": class_counts(test),
        },
        "checks": [
            "schema",
            "file_hashes",
            "complete_disjoint_positions",
            "split_recipe",
            "preserved_records_and_target",
        ],
    }
    write_json(paths.report, report)
    LOGGER.info("validate: integridade confirmada; treino=%d; teste=%d", len(train), len(test))


def _parse_arguments() -> argparse.Namespace:
    """Recebe todos os arquivos declarados como entradas e saída pelo DVC."""
    parser = argparse.ArgumentParser(description="Confere integridade dos dados preparados.")
    for argument in ("raw", "train", "test", "manifest", "report", "params"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Executa o estágio validate; qualquer violação retorna saída diferente de zero."""
    args = _parse_arguments()
    configure_logging("INFO")
    paths = ValidationPaths(args.raw, args.train, args.test, args.manifest, args.report)
    try:
        validate_prepared_data(paths, load_data_parameters(args.params))
    except (ValueError, OSError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
