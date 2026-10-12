"""Separação entre treino e teste — stage `prepare` do pipeline.

Responsabilidade única: dividir o dataset bruto de forma estratificada e
reprodutível, gravando os conjuntos em `data/processed/`. Nenhum encoding é
aplicado aqui — isso pertence ao `Pipeline` do modelo (ver `preprocessing`).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from purchase_intent.config import DataParameters, configure_logging, load_data_parameters
from purchase_intent.data_schema import file_sha256, read_dataset, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSplits:
    """Conjuntos de treino e teste produzidos pela divisão estratificada.

    Existe para que os quatro conjuntos circulem com nome próprio. Como tupla
    posicional, seria possível trocar treino por teste sem que nenhuma
    ferramenta percebesse — o modelo treinaria no conjunto errado e as métricas
    pareceriam válidas.
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features do vetor alvo.

    Args:
        dataframe: DataFrame completo, com a coluna alvo.
        target_column: Nome da coluna alvo.

    Returns:
        Tupla `(features, alvo)`.
    """
    if target_column not in dataframe:
        raise ValueError(f"Coluna alvo ausente: {target_column}")
    return dataframe.drop(columns=[target_column]), dataframe[target_column].copy()


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> DataSplits:
    """Divide os dados preservando a proporção de classes do alvo.

    Args:
        features: Matriz de features.
        target: Vetor alvo.
        test_size: Fração destinada ao conjunto de teste.
        seed: Semente que torna a divisão reprodutível.
        stratify: Quando verdadeiro, preserva a distribuição de classes.

    Returns:
        Conjuntos de treino e teste nomeados.
    """
    if not features.index.equals(target.index):
        raise ValueError("Features e alvo devem ter índices alinhados.")
    if not features.index.is_unique:
        raise ValueError("Posições de linhas devem ser únicas.")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target if stratify else None,
    )
    return DataSplits(x_train, x_test, y_train, y_test)


def save_splits(splits: DataSplits, output_dir: Path) -> None:
    """Grava os conjuntos de treino e teste em `data/processed/`.

    Args:
        splits: Conjuntos produzidos por `stratified_split`.
        output_dir: Diretório de destino.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, features, target in (
        ("train.csv", splits.x_train, splits.y_train),
        ("test.csv", splits.x_test, splits.y_test),
    ):
        frame = pd.concat([features, target], axis=1)
        frame.to_csv(output_dir / name, index=False, encoding="utf-8", lineterminator="\n")


def prepare_dataset(source: Path, output_dir: Path, params: DataParameters) -> None:
    """Produz CSVs e manifesto determinísticos preservando cada posição do bruto."""
    outputs = (output_dir / name for name in ("train.csv", "test.csv", "split_manifest.json"))
    if source.resolve() in {path.resolve() for path in outputs}:
        raise ValueError("A preparação não pode sobrescrever o CSV bruto.")
    raw = read_dataset(source)
    features, target = split_features_target(raw, params.target_column)
    splits = stratified_split(features, target, params.test_size, params.seed, params.stratify)
    save_splits(splits, output_dir)
    manifest = {
        "version": 1,
        "source": {"sha256": file_sha256(source), "rows": len(raw), "columns": list(raw)},
        "parameters": asdict(params),
        "train": _split_metadata(output_dir / "train.csv", splits.x_train),
        "test": _split_metadata(output_dir / "test.csv", splits.x_test),
    }
    write_json(output_dir / "split_manifest.json", manifest)
    LOGGER.info(
        "prepare: %d linhas treino; %d teste; seed=%d",
        len(splits.x_train),
        len(splits.x_test),
        params.seed,
    )


def _split_metadata(path: Path, features: pd.DataFrame) -> dict[str, object]:
    """Registra identidade do CSV e posições originais, incluindo linhas iguais."""
    return {
        "sha256": file_sha256(path),
        "rows": len(features),
        "positions": [int(position) for position in features.index],
    }


def _parse_arguments() -> argparse.Namespace:
    """Recebe caminhos explícitos que coincidem com as dependências do DVC."""
    parser = argparse.ArgumentParser(description="Prepara dados sem treinar um modelo.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do stage `prepare` (executado via `python -m`)."""
    args = _parse_arguments()
    configure_logging("INFO")
    try:
        params = load_data_parameters(args.params)
        prepare_dataset(args.input, args.output_dir, params)
    except (ValueError, OSError) as error:
        LOGGER.error("Falha na preparação: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
