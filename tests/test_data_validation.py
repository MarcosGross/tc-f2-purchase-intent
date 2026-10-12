"""Falhas reais de integridade e preservação de registros originalmente iguais."""

from pathlib import Path

import pandas as pd
import pytest

from purchase_intent.config import DataParameters
from purchase_intent.data_schema import file_sha256, read_json, validate_dataframe, write_json
from purchase_intent.data_split import prepare_dataset
from purchase_intent.data_validation import ValidationPaths, validate_prepared_data


def _prepare(raw_path: Path, tmp_path: Path) -> tuple[ValidationPaths, DataParameters]:
    params = DataParameters(42, 0.2, True, "Revenue")
    output = tmp_path / "processed"
    prepare_dataset(raw_path, output, params)
    paths = ValidationPaths(
        raw_path,
        output / "train.csv",
        output / "test.csv",
        output / "split_manifest.json",
        tmp_path / "report.json",
    )
    return paths, params


def test_valid_report_is_deterministic(raw_path: Path, tmp_path: Path):
    paths, params = _prepare(raw_path, tmp_path)
    validate_prepared_data(paths, params)
    before = paths.report.read_bytes()
    validate_prepared_data(paths, params)
    assert paths.report.read_bytes() == before
    report = read_json(paths.report)
    assert report["status"] == "ok"
    assert report["train"]["rows"] == 32
    assert report["test"]["rows"] == 8


def test_changed_csv_invalidates_report(raw_path: Path, tmp_path: Path):
    paths, params = _prepare(raw_path, tmp_path)
    validate_prepared_data(paths, params)
    paths.train.write_text(paths.train.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Hash divergente"):
        validate_prepared_data(paths, params)
    assert not paths.report.exists()


def test_hash_alone_cannot_hide_changed_target(raw_path: Path, tmp_path: Path):
    paths, params = _prepare(raw_path, tmp_path)
    train = pd.read_csv(paths.train)
    train.loc[0, "Revenue"] = not train.loc[0, "Revenue"]
    train.to_csv(paths.train, index=False)
    manifest = read_json(paths.manifest)
    manifest["train"]["sha256"] = file_sha256(paths.train)
    write_json(paths.manifest, manifest)
    with pytest.raises(ValueError, match="alvo divergentes"):
        validate_prepared_data(paths, params)


@pytest.mark.parametrize("issue", ["overlap", "missing", "order", "params", "bad_type"])
def test_manifest_corruption_is_rejected(raw_path: Path, tmp_path: Path, issue: str):
    paths, params = _prepare(raw_path, tmp_path)
    manifest = read_json(paths.manifest)
    if issue == "overlap":
        manifest["test"]["positions"][0] = manifest["train"]["positions"][0]
    elif issue == "missing":
        manifest["test"]["positions"].pop()
    elif issue == "order":
        manifest["train"]["positions"].reverse()
    elif issue == "params":
        manifest["parameters"]["seed"] = 17
    else:
        manifest["test"]["positions"][0] = False
    write_json(paths.manifest, manifest)
    with pytest.raises(ValueError):
        validate_prepared_data(paths, params)


def test_equal_rows_remain_distinct_positions(raw_frame: pd.DataFrame, tmp_path: Path):
    raw_frame.iloc[2] = raw_frame.iloc[0]
    path = tmp_path / "duplicates.csv"
    raw_frame.to_csv(path, index=False)
    paths, params = _prepare(path, tmp_path)
    validate_prepared_data(paths, params)
    assert read_json(paths.report)["raw"]["rows"] == 40


@pytest.mark.parametrize("issue", ["empty", "schema", "null", "infinity", "target"])
def test_bad_raw_schema_is_rejected(raw_frame: pd.DataFrame, issue: str):
    if issue == "empty":
        raw_frame = raw_frame.iloc[:0]
    elif issue == "schema":
        raw_frame["Unnamed: 0"] = 0
    elif issue == "null":
        raw_frame.loc[0, "Month"] = None
    elif issue == "infinity":
        raw_frame.loc[0, "PageValues"] = float("inf")
    else:
        raw_frame["Revenue"] = 2
    with pytest.raises(ValueError):
        validate_dataframe(raw_frame)
