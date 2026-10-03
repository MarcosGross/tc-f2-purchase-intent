"""Contratos da divisão e preservação de dados sem operações de modelagem."""

import json
from pathlib import Path

import pandas as pd
import pytest

from purchase_intent.config import DataParameters, load_data_parameters
from purchase_intent.data_schema import EXPECTED_COLUMNS, read_dataset
from purchase_intent.data_split import prepare_dataset, split_features_target, stratified_split


def test_split_is_complete_disjoint_and_repeatable(raw_frame: pd.DataFrame):
    features, target = split_features_target(raw_frame, "Revenue")
    first = stratified_split(features, target, 0.2, 42)
    second = stratified_split(features, target, 0.2, 42)
    assert first.x_train.index.tolist() == second.x_train.index.tolist()
    assert set(first.x_train.index).isdisjoint(first.x_test.index)
    assert set(first.x_train.index) | set(first.x_test.index) == set(raw_frame.index)
    assert len(first.x_train) == 32 and len(first.x_test) == 8
    assert first.y_train.mean() == first.y_test.mean() == 0.5
    assert "Revenue" not in first.x_train


def test_prepare_preserves_records_and_bytes(raw_path: Path, tmp_path: Path):
    output = tmp_path / "processed"
    params = DataParameters(42, 0.2, True, "Revenue")
    prepare_dataset(raw_path, output, params)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    raw = read_dataset(raw_path)
    manifest = json.loads((output / "split_manifest.json").read_text())
    for name in ("train", "test"):
        actual = read_dataset(output / f"{name}.csv")
        expected = raw.iloc[manifest[name]["positions"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)
        assert tuple(actual.columns) == EXPECTED_COLUMNS
    prepare_dataset(raw_path, output, params)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_non_stratified_split_is_supported(raw_frame: pd.DataFrame):
    features, target = split_features_target(raw_frame, "Revenue")
    result = stratified_split(features, target, 0.25, 7, stratify=False)
    assert len(result.x_test) == 10
    assert result.x_train.index.equals(result.y_train.index)


def test_misaligned_target_is_rejected(raw_frame: pd.DataFrame):
    features, target = split_features_target(raw_frame, "Revenue")
    with pytest.raises(ValueError, match="alinhados"):
        stratified_split(features, target.iloc[::-1], 0.2, 42)


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", True),
        ("seed", -1),
        ("test_size", 0),
        ("test_size", float("nan")),
        ("stratify", "false"),
        ("target_column", "Missing"),
    ],
)
def test_invalid_parameters_are_rejected(tmp_path: Path, key: str, value: object):
    import yaml

    params = {
        "seed": 42,
        "dataset": {"target_column": "Revenue"},
        "split": {"test_size": 0.2, "stratify": True},
    }
    if key == "seed":
        params[key] = value
    elif key == "target_column":
        params["dataset"][key] = value
    else:
        params["split"][key] = value
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params), encoding="utf-8")
    with pytest.raises(ValueError):
        load_data_parameters(path)
