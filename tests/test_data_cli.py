"""Comportamento dos pontos de entrada e proteção dos arquivos de origem."""

from pathlib import Path

import pytest

from purchase_intent import data_split, data_validation
from purchase_intent.config import DataParameters
from purchase_intent.data_schema import read_json
from purchase_intent.data_validation import ValidationPaths


def test_prepare_cli_and_validation_cli(
    raw_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    params = tmp_path / "params.yaml"
    params.write_text(
        "seed: 42\ndataset:\n  target_column: Revenue\n"
        "split:\n  test_size: 0.2\n  stratify: true\n",
        encoding="utf-8",
    )
    output = tmp_path / "processed"
    monkeypatch.setattr(
        "sys.argv",
        ["prepare", "--input", str(raw_path), "--output-dir", str(output), "--params", str(params)],
    )
    data_split.main()
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate",
            "--raw",
            str(raw_path),
            "--train",
            str(output / "train.csv"),
            "--test",
            str(output / "test.csv"),
            "--manifest",
            str(output / "split_manifest.json"),
            "--report",
            str(report),
            "--params",
            str(params),
        ],
    )
    data_validation.main()
    assert read_json(report)["status"] == "ok"
    (output / "train.csv").write_text("invalid")
    with pytest.raises(SystemExit) as error:
        data_validation.main()
    assert error.value.code == 1
    assert not report.exists()


def test_report_cannot_overwrite_raw(raw_path: Path, tmp_path: Path):
    before = raw_path.read_bytes()
    paths = ValidationPaths(
        raw_path, tmp_path / "train", tmp_path / "test", tmp_path / "manifest", raw_path
    )
    with pytest.raises(ValueError, match="sobrescrever"):
        data_validation.validate_prepared_data(paths, DataParameters(42, 0.2, True, "Revenue"))
    assert raw_path.read_bytes() == before


def test_prepare_cannot_overwrite_raw(raw_path: Path, tmp_path: Path):
    source = tmp_path / "train.csv"
    source.write_bytes(raw_path.read_bytes())
    before = source.read_bytes()
    with pytest.raises(ValueError, match="sobrescrever"):
        data_split.prepare_dataset(source, tmp_path, DataParameters(42, 0.2, True, "Revenue"))
    assert source.read_bytes() == before
