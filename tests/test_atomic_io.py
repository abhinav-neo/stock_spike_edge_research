from __future__ import annotations

import pandas as pd
import pytest

from src.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet


def test_atomic_writers_replace_targets(tmp_path) -> None:
    csv_path = tmp_path / "nested" / "frame.csv"
    parquet_path = tmp_path / "nested" / "frame.parquet"
    json_path = tmp_path / "nested" / "payload.json"

    atomic_write_csv(pd.DataFrame({"value": [1]}), csv_path)
    atomic_write_parquet(pd.DataFrame({"value": [2]}), parquet_path)
    atomic_write_json({"value": 3}, json_path)

    assert pd.read_csv(csv_path).iloc[0]["value"] == 1
    assert pd.read_parquet(parquet_path).iloc[0]["value"] == 2
    assert json_path.read_text(encoding="utf-8") == '{\n  "value": 3\n}'


def test_failed_atomic_csv_write_preserves_existing_target(monkeypatch, tmp_path) -> None:
    target = tmp_path / "ledger.csv"
    target.write_text("original\n", encoding="utf-8")

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_csv(pd.DataFrame({"value": [1]}), target)

    assert target.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".*.tmp")) == []
