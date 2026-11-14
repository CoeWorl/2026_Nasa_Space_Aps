"""CLI tests. The selftest is exercised against a temporary directory,
so these pass outside Docker too."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from groundtruth.cli import app

runner = CliRunner()


def _make_data_tree(root: Path) -> None:
    for sub in ("raw/nisar", "raw/nyc", "interim", "processed"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def test_info_reports_a_present_data_root(tmp_path, monkeypatch) -> None:
    _make_data_tree(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CONFIG_ROOT", str(tmp_path))

    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0, result.output
    assert str(tmp_path) in result.output
    assert "MISSING" not in result.output


def test_info_fails_when_the_mount_is_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "not-mounted"))
    monkeypatch.setenv("CONFIG_ROOT", str(tmp_path / "not-mounted"))

    result = runner.invoke(app, ["info"])

    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_check_data_lists_a_file(tmp_path, monkeypatch) -> None:
    _make_data_tree(tmp_path)
    (tmp_path / "raw" / "nisar" / "granule.txt").write_text("x" * 100)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = runner.invoke(app, ["check-data"])

    assert result.exit_code == 0, result.output
    assert "granule.txt" in result.output
    assert "1 file(s)" in result.output


def test_selftest_writes_and_reads_back(tmp_path, monkeypatch) -> None:
    _make_data_tree(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = runner.invoke(app, ["selftest"])

    assert result.exit_code == 0, result.output
    assert "SELFTEST PASSED" in result.output
    assert (tmp_path / "interim" / "selftest.h5").stat().st_size > 0
    assert (tmp_path / "interim" / "selftest.tif").stat().st_size > 0
