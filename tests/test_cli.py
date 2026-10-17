"""CLI smoke tests: arg parsing + backends command (no network, no clustering)."""

from __future__ import annotations

import pytest

from flashtopic.cli import main


def test_backends_command(capsys):
    rc = main(["backends"])
    out = capsys.readouterr().out
    assert "clustering backends" in out
    assert rc in (0, 1)  # 0 if a backend is installed, 1 (+stderr hint) if none


def test_cluster_requires_k_or_range():
    with pytest.raises(SystemExit):  # mutually-exclusive group is required
        main(["cluster", "corpus.jsonl"])


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "backends" in out and "cluster" in out
