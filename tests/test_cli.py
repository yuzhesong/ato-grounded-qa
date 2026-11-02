"""CLI smoke tests.

The exit code is part of the contract: 0 when answered, 2 when declined, so the
command is usable in a shell pipeline without parsing its output.
"""

import json

import pytest

from atoqa.cli import main

CORPUS = ["--corpus", "tests/fixtures/corpus"]


def test_answered_question_exits_zero(capsys):
    assert main(["ask", "What is the GST registration turnover threshold?", *CORPUS]) == 0
    out = capsys.readouterr().out
    assert "ANSWER" in out
    assert "$75,000" in out
    assert "Verification" in out, "guard results should be visible to the operator"


def test_declined_question_exits_two(capsys):
    assert main(["ask", "What is the instant asset write-off threshold?", *CORPUS]) == 2
    assert "ABSTAIN" in capsys.readouterr().out


def test_refused_question_exits_two(capsys):
    assert main(["ask", "Who won the 1998 AFL grand final?", *CORPUS]) == 2
    assert "REFUSE" in capsys.readouterr().out


def test_json_output_is_valid_and_complete(capsys):
    main(["ask", "What is the super guarantee rate for 2025-26?", "--json", *CORPUS])
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] in ("answer", "answer_with_caveat")
    assert payload["claims"] and payload["guards"] and payload["trace_id"]


def test_index_subcommand_persists_and_reloads(tmp_path, monkeypatch, capsys):
    from atoqa.config import settings

    monkeypatch.setattr(settings, "index_dir", tmp_path / "index")
    assert main(["index", *CORPUS]) == 0
    assert "Indexed" in capsys.readouterr().out
    assert (tmp_path / "index" / "chunks.json").exists()


def test_unknown_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main(["frobnicate"])

def test_postgres_ask_rejects_corpus_override(monkeypatch, capsys):
    from atoqa.config import settings

    monkeypatch.setattr(settings, "database_url", "postgresql://unused")
    with pytest.raises(SystemExit) as exc:
        main(["ask", "What is the GST threshold?", "--corpus", "missing-corpus"])

    assert exc.value.code == 2
    assert "atoqa index --corpus PATH" in capsys.readouterr().err