import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

dotenv = pytest.importorskip("dotenv")
from dotenv import load_dotenv

from scripts import evaluate_templates


def _write_fixture(tmp_path: Path) -> Path:
    data = {
        "template_variants": {"A": "{prompt}"},
        "samples": [
            {
                "prompt": "salut",
                "expected_intent": "GREETING",
                "expected_entities": {},
            }
        ],
    }
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(data))
    return fixture


def test_missing_api_key(monkeypatch, tmp_path):
    """Running without OPENAI_API_KEY produces an explicit error."""
    fixture = _write_fixture(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_openai():
        if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
            raise RuntimeError("OPENAI_API_KEY is missing")
        return MagicMock()

    monkeypatch.setattr(evaluate_templates, "OpenAI", fake_openai)
    monkeypatch.setattr(sys, "argv", ["evaluate_templates", str(fixture)])

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is missing"):
        evaluate_templates.main()


def test_dotenv_loads_key_and_calls(monkeypatch, tmp_path):
    """A key loaded via python-dotenv allows _openai_call to run."""

    fixture = _write_fixture(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=fake-key\n")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_dotenv(env_file)
    assert os.getenv("OPENAI_API_KEY") == "fake-key"

    mock_call = MagicMock(return_value={
        "intent": "GREETING",
        "entities": {},
        "raw_response": "{}",
    })
    monkeypatch.setattr(evaluate_templates, "_openai_call", mock_call)
    monkeypatch.setattr(evaluate_templates, "OpenAI", MagicMock())
    monkeypatch.setattr(sys, "argv", ["evaluate_templates", str(fixture)])

    evaluate_templates.main()
    assert mock_call.call_count > 0
