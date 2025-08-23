import ast
import json
import pathlib

import pytest


def parse_taxonomy():
    """Parse INTENTS.md into intent -> {category, actions} mapping."""
    lines = pathlib.Path("INTENTS.md").read_text(encoding="utf-8").splitlines()
    rows: list[str] = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer.endswith("|"):
                rows.append(buffer)
                buffer = ""
            continue
        if stripped.startswith("|") or buffer:
            buffer += stripped
            if stripped.endswith("|"):
                rows.append(buffer)
                buffer = ""
    taxonomy: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        if row.startswith("| Intent Type") or "---" in row:
            continue
        cols = [c.strip() for c in row.strip("|").split("|")]
        if len(cols) >= 4:
            intent = cols[0]
            category = cols[1].split()[0]
            actions_str = cols[3]
            try:
                actions = ast.literal_eval(actions_str)
                if not isinstance(actions, list):
                    actions = []
            except Exception:
                actions = []
            taxonomy[intent] = {"category": category, "actions": actions}
    return taxonomy


def build_examples(taxonomy: dict[str, dict[str, list[str]]]):
    examples: list[str] = []
    for intent, data in taxonomy.items():
        for lang in ("fr", "en"):
            sample = {
                "message": f"Example {intent} in {lang}",
                "language": lang,
                "intent_type": intent,
                "intent_category": data["category"],
            }
            if data["actions"]:
                sample["actions"] = data["actions"]
            if lang == "en":
                sample["optional_field"] = "foo"
            examples.append(json.dumps(sample))
    return examples


def test_examples_json_and_taxonomy():
    taxonomy = parse_taxonomy()
    examples = build_examples(taxonomy)
    assert len(examples) >= 60
    for example in examples:
        data = json.loads(example)
        intent = data["intent_type"]
        assert intent in taxonomy
        expected = taxonomy[intent]
        assert data["intent_category"] == expected["category"]
        if expected["actions"]:
            assert data["actions"] == expected["actions"]
        else:
            assert "actions" not in data or data["actions"] == []
        if data["language"] == "en":
            assert data.get("optional_field") == "foo"
        else:
            assert "optional_field" not in data


def test_security_and_out_of_taxonomy():
    taxonomy = parse_taxonomy()
    # Unsupported or unclear intents should not expose actions
    safe_actions = {"ask_to_rephrase", "no_action", "retry_or_contact_support"}
    for intent, info in taxonomy.items():
        if info["category"] in {"UNSUPPORTED", "UNCLEAR_INTENT"}:
            assert set(info["actions"]) <= safe_actions
    # Example outside the taxonomy
    invalid = {"intent_type": "MALICIOUS_INTENT", "intent_category": "UNSUPPORTED"}
    with pytest.raises(KeyError):
        taxonomy[invalid["intent_type"]]
