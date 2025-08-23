import importlib.util
import pathlib

import pytest

# Load enums module directly to avoid heavy package imports
spec = importlib.util.spec_from_file_location(
    "enums", pathlib.Path("conversation_service/models/enums.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

IntentType = module.IntentType
EntityType = module.EntityType
QueryType = module.QueryType


def parse_intents_md():
    text = pathlib.Path("INTENTS.md").read_text()
    intents = []
    categories = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and not line.startswith("| Intent Type") and "---" not in line:
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) >= 2:
                intents.append(cols[0])
                categories.append(cols[1].split()[0])
    return intents, categories


def test_intent_type_matches_intents_md():
    intents, _ = parse_intents_md()
    assert set(intents) == {i.value for i in IntentType}


def test_query_type_matches_categories():
    _, categories = parse_intents_md()
    assert set(categories) == {q.value for q in QueryType}


def test_entity_type_expected_values():
    expected = {
        "AMOUNT",
        "TEMPORAL",
        "MERCHANT",
        "CATEGORY",
        "ACCOUNT",
        "OPERATION_TYPE",
        "LOCATION",
        "TEXT_QUERY",
    }
    assert expected == {e.value for e in EntityType}
