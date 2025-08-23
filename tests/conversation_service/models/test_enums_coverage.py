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
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and not line.startswith("| Intent Type") and "---" not in line:
            cols = [c.strip() for c in line.strip("|").split("|")]
            if cols:
                intents.append(cols[0])
    return intents


def test_intent_type_matches_intents_md():
    intents = parse_intents_md()
    assert set(intents) == {i.value for i in IntentType}


def test_query_type_expected_values():
    expected = {
        "FINANCIAL_QUERY",
        "SPENDING_ANALYSIS",
        "ACCOUNT_BALANCE",
        "CONVERSATION",
        "UNSUPPORTED",
    }
    assert expected == {q.value for q in QueryType}


def test_entity_type_expected_values():
    expected = {
        "ACCOUNT",
        "TRANSACTION",
        "MERCHANT",
        "CATEGORY",
        "DATE",
        "PERIOD",
        "AMOUNT",
        "OPERATION_TYPE",
        "TEXT",
    }
    assert expected == {e.value for e in EntityType}
