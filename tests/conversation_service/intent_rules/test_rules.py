import sys
from pathlib import Path
import pytest

# Assure que le répertoire racine est dans le PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from conversation_service.intent_rules import (
    RuleLoader,
    create_pattern_matcher,
    create_rule_engine,
)


@pytest.fixture(scope="module")
def rule_loader():
    """Charge le RuleLoader avec les fichiers JSON de règles."""
    return RuleLoader()


@pytest.fixture(scope="module")
def pattern_matcher(rule_loader):
    """Crée le PatternMatcher à partir des règles chargées."""
    return create_pattern_matcher(rule_loader)


@pytest.fixture(scope="module")
def rule_engine(rule_loader, pattern_matcher):
    """Instancie le RuleEngine avec le PatternMatcher configuré."""
    return create_rule_engine(rule_loader, pattern_matcher)


def test_exact_match(rule_engine):
    """Vérifie la détection d'une intention via correspondance exacte."""
    result = rule_engine.match_intent("bonjour")
    assert result is not None
    assert result.intent == "GREETING"
    assert result.method == "exact_match"


def test_fuzzy_match(rule_engine):
    """Vérifie la détection d'une intention via pattern (fuzzy)."""
    result = rule_engine.match_intent("transactions amazon")
    assert result is not None
    assert result.intent == "SEARCH_BY_MERCHANT"
    assert result.method == "pattern_match"
    merchants = result.entities.get("merchant")
    assert merchants
    assert merchants[0].normalized_value["merchant"] == "AMAZON"


def test_fallback_rule(rule_engine):
    """Vérifie le comportement de fallback lorsqu'aucune règle ne correspond."""
    result = rule_engine.match_intent("phrase sans correspondance")
    assert result is None
