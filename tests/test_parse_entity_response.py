import sys
import types


def _make_agent(monkeypatch):
    """Create a SearchQueryAgent instance with minimal dependencies."""
    # Provide minimal modules to satisfy import requirements
    monkeypatch.setitem(sys.modules, "httpx", types.ModuleType("httpx"))
    dummy_deepseek = types.ModuleType("deepseek_client")

    class _DummyDeepSeek:
        pass

    dummy_deepseek.DeepSeekClient = _DummyDeepSeek
    monkeypatch.setitem(
        sys.modules, "conversation_service.core.deepseek_client", dummy_deepseek
    )

    dummy_base = types.ModuleType("base_financial_agent")

    class _DummyBase:  # placeholder for BaseFinancialAgent
        pass

    dummy_base.BaseFinancialAgent = _DummyBase
    monkeypatch.setitem(
        sys.modules, "conversation_service.agents.base_financial_agent", dummy_base
    )

    dummy_agent_models = types.ModuleType("agent_models")
    dummy_agent_models.AgentConfig = object
    dummy_agent_models.AgentResponse = object
    monkeypatch.setitem(
        sys.modules, "conversation_service.models.agent_models", dummy_agent_models
    )

    dummy_service_contracts = types.ModuleType("service_contracts")
    dummy_service_contracts.SearchServiceQuery = object
    dummy_service_contracts.SearchServiceResponse = object
    dummy_service_contracts.QueryMetadata = object
    dummy_service_contracts.SearchParameters = object
    dummy_service_contracts.SearchFilters = object
    monkeypatch.setitem(
        sys.modules, "conversation_service.models.service_contracts", dummy_service_contracts
    )

    dummy_financial_models = types.ModuleType("financial_models")

    class _IntentResult:
        pass

    class _FinancialEntity:
        pass

    class _EntityType:
        pass

    dummy_financial_models.IntentResult = _IntentResult
    dummy_financial_models.FinancialEntity = _FinancialEntity
    dummy_financial_models.EntityType = _EntityType
    monkeypatch.setitem(
        sys.modules, "conversation_service.models.financial_models", dummy_financial_models
    )

    dummy_validators = types.ModuleType("validators")
    dummy_validators.ContractValidator = object
    monkeypatch.setitem(
        sys.modules, "conversation_service.utils.validators", dummy_validators
    )

    from conversation_service.agents import search_query_agent

    class DummyEntity:
        def __init__(self, entity_type, raw_value, normalized_value, confidence):
            self.entity_type = entity_type
            self.raw_value = raw_value
            self.normalized_value = normalized_value
            self.confidence = confidence

    # Patch EntityType to return the string itself
    monkeypatch.setattr(search_query_agent, "EntityType", lambda s: s)
    # Patch FinancialEntity to a simple container that keeps the string type
    monkeypatch.setattr(search_query_agent, "FinancialEntity", DummyEntity)

    return search_query_agent.SearchQueryAgent.__new__(
        search_query_agent.SearchQueryAgent
    )


def test_parse_entity_response_accepts_string_entity_type(monkeypatch):
    agent = _make_agent(monkeypatch)
    content = '[{"type": "MERCHANT", "value": "Carrefour"}]'

    entities = agent._parse_entity_response(content)

    assert entities and entities[0].entity_type == "MERCHANT"


def test_parse_entity_response_handles_invalid_json(monkeypatch):
    agent = _make_agent(monkeypatch)
    content = "not json"

    entities = agent._parse_entity_response(content)

    assert entities == []

