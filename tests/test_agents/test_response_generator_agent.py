import pytest

agent_module = pytest.importorskip("conversation_service.agents.response_generator_agent")


@pytest.mark.skip(reason="Response generator agent dependencies incomplete")
def test_placeholder():
    assert agent_module is not None
