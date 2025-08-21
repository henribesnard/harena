"""Ensure the :mod:`conversation_service.agents` package imports cleanly."""


def test_agents_package_import() -> None:
    """Import the package and access the exported utility."""
    from conversation_service import agents

    assert hasattr(agents, "QueryOptimizer")

