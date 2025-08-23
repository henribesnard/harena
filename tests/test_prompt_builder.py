from conversation_service.prompts.utils.prompt_builder import PromptBuilder, PromptSection


def test_prompt_builder_section_selection_and_context():
    sections = [
        PromptSection(text="Global {name}", priority=1),
        PromptSection(text="Intent: {intent}", intents=["greet"], priority=1),
        PromptSection(text="Unused", intents=["other"], priority=1),
    ]
    builder = PromptBuilder(sections)
    result = builder.build("greet", {"name": "Alice", "intent": "greet"})
    assert "Global Alice" in result
    assert "Intent: greet" in result
    assert "Unused" not in result


def test_prompt_builder_cache_uses_context_and_intent():
    sections = [PromptSection(text="Hello {name} {age}")]
    builder = PromptBuilder(sections)
    first = builder.build("greet", {"name": "Bob", "age": 30})
    second = builder.build("greet", {"age": 30, "name": "Bob"})
    assert first == second
    # the cache should only contain one entry even though context dict order differed
    assert len(builder._cache) == 1


def test_prompt_builder_length_optimization():
    sections = [
        PromptSection(text="A" * 40, priority=1),
        PromptSection(text="B" * 40, priority=0),
    ]
    builder = PromptBuilder(sections, max_length=50)
    result = builder.build("any")
    assert result.startswith("A" * 40)
    assert "B" in result
    assert "B" * 40 not in result
    assert len(result) <= 50
