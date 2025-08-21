from conversation_service.prompts import query_prompts


def test_system_prompt_mentions_elasticsearch():
    assert "Elasticsearch" in query_prompts.QUERY_GENERATION_SYSTEM_PROMPT
    assert "user_id" in query_prompts.QUERY_GENERATION_SYSTEM_PROMPT


def test_few_shot_examples_have_input_and_output():
    example = query_prompts.QUERY_FEW_SHOT_EXAMPLES[0]
    assert "input" in example and "output" in example
