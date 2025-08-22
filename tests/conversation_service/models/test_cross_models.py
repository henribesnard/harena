from datetime import datetime, timedelta, timezone

from conversation_service.models import AgentStep, AgentTrace, ConversationTurn


def test_agent_steps_roundtrip_with_conversation_turn():
    now = datetime.now(timezone.utc)
    later = now + timedelta(seconds=5)
    step = AgentStep(agent="retriever", status="ok")
    turn = ConversationTurn(
        id=1,
        turn_id="t1",
        conversation_id=1,
        turn_number=1,
        user_message="hi",
        assistant_response="ok",
        error_occurred=False,
        agent_chain=[step.model_dump()],
        search_results_count=0,
        created_at=now,
        updated_at=later,
    )
    recovered = AgentStep(**turn.agent_chain[0])
    assert recovered.model_dump() == step.model_dump()


def test_agent_trace_processing_time_matches():
    now = datetime.now(timezone.utc)
    later = now + timedelta(seconds=5)
    step = AgentStep(agent="retriever", status="ok")
    trace = AgentTrace(steps=[step], total_time_ms=20.0)
    turn = ConversationTurn(
        id=1,
        turn_id="t1",
        conversation_id=1,
        turn_number=1,
        user_message="hi",
        assistant_response="ok",
        error_occurred=False,
        agent_chain=[s.model_dump() for s in trace.steps],
        search_results_count=0,
        processing_time_ms=trace.total_time_ms,
        created_at=now,
        updated_at=later,
    )
    assert turn.processing_time_ms == trace.total_time_ms
