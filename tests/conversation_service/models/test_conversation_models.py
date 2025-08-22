from datetime import datetime, timedelta, timezone

import pytest
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from conversation_service.models import Conversation, ConversationSummary, ConversationTurn


@pytest.fixture()
def timestamps():
    now = datetime.now(timezone.utc)
    later = now + timedelta(seconds=10)
    return now, later


def test_conversation_validation_and_json(timestamps):
    now, later = timestamps
    conv = Conversation(
        id=1,
        conversation_id="c1",
        user_id=42,
        title="Budget",
        status="active",
        language="fr",
        domain="finance",
        total_turns=1,
        max_turns=5,
        last_activity_at=later,
        conversation_metadata={"topic": "budget"},
        user_preferences={},
        session_metadata={},
        created_at=now,
        updated_at=later,
    )
    import json
    from datetime import datetime

    data = {
        k: (v.isoformat() if isinstance(v, datetime) else v)
        for k, v in conv.dict().items()
    }
    json_data = json.dumps(data)
    loaded = Conversation(**json.loads(json_data))
    assert loaded.conversation_id == conv.conversation_id
    assert loaded.user_id == conv.user_id

    with pytest.raises(ValidationError):
        Conversation(
            id=1,
            conversation_id="c1",
            user_id=-1,
            status="active",
            language="fr",
            domain="finance",
            total_turns=1,
            max_turns=5,
            last_activity_at=now,
            created_at=now,
            updated_at=now,
        )

    with pytest.raises(ValidationError):
        Conversation(
            id=1,
            conversation_id="c1",
            user_id=42,
            status="active",
            language="fr",
            domain="finance",
            total_turns=6,
            max_turns=5,
            last_activity_at=later,
            created_at=now,
            updated_at=later,
        )

    with pytest.raises(ValidationError):
        Conversation(
            id=1,
            conversation_id="c1",
            user_id=42,
            status="active",
            language="fr",
            domain="finance",
            total_turns=1,
            max_turns=5,
            last_activity_at=now,
            created_at=later,
            updated_at=later,
        )

    with pytest.raises(ValidationError):
        Conversation(
            id=1,
            conversation_id="c1",
            user_id=42,
            status="active",
            language="fr",
            domain="finance",
            total_turns=1,
            max_turns=5,
            last_activity_at=now,
            created_at=later,
            updated_at=now,
        )


def test_summary_validation_and_json(timestamps):
    now, later = timestamps
    summary = ConversationSummary(
        id=1,
        conversation_id=1,
        start_turn=1,
        end_turn=2,
        summary_text="ok",
        key_topics=["budget"],
        important_entities=[{"name": "Paris", "type": "city"}],
        summary_method="llm",
        created_at=now,
        updated_at=later,
    )
    import json
    from datetime import datetime

    data = {
        k: (v.isoformat() if isinstance(v, datetime) else v)
        for k, v in summary.dict().items()
    }
    json_data = json.dumps(data)
    loaded = ConversationSummary(**json.loads(json_data))
    assert loaded.conversation_id == summary.conversation_id
    assert loaded.start_turn == summary.start_turn

    with pytest.raises(ValidationError):
        ConversationSummary(
            id=1,
            conversation_id=-1,
            start_turn=1,
            end_turn=2,
            summary_text="bad",
            summary_method="llm",
            created_at=now,
            updated_at=later,
        )

    with pytest.raises(ValidationError):
        ConversationSummary(
            id=1,
            conversation_id=1,
            start_turn=2,
            end_turn=1,
            summary_text="bad",
            summary_method="llm",
            created_at=now,
            updated_at=later,
        )


def test_turn_validation_and_json(timestamps):
    now, later = timestamps
    turn = ConversationTurn(
        id=1,
        turn_id="t1",
        conversation_id=1,
        turn_number=1,
        user_message="hi",
        assistant_response="ok",
        error_occurred=False,
        agent_chain=[],
        search_results_count=0,
        created_at=now,
        updated_at=later,
    )
    import json
    from datetime import datetime

    data = {
        k: (v.isoformat() if isinstance(v, datetime) else v)
        for k, v in turn.dict().items()
    }
    json_data = json.dumps(data)
    loaded = ConversationTurn(**json.loads(json_data))
    assert loaded.turn_id == turn.turn_id
    assert loaded.conversation_id == turn.conversation_id

    with pytest.raises(ValidationError):
        ConversationTurn(
            id=1,
            turn_id="t1",
            conversation_id=-1,
            turn_number=1,
            user_message="hi",
            assistant_response="ok",
            error_occurred=False,
            agent_chain=[],
            search_results_count=0,
            created_at=now,
            updated_at=later,
        )

    with pytest.raises(ValidationError):
        ConversationTurn(
            id=1,
            turn_id="t1",
            conversation_id=1,
            turn_number=1,
            user_message="hi",
            assistant_response="ok",
            error_occurred=False,
            agent_chain=[],
            search_results_count=-1,
            created_at=now,
            updated_at=later,
        )

    with pytest.raises(ValidationError):
        ConversationTurn(
            id=1,
            turn_id="t1",
            conversation_id=1,
            turn_number=1,
            user_message="hi",
            assistant_response="ok",
            error_occurred=False,
            agent_chain=[],
            search_results_count=0,
            processing_time_ms=-5.0,
            created_at=now,
            updated_at=later,
        )

    with pytest.raises(ValidationError):
        ConversationTurn(
            id=1,
            turn_id="t1",
            conversation_id=1,
            turn_number=1,
            user_message="hi",
            assistant_response="ok",
            error_occurred=False,
            agent_chain=[],
            search_results_count=0,
            created_at=later,
            updated_at=now,
        )
