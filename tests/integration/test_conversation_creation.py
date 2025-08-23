from conversation_service.repositories import ConversationRepository
from conversation_service.schemas import ConversationCreate
from tests.test_phase_0.conftest import db_session, user  # noqa: F401


def test_conversation_create_with_metadata(db_session, user):
    repo = ConversationRepository(db_session)
    conv = repo.create(
        ConversationCreate(
            user_id=user.id,
            conversation_metadata={"topic": "budget"},
            user_preferences={"tone": "formal"},
            session_metadata={"ip": "127.0.0.1"},
        )
    )

    fetched = repo.get_conversation(conv.conversation_id, user_id=user.id)

    assert fetched.conversation_metadata == {"topic": "budget"}
    assert fetched.user_preferences == {"tone": "formal"}
    assert fetched.session_metadata == {"ip": "127.0.0.1"}
