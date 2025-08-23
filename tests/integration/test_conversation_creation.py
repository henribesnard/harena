from conversation_service.repositories import ConversationRepository
from conversation_service.schemas import ConversationCreate
from tests.test_phase_0.conftest import db_session, user  # noqa: F401


def test_conversation_create_with_metadata(db_session, user):
    repo = ConversationRepository(db_session)
    conv = repo.create(
        ConversationCreate(
            user_id=user.id,
            financial_context={"balance": 100},
            user_preferences_ai={"tone": "formal"},
            key_entities_history=[{"name": "Account", "type": "bank_account"}],
        )
    )

    fetched = repo.get_conversation(conv.conversation_id, user_id=user.id)

    assert fetched.financial_context == {"balance": 100}
    assert fetched.user_preferences_ai == {"tone": "formal"}
    assert fetched.key_entities_history == [
        {"name": "Account", "type": "bank_account"}
    ]
