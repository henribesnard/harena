from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_service.base import Base
from db_service.models.conversation import Conversation
from db_service.models.user import User
from conversation_service.message_repository import ConversationMessageRepository


def test_add_and_list_messages_with_int_conversation_id():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        conv = Conversation(user_id=user.id, conversation_id="conv1")
        session.add(conv)
        session.commit()
        session.refresh(conv)

        repo = ConversationMessageRepository(session)
        repo.add(
            conversation_id=conv.conversation_id,
            conversation_db_id=conv.id,
            user_id=user.id,
            role="user",
            content="hello",
        )

        messages = repo.list_models(conv.conversation_id)
        assert len(messages) == 1
        assert messages[0].conversation_id == conv.conversation_id
        assert messages[0].content == "hello"
