"""Simple orchestrator that chains classification, extraction, querying and response."""

from typing import Any, Dict, List

from agent_types import ChatMessage, TaskResult


class TeamOrchestrator:
    """Run a pipeline of assistant agents sequentially."""

    def __init__(
        self,
        classifier,
        extractor,
        query_agent,
        responder,
    ) -> None:
        self._classifier = classifier
        self._extractor = extractor
        self._query_agent = query_agent
        self._responder = responder
        self.context: Dict[str, Any] = {}

    async def run(self, task: str) -> TaskResult:
        """Execute the pipeline and return the resulting messages."""
        messages: List[ChatMessage] = [ChatMessage(content=task, source="user")]
        self.context = {}
        for agent in [
            self._classifier,
            self._extractor,
            self._query_agent,
            self._responder,
        ]:
            if agent is None:
                continue
            response = await agent.on_messages(messages, None)
            msg = response.chat_message
            messages.append(msg)
            name = getattr(agent, "name", agent.__class__.__name__)
            self.context[name] = msg.content
        return TaskResult(messages=messages)

    async def query_agents(
        self, conversation_id: str, message: str, user_id: int, db: Session
    ) -> str:
        start = time.time()
        history_models = self.get_history(conversation_id, db) or []
        context: Dict[str, Any] = {
            "user_message": message,
            "user_id": user_id,
            "history": [m.model_dump() for m in history_models],
        }
        repo = ConversationMessageRepository(db)
        repo.add(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=message,
        )
        self._total_calls += 1
        try:
            context = await self._call_agent(
                self._classifier, context, repo, conversation_id, user_id
            )
            context = await self._call_agent(
                self._extractor, context, repo, conversation_id, user_id
            )
            context = await self._call_agent(
                self._query_agent, context, repo, conversation_id, user_id
            )
            context = await self._call_agent(
                self._responder, context, repo, conversation_id, user_id
            )
            reply = context.get("response", "")
        except Exception:
            self._error_calls += 1
            logger.exception("Agent processing failed")
            reply = (
                "Désolé, une erreur est survenue lors du traitement de votre demande."
            )
        repo.add(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=reply,
        )
        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="query_agents", success=True, processing_time_ms=duration
        )
        return reply

    def get_error_metrics(self) -> Dict[str, float]:
        return {
            "total_calls": float(self._total_calls),
            "error_calls": float(self._error_calls),
            "error_rate": (
                self._error_calls / self._total_calls if self._total_calls else 0.0
            ),
        }

