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

