"""Query generation agent."""

from typing import Any, Dict, Optional

from pydantic import ValidationError

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts import query_prompts
from ..clients import SearchClient, OpenAIClient
from search_service.models import SearchRequest


class QueryGeneratorAgent(BaseFinancialAgent):
    """Generate search queries based on extracted data."""

    def __init__(
        self,
        search_client: SearchClient,
        openai_client: Optional[OpenAIClient] = None,
    ):
        """Create a new :class:`QueryGeneratorAgent`.

        Parameters
        ----------
        search_client:
            Client used to execute search queries.
        openai_client:
            Optional OpenAI client instance forwarded to
            :class:`BaseFinancialAgent`. When ``None``, the base class will not
            create an AutoGen ``AssistantAgent``.
        """
        config = AgentConfig(
            name="query_generator",
            system_message=query_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.search_client = search_client

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build and execute a search request based on ``input_data``."""

        if input_data.get("intent") is None or input_data.get("entities") is None:
            raise ValueError("intent and entities are required")

        context = input_data.get("context", {})
        user_id = context.get("user_id")

        filters = (context.get("filters") or {}).copy()
        filters["user_id"] = user_id

        try:
            search_request = SearchRequest(
                user_id=user_id,
                query=context.get("query", ""),
                filters=filters,
                aggregations=context.get("aggregations"),
            )
        except ValidationError:
            # If validation fails we do not query the search service
            return None

        response = await self.search_client.search(
            search_request.user_id, search_request.model_dump()
        )

        return {
            "input": input_data,
            "context": context,
            "search_request": search_request.model_dump(),
            "search_response": response,
        }
