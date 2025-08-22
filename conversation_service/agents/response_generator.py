"""Response generation agent."""

from typing import Any, Dict, Optional
import asyncio
import json

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts import response_prompts


class ResponseGeneratorAgent(BaseFinancialAgent):
    """Craft natural language responses from search results."""

    def __init__(self, openai_client):
        config = AgentConfig(
            name="response_generator",
            system_message=response_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = response_prompts.get_examples()

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a personalised response from search results.

        The method leverages the OpenAI client to craft a natural language
        summary of ``search_response`` while incorporating user profile
        information for formatting and tailored insights.
        """

        context = input_data.get("context", {})
        search_response = input_data.get("search_response", {})
        user_profile = context.get("user_profile", {})

        # Prepare a short JSON serialisation of the search results for the
        # language model.  Only include the top few entries to keep the prompt
        # compact.
        results = search_response.get("results", [])
        top_results = results[:3]
        results_json = json.dumps(top_results, ensure_ascii=False)

        prompt = (
            "Tu es un assistant financier. Résume les résultats suivants en "
            "fournissant des conseils adaptés au profil utilisateur.\n\n"
            f"Profil utilisateur: {json.dumps(user_profile, ensure_ascii=False)}\n"
            f"Résultats de recherche: {results_json}"
        )

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    self._call_openai(prompt, few_shot_examples=self.examples),
                    timeout=self.config.timeout_seconds,
                )
                message = response["content"].strip()
                break
            except Exception as exc:  # pragma: no cover - network/timeout
                last_error = exc
                if attempt >= 2:
                    raise
                await asyncio.sleep(2 ** attempt)

        if last_error:
            raise last_error

        user_name = user_profile.get("name", "client")
        formatted = f"### Résumé personnalisé pour {user_name}\n\n{message}"
        insights = {"result_count": len(results)}

        return {
            "input": input_data,
            "context": context,
            "response": formatted,
            "insights": insights,
        }
