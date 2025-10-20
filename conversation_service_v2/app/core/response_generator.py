"""Module 6: Natural Language Response Generation using DeepSeek API."""

from openai import AsyncOpenAI
from typing import Dict, Any
import os

# DeepSeek client
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-your-deepseek-api-key"),
    base_url="https://api.deepseek.com"
)

RESPONSE_GENERATION_SYSTEM_PROMPT = """
Tu es Harena, un assistant financier intelligent et bienveillant.

Ton rôle est d'analyser les données financières de l'utilisateur et de lui fournir:
1. Une réponse claire et naturelle à sa question
2. Des insights pertinents sur ses finances
3. Des recommandations actionnables pour améliorer sa gestion

STYLE DE COMMUNICATION:
- Utilise un ton amical et conversationnel
- Sois précis avec les chiffres (toujours 2 décimales pour les montants)
- Mets en évidence les tendances importantes
- Donne des recommandations concrètes et réalistes
- Utilise des emojis de manière subtile (pas trop)

STRUCTURE DE RÉPONSE:
1. Réponse directe à la question (1-2 phrases)
2. Détails complémentaires pertinents
3. Contexte et comparaisons si disponibles

INSIGHTS À GÉNÉRER (3-5 maximum):
- Tendances (hausse/baisse par rapport aux périodes précédentes)
- Alertes budget (dépassement ou respect)
- Patterns de dépenses (commerces récurrents, catégories dominantes)
- Anomalies (transactions inhabituelles)

RECOMMANDATIONS À GÉNÉRER (2-3 maximum):
- Actions concrètes pour optimiser les finances
- Suggestions d'économies basées sur les données
- Conseils de budgétisation personnalisés

IMPORTANT:
- Base-toi UNIQUEMENT sur les données fournies dans le contexte
- Si les données sont insuffisantes, dis-le clairement
- Ne fais pas d'hypothèses non fondées
"""

RESPONSE_GENERATION_USER_PROMPT = """
Question de l'utilisateur: "{user_query}"

CONTEXTE (données financières):
{context}

Génère une réponse complète avec:
1. "answer": La réponse naturelle à la question
2. "insights": Liste de 3-5 insights pertinents
3. "recommendations": Liste de 2-3 recommandations actionnables

Format de sortie JSON:
{{
  "answer": "...",
  "insights": ["...", "...", "..."],
  "recommendations": ["...", "..."]
}}

Réponds UNIQUEMENT avec le JSON, sans texte avant ni après.
"""


class ResponseGenerator:
    """Generator for natural language responses using DeepSeek API."""

    def __init__(self):
        """Initialize the response generator."""
        self.client = deepseek_client

    async def generate(
        self,
        user_query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Generate natural language response from context.

        Args:
            user_query: Original user question
            context: Formatted context from SQL results

        Returns:
            dict: Response with answer, insights, and recommendations

        Raises:
            Exception: If API call fails
        """
        # Prepare prompt
        user_prompt = RESPONSE_GENERATION_USER_PROMPT.format(
            user_query=user_query,
            context=context
        )

        # Call DeepSeek API
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": RESPONSE_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Slightly creative but mostly factual
            max_tokens=1024,
            response_format={"type": "json_object"}
        )

        # Parse JSON response
        import json
        result = json.loads(response.choices[0].message.content)

        return {
            "answer": result.get("answer", ""),
            "insights": result.get("insights", []),
            "recommendations": result.get("recommendations", []),
            "tokens_used": {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
        }
