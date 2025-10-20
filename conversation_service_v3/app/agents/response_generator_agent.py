"""
Response Generator Agent - Génère la réponse finale
Utilise les agrégations + résumé + transactions pour créer une réponse pertinente
"""
import logging
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..models import SearchResults, AgentResponse, AgentRole, ConversationResponse

logger = logging.getLogger(__name__)


class ResponseGeneratorAgent:
    """
    Agent de génération de réponse finale

    Responsabilités:
    - Analyser les agrégations Elasticsearch
    - Résumer les résultats de recherche
    - Créer une réponse naturelle et pertinente
    - Inclure les détails des premières transactions si pertinent
    """

    def __init__(self, llm_model: str = "gpt-4o", temperature: float = 0.3):
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )

        # Prompt pour la génération de réponse
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un assistant financier personnel qui aide les utilisateurs à comprendre leurs transactions bancaires.

Ton rôle est de créer une réponse claire, précise et utile basée sur:
1. Les agrégations Elasticsearch (totaux, moyennes, statistiques)
2. Un résumé des résultats de recherche
3. Les premières transactions détaillées (max 50)

Règles de réponse:
- Commence par répondre directement à la question
- Utilise les chiffres des agrégations pour donner des totaux/statistiques
- Mentionne les insights intéressants (plus grosse dépense, catégorie principale, etc.)
- Sois naturel et conversationnel
- Utilise des émojis modérément pour rendre la réponse agréable
- Si aucun résultat, explique pourquoi et propose des alternatives

Format de réponse:
1. Réponse directe à la question avec les chiffres clés
2. Insights et observations
3. Détails des principales transactions (si pertinent)
4. Suggestion d'action ou question de suivi (optionnel)

Exemples:

Question: "Combien j'ai dépensé en courses ce mois-ci ?"
Agrégations: {{"total_amount": {{"value": -342.50}}, "count": 12}}
Réponse:
"Tu as dépensé **342,50 €** en courses ce mois-ci, répartis sur 12 transactions.

📊 **Observations:**
- Dépense moyenne par course: ~28,54 €
- Ta plus grosse course était de 78,30 € chez Carrefour le 12 janvier

Les principales transactions:
1. Carrefour - 78,30 € (12 jan)
2. Monoprix - 45,20 € (08 jan)
3. Lidl - 32,10 € (05 jan)
..."

Question: "Montre-moi mes transactions chez Carrefour"
Transactions: 8 résultats trouvés
Réponse:
"J'ai trouvé **8 transactions chez Carrefour** dans ton historique.

💰 **Statistiques:**
- Total dépensé: 456,80 €
- Dépense moyenne: 57,10 €
- Plus grosse transaction: 98,50 €

📝 **Dernières transactions:**
1. 12 jan - 78,30 € (Alimentation)
2. 05 jan - 98,50 € (Alimentation)
3. 28 déc - 45,60 € (Alimentation)
..."
"""),
            ("user", """Question utilisateur: {user_message}

**Agrégations Elasticsearch:**
{aggregations}

**Résumé des résultats:**
- Nombre total de résultats: {total_results}
- Nombre de transactions retournées: {transactions_count}

**Transactions (premières {transactions_count}):**
{transactions}

Génère une réponse complète et utile.""")
        ])

        self.chain = self.prompt | self.llm

        self.stats = {
            "responses_generated": 0,
            "avg_response_length": 0
        }

        logger.info(f"ResponseGeneratorAgent initialized with model {llm_model}")

    async def generate_response(
        self,
        user_message: str,
        search_results: SearchResults,
        original_query_analysis: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Génère une réponse finale basée sur les résultats de recherche

        Args:
            user_message: Message original de l'utilisateur
            search_results: Résultats Elasticsearch (hits + agrégations)
            original_query_analysis: Analyse originale (pour contexte)

        Returns:
            AgentResponse contenant ConversationResponse
        """
        try:
            logger.info(f"Generating response for query: {user_message[:100]}")

            # Préparer les agrégations pour le LLM
            aggs_summary = self._format_aggregations(search_results.aggregations)

            # Préparer les transactions pour le LLM
            transactions_text = self._format_transactions(search_results.hits[:50])  # Max 50

            # Invoquer le LLM
            result = await self.chain.ainvoke({
                "user_message": user_message,
                "aggregations": aggs_summary,
                "total_results": search_results.total,
                "transactions_count": len(search_results.hits[:50]),
                "transactions": transactions_text
            })

            response_text = result.content

            # Créer la réponse de conversation
            conversation_response = ConversationResponse(
                success=True,
                message=response_text,
                search_results=search_results,
                aggregations_summary=aggs_summary,
                metadata={
                    "total_results": search_results.total,
                    "response_length": len(response_text),
                    "took_ms": search_results.took_ms
                }
            )

            # Mise à jour des stats
            self.stats["responses_generated"] += 1
            current_avg = self.stats["avg_response_length"]
            total_responses = self.stats["responses_generated"]
            self.stats["avg_response_length"] = (
                (current_avg * (total_responses - 1) + len(response_text)) / total_responses
            )

            logger.info(f"Response generated successfully. Length: {len(response_text)} chars")

            return AgentResponse(
                success=True,
                data=conversation_response,
                agent_role=AgentRole.RESPONSE_GENERATOR,
                metadata={
                    "response_length": len(response_text),
                    "total_results": search_results.total
                }
            )

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return AgentResponse(
                success=False,
                data=None,
                agent_role=AgentRole.RESPONSE_GENERATOR,
                error=str(e)
            )

    def _format_aggregations(self, aggregations: Optional[Dict[str, Any]]) -> str:
        """
        Formate les agrégations Elasticsearch de manière lisible pour le LLM

        Args:
            aggregations: Agrégations brutes d'Elasticsearch

        Returns:
            String formatée
        """
        if not aggregations:
            return "Aucune agrégation disponible"

        formatted_lines = []

        for agg_name, agg_data in aggregations.items():
            if isinstance(agg_data, dict):
                # Agrégations de valeur simple (sum, avg, etc.)
                if "value" in agg_data:
                    formatted_lines.append(f"- {agg_name}: {agg_data['value']:.2f}")

                # Statistiques
                elif "count" in agg_data and "sum" in agg_data:
                    formatted_lines.append(f"- {agg_name}:")
                    formatted_lines.append(f"  - Count: {agg_data.get('count', 0)}")
                    formatted_lines.append(f"  - Sum: {agg_data.get('sum', 0):.2f}")
                    formatted_lines.append(f"  - Avg: {agg_data.get('avg', 0):.2f}")
                    formatted_lines.append(f"  - Min: {agg_data.get('min', 0):.2f}")
                    formatted_lines.append(f"  - Max: {agg_data.get('max', 0):.2f}")

                # Agrégations terms (buckets)
                elif "buckets" in agg_data:
                    formatted_lines.append(f"- {agg_name}:")
                    buckets = agg_data["buckets"][:10]  # Top 10
                    for bucket in buckets:
                        key = bucket.get("key", "Unknown")
                        doc_count = bucket.get("doc_count", 0)
                        formatted_lines.append(f"  - {key}: {doc_count} transactions")

                        # Sous-agrégations
                        for sub_agg_name, sub_agg_data in bucket.items():
                            if sub_agg_name not in ["key", "doc_count", "key_as_string"] and isinstance(sub_agg_data, dict):
                                if "value" in sub_agg_data:
                                    formatted_lines.append(f"    - {sub_agg_name}: {sub_agg_data['value']:.2f}")

        return "\n".join(formatted_lines) if formatted_lines else "Pas d'agrégations pertinentes"

    def _format_transactions(self, transactions: List[Dict[str, Any]]) -> str:
        """
        Formate les transactions de manière lisible pour le LLM

        Args:
            transactions: Liste des transactions

        Returns:
            String formatée
        """
        if not transactions:
            return "Aucune transaction trouvée"

        formatted_lines = []

        for idx, transaction in enumerate(transactions[:50], 1):  # Max 50
            source = transaction.get("_source", {})

            # Extraire les informations clés
            date = source.get("date", "Date inconnue")
            amount = source.get("amount", 0)
            merchant = source.get("merchant_name", "Marchand inconnu")
            category = source.get("category_name", "")
            description = source.get("primary_description", "")

            # Formater le montant
            amount_str = f"{abs(amount):.2f} €"
            if amount < 0:
                amount_str = f"-{amount_str}"

            # Ligne de transaction
            line = f"{idx}. {date} - {merchant} - {amount_str}"
            if category:
                line += f" ({category})"

            formatted_lines.append(line)

            # Ajouter description si disponible et différente du marchand
            if description and description.lower() != merchant.lower():
                formatted_lines.append(f"   Description: {description[:100]}")

        return "\n".join(formatted_lines)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        return {
            "agent": "response_generator",
            "model": self.llm.model_name,
            **self.stats
        }
