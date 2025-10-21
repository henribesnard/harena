"""
Response Generator Agent - Génère la réponse finale
Utilise les agrégations + résumé + transactions pour créer une réponse pertinente
"""
import logging
import json
import os
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
        # ChatOpenAI charge automatiquement OPENAI_API_KEY depuis l'environnement
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )

        # Prompt pour la génération de réponse
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un assistant financier personnel expert en analyse de données.

IMPORTANT - Utilisation des données:
- Les AGRÉGATIONS contiennent les STATISTIQUES GLOBALES sur TOUS les résultats
- Les transactions détaillées sont des EXEMPLES ILLUSTRATIFS (limités à {transactions_count})
- TOUJOURS utiliser les AGRÉGATIONS pour les chiffres totaux et statistiques
- JAMAIS dire "j'ai trouvé {transactions_count} transactions" si le total est différent
- Les agrégations sont PRIORITAIRES sur les transactions détaillées

Ton rôle est de créer une réponse claire, précise et utile basée sur:
1. Les agrégations Elasticsearch (totaux, moyennes, statistiques) - SOURCE DE VÉRITÉ
2. Un résumé des résultats de recherche
3. Les premières transactions détaillées (exemples illustratifs)

Règles de réponse:
- Commence TOUJOURS par les chiffres des AGRÉGATIONS
- Utilise "vos/votre" (jamais "utilisateur 123")
- Mentionne les insights importants des agrégations
- Inclus des exemples de transactions SI pertinent
- Sois naturel et conversationnel
- Si aucun résultat, explique pourquoi et propose des alternatives

Format de réponse:
1. Réponse directe à la question avec les chiffres clés
2. Insights et observations
3. Détails des principales transactions (si pertinent)
4. Suggestion d'action ou question de suivi (optionnel)

Exemples de bonnes réponses:

Question: "Combien j'ai dépensé en courses ce mois-ci ?"
Agrégations: total_spent: 342.50, transaction_count: 12, avg_transaction: 28.54
✅ BON: "Vous avez dépensé **342,50 €** en courses ce mois-ci (basé sur 12 transactions).
         Dépense moyenne: 28,54€ par visite."
❌ MAUVAIS: "J'ai trouvé 10 transactions pour un total de 250€"
            (si les agrégations montrent 12 transactions et 342,50€)

Question: "Montre-moi mes achats Amazon"
Agrégations: total: 456.80, count: 8
Transactions détaillées: 5 affichées
✅ BON: "Vous avez **8 transactions** chez Amazon pour un total de **456,80€**.
         Voici vos principales transactions: [liste des 5 transactions]"
❌ MAUVAIS: "Voici vos 5 transactions Amazon pour 250€"
            (si les agrégations en montrent 8 pour 456,80€)

Question: "Répartition de mes dépenses par catégorie"
Agrégations: by_category avec 15 catégories, totaux et comptages
✅ BON: "Voici la répartition complète de vos dépenses par catégorie (15 catégories analysées):
         1. Alimentation: 342,50€ (12 transactions)
         2. Transport: 156,80€ (8 transactions)
         ..."
❌ MAUVAIS: "D'après les 10 transactions que je vois..."
            (les agrégations contiennent TOUTES les catégories)
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

        OPTIMISATION: Ne met en contexte que:
        - Le résumé de la recherche (total, etc.)
        - Les agrégations complètes
        - Les N premières transactions (défini par MAX_TRANSACTIONS_IN_CONTEXT)

        Args:
            user_message: Message original de l'utilisateur
            search_results: Résultats Elasticsearch (hits + agrégations)
            original_query_analysis: Analyse originale (pour contexte)

        Returns:
            AgentResponse contenant ConversationResponse
        """
        try:
            logger.info(f"Generating response for query: {user_message[:100]}")

            from ..config.settings import settings

            # Préparer les agrégations pour le LLM (toujours complètes)
            aggs_summary = self._format_aggregations(search_results.aggregations)

            # Limiter le nombre de transactions dans le contexte
            max_transactions = min(settings.MAX_TRANSACTIONS_IN_CONTEXT, len(search_results.hits))
            limited_transactions = search_results.hits[:max_transactions]

            # DEBUG: Log pour identifier le problème
            logger.info(f"Transactions received: {len(search_results.hits)}, Limited to: {len(limited_transactions)}")
            if len(limited_transactions) > 0:
                logger.info(f"🔍 First transaction keys: {list(limited_transactions[0].keys())}")
                logger.debug(f"🔍 First transaction full: {json.dumps(limited_transactions[0], indent=2, default=str)}")
            else:
                logger.warning("⚠️ No transactions in search_results.hits despite total > 0")

            if len(search_results.hits) > max_transactions:
                logger.debug(
                    f"Context limitation: {len(search_results.hits)} → {max_transactions} transactions"
                )

            # Préparer les transactions pour le LLM (limitées)
            transactions_text = self._format_transactions(limited_transactions)

            # DEBUG: Log formatted transactions
            logger.info(f"🔍 Formatted transactions (first 500 chars): {transactions_text[:500]}")

            # Invoquer le LLM
            result = await self.chain.ainvoke({
                "user_message": user_message,
                "aggregations": aggs_summary,
                "total_results": search_results.total,
                "transactions_count": len(limited_transactions),
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
        Formate les agrégations Elasticsearch de manière détaillée pour le LLM

        VERSION AMÉLIORÉE avec interprétation et contexte enrichi

        Args:
            aggregations: Agrégations brutes d'Elasticsearch

        Returns:
            String formatée avec contexte enrichi
        """
        if not aggregations:
            return "Aucune agrégation disponible"

        formatted_lines = []
        formatted_lines.append("📊 RÉSUMÉ STATISTIQUE COMPLET (SOURCE DE VÉRITÉ):\n")

        for agg_name, agg_data in aggregations.items():
            if isinstance(agg_data, dict):
                # Agrégation de valeur unique (sum, avg, etc.)
                if "value" in agg_data:
                    value = agg_data['value']
                    if value is not None:
                        formatted_lines.append(f"✅ {agg_name}: {value:.2f}")

                        # Ajouter interprétation
                        if "total" in agg_name.lower() or "sum" in agg_name.lower():
                            formatted_lines.append(f"   → Montant total calculé sur tous les résultats")
                        elif "avg" in agg_name.lower() or "moyenne" in agg_name.lower():
                            formatted_lines.append(f"   → Moyenne calculée")
                        elif "count" in agg_name.lower():
                            formatted_lines.append(f"   → Nombre total de transactions")

                # Statistiques complètes (stats aggregation)
                elif "count" in agg_data and "sum" in agg_data:
                    formatted_lines.append(f"\n📈 {agg_name} (Statistiques complètes):")
                    formatted_lines.append(f"   • Nombre: {agg_data.get('count', 0)}")
                    formatted_lines.append(f"   • Total: {agg_data.get('sum', 0):.2f}€")
                    formatted_lines.append(f"   • Moyenne: {agg_data.get('avg', 0):.2f}€")
                    formatted_lines.append(f"   • Min: {agg_data.get('min', 0):.2f}€")
                    formatted_lines.append(f"   • Max: {agg_data.get('max', 0):.2f}€")

                # Terms aggregation (groupements par catégorie, marchand, etc.)
                elif "buckets" in agg_data:
                    buckets = agg_data["buckets"]
                    total_buckets = len(buckets)
                    displayed_buckets = buckets[:15]  # Top 15

                    formatted_lines.append(f"\n🏷️  {agg_name} ({total_buckets} groupes au total):")

                    for idx, bucket in enumerate(displayed_buckets, 1):
                        key = bucket.get("key", "Unknown")
                        doc_count = bucket.get("doc_count", 0)

                        line = f"   {idx}. {key}: {doc_count} transactions"

                        # Sous-agrégations (montants, moyennes, etc.)
                        sub_agg_parts = []
                        for sub_agg_name, sub_agg_data in bucket.items():
                            if sub_agg_name not in ["key", "doc_count", "key_as_string"]:
                                if isinstance(sub_agg_data, dict) and "value" in sub_agg_data:
                                    value = sub_agg_data['value']
                                    if value is not None:
                                        sub_agg_parts.append(f"{sub_agg_name}: {value:.2f}€")

                        if sub_agg_parts:
                            line += f" | {' | '.join(sub_agg_parts)}"

                        formatted_lines.append(line)

                    if total_buckets > 15:
                        formatted_lines.append(f"   ... et {total_buckets - 15} autres groupes")

                # Date histogram aggregation (tendances temporelles)
                elif agg_data.get("buckets") and len(agg_data.get("buckets", [])) > 0 and "key_as_string" in agg_data["buckets"][0]:
                    formatted_lines.append(f"\n📅 {agg_name} (Évolution temporelle):")
                    buckets = agg_data.get("buckets", [])

                    for bucket in buckets[:12]:  # Max 12 périodes
                        period = bucket.get("key_as_string", bucket.get("key", "Unknown"))
                        doc_count = bucket.get("doc_count", 0)

                        line = f"   • {period}: {doc_count} transactions"

                        # Sous-agrégations
                        for sub_agg_name, sub_agg_data in bucket.items():
                            if sub_agg_name not in ["key", "doc_count", "key_as_string"]:
                                if isinstance(sub_agg_data, dict) and "value" in sub_agg_data:
                                    value = sub_agg_data['value']
                                    if value is not None:
                                        line += f" | {sub_agg_name}: {value:.2f}€"

                        formatted_lines.append(line)

        formatted_lines.append(f"\n💡 IMPORTANT: Ces statistiques couvrent TOUS les résultats, pas seulement les exemples de transactions listés ci-dessous.")

        return "\n".join(formatted_lines)

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
            # search_service retourne les transactions avec les champs directement
            # (pas de wrapper _source comme Elasticsearch brut)

            # Extraire les informations clés
            date = transaction.get("date", "Date inconnue")
            amount = transaction.get("amount", 0)
            merchant = transaction.get("merchant_name", "Marchand inconnu")
            category = transaction.get("category_name", "")
            description = transaction.get("primary_description", "")

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
