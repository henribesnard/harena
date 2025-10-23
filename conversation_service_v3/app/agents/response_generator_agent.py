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

IMPORTANT - Contexte conversationnel:
- Tu as accès à l'historique de la conversation précédente
- Utilise ce contexte pour des réponses cohérentes et personnalisées
- Si l'utilisateur fait référence à une question précédente, utilise cet historique
- Reste naturel et conversationnel en tenant compte du contexte

Ton rôle est de créer une réponse claire, précise et utile basée sur:
1. Les agrégations Elasticsearch (totaux, moyennes, statistiques) - SOURCE DE VÉRITÉ
2. Un résumé des résultats de recherche
3. Les premières transactions détaillées (exemples illustratifs)
4. L'historique de conversation (si disponible)

Règles de réponse:
- Commence TOUJOURS par les chiffres des AGRÉGATIONS
- Utilise "vos/votre" (jamais "utilisateur 123")
- Mentionne les insights importants des agrégations
- Inclus des exemples de transactions SI pertinent
- Sois naturel et conversationnel
- Si aucun résultat, explique pourquoi et propose des alternatives
- Utilise le contexte conversationnel pour des réponses plus pertinentes

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
            ("user", """**Historique de conversation:**
{conversation_history}

**Question utilisateur actuelle:** {user_message}

**Agrégations Elasticsearch:**
{aggregations}

**Résumé des résultats:**
- Nombre total de résultats: {total_results}
- Nombre de transactions retournées: {transactions_count}

**Transactions (premières {transactions_count}):**
{transactions}

Génère une réponse complète et utile en tenant compte du contexte de la conversation.""")
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
        original_query_analysis: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AgentResponse:
        """
        Génère une réponse finale basée sur les résultats de recherche

        OPTIMISATION: Ne met en contexte que:
        - Le résumé de la recherche (total, etc.)
        - Les agrégations complètes
        - Les N premières transactions (défini par MAX_TRANSACTIONS_IN_CONTEXT)
        - L'historique de conversation récent (si disponible)

        Args:
            user_message: Message original de l'utilisateur
            search_results: Résultats Elasticsearch (hits + agrégations)
            original_query_analysis: Analyse originale (pour contexte)
            conversation_history: Historique de conversation (format OpenAI chat)

        Returns:
            AgentResponse contenant ConversationResponse
        """
        try:
            logger.info("Generating response for query")
            logger.info("DEBUG: Line 139 - before import settings")

            from ..config.settings import settings

            logger.info("DEBUG: Line 142 - after import settings")
            # DEBUG: Log aggregations
            logger.info("DEBUG: About to log aggregations")
            logger.info(f"DEBUG: Aggregations type: {type(search_results.aggregations)}")

            # Préparer les agrégations pour le LLM (toujours complètes)
            try:
                logger.info("Step A: Starting aggregation formatting")
                aggs_summary = self._format_aggregations(search_results.aggregations)
                logger.info(f"Step A: Aggregations formatted successfully - length: {len(aggs_summary)}")
            except Exception as agg_error:
                logger.error(f"Step A ERROR: Error formatting aggregations: {agg_error}", exc_info=True)
                aggs_summary = "Agrégations non disponibles (erreur de formatting)"

            # Limiter le nombre de transactions dans le contexte
            logger.info("Step B: Limiting transactions")
            max_transactions = min(settings.MAX_TRANSACTIONS_IN_CONTEXT, len(search_results.hits))
            limited_transactions = search_results.hits[:max_transactions]

            # Step B.5: Filtrer les champs pour réduire le contexte (~50% de tokens économisés)
            # Ne garder que les champs essentiels pour le LLM
            essential_fields = [
                'amount', 'currency_code', 'transaction_type', 'date',
                'primary_description', 'merchant_name', 'category_name', 'operation_type'
            ]

            filtered_transactions = []
            for transaction in limited_transactions:
                filtered = {field: transaction.get(field) for field in essential_fields if field in transaction}
                filtered_transactions.append(filtered)

            limited_transactions = filtered_transactions

            # DEBUG: Log pour identifier le problème
            logger.info(f"Transactions received: {len(search_results.hits)}, Limited to: {len(limited_transactions)}")
            if len(limited_transactions) > 0:
                logger.info(f"Filtered transaction keys: {list(limited_transactions[0].keys())} (was 16, now {len(limited_transactions[0])})")
            else:
                logger.warning("No transactions in search_results.hits")

            if len(search_results.hits) > max_transactions:
                logger.debug(
                    f"Context limitation: {len(search_results.hits)} → {max_transactions} transactions"
                )

            # Préparer les transactions pour le LLM (limitées)
            logger.debug("Step C: Formatting transactions")
            transactions_text = self._format_transactions(limited_transactions)
            logger.debug(f"Step C: Transactions formatted - length: {len(transactions_text)}")

            # DEBUG: Log formatted transactions
            logger.debug(f"Transactions text preview: {transactions_text[:min(200, len(transactions_text))]}")

            # S'assurer qu'aucune valeur n'est None (LangChain ne gère pas bien None)
            total_results_safe = search_results.total if search_results.total is not None else 0

            # Formater l'historique de conversation
            logger.debug("Step D1: Formatting conversation history")
            history_text = self._format_conversation_history(conversation_history)
            logger.debug(f"Step D1: History formatted - length: {len(history_text)}")

            logger.debug("Step D: Preparing chain parameters")
            logger.debug(f"Chain params: total_results={total_results_safe}, transactions_count={len(limited_transactions)}")

            # Invoquer le LLM
            logger.debug("Step E: Invoking LLM chain")
            chain_params = {
                "user_message": user_message,
                "conversation_history": history_text,
                "aggregations": aggs_summary,
                "total_results": total_results_safe,
                "transactions_count": len(limited_transactions),
                "transactions": transactions_text
            }
            logger.debug(f"Chain params types: user_message={type(user_message)}, history={type(history_text)}, aggs={type(aggs_summary)}, total={type(total_results_safe)}, count={type(len(limited_transactions))}, trans={type(transactions_text)}")

            result = await self.chain.ainvoke(chain_params)
            logger.debug("Step E: LLM chain invoked successfully")

            response_text = result.content
            logger.debug(f"Step F: Response text extracted - length: {len(response_text)}")

            # Créer la réponse de conversation
            conversation_response = ConversationResponse(
                success=True,
                message=response_text,
                search_results=search_results,
                aggregations_summary=aggs_summary,
                metadata={
                    "total_results": total_results_safe,
                    "response_length": len(response_text),
                    "took_ms": search_results.took_ms if search_results.took_ms is not None else 0
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
                    "total_results": total_results_safe
                }
            )

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return AgentResponse(
                success=False,
                data=None,
                agent_role=AgentRole.RESPONSE_GENERATOR,
                error=str(e)
            )

    async def generate_response_stream(
        self,
        user_message: str,
        search_results: SearchResults,
        original_query_analysis: Optional[Dict[str, Any]] = None
    ):
        """
        Génère une réponse en mode streaming (yield chunks)

        Args:
            user_message: Message original de l'utilisateur
            search_results: Résultats Elasticsearch (hits + agrégations)
            original_query_analysis: Analyse originale (pour contexte)

        Yields:
            Chunks de texte au fur et à mesure de la génération
        """
        try:
            logger.info(f"Generating streaming response for query: {user_message[:100]}")

            from ..config.settings import settings

            # Préparer les agrégations pour le LLM
            aggs_summary = self._format_aggregations(search_results.aggregations)

            # Limiter le nombre de transactions dans le contexte
            max_transactions = min(settings.MAX_TRANSACTIONS_IN_CONTEXT, len(search_results.hits))
            limited_transactions = search_results.hits[:max_transactions]

            # Filtrer les champs pour réduire le contexte (~50% de tokens économisés)
            # Ne garder que les champs essentiels pour le LLM
            essential_fields = [
                'amount', 'currency_code', 'transaction_type', 'date',
                'primary_description', 'merchant_name', 'category_name', 'operation_type'
            ]

            filtered_transactions = []
            for transaction in limited_transactions:
                filtered = {field: transaction.get(field) for field in essential_fields if field in transaction}
                filtered_transactions.append(filtered)

            limited_transactions = filtered_transactions

            logger.info(f"[STREAM] Transactions: {len(search_results.hits)} → {len(limited_transactions)} (filtered to {len(essential_fields)} fields)")

            # Préparer les transactions pour le LLM
            transactions_text = self._format_transactions(limited_transactions)

            # Stream la réponse du LLM
            async for chunk in self.chain.astream({
                "user_message": user_message,
                "conversation_history": "",  # Pas d'historique en mode streaming (pour compatibilité prompt)
                "aggregations": aggs_summary,
                "total_results": search_results.total,
                "transactions_count": len(limited_transactions),
                "transactions": transactions_text
            }):
                # Chaque chunk contient le contenu généré
                if hasattr(chunk, 'content'):
                    yield chunk.content

            logger.info("Streaming response completed")

        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Erreur lors de la génération de la réponse: {str(e)}"

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

        logger.debug(f"Formatting {len(aggregations)} aggregations")

        for agg_name, agg_data in aggregations.items():
            try:
                logger.debug(f"Processing aggregation: {agg_name}, type: {type(agg_data)}")
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

                        # Gérer les valeurs None (quand 0 résultats)
                        sum_val = agg_data.get('sum') or 0
                        avg_val = agg_data.get('avg') or 0
                        min_val = agg_data.get('min') or 0
                        max_val = agg_data.get('max') or 0

                        formatted_lines.append(f"   • Total: {sum_val:.2f}€")
                        formatted_lines.append(f"   • Moyenne: {avg_val:.2f}€")
                        formatted_lines.append(f"   • Min: {min_val:.2f}€")
                        formatted_lines.append(f"   • Max: {max_val:.2f}€")

                    # Filter aggregation (ex: debit_stats, credit_stats)
                    elif "doc_count" in agg_data:
                        doc_count = agg_data.get("doc_count", 0)
                        formatted_lines.append(f"\n📊 {agg_name}:")
                        formatted_lines.append(f"   • Nombre: {doc_count} transactions")

                        # Parser les sous-agrégations
                        for sub_agg_name, sub_agg_data in agg_data.items():
                            if sub_agg_name != "doc_count" and isinstance(sub_agg_data, dict):
                                if "value" in sub_agg_data:
                                    value = sub_agg_data.get("value")
                                    # Gérer None
                                    if value is not None:
                                        formatted_lines.append(f"   • {sub_agg_name}: {value:.2f}€")
                                    else:
                                        formatted_lines.append(f"   • {sub_agg_name}: 0.00€")

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

            except Exception as agg_error:
                logger.error(f"Error formatting aggregation '{agg_name}': {agg_error}", exc_info=True)
                formatted_lines.append(f"\n⚠️ {agg_name}: Erreur de formatting")

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
            amount = transaction.get("amount") or 0  # Gérer None
            merchant = transaction.get("merchant_name", "Marchand inconnu")
            category = transaction.get("category_name", "")
            description = transaction.get("primary_description", "")

            # Formater le montant (gérer None explicitement)
            amount_safe = abs(amount) if amount is not None else 0
            amount_str = f"{amount_safe:.2f} €"
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

    def _format_conversation_history(
        self,
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> str:
        """
        Formate l'historique de conversation pour le LLM

        Args:
            conversation_history: Liste de messages {"role": "user/assistant", "content": "..."}

        Returns:
            String formatée de l'historique
        """
        if not conversation_history or len(conversation_history) == 0:
            return "(Aucun historique de conversation - première interaction)"

        formatted_lines = []
        for idx, message in enumerate(conversation_history, 1):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            # Mapper les rôles pour plus de clarté
            role_display = "👤 Utilisateur" if role == "user" else "🤖 Assistant"

            formatted_lines.append(f"{role_display}: {content}")

        history_count = len(conversation_history)
        return f"({history_count} messages précédents)\n" + "\n".join(formatted_lines)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        return {
            "agent": "response_generator",
            "model": self.llm.model_name,
            **self.stats
        }
