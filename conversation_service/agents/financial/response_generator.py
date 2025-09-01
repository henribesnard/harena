"""
Agent de génération de réponses contextualisées Phase 5
Génère des réponses naturelles avec insights et suggestions actionnables
"""
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

from conversation_service.agents.base.base_agent import BaseAgent
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.models.responses.conversation_responses import (
    ResponseContent, ResponseQuality, ResponseGenerationMetrics, 
    Insight, Suggestion, StructuredData
)
from conversation_service.models.contracts.search_service import SearchResponse
from conversation_service.prompts.templates.response_templates import get_response_template

logger = logging.getLogger(__name__)


class ResponseGeneratorAgent(BaseAgent):
    """Agent spécialisé pour la génération de réponses contextualisées"""
    
    def __init__(self, client: DeepSeekClient):
        super().__init__(
            name="response_generator",
            deepseek_client=client,
            cache_manager=None  # Peut être ajouté plus tard
        )
        self.agent_name = "response_generator"
        
        # Templates de base par intention
        self.response_templates = {
            "SEARCH_BY_MERCHANT": self._get_merchant_template,
            "SPENDING_ANALYSIS": self._get_spending_analysis_template,
            "BALANCE_INQUIRY": self._get_balance_template,
            "TRANSACTION_SEARCH": self._get_transaction_template,
            "CATEGORY_ANALYSIS": self._get_category_template,
            "BUDGET_INQUIRY": self._get_budget_template
        }
    
    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Implémentation de la méthode abstraite execute de BaseAgent
        """
        # Pour l'instant, délégation vers generate_response
        if isinstance(input_data, dict):
            return await self.generate_response(
                user_message=input_data.get("user_message", ""),
                intent=input_data.get("intent", {}),
                entities=input_data.get("entities", {}),
                search_results=input_data.get("search_results"),
                user_context=input_data.get("user_context"),
                request_id=input_data.get("request_id")
            )
        else:
            raise ValueError("input_data doit être un dictionnaire")
    
    async def generate_response(
        self,
        user_message: str,
        intent: Dict[str, Any],
        entities: Dict[str, Any],
        search_results: Optional[SearchResponse],
        user_context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Tuple[ResponseContent, ResponseQuality, ResponseGenerationMetrics]:
        """Génère une réponse complète avec insights et suggestions"""
        
        start_time = time.time()
        
        try:
            logger.info(f"[{request_id}] 🤖 Génération réponse pour intent: {intent.get('intent_type')}")
            
            # 1. Analyser les données disponibles
            analysis_data = self._analyze_search_results(search_results, intent, entities)
            
            # 2. Générer le contenu principal
            response_content = await self._generate_main_response(
                user_message, intent, entities, analysis_data, user_context, request_id
            )
            
            # 3. Générer insights automatiques
            insights = self._generate_insights(analysis_data, intent, entities)
            response_content.insights = insights
            
            # 4. Générer suggestions actionnables
            suggestions = self._generate_suggestions(analysis_data, intent, entities, user_context)
            response_content.suggestions = suggestions
            
            # 5. Générer actions suivantes
            next_actions = self._generate_next_actions(intent, entities, analysis_data)
            response_content.next_actions = next_actions
            
            # 6. Évaluer qualité
            quality = self._evaluate_response_quality(response_content, intent, search_results)
            
            # 7. Métriques génération
            generation_time_ms = int((time.time() - start_time) * 1000)
            metrics = ResponseGenerationMetrics(
                generation_time_ms=generation_time_ms,
                tokens_response=len(response_content.message.split()),
                quality_score=quality.relevance_score,
                insights_generated=len(insights),
                suggestions_generated=len(suggestions),
                context_items_used=len(user_context) if user_context else 0,
                personalization_applied=user_context is not None,
                template_used=intent.get('intent_type')
            )
            
            logger.info(
                f"[{request_id}] ✅ Réponse générée: {len(response_content.message)} chars, "
                f"{len(insights)} insights, {len(suggestions)} suggestions"
            )
            
            return response_content, quality, metrics
            
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Erreur génération réponse: {str(e)}")
            
            # Réponse de fallback
            fallback_content = self._create_fallback_response(user_message, intent, entities)
            fallback_quality = ResponseQuality(
                relevance_score=0.3,
                completeness="minimal",
                actionability="low",
                personalization_level="none",
                tone="professional"
            )
            fallback_metrics = ResponseGenerationMetrics(
                generation_time_ms=int((time.time() - start_time) * 1000),
                quality_score=0.3
            )
            
            return fallback_content, fallback_quality, fallback_metrics
    
    def _analyze_search_results(
        self, 
        search_results: Optional[SearchResponse], 
        intent: Dict[str, Any], 
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse les résultats de recherche pour extraction de données"""
        
        if not search_results or not search_results.hits:
            return {
                "has_results": False,
                "total_hits": 0,
                "analysis_type": intent.get('intent_type'),
                "primary_entity": self._extract_primary_entity(entities)
            }
        
        analysis = {
            "has_results": True,
            "total_hits": search_results.total_hits,
            "returned_hits": len(search_results.hits),
            "analysis_type": intent.get('intent_type'),
            "primary_entity": self._extract_primary_entity(entities)
        }
        
        # Analyse des agrégations si disponibles
        if search_results.aggregations:
            analysis.update(self._analyze_aggregations(search_results.aggregations))
        
        # Analyse des transactions individuelles
        if search_results.hits:
            analysis.update(self._analyze_transactions(search_results.hits))
        
        return analysis
    
    def _extract_primary_entity(self, entities: Dict[str, Any]) -> Optional[str]:
        """Extrait l'entité principale de la requête"""
        
        if entities.get("merchants"):
            merchants = entities["merchants"]
            if isinstance(merchants, list) and len(merchants) > 0:
                return merchants[0] if isinstance(merchants[0], str) else merchants[0].get("name")
        
        if entities.get("categories"):
            categories = entities["categories"]
            if isinstance(categories, list) and len(categories) > 0:
                return categories[0]
        
        return None
    
    def _analyze_aggregations(self, aggregations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les agrégations pour extraire métriques clés"""
        
        analysis = {}
        
        # Analyse montant total
        if "total_spent" in aggregations:
            total_spent = aggregations["total_spent"]
            if isinstance(total_spent, dict) and "value" in total_spent:
                analysis["total_amount"] = abs(total_spent["value"])
        
        # Analyse par marchand
        if "merchant_analysis" in aggregations:
            merchant_agg = aggregations["merchant_analysis"]
            if "buckets" in merchant_agg:
                merchants_data = []
                for bucket in merchant_agg["buckets"]:
                    merchants_data.append({
                        "name": bucket.get("key"),
                        "amount": abs(bucket.get("total_spent", {}).get("value", 0)),
                        "count": bucket.get("transaction_count", {}).get("value", 0)
                    })
                analysis["merchants_breakdown"] = merchants_data
        
        # Analyse par catégorie
        if "category_analysis" in aggregations:
            category_agg = aggregations["category_analysis"]
            if "buckets" in category_agg:
                categories_data = []
                for bucket in category_agg["buckets"]:
                    categories_data.append({
                        "name": bucket.get("key"),
                        "amount": abs(bucket.get("total_spent", {}).get("value", 0)),
                        "percentage": bucket.get("percentage", 0)
                    })
                analysis["categories_breakdown"] = categories_data
        
        return analysis
    
    def _analyze_transactions(self, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse les transactions individuelles"""
        
        amounts = []
        merchants = set()
        dates = []
        transactions_detail = []  # Pour passer les détails des transactions
        
        for hit in hits:
            # SearchHit has a .source attribute, not a .get() method
            source = hit.source if hasattr(hit, 'source') else hit.get("_source", {})
            
            if "amount" in source:
                amounts.append(abs(source["amount"]))
            
            if "merchant_name" in source and source["merchant_name"] is not None:
                merchants.add(source["merchant_name"])
            
            if "date" in source:
                dates.append(source["date"])
            
            # Ajouter les détails de la transaction pour la réponse
            transaction_detail = {
                "amount": abs(source.get("amount", 0)),
                "merchant": source.get("merchant_name", "N/A"),
                "date": source.get("date", "N/A"),
                "description": source.get("primary_description", ""),
                "category": source.get("category_name", "")
            }
            transactions_detail.append(transaction_detail)
        
        analysis = {}
        
        if amounts:
            analysis["transaction_count"] = len(amounts)
            analysis["total_amount"] = sum(amounts)  # AJOUT DU MONTANT TOTAL MANQUANT
            analysis["average_amount"] = analysis["total_amount"] / len(amounts)
            analysis["min_amount"] = min(amounts)
            analysis["max_amount"] = max(amounts)
        
        if merchants:
            analysis["unique_merchants"] = len(merchants)
            analysis["merchant_list"] = list(merchants)
        
        # Ajouter les détails des transactions (limité aux 10 premières pour éviter la surcharge)
        if transactions_detail:
            analysis["transactions_detail"] = transactions_detail[:10]  
        
        return analysis
    
    async def _generate_main_response(
        self,
        user_message: str,
        intent: Dict[str, Any],
        entities: Dict[str, Any],
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]],
        request_id: Optional[str]
    ) -> ResponseContent:
        """Génère le message principal de réponse"""
        
        intent_type = intent.get('intent_type')
        
        # Utiliser les templates contextualisés
        prompt = get_response_template(
            intent_type=intent_type,
            user_message=user_message,
            entities=entities,
            analysis_data=analysis_data,
            user_context=user_context,
            use_personalization=user_context is not None
        )
        
        try:
            chat_response = await self.deepseek_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            response_text = chat_response["choices"][0]["message"]["content"]
            
            # Créer données structurées
            structured_data = self._create_structured_data(analysis_data, entities)
            
            return ResponseContent(
                message=response_text.strip(),
                structured_data=structured_data
            )
            
        except Exception as e:
            logger.error(f"[{request_id}] Erreur génération message principal: {str(e)}")
            return ResponseContent(
                message=self._create_fallback_message(intent, entities, analysis_data),
                structured_data=self._create_structured_data(analysis_data, entities)
            )
    
    def _create_structured_data(self, analysis_data: Dict[str, Any], entities: Dict[str, Any]) -> StructuredData:
        """Crée les données structurées à partir de l'analyse"""
        
        data = StructuredData(
            analysis_type=analysis_data.get("analysis_type"),
            primary_entity=analysis_data.get("primary_entity")
        )
        
        if analysis_data.get("total_amount"):
            data.total_amount = analysis_data["total_amount"]
        
        if analysis_data.get("transaction_count"):
            data.transaction_count = analysis_data["transaction_count"]
        
        if analysis_data.get("average_amount"):
            data.average_amount = analysis_data["average_amount"]
        
        # Période d'analyse
        if entities.get("dates") and isinstance(entities["dates"], dict):
            dates_info = entities["dates"]
            if "normalized" in dates_info:
                normalized = dates_info["normalized"]
                if isinstance(normalized, dict):
                    data.period_start = normalized.get("gte")
                    data.period_end = normalized.get("lte")
                    if data.period_start and data.period_end:
                        data.period = f"{data.period_start} to {data.period_end}"
        
        return data
    
    def _generate_insights(
        self, 
        analysis_data: Dict[str, Any], 
        intent: Dict[str, Any], 
        entities: Dict[str, Any]
    ) -> List[Insight]:
        """Génère des insights automatiques basés sur l'analyse"""
        
        insights = []
        
        if not analysis_data.get("has_results"):
            insights.append(Insight(
                type="pattern",
                title="Aucune donnée trouvée",
                description="Aucune transaction correspondante trouvée pour cette période ou ces critères",
                severity="info",
                confidence=0.9
            ))
            return insights
        
        # Insight sur le volume de transactions
        total_hits = analysis_data.get("total_hits", 0)
        if total_hits > 20:
            insights.append(Insight(
                type="pattern",
                title="Volume élevé de transactions",
                description=f"Vous avez {total_hits} transactions correspondant à vos critères",
                severity="info",
                confidence=0.8
            ))
        elif total_hits < 3:
            insights.append(Insight(
                type="pattern",
                title="Peu de transactions",
                description=f"Seulement {total_hits} transactions trouvées",
                severity="neutral",
                confidence=0.8
            ))
        
        # Insight sur les montants
        if analysis_data.get("average_amount"):
            avg_amount = analysis_data["average_amount"]
            if avg_amount > 100:
                insights.append(Insight(
                    type="spending",
                    title="Montant moyen élevé",
                    description=f"Vos transactions ont un montant moyen de {avg_amount:.2f}€",
                    severity="info",
                    confidence=0.7
                ))
        
        # Insight sur la diversification des marchands
        unique_merchants = analysis_data.get("unique_merchants", 0)
        if unique_merchants > 10:
            insights.append(Insight(
                type="pattern",
                title="Diversification élevée",
                description=f"Vous avez des transactions chez {unique_merchants} marchands différents",
                severity="positive",
                confidence=0.7
            ))
        elif unique_merchants == 1:
            merchant_name = analysis_data.get("primary_entity", "ce marchand")
            insights.append(Insight(
                type="pattern",
                title="Concentration sur un marchand",
                description=f"Toutes vos transactions sont chez {merchant_name}",
                severity="neutral",
                confidence=0.8
            ))
        
        return insights
    
    def _generate_suggestions(
        self, 
        analysis_data: Dict[str, Any], 
        intent: Dict[str, Any], 
        entities: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> List[Suggestion]:
        """Génère des suggestions actionnables"""
        
        suggestions = []
        
        if not analysis_data.get("has_results"):
            suggestions.append(Suggestion(
                type="action",
                title="Élargir la recherche",
                description="Essayez d'élargir vos critères de recherche (période, montant, etc.)",
                priority="medium"
            ))
            return suggestions
        
        # Suggestion budget si montant élevé
        total_amount = analysis_data.get("total_amount")
        if total_amount and total_amount > 500:
            budget_amount = int(total_amount * 1.1)  # 10% de marge
            suggestions.append(Suggestion(
                type="budget",
                title="Définir un budget",
                description=f"Avec {total_amount:.2f}€ dépensés, considérez un budget de {budget_amount}€",
                action=f"Créer un budget de {budget_amount}€",
                priority="medium"
            ))
        
        # Suggestion analyse si beaucoup de transactions
        total_hits = analysis_data.get("total_hits", 0)
        if total_hits > 15:
            suggestions.append(Suggestion(
                type="optimization",
                title="Analyser les patterns",
                description="Avec de nombreuses transactions, une analyse détaillée pourrait révéler des opportunités d'économie",
                action="Voir l'analyse détaillée des dépenses",
                priority="low"
            ))
        
        # Suggestion catégorisation
        if analysis_data.get("unique_merchants", 0) > 5:
            suggestions.append(Suggestion(
                type="action",
                title="Analyser par catégorie",
                description="Analysez vos dépenses par catégorie pour mieux comprendre vos habitudes",
                action="Voir la répartition par catégorie",
                priority="low"
            ))
        
        return suggestions
    
    def _generate_next_actions(
        self, 
        intent: Dict[str, Any], 
        entities: Dict[str, Any], 
        analysis_data: Dict[str, Any]
    ) -> List[str]:
        """Génère la liste des actions suivantes possibles"""
        
        actions = []
        intent_type = intent.get('intent_type')
        
        # Actions communes
        actions.append("Voir le détail des transactions")
        
        # Actions spécifiques par intention
        if intent_type == "SEARCH_BY_MERCHANT":
            primary_entity = analysis_data.get("primary_entity")
            if primary_entity:
                actions.append(f"Comparer avec d'autres marchands")
                actions.append(f"Analyser l'évolution chez {primary_entity}")
        
        elif intent_type == "SPENDING_ANALYSIS":
            actions.append("Voir la répartition par catégorie")
            actions.append("Comparer avec les mois précédents")
        
        elif intent_type == "CATEGORY_ANALYSIS":
            actions.append("Voir les autres catégories")
            actions.append("Analyser les tendances par catégorie")
        
        # Actions conditionnelles
        if analysis_data.get("total_hits", 0) > 10:
            actions.append("Filtrer par montant")
            actions.append("Grouper par période")
        
        return actions[:5]  # Maximum 5 actions
    
    def _evaluate_response_quality(
        self, 
        response_content: ResponseContent, 
        intent: Dict[str, Any], 
        search_results: Optional[SearchResponse]
    ) -> ResponseQuality:
        """Évalue la qualité de la réponse générée"""
        
        # Score de pertinence basé sur la présence de données
        relevance_score = 0.5  # Base
        
        if search_results and search_results.hits:
            relevance_score += 0.2  # Données disponibles
        
        if response_content.structured_data and response_content.structured_data.total_amount:
            relevance_score += 0.1  # Montants structurés
        
        if len(response_content.insights) > 0:
            relevance_score += 0.1  # Insights présents
        
        if len(response_content.suggestions) > 0:
            relevance_score += 0.1  # Suggestions présentes
        
        relevance_score = min(1.0, relevance_score)
        
        # Complétude
        completeness = "minimal"
        if search_results and len(response_content.insights) > 0:
            completeness = "partial"
        if search_results and len(response_content.insights) > 1 and len(response_content.suggestions) > 0:
            completeness = "full"
        
        # Actionnabilité
        actionability = "none"
        if len(response_content.suggestions) > 0:
            actionability = "low"
        if len(response_content.suggestions) >= 2:
            actionability = "medium"
        if len(response_content.suggestions) >= 3 or len(response_content.next_actions) > 3:
            actionability = "high"
        
        return ResponseQuality(
            relevance_score=relevance_score,
            completeness=completeness,
            actionability=actionability,
            personalization_level="basic",
            tone="professional_friendly"
        )
    
    def _create_fallback_response(
        self, 
        user_message: str, 
        intent: Dict[str, Any], 
        entities: Dict[str, Any]
    ) -> ResponseContent:
        """Crée une réponse de fallback en cas d'erreur"""
        
        intent_type = intent.get('intent_type', 'unknown')
        
        fallback_messages = {
            "SEARCH_BY_MERCHANT": "Je n'ai pas pu récupérer vos données de transactions pour ce marchand actuellement.",
            "SPENDING_ANALYSIS": "Je n'ai pas pu analyser vos dépenses pour le moment.",
            "BALANCE_INQUIRY": "Je n'ai pas pu récupérer votre solde actuellement.",
            "CATEGORY_ANALYSIS": "Je n'ai pas pu analyser cette catégorie de dépenses.",
            "TRANSACTION_SEARCH": "Je n'ai pas pu rechercher ces transactions."
        }
        
        message = fallback_messages.get(intent_type, "Je n'ai pas pu traiter votre demande actuellement.")
        message += " Veuillez réessayer dans quelques instants."
        
        return ResponseContent(
            message=message,
            suggestions=[
                Suggestion(
                    type="action",
                    title="Réessayer",
                    description="Réessayez votre demande dans quelques instants",
                    priority="medium"
                )
            ]
        )
    
    def _create_fallback_message(
        self, 
        intent: Dict[str, Any], 
        entities: Dict[str, Any], 
        analysis_data: Dict[str, Any]
    ) -> str:
        """Crée un message de fallback simple"""
        
        if not analysis_data.get("has_results"):
            return "Aucune transaction trouvée pour vos critères de recherche."
        
        total_hits = analysis_data.get("total_hits", 0)
        total_amount = analysis_data.get("total_amount")
        
        if total_amount:
            return f"J'ai trouvé {total_hits} transactions pour un montant total de {total_amount:.2f}€."
        else:
            return f"J'ai trouvé {total_hits} transactions correspondant à votre recherche."
    
    # Templates de réponse par intention
    def _get_merchant_template(self, message: str, entities: Dict, analysis: Dict, context: Dict) -> str:
        return f"""Tu es un assistant financier. L'utilisateur demande: "{message}"

Données analysées:
- Marchand principal: {analysis.get('primary_entity', 'Non spécifié')}
- Nombre de transactions: {analysis.get('total_hits', 0)}
- Montant total: {analysis.get('total_amount', 0)}€
- Montant moyen: {analysis.get('average_amount', 0):.2f}€

Génère une réponse naturelle, informative et engageante qui:
1. Répond directement à la question
2. Présente les chiffres clés de manière claire
3. Utilise un ton professionnel mais amical
4. Reste concis (max 200 mots)

Réponse:"""

    def _get_spending_analysis_template(self, message: str, entities: Dict, analysis: Dict, context: Dict) -> str:
        return f"""Tu es un assistant financier. L'utilisateur demande: "{message}"

Données d'analyse:
- Période analysée: {entities.get('dates', 'Non spécifiée')}
- Total dépensé: {analysis.get('total_amount', 0)}€
- Nombre de transactions: {analysis.get('total_hits', 0)}
- Marchands différents: {analysis.get('unique_merchants', 0)}

Génère une analyse claire et actionnable qui présente les informations principales de manière structurée.

Réponse:"""

    def _get_balance_template(self, message: str, entities: Dict, analysis: Dict, context: Dict) -> str:
        return f"""Tu es un assistant financier. L'utilisateur demande son solde.

Génère une réponse sur le solde disponible en restant informatif et rassurant.

Réponse:"""

    def _get_transaction_template(self, message: str, entities: Dict, analysis: Dict, context: Dict) -> str:
        return f"""Tu es un assistant financier. L'utilisateur recherche des transactions spécifiques.

Résultats trouvés: {analysis.get('total_hits', 0)} transactions
Montant total: {analysis.get('total_amount', 0)}€

Présente les résultats de recherche de manière claire et utile.

Réponse:"""

    def _get_category_template(self, message: str, entities: Dict, analysis: Dict, context: Dict) -> str:
        return f"""Tu es un assistant financier. L'utilisateur analyse une catégorie de dépenses.

Données de catégorie:
- Transactions: {analysis.get('total_hits', 0)}
- Montant: {analysis.get('total_amount', 0)}€

Présente l'analyse de cette catégorie avec des insights utiles.

Réponse:"""

    def _get_budget_template(self, message: str, entities: Dict, analysis: Dict, context: Dict) -> str:
        return f"""Tu es un assistant financier. L'utilisateur s'intéresse à son budget.

Génère des conseils budgétaires personnalisés basés sur ses dépenses.

Réponse:"""

    def _get_generic_template(self, message: str, intent: Dict, entities: Dict, analysis: Dict) -> str:
        return f"""Tu es un assistant financier. L'utilisateur demande: "{message}"

Intent détecté: {intent.get('intent_type', 'Inconnu')}
Données disponibles: {analysis.get('has_results', False)}

Génère une réponse appropriée et utile.

Réponse:"""