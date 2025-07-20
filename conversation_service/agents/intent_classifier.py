"""
🧠 Agent principal orchestrant la détection d'intention

Agent high-level pour classification intentions avec orchestration
Intent Detection Engine, métriques agent et fallbacks gracieux.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any

from conversation_service.intent_detection.engine import IntentDetectionEngine
from conversation_service.intent_detection.models import IntentResult, IntentType, IntentLevel
from conversation_service.config.settings import settings
from conversation_service.utils.logging import log_intent_detection, log_business_event

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    🤖 Agent principal classification intentions financières
    
    Interface high-level pour classification intentions avec:
    - Orchestration Intent Detection Engine
    - Métriques agent (succès, échecs, temps moyen)
    - Fallbacks gracieux en cas d'erreur
    - Cache local patterns fréquents
    - Suggestions d'actions basées intentions
    """
    
    def __init__(self):
        self.intent_engine: Optional[IntentDetectionEngine] = None
        
        # Configuration agent
        self.confidence_threshold = settings.MIN_CONFIDENCE_THRESHOLD
        self.enable_suggestions = True
        self.enable_response_generation = True
        
        # Métriques agent
        self._agent_metrics = {
            "total_classifications": 0,
            "successful_classifications": 0,
            "failed_classifications": 0,
            "average_confidence": 0.0,
            "average_latency_ms": 0.0,
            "intent_distribution": {},
            "level_usage": {"L0": 0, "L1": 0, "L2": 0, "FALLBACK": 0}
        }
        
        # Cache local patterns agent (différent du cache engine)
        self._agent_cache = {}
        self._agent_cache_max_size = 100
        
        # Patterns suggestions actions par intention
        self._action_suggestions = self._initialize_action_suggestions()
        
        # Messages contextuels par intention
        self._response_templates = self._initialize_response_templates()
        
        logger.info("🤖 Intent Classifier Agent initialisé")
    
    async def initialize(self):
        """Initialisation agent avec Intent Detection Engine"""
        try:
            logger.info("🔧 Initialisation Intent Classifier Agent...")
            
            # Initialisation Intent Detection Engine
            self.intent_engine = IntentDetectionEngine()
            await self.intent_engine.initialize()
            
            logger.info("✅ Intent Classifier Agent prêt")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Intent Classifier Agent: {e}")
            self.intent_engine = None
            return False
    
    async def classify_intent(self, user_query: str, user_id: str = "anonymous") -> IntentResult:
        """
        Classification intention principal avec enrichissements agent
        
        Args:
            user_query: Message utilisateur à classifier
            user_id: ID utilisateur pour personnalisation
            
        Returns:
            IntentResult: Résultat enrichi avec suggestions et contexte
        """
        start_time = time.time()
        self._agent_metrics["total_classifications"] += 1
        
        # Log événement business
        log_business_event(
            event_type="intent_classification_request",
            user_id=user_id,
            metadata={"query_length": len(user_query)}
        )
        
        try:
            # 1. Vérification cache agent local
            cache_result = self._check_agent_cache(user_query, user_id)
            if cache_result:
                log_intent_detection(
                    "agent_cache_hit",
                    user_id=user_id,
                    intent=cache_result.intent_type.value,
                    confidence=cache_result.confidence.score
                )
                return cache_result
            
            # 2. Classification via Intent Detection Engine
            if not self.intent_engine:
                raise RuntimeError("Intent Detection Engine non disponible")
            
            engine_result = await self.intent_engine.detect_intent(user_query, user_id)
            
            # 3. Enrichissement résultat agent
            enriched_result = await self._enrich_result(engine_result, user_query, user_id)
            
            # 4. Mise à jour métriques agent
            processing_time = (time.time() - start_time) * 1000
            self._update_agent_metrics(enriched_result, processing_time, success=True)
            
            # 5. Cache agent si haute confiance
            if enriched_result.confidence.score >= 0.90:
                self._cache_agent_result(user_query, user_id, enriched_result)
            
            # 6. Log événement business succès
            log_business_event(
                event_type="intent_classification_success",
                user_id=user_id,
                intent=enriched_result.intent_type.value,
                success=True,
                value=enriched_result.confidence.score,
                metadata={
                    "level_used": enriched_result.level.value,
                    "processing_time_ms": processing_time
                }
            )
            
            log_intent_detection(
                "agent_classification_success",
                user_id=user_id,
                intent=enriched_result.intent_type.value,
                confidence=enriched_result.confidence.score,
                level=enriched_result.level.value,
                latency_ms=processing_time
            )
            
            return enriched_result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_agent_metrics(None, processing_time, success=False)
            
            # Fallback gracieux
            fallback_result = self._create_fallback_result(user_query, user_id, str(e))
            
            # Log événement business échec
            log_business_event(
                event_type="intent_classification_failure",
                user_id=user_id,
                success=False,
                metadata={"error": str(e), "processing_time_ms": processing_time}
            )
            
            log_intent_detection(
                "agent_classification_error",
                user_id=user_id,
                error=str(e),
                latency_ms=processing_time
            )
            
            logger.warning(f"⚠️ Erreur classification agent: {e}")
            return fallback_result
    
    async def classify_batch(self, requests: List[Any]) -> List[IntentResult]:
        """Classification batch pour optimisation performance"""
        logger.info(f"📦 Classification batch: {len(requests)} requêtes")
        
        # Traitement parallèle avec limitation concurrence
        semaphore = asyncio.Semaphore(5)  # Max 5 parallèles
        
        async def classify_single(request: Any) -> IntentResult:
            async with semaphore:
                return await self.classify_intent(request.message, request.user_id)
        
        tasks = [classify_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Traitement résultats avec gestion exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                fallback = self._create_fallback_result(
                    requests[i].message, 
                    requests[i].user_id, 
                    str(result)
                )
                final_results.append(fallback)
            else:
                final_results.append(result)
        
        logger.info(f"📦 Batch terminé: {len(final_results)} résultats")
        return final_results
    
    def to_chat_response(self, intent_result: IntentResult, request_id: str = None) -> Any:
        """Conversion IntentResult vers ChatResponse API"""
        # Import local pour éviter les références circulaires
        from conversation_service.models.conversation_models import ChatResponse, FinancialIntent, ProcessingMetadata
        
        # Mapping IntentType vers FinancialIntent
        financial_intent = self._map_to_financial_intent(intent_result.intent_type)
        
        # Construction métadonnées processing
        processing_metadata = ProcessingMetadata(
            request_id=request_id or f"agent_{int(time.time() * 1000)}",
            level_used=intent_result.level.value,
            processing_time_ms=intent_result.latency_ms,
            cache_hit=intent_result.from_cache,
            engine_latency_ms=intent_result.latency_ms,
            timestamp=int(time.time())
        )
        
        # Actions suggérées si activées
        suggested_actions = None
        if self.enable_suggestions:
            suggested_actions = self._get_suggested_actions(intent_result.intent_type)
        
        # Message de réponse contextuel
        response_message = None
        if self.enable_response_generation:
            response_message = self._generate_response_message(intent_result)
        
        return ChatResponse(
            intent=financial_intent,
            entities=intent_result.entities or {},
            confidence=intent_result.confidence.score,
            processing_metadata=processing_metadata,
            suggested_actions=suggested_actions,
            response_message=response_message
        )
    
    # ==========================================
    # MÉTHODES PRIVÉES ENRICHISSEMENT
    # ==========================================
    
    async def _enrich_result(self, engine_result: IntentResult, query: str, user_id: str) -> IntentResult:
        """Enrichissement résultat engine avec contexte agent"""
        
        # Copie résultat engine
        enriched = IntentResult(
            intent_type=engine_result.intent_type,
            confidence=engine_result.confidence,
            level=engine_result.level,
            latency_ms=engine_result.latency_ms,
            from_cache=engine_result.from_cache,
            entities=engine_result.entities.copy() if engine_result.entities else {},
            user_id=user_id,
            timestamp=engine_result.timestamp,
            processing_details=engine_result.processing_details.copy() if engine_result.processing_details else {}
        )
        
        # Enrichissement entités agent
        enriched.entities.update(await self._extract_agent_entities(query, engine_result.intent_type))
        
        # Métadonnées agent
        enriched.processing_details["agent_version"] = "1.0.0"
        enriched.processing_details["agent_enrichments"] = True
        
        return enriched
    
    async def _extract_agent_entities(self, query: str, intent_type: IntentType) -> Dict[str, Any]:
        """Extraction entités supplémentaires niveau agent"""
        agent_entities = {}
        
        try:
            query_lower = query.lower()
            
            # Entités urgence/priorité
            if any(word in query_lower for word in ["urgent", "rapidement", "vite", "immédiatement"]):
                agent_entities["urgency"] = "high"
            elif any(word in query_lower for word in ["quand possible", "pas pressé"]):
                agent_entities["urgency"] = "low"
            else:
                agent_entities["urgency"] = "normal"
            
            # Entités sentiment/politesse
            if any(word in query_lower for word in ["s'il vous plaît", "merci", "svp"]):
                agent_entities["politeness"] = "polite"
            
            # Entités spécifiques par intention
            if intent_type == IntentType.TRANSFER:
                # Détection IBAN/RIB basique
                import re
                iban_pattern = r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,28}\b'
                if re.search(iban_pattern, query.upper()):
                    agent_entities["contains_iban"] = True
            
            elif intent_type == IntentType.EXPENSE_ANALYSIS:
                # Détection comparaison temporelle
                if any(word in query_lower for word in ["comparé", "versus", "par rapport"]):
                    agent_entities["comparison_request"] = True
        
        except Exception as e:
            logger.debug(f"⚠️ Erreur extraction entités agent: {e}")
        
        return agent_entities
    
    def _check_agent_cache(self, query: str, user_id: str) -> Optional[IntentResult]:
        """Vérification cache agent local"""
        cache_key = f"{user_id}:{hash(query.lower().strip())}"
        
        cached_entry = self._agent_cache.get(cache_key)
        if cached_entry:
            # Vérification TTL (5 minutes)
            if time.time() - cached_entry["timestamp"] < 300:
                result = cached_entry["result"]
                result.from_cache = True
                return result
            else:
                # Entrée expirée
                del self._agent_cache[cache_key]
        
        return None
    
    def _cache_agent_result(self, query: str, user_id: str, result: IntentResult):
        """Mise en cache résultat agent"""
        cache_key = f"{user_id}:{hash(query.lower().strip())}"
        
        # Éviction LRU si cache plein
        if len(self._agent_cache) >= self._agent_cache_max_size:
            oldest_key = min(self._agent_cache.keys(), 
                           key=lambda k: self._agent_cache[k]["timestamp"])
            del self._agent_cache[oldest_key]
        
        self._agent_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def _create_fallback_result(self, query: str, user_id: str, error: str) -> IntentResult:
        """Création résultat fallback en cas d'erreur"""
        from conversation_service.intent_detection.models import IntentConfidence
        
        # Analyse basique pour fallback intelligent
        fallback_intent = self._analyze_fallback_intent(query)
        
        fallback_confidence = IntentConfidence(
            score=0.5,
            reasoning=f"Fallback agent - Erreur: {error}",
            factors={"fallback": True, "error": error}
        )
        
        return IntentResult(
            intent_type=fallback_intent,
            confidence=fallback_confidence,
            level=IntentLevel.ERROR_FALLBACK,
            latency_ms=0.0,
            from_cache=False,
            entities={"fallback_reason": error},
            user_id=user_id,
            processing_details={
                "agent_fallback": True,
                "original_error": error,
                "fallback_analysis": "basic_keyword_matching"
            }
        )
    
    def _analyze_fallback_intent(self, query: str) -> IntentType:
        """Analyse fallback basique par mots-clés"""
        query_lower = query.lower()
        
        # Mots-clés simples par intention
        fallback_keywords = {
            IntentType.BALANCE_CHECK: ["solde", "compte", "argent", "combien"],
            IntentType.TRANSFER: ["virement", "transférer", "envoyer", "virer"],
            IntentType.EXPENSE_ANALYSIS: ["dépenses", "dépensé", "budget"],
            IntentType.CARD_MANAGEMENT: ["carte", "bloquer", "limites"],
            IntentType.GREETING: ["bonjour", "salut", "hello"],
            IntentType.HELP: ["aide", "comment", "expliquer"]
        }
        
        # Score par intention
        intent_scores = {}
        for intent, keywords in fallback_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Retourne intention avec meilleur score ou UNKNOWN
        if intent_scores:
            return max(intent_scores.keys(), key=lambda k: intent_scores[k])
        else:
            return IntentType.UNKNOWN
    
    # ==========================================
    # SUGGESTIONS ET RÉPONSES
    # ==========================================
    
    def _initialize_action_suggestions(self) -> Dict[IntentType, List[str]]:
        """Initialisation suggestions actions par intention"""
        return {
            IntentType.BALANCE_CHECK: [
                "Consulter solde détaillé",
                "Voir historique récent",
                "Configurer alertes solde",
                "Télécharger relevé"
            ],
            IntentType.EXPENSE_ANALYSIS: [
                "Voir détails par catégorie", 
                "Comparer avec mois précédent",
                "Créer budget personnalisé",
                "Exporter analyse Excel"
            ],
            IntentType.TRANSFER: [
                "Ajouter nouveau bénéficiaire",
                "Programmer virement récurrent",
                "Vérifier limites de virement",
                "Historique virements"
            ],
            IntentType.CARD_MANAGEMENT: [
                "Voir paramètres carte",
                "Modifier limites",
                "Gérer notifications",
                "Commander nouvelle carte"
            ],
            IntentType.BILL_PAYMENT: [
                "Configurer prélèvement auto",
                "Voir factures en attente", 
                "Historique paiements",
                "Ajouter nouveau créancier"
            ],
            IntentType.INVESTMENT_QUERY: [
                "Consulter portefeuille détaillé",
                "Analyser performance",
                "Rebalancer allocation",
                "Simuler investissement"
            ],
            IntentType.HELP: [
                "Voir fonctionnalités disponibles",
                "Guide utilisation",
                "Contacter support",
                "FAQ"
            ]
        }
    
    def _initialize_response_templates(self) -> Dict[IntentType, str]:
        """Initialisation templates réponses contextuelles"""
        return {
            IntentType.BALANCE_CHECK: "Je peux vous aider à consulter votre solde. Voulez-vous voir le détail par compte ?",
            IntentType.EXPENSE_ANALYSIS: "Parfait ! Je vais analyser vos dépenses. Sur quelle période souhaitez-vous l'analyse ?",
            IntentType.TRANSFER: "Je peux vous accompagner pour votre virement. Avez-vous déjà le bénéficiaire enregistré ?",
            IntentType.CARD_MANAGEMENT: "Je peux vous aider avec votre carte bancaire. Quelle action souhaitez-vous effectuer ?",
            IntentType.BILL_PAYMENT: "Pour vos paiements de factures, je peux vous montrer les options disponibles.",
            IntentType.INVESTMENT_QUERY: "Concernant vos investissements, que souhaitez-vous savoir précisément ?",
            IntentType.BUDGET_PLANNING: "Excellente idée de planifier votre budget ! Par où souhaitez-vous commencer ?",
            IntentType.SAVINGS_GOAL: "C'est formidable d'épargner ! Quel est votre objectif d'épargne ?",
            IntentType.GREETING: "Bonjour ! Je suis votre assistant bancaire. Comment puis-je vous aider aujourd'hui ?",
            IntentType.HELP: "Je suis là pour vous aider ! Voici ce que je peux faire pour vous...",
            IntentType.UNKNOWN: "Je n'ai pas bien compris votre demande. Pouvez-vous reformuler ou être plus spécifique ?"
        }
    
    def _get_suggested_actions(self, intent_type: IntentType) -> List[str]:
        """Récupération suggestions actions pour intention"""
        return self._action_suggestions.get(intent_type, [
            "Voir plus d'options",
            "Parler à un conseiller",
            "Retour au menu principal"
        ])
    
    def _generate_response_message(self, intent_result: IntentResult) -> str:
        """Génération message réponse contextuel"""
        base_message = self._response_templates.get(
            intent_result.intent_type,
            "Comment puis-je vous aider avec cette demande ?"
        )
        
        # Personnalisation selon confiance
        if intent_result.confidence.score < 0.7:
            base_message = f"Je pense que vous voulez... {base_message.lower()}"
        elif intent_result.confidence.score > 0.95:
            base_message = f"Parfait ! {base_message}"
        
        # Ajout contexte urgence si détectée
        if intent_result.entities.get("urgency") == "high":
            base_message += " Je vais traiter cela en priorité."
        
        return base_message
    
    def _map_to_financial_intent(self, intent_type: IntentType) -> Any:
        """Mapping IntentType vers FinancialIntent API"""
        # Import local pour éviter les références circulaires
        from conversation_service.models.conversation_models import FinancialIntent
        
        mapping = {
            IntentType.BALANCE_CHECK: FinancialIntent.BALANCE_CHECK,
            IntentType.EXPENSE_ANALYSIS: FinancialIntent.EXPENSE_ANALYSIS,
            IntentType.TRANSFER: FinancialIntent.TRANSFER,
            IntentType.BILL_PAYMENT: FinancialIntent.BILL_PAYMENT,
            IntentType.INVESTMENT_QUERY: FinancialIntent.INVESTMENT_QUERY,
            IntentType.LOAN_INQUIRY: FinancialIntent.LOAN_INQUIRY,
            IntentType.CARD_MANAGEMENT: FinancialIntent.CARD_MANAGEMENT,
            IntentType.TRANSACTION_HISTORY: FinancialIntent.TRANSACTION_HISTORY,
            IntentType.BUDGET_PLANNING: FinancialIntent.BUDGET_PLANNING,
            IntentType.SAVINGS_GOAL: FinancialIntent.SAVINGS_GOAL,
            IntentType.ACCOUNT_MANAGEMENT: FinancialIntent.ACCOUNT_MANAGEMENT,
            IntentType.FINANCIAL_ADVICE: FinancialIntent.FINANCIAL_ADVICE,
            IntentType.GREETING: FinancialIntent.GREETING,
            IntentType.HELP: FinancialIntent.HELP,
            IntentType.UNKNOWN: FinancialIntent.UNKNOWN
        }
        
        return mapping.get(intent_type, FinancialIntent.UNKNOWN)
    
    # ==========================================
    # MÉTRIQUES AGENT
    # ==========================================
    
    def _update_agent_metrics(self, result: Optional[IntentResult], latency_ms: float, success: bool):
        """Mise à jour métriques agent"""
        if success and result:
            self._agent_metrics["successful_classifications"] += 1
            
            # Confiance moyenne
            current_avg = self._agent_metrics["average_confidence"]
            total_successful = self._agent_metrics["successful_classifications"]
            new_confidence = result.confidence.score
            self._agent_metrics["average_confidence"] = (
                (current_avg * (total_successful - 1) + new_confidence) / total_successful
            )
            
            # Distribution intentions
            intent_name = result.intent_type.value
            self._agent_metrics["intent_distribution"][intent_name] = \
                self._agent_metrics["intent_distribution"].get(intent_name, 0) + 1
            
            # Usage niveaux
            level_key = result.level.value.split('_')[0]  # L0, L1, L2, ERROR
            if level_key in self._agent_metrics["level_usage"]:
                self._agent_metrics["level_usage"][level_key] += 1
            else:
                self._agent_metrics["level_usage"]["FALLBACK"] += 1
        else:
            self._agent_metrics["failed_classifications"] += 1
        
        # Latence moyenne
        current_avg_latency = self._agent_metrics["average_latency_ms"]
        total_classifications = self._agent_metrics["total_classifications"]
        self._agent_metrics["average_latency_ms"] = (
            (current_avg_latency * (total_classifications - 1) + latency_ms) / total_classifications
        )
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Métriques performance agent"""
        total = self._agent_metrics["total_classifications"]
        success_rate = (
            self._agent_metrics["successful_classifications"] / max(1, total)
        )
        
        return {
            "agent_version": "1.0.0",
            "total_classifications": total,
            "successful_classifications": self._agent_metrics["successful_classifications"],
            "failed_classifications": self._agent_metrics["failed_classifications"],
            "success_rate": round(success_rate, 3),
            "average_confidence": round(self._agent_metrics["average_confidence"], 3),
            "average_latency_ms": round(self._agent_metrics["average_latency_ms"], 2),
            "cache_size": len(self._agent_cache),
            "cache_max_size": self._agent_cache_max_size,
            "intent_distribution": self._agent_metrics["intent_distribution"].copy(),
            "level_usage": self._agent_metrics["level_usage"].copy(),
            "level_usage_percentages": {
                level: round((count / max(1, total)) * 100, 1)
                for level, count in self._agent_metrics["level_usage"].items()
            }
        }
    
    def get_top_intents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Top intentions les plus fréquentes"""
        distribution = self._agent_metrics["intent_distribution"]
        
        sorted_intents = sorted(
            distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        total_classifications = sum(distribution.values())
        
        return [
            {
                "intent": intent,
                "count": count,
                "percentage": round((count / max(1, total_classifications)) * 100, 1)
            }
            for intent, count in sorted_intents[:limit]
        ]
    
    # ==========================================
    # MÉTHODES DEBUG ET TESTING
    # ==========================================
    
    async def test_classification(self, query: str, expected_intent: str = None) -> Dict[str, Any]:
        """Test classification avec détails debug"""
        start_time = time.time()
        
        result = await self.classify_intent(query, "test_user")
        
        test_result = {
            "query": query,
            "expected_intent": expected_intent,
            "classification": {
                "intent": result.intent_type.value,
                "confidence": result.confidence.score,
                "level": result.level.value,
                "entities": result.entities,
                "from_cache": result.from_cache
            },
            "agent_enrichments": {
                "suggested_actions": self._get_suggested_actions(result.intent_type),
                "response_message": self._generate_response_message(result)
            },
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Vérification attente
        if expected_intent:
            test_result["expectation_met"] = (result.intent_type.value == expected_intent)
        
        return test_result
    
    async def benchmark_agent(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Benchmark agent sur cas de test"""
        logger.info(f"🏁 Benchmark Agent sur {len(test_cases)} cas...")
        
        start_time = time.time()
        results = []
        
        for case in test_cases:
            case_result = await self.test_classification(
                case["query"], 
                case.get("expected_intent")
            )
            results.append(case_result)
        
        # Analyse résultats
        successful = sum(1 for r in results if r["classification"]["intent"] != "UNKNOWN")
        correct = sum(1 for r in results if r.get("expectation_met", False))
        
        benchmark_results = {
            "total_cases": len(test_cases),
            "successful_classifications": successful,
            "correct_predictions": correct,
            "success_rate": round(successful / len(test_cases), 3),
            "accuracy": round(correct / max(1, len([c for c in test_cases if c.get("expected_intent")])), 3),
            "total_time_ms": round((time.time() - start_time) * 1000, 2),
            "average_latency_ms": round(sum(r["processing_time_ms"] for r in results) / len(results), 2),
            "agent_metrics": self.get_agent_metrics(),
            "results_sample": results[:5]
        }
        
        logger.info(f"🏁 Benchmark Agent terminé - Success: {benchmark_results['success_rate']:.1%}, "
                   f"Accuracy: {benchmark_results['accuracy']:.1%}")
        
        return benchmark_results
    
    def clear_agent_cache(self):
        """Vide cache agent"""
        self._agent_cache.clear()
        logger.info("🧹 Cache agent vidé")
    
    def reset_agent_metrics(self):
        """Reset métriques agent"""
        self._agent_metrics = {
            "total_classifications": 0,
            "successful_classifications": 0,
            "failed_classifications": 0,
            "average_confidence": 0.0,
            "average_latency_ms": 0.0,
            "intent_distribution": {},
            "level_usage": {"L0": 0, "L1": 0, "L2": 0, "FALLBACK": 0}
        }
        logger.info("🔄 Métriques agent reset")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Status santé agent"""
        engine_healthy = self.intent_engine is not None
        
        if engine_healthy:
            engine_status = await self.intent_engine.get_health_status()
        else:
            engine_status = {"available": False}
        
        agent_metrics = self.get_agent_metrics()
        
        # Détermination santé globale agent
        is_healthy = (
            engine_healthy and
            agent_metrics["success_rate"] > 0.8 and
            agent_metrics["average_latency_ms"] < 1000
        )
        
        return {
            "agent_healthy": is_healthy,
            "intent_engine_available": engine_healthy,
            "intent_engine_status": engine_status,
            "agent_metrics_summary": {
                "total_classifications": agent_metrics["total_classifications"],
                "success_rate": agent_metrics["success_rate"],
                "average_latency_ms": agent_metrics["average_latency_ms"],
                "top_intent": max(agent_metrics["intent_distribution"].items(), 
                                key=lambda x: x[1])[0] if agent_metrics["intent_distribution"] else None
            },
            "configuration": {
                "confidence_threshold": self.confidence_threshold,
                "suggestions_enabled": self.enable_suggestions,
                "response_generation_enabled": self.enable_response_generation
            }
        }
    
    async def shutdown(self):
        """Arrêt propre agent"""
        logger.info("🛑 Arrêt Intent Classifier Agent...")
        
        try:
            # Métriques finales
            final_metrics = self.get_agent_metrics()
            logger.info(f"📊 Métriques finales Agent: "
                       f"Classifications: {final_metrics['total_classifications']}, "
                       f"Success rate: {final_metrics['success_rate']:.1%}")
            
            # Arrêt Intent Detection Engine
            if self.intent_engine:
                await self.intent_engine.shutdown()
            
            # Clear caches
            self._agent_cache.clear()
            
            logger.info("✅ Intent Classifier Agent arrêté")
            
        except Exception as e:
            logger.error(f"❌ Erreur arrêt Intent Classifier Agent: {e}")


# ==========================================
# HELPERS ET UTILITAIRES
# ==========================================

def create_agent_test_cases() -> List[Dict[str, str]]:
    """Cas de test pour validation agent"""
    return [
        # Balance checks
        {"query": "quel est mon solde", "expected_intent": "BALANCE_CHECK"},
        {"query": "combien me reste-t-il sur mon compte", "expected_intent": "BALANCE_CHECK"},
        {"query": "position de mes comptes", "expected_intent": "BALANCE_CHECK"},
        
        # Transfers
        {"query": "faire un virement de 200 euros", "expected_intent": "TRANSFER"},
        {"query": "envoyer de l'argent à marie", "expected_intent": "TRANSFER"},
        {"query": "virement urgent", "expected_intent": "TRANSFER"},
        
        # Expenses
        {"query": "mes dépenses ce mois", "expected_intent": "EXPENSE_ANALYSIS"},
        {"query": "combien j'ai dépensé en restaurant", "expected_intent": "EXPENSE_ANALYSIS"},
        {"query": "analyser mon budget", "expected_intent": "EXPENSE_ANALYSIS"},
        
        # Cards
        {"query": "bloquer ma carte", "expected_intent": "CARD_MANAGEMENT"},
        {"query": "limites de ma carte", "expected_intent": "CARD_MANAGEMENT"},
        {"query": "paramètres carte", "expected_intent": "CARD_MANAGEMENT"},
        
        # System
        {"query": "bonjour", "expected_intent": "GREETING"},
        {"query": "j'ai besoin d'aide", "expected_intent": "HELP"},
        {"query": "comment ça marche", "expected_intent": "HELP"},
        
        # Complex/ambiguous
        {"query": "je veux optimiser mes finances", "expected_intent": "FINANCIAL_ADVICE"},
        {"query": "préparer achat maison", "expected_intent": "FINANCIAL_ADVICE"},
        
        # Edge cases
        {"query": "xyz abc 123", "expected_intent": "UNKNOWN"},
        {"query": "", "expected_intent": "UNKNOWN"}
    ]

async def validate_agent_performance(agent: IntentClassifier) -> Dict[str, Any]:
    """Validation performance agent complète"""
    test_cases = create_agent_test_cases()
    
    # Benchmark principal
    benchmark = await agent.benchmark_agent(test_cases)
    
    # Métriques détaillées
    agent_metrics = agent.get_agent_metrics()
    top_intents = agent.get_top_intents(5)
    health_status = await agent.get_health_status()
    
    # Validation targets agent
    target_success_rate = 0.90
    target_accuracy = 0.85
    target_latency = 100.0  # ms
    
    validation = {
        "performance_validation": {
            "target_success_rate": target_success_rate,
            "actual_success_rate": benchmark["success_rate"],
            "success_rate_met": benchmark["success_rate"] >= target_success_rate,
            
            "target_accuracy": target_accuracy,
            "actual_accuracy": benchmark["accuracy"],
            "accuracy_met": benchmark["accuracy"] >= target_accuracy,
            
            "target_latency_ms": target_latency,
            "actual_latency_ms": benchmark["average_latency_ms"],
            "latency_met": benchmark["average_latency_ms"] <= target_latency
        },
        "benchmark_results": benchmark,
        "agent_metrics": agent_metrics,
        "top_intents": top_intents,
        "health_status": health_status
    }
    
    return validation