#!/usr/bin/env python3
"""
📏 Moteur de Règles Intelligent - Cœur du système de détection

Moteur de règles heuristiques ultra-optimisé reprenant la logique éprouvée
du fichier original avec améliorations modulaires et performance.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Pattern
from dataclasses import dataclass
from conversation_service.models.enums import IntentType, DetectionMethod
from conversation_service.models.exceptions import RuleEngineError, PatternCompilationError
from conversation_service.config import config
from conversation_service.utils.helpers.text_helpers import get_compiled_pattern
from ..entity_extraction.pattern_matcher import get_pattern_matcher

logger = logging.getLogger(__name__)


@dataclass
class IntentRule:
    """Configuration d'une règle de détection d'intention"""
    intent: IntentType
    patterns: List[str]
    entity_patterns: List[Tuple[str, str, Optional[str]]]  # (pattern, name, default_value)
    confidence_boost: float
    priority: int = 5
    requires_entities: bool = False


class IntelligentRuleEngine:
    """
    Moteur de règles intelligent pour détection d'intention ultra-rapide
    
    Reprend exactement la logique du fichier original qui fonctionne bien,
    avec améliorations modulaires et monitoring avancé.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration des règles par intention (du fichier original)
        self.intent_rules = {
            IntentType.ACCOUNT_BALANCE: IntentRule(
                intent=IntentType.ACCOUNT_BALANCE,
                patterns=[
                    r"\b(solde|combien.*ai|argent.*compte|euros?\s+sur|balance|disponible|reste)\b",
                    r"\b(compte.*courant|livret|épargne)\b.*\b(solde|combien)\b", 
                    r"\bmon\s+(solde|compte)\b"
                ],
                entity_patterns=[
                    (r"\b(compte\s+courant|livret\s+a|épargne|livret)\b", "account_type", None),
                    (r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b", "month", None)
                ],
                confidence_boost=0.9,
                priority=1
            ),
            
            IntentType.SEARCH_BY_CATEGORY: IntentRule(
                intent=IntentType.SEARCH_BY_CATEGORY,
                patterns=[
                    r"\b(restaurant|resto|repas|dîner|déjeuner)\b",
                    r"\b(courses|alimentation|supermarché|carrefour|leclerc)\b",
                    r"\b(transport|essence|carburant|taxi|uber)\b", 
                    r"\b(shopping|vêtements|achats|boutique)\b",
                    r"\b(loisirs|cinéma|sport|vacances)\b",
                    r"\b(santé|pharmacie|médecin|dentiste)\b",
                    r"\b(dépenses?.*\b(restaurant|courses|transport|shopping|loisirs|santé)\b)"
                ],
                entity_patterns=[
                    (r"\b(restaurant|resto|repas|dîner|déjeuner)\b", "category", "restaurant"),
                    (r"\b(courses|alimentation|supermarché)\b", "category", "alimentation"),
                    (r"\b(transport|essence|taxi|uber)\b", "category", "transport"),
                    (r"\b(shopping|vêtements|achats)\b", "category", "shopping"),
                    (r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b", "month", None),
                    (r"\b(ce\s+mois|mois\s+dernier|cette\s+semaine|semaine\s+dernière)\b", "period", None)
                ],
                confidence_boost=0.85,
                priority=2,
                requires_entities=True
            ),
            
            IntentType.BUDGET_ANALYSIS: IntentRule(
                intent=IntentType.BUDGET_ANALYSIS,
                patterns=[
                    r"\b(budget|dépensé|combien.*dépensé|coûté|montant)\b",
                    r"\b(analyse|bilan|résumé|total).*\b(dépenses?|budget)\b",
                    r"\b(j'ai\s+dépensé|ça\s+m'a\s+coûté)\b"
                ],
                entity_patterns=[
                    (r"\b(\d+)\s*euros?\b", "amount", None),
                    (r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b", "month", None),
                    (r"\b(ce\s+mois|mois\s+dernier|cette\s+année|année\s+dernière)\b", "period", None)
                ],
                confidence_boost=0.8,
                priority=3
            ),
            
            IntentType.TRANSFER: IntentRule(
                intent=IntentType.TRANSFER,
                patterns=[
                    r"\b(virer|virement|transfert|transférer)\b",
                    r"\b(envoyer|verser|payer).*\b(argent|euros?)\b",
                    r"\b(donner|prêter).*\b(\d+.*euros?)\b"
                ],
                entity_patterns=[
                    (r"\b(\d+(?:[,\.]\d+)?)\s*euros?\b", "amount", None),
                    (r"\b(à|vers|pour)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", "recipient", None)
                ],
                confidence_boost=0.9,
                priority=1
            ),
            
            IntentType.SEARCH_BY_DATE: IntentRule(
                intent=IntentType.SEARCH_BY_DATE,
                patterns=[
                    r"\b(historique|transactions|opérations|mouvements)\b",
                    r"\b(hier|avant-hier|semaine|mois).*\b(dernier|dernière|passé)\b",
                    r"\b(récent|dernier|précédent)\b"
                ],
                entity_patterns=[
                    (r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b", "month", None),
                    (r"\b(hier|avant-hier|semaine\s+dernière|mois\s+dernier)\b", "period", None),
                    (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "date", None)
                ],
                confidence_boost=0.75,
                priority=4
            ),
            
            IntentType.CARD_MANAGEMENT: IntentRule(
                intent=IntentType.CARD_MANAGEMENT,
                patterns=[
                    r"\b(carte|cb|visa|mastercard)\b.*\b(bloquer|débloquer|opposition)\b",
                    r"\b(bloquer|annuler|gérer).*\bcarte\b",
                    r"\b(limite|plafond).*\bcarte\b"
                ],
                entity_patterns=[
                    (r"\b(visa|mastercard|cb|carte\s+bleue)\b", "card_type", None),
                    (r"\b(\d+)\s*euros?\b", "amount", None)
                ],
                confidence_boost=0.95,
                priority=1
            ),
            
            IntentType.GREETING: IntentRule(
                intent=IntentType.GREETING,
                patterns=[
                    r"^\s*(bonjour|salut|hello|bonsoir|coucou|hey)\b",
                    r"\b(bonjour|salut).*\b(comment\s+(ça\s+va|allez-vous))\b"
                ],
                entity_patterns=[],
                confidence_boost=0.95,
                priority=5
            ),
            
            IntentType.HELP: IntentRule(
                intent=IntentType.HELP,
                patterns=[
                    r"\b(aide|aidez|help|comment)\b",
                    r"\b(expliquer|ne\s+comprends?\s+pas|aide-moi)\b",
                    r"^\s*(que|qu'est-ce|comment).*\b(faire|fonctionne)\b"
                ],
                entity_patterns=[],
                confidence_boost=0.8,
                priority=5
            ),
            
            IntentType.GOODBYE: IntentRule(
                intent=IntentType.GOODBYE,
                patterns=[
                    r"\b(au\s+revoir|bye|ciao|à\s+bientôt|goodbye|salut)\b",
                    r"\b(merci.*bye|c'est\s+tout|terminé)\b"
                ],
                entity_patterns=[],
                confidence_boost=0.95,
                priority=5
            )
        }
        
        # Cache des patterns compilés pour performance
        self._compiled_patterns: Dict[IntentType, Dict[str, Any]] = {}
        self._compile_patterns()
        
        # Métriques du moteur
        self._metrics = {
            "total_detections": 0,
            "successful_detections": 0,
            "detection_times": [],
            "intent_distribution": {},
            "confidence_distribution": {}
        }
        
        # Pattern matcher pour entités
        self.pattern_matcher = get_pattern_matcher()
    
    def _compile_patterns(self):
        """Pre-compilation des patterns regex pour performance optimale"""
        self.logger.info("Compilation des patterns de règles...")
        
        for intent, rule in self.intent_rules.items():
            try:
                compiled_rule = {
                    "patterns": [
                        get_compiled_pattern(pattern, re.IGNORECASE) 
                        for pattern in rule.patterns
                    ],
                    "entities": [
                        (
                            get_compiled_pattern(pattern, re.IGNORECASE), 
                            name, 
                            default_value
                        )
                        for pattern, name, default_value in rule.entity_patterns
                    ],
                    "confidence_boost": rule.confidence_boost,
                    "priority": rule.priority,
                    "requires_entities": rule.requires_entities
                }
                self._compiled_patterns[intent] = compiled_rule
                
            except re.error as e:
                raise PatternCompilationError(
                    pattern=f"Intent {intent} patterns",
                    regex_error=str(e)
                )
        
        self.logger.info(f"✅ {len(self._compiled_patterns)} intentions avec patterns compilés")
    
    def detect_intent(self, query):
        # type: (str) -> Tuple[IntentType, float, Dict[str, Any]]
        """
        Détection ultra-rapide par règles (méthode principale)
        
        Args:
            query: Requête utilisateur nettoyée
            
        Returns:
            (intent, confidence, entities)
            
        Raises:
            RuleEngineError: Si erreur pendant la détection
        """
        if not query:
            raise RuleEngineError(
                "Query vide fournie au moteur de règles",
                query=query
            )
        
        try:
            query_clean = query.strip().lower()
            self._metrics["total_detections"] += 1
            
            # Score par intention avec priorité
            intent_scores = {}
            
            for intent, compiled_rule in self._compiled_patterns.items():
                score = self._calculate_intent_score(query_clean, intent, compiled_rule)
                if score > 0:
                    # Ajustement score selon priorité intention
                    priority_weight = 1.0 + (5 - compiled_rule["priority"]) * 0.02
                    intent_scores[intent] = score * priority_weight
            
            # Sélection meilleure intention
            if intent_scores:
                best_intent = max(intent_scores, key=intent_scores.get)
                best_score = intent_scores[best_intent]
                
                # Seuil minimum de confiance
                if best_score > config.rule_engine.min_match_threshold:
                    # Extraction entités pour cette intention
                    entities = self._extract_entities_for_intent(query, best_intent)
                    
                    # Bonus si entités requises trouvées
                    if self._compiled_patterns[best_intent]["requires_entities"]:
                        if entities:
                            best_score *= 1.1
                        else:
                            best_score *= 0.8  # Pénalité si entités manquantes
                    
                    # Mise à jour métriques
                    self._update_metrics(best_intent, best_score)
                    
                    return best_intent, min(best_score, 1.0), entities
            
            return IntentType.UNKNOWN, 0.05, {}
            
        except Exception as e:
            # Wrapper toute exception en RuleEngineError
            raise RuleEngineError(
                f"Erreur lors de la détection d'intention: {str(e)}",
                query=query,
                details={"original_exception": str(e)}
            )
    
    def _calculate_intent_score(
        self, 
        query: str, 
        intent: IntentType, 
        compiled_rule: Dict[str, Any]
    ) -> float:
        """Calcule le score de correspondance pour une intention"""
        score = 0.0
        matches = 0
        
        # Test de chaque pattern
        for pattern in compiled_rule["patterns"]:
            if pattern.search(query):
                matches += 1
                # Score basé sur spécificité du pattern
                pattern_specificity = len(pattern.pattern) / len(query) if len(query) > 0 else 0
                pattern_score = pattern_specificity * compiled_rule["confidence_boost"]
                score += pattern_score
        
        # Bonus pour multiples matches
        if matches > 1:
            score *= config.rule_engine.multi_pattern_boost
        
        return min(score, 1.0)
    
    def _extract_entities_for_intent(
        self, 
        query: str, 
        intent: IntentType
    ) -> Dict[str, Any]:
        """Extrait les entités spécifiques à une intention"""
        entities = {}
        
        if intent not in self._compiled_patterns:
            return entities
        
        entity_patterns = self._compiled_patterns[intent]["entities"]
        
        for pattern, name, default_value in entity_patterns:
            matches = pattern.findall(query)
            if matches:
                if default_value:
                    entities[name] = default_value
                else:
                    # Prendre première valeur trouvée
                    match_value = matches[0]
                    if isinstance(match_value, tuple):
                        # Pattern avec groupes multiples (ex: date)
                        entities[name] = match_value
                    else:
                        entities[name] = match_value
        
        return entities
    
    def get_suggestions(self, intent: IntentType, entities: Dict[str, Any]) -> List[str]:
        """
        Génère suggestions contextuelles intelligentes
        Reprend la logique du fichier original
        """
        suggestions_map = {
            IntentType.ACCOUNT_BALANCE: [
                "Voir l'historique des soldes",
                "Comparer avec le mois dernier", 
                "Afficher tous mes comptes"
            ],
            IntentType.SEARCH_BY_CATEGORY: [
                f"Détails des dépenses {entities.get('category', '')}".strip(),
                "Comparer avec la période précédente",
                "Voir le budget par catégorie"
            ],
            IntentType.BUDGET_ANALYSIS: [
                "Analyse détaillée par catégorie",
                "Évolution sur plusieurs mois",
                "Comparaison avec mes objectifs"
            ],
            IntentType.TRANSFER: [
                "Voir mes bénéficiaires récents",
                "Programmer un virement récurrent",
                "Vérifier mes limites de virement"
            ],
            IntentType.SEARCH_BY_DATE: [
                "Filtrer par montant",
                "Exporter les données",
                "Recherche par marchand"
            ],
            IntentType.CARD_MANAGEMENT: [
                "Voir l'historique de la carte",
                "Modifier les limites",
                "Gérer les alertes"
            ],
            IntentType.GREETING: [
                "Que puis-je faire pour vous ?",
                "Consulter votre solde",
                "Voir vos dernières transactions"
            ],
            IntentType.HELP: [
                "Guide d'utilisation",
                "Questions fréquentes", 
                "Contacter le support"
            ],
            IntentType.GOODBYE: [
                "À bientôt !",
                "N'hésitez pas à revenir",
                "Service disponible 24h/24"
            ]
        }
        
        base_suggestions = suggestions_map.get(intent, [
            "Que puis-je faire d'autre ?",
            "Voir l'aide complète",
            "Retour au menu principal"
        ])
        
        return base_suggestions[:3]
    
    def _update_metrics(self, intent: IntentType, confidence: float):
        """Met à jour les métriques du moteur de règles"""
        self._metrics["successful_detections"] += 1
        
        # Distribution par intention
        intent_key = intent.value
        if intent_key not in self._metrics["intent_distribution"]:
            self._metrics["intent_distribution"][intent_key] = 0
        self._metrics["intent_distribution"][intent_key] += 1
        
        # Distribution par niveau de confiance
        confidence_level = "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low"
        if confidence_level not in self._metrics["confidence_distribution"]:
            self._metrics["confidence_distribution"][confidence_level] = 0
        self._metrics["confidence_distribution"][confidence_level] += 1
    
    def get_rule_metrics(self):
        # type: () -> Dict[str, Any]
        """Retourne métriques détaillées du moteur de règles"""
        if self._metrics["total_detections"] == 0:
            return {"total_detections": 0, "success_rate": 0.0, "detection_method": DetectionMethod.RULES.value}
        
        success_rate = self._metrics["successful_detections"] / self._metrics["total_detections"]
        
        return {
            "detection_method": DetectionMethod.RULES.value,
            "method_info": self.get_detection_method_info(),
            "total_detections": self._metrics["total_detections"],
            "successful_detections": self._metrics["successful_detections"],
            "success_rate": round(success_rate, 3),
            "intent_distribution": self._metrics["intent_distribution"],
            "confidence_distribution": self._metrics["confidence_distribution"],
            "supported_intents": len(self.intent_rules),
            "compiled_patterns": len(self._compiled_patterns),
            "performance": {
                "method": DetectionMethod.RULES,
                "expected_latency_ms": 5.0,
                "deterministic": True,
                "requires_training": False
            }
        }
    
    def get_detection_method_info(self):
        # type: () -> Dict[str, Any]
        """
        Informations sur la méthode de détection utilisée
        
        Returns:
            Dict avec informations DetectionMethod.RULES
        """
        return {
            "method": DetectionMethod.RULES,
            "method_name": DetectionMethod.RULES.value,
            "description": "Moteur de règles heuristiques intelligent",
            "expected_latency_ms": 5.0,
            "strengths": [
                "Ultra-rapide (< 5ms)",
                "Déterministe et explicable", 
                "Optimisé pour français financier",
                "Extraction entités intégrée"
            ],
            "use_cases": [
                "Détection intention première ligne",
                "Requêtes financières standards",
                "Performance critique"
            ],
            "confidence_range": {
                "min": 0.0,
                "max": 1.0,
                "typical_high": 0.9,
                "typical_low": 0.1
            }
        }
    
    def validate_rule_engine_health(self):
        # type: () -> Dict[str, Any]
        """
        Validation santé du moteur de règles
        
        Returns:
            Dict avec statut santé et diagnostics
            
        Raises:
            RuleEngineError: Si problèmes critiques détectés
        """
        health_report = {
            "status": "healthy",
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check 1: Patterns compilés
            if not self._compiled_patterns:
                health_report["errors"].append("Aucun pattern compilé")
                health_report["status"] = "unhealthy"
            else:
                health_report["checks"]["compiled_patterns"] = len(self._compiled_patterns)
            
            # Check 2: Test détection basique
            test_queries = [
                ("bonjour", IntentType.GREETING),
                ("mon solde", IntentType.ACCOUNT_BALANCE),
                ("au revoir", IntentType.GOODBYE)
            ]
            
            successful_tests = 0
            for test_query, expected_intent in test_queries:
                try:
                    detected_intent, confidence, _ = self.detect_intent(test_query)
                    if detected_intent == expected_intent and confidence > 0.5:
                        successful_tests += 1
                except Exception as e:
                    health_report["warnings"].append(f"Test échoué pour '{test_query}': {e}")
            
            health_report["checks"]["detection_tests"] = f"{successful_tests}/{len(test_queries)}"
            
            if successful_tests < len(test_queries) * 0.5:  # Moins de 50% réussis
                health_report["status"] = "degraded"
                health_report["warnings"].append("Taux de réussite tests < 50%")
            
            # Check 3: Performance patterns
            if len(self._compiled_patterns) != len(self.intent_rules):
                health_report["warnings"].append("Incohérence nombre patterns compilés")
            
            # Check 4: Métriques basiques
            if hasattr(self, '_metrics'):
                health_report["checks"]["total_detections"] = self._metrics.get("total_detections", 0)
                health_report["checks"]["successful_detections"] = self._metrics.get("successful_detections", 0)
            
            return health_report
            
        except Exception as e:
            raise RuleEngineError(
                "Erreur lors de la validation santé moteur de règles",
                details={"validation_error": str(e)}
            )
        """Teste un pattern contre une liste de requêtes (debug)"""
        try:
            compiled_pattern = get_compiled_pattern(pattern, re.IGNORECASE)
            results = []
            
            for query in test_queries:
                match = compiled_pattern.search(query.lower())
                results.append({
                    "query": query,
                    "matched": bool(match),
                    "match_text": match.group() if match else None
                })
            
            return {
                "pattern": pattern,
                "test_results": results,
                "match_rate": sum(1 for r in results if r["matched"]) / len(results)
            }
            
        except re.error as e:
            return {
                "pattern": pattern,
                "error": str(e),
                "test_results": []
            }
    
    def get_intent_to_search_code(self, intent: IntentType) -> str:
        """Mapping intention vers code search service"""
        return config.get_intent_search_code(intent.value)
    
    def reset_metrics(self):
        """Remet à zéro les métriques (utile pour tests)"""
        self._metrics = {
            "total_detections": 0,
            "successful_detections": 0,
            "detection_times": [],
            "intent_distribution": {},
            "confidence_distribution": {}
        }


# Instance singleton du moteur de règles
_rule_engine_instance = None

def get_rule_engine() -> IntelligentRuleEngine:
    """Factory function pour récupérer instance RuleEngine singleton"""
    global _rule_engine_instance
    if _rule_engine_instance is None:
        _rule_engine_instance = IntelligentRuleEngine()
    return _rule_engine_instance


# Fonction utilitaire de détection rapide
def quick_intent_detection(query: str) -> Tuple[str, float]:
    """Détection rapide d'intention sans extraction d'entités"""
    engine = get_rule_engine()
    intent, confidence, _ = engine.detect_intent(query)
    return intent.value, confidence


# Exports publics
__all__ = [
    "IntelligentRuleEngine",
    "IntentRule",
    "get_rule_engine",
    "quick_intent_detection"
]