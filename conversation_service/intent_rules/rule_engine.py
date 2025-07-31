"""
⚙️ Rule Engine - Moteur de règles de détection d'intentions

Ce module implémente le moteur de règles complet avec scoring intelligent,
cache haute performance et extraction d'entités intégrée. Il constitue le
cœur du système de détection hybride niveau 0.

Responsabilité : Détection d'intentions via règles configurables
Dépendances : rule_loader.py, pattern_matcher.py
Performance : <1ms exact match, <10ms pattern match avec entités

VERSION CORRIGÉE : Résolution du conflit de noms PatternMatcher
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, NamedTuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import hashlib

# Imports locaux
from .rule_loader import RuleLoader, IntentRule, IntentCategory, RuleMatch as LoaderRuleMatch
from .pattern_matcher import PatternMatcher as EntityPatternMatcher, EntityMatch, ExtractionResult, create_pattern_matcher

logger = logging.getLogger(__name__)


class RuleMatch(NamedTuple):
    """Résultat d'un match de règle avec entités"""
    intent: str
    intent_category: str
    confidence: float 
    entities: Dict[str, List[EntityMatch]]
    method: str  # "exact_match" | "pattern_match"
    execution_time_ms: float
    pattern_matched: str
    rule_priority: int
    entity_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le match en dictionnaire pour sérialisation"""
        return {
            "intent": self.intent,
            "intent_category": self.intent_category,
            "confidence": self.confidence,
            "entities": {
                entity_type: [
                    {
                        "entity_type": e.entity_type,
                        "raw_value": e.raw_value,
                        "normalized_value": e.normalized_value,
                        "confidence": e.confidence,
                        "position": e.position,
                        "extraction_method": e.extraction_method
                    }
                    for e in entity_list
                ]
                for entity_type, entity_list in self.entities.items()
            },
            "method": self.method,
            "execution_time_ms": self.execution_time_ms,
            "pattern_matched": self.pattern_matched,
            "rule_priority": self.rule_priority,
            "entity_count": self.entity_count
        }


@dataclass
class MatchingContext:
    """Contexte d'exécution pour le matching"""
    original_text: str
    normalized_text: str
    text_hash: str
    timestamp: datetime
    user_id: Optional[int] = None
    conversation_id: Optional[str] = None
    
    @classmethod
    def create(cls, text: str, user_id: Optional[int] = None, 
               conversation_id: Optional[str] = None) -> 'MatchingContext':
        """Factory method pour créer un contexte"""
        normalized = text.lower().strip()
        text_hash = hashlib.md5(normalized.encode()).hexdigest()[:8]
        
        return cls(
            original_text=text,
            normalized_text=normalized,
            text_hash=text_hash,
            timestamp=datetime.now(),
            user_id=user_id,
            conversation_id=conversation_id
        )


class ExactMatcher:
    """Matcher ultra-rapide pour correspondances exactes"""
    
    def __init__(self, rules: Dict[str, IntentRule]):
        """
        Initialise le matcher exact avec index optimisé
        
        Args:
            rules: Dictionnaire des règles d'intention
        """
        self.rules = rules
        
        # Index inversé pour recherche rapide : exact_match -> rule
        self.exact_match_index: Dict[str, IntentRule] = {}
        self._build_exact_index()
        
        # Statistiques
        self.match_count = 0
        self.total_time_ms = 0.0
    
    def _build_exact_index(self) -> None:
        """Construit l'index des correspondances exactes"""
        for rule in self.rules.values():
            for exact_match in rule.exact_matches:
                normalized_match = exact_match.lower().strip()
                if normalized_match in self.exact_match_index:
                    # Conflit : prendre la règle avec la plus haute priorité
                    existing_rule = self.exact_match_index[normalized_match]
                    if rule.priority < existing_rule.priority:  # Plus petit = plus prioritaire
                        self.exact_match_index[normalized_match] = rule
                else:
                    self.exact_match_index[normalized_match] = rule
        
        logger.info(f"Built exact match index with {len(self.exact_match_index)} entries")
    
    def find_exact_match(self, context: MatchingContext) -> Optional[RuleMatch]:
        """
        Recherche une correspondance exacte
        
        Args:
            context: Contexte de matching
            
        Returns:
            RuleMatch si trouvé, None sinon
        """
        start_time = datetime.now()
        
        # Recherche dans l'index
        rule = self.exact_match_index.get(context.normalized_text)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        self.match_count += 1
        self.total_time_ms += execution_time
        
        if rule:
            return RuleMatch(
                intent=rule.intent,
                intent_category=rule.intent_category.value,
                confidence=rule.confidence,
                entities={},  # Pas d'entités pour match exact simple
                method="exact_match",
                execution_time_ms=execution_time,
                pattern_matched=f"exact:{context.normalized_text}",
                rule_priority=rule.priority,
                entity_count=0
            )
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du matcher exact"""
        avg_time = self.total_time_ms / self.match_count if self.match_count > 0 else 0
        
        return {
            "matcher_type": "exact",
            "index_size": len(self.exact_match_index),
            "match_count": self.match_count,
            "avg_execution_time_ms": round(avg_time, 3),
            "total_time_ms": round(self.total_time_ms, 2)
        }


class PatternMatchScorer:
    """Calculateur de scores pour les matches de patterns"""
    
    @staticmethod
    def calculate_base_score(rule: IntentRule, pattern_matches: List[Tuple[Any, float]]) -> float:
        """
        Calcule le score de base d'une règle
        
        Args:
            rule: Règle d'intention
            pattern_matches: Liste des patterns matchés avec leurs poids
            
        Returns:
            Score de base [0.0, 1.0]
        """
        if not pattern_matches:
            return 0.0
        
        # Score basé sur les poids des patterns matchés
        total_weight = sum(weight for _, weight in pattern_matches)
        max_possible_weight = sum(pattern.weight for pattern in rule.patterns)
        
        if max_possible_weight == 0:
            return 0.0
        
        # Normalisation : score relatif aux patterns matchés
        base_score = min(total_weight / max_possible_weight, 1.0)
        
        # Application de la confidence de la règle
        final_score = base_score * rule.confidence
        
        return min(final_score, 1.0)
    
    @staticmethod
    def apply_entity_bonus(base_score: float, entity_count: int, required_entities: int = 0) -> float:
        """
        Applique un bonus basé sur les entités extraites
        
        Args:
            base_score: Score de base
            entity_count: Nombre d'entités extraites
            required_entities: Nombre d'entités requises par la règle
            
        Returns:
            Score ajusté avec bonus entités
        """
        if entity_count == 0:
            return base_score
        
        # Bonus pour entités extraites (max +10%)
        entity_bonus = min(entity_count * 0.05, 0.1)
        
        # Bonus supplémentaire si toutes les entités requises sont présentes
        if required_entities > 0 and entity_count >= required_entities:
            entity_bonus += 0.05
        
        return min(base_score + entity_bonus, 1.0)
    
    @staticmethod
    def apply_priority_adjustment(score: float, priority: int) -> float:
        """
        Applique un ajustement basé sur la priorité de la règle
        
        Args:
            score: Score de base
            priority: Priorité de la règle (1 = haute, 5 = basse)
            
        Returns:
            Score ajusté selon la priorité
        """
        # Légère pondération selon la priorité
        priority_multiplier = {
            1: 1.02,  # +2% pour très haute priorité
            2: 1.01,  # +1% pour haute priorité
            3: 1.00,  # Neutre
            4: 0.99,  # -1% pour basse priorité
            5: 0.98   # -2% pour très basse priorité
        }.get(priority, 1.00)
        
        return score * priority_multiplier


class RulePatternMatcher:
    """
    RENOMMÉ: Matcher avancé pour patterns regex avec scoring intelligent
    (Anciennement PatternMatcher - renommé pour éviter conflit avec EntityPatternMatcher)
    """
    
    def __init__(self, rules: Dict[str, IntentRule], entity_matcher: EntityPatternMatcher):
        """
        Initialise le matcher de patterns de règles
        
        Args:
            rules: Dictionnaire des règles d'intention
            entity_matcher: Matcher d'entités (EntityPatternMatcher)
        """
        self.rules = rules
        self.entity_matcher = entity_matcher
        self.scorer = PatternMatchScorer()
        
        # Pré-compilation des patterns pour performance
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, float, Dict]]] = {}
        self._precompile_patterns()
        
        # Statistiques
        self.match_count = 0
        self.total_time_ms = 0.0
        self.pattern_hit_stats: Dict[str, int] = defaultdict(int)
    
    def _precompile_patterns(self) -> None:
        """Pré-compile tous les patterns regex pour optimiser les performances"""
        for rule_name, rule in self.rules.items():
            compiled_patterns = []
            
            for pattern_rule in rule.patterns:
                try:
                    compiled_patterns.append((
                        pattern_rule.regex,
                        pattern_rule.weight,
                        {
                            'entity_extract': pattern_rule.entity_extract,
                            'extract_group': pattern_rule.extract_group,
                            'extract_groups': pattern_rule.extract_groups,
                            'normalize': pattern_rule.normalize
                        }
                    ))
                except Exception as e:
                    logger.error(f"Failed to compile pattern for rule {rule_name}: {e}")
                    continue
            
            self._compiled_patterns[rule_name] = compiled_patterns
        
        total_patterns = sum(len(patterns) for patterns in self._compiled_patterns.values())
        logger.info(f"Pre-compiled {total_patterns} patterns for {len(self._compiled_patterns)} rules")
    
    def find_pattern_matches(self, context: MatchingContext, 
                           confidence_threshold: float = 0.5) -> List[RuleMatch]:
        """
        Recherche des correspondances par patterns
        
        Args:
            context: Contexte de matching
            confidence_threshold: Seuil minimum de confidence
            
        Returns:
            Liste des matches trouvés, triés par score décroissant
        """
        start_time = datetime.now()
        matches = []
        
        # CORRECTION CRITIQUE : Extraction d'entités UNE SEULE FOIS avec le bon matcher
        entity_extraction = self.entity_matcher.extract_entities(context.original_text)
        
        # Test de chaque règle
        for rule_name, rule in self.rules.items():
            rule_match = self._test_rule_patterns(
                rule, context, entity_extraction, confidence_threshold
            )
            
            if rule_match:
                matches.append(rule_match)
        
        # Tri par score décroissant puis par priorité
        matches.sort(key=lambda m: (-m.confidence, m.rule_priority))
        
        # Statistiques
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        self.match_count += 1
        self.total_time_ms += execution_time
        
        return matches
    
    def _test_rule_patterns(self, rule: IntentRule, context: MatchingContext,
                          entity_extraction: ExtractionResult, 
                          confidence_threshold: float) -> Optional[RuleMatch]:
        """
        Test les patterns d'une règle spécifique
        
        Args:
            rule: Règle à tester
            context: Contexte de matching
            entity_extraction: Résultat d'extraction d'entités
            confidence_threshold: Seuil de confidence
            
        Returns:
            RuleMatch si la règle matche, None sinon
        """
        compiled_patterns = self._compiled_patterns.get(rule.intent, [])
        if not compiled_patterns:
            return None
        
        pattern_matches = []
        matched_patterns = []
        
        # Test de chaque pattern de la règle
        for regex_pattern, weight, metadata in compiled_patterns:
            matches = list(regex_pattern.finditer(context.original_text))
            
            if matches:
                pattern_matches.append((matches[0], weight))
                matched_patterns.append(regex_pattern.pattern)
                self.pattern_hit_stats[f"{rule.intent}:{regex_pattern.pattern}"] += 1
        
        if not pattern_matches:
            return None
        
        # Calcul du score
        base_score = self.scorer.calculate_base_score(rule, pattern_matches)
        
        # Bonus entités
        entity_count = entity_extraction.total_matches
        required_entities = len(getattr(rule, 'required_entities', []))
        score_with_entities = self.scorer.apply_entity_bonus(
            base_score, entity_count, required_entities
        )
        
        # Ajustement priorité
        final_score = self.scorer.apply_priority_adjustment(
            score_with_entities, rule.priority
        )
        
        # Vérification seuil
        if final_score < confidence_threshold:
            return None
        
        # CORRECTION CRITIQUE : Passage des entités extraites au RuleMatch
        return RuleMatch(
            intent=rule.intent,
            intent_category=rule.intent_category.value,
            confidence=final_score,
            entities=entity_extraction.entities,  # ENTITÉS PASSÉES CORRECTEMENT
            method="pattern_match",
            execution_time_ms=0.0,  # Sera mis à jour par le caller
            pattern_matched=";".join(matched_patterns),
            rule_priority=rule.priority,
            entity_count=entity_count
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du matcher de patterns"""
        avg_time = self.total_time_ms / self.match_count if self.match_count > 0 else 0
        
        # Top 5 des patterns les plus utilisés
        top_patterns = sorted(
            self.pattern_hit_stats.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "matcher_type": "rule_pattern",
            "rules_count": len(self.rules),
            "compiled_patterns_count": sum(len(p) for p in self._compiled_patterns.values()),
            "match_count": self.match_count,
            "avg_execution_time_ms": round(avg_time, 3),
            "total_time_ms": round(self.total_time_ms, 2),
            "top_patterns": [
                {"pattern": pattern, "hits": hits} 
                for pattern, hits in top_patterns
            ]
        }


class RuleEngine:
    """
    Moteur de règles principal avec cache intelligent et métriques avancées
    
    Ce moteur combine recherche exacte ultra-rapide et pattern matching intelligent
    pour détecter les intentions avec extraction d'entités intégrée.
    
    VERSION CORRIGÉE : Utilise EntityPatternMatcher pour l'extraction d'entités
                      et RulePatternMatcher pour le matching de règles
    """
    
    def __init__(self, rule_loader: Optional[RuleLoader] = None, 
                 entity_matcher: Optional[EntityPatternMatcher] = None):
        """
        Initialise le moteur de règles
        
        Args:
            rule_loader: Loader des règles (optionnel)
            entity_matcher: Matcher d'entités (EntityPatternMatcher, optionnel)
        """
        # Initialisation des composants
        if rule_loader is None:
            from . import get_default_loader
            rule_loader = get_default_loader()
        
        if entity_matcher is None:
            entity_matcher = create_pattern_matcher(rule_loader)
        
        self.rule_loader = rule_loader
        self.entity_matcher = entity_matcher  # CORRECTION : EntityPatternMatcher
        
        # Chargement des règles
        self.financial_rules = rule_loader.get_financial_rules()
        self.conversational_rules = rule_loader.get_conversational_rules()
        self.all_rules = {**self.financial_rules, **self.conversational_rules}
        
        # Initialisation des matchers
        self.exact_matcher = ExactMatcher(self.all_rules)
        self.pattern_matcher = RulePatternMatcher(self.all_rules, entity_matcher)  # CORRECTION
        
        # Cache des résultats
        self._result_cache: Dict[str, RuleMatch] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Configuration
        self.default_confidence_threshold = 0.7
        self.max_pattern_matches = 5
        self.enable_cache = True
        
        logger.info(f"RuleEngine initialized with {len(self.all_rules)} rules")
    
    def match_intent(self, text: str, user_id: Optional[int] = None,
                    conversation_id: Optional[str] = None,
                    confidence_threshold: Optional[float] = None) -> Optional[RuleMatch]:
        """
        Détecte l'intention d'un texte (point d'entrée principal)
        
        Args:
            text: Texte à analyser
            user_id: ID utilisateur (optionnel)
            conversation_id: ID conversation (optionnel)
            confidence_threshold: Seuil de confidence personnalisé
            
        Returns:
            Meilleur match trouvé ou None
        """
        if not text or not text.strip():
            return None
        
        start_time = datetime.now()
        threshold = confidence_threshold or self.default_confidence_threshold
        
        # Création du contexte
        context = MatchingContext.create(text, user_id, conversation_id)
        
        # Vérification cache
        cache_key = f"{context.text_hash}_{threshold}"
        if self.enable_cache and cache_key in self._result_cache:
            self._cache_hits += 1
            cached_result = self._result_cache[cache_key]
            logger.debug(f"Cache hit for: {text[:50]}")
            return cached_result
        
        self._cache_misses += 1
        
        # 1. Tentative exact match (ultra-rapide)
        exact_match = self.exact_matcher.find_exact_match(context)
        if exact_match and exact_match.confidence >= threshold:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # CORRECTION : Ajout des entités même pour exact match
            if exact_match.entity_count == 0:
                # Extraction d'entités même pour les matches exacts
                entity_extraction = self.entity_matcher.extract_entities(context.original_text)
                exact_match = exact_match._replace(
                    entities=entity_extraction.entities,
                    entity_count=entity_extraction.total_matches
                )
            
            # Mise à jour du temps d'exécution
            final_match = exact_match._replace(execution_time_ms=execution_time)
            
            # Mise en cache
            self._cache_result(cache_key, final_match)
            
            logger.debug(f"Exact match found for: {text[:50]} -> {exact_match.intent}")
            return final_match
        
        # 2. Pattern matching (plus lent mais intelligent)
        pattern_matches = self.pattern_matcher.find_pattern_matches(context, threshold)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if pattern_matches:
            # Prendre le meilleur match
            best_match = pattern_matches[0]
            
            # Mise à jour du temps d'exécution
            final_match = best_match._replace(execution_time_ms=execution_time)
            
            # Mise en cache
            self._cache_result(cache_key, final_match)
            
            logger.debug(f"Pattern match found for: {text[:50]} -> {best_match.intent} (score: {best_match.confidence:.3f})")
            return final_match
        
        logger.debug(f"No match found for: {text[:50]}")
        return None
    
    def match_exact(self, text: str) -> Optional[RuleMatch]:
        """
        Recherche uniquement par correspondance exacte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Match exact ou None
        """
        context = MatchingContext.create(text)
        return self.exact_matcher.find_exact_match(context)
    
    def match_patterns(self, text: str, confidence_threshold: float = 0.5) -> List[RuleMatch]:
        """
        Recherche uniquement par patterns (retourne tous les matches)
        
        Args:
            text: Texte à analyser
            confidence_threshold: Seuil minimum de confidence
            
        Returns:
            Liste de tous les matches trouvés
        """
        context = MatchingContext.create(text)
        return self.pattern_matcher.find_pattern_matches(context, confidence_threshold)
    
    def match_by_category(self, text: str, category: IntentCategory,
                         confidence_threshold: float = 0.7) -> Optional[RuleMatch]:
        """
        Recherche dans une catégorie spécifique d'intentions
        
        Args:
            text: Texte à analyser
            category: Catégorie d'intentions
            confidence_threshold: Seuil de confidence
            
        Returns:
            Meilleur match dans la catégorie ou None
        """
        # Filtrage des règles par catégorie
        category_rules = {
            name: rule for name, rule in self.all_rules.items()
            if rule.intent_category == category
        }
        
        if not category_rules:
            return None
        
        # Création d'un matcher temporaire avec les règles filtrées
        temp_exact_matcher = ExactMatcher(category_rules)
        temp_pattern_matcher = RulePatternMatcher(category_rules, self.entity_matcher)
        
        context = MatchingContext.create(text)
        
        # Test exact match d'abord
        exact_match = temp_exact_matcher.find_exact_match(context)
        if exact_match and exact_match.confidence >= confidence_threshold:
            return exact_match
        
        # Test pattern match
        pattern_matches = temp_pattern_matcher.find_pattern_matches(context, confidence_threshold)
        return pattern_matches[0] if pattern_matches else None
    
    def validate_rules(self) -> Dict[str, List[str]]:
        """
        Valide toutes les règles chargées
        
        Returns:
            Dictionnaire des erreurs trouvées
        """
        return self.rule_loader.validate_rules()
    
    def get_rule_info(self, intent: str) -> Optional[Dict[str, Any]]:
        """
        Retourne les informations détaillées d'une règle
        
        Args:
            intent: Nom de l'intention
            
        Returns:
            Informations de la règle ou None
        """
        rule = self.all_rules.get(intent)
        if not rule:
            return None
        
        return {
            "intent": rule.intent,
            "description": rule.description,
            "category": rule.intent_category.value,
            "confidence": rule.confidence,
            "priority": rule.priority,
            "patterns_count": len(rule.patterns),
            "exact_matches_count": len(rule.exact_matches),
            "exact_matches": list(rule.exact_matches),
            "examples": rule.examples or []
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques complètes de performance
        
        Returns:
            Dictionnaire avec toutes les métriques
        """
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "engine_info": {
                "total_rules": len(self.all_rules),
                "financial_rules": len(self.financial_rules),
                "conversational_rules": len(self.conversational_rules),
                "default_confidence_threshold": self.default_confidence_threshold
            },
            "cache_stats": {
                "enabled": self.enable_cache,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate_percent": round(cache_hit_rate, 2),
                "cache_size": len(self._result_cache)
            },
            "matcher_stats": {
                "exact_matcher": self.exact_matcher.get_stats(),
                "pattern_matcher": self.pattern_matcher.get_stats()
            },
            "entity_matcher_stats": self.entity_matcher.get_performance_stats()
        }
    
    def clear_cache(self) -> None:
        """Vide tous les caches du moteur"""
        self._result_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.entity_matcher.clear_cache()
        logger.info("RuleEngine cache cleared")
    
    def reload_rules(self) -> None:
        """Recharge toutes les règles depuis les fichiers"""
        logger.info("Reloading rules...")
        
        # Rechargement via le rule loader
        self.rule_loader.reload_rules()
        
        # Rechargement des règles
        self.financial_rules = self.rule_loader.get_financial_rules()
        self.conversational_rules = self.rule_loader.get_conversational_rules()
        self.all_rules = {**self.financial_rules, **self.conversational_rules}
        
        # Réinitialisation des matchers
        self.exact_matcher = ExactMatcher(self.all_rules)
        self.pattern_matcher = RulePatternMatcher(self.all_rules, self.entity_matcher)
        
        # Vidage du cache
        self.clear_cache()
        
        logger.info(f"Rules reloaded: {len(self.all_rules)} total rules")
    
    def _cache_result(self, cache_key: str, result: RuleMatch) -> None:
        """Met en cache un résultat (avec limite de taille)"""
        if not self.enable_cache:
            return
        
        # Limite du cache pour éviter la surcharge mémoire
        if len(self._result_cache) >= 1000:
            # Suppression des plus anciens (stratégie FIFO simple)
            oldest_keys = list(self._result_cache.keys())[:100]
            for key in oldest_keys:
                del self._result_cache[key]
        
        self._result_cache[cache_key] = result


# Factory function pour faciliter l'utilisation
def create_rule_engine(rule_loader: Optional[RuleLoader] = None,
                      entity_matcher: Optional[EntityPatternMatcher] = None) -> RuleEngine:
    """
    Factory function pour créer un RuleEngine
    
    Args:
        rule_loader: RuleLoader personnalisé (optionnel)
        entity_matcher: EntityPatternMatcher personnalisé (optionnel)
        
    Returns:
        Instance RuleEngine configurée
    """
    return RuleEngine(rule_loader, entity_matcher)


# Exports principaux
__all__ = [
    "RuleEngine",
    "RuleMatch", 
    "MatchingContext",
    "ExactMatcher",
    "RulePatternMatcher",  # RENOMMÉ
    "PatternMatchScorer",
    "create_rule_engine"
]