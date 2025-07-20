"""
⚡ Niveau L0 - Pattern Matcher ultra-rapide

Reconnaissance patterns financiers fréquents avec regex pré-compilés
pour objectif performance <10ms sur 85% des requêtes.
"""

import re
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from conversation_service.intent_detection.models import (
    IntentResult, IntentType, IntentLevel, IntentConfidence
)
from conversation_service.intent_detection.cache_manager import CacheManager

logger = logging.getLogger(__name__)

@dataclass
class PatternMatch:
    """Résultat match pattern avec métadonnées"""
    intent_type: IntentType
    confidence: float
    matched_text: str
    pattern_name: str
    entities: Dict[str, Any]
    match_groups: Dict[str, str]

class FinancialPatterns:
    """
    📚 Bibliothèque patterns financiers pré-compilés
    
    Patterns organisés par intention avec extraction entités basique
    et scoring confiance optimisé pour français financier.
    """
    
    def __init__(self):
        self.compiled_patterns: Dict[IntentType, List[re.Pattern]] = {}
        self.pattern_metadata: Dict[str, Dict[str, Any]] = {}
        self._compile_all_patterns()
    
    def _compile_all_patterns(self):
        """Compilation tous les patterns au démarrage pour performance"""
        logger.info("🔧 Compilation patterns financiers...")
        
        # ==========================================
        # PATTERNS BALANCE_CHECK (très fréquent)
        # ==========================================
        balance_patterns = [
            # Patterns directs solde
            (r"(?:quel|combien|voir|consulter|afficher)\s+(?:est\s+)?(?:le\s+)?(?:mon\s+)?solde(?:\s+de\s+mon\s+compte)?", 0.95, "direct_balance"),
            (r"solde\s+(?:de\s+)?(?:mon\s+)?compte(?:\s+courant)?", 0.92, "account_balance"),
            (r"combien\s+(?:ai-je|j'ai)\s+(?:sur\s+)?(?:mon\s+)?compte", 0.90, "how_much_account"),
            (r"(?:mon\s+)?argent\s+(?:disponible|restant)", 0.88, "available_money"),
            (r"position\s+(?:de\s+)?(?:mon\s+)?compte", 0.85, "account_position"),
            
            # Patterns avec types comptes
            (r"solde\s+compte\s+(courant|épargne|livret)", 0.93, "specific_account_balance"),
            (r"(?:voir|consulter)\s+(?:mes\s+)?comptes", 0.87, "view_accounts"),
            
            # Patterns temporels
            (r"solde\s+(?:actuel|maintenant|aujourd'hui)", 0.91, "current_balance"),
            (r"combien\s+(?:me\s+reste|reste)(?:\s+t)?(?:\s+il)?", 0.86, "remaining_money")
        ]
        
        # ==========================================
        # PATTERNS TRANSFER (action fréquente)
        # ==========================================
        transfer_patterns = [
            # Virements directs
            (r"(?:faire\s+un\s+)?virement\s+(?:de\s+)?(\d+(?:\.\d{2})?)\s*(?:euros?|€)", 0.94, "transfer_amount"),
            (r"virer\s+(\d+(?:\.\d{2})?)\s*(?:euros?|€)\s+(?:vers|à|sur)", 0.93, "transfer_to_amount"),
            (r"transférer\s+(\d+(?:\.\d{2})?)\s*(?:euros?|€)", 0.92, "transfer_verb_amount"),
            
            # Patterns génériques
            (r"(?:faire\s+un\s+)?(?:virement|transfert)", 0.88, "generic_transfer"),
            (r"envoyer\s+de\s+l'argent", 0.87, "send_money"),
            (r"virer\s+(?:de\s+l')?argent", 0.86, "wire_money"),
            
            # Patterns avec bénéficiaires
            (r"virement\s+vers\s+([a-zA-ZÀ-ÿ\s]+)", 0.91, "transfer_to_person"),
            (r"payer\s+([a-zA-ZÀ-ÿ\s]+)\s+(\d+(?:\.\d{2})?)", 0.90, "pay_person_amount")
        ]
        
        # ==========================================
        # PATTERNS EXPENSE_ANALYSIS
        # ==========================================
        expense_patterns = [
            # Analyses dépenses
            (r"(?:mes\s+)?dépenses\s+(?:de\s+|du\s+|en\s+)?([a-zA-Zà-ÿ]+)", 0.91, "expenses_category"),
            (r"combien\s+(?:ai-je|j'ai)\s+dépensé(?:\s+(?:en|pour|dans))?\s+([a-zA-Zà-ÿ]+)?", 0.89, "spent_how_much"),
            (r"(?:voir|analyser|consulter)\s+(?:mes\s+)?dépenses", 0.87, "view_expenses"),
            (r"budget\s+([a-zA-Zà-ÿ]+)", 0.85, "budget_category"),
            
            # Patterns temporels dépenses
            (r"dépenses\s+(?:de\s+)?(?:ce\s+)?(mois|trimestre|année)", 0.88, "expenses_period"),
            (r"combien\s+(?:ce\s+)?(mois|semaine)", 0.84, "spent_this_period")
        ]
        
        # ==========================================
        # PATTERNS CARD_MANAGEMENT
        # ==========================================
        card_patterns = [
            # Gestion cartes
            (r"(?:bloquer|suspendre)\s+(?:ma\s+)?carte", 0.95, "block_card"),
            (r"(?:activer|débloquer)\s+(?:ma\s+)?carte", 0.94, "activate_card"),
            (r"(?:changer|modifier)\s+(?:le\s+)?code\s+(?:de\s+)?(?:ma\s+)?carte", 0.92, "change_pin"),
            (r"limites?\s+(?:de\s+)?(?:ma\s+)?carte", 0.89, "card_limits"),
            (r"plafonds?\s+carte", 0.87, "card_ceiling"),
            
            # Opposition carte
            (r"opposition\s+carte", 0.93, "card_opposition"),
            (r"carte\s+(?:volée|perdue)", 0.91, "card_lost_stolen"),
            (r"signaler\s+(?:perte|vol)\s+carte", 0.90, "report_card_issue")
        ]
        
        # ==========================================
        # PATTERNS SYSTÈME (greetings, help)
        # ==========================================
        system_patterns = {
            IntentType.GREETING: [
                (r"^(?:bonjour|bonsoir|salut|hello|hey)", 0.96, "greeting"),
                (r"^(?:comment\s+(?:ça\s+)?va|ça\s+va)", 0.94, "how_are_you"),
                (r"^(?:re)?bonjour", 0.92, "hello_again")
            ],
            IntentType.HELP: [
                (r"(?:aide|aidez)(?:\s+moi)?", 0.95, "help_request"),
                (r"comment\s+(?:faire|procéder)", 0.93, "how_to"),
                (r"que\s+puis(?:\s+je|se)?\s+faire", 0.91, "what_can_do"),
                (r"fonctionnalités", 0.89, "features")
            ],
            IntentType.GOODBYE: [
                (r"(?:au\s+revoir|à\s+bientôt|bye|goodbye)", 0.96, "goodbye"),
                (r"(?:merci|thank you)(?:\s+et\s+au\s+revoir)?", 0.88, "thanks_goodbye")
            ]
        }
        
        # Compilation patterns principaux
        pattern_definitions = {
            IntentType.BALANCE_CHECK: balance_patterns,
            IntentType.TRANSFER: transfer_patterns,
            IntentType.EXPENSE_ANALYSIS: expense_patterns,
            IntentType.CARD_MANAGEMENT: card_patterns
        }
        
        # Ajout patterns système
        pattern_definitions.update(system_patterns)
        
        # Compilation effective
        compiled_count = 0
        for intent_type, patterns in pattern_definitions.items():
            compiled_patterns = []
            
            for pattern_text, confidence, pattern_name in patterns:
                try:
                    compiled_pattern = re.compile(pattern_text, re.IGNORECASE | re.UNICODE)
                    compiled_patterns.append(compiled_pattern)
                    
                    # Métadonnées pattern
                    pattern_key = f"{intent_type.value}_{pattern_name}"
                    self.pattern_metadata[pattern_key] = {
                        "intent": intent_type,
                        "confidence": confidence,
                        "name": pattern_name,
                        "regex": pattern_text
                    }
                    
                    compiled_count += 1
                    
                except re.error as e:
                    logger.warning(f"⚠️ Erreur compilation pattern {pattern_name}: {e}")
            
            self.compiled_patterns[intent_type] = compiled_patterns
        
        logger.info(f"✅ {compiled_count} patterns financiers compilés")
    
    def match_patterns(self, query: str) -> List[PatternMatch]:
        """
        Recherche matches patterns dans requête normalisée
        
        Args:
            query: Requête utilisateur normalisée
            
        Returns:
            List[PatternMatch]: Matches trouvés, triés par confiance
        """
        matches = []
        
        for intent_type, compiled_patterns in self.compiled_patterns.items():
            for i, pattern in enumerate(compiled_patterns):
                try:
                    match = pattern.search(query)
                    if match:
                        # Récupération métadonnées pattern
                        pattern_keys = [k for k in self.pattern_metadata.keys() 
                                      if k.startswith(intent_type.value)]
                        
                        if i < len(pattern_keys):
                            metadata = self.pattern_metadata[pattern_keys[i]]
                            
                            # Extraction entités basiques depuis groupes
                            entities = self._extract_entities_from_match(match, intent_type)
                            
                            pattern_match = PatternMatch(
                                intent_type=intent_type,
                                confidence=metadata["confidence"],
                                matched_text=match.group(0),
                                pattern_name=metadata["name"],
                                entities=entities,
                                match_groups=match.groupdict() if match.groups else {}
                            )
                            
                            matches.append(pattern_match)
                
                except Exception as e:
                    logger.debug(f"⚠️ Erreur match pattern {intent_type}: {e}")
        
        # Tri par confiance décroissante
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches
    
    def _extract_entities_from_match(self, match: re.Match, intent_type: IntentType) -> Dict[str, Any]:
        """Extraction entités basiques depuis match regex"""
        entities = {}
        
        try:
            # Entités montants (groupes numériques)
            if intent_type in [IntentType.TRANSFER, IntentType.EXPENSE_ANALYSIS]:
                for i, group in enumerate(match.groups()):
                    if group and re.match(r'^\d+(?:\.\d{2})?$', group):
                        entities["amount"] = float(group)
                        entities["amount_text"] = group
                        break
            
            # Entités temporelles
            if intent_type == IntentType.EXPENSE_ANALYSIS:
                temporal_keywords = ["mois", "semaine", "année", "trimestre"]
                for keyword in temporal_keywords:
                    if keyword in match.group(0).lower():
                        entities["time_period"] = keyword
                        break
            
            # Entités catégories (texte entre groupes)
            category_keywords = ["restaurant", "transport", "shopping", "épargne", "courant"]
            for keyword in category_keywords:
                if keyword in match.group(0).lower():
                    entities["category"] = keyword
                    break
        
        except Exception as e:
            logger.debug(f"⚠️ Erreur extraction entités: {e}")
        
        return entities

class PatternMatcher:
    """
    ⚡ Gestionnaire principal pattern matching L0
    
    Objectif: <10ms pour 85% des requêtes financières fréquentes
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.patterns = FinancialPatterns()
        
        # Métriques performance
        self._total_matches = 0
        self._successful_matches = 0
        self._cache_hits = 0
        self._average_latency = 0.0
        
        # Cache patterns fréquents en mémoire
        self._frequent_patterns_cache = {}
        self._max_frequent_cache_size = 100
        
        logger.info("⚡ Pattern Matcher L0 initialisé")
    
    async def initialize(self):
        """Initialisation avec préchargement patterns fréquents"""
        logger.info("🔧 Initialisation Pattern Matcher...")
        
        # Préchargement patterns fréquents (simulation)
        frequent_queries = [
            "solde compte", "mes dépenses", "virement", "bloquer carte",
            "bonjour", "aide", "combien j'ai", "dépenses restaurant"
        ]
        
        for query in frequent_queries:
            normalized = self._normalize_for_matching(query)
            pattern_hash = hash(normalized)
            
            # Pre-compute et cache
            matches = self.patterns.match_patterns(normalized)
            if matches:
                self._frequent_patterns_cache[pattern_hash] = matches[0]
        
        logger.info(f"✅ Pattern Matcher initialisé avec {len(self._frequent_patterns_cache)} patterns fréquents")
    
    async def match_intent(self, query: str) -> Optional[IntentResult]:
        """
        Match intention via patterns pré-compilés
        
        Args:
            query: Requête utilisateur normalisée
            
        Returns:
            IntentResult: Résultat match ou None si pas de match
        """
        start_time = time.time()
        self._total_matches += 1
        
        try:
            # 1. Vérification cache fréquents en mémoire
            normalized_query = self._normalize_for_matching(query)
            query_hash = hash(normalized_query)
            
            if query_hash in self._frequent_patterns_cache:
                cached_match = self._frequent_patterns_cache[query_hash]
                self._cache_hits += 1
                
                result = self._convert_match_to_result(cached_match, from_cache=True)
                result.latency_ms = (time.time() - start_time) * 1000
                
                return result
            
            # 2. Pattern matching standard
            matches = self.patterns.match_patterns(normalized_query)
            
            if not matches:
                return None
            
            # 3. Sélection meilleur match (premier = plus haute confiance)
            best_match = matches[0]
            
            # 4. Vérification seuil confiance L0
            if best_match.confidence < 0.85:
                return None
            
            # 5. Conversion en IntentResult
            result = self._convert_match_to_result(best_match, from_cache=False)
            result.latency_ms = (time.time() - start_time) * 1000
            
            # 6. Cache match fréquent si performance excellente
            if result.latency_ms < 5.0 and best_match.confidence > 0.90:
                self._cache_frequent_pattern(query_hash, best_match)
            
            self._successful_matches += 1
            self._update_average_latency(result.latency_ms)
            
            logger.debug(
                f"✅ Pattern match L0: {best_match.intent_type.value} "
                f"(confidence: {best_match.confidence:.2f}, "
                f"latency: {result.latency_ms:.1f}ms)"
            )
            
            return result
        
        except Exception as e:
            logger.warning(f"⚠️ Erreur pattern matching: {e}")
            return None
    
    def _normalize_for_matching(self, query: str) -> str:
        """Normalisation spécialisée pour pattern matching"""
        if not query:
            return ""
        
        # Nettoyage et normalisation
        normalized = query.lower().strip()
        
        # Suppression accents optionnelle pour certains patterns
        accent_map = {
            'à': 'a', 'â': 'a', 'ä': 'a', 'ç': 'c', 'é': 'e', 'è': 'e', 
            'ê': 'e', 'ë': 'e', 'î': 'i', 'ï': 'i', 'ô': 'o', 'ö': 'o',
            'ù': 'u', 'û': 'u', 'ü': 'u', 'ÿ': 'y', 'ñ': 'n'
        }
        
        # Application mapping accents seulement si nécessaire
        for accented, plain in accent_map.items():
            normalized = normalized.replace(accented, plain)
        
        # Nettoyage caractères spéciaux en préservant € et chiffres
        import re
        normalized = re.sub(r'[^\w\s\-\.€]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _convert_match_to_result(self, match: PatternMatch, from_cache: bool = False) -> IntentResult:
        """Conversion PatternMatch vers IntentResult"""
        confidence = IntentConfidence.from_pattern_match(
            match_strength=match.confidence,
            pattern_type=match.pattern_name
        )
        
        return IntentResult(
            intent_type=match.intent_type,
            confidence=confidence,
            level=IntentLevel.L0_PATTERN,
            latency_ms=0.0,  # Sera défini par appelant
            from_cache=from_cache,
            entities=match.entities,
            processing_details={
                "pattern_name": match.pattern_name,
                "matched_text": match.matched_text,
                "match_groups": match.match_groups
            }
        )
    
    def _cache_frequent_pattern(self, query_hash: int, match: PatternMatch):
        """Cache patterns fréquents avec éviction LRU"""
        if len(self._frequent_patterns_cache) >= self._max_frequent_cache_size:
            # Éviction simple : supprime premier élément
            first_key = next(iter(self._frequent_patterns_cache))
            del self._frequent_patterns_cache[first_key]
        
        self._frequent_patterns_cache[query_hash] = match
    
    def _update_average_latency(self, latency_ms: float):
        """Mise à jour latence moyenne avec smoothing"""
        if self._average_latency == 0.0:
            self._average_latency = latency_ms
        else:
            # Smoothing exponentiel (alpha = 0.1)
            self._average_latency = 0.9 * self._average_latency + 0.1 * latency_ms
    
    # ==========================================
    # MÉTRIQUES ET MONITORING
    # ==========================================
    
    def get_status(self) -> Dict[str, Any]:
        """Status et métriques Pattern Matcher"""
        success_rate = (
            self._successful_matches / max(1, self._total_matches)
        )
        cache_hit_rate = (
            self._cache_hits / max(1, self._total_matches)
        )
        
        return {
            "initialized": True,
            "total_patterns": sum(len(patterns) for patterns in self.patterns.compiled_patterns.values()),
            "total_matches_attempted": self._total_matches,
            "successful_matches": self._successful_matches,
            "success_rate": round(success_rate, 3),
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "average_latency_ms": round(self._average_latency, 2),
            "frequent_cache_size": len(self._frequent_patterns_cache),
            "frequent_cache_max": self._max_frequent_cache_size
        }
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Statistiques détaillées patterns par intention"""
        stats = {}
        
        for intent_type, patterns in self.patterns.compiled_patterns.items():
            pattern_count = len(patterns)
            
            # Compter métadonnées associées
            metadata_count = len([
                k for k in self.patterns.pattern_metadata.keys()
                if k.startswith(intent_type.value)
            ])
            
            stats[intent_type.value] = {
                "pattern_count": pattern_count,
                "metadata_entries": metadata_count,
                "avg_confidence": self._get_avg_confidence_for_intent(intent_type)
            }
        
        return stats
    
    def _get_avg_confidence_for_intent(self, intent_type: IntentType) -> float:
        """Confiance moyenne patterns pour une intention"""
        relevant_metadata = [
            metadata for key, metadata in self.patterns.pattern_metadata.items()
            if key.startswith(intent_type.value)
        ]
        
        if not relevant_metadata:
            return 0.0
        
        total_confidence = sum(m["confidence"] for m in relevant_metadata)
        return round(total_confidence / len(relevant_metadata), 3)
    
    # ==========================================
    # MÉTHODES DEBUG ET TESTING
    # ==========================================
    
    async def test_pattern(self, query: str, expected_intent: str = None) -> Dict[str, Any]:
        """Test pattern matching pour debug"""
        start_time = time.time()
        
        normalized = self._normalize_for_matching(query)
        matches = self.patterns.match_patterns(normalized)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            "original_query": query,
            "normalized_query": normalized,
            "matches_found": len(matches),
            "latency_ms": round(latency_ms, 2),
            "expected_intent": expected_intent
        }
        
        if matches:
            best_match = matches[0]
            result["best_match"] = {
                "intent": best_match.intent_type.value,
                "confidence": best_match.confidence,
                "pattern_name": best_match.pattern_name,
                "matched_text": best_match.matched_text,
                "entities": best_match.entities
            }
            
            result["all_matches"] = [
                {
                    "intent": m.intent_type.value,
                    "confidence": m.confidence,
                    "pattern": m.pattern_name
                }
                for m in matches[:5]  # Top 5
            ]
            
            # Vérification attente si fournie
            if expected_intent:
                result["expectation_met"] = (best_match.intent_type.value == expected_intent)
        
        return result
    
    async def benchmark_patterns(self, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark performance patterns sur liste de requêtes"""
        logger.info(f"🏁 Benchmark patterns sur {len(test_queries)} requêtes...")
        
        start_time = time.time()
        results = []
        successful_matches = 0
        total_latency = 0.0
        
        for query in test_queries:
            query_start = time.time()
            
            try:
                intent_result = await self.match_intent(query)
                query_latency = (time.time() - query_start) * 1000
                
                if intent_result:
                    successful_matches += 1
                    results.append({
                        "query": query,
                        "intent": intent_result.intent_type.value,
                        "confidence": intent_result.confidence.score,
                        "latency_ms": query_latency,
                        "success": True
                    })
                else:
                    results.append({
                        "query": query,
                        "intent": None,
                        "confidence": 0.0,
                        "latency_ms": query_latency,
                        "success": False
                    })
                
                total_latency += query_latency
                
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        total_time = (time.time() - start_time) * 1000
        
        benchmark_results = {
            "total_queries": len(test_queries),
            "successful_matches": successful_matches,
            "success_rate": successful_matches / len(test_queries),
            "total_time_ms": round(total_time, 2),
            "average_latency_ms": round(total_latency / len(test_queries), 2),
            "queries_per_second": round(len(test_queries) / (total_time / 1000), 1),
            "target_latency_met": (total_latency / len(test_queries)) < 10.0,
            "results": results[:10]  # Échantillon pour éviter spam
        }
        
        logger.info(f"🏁 Benchmark terminé - Success rate: {benchmark_results['success_rate']:.1%}")
        return benchmark_results
    
    def add_dynamic_pattern(self, intent_type: IntentType, pattern_text: str, confidence: float, name: str):
        """Ajout pattern dynamique en runtime"""
        try:
            compiled_pattern = re.compile(pattern_text, re.IGNORECASE | re.UNICODE)
            
            # Ajout au dictionnaire patterns
            if intent_type not in self.patterns.compiled_patterns:
                self.patterns.compiled_patterns[intent_type] = []
            
            self.patterns.compiled_patterns[intent_type].append(compiled_pattern)
            
            # Ajout métadonnées
            pattern_key = f"{intent_type.value}_{name}"
            self.patterns.pattern_metadata[pattern_key] = {
                "intent": intent_type,
                "confidence": confidence,
                "name": name,
                "regex": pattern_text,
                "dynamic": True
            }
            
            logger.info(f"➕ Pattern dynamique ajouté: {intent_type.value} - {name}")
            return True
            
        except re.error as e:
            logger.warning(f"⚠️ Erreur pattern dynamique {name}: {e}")
            return False
    
    def get_pattern_coverage_report(self, test_queries: List[str]) -> Dict[str, Any]:
        """Rapport couverture patterns sur requêtes test"""
        intent_coverage = {}
        pattern_usage = {}
        
        for query in test_queries:
            normalized = self._normalize_for_matching(query)
            matches = self.patterns.match_patterns(normalized)
            
            if matches:
                best_match = matches[0]
                intent = best_match.intent_type.value
                pattern = best_match.pattern_name
                
                # Couverture intentions
                intent_coverage[intent] = intent_coverage.get(intent, 0) + 1
                
                # Usage patterns
                pattern_key = f"{intent}_{pattern}"
                pattern_usage[pattern_key] = pattern_usage.get(pattern_key, 0) + 1
        
        return {
            "total_test_queries": len(test_queries),
            "covered_queries": sum(intent_coverage.values()),
            "coverage_rate": sum(intent_coverage.values()) / len(test_queries),
            "intent_distribution": intent_coverage,
            "pattern_usage": dict(sorted(pattern_usage.items(), key=lambda x: x[1], reverse=True)[:20]),
            "unused_intents": [
                intent.value for intent in IntentType
                if intent.value not in intent_coverage
            ]
        }
    
    async def shutdown(self):
        """Arrêt propre Pattern Matcher"""
        logger.info("🛑 Arrêt Pattern Matcher...")
        
        # Sauvegarde statistiques finales
        final_stats = self.get_status()
        logger.info(f"📊 Stats finales Pattern Matcher: "
                   f"Success rate = {final_stats['success_rate']:.1%}, "
                   f"Avg latency = {final_stats['average_latency_ms']:.1f}ms")
        
        # Clear caches
        self._frequent_patterns_cache.clear()
        
        logger.info("✅ Pattern Matcher arrêté")


# ==========================================
# UTILITAIRES ET HELPERS
# ==========================================

def create_test_queries() -> List[str]:
    """Génère liste requêtes test pour validation patterns"""
    return [
        # Balance checks
        "quel est mon solde",
        "combien j'ai sur mon compte",
        "solde compte courant",
        "voir mes comptes",
        "position de mon compte",
        
        # Transfers
        "faire un virement de 100 euros",
        "virer 50€ vers Paul",
        "transférer 200 euros",
        "envoyer de l'argent",
        
        # Expenses
        "mes dépenses restaurant",
        "combien j'ai dépensé ce mois",
        "voir mes dépenses",
        "budget transport",
        "dépenses de cette semaine",
        
        # Card management
        "bloquer ma carte",
        "activer carte",
        "changer code carte",
        "limites de carte",
        "opposition carte",
        
        # System
        "bonjour",
        "aide",
        "comment faire",
        "au revoir",
        "merci",
        
        # Edge cases
        "solde compte épargne maintenant",
        "virement urgent 1000€ vers Marie",
        "dépenses shopping cette année",
        "bloquer carte volée immédiatement"
    ]

async def validate_pattern_performance(pattern_matcher: PatternMatcher) -> Dict[str, Any]:
    """Validation performance patterns selon targets L0"""
    test_queries = create_test_queries()
    
    # Benchmark performance
    benchmark_results = await pattern_matcher.benchmark_patterns(test_queries)
    
    # Validation targets L0
    target_latency = 10.0  # ms
    target_success_rate = 0.85
    
    validation = {
        "performance_validation": {
            "target_latency_ms": target_latency,
            "actual_avg_latency_ms": benchmark_results["average_latency_ms"],
            "latency_target_met": benchmark_results["average_latency_ms"] < target_latency,
            
            "target_success_rate": target_success_rate,
            "actual_success_rate": benchmark_results["success_rate"],
            "success_rate_target_met": benchmark_results["success_rate"] >= target_success_rate
        },
        "benchmark_summary": benchmark_results,
        "pattern_stats": pattern_matcher.get_pattern_stats()
    }
    
    return validation