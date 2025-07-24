"""
⚡ Niveau L0 - Pattern Matcher ultra-rapide - PHASE 1

Reconnaissance patterns financiers fréquents avec regex pré-compilés
pour objectif performance <10ms sur 85% des requêtes.

Version Phase 1 : Focus sur ~60 patterns essentiels avec cache intelligent
et intégration directe aux nouveaux modèles Pydantic.
"""

import re
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Import des nouveaux modèles Phase 1
from conversation_service.models.conversation_models import (
    FinancialIntent, PatternType, PatternMatch, FinancialEntity,
    L0PerformanceMetrics, create_l0_success_response, create_l0_error_response
)
from conversation_service.utils.logging import log_intent_detection, log_performance_metric

logger = logging.getLogger(__name__)

# ==========================================
# PATTERNS FINANCIERS OPTIMISÉS PHASE 1
# ==========================================

class FinancialPatterns:
    """
    📚 Bibliothèque patterns financiers Phase 1 - 60 patterns essentiels
    
    Focus sur performance <10ms avec patterns optimisés français financier
    et extraction entités intelligente.
    """
    
    def __init__(self):
        self.compiled_patterns: Dict[FinancialIntent, List[re.Pattern]] = {}
        self.pattern_metadata: Dict[str, Dict[str, Any]] = {}
        self.pattern_count = 0
        self._compile_all_patterns()
    
    def _compile_all_patterns(self):
        """Compilation tous les patterns au démarrage pour performance"""
        logger.info("🔧 Compilation patterns financiers Phase 1...")
        
        # Définition patterns par intention
        pattern_definitions = {
            FinancialIntent.BALANCE_CHECK: self._get_balance_patterns(),
            FinancialIntent.TRANSFER: self._get_transfer_patterns(),
            FinancialIntent.EXPENSE_ANALYSIS: self._get_expense_patterns(),
            FinancialIntent.CARD_MANAGEMENT: self._get_card_patterns(),
            FinancialIntent.GREETING: self._get_greeting_patterns(),
            FinancialIntent.HELP: self._get_help_patterns(),
            FinancialIntent.GOODBYE: self._get_goodbye_patterns()
        }
        
        # Compilation effective
        compiled_count = 0
        for intent, patterns in pattern_definitions.items():
            compiled_patterns = []
            
            for pattern_data in patterns:
                try:
                    pattern_text = pattern_data["regex"]
                    compiled_pattern = re.compile(pattern_text, re.IGNORECASE | re.UNICODE)
                    compiled_patterns.append(compiled_pattern)
                    
                    # Métadonnées pattern avec ID unique
                    pattern_id = f"{intent.value}_{pattern_data['name']}"
                    self.pattern_metadata[pattern_id] = {
                        "intent": intent,
                        "confidence": pattern_data["confidence"],
                        "name": pattern_data["name"],
                        "pattern_type": pattern_data["type"],
                        "regex": pattern_text,
                        "entities_extractable": pattern_data.get("entities", []),
                        "priority": pattern_data.get("priority", 0)
                    }
                    
                    compiled_count += 1
                    
                except re.error as e:
                    logger.warning(f"⚠️ Erreur compilation pattern {pattern_data['name']}: {e}")
            
            self.compiled_patterns[intent] = compiled_patterns
        
        self.pattern_count = compiled_count
        logger.info(f"✅ {compiled_count} patterns financiers Phase 1 compilés")
    
    def _get_balance_patterns(self) -> List[Dict[str, Any]]:
        """Patterns BALANCE_CHECK - 15 patterns"""
        return [
            # Patterns directs haute confiance
            {
                "name": "direct_balance",
                "regex": r"^solde$",
                "confidence": 0.98,
                "type": PatternType.DIRECT_KEYWORD,
                "priority": 1
            },
            {
                "name": "question_balance",
                "regex": r"(?:quel|combien)\s+(?:est\s+)?(?:le\s+)?(?:mon\s+)?solde",
                "confidence": 0.96,
                "type": PatternType.QUESTION_PHRASE,
                "priority": 1
            },
            {
                "name": "account_balance",
                "regex": r"solde\s+(?:de\s+)?(?:mon\s+)?compte(?:\s+(courant|épargne|livret))?",
                "confidence": 0.94,
                "type": PatternType.CATEGORY_SPECIFIC,
                "entities": ["account_type"],
                "priority": 1
            },
            {
                "name": "how_much_money",
                "regex": r"combien\s+(?:ai-je|j'ai)\s+(?:sur\s+)?(?:mon\s+)?compte",
                "confidence": 0.92,
                "type": PatternType.QUESTION_PHRASE,
                "priority": 1
            },
            {
                "name": "available_money",
                "regex": r"(?:mon\s+)?argent\s+(?:disponible|restant)",
                "confidence": 0.90,
                "type": PatternType.DIRECT_KEYWORD,
                "priority": 2
            },
            {
                "name": "current_balance",
                "regex": r"solde\s+(?:actuel|maintenant|aujourd'hui)",
                "confidence": 0.91,
                "type": PatternType.TEMPORAL_CONTEXT,
                "entities": ["time_context"],
                "priority": 2
            },
            {
                "name": "position_account",
                "regex": r"position\s+(?:de\s+)?(?:mon\s+)?compte",
                "confidence": 0.88,
                "type": PatternType.DIRECT_KEYWORD,
                "priority": 2
            },
            {
                "name": "view_accounts",
                "regex": r"(?:voir|consulter|afficher)\s+(?:mes\s+)?comptes?",
                "confidence": 0.87,
                "type": PatternType.ACTION_VERB,
                "priority": 2
            },
            {
                "name": "check_balance",
                "regex": r"(?:vérifier|checker)\s+(?:mon\s+)?solde",
                "confidence": 0.89,
                "type": PatternType.ACTION_VERB,
                "priority": 2
            },
            {
                "name": "remaining_money",
                "regex": r"combien\s+(?:me\s+reste|reste)(?:\s+t)?(?:\s+il)?",
                "confidence": 0.85,
                "type": PatternType.QUESTION_PHRASE,
                "priority": 3
            }
        ]
    
    def _get_transfer_patterns(self) -> List[Dict[str, Any]]:
        """Patterns TRANSFER - 12 patterns"""
        return [
            # Virements avec montants
            {
                "name": "transfer_amount",
                "regex": r"(?:faire\s+un\s+)?virement\s+(?:de\s+)?(\d+(?:[,\.]\d{1,2})?)\s*(?:euros?|€|eur)",
                "confidence": 0.96,
                "type": PatternType.AMOUNT_EXTRACTION,
                "entities": ["amount", "currency"],
                "priority": 1
            },
            {
                "name": "wire_amount",
                "regex": r"virer\s+(\d+(?:[,\.]\d{1,2})?)\s*(?:euros?|€|eur)(?:\s+(?:vers|à|sur))?",
                "confidence": 0.95,
                "type": PatternType.AMOUNT_EXTRACTION,
                "entities": ["amount", "currency"],
                "priority": 1
            },
            {
                "name": "transfer_verb_amount",
                "regex": r"transférer\s+(\d+(?:[,\.]\d{1,2})?)\s*(?:euros?|€|eur)",
                "confidence": 0.94,
                "type": PatternType.AMOUNT_EXTRACTION,
                "entities": ["amount", "currency"],
                "priority": 1
            },
            {
                "name": "pay_amount",
                "regex": r"payer\s+(\d+(?:[,\.]\d{1,2})?)\s*(?:euros?|€|eur)",
                "confidence": 0.93,
                "type": PatternType.AMOUNT_EXTRACTION,
                "entities": ["amount", "currency"],
                "priority": 1
            },
            # Virements génériques
            {
                "name": "generic_transfer",
                "regex": r"(?:faire\s+un\s+)?(?:virement|transfert)(?:\s+bancaire)?",
                "confidence": 0.90,
                "type": PatternType.DIRECT_KEYWORD,
                "priority": 2
            },
            {
                "name": "send_money",
                "regex": r"envoyer\s+de\s+l'argent",
                "confidence": 0.89,
                "type": PatternType.ACTION_VERB,
                "priority": 2
            },
            {
                "name": "wire_money",
                "regex": r"virer\s+(?:de\s+l')?argent",
                "confidence": 0.87,
                "type": PatternType.ACTION_VERB,
                "priority": 2
            },
            # Virements avec bénéficiaires
            {
                "name": "transfer_to_person",
                "regex": r"(?:virement|virer|transférer|envoyer)\s+(?:vers|à|pour)\s+([a-zA-ZÀ-ÿ\s]{2,20})",
                "confidence": 0.92,
                "type": PatternType.CATEGORY_SPECIFIC,
                "entities": ["beneficiary"],
                "priority": 1
            },
            {
                "name": "pay_person",
                "regex": r"payer\s+([a-zA-ZÀ-ÿ\s]{2,20})",
                "confidence": 0.88,
                "type": PatternType.CATEGORY_SPECIFIC,
                "entities": ["beneficiary"],
                "priority": 2
            }
        ]
    
    def _get_expense_patterns(self) -> List[Dict[str, Any]]:
        """Patterns EXPENSE_ANALYSIS - 10 patterns"""
        return [
            {
                "name": "expenses_category",
                "regex": r"(?:mes\s+)?dépenses\s+(?:de\s+|du\s+|en\s+|pour\s+)?([a-zA-Zà-ÿ]+)",
                "confidence": 0.93,
                "type": PatternType.CATEGORY_SPECIFIC,
                "entities": ["category"],
                "priority": 1
            },
            {
                "name": "spent_how_much",
                "regex": r"combien\s+(?:ai-je|j'ai)\s+dépensé(?:\s+(?:en|pour|dans|ce))?\s*([a-zA-Zà-ÿ]*)",
                "confidence": 0.91,
                "type": PatternType.QUESTION_PHRASE,
                "entities": ["category"],
                "priority": 1
            },
            {
                "name": "view_expenses",
                "regex": r"(?:voir|analyser|consulter|afficher)\s+(?:mes\s+)?dépenses",
                "confidence": 0.89,
                "type": PatternType.ACTION_VERB,
                "priority": 2
            },
            {
                "name": "budget_category",
                "regex": r"budget\s+([a-zA-Zà-ÿ]+)",
                "confidence": 0.87,
                "type": PatternType.CATEGORY_SPECIFIC,
                "entities": ["category"],
                "priority": 2
            },
            {
                "name": "expenses_period",
                "regex": r"dépenses\s+(?:de\s+)?(?:ce\s+|cette\s+)?(mois|semaine|année|trimestre)",
                "confidence": 0.90,
                "type": PatternType.TEMPORAL_CONTEXT,
                "entities": ["time_period"],
                "priority": 1
            },
            {
                "name": "spent_period",
                "regex": r"combien\s+(?:ce\s+|cette\s+)?(mois|semaine|année)",
                "confidence": 0.86,
                "type": PatternType.TEMPORAL_CONTEXT,
                "entities": ["time_period"],
                "priority": 2
            },
            {
                "name": "monthly_expenses",
                "regex": r"dépenses\s+mensuelles?",
                "confidence": 0.88,
                "type": PatternType.TEMPORAL_CONTEXT,
                "entities": ["time_period"],
                "priority": 2
            }
        ]
    
    def _get_card_patterns(self) -> List[Dict[str, Any]]:
        """Patterns CARD_MANAGEMENT - 8 patterns"""
        return [
            {
                "name": "block_card",
                "regex": r"(?:bloquer|suspendre|désactiver)\s+(?:ma\s+)?carte",
                "confidence": 0.97,
                "type": PatternType.ACTION_VERB,
                "entities": ["action"],
                "priority": 1
            },
            {
                "name": "activate_card",
                "regex": r"(?:activer|débloquer|réactiver)\s+(?:ma\s+)?carte",
                "confidence": 0.96,
                "type": PatternType.ACTION_VERB,
                "entities": ["action"],
                "priority": 1
            },
            {
                "name": "change_pin",
                "regex": r"(?:changer|modifier)\s+(?:le\s+)?code\s+(?:de\s+)?(?:ma\s+)?carte",
                "confidence": 0.94,
                "type": PatternType.ACTION_VERB,
                "entities": ["action"],
                "priority": 1
            },
            {
                "name": "card_limits",
                "regex": r"limites?\s+(?:de\s+)?(?:ma\s+)?carte",
                "confidence": 0.91,
                "type": PatternType.DIRECT_KEYWORD,
                "priority": 2
            },
            {
                "name": "card_opposition",
                "regex": r"opposition\s+carte",
                "confidence": 0.95,
                "type": PatternType.DIRECT_KEYWORD,
                "entities": ["action"],
                "priority": 1
            },
            {
                "name": "lost_stolen_card",
                "regex": r"carte\s+(?:volée|perdue|disparue)",
                "confidence": 0.93,
                "type": PatternType.CATEGORY_SPECIFIC,
                "entities": ["issue_type"],
                "priority": 1
            }
        ]
    
    def _get_greeting_patterns(self) -> List[Dict[str, Any]]:
        """Patterns GREETING - 5 patterns"""
        return [
            {
                "name": "simple_greeting",
                "regex": r"^(?:bonjour|bonsoir|salut|hello|hey)(?:\s+.*)?$",
                "confidence": 0.98,
                "type": PatternType.GREETING_SYSTEM,
                "priority": 1
            },
            {
                "name": "how_are_you",
                "regex": r"(?:comment\s+(?:ça\s+)?va|ça\s+va|comment\s+allez\s+vous)",
                "confidence": 0.95,
                "type": PatternType.GREETING_SYSTEM,
                "priority": 1
            },
            {
                "name": "good_morning",
                "regex": r"(?:bon|bonne)\s+(?:matin|matinée|journée)",
                "confidence": 0.96,
                "type": PatternType.GREETING_SYSTEM,
                "priority": 1
            }
        ]
    
    def _get_help_patterns(self) -> List[Dict[str, Any]]:
        """Patterns HELP - 5 patterns"""
        return [
            {
                "name": "help_request",
                "regex": r"(?:aide|aidez|help)(?:\s+moi)?",
                "confidence": 0.97,
                "type": PatternType.GREETING_SYSTEM,
                "priority": 1
            },
            {
                "name": "how_to",
                "regex": r"comment\s+(?:faire|procéder|utiliser)",
                "confidence": 0.94,
                "type": PatternType.QUESTION_PHRASE,
                "priority": 1
            },
            {
                "name": "what_can_do",
                "regex": r"que\s+(?:puis|peux)(?:\s+je|se)?\s+faire",
                "confidence": 0.92,
                "type": PatternType.QUESTION_PHRASE,
                "priority": 1
            },
            {
                "name": "features",
                "regex": r"(?:fonctionnalités|services|options)",
                "confidence": 0.90,
                "type": PatternType.DIRECT_KEYWORD,
                "priority": 2
            }
        ]
    
    def _get_goodbye_patterns(self) -> List[Dict[str, Any]]:
        """Patterns GOODBYE - 3 patterns"""
        return [
            {
                "name": "goodbye",
                "regex": r"(?:au\s+revoir|à\s+bientôt|bye|goodbye|ciao)",
                "confidence": 0.98,
                "type": PatternType.GREETING_SYSTEM,
                "priority": 1
            },
            {
                "name": "thanks_goodbye",
                "regex": r"(?:merci|thank\s+you)(?:\s+(?:et\s+)?(?:au\s+revoir|bye))?",
                "confidence": 0.90,
                "type": PatternType.GREETING_SYSTEM,
                "priority": 2
            }
        ]
    
    def match_patterns(self, query: str) -> List[PatternMatch]:
        """
        Recherche matches patterns dans requête normalisée
        
        Args:
            query: Requête utilisateur normalisée
            
        Returns:
            List[PatternMatch]: Matches trouvés, triés par confiance
        """
        matches = []
        
        for intent, compiled_patterns in self.compiled_patterns.items():
            for i, pattern in enumerate(compiled_patterns):
                try:
                    match = pattern.search(query)
                    if match:
                        # Récupération métadonnées pattern
                        pattern_keys = [k for k in self.pattern_metadata.keys() 
                                      if k.startswith(intent.value)]
                        
                        if i < len(pattern_keys):
                            metadata = self.pattern_metadata[pattern_keys[i]]
                            
                            # Extraction entités depuis match
                            entities = self._extract_entities_from_match(match, metadata)
                            
                            # Calcul position match
                            position = {
                                "start": match.start(),
                                "end": match.end()
                            }
                            
                            pattern_match = PatternMatch(
                                pattern_name=metadata["name"],
                                pattern_type=metadata["pattern_type"],
                                confidence=metadata["confidence"],
                                matched_text=match.group(0),
                                position=position,
                                entities=entities
                            )
                            
                            matches.append(pattern_match)
                
                except Exception as e:
                    logger.debug(f"⚠️ Erreur match pattern {intent}: {e}")
        
        # Tri par confiance décroissante puis par priorité
        matches.sort(key=lambda m: (m.confidence, -self._get_pattern_priority(m.pattern_name)), reverse=True)
        return matches
    
    def _extract_entities_from_match(self, match: re.Match, metadata: Dict[str, Any]) -> List[FinancialEntity]:
        """Extraction entités intelligente depuis match regex"""
        entities = []
        
        try:
            # Entités extractibles définies dans métadonnées
            extractable_entities = metadata.get("entities_extractable", [])
            
            # Extraction groupes regex
            if match.groups():
                for i, group in enumerate(match.groups()):
                    if group and group.strip():
                        entity_type = self._determine_entity_type(group, extractable_entities, i)
                        if entity_type:
                            entity = FinancialEntity(
                                type=entity_type,
                                value=group.strip(),
                                confidence=0.9,
                                position={"start": match.start(i+1), "end": match.end(i+1)},
                                extraction_method="regex_group",
                                normalized_value=self._normalize_entity_value(entity_type, group)
                            )
                            entities.append(entity)
            
            # Extraction entités spéciales (montants, dates, etc.)
            entities.extend(self._extract_special_entities(match.group(0)))
            
        except Exception as e:
            logger.debug(f"⚠️ Erreur extraction entités: {e}")
        
        return entities
    
    def _determine_entity_type(self, value: str, extractable: List[str], group_index: int) -> Optional[str]:
        """Détermine le type d'entité basé sur valeur et contexte"""
        value_lower = value.lower().strip()
        
        # Montants
        if re.match(r'^\d+([,\.]\d{1,2})?$', value):
            return "amount"
        
        # Devises
        if value_lower in ["eur", "euro", "euros", "€"]:
            return "currency"
        
        # Types de comptes
        if value_lower in ["courant", "épargne", "livret", "joint"]:
            return "account_type"
        
        # Catégories dépenses
        expense_categories = ["restaurant", "transport", "shopping", "alimentaire", "essence", "loisirs"]
        if value_lower in expense_categories:
            return "category"
        
        # Périodes temporelles
        if value_lower in ["mois", "semaine", "année", "trimestre"]:
            return "time_period"
        
        # Actions carte
        if value_lower in ["bloquer", "activer", "désactiver"]:
            return "action"
        
        # Bénéficiaires (noms propres probables)
        if len(value) > 2 and value[0].isupper():
            return "beneficiary"
        
        # Fallback selon extractable
        if extractable and group_index < len(extractable):
            return extractable[group_index]
        
        return "text"
    
    def _extract_special_entities(self, text: str) -> List[FinancialEntity]:
        """Extraction entités spéciales (montants, dates, etc.)"""
        entities = []
        
        # Montants avec devises
        amount_pattern = r'(\d+(?:[,\.]\d{1,2})?)\s*(euros?|€|eur)'
        for match in re.finditer(amount_pattern, text.lower()):
            amount_text = match.group(1).replace(',', '.')
            currency = match.group(2)
            
            entities.append(FinancialEntity(
                type="amount",
                value=match.group(0),
                confidence=0.95,
                position={"start": match.start(), "end": match.end()},
                extraction_method="regex_special",
                normalized_value=float(amount_text),
                currency="EUR" if currency in ["euro", "euros", "€", "eur"] else currency.upper()
            ))
        
        return entities
    
    def _normalize_entity_value(self, entity_type: str, value: str) -> Union[str, int, float]:
        """Normalisation valeur entité selon type"""
        try:
            if entity_type == "amount":
                return float(value.replace(',', '.'))
            elif entity_type == "currency":
                currency_map = {"euro": "EUR", "euros": "EUR", "€": "EUR", "eur": "EUR"}
                return currency_map.get(value.lower(), value.upper())
            else:
                return value.strip().lower()
        except:
            return value
    
    def _get_pattern_priority(self, pattern_name: str) -> int:
        """Récupère priorité pattern pour tri"""
        for metadata in self.pattern_metadata.values():
            if metadata["name"] == pattern_name:
                return metadata.get("priority", 0)
        return 0

# ==========================================
# PATTERN MATCHER PRINCIPAL PHASE 1
# ==========================================

class PatternMatcher:
    """
    ⚡ Gestionnaire principal pattern matching L0 - Phase 1
    
    Objectif: <10ms pour 85% des requêtes financières fréquentes
    avec cache intelligent et métriques détaillées.
    """
    
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager  # Optionnel en Phase 1
        self.patterns = FinancialPatterns()
        
        # Métriques performance L0 spécialisées
        self.metrics = L0PerformanceMetrics()
        
        # Cache patterns fréquents en mémoire (simple dict)
        self._pattern_cache = {}
        self._cache_max_size = 1000
        
        # Statistiques usage patterns
        self._pattern_usage_stats = {}
        self._pattern_latency_stats = {}
        
        logger.info(f"⚡ Pattern Matcher L0 initialisé - {self.patterns.pattern_count} patterns")
    
    async def initialize(self):
        """Initialisation avec préchargement patterns fréquents"""
        logger.info("🔧 Initialisation Pattern Matcher Phase 1...")
        
        # Préchargement requêtes fréquentes pour cache
        frequent_queries = [
            "solde", "mes dépenses", "virement", "bloquer carte", "bonjour", 
            "aide", "combien j'ai", "dépenses restaurant", "virer 100€",
            "quel est mon solde", "faire un virement", "activer carte"
        ]
        
        for query in frequent_queries:
            normalized = self._normalize_query(query)
            cache_key = self._generate_cache_key(normalized, "preload")
            
            # Pre-compute matches
            matches = self.patterns.match_patterns(normalized)
            if matches:
                self._pattern_cache[cache_key] = matches[0]
        
        # Log métriques initialisation
        log_intent_detection(
            "pattern_matcher_initialized",
            level="L0_PATTERN",
            message=f"Pattern Matcher initialisé avec {len(self._pattern_cache)} patterns pré-chargés"
        )
        
        logger.info(f"✅ Pattern Matcher initialisé - Cache: {len(self._pattern_cache)} patterns")
    
    async def match_intent(self, query: str, user_id: str = "anonymous") -> Optional[PatternMatch]:
        """
        Match intention via patterns pré-compilés - Méthode principale Phase 1
        
        Args:
            query: Requête utilisateur
            user_id: ID utilisateur pour métriques
            
        Returns:
            PatternMatch: Meilleur match ou None si aucun match
        """
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            # 1. Normalisation requête
            normalized_query = self._normalize_query(query)
            if not normalized_query:
                return None
            
            # 2. Vérification cache
            cache_key = self._generate_cache_key(normalized_query, user_id)
            cached_match = self._get_from_cache(cache_key)
            
            if cached_match:
                self.metrics.cache_hit_rate = self._update_cache_rate(True)
                processing_time = (time.time() - start_time) * 1000
                
                log_intent_detection(
                    "l0_cache_hit",
                    level="L0_PATTERN",
                    intent=cached_match.pattern_name,
                    confidence=cached_match.confidence,
                    latency_ms=processing_time,
                    cache_hit=True,
                    user_id=user_id
                )
                
                return cached_match
            
            # 3. Pattern matching
            matches = self.patterns.match_patterns(normalized_query)
            processing_time = (time.time() - start_time) * 1000
            
            if not matches:
                self.metrics.l0_failed_requests += 1
                self._update_avg_latency(processing_time)
                
                log_intent_detection(
                    "l0_no_match",
                    level="L0_PATTERN",
                    latency_ms=processing_time,
                    user_id=user_id,
                    message_preview=query[:50]
                )
                
                return None
            
            # 4. Sélection meilleur match
            best_match = matches[0]
            
            # 5. Vérification seuil confiance L0 (0.85)
            if best_match.confidence < 0.85:
                self.metrics.l0_failed_requests += 1
                self._update_avg_latency(processing_time)
                
                log_intent_detection(
                    "l0_low_confidence",
                    level="L0_PATTERN",
                    intent=best_match.pattern_name,
                    confidence=best_match.confidence,
                    latency_ms=processing_time,
                    user_id=user_id
                )
                
                return None
            
            # 6. Succès L0
            self.metrics.l0_successful_requests += 1
            self._update_avg_latency(processing_time)
            self._update_pattern_stats(best_match.pattern_name, processing_time, True)
            
            # 7. Cache pour usage futur
            self._add_to_cache(cache_key, best_match)
            
            # 8. Logging succès
            log_intent_detection(
                "l0_success",
                level="L0_PATTERN",
                intent=best_match.pattern_name,
                confidence=best_match.confidence,
                latency_ms=processing_time,
                cache_hit=False,
                user_id=user_id,
                matched_text=best_match.matched_text
            )
            
            # 9. Métriques performance
            log_performance_metric(
                "l0_pattern_latency",
                processing_time,
                unit="ms",
                component="pattern_matcher",
                level="L0",
                threshold=10.0,
                threshold_exceeded=processing_time > 10.0
            )
            
            return best_match
        
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.metrics.l0_failed_requests += 1
            
            logger.error(f"❌ Erreur pattern matching L0: {e}")
            log_intent_detection(
                "l0_error",
                level="L0_PATTERN",
                latency_ms=processing_time,
                user_id=user_id,
                error=str(e)
            )
            
            return None
    
    def _normalize_query(self, query: str) -> str:
        """Normalisation spécialisée pour pattern matching Phase 1"""
        if not query:
            return ""
        
        # Nettoyage de base
        normalized = query.lower().strip()
        
        # Normalisation accents français
        accent_replacements = {
            'à': 'a', 'â': 'a', 'ä': 'a', 'ç': 'c', 'é': 'e', 'è': 'e', 
            'ê': 'e', 'ë': 'e', 'î': 'i', 'ï': 'i', 'ô': 'o', 'ö': 'o',
            'ù': 'u', 'û': 'u', 'ü': 'u', 'ÿ': 'y', 'ñ': 'n'
        }
        
        for accented, plain in accent_replacements.items():
            normalized = normalized.replace(accented, plain)
        
        # Nettoyage caractères spéciaux (garde €, chiffres, lettres, espaces, points, virgules)
        import re
        normalized = re.sub(r'[^\w\s\-\.€,]', ' ', normalized)
        
        # Normalisation espaces multiples
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _generate_cache_key(self, normalized_query: str, user_id: str) -> str:
        """Génère clé cache unique et courte"""
        combined = f"{normalized_query}_{user_id}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()[:12]
    
    def _get_from_cache(self, cache_key: str) -> Optional[PatternMatch]:
        """Récupération depuis cache mémoire"""
        return self._pattern_cache.get(cache_key)
    
    def _add_to_cache(self, cache_key: str, match: PatternMatch):
        """Ajout au cache avec éviction LRU simple"""
        if len(self._pattern_cache) >= self._cache_max_size:
            # Éviction simple : supprime le plus ancien
            oldest_key = next(iter(self._pattern_cache))
            del self._pattern_cache[oldest_key]
        
        self._pattern_cache[cache_key] = match
    
    def _update_cache_rate(self, hit: bool) -> float:
        """Mise à jour taux cache hit"""
        if hit:
            self.metrics.cache_hit_rate = (
                self.metrics.cache_hit_rate * 0.9 + 0.1
            )
        else:
            self.metrics.cache_hit_rate = (
                self.metrics.cache_hit_rate * 0.9
            )
        return self.metrics.cache_hit_rate
    
    def _update_avg_latency(self, latency_ms: float):
        """Mise à jour latence moyenne avec smoothing"""
        if self.metrics.avg_l0_latency_ms == 0.0:
            self.metrics.avg_l0_latency_ms = latency_ms
        else:
            # Smoothing exponentiel
            self.metrics.avg_l0_latency_ms = (
                0.9 * self.metrics.avg_l0_latency_ms + 0.1 * latency_ms
            )
    
    def _update_pattern_stats(self, pattern_name: str, latency_ms: float, success: bool):
        """Mise à jour statistiques par pattern"""
        # Usage count
        self.metrics.pattern_usage[pattern_name] = (
            self.metrics.pattern_usage.get(pattern_name, 0) + 1
        )
        
        # Success rate
        if pattern_name not in self.metrics.pattern_success_rate:
            self.metrics.pattern_success_rate[pattern_name] = 1.0 if success else 0.0
        else:
            current_rate = self.metrics.pattern_success_rate[pattern_name]
            self.metrics.pattern_success_rate[pattern_name] = (
                0.9 * current_rate + 0.1 * (1.0 if success else 0.0)
            )
        
        # Average latency
        if pattern_name not in self.metrics.pattern_avg_latency:
            self.metrics.pattern_avg_latency[pattern_name] = latency_ms
        else:
            current_avg = self.metrics.pattern_avg_latency[pattern_name]
            self.metrics.pattern_avg_latency[pattern_name] = (
                0.9 * current_avg + 0.1 * latency_ms
            )
    
    # ==========================================
    # MÉTRIQUES ET MONITORING PHASE 1
    # ==========================================
    
    def get_l0_metrics(self) -> L0PerformanceMetrics:
        """Retourne métriques L0 complètes"""
        # Mise à jour timestamp
        self.metrics.timestamp = int(time.time())
        
        # Calcul distributions confiance
        self.metrics.confidence_distribution = self._calculate_confidence_distribution()
        
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Status Pattern Matcher avec métriques essentielles"""
        metrics = self.get_l0_metrics()
        
        return {
            "status": "ready",
            "phase": "L0_PATTERN_MATCHING",
            "patterns_loaded": self.patterns.pattern_count,
            "total_requests": metrics.total_requests,
            "success_rate": metrics.l0_success_rate,
            "avg_latency_ms": round(metrics.avg_l0_latency_ms, 2),
            "target_usage_percent": round(metrics.target_l0_usage_percent, 1),
            "cache_hit_rate": round(metrics.cache_hit_rate, 3),
            "cache_size": len(self._pattern_cache),
            "targets_met": {
                "latency": metrics.avg_l0_latency_ms < 10.0,
                "usage": metrics.target_l0_usage_percent >= 80.0,
                "success_rate": metrics.l0_success_rate >= 0.85
            }
        }
    
    def get_pattern_usage_report(self) -> Dict[str, Any]:
        """Rapport détaillé usage patterns"""
        sorted_patterns = sorted(
            self.metrics.pattern_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "total_patterns": self.patterns.pattern_count,
            "used_patterns": len(self.metrics.pattern_usage),
            "unused_patterns": self.patterns.pattern_count - len(self.metrics.pattern_usage),
            "top_patterns": dict(sorted_patterns[:10]),
            "pattern_performance": {
                name: {
                    "usage_count": self.metrics.pattern_usage.get(name, 0),
                    "success_rate": round(self.metrics.pattern_success_rate.get(name, 0.0), 3),
                    "avg_latency_ms": round(self.metrics.pattern_avg_latency.get(name, 0.0), 2)
                }
                for name in [p[0] for p in sorted_patterns[:10]]
            }
        }
    
    def _calculate_confidence_distribution(self) -> Dict[str, int]:
        """Calcule distribution confiance patterns utilisés"""
        distribution = {"VERY_HIGH": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for pattern_name, usage_count in self.metrics.pattern_usage.items():
            # Trouve confiance pattern dans métadonnées
            pattern_confidence = 0.0
            for metadata in self.patterns.pattern_metadata.values():
                if metadata["name"] == pattern_name:
                    pattern_confidence = metadata["confidence"]
                    break
            
            # Classification confiance
            if pattern_confidence >= 0.95:
                distribution["VERY_HIGH"] += usage_count
            elif pattern_confidence >= 0.90:
                distribution["HIGH"] += usage_count
            elif pattern_confidence >= 0.85:
                distribution["MEDIUM"] += usage_count
            else:
                distribution["LOW"] += usage_count
        
        return distribution
    
    # ==========================================
    # MÉTHODES DEBUG ET TESTING PHASE 1
    # ==========================================
    
    async def test_pattern(self, query: str, expected_intent: str = None) -> Dict[str, Any]:
        """Test pattern matching pour debug avec détails Phase 1"""
        start_time = time.time()
        
        normalized = self._normalize_query(query)
        matches = self.patterns.match_patterns(normalized)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            "test_info": {
                "original_query": query,
                "normalized_query": normalized,
                "expected_intent": expected_intent,
                "test_timestamp": int(time.time())
            },
            "performance": {
                "latency_ms": round(latency_ms, 2),
                "target_met": latency_ms < 10.0,
                "matches_found": len(matches)
            },
            "analysis": {
                "normalization_changes": query != normalized,
                "cache_would_hit": self._generate_cache_key(normalized, "test") in self._pattern_cache
            }
        }
        
        if matches:
            best_match = matches[0]
            result["best_match"] = {
                "pattern_name": best_match.pattern_name,
                "pattern_type": best_match.pattern_type.value,
                "confidence": best_match.confidence,
                "matched_text": best_match.matched_text,
                "match_position": best_match.position,
                "entities_found": len(best_match.entities),
                "entities": [
                    {
                        "type": entity.type,
                        "value": entity.value,
                        "confidence": entity.confidence,
                        "normalized": entity.normalized_value
                    }
                    for entity in best_match.entities
                ]
            }
            
            # Toutes les alternatives
            result["alternatives"] = [
                {
                    "pattern_name": match.pattern_name,
                    "confidence": match.confidence,
                    "pattern_type": match.pattern_type.value
                }
                for match in matches[:5]
            ]
            
            # Vérification expectation
            if expected_intent:
                # Trouve l'intention du pattern dans les métadonnées
                pattern_intent = None
                for metadata in self.patterns.pattern_metadata.values():
                    if metadata["name"] == best_match.pattern_name:
                        pattern_intent = metadata["intent"].value
                        break
                
                result["expectation"] = {
                    "expected": expected_intent,
                    "actual": pattern_intent,
                    "met": pattern_intent == expected_intent
                }
        else:
            result["no_matches"] = {
                "reason": "No patterns matched above confidence threshold",
                "suggestions": [
                    "Check if query contains financial keywords",
                    "Verify normalization didn't remove important terms",
                    "Consider adding new patterns for this use case"
                ]
            }
        
        return result
    
    async def benchmark_l0_performance(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Benchmark performance L0 avec requêtes test Phase 1"""
        if not test_queries:
            test_queries = self._get_default_test_queries()
        
        logger.info(f"🏁 Benchmark L0 sur {len(test_queries)} requêtes...")
        
        benchmark_start = time.time()
        results = []
        successful_matches = 0
        total_latency = 0.0
        confidence_scores = []
        
        for query in test_queries:
            query_start = time.time()
            
            try:
                match = await self.match_intent(query, "benchmark")
                query_latency = (time.time() - query_start) * 1000
                
                if match:
                    successful_matches += 1
                    confidence_scores.append(match.confidence)
                    results.append({
                        "query": query,
                        "pattern_name": match.pattern_name,
                        "confidence": match.confidence,
                        "latency_ms": round(query_latency, 2),
                        "success": True,
                        "entities_count": len(match.entities)
                    })
                else:
                    results.append({
                        "query": query,
                        "pattern_name": None,
                        "confidence": 0.0,
                        "latency_ms": round(query_latency, 2),
                        "success": False,
                        "reason": "no_match_or_low_confidence"
                    })
                
                total_latency += query_latency
                
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        total_benchmark_time = (time.time() - benchmark_start) * 1000
        avg_latency = total_latency / len(test_queries)
        success_rate = successful_matches / len(test_queries)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Analyse des résultats
        benchmark_results = {
            "summary": {
                "total_queries": len(test_queries),
                "successful_matches": successful_matches,
                "success_rate": round(success_rate, 3),
                "avg_latency_ms": round(avg_latency, 2),
                "avg_confidence": round(avg_confidence, 3),
                "total_benchmark_time_ms": round(total_benchmark_time, 2),
                "queries_per_second": round(len(test_queries) / (total_benchmark_time / 1000), 1)
            },
            "targets_analysis": {
                "latency_target": 10.0,
                "latency_target_met": avg_latency < 10.0,
                "success_rate_target": 0.85,
                "success_rate_target_met": success_rate >= 0.85,
                "confidence_target": 0.90,
                "confidence_target_met": avg_confidence >= 0.90
            },
            "performance_distribution": {
                "under_5ms": len([r for r in results if r.get("latency_ms", 999) < 5]),
                "under_10ms": len([r for r in results if r.get("latency_ms", 999) < 10]),
                "over_10ms": len([r for r in results if r.get("latency_ms", 0) >= 10])
            },
            "sample_results": results[:10],  # Échantillon pour debug
            "pattern_usage_in_test": self._analyze_benchmark_pattern_usage(results)
        }
        
        logger.info(f"🏁 Benchmark L0 terminé - Success: {success_rate:.1%}, Latence: {avg_latency:.1f}ms")
        return benchmark_results
    
    def _get_default_test_queries(self) -> List[str]:
        """Requêtes test par défaut Phase 1"""
        return [
            # Balance checks (haute fréquence)
            "solde", "quel est mon solde", "combien j'ai sur mon compte",
            "solde compte courant", "voir mes comptes", "argent disponible",
            
            # Transfers (fréquence moyenne)
            "virement", "faire un virement de 100 euros", "virer 50€",
            "transférer 200 euros", "envoyer de l'argent", "payer Paul",
            
            # Expenses (fréquence moyenne)
            "mes dépenses", "dépenses restaurant", "combien j'ai dépensé",
            "budget transport", "dépenses ce mois", "voir mes dépenses",
            
            # Card management (fréquence faible)
            "bloquer carte", "activer carte", "changer code carte",
            "limites carte", "opposition carte", "carte volée",
            
            # System (fréquence élevée)
            "bonjour", "aide", "comment faire", "au revoir", "merci",
            "salut", "hello", "help", "fonctionnalités",
            
            # Edge cases et variations
            "solde compte épargne maintenant", "virement urgent 1000€ vers Marie",
            "dépenses shopping cette année", "bloquer carte volée immédiatement",
            "combien me reste t il", "faire opposition carte"
        ]
    
    def _analyze_benchmark_pattern_usage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse usage patterns dans benchmark"""
        pattern_counts = {}
        successful_patterns = {}
        
        for result in results:
            if result["success"] and result.get("pattern_name"):
                pattern_name = result["pattern_name"]
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
                
                if pattern_name not in successful_patterns:
                    successful_patterns[pattern_name] = []
                successful_patterns[pattern_name].append(result["latency_ms"])
        
        # Calcul moyennes par pattern
        pattern_performance = {}
        for pattern, latencies in successful_patterns.items():
            pattern_performance[pattern] = {
                "usage_count": pattern_counts[pattern],
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
                "min_latency_ms": round(min(latencies), 2),
                "max_latency_ms": round(max(latencies), 2)
            }
        
        return {
            "total_unique_patterns_used": len(pattern_counts),
            "most_used_patterns": dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "pattern_performance": pattern_performance
        }
    
    # ==========================================
    # GESTION DYNAMIQUE PATTERNS
    # ==========================================
    
    def add_dynamic_pattern(self, intent: FinancialIntent, pattern_data: Dict[str, Any]) -> bool:
        """Ajout pattern dynamique en runtime"""
        try:
            pattern_text = pattern_data["regex"]
            compiled_pattern = re.compile(pattern_text, re.IGNORECASE | re.UNICODE)
            
            # Ajout au dictionnaire patterns
            if intent not in self.patterns.compiled_patterns:
                self.patterns.compiled_patterns[intent] = []
            
            self.patterns.compiled_patterns[intent].append(compiled_pattern)
            
            # Ajout métadonnées
            pattern_id = f"{intent.value}_{pattern_data['name']}"
            self.patterns.pattern_metadata[pattern_id] = {
                "intent": intent,
                "confidence": pattern_data["confidence"],
                "name": pattern_data["name"],
                "pattern_type": pattern_data.get("type", PatternType.DIRECT_KEYWORD),
                "regex": pattern_text,
                "dynamic": True,
                "priority": pattern_data.get("priority", 0)
            }
            
            self.patterns.pattern_count += 1
            
            logger.info(f"➕ Pattern dynamique ajouté: {intent.value} - {pattern_data['name']}")
            return True
            
        except (re.error, KeyError) as e:
            logger.warning(f"⚠️ Erreur pattern dynamique {pattern_data.get('name', 'unknown')}: {e}")
            return False
    
    def remove_pattern(self, pattern_name: str) -> bool:
        """Suppression pattern dynamique"""
        try:
            # Cherche et supprime des métadonnées
            pattern_id_to_remove = None
            for pattern_id, metadata in self.patterns.pattern_metadata.items():
                if metadata["name"] == pattern_name and metadata.get("dynamic", False):
                    pattern_id_to_remove = pattern_id
                    break
            
            if pattern_id_to_remove:
                intent = self.patterns.pattern_metadata[pattern_id_to_remove]["intent"]
                del self.patterns.pattern_metadata[pattern_id_to_remove]
                
                # Note: Suppression du compiled pattern nécessiterait une recompilation
                # Pour l'instant, on marque juste comme supprimé
                logger.info(f"➖ Pattern dynamique supprimé: {pattern_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur suppression pattern {pattern_name}: {e}")
            return False
    
    # ==========================================
    # CLEANUP ET SHUTDOWN
    # ==========================================
    
    async def shutdown(self):
        """Arrêt propre Pattern Matcher Phase 1"""
        logger.info("🛑 Arrêt Pattern Matcher L0...")
        
        # Sauvegarde statistiques finales
        final_metrics = self.get_l0_metrics()
        final_status = self.get_status()
        
        logger.info(f"📊 Stats finales L0:")
        logger.info(f"   - Total requêtes: {final_metrics.total_requests}")
        logger.info(f"   - Taux succès: {final_status['success_rate']:.1%}")
        logger.info(f"   - Latence moyenne: {final_status['avg_latency_ms']:.1f}ms")
        logger.info(f"   - Usage L0: {final_status['target_usage_percent']:.1f}%")
        logger.info(f"   - Cache hit rate: {final_status['cache_hit_rate']:.3f}")
        
        # Nettoyage caches
        self._pattern_cache.clear()
        
        logger.info("✅ Pattern Matcher L0 arrêté proprement")


# ==========================================
# UTILITAIRES ET VALIDATION PHASE 1
# ==========================================

async def validate_l0_phase1_performance(pattern_matcher: PatternMatcher) -> Dict[str, Any]:
    """Validation complète performance Phase 1 selon targets"""
    logger.info("🎯 Validation performance L0 Phase 1...")
    
    # Benchmark performance
    benchmark_results = await pattern_matcher.benchmark_l0_performance()
    
    # Métriques actuelles
    current_metrics = pattern_matcher.get_l0_metrics()
    current_status = pattern_matcher.get_status()
    
    # Tests de conformité Phase 1
    targets = {
        "latency_ms": 10.0,
        "success_rate": 0.85,
        "usage_percent": 80.0,
        "confidence": 0.90,
        "cache_hit_rate": 0.15
    }
    
    validation = {
        "phase": "L0_PATTERN_MATCHING",
        "validation_timestamp": int(time.time()),
        "targets": targets,
        "current_performance": {
            "avg_latency_ms": current_status["avg_latency_ms"],
            "success_rate": current_status["success_rate"],
            "usage_percent": current_status["target_usage_percent"],
            "cache_hit_rate": current_status["cache_hit_rate"]
        },
        "targets_met": {
            "latency": current_status["avg_latency_ms"] < targets["latency_ms"],
            "success_rate": current_status["success_rate"] >= targets["success_rate"],
            "usage": current_status["target_usage_percent"] >= targets["usage_percent"],
            "cache_performance": current_status["cache_hit_rate"] >= targets["cache_hit_rate"]
        },
        "benchmark_results": benchmark_results,
        "recommendations": []
    }
    
    # Génération recommandations
    if not validation["targets_met"]["latency"]:
        validation["recommendations"].append("Optimiser patterns les plus lents")
    
    if not validation["targets_met"]["success_rate"]:
        validation["recommendations"].append("Ajouter patterns pour cas non couverts")
    
    if not validation["targets_met"]["cache_performance"]:
        validation["recommendations"].append("Améliorer stratégie de cache")
    
    # Status global
    all_targets_met = all(validation["targets_met"].values())
    validation["overall_status"] = "READY_FOR_L1" if all_targets_met else "NEEDS_OPTIMIZATION"
    
    logger.info(f"🎯 Validation terminée - Status: {validation['overall_status']}")
    return validation

def create_test_queries_phase1() -> List[str]:
    """Génère requêtes test spécialisées Phase 1"""
    return [
        # Patterns haute confiance (>0.95)
        "solde", "bonjour", "virement", "bloquer carte", "aide",
        
        # Patterns avec entités
        "virer 100€", "dépenses restaurant", "solde compte courant",
        "faire un virement de 250 euros", "mes dépenses ce mois",
        
        # Patterns questions
        "quel est mon solde", "combien j'ai dépensé", "comment faire",
        
        # Patterns temporels
        "solde maintenant", "dépenses cette semaine", 
        
        # Edge cases
        "solde compte épargne", "opposition carte", "au revoir",
        "virement urgent", "carte perdue", "argent disponible",
        
        # Tests négatives (ne devraient pas matcher)
        "météo", "actualités", "recette cuisine",
        "123456", "................", "qwertyuiop"
    ]

# ==========================================
# EXPORTS PHASE 1
# ==========================================

__all__ = [
    "PatternMatcher",
    "FinancialPatterns", 
    "validate_l0_phase1_performance",
    "create_test_queries_phase1"
]