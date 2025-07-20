"""
Pattern Matcher pour détection rapide L0
Reconnaissance patterns financiers fréquents
"""

import re
from typing import Optional, Dict, List, Any
import time
from dataclasses import dataclass

from .models import IntentResult, IntentLevel, IntentConfidence, IntentType
from conversation_service.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FinancialPattern:
    """Pattern financier avec regex et métadonnées"""
    pattern: str
    regex: re.Pattern
    intent_type: str
    entities: Dict[str, str]
    confidence: float
    examples: List[str]


class PatternMatcher:
    """
    Matcher patterns financiers optimisé pour performance L0
    Reconnaissance sous 10ms des requêtes fréquentes
    """
    
    def __init__(self):
        self.financial_patterns = self._compile_financial_patterns()
        self.entity_extractors = self._compile_entity_extractors()
    
    def _compile_financial_patterns(self) -> List[FinancialPattern]:
        """Compilation patterns financiers avec regex optimisées"""
        
        patterns = [
            # === BALANCE CHECK PATTERNS ===
            FinancialPattern(
                pattern="solde_account",
                regex=re.compile(r"(?:solde|combien|montant).*(?:compte|cb|carte|livret)", re.I),
                intent_type=IntentType.BALANCE_CHECK.value,
                entities={"query_type": "balance"},
                confidence=0.95,
                examples=["solde compte courant", "combien sur ma carte", "montant livret a"]
            ),
            
            # === EXPENSE ANALYSIS PATTERNS ===
            FinancialPattern(
                pattern="expenses_category",
                regex=re.compile(r"(?:dépenses?|achats?|frais).*(?:restaurant|courses|essence|shopping|pharmacie)", re.I),
                intent_type=IntentType.EXPENSE_ANALYSIS.value,
                entities={"analysis_type": "category_expenses"},
                confidence=0.92,
                examples=["mes dépenses restaurant", "achats courses janvier", "frais essence"]
            ),
            
            FinancialPattern(
                pattern="monthly_expenses",
                regex=re.compile(r"(?:dépenses?|total).*(?:mois|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)", re.I),
                intent_type=IntentType.EXPENSE_ANALYSIS.value,
                entities={"analysis_type": "monthly_expenses"},
                confidence=0.90,
                examples=["dépenses janvier", "total mois dernier", "frais février"]
            ),
            
            # === TRANSFER PATTERNS ===
            FinancialPattern(
                pattern="transfer_contact",
                regex=re.compile(r"(?:virement|virer|envoyer|transfert).*(?:vers|à|pour)\s+([a-zA-ZÀ-ÿ]+)", re.I),
                intent_type=IntentType.TRANSFER.value,
                entities={"action_type": "transfer_to_contact"},
                confidence=0.88,
                examples=["virement vers marie", "virer à paul", "transfert pour julie"]
            ),
            
            # === TRANSACTION SEARCH PATTERNS ===
            FinancialPattern(
                pattern="search_merchant",
                regex=re.compile(r"(?:transactions?|achats?|paiements?).*(?:chez|à|sur)\s+([a-zA-Z0-9\s]+)", re.I),
                intent_type=IntentType.TRANSACTION_SEARCH.value,
                entities={"search_type": "merchant"},
                confidence=0.85,
                examples=["transactions chez carrefour", "achats à amazon", "paiements sur netflix"]
            ),
            
            FinancialPattern(
                pattern="search_amount",
                regex=re.compile(r"(?:transactions?|paiements?).*(\d+(?:,\d+)?)\s*(?:€|euros?)", re.I),
                intent_type=IntentType.TRANSACTION_SEARCH.value,
                entities={"search_type": "amount"},
                confidence=0.87,
                examples=["transactions 50 euros", "paiements 25,50€"]
            ),
            
            # === BUDGET INQUIRY PATTERNS ===
            FinancialPattern(
                pattern="budget_category",
                regex=re.compile(r"(?:budget|enveloppe|limite).*(?:restaurant|courses|loisirs|transport)", re.I),
                intent_type=IntentType.BUDGET_INQUIRY.value,
                entities={"budget_type": "category_budget"},
                confidence=0.89,
                examples=["budget restaurant", "enveloppe courses", "limite loisirs"]
            ),
        ]
        
        logger.info(f"Compiled {len(patterns)} financial patterns")
        return patterns
    
    def _compile_entity_extractors(self) -> Dict[str, re.Pattern]:
        """Extracteurs d'entités spécialisés"""
        return {
            "amount": re.compile(r"(\d+(?:,\d+)?)\s*(?:€|euros?)", re.I),
            "date": re.compile(r"(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|\d{1,2}/\d{1,2})", re.I),
            "contact_name": re.compile(r"(?:vers|à|pour)\s+([a-zA-ZÀ-ÿ]+)", re.I),
            "merchant": re.compile(r"(?:chez|à|sur)\s+([a-zA-Z0-9\s]+)", re.I),
            "account_type": re.compile(r"(compte courant|livret a|carte|cb|pel|ldd)", re.I),
            "category": re.compile(r"(restaurant|courses|essence|shopping|pharmacie|loisirs|transport)", re.I)
        }
    
    async def match_financial_patterns(self, normalized_query: str) -> Optional[IntentResult]:
        """
        Matching rapide patterns financiers
        Performance cible: 5-10ms
        """
        start_time = time.time()
        
        try:
            # Test chaque pattern compilé
            for pattern in self.financial_patterns:
                match = pattern.regex.search(normalized_query)
                
                if match:
                    # Extraction entités additionnelles
                    entities = dict(pattern.entities)  # Copy base entities
                    entities.update(self._extract_entities(normalized_query, match))
                    
                    result = IntentResult(
                        intent_type=pattern.intent_type,
                        entities=entities,
                        confidence=IntentConfidence(
                            score=pattern.confidence,
                            level=IntentLevel.L0_PATTERN
                        ),
                        level=IntentLevel.L0_PATTERN,
                        latency_ms=int((time.time() - start_time) * 1000),
                        metadata={
                            "pattern_matched": pattern.pattern,
                            "regex_groups": match.groups(),
                            "match_span": match.span()
                        }
                    )
                    
                    logger.debug(f"Pattern matched: {pattern.pattern} for query: {normalized_query}")
                    return result
            
            # Aucun pattern trouvé
            return None
            
        except Exception as e:
            logger.error(f"Erreur pattern matching: {e}")
            return None
    
    def _extract_entities(self, query: str, primary_match: re.Match) -> Dict[str, str]:
        """Extraction entités avec extracteurs spécialisés"""
        entities = {}
        
        for entity_type, extractor in self.entity_extractors.items():
            match = extractor.search(query)
            if match:
                entities[entity_type] = match.group(1).strip()
        
        return entities
    
    async def preload_frequent_patterns(self) -> bool:
        """Préchargement patterns fréquents en mémoire"""
        try:
            # Validation regex compilation
            for pattern in self.financial_patterns:
                test_query = pattern.examples[0] if pattern.examples else "test"
                pattern.regex.search(test_query.lower())
            
            logger.info("Patterns fréquents préchargés avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur préchargement patterns: {e}")
            return False
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Statistiques patterns compilés"""
        return {
            "total_patterns": len(self.financial_patterns),
            "patterns_by_intent": {
                intent_type: len([p for p in self.financial_patterns if p.intent_type == intent_type])
                for intent_type in set(p.intent_type for p in self.financial_patterns)
            },
            "entity_extractors": list(self.entity_extractors.keys())
        }
