"""
Exemples few-shot optimisés et dynamiques pour classification intentions Harena
"""
import logging
from typing import List, Dict, Set, Optional, Tuple
from functools import lru_cache
import random
from dataclasses import dataclass, field
from enum import Enum
from conversation_service.prompts.harena_intents import HarenaIntentType, INTENT_CATEGORIES

# Configuration du logger
logger = logging.getLogger("conversation_service.examples")


@dataclass(frozen=True)
class IntentExample:
    """
    Exemple d'intention optimisé avec métadonnées pour sélection dynamique
    """
    input: str
    intent: str
    confidence: float
    complexity: str = "simple"  # simple, medium, complex
    keywords: Tuple[str, ...] = field(default_factory=tuple)  # Mots-clés pour matching
    language_pattern: str = "standard"  # standard, colloquial, technical
    priority: int = 1  # 1=high, 2=medium, 3=low pour sélection
    context_tags: Tuple[str, ...] = field(default_factory=tuple)  # Tags contextuels


class ExampleComplexity(str, Enum):
    """Niveaux de complexité des exemples"""
    SIMPLE = "simple"
    MEDIUM = "medium" 
    COMPLEX = "complex"


class LanguagePattern(str, Enum):
    """Patterns linguistiques"""
    STANDARD = "standard"
    COLLOQUIAL = "colloquial"
    TECHNICAL = "technical"


# Exemples few-shot optimisés avec métadonnées enrichies
HARENA_INTENT_EXAMPLES = [
    # TRANSACTIONS - SEARCH_BY_MERCHANT (Haute priorité)
    IntentExample(
        "Mes achats Amazon", "SEARCH_BY_MERCHANT", 0.96, "simple",
        ("amazon", "achats"), "standard", 1, ("e-commerce", "frequent")
    ),
    IntentExample(
        "Qu'est-ce que j'ai acheté chez Amazon ?", "SEARCH_BY_MERCHANT", 0.93, "medium",
        ("amazon", "acheté"), "standard", 1, ("e-commerce", "question")
    ),
    IntentExample(
        "Toutes mes transactions Carrefour", "SEARCH_BY_MERCHANT", 0.95, "simple",
        ("carrefour", "transactions"), "standard", 1, ("supermarket", "frequent")
    ),
    IntentExample(
        "Mes dépenses McDonald's", "SEARCH_BY_MERCHANT", 0.94, "simple",
        ("mcdonald", "dépenses"), "standard", 1, ("restaurant", "frequent")
    ),
    IntentExample(
        "Historique Uber Eats", "SEARCH_BY_MERCHANT", 0.92, "simple",
        ("uber", "historique"), "standard", 1, ("delivery", "app")
    ),
    IntentExample(
        "Combien j'ai claqué chez Netflix ?", "SEARCH_BY_MERCHANT", 0.89, "medium",
        ("netflix", "claqué"), "colloquial", 2, ("subscription", "entertainment")
    ),
    IntentExample(
        "Transactions SNCF Connect", "SEARCH_BY_MERCHANT", 0.91, "simple",
        ("sncf", "transactions"), "standard", 2, ("transport", "travel")
    ),
    
    # TRANSACTIONS - SEARCH_BY_DATE (Haute priorité)
    IntentExample(
        "Mes transactions d'hier", "SEARCH_BY_DATE", 0.97, "simple",
        ("hier", "transactions"), "standard", 1, ("recent", "daily")
    ),
    IntentExample(
        "Qu'est-ce que j'ai dépensé cette semaine ?", "SEARCH_BY_DATE", 0.91, "medium",
        ("semaine", "dépensé"), "standard", 1, ("recent", "weekly")
    ),
    IntentExample(
        "Mes achats du mois dernier", "SEARCH_BY_DATE", 0.93, "simple",
        ("mois", "achats"), "standard", 1, ("monthly", "past")
    ),
    IntentExample(
        "Transactions entre le 1er et le 15 août", "SEARCH_BY_DATE", 0.95, "complex",
        ("entre", "août", "15"), "technical", 2, ("range", "specific")
    ),
    IntentExample(
        "Mes dépenses depuis janvier", "SEARCH_BY_DATE", 0.90, "medium",
        ("depuis", "janvier"), "standard", 1, ("range", "monthly")
    ),
    IntentExample(
        "Ce que j'ai payé aujourd'hui", "SEARCH_BY_DATE", 0.94, "simple",
        ("aujourd'hui", "payé"), "standard", 1, ("today", "recent")
    ),
    IntentExample(
        "Transactions de la semaine passée", "SEARCH_BY_DATE", 0.92, "simple",
        ("semaine", "passée"), "standard", 2, ("weekly", "past")
    ),
    
    # TRANSACTIONS - SEARCH_BY_AMOUNT (Priorité moyenne)
    IntentExample(
        "Mes gros achats", "SEARCH_BY_AMOUNT", 0.87, "simple",
        ("gros", "achats"), "colloquial", 2, ("high_amount", "relative")
    ),
    IntentExample(
        "Transactions supérieures à 100€", "SEARCH_BY_AMOUNT", 0.98, "medium",
        ("supérieures", "100"), "technical", 1, ("threshold", "specific")
    ),
    IntentExample(
        "Mes petites dépenses", "SEARCH_BY_AMOUNT", 0.85, "simple",
        ("petites", "dépenses"), "colloquial", 2, ("low_amount", "relative")
    ),
    IntentExample(
        "Achats entre 50 et 200 euros", "SEARCH_BY_AMOUNT", 0.96, "complex",
        ("entre", "50", "200", "euros"), "technical", 1, ("range", "specific")
    ),
    IntentExample(
        "Dépenses de moins de 10€", "SEARCH_BY_AMOUNT", 0.93, "medium",
        ("moins", "10€"), "standard", 2, ("threshold", "low")
    ),
    IntentExample(
        "Mes achats chers", "SEARCH_BY_AMOUNT", 0.84, "simple",
        ("achats", "chers"), "colloquial", 3, ("high_amount", "relative")
    ),
    
    # TRANSACTIONS - SEARCH_BY_CATEGORY (Haute priorité)
    IntentExample(
        "Mes dépenses restaurants", "SEARCH_BY_CATEGORY", 0.95, "simple",
        ("restaurants", "dépenses"), "standard", 1, ("food", "frequent")
    ),
    IntentExample(
        "Combien j'ai dépensé en courses ?", "SEARCH_BY_CATEGORY", 0.91, "medium",
        ("courses", "dépensé"), "standard", 1, ("grocery", "frequent")
    ),
    IntentExample(
        "Mes frais de transport", "SEARCH_BY_CATEGORY", 0.93, "simple",
        ("transport", "frais"), "standard", 1, ("transport", "frequent")
    ),
    IntentExample(
        "Dépenses santé et médical", "SEARCH_BY_CATEGORY", 0.92, "medium",
        ("santé", "médical"), "standard", 2, ("health", "important")
    ),
    IntentExample(
        "Mes achats de vêtements", "SEARCH_BY_CATEGORY", 0.94, "simple",
        ("vêtements", "achats"), "standard", 2, ("clothing", "shopping")
    ),
    IntentExample(
        "Sorties loisirs et divertissement", "SEARCH_BY_CATEGORY", 0.88, "medium",
        ("loisirs", "divertissement"), "standard", 2, ("entertainment", "leisure")
    ),
    IntentExample(
        "Mes dépenses essence", "SEARCH_BY_CATEGORY", 0.90, "simple",
        ("essence", "dépenses"), "standard", 2, ("fuel", "transport")
    ),
    
    # SPENDING_ANALYSIS (Haute priorité)
    IntentExample(
        "Analyse de mes dépenses", "SPENDING_ANALYSIS", 0.95, "simple",
        ("analyse", "dépenses"), "technical", 1, ("analysis", "overview")
    ),
    IntentExample(
        "Combien j'ai dépensé ce mois ?", "SPENDING_ANALYSIS", 0.92, "medium",
        ("combien", "dépensé", "mois"), "standard", 1, ("monthly", "total")
    ),
    IntentExample(
        "Ma consommation financière", "SPENDING_ANALYSIS", 0.87, "medium",
        ("consommation", "financière"), "technical", 2, ("analysis", "global")
    ),
    IntentExample(
        "Répartition de mes sorties d'argent", "SPENDING_ANALYSIS", 0.89, "complex",
        ("répartition", "sorties", "argent"), "standard", 2, ("breakdown", "analysis")
    ),
    IntentExample(
        "Bilan de mes dépenses mensuelles", "SPENDING_ANALYSIS", 0.93, "medium",
        ("bilan", "mensuelles"), "technical", 1, ("monthly", "summary")
    ),
    IntentExample(
        "Où part mon fric ?", "SPENDING_ANALYSIS", 0.84, "simple",
        ("où", "fric"), "colloquial", 3, ("analysis", "casual")
    ),
    
    # BALANCE_INQUIRY (Très haute priorité - Simple)
    IntentExample(
        "Mon solde", "BALANCE_INQUIRY", 0.99, "simple",
        ("solde",), "standard", 1, ("balance", "quick")
    ),
    IntentExample(
        "Combien j'ai sur mon compte ?", "BALANCE_INQUIRY", 0.97, "medium",
        ("combien", "compte"), "standard", 1, ("balance", "question")
    ),
    IntentExample(
        "Solde actuel", "BALANCE_INQUIRY", 0.98, "simple",
        ("solde", "actuel"), "standard", 1, ("balance", "current")
    ),
    IntentExample(
        "Ma situation financière", "BALANCE_INQUIRY", 0.86, "medium",
        ("situation", "financière"), "standard", 2, ("balance", "overview")
    ),
    IntentExample(
        "Combien il me reste ?", "BALANCE_INQUIRY", 0.92, "simple",
        ("combien", "reste"), "colloquial", 1, ("balance", "available")
    ),
    IntentExample(
        "Mon pognon disponible", "BALANCE_INQUIRY", 0.88, "simple",
        ("pognon", "disponible"), "colloquial", 3, ("balance", "casual")
    ),
    
    # GREETING (Très haute priorité)
    IntentExample(
        "Bonjour", "GREETING", 0.99, "simple",
        ("bonjour",), "standard", 1, ("greeting", "polite")
    ),
    IntentExample(
        "Salut", "GREETING", 0.98, "simple",
        ("salut",), "colloquial", 1, ("greeting", "casual")
    ),
    IntentExample(
        "Hello", "GREETING", 0.97, "simple",
        ("hello",), "standard", 1, ("greeting", "english")
    ),
    IntentExample(
        "Bonsoir", "GREETING", 0.98, "simple",
        ("bonsoir",), "standard", 1, ("greeting", "evening")
    ),
    IntentExample(
        "Hey", "GREETING", 0.95, "simple",
        ("hey",), "colloquial", 2, ("greeting", "casual")
    ),
    IntentExample(
        "Coucou Harena", "GREETING", 0.94, "simple",
        ("coucou", "harena"), "colloquial", 2, ("greeting", "personal")
    ),
    
    # CONFIRMATION (Priorité moyenne)
    IntentExample(
        "Merci", "CONFIRMATION", 0.96, "simple",
        ("merci",), "standard", 1, ("thanks", "polite")
    ),
    IntentExample(
        "Parfait", "CONFIRMATION", 0.93, "simple",
        ("parfait",), "standard", 1, ("confirmation", "positive")
    ),
    IntentExample(
        "C'est bon", "CONFIRMATION", 0.90, "simple",
        ("bon",), "colloquial", 2, ("confirmation", "casual")
    ),
    IntentExample(
        "OK", "CONFIRMATION", 0.95, "simple",
        ("ok",), "standard", 1, ("confirmation", "acknowledgment")
    ),
    IntentExample(
        "Super, merci", "CONFIRMATION", 0.94, "simple",
        ("super", "merci"), "standard", 2, ("thanks", "enthusiastic")
    ),
    IntentExample(
        "Nickel", "CONFIRMATION", 0.91, "simple",
        ("nickel",), "colloquial", 3, ("confirmation", "slang")
    ),
    
    # NON SUPPORTÉES (Important pour formation - Priorité élevée)
    IntentExample(
        "Faire un virement", "TRANSFER_REQUEST", 0.97, "simple",
        ("virement",), "standard", 1, ("unsupported", "banking")
    ),
    IntentExample(
        "Virer 500€ à Paul", "TRANSFER_REQUEST", 0.95, "medium",
        ("virer", "500€"), "standard", 1, ("unsupported", "specific")
    ),
    IntentExample(
        "Transférer de l'argent", "TRANSFER_REQUEST", 0.94, "simple",
        ("transférer", "argent"), "standard", 2, ("unsupported", "transfer")
    ),
    IntentExample(
        "Payer ma facture EDF", "PAYMENT_REQUEST", 0.95, "medium",
        ("payer", "facture"), "standard", 1, ("unsupported", "bill")
    ),
    IntentExample(
        "Effectuer un paiement", "PAYMENT_REQUEST", 0.93, "simple",
        ("paiement",), "technical", 1, ("unsupported", "payment")
    ),
    IntentExample(
        "Bloquer ma carte", "CARD_BLOCK", 0.98, "simple",
        ("bloquer", "carte"), "standard", 1, ("unsupported", "security")
    ),
    IntentExample(
        "Suspendre ma CB", "CARD_BLOCK", 0.96, "simple",
        ("suspendre", "cb"), "colloquial", 2, ("unsupported", "security")
    ),
    IntentExample(
        "Où en est mon budget ?", "BUDGET_INQUIRY", 0.92, "medium",
        ("budget",), "standard", 1, ("unsupported", "budgeting")
    ),
    IntentExample(
        "Mon objectif épargne", "GOAL_TRACKING", 0.89, "medium",
        ("objectif", "épargne"), "standard", 2, ("unsupported", "savings")
    ),
    
    # UNCLEAR_INTENT (Formation importante - Priorité élevée)
    IntentExample(
        "Euh... je sais pas", "UNCLEAR_INTENT", 0.89, "simple",
        ("euh", "sais", "pas"), "colloquial", 1, ("unclear", "hesitation")
    ),
    IntentExample(
        "Peux-tu m'aider ?", "UNCLEAR_INTENT", 0.84, "medium",
        ("peux", "aider"), "standard", 1, ("unclear", "help")
    ),
    IntentExample(
        "Comment ça marche ?", "UNCLEAR_INTENT", 0.82, "medium",
        ("comment", "marche"), "standard", 2, ("unclear", "how")
    ),
    IntentExample(
        "Aide moi stp", "UNCLEAR_INTENT", 0.85, "simple",
        ("aide", "stp"), "colloquial", 2, ("unclear", "help")
    ),
    IntentExample(
        "Je comprends pas", "UNCLEAR_INTENT", 0.87, "simple",
        ("comprends", "pas"), "colloquial", 2, ("unclear", "confusion")
    ),
    IntentExample(
        "Hum...", "UNCLEAR_INTENT", 0.92, "simple",
        ("hum",), "colloquial", 3, ("unclear", "thinking")
    ),
    
    # UNKNOWN (Messages incompréhensibles - Priorité élevée)
    IntentExample(
        "azerty qwerty", "UNKNOWN", 0.96, "simple",
        ("azerty", "qwerty"), "standard", 1, ("gibberish", "keyboard")
    ),
    IntentExample(
        "123 456 !!!", "UNKNOWN", 0.95, "simple",
        ("123", "456"), "standard", 1, ("gibberish", "numbers")
    ),
    IntentExample(
        "jdhgkjdhgk", "UNKNOWN", 0.97, "simple",
        ("jdhgkjdhgk",), "standard", 1, ("gibberish", "random")
    ),
    IntentExample(
        "°°°°°", "UNKNOWN", 0.94, "simple",
        ("°",), "standard", 2, ("gibberish", "symbols")
    ),
]


class DynamicExampleSelector:
    """Sélecteur dynamique d'exemples avec logique adaptative"""
    
    def __init__(self, examples: List[IntentExample]):
        self.examples = examples
        self.usage_stats = {}  # Stats utilisation pour optimisation
        self._build_indices()
        logger.info(f"DynamicExampleSelector initialisé avec {len(examples)} exemples")
    
    def _build_indices(self) -> None:
        """Construction indices pour recherche optimisée"""
        self.by_intent = {}
        self.by_keywords = {}
        self.by_priority = {1: [], 2: [], 3: []}
        self.by_complexity = {}
        
        for example in self.examples:
            # Index par intention
            if example.intent not in self.by_intent:
                self.by_intent[example.intent] = []
            self.by_intent[example.intent].append(example)
            
            # Index par mots-clés
            for keyword in example.keywords:
                if keyword not in self.by_keywords:
                    self.by_keywords[keyword] = []
                self.by_keywords[keyword].append(example)
            
            # Index par priorité
            self.by_priority[example.priority].append(example)
            
            # Index par complexité
            if example.complexity not in self.by_complexity:
                self.by_complexity[example.complexity] = []
            self.by_complexity[example.complexity].append(example)
    
    def select_relevant_examples(
        self,
        user_message: str,
        max_examples: int = 15,
        balance_intents: bool = True,
        prefer_high_confidence: bool = True
    ) -> List[IntentExample]:
        """
        Sélection dynamique d'exemples pertinents
        
        Args:
            user_message: Message utilisateur pour contexte
            max_examples: Nombre maximum d'exemples
            balance_intents: Équilibrer les types d'intentions
            prefer_high_confidence: Privilégier haute confiance
            
        Returns:
            List[IntentExample]: Exemples sélectionnés et ordonnés
        """
        message_lower = user_message.lower()
        scored_examples = []
        
        # Score de pertinence pour chaque exemple
        for example in self.examples:
            relevance_score = self._calculate_relevance_score(example, message_lower)
            scored_examples.append((example, relevance_score))
        
        # Tri par score de pertinence
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        
        if balance_intents:
            selected = self._balance_intent_selection(scored_examples, max_examples)
        else:
            selected = [ex for ex, _ in scored_examples[:max_examples]]
        
        # Mise à jour stats utilisation
        self._update_usage_stats(selected)
        
        logger.debug(f"Sélectionnés {len(selected)} exemples pour: '{user_message[:50]}...'")
        return selected
    
    def _calculate_relevance_score(self, example: IntentExample, message_lower: str) -> float:
        """Calcul score de pertinence avec facteurs multiples"""
        score = 0.0
        
        # Score mots-clés (facteur principal)
        keyword_matches = sum(1 for keyword in example.keywords if keyword in message_lower)
        if example.keywords:
            keyword_score = keyword_matches / len(example.keywords)
            score += keyword_score * 10.0
        
        # Score confiance (facteur qualité)
        score += example.confidence * 5.0
        
        # Score priorité (facteur importance)
        priority_bonus = {1: 3.0, 2: 1.0, 3: 0.0}
        score += priority_bonus.get(example.priority, 0.0)
        
        # Bonus longueur similarité
        message_words = len(message_lower.split())
        example_words = len(example.input.split())
        if abs(message_words - example_words) <= 2:
            score += 1.0
        
        # Pénalité usage excessif (diversification)
        usage_count = self.usage_stats.get(example.input, 0)
        if usage_count > 5:
            score -= min(usage_count * 0.1, 2.0)
        
        return score
    
    def _balance_intent_selection(
        self, 
        scored_examples: List[Tuple[IntentExample, float]], 
        max_examples: int
    ) -> List[IntentExample]:
        """Sélection équilibrée par type d'intention"""
        
        selected = []
        intents_count = {}
        max_per_intent = max(2, max_examples // 8)  # Max 2-3 par intention
        
        # Sélection avec équilibrage
        for example, score in scored_examples:
            if len(selected) >= max_examples:
                break
            
            intent_count = intents_count.get(example.intent, 0)
            
            # Accepter si sous la limite ou très pertinent
            if intent_count < max_per_intent or score > 8.0:
                selected.append(example)
                intents_count[example.intent] = intent_count + 1
        
        # Compléter avec exemples haute priorité si nécessaire
        if len(selected) < max_examples:
            high_priority = [ex for ex, _ in scored_examples if ex.priority == 1 and ex not in selected]
            needed = max_examples - len(selected)
            selected.extend(high_priority[:needed])
        
        return selected
    
    def _update_usage_stats(self, examples: List[IntentExample]) -> None:
        """Mise à jour statistiques utilisation"""
        for example in examples:
            self.usage_stats[example.input] = self.usage_stats.get(example.input, 0) + 1
    
    def get_examples_by_intent(self, intent: str, max_count: int = 5) -> List[IntentExample]:
        """Récupération exemples par intention spécifique"""
        examples = self.by_intent.get(intent, [])
        return sorted(examples, key=lambda x: (x.priority, -x.confidence))[:max_count]
    
    def get_high_confidence_examples(self, min_confidence: float = 0.9) -> List[IntentExample]:
        """Récupération exemples haute confiance"""
        high_conf = [ex for ex in self.examples if ex.confidence >= min_confidence]
        return sorted(high_conf, key=lambda x: (-x.confidence, x.priority))
    
    def get_usage_statistics(self) -> Dict[str, int]:
        """Statistiques utilisation des exemples"""
        return dict(sorted(self.usage_stats.items(), key=lambda x: x[1], reverse=True))


# Instance globale sélecteur
EXAMPLE_SELECTOR = DynamicExampleSelector(HARENA_INTENT_EXAMPLES)


# Fonctions utilitaires pour compatibilité et facilité d'usage
@lru_cache(maxsize=32)
def get_few_shot_examples_by_intent(
    intent_type: str, 
    max_examples: int = 5,
    complexity: Optional[str] = None
) -> List[Dict[str, any]]:
    """
    Récupère des exemples pour une intention spécifique (cache LRU)
    """
    examples = EXAMPLE_SELECTOR.get_examples_by_intent(intent_type, max_examples)
    
    if complexity:
        examples = [ex for ex in examples if ex.complexity == complexity]
    
    # Conversion au format attendu
    return [
        {
            'input': ex.input,
            'intent': ex.intent,
            'confidence': ex.confidence
        }
        for ex in examples
    ]

def get_balanced_few_shot_examples(
    examples_per_intent: int = 2,
    complexity_mix: bool = True
) -> List[Dict[str, any]]:
    """
    Récupère des exemples équilibrés pour toutes les intentions
    """
    # Utiliser le sélecteur dynamique avec message générique
    selected_examples = EXAMPLE_SELECTOR.select_relevant_examples(
        user_message="exemple générique",
        max_examples=examples_per_intent * 15,  # Estimation large
        balance_intents=True
    )
    
    # Conversion au format attendu
    return [
        {
            'input': ex.input,
            'intent': ex.intent,
            'confidence': ex.confidence
        }
        for ex in selected_examples
    ]

def get_contextual_examples(
    user_message: str,
    max_examples: int = 15
) -> List[Dict[str, any]]:
    """
    Récupère des exemples contextuels basés sur le message utilisateur
    """
    selected_examples = EXAMPLE_SELECTOR.select_relevant_examples(
        user_message=user_message,
        max_examples=max_examples,
        balance_intents=True,
        prefer_high_confidence=True
    )
    
    # Conversion au format attendu
    return [
        {
            'input': ex.input,
            'intent': ex.intent,
            'confidence': ex.confidence
        }
        for ex in selected_examples
    ]

def format_examples_for_prompt(
    examples: List[Dict[str, any]],
    format_style: str = "concise"
) -> str:
    """
    Formate exemples pour inclusion dans prompt
    """
    if format_style == "concise":
        formatted = []
        for ex in examples:
            formatted.append(f'"{ex["input"]}" → {ex["intent"]} ({ex["confidence"]:.2f})')
        return "\n".join(formatted)
    
    elif format_style == "detailed":
        formatted = []
        for ex in examples:
            formatted.append(
                f'Message: "{ex["input"]}" → Intention: {ex["intent"]} '
                f'(Confiance: {ex["confidence"]:.2f})'
            )
        return "\n".join(formatted)
    
    elif format_style == "json":
        import json
        formatted = []
        for ex in examples:
            formatted.append(json.dumps(ex, ensure_ascii=False))
        return "\n".join(formatted)
    
    return format_examples_for_prompt(examples, "concise")

# Statistiques globales pour monitoring
def get_examples_statistics() -> Dict[str, any]:
    """Statistiques complètes des exemples"""
    return {
        "total_examples": len(HARENA_INTENT_EXAMPLES),
        "unique_intents": len(EXAMPLE_SELECTOR.by_intent),
        "avg_confidence": sum(ex.confidence for ex in HARENA_INTENT_EXAMPLES) / len(HARENA_INTENT_EXAMPLES),
        "complexity_distribution": {
            complexity: len(examples) 
            for complexity, examples in EXAMPLE_SELECTOR.by_complexity.items()
        },
        "priority_distribution": {
            f"priority_{priority}": len(examples)
            for priority, examples in EXAMPLE_SELECTOR.by_priority.items()
        },
        "usage_stats": EXAMPLE_SELECTOR.get_usage_statistics(),
        "most_used_examples": list(EXAMPLE_SELECTOR.get_usage_statistics().keys())[:10]
    }