"""
Exemples few-shot pour classification intentions Harena
"""
from typing import List, Dict

HARENA_INTENT_EXAMPLES = [
    # TRANSACTIONS - SEARCH_BY_MERCHANT
    {"input": "Mes achats Amazon", "intent": "SEARCH_BY_MERCHANT", "confidence": 0.95},
    {"input": "Qu'est-ce que j'ai acheté chez Amazon ?", "intent": "SEARCH_BY_MERCHANT", "confidence": 0.92},
    {"input": "Toutes mes transactions Carrefour", "intent": "SEARCH_BY_MERCHANT", "confidence": 0.94},
    {"input": "Mes dépenses McDonald's", "intent": "SEARCH_BY_MERCHANT", "confidence": 0.93},
    {"input": "Historique Uber Eats", "intent": "SEARCH_BY_MERCHANT", "confidence": 0.91},
    
    # TRANSACTIONS - SEARCH_BY_DATE  
    {"input": "Mes transactions d'hier", "intent": "SEARCH_BY_DATE", "confidence": 0.96},
    {"input": "Qu'est-ce que j'ai dépensé cette semaine ?", "intent": "SEARCH_BY_DATE", "confidence": 0.89},
    {"input": "Mes achats du mois dernier", "intent": "SEARCH_BY_DATE", "confidence": 0.92},
    {"input": "Transactions entre le 1er et le 15 août", "intent": "SEARCH_BY_DATE", "confidence": 0.94},
    {"input": "Mes dépenses depuis janvier", "intent": "SEARCH_BY_DATE", "confidence": 0.88},
    
    # TRANSACTIONS - SEARCH_BY_AMOUNT
    {"input": "Mes gros achats", "intent": "SEARCH_BY_AMOUNT", "confidence": 0.85},
    {"input": "Transactions supérieures à 100€", "intent": "SEARCH_BY_AMOUNT", "confidence": 0.97},
    {"input": "Mes petites dépenses", "intent": "SEARCH_BY_AMOUNT", "confidence": 0.83},
    {"input": "Achats entre 50 et 200 euros", "intent": "SEARCH_BY_AMOUNT", "confidence": 0.95},
    {"input": "Dépenses de moins de 10€", "intent": "SEARCH_BY_AMOUNT", "confidence": 0.92},
    
    # TRANSACTIONS - SEARCH_BY_CATEGORY
    {"input": "Mes dépenses restaurants", "intent": "SEARCH_BY_CATEGORY", "confidence": 0.94},
    {"input": "Combien j'ai dépensé en courses ?", "intent": "SEARCH_BY_CATEGORY", "confidence": 0.89},
    {"input": "Mes frais de transport", "intent": "SEARCH_BY_CATEGORY", "confidence": 0.92},
    {"input": "Dépenses santé et médical", "intent": "SEARCH_BY_CATEGORY", "confidence": 0.91},
    {"input": "Mes achats de vêtements", "intent": "SEARCH_BY_CATEGORY", "confidence": 0.93},
    
    # TRANSACTIONS - MERCHANT_INQUIRY
    {"input": "Analyse détaillée Amazon", "intent": "MERCHANT_INQUIRY", "confidence": 0.88},
    {"input": "Combien je dépense chez McDonald's par mois ?", "intent": "MERCHANT_INQUIRY", "confidence": 0.90},
    {"input": "Statistiques de mes achats Carrefour", "intent": "MERCHANT_INQUIRY", "confidence": 0.87},
    
    # SPENDING_ANALYSIS
    {"input": "Analyse de mes dépenses", "intent": "SPENDING_ANALYSIS", "confidence": 0.94},
    {"input": "Combien j'ai dépensé ce mois ?", "intent": "SPENDING_ANALYSIS", "confidence": 0.91},
    {"input": "Ma consommation financière", "intent": "SPENDING_ANALYSIS", "confidence": 0.86},
    {"input": "Répartition de mes sorties d'argent", "intent": "SPENDING_ANALYSIS", "confidence": 0.88},
    {"input": "Bilan de mes dépenses mensuelles", "intent": "SPENDING_ANALYSIS", "confidence": 0.92},
    
    # SPENDING_ANALYSIS_BY_CATEGORY
    {"input": "Répartition par catégories", "intent": "SPENDING_ANALYSIS_BY_CATEGORY", "confidence": 0.95},
    {"input": "Mes dépenses par type", "intent": "SPENDING_ANALYSIS_BY_CATEGORY", "confidence": 0.89},
    {"input": "Où va mon argent ?", "intent": "SPENDING_ANALYSIS_BY_CATEGORY", "confidence": 0.84},
    
    # SPENDING_COMPARISON
    {"input": "Comparaison avec le mois dernier", "intent": "SPENDING_COMPARISON", "confidence": 0.93},
    {"input": "Est-ce que je dépense plus qu'avant ?", "intent": "SPENDING_COMPARISON", "confidence": 0.87},
    {"input": "Évolution de mes dépenses", "intent": "SPENDING_COMPARISON", "confidence": 0.90},
    
    # BALANCE_INQUIRY
    {"input": "Mon solde", "intent": "BALANCE_INQUIRY", "confidence": 0.98},
    {"input": "Combien j'ai sur mon compte ?", "intent": "BALANCE_INQUIRY", "confidence": 0.96},
    {"input": "Solde actuel", "intent": "BALANCE_INQUIRY", "confidence": 0.97},
    {"input": "Ma situation financière", "intent": "BALANCE_INQUIRY", "confidence": 0.85},
    
    # GREETING
    {"input": "Bonjour", "intent": "GREETING", "confidence": 0.99},
    {"input": "Salut", "intent": "GREETING", "confidence": 0.98},
    {"input": "Hello", "intent": "GREETING", "confidence": 0.96},
    {"input": "Bonsoir", "intent": "GREETING", "confidence": 0.97},
    
    # CONFIRMATION
    {"input": "Merci", "intent": "CONFIRMATION", "confidence": 0.95},
    {"input": "Parfait", "intent": "CONFIRMATION", "confidence": 0.92},
    {"input": "C'est bon", "intent": "CONFIRMATION", "confidence": 0.89},
    {"input": "OK", "intent": "CONFIRMATION", "confidence": 0.94},
    
    # NON SUPPORTÉES
    {"input": "Faire un virement", "intent": "TRANSFER_REQUEST", "confidence": 0.96},
    {"input": "Payer ma facture", "intent": "PAYMENT_REQUEST", "confidence": 0.94},
    {"input": "Bloquer ma carte", "intent": "CARD_BLOCK", "confidence": 0.97},
    {"input": "Où en est mon budget ?", "intent": "BUDGET_INQUIRY", "confidence": 0.91},
    
    # UNCLEAR_INTENT
    {"input": "Euh... je sais pas", "intent": "UNCLEAR_INTENT", "confidence": 0.88},
    {"input": "Peux-tu m'aider ?", "intent": "UNCLEAR_INTENT", "confidence": 0.82},
    {"input": "azerty", "intent": "UNKNOWN", "confidence": 0.95},
    {"input": "", "intent": "UNKNOWN", "confidence": 0.99}
]

def get_few_shot_examples_by_intent(intent_type: str, max_examples: int = 5) -> List[Dict]:
    """Récupère des exemples pour une intention spécifique"""
    examples = [ex for ex in HARENA_INTENT_EXAMPLES if ex["intent"] == intent_type]
    return examples[:max_examples]

def get_balanced_few_shot_examples(examples_per_intent: int = 2) -> List[Dict]:
    """Récupère des exemples équilibrés pour toutes les intentions"""
    balanced_examples = []
    unique_intents = list(set(ex["intent"] for ex in HARENA_INTENT_EXAMPLES))
    
    for intent in unique_intents:
        intent_examples = get_few_shot_examples_by_intent(intent, examples_per_intent)
        balanced_examples.extend(intent_examples)
    
    return balanced_examples

def get_high_confidence_examples(min_confidence: float = 0.9) -> List[Dict]:
    """Récupère exemples avec confiance élevée"""
    return [ex for ex in HARENA_INTENT_EXAMPLES if ex["confidence"] >= min_confidence]

def format_examples_for_prompt(examples: List[Dict]) -> str:
    """Formate exemples pour inclusion dans prompt"""
    formatted = []
    for ex in examples:
        formatted.append(
            f"Message: \"{ex['input']}\" → Intention: {ex['intent']} (Confiance: {ex['confidence']})"
        )
    return "\n".join(formatted)