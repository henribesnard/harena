"""
🧠 Intent Detection Prompts - IA Principale pour Classification

Ce module contient les prompts optimisés DeepSeek pour la détection d'intention
en mode LLM principal sans recours au pattern matching.

Responsabilité :
- Classification précise des intentions utilisateur
- Extraction des entités financières associées
- Gestion du contexte conversationnel
- Format de sortie strict pour les agents DeepSeek
"""

from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# PROMPTS SYSTÈME PRINCIPAUX
# =============================================================================

INTENT_SYSTEM_PROMPT = """Vous êtes un expert en classification d'intentions pour un assistant financier personnel.

VOTRE MISSION :
Analyser les messages utilisateur et identifier précisément leur intention financière, même quand le message est ambigu, conversationnel, ou complexe.

TAXONOMIE DES INTENTIONS :
1. **TRANSACTION_SEARCH** - Toutes les transactions sans filtre
2. **SEARCH_BY_DATE** - Transactions pour une date ou période
3. **SEARCH_BY_AMOUNT** - Transactions par montant
4. **SEARCH_BY_MERCHANT** - Transactions liées à un marchand
5. **SEARCH_BY_CATEGORY** - Transactions par catégorie
6. **SEARCH_BY_AMOUNT_AND_DATE** - Combinaison montant + date
7. **SEARCH_BY_OPERATION_TYPE** - Transactions filtrées par type d'opération
8. **SEARCH_BY_TEXT** - Recherche textuelle libre
9. **COUNT_TRANSACTIONS** - Compter les transactions
10. **MERCHANT_INQUIRY** - Analyse détaillée par marchand
11. **FILTER_REQUEST** - Raffiner une requête transactionnelle
12. **SPENDING_ANALYSIS** - Analyse globale des dépenses
13. **SPENDING_ANALYSIS_BY_CATEGORY** - Analyse des dépenses par catégorie
14. **SPENDING_ANALYSIS_BY_PERIOD** - Analyse des dépenses par période
15. **SPENDING_COMPARISON** - Comparaison de périodes ou catégories
16. **TREND_ANALYSIS** - Tendance ou évolution des dépenses
17. **CATEGORY_ANALYSIS** - Répartition par catégories
18. **COMPARISON_QUERY** - Comparaison ciblée
19. **BALANCE_INQUIRY** - Solde global actuel
20. **ACCOUNT_BALANCE_SPECIFIC** - Solde d'un compte précis
21. **BALANCE_EVOLUTION** - Historique du solde
22. **GREETING** - Salutation simple
23. **CONFIRMATION** - Remerciement ou acquiescement
24. **CLARIFICATION** - Demande de précision
25. **GENERAL_QUESTION** - Question générale hors finance
26. **TRANSFER_REQUEST** - Demande de virement (non supporté)
27. **PAYMENT_REQUEST** - Paiement de facture (non supporté)
28. **CARD_BLOCK** - Blocage de carte (non supporté)
29. **BUDGET_INQUIRY** - Question sur le budget (non supporté)
30. **GOAL_TRACKING** - Suivi d'objectif d'épargne (non supporté)
31. **EXPORT_REQUEST** - Exporter des transactions (non supporté)
32. **OUT_OF_SCOPE** - Requête hors domaine
33. **UNCLEAR_INTENT** - Intention ambiguë
34. **UNKNOWN** - Phrase inintelligible
35. **TEST_INTENT** - Message de test
36. **ERROR** - Entrée corrompue

ENTITÉS FINANCIÈRES À EXTRAIRE :
- **montants** : 50€, 1000 euros, moins de 100€
- **dates** : hier, janvier, le 15/03, cette semaine
- **marchands** : Carrefour, Amazon, restaurant
- **catégories** : alimentation, transport, loisirs
- **comptes** : compte courant, épargne, carte
- **périodes** : mensuel, hebdomadaire, ce trimestre

FORMAT DE SORTIE STRICT :
Réponds uniquement avec un JSON ayant la structure suivante :
```
{
  "intent_type": "...",
  "intent_category": "...",
  "confidence": 0.0-1.0,
  "entities": [
    {"entity_type": "...", "value": "...", "confidence": 0.0-1.0}
  ],
  "suggested_actions": ["filter_by_amount_greater"]
}
```
Chaque entité doit contenir les champs ``entity_type``, ``value`` et ``confidence``.

Inclure le champ optionnel ``suggested_actions`` lorsque le message contient des
comparatifs de montant ("supérieur", "plus de", "inférieur", "moins de") en
utilisant ``filter_by_amount_greater`` ou ``filter_by_amount_less`` selon le cas.

**Exemple avec montant** :
```
MESSAGE: "transactions supérieures à 100 €"
→ {
  "intent_type": "TRANSACTION_SEARCH",
  "intent_category": "TRANSACTION_SEARCH",
  "confidence": 0.9,
  "entities": [
    {"entity_type": "AMOUNT", "value": "100", "confidence": 0.9}
  ],
  "suggested_actions": ["filter_by_amount_greater"]
}
```

INSTRUCTIONS IMPORTANTES :
- Soyez précis mais pas trop restrictif dans la classification
- Si l'intention est ambiguë, choisissez la plus probable et réduisez la confidence
- Extrayez TOUTES les entités financières même approximatives
- En cas d'incertitude majeure, utilisez "UNCLEAR_INTENT" avec confidence < 0.5
- Respectez EXACTEMENT le format de sortie"""

# =============================================================================
# TEMPLATE UTILISATEUR AVEC CONTEXTE
# =============================================================================

INTENT_USER_TEMPLATE = """Analysez ce message utilisateur et identifiez son intention financière :

MESSAGE: "{user_message}"

{context_section}

Analysez précisément l'intention et extrayez toutes les entités financières pertinentes.
Répondez dans le format requis."""

# =============================================================================
# EXEMPLES FEW-SHOT POUR AMÉLIORER LA PRÉCISION
# =============================================================================

INTENT_EXAMPLES_FEW_SHOT = """EXEMPLES DE CLASSIFICATION :

**Exemple 1 - Recherche par montant :**
MESSAGE: "Transactions de 50 euros"
INTENT_TYPE: SEARCH_BY_AMOUNT
INTENT_CATEGORY: FINANCIAL_QUERY
CONFIDENCE: 0.95
ENTITIES: {"amounts": ["50 euros"]}
SUGGESTED_ACTIONS: []

**Exemple 2 - Recherche par date :**
MESSAGE: "Transactions de mars 2024"
INTENT_TYPE: SEARCH_BY_DATE
INTENT_CATEGORY: FINANCIAL_QUERY
CONFIDENCE: 0.92
ENTITIES: {"dates": ["mars 2024"]}
SUGGESTED_ACTIONS: []

**Exemple 3 - Analyse par catégorie :**
MESSAGE: "Analyse des dépenses alimentaires"
INTENT_TYPE: SPENDING_ANALYSIS_BY_CATEGORY
INTENT_CATEGORY: SPENDING_ANALYSIS
CONFIDENCE: 0.90
ENTITIES: {"categories": ["alimentaire"]}
SUGGESTED_ACTIONS: []

**Exemple 4 - Solde de compte :**
MESSAGE: "Quel est mon solde actuel ?"
INTENT_TYPE: BALANCE_INQUIRY
INTENT_CATEGORY: ACCOUNT_BALANCE
CONFIDENCE: 0.93
ENTITIES: {}
SUGGESTED_ACTIONS: []

**Exemple 5 - Salutation :**
MESSAGE: "Bonjour !"
INTENT_TYPE: GREETING
INTENT_CATEGORY: GREETING
CONFIDENCE: 0.30
ENTITIES: {}
SUGGESTED_ACTIONS: []

**Exemple 6 - Intention ambiguë :**
MESSAGE: "Je ne sais pas, fais quelque chose"
INTENT_TYPE: UNCLEAR_INTENT
INTENT_CATEGORY: UNCLEAR_INTENT
CONFIDENCE: 0.20
ENTITIES: {}
SUGGESTED_ACTIONS: []

**Exemple 7 - Comparatif montant supérieur :**
MESSAGE: "Transactions supérieures à 100 €"
INTENT_TYPE: SEARCH_BY_AMOUNT
INTENT_CATEGORY: FINANCIAL_QUERY
CONFIDENCE: 0.96
ENTITIES: {"amounts": ["100 €"]}
SUGGESTED_ACTIONS: ["filter_by_amount_greater"]

**Exemple 8 - Comparatif montant inférieur :**
MESSAGE: "Transactions inférieures à 50 €"
INTENT_TYPE: SEARCH_BY_AMOUNT
INTENT_CATEGORY: FINANCIAL_QUERY
CONFIDENCE: 0.95
ENTITIES: {"amounts": ["50 €"]}
SUGGESTED_ACTIONS: ["filter_by_amount_less"]
"""

# =============================================================================
# FONCTIONS DE FORMATAGE
# =============================================================================

def format_intent_prompt(user_message: str, context: str = "") -> str:
    """
    Formate le prompt complet pour la détection d'intention.
    
    Args:
        user_message: Message utilisateur à analyser
        context: Contexte conversationnel optionnel
        
    Returns:
        Prompt formaté prêt pour DeepSeek
        
    Example:
        >>> prompt = format_intent_prompt("Mes achats Amazon", "L'utilisateur cherchait ses factures")
        >>> # Utilisation avec DeepSeek client
    """
    if not user_message or not user_message.strip():
        raise ValueError("user_message ne peut pas être vide")
    
    # Construction de la section contexte
    context_section = ""
    if context and context.strip():
        context_section = f"\nCONTEXTE CONVERSATIONNEL:\n{context.strip()}\n"
    
    # Formatage du prompt utilisateur
    user_prompt = INTENT_USER_TEMPLATE.format(
        user_message=user_message.strip(),
        context_section=context_section
    )
    
    return user_prompt

def build_context_summary(conversation_history: List[Dict[str, Any]], max_tokens: int = 500) -> str:
    """
    Construit un résumé du contexte conversationnel en respectant la limite de tokens.
    
    Args:
        conversation_history: Historique des échanges [{"user": "...", "assistant": "..."}]
        max_tokens: Limite approximative en tokens pour le contexte
        
    Returns:
        Résumé contexte formaté ou string vide si pas d'historique
        
    Example:
        >>> history = [{"user": "Mes achats", "assistant": "Voici vos transactions..."}]
        >>> context = build_context_summary(history)
    """
    if not conversation_history:
        return ""
    
    # Estimation approximative : 1 token ≈ 4 caractères en français
    max_chars = max_tokens * 4
    
    context_parts = []
    current_length = 0
    
    # Prendre les échanges les plus récents en premier
    for turn in reversed(conversation_history[-5:]):  # Max 5 derniers échanges
        user_msg = turn.get("user", "")
        assistant_msg = turn.get("assistant", "")
        
        turn_text = f"User: {user_msg}\nAssistant: {assistant_msg[:100]}..."
        turn_length = len(turn_text)
        
        if current_length + turn_length > max_chars:
            break
            
        context_parts.insert(0, turn_text)  # Insérer au début pour garder l'ordre chronologique
        current_length += turn_length
    
    if not context_parts:
        return ""
    
    context_summary = "\n---\n".join(context_parts)
    
    # Alerte si contexte trop long
    if len(conversation_history) > 10:
        logger.warning(f"Contexte conversationnel très long ({len(conversation_history)} échanges). "
                      f"Considérez proposer une nouvelle conversation.")
        context_summary += "\n\n[CONTEXTE TRONQUÉ - Conversation longue détectée]"
    
    return context_summary

def parse_intent_response(response: str) -> Dict[str, Any]:
    """
    Parse la réponse formatée de DeepSeek pour extraire les composants structurés.
    
    Args:
        response: Réponse brute de DeepSeek
        
    Returns:
        Dict avec intent, confidence, entities, reasoning
        
    Raises:
        ValueError: Si le format de réponse est invalide
        
    Example:
        >>> response = "INTENT: transaction_query\\nCONFIDENCE: 0.9\\n..."
        >>> parsed = parse_intent_response(response)
        >>> print(parsed["intent"])  # "transaction_query"
    """
    if not response or not response.strip():
        raise ValueError("Réponse vide de DeepSeek")
    
    try:
        lines = response.strip().split('\n')
        result = {
            "intent": None,
            "confidence": 0.0,
            "entities": {},
            "reasoning": ""
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith("INTENT:"):
                result["intent"] = line.replace("INTENT:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                confidence_str = line.replace("CONFIDENCE:", "").strip()
                result["confidence"] = float(confidence_str)
            elif line.startswith("ENTITIES:"):
                entities_str = line.replace("ENTITIES:", "").strip()
                if entities_str:
                    result["entities"] = json.loads(entities_str)
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()
        
        # Validation des champs obligatoires
        if not result["intent"]:
            raise ValueError("Intent manquant dans la réponse")
        
        if not 0.0 <= result["confidence"] <= 1.0:
            raise ValueError(f"Confidence invalide: {result['confidence']}")
            
        return result
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Erreur parsing réponse intent: {e}")
        logger.error(f"Réponse brute: {response}")
        
        # Gestion d'erreur par défaut
        return {
            "intent": "other",
            "confidence": 0.1,
            "entities": {},
            "reasoning": f"Erreur parsing: {str(e)}"
        }

# =============================================================================
# CONSTANTES UTILES
# =============================================================================

VALID_INTENTS = {
    "transaction_query", "spending_analysis", "budget_inquiry", 
    "category_analysis", "merchant_inquiry", "balance_inquiry",
    "trend_analysis", "comparison_query", "goal_tracking",
    "alert_management", "conversational", "other"
}

FINANCIAL_ENTITY_TYPES = {
    "amounts", "dates", "merchants", "categories", 
    "accounts", "periods", "sentiment", "budget_type", 
    "analysis_type"
}

# Export des éléments principaux
__all__ = [
    "INTENT_SYSTEM_PROMPT",
    "INTENT_USER_TEMPLATE",
    "INTENT_EXAMPLES_FEW_SHOT",
    "format_intent_prompt",
    "build_context_summary",
    "parse_intent_response",
    "VALID_INTENTS",
    "FINANCIAL_ENTITY_TYPES"
]
