"""
🧠 Intent Detection Prompts

Ce module contient les prompts optimisés DeepSeek pour la détection d'intention
dans les conversations financières.

Responsabilité :
- Classification précise des intentions utilisateur
- Extraction des entités financières associées
- Gestion du contexte conversationnel
- Format de sortie standardisé pour les agents AutoGen
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
1. **transaction_query** - Recherche de transactions spécifiques
2. **spending_analysis** - Analyse des dépenses et habitudes
3. **budget_inquiry** - Questions sur budget et planification
4. **category_analysis** - Analyse par catégorie de dépenses
5. **merchant_inquiry** - Questions sur marchands spécifiques
6. **balance_inquiry** - Consultation soldes et positions
7. **trend_analysis** - Analyse tendances et évolutions
8. **comparison_query** - Comparaisons temporelles ou catégorielles
9. **goal_tracking** - Suivi objectifs financiers
10. **alert_management** - Gestion alertes et notifications
11. **conversational** - Échange conversationnel sans intention financière spécifique
12. **other** - Intentions non classifiables dans les catégories précédentes

ENTITÉS FINANCIÈRES À EXTRAIRE :
- **montants** : 50€, 1000 euros, moins de 100€
- **dates** : hier, janvier, le 15/03, cette semaine
- **marchands** : Carrefour, Amazon, restaurant
- **catégories** : alimentation, transport, loisirs
- **comptes** : compte courant, épargne, carte
- **périodes** : mensuel, hebdomadaire, ce trimestre

FORMAT DE RÉPONSE OBLIGATOIRE :
```json
{
  "intent": "[intention_identifiée]",
  "confidence": [0.0-1.0],
  "entities": {json_des_entités_extraites},
  "reasoning": "[explication_courte_du_raisonnement]"
}
```

RÈGLES IMPORTANTES :
- Soyez précis mais pas trop restrictif dans la classification
- Si l'intention est ambiguë, choisissez la plus probable et réduisez la confidence
- Extrayez TOUTES les entités financières même approximatives
- En cas d'incertitude majeure, utilisez "conversational" avec confidence < 0.5
- Gardez le REASONING concis (max 1 phrase)
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

**Exemple 1 - Transaction Query Simple :**
MESSAGE: "Mes achats chez Carrefour le mois dernier"
INTENT: transaction_query
CONFIDENCE: 0.95
ENTITIES: {"merchants": ["Carrefour"], "periods": ["mois dernier"]}
REASONING: Recherche explicite de transactions avec marchand et période spécifiés.

**Exemple 2 - Spending Analysis Complexe :**
MESSAGE: "J'ai l'impression de trop dépenser en restaurant ces derniers temps"
INTENT: spending_analysis
CONFIDENCE: 0.85
ENTITIES: {"categories": ["restaurant"], "periods": ["ces derniers temps"], "sentiment": ["trop dépenser"]}
REASONING: Analyse subjective des habitudes de dépense dans une catégorie.

**Exemple 3 - Budget Inquiry :**
MESSAGE: "Il me reste combien sur mon budget courses ce mois-ci ?"
INTENT: budget_inquiry
CONFIDENCE: 0.90
ENTITIES: {"categories": ["courses"], "periods": ["ce mois-ci"], "budget_type": ["remaining"]}
REASONING: Question directe sur le budget restant dans une catégorie.

**Exemple 4 - Conversational Ambigu :**
MESSAGE: "Salut ! Comment ça va ?"
INTENT: conversational
CONFIDENCE: 0.30
ENTITIES: {}
REASONING: Salutation sans intention financière identifiable.

**Exemple 5 - Trend Analysis avec Montants :**
MESSAGE: "Est-ce que je dépense plus que 500€ par mois en moyenne ?"
INTENT: trend_analysis
CONFIDENCE: 0.88
ENTITIES: {"amounts": ["500€"], "periods": ["par mois"], "analysis_type": ["average", "comparison"]}
REASONING: Analyse comparative des dépenses avec seuil monétaire.

**Exemple 6 - Merchant Inquiry avec Salutation :**
MESSAGE: "Salut, combien ai-je dépensé chez Amazon cette semaine ?"
INTENT: merchant_inquiry
CONFIDENCE: 0.92
ENTITIES: {"merchants": ["Amazon"], "periods": ["cette semaine"]}
REASONING: Recherche de dépenses pour un marchand spécifique après salutation.

**Exemple 7 - Balance Inquiry :**
MESSAGE: "Quel est le solde de mon compte épargne ?"
INTENT: balance_inquiry
CONFIDENCE: 0.93
ENTITIES: {"accounts": ["compte épargne"]}
REASONING: Demande directe de consultation de solde.

**Exemple 8 - Goal Tracking :**
MESSAGE: "Suis-je proche de mon objectif d'épargne de 5000€ ?"
INTENT: goal_tracking
CONFIDENCE: 0.89
ENTITIES: {"amounts": ["5000€"], "goal_type": ["épargne"]}
REASONING: Vérification de la progression vers un objectif financier."""

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
    Parse la réponse JSON de DeepSeek pour extraire les composants structurés.

    Args:
        response: Réponse brute de DeepSeek

    Returns:
        Dict avec intent, confidence, entities, reasoning

    Raises:
        ValueError: Si le format de réponse est invalide

    Example:
        >>> response = '{"intent": "transaction_query", "confidence": 0.9, "entities": {}, "reasoning": "ok"}'
        >>> parsed = parse_intent_response(response)
        >>> print(parsed["intent"])  # "transaction_query"
    """
    if not response or not response.strip():
        raise ValueError("Réponse vide de DeepSeek")

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        data = json.loads(cleaned)
        result = {
            "intent": data.get("intent"),
            "confidence": float(data.get("confidence", 0.0)),
            "entities": data.get("entities", {}),
            "reasoning": data.get("reasoning", ""),
        }
        if not result["intent"]:
            raise ValueError("Intent manquant dans la réponse")
        if not 0.0 <= result["confidence"] <= 1.0:
            raise ValueError(f"Confidence invalide: {result['confidence']}")
        return result
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        logger.error(f"Erreur parsing réponse intent: {e}")
        logger.error(f"Réponse brute: {response}")
        return {
            "intent": "other",
            "confidence": 0.1,
            "entities": {},
            "reasoning": f"Erreur parsing: {str(e)}",
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
