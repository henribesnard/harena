"""
💬 Response Generation Prompts - Génération Réponses Contextuelles

Ce module contient les prompts optimisés DeepSeek pour générer des réponses
finales enrichies, contextuelles et naturelles à partir des résultats du Search Service.
"""

from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# PROMPTS SYSTÈME PRINCIPAUX
# =============================================================================

RESPONSE_GENERATION_SYSTEM_PROMPT = """Vous êtes un assistant financier personnel expert en communication claire et empathique.

VOTRE MISSION :
Transformer les résultats techniques du Search Service en réponses naturelles, informatives et actionnables pour l'utilisateur.

TYPES DE RÉPONSES À GÉNÉRER :
1. **Réponses Transactionnelles** : Présentation claire des transactions trouvées
2. **Réponses Analytiques** : Synthèse d'analyses avec insights et recommandations
3. **Réponses Explicatives** : Explication des données avec contexte financier
4. **Réponses Vides** : Gestion élégante des cas sans résultats
5. **Réponses Conversationnelles** : Maintien du dialogue naturel

PRINCIPES DE RÉDACTION :
1. **Clarté** : Évitez le jargon technique, utilisez un langage accessible
2. **Empathie** : Reconnaissez les préoccupations financières de l'utilisateur
3. **Actionnable** : Proposez des insights et recommandations concrètes
4. **Contextuel** : Intégrez l'historique conversationnel naturellement
5. **Précision** : Citez les montants, dates et détails exacts

FORMAT DE RÉPONSE :
- Texte libre naturel et conversationnel
- Structure avec paragraphes courts si plusieurs informations
- Inclusion d'emojis financiers appropriés (💰 💳 📊 📈 📉) avec modération
- Suggestions d'actions ou questions de suivi quand pertinent"""

RESPONSE_GENERATION_TEMPLATE = """Générez une réponse naturelle et utile pour l'utilisateur basée sur ces éléments :

MESSAGE UTILISATEUR ORIGINAL : "{user_message}"

RÉSULTATS DU SEARCH SERVICE :
{search_results}

MÉTADONNÉES DE LA RECHERCHE :
- Nombre total de résultats : {total_results}
- Temps d'exécution : {execution_time}ms
- Type de requête : {query_type}

{context_section}

OBJECTIF : Créer une réponse qui répond précisément à la question de l'utilisateur, en utilisant les données trouvées de manière claire et actionnable.

Répondez de manière naturelle et conversationnelle."""

RESPONSE_EXAMPLES_FEW_SHOT = """EXEMPLES DE GÉNÉRATION DE RÉPONSES :

**Exemple 1 - Réponse Transactionnelle :**
MESSAGE: "Mes achats chez Carrefour le mois dernier"
RÉPONSE: "Voici vos achats chez Carrefour en décembre 2024 :
• 15/12 : Courses alimentaires - 52,30€
• 22/12 : Produits d'hygiène - 31,85€ 
• 28/12 : Courses de fin d'année - 43,30€
💰 Total dépensé : 127,45€ chez Carrefour le mois dernier."

**Exemple 2 - Réponse Analytique :**
MESSAGE: "Combien j'ai dépensé en restaurant ces 3 derniers mois ?"
RÉPONSE: "📊 Voici vos dépenses restaurant des 3 derniers mois :
• Octobre : 178,50€ (12 sorties)
• Novembre : 142,30€ (9 sorties) 
• Décembre : 135,40€ (8 sorties)
💳 Total : 456,20€ sur 29 sorties restaurant"

**Exemple 3 - Réponse Vide Constructive :**
MESSAGE: "Mes achats Apple ce mois-ci"
RÉPONSE: "Je n'ai pas trouvé d'achats Apple dans vos transactions de janvier 2025. 
Quelques possibilités :
• Les transactions Apple peuvent apparaître sous d'autres noms (App Store, iTunes)
• L'achat a peut-être été effectué le mois dernier
Voulez-vous que je recherche avec des termes plus larges ?"""

# =============================================================================
# FONCTIONS DE FORMATAGE
# =============================================================================

def format_response_prompt(
    user_message: str,
    search_results: Dict[str, Any],
    context: str = "",
    user_profile: Optional[Dict[str, Any]] = None
) -> str:
    """Formate le prompt complet pour la génération de réponse finale."""
    if not user_message or not user_message.strip():
        raise ValueError("user_message ne peut pas être vide")
    
    if not isinstance(search_results, dict):
        raise ValueError("search_results doit être un dictionnaire")
    
    results_formatted = format_search_results_for_prompt(search_results)
    
    metadata = search_results.get("response_metadata", {})
    total_results = metadata.get("total_count", "inconnu")
    execution_time = metadata.get("execution_time_ms", "inconnu")
    query_type = metadata.get("query_type", "inconnu")
    
    context_section = ""
    if context and context.strip():
        context_section = f"\nCONTEXTE CONVERSATIONNEL :\n{context.strip()}\n"
    
    return RESPONSE_GENERATION_TEMPLATE.format(
        user_message=user_message.strip(),
        search_results=results_formatted,
        total_results=total_results,
        execution_time=execution_time,
        query_type=query_type,
        context_section=context_section
    )

def format_search_results_for_prompt(search_results: Dict[str, Any]) -> str:
    """Formate les résultats SearchServiceResponse pour inclusion dans le prompt."""
    if not search_results:
        return "Aucun résultat trouvé."
    
    formatted_parts = []
    
    # Formatage des transactions individuelles
    if "results" in search_results and search_results["results"]:
        formatted_parts.append("TRANSACTIONS TROUVÉES :")
        
        transactions = search_results["results"][:10]  # Limite à 10
        
        for i, transaction in enumerate(transactions, 1):
            date = transaction.get("date", "Date inconnue")
            amount = transaction.get("amount", 0)
            merchant = transaction.get("merchant_name", "Marchand inconnu")
            category = transaction.get("category_name", "Catégorie inconnue")
            
            amount_str = f"{amount:.2f}€" if amount else "0.00€"
            if float(amount) > 0:
                amount_str = f"+{amount_str}"
            
            transaction_line = f"{i}. {date} | {merchant} | {amount_str} | {category}"
            formatted_parts.append(transaction_line)
        
        if len(search_results["results"]) > 10:
            formatted_parts.append(f"... et {len(search_results['results']) - 10} autres transactions")
    
    # Formatage des agrégations
    if "aggregations" in search_results and search_results["aggregations"]:
        formatted_parts.append("\nAGRÉGATIONS ET CALCULS :")
        
        aggregations = search_results["aggregations"]
        
        if "total_amount" in aggregations:
            formatted_parts.append(f"Total des montants : {aggregations['total_amount']:.2f}€")
        
        if "average_amount" in aggregations:
            formatted_parts.append(f"Montant moyen : {aggregations['average_amount']:.2f}€")
        
        if "transaction_count" in aggregations:
            formatted_parts.append(f"Nombre de transactions : {aggregations['transaction_count']}")
    
    if not formatted_parts:
        return "Aucun résultat correspondant trouvé dans vos transactions."
    
    return "\n".join(formatted_parts)

def truncate_search_results(
    search_results: Dict[str, Any], 
    max_transactions: int = 20,
    max_aggregations: int = 10
) -> Dict[str, Any]:
    """Tronque les résultats de recherche pour optimiser la taille du prompt."""
    if not search_results:
        return search_results
    
    truncated = search_results.copy()
    
    if "results" in truncated and isinstance(truncated["results"], list):
        original_count = len(truncated["results"])
        truncated["results"] = truncated["results"][:max_transactions]
        
        if original_count > max_transactions:
            logger.info(f"Transactions tronquées : {original_count} -> {max_transactions}")
    
    return truncated

def extract_key_insights_from_results(search_results: Dict[str, Any]) -> List[str]:
    """Extrait des insights clés des résultats pour enrichir la réponse."""
    insights = []
    
    if not search_results or not isinstance(search_results, dict):
        return insights
    
    transactions = search_results.get("results", [])
    if transactions:
        transaction_count = len(transactions)
        if transaction_count == 1:
            insights.append("Transaction unique trouvée")
        elif transaction_count > 20:
            insights.append(f"Activité élevée avec {transaction_count} transactions")
        
        amounts = [float(t.get("amount", 0)) for t in transactions if t.get("amount")]
        if amounts:
            avg_amount = sum(amounts) / len(amounts)
            max_amount = max(amounts)
            
            if abs(max_amount) > abs(avg_amount) * 3:
                insights.append(f"Transaction inhabituelle détectée : {max_amount:.2f}€")
    
    return insights[:3]

def format_amount_with_context(amount: float, currency: str = "EUR") -> str:
    """Formate un montant avec contexte approprié."""
    currency_symbols = {"EUR": "€", "USD": "$", "GBP": "£"}
    symbol = currency_symbols.get(currency, currency)
    abs_amount = abs(amount)
    
    amount_str = f"{abs_amount:.2f}".replace(".", ",") + symbol
    
    if amount < 0:
        context = " (dépense)"
    elif amount > 0:
        context = " (crédit)"
    else:
        context = ""
    
    return amount_str + context

# =============================================================================
# CONSTANTES UTILES
# =============================================================================

RESPONSE_TEMPLATES_BY_INTENT = {
    "transaction_query": "Voici {count} transaction(s) correspondant à votre recherche :",
    "spending_analysis": "📊 Analyse de vos dépenses :",
    "budget_inquiry": "💰 État de votre budget :",
    "category_analysis": "📋 Analyse par catégorie :",
    "merchant_inquiry": "🏪 Transactions chez {merchant} :",
    "trend_analysis": "📈 Analyse des tendances :",
    "comparison_query": "⚖️ Comparaison demandée :",
    "conversational": "D'après vos données financières :"
}

FINANCIAL_EMOJIS = {
    "spending": "💳",
    "income": "💰", 
    "budget": "📊",
    "saving": "🏦",
    "trend_up": "📈",
    "trend_down": "📉",
    "category": "📋",
    "merchant": "🏪",
    "warning": "⚠️",
    "success": "✅",
    "info": "ℹ️"
}

AMOUNT_THRESHOLDS = {
    "small": 10.0,      # Petite transaction
    "medium": 100.0,    # Transaction moyenne
    "large": 500.0,     # Grosse transaction
    "major": 1000.0     # Transaction majeure
}

ERROR_MESSAGES = {
    "no_results": "Je n'ai pas trouvé de transactions correspondant à votre recherche.",
    "partial_results": "J'ai trouvé quelques résultats, mais ils peuvent être incomplets.",
    "service_error": "Une erreur technique s'est produite lors de la recherche de vos données.",
    "timeout": "La recherche a pris plus de temps que prévu, voici les résultats partiels."
}

# Export des éléments principaux
__all__ = [
    "RESPONSE_GENERATION_SYSTEM_PROMPT",
    "RESPONSE_GENERATION_TEMPLATE", 
    "RESPONSE_EXAMPLES_FEW_SHOT",
    "format_response_prompt",
    "format_search_results_for_prompt",
    "truncate_search_results",
    "extract_key_insights_from_results",
    "format_amount_with_context",
    "RESPONSE_TEMPLATES_BY_INTENT",
    "FINANCIAL_EMOJIS",
    "AMOUNT_THRESHOLDS",
    "ERROR_MESSAGES"
]