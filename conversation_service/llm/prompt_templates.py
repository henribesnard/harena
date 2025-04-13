"""
Templates de prompts pour différents cas d'usage.

Ce module fournit des templates de prompts pour différentes tâches
comme la classification d'intention, la génération de réponse, etc.
"""

from typing import Dict, Any, List
from ..config.settings import settings
from ..config.constants import DEFAULT_SYSTEM_MESSAGE, INTENT_TYPES


def load_system_prompt() -> str:
    """
    Charge le prompt système par défaut.
    
    Returns:
        str: Prompt système par défaut
    """
    return settings.DEFAULT_SYSTEM_PROMPT or DEFAULT_SYSTEM_MESSAGE


def get_intent_classification_prompt() -> str:
    """
    Génère le prompt pour la classification d'intention.
    
    Returns:
        str: Prompt pour la classification d'intention
    """
    # Créer une description des intentions disponibles
    intent_descriptions = "\n".join([
        f"- {intent}: {details['description']}" +
        f"\n  Exemples: {', '.join(details['examples'][:2])}"
        for intent, details in INTENT_TYPES.items()
    ])
    
    prompt = f"""Vous êtes un assistant spécialisé dans l'analyse des requêtes financières.
Votre tâche est de classifier l'intention de l'utilisateur parmi les catégories suivantes:

{intent_descriptions}

Répondez exactement au format JSON avec les champs suivants:
{{
  "intent": "NOM_DE_L_INTENTION",
  "confidence": 0.XX,
  "entities": {{
    "entity1": "valeur1",
    "entity2": "valeur2"
  }}
}}

Les entités peuvent inclure: date_start, date_end, amount, merchant, category, account_id, etc.
Soyez précis et extrayez toutes les entités pertinentes de la requête.
"""
    
    return prompt


def get_query_building_prompt() -> str:
    """
    Génère le prompt pour la construction de requête.
    
    Returns:
        str: Prompt pour la construction de requête
    """
    prompt = """Vous êtes un assistant spécialisé dans la transformation de requêtes
en langage naturel en requêtes structurées pour un système financier.

Votre tâche est de créer une requête JSON pour notre API de recherche de transactions.
Utilisez uniquement les paramètres mentionnés dans la requête de l'utilisateur.

Paramètres disponibles:
- query: texte de recherche libre
- start_date: date de début au format YYYY-MM-DD
- end_date: date de fin au format YYYY-MM-DD
- min_amount: montant minimum (nombre)
- max_amount: montant maximum (nombre)
- categories: liste d'IDs de catégories
- merchant_names: liste de noms de commerçants
- account_ids: liste d'IDs de comptes

Format de réponse:
{
  "query": "texte",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "min_amount": X.XX,
  "max_amount": X.XX,
  "categories": [ID1, ID2],
  "merchant_names": ["nom1", "nom2"],
  "account_ids": [ID1, ID2]
}

Ne renvoyez que le JSON, sans explications ni formatage supplémentaire.
"""
    
    return prompt


def get_response_generation_prompt(context: Dict[str, Any]) -> str:
    """
    Génère le prompt pour la génération de réponse.
    
    Args:
        context: Contexte de la conversation et données financières
        
    Returns:
        str: Prompt pour la génération de réponse
    """
    # Extraire les informations de contexte
    intent = context.get("intent", "GENERAL_QUERY")
    transaction_data = context.get("transaction_data", {})
    account_data = context.get("account_data", {})
    
    # Construire la section des données disponibles
    data_sections = []
    
    # Ajouter les données de transactions si disponibles
    if transaction_data:
        tx_count = transaction_data.get("count", 0)
        tx_section = f"""Vous disposez de {tx_count} transactions. """
        
        if tx_count > 0:
            tx_section += "Voici un résumé des transactions:\n"
            tx_section += f"- Montant total: {transaction_data.get('total_amount', 0)} €\n"
            tx_section += f"- Période: du {transaction_data.get('start_date', 'N/A')} au {transaction_data.get('end_date', 'N/A')}\n"
            
            if "transactions" in transaction_data and transaction_data["transactions"]:
                tx_section += "Voici quelques transactions à titre d'exemple:\n"
                for i, tx in enumerate(transaction_data["transactions"][:3]):
                    tx_section += f"- {tx.get('date', 'N/A')}: {tx.get('description', 'N/A')} - {tx.get('amount', 0)} €\n"
        
        data_sections.append(tx_section)
    
    # Ajouter les données de compte si disponibles
    if account_data:
        acc_section = "Informations sur les comptes:\n"
        for acc in account_data.get("accounts", []):
            acc_section += f"- {acc.get('name', 'N/A')}: Solde de {acc.get('balance', 0)} €\n"
        
        data_sections.append(acc_section)
    
    # Construire le prompt final
    data_context = "\n\n".join(data_sections) if data_sections else "Aucune donnée financière n'est disponible pour cette requête."
    
    prompt = f"""Vous êtes Harena, un assistant financier intelligent et conversationnel.
Vous aidez l'utilisateur à comprendre et gérer ses finances personnelles.

Vous avez identifié l'intention suivante: {intent}

{data_context}

Répondez de manière conversationnelle, claire et concise. Utilisez un ton amical mais professionnel.
Si vous n'avez pas assez d'informations, n'hésitez pas à le mentionner.
Structurez votre réponse pour faciliter la lecture, mais restez naturel.
Proposez des conseils pertinents lorsque cela est approprié.
"""
    
    return prompt