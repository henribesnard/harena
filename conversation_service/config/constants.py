"""
Constants used throughout the conversation service.

Ce module définit les constantes utilisées par le service de conversation
pour garantir la cohérence et éviter la duplication.
"""

# Limites et valeurs par défaut
MAX_TITLE_LENGTH = 255
DEFAULT_CONVERSATIONS_LIMIT = 50
MAX_CONVERSATIONS_LIMIT = 200
CACHE_EXPIRATION_SECONDS = 3600  # 1 heure

# Rôles de message
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant" 
ROLE_SYSTEM = "system"

# États de conversation
CONVERSATION_ACTIVE = "active"
CONVERSATION_ARCHIVED = "archived"
CONVERSATION_DELETED = "deleted"

# Types d'intentions
INTENT_TYPES = {
    "CHECK_BALANCE": {
        "description": "Vérifier le solde d'un compte",
        "examples": ["Quel est mon solde ?", "Combien d'argent ai-je sur mon compte ?"]
    },
    "SEARCH_TRANSACTION": {
        "description": "Rechercher des transactions",
        "examples": ["Montre mes dépenses chez Carrefour", "Combien ai-je dépensé le mois dernier ?"]
    },
    "ANALYZE_SPENDING": {
        "description": "Analyser les dépenses par catégorie",
        "examples": ["Quelle est ma catégorie de dépense principale ?", "Analyse mes dépenses du mois"]
    },
    "FORECAST_BALANCE": {
        "description": "Prévisions de solde futur",
        "examples": ["Comment sera mon solde à la fin du mois ?", "Puis-je me permettre d'acheter X ?"]
    },
    "SUBSCRIPTION_MANAGEMENT": {
        "description": "Gestion des abonnements et paiements récurrents",
        "examples": ["Quels sont mes abonnements ?", "Combien je paye pour Netflix ?"]
    },
    "SAVINGS_GOAL": {
        "description": "Définition et suivi d'objectifs d'épargne",
        "examples": ["Comment économiser pour des vacances ?", "Quel est l'état de mon objectif vacances ?"]
    },
    "BUDGET_TRACKING": {
        "description": "Suivi de budget",
        "examples": ["Suis-je dans mon budget alimentaire ?", "Montre mon budget de ce mois"]
    },
    "GENERAL_QUERY": {
        "description": "Requête générale sur les finances",
        "examples": ["Explique-moi ce qu'est un REER", "Comment fonctionne le crédit ?"]
    },
    "ACCOUNT_INFO": {
        "description": "Informations sur les comptes",
        "examples": ["Quels comptes ai-je ?", "Montre-moi mes comptes d'épargne"]
    },
    "HELP": {
        "description": "Demande d'aide sur l'utilisation de l'assistant",
        "examples": ["Comment ça marche ?", "Que peux-tu faire ?"]
    }
}

# Formats de dates
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Messages système par défaut
DEFAULT_SYSTEM_MESSAGE = """Vous êtes Harena, un assistant financier intelligent conçu pour aider les utilisateurs à comprendre et gérer leurs finances personnelles. Vous êtes conversationnel, précis, et vous vous concentrez sur les données financières de l'utilisateur pour fournir des réponses et analyses personnalisées.

Vos capacités incluent:
- Rechercher et analyser les transactions de l'utilisateur
- Fournir des résumés des dépenses et revenus
- Aider à la budgétisation et à la planification financière
- Détecter des tendances dans les habitudes de dépenses
- Répondre aux questions générales sur la finance personnelle

Vous répondez toujours en français et de manière conversationnelle. Vous utilisez les données réelles de l'utilisateur quand elles sont disponibles, et vous êtes transparent quand vous n'avez pas accès à certaines informations.
"""

# Limites de tokens
MAX_TOKENS_PER_REQUEST = 8000
MAX_TOKENS_RESPONSE = 4000
MAX_HISTORY_TOKENS = 4000

# Templates de réponse pour les erreurs courantes
ERROR_MESSAGES = {
    "no_data": "Je n'ai pas pu trouver de données pour cette requête. Pourriez-vous reformuler ou préciser votre demande ?",
    "service_unavailable": "Je suis désolé, mais je ne peux pas accéder à cette information pour le moment. Veuillez réessayer plus tard.",
    "rate_limited": "Vous avez effectué trop de requêtes en peu de temps. Veuillez patienter un moment avant de réessayer.",
    "unauthorized": "Vous n'êtes pas autorisé à accéder à ces informations. Veuillez vous connecter ou vérifier vos autorisations.",
    "invalid_request": "Je n'ai pas compris votre demande. Pourriez-vous la reformuler ?",
    "generic_error": "Une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer."
}

# Langues prises en charge
SUPPORTED_LANGUAGES = ["fr", "en"]
DEFAULT_LANGUAGE = "fr"