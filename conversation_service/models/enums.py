from enum import Enum


class IntentType(str, Enum):
    """Types d'intention pris en charge par la plateforme."""

    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"
    """Rechercher toutes transactions sans filtre.

    Exemple: "Montre-moi toutes mes transactions."""

    SEARCH_BY_DATE = "SEARCH_BY_DATE"
    """Transactions pour une date ou une période.

    Exemple: "Transactions du 5 mai"."""

    SEARCH_BY_AMOUNT = "SEARCH_BY_AMOUNT"
    """Transactions par montant.

    Exemple: "Transactions supérieures à 50 euros"."""

    SEARCH_BY_MERCHANT = "SEARCH_BY_MERCHANT"
    """Transactions liées à un marchand précis.

    Exemple: "Achats chez Carrefour"."""

    SEARCH_BY_CATEGORY = "SEARCH_BY_CATEGORY"
    """Transactions par catégorie.

    Exemple: "Dépenses en restauration"."""

    SEARCH_BY_AMOUNT_AND_DATE = "SEARCH_BY_AMOUNT_AND_DATE"
    """Combinaison montant et date.

    Exemple: "Dépenses de plus de 100€ en mars"."""

    SEARCH_BY_OPERATION_TYPE = "SEARCH_BY_OPERATION_TYPE"
    """Transactions filtrées par type d'opération.

    Exemple: "Seulement les débits"."""

    SEARCH_BY_TEXT = "SEARCH_BY_TEXT"
    """Recherche textuelle libre.

    Exemple: "Transactions contenant 'abonnement'"."""

    COUNT_TRANSACTIONS = "COUNT_TRANSACTIONS"
    """Compter les transactions correspondant à une requête.

    Exemple: "Combien de transactions ce mois-ci ?"""

    MERCHANT_INQUIRY = "MERCHANT_INQUIRY"
    """Analyse détaillée par marchand.

    Exemple: "Analyse des dépenses chez Amazon"."""

    FILTER_REQUEST = "FILTER_REQUEST"
    """Raffiner une requête transactionnelle.

    Exemple: "Ajoute un filtre pour les débits"."""

    SPENDING_ANALYSIS = "SPENDING_ANALYSIS"
    """Analyse globale des dépenses.

    Exemple: "Analyse mes dépenses du mois"."""

    SPENDING_ANALYSIS_BY_CATEGORY = "SPENDING_ANALYSIS_BY_CATEGORY"
    """Analyse des dépenses par catégorie.

    Exemple: "Analyse de mes dépenses en loisirs"."""

    SPENDING_ANALYSIS_BY_PERIOD = "SPENDING_ANALYSIS_BY_PERIOD"
    """Analyse des dépenses par période.

    Exemple: "Dépenses sur les trois derniers mois"."""

    SPENDING_COMPARISON = "SPENDING_COMPARISON"
    """Comparaison de dépenses entre périodes ou catégories.

    Exemple: "Comparer janvier et février"."""

    TREND_ANALYSIS = "TREND_ANALYSIS"
    """Tendance ou évolution des dépenses.

    Exemple: "Évolution de mes dépenses"."""

    CATEGORY_ANALYSIS = "CATEGORY_ANALYSIS"
    """Répartition des dépenses par catégorie.

    Exemple: "Distribution de mes dépenses"."""

    COMPARISON_QUERY = "COMPARISON_QUERY"
    """Comparaison ciblée entre catégories.

    Exemple: "Restaurants vs courses"."""

    BALANCE_INQUIRY = "BALANCE_INQUIRY"
    """Solde général actuel.

    Exemple: "Quel est mon solde ?"""

    ACCOUNT_BALANCE_SPECIFIC = "ACCOUNT_BALANCE_SPECIFIC"
    """Solde d'un compte précis.

    Exemple: "Solde du compte épargne"."""

    BALANCE_EVOLUTION = "BALANCE_EVOLUTION"
    """Historique ou évolution du solde.

    Exemple: "Comment a évolué mon solde ?"""

    GREETING = "GREETING"
    """Message de salutation utilisateur.

    Exemple: "Bonjour !"""

    CONFIRMATION = "CONFIRMATION"
    """Message de confirmation ou d'accord.

    Exemple: "Merci, parfait"."""

    CLARIFICATION = "CLARIFICATION"
    """Demande de clarification.

    Exemple: "Peux-tu préciser ?"""

    GENERAL_QUESTION = "GENERAL_QUESTION"
    """Question générale hors autre intention.

    Exemple: "Que peux-tu faire ?"""

    TRANSFER_REQUEST = "TRANSFER_REQUEST"
    """Demande de virement (non supportée).

    Exemple: "Fais un virement de 100€"."""

    PAYMENT_REQUEST = "PAYMENT_REQUEST"
    """Demande de paiement d'une facture (non supportée).

    Exemple: "Payer ma facture EDF"."""

    CARD_BLOCK = "CARD_BLOCK"
    """Demande de blocage de carte (non supportée).

    Exemple: "Bloque ma carte"."""

    BUDGET_INQUIRY = "BUDGET_INQUIRY"
    """Question sur le suivi budgétaire (non supportée).

    Exemple: "Où en est mon budget ?"""

    GOAL_TRACKING = "GOAL_TRACKING"
    """Suivi d'objectifs d'épargne (non supporté).

    Exemple: "Progrès vers mon objectif"."""

    EXPORT_REQUEST = "EXPORT_REQUEST"
    """Demande d'export de données (non supportée).

    Exemple: "Export mes transactions"."""

    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    """Requête hors domaine.

    Exemple: "Donne-moi une recette"."""

    UNCLEAR_INTENT = "UNCLEAR_INTENT"
    """Intention ambiguë ou non reconnue.

    Exemple: "Je veux truc"."""

    UNKNOWN = "UNKNOWN"
    """Phrase inintelligible.

    Exemple: "ghjk lkj"."""

    TEST_INTENT = "TEST_INTENT"
    """Message de test.

    Exemple: "[TEST] ping"."""

    ERROR = "ERROR"
    """Entrée corrompue ou erronée.

    Exemple: données illisibles."""


class EntityType(str, Enum):
    """Catégories d'entités extraites d'un message."""

    ACCOUNT = "ACCOUNT"
    """Identifiant de compte bancaire.

    Exemple: "compte courant"."""

    TRANSACTION = "TRANSACTION"
    """Identifiant de transaction.

    Exemple: "txn_123"."""

    MERCHANT = "MERCHANT"
    """Nom d'un marchand.

    Exemple: "Carrefour"."""

    CATEGORY = "CATEGORY"
    """Catégorie de dépense.

    Exemple: "restaurants"."""

    DATE = "DATE"
    """Date explicite.

    Exemple: "2024-05-01"."""

    PERIOD = "PERIOD"
    """Période ou intervalle de dates.

    Exemple: "janvier 2024"."""

    AMOUNT = "AMOUNT"
    """Valeur monétaire.

    Exemple: "50 euros"."""

    OPERATION_TYPE = "OPERATION_TYPE"
    """Type d'opération financière.

    Exemple: "débit"."""

    TEXT = "TEXT"
    """Texte libre de recherche.

    Exemple: "abonnement"."""


class QueryType(str, Enum):
    """Catégories principales des requêtes utilisateur."""

    FINANCIAL_QUERY = "FINANCIAL_QUERY"
    """Questions sur les transactions ou autres données financières."""

    SPENDING_ANALYSIS = "SPENDING_ANALYSIS"
    """Demandes d'analyse des dépenses ou des tendances."""

    ACCOUNT_BALANCE = "ACCOUNT_BALANCE"
    """Questions sur le solde actuel ou historique d'un compte."""

    CONVERSATION = "CONVERSATION"
    """Messages conversationnels généraux comme les salutations ou clarifications."""

    UNSUPPORTED = "UNSUPPORTED"
    """Requêtes hors du périmètre de la plateforme."""


class ConfidenceThreshold(float, Enum):
    """Seuils de confiance utilisés pour les classifications."""

    LOW = 0.3
    """Seuil de confiance faible."""

    MEDIUM = 0.6
    """Seuil de confiance moyen."""

    HIGH = 0.9
    """Seuil de confiance élevé."""

    def __new__(cls, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError("Le seuil doit être compris entre 0.0 et 1.0")
        obj = float.__new__(cls, value)
        obj._value_ = value
        return obj
