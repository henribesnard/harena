"""Utilities and few-shot prompts for intent classification.

This module defines the intent taxonomy, grouped categories and
few-shot examples used for intent detection.  It also exposes helper
functions to build prompts and perform basic preprocessing like
salutation detection to support more advanced classification such as
multi-intent messages or ambiguous queries.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Tuple

from conversation_service.models.enums import IntentType

# ---------------------------------------------------------------------------
# Intent groups as defined in ``INTENTS.md``
INTENT_GROUPS: Dict[str, List[IntentType]] = {
    "TRANSACTIONS": [
        IntentType.TRANSACTION_SEARCH,
        IntentType.SEARCH_BY_DATE,
        IntentType.SEARCH_BY_AMOUNT,
        IntentType.SEARCH_BY_MERCHANT,
        IntentType.SEARCH_BY_CATEGORY,
        IntentType.SEARCH_BY_AMOUNT_AND_DATE,
        IntentType.SEARCH_BY_OPERATION_TYPE,
        IntentType.SEARCH_BY_TEXT,
        IntentType.COUNT_TRANSACTIONS,
        IntentType.MERCHANT_INQUIRY,
        IntentType.FILTER_REQUEST,
    ],
    "SPENDING_ANALYSIS": [
        IntentType.SPENDING_ANALYSIS,
        IntentType.SPENDING_ANALYSIS_BY_CATEGORY,
        IntentType.SPENDING_ANALYSIS_BY_PERIOD,
        IntentType.SPENDING_COMPARISON,
        IntentType.TREND_ANALYSIS,
        IntentType.CATEGORY_ANALYSIS,
        IntentType.COMPARISON_QUERY,
    ],
    "BALANCE": [
        IntentType.BALANCE_INQUIRY,
        IntentType.ACCOUNT_BALANCE_SPECIFIC,
        IntentType.BALANCE_EVOLUTION,
    ],
    "CONVERSATIONAL": [
        IntentType.GREETING,
        IntentType.CONFIRMATION,
        IntentType.CLARIFICATION,
        IntentType.GENERAL_QUESTION,
    ],
    "UNSUPPORTED_UNCLEAR": [
        IntentType.TRANSFER_REQUEST,
        IntentType.PAYMENT_REQUEST,
        IntentType.CARD_BLOCK,
        IntentType.BUDGET_INQUIRY,
        IntentType.GOAL_TRACKING,
        IntentType.EXPORT_REQUEST,
        IntentType.OUT_OF_SCOPE,
        IntentType.UNCLEAR_INTENT,
        IntentType.UNKNOWN,
        IntentType.TEST_INTENT,
        IntentType.ERROR,
    ],
}

# ---------------------------------------------------------------------------
# Few-shot examples per intent.  Each intent contains between three and
# five examples for training the classifier.  In total this provides more
# than 60 examples as required.
FEW_SHOT_EXAMPLES: Dict[IntentType, List[str]] = {
    IntentType.TRANSACTION_SEARCH: [
        "Montre-moi toutes mes transactions.",
        "Liste toutes mes opérations.",
        "Quelles sont toutes mes dépenses récentes ?",
    ],
    IntentType.SEARCH_BY_DATE: [
        "Transactions du 5 mai.",
        "Opérations entre janvier et mars.",
        "Que s'est-il passé le 10/02 ?",
    ],
    IntentType.SEARCH_BY_AMOUNT: [
        "Transactions supérieures à 50€.",
        "Opérations inférieures à 20 euros.",
        "Dépenses d'au moins 100 euros.",
    ],
    IntentType.SEARCH_BY_MERCHANT: [
        "Achats chez Carrefour.",
        "Paiements à Amazon.",
        "Transactions chez SNCF.",
    ],
    IntentType.SEARCH_BY_CATEGORY: [
        "Dépenses en restauration.",
        "Achats dans la catégorie loisirs.",
        "Transactions liées aux voyages.",
    ],
    IntentType.SEARCH_BY_AMOUNT_AND_DATE: [
        "Dépenses de plus de 100€ en mars.",
        "Transactions supérieures à 50 euros en janvier.",
        "Opérations d'au moins 200€ la semaine dernière.",
    ],
    IntentType.SEARCH_BY_OPERATION_TYPE: [
        "Seulement les débits.",
        "Montre les crédits.",
        "Transactions par carte uniquement.",
    ],
    IntentType.SEARCH_BY_TEXT: [
        "Transactions contenant 'abonnement'.",
        "Opérations avec le mot 'loyer'.",
        "Recherche les transactions marquées 'frais bancaires'.",
    ],
    IntentType.COUNT_TRANSACTIONS: [
        "Combien de transactions ce mois-ci ?",
        "Nombre d'opérations en avril.",
        "Compte le total de mes achats récents.",
    ],
    IntentType.MERCHANT_INQUIRY: [
        "Analyse des dépenses chez Amazon.",
        "Quel est mon total chez Uber ?",
        "Dépenses détaillées pour Monoprix.",
    ],
    IntentType.FILTER_REQUEST: [
        "Ajoute un filtre pour les débits.",
        "Limite aux transactions en ligne.",
        "Peux-tu filtrer par carte bleue ?",
    ],
    IntentType.SPENDING_ANALYSIS: [
        "Analyse mes dépenses du mois.",
        "Fais un bilan de mes dépenses.",
        "Je veux un résumé de mes dépenses totales.",
    ],
    IntentType.SPENDING_ANALYSIS_BY_CATEGORY: [
        "Analyse de mes dépenses en loisirs.",
        "Combien ai-je dépensé pour le logement ?",
        "Répartition de mes achats par catégorie.",
    ],
    IntentType.SPENDING_ANALYSIS_BY_PERIOD: [
        "Dépenses sur les trois derniers mois.",
        "Analyse des dépenses de janvier à mars.",
        "Bilan de mes dépenses de l'année dernière.",
    ],
    IntentType.SPENDING_COMPARISON: [
        "Comparer janvier et février.",
        "Mes dépenses de ce mois versus le mois dernier.",
        "Compare mes frais de transport et de nourriture.",
    ],
    IntentType.TREND_ANALYSIS: [
        "Évolution de mes dépenses.",
        "Tendance de mes dépenses sur l'année.",
        "Ma dépense mensuelle augmente-t-elle ?",
    ],
    IntentType.CATEGORY_ANALYSIS: [
        "Distribution de mes dépenses.",
        "Répartition par catégorie.",
        "Quel pourcentage pour les loisirs ?",
    ],
    IntentType.COMPARISON_QUERY: [
        "Restaurants vs courses.",
        "Compare dépenses carburant et transport public.",
        "Qui coûte plus: shopping ou nourriture ?",
    ],
    IntentType.BALANCE_INQUIRY: [
        "Quel est mon solde ?",
        "Donne-moi mon solde actuel.",
        "Combien me reste-t-il sur mon compte ?",
    ],
    IntentType.ACCOUNT_BALANCE_SPECIFIC: [
        "Solde du compte épargne.",
        "Quel est le solde de mon compte courant ?",
        "Montre le solde de ma carte de crédit.",
    ],
    IntentType.BALANCE_EVOLUTION: [
        "Comment a évolué mon solde ?",
        "Historique de mon solde sur trois mois.",
        "Mon solde a-t-il augmenté depuis janvier ?",
    ],
    IntentType.GREETING: [
        "Bonjour !",
        "Salut, ça va ?",
        "Coucou.",
    ],
    IntentType.CONFIRMATION: [
        "Merci, parfait.",
        "C'est bon pour moi.",
        "D'accord, merci.",
    ],
    IntentType.CLARIFICATION: [
        "Peux-tu préciser ?",
        "Tu peux être plus clair ?",
        "Que veux-tu dire exactement ?",
    ],
    IntentType.GENERAL_QUESTION: [
        "Que peux-tu faire ?",
        "Comment ça marche ?",
        "Peux-tu m'aider ?",
    ],
    IntentType.TRANSFER_REQUEST: [
        "Fais un virement de 100€.",
        "Transfère 50 euros à Paul.",
        "Je veux envoyer de l'argent.",
    ],
    IntentType.PAYMENT_REQUEST: [
        "Payer ma facture EDF.",
        "Règle ma note d'électricité.",
        "Peux-tu payer cette facture ?",
    ],
    IntentType.CARD_BLOCK: [
        "Bloque ma carte.",
        "Je veux opposer ma carte.",
        "Désactive ma carte de crédit.",
    ],
    IntentType.BUDGET_INQUIRY: [
        "Où en est mon budget ?",
        "Montre-moi mon budget mensuel.",
        "Ai-je dépassé mon budget ?",
    ],
    IntentType.GOAL_TRACKING: [
        "Progrès vers mon objectif d'épargne.",
        "Combien manque-t-il pour mon voyage ?",
        "Où en est mon objectif de 5000€ ?",
    ],
    IntentType.EXPORT_REQUEST: [
        "Export mes transactions.",
        "Peux-tu me donner un fichier CSV de mes dépenses ?",
        "Télécharge mes opérations.",
    ],
    IntentType.OUT_OF_SCOPE: [
        "Donne-moi une recette.",
        "Quel temps fait-il ?",
        "Raconte une blague.",
    ],
    IntentType.UNCLEAR_INTENT: [
        "Je veux quelque chose.",
        "Tu sais, le truc là.",
        "Fais ce que tu sais.",
    ],
    IntentType.UNKNOWN: [
        "hjkslq sdlj.",
        "???!!!",
        "lalalaaaaa.",
    ],
    IntentType.TEST_INTENT: [
        "[TEST] ping",
        "test: vérifier",
        "PING de test",
    ],
    IntentType.ERROR: [
        "[ERREUR] données illisibles",
        "message corrompu ###",
        "<<input invalide>>",
    ],
}

# ---------------------------------------------------------------------------
# Salutation detection and prompt generation
GREETING_KEYWORDS = {"bonjour", "salut", "coucou", "hello", "hey"}
GREETING_RE = re.compile(r"\b(" + "|".join(GREETING_KEYWORDS) + r")\b", re.IGNORECASE)


def detect_greeting(message: str) -> Tuple[bool, str]:
    """Detect and strip greetings from ``message``.

    Returns a tuple ``(has_greeting, cleaned_message)`` where
    ``has_greeting`` indicates if a salutation was found.
    """

    match = GREETING_RE.search(message)
    if not match:
        return False, message
    cleaned = GREETING_RE.sub("", message).strip()
    return True, cleaned


def _format_examples() -> str:
    """Format few-shot examples for inclusion in prompts."""

    lines: List[str] = []
    for intent, examples in FEW_SHOT_EXAMPLES.items():
        for ex in examples:
            lines.append(f"Utilisateur: {ex}\nIntention: {intent.value}")
    return "\n".join(lines)


EXAMPLES_TEXT = _format_examples()


def build_intent_prompt(message: str) -> str:
    """Build the full classification prompt for ``message``.

    The prompt lists the taxonomy, contains few-shot examples and
    instructions on how to handle ambiguities, multi-intent requests and
    greetings.  The caller should run :func:`detect_greeting` beforehand
    if it wishes to treat greetings separately.
    """

    taxonomy_lines: List[str] = []
    for group, intents in INTENT_GROUPS.items():
        taxonomy_lines.append(f"{group}:")
        taxonomy_lines.extend(f"  - {intent.value}" for intent in intents)
    taxonomy = "\n".join(taxonomy_lines)

    instructions = (
        "Tu es un classifieur d'intentions bancaires. Analyse le message "
        "et retourne toutes les intentions correspondantes au format JSON.\n"
        "- Renvoie toutes les intentions présentes dans le message.\n"
        "- Si une salutation est combinée à une autre intention, inclue \"GREETING\".\n"
        "- En cas de plusieurs demandes distinctes, retourne chaque intention.\n"
        "- Si la requête est ambiguë ou ne correspond à rien, utilise \"UNCLEAR_INTENT\".\n"
        "- Pour les fonctionnalités non prises en charge, utilise l'intention appropriée.\n"
        "Réponds uniquement par une liste JSON d'intentions."
    )

    return (
        f"{instructions}\n\nTaxonomie:\n{taxonomy}\n\nExemples:\n"
        f"{EXAMPLES_TEXT}\n\nMessage utilisateur: {message}\nIntention(s):"
    )


def classify_message(message: str, llm_client: Callable[[str], str]) -> List[IntentType]:
    """Classify ``message`` using ``llm_client`` for the heavy lifting.

    The function performs basic preprocessing to handle greetings and
    delegates the remaining text to ``llm_client`` which should execute a
    language model call with the prompt produced by
    :func:`build_intent_prompt`.  The LLM is expected to return a JSON
    array of intent names.
    """

    intents: List[IntentType] = []
    has_greeting, cleaned = detect_greeting(message)
    if has_greeting:
        intents.append(IntentType.GREETING)

    prompt = build_intent_prompt(cleaned or message)
    raw = llm_client(prompt)
    try:
        data = json.loads(raw)
        for item in data:
            try:
                intent = IntentType(item)
            except ValueError:
                intent = IntentType.UNCLEAR_INTENT
            if intent not in intents:
                intents.append(intent)
    except Exception:
        if IntentType.UNCLEAR_INTENT not in intents:
            intents.append(IntentType.UNCLEAR_INTENT)

    if not intents:
        return [IntentType.UNCLEAR_INTENT]
    return intents

