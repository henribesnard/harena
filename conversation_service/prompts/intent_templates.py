"""Intent-Specific Prompt Templates.

This module définit des templates de prompts spécialisés par intention
ou domaine fonctionnel. Chaque template inclut un prompt système dédié,
un template utilisateur et plusieurs exemples few-shot détaillés afin de
guider efficacement le modèle.

L'API expose des fonctions utilitaires permettant de sélectionner
dynamiquement le template approprié en fonction du contexte et de
formater un prompt complet prêt à être envoyé au modèle.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class IntentPromptTemplate:
    """Structure d'un template de prompt spécialisé."""

    system_prompt: str
    user_template: str
    examples: List[str]


# ---------------------------------------------------------------------------
# Templates par intention
# ---------------------------------------------------------------------------

INTENT_PROMPT_TEMPLATES: Dict[str, IntentPromptTemplate] = {
    "transaction_query": IntentPromptTemplate(
        system_prompt="Vous êtes un assistant capable de retrouver des transactions précises pour l'utilisateur.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Fournissez les transactions correspondantes.",
        examples=[
            "MESSAGE: \"Achats chez Carrefour le mois dernier\" => Retourner la liste filtrée par marchand et période.",
            "MESSAGE: \"Paiement de loyer d'avril\" => Retourner la transaction unique du loyer d'avril.",
        ],
    ),
    "spending_analysis": IntentPromptTemplate(
        system_prompt="Vous analysez les habitudes de dépense de l'utilisateur et détectez des tendances.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Analysez les dépenses et fournissez un résumé clair.",
        examples=[
            "MESSAGE: \"Je dépense trop en restaurants\" => Identifier la part des restaurants dans les dépenses.",
            "MESSAGE: \"Où part mon salaire chaque mois ?\" => Fournir une répartition mensuelle des dépenses.",
        ],
    ),
    "budget_inquiry": IntentPromptTemplate(
        system_prompt="Vous aidez l'utilisateur à suivre et comprendre ses budgets.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Répondez avec l'état actuel du budget concerné.",
        examples=[
            "MESSAGE: \"Il me reste combien pour les courses ce mois-ci ?\" => Retourner le budget restant de la catégorie courses.",
            "MESSAGE: \"Quel est mon budget transport ?\" => Fournir le budget alloué au transport.",
        ],
    ),
    "category_analysis": IntentPromptTemplate(
        system_prompt="Vous fournissez des analyses sur une catégorie de dépense spécifique.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Détaillez l'activité pour la catégorie mentionnée.",
        examples=[
            "MESSAGE: \"Parle-moi de mes dépenses en loisirs\" => Résumer les montants et la tendance pour les loisirs.",
            "MESSAGE: \"Combien pour l'alimentation cette semaine ?\" => Totaliser les dépenses d'alimentation de la semaine.",
        ],
    ),
    "merchant_inquiry": IntentPromptTemplate(
        system_prompt="Vous répondez aux questions concernant des marchands spécifiques.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Détaillez les interactions financières avec ce marchand.",
        examples=[
            "MESSAGE: \"Combien ai-je dépensé chez Amazon ?\" => Total des dépenses liées à Amazon.",
            "MESSAGE: \"Des transactions chez Carrefour la semaine dernière ?\" => Liste des transactions Carrefour de la semaine précédente.",
        ],
    ),
    "balance_inquiry": IntentPromptTemplate(
        system_prompt="Vous indiquez les soldes des comptes et cartes de l'utilisateur.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Fournissez les soldes demandés avec précision.",
        examples=[
            "MESSAGE: \"Quel est le solde de mon compte courant ?\" => Afficher le solde actuel du compte courant.",
            "MESSAGE: \"Combien sur ma carte épargne ?\" => Indiquer le solde du compte épargne.",
        ],
    ),
    "trend_analysis": IntentPromptTemplate(
        system_prompt="Vous détectez et expliquez les tendances financières sur différentes périodes.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Analysez la tendance et fournissez une interprétation.",
        examples=[
            "MESSAGE: \"Mes dépenses augmentent-elles ?\" => Comparer les dépenses récentes aux périodes précédentes.",
            "MESSAGE: \"Dépensé plus que 500€ par mois ?\" => Évaluer si la moyenne mensuelle dépasse 500€.",
        ],
    ),
    "comparison_query": IntentPromptTemplate(
        system_prompt="Vous réalisez des comparaisons entre périodes, catégories ou montants.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Effectuez la comparaison et résumez le résultat.",
        examples=[
            "MESSAGE: \"Dépenses de février vs mars\" => Comparer les totaux entre ces mois.",
            "MESSAGE: \"Plus chez Uber ou Lyft ?\" => Comparer les montants dépensés chez les deux marchands.",
        ],
    ),
    "goal_tracking": IntentPromptTemplate(
        system_prompt="Vous suivez les objectifs financiers de l'utilisateur.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Indiquez la progression et les prochaines étapes pour l'objectif.",
        examples=[
            "MESSAGE: \"Où en est mon objectif épargne vacances ?\" => Progrès vers l'objectif vacances.",
            "MESSAGE: \"J'ai atteint mon objectif de réserve ?\" => Confirmer si le seuil prévu est atteint.",
        ],
    ),
    "alert_management": IntentPromptTemplate(
        system_prompt="Vous gérez les alertes et notifications financières de l'utilisateur.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Traitez ou résumez les alertes demandées.",
        examples=[
            "MESSAGE: \"Active une alerte si je dépense plus de 100€\" => Configurer une alerte sur seuil de dépense.",
            "MESSAGE: \"Quelles alertes sont actives ?\" => Lister les alertes financières actuelles.",
        ],
    ),
    "conversational": IntentPromptTemplate(
        system_prompt="Vous engagez un dialogue simple sans objectif financier spécifique.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Répondez de manière conversationnelle et courtoise.",
        examples=[
            "MESSAGE: \"Salut, comment ça va ?\" => Réponse polie sans contenu financier.",
            "MESSAGE: \"Merci pour ton aide !\" => Réponse empathique de conclusion.",
        ],
    ),
    "other": IntentPromptTemplate(
        system_prompt="Vous gérez les demandes ne rentrant dans aucune catégorie définie.",
        user_template="MESSAGE: \"{message}\"{context}\nTÂCHE: Fournissez une réponse utile ou demandez des précisions.",
        examples=[
            "MESSAGE: \"Peux-tu m'expliquer ceci ?\" => Demander plus de détails pour clarifier.",
            "MESSAGE: \"Je veux quelque chose mais je ne sais pas quoi\" => Proposer des catégories possibles.",
        ],
    ),
}


# ---------------------------------------------------------------------------
# API de sélection et de formatage
# ---------------------------------------------------------------------------

def get_intent_prompt_template(intent: str) -> IntentPromptTemplate:
    """Récupère le template correspondant à une intention.

    Args:
        intent: Intention recherchée.

    Returns:
        Template associé ou template "other" si inconnu.
    """

    return INTENT_PROMPT_TEMPLATES.get(intent, INTENT_PROMPT_TEMPLATES["other"])


def build_intent_prompt(intent: str, message: str, context: str = "") -> str:
    """Construit un prompt complet en fonction de l'intention et du message.

    Args:
        intent: Intention détectée ou ciblée.
        message: Message utilisateur.
        context: Contexte additionnel optionnel.

    Returns:
        Prompt formaté comprenant prompt système, exemples et instruction utilisateur.
    """

    template = get_intent_prompt_template(intent)
    context_section = f"\nCONTEXTE: {context}" if context else ""
    examples_block = "\n\n".join(template.examples)

    return (
        f"{template.system_prompt}\n\nEXEMPLES:\n{examples_block}\n\n"
        f"{template.user_template.format(message=message, context=context_section)}"
    )


__all__ = [
    "IntentPromptTemplate",
    "INTENT_PROMPT_TEMPLATES",
    "get_intent_prompt_template",
    "build_intent_prompt",
]

