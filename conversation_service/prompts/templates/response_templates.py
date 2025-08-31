"""
Templates de réponse Phase 5
Templates contextualisés pour la génération de réponses par intention et situation
"""
from typing import Dict, Any, Optional, List
from datetime import datetime


class ResponseTemplates:
    """Gestionnaire de templates de réponse contextualisés"""
    
    def __init__(self):
        self.base_instructions = """Tu es un assistant financier personnel intelligent et bienveillant.
Ton rôle est d'aider l'utilisateur à comprendre et gérer ses finances personnelles.

Principes de réponse:
1. Sois précis et factuel avec les chiffres
2. Utilise un ton professionnel mais chaleureux
3. Structure tes réponses de manière claire
4. Mets en avant les informations importantes avec **markdown**
5. Propose des insights utiles et actionnables
6. Reste concis tout en étant informatif

Format de réponse attendu:
- Message principal clair et direct
- Présentation des chiffres clés
- Insights pertinents si disponibles
- Évite les répétitions inutiles
"""
    
    def get_template(
        self, 
        intent_type: str, 
        user_message: str,
        entities: Dict[str, Any],
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Récupère le template approprié selon l'intention et le contexte"""
        
        template_method = getattr(self, f"_get_{intent_type.lower()}_template", None)
        if template_method:
            return template_method(user_message, entities, analysis_data, user_context)
        else:
            return self._get_generic_template(user_message, intent_type, entities, analysis_data, user_context)
    
    def _get_search_by_merchant_template(
        self, 
        user_message: str, 
        entities: Dict[str, Any], 
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Template pour les recherches par marchand"""
        
        merchant_name = analysis_data.get("primary_entity", "ce marchand")
        has_results = analysis_data.get("has_results", False)
        is_returning_user = user_context.get("is_returning_user", False) if user_context else False
        
        context_addon = ""
        if is_returning_user:
            frequent_merchants = user_context.get("frequent_merchants", [])
            if merchant_name in frequent_merchants:
                context_addon = f"\nNote: L'utilisateur consulte régulièrement {merchant_name}."
        
        if not has_results:
            return f"""{self.base_instructions}

L'utilisateur demande: "{user_message}"

Situation: Aucune transaction trouvée pour {merchant_name}
Période recherchée: {self._format_period(entities)}

{context_addon}

Génère une réponse empathique qui:
1. Confirme qu'aucune transaction n'a été trouvée
2. Suggère des raisons possibles (période, filtres)
3. Propose des alternatives de recherche

Exemple de réponse:
"Je n'ai trouvé aucune transaction chez {merchant_name} pour la période demandée. Cela peut signifier que vous n'avez pas effectué d'achat récemment ou que les transactions sont enregistrées sous un autre nom de marchand."

Réponse:"""

        total_amount = analysis_data.get("total_amount", 0)
        transaction_count = analysis_data.get("transaction_count", 0)
        avg_amount = analysis_data.get("average_amount", 0)
        
        return f"""{self.base_instructions}

L'utilisateur demande: "{user_message}"

Données analysées:
- Marchand: {merchant_name}
- Montant total: {total_amount:.2f}€
- Nombre de transactions: {transaction_count}
- Montant moyen: {avg_amount:.2f}€
- Période: {self._format_period(entities)}

{context_addon}

Génère une réponse claire qui:
1. Répond directement à la question avec les chiffres clés
2. Met en valeur les montants importants avec **markdown**
3. Ajoute du contexte sur les habitudes de dépense
4. Reste naturelle et engageante

Exemple de format:
"Ce mois-ci, vous avez dépensé **234,56€** chez Amazon sur **12 transactions**. Votre montant moyen par achat est de 19,55€."

Réponse:"""
    
    def _get_spending_analysis_template(
        self, 
        user_message: str, 
        entities: Dict[str, Any], 
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Template pour l'analyse des dépenses globales"""
        
        has_results = analysis_data.get("has_results", False)
        
        if not has_results:
            return f"""{self.base_instructions}

L'utilisateur demande une analyse de dépenses: "{user_message}"

Situation: Aucune donnée disponible pour l'analyse
Période: {self._format_period(entities)}

Génère une réponse qui explique l'absence de données et propose des alternatives.

Réponse:"""
        
        total_amount = analysis_data.get("total_amount", 0)
        transaction_count = analysis_data.get("transaction_count", 0)
        unique_merchants = analysis_data.get("unique_merchants", 0)
        
        breakdown_info = ""
        if "merchants_breakdown" in analysis_data:
            merchants_data = analysis_data["merchants_breakdown"][:3]  # Top 3
            breakdown_info = "Top marchands:\n" + "\n".join([
                f"- {m['name']}: {m['amount']:.2f}€ ({m['count']} transactions)"
                for m in merchants_data
            ])
        
        personalization = ""
        if user_context and user_context.get("is_returning_user"):
            detail_level = user_context.get("detail_level", "medium")
            if detail_level == "advanced":
                personalization = "Fournis une analyse détaillée avec des métriques avancées."
            elif detail_level == "basic":
                personalization = "Reste simple et évite les termes techniques complexes."
        
        return f"""{self.base_instructions}

L'utilisateur demande: "{user_message}"

Données d'analyse:
- Montant total: {total_amount:.2f}€
- Nombre de transactions: {transaction_count}
- Marchands différents: {unique_merchants}
- Période: {self._format_period(entities)}

{breakdown_info}

{personalization}

Génère une analyse synthétique qui:
1. Présente le montant total avec impact visuel
2. Met en perspective le nombre de transactions
3. Évoque la diversification si pertinente
4. Structure l'information de manière digestible

Réponse:"""
    
    def _get_balance_inquiry_template(
        self, 
        user_message: str, 
        entities: Dict[str, Any], 
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Template pour les demandes de solde"""
        
        return f"""{self.base_instructions}

L'utilisateur demande son solde: "{user_message}"

Contexte: Demande d'information sur le solde de compte

Génère une réponse professionnelle qui:
1. Indique que les informations de solde nécessitent une connexion sécurisée
2. Explique les étapes pour accéder au solde
3. Propose des alternatives d'information disponibles
4. Reste utile malgré la limitation

Ton: Professionnel et rassurant

Réponse:"""
    
    def _get_category_analysis_template(
        self, 
        user_message: str, 
        entities: Dict[str, Any], 
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Template pour l'analyse par catégorie"""
        
        category = analysis_data.get("primary_entity", "cette catégorie")
        has_results = analysis_data.get("has_results", False)
        
        if not has_results:
            return f"""{self.base_instructions}

L'utilisateur analyse la catégorie: "{user_message}"

Situation: Aucune donnée trouvée pour la catégorie {category}
Période: {self._format_period(entities)}

Génère une réponse qui explique l'absence de données pour cette catégorie et suggère des alternatives.

Réponse:"""
        
        total_amount = analysis_data.get("total_amount", 0)
        transaction_count = analysis_data.get("transaction_count", 0)
        
        return f"""{self.base_instructions}

L'utilisateur demande: "{user_message}"

Analyse de catégorie:
- Catégorie: {category}
- Montant total: {total_amount:.2f}€
- Nombre de transactions: {transaction_count}
- Période: {self._format_period(entities)}

Génère une analyse de cette catégorie qui:
1. Met en avant les montants avec formatage visuel
2. Contextualise par rapport aux habitudes moyennes
3. Propose des insights sur cette catégorie
4. Reste informatif et actionnable

Réponse:"""
    
    def _get_transaction_search_template(
        self, 
        user_message: str, 
        entities: Dict[str, Any], 
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Template pour la recherche de transactions"""
        
        has_results = analysis_data.get("has_results", False)
        total_hits = analysis_data.get("total_hits", 0)
        
        search_criteria = self._extract_search_criteria(entities)
        
        return f"""{self.base_instructions}

L'utilisateur recherche des transactions: "{user_message}"

Critères de recherche: {search_criteria}
Résultats trouvés: {total_hits} transactions
Données disponibles: {has_results}

Génère une réponse de recherche qui:
1. Confirme les critères de recherche
2. Présente le nombre de résultats trouvés
3. Résume les informations principales si disponibles
4. Guide vers les prochaines actions possibles

Réponse:"""
    
    def _get_budget_inquiry_template(
        self, 
        user_message: str, 
        entities: Dict[str, Any], 
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Template pour les demandes liées au budget"""
        
        return f"""{self.base_instructions}

L'utilisateur s'intéresse à son budget: "{user_message}"

Génère une réponse sur la gestion budgétaire qui:
1. Encourage la planification financière
2. Propose des conseils budgétaires généraux
3. Suggère des actions concrètes
4. Reste motivante et constructive

Réponse:"""
    
    def _get_generic_template(
        self, 
        user_message: str, 
        intent_type: str,
        entities: Dict[str, Any], 
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Template générique pour les intentions non spécifiées"""
        
        has_results = analysis_data.get("has_results", False)
        
        return f"""{self.base_instructions}

L'utilisateur demande: "{user_message}"

Intention détectée: {intent_type}
Données disponibles: {"Oui" if has_results else "Non"}
Contexte: {entities}

Génère une réponse appropriée qui:
1. Répond du mieux possible à la demande
2. Utilise les données disponibles
3. Reste utile même avec des informations limitées
4. Guide vers des alternatives si nécessaire

Réponse:"""
    
    # Méthodes utilitaires
    
    def _format_period(self, entities: Dict[str, Any]) -> str:
        """Formate la période à partir des entités de date"""
        
        if not entities.get("dates"):
            return "Non spécifiée"
        
        dates_info = entities["dates"]
        if isinstance(dates_info, dict):
            if "original" in dates_info:
                return dates_info["original"]
            elif "normalized" in dates_info:
                normalized = dates_info["normalized"]
                if isinstance(normalized, dict):
                    start = normalized.get("gte", "")
                    end = normalized.get("lte", "")
                    if start and end:
                        return f"{start} au {end}"
                    elif start:
                        return f"Depuis {start}"
                    elif end:
                        return f"Jusqu'au {end}"
        
        return str(dates_info)
    
    def _extract_search_criteria(self, entities: Dict[str, Any]) -> str:
        """Extrait les critères de recherche lisibles"""
        
        criteria = []
        
        if entities.get("merchants"):
            merchants = entities["merchants"]
            if isinstance(merchants, list) and merchants:
                criteria.append(f"Marchand: {merchants[0]}")
        
        if entities.get("categories"):
            categories = entities["categories"] 
            if isinstance(categories, list) and categories:
                criteria.append(f"Catégorie: {categories[0]}")
        
        if entities.get("amounts"):
            amounts = entities["amounts"]
            if isinstance(amounts, dict):
                if amounts.get("min"):
                    criteria.append(f"Montant min: {amounts['min']}€")
                if amounts.get("max"):
                    criteria.append(f"Montant max: {amounts['max']}€")
        
        period = self._format_period(entities)
        if period != "Non spécifiée":
            criteria.append(f"Période: {period}")
        
        return ", ".join(criteria) if criteria else "Critères généraux"


class ContextualResponseTemplates(ResponseTemplates):
    """Version avancée avec personnalisation contextuelle"""
    
    def get_personalized_template(
        self,
        intent_type: str,
        user_message: str,
        entities: Dict[str, Any],
        analysis_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        """Récupère un template personnalisé selon le contexte utilisateur"""
        
        base_template = self.get_template(intent_type, user_message, entities, analysis_data, user_context)
        
        # Ajouts contextuels
        personalization_addons = self._build_personalization_addons(user_context)
        
        if personalization_addons:
            return f"{base_template}\n\nPersonnalisation:\n{personalization_addons}"
        
        return base_template
    
    def _build_personalization_addons(self, user_context: Dict[str, Any]) -> str:
        """Construit les ajouts de personnalisation"""
        
        addons = []
        
        # Style de communication
        communication_style = user_context.get("communication_style", "balanced")
        if communication_style == "concise":
            addons.append("- Reste concis et va à l'essentiel")
        elif communication_style == "detailed":
            addons.append("- Fournis des explications détaillées et approfondies")
        
        # Niveau d'expérience
        detail_level = user_context.get("detail_level", "medium")
        if detail_level == "basic":
            addons.append("- Utilise un langage simple, évite le jargon financier")
        elif detail_level == "advanced":
            addons.append("- Tu peux utiliser des termes financiers techniques")
        
        # Historique des interactions
        if user_context.get("is_returning_user"):
            interaction_count = user_context.get("interaction_count", 0)
            if interaction_count > 5:
                addons.append(f"- L'utilisateur est expérimenté ({interaction_count} interactions)")
        
        # Préférences déduites
        frequent_merchants = user_context.get("frequent_merchants", [])
        if frequent_merchants:
            merchants_str = ", ".join(frequent_merchants[:3])
            addons.append(f"- Marchands fréquents: {merchants_str}")
        
        preferred_intents = user_context.get("preferred_intents", [])
        if preferred_intents:
            intents_str = ", ".join(preferred_intents[:2])
            addons.append(f"- Sujets d'intérêt: {intents_str}")
        
        return "\n".join(addons)


# Instance globale par défaut
default_templates = ResponseTemplates()
contextual_templates = ContextualResponseTemplates()


def get_response_template(
    intent_type: str,
    user_message: str,
    entities: Dict[str, Any],
    analysis_data: Dict[str, Any],
    user_context: Optional[Dict[str, Any]] = None,
    use_personalization: bool = True
) -> str:
    """
    Fonction utilitaire pour récupérer le template approprié
    
    Args:
        intent_type: Type d'intention détectée
        user_message: Message original de l'utilisateur
        entities: Entités extraites
        analysis_data: Données d'analyse des résultats
        user_context: Contexte utilisateur (optionnel)
        use_personalization: Utiliser la personnalisation contextuelle
    
    Returns:
        Template de prompt personnalisé
    """
    
    if use_personalization and user_context:
        return contextual_templates.get_personalized_template(
            intent_type, user_message, entities, analysis_data, user_context
        )
    else:
        return default_templates.get_template(
            intent_type, user_message, entities, analysis_data, user_context
        )


# Configuration des templates par environnement
TEMPLATE_CONFIG = {
    "production": {
        "use_personalization": True,
        "max_template_length": 2000,
        "enable_context_memory": True
    },
    "development": {
        "use_personalization": True,
        "max_template_length": 3000,
        "enable_context_memory": True,
        "debug_mode": True
    },
    "testing": {
        "use_personalization": False,
        "max_template_length": 1500,
        "enable_context_memory": False
    }
}