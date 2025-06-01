"""
Formateur de requêtes pour le search_service.

Ce module convertit les intentions détectées en requêtes structurées
pour le service de recherche hybride.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from conversation_service.models import DetectedIntent, IntentType

logger = logging.getLogger(__name__)


class QueryFormatter:
    """Formateur de requêtes de recherche."""
    
    def __init__(self):
        # Mapping des catégories courantes
        self.category_mapping = {
            'restaurant': ['restaurant', 'resto', 'restauration'],
            'courses': ['supermarché', 'course', 'alimentaire', 'grocery'],
            'carburant': ['carburant', 'essence', 'station', 'fuel'],
            'transport': ['transport', 'metro', 'bus', 'train', 'taxi'],
            'santé': ['pharmacie', 'médecin', 'dentiste', 'health'],
            'shopping': ['shopping', 'vêtement', 'magasin', 'achat'],
            'banque': ['banque', 'virement', 'frais', 'commission'],
            'loisirs': ['loisir', 'cinéma', 'sport', 'entertainment']
        }
        
        # Mapping des périodes temporelles
        self.time_mapping = {
            'today': {'days': 0},
            'yesterday': {'days': 1},
            'this_week': {'days': 7},
            'last_week': {'days': 14, 'offset': 7},
            'this_month': {'days': 30},
            'last_month': {'days': 60, 'offset': 30},
            'this_year': {'days': 365},
            'last_year': {'days': 730, 'offset': 365}
        }
    
    def format_search_query(
        self,
        intent: DetectedIntent,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Formate une requête de recherche basée sur l'intention.
        
        Args:
            intent: Intention détectée
            user_id: ID de l'utilisateur
            context: Contexte additionnel
            
        Returns:
            Dict: Requête formatée pour le search_service ou None
        """
        if intent.intent_type == IntentType.SEARCH_TRANSACTIONS:
            return self._format_transaction_search(intent, user_id, context)
        elif intent.intent_type == IntentType.SPENDING_ANALYSIS:
            return self._format_spending_analysis(intent, user_id, context)
        elif intent.intent_type == IntentType.CATEGORY_ANALYSIS:
            return self._format_category_analysis(intent, user_id, context)
        elif intent.intent_type == IntentType.MERCHANT_ANALYSIS:
            return self._format_merchant_analysis(intent, user_id, context)
        elif intent.intent_type == IntentType.TIME_ANALYSIS:
            return self._format_time_analysis(intent, user_id, context)
        elif intent.intent_type == IntentType.ACCOUNT_SUMMARY:
            return self._format_account_summary(intent, user_id, context)
        elif intent.intent_type == IntentType.COMPARISON:
            return self._format_comparison(intent, user_id, context)
        else:
            logger.warning(f"Type d'intention non supporté pour la recherche: {intent.intent_type}")
            return None
    
    def _format_transaction_search(
        self,
        intent: DetectedIntent,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formate une recherche de transactions."""
        parameters = intent.parameters
        
        # Construire la requête de base
        query_text = ""
        
        # Ajouter le marchand si spécifié
        if 'merchant' in parameters:
            query_text += parameters['merchant']
        
        # Ajouter la catégorie si spécifiée
        if 'category' in parameters:
            category_terms = self.category_mapping.get(
                parameters['category'], 
                [parameters['category']]
            )
            query_text += " " + " ".join(category_terms)
        
        # Ajouter le montant si spécifié
        if 'amount' in parameters:
            query_text += f" {parameters['amount']}€"
        
        # Par défaut, rechercher toutes les transactions récentes
        if not query_text.strip():
            query_text = "transaction"
        
        # Construire la requête complète
        search_query = {
            "user_id": user_id,
            "query": query_text.strip(),
            "search_type": "hybrid",
            "limit": parameters.get('limit', 20),
            "lexical_weight": 0.6,
            "semantic_weight": 0.4,
            "use_reranking": True,
            "include_highlights": True
        }
        
        # Ajouter les filtres temporels
        date_filters = self._extract_date_filters(parameters)
        search_query.update(date_filters)
        
        # Ajouter les filtres de montant
        amount_filters = self._extract_amount_filters(parameters)
        search_query.update(amount_filters)
        
        return search_query
    
    def _format_spending_analysis(
        self,
        intent: DetectedIntent,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formate une analyse des dépenses."""
        parameters = intent.parameters
        
        # Pour une analyse des dépenses, on veut toutes les transactions
        # dans la période spécifiée
        query_text = "dépense transaction"
        
        # Ajouter la catégorie si spécifiée
        if 'category' in parameters:
            category_terms = self.category_mapping.get(
                parameters['category'], 
                [parameters['category']]
            )
            query_text += " " + " ".join(category_terms)
        
        search_query = {
            "user_id": user_id,
            "query": query_text,
            "search_type": "hybrid",
            "limit": 100,  # Plus de résultats pour l'analyse
            "lexical_weight": 0.4,
            "semantic_weight": 0.6,  # Plus de poids sémantique pour l'analyse
            "use_reranking": False,  # Pas besoin de reranking pour l'analyse
            "include_highlights": False
        }
        
        # Filtres temporels (important pour l'analyse)
        date_filters = self._extract_date_filters(parameters)
        if not date_filters:
            # Par défaut, ce mois-ci
            now = datetime.now()
            search_query.update({
                "date_from": now.replace(day=1).isoformat(),
                "date_to": now.isoformat()
            })
        else:
            search_query.update(date_filters)
        
        # Filtrer seulement les débits (dépenses)
        search_query["transaction_types"] = ["debit"]
        
        return search_query
    
    def _format_category_analysis(
        self,
        intent: DetectedIntent,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formate une analyse par catégorie."""
        parameters = intent.parameters
        
        query_text = "catégorie"
        
        # Catégorie spécifique
        if 'category' in parameters:
            category_terms = self.category_mapping.get(
                parameters['category'], 
                [parameters['category']]
            )
            query_text = " ".join(category_terms)
        
        search_query = {
            "user_id": user_id,
            "query": query_text,
            "search_type": "semantic",  # Recherche sémantique pour les catégories
            "limit": 50,
            "lexical_weight": 0.3,
            "semantic_weight": 0.7,
            "use_reranking": True,
            "include_highlights": True
        }
        
        # Filtres temporels
        date_filters = self._extract_date_filters(parameters)
        search_query.update(date_filters)
        
        return search_query
    
    def _format_merchant_analysis(
        self,
        intent: DetectedIntent,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formate une analyse par marchand."""
        parameters = intent.parameters
        
        query_text = parameters.get('merchant', 'marchand')
        
        search_query = {
            "user_id": user_id,
            "query": query_text,
            "search_type": "lexical",  # Recherche lexicale pour les noms de marchands
            "limit": 30,
            "lexical_weight": 0.8,
            "semantic_weight": 0.2,
            "use_reranking": True,
            "include_highlights": True
        }
        
        # Filtres temporels
        date_filters = self._extract_date_filters(parameters)
        search_query.update(date_filters)
        
        return search_query
    
    def _format_time_analysis(
        self,
        intent: DetectedIntent,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formate une analyse temporelle."""
        parameters = intent.parameters
        
        query_text = "transaction"
        
        search_query = {
            "user_id": user_id,
            "query": query_text,
            "search_type": "hybrid",
            "limit": 100,  # Beaucoup de résultats pour l'analyse temporelle
            "lexical_weight": 0.3,
            "semantic_weight": 0.7,
            "use_reranking": False,
            "include_highlights": False
        }
        
        # Les filtres temporels sont essentiels pour ce type d'analyse
        date_filters = self._extract_date_filters(parameters)
        if date_filters:
            search_query.update(date_filters)
        else:
            # Par défaut, les 3 derniers mois
            now = datetime.now()
            three_months_ago = now - timedelta(days=90)
            search_query.update({
                "date_from": three_months_ago.isoformat(),
                "date_to": now.isoformat()
            })
        
        return search_query
    
    def _format_account_summary(
        self,
        intent: DetectedIntent,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formate une demande de résumé de compte."""
        query_text = "transaction récente"
        
        search_query = {
            "user_id": user_id,
            "query": query_text,
            "search_type": "hybrid",
            "limit": 50,
            "lexical_weight": 0.5,
            "semantic_weight": 0.5,
            "use_reranking": False,
            "include_highlights": False
        }
        
        # Derniers 7 jours par défaut
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        search_query.update({
            "date_from": week_ago.isoformat(),
            "date_to": now.isoformat()
        })
        
        return search_query
    
    def _format_comparison(
        self,
        intent: DetectedIntent,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formate une demande de comparaison."""
        parameters = intent.parameters
        
        query_text = "transaction"
        
        # Ajouter les éléments à comparer
        if 'compare_elements' in parameters:
            query_text += " " + " ".join(parameters['compare_elements'])
        
        search_query = {
            "user_id": user_id,
            "query": query_text,
            "search_type": "hybrid",
            "limit": 100,  # Plus de résultats pour faire des comparaisons
            "lexical_weight": 0.4,
            "semantic_weight": 0.6,
            "use_reranking": True,
            "include_highlights": True
        }
        
        # Période étendue pour les comparaisons
        date_filters = self._extract_date_filters(parameters)
        if not date_filters:
            # Par défaut, 6 derniers mois
            now = datetime.now()
            six_months_ago = now - timedelta(days=180)
            search_query.update({
                "date_from": six_months_ago.isoformat(),
                "date_to": now.isoformat()
            })
        else:
            search_query.update(date_filters)
        
        return search_query
    
    def _extract_date_filters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les filtres de date des paramètres."""
        filters = {}
        
        if 'time_period' in parameters:
            period = parameters['time_period']
            
            if period in self.time_mapping:
                mapping = self.time_mapping[period]
                now = datetime.now()
                
                if 'offset' in mapping:
                    # Période avec décalage (ex: mois dernier)
                    end_date = now - timedelta(days=mapping['offset'])
                    start_date = end_date - timedelta(days=mapping['days'] - mapping['offset'])
                else:
                    # Période depuis maintenant
                    end_date = now
                    start_date = now - timedelta(days=mapping['days'])
                
                filters.update({
                    "date_from": start_date.isoformat(),
                    "date_to": end_date.isoformat()
                })
        
        # Dates spécifiques
        if 'date_from' in parameters:
            filters['date_from'] = parameters['date_from']
        
        if 'date_to' in parameters:
            filters['date_to'] = parameters['date_to']
        
        return filters
    
    def _extract_amount_filters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les filtres de montant des paramètres."""
        filters = {}
        
        if 'amount_min' in parameters:
            filters['amount_min'] = parameters['amount_min']
        
        if 'amount_max' in parameters:
            filters['amount_max'] = parameters['amount_max']
        
        if 'amount' in parameters:
            # Montant exact -> recherche dans une fourchette
            amount = parameters['amount']
            tolerance = amount * 0.1  # 10% de tolérance
            filters.update({
                'amount_min': amount - tolerance,
                'amount_max': amount + tolerance
            })
        
        return filters
    
    def get_supported_intents(self) -> List[str]:
        """Retourne la liste des intentions supportées."""
        return [
            IntentType.SEARCH_TRANSACTIONS,
            IntentType.SPENDING_ANALYSIS,
            IntentType.CATEGORY_ANALYSIS,
            IntentType.MERCHANT_ANALYSIS,
            IntentType.TIME_ANALYSIS,
            IntentType.ACCOUNT_SUMMARY,
            IntentType.COMPARISON
        ]


# Instance globale
query_formatter = QueryFormatter()