"""
Construction des requêtes vers les services financiers.

Ce module construit des requêtes structurées vers les services
financiers en fonction des intentions et entités détectées.
"""

import json
import httpx
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta

from ..config.settings import settings
from ..config.logging import get_logger
from ..models.intent import IntentType, IntentClassification

logger = get_logger(__name__)


class QueryBuilder:
    """
    Constructeur de requêtes vers les services financiers.
    
    Cette classe traduit les intentions et entités en requêtes
    structurées vers les différents services de données.
    """
    
    def __init__(self):
        """Initialise le constructeur de requêtes."""
        self.transaction_service_url = settings.TRANSACTION_VECTOR_SERVICE_URL
        self.sync_service_url = settings.SYNC_SERVICE_URL
        self.user_service_url = settings.USER_SERVICE_URL
        
        # Client HTTP pour les requêtes aux services
        self.client = httpx.AsyncClient(timeout=10.0)
        
        logger.info("Constructeur de requêtes initialisé")
    
    async def build_query(
        self,
        intent: IntentClassification,
        raw_query: str
    ) -> Dict[str, Any]:
        """
        Construit une requête structurée à partir de l'intention et des entités.
        
        Args:
            intent: Classification d'intention
            raw_query: Requête brute de l'utilisateur
            
        Returns:
            Requête structurée
        """
        logger.info(f"Construction d'une requête pour l'intention: {intent.intent}")
        
        # Structure de base de la requête
        query = {
            "intent": intent.intent,
            "entities": intent.entities,
            "raw_query": raw_query
        }
        
        # Ajouter les paramètres spécifiques selon l'intention
        if intent.intent == IntentType.SEARCH_TRANSACTION:
            query["search_params"] = self._build_transaction_search_params(intent.entities, raw_query)
        
        elif intent.intent == IntentType.ANALYZE_SPENDING:
            query["search_params"] = self._build_spending_analysis_params(intent.entities, raw_query)
        
        elif intent.intent == IntentType.CHECK_BALANCE:
            query["account_params"] = self._build_account_params(intent.entities, raw_query)
        
        elif intent.intent == IntentType.ACCOUNT_INFO:
            query["account_params"] = self._build_account_params(intent.entities, raw_query)
        
        return query
    
    async def execute_transaction_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute une requête vers le service de transactions.
        
        Args:
            query_data: Données de la requête
            
        Returns:
            Résultats de la requête
        """
        try:
            # Pour test et développement, utiliser des données simulées
            # Dans un environnement de production, appeler réellement le service
            if settings.DEBUG:
                return self._get_mock_transaction_data(query_data)
            
            # Construction de la requête
            search_params = query_data.get("search_params", {})
            
            # Récupérer les paramètres de requête
            user_id = 1  # À remplacer par l'utilisateur réel
            endpoint = f"{self.transaction_service_url}/api/v1/transactions"
            
            # Exécuter la requête
            response = await self.client.get(
                endpoint,
                params=search_params,
                headers={
                    "Authorization": f"Bearer YOUR_TOKEN_HERE"  # À remplacer par un vrai token
                }
            )
            
            # Vérifier la réponse
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Erreur lors de la requête au service de transactions: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la requête de transactions: {str(e)}")
            return {}
    
    async def execute_account_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute une requête vers le service de comptes.
        
        Args:
            query_data: Données de la requête
            
        Returns:
            Résultats de la requête
        """
        try:
            # Pour test et développement, utiliser des données simulées
            if settings.DEBUG:
                return self._get_mock_account_data(query_data)
            
            # Construction de la requête
            account_params = query_data.get("account_params", {})
            
            # Récupérer les paramètres de requête
            user_id = 1  # À remplacer par l'utilisateur réel
            endpoint = f"{self.user_service_url}/api/v1/accounts"
            
            # Exécuter la requête
            response = await self.client.get(
                endpoint,
                params=account_params,
                headers={
                    "Authorization": f"Bearer YOUR_TOKEN_HERE"  # À remplacer par un vrai token
                }
            )
            
            # Vérifier la réponse
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Erreur lors de la requête au service de comptes: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la requête de comptes: {str(e)}")
            return {}
    
    def _build_transaction_search_params(
        self,
        entities: Dict[str, Any],
        raw_query: str
    ) -> Dict[str, Any]:
        """
        Construit les paramètres de recherche de transactions.
        
        Args:
            entities: Entités extraites
            raw_query: Requête brute
            
        Returns:
            Paramètres de recherche
        """
        params = {}
        
        # Ajouter les dates
        if "date_start" in entities:
            params["start_date"] = entities["date_start"]
        
        if "date_end" in entities:
            params["end_date"] = entities["date_end"]
        
        # Si aucune date n'est spécifiée, utiliser le mois en cours par défaut
        if "date_start" not in params and "date_end" not in params:
            today = date.today()
            first_day_of_month = date(today.year, today.month, 1)
            params["start_date"] = first_day_of_month.isoformat()
            params["end_date"] = today.isoformat()
        
        # Ajouter les montants
        if "min_amount" in entities:
            params["min_amount"] = entities["min_amount"]
        
        if "max_amount" in entities:
            params["max_amount"] = entities["max_amount"]
        
        # Ajouter le commerçant
        if "merchant" in entities:
            params["merchant_names"] = [entities["merchant"]]
        
        # Ajouter la catégorie
        if "category" in entities:
            # Dans un système réel, il faudrait convertir le nom en ID
            if entities["category"].isdigit():
                params["categories"] = [int(entities["category"])]
            else:
                # Simulation - dans un système réel, rechercher l'ID
                params["category_query"] = entities["category"]
        
        # Ajouter le texte de recherche
        search_terms = []
        for term in raw_query.split():
            if len(term) > 3 and term.lower() not in ["quel", "quelle", "quels", "quelles", "dans", "pour", "avec", "chez", "depuis"]:
                search_terms.append(term)
        
        if search_terms:
            params["query"] = " ".join(search_terms)
        
        # Paramètres de pagination et tri
        params["limit"] = 50
        params["sort_by"] = "transaction_date"
        params["sort_order"] = "desc"
        
        return params
    
    def _build_spending_analysis_params(
        self,
        entities: Dict[str, Any],
        raw_query: str
    ) -> Dict[str, Any]:
        """
        Construit les paramètres d'analyse de dépenses.
        
        Args:
            entities: Entités extraites
            raw_query: Requête brute
            
        Returns:
            Paramètres d'analyse
        """
        # Commencer avec les paramètres de recherche de base
        params = self._build_transaction_search_params(entities, raw_query)
        
        # Ajouter des paramètres spécifiques à l'analyse
        params["include_stats"] = True
        params["group_by"] = "category"
        
        # Déterminer la période d'analyse
        if "period" in entities:
            params["period"] = entities["period"]
        elif not ("date_start" in params and "date_end" in params):
            # Si aucune période spécifique, utiliser le mois en cours par défaut
            today = date.today()
            first_day_of_month = date(today.year, today.month, 1)
            params["start_date"] = first_day_of_month.isoformat()
            params["end_date"] = today.isoformat()
            params["period"] = "this_month"
        
        return params
    
    def _build_account_params(
        self,
        entities: Dict[str, Any],
        raw_query: str
    ) -> Dict[str, Any]:
        """
        Construit les paramètres de requête de compte.
        
        Args:
            entities: Entités extraites
            raw_query: Requête brute
            
        Returns:
            Paramètres de compte
        """
        params = {}
        
        # Identifier le compte spécifique si mentionné
        if "account_id" in entities:
            params["account_id"] = entities["account_id"]
        
        # Détecter le type de compte
        account_types = []
        if "checking" in raw_query.lower() or "courant" in raw_query.lower():
            account_types.append("checking")
        if "savings" in raw_query.lower() or "épargne" in raw_query.lower():
            account_types.append("savings")
        if "credit" in raw_query.lower() or "crédit" in raw_query.lower():
            account_types.append("credit_card")
        
        if account_types:
            params["account_types"] = account_types
        
        # Inclure les soldes
        params["include_balance"] = True
        
        return params
    
    def _get_mock_transaction_data(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère des données de transaction simulées pour le développement.
        
        Args:
            query_data: Données de la requête
            
        Returns:
            Données simulées
        """
        # Récupérer les paramètres
        search_params = query_data.get("search_params", {})
        merchant_names = search_params.get("merchant_names", [])
        
        # Créer des données simulées
        mock_data = {
            "count": 5,
            "total_amount": 235.45,
            "start_date": search_params.get("start_date", date.today().replace(day=1).isoformat()),
            "end_date": search_params.get("end_date", date.today().isoformat()),
            "transactions": []
        }
        
        # Générer quelques transactions simulées
        merchant = merchant_names[0] if merchant_names else "Carrefour"
        
        mock_data["transactions"] = [
            {
                "id": "tx1",
                "date": "2025-04-10",
                "description": f"Achat chez {merchant}",
                "amount": -42.50,
                "category": "Alimentation"
            },
            {
                "id": "tx2",
                "date": "2025-04-05",
                "description": f"Achat chez {merchant}",
                "amount": -67.80,
                "category": "Alimentation"
            },
            {
                "id": "tx3",
                "date": "2025-03-28",
                "description": f"Achat chez {merchant}",
                "amount": -38.15,
                "category": "Alimentation"
            },
            {
                "id": "tx4",
                "date": "2025-03-20",
                "description": f"Achat chez {merchant}",
                "amount": -45.90,
                "category": "Alimentation"
            },
            {
                "id": "tx5",
                "date": "2025-03-12",
                "description": f"Achat chez {merchant}",
                "amount": -41.10,
                "category": "Alimentation"
            }
        ]
        
        return mock_data
    
    def _get_mock_account_data(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère des données de compte simulées pour le développement.
        
        Args:
            query_data: Données de la requête
            
        Returns:
            Données simulées
        """
        # Récupérer les paramètres
        account_params = query_data.get("account_params", {})
        account_types = account_params.get("account_types", [])
        
        # Créer des données simulées
        mock_data = {
            "accounts": []
        }
        
        # Si aucun type spécifié, inclure tous les types
        if not account_types:
            account_types = ["checking", "savings", "credit_card"]
        
        # Générer des comptes simulés
        if "checking" in account_types:
            mock_data["accounts"].append({
                "id": 1,
                "name": "Compte Courant",
                "type": "checking",
                "balance": 1250.45,
                "currency": "EUR",
                "last_update": datetime.now().isoformat()
            })
        
        if "savings" in account_types:
            mock_data["accounts"].append({
                "id": 2,
                "name": "Livret A",
                "type": "savings",
                "balance": 5478.92,
                "currency": "EUR",
                "last_update": datetime.now().isoformat()
            })
        
        if "credit_card" in account_types:
            mock_data["accounts"].append({
                "id": 3,
                "name": "Carte Gold",
                "type": "credit_card",
                "balance": -320.18,
                "currency": "EUR",
                "last_update": datetime.now().isoformat()
            })
        
        return mock_data