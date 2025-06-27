# enrichment_service/storage/elasticsearch_client.py
"""
Client Elasticsearch pour enrichment_service.
Gère l'indexation des transactions dans Bonsai Elasticsearch.
"""
import logging
import aiohttp
import json
import ssl
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from config_service.config import settings

logger = logging.getLogger("enrichment_service.elasticsearch")

class ElasticsearchClient:
    """Client HTTP pour indexer dans Bonsai Elasticsearch."""
    
    def __init__(self):
        self.base_url = settings.BONSAI_URL
        self.index_name = "harena_transactions"
        self.session = None
        self._initialized = False
        
    async def initialize(self):
        """Initialise la connexion Elasticsearch."""
        if not self.base_url:
            raise ValueError("BONSAI_URL is required")
        
        # Créer une session HTTP persistante
        ssl_context = ssl.create_default_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        
        # Tester la connexion
        try:
            async with self.session.get(self.base_url) as response:
                if response.status == 200:
                    cluster_info = await response.json()
                    logger.info(f"✅ Connexion Elasticsearch établie: {cluster_info.get('cluster_name', 'unknown')}")
                else:
                    logger.warning(f"⚠️ Elasticsearch répond avec status {response.status}")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Elasticsearch: {e}")
            raise
        
        # Créer l'index s'il n'existe pas
        await self._setup_index()
        self._initialized = True
        logger.info(f"🔍 Client Elasticsearch initialisé pour index '{self.index_name}'")
    
    async def _setup_index(self):
        """Crée l'index s'il n'existe pas."""
        # Vérifier l'existence
        async with self.session.head(f"{self.base_url}/{self.index_name}") as response:
            if response.status == 200:
                logger.info(f"📚 Index '{self.index_name}' existe déjà")
                return
        
        # Créer l'index avec mapping optimisé
        mapping = {
            "mappings": {
                "properties": {
                    # Identifiants
                    "user_id": {"type": "integer"},
                    "transaction_id": {"type": "keyword"},
                    "account_id": {"type": "integer"},
                    
                    # Contenu recherchable
                    "searchable_text": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "primary_description": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "merchant_name": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    
                    # Données financières
                    "amount": {"type": "float"},
                    "amount_abs": {"type": "float"},
                    "transaction_type": {"type": "keyword"},
                    "currency_code": {"type": "keyword"},
                    
                    # Dates
                    "date": {"type": "date"},
                    "transaction_date": {"type": "date"},
                    "month_year": {"type": "keyword"},
                    "weekday": {"type": "keyword"},
                    
                    # Catégorisation
                    "category_id": {"type": "integer"},
                    "category_name": {"type": "keyword"},
                    "operation_type": {"type": "keyword"},
                    
                    # Flags
                    "is_future": {"type": "boolean"},
                    "is_deleted": {"type": "boolean"},
                    
                    # Métadonnées
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "max_result_window": 10000
                }
            }
        }
        
        async with self.session.put(f"{self.base_url}/{self.index_name}", json=mapping) as response:
            if response.status in [200, 201]:
                logger.info(f"✅ Index '{self.index_name}' créé avec succès")
            else:
                error_text = await response.text()
                logger.error(f"❌ Erreur création index: {response.status} - {error_text}")
                raise Exception(f"Failed to create index: {error_text}")
    
    async def index_transaction(self, structured_transaction) -> bool:
        """
        Indexe une transaction structurée dans Elasticsearch.
        
        Args:
            structured_transaction: Instance de StructuredTransaction
            
        Returns:
            bool: True si l'indexation a réussi
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        try:
            # Créer le document Elasticsearch
            doc = {
                # Identifiants
                "user_id": structured_transaction.user_id,
                "transaction_id": structured_transaction.transaction_id,
                "account_id": structured_transaction.account_id,
                
                # Contenu recherchable
                "searchable_text": structured_transaction.searchable_text,
                "primary_description": structured_transaction.primary_description,
                "merchant_name": getattr(structured_transaction, 'merchant_name', ''),
                
                # Données financières
                "amount": structured_transaction.amount,
                "amount_abs": structured_transaction.amount_abs,
                "transaction_type": structured_transaction.transaction_type,
                "currency_code": structured_transaction.currency_code,
                
                # Dates
                "date": structured_transaction.date_str,
                "transaction_date": structured_transaction.date_str,
                "month_year": structured_transaction.month_year,
                "weekday": structured_transaction.weekday,
                
                # Catégorisation
                "category_id": structured_transaction.category_id,
                "operation_type": structured_transaction.operation_type,
                
                # Flags
                "is_future": structured_transaction.is_future,
                "is_deleted": structured_transaction.is_deleted,
                
                # Métadonnées
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # ID du document (unique par utilisateur et transaction)
            doc_id = f"user_{structured_transaction.user_id}_tx_{structured_transaction.transaction_id}"
            
            # Indexer le document
            async with self.session.put(
                f"{self.base_url}/{self.index_name}/_doc/{doc_id}",
                json=doc
            ) as response:
                if response.status in [200, 201]:
                    logger.debug(f"📝 Transaction {doc_id} indexée avec succès")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur indexation {doc_id}: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Exception lors de l'indexation: {e}")
            return False
    
    async def index_transactions_batch(self, structured_transactions: List) -> Dict[str, Any]:
        """
        Indexe un lot de transactions dans Elasticsearch.
        
        Args:
            structured_transactions: Liste de StructuredTransaction
            
        Returns:
            Dict: Résumé de l'indexation
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        if not structured_transactions:
            return {"indexed": 0, "errors": 0, "total": 0}
        
        # Préparer le bulk request
        bulk_body = []
        
        for tx in structured_transactions:
            doc_id = f"user_{tx.user_id}_tx_{tx.transaction_id}"
            
            # Action header
            bulk_body.append(json.dumps({
                "index": {
                    "_index": self.index_name,
                    "_id": doc_id
                }
            }))
            
            # Document data
            doc = {
                "user_id": tx.user_id,
                "transaction_id": tx.transaction_id,
                "account_id": tx.account_id,
                "searchable_text": tx.searchable_text,
                "primary_description": tx.primary_description,
                "merchant_name": getattr(tx, 'merchant_name', ''),
                "amount": tx.amount,
                "amount_abs": tx.amount_abs,
                "transaction_type": tx.transaction_type,
                "currency_code": tx.currency_code,
                "date": tx.date_str,
                "transaction_date": tx.date_str,
                "month_year": tx.month_year,
                "weekday": tx.weekday,
                "category_id": tx.category_id,
                "operation_type": tx.operation_type,
                "is_future": tx.is_future,
                "is_deleted": tx.is_deleted,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            bulk_body.append(json.dumps(doc))
        
        # Joindre avec des nouvelles lignes (format bulk)
        bulk_data = "\n".join(bulk_body) + "\n"
        
        try:
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_bulk",
                data=bulk_data,
                headers={"Content-Type": "application/x-ndjson"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Analyser les résultats
                    indexed_count = 0
                    error_count = 0
                    
                    for item in result.get("items", []):
                        if "index" in item:
                            if item["index"].get("status") in [200, 201]:
                                indexed_count += 1
                            else:
                                error_count += 1
                                logger.error(f"❌ Erreur bulk item: {item['index']}")
                    
                    summary = {
                        "indexed": indexed_count,
                        "errors": error_count,
                        "total": len(structured_transactions)
                    }
                    
                    logger.info(f"📦 Bulk indexation: {summary}")
                    return summary
                    
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur bulk request: {response.status} - {error_text}")
                    return {"indexed": 0, "errors": len(structured_transactions), "total": len(structured_transactions)}
                    
        except Exception as e:
            logger.error(f"❌ Exception bulk indexation: {e}")
            return {"indexed": 0, "errors": len(structured_transactions), "total": len(structured_transactions)}
    
    async def delete_user_transactions(self, user_id: int) -> bool:
        """
        Supprime toutes les transactions d'un utilisateur de l'index.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si la suppression a réussi
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        try:
            # Requête de suppression par user_id
            delete_query = {
                "query": {
                    "term": {
                        "user_id": user_id
                    }
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_delete_by_query",
                json=delete_query
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    deleted_count = result.get("deleted", 0)
                    logger.info(f"🗑️ {deleted_count} transactions supprimées pour user {user_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur suppression user {user_id}: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Exception suppression user {user_id}: {e}")
            return False
    
    async def get_user_transaction_count(self, user_id: int) -> int:
        """
        Compte les transactions d'un utilisateur dans l'index.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            int: Nombre de transactions
        """
        if not self._initialized:
            return 0
        
        try:
            count_query = {
                "query": {
                    "term": {
                        "user_id": user_id
                    }
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_count",
                json=count_query
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("count", 0)
                else:
                    logger.error(f"❌ Erreur comptage user {user_id}: {response.status}")
                    return 0
                    
        except Exception as e:
            logger.error(f"❌ Exception comptage user {user_id}: {e}")
            return 0
    
    async def close(self):
        """Ferme la session HTTP."""
        if self.session:
            await self.session.close()
            logger.info("🔌 Session Elasticsearch fermée")