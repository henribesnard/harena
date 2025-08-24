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
import asyncio
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
        # Batch size used as a starting point for adaptive bulk indexing
        self.default_batch_size = 500
        
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
        # Précharger certaines requêtes pour réchauffer les caches
        await self._warm_index()
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

    async def _warm_index(self):
        """Exécute des requêtes pour réchauffer les caches Elasticsearch."""
        warm_queries = [
            # Statistiques sur le solde pour forcer le chargement du champ
            {"size": 0, "aggs": {"balance_stats": {"stats": {"field": "account_balance"}}}},
            # Requête sur merchant_name.keyword pour réchauffer l'agrégation sur ce champ
            {"size": 0, "query": {"match_all": {}}, "aggs": {"merchants": {"terms": {"field": "merchant_name.keyword", "size": 1}}}}
        ]

        for query in warm_queries:
            try:
                async with self.session.post(
                    f"{self.base_url}/{self.index_name}/_search",
                    json=query
                ) as response:
                    # On lit le corps pour s'assurer que la requête est exécutée
                    await response.text()
            except Exception as e:
                logger.debug(f"Warmup query failed: {e}")
    
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
    
    async def index_transactions_batch(
        self,
        structured_transactions: List,
        initial_batch_size: int = None,
        max_retries: int = 3,
        target_time: float = 2.0,
    ) -> Dict[str, Any]:
        """Indexe un lot de transactions avec adaptation de la taille de batch.

        La méthode ajuste dynamiquement la taille des sous-batches en fonction
        du temps d'indexation précédent et applique un retry avec backoff
        exponentiel en cas d'échec.

        Args:
            structured_transactions: Liste de StructuredTransaction
            initial_batch_size: Taille initiale des lots (par défaut self.default_batch_size)
            max_retries: Nombre maximum de tentatives par sous-batch
            target_time: Temps visé pour une opération d'indexation (en secondes)

        Returns:
            Dict: Résumé de l'indexation avec réponses individuelles
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")

        if not structured_transactions:
            return {"indexed": 0, "errors": 0, "total": 0, "responses": []}

        batch_size = initial_batch_size or self.default_batch_size
        min_batch_size = 50
        max_batch_size = 2000

        total_indexed = 0
        total_errors = 0
        responses: List[Dict[str, Any]] = []

        idx = 0
        while idx < len(structured_transactions):
            current_batch = structured_transactions[idx: idx + batch_size]
            attempt = 0
            backoff = 1
            while True:
                # Préparer les données bulk pour ce sous-batch
                bulk_body = []
                for tx in current_batch:
                    doc_id = f"user_{tx.user_id}_tx_{tx.transaction_id}"
                    bulk_body.append(json.dumps({"index": {"_index": self.index_name, "_id": doc_id}}))
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
                        "updated_at": datetime.now().isoformat(),
                    }
                    bulk_body.append(json.dumps(doc))

                bulk_data = "\n".join(bulk_body) + "\n"

                start_time = time.perf_counter()
                try:
                    async with self.session.post(
                        f"{self.base_url}/{self.index_name}/_bulk",
                        data=bulk_data,
                        headers={"Content-Type": "application/x-ndjson"},
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            elapsed = time.perf_counter() - start_time

                            indexed_count = 0
                            error_count = 0
                            for i, item in enumerate(result.get("items", [])):
                                tx_id = current_batch[i].transaction_id
                                if item.get("index", {}).get("status") in [200, 201]:
                                    indexed_count += 1
                                    responses.append({"transaction_id": tx_id, "success": True})
                                else:
                                    error_count += 1
                                    err = item.get("index", {}).get("error", {}).get("reason", "Unknown error")
                                    responses.append({"transaction_id": tx_id, "success": False, "error": err})
                                    logger.error(f"❌ Erreur bulk item: {item['index']}")

                            total_indexed += indexed_count
                            total_errors += error_count

                            # Adapter la taille du prochain batch en fonction du temps
                            if elapsed > target_time and batch_size > min_batch_size:
                                batch_size = max(min_batch_size, batch_size // 2)
                            elif elapsed < target_time / 2 and batch_size < max_batch_size:
                                batch_size = min(max_batch_size, batch_size * 2)

                            break  # sortie de la boucle de retry
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"Bulk request failed (status {response.status}): {error_text}. Retrying..."
                            )
                except Exception as e:
                    logger.warning(f"Bulk indexation exception: {e}. Retrying...")

                attempt += 1
                if attempt >= max_retries:
                    # Considérer tout le batch en erreur
                    total_errors += len(current_batch)
                    for tx in current_batch:
                        responses.append({"transaction_id": tx.transaction_id, "success": False, "error": "max_retries"})
                    break
                await asyncio.sleep(backoff)
                backoff *= 2

            idx += len(current_batch)

        summary = {
            "indexed": total_indexed,
            "errors": total_errors,
            "total": len(structured_transactions),
            "responses": responses,
        }

        logger.info(
            f"📦 Bulk indexation adaptative: {summary['indexed']}/{summary['total']} succès"
        )
        return summary
    
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
    
    async def document_exists(self, document_id: str) -> bool:
        """
        Vérifie si un document existe dans l'index.
        
        Args:
            document_id: ID du document à vérifier
            
        Returns:
            bool: True si le document existe
        """
        if not self._initialized:
            return False
        
        try:
            async with self.session.head(
                f"{self.base_url}/{self.index_name}/_doc/{document_id}"
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification existence document {document_id}: {e}")
            return False
    
    async def index_document(self, document_id: str, document: Dict[str, Any]) -> bool:
        """
        Indexe un document unique dans Elasticsearch.
        
        Args:
            document_id: ID du document
            document: Données du document
            
        Returns:
            bool: True si l'indexation a réussi
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        try:
            async with self.session.put(
                f"{self.base_url}/{self.index_name}/_doc/{document_id}",
                json=document
            ) as response:
                if response.status in [200, 201]:
                    logger.debug(f"📝 Document {document_id} indexé avec succès")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur indexation {document_id}: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Exception lors de l'indexation {document_id}: {e}")
            return False
    
    async def bulk_index_documents(self, documents_to_index: List[Dict], force_update: bool = False) -> Dict[str, Any]:
        """
        Indexe un lot de documents dans Elasticsearch (format adapté au nouveau processor).
        
        Args:
            documents_to_index: Liste de dicts avec 'id', 'document', 'transaction_id'
            force_update: Force la mise à jour même si les documents existent
            
        Returns:
            Dict: Résumé de l'indexation avec format adapté
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        if not documents_to_index:
            return {"indexed": 0, "errors": 0, "total": 0, "responses": []}
        
        # Préparer le bulk request
        bulk_body = []
        
        for item in documents_to_index:
            doc_id = item["id"]
            document = item["document"]
            
            # Action header
            bulk_body.append(json.dumps({
                "index": {
                    "_index": self.index_name,
                    "_id": doc_id
                }
            }))
            
            # Document data
            bulk_body.append(json.dumps(document))
        
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
                    
                    # Analyser les résultats et créer la réponse adaptée
                    indexed_count = 0
                    error_count = 0
                    responses = []
                    
                    for i, item in enumerate(result.get("items", [])):
                        if "index" in item:
                            if item["index"].get("status") in [200, 201]:
                                indexed_count += 1
                                responses.append({
                                    "success": True,
                                    "transaction_id": documents_to_index[i].get("transaction_id"),
                                    "document_id": item["index"]["_id"]
                                })
                            else:
                                error_count += 1
                                error_msg = item["index"].get("error", {}).get("reason", "Unknown error")
                                responses.append({
                                    "success": False,
                                    "transaction_id": documents_to_index[i].get("transaction_id"),
                                    "error": error_msg
                                })
                                logger.error(f"❌ Erreur bulk item: {item['index']}")
                    
                    summary = {
                        "indexed": indexed_count,
                        "errors": error_count,
                        "total": len(documents_to_index),
                        "responses": responses
                    }
                    
                    logger.info(f"📦 Bulk indexation: {summary['indexed']}/{summary['total']} succès")
                    return summary
                    
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur bulk request: {response.status} - {error_text}")
                    
                    # Retourner des erreurs pour tous les documents
                    responses = [
                        {
                            "success": False,
                            "transaction_id": item.get("transaction_id"),
                            "error": f"Bulk request failed: {response.status}"
                        }
                        for item in documents_to_index
                    ]
                    
                    return {
                        "indexed": 0, 
                        "errors": len(documents_to_index), 
                        "total": len(documents_to_index),
                        "responses": responses
                    }
                    
        except Exception as e:
            logger.error(f"❌ Exception bulk indexation: {e}")
            
            # Retourner des erreurs pour tous les documents
            responses = [
                {
                    "success": False,
                    "transaction_id": item.get("transaction_id"),
                    "error": f"Exception: {str(e)}"
                }
                for item in documents_to_index
            ]
            
            return {
                "indexed": 0, 
                "errors": len(documents_to_index), 
                "total": len(documents_to_index),
                "responses": responses
            }
    
    async def delete_user_transactions(self, user_id: int) -> int:
        """
        Supprime toutes les transactions d'un utilisateur et retourne le nombre supprimé.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            int: Nombre de documents supprimés
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
                    return deleted_count
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur suppression user {user_id}: {response.status} - {error_text}")
                    return 0
                    
        except Exception as e:
            logger.error(f"❌ Exception suppression user {user_id}: {e}")
            return 0
    
    async def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère les statistiques d'un utilisateur dans Elasticsearch.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Statistiques de l'utilisateur
        """
        if not self._initialized:
            return {"error": "Client not initialized"}
        
        try:
            # Requête de statistiques
            stats_query = {
                "query": {
                    "term": {
                        "user_id": user_id
                    }
                },
                "aggs": {
                    "total_transactions": {
                        "value_count": {
                            "field": "transaction_id"
                        }
                    },
                    "total_amount": {
                        "sum": {
                            "field": "amount"
                        }
                    },
                    "average_amount": {
                        "avg": {
                            "field": "amount"
                        }
                    },
                    "transaction_types": {
                        "terms": {
                            "field": "transaction_type"
                        }
                    },
                    "date_range": {
                        "date_range": {
                            "field": "date",
                            "ranges": [
                                {"key": "last_30_days", "from": "now-30d"},
                                {"key": "last_90_days", "from": "now-90d"}
                            ]
                        }
                    }
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_search",
                json=stats_query
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    aggregations = result.get("aggregations", {})
                    
                    return {
                        "user_id": user_id,
                        "total_transactions": aggregations.get("total_transactions", {}).get("value", 0),
                        "total_amount": aggregations.get("total_amount", {}).get("value", 0),
                        "average_amount": aggregations.get("average_amount", {}).get("value", 0),
                        "transaction_types": [
                            {"type": bucket["key"], "count": bucket["doc_count"]}
                            for bucket in aggregations.get("transaction_types", {}).get("buckets", [])
                        ],
                        "date_ranges": {
                            bucket["key"]: bucket["doc_count"]
                            for bucket in aggregations.get("date_range", {}).get("buckets", [])
                        }
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur stats user {user_id}: {response.status} - {error_text}")
                    return {"error": f"Query failed: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Exception stats user {user_id}: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """
        Vérifie la santé de la connexion Elasticsearch.
        
        Returns:
            bool: True si Elasticsearch est accessible
        """
        if not self._initialized:
            return False
        
        try:
            async with self.session.get(
                f"{self.base_url}/_cluster/health"
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """
        Récupère les informations du cluster Elasticsearch.
        
        Returns:
            Dict: Informations du cluster
        """
        if not self._initialized:
            return {"error": "Client not initialized"}
        
        try:
            async with self.session.get(self.base_url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Request failed: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Erreur cluster info: {e}")
            return {"error": str(e)}
        
    async def close(self):
        """Ferme la session HTTP."""
        if self.session:
            await self.session.close()
            logger.info("🔌 Session Elasticsearch fermée")