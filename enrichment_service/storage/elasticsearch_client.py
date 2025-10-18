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
from enrichment_service.storage.index_management import ensure_template_and_policy

logger = logging.getLogger("enrichment_service.elasticsearch")

class ElasticsearchClient:
    """Client HTTP pour indexer dans Elasticsearch - Support dual index."""

    def __init__(self):
        self.base_url = settings.ELASTICSEARCH_URL
        self.transactions_index = "harena_transactions"  # Index des transactions (nettoyé)
        self.accounts_index = "harena_accounts"         # Nouvel index des comptes
        self.index_name = self.transactions_index  # Rétrocompatibilité
        self.session = None
        self._initialized = False
        # Batch size used as a starting point for adaptive bulk indexing
        self.default_batch_size = 500

    async def initialize(self):
        """Initialise la connexion Elasticsearch."""
        if not self.base_url:
            raise ValueError("ELASTICSEARCH_URL is required")
        
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
        
        # Créer les index s'ils n'existent pas (transactions + accounts)
        await self._setup_indexes()
        # Précharger certaines requêtes pour réchauffer les caches
        await self._warm_indexes()
        self._initialized = True
        logger.info(f"📄 Client Elasticsearch initialisé (transactions: '{self.transactions_index}', accounts: '{self.accounts_index}')")
    
    async def _setup_indexes(self):
        """Crée les index transactions et accounts s'ils n'existent pas."""
        # S'assurer que le template et la politique ILM existent
        await ensure_template_and_policy(self.session, self.base_url)

        # 1. Créer l'index transactions (nettoyé)
        await self._create_transactions_index()
        
        # 2. Créer l'index accounts (nouveau)
        await self._create_accounts_index()

    async def _create_transactions_index(self):
        """Crée l'index des transactions (version nettoyée)."""
        # Vérifier l'existence de l'alias (index de rollover)
        async with self.session.head(f"{self.base_url}/{self.transactions_index}") as response:
            if response.status == 200:
                logger.info(f"📚 Index transactions '{self.transactions_index}' existe déjà")
                return

        # Créer l'index initial avec alias pour le rollover
        index_name = f"{self.transactions_index}-000001"
        body = {
            "aliases": {
                self.transactions_index: {"is_write_index": True}
            }
        }

        # Créer l'index avec mapping optimisé (NETTOYÉ - sans données de compte)
        mapping = {
            "mappings": {
                "properties": {
                    # Identifiants
                    "user_id": {"type": "integer"},
                    "transaction_id": {"type": "keyword"},
                    "account_id": {"type": "integer"},  # 🔗 LIEN vers index accounts

                    # Contenu recherchable
                    "searchable_text": {
                        "type": "text",
                        "analyzer": "french_financial",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "primary_description": {
                        "type": "text",
                        "analyzer": "french_financial",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "merchant_name": {
                        "type": "text",
                        "analyzer": "merchant_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },

                    # Données financières
                    "amount": {"type": "float"},
                    "amount_abs": {"type": "float"},
                    "transaction_type": {"type": "keyword"},
                    "currency_code": {"type": "keyword"},
                    "quality_score": {"type": "float"},
                    

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
                    "indexed_at": {"type": "date"},
                    "version": {"type": "keyword"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "filter": {
                        "french_stop": {
                            "type": "stop",
                            "stopwords": "_french_"
                        },
                        "french_elision": {
                            "type": "elision",
                            "articles_case": True,
                            "articles": [
                                "l", "m", "t", "qu", "n", "s", "j", "d", "c", "jusqu",
                                "quoiqu", "lorsqu", "puisqu"
                            ]
                        },
                        "french_stemmer": {
                            "type": "stemmer",
                            "language": "light_french"
                        }
                    },
                    "analyzer": {
                        "french_financial": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "asciifolding",
                                "french_elision",
                                "french_stop",
                                "french_stemmer"
                            ]
                        },
                        "merchant_analyzer": {
                            "tokenizer": "simple",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                },
                "index": {
                    "max_result_window": 10000
                }
            }
        }

        async with self.session.put(f"{self.base_url}/{index_name}", json=body) as response:
            if response.status in [200, 201]:
                logger.info(f"✅ Index transactions '{index_name}' créé avec succès")
            else:
                error_text = await response.text()
                logger.error(f"❌ Erreur création index transactions: {response.status} - {error_text}")
                raise Exception(f"Failed to create transactions index: {error_text}")

    async def _create_accounts_index(self):
        """Crée l'index des comptes (nouveau)."""
        # Vérifier l'existence de l'index
        async with self.session.head(f"{self.base_url}/{self.accounts_index}") as response:
            if response.status == 200:
                logger.info(f"📚 Index accounts '{self.accounts_index}' existe déjà")
                return

        # Mapping spécialisé pour les comptes
        accounts_mapping = {
            "mappings": {
                "properties": {
                    # Identifiants
                    "user_id": {"type": "integer"},
                    "account_id": {"type": "integer"},  # Clé primaire
                    
                    # Données du compte
                    "account_name": {
                        "type": "text",
                        "analyzer": "simple",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "account_type": {"type": "keyword"},
                    "account_balance": {"type": "float"},  # ⚡ SOLDE ACTUEL
                    "account_currency": {"type": "keyword"},
                    
                    # Métadonnées
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "last_sync_timestamp": {"type": "date"},
                    "is_active": {"type": "boolean"},
                    
                    # Statistiques optionnelles (calculées)
                    "total_transactions": {"type": "integer"},
                    "last_transaction_date": {"type": "date"}
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

        async with self.session.put(f"{self.base_url}/{self.accounts_index}", json=accounts_mapping) as response:
            if response.status in [200, 201]:
                logger.info(f"✅ Index accounts '{self.accounts_index}' créé avec succès")
            else:
                error_text = await response.text()
                logger.error(f"❌ Erreur création index accounts: {response.status} - {error_text}")
                raise Exception(f"Failed to create accounts index: {error_text}")

    async def _warm_indexes(self):
        """Exécute des requêtes pour réchauffer les caches Elasticsearch."""
        # Requêtes de warmup pour les transactions
        transactions_queries = [
            # Requête sur merchant_name.keyword pour réchauffer l'agrégation
            {"size": 0, "query": {"match_all": {}}, "aggs": {"merchants": {"terms": {"field": "merchant_name.keyword", "size": 1}}}}
        ]
        
        # Requêtes de warmup pour les comptes
        accounts_queries = [
            # Statistiques sur le solde pour forcer le chargement du champ
            {"size": 0, "aggs": {"balance_stats": {"stats": {"field": "account_balance"}}}},
        ]

        # Warmup transactions
        for query in transactions_queries:
            try:
                async with self.session.post(
                    f"{self.base_url}/{self.transactions_index}/_search",
                    json=query
                ) as response:
                    await response.text()
            except Exception as e:
                logger.debug(f"Transactions warmup query failed: {e}")
        
        # Warmup accounts
        for query in accounts_queries:
            try:
                async with self.session.post(
                    f"{self.base_url}/{self.accounts_index}/_search",
                    json=query
                ) as response:
                    await response.text()
            except Exception as e:
                logger.debug(f"Accounts warmup query failed: {e}")
    
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
                # 🔧 NOUVEAU : Type de document
                "document_type": "transaction",
                
                # Identifiants
                "user_id": structured_transaction.user_id,
                "transaction_id": structured_transaction.transaction_id,
                "account_id": structured_transaction.account_id,
                "account_name": getattr(structured_transaction, 'account_name', ''),
                "account_type": getattr(structured_transaction, 'account_type', ''),
                "account_balance": getattr(structured_transaction, 'account_balance', None),
                "account_currency": getattr(structured_transaction, 'account_currency', ''),

                # Contenu recherchable
                "searchable_text": structured_transaction.searchable_text,
                "primary_description": structured_transaction.primary_description,
                "merchant_name": getattr(structured_transaction, 'merchant_name', ''),

                # Données financières
                "amount": structured_transaction.amount,
                "amount_abs": structured_transaction.amount_abs,
                "transaction_type": structured_transaction.transaction_type,
                "currency_code": structured_transaction.currency_code,
                "quality_score": getattr(structured_transaction, 'quality_score', 1.0),
                
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
                    logger.debug(f"📄 Transaction {doc_id} indexée avec succès")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur indexation {doc_id}: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Exception lors de l'indexation: {e}")
            return False
    
    async def index_transactions_batch(
        self, structured_transactions: List
    ) -> Dict[str, Any]:
        """Indexe un lot de transactions en déléguant à ``bulk_index_documents``.

        Cette méthode est conservée pour compatibilité mais utilise désormais
        exclusivement ``StructuredTransaction.to_elasticsearch_document()`` pour
        générer les documents avant de les passer au bulk indexer.
        """

        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")

        if not structured_transactions:
            return {"indexed": 0, "errors": 0, "total": 0, "responses": []}

        documents_to_index = []
        for tx in structured_transactions:
            documents_to_index.append(
                {
                    "id": tx.get_document_id(),
                    "document": tx.to_elasticsearch_document(),
                    "transaction_id": tx.transaction_id,
                }
            )

        return await self.bulk_index_documents(documents_to_index)
    
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
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}},
                            {"term": {"document_type": "transaction"}}
                        ]
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

    async def index_accounts(self, accounts: List[Any], user_id: int) -> int:
        """
        🔧 CORRIGÉ : Indexe des documents de comptes dans le MÊME index que les transactions.

        Args:
            accounts: Liste d'objets représentant les comptes à indexer
            user_id: Identifiant de l'utilisateur propriétaire des comptes

        Returns:
            int: Nombre de comptes effectivement indexés
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")

        if not accounts:
            return 0

        bulk_body: List[str] = []
        for acc in accounts:
            # Utiliser en priorité l'identifiant métier (bridge_account_id)
            account_id = getattr(acc, "bridge_account_id", getattr(acc, "account_id", getattr(acc, "id", None)))
            if account_id is None:
                continue

            document = {
                # 🔧 NOUVEAU : Type de document pour différencier
                "document_type": "account",
                
                # Données de compte
                "account_id": account_id,
                "user_id": user_id,
                "account_name": getattr(acc, "account_name", None),
                "account_type": getattr(acc, "account_type", None),
                "account_balance": getattr(acc, "balance", None),
                "account_currency": getattr(acc, "currency_code", None),
                "last_sync_timestamp": getattr(acc, "last_sync_timestamp", None).isoformat()
                if getattr(acc, "last_sync_timestamp", None)
                else None,
                
                # Métadonnées
                "indexed_at": datetime.now().isoformat(),
                "version": "2.0-elasticsearch"
            }

            doc_id = f"user_{user_id}_acc_{account_id}"
            
            # 🔧 CORRIGÉ : Utiliser le même index (harena_transactions)
            bulk_body.append(json.dumps({"index": {"_index": self.index_name, "_id": doc_id}}))
            bulk_body.append(json.dumps(document))

        if not bulk_body:
            return 0

        bulk_data = "\n".join(bulk_body) + "\n"

        try:
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_bulk",
                data=bulk_data,
                headers={"Content-Type": "application/x-ndjson"},
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    indexed = sum(
                        1
                        for item in result.get("items", [])
                        if item.get("index", {}).get("status") in [200, 201]
                    )
                    logger.info(f"✅ {indexed}/{len(accounts)} comptes indexés pour user {user_id}")
                    return indexed
                else:
                    error_text = await response.text()
                    logger.error(
                        f"❌ Erreur indexation comptes: {response.status} - {error_text}"
                    )
                    return 0
        except Exception as e:
            logger.error(f"❌ Exception indexation comptes: {e}")
            return 0
    
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
                    logger.debug(f"📄 Document {document_id} indexé avec succès")
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
                    
                    logger.debug(f"📦 Bulk indexation: {summary['indexed']}/{summary['total']} succès")
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

    async def bulk_update_documents(self, updates: List[Dict]) -> Dict[str, Any]:
        """
        Met à jour un lot de documents dans Elasticsearch (mise à jour partielle).
        
        Args:
            updates: Liste de dicts avec 'id' et 'update' (les champs à mettre à jour)
            
        Returns:
            Dict: Résumé de la mise à jour
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        if not updates:
            return {"updated": 0, "errors": 0, "total": 0}
        
        # Préparer le bulk request pour updates
        bulk_body = []
        
        for item in updates:
            doc_id = item["id"]
            update_data = item["update"]
            
            # Action header pour update
            bulk_body.append(json.dumps({
                "update": {
                    "_index": self.index_name,
                    "_id": doc_id
                }
            }))
            
            # Document data pour update
            bulk_body.append(json.dumps({
                "doc": update_data,
                "doc_as_upsert": False  # Ne pas créer si le doc n'existe pas
            }))
        
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
                    updated_count = 0
                    error_count = 0
                    
                    for item in result.get("items", []):
                        if "update" in item:
                            if item["update"].get("status") in [200, 201]:
                                updated_count += 1
                            else:
                                error_count += 1
                                error_msg = item["update"].get("error", {}).get("reason", "Unknown error")
                                logger.warning(f"⚠️ Erreur update: {error_msg}")
                    
                    logger.info(f"📦 Bulk update: {updated_count}/{len(updates)} documents mis à jour")
                    return {
                        "updated": updated_count,
                        "errors": error_count,
                        "total": len(updates)
                    }
                    
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur bulk update: {response.status} - {error_text}")
                    return {
                        "updated": 0,
                        "errors": len(updates),
                        "total": len(updates)
                    }
                    
        except Exception as e:
            logger.error(f"❌ Exception lors du bulk update: {e}")
            return {
                "updated": 0,
                "errors": len(updates),
                "total": len(updates)
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
            # 🔧 CORRIGÉ : Supprimer seulement les transactions (pas les comptes)
            delete_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}},
                            {"term": {"document_type": "transaction"}}
                        ]
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
            # 🔧 CORRIGÉ : Filtrer seulement les transactions
            stats_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}},
                            {"term": {"document_type": "transaction"}}
                        ]
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
    
    async def ping(self) -> bool:
        """
        🔧 NOUVEAU : Ping Elasticsearch pour health check.
        
        Returns:
            bool: True si Elasticsearch répond
        """
        if not self._initialized:
            return False
        
        try:
            async with self.session.get(
                f"{self.base_url}/_cluster/health"
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"❌ Ping failed: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        Vérifie la santé de la connexion Elasticsearch.
        
        Returns:
            bool: True si Elasticsearch est accessible
        """
        return await self.ping()
    
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
        
    # =============================================
    # 🏦 MÉTHODES POUR GESTION DES COMPTES
    # =============================================
    
    async def index_account(self, account_data: Dict[str, Any]) -> bool:
        """
        Indexe ou met à jour un compte dans l'index accounts.
        
        Args:
            account_data: Données du compte à indexer
            
        Returns:
            bool: True si l'indexation a réussi
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        account_id = account_data.get('account_id')
        user_id = account_data.get('user_id')
        
        if not account_id or not user_id:
            logger.error("❌ account_id et user_id requis pour indexer un compte")
            return False
        
        # ID du document : user_<user_id>_acc_<account_id>
        document_id = f"user_{user_id}_acc_{account_id}"
        
        try:
            # Ajouter timestamp de mise à jour
            account_data["updated_at"] = datetime.now().isoformat()
            
            async with self.session.put(
                f"{self.base_url}/{self.accounts_index}/_doc/{document_id}",
                json=account_data
            ) as response:
                if response.status in [200, 201]:
                    logger.debug(f"✅ Compte {account_id} indexé avec succès")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur indexation compte {account_id}: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Exception lors de l'indexation compte {account_id}: {e}")
            return False
    
    async def bulk_index_accounts(self, accounts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Indexe un lot de comptes en mode bulk.
        
        Args:
            accounts: Liste des comptes à indexer
            
        Returns:
            Dict: Résumé de l'indexation
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        if not accounts:
            return {"indexed": 0, "errors": 0, "total": 0}
        
        # Préparer le bulk request
        bulk_body = []
        
        for account in accounts:
            account_id = account.get('account_id')
            user_id = account.get('user_id')
            
            if not account_id or not user_id:
                logger.warning(f"⚠️ Compte invalide ignoré: {account}")
                continue
            
            document_id = f"user_{user_id}_acc_{account_id}"
            account["updated_at"] = datetime.now().isoformat()
            
            # Action header
            bulk_body.append(json.dumps({
                "index": {
                    "_index": self.accounts_index,
                    "_id": document_id
                }
            }))
            
            # Document data
            bulk_body.append(json.dumps(account))
        
        if not bulk_body:
            return {"indexed": 0, "errors": 0, "total": 0}
        
        # Joindre avec des nouvelles lignes (format bulk)
        bulk_data = "\n".join(bulk_body) + "\n"
        
        try:
            async with self.session.post(
                f"{self.base_url}/{self.accounts_index}/_bulk",
                data=bulk_data,
                headers={"Content-Type": "application/x-ndjson"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Analyser les résultats
                    total = len(accounts)
                    indexed = 0
                    errors = 0
                    
                    for item in result.get("items", []):
                        if "index" in item:
                            if item["index"]["status"] in [200, 201]:
                                indexed += 1
                            else:
                                errors += 1
                    
                    logger.info(f"📦 Bulk accounts: {indexed}/{total} comptes indexés")
                    return {"indexed": indexed, "errors": errors, "total": total}
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Bulk accounts failed: {response.status} - {error_text}")
                    return {"indexed": 0, "errors": len(accounts), "total": len(accounts)}
                    
        except Exception as e:
            logger.error(f"❌ Exception bulk accounts: {e}")
            return {"indexed": 0, "errors": len(accounts), "total": len(accounts)}
    
    async def get_user_accounts(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Récupère tous les comptes d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List: Liste des comptes avec leurs soldes
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        try:
            query = {
                "query": {"term": {"user_id": user_id}},
                "sort": [{"account_id": {"order": "asc"}}],
                "size": 100
            }
            
            async with self.session.post(
                f"{self.base_url}/{self.accounts_index}/_search",
                json=query
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    hits = result.get("hits", {}).get("hits", [])
                    
                    accounts = []
                    for hit in hits:
                        account = hit["_source"]
                        accounts.append(account)
                    
                    logger.debug(f"📋 {len(accounts)} comptes trouvés pour user {user_id}")
                    return accounts
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur récupération comptes user {user_id}: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Exception récupération comptes user {user_id}: {e}")
            return []
    
    async def get_account_balance(self, user_id: int, account_id: int) -> Optional[float]:
        """
        Récupère le solde d'un compte spécifique.
        
        Args:
            user_id: ID de l'utilisateur  
            account_id: ID du compte
            
        Returns:
            Optional[float]: Solde du compte ou None si non trouvé
        """
        if not self._initialized:
            raise ValueError("ElasticsearchClient not initialized")
        
        document_id = f"user_{user_id}_acc_{account_id}"
        
        try:
            async with self.session.get(
                f"{self.base_url}/{self.accounts_index}/_doc/{document_id}"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    account_balance = result.get("_source", {}).get("account_balance")
                    return account_balance
                elif response.status == 404:
                    logger.debug(f"🔍 Compte {account_id} non trouvé pour user {user_id}")
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Erreur récupération solde compte {account_id}: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Exception récupération solde compte {account_id}: {e}")
            return None

    async def close(self):
        """Ferme la session HTTP."""
        if self.session:
            await self.session.close()
            logger.info("🔌 Session Elasticsearch fermée")
