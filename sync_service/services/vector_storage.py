# sync_service/services/vector_storage.py
"""
Service pour le stockage vectoriel des données financières Harena.

Ce module fournit des fonctionnalités pour créer, gérer, stocker et interroger
les données (transactions, comptes, catégories, marchands, insights, actions)
dans une base de données vectorielle Qdrant.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timezone
import uuid
from uuid import UUID, uuid4

# Import des dépendances Qdrant
from qdrant_client import QdrantClient, models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# Import du service d'embedding local
from sync_service.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# --- Namespaces UUID constants pour la génération d'ID déterministes ---
# UUIDs fixes pour garantir que les mêmes données produisent les mêmes IDs
HARENA_NAMESPACE = UUID('f81d4fae-7dec-11d0-a765-00a0c91e6bf6')
HARENA_TRANSACTION_NAMESPACE = UUID('a1a2a3a4-b1b2-c1c2-d1d2-e1e2e3e4e5e6')
HARENA_ACCOUNT_NAMESPACE = UUID('b1b2b3b4-c1c2-d1d2-e1e2-f1f2f3f4f5f6')
HARENA_USER_METADATA_NAMESPACE = UUID('c1c2c3c4-d1d2-e1e2-f1f2-a1a2a3a4a5a6')
HARENA_MERCHANT_NAMESPACE = UUID('d1d2d3d4-e1e2-f1f2-a1a2-b1b2b3b4b5b6')


class VectorStorageService:
    """Service pour le stockage et la recherche vectorielle des données Harena."""

    # Noms des collections
    TRANSACTIONS_COLLECTION = "transactions"
    ACCOUNTS_COLLECTION = "accounts"
    CATEGORIES_COLLECTION = "categories"
    MERCHANTS_COLLECTION = "merchants"
    INSIGHTS_COLLECTION = "insights"
    STOCKS_COLLECTION = "stocks"
    USER_METADATA_COLLECTION = "user_metadata"

    # Configuration par défaut
    DEFAULT_VECTOR_SIZE = 1536
    DEFAULT_DISTANCE = qmodels.Distance.COSINE
    DEFAULT_INDEXING_THRESHOLD = 10000

    def __init__(self):
        """Initialise le service de stockage vectoriel."""
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_size = self.DEFAULT_VECTOR_SIZE

        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=60.0
            )
            self.client.get_collections()
            logger.info("Client Qdrant connecté avec succès à %s", self.qdrant_url)
        except Exception as e:
            logger.error("Erreur lors de la connexion à Qdrant (%s): %s", self.qdrant_url, e, exc_info=True)
            self.client = None

        self.embedding_service = EmbeddingService()

        if self.client:
            self._ensure_collections()
            logger.info("Service de stockage vectoriel initialisé.")
        else:
            logger.error("Service de stockage vectoriel NON initialisé en raison d'une erreur de connexion Qdrant.")

    def _create_collection_with_indexes(self, collection_name: str, indexes: List[Tuple[str, str]]):
        """Helper pour créer une collection et ses index de payload."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=self.DEFAULT_DISTANCE,
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=self.DEFAULT_INDEXING_THRESHOLD,
                )
            )
            logger.info(f"Collection Qdrant créée: {collection_name}")

            for field_name, field_type in indexes:
                try:
                    if field_type == "keyword": schema_type = qmodels.PayloadSchemaType.KEYWORD
                    elif field_type == "integer": schema_type = qmodels.PayloadSchemaType.INTEGER
                    elif field_type == "float": schema_type = qmodels.PayloadSchemaType.FLOAT
                    elif field_type == "boolean": schema_type = qmodels.PayloadSchemaType.BOOL
                    elif field_type == "datetime": schema_type = qmodels.PayloadSchemaType.DATETIME
                    elif field_type == "text":
                         schema_type = qmodels.TextIndexParams(
                             type="text", tokenizer=qmodels.TokenizerType.WORD,
                             min_token_len=2, max_token_len=15, lowercase=True
                         )
                    else:
                        logger.warning(f"Type d'index non supporté '{field_type}' pour {field_name} dans {collection_name}. Ignoré.")
                        continue

                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=schema_type
                    )
                    logger.debug(f"Index créé pour {field_name} ({field_type}) dans {collection_name}")
                except Exception as index_error:
                    logger.error(f"Erreur lors de la création de l'index {field_name} dans {collection_name}: {index_error}")
            logger.info(f"Index créés pour la collection: {collection_name}")

        except UnexpectedResponse as e:
            if e.status_code in [400, 409] and "already exists" in str(e.content).lower():
                logger.info(f"Collection Qdrant existante ({e.status_code}): {collection_name}")
            else:
                logger.error(f"Erreur inattendue lors de la création de la collection {collection_name}: {e.status_code} {e.content}")
                raise
        except Exception as e:
            logger.error(f"Erreur générale lors de la création de la collection {collection_name}: {e}")
            raise

    def _ensure_collections(self):
        """S'assure que toutes les collections requises existent et ont les bons index."""
        if not self.client:
            logger.error("Client Qdrant non disponible, impossible de vérifier les collections.")
            return

        try:
            existing_collections = {col.name for col in self.client.get_collections().collections}
            logger.info(f"Collections Qdrant existantes: {existing_collections}")

            # Définitions des index pour chaque collection
            collection_definitions = {
                self.TRANSACTIONS_COLLECTION: [
                    ("user_id", "integer"), ("account_id", "integer"), ("bridge_transaction_id", "keyword"),
                    ("transaction_date", "datetime"), ("category_id", "integer"), ("merchant_id", "keyword"),
                    ("amount", "float"), ("operation_type", "keyword"), ("is_recurring", "boolean"),
                    ("currency_code", "keyword"),
                ],
                self.ACCOUNTS_COLLECTION: [
                    ("user_id", "integer"), ("item_id", "integer"), ("bridge_account_id", "integer"),
                    ("type", "keyword"), ("currency_code", "keyword"), ("provider_id", "integer"),
                    ("pro", "boolean"), ("balance", "float"), ("iban", "keyword"),
                ],
                self.CATEGORIES_COLLECTION: [
                    ("bridge_category_id", "integer"), ("parent_id", "integer"), ("name", "keyword"),
                ],
                self.MERCHANTS_COLLECTION: [
                    ("normalized_name", "keyword"), ("category_id", "integer"), ("display_name", "text"),
                ],
                self.INSIGHTS_COLLECTION: [
                    ("user_id", "integer"), ("category_id", "integer"), ("period_type", "keyword"),
                    ("period_start", "datetime"), ("period_end", "datetime"),
                ],
                self.STOCKS_COLLECTION: [
                    ("user_id", "integer"), ("account_id", "integer"), ("bridge_stock_id", "integer"),
                    ("ticker", "keyword"), ("isin", "keyword"), ("currency_code", "keyword"),
                    ("value_date", "datetime"),
                ],
            }

            # Créer les collections standards avec vecteurs
            for name, indexes in collection_definitions.items():
                if (name not in existing_collections) and (name != self.USER_METADATA_COLLECTION):
                    self._create_collection_with_indexes(name, indexes)
                else:
                    logger.info(f"Collection Qdrant existante: {name}")

            # --- Collection User Metadata (sans vecteurs) ---
            if self.USER_METADATA_COLLECTION not in existing_collections:
                meta_indexes = [("user_id", "integer"), ("status", "keyword")]
                try:
                    self.client.create_collection(
                        collection_name=self.USER_METADATA_COLLECTION,
                        vectors_config={} # Config vide car pas de vecteurs mais argument requis
                    )
                    logger.info(f"Collection Qdrant créée: {self.USER_METADATA_COLLECTION}")
                    for field_name, field_type in meta_indexes:
                        try:
                            schema_type = qmodels.PayloadSchemaType.KEYWORD if field_type == 'keyword' else qmodels.PayloadSchemaType.INTEGER
                            self.client.create_payload_index(
                                collection_name=self.USER_METADATA_COLLECTION,
                                field_name=field_name,
                                field_schema=schema_type
                            )
                            logger.debug(f"Index créé pour {field_name} ({field_type}) dans {self.USER_METADATA_COLLECTION}")
                        except Exception as index_error:
                            logger.error(f"Erreur lors de la création de l'index {field_name} dans {self.USER_METADATA_COLLECTION}: {index_error}")
                    logger.info(f"Index créés pour la collection: {self.USER_METADATA_COLLECTION}")
                except UnexpectedResponse as e:
                     if e.status_code in [400, 409] and "already exists" in str(e.content).lower():
                         logger.info(f"Collection Qdrant existante ({e.status_code}): {self.USER_METADATA_COLLECTION}")
                     else:
                         logger.error(f"Erreur inattendue lors de la création de la collection {self.USER_METADATA_COLLECTION}: {e.status_code} {e.content}")
                         raise
                except Exception as meta_e:
                     logger.error(f"Erreur générale lors de la création de la collection {self.USER_METADATA_COLLECTION}: {meta_e}", exc_info=True)
            else:
                logger.info(f"Collection Qdrant existante: {self.USER_METADATA_COLLECTION}")

        except Exception as e:
            logger.error(f"Erreur générale lors de la vérification/création des collections: {e}", exc_info=True)

    def _prepare_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoie et prépare le payload pour Qdrant."""
        payload = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                if value.tzinfo is None: value = value.replace(tzinfo=timezone.utc)
                payload[key] = value.isoformat()
            elif isinstance(value, UUID):
                payload[key] = str(value)
            elif value is not None:
                payload[key] = value
        return payload

    # --- Méthodes pour générer les IDs de points Qdrant de façon cohérente ---

    def _get_transaction_point_id(self, user_id: int, bridge_transaction_id: str) -> str:
        """Génère un ID de point UUID déterministe pour une transaction."""
        return str(uuid.uuid5(HARENA_TRANSACTION_NAMESPACE, f"user_{user_id}_tx_{bridge_transaction_id}"))

    def _get_account_point_id(self, user_id: int, bridge_account_id: int) -> str:
        """Génère un ID de point UUID déterministe pour un compte."""
        return str(uuid.uuid5(HARENA_ACCOUNT_NAMESPACE, f"user_{user_id}_account_{bridge_account_id}"))

    def _get_merchant_point_id(self, normalized_name: str) -> str:
        """Génère un ID de point UUID déterministe pour un marchand basé sur son nom normalisé."""
        return str(uuid.uuid5(HARENA_MERCHANT_NAMESPACE, normalized_name))

    def _get_user_metadata_point_id(self, user_id: int) -> str:
        """Génère un ID de point UUID déterministe pour les métadonnées utilisateur."""
        return str(uuid.uuid5(HARENA_USER_METADATA_NAMESPACE, str(user_id)))

    # --- Méthodes de Stockage (Transactions) ---

    async def store_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Stocke une transaction unique."""
        if not self.client: return False
        try:
            transaction_id_str = str(transaction.get("bridge_transaction_id"))
            user_id = transaction.get("user_id")
            if not transaction_id_str or user_id is None:
                logger.error("ID transaction Bridge ou User ID manquant.")
                return False

            # Utiliser l'ID UUID déterministe
            point_id_str = self._get_transaction_point_id(user_id, transaction_id_str)

            description = transaction.get("clean_description") or transaction.get("description", "")
            embedding = await self.embedding_service.get_embedding(description)

            payload = self._prepare_payload({
                "user_id": user_id,
                "account_id": transaction.get("account_id"),
                "bridge_transaction_id": transaction_id_str,
                "amount": transaction.get("amount", 0.0),
                "currency_code": transaction.get("currency_code", "EUR"),
                "description": transaction.get("description", ""),
                "clean_description": description,
                "transaction_date": transaction.get("transaction_date"),
                "category_id": transaction.get("category_id"),
                "operation_type": transaction.get("operation_type"),
                "is_recurring": transaction.get("is_recurring", False),
                "merchant_id": transaction.get("merchant_id"),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            })

            self.client.upsert(
                collection_name=self.TRANSACTIONS_COLLECTION,
                points=[qmodels.PointStruct(id=point_id_str, vector=embedding, payload=payload)],
                wait=False
            )
            logger.debug(f"Transaction upserted: {point_id_str}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du stockage de la transaction {transaction.get('bridge_transaction_id')}: {e}", exc_info=True)
            return False

    async def batch_store_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot de transactions."""
        if not self.client or not transactions:
            logger.info("batch_store_transactions: Pas de client Qdrant ou liste vide")
            return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        logger.info(f"Début du batch_store_transactions pour {len(transactions)} transactions")
        
        points_to_upsert = []
        failed_ids = []
        processed_count = 0
        categories = set()  # Pour des statistiques supplémentaires

        logger.info(f"Préparation de l'embedding pour {len(transactions)} transactions")
        for transaction in transactions:
            processed_count += 1
            try:
                transaction_id_str = str(transaction.get("bridge_transaction_id"))
                user_id = transaction.get("user_id")
                if not transaction_id_str or user_id is None:
                    logger.warning(f"ID transaction Bridge ou User ID manquant dans le lot (transaction #{processed_count}). Skipping.")
                    failed_ids.append(transaction.get("bridge_transaction_id", "unknown"))
                    continue

                # Utiliser l'ID UUID déterministe
                point_id_str = self._get_transaction_point_id(user_id, transaction_id_str)
                description = transaction.get("clean_description") or transaction.get("description", "")
                
                # Collecter des statistiques sur les catégories
                if transaction.get("category_id"):
                    categories.add(transaction.get("category_id"))
                    
                # Générer l'embedding vectoriel
                embedding = await self.embedding_service.get_embedding(description)

                payload = self._prepare_payload({
                    "user_id": user_id,
                    "account_id": transaction.get("account_id"),
                    "bridge_transaction_id": transaction_id_str,
                    "amount": transaction.get("amount", 0.0),
                    "currency_code": transaction.get("currency_code", "EUR"),
                    "description": transaction.get("description", ""),
                    "clean_description": description,
                    "transaction_date": transaction.get("transaction_date"),
                    "booking_date": transaction.get("booking_date"),
                    "value_date": transaction.get("value_date"),
                    "category_id": transaction.get("category_id"),
                    "operation_type": transaction.get("operation_type"),
                    "is_recurring": transaction.get("is_recurring", False),
                    "merchant_id": transaction.get("merchant_id"),
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                })

                points_to_upsert.append(qmodels.PointStruct(id=point_id_str, vector=embedding, payload=payload))

            except Exception as e:
                logger.error(f"Erreur lors de la préparation de la transaction {transaction.get('bridge_transaction_id')} pour le batch: {e}")
                failed_ids.append(transaction.get("bridge_transaction_id", "unknown"))

        successful_count = 0
        if points_to_upsert:
            logger.info(f"Stockage vectoriel de {len(points_to_upsert)} transactions en {(len(points_to_upsert) + 99) // 100} chunks")
            try:
                chunk_size = 100
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    chunk_start = i + 1
                    chunk_end = min(i + chunk_size, len(points_to_upsert))
                    logger.debug(f"Uploading transactions chunk {i//chunk_size + 1}: {chunk_start}-{chunk_end} of {len(points_to_upsert)}")
                    
                    self.client.upsert(
                        collection_name=self.TRANSACTIONS_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} transactions terminé. Catégories: {len(categories)}")
            except UnexpectedResponse as e:
                logger.error(f"Erreur Qdrant lors du batch upsert des transactions: {e.status_code} - {e.content}")
                logger.error(f"Erreur lors du batch upsert des transactions: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_ids)
                successful_count = 0
            except Exception as e:
                logger.error(f"Erreur générale lors du batch upsert des transactions: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_ids)
                successful_count = 0
            else:
                failed_count = len(failed_ids)
        else:
            logger.warning("Aucun point transaction à upserter dans Qdrant")
            failed_count = len(failed_ids)

        result = {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_ids": failed_ids[:10],
            "categories_count": len(categories)
        }
        
        logger.info(f"Résultat batch_store_transactions: {result['status']}, {result['successful']}/{result['total']} réussis, {len(categories)} catégories")
        return result

    # --- Méthodes de Stockage (Comptes) ---

    async def store_account(self, account_data: Dict[str, Any]) -> bool:
        """Stocke un compte unique."""
        if not self.client: return False
        try:
            account_id = account_data.get("bridge_account_id")
            user_id = account_data.get("user_id")
            if account_id is None or user_id is None:
                logger.error("ID compte Bridge ou User ID manquant.")
                return False

            # Générer un ID UUID déterministe
            point_id = self._get_account_point_id(user_id, account_id)

            text_to_embed = f"{account_data.get('name', '')} {account_data.get('type', '')}"
            embedding = await self.embedding_service.get_embedding(text_to_embed.strip())

            payload = self._prepare_payload({
                "user_id": user_id,
                "item_id": account_data.get("item_id"),
                "bridge_account_id": account_id,
                "name": account_data.get("name"),
                "type": account_data.get("type"),
                "balance": account_data.get("balance"),
                "currency_code": account_data.get("currency_code"),
                "iban": account_data.get("iban"),
                "provider_id": account_data.get("provider_id"),
                "pro": account_data.get("pro", False),
                "loan_details": account_data.get("loan_details"),
                "bridge_updated_at": account_data.get("bridge_updated_at"),
                "last_synced_at": datetime.now(timezone.utc)
            })

            self.client.upsert(
                collection_name=self.ACCOUNTS_COLLECTION,
                points=[qmodels.PointStruct(id=point_id, vector=embedding, payload=payload)],
                wait=False
            )
            logger.debug(f"Compte upserted: {point_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du stockage du compte {account_id}: {e}", exc_info=True)
            return False

    async def batch_store_accounts(self, accounts_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot de comptes."""
        if not self.client or not accounts_list:
            logger.info("batch_store_accounts: Pas de client Qdrant ou liste vide")
            return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        logger.info(f"Début du batch_store_accounts pour {len(accounts_list)} comptes")
        
        points_to_upsert = []
        failed_ids = []
        processed_count = 0

        for account in accounts_list:
            processed_count += 1
            try:
                account_id = account.get("bridge_account_id")
                user_id = account.get("user_id")
                if account_id is None or user_id is None:
                    logger.warning(f"ID compte Bridge ou User ID manquant dans le lot. Skipping.")
                    failed_ids.append(account.get("bridge_account_id", "unknown"))
                    continue

                # Générer un ID UUID déterministe
                point_id = self._get_account_point_id(user_id, account_id)

                text_to_embed = f"{account.get('name', '')} {account.get('type', '')}"
                logger.debug(f"Génération d'embedding pour compte {account_id}: '{text_to_embed}'")
                embedding = await self.embedding_service.get_embedding(text_to_embed.strip())

                payload = self._prepare_payload({
                    "user_id": user_id,
                    "item_id": account.get("item_id"),
                    "bridge_account_id": account_id,
                    "name": account.get("name"),
                    "type": account.get("type"),
                    "balance": account.get("balance"),
                    "currency_code": account.get("currency_code"),
                    "iban": account.get("iban"),
                    "provider_id": account.get("provider_id"),
                    "pro": account.get("pro", False),
                    "loan_details": account.get("loan_details"),
                    "bridge_updated_at": account.get("bridge_updated_at"),
                    "last_synced_at": datetime.now(timezone.utc)
                })
                points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
                logger.debug(f"Compte {account_id} préparé pour upsert")
            except Exception as e:
                logger.error(f"Erreur lors de la préparation du compte {account.get('bridge_account_id')} pour le batch: {e}")
                failed_ids.append(account.get("bridge_account_id", "unknown"))

        successful_count = 0
        if points_to_upsert:
            try:
                logger.info(f"Tentative d'upsert de {len(points_to_upsert)} points comptes dans Qdrant")
                chunk_size = 100
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    logger.debug(f"Upsert du chunk de comptes {i//chunk_size + 1} (taille: {len(chunk)})")
                    self.client.upsert(
                        collection_name=self.ACCOUNTS_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} comptes terminé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des comptes: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_ids)
                successful_count = 0
            else:
                failed_count = len(failed_ids)
        else:
            logger.warning("Aucun point compte prêt pour l'upsert")
            failed_count = len(failed_ids)

        result = {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_ids": failed_ids[:10] if failed_ids else []
        }
        
        logger.info(f"Résultat batch_store_accounts: {result['status']}, {result['successful']}/{result['total']} réussis")
        return result

    # --- Méthodes de Stockage (Catégories) ---

    async def batch_store_categories(self, categories_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot de catégories."""
        if not self.client or not categories_list:
            logger.info("batch_store_categories: Pas de client Qdrant ou liste vide")
            return {"status": "noop", "total": 0, "successful": 0, "failed": 0}
        logger.info(f"Début du batch_store_categories pour {len(categories_list)} catégories")
        
        points_to_upsert = []
        failed_ids = []
        processed_count = 0
        for category in categories_list:
            processed_count += 1
            try:
                category_id = category.get("bridge_category_id")
                if category_id is None or (not isinstance(category_id, int)) or category_id < 0:
                    logger.warning(f"ID catégorie Bridge invalide ou manquant ({category_id}) dans le lot. Skipping.")
                    failed_ids.append(category.get("bridge_category_id", "unknown"))
                    continue
                # Utiliser directement l'ID numérique bridge_category_id
                point_id = str(category_id)
                text_to_embed = category.get("name", "")
                
                logger.debug(f"Génération d'embedding pour catégorie {category_id}: '{text_to_embed}'")
                embedding = await self.embedding_service.get_embedding(text_to_embed)
                payload = self._prepare_payload({
                    "bridge_category_id": category_id,
                    "name": text_to_embed,
                    "parent_id": category.get("parent_id"),
                    "last_synced_at": datetime.now(timezone.utc)
                })
                points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
                logger.debug(f"Catégorie {category_id} préparée pour upsert")
            except Exception as e:
                logger.error(f"Erreur lors de la préparation de la catégorie {category.get('bridge_category_id')} pour le batch: {e}")
                failed_ids.append(category.get("bridge_category_id", "unknown"))
        successful_count = 0
        if points_to_upsert:
            try:
                logger.info(f"Tentative d'upsert de {len(points_to_upsert)} points catégories dans Qdrant")
                chunk_size = 100
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    logger.debug(f"Upsert du chunk de catégories {i//chunk_size + 1} (taille: {len(chunk)})")
                    self.client.upsert(
                        collection_name=self.CATEGORIES_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} catégories terminé avec succès.")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des catégories: {e}")
                failed_count = len(points_to_upsert) - successful_count + len(failed_ids)
                successful_count = 0
            else:
                failed_count = len(failed_ids)
        else:
            logger.warning("Aucun point catégorie prêt pour l'upsert")
            failed_count = len(failed_ids)
        result = {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_ids": failed_ids[:10] if failed_ids else []
        }
        
        logger.info(f"Résultat batch_store_categories: {result['status']}, {result['successful']}/{result['total']} réussis")
        return result

    # --- Méthodes de Stockage (Marchands) ---

    async def store_merchant(self, merchant_data: Dict[str, Any]) -> Optional[str]:
        """Stocke un marchand unique. Retourne l'ID du point si succès."""
        if not self.client: return None
        try:
            normalized_name = merchant_data.get("normalized_name")
            if not normalized_name:
                logger.error("Nom normalisé du marchand manquant.")
                return None
                
            # Utiliser un ID UUID déterministe basé sur le nom normalisé
            point_id = self._get_merchant_point_id(normalized_name)

            text_to_embed = f"{merchant_data.get('display_name', '')} {normalized_name}"
            embedding = await self.embedding_service.get_embedding(text_to_embed.strip())

            payload = self._prepare_payload({
                "normalized_name": normalized_name,
                "display_name": merchant_data.get("display_name"),
                "category_id": merchant_data.get("category_id"),
                "source": merchant_data.get("source", "inferred"),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            })

            self.client.upsert(
                collection_name=self.MERCHANTS_COLLECTION,
                points=[qmodels.PointStruct(id=point_id, vector=embedding, payload=payload)],
                wait=False
            )
            logger.debug(f"Marchand upserted: {point_id} ({normalized_name})")
            return point_id
        except Exception as e:
            logger.error(f"Erreur lors du stockage du marchand {merchant_data.get('normalized_name')}: {e}", exc_info=True)
            return None

    async def find_merchant(self, normalized_name: str) -> Optional[str]:
        """Trouve un marchand par son nom normalisé. Retourne l'ID du point si trouvé."""
        if not self.client: return None
        try:
            # Utiliser l'ID déterministe directement plutôt que de faire une recherche
            point_id = self._get_merchant_point_id(normalized_name)
            
            # Vérifier que le point existe
            results = self.client.retrieve(
                collection_name=self.MERCHANTS_COLLECTION,
                ids=[point_id],
                with_payload=False,
                with_vectors=False
            )
            
            if results:
                return point_id
                
            # Si pas trouvé par ID direct, essayer une recherche par filtrage (fallback)
            filter_query = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="normalized_name", match=qmodels.MatchValue(value=normalized_name))
                ]
            )
            
            search_result = self.client.search(
                 collection_name=self.MERCHANTS_COLLECTION,
                 query_filter=filter_query,
                 query_vector=[0.0] * self.vector_size,  # Vecteur factice car on filtre seulement
                 limit=1,
                 with_payload=False,
                 with_vectors=False
            )
            
            if search_result:
                return search_result[0].id
                
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche du marchand '{normalized_name}': {e}", exc_info=True)
            return None

    # --- Méthodes de Stockage (Insights) ---

    async def batch_store_insights(self, insights_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot d'insights."""
        if not self.client or not insights_list:
            logger.info("batch_store_insights: Pas de client Qdrant ou liste vide")
            return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        logger.info(f"Début du batch_store_insights pour {len(insights_list)} insights")
        
        points_to_upsert = []
        failed_items = []
        processed_count = 0

        for insight in insights_list:
            processed_count += 1
            try:
                # Générer un ID consistant basé sur les propriétés uniques
                user_id = insight.get("user_id")
                period_type = insight.get("period_type")
                period_start = insight.get("period_start")
                category_id = insight.get("category_id")
                
                if not all([user_id, period_type, period_start]):
                    logger.warning(f"Données manquantes pour insight dans batch. Skipping.")
                    failed_items.append(f"user_{user_id}_{period_type}_{period_start}")
                    continue

                # Créer un ID déterministe pour les insights
                id_components = f"user_{user_id}_{period_type}_{period_start.isoformat() if isinstance(period_start, datetime) else period_start}"
                if category_id is not None:
                    id_components += f"_cat_{category_id}"
                
                point_id = str(uuid.uuid5(HARENA_NAMESPACE, id_components))

                category_name = insight.get("category_name", "Toutes catégories")
                period_start_str = period_start.strftime('%Y-%m-%d') if isinstance(period_start, datetime) else str(period_start)
                text_to_embed = f"Résumé {period_type} pour {category_name} démarrant le {period_start_str}"
                logger.debug(f"Génération d'embedding pour insight {id_components}: '{text_to_embed}'")
                embedding = await self.embedding_service.get_embedding(text_to_embed)

                payload = self._prepare_payload({
                    "user_id": user_id,
                    "category_id": category_id,
                    "category_name": category_name,
                    "period_type": period_type,
                    "period_start": period_start,
                    "period_end": insight.get("period_end"),
                    "aggregates": insight.get("aggregates"),
                    "kpi_data": insight.get("kpi_data"),
                    "generated_at": datetime.now(timezone.utc)
                })
                points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
                logger.debug(f"Insight pour {category_name} (période {period_type}) préparé pour upsert")
            except Exception as e:
                logger.error(f"Erreur lors de la préparation de l'insight pour le batch: {e}")
                failed_items.append(f"user_{user_id}_{period_type}_{period_start}")

        successful_count = 0
        if points_to_upsert:
            try:
                logger.info(f"Tentative d'upsert de {len(points_to_upsert)} points insights dans Qdrant")
                chunk_size = 50
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    logger.debug(f"Upsert du chunk d'insights {i//chunk_size + 1} (taille: {len(chunk)})")
                    self.client.upsert(
                        collection_name=self.INSIGHTS_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} insights terminé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des insights: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_items)
                successful_count = 0
            else:
                failed_count = len(failed_items)
        else:
            logger.warning("Aucun point insight prêt pour l'upsert")
            failed_count = len(failed_items)

        result = {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_items": failed_items[:10] if failed_items else []
        }
        
        logger.info(f"Résultat batch_store_insights: {result['status']}, {result['successful']}/{result['total']} réussis")
        return result

    # --- Méthodes de Stockage (Stocks) ---

    async def batch_store_stocks(self, stocks_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot d'actions/stocks."""
        if not self.client or not stocks_list:
            logger.info("batch_store_stocks: Pas de client Qdrant ou liste vide")
            return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        logger.info(f"Début du batch_store_stocks pour {len(stocks_list)} stocks")
        
        points_to_upsert = []
        failed_items = []
        processed_count = 0

        for stock in stocks_list:
            processed_count += 1
            try:
                user_id = stock.get("user_id")
                account_id = stock.get("account_id")
                bridge_stock_id = stock.get("bridge_stock_id")
                isin = stock.get("isin")
                ticker = stock.get("ticker")
                
                if user_id is None or account_id is None or not (isin or ticker or bridge_stock_id):
                    logger.warning(f"Données d'identification manquantes pour stock dans batch. Skipping.")
                    failed_items.append(f"user_{user_id}_acc_{account_id}_{ticker or isin or bridge_stock_id}")
                    continue

                # Créer un ID déterministe pour le stock
                id_components = f"user_{user_id}_account_{account_id}"
                if bridge_stock_id:
                    id_components += f"_stock_{bridge_stock_id}"
                elif isin:
                    id_components += f"_isin_{isin}"
                elif ticker:
                    id_components += f"_ticker_{ticker}"
                
                point_id = str(uuid.uuid5(HARENA_NAMESPACE, id_components))

                text_to_embed = f"{stock.get('label', '')} {ticker or ''} {isin or ''}"
                logger.debug(f"Génération d'embedding pour stock {id_components}: '{text_to_embed}'")
                embedding = await self.embedding_service.get_embedding(text_to_embed.strip())

                payload = self._prepare_payload({
                    "user_id": user_id,
                    "account_id": account_id,
                    "bridge_stock_id": bridge_stock_id,
                    "label": stock.get("label"),
                    "ticker": ticker,
                    "isin": isin,
                    "quantity": stock.get("quantity"),
                    "current_price": stock.get("current_price"),
                    "total_value": stock.get("total_value"),
                    "currency_code": stock.get("currency_code"),
                    "value_date": stock.get("value_date"),
                    "average_purchase_price": stock.get("average_purchase_price"),
                    "bridge_updated_at": stock.get("bridge_updated_at"),
                    "last_synced_at": datetime.now(timezone.utc)
                })
                points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
                logger.debug(f"Stock {ticker or isin or bridge_stock_id} préparé pour upsert")
            except Exception as e:
                logger.error(f"Erreur lors de la préparation du stock {ticker or isin} pour le batch: {e}")
                failed_items.append(f"user_{user_id}_acc_{account_id}_{ticker or isin}")

        successful_count = 0
        if points_to_upsert:
            try:
                logger.info(f"Tentative d'upsert de {len(points_to_upsert)} points stocks dans Qdrant")
                chunk_size = 100
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    logger.debug(f"Upsert du chunk de stocks {i//chunk_size + 1} (taille: {len(chunk)})")
                    self.client.upsert(
                        collection_name=self.STOCKS_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} stocks terminé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des stocks: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_items)
                successful_count = 0
            else:
                failed_count = len(failed_items)
        else:
            logger.warning("Aucun point stock prêt pour l'upsert")
            failed_count = len(failed_items)

        result = {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_items": failed_items[:10] if failed_items else []
        }
        
        logger.info(f"Résultat batch_store_stocks: {result['status']}, {result['successful']}/{result['total']} réussis")
        return result

    # --- Méthodes de gestion utilisateur ---

    async def initialize_user_storage(self, user_id: int) -> bool:
        """Initialise le stockage pour un utilisateur (marqueur)."""
        if not self.client: return False
        try:
            point_id_str = self._get_user_metadata_point_id(user_id)
            payload = {
                "user_id": user_id,
                "initialized_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.1",
                "status": "active"
            }
            self.client.upsert(
                collection_name=self.USER_METADATA_COLLECTION,
                points=[qmodels.PointStruct(id=point_id_str, payload=payload)],
                wait=True
            )
            logger.info(f"Stockage vectoriel initialisé (métadata) pour l'utilisateur {user_id} avec ID {point_id_str}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du stockage pour user {user_id}: {e}", exc_info=True)
            return False

    async def check_user_storage_initialized(self, user_id: int) -> bool:
        """Vérifie si le stockage pour un utilisateur est initialisé."""
        if not self.client: return False
        try:
            point_id_str = self._get_user_metadata_point_id(user_id)
            results = self.client.retrieve(
                collection_name=self.USER_METADATA_COLLECTION,
                ids=[point_id_str],
                with_payload=False,
                with_vectors=False
            )
            return len(results) > 0
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.debug(f"Point metadata {point_id_str} non trouvé pour user {user_id}. Initialisation non faite.")
                return False
            logger.error(f"Erreur Qdrant {e.status_code} lors de la vérification de l'initialisation pour user {user_id}: {e.content}")
            return False
        except Exception as e:
            logger.error(f"Erreur générale lors de la vérification de l'initialisation pour user {user_id}: {e}", exc_info=True)
            return False

    async def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """Récupère des statistiques simples pour un utilisateur."""
        if not self.client:
            return {"user_id": user_id, "error": "Qdrant client not available", "initialized": False}

        stats = {"user_id": user_id, "initialized": False}
        try:
            stats["initialized"] = await self.check_user_storage_initialized(user_id)
            user_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))])
            collections = {col.name for col in self.client.get_collections().collections}

            for collection_name in [
                self.TRANSACTIONS_COLLECTION, self.ACCOUNTS_COLLECTION,
                self.STOCKS_COLLECTION, self.INSIGHTS_COLLECTION
            ]:
                try:
                    if collection_name in collections:
                        count_response = self.client.count(
                            collection_name=collection_name,
                            count_filter=user_filter, exact=False
                        )
                        stats[f"{collection_name}_count"] = count_response.count
                    else:
                        stats[f"{collection_name}_count"] = 0
                except Exception as count_e:
                    logger.warning(f"Impossible de compter les points dans {collection_name} pour user {user_id}: {count_e}")
                    stats[f"{collection_name}_count"] = -1

            return stats
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques pour user {user_id}: {e}", exc_info=True)
            stats["error"] = str(e)
            return stats

    async def get_user_storage_metadata(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Récupère les métadonnées de stockage d'un utilisateur."""
        if not self.client: return None
        try:
            point_id_str = self._get_user_metadata_point_id(user_id)
            results = self.client.retrieve(
                collection_name=self.USER_METADATA_COLLECTION,
                ids=[point_id_str],
                with_payload=True
            )
            if results:
                return results[0].payload
            return None
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.debug(f"Métadonnées non trouvées pour user {user_id} (point ID: {point_id_str})")
            else:
                logger.error(f"Erreur Qdrant lors de la récupération des métadonnées pour user {user_id}: {e.status_code} - {e.content}")
            return None
        except Exception as e:
            logger.error(f"Erreur générale lors de la récupération des métadonnées pour user {user_id}: {e}", exc_info=True)
            return None

    async def update_user_storage_metadata(self, user_id: int, metadata_updates: Dict[str, Any]) -> bool:
        """Met à jour les métadonnées de stockage d'un utilisateur."""
        if not self.client: return False
        try:
            point_id_str = self._get_user_metadata_point_id(user_id)
            current_metadata = await self.get_user_storage_metadata(user_id)

            if not current_metadata:
                logger.warning(f"Métadonnées non trouvées pour user {user_id}. Initialisation avec les nouvelles données.")
                init_payload = {
                    "user_id": user_id,
                    "initialized_at": datetime.now(timezone.utc).isoformat(),
                    "status": "active",
                    **metadata_updates
                }
                self.client.upsert(
                    collection_name=self.USER_METADATA_COLLECTION,
                    points=[qmodels.PointStruct(id=point_id_str, payload=init_payload)],
                    wait=True
                )
                return True

            updated_payload = {**current_metadata, **metadata_updates, "updated_at": datetime.now(timezone.utc).isoformat()}
            self.client.set_payload(
                collection_name=self.USER_METADATA_COLLECTION,
                payload=updated_payload,
                points=[point_id_str],
                wait=True
            )
            logger.info(f"Métadonnées mises à jour pour l'utilisateur {user_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métadonnées pour user {user_id}: {e}", exc_info=True)
            return False

    async def delete_user_data(self, user_id: int) -> bool:
        """Supprime toutes les données associées à un utilisateur dans Qdrant."""
        if not self.client: return False
        logger.warning(f"Tentative de suppression de TOUTES les données Qdrant pour user_id={user_id}")
        success = True
        user_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))])
        collections = {col.name for col in self.client.get_collections().collections}

        collections_to_clean = [
            self.TRANSACTIONS_COLLECTION, self.ACCOUNTS_COLLECTION,
            self.INSIGHTS_COLLECTION, self.STOCKS_COLLECTION,
        ]

        for collection_name in collections_to_clean:
            try:
                if collection_name in collections:
                    count_before = self.client.count(collection_name=collection_name, count_filter=user_filter, exact=False).count
                    if count_before > 0:
                        logger.info(f"Suppression de ~{count_before} points de {collection_name} pour user {user_id}")
                        delete_result = self.client.delete(
                            collection_name=collection_name,
                            points_selector=qmodels.FilterSelector(filter=user_filter),
                            wait=True
                        )
                        logger.info(f"Points supprimés de {collection_name} pour user {user_id}. Résultat: {delete_result.status}")
                    else:
                        logger.info(f"Aucun point à supprimer dans {collection_name} pour user {user_id}")
                else:
                    logger.debug(f"Collection {collection_name} n'existe pas, suppression ignorée.")
            except Exception as e:
                logger.error(f"Erreur lors de la suppression des points de {collection_name} pour user {user_id}: {e}")
                success = False

        # Supprimer les métadonnées utilisateur
        try:
            if self.USER_METADATA_COLLECTION in collections:
                point_id_str = self._get_user_metadata_point_id(user_id)
                delete_result = self.client.delete(
                    collection_name=self.USER_METADATA_COLLECTION,
                    points_selector=qmodels.PointIdsList(points=[point_id_str]),
                    wait=True
                )
                logger.info(f"Tentative de suppression des métadonnées pour user {user_id}. Résultat: {delete_result.status}")
            else:
                logger.debug(f"Collection {self.USER_METADATA_COLLECTION} n'existe pas, suppression métadata ignorée.")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des métadonnées pour user {user_id}: {e}")
            success = False

        logger.warning(f"Suppression des données Qdrant pour user_id={user_id} terminée avec succès global: {success}")
        return success