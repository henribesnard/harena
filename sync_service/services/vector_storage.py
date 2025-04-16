
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
from datetime import datetime
from uuid import UUID, uuid4
import hashlib # Pour générer des ID marchands potentiels

# Import des dépendances Qdrant
from qdrant_client import QdrantClient, models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# Import du service d'embedding local
from sync_service.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStorageService:
    """Service pour le stockage et la recherche vectorielle des données Harena."""

    # Noms des collections
    TRANSACTIONS_COLLECTION = "transactions"
    ACCOUNTS_COLLECTION = "accounts"
    CATEGORIES_COLLECTION = "categories"
    MERCHANTS_COLLECTION = "merchants"
    INSIGHTS_COLLECTION = "insights"
    STOCKS_COLLECTION = "stocks"
    USER_METADATA_COLLECTION = "user_metadata" # Collection séparée pour les métadonnées

    # Configuration par défaut
    DEFAULT_VECTOR_SIZE = 1536  # Taille pour text-embedding-3-small
    DEFAULT_DISTANCE = qmodels.Distance.COSINE
    DEFAULT_INDEXING_THRESHOLD = 10000 # Indexer après ce nombre de points

    def __init__(self):
        """Initialise le service de stockage vectoriel."""
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_size = self.DEFAULT_VECTOR_SIZE

        # Initialiser le client Qdrant
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=60.0  # Augmenter le timeout
            )
            # Test de connexion simple
            self.client.get_collections()
            logger.info("Client Qdrant connecté avec succès à %s", self.qdrant_url)
        except Exception as e:
            logger.error("Erreur lors de la connexion à Qdrant (%s): %s", self.qdrant_url, e, exc_info=True)
            # Vous pourriez vouloir lever une exception ici ou avoir un état 'non initialisé'
            self.client = None # Marquer le client comme non disponible

        # Service d'embedding
        self.embedding_service = EmbeddingService()

        # Initialiser les collections si le client est disponible
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

            # Créer les index de payload
            for field_name, field_type in indexes:
                try:
                    # Déterminer le type de schéma Qdrant
                    if field_type == "keyword":
                        schema_type = qmodels.PayloadSchemaType.KEYWORD
                    elif field_type == "integer":
                        schema_type = qmodels.PayloadSchemaType.INTEGER
                    elif field_type == "float":
                        schema_type = qmodels.PayloadSchemaType.FLOAT
                    elif field_type == "boolean":
                        schema_type = qmodels.PayloadSchemaType.BOOL
                    elif field_type == "datetime":
                         schema_type = qmodels.PayloadSchemaType.DATETIME
                    elif field_type == "text":
                         schema_type = qmodels.TextIndexParams(
                             type="text",
                             tokenizer=qmodels.TokenizerType.WORD,
                             min_token_len=2,
                             max_token_len=15,
                             lowercase=True
                         )
                    else:
                        logger.warning(f"Type d'index non supporté '{field_type}' pour {field_name} dans {collection_name}. Ignoré.")
                        continue

                    # Créer l'index
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
            # Gérer le cas où la collection existe déjà (code 400 ou 409 typiquement)
            if e.status_code == 400 and "already exists" in str(e.content):
                logger.info(f"Collection Qdrant existante: {collection_name}")
            elif e.status_code == 409: # Conflit, peut aussi signifier existe déjà
                 logger.info(f"Collection Qdrant existante (conflit): {collection_name}")
            else:
                logger.error(f"Erreur inattendue lors de la création de la collection {collection_name}: {e.status_code} {e.content}")
                raise # Renvoyer l'erreur si ce n'est pas une erreur "existe déjà"
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

            # --- Collection Transactions ---
            if self.TRANSACTIONS_COLLECTION not in existing_collections:
                indexes = [
                    ("user_id", "integer"), # Assurez-vous que user_id est bien un entier
                    ("account_id", "integer"), # Assurez-vous que account_id est bien un entier
                    ("bridge_transaction_id", "keyword"), # ID Bridge comme keyword
                    ("transaction_date", "datetime"),
                    ("category_id", "integer"), # ID catégorie comme entier
                    ("merchant_id", "keyword"), # ID du marchand (si utilisé)
                    ("amount", "float"),
                    ("operation_type", "keyword"),
                    ("is_recurring", "boolean"),
                    ("currency_code", "keyword"),
                ]
                self._create_collection_with_indexes(self.TRANSACTIONS_COLLECTION, indexes)
            else:
                logger.info(f"Collection Qdrant existante: {self.TRANSACTIONS_COLLECTION}")

            # --- Collection Accounts ---
            if self.ACCOUNTS_COLLECTION not in existing_collections:
                indexes = [
                    ("user_id", "integer"),
                    ("item_id", "integer"),
                    ("bridge_account_id", "integer"),
                    ("type", "keyword"),
                    ("currency_code", "keyword"),
                    ("provider_id", "integer"),
                    ("pro", "boolean"),
                    ("balance", "float"), # Pour requêtes de plage de solde
                    ("iban", "keyword"), # Pour recherche exacte
                ]
                self._create_collection_with_indexes(self.ACCOUNTS_COLLECTION, indexes)
            else:
                 logger.info(f"Collection Qdrant existante: {self.ACCOUNTS_COLLECTION}")

            # --- Collection Categories ---
            if self.CATEGORIES_COLLECTION not in existing_collections:
                indexes = [
                    ("bridge_category_id", "integer"),
                    ("parent_id", "integer"),
                    ("name", "keyword"), # Pour recherche exacte par nom
                ]
                self._create_collection_with_indexes(self.CATEGORIES_COLLECTION, indexes)
            else:
                 logger.info(f"Collection Qdrant existante: {self.CATEGORIES_COLLECTION}")

            # --- Collection Merchants ---
            if self.MERCHANTS_COLLECTION not in existing_collections:
                indexes = [
                    # Attention: Si les marchands sont globaux, pas de user_id ici.
                    # Si spécifiques à l'enrichissement par user, ajouter: ("user_id", "integer"),
                    ("normalized_name", "keyword"),
                    ("category_id", "integer"),
                    ("display_name", "text"), # Pour recherche textuelle
                ]
                self._create_collection_with_indexes(self.MERCHANTS_COLLECTION, indexes)
            else:
                 logger.info(f"Collection Qdrant existante: {self.MERCHANTS_COLLECTION}")

            # --- Collection Insights ---
            if self.INSIGHTS_COLLECTION not in existing_collections:
                indexes = [
                    ("user_id", "integer"),
                    ("category_id", "integer"),
                    ("period_type", "keyword"), # 'monthly', 'global', etc.
                    ("period_start", "datetime"),
                    ("period_end", "datetime"),
                ]
                self._create_collection_with_indexes(self.INSIGHTS_COLLECTION, indexes)
            else:
                 logger.info(f"Collection Qdrant existante: {self.INSIGHTS_COLLECTION}")

            # --- Collection Stocks ---
            if self.STOCKS_COLLECTION not in existing_collections:
                indexes = [
                    ("user_id", "integer"),
                    ("account_id", "integer"),
                    ("bridge_stock_id", "integer"),
                    ("ticker", "keyword"),
                    ("isin", "keyword"),
                    ("currency_code", "keyword"),
                    ("value_date", "datetime"),
                ]
                self._create_collection_with_indexes(self.STOCKS_COLLECTION, indexes)
            else:
                 logger.info(f"Collection Qdrant existante: {self.STOCKS_COLLECTION}")

            # --- Collection User Metadata ---
            if self.USER_METADATA_COLLECTION not in existing_collections:
                indexes = [
                    ("user_id", "integer"),
                    ("status", "keyword"),
                ]
                # Pas besoin de vecteurs pour les métadonnées pures
                try:
                    self.client.create_collection(collection_name=self.USER_METADATA_COLLECTION)
                    logger.info(f"Collection Qdrant créée: {self.USER_METADATA_COLLECTION}")
                    for field, ftype in indexes:
                        self.client.create_payload_index(
                             collection_name=self.USER_METADATA_COLLECTION,
                             field_name=field,
                             field_schema=qmodels.PayloadSchemaType.KEYWORD if ftype == 'keyword' else qmodels.PayloadSchemaType.INTEGER
                        )
                    logger.info(f"Index créés pour la collection: {self.USER_METADATA_COLLECTION}")
                except Exception as meta_e:
                     logger.error(f"Erreur lors de la création de la collection {self.USER_METADATA_COLLECTION}: {meta_e}")
            else:
                logger.info(f"Collection Qdrant existante: {self.USER_METADATA_COLLECTION}")


        except Exception as e:
            logger.error(f"Erreur générale lors de la vérification/création des collections: {e}", exc_info=True)
            # Selon la criticité, vous pourriez vouloir arrêter l'application ici.

    def _prepare_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoie et prépare le payload pour Qdrant (ex: dates en ISO)."""
        payload = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
            # Ajouter d'autres conversions si nécessaire (ex: Enum en string)
            elif value is not None: # Exclure les clés avec valeur None
                payload[key] = value
        return payload

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

            # Utiliser un ID de point basé sur l'ID Bridge et l'User ID pour l'idempotence
            point_id = hashlib.sha256(f"user_{user_id}_tx_{transaction_id_str}".encode()).hexdigest()

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
                "merchant_id": transaction.get("merchant_id"), # Si l'enrichissement est fait
                "created_at": datetime.now(datetime.timezone.utc),
                "updated_at": datetime.now(datetime.timezone.utc)
            })

            self.client.upsert(
                collection_name=self.TRANSACTIONS_COLLECTION,
                points=[qmodels.PointStruct(id=point_id, vector=embedding, payload=payload)],
                wait=False # Peut être mis à True pour garantir l'écriture avant retour
            )
            logger.debug(f"Transaction upserted: {point_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du stockage de la transaction {transaction.get('bridge_transaction_id')}: {e}", exc_info=True)
            return False

    async def batch_store_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot de transactions."""
        if not self.client or not transactions:
            return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        points_to_upsert = []
        failed_ids = []
        processed_count = 0

        for transaction in transactions:
            processed_count += 1
            try:
                transaction_id_str = str(transaction.get("bridge_transaction_id"))
                user_id = transaction.get("user_id")
                if not transaction_id_str or user_id is None:
                    logger.warning(f"ID transaction Bridge ou User ID manquant dans le lot. Skipping.")
                    failed_ids.append(transaction.get("bridge_transaction_id", "unknown"))
                    continue

                point_id = hashlib.sha256(f"user_{user_id}_tx_{transaction_id_str}".encode()).hexdigest()
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
                    "created_at": datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.now(datetime.timezone.utc)
                })

                points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))

            except Exception as e:
                logger.error(f"Erreur lors de la préparation de la transaction {transaction.get('bridge_transaction_id')} pour le batch: {e}")
                failed_ids.append(transaction.get("bridge_transaction_id", "unknown"))

        successful_count = 0
        if points_to_upsert:
            try:
                # Upsert par lots plus petits pour éviter les timeouts / grosses requêtes
                chunk_size = 100
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    self.client.upsert(
                        collection_name=self.TRANSACTIONS_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} transactions terminé.")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des transactions: {e}", exc_info=True)
                # Dans ce cas, difficile de savoir lesquels ont échoué, on considère le batch entier comme échoué
                failed_count = len(points_to_upsert) - successful_count + len(failed_ids)
                successful_count = 0 # Reset car le batch a échoué
            else:
                 failed_count = len(failed_ids)
        else:
            failed_count = len(failed_ids)


        return {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_ids": failed_ids[:10] # Limiter la taille des IDs échoués retournés
        }

    async def check_transaction_exists(self, bridge_transaction_id: str, user_id: int) -> bool:
        """Vérifie si une transaction existe déjà pour cet utilisateur."""
        if not self.client: return False
        try:
            filter_query = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="bridge_transaction_id", match=qmodels.MatchValue(value=str(bridge_transaction_id))),
                    qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))
                ]
            )
            results = self.client.scroll(
                collection_name=self.TRANSACTIONS_COLLECTION, scroll_filter=filter_query, limit=1, with_payload=False, with_vectors=False
            )
            return len(results[0]) > 0
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'existence de la transaction {bridge_transaction_id} pour user {user_id}: {e}", exc_info=True)
            return False # Prudence en cas d'erreur

    # --- Méthodes de Stockage (Comptes) ---

    async def store_account(self, account_data: Dict[str, Any]) -> bool:
        """Stocke un compte unique."""
        if not self.client: return False
        # Implémentation similaire à store_transaction
        try:
            account_id = account_data.get("bridge_account_id")
            user_id = account_data.get("user_id")
            if account_id is None or user_id is None:
                 logger.error("ID compte Bridge ou User ID manquant.")
                 return False

            point_id = f"user_{user_id}_account_{account_id}" # ID déterministe
            text_to_embed = f"{account_data.get('name', '')} {account_data.get('type', '')}"
            embedding = await self.embedding_service.get_embedding(text_to_embed.strip())

            # Préparer le payload spécifique aux comptes
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
                "loan_details": account_data.get("loan_details"), # Peut être un dict/JSON
                "bridge_updated_at": account_data.get("bridge_updated_at"), # Date de Bridge
                "last_synced_at": datetime.now(datetime.timezone.utc) # Date de notre synchro
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
            return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        # Implémentation similaire à batch_store_transactions
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

                point_id = f"user_{user_id}_account_{account_id}"
                text_to_embed = f"{account.get('name', '')} {account.get('type', '')}"
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
                    "last_synced_at": datetime.now(datetime.timezone.utc)
                })
                points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
            except Exception as e:
                logger.error(f"Erreur lors de la préparation du compte {account.get('bridge_account_id')} pour le batch: {e}")
                failed_ids.append(account.get("bridge_account_id", "unknown"))

        # Logique de batch upsert similaire à celle des transactions...
        successful_count = 0
        if points_to_upsert:
            try:
                chunk_size = 100
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    self.client.upsert(
                        collection_name=self.ACCOUNTS_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} comptes terminé.")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des comptes: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_ids)
                successful_count = 0
            else:
                 failed_count = len(failed_ids)
        else:
            failed_count = len(failed_ids)

        return {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_ids": failed_ids[:10]
        }

    async def check_account_exists(self, bridge_account_id: int, user_id: int) -> bool:
        """Vérifie si un compte existe déjà pour cet utilisateur."""
        if not self.client: return False
        try:
            point_id = f"user_{user_id}_account_{bridge_account_id}"
            # Utiliser fetch qui est plus direct pour vérifier l'existence par ID
            results = self.client.retrieve(
                collection_name=self.ACCOUNTS_COLLECTION,
                ids=[point_id],
                with_payload=False,
                with_vectors=False
            )
            return len(results) > 0
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'existence du compte {bridge_account_id} pour user {user_id}: {e}", exc_info=True)
            return False

    async def update_account_payload(self, bridge_account_id: int, user_id: int, updates: Dict[str, Any]) -> bool:
        """Met à jour le payload d'un compte existant (ex: solde)."""
        if not self.client: return False
        try:
            point_id = f"user_{user_id}_account_{bridge_account_id}"
            # Préparer le payload de mise à jour
            update_payload = self._prepare_payload(updates)
            update_payload["last_synced_at"] = datetime.now(datetime.timezone.utc).isoformat() # Toujours màj la date

            # Utiliser set_payload pour ne mettre à jour que certains champs
            self.client.set_payload(
                collection_name=self.ACCOUNTS_COLLECTION,
                payload=update_payload,
                points=[point_id],
                wait=False
            )
            logger.info(f"Payload du compte {point_id} mis à jour avec: {list(update_payload.keys())}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du payload du compte {bridge_account_id}: {e}", exc_info=True)
            return False

    # --- Méthodes de Stockage (Catégories) ---

    async def store_category(self, category_data: Dict[str, Any]) -> bool:
        """Stocke une catégorie unique. Les catégories sont souvent globales."""
        if not self.client: return False
        try:
            category_id = category_data.get("bridge_category_id") # Utiliser un champ spécifique
            if category_id is None:
                 logger.error("ID catégorie Bridge manquant.")
                 return False

            point_id = f"category_{category_id}" # ID déterministe global
            text_to_embed = category_data.get("name", "")
            embedding = await self.embedding_service.get_embedding(text_to_embed)

            payload = self._prepare_payload({
                "bridge_category_id": category_id,
                "name": text_to_embed,
                "parent_id": category_data.get("parent_id"),
                # Ajouter d'autres métadonnées si fournies par Bridge
                "last_synced_at": datetime.now(datetime.timezone.utc)
            })

            self.client.upsert(
                collection_name=self.CATEGORIES_COLLECTION,
                points=[qmodels.PointStruct(id=point_id, vector=embedding, payload=payload)],
                wait=False
            )
            logger.debug(f"Catégorie upserted: {point_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du stockage de la catégorie {category_id}: {e}", exc_info=True)
            return False

    async def batch_store_categories(self, categories_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot de catégories."""
        if not self.client or not categories_list:
            return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        # Implémentation similaire à batch_store_transactions
        points_to_upsert = []
        failed_ids = []
        processed_count = 0

        for category in categories_list:
             processed_count += 1
             try:
                 category_id = category.get("bridge_category_id")
                 if category_id is None:
                     logger.warning(f"ID catégorie Bridge manquant dans le lot. Skipping.")
                     failed_ids.append(category.get("bridge_category_id", "unknown"))
                     continue

                 point_id = f"category_{category_id}"
                 text_to_embed = category.get("name", "")
                 embedding = await self.embedding_service.get_embedding(text_to_embed)

                 payload = self._prepare_payload({
                     "bridge_category_id": category_id,
                     "name": text_to_embed,
                     "parent_id": category.get("parent_id"),
                     "last_synced_at": datetime.now(datetime.timezone.utc)
                 })
                 points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
             except Exception as e:
                 logger.error(f"Erreur lors de la préparation de la catégorie {category.get('bridge_category_id')} pour le batch: {e}")
                 failed_ids.append(category.get("bridge_category_id", "unknown"))

        # Logique de batch upsert...
        successful_count = 0
        if points_to_upsert:
            try:
                chunk_size = 100
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    self.client.upsert(
                        collection_name=self.CATEGORIES_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} catégories terminé.")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des catégories: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_ids)
                successful_count = 0
            else:
                 failed_count = len(failed_ids)
        else:
            failed_count = len(failed_ids)

        return {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_ids": failed_ids[:10]
        }

    async def check_category_exists(self, bridge_category_id: int) -> bool:
        """Vérifie si une catégorie existe déjà."""
        if not self.client: return False
        try:
            point_id = f"category_{bridge_category_id}"
            results = self.client.retrieve(
                collection_name=self.CATEGORIES_COLLECTION,
                ids=[point_id],
                with_payload=False,
                with_vectors=False
            )
            return len(results) > 0
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'existence de la catégorie {bridge_category_id}: {e}", exc_info=True)
            return False

    # --- Méthodes de Stockage (Marchands) ---
    # NOTE: La logique exacte dépendra de votre stratégie d'enrichissement

    async def store_merchant(self, merchant_data: Dict[str, Any]) -> Optional[str]:
        """Stocke un marchand unique. Retourne l'ID du point si succès."""
        if not self.client: return None
        try:
            # Générer un ID unique pour le marchand (UUID ou basé sur hash normalisé)
            # Utiliser UUID pour éviter collisions et simplifier la vérification d'existence
            point_id = str(uuid4())
            # Alternative :
            # normalized_name = merchant_data.get("normalized_name", "").lower()
            # if not normalized_name: return None
            # point_id = f"merchant_{hashlib.sha256(normalized_name.encode()).hexdigest()}"

            text_to_embed = f"{merchant_data.get('display_name', '')} {merchant_data.get('normalized_name', '')}"
            embedding = await self.embedding_service.get_embedding(text_to_embed.strip())

            payload = self._prepare_payload({
                "normalized_name": merchant_data.get("normalized_name"),
                "display_name": merchant_data.get("display_name"),
                "category_id": merchant_data.get("category_id"), # Catégorie associée
                # "user_id": merchant_data.get("user_id"), # Si spécifique à l'utilisateur
                "source": merchant_data.get("source", "inferred"), # D'où vient ce marchand?
                "created_at": datetime.now(datetime.timezone.utc),
                "updated_at": datetime.now(datetime.timezone.utc)
            })

            self.client.upsert(
                collection_name=self.MERCHANTS_COLLECTION,
                points=[qmodels.PointStruct(id=point_id, vector=embedding, payload=payload)],
                wait=False
            )
            logger.debug(f"Marchand upserted: {point_id} ({payload.get('normalized_name')})")
            return point_id # Retourne l'ID généré
        except Exception as e:
            logger.error(f"Erreur lors du stockage du marchand {merchant_data.get('normalized_name')}: {e}", exc_info=True)
            return None

    async def batch_store_merchants(self, merchants_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot de marchands."""
        # Logique similaire aux autres batch stores, en générant des UUIDs pour les points
        if not self.client or not merchants_list:
             return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        points_to_upsert = []
        failed_items = []
        processed_count = 0

        for merchant in merchants_list:
            processed_count += 1
            try:
                point_id = str(uuid4()) # Utiliser UUID
                text_to_embed = f"{merchant.get('display_name', '')} {merchant.get('normalized_name', '')}"
                embedding = await self.embedding_service.get_embedding(text_to_embed.strip())

                payload = self._prepare_payload({
                    "normalized_name": merchant.get("normalized_name"),
                    "display_name": merchant.get("display_name"),
                    "category_id": merchant.get("category_id"),
                    "source": merchant.get("source", "inferred"),
                    "created_at": datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.now(datetime.timezone.utc)
                })
                points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
            except Exception as e:
                logger.error(f"Erreur lors de la préparation du marchand {merchant.get('normalized_name')} pour le batch: {e}")
                failed_items.append(merchant.get('normalized_name', "unknown"))

        # Logique de batch upsert...
        successful_count = 0
        if points_to_upsert:
            try:
                chunk_size = 100
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    self.client.upsert(
                        collection_name=self.MERCHANTS_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} marchands terminé.")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des marchands: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_items)
                successful_count = 0
            else:
                 failed_count = len(failed_items)
        else:
            failed_count = len(failed_items)

        return {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_items": failed_items[:10]
        }

    async def find_merchant(self, normalized_name: str) -> Optional[str]:
        """Trouve un marchand par son nom normalisé. Retourne l'ID du point si trouvé."""
        if not self.client: return None
        try:
            filter_query = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="normalized_name", match=qmodels.MatchValue(value=normalized_name))
                ]
            )
            results = self.client.scroll(
                collection_name=self.MERCHANTS_COLLECTION,
                scroll_filter=filter_query,
                limit=1,
                with_payload=False, # Juste besoin de l'ID
                with_vectors=False
            )
            if results and results[0]:
                return results[0][0].id
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la recherche du marchand '{normalized_name}': {e}", exc_info=True)
            return None


    # --- Méthodes de Stockage (Insights) ---

    async def store_insight(self, insight_data: Dict[str, Any]) -> bool:
        """Stocke un insight unique."""
        if not self.client: return False
        try:
            # Générer un ID (UUID ou basé sur user/cat/period)
            point_id = str(uuid4())
            user_id = insight_data.get("user_id")
            category_id = insight_data.get("category_id")
            period_type = insight_data.get("period_type")
            period_start = insight_data.get("period_start")

            if not all([user_id, period_type, period_start]):
                 logger.error("Données manquantes pour l'insight (user, type, start).")
                 return False
            # Alternative ID : point_id = f"user_{user_id}_insight_{category_id or 'global'}_{period_type}_{period_start.strftime('%Y%m%d')}"

            # Créer un texte descriptif pour l'embedding
            category_name = insight_data.get("category_name", "Toutes catégories") # Assumer qu'on a le nom
            text_to_embed = f"Résumé {period_type} pour {category_name} démarrant le {period_start.strftime('%Y-%m-%d')}"
            embedding = await self.embedding_service.get_embedding(text_to_embed)

            payload = self._prepare_payload({
                "user_id": user_id,
                "category_id": category_id,
                "category_name": category_name,
                "period_type": period_type,
                "period_start": period_start,
                "period_end": insight_data.get("period_end"),
                "aggregates": insight_data.get("aggregates"), # Dict des agrégats (sum, avg, etc.)
                "kpi_data": insight_data.get("kpi_data"), # Autres KPIs spécifiques
                "generated_at": datetime.now(datetime.timezone.utc)
            })

            self.client.upsert(
                collection_name=self.INSIGHTS_COLLECTION,
                points=[qmodels.PointStruct(id=point_id, vector=embedding, payload=payload)],
                wait=False
            )
            logger.debug(f"Insight upserted: {point_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du stockage de l'insight: {e}", exc_info=True)
            return False

    async def batch_store_insights(self, insights_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot d'insights."""
        # Logique similaire aux autres batch stores, en générant des UUIDs
        if not self.client or not insights_list:
             return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        points_to_upsert = []
        failed_items = []
        processed_count = 0

        for insight in insights_list:
            processed_count += 1
            try:
                point_id = str(uuid4())
                user_id = insight.get("user_id")
                period_type = insight.get("period_type")
                period_start = insight.get("period_start")
                if not all([user_id, period_type, period_start]):
                    logger.warning(f"Données manquantes pour insight dans batch. Skipping.")
                    failed_items.append(f"user_{user_id}_{period_type}_{period_start}")
                    continue

                category_name = insight.get("category_name", "Toutes catégories")
                text_to_embed = f"Résumé {period_type} pour {category_name} démarrant le {period_start.strftime('%Y-%m-%d')}"
                embedding = await self.embedding_service.get_embedding(text_to_embed)

                payload = self._prepare_payload({
                    "user_id": user_id,
                    "category_id": insight.get("category_id"),
                    "category_name": category_name,
                    "period_type": period_type,
                    "period_start": period_start,
                    "period_end": insight.get("period_end"),
                    "aggregates": insight.get("aggregates"),
                    "kpi_data": insight.get("kpi_data"),
                    "generated_at": datetime.now(datetime.timezone.utc)
                })
                points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
            except Exception as e:
                logger.error(f"Erreur lors de la préparation de l'insight pour le batch: {e}")
                failed_items.append(f"user_{user_id}_{period_type}_{period_start}")

        # Logique de batch upsert...
        successful_count = 0
        if points_to_upsert:
            try:
                chunk_size = 50 # Plus petit pour insights potentiellement plus gros
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    self.client.upsert(
                        collection_name=self.INSIGHTS_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} insights terminé.")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des insights: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_items)
                successful_count = 0
            else:
                 failed_count = len(failed_items)
        else:
            failed_count = len(failed_items)

        return {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_items": failed_items[:10]
        }

    # --- Méthodes de Stockage (Stocks) ---

    async def store_stock(self, stock_data: Dict[str, Any]) -> bool:
        """Stocke une action/stock unique."""
        if not self.client: return False
        try:
            stock_id = stock_data.get("bridge_stock_id") # S'il y a un ID bridge
            user_id = stock_data.get("user_id")
            account_id = stock_data.get("account_id")
            isin = stock_data.get("isin")
            ticker = stock_data.get("ticker")

            if user_id is None or account_id is None or not (isin or ticker):
                 logger.error("Données manquantes pour stock (user, account, isin/ticker).")
                 return False

            # Générer ID (UUID ou basé sur user/account/isin_ticker)
            point_id = str(uuid4())
            # Alternative ID: point_id = f"user_{user_id}_stock_{account_id}_{isin or ticker}"

            text_to_embed = f"{stock_data.get('label', '')} {ticker or ''} {isin or ''}"
            embedding = await self.embedding_service.get_embedding(text_to_embed.strip())

            payload = self._prepare_payload({
                "user_id": user_id,
                "account_id": account_id,
                "bridge_stock_id": stock_id,
                "label": stock_data.get("label"),
                "ticker": ticker,
                "isin": isin,
                "quantity": stock_data.get("quantity"),
                "current_price": stock_data.get("current_price"),
                "total_value": stock_data.get("total_value"),
                "currency_code": stock_data.get("currency_code"),
                "value_date": stock_data.get("value_date"),
                "average_purchase_price": stock_data.get("average_purchase_price"),
                "bridge_updated_at": stock_data.get("bridge_updated_at"), # Date de Bridge
                "last_synced_at": datetime.now(datetime.timezone.utc)
            })

            self.client.upsert(
                collection_name=self.STOCKS_COLLECTION,
                points=[qmodels.PointStruct(id=point_id, vector=embedding, payload=payload)],
                wait=False
            )
            logger.debug(f"Stock upserted: {point_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du stockage du stock {ticker or isin}: {e}", exc_info=True)
            return False

    async def batch_store_stocks(self, stocks_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stocke un lot d'actions/stocks."""
        # Logique similaire aux autres batch stores, en générant des UUIDs
        if not self.client or not stocks_list:
             return {"status": "noop", "total": 0, "successful": 0, "failed": 0}

        points_to_upsert = []
        failed_items = []
        processed_count = 0

        for stock in stocks_list:
            processed_count += 1
            try:
                point_id = str(uuid4())
                user_id = stock.get("user_id")
                account_id = stock.get("account_id")
                isin = stock.get("isin")
                ticker = stock.get("ticker")
                if user_id is None or account_id is None or not (isin or ticker):
                    logger.warning(f"Données manquantes pour stock dans batch. Skipping.")
                    failed_items.append(f"user_{user_id}_acc_{account_id}_{ticker or isin}")
                    continue

                text_to_embed = f"{stock.get('label', '')} {ticker or ''} {isin or ''}"
                embedding = await self.embedding_service.get_embedding(text_to_embed.strip())

                payload = self._prepare_payload({
                    "user_id": user_id,
                    "account_id": account_id,
                    "bridge_stock_id": stock.get("bridge_stock_id"),
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
                    "last_synced_at": datetime.now(datetime.timezone.utc)
                })
                points_to_upsert.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
            except Exception as e:
                logger.error(f"Erreur lors de la préparation du stock {ticker or isin} pour le batch: {e}")
                failed_items.append(f"user_{user_id}_acc_{account_id}_{ticker or isin}")

        # Logique de batch upsert...
        successful_count = 0
        if points_to_upsert:
            try:
                chunk_size = 100
                for i in range(0, len(points_to_upsert), chunk_size):
                    chunk = points_to_upsert[i:i + chunk_size]
                    self.client.upsert(
                        collection_name=self.STOCKS_COLLECTION,
                        points=chunk,
                        wait=False
                    )
                    successful_count += len(chunk)
                logger.info(f"Batch upsert de {successful_count} stocks terminé.")
            except Exception as e:
                logger.error(f"Erreur lors du batch upsert des stocks: {e}", exc_info=True)
                failed_count = len(points_to_upsert) - successful_count + len(failed_items)
                successful_count = 0
            else:
                 failed_count = len(failed_items)
        else:
            failed_count = len(failed_items)

        return {
            "status": "success" if failed_count == 0 else ("partial" if successful_count > 0 else "error"),
            "total": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "failed_items": failed_items[:10]
        }

    async def check_stock_exists(self, user_id: int, account_id: int, isin: Optional[str] = None, ticker: Optional[str] = None) -> bool:
        """Vérifie si un stock existe déjà pour cet utilisateur/compte."""
        if not self.client or not (isin or ticker): return False
        try:
            filters = [
                qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)),
                qmodels.FieldCondition(key="account_id", match=qmodels.MatchValue(value=account_id)),
            ]
            if isin:
                filters.append(qmodels.FieldCondition(key="isin", match=qmodels.MatchValue(value=isin)))
            if ticker:
                 filters.append(qmodels.FieldCondition(key="ticker", match=qmodels.MatchValue(value=ticker)))

            filter_query = qmodels.Filter(must=filters)
            results = self.client.scroll(
                collection_name=self.STOCKS_COLLECTION, scroll_filter=filter_query, limit=1, with_payload=False, with_vectors=False
            )
            return len(results[0]) > 0
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'existence du stock {ticker or isin} pour user {user_id}, account {account_id}: {e}", exc_info=True)
            return False

    # --- Méthodes de gestion utilisateur ---

    async def initialize_user_storage(self, user_id: int) -> bool:
        """Initialise le stockage pour un utilisateur (marqueur)."""
        if not self.client: return False
        try:
            point_id = f"user_meta_{user_id}"
            payload = {
                "user_id": user_id,
                "initialized_at": datetime.now(datetime.timezone.utc).isoformat(),
                "version": "1.1", # Version de la structure de données user
                "status": "active"
            }
            # Upsert sans vecteur car c'est juste un marqueur
            self.client.upsert(
                collection_name=self.USER_METADATA_COLLECTION,
                points=[qmodels.PointStruct(id=point_id, payload=payload)], # Pas de vecteur
                wait=True # Attendre confirmation pour la métadonnée
            )
            logger.info(f"Stockage vectoriel initialisé (métadata) pour l'utilisateur {user_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du stockage pour user {user_id}: {e}", exc_info=True)
            return False

    async def check_user_storage_initialized(self, user_id: int) -> bool:
        """Vérifie si le stockage pour un utilisateur est initialisé."""
        if not self.client: return False
        try:
            point_id = f"user_meta_{user_id}"
            results = self.client.retrieve(
                collection_name=self.USER_METADATA_COLLECTION,
                ids=[point_id],
                with_payload=False,
                with_vectors=False
            )
            return len(results) > 0
        except Exception as e:
            # Gérer l'erreur si la collection n'existe pas encore ? Non, _ensure_collection devrait le faire.
            logger.error(f"Erreur lors de la vérification de l'initialisation pour user {user_id}: {e}")
            return False

    async def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """Récupère des statistiques simples pour un utilisateur."""
        if not self.client:
            return {"user_id": user_id, "error": "Qdrant client not available", "initialized": False}

        stats = {"user_id": user_id, "initialized": False}
        try:
            stats["initialized"] = await self.check_user_storage_initialized(user_id)

            user_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))])

            for collection_name in [
                self.TRANSACTIONS_COLLECTION,
                self.ACCOUNTS_COLLECTION,
                self.STOCKS_COLLECTION, # Ajouter d'autres si pertinent
                self.INSIGHTS_COLLECTION
            ]:
                try:
                    count_response = self.client.count(
                        collection_name=collection_name,
                        count_filter=user_filter,
                        exact=False # Utiliser estimation pour performance
                    )
                    stats[f"{collection_name}_count"] = count_response.count
                except Exception as count_e:
                    # La collection n'existe peut-être pas encore ou erreur
                    logger.warning(f"Impossible de compter les points dans {collection_name} pour user {user_id}: {count_e}")
                    stats[f"{collection_name}_count"] = 0

            return stats
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques pour user {user_id}: {e}", exc_info=True)
            stats["error"] = str(e)
            return stats

    async def get_user_storage_metadata(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Récupère les métadonnées de stockage d'un utilisateur."""
        if not self.client: return None
        try:
            point_id = f"user_meta_{user_id}"
            results = self.client.retrieve(
                collection_name=self.USER_METADATA_COLLECTION,
                ids=[point_id],
                with_payload=True
            )
            if results:
                return results[0].payload
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métadonnées pour user {user_id}: {e}", exc_info=True)
            return None

    async def update_user_storage_metadata(self, user_id: int, metadata_updates: Dict[str, Any]) -> bool:
        """Met à jour les métadonnées de stockage d'un utilisateur."""
        if not self.client: return False
        try:
            point_id = f"user_meta_{user_id}"
            current_metadata = await self.get_user_storage_metadata(user_id)

            if not current_metadata:
                logger.warning(f"Métadonnées non trouvées pour user {user_id}, tentative d'initialisation.")
                # Initialiser avec les updates si non trouvé
                init_payload = {
                    "user_id": user_id,
                    "initialized_at": datetime.now(datetime.timezone.utc).isoformat(),
                    "status": "active",
                    **metadata_updates # Ajouter les updates
                }
                self.client.upsert(
                    collection_name=self.USER_METADATA_COLLECTION,
                    points=[qmodels.PointStruct(id=point_id, payload=init_payload)],
                    wait=True
                )
                return True

            # Mettre à jour le payload existant
            updated_payload = {**current_metadata, **metadata_updates, "updated_at": datetime.now(datetime.timezone.utc).isoformat()}

            self.client.set_payload(
                collection_name=self.USER_METADATA_COLLECTION,
                payload=updated_payload,
                points=[point_id],
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

        # Collections contenant potentiellement des données utilisateur
        collections_to_clean = [
            self.TRANSACTIONS_COLLECTION,
            self.ACCOUNTS_COLLECTION,
            self.INSIGHTS_COLLECTION,
            self.STOCKS_COLLECTION,
            # MERCHANTS si spécifique à l'utilisateur
            # self.MERCHANTS_COLLECTION,
        ]

        # Supprimer les points filtrés dans chaque collection
        for collection_name in collections_to_clean:
            try:
                 # Compter avant pour log
                 # count_before = self.client.count(collection_name=collection_name, count_filter=user_filter, exact=True).count
                 # logger.info(f"Suppression de {count_before} points de {collection_name} pour user {user_id}")

                 # Supprimer par filtre
                 self.client.delete(
                     collection_name=collection_name,
                     points_selector=qmodels.FilterSelector(filter=user_filter),
                     wait=True # Attendre la fin de la suppression
                 )
                 logger.info(f"Points supprimés de {collection_name} pour user {user_id}")
            except Exception as e:
                 # Ne pas bloquer si une collection n'existe pas ou autre erreur
                 logger.error(f"Erreur lors de la suppression des points de {collection_name} pour user {user_id}: {e}")
                 success = False # Marquer comme échoué partiellement

        # Supprimer les métadonnées utilisateur
        try:
            point_id = f"user_meta_{user_id}"
            self.client.delete(
                collection_name=self.USER_METADATA_COLLECTION,
                points_selector=qmodels.PointIdsList(points=[point_id]),
                wait=True
            )
            logger.info(f"Métadonnées supprimées pour user {user_id}")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des métadonnées pour user {user_id}: {e}")
            success = False

        logger.warning(f"Suppression des données Qdrant pour user_id={user_id} terminée avec succès: {success}")
        return success