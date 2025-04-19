"""
Utilitaires d'indexation pour le service de recherche.

Ce module fournit des fonctions pour indexer des données provenant
de différentes sources dans les moteurs de recherche.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from search_service.storage.unified_engine import get_unified_engine, SearchEngineType
from config_service.config import settings

logger = logging.getLogger(__name__)

async def index_transactions_from_sync_service(user_id: int, limit: int = 1000) -> Dict[str, Any]:
    """
    Indexe les transactions d'un utilisateur depuis le service de synchronisation.
    
    Args:
        user_id: ID de l'utilisateur
        limit: Nombre maximum de transactions à indexer
        
    Returns:
        Résultat de l'indexation
    """
    try:
        logger.info(f"Indexation des transactions depuis sync_service pour l'utilisateur {user_id}")
        
        # Import le service vectoriel depuis sync_service
        try:
            from sync_service.services.vector_storage import VectorStorageService
            vector_storage = VectorStorageService()
        except ImportError:
            logger.error("Impossible d'importer VectorStorageService depuis sync_service")
            return {
                "status": "error",
                "message": "sync_service not available",
                "documents_processed": 0
            }
        
        # Créer un filtre pour récupérer les transactions de l'utilisateur
        from qdrant_client import models as qmodels
        user_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))]
        )
        
        # Récupérer les transactions depuis Qdrant
        logger.debug(f"Récupération des transactions depuis Qdrant pour l'utilisateur {user_id}")
        
        # Utilisation de scroll pour pagination
        offset = None
        all_transactions = []
        
        while True:
            transactions_page, next_offset = vector_storage.client.scroll(
                collection_name=vector_storage.TRANSACTIONS_COLLECTION,
                scroll_filter=user_filter,
                limit=min(100, limit - len(all_transactions)),
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not transactions_page:
                break
                
            # Convertir en liste de dictionnaires
            all_transactions.extend([point.payload for point in transactions_page])
            
            if len(all_transactions) >= limit or next_offset is None:
                break
                
            offset = next_offset
        
        logger.info(f"Récupération de {len(all_transactions)} transactions pour l'utilisateur {user_id}")
        
        if not all_transactions:
            return {
                "status": "success",
                "message": "No transactions found to index",
                "documents_processed": 0
            }
        
        # Standardiser le format des transactions si nécessaire
        standardized_transactions = []
        for tx in all_transactions:
            # Convertir les dates si elles sont des chaînes
            transaction_date = tx.get("transaction_date")
            if isinstance(transaction_date, str):
                try:
                    tx["transaction_date"] = datetime.fromisoformat(transaction_date.replace('Z', '+00:00'))
                except ValueError:
                    pass
            
            # Ajouter un ID unique si manquant
            if "id" not in tx:
                tx["id"] = f"tx_{tx.get('bridge_transaction_id', '')}"
            
            standardized_transactions.append(tx)
        
        # Indexer les transactions dans les moteurs de recherche
        engine = get_unified_engine()
        result = await engine.index_documents(user_id, standardized_transactions)
        
        logger.info(f"Indexation terminée: {result['status']}, {len(standardized_transactions)} transactions traitées")
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation des transactions: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "documents_processed": 0
        }

async def index_batch_documents(user_id: int, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Indexe un lot de documents fourni directement.
    
    Args:
        user_id: ID de l'utilisateur
        documents: Liste de documents à indexer
        
    Returns:
        Résultat de l'indexation
    """
    try:
        logger.info(f"Indexation d'un lot de {len(documents)} documents pour l'utilisateur {user_id}")
        
        # Standardiser le format des documents si nécessaire
        standardized_documents = []
        for doc in documents:
            # Convertir les dates si elles sont des chaînes
            for date_field in ["transaction_date", "booking_date", "value_date"]:
                if date_field in doc and isinstance(doc[date_field], str):
                    try:
                        doc[date_field] = datetime.fromisoformat(doc[date_field].replace('Z', '+00:00'))
                    except ValueError:
                        pass
            
            # Ajouter un ID unique si manquant
            if "id" not in doc:
                doc["id"] = f"doc_{doc.get('bridge_transaction_id', '')}"
            
            standardized_documents.append(doc)
        
        # Indexer les documents dans les moteurs de recherche
        engine = get_unified_engine()
        result = await engine.index_documents(user_id, standardized_documents)
        
        logger.info(f"Indexation terminée: {result['status']}, {len(standardized_documents)} documents traités")
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation du lot de documents: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "documents_processed": 0
        }

async def reindex_user_data(user_id: int) -> Dict[str, Any]:
    """
    Réindexe toutes les données d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        
    Returns:
        Résultat de la réindexation
    """
    try:
        logger.info(f"Début de la réindexation pour l'utilisateur {user_id}")
        
        # 1. Supprimer les index existants
        engine = get_unified_engine()
        delete_result = engine.delete_user_index(user_id)
        
        if delete_result["status"] not in ["success", "partial"]:
            logger.error(f"Échec de la suppression des index existants: {delete_result}")
            return {
                "status": "error",
                "message": "Failed to delete existing indexes",
                "delete_result": delete_result
            }
        
        # 2. Réindexer les données depuis sync_service
        index_result = await index_transactions_from_sync_service(user_id, limit=10000)
        
        logger.info(f"Réindexation terminée pour l'utilisateur {user_id}: {index_result['status']}")
        
        return {
            "status": index_result["status"],
            "delete_result": delete_result,
            "index_result": index_result
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la réindexation pour l'utilisateur {user_id}: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

async def schedule_background_indexing(
    interval_seconds: int = 3600,
    user_callback: Optional[Callable[[int], bool]] = None
):
    """
    Planifie l'indexation périodique en arrière-plan.
    
    Args:
        interval_seconds: Intervalle entre chaque cycle d'indexation
        user_callback: Fonction appelée pour déterminer si un utilisateur doit être indexé
    """
    logger.info(f"Planification de l'indexation en arrière-plan toutes les {interval_seconds} secondes")
    
    while True:
        try:
            # Récupérer la liste des utilisateurs à indexer
            user_ids = await _get_users_to_index(user_callback)
            
            if not user_ids:
                logger.info("Aucun utilisateur à indexer dans ce cycle")
            else:
                logger.info(f"Indexation planifiée pour {len(user_ids)} utilisateurs")
                
                for user_id in user_ids:
                    try:
                        # Indexer les transactions de l'utilisateur
                        result = await index_transactions_from_sync_service(user_id)
                        logger.info(f"Indexation pour l'utilisateur {user_id}: {result['status']}")
                    except Exception as user_error:
                        logger.error(f"Erreur lors de l'indexation pour l'utilisateur {user_id}: {user_error}", exc_info=True)
                    
                    # Petite pause entre les utilisateurs pour éviter de surcharger le système
                    await asyncio.sleep(5)
        
        except Exception as e:
            logger.error(f"Erreur dans la boucle d'indexation en arrière-plan: {e}", exc_info=True)
        
        # Attendre jusqu'au prochain cycle
        logger.info(f"Prochain cycle d'indexation dans {interval_seconds} secondes")
        await asyncio.sleep(interval_seconds)

async def _get_users_to_index(user_callback: Optional[Callable[[int], bool]] = None) -> List[int]:
    """
    Récupère la liste des utilisateurs à indexer.
    
    Args:
        user_callback: Fonction appelée pour déterminer si un utilisateur doit être indexé
        
    Returns:
        Liste des IDs d'utilisateurs à indexer
    """
    try:
        # Si une fonction de callback est fournie, l'utiliser
        if user_callback:
            # Cette fonction devrait interroger la base de données et appliquer le callback
            # Pour cet exemple, on suppose que cette logique sera implémentée plus tard
            return []
        
        # Sinon, utiliser une approche par défaut
        # Pour cet exemple, on peut utiliser les statistiques du moteur pour trouver les utilisateurs
        engine = get_unified_engine()
        stats = engine.get_stats()
        
        # Extraire les IDs utilisateurs des statistiques
        user_ids = []
        for engine_name, engine_stats in stats["engines"].items():
            if engine_name == "qdrant":
                continue  # Pas de statistiques par utilisateur dans Qdrant via cette interface
                
            if "users" in engine_stats:
                for user_id in engine_stats["users"]:
                    user_ids.append(int(user_id))
        
        # Éliminer les doublons
        return list(set(user_ids))
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des utilisateurs à indexer: {e}", exc_info=True)
        return []